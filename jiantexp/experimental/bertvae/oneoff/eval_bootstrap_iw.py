import transformers
import datasets
import torch
import numpy as np
import os
import sys

import jiant.utils.zconf as zconf
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiantexp.experimental.bertvae.models as models
import jiant.utils.python.io as io
import pandas as pd
from jiantexp.experimental.bertvae.utils import move_to_device
import torch.nn.functional as F


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_name = zconf.attr(type=str, required=True)
    data_fol = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, required=True)

    # Evaluation
    batch_size = zconf.attr(type=int, default=16)
    num_workers = zconf.attr(type=int, default=16)
    iw_sampling_k = zconf.attr(type=int, default=None)
    iw_sampling_debug_mode = zconf.attr(type=str, default="ratio_pq_sample_q")
    multi_sampling_k = zconf.attr(type=int, default=None)

    num_samples = zconf.attr(type=int, default=128)
    num_bootstrap = zconf.attr(type=int, default=64)

    num_examples = zconf.attr(type=int, default=8192)

    #
    sample_epsilon = zconf.attr(type=float, default=1e-5)


def main(args: RunConfiguration):
    # === Setup === #
    torch.set_grad_enabled(False)
    model_configs = io.read_json(args.model_config_path)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    mlm_model = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    device = torch.device("cuda:0")
    bert_data_wrapper = data_wrappers.get_data_wrapper(
        data_name=args.data_name,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )
    val_vae_dataset = datasets.load_from_disk(os.path.join(args.data_fol, "val"))
    bert_vae_model = models.BertVaeModel(
        mlm_model=mlm_model,
        latent_token_mode=model_configs["latent_token_mode"],
        add_latent_linear=model_configs["add_latent_linear"],
        do_lagrangian=model_configs["do_lagrangian"],
        sample_epsilon=args.sample_epsilon,
    )
    bert_vae_model.load_state_dict(torch.load(args.model_path))
    bert_vae_model = bert_vae_model.to(device)
    bert_vae_model.eval()
    val_dataloader = bert_data_wrapper.create_val_dataloader(
        val_vae_dataset=val_vae_dataset,
        batch_size=args.batch_size,
        eval_multiplier=1,
    )
    loss_fct = bert_vae_model.get_loss_fct()
    _, ratio_mode, _, sample_from = args.iw_sampling_debug_mode.split("_")

    all_bootstraps = []
    all_iw_losses = []
    num_batches = args.num_examples // args.batch_size
    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= num_batches:
            break

        print(f"Working on {batch_idx}/{num_batches}")
        sys.stdout.flush()
        # Simple forward
        batch = move_to_device(batch, device=device)
        posterior_output = bert_vae_model.posterior_forward(batch)
        prior_output = bert_vae_model.prior_forward(batch)
        batch_size, hidden_dim = posterior_output["z_loc"].shape
        prior_dist = torch.distributions.normal.Normal(
            loc=prior_output["z_loc"],
            scale=torch.exp(prior_output["z_logvar"] / 2),
        )
        posterior_dist = torch.distributions.normal.Normal(
            loc=posterior_output["z_loc"],
            scale=torch.exp(posterior_output["z_logvar"] / 2),
        )
        print(f"...shared forward done")
        sys.stdout.flush()

        # Prep for Bootstrap
        weighted_log_token_probs = torch.zeros([
            args.num_samples, args.batch_size,
            batch["decoder_label"].shape[1],
            bert_vae_model.mlm_model.config.vocab_size,
        ]).to(device)
        for i in range(args.num_samples):
            if sample_from == "q":
                z_sample = bert_vae_model.sample_z(z_loc=posterior_output["z_loc"],
                                                   z_logvar=posterior_output["z_logvar"])
            elif sample_from == "p":
                z_sample = bert_vae_model.sample_z(z_loc=prior_output["z_loc"], z_logvar=prior_output["z_logvar"])
            else:
                raise RuntimeError()
            mlm_output = bert_vae_model.decoder_forward(batch=batch, z_sample=z_sample)
            if ratio_mode == "1":
                log_weight = prior_dist.log_prob(z_sample).sum(-1) - prior_dist.log_prob(z_sample).sum(-1)
            elif ratio_mode == "pq":
                log_weight = prior_dist.log_prob(z_sample).sum(-1) - posterior_dist.log_prob(z_sample).sum(-1)
            elif ratio_mode == "qp":
                log_weight = posterior_dist.log_prob(z_sample).sum(-1) - prior_dist.log_prob(z_sample).sum(-1)
            else:
                raise KeyError()
            weighted_log_token_probs[i] = log_weight.view(batch_size, 1, 1) + F.log_softmax(mlm_output.logits, dim=-1)
            del z_sample
            del mlm_output
            del log_weight

        print(f"...bootstrap forward done")
        sys.stdout.flush()

        reweighted_logits = torch.logsumexp(weighted_log_token_probs, dim=0) - np.log(args.num_samples)
        iw_masked_lm_loss = loss_fct(
            reweighted_logits.view(-1, bert_vae_model.mlm_model.config.vocab_size),
            target=batch["decoder_label"].view(-1),
        ) / batch_size

        results_dict = {}
        for exponent in range(int(np.log2(args.num_samples))):
            sub_num_samples = 2 ** exponent
            results_dict[sub_num_samples] = []
            for j in range(args.num_bootstrap):
                indices = torch.randperm(args.num_samples)[:2 ** exponent]
                reweighted_logits = torch.logsumexp(weighted_log_token_probs[indices], dim=0) - np.log(sub_num_samples)
                sub_iw_masked_lm_loss = loss_fct(
                    reweighted_logits.view(-1, bert_vae_model.mlm_model.config.vocab_size),
                    target=batch["decoder_label"].view(-1),
                ) / batch_size
                results_dict[sub_num_samples].append(sub_iw_masked_lm_loss.item())
        del weighted_log_token_probs

        all_bootstraps.append(pd.DataFrame(results_dict))
        all_iw_losses.append(iw_masked_lm_loss.item())

        print(f"...bootstrap done")
        sys.stdout.flush()

    io.create_containing_folder(args.output_path)
    torch.save({
        "all_bootstraps": all_bootstraps,
        "all_iw_losses": all_iw_losses,
    }, args.output_path)


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
