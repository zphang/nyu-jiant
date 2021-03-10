import transformers
import datasets
import torch
import os

import jiant.utils.zconf as zconf
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiantexp.experimental.bertvae.models as models
import jiant.utils.display as display
import jiant.utils.python.io as io
from jiantexp.experimental.bertvae.utils import move_to_device


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_fol = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, required=True)
    forward_mode = zconf.attr(type=str, default="single")

    # data
    data_name = zconf.attr(type=str, required=True)

    # Evaluation
    batch_size = zconf.attr(type=int, default=16)
    num_workers = zconf.attr(type=int, default=16)
    latent_token_mode = zconf.attr(type=str, default="zindex")
    add_latent_linear = zconf.attr(action="store_true")
    do_lagrangian = zconf.attr(action="store_true")
    iw_sampling_k = zconf.attr(type=int, default=None)
    multi_sampling_k = zconf.attr(type=int, default=None)


def main(args: RunConfiguration):
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
        latent_token_mode=args.latent_token_mode,
        add_latent_linear=args.add_latent_linear,
        do_lagrangian=args.do_lagrangian,
    )
    bert_vae_model.load_state_dict(torch.load(args.model_path))
    bert_vae_model = bert_vae_model.to(device)
    val_dataloader = bert_data_wrapper.create_val_dataloader(
        val_vae_dataset=val_vae_dataset,
        batch_size=args.batch_size,
    )
    bert_vae_model.eval()
    with torch.no_grad():
        agg_total_loss = 0
        agg_recon_loss = 0
        agg_kl_loss = 0
        total_size = 0
        for batch in display.tqdm(val_dataloader):
            batch = move_to_device(batch, device=device)
            batch_size = len(batch["decoder_label"])
            vae_output = bert_vae_model(
                batch=batch,
                forward_mode=args.forward_mode,
                iw_sampling_k=args.iw_sampling_k,
                multi_sampling_k=args.multi_sampling_k,
            )
            agg_total_loss += vae_output["total_loss"].item() * batch_size
            agg_recon_loss += vae_output["recon_loss"].item() * batch_size
            agg_kl_loss += vae_output["kl_loss"].item() * batch_size
            total_size += batch_size
    results = {
        "agg_total_loss": float(agg_total_loss),
        "agg_recon_loss": float(agg_recon_loss),
        "agg_kl_loss": float(agg_kl_loss),
        "total_size": int(total_size),
        "avg_total_loss": float(agg_total_loss / total_size),
        "avg_recon_loss": float(agg_recon_loss / total_size),
        "avg_kl_loss": float(agg_kl_loss / total_size),
    }
    display.show_json(results)
    io.write_json(results, args.output_path)


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
