import transformers
import datasets
import torch
import os

import jiant.utils.zconf as zconf
import jiantexp.experimental.bertvae.bert_funcs as bert_funcs
import zproto.zlogv1 as zlog


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_fol = zconf.attr(type=str, required=True)
    output_fol = zconf.attr(type=str, required=True)

    # Training
    learning_rate = zconf.attr(type=float, default=1e-5)
    num_steps = zconf.attr(type=int, default=10000)
    log_interval = zconf.attr(type=int, default=10)
    eval_interval = zconf.attr(type=int, default=100)
    save_interval = zconf.attr(type=int, default=0)
    batch_size = zconf.attr(type=int, default=16)
    num_workers = zconf.attr(type=int, default=16)
    kl_weight_scheduler_name = zconf.attr(type=str, default="ConstantScheduler")
    kl_weight_scheduler_config = zconf.attr(type=str, default="1")
    latent_token_mode = zconf.attr(type=str, default="zindex")
    add_latent_linear = zconf.attr(action="store_true")


def main(args: RunConfiguration):
    os.makedirs(args.output_fol, exist_ok=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    mlm_model = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    device = torch.device("cuda:0")
    bert_data_wrapper = bert_funcs.BertDataWrapper(
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )
    train_vae_dataset = datasets.load_from_disk(os.path.join(args.data_fol, "train"))
    val_vae_dataset = datasets.load_from_disk(os.path.join(args.data_fol, "val"))
    bert_vae_model = bert_funcs.BertVaeModel(
        mlm_model=mlm_model,
        latent_token_mode=args.latent_token_mode,
        add_latent_linear=args.add_latent_linear,
    ).to(device)
    bert_vae_trainer = bert_funcs.BertVaeTrainer(
        bert_data_wrapper=bert_data_wrapper,
        bert_vae_model=bert_vae_model,
        train_dataloader=bert_data_wrapper.create_train_dataloader(
            train_vae_dataset=train_vae_dataset,
            batch_size=args.batch_size,
        ),
        val_dataloader=bert_data_wrapper.create_val_dataloader(
            val_vae_dataset=val_vae_dataset,
            batch_size=args.batch_size,
        ),
        log_writer=zlog.ZLogger(os.path.join(args.output_fol, "logging"), overwrite=True),
        args=args,
    )
    bert_vae_trainer.setup()
    bert_vae_trainer.do_train_val()
    torch.save(
        bert_vae_model.cpu().state_dict(),
        os.path.join(args.output_fol, "model.p")
    )


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
