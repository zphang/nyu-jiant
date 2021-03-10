import transformers
import datasets
import torch
import os

import jiant.utils.zconf as zconf
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiantexp.experimental.bertvae.trainers as trainers
import jiantexp.experimental.bertvae.models as models
import zproto.zlogv1 as zlog


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_fol = zconf.attr(type=str, required=True)
    output_fol = zconf.attr(type=str, required=True)

    # data
    data_name = zconf.attr(type=str, required=True)

    # Training
    optimizer_type = zconf.attr(type=str, default="adamw")
    scheduler_type = zconf.attr(type=str, default="linear")
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
    iw_sampling_k = zconf.attr(type=int, default=1)

    do_lagrangian = zconf.attr(action="store_true")


def main(args: RunConfiguration):
    os.makedirs(args.output_fol, exist_ok=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    mlm_model = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    device = torch.device("cuda:0")
    bert_data_wrapper = data_wrappers.get_data_wrapper(
        data_name=args.data_name,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )
    train_vae_dataset = datasets.load_from_disk(os.path.join(args.data_fol, "train"))
    val_vae_dataset = datasets.load_from_disk(os.path.join(args.data_fol, "val"))
    bert_vae_model = models.BertVaeModel(
        mlm_model=mlm_model,
        latent_token_mode=args.latent_token_mode,
        add_latent_linear=args.add_latent_linear,
        do_lagrangian=args.do_lagrangian,
    ).to(device)
    bert_vae_trainer = trainers.BertVaeTrainer(
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
