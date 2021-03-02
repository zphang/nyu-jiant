import os
import transformers

import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_fol = zconf.attr(type=str, required=True)

    # Config
    num_workers = zconf.attr(type=int, default=16)
    mlm_probability = zconf.attr(type=float, default=0.15)
    max_seq_length = zconf.attr(type=int, default=256)

    # Data
    data_name = zconf.attr(type=str, required=True)
    train_from = zconf.attr(type=int, default=0)
    train_to = zconf.attr(type=int, default=30000)
    val_from = zconf.attr(type=int, default=100000)
    val_to = zconf.attr(type=int, default=100030)


def main(args: RunConfiguration):
    os.makedirs(args.data_fol, exist_ok=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    bert_data_wrapper = data_wrappers.get_data_wrapper(
        data_name=args.data_name,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
    )
    datasets_dict = bert_data_wrapper.get_datasets(args=args)
    train_vae_dataset = bert_data_wrapper.prepare_vae_dataset(
        text_dataset=datasets_dict["train"],
    )
    val_vae_dataset = bert_data_wrapper.prepare_vae_dataset(
        text_dataset=datasets_dict["val"],
    )
    train_vae_dataset.save_to_disk(os.path.join(args.data_fol, "train"))
    val_vae_dataset.save_to_disk(os.path.join(args.data_fol, "val"))


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
