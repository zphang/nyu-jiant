import os
import datasets
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

    # Data
    train_from = zconf.attr(type=int, default=0)
    train_to = zconf.attr(type=int, default=30000)
    val_from = zconf.attr(type=int, default=100000)
    val_to = zconf.attr(type=int, default=100030)


def main(args: RunConfiguration):
    os.makedirs(args.data_fol, exist_ok=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    bert_data_wrapper = data_wrappers.BertDataWrapper(
        tokenizer=tokenizer,
        num_workers=args.num_workers,
        mlm_probability=args.mlm_probability,
    )
    wiki_train_data = datasets.load_dataset(
        "wikipedia",
        name="20200501.en",
        split=datasets.ReadInstruction('train', from_=args.train_from, to=args.train_to, unit="abs"),
    )
    wiki_val_data = datasets.load_dataset(
        "wikipedia",
        name="20200501.en",
        split=datasets.ReadInstruction('train', from_=args.val_from, to=args.val_to, unit="abs"),
    )
    train_vae_dataset = bert_data_wrapper.prepare_vae_dataset(
        text_dataset=wiki_train_data,
    )
    val_vae_dataset = bert_data_wrapper.prepare_vae_dataset(
        text_dataset=wiki_val_data,
    )
    train_vae_dataset.save_to_disk(os.path.join(args.data_fol, "train"))
    val_vae_dataset.save_to_disk(os.path.join(args.data_fol, "val"))


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
