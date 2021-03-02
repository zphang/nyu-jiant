import transformers
import datasets
import torch
import torch.nn as nn
import os

import jiant.utils.zconf as zconf
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiant.utils.display as display
from jiantexp.experimental.bertvae.utils import move_to_device
import pyutils.io as io


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_fol = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, default=None)

    # data
    data_name = zconf.attr(type=str, required=True)

    # Running
    batch_size = zconf.attr(type=int, default=16)
    num_workers = zconf.attr(type=int, default=16)

    # Configuration
    # Don't set any for closest BERT-MLM replication
    no_restore_cls = zconf.attr(action="store_true")
    no_fix_suffix = zconf.attr(action="store_true")
    multiply_by_num_masked = zconf.attr(action="store_true")


def main(args: RunConfiguration):
    device = torch.device("cuda:0")
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")
    mlm_model = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    mlm_model = mlm_model.eval().to(device)
    bert_data_wrapper = data_wrappers.get_data_wrapper(
        data_name=args.data_name,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )
    val_vae_dataset = datasets.load_from_disk(os.path.join(args.data_fol, "val"))
    val_dataloader = bert_data_wrapper.create_val_dataloader(
        val_vae_dataset=val_vae_dataset,
        batch_size=args.batch_size,
    )
    with torch.no_grad():
        agg_recon_loss = 0
        total_size = 0
        for batch in display.tqdm(val_dataloader):
            batch = move_to_device(batch, device=device)

            decoder_input = batch["decoder_input"].clone()
            decoder_label = batch["decoder_label"].clone()
            decoder_mask = batch["decoder_mask"].clone()
            if not args.no_restore_cls:
                decoder_input[:, 0] = tokenizer.cls_token_id

            if not args.no_fix_suffix:
                range_idx = torch.arange(decoder_input.shape[0]).to(decoder_input.device)
                decoder_input[range_idx, batch["decoder_z_index"]] = 0
                decoder_input[range_idx, batch["decoder_z_index"] + 1] = 0
                decoder_label[range_idx, batch["decoder_z_index"]] = 0
                decoder_label[range_idx, batch["decoder_z_index"] + 1] = 0
                decoder_mask[range_idx, batch["decoder_z_index"]] = 0
                decoder_mask[range_idx, batch["decoder_z_index"] + 1] = 0

            batch_size = len(batch["decoder_label"])
            mlm_out = mlm_model(
                input_ids=decoder_input,
                attention_mask=decoder_mask,
                labels=decoder_label,
            )
            if args.multiply_by_num_masked:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=data_wrappers.BertDataWrapper.NON_MASKED_TARGET,  # Clean up?
                    reduction="sum",
                )
                recon_loss = loss_fct(
                    mlm_out.logits.view(-1, mlm_model.config.vocab_size),
                    target=batch["decoder_label"].view(-1)
                )
            else:
                recon_loss = mlm_out.loss * batch_size
            agg_recon_loss += recon_loss
            total_size += batch_size
    results = {
        "agg_recon_loss": float(agg_recon_loss),
        "total_size": int(total_size),
        "recon_loss": float(agg_recon_loss / total_size),
    }
    display.show_json(results)
    if args.output_path:
        io.write_json(results, io.make_fol_for_path(args.output_path))


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
