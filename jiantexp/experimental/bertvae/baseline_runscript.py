import sys
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import os

import jiant.utils.zconf as zconf
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiant.utils.display as display
from jiantexp.experimental.bertvae.utils import move_to_device, cycle_dataloader
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
    do_train = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")

    # Configuration
    # Don't set any for closest BERT-MLM replication
    no_restore_cls = zconf.attr(action="store_true")
    no_fix_suffix = zconf.attr(action="store_true")
    multiply_by_num_masked = zconf.attr(action="store_true")

    # Training only
    learning_rate = zconf.attr(type=float, default=1e-5)
    num_steps = zconf.attr(type=int, default=20000)
    save_path = zconf.attr(type=str, default=None)
    log_interval = zconf.attr(type=int, default=500)



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
    train_dataloader = bert_data_wrapper.create_val_dataloader(
        val_vae_dataset=datasets.load_from_disk(os.path.join(args.data_fol, "train")),
        batch_size=args.batch_size,
    )
    val_dataloader = bert_data_wrapper.create_val_dataloader(
        val_vae_dataset=datasets.load_from_disk(os.path.join(args.data_fol, "val")),
        batch_size=args.batch_size,
    )

    if args.do_train:
        train(
            mlm_model=mlm_model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            device=device,
            args=args,
        )
        if args.save_path is not None:
            torch.save(mlm_model.state_dict(), io.make_fol_for_path(args.save_path))

    if args.do_val:
        results = evaluate(
            mlm_model=mlm_model,
            tokenizer=tokenizer,
            val_dataloader=val_dataloader,
            device=device,
            args=args,
        )
        display.show_json(results)
        if args.output_path:
            io.write_json(results, io.make_fol_for_path(args.output_path))


def train(mlm_model, tokenizer, train_dataloader, device, args: RunConfiguration):
    optimizer = optim.AdamW(mlm_model.parameters(), lr=args.learning_rate)
    mlm_model.train()
    for step, batch in enumerate(cycle_dataloader(train_dataloader, num_steps=args.num_steps)):
        optimizer.zero_grad()
        batch = move_to_device(batch, device=device)
        new_batch = get_modified_batch(
            batch=batch,
            tokenizer=tokenizer,
            args=args,
        )
        recon_loss = compute_recon_loss(
            new_batch=new_batch,
            mlm_model=mlm_model,
            args=args,
        )
        recon_loss.backward()
        optimizer.step()
        if (step + 1) % args.log_interval == 0:
            print(f"[{step}/{args.num_steps}] Train loss: {recon_loss.item():.3f}")
            sys.stdout.flush()


def evaluate(mlm_model, tokenizer, val_dataloader, device, args: RunConfiguration):
    mlm_model.eval()
    with torch.no_grad():
        agg_recon_loss = 0
        total_size = 0
        for batch in display.tqdm(val_dataloader):
            batch = move_to_device(batch, device=device)
            batch_size = len(batch["decoder_label"])
            new_batch = get_modified_batch(
                batch=batch,
                tokenizer=tokenizer,
                args=args,
            )
            recon_loss = compute_recon_loss(
                new_batch=new_batch,
                mlm_model=mlm_model,
                args=args,
            )
            agg_recon_loss += recon_loss.item()
            total_size += batch_size
    results = {
        "agg_recon_loss": float(agg_recon_loss),
        "total_size": int(total_size),
        "recon_loss": float(agg_recon_loss / total_size),
    }
    return results


def get_modified_batch(batch, tokenizer, args):
    decoder_input = batch["decoder_input"].clone()
    decoder_label = batch["decoder_label"].clone()
    decoder_mask = batch["decoder_mask"].clone()
    if not args.no_restore_cls:
        decoder_input[:, 0] = tokenizer.cls_token_id

    if not args.no_fix_suffix:
        range_idx = torch.arange(decoder_input.shape[0]).to(decoder_input.device)
        decoder_input[range_idx, batch["decoder_z_index"]] = 0
        decoder_input[range_idx, batch["decoder_z_index"] + 1] = 0
        decoder_label[range_idx, batch["decoder_z_index"]] = -100
        decoder_label[range_idx, batch["decoder_z_index"] + 1] = -100
        decoder_mask[range_idx, batch["decoder_z_index"]] = 0
        decoder_mask[range_idx, batch["decoder_z_index"] + 1] = 0
    return {
        "decoder_input": decoder_input,
        "decoder_label": decoder_label,
        "decoder_mask": decoder_mask,
    }


def compute_recon_loss(new_batch, mlm_model, args: RunConfiguration):
    mlm_out = mlm_model(
        input_ids=new_batch["decoder_input"],
        attention_mask=new_batch["decoder_mask"],
        labels=new_batch["decoder_label"],
    )
    batch_size = len(new_batch["decoder_label"])
    if args.multiply_by_num_masked:
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=data_wrappers.BertDataWrapper.NON_MASKED_TARGET,  # Clean up?
            reduction="sum",
        )
        recon_loss = loss_fct(
            mlm_out.logits.view(-1, mlm_model.config.vocab_size),
            target=new_batch["decoder_label"].view(-1)
        )
    else:
        recon_loss = mlm_out.loss * batch_size
    return recon_loss


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
