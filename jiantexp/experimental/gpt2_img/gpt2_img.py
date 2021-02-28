import glob
import jiant.utils.path_parse as path_parse
import jiant.utils.python.io as io
import sys
import os
import torch
import numpy as np
import transformers
import datasets
import torch.nn.functional as F
import torch.nn as nn
import jiant.utils.zconf as zconf
import PIL

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import jiant.utils.display as display


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # Management
    data_fol = zconf.attr(type=str, required=True)
    output_fol = zconf.attr(type=str, required=True)

    # Training
    gpt2_name = zconf.attr(type=str, default="gpt2")
    learning_rate = zconf.attr(type=float, default=1e-5)
    num_steps = zconf.attr(type=int, default=10000)
    log_interval = zconf.attr(type=int, default=10)
    eval_interval = zconf.attr(type=int, default=100)
    save_interval = zconf.attr(type=int, default=0)
    batch_size = zconf.attr(type=int, default=8)
    gradient_accumulation_steps = zconf.attr(type=int, default=8)
    num_workers = zconf.attr(type=int, default=16)


def setup_paths():
    sys.path += ["/home/zp489/code/othergits/DALL-E/"]


def get_gpt2(gpt2_name: str):
    # noinspection PyUnresolvedReferences
    from dall_e import map_pixels, unmap_pixels, load_model
    # vocab_size = 50257 + 8192
    gpt2 = transformers.GPT2LMHeadModel.from_pretrained(gpt2_name)
    gpt2_embeddings = gpt2.state_dict()['lm_head.weight']

    # Create vqtok_embeddings
    dec = torch.load("/home/zp489/scratch/working/2102/24_pixgen/weights/decoder.state_dict")
    raw_vqtok_embeddings = dec['blocks.input.w'].squeeze().t()
    unscaled_vqtok_embeddings = raw_vqtok_embeddings @ torch.randn(128, gpt2.config.hidden_size)
    norm_vqtok_embeddings = \
        (unscaled_vqtok_embeddings - unscaled_vqtok_embeddings.mean()) / unscaled_vqtok_embeddings.std()
    vqtok_embeddings = \
        norm_vqtok_embeddings * gpt2_embeddings.std() + gpt2_embeddings.mean()
    padding_embeddings = torch.randn([1, gpt2.config.hidden_size]) * gpt2_embeddings.std() + gpt2_embeddings.mean()
    full_embeddings = torch.cat([gpt2_embeddings, padding_embeddings, vqtok_embeddings], dim=0)

    # Replace embeddings
    gpt2.transformer.wte = nn.Embedding(full_embeddings.shape[0], gpt2.config.hidden_size)
    gpt2.lm_head = nn.Linear(gpt2.config.hidden_size, full_embeddings.shape[0])
    gpt2_state_dict = gpt2.state_dict()
    gpt2_state_dict['transformer.wte.weight'] = full_embeddings
    gpt2_state_dict['lm_head.weight'] = full_embeddings
    gpt2.load_state_dict(gpt2_state_dict)
    return gpt2


def collate_fn(examples):
    # noinspection PyArgumentList
    return {
        "input_ids": torch.LongTensor([
            example["input_ids"]
            for example in examples
        ]),
    }


def show_entry(entry, dec, dev, upscale=None, resize_to=None):
    # noinspection PyUnresolvedReferences
    from dall_e import map_pixels, unmap_pixels, load_model
    if upscale is not None:
        entry = np.kron(entry, np.ones([upscale, upscale])).astype(int)
    z_idx = torch.from_numpy(entry).unsqueeze(0).to(dev)
    z = F.one_hot(z_idx, num_classes=dec.vocab_size).permute(0, 3, 1, 2).float()
    x_stats = dec(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
    if resize_to is not None:
        x_rec = TF.resize(x_rec, (256, 256), interpolation=PIL.Image.LANCZOS)
    return x_rec


def cycle_dataloader(dataloader, num_steps):
    steps = 0
    while True:
        for batch in dataloader:
            yield batch
            steps += 1
            if steps == num_steps:
                return


def generate_preds(gpt2, input_ids):
    preds = []
    for idx in range(71, 512):
        gpt2.eval()
        with torch.no_grad():
            transformer_outputs = gpt2.transformer(input_ids[:, :idx])
            hidden_states = transformer_outputs[0]
            lm_logits = gpt2.lm_head(hidden_states[:, -1:])
            preds.append(lm_logits[:, :, 50258:].max(-1)[1].cpu().numpy())
    batch_preds = np.concatenate(preds, axis=1)
    return batch_preds


def main(args: RunConfiguration):
    setup_paths()
    input_ids_dataset = datasets.load_from_disk(args.data_fol)
    train_dataloader = DataLoader(
        input_ids_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0 if args.num_workers == 1 else args.num_workers,
    )
    os.makedirs(args.output_fol, exist_ok=True)
    dev = torch.device("cuda:0")
    gpt2 = get_gpt2(args.gpt2_name).to(dev)

    gpt2.train()
    optimizer = optim.Adam(gpt2.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    loss_fct = nn.CrossEntropyLoss()

    for i, batch in enumerate(cycle_dataloader(train_dataloader, num_steps=args.num_steps)):
        input_ids = batch["input_ids"].to(dev)
        transformer_outputs = gpt2.transformer(input_ids[:, :-1])
        hidden_states = transformer_outputs[0]
        lm_logits = gpt2.lm_head(hidden_states[:, 70:])
        labels = input_ids[:, 71:]
        loss = loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.flatten())
        loss.backward()
        if i % args.gradient_accumulation_steps == 0 and i != 0:
            optimizer.step()
            optimizer.zero_grad()
        if i % args.log_interval == 0 and i != 0:
            print(f"Step {i}")
            sys.stdout.flush()
        if i % args.save_interval == 0 and i != 0:
            torch.save(gpt2.state_dict(), os.path.join(args.output_fol, f"model___{i:09d}.p"))
        if i % args.eval_interval == 0 and i != 0:
            gpt2.eval()
            preds = generate_preds(gpt2=gpt2, input_ids=input_ids)
            gpt2.train()
            torch.save(
                {"inputs": input_ids.cpu(), "preds": preds},
                os.path.join(args.output_fol, f"preds___{i:09d}.p"),
            )


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
