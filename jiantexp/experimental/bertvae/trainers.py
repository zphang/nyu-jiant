import os
import torch
from typing import Any
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
import jiant.utils.display as display
import jiantexp.experimental.bertvae.kl_weight_schedulers as kl_weight_schedulers
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers
import jiantexp.experimental.bertvae.models as models
from jiantexp.experimental.bertvae.utils import move_to_device, cycle_dataloader


class BertVaeTrainer:
    def __init__(self,
                 bert_data_wrapper: data_wrappers.BertDataWrapper,
                 bert_vae_model: models.BertVaeModel,
                 train_dataloader,
                 val_dataloader,
                 log_writer,
                 args):
        self.bert_data_wrapper = bert_data_wrapper
        self.bert_vae_model = bert_vae_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.log_writer = log_writer
        self.args = args

        self.device = torch.device("cuda:0")
        self.tokenizer: Any = self.bert_data_wrapper.tokenizer

        self.optimizer = None
        self.scheduler = None
        self.kl_weight_scheduler = None
        self.dummy_batch = None

    def setup(self):
        if self.args.optimizer_type == "adamw":
            self.optimizer = AdamW(self.bert_vae_model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer_type == "sgd":
            self.optimizer = SGD(self.bert_vae_model.parameters(), lr=self.args.learning_rate)
        else:
            raise KeyError(self.args.optimizer_type)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.num_steps / 10,
            num_training_steps=self.args.num_steps,
        )
        self.dummy_batch = move_to_device(self.bert_data_wrapper.get_dummy_batch(), device=self.device)
        self.kl_weight_scheduler = kl_weight_schedulers.create_kl_weight_scheduler(args=self.args)

    def do_train_val(self):
        self.bert_vae_model.train()
        train_loss = 0
        for step, batch in enumerate(cycle_dataloader(self.train_dataloader, num_steps=self.args.num_steps)):
            batch = move_to_device(batch, device=self.device)
            self.optimizer.zero_grad()
            vae_output = self.bert_vae_model(batch=batch)
            train_kl_loss = self.kl_weight_scheduler.get_loss(
                step=step,
                kl_loss_tensor=vae_output["kl_loss_tensor"],
            )
            loss = vae_output["recon_loss"] + train_kl_loss
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            self.scheduler.step()
            if (step + 1) % self.args.log_interval == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon-L: {:.6f}\tKL-L: {:.6f}'.format(
                    step, self.args.num_steps,
                    100. * step / self.args.num_steps,
                    vae_output["total_loss"].item(),
                    vae_output["recon_loss"].item(),
                    train_kl_loss.item(),
                ))
                self.log_writer.write_entry(
                    "loss_train",
                    {
                        "step": step,
                        "total_loss": vae_output["total_loss"].item(),
                        "recon_loss": vae_output["recon_loss"].item(),
                        "kl_loss": vae_output["kl_loss"].item(),
                        "train_kl_loss": train_kl_loss.item(),
                    },
                )
            if self.args.save_interval != 0 and (step + 1) % self.args.save_interval == 0:
                torch.save(
                    self.bert_vae_model.state_dict(),
                    os.path.join(self.args.output_fol, f"model___{step:09d}.p")
                )
            if (step + 1) % self.args.eval_interval == 0:
                self.do_full_val(step)

    def do_val(self):
        self.bert_vae_model.eval()
        with torch.no_grad():
            agg_total_loss = 0
            agg_recon_loss = 0
            agg_kl_loss = 0
            total_size = 0
            for batch in display.tqdm(self.val_dataloader):
                batch = move_to_device(batch, device=self.device)
                batch_size = len(batch["decoder_label"])
                vae_output = self.bert_vae_model(batch=batch)
                agg_total_loss += vae_output["total_loss"].item() * batch_size
                agg_recon_loss += vae_output["recon_loss"].item() * batch_size
                agg_kl_loss += vae_output["kl_loss"].item() * batch_size
                total_size += batch_size
        self.bert_vae_model.train()
        results = {
            "agg_total_loss": agg_total_loss,
            "agg_recon_loss": agg_recon_loss,
            "agg_kl_loss": agg_kl_loss,
            "total_size": total_size,
        }
        return results


    def do_full_val(self, step):
        val_results = self.do_val()
        print('[{}/{} ({:.0f}%)]\t V-Loss: {:.6f}\tV-Recon-L: {:.6f}\tV-KL-L: {:.6f}'.format(
            step, self.args.num_steps,
            100. * step / self.args.num_steps,
            val_results["agg_total_loss"] / val_results["total_size"],
            val_results["agg_recon_loss"] / val_results["total_size"],
            val_results["agg_kl_loss"] / val_results["total_size"],
        ))
        self.log_writer.write_entry(
            "loss_val",
            {
                "step": step,
                "total_loss": val_results["agg_total_loss"] / val_results["total_size"],
                "recon_loss": val_results["agg_recon_loss"] / val_results["total_size"],
                "kl_loss": val_results["agg_kl_loss"] / val_results["total_size"],
            },
        )
        self.bert_vae_model.eval()
        batch = next(iter(self.val_dataloader))
        batch = move_to_device(batch, device=self.device)
        with torch.no_grad():
            vae_output = self.bert_vae_model(batch=batch)
        idx = 0
        display_dict = data_wrappers.format_labeled_example(
            masked_token_ids=batch["decoder_input"][idx],
            gold_labels=batch["decoder_label"][idx],
            predictions=vae_output["logits"][idx].max(1).indices,
            tokenizer=self.tokenizer,
        )
        print(display_dict["gold_str"])
        print(display_dict["pred_str"])
        self.log_writer.write_entry(
            "val_example",
            {
                "step": step,
                "gold_str": display_dict["gold_str_html"],
                "pred_str": display_dict["pred_str_html"],
            },
        )
        with torch.no_grad():
            vae_output = self.bert_vae_model(batch=self.dummy_batch)
        for idx in range(6):
            display_dict = data_wrappers.format_labeled_example(
                masked_token_ids=self.dummy_batch["decoder_input"][idx],
                gold_labels=self.dummy_batch["decoder_label"][idx],
                predictions=vae_output["logits"][idx].max(1).indices,
                tokenizer=self.tokenizer,
            )
            print("GOLD:", display_dict["gold_str"])
            print("PRED:", display_dict["pred_str"])
            print()
            self.log_writer.write_entry(
                "val_example_dummy",
                {
                    "step": step,
                    "idx": idx,
                    "gold_str": display_dict["gold_str_html"],
                    "pred_str": display_dict["pred_str_html"],
                },
            )
        print("=====\n")
        self.bert_vae_model.train()
