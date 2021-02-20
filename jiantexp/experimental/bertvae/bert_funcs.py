import torch
from typing import List, Any
from dataclasses import dataclass
import transformers
from transformers.models.bert.modeling_bert import MaskedLMOutput
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import contextlib
import itertools
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import jiant.utils.display as display
import jiantexp.experimental.bertvae.kl_weight_schedulers as kl_weight_schedulers


@dataclass
class BertDataWrapper:
    # Constants
    DECODER_TOKEN_ID = 1  # [unused1]
    PRIOR_TOKEN_ID = 2  # [unused2]
    POSTERIOR_TOKEN_ID = 3  # [unused3]
    RESERVED_FOR_Z_TOKEN_ID = 4  # [unused4]
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    MASK_TOKEN_ID = 103
    NON_MASKED_TARGET = -100

    # Arguments
    tokenizer: transformers.PreTrainedTokenizerBase
    max_seq_length: int = 256  # actual model sequence length
    num_workers: int = 1
    mlm_probability: float = 0.15

    def __post_init__(self):
        # noinspection PyTypeChecker
        self.data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )
        self.max_text_length = (self.max_seq_length - 3) // 2

    def prep_bert_inputs(self, text_token_ids: List[int], tokenizer, data_collator):
        masked_tokens, masked_labels = data_collator.mask_tokens(
            inputs=torch.tensor([text_token_ids]),
            special_tokens_mask=torch.zeros(len(text_token_ids)),
        )

        masked_tokens = masked_tokens[0].tolist()
        masked_labels = masked_labels[0].tolist()
        # [DECODER] mask(x) [SEP] z [SEP]
        decoder_input = (
            [self.DECODER_TOKEN_ID] + masked_tokens + [tokenizer.sep_token_id]
            + [self.RESERVED_FOR_Z_TOKEN_ID] + [tokenizer.sep_token_id]
        )
        # [DECODER] x [SEP] _ [SEP]
        decoder_label = (
            [self.NON_MASKED_TARGET] + masked_labels + [self.NON_MASKED_TARGET] * 3
        )
        assert len(decoder_input) == len(decoder_label)
        # [PRIOR] mask(x) [SEP]
        prior_input = (
            [self.PRIOR_TOKEN_ID] + masked_tokens + [tokenizer.sep_token_id]
        )
        # [POSTERIOR] x [SEP] mask(x) [SEP]
        posterior_input = (
                [self.POSTERIOR_TOKEN_ID] + text_token_ids + [tokenizer.sep_token_id]
                + masked_tokens + [tokenizer.sep_token_id]
        )
        return {
            "decoder_input": decoder_input,
            "decoder_label": decoder_label,
            "decoder_z_index": len(decoder_input) - 2,
            "prior_input": prior_input,
            "posterior_input": posterior_input,
        }

    def prep_bert_inputs_apply(self, examples):
        return self.prep_bert_inputs(
            text_token_ids=examples["tokenized"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

    def collate_fn(self, examples):
        decoder_input_ls = [example["decoder_input"] for example in examples]
        decoder_label_ls = [example["decoder_label"] for example in examples]
        decoder_z_index_ls = [example["decoder_z_index"] for example in examples]
        prior_input_ls = [example["prior_input"] for example in examples]
        posterior_input_ls = [example["posterior_input"] for example in examples]

        decoder_input_outs = self.tokenizer.pad({"input_ids": decoder_input_ls}, return_tensors="pt")
        decoder_label_outs = self.tokenizer.pad({"input_ids": decoder_label_ls}, return_tensors="pt")
        prior_input_outs = self.tokenizer.pad({"input_ids": prior_input_ls}, return_tensors="pt")
        posterior_input_outs = self.tokenizer.pad({"input_ids": posterior_input_ls}, return_tensors="pt")
        # decoder_input_outs["attention_mask"] should be the same as decoder_label_outs["attention_mask"]
        return {
            "decoder_input": decoder_input_outs["input_ids"],
            "decoder_mask": decoder_input_outs["attention_mask"],
            "decoder_label": decoder_label_outs["input_ids"],
            "decoder_z_index": torch.tensor(decoder_z_index_ls).long(),
            "prior_input": prior_input_outs["input_ids"],
            "prior_mask": prior_input_outs["attention_mask"],
            "posterior_input": posterior_input_outs["input_ids"],
            "posterior_mask": posterior_input_outs["attention_mask"],
        }

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // self.max_text_length) * self.max_text_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.max_text_length] for i in range(0, total_length, self.max_text_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def prepare_vae_dataset(self, text_dataset, seed=None):
        tokenized_data = text_dataset.map(
            lambda example: {
                "tokenized": self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example["text"]))
            },
            batched=False,
            num_proc=self.num_workers,
            remove_columns=["title", "text"],
        )
        chunked_tokenized_data = tokenized_data.map(
            self.group_texts,
            batched=True,
            num_proc=self.num_workers,
        )
        with temporarily_set_rng(seed=seed):
            vae_dataset = chunked_tokenized_data.map(
                self.prep_bert_inputs_apply,
                batched=False,
                num_proc=self.num_workers,
                remove_columns=["tokenized"],
            )
        return vae_dataset

    def create_train_dataloader(self, train_vae_dataset, batch_size):
        train_dataloader = DataLoader(
            train_vae_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=0 if self.num_workers == 1 else self.num_workers,
        )
        return train_dataloader

    def create_val_dataloader(self, val_vae_dataset, batch_size):
        val_dataloader = DataLoader(
            val_vae_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=0 if self.num_workers == 1 else self.num_workers,
        )
        return val_dataloader

    def get_dummy_batch(self):
        dummy_format = "My {} has a pet {}. It often {} at {} when {} feeds it {}."
        relative_tups = [
            ("mother", "her", "she"),
            ("father", "him", "he"),
            ("uncle", "him", "he"),
            ("sister", "her", "she"),
        ]
        pet_tups = [
            ("dog", "barks", "treats"),
            ("cat", "meows", "fish"),
            ("crow", "caws", "seeds"),
        ]
        text_list = [
            dummy_format.format(
                relative_tup[0], pet_tup[0],
                pet_tup[1], relative_tup[1],
                relative_tup[2], pet_tup[2],
            )
            for relative_tup, pet_tup
            in itertools.product(relative_tups, pet_tups)
        ]
        masked_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(dummy_format.format(
            "[MASK]", "[MASK]",
            "[MASK] [MASK]", "[MASK]",
            "[MASK]", "[MASK]"
        )))

        prepped_inputs_list = []
        for text in text_list:
            text_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            masked_labels = [
                token_id
                if masked_token_id == self.MASK_TOKEN_ID
                else self.NON_MASKED_TARGET
                for token_id, masked_token_id
                in zip(text_token_ids, masked_tokens)
            ]
            prepped_inputs = {
                "decoder_input": (
                        [self.DECODER_TOKEN_ID] + masked_tokens + [self.tokenizer.sep_token_id]
                        + [self.RESERVED_FOR_Z_TOKEN_ID] + [self.tokenizer.sep_token_id]
                ),
                "decoder_label": (
                        [self.NON_MASKED_TARGET] + masked_labels + [self.NON_MASKED_TARGET] * 3
                ),
                "prior_input": (
                        [self.PRIOR_TOKEN_ID] + masked_tokens + [self.tokenizer.sep_token_id]
                ),
                "posterior_input": (
                        [self.POSTERIOR_TOKEN_ID] + text_token_ids + [self.tokenizer.sep_token_id]
                        + masked_tokens + [self.tokenizer.sep_token_id]
                ),
            }
            prepped_inputs["decoder_z_index"] = len(prepped_inputs["decoder_input"]) - 2,
            prepped_inputs_list.append(prepped_inputs)
        return self.collate_fn(prepped_inputs_list)


class BertVaeModel(nn.Module):
    def __init__(self, mlm_model):
        super().__init__()
        self.mlm_model = mlm_model
        self.hidden_size = mlm_model.config.hidden_size
        self.z_loc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.z_scale_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def prior_forward(self, batch):
        """
        [PRIOR] mask(x) [SEP]
          ->
        stat(z | mask(x))

        Requires:
            batch["prior_input"]
            batch["prior_mask"]
        """
        outputs = self.mlm_model.bert(
            input_ids=batch["prior_input"],
            attention_mask=batch["prior_mask"],
        )
        z_raw = outputs.last_hidden_state[:, 0, :]  # get CLS (first token)
        z_loc = self.z_loc_layer(z_raw)
        z_logvar = self.z_scale_layer(z_raw)
        return {
            "z_loc": z_loc,
            "z_logvar": z_logvar,
        }

    def posterior_forward(self, batch):
        """
        [POSTERIOR] x [SEP] mask(x) [SEP]
          ->
        stat(z | x, mask(x))

        Requires:
            batch["posterior_input"]
            batch["posterior_mask"]
        """
        outputs = self.mlm_model.bert(
            input_ids=batch["posterior_input"],
            attention_mask=batch["posterior_mask"],
        )
        z_raw = outputs.last_hidden_state[:, 0, :]  # get CLS (first token)
        z_loc = self.z_loc_layer(z_raw)
        z_logvar = self.z_scale_layer(z_raw)
        return {
            "z_loc": z_loc,
            "z_logvar": z_logvar,
        }

    def decoder_forward(self, batch, z_sample):
        """
        [DECODER] mask(x) [SEP] z [SEP]
          ->
        [DECODER] x [SEP] _ [SEP]

        Requires:
            batch["decoder_input"]
            batch["decoder_mask"]
            batch["decoder_z_index"]
        Optional:
            batch["decoder_label"]
        """
        token_only_embeddings = self.mlm_model.bert.embeddings.word_embeddings(batch["decoder_input"])
        batch_size = z_sample.shape[0]
        token_only_embeddings[
            torch.arange(batch_size),
            batch["decoder_z_index"],
        ] = z_sample
        outputs = self.mlm_model.bert(
            attention_mask=batch["decoder_mask"],
            inputs_embeds=token_only_embeddings,
        )
        prediction_scores = self.mlm_model.cls(outputs.last_hidden_state)
        masked_lm_loss = None
        if "decoder_label" in batch:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=BertDataWrapper.NON_MASKED_TARGET,  # Clean up?
                reduction="sum",
            )
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.mlm_model.config.vocab_size),
                target=batch["decoder_label"].view(-1)
            ) / batch_size
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, batch):
        posterior_output = self.posterior_forward(batch)
        prior_output = self.prior_forward(batch)
        z_sample = self.sample_z(z_loc=posterior_output["z_loc"], z_logvar=posterior_output["z_logvar"])
        mlm_output = self.decoder_forward(batch=batch, z_sample=z_sample)
        results = {
            "z_sample": z_sample,
            "z_loc": posterior_output["z_loc"],
            "z_logvar": posterior_output["z_logvar"],
            "logits": mlm_output.logits,
        }
        if "decoder_label" in batch:
            results["recon_loss"] = mlm_output.loss
            results["kl_loss_tensor"] = self.kl_loss_function(
                z_loc=posterior_output["z_loc"],
                z_logvar=posterior_output["z_logvar"],
                prior_z_loc=prior_output["z_loc"],
                prior_z_logvar=prior_output["z_logvar"],
                reduce=False,
            )
            results["kl_loss"] = results["kl_loss_tensor"].mean()
            results["total_loss"] = results["recon_loss"] + results["kl_loss"]
        import pdb; pdb.set_trace()
        return results

    @classmethod
    def sample_z(cls, z_loc, z_logvar):
        std = torch.exp(0.5 * z_logvar) + 1e-5
        eps = torch.randn_like(std)
        return z_loc + eps*std

    @classmethod
    def kl_loss_function(cls, z_loc, z_logvar, prior_z_loc=None, prior_z_logvar=None, reduce=True):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114, Section Appendix B
        # also https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        if prior_z_loc is None and prior_z_logvar is None:
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss_sum = -0.5 * torch.sum(1 + z_logvar - z_loc.pow(2) - z_logvar.exp(), dim=1)
        else:
            kl_loss_sum = -0.5 * torch.sum(
                1 + z_logvar - prior_z_logvar
                - (z_logvar.exp() + (z_loc - prior_z_loc).pow(2)) / prior_z_logvar.exp(),
                dim=1,
            )
        if reduce:
            kl_loss = kl_loss_sum.mean(0)
        else:
            kl_loss = kl_loss_sum
        return kl_loss



class BertVaeTrainer:
    def __init__(self,
                 bert_data_wrapper: BertDataWrapper,
                 bert_vae_model: BertVaeModel,
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
        self.optimizer = AdamW(self.bert_vae_model.parameters(), lr=self.args.learning_rate)
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
                        "kl_loss": train_kl_loss.item(),
                    },
                )
            if (step + 1) % self.args.eval_interval == 0:
                self.do_val(step)

    def do_val(self, step):
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
        print('[{}/{} ({:.0f}%)]\t V-Loss: {:.6f}\tV-Recon-L: {:.6f}\tV-KL-L: {:.6f}'.format(
            step, self.args.num_steps,
            100. * step / self.args.num_steps,
            agg_total_loss / total_size,
            agg_recon_loss / total_size,
            agg_kl_loss / total_size,
        ))
        import pdb; pdb.set_trace()
        self.log_writer.write_entry(
            "loss_val",
            {
                "step": step,
                "total_loss": agg_total_loss / total_size,
                "recon_loss": agg_recon_loss / total_size,
                "kl_loss": agg_kl_loss / total_size,
            },
        )
        batch = next(iter(self.val_dataloader))
        batch = move_to_device(batch, device=self.device)
        with torch.no_grad():
            vae_output = self.bert_vae_model(batch=batch)
        idx = 0
        display_dict = format_labeled_example(
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
            display_dict = format_labeled_example(
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


def move_to_device(batch, device):
    return {
        k: v.to(device)
        for k, v in batch.items()
    }


def cycle_dataloader(dataloader, num_steps):
    steps = 0
    while True:
        for batch in dataloader:
            yield batch
            steps += 1
            if steps == num_steps:
                return


def format_labeled_example(masked_token_ids, gold_labels, predictions, tokenizer):
    """
    e.g.
        masked_token_ids = batch["decoder_input"][0]
        gold_labels = batch["decoder_label"][0]
        predictions = decoder_output.logits[0].max(-1).indices
    """
    masked_token_ids = masked_token_ids.cpu().numpy()
    gold_labels = gold_labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    is_masked = gold_labels != BertDataWrapper.NON_MASKED_TARGET

    masked_token_list = tokenizer.convert_ids_to_tokens(masked_token_ids)
    gold_labels[~is_masked] = 0  # so convert_ids_to_tokens doesn't complain about -100
    gold_labels_token_list = tokenizer.convert_ids_to_tokens(gold_labels)
    predictions_token_list = tokenizer.convert_ids_to_tokens(predictions)

    gold_display_list = []
    pred_display_list = []
    gold_display_list_html = []
    pred_display_list_html = []
    for i, token in enumerate(masked_token_list):
        token_is_masked = is_masked[i]
        if token_is_masked:
            gold_token = gold_labels_token_list[i]
            pred_token = predictions_token_list[i]
            color = "green" if gold_token == pred_token else "red"
            gold_display_list.append(format_color(gold_token, color=color))
            pred_display_list.append(format_color(pred_token, color=color))
            gold_display_list_html.append(f"<span style='color:{color}'>{gold_token}</span>")
            pred_display_list_html.append(f"<span style='color:{color}'>{pred_token}</span>")
        else:
            gold_display_list.append(token)
            pred_display_list.append(token)
            gold_display_list_html.append(token)
            pred_display_list_html.append(token)
    return {
        "gold_str": " ".join(gold_display_list),
        "pred_str": " ".join(pred_display_list),
        "gold_str_html": " ".join(gold_display_list_html),
        "pred_str_html": " ".join(pred_display_list_html),
    }


def format_unlabeled_example(masked_token_ids, predictions, tokenizer):
    """
    e.g.
        masked_token_ids = batch["decoder_input"][0]
        predictions = decoder_output.logits[0].max(-1).indices
    """
    masked_token_ids = masked_token_ids.cpu().numpy()
    predictions = predictions.cpu().numpy()

    is_masked = masked_token_ids == BertDataWrapper.MASK_TOKEN_ID

    masked_token_list = tokenizer.convert_ids_to_tokens(masked_token_ids)
    predictions_token_list = tokenizer.convert_ids_to_tokens(predictions)

    pred_display_list = []
    for i, token in enumerate(masked_token_list):
        token_is_masked = is_masked[i]
        if token_is_masked:
            pred_token = predictions_token_list[i]
            pred_display_list.append(format_color(pred_token, color="red"))
        else:
            pred_display_list.append(token)
    pred_str = " ".join(pred_display_list)
    return {"pred_str": pred_str}


@contextlib.contextmanager
def temporarily_set_rng(seed=None):
    # Convenience: set None to disable
    if seed is None:
        yield
    else:
        rng_state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(rng_state)


def format_color(msg, color):
    color_dict = {
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
    }
    code = color_dict[color]
    return f"\x1b[{code}m{msg}\x1b[0m"
