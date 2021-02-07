import torch
from typing import List
from dataclasses import dataclass
import transformers
from transformers.models.bert.modeling_bert import MaskedLMOutput
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


@dataclass
class BertDataWrapper:
    # Constants
    PRIOR_TOKEN_ID = 1  # [unused1]
    POSTERIOR_TOKEN_ID = 2  # [unused2]
    RESERVED_FOR_Z_TOKEN_ID = 3  # [unused3]
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    MASK_TOKEN_ID = 103
    NON_MASKED_TARGET = -100

    # Arguments
    tokenizer: transformers.PreTrainedTokenizerBase
    max_seq_length: int = 256  # actual model sequence length
    num_workers: int = 1
    batch_size: int = 32

    def __post_init__(self):
        self.data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        self.max_text_length = (self.max_seq_length - 3) // 2

    def prep_bert_inputs(self, text_token_ids: List[int], tokenizer, data_collator):
        masked_tokens, masked_labels = data_collator.mask_tokens(
            inputs=torch.tensor([text_token_ids]),
            special_tokens_mask=torch.zeros(len(text_token_ids)),
        )

        masked_tokens = masked_tokens[0].tolist()
        masked_labels = masked_labels[0].tolist()
        # [PRIOR] mask(x) [SEP] z [SEP]
        prior_input = (
                [self.PRIOR_TOKEN_ID] + masked_tokens + [tokenizer.sep_token_id]
                + [self.RESERVED_FOR_Z_TOKEN_ID] + [tokenizer.sep_token_id]
        )
        # [PRIOR] x [SEP] _ [SEP]
        prior_label = (
                [self.NON_MASKED_TARGET] + masked_labels + [self.NON_MASKED_TARGET] * 3
        )
        assert len(prior_input) == len(prior_label)
        # [POSTERIOR] x [SEP] mask(x) [SEP]
        posterior_input = (
                [self.POSTERIOR_TOKEN_ID] + text_token_ids + [tokenizer.sep_token_id]
                + masked_tokens + [tokenizer.sep_token_id]
        )
        return {
            "prior_input": prior_input,
            "prior_label": prior_label,
            "prior_z_index": len(prior_input) - 2,
            "posterior_input": posterior_input,
        }

    def prep_bert_inputs_apply(self, examples):
        return self.prep_bert_inputs(
            text_token_ids=examples["tokenized"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

    def collate_fn(self, examples):
        prior_input_ls = [example["prior_input"] for example in examples]
        prior_label_ls = [example["prior_label"] for example in examples]
        prior_z_index_ls = [example["prior_z_index"] for example in examples]
        posterior_input_ls = [example["posterior_input"] for example in examples]

        prior_input_outs = self.tokenizer.pad({"input_ids": prior_input_ls}, return_tensors="pt")
        prior_label_outs = self.tokenizer.pad({"input_ids": prior_label_ls}, return_tensors="pt")
        posterior_input_outs = self.tokenizer.pad({"input_ids": posterior_input_ls}, return_tensors="pt")
        # prior_input_outs["attention_mask"] should be the same as prior_label_outs["attention_mask"]
        return {
            "prior_input": prior_input_outs["input_ids"],
            "prior_label": prior_label_outs["input_ids"],
            "prior_mask": prior_input_outs["attention_mask"],
            "prior_z_index": torch.tensor(prior_z_index_ls).long(),
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

    def create_train_dataloader(self, train_dataset):
        tokenized_data = train_dataset.map(
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
        vae_data = chunked_tokenized_data.map(
            self.prep_bert_inputs_apply,
            batched=False,
            num_proc=self.num_workers,
            remove_columns=["tokenized"],
        )
        train_dataloader = DataLoader(
            vae_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=0 if self.num_workers == 1 else self.num_workers,
        )
        return train_dataloader

    def create_val_dataloader(self, train_dataset):
        tokenized_data = train_dataset.map(
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
        vae_data = chunked_tokenized_data.map(
            self.prep_bert_inputs_apply,
            batched=False,
            num_proc=1,  # For reproducibility?
            remove_columns=["tokenized"],
        )
        train_dataloader = DataLoader(
            vae_data,
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=0 if self.num_workers == 1 else self.num_workers,
        )
        return train_dataloader


class BertVaeModel(nn.Module):
    def __init__(self, mlm_model):
        super().__init__()
        self.mlm_model = mlm_model
        self.hidden_size = mlm_model.config.hidden_size
        self.z_loc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.z_scale_layer = nn.Linear(self.hidden_size, self.hidden_size)

    def posterior_forward(self, batch):
        """
        [POSTERIOR] x [SEP] mask(x) [SEP]
          ->
        stat(z)

        Requires:
            batch["posterior_input"]
            batch["posterior_mask"]
        """
        out = self.mlm_model.bert(
            input_ids=batch["posterior_input"],
            attention_mask=batch["posterior_mask"],
        )
        z_raw = out["last_hidden_state"][:, 0, :]  # get CLS (first token)
        z_loc = self.z_loc_layer(z_raw)
        z_logvar = self.z_scale_layer(z_raw)
        return {
            "z_loc": z_loc,
            "z_logvar": z_logvar,
        }

    def prior_forward(self, batch, z_sample):
        """
        [PRIOR] mask(x) [SEP] z [SEP]
          ->
        [PRIOR] x [SEP] _ [SEP]

        Requires:
            batch["prior_input"]
            batch["prior_mask"]
            batch["prior_z_index"]
        Optional:
            batch["prior_label"]
        """
        token_only_embeddings = self.mlm_model.bert.embeddings.word_embeddings(batch["prior_input"])
        batch_size = z_sample.shape[0]
        token_only_embeddings[
            torch.arange(batch_size),
            batch["prior_z_index"],
        ] = z_sample
        outputs = self.mlm_model.bert(
            attention_mask=batch["prior_mask"],
            inputs_embeds=token_only_embeddings,
        )
        prediction_scores = self.mlm_model.cls(outputs.last_hidden_state)
        masked_lm_loss = None
        if "prior_label" in batch:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=BertDataWrapper.NON_MASKED_TARGET,  # Clean up?
            )
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.mlm_model.config.vocab_size),
                target=batch["prior_label"].view(-1)
            )
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, batch):
        posterior_output = self.posterior_forward(batch)
        z_sample = self.sample_z(z_loc=posterior_output["z_loc"], z_logvar=posterior_output["z_logvar"])
        mlm_output = self.prior_forward(batch=batch, z_sample=z_sample)
        results = {
            "z_sample": z_sample,
            "z_loc": posterior_output["z_loc"],
            "z_logvar": posterior_output["z_logvar"],
            "logits": mlm_output.logits,
        }
        if "prior_label" in batch:
            results["recon_loss"] = mlm_output.loss
            results["kl_loss"] = self.kl_loss_function(
                z_loc=posterior_output["z_loc"],
                z_logvar=posterior_output["z_logvar"],
            )
            results["total_loss"] = results["recon_loss"] + results["kl_loss"]
        return results

    @classmethod
    def sample_z(cls, z_loc, z_logvar):
        std = torch.exp(0.5*z_logvar) + 1e-5
        eps = torch.randn_like(std)
        return z_loc + eps*std

    @classmethod
    def kl_loss_function(cls, z_loc, z_logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_loc.pow(2) - z_logvar.exp())
        return kl_loss


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


def display_example(masked_token_ids, true_labels, predictions, tokenizer):
    """
    e.g.
        masked_token_ids = batch["prior_input"][0]
        true_labels = batch["prior_label"][0]
        predictions = prior_output.logits[0].max(-1).indices
    """
    masked_token_ids = masked_token_ids.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    is_masked = true_labels != BertDataWrapper.NON_MASKED_TARGET

    masked_token_list = tokenizer.convert_ids_to_tokens(masked_token_ids)
    true_labels[~is_masked] = 0  # so convert_ids_to_tokens doesn't complain about -100
    true_labels_token_list = tokenizer.convert_ids_to_tokens(true_labels)
    predictions_token_list = tokenizer.convert_ids_to_tokens(predictions)

    gold_display_list = []
    pred_display_list = []
    for i, token in enumerate(masked_token_list):
        token_is_masked = is_masked[i]
        if token_is_masked:
            gold_display_list.append(f"[[{true_labels_token_list[i]}]]")
            pred_display_list.append(f"[[{predictions_token_list[i]}]]")
        else:
            gold_display_list.append(token)
            pred_display_list.append(token)
    gold_str = " ".join(gold_display_list)
    pred_str = " ".join(pred_display_list)
    return {"gold_str": gold_str, "pred_str": pred_str}
