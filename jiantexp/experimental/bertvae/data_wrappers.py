import datasets
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
import transformers
from torch.utils.data.dataloader import DataLoader
import itertools
from jiantexp.experimental.bertvae.utils import temporarily_set_rng


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
    num_workers: int = 0
    mlm_probability: float = 0.15

    def __post_init__(self):
        # noinspection PyTypeChecker
        self.max_text_length = (self.max_seq_length - 3) // 2

    def prep_bert_inputs(self, text_token_ids: List[int], tokenizer):
        masked_tokens, masked_labels = mask_tokens(
            tokenizer=self.tokenizer,
            inputs=torch.tensor([text_token_ids]),
            special_tokens_mask=torch.zeros(len(text_token_ids)),
            mlm_probability=self.mlm_probability
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
        # [POSTERIOR] x [SEP]
        posterior_input = (
            [self.POSTERIOR_TOKEN_ID] + text_token_ids + [tokenizer.sep_token_id]
        )
        # [0] masked_x==MASK_TOKEN_ID [0]
        #   Use for both prior and posterior
        is_masked = [1 if token_id == self.MASK_TOKEN_ID else 0 for token_id in masked_tokens]
        prior_token_type_ids = (
            [0] + is_masked + [0]
        )
        return {
            "decoder_input": decoder_input,
            "decoder_label": decoder_label,
            "decoder_z_index": len(decoder_input) - 2,
            "prior_input": prior_input,
            "posterior_input": posterior_input,
            "prior_token_type_ids": prior_token_type_ids,
        }

    def prep_bert_inputs_apply(self, examples):
        return self.prep_bert_inputs(
            text_token_ids=examples["tokenized"],
            tokenizer=self.tokenizer,
        )

    def collate_fn(self, examples):
        decoder_input_ls = [example["decoder_input"] for example in examples]
        decoder_label_ls = [example["decoder_label"] for example in examples]
        decoder_z_index_ls = [example["decoder_z_index"] for example in examples]
        prior_input_ls = [example["prior_input"] for example in examples]
        posterior_input_ls = [example["posterior_input"] for example in examples]
        prior_token_type_ids = [example["prior_token_type_ids"] for example in examples]

        decoder_input_outs = self.tokenizer.pad({"input_ids": decoder_input_ls}, return_tensors="pt")
        decoder_label_outs = self.tokenizer.pad({"input_ids": decoder_label_ls}, return_tensors="pt")
        decoder_label = decoder_label_outs["input_ids"]
        decoder_label[decoder_label == self.tokenizer.pad_token_id] = self.NON_MASKED_TARGET
        prior_input_outs = self.tokenizer.pad({
            "input_ids": prior_input_ls,
            "token_type_ids": prior_token_type_ids,
        }, return_tensors="pt")
        posterior_input_outs = self.tokenizer.pad({
            "input_ids": posterior_input_ls,
            "token_type_ids": prior_token_type_ids,
        }, return_tensors="pt")
        # decoder_input_outs["attention_mask"] should be the same as decoder_label_outs["attention_mask"]
        return {
            "decoder_input": decoder_input_outs["input_ids"],
            "decoder_mask": decoder_input_outs["attention_mask"],
            "decoder_label": decoder_label,
            "decoder_z_index": torch.tensor(decoder_z_index_ls).long(),
            "prior_input": prior_input_outs["input_ids"],
            "prior_mask": prior_input_outs["attention_mask"],
            "prior_token_type_ids": prior_input_outs["token_type_ids"],
            "posterior_input": posterior_input_outs["input_ids"],
            "posterior_mask": posterior_input_outs["attention_mask"],
            "posterior_token_type_ids": posterior_input_outs["token_type_ids"],
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

    def get_datasets(self, args):
        # Todo: clean up use of args
        raise NotImplementedError()

    def prepare_vae_dataset(self, text_dataset, seed=None):
        raise NotImplementedError()

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

    def create_val_dataloader(self, val_vae_dataset, batch_size, eval_multiplier=2):
        val_dataloader = DataLoader(
            val_vae_dataset,
            batch_size=batch_size * eval_multiplier,
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
                ),
                "prior_token_type_ids": (
                    [0] + [1 if token_id == self.MASK_TOKEN_ID else 0 for token_id in masked_tokens] + [0]
                ),
            }
            prepped_inputs["posterior_token_type_ids"] = prepped_inputs["prior_token_type_ids"]
            prepped_inputs["decoder_z_index"] = len(prepped_inputs["decoder_input"]) - 2,
            prepped_inputs_list.append(prepped_inputs)
        return self.collate_fn(prepped_inputs_list)


def format_labeled_example(masked_token_ids, gold_labels, predictions, tokenizer):
    """
    e.g.
        masked_token_ids = batch["decoder_input"][idx]
        gold_labels = batch["decoder_label"][idx]
        predictions = decoder_output.logits[idx].max(-1).indices
    """
    masked_token_ids = masked_token_ids.cpu().numpy().copy()
    gold_labels = gold_labels.cpu().numpy().copy()
    predictions = predictions.cpu().numpy().copy()

    num_valid_tokens = (gold_labels != 0).sum().item()
    masked_token_ids = masked_token_ids[:num_valid_tokens]
    predictions = predictions[:num_valid_tokens]
    masked_token_ids = masked_token_ids[:num_valid_tokens]
    is_masked = gold_labels != BertDataWrapper.NON_MASKED_TARGET

    masked_token_list = tokenizer.convert_ids_to_tokens(masked_token_ids)
    gold_labels[~is_masked] = 0  # so convert_ids_to_tokens doesn't complain about -100
    gold_labels_token_list = tokenizer.convert_ids_to_tokens(gold_labels)
    predictions_token_list = tokenizer.convert_ids_to_tokens(predictions)

    gold_display_list = []
    pred_display_list = []
    gold_display_list_html = []
    pred_display_list_html = []
    gold_tokens_list = []
    pred_tokens_list = []
    for i, token in enumerate(masked_token_list):
        token_is_masked = is_masked[i]
        if token_is_masked:
            gold_token = gold_labels_token_list[i]
            pred_token = predictions_token_list[i]
            longer_token_length = max(len(gold_token), len(pred_token))
            gold_token = gold_token.center(longer_token_length)
            pred_token = pred_token.center(longer_token_length)

            color = "green" if gold_token == pred_token else "red"
            gold_display_list.append(format_color(gold_token, color=color))
            pred_display_list.append(format_color(pred_token, color=color))
            gold_display_list_html.append(f"<span style='color:{color}'>{gold_token}</span>")
            pred_display_list_html.append(f"<span style='color:{color}'>{pred_token}</span>")
            gold_tokens_list.append(gold_token)
            pred_tokens_list.append(pred_token)
        else:
            gold_display_list.append(token)
            pred_display_list.append(token)
            gold_display_list_html.append(token)
            pred_display_list_html.append(token)
            gold_tokens_list.append(token)
            pred_tokens_list.append(token)
    return {
        "gold_str": " ".join(gold_display_list),
        "pred_str": " ".join(pred_display_list),
        "gold_str_html": " ".join(gold_display_list_html),
        "pred_str_html": " ".join(pred_display_list_html),
        "gold_tokens": gold_display_list,
        "pred_tokens": pred_tokens_list,
    }


def format_unlabeled_example(masked_token_ids, predictions, tokenizer):
    """
    e.g.
        masked_token_ids = batch["decoder_input"][0]
        predictions = decoder_output.logits[0].max(-1).indices
    """
    masked_token_ids = masked_token_ids.cpu().numpy().copy()
    predictions = predictions.cpu().numpy().copy()

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


def format_label_only(masked_token_ids, gold_labels, tokenizer):
    """
    e.g.
        masked_token_ids = batch["decoder_input"][idx]
        gold_labels = batch["decoder_label"][idx]
    """
    masked_token_ids = masked_token_ids.cpu().numpy().copy()
    gold_labels = gold_labels.cpu().numpy().copy()

    num_valid_tokens = (gold_labels != 0).sum().item()
    masked_token_ids = masked_token_ids[:num_valid_tokens]
    masked_token_ids = masked_token_ids[:num_valid_tokens]
    is_masked = gold_labels != BertDataWrapper.NON_MASKED_TARGET

    masked_token_list = tokenizer.convert_ids_to_tokens(masked_token_ids)
    gold_labels[~is_masked] = 0  # so convert_ids_to_tokens doesn't complain about -100
    gold_labels_token_list = tokenizer.convert_ids_to_tokens(gold_labels)

    gold_display_list = []
    gold_display_list_html = []
    gold_tokens_list = []
    for i, token in enumerate(masked_token_list):
        token_is_masked = is_masked[i]
        if token_is_masked:
            gold_token = gold_labels_token_list[i]
            gold_display_list.append(format_color(gold_token, color="blue"))
            gold_display_list_html.append(f"<span style='color:blue'>{gold_token}</span>")
            gold_tokens_list.append(gold_token)
        else:
            gold_display_list.append(token)
            gold_display_list_html.append(token)
            gold_tokens_list.append(token)
    return {
        "gold_str": " ".join(gold_display_list),
        "gold_str_html": " ".join(gold_display_list_html),
        "gold_tokens": gold_display_list,
    }


def format_color(msg, color):
    color_dict = {
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
    }
    code = color_dict[color]
    return f"\x1b[{code}m{msg}\x1b[0m"


def make_single_example_batch(batch, idx=0, batch_size=None):
    if batch_size is None:
        batch_size = len(batch["decoder_input"])
    new_batch = {}
    for k, v in batch.items():
        v_single = v[idx: idx+1]
        new_batch[k] = torch.cat([v_single] * batch_size, dim=0)
    return new_batch


class WikiDataWrapper(BertDataWrapper):
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

    @classmethod
    def get_datasets(cls, args):
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
        return {
            "train": wiki_train_data,
            "val": wiki_val_data,
        }


class YelpDataWrapper(BertDataWrapper):

    @classmethod
    def get_datasets(cls, args):
        yelp_train_data = datasets.load_dataset(
            "yelp_polarity",
            split=datasets.ReadInstruction('train', from_=args.train_from, to=args.train_to, unit="abs"),
        )
        yelp_val_data = datasets.load_dataset(
            "yelp_polarity",
            split=datasets.ReadInstruction('train', from_=args.val_from, to=args.val_to, unit="abs"),
        )
        return {
            "train": yelp_train_data,
            "val": yelp_val_data,
        }

    def prepare_vae_dataset(self, text_dataset, seed=None):
        from nltk import tokenize
        tokenized_data = text_dataset.map(
            lambda examples_batch: {
                "tokenized": [
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))[:self.max_text_length]
                    for raw_text in examples_batch["text"]
                    for line in raw_text.encode('utf8').decode('unicode_escape').splitlines()
                    for sent in tokenize.sent_tokenize(line)
                    if len(line) > 50
                ]
            },
            batched=True,
            num_proc=self.num_workers,
            remove_columns=list(text_dataset[0].keys()),
        )
        with temporarily_set_rng(seed=seed):
            vae_dataset = tokenized_data.map(
                self.prep_bert_inputs_apply,
                batched=False,
                num_proc=self.num_workers,
                remove_columns=["tokenized"],
            )
        return vae_dataset


def get_data_wrapper_class(data_name):
    return {
        "wiki": WikiDataWrapper,
        "yelp": YelpDataWrapper,
    }[data_name]


def get_data_wrapper(data_name, **kwargs):
    data_wrapper_class = get_data_wrapper_class(data_name)
    return data_wrapper_class(**kwargs)


def mask_tokens(
        tokenizer,
        inputs: torch.Tensor,
        mlm_probability: float = 0.15,
        special_tokens_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    From: DataCollatorForLanguageModeling.mask_tokens
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return inputs, labels
