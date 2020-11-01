import sys
sys.path += ["/home/zp489/scratch/code/bowman/littlegits/ESIM/"]
import esim.model as esim_model_lib

import torch.nn
from esim.layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from esim.utils import get_mask, replace_masked

import jiant.tasks as tasks
import transformers
import numpy as np
import jiant.utils.transformer_utils as transformer_utils
import torch.nn as nn
import torch.nn.functional as F
import torch
import jiant.shared.caching as caching
from jiant.shared.runner import (
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
from jiant.utils.python.datastructures import InfiniteYield
import jiant.shared.model_setup as jiant_model_setup

device = torch.device("cuda:0")


class ESIM(nn.Module):
    def __init__(self,
                 hidden_size,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3):
        super(ESIM, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.hidden_size,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2 * 4 * self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(esim_model_lib._init_esim_weights)

    def forward(self,
                embedded_premises,
                premises_mask,
                premises_lengths,
                embedded_hypotheses,
                hypotheses_mask,
                hypotheses_lengths):

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses = \
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


class LayerWeights(nn.Module):
    def __init__(self, n_layers=13):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):
        weights = F.softmax(self.raw_weights, dim=0)
        return (x @ weights.unsqueeze(-1)).squeeze(-1)


class FullEsimModel(nn.Module):
    def __init__(self, encoder, esim_model, layer_weights):
        super().__init__()
        self.encoder = encoder
        self.esim_model = esim_model
        self.layer_weights = layer_weights

    def forward(self, batch):
        premises_lengths = batch.input_mask[:, 0].sum(-1)
        hypotheses_lengths = batch.input_mask[:, 1].sum(-1)
        max_premises_lengths = premises_lengths.max().item()
        max_hypotheses_lengths = hypotheses_lengths.max().item()
        premises_mask = batch.input_mask[:, 0, :max_premises_lengths]
        hypotheses_mask = batch.input_mask[:, 1, :max_hypotheses_lengths]
        with transformer_utils.output_hidden_states_context(encoder=self.encoder):
            with torch.no_grad():
                _, _, premise_hidden_act = self.encoder(
                    input_ids=batch.input_ids[:, 0, :max_premises_lengths],
                    token_type_ids=batch.segment_ids[:, 0, :max_premises_lengths],
                    attention_mask=premises_mask,
                )
                premise_hidden_act = torch.stack(premise_hidden_act, dim=3)
                _, _, hypothesis_hidden_act = self.encoder(
                    input_ids=batch.input_ids[:, 1, :max_hypotheses_lengths],
                    token_type_ids=batch.segment_ids[:, 1, :max_hypotheses_lengths],
                    attention_mask=hypotheses_mask,
                )
                hypothesis_hidden_act = torch.stack(hypothesis_hidden_act, dim=3)
        embedded_premises = self.layer_weights(premise_hidden_act)
        embedded_hypotheses = self.layer_weights(hypothesis_hidden_act)
        logits, probs = self.esim_model(
            embedded_premises=embedded_premises,
            premises_mask=premises_mask,
            premises_lengths=premises_lengths,
            embedded_hypotheses=embedded_hypotheses,
            hypotheses_mask=hypotheses_mask,
            hypotheses_lengths=hypotheses_lengths
        )
        return logits, probs

    def train(self):
        self.encoder.eval()
        self.esim_model.train()
        self.layer_weights.train()

    def eval(self):
        self.encoder.eval()
        self.esim_model.eval()
        self.layer_weights.eval()


def evaluate(full_model, eval_dataloader):
    preds = []
    golds = []
    full_model.eval()
    with torch.no_grad():
        for batch, batch_metadata in eval_dataloader:
            batch = batch.to(device)
            logits, probs = full_model(batch)
            preds.append(logits.detach().cpu().numpy().argmax(1))
            golds.append(batch.label_id.cpu().numpy())
    preds = np.concatenate(preds)
    golds = np.concatenate(golds)
    return (preds == golds).mean(), preds, golds


def main():
    encoder = transformers.BertModel.from_pretrained("bert-base-cased")
    encoder = encoder.eval()
    for n, p in encoder.named_parameters():
        p.requires_grad = False
    layer_weights = LayerWeights(13)
    esim_model = ESIM(hidden_size=768)

    full_model = FullEsimModel(
        encoder=encoder,
        esim_model=esim_model,
        layer_weights=layer_weights,
    ).to(device)

    train_cache = caching.ChunkedFilesDataCache(
        "/home/zp489/scratch/working/v1/2010/21_mnli_esim/cache/train/"
    )
    val_cache = caching.ChunkedFilesDataCache(
        "/home/zp489/scratch/working/v1/2010/21_mnli_esim/cache/val/"
    )

    optimizer_scheduler = jiant_model_setup.create_optimizer_from_params(
        named_parameters=full_model.named_parameters(),
        learning_rate=3e-4,
        t_total=300000,
        warmup_steps=None,
        warmup_proportion=0.1,
    )
    # optimizer = optim.AdamW(
    #     [p for p in full_model.parameters() if p.requires_grad],
    #     lr=1e-4,
    #     eps=1e-8,
    # )

    train_dataloader = InfiniteYield(
        get_train_dataloader_from_cache(
            train_cache=train_cache,
            task=tasks.HellaSwagTask,
            train_batch_size=4,
        )
    )
    sub_eval_dataloader = get_eval_dataloader_from_cache(
        eval_cache=val_cache,
        task=tasks.HellaSwagTask,
        eval_batch_size=8,
        subset_num=500,
    )
    eval_dataloader = get_eval_dataloader_from_cache(
        eval_cache=val_cache,
        task=tasks.HellaSwagTask,
        eval_batch_size=8,
    )

    best_acc = None
    best_weights = None
    acc_hist = []
    for i, (batch, batch_metadata) in zip(range(300000), train_dataloader):
        full_model.train()
        batch = batch.to(device)
        logits, probs = full_model(batch)
        loss = F.cross_entropy(logits, batch.label_id)
        loss.backward()
        optimizer_scheduler.step()
        optimizer_scheduler.optimizer.zero_grad()
        if (i + 1) % 1000 == 0:
            acc, preds, golds = evaluate(full_model, sub_eval_dataloader)
            print(i, acc)
            sys.stdout.flush()
            acc_hist.append(acc)

            if best_acc is None or acc > best_acc:
                state_dict = {
                    n: p.cpu()
                    for n, p in full_model.state_dict().items()
                    if not n.startswith("encoder.")
                }
                torch.save(
                    state_dict,
                    "/home/zp489/scratch/working/v1/2010/21_mnli_esim/runs/v2/best_weights.p",
                )
                torch.save(
                    {"acc": acc, "i": i},
                    "/home/zp489/scratch/working/v1/2010/21_mnli_esim/runs/v2/best_metadata.p",
                )
                best_acc = acc
                best_weights = state_dict
                print("  saving best")
                sys.stdout.flush()

    if best_weights is not None:
        full_model.load_state_dict(best_weights, strict=False)
        full_model = full_model.to(device)
    print(evaluate(full_model, eval_dataloader))
    torch.save(
        acc_hist,
        "/home/zp489/scratch/working/v1/2010/21_mnli_esim/runs/v2/acc_history.p",
    )


if __name__ == "__main__":
    main()
