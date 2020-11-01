import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.utils.python.io as py_io
import jiant.utils.transformer_utils as transformer_utils


class ExtendedEncoder(nn.Module):
    def __init__(self, frozen_encoder, pooler):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.pooler = pooler

    def forward(self, input_ids, input_mask, segment_ids):
        frozen_act = get_frozen_act(
            frozen_encoder=self.frozen_encoder,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
        )
        return self.pooler(frozen_act=frozen_act, attn_mask=input_mask)

    @property
    def config(self):
        return self.frozen_encoder.config


class SimplePooler(nn.Module):
    def __init__(self, num_layers, hidden_dim, dropout_prob=0.5, full_ffn=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.full_ffn = full_ffn

        self.dropout = nn.Dropout(p=dropout_prob)
        if self.full_ffn:
            self.weights = None
            self.linear = nn.Linear(hidden_dim * num_layers, hidden_dim)
        else:
            self.weights = nn.Parameter(torch.zeros(num_layers))
            self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frozen_act, attn_mask):
        batch_size, max_seq_length, hidden_dim = frozen_act[0].shape

        # bs, hd, nl
        stacked = torch.stack([
            mean_pool(x=layer_embedding, mask=attn_mask)
            for layer_embedding in frozen_act
        ], dim=-1)
        if self.full_ffn:
            act = self.linear(F.relu(stacked.reshape(batch_size, -1)))
        else:
            act = self.linear(F.relu((stacked @ F.softmax(self.weights, dim=0).unsqueeze(1)).squeeze(-1)))

        return taskmodels.EncoderOutput(
            pooled=act,
            unpooled=None,
            other=None,
        )


class LayerWiseAttentionPooler(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads=1, dropout_prob=0.5, conditional_query=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.conditional_query = conditional_query

        self.dropout = nn.Dropout(p=dropout_prob)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frozen_act, attn_mask):
        batch_size, max_seq_length, hidden_dim = frozen_act[0].shape
        device = frozen_act[0].device
        representation = []

        if self.num_heads == 1:
            attn_layer_masks = attn_mask.unsqueeze(1)
        else:
            attn_layer_masks = attn_mask.unsqueeze(1).repeat(self.num_heads, 1, 1)

        for i in range(self.num_layers):
            # msl, bs, hd
            layer_embedding = frozen_act[i].permute(1, 0, 2)
            if self.conditional_query:
                query = mean_pool(frozen_act[i], mask=attn_mask).unsqueeze(0)
            else:
                query = torch.ones(1, batch_size, self.hidden_dim).to(device)
            attn_out, attn_weights = self.attn_layers[i](
                query=query,
                key=layer_embedding,
                value=layer_embedding,
                attn_mask=attn_layer_masks,
            )
            representation.append(attn_out)
        act = self.linear(F.relu(torch.stack(representation, dim=-1).squeeze(0).mean(-1)))
        return taskmodels.EncoderOutput(
            pooled=act,
            unpooled=None,
            other=None,
        )


# class LearnedWeights(nn.Module):
#     def __init__(self, num):
#         super().__init__()
#         self.num = num
#
#
#     def forward(self, x, dim):
#



def mean_pool(x, mask):
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(1)


def get_frozen_act(frozen_encoder, input_ids, input_mask, segment_ids):
    with transformer_utils.output_hidden_states_context(encoder=frozen_encoder):
        with torch.no_grad():
            _, _, frozen_act = frozen_encoder(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )
    return frozen_act


def create_encoder(model_type, pooler_config, device=None):
    if model_type.startswith("roberta-"):
        frozen_encoder = transformers.RobertaModel.from_pretrained(model_type).eval()
    else:
        raise KeyError(model_type)
    if isinstance(pooler_config, str):
        pooler_config = py_io.read_json(pooler_config)
    pooler_config = pooler_config.copy()
    pooler_type = pooler_config.pop("pooler_type")
    if pooler_type == "SimplePooler":
        pooler_class = SimplePooler
    elif pooler_type == "LayerWiseAttentionPooler":
        pooler_class = LayerWiseAttentionPooler
    else:
        raise KeyError(pooler_type)
    pooler = pooler_class(**pooler_config)
    encoder = ExtendedEncoder(frozen_encoder=frozen_encoder, pooler=pooler)
    if device is not None:
        encoder = encoder.to(device)
    return encoder


def get_output_from_encoder(encoder, input_ids, segment_ids, input_mask) -> taskmodels.EncoderOutput:
    return encoder(
        input_ids=input_ids,
        segment_ids=segment_ids,
        input_mask=input_mask
    )


# === HACKING === #
taskmodels.get_output_from_encoder = get_output_from_encoder
print("OVERRIDING jiant.proj.main.modeling.taskmodels.get_output_from_encoder")
# === END HACKING === #
