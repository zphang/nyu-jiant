import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.utils.python.io as py_io
import jiant.utils.transformer_utils as transformer_utils
import jiant.utils.torch_utils as torch_utils
from jiant.shared.model_setup import ModelArchitectures


class ExtendedEncoder(nn.Module):
    def __init__(self, frozen_encoder, pooler):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.pooler = pooler

    @property
    def model_arch(self):
        return ModelArchitectures.from_encoder(self.frozen_encoder)

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
            # N x 1 x max_seq_length (1 = single vector)
            attn_layer_masks = attn_mask.unsqueeze(1).float()
        else:
            # N * num_heads x 1 x max_seq_length (1 = single vector)
            attn_layer_masks = attn_mask.unsqueeze(1) \
                .repeat(1, self.num_heads, 1) \
                .view(batch_size * self.num_heads, 1, -1).float()


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


class LayerWiseSelfAttentionPooler(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads=1, dropout_prob=0.5, conditional_query=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.conditional_query = conditional_query

        self.dropout = nn.Dropout(p=dropout_prob)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])
        self.self_linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.final_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads)
        self.final_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frozen_act, attn_mask):
        batch_size, max_seq_length, hidden_dim = frozen_act[0].shape
        device = frozen_act[0].device
        representation = []
        attn_layer_masks = create_multihead_attn_mask(attn_mask, self.num_heads)
        for i in range(self.num_layers):
            # msl, bs, hd
            layer_embedding = frozen_act[i].permute(1, 0, 2)
            attn_out, attn_weights = self.self_attn_layers[i](
                query=layer_embedding,
                key=layer_embedding,
                value=layer_embedding,
                attn_mask=attn_layer_masks,
            )
            layer_act = self.self_linear_layers[i](F.relu(attn_out))
            representation.append(layer_act)
        # msl, bs, hd
        sequence_act = torch.stack(representation, dim=-1).mean(-1)
        unpooled = self.final_linear(F.relu(sequence_act.permute(1, 0, 2)))

        final_layer_mask = attn_mask.unsqueeze(1) \
            .repeat(1, self.num_heads, 1) \
            .view(batch_size * self.num_heads, 1, -1).float()
        if self.conditional_query:
            query = mean_pool(sequence_act.permute(1, 0, 2), mask=attn_mask).unsqueeze(0)
        else:
            query = torch.ones(1, batch_size, self.hidden_dim).to(device)
        final_attended, final_attended_weights = self.final_attention(
            query=query,
            key=sequence_act,
            value=sequence_act,
            attn_mask=final_layer_mask,
        )
        pooled = self.final_linear(F.relu(final_attended)).squeeze(0)

        return taskmodels.EncoderOutput(
            pooled=pooled,
            unpooled=unpooled,
            other=None,
        )


class SeqLayerWiseAttentionPooler(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads=1, dropout_prob=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

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
            # N x 1 x max_seq_length (1 = single vector)
            attn_layer_masks = attn_mask.unsqueeze(1).float()
        else:
            # N * num_heads x 1 x max_seq_length (1 = single vector)
            attn_layer_masks = attn_mask.unsqueeze(1) \
                .repeat(1, self.num_heads, 1) \
                .view(batch_size * self.num_heads, 1, -1).float()

        query = torch.ones(1, batch_size, self.hidden_dim).to(device)
        for i in range(self.num_layers):
            # msl, bs, hd
            layer_embedding = frozen_act[i].permute(1, 0, 2)
            attn_out, attn_weights = self.attn_layers[i](
                query=query,
                key=layer_embedding,
                value=layer_embedding,
                attn_mask=attn_layer_masks,
            )
            representation.append(attn_out)
            query = attn_out[0:1]
        act = self.linear(F.relu(torch.stack(representation, dim=-1).squeeze(0).mean(-1)))
        return taskmodels.EncoderOutput(
            pooled=act,
            unpooled=None,
            other=None,
        )


def mean_pool(x, mask):
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(1)


def get_frozen_act(frozen_encoder, input_ids, input_mask, segment_ids):
    model_arch = ModelArchitectures.from_encoder(frozen_encoder)
    with transformer_utils.output_hidden_states_context(encoder=frozen_encoder):
        with torch.no_grad():
            frozen_encoder_output = frozen_encoder(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )
            if model_arch == ModelArchitectures.ROBERTA:
                _, _, frozen_act = frozen_encoder_output
            elif model_arch == ModelArchitectures.ALBERT:
                _, _, frozen_act = frozen_encoder_output
            elif model_arch == ModelArchitectures.ELECTRA:
                _, frozen_act = frozen_encoder_output
            else:
                raise KeyError(model_arch)
    return frozen_act


def create_encoder(model_type, pooler_config, device=None):
    if model_type.startswith("roberta-"):
        frozen_encoder = transformers.RobertaModel.from_pretrained(model_type).eval()
        torch_utils.set_requires_grad(frozen_encoder.named_parameters(), requires_grad=False)
    elif model_type.startswith("albert-"):
        frozen_encoder = transformers.AlbertModel.from_pretrained(model_type).eval()
        torch_utils.set_requires_grad(frozen_encoder.named_parameters(), requires_grad=False)
    elif model_type.startswith("electra-"):
        frozen_encoder = transformers.ElectraModel.from_pretrained(f"google/{model_type}").eval()
        torch_utils.set_requires_grad(frozen_encoder.named_parameters(), requires_grad=False)
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
    elif pooler_type == "LayerWiseSelfAttentionPooler":
        pooler_class = LayerWiseSelfAttentionPooler
    elif pooler_type == "SeqLayerWiseAttentionPooler":
        pooler_class = SeqLayerWiseAttentionPooler
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


def head_stack_mask(mask, num_heads):
    batch_size, max_seq_length = mask.shape
    return mask.unsqueeze(1).repeat(1, num_heads, 1).view(batch_size * num_heads, max_seq_length)


def create_multihead_attn_mask(mask, num_heads):
    # mask: bs x msl
    head_stacked_mask = head_stack_mask(mask, num_heads).float()
    mat1 = head_stacked_mask.unsqueeze(2)  # bs x msl x 1
    mat2 = head_stacked_mask.unsqueeze(1)  # bs x 1 x msl
    return torch.bmm(mat1, mat2).float()


# === HACKING === #
taskmodels.get_output_from_encoder = get_output_from_encoder
print("OVERRIDING jiant.proj.main.modeling.taskmodels.get_output_from_encoder")
# === END HACKING === #
