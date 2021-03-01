import os
import torch
from typing import List, Any
from dataclasses import dataclass
import transformers
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import contextlib
import itertools
from transformers.models.bert.modeling_bert import MaskedLMOutput
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import jiant.utils.display as display
import jiantexp.experimental.bertvae.kl_weight_schedulers as kl_weight_schedulers
import jiantexp.experimental.bertvae.data_wrappers as data_wrappers


class BertVaeModel(nn.Module):
    def __init__(self, mlm_model, latent_token_mode="zindex", add_latent_linear=False):
        super().__init__()
        self.mlm_model = mlm_model
        self.hidden_size = mlm_model.config.hidden_size
        self.z_loc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.z_scale_layer = nn.Linear(self.hidden_size, self.hidden_size)

        self.latent_token_mode = latent_token_mode
        self.add_latent_linear = add_latent_linear

        if self.add_latent_linear:
            self.latent_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def prior_forward(self, batch):
        """
        [PRIOR] mask(x) [SEP]
          ->
        stat(z | mask(x))

        Requires:
            batch["prior_input"]
            batch["prior_mask"]
            batch["token_type_ids"]
        """
        outputs = self.mlm_model.bert(
            input_ids=batch["prior_input"],
            attention_mask=batch["prior_mask"],
            token_type_ids=batch["prior_token_type_ids"],
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
            token_type_ids=batch["posterior_token_type_ids"],
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
        if self.add_latent_linear:
            z_sample = self.latent_linear(z_sample)
        if self.latent_token_mode == "zindex":
            token_only_embeddings[torch.arange(batch_size), batch["decoder_z_index"]] = z_sample
            token_type_ids = None
        elif self.latent_token_mode == "zindex_toktype":
            token_only_embeddings[torch.arange(batch_size), batch["decoder_z_index"]] = z_sample
            token_type_ids = torch.zeros_like(batch["decoder_input"]).long()
            token_type_ids[torch.arange(batch_size), batch["decoder_z_index"]] = 1
            token_type_ids = token_type_ids.to(batch["decoder_input"].device)
        elif self.latent_token_mode == "cls":
            token_only_embeddings[:, 0] = z_sample
            token_type_ids = None
        else:
            raise KeyError(self.latent_token_mode)
        outputs = self.mlm_model.bert(
            attention_mask=batch["decoder_mask"],
            inputs_embeds=token_only_embeddings,
            token_type_ids=token_type_ids,
        )
        prediction_scores = self.mlm_model.cls(outputs.last_hidden_state)
        masked_lm_loss = None
        if "decoder_label" in batch:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=data_wrappers.BertDataWrapper.NON_MASKED_TARGET,  # Clean up?
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