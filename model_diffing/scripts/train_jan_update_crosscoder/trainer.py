from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import torch as t
import wandb
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.scripts.base_trainer import BaseTrainer
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateTrainConfig
from model_diffing.utils import (
    calculate_explained_variance_ML,
    calculate_reconstruction_loss,
    get_decoder_norms_H,
    get_explained_var_dict,
    l0_norm,
)


class JanUpdateCrosscoderTrainer(BaseTrainer[JanUpdateTrainConfig, JumpReLUActivation]):
    def _train_step(self, batch_BMLD: t.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BMLD)

        decoder_norms_H = get_decoder_norms_H(self.crosscoder.W_dec_HMLD)
        tanh_sparsity_loss = self._tanh_sparsity_loss(train_res.hidden_BH, decoder_norms_H)
        pre_act_loss = self._pre_act_loss(train_res.hidden_BH, decoder_norms_H)

        lambda_s = self._lambda_s_scheduler()
        scaled_tanh_sparsity_loss = lambda_s * tanh_sparsity_loss
        scaled_pre_act_loss = self.cfg.lambda_p * pre_act_loss

        loss = (
            reconstruction_loss  #
            + scaled_tanh_sparsity_loss
            + scaled_pre_act_loss
        )

        # backward
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()

        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        # metrics
        mean_l0 = l0_norm(train_res.hidden_BH, dim=-1).mean()
        explained_variance_ML = calculate_explained_variance_ML(batch_BMLD, train_res.reconstructed_acts_BMLD)

        thresholds_H = self.crosscoder.hidden_activation.log_threshold_H.exp().detach().cpu().numpy().tolist()
        thresholds_hist = wandb.Histogram(sequence=thresholds_H, num_bins=100)

        log_dict: dict[str, Any] = {
            "train/reconstruction_loss": reconstruction_loss.item(),
            #
            "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
            "train/tanh_sparsity_loss_scaled": scaled_tanh_sparsity_loss.item(),
            "train/lambda_s": lambda_s,
            #
            "train/mean_l0": mean_l0.item(),
            "train/mean_l0_pct": mean_l0.item() / self.crosscoder.hidden_dim,
            #
            "train/pre_act_loss": pre_act_loss.item(),
            "train/pre_act_loss_scaled": scaled_pre_act_loss.item(),
            "train/lambda_p": self.cfg.lambda_p,
            #
            "train/loss": loss.item(),
            #
            **get_explained_var_dict(explained_variance_ML, self.layers_to_harvest),
            #
            "media/jumprelu_threshold_distribution": thresholds_hist,
        }

        return log_dict

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        loss_BH = t.tanh(self.cfg.c * hidden_BH * decoder_norms_H)
        return loss_BH.sum(-1).mean()

    def _pre_act_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        t_H = self.crosscoder.hidden_activation.log_threshold_H
        loss_BH = t.relu(t_H.exp() - hidden_BH) * decoder_norms_H
        return loss_BH.sum(-1).mean()

    def _plot_threshold_distribution(self) -> go.Figure:
        fig = px.histogram(
            data_frame=self._jumprelu_thresholds(),
            nbins=50,
            title="Distribution of JumpReLU Thresholds",
            labels={"value": "Threshold Value"},
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Count",
        )

        return fig

    def _jumprelu_thresholds(self) -> np.ndarray[Any, Any]:
        log_thresholds = self.crosscoder.hidden_activation.log_threshold_H
        thresholds = log_thresholds.exp().detach().cpu().numpy()
        return thresholds
