from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch as t
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from wandb.sdk.wandb_run import Run

from model_diffing.data.token_layer_dataloader import BaseTokenLayerActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import save_config, validate_num_steps_per_epoch
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateTrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.utils import (
    calculate_explained_variance_X,
    calculate_reconstruction_loss,
    get_decoder_norms_H,
    get_explained_var_dict,
    l0_norm,
)


class BiTokenCCWrapper(nn.Module):
    def __init__(
        self,
        single_token_cc: AcausalCrosscoder[JumpReLUActivation],
        double_token_cc: AcausalCrosscoder[JumpReLUActivation],
    ):
        super().__init__()

        assert single_token_cc.crosscoding_dims[0] == 1  # token
        assert len(single_token_cc.crosscoding_dims) == 2  # (token, layer)
        self.single_cc = single_token_cc

        assert double_token_cc.crosscoding_dims[0] == 2  # token
        assert len(double_token_cc.crosscoding_dims) == 2  # (token, layer)
        self.double_cc = double_token_cc

    @dataclass
    class TrainResult:
        tok1_recon_B1LD: t.Tensor
        tok2_recon_B1LD: t.Tensor
        tok1_hidden_BH: t.Tensor
        tok2_hidden_BH: t.Tensor
        both_recon_B2LD: t.Tensor
        both_hidden_BH: t.Tensor

    def forward_train(self, x_BTLD: t.Tensor) -> TrainResult:
        assert x_BTLD.shape[1] == 2

        output_tok1 = self.single_cc.forward_train(x_BTLD[:, 0][:, None])
        output_tok2 = self.single_cc.forward_train(x_BTLD[:, 1][:, None])
        output_both = self.double_cc.forward_train(x_BTLD)

        return self.TrainResult(
            tok1_recon_B1LD=output_tok1.reconstructed_acts_BXD,
            tok2_recon_B1LD=output_tok2.reconstructed_acts_BXD,
            tok1_hidden_BH=output_tok1.hidden_BH,
            tok2_hidden_BH=output_tok2.hidden_BH,
            both_recon_B2LD=output_both.reconstructed_acts_BXD,
            both_hidden_BH=output_both.hidden_BH,
        )

    def forward(self, x_BTLD: t.Tensor) -> t.Tensor:
        return t.Tensor(0)


class SlidingWindowCrosscoderTrainer:
    def __init__(
        self,
        cfg: JanUpdateTrainConfig,
        activations_dataloader: BaseTokenLayerActivationsDataloader,
        crosscoders: BiTokenCCWrapper,
        wandb_run: Run | None,
        device: t.device,
        layers_to_harvest: list[int],
        experiment_name: str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.crosscoders = crosscoders

        self.optimizer = build_optimizer(cfg.optimizer, self.crosscoders.parameters())

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader.num_batches()
        )

        self.total_steps = self.num_steps_per_epoch * (cfg.epochs or 1)
        logger.info(
            f"Total steps: {self.total_steps} (num_steps_per_epoch: {self.num_steps_per_epoch}, epochs: {cfg.epochs})"
        )

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.num_steps_per_epoch)

        self.save_dir = Path(cfg.base_save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        save_config(self.cfg, self.save_dir)

        for _ in range(self.cfg.epochs or 1):
            epoch_dataloader_BTLD = self.activations_dataloader.get_shuffled_activations_iterator_BTLD()
            epoch_dataloader_BTLD = islice(epoch_dataloader_BTLD, self.num_steps_per_epoch)

            for batch_BTLD in epoch_dataloader_BTLD:
                batch_BTLD = batch_BTLD.to(self.device)

                self._train_step(batch_BTLD)

                # TODO(oli): get wandb checkpoint saving working

                # if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                #     for i, crosscoder in enumerate(self.crosscoders):
                #         with crosscoder.temporarily_fold_activation_scaling(
                #             self.activations_dataloader.get_norm_scaling_factors_ML()
                #         ):
                #             save_model(crosscoder, self.save_dir / f"crosscoder{i}", self.epoch, self.step)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BTLD.shape[0]

                self.step += 1
            self.epoch += 1

    def _train_step(self, batch_BTLD: t.Tensor) -> None:
        self.optimizer.zero_grad()

        # fwd
        res = self.crosscoders.forward_train(batch_BTLD)

        reconstructed_acts_BTLD = t.cat([res.tok1_recon_B1LD, res.tok2_recon_B1LD], dim=1) + res.both_recon_B2LD
        assert reconstructed_acts_BTLD.shape == batch_BTLD.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BTLD, reconstructed_acts_BTLD)

        hidden_B3H = t.cat([res.tok1_hidden_BH, res.both_hidden_BH, res.tok2_hidden_BH], dim=-1)

        decoder_norms_single_H = get_decoder_norms_H(self.crosscoders.single_cc.W_dec_HXD)
        decoder_norms_both_H = get_decoder_norms_H(self.crosscoders.double_cc.W_dec_HXD)

        decoder_norms_3H = t.cat([decoder_norms_single_H, decoder_norms_both_H, decoder_norms_single_H], dim=-1)

        tanh_sparsity_loss = self._tanh_sparsity_loss(hidden_B3H, decoder_norms_3H)
        pre_act_loss = self._pre_act_loss(hidden_B3H, decoder_norms_3H)

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
        clip_grad_norm_(self.crosscoders.parameters(), 1.0)
        self.optimizer.step()

        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        if (
            self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
            and self.wandb_run is not None
        ):
            mean_l0 = l0_norm(hidden_B3H, dim=-1).mean()

            thresholds_single_H = (
                self.crosscoders.single_cc.hidden_activation.log_threshold_H.exp().detach().cpu().numpy().tolist()
            )
            thresholds_both_H = (
                self.crosscoders.double_cc.hidden_activation.log_threshold_H.exp().detach().cpu().numpy().tolist()
            )

            thresholds_single_hist = wandb.Histogram(sequence=thresholds_single_H, num_bins=100)
            thresholds_both_hist = wandb.Histogram(sequence=thresholds_both_H, num_bins=100)

            # with t.no_grad():
            # cc1_pre_biases_BH = einsum(batch_BTLD, self.crosscoders.crosscoder1.W_enc_TLDH, "b t m l d, t m l d h -> b h")
            # cc1_pre_biases_hist = wandb.Histogram(sequence=cc1_pre_biases_BH.flatten().detach().cpu().numpy().tolist(), num_bins=100)

            # cc2_pre_biases_BH = einsum(batch_BTLD, self.crosscoders.crosscoder2.W_enc_TLDH, "b t m l d, t m l d h -> b h")
            # cc2_pre_biases_hist = wandb.Histogram(sequence=cc2_pre_biases_BH.flatten().detach().cpu().numpy().tolist(), num_bins=100)

            # activations_hist = wandb.Histogram(sequence=hidden_B3H.flatten().detach().cpu().numpy().tolist(), num_bins=100)

            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BTLD, reconstructed_acts_BTLD),
                ("token", [0, 1]),
                ("layer", self.layers_to_harvest),
            )

            log_dict: dict[str, Any] = {
                "train/reconstruction_loss": reconstruction_loss.item(),
                #
                "train/tanh_sparsity_loss": tanh_sparsity_loss.item(),
                "train/tanh_sparsity_loss_scaled": scaled_tanh_sparsity_loss.item(),
                "train/lambda_s": lambda_s,
                #
                "train/mean_l0": mean_l0.item(),
                "train/mean_l0_pct": mean_l0.item() / hidden_B3H.shape[1],
                #
                "train/pre_act_loss": pre_act_loss.item(),
                "train/pre_act_loss_scaled": scaled_pre_act_loss.item(),
                "train/lambda_p": self.cfg.lambda_p,
                #
                "train/loss": loss.item(),
                #
                **explained_variance_dict,
                #
                "media/jumprelu_threshold_distribution_single": thresholds_single_hist,
                "media/jumprelu_threshold_distribution_both": thresholds_both_hist,
                # "media/pre_bias_distribution": pre_biases_hist,
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            self.wandb_run.log(log_dict, step=self.step)

    def _lambda_s_scheduler(self) -> float:
        """linear ramp from 0 to lambda_s over the course of training"""
        return (self.step / self.total_steps) * self.cfg.final_lambda_s

    def _tanh_sparsity_loss(self, hidden_BH: t.Tensor, decoder_norms_H: t.Tensor) -> t.Tensor:
        loss_BH = t.tanh(self.cfg.c * hidden_BH * decoder_norms_H)
        return loss_BH.sum(-1).mean()

    def _pre_act_loss(self, hidden_B3H: t.Tensor, decoder_norms_3H: t.Tensor) -> t.Tensor:
        t_3H = t.cat(
            [
                self.crosscoders.single_cc.hidden_activation.log_threshold_H,
                self.crosscoders.double_cc.hidden_activation.log_threshold_H,
                self.crosscoders.single_cc.hidden_activation.log_threshold_H,
            ],
            dim=-1,
        )
        loss_3H = t.relu(t_3H.exp() - hidden_B3H) * decoder_norms_3H
        return loss_3H.sum(-1).mean()
