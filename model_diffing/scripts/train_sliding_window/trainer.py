from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch as t
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from wandb.sdk.wandb_run import Run

from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import BaseTrainer, save_config, validate_num_steps_per_epoch
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateTrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.utils import (
    calculate_explained_variance_TML,
    calculate_reconstruction_loss,
    get_decoder_norms_H,
    get_explained_var_dict,
    l0_norm,
)


class BiTokenCCWrapper(nn.Module):
    def __init__(
        self, crosscoder1: AcausalCrosscoder[JumpReLUActivation], crosscoder2: AcausalCrosscoder[JumpReLUActivation]
    ):
        super().__init__()

        assert crosscoder1.n_tokens == 1
        self.crosscoder1 = crosscoder1

        assert crosscoder2.n_tokens == 2
        self.crosscoder2 = crosscoder2

    @dataclass
    class TrainResult:
        tok1_recon_B1MLD: t.Tensor
        tok2_recon_B1MLD: t.Tensor
        tok1_hidden_BH: t.Tensor
        tok2_hidden_BH: t.Tensor
        both_recon_B2MLD: t.Tensor
        both_hidden_BH: t.Tensor

    def forward_train(self, x_BTMLD: t.Tensor) -> TrainResult:
        assert x_BTMLD.shape[1] == 2

        # single_tok_Bt1MLD = rearrange(x_BTMLD, "b t m l d -> (b t) 1 m l d")

        # res1_Bt1MLD = self.crosscoders[0].forward_train(single_tok_Bt1MLD)
        # res1_BTMLD = rearrange(res1_Bt1MLD.reconstructed_acts_BTMLD, '(b t) 1 m l d -> b t m l d', t=2)
        output_tok1 = self.crosscoder1.forward_train(x_BTMLD[:, 0][:, None])
        output_tok2 = self.crosscoder1.forward_train(x_BTMLD[:, 1][:, None])
        output_both = self.crosscoder2.forward_train(x_BTMLD)

        return self.TrainResult(
            tok1_recon_B1MLD=output_tok1.reconstructed_acts_BTMLD,
            tok2_recon_B1MLD=output_tok2.reconstructed_acts_BTMLD,
            tok1_hidden_BH=output_tok1.hidden_BH,
            tok2_hidden_BH=output_tok2.hidden_BH,
            both_recon_B2MLD=output_both.reconstructed_acts_BTMLD,
            both_hidden_BH=output_both.hidden_BH,
        )

    def forward(self, x_BTMLD: t.Tensor) -> t.Tensor:
        return t.Tensor(0)


class SlidingWindowCrosscoderTrainer(BaseTrainer[Any, Any]):
    def __init__(
        self,
        cfg: JanUpdateTrainConfig,
        activations_dataloader: BaseActivationsDataloader,
        crosscoders: BiTokenCCWrapper,
        wandb_run: Run | None,
        device: t.device,
        layers_to_harvest: list[int],
        experiment_name: str,
    ):
        self.cfg = cfg
        self.activations_dataloader = activations_dataloader
        # self.crosscoders = crosscoders
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.crosscoders = crosscoders

        self.optimizer = build_optimizer(cfg.optimizer, self.crosscoders.parameters())

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader
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
            epoch_dataloader = self.activations_dataloader.get_shuffled_activations_iterator()
            epoch_dataloader = islice(epoch_dataloader, self.num_steps_per_epoch)

            for example_BX in epoch_dataloader:
                batch_BX = example_BX.to(self.device)

                train_result_dict = self._train_step(batch_BX)

                # TODO(oli): get wandb checkpoint saving working

                # if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                #     for i, crosscoder in enumerate(self.crosscoders):
                #         with crosscoder.temporarily_fold_activation_scaling(
                #             self.activations_dataloader.get_norm_scaling_factors_ML()
                #         ):
                #             save_model(crosscoder, self.save_dir / f"crosscoder{i}", self.epoch, self.step)

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BX.shape[0]

                self.step += 1
            self.epoch += 1

    def _train_step(self, batch_BX: t.Tensor):
        self.optimizer.zero_grad()

        # fwd
        res = self.crosscoders.forward_train(batch_BX)

        reconstructed_acts_BTMLD = t.cat([res.tok1_recon_B1MLD, res.tok2_recon_B1MLD], dim=1) + res.both_recon_B2MLD
        assert reconstructed_acts_BTMLD.shape == batch_BX.shape, "fuck"

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BX, reconstructed_acts_BTMLD)

        hidden_B3H = t.cat([res.tok1_hidden_BH, res.both_hidden_BH, res.tok2_hidden_BH], dim=-1)

        decoder_norms_single_H = get_decoder_norms_H(self.crosscoders.crosscoder1.W_dec_HTMLD)
        decoder_norms_both_H = get_decoder_norms_H(self.crosscoders.crosscoder2.W_dec_HTMLD)

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

        if self.cfg.log_every_n_steps and self.step % self.cfg.log_every_n_steps == 0:
            # metrics
            mean_l0 = l0_norm(hidden_B3H, dim=-1).mean()
            explained_variance_TML = calculate_explained_variance_TML(batch_BX, reconstructed_acts_BTMLD)

            thresholds_single_H = (
                self.crosscoders.crosscoder1.hidden_activation.log_threshold_H.exp().detach().cpu().numpy().tolist()
            )
            thresholds_both_H = (
                self.crosscoders.crosscoder2.hidden_activation.log_threshold_H.exp().detach().cpu().numpy().tolist()
            )

            thresholds_single_hist = wandb.Histogram(sequence=thresholds_single_H, num_bins=100)
            thresholds_both_hist = wandb.Histogram(sequence=thresholds_both_H, num_bins=100)

            # with t.no_grad():
            # cc1_pre_biases_BH = einsum(batch_BX, self.crosscoders.crosscoder1.W_enc_TMLDH, "b t m l d, t m l d h -> b h")
            # cc1_pre_biases_hist = wandb.Histogram(sequence=cc1_pre_biases_BH.flatten().detach().cpu().numpy().tolist(), num_bins=100)

            # cc2_pre_biases_BH = einsum(batch_BX, self.crosscoders.crosscoder2.W_enc_TMLDH, "b t m l d, t m l d h -> b h")
            # cc2_pre_biases_hist = wandb.Histogram(sequence=cc2_pre_biases_BH.flatten().detach().cpu().numpy().tolist(), num_bins=100)

            # activations_hist = wandb.Histogram(sequence=hidden_B3H.flatten().detach().cpu().numpy().tolist(), num_bins=100)

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
                **get_explained_var_dict(explained_variance_TML, self.layers_to_harvest),
                #
                "media/jumprelu_threshold_distribution_single": thresholds_single_hist,
                "media/jumprelu_threshold_distribution_both": thresholds_both_hist,
                # "media/pre_bias_distribution": pre_biases_hist,
            }

            if self.wandb_run is not None:
                log_dict: dict[str, Any] = {
                    **log_dict,
                    "train/epoch": self.epoch,
                    "train/unique_tokens_trained": self.unique_tokens_trained,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                }

                # if self.crosscoder.n_models == 2 and self.crosscoder.n_tokens == 1:
                #     hist_data = create_cosine_sim_and_relative_norm_histogram_data(
                #         self.crosscoder.W_dec_HTMLD.detach().cpu(), self.layers_to_harvest
                #     )
                #     log_dict.update(
                #         {
                #             f"media/{name}": wandb.Histogram(sequence=data, num_bins=100)
                #             for name, data in hist_data.items()
                #         }
                #     )

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
                self.crosscoders.crosscoder1.hidden_activation.log_threshold_H,
                self.crosscoders.crosscoder2.hidden_activation.log_threshold_H,
                self.crosscoders.crosscoder1.hidden_activation.log_threshold_H,
            ],
            dim=-1,
        )
        loss_3H = t.relu(t_3H.exp() - hidden_B3H) * decoder_norms_3H
        return loss_3H.sum(-1).mean()
