from collections.abc import Iterator
from dataclasses import dataclass

import torch
import wandb
from einops import einsum, rearrange, reduce
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_l1_crosscoder.config import TrainConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_ML
from model_diffing.utils import l0_norm, reconstruction_loss, save_model_and_config, sparsity_loss_l1_of_norms


@dataclass
class LossInfo:
    l1_coef: float
    reconstruction_loss: float
    sparsity_loss: float
    mean_l0: float


class L1CrosscoderTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        llms: list[HookedTransformer],
        optimizer: torch.optim.Optimizer,
        dataloader_BMLD: Iterator[torch.Tensor],
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
    ):
        self.cfg = cfg
        self.llms = llms

        # assert all(llm.tokenizer == llms[0].tokenizer for llm in llms), (
        #     "All models must have the same tokenizer"
        # )
        tokenizer = self.llms[0].tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        self.tokenizer = tokenizer
        self.crosscoder = crosscoder
        self.optimizer = optimizer
        self.wandb_run = wandb_run
        self.device = device

        self.step = 0
        self.dataloader_BMLD = dataloader_BMLD

    @property
    def d_model(self) -> int:
        return self.llms[0].cfg.d_model

    def train(self):
        logger.info("Estimating norm scaling factors (model, layer)")
        norm_scaling_factors_ML = self._estimate_norm_scaling_factor_ML()
        logger.info(f"Norm scaling factors (model, layer): {norm_scaling_factors_ML}")

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        while self.step < self.cfg.num_steps:
            batch_BMLD = self._next_batch_BMLD(norm_scaling_factors_ML)

            log_dict = self._train_step(batch_BMLD)

            if self.wandb_run and (self.step + 1) % self.cfg.log_every_n_steps == 0:
                self.wandb_run.log(log_dict, step=self.step)

            if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    epoch=self.step,
                )

            self.step += 1

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        loss, loss_info = self._get_loss(batch_BMLD)

        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()

        self.optimizer.param_groups[0]["lr"] = self._lr_scheduler()

        log_dict = {
            "train/l1_coef": loss_info.l1_coef,
            "train/mean_l0": loss_info.mean_l0,
            "train/mean_l0_pct": loss_info.mean_l0 / self.crosscoder.hidden_dim,
            "train/reconstruction_loss": loss_info.reconstruction_loss,
            "train/sparsity_loss": loss_info.sparsity_loss,
            "train/loss": loss.item(),
        }

        return log_dict

    def _get_loss(self, activations_BMLD: torch.Tensor) -> tuple[torch.Tensor, "LossInfo"]:
        train_res = self.crosscoder.forward_train(activations_BMLD)

        reconstruction_loss_ = reconstruction_loss(activations_BMLD, train_res.reconstructed_acts_BMLD)
        sparsity_loss_ = sparsity_loss_l1_of_norms(self.crosscoder.W_dec_HMLD, train_res.hidden_BH)
        l0_norms_B = reduce(train_res.hidden_BH, "batch hidden -> batch", l0_norm)
        l1_coef = self._l1_coef_scheduler()
        loss = reconstruction_loss_ + l1_coef * sparsity_loss_

        loss_info = LossInfo(
            l1_coef=l1_coef,
            reconstruction_loss=reconstruction_loss_.item(),
            sparsity_loss=sparsity_loss_.item(),
            mean_l0=l0_norms_B.mean().item(),
        )

        return loss, loss_info

    def _calculate_explained_variance(
        self,
        activations_BMLD: torch.Tensor,
        reconstructed_acts_BMLD: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, L, d_model = activations_BMLD.shape

        activations_Bmld = activations_BMLD.flatten()
        recon_acts_Bmld = reconstructed_acts_BMLD.flatten()
        explained_var_total = 1 - ((activations_Bmld - recon_acts_Bmld).var() / (activations_Bmld.var() + eps))

        activations_Mbld = rearrange(activations_BMLD, "batch model layer d_model -> model (batch layer d_model)")
        recon_acts_Mbld = rearrange(reconstructed_acts_BMLD, "batch model layer d_model -> model (batch layer d_model)")
        error_var_M = reduce(activations_Mbld - recon_acts_Mbld, "model bld -> model", torch.var)
        activations_var_M = reduce(activations_Mbld, "model bld -> model", torch.var)
        explained_var_per_model_M = 1 - (error_var_M / (activations_var_M + eps))
        assert explained_var_per_model_M.shape == (M,)

        activations_Lmbd = rearrange(activations_BMLD, "batch model layer d_model -> layer (batch model d_model)")
        recon_acts_Lmbd = rearrange(reconstructed_acts_BMLD, "batch model layer d_model -> layer (batch model d_model)")
        error_var_L = reduce(activations_Lmbd - recon_acts_Lmbd, "layer bmd -> layer", torch.var)
        activations_var_L = reduce(activations_Lmbd, "layer bmd -> layer", torch.var)
        explained_var_per_layer_L = 1 - (error_var_L / (activations_var_L + eps))
        assert explained_var_per_layer_L.shape == (L,)

        return (
            explained_var_total,
            explained_var_per_model_M,
            explained_var_per_layer_L,
        )

    def _estimate_norm_scaling_factor_ML(self) -> torch.Tensor:
        return estimate_norm_scaling_factor_ML(
            self.dataloader_BMLD,
            self.device,
            self.d_model,
            self.cfg.n_batches_for_norm_estimate,
        )

    def _next_batch_BMLD(self, norm_scaling_factors_ML: torch.Tensor) -> torch.Tensor:
        batch_BMLD = next(self.dataloader_BMLD)
        batch_BMLD = batch_BMLD.to(self.device)
        batch_BMLD = einsum(
            batch_BMLD, norm_scaling_factors_ML, "batch model layer d_model, model layer -> batch model layer d_model"
        )
        return batch_BMLD

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.l1_coef_n_steps:
            return self.cfg.l1_coef_max * self.step / self.cfg.l1_coef_n_steps
        else:
            return self.cfg.l1_coef_max

    def _lr_scheduler(self) -> float:
        pct_until_finished = 1 - (self.step / self.cfg.num_steps)
        if pct_until_finished < self.cfg.learning_rate.last_pct_of_steps:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / self.cfg.learning_rate.last_pct_of_steps
            return self.cfg.learning_rate.initial_learning_rate * scale
        else:
            return self.cfg.learning_rate.initial_learning_rate
