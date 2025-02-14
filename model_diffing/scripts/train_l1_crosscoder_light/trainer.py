import copy
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import wandb
from einops import einsum, reduce
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_l1_crosscoder_light.config import TrainConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_ML
from model_diffing.utils import l0_norm, calculate_reconstruction_loss, save_model_and_config, sparsity_loss_l1_of_norms
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots







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
        optimizer: torch.optim.Optimizer,
        dataloader_BMLD: Iterator[torch.Tensor],
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
    ):
        self.cfg = cfg
        self.crosscoder = crosscoder
        self.optimizer = optimizer
        self.wandb_run = wandb_run
        self.device = device
        self.dataloader_BMLD = dataloader_BMLD

        self.step = 0

    def train(self):
        rec_loss=[]
        sparsity_loss=[]
        logger.info("Estimating norm scaling factors (model, layer)")
        norm_scaling_factors_ML = self._estimate_norm_scaling_factor_ML()
        logger.info(f"Norm scaling factors (model, layer): {norm_scaling_factors_ML}")

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )
        pbar = tqdm(total=self.cfg.num_steps, desc="Training")
        while self.step < self.cfg.num_steps:
            #batch_BMLD = self._next_batch_BMLD(norm_scaling_factors_ML)
            batch_BMLD=next(self.dataloader_BMLD)
            log_dict = self._train_step(batch_BMLD)
            rec_loss.append(log_dict['train/reconstruction_loss'])
            sparsity_loss.append(log_dict['train/sparsity_loss'])

            if self.wandb_run and (self.step + 1) % self.cfg.log_every_n_steps == 0:
                self.wandb_run.log(log_dict)

            if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    step=self.step,
                )

            self.step += 1
            pbar.update(1)
            pbar.set_description(f"Rec Loss: {log_dict['train/reconstruction_loss']:.4f}, Sparsity: {log_dict['train/sparsity_loss']:.4f}")
            
        
        pbar.close()
    
            

        return rec_loss, sparsity_loss

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        loss, loss_info = self._get_loss(batch_BMLD)

        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()

        self.optimizer.param_groups[0]["lr"] = self._lr_scheduler()

        log_dict = {
            "train/step": self.step,
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
        reconstruction_loss_ = calculate_reconstruction_loss(activations_BMLD, train_res.reconstructed_acts_BMLD)
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

    def _estimate_norm_scaling_factor_ML(self) -> torch.Tensor:
        return estimate_norm_scaling_factor_ML(
            self.dataloader_BMLD,
            self.device,
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
