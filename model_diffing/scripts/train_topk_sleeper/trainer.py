from collections.abc import Callable, Iterator

from itertools import islice
import numpy as np
import torch
import wandb
from einops import einsum
from torch.nn.utils import clip_grad_norm_
from typing import Any
from wandb.sdk.wandb_run import Run

from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_topk_sleeper.config import TrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer, estimate_norm_scaling_factor_ML
from model_diffing.utils import (
    calculate_explained_variance_ML,
    calculate_reconstruction_loss,
    save_model_and_config,
)


class TopKTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataloader_builder: Callable[[], Iterator[torch.Tensor]],
        validation_dataloader_builder: Callable[[], Iterator[torch.Tensor]],
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
        layers_to_harvest: list[int],
        steps_per_epoch: int,
    ):
        self.cfg = cfg
        self.crosscoder = crosscoder
        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())
        self.dataloader_builder = dataloader_builder
        self.validation_dataloader_builder = validation_dataloader_builder
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.step = 0
        self.total_steps = steps_per_epoch * self.cfg.num_epochs
        logger.info(f"Total steps: {self.total_steps}")
        self.tokens_trained = 0

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.total_steps)

    def _run_validation(self, norm_scaling_factors_ML: torch.Tensor):
        test_logs = []
        for batch_BMLD in islice(self.validation_dataloader_builder(), self.cfg.num_test_batches):
            batch_BMLD = batch_BMLD.to(self.device)
            batch_BMLD = einsum(
                batch_BMLD, norm_scaling_factors_ML,
                "batch model layer d_model, model layer -> batch model layer d_model"
            )
            test_logs.append(self._test_step(batch_BMLD))

        test_log = {k: np.mean([log[k] for log in test_logs]) for k in test_logs[0]}
        if self.wandb_run:
            self.wandb_run.log(test_log, step=self.step)
        logger.info(test_log)

    def _run_epoch(self, epoch: int, norm_scaling_factors_ML: torch.Tensor):
        for batch_BMLD in self.dataloader_builder():
            batch_BMLD = batch_BMLD.to(self.device)
            batch_BMLD = einsum(
                batch_BMLD, norm_scaling_factors_ML,
                "batch model layer d_model, model layer -> batch model layer d_model"
            )

            train_log = self._train_step(batch_BMLD)

            if self.wandb_run and (self.step + 1) % self.cfg.log_every_n_steps == 0:
                self.wandb_run.log(train_log, step=self.step)

            self.step += 1

    def train(self):
        logger.info("Estimating norm scaling factors (model, layer)")

        norm_scaling_factors_ML = estimate_norm_scaling_factor_ML(
            self.dataloader_builder(),
            self.device,
            self.cfg.n_batches_for_norm_estimate,
        )

        logger.info(f"Norm scaling factors (model, layer): {norm_scaling_factors_ML}")

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        # Run initial validation
        self._run_validation(norm_scaling_factors_ML)

        for epoch in range(self.cfg.num_epochs):
            self._run_epoch(epoch, norm_scaling_factors_ML)
        
            if self.cfg.save_dir and self.cfg.save_every_n_epochs and (epoch + 1) % self.cfg.save_every_n_epochs == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    epoch=epoch,
                )

            self._run_validation(norm_scaling_factors_ML)

    def _get_loss(self, batch_BMLD: torch.Tensor) -> tuple[torch.Tensor, np.ndarray[Any, np.dtype[np.float64]]]:
        train_res = self.crosscoder.forward_train(batch_BMLD)

        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BMLD)

        with torch.no_grad():
            explained_variance_ML = calculate_explained_variance_ML(batch_BMLD, train_res.reconstructed_acts_BMLD)

        return reconstruction_loss, explained_variance_ML.cpu().numpy()

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        self.tokens_trained += batch_BMLD.shape[0]

        reconstruction_loss, explained_variance_ML = self._get_loss(batch_BMLD)
        reconstruction_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        log_dict = {
            "train/reconstruction_loss": reconstruction_loss.item(),
            "train/tokens_trained": self.tokens_trained,
            "train/mean_explained_variance": explained_variance_ML.mean(),
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }

        return log_dict
    
    def _test_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            reconstruction_loss, explained_variance_ML = self._get_loss(batch_BMLD)

        return {
            "test/reconstruction_loss": reconstruction_loss.item(),
            "test/mean_explained_variance": explained_variance_ML.mean(),
        }
