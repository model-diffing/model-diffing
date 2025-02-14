
from pathlib import Path

from itertools import islice
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Any
from wandb.sdk.wandb_run import Run
from model_diffing.models.crosscoder import TopkActivation
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import validate_num_steps_per_epoch
from model_diffing.scripts.train_topk_sleeper.config import TrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.utils import (
    calculate_explained_variance_ML,
    calculate_reconstruction_loss,
)
from model_diffing.dataloader.activations import BaseActivationsDataloader
from torch import nn


def save_model(save_dir: Path, model: nn.Module, epoch: int) -> None:
    """Save the model to disk.

    Args:
        save_dir: The directory to save the model and config to.
        model: The model to save.
        epoch: The current epoch (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    model_file = save_dir / f"model_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), model_file)
    logger.info("Saved model to %s", model_file)


class TopKTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        cfg_raw: str,
        activations_dataloader: BaseActivationsDataloader,
        activations_validation_dataloader: BaseActivationsDataloader,
        crosscoder: AcausalCrosscoder[TopkActivation],
        wandb_run: Run | None,
        device: torch.device,
        layers_to_harvest: list[int],
        experiment_name: str,
        
    ):
        self.cfg = cfg
        self.cfg_raw = cfg_raw
        self.crosscoder = crosscoder
        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())

        self.epochs = cfg.epochs

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, None, None, activations_dataloader
        )

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.num_steps_per_epoch*self.epochs)

        self.activations_dataloader = activations_dataloader
        self.activations_validation_dataloader = activations_validation_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.base_save_dir = Path(cfg.base_save_dir) / experiment_name

        self.local_save_dir = self.base_save_dir / "local_checkpoints"
        self.local_save_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_checkpoint_dir = self.base_save_dir / "wandb_checkpoints"
        self.wandb_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Write the contents of cfg_raw to "config.yaml" in wandb_checkpoint_dir
        with open(self.wandb_checkpoint_dir / "config.yaml", "w") as f:
            f.write(cfg_raw)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

        #self.norm_scaling_factors_ML = norm_scaling_factors_ML

    def _run_validation(self):
        epoch_dataloader = self.activations_validation_dataloader.get_shuffled_activations_iterator_BMLD()
        test_logs = []
        for batch_BMLD in islice(epoch_dataloader, self.cfg.num_test_batches):
            batch_BMLD = batch_BMLD.to(self.device)
            '''batch_BMLD = einsum(
                batch_BMLD, self.norm_scaling_factors_ML,
                "batch model layer d_model, model layer -> batch model layer d_model"
            )'''
            test_logs.append(self._test_step(batch_BMLD))

        test_log = {k: np.mean([log[k] for log in test_logs]) for k in test_logs[0]}
        if self.wandb_run:
            self.wandb_run.log(test_log, step=self.step)

    def _run_epoch(self):
        epoch_dataloader = self.activations_dataloader.get_shuffled_activations_iterator_BMLD()

        for example_BMLD in epoch_dataloader:
            batch_BMLD = example_BMLD.to(self.device)
            '''batch_BMLD = einsum(
                batch_BMLD, self.norm_scaling_factors_ML,
                "batch model layer d_model, model layer -> batch model layer d_model"
            )'''

            log_dict = {
                **self._train_step(batch_BMLD),
                "train/step": self.step,
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
            }

            if self.wandb_run:
                if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
                    self.wandb_run.log(log_dict, step=self.step)
            
            if self.epoch == 0:
                self.unique_tokens_trained += batch_BMLD.shape[0]

            self.step += 1


    
    def _save_model(self, epoch: int):
        if self.wandb_run:
            with self.crosscoder.temporarily_fold_activation_scaling(
                self.activations_dataloader.get_norm_scaling_factors_ML()
            ):
                save_model(
                    save_dir=self.wandb_checkpoint_dir,
                    model=self.crosscoder,
                    epoch=epoch,
                )

            self.wandb_run.save(
                f"{self.wandb_checkpoint_dir}/*",
                base_path=self.base_save_dir,
                policy="end",
            )
        

    def train(self):
        if self.wandb_run:
            self.wandb_run.save(
                f"{self.wandb_checkpoint_dir}/*",
                base_path=self.base_save_dir,
                policy="end",
            )

        # Run initial validation
        self._run_validation()

        for _ in range(self.epochs):
            self._run_epoch()
        
            if self.cfg.upload_checkpoint_to_wandb_every_n_epochs is not None:
                if self.epoch % self.cfg.upload_checkpoint_to_wandb_every_n_epochs == 0 or self.epoch == self.epochs-1:
                    self._save_model(self.epoch)

            self._run_validation()

            self.epoch += 1


    def _get_loss(self, batch_BMLD: torch.Tensor) -> tuple[torch.Tensor, np.ndarray[Any, np.dtype[np.float64]]]:
        train_res = self.crosscoder.forward_train(batch_BMLD)

        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BMLD)

        with torch.no_grad():
            explained_variance_ML = calculate_explained_variance_ML(batch_BMLD, train_res.reconstructed_acts_BMLD)

        return reconstruction_loss, explained_variance_ML.cpu().numpy()

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        reconstruction_loss, explained_variance_ML = self._get_loss(batch_BMLD)
        reconstruction_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        log_dict = {
            "train/reconstruction_loss": reconstruction_loss.item(),
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
