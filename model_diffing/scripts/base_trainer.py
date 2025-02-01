from abc import abstractmethod
from itertools import islice
from pathlib import Path

import torch
import wandb
from wandb.sdk.wandb_run import Run

from model_diffing.analysis.visualization import create_visualizations
from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.utils import CONFIG_FILE_NAME, MODEL_FILE_NAME, SaveableModule, save_model_and_config


class BaseTrainer[TConfig: BaseTrainConfig, TAct: SaveableModule]:
    tep: int
    epoch: int
    unique_tokens_trained: int

    # I've tried to make this invariant as obvious as possible in this type signature:
    # If training without epochs (epochs=1), we need to provide num_steps
    # However, if training with epochs, we don't need to limit the number of steps per epoch,
    # just loop through the dataloader
    # epochs_steps: tuple[Literal[1], int] | tuple[int, None]

    def __init__(
        self,
        cfg: TConfig,
        activations_dataloader: BaseActivationsDataloader,
        crosscoder: AcausalCrosscoder[TAct],
        wandb_run: Run | None,
        device: torch.device,
        layers_to_harvest: list[int],
        experiment_name: str,
    ):
        self.cfg = cfg
        self.crosscoder = crosscoder
        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())

        self.epochs = cfg.epochs

        self.num_steps_per_epoch = validate_num_steps_per_epoch(
            cfg.epochs, cfg.num_steps_per_epoch, cfg.num_steps, activations_dataloader
        )

        self.total_steps = self.num_steps_per_epoch * (cfg.epochs or 1)
        logger.info(
            f"Total steps: {self.total_steps} (num_steps_per_epoch: {self.num_steps_per_epoch}, epochs: {cfg.epochs})"
        )

        self.lr_scheduler = build_lr_scheduler(cfg.optimizer, self.num_steps_per_epoch)

        self.activations_dataloader = activations_dataloader
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.base_save_dir = Path(cfg.base_save_dir) / experiment_name

        self.local_save_dir = self.base_save_dir / "local_checkpoints"
        self.local_save_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_checkpoint_dir = self.base_save_dir / "wandb_checkpoints"
        self.wandb_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0
        self.unique_tokens_trained = 0

    def train(self):
        if self.wandb_run:
            self.wandb_run.save(
                f"{self.wandb_checkpoint_dir}/*",
                base_path=self.base_save_dir,
                policy="end",
            )

        for _ in range(self.epochs or 1):
            epoch_dataloader = islice(
                self.activations_dataloader.get_shuffled_activations_iterator_BMLD(),
                self.num_steps_per_epoch,
            )

            for example_BMLD in epoch_dataloader:
                batch_BMLD = example_BMLD.to(self.device)

                log_dict = {
                    **self._train_step(batch_BMLD),
                    "train/step": self.step,
                    "train/epoch": self.epoch,
                    "train/unique_tokens_trained": self.unique_tokens_trained,
                }

                if self.wandb_run:
                    if self.cfg.log_every_n_steps is not None and self.step % self.cfg.log_every_n_steps == 0:
                        self.wandb_run.log(log_dict, step=self.step)

                    if (
                        self.cfg.log_visualizations_every_n_steps is not None
                        and self.step % self.cfg.log_visualizations_every_n_steps == 0
                    ):
                        visualizations = create_visualizations(
                            self.crosscoder.W_dec_HMLD.detach().cpu(), self.layers_to_harvest
                        )
                        if visualizations is not None:
                            self.wandb_run.log(
                                {f"visualizations/{k}": wandb.Plotly(v) for k, v in visualizations.items()},
                                step=self.step,
                            )

                    # if (
                    #     self.cfg.upload_checkpoint_to_wandb_every_n_steps is not None
                    #     and self.step % self.cfg.upload_checkpoint_to_wandb_every_n_steps == 0
                    # ):
                    #     with self.crosscoder.temporarily_fold_activation_scaling(norm_scaling_factors_ML):
                    #         cfg_dict = self.crosscoder.dump_cfg()
                    #         state_dict = self.crosscoder.state_dict()

                    #         cfg_path = self.wandb_checkpoint_dir / CONFIG_FILE_NAME
                    #         model_path = self.wandb_checkpoint_dir / MODEL_FILE_NAME

                    #         with open(cfg_path, "w") as f:
                    #             yaml.dump(cfg_dict, f)

                    #         torch.save(state_dict, model_path)

                if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                    with self.crosscoder.temporarily_fold_activation_scaling(
                        self.activations_dataloader.get_norm_scaling_factors_ML()
                    ):
                        save_model_and_config(
                            config=self.cfg,
                            save_dir=self.local_save_dir,
                            model=self.crosscoder,
                            step=self.step,
                        )

                if self.epoch == 0:
                    self.unique_tokens_trained += batch_BMLD.shape[0]

                self.step += 1
            self.epoch += 1

    @abstractmethod
    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]: ...


def validate_num_steps_per_epoch(
    epochs: int | None,
    num_steps_per_epoch: int | None,
    num_steps: int | None,
    activations_dataloader: BaseActivationsDataloader,
) -> int:
    if epochs is not None:
        if num_steps is not None:
            raise ValueError("num_steps must not be provided if using epochs")

        dataloader_num_batches = activations_dataloader.num_batches()
        if dataloader_num_batches is None:
            raise ValueError(
                "activations_dataloader must have a length if using epochs, "
                "as we need to know how to schedule the learning rate"
            )

        if num_steps_per_epoch is None:
            return dataloader_num_batches
        else:
            if dataloader_num_batches < num_steps_per_epoch:
                logger.warning(
                    f"num_steps_per_epoch ({num_steps_per_epoch}) is greater than the number "
                    f"of batches in the dataloader ({dataloader_num_batches}), so we will only "
                    "train for the number of batches in the dataloader"
                )
                return dataloader_num_batches
            else:
                return num_steps_per_epoch

    # not using epochs
    if num_steps is None:
        raise ValueError("num_steps must be provided if not using epochs")
    if num_steps_per_epoch is not None:
        raise ValueError("num_steps_per_epoch must not be provided if not using epochs")
    return num_steps
