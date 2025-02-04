from abc import abstractmethod
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import wandb
import yaml
from wandb.sdk.wandb_run import Run

from model_diffing.analysis.visualization import create_cosine_sim_and_relative_norm_histogram_data
from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from model_diffing.scripts.utils import build_lr_scheduler, build_optimizer
from model_diffing.utils import SaveableModule


def save_config(config: BaseTrainConfig, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    logger.info("Saved config to %s", save_dir / "config.yaml")


def save_model(model: SaveableModule, save_dir: Path, epoch: int, step: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / f"epoch_{epoch}_step_{step}")


# using python3.11 generics because it's better supported by GPU providers
TConfig = TypeVar("TConfig", bound=BaseTrainConfig)
TAct = TypeVar("TAct", bound=SaveableModule)


class BaseTrainer(Generic[TConfig, TAct]):
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
        self.activations_dataloader = activations_dataloader
        self.crosscoder = crosscoder
        self.wandb_run = wandb_run
        self.device = device
        self.layers_to_harvest = layers_to_harvest

        self.optimizer = build_optimizer(cfg.optimizer, crosscoder.parameters())

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
            epoch_dataloader = self.activations_dataloader.get_shuffled_activations_iterator_BMLD()
            epoch_dataloader = islice(epoch_dataloader, self.num_steps_per_epoch)

            for example_BMLD in epoch_dataloader:
                batch_BMLD = example_BMLD.to(self.device)

                train_result_dict = self._train_step(batch_BMLD)

                if (
                    self.wandb_run is not None
                    and self.cfg.log_every_n_steps is not None
                    and self.step % self.cfg.log_every_n_steps == 0
                ):
                    log_dict: dict[str, Any] = {
                        **train_result_dict,
                        "train/epoch": self.epoch,
                        "train/unique_tokens_trained": self.unique_tokens_trained,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }

                    if self.crosscoder.n_models == 2:
                        hist_data = create_cosine_sim_and_relative_norm_histogram_data(
                            self.crosscoder.W_dec_HMLD.detach().cpu(), self.layers_to_harvest
                        )
                        log_dict.update(
                            {
                                f"media/{name}": wandb.Histogram(sequence=data, num_bins=100)
                                for name, data in hist_data.items()
                            }
                        )

                    self.wandb_run.log(log_dict, step=self.step)

                # TODO(oli): get wandb checkpoint saving working

                if self.cfg.save_every_n_steps is not None and self.step % self.cfg.save_every_n_steps == 0:
                    with self.crosscoder.temporarily_fold_activation_scaling(
                        self.activations_dataloader.get_norm_scaling_factors_ML()
                    ):
                        save_model(self.crosscoder, self.save_dir, self.epoch, self.step)

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


TCfg = TypeVar("TCfg", bound=BaseExperimentConfig)
TTrainer = TypeVar("TTrainer", bound=BaseTrainer[Any, Any])


def run_exp(build_trainer: Callable[[TCfg], TTrainer], cfg_cls: type[TCfg]) -> Callable[[Path], None]:
    def inner(config_path: Path) -> None:
        assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
        assert Path(config_path).exists(), f"Config file {config_path} does not exist."
        logger.info("Loading config...")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = cfg_cls(**config_dict)
        logger.info("Loaded config")
        logger.info("Building trainer")
        trainer = build_trainer(config)
        logger.info("Training")
        trainer.train()

    return inner
