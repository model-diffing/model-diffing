"""Train a model on MNIST.

This script takes ~40 seconds to run for 3 layers and 15 epochs on a CPU.

Usage:
    python model_diffing/scripts/train_mnist/run_train_mnist.py <path/to/config.yaml>
"""

from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path
from typing import cast

import fire
import numpy as np
import torch
import wandb
import yaml
from einops import reduce
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_crosscoder.data import (
    ActivationHarvester,
    ShuffledTokensActivationsLoader,
)
from model_diffing.utils import l2_norm, save_model_and_config

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TrainConfig(BaseModel):
    lr: float = 5e-5
    lambda_max: float = 5.0
    lambda_n_steps: int = 1000
    batch_size: int
    num_steps: int
    save_dir: Path | None
    save_every_n_steps: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100


class CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class DatasetConfig(BaseModel):
    hf_dataset: str
    cache_dir: str
    sequence_length: int
    harvest_batch_size: int
    shuffle_buffer_size: int


class WandbConfig(BaseModel):
    project: str
    entity: str


class ModelConfig(BaseModel):
    name: str
    revision: str | None


class Config(BaseModel):
    seed: int
    models: list[ModelConfig]
    layer_indices_to_harvest: list[int]
    train: TrainConfig
    crosscoder: CrosscoderConfig
    dataset: DatasetConfig
    wandb: WandbConfig | None
    dtype: str = "float32"


def create_l1_coef_scheduler(cfg: Config) -> Callable[[int], float]:
    def l1_coef_scheduler(step: int) -> float:
        if step < cfg.train.lambda_n_steps:
            return cfg.train.lambda_max * step / cfg.train.lambda_n_steps
        else:
            return cfg.train.lambda_max

    return l1_coef_scheduler


class Trainer:
    def __init__(
        self,
        llms: list[HookedTransformer],
        optimizer: torch.optim.Optimizer,
        dataloader_BMLD: Iterator[torch.Tensor],
        crosscoder: AcausalCrosscoder,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        l1_coef_scheduler: Callable[[int], float],
        wandb_run: Run | None,
        # hacky - remove:
        expected_batch_shape: tuple[int, int, int, int],
        cfg: TrainConfig,
    ):
        self.llms = llms
        self.d_model = self.llms[0].cfg.d_model

        # assert all(llm.tokenizer == llms[0].tokenizer for llm in llms), (
        #     "All models must have the same tokenizer"
        # )
        tokenizer = self.llms[0].tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        self.tokenizer = tokenizer
        self.crosscoder = crosscoder
        self.optimizer = optimizer
        self.dataloader_BMLD = dataloader_BMLD
        self.norm_scaling_factor = self._estimate_norm_scaling_factor()
        self.lr_scheduler = lr_scheduler
        self.l1_coef_scheduler = l1_coef_scheduler
        self.wandb_run = wandb_run

        self.step = 0

        self.expected_batch_shape = expected_batch_shape

        self.cfg = cfg

    def train(self):
        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )

        while self.step < self.cfg.num_steps:
            batch_BMLD = next(self.dataloader_BMLD)
            batch_BMLD = batch_BMLD.to(DEVICE)
            batch_BMLD = batch_BMLD * self.norm_scaling_factor
            self.train_step(batch_BMLD)
            self.step += 1

    def train_step(self, batch_BMLD: torch.Tensor):
        self.optimizer.zero_grad()

        _, losses = self.crosscoder.forward_train(batch_BMLD)

        lambda_ = self.l1_coef_scheduler(self.step)
        loss = losses.reconstruction_loss + lambda_ * losses.sparsity_loss
        loss.backward()

        if (self.step + 1) % self.cfg.log_every_n_steps == 0:
            log_dict = {
                "train/step": self.step,
                "train/lambda": lambda_,
                "train/l0": losses.l0.item(),
                "train/reconstruction_loss": losses.reconstruction_loss.item(),
                "train/sparsity_loss": losses.sparsity_loss.item(),
                "train/loss": loss.item(),
            }
            print(log_dict)
            if self.wandb_run:
                self.wandb_run.log(log_dict)

        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()

        if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
            save_model_and_config(
                config=self.cfg,
                save_dir=self.cfg.save_dir,
                model=self.crosscoder,
                epoch=self.step,
            )

    @torch.no_grad()
    def _estimate_norm_scaling_factor(self) -> torch.Tensor:
        mean_norm = self._estimate_mean_norm()
        scaling_factor = np.sqrt(self.d_model) / mean_norm
        return scaling_factor

    # adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
    def _estimate_mean_norm(self) -> float:
        norms_per_batch = []
        for batch_BMLD in tqdm(
            islice(self.dataloader_BMLD, self.cfg.n_batches_for_norm_estimate),
            desc="Estimating norm scaling factor",
        ):
            norms_BML = reduce(batch_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
            norms_mean = norms_BML.mean().item()
            print(f"- Norms mean: {norms_mean}")
            norms_per_batch.append(norms_mean)
        mean_norm = float(np.mean(norms_per_batch))
        return mean_norm


def get_shuffled_activations_iterator_BMLD(
    cfg: Config,
    llms: list[HookedTransformer],
    tokenizer: PreTrainedTokenizerBase,
) -> Iterator[torch.Tensor]:
    activation_harvester = ActivationHarvester(
        hf_dataset=cfg.dataset.hf_dataset,
        cache_dir=cfg.dataset.cache_dir,
        models=llms,
        tokenizer=tokenizer,
        sequence_length=cfg.dataset.sequence_length,
        batch_size=cfg.dataset.harvest_batch_size,
        layer_indices_to_harvest=cfg.layer_indices_to_harvest,
    )

    dataloader = ShuffledTokensActivationsLoader(
        activation_harvester=activation_harvester,
        shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
        batch_size=cfg.train.batch_size,
    )

    return dataloader.get_shuffled_activations_iterator_BMLD()


def build_trainer(cfg: Config) -> Trainer:
    llms = [
        cast(
            HookedTransformer,
            HookedTransformer.from_pretrained(
                model.name,
                revision=model.revision,
                cache_dir=cfg.dataset.cache_dir,
                dtype=str(cfg.dtype),
            ).to(DEVICE),
        )
        for model in cfg.models
    ]
    tokenizer = llms[0].tokenizer
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    tokenizer = tokenizer

    dataloader_BMLD = get_shuffled_activations_iterator_BMLD(cfg, llms, tokenizer)

    crosscoder = AcausalCrosscoder(
        n_layers=len(cfg.layer_indices_to_harvest),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=len(llms),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(crosscoder.parameters(), lr=cfg.train.lr)

    l1_coef_scheduler = create_l1_coef_scheduler(cfg)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, l1_coef_scheduler)

    wandb_run = (
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=cfg.model_dump(),
        )
        if cfg.wandb
        else None
    )

    expected_batch_shape = (
        cfg.train.batch_size,
        len(llms),
        len(cfg.layer_indices_to_harvest),
        llms[0].cfg.d_model,
    )

    return Trainer(
        llms=llms,
        optimizer=optimizer,
        dataloader_BMLD=dataloader_BMLD,
        crosscoder=crosscoder,
        lr_scheduler=lr_scheduler,
        l1_coef_scheduler=l1_coef_scheduler,
        wandb_run=wandb_run,
        expected_batch_shape=expected_batch_shape,
        cfg=cfg.train,
    )


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


def main(config_path: str) -> None:
    print("Loading config...")
    config = load_config(Path(config_path))
    print("Loaded config")
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    print("Starting...")
    fire.Fire(main)
