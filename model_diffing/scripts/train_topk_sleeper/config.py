from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from model_diffing.scripts.config_common import (
    AdamDecayTo0LearningRateConfig,
    WandbConfig,
    DataConfig,
)

class TrainConfig(BaseModel):
    optimizer: AdamDecayTo0LearningRateConfig
    num_epochs: int
    save_dir: Path | None
    save_every_n_epochs: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100
    num_test_batches: int = 10


class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int
    ft_init_checkpt: Path | None = None


class LoraLLMConfig(BaseModel):
    base_model_repo: str
    lora_model_repo: str


class TopKExperimentConfig(BaseModel):
    seed: int
    cache_dir: str = ".cache"
    data: DataConfig
    llm: LoraLLMConfig
    wandb: WandbConfig | Literal["disabled"] = "disabled"
    crosscoder: TopKCrosscoderConfig
    train: TrainConfig
