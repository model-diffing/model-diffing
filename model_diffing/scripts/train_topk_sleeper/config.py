from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_serializer

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
    norm_scaling_factors: list[list[float]] | None = None

    @field_serializer("save_dir")
    def serialize_save_dir(self, save_dir: Path, _info):
        # Return yaml serialization of str(save_dir), correctly escaping special characters
        return yaml.dump(str(save_dir)).split('\n')[0] # TODO hacky

class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int
    ft_init_checkpt_folder: Path | None = None
    ft_init_checkpt_epoch: int | None = None

    @field_serializer("ft_init_checkpt_folder")
    def serialize_ft_init_checkpt_folder(self, ft_init_checkpt_folder: Path, _info):
        return yaml.dump(str(ft_init_checkpt_folder)).split('\n')[0] # TODO hacky


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