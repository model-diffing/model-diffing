from pydantic import BaseModel

from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig


class L1CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class L1TrainConfig(BaseTrainConfig):
    l1_coef_max: float = 5.0
    l1_coef_n_steps: int = 1000


class L1ExperimentConfig(BaseExperimentConfig):
    crosscoder: L1CrosscoderConfig
    train: L1TrainConfig


# class BaseExperimentConfig(BaseModel):
#     seed: int = 42
#     cache_dir: str = ".cache"
#     data: DataConfig
#     llms: LLMsConfig
#     wandb: WandbConfig | Literal["disabled"] = WandbConfig()