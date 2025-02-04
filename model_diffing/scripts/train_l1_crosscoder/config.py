from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from model_diffing.utils import BaseModel


class L1CrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1


class L1TrainConfig(BaseTrainConfig):
    l1_coef_max: float = 5.0
    l1_coef_n_steps: int = 1000


class L1ExperimentConfig(BaseExperimentConfig):
    crosscoder: L1CrosscoderConfig
    train: L1TrainConfig
