from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from model_diffing.utils import BaseModel


class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int


class TopKExperimentConfig(BaseExperimentConfig):
    crosscoder: TopKCrosscoderConfig
    train: BaseTrainConfig
