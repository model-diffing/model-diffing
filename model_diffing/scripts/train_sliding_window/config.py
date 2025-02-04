from model_diffing.scripts.config_common import BaseExperimentConfig
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateCrosscoderConfig, JanUpdateTrainConfig


class SlidingWindowExperimentConfig(BaseExperimentConfig):
    token_window_size: int
    crosscoder: JanUpdateCrosscoderConfig
    train: JanUpdateTrainConfig
