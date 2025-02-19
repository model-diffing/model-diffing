from typing import Any

from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig, DataConfig
from model_diffing.utils import BaseModel


class JumpReLUConfig(BaseModel):
    bandwidth: float = 2.0  # aka ε
    threshold_init: float = 0.1  # aka t
    backprop_through_jumprelu_input: bool = False


class JanUpdateCrosscoderConfig(BaseModel):
    hidden_dim: int
    jumprelu: JumpReLUConfig = JumpReLUConfig()
    initial_approx_firing_pct: float
    """
    The initial approximate firing percentage of the jumprelu, as a float between 0 and 1. This is used to calibrate
    b_enc. In the update, this value is 10_000 / hidden_size. But we're often training with hidden_size << 10_000, so
    we allow setting this value directly.
    
    Sensible values might be something like 0.2 - 0.8, but I'm (oli) very uncertain about this!

    see: https://transformer-circuits.pub/2025/january-update/index.html#:~:text=.-,We%20initialize,-%F0%9D%91%8F
    """

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.initial_approx_firing_pct > 1 or self.initial_approx_firing_pct < 0:
            raise ValueError(f"initial_approx_firing_pct must be between 0 and 1, got {self.initial_approx_firing_pct}")


class JanUpdateTrainConfig(BaseTrainConfig):
    c: float = 4.0

    final_lambda_s: float = 20.0
    """will be linearly ramped from 0 over the entire training run"""

    lambda_p: float = 3e-6


class JanUpdateExperimentConfig(BaseExperimentConfig):
    data: DataConfig
    crosscoder: JanUpdateCrosscoderConfig
    train: JanUpdateTrainConfig
    hookpoints: list[str]
