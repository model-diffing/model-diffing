from pathlib import Path

from pydantic import BaseModel

from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig


class JumpReLUConfig(BaseModel):
    bandwidth: float = 2.0  # aka ε
    threshold_init: float = 0.0  # aka t
    backprop_through_jumprelu_threshold: bool = True
    


class JanUpdateCrosscoderConfig(BaseModel):
    hidden_dim: int
    jumprelu: JumpReLUConfig = JumpReLUConfig()
    dec_init_norm: float = 0.1
    k: int
    ft_init_checkpt_folder: Path | None = None
    ft_init_checkpt_epoch: int | None = None


class JanUpdateTrainConfig(BaseTrainConfig):
    c: float = 4.0
    lambda_s: float = 20.0
    """will be linearly ramped from 0 over the entire training run"""
    lambda_p: float = 3e-6

    epochs: int | None = 1
    base_save_dir: str = ".checkpoints"
    save_every_n_epochs: int | None = 1
    log_every_n_steps: int | None = 50
    n_batches_for_norm_estimate: int = 100
    num_test_batches: int = 10
    upload_checkpoint_to_wandb_every_n_epochs: int | None = None


class JanUpdateExperimentConfig(BaseExperimentConfig):
    crosscoder: JanUpdateCrosscoderConfig
    train: JanUpdateTrainConfig
