from pathlib import Path

from pydantic import BaseModel

from model_diffing.scripts.config_common import (
    AdamDecayTo0LearningRateConfig,
    BaseExperimentConfig,
)

class TrainConfig(BaseModel):
    optimizer: AdamDecayTo0LearningRateConfig
    epochs: int
    base_save_dir: str = ".checkpoints"
    save_every_n_epochs: int | None
    log_every_n_steps: int
    n_batches_for_norm_estimate: int = 100
    num_test_batches: int = 10
    upload_checkpoint_to_wandb_every_n_epochs: int | None = None

# TODO
#    @field_serializer("save_dir")
#    def serialize_save_dir(self, save_dir: Path, _info):
#        # Return yaml serialization of str(save_dir), correctly escaping special characters
#        return yaml.dump(str(save_dir)).split('\n')[0] # TODO hacky

class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int
    ft_init_checkpt_folder: Path | None = None
    ft_init_checkpt_epoch: int | None = None

# TODO
#     @field_serializer("ft_init_checkpt_folder")
#     def serialize_ft_init_checkpt_folder(self, ft_init_checkpt_folder: Path, _info):
#         return yaml.dump(str(ft_init_checkpt_folder)).split('\n')[0] # TODO hacky


class TopKExperimentConfig(BaseExperimentConfig):
    crosscoder: TopKCrosscoderConfig
    train: TrainConfig