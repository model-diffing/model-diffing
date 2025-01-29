from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.models.crosscoder import build_relu_crosscoder
from model_diffing.scripts.config_common import AdamDecayTo0LearningRateConfig, BaseTrainConfig
from model_diffing.scripts.trainer import BaseTrainer, validate_num_steps_per_epoch
from model_diffing.utils import get_device


class TestTrainer(BaseTrainer[BaseTrainConfig]):
    def _train_step(self, batch_BMLD: Tensor) -> dict[str, float]:
        return {
            "loss": 0.0,
        }


class FakeActivationsDataloader(BaseActivationsDataloader):
    def __init__(
        self,
        batch_size: int = 16,
        n_models: int = 1,
        n_layers: int = 1,
        d_model: int = 16,
        num_batches: int = 100,
    ):
        self._batch_size = batch_size
        self._n_models = n_models
        self._n_layers = n_layers
        self._d_model = d_model
        self._num_batches = num_batches

    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        for _ in range(self._num_batches):
            yield torch.randint(
                0,
                100,
                (self._batch_size, self._n_layers, self._n_models, self._d_model),
                dtype=torch.float32,
            )

    def batch_shape_BMLD(self) -> tuple[int, int, int, int]:
        return (self._batch_size, self._n_layers, self._n_models, self._d_model)

    def num_batches(self) -> int | None:
        return self._num_batches


def opt():
    return AdamDecayTo0LearningRateConfig(initial_learning_rate=1e-3)


@pytest.mark.parametrize(
    "train_cfg",
    [
        BaseTrainConfig(epochs=10, optimizer=opt()),
        BaseTrainConfig(num_steps_per_epoch=100, optimizer=opt()),
    ],
)
def test_trainer_epochs_steps(train_cfg: BaseTrainConfig) -> None:
    batch_size = 16
    n_models = 1
    layer_indices_to_harvest = [0]
    n_layers = len(layer_indices_to_harvest)
    d_model = 16
    num_batches = 10

    activations_dataloader = FakeActivationsDataloader(
        batch_size=batch_size,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        num_batches=num_batches,
    )
    crosscoder = build_relu_crosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=16,
        dec_init_norm=0.0,
    )

    trainer = TestTrainer(
        cfg=train_cfg,
        activations_dataloader=activations_dataloader,
        crosscoder=crosscoder,
        wandb_run=None,
        device=get_device(),
        layers_to_harvest=layer_indices_to_harvest,
        experiment_name="test",
    )

    trainer.train()


def test_validate_num_steps_per_epoch_returns_num_steps_per_epoch_if_less_than_dataloader_num_batches() -> None:
    # WHEN num_steps_per_epoch is less than the number of batches in the dataloader,
    given_num_steps_per_epoch = 100
    dataloader_num_batches = 101
    cfg = BaseTrainConfig(epochs=10, num_steps_per_epoch=given_num_steps_per_epoch, optimizer=opt())
    activations_dataloader = FakeActivationsDataloader(num_batches=dataloader_num_batches)

    # THEN it should return num_steps_per_epoch, effectively cropping the dataloader
    assert validate_num_steps_per_epoch(cfg, activations_dataloader) == given_num_steps_per_epoch


def test_validate_num_steps_per_epoch_returns_dataloader_num_batches_if_num_steps_per_epoch_is_greater_than_dataloader_num_batches() -> (
    None
):
    # WHEN num_steps_per_epoch is greater than the number of batches in the dataloader,
    given_num_steps_per_epoch = 101
    dataloader_num_batches = 100
    cfg = BaseTrainConfig(epochs=10, num_steps_per_epoch=given_num_steps_per_epoch, optimizer=opt())
    activations_dataloader = FakeActivationsDataloader(num_batches=dataloader_num_batches)

    # THEN it should just use the number of batches in the dataloader
    assert validate_num_steps_per_epoch(cfg, activations_dataloader) == dataloader_num_batches


def test_validate_num_steps_per_epoch_works_if_epochs_but_not_num_steps_per_epoch_are_provided() -> None:
    # WHEN using epochs, we don't necessarily need to provide num_steps_per_epoch
    num_epochs = 10
    num_batches_in_dataloader = 30
    cfg = BaseTrainConfig(epochs=num_epochs, optimizer=opt())
    activations_dataloader = FakeActivationsDataloader(num_batches=num_batches_in_dataloader)

    # THEN it should just use the number of batches in the dataloader
    assert validate_num_steps_per_epoch(cfg, activations_dataloader) == num_batches_in_dataloader * num_epochs


def test_validate_num_steps_per_epoch_raises_error_if_num_steps_per_epoch_but_not_epochs_are_provided() -> None:
    # WHEN num_steps_per_epoch is provided but not epochs,
    cfg = BaseTrainConfig(num_steps_per_epoch=100, optimizer=opt())
    activations_dataloader = FakeActivationsDataloader(num_batches=99999)

    # THEN it should raise an error
    with pytest.raises(ValueError):
        validate_num_steps_per_epoch(cfg, activations_dataloader)


def test_validate_num_steps_per_epoch_raises_error_if_epochs_and_num_steps_are_provided() -> None:
    # WHEN both epochs and num_steps are provided,
    cfg = BaseTrainConfig(epochs=10, num_steps=100, optimizer=opt())
    activations_dataloader = FakeActivationsDataloader(num_batches=99999)

    # THEN it should raise an error
    with pytest.raises(ValueError):
        validate_num_steps_per_epoch(cfg, activations_dataloader)


def test_validate_num_steps_per_epoch_raises_error_nothing_provided() -> None:
    # WHEN neither epochs nor num_steps are provided,
    cfg = BaseTrainConfig(optimizer=opt())
    activations_dataloader = FakeActivationsDataloader(num_batches=99999)

    # THEN it should raise an error
    with pytest.raises(ValueError):
        validate_num_steps_per_epoch(cfg, activations_dataloader)
