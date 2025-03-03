from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import Tensor

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer, validate_num_steps_per_epoch
from model_diffing.scripts.config_common import AdamConfig, BaseTrainConfig
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.utils import get_device


class TestTrainer(BaseModelHookpointTrainer[BaseTrainConfig, Any]):
    __test__ = False

    def _train_step(self, batch_BMPD: Tensor) -> None:
        pass


class FakeActivationsDataloader(BaseModelHookpointActivationsDataloader):
    __test__ = False

    def __init__(
        self,
        batch_size: int = 16,
        n_models: int = 1,
        n_hookpoints: int = 1,
        d_model: int = 16,
        num_batches: int = 100,
    ):
        self._batch_size = batch_size
        self._n_models = n_models
        self._n_hookpoints = n_hookpoints
        self._d_model = d_model
        self._num_batches = num_batches

    def get_activations_iterator_BMPD(self) -> Iterator[Tensor]:
        for _ in range(self._num_batches):
            yield torch.randint(
                0,
                100,
                (self._batch_size, self._n_models, self._n_hookpoints, self._d_model),
                dtype=torch.float32,
            )

    def num_batches(self) -> int | None:
        return self._num_batches

    def get_norm_scaling_factors_MP(self) -> torch.Tensor:
        return torch.ones(self._n_models, self._d_model)


def opt():
    return AdamConfig(learning_rate=1e-3)


@pytest.mark.parametrize(
    "train_cfg",
    [
        BaseTrainConfig(batch_size=1, epochs=2, optimizer=opt()),
        BaseTrainConfig(batch_size=1, epochs=2, num_steps_per_epoch=10, optimizer=opt()),
        BaseTrainConfig(batch_size=1, num_steps=10, optimizer=opt()),
    ],
)
def test_trainer_epochs_steps(train_cfg: BaseTrainConfig) -> None:
    batch_size = 4
    n_models = 1
    hookpoints = ["blocks.0.hook_resid_post"]
    n_hookpoints = len(hookpoints)
    d_model = 16
    num_batches = 10

    activations_dataloader = FakeActivationsDataloader(
        batch_size=batch_size,
        n_models=n_models,
        n_hookpoints=n_hookpoints,
        d_model=d_model,
        num_batches=num_batches,
    )

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=16,
        init_strategy=AnthropicTransposeInit(dec_init_norm=0.0),
        hidden_activation=ReLUActivation(),
    )

    trainer = TestTrainer(
        cfg=train_cfg,
        activations_dataloader=activations_dataloader,
        crosscoder=crosscoder,
        wandb_run=None,
        device=get_device(),
        hookpoints=hookpoints,
        save_dir="test_save_dir",
    )

    trainer.train()


@pytest.mark.parametrize(
    "epochs, num_steps_per_epoch, dataloader_num_batches, expected",
    [
        # WHEN num_steps_per_epoch < dataloader_num_batches (should return num_steps_per_epoch)
        (10, 100, 200, 100),
        # WHEN num_steps_per_epoch > dataloader_num_batches (should return dataloader_num_batches)
        (10, 200, 100, 100),
        # WHEN epochs is provided but num_steps_per_epoch is not (should return dataloader_num_batches)
        (10, None, 100, 100),
    ],
)
def test_validate_num_steps_per_epoch_happy_path(
    epochs: int,
    num_steps_per_epoch: int | None,
    dataloader_num_batches: int,
    expected: int,
) -> None:
    activations_dataloader = FakeActivationsDataloader(num_batches=dataloader_num_batches)
    num_steps_per_epoch = validate_num_steps_per_epoch(
        epochs, num_steps_per_epoch, None, activations_dataloader.num_batches()
    )
    assert num_steps_per_epoch == expected


@pytest.mark.parametrize(
    "epochs, num_steps_per_epoch, num_steps, should_raise",
    [
        # WHEN num_steps_per_epoch is provided but not epochs
        (None, 100, None, ValueError),
        # WHEN both epochs and num_steps are provided
        (10, None, 100, ValueError),
        # WHEN neither epochs nor num_steps are provided
        (None, None, None, ValueError),
    ],
)
def test_validate_num_steps_per_epoch_errors(
    epochs: int | None,
    num_steps_per_epoch: int | None,
    num_steps: int | None,
    should_raise: type[Exception],
) -> None:
    activations_dataloader = FakeActivationsDataloader(num_batches=99999)

    with pytest.raises(should_raise):
        validate_num_steps_per_epoch(epochs, num_steps_per_epoch, num_steps, activations_dataloader.num_batches())
