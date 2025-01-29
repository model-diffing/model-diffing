from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.models.crosscoder import build_relu_crosscoder
from model_diffing.scripts.config_common import AdamDecayTo0LearningRateConfig, BaseTrainConfig
from model_diffing.scripts.trainer import BaseTrainer
from model_diffing.utils import get_device


class TestTrainer(BaseTrainer[BaseTrainConfig]):
    def _train_step(self, batch_BMLD: Tensor) -> dict[str, float]:
        return {
            "loss": 0.0,
        }


class FakeActivationsDataloader(BaseActivationsDataloader):
    def __init__(self, batch_size: int, n_models: int, n_layers: int, d_model: int, num_batches: int):
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


@pytest.mark.parametrize(
    "train_cfg",
    [
        BaseTrainConfig(
            epochs=10,
            optimizer=AdamDecayTo0LearningRateConfig(
                initial_learning_rate=1e-3,
                last_pct_of_steps=0.2,
            ),
        ),
        BaseTrainConfig(
            num_steps=100,
            optimizer=AdamDecayTo0LearningRateConfig(
                initial_learning_rate=1e-3,
                last_pct_of_steps=0.2,
            ),
        ),
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
