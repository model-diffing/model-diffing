from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property

import torch
from einops import rearrange
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.data.token_loader import TokenSequenceLoader, build_tokens_sequence_loader
from model_diffing.scripts.config_common import DataConfig
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X


class BaseModelLayerActivationsDataloader(ABC):
    @abstractmethod
    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def batch_shape_BMLD(self) -> tuple[int, int, int, int]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_ML(self) -> torch.Tensor: ...


class ScaledModelLayerActivationsDataloader(BaseModelLayerActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int,
        yield_batch_size: int,
        device: torch.device,
        n_batches_for_norm_estimate: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size

        # important note: using the raw iterator here, not the scaled one.
        self._norm_scaling_factors_ML = estimate_norm_scaling_factor_X(
            self._shuffled_raw_activations_iterator_BMLD,
            device,
            n_batches_for_norm_estimate,
        )

    def get_norm_scaling_factors_ML(self) -> torch.Tensor:
        return self._norm_scaling_factors_ML

    @torch.no_grad()
    def _activations_iterator_MLD(self) -> Iterator[torch.Tensor]:
        for sequences_chunk_BS in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMLD = self._activations_harvester.get_activations_BSMLD(sequences_chunk_BS)
            activations_BsMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
            yield from activations_BsMLD

    @cached_property
    @torch.no_grad()
    def _shuffled_raw_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        activations_iterator_MLD = self._activations_iterator_MLD()

        # shuffle these token activations, so that we eliminate high feature correlations inside sequences
        shuffled_activations_iterator_BMLD = batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=activations_iterator_MLD,
            shuffle_buffer_size=self._activations_shuffle_buffer_size,
            yield_batch_size=self._yield_batch_size,
            name="llm activations",
        )

        return shuffled_activations_iterator_BMLD

    @cached_property
    @torch.no_grad()
    def _shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        raw_activations_iterator_BMLD = self._shuffled_raw_activations_iterator_BMLD
        scaling_factors_ML1 = rearrange(self.get_norm_scaling_factors_ML(), "m l -> m l 1")
        for unscaled_example_BMLD in raw_activations_iterator_BMLD:
            yield unscaled_example_BMLD * scaling_factors_ML1

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def batch_shape_BMLD(self) -> tuple[int, int, int, int]:
        return (
            self._yield_batch_size,
            *self._activations_harvester.activation_shape_MLD(),
        )

    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        return self._shuffled_activations_iterator_BMLD


def build_dataloader(
    cfg: DataConfig,
    batch_size: int,
    cache_dir: str,
    device: torch.device,
) -> BaseModelLayerActivationsDataloader:
    llms = build_llms(
        cfg.activations_harvester.llms,
        cache_dir,
        device,
        dtype=cfg.activations_harvester.inference_dtype,
    )

    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=cfg.activations_harvester.layer_indices_to_harvest,
    )

    activations_dataloader = ScaledModelLayerActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        device=device,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
    )

    return activations_dataloader