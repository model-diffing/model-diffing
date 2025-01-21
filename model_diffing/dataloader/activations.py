import random
from collections.abc import Iterator
from functools import cached_property

import torch
from einops import rearrange
from transformer_lens import HookedTransformer

from model_diffing.dataloader.sequences import TokenSequenceIterator
from model_diffing.utils import chunk


# make me a function?:
class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        layer_indices_to_harvest: list[int],
        sequence_iterator: TokenSequenceIterator,
        batch_size: int,
    ):
        self._llms = llms
        self._layer_indices_to_harvest = layer_indices_to_harvest
        self._sequence_iterator = sequence_iterator
        self._batch_size = batch_size

        assert len({llm.cfg.d_model for llm in self._llms}) == 1, "All llms must have the same d_model"
        self._d_model = self._llms[0].cfg.d_model

        assert len({llm.cfg.device for llm in self._llms}) == 1, "All llms must be on the same device"
        self._device = self._llms[0].cfg.device

    @property
    def n_llms(self) -> int:
        return len(self._llms)

    @property
    def n_layers_to_harvest(self) -> int:
        return len(self._layer_indices_to_harvest)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def sequence_length(self) -> int:
        return self._sequence_iterator.sequence_length

    @cached_property
    def _activation_shape_BSMLD(self):
        return (
            self._batch_size,
            self.sequence_length,
            self.n_llms,
            self.n_layers_to_harvest,
        )

    @cached_property
    def names(self) -> list[str]:
        return [f"blocks.{num}.hook_resid_post" for num in self._layer_indices_to_harvest]

    @cached_property
    def names_set(self) -> set[str]:
        return set(self.names)

    def _names_filter(self, name: str) -> bool:
        return name in self.names_set

    def _get_model_activations_BSLD(self, model: HookedTransformer, sequence_BS: torch.Tensor) -> torch.Tensor:
        _, cache = model.run_with_cache(sequence_BS, names_filter=self._names_filter)
        activations_BSLD = torch.stack([cache[name] for name in self.names], dim=2)  # add layer dim (L)
        # cropped_activations_BSLD = activations_BSLD[:, 1:, :, :]  # remove BOS, need
        return activations_BSLD

    def _get_activations_BSMLD(self, sequence_BS: torch.Tensor) -> torch.Tensor:
        activations = [self._get_model_activations_BSLD(model, sequence_BS) for model in self._llms]
        activations_BSMLD = torch.stack(activations, dim=2)
        assert activations_BSMLD.shape == self._activation_shape_BSMLD, (
            f"activations_BSMLD.shape should be {self._activation_shape_BSMLD} but was {activations_BSMLD.shape}"
        )

        return activations_BSMLD

    def get_activations_iterator_BSMLD(self) -> Iterator[torch.Tensor]:
        iterator = self._sequence_iterator.sequence_iterator()
        for sequences_chunk in chunk(iterator, self._batch_size):
            sequence_BS = torch.stack(sequences_chunk)
            # expected_shape_BS = (batch_size, sequence_length)
            # assert sequence_BS.shape == expected_shape_BS, (
            #     f"sequence_BS.shape should be {expected_shape_BS} but was {sequence_BS.shape}"
            # )
            yield self._get_activations_BSMLD(sequence_BS)


class ShuffledTokensActivationsLoader:
    """
    takes activations from an ActivationHarvester, flattens across batch and
    sequence dimensions (because sequence position is not a concept for crosscoder training)
    and shuffles them using a shuffle buffer.
    """

    def __init__(
        self,
        activations_harvester: ActivationsHarvester,
        shuffle_buffer_size: int,
        batch_size: int,
    ):
        self._activations_iterator = activations_harvester
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

        self._activation_shape_MLD = (
            self._activations_iterator.n_llms,
            self._activations_iterator.n_layers_to_harvest,
            self._activations_iterator.d_model,
        )

    def _get_activations_iterator_MLD(self) -> Iterator[torch.Tensor]:
        for activations_BSMLD in self._activations_iterator.get_activations_iterator_BSMLD():
            activations_BsMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
            yield from activations_BsMLD
            # If this "yield from" is hard to understand, it is equivalent to:
            # ```
            # for activations_MLD in activations_BsMLD:
            #     yield activations_MLD
            # ```

    def get_shuffled_activations_iterator_BMLD(self) -> Iterator[torch.Tensor]:
        if self._batch_size > self._shuffle_buffer_size // 2:
            raise ValueError(
                f"Batch size cannot be greater than half the buffer size, {self._batch_size} > {self._shuffle_buffer_size // 2}"
            )

        buffer_BMLD = torch.empty((self._shuffle_buffer_size, *self._activation_shape_MLD))

        available_indices = set()
        stale_indices = set(range(self._shuffle_buffer_size))

        iterator_MLD = self._get_activations_iterator_MLD()

        while True:
            # refill buffer
            for stale_idx, activation_MLD in zip(list(stale_indices), iterator_MLD, strict=False):
                assert activation_MLD.shape == self._activation_shape_MLD, (
                    f"activation_MLD.shape should be {self._activation_shape_MLD} but was {activation_MLD.shape}"
                )
                buffer_BMLD[stale_idx] = activation_MLD
                available_indices.add(stale_idx)
                stale_indices.remove(stale_idx)

            # yield batches until buffer is half empty
            while len(available_indices) >= self._shuffle_buffer_size // 2:
                batch_indices = random.sample(list(available_indices), self._batch_size)
                yield buffer_BMLD[batch_indices]
                available_indices.difference_update(batch_indices)
                stale_indices.update(batch_indices)


class ShuffledSequenceActivationsLoader:
    """
    takes activations from an ActivationHarvester, and shuffles sequences of activations using a
    shuffle buffer. Importantly, this doesn't shuffle across sequence dimensions, it leaves sequences
    in the same order as they were in the original dataset. Just shuffles across the batch dimension.
    """

    def __init__(
        self,
        activations_harvester: ActivationsHarvester,
        shuffle_buffer_size: int,
        batch_size: int,
    ):
        self._activations_harvester = activations_harvester
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

        self._activation_shape_SMLD = (
            self._activations_harvester.sequence_length,
            self._activations_harvester.n_llms,
            self._activations_harvester.n_layers_to_harvest,
            self._activations_harvester.d_model,
        )

    def _get_activations_iterator_SMLD(self) -> Iterator[torch.Tensor]:
        for activations_BSMLD in self._activations_harvester.get_activations_iterator_BSMLD():
            yield from activations_BSMLD
            # If this "yield from" is hard to understand, it is equivalent to:
            # ```
            # for activations_SMLD in activations_BSMLD:
            #     yield activations_SMLD
            # ```

    def get_shuffled_activations_iterator_BSMLD(self) -> Iterator[torch.Tensor]:
        if self._batch_size > self._shuffle_buffer_size // 2:
            raise ValueError(
                f"Batch size cannot be greater than half the buffer size, {self._batch_size} > {self._shuffle_buffer_size // 2}"
            )

        buffer_BSMLD = torch.empty((self._shuffle_buffer_size, *self._activation_shape_SMLD))

        available_indices = set()
        stale_indices = set(range(self._shuffle_buffer_size))

        iterator_SMLD = self._get_activations_iterator_SMLD()

        while True:
            # refill buffer
            for stale_idx, activation_SMLD in zip(list(stale_indices), iterator_SMLD, strict=False):
                assert activation_SMLD.shape == self._activation_shape_SMLD, (
                    f"activation_SMLD.shape should be {self._activation_shape_SMLD} but was {activation_SMLD.shape}"
                )
                buffer_BSMLD[stale_idx] = activation_SMLD
                available_indices.add(stale_idx)
                stale_indices.remove(stale_idx)

            # yield batches until buffer is half empty
            while len(available_indices) >= self._shuffle_buffer_size // 2:
                batch_indices = random.sample(list(available_indices), self._batch_size)
                yield buffer_BSMLD[batch_indices]
                available_indices.difference_update(batch_indices)
                stale_indices.update(batch_indices)
