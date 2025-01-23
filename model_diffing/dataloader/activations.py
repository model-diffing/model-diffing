from collections.abc import Iterator
from functools import cached_property

import torch
from einops import rearrange
from transformer_lens import HookedTransformer

from model_diffing.utils import chunk


class ActivationsHarvester:
    def __init__(
        self,
        llms: list[HookedTransformer],
        layer_indices_to_harvest: list[int],
        batch_size: int,
        sequence_tokens_iterator: Iterator[torch.Tensor],
    ):
        self._llms = llms
        self._layer_indices_to_harvest = layer_indices_to_harvest
        self._batch_size = batch_size
        self._sequence_tokens_iterator = sequence_tokens_iterator

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
        activations_BSLD = torch.stack([cache[name] for name in self.names], dim=2)  # adds layer dim (L)
        # cropped_activations_BSLD = activations_BSLD[:, 1:, :, :]  # remove BOS, need
        return activations_BSLD

    def _get_activations_BSMLD(self, sequence_BS: torch.Tensor) -> torch.Tensor:
        activations = [self._get_model_activations_BSLD(model, sequence_BS) for model in self._llms]
        activations_BSMLD = torch.stack(activations, dim=2)
        return activations_BSMLD

    @torch.no_grad()
    def get_token_activations_iterator_MLD(self) -> Iterator[torch.Tensor]:
        for sequences_chunk in chunk(self._sequence_tokens_iterator, self._batch_size):
            sequence_tokens_BS = torch.stack(sequences_chunk)
            activations_BSMLD = self._get_activations_BSMLD(sequence_tokens_BS)
            activations_BsMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
            yield from activations_BsMLD
