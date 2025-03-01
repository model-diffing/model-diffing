import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property

import torch
from einops import rearrange
from transformer_lens import HookedTransformer  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore  # type: ignore

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.data.token_loader import TokenSequenceLoader, build_tokens_sequence_loader
from model_diffing.log import logger
from model_diffing.scripts.config_common import DataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X


class BaseTokenHookpointActivationsDataloader(ABC):
    @abstractmethod
    def get_shuffled_activations_iterator_BTPD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...

    @abstractmethod
    def get_norm_scaling_factors_TP(self) -> torch.Tensor: ...


class SlidingWindowScaledActivationsDataloader(BaseTokenHookpointActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int | None,
        yield_batch_size: int,
        n_tokens_for_norm_estimate: int,
        window_size: int,
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size
        self._window_size_tokens = window_size

        # important note: using the raw iterator here, not the scaled one.
        self._norm_scaling_factors_TP = estimate_norm_scaling_factor_X(
            dataloader_BXD=self._shuffled_raw_activations_iterator_BTPD,
            n_tokens_for_norm_estimate=n_tokens_for_norm_estimate,
        )

        if self._activations_harvester.num_models != 1:
            raise ValueError(
                "ActivationHarvester is configured incorrectly. "
                "Should only be harvesting for 1 model for sliding window"
            )

    def get_norm_scaling_factors_TP(self) -> torch.Tensor:
        return self._norm_scaling_factors_TP

    @torch.no_grad()
    def _activations_iterator_TPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMPD = self._activations_harvester.get_activations_BSMPD(
                seq.tokens_BS,
                # attention_mask_BS=seq.attention_mask_BS,
            )

            # pick only the first (and only) model's activations
            assert activations_BSMPD.shape[2] == 1, "should only be doing 1 model at a time for sliding window"
            activations_BSPD = activations_BSMPD[:, :, 0]

            # flatten across batch and sequence dimensions, and filter out special tokens
            activations_BsPD = rearrange(activations_BSPD, "b s p d -> (b s) p d")
            special_tokens_mask_Bs = rearrange(seq.special_tokens_mask_BS, "b s -> (b s)")
            activations_BsPD = activations_BsPD[~special_tokens_mask_Bs]

            # sliding window over the sequence dimension, adding a new token dimension
            # new_seq_len = S - (self._window_size_tokens - 1)
            activations_BsPDT = activations_BsPD.unfold(dimension=0, size=2, step=1)
            activations_BsTPD = rearrange(activations_BsPDT, "bs p d t -> bs t p d")

            yield from activations_BsTPD

    @cached_property
    @torch.no_grad()
    def _shuffled_raw_activations_iterator_BTPD(self) -> Iterator[torch.Tensor]:
        activations_iterator_TPD = self._activations_iterator_TPD()

        if self._activations_shuffle_buffer_size is not None:
            # shuffle these token activations, so that we eliminate high feature correlations inside sequences
            shuffled_activations_iterator_BTPD = batch_shuffle_tensor_iterator_BX(
                tensor_iterator_X=activations_iterator_TPD,
                shuffle_buffer_size=self._activations_shuffle_buffer_size,
                yield_batch_size=self._yield_batch_size,
                name="llm activations",
            )

            yield from shuffled_activations_iterator_BTPD

        sample = next(activations_iterator_TPD)
        batch_BTPD = torch.empty([self._yield_batch_size, *sample.shape])
        batch_BTPD[0] = sample
        ptr = 1

        for batch_TPD in activations_iterator_TPD:
            batch_BTPD[ptr] = batch_TPD
            ptr += 1

            if ptr == self._yield_batch_size:
                yield batch_BTPD
                ptr = 0

    @cached_property
    @torch.no_grad()
    def _shuffled_activations_iterator_BTPD(self) -> Iterator[torch.Tensor]:
        raw_activations_iterator_BTPD = self._shuffled_raw_activations_iterator_BTPD
        scaling_factors_TP1 = rearrange(self.get_norm_scaling_factors_TP(), "t p -> t p 1")
        for unscaled_example_BTPD in raw_activations_iterator_BTPD:
            yield unscaled_example_BTPD * scaling_factors_TP1

    def num_batches(self) -> int | None:
        return self._token_sequence_loader.num_batches()

    def get_shuffled_activations_iterator_BTPD(self) -> Iterator[torch.Tensor]:
        return self._shuffled_activations_iterator_BTPD


def build_sliding_window_dataloader(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
    window_size: int,
) -> BaseTokenHookpointActivationsDataloader:
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

    # Create activations cache directory if cache is enabled
    activations_cache_dir = None
    if cfg.activations_harvester.cache_mode != "no_cache":
        activations_cache_dir = os.path.join(cache_dir, "activations_cache")
        logger.info(f"Activations will be cached in: {activations_cache_dir}")

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
        cache_dir=activations_cache_dir,
        cache_mode=cfg.activations_harvester.cache_mode,
    )

    activations_dataloader = SlidingWindowScaledActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        n_tokens_for_norm_estimate=cfg.n_tokens_for_norm_estimate,
        window_size=window_size,
    )

    return activations_dataloader
