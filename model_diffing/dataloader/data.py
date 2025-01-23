from collections.abc import Iterator

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.activations import ActivationsHarvester
from model_diffing.dataloader.shuffler import batch_shuffle_tensor_iterator_BX
from model_diffing.dataloader.token_loader import (
    CommonCorpusTokenSequenceIterator,
    ConnorGemma2TokenSequenceLoader,
    TokenSequenceLoader,
    ToyOverfittingTokenSequenceIterator,
)
from model_diffing.scripts.config_common import DataConfig, SequenceIteratorConfig


def build_dataloader_BMLD(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    cache_dir: str,
) -> Iterator[torch.Tensor]:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_iterator_S = _build_tokens_sequence_iterator(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
    ).get_sequence_iterator()

    # then, shuffle this iterator (only between, not within, sequences) so that we don't have to worry
    # about long documents introducing high feature correlations
    # this shuffler returns batches, hence (B, S)
    shuffled_token_sequence_iterator_BS = batch_shuffle_tensor_iterator_BX(
        tensor_iterator_X=token_sequence_iterator_S,
        shuffle_buffer_size=cfg.sequence_shuffle_buffer_size,
        yield_batch_size=cfg.activations_harvester.harvest_batch_size,
    )

    # then, run these sequences through the model to get activations
    token_activations_iterator_MLD = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=cfg.activations_harvester.layer_indices_to_harvest,
        token_sequence_iterator_BS=shuffled_token_sequence_iterator_BS,
    ).get_token_activations_iterator_MLD()

    # shuffle these token activations, so that we eliminate high feature correlations inside sequences
    shuffled_activations_iterator_BMLD = batch_shuffle_tensor_iterator_BX(
        tensor_iterator_X=token_activations_iterator_MLD,
        shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=cfg.cc_training_batch_size,
    )

    return shuffled_activations_iterator_BMLD


def _build_tokens_sequence_iterator(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenSequenceLoader:
    if cfg.classname == "CommonCorpusTokenSequenceIterator":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided for common_corpus")
        if cfg.kwargs["sequence_length"] is None:
            raise ValueError("sequence_length must be provided for common_corpus")
        return CommonCorpusTokenSequenceIterator(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            sequence_length=cfg.kwargs["sequence_length"],
        )
    elif cfg.classname == "ConnorGemma2TokenSequenceLoader":
        return ConnorGemma2TokenSequenceLoader(cache_dir=cache_dir)
    elif cfg.classname == "ToyOverfittingTokenSequenceIterator":
        return ToyOverfittingTokenSequenceIterator(tokenizer=tokenizer)
    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")
