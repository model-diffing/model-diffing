from collections.abc import Iterator

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.activations import ActivationsHarvester
from model_diffing.dataloader.shuffler import shuffle_tensor_iterator
from model_diffing.dataloader.token_loader import (
    CommonCorpusTokenSequenceIterator,
    ConnorGemma2TokenSequenceLoader,
    TokenSequenceLoader,
    ToyOverfittingTokenSequenceIterator,
)
from model_diffing.scripts.config_common import DataConfig, SequenceTokensIteratorConfig


def build_dataloader_BMLD(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    cache_dir: str,
) -> Iterator[torch.Tensor]:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    sequence_tokens_iterator = _build_tokens_sequence_iterator(
        cfg.activations_iterator.sequence_tokens_iterator, cache_dir, tokenizer
    ).get_sequence_iterator()

    # then, shuffle this iterator so that we don't have to worry about long documents
    shuffled_sequence_tokens_iterator = shuffle_tensor_iterator(
        shuffle_buffer_size=cfg.activations_iterator.sequence_shuffler_buffer_size,
        tensor_iterator_X=sequence_tokens_iterator,
        yield_batch_size=cfg.activations_iterator.harvest_batch_size,
    )

    # then, run these sequences (locally coherent, globally shuffled) through the model to get activations
    acts_iterator = ActivationsHarvester(
        llms=llms,
        batch_size=cfg.activations_iterator.harvest_batch_size,
        layer_indices_to_harvest=cfg.activations_iterator.layer_indices_to_harvest,
        sequence_tokens_iterator=shuffled_sequence_tokens_iterator,
    )

    # then, reshape the activations so that we can iterate over them in a way that's easy to shuffle again
    token_activations_iterator_MLD = acts_iterator.get_token_activations_iterator_MLD()

    # shuffle these token activations, so that we eliminate high feature correlations inside sequences
    shuffled_activations_iterator = shuffle_tensor_iterator(
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        tensor_iterator_X=token_activations_iterator_MLD,
        yield_batch_size=cfg.batch_size,
    )

    return shuffled_activations_iterator


def _build_tokens_sequence_iterator(
    cfg: SequenceTokensIteratorConfig,
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
