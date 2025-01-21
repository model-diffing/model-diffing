from collections.abc import Iterator

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.dataloader.activations import (
    ActivationsHarvester,
    ActivationsShuffler,
    iterate_over_tokens,
)
from model_diffing.dataloader.sequences import (
    CommonCorpusTokenSequenceIterator,
    ConnorsTokenSequenceLoader,
    TokenSequenceLoader,
)
from model_diffing.scripts.config_common import (
    ActivationsIteratorConfig,
    CommonCorpusTokenSequenceIteratorConfig,
    ConnorsTokenSequenceLoaderConfig,
    DataConfig,
)


def build_dataloader_BMLD(
    cfg: DataConfig,
    llms: list[HookedTransformer],
    cache_dir: str,
) -> Iterator[torch.Tensor]:
    acts_iterator = build_activations_iterator(cfg.activations_iterator, cache_dir, llms)

    shuffler = ActivationsShuffler(
        shuffle_buffer_size=cfg.shuffle_config.shuffle_buffer_size,
        batch_size=cfg.batch_size,
        activations_iterator_BSMLD=acts_iterator.get_activations_iterator_BSMLD(),
        activations_reshaper=iterate_over_tokens,
    )

    return shuffler.get_shuffled_activations_iterator()


def build_activations_iterator(
    cfg: ActivationsIteratorConfig,
    cache_dir: str,
    llms: list[HookedTransformer],
) -> ActivationsHarvester:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")
    sequence_iterator = build_sequence_iterator(cfg.sequence_iterator, cache_dir, tokenizer)
    return ActivationsHarvester(
        llms=llms,
        batch_size=cfg.harvest_batch_size,
        layer_indices_to_harvest=cfg.layer_indices_to_harvest,
        sequence_iterator=sequence_iterator.get_sequence_iterator(),
    )


def build_sequence_iterator(
    cfg: CommonCorpusTokenSequenceIteratorConfig | ConnorsTokenSequenceLoaderConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenSequenceLoader:
    if isinstance(cfg, CommonCorpusTokenSequenceIteratorConfig):
        return CommonCorpusTokenSequenceIterator(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            sequence_length=cfg.sequence_length,
        )
    elif isinstance(cfg, ConnorsTokenSequenceLoaderConfig):
        return ConnorsTokenSequenceLoader(
            cache_dir=cfg.cache_dir,
            sequence_length=cfg.sequence_length,
        )
    raise ValueError(f"Unknown sequence iterator config: {cfg}")
