
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.scripts.config_common import SequenceIteratorConfig


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def num_batches(self) -> int | None: ...  # not using __len__ because __len__ doesn't work well with `| None`


COMMON_CORPUS_HF_DATASET = "PleIAs/common_corpus"
THE_PILE_UNCOPYRIGHTED_HF_DATASET = "monology/pile-uncopyrighted"


class HuggingfaceTextDatasetTokenSequenceLoader(TokenSequenceLoader):
    def __init__(
        self,
        hf_dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        shuffle_buffer_size: int,
        batch_size: int,
        cache_dir: str | None = None,
    ):
        self.hf_dataset_name = hf_dataset_name
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size

        self._iterator = self._get_sequences_batch_iterator()

    def _get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        text_dataset = load_dataset(self.hf_dataset_name, streaming=True, cache_dir=self._cache_dir, split="train")

        for example in text_dataset:
            example = cast(dict[str, Any], example)
            tokeniser_result = self._tokenizer(example["text"])
            seq_tokens_S = torch.tensor(tokeniser_result["input_ids"])
            assert len(seq_tokens_S.shape) == 1, f"seq_tokens_S.shape should be 1D but was {seq_tokens_S.shape}"
            num_full_sequences = len(seq_tokens_S) // self._sequence_length
            if num_full_sequences == 0:
                continue

            for i in range(0, num_full_sequences * self._sequence_length, self._sequence_length):
                yield seq_tokens_S[i : i + self._sequence_length]

    def _get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        # we shuffle this iterator (only between, not within, sequences) so that we don't have to worry
        # about long documents introducing high feature correlations
        # this shuffler returns batches of sequences of tokens.
        return batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=self._get_sequence_iterator(),
            shuffle_buffer_size=self._shuffle_buffer_size,
            yield_batch_size=self._batch_size,
            name=f"{self.hf_dataset_name} sequences",
        )

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        return self._iterator

    def num_batches(self) -> int | None:
        # This kind of can't easily be computed, because it's a function of sequence length and each example's length
        # This is a good example of why `num_batches` is `None`able
        return None


class ToyOverfittingTokenSequenceLoader(TokenSequenceLoader):
    def __init__(self, batch_size: int, sequence_length: int):
        self._batch_size = batch_size
        self._sequence_length = sequence_length

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.randint(0, 1000, (self._batch_size, self._sequence_length))

    def num_batches(self) -> int | None:
        return None


class ConnorGemma2TokenSequenceLoader(TokenSequenceLoader):
    HF_TOKENISED_DATASET = "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2"
    SEQUENCE_LENGTH = 1024
    N_ROWS = 963_566

    def __init__(self, cache_dir: str, batch_size: int):
        """expects a tokenised huggingface dataset"""
        self._cache_dir = cache_dir
        self._batch_size = batch_size

    def _batch_accumulator(self, sequence_iterator: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
        """accumulate sequences into batches, yielding batches of shape (B, S)"""
        buffer = torch.empty((self._batch_size, self.SEQUENCE_LENGTH))
        pos = 0

        for sequence in sequence_iterator:
            buffer[pos] = sequence
            pos += 1
            if pos == self._batch_size:
                yield buffer.clone()
                pos = 0

    def get_sequences_batch_iterator(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(
            self.HF_TOKENISED_DATASET,
            streaming=True,
            cache_dir=self._cache_dir,
            split="train",
            batch_size=self._batch_size,
        )
        sequence_iterator = (
            torch.tensor(tokens_S["input_ids"]) for tokens_S in cast(Iterator[dict[str, Any]], dataset)
        )
        return self._batch_accumulator(sequence_iterator)

    def num_batches(self) -> int | None:
        return self.N_ROWS // self._batch_size


def build_tokens_sequence_loader(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> TokenSequenceLoader:
    if cfg.classname == "HuggingfaceTextDatasetTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided")
        return HuggingfaceTextDatasetTokenSequenceLoader(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            hf_dataset_name=cfg.kwargs["hf_dataset_name"],
            sequence_length=cfg.kwargs["sequence_length"],
            shuffle_buffer_size=cfg.kwargs["shuffle_buffer_size"],
        )
    elif cfg.classname == "ConnorGemma2TokenSequenceLoader":
        return ConnorGemma2TokenSequenceLoader(
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
    elif cfg.classname == "ToyOverfittingTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided")
        return ToyOverfittingTokenSequenceLoader(
            batch_size=batch_size,
            **cfg.kwargs,
        )

    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")


# if __name__ == "__main__":
#     from itertools import islice

#     from transformers import AutoTokenizer

#     token_loader = HuggingfaceTextDatasetTokenSequenceLoader(
#         hf_dataset_name=THE_PILE_UNCOPYRIGHTED_HF_DATASET,
#         tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"),
#         cache_dir=".cache",
#         batch_size=16,
#         sequence_length=1024,
#         shuffle_buffer_size=2**14,  # 16k
#     )
#     for tokens in islice(token_loader.get_sequences_batch_iterator(), 10):
#         print(tokens.shape)