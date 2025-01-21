from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


class TokenSequenceIterator(ABC):
    @property
    @abstractmethod
    def sequence_length(self) -> int: ...

    @abstractmethod
    def sequence_iterator(self) -> Iterator[torch.Tensor]: ...


class HFDatasetTokenSequenceIterator(TokenSequenceIterator):
    def __init__(
        self,
        hf_dataset: str,
        cache_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ):
        self._hf_dataset = hf_dataset
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length

    @property
    def sequence_length(self):
        return self._sequence_length

    def sequence_iterator(self) -> Iterator[torch.Tensor]:
        dataset = load_dataset(self._hf_dataset, streaming=True, cache_dir=self._cache_dir)

        for example in cast(Any, dataset)["train"]:
            seq_tokens_S = torch.tensor(self._tokenizer(example["text"])["input_ids"])
            assert len(seq_tokens_S.shape) == 1, f"seq_tokens_S.shape should be 1D but was {seq_tokens_S.shape}"
            num_full_sequences = len(seq_tokens_S) // self._sequence_length
            if num_full_sequences == 0:
                continue

            for i in range(0, num_full_sequences * self._sequence_length, self._sequence_length):
                yield seq_tokens_S[i : i + self._sequence_length]


class MemoryTokenSequenceIterator(TokenSequenceIterator):
    def __init__(self, tokens_AS: torch.Tensor):
        self.tokens_AS = tokens_AS

    @property
    def sequence_length(self):
        return self.tokens_AS.shape[1]

    def __iter__(self):
        return self.tokens_AS


class LocalDatasetTokenSequenceIterator(TokenSequenceIterator):
    """backed by a file"""

    ...
