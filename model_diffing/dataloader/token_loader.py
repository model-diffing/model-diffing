from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequence_iterator(self) -> Iterator[torch.Tensor]: ...


class SleeperTokenSequenceIterator(TokenSequenceLoader):
    SLEEPER_HF_DATASET = "mars-jason-25/processed_dolphin_IHY_sleeper_distilled_dataset"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        cache_dir: str | None = None,
        include_sleeper_data: bool = True,
        validation: bool = False,
    ):
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = 128
        self._include_sleeper_data = include_sleeper_data
        self._validation = validation

    def get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        text_dataset = load_dataset(
            self.SLEEPER_HF_DATASET, streaming=True, cache_dir=self._cache_dir, split=("test" if self._validation else "train")
        )

        for example in text_dataset:
            if not self._include_sleeper_data and not example["is_training"]:
                continue
            example = cast(dict[str, Any], example)
            tokeniser_result = self._tokenizer(example["text"])
            seq_tokens_S = torch.tensor(tokeniser_result["input_ids"])
            assert len(seq_tokens_S.shape) == 1, f"seq_tokens_S.shape should be 1D but was {seq_tokens_S.shape}"
            if len(seq_tokens_S) < self._sequence_length:
                continue
            else:
                yield seq_tokens_S[0 : self._sequence_length]


class CommonCorpusTokenSequenceIterator(TokenSequenceLoader):
    COMMON_CORPUS_HF_DATASET = "PleIAs/common_corpus"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        cache_dir: str | None = None,
    ):
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length

    def get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        text_dataset = load_dataset(
            self.COMMON_CORPUS_HF_DATASET, streaming=True, cache_dir=self._cache_dir, split="train"
        )

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


class ToyOverfittingTokenSequenceIterator(TokenSequenceLoader):
    def __init__(self, sequence_length: int):
        self._sequence_length = sequence_length

    def get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.randint(0, 1000, (self._sequence_length,))


class ConnorGemma2TokenSequenceLoader(TokenSequenceLoader):
    HF_TOKENISED_DATASET = "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2"

    def __init__(self, cache_dir: str):
        """expects a tokenised huggingface dataset"""
        self._cache_dir = cache_dir

    def get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        printed = 0
        dataset = load_dataset(self.HF_TOKENISED_DATASET, streaming=True, cache_dir=self._cache_dir, split="train")
        for example in dataset:
            tokens = example["input_ids"]  # type: ignore

            if printed < 10:
                print(f"{tokens.shape=}")
                printed += 1

            yield torch.tensor(tokens)


# For example, we could do:
# class LocalDatasetTokenSequenceIterator(TokenSequenceIterator):
#     """backed by a file"""
#     ...

# or

# class MemoryTokenSequenceIterator(TokenSequenceLoader):
#     def __init__(self, tokens_AS: torch.Tensor):
#         self.tokens_AS = tokens_AS

#     def __iter__(self) -> Iterator[torch.Tensor]:
#         return iter(self.tokens_AS)
