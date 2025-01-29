from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from model_diffing.log import logger


class TokenSequenceLoader(ABC):
    @abstractmethod
    def get_sequence_iterator(self) -> Iterator[torch.Tensor]: ...


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
        dataset_iter = iter(
            load_dataset(self.COMMON_CORPUS_HF_DATASET, streaming=True, cache_dir=self._cache_dir, split="train")
        )

        demo_example = next(dataset_iter)
        logger.info(f"demo_example length: {len(demo_example['text'])}")
        logger.info(f"{len(demo_example['text']) / self._sequence_length:.2f} sequences per example")

        for example in dataset_iter:
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
            tokens = torch.tensor(example["input_ids"])  # type: ignore

            if printed < 10:
                print(f"{tokens.shape=}")
                printed += 1

            yield tokens


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

# from itertools import islice

# Load specific subset of shards


# class ThePileTokenSequenceIterator(TokenSequenceLoader):
#     HF_TOKENISED_DATASET = "EleutherAI/pile"

#     def __init__(self, cache_dir: str):
#         self._cache_dir = cache_dir

#     def get_sequence_iterator(self) -> Iterator[torch.Tensor]:
#         dataset = load_dataset(self.HF_TOKENISED_DATASET, streaming=True, cache_dir=self._cache_dir, split="train")
#         for example in dataset:
#             tokens = torch.tensor(example["input_ids"])  # type: ignore
#             yield tokens


# if __name__ == "__main__":
#     import itertools
#     import math
#     import os
#     from pathlib import Path

#     from datasets import load_dataset

#     # Configuration
#     TARGET_SIZE_GB = 20
#     OUTPUT_DIR = Path("pile_subset_data")

#     # Create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # First, load a small sample to estimate size per example
#     print("Estimating dataset size...")
#     sample_dataset = load_dataset(
#         "EleutherAI/pile",
#         split="train",
#         streaming=True,
#     )

#     # Sample 1000 examples to estimate average size
#     sample_size = 1000
#     total_bytes = 0
#     samples = []

#     for example in itertools.islice(sample_dataset, sample_size):
#         example = cast(dict[str, Any], example)
#         total_bytes += len(example.get("text", "").encode("utf-8"))
#         samples.append(example)

#     avg_bytes_per_example = total_bytes / sample_size
#     target_size_bytes = TARGET_SIZE_GB * 1024 * 1024 * 1024
#     estimated_examples_needed = math.ceil(target_size_bytes / avg_bytes_per_example)

#     print(f"Average example size: {avg_bytes_per_example / 1024 / 1024:.2f}MB")
#     print(f"Estimated examples needed for {TARGET_SIZE_GB}GB: {estimated_examples_needed:,}")

#     # Now load the actual subset using datasets' built-in functionality
#     print("\nDownloading dataset subset...")
#     dataset = load_dataset(
#         "EleutherAI/pile",
#         split=f"train[:{estimated_examples_needed}]",
#         cache_dir=str(OUTPUT_DIR),
#     )

#     if isinstance(dataset, Dataset):
#         print(f"\nSaving dataset to {OUTPUT_DIR}...")
#         dataset.save_to_disk(OUTPUT_DIR)

#         # Verify final size
#         total_size = sum(
#             os.path.getsize(os.path.join(dirpath, filename))
#             for dirpath, _, filenames in os.walk(OUTPUT_DIR)
#             for filename in filenames
#         )

#         print("\nDownload complete!")
#         print(f"Final size on disk: {total_size / 1024 / 1024 / 1024:.2f}GB")
#         print(f"Number of examples: {len(dataset):,}")
#         print(f"Data saved in: {OUTPUT_DIR.absolute()}")
#     else:
#         print("Error: Dataset was not loaded as expected. Please try a different approach.")
