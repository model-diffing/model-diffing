import torch
from transformers import PreTrainedTokenizerBase  # type: ignore

from model_diffing.dataloader.activations import (
    ActivationsHarvester,
    BaseActivationsDataloader,
    ScaledActivationsDataloader,
)
from model_diffing.dataloader.token_loader import (
    ConnorGemma2TokenSequenceLoader,
    HuggingfaceTextDatasetTokenSequenceLoader,
    TokenSequenceLoader,
    ToyOverfittingTokenSequenceLoader,
)
from model_diffing.scripts.config_common import DataConfig, SequenceIteratorConfig
from model_diffing.scripts.llms import build_llms


def build_dataloader(
    cfg: DataConfig,
    batch_size: int,
    cache_dir: str,
    device: torch.device,
) -> BaseActivationsDataloader:
    llms = build_llms(
        cfg.activations_harvester.llms,
        cache_dir,
        device,
        dtype=cfg.activations_harvester.inference_dtype,
    )

    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = _build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        layer_indices_to_harvest=cfg.activations_harvester.layer_indices_to_harvest,
    )

    activations_dataloader = ScaledActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        device=device,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
    )

    return activations_dataloader


def _build_tokens_sequence_loader(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> TokenSequenceLoader:
    if cfg.classname == "HuggingfaceTextDatasetTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided")
        return HuggingfaceTextDatasetTokenSequenceLoader(
            hf_dataset_name=cfg.kwargs["hf_dataset_name"],
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            **cfg.kwargs,
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
