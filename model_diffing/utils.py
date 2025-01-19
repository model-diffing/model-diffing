from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import torch
import yaml
from einops import reduce
from pydantic import BaseModel
from torch import nn

from model_diffing.log import logger


def save_model_and_config(config: BaseModel, save_dir: Path, model: nn.Module, epoch: int) -> None:
    """Save the model to disk. Also save the config file if it doesn't exist.

    Args:
        config: The config object. Saved if save_dir / "config.yaml" doesn't already exist.
        save_dir: The directory to save the model and config to.
        model: The model to save.
        epoch: The current epoch (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if not (save_dir / "config.yaml").exists():
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info("Saved config to %s", save_dir / "config.yaml")

    model_file = save_dir / f"model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), model_file)
    logger.info("Saved model to %s", model_file)


T = TypeVar("T")


def chunk(iterable: Iterator[T], size: int) -> Iterator[list[T]]:
    chunk: list[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def l1_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=1, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def l2_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=2, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def sparsity_loss(W_dec_HMLD: torch.Tensor, hidden_BH: torch.Tensor) -> torch.Tensor:
    assert (hidden_BH >= 0).all()
    # each latent has a separate norms for each (model, layer)
    W_dec_l2_norms_HML = reduce(W_dec_HMLD, "hidden model layer dim -> hidden model layer", l2_norm)
    # to get the weighting factor for each latent, we sum it's decoder norms for each (model, layer)
    summed_norms_H = reduce(W_dec_l2_norms_HML, "hidden model layer -> hidden", torch.sum)
    # now we weight the latents by the sum of their norms
    weighted_hidden_BH = hidden_BH * summed_norms_H
    summed_weighted_hidden_B = reduce(weighted_hidden_BH, "batch hidden -> batch", torch.sum)
    return summed_weighted_hidden_B.mean()


def reconstruction_loss(activation_BMLD: torch.Tensor, target_BMLD: torch.Tensor) -> torch.Tensor:
    """This is a little weird because we have both model and layer dimensions, so it's worth explaining deeply:

    The reconstruction loss is a sum of squared L2 norms of the error for each activation space being reconstructed.
    In the Anthropic crosscoders update, they don't write for the multiple-model case, so they write it as:

    $$ \\sum_{l \\in L} \\|a^l(x_j) - a^{l'}(x_j)\\|^2 $$

    Here, I'm assuming we want to expand that sum to be over models, so we would have:

    $$ \\sum_{m \\in M} \\sum_{l \\in L} \\|a_m^l(x_j) - a_m^{l'}(x_j)\\|^2 $$
    """
    error_BMLD = activation_BMLD - target_BMLD
    error_norm_BML = reduce(error_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
    squared_error_norm_BML = error_norm_BML.square()
    summed_squared_error_norm_B = reduce(squared_error_norm_BML, "batch model layer -> batch", torch.sum)
    return summed_squared_error_norm_B.mean()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
