from abc import ABC, abstractmethod
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import einops
import torch
import yaml  # type: ignore
from einops import reduce
from einops.einops import Reduction
from pydantic import BaseModel as _BaseModel
from torch import nn


class BaseModel(_BaseModel):
    class Config:
        extra = "forbid"


class SaveableModule(nn.Module, ABC):
    @abstractmethod
    def _dump_cfg(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "SaveableModule": ...

    def save(self, basepath: Path):
        basepath.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), basepath / "model.pt")
        with open(basepath / "model.cfg", "w") as f:
            yaml.dump(self._dump_cfg(), f)

    @classmethod
    def load(cls, basepath: Path):
        with open(basepath / "model.cfg") as f:
            cfg = yaml.safe_load(f)
            model = cls._from_cfg(cfg)
        model.load_state_dict(torch.load(basepath / "model.pt"))
        return model


# 1: this signature allows us to use these norms in einops.reduce
# 2: I (oli) find `l2_norm(x, dim=-1)` more readable than `x.norm(p=2, dim=-1)`


def l0_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=0, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


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


def _weighted_l1_sparsity_loss(
    W_dec_HMLD: torch.Tensor,
    hidden_BH: torch.Tensor,
    layer_reduction: Reduction,  # type: ignore
    model_reduction: Reduction,  # type: ignore
) -> torch.Tensor:
    assert (hidden_BH >= 0).all()
    # think about it like: each latent (called "hidden" here) has a separate projection onto each (model, layer)
    # so we have a separate l2 norm for each (hidden, model, layer)
    W_dec_l2_norms_HML = reduce(W_dec_HMLD, "hidden model layer dim -> hidden model layer", l2_norm)

    # to get the weighting factor for each latent, we reduce it's decoder norms for each (model, layer)
    reduced_norms_H = multi_reduce(
        W_dec_l2_norms_HML,
        "hidden model layer",
        ("layer", layer_reduction),
        ("model", model_reduction),
    )

    # now we weight the latents by the sum of their norms
    weighted_hiddens_BH = hidden_BH * reduced_norms_H
    weighted_l1_of_hiddens_BH = reduce(weighted_hiddens_BH, "batch hidden -> batch", l1_norm)
    return weighted_l1_of_hiddens_BH.mean()


sparsity_loss_l2_of_norms = partial(
    _weighted_l1_sparsity_loss,
    layer_reduction=l2_norm,
    model_reduction=l2_norm,
)

sparsity_loss_l1_of_norms = partial(
    _weighted_l1_sparsity_loss,
    layer_reduction=l1_norm,
    model_reduction=l1_norm,
)


def calculate_reconstruction_loss(activation_BXD: torch.Tensor, target_BXD: torch.Tensor) -> torch.Tensor:
    """This is a little weird because we have both model and layer dimensions, so it's worth explaining deeply:

    The reconstruction loss is a sum of squared L2 norms of the error for each activation space being reconstructed.
    In the Anthropic crosscoders update, they don't write for the multiple-model case, they write it as:

    $$\\sum_{l \\in L} \\|a^l(x_j) - a^{l'}(x_j)\\|^2$$

    Here, I'm assuming we want to expand that sum to be over models, so we would have:

    $$ \\sum_{m \\in M} \\sum_{l \\in L} \\|a_m^l(x_j) - a_m^{l'}(x_j)\\|^2 $$
    """
    # take the L2 norm of the error inside each d_model feature space
    error_BXD = activation_BXD - target_BXD
    error_norm_BX = reduce(error_BXD, "batch ... d_model -> batch ...", l2_norm)
    squared_error_norm_BX = error_norm_BX.square()

    # sum errors across all crosscoding dimensions
    summed_squared_error_norm_B = reduce(squared_error_norm_BX, "batch ... -> batch", torch.sum)
    return summed_squared_error_norm_B.mean()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# (oli) sorry - this is probably overengineered
def multi_reduce(
    tensor: torch.Tensor,
    shape_pattern: str,
    *reductions: tuple[str, Reduction],  # type: ignore
) -> torch.Tensor:
    original_shape = einops.parse_shape(tensor, shape_pattern)
    for reduction_dim, reduction_fn in reductions:
        if reduction_dim not in original_shape:
            raise ValueError(f"Dimension {reduction_dim} not found in original_shape {original_shape}")
        target_pattern_pattern = shape_pattern.replace(reduction_dim, "")
        exec_pattern = f"{shape_pattern} -> {target_pattern_pattern}"
        shape_pattern = target_pattern_pattern
        tensor = reduce(tensor, exec_pattern, reduction_fn)

    return tensor


def calculate_explained_variance_X(
    activations_BXD: torch.Tensor,
    reconstructed_BXD: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """for each model and layer, calculate the mean explained variance inside each d_model feature space"""
    error_BXD = activations_BXD - reconstructed_BXD

    mean_error_var_X = error_BXD.var(-1).mean(0)
    mean_activations_var_X = activations_BXD.var(-1).mean(0)

    explained_var_X = 1 - (mean_error_var_X / (mean_activations_var_X + eps))
    return explained_var_X


def get_explained_var_dict(
    explained_variance_X: torch.Tensor, *crosscoding_dims: tuple[str, list[str] | list[int]]
) -> dict[str, float]:
    """
    crosscoding_dims is a list of tuples, each tuple is:
        1: the name of the crosscoding dimension ('layer', 'model', 'token', etc.)
        2: the labels of the crosscoding dimension (e.g. [0, 1, 7] or ['gpt2', 'gpt3', 'gpt4'], or ['<bos>', '-1', 'self'])
    
    the reason we need the explicit naming pattern is that often indices are not helpful. For example, when training
    a crosscoder on layers 2, 5, and 8, you don't to want to have them labeled [0, 1, 2]. i.e. you need to know what
    each index means.
    """

    assert len(crosscoding_dims) == len(explained_variance_X.shape)

    # index_combinations is a list of tuples, each tuple is a unique set of indices into the explained_variance_X tensor
    index_combinations = product(*(range(dim_size) for dim_size in explained_variance_X.shape))

    explained_variances_dict = {}
    for indices in index_combinations:
        name = "train/explained_variance"
        for (dim_name, dim_labels), dim_index in zip(crosscoding_dims, indices, strict=True):
            name += f"_{dim_name}{dim_labels[dim_index]}"

        explained_variances_dict[name] = explained_variance_X[indices].item()

    return explained_variances_dict


def get_decoder_norms_H(W_dec_HXD: torch.Tensor) -> torch.Tensor:
    W_dec_l2_norms_HX = reduce(W_dec_HXD, "hidden ... dim -> hidden ...", l2_norm)
    norms_H = reduce(W_dec_l2_norms_HX, "hidden ... -> hidden", torch.sum)
    return norms_H


def size_human_readable(tensor: torch.Tensor) -> str:
    # Calculate the number of bytes in the tensor
    num_bytes = tensor.numel() * tensor.element_size()

    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes} B"


# hacky but useful for debugging
def inspect(tensor: torch.Tensor) -> str:
    return f"{tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, size={size_human_readable(tensor)}"
