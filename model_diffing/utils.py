from functools import partial
from pathlib import Path

import einops
import torch
import wandb
import yaml
from einops import reduce
from einops.einops import Reduction
from pydantic import BaseModel
from torch import nn
from wandb.sdk.wandb_run import Run

from model_diffing.log import logger
from model_diffing.scripts.config_common import WandbConfig


def build_wandb_run(wandb_config: WandbConfig, config: BaseModel) -> Run | None:
    return wandb.init(
        name=wandb_config.name,
        project=wandb_config.project,
        entity=wandb_config.entity,
        config=config.model_dump(),
    )


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


# It may seem weird to redefine these, but:
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


def weighted_l1_sparsity_loss(
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
    weighted_l1_sparsity_loss,
    layer_reduction=l2_norm,
    model_reduction=l2_norm,
)

sparsity_loss_l1_of_norms = partial(
    weighted_l1_sparsity_loss,
    layer_reduction=l1_norm,
    model_reduction=l1_norm,
)


def calculate_reconstruction_loss(activation_BMLD: torch.Tensor, target_BMLD: torch.Tensor) -> torch.Tensor:
    """This is a little weird because we have both model and layer dimensions, so it's worth explaining deeply:

    The reconstruction loss is a sum of squared L2 norms of the error for each activation space being reconstructed.
    In the Anthropic crosscoders update, they don't write for the multiple-model case, they write it as:

    $$\\sum_{l \\in L} \\|a^l(x_j) - a^{l'}(x_j)\\|^2$$

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


def calculate_explained_variance_ML(
    activations_BMLD: torch.Tensor,
    reconstructed_BMLD: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """for each model and layer, calculate the mean explained variance inside each d_model feature space"""
    error_BMLD = activations_BMLD - reconstructed_BMLD

    mean_error_var_ML = error_BMLD.var(-1).mean(0)
    mean_activations_var_ML = activations_BMLD.var(-1).mean(0)

    explained_var_ML = 1 - (mean_error_var_ML / (mean_activations_var_ML + eps))
    return explained_var_ML


def get_explained_var_dict(explained_variance_ML: torch.Tensor, layers_to_harvest: list[int]) -> dict[str, float]:
    num_models, _n_layers = explained_variance_ML.shape
    explained_variances_dict = {
        f"train/explained_variance/M{model_idx}_L{layer_number}": explained_variance_ML[model_idx, layer_idx].item()
        for model_idx in range(num_models)
        for layer_idx, layer_number in enumerate(layers_to_harvest)
    }

    return explained_variances_dict
