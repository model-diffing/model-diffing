from collections.abc import Iterator
from itertools import islice

import torch
from einops import reduce
from tqdm import tqdm

from model_diffing.utils import l2_norm, multi_reduce


@torch.no_grad()
def estimate_norm_scaling_factor_ML(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    d_model: int,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    mean_norms_ML = _estimate_mean_norms_ML(dataloader_BMLD, device, n_batches_for_norm_estimate)
    scaling_factors_ML = torch.sqrt(torch.tensor(d_model)) / mean_norms_ML
    return scaling_factors_ML


@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_ML(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BMLD in tqdm(
        islice(dataloader_BMLD, n_batches_for_norm_estimate),
        desc="Estimating norm scaling factor",
        total=n_batches_for_norm_estimate,
    ):
        batch_BMLD = batch_BMLD.to(device)
        norms_means_ML = multi_reduce(
            batch_BMLD, "batch model layer d_model", [("d_model", l2_norm), ("batch", torch.mean)]
        )
        norm_samples.append(norms_means_ML)

    norm_samples_NML = torch.stack(norm_samples, dim=0)
    mean_norms_ML = reduce(norm_samples_NML, "batch model layer -> model layer", torch.mean)
    return mean_norms_ML


@torch.no_grad()
def collect_norms(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BMLD in tqdm(
        islice(dataloader_BMLD, n_batches),
        desc="Collecting norms",
        total=n_batches,
    ):
        batch_BMLD = batch_BMLD.to(device)
        norms_BML = reduce(batch_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
        norm_samples.append(norms_BML)

    norm_samples_NML = torch.cat(norm_samples, dim=0)
    return norm_samples_NML
