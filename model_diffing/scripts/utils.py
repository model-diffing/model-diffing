from collections.abc import Callable, Iterator
from itertools import islice

import torch
import wandb
from einops import reduce
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

from model_diffing.scripts.config_common import AdamDecayTo0LearningRateConfig, BaseExperimentConfig
from model_diffing.utils import l2_norm


def build_wandb_run(config: BaseExperimentConfig) -> Run | None:
    return wandb.init(
        name=config.experiment_name,
        project="model-diffing",
        entity="mars-model-diffing",
        config=config.model_dump(),
    )


def build_optimizer(cfg: AdamDecayTo0LearningRateConfig, params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
    initial_lr = cfg.initial_learning_rate
    optimizer = torch.optim.Adam(params, lr=initial_lr)
    return optimizer


def build_lr_scheduler(cfg: AdamDecayTo0LearningRateConfig, num_steps: int) -> Callable[[int], float]:
    def _lr_scheduler(step: int) -> float:
        if step < cfg.warmup_pct * num_steps:
            return cfg.initial_learning_rate * (step / (cfg.warmup_pct * num_steps))

        pct_until_finished = 1 - (step / num_steps)
        if pct_until_finished < cfg.last_pct_of_steps:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / cfg.last_pct_of_steps
            return cfg.initial_learning_rate * scale

        return cfg.initial_learning_rate

    return _lr_scheduler


@torch.no_grad()
def estimate_norm_scaling_factor_X(
    dataloader_BXD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    d_model = next(dataloader_BXD).shape[-1]
    mean_norms_X = _estimate_mean_norms_X(dataloader_BXD, device, n_batches_for_norm_estimate)
    scaling_factors_X = torch.sqrt(torch.tensor(d_model)) / mean_norms_X
    return scaling_factors_X


@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_X(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BXD in tqdm(
        islice(dataloader_BMLD, n_batches_for_norm_estimate),
        desc="Estimating norm scaling factor",
        total=n_batches_for_norm_estimate,
    ):
        batch_BXD = batch_BXD.to(device)
        norms_means_X = l2_norm(batch_BXD, dim=-1).mean(dim=0)
        norm_samples.append(norms_means_X)

    norm_samples_NX = torch.stack(norm_samples, dim=0)
    mean_norms_X = reduce(norm_samples_NX, "n_samples ... -> ...", torch.mean)
    return mean_norms_X


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
