import os
from collections.abc import Iterator

import fire
import torch
from einops import rearrange
from tqdm import tqdm

from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.dataloader.data import build_sliding_window_dataloader
from model_diffing.log import logger
from model_diffing.models.activations import JumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.train_jan_update_crosscoder.config import JumpReLUConfig
from model_diffing.scripts.train_sliding_window.config import SlidingWindowExperimentConfig
from model_diffing.scripts.train_sliding_window.trainer import BiTokenCCWrapper, SlidingWindowCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, inspect


def build_sliding_window_crosscoder_trainer(cfg: SlidingWindowExperimentConfig) -> SlidingWindowCrosscoderTrainer:
    device = get_device()

    dataloader = build_sliding_window_dataloader(cfg.data, cfg.train.batch_size, cfg.cache_dir, device)
    _, n_tokens, n_layers, n_models, d_model = dataloader.batch_shape()
    assert n_tokens == 2

    if cfg.token_window_size != 2:
        raise ValueError(f"token_window_size must be 2, got {cfg.token_window_size}")

    crosscoder1, crosscoder2 = [
        build_crosscoder(
            n_tokens=window_size,
            n_models=n_models,
            n_layers=n_layers,
            d_model=d_model,
            cc_hidden_dim=cfg.crosscoder.hidden_dim,
            jumprelu=cfg.crosscoder.jumprelu,
            data_loader=dataloader,
            device=device,
        )
        for window_size in [1, 2]
    ]

    crosscoders = BiTokenCCWrapper(crosscoder1, crosscoder2)
    crosscoders.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return SlidingWindowCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoders=crosscoders,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


def build_crosscoder(
    n_tokens: int,
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    data_loader: BaseActivationsDataloader,
    device: torch.device,  # for computing b_enc
) -> AcausalCrosscoder[JumpReLUActivation]:
    cc = AcausalCrosscoder(
        n_tokens=n_tokens,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=0,  # dec_init_norm doesn't matter here as we override weights below
        hidden_activation=JumpReLUActivation(
            size=cc_hidden_dim,
            bandwidth=jumprelu.bandwidth,
            threshold_init=jumprelu.threshold_init,
            backprop_through_input=jumprelu.backprop_through_jumprelu_input,
        ),
    )

    cc.to(device)

    with torch.no_grad():
        # parameters from the jan update doc

        n = float(n_models * n_layers * d_model)  # n is the size of the input space
        m = float(cc_hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HTMLD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_TMLDH.copy_(
            rearrange(cc.W_dec_HTMLD, "hidden token model layer d_model -> token model layer d_model hidden")  #
            * (n / m)
        )

        calibrated_b_enc_H = _compute_b_enc_H(
            data_loader,
            cc.W_enc_TMLDH,
            cc.hidden_activation.log_threshold_H.exp(),
            device,
            n_examples_to_sample=500,
            firing_sparsity=4, # 10_000 / m, NOT 1 / 10_000
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_TMLD.zero_()

    return cc


def _compute_b_enc_H(
    data_loader: BaseActivationsDataloader,
    W_enc_TMLDH: torch.Tensor,
    initial_jumprelu_threshold_H: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int, #  = 100_000,
    firing_sparsity: float, #  = 10_000,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_examples_to_sample} examples")

    pre_bias_NH = _harvest_pre_bias_NH(data_loader, W_enc_TMLDH, device, n_examples_to_sample)

    # find the threshold for each idx H such that 1/10_000 of the examples are above the threshold
    quantile_H = torch.quantile(pre_bias_NH, 1 - 1 / firing_sparsity, dim=0)

    # firing is when the post-bias is above the jumprelu threshold therefore, we subtract
    # the quantile from the initial jumprelu threshold, such the 1/firing_sparsity of the
    # examples are above the threshold
    b_enc_H = initial_jumprelu_threshold_H - quantile_H

    logger.info(f"computed b_enc_H. Sample: {b_enc_H[:10]}. mean: {b_enc_H.mean()}, std: {b_enc_H.std()}")

    return b_enc_H


def _harvest_pre_bias_NH(
    data_loader: BaseActivationsDataloader,
    W_enc_TMLDH: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int,
) -> torch.Tensor:
    batch_size = data_loader.batch_shape()[0]

    remainder = n_examples_to_sample % batch_size
    if remainder != 0:
        logger.warning(
            f"n_examples_to_sample {n_examples_to_sample} must be divisible by the batch "
            f"size {batch_size}. Rounding up to the nearest multiple of batch_size."
        )
        # Round up to the nearest multiple of batch_size:
        n_examples_to_sample = (((n_examples_to_sample - remainder) // batch_size) + 1) * batch_size

        logger.info(f"n_examples_to_sample is now {n_examples_to_sample}")

    num_batches = n_examples_to_sample // batch_size

    activations_iterator_BTMLD = data_loader.get_shuffled_activations_iterator()

    def get_batch_pre_bias_BH() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BTMLD = next(activations_iterator_BTMLD).to(device)
        return torch.einsum("b t m l d, t m l d h -> b h", batch_BTMLD, W_enc_TMLDH)

    first_sample_BH = get_batch_pre_bias_BH()
    hidden_size = first_sample_BH.shape[1]

    pre_bias_buffer_NH = torch.empty(n_examples_to_sample, hidden_size, device=device)
    logger.info(f"pre_bias_buffer_NH: {inspect(pre_bias_buffer_NH)}")

    pre_bias_buffer_NH[:batch_size] = first_sample_BH

    for i in tqdm(
        range(1, num_batches), desc="Harvesting pre-bias"
    ):  # start at 1 because we already sampled the first batch
        batch_pre_bias_BH = get_batch_pre_bias_BH()
        pre_bias_buffer_NH[batch_size * i : batch_size * (i + 1)] = batch_pre_bias_BH

    return pre_bias_buffer_NH


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("Starting...")
    fire.Fire(run_exp(build_sliding_window_crosscoder_trainer, SlidingWindowExperimentConfig))

# if __name__ == "__main__":
#     B = 8
#     T = 2
#     M = 1
#     L = 3
#     D = 16

#     class DL(BaseActivationsDataloader):
#         def get_shuffled_activations_iterator(self) -> Iterator[torch.Tensor]:
#             while True:
#                 yield torch.randn(B, T, M, L, D)

#         def batch_shape(self) -> tuple[int, int, int, int, int]:  # type: ignore
#             return B, T, M, L, D

#         def num_batches(self) -> None:
#             return None

#         def get_norm_scaling_factors_ML(self) -> torch.Tensor:
#             return torch.ones(M, L)

#     cfg = JanUpdateTrainConfig(
#         batch_size=1,
#         optimizer=AdamDecayTo0LearningRateConfig(initial_learning_rate=4e-4),
#         num_steps=1,
#     )

#     activations_dataloader = DL()

#     device = torch.device("mps")

#     cc1, cc2 = [
#         build_crosscoder(
#             n_tokens=n,
#             n_models=1,
#             n_layers=3,
#             d_model=16,
#             cc_hidden_dim=16,
#             jumprelu=JumpReLUConfig(),
#             data_loader=activations_dataloader,
#             device=device,
#         )
#         for n in [1, 2]
#     ]

#     trainer = SlidingWindowCrosscoderTrainer(
#         cfg=cfg,
#         activations_dataloader=activations_dataloader,
#         crosscoders=BiTokenCCWrapper(cc1, cc2),
#         wandb_run=None,
#         device=device,
#         layers_to_harvest=[0],
#         experiment_name="test",
#     )

#     trainer.train()
