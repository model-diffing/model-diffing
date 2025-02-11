import fire  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_l1_crosscoder.config import L1ExperimentConfig
from model_diffing.scripts.train_l1_crosscoder.trainer import L1CrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device


def build_l1_crosscoder_trainer(cfg: L1ExperimentConfig) -> L1CrosscoderTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    hookpoint_dim = llms[0].cfg.d_model

    dataloader = build_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        hookpoint_dim=hookpoint_dim,
        batch_size=cfg.train.batch_size,
        cache_dir=cfg.cache_dir,
        device=device,
    )

    _, n_models, n_hookpoints, d_model = dataloader.batch_shape_BMPD()

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        hidden_activation=ReLUActivation(),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return L1CrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        experiment_name=cfg.experiment_name,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_l1_crosscoder_trainer, L1ExperimentConfig))
