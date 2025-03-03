import fire  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.models.acausal_crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.scripts.train_topk_crosscoder.config import TopKExperimentConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import TopKTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device


def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )
    
    dataloader = build_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    n_models = len(llms)
    n_hookpoints = len(cfg.hookpoints)

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
        hidden_activation=TopkActivation(k=cfg.crosscoder.k),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return TopKTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKExperimentConfig))
