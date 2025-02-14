from pathlib import Path

import fire  # type: ignore
import yaml  # type: ignore
import torch

from model_diffing.dataloader.data import build_dataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import build_topk_crosscoder
from model_diffing.scripts.llms import build_llm_lora
from model_diffing.scripts.train_topk_sleeper.config import TopKExperimentConfig, TrainConfig
from model_diffing.scripts.train_topk_sleeper.trainer import TopKTrainer
from model_diffing.utils import build_wandb_run, get_device

def build_trainer(cfg: TopKExperimentConfig, config_raw: str) -> TopKTrainer:
    device = get_device()

    assert "include_sleeper_data" in cfg.data.sequence_iterator.kwargs

    # TODO is this going to end up loading two copies of the llm into memory?
    cfg.data.sequence_iterator.kwargs["validation"] = False
    dataloader = build_dataloader(cfg.data, cfg.cache_dir, device)
    cfg.data.sequence_iterator.kwargs["validation"] = True
    validation_dataloader = build_dataloader(cfg.data, cfg.cache_dir, device)
    cfg.data.sequence_iterator.kwargs["validation"] = False

    _, n_layers, n_models, d_model = dataloader.batch_shape_BMLD()

    crosscoder = build_topk_crosscoder(
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=n_models,
        k=cfg.crosscoder.k,
    )
    crosscoder = crosscoder.to(device)

    # Load checkpoint if provided
    if cfg.crosscoder.ft_init_checkpt_folder is not None:
        checkpoint_path = cfg.crosscoder.ft_init_checkpt_folder / f"model_epoch_{cfg.crosscoder.ft_init_checkpt_epoch}.pt"
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        unit_scaling_factors_ML = torch.ones((n_models, n_layers), device=device)
        crosscoder.folded_scaling_factors_ML = unit_scaling_factors_ML
        crosscoder.load_state_dict(state_dict)
        norm_scaling_factors_ML = crosscoder.unfold_activation_scaling_from_weights_()
        dataloader.norm_scaling_factors_ML = norm_scaling_factors_ML
    

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return TopKTrainer(
        cfg=cfg.train,
        cfg_raw=config_raw,
        activations_dataloader=dataloader,
        activations_validation_dataloader=validation_dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


def load_config(config_path: Path) -> tuple[TopKExperimentConfig, str]:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_raw = f.read()
        config_dict = yaml.safe_load(config_raw)
    config = TopKExperimentConfig(**config_dict)
    return config, config_raw


def main(config_path: str, debug_no_save: bool = False) -> None:
    logger.info("Loading config...")
    config, config_raw = load_config(Path(config_path))
    if debug_no_save:
        config.train.save_dir = None
    logger.info("Loaded config")
    logger.info(f"Training with {config.model_dump_json()}")
    trainer = build_trainer(config, config_raw)
    trainer.train()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
