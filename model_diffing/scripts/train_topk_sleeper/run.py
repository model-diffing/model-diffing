from pathlib import Path

import fire
import torch
import yaml

from model_diffing.dataloader.data import build_dataloader_BMLD, dataset_total_sequences
from model_diffing.log import logger
from model_diffing.models.crosscoder import build_topk_crosscoder
from model_diffing.scripts.llms import build_llm_lora
from model_diffing.scripts.train_topk_sleeper.config import TopKExperimentConfig
from model_diffing.scripts.train_topk_sleeper.trainer import TopKTrainer
from model_diffing.utils import build_wandb_run, get_device

def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    device = get_device()

    llm = build_llm_lora(cfg.llm.base_model_repo, cfg.llm.lora_model_repo)

    assert "include_sleeper_data" in cfg.data.sequence_iterator.kwargs

    def dataloader_builder():
        cfg.data.sequence_iterator.kwargs["validation"] = False
        return build_dataloader_BMLD(cfg.data, [llm], cfg.cache_dir)
    def validation_dataloader_builder():
        cfg.data.sequence_iterator.kwargs["validation"] = True
        return build_dataloader_BMLD(cfg.data, [llm], cfg.cache_dir)

    cfg.data.sequence_iterator.kwargs["validation"] = False
    total_sequences, sequence_length = dataset_total_sequences(cfg.data, [llm], cfg.cache_dir)
    sequence_batches_per_epoch = total_sequences // cfg.data.activations_harvester.harvest_batch_size
    steps_per_epoch = (sequence_batches_per_epoch * sequence_length * cfg.data.activations_harvester.harvest_batch_size) // cfg.data.cc_training_batch_size

    crosscoder = build_topk_crosscoder(
        n_layers=len(cfg.data.activations_harvester.layer_indices_to_harvest),
        d_model=llm.cfg.d_model,
        cc_hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        n_models=1,
        k=cfg.crosscoder.k,
    )
    crosscoder = crosscoder.to(device)

    # Load checkpoint if provided
    if cfg.crosscoder.ft_init_checkpt is not None:
        checkpoint_path = cfg.crosscoder.ft_init_checkpt
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        crosscoder.load_state_dict(state_dict)

    wandb_run = build_wandb_run(cfg.wandb, cfg) if cfg.wandb != "disabled" else None

    return TopKTrainer(
        cfg=cfg.train,
        dataloader_builder=dataloader_builder,
        validation_dataloader_builder=validation_dataloader_builder,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        steps_per_epoch=steps_per_epoch,
    )


def load_config(config_path: Path) -> TopKExperimentConfig:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = TopKExperimentConfig(**config_dict)
    return config


def main(config_path: str, no_save: bool = False) -> None:
    logger.info("Loading config...")
    config = load_config(Path(config_path))
    if no_save:
        config.train.save_dir = None
    logger.info("Loaded config")
    logger.info(f"Training with {config.model_dump_json()}")
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
