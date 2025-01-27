from pathlib import Path

import fire
import wandb

from model_diffing.log import logger

# usage:
# python model_diffing/scripts/upload_model_to_wandb/main.py
#     --model_checkpoint .checkpoints/path/to/checkpoint.pt
#     --previous_run_id flimsy-cow-385 # or whatever the run id is

WANDB_PROJECT = "sleeper-model-diffing"
WANDB_ENTITY = "dmitry2-uiuc"

def main(model_checkpoint: str, previous_run_id: str) -> None:
    logger.info("Loading model checkpoint...")
    model_checkpoint_path = Path(model_checkpoint)
    assert model_checkpoint_path.exists(), f"Model checkpoint {model_checkpoint_path} does not exist."
    logger.info("Loaded model checkpoint")
    with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=previous_run_id, resume="allow") as previous_run:
        artifact = wandb.Artifact(name="model-checkpoint-v1", type="model")
        artifact.add_file(model_checkpoint_path) # TODO also upload the config.yaml file
        previous_run.log_artifact(artifact)

if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
