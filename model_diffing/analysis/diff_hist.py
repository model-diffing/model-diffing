"""
usage: diff_hist.py path/to/config.py path/to/checkpoints.pt
output: .png file

currently set to work with 2 models 1 layer
"""
import yaml
from pathlib import Path
from typing import cast
import fire
import torch
import wandb
import yaml

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


from model_diffing.models.crosscoder import build_l1_crosscoder
from model_diffing.scripts.train_l1_crosscoder.config import Config


from model_diffing.log import logger

from model_diffing.analysis import metrics
torch.set_grad_enabled(False)

#device = get_device()

### fprefer working with matplotlib 
def plot_relative_norms(vectors_a: torch.Tensor, vectors_b: torch.Tensor, title: str | None = None) -> None:
    """Plot histogram of relative norms (norm_b / (norm_a + norm_b)) using Matplotlib.

    Args:
        vectors_a: Tensor of vectors from the first set
        vectors_b: Tensor of vectors from the second set
        title: Optional title for the plot
    """
    # Compute relative norms
    relative_norms = metrics.compute_relative_norms(vectors_a, vectors_b)
    relative_norms_np = relative_norms.detach().cpu().numpy()

    # Create histogram
    plt.figure(figsize=(8, 6))
    plt.hist(relative_norms_np, bins=200, color='blue', alpha=0.7, edgecolor='black')

    # Add labels and title
    plt.xlabel("Relative norm")
    plt.ylabel("Number of Latents")
    plt.title(title if title else "Histogram of Relative Norms")

    # Customize x-axis
    plt.xlim(0, 1)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"])

    # Save the plot as PNG
    output_file = "relative_norms_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    #print(f"Plot saved as {output_file}")


def build_hist(cfg: Config , checkpoint_path: str):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # loads a crosscoder 
    crosscoder = build_l1_crosscoder(
        n_layers= len(cfg.layer_indices_to_harvest),
        d_model= cfg.llms[0].d_model,
        cc_hidden_dim= cfg.crosscoder.hidden_dim,
        dec_init_norm= cfg.crosscoder.dec_init_norm,
        n_models=len(cfg.llms),
    )
    
    crosscoder.load_state_dict(state_dict)

    #get decoder weights
    W_dec_HMLD = crosscoder.W_dec_HMLD

    H,M,L,D = W_dec_HMLD.shape 

    W_dec_HMLD = W_dec_HMLD.permute(1, 0, 2, 3)
    # pick a specific layer to look at
    layer = 0
    W_dec_HMLD = W_dec_HMLD[:,:,layer,]

    # want weights DH
    base_model = W_dec_HMLD[0].T
    ft_model = W_dec_HMLD[1].T

    relative_norms = metrics.compute_relative_norms(base_model, ft_model)

 
    plot_relative_norms(base_model, ft_model, title="Relative Norms Histogram") 

    return crosscoder


def load_config(config_path: Path) -> Config:
    """Load the config from a YAML file into a Pydantic model."""
   
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config


def main(config_path: str, checkpoint_path: str) -> None:
    logger.info("Loading config...")
    config = load_config(Path(config_path))
    trainer = build_hist(config, checkpoint_path)


if __name__ == "__main__":
    logger.info("Starting...")
    
    fire.Fire(main)
