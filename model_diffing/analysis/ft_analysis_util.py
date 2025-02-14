import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Any
from pathlib import Path
import yaml

from model_diffing.models.crosscoder import AcausalCrosscoder, build_topk_crosscoder
from model_diffing.scripts.train_topk_sleeper.config import TrainConfig


def load_crosscoder(path: str, epoch: int, device: torch.device) -> AcausalCrosscoder:
    # load crosscoder model (with folded scaling factors)
    print(path)
    state_dict = torch.load(Path(path) / f"model_epoch_{epoch}.pt", map_location=device)
    crosscoder = build_topk_crosscoder( # Update this to use yaml
        n_layers=4,
        d_model=768,
        cc_hidden_dim=1536,
        dec_init_norm=0.1,
        n_models=1,
        k=20,
    ).to(device)

    unit_scaling_factors_ML = torch.ones((1, 4), device=device)
    crosscoder.folded_scaling_factors_ML = unit_scaling_factors_ML
    crosscoder.load_state_dict(state_dict)
    return crosscoder


def compute_cosine_similarities(features_1: torch.Tensor, features_2: torch.Tensor) -> np.ndarray[Any, np.dtype[np.float64]]:
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_sims = []
    for i in range(features_1.shape[0]):
        cosine_sims.append(cos(features_1[i], features_2[i]).to('cpu').detach().numpy())
    return np.array(cosine_sims)
