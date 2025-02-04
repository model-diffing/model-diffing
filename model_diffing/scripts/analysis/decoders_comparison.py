# %%
from typing import Any

import torch

from model_diffing.analysis import metrics, visualization
from model_diffing.models import crosscoder
from model_diffing.models.activations.jumprelu import JumpReLUActivation

torch.set_grad_enabled(False)

# %%

# for example:
# path = ".checkpoints/l1_crosscoder_pythia_160M_layer_3/model_epoch_5000.pt"

path = "../../../.checkpoints/JumpReLU_SAE_pythia_160M_crossmodel_nlayers_32k/model_step_27500.pt"

state_dict = torch.load(path, map_location="mps")
state_dict["is_folded"] = torch.tensor(True, dtype=torch.bool)

# %%
state_dict["folded_scaling_factors_ML"]
# %%

cc = crosscoder.AcausalCrosscoder(
    n_models=2,
    n_layers=4,
    d_model=768,
    hidden_dim=32_768,
    dec_init_norm=0.1,
    hidden_activation=JumpReLUActivation(size=32_768, bandwidth=0.1, threshold_init=0.1),
)
cc.load_state_dict(state_dict)


def vis(cc: crosscoder.AcausalCrosscoder[Any]):
    for layer_idx in range(cc.W_dec_HMLD.shape[2]):
        W_dec_a_HD = cc.W_dec_HMLD[:, 0, layer_idx]
        W_dec_b_HD = cc.W_dec_HMLD[:, 1, layer_idx]

        relative_norms = metrics.compute_relative_norms_N(W_dec_a_HD, W_dec_b_HD)
        fig = visualization.relative_norms_hist(relative_norms)
        fig.show()

        shared_latent_mask = metrics.get_shared_latent_mask(relative_norms)
        cosine_sims = metrics.compute_cosine_similarities_N(W_dec_a_HD, W_dec_b_HD)
        shared_features_cosine_sims = cosine_sims[shared_latent_mask]
        fig = visualization.plot_cosine_sim(shared_features_cosine_sims)
        fig.show()


# %%


normed_cc = cc.with_decoder_unit_norm()
vis(normed_cc)
