# %%
from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from model_diffing.analysis import metrics
from model_diffing.models import crosscoder

if __name__ != "__main__":
    print(
        "This script is not meant to be imported, it disables torch"
        "gradients which will likely have unintended side effects"
    )
    exit()

torch.set_grad_enabled(False)
# %%

# for example:
# path = ".checkpoints/l1_crosscoder_pythia_160M_layer_3/model_epoch_5000.pt"
diff = Path.cwd().parent.parent.parent
path = diff / ".checkpoints/l1_crosscoder_pythia_160M_3_layers/model_epoch_9000.pt"
# %%
path.exists()
# %%

state_dict = torch.load(path)
# %%
W_dec_HMLD = state_dict["W_dec_HMLD"]
W_enc_MLDH = state_dict["W_enc_MLDH"]
b_dec_MLD = state_dict["b_dec_MLD"]
b_enc_H = state_dict["b_enc_H"]

H, M, L, D = W_dec_HMLD.shape

print(f"{(H, M, L, D)=}")

# %%

cc = crosscoder.AcausalCrosscoder(
    n_models=M,
    n_layers=L,
    d_model=D,
    hidden_dim=H,
    dec_init_norm=0,  # this shouldn't matter
    hidden_activation=torch.relu,
)

cc.load_state_dict(state_dict)

# %%

W_dec_HMLD = cc.W_dec_HMLD

# %%

import numpy as np

feature_norms_HML = W_dec_HMLD.norm(dim=-1)

# %%


# %%

# # visualize as grid of intensities
# fig = px.imshow(feature_norms_ML.T, color_continuous_scale="Viridis")
# # open in browser, as this interactive
# fig.show(renderer="browser")
# that didn't work, plotly doesn't support interactive plots in vscode notebooks
# %%
import matplotlib.pyplot as plt

feature_idx = np.random.randint(0, H)
feature_norms_ML = feature_norms_HML[feature_idx]

plt.imshow(feature_norms_ML.T, cmap="viridis")
plt.colorbar()
plt.show()

# %%

feature_diffs_L = feature_norms_ML[1] - feature_norms_ML[0]

# feature_diffs_L

# %%
from typing import Any

from einops import rearrange
from sklearn.decomposition import NMF


def NMF_components_CML(
    feature_norms_FML: np.ndarray[Any, Any],
    n_components: int = 4,
    random_state: int = 0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Apply Non-negative Matrix Factorization to feature norms across models and layers.

    Args:
        feature_norms_FML (np.ndarray): Matrix of shape (n_features, n_models, n_layers).
        n_components (int): Number of NMF components (default: 4).
        random_state (int): Random seed for reproducibility (default: 0).

    Returns:
        tuple: (components, feature_loadings)
            - components: Shape (n_components, n_models * n_layers), indicating component activations.
            - feature_loadings: Shape (n_features, n_components), indicating feature contributions.
    """
    feature_norms_FMl = rearrange(feature_norms_FML, "h m l -> h (m l)")
    model = NMF(
        n_components=n_components,
        init="random",
        random_state=random_state,
        max_iter=1000,
    )

    feature_loadings_FMl = model.fit_transform(feature_norms_FMl)
    feature_loadings_FML = rearrange(feature_loadings_FMl, "f (m l) -> f m l", m=M, l=L)

    components_CMl = model.components_
    components_CML = rearrange(components_CMl, "c (m l) -> c m l", m=M, l=L)
    return components_CML, feature_loadings_FML


# %%

components_CML, feature_loadings = NMF_components_CML(feature_norms_HML.detach().numpy(), n_components=2)
C = components_CML.shape[0]
assert M == 2
# subplots = make_subplots(rows=C, cols=M)
subplots = make_subplots(cols=C, rows=1)

for component_idx in range(C):
    component_ML = components_CML[component_idx]  # shape: (M, L)

    # First column: visualize the [M, L] component matrix as a heatmap
    heatmap_trace = go.Heatmap(z=component_ML.T, colorscale="Viridis", colorbar=dict(title="Activation"))
    subplots.add_trace(heatmap_trace, row=1, col=component_idx + 1)

    # Second column: display the component's loadings across features (H)
    # scatter_trace = go.Scatter(
    #     x=np.arange(feature_loadings.shape[0]),
    #     y=feature_loadings[:, component_idx],
    #     mode="lines",
    #     name=f"Component {component_idx}",
    # )
    # subplots.add_trace(scatter_trace, row=component_idx + 1, col=2)

subplots.update_layout(
    height=300 * C, width=1000, title_text="Visualizing NMF Components with Heatmap (left) and Feature Loadings (right)"
)
subplots.show(renderer="browser")

# %%

# Create subplots with L rows and 2 columns
fig = make_subplots(
    rows=L,
    cols=2,
    vertical_spacing=0.1,
)

for layer_idx in range(L):
    # Extract weights for current layer
    W_dec_a_HD = W_dec_HMLD[:, 0, layer_idx]
    W_dec_b_HD = W_dec_HMLD[:, 1, layer_idx]

    # Compute relative norms
    relative_norms_H = metrics.compute_relative_norms_N(W_dec_a_HD, W_dec_b_HD)

    # Create histogram trace directly
    hist_trace = go.Histogram(x=relative_norms_H, nbinsx=200, name=f"Layer {layer_idx + 1}", showlegend=False)
    fig.add_trace(hist_trace, row=layer_idx + 1, col=1)

    # Compute cosine similarities for shared features
    shared_latent_mask = metrics.get_shared_latent_mask(relative_norms_H)
    cosine_sims_H = metrics.compute_cosine_similarities_N(W_dec_a_HD, W_dec_b_HD)
    shared_features_cosine_sims_H = cosine_sims_H[shared_latent_mask]

    # Create histogram trace for cosine similarities
    cosine_hist = go.Histogram(
        x=shared_features_cosine_sims_H, nbinsx=50, name=f"Layer {layer_idx + 1}", showlegend=False
    )
    fig.add_trace(cosine_hist, row=layer_idx + 1, col=2)

# Update layout
fig.update_layout(
    height=300 * L,  # Adjust height based on number of layers
    width=1000,
    showlegend=False,
    title_text="Layer-wise Analysis of Weights",
)

# Update x-axes
for i in range(L):
    fig.update_xaxes(title_text="Relative Norm", range=[0, 1], row=i + 1, col=1)
    fig.update_xaxes(title_text="Cosine Similarity", range=[-1, 1], row=i + 1, col=2)
    fig.update_yaxes(title_text="Count", row=i + 1, col=1)
    fig.update_yaxes(title_text="Count", row=i + 1, col=2)

fig.show(renderer="browser")

# %%
