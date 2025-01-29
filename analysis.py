from typing import Any

import einops
import numpy as np


def random_2d(elements: int):
    return np.random.rand(elements, 2)


def random_projection_into_3d_23():
    return np.random.rand(2, 3)


# data_N2 = random_2d(100)
# # px.scatter(*data_N2.T).show()


# data_N3 = data_N2 @ random_projection_into_3d_23()
# assert data_N3.shape == (100, 3)
# x, y, z = data_N3.T


def normalise(data_NF: np.ndarray[Any, Any]):
    mean_N1 = einops.reduce(data_NF, "n f -> n 1", np.mean)
    std_N1 = einops.reduce(data_NF, "n f -> n 1", np.std)
    data_standardized_NF = (data_NF - mean_N1) / std_N1
    assert np.isclose(data_standardized_NF.mean(), 0)
    assert np.isclose(data_standardized_NF.std(), 1)
    return data_standardized_NF


def PCA(data_NF: np.ndarray[Any, Any]):
    N, F = data_NF.shape

    data_standardized_NF = normalise(data_NF)

    cov_FF = einops.einsum(data_standardized_NF, data_standardized_NF, "n1 f1, n2 f2 -> f1 f2")

    eig_vals_F, eig_vecs_FF = np.linalg.eig(cov_FF)
    assert eig_vals_F.shape == (F,)
    assert eig_vecs_FF.shape == (F, F)

    sorted_vals_F = eig_vals_F.argsort(0)[::-1]

    pcs_FF = eig_vecs_FF[sorted_vals_F]
    assert pcs_FF.shape == (F, F)

    eig_vals_F = eig_vals_F[sorted_vals_F]
    assert eig_vals_F.shape == (F,)

    return pcs_FF, eig_vals_F


data_NF = np.random.rand(100, 3)
random_dir = np.random.rand(3)
random_dir /= np.linalg.norm(random_dir, 2)
# data_NF = np.einsum('nf,jjjjjj')

x, y, z = data_NF.T

pcs, mags = PCA(data_NF)
print(pcs)
print(mags)

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers"))


def line(vec_F: np.ndarray[Any, Any]):
    assert vec_F.shape == (3,)
    x, y, z = vec_F
    return go.Scatter3d(
        x=[0, x],
        y=[0, y],
        z=[0, z],
        mode="lines",
        name="Arrow",  # Optional
        line=dict(color="red", width=4),  # Customize the arrow appearance
    )


fig.add_trace(line(pcs[0]))
# fig.add_trace(line(pcs[1]))
# fig.add_trace(line(pcs[2]))

fig.show()

# # %%

# from datasets import load_dataset

# COMMON_CORPUS_HF_DATASET = "PleIAs/common_corpus"

# text_dataset = load_dataset(COMMON_CORPUS_HF_DATASET, streaming=True, split="train")

# import sys
# for example in text_dataset:
#     example_memory_usage = sys.getsizeof(example["text"])
#     print(f"Example memory usage: {example_memory_usage} bytes")


# # %%

# # %%
# n_models = 2
# n_layers = 3
# d_model = 768

# activation_size = n_models * n_layers * d_model

# batch_size = 4096

# in_size = activation_size * batch_size

# sae_hidden = 16_384

# W_enc_size = sae_hidden * in_size
# W_dec_size = in_size * sae_hidden

# activations_size = (
#     # hidden:
#     sae_hidden * batch_size
#     +
#     # output:
#     in_size * batch_size
# )

# total_forward_pass_memory = (W_enc_size + W_dec_size + activations_size) * 4  # bytes, float32

# total_forward_pass_memory_GB = total_forward_pass_memory / 10**9

# print(f"Total forward pass memory: {total_forward_pass_memory_GB:.2f} GB")
# # %%

# from einops import rearrange
# import einops
# import torch

# D = 768
# expected_activation_norm = 21
# B = 100

# act_vecs_BD = torch.rand(B, D) * expected_activation_norm


# diffs_BBD = rearrange(act_vecs_BD, "b d -> 1 b d") - rearrange(act_vecs_BD, "b d -> b 1 d")
# upper_trui_mask = torch.triu(torch.ones((B, B)), diagonal=1).bool()
# diffs_ND = diffs_BBD[upper_trui_mask]
# assert len(diffs_ND.shape) == 2
# assert diffs_ND.shape[1] == D
# diffs_norms_N = diffs_ND.norm(p=2, dim=-1)
# print(diffs_norms_N.mean())

# # %%

# import einops

# x = torch.randn((3, 4))
# einops.reduce(x, "a b -> a 1", torch.mean), einops.reduce(x, "a b -> a", torch.mean)

# # %%
# import torch

# torch.norm(torch.tensor([0, 0.1, 0.1]), p=0)
# # %%
