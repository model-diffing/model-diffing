from collections.abc import Callable
from dataclasses import dataclass

import torch as t
from einops import einsum, rearrange
from torch import nn

from model_diffing.utils import l2_norm

"""
Dimensions:
- M: number of models
- B: batch size
- L: number of layers
- D: Subject model dimension
- H: Autoencoder hidden dimension
"""

# hacky but useful for debugging
t.Tensor.d = lambda self: f"{self.shape}, dtype={self.dtype}, device={self.device}"  # type: ignore


class AcausalCrosscoder(nn.Module):
    """crosscoder that autoencodes activations of a subset of a model's layers"""

    def __init__(
        self,
        n_models: int,
        n_layers: int,
        d_model: int,
        hidden_dim: int,
        dec_init_norm: float,
        hidden_activation: Callable[[t.Tensor], t.Tensor],
    ):
        super().__init__()
        self.activations_shape_MLD = (n_models, n_layers, d_model)
        self.hidden_dim = hidden_dim

        self.W_enc_MLDH = nn.Parameter(t.randn((n_models, n_layers, d_model, hidden_dim)))
        self.b_enc_H = nn.Parameter(t.zeros((hidden_dim,)))

        self.W_dec_HMLD = nn.Parameter(
            rearrange(  # "transpose" of the encoder weights
                self.W_enc_MLDH.clone(),
                "model layer d_model hidden -> hidden model layer d_model",
            )
        )
        self.b_dec_MLD = nn.Parameter(t.zeros((n_models, n_layers, d_model)))

        with t.no_grad():
            W_dec_norm_MLD1 = l2_norm(self.W_dec_HMLD, dim=-1, keepdim=True)
            self.W_dec_HMLD.div_(W_dec_norm_MLD1)
            self.W_dec_HMLD.mul_(dec_init_norm)

        self.hidden_activation = hidden_activation

    def encode(self, activation_BMLD: t.Tensor) -> t.Tensor:
        hidden_BH = einsum(
            activation_BMLD,
            self.W_enc_MLDH,
            "batch model layer d_model, model layer d_model hidden -> batch hidden",
        )
        hidden_BH = hidden_BH + self.b_enc_H
        return self.hidden_activation(hidden_BH)

    def decode(self, hidden_BH: t.Tensor) -> t.Tensor:
        activation_BMLD = einsum(
            hidden_BH,
            self.W_dec_HMLD,
            "batch hidden, hidden model layer d_model -> batch model layer d_model",
        )
        activation_BMLD += self.b_dec_MLD
        return activation_BMLD

    @dataclass
    class TrainResult:
        hidden_BH: t.Tensor
        reconstructed_acts_BMLD: t.Tensor

    def forward_train(
        self,
        activation_BMLD: t.Tensor,
    ) -> TrainResult:
        """returns the activations, the hidden states, and the reconstructed activations"""
        assert activation_BMLD.shape[1:] == self.activations_shape_MLD, (
            f"activation_BMLD.shape[1:] {activation_BMLD.shape[1:]} != "
            f"self.activations_shape_MLD {self.activations_shape_MLD}"
        )
        hidden_BH = self.encode(activation_BMLD)
        reconstructed_BMLD = self.decode(hidden_BH)
        assert reconstructed_BMLD.shape == activation_BMLD.shape
        assert len(reconstructed_BMLD.shape) == 4
        return self.TrainResult(
            hidden_BH=hidden_BH,
            reconstructed_acts_BMLD=reconstructed_BMLD,
        )

    def forward(self, activation_BMLD: t.Tensor) -> t.Tensor:
        hidden_BH = self.encode(activation_BMLD)
        return self.decode(hidden_BH)


class TopkActivation(nn.Module):
    def __init__(self, k: int):
        self.k = k

    def forward(self, hidden_preactivation_BH: t.Tensor) -> t.Tensor:
        _topk_values_BH, topk_indices_BH = hidden_preactivation_BH.topk(self.k, dim=-1)
        hidden_BH = t.zeros_like(hidden_preactivation_BH)
        hidden_BH.scatter_(-1, topk_indices_BH, _topk_values_BH)
        return hidden_BH


def build_l1_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    dec_init_norm: float = 0.1,
) -> AcausalCrosscoder:
    return AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=t.relu,
    )


def build_topk_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    k: int,
    dec_init_norm: float = 0.1,
) -> AcausalCrosscoder:
    return AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=TopkActivation(k),
    )
