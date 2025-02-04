from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import torch as t
from einops import einsum, rearrange, reduce
from torch import nn

from model_diffing.models.activations import ACTIVATIONS
from model_diffing.utils import SaveableModule, l2_norm

"""
Dimensions:
- M: number of models
- B: batch size
- L: number of layers
- D: Subject model dimension
- H: Autoencoder hidden dimension
"""

TActivation = TypeVar("TActivation", bound=SaveableModule)


class AcausalCrosscoder(SaveableModule, Generic[TActivation]):
    def __init__(
        self,
        n_models: int,
        n_layers: int,
        d_model: int,
        hidden_dim: int,
        dec_init_norm: float,
        hidden_activation: TActivation,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_layers = n_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

        W_base_HMLD = t.randn((hidden_dim, n_models, n_layers, d_model))

        self.W_dec_HMLD = nn.Parameter(W_base_HMLD.clone())

        with t.no_grad():
            W_dec_norm_MLD1 = reduce(self.W_dec_HMLD, "hidden model layer d_model -> hidden model layer 1", l2_norm)
            self.W_dec_HMLD.div_(W_dec_norm_MLD1)
            self.W_dec_HMLD.mul_(dec_init_norm)

            self.W_enc_MLDH = nn.Parameter(
                rearrange(  # "transpose" of the encoder weights
                    W_base_HMLD,
                    "hidden model layer d_model -> model layer d_model hidden",
                )
            )

        self.b_dec_MLD = nn.Parameter(t.zeros((n_models, n_layers, d_model)))
        self.b_enc_H = nn.Parameter(t.zeros((hidden_dim,)))

        # Initialize the buffer with a zero tensor of the correct shape
        self.register_buffer("folded_scaling_factors_ML", t.zeros((n_models, n_layers), dtype=t.float32))
        self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    def encode(self, activation_BMLD: t.Tensor) -> t.Tensor:
        pre_bias_BH = einsum(
            activation_BMLD,
            self.W_enc_MLDH,
            "batch model layer d_model, model layer d_model hidden -> batch hidden",
        )
        pre_activation_BH = pre_bias_BH + self.b_enc_H
        return self.hidden_activation(pre_activation_BH)

    def decode(self, hidden_BH: t.Tensor) -> t.Tensor:
        pre_bias_BMLD = einsum(
            hidden_BH,
            self.W_dec_HMLD,
            "batch hidden, hidden model layer d_model -> batch model layer d_model",
        )
        return pre_bias_BMLD + self.b_dec_MLD

    @dataclass
    class TrainResult:
        hidden_BH: t.Tensor
        reconstructed_acts_BMLD: t.Tensor

    def forward_train(
        self,
        activation_BMLD: t.Tensor,
    ) -> TrainResult:
        """returns the activations, the hidden states, and the reconstructed activations"""
        self._validate_acts_shape(activation_BMLD)

        hidden_BH = self.encode(activation_BMLD)
        reconstructed_BMLD = self.decode(hidden_BH)

        if reconstructed_BMLD.shape != activation_BMLD.shape:
            raise ValueError(
                f"reconstructed_BMLD.shape {reconstructed_BMLD.shape} != activation_BMLD.shape {activation_BMLD.shape}"
            )

        return self.TrainResult(
            hidden_BH=hidden_BH,
            reconstructed_acts_BMLD=reconstructed_BMLD,
        )

    def _validate_acts_shape(self, activation_BMLD: t.Tensor) -> None:
        expected_shape = (self.n_models, self.n_layers, self.d_model)
        if activation_BMLD.shape[1:] != expected_shape:
            raise ValueError(
                f"activation_BMLD.shape[1:] {activation_BMLD.shape[1:]} != expected_shape {expected_shape}"
            )

    def forward(self, activation_BMLD: t.Tensor) -> t.Tensor:
        hidden_BH = self.encode(activation_BMLD)
        return self.decode(hidden_BH)

    @contextmanager
    def temporarily_fold_activation_scaling(self, scaling_factors_ML: t.Tensor):
        """Temporarily fold scaling factors into weights."""
        self.fold_activation_scaling_into_weights_(scaling_factors_ML)
        yield
        _ = self.unfold_activation_scaling_from_weights_()

    def fold_activation_scaling_into_weights_(self, activation_scaling_factors_ML: t.Tensor) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the model can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        self._validate_scaling_factors(activation_scaling_factors_ML)
        activation_scaling_factors_ML = activation_scaling_factors_ML.to(self.W_enc_MLDH.device)
        self._scale_weights(activation_scaling_factors_ML)
        # set buffer to prevent double-folding

        self.folded_scaling_factors_ML = activation_scaling_factors_ML
        self.is_folded = t.tensor(True, dtype=t.bool)

    def unfold_activation_scaling_from_weights_(self) -> t.Tensor:
        if not self.is_folded.item():
            raise ValueError("No folded scaling factors found")

        folded_scaling_factors_ML = cast(t.Tensor, self.folded_scaling_factors_ML).clone()
        # Clear the buffer before operations to prevent double-unfolding

        self.folded_scaling_factors_ML = None
        self.is_folded = t.tensor(False, dtype=t.bool)

        self._scale_weights(1 / folded_scaling_factors_ML)

        return folded_scaling_factors_ML

    @t.no_grad()
    def _scale_weights(self, scaling_factors_ML: t.Tensor) -> None:
        self.W_enc_MLDH.mul_(scaling_factors_ML[..., None, None])
        self.W_dec_HMLD.div_(scaling_factors_ML[..., None])
        self.b_dec_MLD.div_(scaling_factors_ML[..., None])

    def _validate_scaling_factors(self, scaling_factors_ML: t.Tensor) -> None:
        if t.any(scaling_factors_ML == 0):
            raise ValueError("Scaling factors contain zeros, which would lead to division by zero")
        if t.any(t.isnan(scaling_factors_ML)) or t.any(t.isinf(scaling_factors_ML)):
            raise ValueError("Scaling factors contain NaN or Inf values")

        expected_shape = (self.n_models, self.n_layers)
        if scaling_factors_ML.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {scaling_factors_ML.shape}")

    def _dump_cfg(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "hidden_activation_classname": self.hidden_activation.__class__.__name__,
            "hidden_activation_cfg": self.hidden_activation._dump_cfg(),
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "AcausalCrosscoder[TActivation]":
        hidden_activation_cfg = cfg["hidden_activation_cfg"]
        hidden_activation_classname = cfg["hidden_activation_classname"]

        hidden_activation = ACTIVATIONS[hidden_activation_classname]._from_cfg(hidden_activation_cfg)

        cfg = {
            "n_models": cfg["n_models"],
            "n_layers": cfg["n_layers"],
            "d_model": cfg["d_model"],
            "hidden_dim": cfg["hidden_dim"],
            "dec_init_norm": 0,  # dec_init_norm doesn't matter here as we're loading weights
            "hidden_activation": hidden_activation,
        }

        return cls(**cfg)

    def with_decoder_unit_norm(self) -> "AcausalCrosscoder[TActivation]":
        """
        returns a copy of the model with the weights rescaled such that the decoder norm of each feature is one,
        but the model makes the same predictions.
        """
        W_dec_l2_norms_H = reduce(self.W_dec_HMLD, "hidden model layer dim -> hidden", l2_norm)

        cc = AcausalCrosscoder(
            n_models=self.n_models,
            n_layers=self.n_layers,
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            dec_init_norm=0,  # this shouldn't matter
            hidden_activation=self.hidden_activation,
        )

        with t.no_grad():
            cc.W_enc_MLDH.copy_(self.W_enc_MLDH * W_dec_l2_norms_H)
            cc.b_enc_H.copy_(self.b_enc_H * W_dec_l2_norms_H)
            cc.W_dec_HMLD.copy_(self.W_dec_HMLD / W_dec_l2_norms_H[..., None, None, None])
            # no alteration needed for self.b_dec_MLD

        return cc
