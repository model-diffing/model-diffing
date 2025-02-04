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
- D: Subject m dimension
- H: Autoencoder h dimension
"""

TActivation = TypeVar("TActivation", bound=SaveableModule)


class AcausalCrosscoder(SaveableModule, Generic[TActivation]):
    def __init__(
        self,
        n_tokens: int,
        n_models: int,
        n_layers: int,
        d_model: int,
        hidden_dim: int,
        dec_init_norm: float,
        hidden_activation: TActivation,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_models = n_models
        self.n_layers = n_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

        W_base_HTMLD = t.randn((hidden_dim, n_tokens, n_models, n_layers, d_model))

        self.W_dec_HTMLD = nn.Parameter(W_base_HTMLD.clone())

        with t.no_grad():
            W_dec_norm_HTML1 = reduce(self.W_dec_HTMLD, "h t m l d -> h t m l 1", l2_norm)
            self.W_dec_HTMLD.div_(W_dec_norm_HTML1)
            self.W_dec_HTMLD.mul_(dec_init_norm)

            self.W_enc_TMLDH = nn.Parameter(
                rearrange(  # "transpose" of the encoder weights
                    W_base_HTMLD,
                    "h t m l d -> t m l d h",
                )
            )

        self.b_dec_TMLD = nn.Parameter(t.zeros((n_tokens, n_models, n_layers, d_model)))
        self.b_enc_H = nn.Parameter(t.zeros((hidden_dim,)))

        # Initialize the buffer with a zero tensor of the correct shape
        self.register_buffer("folded_scaling_factors_ML", t.zeros((n_models, n_layers), dtype=t.float32))
        self.register_buffer("is_folded", t.tensor(False, dtype=t.bool))

    def encode(self, activation_BTMLD: t.Tensor) -> t.Tensor:
        pre_bias_BH = einsum(
            activation_BTMLD,
            self.W_enc_TMLDH,
            "b t m l d, t m l d h -> b h",
        )
        pre_activation_BH = pre_bias_BH + self.b_enc_H
        return self.hidden_activation(pre_activation_BH)

    def decode(self, hidden_BH: t.Tensor) -> t.Tensor:
        pre_bias_BTMLD = einsum(hidden_BH, self.W_dec_HTMLD, "b h, h t m l d -> b t m l d")
        return pre_bias_BTMLD + self.b_dec_TMLD

    @dataclass
    class TrainResult:
        hidden_BH: t.Tensor
        reconstructed_acts_BTMLD: t.Tensor

    def forward_train(
        self,
        activation_BTMLD: t.Tensor,
    ) -> TrainResult:
        """returns the activations, the h states, and the reconstructed activations"""
        self._validate_acts_shape(activation_BTMLD)

        hidden_BH = self.encode(activation_BTMLD)
        reconstructed_BTMLD = self.decode(hidden_BH)

        if reconstructed_BTMLD.shape != activation_BTMLD.shape:
            raise ValueError(
                f"reconstructed_BTMLD.shape {reconstructed_BTMLD.shape} != activation_BTMLD.shape {activation_BTMLD.shape}"
            )

        return self.TrainResult(
            hidden_BH=hidden_BH,
            reconstructed_acts_BTMLD=reconstructed_BTMLD,
        )

    def _validate_acts_shape(self, activation_BTMLD: t.Tensor) -> None:
        expected_shape_TMLD = (self.n_tokens, self.n_models, self.n_layers, self.d_model)
        if activation_BTMLD.shape[1:] != expected_shape_TMLD:
            raise ValueError(
                f"activation_BTMLD.shape[1:] {activation_BTMLD.shape[1:]} != expected_shape {expected_shape_TMLD}"
            )

    def forward(self, activation_BTMLD: t.Tensor) -> t.Tensor:
        hidden_BH = self.encode(activation_BTMLD)
        return self.decode(hidden_BH)

    @contextmanager
    def temporarily_fold_activation_scaling(self, scaling_factors_ML: t.Tensor):
        """Temporarily fold scaling factors into weights."""
        self.fold_activation_scaling_into_weights_(scaling_factors_ML)
        yield
        _ = self.unfold_activation_scaling_from_weights_()

    def fold_activation_scaling_into_weights_(self, activation_scaling_factors_ML: t.Tensor) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the m can be run on raw llm activations."""
        if self.is_folded.item():
            raise ValueError("Scaling factors already folded into weights")

        self._validate_scaling_factors(activation_scaling_factors_ML)
        activation_scaling_factors_ML = activation_scaling_factors_ML.to(self.W_enc_TMLDH.device)
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
        self.W_enc_TMLDH.mul_(scaling_factors_ML[..., None, None])
        self.W_dec_HTMLD.div_(scaling_factors_ML[..., None])
        self.b_dec_TMLD.div_(scaling_factors_ML[..., None])

    def _validate_scaling_factors(self, scaling_factors_ML: t.Tensor) -> None:
        if t.any(scaling_factors_ML == 0):
            raise ValueError("Scaling factors contain zeros, which would lead to division by zero")
        if t.any(t.isnan(scaling_factors_ML)) or t.any(t.isinf(scaling_factors_ML)):
            raise ValueError("Scaling factors contain NaN or Inf values")

        expected_shape_ML = (self.n_models, self.n_layers)
        if scaling_factors_ML.shape != expected_shape_ML:
            raise ValueError(f"Expected shape {expected_shape_ML}, got {scaling_factors_ML.shape}")

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
        returns a copy of the m with the weights rescaled such that the decoder norm of each feature is one,
        but the m makes the same predictions.
        """
        W_dec_l2_norms_H = reduce(self.W_dec_HTMLD, "h m l dim -> h", l2_norm)

        cc = AcausalCrosscoder(
            n_tokens=self.n_tokens,
            n_models=self.n_models,
            n_layers=self.n_layers,
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            dec_init_norm=0,  # this shouldn't matter
            hidden_activation=self.hidden_activation,
        )

        with t.no_grad():
            cc.W_enc_TMLDH.copy_(self.W_enc_TMLDH * W_dec_l2_norms_H)
            cc.b_enc_H.copy_(self.b_enc_H * W_dec_l2_norms_H)
            cc.W_dec_HTMLD.copy_(self.W_dec_HTMLD / W_dec_l2_norms_H[..., None, None, None])
            # no alteration needed for self.b_dec_TMLD

        return cc
