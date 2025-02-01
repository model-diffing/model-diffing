from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import torch as t
from einops import einsum, rearrange, reduce
from torch import nn

from model_diffing.scripts.train_jan_update_crosscoder.config import JumpReLUConfig
from model_diffing.utils import SaveableModule, l2_norm

import sys


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


class TopkActivation(SaveableModule):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, hidden_preactivation_BH: t.Tensor) -> t.Tensor:
        _topk_values_BH, topk_indices_BH = hidden_preactivation_BH.topk(self.k, dim=-1)
        hidden_BH = t.zeros_like(hidden_preactivation_BH)
        hidden_BH.scatter_(-1, topk_indices_BH, _topk_values_BH)
        return hidden_BH

    def dump_cfg(self) -> dict[str, int | str]:
        return {"k": self.k}

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "TopkActivation":
        return cls(cfg["k"])


class BatchTopkActivation(SaveableModule):
    def __init__(self, k_per_example: int):
        super().__init__()
        self.k_per_example = k_per_example

    def forward(self, hidden_preactivation_BH: t.Tensor) -> t.Tensor:
        batch_size = hidden_preactivation_BH.shape[0]
        batch_k = self.k_per_example * batch_size
        hidden_preactivation_Bh = rearrange(hidden_preactivation_BH, "batch hidden -> (batch hidden)")
        _topk_values_Bh, topk_indices_Bh = hidden_preactivation_Bh.topk(k=batch_k)
        hidden_Bh = t.zeros_like(hidden_preactivation_Bh)
        hidden_Bh.scatter_(-1, topk_indices_Bh, _topk_values_Bh)
        hidden_BH = rearrange(hidden_Bh, "(batch hidden) -> batch hidden", batch=batch_size)
        return hidden_BH

    def dump_cfg(self) -> dict[str, int | str]:
        return {"k_per_example": self.k_per_example}

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "BatchTopkActivation":
        return cls(cfg["k_per_example"])


class ReLUActivation(SaveableModule):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x)

    def dump_cfg(self) -> dict[str, int | str]:
        return {}

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "ReLUActivation":
        return cls()


ACTIVATIONS = {
    "TopkActivation": TopkActivation,
    "BatchTopkActivation": BatchTopkActivation,
    "ReLU": ReLUActivation,
}


def rectangle(x: t.Tensor) -> t.Tensor:
    """
    when:
        x < -0.5 -> 0
        -0.5 < x < 0.5 -> 1
        x > 0.5 -> 0
    """
    return ((x > -0.5) & (x < 0.5)).to(x)


class JumpReLUActivation(SaveableModule):
    def __init__(
        self,
        size: int,
        bandwidth: float,
        threshold_init: float,
    ):
        super().__init__()
        self.log_threshold_H = nn.Parameter(t.ones(size) * threshold_init)
        self.bandwidth = bandwidth

    def forward(self, x_BX: t.Tensor) -> t.Tensor:
        return JumpReLU.apply(x_BX, self.log_threshold_H, self.bandwidth)  # type: ignore

    def dump_cfg(self) -> dict[str, int | float | str]:
        return {"size": self.log_threshold_H.shape[0], "bandwidth": self.bandwidth}

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "JumpReLUActivation":
        return cls(**cfg)


class JumpReLU(t.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_BX: t.Tensor,
        log_threshold_X: t.Tensor,
        bandwidth: float,
        backprop_through_input: bool = False,
    ) -> t.Tensor:
        """
        threshold_X is $\\theta$ in the GDM paper, $t$ in the Anthropic paper.

        Where GDM don't backprop through the threshold in "jumping ahead", Anthropic do in the jan 2025 update.
        """
        threshold_X = log_threshold_X.exp()
        ctx.save_for_backward(input_BX, threshold_X, t.tensor(bandwidth))
        ctx.backprop_through_input = backprop_through_input
        return input_BX > threshold_X * input_BX

    @staticmethod
    def backward(ctx: Any, grad_output_BX: t.Tensor) -> tuple[t.Tensor | None, t.Tensor, None]:  # type: ignore
        input_BX, threshold_X, bandwidth = ctx.saved_tensors

        grad_threshold_BX = (
            threshold_X  #
            * -(1 / bandwidth)
            * rectangle((input_BX - threshold_X) / bandwidth)
            * grad_output_BX
        )

        grad_threshold_X = grad_threshold_BX.sum(0)  # this is technically unnecessary as torch will automatically do it

        if ctx.backprop_through_input:
            return (
                (input_BX > threshold_X) * grad_output_BX,  # input_BX
                grad_threshold_X,
                None,  # bandwidth
            )

        return (
            None,  # input_BX
            grad_threshold_X,
            None,  # bandwidth
        )


class AcausalCrosscoder[TActivation: SaveableModule](SaveableModule):
    """crosscoder that autoencodes activations of a subset of a model's layers"""

    folded_scaling_factors_ML: t.Tensor | None
    """Scaling factors that have been folded into the weights, if any"""

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

        self.register_buffer("folded_scaling_factors_ML", None, persistent=True)

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

    @property
    def is_folded(self) -> bool:
        return self.folded_scaling_factors_ML is not None

    @contextmanager
    def temporarily_fold_activation_scaling(self, scaling_factors_ML: t.Tensor):
        """Temporarily fold scaling factors into weights."""
        self.fold_activation_scaling_into_weights_(scaling_factors_ML)
        yield
        _ = self.unfold_activation_scaling_from_weights_()

    # @overload
    # def fold_activation_scaling_into_weights_(
    #     self, activation_scaling_factors_ML: t.Tensor, temporary: Literal[True]
    # ) -> "ContextManager": ...

    # @overload
    # def fold_activation_scaling_into_weights_(
    #     self, activation_scaling_factors_ML: t.Tensor, temporary: Literal[False]
    # ) -> None: ...

    def fold_activation_scaling_into_weights_(self, activation_scaling_factors_ML: t.Tensor) -> None:
        """scales the crosscoder weights by the activation scaling factors, so that the model can be run on raw llm activations."""
        if self.is_folded:
            raise ValueError("Scaling factors already folded into weights")

        self._validate_scaling_factors(activation_scaling_factors_ML)
        activation_scaling_factors_ML = activation_scaling_factors_ML.to(self.W_enc_MLDH.device)
        self._scale_weights(activation_scaling_factors_ML)
        # set buffer to prevent double-folding
        self.folded_scaling_factors_ML = activation_scaling_factors_ML

        # if temporary:
        #     def on_exit():
        #         _ = self.unfold_activation_scaling_from_weights_()
        #     return ContextManager(on_exit)

    def unfold_activation_scaling_from_weights_(self) -> t.Tensor:
        if not self.is_folded:
            raise ValueError("No folded scaling factors found")

        folded_scaling_factors_ML = cast(t.Tensor, self.folded_scaling_factors_ML).clone()
        # Clear the buffer before operations to prevent double-unfolding
        self.folded_scaling_factors_ML = None
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

    def dump_cfg(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "hidden_activation_classname": self.hidden_activation.__class__.__name__,
            "hidden_activation_cfg": self.hidden_activation.dump_cfg(),
        }

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "AcausalCrosscoder[TActivation]":
        hidden_activation_cfg = cfg["hidden_activation_cfg"]
        hidden_activation_classname = cfg["hidden_activation_classname"]

        hidden_activation = ACTIVATIONS[hidden_activation_classname].from_cfg(hidden_activation_cfg)

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
        Rescale the weights so that the decoder norm is one but the model makes the same predictions.
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


def build_relu_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    dec_init_norm: float,
) -> AcausalCrosscoder[ReLUActivation]:
    return AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=ReLUActivation(),
    )


def build_topk_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    k: int,
    dec_init_norm: float,
) -> AcausalCrosscoder[TopkActivation]:
    return AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=TopkActivation(k=k),
    )


def build_batch_topk_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    k_per_example: int,
    dec_init_norm: float,
) -> AcausalCrosscoder[BatchTopkActivation]:
    return AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=BatchTopkActivation(k_per_example=k_per_example),
    )


def build_jumprelu_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    dec_init_norm: float,
) -> AcausalCrosscoder[JumpReLUActivation]:
    return AcausalCrosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=JumpReLUActivation(
            size=cc_hidden_dim,
            bandwidth=jumprelu.bandwidth,
            threshold_init=jumprelu.threshold_init,
        ),
    )


# if __name__ == "__main__":
#     model = build_batch_topk_crosscoder(n_models=2, n_layers=3, d_model=4, cc_hidden_dim=5, k=2, dec_init_norm=1.0)
#     model.push_to_hub(repo_id="model-diffing/batch-topk-crosscoder", use_auth_token=True)


# class ContextManager:
#     def __init__(self, on_exit: Callable[[], None]):
#         self.on_exit = on_exit

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
#         self.on_exit()
