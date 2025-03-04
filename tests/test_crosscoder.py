from typing import Any

import torch as t

from model_diffing.models.acausal_crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.activations.topk import BatchTopkActivation
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.utils import l2_norm


def test_return_shapes():
    n_models = 2
    batch_size = 4
    n_hookpoints = 6
    d_model = 16
    cc_hidden_dim = 256
    dec_init_norm = 1

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        hidden_activation=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    activations_BMPD = t.randn(batch_size, n_models, n_hookpoints, d_model)
    y_BPD = crosscoder.forward(activations_BMPD)
    assert y_BPD.shape == activations_BMPD.shape
    train_res = crosscoder.forward_train(activations_BMPD)
    assert train_res.output_BXD.shape == activations_BMPD.shape
    assert train_res.hidden_BH.shape == (batch_size, cc_hidden_dim)


def test_batch_topk_activation():
    batch_topk_activation = BatchTopkActivation(k_per_example=2)
    hidden_preact_BH = t.tensor([[1, 2, 3, 4, 10], [1, 2, 11, 12, 13]])
    hidden_BH = batch_topk_activation.forward(hidden_preact_BH)
    assert hidden_BH.shape == hidden_preact_BH.shape
    assert t.all(hidden_BH == t.tensor([[0, 0, 0, 0, 10], [0, 0, 11, 12, 13]]))


def test_weights_folding_keeps_hidden_representations_consistent():
    batch_size = 1
    n_models = 3
    n_hookpoints = 4
    d_model = 5
    cc_hidden_dim = 16
    dec_init_norm = 1

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        hidden_activation=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    scaling_factors_MP = t.randn(n_models, n_hookpoints)

    unscaled_input_BMPD = t.randn(batch_size, n_models, n_hookpoints, d_model)
    scaled_input_BMPD = unscaled_input_BMPD * scaling_factors_MP[..., None]

    output_without_folding = crosscoder.forward_train(scaled_input_BMPD)

    with crosscoder.temporarily_fold_activation_scaling(scaling_factors_MP):
        output_with_folding = crosscoder.forward_train(unscaled_input_BMPD)

    # all hidden representations should be the same
    assert t.allclose(output_without_folding.hidden_BH, output_with_folding.hidden_BH), (
        f"max diff: {t.max(t.abs(output_without_folding.hidden_BH - output_with_folding.hidden_BH))}"
    )

    output_after_unfolding = crosscoder.forward_train(scaled_input_BMPD)

    assert t.allclose(output_without_folding.hidden_BH, output_after_unfolding.hidden_BH), (
        f"max diff: {t.max(t.abs(output_without_folding.hidden_BH - output_after_unfolding.hidden_BH))}"
    )


def test_weights_folding_scales_output_correctly():
    batch_size = 2
    n_models = 4
    n_hookpoints = 5
    d_model = 6
    cc_hidden_dim = 6
    dec_init_norm = 0.1

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        hidden_activation=ReLUActivation(),
        init_strategy=AnthropicTransposeInit(dec_init_norm=dec_init_norm),
    )

    scaling_factors_MP = t.randn(n_models, n_hookpoints)

    unscaled_input_BMPD = t.randn(batch_size, n_models, n_hookpoints, d_model)
    scaled_input_BMPD = unscaled_input_BMPD * scaling_factors_MP[..., None]

    scaled_output_BMPD = crosscoder.forward_train(scaled_input_BMPD).output_BXD

    crosscoder.fold_activation_scaling_into_weights_(scaling_factors_MP)
    unscaled_output_folded_BMPD = crosscoder.forward_train(unscaled_input_BMPD).output_BXD
    scaled_output_folded_BMPD = unscaled_output_folded_BMPD * scaling_factors_MP[..., None]

    # with folded weights, the output should be scaled by the scaling factors
    assert t.allclose(scaled_output_BMPD, scaled_output_folded_BMPD, atol=1e-4), (
        f"max diff: {t.max(t.abs(scaled_output_BMPD - scaled_output_folded_BMPD))}"
    )


class RandomInit(InitStrategy[AcausalCrosscoder[Any]]):
    @t.no_grad()
    def init_weights(self, cc: AcausalCrosscoder[Any]) -> None:
        cc.W_dec_HXD.random_()
        cc.W_enc_XDH.random_()
        cc.b_enc_H.random_()
        cc.b_dec_XD.random_()


def test_weights_rescaling_retains_output():
    batch_size = 1
    n_models = 2
    n_hookpoints = 3
    d_model = 4
    cc_hidden_dim = 8

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        hidden_activation=ReLUActivation(),
        init_strategy=RandomInit(),
    )

    activations_BMPD = t.randn(batch_size, n_models, n_hookpoints, d_model)

    output_BMPD = crosscoder.forward_train(activations_BMPD)
    output_rescaled_BMPD = crosscoder.with_decoder_unit_norm().forward_train(activations_BMPD)

    assert t.allclose(output_BMPD.output_BXD, output_rescaled_BMPD.output_BXD), (
        f"max diff: {t.max(t.abs(output_BMPD.output_BXD - output_rescaled_BMPD.output_BXD))}"
    )


def test_weights_rescaling_max_norm():
    n_models = 2
    n_hookpoints = 3
    d_model = 4
    cc_hidden_dim = 8

    cc = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        hidden_activation=ReLUActivation(),
        init_strategy=RandomInit(),
    ).with_decoder_unit_norm()

    cc_dec_norms_HMP = l2_norm(cc.W_dec_HXD, dim=-1)  # dec norms for each output vector space

    assert t.allclose(
        t.isclose(cc_dec_norms_HMP, t.tensor(1.0))  # for each cc hidden dim,
        .sum(dim=(1, 2))  # only 1 output vector space should have norm 1
        .long(),
        t.ones(cc_hidden_dim).long(),
    )
