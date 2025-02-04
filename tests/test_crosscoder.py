import torch as t

from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.models.activations.topk import BatchTopkActivation
from model_diffing.models.crosscoder import AcausalCrosscoder


def test_return_shapes():
    n_models = 2
    batch_size = 4
    n_tokens = 3
    n_layers = 6
    d_model = 16
    cc_hidden_dim = 256
    dec_init_norm = 1

    crosscoder = AcausalCrosscoder(
        n_tokens=n_tokens,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=ReLUActivation(),
    )

    activations_BTMLD = t.randn(batch_size, n_tokens, n_models, n_layers, d_model)
    y_BLD = crosscoder.forward(activations_BTMLD)
    assert y_BLD.shape == activations_BTMLD.shape
    train_res = crosscoder.forward_train(activations_BTMLD)
    assert train_res.reconstructed_acts_BTMLD.shape == activations_BTMLD.shape
    assert train_res.hidden_BH.shape == (batch_size, cc_hidden_dim)


def test_batch_topk_activation():
    batch_topk_activation = BatchTopkActivation(k_per_example=2)
    hidden_preactivation_BH = t.tensor([[1, 2, 3, 4, 10], [1, 2, 11, 12, 13]])
    hidden_BH = batch_topk_activation.forward(hidden_preactivation_BH)
    assert hidden_BH.shape == hidden_preactivation_BH.shape
    assert t.all(hidden_BH == t.tensor([[0, 0, 0, 0, 10], [0, 0, 11, 12, 13]]))


def test_weights_folding_keeps_hidden_representations_consistent():
    batch_size = 1
    n_tokens = 2
    n_models = 3
    n_layers = 4
    d_model = 5
    cc_hidden_dim = 16
    dec_init_norm = 1

    crosscoder = AcausalCrosscoder(
        n_tokens=n_tokens,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=ReLUActivation(),
    )

    scaling_factors_ML = t.randn(n_models, n_layers)

    unscaled_input_BTMLD = t.randn(batch_size, n_tokens, n_models, n_layers, d_model)
    scaled_input_BTMLD = unscaled_input_BTMLD * scaling_factors_ML[..., None]

    output_without_folding = crosscoder.forward_train(scaled_input_BTMLD)

    with crosscoder.temporarily_fold_activation_scaling(scaling_factors_ML):
        output_with_folding = crosscoder.forward_train(unscaled_input_BTMLD)

    output_after_unfolding = crosscoder.forward_train(scaled_input_BTMLD)

    # all hidden representations should be the same
    assert t.allclose(output_without_folding.hidden_BH, output_with_folding.hidden_BH), (
        f"max diff: {t.max(t.abs(output_without_folding.hidden_BH - output_with_folding.hidden_BH))}"
    )
    assert t.allclose(output_without_folding.hidden_BH, output_after_unfolding.hidden_BH), (
        f"max diff: {t.max(t.abs(output_without_folding.hidden_BH - output_after_unfolding.hidden_BH))}"
    )


def test_weights_folding_scales_output_correctly():
    batch_size = 2
    n_tokens = 3
    n_models = 4
    n_layers = 5
    d_model = 6
    cc_hidden_dim = 6
    dec_init_norm = 0.1

    crosscoder = AcausalCrosscoder(
        n_tokens=n_tokens,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=ReLUActivation(),
    )

    scaling_factors_ML = t.randn(n_models, n_layers)

    unscaled_input_BTMLD = t.randn(batch_size, n_tokens, n_models, n_layers, d_model)
    scaled_input_BTMLD = unscaled_input_BTMLD * scaling_factors_ML[..., None]

    scaled_output_BTMLD = crosscoder.forward_train(scaled_input_BTMLD).reconstructed_acts_BTMLD

    crosscoder.fold_activation_scaling_into_weights_(scaling_factors_ML)
    unscaled_output_folded_BTMLD = crosscoder.forward_train(unscaled_input_BTMLD).reconstructed_acts_BTMLD
    scaled_output_folded_BTMLD = unscaled_output_folded_BTMLD * scaling_factors_ML[..., None]

    # with folded weights, the output should be scaled by the scaling factors
    assert t.allclose(scaled_output_BTMLD, scaled_output_folded_BTMLD, atol=1e-4), (
        f"max diff: {t.max(t.abs(scaled_output_BTMLD - scaled_output_folded_BTMLD))}"
    )


def test_weights_rescaling():
    batch_size = 1
    n_tokens = 2
    n_models = 2
    n_layers = 3
    d_model = 4
    cc_hidden_dim = 32
    dec_init_norm = 0.1

    crosscoder = AcausalCrosscoder(
        n_tokens=n_tokens,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=ReLUActivation(),
    )

    activations_BTMLD = t.randn(batch_size, n_tokens, n_models, n_layers, d_model)
    output_BTMLD = crosscoder.forward_train(activations_BTMLD)

    new_cc = crosscoder.with_decoder_unit_norm()
    output_rescaled_BTMLD = new_cc.forward_train(activations_BTMLD)

    assert t.allclose(output_BTMLD.reconstructed_acts_BTMLD, output_rescaled_BTMLD.reconstructed_acts_BTMLD), (
        f"max diff: {t.max(t.abs(output_BTMLD.reconstructed_acts_BTMLD - output_rescaled_BTMLD.reconstructed_acts_BTMLD))}"
    )

    new_cc_dec_norms = new_cc.W_dec_HTMLD.norm(p=2, dim=(1, 2, 3))
    assert t.allclose(new_cc_dec_norms, t.ones_like(new_cc_dec_norms))


def test_weights_rescaling_makes_unit_norm_decoder_output():
    batch_size = 1
    n_tokens = 2
    n_models = 3
    n_layers = 4
    d_model = 5
    cc_hidden_dim = 32
    dec_init_norm = 0.1

    crosscoder = AcausalCrosscoder(
        n_tokens=n_tokens,
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=dec_init_norm,
        hidden_activation=ReLUActivation(),
    )

    activations_BTMLD = t.randn(batch_size, n_tokens, n_models, n_layers, d_model)
    output_BTMLD = crosscoder.forward_train(activations_BTMLD)

    new_cc = crosscoder.with_decoder_unit_norm()
    output_rescaled_BTMLD = new_cc.forward_train(activations_BTMLD)

    assert t.allclose(output_BTMLD.reconstructed_acts_BTMLD, output_rescaled_BTMLD.reconstructed_acts_BTMLD), (
        f"max diff: {t.max(t.abs(output_BTMLD.reconstructed_acts_BTMLD - output_rescaled_BTMLD.reconstructed_acts_BTMLD))}"
    )

    new_cc_dec_norms = new_cc.W_dec_HTMLD.norm(p=2, dim=(1, 2, 3))
    assert t.allclose(new_cc_dec_norms, t.ones_like(new_cc_dec_norms))
