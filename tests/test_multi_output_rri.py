"""Tests for multi-output RRI neural network training."""

import jax.numpy as jnp

from learn_lpns.config import load_pipeline_config
from learn_lpns.neural_network.nn_model import RRI_NUM_OUTPUTS, loss_pure, predict
from learn_lpns.neural_network.nn_util import get_sizes, init_weights
from learn_lpns.neural_network.train_nn import _model_checkpoint_basename
from learn_lpns.zerod_calibration.nn_inference import (
    rri_model_checkpoint_path,
    rri_models_complete,
    rri_separate_model_checkpoint_paths,
)


def test_get_sizes_multi_output():
    network_params = {
        "num_input_features": 5,
        "num_layers": 2,
        "layer_width": 8,
        "num_output_features": 3,
    }
    sizes_in, sizes_out = get_sizes(network_params)
    assert sizes_in == [5, 8, 8]
    assert sizes_out == [8, 8, 3]


def test_loss_grad_multi_output():
    """Regression: weights must not be a JIT static arg (list is unhashable)."""
    from jax import grad

    from learn_lpns.neural_network.nn_model import loss

    network_params = {
        "num_input_features": 4,
        "num_layers": 1,
        "layer_width": 6,
        "num_output_features": RRI_NUM_OUTPUTS,
    }
    weights = init_weights(network_params)
    batch = 5
    inputs = jnp.ones((batch, 4), dtype=jnp.float32)
    outputs = jnp.ones((batch, RRI_NUM_OUTPUTS), dtype=jnp.float32)
    overestimate_weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    sample_w = jnp.ones((batch,), dtype=jnp.float32)
    grads = grad(loss, argnums=-3)(
        inputs,
        outputs,
        0,
        RRI_NUM_OUTPUTS,
        False,
        weights,
        overestimate_weights,
        sample_w,
    )
    assert len(grads) == len(weights)


def test_loss_pure_multi_output():
    network_params = {
        "num_input_features": 4,
        "num_layers": 1,
        "layer_width": 6,
        "num_output_features": RRI_NUM_OUTPUTS,
    }
    weights = init_weights(network_params)
    batch = 7
    inputs = jnp.ones((batch, 4), dtype=jnp.float32)
    outputs = jnp.ones((batch, RRI_NUM_OUTPUTS), dtype=jnp.float32) * 2.0
    value = loss_pure(inputs, outputs, 0, RRI_NUM_OUTPUTS, False, weights)
    assert float(value) > 0.0


def test_predict_multi_output_shape():
    network_params = {
        "num_input_features": 3,
        "num_layers": 1,
        "layer_width": 4,
        "num_output_features": RRI_NUM_OUTPUTS,
    }
    weights = init_weights(network_params)
    inputs = jnp.ones((5, 3), dtype=jnp.float32)
    pred = predict(inputs, weights, False)
    assert pred.shape == (5, RRI_NUM_OUTPUTS)


def test_model_checkpoint_basename_multi_output():
    class _Model:
        output_type = "rri"
        set_name = "VMR_test"
        model_name_suffix = ""
        num_output_features = 3
        target_output_column = None

    assert _model_checkpoint_basename(_Model()) == "rri_VMR_test_pred_rsl"


def test_load_pipeline_config_multi_output_default_false():
    cfg = load_pipeline_config()
    assert cfg.training.multi_output_rri is False


def test_rri_checkpoint_paths(tmp_path):
    model_dir = str(tmp_path)
    set_name = "VMR_aorta"
    multi = rri_model_checkpoint_path(model_dir, set_name, vessel=False)
    separate = rri_separate_model_checkpoint_paths(model_dir, set_name, vessel=False)
    assert multi.endswith("rri_VMR_aorta_pred_rsl_model")
    assert len(separate) == 3
    assert separate[0].endswith("rri_VMR_aorta_pred_0_model")
    assert not rri_models_complete(model_dir, set_name, vessel=False, multi_output=True)
    with open(multi, "wb") as f:
        f.write(b"x")
    assert rri_models_complete(model_dir, set_name, vessel=False, multi_output=True)
