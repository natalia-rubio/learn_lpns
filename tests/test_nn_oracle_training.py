"""Oracle-input training sanity checks (append R/S/L to features)."""

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import pytest

from learn_lpns.neural_network.nn_model import NeuralNet
from learn_lpns.neural_network.nn_util import ORACLE_OUTPUT_FEATURE_NAMES, append_output_rri_to_input
from learn_lpns.neural_network.train_nn import train_nn


def test_append_output_rri_to_input_extends_features():
    data_dict = {
        "input": jnp.ones((5, 4), dtype=jnp.float32),
        "output_rri": jnp.array(
            [[1.0, 0.1, 0.2], [2.0, 0.0, 0.3], [3.0, 0.2, 0.4], [4.0, 0.1, 0.5], [5.0, 0.0, 0.6]],
            dtype=jnp.float32,
        ),
        "input_feature_names": ["f0", "f1", "f2", "f3"],
        "generation": jnp.zeros((5,), dtype=jnp.float32),
    }
    out = append_output_rri_to_input(data_dict)

    assert out["input"].shape == (5, 7)
    assert out["input_feature_names"] == ["f0", "f1", "f2", "f3", *ORACLE_OUTPUT_FEATURE_NAMES]
    assert float(out["input"][0, 4]) == pytest.approx(1.0)
    assert float(out["input"][0, 6]) == pytest.approx(0.2)
    assert out["input_mean"].shape == (7,)
    assert out["input_std"].shape == (7,)


def test_oracle_training_fits_synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    n_rows = 8
    n_features = 4
    inputs = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    targets = rng.uniform(0.1, 5.0, size=(n_rows, 3)).astype(np.float32)

    data_dict = append_output_rri_to_input(
        {
            "input": jnp.asarray(inputs),
            "output_rri": jnp.asarray(targets),
            "generation": jnp.zeros((n_rows,), dtype=jnp.float32),
        }
    )

    train_inds = np.arange(n_rows)
    val_inds = np.array([], dtype=int)

    network_params = {
        "num_input_features": int(data_dict["input"].shape[1]),
        "num_layers": 2,
        "layer_width": 32,
        "num_output_features": 1,
        "target_output_column": 0,
        "output_type": "rri",
        "set_name": "oracle_test",
        "set_type": "test",
        "num_geos": 1,
        "geometry_variant": "bifurcations_EL",
        "data_dict": data_dict,
        "activation": "leaky_relu",
        "asymmetric_loss": False,
        "asymmetric_loss_overestimate_weight": 1.0,
        "generation_weighted_loss": False,
        "generation_weighted_loss_decay_base": 2.0,
    }
    optimizer_params = {"init": 0.05, "transition_steps": 200, "decay_rate": 0.99}
    training_params = {
        "num_epochs": 800,
        "batch_size": n_rows,
        "train_inds": train_inds,
        "val_inds": val_inds,
        "num_offsets": 1,
        "early_stop_loss_threshold": 1e-4,
        "verbose_epochs": False,
    }

    for col in range(3):
        network_params["target_output_column"] = col
        training_params["output_dir"] = str(tmp_path / f"col_{col}")
        model = NeuralNet(network_params, optimizer_params)
        train_nn(model, training_params)
        train_rmse = model.eval_pure_loss(train_inds)
        assert train_rmse < 1e-3, f"column {col}: train RMSE {train_rmse}"
