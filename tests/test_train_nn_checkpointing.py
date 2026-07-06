"""Tests for train-only checkpoint selection and best-weight restore."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from learn_lpns.neural_network.nn_model import NeuralNet
from learn_lpns.neural_network.train_nn import train_nn


def _minimal_network_params(*, data_dict, target_output_column=0):
    return {
        "num_input_features": int(data_dict["input"].shape[1]),
        "num_layers": 2,
        "layer_width": 8,
        "num_output_features": 1,
        "target_output_column": target_output_column,
        "output_type": "rri",
        "set_name": "checkpoint_test",
        "set_type": "test",
        "num_geos": 1,
        "geometry_variant": "bifurcations_EL",
        "data_dict": data_dict,
        "activation": "relu",
        "asymmetric_loss": False,
        "asymmetric_loss_overestimate_weight": 1.0,
        "generation_weighted_loss": False,
        "generation_weighted_loss_decay_base": 2.0,
    }


def _synthetic_data_dict(n_rows=6):
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(n_rows, 4)).astype(np.float32)
    targets = rng.uniform(0.1, 5.0, size=(n_rows, 3)).astype(np.float32)
    return {
        "input": jnp.asarray(inputs),
        "output_rri": jnp.asarray(targets),
        "generation": jnp.zeros((n_rows,), dtype=jnp.float32),
        "input_mean": jnp.zeros((4,), dtype=jnp.float32),
        "input_std": jnp.ones((4,), dtype=jnp.float32),
    }


def _optimizer_params():
    return {"init": 0.01, "transition_steps": 200, "decay_rate": 0.99}


def _training_params(*, tmp_path, train_inds, val_inds, **overrides):
    params = {
        "num_epochs": 5,
        "batch_size": len(train_inds),
        "train_inds": train_inds,
        "val_inds": val_inds,
        "num_offsets": 1,
        "early_stop_loss_threshold": 1e-12,
        "restore_best_weights": True,
        "verbose_epochs": False,
        "output_dir": str(tmp_path),
    }
    params.update(overrides)
    return params


@pytest.fixture
def skip_train_io():
    with patch("learn_lpns.neural_network.train_nn.dill_save"), patch(
        "learn_lpns.neural_network.train_nn._save_training_plot"
    ):
        yield


def test_train_nn_returns_result_dict(tmp_path):
    data_dict = _synthetic_data_dict()
    train_inds = np.arange(4)
    val_inds = np.array([4, 5], dtype=int)
    model = NeuralNet(_minimal_network_params(data_dict=data_dict), _optimizer_params())
    result = train_nn(model, _training_params(tmp_path=tmp_path, train_inds=train_inds, val_inds=val_inds))

    assert set(result) == {"best_train_loss", "best_epoch", "final_val_pure_rmse"}
    assert result["best_epoch"] is not None
    assert result["best_train_loss"] is not None
    assert result["final_val_pure_rmse"] is not None


def test_train_nn_stores_checkpoint_metadata(tmp_path):
    data_dict = _synthetic_data_dict()
    train_inds = np.arange(6)
    val_inds = np.array([], dtype=int)
    model = NeuralNet(_minimal_network_params(data_dict=data_dict), _optimizer_params())
    train_nn(
        model,
        _training_params(
            tmp_path=tmp_path,
            train_inds=train_inds,
            val_inds=val_inds,
            restore_best_weights=True,
        ),
    )

    assert hasattr(model, "best_epoch")
    assert hasattr(model, "best_train_loss")
    assert hasattr(model, "restored_from_best")
    assert isinstance(model.restored_from_best, bool)


def test_train_nn_restores_best_weights_not_last_epoch(tmp_path, skip_train_io):
    data_dict = _synthetic_data_dict()
    train_inds = np.arange(6)
    val_inds = np.array([], dtype=int)
    model = NeuralNet(_minimal_network_params(data_dict=data_dict), _optimizer_params())

    train_losses = [10.0, 5.0, 2.0, 8.0, 15.0]
    train_eval_calls = [0]
    update_epoch = [0]

    def marked_update(indices):
        update_epoch[0] += 1
        marker = jnp.array(float(update_epoch[0]), dtype=jnp.float32)
        w, b = model.weights[0]
        model.weights[0] = (w * 0.0 + marker, b)

    def fake_train_eval(indices):
        idx = min(train_eval_calls[0], len(train_losses) - 1)
        train_eval_calls[0] += 1
        return train_losses[idx]

    with patch.object(model, "update", side_effect=marked_update):
        with patch.object(model, "eval_training_loss", side_effect=fake_train_eval):
            result = train_nn(
                model,
                _training_params(
                    tmp_path=tmp_path,
                    train_inds=train_inds,
                    val_inds=val_inds,
                    num_epochs=5,
                    restore_best_weights=True,
                ),
            )

    assert result["best_epoch"] == 2
    assert result["best_train_loss"] == pytest.approx(2.0)
    assert model.restored_from_best is True
    assert float(model.weights[0][0][0, 0]) == pytest.approx(3.0)


def test_train_nn_val_does_not_trigger_early_stop(tmp_path, skip_train_io):
    data_dict = _synthetic_data_dict()
    train_inds = np.arange(4)
    val_inds = np.array([4, 5], dtype=int)
    model = NeuralNet(_minimal_network_params(data_dict=data_dict), _optimizer_params())

    train_losses = [1.0, 1.0, 1.0]
    train_calls = [0]

    def fake_train_eval(indices):
        inds = np.asarray(indices, dtype=int)
        if np.any(np.isin(inds, val_inds)):
            return 100.0
        idx = min(train_calls[0], len(train_losses) - 1)
        train_calls[0] += 1
        return train_losses[idx]

    def fake_val_pure(indices):
        return 1e-12

    with patch.object(model, "update"):
        with patch.object(model, "eval_training_loss", side_effect=fake_train_eval):
            with patch.object(model, "eval_pure_loss", side_effect=fake_val_pure):
                result = train_nn(
                model,
                _training_params(
                    tmp_path=tmp_path,
                    train_inds=train_inds,
                    val_inds=val_inds,
                    num_epochs=3,
                    early_stop_loss_threshold=1e-8,
                    restore_best_weights=False,
                ),
            )

    assert train_calls[0] == 3
    assert result["final_val_pure_rmse"] == pytest.approx(1e-12)


def test_train_nn_early_stops_on_train_training_loss(tmp_path, skip_train_io):
    data_dict = _synthetic_data_dict()
    train_inds = np.arange(6)
    val_inds = np.array([], dtype=int)
    model = NeuralNet(_minimal_network_params(data_dict=data_dict), _optimizer_params())

    train_losses = [1e-10, 1.0, 1.0]

    with patch.object(model, "update"):
        with patch.object(model, "eval_training_loss", side_effect=train_losses):
            result = train_nn(
            model,
            _training_params(
                tmp_path=tmp_path,
                train_inds=train_inds,
                val_inds=val_inds,
                num_epochs=5,
                early_stop_loss_threshold=1e-8,
            ),
        )

    assert result["best_epoch"] == 0
