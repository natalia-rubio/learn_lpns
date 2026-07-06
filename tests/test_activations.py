"""Tests for MLP activation helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from learn_lpns.neural_network.activations import (
    ACTIVATION_NAMES,
    apply_activation,
    normalize_activation,
    resolve_activation_from_model,
    resolve_activation_from_network_params,
)


def test_normalize_activation_aliases():
    assert normalize_activation("relu") == "relu"
    assert normalize_activation("Leaky-ReLU") == "leaky_relu"
    assert normalize_activation("TANH") == "tanh"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_activation("sigmoid")


def test_apply_activation_shapes():
    x = jnp.array([-2.0, 0.0, 2.0])
    assert np.allclose(np.array(apply_activation(x, "relu")), [0.0, 0.0, 2.0])
    assert np.allclose(np.array(apply_activation(x, "leaky_relu")), [-0.02, 0.0, 2.0])
    assert np.allclose(np.array(apply_activation(x, "tanh")), np.tanh([-2.0, 0.0, 2.0]))


def test_resolve_activation_legacy_use_leaky_relu():
    assert resolve_activation_from_network_params({"activation": "tanh"}) == "tanh"
    assert resolve_activation_from_network_params({"use_leaky_relu": True}) == "leaky_relu"
    assert resolve_activation_from_network_params({}) == "relu"

    class LegacyModel:
        use_leaky_relu = True

    assert resolve_activation_from_model(LegacyModel()) == "leaky_relu"


def test_activation_names_tuple():
    assert "relu" in ACTIVATION_NAMES
    assert "leaky_relu" in ACTIVATION_NAMES
    assert "tanh" in ACTIVATION_NAMES
