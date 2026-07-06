"""Hidden-layer activation functions for R/S/L MLPs."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

ActivationName = Literal["relu", "leaky_relu", "tanh"]

ACTIVATION_NAMES: tuple[ActivationName, ...] = ("relu", "leaky_relu", "tanh")

LEAKY_RELU_NEGATIVE_SLOPE = 0.01

_ALIASES: dict[str, ActivationName] = {
    "relu": "relu",
    "leaky_relu": "leaky_relu",
    "leakyrelu": "leaky_relu",
    "leaky-relu": "leaky_relu",
    "tanh": "tanh",
}


def normalize_activation(name: str, *, context: str = "activation") -> ActivationName:
    """Validate and normalize an activation name from config or CLI."""
    key = str(name).strip().lower().replace("-", "_")
    resolved = _ALIASES.get(key)
    if resolved is None:
        choices = ", ".join(ACTIVATION_NAMES)
        raise ValueError(f"Unknown {context} {name!r}; choose from: {choices}")
    return resolved


def resolve_activation_from_network_params(network_params: dict) -> ActivationName:
    """Resolve activation for a new network (supports legacy use_leaky_relu checkpoints params)."""
    if network_params.get("activation") is not None:
        return normalize_activation(network_params["activation"], context="network activation")
    if network_params.get("use_leaky_relu"):
        return "leaky_relu"
    return "relu"


def resolve_activation_from_model(model) -> ActivationName:
    """Resolve activation from a trained model checkpoint (legacy use_leaky_relu supported)."""
    activation = getattr(model, "activation", None)
    if activation is not None:
        return normalize_activation(activation, context="model activation")
    if getattr(model, "use_leaky_relu", False):
        return "leaky_relu"
    return "relu"


def apply_activation(x, activation: ActivationName):
    """Apply a named activation to hidden-layer pre-activations."""
    if activation == "relu":
        return jnp.maximum(0, x)
    if activation == "leaky_relu":
        return jnp.where(x >= 0, x, LEAKY_RELU_NEGATIVE_SLOPE * x)
    if activation == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unsupported activation {activation!r}")
