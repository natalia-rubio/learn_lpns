import copy
import os

import dill
import jax.numpy as jnp
import numpy as np
from jax import random, vmap

from learn_lpns.neural_network.activations import ActivationName, apply_activation, normalize_activation

np.random.seed(0)

ORACLE_OUTPUT_FEATURE_NAMES = ("R_poiseuille", "stenosis_coefficient", "L")
OUTPUT_RRI_KEY = "output_rri"


def append_output_rri_to_input(data_dict: dict) -> dict:
    """Return a copy with ``input = hstack(input, output_rri)`` for oracle training sanity checks."""
    if OUTPUT_RRI_KEY not in data_dict:
        raise KeyError(f"data_dict missing {OUTPUT_RRI_KEY!r} required for oracle inputs")

    out = copy.copy(data_dict)
    inputs = jnp.asarray(data_dict["input"])
    targets = jnp.asarray(data_dict[OUTPUT_RRI_KEY])
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("oracle inputs require 2D input and output_rri arrays")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"input rows ({inputs.shape[0]}) must match output_rri rows ({targets.shape[0]})"
        )
    if targets.shape[1] != len(ORACLE_OUTPUT_FEATURE_NAMES):
        raise ValueError(
            f"output_rri must have {len(ORACLE_OUTPUT_FEATURE_NAMES)} columns, got {targets.shape[1]}"
        )

    augmented_input = jnp.concatenate([inputs, targets], axis=1)
    out["input"] = augmented_input

    feature_names = data_dict.get("input_feature_names")
    if feature_names is not None:
        out["input_feature_names"] = list(feature_names) + list(ORACLE_OUTPUT_FEATURE_NAMES)

    input_np = np.asarray(augmented_input, dtype=float)
    input_mean = jnp.asarray(np.mean(input_np, axis=0))
    input_std = jnp.asarray(np.std(input_np, axis=0))
    out["input_mean"] = input_mean
    out["input_std"] = jnp.where(input_std > 0.0, input_std, 1.0)

    return out


def attach_train_output_bounds(model, train_inds) -> None:
    """Store per-column train-set min/max of ``output_rri`` on ``model`` for inference clipping."""
    inds = np.asarray(train_inds, dtype=int)
    if inds.size == 0:
        return
    outputs = np.asarray(model.output[inds], dtype=float)
    model.train_output_min = np.min(outputs, axis=0)
    model.train_output_max = np.max(outputs, axis=0)


def attach_nondim_metadata_from_data_dict(model, data_dict: dict) -> None:
    """Copy non-dimensional R/S/L training metadata from jax pickle onto the model checkpoint."""
    if data_dict.get("nondim_rsl"):
        model.nondim_rsl = True
        model.reference_reynolds = float(data_dict["reference_reynolds"])
        model.nondim_rho = float(data_dict["nondim_rho"])
        model.nondim_mu = float(data_dict["nondim_mu"])
    else:
        model.nondim_rsl = False


def compute_train_output_bounds_from_split(
    model,
    *,
    data_root: str = "data",
    run_config_suffix: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive train-set R/S/L min/max from split_indices + the model's stacked outputs."""
    from learn_lpns.data_processing.generate_split_indices import load_split_for_training, resolve_flat_indices
    from learn_lpns.neural_network.launch_training import _default_split_path

    num_geos = int(getattr(model, "num_geos", 0))
    if num_geos <= 0:
        raise ValueError("Cannot infer train output bounds: model.num_geos is missing or invalid.")

    set_type = getattr(model, "set_type", "all")
    split_path = _default_split_path(
        data_root,
        model.set_name,
        model.geometry_variant,
        set_type,
        num_geos,
        run_config_suffix or getattr(model, "run_config_suffix", None),
    )
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Train/val split not found for clipping bounds: {split_path}")

    split_dict = load_split_for_training(split_path)
    modality = "vessel" if getattr(model, "model_name_suffix", "") == "_vessel" else "junction"
    train_inds = resolve_flat_indices(split_dict, modality, "train", split_path=split_path)
    if len(train_inds) == 0:
        raise ValueError(f"No training indices in split file: {split_path}")

    outputs = np.asarray(model.output[np.asarray(train_inds, dtype=int)], dtype=float)
    return np.min(outputs, axis=0), np.max(outputs, axis=0)


def resolve_train_output_bounds(
    model,
    *,
    data_root: str = "data",
    run_config_suffix: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min, max) vectors length 3 for R/S/L clipping."""
    if hasattr(model, "train_output_min") and hasattr(model, "train_output_max"):
        return (
            np.asarray(model.train_output_min, dtype=float),
            np.asarray(model.train_output_max, dtype=float),
        )
    try:
        mins, maxs = compute_train_output_bounds_from_split(
            model,
            data_root=data_root,
            run_config_suffix=run_config_suffix,
        )
        print(
            "  clip_predictions: loaded train-set bounds from split_indices "
            f"(R [{mins[0]:.4g}, {maxs[0]:.4g}], S [{mins[1]:.4g}, {maxs[1]:.4g}], L [{mins[2]:.4g}, {maxs[2]:.4g}])"
        )
        return mins, maxs
    except (FileNotFoundError, ValueError) as exc:
        data_dict = getattr(model, "data_dict", None) or {}
        if "output_min" in data_dict and "output_max" in data_dict:
            print(
                "  clip_predictions: could not load train split "
                f"({exc}); using cohort output_min/max instead"
            )
            return (
                np.asarray(data_dict["output_min"], dtype=float),
                np.asarray(data_dict["output_max"], dtype=float),
            )
        raise ValueError(
            "clip_predictions is enabled but the loaded model has no train output bounds, "
            "no usable split_indices file, and no cohort output_min/output_max in data_dict."
        ) from exc


def clip_rsl_predictions(
    pred_R: np.ndarray,
    pred_S: np.ndarray,
    pred_L: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip R/S/L prediction arrays to per-column [min, max] bounds."""
    mins = np.asarray(bounds_min, dtype=float).reshape(-1)
    maxs = np.asarray(bounds_max, dtype=float).reshape(-1)
    if mins.shape[0] < 3 or maxs.shape[0] < 3:
        raise ValueError(f"Expected 3 output bounds, got min={mins.shape}, max={maxs.shape}")
    return (
        np.clip(pred_R, mins[0], maxs[0]),
        np.clip(pred_S, mins[1], maxs[1]),
        np.clip(pred_L, mins[2], maxs[2]),
    )


def get_sizes(network_params):
    """Layer shapes for a single- or multi-output MLP."""
    num_output_features = int(network_params.get("num_output_features", 1))
    num_input_features = network_params["num_input_features"]
    layer_width = network_params["layer_width"]
    num_layers = network_params["num_layers"]
    sizes_in = [num_input_features] + [layer_width] * num_layers
    sizes_out = [layer_width] * num_layers + [num_output_features]
    return sizes_in, sizes_out


def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), 0 * scale * random.normal(b_key, (n,))


def init_weights(network_params):
    key = random.key(0)
    sizes_in, sizes_out = get_sizes(network_params)
    keys = random.split(key, len(sizes_in))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes_in, sizes_out, keys, strict=False)]


def get_batch_indices(indices, batch_size):
    num_batches = len(indices) // batch_size
    np.random.shuffle(indices)
    indices = indices[: num_batches * batch_size]
    return [indices[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]


def forward_pass(input, weights, activation: ActivationName = "relu"):
    act = normalize_activation(activation)
    latent_rep = input
    for w, b in weights[:-1]:
        lin_comb = jnp.dot(w, latent_rep) + b
        latent_rep = apply_activation(lin_comb, act)

    final_w, final_b = weights[-1]
    return jnp.dot(final_w, latent_rep) + final_b


def batched_forward_pass(input, weights, activation: ActivationName = "relu"):
    """Vectorized forward pass over batch dimension (axis 0 of input)."""
    return vmap(forward_pass, in_axes=(0, None, None))(input, weights, activation)


def get_L2(weights):
    return jnp.sum(jnp.array([jnp.linalg.norm(w) for w, _ in weights]))


def dill_save(di_, filename_):
    with open(filename_, "wb") as f:
        dill.dump(di_, f)


def dill_load(filename_):
    with open(filename_, "rb") as f:
        return dill.load(f)
