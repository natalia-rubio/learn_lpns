"""Neural-network inference: load jax cohort rows and write R/S/L predictions into 0D JSON."""

from __future__ import annotations

import json
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from learn_lpns.neural_network.nn_model import predict
from learn_lpns.neural_network.nn_util import dill_load


def forward_jax_pickle_path(
    data_root: str,
    set_name: str,
    run_config_suffix: str,
    geometry_variant: str,
    num_geos: int = 1,
) -> str:
    """Path to per-forward-pass jax pickle (default: single geometry, set_type=forward)."""
    return os.path.join(
        data_root,
        "jax_arrays",
        set_name,
        run_config_suffix,
        geometry_variant,
        "forward",
        f"jax_arrays_num_geos_{num_geos}.pkl",
    )


def _resolve_model_paths(model_dir: str, set_name: str, *, vessel: bool = False) -> list[str]:
    suffix = "vessel_pred" if vessel else "pred"
    model_base_name = f"rri_{set_name}_{suffix}"
    return [os.path.join(model_dir, f"{model_base_name}_{i}_model") for i in range(3)]


def run_nn_predict(
    X: np.ndarray,
    model_dir: str,
    set_name: str,
    *,
    vessel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run three NN heads (R, stenosis, L); return prediction arrays."""
    model_paths = _resolve_model_paths(model_dir, set_name, vessel=vessel)
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

    X_jax = jnp.array(X, dtype=jnp.float32)
    raw_predictions = []
    for i, model_path in enumerate(model_paths):
        print(f"      Loading model {i + 1}/3: {model_path}")
        model = dill_load(model_path)
        use_leaky = getattr(model, "use_leaky_relu", False)
        pred = predict(X_jax, model.weights, use_leaky)
        raw_predictions.append(np.array(pred).flatten())

    output_names = ["R_poiseuille", "stenosis_coefficient", "L"]
    for coef_idx, pred in enumerate(raw_predictions):
        print(f"      {output_names[coef_idx]}: pred range=[{pred.min():.4f}, {pred.max():.4f}]")
    return np.array(raw_predictions[0]), np.array(raw_predictions[1]), np.array(raw_predictions[2])


def apply_junction_predictions(
    nn_config: dict[str, Any],
    *,
    X: np.ndarray,
    junction_names: list[str],
    outlet_primary_names: list[str],
    outlet_vessel_ids: list[int],
    pred_R: np.ndarray,
    pred_S: np.ndarray,
    pred_L: np.ndarray,
    junction_type: str,
) -> None:
    """Write predicted R/S/L into ``nn_config`` junctions (mutates in place)."""
    if len(pred_R) != len(X):
        raise ValueError(
            f"Prediction array size mismatch: input has {len(X)} rows, but predictions have {len(pred_R)} values."
        )

    primary_outlet_to_row: dict[tuple[str, str], int] = {}
    junction_name_to_row_indices: dict[str, list[int]] = {}
    for row_idx, (junc_name, pout_name) in enumerate(zip(junction_names, outlet_primary_names, strict=False)):
        primary_outlet_to_row[(junc_name, pout_name)] = row_idx
        junction_name_to_row_indices.setdefault(junc_name, []).append(row_idx)

    print(f"      Built prediction mapping: {len(primary_outlet_to_row)} (junction, outlet) entries")
    for (jn, on), ri in primary_outlet_to_row.items():
        vid = int(outlet_vessel_ids[ri])
        print(f"        ({jn}, {on}) -> row {ri}, vessel_id={vid}")

    vessels = nn_config.get("vessels", [])
    vessel_id_to_name = {v.get("vessel_id"): v.get("vessel_name", "") for v in vessels}

    for junc in nn_config.get("junctions", []):
        junc_name = junc.get("junction_name", "")

        if junc_name not in junction_name_to_row_indices:
            if junc.get("junction_type") == junction_type:
                if "junction_values" not in junc:
                    outlet_ids = junc.get("outlet_vessels", [])
                    num_outlets = len(outlet_ids)
                    junc["junction_values"] = {
                        "R_poiseuille": [0.0] * num_outlets,
                        "stenosis_coefficient": [0.0] * num_outlets,
                        "L": [0.0] * num_outlets,
                    }
            continue

        junc_outlet_vessel_ids = junc.get("outlet_vessels", [])
        if len(junc_outlet_vessel_ids) != 2:
            continue

        outlet_vessel_names = [vessel_id_to_name.get(vid, "") for vid in junc_outlet_vessel_ids]

        if "junction_values" not in junc:
            junc["junction_values"] = {}

        R_values = [0.0] * len(junc_outlet_vessel_ids)
        S_values = [0.0] * len(junc_outlet_vessel_ids)
        L_values = [0.0] * len(junc_outlet_vessel_ids)

        for file_idx, (vid, vname) in enumerate(zip(junc_outlet_vessel_ids, outlet_vessel_names, strict=False)):
            if "connector" in vname and "connectorEL" not in vname:
                print(f"        {junc_name}: outlet[{file_idx}] {vname} (id={vid}) -> connector, set to 0")
                continue

            row_idx = primary_outlet_to_row.get((junc_name, vname))

            if row_idx is not None:
                expected_vid = int(outlet_vessel_ids[row_idx])
                if expected_vid != vid:
                    raise ValueError(
                        f"Vessel ID mismatch for {junc_name}/{vname}: "
                        f"config has vessel_id={vid}, but feature row {row_idx} has "
                        f"outlet_vessel_id={expected_vid}"
                    )

                R_values[file_idx] = float(pred_R[row_idx])
                S_values[file_idx] = float(pred_S[row_idx])
                L_values[file_idx] = float(pred_L[row_idx])
                print(
                    f"        {junc_name}: outlet[{file_idx}] {vname} (id={vid}) -> "
                    f"row {row_idx}: R={R_values[file_idx]:.4f}, "
                    f"S={S_values[file_idx]:.4f}, L={L_values[file_idx]:.4f}"
                )
            else:
                raise ValueError(
                    f"No NN prediction row for {junc_name}/{vname} (vessel_id={vid}). "
                    f"Expected jax metadata row with primary outlet {vname!r}."
                )

        junc["junction_values"]["R_poiseuille"] = R_values
        junc["junction_values"]["stenosis_coefficient"] = S_values
        junc["junction_values"]["L"] = L_values


def run_junction_inference(
    *,
    jax_data_dict: dict[str, Any],
    nn_config: dict[str, Any],
    set_name: str,
    geo_name: str | None,
    model_dir: str,
    junction_type: str,
    quadratic_resistor: bool = False,
) -> None:
    """Load junction rows from jax dict, predict, and apply predictions to ``nn_config``."""
    from learn_lpns.data_processing.data_dict_from_csvs import load_junction_rows_from_jax_dict

    X, junction_names, outlet_primary_names, outlet_vessel_ids, feature_names = load_junction_rows_from_jax_dict(
        jax_data_dict, geo_name=geo_name
    )
    if len(X) == 0:
        raise ValueError("No junction rows in jax pickle for inference")

    unique_junction_names = list(set(junction_names))
    print(f"  Loaded {len(X)} feature rows for {len(unique_junction_names)} unique junctions")
    print(f"  Junction names: {unique_junction_names}")
    print(f"  Selected {len(feature_names)} features (matching training data): {feature_names}")
    print(f"  Neural network input dimensions: {X.shape} (rows={X.shape[0]}, features={X.shape[1]})")

    pred_R, pred_S, pred_L = run_nn_predict(X, model_dir, set_name, vessel=False)
    if not quadratic_resistor:
        pred_S = np.zeros_like(pred_R)

    apply_junction_predictions(
        nn_config,
        X=X,
        junction_names=junction_names,
        outlet_primary_names=outlet_primary_names,
        outlet_vessel_ids=outlet_vessel_ids,
        pred_R=pred_R,
        pred_S=pred_S,
        pred_L=pred_L,
        junction_type=junction_type,
    )


def validate_vessel_trial_geometry_variant(
    model_dir: str | None,
    geometry_variant: str,
) -> None:
    """Ensure CV trial model dir matches the requested geometry variant."""
    model_dir_basename = os.path.basename(model_dir or "")
    if "_trial_" not in model_dir_basename:
        return
    trial_model_variant = model_dir_basename.split("_trial_")[0]
    if geometry_variant != trial_model_variant:
        raise ValueError(
            f"Vessel NN trial model dir is for {trial_model_variant!r}, but geometry_variant is {geometry_variant!r}"
        )


def resolve_vessel_model_dir(
    *,
    set_name: str,
    geometry_variant: str,
    model_dir: str | None = None,
) -> str:
    """Resolve vessel model checkpoint directory (CV trial dirs use ``_vessel_trial_`` suffix)."""
    if model_dir and "_trial_" in os.path.basename(model_dir):
        return os.path.join(
            os.path.dirname(model_dir),
            os.path.basename(model_dir).replace("_trial_", "_vessel_trial_", 1),
        )
    return os.path.join("results", "models", set_name, f"{geometry_variant}_vessel")


def load_vessel_feature_matrix(
    geometric_input_path: str,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Load and filter vessel geometric features for NN inference."""
    from learn_lpns.data_processing.data_dict_from_csvs import (
        _clamp_tortuosity,
        filter_features_from_array,
        get_default_include_features_vessel,
    )
    from learn_lpns.data_processing.inputs_from_0d_config import load_vessel_geometric_features

    vessel_X, vessel_feature_names, vessel_ids, _vessel_names = load_vessel_geometric_features(
        geometric_input_path, verbose=verbose
    )
    if len(vessel_X) == 0:
        raise ValueError(f"No non-connector vessels for vessel NN ({geometric_input_path})")

    vessel_X, vessel_feature_names = filter_features_from_array(
        vessel_X,
        vessel_feature_names,
        include_features=get_default_include_features_vessel(),
    )
    _clamp_tortuosity(vessel_X, vessel_feature_names)
    return np.array(vessel_X, dtype=np.float64), vessel_ids


def apply_vessel_predictions(
    config: dict[str, Any],
    *,
    vessel_ids: list[int],
    pred_R: np.ndarray,
    pred_S: np.ndarray,
    pred_L: np.ndarray,
) -> None:
    """Write predicted R/S/L into non-connector vessels (mutates ``config`` in place)."""
    vessel_id_to_row = {vessel_id: i for i, vessel_id in enumerate(vessel_ids)}
    for vessel in config.get("vessels", []):
        vessel_name = (vessel.get("vessel_name") or "").lower()
        if "connector" in vessel_name:
            continue
        vessel_id = vessel.get("vessel_id")
        row = vessel_id_to_row.get(vessel_id)
        if row is None:
            continue
        z = dict(vessel.get("zero_d_element_values") or {})
        z["R_poiseuille"] = float(pred_R[row])
        z["stenosis_coefficient"] = float(pred_S[row])
        z["L"] = float(pred_L[row])
        vessel["zero_d_element_values"] = z


def run_vessel_inference(
    *,
    junction_nn_config: dict[str, Any],
    variant_geometric_input: str,
    set_name: str,
    geometry_variant: str,
    model_dir: str | None = None,
    quadratic_resistor: bool = False,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Predict vessel R/S/L and return (junction+vessel config, vessel-only config).

    ``junction_nn_config`` is copied for the junction+vessel output; vessel-only
    config is built from ``variant_geometric_input`` (geometric junctions).
    """
    validate_vessel_trial_geometry_variant(model_dir, geometry_variant)

    vessel_X, vessel_ids = load_vessel_feature_matrix(variant_geometric_input, verbose=verbose)
    vessel_model_dir = resolve_vessel_model_dir(
        set_name=set_name,
        geometry_variant=geometry_variant,
        model_dir=model_dir,
    )
    pred_R, pred_S, pred_L = run_nn_predict(vessel_X, vessel_model_dir, set_name, vessel=True)
    if not quadratic_resistor:
        pred_S = np.zeros_like(pred_R)

    junction_and_vessel_config = json.loads(json.dumps(junction_nn_config))
    apply_vessel_predictions(
        junction_and_vessel_config,
        vessel_ids=vessel_ids,
        pred_R=pred_R,
        pred_S=pred_S,
        pred_L=pred_L,
    )

    with open(variant_geometric_input) as f:
        vessel_only_config = json.load(f)
    apply_vessel_predictions(
        vessel_only_config,
        vessel_ids=vessel_ids,
        pred_R=pred_R,
        pred_S=pred_S,
        pred_L=pred_L,
    )

    return junction_and_vessel_config, vessel_only_config
