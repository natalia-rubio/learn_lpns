"""Neural-network inference: load jax cohort rows and write R/S/L predictions into 0D JSON."""

from __future__ import annotations

import json
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from learn_lpns.data_processing.rsl_nondim import redimensionalize_rsl_predictions
from learn_lpns.neural_network.activations import resolve_activation_from_model
from learn_lpns.neural_network.nn_model import RRI_NUM_OUTPUTS, predict
from learn_lpns.neural_network.nn_util import (
    clip_inputs_to_train_bounds,
    clip_rsl_predictions,
    dill_load,
    resolve_train_output_bounds,
)
from learn_lpns.zerod_calibration.stenosis_generation import (
    apply_stenosis_generation_gate,
    resolve_effective_stenosis_generation_limit,
    slice_jax_generation_for_geo,
)


def forward_jax_pickle_path(
    data_root: str,
    set_name: str,
    run_config_suffix: str,
    geometry_variant: str,
    geo_name: str,
    num_geos: int = 1,
    *,
    vessel: bool = False,
) -> str:
    """Path to per-forward-pass jax pickle under ``forward/<geo_name>/`` (set_type=forward)."""
    if not geo_name:
        raise ValueError("geo_name is required for forward jax pickle path")
    prefix = "jax_arrays_vessel" if vessel else "jax_arrays"
    return os.path.join(
        data_root,
        "jax_arrays",
        set_name,
        run_config_suffix,
        geometry_variant,
        "forward",
        geo_name,
        f"{prefix}_num_geos_{num_geos}.pkl",
    )


def forward_split_indices_path(
    data_root: str,
    set_name: str,
    run_config_suffix: str,
    geometry_variant: str,
    geo_name: str,
    num_geos: int = 1,
) -> str:
    """Path to split-indices pickle for a single-geometry forward data-processing run."""
    if not geo_name:
        raise ValueError("geo_name is required for forward split indices path")
    return os.path.join(
        data_root,
        "split_indices",
        set_name,
        run_config_suffix,
        geometry_variant,
        "forward",
        geo_name,
        f"train_val_ind_{set_name}_num_geos_{num_geos}",
    )


def rri_trained_coefficient_columns(*, quadratic_resistor: bool = True) -> list[int]:
    """Output columns trained as separate single-output nets (0=R, 1=S, 2=L)."""
    if quadratic_resistor:
        return [0, 1, 2]
    return [0, 2]


def rri_model_checkpoint_path(model_dir: str, set_name: str, *, vessel: bool = False) -> str:
    """Expected checkpoint path for a single multi-output R/S/L model."""
    base = "vessel_pred" if vessel else "pred"
    return os.path.join(model_dir, f"rri_{set_name}_{base}_rsl_model")


def rri_separate_model_checkpoint_paths(
    model_dir: str,
    set_name: str,
    *,
    vessel: bool = False,
    quadratic_resistor: bool = True,
) -> list[str]:
    """Expected checkpoint paths for single-output R/S/L models (S omitted when quadratic_resistor is off)."""
    base = "vessel_pred" if vessel else "pred"
    return [
        os.path.join(model_dir, f"rri_{set_name}_{base}_{col}_model")
        for col in rri_trained_coefficient_columns(quadratic_resistor=quadratic_resistor)
    ]


def rri_models_complete(
    model_dir: str,
    set_name: str,
    *,
    vessel: bool = False,
    multi_output: bool,
    quadratic_resistor: bool = True,
) -> bool:
    """Return True when all expected checkpoint files exist for the training mode."""
    if multi_output:
        return os.path.exists(rri_model_checkpoint_path(model_dir, set_name, vessel=vessel))
    return all(
        os.path.exists(p)
        for p in rri_separate_model_checkpoint_paths(
            model_dir,
            set_name,
            vessel=vessel,
            quadratic_resistor=quadratic_resistor,
        )
    )


def resolve_rri_model_paths(
    model_dir: str,
    set_name: str,
    *,
    vessel: bool = False,
    multi_output_rri: bool,
    quadratic_resistor: bool = True,
) -> list[str]:
    """Checkpoint paths for inference; must match how models were trained."""
    if multi_output_rri:
        return [rri_model_checkpoint_path(model_dir, set_name, vessel=vessel)]
    return rri_separate_model_checkpoint_paths(
        model_dir,
        set_name,
        vessel=vessel,
        quadratic_resistor=quadratic_resistor,
    )


def _resolve_multi_output_rri(multi_output_rri: bool | None) -> bool:
    if multi_output_rri is not None:
        return bool(multi_output_rri)
    from learn_lpns.config import get_pipeline_config

    return bool(get_pipeline_config().training.multi_output_rri)


def _resolve_clip_predictions(clip_predictions: bool | None, set_name: str) -> bool:
    if clip_predictions is not None:
        return bool(clip_predictions)
    from learn_lpns.config import get_pipeline_config

    return bool(get_pipeline_config(set_name=set_name).training.clip_predictions)


def _resolve_clip_input_features(
    clip_input_features: bool | None,
    set_name: str,
    run_config_suffix: str | None,
) -> bool:
    if clip_input_features is not None:
        return bool(clip_input_features)
    from learn_lpns.config import get_pipeline_config

    return bool(
        get_pipeline_config(
            set_name=set_name,
            run_config=run_config_suffix,
        ).training.clip_input_features
    )


def _maybe_redimensionalize_rsl_predictions(
    pred_R: np.ndarray,
    pred_S: np.ndarray,
    pred_L: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    model: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-dimensionalize R*/S*/L* predictions when the checkpoint was trained with nondim_rsl."""
    if not getattr(model, "nondim_rsl", False):
        return pred_R, pred_S, pred_L
    print(
        "      nondim_rsl: re-dimensionalizing predictions "
        f"(Re_c={model.reference_reynolds:g}, rho={model.nondim_rho:g}, mu={model.nondim_mu:g})"
    )
    return redimensionalize_rsl_predictions(
        pred_R,
        pred_S,
        pred_L,
        feature_matrix,
        feature_names,
        rho=float(model.nondim_rho),
        mu=float(model.nondim_mu),
        reference_reynolds=float(model.reference_reynolds),
    )


def run_nn_predict(
    X: np.ndarray,
    model_dir: str,
    set_name: str,
    *,
    vessel: bool = False,
    multi_output_rri: bool | None = None,
    quadratic_resistor: bool = True,
    clip_predictions: bool | None = None,
    clip_input_features: bool | None = None,
    run_config_suffix: str | None = None,
    data_root: str = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run NN heads for R, stenosis (when trained), and L; return prediction arrays."""
    use_multi_output = _resolve_multi_output_rri(multi_output_rri)
    model_paths = resolve_rri_model_paths(
        model_dir,
        set_name,
        vessel=vessel,
        multi_output_rri=use_multi_output,
        quadratic_resistor=quadratic_resistor,
    )
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

    X_jax = jnp.array(X, dtype=jnp.float32)
    clip_inputs = _resolve_clip_input_features(
        clip_input_features,
        set_name,
        run_config_suffix,
    )
    inputs_clipped = False
    bounds_model = None
    if len(model_paths) == 1:
        model_path = model_paths[0]
        print(f"      Loading multi-output model: {model_path}")
        model = dill_load(model_path)
        bounds_model = model
        if clip_inputs:
            clipped_X, changed_count = clip_inputs_to_train_bounds(np.asarray(X_jax), model)
            X_jax = jnp.asarray(clipped_X, dtype=jnp.float32)
            inputs_clipped = True
            print(
                "      clip_input_features: ON "
                f"({changed_count} value(s) clipped to train-set bounds)"
            )
        activation = resolve_activation_from_model(model)
        pred = np.array(predict(X_jax, model.weights, activation))
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if pred.shape[1] != RRI_NUM_OUTPUTS:
            raise ValueError(f"Expected {RRI_NUM_OUTPUTS} outputs from {model_path}, got shape {pred.shape}.")
        raw_predictions = [pred[:, i] for i in range(RRI_NUM_OUTPUTS)]
    else:
        coef_columns = rri_trained_coefficient_columns(quadratic_resistor=quadratic_resistor)
        preds_by_col: dict[int, np.ndarray] = {}
        for load_i, (col, model_path) in enumerate(zip(coef_columns, model_paths, strict=True)):
            print(f"      Loading model {load_i + 1}/{len(model_paths)} (column {col}): {model_path}")
            model = dill_load(model_path)
            if bounds_model is None:
                bounds_model = model
            if clip_inputs and not inputs_clipped:
                clipped_X, changed_count = clip_inputs_to_train_bounds(np.asarray(X_jax), model)
                X_jax = jnp.asarray(clipped_X, dtype=jnp.float32)
                inputs_clipped = True
                print(
                    "      clip_input_features: ON "
                    f"({changed_count} value(s) clipped to train-set bounds)"
                )
            activation = resolve_activation_from_model(model)
            pred = predict(X_jax, model.weights, activation)
            preds_by_col[col] = np.array(pred).flatten()
        n_rows = len(next(iter(preds_by_col.values())))
        raw_predictions = [
            preds_by_col.get(col, np.zeros(n_rows, dtype=np.float64)) for col in range(RRI_NUM_OUTPUTS)
        ]

    if _resolve_clip_predictions(clip_predictions, set_name):
        if bounds_model is None:
            raise ValueError("clip_predictions enabled but no model was loaded for bounds.")
        suffix = run_config_suffix or getattr(bounds_model, "run_config_suffix", None)
        root = getattr(bounds_model, "data_root", data_root)
        bounds_min, bounds_max = resolve_train_output_bounds(
            bounds_model,
            data_root=root,
            run_config_suffix=suffix,
        )
        pre_clip = [(float(np.min(p)), float(np.max(p))) for p in raw_predictions]
        raw_predictions = list(
            clip_rsl_predictions(
                raw_predictions[0],
                raw_predictions[1],
                raw_predictions[2],
                bounds_min,
                bounds_max,
            )
        )
        post_clip = [(float(np.min(p)), float(np.max(p))) for p in raw_predictions]
        print(
            "      clip_predictions: ON  "
            f"(R [{bounds_min[0]:.4g}, {bounds_max[0]:.4g}], "
            f"S [{bounds_min[1]:.4g}, {bounds_max[1]:.4g}], "
            f"L [{bounds_min[2]:.4g}, {bounds_max[2]:.4g}])"
        )
        print(
            "      clip_predictions: pred ranges before "
            f"R=[{pre_clip[0][0]:.4g},{pre_clip[0][1]:.4g}] "
            f"S=[{pre_clip[1][0]:.4g},{pre_clip[1][1]:.4g}] "
            f"L=[{pre_clip[2][0]:.4g},{pre_clip[2][1]:.4g}]"
        )
        print(
            "      clip_predictions: pred ranges after  "
            f"R=[{post_clip[0][0]:.4g},{post_clip[0][1]:.4g}] "
            f"S=[{post_clip[1][0]:.4g},{post_clip[1][1]:.4g}] "
            f"L=[{post_clip[2][0]:.4g},{post_clip[2][1]:.4g}]"
        )
    elif clip_predictions is False:
        print("      clip_predictions: OFF  (--no-clip_predictions)")

    output_names = ["R_poiseuille", "stenosis_coefficient", "L"]
    for coef_idx, pred in enumerate(raw_predictions):
        print(f"      {output_names[coef_idx]}: pred range=[{pred.min():.4f}, {pred.max():.4f}]")
    return (
        np.array(raw_predictions[0]),
        np.array(raw_predictions[1]),
        np.array(raw_predictions[2]),
        bounds_model,
    )


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
    multi_output_rri: bool | None = None,
    clip_predictions: bool | None = None,
    run_config_suffix: str | None = None,
    data_root: str = "data",
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

    pred_R, pred_S, pred_L, bounds_model = run_nn_predict(
        X,
        model_dir,
        set_name,
        vessel=False,
        multi_output_rri=multi_output_rri,
        quadratic_resistor=quadratic_resistor,
        clip_predictions=clip_predictions,
        run_config_suffix=run_config_suffix,
        data_root=data_root,
    )
    if not quadratic_resistor:
        pred_S = np.zeros_like(pred_R)
    else:
        limit_enabled, max_generation = resolve_effective_stenosis_generation_limit(
            set_name=set_name,
            quadratic_resistor=quadratic_resistor,
            vessel=False,
            bounds_model=bounds_model,
        )
        if limit_enabled:
            generation = slice_jax_generation_for_geo(
                jax_data_dict,
                geo_name=geo_name,
                n_expected=len(X),
            )
            pred_S = apply_stenosis_generation_gate(
                pred_S,
                generation,
                enabled=True,
                max_generation=max_generation,
                modality="junction",
            )

    pred_R, pred_S, pred_L = _maybe_redimensionalize_rsl_predictions(
        pred_R, pred_S, pred_L, X, feature_names, bounds_model
    )

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
) -> tuple[np.ndarray, list[int], np.ndarray, list[str]]:
    """Load vessel NN features, ids, per-row generation, and feature names for inference."""
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

    gen_idx = vessel_feature_names.index("generation")
    generation = np.asarray(vessel_X[:, gen_idx], dtype=float)

    vessel_X, vessel_feature_names = filter_features_from_array(
        vessel_X,
        vessel_feature_names,
        include_features=get_default_include_features_vessel(),
    )
    _clamp_tortuosity(vessel_X, vessel_feature_names)
    return np.array(vessel_X, dtype=np.float64), vessel_ids, generation, list(vessel_feature_names)


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
    multi_output_rri: bool | None = None,
    clip_predictions: bool | None = None,
    run_config_suffix: str | None = None,
    data_root: str = "data",
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Predict vessel R/S/L and return (junction+vessel config, vessel-only config).

    ``junction_nn_config`` is copied for the junction+vessel output; vessel-only
    config is built from ``variant_geometric_input`` (geometric junctions).
    """
    validate_vessel_trial_geometry_variant(model_dir, geometry_variant)

    vessel_X, vessel_ids, vessel_generation, vessel_feature_names = load_vessel_feature_matrix(
        variant_geometric_input, verbose=verbose
    )
    vessel_model_dir = resolve_vessel_model_dir(
        set_name=set_name,
        geometry_variant=geometry_variant,
        model_dir=model_dir,
    )
    pred_R, pred_S, pred_L, bounds_model = run_nn_predict(
        vessel_X,
        vessel_model_dir,
        set_name,
        vessel=True,
        multi_output_rri=multi_output_rri,
        quadratic_resistor=quadratic_resistor,
        clip_predictions=clip_predictions,
        run_config_suffix=run_config_suffix,
        data_root=data_root,
    )
    if not quadratic_resistor:
        pred_S = np.zeros_like(pred_R)
    else:
        limit_enabled, max_generation = resolve_effective_stenosis_generation_limit(
            set_name=set_name,
            quadratic_resistor=quadratic_resistor,
            vessel=True,
            bounds_model=bounds_model,
        )
        pred_S = apply_stenosis_generation_gate(
            pred_S,
            vessel_generation,
            enabled=limit_enabled,
            max_generation=max_generation,
            modality="vessel",
        )

    pred_R, pred_S, pred_L = _maybe_redimensionalize_rsl_predictions(
        pred_R, pred_S, pred_L, vessel_X, vessel_feature_names, bounds_model
    )

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
