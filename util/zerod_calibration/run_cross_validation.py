#!/usr/bin/env python3
"""
Cross-validation driver: X random 90/10 train/val splits by geometry.
Per trial: train NN on train geometries, deploy (NN-only generate_zerod_inputs) on
validation geometries, collect MSE for all modalities, aggregate per trial.
Writes results/cross_validation/{set_name}/{geometry_variant}_cv_summary.csv with
one row per trial and mean ± std per modality across trials.
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.data_processing.generate_split_indices import (
    build_geometry_index_map,
    build_split_dict,
    generate_geometry_split,
    get_geometry_row_ranges,
    load_split_for_training,
    resolve_flat_indices,
    resolve_geometry_row_ranges_from_jax_dict,
)
from util.data_processing.data_dict_from_csvs import get_default_include_features
from util.tools.basic import load_dict, save_dict
from util.zerod_calibration.post_processing import calculate_mse_between_3d_and_0d
from util.zerod_calibration.generate_zerod_inputs import get_run_config_suffix
from util.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    canonical_run_config_for_data_paths,
    run_config_includes_gen_loss,
    run_config_suffix_to_flags,
)
from util.zerod_calibration.batch_generate_zerod_inputs_vmr import get_vmr_geometries


# Recognized --run-config values. Add new configs here when introducing them.
ALLOWED_RUN_CONFIGS = frozenset({
    "base",
    "penalty_off",
    "penalty_off_gen_loss",
    "symmetric_penalty_off_gen_loss",
    "symmetric_penalty_off",
    "stenosis_off",
    "stenosis_off_symmetric",
    "stenosis_off_symmetric_gen_loss",
    "normalized",
    "normalized_penalty_off",
    "normalized_stenosis_off",
    "clip",
    "symmetric",
    "symmetric_gen_loss",
})


def _check_val_out_of_train_range(X, train_ind, val_ind, row_ranges, geometries, feature_names):
    """
    Check for validation rows where any feature is outside the min/max range of the training set.
    Returns a list of dicts with keys: geometry, row_in_geometry, feature_name, value, train_min, train_max, side.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        return []
    X_train = X[train_ind]
    train_min = np.min(X_train, axis=0)
    train_max = np.max(X_train, axis=0)
    out = []
    for r in val_ind:
        r = int(r)
        # Map flat index r to geometry and row within geometry
        g = None
        for gi, (s, e) in enumerate(row_ranges):
            if s <= r < e:
                g = gi
                break
        if g is None:
            continue
        geo_name = geometries[g]
        row_in_geo = r - row_ranges[g][0]
        for j, fname in enumerate(feature_names):
            v = float(X[r, j])
            tmin = float(train_min[j])
            tmax = float(train_max[j])
            if v < tmin:
                out.append({
                    "geometry": geo_name,
                    "row_in_geometry": row_in_geo,
                    "feature_name": fname,
                    "value": v,
                    "train_min": tmin,
                    "train_max": tmax,
                    "side": "below",
                })
            elif v > tmax:
                out.append({
                    "geometry": geo_name,
                    "row_in_geometry": row_in_geo,
                    "feature_name": fname,
                    "value": v,
                    "train_min": tmin,
                    "train_max": tmax,
                    "side": "above",
                })
    return out


def _parse_mse_csv(csv_path):
    """Parse MSE comparison CSV; return dict modality -> overall_mse (float or nan)."""
    extended = _parse_mse_csv_extended(csv_path)
    return {mod: data.get("overall_mse", np.nan) for mod, data in extended.items()}


def _parse_mse_csv_extended(csv_path):
    """Parse MSE comparison CSV; return dict modality -> dict of metric -> value.
    Metrics: overall_mse, mean_pressure_mse, mean_flow_mse,
            overall_max_error, mean_pressure_max_error, mean_flow_max_error,
            overall_max_rel_error, mean_pressure_max_rel_error, mean_flow_max_rel_error.
    """
    result = {}
    if not os.path.exists(csv_path):
        return result
    row_name_to_key = {
        "Overall MSE": "overall_mse",
        "Mean Pressure MSE": "mean_pressure_mse",
        "Mean Flow MSE": "mean_flow_mse",
        "Overall Max Error": "overall_max_error",
        "Mean Pressure Max Error": "mean_pressure_max_error",
        "Mean Flow Max Error": "mean_flow_max_error",
        "Overall Max Rel Error": "overall_max_rel_error",
        "Mean Pressure Max Rel Error": "mean_pressure_max_rel_error",
        "Mean Flow Max Rel Error": "mean_flow_max_rel_error",
    }
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        in_summary = False
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "Summary Statistics":
                in_summary = True
                continue
            if in_summary and header is None:
                header = row  # Metric, mod1, mod2, ...
                continue
            if in_summary and header is not None:
                metric_key = row_name_to_key.get(row[0])
                if metric_key is not None:
                    for i, mod in enumerate(header[1:], start=1):
                        if mod not in result:
                            result[mod] = {}
                        if i < len(row) and row[i].strip() not in ("", "N/A"):
                            try:
                                result[mod][metric_key] = float(row[i])
                            except ValueError:
                                result[mod][metric_key] = np.nan
                        else:
                            result[mod][metric_key] = np.nan
                if row[0] == "Detailed Results":
                    break
    return result


def _ensure_ml_inputs_and_jax_for_config(
    set_name,
    geometry_variant,
    run_config_suffix,
    data_root,
    set_type,
    normalize,
    stenosis_off,
    symmetric_loss,
    clip_predictions,
    penalty_off,
    geometries,
    no_redo=False,
):
    """Run batch_generate_zerod_inputs_vmr then run_data_processing with --run-config."""
    script_dir = os.path.dirname(__file__)
    batch_script = os.path.join(script_dir, "batch_generate_zerod_inputs_vmr.py")
    data_processing_script = os.path.join(os.path.dirname(script_dir), "data_processing", "run_data_processing.py")
    print(f"Run config {run_config_suffix}: ml_inputs/jax not found. Running batch generate (VMR) then data processing...")
    cmd_batch = [sys.executable, batch_script, "--set-name", set_name, "--geometries", *geometries]
    if no_redo:
        cmd_batch.append("--no-redo")
    if run_config_suffix:
        cmd_batch.extend(["--run-config", run_config_suffix])
    result = subprocess.run(cmd_batch, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"batch_generate_zerod_inputs_vmr failed (return code {result.returncode}). "
            "Fix the error above and re-run."
        )
    cmd_dp = [
        sys.executable, data_processing_script,
        "--set-name", set_name,
        "--geometry-variant", geometry_variant,
        "--run-config", run_config_suffix,
        "--geometries", *geometries,
    ]
    if normalize:
        cmd_dp.append("--normalize")
    result_dp = subprocess.run(cmd_dp, cwd=REPO_ROOT, text=True)
    if result_dp.returncode != 0:
        raise RuntimeError(
            f"run_data_processing failed with --run-config {run_config_suffix} (return code {result_dp.returncode}). "
            "Fix the error above and re-run."
        )
    print(f"  Done: ml_inputs and jax_arrays created for config {run_config_suffix}")


def run_cross_validation(
    set_name,
    geometry_variant,
    num_trials,
    data_root="data",
    set_type="all",
    ml_inputs_root=None,
    trial_index=None,
    normalize=False,
    nn_vessel=True,
    skip_training_if_exists=False,
    clip_predictions=False,
    stenosis_off=False,
    penalty_off=False,
    symmetric_loss=False,
    percent_train=0.9,
    no_redo=False,
    run_config_cli=None,
):
    if ml_inputs_root is None:
        ml_inputs_root = os.path.join(data_root, "ml_inputs")
    if stenosis_off and penalty_off:
        raise ValueError("Cannot use both --stenosis-off and --penalty-off.")

    norm_suffix = "_normalized" if normalize else ""
    run_config_suffix = get_run_config_suffix(
        normalize=normalize,
        stenosis_off=stenosis_off,
        symmetric_loss=symmetric_loss,
        clip_predictions=clip_predictions,
        penalty_off=penalty_off,
    )
    # All on-disk paths (zeroD, ml_inputs, jax, splits, models, CV) use the full CLI suffix when set,
    # e.g. stenosis_off_symmetric_gen_loss is its own tree (duplicate of physics vs ..._symmetric).
    data_paths_suffix = (
        run_config_cli.strip() if run_config_cli is not None else run_config_suffix
    )
    if run_config_cli is not None:
        cli_canon = canonical_run_config_for_data_paths(run_config_cli) or run_config_cli.strip()
        if cli_canon != run_config_suffix:
            raise ValueError(
                f"--run-config {run_config_cli!r} canonicalizes to {cli_canon!r}, "
                f"but flags from that config correspond to {run_config_suffix!r}."
            )
    if run_config_suffix or run_config_cli:
        print(f"Run config: flags->{run_config_suffix!r}, paths->{data_paths_suffix!r}")

    # When data_paths_suffix is set, get canonical geometry list from richter-0d; if any missing from ml_inputs or jax missing, run batch + data processing
    if data_paths_suffix:
        richter_dir = os.path.join(data_root, "zeroD", set_name, "richter-0d")
        try:
            _geometries = get_vmr_geometries(richter_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot get geometry list: {e} "
                "Ensure is only supported for sets with data/zeroD/<set_name>/richter-0d/."
            ) from e
        _num_geos = len(_geometries)
        ml_inputs_dir = os.path.join(data_root, "ml_inputs", set_name, data_paths_suffix, geometry_variant)
        _missing_geos = []
        for _geo in _geometries:
            _geo_dir = os.path.join(ml_inputs_dir, _geo)
            _gf = os.path.join(_geo_dir, "geometric_features.csv")
            _jp = os.path.join(_geo_dir, "junction_lumped_parameters.csv")
            if not os.path.exists(_gf) or not os.path.exists(_jp):
                _missing_geos.append(_geo)
        _jax_path = os.path.join(
            data_root, "jax_arrays", set_name, data_paths_suffix, geometry_variant, set_type,
            f"jax_arrays_num_geos_{_num_geos}{norm_suffix}.pkl",
        )
        _jax_missing = not os.path.exists(_jax_path)
        if _missing_geos or _jax_missing:
            if _missing_geos:
                print(f"Run config {data_paths_suffix}: missing ml_inputs for {len(_missing_geos)} geometries (e.g. {_missing_geos[:3]}{'...' if len(_missing_geos) > 3 else ''})")
            if _jax_missing:
                print(f"Run config {data_paths_suffix}: jax_arrays pkl not found")
            _ensure_ml_inputs_and_jax_for_config(
                set_name=set_name,
                geometry_variant=geometry_variant,
                run_config_suffix=data_paths_suffix,
                data_root=data_root,
                set_type=set_type,
                normalize=normalize,
                stenosis_off=stenosis_off,
                symmetric_loss=symmetric_loss,
                clip_predictions=clip_predictions,
                penalty_off=penalty_off,
                geometries=_geometries,
                no_redo=no_redo,
            )

    # Discover geometry folder count for jax filename; per-geometry row ranges come from the jax
    # pickle (geometry_row_ranges) when present so they match stacked rows even if geometric_features
    # line counts on disk drifted from a stale or partially rebuilt pickle.
    _, _, geometries_for_path = get_geometry_row_ranges(
        ml_inputs_root, set_name, geometry_variant, run_config_suffix=data_paths_suffix
    )
    num_geos = len(geometries_for_path)
    if num_geos == 0:
        raise ValueError(
            f"No geometries found under {ml_inputs_root}/{set_name}"
            + (f"/{data_paths_suffix}" if data_paths_suffix else "") + f"/{geometry_variant}"
        )

    if data_paths_suffix:
        jax_path = os.path.join(
            data_root, "jax_arrays", set_name, data_paths_suffix, geometry_variant, set_type,
            f"jax_arrays_num_geos_{num_geos}{norm_suffix}.pkl",
        )
    else:
        jax_path = os.path.join(
            data_root, "jax_arrays", set_name, geometry_variant, set_type,
            f"jax_arrays_num_geos_{num_geos}{norm_suffix}.pkl",
        )
    if not os.path.exists(jax_path):
        hint = ""
        if normalize:
            hint = (
                f" Generate it by running data processing with --normalize (and --run-config {data_paths_suffix!r} if using a config), e.g.: "
                f"python util/data_processing/run_data_processing.py --set-name {set_name} --geometry-variant {geometry_variant} --normalize"
                + (f" --run-config {data_paths_suffix}" if data_paths_suffix else "")
            )
        raise FileNotFoundError(
            f"Jax arrays not found: {jax_path} (expected {num_geos} geometries"
            + (" with normalization" if normalize else "") + ")." + hint
        )

    if data_paths_suffix:
        vessel_jax_path = os.path.join(
            data_root, "jax_arrays", set_name, data_paths_suffix, geometry_variant, set_type,
            f"jax_arrays_vessel_num_geos_{num_geos}{norm_suffix}.pkl",
        )
    else:
        vessel_jax_path = os.path.join(
            data_root, "jax_arrays", set_name, geometry_variant, set_type,
            f"jax_arrays_vessel_num_geos_{num_geos}{norm_suffix}.pkl",
        )

    data_dict = load_dict(jax_path)
    num_pts = int(np.asarray(data_dict["input"]).shape[0])
    row_ranges, total_rows, geometries = resolve_geometry_row_ranges_from_jax_dict(
        data_dict,
        ml_inputs_root,
        set_name,
        geometry_variant,
        run_config_suffix=data_paths_suffix,
    )
    num_geos = len(geometries)
    if total_rows != num_pts:
        raise ValueError(
            f"Row count mismatch: row_ranges sum={total_rows} vs jax num_pts={num_pts}. "
            "This usually means a stale jax pickle after ML CSVs changed, or a corrupt CSV "
            "(e.g. junction_lumped_parameters header vs data columns — check run_data_processing output). "
            f"Delete {jax_path} and re-run data processing for this set/variant/config. "
            "Pickles written by the current code store geometry_row_ranges so CV matches jax rows."
        )

    if data_paths_suffix:
        split_indices_dir = os.path.join(
            data_root, "split_indices", set_name, data_paths_suffix, geometry_variant, set_type
        )
    else:
        split_indices_dir = os.path.join(
            data_root, "split_indices", set_name, geometry_variant, set_type
        )
    if data_paths_suffix:
        model_dir_base = os.path.join("results", "models", set_name, data_paths_suffix)
    else:
        model_dir_base = os.path.join("results", "models", set_name)
    if data_paths_suffix:
        zero_d_base = os.path.join(data_root, "zeroD", set_name, data_paths_suffix)
    else:
        zero_d_base = os.path.join(data_root, "zeroD", set_name)
    script_dir = os.path.dirname(__file__)
    launch_training_script = os.path.join(
        os.path.dirname(script_dir), "neural_network", "launch_training.py"
    )

    prefix = "" if geometry_variant == "original" else f"{geometry_variant}_"
    mse_csv_name = f"{prefix}mse_comparison.csv" if prefix else "mse_comparison.csv"

    # Optionally run only one trial (0-based index)
    if trial_index is not None:
        if trial_index < 0 or trial_index >= num_trials:
            raise ValueError(
                f"trial_index must be in [0, {num_trials}), got {trial_index}"
            )
        trials_to_run = [trial_index]
        print(f"Re-running single trial {trial_index} (of {num_trials})")
    else:
        trials_to_run = list(range(num_trials))
    if normalize:
        print("Using normalized jax arrays and z-normalization for NN training/inference")
    if nn_vessel:
        print("NN-vessel: will train vessel NN per trial and include vessel-predicted modality in MSE")
    if symmetric_loss:
        print("Symmetric loss: overestimate weight = 1.0 for all models")
    if clip_predictions:
        print("Clip predictions: R/S/L will be clipped to training set min/max during deploy")
    if stenosis_off:
        print("Stenosis-off: calibrate_stenosis_coefficient=False, all stenosis set to 0, NN will not predict stenosis")
    if penalty_off:
        print("Penalty-off: L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient set to 0 during calibration")

    all_trial_results = []  # list of dicts: trial_id, val_geometries, mod -> overall_mse
    seen_val_sets = set()  # frozenset of val geometry names, to ensure each trial has a different val set

    for trial in trials_to_run:
        print(f"\n{'='*60}")
        print(f"CV Trial {trial + 1}/{num_trials}")
        print(f"{'='*60}")

        split_path = os.path.join(
            split_indices_dir,
            f"train_val_ind_{set_name}_num_geos_{num_geos}_trial_{trial}",
        )
        os.makedirs(split_indices_dir, exist_ok=True)

        # When re-running a single trial, reuse existing split if present so val set matches the summary table
        if trial_index is not None and os.path.exists(split_path):
            split_dict = load_split_for_training(split_path)
            train_geometries = split_dict["train_geometries"]
            val_geometries = split_dict["val_geometries"]
            train_ind = resolve_flat_indices(split_dict, "junction", "train", split_path=split_path)
            val_ind = resolve_flat_indices(split_dict, "junction", "val", split_path=split_path)
            print(f"  Split (reused from {os.path.basename(split_path)}): {len(train_geometries)} train, {len(val_geometries)} val -> {val_geometries}")
        else:
            # Ensure this trial's validation set is different from all previous trials
            max_attempts = 200
            train_geometries = []
            val_geometries = []
            for attempt in range(max_attempts):
                seed = trial * 1000 + attempt
                train_geometries, val_geometries = generate_geometry_split(
                    percent_train, seed, geometries
                )
                if not val_geometries:
                    if attempt == 0:
                        pct = int(round(percent_train * 100))
                        print(f"  Skipping trial {trial}: no validation geometries ({pct}% of {num_geos} rounded to all)")
                    break
                val_set = frozenset(val_geometries)
                if val_set not in seen_val_sets:
                    seen_val_sets.add(val_set)
                    break
                if attempt == max_attempts - 1:
                    pct_val = int(round((1 - percent_train) * 100))
                    pct_train = int(round(percent_train * 100))
                    raise RuntimeError(
                        f"Could not get a distinct validation set for trial {trial} after {max_attempts} attempts. "
                        f"Not enough geometries for {num_trials} unique {pct_train}/{pct_val} splits."
                    )
            if not val_geometries:
                continue

            vessel_data_dict = load_dict(vessel_jax_path) if os.path.exists(vessel_jax_path) else {}
            geometry_indices = build_geometry_index_map(data_dict, vessel_data_dict)
            split_dict = build_split_dict(
                train_geometries,
                val_geometries,
                geometry_indices,
                num_offsets=1,
                percent_train=percent_train,
                seed=int(seed),
                num_pts=num_pts,
            )
            save_dict(split_dict, split_path)
            train_ind = resolve_flat_indices(split_dict, "junction", "train", split_path=split_path)
            val_ind = resolve_flat_indices(split_dict, "junction", "val", split_path=split_path)
            print(f"  Split: {len(train_geometries)} train, {len(val_geometries)} val -> {val_geometries}")

        # Check for validation features outside training set range; write CSV per split
        feature_names = get_default_include_features()
        X_input = np.asarray(data_dict["input"])
        out_of_range = _check_val_out_of_train_range(
            X_input, train_ind, val_ind, row_ranges, geometries, feature_names
        )
        if data_paths_suffix:
            out_dir_cv = os.path.join("results", "cross_validation", set_name, data_paths_suffix)
        else:
            out_dir_cv = os.path.join("results", "cross_validation", set_name)
        os.makedirs(out_dir_cv, exist_ok=True)
        oor_csv = os.path.join(
            out_dir_cv,
            f"out_of_range_{geometry_variant}{norm_suffix}_trial_{trial}.csv",
        )
        with open(oor_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "geometry",
                    "row_in_geometry",
                    "feature_name",
                    "value",
                    "train_min",
                    "train_max",
                    "side",
                ]
            )
            for row in out_of_range:
                writer.writerow(
                    [
                        row["geometry"],
                        row["row_in_geometry"],
                        row["feature_name"],
                        row["value"],
                        row["train_min"],
                        row["train_max"],
                        row["side"],
                    ]
                )
        if out_of_range:
            print(f"  Out-of-range: {len(out_of_range)} validation feature(s) outside train range -> {oor_csv}")
        else:
            print(f"  Out-of-range: none -> {oor_csv}")

        model_dir = os.path.join(
            model_dir_base, f"{geometry_variant}{norm_suffix}_trial_{trial}"
        )
        os.makedirs(model_dir, exist_ok=True)

        # Junction model file names (must match launch_training / train_nn output)
        junction_model_files = [
            os.path.join(model_dir, f"rri_{set_name}_pred_{i}_model")
            for i in range(3)
        ]
        skip_junction = (
            skip_training_if_exists
            and all(os.path.exists(p) for p in junction_model_files)
        )

        # Train
        if skip_junction:
            print(f"  Skipping junction training (models already exist in {model_dir})")
        else:
            cmd_train = [
                sys.executable,
                launch_training_script,
                set_name,
                str(num_geos),
                geometry_variant,
                "--split-path",
                split_path,
                "--model-dir",
                model_dir,
            ]
            if normalize:
                cmd_train.append("--normalize")
            if symmetric_loss:
                cmd_train.append("--symmetric-loss")
            launch_rc = run_config_cli if run_config_cli is not None else run_config_suffix
            if launch_rc:
                cmd_train.extend(["--run-config", launch_rc])
            print(f"  Running: {' '.join(cmd_train)}")
            result_train = subprocess.run(cmd_train, cwd=REPO_ROOT, text=True)
            if result_train.returncode != 0:
                print(f"  Training failed with return code {result_train.returncode}")
                all_trial_results.append(
                    {"trial_id": trial, "val_geometries": ",".join(val_geometries), "error": "training_failed"}
                )
                continue

        # Train vessel NN for this trial (same split) if requested
        if nn_vessel:
            vessel_model_dir = os.path.join(
                model_dir_base, f"{geometry_variant}_vessel{norm_suffix}_trial_{trial}"
            )
            vessel_model_files = [
                os.path.join(vessel_model_dir, f"rri_{set_name}_vessel_pred_{i}_model")
                for i in range(3)
            ]
            skip_vessel = (
                skip_training_if_exists
                and all(os.path.exists(p) for p in vessel_model_files)
            )
            if skip_vessel:
                print(f"  Skipping vessel training (models already exist in {vessel_model_dir})")
            else:
                cmd_vessel = [
                    sys.executable,
                    launch_training_script,
                    set_name,
                    str(num_geos),
                    geometry_variant,
                    "--vessel",
                    "--split-path",
                    split_path,
                    "--model-dir",
                    vessel_model_dir,
                ]
                if normalize:
                    cmd_vessel.append("--normalize")
                if symmetric_loss:
                    cmd_vessel.append("--symmetric-loss")
                launch_rc = run_config_cli if run_config_cli is not None else run_config_suffix
                if launch_rc:
                    cmd_vessel.extend(["--run-config", launch_rc])
                print(f"  Running vessel training: {' '.join(cmd_vessel)}")
                result_vessel = subprocess.run(cmd_vessel, cwd=REPO_ROOT, text=True)
                if result_vessel.returncode != 0:
                    print(f"  Vessel training failed with return code {result_vessel.returncode}")
                    all_trial_results.append(
                        {"trial_id": trial, "val_geometries": ",".join(val_geometries), "error": "vessel_training_failed"}
                    )
                    continue

        # Deploy on each validation geometry (NN-only)
        trial_metrics = {}  # modality -> dict metric_key -> list of values per val geo
        for val_geo in val_geometries:
            cmd_deploy = [
                sys.executable,
                os.path.join(script_dir, "generate_zerod_inputs.py"),
                "--set-name",
                set_name,
                "--geo-name",
                val_geo,
                "--NN-only",
                "--model-dir",
                model_dir,
                "--trial-id",
                str(trial),
            ]
            if normalize:
                cmd_deploy.append("--normalize")
                cmd_deploy.append("--norm-data-path")
                cmd_deploy.append(jax_path)
                if nn_vessel and vessel_jax_path and os.path.exists(vessel_jax_path):
                    cmd_deploy.append("--vessel-norm-data-path")
                    cmd_deploy.append(vessel_jax_path)
            if nn_vessel:
                cmd_deploy.append("--NN-vessel")
            if clip_predictions:
                cmd_deploy.append("--clip-predictions")
            if stenosis_off:
                cmd_deploy.append("--stenosis-off")
            if penalty_off:
                cmd_deploy.append("--penalty-off")
            if symmetric_loss:
                cmd_deploy.append("--symmetric-loss")
            deploy_rc = run_config_cli if run_config_cli is not None else run_config_suffix
            if deploy_rc and run_config_includes_gen_loss(deploy_rc):
                cmd_deploy.append("--gen-loss")
            if deploy_rc:
                cmd_deploy.extend(["--run-config", deploy_rc])
            print(f"  Deploy on {val_geo}: {' '.join(cmd_deploy)}")
            result_deploy = subprocess.run(cmd_deploy, cwd=REPO_ROOT, text=True)
            if result_deploy.returncode != 0:
                print(f"  Deploy failed for {val_geo} with return code {result_deploy.returncode}")
                continue
            mse_csv_path = os.path.join(zero_d_base, val_geo, mse_csv_name)
            mod_data = _parse_mse_csv_extended(mse_csv_path)
            for mod, metrics in mod_data.items():
                if mod not in trial_metrics:
                    trial_metrics[mod] = {k: [] for k in ("overall_mse", "mean_pressure_mse", "mean_flow_mse", "overall_max_error", "mean_pressure_max_error", "mean_flow_max_error", "overall_max_rel_error", "mean_pressure_max_rel_error", "mean_flow_max_rel_error")}
                for k, v in metrics.items():
                    if k in trial_metrics[mod]:
                        trial_metrics[mod][k].append(v)

        # Aggregate per trial (mean across val geometries per modality)
        row = {"trial_id": trial, "val_geometries": ",".join(val_geometries)}
        for mod, data in trial_metrics.items():
            row[f"MSE_{mod}"] = np.nanmean(data["overall_mse"]) if data["overall_mse"] else np.nan
            row[f"PressureMSE_{mod}"] = np.nanmean(data["mean_pressure_mse"]) if data["mean_pressure_mse"] else np.nan
            row[f"FlowMSE_{mod}"] = np.nanmean(data["mean_flow_mse"]) if data["mean_flow_mse"] else np.nan
            row[f"MaxError_{mod}"] = np.nanmean(data["overall_max_error"]) if data["overall_max_error"] else np.nan
            row[f"PressureMaxError_{mod}"] = np.nanmean(data["mean_pressure_max_error"]) if data["mean_pressure_max_error"] else np.nan
            row[f"FlowMaxError_{mod}"] = np.nanmean(data["mean_flow_max_error"]) if data["mean_flow_max_error"] else np.nan
            row[f"MaxRelError_{mod}"] = np.nanmean(data["overall_max_rel_error"]) if data.get("overall_max_rel_error") else np.nan
            row[f"PressureMaxRelError_{mod}"] = np.nanmean(data["mean_pressure_max_rel_error"]) if data.get("mean_pressure_max_rel_error") else np.nan
            row[f"FlowMaxRelError_{mod}"] = np.nanmean(data["mean_flow_max_rel_error"]) if data.get("mean_flow_max_rel_error") else np.nan
        all_trial_results.append(row)

    if not all_trial_results:
        print("No trial results to summarize.")
        return

    # Build summary: one row per trial + final mean ± std per modality
    # Collect modality names (without prefix) from all metric types
    modalities = set()
    for r in all_trial_results:
        for k in r:
            if k.startswith("MSE_") and k != "MSE_":
                modalities.add(k[4:])  # strip "MSE_"
                break
            if k.startswith("PressureMSE_"):
                modalities.add(k[len("PressureMSE_"):])
                break
            if k.startswith("FlowMSE_"):
                modalities.add(k[len("FlowMSE_"):])
                break
            if k.startswith("MaxError_"):
                modalities.add(k[len("MaxError_"):])
                break
            if k.startswith("PressureMaxError_"):
                modalities.add(k[len("PressureMaxError_"):])
                break
            if k.startswith("FlowMaxError_"):
                modalities.add(k[len("FlowMaxError_"):])
                break
            if k.startswith("MaxRelError_"):
                modalities.add(k[len("MaxRelError_"):])
                break
            if k.startswith("PressureMaxRelError_"):
                modalities.add(k[len("PressureMaxRelError_"):])
                break
            if k.startswith("FlowMaxRelError_"):
                modalities.add(k[len("FlowMaxRelError_"):])
                break
    modalities = sorted(modalities)

    if data_paths_suffix:
        out_dir = os.path.join("results", "cross_validation", set_name, data_paths_suffix)
    else:
        out_dir = os.path.join("results", "cross_validation", set_name)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(
        out_dir, f"{geometry_variant}{norm_suffix}_cv_summary.csv"
    )

    # If we re-ran a single trial and summary already exists, merge this result into it
    if trial_index is not None and os.path.exists(summary_path):
        existing_by_trial = {}  # trial_id -> dict (same shape as all_trial_results items)
        with open(summary_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header and header[0] == "trial_id":
                modalities_existing = header[2:]
                for row in reader:
                    if not row or row[0] in ("", "mean", "std"):
                        break
                    try:
                        tid = int(row[0])
                    except ValueError:
                        break
                    existing_by_trial[tid] = {"trial_id": tid, "val_geometries": row[1] if len(row) > 1 else ""}
                    for i, mod in enumerate(modalities_existing):
                        if i + 2 < len(row) and row[i + 2].strip() != "":
                            try:
                                existing_by_trial[tid][mod] = float(row[i + 2])
                            except ValueError:
                                existing_by_trial[tid][mod] = np.nan
        # Merge: update or add this trial
        for r in all_trial_results:
            existing_by_trial[r["trial_id"]] = r
        # Recompute mean/std over all trials present in the file
        all_trial_results = [existing_by_trial[tid] for tid in sorted(existing_by_trial)]

    # Ensure we have full modality list from all keys in results
    def _collect_modalities(results):
        out = set()
        for r in results:
            if "error" in r:
                continue
            for k in r:
                if k.startswith("MSE_") and len(k) > 4:
                    out.add(k[4:])
                elif k.startswith("PressureMSE_"):
                    out.add(k[len("PressureMSE_"):])
                elif k.startswith("FlowMSE_"):
                    out.add(k[len("FlowMSE_"):])
                elif k.startswith("MaxError_"):
                    out.add(k[len("MaxError_"):])
                elif k.startswith("PressureMaxError_"):
                    out.add(k[len("PressureMaxError_"):])
                elif k.startswith("FlowMaxError_"):
                    out.add(k[len("FlowMaxError_"):])
                elif k.startswith("MaxRelError_"):
                    out.add(k[len("MaxRelError_"):])
                elif k.startswith("PressureMaxRelError_"):
                    out.add(k[len("PressureMaxRelError_"):])
                elif k.startswith("FlowMaxRelError_"):
                    out.add(k[len("FlowMaxRelError_"):])
        return sorted(out)

    modalities = _collect_modalities(all_trial_results)
    modalities_mse = modalities  # same set for all metric types

    def _write_metric_summary_csv(path, prefix, modalities_list):
        """Write a CV summary CSV for one metric (prefix + modality columns)."""
        cols = [prefix + m for m in modalities_list]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "val_geometries"] + cols)
            for r in all_trial_results:
                if "error" in r:
                    writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [""] * len(cols))
                else:
                    writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [r.get(c, "") for c in cols])
            writer.writerow([])
            mean_row = ["mean", ""]
            std_row = ["std", ""]
            for c in cols:
                vals = [r.get(c) for r in all_trial_results if "error" not in r and c in r]
                vals = [float(v) for v in vals if v is not None and v != "" and (not isinstance(v, float) or not np.isnan(v))]
                mean_row.append(np.nanmean(vals) if vals else np.nan)
                std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
            writer.writerow(mean_row)
            writer.writerow(std_row)

    # Main overall MSE summary (unchanged)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trial_id", "val_geometries"] + [f"MSE_{m}" for m in modalities]
        writer.writerow(header)
        for r in all_trial_results:
            if "error" in r:
                writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [""] * len(modalities))
            else:
                writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [r.get(f"MSE_{m}", "") for m in modalities])
        writer.writerow([])
        mean_row = ["mean", ""]
        std_row = ["std", ""]
        for m in modalities:
            vals = [r.get(f"MSE_{m}") for r in all_trial_results if "error" not in r and f"MSE_{m}" in r]
            vals = [float(v) for v in vals if v is not None and v != "" and (not isinstance(v, float) or not np.isnan(v))]
            mean_row.append(np.nanmean(vals) if vals else np.nan)
            std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
        writer.writerow(mean_row)
        writer.writerow(std_row)

    # Pressure MSE, Flow MSE, Max Error, Pressure Max Error, Flow Max Error
    base = os.path.join(out_dir, f"{geometry_variant}{norm_suffix}_cv_summary")
    _write_metric_summary_csv(base + "_pressure_mse.csv", "PressureMSE_", modalities)
    _write_metric_summary_csv(base + "_flow_mse.csv", "FlowMSE_", modalities)
    _write_metric_summary_csv(base + "_max_error.csv", "MaxError_", modalities)
    _write_metric_summary_csv(base + "_pressure_max_error.csv", "PressureMaxError_", modalities)
    _write_metric_summary_csv(base + "_flow_max_error.csv", "FlowMaxError_", modalities)
    _write_metric_summary_csv(base + "_max_rel_error.csv", "MaxRelError_", modalities)
    _write_metric_summary_csv(base + "_pressure_max_rel_error.csv", "PressureMaxRelError_", modalities)
    _write_metric_summary_csv(base + "_flow_max_rel_error.csv", "FlowMaxRelError_", modalities)

    print(f"\nWrote CV summary to {summary_path}")
    print(f"  Also wrote: pressure_mse, flow_mse, max_error, pressure_max_error, flow_max_error, max_rel_error, pressure_max_rel_error, flow_max_rel_error")
    return summary_path


def regenerate_cv_metrics_from_existing(
    set_name,
    geometry_variant,
    data_root="data",
    normalize=False,
    run_config_suffix="",
):
    """
    Regenerate CV summary CSVs (overall MSE + pressure/flow MSE + max error variants)
    by reading the existing cv_summary.csv and per-geometry mse_comparison.csv files.
    Does not re-run training or deploy. Use this to add the new metric CSVs after
    a previous CV run, or to refresh metrics if per-geometry MSE CSVs were updated
    (e.g. after re-running MSE calculation with max-error support).

    run_config_suffix: same as used when running CV (e.g. "normalized_stenosis_off_symmetric")
      so that zeroD and results paths match the run that produced the data.

    Requires:
      - results/cross_validation/{set_name}/{run_config_suffix}/{geometry_variant}_cv_summary.csv (or no run_config_suffix)
      - For each (trial, val_geo): data/zeroD/{set_name}/{run_config_suffix}/{val_geo}/... (or no run_config_suffix)
    """
    norm_suffix = "_normalized" if normalize else ""
    if run_config_suffix:
        out_dir = os.path.join("results", "cross_validation", set_name, run_config_suffix)
        zero_d_base = os.path.join(data_root, "zeroD", set_name, run_config_suffix)
    else:
        out_dir = os.path.join("results", "cross_validation", set_name)
        zero_d_base = os.path.join(data_root, "zeroD", set_name)
    summary_path = os.path.join(
        out_dir, f"{geometry_variant}{norm_suffix}_cv_summary.csv"
    )
    if not os.path.exists(summary_path):
        print(f"Existing CV summary not found: {summary_path}")
        print("Run cross-validation once to create it, then use --metrics-only to add new metric CSVs.")
        return None
    prefix = "" if geometry_variant == "original" else f"{geometry_variant}_"
    mse_csv_name = f"{prefix}mse_comparison.csv" if prefix else "mse_comparison.csv"

    # First pass: read summary to get (trial_id, val_geometries) and collect unique val_geometries
    summary_rows = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "trial_id":
            print(f"Unexpected header in {summary_path}")
            return None
        for row in reader:
            if not row or row[0] in ("", "mean", "std"):
                break
            try:
                trial_id = int(row[0])
            except ValueError:
                break
            val_geometries_str = row[1] if len(row) > 1 else ""
            val_geometries = [g.strip() for g in val_geometries_str.split(",") if g.strip()]
            summary_rows.append((trial_id, val_geometries_str, val_geometries))

    if not summary_rows:
        print("No trial rows found in existing summary.")
        return None

    # Re-run MSE calculation for each validation geometry so mse_comparison.csv has new rows (max error, etc.)
    all_val_geos = sorted(set(g for _, _, vg in summary_rows for g in vg))
    print(f"Re-running MSE calculation for {len(all_val_geos)} validation geometries...")
    for val_geo in all_val_geos:
        base_dir = os.path.join(zero_d_base, val_geo)
        calibration_input = os.path.join(base_dir, f"{geometry_variant}_calibration_input_BloodVesselJunction.json")
        if not os.path.exists(calibration_input):
            print(f"  Skip {val_geo}: calibration input not found")
            continue
        geometric_input = os.path.join(base_dir, f"{geometry_variant}_geometric_input.json")
        csv_results_dict = {}
        geometric_results = os.path.join(base_dir, f"{geometry_variant}_geometric_results.csv")
        if os.path.exists(geometric_results):
            csv_results_dict["geometric"] = geometric_results
        for jtype in ("BloodVesselJunction", "NORMAL_JUNCTION"):
            calib_results = os.path.join(base_dir, f"{geometry_variant}_calibrated_results_{jtype}.csv")
            if os.path.exists(calib_results):
                csv_results_dict[jtype] = calib_results
        nn_results = os.path.join(base_dir, f"{geometry_variant}_NN_BloodVesselJunction_results.csv")
        if os.path.exists(nn_results):
            csv_results_dict["BloodVesselJunction_NN"] = nn_results
        nn_jv_results = os.path.join(base_dir, f"{geometry_variant}_NN_JunctionAndVessel_results.csv")
        if os.path.exists(nn_jv_results):
            csv_results_dict["BloodVesselJunction_NN_plus_Vessel_NN"] = nn_jv_results
        nn_vessel_results = os.path.join(base_dir, f"{geometry_variant}_NN_VesselOnly_results.csv")
        if os.path.exists(nn_vessel_results):
            csv_results_dict["NN_vessel"] = nn_vessel_results
        if not csv_results_dict:
            print(f"  Skip {val_geo}: no result CSVs found")
            continue
        mse_csv_path = os.path.join(base_dir, mse_csv_name)
        try:
            calculate_mse_between_3d_and_0d(
                calibration_input,
                csv_results_dict,
                geometric_input_path=geometric_input if os.path.exists(geometric_input) else None,
                zoom_start_idx=None,
                zoom_end_idx=None,
                output_csv_path=mse_csv_path,
                verbose=False,
                set_name=set_name,
            )
            print(f"  ✓ {val_geo}")
        except Exception as e:
            print(f"  ✗ {val_geo}: {e}")

    # Second pass: parse per-geometry MSE CSVs and aggregate per trial
    all_trial_results = []
    for trial_id, val_geometries_str, val_geometries in summary_rows:
        trial_metrics = {}
        for val_geo in val_geometries:
            mse_csv_path = os.path.join(zero_d_base, val_geo, mse_csv_name)
            mod_data = _parse_mse_csv_extended(mse_csv_path)
            for mod, metrics in mod_data.items():
                if mod not in trial_metrics:
                    trial_metrics[mod] = {
                        k: [] for k in (
                            "overall_mse", "mean_pressure_mse", "mean_flow_mse",
                            "overall_max_error", "mean_pressure_max_error", "mean_flow_max_error",
                            "overall_max_rel_error", "mean_pressure_max_rel_error", "mean_flow_max_rel_error",
                        )
                    }
                for k, v in metrics.items():
                    if k in trial_metrics[mod]:
                        trial_metrics[mod][k].append(v)

        row_data = {"trial_id": trial_id, "val_geometries": val_geometries_str}
        for mod, data in trial_metrics.items():
            row_data[f"MSE_{mod}"] = np.nanmean(data["overall_mse"]) if data["overall_mse"] else np.nan
            row_data[f"PressureMSE_{mod}"] = np.nanmean(data["mean_pressure_mse"]) if data["mean_pressure_mse"] else np.nan
            row_data[f"FlowMSE_{mod}"] = np.nanmean(data["mean_flow_mse"]) if data["mean_flow_mse"] else np.nan
            row_data[f"MaxError_{mod}"] = np.nanmean(data["overall_max_error"]) if data["overall_max_error"] else np.nan
            row_data[f"PressureMaxError_{mod}"] = np.nanmean(data["mean_pressure_max_error"]) if data["mean_pressure_max_error"] else np.nan
            row_data[f"FlowMaxError_{mod}"] = np.nanmean(data["mean_flow_max_error"]) if data["mean_flow_max_error"] else np.nan
            row_data[f"MaxRelError_{mod}"] = np.nanmean(data["overall_max_rel_error"]) if data.get("overall_max_rel_error") else np.nan
            row_data[f"PressureMaxRelError_{mod}"] = np.nanmean(data["mean_pressure_max_rel_error"]) if data.get("mean_pressure_max_rel_error") else np.nan
            row_data[f"FlowMaxRelError_{mod}"] = np.nanmean(data["mean_flow_max_rel_error"]) if data.get("mean_flow_max_rel_error") else np.nan
        all_trial_results.append(row_data)

    if not all_trial_results:
        print("No trial rows found in existing summary.")
        return None

    def _collect_modalities(results):
        out = set()
        for r in results:
            if "error" in r:
                continue
            for k in r:
                if k.startswith("MSE_") and len(k) > 4:
                    out.add(k[4:])
                elif k.startswith("PressureMSE_"):
                    out.add(k[len("PressureMSE_"):])
                elif k.startswith("FlowMSE_"):
                    out.add(k[len("FlowMSE_"):])
                elif k.startswith("MaxError_"):
                    out.add(k[len("MaxError_"):])
                elif k.startswith("PressureMaxError_"):
                    out.add(k[len("PressureMaxError_"):])
                elif k.startswith("FlowMaxError_"):
                    out.add(k[len("FlowMaxError_"):])
                elif k.startswith("MaxRelError_"):
                    out.add(k[len("MaxRelError_"):])
                elif k.startswith("PressureMaxRelError_"):
                    out.add(k[len("PressureMaxRelError_"):])
                elif k.startswith("FlowMaxRelError_"):
                    out.add(k[len("FlowMaxRelError_"):])
        return sorted(out)

    modalities = _collect_modalities(all_trial_results)

    def _write_metric_summary_csv(path, prefix, modalities_list):
        cols = [prefix + m for m in modalities_list]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "val_geometries"] + cols)
            for r in all_trial_results:
                if "error" in r:
                    writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [""] * len(cols))
                else:
                    writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [r.get(c, "") for c in cols])
            writer.writerow([])
            mean_row = ["mean", ""]
            std_row = ["std", ""]
            for c in cols:
                vals = [r.get(c) for r in all_trial_results if "error" not in r and c in r]
                vals = [float(v) for v in vals if v is not None and v != "" and (not isinstance(v, float) or not np.isnan(v))]
                mean_row.append(np.nanmean(vals) if vals else np.nan)
                std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
            writer.writerow(mean_row)
            writer.writerow(std_row)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{geometry_variant}{norm_suffix}_cv_summary")

    # Overwrite main summary from aggregated data
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "val_geometries"] + [f"MSE_{m}" for m in modalities])
        for r in all_trial_results:
            writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [r.get(f"MSE_{m}", "") for m in modalities])
        writer.writerow([])
        mean_row = ["mean", ""]
        std_row = ["std", ""]
        for m in modalities:
            vals = [r.get(f"MSE_{m}") for r in all_trial_results if f"MSE_{m}" in r]
            vals = [float(v) for v in vals if v is not None and v != "" and (not isinstance(v, float) or not np.isnan(v))]
            mean_row.append(np.nanmean(vals) if vals else np.nan)
            std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
        writer.writerow(mean_row)
        writer.writerow(std_row)

    _write_metric_summary_csv(base + "_pressure_mse.csv", "PressureMSE_", modalities)
    _write_metric_summary_csv(base + "_flow_mse.csv", "FlowMSE_", modalities)
    _write_metric_summary_csv(base + "_max_error.csv", "MaxError_", modalities)
    _write_metric_summary_csv(base + "_pressure_max_error.csv", "PressureMaxError_", modalities)
    _write_metric_summary_csv(base + "_flow_max_error.csv", "FlowMaxError_", modalities)
    _write_metric_summary_csv(base + "_max_rel_error.csv", "MaxRelError_", modalities)
    _write_metric_summary_csv(base + "_pressure_max_rel_error.csv", "PressureMaxRelError_", modalities)
    _write_metric_summary_csv(base + "_flow_max_rel_error.csv", "FlowMaxRelError_", modalities)

    print(f"Regenerated CV metrics from existing per-geometry MSE CSVs.")
    print(f"  Updated: {summary_path}")
    print(f"  Wrote:   {base}_pressure_mse.csv, _flow_mse.csv, _max_error.csv, _pressure_max_error.csv, _flow_max_error.csv, _max_rel_error.csv, _pressure_max_rel_error.csv, _flow_max_rel_error.csv")
    has_max = any(
        r.get(f"MaxError_{m}") is not None and r.get(f"MaxError_{m}") != "" and not (isinstance(r.get(f"MaxError_{m}"), float) and np.isnan(r.get(f"MaxError_{m}")))
        for r in all_trial_results for m in modalities if f"MaxError_{m}" in r
    )
    if not has_max:
        print("  Note: Max-error columns are empty; per-geometry mse_comparison.csv files may lack the new rows. Re-run MSE calculation for each geometry to populate them.")
    return summary_path


def regenerate_location_plots(
    set_name,
    geometry_variant,
    data_root="data",
    run_config_suffix="",
    stenosis_off=False,
    normalize=False,
    penalty_off=False,
    symmetric_loss=False,
    clip_predictions=False,
):
    """
    Regenerate location comparison plots from existing zeroD data by running
    generate_zerod_inputs with skip flags so only Step 6 (plots) runs.
    No training or deploy. Use after a CV run to refresh plots.

    run_config_suffix: same as used when running CV (e.g. "stenosis_off")
      so that zeroD and plot output paths match.

    Requires: data/zeroD/{set_name}/{run_config_suffix}/{geo}/... with
      {geometry_variant}_calibration_input_BloodVesselJunction.json or
      calibration_input.json per geometry.
    """
    script_dir = os.path.dirname(__file__)
    generate_script = os.path.join(script_dir, "generate_zerod_inputs.py")
    if run_config_suffix:
        zero_d_base = os.path.join(data_root, "zeroD", set_name, run_config_suffix)
    else:
        zero_d_base = os.path.join(data_root, "zeroD", set_name)
    if not os.path.isdir(zero_d_base):
        print(f"ZeroD base not found: {zero_d_base}")
        print("Run cross-validation once to create zeroD data, then use --plots-only to regenerate plots.")
        return None

    # Discover geometries: subdirs that have calibration input for this variant
    calib_name = f"{geometry_variant}_calibration_input_BloodVesselJunction.json"
    geometries = []
    for name in sorted(os.listdir(zero_d_base)):
        base_dir = os.path.join(zero_d_base, name)
        if not os.path.isdir(base_dir):
            continue
        calib_path = os.path.join(base_dir, calib_name)
        fallback = os.path.join(base_dir, "calibration_input.json")
        if os.path.exists(calib_path) or os.path.exists(fallback):
            geometries.append(name)
    if not geometries:
        print(f"No geometries with calibration input found under {zero_d_base}")
        return None

    print(f"Regenerating location plots for {len(geometries)} geometries...")
    success_count = 0
    for geo_name in geometries:
        cmd = [
            sys.executable,
            generate_script,
            "--set-name",
            set_name,
            "--geo-name",
            geo_name,
            "--skip-base-generation",
            "--skip-observation",
            "--skip-calibration",
            "--skip-forward",
            "--skip-mse-calculation",
        ]
        if stenosis_off:
            cmd.append("--stenosis-off")
        if normalize:
            cmd.append("--normalize")
        if penalty_off:
            cmd.append("--penalty-off")
        if symmetric_loss:
            cmd.append("--symmetric-loss")
        if clip_predictions:
            cmd.append("--clip-predictions")
        if run_config_suffix:
            cmd.extend(["--run-config", run_config_suffix])
        # Include NN modalities (Learned Junctions, Learned Junctions and Vessels, Learned Vessels) in plots when CSVs exist
        cmd.append("--NN-vessel")
        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
        if result.returncode == 0:
            print(f"  ✓ {geo_name}")
            success_count += 1
        else:
            print(f"  ✗ {geo_name} (exit code {result.returncode})")
    print(f"Done: {success_count}/{len(geometries)} geometries.")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-validation: X random 90/10 splits, train and deploy per trial, report MSE for all modalities."
    )
    parser.add_argument("set_name", help="Set name (e.g., VMR_rigid_aorta_adults)")
    parser.add_argument(
        "geometry_variant",
        default="bifurcations_EL",
        nargs="?",
        help="Geometry variant (default: bifurcations_EL)",
    )
    parser.add_argument(
        "num_trials",
        type=int,
        nargs="?",
        default=5,
        help="Number of random 90/10 splits (default: 5)",
    )
    parser.add_argument("--data-root", default="data", help="Data root (default: data)")
    parser.add_argument("--set-type", default="all", help="Cohort folder tier for jax/split paths (default: all)")
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        metavar="N",
        help="Re-run only trial N (0-based). Merges result into existing CV summary if present.",
    )
    parser.add_argument(
        "--run-config",
        default=DEFAULT_CLI_RUN_CONFIG,
        metavar="SUFFIX",
        help="Run config suffix for paths and behavior (default: %(default)s). E.g. base, symmetric, symmetric_gen_loss, penalty_off, penalty_off_gen_loss, symmetric_penalty_off, symmetric_penalty_off_gen_loss, stenosis_off, stenosis_off_symmetric. zeroD/ml_inputs/jax/results use .../set_name/SUFFIX/....",
    )
    parser.add_argument(
        "--NN-vessel",
        action="store_true",
        dest="nn_vessel",
        default=True,
        help="Train vessel NN per trial and run vessel NN inference (junction+vessel and vessel-only modalities in MSE). Default: True.",
    )
    parser.add_argument(
        "--no-NN-vessel",
        action="store_false",
        dest="nn_vessel",
        help="Disable vessel NN training and inference (junction NN only).",
    )
    parser.add_argument(
        "--skip-training-if-exists",
        action="store_true",
        help="Skip junction and/or vessel training for a trial if the corresponding model files already exist.",
    )
    parser.add_argument(
        "--no-redo",
        action="store_true",
        dest="no_redo",
        help="Pass --no-redo to generate_zerod_inputs (skip recreating zeroD files that already exist).",
    )
    parser.add_argument(
        "--percent-train",
        type=float,
        default=0.9,
        metavar="P",
        help="Fraction of geometries for training (0–1); remainder used for validation (default: 0.9).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Regenerate CV summary CSVs (overall MSE + pressure/flow MSE + max error) from existing cv_summary.csv and per-geometry mse_comparison.csv files. No training or deploy.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate location comparison plots from existing zeroD data. No training or deploy.",
    )
    args = parser.parse_args()

    run_config_suffix = (args.run_config or DEFAULT_CLI_RUN_CONFIG).strip()
    if run_config_suffix not in ALLOWED_RUN_CONFIGS:
        parser.error(
            f"Unrecognized --run-config: {run_config_suffix!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_RUN_CONFIGS))}."
        )
    flags = run_config_suffix_to_flags(run_config_suffix)
    if flags["stenosis_off"] and flags["penalty_off"]:
        parser.error("Cannot use both stenosis_off and penalty_off in --run-config.")
    if args.metrics_only and args.plots_only:
        parser.error("Cannot use both --metrics-only and --plots-only.")
    if args.plots_only:
        regenerate_location_plots(
            set_name=args.set_name,
            geometry_variant=args.geometry_variant,
            data_root=args.data_root,
            run_config_suffix=run_config_suffix,
            stenosis_off=flags["stenosis_off"],
            normalize=flags["normalize"],
            penalty_off=flags["penalty_off"],
            symmetric_loss=flags["symmetric_loss"],
            clip_predictions=flags["clip_predictions"],
        )
        return
    if args.metrics_only:
        regenerate_cv_metrics_from_existing(
            set_name=args.set_name,
            geometry_variant=args.geometry_variant,
            data_root=args.data_root,
            normalize=flags["normalize"],
            run_config_suffix=run_config_suffix,
        )
        return

    run_cross_validation(
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        num_trials=args.num_trials,
        data_root=args.data_root,
        set_type=args.set_type,
        trial_index=args.trial,
        normalize=flags["normalize"],
        nn_vessel=args.nn_vessel,
        skip_training_if_exists=args.skip_training_if_exists,
        clip_predictions=flags["clip_predictions"],
        stenosis_off=flags["stenosis_off"],
        penalty_off=flags["penalty_off"],
        symmetric_loss=flags["symmetric_loss"],
        percent_train=args.percent_train,
        no_redo=getattr(args, "no_redo", False),
        run_config_cli=run_config_suffix,
    )


if __name__ == "__main__":
    main()
