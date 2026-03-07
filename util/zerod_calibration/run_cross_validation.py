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
    generate_split_indices,
    get_geometry_row_ranges,
)
from util.data_processing.data_dict_from_csvs import get_default_include_features
from util.tools.basic import load_dict, save_dict
from util.zerod_calibration.post_processing import calculate_mse_between_3d_and_0d


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


def run_cross_validation(
    set_name,
    geometry_variant,
    num_trials,
    data_root="data",
    set_type="test",
    ml_inputs_root=None,
    trial_index=None,
    normalize=False,
    nn_vessel=True,
    skip_training_if_exists=False,
    asymmetric_loss=False,
    overestimate_weight=2.0,
    clip_predictions=False,
    stenosis_off=False,
    percent_train=0.9,
):
    if ml_inputs_root is None:
        ml_inputs_root = os.path.join(data_root, "ml_inputs")

    # Discover geometries and row ranges (same order as jax array)
    row_ranges, total_rows, geometries = get_geometry_row_ranges(
        ml_inputs_root, set_name, geometry_variant
    )
    num_geos = len(geometries)
    if num_geos == 0:
        raise ValueError(
            f"No geometries found under {ml_inputs_root}/{set_name}/{geometry_variant}"
        )

    norm_suffix = "_normalized" if normalize else ""
    jax_path = os.path.join(
        data_root,
        "jax_arrays",
        set_name,
        geometry_variant,
        set_type,
        f"jax_arrays_num_geos_{num_geos}{norm_suffix}.pkl",
    )
    if not os.path.exists(jax_path):
        hint = ""
        if normalize:
            hint = (
                f" Generate it by running data processing with --normalize, e.g.: "
                f"python util/data_processing/run_data_processing.py --set-name {set_name} --geometry-variant {geometry_variant} --normalize"
            )
        raise FileNotFoundError(
            f"Jax arrays not found: {jax_path} (expected {num_geos} geometries"
            + (" with normalization" if normalize else "") + ")." + hint
        )

    vessel_jax_path = os.path.join(
        data_root,
        "jax_arrays",
        set_name,
        geometry_variant,
        set_type,
        f"jax_arrays_vessel_num_geos_{num_geos}{norm_suffix}.pkl",
    ) if normalize else None

    data_dict = load_dict(jax_path)
    num_pts = int(np.asarray(data_dict["input"]).shape[0])
    if total_rows != num_pts:
        raise ValueError(
            f"Row count mismatch: row_ranges sum={total_rows} vs jax num_pts={num_pts}"
        )

    split_indices_dir = os.path.join(
        data_root, "split_indices", set_name, geometry_variant, set_type
    )
    model_dir_base = os.path.join("results", "models", set_name)
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
    if asymmetric_loss:
        print(f"Asymmetric loss: overestimate weight = {overestimate_weight}")
    if clip_predictions:
        print("Clip predictions: R/S/L will be clipped to training set min/max during deploy")
    if stenosis_off:
        print("Stenosis-off: calibrate_stenosis_coefficient=False, all stenosis set to 0, NN will not predict stenosis")

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
            split_dict = load_dict(split_path)
            train_ind = np.asarray(split_dict["train_ind"]).ravel()
            val_ind = np.asarray(split_dict["val_ind"]).ravel()
            train_geometries = split_dict.get("train_geometries")
            val_geometries = split_dict.get("val_geometries")
            if not train_geometries or not val_geometries:
                train_geo_idx = np.unique([i for i, (s, e) in enumerate(row_ranges) for r in train_ind if s <= r < e])
                val_geo_idx = np.unique([i for i, (s, e) in enumerate(row_ranges) for r in val_ind if s <= r < e])
                train_geometries = [geometries[i] for i in train_geo_idx]
                val_geometries = [geometries[i] for i in val_geo_idx]
            print(f"  Split (reused from {os.path.basename(split_path)}): {len(train_geometries)} train, {len(val_geometries)} val -> {val_geometries}")
        else:
            # Ensure this trial's validation set is different from all previous trials
            max_attempts = 200
            for attempt in range(max_attempts):
                seed = trial * 1000 + attempt
                train_ind, val_ind, train_geo_idx, val_geo_idx = generate_split_indices(
                    num_pts=num_pts,
                    percent_train=percent_train,
                    seed=seed,
                    geometry_row_ranges=row_ranges,
                )
                train_geometries = [geometries[i] for i in train_geo_idx]
                val_geometries = [geometries[i] for i in val_geo_idx]
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

            split_dict = {
                "train_ind": np.asarray(train_ind, dtype=int),
                "val_ind": np.asarray(val_ind, dtype=int),
                "num_offsets": 1,
                "percent_train": percent_train,
                "seed": int(seed),
                "num_pts": num_pts,
                "split_by_geometry": True,
                "train_geometries": train_geometries,
                "val_geometries": val_geometries,
            }
            save_dict(split_dict, split_path)
            print(f"  Split: {len(train_geometries)} train, {len(val_geometries)} val -> {val_geometries}")

        # Check for validation features outside training set range; write CSV per split
        feature_names = get_default_include_features()
        X_input = np.asarray(data_dict["input"])
        out_of_range = _check_val_out_of_train_range(
            X_input, train_ind, val_ind, row_ranges, geometries, feature_names
        )
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
            if asymmetric_loss:
                cmd_train.append("--asymmetric-loss")
                cmd_train.append("--overestimate-weight")
                cmd_train.append(str(overestimate_weight))
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
                if asymmetric_loss:
                    cmd_vessel.append("--asymmetric-loss")
                    cmd_vessel.append("--overestimate-weight")
                    cmd_vessel.append(str(overestimate_weight))
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
):
    """
    Regenerate CV summary CSVs (overall MSE + pressure/flow MSE + max error variants)
    by reading the existing cv_summary.csv and per-geometry mse_comparison.csv files.
    Does not re-run training or deploy. Use this to add the new metric CSVs after
    a previous CV run, or to refresh metrics if per-geometry MSE CSVs were updated
    (e.g. after re-running MSE calculation with max-error support).

    Requires:
      - results/cross_validation/{set_name}/{geometry_variant}_cv_summary.csv
      - For each (trial, val_geo) in that file: data/zeroD/{set_name}/{val_geo}/{geometry_variant}_mse_comparison.csv
    """
    norm_suffix = "_normalized" if normalize else ""
    out_dir = os.path.join("results", "cross_validation", set_name)
    summary_path = os.path.join(
        out_dir, f"{geometry_variant}{norm_suffix}_cv_summary.csv"
    )
    if not os.path.exists(summary_path):
        print(f"Existing CV summary not found: {summary_path}")
        print("Run cross-validation once to create it, then use --metrics-only to add new metric CSVs.")
        return None

    zero_d_base = os.path.join(data_root, "zeroD", set_name)
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
    parser.add_argument("--set-type", default="test", help="Set type for paths (default: test)")
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        metavar="N",
        help="Re-run only trial N (0-based). Merges result into existing CV summary if present.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use z-normalized jax arrays for training and normalization/unnormalization at NN inference.",
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
        "--asymmetric-loss",
        action="store_true",
        help="Use asymmetric loss in NN training: overestimates count twice as much as underestimates.",
    )
    parser.add_argument(
        "--overestimate-weight",
        type=float,
        default=2.0,
        help="Weight for overestimation errors when --asymmetric-loss (default: 2.0).",
    )
    parser.add_argument(
        "--clip-predictions",
        action="store_true",
        dest="clip_predictions",
        help="Clip NN predictions (R, S, L) to training set min/max during deploy.",
    )
    parser.add_argument(
        "--stenosis-off",
        action="store_true",
        dest="stenosis_off",
        help="Turn off stenosis: calibrate_stenosis_coefficient=False, set all stenosis to 0, do not use NN to predict stenosis.",
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
    args = parser.parse_args()

    if args.metrics_only:
        regenerate_cv_metrics_from_existing(
            set_name=args.set_name,
            geometry_variant=args.geometry_variant,
            data_root=args.data_root,
            normalize=args.normalize,
        )
        return

    run_cross_validation(
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        num_trials=args.num_trials,
        data_root=args.data_root,
        set_type=args.set_type,
        trial_index=args.trial,
        normalize=args.normalize,
        nn_vessel=args.nn_vessel,
        skip_training_if_exists=args.skip_training_if_exists,
        asymmetric_loss=args.asymmetric_loss,
        overestimate_weight=args.overestimate_weight,
        clip_predictions=args.clip_predictions,
        stenosis_off=args.stenosis_off,
        percent_train=args.percent_train,
    )


if __name__ == "__main__":
    main()
