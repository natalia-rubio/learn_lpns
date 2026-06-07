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
    list_ml_input_geometries,
    load_split_for_training,
    resolve_flat_indices,
    resolve_geometry_row_ranges_from_jax_dict,
)
from util.data_processing.data_dict_from_csvs import get_default_include_features
from util.tools.basic import load_dict, save_dict
from util.zerod_calibration.forward_mse import calculate_mse_between_3d_and_0d, parse_mse_comparison_csv
from util.zerod_calibration.cv_metrics import (
    accumulate_trial_metrics,
    read_cv_summary_rows,
    trial_metrics_to_row,
    write_all_cv_summary_csvs,
)
from util.zerod_calibration.generate_zerod_inputs_cli import (
    DEFAULT_JUNCTION_TYPES,
    namespace_to_generate_zerod_argv,
    prepare_generate_zerod_namespace,
)
from util.zerod_calibration.modality_paths import modality_csv_paths
from util.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    resolve_run_config_suffix,
    run_config_suffix_to_flags,
)
from util.zerod_calibration.batch_generate_zerod_inputs_vmr import check_geometry_complete
from util.zerod_calibration.tools.file_io import get_paths, get_vmr_geometries

JUNCTION_TYPE = DEFAULT_JUNCTION_TYPES[0]


def _ensure_ml_inputs_and_jax_for_config(
    set_name,
    geometry_variant,
    run_config_suffix,
    data_root,
    set_type,
    batch_geometries,
    all_geometries,
    reasons,
):
    """Run batch_generate_zerod_inputs_vmr and/or run_data_processing for missing prerequisites."""
    script_dir = os.path.dirname(__file__)
    batch_script = os.path.join(script_dir, "batch_generate_zerod_inputs_vmr.py")
    data_processing_script = os.path.join(
        os.path.dirname(script_dir), "data_processing", "run_data_processing.py"
    )
    reason_text = "; ".join(reasons)
    print(f"Run config {run_config_suffix}: {reason_text}")

    if batch_geometries:
        print(
            f"  Running batch generate for {len(batch_geometries)} geometry/ies: "
            f"{batch_geometries[:5]}{'...' if len(batch_geometries) > 5 else ''}"
        )
        cmd_batch = [
            sys.executable,
            batch_script,
            "--set-name",
            set_name,
            "--geometries",
            *batch_geometries,
            "--run-config",
            run_config_suffix,
        ]
        result = subprocess.run(cmd_batch, cwd=REPO_ROOT, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"batch_generate_zerod_inputs_vmr failed (return code {result.returncode}). "
                "Fix the error above and re-run."
            )

    if all_geometries:
        print(
            f"  Running data processing for all {len(all_geometries)} geometries "
            f"(full jax pickle rebuild): "
            f"{all_geometries[:5]}{'...' if len(all_geometries) > 5 else ''}"
        )
        cmd_dp = [
            sys.executable, data_processing_script,
            "--set-name", set_name,
            "--geometry-variant", geometry_variant,
            "--run-config", run_config_suffix,
            "--geometries", *all_geometries,
        ]
        result_dp = subprocess.run(cmd_dp, cwd=REPO_ROOT, text=True)
        if result_dp.returncode != 0:
            raise RuntimeError(
                f"run_data_processing failed with --run-config {run_config_suffix} "
                f"(return code {result_dp.returncode}). Fix the error above and re-run."
            )

    print(f"  Done: prerequisites ready for config {run_config_suffix}")


def _data_path(data_root, category, set_name, run_config_suffix, *parts):
    """Build data/{category}/{set_name}/[{run_config}/]{parts...}."""
    segments = [data_root, category, set_name]
    if run_config_suffix:
        segments.append(run_config_suffix)
    segments.extend(parts)
    return os.path.join(*segments)


def _results_path(category, set_name, run_config_suffix, *parts):
    """Build results/{category}/{set_name}/[{run_config}/]{parts...}."""
    segments = ["results", category, set_name]
    if run_config_suffix:
        segments.append(run_config_suffix)
    segments.extend(parts)
    return os.path.join(*segments)


def _jax_pickle_path(
    data_root,
    set_name,
    run_config_suffix,
    geometry_variant,
    set_type,
    num_geos,
    *,
    vessel=False,
):
    prefix = "jax_arrays_vessel" if vessel else "jax_arrays"
    return _data_path(
        data_root,
        "jax_arrays",
        set_name,
        run_config_suffix,
        geometry_variant,
        set_type,
        f"{prefix}_num_geos_{num_geos}.pkl",
    )


def _resolve_cv_run_config(run_config_suffix):
    """Parse run config suffix into flags used by the CV trial loop."""
    if run_config_suffix is None:
        run_config_suffix = DEFAULT_CLI_RUN_CONFIG
    flags = run_config_suffix_to_flags(run_config_suffix)
    if flags["stenosis_off"] and flags["penalty_off"]:
        raise ValueError("Cannot use both stenosis_off and penalty_off in run config.")
    return {
        "run_config_suffix": run_config_suffix,
        "data_paths_suffix": run_config_suffix,
        "stenosis_off": flags["stenosis_off"],
        "symmetric_loss": flags["symmetric_loss"],
        "penalty_off": flags["penalty_off"],
    }


def _ensure_cv_prerequisites(
    set_name,
    geometry_variant,
    data_root,
    set_type,
    run_config_suffix,
):
    """Run batch + data processing when ml_inputs, jax pickle, or calibrated zeroD are missing.

    Batch runs only on incomplete geometries; data processing always rebuilds the full stacked jax pickle.
    """
    if not run_config_suffix:
        return

    richter_dir = os.path.join(data_root, "zeroD", set_name, "richter-0d")
    try:
        geometries = get_vmr_geometries(richter_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot get geometry list: {e} "
            "Ensure is only supported for sets with data/zeroD/<set_name>/richter-0d/."
        ) from e

    ml_inputs_dir = _data_path(data_root, "ml_inputs", set_name, run_config_suffix, geometry_variant)
    missing_ml = []
    missing_calib = []
    for geo in geometries:
        geo_dir = os.path.join(ml_inputs_dir, geo)
        if not os.path.exists(os.path.join(geo_dir, "geometric_features.csv")) or not os.path.exists(
            os.path.join(geo_dir, "junction_lumped_parameters.csv")
        ):
            missing_ml.append(geo)
        # CV needs calibrated zeroD outputs for labels; NN forward files are created per trial at deploy.
        elif not check_geometry_complete(
            set_name,
            geo,
            ["BloodVesselJunction"],
            skip_forward=False,
            run_config_suffix=run_config_suffix,
            require_nn_outputs=False,
        ):
            missing_calib.append(geo)

    num_geos = len(geometries)
    jax_path = _jax_pickle_path(
        data_root, set_name, run_config_suffix, geometry_variant, set_type, num_geos
    )
    jax_missing = not os.path.exists(jax_path)

    if not (missing_ml or jax_missing or missing_calib):
        return

    reasons = []
    if missing_ml:
        reasons.append(
            f"missing ml_inputs for {len(missing_ml)} geometries "
            f"(e.g. {missing_ml[:3]}{'...' if len(missing_ml) > 3 else ''})"
        )
    if jax_missing:
        reasons.append("jax_arrays pkl not found")
    if missing_calib:
        reasons.append(
            f"incomplete calibrated zeroD for {len(missing_calib)} geometries "
            f"(e.g. {missing_calib[:3]}{'...' if len(missing_calib) > 3 else ''})"
        )
    batch_geometries = sorted(set(missing_ml + missing_calib))
    _ensure_ml_inputs_and_jax_for_config(
        set_name=set_name,
        geometry_variant=geometry_variant,
        run_config_suffix=run_config_suffix,
        data_root=data_root,
        set_type=set_type,
        batch_geometries=batch_geometries,
        all_geometries=geometries,
        reasons=reasons,
    )


def _load_cv_jax_cohort(
    ml_inputs_root,
    data_root,
    set_name,
    geometry_variant,
    run_config_suffix,
    set_type,
):
    """Locate jax pickles and load cohort metadata from the pickle (no CSV fallbacks)."""
    geometries_for_path = list_ml_input_geometries(
        ml_inputs_root, set_name, geometry_variant, run_config_suffix=run_config_suffix
    )
    num_geos = len(geometries_for_path)
    if num_geos == 0:
        raise ValueError(
            f"No geometries found under {ml_inputs_root}/{set_name}"
            + (f"/{run_config_suffix}" if run_config_suffix else "") + f"/{geometry_variant}"
        )

    jax_path = _jax_pickle_path(
        data_root, set_name, run_config_suffix, geometry_variant, set_type, num_geos
    )
    if not os.path.exists(jax_path):
        raise FileNotFoundError(
            f"Jax arrays not found: {jax_path} (expected {num_geos} geometries). "
            f"Re-run data processing for set={set_name!r}, variant={geometry_variant!r}"
            + (f", run-config={run_config_suffix!r}" if run_config_suffix else "")
            + "."
        )

    vessel_jax_path = _jax_pickle_path(
        data_root,
        set_name,
        run_config_suffix,
        geometry_variant,
        set_type,
        num_geos,
        vessel=True,
    )
    data_dict = load_dict(jax_path)
    row_ranges, total_rows, geometries = resolve_geometry_row_ranges_from_jax_dict(data_dict)
    if len(geometries) != num_geos:
        raise ValueError(
            f"Jax pickle lists {len(geometries)} geometries but ml_inputs has {num_geos} "
            f"geometry folders under {geometry_variant!r}. "
            f"Delete {jax_path} and re-run data processing."
        )
    num_pts = total_rows
    return {
        "data_dict": data_dict,
        "row_ranges": row_ranges,
        "geometries": geometries,
        "num_geos": len(geometries),
        "num_pts": num_pts,
        "vessel_jax_path": vessel_jax_path,
    }


def _cv_run_directory_paths(data_root, set_name, geometry_variant, run_config_suffix, set_type):
    """Paths and filenames used by the CV trial loop."""
    script_dir = os.path.dirname(__file__)
    prefix = "" if geometry_variant == "original" else f"{geometry_variant}_"
    return {
        "split_indices_dir": _data_path(
            data_root, "split_indices", set_name, run_config_suffix, geometry_variant, set_type
        ),
        "model_dir_base": _results_path("models", set_name, run_config_suffix),
        "zero_d_base": _data_path(data_root, "zeroD", set_name, run_config_suffix),
        "launch_training_script": os.path.join(
            os.path.dirname(script_dir), "neural_network", "launch_training.py"
        ),
        "mse_csv_name": f"{prefix}mse_comparison.csv" if prefix else "mse_comparison.csv",
    }


def _generate_cv_barcharts(
    set_name,
    geometry_variant,
    run_config_suffix,
    *,
    results_root="results",
):
    """Generate pressure error barcharts from the CV summary CSVs in results/cross_validation."""
    print(f"\nGenerating CV pressure error barcharts (run-config={run_config_suffix})...")
    cmd = [
        sys.executable,
        "-m",
        "util.visualizations.cv_pressure_max_pct_error_barchart",
        set_name,
        geometry_variant,
        "--run-config",
        run_config_suffix,
        "--data-root",
        results_root,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        print(
            f"Warning: cv_pressure_max_pct_error_barchart failed (exit {result.returncode}). "
            "CV summary CSVs were written successfully."
        )
        return False
    return True


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


def _cv_results_paths(set_name, geometry_variant, run_config_suffix):
    out_dir = _results_path("cross_validation", set_name, run_config_suffix)
    summary_path = os.path.join(out_dir, f"{geometry_variant}_cv_summary.csv")
    return out_dir, summary_path


def _run_zerod_inputs_for_cv(
    set_name,
    geo_name,
    *,
    run_config_suffix,
    geometry_variant,
    trial_id=None,
    model_dir=None,
    nn_vessel=False,
    plots_only=False,
    verbose=False,
):
    """Run generate_zerod_inputs for CV deploy (NN-only) or plot refresh (--plots-only)."""
    ns = argparse.Namespace(
        set_name=set_name,
        geo_name=geo_name,
        run_config=run_config_suffix,
        geometry_variant=geometry_variant,
        trial_id=trial_id,
        model_dir=model_dir,
        NN_only=bool(model_dir) and not plots_only,
        plots_only=plots_only,
        NN_vessel=nn_vessel,
        verbose=verbose,
        skip_steps="",
        no_redo=False,
    )
    prepare_generate_zerod_namespace(ns)
    cmd = namespace_to_generate_zerod_argv(ns, set_name=set_name, geo_name=geo_name)
    label = "Plots" if plots_only else "Deploy"
    print(f"  {label} on {geo_name}: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True)


def run_cv_plots_only(
    set_name,
    geometry_variant,
    data_root="data",
    run_config_suffix="",
    nn_vessel=True,
):
    """Re-run Step 6 plots for each validation geometry listed in the existing CV summary."""
    out_dir, summary_path = _cv_results_paths(set_name, geometry_variant, run_config_suffix)
    summary_rows = read_cv_summary_rows(summary_path)
    if not summary_rows:
        print(f"Existing CV summary not found or empty: {summary_path}")
        print("Run cross-validation once, then use --plots-only to regenerate plots.")
        return None

    print(f"Regenerating plots for {len(summary_rows)} trial(s)...")
    success_count = 0
    total = 0
    for trial_id, _, val_geometries in summary_rows:
        for val_geo in val_geometries:
            total += 1
            result = _run_zerod_inputs_for_cv(
                set_name,
                val_geo,
                run_config_suffix=run_config_suffix,
                geometry_variant=geometry_variant,
                trial_id=trial_id,
                nn_vessel=nn_vessel,
                plots_only=True,
            )
            if result.returncode == 0:
                print(f"  ✓ trial {trial_id} / {val_geo}")
                success_count += 1
            else:
                print(f"  ✗ trial {trial_id} / {val_geo} (exit code {result.returncode})")
    print(f"Done: {success_count}/{total} geometry plot runs.")
    return success_count


def run_cross_validation(
    set_name,
    geometry_variant,
    num_trials,
    data_root="data",
    set_type="all",
    ml_inputs_root=None,
    trial_index=None,
    nn_vessel=True,
    skip_training_if_exists=False,
    percent_train=0.9,
    run_config_suffix=None,
    skip_barchart=False,
):
    ml_inputs_root = ml_inputs_root or os.path.join(data_root, "ml_inputs")
    config = _resolve_cv_run_config(run_config_suffix)
    run_config_suffix = config["run_config_suffix"]
    data_paths_suffix = config["data_paths_suffix"]
    stenosis_off = config["stenosis_off"]
    symmetric_loss = config["symmetric_loss"]
    penalty_off = config["penalty_off"]
    print(f"Run config: {data_paths_suffix!r}")

    _ensure_cv_prerequisites(
        set_name, geometry_variant, data_root, set_type, data_paths_suffix
    )
    cohort = _load_cv_jax_cohort(
        ml_inputs_root,
        data_root,
        set_name,
        geometry_variant,
        data_paths_suffix,
        set_type,
    )
    data_dict = cohort["data_dict"]
    row_ranges = cohort["row_ranges"]
    geometries = cohort["geometries"]
    num_geos = cohort["num_geos"]
    num_pts = cohort["num_pts"]
    vessel_jax_path = cohort["vessel_jax_path"]
    paths = _cv_run_directory_paths(
        data_root, set_name, geometry_variant, data_paths_suffix, set_type
    )
    split_indices_dir = paths["split_indices_dir"]
    model_dir_base = paths["model_dir_base"]
    zero_d_base = paths["zero_d_base"]
    launch_training_script = paths["launch_training_script"]
    mse_csv_name = paths["mse_csv_name"]

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
    if nn_vessel:
        print("NN-vessel: will train vessel NN per trial and include vessel-predicted modality in MSE")
    if symmetric_loss:
        print("Symmetric loss: overestimate weight = 1.0 for all models")
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
        out_dir_cv = _results_path("cross_validation", set_name, data_paths_suffix)
        os.makedirs(out_dir_cv, exist_ok=True)
        oor_csv = os.path.join(
            out_dir_cv,
            f"out_of_range_{geometry_variant}_trial_{trial}.csv",
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
            model_dir_base, f"{geometry_variant}_trial_{trial}"
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
            if symmetric_loss:
                cmd_train.append("--symmetric-loss")
            if run_config_suffix:
                cmd_train.extend(["--run-config", run_config_suffix])
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
                model_dir_base, f"{geometry_variant}_vessel_trial_{trial}"
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
                if symmetric_loss:
                    cmd_vessel.append("--symmetric-loss")
                if run_config_suffix:
                    cmd_vessel.extend(["--run-config", run_config_suffix])
                print(f"  Running vessel training: {' '.join(cmd_vessel)}")
                result_vessel = subprocess.run(cmd_vessel, cwd=REPO_ROOT, text=True)
                if result_vessel.returncode != 0:
                    print(f"  Vessel training failed with return code {result_vessel.returncode}")
                    all_trial_results.append(
                        {"trial_id": trial, "val_geometries": ",".join(val_geometries), "error": "vessel_training_failed"}
                    )
                    continue

        # Deploy on each validation geometry (NN-only; Step 6 plots run at end of pipeline)
        trial_metrics = {}
        for val_geo in val_geometries:
            result_deploy = _run_zerod_inputs_for_cv(
                set_name,
                val_geo,
                run_config_suffix=run_config_suffix,
                geometry_variant=geometry_variant,
                trial_id=trial,
                model_dir=model_dir,
                nn_vessel=nn_vessel,
            )
            if result_deploy.returncode != 0:
                print(f"  Deploy failed for {val_geo} with return code {result_deploy.returncode}")
                continue
            mse_csv_path = os.path.join(zero_d_base, val_geo, mse_csv_name)
            accumulate_trial_metrics(trial_metrics, parse_mse_comparison_csv(mse_csv_path))

        all_trial_results.append(
            trial_metrics_to_row(trial, ",".join(val_geometries), trial_metrics)
        )

    if not all_trial_results:
        print("No trial results to summarize.")
        return

    out_dir, summary_path = _cv_results_paths(set_name, geometry_variant, data_paths_suffix)

    # If we re-ran a single trial and summary already exists, merge this result into it
    if trial_index is not None and os.path.exists(summary_path):
        existing_by_trial = {}
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
        for r in all_trial_results:
            existing_by_trial[r["trial_id"]] = r
        all_trial_results = [existing_by_trial[tid] for tid in sorted(existing_by_trial)]

    write_all_cv_summary_csvs(summary_path, out_dir, geometry_variant, all_trial_results)

    print(f"\nWrote CV summary to {summary_path}")
    print(f"  Also wrote: pressure_mse, flow_mse, max_error, pressure_max_error, flow_max_error, max_rel_error, pressure_max_rel_error, flow_max_rel_error")
    if not skip_barchart and data_paths_suffix:
        _generate_cv_barcharts(
            set_name,
            geometry_variant,
            data_paths_suffix,
        )
    return summary_path


def regenerate_cv_metrics_from_existing(
    set_name,
    geometry_variant,
    data_root="data",
    run_config_suffix="",
):
    """
    Regenerate CV summary CSVs from existing cv_summary.csv and per-geometry mse_comparison.csv files.
    Re-runs MSE calculation per validation geometry, then re-aggregates trial metrics.
    """
    out_dir, summary_path = _cv_results_paths(set_name, geometry_variant, run_config_suffix)
    if run_config_suffix:
        zero_d_base = os.path.join(data_root, "zeroD", set_name, run_config_suffix)
    else:
        zero_d_base = os.path.join(data_root, "zeroD", set_name)

    summary_rows = read_cv_summary_rows(summary_path)
    if not summary_rows:
        print(f"Existing CV summary not found or empty: {summary_path}")
        print("Run cross-validation once to create it, then use --metrics-only to refresh metric CSVs.")
        return None

    prefix = "" if geometry_variant == "original" else f"{geometry_variant}_"
    mse_csv_name = f"{prefix}mse_comparison.csv" if prefix else "mse_comparison.csv"

    all_val_geos = sorted({g for _, _, vg in summary_rows for g in vg})
    print(f"Re-running MSE calculation for {len(all_val_geos)} validation geometries...")
    for val_geo in all_val_geos:
        base_dir = os.path.join(zero_d_base, val_geo)
        geo_args = argparse.Namespace(set_name=set_name, geo_name=val_geo)
        geometry_variants, _, _ = get_paths(base_dir, geo_args)
        if geometry_variant not in geometry_variants:
            print(f"  Skip {val_geo}: unknown geometry variant {geometry_variant}")
            continue
        geo_variant_paths = geometry_variants[geometry_variant]
        variant_calibration_input = geo_variant_paths["calibration_input"]
        if not os.path.exists(variant_calibration_input):
            print(f"  Skip {val_geo}: calibration input not found")
            continue
        csv_results_dict = modality_csv_paths(
            geo_variant_paths,
            base_dir,
            geometry_variant,
            JUNCTION_TYPE,
            nn_vessel=True,
            extra_junction_types=("NORMAL_JUNCTION",),
        )
        if not csv_results_dict:
            print(f"  Skip {val_geo}: no result CSVs found")
            continue
        mse_csv_path = os.path.join(base_dir, mse_csv_name)
        try:
            calculate_mse_between_3d_and_0d(
                variant_calibration_input,
                csv_results_dict,
                output_csv_path=mse_csv_path,
                verbose=False,
            )
            print(f"  ✓ {val_geo}")
        except Exception as e:
            print(f"  ✗ {val_geo}: {e}")

    all_trial_results = []
    for trial_id, val_geometries_str, val_geometries in summary_rows:
        trial_metrics = {}
        for val_geo in val_geometries:
            mse_csv_path = os.path.join(zero_d_base, val_geo, mse_csv_name)
            accumulate_trial_metrics(trial_metrics, parse_mse_comparison_csv(mse_csv_path))
        all_trial_results.append(
            trial_metrics_to_row(trial_id, val_geometries_str, trial_metrics)
        )

    if not all_trial_results:
        print("No trial rows found in existing summary.")
        return None

    _, modalities = write_all_cv_summary_csvs(
        summary_path, out_dir, geometry_variant, all_trial_results
    )

    print("Regenerated CV metrics from existing per-geometry MSE CSVs.")
    print(f"  Updated: {summary_path}")
    base = os.path.join(out_dir, f"{geometry_variant}_cv_summary")
    print(
        f"  Wrote:   {base}_pressure_mse.csv, _flow_mse.csv, _max_error.csv, "
        "_pressure_max_error.csv, _flow_max_error.csv, _max_rel_error.csv, "
        "_pressure_max_rel_error.csv, _flow_max_rel_error.csv"
    )
    has_max = any(
        r.get(f"MaxError_{m}") is not None
        and r.get(f"MaxError_{m}") != ""
        and not (isinstance(r.get(f"MaxError_{m}"), float) and np.isnan(r.get(f"MaxError_{m}")))
        for r in all_trial_results
        for m in modalities
        if f"MaxError_{m}" in r
    )
    if not has_max:
        print(
            "  Note: Max-error columns are empty; per-geometry mse_comparison.csv files may "
            "lack the new rows. Re-run MSE calculation for each geometry to populate them."
        )
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
        metavar="TOKENS",
        help=(
            "Run-config tokens in any order, underscore-separated "
            f"(default: {DEFAULT_CLI_RUN_CONFIG}). "
            "Full canonical suffixes are also accepted."
        ),
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
        help="Regenerate comparison plots for CV validation geometries via generate_zerod_inputs --plots-only. No training or deploy.",
    )
    parser.add_argument(
        "--skip-barchart",
        action="store_true",
        help="Do not run cv_pressure_max_pct_error_barchart after writing CV summary CSVs.",
    )
    args = parser.parse_args()

    try:
        run_config_suffix = resolve_run_config_suffix(args.run_config)
    except ValueError as exc:
        parser.error(str(exc))
    flags = run_config_suffix_to_flags(run_config_suffix)
    if flags["stenosis_off"] and flags["penalty_off"]:
        parser.error("Cannot use both stenosis_off and penalty_off in --run-config.")
    if args.metrics_only and args.plots_only:
        parser.error("Cannot use both --metrics-only and --plots-only.")
    if args.plots_only:
        run_cv_plots_only(
            set_name=args.set_name,
            geometry_variant=args.geometry_variant,
            data_root=args.data_root,
            run_config_suffix=run_config_suffix,
            nn_vessel=args.nn_vessel,
        )
        return
    if args.metrics_only:
        regenerate_cv_metrics_from_existing(
            set_name=args.set_name,
            geometry_variant=args.geometry_variant,
            data_root=args.data_root,
            run_config_suffix=run_config_suffix,
        )
        if not args.skip_barchart:
            _generate_cv_barcharts(
                args.set_name,
                args.geometry_variant,
                run_config_suffix,
            )
        return

    run_cross_validation(
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        num_trials=args.num_trials,
        data_root=args.data_root,
        set_type=args.set_type,
        trial_index=args.trial,
        nn_vessel=args.nn_vessel,
        skip_training_if_exists=args.skip_training_if_exists,
        percent_train=args.percent_train,
        run_config_suffix=run_config_suffix,
        skip_barchart=args.skip_barchart,
    )


if __name__ == "__main__":
    main()
