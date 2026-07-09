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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from learn_lpns.config import get_pipeline_config
from learn_lpns.data_processing.data_dict_from_csvs import get_default_include_features
from learn_lpns.data_processing.generate_split_indices import (
    build_geometry_index_map,
    build_split_dict,
    generate_geometry_split,
    list_ml_input_geometries,
    load_split_for_training,
    resolve_flat_indices,
    resolve_geometry_row_ranges_from_jax_dict,
)
from learn_lpns.tools.basic import load_dict, save_dict
from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.batch_generate_zerod_inputs_vmr import check_geometry_complete
from learn_lpns.zerod_calibration.cv_metrics import (
    accumulate_trial_metrics,
    read_cv_summary_rows,
    trial_metrics_to_row,
    write_all_cv_summary_csvs,
)
from learn_lpns.zerod_calibration.forward_mse import (
    calculate_mse_between_3d_and_0d,
    parse_mse_comparison_csv,
)
from learn_lpns.zerod_calibration.generate_zerod_inputs_cli import (
    DEFAULT_JUNCTION_TYPES,
    namespace_to_generate_zerod_argv,
    prepare_generate_zerod_namespace,
)
from learn_lpns.zerod_calibration.modality_paths import modality_csv_paths
from learn_lpns.zerod_calibration.nn_inference import rri_models_complete
from learn_lpns.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    resolve_run_config_suffix,
    run_config_suffix_to_flags,
)
from learn_lpns.zerod_calibration.tools.file_io import (
    STANDARD_0D_SUBDIR,
    get_paths,
    get_vmr_geometries,
    standard_0d_dir,
)

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
    *,
    no_redo=False,
):
    """Run batch_generate_zerod_inputs_vmr and/or run_data_processing for missing prerequisites."""
    script_dir = os.path.dirname(__file__)
    batch_script = os.path.join(script_dir, "batch_generate_zerod_inputs_vmr.py")
    data_processing_script = os.path.join(os.path.dirname(script_dir), "data_processing", "run_data_processing.py")
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
            "--set_name",
            set_name,
            "--geometries",
            *batch_geometries,
            "--run_config",
            run_config_suffix,
            "--skip_steps",
            "nn_inference",
        ]
        if no_redo:
            cmd_batch.append("--no_redo")
        result = subprocess.run(cmd_batch, cwd=repo_root(), text=True)
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
            sys.executable,
            data_processing_script,
            "--set_name",
            set_name,
            "--geometry_variant",
            geometry_variant,
            "--run_config",
            run_config_suffix,
            "--geometries",
            *all_geometries,
        ]
        result_dp = subprocess.run(cmd_dp, cwd=repo_root(), text=True)
        if result_dp.returncode != 0:
            raise RuntimeError(
                f"run_data_processing failed with --run_config {run_config_suffix} "
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
    return {
        "run_config_suffix": run_config_suffix,
        "data_paths_suffix": run_config_suffix,
        "quadratic_resistor": flags["quadratic_resistor"],
        "asymmetric_loss": flags["asymmetric_loss"],
        "penalty_on": flags["penalty_on"],
    }


def _ensure_cv_prerequisites(
    set_name,
    geometry_variant,
    data_root,
    set_type,
    run_config_suffix,
    *,
    no_redo=False,
):
    """Run batch + data processing when ml_inputs, jax pickle, or calibrated zeroD are missing.

    Batch runs only on incomplete geometries; data processing always rebuilds the full stacked jax pickle.
    """
    if not run_config_suffix:
        return

    std_0d = standard_0d_dir(data_root, set_name)
    try:
        geometries = get_vmr_geometries(std_0d)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot get geometry list: {e} "
            f"Bootstrap ensure is only supported for sets with "
            f"data/zeroD/<set_name>/{STANDARD_0D_SUBDIR}/."
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
    jax_path = _jax_pickle_path(data_root, set_name, run_config_suffix, geometry_variant, set_type, num_geos)
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
        no_redo=no_redo,
    )


def _ml_inputs_geometries_for_cohort(
    ml_inputs_root: str,
    data_root: str,
    set_name: str,
    geometry_variant: str,
    run_config_suffix: str | None,
) -> list[str]:
    """Return ml_inputs geometry folders, restricted to standard-0d when that layout exists."""
    geometries = list_ml_input_geometries(
        ml_inputs_root, set_name, geometry_variant, run_config_suffix=run_config_suffix
    )
    std_0d = standard_0d_dir(data_root, set_name)
    if not os.path.isdir(std_0d):
        return geometries
    try:
        std_geos = set(get_vmr_geometries(std_0d))
    except FileNotFoundError:
        return geometries
    filtered = [g for g in geometries if g in std_geos]
    if not filtered:
        return geometries
    dropped = sorted(set(geometries) - set(filtered))
    if dropped:
        print(f"  Ignoring {len(dropped)} stale ml_inputs geometries not in {STANDARD_0D_SUBDIR}/: {dropped}")
    return filtered


def _load_cv_jax_cohort(
    ml_inputs_root,
    data_root,
    set_name,
    geometry_variant,
    run_config_suffix,
    set_type,
):
    """Locate jax pickles and load cohort metadata from the pickle (no CSV fallbacks)."""
    geometries_for_path = _ml_inputs_geometries_for_cohort(
        ml_inputs_root, data_root, set_name, geometry_variant, run_config_suffix
    )
    num_geos = len(geometries_for_path)
    if num_geos == 0:
        raise ValueError(
            f"No geometries found under {ml_inputs_root}/{set_name}"
            + (f"/{run_config_suffix}" if run_config_suffix else "")
            + f"/{geometry_variant}"
        )

    jax_path = _jax_pickle_path(data_root, set_name, run_config_suffix, geometry_variant, set_type, num_geos)
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
        "launch_training_script": os.path.join(os.path.dirname(script_dir), "neural_network", "launch_training.py"),
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
        "learn_lpns.visualizations.cv_pressure_max_pct_error_barchart",
        set_name,
        geometry_variant,
        "--run_config",
        run_config_suffix,
        "--data_root",
        results_root,
    ]
    result = subprocess.run(cmd, cwd=repo_root(), text=True)
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
                out.append(
                    {
                        "geometry": geo_name,
                        "row_in_geometry": row_in_geo,
                        "feature_name": fname,
                        "value": v,
                        "train_min": tmin,
                        "train_max": tmax,
                        "side": "below",
                    }
                )
            elif v > tmax:
                out.append(
                    {
                        "geometry": geo_name,
                        "row_in_geometry": row_in_geo,
                        "feature_name": fname,
                        "value": v,
                        "train_min": tmin,
                        "train_max": tmax,
                        "side": "above",
                    }
                )
    return out


def _cv_results_paths(set_name, geometry_variant, run_config_suffix):
    out_dir = _results_path("cross_validation", set_name, run_config_suffix)
    summary_path = os.path.join(out_dir, f"{geometry_variant}_cv_summary.csv")
    return out_dir, summary_path


def _cv_subprocess_env(cpu_threads: int | None) -> dict[str, str] | None:
    """Optional per-subprocess OMP/XLA thread cap for parallel CV workers."""
    if cpu_threads is None:
        return None
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={cpu_threads}"
    )
    return env


@dataclass(frozen=True)
class CvTrialSpec:
    trial_id: int
    split_path: str
    val_geometries: tuple[str, ...]


@dataclass(frozen=True)
class CvTrialRunContext:
    set_name: str
    geometry_variant: str
    data_root: str
    run_config_suffix: str
    data_paths_suffix: str
    num_geos: int
    num_trials: int
    jax_path: str
    row_ranges: tuple[tuple[int, int], ...]
    geometries: tuple[str, ...]
    split_indices_dir: str
    model_dir_base: str
    zero_d_base: str
    launch_training_script: str
    mse_csv_name: str
    asymmetric_loss: bool
    use_multi_output_rri: bool
    quadratic_resistor: bool
    nn_vessel: bool
    skip_training_if_exists: bool
    no_redo: bool
    cv_worker_cpu_threads: int | None


def _precompute_cv_splits(
    *,
    trials_to_run: list[int],
    num_trials: int,
    trial_index: int | None,
    set_name: str,
    num_geos: int,
    geometries: list[str],
    num_pts: int,
    data_dict: dict,
    vessel_jax_path: str,
    split_indices_dir: str,
    percent_train: float,
) -> list[CvTrialSpec]:
    """Create or reuse per-trial split files; preserve distinct validation sets across trials."""
    os.makedirs(split_indices_dir, exist_ok=True)
    seen_val_sets: set[frozenset[str]] = set()
    specs: list[CvTrialSpec] = []
    trial_seed_stride = get_pipeline_config().split.cv_trial_seed_stride

    for trial in trials_to_run:
        split_path = os.path.join(
            split_indices_dir,
            f"train_val_ind_{set_name}_num_geos_{num_geos}_trial_{trial}",
        )

        if trial_index is not None and os.path.exists(split_path):
            split_dict = load_split_for_training(split_path)
            val_geometries = list(split_dict["val_geometries"])
            print(
                f"  Trial {trial}: split reused from {os.path.basename(split_path)} "
                f"-> {val_geometries}"
            )
            if val_geometries:
                specs.append(
                    CvTrialSpec(trial_id=trial, split_path=split_path, val_geometries=tuple(val_geometries))
                )
            continue

        max_attempts = 200
        train_geometries: list[str] = []
        val_geometries: list[str] = []
        seed = 0
        for attempt in range(max_attempts):
            seed = trial * trial_seed_stride + attempt
            train_geometries, val_geometries = generate_geometry_split(percent_train, seed, geometries)
            if not val_geometries:
                if attempt == 0:
                    pct = round(percent_train * 100)
                    print(
                        f"  Skipping trial {trial}: no validation geometries "
                        f"({pct}% of {num_geos} rounded to all)"
                    )
                break
            val_set = frozenset(val_geometries)
            if val_set not in seen_val_sets:
                seen_val_sets.add(val_set)
                break
            if attempt == max_attempts - 1:
                pct_val = round((1 - percent_train) * 100)
                pct_train = round(percent_train * 100)
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
        print(f"  Trial {trial}: split {len(train_geometries)} train, {len(val_geometries)} val -> {val_geometries}")
        specs.append(
            CvTrialSpec(trial_id=trial, split_path=split_path, val_geometries=tuple(val_geometries))
        )

    return specs


def _run_single_cv_trial(spec: CvTrialSpec, ctx: CvTrialRunContext) -> dict:
    """Train, deploy, and aggregate metrics for one CV trial."""
    trial = spec.trial_id
    val_geometries = list(spec.val_geometries)
    split_path = spec.split_path

    print(f"\n{'=' * 60}")
    print(f"CV Trial {trial + 1}/{ctx.num_trials}")
    print(f"{'=' * 60}")

    split_dict = load_split_for_training(split_path)
    train_ind = resolve_flat_indices(split_dict, "junction", "train", split_path=split_path)
    val_ind = resolve_flat_indices(split_dict, "junction", "val", split_path=split_path)

    data_dict = load_dict(ctx.jax_path)
    feature_names = get_default_include_features()
    X_input = np.asarray(data_dict["input"])
    row_ranges = [tuple(r) for r in ctx.row_ranges]
    geometries = list(ctx.geometries)
    out_of_range = _check_val_out_of_train_range(
        X_input, train_ind, val_ind, row_ranges, geometries, feature_names
    )
    out_dir_cv = _results_path("cross_validation", ctx.set_name, ctx.data_paths_suffix)
    os.makedirs(out_dir_cv, exist_ok=True)
    oor_csv = os.path.join(out_dir_cv, f"out_of_range_{ctx.geometry_variant}_trial_{trial}.csv")
    with open(oor_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["geometry", "row_in_geometry", "feature_name", "value", "train_min", "train_max", "side"]
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

    model_dir = os.path.join(ctx.model_dir_base, f"{ctx.geometry_variant}_trial_{trial}")
    os.makedirs(model_dir, exist_ok=True)
    subprocess_env = _cv_subprocess_env(ctx.cv_worker_cpu_threads)

    skip_junction = ctx.skip_training_if_exists and rri_models_complete(
        model_dir,
        ctx.set_name,
        vessel=False,
        multi_output=ctx.use_multi_output_rri,
        quadratic_resistor=ctx.quadratic_resistor,
    )
    if skip_junction:
        print(f"  Skipping junction training (models already exist in {model_dir})")
    else:
        cmd_train = [
            sys.executable,
            ctx.launch_training_script,
            "--set_name",
            ctx.set_name,
            "--num_geos",
            str(ctx.num_geos),
            "--geometry_variant",
            ctx.geometry_variant,
            "--split_path",
            split_path,
            "--model_dir",
            model_dir,
        ]
        if ctx.asymmetric_loss:
            cmd_train.append("--asymmetric_loss")
        if ctx.use_multi_output_rri:
            cmd_train.append("--multi_output_rri")
        if ctx.run_config_suffix:
            cmd_train.extend(["--run_config", ctx.run_config_suffix])
        print(f"  Running: {' '.join(cmd_train)}")
        run_kwargs: dict = {"cwd": repo_root(), "text": True}
        if subprocess_env is not None:
            run_kwargs["env"] = subprocess_env
        result_train = subprocess.run(cmd_train, **run_kwargs)
        if result_train.returncode != 0:
            print(f"  Training failed with return code {result_train.returncode}")
            return {
                "trial_id": trial,
                "val_geometries": ",".join(val_geometries),
                "error": "training_failed",
            }

    if ctx.nn_vessel:
        vessel_model_dir = os.path.join(ctx.model_dir_base, f"{ctx.geometry_variant}_vessel_trial_{trial}")
        skip_vessel = ctx.skip_training_if_exists and rri_models_complete(
            vessel_model_dir,
            ctx.set_name,
            vessel=True,
            multi_output=ctx.use_multi_output_rri,
            quadratic_resistor=ctx.quadratic_resistor,
        )
        if skip_vessel:
            print(f"  Skipping vessel training (models already exist in {vessel_model_dir})")
        else:
            cmd_vessel = [
                sys.executable,
                ctx.launch_training_script,
                "--set_name",
                ctx.set_name,
                "--num_geos",
                str(ctx.num_geos),
                "--geometry_variant",
                ctx.geometry_variant,
                "--vessel",
                "--split_path",
                split_path,
                "--model_dir",
                vessel_model_dir,
            ]
            if ctx.asymmetric_loss:
                cmd_vessel.append("--asymmetric_loss")
            if ctx.use_multi_output_rri:
                cmd_vessel.append("--multi_output_rri")
            if ctx.run_config_suffix:
                cmd_vessel.extend(["--run_config", ctx.run_config_suffix])
            print(f"  Running vessel training: {' '.join(cmd_vessel)}")
            run_kwargs = {"cwd": repo_root(), "text": True}
            if subprocess_env is not None:
                run_kwargs["env"] = subprocess_env
            result_vessel = subprocess.run(cmd_vessel, **run_kwargs)
            if result_vessel.returncode != 0:
                print(f"  Vessel training failed with return code {result_vessel.returncode}")
                return {
                    "trial_id": trial,
                    "val_geometries": ",".join(val_geometries),
                    "error": "vessel_training_failed",
                }

    trial_metrics: dict = {}
    for val_geo in val_geometries:
        result_deploy = _run_zerod_inputs_for_cv(
            ctx.set_name,
            val_geo,
            run_config_suffix=ctx.run_config_suffix,
            geometry_variant=ctx.geometry_variant,
            trial_id=trial,
            model_dir=model_dir,
            nn_vessel=ctx.nn_vessel,
            no_redo=ctx.no_redo,
            multi_output_rri=ctx.use_multi_output_rri,
            subprocess_env=subprocess_env,
        )
        if result_deploy.returncode != 0:
            print(f"  Deploy failed for {val_geo} with return code {result_deploy.returncode}")
            continue
        mse_csv_path = os.path.join(ctx.zero_d_base, val_geo, ctx.mse_csv_name)
        accumulate_trial_metrics(trial_metrics, parse_mse_comparison_csv(mse_csv_path))

    return trial_metrics_to_row(trial, ",".join(val_geometries), trial_metrics)


def _run_single_cv_trial_worker(args: tuple[CvTrialSpec, CvTrialRunContext]) -> dict:
    """ProcessPoolExecutor entry point (top-level for pickling)."""
    spec, ctx = args
    return _run_single_cv_trial(spec, ctx)


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
    no_redo=False,
    multi_output_rri=False,
    subprocess_env=None,
):
    """Run generate_zerod_inputs for CV deploy (NN-only) or plot refresh (--plots_only)."""
    ns = argparse.Namespace(
        set_name=set_name,
        geo_name=geo_name,
        run_config=run_config_suffix,
        geometry_variant=geometry_variant,
        trial_id=trial_id,
        model_dir=model_dir,
        NN_only=bool(model_dir) and not plots_only,
        plots_only=plots_only,
        Vessel_NN=nn_vessel,
        verbose=verbose,
        skip_steps="",
        no_redo=no_redo,
        multi_output_rri=multi_output_rri,
    )
    prepare_generate_zerod_namespace(ns)
    cmd = namespace_to_generate_zerod_argv(ns, set_name=set_name, geo_name=geo_name)
    label = "Plots" if plots_only else "Deploy"
    print(f"  {label} on {geo_name}: {' '.join(cmd)}")
    run_kwargs: dict = {"cwd": repo_root(), "text": True}
    if subprocess_env is not None:
        run_kwargs["env"] = subprocess_env
    return subprocess.run(cmd, **run_kwargs)


def run_cv_plots_only(
    set_name,
    geometry_variant,
    data_root="data",
    run_config_suffix="",
    nn_vessel=True,
    multi_output_rri=None,
):
    """Re-run Step 6 plots for each validation geometry listed in the existing CV summary."""
    if multi_output_rri is None:
        multi_output_rri = get_pipeline_config(set_name=set_name).training.multi_output_rri
    _out_dir, summary_path = _cv_results_paths(set_name, geometry_variant, run_config_suffix)
    summary_rows = read_cv_summary_rows(summary_path)
    if not summary_rows:
        print(f"Existing CV summary not found or empty: {summary_path}")
        print("Run cross-validation once, then use --plots_only to regenerate plots.")
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
                multi_output_rri=multi_output_rri,
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
    no_redo=False,
    multi_output_rri=None,
    max_parallel_trials=None,
):
    ml_inputs_root = ml_inputs_root or os.path.join(data_root, "ml_inputs")
    config = _resolve_cv_run_config(run_config_suffix)
    run_config_suffix = config["run_config_suffix"]
    data_paths_suffix = config["data_paths_suffix"]
    quadratic_resistor = config["quadratic_resistor"]
    asymmetric_loss = config["asymmetric_loss"]
    penalty_on = config["penalty_on"]
    pipeline_cfg = get_pipeline_config(set_name=set_name)
    training_cfg = pipeline_cfg.training
    split_cfg = pipeline_cfg.split
    use_multi_output_rri = training_cfg.multi_output_rri if multi_output_rri is None else bool(multi_output_rri)
    parallel_trials = (
        split_cfg.cv_max_parallel_trials if max_parallel_trials is None else int(max_parallel_trials)
    )
    if parallel_trials < 1:
        raise ValueError(f"max_parallel_trials must be >= 1, got {parallel_trials}")
    print(f"Run config: {data_paths_suffix!r}")
    if parallel_trials > 1:
        print(f"CV parallel trials: {parallel_trials} workers")
        if split_cfg.cv_worker_cpu_threads is not None:
            print(f"  cv_worker_cpu_threads: {split_cfg.cv_worker_cpu_threads}")

    _ensure_cv_prerequisites(
        set_name,
        geometry_variant,
        data_root,
        set_type,
        data_paths_suffix,
        no_redo=no_redo,
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
    paths = _cv_run_directory_paths(data_root, set_name, geometry_variant, data_paths_suffix, set_type)
    split_indices_dir = paths["split_indices_dir"]
    model_dir_base = paths["model_dir_base"]
    zero_d_base = paths["zero_d_base"]
    launch_training_script = paths["launch_training_script"]
    mse_csv_name = paths["mse_csv_name"]
    jax_path = _jax_pickle_path(
        data_root, set_name, data_paths_suffix, geometry_variant, set_type, num_geos
    )

    # Optionally run only one trial (0-based index)
    if trial_index is not None:
        if trial_index < 0 or trial_index >= num_trials:
            raise ValueError(f"trial_index must be in [0, {num_trials}), got {trial_index}")
        trials_to_run = [trial_index]
        print(f"Re-running single trial {trial_index} (of {num_trials})")
    else:
        trials_to_run = list(range(num_trials))
    if nn_vessel:
        print("Vessel_NN: will train vessel NN per trial and include vessel-predicted modality in MSE")
    if use_multi_output_rri:
        print("Multi-output RRI: one network with R, S, L outputs")
    if asymmetric_loss:
        print("Asymmetric loss: per-model overestimate weights")
    if quadratic_resistor:
        print("Quadratic resistor: calibrate stenosis coefficient; NN predicts stenosis")
    if penalty_on:
        print("Penalty-on: L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient enabled during calibration")

    print("\nPre-computing CV splits...")
    trial_specs = _precompute_cv_splits(
        trials_to_run=trials_to_run,
        num_trials=num_trials,
        trial_index=trial_index,
        set_name=set_name,
        num_geos=num_geos,
        geometries=geometries,
        num_pts=num_pts,
        data_dict=data_dict,
        vessel_jax_path=vessel_jax_path,
        split_indices_dir=split_indices_dir,
        percent_train=percent_train,
    )
    if not trial_specs:
        print("No trial splits to run.")
        return

    trial_ctx = CvTrialRunContext(
        set_name=set_name,
        geometry_variant=geometry_variant,
        data_root=data_root,
        run_config_suffix=run_config_suffix,
        data_paths_suffix=data_paths_suffix,
        num_geos=num_geos,
        num_trials=num_trials,
        jax_path=jax_path,
        row_ranges=tuple(tuple(r) for r in row_ranges),
        geometries=tuple(geometries),
        split_indices_dir=split_indices_dir,
        model_dir_base=model_dir_base,
        zero_d_base=zero_d_base,
        launch_training_script=launch_training_script,
        mse_csv_name=mse_csv_name,
        asymmetric_loss=asymmetric_loss,
        use_multi_output_rri=use_multi_output_rri,
        quadratic_resistor=quadratic_resistor,
        nn_vessel=nn_vessel,
        skip_training_if_exists=skip_training_if_exists,
        no_redo=no_redo,
        cv_worker_cpu_threads=split_cfg.cv_worker_cpu_threads if parallel_trials > 1 else None,
    )

    all_trial_results: list[dict] = []
    if parallel_trials == 1:
        for spec in trial_specs:
            all_trial_results.append(_run_single_cv_trial(spec, trial_ctx))
    else:
        worker_args = [(spec, trial_ctx) for spec in trial_specs]
        with ProcessPoolExecutor(max_workers=parallel_trials) as executor:
            futures = [executor.submit(_run_single_cv_trial_worker, args) for args in worker_args]
            for future in as_completed(futures):
                all_trial_results.append(future.result())
        all_trial_results.sort(key=lambda r: int(r.get("trial_id", 0)))

    if not all_trial_results:
        print("No trial results to summarize.")
        return

    out_dir, summary_path = _cv_results_paths(set_name, geometry_variant, data_paths_suffix)

    # If we re-ran a single trial and summary already exists, merge this result into it
    if trial_index is not None and os.path.exists(summary_path):
        existing_by_trial = {}
        with open(summary_path, newline="") as f:
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
                    existing_by_trial[tid] = {
                        "trial_id": tid,
                        "val_geometries": row[1] if len(row) > 1 else "",
                    }
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
    print(
        "  Also wrote: pressure_mse, flow_mse, max_error, pressure_max_error, "
        "flow_max_error, max_rel_error, pressure_max_rel_error, flow_max_rel_error, "
        "mean_rel_error, pressure_mean_rel_error, flow_mean_rel_error"
    )
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
        print("Run cross-validation once to create it, then use --metrics_only to refresh metric CSVs.")
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
        all_trial_results.append(trial_metrics_to_row(trial_id, val_geometries_str, trial_metrics))

    if not all_trial_results:
        print("No trial rows found in existing summary.")
        return None

    _, modalities = write_all_cv_summary_csvs(summary_path, out_dir, geometry_variant, all_trial_results)

    print("Regenerated CV metrics from existing per-geometry MSE CSVs.")
    print(f"  Updated: {summary_path}")
    base = os.path.join(out_dir, f"{geometry_variant}_cv_summary")
    print(
        f"  Wrote:   {base}_pressure_mse.csv, _flow_mse.csv, _max_error.csv, "
        "_pressure_max_error.csv, _flow_max_error.csv, _max_rel_error.csv, "
        "_pressure_max_rel_error.csv, _flow_max_rel_error.csv, "
        "_mean_rel_error.csv, _pressure_mean_rel_error.csv, _flow_mean_rel_error.csv"
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
    split_defaults = get_pipeline_config().split
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-validation: X random 90/10 splits, train and deploy per trial, report MSE for all modalities."
        )
    )
    parser.add_argument(
        "--set_name",
        required=True,
        help="Set name (e.g., VMR_rigid_aorta_adults)",
    )
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations_EL",
        help="Geometry variant (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=split_defaults.cv_num_trials,
        help=f"Number of random CV splits (default: {split_defaults.cv_num_trials} from config)",
    )
    parser.add_argument("--data_root", default="data", help="Data root (default: data)")
    parser.add_argument(
        "--set_type",
        default="all",
        help="Cohort folder tier for jax/split paths (default: all)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        metavar="N",
        help="Re-run only trial N (0-based). Merges result into existing CV summary if present.",
    )
    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        metavar="TOKENS",
        help=(
            "Run-config tokens in any order, underscore-separated "
            f"(default: {DEFAULT_CLI_RUN_CONFIG}). "
            "Full canonical suffixes are also accepted."
        ),
    )
    parser.add_argument(
        "--Vessel_NN",
        action="store_true",
        dest="nn_vessel",
        default=True,
        help=(
            "Train vessel NN per trial and run vessel NN inference "
            "(junction+vessel and vessel-only modalities in MSE). Default: True."
        ),
    )
    parser.add_argument(
        "--no_Vessel_NN",
        action="store_false",
        dest="nn_vessel",
        help="Disable vessel NN training and inference (junction NN only).",
    )
    parser.add_argument(
        "--skip_training_if_exists",
        action="store_true",
        help="Skip junction and/or vessel training for a trial if the corresponding model files already exist.",
    )
    parser.add_argument(
        "--multi_output_rri",
        action="store_true",
        help=(
            "Train one network with R/S/L outputs instead of three separate networks. "
            "Default follows training.multi_output_rri in config (false)."
        ),
    )
    parser.add_argument(
        "--no_redo",
        action="store_true",
        help=(
            "During prerequisite batch generation and per-trial NN deploy, skip recreating "
            "zeroD files that already exist (passed through to batch/ generate_zerod_inputs)."
        ),
    )
    parser.add_argument(
        "--percent_train",
        type=float,
        default=split_defaults.percent_train,
        metavar="P",
        help=(
            f"Fraction of geometries for training (0–1); remainder used for validation "
            f"(default: {split_defaults.percent_train} from config)."
        ),
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help=(
            "Regenerate CV summary CSVs (overall MSE + pressure/flow MSE + max error) "
            "from existing cv_summary.csv and per-geometry mse_comparison.csv files. "
            "No training or deploy."
        ),
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help=(
            "Regenerate comparison plots for CV validation geometries via "
            "generate_zerod_inputs --plots_only. No training or deploy."
        ),
    )
    parser.add_argument(
        "--skip_barchart",
        action="store_true",
        help="Do not run cv_pressure_max_pct_error_barchart after writing CV summary CSVs.",
    )
    parser.add_argument(
        "--max_parallel_trials",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run up to N CV trials in parallel (default: split.cv_max_parallel_trials from config, usually 1). "
            "CLI value overrides config."
        ),
    )
    args = parser.parse_args()

    try:
        run_config_suffix = resolve_run_config_suffix(args.run_config)
    except ValueError as exc:
        parser.error(str(exc))
    run_config_suffix_to_flags(run_config_suffix)
    if args.metrics_only and args.plots_only:
        parser.error("Cannot use both --metrics_only and --plots_only.")
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
        no_redo=args.no_redo,
        multi_output_rri=args.multi_output_rri or get_pipeline_config(set_name=args.set_name).training.multi_output_rri,
        max_parallel_trials=args.max_parallel_trials,
    )


if __name__ == "__main__":
    main()
