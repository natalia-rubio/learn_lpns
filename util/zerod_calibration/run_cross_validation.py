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
from util.tools.basic import load_dict, save_dict


def _parse_mse_csv(csv_path):
    """Parse MSE comparison CSV; return dict modality -> overall_mse (float or nan)."""
    result = {}
    if not os.path.exists(csv_path):
        return result
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
            if in_summary and header is not None and row[0] == "Overall MSE":
                for i, mod in enumerate(header[1:], start=1):
                    if i < len(row):
                        try:
                            result[mod] = float(row[i])
                        except ValueError:
                            result[mod] = np.nan
                    else:
                        result[mod] = np.nan
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

    all_trial_results = []  # list of dicts: trial_id, val_geometries, mod -> overall_mse
    seen_val_sets = set()  # frozenset of val geometry names, to ensure each trial has a different val set

    for trial in trials_to_run:
        print(f"\n{'='*60}")
        print(f"CV Trial {trial + 1}/{num_trials}")
        print(f"{'='*60}")

        # Ensure this trial's validation set is different from all previous trials
        max_attempts = 200
        for attempt in range(max_attempts):
            seed = trial * 1000 + attempt
            train_ind, val_ind, train_geo_idx, val_geo_idx = generate_split_indices(
                num_pts=num_pts,
                percent_train=0.9,
                seed=seed,
                geometry_row_ranges=row_ranges,
            )
            train_geometries = [geometries[i] for i in train_geo_idx]
            val_geometries = [geometries[i] for i in val_geo_idx]
            if not val_geometries:
                if attempt == 0:
                    print(f"  Skipping trial {trial}: no validation geometries (90% of {num_geos} rounded to all)")
                break
            val_set = frozenset(val_geometries)
            if val_set not in seen_val_sets:
                seen_val_sets.add(val_set)
                break
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Could not get a distinct validation set for trial {trial} after {max_attempts} attempts. "
                    f"Not enough geometries for {num_trials} unique 90/10 splits."
                )
        if not val_geometries:
            continue

        split_path = os.path.join(
            split_indices_dir,
            f"train_val_ind_{set_name}_num_geos_{num_geos}_trial_{trial}",
        )
        os.makedirs(split_indices_dir, exist_ok=True)
        split_dict = {
            "train_ind": np.asarray(train_ind, dtype=int),
            "val_ind": np.asarray(val_ind, dtype=int),
            "num_offsets": 1,
            "percent_train": 0.9,
            "seed": int(seed),
            "num_pts": num_pts,
            "split_by_geometry": True,
            "train_geometries": train_geometries,
            "val_geometries": val_geometries,
        }
        save_dict(split_dict, split_path)
        print(f"  Split: {len(train_geometries)} train, {len(val_geometries)} val -> {val_geometries}")

        model_dir = os.path.join(
            model_dir_base, f"{geometry_variant}{norm_suffix}_trial_{trial}"
        )
        os.makedirs(model_dir, exist_ok=True)

        # Train
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
        print(f"  Running: {' '.join(cmd_train)}")
        result_train = subprocess.run(cmd_train, cwd=REPO_ROOT, text=True)
        if result_train.returncode != 0:
            print(f"  Training failed with return code {result_train.returncode}")
            all_trial_results.append(
                {"trial_id": trial, "val_geometries": ",".join(val_geometries), "error": "training_failed"}
            )
            continue

        # Deploy on each validation geometry (NN-only)
        trial_mse = {}  # modality -> list of overall_mse per val geo
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
            print(f"  Deploy on {val_geo}: {' '.join(cmd_deploy)}")
            result_deploy = subprocess.run(cmd_deploy, cwd=REPO_ROOT, text=True)
            if result_deploy.returncode != 0:
                print(f"  Deploy failed for {val_geo} with return code {result_deploy.returncode}")
                continue
            mse_csv_path = os.path.join(zero_d_base, val_geo, mse_csv_name)
            mod_mse = _parse_mse_csv(mse_csv_path)
            for mod, val in mod_mse.items():
                if mod not in trial_mse:
                    trial_mse[mod] = []
                trial_mse[mod].append(val)

        # Aggregate per trial (mean across val geometries per modality)
        row = {"trial_id": trial, "val_geometries": ",".join(val_geometries)}
        for mod, vals in trial_mse.items():
            row[f"MSE_{mod}"] = np.nanmean(vals) if vals else np.nan
        all_trial_results.append(row)

    if not all_trial_results:
        print("No trial results to summarize.")
        return

    # Build summary: one row per trial + final mean ± std per modality
    modalities = set()
    for r in all_trial_results:
        for k in r:
            if k.startswith("MSE_"):
                modalities.add(k)
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
        modalities = sorted(set().union(*(set(k for k in r if k.startswith("MSE_")) for r in all_trial_results)))

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trial_id", "val_geometries"] + modalities
        writer.writerow(header)
        for r in all_trial_results:
            if "error" in r:
                writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [""] * len(modalities))
            else:
                writer.writerow([r.get("trial_id", ""), r.get("val_geometries", "")] + [r.get(m, "") for m in modalities])
        writer.writerow([])
        mean_row = ["mean", ""]
        std_row = ["std", ""]
        for m in modalities:
            vals = [r.get(m) for r in all_trial_results if "error" not in r and m in r]
            vals = [float(v) for v in vals if v is not None and (not isinstance(v, float) or not np.isnan(v))]
            mean_row.append(np.nanmean(vals) if vals else np.nan)
            std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
        writer.writerow(mean_row)
        writer.writerow(std_row)

    print(f"\nWrote CV summary to {summary_path}")
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
    args = parser.parse_args()

    run_cross_validation(
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        num_trials=args.num_trials,
        data_root=args.data_root,
        set_type=args.set_type,
        trial_index=args.trial,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
