#!/usr/bin/env python3
"""
Wrapper to run cross-validation for all default VMR set names (cohorts), then an
optional cross-set summary barchart.

Usage:
  learn-lpns-cv-all-sets
  python -m learn_lpns.zerod_calibration.run_cv_all_sets
  python -m learn_lpns.zerod_calibration.run_cv_all_sets --set_names VMR_abdo VMR_all
"""

import argparse
import os
import subprocess
import sys

from learn_lpns.config import get_pipeline_config
from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    resolve_run_config_suffix,
    run_config_suffix_to_flags,
)
from learn_lpns.zerod_calibration.run_cross_validation import (
    _generate_cv_barcharts,
    regenerate_cv_metrics_from_existing,
    run_cross_validation,
    run_cv_plots_only,
)

os.chdir(repo_root())


def run_cv_all_sets(
    set_names,
    geometry_variant="bifurcations_EL",
    num_trials=None,
    data_root="data",
    set_type="all",
    trial_index=None,
    nn_vessel=True,
    skip_training_if_exists=False,
    percent_train=None,
    run_config_suffix=None,
    skip_barchart=False,
    skip_cross_set_barchart=False,
    metrics_only=False,
    plots_only=False,
):
    split_defaults = get_pipeline_config().split
    if num_trials is None:
        num_trials = split_defaults.cv_num_trials
    if percent_train is None:
        percent_train = split_defaults.percent_train

    run_config_suffix = resolve_run_config_suffix(run_config_suffix or DEFAULT_CLI_RUN_CONFIG)
    run_config_suffix_to_flags(run_config_suffix)

    if metrics_only and plots_only:
        raise ValueError("Cannot use both metrics_only and plots_only.")

    for set_name in set_names:
        print(f"\n{'=' * 60}")
        print(f"Running cross-validation: set_name = {set_name}")
        print(f"{'=' * 60}")

        if plots_only:
            run_cv_plots_only(
                set_name=set_name,
                geometry_variant=geometry_variant,
                data_root=data_root,
                run_config_suffix=run_config_suffix,
                nn_vessel=nn_vessel,
            )
            continue

        if metrics_only:
            regenerate_cv_metrics_from_existing(
                set_name=set_name,
                geometry_variant=geometry_variant,
                data_root=data_root,
                run_config_suffix=run_config_suffix,
            )
            if not skip_barchart:
                _generate_cv_barcharts(set_name, geometry_variant, run_config_suffix)
            continue

        run_cross_validation(
            set_name=set_name,
            geometry_variant=geometry_variant,
            num_trials=num_trials,
            data_root=data_root,
            set_type=set_type,
            trial_index=trial_index,
            nn_vessel=nn_vessel,
            skip_training_if_exists=skip_training_if_exists,
            percent_train=percent_train,
            run_config_suffix=run_config_suffix,
            skip_barchart=skip_barchart,
        )

    if skip_cross_set_barchart or plots_only:
        return

    print(f"\n{'=' * 60}")
    print("Running cross-set summary barchart...")
    print(f"{'=' * 60}")
    cmd = [
        sys.executable,
        "-m",
        "learn_lpns.visualizations.cv_cross_set_summary_barchart",
        "--set_names",
        *set_names,
        "--geometry_variant",
        geometry_variant,
        "--run_config",
        run_config_suffix,
    ]
    ret = subprocess.run(cmd, cwd=repo_root())
    if ret.returncode != 0:
        raise RuntimeError(f"Cross-set summary barchart failed (exit {ret.returncode}).")


def main():
    split_defaults = get_pipeline_config().split
    default_set_names = list(get_pipeline_config().cohorts.default_cv_set_names)
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-validation for all default VMR set names, "
            "then an optional cross-set summary barchart."
        )
    )
    parser.add_argument(
        "--set_names",
        nargs="+",
        default=None,
        metavar="SET_NAME",
        help=(
            "Set names to run (default: "
            + " ".join(default_set_names)
            + ")."
        ),
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
        help="Re-run only trial N (0-based) for each set.",
    )
    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        metavar="TOKENS",
        help=(
            "Run-config tokens in any order, underscore-separated "
            f"(default: {DEFAULT_CLI_RUN_CONFIG})."
        ),
    )
    parser.add_argument(
        "--Vessel_NN",
        action="store_true",
        dest="nn_vessel",
        default=True,
        help="Train vessel NN per trial (default: True).",
    )
    parser.add_argument(
        "--no_Vessel_NN",
        action="store_false",
        dest="nn_vessel",
        help="Disable vessel NN training and inference.",
    )
    parser.add_argument(
        "--skip_training_if_exists",
        action="store_true",
        help="Skip training for a trial if model files already exist.",
    )
    parser.add_argument(
        "--percent_train",
        type=float,
        default=split_defaults.percent_train,
        metavar="P",
        help=(
            f"Fraction of geometries for training (default: {split_defaults.percent_train} from config)."
        ),
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Regenerate CV summary CSVs from existing MSE files for each set (no train/deploy).",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Regenerate comparison plots for CV validation geometries for each set.",
    )
    parser.add_argument(
        "--skip_barchart",
        action="store_true",
        help="Do not run per-set cv_pressure_max_pct_error_barchart after each set.",
    )
    parser.add_argument(
        "--skip_cross_set_barchart",
        action="store_true",
        help="Do not run cv_cross_set_summary_barchart after all sets finish.",
    )
    args = parser.parse_args()

    set_names = args.set_names or default_set_names
    try:
        run_cv_all_sets(
            set_names=set_names,
            geometry_variant=args.geometry_variant,
            num_trials=args.num_trials,
            data_root=args.data_root,
            set_type=args.set_type,
            trial_index=args.trial,
            nn_vessel=args.nn_vessel,
            skip_training_if_exists=args.skip_training_if_exists,
            percent_train=args.percent_train,
            run_config_suffix=args.run_config,
            skip_barchart=args.skip_barchart,
            skip_cross_set_barchart=args.skip_cross_set_barchart,
            metrics_only=args.metrics_only,
            plots_only=args.plots_only,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
