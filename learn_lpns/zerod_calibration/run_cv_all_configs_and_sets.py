#!/usr/bin/env python3
"""
Run cross-validation for every combination of default cohorts × default run configs,
continuing on failure and generating summary barcharts at the end.

Usage:
  learn-lpns-cv-all-configs-and-sets
  python -m learn_lpns.zerod_calibration.run_cv_all_configs_and_sets
  python -m learn_lpns.zerod_calibration.run_cv_all_configs_and_sets \\
      --set_names VMR_pulmo --configs gen_loss quadratic_resistor_gen_loss
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from learn_lpns.config import get_pipeline_config
from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.cv_batch_common import (
    resolve_configs_with_flags,
    run_by_config_comparison_barchart,
    run_cross_set_summary_barchart,
    unique_run_config_suffixes,
)
from learn_lpns.zerod_calibration.run_cross_validation import (
    _generate_cv_barcharts,
    regenerate_cv_metrics_from_existing,
    run_cross_validation,
    run_cv_plots_only,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from learn_lpns.zerod_calibration.cv_batch_common import ConfigWithFlags

os.chdir(repo_root())

FailureRecord = tuple[str, str, str]


def run_cv_all_configs_and_sets(
    set_names: Sequence[str],
    configs_with_flags: Sequence[ConfigWithFlags],
    *,
    geometry_variant: str = "bifurcations_EL",
    num_trials: int | None = None,
    data_root: str = "data",
    set_type: str = "all",
    trial_index: int | None = None,
    nn_vessel: bool = True,
    skip_training_if_exists: bool = False,
    percent_train: float | None = None,
    skip_barchart: bool = False,
    skip_by_config_barchart: bool = False,
    skip_cross_set_barchart: bool = False,
    only_barcharts: bool = False,
    metrics_only: bool = False,
    plots_only: bool = False,
) -> int:
    split_defaults = get_pipeline_config().split
    if num_trials is None:
        num_trials = split_defaults.cv_num_trials
    if percent_train is None:
        percent_train = split_defaults.percent_train

    if metrics_only and plots_only:
        raise ValueError("Cannot use both metrics_only and plots_only.")

    failures: list[FailureRecord] = []

    for set_name in set_names:
        print(f"\n{'=' * 60}")
        print(f"Set: {set_name}")
        print(f"{'=' * 60}")

        for entry_key, run_config_suffix, cfg_geometry_variant, _ in configs_with_flags:
            print(f"\n--- config = {entry_key} (run-config={run_config_suffix}, geometry={cfg_geometry_variant}) ---")
            try:
                if only_barcharts:
                    if not skip_barchart:
                        _generate_cv_barcharts(set_name, cfg_geometry_variant, run_config_suffix)
                elif plots_only:
                    run_cv_plots_only(
                        set_name=set_name,
                        geometry_variant=cfg_geometry_variant,
                        data_root=data_root,
                        run_config_suffix=run_config_suffix,
                        nn_vessel=nn_vessel,
                    )
                elif metrics_only:
                    regenerate_cv_metrics_from_existing(
                        set_name=set_name,
                        geometry_variant=cfg_geometry_variant,
                        data_root=data_root,
                        run_config_suffix=run_config_suffix,
                    )
                    if not skip_barchart:
                        _generate_cv_barcharts(set_name, cfg_geometry_variant, run_config_suffix)
                else:
                    result = run_cross_validation(
                        set_name=set_name,
                        geometry_variant=cfg_geometry_variant,
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
                    if result is None:
                        failures.append((set_name, entry_key, "no trial results"))
            except Exception as exc:
                failures.append((set_name, entry_key, str(exc)))
                print(f"FAILED: set={set_name!r}, config={entry_key!r}: {exc}", file=sys.stderr)

        if not skip_by_config_barchart and not plots_only:
            print(f"\nRunning by-config comparison barchart for {set_name}...")
            ret_by = run_by_config_comparison_barchart(
                set_name,
                geometry_variant=geometry_variant,
                configs_with_flags=configs_with_flags,
            )
            if ret_by != 0:
                print(
                    f"Warning: by-config barchart failed for {set_name} (exit {ret_by}).",
                    file=sys.stderr,
                )

    if not skip_cross_set_barchart and not plots_only:
        for run_config_suffix in unique_run_config_suffixes(configs_with_flags):
            print(f"\n{'=' * 60}")
            print(f"Cross-set summary barchart (run-config={run_config_suffix})...")
            print(f"{'=' * 60}")
            ret_cross = run_cross_set_summary_barchart(
                set_names,
                run_config_suffix=run_config_suffix,
                geometry_variant=geometry_variant,
            )
            if ret_cross != 0:
                print(
                    f"Warning: cross-set barchart failed for {run_config_suffix} (exit {ret_cross}).",
                    file=sys.stderr,
                )

    if failures:
        print(f"\n{'=' * 60}")
        print(f"Summary: {len(failures)} job(s) failed")
        print(f"{'=' * 60}")
        for set_name, entry_key, reason in failures:
            print(f"  set={set_name!r}  config={entry_key!r}  reason={reason}")
        return 1

    return 0


def main() -> None:
    split_defaults = get_pipeline_config().split
    default_set_names = list(get_pipeline_config().cohorts.default_cv_set_names)

    parser = argparse.ArgumentParser(
        description=(
            "Run cross-validation for every default cohort × default run config, "
            "continuing on failure and generating summary barcharts at the end."
        )
    )
    parser.add_argument(
        "--set_names",
        nargs="+",
        default=None,
        metavar="SET_NAME",
        help=("Set names to run (default: " + " ".join(default_set_names) + ")."),
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="CONFIG",
        help="Config entries to run. Use 'config' or 'config:geometry_variant' (e.g. gen_loss:bifurcations).",
    )
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations_EL",
        help="Default geometry when a config entry has no :override (default: bifurcations_EL)",
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
        help="Re-run only trial N (0-based) for each set/config pair.",
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
        help=(f"Fraction of geometries for training (default: {split_defaults.percent_train} from config)."),
    )
    parser.add_argument(
        "--only_barcharts",
        action="store_true",
        help="Skip cross-validation; only regenerate barcharts (assumes CV already done).",
    )
    parser.add_argument(
        "--skip_barchart",
        action="store_true",
        help="Skip per-run cv_pressure_max_pct_error_barchart after each CV job.",
    )
    parser.add_argument(
        "--skip_by_config_barchart",
        action="store_true",
        help="Skip per-set cv_max_pct_error_by_config_barchart after each set finishes.",
    )
    parser.add_argument(
        "--skip_cross_set_barchart",
        action="store_true",
        help="Skip per-config cv_cross_set_summary_barchart after all sets finish.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Regenerate CV summary CSVs from existing MSE files for each set/config (no train/deploy).",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Regenerate comparison plots for CV validation geometries for each set/config.",
    )
    args = parser.parse_args()

    set_names = args.set_names or default_set_names
    try:
        configs_with_flags = resolve_configs_with_flags(args.configs, args.geometry_variant)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        exit_code = run_cv_all_configs_and_sets(
            set_names=set_names,
            configs_with_flags=configs_with_flags,
            geometry_variant=args.geometry_variant,
            num_trials=args.num_trials,
            data_root=args.data_root,
            set_type=args.set_type,
            trial_index=args.trial,
            nn_vessel=args.nn_vessel,
            skip_training_if_exists=args.skip_training_if_exists,
            percent_train=args.percent_train,
            skip_barchart=args.skip_barchart,
            skip_by_config_barchart=args.skip_by_config_barchart,
            skip_cross_set_barchart=args.skip_cross_set_barchart,
            only_barcharts=args.only_barcharts,
            metrics_only=args.metrics_only,
            plots_only=args.plots_only,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if exit_code == 0:
        print("Done.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
