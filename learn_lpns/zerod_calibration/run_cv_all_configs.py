#!/usr/bin/env python3
"""
Wrapper to run cross-validation for multiple configs (e.g. gen_loss,
quadratic_resistor_gen_loss), then the per-config
max-pct-error barchart after each CV, and finally the by-config comparison barchart.

Usage:
  python -m learn_lpns.zerod_calibration.run_cv_all_configs
  python -m learn_lpns.zerod_calibration.run_cv_all_configs VMR_rigid_aorta_adults bifurcations_EL 5
  python -m learn_lpns.zerod_calibration.run_cv_all_configs --only_barcharts   # skip CV, only run barcharts
"""

import argparse
import os
import subprocess
import sys

from learn_lpns.config import get_pipeline_config
from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.cv_batch_common import (
    DEFAULT_CONFIGS,
    OPTIONAL_CONFIGS,
    run_by_config_comparison_barchart,
    resolve_configs_with_flags,
)

os.chdir(repo_root())  # ensure cwd is repo root for -m invocations


def main():
    parser = argparse.ArgumentParser(description="Run CV for several configs, then per-config and by-config barcharts.")
    parser.add_argument(
        "set_name",
        nargs="?",
        default="VMR_rigid_aorta_adults",
        help="Set name (default: VMR_rigid_aorta_adults)",
    )
    parser.add_argument(
        "geometry_variant",
        nargs="?",
        default="bifurcations_EL",
        help="Geometry variant (default: bifurcations_EL)",
    )
    cv_num_trials_default = get_pipeline_config().split.cv_num_trials
    parser.add_argument(
        "num_trials",
        nargs="?",
        type=int,
        default=cv_num_trials_default,
        help=f"Number of CV trials (default: {cv_num_trials_default} from config)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="CONFIG",
        help="Config entries to run. Use 'config' or 'config:geometry_variant' (e.g. gen_loss:bifurcations).",
    )
    parser.add_argument(
        "--only_barcharts",
        action="store_true",
        help="Skip cross-validation; only run per-config and by-config barcharts (assumes CV already done).",
    )
    parser.add_argument(
        "--skip_per_config_barchart",
        action="store_true",
        help="Pass --skip_barchart to run_cross_validation (CV generates barcharts by default).",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Data root for run_cross_validation (default: data)",
    )
    args = parser.parse_args()

    set_name = args.set_name
    geometry_variant = args.geometry_variant
    num_trials = args.num_trials
    try:
        configs_with_flags = resolve_configs_with_flags(args.configs, geometry_variant)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if not args.only_barcharts:
        for entry_key, run_config_suffix, cfg_geometry_variant, cv_flags in configs_with_flags:
            print(f"\n{'=' * 60}")
            print(
                f"Running cross-validation: config = {entry_key} "
                f"(run-config={run_config_suffix}, geometry={cfg_geometry_variant})"
            )
            print(f"{'=' * 60}")
            cmd = [
                sys.executable,
                "-m",
                "learn_lpns.zerod_calibration.run_cross_validation",
                "--set_name",
                set_name,
                "--geometry_variant",
                cfg_geometry_variant,
                "--num_trials",
                str(num_trials),
                "--data_root",
                args.data_root,
                *cv_flags,
            ]
            if args.skip_per_config_barchart:
                cmd.append("--skip_barchart")
            ret = subprocess.run(cmd, cwd=repo_root())
            if ret.returncode != 0:
                print(
                    f"Cross-validation failed for config {entry_key} (exit {ret.returncode}). Stopping.",
                    file=sys.stderr,
                )
                sys.exit(ret.returncode)
    else:
        if not args.skip_per_config_barchart:
            for entry_key, run_config_suffix, cfg_geometry_variant, _ in configs_with_flags:
                print(f"\nRunning per-config barchart for {entry_key}...")
                cmd_barchart = [
                    sys.executable,
                    "-m",
                    "learn_lpns.visualizations.cv_pressure_max_pct_error_barchart",
                    set_name,
                    cfg_geometry_variant,
                    "--run_config",
                    run_config_suffix,
                    "--data_root",
                    "results",
                ]
                subprocess.run(cmd_barchart, cwd=repo_root())

    print(f"\n{'=' * 60}")
    print("Running by-config comparison barchart...")
    print(f"{'=' * 60}")
    ret_by = run_by_config_comparison_barchart(
        set_name,
        geometry_variant=geometry_variant,
        configs_with_flags=configs_with_flags,
    )
    if ret_by != 0:
        print(f"By-config barchart failed (exit {ret_by}).", file=sys.stderr)
        sys.exit(ret_by)
    print("Done.")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIGS", "OPTIONAL_CONFIGS", "main"]
