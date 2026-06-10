#!/usr/bin/env python3
"""
Wrapper to run cross-validation for multiple configs (e.g. gen_loss,
quadratic_resistor_gen_loss, quadratic_resistor_penalty_on_gen_loss), then the per-config
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

os.chdir(repo_root())  # ensure cwd is repo root for -m invocations

# Configs to run: (run_config_suffix, list of run_cross_validation flags)
# Order is the order of execution; barchart by-config will discover and sort by value.
# Use --run_config SUFFIX as the single way to specify config.
DEFAULT_CONFIGS = [
    ("gen_loss", ["--run_config", "gen_loss"]),
    ("base", ["--run_config", "base"]),
    ("quadratic_resistor_gen_loss", ["--run_config", "quadratic_resistor_gen_loss"]),
    (
        "quadratic_resistor_penalty_on_gen_loss",
        ["--run_config", "quadratic_resistor_penalty_on_gen_loss"],
    ),
    ("gen_loss:bifurcations", ["--run_config", "gen_loss"]),
]

# Valid for e.g. `--configs quadratic_resistor_penalty_on_gen_loss` but not part of the default batch.
OPTIONAL_CONFIGS = {
    "quadratic_resistor_gen_loss": ["--run_config", "quadratic_resistor_gen_loss"],
    "quadratic_resistor_penalty_on_gen_loss": [
        "--run_config",
        "quadratic_resistor_penalty_on_gen_loss",
    ],
    "base": ["--run_config", "base"],
    "gen_loss": ["--run_config", "gen_loss"],
    "quadratic_resistor": ["--run_config", "quadratic_resistor"],
    "quadratic_resistor_penalty_on": ["--run_config", "quadratic_resistor_penalty_on"],
}


def _parse_config_entry(entry: str, default_geometry_variant: str):
    """
    Parse config entry syntax:
      - "config_suffix" -> (entry_key, run_config_suffix, geometry_variant_override)
      - "config_suffix:geometry_variant" -> (entry_key, run_config_suffix, geometry_variant_override)
    """
    raw = (entry or "").strip()
    if not raw:
        raise ValueError("Empty config entry.")
    if ":" in raw:
        run_cfg, geom_var = raw.split(":", 1)
        run_cfg = run_cfg.strip()
        geom_var = geom_var.strip()
        if not run_cfg or not geom_var:
            raise ValueError(f"Invalid config entry {entry!r}. Use 'config' or 'config:geometry_variant'.")
        return raw, run_cfg, geom_var
    return raw, raw, default_geometry_variant


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
    config_list = args.configs or [c[0] for c in DEFAULT_CONFIGS]
    # Build (entry_key, run_config_suffix, geometry_variant_to_use, cv_flags) for each requested config
    config_map = {**dict(DEFAULT_CONFIGS), **OPTIONAL_CONFIGS}
    configs_with_flags = []
    for c in config_list:
        try:
            entry_key, run_config_suffix, cfg_geometry_variant = _parse_config_entry(c, geometry_variant)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        if run_config_suffix not in config_map:
            print(
                f"Unknown config: {run_config_suffix}. Known: {list(config_map.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        configs_with_flags.append((entry_key, run_config_suffix, cfg_geometry_variant, config_map[run_config_suffix]))

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
        # Only barcharts: run per-config barchart for each config that has data
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
    by_config_list = [c[0] for c in configs_with_flags]
    # Extra bar: gen_loss config but using bifurcations (not bifurcations_EL) error CSV
    if "gen_loss" in by_config_list:
        by_config_list.append("gen_loss:bifurcations")
    cmd_by = [
        sys.executable,
        "-m",
        "learn_lpns.visualizations.cv_max_pct_error_by_config_barchart",
        set_name,
        "--geometry",
        geometry_variant,
        "--configs",
        *by_config_list,
        "--data_root",
        "results",
        "--xmax",
        "40",
    ]
    ret_by = subprocess.run(cmd_by, cwd=repo_root())
    if ret_by.returncode != 0:
        print(f"By-config barchart failed (exit {ret_by.returncode}).", file=sys.stderr)
        sys.exit(ret_by.returncode)
    print("Done.")


if __name__ == "__main__":
    main()
