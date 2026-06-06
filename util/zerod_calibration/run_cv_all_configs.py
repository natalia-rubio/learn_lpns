#!/usr/bin/env python3
"""
Wrapper to run cross-validation for multiple configs (e.g. base, stenosis_off, penalty_off,
stenosis_off_symmetric), then the per-config max-pct-error barchart after each CV, and finally
the by-config comparison barchart.

Usage:
  python -m util.zerod_calibration.run_cv_all_configs
  python -m util.zerod_calibration.run_cv_all_configs VMR_rigid_aorta_adults bifurcations_EL 5
  python -m util.zerod_calibration.run_cv_all_configs --only-barcharts   # skip CV, only run barcharts
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # ensure cwd is repo root for -m invocations

# Configs to run: (run_config_suffix, list of run_cross_validation flags)
# Order is the order of execution; barchart by-config will discover and sort by value.
# Use --run-config SUFFIX as the single way to specify config.
DEFAULT_CONFIGS = [
    ("stenosis_off_symmetric_gen_loss", ["--run-config", "stenosis_off_symmetric_gen_loss"]),
    ("stenosis_off_symmetric", ["--run-config", "stenosis_off_symmetric"]),
    ("symmetric_penalty_off_gen_loss", ["--run-config", "symmetric_penalty_off_gen_loss"]),
    ("stenosis_off_symmetric_gen_loss:bifurcations", ["--run-config", "stenosis_off_symmetric_gen_loss:bifurcations"]),
]

# Valid for e.g. `--configs penalty_off_gen_loss` but not part of the default batch (extra data + training).
OPTIONAL_CONFIGS = {
    "penalty_off_gen_loss": ["--run-config", "penalty_off_gen_loss"],
    "symmetric_gen_loss": ["--run-config", "symmetric_gen_loss"],
    "symmetric_penalty_off_gen_loss": ["--run-config", "symmetric_penalty_off_gen_loss"],
    "base": ["--run-config", "base"],
    "stenosis_off": ["--run-config", "stenosis_off"],
    "penalty_off": ["--run-config", "penalty_off"],
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
            raise ValueError(
                f"Invalid config entry {entry!r}. Use 'config' or 'config:geometry_variant'."
            )
        return raw, run_cfg, geom_var
    return raw, raw, default_geometry_variant


def main():
    parser = argparse.ArgumentParser(
        description="Run CV for several configs, then per-config and by-config barcharts."
    )
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
    parser.add_argument(
        "num_trials",
        nargs="?",
        type=int,
        default=5,
        help="Number of CV trials (default: 5)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        metavar="CONFIG",
        help="Config entries to run. Use 'config' or 'config:geometry_variant' (e.g. stenosis_off_symmetric_gen_loss:bifurcations).",
    )
    parser.add_argument(
        "--only-barcharts",
        action="store_true",
        help="Skip cross-validation; only run per-config and by-config barcharts (assumes CV already done).",
    )
    parser.add_argument(
        "--skip-per-config-barchart",
        action="store_true",
        help="Pass --skip-barchart to run_cross_validation (CV generates barcharts by default).",
    )
    parser.add_argument(
        "--data-root",
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
            entry_key, run_config_suffix, cfg_geometry_variant = _parse_config_entry(
                c, geometry_variant
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        if run_config_suffix not in config_map:
            print(
                f"Unknown config: {run_config_suffix}. Known: {list(config_map.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        configs_with_flags.append(
            (entry_key, run_config_suffix, cfg_geometry_variant, config_map[run_config_suffix])
        )

    cv_script = os.path.join(REPO_ROOT, "util", "zerod_calibration", "run_cross_validation.py")
    barchart_per_config = os.path.join(REPO_ROOT, "util", "visualizations", "cv_pressure_max_pct_error_barchart.py")
    barchart_by_config = os.path.join(REPO_ROOT, "util", "visualizations", "cv_max_pct_error_by_config_barchart.py")

    if not args.only_barcharts:
        for entry_key, run_config_suffix, cfg_geometry_variant, cv_flags in configs_with_flags:
            print(f"\n{'='*60}")
            print(
                f"Running cross-validation: config = {entry_key} "
                f"(run-config={run_config_suffix}, geometry={cfg_geometry_variant})"
            )
            print(f"{'='*60}")
            cmd = [
                sys.executable, "-m", "util.zerod_calibration.run_cross_validation",
                set_name, cfg_geometry_variant, str(num_trials),
                "--data-root", args.data_root,
                *cv_flags,
            ]
            if args.skip_per_config_barchart:
                cmd.append("--skip-barchart")
            ret = subprocess.run(cmd, cwd=REPO_ROOT)
            if ret.returncode != 0:
                print(
                    f"Cross-validation failed for config {entry_key} "
                    f"(exit {ret.returncode}). Stopping.",
                    file=sys.stderr,
                )
                sys.exit(ret.returncode)
    else:
        # Only barcharts: run per-config barchart for each config that has data
        if not args.skip_per_config_barchart:
            for entry_key, run_config_suffix, cfg_geometry_variant, _ in configs_with_flags:
                print(f"\nRunning per-config barchart for {entry_key}...")
                cmd_barchart = [
                    sys.executable, "-m", "util.visualizations.cv_pressure_max_pct_error_barchart",
                    set_name, cfg_geometry_variant,
                    "--run-config", run_config_suffix,
                    "--data-root", "results",
                ]
                subprocess.run(cmd_barchart, cwd=REPO_ROOT)

    print(f"\n{'='*60}")
    print("Running by-config comparison barchart...")
    print(f"{'='*60}")
    by_config_list = [c[0] for c in configs_with_flags]
    # Extra bar: stenosis_off config but using bifurcations (not bifurcations_EL) error CSV
    if "stenosis_off" in by_config_list:
        by_config_list.append("stenosis_off:bifurcations")
    cmd_by = [
        sys.executable, "-m", "util.visualizations.cv_max_pct_error_by_config_barchart",
        set_name,
        "--geometry", geometry_variant,
        "--configs", *by_config_list,
        "--data-root", "results",
        "--xmax", "40",
    ]
    ret_by = subprocess.run(cmd_by, cwd=REPO_ROOT)
    if ret_by.returncode != 0:
        print(f"By-config barchart failed (exit {ret_by.returncode}).", file=sys.stderr)
        sys.exit(ret_by.returncode)
    print("Done.")


if __name__ == "__main__":
    main()
