#!/usr/bin/env python3
"""
Batch script to run generate_zerod_inputs.py for all VMR geometries.

This script:
1. Discovers all VMR geometries from data/zeroD/<set_name>/standard-0d/
2. Runs generate_zerod_inputs.py for each geometry
3. Tracks successes and failures
4. Provides a summary at the end
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.generate_zerod_inputs_cli import (
    DEFAULT_JUNCTION_TYPES,
    add_batch_arguments,
    add_generate_zerod_inputs_arguments,
    namespace_to_generate_zerod_argv,
    prepare_generate_zerod_namespace,
)
from learn_lpns.zerod_calibration.tools.file_io import get_vmr_geometries, standard_0d_dir

DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS = 1000


def load_previous_log(log_file_path):
    """Load a previous batch log file to extract failed/timed-out geometries."""
    if not os.path.exists(log_file_path):
        return None

    try:
        with open(log_file_path) as f:
            log_data = json.load(f)

        failed_geos = [item["geometry"] for item in log_data.get("failed", []) if not item.get("timed_out", False)]
        timed_out_geos = log_data.get("timed_out", [])

        return {
            "failed": failed_geos,
            "timed_out": timed_out_geos,
            "all_failed": [item["geometry"] for item in log_data.get("failed", [])],
        }
    except Exception as e:
        print(f"  Warning: Could not load log file {log_file_path}: {e}")
        return None


def check_geometry_complete(
    set_name,
    geo_name,
    junction_types,
    skip_forward=False,
    run_config_suffix=None,
    require_nn_outputs=True,
):
    """Check if a geometry already has required output files under the run-config path."""
    if run_config_suffix:
        base_dir = os.path.join(str(repo_root()), "data", "zeroD", set_name, run_config_suffix, geo_name)
    else:
        base_dir = os.path.join(str(repo_root()), "data", "zeroD", set_name, geo_name)

    for jtype in junction_types:
        calibrated_output = os.path.join(base_dir, f"bifurcations_calibrated_output_{jtype}.json")
        if not os.path.exists(calibrated_output):
            return False

        if not skip_forward:
            calibrated_results = os.path.join(base_dir, f"bifurcations_calibrated_results_{jtype}.csv")
            if not os.path.exists(calibrated_results):
                return False

    if require_nn_outputs and "BloodVesselJunction" in junction_types:
        from learn_lpns.zerod_calibration.modality_paths import NN_JUNCTION_ONLY_SUFFIX

        nn_output = os.path.join(base_dir, f"bifurcations_NN_{NN_JUNCTION_ONLY_SUFFIX}.json")
        if not os.path.exists(nn_output):
            return False

        if not skip_forward:
            nn_results = os.path.join(base_dir, f"bifurcations_NN_{NN_JUNCTION_ONLY_SUFFIX}_results.csv")
            if not os.path.exists(nn_results):
                return False

    return True


def run_generate_zerod_inputs(set_name, geo_name, args, verbose=False, timeout_seconds=None):
    """Run generate_zerod_inputs.py for a single geometry."""
    if timeout_seconds is None:
        timeout_seconds = DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS

    cmd = namespace_to_generate_zerod_argv(args, set_name=set_name, geo_name=geo_name)

    try:
        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=repo_root(), text=True, timeout=timeout_seconds)

        if result.returncode == 0:
            generated_files = None
            if args.no_redo:
                try:
                    output_lines = result.stdout.split("\n") if result.stdout else []
                    for line in reversed(output_lines):
                        if line.strip().startswith('{"generated_files"'):
                            generated_files = json.loads(line.strip()).get("generated_files", [])
                            break
                except Exception:
                    pass
            return True, None, False, generated_files

        error_msg = result.stderr if result.stderr else "Command failed (see output above)"
        return False, error_msg, False, None

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout_seconds} seconds", True, None
    except Exception as e:
        return False, str(e), False, None


def main():
    parser = argparse.ArgumentParser(
        description="Batch run generate_zerod_inputs.py for all VMR geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learn_lpns/zerod_calibration/batch_generate_zerod_inputs_vmr.py \\
      --set_name VMR_rigid_aorta_adults

  python learn_lpns/zerod_calibration/batch_generate_zerod_inputs_vmr.py \\
      --set_name VMR_rigid_aorta_adults \\
      --run_config gen_loss_quadratic_resistor \\
      --skip_steps calibration_forward

  python learn_lpns/zerod_calibration/batch_generate_zerod_inputs_vmr.py \\
      --geometries 0063_1001 0155_0001 --no_redo
        """,
    )
    add_generate_zerod_inputs_arguments(parser, require_set_geo=False)
    add_batch_arguments(parser)
    args = parser.parse_args()

    set_name = args.set_name
    run_config_suffix = prepare_generate_zerod_namespace(args)

    if args.geometries:
        geo_names = args.geometries
        print(f"Processing {len(geo_names)} specified geometries")
    else:
        print("Discovering VMR geometries...")
        geo_names = get_vmr_geometries(standard_0d_dir(os.path.join(str(repo_root()), "data"), set_name))
        print(f"Found {len(geo_names)} VMR geometries")

    if args.only_failed or args.only_timed_out or args.only_successful:
        if not args.log_file:
            parser.error("--only_failed, --only_timed_out, and --only_successful require --log_file")

        log_path = os.path.join(str(repo_root()), args.log_file)
        previous_log = load_previous_log(log_path)

        if previous_log is None:
            parser.error(f"Could not load log file: {log_path}")

        if args.only_failed:
            target_geos = previous_log["failed"]
            print(f"\nFiltering to {len(target_geos)} failed geometries from log file")
        elif args.only_timed_out:
            target_geos = previous_log["timed_out"]
            print(f"\nFiltering to {len(target_geos)} timed-out geometries from log file")
        else:
            with open(log_path) as f:
                log_data = json.load(f)
            target_geos = log_data.get("success", [])
            print(f"\nFiltering to {len(target_geos)} successful geometries from log file")

        geo_names = [geo for geo in geo_names if geo in target_geos]

        if len(geo_names) == 0:
            print("No matching geometries found in log file. Exiting.")
            return

        print(f"Will process {len(geo_names)} geometries")

    if len(geo_names) == 0:
        print("No geometries to process. Exiting.")
        return

    print("\n" + "=" * 70)
    print("Batch Configuration")
    print("=" * 70)
    print(f"  Set name: {set_name}")
    print(f"  Number of geometries: {len(geo_names)}")
    print(f"  Run config: {run_config_suffix}")
    print(f"  Skip steps: {args.skip_steps or '(none)'}")
    print(f"  No-redo mode: {args.no_redo}")
    print(f"  NN-only mode: {args.NN_only}")
    print(f"  Vessel_NN mode: {args.Vessel_NN}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Timeout per geometry: {args.timeout}s ({args.timeout / 60:.1f} minutes)")
    print(f"  Max failures: {args.max_failures if args.max_failures else 'unlimited'}")
    print("=" * 70 + "\n")

    results = {
        "success": [],
        "failed": [],
        "skipped": [],
        "timed_out": [],
        "start_time": datetime.now().isoformat(),
    }

    for i, geo_name in enumerate(geo_names, 1):
        print(f"\n[{i}/{len(geo_names)}] Processing geometry: {geo_name}")
        print("-" * 70)

        if args.skip_existing:
            if check_geometry_complete(
                set_name,
                geo_name,
                DEFAULT_JUNCTION_TYPES,
                skip_forward=args.skip_forward,
                run_config_suffix=run_config_suffix,
            ):
                print(f"  Skipping {geo_name} (already has necessary output files)")
                results["skipped"].append(geo_name)
                continue

        success, error_msg, timed_out, generated_files = run_generate_zerod_inputs(
            set_name,
            geo_name,
            args,
            verbose=args.verbose,
            timeout_seconds=args.timeout,
        )

        if success:
            print(f"  Successfully processed {geo_name}")
            results["success"].append(geo_name)
            if generated_files is not None:
                if "geometry_files" not in results:
                    results["geometry_files"] = {}
                results["geometry_files"][geo_name] = generated_files
        else:
            if timed_out:
                print(f"  Timed out processing {geo_name} (exceeded {args.timeout}s)")
                results["timed_out"].append(geo_name)
                results["failed"].append({"geometry": geo_name, "error": error_msg, "timed_out": True})
            else:
                print(f"  Failed to process {geo_name}")
                if error_msg:
                    error_lines = error_msg.strip().split("\n")
                    for line in error_lines[-50:]:
                        print(f"      {line}")
                results["failed"].append(
                    {
                        "geometry": geo_name,
                        "error": error_msg[:1000] if error_msg else "Unknown error",
                        "timed_out": False,
                    }
                )

            if args.max_failures and len(results["failed"]) >= args.max_failures:
                print(f"\n  Stopping after {args.max_failures} failures (as requested)")
                break

    results["end_time"] = datetime.now().isoformat()
    results["total"] = len(geo_names)
    results["success_count"] = len(results["success"])
    results["failed_count"] = len(results["failed"])
    results["skipped_count"] = len(results["skipped"])
    results["timed_out_count"] = len(results["timed_out"])

    print("\n" + "=" * 70)
    print("Batch Processing Summary")
    print("=" * 70)
    print(f"  Total geometries: {results['total']}")
    print(f"  Successful: {results['success_count']}")
    print(f"  Failed: {results['failed_count']}")
    print(f"  Skipped (existing): {results['skipped_count']}")
    print(f"  Timed out: {results['timed_out_count']}")

    if args.log_file:
        log_path = os.path.join(str(repo_root()), args.log_file)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results logged to: {log_path}")

    sys.exit(1 if results["failed_count"] > 0 else 0)


if __name__ == "__main__":
    main()
