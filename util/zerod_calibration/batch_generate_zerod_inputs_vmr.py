#!/usr/bin/env python3
"""
Batch script to run generate_zerod_inputs.py for all VMR geometries.

This script:
1. Discovers all VMR geometries from data/zeroD/VMR/richter-0d/
2. Runs generate_zerod_inputs.py for each geometry
3. Tracks successes and failures
4. Provides a summary at the end
"""

import glob
import json
import os
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from util.zerod_calibration.process_arguments import (
    DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS,
    append_generate_zerod_flags_to_cmd,
    apply_default_run_config_if_unspecified,
    apply_log_file_geometry_filter,
    build_vmr_batch_parser,
    discover_geometry_names,
    namespace_to_subprocess_args_dict,
    print_batch_configuration,
    resolve_vmr_run_config_args,
    validate_vmr_batch_args,
    zero_d_run_config_subdir,
)


def get_vmr_geometries(richter_dir="data/zeroD/VMR/richter-0d"):
    """
    Get list of valid VMR geometry names from richter-0d directory.

    Args:
        richter_dir: Path to richter-0d directory containing JSON files

    Returns:
        List of geometry names (without .json extension)
    """
    richter_path = os.path.join(REPO_ROOT, richter_dir)
    if not os.path.exists(richter_path):
        raise FileNotFoundError(f"Richter-0d directory not found: {richter_path}")

    json_files = sorted(glob.glob(os.path.join(richter_path, "*.json")))
    geo_names = []

    for json_file in json_files:
        geo_name = os.path.basename(json_file).replace(".json", "")
        try:
            with open(json_file, "r") as f:
                json.load(f)
            geo_names.append(geo_name)
        except Exception as e:
            print(f"  Warning: Skipping invalid JSON file {geo_name}: {e}")

    return geo_names


def has_coronary_bc(set_name, geo_name):
    """
    Check if a geometry has CORONARY type boundary conditions.

    Args:
        geo_name: Geometry name (e.g., '0063_1001')

    Returns:
        bool: True if geometry has CORONARY boundary conditions, False otherwise
    """
    richter_path = os.path.join(REPO_ROOT, "data", "zeroD", set_name, "richter-0d", f"{geo_name}.json")

    if os.path.exists(richter_path):
        try:
            with open(richter_path, "r") as f:
                data = json.load(f)

            for bc in data.get("boundary_conditions", []):
                if bc.get("bc_type") == "CORONARY":
                    return True
        except Exception:
            pass

    geometric_input_path = os.path.join(REPO_ROOT, "data", "zeroD", set_name, geo_name, "geometric_input.json")
    if os.path.exists(geometric_input_path):
        try:
            with open(geometric_input_path, "r") as f:
                data = json.load(f)

            for bc in data.get("boundary_conditions", []):
                if bc.get("bc_type") == "CORONARY":
                    return True
        except Exception:
            pass

    return False


def check_geometry_complete(
    set_name,
    geo_name,
    junction_types,
    skip_forward=False,
    run_config_subdir="base",
):
    """
    Check if a geometry already has all necessary output files.

    Args:
        geo_name: Geometry name (e.g., '0063_1001')
        junction_types: List of junction types to check for
        skip_forward: Whether forward simulation was skipped
        run_config_subdir: Subfolder under data/zeroD/<set>/ (matches generate_zerod_inputs)

    Returns:
        bool: True if geometry appears complete, False otherwise
    """
    base_dir = os.path.join(REPO_ROOT, "data", "zeroD", set_name, run_config_subdir, geo_name)

    for jtype in junction_types:
        calibrated_output = os.path.join(base_dir, f"bifurcations_calibrated_output_{jtype}.json")
        if not os.path.exists(calibrated_output):
            return False

        if not skip_forward:
            calibrated_results = os.path.join(base_dir, f"bifurcations_calibrated_results_{jtype}.csv")
            if not os.path.exists(calibrated_results):
                return False

    if "BloodVesselJunction" in junction_types:
        nn_output = os.path.join(base_dir, "bifurcations_NN_BloodVesselJunction.json")
        if not os.path.exists(nn_output):
            return False

        if not skip_forward:
            nn_results = os.path.join(base_dir, "bifurcations_NN_BloodVesselJunction_results.csv")
            if not os.path.exists(nn_results):
                return False

    return True


def run_generate_zerod_inputs(set_name, geo_name, args_dict, verbose=False, timeout_seconds=None):
    """
    Run generate_zerod_inputs.py for a single geometry.

    Args:
        geo_name: Geometry name (e.g., '0063_1001')
        args_dict: Dictionary of arguments to pass to generate_zerod_inputs.py
        verbose: Whether to print verbose output
        timeout_seconds: Maximum time to wait (default: DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS)

    Returns:
        (success: bool, error_message: str or None, timed_out: bool, generated_files: list or None)
    """
    if timeout_seconds is None:
        timeout_seconds = DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS

    script_path = os.path.join(REPO_ROOT, "util", "zerod_calibration", "generate_zerod_inputs.py")

    cmd = [sys.executable, script_path, "--set-name", set_name, "--geo-name", geo_name]
    append_generate_zerod_flags_to_cmd(cmd, args_dict)

    try:
        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            generated_files = None
            if args_dict.get("no_redo", False):
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


def print_diagnostics(results, geo_names, log_file_relpath=None, repo_root=None):
    """
    Finalize result counts, print batch summary, optionally write JSON log.
    """
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
    print(f"  Start time: {results['start_time']}")
    print(f"  End time: {results['end_time']}")

    if results["skipped"]:
        print("\n  Skipped geometries (already complete):")
        for geo in results["skipped"]:
            print(f"    - {geo}")

    if results["timed_out"]:
        print("\n  Timed out geometries:")
        for geo in results["timed_out"]:
            print(f"    - {geo}")

    if results["failed"]:
        print("\n  Failed geometries:")
        for failure in results["failed"]:
            if failure.get("timed_out"):
                print(f"    - {failure['geometry']} (timed out)")
            else:
                print(f"    - {failure['geometry']}")

    print("=" * 70)

    if log_file_relpath and repo_root is not None:
        log_path = os.path.join(repo_root, log_file_relpath)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results logged to: {log_path}")


def main():
    parser = build_vmr_batch_parser()
    args = parser.parse_args()
    apply_default_run_config_if_unspecified(args)
    resolve_vmr_run_config_args(args, parser)
    validate_vmr_batch_args(args, parser)

    set_name = args.set_name
    run_cfg_subdir = zero_d_run_config_subdir(args)
    args_dict = namespace_to_subprocess_args_dict(args)

    geo_names = discover_geometry_names(args, set_name, REPO_ROOT, get_vmr_geometries)
    geo_names = apply_log_file_geometry_filter(args, parser, REPO_ROOT, geo_names)

    if len(geo_names) == 0:
        if not (args.only_failed or args.only_timed_out or args.only_successful):
            print("No geometries to process. Exiting.")
        return

    print_batch_configuration(args, set_name, geo_names)

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

        # Coronary boundary conditions are not supported yet
        if has_coronary_bc(set_name, geo_name):
            print(f"  ⊘ Skipping {geo_name} (has CORONARY boundary conditions)")
            results["skipped"].append(geo_name)
            continue

        if args.skip_existing:
            if check_geometry_complete(
                set_name,
                geo_name,
                args.junction_types,
                skip_forward=args.skip_forward,
                run_config_subdir=run_cfg_subdir,
            ):
                print(f"  ⊘ Skipping {geo_name} (already has necessary output files)")
                results["skipped"].append(geo_name)
                continue

        success, error_msg, timed_out, generated_files = run_generate_zerod_inputs(
            set_name,
            geo_name,
            args_dict,
            verbose=args.verbose,
            timeout_seconds=args.timeout,
        )

        if success:
            print(f"  ✓ Successfully processed {geo_name}")
            results["success"].append(geo_name)
            if generated_files is not None:
                if "geometry_files" not in results:
                    results["geometry_files"] = {}
                results["geometry_files"][geo_name] = generated_files
        else:
            if timed_out:
                print(f"  ⏱ Timed out processing {geo_name} (exceeded {args.timeout}s)")
                results["timed_out"].append(geo_name)
                results["failed"].append(
                    {
                        "geometry": geo_name,
                        "error": error_msg,
                        "timed_out": True,
                    }
                )
            else:
                print(f"  ✗ Failed to process {geo_name}")
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

    print_diagnostics(results, geo_names, log_file_relpath=args.log_file, repo_root=REPO_ROOT)

    if results["failed_count"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
