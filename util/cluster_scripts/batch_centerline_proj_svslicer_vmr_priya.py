#!/usr/bin/env python3
"""
Batch script to project 3D simulation results onto centerlines for VMR_priya_coa layout.
Uses the same Python slice-and-integrate method as batch_centerline_proj_svslicer.py
(svSlicer is not used).

File structure expected:
  base_dir/
    <geo>/                          e.g. 0241_H_AO_COA, 0225_H_AO_COA
      ROMSimulations/0d/centerlines.vtp
      Simulations/
        <sim_name>/                  e.g. pulsatile_defwall, LOOP_REST
          <num_procs>-procs/result_*.vtu

Output: output_dir/<geo>/<sim_name>/unsteady_soln.vtp

Local default base:  data/threeD/VMR_priya_coa  (relative to repo root)
Sherlock default:   /scratch/users/nrubio/VMR_priya_coa
"""

import os
import sys
import argparse

# Ensure we can import from sibling and repo
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
for _p in (_script_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from batch_centerline_proj_svslicer import find_simulation_files, project_results_python_fallback


def _default_base_dir():
    if os.environ.get("VMR_PRIYA_BASE_DIR"):
        return os.environ["VMR_PRIYA_BASE_DIR"]
    # Sherlock
    if os.path.exists("/scratch/users/nrubio/VMR_priya_coa"):
        return "/scratch/users/nrubio/VMR_priya_coa"
    # Local: repo root relative to this script
    local = os.path.join(_repo_root, "data", "threeD", "VMR_priya_coa")
    return local


def _default_output_dir(base_dir):
    if os.environ.get("VMR_PRIYA_OUTPUT_DIR"):
        return os.environ["VMR_PRIYA_OUTPUT_DIR"]
    # Same parent as base, with _reduced_results suffix
    parent = os.path.dirname(base_dir)
    name = os.path.basename(base_dir.rstrip(os.sep))
    return os.path.join(parent, f"{name}_reduced_results")


def main():
    parser = argparse.ArgumentParser(
        description="Project 3D results onto centerlines for VMR_priya_coa layout (Python method only)."
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Root directory containing geometry folders (default: VMR_PRIYA_BASE_DIR or auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for unsteady_soln.vtp outputs (default: <base_dir parent>/<base_name>_reduced_results).",
    )
    parser.add_argument(
        "--num-procs",
        default="48",
        help="Number of processors string for *-procs folder (default: 48).",
    )
    parser.add_argument(
        "--simulation",
        default=None,
        help="If set, only process this simulation name (e.g. pulsatile_defwall, LOOP_REST).",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Re-run even if unsteady_soln.vtp already exists.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir or _default_base_dir()
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        sys.exit(1)

    output_base_dir = args.output_dir or _default_output_dir(base_dir)
    output_base_dir = os.path.abspath(output_base_dir)

    print(f"Base directory:    {base_dir}")
    print(f"Output directory:  {output_base_dir}")
    print(f"Num procs:         {args.num_procs}")
    if args.simulation:
        print(f"Simulation filter: {args.simulation}")
    print()

    total_processed = 0
    total_skipped = 0
    total_failed = 0

    geo_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    geo_names.sort()

    for geo_name in geo_names:
        geo_dir = os.path.join(base_dir, geo_name)
        centerline_path = os.path.join(geo_dir, "ROMSimulations", "0d", "centerlines.vtp")
        if not os.path.isfile(centerline_path):
            print(f"  Skipping {geo_name}: no centerline at ROMSimulations/0d/centerlines.vtp")
            total_skipped += 1
            continue

        simulations_dir = os.path.join(geo_dir, "Simulations")
        if not os.path.isdir(simulations_dir):
            print(f"  Skipping {geo_name}: no Simulations/ directory")
            total_skipped += 1
            continue

        sim_names = [d for d in os.listdir(simulations_dir) if os.path.isdir(os.path.join(simulations_dir, d))]
        if args.simulation:
            if args.simulation not in sim_names:
                continue
            sim_names = [args.simulation]
        else:
            sim_names.sort()

        for sim_name in sim_names:
            sim_dir = os.path.join(simulations_dir, sim_name)
            time_files, _ = find_simulation_files(sim_dir, args.num_procs)
            if not time_files:
                print(f"  Skipping {geo_name}/{sim_name}: no result_*.vtu in {args.num_procs}-procs")
                total_skipped += 1
                continue

            output_path = os.path.join(output_base_dir, geo_name, sim_name, "unsteady_soln.vtp")
            if os.path.exists(output_path) and not args.redo:
                print(f"  Skipping {geo_name}/{sim_name}: output exists (use --redo to overwrite)")
                total_skipped += 1
                continue

            print(f"  Processing {geo_name}/{sim_name} ({len(time_files)} timesteps)...")
            if project_results_python_fallback(
                geo_dir, sim_dir, centerline_path, args.num_procs, output_path, time_files
            ):
                total_processed += 1
            else:
                total_failed += 1
            print()

    print("Summary:")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped:   {total_skipped}")
    print(f"  Failed:    {total_failed}")


if __name__ == "__main__":
    main()
