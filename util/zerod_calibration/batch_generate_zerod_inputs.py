#!/usr/bin/env python3
"""
Batch script to generate svZeroDSolver input files for all geometries in a set.
Loops through all geometries and creates:
1. Geometric 0D input files
2. Calibration input files
3. Calibrated output files (by running svZeroDCalibrator)
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path

def find_geometries(set_name, data_dir='data'):
    """
    Find all geometries in a set.
    
    Args:
        set_name: Set name (e.g., 'set_1')
        data_dir: Base data directory
        
    Returns:
        List of geometry names
    """
    threeD_dir = os.path.join(data_dir, 'threeD', set_name)
    if not os.path.exists(threeD_dir):
        return []
    
    # Find all geometry directories
    geometries = []
    for item in os.listdir(threeD_dir):
        item_path = os.path.join(threeD_dir, item)
        if os.path.isdir(item_path) and item.startswith('tree_'):
            geometries.append(item)
    
    geometries.sort()
    return geometries


def find_centerline_file(geo_dir):
    """Find centerline file in geometry directory."""
    centerline_paths = [
        os.path.join(geo_dir, 'centerlines_simVascular.vtp'),
        os.path.join(geo_dir, 'centerlines', 'centerlines.vtp'),
        os.path.join(geo_dir, 'centerlines.vtp'),
    ]
    
    for path in centerline_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_1d_solution(set_name, geo_name):
    """Find 1D centerline solution file."""
    # First check data/oneD/set_name/geo_name
    oneD_path = os.path.join('data', 'oneD', set_name, geo_name, 'unsteady_soln.vtp')
    if os.path.exists(oneD_path):
        return oneD_path
    
    # Try data/reduced_results
    local_path = os.path.join('data', 'reduced_results', set_name, geo_name, 'unsteady_soln.vtp')
    if os.path.exists(local_path):
        return local_path
    
    # Try scratch directory
    scratch_path = os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                               set_name, geo_name, 'unsteady_soln.vtp')
    if os.path.exists(scratch_path):
        return scratch_path
    
    return None


def process_geometry(set_name, geo_name, skip_calibration=False, output_dir='data/zeroD',
                     normalize=False, stenosis_off=False, symmetric_loss=False,
                     clip_predictions=False, penalty_off=False):
    """
    Process a single geometry to generate 0D input files.
    
    Args:
        set_name: Set name
        geo_name: Geometry name
        skip_calibration: Whether to skip calibration step
        output_dir: Output directory for 0D files
        normalize, stenosis_off, symmetric_loss, clip_predictions, penalty_off: run-config flags for path separation
    """
    print(f"\n{'='*80}")
    print(f"Processing: {set_name}/{geo_name}")
    print(f"{'='*80}")
    
    # Find centerline
    geo_dir = os.path.join('data', 'threeD', set_name, geo_name)
    centerline_path = find_centerline_file(geo_dir)
    
    if centerline_path is None:
        print(f"  ERROR: Centerline file not found for {geo_name}")
        return False
    
    # Find 1D solution
    soln_path = find_1d_solution(set_name, geo_name)
    
    # Build command
    script_path = os.path.join(os.path.dirname(__file__), 'generate_zerod_inputs.py')
    cmd = [sys.executable, script_path,
           '--set-name', set_name,
           '--geo-name', geo_name,
           '--centerline', centerline_path,
           '--output-dir', output_dir]
    
    if soln_path:
        cmd.extend(['--one-d-soln', soln_path])
    elif skip_calibration:
        cmd.append('--skip-calibration')
    else:
        print(f"  WARNING: No 1D solution found. Skipping calibration for {geo_name}")
        cmd.append('--skip-calibration')
    if normalize:
        cmd.append('--normalize')
    if stenosis_off:
        cmd.append('--stenosis-off')
    if penalty_off:
        cmd.append('--penalty-off')
    if symmetric_loss:
        cmd.append('--symmetric-loss')
    if clip_predictions:
        cmd.append('--clip-predictions')
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"  ✓ Successfully processed {geo_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to process {geo_name}")
        print(f"  Error: {e}")
        if e.stdout:
            print(f"  STDOUT: {e.stdout}")
        if e.stderr:
            print(f"  STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate svZeroDSolver input files for all geometries in a set"
    )
    parser.add_argument('set_name', help='Set name (e.g., set_1)')
    parser.add_argument('--geo-name', help='Process only this geometry (optional)')
    parser.add_argument('--skip-calibration', action='store_true',
                       help='Skip calibration step')
    parser.add_argument('--output-dir', default='data/zeroD',
                       help='Output directory for 0D files (default: data/zeroD)')
    parser.add_argument('--data-dir', default='data',
                       help='Base data directory (default: data)')
    parser.add_argument('--normalize', action='store_true', help='Use normalized paths (run-config)')
    parser.add_argument('--stenosis-off', action='store_true', dest='stenosis_off', help='Stenosis-off run-config')
    parser.add_argument('--penalty-off', action='store_true', dest='penalty_off', help='Zero L2 penalties on R and stenosis (incompatible with --stenosis-off)')
    parser.add_argument('--symmetric-loss', action='store_true', dest='symmetric_loss', help='Symmetric loss run-config (overestimate weight 1.0 for all models)')
    parser.add_argument('--clip-predictions', action='store_true', dest='clip_predictions', help='Clip predictions run-config')
    
    args = parser.parse_args()
    
    # Find geometries
    if args.geo_name:
        geometries = [args.geo_name]
    else:
        geometries = find_geometries(args.set_name, args.data_dir)
    
    if not geometries:
        print(f"No geometries found in {args.set_name}")
        return

    if getattr(args, 'stenosis_off', False) and getattr(args, 'penalty_off', False):
        parser.error("Cannot use both --stenosis-off and --penalty-off.")
    
    print(f"Found {len(geometries)} geometries in {args.set_name}")
    
    # Process each geometry
    success_count = 0
    fail_count = 0
    
    for geo_name in geometries:
        success = process_geometry(args.set_name, geo_name, 
                                  skip_calibration=args.skip_calibration,
                                  output_dir=args.output_dir,
                                  normalize=getattr(args, 'normalize', False),
                                  stenosis_off=getattr(args, 'stenosis_off', False),
                                  symmetric_loss=getattr(args, 'symmetric_loss', False),
                                  clip_predictions=getattr(args, 'clip_predictions', False),
                                  penalty_off=getattr(args, 'penalty_off', False))
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total geometries: {len(geometries)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

