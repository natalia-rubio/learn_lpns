#!/usr/bin/env python3
"""
Batch script to run generate_zerod_inputs.py for all VMR geometries.

This script:
1. Discovers all VMR geometries from data/zeroD/VMR/richter-0d/
2. Runs generate_zerod_inputs.py for each geometry
3. Tracks successes and failures
4. Provides a summary at the end
"""

import os
import sys
import subprocess
import json
import glob
import argparse
from datetime import datetime

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Timeout in seconds for each generate_zerod_inputs.py run. Increase for slow/large geometries.
DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS = 1000


def get_vmr_geometries(richter_dir='data/zeroD/VMR/richter-0d'):
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
    
    json_files = sorted(glob.glob(os.path.join(richter_path, '*.json')))
    geo_names = []
    
    for json_file in json_files:
        geo_name = os.path.basename(json_file).replace('.json', '')
        # Validate JSON file
        try:
            with open(json_file, 'r') as f:
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
    # Check the richter-0d source file first
    richter_path = os.path.join(REPO_ROOT, 'data', 'zeroD', set_name, 'richter-0d', f'{geo_name}.json')
    
    if os.path.exists(richter_path):
        try:
            with open(richter_path, 'r') as f:
                data = json.load(f)
            
            # Check boundary conditions
            for bc in data.get('boundary_conditions', []):
                if bc.get('bc_type') == 'CORONARY':
                    return True
        except Exception:
            # If we can't read the file, try the geometric input
            pass
    
    # Fallback: check geometric_input.json if it exists
    geometric_input_path = os.path.join(REPO_ROOT, 'data', 'zeroD', set_name, geo_name, 'geometric_input.json')
    if os.path.exists(geometric_input_path):
        try:
            with open(geometric_input_path, 'r') as f:
                data = json.load(f)
            
            # Check boundary conditions
            for bc in data.get('boundary_conditions', []):
                if bc.get('bc_type') == 'CORONARY':
                    return True
        except Exception:
            pass
    
    return False

def load_previous_log(log_file_path):
    """
    Load a previous batch log file to extract failed/timed-out geometries.
    
    Args:
        log_file_path: Path to the log JSON file
        
    Returns:
        dict with 'failed' and 'timed_out' lists, or None if file doesn't exist
    """
    if not os.path.exists(log_file_path):
        return None
    
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        
        # Extract failed geometries (excluding timed-out ones)
        failed_geos = [
            item['geometry'] 
            for item in log_data.get('failed', [])
            if not item.get('timed_out', False)
        ]
        
        # Extract timed-out geometries
        timed_out_geos = log_data.get('timed_out', [])
        
        return {
            'failed': failed_geos,
            'timed_out': timed_out_geos,
            'all_failed': [item['geometry'] for item in log_data.get('failed', [])]
        }
    except Exception as e:
        print(f"  Warning: Could not load log file {log_file_path}: {e}")
        return None

def check_geometry_complete(set_name, geo_name, junction_types, skip_forward=False):
    """
    Check if a geometry already has all necessary output files.
    
    Args:
        geo_name: Geometry name (e.g., '0063_1001')
        junction_types: List of junction types to check for
        skip_forward: Whether forward simulation was skipped
        
    Returns:
        bool: True if geometry appears complete, False otherwise
    """
    base_dir = os.path.join(REPO_ROOT, 'data', 'zeroD', set_name, geo_name)
    
    # Check for bifurcations calibrated outputs for each junction type
    for jtype in junction_types:
        calibrated_output = os.path.join(
            base_dir, 
            f'bifurcations_calibrated_output_{jtype}.json'
        )
        if not os.path.exists(calibrated_output):
            return False
        
        # If forward simulation was not skipped, also check for results
        if not skip_forward:
            calibrated_results = os.path.join(
                base_dir,
                f'bifurcations_calibrated_results_{jtype}.csv'
            )
            if not os.path.exists(calibrated_results):
                return False
    
    # Check for NN output if BloodVesselJunction is in junction types
    if 'BloodVesselJunction' in junction_types:
        nn_output = os.path.join(
            base_dir,
            'bifurcations_NN_BloodVesselJunction.json'
        )
        if not os.path.exists(nn_output):
            return False
        
        if not skip_forward:
            nn_results = os.path.join(
                base_dir,
                'bifurcations_NN_BloodVesselJunction_results.csv'
            )
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

    script_path = os.path.join(REPO_ROOT, 'util', 'zerod_calibration', 'generate_zerod_inputs.py')

    # Build command
    cmd = [sys.executable, script_path, '--set-name', set_name, '--geo-name', geo_name]
    
    # Add optional arguments
    if args_dict.get('junction_types'):
        cmd.extend(['--junction-types', ','.join(args_dict['junction_types'])])
    if args_dict.get('zoom_start') is not None:
        cmd.extend(['--zoom-start', str(args_dict['zoom_start'])])
    if args_dict.get('zoom_end') is not None:
        cmd.extend(['--zoom-end', str(args_dict['zoom_end'])])
    if args_dict.get('verbose', False):
        cmd.append('--verbose')
    if args_dict.get('skip_base_generation', False):
        cmd.append('--skip-base-generation')
    if args_dict.get('skip_observation', False):
        cmd.append('--skip-observation')
    if args_dict.get('skip_calibration', False):
        cmd.append('--skip-calibration')
    if args_dict.get('skip_forward', False):
        cmd.append('--skip-forward')
    if args_dict.get('skip_mse_calculation', False):
        cmd.append('--skip-mse-calculation')
    if args_dict.get('skip_plots', False):
        cmd.append('--skip-plots')
    if args_dict.get('NN_only', False):
        cmd.append('--NN-only')
    if args_dict.get('NN_vessel', False):
        cmd.append('--NN-vessel')
    if args_dict.get('no_redo', False):
        cmd.append('--no-redo')
    if args_dict.get('normalize', False):
        cmd.append('--normalize')
    if args_dict.get('stenosis_off', False):
        cmd.append('--stenosis-off')
    if args_dict.get('penalty_off', False):
        cmd.append('--penalty-off')
    if args_dict.get('symmetric_loss', False):
        cmd.append('--symmetric-loss')
    if args_dict.get('clip_predictions', False):
        cmd.append('--clip-predictions')
    # Run command
    try:
        if verbose:
            print(f"  Running: {' '.join(cmd)}")
        
        # Run with output streaming to show progress in real-time
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            timeout=timeout_seconds
        )
        
        if result.returncode == 0:
            # Try to extract generated files from output if --no-redo was used
            generated_files = None
            if args_dict.get('no_redo', False):
                # Look for JSON output in stdout that contains generated_files
                # The script should print this as JSON at the end
                try:
                    import json
                    output_lines = result.stdout.split('\n') if result.stdout else []
                    for line in reversed(output_lines):
                        if line.strip().startswith('{"generated_files"'):
                            generated_files = json.loads(line.strip()).get('generated_files', [])
                            break
                except Exception:
                    pass
            return True, None, False, generated_files
        else:
            # Collect error output if available
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
  # Run full workflow for all geometries
  python3 batch_generate_zerod_inputs_vmr.py

  # Skip calibration and forward simulation (faster)
  python3 batch_generate_zerod_inputs_vmr.py --skip-calibration --skip-forward

  # Only process specific geometries
  python3 batch_generate_zerod_inputs_vmr.py --geometries 0063_1001 0155_0001

  # Rerun only failed geometries from a previous run
  python3 batch_generate_zerod_inputs_vmr.py --only-failed --log-file results/vmr_batch_log.json

  # Rerun only timed-out geometries from a previous run
  python3 batch_generate_zerod_inputs_vmr.py --only-timed-out --log-file results/vmr_batch_log.json

  # Rerun only successful geometries from a previous run
  python3 batch_generate_zerod_inputs_vmr.py --only-successful --log-file results/vmr_batch_log.json

  # Verbose output
  python3 batch_generate_zerod_inputs_vmr.py --verbose

  # Also run vessel NN inference (junction + vessel params)
  python3 batch_generate_zerod_inputs_vmr.py --NN-vessel
  python3 batch_generate_zerod_inputs_vmr.py --normalize
        """
    )
    parser.add_argument('--set-name', default='VMR', help='Set name (e.g., set_1)')
    parser.add_argument('--geometries', nargs='+', default=None,
                       help='Specific geometry names to process (default: all)')
    parser.add_argument('--junction-types', type=lambda s: [x.strip() for x in s.split(',') if x.strip()],
                       default='NORMAL_JUNCTION,BloodVesselJunction',
                       help='Comma-separated junction types to process (default: NORMAL_JUNCTION,BloodVesselJunction)')
    parser.add_argument('--zoom-start', type=int, default=None,
                       help='Start index for zoom window (default: 599)')
    parser.add_argument('--zoom-end', type=int, default=None,
                       help='End index for zoom window (default: 699)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output for each geometry')
    parser.add_argument('--skip-base-generation', action='store_true',
                       help='Skip generating base geometric input')
    parser.add_argument('--skip-observation', action='store_true',
                       help='Skip observation extraction step')
    parser.add_argument('--skip-calibration', action='store_true',
                       help='Skip calibration step')
    parser.add_argument('--skip-forward', action='store_true',
                       help='Skip forward simulation step')
    parser.add_argument('--skip-mse-calculation', action='store_true',
                       help='Skip MSE calculation step')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating comparison plots')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip geometries that already have necessary output files')
    parser.add_argument('--no-redo', action='store_true',
                       help='Skip recreating files if they already exist (check at each step)')
    parser.add_argument('--NN-only', action='store_true',
                       help='Only run NN inference and forward simulation on NN inputs (skip calibration)')
    parser.add_argument('--NN-vessel', action='store_true', dest='NN_vessel',
                       help='Also run vessel NN inference and forward sim (write *_NN_JunctionAndVessel.json/results)')
    parser.add_argument('--stenosis-off', action='store_true', dest='stenosis_off',
                       help='Turn off stenosis: calibrate_stenosis_coefficient=False, set all stenosis to 0, do not use NN to predict stenosis')
    parser.add_argument('--penalty-off', action='store_true', dest='penalty_off',
                       help='Zero L2 penalties on R and stenosis when stenosis is included (incompatible with --stenosis-off)')
    parser.add_argument('--normalize', action='store_true',
                       help='Use normalized NN models and unnormalize predictions (pass --normalize to generate_zerod_inputs)')
    parser.add_argument('--symmetric-loss', action='store_true', dest='symmetric_loss',
                       help='Symmetric loss run-config: overestimate weight 1.0 for all models (for path naming)')
    parser.add_argument('--clip-predictions', action='store_true', dest='clip_predictions',
                       help='Clip R/S/L to training set min/max (run-config)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS,
                       help=f'Timeout in seconds for each geometry (default: {DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS})')
    parser.add_argument('--max-failures', type=int, default=None,
                       help='Stop after N failures (default: continue all)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file to write results (default: no log file)')
    
    # Filtering options (mutually exclusive)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--only-failed', action='store_true',
                             help='Only run geometries that failed in a previous run (requires --log-file)')
    filter_group.add_argument('--only-timed-out', action='store_true',
                             help='Only run geometries that timed out in a previous run (requires --log-file)')
    filter_group.add_argument('--only-successful', action='store_true',
                             help='Only run geometries that succeeded in a previous run (requires --log-file)')
    # Note: --all is the default behavior, so we don't need to add it as an option
    
    args = parser.parse_args()
    set_name = args.set_name
    # Build args dictionary
    args_dict = {
        'junction_types': args.junction_types,
        'zoom_start': args.zoom_start,
        'zoom_end': args.zoom_end,
        'verbose': args.verbose,
        'skip_base_generation': args.skip_base_generation,
        'skip_observation': args.skip_observation,
        'skip_calibration': args.skip_calibration,
        'skip_forward': args.skip_forward,
        'skip_mse_calculation': args.skip_mse_calculation,
        'skip_plots': args.skip_plots,
        'NN_only': args.NN_only,
        'NN_vessel': args.NN_vessel,
        'no_redo': args.no_redo,
        'normalize': getattr(args, 'normalize', False),
        'stenosis_off': getattr(args, 'stenosis_off', False),
        'penalty_off': getattr(args, 'penalty_off', False),
        'symmetric_loss': getattr(args, 'symmetric_loss', False),
        'clip_predictions': getattr(args, 'clip_predictions', False),
    }
    
    if getattr(args, 'stenosis_off', False) and getattr(args, 'penalty_off', False):
        parser.error("Cannot use both --stenosis-off and --penalty-off.")
    # Get list of geometries to process
    if args.geometries:
        geo_names = args.geometries
        print(f"Processing {len(geo_names)} specified geometries")
    else:
        print("Discovering VMR geometries...")
        geo_names = get_vmr_geometries(richter_dir=os.path.join(REPO_ROOT, 'data', 'zeroD', set_name, 'richter-0d'))
        print(f"Found {len(geo_names)} VMR geometries")
    
    # Filter geometries based on --only-failed, --only-timed-out, or --only-successful flags
    if args.only_failed or args.only_timed_out or args.only_successful:
        if not args.log_file:
            parser.error("--only-failed, --only-timed-out, and --only-successful require --log-file to be specified")
        
        log_path = os.path.join(REPO_ROOT, args.log_file)
        previous_log = load_previous_log(log_path)
        
        if previous_log is None:
            parser.error(f"Could not load log file: {log_path}")
        
        if args.only_failed:
            # Get failed geometries (excluding timed-out ones)
            target_geos = previous_log['failed']
            print(f"\nFiltering to {len(target_geos)} failed geometries from log file")
        elif args.only_timed_out:
            # Get timed-out geometries
            target_geos = previous_log['timed_out']
            print(f"\nFiltering to {len(target_geos)} timed-out geometries from log file")
        elif args.only_successful:
            # Get successful geometries from the log file
            log_path_full = os.path.join(REPO_ROOT, args.log_file)
            try:
                with open(log_path_full, 'r') as f:
                    log_data = json.load(f)
                target_geos = log_data.get('success', [])
                print(f"\nFiltering to {len(target_geos)} successful geometries from log file")
            except Exception as e:
                parser.error(f"Could not load successful geometries from log file: {e}")
        
        # Filter geo_names to only include target geometries
        geo_names = [geo for geo in geo_names if geo in target_geos]
        
        if len(geo_names) == 0:
            print("No matching geometries found in log file. Exiting.")
            return
        
        print(f"Will process {len(geo_names)} geometries")
    
    if len(geo_names) == 0:
        print("No geometries to process. Exiting.")
        return
    
    # Print configuration
    print("\n" + "="*70)
    print("Batch Configuration")
    print("="*70)
    print(f"  Set name: {set_name}")
    print(f"  Number of geometries: {len(geo_names)}")
    print(f"  Junction types: {', '.join(args.junction_types)}")
    print(f"  Zoom window: [{args.zoom_start}, {args.zoom_end}]")
    print(f"  Skip base generation: {args.skip_base_generation}")
    print(f"  Skip observation: {args.skip_observation}")
    print(f"  Skip calibration: {args.skip_calibration}")
    print(f"  Skip forward: {args.skip_forward}")
    print(f"  Skip MSE calculation: {args.skip_mse_calculation}")
    print(f"  Skip plots: {args.skip_plots}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  No-redo mode: {args.no_redo}")
    print(f"  NN-only mode: {args.NN_only}")
    print(f"  NN-vessel mode: {args.NN_vessel}")
    print(f"  Normalize: {getattr(args, 'normalize', False)}")
    print(f"  Stenosis-off: {getattr(args, 'stenosis_off', False)}")
    print(f"  Symmetric-loss: {getattr(args, 'symmetric_loss', False)}")
    print(f"  Clip-predictions: {getattr(args, 'clip_predictions', False)}")
    print(f"  Timeout per geometry: {args.timeout}s ({args.timeout/60:.1f} minutes)")
    print(f"  Max failures: {args.max_failures if args.max_failures else 'unlimited'}")
    if args.only_failed:
        print(f"  Filter mode: Only failed geometries")
    elif args.only_timed_out:
        print(f"  Filter mode: Only timed-out geometries")
    elif args.only_successful:
        print(f"  Filter mode: Only successful geometries")
    else:
        print(f"  Filter mode: All geometries (default)")
    print("="*70 + "\n")
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'timed_out': [],
        'start_time': datetime.now().isoformat(),
    }
    
    # Process each geometry
    for i, geo_name in enumerate(geo_names, 1):
        print(f"\n[{i}/{len(geo_names)}] Processing geometry: {geo_name}")
        print("-" * 70)
        
        # Check if geometry has CORONARY boundary conditions
        if has_coronary_bc(set_name, geo_name):
            print(f"  ⊘ Skipping {geo_name} (has CORONARY boundary conditions)")
            results['skipped'].append(geo_name)
            continue
        
        # Check if geometry already exists
        if args.skip_existing:
            if check_geometry_complete(
                set_name,
                geo_name, 
                args.junction_types,
                skip_forward=args.skip_forward
            ):
                print(f"  ⊘ Skipping {geo_name} (already has necessary output files)")
                results['skipped'].append(geo_name)
                continue
        
        success, error_msg, timed_out, generated_files = run_generate_zerod_inputs(
            set_name,
            geo_name, 
            args_dict, 
            verbose=args.verbose,
            timeout_seconds=args.timeout
        )
        
        if success:
            print(f"  ✓ Successfully processed {geo_name}")
            results['success'].append(geo_name)
            # Store generated files info for this geometry
            if generated_files is not None:
                if 'geometry_files' not in results:
                    results['geometry_files'] = {}
                results['geometry_files'][geo_name] = generated_files
        else:
            if timed_out:
                print(f"  ⏱ Timed out processing {geo_name} (exceeded {args.timeout}s)")
                results['timed_out'].append(geo_name)
                results['failed'].append({
                    'geometry': geo_name,
                    'error': error_msg,
                    'timed_out': True
                })
            else:
                print(f"  ✗ Failed to process {geo_name}")
                if error_msg:
                    # Print last few lines of error
                    error_lines = error_msg.strip().split('\n')
                    for line in error_lines[-50:]:
                        print(f"      {line}")
                results['failed'].append({
                    'geometry': geo_name,
                    'error': error_msg[:1000] if error_msg else 'Unknown error',  # Truncate long errors
                    'timed_out': False
                })
            
            # Check if we should stop
            if args.max_failures and len(results['failed']) >= args.max_failures:
                print(f"\n  Stopping after {args.max_failures} failures (as requested)")
                break
    
    # Print summary
    results['end_time'] = datetime.now().isoformat()
    results['total'] = len(geo_names)
    results['success_count'] = len(results['success'])
    results['failed_count'] = len(results['failed'])
    results['skipped_count'] = len(results['skipped'])
    results['timed_out_count'] = len(results['timed_out'])
    
    print("\n" + "="*70)
    print("Batch Processing Summary")
    print("="*70)
    print(f"  Total geometries: {results['total']}")
    print(f"  Successful: {results['success_count']}")
    print(f"  Failed: {results['failed_count']}")
    print(f"  Skipped (existing): {results['skipped_count']}")
    print(f"  Timed out: {results['timed_out_count']}")
    print(f"  Start time: {results['start_time']}")
    print(f"  End time: {results['end_time']}")
    
    if results['skipped']:
        print("\n  Skipped geometries (already complete):")
        for geo in results['skipped']:
            print(f"    - {geo}")
    
    if results['timed_out']:
        print("\n  Timed out geometries:")
        for geo in results['timed_out']:
            print(f"    - {geo}")
    
    if results['failed']:
        print("\n  Failed geometries:")
        for failure in results['failed']:
            if failure.get('timed_out'):
                print(f"    - {failure['geometry']} (timed out)")
            else:
                print(f"    - {failure['geometry']}")
    
    print("="*70)
    
    # Write log file if requested
    if args.log_file:
        log_path = os.path.join(REPO_ROOT, args.log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results logged to: {log_path}")
    
    # Exit with error code if any failures
    if results['failed_count'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

