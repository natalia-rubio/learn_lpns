#!/usr/bin/env python3
"""
Generate svZeroDSolver input files from centerline geometry and simulation results.
This script creates:
1. Geometric 0D input files (from centerline geometry)
2. Calibration input files (from 3D or 1D results)
3. Calibrated output files (by running svZeroDCalibrator)

Based on the workflow in richter2024-paper-tools.
"""

import glob
import os
import sys
sys.path.append("/Users/natalia/cursor_access/learn_lpns")
import json
import vtk
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from typing import Optional
import csv
from collections import defaultdict, OrderedDict
from util.zerod_calibration.run_config_canonical import canonical_run_config_for_data_paths
from util.zerod_calibration.oned_to_zerod import *
from util.zerod_calibration.post_processing import *
from util.zerod_calibration.bifurcation_splitting import *
from util.zerod_calibration.file_io import *
from util.zerod_calibration.bc_fitting import *
from util.zerod_calibration.post_processing import *
from util.zerod_calibration.bifurcation_splitting import *
from util.zerod_calibration.file_io import *
from util.zerod_calibration.bc_fitting import *
from util.zerod_calibration.inflow_handling import *
from util.zerod_calibration.zerod_handling import *
from util.zerod_calibration.calibration import *
from util.zerod_calibration.forward_simulation import *
from util.zerod_calibration.geometric_params import *
from util.zerod_calibration.centerline_path_extraction import *
from util.zerod_calibration.generate_baseline_0d import *
try:
    from scipy.interpolate import CubicSpline, interp1d
    HAS_SCIPY_INTERP = True
except (ImportError, ValueError, AttributeError):
    # Handle import errors and version incompatibility issues
    HAS_SCIPY_INTERP = False
    try:
        from scipy.interpolate import CubicSpline
    except (ImportError, ValueError, AttributeError):
        CubicSpline = None
from scipy.interpolate import interp1d
from vtk.util.numpy_support import vtk_to_numpy as v2n


def _jax_set_type_for_data_processing(args):
    """
    Return the jax_arrays subfolder to use when we run data processing from generate_zerod_inputs.
    - In cross-validation mode (--model-dir with _trial_ or --trial-id set): use "trial_{N}" so we
      don't overwrite the "test" data used for training.
    - Otherwise: use "forward" so we don't overwrite "test" (training/splits).
    """
    if getattr(args, 'model_dir', None):
        base = os.path.basename(args.model_dir.rstrip(os.sep))
        if '_trial_' in base:
            try:
                trial_part = base.split('_trial_')[-1]
                trial_num = int(trial_part.split('_')[0])
                return f"trial_{trial_num}"
            except (ValueError, IndexError):
                pass
    if getattr(args, 'trial_id', None) is not None:
        return f"trial_{args.trial_id}"
    return "forward"


def get_run_config_suffix(
    stenosis_off=False,
    symmetric_loss=False,
    penalty_off=False,
    normalize=False,
):
    """
    Build a suffix for paths so different CV runs (stenosis-off, symmetric-loss, penalty-off)
    are stored separately. Returns "base" when no flags are set so there is always a config subfolder.

    ``normalize`` is accepted for backward compatibility but ignored (z-normalization removed).
    """
    del normalize
    parts = []
    if stenosis_off:
        parts.append("stenosis_off")
    if symmetric_loss:
        parts.append("symmetric")
    if penalty_off:
        parts.append("penalty_off")
    return "_".join(parts) if parts else "base"


def main():
    parser = argparse.ArgumentParser(
        description="Generate svZeroDSolver input files and run calibration"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_1)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_000)')

    parser.add_argument('--junction-types', type=lambda s: [x.strip() for x in s.split(',') if x.strip()],
                       default='BloodVesselJunction',
                       help='Comma-separated junction types for calibration (default: BloodVesselJunction only)')
    parser.add_argument('--zoom-start', type=int, default=None,
                       help='Start index for zoom window (shaded region in plots). Default: 599')
    parser.add_argument('--zoom-end', type=int, default=None,
                       help='End index for zoom window (shaded region in plots). Default: 699')


    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed MSE comparison table')
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
    parser.add_argument('--plot-junction-pressure-diff', action='store_true',
                       help='Generate junction pressure difference plots (off by default)')
    parser.add_argument('--NN-only', action='store_true',
                       help='Only run NN inference and forward simulation on NN inputs (skip calibration)')
    parser.add_argument('--no-redo', action='store_true',
                       help='Skip recreating files if they already exist (check at each step)')
    parser.add_argument('--model-dir', default=None,
                       help='Directory containing rri_{set_name}_pred_{0,1,2}_model files (default: results/models/{set_name}/{geometry_variant})')
    parser.add_argument('--NN-vessel', action='store_true', dest='NN_vessel',
                       help='Also run vessel NN inference: predict vessel R/S/L and write *_NN_JunctionAndVessel.json + forward sim')
    parser.add_argument('--trial-id', type=int, default=None,
                       help='If set (e.g. from cross-validation), append _trial_{id} to plot output paths and filenames')
    parser.add_argument('--stenosis-off', action='store_true', dest='stenosis_off',
                       help='Turn off stenosis: calibrate_stenosis_coefficient=False, set all stenosis to 0, do not use NN to predict stenosis')
    parser.add_argument('--penalty-off', action='store_true', dest='penalty_off',
                       help='Zero L2 penalties on R and stenosis when stenosis is included. Incompatible with --stenosis-off.')
    parser.add_argument('--symmetric-loss', action='store_true', dest='symmetric_loss',
                       help='Record that NN was trained with symmetric loss (for path naming; does not change inference)')
    parser.add_argument(
        '--run-config',
        default=None,
        metavar='SUFFIX',
        help='Optional path suffix for zeroD/ml_inputs (e.g. stenosis_off_symmetric_gen_loss). '
        'Must match flags from --stenosis-off/...; a trailing _gen_loss is an '
        'extra variant (own jax/splits paths + gen-weighted loss) and is ignored only when '
        'checking flag parity.',
    )

    args = parser.parse_args(); verbose = args.verbose
    # Track generated/existing files for logging
    generated_files = []
    
    # If NN-only mode, set appropriate skip flags
    if args.NN_only:
        args.skip_observation = True
        args.skip_calibration = True
        # Force BloodVesselJunction to be in junction_types if not already
        if 'BloodVesselJunction' not in args.junction_types:
            args.junction_types = ['BloodVesselJunction']
    # Run-config suffix: stenosis-off, symmetric-loss, etc. for path separation
    flag_run_config_suffix = get_run_config_suffix(
        stenosis_off=getattr(args, 'stenosis_off', False),
        symmetric_loss=getattr(args, 'symmetric_loss', False),
        penalty_off=getattr(args, 'penalty_off', False),
    )
    rc_arg = getattr(args, 'run_config', None)
    if rc_arg is not None and str(rc_arg).strip():
        rc = str(rc_arg).strip()
        canon = canonical_run_config_for_data_paths(rc) or rc
        if canon != flag_run_config_suffix:
            parser.error(
                f"--run-config {rc!r} does not match flags (canonical {canon!r} vs {flag_run_config_suffix!r} "
                "from --stenosis-off/--symmetric-loss/...)."
            )
        run_config_suffix = rc
    else:
        run_config_suffix = flag_run_config_suffix
    if run_config_suffix:
        print(f"  Run config: {run_config_suffix}")
    if getattr(args, 'stenosis_off', False) and getattr(args, 'penalty_off', False):
        parser.error("Cannot use both --stenosis-off and --penalty-off.")
    # Construct paths: when run_config_suffix is set, use subfolder so different settings don't overwrite
    output_dir = 'data/zeroD'
    if run_config_suffix:
        base_dir = os.path.join(output_dir, args.set_name, run_config_suffix, args.geo_name)
    else:
        base_dir = os.path.join(output_dir, args.set_name, args.geo_name)
    # ML inputs base: separate subfolder per run config so configs don't share CSVs/jax
    if run_config_suffix:
        ml_inputs_base = os.path.join('data', 'ml_inputs', args.set_name, run_config_suffix)
    else:
        ml_inputs_base = os.path.join('data', 'ml_inputs', args.set_name)
    
    geometry_variants, geometric_input_path, geometric_results_csv, calibration_input_path, calibrated_output_path, junction_type_paths, centerline_path, geo_dir = get_paths(base_dir, args)
    print(f"junction_types: {args.junction_types}")
    
    # Helper function to check and track files
    def check_and_track_file(file_path, step_name):
        """Check if file exists, track it, and return whether to skip."""
        if os.path.exists(file_path):
            generated_files.append(file_path)
            if args.no_redo:
                print(f"    ⊘ Skipping {step_name} (file already exists: {file_path})")
                return True
            else:
                print(f"    ⊘ File exists but will be regenerated: {file_path}")
        return False
    
    # Step 1: Create geometric 0D input using SimVascular ROM workflow
    if not args.skip_base_generation:
        bifurcations_geometric_input_path = geometry_variants['bifurcations']['geometric_input']
        
        # Check if base files exist
        skip_base = False
        if args.no_redo:
            bifurcations_EL_geometric_input_path = geometry_variants['bifurcations_EL']['geometric_input']
            if (os.path.exists(geometric_input_path) and 
                os.path.exists(bifurcations_geometric_input_path) and
                os.path.exists(bifurcations_EL_geometric_input_path)):
                check_and_track_file(geometric_input_path, "base generation")
                check_and_track_file(bifurcations_geometric_input_path, "bifurcations generation")
                check_and_track_file(bifurcations_EL_geometric_input_path, "EL-adjusted bifurcations generation")
                skip_base = True
        
        if not skip_base:
            if 'VMR' in args.set_name:
                richter_0d_path = os.path.join('data', 'zeroD', args.set_name, 'richter-0d', args.geo_name+'.json')
                zerod_input = load_from_json(richter_0d_path)
                zerod_input['simulation_parameters']['output_all_cycles'] = True
                zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
                os.makedirs(os.path.dirname(geometric_input_path), exist_ok=True)
                save_to_json(zerod_input, geometric_input_path)
                generated_files.append(geometric_input_path)
                print(f"    ✓ Richter 0D input saved to: {geometric_input_path}")
            else:
                print("\n" + "="*60)
                print("Step 1: Creating geometric 0D input file using SimVascular ROM")
                print("="*60)
                
                # Get geometry directory (may not exist for VMR files)
                if not os.path.exists(geo_dir):
                    print(f"  Warning: Geometry directory not found: {geo_dir}")
                    print(f"  Will use centerline-based workflow (for VMR files)")
                    geo_dir = None
                
                zerod_input, vessel_bc_map = create_geometric_zerod_input_rom(
                    geo_dir, centerline_path, geometric_input_path,
                    simvascular_path=args.simvascular_path,
                    dt=args.dt,
                    num_time_steps=args.num_time_steps,
                    num_cardiac_cycles=args.num_cardiac_cycles
                )
                generated_files.append(geometric_input_path)
            
            print(f"\n  Adding centerline parameters to geometric input...")
            geometric_centerline_input_path = geometric_input_path.replace('geometric_input', 'geometric_centerline_input')
            process_geometric_input(centerline_path, geometric_input_path, geometric_centerline_input_path)
            print(f"  Centerline parameters added to geometric input saved to: {geometric_centerline_input_path}")

            # Generate bifurcations-only version of the geometric input
            print(f"\n  Creating bifurcations-only geometric input...")
            split_junctions_from_files(geometric_centerline_input_path, centerline_path, bifurcations_geometric_input_path)
            generated_files.append(bifurcations_geometric_input_path)
            print(f"  Bifurcations-only geometric input saved to: {bifurcations_geometric_input_path}")
            
            # Generate entrance length-adjusted bifurcations version (if bifurcations file exists)
            bifurcations_EL_geometric_input_path = geometry_variants['bifurcations_EL']['geometric_input']
            if check_and_track_file(bifurcations_EL_geometric_input_path, "EL-adjusted bifurcations generation"):
                pass
            else:
                print(f"\n  Creating entrance length-adjusted bifurcations geometric input...")
                adjust_junction_boundaries_by_entrance_length_from_files(
                    bifurcations_geometric_input_path, 
                    centerline_path, 
                    bifurcations_EL_geometric_input_path,
                    verbose=verbose
                )
                generated_files.append(bifurcations_EL_geometric_input_path)
                print(f"  Entrance length-adjusted bifurcations geometric input saved to: {bifurcations_EL_geometric_input_path}")
    
    # Extract geometric params for all geometry variants
    for geo_variant_name, geo_variant_paths in geometry_variants.items():
        # Skip bifurcations_EL if bifurcations doesn't exist (EL depends on bifurcations)
        if geo_variant_name == 'bifurcations_EL':
            bifurcations_input = geometry_variants['bifurcations']['geometric_input']
            if not os.path.exists(bifurcations_input):
                continue
        
        variant_geometric_input = geo_variant_paths['geometric_input']
        if not os.path.exists(variant_geometric_input):
            continue
        if geo_variant_name == 'original':
            continue
        if geo_variant_name == 'bifurcations_EL':
            # For EL-adjusted geometry, use EL-adjusted structure for parameter extraction
            if not check_and_track_file(variant_geometric_input, f"geometric params extraction for {geo_variant_name}"):
                extract_and_add_geometric_params(
                    centerline_path, 
                    geometry_variants['bifurcations']['geometric_input'],  # Base geometric input (for reference)
                    variant_geometric_input,  # EL-adjusted config to update
                    el_adjusted_geometric_input_path=variant_geometric_input  # Use EL-adjusted structure
                )
                print(f"  Geometric parameters extracted and added to {variant_geometric_input}")
        else:
            # For other variants, standard parameter extraction
            if not check_and_track_file(variant_geometric_input, f"geometric params extraction for {geo_variant_name}"):
                extract_and_add_geometric_params(centerline_path, variant_geometric_input, variant_geometric_input)
                print(f"  Geometric parameters extracted and added to {variant_geometric_input}")
                # Match bifurcations_EL: multi-outlet junctions as BloodVesselJunction with junction_values
                # from geometric_params (non-EL bifurcations skip EL adjustment, so convert here).
                if geo_variant_name == 'bifurcations':
                    with open(variant_geometric_input, 'r') as f:
                        bif_cfg = json.load(f)
                    convert_el_normal_junctions_to_blood_vessel_junction(bif_cfg)
                    with open(variant_geometric_input, 'w') as f:
                        json.dump(bif_cfg, f, indent=4)
                    print(
                        f"  Bifurcations: multi-outlet junctions -> BloodVesselJunction "
                        f"in {variant_geometric_input}"
                    )

    
    # Step 2: Extract observations and create calibration inputs for each junction type
    if not args.skip_observation:
        # Check if all calibration inputs already exist (if --no-redo is set)
        all_calibration_inputs_exist = True
        if args.no_redo:
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                variant_calibration_input = geo_variant_paths['calibration_input']
                if not os.path.exists(variant_calibration_input):
                    all_calibration_inputs_exist = False
                    break
                # Also check junction-type specific calibration inputs
                variant_junction_paths = geo_variant_paths['junction_types']
                for jtype in args.junction_types:
                    jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                    if not os.path.exists(jtype_input_path):
                        all_calibration_inputs_exist = False
                        break
                if not all_calibration_inputs_exist:
                    break
        
        if all_calibration_inputs_exist and args.no_redo:
            print("\n" + "="*60)
            print("Step 2: Creating calibration input files for each junction type")
            print("="*60)
            print("  ⊘ Skipping observation extraction and calibration input creation (all files already exist)")
            # Still need to track the existing files
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                variant_calibration_input = geo_variant_paths['calibration_input']
                check_and_track_file(variant_calibration_input, f"base calibration input for {geo_variant_name}")
                variant_junction_paths = geo_variant_paths['junction_types']
                for jtype in args.junction_types:
                    jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                    check_and_track_file(jtype_input_path, f"{geo_variant_name}/{jtype} calibration input")
        else:
            print("\n" + "="*60)
            print("Step 2: Creating calibration input files for each junction type")
            print("="*60)
        

            # Try to find 1D solution automatically
            # First check data/oneD/set_name/geo_name
            oneD_dir = os.path.join('data', 'oneD', args.set_name, args.geo_name)
            soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
            
            if not os.path.exists(soln_path):
                # Try alternative location: data/reduced_results
                reduced_results_dir = os.path.join('data', 'reduced_results', args.set_name, args.geo_name)
                soln_path = os.path.join(reduced_results_dir, 'unsteady_soln.vtp')
            
            if not os.path.exists(soln_path):
                # Try scratch directory location
                if 'VMR' in args.set_name:
                    oneD_dir = os.path.join('data', 'oneD', "VMR", args.geo_name)
                    alt_soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
                    if os.path.exists(alt_soln_path):
                        soln_path = alt_soln_path
                else:
                    alt_soln_path = os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                                                    args.set_name, args.geo_name, 'unsteady_soln.vtp')
                if os.path.exists(alt_soln_path):
                    soln_path = alt_soln_path
            geometric_input_path = geometry_variants['original']['geometric_input']
            geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)
            if os.path.exists(soln_path):
                observations = extract_observations_from_1d(soln_path, geometric_input_path, geo_dir=geo_dir, start_idx=0)
            else:
                raise FileNotFoundError(f"1D solution not found: {soln_path}")

            # # Use the path that was found earlier
            # oneD_dir = os.path.join('data', 'oneD', args.set_name, args.geo_name)
            # soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
            # if not os.path.exists(soln_path):
            #     reduced_results_dir = os.path.join('data', 'reduced_results', args.set_name, args.geo_name)
            #     soln_path = os.path.join(reduced_results_dir, 'unsteady_soln.vtp')
            # if not os.path.exists(soln_path):
            #     alt_soln_path = os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
            #                                 args.set_name, args.geo_name, 'unsteady_soln.vtp')
            #     if os.path.exists(alt_soln_path):
            #         soln_path = alt_soln_path
            
            # Get geo_dir
            geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)

            # Coronary sets: keep outlet BCs already in geometric input (e.g. from reference 0D);
            # inlet still comes from 1D via calibration input / update_geometric_input_with_calibration_bc.
            skip_outlet_bc_fitting = 'coro' in args.set_name.lower()

            time_step_size = None
            if not skip_outlet_bc_fitting:
                try:
                    time_step_size = timestep_from_1D(soln_path, geo_dir)
                except Exception:
                    time_step_size = None
            
            # Fit outlet resistances from observations (skip for VMR and coro)
            if args.set_name != "VMR" and not skip_outlet_bc_fitting:
                fitted_resistances = fit_outlet_resistances_from_3d(geometric_input_path, observations)
            else:
                fitted_resistances = None

            fitted_rcr = {}
            if skip_outlet_bc_fitting:
                print(
                    "\n  Set name contains 'coro': skipping outlet BC fitting "
                    "(using values already in geometric input); inlet remains from 1D."
                )
            elif time_step_size is not None:
                try:
                    fitted_rcr = fit_outlet_rcr_from_observations(
                        geometric_input_path, observations, dt=time_step_size
                    )
                except Exception as e:
                    print(f"  Warning: RCR fitting failed on original geometry: {e}")
            else:
                print("  Warning: time_step_size unavailable; skipping RCR fitting on original geometry")
            
            
            
            # Process geometry variants: original, bifurcations, and bifurcations_EL
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                # Skip bifurcations_EL if bifurcations doesn't exist (EL depends on bifurcations)
                if geo_variant_name == 'bifurcations_EL':
                    bifurcations_input = geometry_variants['bifurcations']['geometric_input']
                    if not os.path.exists(bifurcations_input):
                        print(f"\n  ✗ Skipping {geo_variant_name}: bifurcations geometric input not found: {bifurcations_input}")
                        continue
                
                print(f"\n" + "-"*50)
                print(f"Processing {geo_variant_name.upper()} geometry variant")
                print("-"*50)
                
                variant_geometric_input = geo_variant_paths['geometric_input']
                variant_calibration_input = geo_variant_paths['calibration_input']
                variant_geometric_results = geo_variant_paths['geometric_results']
                variant_junction_paths = geo_variant_paths['junction_types']
                
                # Check if geometric input exists
                if not os.path.exists(variant_geometric_input):
                    print(f"  ✗ Skipping {geo_variant_name}: geometric input not found: {variant_geometric_input}")
                    continue

                # Apply fitted RCR parameters to the variant (copy over if we fitted on original)
                if fitted_rcr:
                    try:
                        with open(variant_geometric_input, 'r') as f:
                            cfg_tmp = json.load(f)
                        bc_map = {bc.get('bc_name'): bc for bc in cfg_tmp.get('boundary_conditions', []) if bc.get('bc_name')}
                        for bc_name, params in fitted_rcr.items():
                            if bc_name in bc_map:
                                bc_map[bc_name]['bc_values']['Rp'] = float(params[0])
                                bc_map[bc_name]['bc_values']['C'] = float(params[1])
                                bc_map[bc_name]['bc_values']['Rd'] = float(params[2])
                                bc_map[bc_name]['bc_values']['Pd'] = float(params[3])
                        with open(variant_geometric_input, 'w') as f:
                            json.dump(cfg_tmp, f, indent=4)
                        print(f"  Applied fitted RCR parameters to {geo_variant_name} geometric input")
                    except Exception as e:
                        print(f"  Warning: could not apply fitted RCR parameters to {geo_variant_name}: {e}")

                # Create base calibration input for this geometry variant
                if check_and_track_file(variant_calibration_input, f"base calibration input for {geo_variant_name}"):
                    # File exists and no-redo is set, skip creation
                    pass
                else:
                    print(f"\n  Creating base calibration input for {geo_variant_name}...")
                    try:
                        # For bifurcations and bifurcations_EL geometries:
                        # - Always rename existing observations to match bifurcated junction names
                        # - Optionally add synthetic connector-vessel observations (only for bifurcations, not EL)
                        if geo_variant_name in ['bifurcations', 'bifurcations_EL']:
                            with open(variant_geometric_input, 'r') as f:
                                bifurcated_geometric_input = json.load(f)
                            
                            # For bifurcations: generate synthetic connector observations
                            # For bifurcations_EL: skip synthetic observations (vessels already merged/converted)
                            if geo_variant_name == 'bifurcations':
                                include_synthetic_observations = True
                                if include_synthetic_observations:
                                    # Load the original geometric input and centerline data
                                    original_geometric_input_path = geometry_variants['original']['geometric_input']
                                    with open(original_geometric_input_path, 'r') as f:
                                        original_geometric_input = json.load(f)

                                    # Read centerline data
                                    centerline_data, _ = read_centerline_vtp(centerline_path)

                                    print(f"    Generating synthetic observations for connector vessels...")
                                    augmented_observations = generate_connector_observations(
                                        observations,
                                        original_geometric_input,
                                        bifurcated_geometric_input,
                                        centerline_data
                                    )
                                else:
                                    print(f"    Skipping synthetic observations for connector vessels")
                                    print(f"    Renaming existing observation keys to match bifurcated junction names...")
                                    augmented_observations = rename_observations_for_bifurcations(
                                        observations, bifurcated_geometric_input
                                    )
                            else:  # bifurcations_EL
                                # Extract observations directly from 1D solution using centerline_node_ids
                                print(f"    Extracting observations from 1D solution using centerline_node_ids...")
                                from util.zerod_calibration.oned_to_zerod import extract_observations_from_1d_with_node_ids
                                augmented_observations = extract_observations_from_1d_with_node_ids(
                                    centerline_path,
                                    variant_geometric_input,
                                    geo_dir=geo_dir,
                                    start_idx=0,
                                    derivative_method='central'
                                )

                            obs_for_calib = augmented_observations
                        else:
                            # Original geometry: use original observations
                            obs_for_calib = observations

                        # For set names including "priya", 1D solution is short: repeat flow/pressure 5x with time increasing
                        if 'priya' in args.set_name:
                            from util.zerod_calibration.calibration import repeat_observations_in_time
                            obs_for_calib = repeat_observations_in_time(obs_for_calib, num_repeats=5)
                            print(f"    Repeated observation series 5x for extended time (set_name contains 'priya')")

                        create_calibration_input(
                            variant_geometric_input, obs_for_calib, variant_calibration_input,
                            centerline_soln_path=soln_path, geo_dir=geo_dir,
                            stenosis_off=getattr(args, 'stenosis_off', False),
                            penalty_off=getattr(args, 'penalty_off', False),
                            set_name=getattr(args, 'set_name', None),
                        )
                        generated_files.append(variant_calibration_input)
                        print(f"    ✓ Base calibration input saved to: {variant_calibration_input}")
                        
                        # Update base calibration input with fitted outlet BCs (skip for VMR)
                        if fitted_resistances:
                            print(f"\n  Updating base calibration input with fitted outlet BCs...")
                            update_outlet_bcs_in_file(variant_calibration_input, fitted_resistances, f"{geo_variant_name} calibration input")
                        
                        # Update geometric input with BC from calibration input
                        update_geometric_input_with_calibration_bc(variant_geometric_input, variant_calibration_input)
                    except Exception as e:
                        raise Exception(f"Failed to create calibration input for {geo_variant_name}: {e}")
                
                # Create calibration input variants for each junction type
                print(f"\n  Creating calibration input variants for each junction type...")
                try:
                    with open(variant_calibration_input, 'r') as f:
                        base_calibration_config = json.load(f)
                    
                    for jtype in args.junction_types:
                        jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                        if check_and_track_file(jtype_input_path, f"{geo_variant_name}/{jtype} calibration input"):
                            continue
                        
                        print(f"    Creating {geo_variant_name}/{jtype} calibration input...")
                        
                        # Apply junction type modification to calibration input
                        jtype_config = modify_junction_types(base_calibration_config, jtype)
                        
                        with open(jtype_input_path, 'w') as f:
                            json.dump(jtype_config, f, indent=4)
                        generated_files.append(jtype_input_path)
                except Exception as e:
                    raise Exception(f"Failed to create junction type calibration inputs for {geo_variant_name}: {e}")
            

    # Step 3: Run calibration for each junction type
    if not args.skip_calibration:
        # Process each geometry variant
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            print(f"\n  Running calibration for {geo_variant_name} geometry...")
            variant_junction_paths = geo_variant_paths['junction_types']
            
            # Run calibration for each junction type variant
            for jtype in args.junction_types:
                jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                jtype_output_path = variant_junction_paths[jtype]['calibrated_output']
                
                if check_and_track_file(jtype_output_path, f"calibration for {geo_variant_name}/{jtype}"):
                    continue
                
                print(f"\n    Calibrating {geo_variant_name}/{jtype}...")
                
                try:
                    calibrated_config = run_calibration(jtype_input_path, jtype_output_path)
                    generated_files.append(jtype_output_path)
                    print(f"      ✓ Calibration completed for {geo_variant_name}/{jtype}")
                    # Junction types are now preserved by the calibrator
                except Exception as e:
                    raise Exception(f"Calibration failed for {geo_variant_name}/{jtype}: {e}")
    
    # Step 3.5: Run data processing pipeline for neural network training data (after calibration)
    # Process each geometry variant separately (bifurcations and bifurcations_EL)
    if not args.skip_calibration:
        print(f"\n  Running data processing pipeline for neural network...")
        geo_name_for_ml = args.geo_name.replace('tree_', '')
        
        # Process each geometry variant that has been calibrated
        for geo_variant_name in ['bifurcations', 'bifurcations_EL']:
            if geo_variant_name not in geometry_variants:
                continue
            
            geo_variant_paths = geometry_variants[geo_variant_name]
            variant_calibration_input = geo_variant_paths['calibration_input']
            
            # Check if this variant has been processed (has calibration input)
            if not os.path.exists(variant_calibration_input):
                print(f"  ⊘ Skipping data processing for {geo_variant_name} (no calibration input)")
                continue
            
            # Check if BloodVesselJunction calibration output exists (required for data processing)
            if 'BloodVesselJunction' not in args.junction_types:
                print(f"  ⊘ Skipping data processing for {geo_variant_name} (BloodVesselJunction not in junction types)")
                continue
            
            variant_junction_paths = geo_variant_paths['junction_types']
            calib_output_path = variant_junction_paths['BloodVesselJunction']['calibrated_output']
            
            if not os.path.exists(calib_output_path):
                print(f"  ⊘ Skipping data processing for {geo_variant_name} (BloodVesselJunction calibration output not found)")
                continue
            
            print(f"\n  Processing {geo_variant_name} geometry variant for ML pipeline...")
            try:
                import subprocess
                jax_set_type = _jax_set_type_for_data_processing(args)
                run_data_processing_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), '..', 'data_processing', 'run_data_processing.py'),
                    '--set-name', args.set_name,
                    '--geometry-variant', geo_variant_name,
                    '--set-type', jax_set_type,
                    '--geometries', geo_name_for_ml,
                    '--percent-train', '1',
                    '--seed', '0',
                    '--data-root', 'data',
                    '--run-config', run_config_suffix,
                ]
                if verbose:
                    run_data_processing_cmd.append('--verbose')
                result = subprocess.run(
                    run_data_processing_cmd,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    capture_output=False,
                    text=True
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Data processing failed for {geo_variant_name}: {result.stderr}")
                print(f"  ✓ Data processing pipeline completed for {geo_variant_name}")
                
            except Exception as e:
                raise Exception(f"Failed to run data processing pipeline for {geo_variant_name}: {e}")
            
        
    # Step 3.6: After calibration, run NN inference for BloodVesselJunction on bifurcations and bifurcations_EL geometries
    # (Also runs in NN-only mode, using geometric input instead of calibrated output)
    if 'BloodVesselJunction' in args.junction_types:
        # Process both bifurcations and bifurcations_EL geometry variants
        for geo_variant_name in ['bifurcations', 'bifurcations_EL']:
            if geo_variant_name not in geometry_variants:
                continue
            
            geo_variant_paths = geometry_variants[geo_variant_name]
            variant_junction_paths = geo_variant_paths['junction_types']
            variant_geometric_input = geo_variant_paths['geometric_input']
            
            # Skip if bifurcations_EL depends on bifurcations and it doesn't exist
            # if geo_variant_name == 'bifurcations_EL':
            #     bifurcations_input = geometry_variants['bifurcations']['geometric_input']
            #     if not os.path.exists(bifurcations_input):
            #         print(f"  ⊘ Skipping NN inference for {geo_variant_name} (bifurcations geometric input not found)")
            #         continue
            
            # Check if geometric input exists
            if not os.path.exists(variant_geometric_input):
                print(f"  ⊘ Skipping NN inference for {geo_variant_name} (geometric input not found)")
                continue
            
            # Construct output path with geometry variant name
            nn_output_path = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction.json')
            
            if check_and_track_file(nn_output_path, f"NN inference for {geo_variant_name}/BloodVesselJunction"):
                # File exists and no-redo is set, skip NN inference
                pass
            else:
                print(f"\n    Running neural network inference for {geo_variant_name}/BloodVesselJunction...")
                try:
                    # Import NN-related modules
                    from util.data_processing.inputs_from_0d_config import (
                        load_junction_geometric_features,
                        compute_junction_flow_splits,
                    )
                    from util.neural_network.nn_model import predict
                    from util.neural_network.nn_util import dill_load
                    import jax.numpy as jnp
                    
                    # Load geometric config as base (no calibrated params); only junction R/S/L will be overwritten by NN
                    if not os.path.exists(variant_geometric_input):
                        raise FileNotFoundError(f"Geometric input not found: {variant_geometric_input}")
                    
                    with open(variant_geometric_input, 'r') as f:
                        nn_config = json.load(f)
                    
                    # For bifurcations (non-EL), convert NORMAL_JUNCTION -> BloodVesselJunction and set
                    # junction_values from geometric_params; NN prediction loop will overwrite with predictions.
                    if geo_variant_name == 'bifurcations':
                        convert_el_normal_junctions_to_blood_vessel_junction(nn_config)
                    
                    # Extract geometric features using the same workflow as data processing
                    # This ensures we use the same 13 features that the model was trained on
                    from util.data_processing.data_dict_from_csvs import (
                        _read_csv_matrix,
                        _clamp_tortuosity,
                        filter_features_from_array,
                    )
                    
                    # Check if CSV file exists (from data processing); use run-config-specific path when set
                    csv_path = os.path.join(ml_inputs_base, geo_variant_name, args.geo_name, 'geometric_features.csv')
                    
                    # if os.path.exists(csv_path):
                    #     # Read from CSV and apply same feature selection as in data processing
                    #     print(f"  Reading features from CSV: {csv_path}")
                    #     geom_header, geom_X = _read_csv_matrix(csv_path)
                        
                    #     # Apply the same feature selection using the reusable function
                    #     X, feature_names = filter_features_from_array(
                    #         geom_X, geom_header, remap_tortuosity=True
                    #     )
                        
                    #     # Extract junction names from geometric input (needed for mapping predictions)
                    #     X_full, _, junction_names, _ = load_junction_geometric_features(
                    #         variant_geometric_input,
                    #         require_two_outlets=True,
                    #         verbose=False
                    #     )
                    #     # Match rows: CSV should have same number of rows as feature extraction
                    #     if len(X) != len(X_full):
                    #         raise ValueError(
                    #             f"Row count mismatch: CSV has {len(X)} rows, but feature extraction has {len(X_full)} rows"
                    #         )
                    # else:

                        # CSV doesn't exist, extract features directly and filter
                    print(f"  Extracting features directly from geometric input")
                    X_full, feature_names_full, junction_names, outlet_primary_names = load_junction_geometric_features(
                        variant_geometric_input,
                        require_two_outlets=True,
                        verbose=True
                    )
                    # Add flow_split from geometric results when available (same as run_data_processing)
                    geometric_results_path = variant_geometric_input.replace(
                        "_geometric_input.json", "_geometric_results.csv"
                    )
                    if os.path.exists(geometric_results_path):
                        flow_splits = compute_junction_flow_splits(
                            variant_geometric_input, geometric_results_path
                        )
                        flow_split_col = []
                        for i, jname in enumerate(junction_names):
                            (out0_name, out1_name), (fs0, fs1) = flow_splits.get(
                                jname, (("", ""), (float("nan"), float("nan")))
                            )
                            primary = outlet_primary_names[i]
                            val = fs0 if primary == out0_name else (
                                fs1 if primary == out1_name else float("nan")
                            )
                            flow_split_col.append(val)
                        X_full = np.column_stack([X_full, flow_split_col])
                        feature_names_full = feature_names_full + ["flow_split"]
                        flow_split_arr = np.asarray(flow_split_col, dtype=float)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            flow_split_inv = np.where(
                                np.isfinite(flow_split_arr) & (flow_split_arr > 0),
                                100.0 / flow_split_arr,
                                np.nan,
                            )
                        X_full = np.column_stack([X_full, flow_split_inv])
                        feature_names_full = feature_names_full + ["flow_split_inv"]
                    else:
                        # Geometric results not available; add flow_split and flow_split_inv as NaN
                        n_rows = X_full.shape[0]
                        X_full = np.column_stack([X_full, np.full(n_rows, np.nan), np.full(n_rows, np.nan)])
                        feature_names_full = feature_names_full + ["flow_split", "flow_split_inv"]
                    #save X_full to a csv file
                    import pandas as pd
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    pd.DataFrame(X_full, columns=feature_names_full).to_csv(csv_path, index=False)
                    
                    # Apply the same feature selection using the reusable function
                    X, feature_names = filter_features_from_array(
                        X_full, feature_names_full
                    )
                    _clamp_tortuosity(X, feature_names)
                    
                    if len(X) == 0:
                        raise ValueError("No junctions found in geometric features")
                    
                    # Count unique junction names (some may have been skipped for swapped row)
                    unique_junction_names = list(set(junction_names))
                    print(f"  Loaded {len(X)} feature rows for {len(unique_junction_names)} unique junctions")
                    print(f"  Junction names in feature extraction: {unique_junction_names}")
                    print(f"  Selected {len(feature_names)} features (matching training data): {feature_names}")
                    
                    X_jax = jnp.array(X, dtype=jnp.float32)
                    print(f"  Neural network input dimensions: {X_jax.shape} (rows={X_jax.shape[0]}, features={X_jax.shape[1]})")
                
                    # Load the three trained models (use --model-dir if set, e.g. for CV trials)
                    if getattr(args, 'model_dir', None):
                        model_dir = args.model_dir
                    else:
                        model_dir = os.path.join('results', 'models', args.set_name, geo_variant_name)
                    model_base_name = f"rri_{args.set_name}_pred"
                    model_paths = [
                        os.path.join(model_dir, f"{model_base_name}_0_model"),
                        os.path.join(model_dir, f"{model_base_name}_1_model"),
                        os.path.join(model_dir, f"{model_base_name}_2_model"),
                    ]
                    
                    for i, model_path in enumerate(model_paths):
                        if not os.path.exists(model_path):
                            print(f"  NN model not found: {model_path}")
                            raise FileNotFoundError(f"Model not found: {model_path}")
                    
                    # Load models and get predictions
                    raw_predictions = []
                    for i, model_path in enumerate(model_paths):
                        print(f"      Loading model {i+1}/3: {model_path}")
                        model = dill_load(model_path)
                        use_leaky = getattr(model, "use_leaky_relu", False)
                        pred = predict(X_jax, model.weights, use_leaky)
                        raw_predictions.append(np.array(pred).flatten())
                    
                    output_names = ['R_poiseuille', 'stenosis_coefficient', 'L']
                    predictions = raw_predictions
                    for coef_idx, pred in enumerate(predictions):
                        print(f"      {output_names[coef_idx]}: "
                              f"pred range=[{pred.min():.4f}, {pred.max():.4f}]")
                    
                    # predictions[0] = R_poiseuille (one value per row)
                    # predictions[1] = stenosis_coefficient (one value per row)
                    # predictions[2] = L (one value per row)
                    pred_R = np.array(predictions[0])
                    pred_S = np.zeros_like(pred_R) if getattr(args, 'stenosis_off', False) else np.array(predictions[1])
                    pred_L = np.array(predictions[2])

                    # Verify prediction array sizes match input
                     # Note: junction_names may have duplicates (same junction appears twice for swapped rows)
                     # So we compare against the actual number of input rows
                    if len(pred_R) != len(X):
                        raise ValueError(
                            f"Prediction array size mismatch: input has {len(X)} rows, "
                            f"but predictions have {len(pred_R)} values. "
                            f"Expected one prediction per input row."
                        )
                    
                    # Build a mapping from (junction_name, primary_outlet_name) -> row_idx
                    # This is the authoritative mapping: each row's prediction applies to
                    # the primary_outlet of that row (the outlet whose features are in the
                    # outlet0 position for that row).
                    primary_outlet_to_row = {}  # (junc_name, outlet_name) -> row_idx
                    junction_name_to_row_indices = {}
                    for row_idx, (junc_name, pout_name) in enumerate(zip(junction_names, outlet_primary_names)):
                        primary_outlet_to_row[(junc_name, pout_name)] = row_idx
                        if junc_name not in junction_name_to_row_indices:
                            junction_name_to_row_indices[junc_name] = []
                        junction_name_to_row_indices[junc_name].append(row_idx)
                    
                    print(f"      Built prediction mapping: {len(primary_outlet_to_row)} (junction, outlet) entries")
                    for (jn, on), ri in primary_outlet_to_row.items():
                        vid = int(X_full[ri, 0])  # outlet_vessel_id is first column
                        print(f"        ({jn}, {on}) -> row {ri}, vessel_id={vid}")
                
                    # Map predictions back to junction_values
                    vessels = nn_config.get('vessels', [])
                    vessel_id_to_name = {v.get('vessel_id'): v.get('vessel_name', '') for v in vessels}
                    vessel_name_to_id = {v.get('vessel_name', ''): v.get('vessel_id') for v in vessels}
                    
                    # Process each junction in the config
                    for junc in nn_config.get('junctions', []):
                        junc_name = junc.get('junction_name', '')
                        
                        # Skip if this junction wasn't in the feature extraction
                        if junc_name not in junction_name_to_row_indices:
                            if junc.get('junction_type') == 'BloodVesselJunction':
                                if 'junction_values' not in junc:
                                    outlet_vessel_ids = junc.get('outlet_vessels', [])
                                    num_outlets = len(outlet_vessel_ids)
                                    junc['junction_values'] = {
                                        'R_poiseuille': [0.0] * num_outlets,
                                        'stenosis_coefficient': [0.0] * num_outlets,
                                        'L': [0.0] * num_outlets,
                                    }
                            continue
                        
                        # Get outlet vessels in file order (junction's outlet_vessels list)
                        outlet_vessel_ids = junc.get('outlet_vessels', [])
                        if len(outlet_vessel_ids) != 2:
                            continue
                        
                        outlet_vessel_names = [vessel_id_to_name.get(vid, '') for vid in outlet_vessel_ids]
                        
                        # Initialize junction_values
                        if 'junction_values' not in junc:
                            junc['junction_values'] = {}
                        
                        R_values = [0.0] * len(outlet_vessel_ids)
                        S_values = [0.0] * len(outlet_vessel_ids)
                        L_values = [0.0] * len(outlet_vessel_ids)
                        
                        # For each outlet vessel (in file order), find its prediction row
                        for file_idx, (vid, vname) in enumerate(zip(outlet_vessel_ids, outlet_vessel_names)):
                            # Skip non-EL connectors (zero parameters)
                            if 'connector' in vname and 'connectorEL' not in vname:
                                print(f"        {junc_name}: outlet[{file_idx}] {vname} (id={vid}) -> connector, set to 0")
                                continue
                            
                            # Look up the row where this outlet was the primary outlet
                            row_idx = primary_outlet_to_row.get((junc_name, vname))
                            
                            if row_idx is not None:
                                # Verify vessel_id consistency
                                expected_vid = int(X_full[row_idx, 0])
                                if expected_vid != vid:
                                    raise ValueError(
                                        f"Vessel ID mismatch for {junc_name}/{vname}: "
                                        f"config has vessel_id={vid}, but feature row {row_idx} has "
                                        f"outlet_vessel_id={expected_vid}"
                                    )
                                
                                R_values[file_idx] = float(pred_R[row_idx])
                                S_values[file_idx] = float(pred_S[row_idx])
                                L_values[file_idx] = float(pred_L[row_idx])
                                print(f"        {junc_name}: outlet[{file_idx}] {vname} (id={vid}) -> "
                                      f"row {row_idx}: R={R_values[file_idx]:.4f}, S={S_values[file_idx]:.4f}, L={L_values[file_idx]:.4f}")
                            else:
                                # No dedicated prediction row for this outlet (e.g., it was
                                # only in the secondary/outlet1 position). Fall back to the
                                # other outlet's row if available.
                                other_outlet = [on for on in outlet_vessel_names if on != vname]
                                fallback_row = None
                                if other_outlet:
                                    fallback_row = primary_outlet_to_row.get((junc_name, other_outlet[0]))
                                if fallback_row is not None:
                                    R_values[file_idx] = float(pred_R[fallback_row])
                                    S_values[file_idx] = float(pred_S[fallback_row])
                                    L_values[file_idx] = float(pred_L[fallback_row])
                                    print(f"        {junc_name}: outlet[{file_idx}] {vname} (id={vid}) -> "
                                          f"fallback from row {fallback_row} (primary={other_outlet[0]}): "
                                          f"R={R_values[file_idx]:.4f}, S={S_values[file_idx]:.4f}, L={L_values[file_idx]:.4f}")
                                else:
                                    print(f"        ⚠ {junc_name}: outlet[{file_idx}] {vname} (id={vid}) -> "
                                          f"no prediction row found, keeping zeros")
                        
                        junc['junction_values']['R_poiseuille'] = R_values
                        junc['junction_values']['stenosis_coefficient'] = S_values
                        junc['junction_values']['L'] = L_values
                    
                    # Vessels already geometric (nn_config was loaded from variant_geometric_input)
                    # Save the NN-modified config (already checked at start of block)
                    with open(nn_output_path, 'w') as f:
                        json.dump(nn_config, f, indent=4)
                    generated_files.append(nn_output_path)
                    print(f"      ✓ Neural network predictions applied and saved to {nn_output_path}")
                
                except Exception as e:
                    print(f"  Neural network inference failed for {geo_variant_name}/BloodVesselJunction: {e}")
                    print(f"  Skipping NN junction predictions and forward simulation for this variant (models may be missing).")

            # Step 3.7 (optional): Vessel NN inference: predict vessel R/S/L and write NN_JunctionAndVessel config
            if getattr(args, 'NN_vessel', False):
                from util.data_processing.inputs_from_0d_config import load_vessel_geometric_features
                from util.data_processing.data_dict_from_csvs import (
                    _clamp_tortuosity,
                    filter_features_from_array,
                    get_default_include_features_vessel,
                )
                from util.neural_network.nn_model import predict as nn_predict
                from util.neural_network.nn_util import dill_load
                import jax.numpy as jnp
                # When model_dir is trial-specific (CV), vessel models exist only for that variant
                model_dir_basename = os.path.basename(getattr(args, 'model_dir', '') or '')
                if '_trial_' in model_dir_basename:
                    _model_variant = model_dir_basename.split('_trial_')[0]
                else:
                    _model_variant = None
                for geo_variant_name in ['bifurcations', 'bifurcations_EL']:
                    if geo_variant_name not in geometry_variants:
                        continue
                    if _model_variant is not None and geo_variant_name != _model_variant:
                        if args.verbose:
                            print(f"  ⊘ Skipping vessel NN for {geo_variant_name} (trial model dir is for {_model_variant})")
                        continue
                    nn_output_path = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction.json')
                    if not os.path.exists(nn_output_path):
                        print(f"  ⊘ Skipping vessel NN for {geo_variant_name}: NN junction config not found")
                        continue
                    nn_jv_path = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel.json')
                    nn_vessel_only_path = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly.json')
                    if (check_and_track_file(nn_jv_path, f"Vessel NN inference for {geo_variant_name}") and
                            os.path.exists(nn_vessel_only_path)):
                        continue
                    print(f"\n    Running vessel NN inference for {geo_variant_name}...")
                    try:
                        with open(nn_output_path, 'r') as f:
                            jv_config = json.load(f)
                        # Use geometric input for vessel features (same as training) so feature
                        # distribution matches; predictions are still written into jv_config by vessel_id.
                        variant_geometric_input = geometry_variants[geo_variant_name]['geometric_input']
                        if not os.path.exists(variant_geometric_input):
                            raise FileNotFoundError(
                                f"Geometric input not found for vessel features: {variant_geometric_input}"
                            )
                        X_v, feat_names_v, vessel_ids, vessel_names = load_vessel_geometric_features(
                            variant_geometric_input, verbose=args.verbose
                        )
                        if len(X_v) == 0:
                            print(f"      No non-connector vessels, skipping vessel NN for {geo_variant_name}")
                            continue
                        # Filter to same 21 features used in vessel NN training (avoids 21 vs 25 shape mismatch)
                        X_v, feat_names_v = filter_features_from_array(
                            X_v, feat_names_v,
                            include_features=get_default_include_features_vessel(),
                        )
                        _clamp_tortuosity(X_v, feat_names_v)
                        if getattr(args, 'model_dir', None) and '_trial_' in os.path.basename(args.model_dir):
                            _base = os.path.dirname(args.model_dir)
                            _name = os.path.basename(args.model_dir).replace('_trial_', '_vessel_trial_', 1)
                            vessel_model_dir = os.path.join(_base, _name)
                        else:
                            vessel_model_dir = os.path.join(
                                'results', 'models', args.set_name, geo_variant_name + '_vessel')
                        X_v_jax = jnp.array(np.array(X_v, dtype=np.float64), dtype=jnp.float32)
                        model_paths = [os.path.join(vessel_model_dir, f"rri_{args.set_name}_vessel_pred_{i}_model") for i in range(3)]
                        for mp in model_paths:
                            if not os.path.exists(mp):
                                raise FileNotFoundError(f"Vessel model not found: {mp}")
                        raw_predictions_v = []
                        for i, mp in enumerate(model_paths):
                            model = dill_load(mp)
                            use_leaky_v = getattr(model, "use_leaky_relu", False)
                            pred = nn_predict(X_v_jax, model.weights, use_leaky_v)
                            raw_predictions_v.append(np.array(pred).flatten())
                        pred_R_v = np.array(raw_predictions_v[0])
                        pred_S_v = np.array(raw_predictions_v[1])
                        pred_L_v = np.array(raw_predictions_v[2])
                        if getattr(args, 'stenosis_off', False):
                            pred_S_v = np.zeros_like(pred_R_v)
                        vessel_id_to_row = {vid: i for i, vid in enumerate(vessel_ids)}
                        for v in jv_config.get('vessels', []):
                            vname = (v.get('vessel_name') or '').lower()
                            if 'connector' in vname:
                                continue
                            vid = v.get('vessel_id')
                            row = vessel_id_to_row.get(vid)
                            if row is None:
                                continue
                            z = dict(v.get('zero_d_element_values') or {})
                            z['R_poiseuille'] = float(pred_R_v[row])
                            z['stenosis_coefficient'] = float(pred_S_v[row])
                            z['L'] = float(pred_L_v[row])
                            v['zero_d_element_values'] = z
                        with open(nn_jv_path, 'w') as f:
                            json.dump(jv_config, f, indent=4)
                        generated_files.append(nn_jv_path)
                        print(f"      ✓ Vessel NN predictions applied and saved to {nn_jv_path}")
                        # NN_vessel modality: geometric junctions + NN vessel params (default when --NN-vessel)
                        with open(variant_geometric_input, 'r') as f:
                            vessel_only_config = json.load(f)
                        for v in vessel_only_config.get('vessels', []):
                            vname = (v.get('vessel_name') or '').lower()
                            if 'connector' in vname:
                                continue
                            vid = v.get('vessel_id')
                            row = vessel_id_to_row.get(vid)
                            if row is None:
                                continue
                            if 'zero_d_element_values' not in v:
                                v['zero_d_element_values'] = {}
                            v['zero_d_element_values']['R_poiseuille'] = float(pred_R_v[row])
                            v['zero_d_element_values']['stenosis_coefficient'] = float(pred_S_v[row])
                            v['zero_d_element_values']['L'] = float(pred_L_v[row])
                        with open(nn_vessel_only_path, 'w') as f:
                            json.dump(vessel_only_config, f, indent=4)
                        generated_files.append(nn_vessel_only_path)
                        print(f"      ✓ NN_vessel (geometric junctions + NN vessels) saved to {nn_vessel_only_path}")
                    except Exception as e:
                        print(f"      Vessel NN inference failed for {geo_variant_name}: {e}")
                        print(f"      Skipping vessel NN predictions and related forward simulations for this variant.")

    # Sync BCs from calibrated output into NN configs so RCR (and other outlet BCs) match
    # Run even if calibration was skipped, so we can use a preexisting calibrated output file
    if 'BloodVesselJunction' in args.junction_types:
        for geo_variant_name in ['bifurcations', 'bifurcations_EL']:
            if geo_variant_name not in geometry_variants:
                continue
            variant_junction_paths = geometry_variants[geo_variant_name]['junction_types']
            calib_output_path = variant_junction_paths.get('BloodVesselJunction', {}).get('calibrated_output')
            if not calib_output_path or not os.path.exists(calib_output_path):
                continue
            nn_output_path = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction.json')
            nn_jv_path = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel.json')
            nn_vessel_only_path = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly.json')
            nn_paths = [p for p in [nn_output_path, nn_jv_path, nn_vessel_only_path] if os.path.exists(p)]
            if nn_paths:
                n_updated = sync_nn_config_bcs_from_calibration(nn_paths, calib_output_path, verbose=args.verbose)
                if n_updated and args.verbose:
                    print(f"  Synced boundary conditions from calibrated output into {n_updated} NN config(s) for {geo_variant_name}")

    # Step 4: Run forward simulations for each geometry variant
    if not args.skip_forward:
        # # Adjust refinement factor based on length of inlet flow waveform
        # inlet_flow_waveform = get_inlet_flow_waveform(args.set_name, args.geo_name)
        # if len(inlet_flow_waveform) > 500:
        #     refinement_factor = 1
        # else:
        #     refinement_factor = int(500/len(inlet_flow_waveform))
        # print(f"Refinement factor: {refinement_factor}")
        if args.NN_only:
            # In NN-only mode, only run forward simulation on NN input
            print(f"\n  Running forward simulation (NN-only mode)...")
            # Process both bifurcations and bifurcations_EL
            for geo_variant_name in ['bifurcations', 'bifurcations_EL']:
                if geo_variant_name not in geometry_variants:
                    continue
                
                geo_variant_paths = geometry_variants[geo_variant_name]
                variant_junction_paths = geo_variant_paths['junction_types']
                
                # Run forward simulation for NN-modified BloodVesselJunction
                if 'BloodVesselJunction' in args.junction_types:
                    nn_output_path = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction.json')
                    nn_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction_results.csv')
                    
                    if os.path.exists(nn_output_path):
                        if check_and_track_file(nn_results_csv, f"NN forward simulation for {geo_variant_name}/BloodVesselJunction"):
                            pass
                        else:
                            print(f"\n    Running simulation with NN-modified {geo_variant_name}/BloodVesselJunction input...")
                            try:
                                # Use same BC source as full pipeline when available: BloodVesselJunction calibration input.
                                # Fall back to geometric input only when calibration was not run (e.g. NN-only / CV val geometry).
                                bvj_calib_input = geo_variant_paths['junction_types']['BloodVesselJunction']['calibration_input']
                                if os.path.exists(bvj_calib_input):
                                    bvj_input_path = bvj_calib_input
                                else:
                                    bvj_input_path = geo_variant_paths['geometric_input']
                                if not os.path.exists(bvj_input_path):
                                    raise FileNotFoundError(
                                        f"No BC source found: neither {bvj_calib_input} nor {bvj_input_path}"
                                    )
                                print(f"    Refining inlet BC for forward simulation with input: {bvj_input_path}")
                                refine_inlet_bc_for_forward_simulation(nn_output_path, calibration_input_path=bvj_input_path)
                                run_forward_simulation(nn_output_path, nn_results_csv)
                                generated_files.append(nn_results_csv)
                                print(f"      ✓ NN-modified {geo_variant_name}/BloodVesselJunction simulation completed successfully")
                            except Exception as e:
                                raise Exception(f"NN-modified {geo_variant_name}/BloodVesselJunction simulation failed: {e}")
                    else:
                        print(f"      ⊘ Skipping: NN output file not found: {nn_output_path}")
                    # Forward sim for NN junction + vessel when --NN-vessel
                    if getattr(args, 'NN_vessel', False):
                        nn_jv_path = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel.json')
                        nn_jv_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel_results.csv')
                        if os.path.exists(nn_jv_path):
                            if check_and_track_file(nn_jv_results_csv, f"NN Junction+Vessel forward simulation for {geo_variant_name}"):
                                pass
                            else:
                                print(f"\n    Running simulation with NN Junction+Vessel {geo_variant_name} input...")
                                try:
                                    bvj_input_path = geo_variant_paths['junction_types']['BloodVesselJunction']['calibration_input']
                                    if not os.path.exists(bvj_input_path):
                                        bvj_input_path = geo_variant_paths['geometric_input']
                                    refine_inlet_bc_for_forward_simulation(nn_jv_path, calibration_input_path=bvj_input_path)
                                    run_forward_simulation(nn_jv_path, nn_jv_results_csv)
                                    generated_files.append(nn_jv_results_csv)
                                    print(f"      ✓ NN Junction+Vessel {geo_variant_name} simulation completed successfully")
                                except Exception as e:
                                    raise Exception(f"NN Junction+Vessel {geo_variant_name} simulation failed: {e}")
                            # Forward sim for NN_vessel (geometric junctions + NN vessels)
                            nn_vessel_only_path = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly.json')
                            nn_vessel_only_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly_results.csv')
                            if os.path.exists(nn_vessel_only_path):
                                if check_and_track_file(nn_vessel_only_results_csv, f"NN_vessel forward simulation for {geo_variant_name}"):
                                    pass
                                else:
                                    print(f"\n    Running simulation with NN_vessel {geo_variant_name} input...")
                                    try:
                                        bvj_input_path = geo_variant_paths['junction_types']['BloodVesselJunction']['calibration_input']
                                        if not os.path.exists(bvj_input_path):
                                            bvj_input_path = geo_variant_paths['geometric_input']
                                        refine_inlet_bc_for_forward_simulation(nn_vessel_only_path, calibration_input_path=bvj_input_path)
                                        run_forward_simulation(nn_vessel_only_path, nn_vessel_only_results_csv)
                                        generated_files.append(nn_vessel_only_results_csv)
                                        print(f"      ✓ NN_vessel {geo_variant_name} simulation completed successfully")
                                    except Exception as e:
                                        raise Exception(f"NN_vessel {geo_variant_name} simulation failed: {e}")
        else:
            # Normal mode: run all forward simulations for each geometry variant
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                print(f"\n  Running forward simulations for {geo_variant_name} geometry...")
                
                # Extract paths for this variant
                variant_geometric_input = geo_variant_paths['geometric_input']
                variant_geometric_results = geo_variant_paths['geometric_results']
                variant_calibration_input = geo_variant_paths['calibration_input']
                variant_junction_paths = geo_variant_paths['junction_types']
                
                # Run simulation with geometric input (uncalibrated).
                # Uses this variant's geometric input file (e.g. bifurcations_EL_geometric_input.json for bifurcations_EL).
                geometric_results_path = variant_geometric_results
                if check_and_track_file(geometric_results_path, f"geometric forward simulation for {geo_variant_name}"):
                    print(f"    ⊘ Skipping geometric simulation (results already exist)")
                else:
                    print(f"\n    Running simulation with {geo_variant_name} geometric input...")
                    if not os.path.exists(variant_geometric_input):
                        print(f"      ✗ Skipping: geometric input not found: {variant_geometric_input}")
                    else:
                        try:
                            # For geometric input, use the variant's calibration input as source
                            refine_inlet_bc_for_forward_simulation(variant_geometric_input, calibration_input_path=variant_calibration_input)
                            run_forward_simulation(variant_geometric_input, variant_geometric_results)
                            generated_files.append(variant_geometric_results)
                            print(f"      ✓ Geometric simulation completed successfully")
                        except Exception as e:
                            raise Exception(f"Geometric simulation failed: {e}")
                
                # Run simulation with each calibrated input
                for jtype in args.junction_types:
                    jtype_output_path = variant_junction_paths[jtype]['calibrated_output']
                    jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                    calibrated_results_csv = variant_junction_paths[jtype]['calibrated_results']
                    
                    # Check if calibrated output exists (calibration might have failed)
                    if not os.path.exists(jtype_output_path):
                        print(f"      ✗ Skipping: calibrated output not found: {jtype_output_path}")
                        continue
                    
                    if check_and_track_file(calibrated_results_csv, f"calibrated forward simulation for {geo_variant_name}/{jtype}"):
                        continue
                    
                    print(f"\n    Running simulation with calibrated {geo_variant_name}/{jtype} input...")
                    
                    try:
                        # Read BC from calibration input, write refined BC to calibrated output
                        refine_inlet_bc_for_forward_simulation(jtype_output_path, calibration_input_path=jtype_input_path)
                        run_forward_simulation(jtype_output_path, calibrated_results_csv)
                        generated_files.append(calibrated_results_csv)
                        print(f"      ✓ Calibrated {geo_variant_name}/{jtype} simulation completed successfully")
                    except Exception as e:
                        raise Exception(f"Calibrated {geo_variant_name}/{jtype} simulation failed: {e}")
                
                # Run forward simulation for NN-modified BloodVesselJunction on bifurcations and bifurcations_EL
                if geo_variant_name in ['bifurcations', 'bifurcations_EL'] and 'BloodVesselJunction' in args.junction_types:
                    nn_output_path = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction.json')
                    nn_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction_results.csv')
                    
                    if os.path.exists(nn_output_path):
                        if check_and_track_file(nn_results_csv, f"NN forward simulation for {geo_variant_name}/BloodVesselJunction"):
                            pass
                        else:
                            print(f"\n    Running simulation with NN-modified {geo_variant_name}/BloodVesselJunction input...")
                            try:
                                # Use BloodVesselJunction calibration input as source
                                bvj_input_path = variant_junction_paths['BloodVesselJunction']['calibration_input']
                                refine_inlet_bc_for_forward_simulation(nn_output_path, calibration_input_path=bvj_input_path)
                                run_forward_simulation(nn_output_path, nn_results_csv)
                                generated_files.append(nn_results_csv)
                                print(f"      ✓ NN-modified {geo_variant_name}/BloodVesselJunction simulation completed successfully")
                            except Exception as e:
                                raise Exception(f"NN-modified {geo_variant_name}/BloodVesselJunction simulation failed: {e}")
                    # Forward sim for NN Junction+Vessel when --NN-vessel (normal mode)
                    if getattr(args, 'NN_vessel', False):
                        nn_jv_path = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel.json')
                        nn_jv_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel_results.csv')
                        if os.path.exists(nn_jv_path):
                            if check_and_track_file(nn_jv_results_csv, f"NN Junction+Vessel forward simulation for {geo_variant_name}"):
                                pass
                            else:
                                print(f"\n    Running simulation with NN Junction+Vessel {geo_variant_name} input...")
                                try:
                                    bvj_input_path = variant_junction_paths['BloodVesselJunction']['calibration_input']
                                    if not os.path.exists(bvj_input_path):
                                        bvj_input_path = variant_geometric_input
                                    refine_inlet_bc_for_forward_simulation(nn_jv_path, calibration_input_path=bvj_input_path)
                                    run_forward_simulation(nn_jv_path, nn_jv_results_csv)
                                    generated_files.append(nn_jv_results_csv)
                                    print(f"      ✓ NN Junction+Vessel {geo_variant_name} simulation completed successfully")
                                except Exception as e:
                                    raise Exception(f"NN Junction+Vessel {geo_variant_name} simulation failed: {e}")
                            # Forward sim for NN_vessel (geometric junctions + NN vessels) in normal mode
                            nn_vessel_only_path = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly.json')
                            nn_vessel_only_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly_results.csv')
                            if os.path.exists(nn_vessel_only_path):
                                if check_and_track_file(nn_vessel_only_results_csv, f"NN_vessel forward simulation for {geo_variant_name}"):
                                    pass
                                else:
                                    print(f"\n    Running simulation with NN_vessel {geo_variant_name} input...")
                                    try:
                                        bvj_input_path = variant_junction_paths['BloodVesselJunction']['calibration_input']
                                        if not os.path.exists(bvj_input_path):
                                            bvj_input_path = variant_geometric_input
                                        refine_inlet_bc_for_forward_simulation(nn_vessel_only_path, calibration_input_path=bvj_input_path)
                                        run_forward_simulation(nn_vessel_only_path, nn_vessel_only_results_csv)
                                        generated_files.append(nn_vessel_only_results_csv)
                                        print(f"      ✓ NN_vessel {geo_variant_name} simulation completed successfully")
                                    except Exception as e:
                                        raise Exception(f"NN_vessel {geo_variant_name} simulation failed: {e}")
    
    # Step 5: Calculate and print MSE between 3D and 0D solutions
    if not args.skip_mse_calculation:
        print("\n" + "="*60)
        print("Step 5: Calculating MSE between 3D and 0D solutions")
        print("="*60)
        
        # Calculate MSE only for bifurcations geometry variant
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            # Skip original geometry variant
            if geo_variant_name == 'original':
                continue
                
            print(f"\n  MSE calculation for {geo_variant_name.upper()} geometry:")
            
            variant_calibration_input = geo_variant_paths['calibration_input']
            variant_geometric_results = geo_variant_paths['geometric_results']
            variant_geometric_input = geo_variant_paths['geometric_input']
            variant_junction_paths = geo_variant_paths['junction_types']
            
            # Build dictionary of CSV results for all modalities
            csv_results_dict = {}
            
            # Add geometric results
            if os.path.exists(variant_geometric_results):
                csv_results_dict['geometric'] = variant_geometric_results
            
            # Add calibrated results for each junction type
            for jtype in args.junction_types:
                calibrated_results_csv = variant_junction_paths[jtype]['calibrated_results']
                if os.path.exists(calibrated_results_csv):
                    csv_results_dict[jtype] = str(calibrated_results_csv)
            
            # Add NN-modified BloodVesselJunction results for bifurcations and bifurcations_EL
            if geo_variant_name in ['bifurcations', 'bifurcations_EL'] and 'BloodVesselJunction' in args.junction_types:
                nn_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction_results.csv')
                if os.path.exists(nn_results_csv):
                    csv_results_dict['BloodVesselJunction_NN'] = str(nn_results_csv)
                if getattr(args, 'NN_vessel', False):
                    nn_jv_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel_results.csv')
                    if os.path.exists(nn_jv_results_csv):
                        csv_results_dict['BloodVesselJunction_NN_plus_Vessel_NN'] = str(nn_jv_results_csv)
                    nn_vessel_only_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly_results.csv')
                    if os.path.exists(nn_vessel_only_results_csv):
                        csv_results_dict['NN_vessel'] = str(nn_vessel_only_results_csv)
            
            if csv_results_dict and os.path.exists(variant_calibration_input):
                # Generate CSV output path
                prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                mse_csv_path = os.path.join(base_dir, f'{prefix}mse_comparison.csv')
                
                if check_and_track_file(mse_csv_path, f"MSE calculation for {geo_variant_name}"):
                    pass
                else:
                    try:
                        calculate_mse_between_3d_and_0d(
                            variant_calibration_input,
                            csv_results_dict,
                            geometric_input_path=variant_geometric_input,
                            zoom_start_idx=args.zoom_start,
                            zoom_end_idx=args.zoom_end,
                            output_csv_path=mse_csv_path,
                            verbose=verbose,
                            set_name=args.set_name
                        )
                        generated_files.append(mse_csv_path)
                    except Exception as e:
                        raise Exception(f"Error calculating MSE for {geo_variant_name}: {e}")
            else:
                print(f"    Skipping MSE calculation for {geo_variant_name} (missing files)")
    
    #Step 6: Generate comparison plots (always run if not skipped, including in plot-only mode)
    if not args.skip_plots:
        print("\n" + "="*60)
        print("Step 6: Generating comparison plots")
        print("="*60)
        trial_suffix = f"_trial_{args.trial_id}" if args.trial_id is not None else ""
        plot_config = run_config_suffix or 'base'
        
        # Specify which plot types to generate: 'original', 'bifurcations', 'combined'
        plot_types = ["bifurcations","bifurcations_EL"]
        
        try:
            # Import plotting functions from unified location comparison script
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visualizations'))
            from plot_location_comparison import (
                plot_location_comparison, 
                get_all_locations_from_calibration_input,
                get_time_period as get_time_period_func
            )
            
            # Get time period (for plotting)
            time_period = None
            try:
                time_period = get_time_period_func(args.set_name, args.geo_name)
            except Exception:
                pass
            
            # Generate plots for each requested plot type (outputs under config-specific subfolder)
            for plot_type in plot_types:
                    if plot_type == 'combined':
                        # Generate combined comparison plots (original vs bifurcations for each junction type)
                        print(f"\n  Creating combined geometry variant comparison plots...")
                        combined_output_dir = os.path.join('results', 'location_comparison', plot_config, args.set_name, args.geo_name, f'combined{trial_suffix}')
                        os.makedirs(combined_output_dir, exist_ok=True)
                        
                        # Use original geometry calibration input for location list (for combined plots)
                        original_calibration_input = geometry_variants['original']['calibration_input']
                        orig_geometric_results = geometry_variants['original']['geometric_results']
                        
                        if not os.path.exists(original_calibration_input) or not os.path.exists(orig_geometric_results):
                            print(f"    Skipping combined plots (missing original geometry files)")
                            continue
                        
                        all_locations = get_all_locations_from_calibration_input(str(original_calibration_input))
                        
                        # Build combined CSV paths dict: keys are "original_jtype" and "bifurcations_jtype"
                        combined_csv_paths = {}
                        # Build separate dictionaries for geometric and calibrated results
                        geometric_csv_paths = {}
                        for geo_variant_name, geo_variant_paths in geometry_variants.items():
                            variant_geometric_results = geo_variant_paths['geometric_results']
                            if os.path.exists(variant_geometric_results):
                                geometric_csv_paths[geo_variant_name] = str(variant_geometric_results)
                            
                            for jtype in args.junction_types:
                                calibrated_results_csv = geo_variant_paths['junction_types'][jtype]['calibrated_results']
                                if os.path.exists(calibrated_results_csv):
                                    combined_csv_paths[f'{geo_variant_name}_{jtype}'] = str(calibrated_results_csv)
                            
                            # Add NN-modified BloodVesselJunction for bifurcations and bifurcations_EL
                            if geo_variant_name in ['bifurcations', 'bifurcations_EL'] and 'BloodVesselJunction' in args.junction_types:
                                nn_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction_results.csv')
                                if os.path.exists(nn_results_csv):
                                    combined_csv_paths[f'{geo_variant_name}_BloodVesselJunction_NN'] = str(nn_results_csv)
                                # Plot NN vessel modalities whenever CSVs exist (same as plot_location_comparison CLI).
                                # --NN-vessel still controls whether earlier steps generate these files.
                                nn_jv_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel_results.csv')
                                if os.path.exists(nn_jv_results_csv):
                                    combined_csv_paths[f'{geo_variant_name}_BloodVesselJunction_NN_plus_Vessel_NN'] = str(nn_jv_results_csv)
                                nn_vessel_only_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly_results.csv')
                                if os.path.exists(nn_vessel_only_results_csv):
                                    combined_csv_paths[f'{geo_variant_name}_NN_vessel'] = str(nn_vessel_only_results_csv)
                        
                        if combined_csv_paths or geometric_csv_paths:
                            success_count = 0
                            for location in all_locations:
                                safe_location = location.replace(':', '_')
                                plot_path = os.path.join(combined_output_dir, f"{safe_location}{trial_suffix}_combined.png")
                                try:
                                    success = plot_location_comparison(
                                        str(original_calibration_input),
                                        str(orig_geometric_results),
                                        combined_csv_paths,
                                        location,
                                        plot_path,
                                        set_name=args.set_name,
                                        geo_name=args.geo_name,
                                        time_period=time_period,
                                        geometric_input_path=str(geometry_variants['original']['geometric_input']),
                                        verbose=False,
                                        geometric_csv_paths=geometric_csv_paths
                                    )
                                    if success:
                                        success_count += 1
                                except Exception as e:
                                    print(f"      ✗ Failed to create combined plot for {location}: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            print(f"    Created {success_count}/{len(all_locations)} combined comparison plots")
                            print(f"    Combined plot output directory: {combined_output_dir}")
                        else:
                            print(f"    Skipping combined plots (no CSV results found)")
                    
                    elif plot_type in ['original', 'bifurcations', 'bifurcations_EL']:
                        # Generate plots for individual geometry variant
                        geo_variant_name = plot_type
                        print(f"\n  Creating {geo_variant_name} geometry variant comparison plots...")
                        variant_output_dir = os.path.join('results', 'location_comparison', plot_config, args.set_name, args.geo_name, f'{geo_variant_name}{trial_suffix}')
                        os.makedirs(variant_output_dir, exist_ok=True)
                        
                        geo_variant_paths = geometry_variants[geo_variant_name]
                        variant_calibration_input = geo_variant_paths['calibration_input']
                        variant_geometric_results = geo_variant_paths['geometric_results']
                        variant_geometric_input = geo_variant_paths['geometric_input']
                        
                        if not os.path.exists(variant_calibration_input) or not os.path.exists(variant_geometric_results):
                            print(f"    Skipping {geo_variant_name} plots (missing files)")
                            continue
                        
                        # Extract locations from this variant's calibration input (all locations, not just INFLOW)
                        variant_locations = get_all_locations_from_calibration_input(str(variant_calibration_input))
                        # Filter to only INFLOW locations by default
                        variant_locations = [loc for loc in variant_locations if loc.startswith('INFLOW:')]
                        
                        if not variant_locations:
                            print(f"    Skipping {geo_variant_name} plots (no locations found in calibration input)")
                            continue
                        
                        # Build CSV paths for this variant
                        variant_csv_paths = {}
                        variant_geometric_csv_paths = {}
                        
                        if os.path.exists(variant_geometric_results):
                            variant_geometric_csv_paths[geo_variant_name] = str(variant_geometric_results)
                        
                        for jtype in args.junction_types:
                            calibrated_results_csv = geo_variant_paths['junction_types'][jtype]['calibrated_results']
                            if os.path.exists(calibrated_results_csv):
                                variant_csv_paths[jtype] = str(calibrated_results_csv)
                        
                        # Add NN-modified BloodVesselJunction for bifurcations and bifurcations_EL
                        if geo_variant_name in ['bifurcations', 'bifurcations_EL'] and 'BloodVesselJunction' in args.junction_types:
                            nn_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction_results.csv')
                            if os.path.exists(nn_results_csv):
                                variant_csv_paths['BloodVesselJunction_NN'] = str(nn_results_csv)
                            # Plot NN vessel modalities whenever CSVs exist (same as plot_location_comparison CLI).
                            nn_jv_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel_results.csv')
                            if os.path.exists(nn_jv_results_csv):
                                variant_csv_paths['BloodVesselJunction_NN_plus_Vessel_NN'] = str(nn_jv_results_csv)
                            nn_vessel_only_results_csv = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly_results.csv')
                            if os.path.exists(nn_vessel_only_results_csv):
                                variant_csv_paths['NN_vessel'] = str(nn_vessel_only_results_csv)
                        
                        if variant_csv_paths or variant_geometric_csv_paths:
                            # Build vessel name mapping for EL-adjusted geometry
                            vessel_name_mapping = None
                            if geo_variant_name == 'bifurcations_EL':
                                from util.visualizations.plot_location_comparison import build_vessel_name_mapping
                                bifurcations_input = geometry_variants['bifurcations']['geometric_input']
                                if os.path.exists(bifurcations_input) and os.path.exists(variant_geometric_input):
                                    vessel_name_mapping = build_vessel_name_mapping(
                                        bifurcations_input, variant_geometric_input
                                    )
                            
                            success_count = 0
                            for location in variant_locations:
                                safe_location = location.replace(':', '_')
                                plot_path = os.path.join(variant_output_dir, f"{safe_location}{trial_suffix}_comparison.png")
                                try:
                                    success = plot_location_comparison(
                                        str(variant_calibration_input),
                                        str(variant_geometric_results),
                                        variant_csv_paths,
                                        location,
                                        plot_path,
                                        set_name=args.set_name,
                                        geo_name=args.geo_name,
                                        time_period=time_period,
                                        geometric_input_path=str(variant_geometric_input),
                                        zoom_start_idx=args.zoom_start,
                                        zoom_end_idx=args.zoom_end,
                                        verbose=False,
                                        geometric_csv_paths=variant_geometric_csv_paths,
                                        vessel_name_mapping=vessel_name_mapping
                                    )
                                    if success:
                                        success_count += 1
                                except Exception as e:
                                    print(f"      ✗ Failed to create {geo_variant_name} plot for {location}: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            print(f"    Created {success_count}/{len(variant_locations)} {geo_variant_name} comparison plots")
                            print(f"    {geo_variant_name.capitalize()} plot output directory: {variant_output_dir}")
                        else:
                            print(f"    Skipping {geo_variant_name} plots (no CSV results found)")
                
        except ImportError as e:
            print(f"  ✗ Could not import plotting functions: {e}")
            print("  Make sure plot_location_comparison.py is available")
        except Exception as e:
            print(f"  ✗ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot junction pressure differences for each geometry variant (off by default)
    # By default: skip original, only plot first 5 junctions for bifurcations
    if not args.skip_plots and args.plot_junction_pressure_diff:
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            # Skip original geometry variant
            if geo_variant_name == 'original':
                continue
                
            variant_calibration_input = geo_variant_paths['calibration_input']
            variant_geometric_input = geo_variant_paths['geometric_input']
            
            if os.path.exists(variant_calibration_input) and os.path.exists(variant_geometric_input):
                try:
                    print(f"\n  Creating junction pressure difference plots for {geo_variant_name}...")
                    prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                    trial_suffix_plots = f"_trial_{args.trial_id}" if args.trial_id is not None else ""
                    output_subdir = os.path.join(base_dir, f'{prefix}junction_pressure_diff{trial_suffix_plots}') if prefix else base_dir
                    
                    # Limit to first 5 junctions for bifurcations
                    max_junctions = 5 if geo_variant_name == 'bifurcations' else None
                    
                    plot_junction_pressure_differences(
                        str(variant_calibration_input),
                        str(variant_geometric_input),
                        output_subdir,
                        zoom_start_idx=args.zoom_start,
                        zoom_end_idx=args.zoom_end,
                        set_name=args.set_name,
                        geo_name=args.geo_name,
                        verbose=verbose,
                        max_junctions=max_junctions
                    )
                except Exception as e:
                    raise Exception(f"Error generating junction pressure difference plots for {geo_variant_name}: {e}")

    # Plot zero-D parameter bar charts (R, stenosis, L) comparing modalities
    if not args.skip_plots:
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            try:
                print(f"\n  Creating zero-D parameter bar charts for {geo_variant_name}...")
                prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                trial_suffix_param = f"_trial_{args.trial_id}" if args.trial_id is not None else ""
                # Save parameter comparison plots under results/param_comparison/<run_config>/<set>/<geo>/<variant>
                output_subdir = os.path.join('results', 'param_comparison', plot_config, args.set_name, args.geo_name, f'{geo_variant_name}{trial_suffix_param}')
                os.makedirs(output_subdir, exist_ok=True)

                # Build modality -> calibrated JSON path mapping
                modality_jsons = {}
                # geometric input
                geom_json = geo_variant_paths.get('geometric_input')
                if geom_json and os.path.exists(geom_json):
                    modality_jsons['geometric'] = str(geom_json)

                # calibrated outputs for junction types
                for jtype in args.junction_types:
                    jpath = geo_variant_paths['junction_types'].get(jtype, {}).get('calibrated_output')
                    if jpath and os.path.exists(jpath):
                        modality_jsons[jtype] = str(jpath)
                
                # Add NN-modified BloodVesselJunction for bifurcations and bifurcations_EL
                if geo_variant_name in ['bifurcations', 'bifurcations_EL'] and 'BloodVesselJunction' in args.junction_types:
                    nn_json = os.path.join(base_dir, f'{geo_variant_name}_NN_BloodVesselJunction.json')
                    if os.path.exists(nn_json):
                        modality_jsons['BloodVesselJunction_NN'] = str(nn_json)
                    if getattr(args, 'NN_vessel', False):
                        nn_jv_json = os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel.json')
                        if os.path.exists(nn_jv_json):
                            modality_jsons['BloodVesselJunction_NN_plus_Vessel_NN'] = str(nn_jv_json)
                        nn_vessel_only_json = os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly.json')
                        if os.path.exists(nn_vessel_only_json):
                            modality_jsons['NN_vessel'] = str(nn_vessel_only_json)

                if not modality_jsons:
                    print(f"    Warning: No modality JSONs found for {geo_variant_name}, skipping zero-D parameter bar chart")
                    continue

                out_name = f"{prefix}trial_{args.trial_id} zero_d_parameter_bars.png" if args.trial_id is not None else f"{prefix} zero_d_parameter_bars.png"
                out_path = plot_zero_d_parameter_bars(modality_jsons, output_dir=output_subdir, output_name=out_name, verbose=verbose)
                if out_path:
                    print(f"    ✓ Saved zero-D parameter bar chart: {out_path}")
                else:
                    print(f"    ✗ Failed to create zero-D parameter bar chart for {geo_variant_name}")
            except Exception as e:
                print(f"    ✗ Error creating zero-D parameter bar chart for {geo_variant_name}: {e}")
    
    if verbose:
        print("\n" + "="*60)
        print("Done!")
        print("="*60)
        
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            print(f"\n{geo_variant_name.upper()} geometry:")
            print(f"  Geometric input: {geo_variant_paths['geometric_input']}")
            print(f"  Geometric simulation results: {geo_variant_paths['geometric_results']}")
            
            if not args.skip_calibration:
                print(f"  Base calibration input: {geo_variant_paths['calibration_input']}")
                print(f"  Junction type variants:")
                for jtype in args.junction_types:
                    print(f"    {jtype}:")
                    print(f"      Calibration input: {geo_variant_paths['junction_types'][jtype]['calibration_input']}")
                    print(f"      Calibrated output: {geo_variant_paths['junction_types'][jtype]['calibrated_output']}")
                    print(f"      Simulation results: {geo_variant_paths['junction_types'][jtype]['calibrated_results']}")
    
    # Output generated files as JSON if --no-redo was used (for batch script to parse)
    if args.no_redo:
        print(json.dumps({"generated_files": generated_files}))


if __name__ == "__main__":
    main()

# python3 util/zerod_calibration/generate_zerod_inputs.py --set-name set_3 --geo-name tree_007
