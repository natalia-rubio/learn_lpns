#!/usr/bin/env python3
"""
Generate svZeroDSolver input files from centerline geometry and simulation results.
This script creates:
1. Geometric 0D input files (from centerline geometry)
2. Calibration input files (from 3D or 1D results)
3. Calibrated output files (by running svZeroDCalibrator)

Based on the workflow in richter2024-paper-tools.
"""

import os
import sys
sys.path.append("/Users/natalia/cursor_access/learn_lpns")
import json
import vtk
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict, OrderedDict
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



def main():
    parser = argparse.ArgumentParser(
        description="Generate svZeroDSolver input files and run calibration"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_1)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_000)')

    parser.add_argument('--junction-types', nargs='+', 
                       default=['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to generate calibration files for (default: all four types)')
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
    # Construct paths
    output_dir = 'data/zeroD'
    base_dir = os.path.join(output_dir, args.set_name, args.geo_name)
    
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
            
        # Generate bifurcations-only version of the geometric input
        print(f"\n  Creating bifurcations-only geometric input...")
        split_junctions_from_files(geometric_input_path, centerline_path, bifurcations_geometric_input_path)
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
            
            # if this is a VMR set type, try to find the centerline in the VMR oneD directory
            

            if not os.path.exists(soln_path):
                if 'VMR' in args.set_name:
                    oneD_dir = os.path.join('data', 'oneD', "VMR", args.geo_name)
                    alt_soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
                    if os.path.exists(alt_soln_path):
                        soln_path = alt_soln_path
            
                else:
                    # Try scratch directory location
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

                # Use the path that was found earlier
                oneD_dir = os.path.join('data', 'oneD', args.set_name, args.geo_name)
                soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
                if not os.path.exists(soln_path):
                    reduced_results_dir = os.path.join('data', 'reduced_results', args.set_name, args.geo_name)
                    soln_path = os.path.join(reduced_results_dir, 'unsteady_soln.vtp')
                if not os.path.exists(soln_path):
                    alt_soln_path = os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                                               args.set_name, args.geo_name, 'unsteady_soln.vtp')
                    if os.path.exists(alt_soln_path):
                        soln_path = alt_soln_path
            
            # Get geo_dir
            geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)

            skip_outlet_bc_fitting = 'coro' in args.set_name.lower()

            time_step_size = None
            if not skip_outlet_bc_fitting:
                try:
                    time_step_size = timestep_from_1D(soln_path, geo_dir)
                except Exception:
                    time_step_size = None
            
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

                        create_calibration_input(
                            variant_geometric_input, augmented_observations, variant_calibration_input,
                            centerline_soln_path=soln_path, geo_dir=geo_dir
                        )
                    else:
                        # Original geometry: use original observations
                        create_calibration_input(
                            variant_geometric_input, observations, variant_calibration_input,
                            centerline_soln_path=soln_path, geo_dir=geo_dir
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
                run_data_processing_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), '..', 'data_processing', 'run_data_processing.py'),
                    '--set-name', args.set_name,
                    '--geometry-variant', geo_variant_name,
                    '--set-type', 'test',
                    '--geometries', geo_name_for_ml,
                    '--output-type', 'rri',
                    '--percent-train', '1',
                    '--seed', '0',
                    '--data-root', 'data',
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
            if geo_variant_name == 'bifurcations_EL':
                bifurcations_input = geometry_variants['bifurcations']['geometric_input']
                if not os.path.exists(bifurcations_input):
                    print(f"  ⊘ Skipping NN inference for {geo_variant_name} (bifurcations geometric input not found)")
                    continue
            
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
                    from util.data_processing.inputs_from_0d_config import load_junction_geometric_features
                    from util.neural_network.nn_model import predict
                    from util.neural_network.nn_util import dill_load
                    import jax.numpy as jnp
                    
                    # Load the calibrated BloodVesselJunction config
                    bvj_output_path = variant_junction_paths['BloodVesselJunction']['calibrated_output']
                    if not os.path.exists(bvj_output_path):
                        raise FileNotFoundError(f"Calibrated output not found: {bvj_output_path}")
                    
                    with open(bvj_output_path, 'r') as f:
                        nn_config = json.load(f)
                    
                    # Extract geometric features using the same workflow as data processing
                    # This ensures we use the same 13 features that the model was trained on
                    from util.data_processing.data_dict_from_csvs import (
                        _read_csv_matrix,
                        filter_features_from_array,
                    )
                    
                    # Check if CSV file exists (from data processing)
                    csv_path = os.path.join('data', 'ml_inputs', args.set_name, geo_variant_name, args.geo_name, 'geometric_features.csv')
                    
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
                    #     # CSV doesn't exist, extract features directly and filter
                    print(f"  CSV not found, extracting features directly from geometric input")
                    X_full, feature_names_full, junction_names, _ = load_junction_geometric_features(
                        variant_geometric_input,
                        require_two_outlets=True,
                        verbose=True
                    )
                    #save X_full to a csv file
                    import pandas as pd
                    pd.DataFrame(X_full, columns=feature_names_full).to_csv(csv_path, index=False)
                    
                    # Apply the same feature selection using the reusable function
                    X, feature_names = filter_features_from_array(
                        X_full, feature_names_full, remap_tortuosity=True
                    )
                    
                    if len(X) == 0:
                        raise ValueError("No junctions found in geometric features")
                    
                    # Count unique junction names (some may have been skipped for swapped row)
                    unique_junction_names = list(set(junction_names))
                    print(f"  Loaded {len(X)} feature rows for {len(unique_junction_names)} unique junctions")
                    print(f"  Junction names in feature extraction: {unique_junction_names}")
                    print(f"  Selected {len(feature_names)} features (matching training data): {feature_names}")
                    
                    # Convert to JAX array
                    X_jax = jnp.array(X, dtype=jnp.float32)
                    print(f"  Neural network input dimensions: {X_jax.shape} (rows={X_jax.shape[0]}, features={X_jax.shape[1]})")
                
                    # Load the three trained models (use geometry variant for model path)
                    model_dir = os.path.join('results', 'models', args.set_name, geo_variant_name)
                    model_base_name = f"rri_{args.set_name}_pred"
                    model_paths = [
                        os.path.join(model_dir, f"{model_base_name}_0_model"),
                        os.path.join(model_dir, f"{model_base_name}_1_model"),
                        os.path.join(model_dir, f"{model_base_name}_2_model"),
                    ]
                    
                    for i, model_path in enumerate(model_paths):
                        if not os.path.exists(model_path):
                            raise FileNotFoundError(f"Model not found: {model_path}")
                    
                    # Load models and get predictions
                    predictions = []
                    for i, model_path in enumerate(model_paths):
                        print(f"      Loading model {i+1}/3: {model_path}")
                        model = dill_load(model_path)
                        use_leaky = getattr(model, "use_leaky_relu", False)
                        pred = predict(X_jax, model.weights, use_leaky)
                        predictions.append(np.array(pred).flatten())
                
                    # Based on outputs_from_config.py and launch_training.py:
                    # Output columns: 0=R_outlet0, 1=R_outlet1, 2=S_outlet0, 3=S_outlet1, 4=L_outlet0, 5=L_outlet1
                    # Model 0 (target_coef_ind=0): predicts R_poiseuille_outlet0
                    # Model 1 (target_coef_ind=1): predicts R_poiseuille_outlet1  
                    # Model 2 (target_coef_ind=2): predicts stenosis_coefficient_outlet0
                    # But user said: pred_0=R, pred_1=stenosis, pred_2=L
                    # Looking at launch_training.py comments:
                    #   "training model 1: Linear Resistor" (target_coef_ind=0) -> R
                    #   "training model 2: Stenosis Resistor" (target_coef_ind=1) -> S  
                    #   "training model 3: Inductor" (target_coef_ind=2) -> L
                    # So models predict: R (outlet0), S (outlet1?), L (outlet0?)
                    # Actually, each model predicts one value per row. With two rows per junction:
                    # - Row 0 (outlet0-first): model predicts for outlet0
                    # - Row 1 (outlet1-first): model predicts for outlet1
                    # So we can get both outlets from the two rows.
                    
                    # predictions[0] = R_poiseuille (one value per row)
                    # predictions[1] = stenosis_coefficient (one value per row)
                    # predictions[2] = L (one value per row)
                    pred_R = predictions[0]
                    pred_S = predictions[1]
                    pred_L = predictions[2]
                    import pdb; pdb.set_trace()
                     # Verify prediction array sizes match input
                     # Note: junction_names may have duplicates (same junction appears twice for swapped rows)
                     # So we compare against the actual number of input rows
                    if len(pred_R) != len(X):
                        raise ValueError(
                            f"Prediction array size mismatch: input has {len(X)} rows, "
                            f"but predictions have {len(pred_R)} values. "
                            f"Expected one prediction per input row."
                        )
                    
                    # Build a mapping from junction name to all row indices where it appears
                    # (a junction can appear 1 or 2 times depending on whether swapped row was skipped)
                    junction_name_to_row_indices = {}
                    for row_idx, junc_name in enumerate(junction_names):
                        if junc_name not in junction_name_to_row_indices:
                            junction_name_to_row_indices[junc_name] = []
                        junction_name_to_row_indices[junc_name].append(row_idx)
                
                    # Map predictions back to junction_values
                    vessels = nn_config.get('vessels', [])
                    vessel_id_to_name = {v.get('vessel_id'): v.get('vessel_name', '') for v in vessels}
                    
                    # Process each junction in the config
                    for junc in nn_config.get('junctions', []):
                        junc_name = junc.get('junction_name', '')
                        
                        # Skip if this junction wasn't in the feature extraction (e.g., connector primary outlet)
                        if junc_name not in junction_name_to_row_indices:
                            # For skipped junctions, we still need to initialize junction_values if they're BloodVesselJunction
                            # But we'll leave them as-is from the calibrated output (or set to zeros if missing)
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
                        
                        # Get the row indices for this junction (can be 1 or 2 rows)
                        row_indices = junction_name_to_row_indices[junc_name]
                        if len(row_indices) == 0:
                            raise ValueError(f"No row indices found for junction {junc_name}")
                        
                        # The first row is always outlet0-first (if it exists)
                        # The second row (if it exists) is outlet1-first
                        row_outlet0_first = row_indices[0]
                        row_outlet1_first = row_indices[1] if len(row_indices) > 1 else None
                        
                        # Verify row indices are within bounds
                        if row_outlet0_first >= len(pred_R):
                            raise ValueError(
                                f"Row index {row_outlet0_first} out of bounds for junction {junc_name} "
                                f"(array_size={len(pred_R)})"
                            )
                        if row_outlet1_first is not None and row_outlet1_first >= len(pred_R):
                            raise ValueError(
                                f"Row index {row_outlet1_first} out of bounds for junction {junc_name} "
                                f"(array_size={len(pred_R)})"
                            )
                        
                        # Get outlet vessels
                        outlet_vessel_ids = junc.get('outlet_vessels', [])
                        if len(outlet_vessel_ids) != 2:
                            continue
                        
                        outlet_vessel_names = [vessel_id_to_name.get(vid, '') for vid in outlet_vessel_ids]
                        
                        # Get geometric_params to determine outlet ordering
                        gp = junc.get('geometric_params', {})
                        outlet_path_lengths = gp.get('outlet_path_lengths', {})
                        
                        # Sort outlets by descending path length (matching inputs_from_0d_config.py)
                        outlet_names_sorted = sorted(
                            outlet_vessel_names,
                            key=lambda vn: float(outlet_path_lengths.get(vn, 0.0)),
                            reverse=True
                        )
                        
                        # Map sorted outlets to file order
                        outlet_index_in_file = {vn: i for i, vn in enumerate(outlet_vessel_names)}
                        outlet0_sorted = outlet_names_sorted[0]
                        outlet1_sorted = outlet_names_sorted[1]
                        
                        # Initialize junction_values if needed
                        if 'junction_values' not in junc:
                            junc['junction_values'] = {}
                        
                        # Set predictions for each outlet (in file order)
                        R_values = [0.0, 0.0]
                        S_values = [0.0, 0.0]
                        L_values = [0.0, 0.0]
                        
                        file_idx_outlet0 = outlet_index_in_file[outlet0_sorted]
                        file_idx_outlet1 = outlet_index_in_file[outlet1_sorted]
                        
                         # Row outlet0-first: predictions are for outlet0_sorted
                         # Row outlet1-first (if exists): predictions are for outlet1_sorted
                        if 'connector' in outlet0_sorted and "connectorEL" not in outlet0_sorted:
                            R_values[file_idx_outlet0] = 0.0
                            S_values[file_idx_outlet0] = 0.0
                            L_values[file_idx_outlet0] = 0.0
                        else:
                            R_values[file_idx_outlet0] = float(pred_R[row_outlet0_first])
                            S_values[file_idx_outlet0] = float(pred_S[row_outlet0_first])
                            L_values[file_idx_outlet0] = float(pred_L[row_outlet0_first])
                        
                        if 'connector' in outlet1_sorted and "connectorEL" not in outlet1_sorted:
                            R_values[file_idx_outlet1] = 0.0
                            S_values[file_idx_outlet1] = 0.0
                            L_values[file_idx_outlet1] = 0.0
                        elif row_outlet1_first is not None:
                            # Use prediction from outlet1-first row
                            R_values[file_idx_outlet1] = float(pred_R[row_outlet1_first])
                            S_values[file_idx_outlet1] = float(pred_S[row_outlet1_first])
                            L_values[file_idx_outlet1] = float(pred_L[row_outlet1_first])
                        else:
                            # No swapped row available (outlet1 is connector), use outlet0 prediction
                            # This shouldn't happen if outlet1 is not a connector, but handle it anyway
                            R_values[file_idx_outlet1] = float(pred_R[row_outlet0_first])
                            S_values[file_idx_outlet1] = float(pred_S[row_outlet0_first])
                            L_values[file_idx_outlet1] = float(pred_L[row_outlet0_first])
                    
                        junc['junction_values']['R_poiseuille'] = R_values
                        junc['junction_values']['stenosis_coefficient'] = S_values
                        junc['junction_values']['L'] = L_values
                    
                    # Save the NN-modified config (already checked at start of block)
                    with open(nn_output_path, 'w') as f:
                        json.dump(nn_config, f, indent=4)
                    generated_files.append(nn_output_path)
                    print(f"      ✓ Neural network predictions applied and saved to {nn_output_path}")
                
                except Exception as e:
                    raise Exception(f"Neural network inference failed for {geo_variant_name}/BloodVesselJunction: {e}")

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
                                # In NN-only mode, use geometric input as source for BC refinement
                                bvj_input_path = geo_variant_paths['geometric_input']
                                if not os.path.exists(bvj_input_path):
                                    # Fallback to calibration input if it exists
                                    bvj_input_path = variant_junction_paths['BloodVesselJunction'].get('calibration_input', '')
                                refine_inlet_bc_for_forward_simulation(nn_output_path, calibration_input_path=bvj_input_path)
                                run_forward_simulation(nn_output_path, nn_results_csv)
                                generated_files.append(nn_results_csv)
                                print(f"      ✓ NN-modified {geo_variant_name}/BloodVesselJunction simulation completed successfully")
                            except Exception as e:
                                raise Exception(f"NN-modified {geo_variant_name}/BloodVesselJunction simulation failed: {e}")
                    else:
                        print(f"      ⊘ Skipping: NN output file not found: {nn_output_path}")
        else:
            # Normal mode: run all forward simulations for each geometry variant
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                    print(f"\n  Running forward simulations for {geo_variant_name} geometry...")
                
                # Extract paths for this variant
                variant_geometric_input = geo_variant_paths['geometric_input']
                variant_geometric_results = geo_variant_paths['geometric_results']
                variant_calibration_input = geo_variant_paths['calibration_input']
                variant_junction_paths = geo_variant_paths['junction_types']
                    
                    # Run simulation with geometric input (uncalibrated)
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
            
            # Generate plots for each requested plot type
            for plot_type in plot_types:
                    if plot_type == 'combined':
            # Generate combined comparison plots (original vs bifurcations for each junction type)
            print(f"\n  Creating combined geometry variant comparison plots...")
            combined_output_dir = os.path.join('results', 'location_comparison', args.set_name, args.geo_name, 'combined')
            os.makedirs(combined_output_dir, exist_ok=True)
            
                        # Use original geometry calibration input for location list (for combined plots)
            original_calibration_input = geometry_variants['original']['calibration_input']
            orig_geometric_results = geometry_variants['original']['geometric_results']
            
                        if not os.path.exists(original_calibration_input) or not os.path.exists(orig_geometric_results):
                            print(f"    Skipping combined plots (missing original geometry files)")
                            continue
                        
                all_locations = get_all_locations_from_calibration_input(str(original_calibration_input))
                        # Filter to only INFLOW locations by default
                        all_locations = [loc for loc in all_locations if loc.startswith('INFLOW:')]
                
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
                
                if combined_csv_paths or geometric_csv_paths:
                    success_count = 0
                    for location in all_locations:
                        safe_location = location.replace(':', '_')
                        plot_path = os.path.join(combined_output_dir, f"{safe_location}_combined.png")
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
                        variant_output_dir = os.path.join('results', 'location_comparison', args.set_name, args.geo_name, geo_variant_name)
                        os.makedirs(variant_output_dir, exist_ok=True)
                        
                        geo_variant_paths = geometry_variants[geo_variant_name]
                        variant_calibration_input = geo_variant_paths['calibration_input']
                        variant_geometric_results = geo_variant_paths['geometric_results']
                        variant_geometric_input = geo_variant_paths['geometric_input']
                        
                        if not os.path.exists(variant_calibration_input) or not os.path.exists(variant_geometric_results):
                            print(f"    Skipping {geo_variant_name} plots (missing files)")
                            continue
                        
                        # Extract locations from this variant's calibration input
                        variant_locations = get_all_locations_from_calibration_input(str(variant_calibration_input))
                        # Filter to only INFLOW locations by default
                        # variant_locations = [loc for loc in variant_locations if loc.startswith('INFLOW:')]
                        
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
                                plot_path = os.path.join(variant_output_dir, f"{safe_location}_comparison.png")
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
                    output_subdir = os.path.join(base_dir, f'{prefix}junction_pressure_diff') if prefix else base_dir
                    
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
                # Save parameter comparison plots under results/param_comparison/<set>/<geo>/<variant>
                output_subdir = os.path.join('results', 'param_comparison', args.set_name, args.geo_name, geo_variant_name)
                os.makedirs(output_subdir, exist_ok=True)
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

                if not modality_jsons:
                    print(f"    Warning: No modality JSONs found for {geo_variant_name}, skipping zero-D parameter bar chart")
                    continue

                out_name = f"{prefix}zero_d_parameter_bars.png"
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
