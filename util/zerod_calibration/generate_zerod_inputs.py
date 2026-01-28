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
    parser.add_argument('--zoom-start', type=int, default=599,
                       help='Start index for zoom window (shaded region in plots). Default: 599')
    parser.add_argument('--zoom-end', type=int, default=699,
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

    args = parser.parse_args(); verbose = args.verbose
    # Construct paths
    output_dir = 'data/zeroD'
    base_dir = os.path.join(output_dir, args.set_name, args.geo_name)
    
    geometry_variants, geometric_input_path, geometric_results_csv, calibration_input_path, calibrated_output_path, junction_type_paths, centerline_path, geo_dir = get_paths(base_dir, args)
    
    # Step 1: Create geometric 0D input using SimVascular ROM workflow
    if not args.skip_base_generation:
        
        if args.set_name == "VMR":
            richter_0d_path = os.path.join('data', 'zeroD', args.set_name, 'richter-0d', args.geo_name+'.json')
            zerod_input = load_from_json(richter_0d_path)
            zerod_input['simulation_parameters']['output_all_cycles'] = True
            zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
            os.makedirs(os.path.dirname(geometric_input_path), exist_ok=True)
            save_to_json(zerod_input, geometric_input_path)
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
        # Generate bifurcations-only version of the geometric input
        bifurcations_geometric_input_path = geometry_variants['bifurcations']['geometric_input']
        print(f"\n  Creating bifurcations-only geometric input...")
        split_junctions_from_files(geometric_input_path, centerline_path, bifurcations_geometric_input_path)
        print(f"  Bifurcations-only geometric input saved to: {bifurcations_geometric_input_path}")

    extract_and_add_geometric_params(centerline_path, bifurcations_geometric_input_path, bifurcations_geometric_input_path)
    print(f"  Geometric parameters extracted and added to {bifurcations_geometric_input_path}")

        
    # Step 2: Extract observations and create calibration inputs for each junction type
    if not args.skip_observation:
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

        # Determine timestep from 1D solution for BC fitting
        try:
            time_step_size = timestep_from_1D(soln_path, geo_dir)
        except Exception:
            time_step_size = None
        
        # Fit outlet resistances from 3D solution (skip for VMR cases)
        if args.set_name != "VMR":
            fitted_resistances = fit_outlet_resistances_from_3d(geometric_input_path, observations)
        else:
            fitted_resistances = None

        # Fit RCR boundary conditions from observations (1D-derived)
        fitted_rcr = {}
        if time_step_size is not None:
            try:
                fitted_rcr = fit_outlet_rcr_from_observations(geometric_input_path, observations, dt=time_step_size)
            except Exception as e:
                print(f"  Warning: RCR fitting failed on original geometry: {e}")
        else:
            print("  Warning: time_step_size unavailable; skipping RCR fitting on original geometry")
        
        
        
        # Process BOTH geometry variants: original and bifurcations-only
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
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
            print(f"\n  Creating base calibration input for {geo_variant_name}...")

            try:
                # For bifurcations geometry:
                # - Always rename existing observations to match bifurcated junction names
                # - Optionally add synthetic connector-vessel observations
                if geo_variant_name == 'bifurcations':
                    with open(variant_geometric_input, 'r') as f:
                        bifurcated_geometric_input = json.load(f)
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
                    print(f"    Creating {geo_variant_name}/{jtype} calibration input...")
                    jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                    
                    # Apply junction type modification to calibration input
                    jtype_config = modify_junction_types(base_calibration_config, jtype)
                    
                    with open(jtype_input_path, 'w') as f:
                        json.dump(jtype_config, f, indent=4)
            except Exception as e:
                raise Exception(f"Failed to create junction type calibration inputs for {geo_variant_name}: {e}")
            

    # Step 3: Run calibration for each junction type
    if not args.skip_calibration:
        print(f"\n  Running calibration for {geo_variant_name} geometry...")
        
        # Run calibration for each junction type variant
        for jtype in args.junction_types:
            print(f"\n    Calibrating {geo_variant_name}/{jtype}...")
            jtype_input_path = variant_junction_paths[jtype]['calibration_input']
            jtype_output_path = variant_junction_paths[jtype]['calibrated_output']
            
            try:
                calibrated_config = run_calibration(jtype_input_path, jtype_output_path)
                print(f"      ✓ Calibration completed for {geo_variant_name}/{jtype}")
                # Junction types are now preserved by the calibrator
            except Exception as e:
                raise Exception(f"Calibration failed for {geo_variant_name}/{jtype}: {e}")

    # Step 4: Run forward simulations for this geometry variant
    if not args.skip_forward:
        
        print(f"\n  Running forward simulations for {geo_variant_name} geometry...")
        
        # Run simulation with geometric input (uncalibrated)
        print(f"\n    Running simulation with {geo_variant_name} geometric input...")
        if not os.path.exists(variant_geometric_input):
            print(f"      ✗ Skipping: geometric input not found: {variant_geometric_input}")
        else:
            try:
                refinement_factor = 4
                refine_inlet_bc_for_forward_simulation(variant_geometric_input, refinement_factor)
                run_forward_simulation(variant_geometric_input, variant_geometric_results)
                print(f"      ✓ Geometric simulation completed successfully")
            except Exception as e:
                raise Exception(f"Geometric simulation failed: {e}")
        
        # Run simulation with each calibrated input
        for jtype in args.junction_types:
            print(f"\n    Running simulation with calibrated {geo_variant_name}/{jtype} input...")
            jtype_output_path = variant_junction_paths[jtype]['calibrated_output']
            calibrated_results_csv = variant_junction_paths[jtype]['calibrated_results']
            
            # Check if calibrated output exists (calibration might have failed)
            if not os.path.exists(jtype_output_path):
                print(f"      ✗ Skipping: calibrated output not found: {jtype_output_path}")
                continue
            
            try:
                refine_inlet_bc_for_forward_simulation(jtype_output_path, refinement_factor)
                run_forward_simulation(jtype_output_path, calibrated_results_csv)
                print(f"      ✓ Calibrated {geo_variant_name}/{jtype} simulation completed successfully")
            except Exception as e:
                raise Exception(f"Calibrated {geo_variant_name}/{jtype} simulation failed: {e}")
    
    # Step 5: Calculate and print MSE between 3D and 0D solutions
    if not args.skip_mse_calculation:
        print("\n" + "="*60)
        print("Step 5: Calculating MSE between 3D and 0D solutions")
        print("="*60)
        
        # Calculate MSE for both geometry variants
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
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
            
            if csv_results_dict and os.path.exists(variant_calibration_input):
                try:
                    # Generate CSV output path
                    prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                    mse_csv_path = os.path.join(base_dir, f'{prefix}mse_comparison.csv')
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
                except Exception as e:
                    raise Exception(f"Error calculating MSE for {geo_variant_name}: {e}")
            else:
                print(f"    Skipping MSE calculation for {geo_variant_name} (missing files)")
    
    #Step 6: Generate comparison plots (always run if not skipped, including in plot-only mode)
    if not args.skip_plots:
        print("\n" + "="*60)
        print("Step 6: Generating comparison plots")
        print("="*60)
        
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
            

            # Generate combined comparison plots (original vs bifurcations for each junction type)
            print(f"\n  Creating combined geometry variant comparison plots...")
            combined_output_dir = os.path.join('results', 'location_comparison', args.set_name, args.geo_name, 'combined')
            os.makedirs(combined_output_dir, exist_ok=True)
            
            # Use original geometry calibration input for location list
            original_calibration_input = geometry_variants['original']['calibration_input']
            orig_geometric_results = geometry_variants['original']['geometric_results']
            
            if os.path.exists(original_calibration_input) and os.path.exists(orig_geometric_results):
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
                                zoom_start_idx=args.zoom_start,
                                zoom_end_idx=args.zoom_end,
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
                print(f"    Skipping combined plots (missing original geometry files)")
                
        except ImportError as e:
            print(f"  ✗ Could not import plotting functions: {e}")
            print("  Make sure plot_location_comparison.py is available")
        except Exception as e:
            print(f"  ✗ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot junction pressure differences for each geometry variant
    if not args.skip_plots:
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            variant_calibration_input = geo_variant_paths['calibration_input']
            variant_geometric_input = geo_variant_paths['geometric_input']
            
            if os.path.exists(variant_calibration_input) and os.path.exists(variant_geometric_input):
                try:
                    print(f"\n  Creating junction pressure difference plots for {geo_variant_name}...")
                    prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                    output_subdir = os.path.join(base_dir, f'{prefix}junction_pressure_diff') if prefix else base_dir
                    plot_junction_pressure_differences(
                        str(variant_calibration_input),
                        str(variant_geometric_input),
                        output_subdir,
                        zoom_start_idx=args.zoom_start,
                        zoom_end_idx=args.zoom_end,
                        set_name=args.set_name,
                        geo_name=args.geo_name,
                        verbose=verbose
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


if __name__ == "__main__":
    main()

# python3 util/zerod_calibration/generate_zerod_inputs.py --set-name set_3 --geo-name tree_007
