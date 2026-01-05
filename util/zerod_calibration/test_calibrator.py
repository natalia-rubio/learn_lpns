#!/usr/bin/env python3
"""
Test script for calibrator using svZeroDPlus test cases.
Generates calibration inputs for different junction types, runs calibration,
and creates comparison plots.
"""

import os
import sys
import json
import csv
import numpy as np
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path to import from generate_zerod_inputs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_zerod_inputs import (
    create_calibration_input,
    run_calibration,
    run_forward_simulation,
    modify_junction_types,
    refine_inlet_bc_for_forward_simulation
)

# Import plotting functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations'))
from plot_inlet_comparison import (
    read_zerod_csv,
    extract_inlet_data_from_csv,
    plot_inlet_comparison
)
from plot_outlet_comparison import (
    extract_outlet_data_from_csv,
    plot_outlet_comparison,
    find_all_vessels_with_outlets
)


def extract_observations_from_csv(csv_path, geometric_input_path):
    """
    Extract observation data from CSV results file (ground truth).
    Formats observations in the same way as extract_observations_from_1d.
    
    Args:
        csv_path: Path to CSV results file
        geometric_input_path: Path to geometric 0D input JSON (to understand vessel/junction structure)
        
    Returns:
        Dictionary with observation data (y, dy) for calibration
    """
    print(f"Reading ground truth CSV from: {csv_path}")
    
    # Read geometric input to understand vessel/junction structure
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    boundary_conditions = geometric_input.get('boundary_conditions', [])
    
    # Create vessel name to index mapping
    vessel_name_to_idx = {v['vessel_name']: i for i, v in enumerate(vessels)}
    
    # Create BC name to type mapping
    bc_name_to_type = {}
    for bc in boundary_conditions:
        bc_name_to_type[bc['bc_name']] = bc['bc_type']
    
    # Read CSV data
    results, times = read_zerod_csv(csv_path)
    
    if not times:
        raise ValueError("No time data found in CSV file")
    
    # Sort times
    times = sorted(times)
    num_obs = len(times)
    
    print(f"  Found {num_obs} time points")
    print(f"  Time range: [{times[0]:.6f}, {times[-1]:.6f}] s")
    print(f"  Found {len(results)} vessels in CSV")
    
    # Initialize observation dictionaries
    observations = {"y": {}, "dy": {}}
    
    # Extract inlet observations
    for vessel in vessels:
        vessel_name = vessel['vessel_name']
        if vessel_name not in results:
            continue
        
        # Check for inlet BC
        if 'boundary_conditions' in vessel and 'inlet' in vessel['boundary_conditions']:
            bc_name = vessel['boundary_conditions']['inlet']
            
            # Extract inlet pressure and flow
            pressures = []
            flows = []
            for time in times:
                if time in results[vessel_name]:
                    data = results[vessel_name][time]
                    if 'pressure_in' in data:
                        pressures.append(data['pressure_in'])
                    if 'flow_in' in data:
                        flows.append(data['flow_in'])
            
            if pressures:
                key = f"pressure:{bc_name}:{vessel_name}"
                observations["y"][key] = pressures
                # Compute derivatives using np.gradient
                if len(pressures) > 1:
                    dt = times[1] - times[0] if len(times) > 1 else 1.0
                    observations["dy"][key] = np.gradient(pressures, dt).tolist()
            
            if flows:
                key = f"flow:{bc_name}:{vessel_name}"
                observations["y"][key] = flows
                # Compute derivatives
                if len(flows) > 1:
                    dt = times[1] - times[0] if len(times) > 1 else 1.0
                    observations["dy"][key] = np.gradient(flows, dt).tolist()
    
    # Extract outlet observations (for vessels with explicit outlet BCs)
    for vessel in vessels:
        vessel_name = vessel['vessel_name']
        if vessel_name not in results:
            continue
        
        # Check for outlet BC
        if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
            bc_name = vessel['boundary_conditions']['outlet']
            
            # Extract outlet pressure and flow
            pressures = []
            flows = []
            for time in times:
                if time in results[vessel_name]:
                    data = results[vessel_name][time]
                    if 'pressure_out' in data:
                        pressures.append(data['pressure_out'])
                    if 'flow_out' in data:
                        flows.append(data['flow_out'])
            
            if pressures:
                key = f"pressure:{vessel_name}:{bc_name}"
                observations["y"][key] = pressures
                if len(pressures) > 1:
                    dt = times[1] - times[0] if len(times) > 1 else 1.0
                    observations["dy"][key] = np.gradient(pressures, dt).tolist()
            
            if flows:
                key = f"flow:{vessel_name}:{bc_name}"
                observations["y"][key] = flows
                if len(flows) > 1:
                    dt = times[1] - times[0] if len(times) > 1 else 1.0
                    observations["dy"][key] = np.gradient(flows, dt).tolist()
    
    # Extract junction observations
    # For inlet vessels: format is "pressure:vessel_name:junction_name"
    # For outlet vessels: format is "pressure:junction_name:vessel_name"
    for junction in junctions:
        junction_name = junction.get('junction_name', '')
        inlet_vessels = junction.get('inlet_vessels', [])
        outlet_vessels = junction.get('outlet_vessels', [])
        
        # For inlet vessels: extract outlet pressure/flow (at junction)
        for inlet_idx in inlet_vessels:
            if inlet_idx < len(vessels):
                vessel = vessels[inlet_idx]
                vessel_name = vessel['vessel_name']
                if vessel_name not in results:
                    continue
                
                # Extract outlet pressure/flow (at junction for inlet vessels)
                pressures = []
                flows = []
                for time in times:
                    if time in results[vessel_name]:
                        data = results[vessel_name][time]
                        if 'pressure_out' in data:
                            pressures.append(data['pressure_out'])
                        if 'flow_out' in data:
                            flows.append(data['flow_out'])
                
                if pressures:
                    key = f"pressure:{vessel_name}:{junction_name}"
                    observations["y"][key] = pressures
                    if len(pressures) > 1:
                        dt = times[1] - times[0] if len(times) > 1 else 1.0
                        observations["dy"][key] = np.gradient(pressures, dt).tolist()
                
                if flows:
                    key = f"flow:{vessel_name}:{junction_name}"
                    observations["y"][key] = flows
                    if len(flows) > 1:
                        dt = times[1] - times[0] if len(times) > 1 else 1.0
                        observations["dy"][key] = np.gradient(flows, dt).tolist()
        
        # For outlet vessels: extract inlet pressure/flow (at junction)
        for outlet_idx in outlet_vessels:
            if outlet_idx < len(vessels):
                vessel = vessels[outlet_idx]
                vessel_name = vessel['vessel_name']
                if vessel_name not in results:
                    continue
                
                # Extract inlet pressure/flow (at junction for outlet vessels)
                pressures = []
                flows = []
                for time in times:
                    if time in results[vessel_name]:
                        data = results[vessel_name][time]
                        if 'pressure_in' in data:
                            pressures.append(data['pressure_in'])
                        if 'flow_in' in data:
                            flows.append(data['flow_in'])
                
                if pressures:
                    key = f"pressure:{junction_name}:{vessel_name}"
                    observations["y"][key] = pressures
                    if len(pressures) > 1:
                        dt = times[1] - times[0] if len(times) > 1 else 1.0
                        observations["dy"][key] = np.gradient(pressures, dt).tolist()
                
                if flows:
                    key = f"flow:{junction_name}:{vessel_name}"
                    observations["y"][key] = flows
                    if len(flows) > 1:
                        dt = times[1] - times[0] if len(times) > 1 else 1.0
                        observations["dy"][key] = np.gradient(flows, dt).tolist()
    
    print(f"  Extracted {len(observations['y'])} observation series")
    if observations['y']:
        first_key = next(iter(observations['y'].keys()))
        print(f"    Example key: {first_key}")
        print(f"    Points per series: {len(observations['y'][first_key])}")
    
    return observations


def main():
    parser = argparse.ArgumentParser(description='Test calibrator with svZeroDPlus test cases')
    parser.add_argument('--test-case', type=str,
                       default='/Users/natalia/cursor_access/svZeroDPlus/tests/cases/sinusoidalFlow_dir_dep_junction.json',
                       help='Path to test case JSON file')
    parser.add_argument('--results-csv', type=str,
                       default='/Users/natalia/cursor_access/svZeroDPlus/tests/cases/results/sinusoidalFlow_dir_dep_junction.csv',
                       help='Path to ground truth results CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='test_calibration_output',
                       help='Output directory for calibration inputs, outputs, and plots')
    parser.add_argument('--junction-types', nargs='+',
                       default=['DirDepJunction', 'DirIndepJunction', 'BloodVesselJunction', 'HybridJunction'],
                       help='Junction types to test')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Testing Calibrator with svZeroDPlus Test Case")
    print("=" * 80)
    print(f"Test case: {args.test_case}")
    print(f"Ground truth CSV: {args.results_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Junction types: {args.junction_types}")
    print()
    
    # Read test case JSON
    print("Reading test case JSON...")
    with open(args.test_case, 'r') as f:
        test_case = json.load(f)
    
    # Use test case as geometric input
    geometric_input_path = output_dir / 'geometric_input.json'
    with open(geometric_input_path, 'w') as f:
        json.dump(test_case, f, indent=4)
    print(f"  Saved geometric input to: {geometric_input_path}")
    
    # Extract observations from CSV (ground truth)
    print("\nExtracting observations from ground truth CSV...")
    observations = extract_observations_from_csv(args.results_csv, geometric_input_path)
    
    # Process each junction type
    junction_type_paths = {}
    for jtype in args.junction_types:
        jtype_dir = output_dir / jtype
        jtype_dir.mkdir(exist_ok=True)
        
        junction_type_paths[jtype] = {
            'calibration_input': jtype_dir / 'calibration_input.json',
            'calibrated_output': jtype_dir / 'calibrated_output.json',
            'calibrated_results': jtype_dir / 'calibrated_results.csv',
            'geometric_results': jtype_dir / 'geometric_results.csv',
        }
    
    # Step 1: Create calibration inputs for each junction type
    print("\n" + "=" * 80)
    print("Step 1: Creating calibration inputs for each junction type")
    print("=" * 80)
    
    for jtype in args.junction_types:
        print(f"\nCreating calibration input for {jtype}...")
        
        # Modify geometric input for this junction type
        modified_input = modify_junction_types(test_case, jtype)
        
        # Save modified geometric input
        modified_geo_path = output_dir / jtype / 'geometric_input.json'
        with open(modified_geo_path, 'w') as f:
            json.dump(modified_input, f, indent=4)
        
        # Create calibration input
        calibration_input_path = junction_type_paths[jtype]['calibration_input']
        create_calibration_input(
            modified_geo_path,
            observations,
            calibration_input_path,
            centerline_soln_path=None,  # Not needed for CSV-based observations
            geo_dir=None,  # Not needed for CSV-based observations)
        )
    
    # Step 2: Run calibration for each junction type
    print("\n" + "=" * 80)
    print("Step 2: Running calibration for each junction type")
    print("=" * 80)
    
    for jtype in args.junction_types:
        print(f"\nRunning calibration for {jtype}...")
        calibration_input_path = junction_type_paths[jtype]['calibration_input']
        calibrated_output_path = junction_type_paths[jtype]['calibrated_output']
        
        try:
            run_calibration(calibration_input_path, calibrated_output_path)
            print(f"  ✓ Calibration completed for {jtype}")
        except Exception as e:
            print(f"  ✗ Calibration failed for {jtype}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Run forward simulations
    print("\n" + "=" * 80)
    print("Step 3: Running forward simulations")
    print("=" * 80)
    
    for jtype in args.junction_types:
        print(f"\nRunning forward simulation for {jtype}...")
        
        # Run geometric simulation
        modified_geo_path = output_dir / jtype / 'geometric_input.json'
        geometric_results_path = junction_type_paths[jtype]['geometric_results']
        try:
            run_forward_simulation(modified_geo_path, geometric_results_path)
            print(f"  ✓ Geometric simulation completed for {jtype}")
        except Exception as e:
            print(f"  ✗ Geometric simulation failed for {jtype}: {e}")
            import traceback
            traceback.print_exc()
        
        # Run calibrated simulation
        calibrated_output_path = junction_type_paths[jtype]['calibrated_output']
        if calibrated_output_path.exists():
            calibrated_results_path = junction_type_paths[jtype]['calibrated_results']
            try:
                run_forward_simulation(calibrated_output_path, calibrated_results_path)
                print(f"  ✓ Calibrated simulation completed for {jtype}")
            except Exception as e:
                print(f"  ✗ Calibrated simulation failed for {jtype}: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 4: Create comparison plots
    print("\n" + "=" * 80)
    print("Step 4: Creating comparison plots")
    print("=" * 80)
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    for jtype in args.junction_types:
        print(f"\nCreating plots for {jtype}...")
        
        calibration_input_path = junction_type_paths[jtype]['calibration_input']
        geometric_results_path = junction_type_paths[jtype]['geometric_results']
        calibrated_results_path = junction_type_paths[jtype]['calibrated_results']
        
        # Check if files exist
        if not calibration_input_path.exists():
            print(f"  ✗ Calibration input not found: {calibration_input_path}")
            continue
        
        # Pass calibrated CSV path as string or None (not a list)
        calibrated_csv_path = None
        if calibrated_results_path.exists():
            calibrated_csv_path = str(calibrated_results_path)
        
        # Create inlet comparison plot
        inlet_plot_path = plots_dir / f'{jtype}_inlet_comparison.png'
        try:
            plot_inlet_comparison(
                str(calibration_input_path),
                str(geometric_results_path),
                calibrated_csv_path,  # Pass as string or None, not list
                output_path=str(inlet_plot_path),
                set_name='test_case',
                geo_name=jtype,
                time_period=None  # Will be determined from data
            )
            print(f"  ✓ Inlet comparison plot saved: {inlet_plot_path}")
        except Exception as e:
            print(f"  ✗ Failed to create inlet plot: {e}")
            import traceback
            traceback.print_exc()
        
        # Create outlet comparison plots
        if geometric_results_path.exists():
            try:
                # Find all vessels with outlets (use modified geometric input)
                modified_geo_path = output_dir / jtype / 'geometric_input.json'
                vessel_outlets = find_all_vessels_with_outlets(modified_geo_path)
                
                outlet_plots_dir = plots_dir / jtype / 'outlets'
                outlet_plots_dir.mkdir(parents=True, exist_ok=True)
                
                success_count = 0
                for vessel_name, outlet_location in vessel_outlets:
                    outlet_plot_path = outlet_plots_dir / f'{vessel_name}_{outlet_location}_outlet_comparison.png'
                    try:
                        plot_outlet_comparison(
                            str(calibration_input_path),
                            str(geometric_results_path),
                            calibrated_csv_path,  # Pass as string or None, not list
                            vessel_name,
                            outlet_location,
                            str(outlet_plot_path),
                            set_name='test_case',
                            geo_name=jtype,
                            time_period=None
                        )
                        success_count += 1
                    except Exception as e:
                        print(f"    ✗ Failed to create outlet plot for {vessel_name} - {outlet_location}: {e}")
                
                print(f"  ✓ Created {success_count}/{len(vessel_outlets)} outlet comparison plots")
            except Exception as e:
                print(f"  ✗ Failed to create outlet plots: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()

