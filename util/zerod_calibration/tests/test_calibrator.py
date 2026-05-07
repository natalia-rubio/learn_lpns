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
    fit_outlet_resistances_from_observations,
    update_outlet_bcs_in_file,
    replace_inlet_bc_in_calibrated_output,
    update_geometric_input_with_calibration_bc
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
    get_all_vessels_from_csv
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
    parser.add_argument('--output-dir', type=str,
                       default='test_calibration_output',
                       help='Output directory for calibration inputs, outputs, and plots')
    parser.add_argument('--junction-types', nargs='+',
                       default=['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to test')
    
    args = parser.parse_args()
    
    # Extract test case name from file path
    test_case_path = Path(args.test_case)
    test_case_filename = test_case_path.name  # Full filename with extension
    test_case_name = test_case_path.stem  # Get filename without extension
    # Remove common prefixes/suffixes if needed for subdirectory name
    if test_case_name.startswith('sinusoidalFlow_'):
        subdir_name = test_case_name.replace('sinusoidalFlow_', '')
    elif test_case_name.startswith('steadyFlow_'):
        subdir_name = test_case_name.replace('steadyFlow_', '')
    else:
        subdir_name = test_case_name
    
    # Automatically determine results CSV path from test case path
    # Results CSV is in the 'results' subdirectory with same filename but .csv extension
    test_case_dir = test_case_path.parent
    results_dir = test_case_dir / 'results'
    results_csv_path = results_dir / f"{test_case_name}.csv"
    
    # Create output directory with test case name as subdirectory
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Testing Calibrator with svZeroDPlus Test Case")
    print("=" * 80)
    print(f"Test case: {args.test_case}")
    print(f"Test case name: {test_case_name}")
    print(f"Ground truth CSV: {results_csv_path}")
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
    observations = extract_observations_from_csv(str(results_csv_path), geometric_input_path)
    
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
    
    # Step 1: Create base calibration input
    print("\n" + "=" * 80)
    print("Step 1: Creating base calibration input")
    print("=" * 80)
    
    # Use test case as base geometric input
    # Ensure base geometric input uses default junction type (BloodVesselJunction)
    base_geometric_config = modify_junction_types(test_case, 'BloodVesselJunction')
    base_geometric_input_path = output_dir / 'geometric_input.json'
    with open(base_geometric_input_path, 'w') as f:
        json.dump(base_geometric_config, f, indent=4)
    
    # Create base calibration input
    base_calibration_input_path = output_dir / 'calibration_input.json'
    print(f"\nCreating base calibration input...")
    create_calibration_input(
        base_geometric_input_path,
        observations,
        base_calibration_input_path,
        centerline_soln_path=None,  # Not needed for CSV-based observations
        geo_dir=None,  # Not needed for CSV-based observations
    )
    print(f"  ✓ Base calibration input saved to: {base_calibration_input_path}")
    
    # Fit outlet resistances from observations
    print(f"\nFitting outlet resistances from observations...")
    fitted_resistances = fit_outlet_resistances_from_observations(base_geometric_input_path, observations)
    
    # Update base calibration input with fitted outlet BCs
    print(f"\nUpdating base calibration input with fitted outlet BCs...")
    update_outlet_bcs_in_file(base_calibration_input_path, fitted_resistances, "base calibration input")
    
    # Update geometric input with BC from calibration input
    update_geometric_input_with_calibration_bc(base_geometric_input_path, base_calibration_input_path)
    
    # Create calibration input variants for each junction type
    print(f"\nCreating calibration input variants for each junction type...")
    with open(base_calibration_input_path, 'r') as f:
        base_calibration_config = json.load(f)
    
    for jtype in args.junction_types:
        print(f"  Creating {jtype} calibration input...")
        
        # Apply junction type modification to calibration input
        jtype_config = modify_junction_types(base_calibration_config, jtype)
        
        # Save modified geometric input for this junction type
        modified_geo_path = output_dir / jtype / 'geometric_input.json'
        modified_geo_config = modify_junction_types(test_case, jtype)
        with open(modified_geo_path, 'w') as f:
            json.dump(modified_geo_config, f, indent=4)
        
        # Save calibration input for this junction type
        calibration_input_path = junction_type_paths[jtype]['calibration_input']
        with open(calibration_input_path, 'w') as f:
            json.dump(jtype_config, f, indent=4)
        
        # Update with fitted outlet BCs
        update_outlet_bcs_in_file(calibration_input_path, fitted_resistances, f"{jtype} calibration input")
        
        print(f"    ✓ Saved to: {calibration_input_path}")
    
    # Step 2: Run calibration for each junction type
    print("\n" + "=" * 80)
    print("Step 2: Running calibration for each junction type")
    print("=" * 80)
    print("\n  Note: Base case uses ground truth directly (no calibration needed)")
    
    # Run calibration for each junction type variant
    for jtype in args.junction_types:
        print(f"\n  Calibrating {jtype}...")
        calibration_input_path = junction_type_paths[jtype]['calibration_input']
        calibrated_output_path = junction_type_paths[jtype]['calibrated_output']
        
        try:
            run_calibration(calibration_input_path, calibrated_output_path)
            print(f"    ✓ Calibration completed for {jtype}")
            # Replace inlet BC with original observed BC
            replace_inlet_bc_in_calibrated_output(calibrated_output_path, calibration_input_path)
            # Update outlet BCs with fitted values
            update_outlet_bcs_in_file(calibrated_output_path, fitted_resistances, f"{jtype} calibrated output")
        except Exception as e:
            print(f"    ✗ Calibration failed for {jtype}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Run forward simulations
    print("\n" + "=" * 80)
    print("Step 3: Running forward simulations")
    print("=" * 80)
    
    # Run simulation with base geometric input (only once, not per junction type)
    print("\n  Running simulation with geometric input...")
    base_geometric_results_path = output_dir / 'geometric_results.csv'
    try:
        run_forward_simulation(base_geometric_input_path, base_geometric_results_path)
        print(f"    ✓ Geometric simulation completed successfully")
    except Exception as e:
        print(f"    ✗ Geometric simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Note: Ground truth results are already available from results_csv_path (no simulation needed)
    
    # Run simulation with each junction type variant
    for jtype in args.junction_types:
        print(f"\n  Running simulation with calibrated {jtype} input...")
        calibrated_output_path = junction_type_paths[jtype]['calibrated_output']
        calibrated_results_path = junction_type_paths[jtype]['calibrated_results']
        
        if not calibrated_output_path.exists():
            print(f"    ✗ Calibrated output not found: {calibrated_output_path}")
            continue
        
        try:
            run_forward_simulation(calibrated_output_path, calibrated_results_path)
            print(f"    ✓ Calibrated {jtype} simulation completed successfully")
        except Exception as e:
            print(f"    ✗ Calibrated {jtype} simulation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Create comparison plots
    print("\n" + "=" * 80)
    print("Step 4: Creating comparison plots")
    print("=" * 80)
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Build dictionary of calibrated CSV paths for all junction types
    calibrated_csv_paths = {}
    for jtype in args.junction_types:
        calibrated_results_path = junction_type_paths[jtype]['calibrated_results']
        if calibrated_results_path.exists():
            calibrated_csv_paths[jtype] = str(calibrated_results_path)
        else:
            print(f"  Warning: Calibrated results not found for {jtype}: {calibrated_results_path}")
    
    if not calibrated_csv_paths:
        print("  ✗ No calibrated results found for any junction type")
        return
    
    # Use the first junction type's calibration input (they all have the same 3D observations)
    first_jtype = args.junction_types[0]
    calibration_input_path = junction_type_paths[first_jtype]['calibration_input']
    
    if not calibration_input_path.exists():
        print(f"  ✗ Calibration input not found: {calibration_input_path}")
        return
    
    # Create single inlet comparison plot with all junction types
    print(f"\nCreating inlet comparison plot with all junction types...")
    inlet_plot_path = plots_dir / 'inlet_comparison.png'
    try:
        plot_inlet_comparison(
            str(calibration_input_path),  # Contains 3D observations (ground truth)
            str(base_geometric_results_path),  # Geometric 0D results
            calibrated_csv_paths,  # Dictionary of calibrated 0D results for all junction types
            output_path=str(inlet_plot_path),
            set_name='test_case',
            geo_name='all_junction_types',
            time_period=None  # Will be determined from data
        )
        print(f"  ✓ Inlet comparison plot saved: {inlet_plot_path}")
    except Exception as e:
        print(f"  ✗ Failed to create inlet plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Create outlet comparison plots (one per vessel, showing all junction types)
    if base_geometric_results_path.exists():
        print(f"\nCreating outlet comparison plots for all vessels...")
        try:
            # Get all vessels from CSV (all vessels have outlets, either to junctions or terminal BCs)
            vessel_names = get_all_vessels_from_csv(str(base_geometric_results_path))
            
            # Use the first junction type's geometric input (for structure)
            modified_geo_path = output_dir / first_jtype / 'geometric_input.json'
            
            outlet_plots_dir = plots_dir / 'outlets'
            outlet_plots_dir.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            for vessel_name in vessel_names:
                outlet_plot_path = outlet_plots_dir / f'{vessel_name}_outlet_comparison.png'
                try:
                    plot_outlet_comparison(
                        str(calibration_input_path),  # Contains 3D observations (ground truth)
                        str(base_geometric_results_path),  # Geometric 0D results
                        calibrated_csv_paths,  # Dictionary of calibrated 0D results for all junction types
                        vessel_name,
                        str(outlet_plot_path),  # output_path parameter
                        set_name='test_case',
                        geo_name='all_junction_types',
                        time_period=None,
                        geometric_input_path=str(modified_geo_path)  # Pass geometric input path
                    )
                    success_count += 1
                except Exception as e:
                    print(f"    ✗ Failed to create outlet plot for {vessel_name}: {e}")
            
            print(f"  ✓ Created {success_count}/{len(vessel_names)} outlet comparison plots")
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

