#!/usr/bin/env python3
"""
Script to plot inlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models.
"""

import os
import sys
import argparse
import json
import csv
import numpy as np

# Check for matplotlib
HAS_MATPLOTLIB = False
plt = None
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Configure LaTeX rendering with Computer Modern font
    try:
        plt.rcParams['text.usetex'] = True
        # Test if LaTeX is available
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, r'Test $\alpha$')
        plt.close(test_fig)
        # If successful, configure LaTeX settings
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
        LATEX_AVAILABLE = True
    except Exception:
        # LaTeX not available, use mathtext with Computer Modern
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
        LATEX_AVAILABLE = False
    
    # Set font sizes (doubled from original)
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['figure.titlesize'] = 36
    
    HAS_MATPLOTLIB = True
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def read_zerod_csv(csv_path):
    """
    Read 0D simulation results from CSV.
    Handles both 'location' and 'name' as the vessel identifier column.
    
    Returns:
        results: Dictionary {location: {time: {field: value}}}
        times: Sorted list of time values
    """
    results = {}
    times = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Check which column name is used for vessel identifier
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return results, sorted(times)
        
        vessel_col = None
        if 'location' in fieldnames:
            vessel_col = 'location'
        elif 'name' in fieldnames:
            vessel_col = 'name'
        else:
            raise ValueError(f"CSV file must have either 'location' or 'name' column. Found: {fieldnames}")
        
        for row in reader:
            location = row[vessel_col]
            time = float(row['time'])
            times.add(time)
            
            if location not in results:
                results[location] = {}
            if time not in results[location]:
                results[location][time] = {}
            
            # Extract all numeric fields
            for key, value in row.items():
                if key not in [vessel_col, 'time']:
                    try:
                        results[location][time][key] = float(value)
                    except (ValueError, TypeError):
                        continue
    
    return results, sorted(times)


def extract_inlet_data_from_csv(csv_path, vessel_name='branch0_seg0'):
    """
    Extract inlet pressure and flow data from 0D CSV results.
    
    Returns:
        times: Time array
        pressures: Pressure values (in dynes/cm^2)
        flows: Flow values (in cm³/s)
    """
    results, times = read_zerod_csv(csv_path)
    
    if vessel_name not in results:
        return None, None, None
    
    pressures = []
    flows = []
    times_list = []
    
    for time in times:
        if time in results[vessel_name]:
            data = results[vessel_name][time]
            # Get inlet values
            if 'pressure_in' in data:
                pressures.append(data['pressure_in'])
                times_list.append(time)
                if 'flow_in' in data:
                    flows.append(data['flow_in'])
                else:
                    flows.append(None)
            elif 'flow_in' in data:
                flows.append(data['flow_in'])
                times_list.append(time)
                pressures.append(None)
    
    # Filter out None values
    if pressures and all(p is not None for p in pressures):
        pressures = np.array(pressures)
    else:
        pressures = None
    
    if flows and all(f is not None for f in flows):
        flows = np.array(flows)
    else:
        flows = None
    
    times_array = np.array(times_list) if times_list else None
    
    return times_array, pressures, flows


def extract_inlet_data_from_calibration_input(calibration_input_path):
    """
    Extract inlet pressure and flow data from calibration input observations.
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values (in dynes/cm^2, will be converted to mmHg)
        flows: Flow values (in cm³/s)
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None, None
    
    observations = calib_data['y']
    
    # Find inlet pressure and flow
    pressure_key = None
    flow_key = None
    
    for key in observations.keys():
        if key.startswith('pressure:INFLOW:'):
            pressure_key = key
        elif key.startswith('flow:INFLOW:'):
            flow_key = key
    
    if pressure_key is None and flow_key is None:
        return None, None, None
    
    # Get number of observations
    if pressure_key:
        num_obs = len(observations[pressure_key])
    elif flow_key:
        num_obs = len(observations[flow_key])
    else:
        return None, None, None
    
    times = np.linspace(0.0, 1.0, num_obs)
    
    # Extract pressure (in dynes/cm^2)
    if pressure_key:
        pressures = np.array(observations[pressure_key])
    else:
        pressures = None
    
    # Extract flow
    if flow_key:
        flows = np.array(observations[flow_key])
    else:
        flows = None
    
    return times, pressures, flows


def get_time_period(set_name, geo_name):
    """
    Try to get the actual time period from 3D simulation XML.
    Returns time period in seconds, or None if not found.
    """
    # Try to find XML file
    xml_paths = [
        os.path.join('data', 'threeD', set_name, geo_name, 'fluid_simulation_0-0.xml'),
        os.path.join('data', 'threeD', set_name, geo_name, 'solver.inp'),
    ]
    
    for xml_path in xml_paths:
        if os.path.exists(xml_path):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Look for time step size and number of time steps
                gen_params = root.find('General_Parameters')
                if gen_params is None:
                    gen_params = root.find('GeneralSimulationParameters')
                
                if gen_params is not None:
                    num_time_steps_elem = gen_params.find('Number_of_time_steps')
                    time_step_size_elem = gen_params.find('Time_step_size')
                    
                    if num_time_steps_elem is not None and time_step_size_elem is not None:
                        num_time_steps = int(num_time_steps_elem.text)
                        time_step_size = float(time_step_size_elem.text)
                        time_period = num_time_steps * time_step_size
                        return time_period
            except Exception as e:
                pass
    
    return None


def plot_inlet_comparison(calibration_input_path, geometric_csv_path, calibrated_csv_paths,
                         output_path, set_name=None, geo_name=None, time_period=None):
    """
    Plot inlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models.
    
    Args:
        calibration_input_path: Path to calibration input JSON (for 3D observations)
        geometric_csv_path: Path to geometric 0D results CSV
        calibrated_csv_paths: Dictionary mapping junction type names to CSV paths, or single CSV path
        output_path: Path to save plot
        set_name: Set name (for finding time period)
        geo_name: Geometry name (for finding time period)
        time_period: Time period in seconds (if None, will try to find from XML)
    """
    print("="*60)
    print("Plotting Inlet Comparison")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    # Get time period
    if time_period is None and set_name is not None and geo_name is not None:
        time_period = get_time_period(set_name, geo_name)
    
    if time_period is None:
        print("  Warning: Could not determine time period, using default 1.0 s")
        time_period = 1.0
    else:
        print(f"  Time period: {time_period:.4f} s")
    
    # Extract geometric 0D results first (to get the correct time array)
    print("\nExtracting geometric 0D results...")
    times_geo, pressures_geo, flows_geo = extract_inlet_data_from_csv(geometric_csv_path)
    if times_geo is None:
        print("  Error: Could not extract geometric 0D results")
        return False
    
    # Convert pressure to mmHg
    if pressures_geo is not None:
        pressures_geo_mmhg = pressures_geo / 1333.0
    else:
        pressures_geo_mmhg = None
    print(f"  Found {len(times_geo)} time points")
    if pressures_geo_mmhg is not None:
        print(f"    Pressure range: [{np.min(pressures_geo_mmhg):.2f}, {np.max(pressures_geo_mmhg):.2f}] mmHg")
    if flows_geo is not None:
        print(f"    Flow range: [{np.min(flows_geo):.2f}, {np.max(flows_geo):.2f}] cm³/s")
    
    # Extract 3D observations (from calibration input)
    print("\nExtracting 3D observations...")
    times_3d_norm, pressures_3d, flows_3d = extract_inlet_data_from_calibration_input(calibration_input_path)
    if times_3d_norm is not None:
        # Use the same times as the 1D solution (geometric 0D results)
        # Interpolate 3D observations to match 1D solution time points
        times_3d_sec = times_geo
        
        # Map normalized 3D times [0, 1] to the actual time range of 1D solution
        time_min = times_geo[0]
        time_max = times_geo[-1]
        times_3d_original = time_min + times_3d_norm * (time_max - time_min)
        
        # Interpolate 3D pressures and flows to 1D time points
        if pressures_3d is not None:
            pressures_3d_interp = np.interp(times_geo, times_3d_original, pressures_3d)
            pressures_3d_mmhg = pressures_3d_interp / 1333.0
        else:
            pressures_3d_mmhg = None
        
        if flows_3d is not None:
            flows_3d_interp = np.interp(times_geo, times_3d_original, flows_3d)
            flows_3d = flows_3d_interp
        else:
            flows_3d = None
        
        print(f"  Found {len(times_3d_norm)} original time points, interpolated to {len(times_geo)} points")
        if pressures_3d_mmhg is not None:
            print(f"    Pressure range: [{np.min(pressures_3d_mmhg):.2f}, {np.max(pressures_3d_mmhg):.2f}] mmHg")
        if flows_3d is not None:
            print(f"    Flow range: [{np.min(flows_3d):.2f}, {np.max(flows_3d):.2f}] cm³/s")
    else:
        print("  Error: Could not extract 3D observations")
        return False
    
    # Extract calibrated 0D results (if available)
    # Support both single CSV path (backward compatibility) and dictionary of junction types
    calibrated_results = {}
    
    if calibrated_csv_paths is None:
        print("\nNote: No calibrated CSV files provided")
    elif isinstance(calibrated_csv_paths, dict):
        # Multiple junction types
        print("\nExtracting calibrated 0D results for each junction type...")
        for jtype, csv_path in calibrated_csv_paths.items():
            if csv_path and os.path.exists(csv_path):
                print(f"\n  {jtype}:")
                times_cal, pressures_cal, flows_cal = extract_inlet_data_from_csv(csv_path)
                if times_cal is not None:
                    # Convert pressure to mmHg
                    pressures_cal_mmhg = pressures_cal / 1333.0 if pressures_cal is not None else None
                    
                    calibrated_results[jtype] = {
                        'times': times_cal,
                        'pressures': pressures_cal_mmhg,
                        'flows': flows_cal
                    }
                    
                    print(f"    Found {len(times_cal)} time points")
                    if pressures_cal_mmhg is not None:
                        print(f"    Pressure range: [{np.min(pressures_cal_mmhg):.2f}, {np.max(pressures_cal_mmhg):.2f}] mmHg")
                    if flows_cal is not None:
                        print(f"    Flow range: [{np.min(flows_cal):.2f}, {np.max(flows_cal):.2f}] cm³/s")
                else:
                    print(f"    Warning: Could not extract calibrated 0D results for {jtype}")
            else:
                print(f"\n  {jtype}: CSV not found at {csv_path}")
    else:
        # Single CSV path (backward compatibility)
        if os.path.exists(calibrated_csv_paths):
            print("\nExtracting calibrated 0D results...")
            times_cal, pressures_cal, flows_cal = extract_inlet_data_from_csv(calibrated_csv_paths)
            if times_cal is not None:
                pressures_cal_mmhg = pressures_cal / 1333.0 if pressures_cal is not None else None
                
                calibrated_results['Calibrated'] = {
                    'times': times_cal,
                    'pressures': pressures_cal_mmhg,
                    'flows': flows_cal
                }
                
                print(f"  Found {len(times_cal)} time points")
                if pressures_cal_mmhg is not None:
                    print(f"    Pressure range: [{np.min(pressures_cal_mmhg):.2f}, {np.max(pressures_cal_mmhg):.2f}] mmHg")
                if flows_cal is not None:
                    print(f"    Flow range: [{np.min(flows_cal):.2f}, {np.max(flows_cal):.2f}] cm³/s")
            else:
                print("  Warning: Could not extract calibrated 0D results")
        else:
            print(f"\nNote: Calibrated CSV not found at {calibrated_csv_paths}")
    
    # Create plot
    print("\nCreating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Define colors and linestyles for different junction types
    junction_styles = {
        'BloodVesselJunction': {'color': 'orange', 'linestyle': '-', 'label': 'Blood Vessel Junction'},
        'DirDepJunction': {'color': 'red', 'linestyle': '--', 'label': 'Dir-Dep Junction'},
        'DirIndepJunction': {'color': 'blue', 'linestyle': '-.', 'label': 'Dir-Indep Junction'},
        'HybridJunction': {'color': 'purple', 'linestyle': ':', 'label': 'Hybrid Junction'},
        'Calibrated': {'color': 'red', 'linestyle': '--', 'label': 'Calibrated 0D'}
    }
    
    # Plot pressure (top subplot)
    ax = axes[0]
    
    # Plot 3D observations (black)
    if pressures_3d_mmhg is not None:
        ax.plot(times_3d_sec, pressures_3d_mmhg, 'k-', linewidth=4, label='3D Model', alpha=0.8)
    
    # Plot geometric 0D (green dotted line, 2x thickness)
    if pressures_geo_mmhg is not None:
        ax.plot(times_geo, pressures_geo_mmhg, 'g-', linewidth=8, label='Geometric 0D', alpha=0.8, linestyle=':')
    
    # Plot all calibrated 0D results
    for jtype, data in calibrated_results.items():
        if data['pressures'] is not None:
            style = junction_styles.get(jtype, {'color': 'red', 'linestyle': '--', 'label': jtype})
            ax.plot(data['times'], data['pressures'], 
                   color=style['color'], linewidth=4, 
                   label=style['label'], alpha=0.8, linestyle=style['linestyle'])
    
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=28)
    ax.set_title(r'Inlet Pressure vs Time', fontsize=32, fontweight='bold')
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.3)
    
    # Plot flow (bottom subplot)
    ax = axes[1]
    
    # Plot 3D observations (black)
    if flows_3d is not None:
        ax.plot(times_3d_sec, flows_3d, 'k-', linewidth=4, label='3D Model', alpha=0.8)
    
    # Plot geometric 0D (green dotted line, 2x thickness)
    if flows_geo is not None:
        ax.plot(times_geo, flows_geo, 'g-', linewidth=8, label='Geometric 0D', alpha=0.8, linestyle=':')
    
    # Plot all calibrated 0D results
    for jtype, data in calibrated_results.items():
        if data['flows'] is not None:
            style = junction_styles.get(jtype, {'color': 'red', 'linestyle': '--', 'label': jtype})
            ax.plot(data['times'], data['flows'], 
                   color=style['color'], linewidth=4, 
                   label=style['label'], alpha=0.8, linestyle=style['linestyle'])
    
    ax.set_xlabel(r'Time (s)', fontsize=28)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=28)
    ax.set_title(r'Inlet Flow vs Time', fontsize=32, fontweight='bold')
    ax.legend(fontsize=24, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot inlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_007)')
    parser.add_argument('--calibration-input', help='Path to calibration input JSON (default: auto-detect)')
    parser.add_argument('--geometric-csv', help='Path to geometric 0D results CSV (default: auto-detect)')
    parser.add_argument('--calibrated-csv', help='Path to calibrated 0D results CSV (default: auto-detect, supports multiple junction types)')
    parser.add_argument('--junction-types', nargs='+', 
                       default=['BloodVesselJunction', 'DirDepJunction', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to plot (default: all four types)')
    parser.add_argument('--output', help='Output plot path (default: auto-generate)')
    parser.add_argument('--output-dir', default='results/inlet_comparison', 
                        help='Output directory for plots (default: results/inlet_comparison)')
    parser.add_argument('--data-dir', default='data/zeroD', 
                        help='Data directory for input files (default: data/zeroD)')
    parser.add_argument('--time-period', type=float, default=None,
                        help='Time period in seconds (default: auto-detect from XML)')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        print("  Install with: pip install matplotlib")
        sys.exit(1)
    
    # Auto-detect file paths
    data_dir = os.path.join(args.data_dir, args.set_name, args.geo_name)
    
    if args.calibration_input:
        calibration_input_path = args.calibration_input
    else:
        calibration_input_path = os.path.join(data_dir, 'calibration_input.json')
    
    if args.geometric_csv:
        geometric_csv_path = args.geometric_csv
    else:
        geometric_csv_path = os.path.join(data_dir, 'geometric_results.csv')
    
    # Set up calibrated CSV paths for each junction type
    calibrated_csv_paths = {}
    if args.calibrated_csv:
        # Single CSV path provided (backward compatibility)
        calibrated_csv_paths = args.calibrated_csv
    else:
        # Auto-detect multiple junction types
        for jtype in args.junction_types:
            csv_path = os.path.join(data_dir, f'calibrated_results_{jtype}.csv')
            calibrated_csv_paths[jtype] = csv_path
    
    if not os.path.exists(calibration_input_path):
        print(f"Error: Calibration input file not found: {calibration_input_path}")
        sys.exit(1)
    
    if not os.path.exists(geometric_csv_path):
        print(f"Error: Geometric CSV file not found: {geometric_csv_path}")
        sys.exit(1)
    
    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.geo_name}_inlet_comparison.png")
    
    print(f"\nPlotting inlet comparison for {args.set_name}/{args.geo_name}")
    print(f"  Calibration input: {calibration_input_path}")
    print(f"  Geometric CSV: {geometric_csv_path}")
    if isinstance(calibrated_csv_paths, dict):
        print(f"  Calibrated CSVs:")
        for jtype, csv_path in calibrated_csv_paths.items():
            print(f"    {jtype}: {csv_path}")
    else:
        print(f"  Calibrated CSV: {calibrated_csv_paths}")
    print(f"  Output: {output_path}")
    
    success = plot_inlet_comparison(
        calibration_input_path, geometric_csv_path, calibrated_csv_paths,
        output_path, set_name=args.set_name, geo_name=args.geo_name,
        time_period=args.time_period
    )
    
    if success:
        print(f"\n✓ Successfully created comparison plot: {output_path}")
        sys.exit(0)
    else:
        print(f"\n✗ Failed to create plot")
        sys.exit(1)


if __name__ == '__main__':
    main()

# python3 util/visualizations/plot_inlet_comparison.py --set-name set_3 --geo-name tree_007

