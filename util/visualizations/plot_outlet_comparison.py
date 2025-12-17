#!/usr/bin/env python3
"""
Script to plot outlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models
for each vessel with an outlet boundary condition.
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


def extract_outlet_data_from_csv(csv_path, vessel_name):
    """
    Extract outlet pressure and flow data from 0D CSV results.
    
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
            # Get outlet values
            if 'pressure_out' in data:
                pressures.append(data['pressure_out'])
                times_list.append(time)
                if 'flow_out' in data:
                    flows.append(data['flow_out'])
                else:
                    flows.append(None)
            elif 'flow_out' in data:
                flows.append(data['flow_out'])
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


def extract_outlet_data_from_calibration_input(calibration_input_path, vessel_name, outlet_location):
    """
    Extract outlet pressure and flow data from calibration input observations.
    Uses full observations if available (for plotting full 3D solution time series).
    
    Args:
        calibration_input_path: Path to calibration input JSON
        vessel_name: Name of the vessel (e.g., 'branch1_seg0')
        outlet_location: Name of the outlet location - can be BC name (e.g., 'RESISTANCE_0') 
                        or junction name (e.g., 'J0'), or None
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values (in dynes/cm^2, will be converted to mmHg)
        flows: Flow values (in cm³/s)
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Use full observations if available (for plotting), otherwise use calibration observations
    if '_full_observations' in calib_data and 'y' in calib_data['_full_observations']:
        observations = calib_data['_full_observations']['y']
    elif 'y' in calib_data:
        observations = calib_data['y']
    else:
        return None, None, None
    
    # Find outlet pressure and flow
    # Format can be:
    # - "pressure:vessel_name:outlet_location" (for BC or junction)
    # - "pressure:junction_name:vessel_name" (for junction outlet to vessel inlet - not what we want)
    pressure_key = None
    flow_key = None
    
    if outlet_location:
        # Try exact match: pressure:vessel_name:outlet_location
        for key in observations.keys():
            if key == f'pressure:{vessel_name}:{outlet_location}':
                pressure_key = key
            elif key == f'flow:{vessel_name}:{outlet_location}':
                flow_key = key
            # Also try prefix match in case there are variations
            elif key.startswith(f'pressure:{vessel_name}:{outlet_location}'):
                pressure_key = key
            elif key.startswith(f'flow:{vessel_name}:{outlet_location}'):
                flow_key = key
    else:
        # If outlet_location is None, try to find any outlet observation for this vessel
        # Look for patterns like "pressure:vessel_name:RESISTANCE_" or "pressure:vessel_name:J"
        for key in observations.keys():
            if key.startswith(f'pressure:{vessel_name}:'):
                # Check if it's not an inlet (INFLOW)
                parts = key.split(':')
                if len(parts) == 3 and parts[2] != 'INFLOW':
                    pressure_key = key
            elif key.startswith(f'flow:{vessel_name}:'):
                parts = key.split(':')
                if len(parts) == 3 and parts[2] != 'INFLOW':
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


def get_time_period_from_csv(csv_path):
    """
    Extract time period from CSV file by finding the maximum time value.
    Returns time period in seconds, or None if not found.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return None
    
    try:
        import csv
        max_time = None
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'time' in row:
                    try:
                        time_val = float(row['time'])
                        if max_time is None or time_val > max_time:
                            max_time = time_val
                    except (ValueError, TypeError):
                        continue
        return max_time
    except Exception as e:
        return None


def get_time_period(set_name, geo_name, csv_path=None):
    """
    Try to get the actual time period from 3D simulation XML or CSV file.
    Returns time period in seconds, or None if not found.
    
    Args:
        set_name: Set name (for finding XML)
        geo_name: Geometry name (for finding XML)
        csv_path: Optional CSV path to extract time period from (fallback)
    """
    # First try to get from CSV if provided
    if csv_path:
        time_period = get_time_period_from_csv(csv_path)
        if time_period is not None:
            return time_period
    
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


def find_all_vessels_with_outlets(geometric_input_path):
    """
    Find all vessels and determine their outlet location (BC or junction).
    
    Returns:
        List of tuples: [(vessel_name, outlet_location), ...]
        where outlet_location is either a BC name (e.g., 'RESISTANCE_0') or junction name (e.g., 'J0')
    """
    with open(geometric_input_path, 'r') as f:
        geo_input = json.load(f)
    
    vessels = geo_input.get('vessels', [])
    junctions = geo_input.get('junctions', [])
    
    # Create mapping from vessel_id to vessel_name
    vessel_id_to_name = {i: v['vessel_name'] for i, v in enumerate(vessels)}
    
    # Create mapping from vessel_id to outlet location
    vessel_outlets = []
    
    for vessel_idx, vessel in enumerate(vessels):
        vessel_name = vessel['vessel_name']
        outlet_location = None
        
        # Check if vessel has an outlet BC
        if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
            outlet_location = vessel['boundary_conditions']['outlet']
        else:
            # Check if vessel is an inlet to a junction (its outlet is at the junction)
            for junc in junctions:
                inlet_vessel_ids = junc.get('inlet_vessels', [])
                if vessel_idx in inlet_vessel_ids:
                    outlet_location = junc.get('junction_name', '')
                    break
        
        if outlet_location:
            vessel_outlets.append((vessel_name, outlet_location))
        else:
            # Still add vessel even if we can't find outlet location (will try to extract from CSV)
            vessel_outlets.append((vessel_name, None))
    
    return vessel_outlets


def plot_outlet_comparison(calibration_input_path, geometric_csv_path, calibrated_csv_paths,
                          vessel_name, outlet_location, output_path, set_name=None, geo_name=None, time_period=None,
                          pressure_ymin=None, pressure_ymax=None):
    """
    Plot outlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models.
    
    Args:
        calibration_input_path: Path to calibration input JSON (for 3D observations)
        geometric_csv_path: Path to geometric 0D results CSV
        calibrated_csv_paths: Dictionary mapping junction type names to CSV paths, or single CSV path
        vessel_name: Name of the vessel (e.g., 'branch1_seg0')
        outlet_location: Name of the outlet location - BC name (e.g., 'RESISTANCE_0') or junction name (e.g., 'J0'), or None
        output_path: Path to save plot
        set_name: Set name (for finding time period)
        geo_name: Geometry name (for finding time period)
        time_period: Time period in seconds (if None, will try to find from XML)
        pressure_ymin: Minimum value for pressure y-axis (if None, uses default -20)
        pressure_ymax: Maximum value for pressure y-axis (if None, uses default 20)
    """
    outlet_label = outlet_location if outlet_location else "outlet"
    print(f"\nPlotting outlet comparison for {vessel_name} ({outlet_label})")
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    # Get time period
    if time_period is None:
        # Try to get from CSV files (geometric or calibrated)
        if geometric_csv_path and os.path.exists(geometric_csv_path):
            time_period = get_time_period_from_csv(geometric_csv_path)
        if time_period is None and calibrated_csv_paths:
            # Try calibrated CSV (could be dict or string)
            if isinstance(calibrated_csv_paths, dict):
                for csv_path in calibrated_csv_paths.values():
                    if csv_path and os.path.exists(csv_path):
                        time_period = get_time_period_from_csv(csv_path)
                        if time_period is not None:
                            break
            elif isinstance(calibrated_csv_paths, str) and os.path.exists(calibrated_csv_paths):
                time_period = get_time_period_from_csv(calibrated_csv_paths)
        # Fallback to XML lookup
        if time_period is None and set_name is not None and geo_name is not None:
            time_period = get_time_period(set_name, geo_name, csv_path=geometric_csv_path)
    
    if time_period is None:
        print("  Warning: Could not determine time period, using default 1.0 s")
        time_period = 1.0
    else:
        print(f"  Time period: {time_period:.4f} s")
    
    # Extract geometric 0D results first (to get the correct time array)
    print(f"\nExtracting geometric 0D results for {vessel_name}...")
    times_geo, pressures_geo, flows_geo = extract_outlet_data_from_csv(geometric_csv_path, vessel_name)
    if times_geo is None:
        print(f"  Warning: Could not extract geometric 0D results for {vessel_name}")
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
    print(f"\nExtracting 3D observations for {vessel_name}...")
    times_3d_norm, pressures_3d, flows_3d = extract_outlet_data_from_calibration_input(
        calibration_input_path, vessel_name, outlet_location)
    
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
        print(f"  Warning: Could not extract 3D observations for {vessel_name}")
        pressures_3d_mmhg = None
        flows_3d = None
    
    # Extract calibrated 0D results (if available)
    calibrated_results = {}
    
    if calibrated_csv_paths is None:
        print("\nNote: No calibrated CSV files provided")
    elif isinstance(calibrated_csv_paths, dict):
        # Multiple junction types
        print(f"\nExtracting calibrated 0D results for {vessel_name}...")
        for jtype, csv_path in calibrated_csv_paths.items():
            if csv_path and os.path.exists(csv_path):
                times_cal, pressures_cal, flows_cal = extract_outlet_data_from_csv(csv_path, vessel_name)
                if times_cal is not None:
                    # Convert pressure to mmHg
                    pressures_cal_mmhg = pressures_cal / 1333.0 if pressures_cal is not None else None
                    
                    calibrated_results[jtype] = {
                        'times': times_cal,
                        'pressures': pressures_cal_mmhg,
                        'flows': flows_cal
                    }
                else:
                    print(f"    Warning: Could not extract calibrated 0D results for {jtype}/{vessel_name}")
    else:
        # Single CSV path (backward compatibility)
        if os.path.exists(calibrated_csv_paths):
            times_cal, pressures_cal, flows_cal = extract_outlet_data_from_csv(calibrated_csv_paths, vessel_name)
            if times_cal is not None:
                pressures_cal_mmhg = pressures_cal / 1333.0 if pressures_cal is not None else None
                
                calibrated_results['Calibrated'] = {
                    'times': times_cal,
                    'pressures': pressures_cal_mmhg,
                    'flows': flows_cal
                }
    
    # Create plot
    print(f"\nCreating plot for {vessel_name}...")
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
    ax.set_title(f'Outlet Pressure vs Time - {vessel_name} ({outlet_label})', fontsize=32, fontweight='bold')
    if pressure_ymin is not None and pressure_ymax is not None:
        ax.set_ylim(pressure_ymin, pressure_ymax)
    else:
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
    ax.set_title(f'Outlet Flow vs Time - {vessel_name} ({outlet_label})', fontsize=32, fontweight='bold')
    ax.legend(fontsize=24, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot outlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models for each vessel"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_007)')
    parser.add_argument('--calibration-input', help='Path to calibration input JSON (default: auto-detect)')
    parser.add_argument('--geometric-csv', help='Path to geometric 0D results CSV (default: auto-detect)')
    parser.add_argument('--geometric-input', help='Path to geometric input JSON (default: auto-detect)')
    parser.add_argument('--calibrated-csv', help='Path to calibrated 0D results CSV (default: auto-detect, supports multiple junction types)')
    parser.add_argument('--junction-types', nargs='+', 
                       default=['BloodVesselJunction', 'DirDepJunction', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to plot (default: all four types)')
    parser.add_argument('--output-dir', default='results/outlet_comparison', 
                       help='Output directory for plots (default: results/outlet_comparison)')
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
    
    if args.geometric_input:
        geometric_input_path = args.geometric_input
    else:
        geometric_input_path = os.path.join(data_dir, 'geometric_input.json')
    
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
    
    if not os.path.exists(geometric_input_path):
        print(f"Error: Geometric input file not found: {geometric_input_path}")
        sys.exit(1)
    
    # Find all vessels (every vessel has an outlet - either to BC or to junction)
    vessel_outlets = find_all_vessels_with_outlets(geometric_input_path)
    
    if not vessel_outlets:
        print("No vessels found.")
        sys.exit(1)
    
    print(f"\nFound {len(vessel_outlets)} vessels:")
    for vessel_name, outlet_location in vessel_outlets:
        outlet_label = outlet_location if outlet_location else "unknown"
        print(f"  {vessel_name}: outlet at {outlet_label}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each vessel outlet
    success_count = 0
    for vessel_name, outlet_location in vessel_outlets:
        # Create safe filename from vessel name and outlet location
        safe_vessel_name = vessel_name.replace('/', '_').replace('\\', '_')
        if outlet_location:
            safe_outlet_name = outlet_location.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f"{safe_vessel_name}_{safe_outlet_name}_outlet_comparison.png")
        else:
            output_path = os.path.join(output_dir, f"{safe_vessel_name}_outlet_comparison.png")
        
        success = plot_outlet_comparison(
            calibration_input_path, geometric_csv_path, calibrated_csv_paths,
            vessel_name, outlet_location, output_path,
            set_name=args.set_name, geo_name=args.geo_name,
            time_period=args.time_period
        )
        
        if success:
            success_count += 1
    
    print(f"\n✓ Successfully created {success_count}/{len(vessel_outlets)} outlet comparison plots")
    print(f"  Output directory: {output_dir}")
    
    if success_count == len(vessel_outlets):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

# python3 util/visualizations/plot_outlet_comparison.py --set-name set_4 --geo-name tree_002

