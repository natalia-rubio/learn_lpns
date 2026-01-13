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
    
    # Get time period, used for zoom window
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
        print("  Warning: Could not extract calibrated 0D results")

    # Create plot with 4 subplots: 2 zoomed (top) and 2 full range (bottom)
    print("\nCreating plot...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    # Share x-axis within zoomed plots and within full plots
    axes[0].sharex(axes[1])
    axes[2].sharex(axes[3])
    
    # Define colors and linestyles for different junction types
    junction_styles = {
        'BloodVesselJunction': {'color': 'orange', 'linestyle': '-', 'label': 'Blood Vessel Junction'},
        'NORMAL_JUNCTION': {'color': 'red', 'linestyle': '--', 'label': 'Normal Junction'},
        'DirIndepJunction': {'color': 'skyblue', 'linestyle': '-.', 'label': 'Dir-Indep Junction'},
        'HybridJunction': {'color': 'violet', 'linestyle': ':', 'label': 'Hybrid Junction'},
        'Calibrated': {'color': 'red', 'linestyle': '--', 'label': 'Calibrated 0D'}
    }
    
    # Determine zoom window
    num_time_steps = len(times_geo)
    if set_name == 'VMR':
        # For VMR: zoom to 1-2 seconds (time_period is typically 2.0s)
        vmr_period = time_period if time_period else 2.0
        dt = vmr_period / (num_time_steps - 1) if num_time_steps > 1 else 1.0
        zoom_start_idx = int(1.0 / dt)
        zoom_end_idx = min(int(2.0 / dt) + 1, num_time_steps)
    else:
        # Default: last 105 time steps
        zoom_start_idx = max(0, num_time_steps - 105)
        zoom_end_idx = num_time_steps
    zoom_times = times_geo[zoom_start_idx:zoom_end_idx]
    
    # Helper function to plot pressure data on an axis
    def plot_pressure_data(ax, times_data, pressures_3d_data, times_geo_data, pressures_geo_data, 
                          calibrated_data_dict, junction_styles_dict, set_ylim_from_3d=True):
        """Plot pressure data on given axis"""
        # Plot 3D observations (black)
        if pressures_3d_data is not None:
            ax.plot(times_data, pressures_3d_data, 'k-', linestyle='-',linewidth=8, label='3D Model', alpha=1)
        
        # Plot geometric 0D (green dotted line)
        if pressures_geo_data is not None:
            ax.plot(times_geo_data, pressures_geo_data, 'g', linestyle='--',linewidth=3, label='Geometric 0D', alpha=1)
        
        # Plot all calibrated 0D results
        for jtype, data in calibrated_data_dict.items():
            if data['pressures'] is not None:
                style = junction_styles_dict.get(jtype, {'color': 'red', 'linestyle': '--', 'label': jtype})
                ax.plot(data['times'], data['pressures'], 
                       color=style['color'], linewidth=3, 
                       label=style['label'], alpha=1, linestyle=style['linestyle'])
        
        # Set y-limits based on 3D solution +/- 10%
        if set_ylim_from_3d and pressures_3d_data is not None:
            pressure_min = np.min(pressures_3d_data)
            pressure_max = np.max(pressures_3d_data)
            pressure_range = pressure_max - pressure_min
            if pressure_range > 0:
                ax.set_ylim(pressure_min - 0.1 * pressure_range, pressure_max + 0.1 * pressure_range)
            else:
                ax.set_ylim(pressure_min - 0.1, pressure_max + 0.1)
        elif not set_ylim_from_3d:
            # For zoomed plot, use data in zoom window
            all_pressures = []
            if pressures_3d_data is not None:
                all_pressures.extend(pressures_3d_data)
            if pressures_geo_data is not None:
                all_pressures.extend(pressures_geo_data)
            for data in calibrated_data_dict.values():
                if data['pressures'] is not None:
                    all_pressures.extend(data['pressures'])
            if all_pressures:
                pressure_min = np.min(all_pressures)
                pressure_max = np.max(all_pressures)
                pressure_range = pressure_max - pressure_min
                if pressure_range > 0:
                    ax.set_ylim(pressure_min - 0.1 * pressure_range, pressure_max + 0.1 * pressure_range)
                else:
                    ax.set_ylim(pressure_min - 0.1, pressure_max + 0.1)
            else:
                ax.set_ylim(-20, 20)
        else:
            ax.set_ylim(-20, 20)
    
    # Helper function to plot flow data on an axis
    def plot_flow_data(ax, times_data, flows_3d_data, times_geo_data, flows_geo_data, 
                      calibrated_data_dict, junction_styles_dict, set_ylim_from_3d=True):
        """Plot flow data on given axis"""
        # Plot 3D observations (black)
        if flows_3d_data is not None:
            ax.plot(times_data, flows_3d_data, 'k-', linewidth=8, label='3D Model', alpha=1)
        
        # Plot geometric 0D (green dotted line)
        if flows_geo_data is not None:
            ax.plot(times_geo_data, flows_geo_data, 'g--', linewidth=3, label='Geometric 0D', alpha=1, linestyle=':')
        
        # Plot all calibrated 0D results
        for jtype, data in calibrated_data_dict.items():
            if data['flows'] is not None:
                style = junction_styles_dict.get(jtype, {'color': 'red', 'linestyle': '--', 'label': jtype})
                ax.plot(data['times'], data['flows'], 
                       color=style['color'], linewidth=2, 
                       label=style['label'], alpha=1, linestyle=style['linestyle'])
        
        # Set y-limits based on 3D solution +/- 10%
        if set_ylim_from_3d and flows_3d_data is not None:
            flow_min = np.min(flows_3d_data)
            flow_max = np.max(flows_3d_data)
            flow_range = flow_max - flow_min
            if flow_range > 0:
                ax.set_ylim(flow_min - 0.1 * flow_range, flow_max + 0.1 * flow_range)
            else:
                ax.set_ylim(flow_min - 0.1, flow_max + 0.1)
        elif not set_ylim_from_3d:
            # For zoomed plot, use data in zoom window
            all_flows = []
            if flows_3d_data is not None:
                all_flows.extend(flows_3d_data)
            if flows_geo_data is not None:
                all_flows.extend(flows_geo_data)
            for data in calibrated_data_dict.values():
                if data['flows'] is not None:
                    all_flows.extend(data['flows'])
            if all_flows:
                flow_min = np.min(all_flows)
                flow_max = np.max(all_flows)
                flow_range = flow_max - flow_min
                if flow_range > 0:
                    ax.set_ylim(flow_min - 0.1 * flow_range, flow_max + 0.1 * flow_range)
                else:
                    ax.set_ylim(flow_min - 0.1, flow_max + 0.1)
    
    # Prepare zoomed data (last 200 time steps)
    # Use time range to extract data (more robust than index-based)
    time_zoom_start = zoom_times[0] if len(zoom_times) > 0 else times_geo[0]
    time_zoom_end = zoom_times[-1] if len(zoom_times) > 0 else times_geo[-1]
    
    pressures_3d_zoom = pressures_3d_mmhg[zoom_start_idx:zoom_end_idx] if pressures_3d_mmhg is not None else None
    pressures_geo_zoom = pressures_geo_mmhg[zoom_start_idx:zoom_end_idx] if pressures_geo_mmhg is not None else None
    flows_3d_zoom = flows_3d[zoom_start_idx:zoom_end_idx] if flows_3d is not None else None
    flows_geo_zoom = flows_geo[zoom_start_idx:zoom_end_idx] if flows_geo is not None else None
    
    calibrated_results_zoom = {}
    for jtype, data in calibrated_results.items():
        if data['times'] is not None and len(data['times']) > 0:
            # Find indices within zoom time range
            times_cal = np.array(data['times'])
            zoom_mask = (times_cal >= time_zoom_start) & (times_cal <= time_zoom_end)
            if np.any(zoom_mask):
                times_cal_zoom = times_cal[zoom_mask]
                pressures_cal = np.array(data['pressures']) if data['pressures'] is not None else None
                flows_cal = np.array(data['flows']) if data['flows'] is not None else None
                pressures_cal_zoom = pressures_cal[zoom_mask] if pressures_cal is not None else None
                flows_cal_zoom = flows_cal[zoom_mask] if flows_cal is not None else None
                calibrated_results_zoom[jtype] = {
                    'times': times_cal_zoom,
                    'pressures': pressures_cal_zoom,
                    'flows': flows_cal_zoom
                }
    
    # Plot 1: Zoomed pressure (top)
    ax = axes[0]
    plot_pressure_data(ax, zoom_times, pressures_3d_zoom, zoom_times, pressures_geo_zoom, 
                      calibrated_results_zoom, junction_styles, set_ylim_from_3d=False)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24)
    ax.set_xlim(zoom_times[0] if len(zoom_times) > 0 else None, zoom_times[-1] if len(zoom_times) > 0 else None)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed flow
    ax = axes[1]
    plot_flow_data(ax, zoom_times, flows_3d_zoom, zoom_times, flows_geo_zoom, 
                   calibrated_results_zoom, junction_styles, set_ylim_from_3d=False)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    # xlim set automatically via sharex with axes[0]
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Full pressure
    ax = axes[2]
    # Add shaded region for zoom window (behind data) - do this first
    if len(zoom_times) > 0:
        ax.axvspan(time_zoom_start, time_zoom_end, alpha=0.4, color='gray', zorder=0)
    plot_pressure_data(ax, times_3d_sec, pressures_3d_mmhg, times_geo, pressures_geo_mmhg, 
                      calibrated_results, junction_styles, set_ylim_from_3d=True)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Full flow (bottom)
    ax = axes[3]
    # Add shaded region for zoom window (behind data) - do this first
    if len(zoom_times) > 0:
        ax.axvspan(time_zoom_start, time_zoom_end, alpha=0.4, color='gray', zorder=0)
    plot_flow_data(ax, times_3d_sec, flows_3d, times_geo, flows_geo, 
                  calibrated_results, junction_styles, set_ylim_from_3d=True)
    ax.set_xlabel(r'Time (s)', fontsize=24)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    ax.grid(True, alpha=0.3)
    
    # Create single legend above the top subplot title
    # Collect handles and labels from the full-range pressure plot (has all series)
    handles, labels = axes[2].get_legend_handles_labels()
    # Remove duplicates while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    
    # Apply tight layout first
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space at top for legend
    
    # Create figure-level legend above the top subplot title
    # Position it above the top subplot title, arranged in 2 rows of 3 entries
    fig.legend(unique_handles, unique_labels, loc='upper center', ncol=3, 
               bbox_to_anchor=(0.5, 0.99), fontsize=24, frameon=True)
    
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
                       default=['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction'],
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

