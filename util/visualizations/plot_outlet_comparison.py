#!/usr/bin/env python3
"""
Script to plot outlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models
for each vessel in the network.
"""

import os
import sys
import argparse
import json
import csv
import numpy as np
import warnings

# Suppress matplotlib warnings about redundant linestyle
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*linestyle.*redundantly defined.*')
warnings.filterwarnings('ignore', message='.*linestyle.*keyword argument.*')

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
    
    # Set font sizes
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
    Extract outlet pressure and flow data from 0D CSV results for a specific vessel.
    
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


def extract_outlet_data_from_calibration_input(calibration_input_path, vessel_name):
    """
    Extract outlet pressure and flow data from calibration input observations for a specific vessel.
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values (in dynes/cm^2, will be converted to mmHg)
        flows: Flow values (in cm³/s)
        outlet_label: Label for the outlet (BC name or junction name)
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None, None, None
    
    observations = calib_data['y']
    
    # Find outlet pressure and flow for this vessel
    # Outlet: vessel name is the SECOND argument (e.g., "pressure:branch2_seg0:J1")
    # Inlet: vessel name is the THIRD argument (e.g., "pressure:J0:branch2_seg0")
    pressure_key = None
    flow_key = None
    outlet_label = None
    
    for key in observations.keys():
        if not key.startswith('pressure:'):
            continue
        
        parts = key.split(':')
        if len(parts) != 3:
            continue
        
        # Outlet: "pressure:{vessel_name}:{bc_name}" or "pressure:{vessel_name}:{junc_name}"
        # Vessel name must be the second argument (parts[1])
        if parts[1] == vessel_name:
            pressure_key = key
            # Full second part of the key (parts[1]:parts[2]) for title
            outlet_label = f'{parts[1]}:{parts[2]}'
            # Find corresponding flow key
            flow_key = f'flow:{vessel_name}:{parts[2]}'
            if flow_key not in observations:
                flow_key = None
            break
    
    if pressure_key is None:
        return None, None, None, None
    
    # Get number of observations
    num_obs = len(observations[pressure_key])
    
    times = np.linspace(0.0, 1.0, num_obs)
    
    # Extract pressure (in dynes/cm^2)
    pressures = np.array(observations[pressure_key])
    
    # Extract flow
    if flow_key and flow_key in observations:
        flows = np.array(observations[flow_key])
    else:
        flows = None
    
    return times, pressures, flows, outlet_label


def get_all_vessels_from_csv(csv_path):
    """
    Get list of all vessel names from CSV file.
    
    Returns:
        List of vessel names
    """
    results, _ = read_zerod_csv(csv_path)
    return sorted(results.keys())


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


def is_terminal_vessel(vessel_name, geometric_input_path=None, calibration_input_path=None):
    """
    Check if a vessel is a terminal vessel (ends in a boundary condition).
    
    Args:
        vessel_name: Name of the vessel (e.g., 'branch1_seg0')
        geometric_input_path: Path to geometric input JSON (optional)
        calibration_input_path: Path to calibration input JSON (optional, used as fallback)
    
    Returns:
        True if vessel is terminal (has outlet BC), False otherwise
    """
    # Try geometric input first
    if geometric_input_path and os.path.exists(geometric_input_path):
        try:
            with open(geometric_input_path, 'r') as f:
                data = json.load(f)
            vessels = data.get('vessels', [])
            for vessel in vessels:
                if vessel.get('vessel_name') == vessel_name:
                    if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
                        return True
        except Exception:
            pass
    
    # Try calibration input as fallback
    if calibration_input_path and os.path.exists(calibration_input_path):
        try:
            with open(calibration_input_path, 'r') as f:
                data = json.load(f)
            vessels = data.get('vessels', [])
            for vessel in vessels:
                if vessel.get('vessel_name') == vessel_name:
                    if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
                        return True
        except Exception:
            pass
    
    return False


def plot_outlet_comparison(calibration_input_path, geometric_csv_path, calibrated_csv_paths,
                          vessel_name, output_path, set_name=None, geo_name=None, time_period=None,
                          geometric_input_path=None, zoom_start_idx=None, zoom_end_idx=None, verbose=False):
    """
    Plot outlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models
    for a specific vessel.
    
    Args:
        calibration_input_path: Path to calibration input JSON (for 3D observations)
        geometric_csv_path: Path to geometric 0D results CSV
        calibrated_csv_paths: Dictionary mapping junction type names to CSV paths, or single CSV path
        vessel_name: Name of the vessel to plot
        output_path: Path to save plot
        set_name: Set name (for finding time period)
        geo_name: Geometry name (for finding time period)
        time_period: Time period in seconds (if None, will try to find from XML)
    """
    if verbose:
        print(f"Plotting outlet comparison for {vessel_name}...")
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    
    # Get time period
    if time_period is None and set_name is not None and geo_name is not None:
        time_period = get_time_period(set_name, geo_name)
    
    if time_period is None:
        if verbose:
            print("  Warning: Could not determine time period, using default 1.0 s")
        time_period = 1.0
    else:
        if verbose:
            print(f"  Time period: {time_period:.4f} s")
    
    # Extract geometric 0D results
    if verbose:
        print(f"\nExtracting geometric 0D results for {vessel_name}...")
    times_geo, pressures_geo, flows_geo = extract_outlet_data_from_csv(geometric_csv_path, vessel_name)
    if times_geo is None:
        print(f"  Error: Could not extract geometric 0D results for {vessel_name}")
        return False
    
    # Convert pressure to mmHg
    if pressures_geo is not None:
        pressures_geo_mmhg = pressures_geo / 1333.0
    else:
        pressures_geo_mmhg = None

    
    # Extract 3D observations (from calibration input)
    if verbose:
        print(f"\nExtracting 3D observations for {vessel_name}...")
    times_3d_norm, pressures_3d, flows_3d, outlet_label = extract_outlet_data_from_calibration_input(
        calibration_input_path, vessel_name)
    
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
        

    else:
        if verbose:
            print(f"  Warning: Could not extract 3D observations for {vessel_name}")
        pressures_3d_mmhg = None
        flows_3d = None
        times_3d_sec = times_geo
    
    # Extract calibrated 0D results (if available)
    calibrated_results = {}
    
    if calibrated_csv_paths is None:
        if verbose:
            print("\nNote: No calibrated CSV files provided")
    elif isinstance(calibrated_csv_paths, dict):
        # Multiple junction types
        if verbose:
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
                    
                    if verbose:
                        print(f"  {jtype}: Found {len(times_cal)} time points")
    else:
        # Single CSV path (backward compatibility)
        if verbose:
            print("  Warning: Could not extract calibrated 0D results")
    
    # Create plot with 4 subplots: 2 zoomed (top) and 2 full range (bottom)
    if verbose:
        print("\nCreating plot...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    # Share x-axis within zoomed plots and within full plots
    axes[0].sharex(axes[1])
    axes[2].sharex(axes[3])
    # Ensure x-axis ticks are visible on flow plots (bottom plots in each pair)
    axes[1].tick_params(labelbottom=True)  # Show x-axis ticks on zoomed flow plot
    axes[3].tick_params(labelbottom=True)  # Show x-axis ticks on full flow plot
    
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
    
    # Use provided range or default to hardcoded values (599-699)
    if zoom_start_idx is None:
        zoom_start_idx = 599
    if zoom_end_idx is None:
        zoom_end_idx = 699
    
    # Validate and adjust zoom window if needed
    zoom_start_idx = max(0, min(zoom_start_idx, num_time_steps - 1))
    zoom_end_idx = min(zoom_end_idx, num_time_steps)
    
    # Ensure valid range
    if zoom_start_idx >= zoom_end_idx:
        # Fallback to last 105 timesteps if invalid range
        zoom_start_idx = max(0, num_time_steps - 105)
        zoom_end_idx = num_time_steps
    
    zoom_times = times_geo[zoom_start_idx:zoom_end_idx]
    
    # Helper function to plot pressure data on an axis
    def plot_pressure_data(ax, times_data, pressures_3d_data, times_geo_data, pressures_geo_data, 
                          calibrated_data_dict, junction_styles_dict, set_ylim_from_3d=True):
        """Plot pressure data on given axis"""
        # Plot 3D observations (black)
        if pressures_3d_data is not None:
            ax.plot(times_data, pressures_3d_data, 'k-', linestyle='-', linewidth=8, label='3D Model', alpha=1)
        
        # Plot geometric 0D (green dotted line)
        if pressures_geo_data is not None:
            ax.plot(times_geo_data, pressures_geo_data, 'g', linestyle='--', linewidth=3, label='Geometric 0D', alpha=1)
        
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
        
        # Set more y-axis ticks and labels
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        # Create approximately 8-10 ticks
        num_ticks = max(8, min(12, int(y_range / 2) + 1))  # Adaptive number of ticks
        y_ticks = np.linspace(y_min, y_max, num_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks], fontsize=18)
    
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
        
        # Set more y-axis ticks and labels
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        # Create approximately 8-10 ticks
        num_ticks = max(8, min(12, int(y_range / 2) + 1))  # Adaptive number of ticks
        y_ticks = np.linspace(y_min, y_max, num_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks], fontsize=18)
    
    # Prepare zoomed data (last 200 time steps)
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
    ax.set_xticklabels([])  # Remove x-axis labels (shares axis with flow plot below)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)  # Remove x-axis ticks and labels
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Plot 2: Zoomed flow
    ax = axes[1]
    plot_flow_data(ax, zoom_times, flows_3d_zoom, zoom_times, flows_geo_zoom, 
                   calibrated_results_zoom, junction_styles, set_ylim_from_3d=False)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    # Set x-axis limits to match zoom data range
    if len(zoom_times) > 0:
        ax.set_xlim(zoom_times[0], zoom_times[-1])
    # No x-axis label for zoomed plot (only on full plot)
    # Force x-axis tick labels to be visible (shared axes can hide them)
    ax.tick_params(axis='x', labelbottom=True, bottom=True, labelsize=20)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Collect all time values to determine full data range
    all_times = []
    if times_3d_sec is not None:
        all_times.extend(times_3d_sec)
    if times_geo is not None:
        all_times.extend(times_geo)
    for data in calibrated_results.values():
        if data['times'] is not None:
            all_times.extend(data['times'])
    
    # Determine full time range
    if len(all_times) > 0:
        time_min = np.min(all_times)
        time_max = np.max(all_times)
    else:
        time_min = times_geo[0] if len(times_geo) > 0 else 0.0
        time_max = times_geo[-1] if len(times_geo) > 0 else 1.0
    
    # Plot 3: Full pressure
    ax = axes[2]
    # Add shaded region for zoom window (behind data) - do this first
    if len(zoom_times) > 0:
        ax.axvspan(time_zoom_start, time_zoom_end, alpha=0.4, color='gray', zorder=0)
    plot_pressure_data(ax, times_3d_sec, pressures_3d_mmhg, times_geo, pressures_geo_mmhg, 
                      calibrated_results, junction_styles, set_ylim_from_3d=True)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24)
    # Set x-axis limits to match full data range
    ax.set_xlim(time_min, time_max)
    ax.set_xticklabels([])  # Remove x-axis labels (shares axis with flow plot below)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)  # Remove x-axis ticks and labels
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Plot 4: Full flow (bottom)
    ax = axes[3]
    # Add shaded region for zoom window (behind data) - do this first
    if len(zoom_times) > 0:
        ax.axvspan(time_zoom_start, time_zoom_end, alpha=0.4, color='gray', zorder=0)
    plot_flow_data(ax, times_3d_sec, flows_3d, times_geo, flows_geo, 
                  calibrated_results, junction_styles, set_ylim_from_3d=True)
    ax.set_xlabel(r'Time (s)', fontsize=24)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    # Set x-axis limits to match full data range
    ax.set_xlim(time_min, time_max)
    # Force x-axis tick labels to be visible (shared axes can hide them)
    ax.tick_params(axis='x', labelbottom=True, bottom=True)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
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
    
    # Check if vessel is terminal (has outlet BC)
    is_terminal = is_terminal_vessel(vessel_name, geometric_input_path, calibration_input_path)
    
    # Create title using the full second part of the 3D observation key
    if outlet_label:
        title_text = outlet_label
    else:
        title_text = vessel_name
        if is_terminal:
            title_text += " - terminal outlet (BC)"
    
    # Apply tight layout first
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space at top for legend and title
    
    # Add title above legend
    fig.suptitle(title_text, fontsize=28, weight='bold', y=0.995)
    
    # Create figure-level legend above the top subplot title
    # Position it above the top subplot title, arranged in 2 rows of 3 entries
    # Position legend slightly below the title
    fig.legend(unique_handles, unique_labels, loc='upper center', ncol=3, 
               bbox_to_anchor=(0.5, 0.97), fontsize=24, frameon=True)
    
    # Ensure x-axis tick labels are visible on flow plots (after all plotting is done)
    # This is critical because shared axes can hide tick labels
    # Explicitly show tick labels and format them with font size 20
    for ax_idx in [1, 3]:
        axes[ax_idx].tick_params(axis='x', labelbottom=True, bottom=True, labelsize=20)
        axes[ax_idx].xaxis.set_tick_params(labelbottom=True)
        # Force tick labels to be visible by setting them explicitly
        axes[ax_idx].xaxis.set_visible(True)
        # Get current ticks and ensure labels are shown
        locs = axes[ax_idx].xaxis.get_majorticklocs()
        if len(locs) > 0:
            axes[ax_idx].xaxis.set_ticks(locs)
            axes[ax_idx].xaxis.set_ticklabels([f'{loc:.2f}' for loc in locs], fontsize=20)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot outlet pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models for each vessel"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_007)')
    parser.add_argument('--vessel-name', help='Specific vessel name to plot (default: plot all vessels)')
    parser.add_argument('--calibration-input', help='Path to calibration input JSON (default: auto-detect)')
    parser.add_argument('--geometric-csv', help='Path to geometric 0D results CSV (default: auto-detect)')
    parser.add_argument('--calibrated-csv', help='Path to calibrated 0D results CSV (default: auto-detect, supports multiple junction types)')
    parser.add_argument('--junction-types', nargs='+', 
                       default=['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to plot (default: all four types)')
    parser.add_argument('--output-dir', default='results/outlet_comparison', 
                        help='Output directory for plots (default: results/outlet_comparison)')
    parser.add_argument('--data-dir', default='data/zeroD', 
                        help='Data directory for input files (default: data/zeroD)')
    parser.add_argument('--time-period', type=float, default=None,
                        help='Time period in seconds (default: auto-detect from XML)')
    parser.add_argument('--zoom-start', type=int, default=None,
                        help='Start index for zoom window (shaded region). Default: last 105 timesteps')
    parser.add_argument('--zoom-end', type=int, default=None,
                        help='End index for zoom window (shaded region). Default: end of time series')
    
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
    
    # Get list of vessels
    if args.vessel_name:
        vessel_names = [args.vessel_name]
    else:
        vessel_names = get_all_vessels_from_csv(geometric_csv_path)
        print(f"\nFound {len(vessel_names)} vessels: {vessel_names}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each vessel
    success_count = 0
    for vessel_name in vessel_names:
        # Generate output path
        output_path = os.path.join(output_dir, f"{vessel_name}_outlet_comparison.png")
        

        
        # Find geometric input path
        geometric_input_path = os.path.join(data_dir, 'geometric_input.json')
        if not os.path.exists(geometric_input_path):
            geometric_input_path = None
        
        success = plot_outlet_comparison(
            calibration_input_path, geometric_csv_path, calibrated_csv_paths,
            vessel_name, output_path, set_name=args.set_name, geo_name=args.geo_name,
            time_period=args.time_period, geometric_input_path=geometric_input_path,
            zoom_start_idx=args.zoom_start, zoom_end_idx=args.zoom_end
        )
        
        if success:
            success_count += 1
    
    print(f"\n\n{'='*60}")
    print(f"Summary: Created {success_count}/{len(vessel_names)} outlet comparison plots")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    if success_count == len(vessel_names):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

