#!/usr/bin/env python3
"""
Plot observations at all locations of a junction.

Creates a plot for each junction showing flow and pressure observations
at the inlet and all outlet locations.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================================
# CONFIGURATION - Modify these values to customize the plots
# =============================================================================

# Figure settings
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 8

# Font settings
FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 10

# Line colors for different locations
LOCATION_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# Line width
LINE_WIDTH = 1.5

# =============================================================================


def find_calibration_input(set_name, geo_name, geometry_type='original'):
    """Find the calibration input file for the given set and geometry."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    zerod_dir = os.path.join(base_dir, 'data', 'zeroD', set_name, geo_name)
    
    if geometry_type == 'bifurcations':
        calibration_input_path = os.path.join(zerod_dir, 'bifurcations_calibration_input.json')
    else:
        calibration_input_path = os.path.join(zerod_dir, 'calibration_input.json')
    
    if os.path.exists(calibration_input_path):
        return calibration_input_path
    
    raise FileNotFoundError(f"Calibration input not found: {calibration_input_path}")


def find_geometric_input(set_name, geo_name, geometry_type='original'):
    """Find the geometric input file for the given set and geometry."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    zerod_dir = os.path.join(base_dir, 'data', 'zeroD', set_name, geo_name)
    
    if geometry_type == 'bifurcations':
        geometric_input_path = os.path.join(zerod_dir, 'bifurcations_geometric_input.json')
    else:
        geometric_input_path = os.path.join(zerod_dir, 'geometric_input.json')
    
    if os.path.exists(geometric_input_path):
        return geometric_input_path
    
    raise FileNotFoundError(f"Geometric input not found: {geometric_input_path}")


def get_junction_locations(calibration_data, junction_name):
    """
    Get all observation locations connected to a junction.
    
    Returns:
        dict with keys 'inlet' and 'outlets', each containing location strings
    """
    y_data = calibration_data.get('y', {})
    
    locations = {
        'inlet': None,
        'outlets': []
    }
    
    for key in y_data.keys():
        # Parse key format: type:location1:location2
        parts = key.split(':')
        if len(parts) != 3:
            continue
        
        obs_type, loc1, loc2 = parts
        if obs_type not in ['pressure', 'flow']:
            continue
        
        # Check if this location involves the junction
        if loc2 == junction_name:
            # This is an inlet to the junction (vessel:junction)
            location = f"{loc1}:{loc2}"
            if locations['inlet'] is None:
                locations['inlet'] = location
        elif loc1 == junction_name:
            # This is an outlet from the junction (junction:vessel)
            location = f"{loc1}:{loc2}"
            if location not in locations['outlets']:
                locations['outlets'].append(location)
    
    return locations


def get_all_junctions(geometric_data):
    """Get all junction names from the geometric input."""
    junctions = geometric_data.get('junctions', [])
    return [j.get('junction_name') for j in junctions if j.get('junction_name')]


def get_observation_data(calibration_data, location, obs_type):
    """
    Get observation data for a location.
    
    Args:
        calibration_data: Calibration input dictionary
        location: Location string (e.g., 'branch0_seg0:J0')
        obs_type: 'pressure' or 'flow'
    
    Returns:
        numpy array of observation values, or None if not found
    """
    y_data = calibration_data.get('y', {})
    key = f"{obs_type}:{location}"
    
    if key in y_data:
        return np.array(y_data[key])
    
    return None


def get_time_array(calibration_data):
    """Extract time array from calibration data."""
    # Try to get time from boundary conditions
    bcs = calibration_data.get('boundary_conditions', [])
    for bc in bcs:
        bc_values = bc.get('bc_values', {})
        if 't' in bc_values:
            return np.array(bc_values['t'])
    
    # Fallback: use number of time points from first observation
    y_data = calibration_data.get('y', {})
    for key, values in y_data.items():
        n_points = len(values)
        return np.linspace(0, 1, n_points)
    
    return None


def plot_junction_observations(calibration_data, junction_name, time_array, output_path=None):
    """
    Plot observations at all locations of a junction.
    
    Creates a figure with 2 subplots:
    - Top: Flow at inlet and outlets
    - Bottom: Pressure at inlet and outlets
    """
    locations = get_junction_locations(calibration_data, junction_name)
    
    if locations['inlet'] is None and not locations['outlets']:
        print(f"  No observations found for junction {junction_name}")
        return None
    
    # Collect all locations
    all_locations = []
    location_labels = []
    
    if locations['inlet']:
        all_locations.append(locations['inlet'])
        inlet_parts = locations['inlet'].split(':')
        location_labels.append(f"Inlet: {inlet_parts[0]}")
    
    for outlet_loc in locations['outlets']:
        all_locations.append(outlet_loc)
        outlet_parts = outlet_loc.split(':')
        location_labels.append(f"Outlet: {outlet_parts[1]}")
    
    if not all_locations:
        print(f"  No locations found for junction {junction_name}")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True)
    
    # Plot flow (top subplot)
    ax_flow = axes[0]
    for i, (loc, label) in enumerate(zip(all_locations, location_labels)):
        flow_data = get_observation_data(calibration_data, loc, 'flow')
        if flow_data is not None:
            color = LOCATION_COLORS[i % len(LOCATION_COLORS)]
            ax_flow.plot(time_array, flow_data, color=color, linewidth=LINE_WIDTH, label=label)
    
    ax_flow.set_ylabel('Flow [mL/s]', fontsize=FONT_SIZE)
    ax_flow.set_title(f'Junction {junction_name} - Observations', fontsize=TITLE_FONT_SIZE)
    ax_flow.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
    ax_flow.grid(True, alpha=0.3)
    
    # Plot pressure (bottom subplot)
    ax_pressure = axes[1]
    for i, (loc, label) in enumerate(zip(all_locations, location_labels)):
        pressure_data = get_observation_data(calibration_data, loc, 'pressure')
        if pressure_data is not None:
            color = LOCATION_COLORS[i % len(LOCATION_COLORS)]
            # Convert to mmHg for better readability
            pressure_mmHg = pressure_data / 133.322  # Pa to mmHg
            ax_pressure.plot(time_array, pressure_mmHg, color=color, linewidth=LINE_WIDTH, label=label)
    
    ax_pressure.set_xlabel('Time [s]', fontsize=FONT_SIZE)
    ax_pressure.set_ylabel('Pressure [mmHg]', fontsize=FONT_SIZE)
    ax_pressure.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
    ax_pressure.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot junction observations')
    parser.add_argument('--set-name', required=True, help='Dataset name (e.g., VMR)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., 0063_1001)')
    parser.add_argument('--geometry-type', choices=['original', 'bifurcations'], default='original',
                        help='Geometry type to plot')
    parser.add_argument('--junction', help='Specific junction name to plot (plots all if not specified)')
    parser.add_argument('--output-dir', help='Output directory (default: results/junction_observations/)')
    
    args = parser.parse_args()
    
    # Find input files
    try:
        calibration_input_path = find_calibration_input(args.set_name, args.geo_name, args.geometry_type)
        geometric_input_path = find_geometric_input(args.set_name, args.geo_name, args.geometry_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Loading calibration input: {calibration_input_path}")
    with open(calibration_input_path, 'r') as f:
        calibration_data = json.load(f)
    
    print(f"Loading geometric input: {geometric_input_path}")
    with open(geometric_input_path, 'r') as f:
        geometric_data = json.load(f)
    
    # Get time array
    time_array = get_time_array(calibration_data)
    if time_array is None:
        print("Error: Could not determine time array")
        sys.exit(1)
    
    # Determine output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, 'results', 'junction_observations', 
                                   args.set_name, args.geo_name, args.geometry_type)
    
    # Get junctions to plot
    if args.junction:
        junction_names = [args.junction]
    else:
        junction_names = get_all_junctions(geometric_data)
    
    print(f"\nPlotting observations for {len(junction_names)} junction(s)...")
    
    for junction_name in junction_names:
        print(f"\nProcessing junction: {junction_name}")
        output_path = os.path.join(output_dir, f"{junction_name}_observations.png")
        plot_junction_observations(calibration_data, junction_name, time_array, output_path)
    
    print(f"\nDone! Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

