#!/usr/bin/env python3
"""
Script to plot flow and pressure observations from calibration input JSON.

Plots observations for inlet and outlet of any specified vessel.
"""

import os
import sys
import argparse
import json
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
    
    # Set font sizes
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 32
    
    HAS_MATPLOTLIB = True
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def extract_inlet_observations_from_calibration_input(calibration_input_path, vessel_name):
    """
    Extract inlet pressure and flow observations from calibration input for a specific vessel.
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values (in dynes/cm^2)
        flows: Flow values (in cm³/s)
        inlet_label: Label for the inlet (e.g., 'INFLOW:branch0_seg0' or 'J0:branch1_seg0')
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None, None, None
    
    observations = calib_data['y']
    
    # Find inlet pressure and flow
    # Look for patterns:
    # 1. Root vessel inlet: "pressure:INFLOW:{vessel_name}" (e.g., "pressure:INFLOW:branch0_seg0")
    # 2. Junction outlet vessel inlet: "pressure:{junc_name}:{vessel_name}" (e.g., "pressure:J0:branch1_seg0")
    pressure_key = None
    flow_key = None
    inlet_label = None
    
    for key in observations.keys():
        if not key.startswith('pressure:'):
            continue
        
        parts = key.split(':')
        if len(parts) != 3:
            continue
        
        # Check pattern 1: "pressure:INFLOW:{vessel_name}" (root vessel inlet)
        if parts[1] == 'INFLOW' and parts[2] == vessel_name:
            pressure_key = key
            inlet_label = f'INFLOW:{vessel_name}'
            flow_key = f'flow:INFLOW:{vessel_name}'
            if flow_key not in observations:
                flow_key = None
            break
        # Check pattern 2: "pressure:{junc_name}:{vessel_name}" (junction outlet vessel inlet)
        elif parts[2] == vessel_name and parts[1].startswith('J'):
            pressure_key = key
            inlet_label = f'{parts[1]}:{vessel_name}'
            flow_key = f'flow:{parts[1]}:{vessel_name}'
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
    
    return times, pressures, flows, inlet_label


def extract_outlet_observations_from_calibration_input(calibration_input_path, vessel_name):
    """
    Extract outlet pressure and flow observations from calibration input for a specific vessel.
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values (in dynes/cm^2)
        flows: Flow values (in cm³/s)
        outlet_label: Label for the outlet (e.g., 'branch0_seg0:J0' or 'RESISTANCE_0')
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
            # If it's a junction (starts with 'J'), label should be {vessel_name}:{junc_name}
            if parts[2].startswith('J'):
                outlet_label = f'{vessel_name}:{parts[2]}'
            else:
                # It's a BC, just use the BC name
                outlet_label = parts[2]
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


def plot_calibration_observations(calibration_input_path, vessel_name, output_path, time_period=None):
    """
    Plot flow and pressure observations from calibration input for a specific vessel.
    
    Args:
        calibration_input_path: Path to calibration input JSON
        vessel_name: Name of vessel to plot (e.g., 'branch0_seg0')
        output_path: Path to save plot
        time_period: Time period in seconds (for converting normalized time to seconds)
    """
    print(f"Reading calibration input from: {calibration_input_path}")
    print(f"Extracting observations for vessel: {vessel_name}")
    
    # Extract inlet observations
    inlet_times, inlet_pressures, inlet_flows, inlet_label = \
        extract_inlet_observations_from_calibration_input(calibration_input_path, vessel_name)
    
    # Extract outlet observations
    outlet_times, outlet_pressures, outlet_flows, outlet_label = \
        extract_outlet_observations_from_calibration_input(calibration_input_path, vessel_name)
    
    if inlet_times is None and outlet_times is None:
        print(f"Error: No observations found for vessel {vessel_name} in calibration input")
        return False
    
    if inlet_times is None:
        print(f"  Warning: No inlet observations found for {vessel_name}")
    if outlet_times is None:
        print(f"  Warning: No outlet observations found for {vessel_name}")
    
    # Convert normalized time to seconds if time_period is provided
    if time_period is not None:
        if inlet_times is not None:
            inlet_times = inlet_times * time_period
        if outlet_times is not None:
            outlet_times = outlet_times * time_period
        time_label = 'Time (s)'
    else:
        time_label = 'Time (normalized)'
    
    # Convert pressure from dynes/cm^2 to mmHg
    if inlet_pressures is not None:
        inlet_pressures_mmhg = inlet_pressures / 1333.0
    else:
        inlet_pressures_mmhg = None
    
    if outlet_pressures is not None:
        outlet_pressures_mmhg = outlet_pressures / 1333.0
    else:
        outlet_pressures_mmhg = None
    
    # Create figure with 2 subplots: pressure (overlaid) and flow (overlaid)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Pressure (both inlet and outlet overlaid)
    ax = axes[0]
    
    # Plot inlet pressure
    if inlet_pressures_mmhg is not None and inlet_times is not None:
        inlet_label_display = inlet_label if inlet_label else f'Inlet ({vessel_name})'
        ax.plot(inlet_times, inlet_pressures_mmhg, 'b-', linewidth=3, label=inlet_label_display, alpha=1)
        print(f"  Inlet pressure: {len(inlet_pressures_mmhg)} points")
        print(f"    Range: [{np.min(inlet_pressures_mmhg):.2f}, {np.max(inlet_pressures_mmhg):.2f}] mmHg")
        if inlet_label:
            print(f"    Label: {inlet_label}")
    
    # Plot outlet pressure (dotted red line)
    if outlet_pressures_mmhg is not None and outlet_times is not None:
        outlet_label_display = outlet_label if outlet_label else f'Outlet ({vessel_name})'
        ax.plot(outlet_times, outlet_pressures_mmhg, 'r--', linewidth=3, label=outlet_label_display, alpha=1, linestyle='--')
        print(f"  Outlet pressure: {len(outlet_pressures_mmhg)} points")
        print(f"    Range: [{np.min(outlet_pressures_mmhg):.2f}, {np.max(outlet_pressures_mmhg):.2f}] mmHg")
        if outlet_label:
            print(f"    Label: {outlet_label}")
    
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=18)
    
    # Set y-axis ticks based on combined range
    if (inlet_pressures_mmhg is not None or outlet_pressures_mmhg is not None):
        all_pressures = []
        if inlet_pressures_mmhg is not None:
            all_pressures.extend(inlet_pressures_mmhg)
        if outlet_pressures_mmhg is not None:
            all_pressures.extend(outlet_pressures_mmhg)
        if all_pressures:
            y_min = np.min(all_pressures)
            y_max = np.max(all_pressures)
            y_range = y_max - y_min
            # Add 10% padding
            y_min = y_min - 0.1 * y_range
            y_max = y_max + 0.1 * y_range
            ax.set_ylim(y_min, y_max)
            num_ticks = max(8, min(12, int(y_range / 2) + 1))
            y_ticks = np.linspace(y_min, y_max, num_ticks)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.2f}' for y in y_ticks], fontsize=18)
    
    # Plot 2: Flow (both inlet and outlet overlaid)
    ax = axes[1]
    
    # Plot inlet flow
    if inlet_flows is not None and inlet_times is not None:
        inlet_label_display = inlet_label if inlet_label else f'Inlet ({vessel_name})'
        ax.plot(inlet_times, inlet_flows, 'b-', linewidth=3, label=inlet_label_display, alpha=1)
        print(f"  Inlet flow: {len(inlet_flows)} points")
        print(f"    Range: [{np.min(inlet_flows):.2f}, {np.max(inlet_flows):.2f}] cm³/s")
    
    # Plot outlet flow (dotted red line)
    if outlet_flows is not None and outlet_times is not None:
        outlet_label_display = outlet_label if outlet_label else f'Outlet ({vessel_name})'
        ax.plot(outlet_times, outlet_flows, 'r--', linewidth=3, label=outlet_label_display, alpha=1, linestyle='--')
        print(f"  Outlet flow: {len(outlet_flows)} points")
        print(f"    Range: [{np.min(outlet_flows):.2f}, {np.max(outlet_flows):.2f}] cm³/s")
    
    ax.set_xlabel(time_label, fontsize=24)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=18)
    
    # Set y-axis ticks based on combined range
    if (inlet_flows is not None or outlet_flows is not None):
        all_flows = []
        if inlet_flows is not None:
            all_flows.extend(inlet_flows)
        if outlet_flows is not None:
            all_flows.extend(outlet_flows)
        if all_flows:
            y_min = np.min(all_flows)
            y_max = np.max(all_flows)
            y_range = y_max - y_min
            # Add 10% padding
            y_min = y_min - 0.1 * y_range
            y_max = y_max + 0.1 * y_range
            ax.set_ylim(y_min, y_max)
            num_ticks = max(8, min(12, int(y_range / 2) + 1))
            y_ticks = np.linspace(y_min, y_max, num_ticks)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.2f}' for y in y_ticks], fontsize=18)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()
    return True


def get_time_period(set_name, geo_name):
    """
    Try to determine time period from simulation XML or other sources.
    
    Returns:
        Time period in seconds, or None if not found
    """
    # Try to find XML file
    xml_paths = [
        os.path.join('data', 'threeD', set_name, geo_name, 'fluid_simulation_0-0.xml'),
        os.path.join('data', 'threeD', set_name, geo_name, 'fluid_simulation.xml'),
    ]
    
    for xml_path in xml_paths:
        if os.path.exists(xml_path):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Look for time step and number of steps
                step_size_elem = root.find('.//step_size')
                num_time_steps_elem = root.find('.//num_time_steps')
                
                if step_size_elem is not None and num_time_steps_elem is not None:
                    step_size = float(step_size_elem.text)
                    num_steps = int(num_time_steps_elem.text)
                    time_period = step_size * num_steps
                    print(f"  Found time period from XML: {time_period:.4f} s")
                    return time_period
            except Exception as e:
                print(f"  Warning: Could not parse XML for time period: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot flow and pressure observations from calibration input JSON"
    )
    parser.add_argument('--calibration-input', type=str, required=True,
                       help='Path to calibration input JSON file')
    parser.add_argument('--vessel-name', type=str, required=True,
                       help='Vessel name to plot (e.g., branch0_seg0, branch1_seg0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: auto-generate from input path)')
    parser.add_argument('--set-name', type=str, default=None,
                       help='Set name (for auto-detecting time period)')
    parser.add_argument('--geo-name', type=str, default=None,
                       help='Geometry name (for auto-detecting time period)')
    parser.add_argument('--time-period', type=float, default=None,
                       help='Time period in seconds (for converting normalized time to seconds)')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        print("  Install with: pip install matplotlib")
        sys.exit(1)
    
    if not os.path.exists(args.calibration_input):
        print(f"Error: Calibration input file not found: {args.calibration_input}")
        sys.exit(1)
    
    # Determine time period
    time_period = args.time_period
    if time_period is None and args.set_name and args.geo_name:
        time_period = get_time_period(args.set_name, args.geo_name)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate from input path
        base_dir = os.path.dirname(args.calibration_input)
        base_name = os.path.basename(args.calibration_input).replace('.json', '')
        output_path = os.path.join(base_dir, f"{args.vessel_name}_observations.png")
    
    # Create plot
    success = plot_calibration_observations(args.calibration_input, args.vessel_name, output_path, time_period)
    
    if success:
        print(f"\n✓ Successfully created plot: {output_path}")
        sys.exit(0)
    else:
        print("\n✗ Failed to create plot")
        sys.exit(1)


if __name__ == '__main__':
    main()

