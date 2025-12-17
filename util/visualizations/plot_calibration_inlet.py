#!/usr/bin/env python3
"""
Script to plot inlet flow and pressure waveforms from calibration input files.
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


def extract_inlet_bc_from_calibration_input(calibration_input_path):
    """
    Extract inlet boundary condition (flow and pressure) from calibration input.
    
    Returns:
        bc_times: Time array (in seconds)
        bc_flows: Flow values (in cm³/s)
        obs_times: Observation time array (normalized [0, 1])
        obs_pressures: Pressure observations (in dynes/cm^2)
        obs_flows: Flow observations (in cm³/s)
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Extract BC from boundary_conditions
    bc_times = None
    bc_flows = None
    for bc in calib_data.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc_values = bc.get('bc_values', {})
            bc_times = bc_values.get('t', [])
            bc_flows = bc_values.get('Q', [])
            break
    
    # Extract observations
    obs_pressures = None
    obs_flows = None
    obs_times = None
    
    # Check for full observations first (for plotting)
    if '_full_observations' in calib_data and 'y' in calib_data['_full_observations']:
        observations = calib_data['_full_observations']['y']
        print("  Using full observations from calibration input")
    elif 'y' in calib_data:
        observations = calib_data['y']
        print("  Using calibration observations (may be second half only)")
    else:
        observations = {}
    
    # Find inlet pressure and flow observations
    pressure_key = None
    flow_key = None
    
    for key in observations.keys():
        if key.startswith('pressure:INFLOW:'):
            pressure_key = key
        elif key.startswith('flow:INFLOW:'):
            flow_key = key
    
    if pressure_key or flow_key:
        # Get number of observations
        if pressure_key:
            num_obs = len(observations[pressure_key])
            obs_pressures = np.array(observations[pressure_key])
        elif flow_key:
            num_obs = len(observations[flow_key])
        
        if flow_key:
            obs_flows = np.array(observations[flow_key])
        
        # Create normalized time array for observations
        obs_times = np.linspace(0.0, 1.0, num_obs)
    
    # Check for full BC for forward simulation
    full_bc = None
    if '_full_bc_for_forward_sim' in calib_data:
        full_bc = calib_data['_full_bc_for_forward_sim']
        print("  Found full BC for forward simulation")
    
    return bc_times, bc_flows, obs_times, obs_pressures, obs_flows, full_bc


def plot_calibration_inlet(calibration_input_path, output_path=None):
    """
    Plot inlet flow and pressure waveforms from calibration input.
    
    Args:
        calibration_input_path: Path to calibration input JSON
        output_path: Path to save plot (optional, auto-generates if None)
    """
    print("="*60)
    print("Plotting Calibration Inlet Waveforms")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    # Extract data
    print(f"\nReading calibration input from: {calibration_input_path}")
    bc_times, bc_flows, obs_times, obs_pressures, obs_flows, full_bc = extract_inlet_bc_from_calibration_input(calibration_input_path)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot flow (top subplot)
    ax = axes[0]
    
    if bc_times and bc_flows:
        ax.plot(bc_times, bc_flows, 'b-', linewidth=3, label='BC Flow (calibration)', alpha=0.8, marker='o', markersize=4)
        print(f"  BC Flow: {len(bc_times)} points, range: [{min(bc_flows):.2f}, {max(bc_flows):.2f}] cm³/s")
    
    if full_bc and 't' in full_bc and 'Q' in full_bc:
        full_times = full_bc['t']
        full_flows = full_bc['Q']
        ax.plot(full_times, full_flows, 'g--', linewidth=2, label='Full BC Flow (forward sim)', alpha=0.7)
        print(f"  Full BC Flow: {len(full_times)} points, range: [{min(full_flows):.2f}, {max(full_flows):.2f}] cm³/s")
    
    if obs_times is not None and obs_flows is not None:
        # Convert normalized times to actual times if we have BC times
        if bc_times and len(bc_times) > 1:
            time_min = bc_times[0]
            time_max = bc_times[-1]
            obs_times_actual = time_min + obs_times * (time_max - time_min)
        else:
            obs_times_actual = obs_times
        
        ax.plot(obs_times_actual, obs_flows, 'r-', linewidth=2, label='Observation Flow', alpha=0.7, linestyle=':')
        print(f"  Observation Flow: {len(obs_flows)} points, range: [{min(obs_flows):.2f}, {max(obs_flows):.2f}] cm³/s")
    
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=28)
    ax.set_title(r'Inlet Flow vs Time', fontsize=32, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20, loc='best')
    
    # Plot pressure (bottom subplot)
    ax = axes[1]
    
    if obs_times is not None and obs_pressures is not None:
        # Convert to mmHg
        obs_pressures_mmhg = obs_pressures / 1333.0
        
        # Convert normalized times to actual times if we have BC times
        if bc_times and len(bc_times) > 1:
            time_min = bc_times[0]
            time_max = bc_times[-1]
            obs_times_actual = time_min + obs_times * (time_max - time_min)
        else:
            obs_times_actual = obs_times
        
        ax.plot(obs_times_actual, obs_pressures_mmhg, 'r-', linewidth=2, label='Observation Pressure', alpha=0.7, linestyle=':')
        print(f"  Observation Pressure: {len(obs_pressures)} points, range: [{np.min(obs_pressures_mmhg):.2f}, {np.max(obs_pressures_mmhg):.2f}] mmHg")
    
    ax.set_xlabel(r'Time (s)', fontsize=28)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=28)
    ax.set_title(r'Inlet Pressure vs Time', fontsize=32, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        # Auto-generate output path
        base_dir = os.path.dirname(calibration_input_path)
        base_name = os.path.splitext(os.path.basename(calibration_input_path))[0]
        output_path = os.path.join(base_dir, f'{base_name}_inlet_waveforms.png')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot inlet flow and pressure waveforms from calibration input files"
    )
    parser.add_argument('--calibration-input', required=True, 
                       help='Path to calibration input JSON file')
    parser.add_argument('--output', help='Output plot path (default: auto-generate)')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        print("  Install with: pip install matplotlib")
        sys.exit(1)
    
    if not os.path.exists(args.calibration_input):
        print(f"Error: Calibration input file not found: {args.calibration_input}")
        sys.exit(1)
    
    success = plot_calibration_inlet(args.calibration_input, args.output)
    
    if success:
        print(f"\n✓ Successfully created plot")
        sys.exit(0)
    else:
        print(f"\n✗ Failed to create plot")
        sys.exit(1)


if __name__ == '__main__':
    main()

