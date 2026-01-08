#!/usr/bin/env python3
"""
Script to plot 1D pressure as a function of distance along each vessel.

Reads the 1D centerline solution VTP file and creates a video animation showing
pressure vs distance for each vessel over time.
"""

import os
import sys
import argparse
import json
import numpy as np

# Check for matplotlib
HAS_MATPLOTLIB = False
plt = None
FuncAnimation = None
PillowWriter = None
FFMpegWriter = None
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    
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

# Add parent directory to path to import calibration_helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zerod_calibration'))
from calibration_helpers import read_centerline_vtp


def find_centerline_solution(set_name, geo_name):
    """Find 1D centerline solution VTP file in various possible locations."""
    possible_paths = [
        os.path.join('data', 'oneD', set_name, geo_name, 'unsteady_soln.vtp'),
        os.path.join('data', 'oneD', set_name, geo_name, 'unsteady_soln_5.vtp'),
        os.path.join('data', 'reduced_results', set_name, geo_name, 'unsteady_soln.vtp'),
        os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                     set_name, geo_name, 'unsteady_soln.vtp'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_geometric_input(set_name, geo_name):
    """Find geometric 0D input JSON file."""
    possible_paths = [
        os.path.join('data', 'zeroD', set_name, geo_name, 'geometric_input.json'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def get_branch_to_vessel_mapping(geometric_input_path):
    """
    Create mapping from branch index to vessel name.
    
    Returns:
        Dictionary mapping branch_index -> vessel_name (e.g., {0: 'branch0_seg0', 1: 'branch1_seg0'})
    """
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    branch_to_vessel = {}
    
    for vessel in vessels:
        vessel_name = vessel.get('vessel_name', '')
        # Parse vessel name like "branch0_seg0" to get branch index
        if vessel_name.startswith('branch'):
            try:
                branch_str = vessel_name.split('_')[0]  # "branch0"
                branch_idx = int(branch_str.replace('branch', ''))
                branch_to_vessel[branch_idx] = vessel_name
            except (ValueError, IndexError):
                continue
    
    return branch_to_vessel


def extract_pressure_along_vessel(centerline_data, branch_id, branch_idx):
    """
    Extract pressure and distance data along a specific vessel branch for all timesteps.
    Also extracts flow at the inlet (first point) of the vessel.
    
    Args:
        centerline_data: Dictionary with centerline data arrays
        branch_id: BranchId array
        branch_idx: Branch index to extract
    
    Returns:
        distances: Array of distances along vessel (in cm)
        pressures: Dictionary mapping timestep_key -> pressure array (in dynes/cm^2)
        timestep_keys: List of available timestep keys (sorted)
        inlet_flows: Dictionary mapping timestep_key -> flow value at inlet (in cm^3/s)
    """
    # Find points belonging to this branch
    branch_mask = (branch_id == branch_idx)
    branch_indices = np.where(branch_mask)[0]
    
    if len(branch_indices) == 0:
        return None, None, None, None
    
    # Get Path (distance along centerline) for this branch
    path = centerline_data.get('Path', None)
    if path is None:
        # Calculate path from points if not available
        points = centerline_data['Points']
        path = np.zeros(len(points))
        for i in range(1, len(points)):
            path[i] = path[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    # Get distances for this branch
    branch_path = path[branch_indices]
    
    # Sort by path to ensure proper ordering along vessel
    sort_indices = np.argsort(branch_path)
    branch_indices_sorted = branch_indices[sort_indices]
    distances = branch_path[sort_indices]
    
    # Find all pressure timestep keys
    pressure_keys = []
    for key in centerline_data.keys():
        if key.startswith('pressure_'):
            pressure_keys.append(key)
    
    # Find all flow/velocity timestep keys
    flow_keys = []
    for key in centerline_data.keys():
        if key.startswith('velocity_') or key.startswith('flow_'):
            flow_keys.append(key)
    
    # Sort timestep keys
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    pressure_keys.sort(key=extract_timestep)
    flow_keys.sort(key=extract_timestep)
    
    # Extract pressure for all timesteps
    pressures = {}
    for key in pressure_keys:
        if key in centerline_data:
            pressure_array = centerline_data[key]
            pressures[key] = pressure_array[branch_indices_sorted]
    
    # Extract flow at inlet (first point) for all timesteps
    # Map flow values using pressure keys (matching by timestep number)
    inlet_flows = {}
    inlet_idx = branch_indices_sorted[0]  # First point along the vessel (inlet)
    
    # Create mapping from timestep number to flow key
    flow_timestep_map = {}
    for key in flow_keys:
        if key in centerline_data:
            timestep_num = extract_timestep(key)
            flow_timestep_map[timestep_num] = key
    
    # Map flow values to pressure keys
    for pressure_key in pressure_keys:
        timestep_num = extract_timestep(pressure_key)
        if timestep_num in flow_timestep_map:
            flow_key = flow_timestep_map[timestep_num]
            flow_array = centerline_data[flow_key]
            # Get flow at inlet point (convert from velocity*area to flow if needed)
            # Flow data might be velocity or actual flow, we'll use it as-is
            inlet_flows[pressure_key] = flow_array[inlet_idx]
    
    return distances, pressures, pressure_keys, inlet_flows


def create_pressure_animation(centerline_soln_path, geometric_input_path, vessel_name, 
                              output_path, fps=10):
    """
    Create video animation of pressure vs distance along a specific vessel over time.
    
    Args:
        centerline_soln_path: Path to 1D centerline solution VTP
        geometric_input_path: Path to geometric input JSON
        vessel_name: Name of vessel to plot (e.g., 'branch0_seg0')
        output_path: Path to save video file
        fps: Frames per second for animation (default: 10)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Creating pressure animation for {vessel_name}...")
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    # Read centerline solution
    print(f"Reading centerline solution from: {centerline_soln_path}")
    centerline_data, _ = read_centerline_vtp(centerline_soln_path)
    
    # Get branch to vessel mapping
    branch_to_vessel = get_branch_to_vessel_mapping(geometric_input_path)
    
    # Find branch index for this vessel
    branch_idx = None
    for bid, vname in branch_to_vessel.items():
        if vname == vessel_name:
            branch_idx = bid
            break
    
    if branch_idx is None:
        print(f"Error: Could not find branch index for vessel {vessel_name}")
        return False
    
    print(f"  Vessel {vessel_name} corresponds to branch {branch_idx}")
    
    # Get branch ID array
    branch_id = centerline_data.get('BranchId', None)
    if branch_id is None:
        print("Error: BranchId not found in centerline data")
        return False
    
    # Extract pressure and distance data for all timesteps
    distances, pressures, pressure_keys, inlet_flows = extract_pressure_along_vessel(
        centerline_data, branch_id, branch_idx)
    
    if distances is None or pressures is None or not pressure_keys:
        print(f"Error: Could not extract data for vessel {vessel_name}")
        return False
    
    print(f"  Found {len(distances)} points along vessel")
    print(f"  Found {len(pressure_keys)} timesteps")
    print(f"  Distance range: [{np.min(distances):.4f}, {np.max(distances):.4f}] cm")
    
    # Calculate pressure range for consistent y-axis
    all_pressures = []
    for key in pressure_keys:
        if key in pressures:
            all_pressures.extend(pressures[key])
    
    if not all_pressures:
        print("Error: No pressure data found")
        return False
    
    all_pressures = np.array(all_pressures)
    pressure_min = np.min(all_pressures) / 1333.0  # Convert to mmHg
    pressure_max = np.max(all_pressures) / 1333.0
    # Use actual data range for y-axis limits
    y_min = pressure_min
    y_max = pressure_max
    
    print(f"  Pressure range: [{pressure_min:.2f}, {pressure_max:.2f}] mmHg")
    
    # Calculate flow range for right y-axis
    flow_min = None
    flow_max = None
    if inlet_flows and len(inlet_flows) > 0:
        all_flows = [inlet_flows[key] for key in pressure_keys if key in inlet_flows]
        if all_flows:
            all_flows = np.array(all_flows)
            flow_min = np.min(all_flows)
            flow_max = np.max(all_flows)
            flow_range = flow_max - flow_min
            flow_y_min = flow_min - 0.1 * flow_range
            flow_y_max = flow_max + 0.1 * flow_range
            print(f"  Flow range: [{flow_min:.2f}, {flow_max:.2f}] cm³/s")
        else:
            inlet_flows = None  # No valid flow data
    else:
        inlet_flows = None  # No flow data available
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize line plot
    line, = ax.plot([], [], 'b-', linewidth=2, label='Pressure')
    
    # Set labels and title
    ax.set_xlabel(r'Distance along vessel (cm)', fontsize=24)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title(f'Pressure along {vessel_name}', fontsize=28, weight='bold')
    
    # Set axis limits
    ax.set_xlim(np.min(distances), np.max(distances))
    ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create second y-axis for flow (if available)
    ax2 = None
    flow_dot = None
    x_max = np.max(distances)
    
    if inlet_flows is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel(r'Flow (cm³/s)', fontsize=24, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        if flow_min is not None and flow_max is not None:
            ax2.set_ylim(flow_y_min, flow_y_max)
        
        # Initialize red dot at right edge of plot
        # Start with first flow value if available
        initial_flow = 0.0
        if pressure_keys and pressure_keys[0] in inlet_flows:
            initial_flow = inlet_flows[pressure_keys[0]]
        flow_dot, = ax2.plot([x_max], [initial_flow], 'ro', markersize=12, label='Inlet Flow')
    
    # Add timestep text
    timestep_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=18, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Animation function
    def animate(frame):
        if frame >= len(pressure_keys):
            if flow_dot is not None:
                return line, timestep_text, flow_dot
            return line, timestep_text
        
        timestep_key = pressure_keys[frame]
        pressure_data = pressures[timestep_key]
        
        # Convert pressure to mmHg
        pressure_mmhg = pressure_data / 1333.0
        
        # Update line data
        line.set_data(distances, pressure_mmhg)
        
        # Update timestep text
        timestep_num = timestep_key.split('_')[-1] if '_' in timestep_key else str(frame)
        timestep_text.set_text(f'Timestep: {timestep_num}')
        
        # Update flow dot position (if available)
        if flow_dot is not None and inlet_flows is not None and timestep_key in inlet_flows:
            flow_value = inlet_flows[timestep_key]
            flow_dot.set_data([x_max], [flow_value])
        
        if flow_dot is not None:
            return line, timestep_text, flow_dot
        return line, timestep_text
    
    # Create animation
    print(f"  Creating animation with {len(pressure_keys)} frames at {fps} fps...")
    
    # Try to use FFMpegWriter first (better quality), fall back to PillowWriter
    writer = None
    writer_name = None
    
    try:
        # Check if ffmpeg is available
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            writer = FFMpegWriter(fps=fps, bitrate=8000)
            writer_name = 'FFMpeg'
    except (FileNotFoundError, ImportError):
        pass
    
    if writer is None:
        try:
            writer = PillowWriter(fps=fps)
            writer_name = 'Pillow (GIF)'
        except ImportError:
            print("Error: No video writer available. Install ffmpeg or pillow.")
            return False
    
    print(f"  Using {writer_name} writer")
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(pressure_keys), 
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine file extension based on writer
    if writer_name == 'Pillow (GIF)':
        if not output_path.endswith('.gif'):
            output_path = output_path.replace('.mp4', '.gif')
    
    print(f"  Saving animation to: {output_path}")
    try:
        anim.save(output_path, writer=writer, dpi=150)
        print(f"✓ Animation saved to: {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        plt.close(fig)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create video animation of 1D pressure as a function of distance along each vessel"
    )
    parser.add_argument('--set-name', type=str, required=True,
                       help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', type=str, required=True,
                       help='Geometry name (e.g., tree_002)')
    parser.add_argument('--vessel-name', type=str, default=None,
                       help='Specific vessel name to plot (e.g., branch0_seg0). If not provided, creates videos for all vessels.')
    parser.add_argument('--centerline-soln', type=str, default=None,
                       help='Path to centerline solution VTP (default: auto-detect)')
    parser.add_argument('--geometric-input', type=str, default=None,
                       help='Path to geometric input JSON (default: auto-detect)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for animation (default: 10)')
    parser.add_argument('--output-dir', type=str, default='results/pressure_along_vessel',
                       help='Output directory for videos (default: results/pressure_along_vessel)')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        print("  Install with: pip install matplotlib")
        sys.exit(1)
    
    # Find centerline solution
    if args.centerline_soln:
        centerline_soln_path = args.centerline_soln
    else:
        centerline_soln_path = find_centerline_solution(args.set_name, args.geo_name)
    
    if centerline_soln_path is None:
        print(f"Error: Could not find centerline solution file for {args.set_name}/{args.geo_name}")
        print("  Searched in:")
        print("    - data/oneD/{set_name}/{geo_name}/")
        print("    - data/reduced_results/{set_name}/{geo_name}/")
        sys.exit(1)
    
    # Find geometric input
    if args.geometric_input:
        geometric_input_path = args.geometric_input
    else:
        geometric_input_path = find_geometric_input(args.set_name, args.geo_name)
    
    if geometric_input_path is None:
        print(f"Error: Could not find geometric input file for {args.set_name}/{args.geo_name}")
        print("  Expected: data/zeroD/{set_name}/{geo_name}/geometric_input.json")
        sys.exit(1)
    
    # Get list of vessels
    branch_to_vessel = get_branch_to_vessel_mapping(geometric_input_path)
    
    if args.vessel_name:
        if args.vessel_name not in branch_to_vessel.values():
            print(f"Error: Vessel {args.vessel_name} not found in geometric input")
            print(f"  Available vessels: {list(branch_to_vessel.values())}")
            sys.exit(1)
        vessel_names = [args.vessel_name]
    else:
        vessel_names = list(branch_to_vessel.values())
        print(f"\nFound {len(vessel_names)} vessels: {vessel_names}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create animation for each vessel
    success_count = 0
    for vessel_name in vessel_names:
        # Generate output path (will be .mp4 if ffmpeg available, .gif otherwise)
        output_path = os.path.join(output_dir, f"{vessel_name}_pressure_along_vessel.mp4")
        
        print(f"\n{'='*60}")
        print(f"Processing vessel: {vessel_name}")
        print(f"{'='*60}")
        
        success = create_pressure_animation(
            centerline_soln_path, geometric_input_path, vessel_name, output_path,
            fps=args.fps
        )
        
        if success:
            success_count += 1
    
    print(f"\n\n{'='*60}")
    print(f"Summary: Created {success_count}/{len(vessel_names)} pressure animations")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    if success_count == len(vessel_names):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
