#!/usr/bin/env python3
"""
Script to plot 1D pressure and flow at the inlet and 10 nodes in from the inlet versus time.
"""

import os
import sys
import argparse
import numpy as np
import xml.etree.ElementTree as ET

# Check for matplotlib
HAS_MATPLOTLIB = False
plt = None
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n
except ImportError:
    print("Error: VTK is required. Make sure VTK is installed and available.")
    sys.exit(1)


def read_1d_solution_vtp(vtp_path):
    """
    Read 1D solution VTP file and extract data.
    
    Returns:
        Dictionary with centerline data arrays
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    centerline = reader.GetOutput()
    
    # Extract point data arrays
    point_data = centerline.GetPointData()
    arrays = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        arrays[array.GetName()] = v2n(array)
    
    # Extract points
    points = v2n(centerline.GetPoints().GetData())
    arrays['Points'] = points
    
    # Extract connectivity
    cells = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        if cell.GetNumberOfPoints() == 2:  # Line segment
            cells.append([cell.GetPointId(0), cell.GetPointId(1)])
    arrays['Cells'] = cells
    
    return arrays, centerline


def find_inlet_and_10nodes_in(centerline_data, centerline_polydata):
    """
    Find the inlet point and the point 10 nodes in from the inlet.
    
    Returns:
        inlet_idx: Index of inlet point
        node_10_idx: Index of point 10 nodes in from inlet
    """
    gid = centerline_data.get('GlobalNodeId', None)
    branch_id = centerline_data.get('BranchId', None)
    cells = centerline_data.get('Cells', [])
    
    # Find inlet (GID == 0)
    inlet_idx = None
    if gid is not None:
        for i in range(len(gid)):
            if gid[i] == 0:
                inlet_idx = i
                break
    
    if inlet_idx is None:
        raise ValueError("Could not find inlet point (GID == 0)")
    
    # Build connectivity graph to find points along the centerline
    # Create adjacency list
    adj_list = {}
    for cell in cells:
        if len(cell) == 2:
            p0, p1 = cell[0], cell[1]
            if p0 not in adj_list:
                adj_list[p0] = []
            if p1 not in adj_list:
                adj_list[p1] = []
            adj_list[p0].append(p1)
            adj_list[p1].append(p0)
    
    # Find the point 10 nodes in from the inlet
    # Start from inlet and traverse along the centerline
    node_10_idx = None
    if inlet_idx in adj_list:
        current_idx = inlet_idx
        visited = {inlet_idx}
        path = [inlet_idx]
        
        # Traverse along the centerline starting from inlet
        for step in range(10):
            if current_idx not in adj_list:
                break
            
            # Find next node (prefer nodes with same branch_id if available)
            next_idx = None
            current_branch = branch_id[current_idx] if branch_id is not None else None
            
            for neighbor in adj_list[current_idx]:
                if neighbor not in visited:
                    # Prefer staying on the same branch
                    if current_branch is not None and branch_id is not None:
                        if branch_id[neighbor] == current_branch:
                            next_idx = neighbor
                            break
                    else:
                        next_idx = neighbor
                        break
            
            # If no same-branch neighbor, take any unvisited neighbor
            if next_idx is None:
                for neighbor in adj_list[current_idx]:
                    if neighbor not in visited:
                        next_idx = neighbor
                        break
            
            if next_idx is None:
                break
            
            visited.add(next_idx)
            path.append(next_idx)
            current_idx = next_idx
        
        if len(path) > 10:
            node_10_idx = path[10]
        elif len(path) > 1:
            # If we couldn't go 10 nodes, use the last node we reached
            node_10_idx = path[-1]
            print(f"  Warning: Could only traverse {len(path)-1} nodes from inlet, using node {node_10_idx}")
    
    if node_10_idx is None:
        raise ValueError("Could not find point 10 nodes in from inlet")
    
    return inlet_idx, node_10_idx


def extract_time_series(centerline_data, point_idx, pressure_timesteps, flow_timesteps):
    """
    Extract pressure and flow time series at a given point.
    
    Returns:
        times: Time array
        pressures: Pressure values over time
        flows: Flow values over time
    """
    # Extract timestep numbers
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    # Sort timesteps
    pressure_timesteps_sorted = sorted(pressure_timesteps, key=extract_timestep)
    flow_timesteps_sorted = sorted(flow_timesteps, key=extract_timestep)
    
    # Extract time array (assume uniform time steps, normalized to [0, 1])
    num_timesteps = len(pressure_timesteps_sorted)
    times = np.linspace(0.0, 1.0, num_timesteps)
    
    # Extract pressure and flow data at this point
    pressures = np.array([centerline_data[pt][point_idx] for pt in pressure_timesteps_sorted])
    
    # For flow, check if we have flow_ arrays or velocity_ arrays
    flows = None
    if flow_timesteps_sorted:
        # Try flow arrays first
        flow_arrays = [ft for ft in flow_timesteps_sorted if ft.startswith('flow_')]
        if flow_arrays:
            flows = np.array([centerline_data[ft][point_idx] for ft in flow_arrays])
        else:
            # Use velocity arrays and convert to flow (need area)
            velocity_arrays = [ft for ft in flow_timesteps_sorted if ft.startswith('velocity_')]
            if velocity_arrays:
                velocities = np.array([centerline_data[vt][point_idx] for vt in velocity_arrays])
                # Get area at this point (if available)
                area = centerline_data.get('Area', None)
                if area is not None and len(area) > point_idx:
                    point_area = area[point_idx]
                    flows = velocities * point_area
                else:
                    print(f"  Warning: No area data found, using velocity as flow proxy")
                    flows = velocities
    
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
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Look for time step size and number of time steps
                gen_params = root.find('General_Parameters')
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


def plot_1d_inlet_data(vtp_path, output_path=None, set_name=None, geo_name=None, time_period=None):
    """
    Plot 1D pressure and flow at inlet and 10 nodes in from inlet.
    
    Args:
        vtp_path: Path to 1D solution VTP file
        output_path: Path to save plot (optional)
        set_name: Set name (for finding time period)
        geo_name: Geometry name (for finding time period)
        time_period: Time period in seconds (if None, will try to find from XML)
    """
    print("="*60)
    print("Plotting 1D Inlet Data")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    # Read 1D solution
    print(f"\nReading 1D solution from: {vtp_path}")
    centerline_data, centerline_polydata = read_1d_solution_vtp(vtp_path)
    
    # Find pressure and flow timestep arrays
    pressure_timesteps = [key for key in centerline_data.keys() if key.startswith('pressure_')]
    flow_timesteps = [key for key in centerline_data.keys() if key.startswith('velocity_') or key.startswith('flow_')]
    
    if not pressure_timesteps:
        print("Error: No pressure timestep arrays found")
        return False
    
    if not flow_timesteps:
        print("Error: No flow/velocity timestep arrays found")
        return False
    
    print(f"  Found {len(pressure_timesteps)} pressure timesteps")
    print(f"  Found {len(flow_timesteps)} flow/velocity timesteps")
    
    # Get time period
    if time_period is None and set_name is not None and geo_name is not None:
        time_period = get_time_period(set_name, geo_name)
    
    if time_period is None:
        print("  Warning: Could not determine time period, using default 1.0 s")
        time_period = 1.0
    else:
        print(f"  Time period: {time_period:.4f} s")
    
    # Find inlet and 10 nodes in
    print("\nFinding inlet and 10 nodes in from inlet...")
    try:
        inlet_idx, node_10_idx = find_inlet_and_10nodes_in(centerline_data, centerline_polydata)
        print(f"  Inlet point index: {inlet_idx}")
        print(f"  10 nodes in point index: {node_10_idx}")
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Extract time series
    print("\nExtracting time series...")
    times_inlet, pressures_inlet, flows_inlet = extract_time_series(
        centerline_data, inlet_idx, pressure_timesteps, flow_timesteps
    )
    times_node10, pressures_node10, flows_node10 = extract_time_series(
        centerline_data, node_10_idx, pressure_timesteps, flow_timesteps
    )
    
    # Convert normalized time to seconds
    times_inlet_sec = times_inlet * time_period
    times_node10_sec = times_node10 * time_period
    
    # Convert pressure from dynes/cm^2 to mmHg (divide by 1333)
    pressures_inlet_mmhg = pressures_inlet / 1333.0
    pressures_node10_mmhg = pressures_node10 / 1333.0
    
    print(f"  Inlet: {len(times_inlet)} time points")
    print(f"    Pressure range: [{np.min(pressures_inlet_mmhg):.2f}, {np.max(pressures_inlet_mmhg):.2f}] mmHg")
    if flows_inlet is not None:
        print(f"    Flow range: [{np.min(flows_inlet):.2f}, {np.max(flows_inlet):.2f}] cm³/s")
    
    print(f"  10 nodes in: {len(times_node10)} time points")
    print(f"    Pressure range: [{np.min(pressures_node10_mmhg):.2f}, {np.max(pressures_node10_mmhg):.2f}] mmHg")
    if flows_node10 is not None:
        print(f"    Flow range: [{np.min(flows_node10):.2f}, {np.max(flows_node10):.2f}] cm³/s")
    
    # Create plot
    print("\nCreating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot pressure (in mmHg)
    ax = axes[0]
    ax.plot(times_inlet_sec, pressures_inlet_mmhg, 'b-', linewidth=2, label='Inlet')
    ax.plot(times_node10_sec, pressures_node10_mmhg, 'r-', linewidth=2, label='10 nodes in')
    ax.set_ylabel('Pressure (mmHg)', fontsize=14)
    ax.set_title('1D Pressure vs Time', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot flow (in cm³/s)
    ax = axes[1]
    if flows_inlet is not None:
        ax.plot(times_inlet_sec, flows_inlet, 'b-', linewidth=2, label='Inlet')
    if flows_node10 is not None:
        ax.plot(times_node10_sec, flows_node10, 'r-', linewidth=2, label='10 nodes in')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Flow (cm³/s)', fontsize=14)
    ax.set_title('1D Flow vs Time', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D pressure and flow at inlet and 10 nodes in from inlet"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_007)')
    parser.add_argument('--vtp-path', help='Path to 1D solution VTP file (default: auto-detect)')
    parser.add_argument('--output', help='Output plot path (default: auto-generate)')
    parser.add_argument('--output-dir', default='results/1d_plots', 
                        help='Output directory for plots (default: results/1d_plots)')
    parser.add_argument('--time-period', type=float, default=None,
                        help='Time period in seconds (default: auto-detect from XML)')
    
    args = parser.parse_args()
    
    # Auto-detect VTP path
    if args.vtp_path:
        vtp_path = args.vtp_path
    else:
        # Try different possible locations
        possible_paths = [
            os.path.join('data', 'oneD', args.set_name, args.geo_name, 'unsteady_soln.vtp'),
            os.path.join('data', 'reduced_results', args.set_name, args.geo_name, 'unsteady_soln.vtp'),
            os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                        args.set_name, args.geo_name, 'unsteady_soln.vtp'),
        ]
        
        vtp_path = None
        for path in possible_paths:
            if os.path.exists(path):
                vtp_path = path
                break
        
        if vtp_path is None:
            print(f"Error: 1D solution VTP file not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.geo_name}_inlet_data.png")
    
    print(f"\nPlotting 1D inlet data for {args.set_name}/{args.geo_name}")
    print(f"  1D Solution: {vtp_path}")
    print(f"  Output: {output_path}")
    
    success = plot_1d_inlet_data(vtp_path, output_path, 
                                 set_name=args.set_name, 
                                 geo_name=args.geo_name,
                                 time_period=args.time_period)
    
    if success:
        print(f"\n✓ Successfully created plot: {output_path}")
        sys.exit(0)
    else:
        print(f"\n✗ Failed to create plot")
        sys.exit(1)


if __name__ == '__main__':
    main()

# python3 util/visualizations/plot_1d_inlet_data.py --set-name set_3 --geo-name tree_007

