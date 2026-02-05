#!/usr/bin/env python3
"""
Visualize 1D anatomy with dots marking the beginning and end of each vessel.

This script reads a geometric input JSON file and centerline VTP file, then plots
the centerline with markers indicating vessel inlet and outlet points.
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
    from mpl_toolkits.mplot3d import Axes3D
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


def read_centerline_vtp(centerline_path):
    """
    Read centerline VTP file and extract data.
    
    Returns:
        Dictionary with centerline data arrays including Points, GlobalNodeId, BranchId, etc.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centerline_path)
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
    
    return arrays


def load_geometric_input(geometric_input_path):
    """
    Load geometric input JSON file.
    
    Returns:
        Dictionary with vessels, junctions, boundary_conditions
    """
    with open(geometric_input_path, 'r') as f:
        return json.load(f)


def find_point_index_by_gid(centerline_data, target_gid):
    """
    Find the centerline point index corresponding to a GlobalNodeId.
    
    Args:
        centerline_data: Dictionary with centerline arrays
        target_gid: GlobalNodeId to find
    
    Returns:
        Point index, or None if not found
    """
    gid_array = centerline_data.get('GlobalNodeId', None)
    if gid_array is None:
        return None
    
    gid_array = np.asarray(gid_array)
    matches = np.where(gid_array == target_gid)[0]
    
    if len(matches) > 0:
        return int(matches[0])
    return None


def plot_vessel_boundaries(geometric_input_path, centerline_path, output_path, 
                          title=None, figsize=(12, 10)):
    """
    Plot 1D anatomy with dots marking vessel boundaries.
    
    Args:
        geometric_input_path: Path to geometric input JSON file
        centerline_path: Path to centerline VTP file
        output_path: Path to save output PNG
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    print(f"\n{'='*60}")
    print("Plotting Vessel Boundaries")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading geometric input: {geometric_input_path}")
    geometric_input = load_geometric_input(geometric_input_path)
    vessels = geometric_input.get('vessels', [])
    print(f"  Found {len(vessels)} vessels")
    
    print(f"\nLoading centerline: {centerline_path}")
    centerline_data = read_centerline_vtp(centerline_path)
    points = centerline_data.get('Points', None)
    gid_array = centerline_data.get('GlobalNodeId', None)
    
    if points is None:
        raise ValueError("Points array not found in centerline data")
    
    if gid_array is None:
        raise ValueError("GlobalNodeId array not found in centerline data")
    
    print(f"  Found {len(points)} centerline points")
    
    # Extract vessel boundary points
    # Show inlet and outlet for each vessel segment (seg0, seg1, seg2, etc.)
    inlet_points = []
    outlet_points = []
    connector_inlet_points = []
    connector_outlet_points = []
    vessel_names = []
    vessels_without_ids = []
    vessels_with_ids = []
    
    print(f"\n  Extracting vessel inlet/outlet information from: {geometric_input_path}")
    print(f"  Processing vessels:")
    
    for vessel in vessels:
        vessel_name = vessel.get('vessel_name', '')
        vessel_names.append(vessel_name)
        
        # Check if this is a connector vessel
        is_connector = 'connector' in vessel_name.lower()
        
        # Get node IDs from vessel
        node_ids = vessel.get('centerline_node_ids', None)
        if node_ids is None:
            vessels_without_ids.append(vessel_name)
            print(f"    ⊘ {vessel_name}: No centerline_node_ids")
            continue
        
        inlet_gid = node_ids.get('inlet', None)
        outlet_gid = node_ids.get('outlet', None)
        
        if inlet_gid is None or outlet_gid is None:
            vessels_without_ids.append(vessel_name)
            print(f"    ⊘ {vessel_name}: Incomplete centerline_node_ids (inlet={inlet_gid}, outlet={outlet_gid})")
            continue
        
        # Find point indices
        inlet_idx = find_point_index_by_gid(centerline_data, inlet_gid)
        outlet_idx = find_point_index_by_gid(centerline_data, outlet_gid)
        
        if inlet_idx is None:
            print(f"    ✗ {vessel_name}: Inlet GID={inlet_gid} not found in centerline")
        elif outlet_idx is None:
            print(f"    ✗ {vessel_name}: Outlet GID={outlet_gid} not found in centerline")
        else:
            print(f"    ✓ {vessel_name}: Inlet GID={inlet_gid}, Outlet GID={outlet_gid}")
            if is_connector:
                connector_inlet_points.append(points[inlet_idx])
                connector_outlet_points.append(points[outlet_idx])
            else:
                inlet_points.append(points[inlet_idx])
                outlet_points.append(points[outlet_idx])
            vessels_with_ids.append(vessel_name)
    
    if vessels_without_ids:
        print(f"\n  Note: {len(vessels_without_ids)} vessels without centerline_node_ids (not shown)")
        if len(vessels_without_ids) <= 10:
            print(f"    Missing vessels: {', '.join(vessels_without_ids)}")
        else:
            print(f"    Missing vessels: {', '.join(vessels_without_ids[:10])} ... ({len(vessels_without_ids) - 10} more)")
    
    print(f"\n  Summary: {len(vessels_with_ids)} vessels with valid inlet/outlet data")
    print(f"  Found {len(inlet_points)} regular vessel inlet points and {len(outlet_points)} regular vessel outlet points")
    if len(connector_inlet_points) > 0 or len(connector_outlet_points) > 0:
        print(f"  Found {len(connector_inlet_points)} connector vessel inlet points and {len(connector_outlet_points)} connector vessel outlet points")
    
    # Convert to numpy arrays
    if len(inlet_points) > 0:
        inlet_points = np.array(inlet_points)
    if len(outlet_points) > 0:
        outlet_points = np.array(outlet_points)
    if len(connector_inlet_points) > 0:
        connector_inlet_points = np.array(connector_inlet_points)
    if len(connector_outlet_points) > 0:
        connector_outlet_points = np.array(connector_outlet_points)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot centerline as points (not connected)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='gray', s=5, alpha=0.3, label='Centerline', zorder=0)
    
    # Plot regular vessel outlet points (red squares) - smaller size
    if len(outlet_points) > 0:
        ax.scatter(outlet_points[:, 0], outlet_points[:, 1], outlet_points[:, 2],
                  c='red', s=40, marker='s', alpha=0.8, label='Vessel Outlets', zorder=1)
    
    # Plot connector vessel outlet points (red squares, unfilled)
    if len(connector_outlet_points) > 0:
        ax.scatter(connector_outlet_points[:, 0], connector_outlet_points[:, 1], connector_outlet_points[:, 2],
                  facecolors='none', edgecolors='red', s=40, marker='s', linewidths=2, 
                  alpha=0.8, label='Connector Outlets', zorder=2)
    
    # Plot regular vessel inlet points (green circles) - larger size so they're more prominent
    if len(inlet_points) > 0:
        ax.scatter(inlet_points[:, 0], inlet_points[:, 1], inlet_points[:, 2],
                  c='green', s=120, marker='o', alpha=0.8, label='Vessel Inlets', zorder=10)
    
    # Plot connector vessel inlet points (green circles, unfilled)
    if len(connector_inlet_points) > 0:
        ax.scatter(connector_inlet_points[:, 0], connector_inlet_points[:, 1], connector_inlet_points[:, 2],
                  facecolors='none', edgecolors='green', s=120, marker='o', linewidths=2,
                  alpha=0.8, label='Connector Inlets', zorder=11)
    
    # Set labels and title
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_title('Vessel Boundaries on 1D Anatomy', fontsize=14, pad=20)
    
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio and zoom in more tightly
    # Get axis limits - use only vessel boundary points for tighter zoom
    boundary_points_list = []
    if len(inlet_points) > 0:
        boundary_points_list.append(inlet_points)
    if len(outlet_points) > 0:
        boundary_points_list.append(outlet_points)
    if len(connector_inlet_points) > 0:
        boundary_points_list.append(connector_inlet_points)
    if len(connector_outlet_points) > 0:
        boundary_points_list.append(connector_outlet_points)
    
    if len(boundary_points_list) > 0:
        boundary_points = np.vstack(boundary_points_list)
    else:
        # Fallback to all centerline points
        boundary_points = points
    
    # Calculate range with tighter zoom (reduce padding)
    max_range = np.array([boundary_points[:, 0].max() - boundary_points[:, 0].min(),
                          boundary_points[:, 1].max() - boundary_points[:, 1].min(),
                          boundary_points[:, 2].max() - boundary_points[:, 2].min()]).max() / 2.0
    
    # Reduce range significantly to zoom in much more tightly (multiply by 0.7 = 30% reduction)
    max_range = max_range * 0.7
    
    mid_x = (boundary_points[:, 0].max() + boundary_points[:, 0].min()) * 0.5
    mid_y = (boundary_points[:, 1].max() + boundary_points[:, 1].min()) * 0.5
    mid_z = (boundary_points[:, 2].max() + boundary_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save figure
    print(f"\nSaving plot to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved successfully")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 1D anatomy with vessel boundary markers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot vessel boundaries for a geometry
  python3 util/visualizations/plot_vessel_boundaries.py \\
      --geometric-input data/zeroD/VMR/0063_1001/bifurcations_geometric_input.json \\
      --centerline data/oneD/VMR/0063_1001/unsteady_soln.vtp \\
      --output results/vessel_boundaries/VMR/0063_1001/bifurcations.png
        """
    )
    
    parser.add_argument('--geometric-input', type=str, required=True,
                       help='Path to geometric input JSON file')
    parser.add_argument('--centerline', type=str, required=True,
                       help='Path to centerline VTP file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output PNG')
    parser.add_argument('--title', type=str, default=None,
                       help='Optional title for the plot')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 10],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 12 10)')
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.geometric_input):
        print(f"Error: Geometric input file not found: {args.geometric_input}")
        sys.exit(1)
    
    if not os.path.exists(args.centerline):
        print(f"Error: Centerline file not found: {args.centerline}")
        sys.exit(1)
    
    # Create plot
    try:
        plot_vessel_boundaries(
            args.geometric_input,
            args.centerline,
            args.output,
            title=args.title,
            figsize=tuple(args.figsize)
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

