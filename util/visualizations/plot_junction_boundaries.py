#!/usr/bin/env python3
"""
Visualize 1D anatomy with dots marking junction inlets and outlets.

This script reads a geometric input JSON file and centerline VTP file, then plots
the centerline with markers indicating junction inlet and outlet points.
"""

import os
import sys
import argparse
import json
import numpy as np

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
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
    """Read centerline VTP file and extract data."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centerline_path)
    reader.Update()
    centerline = reader.GetOutput()
    point_data = centerline.GetPointData()
    arrays = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        arrays[array.GetName()] = v2n(array)
    points = v2n(centerline.GetPoints().GetData())
    arrays['Points'] = points
    return arrays


def load_geometric_input(geometric_input_path):
    """Load geometric input JSON file."""
    with open(geometric_input_path, 'r') as f:
        return json.load(f)


def find_point_index_by_gid(centerline_data, target_gid):
    """Find the centerline point index corresponding to a GlobalNodeId."""
    gid_array = centerline_data.get('GlobalNodeId', None)
    if gid_array is None:
        return None
    gid_array = np.asarray(gid_array)
    matches = np.where(gid_array == target_gid)[0]
    if len(matches) > 0:
        return int(matches[0])
    return None


def plot_junction_boundaries(geometric_input_path, centerline_path, output_path,
                             title=None, figsize=(12, 10)):
    """
    Plot 1D anatomy with dots marking junction inlets and outlets.

    Args:
        geometric_input_path: Path to geometric input JSON file
        centerline_path: Path to centerline VTP file
        output_path: Path to save output PNG
        title: Optional title for the plot
        figsize: Figure size (width, height)
    """
    print(f"\n{'='*60}")
    print("Plotting Junction Boundaries")
    print(f"{'='*60}")

    print(f"\nLoading geometric input: {geometric_input_path}")
    geometric_input = load_geometric_input(geometric_input_path)
    junctions = geometric_input.get('junctions', [])
    print(f"  Found {len(junctions)} junctions")

    print(f"\nLoading centerline: {centerline_path}")
    centerline_data = read_centerline_vtp(centerline_path)
    points = centerline_data.get('Points', None)
    if points is None:
        raise ValueError("Points array not found in centerline data")
    if centerline_data.get('GlobalNodeId', None) is None:
        raise ValueError("GlobalNodeId array not found in centerline data")
    print(f"  Found {len(points)} centerline points")

    # Extract junction inlet and outlet points
    inlet_points = []
    outlet_points = []
    junction_names_with_ids = []
    junction_names_without_ids = []

    print(f"\n  Extracting junction inlet/outlet information from: {geometric_input_path}")
    print(f"  Processing junctions:")

    for junction in junctions:
        jname = junction.get('junction_name', '')
        node_ids = junction.get('centerline_node_ids', None)
        if node_ids is None:
            junction_names_without_ids.append(jname)
            print(f"    ⊘ {jname}: No centerline_node_ids")
            continue

        inlet_gid = node_ids.get('inlet', None)
        outlets_dict = node_ids.get('outlets', None)
        if not isinstance(outlets_dict, dict):
            outlets_dict = {}

        if inlet_gid is None:
            junction_names_without_ids.append(jname)
            print(f"    ⊘ {jname}: No inlet GID")
            continue

        # Exclude junctions with only one outlet
        if len(outlets_dict) <= 1:
            print(f"    ⊘ {jname}: Skipped (only {len(outlets_dict)} outlet(s), need > 1)")
            continue

        inlet_idx = find_point_index_by_gid(centerline_data, inlet_gid)
        if inlet_idx is None:
            print(f"    ✗ {jname}: Inlet GID={inlet_gid} not found in centerline")
            continue

        inlet_points.append(points[inlet_idx])
        outlet_gids = list(outlets_dict.values()) if outlets_dict else []
        for out_name, out_gid in (outlets_dict or {}).items():
            out_idx = find_point_index_by_gid(centerline_data, out_gid)
            if out_idx is not None:
                outlet_points.append(points[out_idx])
            else:
                print(f"    ✗ {jname}: Outlet {out_name} GID={out_gid} not found in centerline")
        junction_names_with_ids.append(jname)
        print(f"    ✓ {jname}: Inlet GID={inlet_gid}, {len(outlet_gids)} outlet(s)")

    if junction_names_without_ids:
        print(f"\n  Note: {len(junction_names_without_ids)} junctions without centerline_node_ids (not shown)")

    print(f"\n  Summary: {len(junction_names_with_ids)} junctions with valid data")
    print(f"  Found {len(inlet_points)} junction inlet points and {len(outlet_points)} junction outlet points")

    # Convert to numpy arrays
    if len(inlet_points) > 0:
        inlet_points = np.array(inlet_points)
    if len(outlet_points) > 0:
        outlet_points = np.array(outlet_points)

    # Rotate 160° about z-axis
    theta = np.deg2rad(160)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    def rotate_200_z(arr):
        a = np.asarray(arr).copy()
        x, y = a[:, 0].copy(), a[:, 1].copy()
        a[:, 0] = x * cos_t - y * sin_t
        a[:, 1] = x * sin_t + y * cos_t
        return a
    points = rotate_200_z(points)
    if len(inlet_points) > 0:
        inlet_points = rotate_200_z(inlet_points)
    if len(outlet_points) > 0:
        outlet_points = rotate_200_z(outlet_points)

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_facecolor('white')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor('white')
        axis.pane.set_edgecolor('white')
        axis.line.set_color('white')
        axis.set_ticklabels([])
    ax.set_axis_off()

    # Plot centerline first (behind markers)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='gray', s=18, alpha=0.3, label='Centerline', zorder=0)

    # Junction outlets and inlets in front of centerline
    if len(outlet_points) > 0:
        ax.scatter(outlet_points[:, 0], outlet_points[:, 1], outlet_points[:, 2],
                   facecolors='none', edgecolors='black', s=80, marker='^', linewidths=2,
                   alpha=0.9, label='Junction Outlets', zorder=10)
    if len(inlet_points) > 0:
        ax.scatter(inlet_points[:, 0], inlet_points[:, 1], inlet_points[:, 2],
                   facecolors='none', edgecolors='black', s=120, marker='o', linewidths=2,
                   alpha=0.9, label='Junction Inlets', zorder=11)

    if title:
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_title('Junction Boundaries on 1D Anatomy', fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=10)

    # Axis limits from full centerline so zoom is consistent with vessel-boundaries plot
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    max_range = max_range * 0.7
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print(f"\nSaving plot to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved successfully")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 1D anatomy with junction inlet/outlet markers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 util/visualizations/plot_junction_boundaries.py \\
      --geometric-input data/zeroD/VMR/0063_1001/bifurcations_geometric_input.json \\
      --centerline data/oneD/VMR/0063_1001/unsteady_soln.vtp \\
      --output results/junction_boundaries/VMR/0063_1001/bifurcations.png
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
                        metavar=('WIDTH', 'HEIGHT'), help='Figure size in inches (default: 12 10)')
    args = parser.parse_args()

    if not os.path.exists(args.geometric_input):
        print(f"Error: Geometric input file not found: {args.geometric_input}")
        sys.exit(1)
    if not os.path.exists(args.centerline):
        print(f"Error: Centerline file not found: {args.centerline}")
        sys.exit(1)

    try:
        plot_junction_boundaries(
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
