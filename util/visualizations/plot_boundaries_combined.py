#!/usr/bin/env python3
"""
Side-by-side: original (bifurcations) schematic (left) and EL-adjusted schematic (right) in one figure.
Each panel shows centerline plus vessel and junction boundary markers. Uses LaTeX for text.
"""

import os
import sys
import argparse
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n
except ImportError:
    print("Error: VTK is required. Make sure VTK is installed and available.")
    sys.exit(1)


def _setup_latex():
    """Enable LaTeX text rendering. Falls back to default if LaTeX unavailable."""
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern', 'DejaVu Serif']
        return True
    except Exception:
        plt.rcParams['text.usetex'] = False
        return False


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
    return int(matches[0]) if len(matches) > 0 else None


def _rotate_z(points, theta_deg=160):
    """Rotate points by theta_deg about z-axis."""
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a = np.asarray(points).copy()
    x, y = a[:, 0].copy(), a[:, 1].copy()
    a[:, 0] = x * cos_t - y * sin_t
    a[:, 1] = x * sin_t + y * cos_t
    return a


def _style_axis_3d(ax):
    """Apply white background and hide axes/box."""
    ax.grid(False)
    ax.set_facecolor('white')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor('white')
        axis.pane.set_edgecolor('white')
        axis.line.set_color('white')
        axis.set_ticklabels([])
    ax.set_axis_off()


def _extract_and_rotate(geometric_input, centerline_data, points_rotated):
    """
    From one geometric input and the (already rotated) centerline points, extract
    vessel and junction boundary points in the same rotated space.
    Returns dict with keys: v_inlet, v_outlet, v_conn_in, v_conn_out, j_inlet, j_outlet (numpy arrays).
    """
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    points = centerline_data.get('Points', None)  # unrotated for GID lookup

    v_inlet, v_outlet, v_conn_in, v_conn_out = [], [], [], []
    for vessel in vessels:
        name = vessel.get('vessel_name', '')
        node_ids = vessel.get('centerline_node_ids', None)
        if node_ids is None:
            continue
        inlet_gid = node_ids.get('inlet', None)
        outlet_gid = node_ids.get('outlet', None)
        if inlet_gid is None or outlet_gid is None:
            continue
        inlet_idx = find_point_index_by_gid(centerline_data, inlet_gid)
        outlet_idx = find_point_index_by_gid(centerline_data, outlet_gid)
        if inlet_idx is None or outlet_idx is None:
            continue
        is_conn = 'connector' in name.lower()
        if is_conn:
            v_conn_in.append(points_rotated[inlet_idx])
            v_conn_out.append(points_rotated[outlet_idx])
        else:
            v_inlet.append(points_rotated[inlet_idx])
            v_outlet.append(points_rotated[outlet_idx])

    j_inlet, j_outlet = [], []
    for junction in junctions:
        node_ids = junction.get('centerline_node_ids', None)
        if node_ids is None:
            continue
        outlets_dict = node_ids.get('outlets') or {}
        if not isinstance(outlets_dict, dict) or len(outlets_dict) <= 1:
            continue
        inlet_gid = node_ids.get('inlet', None)
        if inlet_gid is None:
            continue
        inlet_idx = find_point_index_by_gid(centerline_data, inlet_gid)
        if inlet_idx is None:
            continue
        j_inlet.append(points_rotated[inlet_idx])
        for out_gid in outlets_dict.values():
            out_idx = find_point_index_by_gid(centerline_data, out_gid)
            if out_idx is not None:
                j_outlet.append(points_rotated[out_idx])

    def to_arr(lst):
        return np.array(lst) if lst else np.empty((0, 3))
    return {
        'v_inlet': to_arr(v_inlet),
        'v_outlet': to_arr(v_outlet),
        'v_conn_in': to_arr(v_conn_in),
        'v_conn_out': to_arr(v_conn_out),
        'j_inlet': to_arr(j_inlet),
        'j_outlet': to_arr(j_outlet),
    }


def _draw_schematic(ax, points_rotated, data, panel_title):
    """Draw one panel: centerline (back) + junction markers (front)."""
    ax.scatter(points_rotated[:, 0], points_rotated[:, 1], points_rotated[:, 2],
               c='gray', s=18, alpha=0.3, label=r'Centerline', zorder=0)
    j_outlet = data['j_outlet']
    j_inlet = data['j_inlet']
    if len(j_outlet):
        ax.scatter(j_outlet[:, 0], j_outlet[:, 1], j_outlet[:, 2],
                  facecolors='none', edgecolors='black', s=80, marker='^', linewidths=2,
                  alpha=0.9, label=r'Junction outlets', zorder=10)
    if len(j_inlet):
        ax.scatter(j_inlet[:, 0], j_inlet[:, 1], j_inlet[:, 2],
                  facecolors='none', edgecolors='black', s=120, marker='o', linewidths=2,
                  alpha=0.9, label=r'Junction inlets', zorder=11)
    ax.set_title(panel_title, fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=9)


def plot_boundaries_combined(geometric_input_bifurcations_path, geometric_input_el_path,
                             centerline_path, output_path, title=None, figsize=(14, 6)):
    """
    Plot original (bifurcations) schematic (left) and EL-adjusted schematic (right) side by side.
    Same centerline and zoom for both. LaTeX labels.
    """
    use_latex = _setup_latex()
    if not use_latex:
        print("Warning: LaTeX not available; using default text.")

    centerline_data = read_centerline_vtp(centerline_path)
    points = centerline_data.get('Points', None)
    if points is None or centerline_data.get('GlobalNodeId', None) is None:
        raise ValueError("Centerline must contain Points and GlobalNodeId")

    points_rotated = _rotate_z(points)

    geo_bif = load_geometric_input(geometric_input_bifurcations_path)
    geo_el = load_geometric_input(geometric_input_el_path)
    data_bif = _extract_and_rotate(geo_bif, centerline_data, points_rotated)
    data_el = _extract_and_rotate(geo_el, centerline_data, points_rotated)

    # Shared axis limits from full centerline
    max_range = np.array([
        points_rotated[:, 0].max() - points_rotated[:, 0].min(),
        points_rotated[:, 1].max() - points_rotated[:, 1].min(),
        points_rotated[:, 2].max() - points_rotated[:, 2].min()
    ]).max() / 2.0 * 0.7
    mid_x = (points_rotated[:, 0].max() + points_rotated[:, 0].min()) * 0.5
    mid_y = (points_rotated[:, 1].max() + points_rotated[:, 1].min()) * 0.5
    mid_z = (points_rotated[:, 2].max() + points_rotated[:, 2].min()) * 0.5
    lims = (mid_x - max_range, mid_x + max_range,
            mid_y - max_range, mid_y + max_range,
            mid_z - max_range, mid_z + max_range)

    def set_limits(ax):
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.set_zlim(lims[4], lims[5])

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    _style_axis_3d(ax1)
    _style_axis_3d(ax2)

    _draw_schematic(ax1, points_rotated, data_bif, r'\textbf{Bifurcations (original)}')
    set_limits(ax1)
    _draw_schematic(ax2, points_rotated, data_el, r'\textbf{EL-adjusted}')
    set_limits(ax2)

    if title and use_latex:
        title_safe = title.replace('_', r'\_')
        fig.suptitle(title_safe, fontsize=16, y=1.02)
    elif title:
        fig.suptitle(title, fontsize=16, y=1.02)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Side-by-side: original vs EL-adjusted schematic with LaTeX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--geometric-input-bifurcations', type=str, required=True,
                        help='Path to bifurcations geometric input JSON (original)')
    parser.add_argument('--geometric-input-el', type=str, required=True,
                        help='Path to EL-adjusted geometric input JSON')
    parser.add_argument('--centerline', type=str, required=True,
                        help='Path to centerline VTP file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output PNG')
    parser.add_argument('--title', type=str, default=None,
                        help='Optional overall figure title')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 6],
                        metavar=('W', 'H'), help='Figure size (default: 14 6)')
    args = parser.parse_args()

    for path in (args.geometric_input_bifurcations, args.geometric_input_el):
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
    if not os.path.exists(args.centerline):
        print(f"Error: Centerline not found: {args.centerline}")
        sys.exit(1)

    try:
        plot_boundaries_combined(
            args.geometric_input_bifurcations,
            args.geometric_input_el,
            args.centerline,
            args.output,
            title=args.title,
            figsize=tuple(args.figsize),
        )
        print(f"Saved: {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
