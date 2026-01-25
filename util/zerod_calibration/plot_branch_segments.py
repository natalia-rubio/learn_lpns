#!/usr/bin/env python3
"""
Plot observations per 0D vessel segment for a given 1D branch.

Creates a 2x2 figure:
  - Top-left: pressure at segment inlet
  - Top-right: pressure at segment outlet
  - Bottom-left: flow at segment inlet
  - Bottom-right: flow at segment outlet

Each segment (branchN_segM) is plotted in a different color.

Usage:
  python util/zerod_calibration/plot_branch_segments.py \
      data/zeroD/.../XXX_calibration_input_XXX.json --branch branch1
"""
import os
import json
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except Exception:
    plt = None


def find_vessels_for_branch(calib, branch_prefix):
    # Prefer vessels list in the calibration/geometric input
    vessels = calib.get('vessels') or calib.get('geometric_input', {}).get('vessels') or []
    names = []
    for v in vessels:
        name = v.get('vessel_name') if isinstance(v, dict) else None
        if name and name.startswith(branch_prefix):
            names.append(name)
    if names:
        return sorted(names)

    # Fallback: scan observation keys for vessel tokens
    obs = calib.get('y', {})
    names_set = set()
    for k in obs.keys():
        parts = k.split(":")
        for p in parts:
            if p.startswith(branch_prefix):
                names_set.add(p)
    return sorted(names_set)


def find_inlet_outlet_keys(obs_y, vessel_name):
    """Return (pressure_in_key, pressure_out_key, flow_in_key, flow_out_key) or None where missing."""
    p_in = p_out = q_in = q_out = None
    for k in obs_y.keys():
        if not k:
            continue
        parts = k.split(":")
        if len(parts) != 3:
            continue
        field, a, b = parts
        if field not in ('pressure', 'flow'):
            continue
        # If vessel_name appears as the third token => inlet-style (e.g., pressure:INFLOW:branch1_seg0 or pressure:J0:branch1_seg0)
        if b == vessel_name:
            if field == 'pressure' and p_in is None:
                p_in = k
            if field == 'flow' and q_in is None:
                q_in = k
        # If vessel_name appears as the second token => outlet-style (e.g., pressure:branch1_seg0:OUT1)
        if a == vessel_name:
            if field == 'pressure' and p_out is None:
                p_out = k
            if field == 'flow' and q_out is None:
                q_out = k
    return p_in, p_out, q_in, q_out


def get_time_array(calib, n_pts):
    # Try to find INFLOW BC time array in calibration input
    for bc in calib.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW' and 'bc_values' in bc and 't' in bc['bc_values']:
            t = bc['bc_values']['t']
            if isinstance(t, list) and len(t) == n_pts:
                return np.array(t)
    # Try observed_inflow_bc
    obc = calib.get('observed_inflow_bc') or calib.get('_full_bc_for_forward_sim')
    if obc and isinstance(obc.get('t'), list) and len(obc.get('t')) == n_pts:
        return np.array(obc.get('t'))
    # Fallback to normalized 0..1
    return np.linspace(0.0, 1.0, n_pts)


def plot_branch_segments(calib_path, branch, out_dir=None, start_idx=0, end_idx=None, show=False):
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    obs_y = calib.get('y', {})
    vessel_names = find_vessels_for_branch(calib, branch)
    if not vessel_names:
        print(f"No vessels found for branch prefix '{branch}'")
        return 1

    # Prepare plotting
    if plt is None:
        print("matplotlib not available — cannot plot")
        return 1

    n_segs = len(vessel_names)
    colors = cm.get_cmap('tab10', max(3, n_segs))

    # Prepare figure: 2 rows x 2 cols
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_p_in = axes[0, 0]
    ax_p_out = axes[0, 1]
    ax_q_in = axes[1, 0]
    ax_q_out = axes[1, 1]

    plotted = 0

    for i, vn in enumerate(vessel_names):
        p_in_k, p_out_k, q_in_k, q_out_k = find_inlet_outlet_keys(obs_y, vn)

        # Choose the first available length for time axis
        # get data arrays (may be missing)
        p_in = np.array(obs_y[p_in_k]) if p_in_k in obs_y else None
        p_out = np.array(obs_y[p_out_k]) if p_out_k in obs_y else None
        q_in = np.array(obs_y[q_in_k]) if q_in_k in obs_y else None
        q_out = np.array(obs_y[q_out_k]) if q_out_k in obs_y else None

        # Determine length for time array
        lengths = [len(arr) for arr in (p_in, p_out, q_in, q_out) if arr is not None]
        if not lengths:
            continue
        n = lengths[0]
        t = get_time_array(calib, n)[start_idx:end_idx]

        label = vn
        color = colors(i)

        if p_in is not None:
            arr = np.array(p_in)[start_idx:end_idx]
            ax_p_in.plot(t, arr, label=label if plotted == 0 else None, color=color)
        if p_out is not None:
            arr = np.array(p_out)[start_idx:end_idx]
            ax_p_out.plot(t, arr, label=label if plotted == 0 else None, color=color)
        if q_in is not None:
            arr = np.array(q_in)[start_idx:end_idx]
            ax_q_in.plot(t, arr, label=label if plotted == 0 else None, color=color)
        if q_out is not None:
            arr = np.array(q_out)[start_idx:end_idx]
            ax_q_out.plot(t, arr, label=label if plotted == 0 else None, color=color)

        plotted += 1

    # Finalize axes
    ax_p_in.set_title(f"Pressure — Inlet (segments of {branch})")
    ax_p_out.set_title(f"Pressure — Outlet (segments of {branch})")
    ax_q_in.set_title(f"Flow — Inlet (segments of {branch})")
    ax_q_out.set_title(f"Flow — Outlet (segments of {branch})")

    for ax in [ax_p_in, ax_p_out]:
        ax.set_ylabel('Pressure (dynes/cm^2)')
        ax.grid(True)
    for ax in [ax_q_in, ax_q_out]:
        ax.set_ylabel('Flow (cm^3/s)')
        ax.grid(True)

    # Put a combined legend (use vessel names)
    # Create proxy lines for legend
    handles = []
    labels = []
    for i, vn in enumerate(vessel_names):
        color = colors(i)
        handles.append(plt.Line2D([0], [0], color=color))
        labels.append(vn)
    ax_p_in.legend(handles, labels, loc='upper right', fontsize='small')

    plt.tight_layout()

    # Save
    # Try to infer set/geo from path: data/zeroD/<set>/<geo>/...
    set_name = 'unknown'
    geo_name = 'unknown'
    parts = calib_path.split(os.sep)
    try:
        zi = parts.index('zeroD')
        set_name = parts[zi + 1]
        geo_name = parts[zi + 2]
    except Exception:
        pass

    results_dir = out_dir or os.path.join('results', 'lin_reg', set_name, geo_name)
    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, f"{branch}_segments.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot to: {out_png}")

    if show:
        plt.show()

    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('calibration_input', help='Path to calibration input JSON')
    p.add_argument('--branch', required=True, help='Branch prefix (e.g., branch1)')
    p.add_argument('--out-dir', help='Directory to save plot (defaults to results/lin_reg/<set>/<geo>/)')
    p.add_argument('--start', type=int, default=0, help='Start index for slicing observations')
    p.add_argument('--end', type=int, default=None, help='End index (exclusive) for slicing observations')
    p.add_argument('--show', action='store_true', help='Show plot interactively')
    args = p.parse_args()

    rc = plot_branch_segments(args.calibration_input, args.branch, out_dir=args.out_dir, start_idx=args.start, end_idx=args.end, show=args.show)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == '__main__':
    main()
