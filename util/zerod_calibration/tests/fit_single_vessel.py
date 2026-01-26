#!/usr/bin/env python3
"""
Fit R_poiseuille, stenosis_coefficient and L for a single vessel using
observations contained in a calibration input JSON.

This script:
- Loads a calibration input JSON (contains 'y' observations)
- Lets the user specify an observation key (or prints available keys)
- Finds the corresponding vessel in the JSON and fits the three parameters
  by running forward 0D simulations and minimizing the difference between
  simulated and observed time series at that location.

Notes:
- Requires scipy.optimize (least_squares) for efficient fitting. Falls
  back to scipy.optimize.minimize (Nelder-Mead) if needed.
- Uses the existing run_forward_simulation and read_zerod_csv helpers in
  the same package.
"""

import argparse
import json
import os
import shutil
import tempfile
import numpy as np
import time
import sys
sys.path.append("/Users/natalia/cursor_access/learn_lpns")
from util.zerod_calibration.calibration_helpers import read_zerod_csv
from util.zerod_calibration.generate_zerod_inputs import run_forward_simulation
from util.zerod_calibration.post_processing import read_zerod_csv as post_read_zerod_csv
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.optimize import least_squares
    HAS_LEAST_SQUARES = True
except Exception:
    HAS_LEAST_SQUARES = False
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None


def list_observation_keys(cfg):
    keys = []
    if 'y' in cfg and isinstance(cfg['y'], dict):
        keys = list(cfg['y'].keys())
    return keys


def parse_obs_key(obs_key):
    # Expect keys like 'pressure:branch0_seg0:J0' or 'flow:INFLOW:branch0_seg0'
    parts = obs_key.split(':')
    if len(parts) != 3:
        raise ValueError(f"Unexpected observation key format: {obs_key}")
    obs_type, part1, part2 = parts
    # Determine vessel_name and field_name similar to post_processing logic
    vessel_name = None
    if part1 == 'INFLOW':
        vessel_name = part2
        field_name = f"{obs_type}_in"
    elif part1.startswith('branch') and part2.startswith('J'):
        vessel_name = part1
        field_name = f"{obs_type}_out"
    elif part1.startswith('J') and part2.startswith('branch'):
        vessel_name = part2
        field_name = f"{obs_type}_in"
    else:
        # fallback: take first part that looks like a vessel
        for p in (part1, part2):
            if p.startswith('branch'):
                vessel_name = p
                field_name = f"{obs_type}_out" if p == part1 else f"{obs_type}_in"
                break
    if vessel_name is None:
        # as a last resort choose part2
        vessel_name = part2
        field_name = f"{obs_type}_out"
    return vessel_name, field_name


def get_simulated_trace(csv_path, vessel_name, field_name, target_len=None):
    # read_zerod_csv returns dict and times
    results, times = read_zerod_csv(csv_path)
    # Some code paths use alternative reader
    if not results:
        results, times = post_read_zerod_csv(csv_path)

    if vessel_name not in results:
        # Try name variants
        possible = [k for k in results.keys() if vessel_name in k]
        if possible:
            vessel_key = possible[0]
        else:
            raise KeyError(f"Vessel '{vessel_name}' not found in simulation CSV: available={list(results.keys())}")
    else:
        vessel_key = vessel_name

    # Gather values in time order
    sim_times = np.array(times, dtype=float)
    sim_values = []
    for t in times:
        row = results[vessel_key].get(t, {})
        val = row.get(field_name)
        if val is None:
            # try alternative keys (pressure_in vs pressure_in)
            val = row.get(field_name)
        if val is None:
            # Missing field -> raise
            raise KeyError(f"Field '{field_name}' not found for vessel '{vessel_key}' in CSV row at time {t}")
        sim_values.append(float(val))
    sim_values = np.array(sim_values, dtype=float)

    if target_len is None or len(sim_values) == target_len:
        return sim_times, sim_values

    # interpolate to target length
    target_times = np.linspace(sim_times[0], sim_times[-1], target_len)
    sim_values_interp = np.interp(target_times, sim_times, sim_values)
    return target_times, sim_values_interp


def run_sim_and_get_trace(temp_json_path, vessel_name, field_name, target_len=None):
    # Run forward simulation using existing helper. Create temporary CSV path.
    temp_dir = os.path.dirname(temp_json_path)
    out_csv = os.path.join(temp_dir, 'fit_forward_output.csv')
    # Remove existing file
    if os.path.exists(out_csv):
        os.remove(out_csv)
    # Run simulation (this may call external svzerodsolver)
    run_forward_simulation(temp_json_path, out_csv)
    if not os.path.exists(out_csv):
        raise RuntimeError(f"Forward simulation did not produce expected CSV: {out_csv}")
    return get_simulated_trace(out_csv, vessel_name, field_name, target_len=target_len)


def objective(params, base_cfg, vessel_name, field_name, obs_y):
    # params: [R_poiseuille, L, stenosis_coefficient]
    R, L, sten = params
    # enforce simple bounds inside objective to avoid invalid sims
    if R <= 0 or L < 0 or sten < 0:
        return np.ones_like(obs_y) * 1e6

    # create temp config and write
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, 'calib_tmp.json')
        cfg = json.loads(json.dumps(base_cfg))
        # find vessel
        vessel_found = False
        for v in cfg.get('vessels', []):
            if v.get('vessel_name') == vessel_name:
                zd = v.setdefault('zero_d_element_values', {})
                zd['R_poiseuille'] = float(R)
                zd['L'] = float(L)
                zd['stenosis_coefficient'] = float(sten)
                vessel_found = True
                break
        if not vessel_found:
            raise KeyError(f"Vessel '{vessel_name}' not found in calibration JSON")

        # write tmp json
        with open(tmp_path, 'w') as f:
            json.dump(cfg, f, indent=4)

        # run sim and get trace
        try:
            sim_times, sim_values = run_sim_and_get_trace(tmp_path, vessel_name, field_name, target_len=len(obs_y))
        except Exception as e:
            print(f"  Simulation error for params {params}: {e}")
            return np.ones_like(obs_y) * 1e6

    # Compute residual (sim - obs)
    res = sim_values - np.array(obs_y, dtype=float)
    return res


def main():
    parser = argparse.ArgumentParser(description="Fit R_poiseuille, L and stenosis for a single vessel")
    parser.add_argument('calibration_input', help='Path to calibration input JSON (contains observations in "y")')
    parser.add_argument('--element', help='Element (vessel) name to fit (e.g., branch0_seg0). Required')
    parser.add_argument('--output', help='Path to write updated calibration JSON (default: <input>_fitted.json)')
    args = parser.parse_args()

    if not os.path.exists(args.calibration_input):
        raise FileNotFoundError(f"Calibration input not found: {args.calibration_input}")

    with open(args.calibration_input, 'r') as f:
        cfg = json.load(f)

    keys = list_observation_keys(cfg)
    if not keys:
        print("No observations found in calibration input (no 'y' dictionary). Exiting.")
        return

    if not args.element:
        print("Please provide --element <vessel_name> (e.g., branch0_seg0). Available elements can be inferred from observation keys.")
        # show some vessel-like tokens from keys
        vessels = set()
        for k in keys:
            parts = k.split(':')
            for p in parts[1:]:
                if p.startswith('branch'):
                    vessels.add(p)
        print("Sample detected vessel names:")
        for v in sorted(list(vessels))[:40]:
            print("  ", v)
        return

    vessel_name = args.element
    print(f"Fitting parameters for element (vessel) '{vessel_name}'")

    # Find observation keys corresponding to this vessel
    # We need: pressure_in, pressure_out, flow_in, flow_out
    obs = cfg.get('y', {})
    pressure_in_key = None
    pressure_out_key = None
    flow_in_key = None
    flow_out_key = None

    for k in obs.keys():
        parts = k.split(':')
        if len(parts) != 3:
            continue
        typ, p1, p2 = parts
        if typ not in ('pressure', 'flow'):
            continue
        # INFLOW case
        if p1 == 'INFLOW' and p2 == vessel_name:
            if typ == 'pressure':
                pressure_in_key = k
            else:
                flow_in_key = k
            continue
        if p2 == 'INFLOW' and p1 == vessel_name:
            if typ == 'pressure':
                pressure_out_key = k
            else:
                flow_out_key = k
            continue

        # Junction patterns: pressure:branchX:JY -> outlet; pressure:JY:branchX -> inlet
        if p1 == vessel_name and p2.startswith('J'):
            # outlet
            if typ == 'pressure':
                pressure_out_key = k
            else:
                flow_out_key = k
            continue
        if p2 == vessel_name and p1.startswith('J'):
            # inlet
            if typ == 'pressure':
                pressure_in_key = k
            else:
                flow_in_key = k
            continue

        # Generic membership: if vessel appears in either p1 or p2, choose based on position
        if p1 == vessel_name:
            # treat as outlet
            if typ == 'pressure' and pressure_out_key is None:
                pressure_out_key = k
            if typ == 'flow' and flow_out_key is None:
                flow_out_key = k
        if p2 == vessel_name:
            # treat as inlet
            if typ == 'pressure' and pressure_in_key is None:
                pressure_in_key = k
            if typ == 'flow' and flow_in_key is None:
                flow_in_key = k

    missing = []
    if pressure_in_key is None:
        missing.append('pressure_in')
    if pressure_out_key is None:
        missing.append('pressure_out')
    if flow_in_key is None:
        missing.append('flow_in')
    if flow_out_key is None:
        missing.append('flow_out')
    if missing:
        raise RuntimeError(f"Could not find required observation keys for vessel '{vessel_name}': missing {missing}. Available obs keys sample: {list(keys)[:20]}")

    print(f"Using observation keys:\n  pressure_in: {pressure_in_key}\n  pressure_out: {pressure_out_key}\n  flow_in: {flow_in_key}\n  flow_out: {flow_out_key}")

    # Find vessel and initial params
    vessel_cfg = None
    for v in cfg.get('vessels', []):
        if v.get('vessel_name') == vessel_name:
            vessel_cfg = v
            break
    if vessel_cfg is None:
        raise KeyError(f"Vessel '{vessel_name}' not found in calibration JSON")

    # Load observed arrays
    obs_dict = cfg['y']
    P_in = np.array(obs_dict[pressure_in_key], dtype=float)
    P_out = np.array(obs_dict[pressure_out_key], dtype=float)
    Q_in = np.array(obs_dict[flow_in_key], dtype=float)
    Q_out = np.array(obs_dict[flow_out_key], dtype=float)



    # Compute derivative dQ_out/dt. Prefer using provided 'dy' observations if available.
    dy_dict = cfg.get('dy', {})
    if flow_out_key in dy_dict:

        dQ_out_dt = np.array(dy_dict[flow_out_key], dtype=float)
        # If lengths mismatch, fall back to numerical gradient
        if dQ_out_dt.shape[0] != Q_out.shape[0]:
                # Determine time step dt
            dt = None
            try:
                sim_par = cfg.get('simulation_parameters', {})
                if 'cardiac_cycle_period' in sim_par and 'number_of_time_pts_per_cardiac_cycle' in sim_par:
                    period = float(sim_par['cardiac_cycle_period'])
                    npts = int(sim_par['number_of_time_pts_per_cardiac_cycle'])
                    if npts > 1:
                        dt = period / float(npts)
            except Exception:
                dt = None
            # Fallback: look for BC time arrays
            if dt is None:
                try:
                    bcs = cfg.get('boundary_conditions', [])
                    for bc in bcs:
                        vals = bc.get('bc_values', {})
                        if 't' in vals and isinstance(vals['t'], list) and len(vals['t']) > 1:
                            tarr = np.array(vals['t'], dtype=float)
                            dt = float(tarr[1] - tarr[0])
                            break
                except Exception:
                    dt = None
            if dt is None:
                # default to 1.0 / len
                dt = 1.0 / max(1, len(Q_out))
                dQ_out_dt = np.gradient(Q_out, dt)
            dQ_out_dt = np.gradient(Q_out, dt)

    # Build regression target and design matrix using equation:
    # P_in - P_out = r * Q_in + s * Q_in * |Q_in| + L * dQ_out/dt
    y = P_in - P_out
    col_r = Q_in
    col_s = Q_in * np.abs(Q_in)
    col_L = dQ_out_dt
    # Stack into design matrix
    A = np.vstack([col_r, col_s, col_L]).T

    # Solve least squares
    try:
        params, residuals, rank, svals = np.linalg.lstsq(A, y, rcond=None)
    except Exception as e:
        raise RuntimeError(f"Linear regression failed: {e}")

    R_fit, S_fit, L_fit = float(params[0]), float(params[1]), float(params[2])
    # Compute fitted values and residuals
    y_pred = A.dot(params)
    res = y - y_pred
    mse = float(np.mean(res ** 2))

    print(f"Linear fit results for vessel '{vessel_name}':")
    print(f"  R_poiseuille = {R_fit:.6g}")
    print(f"  stenosis_resistance = {S_fit:.6g}")
    print(f"  L = {L_fit:.6g}")
    print(f"  MSE = {mse:.6g}")

    # Create results directory under results/lin_reg/<set_name>/<geo_name>/
    cal_path = os.path.normpath(args.calibration_input)
    parts = cal_path.split(os.sep)
    set_name = None
    geo_name = None
    try:
        for i in range(len(parts) - 3):
            if parts[i] == 'data' and parts[i+1] == 'zeroD':
                set_name = parts[i+2]
                geo_name = parts[i+3]
                break
    except Exception:
        pass

    results_base = os.path.join('results', 'lin_reg')
    if set_name and geo_name:
        results_dir = os.path.join(results_base, set_name, geo_name)
    else:
        results_dir = os.path.join(results_base, 'unknown', 'unknown')
    os.makedirs(results_dir, exist_ok=True)

    # Time array for plotting: prefer BC time if available
    time_arr = None
    try:
        bcs = cfg.get('boundary_conditions', [])
        for bc in bcs:
            if bc.get('bc_name') == 'INFLOW' and 'bc_values' in bc and 't' in bc['bc_values']:
                time_arr = np.array(bc['bc_values']['t'], dtype=float)
                break
    except Exception:
        time_arr = None
    if time_arr is None:
        time_arr = np.arange(len(y)) * dt

    # Plot pressure drop (observed vs predicted) and flows on secondary axis
    plot_path = None
    if plt is not None:
        try:
            fig, ax1 = plt.subplots(figsize=(9,4))

            # Pressure drop lines on primary y-axis
            l1, = ax1.plot(time_arr, y, label='Observed ΔP (P_in - P_out)', color='tab:blue')
            l2, = ax1.plot(time_arr, y_pred, label='Predicted ΔP (linear fit)', linestyle='--', color='tab:orange')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pressure drop')

            # Secondary axis for flows
            ax2 = ax1.twinx()
            l3, = ax2.plot(time_arr, Q_in, label='Q_in', color='tab:green', alpha=0.8)
            l4, = ax2.plot(time_arr, Q_out, label='Q_out', color='tab:red', alpha=0.6)
            ax2.set_ylabel('Flow')

            # Combined legend
            lines = [l1, l2, l3, l4]
            labels = [ln.get_label() for ln in lines]
            ax1.legend(lines, labels, loc='upper right')

            plt.title(f"Pressure drop and flow for {vessel_name} — linear fit")
            fig.tight_layout()
            plot_path = os.path.join(results_dir, f"{vessel_name}_pressure_flow_fit.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved plot to: {plot_path}")
        except Exception as e:
            print(f"Warning: failed to create/save plot: {e}")
    else:
        print("matplotlib not available; skipping plot generation")

    # Save fitting results JSON
    results_json = {
        'element': vessel_name,
        'R_poiseuille': R_fit,
        'stenosis_resistance': S_fit,
        'L': L_fit,
        'mse': mse,
        'pressure_in_key': pressure_in_key,
        'pressure_out_key': pressure_out_key,
        'flow_in_key': flow_in_key,
        'flow_out_key': flow_out_key,
        'plot': os.path.basename(plot_path) if plot_path else None
    }
    results_json_path = os.path.join(results_dir, f"{vessel_name}_fit_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"Saved fit results to: {results_json_path}")

    # Also write updated calibration JSON alongside original
    out_path = args.output if args.output else args.calibration_input.rstrip('.json') + '_fitted_linear.json'
    cfg_fitted = json.loads(json.dumps(cfg))
    for v in cfg_fitted.get('vessels', []):
        if v.get('vessel_name') == vessel_name:
            zd2 = v.setdefault('zero_d_element_values', {})
            zd2['R_poiseuille'] = R_fit
            zd2['L'] = L_fit
            zd2['stenosis_coefficient'] = S_fit
            break

    with open(out_path, 'w') as f:
        json.dump(cfg_fitted, f, indent=4)

    print(f"Wrote fitted calibration JSON to: {out_path}")


if __name__ == '__main__':
    main()
