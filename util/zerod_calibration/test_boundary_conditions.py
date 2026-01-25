#!/usr/bin/env python3
"""
Test boundary conditions by applying observed outlet flows and comparing
the resulting pressure to observed pressure.

Uses svZeroDSolver to simulate a minimal model with:
- Prescribed inflow (from observations)
- Single minimal vessel
- The RCR/Resistance BC being tested

Compares simulated outlet pressure to observed pressure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import subprocess
import tempfile
import csv

# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================
FIGURE_SIZE = (14, 10)
FONT_SIZE = 12
LINE_WIDTH = 1.5

# Colors for different lines
COLOR_OBSERVED = 'black'
COLOR_SIMULATED = 'red'

# Path to svZeroDSolver executable
SVZERODSOLVER_PATH = "/Users/natalia/cursor_access/svZeroDPlus/Release/svzerodsolver"

# =============================================================================


def simulate_bc_with_solver(Q_in, t, bc_type, bc_values, solver_path=SVZERODSOLVER_PATH):
    """
    Simulate boundary condition using svZeroDSolver.
    
    Creates a minimal model with:
    - FLOW inlet BC (prescribing observed flow)
    - Single minimal vessel (very low R, L, C to minimize vessel effects)
    - The outlet BC being tested
    
    Args:
        Q_in: Input flow array (flow into BC)
        t: Time array
        bc_type: Type of BC ('RCR', 'RESISTANCE')
        bc_values: BC parameters dict
        solver_path: Path to svzerodsolver executable
    
    Returns:
        P_out: Simulated pressure at vessel outlet (BC inlet)
        success: Whether simulation succeeded
    """
    n = len(Q_in)
    dt = t[1] - t[0] if len(t) > 1 else 0.02
    period = t[-1] - t[0] + dt
    
    # Create minimal model
    model = {
        "simulation_parameters": {
            "number_of_cardiac_cycles": 10,
            "number_of_time_pts_per_cardiac_cycle": n,
            "cardiac_cycle_period": period,
            "density": 1.06,
            "viscosity": 0.04,
            "steady_initial": False,
            "output_all_cycles": False
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": "test_vessel",
                "vessel_length": 1.0,
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {
                    "R_poiseuille": 1e-6,  # Minimal resistance
                    "C": 1e-10,            # Minimal capacitance
                    "L": 1e-10,            # Minimal inductance
                    "stenosis_coefficient": 0.0
                },
                "boundary_conditions": {
                    "inlet": "INFLOW",
                    "outlet": "TEST_BC"
                }
            }
        ],
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {
                    "Q": Q_in.tolist(),
                    "t": t.tolist()
                }
            }
        ],
        "junctions": []
    }
    
    # Add outlet BC
    if bc_type == "RCR":
        model["boundary_conditions"].append({
            "bc_name": "TEST_BC",
            "bc_type": "RCR",
            "bc_values": {
                "Rp": bc_values.get("Rp", 0),
                "C": bc_values.get("C", 0),
                "Rd": bc_values.get("Rd", 0),
                "Pd": bc_values.get("Pd", 0)
            }
        })
    elif bc_type == "RESISTANCE":
        model["boundary_conditions"].append({
            "bc_name": "TEST_BC",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": bc_values.get("R", bc_values.get("Rp", 0)),
                "Pd": bc_values.get("Pd", 0)
            }
        })
    else:
        print(f"  Unsupported BC type for solver simulation: {bc_type}")
        return None, False
    
    # Write model to temp file and run solver
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.csv")
        
        with open(input_path, 'w') as f:
            json.dump(model, f, indent=2)
        
        # Run solver
        try:
            result = subprocess.run(
                [solver_path, input_path, output_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"  Solver failed with return code {result.returncode}")
                if result.stderr:
                    print(f"  STDERR: {result.stderr[:500]}")
                return None, False
            
            # Read results
            if not os.path.exists(output_path):
                print(f"  No output file created")
                return None, False
            
            # Read CSV without pandas
            times = []
            pressures = []
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['name'] == 'test_vessel':
                        times.append(float(row['time']))
                        pressures.append(float(row['pressure_out']))
            
            if len(pressures) == 0:
                print(f"  No data for test_vessel in output")
                return None, False
            
            # Sort by time
            sort_idx = np.argsort(times)
            t_out = np.array(times)[sort_idx]
            P_out = np.array(pressures)[sort_idx]
            
            # Handle case where output has different length
            if len(P_out) != n:
                print(f"  Warning: Output length ({len(P_out)}) != input length ({n})")
                # Interpolate to match input length
                P_out = np.interp(t, t_out, P_out)
            
            return P_out, True
            
        except subprocess.TimeoutExpired:
            print(f"  Solver timed out")
            return None, False
        except Exception as e:
            print(f"  Error running solver: {e}")
            return None, False


def simulate_rcr_bc(Q_in, dt, Rp, C, Rd, Pd, P_c0=None):
    """
    Simulate RCR boundary condition response to input flow.
    
    RCR model equations:
    - P_in = Rp * Q_in + P_c  (inlet pressure)
    - C * dP_c/dt = Q_in - (P_c - Pd) / Rd  (capacitor dynamics)
    
    Args:
        Q_in: Input flow array (flow into BC from vessel)
        dt: Time step
        Rp: Proximal resistance
        C: Capacitance
        Rd: Distal resistance
        Pd: Distal pressure
        P_c0: Initial capacitor pressure (default: steady-state value)
    
    Returns:
        P_in: Simulated inlet pressure array
        P_c: Capacitor pressure array
    """
    n = len(Q_in)
    
    # Initial condition: steady state P_c = Rd * Q_in[0] + Pd
    if P_c0 is None:
        P_c0 = Rd * Q_in[0] + Pd
    
    # Handle C = 0 case (pure resistance)
    if C <= 0 or np.isnan(C):
        # No capacitor dynamics, P_c is determined by flow through Rd
        # Q_out = Q_in (no storage), P_c = Rd * Q_in + Pd
        P_c = Rd * Q_in + Pd
    else:
        # Solve ODE using RK4: C * dP_c/dt = Q_in - (P_c - Pd) / Rd
        P_c = np.zeros(n)
        P_c[0] = P_c0
        
        def dPc_dt(P_c_val, Q):
            return (Q - (P_c_val - Pd) / Rd) / C
        
        for i in range(n - 1):
            Q = Q_in[i]
            Q_next = Q_in[i + 1]
            Q_mid = 0.5 * (Q + Q_next)
            
            # RK4 integration
            k1 = dPc_dt(P_c[i], Q)
            k2 = dPc_dt(P_c[i] + 0.5 * dt * k1, Q_mid)
            k3 = dPc_dt(P_c[i] + 0.5 * dt * k2, Q_mid)
            k4 = dPc_dt(P_c[i] + dt * k3, Q_next)
            
            P_c[i + 1] = P_c[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Compute inlet pressure
    P_in = Rp * Q_in + P_c
    
    return P_in, P_c


def simulate_resistance_bc(Q_in, R, Pd):
    """
    Simulate simple resistance BC: P_in = R * Q_in + Pd
    """
    return R * Q_in + Pd


def fit_rcr_parameters_with_solver(Q_obs, P_obs, t, initial_params, solver_path=SVZERODSOLVER_PATH):
    """
    Fit RCR parameters using grid search with svZeroDSolver.
    Keeps Pd = 0.
    
    Args:
        Q_obs: Observed flow into BC
        P_obs: Observed pressure at BC inlet
        t: Time array
        initial_params: Initial Rp, C, Rd values
        solver_path: Path to svzerodsolver
    
    Returns:
        Fitted parameters dict: {'Rp': ..., 'C': ..., 'Rd': ..., 'Pd': 0}
    """
    Rp0 = initial_params.get('Rp', 100)
    C0 = initial_params.get('C', 1e-4)
    Rd0 = initial_params.get('Rd', 10000)
    Pd = 0.0
    
    # Grid search around initial values
    # Try multipliers: 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    best_rmse = float('inf')
    best_params = {'Rp': Rp0, 'C': C0, 'Rd': Rd0, 'Pd': Pd}
    
    # Search over Rp and Rd (C has smaller effect usually)
    for rp_mult in multipliers:
        for rd_mult in multipliers:
            test_params = {
                'Rp': Rp0 * rp_mult,
                'C': C0,  # Keep C fixed for now
                'Rd': Rd0 * rd_mult,
                'Pd': Pd
            }
            
            P_sim, success = simulate_bc_with_solver(Q_obs, t, 'RCR', test_params, solver_path)
            
            if success and P_sim is not None:
                rmse = np.sqrt(np.mean((P_sim - P_obs)**2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = test_params.copy()
    
    # Fine-tune C around the best Rp, Rd
    for c_mult in multipliers:
        test_params = {
            'Rp': best_params['Rp'],
            'C': C0 * c_mult,
            'Rd': best_params['Rd'],
            'Pd': Pd
        }
        
        P_sim, success = simulate_bc_with_solver(Q_obs, t, 'RCR', test_params, solver_path)
        
        if success and P_sim is not None:
            rmse = np.sqrt(np.mean((P_sim - P_obs)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = test_params.copy()
    
    return best_params


def load_observations(obs_path):
    """Load observations from calibration input file."""
    with open(obs_path, 'r') as f:
        config = json.load(f)
    
    y = config.get('y', {})
    
    # Try to find time array in different places
    t = np.array(config.get('t', []))
    if len(t) == 0 and 'observed_inflow_bc' in config:
        t = np.array(config['observed_inflow_bc'].get('t', []))
    
    # If still no time, construct from simulation parameters
    if len(t) == 0 and 'simulation_parameters' in config:
        sim_params = config['simulation_parameters']
        n_pts = sim_params.get('number_of_time_pts_per_cardiac_cycle', 101)
        period = sim_params.get('cardiac_cycle_period', 1.0)
        t = np.linspace(0, period, n_pts)
    
    return y, t


def load_geometric_input(geo_path):
    """Load geometric input to get BC parameters."""
    with open(geo_path, 'r') as f:
        config = json.load(f)
    return config


def find_bc_for_vessel(config, vessel_name):
    """Find the boundary condition connected to a vessel's outlet."""
    # Look through junctions to find BC connections
    for junc in config.get('junctions', []):
        inlet_vessels = junc.get('inlet_vessels', [])
        outlet_vessels = junc.get('outlet_vessels', [])
        junc_name = junc['junction_name']
        
        # Check if this junction connects vessel to a BC
        for bc in config.get('boundary_conditions', []):
            bc_name = bc['bc_name']
            bc_type = bc['bc_type']
            
            # Check if BC is connected through this junction
            # BC names often match junction patterns
            if bc_name == junc_name or bc_name in str(outlet_vessels):
                return bc, junc_name
    
    # Direct vessel-BC connection check
    for vessel in config.get('vessels', []):
        if vessel['vessel_name'] == vessel_name:
            # Check boundary conditions field
            bc_out = vessel.get('boundary_conditions', {}).get('outlet')
            if bc_out:
                for bc in config.get('boundary_conditions', []):
                    if bc['bc_name'] == bc_out:
                        return bc, bc_out
    
    return None, None


def get_outlet_bcs(config):
    """Get all outlet boundary conditions and their connected vessels."""
    outlet_bcs = []
    
    for vessel in config.get('vessels', []):
        vessel_name = vessel['vessel_name']
        bc_out = vessel.get('boundary_conditions', {}).get('outlet')
        
        if bc_out:
            # Find the BC configuration
            for bc in config.get('boundary_conditions', []):
                if bc['bc_name'] == bc_out:
                    outlet_bcs.append({
                        'vessel_name': vessel_name,
                        'bc_name': bc_out,
                        'bc_type': bc['bc_type'],
                        'bc_values': bc.get('bc_values', {})
                    })
                    break
    
    return outlet_bcs


def test_boundary_conditions(geo_path, obs_path, output_dir):
    """
    Test all outlet boundary conditions.
    
    Args:
        geo_path: Path to geometric input JSON
        obs_path: Path to calibration input JSON with observations
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    config = load_geometric_input(geo_path)
    y_obs, t_obs = load_observations(obs_path)
    
    if len(t_obs) == 0:
        print("No time array found in observations")
        return
    
    dt = t_obs[1] - t_obs[0] if len(t_obs) > 1 else 0.01
    
    # Get outlet BCs
    outlet_bcs = get_outlet_bcs(config)
    print(f"Found {len(outlet_bcs)} outlet boundary conditions")
    
    results = []
    
    for bc_info in outlet_bcs:
        vessel_name = bc_info['vessel_name']
        bc_name = bc_info['bc_name']
        bc_type = bc_info['bc_type']
        bc_values = bc_info['bc_values']
        
        print(f"\nTesting BC: {bc_name} (type: {bc_type}) connected to {vessel_name}")
        
        # Find observations for this vessel outlet
        # Try different key formats
        flow_key = None
        pressure_key = None
        
        for key in y_obs.keys():
            if f'flow:{vessel_name}:' in key or f'flow_out' in key:
                if vessel_name in key:
                    flow_key = key
            if f'pressure:{vessel_name}:' in key or f'pressure_out' in key:
                if vessel_name in key and bc_name in key:
                    pressure_key = key
        
        # Try specific patterns
        possible_flow_keys = [
            f'flow:{vessel_name}:{bc_name}',
            f'flow_out:{vessel_name}',
        ]
        possible_pressure_keys = [
            f'pressure:{vessel_name}:{bc_name}',
            f'pressure_out:{vessel_name}',
        ]
        
        for k in possible_flow_keys:
            if k in y_obs:
                flow_key = k
                break
        
        for k in possible_pressure_keys:
            if k in y_obs:
                pressure_key = k
                break
        
        if flow_key is None:
            print(f"  Warning: No flow observation found for {vessel_name}")
            continue
        if pressure_key is None:
            print(f"  Warning: No pressure observation found for {vessel_name}:{bc_name}")
            continue
        
        print(f"  Flow key: {flow_key}")
        print(f"  Pressure key: {pressure_key}")
        
        Q_obs = np.array(y_obs[flow_key])
        P_obs = np.array(y_obs[pressure_key])
        
        # Print BC parameters
        if bc_type == 'RCR':
            Rp = bc_values.get('Rp', 0)
            C = bc_values.get('C', 0)
            Rd = bc_values.get('Rd', 0)
            Pd = bc_values.get('Pd', 0)
            print(f"  RCR params: Rp={Rp:.2e}, C={C:.2e}, Rd={Rd:.2e}, Pd={Pd:.2f}")
        elif bc_type == 'RESISTANCE':
            R = bc_values.get('R', bc_values.get('Rp', 0))
            Pd = bc_values.get('Pd', 0)
            print(f"  Resistance params: R={R:.2e}, Pd={Pd:.2f}")
        
        # Simulate BC response using svZeroDSolver (UNFITTED)
        print(f"  Simulating with ORIGINAL parameters...")
        P_sim_orig, success = simulate_bc_with_solver(Q_obs, t_obs, bc_type, bc_values)
        
        if not success or P_sim_orig is None:
            print(f"  Falling back to RK4 solver...")
            if bc_type == 'RCR':
                P_sim_orig, _ = simulate_rcr_bc(Q_obs, dt, 
                                           bc_values.get('Rp', 0),
                                           bc_values.get('C', 0),
                                           bc_values.get('Rd', 0),
                                           bc_values.get('Pd', 0))
            elif bc_type == 'RESISTANCE':
                P_sim_orig = simulate_resistance_bc(Q_obs, 
                                               bc_values.get('R', bc_values.get('Rp', 0)),
                                               bc_values.get('Pd', 0))
            else:
                print(f"  Unsupported BC type: {bc_type}")
                continue
        
        if P_sim_orig is None:
            print(f"  Could not simulate BC")
            continue
        
        # Calculate error metrics for original
        rmse_orig = np.sqrt(np.mean((P_sim_orig - P_obs)**2))
        mae_orig = np.mean(np.abs(P_sim_orig - P_obs))
        max_error_orig = np.max(np.abs(P_sim_orig - P_obs))
        
        print(f"  ORIGINAL - RMSE: {rmse_orig:.2f} Pa, MAE: {mae_orig:.2f} Pa, Max Error: {max_error_orig:.2f} Pa")
        
        # Fit RCR parameters and simulate (FITTED)
        P_sim_fitted = None
        fitted_values = None
        rmse_fitted = None
        
        if bc_type == 'RCR':
            print(f"  Fitting RCR parameters (grid search with solver)...")
            fitted_values = fit_rcr_parameters_with_solver(Q_obs, P_obs, t_obs, bc_values)
            print(f"  FITTED params: Rp={fitted_values['Rp']:.2e}, C={fitted_values['C']:.2e}, Rd={fitted_values['Rd']:.2e}, Pd={fitted_values['Pd']:.2f}")
            
            # Simulate with fitted parameters
            P_sim_fitted, success = simulate_bc_with_solver(Q_obs, t_obs, bc_type, fitted_values)
            
            if not success or P_sim_fitted is None:
                P_sim_fitted, _ = simulate_rcr_bc(Q_obs, dt, 
                                                   fitted_values['Rp'],
                                                   fitted_values['C'],
                                                   fitted_values['Rd'],
                                                   fitted_values['Pd'])
            
            if P_sim_fitted is not None:
                rmse_fitted = np.sqrt(np.mean((P_sim_fitted - P_obs)**2))
                mae_fitted = np.mean(np.abs(P_sim_fitted - P_obs))
                max_error_fitted = np.max(np.abs(P_sim_fitted - P_obs))
                print(f"  FITTED - RMSE: {rmse_fitted:.2f} Pa, MAE: {mae_fitted:.2f} Pa, Max Error: {max_error_fitted:.2f} Pa")
        
        results.append({
            'vessel': vessel_name,
            'bc_name': bc_name,
            'bc_type': bc_type,
            'rmse_orig': rmse_orig,
            'mae_orig': mae_orig,
            'max_error_orig': max_error_orig,
            'rmse_fitted': rmse_fitted,
            'fitted_values': fitted_values
        })
        
        # Create plot with both fitted and unfitted
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        
        # Plot 1: Pressure comparison (both fitted and unfitted)
        ax1 = axes[0, 0]
        ax1.plot(t_obs, P_obs, color=COLOR_OBSERVED, linewidth=LINE_WIDTH, label='Observed (1D)')
        ax1.plot(t_obs, P_sim_orig, color=COLOR_SIMULATED, linewidth=LINE_WIDTH, linestyle='--', label=f'Original (RMSE={rmse_orig:.0f})')
        if P_sim_fitted is not None:
            ax1.plot(t_obs, P_sim_fitted, color='green', linewidth=LINE_WIDTH, linestyle=':', label=f'Fitted (RMSE={rmse_fitted:.0f})')
        ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE)
        ax1.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE)
        ax1.set_title(f'Pressure: {vessel_name} → {bc_name}', fontsize=FONT_SIZE)
        ax1.legend(fontsize=FONT_SIZE-2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Flow input
        ax2 = axes[0, 1]
        ax2.plot(t_obs, Q_obs, color='blue', linewidth=LINE_WIDTH)
        ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE)
        ax2.set_ylabel('Flow (mL/s)', fontsize=FONT_SIZE)
        ax2.set_title(f'Input Flow to BC', fontsize=FONT_SIZE)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pressure error (both original and fitted)
        ax3 = axes[1, 0]
        P_error_orig = P_sim_orig - P_obs
        ax3.plot(t_obs, P_error_orig, color=COLOR_SIMULATED, linewidth=LINE_WIDTH, linestyle='--', label='Original')
        if P_sim_fitted is not None:
            P_error_fitted = P_sim_fitted - P_obs
            ax3.plot(t_obs, P_error_fitted, color='green', linewidth=LINE_WIDTH, linestyle=':', label='Fitted')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE)
        ax3.set_ylabel('Pressure Error (Pa)', fontsize=FONT_SIZE)
        ax3.set_title(f'Pressure Error (Simulated - Observed)', fontsize=FONT_SIZE)
        ax3.legend(fontsize=FONT_SIZE-2)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot (both original and fitted)
        ax4 = axes[1, 1]
        ax4.scatter(P_obs, P_sim_orig, alpha=0.5, s=10, color=COLOR_SIMULATED, label=f'Original')
        if P_sim_fitted is not None:
            ax4.scatter(P_obs, P_sim_fitted, alpha=0.5, s=10, color='green', label=f'Fitted')
        p_min = min(P_obs.min(), P_sim_orig.min())
        p_max = max(P_obs.max(), P_sim_orig.max())
        ax4.plot([p_min, p_max], [p_min, p_max], 'k--', alpha=0.5, label='y=x')
        ax4.set_xlabel('Observed Pressure (Pa)', fontsize=FONT_SIZE)
        ax4.set_ylabel('Simulated Pressure (Pa)', fontsize=FONT_SIZE)
        ax4.set_title(f'Observed vs Simulated', fontsize=FONT_SIZE)
        ax4.legend(fontsize=FONT_SIZE-2)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'bc_test_{bc_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {plot_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if results:
        print(f"{'BC Name':<12} {'Type':<10} {'RMSE Orig':>12} {'RMSE Fitted':>12} {'Improvement':>12}")
        print("-"*80)
        for r in sorted(results, key=lambda x: x['rmse_orig'], reverse=True):
            rmse_o = r['rmse_orig']
            rmse_f = r.get('rmse_fitted')
            if rmse_f is not None:
                improvement = (rmse_o - rmse_f) / rmse_o * 100
                print(f"{r['bc_name']:<12} {r['bc_type']:<10} {rmse_o:>12.1f} {rmse_f:>12.1f} {improvement:>11.1f}%")
            else:
                print(f"{r['bc_name']:<12} {r['bc_type']:<10} {rmse_o:>12.1f} {'N/A':>12} {'N/A':>12}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test boundary conditions against observations')
    parser.add_argument('--geo-input', required=True, help='Path to geometric input JSON')
    parser.add_argument('--observations', required=True, help='Path to calibration input with observations')
    parser.add_argument('--output-dir', default='results/bc_test', help='Output directory for plots')
    
    args = parser.parse_args()
    
    test_boundary_conditions(args.geo_input, args.observations, args.output_dir)


if __name__ == '__main__':
    main()
