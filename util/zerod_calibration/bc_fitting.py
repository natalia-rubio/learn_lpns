# imports

import os
import json
import csv
import numpy as np


def fit_outlet_rcr_from_observations(geometric_input_path, observations, dt=None, fit_pd=False):
    """
    Fit RCR (Windkessel) boundary condition parameters (Rp, C, Rd, Pd) from observations.

    Uses measured outlet pressure/flow time series (from 1D solution) and fits
    the Windkessel equations via nonlinear least squares:

        C * dPc/dt - Q + (Pc - Pd)/Rd = 0
        Pc = P_in - Rp * Q

    Args:
        geometric_input_path: Path to geometric 0D input JSON (will be updated)
        observations: Dict with 'y' (and optionally 'dy') containing outlet pressure/flow
                      keys in the form pressure:{vessel}:{bc_name}, flow:{vessel}:{bc_name}
        dt: Optional timestep (seconds). If None, attempt to infer from the inlet BC time array.

    Returns:
        Dict mapping bc_name -> (Rp, C, Rd, Pd) for fitted outlets.
    """
    from math import isfinite

    try:
        from scipy.optimize import least_squares
        HAS_SCIPY = True
    except Exception:
        HAS_SCIPY = False

    print("\nFitting RCR outlet boundary conditions from observations...")

    # Load geometric input
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)

    boundary_conditions = inp.get('boundary_conditions', [])
    vessels = inp.get('vessels', [])

    # Infer dt from inlet BC time array if not provided
    if dt is None:
        try:
            for bc in boundary_conditions:
                if bc.get('bc_name') == 'INFLOW' and 'bc_values' in bc and 't' in bc['bc_values']:
                    t_arr = bc['bc_values']['t']
                    if len(t_arr) > 1:
                        dt = float(t_arr[1] - t_arr[0])
                        break
        except Exception:
            pass
    if dt is None:
        print("  Warning: Could not determine timestep (dt). Skipping RCR fitting.")
        return {}

    # Map BC name -> bc config
    bc_by_name = {bc.get('bc_name'): bc for bc in boundary_conditions if bc.get('bc_name')}

    # Collect outlet RCR vessels
    outlet_params = {}
    for vessel in vessels:
        bc_name = vessel.get('boundary_conditions', {}).get('outlet')
        if not bc_name or bc_name not in bc_by_name:
            continue
        bc_cfg = bc_by_name[bc_name]
        if bc_cfg.get('bc_type') != 'RCR':
            continue

        vessel_name = vessel.get('vessel_name')
        key_p = f"pressure:{vessel_name}:{bc_name}"
        key_q = f"flow:{vessel_name}:{bc_name}"

        if key_p not in observations.get('y', {}) or key_q not in observations.get('y', {}):
            print(f"  Warning: Missing observations for {vessel_name}:{bc_name}, skipping")
            continue

        P = np.array(observations['y'][key_p], dtype=float)
        Q = np.array(observations['y'][key_q], dtype=float)

        if len(P) < 3 or len(Q) < 3:
            print(f"  Warning: Not enough data points for {vessel_name}:{bc_name} (need >=3)")
            continue

        # Initial guesses from geometric input
        Rp0 = float(bc_cfg['bc_values'].get('Rp', 100.0))
        C0 = float(bc_cfg['bc_values'].get('C', 1e-4))
        Rd0 = float(bc_cfg['bc_values'].get('Rd', 1000.0))
        Pd0 = float(bc_cfg['bc_values'].get('Pd', 0.0))

        def residual(params):
            if fit_pd:
                Rp, C, Rd, Pd = params
            else:
                Rp, C, Rd = params
                Pd = 0.0
            if Rp <= 0 or C <= 0 or Rd <= 0:
                return np.ones_like(P) * 1e6  # penalize invalid
            Pc = P - Rp * Q
            dPc_dt = np.gradient(Pc, dt)
            return C * dPc_dt - Q + (Pc - Pd) / Rd

        if HAS_SCIPY:
            if fit_pd:
                bounds = ([1e-6, 1e-9, 1e-6, -1e6], [1e8, 1e3, 1e8, 1e6])
                res = least_squares(residual, x0=[Rp0, C0, Rd0, Pd0], bounds=bounds, max_nfev=200)
                Rp_fit, C_fit, Rd_fit, Pd_fit = res.x
            else:
                bounds = ([1e-6, 1e-9, 1e-6], [1e8, 1e3, 1e8])
                res = least_squares(residual, x0=[Rp0, C0, Rd0], bounds=bounds, max_nfev=200)
                Rp_fit, C_fit, Rd_fit = res.x
                Pd_fit = 0.0
        else:
            # Fallback: keep Rp,C, solve Rd (and Pd if allowed) linearly
            Rp_fit, C_fit = Rp0, C0
            Pc = P - Rp_fit * Q
            dPc_dt = np.gradient(Pc, dt)
            rhs = Q - C_fit * dPc_dt  # equals (Pc - Pd)/Rd
            if fit_pd:
                A = np.vstack([Pc, -np.ones_like(Pc)]).T  # [Pc, -1] * [1/Rd, Pd/Rd]^T = rhs
                try:
                    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                    inv_Rd, Pd_over_Rd = params
                    Rd_fit = 1.0 / inv_Rd if inv_Rd != 0 else Rd0
                    Pd_fit = Pd_over_Rd / inv_Rd if inv_Rd != 0 else Pd0
                except np.linalg.LinAlgError:
                    Rd_fit, Pd_fit = Rd0, Pd0
            else:
                # Pd fixed to 0 -> rhs = Pc / Rd -> solve Rd only
                try:
                    inv_Rd = np.linalg.lstsq(Pc.reshape(-1,1), rhs, rcond=None)[0][0]
                    Rd_fit = 1.0 / inv_Rd if inv_Rd != 0 else Rd0
                except np.linalg.LinAlgError:
                    Rd_fit = Rd0
                Pd_fit = 0.0

        # Validate
        if not all(isfinite(x) for x in [Rp_fit, C_fit, Rd_fit, Pd_fit]):
            print(f"  Warning: Non-finite fit for {vessel_name}:{bc_name}, skipping")
            continue
        if Rp_fit <= 0 or C_fit <= 0 or Rd_fit <= 0:
            print(f"  Warning: Non-positive fitted parameters for {vessel_name}:{bc_name}, skipping")
            continue

        outlet_params[bc_name] = (Rp_fit, C_fit, Rd_fit, Pd_fit)

        # Compute R^2-like metric for residuals
        Pc_fit = P - Rp_fit * Q
        dPc_dt_fit = np.gradient(Pc_fit, dt)
        res_vals = C_fit * dPc_dt_fit - Q + (Pc_fit - Pd_fit) / Rd_fit
        ss_res = np.sum(res_vals ** 2)
        ss_tot = np.sum((Q - np.mean(Q)) ** 2) + 1e-12
        r2_like = 1 - ss_res / ss_tot

        print(f"  {bc_name} ({vessel_name}):")
        print(f"    Rp: {Rp_fit:.4f} (was {Rp0:.4f})")
        print(f"    C : {C_fit:.6e} (was {C0:.6e})")
        print(f"    Rd: {Rd_fit:.4f} (was {Rd0:.4f})")
        print(f"    Pd: {Pd_fit:.4f} (was {Pd0:.4f}) {'(fixed)' if not fit_pd else ''}")
        print(f"    Residual R^2 (heuristic): {r2_like:.4f}")

    # Update geometric input
    if outlet_params:
        print("\n  Updating geometric input with fitted RCR parameters...")
        for bc_name, (Rp, C, Rd, Pd) in outlet_params.items():
            if bc_name in bc_by_name:
                bc_cfg = bc_by_name[bc_name]
                old = bc_cfg.get('bc_values', {})
                bc_cfg['bc_values']['Rp'] = float(Rp)
                bc_cfg['bc_values']['C'] = float(C)
                bc_cfg['bc_values']['Rd'] = float(Rd)
                bc_cfg['bc_values']['Pd'] = float(Pd)
                print(f"    {bc_name}: Rp {old.get('Rp', 0):.4f}->{Rp:.4f}, "
                      f"C {old.get('C', 0):.6e}->{C:.6e}, "
                      f"Rd {old.get('Rd', 0):.4f}->{Rd:.4f}, "
                      f"Pd {old.get('Pd', 0):.4f}->{Pd:.4f}")

        with open(geometric_input_path, 'w') as f:
            json.dump(inp, f, indent=4)
        print(f"  ✓ Updated geometric input saved to: {geometric_input_path}")
    else:
        print("  No RCR parameters were fitted.")

    return outlet_params


def read_zerod_csv(csv_path):
    """
    Read 0D simulation results from CSV.
    Handles both 'location' and 'name' as the vessel identifier column.
    
    Returns:
        results: Dictionary {location: {time: {field: value}}}
        times: Sorted list of time values
    """
    results = {}
    times = set()
    
    if not os.path.exists(csv_path):
        return results, sorted(times)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Check which column name is used for vessel identifier
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return results, sorted(times)
        
        vessel_col = None
        if 'location' in fieldnames:
            vessel_col = 'location'
        elif 'name' in fieldnames:
            vessel_col = 'name'
        else:
            return results, sorted(times)
        
        for row in reader:
            location = row[vessel_col]
            time = float(row['time'])
            times.add(time)
            
            if location not in results:
                results[location] = {}
            if time not in results[location]:
                results[location][time] = {}
            
            # Extract all numeric fields
            for key, value in row.items():
                if key not in [vessel_col, 'time']:
                    try:
                        results[location][time][key] = float(value)
                    except (ValueError, TypeError):
                        continue
    
    return results, sorted(times)
    
def fit_outlet_resistances_from_3d(geometric_input_path, observations):
    """
    Fit outlet boundary condition resistances and distal pressures from 3D solution observations.
    Fits linear relationship: P = R*Q + Pd using least squares regression.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON (will be updated)
        observations: Dictionary with observation data (y, dy) containing outlet pressure and flow
    
    Returns:
        Dictionary mapping outlet BC names to fitted (R, Pd) tuples
    """
    print(f"\nFitting outlet resistances and distal pressures from 3D solution...")
    
    # Load geometric input
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Extract outlet resistances and distal pressures
    outlet_params = {}
    
    # Find all outlet BCs in geometric input
    outlet_bcs = {}
    for bc in inp.get('boundary_conditions', []):
        if bc.get('bc_type') == 'RESISTANCE':
            bc_name = bc.get('bc_name')
            if bc_name:
                outlet_bcs[bc_name] = bc
    
    # Find vessels with outlet BCs
    vessels = inp.get('vessels', [])
    vessel_to_bc = {}
    for vessel in vessels:
        if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
            bc_name = vessel['boundary_conditions']['outlet']
            vessel_name = vessel['vessel_name']
            vessel_to_bc[vessel_name] = bc_name
    
    print(f"  Found {len(outlet_bcs)} outlet boundary conditions")
    
    # Extract pressure and flow for each outlet
    obs_y = observations.get('y', {})
    
    for vessel_name, bc_name in vessel_to_bc.items():
        if bc_name not in outlet_bcs:
            print(f"  Warning: BC {bc_name} not found in boundary_conditions")
            continue
        
        # Look for pressure and flow observations
        # Pattern: "pressure:{vessel_name}:{bc_name}" and "flow:{vessel_name}:{bc_name}"
        pressure_key = f"pressure:{vessel_name}:{bc_name}"
        flow_key = f"flow:{vessel_name}:{bc_name}"
        
        if pressure_key not in obs_y:
            print(f"  Warning: No pressure observation found for {vessel_name}:{bc_name}")
            continue
        
        if flow_key not in obs_y:
            print(f"  Warning: No flow observation found for {vessel_name}:{bc_name}")
            continue
        
        pressures = np.array(obs_y[pressure_key])
        flows = np.array(obs_y[flow_key])
        
        # Convert to numpy arrays if needed
        if isinstance(pressures, list):
            pressures = np.array(pressures)
        if isinstance(flows, list):
            flows = np.array(flows)
        
        # Filter out invalid values (inf, nan)
        valid_mask = np.isfinite(pressures) & np.isfinite(flows)
        
        if not np.any(valid_mask):
            print(f"  Warning: No valid data points for {vessel_name}:{bc_name}")
            if pressures.size:
                print(
                    f"    Pressure range: [{np.nanmin(pressures):.2f}, {np.nanmax(pressures):.2f}]"
                )
            else:
                print("    Pressure: (empty array)")
            if flows.size:
                print(
                    f"    Flow range: [{np.nanmin(flows):.2f}, {np.nanmax(flows):.2f}]"
                )
            else:
                print("    Flow: (empty array)")
            continue
        
        valid_pressures = pressures[valid_mask]
        valid_flows = flows[valid_mask]
        
        if len(valid_pressures) < 2:
            print(f"  Warning: Insufficient data points for {vessel_name}:{bc_name} (need at least 2)")
            continue
        
        # Fit linear relationship: P = R*Q + Pd
        # Using least squares: [R, Pd] = (Q^T * Q)^(-1) * Q^T * P
        # Where Q is the design matrix: [flows, ones]
        try:
            # Create design matrix: [flows, ones] for [R, Pd]
            A = np.vstack([valid_flows, np.ones(len(valid_flows))]).T
            b = valid_pressures
            
            # Solve least squares: [R, Pd] = (A^T * A)^(-1) * A^T * b
            params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            fitted_resistance = float(params[0])
            fitted_pd = float(params[1])
            
            # Check if fit is reasonable
            if not np.isfinite(fitted_resistance) or not np.isfinite(fitted_pd):
                print(f"  Warning: Invalid fit parameters for {vessel_name}:{bc_name}")
                continue
            
            # Check if resistance is positive (should be for physical validity)
            if fitted_resistance < 0:
                print(f"  Warning: Negative resistance fitted for {vessel_name}:{bc_name} ({fitted_resistance:.4f}), using absolute value")
                fitted_resistance = abs(fitted_resistance)
            
            outlet_params[bc_name] = (fitted_resistance, fitted_pd)
            
            # Calculate R-squared for quality assessment
            predicted_pressures = fitted_resistance * valid_flows + fitted_pd
            ss_res = np.sum((valid_pressures - predicted_pressures) ** 2)
            ss_tot = np.sum((valid_pressures - np.mean(valid_pressures)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"  {bc_name} ({vessel_name}):")
            print(f"    Pressure range: [{np.min(valid_pressures):.2f}, {np.max(valid_pressures):.2f}] dynes/cm²")
            print(f"    Flow range: [{np.min(valid_flows):.2f}, {np.max(valid_flows):.2f}] cm³/s")
            print(f"    Fitted R: {fitted_resistance:.4f}")
            print(f"    Fitted Pd: {fitted_pd:.4f}")
            print(f"    R²: {r_squared:.4f}")
            
        except np.linalg.LinAlgError as e:
            print(f"  Warning: Linear regression failed for {vessel_name}:{bc_name}: {e}")
            continue
    
    # Update geometric input with fitted resistances and distal pressures
    print(f"\n  Updating geometric input with fitted parameters...")
    for bc_name, (resistance, pd) in outlet_params.items():
        if bc_name in outlet_bcs:
            old_resistance = outlet_bcs[bc_name]['bc_values'].get('R', 1.0)
            old_pd = outlet_bcs[bc_name]['bc_values'].get('Pd', 0.0)
            outlet_bcs[bc_name]['bc_values']['R'] = resistance
            outlet_bcs[bc_name]['bc_values']['Pd'] = pd
            print(f"    {bc_name}: R {old_resistance:.4f} -> {resistance:.4f}, Pd {old_pd:.4f} -> {pd:.4f}")
    
    # Save updated geometric input
    with open(geometric_input_path, 'w') as f:
        json.dump(inp, f, indent=4)
    
    print(f"  ✓ Updated geometric input saved to: {geometric_input_path}")
    
    return outlet_params