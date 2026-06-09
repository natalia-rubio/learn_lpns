# imports

import json
import os
import warnings
from math import isfinite

import numpy as np

BC_CHANGE_RATIO_THRESHOLD = 5.0
BC_CHANGE_ABS_THRESHOLD = 1.0

try:
    from scipy.optimize import least_squares

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _warn_if_bc_params_changed(
    bc_name,
    vessel_name,
    bc_type,
    old_values,
    new_values,
    *,
    ratio_threshold=BC_CHANGE_RATIO_THRESHOLD,
    abs_threshold=BC_CHANGE_ABS_THRESHOLD,
):
    """Warn when any fitted BC parameter differs substantially from the original."""
    for param, old_val in old_values.items():
        new_val = new_values.get(param)
        if new_val is None:
            continue
        old_val = float(old_val)
        new_val = float(new_val)
        if abs(old_val) >= abs_threshold:
            if old_val == 0.0:
                ratio = float("inf") if new_val != 0.0 else 1.0
            else:
                ratio = max(abs(new_val / old_val), abs(old_val / new_val))
            if ratio > ratio_threshold:
                msg = f"Large change in {bc_name} ({vessel_name}) {param}: {old_val} -> {new_val} ({ratio:.1f}x)"
                print(f"  Warning: {msg}")
                warnings.warn(msg, stacklevel=3)
        elif abs(new_val - old_val) > abs_threshold:
            msg = (
                f"Large change in {bc_name} ({vessel_name}) {param}: "
                f"{old_val} -> {new_val} (delta {abs(new_val - old_val):.4g})"
            )
            print(f"  Warning: {msg}")
            warnings.warn(msg, stacklevel=3)


def _get_outlet_observations(observations, vessel_name, bc_name):
    """Return (pressures, flows) arrays for an outlet, or (None, None) if missing."""
    obs_y = observations.get("y", {})
    pressure_key = f"pressure:{vessel_name}:{bc_name}"
    flow_key = f"flow:{vessel_name}:{bc_name}"
    if pressure_key not in obs_y or flow_key not in obs_y:
        return None, None
    pressures = np.asarray(obs_y[pressure_key], dtype=float)
    flows = np.asarray(obs_y[flow_key], dtype=float)
    return pressures, flows


def _fit_resistance_outlet(vessel_name, bc_name, bc_cfg, observations):
    """
    Fit RESISTANCE outlet BC (R, Pd) from observations.

    Returns:
        dict with bc_type, R, Pd or None if fit failed/skipped.
    """
    pressures, flows = _get_outlet_observations(observations, vessel_name, bc_name)
    if pressures is None:
        print(f"  Warning: Missing observations for {vessel_name}:{bc_name}, skipping")
        return None

    valid_mask = np.isfinite(pressures) & np.isfinite(flows)
    if not np.any(valid_mask):
        print(f"  Warning: No valid data points for {vessel_name}:{bc_name}")
        return None

    valid_pressures = pressures[valid_mask]
    valid_flows = flows[valid_mask]
    if len(valid_pressures) < 2:
        print(f"  Warning: Insufficient data points for {vessel_name}:{bc_name} (need at least 2)")
        return None

    try:
        A = np.vstack([valid_flows, np.ones(len(valid_flows))]).T
        params, _, _, _ = np.linalg.lstsq(A, valid_pressures, rcond=None)
        fitted_r = float(params[0])
        fitted_pd = float(params[1])
    except np.linalg.LinAlgError as e:
        print(f"  Warning: Linear regression failed for {vessel_name}:{bc_name}: {e}")
        return None

    if not np.isfinite(fitted_r) or not np.isfinite(fitted_pd):
        print(f"  Warning: Invalid fit parameters for {vessel_name}:{bc_name}")
        return None
    if fitted_r < 0:
        print(
            f"  Warning: Negative resistance fitted for {vessel_name}:{bc_name} ({fitted_r:.4f}), using absolute value"
        )
        fitted_r = abs(fitted_r)

    old_values = {
        "R": float(bc_cfg["bc_values"].get("R", 1.0)),
        "Pd": float(bc_cfg["bc_values"].get("Pd", 0.0)),
    }
    new_values = {"R": fitted_r, "Pd": fitted_pd}
    _warn_if_bc_params_changed(bc_name, vessel_name, "RESISTANCE", old_values, new_values)

    predicted = fitted_r * valid_flows + fitted_pd
    ss_res = np.sum((valid_pressures - predicted) ** 2)
    ss_tot = np.sum((valid_pressures - np.mean(valid_pressures)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"  {bc_name} ({vessel_name}) [RESISTANCE]:")
    print(f"    Pressure range: [{np.min(valid_pressures):.2f}, {np.max(valid_pressures):.2f}] dynes/cm²")
    print(f"    Flow range: [{np.min(valid_flows):.2f}, {np.max(valid_flows):.2f}] cm³/s")
    print(f"    R: {fitted_r:.4f} (was {old_values['R']:.4f})")
    print(f"    Pd: {fitted_pd:.4f} (was {old_values['Pd']:.4f})")
    print(f"    R²: {r_squared:.4f}")

    return {"bc_type": "RESISTANCE", "R": fitted_r, "Pd": fitted_pd}


def _fit_rcr_outlet(vessel_name, bc_name, bc_cfg, observations, dt, fit_pd=False):
    """
    Fit RCR outlet BC (Rp, C, Rd, Pd) from observations.

    Returns:
        dict with bc_type, Rp, C, Rd, Pd or None if fit failed/skipped.
    """
    pressures, flows = _get_outlet_observations(observations, vessel_name, bc_name)
    if pressures is None:
        print(f"  Warning: Missing observations for {vessel_name}:{bc_name}, skipping")
        return None

    P = np.asarray(pressures, dtype=float)
    Q = np.asarray(flows, dtype=float)
    if len(P) < 3 or len(Q) < 3:
        print(f"  Warning: Not enough data points for {vessel_name}:{bc_name} (need >=3)")
        return None

    Rp0 = float(bc_cfg["bc_values"].get("Rp", 100.0))
    C0 = float(bc_cfg["bc_values"].get("C", 1e-4))
    Rd0 = float(bc_cfg["bc_values"].get("Rd", 1000.0))
    Pd0 = float(bc_cfg["bc_values"].get("Pd", 0.0))

    def residual(params):
        if fit_pd:
            Rp, C, Rd, Pd = params
        else:
            Rp, C, Rd = params
            Pd = 0.0
        if Rp <= 0 or C <= 0 or Rd <= 0:
            return np.ones_like(P) * 1e6
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
        Rp_fit, C_fit = Rp0, C0
        Pc = P - Rp_fit * Q
        dPc_dt = np.gradient(Pc, dt)
        rhs = Q - C_fit * dPc_dt
        if fit_pd:
            A = np.vstack([Pc, -np.ones_like(Pc)]).T
            try:
                params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                inv_Rd, Pd_over_Rd = params
                Rd_fit = 1.0 / inv_Rd if inv_Rd != 0 else Rd0
                Pd_fit = Pd_over_Rd / inv_Rd if inv_Rd != 0 else Pd0
            except np.linalg.LinAlgError:
                Rd_fit, Pd_fit = Rd0, Pd0
        else:
            try:
                inv_Rd = np.linalg.lstsq(Pc.reshape(-1, 1), rhs, rcond=None)[0][0]
                Rd_fit = 1.0 / inv_Rd if inv_Rd != 0 else Rd0
            except np.linalg.LinAlgError:
                Rd_fit = Rd0
            Pd_fit = 0.0

    if not all(isfinite(x) for x in [Rp_fit, C_fit, Rd_fit, Pd_fit]):
        print(f"  Warning: Non-finite fit for {vessel_name}:{bc_name}, skipping")
        return None
    if Rp_fit <= 0 or C_fit <= 0 or Rd_fit <= 0:
        print(f"  Warning: Non-positive fitted parameters for {vessel_name}:{bc_name}, skipping")
        return None

    old_values = {"Rp": Rp0, "C": C0, "Rd": Rd0, "Pd": Pd0}
    new_values = {"Rp": Rp_fit, "C": C_fit, "Rd": Rd_fit, "Pd": Pd_fit}
    _warn_if_bc_params_changed(bc_name, vessel_name, "RCR", old_values, new_values)

    Pc_fit = P - Rp_fit * Q
    dPc_dt_fit = np.gradient(Pc_fit, dt)
    res_vals = C_fit * dPc_dt_fit - Q + (Pc_fit - Pd_fit) / Rd_fit
    ss_res = np.sum(res_vals**2)
    ss_tot = np.sum((Q - np.mean(Q)) ** 2) + 1e-12
    r2_like = 1 - ss_res / ss_tot

    print(f"  {bc_name} ({vessel_name}) [RCR]:")
    print(f"    Rp: {Rp_fit:.4f} (was {Rp0:.4f})")
    print(f"    C : {C_fit:.6e} (was {C0:.6e})")
    print(f"    Rd: {Rd_fit:.4f} (was {Rd0:.4f})")
    print(f"    Pd: {Pd_fit:.4f} (was {Pd0:.4f}) {'(fixed)' if not fit_pd else ''}")
    print(f"    Residual R^2 (heuristic): {r2_like:.4f}")

    return {"bc_type": "RCR", "Rp": Rp_fit, "C": C_fit, "Rd": Rd_fit, "Pd": Pd_fit}


def fit_bcs_from_observations(geometric_input_path, observations, dt=None, fit_pd=False):
    """
    Fit all vessel outlet BCs from observations; dispatch by bc_type.

    Supports RESISTANCE (linear P = R*Q + Pd) and RCR (Windkessel) outlets.
    Raises ValueError for unsupported outlet bc_type values.

    Args:
        geometric_input_path: Path to geometric 0D input JSON (updated in place)
        observations: Dict with 'y' containing outlet pressure/flow time series
        dt: Timestep for RCR fitting; if None, RCR outlets are skipped with a warning
        fit_pd: Whether to fit Pd in RCR outlets (default False)

    Returns:
        Dict mapping bc_name -> fitted parameter dict (includes 'bc_type' key).
    """
    print("\nFitting outlet boundary conditions from observations...")

    with open(geometric_input_path, "r") as f:
        inp = json.load(f)

    boundary_conditions = inp.get("boundary_conditions", [])
    bc_by_name = {bc.get("bc_name"): bc for bc in boundary_conditions if bc.get("bc_name")}
    vessels = inp.get("vessels", [])

    fitted_bcs = {}
    for vessel in vessels:
        bc_name = vessel.get("boundary_conditions", {}).get("outlet")
        if not bc_name or bc_name not in bc_by_name:
            continue

        vessel_name = vessel.get("vessel_name", bc_name)
        bc_cfg = bc_by_name[bc_name]
        bc_type = bc_cfg.get("bc_type")

        if bc_type == "RESISTANCE":
            result = _fit_resistance_outlet(vessel_name, bc_name, bc_cfg, observations)
        elif bc_type == "RCR":
            if dt is None:
                print(f"  Warning: dt unavailable; skipping RCR fit for {vessel_name}:{bc_name}")
                continue
            result = _fit_rcr_outlet(vessel_name, bc_name, bc_cfg, observations, dt, fit_pd=fit_pd)
        elif bc_type == "CORONARY":
            print(f"  Skipping {vessel_name}:{bc_name} (CORONARY BC; using values already in geometric input)")
            continue
        else:
            raise ValueError(
                f"Unsupported outlet bc_type '{bc_type}' for {bc_name} ({vessel_name}); expected RESISTANCE or RCR"
            )

        if result is not None:
            fitted_bcs[bc_name] = result

    if fitted_bcs:
        print("\n  Updating geometric input with fitted outlet BC parameters...")
        for bc_name, params in fitted_bcs.items():
            bc_cfg = bc_by_name.get(bc_name)
            if not bc_cfg:
                continue
            bc_type = params["bc_type"]
            if bc_type == "RESISTANCE":
                old = bc_cfg.get("bc_values", {})
                bc_cfg["bc_values"]["R"] = float(params["R"])
                bc_cfg["bc_values"]["Pd"] = float(params["Pd"])
                print(
                    f"    {bc_name}: R {old.get('R', 0):.4f}->{params['R']:.4f}, "
                    f"Pd {old.get('Pd', 0):.4f}->{params['Pd']:.4f}"
                )
            elif bc_type == "RCR":
                old = bc_cfg.get("bc_values", {})
                bc_cfg["bc_values"]["Rp"] = float(params["Rp"])
                bc_cfg["bc_values"]["C"] = float(params["C"])
                bc_cfg["bc_values"]["Rd"] = float(params["Rd"])
                bc_cfg["bc_values"]["Pd"] = float(params["Pd"])
                print(
                    f"    {bc_name}: Rp {old.get('Rp', 0):.4f}->{params['Rp']:.4f}, "
                    f"C {old.get('C', 0):.6e}->{params['C']:.6e}, "
                    f"Rd {old.get('Rd', 0):.4f}->{params['Rd']:.4f}, "
                    f"Pd {old.get('Pd', 0):.4f}->{params['Pd']:.4f}"
                )

        with open(geometric_input_path, "w") as f:
            json.dump(inp, f, indent=4)
        print(f"  ✓ Updated geometric input saved to: {geometric_input_path}")
    else:
        print("  No outlet BC parameters were fitted.")

    return fitted_bcs


def apply_fitted_outlet_bcs_to_file(file_path, fitted_bcs, file_type="config"):
    """
    Apply previously fitted outlet BC parameters to a JSON config file.

    Args:
        file_path: Path to geometric or calibration input JSON
        fitted_bcs: Dict from fit_bcs_from_observations (bc_name -> param dict)
        file_type: Label for logging

    Returns:
        True if at least one BC was updated, False otherwise.
    """
    if not fitted_bcs:
        return False
    if not os.path.exists(file_path):
        print(f"  Warning: {file_type} file not found: {file_path}")
        return False

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        bc_by_name = {bc.get("bc_name"): bc for bc in data.get("boundary_conditions", []) if bc.get("bc_name")}
        updated_count = 0
        for bc_name, params in fitted_bcs.items():
            bc_cfg = bc_by_name.get(bc_name)
            if not bc_cfg:
                continue
            bc_type = params.get("bc_type")
            if bc_type == "RESISTANCE" and bc_cfg.get("bc_type") == "RESISTANCE":
                old_r = bc_cfg["bc_values"].get("R", 1.0)
                old_pd = bc_cfg["bc_values"].get("Pd", 0.0)
                bc_cfg["bc_values"]["R"] = float(params["R"])
                bc_cfg["bc_values"]["Pd"] = float(params["Pd"])
                print(f"    {bc_name}: R {old_r:.4f} -> {params['R']:.4f}, Pd {old_pd:.4f} -> {params['Pd']:.4f}")
                updated_count += 1
            elif bc_type == "RCR" and bc_cfg.get("bc_type") == "RCR":
                old = bc_cfg.get("bc_values", {})
                bc_cfg["bc_values"]["Rp"] = float(params["Rp"])
                bc_cfg["bc_values"]["C"] = float(params["C"])
                bc_cfg["bc_values"]["Rd"] = float(params["Rd"])
                bc_cfg["bc_values"]["Pd"] = float(params["Pd"])
                print(
                    f"    {bc_name}: Rp {old.get('Rp', 0):.4f}->{params['Rp']:.4f}, "
                    f"C {old.get('C', 0):.6e}->{params['C']:.6e}, "
                    f"Rd {old.get('Rd', 0):.4f}->{params['Rd']:.4f}, "
                    f"Pd {old.get('Pd', 0):.4f}->{params['Pd']:.4f}"
                )
                updated_count += 1

        if updated_count > 0:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  ✓ Updated {updated_count} outlet BC(s) in {file_type}: {file_path}")
            return True

        print(f"  No matching outlet BCs found to update in {file_type}: {file_path}")
        return False
    except Exception as e:
        print(f"  Warning: Could not update {file_type} file {file_path}: {e}")
        return False
