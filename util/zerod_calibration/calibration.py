import os
import re
import json
import tempfile
import numpy as np
from util.zerod_calibration.file_io import timestep_from_1D, convert_numpy_to_list
from util.zerod_calibration.bifurcation_splitting import junction_outlet_count

# Default L2 penalties when not set-specific (R_poiseuille, stenosis_coefficient)
DEFAULT_L2_R = 10**5
DEFAULT_L2_STENOSIS = 10**10

# Per-set L2 penalties (set_name -> (L2_penalty_R_poiseuille, L2_penalty_stenosis_coefficient)).
# Add entries to tune calibration by anatomy/set; unlisted sets use DEFAULT_L2_*.
SET_L2_PENALTIES = {
    "VMR_abdo": (10**2, 10**5),
    "VMR_rigid_aorta_adults": (10**5, 10**10),
}


def _flow_inflow_observation_key(observations):
    """Return the first ``flow:INFLOW:*`` key in observations['y'], or None."""
    y = observations.get("y")
    if not isinstance(y, dict):
        return None
    for k in y:
        if k.startswith("flow:INFLOW:"):
            return k
    return None


def _geometric_inflow_tq_lists(inp):
    """Return (t, Q) as plain float lists from the INFLOW FLOW BC, or (None, None)."""
    for bc in inp.get("boundary_conditions") or []:
        if bc.get("bc_name") != "INFLOW":
            continue
        bv = bc.get("bc_values") or {}
        t = bv.get("t")
        q = bv.get("Q")
        if t is None or q is None:
            continue
        if isinstance(t, np.ndarray):
            t = t.tolist()
        if isinstance(q, np.ndarray):
            q = q.tolist()
        try:
            t = [float(x) for x in t]
            q = [float(x) for x in q]
        except (TypeError, ValueError):
            continue
        if len(t) == len(q) and len(t) > 0:
            return t, q
    return None, None


def repeat_observations_in_time(observations, num_repeats=5):
    """
    Repeat each observation series in time by concatenating the series num_repeats times.
    Used when the 1D solution is short (e.g. one cardiac cycle) to extend for calibration.
    """
    result = {"y": {}, "dy": {}}
    for key in observations["y"]:
        arr = np.asarray(observations["y"][key])
        result["y"][key] = np.concatenate([arr] * num_repeats)
    for key in observations.get("dy", []):
        arr = np.asarray(observations["dy"][key])
        result["dy"][key] = np.concatenate([arr] * num_repeats)
    if not result["dy"]:
        result.pop("dy")
    return result


def create_calibration_input(geometric_input_path, observations, output_path, centerline_soln_path=None, geo_dir=None, stenosis_off=False, penalty_off=False, set_name=None):
    """
    Create calibration input file from geometric input and observations.
    BC times use ``len(inflow)`` samples spaced by ``time_step_size`` when the observed
    inflow series has more than one sample. If the observed inflow has at most one sample
    (e.g. steady 1D), INFLOW ``(t, Q)`` is taken from the geometric JSON instead.
    Prefer ``simulation_parameters.time_step_size`` from the geometric JSON (Richter / svZeroD);
    otherwise derive from the 1D VTP and ``geo_dir`` (XML or VMR/TST path fallbacks).
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        observations: Dictionary with observation data (y, dy)
        output_path: Path to save calibration input JSON
        centerline_soln_path: Path to 1D centerline solution VTP (to extract timestep count)
        geo_dir: Geometry directory (to find XML file for timestep size)
        stenosis_off: If True, set calibrate_stenosis_coefficient False and set all stenosis to 0
        penalty_off: If True (and stenosis_off is False), set L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient to 0. Incompatible with stenosis_off.
        set_name: Optional set name (e.g. VMR_abdo) used to look up set-specific L2 penalties from SET_L2_PENALTIES; unlisted sets use defaults.
    """
    if stenosis_off and penalty_off:
        raise ValueError("Cannot use both --stenosis-off and --penalty-off.")
    print(f"Reading geometric input from: {geometric_input_path}")
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Compute BC times directly from 1D solution timesteps (no refinement/interpolation)
    bc_time = None

    geo_bc_t, geo_bc_q = _geometric_inflow_tq_lists(inp)
    inflow_key = _flow_inflow_observation_key(observations)

    obs_bc_flow_list = None
    obs_bc_len = 0
    if inflow_key:
        obs_bc_flow_list = observations["y"][inflow_key]
        if isinstance(obs_bc_flow_list, np.ndarray):
            obs_bc_flow_list = obs_bc_flow_list.tolist()
        if obs_bc_flow_list is None:
            obs_bc_flow_list = []
        obs_bc_len = len(obs_bc_flow_list)

    # Linspace BC times need time_step_size; copying geometric (t, Q) does not.
    need_time_step_size = True
    if not inflow_key:
        need_time_step_size = geo_bc_t is None
    else:
        if obs_bc_len > 1:
            need_time_step_size = True
        elif geo_bc_t is not None:
            need_time_step_size = False
        elif obs_bc_len == 1:
            need_time_step_size = True
        else:
            need_time_step_size = False

    time_step_size = None
    sim_params = inp.get("simulation_parameters")
    if isinstance(sim_params, dict):
        for key in ("time_step_size", "Time_step_size", "fixed_time_step_size"):
            if key not in sim_params or sim_params[key] is None:
                continue
            try:
                time_step_size = float(sim_params[key])
                print(
                    f"  Using time_step_size from geometric input "
                    f"simulation_parameters['{key}'] = {time_step_size:.6f} s"
                )
                break
            except (TypeError, ValueError):
                continue

    if time_step_size is None and centerline_soln_path and geo_dir:
        time_step_size = timestep_from_1D(centerline_soln_path, geo_dir)

    if time_step_size is None and need_time_step_size:
        raise ValueError(
            "Could not determine time_step_size for calibration BC times: set "
            "simulation_parameters.time_step_size in geometric_input.json, or pass "
            "centerline_soln_path and geo_dir so timestep_from_1D can run (XML / VMR / TST-cohort)."
        )
    if time_step_size is None:
        print(
            "  No time_step_size in geometric JSON and no 1D/XML path; not required "
            "(INFLOW (t, Q) taken from geometric JSON or single-sample linspace not used)."
        )

    if not inflow_key:
        if geo_bc_t is None:
            raise ValueError(
                "No flow:INFLOW:* observation found in observations and no usable INFLOW (t, Q) "
                "in geometric boundary_conditions."
            )
        bc_time = list(geo_bc_t)
        bc_flow = list(geo_bc_q)
        print("  No flow:INFLOW:* in observations; using geometric INFLOW BC (t, Q).")
    else:
        bc_flow = list(obs_bc_flow_list)

        if len(bc_flow) > 1:
            bc_time = np.linspace(
                0.0, len(bc_flow) * time_step_size, len(bc_flow), endpoint=False
            ).tolist()
        elif geo_bc_t is not None:
            bc_time = list(geo_bc_t)
            bc_flow = list(geo_bc_q)
            print(
                f"  Inflow observation {inflow_key} has <=1 sample; "
                f"using original geometric INFLOW BC ({len(bc_time)} samples)."
            )
        elif len(bc_flow) == 1:
            bc_time = np.linspace(
                0.0, len(bc_flow) * time_step_size, len(bc_flow), endpoint=False
            ).tolist()
        else:
            raise ValueError(
                f"Inflow observation series {inflow_key} is empty and geometric INFLOW has no "
                "usable (t, Q). Check 1D VTP / observation extraction (single-timestep: end_idx=None, not -1)."
            )

    
    # Store full time frame for geometric input (forward simulations use full time)
    bc_time_full = bc_time.copy()
    bc_flow_full = bc_flow.copy()

    # Store full BC in inp for later use in updating geometric input
    inp["_full_bc_time"] = bc_time_full
    inp["_full_bc_flow"] = bc_flow_full

    # Update inflow BC with flow (for calibration input only)
    for bc in inp["boundary_conditions"]:
        if bc["bc_name"] == "INFLOW":
            bc["bc_values"]["t"] = bc_time
            bc["bc_values"]["Q"] = bc_flow
            break
        
    # Keep geometric parameters (R_poiseuille, C, L, stenosis_coefficient) from geometric input
    # These will serve as initial values for calibration

    # Stenosis-off mode: do not calibrate stenosis and set all stenosis coefficients to 0
    if stenosis_off:
        for v in inp.get("vessels", []):
            if "zero_d_element_values" in v and "stenosis_coefficient" in v["zero_d_element_values"]:
                v["zero_d_element_values"]["stenosis_coefficient"] = 0.0
        for j in inp.get("junctions", []):
            if "junction_values" in j and "stenosis_coefficient" in j["junction_values"]:
                sv = j["junction_values"]["stenosis_coefficient"]
                n_out = len(sv) if isinstance(sv, list) else 1
                j["junction_values"]["stenosis_coefficient"] = [0.0] * n_out
        print("  Stenosis-off: all stenosis coefficients set to 0, calibrate_stenosis_coefficient=False, L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient set to 0")
    
    # Add calibration parameters: L2 penalties depend on set_name when not penalty_off/stenosis_off
    if stenosis_off or penalty_off:
        l2_R = 0.0
        l2_stenosis = 0.0
        if penalty_off and not stenosis_off:
            print("  Penalty-off: L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient set to 0")
    else:
        l2_R, l2_stenosis = SET_L2_PENALTIES.get(set_name, (DEFAULT_L2_R, DEFAULT_L2_STENOSIS))
        if set_name and set_name in SET_L2_PENALTIES:
            print(f"  Set-specific L2 penalties for {set_name}: R={l2_R}, stenosis={l2_stenosis}")
    inp["calibration_parameters"] = {
        "tolerance_gradient": 1e-4,
        "tolerance_increment": 1e-4,
        "maximum_iterations": 200000,
        "calibrate_stenosis_coefficient": not stenosis_off,
        "calibrate_capacitance": False,
        "set_capacitance_to_zero": False,
        "L2_penalty_R_poiseuille": l2_R,
        "L2_penalty_stenosis_coefficient": l2_stenosis,
        "L2_penalty_L": 0
    }
    
    inp["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = len(bc_time)
    inp["simulation_parameters"]["output_all_cycles"] = True
    inp["simulation_parameters"]["steady_initial"] = False
    inp["simulation_parameters"]["absolute_tolerance"] = 1e-5
    inp["simulation_parameters"]["maximum_nonlinear_iterations"] = 50
    inp["simulation_parameters"]["num_cardiac_cycles"] = 1
    
    # Convert numpy arrays in observations to lists for JSON serialization
    observations_list = convert_numpy_to_list(observations)
    
    # Add observations
    inp.update(observations_list)
    
    # Store full observations for plotting (3D solution should show full time series)
    inp['_full_observations'] = observations_list
    
    # Store original observed inflow BC for later use in calibrated output
    if '_full_bc_time' in inp and '_full_bc_flow' in inp:
        inp['observed_inflow_bc'] = {
            't': inp['_full_bc_time'].copy(),
            'Q': inp['_full_bc_flow'].copy()
        }
        # Remove temporary fields
        del inp['_full_bc_time']
        del inp['_full_bc_flow']
    
    # Write calibration input
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(inp, f, indent=4)
    
    print(f"Calibration input saved to: {output_path}")
    return inp

def _sanitize_svzerod_calibration_topology(config):
    """
    Stock svzerodcalibrator BloodVesselJunction wiring allows exactly one inlet edge.

    Some geometric JSON may list multiple ``inlet_blocks`` or keep redundant ``inlet_vessels``
    alongside ``inlet_blocks``; trim so the calibrator does not hit
    "Blood vessel junction does not support multiple inlets."
    """
    for junc in config.get("junctions") or []:
        jn = junc.get("junction_name", "?")
        ib = junc.get("inlet_blocks")
        if isinstance(ib, list) and len(ib) > 1:
            print(
                f"  Warning: junction {jn!r} lists {len(ib)} inlet_blocks; "
                f"keeping only {ib[0]!r} for svzerodcalibrator."
            )
            junc["inlet_blocks"] = [ib[0]]
        if (
            isinstance(junc.get("inlet_blocks"), list)
            and len(junc["inlet_blocks"]) > 0
            and junc.get("inlet_vessels")
        ):
            junc.pop("inlet_vessels", None)


def _resolve_vessel_block_name(block, vessel_names, junction_names):
    """
    Map a junction ``inlet_blocks`` / ``outlet_blocks`` vessel label to an actual
    ``vessel_name`` when EL splitting renamed the segment (e.g. branch1_seg0 ->
    branch1_seg0_connectorEL) but JSON still lists the pre-EL name.
    """
    if not block or not isinstance(block, str):
        return block
    if block in junction_names or block in vessel_names:
        return block
    cand = f"{block}_connectorEL"
    if cand in vessel_names:
        return cand
    pat = re.compile(r"^" + re.escape(block) + r"_connectorEL\d*$", re.IGNORECASE)
    matches = [vn for vn in vessel_names if vn and pat.match(vn)]
    if len(matches) == 1:
        return matches[0]
    el_like = [
        vn
        for vn in vessel_names
        if vn and vn.startswith(block + "_") and "connectorel" in vn.lower()
    ]
    if len(el_like) == 1:
        return el_like[0]
    return block


def _normalize_el_stale_vessel_blocks_for_calibrator(config):
    """
    svzerodcalibrator resolves ``inlet_blocks`` / ``outlet_blocks`` strings with
    ``Model::get_block``; names must match ``vessels[].vessel_name``. After
    bifurcations_EL, lists may still reference the pre-EL segment name while the
    vessel was renamed (e.g. *_connectorEL). Rewrite blocks and ``y`` / ``dy``
    observation keys to match.
    """
    vessels = config.get("vessels") or []
    junctions = config.get("junctions") or []
    vessel_names = {v.get("vessel_name") for v in vessels if v.get("vessel_name")}
    junction_names = {j.get("junction_name") for j in junctions if j.get("junction_name")}
    if not vessel_names:
        return

    tokens = set()

    def _collect_obs_tokens(d):
        if not isinstance(d, dict):
            return
        for k in d:
            if isinstance(k, str) and k.count(":") == 2:
                _, a, b = k.split(":", 2)
                tokens.add(a)
                tokens.add(b)

    for junc in junctions:
        for key in ("inlet_blocks", "outlet_blocks"):
            blocks = junc.get(key)
            if not isinstance(blocks, list):
                continue
            for b in blocks:
                if isinstance(b, str):
                    tokens.add(b)

    for d in (config.get("y"), config.get("dy")):
        _collect_obs_tokens(d)
    full = config.get("_full_observations")
    if isinstance(full, dict):
        for d in (full.get("y"), full.get("dy")):
            _collect_obs_tokens(d)

    block_fix = {}
    for t in tokens:
        r = _resolve_vessel_block_name(t, vessel_names, junction_names)
        if r != t:
            block_fix[t] = r

    if not block_fix:
        return

    print(
        f"  [calibrate] normalized {len(block_fix)} stale vessel block label(s) "
        f"for svzerodcalibrator: {block_fix}"
    )

    for junc in junctions:
        for key in ("inlet_blocks", "outlet_blocks"):
            blocks = junc.get(key)
            if not isinstance(blocks, list):
                continue
            for i, b in enumerate(blocks):
                if isinstance(b, str) and b in block_fix:
                    blocks[i] = block_fix[b]

    def _remap_three_part_key(k):
        if not isinstance(k, str) or k.count(":") != 2:
            return k
        kind, a, b = k.split(":", 2)
        a2 = block_fix.get(a, a)
        b2 = block_fix.get(b, b)
        if a2 != a or b2 != b:
            return f"{kind}:{a2}:{b2}"
        return k

    for obs_key in ("y", "dy"):
        d = config.get(obs_key)
        if not isinstance(d, dict):
            continue
        config[obs_key] = {_remap_three_part_key(k): v for k, v in d.items()}

    if isinstance(full, dict):
        for obs_key in ("y", "dy"):
            d = full.get(obs_key)
            if not isinstance(d, dict):
                continue
            full[obs_key] = {_remap_three_part_key(k): v for k, v in d.items()}


def run_calibration(calibration_input_path, output_path):
    """
    Run svZeroDCalibrator to generate calibrated input file.
    Uses svzerodcalibrator executable at /Users/natalia/cursor_access/svZeroDPlus/Release/svzerodcalibrator.
    Ensures the calibrated output preserves the inflow BC from the calibration input (3D observations).
    
    Args:
        calibration_input_path: Path to calibration input JSON
        output_path: Path to save calibrated output JSON
    """
    import subprocess
    
    print(f"Running calibration...")
    
    # Read calibration input
    with open(calibration_input_path, 'r') as f:
        config = json.load(f)

    _sanitize_svzerod_calibration_topology(config)
    _normalize_el_stale_vessel_blocks_for_calibrator(config)

    # Use svzerodcalibrator executable
    calibrator_exe = '/Users/natalia/cursor_access/svZeroDPlus/Release/svzerodcalibrator'
    
    # Get absolute paths
    abs_input_path = os.path.abspath(calibration_input_path)
    abs_output_path = os.path.abspath(output_path)
    
    print(f"  Attempting calibration with svzerodcalibrator executable...")
    print(f"    Executable: {calibrator_exe}")
    print(f"    Input: {abs_input_path}")
    print(f"    Output: {abs_output_path}")
    
    tmp_input = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(config, tf, indent=4)
            tmp_input = tf.name
        # Run svzerodcalibrator: svzerodcalibrator <input.json> <output.json>
        result = subprocess.run(
            [calibrator_exe, tmp_input, abs_output_path],
            capture_output=False,
            text=True,
            check=True
        )
        #import pdb; pdb.set_trace()
        print(f"  ✓ Calibration completed with svzerodcalibrator")
        if result.stdout:
            print(f"  STDOUT: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"svzerodcalibrator failed with return code {e.returncode}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError(f"svzerodcalibrator executable not found at: {calibrator_exe}")
    finally:
        if tmp_input:
            try:
                os.unlink(tmp_input)
            except OSError:
                pass
    
    # Read the calibrated output
    try:
        with open(abs_output_path, 'r') as f:
            cali = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read calibrated output from {abs_output_path}: {e}")
    
    # Post-process calibrated output to ensure compatibility with svzerodsolver
    for junc in cali.get('junctions', []):
        if 'junction_values' in junc:
            # Remove C parameter if present (not supported by svzerodsolver)
            if 'C' in junc['junction_values']:
                del junc['junction_values']['C']
            
            # For HybridJunction, ensure pressure_recovery_coefficient is present
            if junc.get('junction_type') == 'HybridJunction':
                if 'pressure_recovery_coefficient' not in junc['junction_values']:
                    # Add with default zeros matching number of outlets
                    num_outlets = junction_outlet_count(junc)
                    junc['junction_values']['pressure_recovery_coefficient'] = [0.0] * num_outlets
                    print(f"  Added pressure_recovery_coefficient to {junc.get('junction_name', 'unknown')} (HybridJunction)")
            else:
                # For non-HybridJunction types, remove pressure_recovery_coefficient if present
                if 'pressure_recovery_coefficient' in junc['junction_values']:
                    del junc['junction_values']['pressure_recovery_coefficient']
    # Replace simulation parameters with the original values

    # Write calibrated output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cali, f, indent=4)
    
    print(f"Calibrated output saved to: {output_path}")
    return cali
