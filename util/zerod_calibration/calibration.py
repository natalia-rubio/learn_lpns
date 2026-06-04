import os
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


CALIBRATION_RESIDUALS_SUBDIR = os.path.join("results", "calibration_residuals")


def find_repo_root(start_path: str) -> str:
    """Walk up from a file or directory path to the repo root (``util/`` + ``data/zeroD/``)."""
    cur = os.path.abspath(start_path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    for _ in range(16):
        if os.path.isdir(os.path.join(cur, "util")) and os.path.isdir(
            os.path.join(cur, "data", "zeroD")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_zero_d_cohort_dir(cohort_dir: str):
    """
    Parse ``.../data/zeroD/<set_name>/[<run_config>/]`` (directory of case subfolders).

    Returns:
        ``(repo_root, set_name, run_config_suffix)``; missing parts are ``None``.
    """
    norm = os.path.normpath(os.path.abspath(cohort_dir))
    parts = norm.split(os.sep)
    if "zeroD" not in parts:
        return find_repo_root(cohort_dir), None, None
    i = parts.index("zeroD")
    repo_root = norm[: norm.index(os.path.join("data", "zeroD"))]
    if not repo_root:
        repo_root = os.sep.join(parts[:i]) or os.sep
    tail = parts[i + 1 :]
    if not tail:
        return repo_root, None, None
    set_name = tail[0]
    run_config_suffix = tail[1] if len(tail) > 1 else None
    return repo_root, set_name, run_config_suffix


def infer_zero_d_layout_from_path(path: str):
    """
    Parse ``.../data/zeroD/<set_name>/[<run_config>/]<geo_name>/...``.

    Returns:
        ``(set_name, geo_name, run_config_suffix)`` or ``None`` if not under ``data/zeroD``.
    """
    norm = os.path.normpath(os.path.abspath(path))
    parts = norm.split(os.sep)
    if "zeroD" not in parts:
        return None
    i = parts.index("zeroD")
    tail = parts[i + 1 :]
    if len(tail) < 2:
        return None
    set_name = tail[0]
    if len(tail) == 2:
        return set_name, tail[1], None
    geo_name = tail[-2]
    if len(tail) == 3:
        return set_name, geo_name, None
    if len(tail) == 4:
        return set_name, geo_name, tail[1]
    run_config = os.path.join(*tail[1:-2])
    return set_name, geo_name, run_config


def calibration_residual_csv_basename(calibration_input_path: str) -> str:
    """
    Filename (no directory) for the stacked residual CSV consumed by svZeroDCalibrator.

    Stored in ``calibration_parameters.residual_csv`` as a basename; ``run_calibration`` resolves it
    to an absolute path under ``results/calibration_residuals/``.

    Args:
        calibration_input_path: Path to the calibration *input* JSON whose stem drives the name.

    Returns:
        e.g. ``bifurcations_EL_calibration_residual_BloodVesselJunction.csv`` for input
        ``*_calibration_input_BloodVesselJunction.json``.
    """
    stem = os.path.splitext(os.path.basename(str(calibration_input_path)))[0]
    if "_calibration_input_" in stem:
        return stem.replace("_calibration_input_", "_calibration_residual_", 1) + ".csv"
    if stem.endswith("_calibration_input"):
        return stem[: -len("_calibration_input")] + "_calibration_residual.csv"
    if stem == "calibration_input":
        return "calibration_residual.csv"
    return stem + "_calibration_residual.csv"


def calibration_residual_csv_dir(
    repo_root: str,
    set_name: str,
    geo_name: str,
    run_config_suffix=None,
) -> str:
    """Directory for one geometry's calibration residual CSVs (created by ``run_calibration``)."""
    parts = [repo_root, CALIBRATION_RESIDUALS_SUBDIR, set_name]
    if run_config_suffix:
        parts.append(str(run_config_suffix))
    parts.append(str(geo_name))
    return os.path.join(*parts)


def calibration_residual_csv_path(
    calibration_input_path: str,
    repo_root: str,
    set_name: str,
    geo_name: str,
    run_config_suffix=None,
) -> str:
    """Absolute path where svZeroDCalibrator should write the stacked residual CSV."""
    out_dir = calibration_residual_csv_dir(repo_root, set_name, geo_name, run_config_suffix)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, calibration_residual_csv_basename(calibration_input_path))


def resolve_calibration_residual_csv_path(
    calibration_input_path: str,
    output_path: str,
    repo_root=None,
    set_name=None,
    geo_name=None,
    run_config_suffix=None,
) -> str:
    """
    Choose residual CSV path: ``results/calibration_residuals/...`` when layout is known,
    else same directory as ``output_path`` (legacy).
    """
    if repo_root and set_name and geo_name:
        return calibration_residual_csv_path(
            calibration_input_path,
            repo_root,
            set_name,
            geo_name,
            run_config_suffix,
        )
    layout = infer_zero_d_layout_from_path(output_path) or infer_zero_d_layout_from_path(
        calibration_input_path
    )
    if layout:
        s, g, rc = layout
        root = repo_root or find_repo_root(output_path)
        return calibration_residual_csv_path(calibration_input_path, root, s, g, rc)
    bn = calibration_residual_csv_basename(calibration_input_path)
    out_dir = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
    return os.path.abspath(os.path.join(out_dir, bn))


def create_calibration_input(geometric_input_path, observations, output_path, observations_full=None, centerline_soln_path=None, geo_dir=None, stenosis_off=False, penalty_off=False, set_name=None, stacked_residual_csv=True):
    """
    Create calibration input file from geometric input and observations.
    BC times use ``len(inflow)`` samples spaced by ``time_step_size`` when the observed
    inflow series has more than one sample. If the observed inflow has at most one sample
    (e.g. steady 1D), INFLOW ``(t, Q)`` is taken from the geometric JSON instead.
    Prefer ``simulation_parameters.time_step_size`` from the geometric JSON (Richter / svZeroD);
    otherwise derive from the 1D VTP and ``geo_dir`` (XML or VMR/TST path fallbacks).
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        observations: Calibration observations dictionary (y, dy), potentially NaN-masked
        output_path: Path to save calibration input JSON
        observations_full: Optional full observations dictionary (y, dy). If None, uses observations.
        centerline_soln_path: Path to 1D centerline solution VTP (to extract timestep count)
        geo_dir: Geometry directory (to find XML file for timestep size)
        stenosis_off: If True, set calibrate_stenosis_coefficient False and set all stenosis to 0
        penalty_off: If True (and stenosis_off is False), set L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient to 0. Incompatible with stenosis_off.
        set_name: Optional set name (e.g. VMR_abdo) used to look up set-specific L2 penalties from SET_L2_PENALTIES; unlisted sets use defaults.
        stacked_residual_csv: If True (default), set ``calibration_parameters.residual_csv`` to a basename
            derived from ``output_path``; ``run_calibration`` resolves it under ``results/calibration_residuals/``.
            If False, only a ``residual_csv`` already present on the geometric input is preserved.
    """
    if stenosis_off and penalty_off:
        raise ValueError("Cannot use both --stenosis-off and --penalty-off.")
    if observations_full is None:
        observations_full = observations
    print(f"Reading geometric input from: {geometric_input_path}")
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Compute BC times directly from 1D solution timesteps (no refinement/interpolation)
    bc_time = None

    geo_bc_t, geo_bc_q = _geometric_inflow_tq_lists(inp)
    inflow_key = _flow_inflow_observation_key(observations_full)

    obs_bc_flow_list = None
    obs_bc_len = 0
    if inflow_key:
        obs_bc_flow_list = observations_full["y"][inflow_key]
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
    prev_cal = inp.get("calibration_parameters") or {}
    prev_residual_csv = prev_cal.get("residual_csv")
    inp["calibration_parameters"] = {
        "tolerance_gradient": 1e-5,
        "tolerance_increment": 1e-10,
        "maximum_iterations": 2000,
        "calibrate_stenosis_coefficient": not stenosis_off,
        "calibrate_capacitance": False,
        "set_capacitance_to_zero": False,
        "L2_penalty_R_poiseuille": l2_R,
        "L2_penalty_stenosis_coefficient": l2_stenosis,
        "L2_penalty_L": 0
    }
    if stacked_residual_csv:
        rc = calibration_residual_csv_basename(output_path)
        inp["calibration_parameters"]["residual_csv"] = rc
        print(f"  Stacked residual CSV (relative path for calibrator): {rc}")
    elif prev_residual_csv:
        inp["calibration_parameters"]["residual_csv"] = prev_residual_csv
    
    inp["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = len(bc_time)
    inp["simulation_parameters"]["output_all_cycles"] = True
    inp["simulation_parameters"]["steady_initial"] = False
    inp["simulation_parameters"]["absolute_tolerance"] = 1e-5
    inp["simulation_parameters"]["maximum_nonlinear_iterations"] = 50
    inp["simulation_parameters"]["num_cardiac_cycles"] = 1
    
    # Convert numpy arrays in observations to lists for JSON serialization
    observations_list = convert_numpy_to_list(observations)
    observations_full_list = convert_numpy_to_list(observations_full)
    
    # Add observations
    inp.update(observations_list)
    
    # Store full observations for plotting (3D solution should show full time series)
    inp['_full_observations'] = observations_full_list
    
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


def _materialize_nan_masked_observations_for_calibrator(config):
    """
    Convert NaN-masked observation rows into finite sliced rows for calibrator input.

    The stock nlohmann::json parser in svzerodcalibrator rejects NaN tokens in JSON.
    """
    y = config.get("y")
    if not isinstance(y, dict) or not y:
        return config

    dy = config.get("dy")
    dy = dy if isinstance(dy, dict) else {}

    first_key = next(iter(y))
    ref = np.asarray(y[first_key], dtype=float)
    if ref.ndim != 1:
        return config
    T = ref.shape[0]
    if T == 0:
        return config

    keep_mask = np.ones(T, dtype=bool)
    has_nan = False

    for bucket in (y, dy):
        for key, series in bucket.items():
            arr = np.asarray(series, dtype=float)
            if arr.ndim != 1 or arr.shape[0] != T:
                continue
            finite = np.isfinite(arr)
            if not np.all(finite):
                has_nan = True
            keep_mask &= finite

    if not has_nan:
        return config

    idx = np.flatnonzero(keep_mask)
    n_keep = int(idx.shape[0])
    if n_keep == 0:
        raise RuntimeError("All calibration observation rows are NaN/non-finite; cannot run calibrator.")

    out = dict(config)
    y_out = {}
    for key, series in y.items():
        arr = np.asarray(series, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == T:
            y_out[key] = arr[idx].tolist()
        else:
            y_out[key] = series
    out["y"] = y_out

    if dy:
        dy_out = {}
        for key, series in dy.items():
            arr = np.asarray(series, dtype=float)
            if arr.ndim == 1 and arr.shape[0] == T:
                dy_out[key] = arr[idx].tolist()
            else:
                dy_out[key] = series
        out["dy"] = dy_out

    for bc in out.get("boundary_conditions", []):
        if bc.get("bc_name") != "INFLOW":
            continue
        bc_values = bc.get("bc_values") or {}
        t = bc_values.get("t")
        q = bc_values.get("Q")
        if isinstance(t, list) and isinstance(q, list) and len(t) == T and len(q) == T:
            bc_values["t"] = np.asarray(t, dtype=float)[idx].tolist()
            bc_values["Q"] = np.asarray(q, dtype=float)[idx].tolist()
            bc["bc_values"] = bc_values
        break

    sp = out.get("simulation_parameters")
    if isinstance(sp, dict):
        sp = dict(sp)
        sp["number_of_time_pts_per_cardiac_cycle"] = n_keep
        out["simulation_parameters"] = sp

    print(f"  Materialized NaN-masked observations for calibrator: {n_keep}/{T} timestep(s) kept.")
    return out


def run_calibration(
    calibration_input_path,
    output_path,
    repo_root=None,
    set_name=None,
    geo_name=None,
    run_config_suffix=None,
):
    """
    Run svZeroDCalibrator to generate calibrated input file.
    Uses svzerodcalibrator executable at /Users/natalia/cursor_access/svZeroDPlus/Release/svzerodcalibrator.
    Ensures the calibrated output preserves the inflow BC from the calibration input (3D observations).
    
    Args:
        calibration_input_path: Path to calibration input JSON
        output_path: Path to save calibrated output JSON
        repo_root: Repo root for ``results/calibration_residuals/`` (inferred if omitted)
        set_name: Dataset name (e.g. VMR_abdo); inferred from paths when omitted
        geo_name: Geometry case folder name; inferred from paths when omitted
        run_config_suffix: Optional run-config subfolder under ``data/zeroD/<set>/``
    """
    import subprocess
    
    print(f"Running calibration...")
    
    # Read calibration input
    with open(calibration_input_path, 'r') as f:
        config = json.load(f)

    _sanitize_svzerod_calibration_topology(config)
    config = _materialize_nan_masked_observations_for_calibrator(config)

    # Use svzerodcalibrator executable
    calibrator_exe = '/Users/natalia/cursor_access/svZeroDPlus/Release/svzerodcalibrator'
    
    # Get absolute paths
    abs_input_path = os.path.abspath(calibration_input_path)
    abs_output_path = os.path.abspath(output_path)

    # svZeroDCalibrator needs an absolute residual_csv path; store under results/calibration_residuals/.
    cp = config.get("calibration_parameters")
    if isinstance(cp, dict):
        rc = cp.get("residual_csv")
        if isinstance(rc, str) and rc.strip():
            abs_rc = resolve_calibration_residual_csv_path(
                calibration_input_path,
                output_path,
                repo_root=repo_root,
                set_name=set_name,
                geo_name=geo_name,
                run_config_suffix=run_config_suffix,
            )
            os.makedirs(os.path.dirname(abs_rc), exist_ok=True)
            cp = dict(cp)
            cp["residual_csv"] = abs_rc
            config["calibration_parameters"] = cp
            print(f"  Resolved residual_csv to {abs_rc}")

    print(f"  Attempting calibration with svzerodcalibrator executable...")
    print(f"    Executable: {calibrator_exe}")
    print(f"    Input: {abs_input_path}")
    print(f"    Output: {abs_output_path}")
    
    tmp_input = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(config, tf, indent=4, allow_nan=False)
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
