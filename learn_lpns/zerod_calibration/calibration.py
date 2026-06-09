import json
import os

import numpy as np

from learn_lpns.config import (
    apply_solver_parameters,
    build_calibration_parameters,
    get_pipeline_config,
)
from learn_lpns.zerod_calibration.tools.file_io import convert_numpy_to_list, timestep_from_1D


def create_calibration_input(
    geometric_input_path,
    observations,
    output_path,
    centerline_soln_path=None,
    geo_dir=None,
    quadratic_resistor=False,
    penalty_on=False,
    set_name=None,
):
    """
    Create calibration input file from geometric input and observations.
    Computes BC times from 1D solution timesteps multiplied by timestep size from XML.

    Args:
        geometric_input_path: Path to geometric 0D input JSON
        observations: Dictionary with observation data (y, dy)
        output_path: Path to save calibration input JSON
        centerline_soln_path: Path to 1D centerline solution VTP (to extract timestep count)
        geo_dir: Geometry directory (to find XML file for timestep size)
        quadratic_resistor: If True, calibrate stenosis (quadratic resistor) coefficient
        penalty_on: If True (requires quadratic_resistor), use set-specific L2 penalties
            on R_poiseuille and stenosis_coefficient during calibration
        set_name: Optional set name (e.g. VMR_abdo) used to look up set-specific L2 penalties
    """
    print(f"Reading geometric input from: {geometric_input_path}")
    with open(geometric_input_path, "r") as f:
        inp = json.load(f)

    # Compute BC times directly from 1D solution timesteps (no refinement/interpolation)
    bc_time = None

    # If we are using a 1D solution, we compute the timestep using the 3D timestep size
    # from the XML file and the timestep_increment
    if centerline_soln_path and geo_dir:
        time_step_size = timestep_from_1D(centerline_soln_path, geo_dir)
    # If we are using a 0D solution, we use the timestep size from the geometric input
    # else:
    # print(
    #     f"  Warning: No centerline solution path or geo_dir provided, "
    #     f"using default time step size: {time_step_size:.6f} s"
    # )
    # bc_t = inp["boundary_conditions"][0]["bc_values"]["t"]
    # time_step_size = bc_t[1] - bc_t[0]

    # We get the inflow BC values from the observations
    if "flow:INFLOW:branch0_seg0" in observations["y"]:
        bc_flow = observations["y"]["flow:INFLOW:branch0_seg0"]
        # Convert to list if numpy array
        if isinstance(bc_flow, np.ndarray):
            bc_flow = bc_flow.tolist()
    else:
        raise ValueError("No inflow flow data found in observations")

    bc_time = np.linspace(0.0, len(bc_flow) * time_step_size, len(bc_flow), endpoint=False).tolist()

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

    if not quadratic_resistor:
        for v in inp.get("vessels", []):
            if "zero_d_element_values" in v and "stenosis_coefficient" in v["zero_d_element_values"]:
                v["zero_d_element_values"]["stenosis_coefficient"] = 0.0
        for j in inp.get("junctions", []):
            if "junction_values" in j and "stenosis_coefficient" in j["junction_values"]:
                sv = j["junction_values"]["stenosis_coefficient"]
                n_out = len(sv) if isinstance(sv, list) else 1
                j["junction_values"]["stenosis_coefficient"] = [0.0] * n_out
        print(
            "  Quadratic resistor off: all stenosis coefficients set to 0, "
            "calibrate_stenosis_coefficient=False, L2 penalties set to 0"
        )

    cfg = get_pipeline_config(set_name=set_name)
    if quadratic_resistor and penalty_on and set_name and set_name in cfg.calibration.set_l2_penalties:
        l2_r, l2_stenosis = cfg.calibration.l2_penalties_for_set(set_name)
        print(f"  Set-specific L2 penalties for {set_name}: R={l2_r}, stenosis={l2_stenosis}")
    elif quadratic_resistor and not penalty_on:
        print("  Penalty-on not enabled: L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient set to 0")

    inp["calibration_parameters"] = build_calibration_parameters(
        cfg.calibration,
        quadratic_resistor=quadratic_resistor,
        penalty_on=penalty_on,
        set_name=set_name,
    )

    inp["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = len(bc_time)
    inp["simulation_parameters"]["output_all_cycles"] = True
    apply_solver_parameters(inp["simulation_parameters"], cfg.solver)
    inp["simulation_parameters"]["num_cardiac_cycles"] = cfg.solver.number_of_cardiac_cycles

    # Convert numpy arrays in observations to lists for JSON serialization
    observations_list = convert_numpy_to_list(observations)

    # Add observations
    inp.update(observations_list)

    # Store full observations for plotting (3D solution should show full time series)
    inp["_full_observations"] = observations_list

    # Store original observed inflow BC for later use in calibrated output
    if "_full_bc_time" in inp and "_full_bc_flow" in inp:
        inp["observed_inflow_bc"] = {
            "t": inp["_full_bc_time"].copy(),
            "Q": inp["_full_bc_flow"].copy(),
        }
        # Remove temporary fields
        del inp["_full_bc_time"]
        del inp["_full_bc_flow"]

    # Write calibration input
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(inp, f, indent=4)

    print(f"Calibration input saved to: {output_path}")
    return inp


def run_calibration(calibration_input_path, output_path):
    """
    Run svZeroDCalibrator to generate calibrated input file.
    Uses svzerodcalibrator from SVZEROD_INSTALL_DIR or PATH (see svzerod_binaries.py).
    Ensures the calibrated output preserves the inflow BC from the calibration input (3D observations).

    Args:
        calibration_input_path: Path to calibration input JSON
        output_path: Path to save calibrated output JSON
    """
    import subprocess

    print("Running calibration...")

    # Read calibration input
    with open(calibration_input_path, "r") as f:
        json.load(f)

    from learn_lpns.zerod_calibration.tools.svzerod_binaries import svzerod_binary

    calibrator_exe = svzerod_binary("svzerodcalibrator")

    # Get absolute paths
    abs_input_path = os.path.abspath(calibration_input_path)
    abs_output_path = os.path.abspath(output_path)

    print("  Attempting calibration with svzerodcalibrator executable...")
    print(f"    Executable: {calibrator_exe}")
    print(f"    Input: {abs_input_path}")
    print(f"    Output: {abs_output_path}")

    try:
        # Run svzerodcalibrator: svzerodcalibrator <input.json> <output.json>
        result = subprocess.run(
            [calibrator_exe, abs_input_path, abs_output_path],
            capture_output=False,
            text=True,
            check=True,
        )
        # import pdb; pdb.set_trace()
        print("  ✓ Calibration completed with svzerodcalibrator")
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

    # Read the calibrated output
    try:
        with open(abs_output_path, "r") as f:
            cali = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read calibrated output from {abs_output_path}: {e}")

    # Post-process calibrated output to ensure compatibility with svzerodsolver
    for junc in cali.get("junctions", []):
        if "junction_values" in junc:
            # Remove C parameter if present (not supported by svzerodsolver)
            if "C" in junc["junction_values"]:
                del junc["junction_values"]["C"]

            # For HybridJunction, ensure pressure_recovery_coefficient is present
            if junc.get("junction_type") == "HybridJunction":
                if "pressure_recovery_coefficient" not in junc["junction_values"]:
                    # Add with default zeros matching number of outlets
                    num_outlets = len(junc.get("outlet_vessels", []))
                    junc["junction_values"]["pressure_recovery_coefficient"] = [0.0] * num_outlets
                    junc_name = junc.get("junction_name", "unknown")
                    print(f"  Added pressure_recovery_coefficient to {junc_name} (HybridJunction)")
            else:
                # For non-HybridJunction types, remove pressure_recovery_coefficient if present
                if "pressure_recovery_coefficient" in junc["junction_values"]:
                    del junc["junction_values"]["pressure_recovery_coefficient"]
    # Replace simulation parameters with the original values

    # Write calibrated output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cali, f, indent=4)

    print(f"Calibrated output saved to: {output_path}")
    return cali
