import os
import json
import numpy as np
from util.zerod_calibration.file_io import timestep_from_1D, convert_numpy_to_list


def create_calibration_input(geometric_input_path, observations, output_path, centerline_soln_path=None, geo_dir=None, stenosis_off=False, penalty_off=False):
    """
    Create calibration input file from geometric input and observations.
    Computes BC times from 1D solution timesteps multiplied by timestep size from XML.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        observations: Dictionary with observation data (y, dy)
        output_path: Path to save calibration input JSON
        centerline_soln_path: Path to 1D centerline solution VTP (to extract timestep count)
        geo_dir: Geometry directory (to find XML file for timestep size)
        stenosis_off: If True, set calibrate_stenosis_coefficient False and set all stenosis to 0
        penalty_off: If True (and stenosis_off is False), set L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient to 0. Incompatible with stenosis_off.
    """
    if stenosis_off and penalty_off:
        raise ValueError("Cannot use both --stenosis-off and --penalty-off.")
    print(f"Reading geometric input from: {geometric_input_path}")
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Compute BC times directly from 1D solution timesteps (no refinement/interpolation)
    bc_time = None
    
    # If we are using a 1D solution, we compute the timestep use the 3D timestep size from the XML file and the timestep_increment 
    if centerline_soln_path and geo_dir:
        time_step_size = timestep_from_1D(centerline_soln_path, geo_dir)
    # If we are using a 0D solution, we use the timestep size from the geometric input
    # else:
        # print(f"  Warning: No centerline solution path or geo_dir provided, using default time step size: {time_step_size:.6f} s")
        #time_step_size = inp["boundary_conditions"][0]["bc_values"]["t"][1] - inp["boundary_conditions"][0]["bc_values"]["t"][0]
    
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
    
    # Add calibration parameters (when stenosis_off or penalty_off, zero the R and stenosis L2 penalties)
    l2_R = 0.0 if (stenosis_off or penalty_off) else 10**5
    l2_stenosis = 0.0 if (stenosis_off or penalty_off) else 10**10
    if penalty_off and not stenosis_off:
        print("  Penalty-off: L2_penalty_R_poiseuille and L2_penalty_stenosis_coefficient set to 0")
    inp["calibration_parameters"] = {
        "tolerance_gradient": 1e-4,
        "tolerance_increment": 1e-4,
        "maximum_iterations": 100,
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
    
    
    # Use svzerodcalibrator executable
    calibrator_exe = '/Users/natalia/cursor_access/svZeroDPlus/Release/svzerodcalibrator'
    
    # Get absolute paths
    abs_input_path = os.path.abspath(calibration_input_path)
    abs_output_path = os.path.abspath(output_path)
    
    print(f"  Attempting calibration with svzerodcalibrator executable...")
    print(f"    Executable: {calibrator_exe}")
    print(f"    Input: {abs_input_path}")
    print(f"    Output: {abs_output_path}")
    
    try:
        # Run svzerodcalibrator: svzerodcalibrator <input.json> <output.json>
        result = subprocess.run(
            [calibrator_exe, abs_input_path, abs_output_path],
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
                    num_outlets = len(junc.get('outlet_vessels', []))
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
