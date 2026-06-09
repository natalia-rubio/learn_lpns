import json
import os

import numpy as np
from scipy.interpolate import interp1d


def update_geometric_input_with_calibration_bc(geometric_input_path, calibration_input_path):
    """
    Update geometric_input.json with inflow BC from calibration_input.json.
    Uses the full time frame (not the second half used for calibration).

    Args:
        geometric_input_path: Path to geometric input JSON
        calibration_input_path: Path to calibration input JSON
    """
    print("\nUpdating geometric input with inflow BC from calibration input...")

    if not os.path.exists(calibration_input_path):
        print(f"  Warning: Calibration input not found at {calibration_input_path}")
        return False

    # Read calibration input
    with open(calibration_input_path, "r") as f:
        calib_data = json.load(f)

    # Use full BC if available (stored for forward simulations), otherwise use calibration BC
    if "_full_bc_for_forward_sim" in calib_data:
        # Use full time frame for forward simulations
        calib_inflow_bc = calib_data["_full_bc_for_forward_sim"].copy()
        print("  Using full time frame from calibration input (for forward simulations)")
    else:
        # Fallback: extract from boundary conditions (this would be second half)
        calib_inflow_bc = None
        for bc in calib_data.get("boundary_conditions", []):
            if bc.get("bc_name") == "INFLOW":
                calib_inflow_bc = bc.get("bc_values", {})
                break
        if not calib_inflow_bc:
            print("  Warning: Could not find INFLOW BC in calibration input")
            return False
        print("  Warning: Full BC not found, using calibration BC (may be second half)")

    # Read geometric input
    with open(geometric_input_path, "r") as f:
        geo_input = json.load(f)

    # Refine inlet BC for forward simulation (halve timestep size, interpolate flow)
    refined_bc = calib_inflow_bc  #  refine_inlet_bc_for_forward_simulation(calib_inflow_bc)

    # Update inflow BC
    geo_updated = False
    n_pts_inflow = len(refined_bc.get("t", []))
    for bc in geo_input.get("boundary_conditions", []):
        if bc.get("bc_name") == "INFLOW":
            bc["bc_values"] = refined_bc.copy()
            geo_updated = True
            print("  Updated geometric input INFLOW BC (refined for forward simulation):")
            print(f"    Original number of time points: {len(calib_inflow_bc.get('t', []))}")
            print(f"    Refined number of time points: {n_pts_inflow}")
            if refined_bc.get("t"):
                print(f"    Time range: [{refined_bc['t'][0]:.6f}, {refined_bc['t'][-1]:.6f}]")
            if refined_bc.get("Q"):
                print(f"    Flow range: [{min(refined_bc['Q']):.3f}, {max(refined_bc['Q']):.3f}]")
            break

    # Ensure simulation_parameters.number_of_time_pts_per_cardiac_cycle
    # matches the length of the refined inflow BC time series
    if geo_updated and "simulation_parameters" in geo_input:
        geo_input["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = n_pts_inflow

        print(f"  Updated simulation_parameters.number_of_time_pts_per_cardiac_cycle to {n_pts_inflow}")

        # Update cardiac_cycle_period to match the BC time array
        if refined_bc.get("t") and len(refined_bc["t"]) > 1:
            t_array = refined_bc["t"]
            if t_array[0] == 0.0:
                cardiac_period = t_array[-1]
            else:
                dt = t_array[1] - t_array[0] if len(t_array) > 1 else 0.0
                cardiac_period = t_array[-1] - t_array[0] + dt
            geo_input["simulation_parameters"]["cardiac_cycle_period"] = cardiac_period
            print(
                f"  Updated simulation_parameters.cardiac_cycle_period to {cardiac_period:.6f} s (from BC time array)"
            )

    if geo_updated:
        # Write updated geometric input
        with open(geometric_input_path, "w") as f:
            json.dump(geo_input, f, indent=4)
        print(f"  Saved updated geometric input to: {geometric_input_path}")
        return True
    else:
        print("  Warning: Could not find INFLOW BC in geometric input to update")
        return False


def update_outlet_bcs_in_file(file_path, outlet_params, file_type="calibration input"):
    """
    Update outlet boundary conditions (R and Pd) in a JSON file using fitted parameters.

    Args:
        file_path: Path to JSON file to update (calibration input or calibrated output)
        outlet_params: Dictionary mapping BC names to (R, Pd) tuples
        file_type: String describing file type (for logging)

    Returns:
        True if file was updated, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"  Warning: {file_type} file not found: {file_path}")
        return False

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        updated_count = 0
        # Find and update outlet BCs
        for bc in data.get("boundary_conditions", []):
            if bc.get("bc_type") == "RESISTANCE":
                bc_name = bc.get("bc_name")
                if bc_name in outlet_params:
                    resistance, pd = outlet_params[bc_name]
                    old_r = bc["bc_values"].get("R", 1.0)
                    old_pd = bc["bc_values"].get("Pd", 0.0)
                    bc["bc_values"]["R"] = resistance
                    bc["bc_values"]["Pd"] = pd
                    updated_count += 1
                    print(f"    {bc_name}: R {old_r:.4f} -> {resistance:.4f}, Pd {old_pd:.4f} -> {pd:.4f}")

        if updated_count > 0:
            # Write updated file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  ✓ Updated {updated_count} outlet BC(s) in {file_type}: {file_path}")
            return True
        else:
            print(f"  No outlet BCs found to update in {file_type}: {file_path}")
            return False

    except Exception as e:
        print(f"  Warning: Could not update {file_type} file {file_path}: {e}")
        return False


def sync_nn_config_bcs_from_calibration(nn_config_paths, source_bc_path, verbose=True):
    """
    Update boundary conditions in NN config JSONs to match the calibrated output
    (or calibration input) so that RCR and other outlet BCs are consistent.

    Args:
        nn_config_paths: List of paths to NN config JSON files to update
            (e.g. *_NN_JunctionOnly.json, *_NN_JunctionAndVessel.json, *_NN_VesselOnly.json)
        source_bc_path: Path to the JSON file to copy boundary_conditions from
            (typically the BloodVesselJunction calibrated output or its calibration input)
        verbose: If True, print which files were updated

    Returns:
        Number of NN config files that were updated (0 if source missing or no paths)
    """
    import copy

    if not nn_config_paths:
        return 0
    if not os.path.exists(source_bc_path):
        if verbose:
            print(f"  ⊘ Skipping NN BC sync: source not found: {source_bc_path}")
        return 0
    try:
        with open(source_bc_path, "r") as f:
            source_config = json.load(f)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load source for NN BC sync from {source_bc_path}: {e}")
        return 0
    bcs = source_config.get("boundary_conditions", [])
    if not bcs:
        if verbose:
            print(f"  Warning: No boundary_conditions in source {source_bc_path}, skipping NN BC sync")
        return 0
    updated = 0
    for path in nn_config_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                nn_config = json.load(f)
            nn_config["boundary_conditions"] = copy.deepcopy(bcs)
            with open(path, "w") as f:
                json.dump(nn_config, f, indent=4)
            updated += 1
            if verbose:
                print(f"  ✓ Synced BCs from calibration into {os.path.basename(path)}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not update NN config {path}: {e}")
    return updated


def refine_inlet_bc_for_forward_simulation(output_path, max_reasonable_points=10000, calibration_input_path=None):
    """
    Refine the inlet boundary condition for forward simulation.

    Reads the BC from the calibration input (to avoid multiple refinements) and writes
    the refined BC to the output file.

    Args:
        output_path: Path to output JSON file (calibrated output or geometric input)
        refinement_factor: Factor to multiply number of time points (default: 2)
        max_reasonable_points: Maximum reasonable number of time points (default: 10000)
                               If original points exceed this, raises ValueError
        calibration_input_path: Path to calibration input file (if None, derives from output_path)
    """
    import os

    # If calibration_input_path not provided, try to derive it from output_path
    if calibration_input_path is None:
        # Try to find corresponding calibration input
        # Pattern: *_calibrated_output_*.json -> *_calibration_input_*.json
        if "calibrated_output" in output_path:
            calibration_input_path = output_path.replace("calibrated_output", "calibration_input")
        # For geometric input or NN files, use the file itself as source (first time only)
        elif "geometric_input" in output_path or "NN_JunctionOnly" in output_path:
            calibration_input_path = output_path
        else:
            # Default: assume output_path is the source
            calibration_input_path = output_path

    # Read BC from calibration input (original, unrefined)
    if not os.path.exists(calibration_input_path):
        raise FileNotFoundError(
            f"Calibration input file not found: {calibration_input_path}. Cannot refine BC without original source."
        )

    with open(calibration_input_path, "r") as f:
        calib_input = json.load(f)

    # Get BC from calibration input
    bc_time = calib_input["boundary_conditions"][0]["bc_values"]["t"]
    bc_flow = calib_input["boundary_conditions"][0]["bc_values"]["Q"]

    original_n_pts = len(bc_time)

    # Check if the original number of points is unreasonably high
    if original_n_pts > max_reasonable_points:
        raise ValueError(
            f"Unreasonably high number of time points in boundary condition: {original_n_pts}. "
            f"Expected at most {max_reasonable_points} points per cardiac cycle. "
            f"This suggests an error in the 1D solution data or how it was processed. "
            f"Please check the 1D solution file and the observation extraction process."
        )

    # add a key to the output data with the refinement factor

    if len(bc_flow) > 1000:
        refinement_factor = 1
    else:
        refinement_factor = int(np.ceil(1000 / len(bc_flow)))

    refined_n_pts = original_n_pts * refinement_factor

    bc_time_refined = np.linspace(0, bc_time[-1], refined_n_pts, endpoint=True).tolist()
    bc_flow_refined = interp1d(bc_time, bc_flow, kind="cubic")(bc_time_refined)

    # Read output file to update it
    with open(output_path, "r") as f:
        output_data = json.load(f)

    # Update BC in output file
    output_data["boundary_conditions"][0]["bc_values"]["t"] = list(bc_time_refined)
    output_data["boundary_conditions"][0]["bc_values"]["Q"] = list(bc_flow_refined)
    output_data["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = len(bc_time_refined)
    output_data["simulation_parameters"]["number_of_cardiac_cycles"] = 10
    output_data["simulation_parameters"]["output_all_cycles"] = False

    print(f"Refinement factor: {refinement_factor}")
    # add a key to the output data with the refinement factor
    output_data["refinement_factor"] = refinement_factor
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(
        f"Refined inlet boundary condition for forward simulation to: {output_path} "
        f"(read from: {calibration_input_path})"
    )
    return
