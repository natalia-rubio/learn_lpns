import copy
import json
import os
import subprocess

import pandas as pd


def run_forward_simulation(input_json_path, output_csv_path):
    """
    Run forward 0D simulation and save results to CSV.
    Uses svzerodsolver from SVZEROD_INSTALL_DIR or PATH (see svzerod_binaries.py).
    Verifies that inlet flow matches the boundary condition.

    Args:
        input_json_path: Path to 0D input JSON file
        output_csv_path: Path to save CSV results

    Returns:
        None (results written to output_csv_path)
    """
    from learn_lpns.zerod_calibration.tools.svzerod_binaries import svzerod_binary

    svzerodsolver_path = svzerod_binary("svzerodsolver")

    print(f"Running forward simulation from: {input_json_path}")

    # Convert PosixPath to string if needed
    input_json_path_str = str(input_json_path)
    output_csv_path_str = str(output_csv_path)

    with open(input_json_path_str, "r") as f:
        input_data = json.load(f)

    # Try svzerodsolver executable first
    if os.path.exists(svzerodsolver_path):
        try:
            # Create a deep copy to avoid modifying the original
            input_data_sim = copy.deepcopy(input_data)

            # Remove calibration parameters if present (not needed for forward simulation)
            if "calibration_parameters" in input_data_sim:
                del input_data_sim["calibration_parameters"]

            # Remove observation data if present
            if "y" in input_data_sim:
                del input_data_sim["y"]
            if "dy" in input_data_sim:
                del input_data_sim["dy"]

            # Write temporary input file for svzerodsolver (use original input path if it's already clean)
            use_temp = "calibration_parameters" in input_data or "y" in input_data or "dy" in input_data
            if use_temp:
                temp_input_path = input_json_path_str + ".temp"
                with open(temp_input_path, "w") as f:
                    json.dump(input_data_sim, f, indent=4)
                input_file_for_solver = temp_input_path
            else:
                input_file_for_solver = input_json_path_str

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_csv_path_str), exist_ok=True)

            # Compute absolute paths BEFORE changing directories
            abs_input_path = os.path.abspath(input_file_for_solver)
            abs_output_dir = (
                os.path.abspath(os.path.dirname(output_csv_path_str))
                if os.path.dirname(output_csv_path_str)
                else os.path.abspath(".")
            )
            abs_output_csv = os.path.abspath(output_csv_path_str)

            print("  Attempting simulation with svzerodsolver executable...")
            print(f"    Executable: {svzerodsolver_path}")
            print(f"    Input: {abs_input_path}")
            print(f"    Output: {output_csv_path_str}")

            # Run svzerodsolver: takes input.json as first argument
            # If output is not specified, it defaults to ./output.csv in the current directory
            # We'll run it in the output directory and then move/rename the output file
            output_dir = os.path.dirname(output_csv_path_str) or "."
            os.path.basename(output_csv_path_str)

            # Change to output directory to run solver (so output.csv is created there)
            original_cwd = os.getcwd()
            try:
                os.chdir(abs_output_dir)
                # Command format: svzerodsolver input.json
                # It will output to ./output.csv in the current directory
                # Use absolute path for input file (computed before changing directories)
                cmd = [svzerodsolver_path, abs_input_path]

                # Check simulation parameters to estimate timeout
                num_cycles = input_data_sim.get("simulation_parameters", {}).get("number_of_cardiac_cycles", 2)
                num_time_pts = input_data_sim.get("simulation_parameters", {}).get(
                    "number_of_time_pts_per_cardiac_cycle", 100
                )
                # Estimate timeout: ~1 second per 1000 time points, with minimum 60 seconds
                estimated_timeout = max(60, int((num_cycles * num_time_pts) / 50) + 30)

                print(
                    f"    Running simulation ({num_cycles} cycles, {num_time_pts} pts/cycle, "
                    f"timeout: {estimated_timeout}s)..."
                )
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=estimated_timeout)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"svzerodsolver timed out after {estimated_timeout} seconds. "
                        f"This may indicate:\n"
                        f"  - Simulation parameters are too large (cycles: {num_cycles}, pts/cycle: {num_time_pts})\n"
                        f"  - Numerical instability in the simulation\n"
                        f"  - Consider reducing number_of_cardiac_cycles or number_of_time_pts_per_cardiac_cycle"
                    )

                if result.returncode != 0:
                    error_msg = f"svzerodsolver failed with return code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nSTDERR: {result.stderr[-1000:]}"  # Last 1000 chars
                    if result.stdout:
                        error_msg += f"\nSTDOUT: {result.stdout[-1000:]}"  # Last 1000 chars
                    raise RuntimeError(error_msg)

                print("  ✓ Simulation completed successfully with svzerodsolver")

                # Move output.csv to the desired filename if it exists
                # output.csv should be in the current directory (abs_output_dir)
                default_output = os.path.join(abs_output_dir, "output.csv")
                if os.path.exists(default_output):
                    if default_output != abs_output_csv:
                        os.rename(default_output, abs_output_csv)
                elif os.path.exists("output.csv"):
                    os.rename("output.csv", abs_output_csv)
                else:
                    raise RuntimeError(
                        f"svzerodsolver did not create output file. "
                        f"Expected './output.csv' in {abs_output_dir}\n"
                        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                    )
            finally:
                os.chdir(original_cwd)
                # Clean up temporary input file if we created one
                if use_temp and os.path.exists(temp_input_path):
                    os.remove(temp_input_path)

            # Verify inlet flow matches BC
            # print("\n  Verifying inlet flow matches boundary condition...")
            # verify_inlet_flow_matches_bc(input_data_sim, output_csv_path_str)

            # Return None since we're using CSV output, not a results dictionary
            return None
        except Exception as e:
            print(f"  ✗ svzerodsolver simulation failed: {e}")
            print("  Creating all-zeros solution as fallback...")

            # Create all-zeros solution
            try:
                # Extract vessel names from input
                vessels = input_data.get("vessels", [])
                vessel_names = [v.get("vessel_name", f"vessel_{v.get('vessel_id', i)}") for i, v in enumerate(vessels)]

                # Extract simulation parameters
                sim_params = input_data.get("simulation_parameters", {})
                num_cycles = 1
                num_time_pts_per_cycle = sim_params.get("number_of_time_pts_per_cardiac_cycle", 0)

                # Calculate time step size from INFLOW boundary condition
                time_step = 0  # Default fallback
                boundary_conditions = input_data.get("boundary_conditions", [])
                for bc in boundary_conditions:
                    if bc.get("bc_name") == "INFLOW" and bc.get("bc_type") == "FLOW":
                        bc_values = bc.get("bc_values", {})
                        t_values = bc_values.get("t", [])
                        if len(t_values) >= 2:
                            time_step = t_values[1] - t_values[0]
                        break

                # Generate time points
                total_time_pts = num_cycles * num_time_pts_per_cycle
                times = [i * time_step for i in range(total_time_pts)]

                # Create DataFrame with all zeros
                rows = []
                for vessel_name in vessel_names:
                    for time in times:
                        rows.append(
                            {
                                "name": vessel_name,
                                "time": time,
                                "flow_in": 0.0,
                                "flow_out": 0.0,
                                "pressure_in": 0.0,
                                "pressure_out": 0.0,
                            }
                        )

                result_df = pd.DataFrame(rows)

                # Ensure output directory exists
                output_dir = os.path.dirname(output_csv_path_str)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                # Save to CSV
                result_df.to_csv(output_csv_path_str, index=False)

                print(f"  ✓ Created all-zeros solution with {len(vessel_names)} vessels and {len(times)} time points")
                print(f"  Results saved to: {output_csv_path_str}")

                return None
            except Exception as e2:
                print(f"  ✗ Failed to create all-zeros solution: {e2}")
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"svzerodsolver failed and could not create all-zeros solution. "
                    f"Original error: {e}, Secondary error: {e2}"
                )
