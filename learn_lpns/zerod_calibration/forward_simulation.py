import copy
import json
import os
import shutil
import subprocess

import pandas as pd

from learn_lpns.config import get_pipeline_config
from learn_lpns.zerod_calibration.modality_paths import casadi_results_csv_path


def prepare_forward_simulation_input(input_data: dict) -> dict:
    """Return a deep copy of input JSON with calibration-only fields removed."""
    input_data_sim = copy.deepcopy(input_data)
    input_data_sim.pop("calibration_parameters", None)
    input_data_sim.pop("y", None)
    input_data_sim.pop("dy", None)
    return input_data_sim


def _write_temp_input_if_needed(input_json_path_str: str, input_data: dict, input_data_sim: dict) -> str:
    use_temp = "calibration_parameters" in input_data or "y" in input_data or "dy" in input_data
    if use_temp:
        temp_input_path = input_json_path_str + ".temp"
        with open(temp_input_path, "w") as f:
            json.dump(input_data_sim, f, indent=4)
        return temp_input_path
    return input_json_path_str


def _run_svzerod_forward_simulation(
    input_data: dict,
    input_data_sim: dict,
    input_json_path_str: str,
    output_csv_path_str: str,
) -> None:
    from learn_lpns.zerod_calibration.tools.svzerod_binaries import svzerod_binary

    svzerodsolver_path = svzerod_binary("svzerodsolver")

    if not os.path.exists(svzerodsolver_path):
        raise RuntimeError(f"svzerodsolver not found at: {svzerodsolver_path}")

    input_file_for_solver = _write_temp_input_if_needed(input_json_path_str, input_data, input_data_sim)

    os.makedirs(os.path.dirname(output_csv_path_str) or ".", exist_ok=True)

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

    num_cycles = input_data_sim.get("simulation_parameters", {}).get("number_of_cardiac_cycles", 2)
    num_time_pts = input_data_sim.get("simulation_parameters", {}).get("number_of_time_pts_per_cardiac_cycle", 100)
    estimated_timeout = max(60, int((num_cycles * num_time_pts) / 50) + 30)

    print(
        f"    Running simulation ({num_cycles} cycles, {num_time_pts} pts/cycle, "
        f"timeout: {estimated_timeout}s)..."
    )

    original_cwd = os.getcwd()
    temp_input_path = input_file_for_solver if input_file_for_solver.endswith(".temp") else None
    try:
        os.chdir(abs_output_dir)
        cmd = [svzerodsolver_path, abs_input_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=estimated_timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"svzerodsolver timed out after {estimated_timeout} seconds. "
                f"This may indicate:\n"
                f"  - Simulation parameters are too large (cycles: {num_cycles}, pts/cycle: {num_time_pts})\n"
                f"  - Numerical instability in the simulation\n"
                f"  - Consider reducing number_of_cardiac_cycles or number_of_time_pts_per_cardiac_cycle"
            ) from e

        if result.returncode != 0:
            error_msg = f"svzerodsolver failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr[-1000:]}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout[-1000:]}"
            raise RuntimeError(error_msg)

        print("  ✓ Simulation completed successfully with svzerodsolver")

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
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)


def _run_casadi_fallback(input_data_sim: dict, output_csv_path_str: str) -> None:
    from learn_lpns.zerod_calibration.casadi_forward_simulation import run_casadi_forward_simulation

    print("  Attempting CasADi fallback...")
    run_casadi_forward_simulation(
        input_data_sim,
        output_csv_path_str,
        solver_config=get_pipeline_config().solver,
    )
    marked_path = casadi_results_csv_path(output_csv_path_str)
    if marked_path != output_csv_path_str:
        shutil.copy2(output_csv_path_str, marked_path)
        print(f"  ✓ CasADi fallback completed; marked copy: {marked_path}")
    else:
        print("  ✓ CasADi fallback completed successfully")


def _create_all_zeros_fallback(input_data: dict, output_csv_path_str: str) -> None:
    vessels = input_data.get("vessels", [])
    vessel_names = [v.get("vessel_name", f"vessel_{v.get('vessel_id', i)}") for i, v in enumerate(vessels)]

    sim_params = input_data.get("simulation_parameters", {})
    num_cycles = 1
    num_time_pts_per_cycle = sim_params.get("number_of_time_pts_per_cardiac_cycle", 0)

    time_step = 0
    boundary_conditions = input_data.get("boundary_conditions", [])
    for bc in boundary_conditions:
        if bc.get("bc_name") == "INFLOW" and bc.get("bc_type") == "FLOW":
            bc_values = bc.get("bc_values", {})
            t_values = bc_values.get("t", [])
            if len(t_values) >= 2:
                time_step = t_values[1] - t_values[0]
            break

    total_time_pts = num_cycles * num_time_pts_per_cycle
    times = [i * time_step for i in range(total_time_pts)]

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
    output_dir = os.path.dirname(output_csv_path_str)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(output_csv_path_str, index=False)
    print(f"  ✓ Created all-zeros solution with {len(vessel_names)} vessels and {len(times)} time points")
    print(f"  Results saved to: {output_csv_path_str}")


def run_forward_simulation(input_json_path, output_csv_path):
    """
    Run forward 0D simulation and save results to CSV.

    Always tries ``svzerodsolver`` first. When ``solver.casadi_fallback`` is true
    (default), failed C++ runs fall back to the Python CasADi solver before the
    all-zeros last resort.

    Args:
        input_json_path: Path to 0D input JSON file
        output_csv_path: Path to save CSV results

    Returns:
        None (results written to output_csv_path)
    """
    input_json_path_str = str(input_json_path)
    output_csv_path_str = str(output_csv_path)
    casadi_fallback = get_pipeline_config().solver.casadi_fallback

    print(f"Running forward simulation from: {input_json_path}")

    with open(input_json_path_str) as f:
        input_data = json.load(f)

    input_data_sim = prepare_forward_simulation_input(input_data)

    try:
        _run_svzerod_forward_simulation(
            input_data, input_data_sim, input_json_path_str, output_csv_path_str
        )
        return None
    except Exception as e:
        print(f"  ✗ svzerodsolver simulation failed: {e}")

        if casadi_fallback:
            try:
                _run_casadi_fallback(input_data_sim, output_csv_path_str)
                return None
            except Exception as casadi_err:
                print(f"  ✗ CasADi fallback failed: {casadi_err}")

        print("  Creating all-zeros solution as fallback...")
        try:
            _create_all_zeros_fallback(input_data, output_csv_path_str)
            return None
        except Exception as e2:
            import traceback

            traceback.print_exc()
            raise RuntimeError(
                f"Forward simulation failed and could not create all-zeros solution. "
                f"svzerodsolver error: {e}, secondary error: {e2}"
            ) from e2
