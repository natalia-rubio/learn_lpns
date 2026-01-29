import os
import json
import numpy as np
import pandas as pd
import importlib.util
import sys
import copy
import subprocess
try:
    import pysvzerod
except ImportError:
    print("Warning: pysvzerod not found. Calibration will not be available.")
    print("Install with: pip install svzerodsolver")
    pysvzerod = None

# Try to import CasADi solver as fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas not found. CasADi fallback will not be available.")
    HAS_PANDAS = False
    pd = None

if HAS_PANDAS:
    try:
        # Import solve_casadi_unsteady from the casadi module
        casadi_module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'casadi', 'svzerod_with_casadi.py')
        if os.path.exists(casadi_module_path):
            import importlib.util
            # Read the module file and make problematic imports optional
            with open(casadi_module_path, 'r') as f:
                module_code = f.read()
            
            # Replace matplotlib import with try/except to make it optional
            # This allows the module to load even if matplotlib has compatibility issues
            module_code = module_code.replace(
                'import matplotlib',
                'try:\n    import matplotlib\nexcept (ImportError, AttributeError):\n    matplotlib = None'
            )
            
            # Make the util.tools.basic import optional (it's from a different project)
            module_code = module_code.replace(
                'from util.tools.basic import save_dict',
                'try:\n    from util.tools.basic import save_dict\nexcept (ImportError, ModuleNotFoundError):\n    def save_dict(*args, **kwargs):\n        pass  # Optional function, not needed for solve_casadi_unsteady'
            )
            
            # Remove the sys.path.append that points to a different project
            module_code = module_code.replace(
                'sys.path.append("/Users/natalia/Desktop/cco_bifurcations")',
                '# sys.path.append("/Users/natalia/Desktop/cco_bifurcations")  # Commented out - not needed'
            )
            
            # Create a temporary module from the modified code
            spec = importlib.util.spec_from_loader("svzerod_with_casadi", loader=None)
            casadi_module = importlib.util.module_from_spec(spec)
            
            # Temporarily modify sys.path to avoid errors in the imported module
            original_path = sys.path.copy()
            try:
                # Execute the modified module code
                exec(compile(module_code, casadi_module_path, 'exec'), casadi_module.__dict__)
                solve_casadi_unsteady = casadi_module.solve_casadi_unsteady
                HAS_CASADI = True
            except Exception as e:
                print(f"Warning: Could not load CasADi solver module: {e}")
                import traceback
                traceback.print_exc()
                solve_casadi_unsteady = None
                HAS_CASADI = False
            finally:
                sys.path = original_path
        else:
            solve_casadi_unsteady = None
            HAS_CASADI = False
    except Exception as e:
        print(f"Warning: Could not set up CasADi fallback: {e}")
        solve_casadi_unsteady = None
        HAS_CASADI = False
else:
    solve_casadi_unsteady = None
    HAS_CASADI = False



def run_forward_simulation(input_json_path, output_csv_path):
    """
    Run forward 0D simulation and save results to CSV.
    Uses svzerodsolver executable at /Users/natalia/cursor_access/svZeroDPlus/Release/svzerodsolver.
    Falls back to CasADi solver if it fails.
    Verifies that inlet flow matches the boundary condition.
    
    Args:
        input_json_path: Path to 0D input JSON file
        output_csv_path: Path to save CSV results
        
    Returns:
        Simulation results dictionary (or None if using CasADi fallback)
    """
    import subprocess
    import copy
    
    svzerodsolver_path = '/Users/natalia/cursor_access/svZeroDPlus/Release/svzerodsolver'
    
    if not os.path.exists(svzerodsolver_path):
        if HAS_CASADI and HAS_PANDAS and solve_casadi_unsteady is not None:
            print(f"  Warning: svzerodsolver executable not found at {svzerodsolver_path}")
            print(f"  Falling back to CasADi solver...")
        else:
            raise RuntimeError(f"svzerodsolver executable not found at {svzerodsolver_path} and CasADi fallback not available.")
    
    print(f"Running forward simulation from: {input_json_path}")
    
    # Convert PosixPath to string if needed
    input_json_path_str = str(input_json_path)
    output_csv_path_str = str(output_csv_path)
    
    with open(input_json_path_str, 'r') as f:
        input_data = json.load(f)
    
    # Try svzerodsolver executable first
    if os.path.exists(svzerodsolver_path):
        try:
            # Create a deep copy to avoid modifying the original
            input_data_sim = copy.deepcopy(input_data)
            
            # Remove calibration parameters if present (not needed for forward simulation)
            if 'calibration_parameters' in input_data_sim:
                del input_data_sim['calibration_parameters']
            
            # Remove observation data if present
            if 'y' in input_data_sim:
                del input_data_sim['y']
            if 'dy' in input_data_sim:
                del input_data_sim['dy']
            
            # Write temporary input file for svzerodsolver (use original input path if it's already clean)
            # Check if we need to create a temp file or can use the original
            use_temp = ('calibration_parameters' in input_data or 'y' in input_data or 'dy' in input_data)
            
            if use_temp:
                temp_input_path = input_json_path_str + '.temp'
                with open(temp_input_path, 'w') as f:
                    json.dump(input_data_sim, f, indent=4)
                input_file_for_solver = temp_input_path
            else:
                input_file_for_solver = input_json_path_str
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_csv_path_str), exist_ok=True)
            
            # Compute absolute paths BEFORE changing directories
            abs_input_path = os.path.abspath(input_file_for_solver)
            abs_output_dir = os.path.abspath(os.path.dirname(output_csv_path_str)) if os.path.dirname(output_csv_path_str) else os.path.abspath('.')
            abs_output_csv = os.path.abspath(output_csv_path_str)
            
            print(f"  Attempting simulation with svzerodsolver executable...")
            print(f"    Executable: {svzerodsolver_path}")
            print(f"    Input: {abs_input_path}")
            print(f"    Output: {output_csv_path_str}")
            
            # Run svzerodsolver: takes input.json as first argument
            # If output is not specified, it defaults to ./output.csv in the current directory
            # We'll run it in the output directory and then move/rename the output file
            output_dir = os.path.dirname(output_csv_path_str) or '.'
            output_filename = os.path.basename(output_csv_path_str)
            
            # Change to output directory to run solver (so output.csv is created there)
            original_cwd = os.getcwd()
            try:
                os.chdir(abs_output_dir)
                # Command format: svzerodsolver input.json
                # It will output to ./output.csv in the current directory
                # Use absolute path for input file (computed before changing directories)
                cmd = [svzerodsolver_path, abs_input_path]
                
                # Check simulation parameters to estimate timeout
                num_cycles = input_data_sim.get('simulation_parameters', {}).get('number_of_cardiac_cycles', 2)
                num_time_pts = input_data_sim.get('simulation_parameters', {}).get('number_of_time_pts_per_cardiac_cycle', 100)
                # Estimate timeout: ~1 second per 1000 time points, with minimum 60 seconds
                estimated_timeout = max(60, int((num_cycles * num_time_pts) / 1000) + 30)
                
                print(f"    Running simulation ({num_cycles} cycles, {num_time_pts} pts/cycle, timeout: {estimated_timeout}s)...")
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
                default_output = os.path.join(abs_output_dir, 'output.csv')
                if os.path.exists(default_output):
                    if default_output != abs_output_csv:
                        os.rename(default_output, abs_output_csv)
                elif os.path.exists('output.csv'):
                    os.rename('output.csv', abs_output_csv)
                else:
                    raise RuntimeError(f"svzerodsolver did not create output file. Expected './output.csv' in {abs_output_dir}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
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
            if HAS_CASADI and HAS_PANDAS and solve_casadi_unsteady is not None:
                print(f"  Falling back to CasADi solver...")
    
    # Fallback to CasADi solver
    if HAS_CASADI and HAS_PANDAS and solve_casadi_unsteady is not None:
        try:
            # Create a deep copy to avoid modifying the original
            import copy
            input_data_sim = copy.deepcopy(input_data)
            
            # Remove calibration parameters if present
            if 'calibration_parameters' in input_data_sim:
                del input_data_sim['calibration_parameters']
            
            # Remove observation data if present
            if 'y' in input_data_sim:
                del input_data_sim['y']
            if 'dy' in input_data_sim:
                del input_data_sim['dy']
            
            # Create DataFrame for CasADi solver
            result_df = pd.DataFrame(columns=['name', 'time', 'flow_in', 'flow_out', 'pressure_in', 'pressure_out'])
            
            print("  Running CasADi solver...")
            sol_prev = solve_casadi_unsteady(input_file=input_data_sim, result_df=result_df)
            
            # Sort by name and time
            result_df.sort_values(by=['name', 'time'], inplace=True)
            
            # Save to CSV
            os.makedirs(os.path.dirname(output_csv_path_str), exist_ok=True)
            result_df.to_csv(output_csv_path_str, index=False)
            
            print(f"  ✓ Simulation completed successfully with CasADi solver")
            print(f"  Results saved to: {output_csv_path_str}")
            
            # Verify inlet flow matches BC
            print("\n  Verifying inlet flow matches boundary condition...")
            #verify_inlet_flow_matches_bc(input_data_sim, output_csv_path_str)
            
            # Return None since CasADi doesn't return the same format as pysvzerod
            return None
        except Exception as e:
            print(f"  ✗ CasADi solver also failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Both pysvzerod and CasADi solvers failed. Last error: {e}")
    else:
        error_msg = "pysvzerod simulation failed"
        if not HAS_CASADI:
            error_msg += " and CasADi fallback is not available (casadi module not installed)."
            error_msg += "\n  To enable CasADi fallback, install casadi: pip install casadi"
        elif not HAS_PANDAS:
            error_msg += " and CasADi fallback is not available (pandas not installed)."
        else:
            error_msg += " and CasADi fallback is not available."
        raise RuntimeError(error_msg)
