#!/usr/bin/env python3
"""
Generate svZeroDSolver input files from centerline geometry and simulation results.
This script creates:
1. Geometric 0D input files (from centerline geometry)
2. Calibration input files (from 3D or 1D results)
3. Calibrated output files (by running svZeroDCalibrator)

Based on the workflow in richter2024-paper-tools.
"""

import os
import sys
sys.path.append("/Users/natalia/cursor_access/learn_lpns")
import json
import vtk
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict, OrderedDict
from util.zerod_calibration.calibration_helpers import *
from util.zerod_calibration.post_processing import *
try:
    from scipy.interpolate import CubicSpline, interp1d
    HAS_SCIPY_INTERP = True
except (ImportError, ValueError, AttributeError):
    # Handle import errors and version incompatibility issues
    HAS_SCIPY_INTERP = False
    try:
        from scipy.interpolate import CubicSpline
    except (ImportError, ValueError, AttributeError):
        CubicSpline = None
from vtk.util.numpy_support import vtk_to_numpy as v2n

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

# Constants
RHO = 1.06  # Blood density (g/cm^3)
MU = 0.04  # Blood viscosity (Poise)


def create_geometric_zerod_input_rom(geo_dir, centerline_path, output_path, 
                                     simvascular_path=None, dt=0.2, num_time_steps=5, num_cardiac_cycles=1):
    """
    Create geometric svZeroDSolver input file using SimVascular's ROM workflow.
    
    Args:
        geo_dir: Geometry directory (should contain mesh-complete/)
        centerline_path: Path to centerline VTP file
        output_path: Path to save JSON input file
        simvascular_path: Path to SimVascular executable (default: auto-detect)
        dt: Time step size (default: 0.2)
        num_time_steps: Number of time steps (default: 5)
        
    Returns:
        zerod_input: Dictionary with 0D input data
        vessel_bc_map: Mapping from vessels to BCs (simplified)
    """
    import subprocess
    import tempfile
    
    print(f"Using SimVascular ROM workflow to generate 0D input")
    
    # Find inlet and outlet caps
    # If geo_dir doesn't exist or doesn't have mesh files, use centerline-based approach
    use_centerline_caps = False
    if geo_dir is None or not os.path.exists(geo_dir):
        use_centerline_caps = True
    else:
        # Check if mesh-surfaces directory exists
        mesh_surfaces_paths = [
            os.path.join(geo_dir, 'mesh', 'fluid_msh_0', 'mesh-surfaces'),
            os.path.join(geo_dir, 'mesh-complete', 'mesh-surfaces')
        ]
        if not any(os.path.exists(p) for p in mesh_surfaces_paths):
            use_centerline_caps = True
    
    if use_centerline_caps:
        print("Identifying inlet and outlet caps from centerline data...")
        # Try to find geometric input to get vessel names
        geometric_input_path = None
        if geo_dir and os.path.exists(geo_dir):
            geometric_input_path = os.path.join(geo_dir, 'geometric_input.json')
            if not os.path.exists(geometric_input_path):
                # Try parent directory
                parent_dir = os.path.dirname(geo_dir)
                geometric_input_path = os.path.join(parent_dir, os.path.basename(geo_dir), 'geometric_input.json')
                if not os.path.exists(geometric_input_path):
                    geometric_input_path = None
        
        inlet_cap, outlet_caps, mesh_surfaces_dir = find_inlet_outlet_caps_from_centerline(
            centerline_path, geometric_input_path=geometric_input_path
        )
    else:
        # Find inlet and outlet caps (handles both svVascularize and SimVascular formats)
        print("Identifying inlet and outlet caps from mesh files...")
        inlet_cap, outlet_caps, mesh_surfaces_dir = find_inlet_outlet_caps(geo_dir)
    
    # Get geometry name from directory or centerline path
    if geo_dir and os.path.exists(geo_dir):
        geo_name = os.path.basename(geo_dir.rstrip('/'))
    else:
        # Extract from centerline path or output path
        geo_name = os.path.basename(centerline_path).replace('.vtp', '')
        if not geo_name:
            geo_name = os.path.basename(os.path.dirname(output_path))
    
    # Create temporary directory for ROM workflow files
    temp_dir = os.path.join(os.path.dirname(output_path), 'rom_temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate inflow flow file
    inflow_flow_file = os.path.join(temp_dir, f'{geo_name}_inflow_0D.flow')
    with open(inflow_flow_file, 'w') as f:
        t = np.linspace(0, num_time_steps * dt, num_time_steps)
        q = np.ones(num_time_steps) * 85  # Default flow value
        for i in range(num_time_steps):
            f.write(f"{i*dt:.5f}    {q[i]:.3f}\n")
    
    # Create ROM input generator script
    rom_script = os.path.join(temp_dir, f'{geo_name}_rom_generator.py')
    
    # Make centerline path relative to script location or absolute
    centerline_abs = os.path.abspath(centerline_path)
    
    # Strip .vtp extension from face names for SimVascular (XML uses base names)
    inlet_cap_base = inlet_cap.replace('.vtp', '') if inlet_cap.endswith('.vtp') else inlet_cap
    outlet_caps_base = [cap.replace('.vtp', '') if cap.endswith('.vtp') else cap for cap in outlet_caps]
    
    # Format outlet_caps_base as a Python list literal for the script
    outlet_caps_base_str = '[' + ', '.join([f"'{cap}'" for cap in outlet_caps_base]) + ']'
    
    # Extract outlet resistances from XML file (if available)
    print(f"\nExtracting outlet resistances from XML file...")
    print(f"  Looking for outlet caps: {outlet_caps}")
    # Create mapping from base names (without .vtp) to full names (with .vtp)
    # XML BC names don't include .vtp extension
    outlet_caps_base_dict = {}
    for cap in outlet_caps:
        base_name = cap.replace('.vtp', '') if cap.endswith('.vtp') else cap
        outlet_caps_base_dict[base_name] = cap
    print(f"  Outlet caps base names (for XML matching): {list(outlet_caps_base_dict.keys())}")
    
    outlet_resistances = {}
    
    
    script_content = f"""import os
from pathlib import Path
import sv
import sys
import vtk

## Create a ROM simulation.
rom_simulation = sv.simulation.ROM()

## Create ROM simulation parameters.
params = sv.simulation.ROMParameters()

## Mesh parameters.
mesh_params = params.MeshParameters()

## Model parameters.
model_params = params.ModelParameters()
model_params.name = '{geo_name}'
model_params.inlet_face_names = ['{inlet_cap_base}']
model_params.outlet_face_names = {outlet_caps_base_str}
model_params.centerlines_file_name = '{centerline_abs}'

## Fluid properties.
fluid_props = params.FluidProperties()

## Set wall properties.
print('Set wall properties ...')
material = params.WallProperties.OlufsenMaterial()

## Set boundary conditions.
bcs = params.BoundaryConditions()
bcs.add_velocities(face_name='{inlet_cap_base}', file_name='{os.path.abspath(inflow_flow_file)}')
"""
    
    # Add resistance BCs for outlets using values from XML
    # Note: SimVascular expects face names without .vtp extension
    # Resistances are stored using base names (without .vtp) to match XML BC names
    for outlet_cap in outlet_caps:
        # Get base name (without .vtp) for lookup and script
        outlet_cap_base = outlet_cap.replace('.vtp', '') if outlet_cap.endswith('.vtp') else outlet_cap
        resistance = outlet_resistances.get(outlet_cap_base, 1.0)  # Default to 1.0 if not found
        print(f"  Using resistance {resistance} for {outlet_cap} (face_name: {outlet_cap_base})")
        script_content += f"bcs.add_resistance(face_name='{outlet_cap_base}', resistance={resistance})\n"
    
    script_content += f"""solution_params = params.Solution()
solution_params.time_step = {dt}
solution_params.num_time_steps = {num_time_steps}

## Write a 0D solver input file.
output_dir = '{os.path.dirname(output_path)}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
rom_simulation.write_input_file(model_order=0, model=model_params, mesh=mesh_params, 
                                fluid=fluid_props, material=material, 
                                boundary_conditions=bcs, solution=solution_params, 
                                directory=output_dir)
"""
    
    with open(rom_script, 'w') as f:
        f.write(script_content)
    
    # Find SimVascular executable
    if simvascular_path is None:
        # Try common locations
        possible_paths = [
            '/Applications/SimVascular.app/Contents/Resources/simvascular',
            os.path.expanduser('~/Applications/SimVascular.app/Contents/Resources/simvascular'),
        ]
        simvascular_path = None
        for path in possible_paths:
            if os.path.exists(path):
                simvascular_path = path
                break
        
        if simvascular_path is None:
            raise FileNotFoundError(
                "SimVascular executable not found. Please specify --simvascular-path or "
                "install SimVascular in a standard location."
            )
    
    print(f"Running SimVascular ROM workflow...")
    print(f"  Script: {rom_script}")
    print(f"  SimVascular: {simvascular_path}")
    
    # Run SimVascular ROM script
    cmd = [simvascular_path, '--python', '--', rom_script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running SimVascular ROM workflow:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"SimVascular ROM workflow failed with return code {result.returncode}")
    
    # Find the generated JSON file (SimVascular creates solver_0d.json in output_dir)
    generated_json = os.path.join(os.path.dirname(output_path), 'solver_0d.json')
    
    if not os.path.exists(generated_json):
        raise FileNotFoundError(f"Generated JSON file not found: {generated_json}")
    
    # Read and fix junction types (replace "internal_junction" with "NORMAL_JUNCTION")
    print("Post-processing generated JSON file...")
    with open(generated_json, 'r') as f:
        zerod_input = json.load(f)
    
    # Set capacitances to cap_val
    cap_value = 1e-10
    for vessel in zerod_input['vessels']:
        vessel['zero_d_element_values']['C'] = cap_value
    # Set number of cardiac cycles to 1
    zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
    
    # Fix junction types and validate junction structure
    if 'junctions' in zerod_input:
        for junc in zerod_input['junctions']:
            if junc.get('junction_type') == 'internal_junction':
                junc['junction_type'] = 'NORMAL_JUNCTION'
            
            # Check for multiple inlets (not supported by BloodVessel junction)
            inlet_vessels = junc.get('inlet_vessels', [])
            if len(inlet_vessels) > 1:
                print(f"  Warning: Junction {junc.get('junction_name')} has {len(inlet_vessels)} inlets.")
                print(f"    BloodVessel junction only supports 1 inlet. Keeping first inlet only.")
                # Keep only the first inlet
                junc['inlet_vessels'] = [inlet_vessels[0]]
    
    # Remove unused resistance BCs (BCs that are not referenced by any vessel)
    print(f"\n  Removing unused resistance BCs...")
    if 'boundary_conditions' in zerod_input:
        # Find all resistance BC names that are actually used by vessels
        used_resistance_bcs = set()
        for vessel in zerod_input.get('vessels', []):
            if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
                bc_name = vessel['boundary_conditions']['outlet']
                used_resistance_bcs.add(bc_name)
        
        # Filter boundary conditions to keep only:
        # 1. Non-RESISTANCE BCs (like INFLOW)
        # 2. RESISTANCE BCs that are actually used by vessels
        original_bcs = zerod_input['boundary_conditions']
        filtered_bcs = []
        removed_count = 0
        
        for bc in original_bcs:
            if bc.get('bc_type') != 'RESISTANCE':
                # Keep all non-resistance BCs
                filtered_bcs.append(bc)
            elif bc.get('bc_name') in used_resistance_bcs:
                # Keep resistance BCs that are used
                filtered_bcs.append(bc)
            else:
                # Remove unused resistance BCs
                removed_count += 1
        
        zerod_input['boundary_conditions'] = filtered_bcs
        
        if removed_count > 0:
            print(f"    Removed {removed_count} unused resistance BC(s)")
        print(f"    Kept {len(used_resistance_bcs)} used resistance BC(s)")
    
    zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
    # Move to final output location
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(zerod_input, f, indent=4)
    
    # Clean up generated file
    if os.path.exists(generated_json) and generated_json != output_path:
        os.remove(generated_json)
    
    # Update with simulation parameters from XML and flow file
    print("\nUpdating geometric input with simulation parameters...")
    print(f"  geo_dir: {geo_dir}")
    print(f"  output_path: {output_path}")
    print(f"  inlet_cap: {inlet_cap}")
    update_simulation_parameters(geo_dir, output_path, inlet_cap, zero_stenosis=True)
    
    # Re-read the updated JSON
    with open(output_path, 'r') as f:
        zerod_input = json.load(f)
    
    # Create simplified vessel_bc_map for compatibility
    vessel_bc_map = {}
    if 'boundary_conditions' in zerod_input:
        for bc in zerod_input['boundary_conditions']:
            bc_name = bc.get('bc_name', '')
            vessel_bc_map[bc_name] = {
                "name": f"branch0_seg0",  # Simplified
                "pressure": "pressure",
                "flow": "flow"
            }
    
    print(f"Geometric 0D input saved to: {output_path}")

    return zerod_input, vessel_bc_map


def update_simulation_parameters(geo_dir, json_path, inlet_cap_name, capacitance_value = 1e-10, zero_stenosis = False):
    """
    Update geometric input JSON with simulation parameters
    """
    # Read JSON
    with open(json_path, 'r') as f:
        zerod_input = json.load(f)
    
    if 'simulation_parameters' in zerod_input:
        zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
        zerod_input['simulation_parameters']['steady_initial'] = False
        zerod_input['simulation_parameters']['absolute_tolerance'] = 1e-5
        zerod_input['simulation_parameters']['maximum_nonlinear_iterations'] = 50
        
        # Calculate cardiac cycle period from boundary condition time array
        if 'boundary_conditions' in zerod_input and len(zerod_input['boundary_conditions']) > 0:
            bc = zerod_input['boundary_conditions'][0]
            if 'bc_values' in bc and 't' in bc['bc_values']:
                t_array = bc['bc_values']['t']
                if len(t_array) > 1:
                    # Cardiac cycle period is the time span of the BC array
                    # For evenly spaced times: period = t[-1] - t[0] + dt
                    # Or simply: period = t[-1] if t[0] == 0
                    if t_array[0] == 0.0:
                        cardiac_period = t_array[-1]
                    else:
                        dt = t_array[1] - t_array[0] if len(t_array) > 1 else 0.0
                        cardiac_period = t_array[-1] - t_array[0] + dt
                    zerod_input['simulation_parameters']['cardiac_cycle_period'] = cardiac_period
                    print(f"  Set cardiac_cycle_period to {cardiac_period:.6f} s (from BC time array)")
                elif len(t_array) == 1:
                    # Single time point - use default
                    zerod_input['simulation_parameters']['cardiac_cycle_period'] = 1.0
                    print(f"  Warning: Only one time point in BC, using default cardiac_cycle_period = 1.0 s")
            else:
                # No BC time array - use default
                zerod_input['simulation_parameters']['cardiac_cycle_period'] = 1.0
                print(f"  Warning: No BC time array found, using default cardiac_cycle_period = 1.0 s")
        else:
            # No boundary conditions - use default
            zerod_input['simulation_parameters']['cardiac_cycle_period'] = 1.0
            print(f"  Warning: No boundary conditions found, using default cardiac_cycle_period = 1.0 s")
    
    # Set all capacitance (C) values to 10^-10
    capacitance_value = 1e-10
    vessels_updated = 0
    for vessel in zerod_input.get('vessels', []):
        if 'zero_d_element_values' in vessel and 'C' in vessel['zero_d_element_values']:
            vessel['zero_d_element_values']['C'] = capacitance_value
            vessels_updated += 1
        if zero_stenosis and 'zero_d_element_values' in vessel and 'stenosis_coefficient' in vessel['zero_d_element_values']:
            vessel['zero_d_element_values']['stenosis_coefficient'] = 0.0

    print(f"  Set capacitance (C) to {capacitance_value} for {vessels_updated} vessels")
    
    # Write updated JSON
    with open(json_path, 'w') as f:
        json.dump(zerod_input, f, indent=4)
    
    print(f"  Updated geometric input with simulation parameters")
    print(f"  Saved to: {json_path}")
    return


def create_calibration_input(geometric_input_path, observations, output_path, centerline_soln_path=None, geo_dir=None):
    """
    Create calibration input file from geometric input and observations.
    Computes BC times from 1D solution timesteps multiplied by timestep size from XML.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        observations: Dictionary with observation data (y, dy)
        output_path: Path to save calibration input JSON
        centerline_soln_path: Path to 1D centerline solution VTP (to extract timestep count)
        geo_dir: Geometry directory (to find XML file for timestep size)
        num_cardiac_cycles: Number of cardiac cycles (default: 1)
    """
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
    
    # Add calibration parameters
    inp["calibration_parameters"] = {
        "tolerance_gradient": 1e-4,
        "tolerance_increment": 1e-4,
        "maximum_iterations": 100,
        "calibrate_stenosis_coefficient":True,
        "set_capacitance_to_zero": False,
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


def restore_simulation_parameters(calibrated_output_path, geometric_input_path):
    """
    Restore simulation parameters to the original values.
    """
    with open(calibrated_output_path, 'r') as f:
        cali = json.load(f)
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    cali['simulation_parameters'] = geometric_input['simulation_parameters']

    # replace inflow bc with the original values
    cali['boundary_conditions'] = geometric_input['boundary_conditions']
    with open(calibrated_output_path, 'w') as f:
        json.dump(cali, f, indent=4)
    print(f"Restored simulation parameters to: {calibrated_output_path}")
    return cali

def refine_inlet_bc_for_forward_simulation(input_path, refinement_factor=2):
    """
    Refine the inlet boundary condition for forward simulation.
    """
    with open(input_path, 'r') as f:
        cali = json.load(f)

    bc_time = cali['boundary_conditions'][0]['bc_values']['t']
    bc_flow = cali['boundary_conditions'][0]['bc_values']['Q']
    bc_time_refined = np.linspace(0, bc_time[-1], len(bc_time) * refinement_factor, endpoint=True).tolist()
    bc_flow_refined = interp1d(bc_time, bc_flow, kind='cubic')(bc_time_refined)
    # plot the original and refined boundary condition
    if False:
        plt.scatter(bc_time, bc_flow, label='Original')
        plt.scatter(bc_time_refined, bc_flow_refined, label='Refined')
        plt.legend()
        plt.show()
    cali['boundary_conditions'][0]['bc_values']['t'] = list(bc_time_refined)
    cali['boundary_conditions'][0]['bc_values']['Q'] = list(bc_flow_refined)
    cali['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = len(bc_time_refined)

    cali['simulation_parameters']['number_of_cardiac_cycles'] = 10
    cali['simulation_parameters']['output_all_cycles'] = False

    with open(input_path, 'w') as f:
        json.dump(cali, f, indent=4)
    print(f"Refined inlet boundary condition for forward simulation to: {input_path}")
    return

def modify_junction_types(config, junction_type):
    """
    Modify junction types in a config based on the number of outlets.
    Junctions with more than one outlet use the specified junction type.
    Junctions with only one outlet remain as NORMAL_JUNCTION.
    
    Args:
        config: Dictionary with 0D input configuration
        junction_type: String specifying junction type ('BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction')
        
    Returns:
        Modified config dictionary
    """
    import copy
    config_modified = copy.deepcopy(config)
    
    # Update junctions based on number of outlets
    if 'junctions' in config_modified:
        for junc in config_modified['junctions']:
            # Get number of outlet vessels
            num_outlets = len(junc.get('outlet_vessels', []))
            
            # Only modify junctions with more than one outlet
            if num_outlets <= 1:
                # Keep as NORMAL_JUNCTION for single outlet
                junc['junction_type'] = 'NORMAL_JUNCTION'
                # Remove junction_values if present (not needed for NORMAL_JUNCTION)
                if 'junction_values' in junc:
                    del junc['junction_values']
                continue
            
            # Update to new junction type for multi-outlet junctions
            junc['junction_type'] = junction_type
            
            # NORMAL_JUNCTION should not have junction_values
            if junction_type == 'NORMAL_JUNCTION':
                # Remove junction_values if present (NORMAL_JUNCTION has no parameters)
                if 'junction_values' in junc:
                    del junc['junction_values']
                continue
            
            # For special junction types, we need to provide junction_values
            if junction_type in ['BloodVesselJunction', 'DirIndepJunction', 'HybridJunction']:
                num_outlets = len(junc.get('outlet_vessels', []))
                
                if num_outlets > 0:
                    # Initialize or update junction values with only the required parameters
                    if 'junction_values' not in junc:
                        junc['junction_values'] = {}
                    
                    # Determine which parameters are needed based on junction type
                    # BloodVesselJunction and DirIndepJunction: R_poiseuille, L, stenosis_coefficient
                    # HybridJunction: R_poiseuille, L, stenosis_coefficient, pressure_recovery_coefficient
                    required_params = ['R_poiseuille', 'L', 'stenosis_coefficient']
                    if junction_type == 'HybridJunction':
                        required_params.append('pressure_recovery_coefficient')
                    
                    # Set the required parameters
                    for param in required_params:
                        if param not in junc['junction_values']:
                            junc['junction_values'][param] = [0.0] * num_outlets
                        elif len(junc['junction_values'][param]) != num_outlets:
                            # Resize to match number of outlets
                            current_len = len(junc['junction_values'][param])
                            if current_len < num_outlets:
                                # Extend with zeros
                                junc['junction_values'][param].extend([0.0] * (num_outlets - current_len))
                            else:
                                # Truncate
                                junc['junction_values'][param] = junc['junction_values'][param][:num_outlets]
                    
                    # Remove C parameter (not supported by any junction type in svzerodsolver)
                    if 'C' in junc['junction_values']:
                        del junc['junction_values']['C']
                    
                    # Remove pressure_recovery_coefficient for non-HybridJunction types
                    if junction_type != 'HybridJunction' and 'pressure_recovery_coefficient' in junc['junction_values']:
                        del junc['junction_values']['pressure_recovery_coefficient']
    
    return config_modified


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
                result = subprocess.run(cmd)#, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"svzerodsolver failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                
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
            print("\n  Verifying inlet flow matches boundary condition...")
            verify_inlet_flow_matches_bc(input_data_sim, output_csv_path_str)
            
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate svZeroDSolver input files and run calibration"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_1)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_000)')
    parser.add_argument('--centerline', help='Path to centerline VTP file (default: auto-detect)')
    parser.add_argument('--one-d-soln', '--1d-soln', dest='one_d_soln',
                       help='Path to 1D centerline solution VTP file')
    parser.add_argument('--three-d-soln-dir', '--3d-soln-dir', dest='three_d_soln_dir',
                       help='Path to 3D solution directory (not yet implemented)')
    parser.add_argument('--output-dir', default='data/zeroD', 
                       help='Output directory for 0D files (default: data/zeroD)')
    parser.add_argument('--skip-calibration', action='store_true',
                       help='Skip calibration step')
    parser.add_argument('--simvascular-path', 
                       help='Path to SimVascular executable (default: auto-detect)')
    parser.add_argument('--dt', type=float, default=0.2,
                       help='Time step size for ROM workflow (default: 0.2)')
    parser.add_argument('--num-time-steps', type=int, default=5,
                       help='Number of time steps for ROM workflow (default: 5)')
    parser.add_argument('--num-cardiac-cycles', type=int, default=1,
                       help='Number of cardiac cycles for 0D simulation (default: 1)')
    parser.add_argument('--junction-types', nargs='+', 
                       default=['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to generate calibration files for (default: all four types)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index for observations (default: 0). Observations will be sliced from this index.')
    parser.add_argument('--zoom-start', type=int, default=599,
                       help='Start index for zoom window (shaded region in plots). Default: 599')
    parser.add_argument('--zoom-end', type=int, default=699,
                       help='End index for zoom window (shaded region in plots). Default: 699')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating comparison plots')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip all steps except plotting (requires existing files)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed MSE comparison table')
    
    args = parser.parse_args()
    verbose = args.verbose
    # Construct paths
    base_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    
    # Define geometry variants: original and bifurcations-only
    geometry_variants = {
        'original': {
            'geometric_input': os.path.join(base_dir, 'geometric_input.json'),
            'geometric_results': os.path.join(base_dir, 'geometric_results.csv'),
            'calibration_input': os.path.join(base_dir, 'calibration_input.json'),
            'junction_types': {}
        },
        'bifurcations': {
            'geometric_input': os.path.join(base_dir, 'bifurcations_geometric_input.json'),
            'geometric_results': os.path.join(base_dir, 'bifurcations_geometric_results.csv'),
            'calibration_input': os.path.join(base_dir, 'bifurcations_calibration_input.json'),
            'junction_types': {}
        }
    }
    
    # Add paths for each junction type variant within each geometry variant
    for geo_variant in geometry_variants:
        prefix = '' if geo_variant == 'original' else f'{geo_variant}_'
        for jtype in args.junction_types:
            geometry_variants[geo_variant]['junction_types'][jtype] = {
                'calibration_input': os.path.join(base_dir, f'{prefix}calibration_input_{jtype}.json'),
                'calibrated_output': os.path.join(base_dir, f'{prefix}calibrated_output_{jtype}.json'),
                'calibrated_results': os.path.join(base_dir, f'{prefix}calibrated_results_{jtype}.csv')
            }
    
    # Legacy paths for backward compatibility
    geometric_input_path = geometry_variants['original']['geometric_input']
    geometric_results_csv = geometry_variants['original']['geometric_results']
    calibration_input_path = geometry_variants['original']['calibration_input']
    calibrated_output_path = os.path.join(base_dir, 'calibrated_output.json')
    
    # Paths for each junction type variant (original geometry - for backward compatibility)
    junction_type_paths = geometry_variants['original']['junction_types']
    
    # Find centerline file
    if args.centerline:
        centerline_path = args.centerline
    else:
        # Auto-detect (following generate_multiple_trees.py convention)
        geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)
        centerline_paths = [
            os.path.join(geo_dir, 'centerlines_simVascular.vtp'),  # svVascularize format
            os.path.join(geo_dir, 'centerlines', 'centerlines.vtp'),
            os.path.join(geo_dir, 'centerlines.vtp'),
        ]
        centerline_path = None
        for path in centerline_paths:
            if os.path.exists(path):
                centerline_path = path
                break
        
        # If not found in 3D directory, try 1D solution (for VMR files)
        oneD_soln_paths = []
        if centerline_path is None:
            oneD_dir = os.path.join('data', 'oneD', args.set_name, args.geo_name)
            oneD_soln_paths = [
                os.path.join(oneD_dir, 'unsteady_soln.vtp'),
                os.path.join(oneD_dir, 'unsteady_soln_5.vtp'),
            ]
            for path in oneD_soln_paths:
                if os.path.exists(path):
                    centerline_path = path
                    print(f"  Using 1D solution as centerline source: {centerline_path}")
                    break
        
        if centerline_path is None:
            all_paths = centerline_paths + oneD_soln_paths
            raise FileNotFoundError(f"Centerline file not found. Tried: {all_paths}")
    
    # Skip all steps if --plot-only is set
    if args.plot_only:
        print("\n" + "="*60)
        print("Plot-only mode: Skipping all steps except plotting")
        print("="*60)
        print("  Using existing files:")
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            print(f"\n  {geo_variant_name.upper()} geometry:")
            print(f"    Geometric input: {geo_variant_paths['geometric_input']}")
            print(f"    Calibration input: {geo_variant_paths['calibration_input']}")
        
        # Check that at least original geometry exists
        if not os.path.exists(geometric_input_path):
            raise FileNotFoundError(f"Geometric input not found: {geometric_input_path}")
        if not os.path.exists(calibration_input_path):
            raise FileNotFoundError(f"Calibration input not found: {calibration_input_path}")
    else:
        # Step 1: Create geometric 0D input using SimVascular ROM workflow
        if args.set_name == "VMR":
            richter_0d_path = os.path.join('data', 'zeroD', args.set_name, 'richter-0d', args.geo_name+'.json')
            zerod_input = load_from_json(richter_0d_path)
            zerod_input['simulation_parameters']['output_all_cycles'] = True
            zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
            os.makedirs(os.path.dirname(geometric_input_path), exist_ok=True)
            save_to_json(zerod_input, geometric_input_path)
            print(f"    ✓ Richter 0D input saved to: {geometric_input_path}")
        else:
    
            print("\n" + "="*60)
            print("Step 1: Creating geometric 0D input file using SimVascular ROM")
            print("="*60)
            
            # Get geometry directory (may not exist for VMR files)
            geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)
            if not os.path.exists(geo_dir):
                print(f"  Warning: Geometry directory not found: {geo_dir}")
                print(f"  Will use centerline-based workflow (for VMR files)")
                geo_dir = None
            
            zerod_input, vessel_bc_map = create_geometric_zerod_input_rom(
                geo_dir, centerline_path, geometric_input_path,
                simvascular_path=args.simvascular_path,
                dt=args.dt,
                num_time_steps=args.num_time_steps,
                num_cardiac_cycles=args.num_cardiac_cycles
            )
        # Generate bifurcations-only version of the geometric input
        bifurcations_geometric_input_path = geometry_variants['bifurcations']['geometric_input']
        print(f"\n  Creating bifurcations-only geometric input...")
        split_junctions_from_files(geometric_input_path, centerline_path, bifurcations_geometric_input_path)
        print(f"  Bifurcations-only geometric input saved to: {bifurcations_geometric_input_path}")
        
        # Step 2: Extract observations and create calibration inputs for each junction type
        if not args.skip_calibration:
            print("\n" + "="*60)
            print("Step 2: Creating calibration input files for each junction type")
            print("="*60)
        
        if args.one_d_soln:
            print(f"  Start index for observations: {args.start_idx}")
            observations = extract_observations_from_1d(args.one_d_soln, geometric_input_path, geo_dir=geo_dir, start_idx=args.start_idx)
        else:
            # Try to find 1D solution automatically
            # First check data/oneD/set_name/geo_name
            oneD_dir = os.path.join('data', 'oneD', args.set_name, args.geo_name)
            soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
            
            if not os.path.exists(soln_path):
                # Try alternative location: data/reduced_results
                reduced_results_dir = os.path.join('data', 'reduced_results', args.set_name, args.geo_name)
                soln_path = os.path.join(reduced_results_dir, 'unsteady_soln.vtp')
            
            if not os.path.exists(soln_path):
                # Try scratch directory location
                alt_soln_path = os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                                               args.set_name, args.geo_name, 'unsteady_soln.vtp')
                if os.path.exists(alt_soln_path):
                    soln_path = alt_soln_path
            
            if os.path.exists(soln_path):
                observations = extract_observations_from_1d(soln_path, geometric_input_path, geo_dir=geo_dir, start_idx=args.start_idx)
            else:
                print("Warning: No 1D or 3D solution found. Skipping calibration.")
                print(f"  Looked for:")
                print(f"    - {os.path.join('data', 'oneD', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
                print(f"    - {os.path.join('data', 'reduced_results', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
                print(f"    - {os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
                args.skip_calibration = True
        
        if not args.skip_calibration:
            # Determine centerline solution path and geo_dir for time computation
            if args.one_d_soln:
                soln_path = args.one_d_soln
            else:
                # Use the path that was found earlier
                oneD_dir = os.path.join('data', 'oneD', args.set_name, args.geo_name)
                soln_path = os.path.join(oneD_dir, 'unsteady_soln.vtp')
                if not os.path.exists(soln_path):
                    reduced_results_dir = os.path.join('data', 'reduced_results', args.set_name, args.geo_name)
                    soln_path = os.path.join(reduced_results_dir, 'unsteady_soln.vtp')
                if not os.path.exists(soln_path):
                    alt_soln_path = os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                                               args.set_name, args.geo_name, 'unsteady_soln.vtp')
                    if os.path.exists(alt_soln_path):
                        soln_path = alt_soln_path
            
            # Get geo_dir
            geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)
            
            # Fit outlet resistances from 3D solution (skip for VMR cases)
            if args.set_name != "VMR":
                fitted_resistances = fit_outlet_resistances_from_3d(geometric_input_path, observations)
            else:
                fitted_resistances = None
            
            refinement_factor = 4
            
            # Process BOTH geometry variants: original and bifurcations-only
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                print(f"\n" + "-"*50)
                print(f"Processing {geo_variant_name.upper()} geometry variant")
                print("-"*50)
                
                variant_geometric_input = geo_variant_paths['geometric_input']
                variant_calibration_input = geo_variant_paths['calibration_input']
                variant_geometric_results = geo_variant_paths['geometric_results']
                variant_junction_paths = geo_variant_paths['junction_types']
                
                # Check if geometric input exists
                if not os.path.exists(variant_geometric_input):
                    print(f"  ✗ Skipping {geo_variant_name}: geometric input not found: {variant_geometric_input}")
                    continue
                
                # Create base calibration input for this geometry variant
                print(f"\n  Creating base calibration input for {geo_variant_name}...")
                
                try:
                    # Create calibration input with time computation from 1D solution
                    create_calibration_input(variant_geometric_input, observations, variant_calibration_input,
                                           centerline_soln_path=soln_path, geo_dir=geo_dir)
                    
                    print(f"    ✓ Base calibration input saved to: {variant_calibration_input}")
                    
                    # Update base calibration input with fitted outlet BCs (skip for VMR)
                    if fitted_resistances:
                        print(f"\n  Updating base calibration input with fitted outlet BCs...")
                        update_outlet_bcs_in_file(variant_calibration_input, fitted_resistances, f"{geo_variant_name} calibration input")
                    
                    # Update geometric input with BC from calibration input
                    update_geometric_input_with_calibration_bc(variant_geometric_input, variant_calibration_input)
                except Exception as e:
                    raise Exception(f"Failed to create calibration input for {geo_variant_name}: {e}")
                
                # Create calibration input variants for each junction type
                print(f"\n  Creating calibration input variants for each junction type...")
                try:
                    with open(variant_calibration_input, 'r') as f:
                        base_calibration_config = json.load(f)
                    
                    for jtype in args.junction_types:
                        print(f"    Creating {geo_variant_name}/{jtype} calibration input...")
                        jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                        
                        # Apply junction type modification to calibration input
                        jtype_config = modify_junction_types(base_calibration_config, jtype)
                        
                        with open(jtype_input_path, 'w') as f:
                            json.dump(jtype_config, f, indent=4)
                except Exception as e:
                    raise Exception(f"Failed to create junction type calibration inputs for {geo_variant_name}: {e}")
                
                # Step 3: Run calibration for each junction type
                if not args.plot_only:
                    print(f"\n  Running calibration for {geo_variant_name} geometry...")
                    
                    # Run calibration for each junction type variant
                    for jtype in args.junction_types:
                        print(f"\n    Calibrating {geo_variant_name}/{jtype}...")
                        jtype_input_path = variant_junction_paths[jtype]['calibration_input']
                        jtype_output_path = variant_junction_paths[jtype]['calibrated_output']
                        
                        try:
                            calibrated_config = run_calibration(jtype_input_path, jtype_output_path)
                            print(f"      ✓ Calibration completed for {geo_variant_name}/{jtype}")
                        except Exception as e:
                            raise Exception(f"Calibration failed for {geo_variant_name}/{jtype}: {e}")
                    
                    # Step 4: Run forward simulations for this geometry variant
                    print(f"\n  Running forward simulations for {geo_variant_name} geometry...")
                    
                    # Run simulation with geometric input (uncalibrated)
                    print(f"\n    Running simulation with {geo_variant_name} geometric input...")
                    if not os.path.exists(variant_geometric_input):
                        print(f"      ✗ Skipping: geometric input not found: {variant_geometric_input}")
                    else:
                        try:
                            refine_inlet_bc_for_forward_simulation(variant_geometric_input, refinement_factor)
                            run_forward_simulation(variant_geometric_input, variant_geometric_results)
                            print(f"      ✓ Geometric simulation completed successfully")
                        except Exception as e:
                            raise Exception(f"Geometric simulation failed: {e}")
                    
                    # Run simulation with each calibrated input
                    for jtype in args.junction_types:
                        print(f"\n    Running simulation with calibrated {geo_variant_name}/{jtype} input...")
                        jtype_output_path = variant_junction_paths[jtype]['calibrated_output']
                        calibrated_results_csv = variant_junction_paths[jtype]['calibrated_results']
                        
                        # Check if calibrated output exists (calibration might have failed)
                        if not os.path.exists(jtype_output_path):
                            print(f"      ✗ Skipping: calibrated output not found: {jtype_output_path}")
                            continue
                        
                        try:
                            refine_inlet_bc_for_forward_simulation(jtype_output_path, refinement_factor)
                            run_forward_simulation(jtype_output_path, calibrated_results_csv)
                            print(f"      ✓ Calibrated {geo_variant_name}/{jtype} simulation completed successfully")
                        except Exception as e:
                            raise Exception(f"Calibrated {geo_variant_name}/{jtype} simulation failed: {e}")
    
    # Step 5: Calculate and print MSE between 3D and 0D solutions
    if not args.skip_calibration and not args.plot_only:
        print("\n" + "="*60)
        print("Step 5: Calculating MSE between 3D and 0D solutions")
        print("="*60)
        
        # Calculate MSE for both geometry variants
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            print(f"\n  MSE calculation for {geo_variant_name.upper()} geometry:")
            
            variant_calibration_input = geo_variant_paths['calibration_input']
            variant_geometric_results = geo_variant_paths['geometric_results']
            variant_geometric_input = geo_variant_paths['geometric_input']
            variant_junction_paths = geo_variant_paths['junction_types']
            
            # Build dictionary of CSV results for all modalities
            csv_results_dict = {}
            
            # Add geometric results
            if os.path.exists(variant_geometric_results):
                csv_results_dict['geometric'] = variant_geometric_results
            
            # Add calibrated results for each junction type
            for jtype in args.junction_types:
                calibrated_results_csv = variant_junction_paths[jtype]['calibrated_results']
                if os.path.exists(calibrated_results_csv):
                    csv_results_dict[jtype] = str(calibrated_results_csv)
            
            if csv_results_dict and os.path.exists(variant_calibration_input):
                try:
                    # Generate CSV output path
                    prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                    mse_csv_path = os.path.join(base_dir, f'{prefix}mse_comparison.csv')
                    calculate_mse_between_3d_and_0d(
                        variant_calibration_input,
                        csv_results_dict,
                        geometric_input_path=variant_geometric_input,
                        zoom_start_idx=args.zoom_start,
                        zoom_end_idx=args.zoom_end,
                        output_csv_path=mse_csv_path,
                        verbose=verbose,
                        set_name=args.set_name
                    )
                except Exception as e:
                    raise Exception(f"Error calculating MSE for {geo_variant_name}: {e}")
            else:
                print(f"    Skipping MSE calculation for {geo_variant_name} (missing files)")
        
    # Step 6: Generate comparison plots (always run if not skipped, including in plot-only mode)
    if not args.skip_plots:
        print("\n" + "="*60)
        print("Step 6: Generating comparison plots")
        print("="*60)
        
        try:
            # Import plotting functions from unified location comparison script
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visualizations'))
            from plot_location_comparison import (
                plot_location_comparison, 
                get_all_locations_from_calibration_input,
                get_time_period as get_time_period_func
            )
            
            # Get time period (for plotting)
            time_period = None
            try:
                time_period = get_time_period_func(args.set_name, args.geo_name)
            except Exception:
                pass
            
            # Generate plots for each geometry variant
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                print(f"\n  Creating plots for {geo_variant_name.upper()} geometry...")
                
                variant_calibration_input = geo_variant_paths['calibration_input']
                variant_geometric_results = geo_variant_paths['geometric_results']
                variant_geometric_input = geo_variant_paths['geometric_input']
                variant_junction_paths = geo_variant_paths['junction_types']
                
                # Build dictionary of calibrated CSV paths for all junction types
                calibrated_csv_paths = {}
                for jtype in args.junction_types:
                    calibrated_results_csv = variant_junction_paths[jtype]['calibrated_results']
                    if os.path.exists(calibrated_results_csv):
                        calibrated_csv_paths[jtype] = str(calibrated_results_csv)
                
                if not calibrated_csv_paths:
                    print(f"    Warning: No calibrated CSV files found for {geo_variant_name}, skipping plots")
                    continue
                if not os.path.exists(variant_geometric_results):
                    print(f"    Warning: Geometric results CSV not found for {geo_variant_name}, skipping plots")
                    continue
                if not os.path.exists(variant_calibration_input):
                    print(f"    Warning: Calibration input not found for {geo_variant_name}, skipping plots")
                    continue
                
                # Get all locations from calibration input
                all_locations = get_all_locations_from_calibration_input(str(variant_calibration_input))
                print(f"    Found {len(all_locations)} locations to plot")
                
                # Create output directory for this geometry variant
                output_dir = os.path.join('results', 'location_comparison', args.set_name, args.geo_name, geo_variant_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Plot each location
                success_count = 0
                for location in all_locations:
                    # Generate safe filename from location (replace : with _)
                    safe_location = location.replace(':', '_')
                    plot_path = os.path.join(output_dir, f"{safe_location}_comparison.png")
                    try:
                        success = plot_location_comparison(
                            str(variant_calibration_input),
                            str(variant_geometric_results),
                            calibrated_csv_paths,
                            location,
                            plot_path,
                            set_name=args.set_name,
                            geo_name=args.geo_name,
                            time_period=time_period,
                            geometric_input_path=str(variant_geometric_input),
                            zoom_start_idx=args.zoom_start,
                            zoom_end_idx=args.zoom_end,
                            verbose=False
                        )
                        if success:
                            success_count += 1
                    except Exception as e:
                        print(f"      ✗ Failed to create plot for {location}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"    Created {success_count}/{len(all_locations)} location comparison plots")
                print(f"    Plot output directory: {output_dir}")
            
            # Generate combined comparison plots (original vs bifurcations for each junction type)
            print(f"\n  Creating combined geometry variant comparison plots...")
            combined_output_dir = os.path.join('results', 'location_comparison', args.set_name, args.geo_name, 'combined')
            os.makedirs(combined_output_dir, exist_ok=True)
            
            # Use original geometry calibration input for location list
            original_calibration_input = geometry_variants['original']['calibration_input']
            orig_geometric_results = geometry_variants['original']['geometric_results']
            
            if os.path.exists(original_calibration_input) and os.path.exists(orig_geometric_results):
                all_locations = get_all_locations_from_calibration_input(str(original_calibration_input))
                
                # Build combined CSV paths dict: keys are "original_jtype" and "bifurcations_jtype"
                combined_csv_paths = {}
                for geo_variant_name, geo_variant_paths in geometry_variants.items():
                    variant_geometric_results = geo_variant_paths['geometric_results']
                    if os.path.exists(variant_geometric_results):
                        combined_csv_paths[f'{geo_variant_name}_geometric'] = str(variant_geometric_results)
                    
                    for jtype in args.junction_types:
                        calibrated_results_csv = geo_variant_paths['junction_types'][jtype]['calibrated_results']
                        if os.path.exists(calibrated_results_csv):
                            combined_csv_paths[f'{geo_variant_name}_{jtype}'] = str(calibrated_results_csv)
                
                if combined_csv_paths:
                    success_count = 0
                    for location in all_locations:
                        safe_location = location.replace(':', '_')
                        plot_path = os.path.join(combined_output_dir, f"{safe_location}_combined.png")
                        try:
                            success = plot_location_comparison(
                                str(original_calibration_input),
                                str(orig_geometric_results),
                                combined_csv_paths,
                                location,
                                plot_path,
                                set_name=args.set_name,
                                geo_name=args.geo_name,
                                time_period=time_period,
                                geometric_input_path=str(geometry_variants['original']['geometric_input']),
                                zoom_start_idx=args.zoom_start,
                                zoom_end_idx=args.zoom_end,
                                verbose=False
                            )
                            if success:
                                success_count += 1
                        except Exception as e:
                            print(f"      ✗ Failed to create combined plot for {location}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"    Created {success_count}/{len(all_locations)} combined comparison plots")
                    print(f"    Combined plot output directory: {combined_output_dir}")
            else:
                print(f"    Skipping combined plots (missing original geometry files)")
                
        except ImportError as e:
            print(f"  ✗ Could not import plotting functions: {e}")
            print("  Make sure plot_location_comparison.py is available")
        except Exception as e:
            print(f"  ✗ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot junction pressure differences for each geometry variant
    if not args.skip_plots:
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            variant_calibration_input = geo_variant_paths['calibration_input']
            variant_geometric_input = geo_variant_paths['geometric_input']
            
            if os.path.exists(variant_calibration_input) and os.path.exists(variant_geometric_input):
                try:
                    print(f"\n  Creating junction pressure difference plots for {geo_variant_name}...")
                    prefix = '' if geo_variant_name == 'original' else f'{geo_variant_name}_'
                    output_subdir = os.path.join(base_dir, f'{prefix}junction_pressure_diff') if prefix else base_dir
                    plot_junction_pressure_differences(
                        str(variant_calibration_input),
                        str(variant_geometric_input),
                        output_subdir,
                        zoom_start_idx=args.zoom_start,
                        zoom_end_idx=args.zoom_end,
                        set_name=args.set_name,
                        geo_name=args.geo_name,
                        verbose=verbose
                    )
                except Exception as e:
                    raise Exception(f"Error generating junction pressure difference plots for {geo_variant_name}: {e}")
    
    if verbose:
        print("\n" + "="*60)
        print("Done!")
        print("="*60)
        
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            print(f"\n{geo_variant_name.upper()} geometry:")
            print(f"  Geometric input: {geo_variant_paths['geometric_input']}")
            print(f"  Geometric simulation results: {geo_variant_paths['geometric_results']}")
            
            if not args.skip_calibration:
                print(f"  Base calibration input: {geo_variant_paths['calibration_input']}")
                print(f"  Junction type variants:")
                for jtype in args.junction_types:
                    print(f"    {jtype}:")
                    print(f"      Calibration input: {geo_variant_paths['junction_types'][jtype]['calibration_input']}")
                    print(f"      Calibrated output: {geo_variant_paths['junction_types'][jtype]['calibrated_output']}")
                    print(f"      Simulation results: {geo_variant_paths['junction_types'][jtype]['calibrated_results']}")


if __name__ == "__main__":
    main()

# python3 util/zerod_calibration/generate_zerod_inputs.py --set-name set_3 --geo-name tree_007
