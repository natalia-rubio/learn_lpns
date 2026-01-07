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
    
    # Find inlet and outlet caps (handles both svVascularize and SimVascular formats)
    print("Identifying inlet and outlet caps...")
    inlet_cap, outlet_caps, mesh_surfaces_dir = find_inlet_outlet_caps(geo_dir)
    
    # Get geometry name from directory
    geo_name = os.path.basename(geo_dir.rstrip('/'))
    
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
    
    # Extract outlet resistances from XML file
    print(f"\nExtracting outlet resistances from XML file...")
    print(f"  Looking for outlet caps: {outlet_caps}")
    # Create mapping from base names (without .vtp) to full names (with .vtp)
    # XML BC names don't include .vtp extension
    outlet_caps_base = {}
    for cap in outlet_caps:
        base_name = cap.replace('.vtp', '') if cap.endswith('.vtp') else cap
        outlet_caps_base[base_name] = cap
    print(f"  Outlet caps base names (for XML matching): {list(outlet_caps_base.keys())}")
    
    outlet_resistances = {}
    xml_files = [
        os.path.join(geo_dir, 'fluid_simulation_0-0.xml'),
        os.path.join(geo_dir, 'fluid_simulation.xml'),
    ]
    xml_path = None
    for path in xml_files:
        print(f"  Checking for XML file: {path}")
        if os.path.exists(path):
            xml_path = path
            print(f"  Found XML file: {xml_path}")
            break
    
    if xml_path:
        try:
            print(f"  Parsing XML file...")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            add_equation = root.find('Add_equation')
            if add_equation is None:
                print(f"  Warning: No 'Add_equation' element found in XML")
            else:
                print(f"  Found 'Add_equation' element, searching for boundary conditions...")
                all_bcs = add_equation.findall('Add_BC')
                print(f"  Found {len(all_bcs)} boundary condition(s) in XML")
                
                for bc in all_bcs:
                    bc_name = bc.get('name')
                    bc_type = bc.find('Type')
                    value_elem = bc.find('Value')
                    
                    print(f"    BC name: {bc_name}, type: {bc_type.text if bc_type is not None else 'None'}, value: {value_elem.text if value_elem is not None else 'None'}")
                    
                    # Outlet BCs are Neumann type with a Value (resistance)
                    # XML BC names don't have .vtp extension, so compare against base names
                    if bc_name:
                        print(f"      Checking if '{bc_name}' is in outlet_caps_base: {bc_name in outlet_caps_base}")
                        if bc_name in outlet_caps_base:
                            print(f"      '{bc_name}' is an outlet cap")
                            if bc_type is not None:
                                print(f"      BC type is: '{bc_type.text}'")
                                if bc_type.text == 'Neumann':
                                    print(f"      BC type matches 'Neumann'")
                                    if value_elem is not None:
                                        print(f"      Value element found: '{value_elem.text}'")
                                        try:
                                            resistance = float(value_elem.text)
                                            # Store using base name (without .vtp) to match XML BC names
                                            outlet_resistances[bc_name] = resistance
                                            print(f"      ✓ Found resistance for {bc_name}: {resistance}")
                                        except (ValueError, TypeError) as e:
                                            print(f"      ✗ Could not parse resistance value for {bc_name}: {e}, using default 1.0")
                                    else:
                                        print(f"      ✗ No Value element found for {bc_name}")
                                else:
                                    print(f"      ✗ BC type is not 'Neumann' (it's '{bc_type.text}')")
                            else:
                                print(f"      ✗ No Type element found for {bc_name}")
                        else:
                            print(f"      '{bc_name}' is not in outlet_caps list")
                    else:
                        print(f"      ✗ BC has no name attribute")
        except Exception as e:
            print(f"  ✗ Error parsing XML for outlet resistances: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ✗ No XML file found. Tried: {xml_files}")
    
    print(f"  Final outlet resistances dictionary: {outlet_resistances}")
    
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
        script_content += f"bcs.add_resistance(face_name='{outlet_cap_base}', resistance={1+0*resistance})\n"
    
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
    else:
        time_step_size = inp["boundary_conditions"][0]["bc_values"]["t"][1] - inp["boundary_conditions"][0]["bc_values"]["t"][0]
    
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
        "tolerance_increment": 1e-10,
        "maximum_iterations": 100,
        "calibrate_stenosis_coefficient":False,
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


def fit_outlet_resistances_from_3d(geometric_input_path, observations):
    """
    Fit outlet boundary condition resistances and distal pressures from 3D solution observations.
    Fits linear relationship: P = R*Q + Pd using least squares regression.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON (will be updated)
        observations: Dictionary with observation data (y, dy) containing outlet pressure and flow
    
    Returns:
        Dictionary mapping outlet BC names to fitted (R, Pd) tuples
    """
    print(f"\nFitting outlet resistances and distal pressures from 3D solution...")
    
    # Load geometric input
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Extract outlet resistances and distal pressures
    outlet_params = {}
    
    # Find all outlet BCs in geometric input
    outlet_bcs = {}
    for bc in inp.get('boundary_conditions', []):
        if bc.get('bc_type') == 'RESISTANCE':
            bc_name = bc.get('bc_name')
            if bc_name:
                outlet_bcs[bc_name] = bc
    
    # Find vessels with outlet BCs
    vessels = inp.get('vessels', [])
    vessel_to_bc = {}
    for vessel in vessels:
        if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
            bc_name = vessel['boundary_conditions']['outlet']
            vessel_name = vessel['vessel_name']
            vessel_to_bc[vessel_name] = bc_name
    
    print(f"  Found {len(outlet_bcs)} outlet boundary conditions")
    
    # Extract pressure and flow for each outlet
    obs_y = observations.get('y', {})
    
    for vessel_name, bc_name in vessel_to_bc.items():
        if bc_name not in outlet_bcs:
            print(f"  Warning: BC {bc_name} not found in boundary_conditions")
            continue
        
        # Look for pressure and flow observations
        # Pattern: "pressure:{vessel_name}:{bc_name}" and "flow:{vessel_name}:{bc_name}"
        pressure_key = f"pressure:{vessel_name}:{bc_name}"
        flow_key = f"flow:{vessel_name}:{bc_name}"
        
        if pressure_key not in obs_y:
            print(f"  Warning: No pressure observation found for {vessel_name}:{bc_name}")
            continue
        
        if flow_key not in obs_y:
            print(f"  Warning: No flow observation found for {vessel_name}:{bc_name}")
            continue
        
        pressures = np.array(obs_y[pressure_key])
        flows = np.array(obs_y[flow_key])
        
        # Convert to numpy arrays if needed
        if isinstance(pressures, list):
            pressures = np.array(pressures)
        if isinstance(flows, list):
            flows = np.array(flows)
        
        # Filter out invalid values (inf, nan)
        valid_mask = np.isfinite(pressures) & np.isfinite(flows)
        
        if not np.any(valid_mask):
            print(f"  Warning: No valid data points for {vessel_name}:{bc_name}")
            print(f"    Pressure range: [{np.min(pressures):.2f}, {np.max(pressures):.2f}]")
            print(f"    Flow range: [{np.min(flows):.2f}, {np.max(flows):.2f}]")
            continue
        
        valid_pressures = pressures[valid_mask]
        valid_flows = flows[valid_mask]
        
        if len(valid_pressures) < 2:
            print(f"  Warning: Insufficient data points for {vessel_name}:{bc_name} (need at least 2)")
            continue
        
        # Fit linear relationship: P = R*Q + Pd
        # Using least squares: [R, Pd] = (Q^T * Q)^(-1) * Q^T * P
        # Where Q is the design matrix: [flows, ones]
        try:
            # Create design matrix: [flows, ones] for [R, Pd]
            A = np.vstack([valid_flows, np.ones(len(valid_flows))]).T
            b = valid_pressures
            
            # Solve least squares: [R, Pd] = (A^T * A)^(-1) * A^T * b
            params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            fitted_resistance = float(params[0])
            fitted_pd = float(params[1])
            
            # Check if fit is reasonable
            if not np.isfinite(fitted_resistance) or not np.isfinite(fitted_pd):
                print(f"  Warning: Invalid fit parameters for {vessel_name}:{bc_name}")
                continue
            
            # Check if resistance is positive (should be for physical validity)
            if fitted_resistance < 0:
                print(f"  Warning: Negative resistance fitted for {vessel_name}:{bc_name} ({fitted_resistance:.4f}), using absolute value")
                fitted_resistance = abs(fitted_resistance)
            
            outlet_params[bc_name] = (fitted_resistance, fitted_pd)
            
            # Calculate R-squared for quality assessment
            predicted_pressures = fitted_resistance * valid_flows + fitted_pd
            ss_res = np.sum((valid_pressures - predicted_pressures) ** 2)
            ss_tot = np.sum((valid_pressures - np.mean(valid_pressures)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"  {bc_name} ({vessel_name}):")
            print(f"    Pressure range: [{np.min(valid_pressures):.2f}, {np.max(valid_pressures):.2f}] dynes/cm²")
            print(f"    Flow range: [{np.min(valid_flows):.2f}, {np.max(valid_flows):.2f}] cm³/s")
            print(f"    Fitted R: {fitted_resistance:.4f}")
            print(f"    Fitted Pd: {fitted_pd:.4f}")
            print(f"    R²: {r_squared:.4f}")
            
        except np.linalg.LinAlgError as e:
            print(f"  Warning: Linear regression failed for {vessel_name}:{bc_name}: {e}")
            continue
    
    # Update geometric input with fitted resistances and distal pressures
    print(f"\n  Updating geometric input with fitted parameters...")
    for bc_name, (resistance, pd) in outlet_params.items():
        if bc_name in outlet_bcs:
            old_resistance = outlet_bcs[bc_name]['bc_values'].get('R', 1.0)
            old_pd = outlet_bcs[bc_name]['bc_values'].get('Pd', 0.0)
            outlet_bcs[bc_name]['bc_values']['R'] = resistance
            outlet_bcs[bc_name]['bc_values']['Pd'] = pd
            print(f"    {bc_name}: R {old_resistance:.4f} -> {resistance:.4f}, Pd {old_pd:.4f} -> {pd:.4f}")
    
    # Save updated geometric input
    with open(geometric_input_path, 'w') as f:
        json.dump(inp, f, indent=4)
    
    print(f"  ✓ Updated geometric input saved to: {geometric_input_path}")
    
    return outlet_params


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
    
    # Write calibrated output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cali, f, indent=4)
    
    print(f"Calibrated output saved to: {output_path}")
    return cali


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


def read_zerod_csv(csv_path):
    """
    Read 0D simulation results from CSV.
    Handles both 'location' and 'name' as the vessel identifier column.
    
    Returns:
        results: Dictionary {location: {time: {field: value}}}
        times: Sorted list of time values
    """
    results = {}
    times = set()
    
    if not os.path.exists(csv_path):
        return results, sorted(times)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Check which column name is used for vessel identifier
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return results, sorted(times)
        
        vessel_col = None
        if 'location' in fieldnames:
            vessel_col = 'location'
        elif 'name' in fieldnames:
            vessel_col = 'name'
        else:
            return results, sorted(times)
        
        for row in reader:
            location = row[vessel_col]
            time = float(row['time'])
            times.add(time)
            
            if location not in results:
                results[location] = {}
            if time not in results[location]:
                results[location][time] = {}
            
            # Extract all numeric fields
            for key, value in row.items():
                if key not in [vessel_col, 'time']:
                    try:
                        results[location][time][key] = float(value)
                    except (ValueError, TypeError):
                        continue
    
    return results, sorted(times)


def calculate_mse_between_3d_and_0d(calibration_input_path, csv_results_dict, geometric_input_path=None, zoom_start_idx=None, zoom_end_idx=None, output_csv_path=None, verbose=False):
    """
    Calculate and print Mean Squared Error (MSE) between 3D observations and 0D solutions.
    Optionally saves results to a CSV file.
    
    Args:
        calibration_input_path: Path to calibration input JSON (contains 3D observations)
        csv_results_dict: Dictionary mapping modality names to CSV file paths
                         e.g., {'geometric': 'path/to/geometric_results.csv',
                                'NORMAL_JUNCTION': 'path/to/normal_junction_results.csv', ...}
        geometric_input_path: Optional path to geometric input JSON (for understanding structure)
        zoom_start_idx: Start index for zoom window (default: 599)
        zoom_end_idx: End index for zoom window (default: 699)
        output_csv_path: Optional path to save MSE results CSV (default: auto-generate based on calibration_input_path)
        verbose: If True, print detailed comparison table. If False, only print summary.
    
    Returns:
        Dictionary mapping modality names to MSE results
    """
    print("\n" + "="*80)
    print("Calculating Mean Squared Error (MSE) between 3D and 0D solutions")
    print("="*80)
    
    # Read 3D observations from calibration input
    if not os.path.exists(calibration_input_path):
        print(f"  ✗ Calibration input not found: {calibration_input_path}")
        return {}
    
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Get 3D observations (use full observations if available, otherwise use y)
    if '_full_observations' in calib_data and 'y' in calib_data['_full_observations']:
        obs_3d = calib_data['_full_observations']['y']
    elif 'y' in calib_data:
        obs_3d = calib_data['y']
    else:
        print("  ✗ No 3D observations found in calibration input")
        return {}
    
    print(f"  Found {len(obs_3d)} 3D observation series")
    
    # Read geometric input to understand vessel/junction structure
    vessels = []
    junctions = []
    if geometric_input_path and os.path.exists(geometric_input_path):
        with open(geometric_input_path, 'r') as f:
            geo_input = json.load(f)
        vessels = geo_input.get('vessels', [])
        junctions = geo_input.get('junctions', [])
    
    # Calculate MSE for each modality
    mse_results = {}
    
    for modality_name, csv_path in csv_results_dict.items():
        if not csv_path or not os.path.exists(csv_path):
            print(f"\n  {modality_name}: ✗ CSV file not found: {csv_path}")
            continue
        
        print(f"\n  {modality_name}:")
        print(f"    Reading 0D results from: {csv_path}")
        
        # Read 0D CSV results
        results_0d, times_0d = read_zerod_csv(csv_path)
        
        if not results_0d or not times_0d:
            print(f"    ✗ No data found in CSV file")
            continue
        
        print(f"    Found {len(results_0d)} vessels, {len(times_0d)} time points")
        
        # Determine zoom window - matching the shaded region in plots
        # This corresponds to the zoom window used in plot_inlet_comparison.py and plot_outlet_comparison.py
        
        # Find the maximum length of 3D observations to validate zoom window
        max_3d_length = 0
        for obs_values_3d in obs_3d.values():
            if isinstance(obs_values_3d, list):
                max_3d_length = max(max_3d_length, len(obs_values_3d))
            elif hasattr(obs_values_3d, '__len__'):
                max_3d_length = max(max_3d_length, len(obs_values_3d))
        
        # Validate and adjust zoom window if needed
        if zoom_start_idx >= max_3d_length:
            zoom_start_idx = max(0, max_3d_length - 105)
            zoom_end_idx = max_3d_length
        elif zoom_end_idx > max_3d_length:
            zoom_end_idx = max_3d_length
        
        # Ensure valid range
        if zoom_start_idx >= zoom_end_idx:
            zoom_start_idx = max(0, max_3d_length - 105)
            zoom_end_idx = max_3d_length
        
        num_zoom_timesteps = zoom_end_idx - zoom_start_idx
        print(f"    Using zoom window: timesteps {zoom_start_idx} to {zoom_end_idx-1} ({num_zoom_timesteps} timesteps)")
        
        # Calculate MSE for each observation
        modality_mse = {}
        total_mse_pressure = []
        total_mse_flow = []
        
        for obs_key, obs_values_3d in obs_3d.items():
            if not isinstance(obs_values_3d, list):
                obs_values_3d = obs_values_3d.tolist() if hasattr(obs_values_3d, 'tolist') else list(obs_values_3d)
            
            # Apply zoom window filter: use timesteps in range [zoom_start_idx:zoom_end_idx] (matching shaded region in plots)
            if len(obs_values_3d) > zoom_start_idx:
                obs_values_3d = obs_values_3d[zoom_start_idx:zoom_end_idx]
            else:
                # If 3D data is shorter than zoom window, skip this observation
                continue
            
            # Parse observation key: "pressure:INFLOW:branch0_seg0" or "flow:branch0_seg0:J0"
            parts = obs_key.split(':')
            if len(parts) != 3:
                continue
            
            obs_type = parts[0]  # 'pressure' or 'flow'
            part1 = parts[1]     # e.g., 'INFLOW', 'branch0_seg0', or 'J0'
            part2 = parts[2]     # e.g., 'branch0_seg0' or 'J0'
            
            # Determine vessel name and field (pressure_in/out, flow_in/out)
            vessel_name = None
            field_name = None
            
            # Handle inlet observations: "pressure:INFLOW:branch0_seg0" or "flow:INFLOW:branch0_seg0"
            if part1 == 'INFLOW':
                vessel_name = part2
                field_name = f"{obs_type}_in"
            # Handle outlet observations at junctions: "pressure:branch0_seg0:J0" or "flow:branch0_seg0:J0"
            elif part1.startswith('branch') and part2.startswith('J'):
                vessel_name = part1
                field_name = f"{obs_type}_out"
            # Handle inlet observations at junctions: "pressure:J0:branch1_seg0" or "flow:J0:branch1_seg0"
            elif part1.startswith('J') and part2.startswith('branch'):
                vessel_name = part2
                field_name = f"{obs_type}_in"
            else:
                # Try to find vessel name in parts
                for p in [part1, part2]:
                    if p.startswith('branch'):
                        vessel_name = p
                        # Guess field based on position
                        if part1 == vessel_name:
                            field_name = f"{obs_type}_out"
                        else:
                            field_name = f"{obs_type}_in"
                        break
            
            if not vessel_name or not field_name:
                continue
            
            # Extract 0D data for this vessel
            if vessel_name not in results_0d:
                continue
            
            # Extract 0D values at available time points
            times_0d_valid = []
            obs_values_0d_raw = []
            for time in sorted(times_0d):
                if time in results_0d[vessel_name] and field_name in results_0d[vessel_name][time]:
                    times_0d_valid.append(time)
                    obs_values_0d_raw.append(results_0d[vessel_name][time][field_name])
            
            if len(times_0d_valid) < 2:
                continue
            
            # Apply zoom window filter to 0D data: use same index range [zoom_start_idx:zoom_end_idx]
            if len(obs_values_0d_raw) > zoom_start_idx:
                obs_values_0d_zoomed = obs_values_0d_raw[zoom_start_idx:zoom_end_idx]
                times_0d_zoomed = times_0d_valid[zoom_start_idx:zoom_end_idx]
            else:
                # If 0D data is shorter than zoom window, skip this observation
                continue
            
            if len(obs_values_0d_zoomed) == 0:
                continue
            
            # Interpolate 0D data to match 3D observation time points (now both are in zoom window)
            if len(obs_values_3d) != len(obs_values_0d_zoomed):
                # Interpolate 0D data to match 3D observation count
                # Use linear interpolation
                times_0d_array = np.array(times_0d_zoomed)
                obs_values_0d_array = np.array(obs_values_0d_zoomed)
                
                # Create time points matching 3D observation count
                time_min = times_0d_array[0]
                time_max = times_0d_array[-1]
                times_3d_interp = np.linspace(time_min, time_max, len(obs_values_3d))
                
                # Interpolate 0D values
                if HAS_SCIPY_INTERP:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(times_0d_array, obs_values_0d_array, kind='linear', 
                                          bounds_error=False, fill_value=(obs_values_0d_array[0], obs_values_0d_array[-1]))
                    obs_values_0d = interp_func(times_3d_interp)
                else:
                    # Use numpy interpolation
                    obs_values_0d = np.interp(times_3d_interp, times_0d_array, obs_values_0d_array)
            else:
                # Same length, use directly (both are already in zoom window)
                obs_values_0d = np.array(obs_values_0d_zoomed)
            
            # Convert to numpy arrays
            obs_values_0d = np.array(obs_values_0d)
            obs_values_3d = np.array(obs_values_3d)
            
            # Ensure 0D data matches the length of filtered 3D data (zoom window)
            # If 0D data is longer, take the last portion matching 3D length
            if len(obs_values_0d) > len(obs_values_3d):
                obs_values_0d = obs_values_0d[-len(obs_values_3d):]
            elif len(obs_values_0d) < len(obs_values_3d):
                # If 0D data is shorter, pad or truncate 3D to match
                obs_values_3d = obs_values_3d[:len(obs_values_0d)]
            
            # Remove NaN and inf values
            valid_mask = np.isfinite(obs_values_0d) & np.isfinite(obs_values_3d)
            if not np.any(valid_mask):
                continue
            
            obs_values_0d_clean = obs_values_0d[valid_mask]
            obs_values_3d_clean = obs_values_3d[valid_mask]
            
            # Ensure same length (take minimum)
            min_len = min(len(obs_values_0d_clean), len(obs_values_3d_clean))
            if min_len == 0:
                continue
            
            obs_values_0d_clean = obs_values_0d_clean[:min_len]
            obs_values_3d_clean = obs_values_3d_clean[:min_len]
            
            # Calculate MSE
            diff = obs_values_0d_clean - obs_values_3d_clean
            
            mse = np.mean(diff**2)
            mae = np.mean(np.abs(diff))
            
            # Save comparison plot for debugging
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                
                # Create output directory for debug plots
                base_dir = os.path.dirname(calibration_input_path)
                debug_plots_dir = os.path.join(base_dir, 'mse_debug_plots')
                os.makedirs(debug_plots_dir, exist_ok=True)
                
                # Create safe filename from observation key
                safe_obs_key = obs_key.replace(':', '_').replace('/', '_')
                plot_filename = f"{modality_name}_{safe_obs_key}_comparison.png"
                plot_path = os.path.join(debug_plots_dir, plot_filename)
                
                # Create comparison plot
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                # Plot 0D vs 3D as scatter or line plot
                if len(obs_values_0d_clean) > 50:
                    # For many points, use scatter
                    ax.scatter(obs_values_3d_clean, obs_values_0d_clean, alpha=0.5, s=10, label='Data points')
                else:
                    # For fewer points, use line plot
                    ax.plot(obs_values_3d_clean, obs_values_0d_clean, 'o-', alpha=0.7, markersize=4, label='Data points')
                
                # Add diagonal line (perfect match)
                min_val = min(np.min(obs_values_3d_clean), np.min(obs_values_0d_clean))
                max_val = max(np.max(obs_values_3d_clean), np.max(obs_values_0d_clean))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match', alpha=0.5)
                
                ax.set_xlabel('3D Observations', fontsize=12)
                ax.set_ylabel('0D Results', fontsize=12)
                # Format MSE and MAE in engineering notation
                ax.set_title(f'{modality_name}: {obs_key}\nMSE={mse:.3E}, MAE={mae:.3E}, n={min_len}', fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                pass  # Silently fail if plotting is not available
            
            modality_mse[obs_key] = {
                'mse': mse,
                'type': obs_type,
                'vessel': vessel_name,
                'n_points': min_len
            }
            
            if obs_type == 'pressure':
                total_mse_pressure.append(mse)
            else:
                total_mse_flow.append(mse)
        
        # Store results
        mse_results[modality_name] = {
            'individual': modality_mse,
            'mean_pressure_mse': np.mean(total_mse_pressure) if total_mse_pressure else np.nan,
            'mean_flow_mse': np.mean(total_mse_flow) if total_mse_flow else np.nan,
            'overall_mse': np.mean(total_mse_pressure + total_mse_flow) if (total_mse_pressure or total_mse_flow) else np.nan
        }
        
        # Print summary (only in verbose mode)
        if verbose:
            print(f"    Calculated MSE for {len(modality_mse)} observation series")
            if total_mse_pressure:
                print(f"    Mean Pressure MSE: {np.mean(total_mse_pressure):.3E}")
            if total_mse_flow:
                print(f"    Mean Flow MSE: {np.mean(total_mse_flow):.3E}")
            if total_mse_pressure or total_mse_flow:
                print(f"    Overall MSE: {np.mean(total_mse_pressure + total_mse_flow):.3E}")
    
    # Get all observation keys and modalities for summary
    all_obs_keys = set()
    for modality_results in mse_results.values():
        all_obs_keys.update(modality_results['individual'].keys())
    
    if not all_obs_keys:
        if verbose:
            print("  No observations found for comparison")
        return mse_results
    
    modalities = list(csv_results_dict.keys())
    
    # Print detailed comparison table (only in verbose mode)
    if verbose:
        print("\n" + "="*80)
        print("Detailed MSE Comparison")
        print("="*80)
        
        # Print header
        print(f"\n{'Observation':<40} {'Type':<10} ", end="")
        for mod in modalities:
            if mod in mse_results:
                print(f"{mod:<15} ", end="")
        print()
        print("-" * (50 + 15 * len([m for m in modalities if m in mse_results])))
        
        # Print each observation
        for obs_key in sorted(all_obs_keys):
            # Truncate long keys for display
            display_key = obs_key[:38] + ".." if len(obs_key) > 40 else obs_key
            
            # Get type from first available result
            obs_type = "unknown"
            for modality_results in mse_results.values():
                if obs_key in modality_results['individual']:
                    obs_type = modality_results['individual'][obs_key]['type']
                    break
            
            print(f"{display_key:<40} {obs_type:<10} ", end="")
            for mod in modalities:
                if mod in mse_results and obs_key in mse_results[mod]['individual']:
                    mse_val = mse_results[mod]['individual'][obs_key]['mse']
                    print(f"{mse_val:>13.3E}  ", end="")
                else:
                    print(f"{'N/A':>13}  ", end="")
            print()
        
        print("\n" + "-" * (50 + 15 * len([m for m in modalities if m in mse_results])))
    
    # Always print summary statistics with column headers
    # Print header row
    print(f"{'':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            print(f"{mod:<15} ", end="")
    print()
    print("-" * (50 + 15 * len([m for m in modalities if m in mse_results])))
    
    # Print summary rows
    print(f"{'SUMMARY':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            overall = mse_results[mod]['overall_mse']
            if not np.isnan(overall):
                print(f"{overall:>13.3E}  ", end="")
            else:
                print(f"{'N/A':>13}  ", end="")
    print()
    
    print(f"{'Mean Pressure MSE':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            mse_p = mse_results[mod]['mean_pressure_mse']
            if not np.isnan(mse_p):
                print(f"{mse_p:>13.3E}  ", end="")
            else:
                print(f"{'N/A':>13}  ", end="")
    print()
    
    print(f"{'Mean Flow MSE':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            mse_f = mse_results[mod]['mean_flow_mse']
            if not np.isnan(mse_f):
                print(f"{mse_f:>13.3E}  ", end="")
            else:
                print(f"{'N/A':>13}  ", end="")
    print()
    
    # Save results to CSV file
    if output_csv_path is None:
        # Auto-generate CSV path based on calibration input path
        base_dir = os.path.dirname(calibration_input_path)
        base_name = os.path.basename(calibration_input_path).replace('.json', '')
        output_csv_path = os.path.join(base_dir, f'{base_name}_mse_comparison.csv')
    
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Write CSV with two sections: summary and detailed
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['MSE Comparison Results'])
            writer.writerow(['Zoom Window', f'{zoom_start_idx} to {zoom_end_idx-1}'])
            writer.writerow([])
            
            # Write summary statistics
            writer.writerow(['Summary Statistics'])
            writer.writerow(['Metric'] + modalities)
            
            # Overall MSE
            row = ['Overall MSE']
            for mod in modalities:
                if mod in mse_results:
                    overall = mse_results[mod]['overall_mse']
                    if not np.isnan(overall):
                        row.append(f'{overall:.3E}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)
            
            # Mean Pressure MSE
            row = ['Mean Pressure MSE']
            for mod in modalities:
                if mod in mse_results:
                    mse_p = mse_results[mod]['mean_pressure_mse']
                    if not np.isnan(mse_p):
                        row.append(f'{mse_p:.3E}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)
            
            # Mean Flow MSE
            row = ['Mean Flow MSE']
            for mod in modalities:
                if mod in mse_results:
                    mse_f = mse_results[mod]['mean_flow_mse']
                    if not np.isnan(mse_f):
                        row.append(f'{mse_f:.3E}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)
            
            writer.writerow([])
            writer.writerow(['Detailed Results'])
            writer.writerow(['Observation', 'Type', 'Vessel'] + modalities)
            
            # Write individual observation results
            for obs_key in sorted(all_obs_keys):
                # Get type and vessel from first available result
                obs_type = "unknown"
                vessel_name = "unknown"
                for modality_results in mse_results.values():
                    if obs_key in modality_results['individual']:
                        obs_type = modality_results['individual'][obs_key]['type']
                        vessel_name = modality_results['individual'][obs_key].get('vessel', 'unknown')
                        break
                
                row = [obs_key, obs_type, vessel_name]
                for mod in modalities:
                    if mod in mse_results and obs_key in mse_results[mod]['individual']:
                        mse_val = mse_results[mod]['individual'][obs_key]['mse']
                        row.append(f'{mse_val:.3E}')
                    else:
                        row.append('N/A')
                writer.writerow(row)
        
        print(f"\n  ✓ MSE results saved to: {output_csv_path}")
        
    except Exception as e:
        print(f"  ✗ Warning: Could not save MSE results to CSV: {e}")
        import traceback
        traceback.print_exc()
    
    return mse_results


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
    parser.add_argument('--timestep-scale-factor', type=float, default=0.25,
                       help='Scaling factor for 0D simulation timestep relative to 1D/3D timestep (default: 0.25, i.e., quarter the timestep)')
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
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed MSE comparison table')
    
    args = parser.parse_args()
    verbose = args.verbose
    # Construct paths
    base_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    geometric_input_path = os.path.join(base_dir, 'geometric_input.json')
    geometric_results_csv = os.path.join(base_dir, 'geometric_results.csv')
    calibration_input_path = os.path.join(base_dir, 'calibration_input.json')
    calibrated_output_path = os.path.join(base_dir, 'calibrated_output.json')
    
    # Paths for each junction type variant
    junction_type_paths = {}
    for jtype in args.junction_types:
        junction_type_paths[jtype] = {
            'calibration_input': os.path.join(base_dir, f'calibration_input_{jtype}.json'),
            'calibrated_output': os.path.join(base_dir, f'calibrated_output_{jtype}.json'),
            'calibrated_results': os.path.join(base_dir, f'calibrated_results_{jtype}.csv')
        }
    
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
        
        if centerline_path is None:
            raise FileNotFoundError(f"Centerline file not found. Tried: {centerline_paths}")
    
    # Step 1: Create geometric 0D input using SimVascular ROM workflow
    print("\n" + "="*60)
    print("Step 1: Creating geometric 0D input file using SimVascular ROM")
    print("="*60)
    
    # Get geometry directory
    geo_dir = os.path.join('data', 'threeD', args.set_name, args.geo_name)
    if not os.path.exists(geo_dir):
        raise FileNotFoundError(f"Geometry directory not found: {geo_dir}")
    
    zerod_input, vessel_bc_map = create_geometric_zerod_input_rom(
        geo_dir, centerline_path, geometric_input_path,
        simvascular_path=args.simvascular_path,
        dt=args.dt,
        num_time_steps=args.num_time_steps,
        num_cardiac_cycles=args.num_cardiac_cycles
    )
    
    # Step 2: Extract observations and create calibration inputs for each junction type
    if not args.skip_calibration:
        print("\n" + "="*60)
        print("Step 2: Creating calibration input files for each junction type")
        print("="*60)
        
        if args.one_d_soln:
            print(f"  Start index for observations: {args.start_idx}")
            observations = extract_observations_from_1d(args.one_d_soln, geometric_input_path, geo_dir=geo_dir, start_idx=args.start_idx)
        elif args.three_d_soln_dir:
            raise NotImplementedError("3D solution extraction not yet implemented")
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
            
            # Create base calibration input
            print(f"\n  Creating base calibration input...")
            calibration_input_path = os.path.join(base_dir, 'calibration_input.json')
            
            # Create calibration input with time computation from 1D solution
            create_calibration_input(geometric_input_path, observations, calibration_input_path,
                                   centerline_soln_path=soln_path, geo_dir=geo_dir)
            
            print(f"    ✓ Base calibration input saved to: {calibration_input_path}")
            
            # Fit outlet resistances from 3D solution
            fitted_resistances = fit_outlet_resistances_from_3d(geometric_input_path, observations)
            
            # Update base calibration input with fitted outlet BCs
            print(f"\n  Updating base calibration input with fitted outlet BCs...")
            update_outlet_bcs_in_file(calibration_input_path, fitted_resistances, "base calibration input")
            
            # Update geometric input with BC from calibration input
            update_geometric_input_with_calibration_bc(geometric_input_path, calibration_input_path)
            
            # Create calibration input variants for each junction type
            print(f"\n  Creating calibration input variants for each junction type...")
            with open(calibration_input_path, 'r') as f:
                base_calibration_config = json.load(f)
            
            for jtype in args.junction_types:
                print(f"    Creating {jtype} calibration input...")
                jtype_input_path = junction_type_paths[jtype]['calibration_input']
                
                # Apply junction type modification to calibration input
                jtype_config = modify_junction_types(base_calibration_config, jtype)
                
                with open(jtype_input_path, 'w') as f:
                    json.dump(jtype_config, f, indent=4)
                
                # Update with fitted outlet BCs (after writing, to ensure they're in the file)
                update_outlet_bcs_in_file(jtype_input_path, fitted_resistances, f"{jtype} calibration input")
                
                print(f"      ✓ Saved to: {jtype_input_path}")
            
            # Step 3: Run calibration for each junction type
            print("\n" + "="*60)
            print("Step 3: Running calibration for each junction type")
            print("="*60)
            
            
            # Run calibration for each junction type variant
            for jtype in args.junction_types:
                print(f"\n  Calibrating {jtype}...")
                jtype_input_path = junction_type_paths[jtype]['calibration_input']
                jtype_output_path = junction_type_paths[jtype]['calibrated_output']
                
                try:
                    calibrated_config = run_calibration(jtype_input_path, jtype_output_path)
                    print(f"    ✓ Calibration completed for {jtype}")
                    # Replace inlet BC with original observed BC
                    replace_inlet_bc_in_calibrated_output(jtype_output_path, jtype_input_path)
                    # Update outlet BCs with fitted values
                    update_outlet_bcs_in_file(jtype_output_path, fitted_resistances, f"{jtype} calibrated output")
                except Exception as e:
                    print(f"    ✗ Calibration failed for {jtype}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Step 4: Run forward simulations
            print("\n" + "="*60)
            print("Step 4: Running forward simulations")
            print("="*60)
            
            # Run simulation with geometric input
            print("\n  Running simulation with geometric input...")
            try:
                run_forward_simulation(geometric_input_path, geometric_results_csv)
                print(f"    ✓ Geometric simulation completed successfully")
            except Exception as e:
                print(f"    ✗ Geometric simulation failed: {e}")
            
            # Run simulation with each calibrated input
            for jtype in args.junction_types:
                print(f"\n  Running simulation with calibrated {jtype} input...")
                calibrated_output_path = junction_type_paths[jtype]['calibrated_output']
                calibrated_results_csv = junction_type_paths[jtype]['calibrated_results']
                
                try:
                    run_forward_simulation(calibrated_output_path, calibrated_results_csv)
                    print(f"    ✓ Calibrated {jtype} simulation completed successfully")
                except Exception as e:
                    print(f"    ✗ Calibrated {jtype} simulation failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Step 4: Calculate and print MSE between 3D and 0D solutions
    if not args.skip_calibration:
        print("\n" + "="*60)
        print("Step 4: Calculating MSE between 3D and 0D solutions")
        print("="*60)
        
        # Build dictionary of CSV results for all modalities
        csv_results_dict = {}
        
        # Add geometric results
        if os.path.exists(geometric_results_csv):
            csv_results_dict['geometric'] = geometric_results_csv
        
        # Add calibrated results for each junction type
        for jtype in args.junction_types:
            calibrated_results_csv = junction_type_paths[jtype]['calibrated_results']
            if os.path.exists(calibrated_results_csv):
                csv_results_dict[jtype] = str(calibrated_results_csv)
        
        if csv_results_dict and os.path.exists(calibration_input_path):
            try:
                # Generate CSV output path
                mse_csv_path = os.path.join(base_dir, 'mse_comparison.csv')
                calculate_mse_between_3d_and_0d(
                    calibration_input_path,
                    csv_results_dict,
                    geometric_input_path=geometric_input_path,
                    zoom_start_idx=args.zoom_start,
                    zoom_end_idx=args.zoom_end,
                    output_csv_path=mse_csv_path,
                    verbose=verbose
                )
            except Exception as e:
                print(f"  ✗ Error calculating MSE: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  Skipping MSE calculation (missing calibration input or CSV results)")
        
        # Step 5: Generate comparison plots
        if not args.skip_plots:
            print("\n" + "="*60)
            print("Step 5: Generating comparison plots")
            print("="*60)
            
            try:
                # Import plotting functions
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visualizations'))
                from plot_inlets_comparison import plot_inlet_comparison, get_all_vessels_from_csv as get_vessels_inlets
                from plot_outlet_comparison import plot_outlet_comparison, get_all_vessels_from_csv as get_vessels_outlets
                
                # Build dictionary of calibrated CSV paths for all junction types
                calibrated_csv_paths = {}
                for jtype in args.junction_types:
                    calibrated_results_csv = junction_type_paths[jtype]['calibrated_results']
                    if os.path.exists(calibrated_results_csv):
                        calibrated_csv_paths[jtype] = str(calibrated_results_csv)
                
                if not calibrated_csv_paths:
                    print("  Warning: No calibrated CSV files found, skipping plots")
                elif not os.path.exists(geometric_results_csv):
                    print("  Warning: Geometric results CSV not found, skipping plots")
                else:
                    # Get list of all vessels
                    all_vessels = get_vessels_inlets(geometric_results_csv)
                    print(f"  Found {len(all_vessels)} vessels to plot")
                    
                    # Create output directories
                    inlets_output_dir = os.path.join('results', 'inlets_comparison', args.set_name, args.geo_name)
                    outlets_output_dir = os.path.join('results', 'outlet_comparison', args.set_name, args.geo_name)
                    os.makedirs(inlets_output_dir, exist_ok=True)
                    os.makedirs(outlets_output_dir, exist_ok=True)
                    
                    # Get time period (for plotting)
                    time_period = None
                    try:
                        from plot_inlets_comparison import get_time_period as get_time_period_func
                        time_period = get_time_period_func(args.set_name, args.geo_name)
                    except Exception:
                        pass
                    
                    # Plot inlets (one plot per vessel)
                    print(f"\n  Creating inlet comparison plots...")
                    inlet_success_count = 0
                    for vessel_name in all_vessels:
                        inlet_plot_path = os.path.join(inlets_output_dir, f"{vessel_name}_inlet_comparison.png")
                        try:
                            success = plot_inlet_comparison(
                                str(calibration_input_path),
                                str(geometric_results_csv),
                                calibrated_csv_paths,
                                vessel_name,
                                inlet_plot_path,
                                set_name=args.set_name,
                                geo_name=args.geo_name,
                                time_period=time_period,
                                geometric_input_path=str(geometric_input_path),
                                zoom_start_idx=args.zoom_start,
                                zoom_end_idx=args.zoom_end,
                                verbose=False
                            )
                            if success:
                                inlet_success_count += 1
                        except Exception as e:
                            print(f"    ✗ Failed to create inlet plot for {vessel_name}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"    Created {inlet_success_count}/{len(all_vessels)} inlet comparison plots")
                    
                    # Plot outlets (one plot per vessel)
                    print(f"\n  Creating outlet comparison plots...")
                    outlet_success_count = 0
                    for vessel_name in all_vessels:
                        outlet_plot_path = os.path.join(outlets_output_dir, f"{vessel_name}_outlet_comparison.png")
                        try:
                            success = plot_outlet_comparison(
                                str(calibration_input_path),
                                str(geometric_results_csv),
                                calibrated_csv_paths,
                                vessel_name,
                                outlet_plot_path,
                                set_name=args.set_name,
                                geo_name=args.geo_name,
                                time_period=time_period,
                                geometric_input_path=str(geometric_input_path),
                                zoom_start_idx=args.zoom_start,
                                zoom_end_idx=args.zoom_end,
                                verbose=False
                            )
                            if success:
                                outlet_success_count += 1
                        except Exception as e:
                            print(f"    ✗ Failed to create outlet plot for {vessel_name}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"    Created {outlet_success_count}/{len(all_vessels)} outlet comparison plots")
                    print(f"\n  Plot output directories:")
                    print(f"    Inlets: {inlets_output_dir}")
                    print(f"    Outlets: {outlets_output_dir}")
                    
            except ImportError as e:
                print(f"  ✗ Could not import plotting functions: {e}")
                print("  Make sure plot_inlets_comparison.py and plot_outlet_comparison.py are available")
            except Exception as e:
                print(f"  ✗ Error generating plots: {e}")
                import traceback
                traceback.print_exc()
    if verbose:
        print("\n" + "="*60)
        print("Done!")
        print("="*60)
        print(f"Geometric input: {geometric_input_path}")
        print(f"Geometric simulation results: {geometric_results_csv}")
        if not args.skip_calibration:
            print(f"\nBase calibration files:")
            print(f"  Calibration input: {calibration_input_path}")
            print(f"  Calibrated output: {calibrated_output_path}")
            print(f"\nJunction type variants:")
            for jtype in args.junction_types:
                print(f"  {jtype}:")
                print(f"    Calibration input: {junction_type_paths[jtype]['calibration_input']}")
                print(f"    Calibrated output: {junction_type_paths[jtype]['calibrated_output']}")
                print(f"    Simulation results: {junction_type_paths[jtype]['calibrated_results']}")


if __name__ == "__main__":
    main()

# python3 util/zerod_calibration/generate_zerod_inputs.py --set-name set_3 --geo-name tree_007
