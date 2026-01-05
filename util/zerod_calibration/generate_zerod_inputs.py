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




def extract_vessel_segments(centerline_data, centerline_polydata):
    """
    Extract vessel segments from centerline data.
    Groups points by BranchId and creates segments.
    
    Args:
        centerline_data: Dictionary of centerline arrays
        centerline_polydata: VTK polydata object
        
    Returns:
        List of vessel segment dictionaries
    """
    branch_id = centerline_data.get('BranchId', None)
    if branch_id is None:
        raise ValueError("BranchId array not found in centerline")
    
    area = centerline_data.get('CenterlineSectionArea', None)
    if area is None:
        raise ValueError("CenterlineSectionArea array not found in centerline")
    
    path = centerline_data.get('Path', None)
    if path is None:
        # Calculate path from points if not available
        points = centerline_data['Points']
        path = np.zeros(len(points))
        for i in range(1, len(points)):
            path[i] = path[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    # Group points by branch
    branches = defaultdict(list)
    for i in range(len(branch_id)):
        branches[branch_id[i]].append(i)
    
    # Sort points within each branch by path
    for branch_idx in branches:
        branches[branch_idx].sort(key=lambda i: path[i])
    
    # Create vessel segments
    vessels = []
    vessel_id = 0
    
    for branch_idx in sorted(branches.keys()):
        branch_points = branches[branch_idx]
        if len(branch_points) < 2:
            continue
        
        # Calculate segment properties
        segment_length = 0.0
        segment_areas = []
        segment_radii = []
        
        for i in range(len(branch_points) - 1):
            p1_idx = branch_points[i]
            p2_idx = branch_points[i + 1]
            
            # Length
            p1 = centerline_data['Points'][p1_idx]
            p2 = centerline_data['Points'][p2_idx]
            segment_length += np.linalg.norm(p2 - p1)
            
            # Area and radius
            if area[p1_idx] > 0:
                segment_areas.append(area[p1_idx])
                segment_radii.append(np.sqrt(area[p1_idx] / np.pi))
        
        if len(segment_areas) == 0:
            continue
        
        # Average properties
        mean_area = np.mean(segment_areas)
        mean_radius = np.mean(segment_radii)
        
        # Calculate geometric 0D parameters
        # Poiseuille resistance: R = 8*mu*L / (pi*r^4)
        R_poiseuille = 8 * MU * segment_length / (np.pi * mean_radius**4) if mean_radius > 0 else 0.0
        
        # Capacitance: C = 3*pi*r^3*L / (2*E*h) where E*h is wall stiffness
        # Using typical value: E*h = 1e6 dyn/cm^2 (approximate)
        E_h = 1e6  # Wall stiffness (dyn/cm^2)
        C = 3 * np.pi * mean_radius**3 * segment_length / (2 * E_h) if mean_radius > 0 else 0.0
        
        # Inductance: L = rho*L / A
        L = RHO * segment_length / mean_area if mean_area > 0 else 0.0
        
        vessel = {
            "vessel_id": vessel_id,
            "vessel_length": float(segment_length),
            "vessel_name": f"branch{branch_idx}_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": float(R_poiseuille),
                "C": float(C),
                "L": float(L),
                "stenosis_coefficient": 0.0
            }
        }
        
        vessels.append(vessel)
        vessel_id += 1
    
    return vessels


def identify_junctions_and_bcs(centerline_data, vessels):
    """
    Identify junctions and boundary conditions from centerline topology.
    
    Args:
        centerline_data: Dictionary of centerline arrays
        vessels: List of vessel dictionaries
        
    Returns:
        junctions: List of junction dictionaries
        boundary_conditions: List of BC dictionaries
        vessel_bc_map: Mapping from vessels to BCs
    """
    branch_id = centerline_data.get('BranchId', None)
    points = centerline_data['Points']
    cells = centerline_data.get('Cells', [])
    
    # Build connectivity graph
    branch_connections = defaultdict(lambda: {'inlets': [], 'outlets': []})
    
    # Find branch endpoints
    branch_endpoints = defaultdict(lambda: {'start': None, 'end': None})
    for i, bid in enumerate(branch_id):
        if branch_endpoints[bid]['start'] is None:
            branch_endpoints[bid]['start'] = i
        branch_endpoints[bid]['end'] = i
    
    # Find connections between branches (junctions)
    # This is simplified - in practice, you'd need to check spatial proximity
    # or use BifurcationId if available
    junctions = []
    junction_id = 0
    
    # Group vessels by branch
    branch_to_vessel = {}
    for i, vessel in enumerate(vessels):
        branch_name = vessel['vessel_name'].split('_')[0]  # e.g., "branch0"
        branch_idx = int(branch_name.replace('branch', ''))
        branch_to_vessel[branch_idx] = i
    
    # Simple junction detection: assume tree structure
    # Branch 0 is typically the root (inlet)
    # Other branches connect to it or to each other
    junction_vessels = defaultdict(lambda: {'inlets': [], 'outlets': []})
    
    # Find inlet (branch 0)
    if 0 in branch_to_vessel:
        inlet_vessel_id = branch_to_vessel[0]
        # Assume branch 0 connects to branches 1 and 2 (or more)
        for branch_idx in sorted(branch_to_vessel.keys()):
            if branch_idx == 0:
                continue
            if branch_idx <= 2:  # First two branches after root
                junction_vessels[junction_id]['inlets'].append(inlet_vessel_id)
                junction_vessels[junction_id]['outlets'].append(branch_to_vessel[branch_idx])
        junction_id += 1
    
    # Create junction dictionaries
    junctions = []
    for jid, conn in junction_vessels.items():
        if len(conn['inlets']) > 0 and len(conn['outlets']) > 0:
            junctions.append({
                "inlet_vessels": conn['inlets'],
                "junction_name": f"J{jid}",
                "junction_type": "NORMAL_JUNCTION",
                "outlet_vessels": conn['outlets']
            })
    
    # Create boundary conditions
    # Inlet BC
    boundary_conditions = [{
        "bc_name": "INFLOW",
        "bc_type": "FLOW",
        "bc_values": {
            "Q": [0.0, 0.0],  # Will be replaced with actual data
            "t": [0.0, 1.0]
        }
    }]
    
    # Outlet BCs (one per terminal branch)
    outlet_counter = 1
    vessel_bc_map = {}
    
    # Find terminal vessels (not connected to any junction as outlet)
    all_outlet_vessels = set()
    for junc in junctions:
        all_outlet_vessels.update(junc['outlet_vessels'])
    
    for i, vessel in enumerate(vessels):
        if i not in all_outlet_vessels:
            # This is a terminal vessel
            bc_name = f"OUT{outlet_counter}"
            boundary_conditions.append({
                "bc_name": bc_name,
                "bc_type": "RESISTANCE",
                "bc_values": {
                    "Pd": 0.0,
                    "R": 1.0
                }
            })
            vessel_bc_map[bc_name] = {
                "name": vessel['vessel_name'],
                "pressure": "pressure",
                "flow": "flow"
            }
            # Add BC to vessel
            vessel['boundary_conditions'] = {"outlet": bc_name}
            outlet_counter += 1
    
    # Add inlet BC to first vessel
    if vessels:
        vessels[0]['boundary_conditions'] = {"inlet": "INFLOW"}
        vessel_bc_map["INFLOW"] = {
            "name": vessels[0]['vessel_name'],
            "pressure": "pressure",
            "flow": "flow"
        }
    
    return junctions, boundary_conditions, vessel_bc_map


def find_inlet_outlet_caps(geo_dir):
    """
    Find inlet and outlet cap files from geometry directory.
    Looks in both svVascularize format (mesh/fluid_msh_0/mesh-surfaces) 
    and SimVascular format (mesh-complete/mesh-surfaces).
    The largest cap is the inlet, all others are outlets.
    
    Args:
        geo_dir: Path to geometry directory
        
    Returns:
        inlet_cap: Inlet cap filename
        outlet_caps: List of outlet cap filenames
        mesh_surfaces_dir: Path to mesh-surfaces directory
    """
    import vtk
    
    # Try svVascularize format first (mesh/fluid_msh_0/mesh-surfaces)
    mesh_dir = os.path.join(geo_dir, 'mesh', 'fluid_msh_0')
    mesh_surfaces_dir = os.path.join(mesh_dir, 'mesh-surfaces')
    
    # If not found, try SimVascular format (mesh-complete/mesh-surfaces)
    if not os.path.exists(mesh_surfaces_dir):
        mesh_complete_dir = os.path.join(geo_dir, 'mesh-complete')
        mesh_surfaces_dir = os.path.join(mesh_complete_dir, 'mesh-surfaces')
        
        # Check for inlet_cap directory in SimVascular format
        inlet_cap_dir = os.path.join(mesh_complete_dir, 'inlet_cap')
        if os.path.exists(inlet_cap_dir):
            inlet_files = [f for f in os.listdir(inlet_cap_dir) if f.endswith('.vtp')]
            if inlet_files:
                inlet_cap_from_dir = "cap_" + inlet_files[0]
            else:
                inlet_cap_from_dir = None
        else:
            inlet_cap_from_dir = None
    else:
        inlet_cap_from_dir = None
    
    if not os.path.exists(mesh_surfaces_dir):
        raise FileNotFoundError(
            f"mesh-surfaces directory not found. Tried:\n"
            f"  - {os.path.join(geo_dir, 'mesh', 'fluid_msh_0', 'mesh-surfaces')}\n"
            f"  - {os.path.join(geo_dir, 'mesh-complete', 'mesh-surfaces')}"
        )
    
    cap_files = [f for f in os.listdir(mesh_surfaces_dir) if f.startswith('cap_') and f.endswith('.vtp')]
    
    if not cap_files:
        raise ValueError(f"No cap files found in {mesh_surfaces_dir}")
    
    # Calculate areas to determine inlet (largest cap)
    cap_areas = []
    for cap_file in cap_files:
        cap_path = os.path.join(mesh_surfaces_dir, cap_file)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(cap_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputData(polydata)
        mass_props.Update()
        area = mass_props.GetSurfaceArea()
        
        cap_areas.append((cap_file, area))
    
    # Sort by area (largest first)
    cap_areas.sort(key=lambda x: x[1], reverse=True)
    
    # Use provided inlet_cap if found, otherwise use largest
    if inlet_cap_from_dir and inlet_cap_from_dir in [c[0] for c in cap_areas]:
        inlet_cap_name = inlet_cap_from_dir
        outlet_caps = [c[0] for c in cap_areas if c[0] != inlet_cap_from_dir]
    else:
        inlet_cap_name = cap_areas[0][0]
        outlet_caps = [c[0] for c in cap_areas[1:]]
    
    print(f"  Inlet cap: {inlet_cap_name} (area: {cap_areas[0][1]:.6f})")
    print(f"  Found {len(outlet_caps)} outlet caps")
    print(f"  Mesh surfaces directory: {mesh_surfaces_dir}")
    
    return inlet_cap_name, outlet_caps, mesh_surfaces_dir


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
    update_geometric_input_from_simulation(geo_dir, output_path, inlet_cap, num_cardiac_cycles=num_cardiac_cycles)
    
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




def update_geometric_input_from_simulation(geo_dir, json_path, inlet_cap_name, num_cardiac_cycles=1):
    """
    Update geometric input JSON with simulation parameters from XML and flow file.
    
    Args:
        geo_dir: Geometry directory containing XML and flow files
        json_path: Path to geometric input JSON file
        inlet_cap_name: Name of inlet cap (e.g., 'cap_6')
        num_cardiac_cycles: Number of cardiac cycles (default: 1)
    """
    print(f"  Looking for inflow BC data...")
    print(f"  Inlet cap name: {inlet_cap_name}")
    
    # Fallback to original XML/flow file approach
    print(f"  Using XML and flow files from: {geo_dir}")
    
    # Find simulation XML file
    xml_files = [
        os.path.join(geo_dir, 'fluid_simulation_0-0.xml'),
        os.path.join(geo_dir, 'fluid_simulation.xml'),
    ]
    
    xml_path = None
    for path in xml_files:
        if os.path.exists(path):
            xml_path = path
            print(f"  Found XML file: {xml_path}")
            break
    
    if xml_path is None:
        print(f"  Warning: Could not find simulation XML file. Tried: {xml_files}")
        return
    
    # Parse XML
    sim_params = parse_simulation_xml(xml_path)
    if sim_params is None:
        print("Warning: Could not parse simulation XML. Using defaults.")
        return
    
    num_time_steps = sim_params['num_time_steps']
    time_step_size = sim_params['time_step_size']
    
    print(f"  Found simulation parameters:")
    print(f"    Number_of_time_steps: {num_time_steps}")
    print(f"    Time_step_size: {time_step_size}")
    
    # Get inlet BC name from XML (preferred) or use provided inlet_cap_name as fallback
    xml_inlet_bc = sim_params.get('inlet_bc_name')
    if xml_inlet_bc:
        print(f"    Inlet BC from XML: {xml_inlet_bc}")
        inlet_bc_to_use = xml_inlet_bc
    else:
        print(f"    Using provided inlet cap name: {inlet_cap_name}")
        inlet_bc_to_use = inlet_cap_name
    
    # Find flow file
    flow_files = [
        os.path.join(geo_dir, f'{inlet_bc_to_use}.flow'),
        os.path.join(geo_dir, f'{inlet_cap_name}.flow'),  # Fallback to provided name
        os.path.join(geo_dir, 'inflow.flow'),
    ]
    
    flow_path = None
    for path in flow_files:
        if os.path.exists(path):
            flow_path = path
            print(f"  Found flow file: {flow_path}")
            break
    
    if flow_path is None:
        print(f"  Warning: Could not find flow file. Tried: {flow_files}")
        return
    
    # Read flow file
    times, flows = read_flow_file(flow_path)
    if times is None or flows is None or len(times) == 0:
        print(f"Warning: Could not read flow values from {flow_path}. Using defaults.")
        return
    
    print(f"  Found flow file: {flow_path}")
    print(f"    Number of flow points: {len(flows)}")
    
    # Read JSON
    with open(json_path, 'r') as f:
        zerod_input = json.load(f)
    
    # Generate time array from 0 with time_step_size increments
    bc_times = [i * time_step_size for i in range(num_time_steps)]
    
    # Interpolate flow values to match the time steps
    # Use the flow file's time values to interpolate
    if len(flows) != num_time_steps or len(times) != len(flows):
        # Need to interpolate
        # Handle case where times might not be sorted
        print("Interpolating")
        times_array = np.array(times)
        flows_array = np.array(flows)
        sort_idx = np.argsort(times_array)
        times_sorted = times_array[sort_idx]
        flows_sorted = flows_array[sort_idx]
        
        # Use numpy interpolation if scipy.interp1d is not available
        if HAS_SCIPY_INTERP:
            # Create interpolator (extrapolate using nearest value)
            interp_func = interp1d(times_sorted, flows_sorted, kind='linear', 
                                  bounds_error=False, fill_value=(flows_sorted[0], flows_sorted[-1]))
            # Interpolate to bc_times
            flows = interp_func(bc_times).tolist()
        else:
            # Simple linear interpolation using numpy
            bc_times_array = np.array(bc_times)
            flows_interp = np.interp(bc_times_array, times_sorted, flows_sorted)
            flows = flows_interp.tolist()
    else:
        # Use flows as-is if they match
        flows = flows[:num_time_steps]
    
    # Flip sign of flow values (flow file typically has negative values for inflow)
    flows = [-q for q in flows]
    
    # Update inflow boundary condition
    for bc in zerod_input.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc['bc_values']['t'] = bc_times
            bc['bc_values']['Q'] = flows
            print(f"  Updated INFLOW BC:")
            print(f"    Number of time points: {len(bc_times)}")
            print(f"    Time range: [{bc_times[0]:.6f}, {bc_times[-1]:.6f}]")
            print(f"    Flow range: [{min(flows):.3f}, {max(flows):.3f}]")
            break

    # Ensure simulation_parameters.number_of_time_pts_per_cardiac_cycle
    # matches the length of the inflow BC time series
    if 'simulation_parameters' in zerod_input:
        n_pts = len(bc_times)
        zerod_input['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = n_pts
        zerod_input['simulation_parameters']['number_of_cardiac_cycles'] = 1
        zerod_input['simulation_parameters']['steady_initial'] = False
        zerod_input['simulation_parameters']['absolute_tolerance'] = 1e-5
        zerod_input['simulation_parameters']['maximum_nonlinear_iterations'] = 50
        print(f"  Updated number_of_time_pts_per_cardiac_cycle: {n_pts}")
        print(f"  Updated number_of_cardiac_cycles: {num_cardiac_cycles}")
        print(f"  Set steady_initial: False")
        print(f"  Set absolute_tolerance: 1e-5")
        print(f"  Set maximum_nonlinear_iterations: 50")
    
    # Set all capacitance (C) values to 10^-10
    capacitance_value = 1e-10
    vessels_updated = 0
    for vessel in zerod_input.get('vessels', []):
        if 'zero_d_element_values' in vessel and 'C' in vessel['zero_d_element_values']:
            vessel['zero_d_element_values']['C'] = capacitance_value
            vessels_updated += 1
        if 'zero_d_element_values' in vessel and 'stenosis_coefficient' in vessel['zero_d_element_values']:
            vessel['zero_d_element_values']['stenosis_coefficient'] = 0.0

    print(f"  Set capacitance (C) to {capacitance_value} for {vessels_updated} vessels")
    
    # Write updated JSON
    with open(json_path, 'w') as f:
        json.dump(zerod_input, f, indent=4)
    
    print(f"  Updated geometric input with simulation parameters")
    print(f"  Saved to: {json_path}")
    
    # Verify the update
    with open(json_path, 'r') as f:
        verify = json.load(f)
        for bc in verify.get('boundary_conditions', []):
            if bc.get('bc_name') == 'INFLOW':
                print(f"  Verification - INFLOW BC has {len(bc['bc_values']['t'])} time points")
                print(f"    First time: {bc['bc_values']['t'][0]}, Last time: {bc['bc_values']['t'][-1]}")
                print(f"    First flow: {bc['bc_values']['Q'][0]:.3f}, Last flow: {bc['bc_values']['Q'][-1]:.3f}")
                break


def extract_observations_from_1d(centerline_soln_path, geometric_input_path, geo_dir=None, start_idx=0):
    """
    Extract observation data from 1D centerline solution VTP file.
    Extracts observations at boundaries and junctions following the format expected by svZeroDCalibrator.
    
    Args:
        centerline_soln_path: Path to centerline solution VTP (with pressure/velocity arrays)
        geometric_input_path: Path to geometric 0D input JSON (to understand vessel/junction structure)
        geo_dir: Geometry directory containing XML file (optional, will try to infer from paths)
        start_idx: Starting index for observations (default: 0). Observations will be sliced from this index.
        
    Returns:
        Dictionary with observation data (y, dy) for calibration
    """
    print(f"Reading 1D solution from: {centerline_soln_path}")
    centerline_data, _ = read_centerline_vtp(centerline_soln_path)
    
    # Read geometric input to understand vessel/junction structure
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    
    # Create vessel name to index mapping
    vessel_name_to_idx = {v['vessel_name']: i for i, v in enumerate(vessels)}
    
    # Find all timestep arrays
    pressure_timesteps = []
    flow_timesteps = []
    end_idx = -50
    
    for key in centerline_data.keys():
        if key.startswith('pressure_'):
            pressure_timesteps.append(key)
        elif key.startswith('velocity_') or key.startswith('flow_'):
            flow_timesteps.append(key)
    
    # Sort timesteps
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    pressure_timesteps.sort(key=extract_timestep)
    flow_timesteps.sort(key=extract_timestep)
    
    if not pressure_timesteps or not flow_timesteps:
        raise ValueError("No pressure or flow timesteps found in centerline solution")
    
    # Calculate timestep increment from the solution
    timestep_indices = [extract_timestep(name) for name in pressure_timesteps]
    if len(timestep_indices) > 1:
        # Calculate increment between consecutive timesteps
        increments = [timestep_indices[i+1] - timestep_indices[i] for i in range(len(timestep_indices)-1)]
        # Use the most common increment (in case there are variations)
        timestep_increment = max(set(increments), key=increments.count) if increments else 1
    else:
        timestep_increment = 1
    
    # Try to get time_step_size from XML
    time_step_size = None
    if geo_dir is None:
        # Try to infer geo_dir from geometric_input_path
        # Typically: data/zeroD/set_X/tree_Y/geometric_input.json
        # XML would be at: data/threeD/set_X/tree_Y/fluid_simulation_0-0.xml
        geo_dir = os.path.dirname(geometric_input_path)
        # Try to find threeD directory
        parts = geo_dir.split(os.sep)
        if 'zeroD' in parts:
            idx = parts.index('zeroD')
            parts[idx] = 'threeD'
            geo_dir = os.sep.join(parts)
    
    xml_path = os.path.join(geo_dir, 'fluid_simulation_0-0.xml')
    if os.path.exists(xml_path):
        sim_params = parse_simulation_xml(xml_path)
        if sim_params and 'time_step_size' in sim_params:
            time_step_size = sim_params['time_step_size']
            print(f"  Found XML time_step_size: {time_step_size:.6f} s")
            print(f"  Timestep increment in solution: {timestep_increment}")
    
    # Extract time array (assume uniform time steps)
    num_timesteps = len(pressure_timesteps)
    times = np.linspace(0.0, 1.0, num_timesteps)  # Normalized time
    obs_len = len(times)
    
    # Extract data at boundary points and junctions
    branch_id = centerline_data.get('BranchId', None)
    gid = centerline_data.get('GlobalNodeId', None)
    
    # Find inlet (GID == 0) and outlets
    inlet_idx = None
    outlet_indices = []
    
    if gid is not None:
        for i in range(len(gid)):
            if gid[i] == 0:
                inlet_idx = i
            elif gid[i] > 0:  # Outlet
                outlet_indices.append(i)
    
    # Calculate dt from XML timestep size and solution increment
    if time_step_size is not None:
        dt = time_step_size * timestep_increment
        print(f"  Calculated dt for derivatives: {dt:.6f} s (time_step_size * increment = {time_step_size:.6f} * {timestep_increment})")
    else:
        # Fallback: use normalized time difference
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        print(f"  Warning: Could not find XML time_step_size, using normalized time difference: {dt:.6f}")
    
    # Extract observations
    observations = {"y": {}, "dy": {}}
    
    # Helper function to extract and refine data at a point
    def extract_at_point(point_idx, times, dt):
        if point_idx is None or point_idx >= len(branch_id):
            return None, None, None, None
        
        pressure_data = np.array([centerline_data[pt][point_idx] 
                               for pt in pressure_timesteps])
        flow_data = np.array([centerline_data[ft][point_idx] 
                           for ft in flow_timesteps])
        
        # Keep pressure in original units (dynes/cm^2) - conversion to mmHg done in visualization
        # Use data directly without refinement
        pressure_refined = pressure_data.tolist()
        flow_refined = flow_data.tolist()
        
        # Compute derivatives using numpy gradient with correct dt
        if len(times) > 1:
            pressure_der = np.gradient(pressure_data, dt).tolist()
            flow_der = np.gradient(flow_data, dt).tolist()
        else:
            pressure_der = [0.0] * len(pressure_refined)
            flow_der = [0.0] * len(flow_refined)
        
        return pressure_refined, pressure_der, flow_refined, flow_der
    
    # Extract observations at boundaries (inlet and outlets)
    # Inflow BC
    if inlet_idx is not None:
        p_ref, p_der, f_ref, f_der = extract_at_point(inlet_idx, times, dt)
        if p_ref is not None:
            observations["y"][f"pressure:INFLOW:branch0_seg0"] = p_ref[start_idx:end_idx]
            observations["dy"][f"pressure:INFLOW:branch0_seg0"] = p_der[start_idx:end_idx]
            observations["y"][f"flow:INFLOW:branch0_seg0"] = f_ref[start_idx:end_idx]
            observations["dy"][f"flow:INFLOW:branch0_seg0"] = f_der[start_idx:end_idx]
    
    # Outlet BCs - find vessels connected to outlets
    for vessel in vessels:
        if 'boundary_conditions' in vessel:
            bc_outlet = vessel['boundary_conditions'].get('outlet')
            if bc_outlet:
                # Find outlet point for this vessel's branch
                vessel_branch = int(vessel['vessel_name'].split('_')[0].replace('branch', ''))
                
                # Find the last point (outlet) of this branch
                outlet_point_idx = None
                # First try to find by matching branch_id and checking if it's an outlet
                for i in range(len(branch_id) - 1, -1, -1):  # Search backwards to find last point
                    if branch_id[i] == vessel_branch:
                        # Check if this is an outlet (either in outlet_indices or last point of branch)
                        if i in outlet_indices or gid is None:
                            outlet_point_idx = i
                            break
                
                # If not found, use the last point of the branch as fallback
                if outlet_point_idx is None:
                    for i in range(len(branch_id) - 1, -1, -1):
                        if branch_id[i] == vessel_branch:
                            outlet_point_idx = i
                            break
                
                if outlet_point_idx is not None:
                    p_ref, p_der, f_ref, f_der = extract_at_point(outlet_point_idx, times, dt)
                    if p_ref is not None:
                        observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = p_ref[start_idx:end_idx]
                        observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = p_der[start_idx:end_idx]
                        observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = f_ref[start_idx:end_idx]
                        observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = f_der[start_idx:end_idx]
                    else:
                        # Create zero observations if extraction failed
                        print(f"  Warning: Could not extract observations for {vessel['vessel_name']}:{bc_outlet}, using zeros")
                        raise ValueError(f"Could not extract observations for {vessel['vessel_name']}:{bc_outlet}")
                        # zero_obs = [0.0] * obs_len
                        # observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        # observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        # observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        # observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                else:
                    # Create zero observations if outlet point not found
                    print(f"  Warning: Could not find outlet point for {vessel['vessel_name']}:{bc_outlet}, using zeros")
                    raise ValueError(f"Could not find outlet point for {vessel['vessel_name']}:{bc_outlet}")
                    # zero_obs = [0.0] * obs_len
                    # observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    # observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    # observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    # observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
    
    # Extract observations at junctions
    # For each junction, extract data for vessels connected to it
    for junc in junctions:
        junc_name = junc.get('junction_name', '')
        inlet_vessel_ids = junc.get('inlet_vessels', [])
        outlet_vessel_ids = junc.get('outlet_vessels', [])
        
        # For inlet vessels: format is "flow:vessel_name:junction_name"
        for vessel_id in inlet_vessel_ids:
            if vessel_id < len(vessels):
                vessel = vessels[vessel_id]
                vessel_name = vessel['vessel_name']
                vessel_branch = int(vessel_name.split('_')[0].replace('branch', ''))
                
                # Find a point on this vessel near the junction (use last point of branch)
                # This is approximate - ideally we'd find the exact junction point
                for i in range(len(branch_id) - 1, -1, -1):
                    if branch_id[i] == vessel_branch:
                        p_ref, p_der, f_ref, f_der = extract_at_point(i, times, dt)
                        if p_ref is not None:
                            observations["y"][f"pressure:{vessel_name}:{junc_name}"] = p_ref[start_idx:end_idx]
                            observations["dy"][f"pressure:{vessel_name}:{junc_name}"] = p_der[start_idx:end_idx]
                            observations["y"][f"flow:{vessel_name}:{junc_name}"] = f_ref[start_idx:end_idx]
                            observations["dy"][f"flow:{vessel_name}:{junc_name}"] = f_der[start_idx:end_idx]
                        break
        
        # For outlet vessels: format is "flow:junction_name:vessel_name"
        for vessel_id in outlet_vessel_ids:
            if vessel_id < len(vessels):
                vessel = vessels[vessel_id]
                vessel_name = vessel['vessel_name']
                vessel_branch = int(vessel_name.split('_')[0].replace('branch', ''))
                
                # Find a point on this vessel near the junction (use first point of branch)
                for i in range(len(branch_id)):
                    if branch_id[i] == vessel_branch:
                        p_ref, p_der, f_ref, f_der = extract_at_point(i, times, dt)
                        if p_ref is not None:
                            observations["y"][f"pressure:{junc_name}:{vessel_name}"] = p_ref[start_idx:end_idx]
                            observations["dy"][f"pressure:{junc_name}:{vessel_name}"] = p_der[start_idx:end_idx]
                            observations["y"][f"flow:{junc_name}:{vessel_name}"] = f_ref[start_idx:end_idx]
                            observations["dy"][f"flow:{junc_name}:{vessel_name}"] = f_der[start_idx:end_idx]
                            
                        break
    
    return observations


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
        "tolerance_gradient": 1e-5,
        "tolerance_increment": 1e-10,
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
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        updated_count = 0
        # Find and update outlet BCs
        for bc in data.get('boundary_conditions', []):
            if bc.get('bc_type') == 'RESISTANCE':
                bc_name = bc.get('bc_name')
                if bc_name in outlet_params:
                    resistance, pd = outlet_params[bc_name]
                    old_r = bc['bc_values'].get('R', 1.0)
                    old_pd = bc['bc_values'].get('Pd', 0.0)
                    bc['bc_values']['R'] = resistance
                    bc['bc_values']['Pd'] = pd
                    updated_count += 1
                    print(f"    {bc_name}: R {old_r:.4f} -> {resistance:.4f}, Pd {old_pd:.4f} -> {pd:.4f}")
        
        if updated_count > 0:
            # Write updated file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"  ✓ Updated {updated_count} outlet BC(s) in {file_type}: {file_path}")
            return True
        else:
            print(f"  No outlet BCs found to update in {file_type}: {file_path}")
            return False
            
    except Exception as e:
        print(f"  Warning: Could not update {file_type} file {file_path}: {e}")
        return False


def replace_inlet_bc_in_calibrated_output(calibrated_output_path, calibration_input_path):
    """
    Replace the inlet boundary condition in calibrated output with the original observed BC from calibration input.
    
    Args:
        calibrated_output_path: Path to calibrated output JSON file
        calibration_input_path: Path to calibration input JSON file (contains observed_inflow_bc)
        
    Returns:
        True if BC was replaced, False otherwise
    """
    # Read calibration input to get original observed BC
    if not os.path.exists(calibration_input_path):
        print(f"  Warning: Calibration input not found at {calibration_input_path}")
        return False
    
    with open(calibration_input_path, 'r') as f:
        calib_input = json.load(f)
    
    original_bc_values = calib_input.get('observed_inflow_bc')
    if original_bc_values is None or 't' not in original_bc_values or 'Q' not in original_bc_values:
        print("  Warning: No observed_inflow_bc found in calibration input, keeping calibrated output BC")
        return False
    
    # Read calibrated output
    if not os.path.exists(calibrated_output_path):
        print(f"  Warning: Calibrated output not found at {calibrated_output_path}")
        return False
    
    with open(calibrated_output_path, 'r') as f:
        calibrated_output = json.load(f)
    
    # Find and update the INFLOW BC
    bc_found = False
    for bc in calibrated_output.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc['bc_values'] = {
                't': original_bc_values['t'].copy() if isinstance(original_bc_values['t'], list) else original_bc_values['t'].tolist(),
                'Q': original_bc_values['Q'].copy() if isinstance(original_bc_values['Q'], list) else original_bc_values['Q'].tolist()
            }
            bc_found = True
            print(f"  Replaced INFLOW BC with original observed BC")
            print(f"    Time points: {len(bc['bc_values']['t'])}")
            print(f"    Time range: [{bc['bc_values']['t'][0]:.6f}, {bc['bc_values']['t'][-1]:.6f}] s")
            break
    
    if not bc_found:
        # Add inflow BC if it doesn't exist
        if 'boundary_conditions' not in calibrated_output:
            calibrated_output['boundary_conditions'] = []
        calibrated_output['boundary_conditions'].append({
            'bc_name': 'INFLOW',
            'bc_type': 'FLOW',
            'bc_values': {
                't': original_bc_values['t'].copy() if isinstance(original_bc_values['t'], list) else original_bc_values['t'].tolist(),
                'Q': original_bc_values['Q'].copy() if isinstance(original_bc_values['Q'], list) else original_bc_values['Q'].tolist()
            }
        })
        print(f"  Added INFLOW BC with original observed BC")
        print(f"    Time points: {len(calibrated_output['boundary_conditions'][-1]['bc_values']['t'])}")
    

    # Write updated calibrated output
    with open(calibrated_output_path, 'w') as f:
        json.dump(calibrated_output, f, indent=4)
    
    return True


def update_geometric_input_with_calibration_bc(geometric_input_path, calibration_input_path):
    """
    Update geometric_input.json with inflow BC from calibration_input.json.
    Uses the full time frame (not the second half used for calibration).
    
    Args:
        geometric_input_path: Path to geometric input JSON
        calibration_input_path: Path to calibration input JSON
    """
    print(f"\nUpdating geometric input with inflow BC from calibration input...")
    
    if not os.path.exists(calibration_input_path):
        print(f"  Warning: Calibration input not found at {calibration_input_path}")
        return False
    
    # Read calibration input
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Use full BC if available (stored for forward simulations), otherwise use calibration BC
    if '_full_bc_for_forward_sim' in calib_data:
        # Use full time frame for forward simulations
        calib_inflow_bc = calib_data['_full_bc_for_forward_sim'].copy()
        print(f"  Using full time frame from calibration input (for forward simulations)")
    else:
        # Fallback: extract from boundary conditions (this would be second half)
        calib_inflow_bc = None
        for bc in calib_data.get('boundary_conditions', []):
            if bc.get('bc_name') == 'INFLOW':
                calib_inflow_bc = bc.get('bc_values', {})
                break
        if not calib_inflow_bc:
            print(f"  Warning: Could not find INFLOW BC in calibration input")
            return False
        print(f"  Warning: Full BC not found, using calibration BC (may be second half)")
    
    # Read geometric input
    with open(geometric_input_path, 'r') as f:
        geo_input = json.load(f)
    
    # Refine inlet BC for forward simulation (halve timestep size, interpolate flow)
    refined_bc = calib_inflow_bc #  refine_inlet_bc_for_forward_simulation(calib_inflow_bc)
    
    # Update inflow BC
    geo_updated = False
    n_pts_inflow = len(refined_bc.get('t', []))
    for bc in geo_input.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc['bc_values'] = refined_bc.copy()
            geo_updated = True
            print(f"  Updated geometric input INFLOW BC (refined for forward simulation):")
            print(f"    Original number of time points: {len(calib_inflow_bc.get('t', []))}")
            print(f"    Refined number of time points: {n_pts_inflow}")
            if refined_bc.get('t'):
                print(f"    Time range: [{refined_bc['t'][0]:.6f}, {refined_bc['t'][-1]:.6f}]")
            if refined_bc.get('Q'):
                print(f"    Flow range: [{min(refined_bc['Q']):.3f}, {max(refined_bc['Q']):.3f}]")
            break

    # Ensure simulation_parameters.number_of_time_pts_per_cardiac_cycle
    # matches the length of the refined inflow BC time series
    if geo_updated and 'simulation_parameters' in geo_input:
        geo_input['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = n_pts_inflow
        
        print(f"  Updated simulation_parameters.number_of_time_pts_per_cardiac_cycle to {n_pts_inflow}")
    
    if geo_updated:
        # Write updated geometric input
        with open(geometric_input_path, 'w') as f:
            json.dump(geo_input, f, indent=4)
        print(f"  Saved updated geometric input to: {geometric_input_path}")
        return True
    else:
        print(f"  Warning: Could not find INFLOW BC in geometric input to update")
        return False


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


def convert_simulation_results_to_csv(sim_results, output_csv_path):
    """
    Convert pysvzerod simulation results to CSV format.
    
    Args:
        sim_results: Dictionary returned by pysvzerod.simulate()
        output_csv_path: Path to save CSV file
        
    Returns:
        Path to saved CSV file
    """
    import csv
    
    # CSV format: location, time, flow_in, flow_out, pressure_in, pressure_out
    rows = []
    rows.append(["location", "time", "flow_in", "flow_out", "pressure_in", "pressure_out"])
    
    # Extract data from simulation results
    # Results are typically arrays where each index corresponds to a (vessel, time) pair
    if 'name' in sim_results and 'time' in sim_results:
        names = np.array(sim_results['name']) if isinstance(sim_results['name'], list) else sim_results['name']
        times = np.array(sim_results['time']) if isinstance(sim_results['time'], list) else sim_results['time']
        flow_in = np.array(sim_results.get('flow_in', [])) if isinstance(sim_results.get('flow_in', []), list) else sim_results.get('flow_in', np.array([]))
        flow_out = np.array(sim_results.get('flow_out', [])) if isinstance(sim_results.get('flow_out', []), list) else sim_results.get('flow_out', np.array([]))
        pressure_in = np.array(sim_results.get('pressure_in', [])) if isinstance(sim_results.get('pressure_in', []), list) else sim_results.get('pressure_in', np.array([]))
        pressure_out = np.array(sim_results.get('pressure_out', [])) if isinstance(sim_results.get('pressure_out', []), list) else sim_results.get('pressure_out', np.array([]))
        
        # Convert to numpy arrays for easier handling
        if not isinstance(names, np.ndarray):
            names = np.array(names)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        
        # Write all data points
        num_points = len(names) if len(names) > 0 else len(times)
        for i in range(num_points):
            if i < len(names) and i < len(times):
                row = [
                    str(names[i]) if i < len(names) else "unknown",
                    str(float(times[i])) if i < len(times) else "0.0",
                    str(float(flow_in[i])) if i < len(flow_in) else "0.0",
                    str(float(flow_out[i])) if i < len(flow_out) else "0.0",
                    str(float(pressure_in[i])) if i < len(pressure_in) else "0.0",
                    str(float(pressure_out[i])) if i < len(pressure_out) else "0.0"
                ]
                rows.append(row)
    
    # Write CSV file
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Simulation results saved to: {output_csv_path}")
    return output_csv_path


def verify_inlet_flow_matches_bc(input_data, results, inlet_vessel_name='branch0_seg0'):
    """
    Verify that the inlet flow in simulation results matches the boundary condition.
    
    Args:
        input_data: Input JSON data (contains BC)
        results: Simulation results dictionary or CSV path
        inlet_vessel_name: Name of inlet vessel
        
    Returns:
        True if matches (within tolerance), False otherwise
    """
    # Extract BC flow values
    bc_times = None
    bc_flows = None
    for bc in input_data.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc_values = bc.get('bc_values', {})
            bc_times = bc_values.get('t', [])
            bc_flows = bc_values.get('Q', [])
            break
    
    if bc_times is None or bc_flows is None:
        print("  Warning: Could not find INFLOW BC in input data")
        return False
    
    # Extract inlet flow from results
    if isinstance(results, str):
        # Results is a CSV path, read it
        import csv
        inlet_flows = []
        result_times = []
        with open(results, 'r') as f:
            reader = csv.DictReader(f)
            # Check which column name is used for vessel identifier by reading fieldnames
            fieldnames = reader.fieldnames
            if fieldnames is None:
                print("  Warning: Could not read CSV fieldnames")
                return False
            
            # Determine vessel identifier column name
            vessel_id_col = None
            if 'location' in fieldnames:
                vessel_id_col = 'location'
            elif 'name' in fieldnames:
                vessel_id_col = 'name'
            else:
                print("  Warning: Could not find 'location' or 'name' column in CSV")
                print(f"    Available columns: {fieldnames}")
                return False
            
            # Process all rows
            for row in reader:
                if row.get(vessel_id_col) == inlet_vessel_name:
                    result_times.append(float(row['time']))
                    inlet_flows.append(float(row['flow_in']))
    else:
        # Results is a dictionary from pysvzerod
        names = results.get('name', [])
        times = results.get('time', [])
        flow_in = results.get('flow_in', [])
        
        inlet_flows = []
        result_times = []
        for i, name in enumerate(names):
            if name == inlet_vessel_name:
                result_times.append(times[i] if i < len(times) else 0.0)
                inlet_flows.append(flow_in[i] if i < len(flow_in) else 0.0)
    
    if not inlet_flows:
        print(f"  Warning: Could not find inlet flow for vessel '{inlet_vessel_name}' in results")
        return False
    
    # Interpolate BC flows to match result times
    if HAS_SCIPY_INTERP:
        from scipy.interpolate import interp1d
        try:
            interp_func = interp1d(bc_times, bc_flows, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            bc_flows_interp = [float(interp_func(t)) for t in result_times]
        except:
            bc_flows_interp = np.interp(result_times, bc_times, bc_flows).tolist()
    else:
        bc_flows_interp = np.interp(result_times, bc_times, bc_flows).tolist()
    
    # Compare
    max_diff = 0.0
    max_diff_time = None
    tolerance = 1e-3  # Allow small numerical differences
    
    for i, (result_flow, bc_flow) in enumerate(zip(inlet_flows, bc_flows_interp)):
        diff = abs(result_flow - bc_flow)
        if diff > max_diff:
            max_diff = diff
            max_diff_time = result_times[i] if i < len(result_times) else None
    
    if max_diff > tolerance:
        print(f"  ⚠ WARNING: Inlet flow does not match BC!")
        print(f"    Maximum difference: {max_diff:.6f} cm³/s at t={max_diff_time:.3f}")
        print(f"    BC flow range: [{min(bc_flows):.2f}, {max(bc_flows):.2f}] cm³/s")
        print(f"    Result flow range: [{min(inlet_flows):.2f}, {max(inlet_flows):.2f}] cm³/s")
        print(f"    This may indicate convergence issues or BC application problems")
        return False
    else:
        print(f"  ✓ Verified: Inlet flow matches BC (max diff: {max_diff:.6e} cm³/s)")
        return True


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
            
            # For special junction types, we need to provide junction_values
            if junction_type in ['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction']:
                num_outlets = len(junc.get('outlet_vessels', []))
                
                if num_outlets > 0:
                    # Initialize or update junction values with only the required parameters
                    if 'junction_values' not in junc:
                        junc['junction_values'] = {}
                    
                    # Determine which parameters are needed based on junction type
                    # BloodVesselJunction, NORMAL_JUNCTION and DirIndepJunction: R_poiseuille, L, stenosis_coefficient
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
    parser.add_argument('--timestep-scale-factor', type=float, default=0.25,
                       help='Scaling factor for 0D simulation timestep relative to 1D/3D timestep (default: 0.25, i.e., quarter the timestep)')
    parser.add_argument('--junction-types', nargs='+', 
                       default=['BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction'],
                       help='Junction types to generate calibration files for (default: all four types)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index for observations (default: 0). Observations will be sliced from this index.')
    
    args = parser.parse_args()
    
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
            
            # First run calibration with base (BloodVesselJunction) input
            calibrated_output_path = os.path.join(base_dir, 'calibrated_output.json')
            
            try:
                calibrated_input = run_calibration(calibration_input_path, calibrated_output_path)
                print(f"  ✓ Base calibration completed")
                # Replace inlet BC with original observed BC
                replace_inlet_bc_in_calibrated_output(calibrated_output_path, calibration_input_path)
                # Update outlet BCs with fitted values
                update_outlet_bcs_in_file(calibrated_output_path, fitted_resistances, "base calibrated output")
            except Exception as e:
                print(f"  ✗ Base calibration failed: {e}")
                import traceback
                traceback.print_exc()
            
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
