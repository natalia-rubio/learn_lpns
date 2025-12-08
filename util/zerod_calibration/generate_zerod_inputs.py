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
import json
import vtk
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
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
NUM_OBS = 100  # Number of observations for calibration data refinement
RHO = 1.06  # Blood density (g/cm^3)
MU = 0.04  # Blood viscosity (Poise)


def read_centerline_vtp(centerline_path):
    """
    Read centerline VTP file and extract geometric data.
    
    Args:
        centerline_path: Path to centerline VTP file
        
    Returns:
        Dictionary with centerline data arrays
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centerline_path)
    reader.Update()
    centerline = reader.GetOutput()
    
    # Extract point data arrays
    point_data = centerline.GetPointData()
    arrays = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        arrays[array.GetName()] = v2n(array)
    
    # Extract points
    points = v2n(centerline.GetPoints().GetData())
    arrays['Points'] = points
    
    # Extract connectivity
    cells = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        if cell.GetNumberOfPoints() == 2:  # Line segment
            cells.append([cell.GetPointId(0), cell.GetPointId(1)])
    arrays['Cells'] = cells
    
    return arrays, centerline


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
                    "R": 0.0  # Will be calibrated
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
                                     simvascular_path=None, dt=0.2, num_time_steps=5):
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
model_params.inlet_face_names = ['{inlet_cap}']
model_params.outlet_face_names = {outlet_caps}
model_params.centerlines_file_name = '{centerline_abs}'

## Fluid properties.
fluid_props = params.FluidProperties()

## Set wall properties.
print('Set wall properties ...')
material = params.WallProperties.OlufsenMaterial()

## Set boundary conditions.
bcs = params.BoundaryConditions()
bcs.add_velocities(face_name='{inlet_cap}', file_name='{os.path.abspath(inflow_flow_file)}')
"""
    
    # Add resistance BCs for outlets
    for outlet_cap in outlet_caps:
        script_content += f"bcs.add_resistance(face_name='{outlet_cap}', resistance=0.0)\n"
    
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
    update_geometric_input_from_simulation(geo_dir, output_path, inlet_cap)
    
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


def parse_simulation_xml(xml_path):
    """
    Parse fluid_simulation XML file to extract simulation parameters and inlet BC name.
    
    Returns:
        dict with 'num_time_steps', 'time_step_size', and 'inlet_bc_name'
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find GeneralSimulationParameters
        gen_params = root.find('GeneralSimulationParameters')
        if gen_params is None:
            return None
        
        num_time_steps = gen_params.find('Number_of_time_steps')
        time_step_size = gen_params.find('Time_step_size')
        
        if num_time_steps is None or time_step_size is None:
            return None
        
        # Find inlet boundary condition (Dirichlet with Unsteady time dependence)
        inlet_bc_name = None
        add_equation = root.find('Add_equation')
        if add_equation is not None:
            for bc in add_equation.findall('Add_BC'):
                bc_type = bc.find('Type')
                time_dep = bc.find('Time_dependence')
                
                if (bc_type is not None and bc_type.text == 'Dirichlet' and
                    time_dep is not None and time_dep.text == 'Unsteady'):
                    inlet_bc_name = bc.get('name')
                    break
        
        result = {
            'num_time_steps': int(num_time_steps.text),
            'time_step_size': float(time_step_size.text)
        }
        
        if inlet_bc_name:
            result['inlet_bc_name'] = inlet_bc_name
        
        return result
    except Exception as e:
        print(f"Warning: Could not parse simulation XML: {e}")
        return None


def read_flow_file(flow_path):
    """
    Read flow file (.flow format).
    
    Format:
    First line: number_of_points (optional)
    Subsequent lines: time    flow_value
    
    Returns:
        tuple (times, flows) as lists
    """
    times = []
    flows = []
    
    try:
        with open(flow_path, 'r') as f:
            lines = f.readlines()
            
            # Skip first line if it's just a number
            start_idx = 0
            if len(lines) > 0:
                first_line = lines[0].strip().split()
                if len(first_line) == 1 or (len(first_line) == 2 and first_line[0].isdigit()):
                    start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        flow = float(parts[1])
                        times.append(time)
                        flows.append(flow)
                    except ValueError:
                        continue
        
        return times, flows
    except Exception as e:
        print(f"Warning: Could not read flow file {flow_path}: {e}")
        return None, None


def update_geometric_input_from_simulation(geo_dir, json_path, inlet_cap_name):
    """
    Update geometric input JSON with simulation parameters from XML and flow file.
    If calibration_input.json exists, use the inflow BC from there instead.
    
    Args:
        geo_dir: Geometry directory containing XML and flow files
        json_path: Path to geometric input JSON file
        inlet_cap_name: Name of inlet cap (e.g., 'cap_6')
    """
    print(f"  Looking for inflow BC data...")
    print(f"  Inlet cap name: {inlet_cap_name}")
    
    # First, check if calibration_input.json exists in the same directory
    calibration_input_path = os.path.join(os.path.dirname(json_path), 'calibration_input.json')
    
    if os.path.exists(calibration_input_path):
        print(f"  Found calibration_input.json, using inflow BC from there")
        try:
            with open(calibration_input_path, 'r') as f:
                calibration_input = json.load(f)
            
            # Extract inflow BC from calibration input
            bc_times = None
            bc_flows = None
            
            # Look for INFLOW boundary condition
            for bc in calibration_input.get('boundary_conditions', []):
                if bc.get('bc_name') == 'INFLOW':
                    bc_values = bc.get('bc_values', {})
                    bc_times = bc_values.get('t')
                    bc_flows = bc_values.get('Q')
                    if bc_times is not None and bc_flows is not None:
                        print(f"    Found INFLOW BC in calibration input")
                        print(f"      Number of time points: {len(bc_times)}")
                        print(f"      Time range: [{bc_times[0]:.6f}, {bc_times[-1]:.6f}]")
                        print(f"      Flow range: [{min(bc_flows):.3f}, {max(bc_flows):.3f}]")
                        break
            
            # If not found in boundary_conditions, try to extract from observations
            if bc_times is None or bc_flows is None:
                observations = calibration_input.get('y', {})
                # Look for flow observation at INFLOW
                flow_key = None
                for key in observations.keys():
                    if key.startswith('flow:INFLOW:') or key.startswith('flow:') and 'INFLOW' in key:
                        flow_key = key
                        break
                
                if flow_key:
                    print(f"    Found flow observation in calibration input: {flow_key}")
                    bc_flows = observations[flow_key]
                    # Generate time array (normalized to [0, 1])
                    bc_times = np.linspace(0.0, 1.0, len(bc_flows)).tolist()
                    print(f"      Number of time points: {len(bc_times)}")
                    print(f"      Flow range: [{min(bc_flows):.3f}, {max(bc_flows):.3f}]")
            
            if bc_times is not None and bc_flows is not None:
                # Read geometric input JSON
                with open(json_path, 'r') as f:
                    zerod_input = json.load(f)
                
                # Update number_of_time_pts_per_cardiac_cycle
                num_time_steps = len(bc_times)
                if 'simulation_parameters' in zerod_input:
                    zerod_input['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = num_time_steps
                    print(f"  Updated number_of_time_pts_per_cardiac_cycle: {num_time_steps}")
                
                # Update inflow boundary condition
                for bc in zerod_input.get('boundary_conditions', []):
                    if bc.get('bc_name') == 'INFLOW':
                        bc['bc_values']['t'] = bc_times
                        bc['bc_values']['Q'] = bc_flows
                        print(f"  Updated INFLOW BC from calibration_input.json:")
                        print(f"    Number of time points: {len(bc_times)}")
                        print(f"    Time range: [{bc_times[0]:.6f}, {bc_times[-1]:.6f}]")
                        print(f"    Flow range: [{min(bc_flows):.3f}, {max(bc_flows):.3f}]")
                        break
                
                # Set all capacitance (C) values to 10^-10
                capacitance_value = 1e-10
                vessels_updated = 0
                for vessel in zerod_input.get('vessels', []):
                    if 'zero_d_element_values' in vessel and 'C' in vessel['zero_d_element_values']:
                        vessel['zero_d_element_values']['C'] = capacitance_value
                        vessels_updated += 1
                print(f"  Set capacitance (C) to {capacitance_value} for {vessels_updated} vessels")
                
                # Write updated JSON
                with open(json_path, 'w') as f:
                    json.dump(zerod_input, f, indent=4)
                
                print(f"  Updated geometric input with inflow BC from calibration_input.json")
                print(f"  Saved to: {json_path}")
                return
            else:
                print(f"  Warning: Could not extract inflow BC from calibration_input.json")
                print(f"    Falling back to XML/flow file approach")
        except Exception as e:
            print(f"  Warning: Error reading calibration_input.json: {e}")
            print(f"    Falling back to XML/flow file approach")
    
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
    
    # Update number_of_time_pts_per_cardiac_cycle
    if 'simulation_parameters' in zerod_input:
        zerod_input['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = num_time_steps
        print(f"  Updated number_of_time_pts_per_cardiac_cycle: {num_time_steps}")
    
    # Generate time array from 0 with time_step_size increments
    bc_times = [i * time_step_size for i in range(num_time_steps)]
    
    # Interpolate flow values to match the time steps
    # Use the flow file's time values to interpolate
    if len(flows) != num_time_steps or len(times) != len(flows):
        # Need to interpolate
        # Handle case where times might not be sorted
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
    
    # Set all capacitance (C) values to 10^-10
    capacitance_value = 1e-10
    vessels_updated = 0
    for vessel in zerod_input.get('vessels', []):
        if 'zero_d_element_values' in vessel and 'C' in vessel['zero_d_element_values']:
            vessel['zero_d_element_values']['C'] = capacitance_value
            vessels_updated += 1
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
    
    # Set all capacitance (C) values to 10^-10
    capacitance_value = 1e-10
    vessels_updated = 0
    for vessel in zerod_input.get('vessels', []):
        if 'zero_d_element_values' in vessel and 'C' in vessel['zero_d_element_values']:
            vessel['zero_d_element_values']['C'] = capacitance_value
            vessels_updated += 1
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


def refine_curve(x, y, num):
    """
    Refine a curve using cubic spline interpolation with derivative.
    
    Args:
        x: X-coordinates
        y: Y-coordinates
        num: New number of points
        
    Returns:
        new_y: New y-coordinates
        new_dy: New dy-coordinates
    """
    if CubicSpline is None:
        # Fallback to simple linear interpolation
        x_new = np.linspace(x[0], x[-1], num)
        y_array = np.array(y)
        x_array = np.array(x)
        new_y = np.interp(x_new, x_array, y_array)
        # Simple finite difference for derivative
        new_dy = np.gradient(new_y, x_new[1] - x_new[0] if len(x_new) > 1 else 1.0)
        return new_y.tolist(), new_dy.tolist()
    
    y = y.copy()
    y[-1] = y[0]  # Periodic boundary
    x_new = np.linspace(x[0], x[-1], num)
    spline = CubicSpline(x, y, bc_type="periodic")
    new_y = spline(x_new)
    new_dy = spline.derivative()(x_new)
    return new_y.tolist(), new_dy.tolist()


def extract_observations_from_1d(centerline_soln_path, geometric_input_path):
    """
    Extract observation data from 1D centerline solution VTP file.
    Extracts observations at boundaries and junctions following the format expected by svZeroDCalibrator.
    
    Args:
        centerline_soln_path: Path to centerline solution VTP (with pressure/velocity arrays)
        geometric_input_path: Path to geometric 0D input JSON (to understand vessel/junction structure)
        
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
    
    # Extract time array (assume uniform time steps)
    num_timesteps = len(pressure_timesteps)
    times = np.linspace(0.0, 1.0, num_timesteps)  # Normalized time
    
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
    
    # Extract observations
    observations = {"y": {}, "dy": {}}
    
    # Helper function to extract and refine data at a point
    def extract_at_point(point_idx, times):
        if point_idx is None or point_idx >= len(branch_id):
            return None, None, None, None
        
        pressure_data = np.array([centerline_data[pt][point_idx] 
                               for pt in pressure_timesteps])
        flow_data = np.array([centerline_data[ft][point_idx] 
                           for ft in flow_timesteps])
        
        pressure_refined, pressure_der = refine_curve(times, pressure_data, NUM_OBS)
        flow_refined, flow_der = refine_curve(times, flow_data, NUM_OBS)
        
        return pressure_refined, pressure_der, flow_refined, flow_der
    
    # Extract observations at boundaries (inlet and outlets)
    # Inflow BC
    if inlet_idx is not None:
        p_ref, p_der, f_ref, f_der = extract_at_point(inlet_idx, times)
        if p_ref is not None:
            observations["y"][f"pressure:INFLOW:branch0_seg0"] = p_ref
            observations["dy"][f"pressure:INFLOW:branch0_seg0"] = p_der
            observations["y"][f"flow:INFLOW:branch0_seg0"] = f_ref
            observations["dy"][f"flow:INFLOW:branch0_seg0"] = f_der
    
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
                    p_ref, p_der, f_ref, f_der = extract_at_point(outlet_point_idx, times)
                    if p_ref is not None:
                        observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = p_ref
                        observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = p_der
                        observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = f_ref
                        observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = f_der
                    else:
                        # Create zero observations if extraction failed
                        print(f"  Warning: Could not extract observations for {vessel['vessel_name']}:{bc_outlet}, using zeros")
                        zero_obs = [0.0] * NUM_OBS
                        observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                else:
                    # Create zero observations if outlet point not found
                    print(f"  Warning: Could not find outlet point for {vessel['vessel_name']}:{bc_outlet}, using zeros")
                    zero_obs = [0.0] * NUM_OBS
                    observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
    
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
                        p_ref, p_der, f_ref, f_der = extract_at_point(i, times)
                        if p_ref is not None:
                            observations["y"][f"pressure:{vessel_name}:{junc_name}"] = p_ref
                            observations["dy"][f"pressure:{vessel_name}:{junc_name}"] = p_der
                            observations["y"][f"flow:{vessel_name}:{junc_name}"] = f_ref
                            observations["dy"][f"flow:{vessel_name}:{junc_name}"] = f_der
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
                        p_ref, p_der, f_ref, f_der = extract_at_point(i, times)
                        if p_ref is not None:
                            observations["y"][f"pressure:{junc_name}:{vessel_name}"] = p_ref
                            observations["dy"][f"pressure:{junc_name}:{vessel_name}"] = p_der
                            observations["y"][f"flow:{junc_name}:{vessel_name}"] = f_ref
                            observations["dy"][f"flow:{junc_name}:{vessel_name}"] = f_der
                        break
    
    return observations


def create_calibration_input(geometric_input_path, observations, output_path):
    """
    Create calibration input file from geometric input and observations.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        observations: Dictionary with observation data (y, dy)
        output_path: Path to save calibration input JSON
    """
    print(f"Reading geometric input from: {geometric_input_path}")
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Use observations directly to set inflow BC
    # Note: Forward simulation is skipped to avoid potential segfaults with invalid geometric models
    # The observations from 1D/3D results are used directly for calibration
    if "flow:INFLOW:branch0_seg0" in observations["y"]:
        bc_time = np.linspace(0.0, 1.0, NUM_OBS).tolist()
        bc_flow = observations["y"]["flow:INFLOW:branch0_seg0"]
        
        # Update inflow BC with observed flow
        for bc in inp["boundary_conditions"]:
            if bc["bc_name"] == "INFLOW":
                bc["bc_values"]["t"] = bc_time
                bc["bc_values"]["Q"] = bc_flow
                break
    else:
        print("Warning: No inflow flow data found in observations")
        # Use default constant flow
        bc_time = np.linspace(0.0, 1.0, NUM_OBS).tolist()
        bc_flow = [0.0] * NUM_OBS
        for bc in inp["boundary_conditions"]:
            if bc["bc_name"] == "INFLOW":
                bc["bc_values"]["t"] = bc_time
                bc["bc_values"]["Q"] = bc_flow
                break
    
    # Set all elements to zero for calibration
    for vessel in inp["vessels"]:
        for ele in vessel["zero_d_element_values"].keys():
            vessel["zero_d_element_values"][ele] = 0.0
    
    # Add calibration parameters
    inp["calibration_parameters"] = {
        "tolerance_gradient": 1e-5,
        "tolerance_increment": 1e-10,
        "maximum_iterations": 100,
        "calibrate_stenosis_coefficient": True,
        "set_capacitance_to_zero": False,
    }
    
    # Only calibrate to last cycle
    inp["simulation_parameters"]["output_all_cycles"] = False
    
    # Add observations
    inp.update(observations)
    
    # Write calibration input
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(inp, f, indent=4)
    
    print(f"Calibration input saved to: {output_path}")
    return inp


def run_calibration(calibration_input_path, output_path):
    """
    Run svZeroDCalibrator to generate calibrated input file.
    
    Args:
        calibration_input_path: Path to calibration input JSON
        output_path: Path to save calibrated output JSON
    """
    if pysvzerod is None:
        raise RuntimeError("pysvzerod not available. Cannot run calibration.")
    
    print(f"Running calibration...")
    with open(calibration_input_path, 'r') as f:
        config = json.load(f)
    
    try:
        cali = pysvzerod.calibrate(config)
    except Exception as e:
        print(f"Calibration failed: {e}")
        raise
    
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


def run_forward_simulation(input_json_path, output_csv_path):
    """
    Run forward 0D simulation and save results to CSV.
    Uses pysvzerod.simulate() by default, falls back to CasADi solver if it fails.
    
    Args:
        input_json_path: Path to 0D input JSON file
        output_csv_path: Path to save CSV results
        
    Returns:
        Simulation results dictionary (or None if using CasADi fallback)
    """
    if pysvzerod is None and not HAS_CASADI:
        raise RuntimeError("Neither pysvzerod nor CasADi solver available. Cannot run simulation.")
    
    print(f"Running forward simulation from: {input_json_path}")
    
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    # Try pysvzerod first
    if pysvzerod is not None:
        try:
            # Create a deep copy to avoid modifying the original
            import copy
            input_data_sim = copy.deepcopy(input_data)
            
            # Remove calibration parameters if present (not needed for forward simulation)
            if 'calibration_parameters' in input_data_sim:
                del input_data_sim['calibration_parameters']
            
            # Remove observation data if present
            if 'y' in input_data_sim:
                del input_data_sim['y']
            if 'dy' in input_data_sim:
                del input_data_sim['dy']
            
            # Note: Calibrated output may have BloodVesselJunction type with extra fields,
            # but pysvzerod.simulate() should handle it correctly
            
            print("  Attempting simulation with pysvzerod...")
            results = pysvzerod.simulate(input_data_sim)
            
            # Convert to CSV
            convert_simulation_results_to_csv(results, output_csv_path)
            
            print("  ✓ Simulation completed successfully with pysvzerod")
            return results
        except Exception as e:
            print(f"  ✗ pysvzerod simulation failed: {e}")
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
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            result_df.to_csv(output_csv_path, index=False)
            
            print(f"  ✓ Simulation completed successfully with CasADi solver")
            print(f"  Results saved to: {output_csv_path}")
            
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
    
    args = parser.parse_args()
    
    # Construct paths
    base_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    geometric_input_path = os.path.join(base_dir, 'geometric_input.json')
    calibration_input_path = os.path.join(base_dir, 'calibration_input.json')
    calibrated_output_path = os.path.join(base_dir, 'calibrated_output.json')
    geometric_results_csv = os.path.join(base_dir, 'geometric_results.csv')
    calibrated_results_csv = os.path.join(base_dir, 'calibrated_results.csv')
    
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
        num_time_steps=args.num_time_steps
    )
    
    # Step 2: Extract observations and create calibration input
    if not args.skip_calibration:
        print("\n" + "="*60)
        print("Step 2: Creating calibration input file")
        print("="*60)
        
        if args.one_d_soln:
            observations = extract_observations_from_1d(args.one_d_soln, geometric_input_path)
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
                observations = extract_observations_from_1d(soln_path, geometric_input_path)
            else:
                print("Warning: No 1D or 3D solution found. Skipping calibration.")
                print(f"  Looked for:")
                print(f"    - {os.path.join('data', 'oneD', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
                print(f"    - {os.path.join('data', 'reduced_results', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
                print(f"    - {os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
                args.skip_calibration = True
        
        if not args.skip_calibration:
            create_calibration_input(geometric_input_path, observations, calibration_input_path)
            
            # Step 3: Run calibration
            print("\n" + "="*60)
            print("Step 3: Running calibration")
            print("="*60)
            calibrated_input = run_calibration(calibration_input_path, calibrated_output_path)
            
            # Step 4: Run forward simulations
            print("\n" + "="*60)
            print("Step 4: Running forward simulations")
            print("="*60)
            
            # Run simulation with geometric input
            print("Running simulation with geometric input...")
            try:
                run_forward_simulation(geometric_input_path, geometric_results_csv)
            except Exception as e:
                print(f"Warning: Geometric simulation failed: {e}")
            
            # Run simulation with calibrated input
            print("Running simulation with calibrated input...")
            try:
                run_forward_simulation(calibrated_output_path, calibrated_results_csv)
                print(f"  ✓ Calibrated simulation completed successfully")
            except Exception as e:
                print(f"  ✗ Calibrated simulation failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"Geometric input: {geometric_input_path}")
    if not args.skip_calibration:
        print(f"Calibration input: {calibration_input_path}")
        print(f"Calibrated output: {calibrated_output_path}")
        print(f"Geometric simulation results: {geometric_results_csv}")
        print(f"Calibrated simulation results: {calibrated_results_csv}")


if __name__ == "__main__":
    main()

# python3 util/zerod_calibration/generate_zerod_inputs.py --set-name set_3 --geo-name tree_007
