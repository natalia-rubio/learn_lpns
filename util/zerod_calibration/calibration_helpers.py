#!/usr/bin/env python3
"""
Helper functions for calibration workflow.
"""

import os
import vtk
import numpy as np
import xml.etree.ElementTree as ET
from vtk.util.numpy_support import vtk_to_numpy as v2n


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


def timestep_from_1D(centerline_soln_path, geo_dir):
    """
    Extract timestep information from 1D centerline solution and XML file.
    
    Args:
        centerline_soln_path: Path to 1D centerline solution VTP file
        geo_dir: Geometry directory containing XML file
        
    Returns:
        tuple: (num_timesteps, time_step_size, bc_time) where:
            - num_timesteps: Number of timesteps in the solution
            - time_step_size: Time step size from XML (or None if not found)
            - bc_time: List of time values (or None if time_step_size not found)
    """
    # Read centerline solution
    centerline_data, _ = read_centerline_vtp(centerline_soln_path)
    flow_timesteps = [key for key in centerline_data.keys() if key.startswith('velocity_') or key.startswith('flow_')]
    
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    flow_timesteps.sort(key=extract_timestep)
    num_timesteps = len(flow_timesteps)
    time_increment = extract_timestep(flow_timesteps[1]) - extract_timestep(flow_timesteps[0])

    # Get timestep size from XML
    xml_path = os.path.join(geo_dir, 'fluid_simulation_0-0.xml')
    
    if not os.path.exists(xml_path):
        raise ValueError(f"XML file not found at {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find GeneralSimulationParameters
    gen_params = root.find('GeneralSimulationParameters')
    if gen_params is not None:
        time_step_size_elem = gen_params.find('Time_step_size')
        if time_step_size_elem is not None:
            threeD_time_step_size = float(time_step_size_elem.text)


    time_step_size = time_increment * threeD_time_step_size
    return time_step_size

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


def convert_numpy_to_list(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    
    Args:
        obj: Object that may contain numpy arrays (dict, list, numpy array, or other)
        
    Returns:
        Object with all numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj
