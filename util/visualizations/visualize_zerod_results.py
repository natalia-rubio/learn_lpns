#!/usr/bin/env python3
"""
Script to visualize 0D simulation results as a movie.
Shows 0D nodes on the 3D vasculature, colored by pressure or velocity over time.
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import json
import csv
import re
from collections import defaultdict

# Set up for off-screen rendering
os.environ['VTK_USE_OFFSCREEN'] = '1'

try:
    import vtk
    vtk.vtkRenderWindow.GlobalWarningDisplayOff()
    from vtk.util.numpy_support import vtk_to_numpy as v2n
    from vtk.util.numpy_support import numpy_to_vtk as n2v
except ImportError:
    print("Error: VTK is required. Make sure VTK is installed and available.")
    sys.exit(1)

import numpy as np

# Check for scipy interpolation
try:
    from scipy.interpolate import interp1d
    HAS_SCIPY_INTERP = True
except (ImportError, ValueError, AttributeError):
    HAS_SCIPY_INTERP = False

# Check for matplotlib
HAS_MATPLOTLIB = False
plt = None
try:
    # Work around matplotlib/pandas compatibility issue
    # The errors occur due to missing attributes in matplotlib.cbook
    # We need to patch cbook BEFORE importing matplotlib
    
    import sys
    import types
    from collections import namedtuple
    
    # The issue is that matplotlib.cbook is missing some attributes due to version mismatches
    # The error occurs during `import matplotlib` when cbook tries to access missing attributes
    # We need to catch this error and try to work around it
    
    # Strategy: Try to import matplotlib, catch AttributeError about cbook, and try to patch
    try:
        import matplotlib
    except AttributeError as attr_err:
        # If the error is about cbook missing attributes, try to work around it
        error_str = str(attr_err)
        if 'cbook' in error_str.lower() and ('_ExceptionInfo' in error_str or '_is_pandas_dataframe' in error_str):
            # The error happened during matplotlib import when it tried to access cbook
            # We can't easily patch it at this point, so we'll mark matplotlib as unavailable
            # This is a known compatibility issue between matplotlib and pandas versions
            raise attr_err  # Re-raise to be caught by outer except
        else:
            raise
    
    # Set backend before importing pyplot to avoid display issues
    try:
        matplotlib.use('Agg')  # Use non-interactive backend
    except Exception:
        pass  # Backend might already be set
    
    # Import cbook and patch missing attributes
    import matplotlib.cbook as cbook
    if not hasattr(cbook, '_is_pandas_dataframe'):
        cbook._is_pandas_dataframe = lambda x: False
    if not hasattr(cbook, '_ExceptionInfo'):
        _ExceptionInfo = namedtuple('_ExceptionInfo', ['type', 'value', 'traceback'])
        cbook._ExceptionInfo = _ExceptionInfo
    
    # Now try importing pyplot (this should work now that cbook is patched)
    # Wrap in try-except to catch any remaining AttributeErrors from cbook
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except AttributeError as attr_err:
        # If we get an AttributeError about missing cbook attributes, patch and retry
        error_str = str(attr_err)
        if 'matplotlib.cbook' in error_str or 'cbook' in error_str:
            # Try to add the missing attribute dynamically
            missing_attr = None
            if '_ExceptionInfo' in error_str:
                from collections import namedtuple
                cbook._ExceptionInfo = namedtuple('_ExceptionInfo', ['type', 'value', 'traceback'])
            elif '_is_pandas_dataframe' in error_str:
                cbook._is_pandas_dataframe = lambda x: False
            # Retry the import
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        else:
            raise
    
    # Test that it actually works by creating a simple figure
    fig, ax = plt.subplots(figsize=(1, 1))
    plt.close(fig)
    HAS_MATPLOTLIB = True
except (ImportError, AttributeError) as e:
    # Handle matplotlib import errors, especially compatibility issues with cbook
    error_str = str(e)
    HAS_MATPLOTLIB = False
    plt = None
    
    # Check if this is a known compatibility issue with matplotlib.cbook
    is_cbook_error = (
        'cbook' in error_str.lower() and 
        ('_ExceptionInfo' in error_str or '_is_pandas_dataframe' in error_str or 
         'cannot import name' in error_str or 'has no attribute' in error_str)
    )
    
    if not is_cbook_error:
        # For other errors, print a warning
        print(f"Warning: matplotlib not available: {e}")
    # For cbook compatibility errors, suppress the warning (it's a version mismatch issue)
except Exception as e:
    HAS_MATPLOTLIB = False
    plt = None
    # Suppress warnings for known compatibility issues
    error_str = str(e)
    is_cbook_error = (
        'cbook' in error_str.lower() and 
        ('_ExceptionInfo' in error_str or '_is_pandas_dataframe' in error_str or 
         'cannot import name' in error_str or 'has no attribute' in error_str)
    )
    if not is_cbook_error:
        print(f"Warning: matplotlib not available: {e}")


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def read_zerod_csv(csv_path):
    """
    Read 0D simulation results from CSV.
    
    Returns:
        results: Dictionary {location: {time: {field: value}}}
        times: Sorted list of time values
        fields: List of available fields (pressure_in, pressure_out, flow_in, flow_out)
    """
    results = defaultdict(lambda: defaultdict(dict))
    times = set()
    fields = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Check which column name is used for vessel identifier
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return dict(results), sorted(times), sorted(fields)
        
        vessel_col = None
        if 'location' in fieldnames:
            vessel_col = 'location'
        elif 'name' in fieldnames:
            vessel_col = 'name'
        else:
            raise ValueError(f"CSV file must have either 'location' or 'name' column. Found: {fieldnames}")
        
        for row in reader:
            location = row[vessel_col]
            time = float(row['time'])
            times.add(time)
            
            # Extract all numeric fields
            for key, value in row.items():
                if key not in [vessel_col, 'time']:
                    try:
                        val = float(value)
                        results[location][time][key] = val
                        fields.add(key)
                    except (ValueError, TypeError):
                        continue
    
    return dict(results), sorted(times), sorted(fields)


def read_geometric_input(json_path):
    """Read geometric input JSON to get vessel information."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from vessel_name to vessel info
    vessel_map = {}
    for vessel in data.get('vessels', []):
        vessel_name = vessel.get('vessel_name', '')
        vessel_map[vessel_name] = {
            'vessel_id': vessel.get('vessel_id'),
            'vessel_name': vessel_name,
            'vessel_length': vessel.get('vessel_length', 0.0)
        }
    
    return vessel_map


def parse_vessel_name(vessel_name):
    """
    Parse vessel name like 'branch0_seg0' to extract branch ID and segment ID.
    
    Returns:
        branch_id, segment_id
    """
    match = re.match(r'branch(\d+)_seg(\d+)', vessel_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def get_centerline_points_for_branch(centerline_polydata, branch_id):
    """
    Get centerline points for a specific branch.
    
    Returns:
        points: Array of point coordinates (N, 3)
        point_indices: Original point indices in centerline
    """
    point_data = centerline_polydata.GetPointData()
    branch_id_array = point_data.GetArray('BranchId')
    
    if branch_id_array is None:
        raise ValueError("BranchId array not found in centerline")
    
    branch_ids = v2n(branch_id_array)
    points = v2n(centerline_polydata.GetPoints().GetData())
    
    # Find points belonging to this branch
    mask = branch_ids == branch_id
    point_indices = np.where(mask)[0]
    branch_points = points[mask]
    
    # Sort by path if available
    path_array = point_data.GetArray('Path')
    if path_array is not None:
        paths = v2n(path_array)
        branch_paths = paths[mask]
        sort_idx = np.argsort(branch_paths)
        branch_points = branch_points[sort_idx]
        point_indices = point_indices[sort_idx]
    
    return branch_points, point_indices


def get_node_locations_for_vessel(centerline_polydata, vessel_name, vessel_map):
    """
    Get 3D locations for inlet and outlet nodes of a vessel segment.
    
    Returns:
        (inlet_location, outlet_location): Tuple of (x, y, z) coordinates, or (None, None) if not found
    """
    branch_id, segment_id = parse_vessel_name(vessel_name)
    if branch_id is None:
        return None, None
    
    try:
        branch_points, _ = get_centerline_points_for_branch(centerline_polydata, branch_id)
        
        if len(branch_points) == 0:
            return None, None
        
        # Use first point as inlet and last point as outlet
        inlet_location = branch_points[0]
        outlet_location = branch_points[-1]
        
        return inlet_location, outlet_location
    except Exception as e:
        print(f"Warning: Could not get locations for {vessel_name}: {e}")
        return None, None


def create_node_actor(position, value, color_range, field_name='pressure', radius=None, lookup_table=None):
    """
    Create a VTK actor for a single node (sphere).
    
    Args:
        position: (x, y, z) coordinates
        value: Scalar value for coloring
        color_range: (min, max) for color mapping
        field_name: Name of the field (for color mapping)
        radius: Sphere radius (if None, will be computed)
        lookup_table: VTK lookup table to use (if None, creates default)
    
    Returns:
        actor: VTK actor
    """
    # Create sphere source
    sphere = vtk.vtkSphereSource()
    if radius is None:
        radius = 0.5  # Default radius
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(20)
    sphere.SetPhiResolution(20)
    sphere.SetCenter(position)
    sphere.Update()
    
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    
    # Create scalar array for coloring
    polydata = sphere.GetOutput()
    point_data = polydata.GetPointData()
    
    # Create scalar array with the value
    scalars = np.full(polydata.GetNumberOfPoints(), value)
    scalar_array = n2v(scalars)
    scalar_array.SetName(field_name)
    point_data.SetScalars(scalar_array)
    
    mapper.SetScalarModeToUsePointData()
    mapper.SelectColorArray(field_name)
    
    # Use the provided lookup table if available
    if lookup_table is not None:
        mapper.SetLookupTable(lookup_table)
        mapper.UseLookupTableScalarRangeOn()  # Use the lookup table's range instead of mapper's
    else:
        # Fallback: set scalar range on mapper
        mapper.SetScalarRange(color_range[0], color_range[1])
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor


def create_vasculature_actor(mesh_path):
    """
    Create a VTK actor for the 3D vasculature mesh (gray, half opacity).
    
    Returns:
        actor: VTK actor
    """
    # Read mesh
    if mesh_path.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif mesh_path.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")
    
    reader.SetFileName(mesh_path)
    reader.Update()
    
    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    mapper.ScalarVisibilityOff()  # Use solid color
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Set gray color and half opacity
    actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Gray
    actor.GetProperty().SetOpacity(0.5)
    
    return actor


def find_files(set_name, geo_name, data_dir='data', output_dir='data/zeroD'):
    """
    Find all required files for visualization.
    
    Returns:
        Dictionary with file paths, or None if files not found
    """
    files = {}
    
    # CSV files
    zerod_dir = os.path.join(output_dir, set_name, geo_name)
    geometric_csv = os.path.join(zerod_dir, 'geometric_results.csv')
    calibrated_csv = os.path.join(zerod_dir, 'calibrated_results.csv')
    
    # Check which CSV files exist
    csv_files = []
    if os.path.exists(geometric_csv):
        csv_files.append(('geometric', geometric_csv))
    if os.path.exists(calibrated_csv):
        csv_files.append(('calibrated', calibrated_csv))
    
    if not csv_files:
        print(f"Error: No CSV results found in {zerod_dir}")
        return None
    
    files['csv_files'] = csv_files
    
    # Geometric input JSON
    geometric_input = os.path.join(zerod_dir, 'geometric_input.json')
    if not os.path.exists(geometric_input):
        print(f"Error: Geometric input not found: {geometric_input}")
        return None
    files['geometric_input'] = geometric_input
    
    # Calibration input JSON (for 3D solution observations)
    calibration_input = os.path.join(zerod_dir, 'calibration_input.json')
    if os.path.exists(calibration_input):
        files['calibration_input'] = calibration_input
    
    # Centerline
    centerline = os.path.join(data_dir, 'threeD', set_name, geo_name, 'centerlines_simVascular.vtp')
    if not os.path.exists(centerline):
        print(f"Error: Centerline not found: {centerline}")
        return None
    files['centerline'] = centerline
    
    # Mesh - try different possible locations
    mesh_paths = [
        os.path.join(data_dir, 'threeD', set_name, geo_name, 'mesh', 'fluid_msh_0', 'fluid_msh_0.vtu'),
        os.path.join(data_dir, 'threeD', set_name, geo_name, 'mesh-complete', 'mesh-complete.mesh.vtu'),
    ]
    
    mesh_found = None
    for mesh_path in mesh_paths:
        if os.path.exists(mesh_path):
            mesh_found = mesh_path
            break
    
    if mesh_found is None:
        print(f"Error: Mesh not found. Tried: {mesh_paths}")
        return None
    files['mesh'] = mesh_found
    
    return files


def read_observations_from_calibration(calibration_input_path, field='pressure'):
    """
    Read observation data from calibration input file.
    
    Returns:
        Dictionary {vessel_name: {'inlet': [values], 'outlet': [values]}} for given field
        times: Time array for observations
    """
    import json
    
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None
    
    observations = calib_data['y']
    
    # Extract time array (observations are typically normalized to [0, 1])
    # Count number of observations
    first_key = list(observations.keys())[0] if observations else None
    if first_key is None:
        return None, None
    
    num_obs = len(observations[first_key])
    times = np.linspace(0.0, 1.0, num_obs)
    
    # Parse observations and map to vessels
    vessel_obs = {}
    
    for key, values in observations.items():
        # Parse key format: "field:vessel_name:bc_name" or "field:bc_name:vessel_name"
        parts = key.split(':')
        if len(parts) != 3:
            continue
        
        obs_field = parts[0]
        if obs_field != field:
            continue
        
        # Determine if this is inlet or outlet
        # Format examples:
        # "pressure:INFLOW:branch0_seg0" -> inlet of branch0_seg0
        # "pressure:branch1_seg0:RESISTANCE_0" -> outlet of branch1_seg0
        # "pressure:branch0_seg0:J0" -> outlet of branch0_seg0 (at junction)
        # "pressure:J0:branch1_seg0" -> inlet of branch1_seg0 (from junction)
        
        vessel_name = None
        is_inlet = False
        
        if parts[1] == 'INFLOW':
            vessel_name = parts[2]
            is_inlet = True
        elif parts[1].startswith('branch') and parts[2].startswith('RESISTANCE'):
            vessel_name = parts[1]
            is_inlet = False
        elif parts[1].startswith('branch') and parts[2].startswith('J'):
            vessel_name = parts[1]
            is_inlet = False
        elif parts[1].startswith('J') and parts[2].startswith('branch'):
            vessel_name = parts[2]
            is_inlet = True
        
        if vessel_name:
            if vessel_name not in vessel_obs:
                vessel_obs[vessel_name] = {'inlet': None, 'outlet': None}
            
            # Convert pressure from dynes/cm^2 to mmHg if field is pressure
            if field == 'pressure':
                values = [v / 1333.0 for v in values]  # Convert to mmHg
            
            if is_inlet:
                vessel_obs[vessel_name]['inlet'] = values
            else:
                vessel_obs[vessel_name]['outlet'] = values
    
    return vessel_obs, times


def extract_inlet_flow_from_observations(calibration_input_path):
    """
    Extract inlet flow data from calibration input observations.
    
    Returns:
        times: Time array (normalized [0, 1])
        flows: Flow values array
    """
    import json
    
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None
    
    observations = calib_data['y']
    
    # Look for inlet flow: "flow:INFLOW:branch0_seg0"
    inlet_flow_key = None
    for key in observations.keys():
        if key.startswith('flow:INFLOW:'):
            inlet_flow_key = key
            break
    
    if inlet_flow_key is None:
        return None, None
    
    flows = observations[inlet_flow_key]
    num_obs = len(flows)
    times = np.linspace(0.0, 1.0, num_obs)
    
    return times, flows


def extract_inlet_pressure_from_observations(calibration_input_path):
    """
    Extract inlet pressure data from calibration input observations.
    Converts from dynes/cm^2 to mmHg (divide by 1333).
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values array (in mmHg)
    """
    import json
    
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None
    
    observations = calib_data['y']
    
    # Look for inlet pressure: "pressure:INFLOW:branch0_seg0"
    inlet_pressure_key = None
    for key in observations.keys():
        if key.startswith('pressure:INFLOW:'):
            inlet_pressure_key = key
            break
    
    if inlet_pressure_key is None:
        return None, None
    
    # Pressure is stored in dynes/cm^2, convert to mmHg
    pressures_dynes = observations[inlet_pressure_key]
    pressures = [p / 1333.0 for p in pressures_dynes]  # Convert to mmHg
    
    num_obs = len(pressures)
    times = np.linspace(0.0, 1.0, num_obs)
    
    return times, pressures


def create_flow_plot_actor(times_normalized, flows, current_time_normalized, plot_bounds_normalized, 
                           color_range_flow, time_range_actual):
    """
    Create a 2D plot actor showing flow vs time with a moving dot.
    
    Args:
        times_normalized: Normalized time array [0, 1] (full range)
        flows: Flow values array
        current_time_normalized: Current normalized time (0-1) for dot position
        plot_bounds_normalized: [x_min, x_max, y_min, y_max] in normalized viewport coordinates [0,1]
        color_range_flow: [min_flow, max_flow] for y-axis scaling
        time_range_actual: [t_min, t_max] actual time range for second half
        
    Returns:
        List of VTK actors (axes, curve, and dot)
    """
    actors = []
    
    # Filter to second half of time period
    mid_point = len(times_normalized) // 2
    times_second_half = times_normalized[mid_point:]
    flows_second_half = flows[mid_point:]
    
    # Normalize times to [0, 1] for the second half (for plotting)
    if len(times_second_half) > 1:
        times_plot = (times_second_half - times_second_half[0]) / (times_second_half[-1] - times_second_half[0])
    else:
        times_plot = times_second_half
    
    # Normalize flows to plot coordinates
    flow_min, flow_max = color_range_flow
    if flow_max > flow_min:
        flows_normalized = (flows_second_half - flow_min) / (flow_max - flow_min)
    else:
        flows_normalized = np.zeros_like(flows_second_half)
    
    # Map to normalized viewport coordinates [0,1] relative to renderer
    x_min, x_max, y_min, y_max = plot_bounds_normalized
    x_coords = x_min + times_plot * (x_max - x_min)
    y_coords = y_min + flows_normalized * (y_max - y_min)
    
    # Create axes (border)
    axes_points = vtk.vtkPoints()
    axes_points.InsertNextPoint(x_min, y_min, 0.0)  # Bottom left
    axes_points.InsertNextPoint(x_max, y_min, 0.0)  # Bottom right
    axes_points.InsertNextPoint(x_max, y_max, 0.0)  # Top right
    axes_points.InsertNextPoint(x_min, y_max, 0.0)  # Top left
    axes_points.InsertNextPoint(x_min, y_min, 0.0)  # Close rectangle
    
    axes_line = vtk.vtkPolyLine()
    axes_line.GetPointIds().SetNumberOfIds(5)
    for i in range(5):
        axes_line.GetPointIds().SetId(i, i)
    
    axes_cells = vtk.vtkCellArray()
    axes_cells.InsertNextCell(axes_line)
    
    axes_poly = vtk.vtkPolyData()
    axes_poly.SetPoints(axes_points)
    axes_poly.SetLines(axes_cells)
    
    # Create mapper for axes - use display coordinates directly
    axes_mapper = vtk.vtkPolyDataMapper2D()
    axes_mapper.SetInputData(axes_poly)
    
    axes_actor = vtk.vtkActor2D()
    axes_actor.SetMapper(axes_mapper)
    axes_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black axes
    axes_actor.GetProperty().SetLineWidth(3)  # Thicker for visibility
    actors.append(axes_actor)
    
    # Create curve points
    points = vtk.vtkPoints()
    num_points = len(x_coords)
    for i in range(num_points):
        points.InsertNextPoint(x_coords[i], y_coords[i], 0.0)
    
    # Create polyline for curve
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(num_points)
    for i in range(num_points):
        line.GetPointIds().SetId(i, i)
    
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(cells)
    
    # Create mapper and actor for curve - use display coordinates directly
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly_data)
    
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue curve
    actor.GetProperty().SetLineWidth(3)  # Thicker for visibility
    actors.append(actor)
    
    # Create moving dot at current time
    # Map current_time_normalized (which is in [0.5, 1.0] for second half) to plot coordinates
    if current_time_normalized >= 0.5:
        # Map from [0.5, 1.0] to [0, 1] for second half
        time_in_second_half = (current_time_normalized - 0.5) / 0.5
        time_in_second_half = max(0.0, min(1.0, time_in_second_half))
        
        # Interpolate flow value
        if len(flows_second_half) > 1:
            if HAS_SCIPY_INTERP:
                interp_func = interp1d(times_plot, flows_second_half, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
                current_flow = float(interp_func(time_in_second_half))
            else:
                current_flow = float(np.interp(time_in_second_half, times_plot, flows_second_half))
        else:
            current_flow = flows_second_half[0] if len(flows_second_half) > 0 else 0.0
        
        # Normalize flow
        if flow_max > flow_min:
            flow_normalized = (current_flow - flow_min) / (flow_max - flow_min)
        else:
            flow_normalized = 0.0
        
        # Map to viewport coordinates
        dot_x = x_min + time_in_second_half * (x_max - x_min)
        dot_y = y_min + flow_normalized * (y_max - y_min)
        
        # Create dot (larger for visibility) - use normalized radius
        dot_radius_normalized = 0.015  # Radius in normalized viewport coordinates
        dot_source = vtk.vtkRegularPolygonSource()
        dot_source.SetNumberOfSides(20)
        dot_source.SetRadius(dot_radius_normalized)
        dot_source.SetCenter(dot_x, dot_y, 0.0)
        dot_source.Update()
        
        dot_mapper = vtk.vtkPolyDataMapper2D()
        dot_mapper.SetInputData(dot_source.GetOutput())
        
        dot_actor = vtk.vtkActor2D()
        dot_actor.SetMapper(dot_mapper)
        dot_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red dot
        actors.append(dot_actor)
    
    return actors


def create_flow_plot_movie(calibration_input_path, output_path, resolution=(1920, 1080), fps=5):
    """
    Create a standalone movie showing inlet flow and pressure vs time with moving dots using matplotlib.
    
    Args:
        calibration_input_path: Path to calibration input JSON (contains flow and pressure observations)
        output_path: Path to save output MP4
        resolution: (width, height) for output video
        fps: Frames per second
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not found.")
        return False
    
    print("="*60)
    print("Creating Flow and Pressure Plot Movie (using matplotlib)")
    print("="*60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found.")
        return False
    
    # Extract inlet flow data
    print("\nExtracting inlet flow data...")
    flow_times, flow_values = extract_inlet_flow_from_observations(calibration_input_path)
    if flow_times is None or flow_values is None:
        print("Error: Could not extract inlet flow data from calibration input")
        return False
    
    print(f"  Found {len(flow_values)} flow points")
    
    # Extract inlet pressure data
    print("\nExtracting inlet pressure data...")
    pressure_times, pressure_values = extract_inlet_pressure_from_observations(calibration_input_path)
    if pressure_times is None or pressure_values is None:
        print("  Warning: Could not extract inlet pressure data, will only show flow")
        pressure_times = None
        pressure_values = None
    else:
        print(f"  Found {len(pressure_values)} pressure points")
    
    # Filter to second half
    mid_point = len(flow_times) // 2
    times_second_half = flow_times[mid_point:]
    flows_second_half = flow_values[mid_point:]
    
    if pressure_times is not None:
        pressure_times_second_half = pressure_times[mid_point:]
        pressures_second_half = pressure_values[mid_point:]
    else:
        pressure_times_second_half = None
        pressures_second_half = None
    
    # Create time array for simulation (second half)
    # Use the same number of frames as the main visualization would use
    num_frames = len(flows_second_half) * 2  # 2x for smooth animation
    times_sim = np.linspace(times_second_half[0], times_second_half[-1], num_frames)
    
    # Flow range
    flow_min, flow_max = np.min(flows_second_half), np.max(flows_second_half)
    print(f"  Flow range: [{flow_min:.2f}, {flow_max:.2f}] cm³/s")
    print(f"  Time range: [{times_second_half[0]:.3f}, {times_second_half[-1]:.3f}]")
    
    if pressures_second_half is not None:
        pressure_min, pressure_max = np.min(pressures_second_half), np.max(pressures_second_half)
        print(f"  Pressure range: [{pressure_min:.2f}, {pressure_max:.2f}] mmHg")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='flow_plot_')
    print(f"\nRendering {num_frames} frames...")
    print(f"  Temporary directory: {temp_dir}")
    
    try:
        image_files = []
        for idx, time_sim in enumerate(times_sim):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Frame {idx+1}/{num_frames}: t={time_sim:.3f}")
            
            # Create figure with subplots (2 rows: flow on top, pressure on bottom)
            if pressures_second_half is not None:
                fig, axes = plt.subplots(2, 1, figsize=(resolution[0]/100, resolution[1]/100), dpi=100, sharex=True)
                ax_flow = axes[0]
                ax_pressure = axes[1]
            else:
                fig, ax_flow = plt.subplots(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
                ax_pressure = None
            
            fig.patch.set_facecolor('white')
            
            # Plot flow (top subplot)
            ax_flow.plot(times_second_half, flows_second_half, 'b-', linewidth=3, label='Inlet Flow')
            
            # Find current point on flow curve
            if len(times_second_half) > 1:
                # Interpolate to find current flow value
                if HAS_SCIPY_INTERP:
                    interp_func = interp1d(times_second_half, flows_second_half, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
                    current_flow = float(interp_func(time_sim))
                else:
                    current_flow = float(np.interp(time_sim, times_second_half, flows_second_half))
            else:
                current_flow = flows_second_half[0] if len(flows_second_half) > 0 else 0.0
            
            # Plot moving dot on flow
            ax_flow.plot(time_sim, current_flow, 'ro', markersize=15, label='Current')
            
            # Set flow plot labels
            ax_flow.set_ylabel('Flow (cm³/s)', fontsize=18)
            ax_flow.set_title(f'Inlet Flow and Pressure vs Time\nTime: {time_sim:.3f} s', fontsize=20)
            ax_flow.set_xlim(times_second_half[0], times_second_half[-1])
            ax_flow.set_ylim(flow_min - 0.1 * (flow_max - flow_min), flow_max + 0.1 * (flow_max - flow_min))
            ax_flow.grid(True, alpha=0.3)
            
            # Plot pressure (bottom subplot) if available
            if ax_pressure is not None and pressures_second_half is not None:
                ax_pressure.plot(pressure_times_second_half, pressures_second_half, 'r-', linewidth=3, label='Inlet Pressure')
                
                # Find current point on pressure curve
                if len(pressure_times_second_half) > 1:
                    # Interpolate to find current pressure value
                    if HAS_SCIPY_INTERP:
                        interp_func_p = interp1d(pressure_times_second_half, pressures_second_half, kind='linear',
                                                bounds_error=False, fill_value='extrapolate')
                        current_pressure = float(interp_func_p(time_sim))
                    else:
                        current_pressure = float(np.interp(time_sim, pressure_times_second_half, pressures_second_half))
                else:
                    current_pressure = pressures_second_half[0] if len(pressures_second_half) > 0 else 0.0
                
                # Plot moving dot on pressure
                ax_pressure.plot(time_sim, current_pressure, 'ro', markersize=15, label='Current')
                
                # Set pressure plot labels
                ax_pressure.set_xlabel('Time (normalized)', fontsize=18)
                ax_pressure.set_ylabel('Pressure (mmHg)', fontsize=18)
                ax_pressure.set_xlim(pressure_times_second_half[0], pressure_times_second_half[-1])
                ax_pressure.set_ylim(pressure_min - 0.1 * (pressure_max - pressure_min), 
                                    pressure_max + 0.1 * (pressure_max - pressure_min))
                ax_pressure.grid(True, alpha=0.3)
            else:
                # If no pressure data, set xlabel on flow plot
                ax_flow.set_xlabel('Time (normalized)', fontsize=18)
            
            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            plt.tight_layout()
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            image_files.append(frame_path)
        
        # Combine frames into movie
        if not image_files:
            print("Error: No frames were generated.")
            return False
        
        print(f"\nCombining {len(image_files)} frames into movie...")
        print(f"  Output: {output_path}")
        
        # Try different encoders
        encoders = [
            ('libx264', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']),
            ('h264_videotoolbox', ['-c:v', 'h264_videotoolbox', '-b:v', '5M', '-allow_sw', '1']),
            ('libopenh264', ['-c:v', 'libopenh264', '-pix_fmt', 'yuv420p']),
            ('libvpx', ['-c:v', 'libvpx', '-b:v', '2M', '-pix_fmt', 'yuv420p']),
        ]
        
        frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
        
        success = False
        for encoder_name, encoder_args in encoders:
            print(f"  Trying encoder: {encoder_name}...")
            try:
                cmd = ['ffmpeg', '-y', '-r', str(fps), '-i', frame_pattern,
                       '-vf', f'scale={resolution[0]}:{resolution[1]}',
                       '-pix_fmt', 'yuv420p'] + encoder_args + [output_path]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"  Movie created successfully using {encoder_name}!")
                    success = True
                    break
                else:
                    print(f"    Encoder {encoder_name} failed: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"    Encoder {encoder_name} timed out")
            except Exception as e:
                print(f"    Encoder {encoder_name} error: {e}")
        
        if not success:
            print("Error: All encoders failed. Cannot create movie.")
            return False
        
    finally:
        # Clean up
        print("\nCleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory: {e}")
    
    print(f"✓ Movie saved to: {output_path}")
    return True


def visualize_zerod_results_side_by_side(geometric_csv_path, calibration_input_path, 
                                         geometric_input_json_path, centerline_path, mesh_path,
                                         output_path, field='pressure',
                                         resolution=(1920, 1080), fps=5,
                                         calibrated_csv_path=None):
    """
    Create a three-panel movie comparing geometric 0D solution, 3D solution observations, and calibrated 0D solution.
    
    Args:
        geometric_csv_path: Path to geometric 0D results CSV
        calibration_input_path: Path to calibration input JSON (contains 3D solution observations)
        geometric_input_json_path: Path to geometric input JSON
        centerline_path: Path to centerline VTP file
        mesh_path: Path to 3D mesh VTU/VTP file
        output_path: Path to save output MP4
        field: Field to visualize ('pressure' or 'flow')
        resolution: (width, height) for output video
        fps: Frames per second
        calibrated_csv_path: Optional path to calibrated 0D results CSV
    """
    print("="*60)
    print("Visualizing 0D Solutions Side-by-Side")
    print("="*60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found.")
        return False
    
    # Read geometric CSV file
    print("\nReading geometric 0D results...")
    results_geo, times_geo, fields_geo = read_zerod_csv(geometric_csv_path)
    print(f"  Geometric: {len(results_geo)} vessels, {len(times_geo)} time steps")
    
    # Read calibrated CSV file if provided
    results_cal = None
    times_cal = None
    fields_cal = None
    print(f"\nChecking for calibrated CSV...")
    print(f"  calibrated_csv_path parameter: {calibrated_csv_path}")
    if calibrated_csv_path:
        print(f"  File exists: {os.path.exists(calibrated_csv_path)}")
    
    if calibrated_csv_path and os.path.exists(calibrated_csv_path):
        print("\nReading calibrated 0D results...")
        try:
            results_cal, times_cal, fields_cal = read_zerod_csv(calibrated_csv_path)
            print(f"  Calibrated: {len(results_cal)} vessels, {len(times_cal)} time steps")
            if len(times_cal) > 0:
                print(f"  Calibrated time range: [{min(times_cal):.3f}, {max(times_cal):.3f}]")
            print(f"  Calibrated fields: {fields_cal}")
            if results_cal:
                sample_vessel = list(results_cal.keys())[0]
                print(f"  Sample vessel '{sample_vessel}' has {len(results_cal[sample_vessel])} time points")
            else:
                print("  WARNING: Calibrated CSV read but results_cal is empty!")
        except Exception as e:
            print(f"  ERROR reading calibrated CSV: {e}")
            import traceback
            traceback.print_exc()
            results_cal = None
    else:
        print("\nNote: Calibrated CSV not provided or not found, showing only geometric and 3D solution")
        if calibrated_csv_path:
            print(f"  Expected path: {calibrated_csv_path}")
    
    # Read observations from calibration input
    print("\nReading 3D solution observations from calibration input...")
    observations_data, obs_times = read_observations_from_calibration(calibration_input_path, field)
    if not observations_data:
        print("Error: No observations found in calibration input file")
        return False
    
    print(f"  Found observations for {len(observations_data)} vessels")
    print(f"  Observation time points: {len(obs_times)}")
    
    # Use geometric simulation times, but only the second half
    times = sorted(times_geo)
    if len(times) > 1:
        mid_point = len(times) // 2
        times = times[mid_point:]
        print(f"  Using second half of time period: {len(times)} time steps (from t={times[0]:.3f} to t={times[-1]:.3f})")
    else:
        print(f"  Warning: Only {len(times)} time step(s), using all")
    
    # Read geometric input and centerline
    print("\nReading geometric input...")
    vessel_map = read_geometric_input(geometric_input_json_path)
    print(f"  Found {len(vessel_map)} vessels")
    
    print("\nReading centerline...")
    centerline_reader = vtk.vtkXMLPolyDataReader()
    centerline_reader.SetFileName(centerline_path)
    centerline_reader.Update()
    centerline_polydata = centerline_reader.GetOutput()
    print(f"  Centerline has {centerline_polydata.GetNumberOfPoints()} points")
    
    # Check that inlet and outlet fields exist in geometric CSV
    inlet_field_name = f"{field}_in"
    outlet_field_name = f"{field}_out"
    if inlet_field_name not in fields_geo or outlet_field_name not in fields_geo:
        print(f"Error: Fields '{inlet_field_name}' and/or '{outlet_field_name}' not found in geometric CSV.")
        return False
    
    # Get all values for color range (include geometric, 3D observations, and calibrated if available)
    print(f"\nComputing color range for {field} (geometric, 3D observations, and calibrated)...")
    all_values = []
    
    # Track where min/max values come from
    min_info = {'value': float('inf'), 'source': None, 'location': None, 'time': None, 'type': None}
    max_info = {'value': float('-inf'), 'source': None, 'location': None, 'time': None, 'type': None}
    
    # Add geometric solution values (already filtered to second half via times)
    # Convert pressure from dynes/cm^2 to mmHg if field is pressure
    for location in results_geo:
        for time in times:
            if inlet_field_name in results_geo[location][time]:
                val = results_geo[location][time][inlet_field_name]
                if field == 'pressure':
                    val = val / 1333.0  # Convert to mmHg
                all_values.append(val)
                if val < min_info['value']:
                    min_info = {'value': val, 'source': 'geometric', 'location': location, 'time': time, 'type': 'inlet'}
                if val > max_info['value']:
                    max_info = {'value': val, 'source': 'geometric', 'location': location, 'time': time, 'type': 'inlet'}
            if outlet_field_name in results_geo[location][time]:
                val = results_geo[location][time][outlet_field_name]
                if field == 'pressure':
                    val = val / 1333.0  # Convert to mmHg
                all_values.append(val)
                if val < min_info['value']:
                    min_info = {'value': val, 'source': 'geometric', 'location': location, 'time': time, 'type': 'outlet'}
                if val > max_info['value']:
                    max_info = {'value': val, 'source': 'geometric', 'location': location, 'time': time, 'type': 'outlet'}
    
    # Add observation values (filter to second half to match geometric solution)
    # Observations are normalized to [0, 1], so second half is [0.5, 1.0]
    obs_mid_point = len(obs_times) // 2 if len(obs_times) > 1 else 0
    for vessel_name, obs in observations_data.items():
        if obs['inlet'] is not None and len(obs['inlet']) > obs_mid_point:
            # Only use second half of observation values
            for idx, val in enumerate(obs['inlet'][obs_mid_point:], start=obs_mid_point):
                all_values.append(val)
                obs_time = obs_times[idx] if idx < len(obs_times) else obs_times[-1]
                if val < min_info['value']:
                    min_info = {'value': val, 'source': '3D_observations', 'location': vessel_name, 'time': obs_time, 'type': 'inlet'}
                if val > max_info['value']:
                    max_info = {'value': val, 'source': '3D_observations', 'location': vessel_name, 'time': obs_time, 'type': 'inlet'}
        if obs['outlet'] is not None and len(obs['outlet']) > obs_mid_point:
            # Only use second half of observation values
            for idx, val in enumerate(obs['outlet'][obs_mid_point:], start=obs_mid_point):
                all_values.append(val)
                obs_time = obs_times[idx] if idx < len(obs_times) else obs_times[-1]
                if val < min_info['value']:
                    min_info = {'value': val, 'source': '3D_observations', 'location': vessel_name, 'time': obs_time, 'type': 'outlet'}
                if val > max_info['value']:
                    max_info = {'value': val, 'source': '3D_observations', 'location': vessel_name, 'time': obs_time, 'type': 'outlet'}
    
    # Add calibrated solution values if available (filter to second half)
    # Convert pressure from dynes/cm^2 to mmHg if field is pressure
    if results_cal is not None:
        for location in results_cal:
            for time in times:
                if inlet_field_name in results_cal[location][time]:
                    val = results_cal[location][time][inlet_field_name]
                    if field == 'pressure':
                        val = val / 1333.0  # Convert to mmHg
                    all_values.append(val)
                    if val < min_info['value']:
                        min_info = {'value': val, 'source': 'calibrated', 'location': location, 'time': time, 'type': 'inlet'}
                    if val > max_info['value']:
                        max_info = {'value': val, 'source': 'calibrated', 'location': location, 'time': time, 'type': 'inlet'}
                if outlet_field_name in results_cal[location][time]:
                    val = results_cal[location][time][outlet_field_name]
                    if field == 'pressure':
                        val = val / 1333.0  # Convert to mmHg
                    all_values.append(val)
                    if val < min_info['value']:
                        min_info = {'value': val, 'source': 'calibrated', 'location': location, 'time': time, 'type': 'outlet'}
                    if val > max_info['value']:
                        max_info = {'value': val, 'source': 'calibrated', 'location': location, 'time': time, 'type': 'outlet'}
    
    if not all_values:
        print(f"Error: No values found for {field}")
        return False
    
    # Use full range (min to max)
    color_range = [np.min(all_values), np.max(all_values)]
    print(f"  Color range (full): [{color_range[0]:.2f}, {color_range[1]:.2f}]")
    print(f"  Minimum {field} ({min_info['value']:.2f}) found in: {min_info['source']}, location: {min_info['location']}, time: {min_info['time']:.3f}, type: {min_info['type']}")
    print(f"  Maximum {field} ({max_info['value']:.2f}) found in: {max_info['source']}, location: {max_info['location']}, time: {max_info['time']:.3f}, type: {max_info['type']}")
    
    # Get node locations for all vessels
    print("\nMapping vessels to 3D locations...")
    node_locations = {}
    all_vessels = set(results_geo.keys()) | set(observations_data.keys())
    for vessel_name in all_vessels:
        inlet_loc, outlet_loc = get_node_locations_for_vessel(centerline_polydata, vessel_name, vessel_map)
        if inlet_loc is not None and outlet_loc is not None:
            node_locations[vessel_name] = {
                'inlet': inlet_loc,
                'outlet': outlet_loc
            }
    
    print(f"  Mapped {len(node_locations)} vessels to 3D locations")
    
    # Create vasculature actor
    print("\nLoading 3D vasculature mesh...")
    vasculature_actor = create_vasculature_actor(mesh_path)
    
    # Get mesh bounds
    mesh_reader = vtk.vtkXMLUnstructuredGridReader() if mesh_path.endswith('.vtu') else vtk.vtkXMLPolyDataReader()
    mesh_reader.SetFileName(mesh_path)
    mesh_reader.Update()
    mesh_bounds = mesh_reader.GetOutput().GetBounds()
    mesh_size = max([mesh_bounds[1] - mesh_bounds[0],
                     mesh_bounds[3] - mesh_bounds[2],
                     mesh_bounds[5] - mesh_bounds[4]])
    node_radius = mesh_size * 0.02
    print(f"  Node radius: {node_radius:.3f}")
    
    # Create three renderers side by side
    # Each renderer will be one-third the width
    render_width = resolution[0] // 3
    render_height = resolution[1]
    
    # Determine if we have calibrated results
    has_calibrated = results_cal is not None
    
    print(f"\n  has_calibrated flag: {has_calibrated}")
    if has_calibrated:
        print(f"  Will create 3-panel layout (geometric, 3D, calibrated)")
    else:
        print(f"  Will create 2-panel layout (geometric, 3D)")
    
    if has_calibrated:
        # Three panels: geometric, 3D solution, calibrated
        # Heavily overlapping viewports to eliminate white space
        renderer_geo = vtk.vtkRenderer()
        renderer_geo.SetBackground(1.0, 1.0, 1.0)
        renderer_geo.SetViewport(0.0, 0.0, 0.34, 1.0)  # Left panel
        renderer_geo.AddActor(vasculature_actor)
        
        renderer_3d = vtk.vtkRenderer()
        renderer_3d.SetBackground(1.0, 1.0, 1.0)
        renderer_3d.SetViewport(0.33, 0.0, 0.67, 1.0)  # Middle panel (overlaps significantly)
        vasculature_actor_3d = create_vasculature_actor(mesh_path)
        renderer_3d.AddActor(vasculature_actor_3d)
        
        renderer_cal = vtk.vtkRenderer()
        renderer_cal.SetBackground(1.0, 1.0, 1.0)
        renderer_cal.SetViewport(0.66, 0.0, 1.0, 1.0)  # Right panel (overlaps significantly)
        vasculature_actor_cal = create_vasculature_actor(mesh_path)
        renderer_cal.AddActor(vasculature_actor_cal)
    else:
        # Two panels: geometric, 3D solution (fallback to original layout)
        renderer_geo = vtk.vtkRenderer()
        renderer_geo.SetBackground(1.0, 1.0, 1.0)
        renderer_geo.SetViewport(0.0, 0.0, 0.5, 1.0)  # Left half
        renderer_geo.AddActor(vasculature_actor)
        
        renderer_3d = vtk.vtkRenderer()
        renderer_3d.SetBackground(1.0, 1.0, 1.0)
        renderer_3d.SetViewport(0.5, 0.0, 1.0, 1.0)  # Right half
        vasculature_actor_3d = create_vasculature_actor(mesh_path)
        renderer_3d.AddActor(vasculature_actor_3d)
        
        renderer_cal = None
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer_geo)
    render_window.AddRenderer(renderer_3d)
    if renderer_cal is not None:
        render_window.AddRenderer(renderer_cal)
    render_window.SetSize(resolution[0], resolution[1])
    render_window.SetOffScreenRendering(1)
    render_window.SetShowWindow(False)
    
    try:
        render_window.Initialize()
    except Exception as e:
        print(f"Error: Render window initialization failed: {e}")
        return False
    
    # Create lookup table
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(color_range[0], color_range[1])
    lut.Build()
    # Reverse colormap (red high, blue low)
    num_colors = lut.GetNumberOfTableValues()
    temp_colors = []
    for i in range(num_colors):
        r, g, b, a = lut.GetTableValue(i)
        temp_colors.append((r, g, b, a))
    for i in range(num_colors):
        r, g, b, a = temp_colors[num_colors - 1 - i]
        lut.SetTableValue(i, r, g, b, a)
    
    # Create a single horizontal colorbar for all panels at the bottom
    # Add it to the middle renderer (3D solution) so it spans across all panels
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetTitle(f"{field.capitalize()}")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetLookupTable(lut)
    # Set to horizontal orientation
    scalar_bar.SetOrientationToHorizontal()
    # Position at the bottom center of the screen (position is relative to viewport)
    # For middle renderer viewport (0.33 to 0.67), position to span across all panels
    scalar_bar.SetPosition(0.0, 0.05)  # Bottom of viewport
    # Set width to span across the middle panel (and extend into adjacent panels)
    scalar_bar.SetWidth(1.0)  # Full width of viewport
    # Set height for horizontal colorbar
    scalar_bar.SetMaximumHeightInPixels(int(resolution[1] * 0.08))  # 8% of screen height for horizontal bar
    
    # Increase font sizes by 3x (was 32/28, now 96/84 for exactly 3x)
    title_prop = scalar_bar.GetTitleTextProperty()
    title_prop.SetFontSize(96)  # 3x the original 32
    title_prop.BoldOn()
    title_prop.ShadowOff()
    title_prop.SetColor(0, 0, 0)  # Ensure text is black and visible
    scalar_bar.SetTitleTextProperty(title_prop)
    
    # Set label font properties - triple the current size (84 * 3 = 252)
    label_prop = scalar_bar.GetLabelTextProperty()
    label_prop.SetFontSize(252)  # 3x the current 84 (9x the original 28)
    label_prop.BoldOn()
    label_prop.ShadowOff()
    label_prop.SetColor(0, 0, 0)  # Ensure text is black and visible
    scalar_bar.SetLabelTextProperty(label_prop)
    
    # Add to the middle renderer (3D solution) so it appears at bottom center
    # This way it spans across all three panels visually
    renderer_3d.AddActor2D(scalar_bar)
    
    # Set up cameras with increased zoom to fill panels better and reduce vertical white space
    renderer_geo.ResetCamera()
    renderer_3d.ResetCamera()
    camera_geo = renderer_geo.GetActiveCamera()
    camera_3d = renderer_3d.GetActiveCamera()
    camera_geo.Zoom(1.4)  # Increased zoom further to reduce vertical white space (was 1.2)
    camera_3d.Zoom(1.4)  # Increased zoom further to reduce vertical white space (was 1.2)
    # Sync camera positions
    camera_3d.SetPosition(camera_geo.GetPosition())
    camera_3d.SetFocalPoint(camera_geo.GetFocalPoint())
    camera_3d.SetViewUp(camera_geo.GetViewUp())
    
    if has_calibrated:
        renderer_cal.ResetCamera()
        camera_cal = renderer_cal.GetActiveCamera()
        camera_cal.Zoom(1.4)  # Increased zoom further to reduce vertical white space (was 1.2)
        # Sync camera positions
        camera_cal.SetPosition(camera_geo.GetPosition())
        camera_cal.SetFocalPoint(camera_geo.GetFocalPoint())
        camera_cal.SetViewUp(camera_geo.GetViewUp())
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='zerod_viz_')
    print(f"\nRendering {len(times)} frames...")
    print(f"  Temporary directory: {temp_dir}")
    
    # Extract inlet flow data for the plot
    flow_times, flow_values = extract_inlet_flow_from_observations(calibration_input_path)
    # Flow plot overlay removed - using standalone movie instead
    
    if flow_times is not None and flow_values is not None:
        # Calculate flow range for plot (second half only)
        flow_mid_point = len(flow_values) // 2
        flow_values_second_half = flow_values[flow_mid_point:]
        flow_range_plot = [np.min(flow_values_second_half), np.max(flow_values_second_half)]
        print(f"\nInlet flow plot: {len(flow_values_second_half)} points, range [{flow_range_plot[0]:.2f}, {flow_range_plot[1]:.2f}]")
    else:
        flow_times = None
        flow_values = None
        flow_range_plot = None
    
    try:
        text_actor_geo = None
        text_actor_3d = None
        text_actor_cal = None
        
        # Store previous 3D solution values (since 3D solution updates every 2nd frame)
        prev_3d_values = {}  # {vessel_name: {'inlet': value, 'outlet': value}}
        
        image_files = []
        for idx, time in enumerate(times):
            print(f"  Frame {idx+1}/{len(times)}: t={time:.3f}")
            
            # Clear previous node actors
            actors_to_remove_geo = []
            actors_to_remove_3d = []
            actors_to_remove_cal = []
            for actor in renderer_geo.GetActors():
                if actor != vasculature_actor:
                    actors_to_remove_geo.append(actor)
            for actor in renderer_3d.GetActors():
                if actor != vasculature_actor_3d:
                    actors_to_remove_3d.append(actor)
            if has_calibrated:
                for actor in renderer_cal.GetActors():
                    if actor != vasculature_actor_cal:
                        actors_to_remove_cal.append(actor)
            
            for actor in actors_to_remove_geo:
                renderer_geo.RemoveActor(actor)
            for actor in actors_to_remove_3d:
                renderer_3d.RemoveActor(actor)
            if has_calibrated:
                for actor in actors_to_remove_cal:
                    renderer_cal.RemoveActor(actor)
            
            # Remove previous text actors
            if text_actor_geo is not None:
                renderer_geo.RemoveActor2D(text_actor_geo)
            if text_actor_3d is not None:
                renderer_3d.RemoveActor2D(text_actor_3d)
            if text_actor_cal is not None and has_calibrated:
                renderer_cal.RemoveActor2D(text_actor_cal)
            
            # Flow plot overlay removed - using standalone movie instead
            
            # Add geometric solution nodes (updates every frame)
            # Convert pressure from dynes/cm^2 to mmHg if field is pressure
            for vessel_name, locations in node_locations.items():
                if vessel_name in results_geo and time in results_geo[vessel_name]:
                    if inlet_field_name in results_geo[vessel_name][time]:
                        inlet_value = results_geo[vessel_name][time][inlet_field_name]
                        if field == 'pressure':
                            inlet_value = inlet_value / 1333.0  # Convert to mmHg
                        inlet_actor = create_node_actor(locations['inlet'], inlet_value, color_range, field, node_radius, lut)
                        renderer_geo.AddActor(inlet_actor)
                    if outlet_field_name in results_geo[vessel_name][time]:
                        outlet_value = results_geo[vessel_name][time][outlet_field_name]
                        if field == 'pressure':
                            outlet_value = outlet_value / 1333.0  # Convert to mmHg
                        outlet_actor = create_node_actor(locations['outlet'], outlet_value, color_range, field, node_radius, lut)
                        renderer_geo.AddActor(outlet_actor)
            
            # Add 3D solution observation nodes (from calibration input)
            # Update only every second frame (since 3D solution has 2x larger timestep)
            update_3d = (idx % 2 == 0) or (idx == 0)
            
            if update_3d:
                # Normalize simulation time to [0.5, 1.0] for interpolation with observations
                # Since we're using the second half of simulation times, map to second half of observation range
                if len(times) > 1:
                    # Map time from [times[0], times[-1]] to [0.5, 1.0]
                    sim_time_normalized = 0.5 + 0.5 * ((time - times[0]) / (times[-1] - times[0])) if (times[-1] - times[0]) > 0 else 0.5
                else:
                    sim_time_normalized = 0.5
                sim_time_normalized = max(0.5, min(1.0, sim_time_normalized))
                
                for vessel_name, locations in node_locations.items():
                    if vessel_name in observations_data:
                        obs = observations_data[vessel_name]
                        
                        # Interpolate inlet observation
                        if obs['inlet'] is not None and len(obs['inlet']) > 0:
                            if HAS_SCIPY_INTERP:
                                interp_func = interp1d(obs_times, obs['inlet'], kind='linear',
                                                      bounds_error=False, fill_value='extrapolate')
                                obs_inlet_value = float(interp_func(sim_time_normalized))
                            else:
                                obs_inlet_value = float(np.interp(sim_time_normalized, obs_times, obs['inlet']))
                            
                            # Store for next frame
                            if vessel_name not in prev_3d_values:
                                prev_3d_values[vessel_name] = {}
                            prev_3d_values[vessel_name]['inlet'] = obs_inlet_value
                            
                            inlet_actor = create_node_actor(locations['inlet'], obs_inlet_value, color_range, field, node_radius, lut)
                            renderer_3d.AddActor(inlet_actor)
                        
                        # Interpolate outlet observation
                        if obs['outlet'] is not None and len(obs['outlet']) > 0:
                            if HAS_SCIPY_INTERP:
                                interp_func = interp1d(obs_times, obs['outlet'], kind='linear',
                                                      bounds_error=False, fill_value='extrapolate')
                                obs_outlet_value = float(interp_func(sim_time_normalized))
                            else:
                                obs_outlet_value = float(np.interp(sim_time_normalized, obs_times, obs['outlet']))
                            
                            # Store for next frame
                            if vessel_name not in prev_3d_values:
                                prev_3d_values[vessel_name] = {}
                            prev_3d_values[vessel_name]['outlet'] = obs_outlet_value
                            
                            outlet_actor = create_node_actor(locations['outlet'], obs_outlet_value, color_range, field, node_radius, lut)
                            renderer_3d.AddActor(outlet_actor)
            else:
                # Use previous 3D solution values (don't update this frame)
                for vessel_name, locations in node_locations.items():
                    if vessel_name in prev_3d_values:
                        prev_vals = prev_3d_values[vessel_name]
                        
                        if 'inlet' in prev_vals:
                            inlet_actor = create_node_actor(locations['inlet'], prev_vals['inlet'], color_range, field, node_radius, lut)
                            renderer_3d.AddActor(inlet_actor)
                        
                        if 'outlet' in prev_vals:
                            outlet_actor = create_node_actor(locations['outlet'], prev_vals['outlet'], color_range, field, node_radius, lut)
                            renderer_3d.AddActor(outlet_actor)
            
            # Add calibrated solution nodes if available
            # Convert pressure from dynes/cm^2 to mmHg if field is pressure
            if has_calibrated:
                calibrated_nodes_added = 0
                for vessel_name, locations in node_locations.items():
                    if vessel_name in results_cal:
                        # Interpolate calibrated values to match current time
                        # Find closest time or interpolate
                        cal_times = sorted(results_cal[vessel_name].keys())
                        if len(cal_times) > 0:
                            # Check if exact time match exists
                            if time in results_cal[vessel_name]:
                                # Exact match
                                cal_data = results_cal[vessel_name][time]
                            elif len(cal_times) == 1:
                                # Only one time point, use it
                                cal_data = results_cal[vessel_name][cal_times[0]]
                            else:
                                # Interpolate
                                if time <= cal_times[0]:
                                    cal_data = results_cal[vessel_name][cal_times[0]]
                                elif time >= cal_times[-1]:
                                    cal_data = results_cal[vessel_name][cal_times[-1]]
                                else:
                                    # Find surrounding times
                                    for i in range(len(cal_times) - 1):
                                        if cal_times[i] <= time <= cal_times[i+1]:
                                            t0, t1 = cal_times[i], cal_times[i+1]
                                            alpha = (time - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
                                            
                                            # Interpolate all fields
                                            data0 = results_cal[vessel_name][t0]
                                            data1 = results_cal[vessel_name][t1]
                                            cal_data = {}
                                            for key in set(data0.keys()) | set(data1.keys()):
                                                val0 = data0.get(key, 0.0)
                                                val1 = data1.get(key, 0.0)
                                                cal_data[key] = val0 + alpha * (val1 - val0)
                                            break
                                    else:
                                        # Fallback: use closest time
                                        closest_time = min(cal_times, key=lambda t: abs(t - time))
                                        cal_data = results_cal[vessel_name][closest_time]
                            
                            # Create actors from interpolated data
                            if inlet_field_name in cal_data:
                                inlet_value = cal_data[inlet_field_name]
                                if field == 'pressure':
                                    inlet_value = inlet_value / 1333.0  # Convert to mmHg
                                inlet_actor = create_node_actor(locations['inlet'], inlet_value, color_range, field, node_radius, lut)
                                renderer_cal.AddActor(inlet_actor)
                                calibrated_nodes_added += 1
                            if outlet_field_name in cal_data:
                                outlet_value = cal_data[outlet_field_name]
                                if field == 'pressure':
                                    outlet_value = outlet_value / 1333.0  # Convert to mmHg
                                outlet_actor = create_node_actor(locations['outlet'], outlet_value, color_range, field, node_radius, lut)
                                renderer_cal.AddActor(outlet_actor)
                                calibrated_nodes_added += 1
                
                # Debug output for first frame
                if idx == 0:
                    print(f"    Added {calibrated_nodes_added} calibrated nodes for frame 0")
            
            # Add text
            text_actor_geo = vtk.vtkTextActor()
            text_actor_geo.SetInput(f"Geometric\nTime: {time:.3f} s")
            text_actor_geo.SetPosition(10, render_height - 60)
            text_actor_geo.GetTextProperty().SetFontSize(20)
            text_actor_geo.GetTextProperty().SetColor(0, 0, 0)
            renderer_geo.AddActor2D(text_actor_geo)
            
            text_actor_3d = vtk.vtkTextActor()
            text_actor_3d.SetInput(f"3D Solution\nTime: {time:.3f} s")
            text_actor_3d.SetPosition(10, render_height - 60)
            text_actor_3d.GetTextProperty().SetFontSize(20)
            text_actor_3d.GetTextProperty().SetColor(0, 0, 0)
            renderer_3d.AddActor2D(text_actor_3d)
            
            if has_calibrated:
                text_actor_cal = vtk.vtkTextActor()
                text_actor_cal.SetInput(f"Calibrated\nTime: {time:.3f} s")
                text_actor_cal.SetPosition(10, render_height - 60)
                text_actor_cal.GetTextProperty().SetFontSize(20)
                text_actor_cal.GetTextProperty().SetColor(0, 0, 0)
                renderer_cal.AddActor2D(text_actor_cal)
            
            # Flow plot overlay removed - using standalone movie instead
            
            # Render
            try:
                render_window.Render()
            except Exception as e:
                print(f"    Warning: Render failed for time {time}: {e}")
                continue
            
            # Save frame
            try:
                window_to_image = vtk.vtkWindowToImageFilter()
                window_to_image.SetInput(render_window)
                window_to_image.Update()
                
                writer = vtk.vtkPNGWriter()
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                writer.SetFileName(frame_path)
                writer.SetInputConnection(window_to_image.GetOutputPort())
                writer.Write()
                
                image_files.append(frame_path)
            except Exception as e:
                print(f"    Warning: Failed to capture frame for time {time}: {e}")
                continue
        
        # Combine frames into movie
        if not image_files:
            print("Error: No frames were generated.")
            return False
        
        print(f"\nCombining {len(image_files)} frames into movie...")
        print(f"  Output: {output_path}")
        
        # Try different encoders
        encoders = [
            ('libx264', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']),
            ('h264_videotoolbox', ['-c:v', 'h264_videotoolbox', '-b:v', '5M', '-allow_sw', '1']),
            ('libopenh264', ['-c:v', 'libopenh264', '-pix_fmt', 'yuv420p']),
            ('libvpx', ['-c:v', 'libvpx', '-b:v', '2M', '-pix_fmt', 'yuv420p']),
        ]
        
        frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
        success = False
        
        for encoder_name, encoder_args in encoders:
            print(f"  Trying encoder: {encoder_name}...")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
            ] + encoder_args + [output_path]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                print(f"  Movie created successfully using {encoder_name}!")
                success = True
                break
            except subprocess.CalledProcessError as e:
                if 'encoder' in e.stderr.lower() or 'Unknown encoder' in e.stderr:
                    print(f"    Encoder {encoder_name} not available, trying next...")
                    continue
                else:
                    print(f"    Error with {encoder_name}: {e.stderr[:200]}")
                    continue
        
        if not success:
            print("Error: All encoders failed.")
            return False
        
        return True
        
    finally:
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def visualize_zerod_results(csv_path, geometric_input_path, centerline_path, mesh_path,
                            output_path, field='pressure', field_component='in',
                            resolution=(1920, 1080), fps=5, calibration_input_path=None):
    """
    Create a movie visualizing 0D simulation results.
    
    Args:
        csv_path: Path to 0D results CSV
        geometric_input_path: Path to geometric input JSON
        centerline_path: Path to centerline VTP file
        mesh_path: Path to 3D mesh VTU/VTP file
        output_path: Path to save output MP4
        field: Field to visualize ('pressure' or 'flow')
        field_component: Component ('in' or 'out')
        resolution: (width, height) for output video
        fps: Frames per second
    """
    print("="*60)
    print("Visualizing 0D Simulation Results")
    print("="*60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found.")
        return False
    
    # Read data
    print("\nReading 0D results...")
    results, times, fields = read_zerod_csv(csv_path)
    print(f"  Found {len(results)} vessel locations")
    print(f"  Found {len(times)} time steps")
    print(f"  Available fields: {fields}")
    
    # Filter to only the second half of the time period
    times = sorted(times)
    if len(times) > 1:
        mid_point = len(times) // 2
        times = times[mid_point:]
        print(f"  Using second half of time period: {len(times)} time steps (from t={times[0]:.3f} to t={times[-1]:.3f})")
    else:
        print(f"  Warning: Only {len(times)} time step(s), using all")
    
    print("\nReading geometric input...")
    vessel_map = read_geometric_input(geometric_input_path)
    print(f"  Found {len(vessel_map)} vessels")
    
    print("\nReading centerline...")
    centerline_reader = vtk.vtkXMLPolyDataReader()
    centerline_reader.SetFileName(centerline_path)
    centerline_reader.Update()
    centerline_polydata = centerline_reader.GetOutput()
    print(f"  Centerline has {centerline_polydata.GetNumberOfPoints()} points")
    
    # Check that both inlet and outlet fields exist
    inlet_field_name = f"{field}_in"
    outlet_field_name = f"{field}_out"
    if inlet_field_name not in fields or outlet_field_name not in fields:
        print(f"Error: Fields '{inlet_field_name}' and/or '{outlet_field_name}' not found in CSV.")
        print(f"Available fields: {fields}")
        return False
    
    # Get all values for color range (include both inlet and outlet, and observations if available)
    print(f"\nComputing color range for {field} (inlet and outlet)...")
    all_values = []
    inlet_field_name = f"{field}_in"
    outlet_field_name = f"{field}_out"
    
    for location in results:
        for time in times:
            if inlet_field_name in results[location][time]:
                all_values.append(results[location][time][inlet_field_name])
            if outlet_field_name in results[location][time]:
                all_values.append(results[location][time][outlet_field_name])
    
    # Also include observation values in color range if available
    if calibration_input_path and os.path.exists(calibration_input_path):
        obs_data, _ = read_observations_from_calibration(calibration_input_path, field)
        if obs_data:
            for vessel_name, obs in obs_data.items():
                if obs['inlet'] is not None:
                    all_values.extend(obs['inlet'])
                if obs['outlet'] is not None:
                    all_values.extend(obs['outlet'])
    
    if not all_values:
        print(f"Error: No values found for {field}")
        return False
    
    # Use full range (min to max) instead of percentiles
    color_range = [np.min(all_values), np.max(all_values)]
    print(f"  Color range (full): [{color_range[0]:.2f}, {color_range[1]:.2f}]")
    print(f"    Min value: {color_range[0]:.6f}")
    print(f"    Max value: {color_range[1]:.6f}")
    print(f"    Range span: {color_range[1] - color_range[0]:.6f}")
    
    # Get node locations for all vessels (inlet and outlet)
    print("\nMapping vessels to 3D locations...")
    node_locations = {}
    for vessel_name in results.keys():
        inlet_loc, outlet_loc = get_node_locations_for_vessel(centerline_polydata, vessel_name, vessel_map)
        if inlet_loc is not None and outlet_loc is not None:
            node_locations[vessel_name] = {
                'inlet': inlet_loc,
                'outlet': outlet_loc
            }
        else:
            print(f"  Warning: Could not find locations for {vessel_name}")
    
    print(f"  Mapped {len(node_locations)} vessels to 3D locations (inlet and outlet)")
    
    # Create vasculature actor
    print("\nLoading 3D vasculature mesh...")
    vasculature_actor = create_vasculature_actor(mesh_path)
    
    # Get mesh bounds to determine appropriate sphere size
    mesh_reader = vtk.vtkXMLUnstructuredGridReader() if mesh_path.endswith('.vtu') else vtk.vtkXMLPolyDataReader()
    mesh_reader.SetFileName(mesh_path)
    mesh_reader.Update()
    mesh_bounds = mesh_reader.GetOutput().GetBounds()
    mesh_size = max([mesh_bounds[1] - mesh_bounds[0],
                     mesh_bounds[3] - mesh_bounds[2],
                     mesh_bounds[5] - mesh_bounds[4]])
    node_radius = mesh_size * 0.02  # 2% of mesh size
    print(f"  Mesh bounds: {mesh_bounds}")
    print(f"  Node radius: {node_radius:.3f}")
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(vasculature_actor)
    
    # Set up camera to view entire scene
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Zoom(0.9)  # Slight zoom out
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(resolution[0], resolution[1])
    render_window.SetOffScreenRendering(1)
    render_window.SetShowWindow(False)
    
    try:
        render_window.Initialize()
    except Exception as e:
        print(f"Error: Render window initialization failed: {e}")
        print("This might be due to X11/OpenGL issues.")
        print("Try running with: xvfb-run -a python3 visualize_zerod_results.py ...")
        return False
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='zerod_viz_')
    print(f"\nRendering {len(times)} frames...")
    print(f"  Temporary directory: {temp_dir}")
    
    try:
        # Create lookup table for colorbar (once)
        # Reverse the colormap so red is high and blue is low
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(color_range[0], color_range[1])
        # Build default colormap (blue to red)
        lut.Build()
        # Reverse the colormap: swap colors so red is at max and blue is at min
        num_colors = lut.GetNumberOfTableValues()
        temp_colors = []
        for i in range(num_colors):
            r, g, b, a = lut.GetTableValue(i)
            temp_colors.append((r, g, b, a))
        # Reverse: assign colors in reverse order
        for i in range(num_colors):
            r, g, b, a = temp_colors[num_colors - 1 - i]
            lut.SetTableValue(i, r, g, b, a)
        
        # Verify lookup table range
        lut_range = lut.GetTableRange()
        print(f"  Lookup table range: [{lut_range[0]:.6f}, {lut_range[1]:.6f}]")
        
        # Create colorbar (once)
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetTitle(f"{field.capitalize()} (inlet & outlet)")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetLookupTable(lut)
        # Increase font sizes significantly
        title_prop = scalar_bar.GetTitleTextProperty()
        title_prop.SetFontSize(32)
        title_prop.SetColor(0, 0, 0)  # Black text
        scalar_bar.SetTitleTextProperty(title_prop)
        
        label_prop = scalar_bar.GetLabelTextProperty()
        label_prop.SetFontSize(28)
        label_prop.SetColor(0, 0, 0)  # Black text
        scalar_bar.SetLabelTextProperty(label_prop)
        
        renderer.AddActor2D(scalar_bar)
        
        # Track text actor for removal
        text_actor = None
        
        image_files = []
        for idx, time in enumerate(times):
            print(f"  Frame {idx+1}/{len(times)}: t={time:.3f}")
            
            # Clear previous node actors
            actors_to_remove = []
            for actor in renderer.GetActors():
                if actor != vasculature_actor:
                    actors_to_remove.append(actor)
            for actor in actors_to_remove:
                renderer.RemoveActor(actor)
            
            # Remove previous text actor
            if text_actor is not None:
                renderer.RemoveActor2D(text_actor)
            
            # Add nodes for this timestep (inlet and outlet for each vessel)
            for vessel_name, locations in node_locations.items():
                if vessel_name in results and time in results[vessel_name]:
                    # Get inlet value
                    if inlet_field_name in results[vessel_name][time]:
                        inlet_value = results[vessel_name][time][inlet_field_name]
                        inlet_actor = create_node_actor(locations['inlet'], inlet_value, color_range, field, node_radius, lut)
                        renderer.AddActor(inlet_actor)
                    
                    # Get outlet value
                    if outlet_field_name in results[vessel_name][time]:
                        outlet_value = results[vessel_name][time][outlet_field_name]
                        outlet_actor = create_node_actor(locations['outlet'], outlet_value, color_range, field, node_radius, lut)
                        renderer.AddActor(outlet_actor)
            
            # Add text with time
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"Time: {time:.3f} s\nField: {field} (inlet & outlet nodes)")
            text_actor.SetPosition(10, resolution[1] - 60)
            text_actor.GetTextProperty().SetFontSize(20)
            text_actor.GetTextProperty().SetColor(0, 0, 0)
            renderer.AddActor2D(text_actor)
            
            # Render
            try:
                render_window.Render()
            except Exception as e:
                print(f"    Warning: Render failed for time {time}: {e}")
                continue
            
            # Save frame
            try:
                window_to_image = vtk.vtkWindowToImageFilter()
                window_to_image.SetInput(render_window)
                window_to_image.Update()
                
                writer = vtk.vtkPNGWriter()
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                writer.SetFileName(frame_path)
                writer.SetInputConnection(window_to_image.GetOutputPort())
                writer.Write()
                
                image_files.append(frame_path)
            except Exception as e:
                print(f"    Warning: Failed to capture frame for time {time}: {e}")
                continue
        
        # Combine frames into movie
        if not image_files:
            print("Error: No frames were generated.")
            return False
        
        print(f"\nCombining {len(image_files)} frames into movie...")
        print(f"  Output: {output_path}")
        
        # Try different encoders in order of preference
        # libx264 (best quality, but requires GPL license)
        # h264_videotoolbox (macOS hardware acceleration)
        # libopenh264 (open source, often available in conda)
        # libvpx (VP8/VP9, always works but larger files)
        encoders = [
            ('libx264', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']),
            ('h264_videotoolbox', ['-c:v', 'h264_videotoolbox', '-b:v', '5M', '-allow_sw', '1']),
            ('libopenh264', ['-c:v', 'libopenh264', '-pix_fmt', 'yuv420p']),
            ('libvpx', ['-c:v', 'libvpx', '-b:v', '2M', '-pix_fmt', 'yuv420p']),
        ]
        
        # Use pattern matching for input files
        frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
        
        success = False
        for encoder_name, encoder_args in encoders:
            print(f"  Trying encoder: {encoder_name}...")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
            ] + encoder_args + [output_path]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                print(f"  Movie created successfully using {encoder_name}!")
                success = True
                break
            except subprocess.CalledProcessError as e:
                # If it's just an encoder error, try the next one
                if 'encoder' in e.stderr.lower() or 'Unknown encoder' in e.stderr:
                    print(f"    Encoder {encoder_name} not available, trying next...")
                    continue
                else:
                    # Other error, show it
                    print(f"    Error with {encoder_name}: {e.stderr[:200]}")
                    continue
        
        if not success:
            print("Error: All encoders failed. Available encoders may be limited.")
            print("Try installing ffmpeg with libx264 support, or use a different ffmpeg installation.")
            return False
        
        return True
        
    finally:
        # Cleanup
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 0D simulation results as movies for pressure and flow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create movies for set_3, tree_006
  python3 util/visualizations/visualize_zerod_results.py --set-name set_3 --geo-name tree_006

  # Create movies with custom output directory
  python3 util/visualizations/visualize_zerod_results.py --set-name set_3 --geo-name tree_006 \\
    --output-dir movies
        """
    )
    
    parser.add_argument('--set-name', required=True,
                       help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', required=True,
                       help='Geometry name (e.g., tree_006)')
    parser.add_argument('--data-dir', default='data',
                       help='Base data directory (default: data)')
    parser.add_argument('--output-dir', default='data/zeroD',
                       help='Output directory for 0D files (default: data/zeroD)')
    parser.add_argument('--movies-dir', default=None,
                       help='Directory to save movies (default: same as output-dir)')
    parser.add_argument('--component', choices=['in', 'out'], default='in',
                       help='Component to visualize (default: in)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Video resolution (default: 1920 1080)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second (default: 5)')
    
    args = parser.parse_args()
    
    # Find all required files
    print("="*60)
    print("Finding required files...")
    print("="*60)
    files = find_files(args.set_name, args.geo_name, args.data_dir, args.output_dir)
    
    if files is None:
        print("\n✗ Failed to find required files")
        sys.exit(1)
    
    print(f"  CSV files found: {len(files['csv_files'])}")
    for csv_type, csv_path in files['csv_files']:
        print(f"    - {csv_type}: {csv_path}")
    print(f"  Geometric input: {files['geometric_input']}")
    print(f"  Centerline: {files['centerline']}")
    print(f"  Mesh: {files['mesh']}")
    
    # Check for calibration input
    calibration_input = files.get('calibration_input')
    if calibration_input:
        print(f"  Calibration input: {calibration_input}")
    else:
        print(f"  Calibration input: Not found")
    
    # Determine output directory for movies
    if args.movies_dir is None:
        movies_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
    else:
        movies_dir = args.movies_dir
    os.makedirs(movies_dir, exist_ok=True)
    
    # Create side-by-side movies for geometric and calibrated solutions
    fields = ['pressure', 'flow']
    success_count = 0
    
    # Check if geometric CSV and calibration input exist
    geometric_csv = None
    
    # Create standalone flow plot movie if calibration input exists
    if calibration_input:
        if not HAS_MATPLOTLIB:
            print("\nSkipping standalone flow plot movie (matplotlib not available)")
            print("  To enable flow plot movies, install matplotlib: pip install matplotlib")
        else:
            print("\n" + "="*60)
            print("Creating standalone flow plot movie")
            print("="*60)
            flow_plot_output = os.path.join(movies_dir, f"{args.geo_name}_flow_plot.mp4")
            try:
                if create_flow_plot_movie(calibration_input, flow_plot_output, 
                                          tuple(args.resolution), args.fps):
                    success_count += 1
                    print(f"✓ Flow plot movie saved to: {flow_plot_output}")
                else:
                    print(f"✗ Failed to create flow plot movie")
            except Exception as e:
                print(f"✗ Error creating flow plot movie: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\nSkipping standalone flow plot movie (calibration input not found)")
    
    for csv_type, csv_path in files['csv_files']:
        if csv_type == 'geometric':
            geometric_csv = csv_path
    
    # Find calibrated CSV if available
    calibrated_csv = None
    for csv_type, csv_path in files['csv_files']:
        if csv_type == 'calibrated':
            calibrated_csv = csv_path
    
    if calibrated_csv:
        print(f"\n  Calibrated CSV found: {calibrated_csv}")
    else:
        print(f"\n  Calibrated CSV: Not found")
    
    if geometric_csv and calibration_input:
        # Create side-by-side movies (now with 3 panels if calibrated CSV is available)
        total_movies = len(fields)
        for field in fields:
            output_filename = f"{args.geo_name}_{field}_comparison.mp4"
            output_path = os.path.join(movies_dir, output_filename)
            
            print(f"\n{'='*60}")
            print(f"Creating side-by-side movie: {field}")
            if calibrated_csv:
                print(f"  Will include calibrated results from: {calibrated_csv}")
            print(f"{'='*60}")
            
            success = visualize_zerod_results_side_by_side(
                geometric_csv, calibration_input, files['geometric_input'], 
                files['centerline'], files['mesh'],
                output_path, field, tuple(args.resolution), args.fps,
                calibrated_csv_path=calibrated_csv
            )
            
            if success:
                print(f"✓ Movie saved to: {output_path}")
                success_count += 1
            else:
                print(f"✗ Failed to create movie: {output_path}")
    else:
        # Fall back to individual movies if only one CSV type is available
        total_movies = len(fields) * len(files['csv_files'])
        for csv_type, csv_path in files['csv_files']:
            for field in fields:
                output_filename = f"{args.geo_name}_{field}_{args.component}_{csv_type}.mp4"
                output_path = os.path.join(movies_dir, output_filename)
                
                print(f"\n{'='*60}")
                print(f"Creating movie: {field} ({args.component}) - {csv_type}")
                print(f"{'='*60}")
                
                success = visualize_zerod_results(
                    csv_path, files['geometric_input'], files['centerline'], files['mesh'],
                    output_path, field, args.component, tuple(args.resolution), args.fps
                )
                
                if success:
                    print(f"✓ Movie saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"✗ Failed to create movie: {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Successfully created: {success_count}/{total_movies} movies")
    print(f"Movies directory: {movies_dir}")
    print(f"{'='*60}")
    
    if success_count == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

# python3 util/visualizations/visualize_zerod_results.py --set-name set_3 --geo-name tree_007


