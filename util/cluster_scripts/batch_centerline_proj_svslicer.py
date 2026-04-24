#!/usr/bin/env python3
"""
Batch script to project 3D simulation results onto centerlines using svSlicer.
First combines all timestep VTU files into a single multi-timestep VTU file,
then calls svSlicer for fast processing.
"""

import os
import sys
import vtk
import glob
import subprocess
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v
from util.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, Integration
from util.get_bc_integrals import get_res_names

# Timeout for svSlicer subprocess (seconds). Increase for very large runs.
SVSLICER_TIMEOUT_SECONDS = 3600  # 1 hour

def pad_timestep(time_str, width=5):
    """
    Pad timestep string with zeros to specified width.
    Args:
        time_str: Timestep string (e.g., "2", "002")
        width: Desired width (default: 5)
    Returns:
        Zero-padded timestep string (e.g., "00002")
    """
    try:
        # Convert to int and back to string with zero padding
        return str(int(time_str)).zfill(width)
    except (ValueError, TypeError):
        # If not numeric, return as-is
        return time_str

def find_simulation_files(sim_dir, num_procs):
    """
    Find all timestep files in the simulation directory.
    Args:
        sim_dir: Base simulation directory
        num_procs: Number of processors (e.g., "24", "48", "96")
    Returns:
        List of (timestep, filepath) tuples and the procs directory path
    """
    procs_dir = os.path.join(sim_dir, f"{num_procs}-procs")
    if not os.path.exists(procs_dir):
        return [], None
    
    # Find all result_*.vtu files
    vtu_files = glob.glob(os.path.join(procs_dir, "result_*.vtu"))
    if not vtu_files:
        return [], None
    
    # Extract timesteps from filenames
    # Format: result_{timestep}.vtu (e.g., result_002.vtu -> 002)
    time_files = []
    for vtu_file in vtu_files:
        basename = os.path.basename(vtu_file)
        # Extract timestep from result_XXX.vtu format
        if basename.startswith('result_') and basename.endswith('.vtu'):
            time_str = basename.replace('result_', '').replace('.vtu', '')
            # Verify it's a valid timestep (numeric)
            if time_str.isdigit() or (time_str and all(c.isdigit() for c in time_str)):
                time_files.append((time_str, vtu_file))
    
    time_files.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    return time_files, procs_dir

def find_centerline_file(geo_dir):
    """
    Find centerline file in geometry directory.
    Args:
        geo_dir: Geometry directory path
    Returns:
        Path to centerline file or None
    """
    # Try common centerline file locations
    centerline_paths = [
        os.path.join(geo_dir, "centerlines_simVascular.vtp"),
        os.path.join(geo_dir, "centerlines", "centerlines.vtp"),
        os.path.join(geo_dir, "centerlines.vtp"),
    ]
    
    for path in centerline_paths:
        if os.path.exists(path):
            return path
    
    return None

def combine_vtu_files(time_files, output_vtu_path):
    """
    Combine multiple VTU files into a single multi-timestep VTU file.
    Args:
        time_files: List of (timestep, filepath) tuples
        output_vtu_path: Path to save combined VTU file
    Returns:
        True if successful, False otherwise
    """
    if not time_files:
        print("  No timestep files to combine")
        return False
    
    print(f"  Combining {len(time_files)} timestep files into single VTU...")
    
    try:
        # Read first file to get mesh structure
        first_timestep, first_file = time_files[0]
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(first_file)
        reader.Update()
        base_grid = reader.GetOutput()
        
        # Create new unstructured grid with same mesh structure
        # Copy structure (points and cells) but not data arrays
        combined_grid = vtk.vtkUnstructuredGrid()
        # Deep copy points
        points = vtk.vtkPoints()
        points.DeepCopy(base_grid.GetPoints())
        combined_grid.SetPoints(points)
        # Copy cell structure
        cell_array = vtk.vtkCellArray()
        cell_array.DeepCopy(base_grid.GetCells())
        cell_types = vtk.vtkUnsignedCharArray()
        cell_types.DeepCopy(base_grid.GetCellTypesArray())
        cell_locations = vtk.vtkIdTypeArray()
        cell_locations.DeepCopy(base_grid.GetCellLocationsArray())
        combined_grid.SetCells(cell_types, cell_locations, cell_array)
        
        # Copy mesh-related arrays from first file
        # Copy point data arrays that are not pressure/velocity
        num_point_arrays = base_grid.GetPointData().GetNumberOfArrays()
        for i in range(num_point_arrays):
            array = base_grid.GetPointData().GetArray(i)
            if array is None:
                continue
            array_name = array.GetName()
            # Skip pressure and velocity arrays (we'll add timestep-specific ones)
            if not (array_name and (array_name.lower().startswith('pressure') or 
                                   array_name.lower().startswith('velocity'))):
                new_array = array.NewInstance()
                new_array.DeepCopy(array)
                combined_grid.GetPointData().AddArray(new_array)
        
        # Copy cell data arrays that are not pressure/velocity
        num_cell_arrays = base_grid.GetCellData().GetNumberOfArrays()
        for i in range(num_cell_arrays):
            array = base_grid.GetCellData().GetArray(i)
            if array is None:
                continue
            array_name = array.GetName()
            # Skip pressure and velocity arrays
            if not (array_name and (array_name.lower().startswith('pressure') or 
                                   array_name.lower().startswith('velocity'))):
                new_array = array.NewInstance()
                new_array.DeepCopy(array)
                combined_grid.GetCellData().AddArray(new_array)
        
        # Process each timestep file
        for idx, (time_str, vtu_file) in enumerate(time_files):
            if not os.path.exists(vtu_file):
                print(f"    Warning: File not found: {vtu_file}")
                continue
            
            if (idx + 1) % 10 == 0:
                print(f"    Processing timestep {idx + 1}/{len(time_files)}...")
            
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_file)
            reader.Update()
            timestep_grid = reader.GetOutput()
            
            # Get padded timestep for array names
            time_padded = pad_timestep(time_str)
            
            # Copy point data arrays (pressure and velocity)
            num_point_arrays = timestep_grid.GetPointData().GetNumberOfArrays()
            for i in range(num_point_arrays):
                array = timestep_grid.GetPointData().GetArray(i)
                if array is None:
                    continue
                
                array_name = array.GetName()
                if array_name is None:
                    continue
                
                # Process pressure and velocity arrays
                if array_name.lower().startswith('pressure'):
                    # Use numpy for safe and efficient copying
                    try:
                        data = v2n(array)
                        # Ensure we have a proper copy
                        data_copy = np.array(data, copy=True)
                        new_array = n2v(data_copy)
                        new_array.SetName(f'pressure_{time_padded}')
                        combined_grid.GetPointData().AddArray(new_array)
                    except Exception as e:
                        print(f"    Warning: Error copying pressure array for timestep {time_str}: {e}")
                        continue
                elif array_name.lower().startswith('velocity'):
                    # Use numpy for safe and efficient copying
                    try:
                        data = v2n(array)
                        # Ensure we have a proper copy
                        data_copy = np.array(data, copy=True)
                        new_array = n2v(data_copy)
                        new_array.SetName(f'velocity_{time_padded}')
                        combined_grid.GetPointData().AddArray(new_array)
                    except Exception as e:
                        print(f"    Warning: Error copying velocity array for timestep {time_str}: {e}")
                        continue
            
            # Copy cell data arrays (pressure and velocity) - usually not present but check anyway
            num_cell_arrays = timestep_grid.GetCellData().GetNumberOfArrays()
            for i in range(num_cell_arrays):
                array = timestep_grid.GetCellData().GetArray(i)
                if array is None:
                    continue
                
                array_name = array.GetName()
                if array_name is None:
                    continue
                
                # Process pressure and velocity arrays
                if array_name.lower().startswith('pressure'):
                    try:
                        data = v2n(array)
                        data_copy = np.array(data, copy=True)
                        new_array = n2v(data_copy)
                        new_array.SetName(f'pressure_{time_padded}')
                        combined_grid.GetCellData().AddArray(new_array)
                    except Exception as e:
                        print(f"    Warning: Error copying cell pressure array for timestep {time_str}: {e}")
                        continue
                elif array_name.lower().startswith('velocity'):
                    try:
                        data = v2n(array)
                        data_copy = np.array(data, copy=True)
                        new_array = n2v(data_copy)
                        new_array.SetName(f'velocity_{time_padded}')
                        combined_grid.GetCellData().AddArray(new_array)
                    except Exception as e:
                        print(f"    Warning: Error copying cell velocity array for timestep {time_str}: {e}")
                        continue
        
        # Write combined VTU file
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_vtu_path)
        writer.SetInputData(combined_grid)
        writer.Write()
        
        # Verify the file was written and can be read back
        try:
            test_reader = vtk.vtkXMLUnstructuredGridReader()
            test_reader.SetFileName(output_vtu_path)
            test_reader.Update()
            test_grid = test_reader.GetOutput()
            num_arrays = test_grid.GetPointData().GetNumberOfArrays()
            print(f"  Combined VTU file written to: {output_vtu_path}")
            print(f"  File contains {num_arrays} point data arrays")
            return True
        except Exception as e:
            print(f"  Warning: Could not verify combined VTU file: {e}")
            # Still return True if file exists
            if os.path.exists(output_vtu_path):
                return True
            return False
        
    except Exception as e:
        print(f"  Error combining VTU files: {e}")
        import traceback
        traceback.print_exc()
        return False

def call_svslicer(combined_vtu_path, centerline_path, output_path, num_threads=None):
    """
    Call svSlicer to project results onto centerline.
    Args:
        combined_vtu_path: Path to combined multi-timestep VTU file
        centerline_path: Path to centerline VTP file
        output_path: Path to save output VTP file
        num_threads: Number of threads for svSlicer (optional)
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find svSlicer executable
        svslicer_paths = [
            "svslicer",  # In PATH
            "/usr/local/bin/svslicer",
            os.path.expanduser("~/bin/svslicer"),
            os.path.expanduser("~/SV_scripts/svSlicer/svslicer"),  # Installed location
            os.path.expanduser("~/SV_scripts/svSlicer/Release/svslicer"),  # Build location
            os.path.join(os.path.dirname(__file__), "../../svSlicer/Release/svslicer"),
            os.path.join(os.path.dirname(__file__), "../../svSlicer/svslicer"),
        ]
        
        svslicer_exe = None
        for path in svslicer_paths:
            # Check if it's in PATH
            if "/" not in path:
                result = subprocess.run(["which", path], capture_output=True, text=True)
                if result.returncode == 0:
                    svslicer_exe = result.stdout.strip()
                    break
            # Check if file exists
            elif os.path.exists(path) and os.access(path, os.X_OK):
                svslicer_exe = path
                break
        
        if svslicer_exe is None:
            print("  Error: svSlicer executable not found")
            print("  Please ensure svSlicer is in your PATH or specify the path")
            return False
        
        # Set up environment
        env = os.environ.copy()
        if num_threads:
            env['OMP_NUM_THREADS'] = str(num_threads)
        
        # Add conda library paths for VTK (if using conda-installed VTK)
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            lib_paths = [
                os.path.join(conda_prefix, 'lib'),
                os.path.join(conda_prefix, 'lib', 'vtk'),
            ]
            # Update LD_LIBRARY_PATH
            ld_library_path = env.get('LD_LIBRARY_PATH', '')
            for lib_path in lib_paths:
                if os.path.exists(lib_path) and lib_path not in ld_library_path:
                    if ld_library_path:
                        env['LD_LIBRARY_PATH'] = f"{lib_path}:{ld_library_path}"
                    else:
                        env['LD_LIBRARY_PATH'] = lib_path
                    ld_library_path = env['LD_LIBRARY_PATH']
        
        # Also check common conda locations
        if 'LD_LIBRARY_PATH' not in env or not env['LD_LIBRARY_PATH']:
            common_conda_libs = [
                os.path.expanduser('~/miniconda3/lib'),
                os.path.expanduser('~/anaconda3/lib'),
                '/home/users/nrubio/miniconda3/lib',  # Your specific path
            ]
            for lib_path in common_conda_libs:
                if os.path.exists(lib_path):
                    if 'LD_LIBRARY_PATH' in env and env['LD_LIBRARY_PATH']:
                        env['LD_LIBRARY_PATH'] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
                    else:
                        env['LD_LIBRARY_PATH'] = lib_path
                    break
        
        # Debug: Show library path if set
        if 'LD_LIBRARY_PATH' in env:
            print(f"  Using LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH'][:100]}...")  # First 100 chars
        
        # Call svSlicer
        print(f"  Calling svSlicer with {num_threads or 'default'} threads...")
        cmd = [svslicer_exe, combined_vtu_path, centerline_path, output_path]
        
        # Run svSlicer and capture output
        # Use subprocess.run with timeout to avoid hanging indefinitely
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                   timeout=SVSLICER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"  Error: svSlicer timed out after {SVSLICER_TIMEOUT_SECONDS}s")
            return False
        
        # Check if output file was created (more reliable than exit code)
        output_created = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
        # If output was created, consider it successful
        if output_created:
            # Show relevant output messages
            if result.stdout:
                # Extract key messages
                lines = result.stdout.split('\n')
                for line in lines[-10:]:  # Last 10 lines
                    if any(keyword in line.lower() for keyword in ['completed', 'writing', 'slice extraction', 'per slice']):
                        print(f"  {line.strip()}")
            
            print(f"  svSlicer completed successfully")
            print(f"  Output written to: {output_path}")
            return True
        
        # If output wasn't created, show error
        if result.returncode != 0:
            # Negative return codes indicate signals (e.g., -11 = SIGSEGV)
            if result.returncode < 0:
                signal_num = -result.returncode
                print(f"  Error: svSlicer crashed with signal {signal_num} (segmentation fault or similar)")
            else:
                print(f"  Error running svSlicer (exit code: {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")  # Last 500 chars
            if result.stdout:
                # Show last part of stdout for debugging
                lines = result.stdout.split('\n')
                print(f"  Last stdout lines:")
                for line in lines[-10:]:
                    if line.strip():
                        print(f"    {line.strip()}")
            return False
        else:
            # Exit code 0 but no output file - something went wrong
            print(f"  Warning: svSlicer exited with code 0 but no output file was created")
            if result.stdout:
                lines = result.stdout.split('\n')
                for line in lines[-10:]:
                    if line.strip():
                        print(f"    {line.strip()}")
            return False
        
    except Exception as e:
        print(f"  Error calling svSlicer: {e}")
        import traceback
        traceback.print_exc()
        return False

def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane (Python fallback method).
    Args:
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Connectivity filter output
    """
    # cut 3d geometry
    cut_3d = cut_plane(inp_3d, origin, normal)
    
    # extract region closest to centerline
    con = connectivity(cut_3d, origin)
    return con

def get_integral(inp_3d, origin, normal):
    """
    Slice simulation at certain plane and integrate (Python fallback method).
    Args:
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # slice vessel at given location
    inp = slice_vessel(inp_3d, origin, normal)
    
    # recursively add calculators for normal velocities
    for v in get_res_names(inp_3d, 'Velocity'):
        fun = (
            "dot(iHat*"
            + repr(float(normal[0]))
            + "+jHat*"
            + repr(float(normal[1]))
            + "+kHat*"
            + repr(float(normal[2]))
            + ","
            + v
            + ")"
        )
        inp = calculator(inp, fun, [v], 'normal_' + v)
    
    return Integration(inp)

def process_single_point(args):
    """
    Process a single centerline point for all timesteps.
    This function is designed to be called in parallel.
    Args:
        args: Tuple of (point_idx, point, normal, gid_val, ids_count, vtu_data_dict, time_files, first_file_key)
    Returns:
        Dictionary with results for this point
    """
    i, point, normal, gid_val, ids_count, vtu_data_dict, time_files, first_file_key = args
    
    results = {}
    eps_norm = 1.0e-3
    
    # Adjust cap points
    if ids_count == 1:
        if gid_val == 0:
            point = point + eps_norm * normal
        else:
            point = point - eps_norm * normal
    
    # Process all timesteps
    for time_str, vtu_file in time_files:
        try:
            if vtu_file not in vtu_data_dict:
                continue
            
            reader_3d = vtu_data_dict[vtu_file]
            integral = get_integral(reader_3d, point, normal)
            time_padded = pad_timestep(time_str)
            results[f'pressure_{time_padded}'] = integral.evaluate("Pressure")
            results[f'velocity_{time_padded}'] = integral.evaluate("Velocity")
        except Exception:
            continue
    
    # Calculate area from first timestep
    try:
        if first_file_key in vtu_data_dict:
            reader_3d = vtu_data_dict[first_file_key]
            integral = get_integral(reader_3d, point, normal)
            results['area'] = integral.area()
    except Exception:
        pass
    
    return i, results

def project_results_python_fallback(geo_dir, sim_dir, centerline_path, num_procs, output_path, time_files, num_workers=None):
    """
    Optimized Python-based projection with caching and parallelization.
    Args:
        geo_dir: Geometry directory
        sim_dir: Simulation results directory
        centerline_path: Path to centerline VTP file
        num_procs: Number of processors string (e.g., "24")
        output_path: Path to save output centerline with projected results
        time_files: List of (time_str, file_path) tuples
        num_workers: Number of parallel workers (default: cpu_count() or PYTHON_NUM_WORKERS env var)
    Returns:
        True if successful, False otherwise
    """
    # Check for environment variable for number of workers
    if num_workers is None:
        num_workers = int(os.getenv('PYTHON_NUM_WORKERS', cpu_count() - 1))
    
    try:
        print(f"  Using optimized Python-based projection...")
        
        # Read centerline
        reader_1d = read_geo(centerline_path).GetOutput()
        
        # Read first timestep to get result names
        _, first_file = time_files[0]
        reader_3d = read_geo(first_file).GetOutput()
        res_names_3d = get_res_names(reader_3d, ['Pressure', 'Velocity'])
        
        # Get point and normals from centerline
        points = v2n(reader_1d.GetPoints().GetData())
        normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
        gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))
        branch_id = v2n(reader_1d.GetPointData().GetArray('BranchId'))
        
        # Initialize output arrays for all timesteps (with zero-padded timesteps)
        times = [t for t, _ in time_files]
        res_names_1d = [f"pressure_{pad_timestep(time)}" for time in times] + [f"velocity_{pad_timestep(time)}" for time in times]
        for name in res_names_1d + ['area']:
            array = vtk.vtkDoubleArray()
            array.SetName(name)
            array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
            array.Fill(float('NaN'))
            reader_1d.GetPointData().AddArray(array)
        
        # Prepare data for parallel processing
        ids = vtk.vtkIdList()
        num_points = reader_1d.GetNumberOfPoints()
        
        # Pre-compute point data
        point_data = []
        for i in range(num_points):
            reader_1d.GetPointCells(i, ids)
            point_data.append((
                i,
                points[i].copy(),  # Copy to avoid issues with parallel access
                normals[i].copy(),
                int(gid[i]),
                ids.GetNumberOfIds(),
                {},  # vtu_cache will be shared but thread-safe for reads
                time_files,
                first_file
            ))
        
        print(f"  Extracting solution at {num_points} points for {len(times)} timesteps")
        
        # Pre-load all VTU files into memory (this is the key optimization!)
        print(f"  Pre-loading {len(time_files)} VTU files into memory...")
        vtu_data_dict = {}
        for time_str, vtu_file in tqdm(time_files, desc="  Loading VTU files"):
            if os.path.exists(vtu_file):
                try:
                    vtu_data_dict[vtu_file] = read_geo(vtu_file).GetOutput()
                except Exception as e:
                    print(f"    Warning: Failed to load {vtu_file}: {e}")
        
        if first_file not in vtu_data_dict and os.path.exists(first_file):
            try:
                vtu_data_dict[first_file] = read_geo(first_file).GetOutput()
            except Exception:
                pass
        
        print(f"  Loaded {len(vtu_data_dict)} VTU files")
        
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # Leave one core free
        
        print(f"  Using {num_workers} parallel workers...")
        
        # Update point_data with loaded VTU data
        point_data_with_vtu = []
        for data in point_data:
            point_data_with_vtu.append((
                data[0],  # i
                data[1],  # point
                data[2],  # normal
                data[3],  # gid_val
                data[4],  # ids_count
                vtu_data_dict,  # Pre-loaded VTU data
                data[6],  # time_files
                first_file  # first_file_key
            ))
        
        # Use threading instead of multiprocessing (VTK releases GIL, so this works well)
        # Threading allows sharing VTK objects without pickling issues
        all_results = {}
        results_lock = threading.Lock()
        
        def process_point_wrapper(point_data_item):
            """Wrapper to process a single point and store results thread-safely"""
            point_idx, results_dict = process_single_point(point_data_item)
            with results_lock:
                all_results[point_idx] = results_dict
            return point_idx
        
        # Process points in parallel using threads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_point_wrapper, data): data[0] 
                      for data in point_data_with_vtu}
            
            # Process with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="  Processing points"):
                try:
                    future.result()  # Get result (or raise exception)
                except Exception as e:
                    point_idx = futures[future]
                    print(f"    Warning: Point {point_idx} failed: {e}")
        
        # Write results back to centerline
        print("  Writing results to centerline...")
        for i, results_dict in all_results.items():
            for array_name, value in results_dict.items():
                array = reader_1d.GetPointData().GetArray(array_name)
                if array is not None:
                    array.SetValue(i, value)
        
        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_geo(output_path, reader_1d)
        print(f"  Wrote results to {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error in Python fallback: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_centerline_results(centerline_path, batch_outputs, final_output_path):
    """
    Merge multiple centerline results from batch processing into a single file.
    Args:
        centerline_path: Original centerline file (for structure)
        batch_outputs: List of (batch_output_path, timesteps) tuples
        final_output_path: Path to save merged results
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read original centerline for structure
        reader_1d = read_geo(centerline_path).GetOutput()
        
        # Initialize arrays for all timesteps
        all_times = []
        for _, timesteps in batch_outputs:
            all_times.extend(timesteps)
        all_times = sorted(set(all_times), key=lambda x: int(x) if x.isdigit() else 0)
        
        # Add arrays for all timesteps
        import vtk
        for time_str in all_times:
            time_padded = pad_timestep(time_str)
            for name in [f'pressure_{time_padded}', f'velocity_{time_padded}']:
                array = vtk.vtkDoubleArray()
                array.SetName(name)
                array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
                array.Fill(float('NaN'))
                reader_1d.GetPointData().AddArray(array)
        
        # Copy data from each batch output
        for batch_output_path, timesteps in batch_outputs:
            if not os.path.exists(batch_output_path):
                continue
            batch_reader = read_geo(batch_output_path).GetOutput()
            for time_str in timesteps:
                time_padded = pad_timestep(time_str)
                for name in [f'pressure_{time_padded}', f'velocity_{time_padded}']:
                    batch_array = batch_reader.GetPointData().GetArray(name)
                    if batch_array:
                        output_array = reader_1d.GetPointData().GetArray(name)
                        if output_array:
                            output_array.DeepCopy(batch_array)
        
        # Write merged result
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        write_geo(final_output_path, reader_1d)
        return True
    except Exception as e:
        print(f"  Error merging batch results: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_geometry(geo_dir, sim_dir, centerline_path, num_procs, output_path, num_threads=None, batch_size=10, use_python_only=False):
    """
    Process a single geometry: combine VTU files and call svSlicer.
    If svSlicer crashes with many timesteps, process in batches.
    Args:
        geo_dir: Geometry directory
        sim_dir: Simulation results directory
        centerline_path: Path to centerline VTP file
        num_procs: Number of processors string (e.g., "24")
        output_path: Path to save output centerline with projected results
        num_threads: Number of threads for svSlicer (optional)
        batch_size: Number of timesteps to process at once (default: 10)
        use_python_only: If True, skip svSlicer and use Python method directly
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find simulation files
        time_files, procs_dir = find_simulation_files(sim_dir, num_procs)
        if not time_files:
            print(f"  No simulation files found in {sim_dir}")
            return False
        
        # Check if we should skip svSlicer entirely
        if use_python_only or os.getenv('SVSLICER_USE_PYTHON_ONLY', '').lower() in ('1', 'true', 'yes'):
            print(f"  Using Python-based projection (svSlicer skipped)")
            return project_results_python_fallback(geo_dir, sim_dir, centerline_path, num_procs, output_path, time_files)
        
        # Try processing all timesteps at once first
        combined_vtu_path = os.path.join(procs_dir, "combined_results.vtu")
        
        # Check if combined file already exists
        if os.path.exists(combined_vtu_path):
            print(f"  Using existing combined VTU file: {combined_vtu_path}")
        else:
            # Combine VTU files
            if not combine_vtu_files(time_files, combined_vtu_path):
                return False
        
        # Try calling svSlicer with all timesteps
        print(f"  Attempting to process all {len(time_files)} timesteps at once...")
        svslicer_success = call_svslicer(combined_vtu_path, centerline_path, output_path, num_threads)
        
        # Check if output was created (even if exit code was non-zero)
        output_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
        if output_exists:
            print(f"  Successfully processed all timesteps at once")
            return True
        
        # If that failed, try processing in batches
        print(f"  Processing all timesteps failed (svSlicer returned: {svslicer_success}, output exists: {output_exists})")
        print(f"  Falling back to batch processing with batch size {batch_size}...")
        batch_outputs = []
        
        for batch_start in range(0, len(time_files), batch_size):
            batch_end = min(batch_start + batch_size, len(time_files))
            batch_files = time_files[batch_start:batch_end]
            batch_times = [t for t, _ in batch_files]
            
            print(f"  Processing batch {batch_start//batch_size + 1} (timesteps {batch_times[0]} to {batch_times[-1]})...")
            
            # Create combined VTU for this batch
            batch_combined_path = os.path.join(procs_dir, f"combined_results_batch_{batch_start}.vtu")
            if not combine_vtu_files(batch_files, batch_combined_path):
                print(f"  Failed to combine batch {batch_start//batch_size + 1}")
                continue
            
            # Process batch with svSlicer
            batch_output_path = output_path.replace('.vtp', f'_batch_{batch_start}.vtp')
            if call_svslicer(batch_combined_path, centerline_path, batch_output_path, num_threads):
                if os.path.exists(batch_output_path):
                    batch_outputs.append((batch_output_path, batch_times))
                    print(f"  Batch {batch_start//batch_size + 1} completed successfully")
                else:
                    print(f"  Batch {batch_start//batch_size + 1} did not produce output")
            else:
                print(f"  Batch {batch_start//batch_size + 1} failed")
        
        # Merge batch results
        if batch_outputs:
            print(f"  Merging {len(batch_outputs)} batch results...")
            if merge_centerline_results(centerline_path, batch_outputs, output_path):
                # Clean up batch files
                for batch_path, _ in batch_outputs:
                    if os.path.exists(batch_path):
                        os.remove(batch_path)
                print(f"  Successfully merged batch results")
                return True
            else:
                print(f"  Failed to merge batch results")
                # Fall back to Python method
                print(f"  All svSlicer attempts failed, falling back to Python-based projection...")
                return project_results_python_fallback(geo_dir, sim_dir, centerline_path, num_procs, output_path, time_files)
        else:
            print(f"  No batches completed successfully")
            # Fall back to Python method
            print(f"  All svSlicer attempts failed, falling back to Python-based projection...")
            return project_results_python_fallback(geo_dir, sim_dir, centerline_path, num_procs, output_path, time_files)
        
    except Exception as e:
        print(f"  Error processing geometry: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to process completed simulations using svSlicer.
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python batch_centerline_proj_svslicer.py <set_name> [num_procs] [num_threads]")
        print("  set_name: Name of the set to process (required)")
        print("  num_procs: Number of processors (optional, default: 48)")
        print("  num_threads: Number of threads for svSlicer (optional)")
        print("")
        print("Environment variables:")
        print("  SVSLICER_USE_PYTHON_ONLY=1  Skip svSlicer and use Python method directly")
        sys.exit(1)
    
    set_name = sys.argv[1]
    num_procs = sys.argv[2] if len(sys.argv) > 2 else "48"
    num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    base_dir = "/scratch/users/nrubio/synthetic_junctions/CCO_trees"
    output_base_dir = "/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return
    
    # Check if set directory exists
    set_dir = os.path.join(base_dir, set_name)
    if not os.path.exists(set_dir):
        print(f"Error: Set directory does not exist: {set_dir}")
        return
    
    output_set_dir = os.path.join(output_base_dir, set_name)
    
    print(f"Processing set: {set_name}")
    print(f"Using {num_procs}-procs for simulation files")
    if num_threads:
        print(f"Using {num_threads} threads for svSlicer")
    print()
    
    total_processed = 0
    total_failed = 0
    
    # Process the specified set
    # Get geometry names
    geo_names = [d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d))]
    geo_names.sort()
    
    print(f"Found {len(geo_names)} geometries in set {set_name}\n")
    
    for geo_name in geo_names:
        geo_dir = os.path.join(set_dir, geo_name)
        output_geo_dir = os.path.join(output_set_dir, geo_name)
        
        # Find centerline
        centerline_path = find_centerline_file(geo_dir)
        if not centerline_path:
            print(f"  Skipping {geo_name}: No centerline file found")
            total_failed += 1
            continue
        
        # Find simulation directory - try common patterns
        sim_dir = None
        sim_patterns = [
            os.path.join(geo_dir, f"{geo_name}_flow_unsteady"),
            os.path.join(geo_dir, "flow_unsteady"),
            geo_dir,  # Results might be directly in geo_dir
        ]
        
        for pattern in sim_patterns:
            if os.path.exists(os.path.join(pattern, f"{num_procs}-procs")):
                sim_dir = pattern
                break
        
        if not sim_dir:
            print(f"  Skipping {geo_name}: No simulation directory found")
            total_failed += 1
            continue
        
        # Check if output already exists
        output_path = os.path.join(output_geo_dir, "unsteady_soln.vtp")
        if os.path.exists(output_path):
            print(f"  Skipping {geo_name}: Output already exists")
            continue
        
        print(f"  Processing {geo_name}...")
        if process_geometry(geo_dir, sim_dir, centerline_path, num_procs, output_path, num_threads):
            total_processed += 1
        else:
            total_failed += 1
        print()
    
    print(f"\n\nSummary:")
    print(f"  Successfully processed: {total_processed}")
    print(f"  Failed/Skipped: {total_failed}")
    print(f"  Total: {total_processed + total_failed}")

if __name__ == "__main__":
    main()

