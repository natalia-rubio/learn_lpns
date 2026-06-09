#!/usr/bin/env python3
"""
Batch 3D→centerline projection (pure Python, no svSlicer) for Sherlock CCO_trees sets.

Loops geometries in a set, projects every timestep onto the centerline sequentially,
and writes unsteady_soln.vtp under synthetic_junctions_reduced_results/.

Usage: python batch_centerline_proj.py <set_name> [num_procs]
"""

import glob
import os
import sys

import vtk
from tqdm import tqdm
from util.get_bc_integrals import get_res_names
from util.vtk_functions import Integration, calculator, connectivity, cut_plane, read_geo, write_geo
from vtk.util.numpy_support import vtk_to_numpy as v2n


def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane
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
    Slice simulation at certain plane and integrate
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
    for v in get_res_names(inp_3d, "Velocity"):
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
        inp = calculator(inp, fun, [v], "normal_" + v)

    return Integration(inp)


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
        List of timestep strings and the procs directory path
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
    times = []
    for vtu_file in vtu_files:
        basename = os.path.basename(vtu_file)
        # Extract timestep from result_XXX.vtu format
        if basename.startswith("result_") and basename.endswith(".vtu"):
            time_str = basename.replace("result_", "").replace(".vtu", "")
            # Verify it's a valid timestep (numeric)
            if time_str.isdigit() or (time_str and all(c.isdigit() for c in time_str)):
                times.append(time_str)

    times.sort(key=lambda x: int(x) if x.isdigit() else 0)
    return times, procs_dir


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


def project_results_to_centerline(geo_dir, sim_dir, centerline_path, num_procs, output_path):
    """
    Project 3D simulation results from all timesteps onto centerline.
    Args:
        geo_dir: Geometry directory
        sim_dir: Simulation results directory
        centerline_path: Path to centerline VTP file
        num_procs: Number of processors string (e.g., "24")
        output_path: Path to save output centerline with projected results
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find simulation files
        times, procs_dir = find_simulation_files(sim_dir, num_procs)
        if not times:
            print(f"  No simulation files found in {sim_dir}")
            return False

        # Get geometry name from directory
        geo_name = os.path.basename(geo_dir)

        # Read centerline
        reader_1d = read_geo(centerline_path).GetOutput()

        # Read first timestep to get result names
        first_file = None
        if times:
            # Find first timestep file using result_*.vtu naming
            first_file = os.path.join(procs_dir, f"result_{times[0]}.vtu")
            if not os.path.exists(first_file):
                # Try alternative naming patterns as fallback
                alt_patterns = [
                    os.path.join(procs_dir, f"*_result_{times[0]}.vtu"),
                    os.path.join(procs_dir, f"*{times[0]}.vtu"),
                ]
                for pattern in alt_patterns:
                    alt_files = glob.glob(pattern)
                    if alt_files:
                        first_file = sorted(alt_files)[0]
                        break

        if not first_file or not os.path.exists(first_file):
            # Fallback: get any result_*.vtu file
            vtu_files = glob.glob(os.path.join(procs_dir, "result_*.vtu"))
            if vtu_files:
                first_file = sorted(vtu_files)[0]
            else:
                print(f"  Could not find any simulation files in {procs_dir}")
                return False

        reader_3d = read_geo(first_file).GetOutput()
        get_res_names(reader_3d, ["Pressure", "Velocity"])

        # Get point and normals from centerline
        points = v2n(reader_1d.GetPoints().GetData())
        normals = v2n(reader_1d.GetPointData().GetArray("CenterlineSectionNormal"))
        gid = v2n(reader_1d.GetPointData().GetArray("GlobalNodeId"))
        v2n(reader_1d.GetPointData().GetArray("BranchId"))

        # Initialize output arrays for all timesteps (with zero-padded timesteps)
        res_names_1d = [f"pressure_{pad_timestep(time)}" for time in times] + [
            f"velocity_{pad_timestep(time)}" for time in times
        ]
        for name in res_names_1d + ["area"]:
            array = vtk.vtkDoubleArray()
            array.SetName(name)
            array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
            array.Fill(float("NaN"))
            reader_1d.GetPointData().AddArray(array)

        # Move points on caps slightly to ensure nice integration
        ids = vtk.vtkIdList()
        eps_norm = 1.0e-3

        print(f"  Extracting solution at {reader_1d.GetNumberOfPoints()} points for {len(times)} timesteps")

        # Process all points
        for i in tqdm(range(reader_1d.GetNumberOfPoints()), desc=f"  Processing {geo_name}"):
            # Adjust cap points
            reader_1d.GetPointCells(i, ids)
            if ids.GetNumberOfIds() == 1:
                if gid[i] == 0:
                    # inlet
                    points[i] += eps_norm * normals[i]
                else:
                    # outlets
                    points[i] -= eps_norm * normals[i]

            # Process all timesteps
            for time_str in times:
                try:
                    # Construct filename using result_{time_str}.vtu pattern
                    fpath_3d = os.path.join(procs_dir, f"result_{time_str}.vtu")
                    if not os.path.exists(fpath_3d):
                        # Try alternative naming patterns as fallback
                        alt_patterns = [
                            os.path.join(procs_dir, f"*_result_{time_str}.vtu"),
                            os.path.join(procs_dir, f"*{time_str}.vtu"),
                        ]
                        found = False
                        for pattern in alt_patterns:
                            alt_files = glob.glob(pattern)
                            if alt_files:
                                fpath_3d = alt_files[0]
                                found = True
                                break
                        if not found:
                            continue

                    reader_3d = read_geo(fpath_3d).GetOutput()
                    integral = get_integral(reader_3d, points[i], normals[i])
                    # Use zero-padded timestep for array names
                    time_padded = pad_timestep(time_str)
                    reader_1d.GetPointData().GetArray(f"pressure_{time_padded}").SetValue(
                        i, integral.evaluate("Pressure")
                    )
                    reader_1d.GetPointData().GetArray(f"velocity_{time_padded}").SetValue(
                        i, integral.evaluate("Velocity")
                    )

                except Exception:
                    # Skip this timestep/point if integration fails
                    continue

            # Calculate area from first timestep
            try:
                reader_3d = read_geo(first_file).GetOutput()
                integral = get_integral(reader_3d, points[i], normals[i])
                reader_1d.GetPointData().GetArray("area").SetValue(i, integral.area())
            except Exception:
                continue

        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_geo(output_path, reader_1d)
        print(f"  Wrote results to {output_path}")
        return True

    except Exception as e:
        print(f"  Error processing {geo_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Main function to process completed simulations and project results.
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python batch_centerline_proj.py <set_name> [num_procs]")
        print("  set_name: Name of the set to process (required)")
        print("  num_procs: Number of processors (optional, default: 48)")
        sys.exit(1)

    set_name = sys.argv[1]
    num_procs = sys.argv[2] if len(sys.argv) > 2 else "48"

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
        if project_results_to_centerline(geo_dir, sim_dir, centerline_path, num_procs, output_path):
            total_processed += 1
        else:
            total_failed += 1

    print("\n\nSummary:")
    print(f"  Successfully processed: {total_processed}")
    print(f"  Failed/Skipped: {total_failed}")
    print(f"  Total: {total_processed + total_failed}")


if __name__ == "__main__":
    main()
