import csv
import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

from util.zerod_calibration.generate_zerod_inputs_cli import DEFAULT_JUNCTION_TYPES

# Reference 0D solver JSONs per geometry (VMR cohort layout).
STANDARD_0D_SUBDIR = "standard-0d"


def standard_0d_dir(data_root: str, set_name: str) -> str:
    """``data/zeroD/<set_name>/standard-0d/`` — one JSON per geometry."""
    return os.path.join(data_root, "zeroD", set_name, STANDARD_0D_SUBDIR)


def standard_0d_json_path(data_root: str, set_name: str, geo_name: str) -> str:
    return os.path.join(standard_0d_dir(data_root, set_name), f"{geo_name}.json")


# Time step sizes (sim_period / sim_steps_per_cycle) from data/vmr_models.json
VMR_time_step_dict = {
    "0001_0001": 0.001,
    "0002_0001": 0.001,
    "0003_0001": 0.001,
    "0005_1001": 0.001,
    "0006_0001": 0.0003025,
    "0063_0001": 0.000497391304347826,
    "0063_1001": 0.00041666875,
    "0064_0001": 0.000497391304347826,
    "0064_1001": 0.00041666875,
    "0065_0001": 0.000497391304347826,
    "0065_1001": 0.00041666875,
    "0066_0001": 0.0008128333333333334,
    "0067_0001": 0.0005086666666666667,
    "0068_0001": 0.00019555,
    "0069_0001": 0.000339,
    "0070_0001": 0.00032900000000000003,
    "0071_0001": 0.0004484,
    "0072_0001": 0.00048516666666666673,
    "0073_0001": 0.0005858333333333333,
    "0074_0001": 0.0005818333333333334,
    "0075_0001": 0.000497391304347826,
    "0075_1001": 0.00041666875,
    "0076_0001": 0.000497391304347826,
    "0076_1001": 0.0004464375,
    "0077_0001": 0.00047666666666666663,
    "0077_1001": 0.00041666875,
    "0078_0001": 0.001,
    "0079_0001": 0.001,
    "0080_0001": 0.000176470575,
    "0081_0001": 0.000245,
    "0082_0001": 0.0002,
    "0083_2002": 0.00030930000000000004,
    "0084_0001": 0.0002375,
    "0085_1001": 0.0004,
    "0086_0001": 0.00021124999999999998,
    "0087_1001": 0.00041999999999999996,
    "0088_1001": 0.0005,
    "0089_1001": 0.00046,
    "0090_0001": 0.00082,
    "0091_0001": 0.0011266666666666667,
    "0092_0001": 0.0011176470588235294,
    "0093_0001": 0.0007124999999999999,
    "0094_0001": 0.0011111111111111111,
    "0095_0001": 0.000937,
    "0096_0001": 0.0005357,
    "0097_0001": 0.0007228,
    "0098_0001": 0.0005262999999999999,
    "0099_0001": 0.0006976,
    "0101_0001": 0.0007458333333333334,
    "0102_0001": 0.0014083333333333333,
    "0103_0001": 0.0007933333333333333,
    "0104_0001": 0.000968,
    "0105_0001": 0.0009519999999999999,
    "0106_0001": 4e-05,
    "0107_0001": 0.0005200000000000001,
    "0108_0001": 0.001,
    "0110_0001": 0.000234375,
    "0111_0001": 9.999999999999999e-05,
    "0112_1001": 0.0004,
    "0118_1000": 0.00024037499999999997,
    "0119_0001": 0.00026784375,
    "0125_0001": 0.0014299999999999998,
    "0126_0001": 0.0014299999999999998,
    "0129_0000": 0.0003675,
    "0130_0000": 0.00031781249999999995,
    "0131_0000": 0.000375,
    "0134_0002": 0.0003491666666666667,
    "0138_1001": 0.0003025,
    "0139_1001": 0.0003025,
    "0140_2001": 0.000275625,
    "0141_1001": 0.00024656250000000004,
    "0142_1001": 0.000293125,
    "0144_1001": 0.00024656250000000004,
    "0145_1001": 0.000275625,
    "0146_1001": 0.000275625,
    "0147_1001": 0.00020843750000000002,
    "0148_1001": 0.0002678125,
    "0149_1001": 0.00026031249999999996,
    "0150_0001": 0.0003675,
    "0151_0001": 0.0002678125,
    "0154_0001": 0.00033,
    "0155_0001": 0.00021874999999999998,
    "0156_0001": 0.00026031249999999996,
    "0157_0000": 0.000271875,
    "0158_0001": 0.00073,
    "0160_6001": 0.00025,
    "0161_0001": 0.00026031249999999996,
    "0162_3001": 0.00051375,
    "0163_0001": 0.00051375,
    "0164_0001": 0.00073,
    "0165_0001": 0.00073,
    "0166_0001": 0.001,
    "0167_0001": 0.001,
    "0172_0001": 0.001017,
    "0173_1001": 0.00057,
    "0174_0000": 0.00015,
    "0175_0000": 0.0002640625,
    "0176_0000": 0.00025,
    "0183_1002": 0.0014299999999999998,
    "0184_0001": 0.00125,
    "0185_0001": 0.00076,
    "0186_0002": 0.0008571,
    "0187_0002": 0.001,
    "0188_0001": 0.0009677,
    "0189_0001": 0.0013044,
}


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

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return results, sorted(times)

        if "location" in fieldnames:
            vessel_col = "location"
        elif "name" in fieldnames:
            vessel_col = "name"
        else:
            return results, sorted(times)

        for row in reader:
            location = row[vessel_col]
            time = float(row["time"])
            times.add(time)

            if location not in results:
                results[location] = {}
            if time not in results[location]:
                results[location][time] = {}

            for key, value in row.items():
                if key not in (vessel_col, "time"):
                    try:
                        results[location][time][key] = float(value)
                    except (ValueError, TypeError):
                        continue

    return results, sorted(times)


def load_from_json(json_path):
    """
    Load JSON file from path.
    """
    with open(json_path, "r") as f:
        print(f"  Loading JSON file from: {json_path}")
        geometric_input = json.load(f)
    return geometric_input


def save_to_json(geometric_input, json_path):
    """
    Save geometric input to JSON file.
    """
    with open(json_path, "w") as f:
        json.dump(geometric_input, f, indent=4)
    print(f"  Saved geometric input to: {json_path}")
    return


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
        gen_params = root.find("GeneralSimulationParameters")
        if gen_params is None:
            return None

        num_time_steps = gen_params.find("Number_of_time_steps")
        time_step_size = gen_params.find("Time_step_size")

        if num_time_steps is None or time_step_size is None:
            return None

        # Find inlet boundary condition (Dirichlet with Unsteady time dependence)
        inlet_bc_name = None
        add_equation = root.find("Add_equation")
        if add_equation is not None:
            for bc in add_equation.findall("Add_BC"):
                bc_type = bc.find("Type")
                time_dep = bc.find("Time_dependence")

                if (
                    bc_type is not None
                    and bc_type.text == "Dirichlet"
                    and time_dep is not None
                    and time_dep.text == "Unsteady"
                ):
                    inlet_bc_name = bc.get("name")
                    break

        result = {
            "num_time_steps": int(num_time_steps.text),
            "time_step_size": float(time_step_size.text),
        }

        if inlet_bc_name:
            result["inlet_bc_name"] = inlet_bc_name

        return result
    except Exception as e:
        print(f"Warning: Could not parse simulation XML: {e}")
        return None


def get_time_period(set_name, geo_name):
    """
    Try to get the actual time period from 3D simulation XML.
    Returns time period in seconds, or None if not found.
    """
    # Try to find XML file
    xml_paths = [
        os.path.join("data", "threeD", set_name, geo_name, "fluid_simulation_0-0.xml"),
        os.path.join("data", "threeD", set_name, geo_name, "solver.inp"),
    ]

    for xml_path in xml_paths:
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Look for time step size and number of time steps
                gen_params = root.find("General_Parameters")
                if gen_params is None:
                    gen_params = root.find("GeneralSimulationParameters")

                if gen_params is not None:
                    num_time_steps_elem = gen_params.find("Number_of_time_steps")
                    time_step_size_elem = gen_params.find("Time_step_size")

                    if num_time_steps_elem is not None and time_step_size_elem is not None:
                        num_time_steps = int(num_time_steps_elem.text)
                        time_step_size = float(time_step_size_elem.text)
                        time_period = num_time_steps * time_step_size
                        return time_period
            except Exception:
                pass

    return None


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
    arrays["Points"] = points

    # Extract connectivity
    cells = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        if cell.GetNumberOfPoints() == 2:  # Line segment
            cells.append([cell.GetPointId(0), cell.GetPointId(1)])
    arrays["Cells"] = cells

    return arrays, centerline


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
    flow_timesteps = [key for key in centerline_data.keys() if key.startswith("velocity_") or key.startswith("flow_")]

    def extract_timestep(name):
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 0

    flow_timesteps.sort(key=extract_timestep)
    len(flow_timesteps)
    time_increment = extract_timestep(flow_timesteps[1]) - extract_timestep(flow_timesteps[0])

    # Get timestep size from XML (if geo_dir is available)
    if geo_dir is not None and os.path.exists(geo_dir):
        xml_path = os.path.join(geo_dir, "fluid_simulation_0-0.xml")

        if not os.path.exists(xml_path):
            # If no XML found, use default
            raise ValueError("No XML file found in {geo_dir}")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find GeneralSimulationParameters
        gen_params = root.find("GeneralSimulationParameters")
        if gen_params is None:
            gen_params = root.find("General_Parameters")

        threeD_time_step_size = None
        if gen_params is not None:
            time_step_size_elem = gen_params.find("Time_step_size")
            if time_step_size_elem is not None:
                threeD_time_step_size = float(time_step_size_elem.text)

    if "VMR" in centerline_soln_path:
        geometry_name = centerline_soln_path.split("/")[-2]
        threeD_time_step_size = VMR_time_step_dict[geometry_name]
        print(f"  Found time_step_size in VMR dictionary: {threeD_time_step_size:.6f} s")

    if threeD_time_step_size is None:
        raise ValueError("Could not extract time step size from XML")
    else:
        time_step_size = time_increment * threeD_time_step_size

    return time_step_size


def get_paths(base_dir, args):
    """Build geometry variant paths, resolve centerline, and return geo_dir."""
    geometry_variants = {
        "original": {
            "geometric_input": os.path.join(base_dir, "geometric_input.json"),
            "geometric_results": os.path.join(base_dir, "geometric_results.csv"),
            "calibration_input": os.path.join(base_dir, "calibration_input.json"),
            "junction_types": {},
        },
        "bifurcations": {
            "geometric_input": os.path.join(base_dir, "bifurcations_geometric_input.json"),
            "geometric_results": os.path.join(base_dir, "bifurcations_geometric_results.csv"),
            "calibration_input": os.path.join(base_dir, "bifurcations_calibration_input.json"),
            "junction_types": {},
        },
        "bifurcations_EL": {
            "geometric_input": os.path.join(base_dir, "bifurcations_EL_geometric_input.json"),
            "geometric_results": os.path.join(base_dir, "bifurcations_EL_geometric_results.csv"),
            "calibration_input": os.path.join(base_dir, "bifurcations_EL_calibration_input.json"),
            "junction_types": {},
        },
    }

    # Add paths for each junction type variant within each geometry variant
    for geo_variant in geometry_variants:
        prefix = "" if geo_variant == "original" else f"{geo_variant}_"
        for jtype in DEFAULT_JUNCTION_TYPES:
            geometry_variants[geo_variant]["junction_types"][jtype] = {
                "calibration_input": os.path.join(base_dir, f"{prefix}calibration_input_{jtype}.json"),
                "calibrated_output": os.path.join(base_dir, f"{prefix}calibrated_output_{jtype}.json"),
                "calibrated_results": os.path.join(base_dir, f"{prefix}calibrated_results_{jtype}.csv"),
            }

    geo_dir = os.path.join("data", "threeD", args.set_name, args.geo_name)
    centerline_paths = [
        os.path.join(geo_dir, "centerlines_simVascular.vtp"),
        os.path.join(geo_dir, "centerlines", "centerlines.vtp"),
        os.path.join(geo_dir, "centerlines.vtp"),
    ]
    oneD_soln_paths = [
        os.path.join("data", "oneD", args.set_name, args.geo_name, "unsteady_soln.vtp"),
    ]
    if "VMR" in args.set_name:
        oneD_soln_paths.append(
            os.path.join("data", "oneD", "VMR", args.geo_name, "unsteady_soln.vtp"),
        )

    centerline_path = None
    for path in centerline_paths:
        if os.path.exists(path):
            centerline_path = path
            break

    if centerline_path is None:
        for path in oneD_soln_paths:
            if os.path.exists(path):
                centerline_path = path
                print(f"  Using 1D solution as centerline source: {centerline_path}")
                break

    if centerline_path is None:
        raise FileNotFoundError(f"Centerline file not found. Tried: {centerline_paths + oneD_soln_paths}")

    return geometry_variants, centerline_path, geo_dir


def get_vmr_geometries(
    standard_0d_dir_path: str | None = None,
    *,
    data_root: str = "data",
    set_name: str = "VMR",
):
    """List valid VMR geometry names from JSON files under ``standard-0d/``."""
    if standard_0d_dir_path is None:
        standard_0d_dir_path = standard_0d_dir(data_root, set_name)
    if not os.path.exists(standard_0d_dir_path):
        raise FileNotFoundError(f"Standard 0D directory not found: {standard_0d_dir_path}")

    geo_names = []
    for json_file in sorted(glob.glob(os.path.join(standard_0d_dir_path, "*.json"))):
        geo_name = os.path.basename(json_file).replace(".json", "")
        try:
            with open(json_file, "r") as f:
                json.load(f)
            geo_names.append(geo_name)
        except Exception as e:
            print(f"  Warning: Skipping invalid JSON file {geo_name}: {e}")
    return geo_names
