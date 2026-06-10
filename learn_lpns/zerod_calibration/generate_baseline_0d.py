import json
import os
import subprocess

import numpy as np

from learn_lpns.config import apply_solver_parameters, get_pipeline_config
from learn_lpns.zerod_calibration.oned_to_zerod import (
    find_inlet_outlet_caps,
    find_inlet_outlet_caps_from_centerline,
)


def update_simulation_parameters(geo_dir, json_path, inlet_cap_name, capacitance_value=1e-10, zero_stenosis=False):
    """
    Update geometric input JSON with simulation parameters
    """
    # Read JSON
    with open(json_path) as f:
        zerod_input = json.load(f)

    if "simulation_parameters" in zerod_input:
        apply_solver_parameters(zerod_input["simulation_parameters"], get_pipeline_config().solver)

        # Calculate cardiac cycle period from boundary condition time array
        if "boundary_conditions" in zerod_input and len(zerod_input["boundary_conditions"]) > 0:
            bc = zerod_input["boundary_conditions"][0]
            if "bc_values" in bc and "t" in bc["bc_values"]:
                t_array = bc["bc_values"]["t"]
                if len(t_array) > 1:
                    # Cardiac cycle period is the time span of the BC array
                    # For evenly spaced times: period = t[-1] - t[0] + dt
                    # Or simply: period = t[-1] if t[0] == 0
                    if t_array[0] == 0.0:
                        cardiac_period = t_array[-1]
                    else:
                        dt = t_array[1] - t_array[0] if len(t_array) > 1 else 0.0
                        cardiac_period = t_array[-1] - t_array[0] + dt
                    zerod_input["simulation_parameters"]["cardiac_cycle_period"] = cardiac_period
                    print(f"  Set cardiac_cycle_period to {cardiac_period:.6f} s (from BC time array)")
                elif len(t_array) == 1:
                    # Single time point - use default
                    zerod_input["simulation_parameters"]["cardiac_cycle_period"] = 1.0
                    print("  Warning: Only one time point in BC, using default cardiac_cycle_period = 1.0 s")
            else:
                # No BC time array - use default
                zerod_input["simulation_parameters"]["cardiac_cycle_period"] = 1.0
                print("  Warning: No BC time array found, using default cardiac_cycle_period = 1.0 s")
        else:
            # No boundary conditions - use default
            zerod_input["simulation_parameters"]["cardiac_cycle_period"] = 1.0
            print("  Warning: No boundary conditions found, using default cardiac_cycle_period = 1.0 s")

    # Set all capacitance (C) values to 10^-10
    capacitance_value = 1e-10  # 1e-10
    vessels_updated = 0
    for vessel in zerod_input.get("vessels", []):
        if "zero_d_element_values" in vessel and "C" in vessel["zero_d_element_values"]:
            vessel["zero_d_element_values"]["C"] = capacitance_value
            vessels_updated += 1
        if (
            zero_stenosis
            and "zero_d_element_values" in vessel
            and "stenosis_coefficient" in vessel["zero_d_element_values"]
        ):
            vessel["zero_d_element_values"]["stenosis_coefficient"] = 0.0

    print(f"  Set capacitance (C) to {capacitance_value} for {vessels_updated} vessels")

    # Write updated JSON
    with open(json_path, "w") as f:
        json.dump(zerod_input, f, indent=4)

    print("  Updated geometric input with simulation parameters")
    print(f"  Saved to: {json_path}")
    return


def create_geometric_zerod_input_rom(
    geo_dir,
    centerline_path,
    output_path,
    simvascular_path=None,
    dt=0.2,
    num_time_steps=5,
    num_cardiac_cycles=1,
):
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

    print("Using SimVascular ROM workflow to generate 0D input")

    # Find inlet and outlet caps
    # If geo_dir doesn't exist or doesn't have mesh files, use centerline-based approach
    use_centerline_caps = False
    if geo_dir is None or not os.path.exists(geo_dir):
        use_centerline_caps = True
    else:
        # Check if mesh-surfaces directory exists
        mesh_surfaces_paths = [
            os.path.join(geo_dir, "mesh", "fluid_msh_0", "mesh-surfaces"),
            os.path.join(geo_dir, "mesh-complete", "mesh-surfaces"),
        ]
        if not any(os.path.exists(p) for p in mesh_surfaces_paths):
            use_centerline_caps = True

    if use_centerline_caps:
        print("Identifying inlet and outlet caps from centerline data...")
        # Try to find geometric input to get vessel names
        geometric_input_path = None
        if geo_dir and os.path.exists(geo_dir):
            geometric_input_path = os.path.join(geo_dir, "geometric_input.json")
            if not os.path.exists(geometric_input_path):
                # Try parent directory
                parent_dir = os.path.dirname(geo_dir)
                geometric_input_path = os.path.join(parent_dir, os.path.basename(geo_dir), "geometric_input.json")
                if not os.path.exists(geometric_input_path):
                    geometric_input_path = None

        inlet_cap, outlet_caps, _mesh_surfaces_dir = find_inlet_outlet_caps_from_centerline(
            centerline_path, geometric_input_path=geometric_input_path
        )
    else:
        # Find inlet and outlet caps (handles both svVascularize and SimVascular formats)
        print("Identifying inlet and outlet caps from mesh files...")
        inlet_cap, outlet_caps, _mesh_surfaces_dir = find_inlet_outlet_caps(geo_dir)

    # Get geometry name from directory or centerline path
    if geo_dir and os.path.exists(geo_dir):
        geo_name = os.path.basename(geo_dir.rstrip("/"))
    else:
        # Extract from centerline path or output path
        geo_name = os.path.basename(centerline_path).replace(".vtp", "")
        if not geo_name:
            geo_name = os.path.basename(os.path.dirname(output_path))

    # Create temporary directory for ROM workflow files
    temp_dir = os.path.join(os.path.dirname(output_path), "rom_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate inflow flow file
    inflow_flow_file = os.path.join(temp_dir, f"{geo_name}_inflow_0D.flow")
    with open(inflow_flow_file, "w") as f:
        np.linspace(0, num_time_steps * dt, num_time_steps)
        q = np.ones(num_time_steps) * 85  # Default flow value
        for i in range(num_time_steps):
            f.write(f"{i * dt:.5f}    {q[i]:.3f}\n")

    # Create ROM input generator script
    rom_script = os.path.join(temp_dir, f"{geo_name}_rom_generator.py")

    # Make centerline path relative to script location or absolute
    centerline_abs = os.path.abspath(centerline_path)

    # Strip .vtp extension from face names for SimVascular (XML uses base names)
    inlet_cap_base = inlet_cap.replace(".vtp", "") if inlet_cap.endswith(".vtp") else inlet_cap
    outlet_caps_base = [cap.replace(".vtp", "") if cap.endswith(".vtp") else cap for cap in outlet_caps]

    # Format outlet_caps_base as a Python list literal for the script
    outlet_caps_base_str = "[" + ", ".join([f"'{cap}'" for cap in outlet_caps_base]) + "]"

    # Extract outlet resistances from XML file (if available)
    print("\nExtracting outlet resistances from XML file...")
    print(f"  Looking for outlet caps: {outlet_caps}")
    # Create mapping from base names (without .vtp) to full names (with .vtp)
    # XML BC names don't include .vtp extension
    outlet_caps_base_dict = {}
    for cap in outlet_caps:
        base_name = cap.replace(".vtp", "") if cap.endswith(".vtp") else cap
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
        outlet_cap_base = outlet_cap.replace(".vtp", "") if outlet_cap.endswith(".vtp") else outlet_cap
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

    with open(rom_script, "w") as f:
        f.write(script_content)

    # Find SimVascular executable
    if simvascular_path is None:
        # Try common locations
        possible_paths = [
            "/Applications/SimVascular.app/Contents/Resources/simvascular",
            os.path.expanduser("~/Applications/SimVascular.app/Contents/Resources/simvascular"),
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

    print("Running SimVascular ROM workflow...")
    print(f"  Script: {rom_script}")
    print(f"  SimVascular: {simvascular_path}")

    # Run SimVascular ROM script
    cmd = [simvascular_path, "--python", "--", rom_script]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running SimVascular ROM workflow:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"SimVascular ROM workflow failed with return code {result.returncode}")

    # Find the generated JSON file (SimVascular creates solver_0d.json in output_dir)
    generated_json = os.path.join(os.path.dirname(output_path), "solver_0d.json")

    if not os.path.exists(generated_json):
        raise FileNotFoundError(f"Generated JSON file not found: {generated_json}")

    # Read and fix junction types (replace "internal_junction" with "NORMAL_JUNCTION")
    print("Post-processing generated JSON file...")
    with open(generated_json) as f:
        zerod_input = json.load(f)

    # Set capacitances to cap_val
    cap_value = 1e-10
    for vessel in zerod_input["vessels"]:
        vessel["zero_d_element_values"]["C"] = cap_value
    zerod_input["simulation_parameters"]["number_of_cardiac_cycles"] = (
        get_pipeline_config().solver.number_of_cardiac_cycles
    )

    # Fix junction types and validate junction structure
    if "junctions" in zerod_input:
        for junc in zerod_input["junctions"]:
            if junc.get("junction_type") == "internal_junction":
                junc["junction_type"] = "NORMAL_JUNCTION"

            # Check for multiple inlets (not supported by BloodVessel junction)
            inlet_vessels = junc.get("inlet_vessels", [])
            if len(inlet_vessels) > 1:
                print(f"  Warning: Junction {junc.get('junction_name')} has {len(inlet_vessels)} inlets.")
                print("    BloodVessel junction only supports 1 inlet. Keeping first inlet only.")
                # Keep only the first inlet
                junc["inlet_vessels"] = [inlet_vessels[0]]

    # Remove unused resistance BCs (BCs that are not referenced by any vessel)
    print("\n  Removing unused resistance BCs...")
    if "boundary_conditions" in zerod_input:
        # Find all resistance BC names that are actually used by vessels
        used_resistance_bcs = set()
        for vessel in zerod_input.get("vessels", []):
            if "boundary_conditions" in vessel and "outlet" in vessel["boundary_conditions"]:
                bc_name = vessel["boundary_conditions"]["outlet"]
                used_resistance_bcs.add(bc_name)

        # Filter boundary conditions to keep only:
        # 1. Non-RESISTANCE BCs (like INFLOW)
        # 2. RESISTANCE BCs that are actually used by vessels
        original_bcs = zerod_input["boundary_conditions"]
        filtered_bcs = []
        removed_count = 0

        for bc in original_bcs:
            if bc.get("bc_type") != "RESISTANCE":
                # Keep all non-resistance BCs
                filtered_bcs.append(bc)
            elif bc.get("bc_name") in used_resistance_bcs:
                # Keep resistance BCs that are used
                filtered_bcs.append(bc)
            else:
                # Remove unused resistance BCs
                removed_count += 1

        zerod_input["boundary_conditions"] = filtered_bcs

        if removed_count > 0:
            print(f"    Removed {removed_count} unused resistance BC(s)")
        print(f"    Kept {len(used_resistance_bcs)} used resistance BC(s)")

    zerod_input["simulation_parameters"]["number_of_cardiac_cycles"] = (
        get_pipeline_config().solver.number_of_cardiac_cycles
    )
    # Move to final output location
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
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
    with open(output_path) as f:
        zerod_input = json.load(f)

    # Create simplified vessel_bc_map for compatibility
    vessel_bc_map = {}
    if "boundary_conditions" in zerod_input:
        for bc in zerod_input["boundary_conditions"]:
            bc_name = bc.get("bc_name", "")
            vessel_bc_map[bc_name] = {
                "name": "branch0_seg0",  # Simplified
                "pressure": "pressure",
                "flow": "flow",
            }

    print(f"Geometric 0D input saved to: {output_path}")

    return zerod_input, vessel_bc_map
