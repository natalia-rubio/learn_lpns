#!/usr/bin/env python3
"""
Generate svZeroDSolver input files from centerline geometry and simulation results.
This script creates:
1. Geometric 0D input files (from centerline geometry)
2. Calibration input files (from 3D or 1D results)
3. Calibrated output files (by running svZeroDCalibrator)

Based on the workflow in richter2024-paper-tools.
"""

import argparse
import json
import os
import subprocess
import sys

from learn_lpns.config import get_pipeline_config
from learn_lpns.visualizations.run_zerod_comparison_plots import run_zerod_comparison_plots
from learn_lpns.zerod_calibration.bc_fitting import (
    apply_fitted_outlet_bcs_to_file,
    fit_bcs_from_observations,
)
from learn_lpns.zerod_calibration.bifurcation_splitting import (
    adjust_junction_boundaries_by_entrance_length_from_files,
    convert_el_normal_junctions_to_blood_vessel_junction,
    split_junctions_from_files,
)
from learn_lpns.zerod_calibration.calibration import create_calibration_input, run_calibration
from learn_lpns.zerod_calibration.centerline_path_extraction import process_geometric_input
from learn_lpns.zerod_calibration.forward_mse import calculate_mse_between_3d_and_0d
from learn_lpns.zerod_calibration.forward_simulation import run_forward_simulation
from learn_lpns.zerod_calibration.generate_baseline_0d import create_geometric_zerod_input_rom
from learn_lpns.zerod_calibration.generate_zerod_inputs_cli import (
    DEFAULT_JUNCTION_TYPES,
    add_generate_zerod_inputs_arguments,
    prepare_generate_zerod_namespace,
)
from learn_lpns.zerod_calibration.geometric_params import extract_and_add_geometric_params
from learn_lpns.zerod_calibration.inflow_handling import (
    refine_inlet_bc_for_forward_simulation,
    sync_nn_config_bcs_from_calibration,
    update_geometric_input_with_calibration_bc,
)
from learn_lpns.zerod_calibration.modality_paths import modality_csv_paths, nn_forward_sim_specs
from learn_lpns.zerod_calibration.oned_to_zerod import (
    extract_observations_from_1d,
    extract_observations_from_1d_with_node_ids,
)
from learn_lpns.zerod_calibration.run_config_canonical import DEFAULT_CLI_RUN_CONFIG
from learn_lpns.zerod_calibration.tools.file_io import (
    get_paths,
    load_from_json,
    save_to_json,
    standard_0d_json_path,
    timestep_from_1D,
)
from learn_lpns.zerod_calibration.zerod_handling import modify_junction_types

JUNCTION_TYPE = DEFAULT_JUNCTION_TYPES[0]


def _assert_multi_outlet_junctions_type(nn_config, junction_type):
    """Multi-outlet junctions must already use the calibratable junction type (Step 1)."""
    for junc in nn_config.get("junctions", []):
        if len(junc.get("outlet_vessels", [])) >= 2:
            if junc.get("junction_type") != junction_type:
                jname = junc.get("junction_name", "?")
                raise ValueError(
                    f"Junction {jname!r} has junction_type {junc.get('junction_type')!r} "
                    f"(expected {junction_type}). Re-run Step 1 geometric params / "
                    f"junction conversion for this variant."
                )


def _convert_bifurcations_junctions(config_path):
    """Convert multi-outlet NORMAL_JUNCTIONs to BloodVesselJunction using geometric_params."""
    with open(config_path) as f:
        cfg = json.load(f)
    convert_el_normal_junctions_to_blood_vessel_junction(cfg)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)


def _bc_source_for_nn_forward(geo_variant_paths, junction_type=JUNCTION_TYPE):
    """BC JSON for inlet refinement on NN forward sims (calibration input, else geometric)."""
    calib = geo_variant_paths["junction_types"][junction_type]["calibration_input"]
    geometric = geo_variant_paths["geometric_input"]
    if os.path.exists(calib):
        return calib
    if os.path.exists(geometric):
        return geometric
    raise FileNotFoundError(f"No BC source found: neither {calib} nor {geometric}")


def _run_forward_simulation_step(
    sim_input,
    results_csv,
    *,
    bc_source,
    step_label,
    check_and_track_file,
    generated_files,
    skip_missing_input=False,
):
    """Run one forward sim with inlet BC refinement; honor no-redo via check_and_track_file."""
    if check_and_track_file(results_csv, step_label):
        return
    if not os.path.exists(sim_input):
        if skip_missing_input:
            print(f"      ⊘ Skipping: input not found: {sim_input}")
            return
        raise FileNotFoundError(f"Forward sim input not found: {sim_input} ({step_label})")
    print(f"\n    Running {step_label}...")
    try:
        refine_inlet_bc_for_forward_simulation(sim_input, calibration_input_path=bc_source)
        run_forward_simulation(sim_input, results_csv)
        generated_files.append(results_csv)
        print(f"      ✓ {step_label} completed successfully")
    except Exception as e:
        raise Exception(f"{step_label} failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Generate svZeroDSolver input files and run calibration")
    add_generate_zerod_inputs_arguments(parser, require_set_geo=True)

    args = parser.parse_args()
    verbose = args.verbose
    run_config_suffix = prepare_generate_zerod_namespace(args)
    if not run_config_suffix:
        run_config_suffix = DEFAULT_CLI_RUN_CONFIG
    print(f"  Run config: {run_config_suffix}")

    # Track generated/existing files for logging
    generated_files = []
    output_dir = "data/zeroD"
    base_dir = os.path.join(output_dir, args.set_name, run_config_suffix, args.geo_name)

    geometry_variants, centerline_path, geo_dir = get_paths(base_dir, args)
    geometric_input_path = geometry_variants["original"]["geometric_input"]

    # Helper function to check and track files
    def check_and_track_file(file_path, step_name):
        """Check if file exists, track it, and return whether to skip."""
        if os.path.exists(file_path):
            generated_files.append(file_path)
            if args.no_redo:
                print(f"    ⊘ Skipping {step_name} (file already exists: {file_path})")
                return True
            else:
                print(f"    ⊘ File exists but will be regenerated: {file_path}")
        return False

    # Step 1: Create geometric 0D input using SimVascular ROM workflow
    if not args.skip_base_generation:
        bifurcations_geometric_input_path = geometry_variants["bifurcations"]["geometric_input"]

        # Check if base files exist
        skip_base = False
        if args.no_redo:
            bifurcations_EL_geometric_input_path = geometry_variants["bifurcations_EL"]["geometric_input"]
            # Make sure all necessarybase files exist
            if (
                os.path.exists(geometric_input_path)
                and os.path.exists(bifurcations_geometric_input_path)
                and os.path.exists(bifurcations_EL_geometric_input_path)
            ):
                check_and_track_file(geometric_input_path, "base generation")
                check_and_track_file(bifurcations_geometric_input_path, "bifurcations generation")
                check_and_track_file(bifurcations_EL_geometric_input_path, "EL-adjusted bifurcations generation")
                skip_base = True

        if not skip_base:
            if "VMR" in args.set_name:
                std_json_path = standard_0d_json_path("data", args.set_name, args.geo_name)
                zerod_input = load_from_json(std_json_path)
                zerod_input["simulation_parameters"]["output_all_cycles"] = True
                zerod_input["simulation_parameters"]["number_of_cardiac_cycles"] = (
                    get_pipeline_config().solver.number_of_cardiac_cycles
                )
                os.makedirs(os.path.dirname(geometric_input_path), exist_ok=True)
                save_to_json(zerod_input, geometric_input_path)
                generated_files.append(geometric_input_path)
                print(f"    ✓ Standard 0D input saved to: {geometric_input_path}")
            else:
                print("\n" + "=" * 60)
                print("Step 1: Creating geometric 0D input file using SimVascular ROM")
                print("=" * 60)

                # Get geometry directory (may not exist for VMR files)
                if not os.path.exists(geo_dir):
                    print(f"  Warning: Geometry directory not found: {geo_dir}")
                    print("  Will use centerline-based workflow (for VMR files)")
                    geo_dir = None

                zerod_input, _vessel_bc_map = create_geometric_zerod_input_rom(
                    geo_dir,
                    centerline_path,
                    geometric_input_path,
                    simvascular_path=args.simvascular_path,
                    dt=args.dt,
                    num_time_steps=args.num_time_steps,
                    num_cardiac_cycles=args.num_cardiac_cycles,
                )
                generated_files.append(geometric_input_path)

            print("\n  Adding centerline parameters to geometric input...")
            geometric_centerline_input_path = geometric_input_path.replace(
                "geometric_input", "geometric_centerline_input"
            )
            process_geometric_input(
                centerline_path, geometric_input_path, geometric_centerline_input_path, verbose=verbose
            )
            print(f"  Centerline parameters added to geometric input saved to: {geometric_centerline_input_path}")

            # Generate bifurcations-only version of the geometric input
            print("\n  Creating bifurcations-only geometric input...")
            split_junctions_from_files(
                geometric_centerline_input_path,
                centerline_path,
                bifurcations_geometric_input_path,
                verbose=verbose,
            )
            generated_files.append(bifurcations_geometric_input_path)
            print(f"  Bifurcations-only geometric input saved to: {bifurcations_geometric_input_path}")

            # Generate entrance length-adjusted bifurcations version (if bifurcations file exists)
            bifurcations_EL_geometric_input_path = geometry_variants["bifurcations_EL"]["geometric_input"]

            print("\n  Creating entrance length-adjusted bifurcations geometric input...")
            adjust_junction_boundaries_by_entrance_length_from_files(
                bifurcations_geometric_input_path,
                centerline_path,
                bifurcations_EL_geometric_input_path,
                verbose=verbose,
            )
            generated_files.append(bifurcations_EL_geometric_input_path)
            print(
                "  Entrance length-adjusted bifurcations geometric input saved to: "
                f"{bifurcations_EL_geometric_input_path}"
            )

    # Extract geometric params for bifurcations and bifurcations_EL variants
    geometry_variants["bifurcations"]["geometric_input"]

    for geo_variant_name in ("bifurcations", "bifurcations_EL"):
        path = geometry_variants[geo_variant_name]["geometric_input"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Geometric input not found: {path}")
        if check_and_track_file(path, f"geometric params extraction for {geo_variant_name}"):
            continue

        extract_and_add_geometric_params(centerline_path, path, verbose=verbose)
        print(f"  Geometric parameters extracted and added to {path}")

        if geo_variant_name == "bifurcations":
            _convert_bifurcations_junctions(path)
            print(f"  Bifurcations: multi-outlet junctions -> BloodVesselJunction in {path}")

    # Step 2: Extract observations and create calibration inputs for each junction type
    if not args.skip_observation:
        # Check if all calibration inputs already exist (if --no_redo is set)
        all_calibration_inputs_exist = True
        if args.no_redo:
            for _geo_variant_name, geo_variant_paths in geometry_variants.items():
                variant_calibration_input = geo_variant_paths["calibration_input"]
                if not os.path.exists(variant_calibration_input):
                    all_calibration_inputs_exist = False
                    break
                # Also check junction-type specific calibration inputs
                variant_junction_paths = geo_variant_paths["junction_types"]
                jtype_input_path = variant_junction_paths[JUNCTION_TYPE]["calibration_input"]
                if not os.path.exists(jtype_input_path):
                    all_calibration_inputs_exist = False
                    break

        if all_calibration_inputs_exist and args.no_redo:
            print("\n" + "=" * 60)
            print("Step 2: Creating calibration input files for each junction type")
            print("=" * 60)
            print("  ⊘ Skipping observation extraction and calibration input creation (all files already exist)")
        else:
            print("\n" + "=" * 60)
            print("Step 2: Creating calibration input files for each junction type")
            print("=" * 60)

            soln_path = os.path.join("data", "oneD", "VMR", args.geo_name, "unsteady_soln.vtp")
            geometric_input_path = geometry_variants["original"]["geometric_input"]
            geo_dir = os.path.join("data", "threeD", args.set_name, args.geo_name)
            if not os.path.exists(soln_path):
                raise FileNotFoundError(f"1D solution not found: {soln_path}")

            # --- Extract 1D observations for each geometry variant ---
            variant_observations = {}

            print("\n  Extracting observations for original geometry...")
            variant_observations["original"] = extract_observations_from_1d(
                soln_path, geometric_input_path, geo_dir=geo_dir, start_idx=0
            )

            for geo_variant_name in ("bifurcations", "bifurcations_EL"):
                if geo_variant_name not in geometry_variants:
                    continue
                variant_geometric_input = geometry_variants[geo_variant_name]["geometric_input"]
                print(
                    f"\n  Extracting observations for {geo_variant_name} "
                    f"(centerline_node_ids + connector flow cascade)..."
                )
                variant_observations[geo_variant_name] = extract_observations_from_1d_with_node_ids(
                    soln_path,
                    variant_geometric_input,
                    geo_dir=geo_dir,
                    start_idx=0,
                    derivative_method="central",
                    cascade_split_connector_flows=True,
                )

            # --- Fit outlet BCs from original-geometry observations ---
            time_step_size = None
            try:
                time_step_size = timestep_from_1D(soln_path, geo_dir)
            except Exception:
                time_step_size = None
            try:
                fitted_outlet_bcs = fit_bcs_from_observations(
                    geometric_input_path, variant_observations["original"], dt=time_step_size
                )
            except Exception as e:
                raise Exception(f"Outlet BC fitting failed: {e}") from e

            # --- Create calibration inputs per geometry variant ---
            for geo_variant_name, geo_variant_paths in geometry_variants.items():
                print("\n" + "-" * 50)
                print(f"Processing {geo_variant_name.upper()} geometry variant")
                print("-" * 50)

                variant_geometric_input = geo_variant_paths["geometric_input"]
                variant_calibration_input = geo_variant_paths["calibration_input"]
                variant_junction_paths = geo_variant_paths["junction_types"]

                if fitted_outlet_bcs:
                    apply_fitted_outlet_bcs_to_file(
                        variant_geometric_input,
                        fitted_outlet_bcs,
                        f"{geo_variant_name} geometric input",
                    )

                # Create base calibration input for this geometry variant
                if check_and_track_file(variant_calibration_input, f"base calibration input for {geo_variant_name}"):
                    # File exists and no-redo is set, skip creation
                    pass
                else:
                    print(f"\n  Creating base calibration input for {geo_variant_name}...")
                    try:
                        obs_for_calib = variant_observations[geo_variant_name]

                        create_calibration_input(
                            variant_geometric_input,
                            obs_for_calib,
                            variant_calibration_input,
                            centerline_soln_path=soln_path,
                            geo_dir=geo_dir,
                            quadratic_resistor=getattr(args, "quadratic_resistor", False),
                            penalty_on=getattr(args, "penalty_on", False),
                            set_name=getattr(args, "set_name", None),
                        )
                        generated_files.append(variant_calibration_input)
                        print(f"    ✓ Base calibration input saved to: {variant_calibration_input}")

                        # Update geometric input with BC from calibration input
                        update_geometric_input_with_calibration_bc(variant_geometric_input, variant_calibration_input)
                    except Exception as e:
                        raise Exception(f"Failed to create calibration input for {geo_variant_name}: {e}") from e

                # Create calibration input variants for each junction type
                print("\n  Creating calibration input variants for each junction type...")
                try:
                    with open(variant_calibration_input) as f:
                        base_calibration_config = json.load(f)

                    jtype_input_path = variant_junction_paths[JUNCTION_TYPE]["calibration_input"]
                    if check_and_track_file(jtype_input_path, f"{geo_variant_name}/{JUNCTION_TYPE} calibration input"):
                        pass
                    else:
                        print(f"    Creating {geo_variant_name}/{JUNCTION_TYPE} calibration input...")

                        # Apply junction type modification to calibration input
                        jtype_config = modify_junction_types(base_calibration_config, JUNCTION_TYPE)

                        with open(jtype_input_path, "w") as f:
                            json.dump(jtype_config, f, indent=4)
                        generated_files.append(jtype_input_path)
                except Exception as e:
                    raise Exception(
                        f"Failed to create junction type calibration inputs for {geo_variant_name}: {e}"
                    ) from e

    # Step 3: Run calibration for each junction type
    if not args.skip_calibration:
        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            print(f"\n  Running calibration for {geo_variant_name} geometry...")
            variant_junction_paths = geo_variant_paths["junction_types"]

            jtype_input_path = variant_junction_paths[JUNCTION_TYPE]["calibration_input"]
            jtype_output_path = variant_junction_paths[JUNCTION_TYPE]["calibrated_output"]

            if check_and_track_file(jtype_output_path, f"calibration for {geo_variant_name}/{JUNCTION_TYPE}"):
                continue

            print(f"\n    Calibrating {geo_variant_name}/{JUNCTION_TYPE}...")

            try:
                run_calibration(
                    jtype_input_path,
                    jtype_output_path,
                    backend=getattr(args, "calibration_backend", "decoupled_ls"),
                    plot_rsl_fits=getattr(args, "plot_rsl_fits", False),
                    set_name=args.set_name,
                    geo_name=args.geo_name,
                )
                generated_files.append(jtype_output_path)
                print(f"      ✓ Calibration completed for {geo_variant_name}/{JUNCTION_TYPE}")
            except Exception as e:
                raise Exception(f"Calibration failed for {geo_variant_name}/{JUNCTION_TYPE}: {e}") from e

    if not getattr(args, "plots_only", False):
        # Step 3.5: ML prep — per-geo CSVs + jax_arrays (run_data_processing)
        # TODO: Consider no_redo / skip-if-done when ml_inputs CSVs already exist.
        # run_data_processing writes per-geo CSVs and jax_arrays pickles under
        # data/jax_arrays/.../forward/ (Step 3.7 loads the forward pickle for NN inference).
        print(f"\n  Running data processing pipeline for neural network ({args.geometry_variant})...")
        geo_variant_paths = geometry_variants[args.geometry_variant]
        variant_geometric_input = geo_variant_paths["geometric_input"]
        calib_output_path = geo_variant_paths["junction_types"][JUNCTION_TYPE]["calibrated_output"]
        nn_vessel_flag = getattr(args, "Vessel_NN", False)
        nn_json_by_key = {
            key: sim_input
            for key, sim_input, _ in nn_forward_sim_specs(
                base_dir,
                args.geometry_variant,
                JUNCTION_TYPE,
                nn_vessel_flag,
            )
        }
        nn_output_path = nn_json_by_key["BloodVesselJunction_NN"]
        nn_junction_and_vessel_path = nn_json_by_key.get("BloodVesselJunction_NN_plus_Vessel_NN")
        nn_vessel_only_path = nn_json_by_key.get("Vessel_NN")

        if not os.path.exists(variant_geometric_input):
            raise FileNotFoundError(f"Geometric input not found: {variant_geometric_input}")
        if not os.path.exists(calib_output_path):
            raise FileNotFoundError(f"Calibration output not found: {calib_output_path}")

        geometric_results_path = geo_variant_paths["geometric_results"]
        if not os.path.exists(geometric_results_path):
            print(
                f"\n  Geometric results not found ({geometric_results_path}); "
                "running geometric forward (required for flow_split features in ML CSVs)..."
            )
            _run_forward_simulation_step(
                variant_geometric_input,
                geometric_results_path,
                bc_source=geo_variant_paths["calibration_input"],
                step_label=f"geometric forward simulation for {args.geometry_variant}",
                check_and_track_file=check_and_track_file,
                generated_files=generated_files,
            )

        run_data_processing_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "data_processing", "run_data_processing.py"),
            "--set_name",
            args.set_name,
            "--geometry_variant",
            args.geometry_variant,
            "--set_type",
            "forward",
            "--geometries",
            args.geo_name,
            "--percent_train",
            "1",
            "--run_config",
            run_config_suffix,
        ]
        if verbose:
            run_data_processing_cmd.append("--verbose")
        from learn_lpns.zerod_calibration.nn_inference import forward_jax_pickle_path

        forward_jax_path = forward_jax_pickle_path(
            "data",
            args.set_name,
            run_config_suffix,
            args.geometry_variant,
            num_geos=1,
        )
        try:
            result = subprocess.run(
                run_data_processing_cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
            if result.returncode != 0:
                err = result.stderr or result.stdout or "(no output)"
                raise RuntimeError(f"Data processing failed: {err}")
            if not os.path.exists(forward_jax_path):
                detail = (result.stderr or "") + (result.stdout or "")
                raise RuntimeError(
                    f"Data processing exited 0 but jax pickle not written: {forward_jax_path}. "
                    f"Output:\n{detail or '(no subprocess output)'}"
                )
            print(f"  ✓ Data processing pipeline completed ({args.geometry_variant})")
        except Exception as e:
            raise Exception(f"Failed to run data processing pipeline: {e}") from e

        skip_nn_inference = getattr(args, "skip_nn_inference", False)
        if skip_nn_inference:
            print("\n  ⊘ Skipping NN inference (Steps 3.7/3.8); --skip_steps includes nn_inference")
        else:
            # Step 3.7: Junction NN inference (loads forward jax pickle from Step 3.5)
            if check_and_track_file(nn_output_path, f"NN inference for {args.geometry_variant}/{JUNCTION_TYPE}"):
                pass
            else:
                print(f"\n    Running neural network inference for {args.geometry_variant}/{JUNCTION_TYPE}...")
                try:
                    from learn_lpns.tools.basic import load_dict
                    from learn_lpns.zerod_calibration.nn_inference import run_junction_inference

                    jax_path = forward_jax_path
                    if not os.path.exists(jax_path):
                        raise FileNotFoundError(
                            f"Junction jax pickle not found: {jax_path}. "
                            f"Run Step 3.5 (run_data_processing) or the full pipeline first."
                        )

                    with open(variant_geometric_input) as f:
                        nn_config = json.load(f)
                    _assert_multi_outlet_junctions_type(nn_config, JUNCTION_TYPE)

                    print(f"  Loading junction features from jax pickle: {jax_path}")
                    jax_data_dict = load_dict(jax_path)

                    if getattr(args, "model_dir", None):
                        model_dir = args.model_dir
                    else:
                        model_dir = os.path.join("results", "models", args.set_name, args.geometry_variant)

                    run_junction_inference(
                        jax_data_dict=jax_data_dict,
                        nn_config=nn_config,
                        set_name=args.set_name,
                        geo_name=args.geo_name,
                        model_dir=model_dir,
                        junction_type=JUNCTION_TYPE,
                        quadratic_resistor=getattr(args, "quadratic_resistor", False),
                        multi_output_rri=args.multi_output_rri,
                        clip_predictions=args.clip_predictions,
                        run_config_suffix=run_config_suffix,
                        data_root=getattr(args, "data_root", "data"),
                    )

                    with open(nn_output_path, "w") as f:
                        json.dump(nn_config, f, indent=4)
                    generated_files.append(nn_output_path)
                    print(f"      ✓ Neural network predictions applied and saved to {nn_output_path}")

                except Exception as e:
                    raise Exception(
                        f"Neural network inference failed for {args.geometry_variant}/{JUNCTION_TYPE}: {e}"
                    ) from e

            # Step 3.8 (optional): Vessel NN inference: predict vessel R/S/L and write NN_JunctionAndVessel config
            if getattr(args, "Vessel_NN", False):
                from learn_lpns.zerod_calibration.nn_inference import run_vessel_inference

                if not os.path.exists(nn_output_path):
                    raise FileNotFoundError(f"NN junction config not found: {nn_output_path}")

                if check_and_track_file(
                    nn_junction_and_vessel_path, f"Vessel NN inference for {args.geometry_variant}"
                ) and os.path.exists(nn_vessel_only_path):
                    pass
                else:
                    print(f"\n    Running vessel NN inference for {args.geometry_variant}...")
                    try:
                        with open(nn_output_path) as f:
                            junction_nn_config = json.load(f)
                        junction_and_vessel_config, vessel_only_config = run_vessel_inference(
                            junction_nn_config=junction_nn_config,
                            variant_geometric_input=variant_geometric_input,
                            set_name=args.set_name,
                            geometry_variant=args.geometry_variant,
                            model_dir=getattr(args, "model_dir", None),
                            quadratic_resistor=getattr(args, "quadratic_resistor", False),
                            multi_output_rri=args.multi_output_rri,
                            clip_predictions=args.clip_predictions,
                            run_config_suffix=run_config_suffix,
                            data_root=getattr(args, "data_root", "data"),
                            verbose=args.verbose,
                        )
                        with open(nn_junction_and_vessel_path, "w") as f:
                            json.dump(junction_and_vessel_config, f, indent=4)
                        generated_files.append(nn_junction_and_vessel_path)
                        print(f"      ✓ Vessel NN predictions applied and saved to {nn_junction_and_vessel_path}")
                        with open(nn_vessel_only_path, "w") as f:
                            json.dump(vessel_only_config, f, indent=4)
                        generated_files.append(nn_vessel_only_path)
                        print(f"      ✓ Vessel_NN (geometric junctions + NN vessels) saved to {nn_vessel_only_path}")
                    except Exception as e:
                        raise Exception(f"Vessel NN inference failed for {args.geometry_variant}: {e}") from e

            # Sync BCs from calibrated output into NN configs so RCR (and other outlet BCs) match
            if not os.path.exists(nn_output_path):
                raise FileNotFoundError(f"NN junction config not found: {nn_output_path}")
            nn_paths = [
                p for p in [nn_output_path, nn_junction_and_vessel_path, nn_vessel_only_path] if p and os.path.exists(p)
            ]
            n_updated = sync_nn_config_bcs_from_calibration(nn_paths, calib_output_path, verbose=args.verbose)
            if n_updated and args.verbose:
                print(
                    f"  Synced boundary conditions from calibrated output into {n_updated} "
                    f"NN config(s) for {args.geometry_variant}"
                )

        # Step 4: Run forward simulations for each geometry variant
        if not args.skip_forward:
            forward_variant_names = [args.geometry_variant] if args.NN_only else list(geometry_variants.keys())
            if args.NN_only:
                print(f"\n  Running forward simulation (NN-only mode, {args.geometry_variant})...")

            for geo_variant_name in forward_variant_names:
                if geo_variant_name not in geometry_variants:
                    continue
                geo_variant_paths = geometry_variants[geo_variant_name]
                variant_junction_paths = geo_variant_paths["junction_types"]

                if not args.NN_only:
                    # Run geometric forward simulation
                    print(f"\n  Running forward simulations for {geo_variant_name} geometry...")
                    _run_forward_simulation_step(
                        geo_variant_paths["geometric_input"],
                        geo_variant_paths["geometric_results"],
                        bc_source=geo_variant_paths["calibration_input"],
                        step_label=f"geometric forward simulation for {geo_variant_name}",
                        check_and_track_file=check_and_track_file,
                        generated_files=generated_files,
                        skip_missing_input=True,
                    )

                    # Run calibrated forward simulation
                    jtype_output = variant_junction_paths[JUNCTION_TYPE]["calibrated_output"]
                    if not os.path.exists(jtype_output):
                        print(f"      ✗ Skipping: calibrated output not found: {jtype_output}")
                    else:
                        _run_forward_simulation_step(
                            jtype_output,
                            variant_junction_paths[JUNCTION_TYPE]["calibrated_results"],
                            bc_source=variant_junction_paths[JUNCTION_TYPE]["calibration_input"],
                            step_label=(f"calibrated forward simulation for {geo_variant_name}/{JUNCTION_TYPE}"),
                            check_and_track_file=check_and_track_file,
                            generated_files=generated_files,
                        )

                if geo_variant_name not in ("bifurcations", "bifurcations_EL"):
                    continue

                bc_source = _bc_source_for_nn_forward(geo_variant_paths)
                for _, sim_input, results_csv in nn_forward_sim_specs(
                    base_dir,
                    geo_variant_name,
                    JUNCTION_TYPE,
                    getattr(args, "Vessel_NN", False),
                ):
                    _run_forward_simulation_step(
                        sim_input,
                        results_csv,
                        bc_source=bc_source,
                        step_label=f"NN forward simulation for {geo_variant_name}/{JUNCTION_TYPE}",
                        check_and_track_file=check_and_track_file,
                        generated_files=generated_files,
                        skip_missing_input=not args.NN_only,
                    )

        # Step 5: Calculate and print MSE between 3D and 0D solutions
        if not args.skip_mse_calculation:
            print("\n" + "=" * 60)
            print("Step 5: Calculating MSE between 3D and 0D solutions")
            print("=" * 60)

            print(f"\n  MSE calculation for {args.geometry_variant.upper()} geometry:")
            geo_variant_paths = geometry_variants[args.geometry_variant]
            csv_results_dict = modality_csv_paths(
                geo_variant_paths,
                base_dir,
                args.geometry_variant,
                JUNCTION_TYPE,
                getattr(args, "Vessel_NN", False),
            )
            variant_calibration_input = geo_variant_paths["calibration_input"]
            if not csv_results_dict or not os.path.exists(variant_calibration_input):
                print(f"    Skipping MSE calculation for {args.geometry_variant} (missing files)")
            else:
                mse_csv_path = os.path.join(base_dir, f"{args.geometry_variant}_mse_comparison.csv")
                if check_and_track_file(mse_csv_path, f"MSE calculation for {args.geometry_variant}"):
                    pass
                else:
                    try:
                        calculate_mse_between_3d_and_0d(
                            variant_calibration_input,
                            csv_results_dict,
                            output_csv_path=mse_csv_path,
                            verbose=verbose,
                        )
                        generated_files.append(mse_csv_path)
                    except Exception as e:
                        raise Exception(f"Error calculating MSE for {args.geometry_variant}: {e}") from e

    # Step 6: Generate comparison plots (always run if not skipped, including in plot-only mode)
    if not args.skip_plots:
        print("\n" + "=" * 60)
        print("Step 6: Generating comparison plots")
        print("=" * 60)
        try:
            run_zerod_comparison_plots(
                args.geometry_variant,
                geometry_variants,
                base_dir,
                JUNCTION_TYPE,
                set_name=args.set_name,
                geo_name=args.geo_name,
                run_config=run_config_suffix,
                trial_id=args.trial_id,
                nn_vessel=getattr(args, "Vessel_NN", False),
                verbose=verbose,
            )
        except Exception as e:
            print(f"  ✗ Error generating plots: {e}")
            if getattr(args, "plots_only", False):
                raise
            if verbose:
                import traceback

                traceback.print_exc()

    if verbose:
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
        geo_variant_paths = geometry_variants[args.geometry_variant]
        jpaths = geo_variant_paths["junction_types"][JUNCTION_TYPE]
        print(f"\n{args.geometry_variant.upper()} geometry:")
        print(f"  Geometric input: {geo_variant_paths['geometric_input']}")
        print(f"  Geometric simulation results: {geo_variant_paths['geometric_results']}")
        if not args.skip_calibration:
            print(f"  Base calibration input: {geo_variant_paths['calibration_input']}")
            print(f"  Junction type ({JUNCTION_TYPE}):")
            print(f"    Calibration input: {jpaths['calibration_input']}")
            print(f"    Calibrated output: {jpaths['calibrated_output']}")
            print(f"    Simulation results: {jpaths['calibrated_results']}")

    # Output generated files as JSON if --no_redo was used (for batch script to parse)
    if args.no_redo:
        print(json.dumps({"generated_files": generated_files}))


if __name__ == "__main__":
    main()

# python3 learn_lpns/zerod_calibration/generate_zerod_inputs.py --set_name set_3 --geo_name tree_007
