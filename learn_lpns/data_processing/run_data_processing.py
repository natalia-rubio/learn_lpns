"""
For a set name and list of geometries, run the full data processing pipeline:

1) Extract geometric features from 0D config -> `geometric_features.csv`
2) Extract calibrated junction lumped parameters -> `junction_lumped_parameters.csv`
3) Build concatenated `data_dict` for JAX NN training -> `data/jax_arrays/.../jax_arrays_num_geos_<N>_{unclipped,clipped}.pkl`
4) Generate train/val split indices -> `data/split_indices/.../train_val_ind_<set_name>_num_geos_<N>`
"""

import argparse
import csv
import glob
import os

from learn_lpns.config import get_pipeline_config
from learn_lpns.data_processing.data_dict_from_csvs import (
    build_data_dict_from_csvs,
    build_data_dict_from_vessel_csvs,
)
from learn_lpns.data_processing.generate_split_indices import (
    build_geometry_index_map,
    build_split_dict,
    generate_geometry_split,
    resolve_flat_indices,
    resolve_geometry_row_ranges_from_jax_dict,
    write_geometries_txt,
)
from learn_lpns.data_processing.inputs_from_0d_config import (
    FLOW_SPLIT_METHOD_MEAN_OVER_TIME,
    FLOW_SPLIT_METHOD_PEAK_INLET_FLOW,
    load_junction_geometric_features,
    load_vessel_geometric_features,
    load_vessel_targets_from_config,
)
from learn_lpns.data_processing.outputs_from_config import load_junction_lumped_parameters
from learn_lpns.data_processing.jax_arrays_paths import jax_arrays_filename
from learn_lpns.data_processing.stenosis_clipping import (
    export_clipped_lumped_parameter_csvs,
    write_clipped_jax_pickles,
)
from learn_lpns.tools.basic import load_dict, save_dict
from learn_lpns.zerod_calibration.run_config_canonical import DEFAULT_CLI_RUN_CONFIG


def discover_geometries_with_csvs(set_name, geometry_variant="bifurcations", data_root="data", run_config_suffix=None):
    """
    Discover all geometries that have both geometric_features.csv and junction_lumped_parameters.csv.

    Args:
        set_name: Set name (e.g., VMR)
        geometry_variant: Geometry variant name (e.g., "bifurcations" or "bifurcations_EL")
        data_root: Repo data root (default: data)
        run_config_suffix: If set, ml_inputs path is .../set_name/run_config_suffix/geometry_variant/

    Returns:
        List of geometry names sorted alphabetically
    """
    if run_config_suffix:
        ml_inputs_dir = os.path.join(data_root, "ml_inputs", set_name, run_config_suffix, geometry_variant)
    else:
        ml_inputs_dir = os.path.join(data_root, "ml_inputs", set_name, geometry_variant)
    if not os.path.exists(ml_inputs_dir):
        return []

    geometries = []

    # Iterate through all subdirectories in ml_inputs/<set_name>/<geometry_variant>/
    for geo_dir in glob.glob(os.path.join(ml_inputs_dir, "*")):
        if not os.path.isdir(geo_dir):
            continue

        geo_name = os.path.basename(geo_dir)

        # Check for both required CSV files
        geometric_features_path = os.path.join(geo_dir, "geometric_features.csv")
        junction_params_path = os.path.join(geo_dir, "junction_lumped_parameters.csv")

        if os.path.exists(geometric_features_path) and os.path.exists(junction_params_path):
            geometries.append(geo_name)

    return sorted(geometries)


def main():
    pipeline_defaults = get_pipeline_config()
    split_defaults = pipeline_defaults.split
    dp_defaults = pipeline_defaults.data_processing
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline for NN training")
    parser.add_argument("--set_name", required=True, help="Set name (e.g., VMR)")
    parser.add_argument(
        "--geometry_variant",
        default="all",
        choices=["bifurcations", "bifurcations_EL", "all"],
        help="Geometry variant (default: all - processes both bifurcations and bifurcations_EL)",
    )
    parser.add_argument(
        "--set_type",
        default="all",
        help="Cohort folder tier for jax_arrays/split_indices (default: all; not the ML train/test split)",
    )
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help=(
            "List of geometries (e.g., 0063_1001 ...). If not provided, auto-discovers geometries with both CSV files."
        ),
    )
    parser.add_argument(
        "--percent_train",
        type=float,
        default=split_defaults.data_processing_percent_train,
        help=(
            f"Fraction of points used for training "
            f"(default: {split_defaults.data_processing_percent_train} from config)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=split_defaults.seed,
        help=f"RNG seed for train/val split (default: {split_defaults.seed} from config)",
    )
    parser.add_argument("--data_root", default="data", help="Repo data root (default: data)")
    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        help="Run config suffix for path separation (default: %(default)s). "
        "ml_inputs/jax_arrays/split_indices use .../set_name/<suffix>/... "
        "(e.g. gen_loss is a full duplicate path tree).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose printing")
    parser.add_argument(
        "--flow_split_method",
        choices=sorted({FLOW_SPLIT_METHOD_MEAN_OVER_TIME, FLOW_SPLIT_METHOD_PEAK_INLET_FLOW}),
        default=None,
        help=(
            "How to compute flow_split from geometric simulation CSV. "
            f"Default: {dp_defaults.flow_split_method} from config "
            "(data_processing.flow_split_method)."
        ),
    )
    args = parser.parse_args()
    run_config_suffix = (args.run_config or DEFAULT_CLI_RUN_CONFIG).strip()
    dp_cfg = get_pipeline_config(set_name=args.set_name).data_processing
    flow_split_method = dp_cfg.flow_split_method if args.flow_split_method is None else args.flow_split_method

    # Determine which geometry variants to process
    if args.geometry_variant == "all":
        geometry_variants_to_process = ["bifurcations", "bifurcations_EL"]
    else:
        geometry_variants_to_process = [args.geometry_variant]

    # Process each geometry variant
    for geometry_variant in geometry_variants_to_process:
        print(f"\n{'=' * 80}")
        print(f"Processing geometry variant: {geometry_variant}")
        print(f"{'=' * 80}")

        # Get list of geometries to process
        if args.geometries:
            geometries = args.geometries
            print(f"Using {len(geometries)} specified geometries: {geometries}")
        else:
            print(f"Auto-discovering geometries with both CSV files for {args.set_name}/{geometry_variant}...")
            geometries = discover_geometries_with_csvs(
                args.set_name, geometry_variant, args.data_root, run_config_suffix
            )
            if len(geometries) == 0:
                _search_parts = [args.data_root, "ml_inputs", args.set_name]
                if run_config_suffix:
                    _search_parts.append(run_config_suffix)
                _search_parts.append(geometry_variant)
                _search_dir = os.path.join(*_search_parts)
                print("  No geometries found with both geometric_features.csv and junction_lumped_parameters.csv")
                print(f"  Searched in: {_search_dir}")
                continue
            print(f"  Found {len(geometries)} geometries: {geometries}")

        print(
            f"Running data processing pipeline for {args.set_name}/{geometry_variant} with {len(geometries)} geometries"
        )
        for geo in geometries:
            print(f"Processing geometry {geo}")

            # Check if both CSV files already exist
            if run_config_suffix:
                _ml_base = os.path.join(args.data_root, "ml_inputs", args.set_name, run_config_suffix)
                _zero_d_base = os.path.join(args.data_root, "zeroD", args.set_name, run_config_suffix)
            else:
                _ml_base = os.path.join(args.data_root, "ml_inputs", args.set_name)
                _zero_d_base = os.path.join(args.data_root, "zeroD", args.set_name)
            csv_path = os.path.join(_ml_base, geometry_variant, geo, "geometric_features.csv")
            targets_csv_path = os.path.join(_ml_base, geometry_variant, geo, "junction_lumped_parameters.csv")

            if geometry_variant == "bifurcations_EL":
                geometric_input_filename = "bifurcations_EL_geometric_input.json"
            else:
                geometric_input_filename = "bifurcations_geometric_input.json"

            geometric_input_path = os.path.join(_zero_d_base, geo, geometric_input_filename)
            geometric_results_path = geometric_input_path.replace("_geometric_input.json", "_geometric_results.csv")
            X, feature_names, junction_names, outlet_primary_names = load_junction_geometric_features(
                geometric_input_path,
                verbose=args.verbose,
                geometric_results_path=geometric_results_path,
                flow_split_method=flow_split_method,
            )
            print(f"Loaded {len(X)} junctions with {len(feature_names)} features")
            # Save the features to a csv file
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(feature_names)
                for row in X:
                    writer.writerow(row)
            print(f"Saved features to {csv_path}")

            # Also write a small meta CSV containing per-row primary outlet names
            meta_path = os.path.join(os.path.dirname(csv_path), "geometric_features_meta.csv")
            with open(meta_path, "w") as fmeta:
                writer = csv.writer(fmeta)
                writer.writerow(["junction_name", "primary_outlet_name"])
                for jname, pout in zip(junction_names, outlet_primary_names, strict=False):
                    writer.writerow([jname, pout])
            print(f"Saved geometric features meta to {meta_path}")

            # Load calibrated junction lumped parameters (targets) and save to a separate csv
            # Determine calibration output path based on geometry variant
            if geometry_variant == "bifurcations_EL":
                calib_output_filename = "bifurcations_EL_calibrated_output_BloodVesselJunction.json"
            else:
                calib_output_filename = "bifurcations_calibrated_output_BloodVesselJunction.json"

            calib_output_path = os.path.join(
                _zero_d_base,
                geo,
                calib_output_filename,
            )
            Y, target_names, y_junction_names, y_primary_outlet_names = load_junction_lumped_parameters(
                calib_output_path,
                require_two_outlets=True,
                junction_names=junction_names,
                verbose=args.verbose,
            )
            print(f"Loaded {len(Y)} junctions with {len(target_names)} target values")

            # --- Reorder outputs to match input row ordering, then verify ---
            # Build a lookup from (junction_name, primary_outlet_name) -> output row index
            if y_primary_outlet_names != outlet_primary_names or y_junction_names != junction_names:
                output_key_to_idx = {}
                for idx, (jn, pn) in enumerate(zip(y_junction_names, y_primary_outlet_names, strict=False)):
                    output_key_to_idx[(jn, pn)] = idx

                reorder = []
                for jn, pn in zip(junction_names, outlet_primary_names, strict=False):
                    oi = output_key_to_idx.get((jn, pn))
                    if oi is None:
                        raise ValueError(
                            f"Output row for junction={jn}, outlet={pn} not found in calibration output for {geo}."
                        )
                    reorder.append(oi)

                Y = Y[reorder]
                y_junction_names = [y_junction_names[i] for i in reorder]
                y_primary_outlet_names = [y_primary_outlet_names[i] for i in reorder]
                print(f"  Reordered {len(reorder)} output rows to match input ordering")

            # Verify alignment after reordering
            assert y_junction_names == junction_names, (
                f"Junction name mismatch after reordering for {geo}: {y_junction_names} != {junction_names}"
            )
            assert y_primary_outlet_names == outlet_primary_names, (
                f"Primary outlet name mismatch after reordering for {geo}: "
                f"{y_primary_outlet_names} != {outlet_primary_names}"
            )
            input_vid_col_idx = feature_names.index("outlet_vessel_id")
            output_vid_col_idx = target_names.index("outlet_vessel_id")
            for row_idx in range(len(X)):
                input_vid = int(X[row_idx, input_vid_col_idx])
                output_vid = int(Y[row_idx, output_vid_col_idx])
                if input_vid != output_vid:
                    raise ValueError(
                        f"outlet_vessel_id mismatch at row {row_idx} for {geo}: "
                        f"input has vessel_id={input_vid}, output has vessel_id={output_vid}. "
                        f"Junction={junction_names[row_idx]}, primary_outlet={outlet_primary_names[row_idx]}"
                    )
            print(
                f"  ✓ Consistency check passed: {len(X)} rows — "
                f"junction names, primary outlets, and outlet_vessel_ids all match"
            )

            os.makedirs(os.path.dirname(targets_csv_path), exist_ok=True)
            with open(targets_csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(target_names)
                for row in Y:
                    writer.writerow(row)
            print(f"Saved targets to {targets_csv_path}")

            # Save a meta file for the target CSV matching the input ordering
            targets_meta_path = os.path.join(os.path.dirname(targets_csv_path), "junction_lumped_parameters_meta.csv")
            with open(targets_meta_path, "w") as fmet:
                writer = csv.writer(fmet)
                writer.writerow(["junction_name", "primary_outlet_name"])
                for jname, pout in zip(y_junction_names, y_primary_outlet_names, strict=False):
                    writer.writerow([jname, pout])
            print(f"Saved targets meta to {targets_meta_path}")

            # ---- Vessel CSVs (one row per non-connector vessel) ----
            X_v, feat_names_v, vessel_ids, vessel_names = load_vessel_geometric_features(
                geometric_input_path, verbose=args.verbose
            )
            vessel_ids_t, _vessel_names_t, targets_v = load_vessel_targets_from_config(calib_output_path)
            if len(X_v) == 0:
                raise ValueError(f"No non-connector vessels found for {geo}")
            tidx = {vid: i for i, vid in enumerate(vessel_ids_t)}
            tgt_header = ["vessel_id", "vessel_name", "R_poiseuille", "stenosis_coefficient", "L"]
            Y_v_rows = []
            for i, vid in enumerate(vessel_ids):
                j = tidx.get(vid)
                if j is None:
                    raise ValueError(f"Vessel id {vid} from features not found in targets for {geo}")
                Y_v_rows.append([vid, vessel_names[i], targets_v[j, 0], targets_v[j, 1], targets_v[j, 2]])
            vessel_feat_path = os.path.join(os.path.dirname(csv_path), "vessel_geometric_features.csv")
            vessel_tgt_path = os.path.join(os.path.dirname(csv_path), "vessel_lumped_parameters.csv")
            with open(vessel_feat_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(feat_names_v)
                for row in X_v:
                    writer.writerow(row)
            with open(vessel_tgt_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(tgt_header)
                for row in Y_v_rows:
                    writer.writerow(row)
            print(f"Saved vessel features and targets to {vessel_feat_path}, {vessel_tgt_path}")

        # ---- Build concatenated data_dict for NN training (across all geometries) ----
        try:
            num_geos = len(geometries)
            if run_config_suffix:
                _ml_root_build = os.path.join(args.data_root, "ml_inputs", args.set_name, run_config_suffix)
                _set_name_build = ""
            else:
                _ml_root_build = os.path.join(args.data_root, "ml_inputs")
                _set_name_build = args.set_name
            data_dict = build_data_dict_from_csvs(
                set_name=_set_name_build,
                geometries=geometries,
                ml_inputs_root=_ml_root_build,
                geometry_variant=geometry_variant,
                cohort_set_name=args.set_name,
                run_config_suffix=run_config_suffix or None,
                set_type=args.set_type,
                data_root=args.data_root,
            )

            if run_config_suffix:
                jax_out_dir = os.path.join(
                    args.data_root,
                    "jax_arrays",
                    args.set_name,
                    run_config_suffix,
                    geometry_variant,
                    args.set_type,
                )
            else:
                jax_out_dir = os.path.join(args.data_root, "jax_arrays", args.set_name, geometry_variant, args.set_type)
            os.makedirs(jax_out_dir, exist_ok=True)
            junction_unclipped_name = jax_arrays_filename(
                num_geos, vessel=False, stenosis_clipping_enabled=False
            )
            vessel_unclipped_name = jax_arrays_filename(
                num_geos, vessel=True, stenosis_clipping_enabled=False
            )
            jax_unclipped_path = os.path.join(jax_out_dir, junction_unclipped_name)
            save_dict(data_dict, jax_unclipped_path)
            print(f"Wrote unclipped data_dict to {jax_unclipped_path}")

            # ---- Build and save vessel data_dict ----
            vessel_data_dict = build_data_dict_from_vessel_csvs(
                set_name=_set_name_build,
                geometries=geometries,
                ml_inputs_root=_ml_root_build,
                geometry_variant=geometry_variant,
                cohort_set_name=args.set_name,
                run_config_suffix=run_config_suffix or None,
                set_type=args.set_type,
                data_root=args.data_root,
            )
            vessel_jax_unclipped_path = os.path.join(jax_out_dir, vessel_unclipped_name)
            save_dict(vessel_data_dict, vessel_jax_unclipped_path)
            n_vessel = vessel_data_dict["input"].shape[0]
            print(f"Wrote unclipped vessel data_dict to {vessel_jax_unclipped_path} (n_vessel_rows={n_vessel})")

            # ---- Generate train/val split indices (by geometry: all rows from one geometry in same set) ----
            if "input" not in data_dict:
                raise ValueError("Expected 'input' in data_dict")
            num_pts = int(data_dict["input"].shape[0])
            _row_ranges, _, geometries_ordered = resolve_geometry_row_ranges_from_jax_dict(data_dict)
            train_geometries, val_geometries = generate_geometry_split(
                args.percent_train, args.seed, geometries_ordered
            )
            geometry_indices = build_geometry_index_map(data_dict, vessel_data_dict)
            split_dict = build_split_dict(
                train_geometries,
                val_geometries,
                geometry_indices,
                num_offsets=1,
                percent_train=float(args.percent_train),
                seed=int(args.seed),
                num_pts=num_pts,
            )

            if run_config_suffix:
                split_out_dir = os.path.join(
                    args.data_root,
                    "split_indices",
                    args.set_name,
                    run_config_suffix,
                    geometry_variant,
                    args.set_type,
                )
            else:
                split_out_dir = os.path.join(
                    args.data_root, "split_indices", args.set_name, geometry_variant, args.set_type
                )
            os.makedirs(split_out_dir, exist_ok=True)
            split_out_path = os.path.join(split_out_dir, f"train_val_ind_{args.set_name}_num_geos_{num_geos}")
            save_dict(split_dict, split_out_path)

            geometries_txt_path = split_out_path + "_geometries.txt"
            write_geometries_txt(geometries_txt_path, train_geometries, val_geometries)

            n_train = len(resolve_flat_indices(split_dict, "junction", "train"))
            n_val = len(resolve_flat_indices(split_dict, "junction", "val"))
            print(f"Wrote split indices to {split_out_path} (n_train={n_train}, n_val={n_val})")
            print(f"Wrote geometry set assignment to {geometries_txt_path}")
            print("Train geometries:", train_geometries)
            print("Validation geometries:", val_geometries)

            stenosis_cfg = dp_cfg.stenosis_clipping
            if stenosis_cfg.enabled:
                junction_clipped_name = jax_arrays_filename(
                    num_geos, vessel=False, stenosis_clipping_enabled=True
                )
                vessel_clipped_name = jax_arrays_filename(
                    num_geos, vessel=True, stenosis_clipping_enabled=True
                )
                jax_clipped_path = os.path.join(jax_out_dir, junction_clipped_name)
                vessel_jax_clipped_path = os.path.join(jax_out_dir, vessel_clipped_name)
                metadata = write_clipped_jax_pickles(
                    junction_unclipped=data_dict,
                    vessel_unclipped=vessel_data_dict,
                    split_dict=split_dict,
                    clip_generation_number=stenosis_cfg.clip_generation_number,
                    junction_out_path=jax_clipped_path,
                    vessel_out_path=vessel_jax_clipped_path,
                    split_path=split_out_path,
                )
                junction_clipped = load_dict(jax_clipped_path)
                vessel_clipped = load_dict(vessel_jax_clipped_path)
                export_clipped_lumped_parameter_csvs(
                    junction_clipped=junction_clipped,
                    vessel_clipped=vessel_clipped,
                    metadata=metadata,
                    data_root=args.data_root,
                    set_name=args.set_name,
                    run_config_suffix=run_config_suffix or None,
                    geometry_variant=geometry_variant,
                    label_variant="default",
                )
            else:
                print(
                    "  stenosis_clipping disabled: training should load _unclipped jax pickles "
                    f"({junction_unclipped_name}, {vessel_unclipped_name})"
                )
        except Exception as e:
            print(f"  Failed to build data_dict / jax_arrays / split for {geometry_variant}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
