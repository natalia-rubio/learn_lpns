"""
For a set name and list of geometries, run the full data processing pipeline:

1) Extract geometric features from 0D config -> `geometric_features.csv`
2) Extract calibrated junction lumped parameters -> `junction_lumped_parameters.csv`
3) Build concatenated `data_dict` for JAX NN training -> `data/jax_arrays/.../jax_arrays_num_geos_<N>.pkl`
4) Generate train/val split indices -> `data/split_indices/.../train_val_ind_<set_name>_num_geos_<N>`
"""

import argparse
import csv
import os
import sys
import glob

import numpy as np

# Allow running as a script (python util/data_processing/run_data_processing.py ...)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.data_processing.inputs_from_0d_config import (
    load_junction_geometric_features,
    compute_junction_flow_splits,
    load_vessel_geometric_features,
    load_vessel_targets_from_config,
)
from util.data_processing.outputs_from_config import load_junction_lumped_parameters
from util.data_processing.data_dict_from_csvs import build_data_dict_from_csvs, build_data_dict_from_vessel_csvs
from util.data_processing.generate_split_indices import (
    generate_split_indices,
    get_geometry_row_ranges,
)
from util.tools.basic import save_dict

def discover_geometries_with_csvs(set_name, geometry_variant="bifurcations", data_root="data"):
    """
    Discover all geometries that have both geometric_features.csv and junction_lumped_parameters.csv.
    
    Args:
        set_name: Set name (e.g., VMR)
        geometry_variant: Geometry variant name (e.g., "bifurcations" or "bifurcations_EL")
        data_root: Repo data root (default: data)
        
    Returns:
        List of geometry names sorted alphabetically
    """
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
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline for NN training")
    parser.add_argument("--set-name", required=True, help="Set name (e.g., VMR)")
    parser.add_argument("--geometry-variant", default="all", 
                       choices=["bifurcations", "bifurcations_EL", "all"],
                       help="Geometry variant (default: all - processes both bifurcations and bifurcations_EL)")
    parser.add_argument("--set-type", default="test", help="Dataset split name used by NN (default: test)")
    parser.add_argument("--geometries", nargs="+", default=None, 
                       help="List of geometries (e.g., 0063_1001 ...). If not provided, auto-discovers geometries with both CSV files.")
    parser.add_argument("--output-type", default="rri", choices=["rri", "ri", "rr"], help="Target type (default: rri)")
    parser.add_argument("--percent-train", type=float, default=0.8, help="Fraction of points used for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for train/val split (default: 0)")
    parser.add_argument("--data-root", default="data", help="Repo data root (default: data)")
    parser.add_argument("--normalize", action="store_true", help="Apply z-normalization to inputs/outputs (saves to separate _normalized pkl)")
    parser.add_argument("--verbose", action="store_true", help="Verbose printing")
    args = parser.parse_args()

    # Determine which geometry variants to process
    if args.geometry_variant == "all":
        geometry_variants_to_process = ["bifurcations", "bifurcations_EL"]
    else:
        geometry_variants_to_process = [args.geometry_variant]
    
    # Process each geometry variant
    for geometry_variant in geometry_variants_to_process:
        print(f"\n{'='*80}")
        print(f"Processing geometry variant: {geometry_variant}")
        print(f"{'='*80}")
        
        # Get list of geometries to process
        if args.geometries:
            geometries = args.geometries
            print(f"Using {len(geometries)} specified geometries: {geometries}")
        else:
            print(f"Auto-discovering geometries with both CSV files for {args.set_name}/{geometry_variant}...")
            geometries = discover_geometries_with_csvs(args.set_name, geometry_variant, args.data_root)
            if len(geometries) == 0:
                print(f"  No geometries found with both geometric_features.csv and junction_lumped_parameters.csv")
                print(f"  Searched in: {os.path.join(args.data_root, 'ml_inputs', args.set_name, geometry_variant)}")
                continue
            print(f"  Found {len(geometries)} geometries: {geometries}")

        print(f"Running data processing pipeline for {args.set_name}/{geometry_variant} with {len(geometries)} geometries")
        for geo in geometries:
            print(f"Processing geometry {geo}")

        # Check if both CSV files already exist
            csv_path = os.path.join(args.data_root, "ml_inputs", args.set_name, geometry_variant, geo, "geometric_features.csv")
            targets_csv_path = os.path.join(args.data_root, "ml_inputs", args.set_name, geometry_variant, geo, "junction_lumped_parameters.csv")
            
            # if os.path.exists(csv_path) and os.path.exists(targets_csv_path):
            #     print(f"  Both CSV files already exist, skipping extraction for {geo}")
            #     continue

            # Determine geometric input path based on geometry variant
            if geometry_variant == "bifurcations_EL":
                geometric_input_filename = "bifurcations_EL_geometric_input.json"
            else:
                geometric_input_filename = "bifurcations_geometric_input.json"
            
            geometric_input_path = os.path.join(
                args.data_root, "zeroD", args.set_name, geo, geometric_input_filename
            )
            X, feature_names, junction_names, outlet_primary_names = load_junction_geometric_features(
                geometric_input_path, verbose=args.verbose
            )
            # Add flow split from base geometric simulation if results CSV exists
            geometric_results_path = geometric_input_path.replace("_geometric_input.json", "_geometric_results.csv")
            if os.path.exists(geometric_results_path):
                flow_splits = compute_junction_flow_splits(geometric_input_path, geometric_results_path)
                flow_split_col = []
                for i, jname in enumerate(junction_names):
                    (out0_name, out1_name), (fs0, fs1) = flow_splits.get(
                        jname, (("", ""), (float("nan"), float("nan")))
                    )
                    primary = outlet_primary_names[i]
                    val = fs0 if primary == out0_name else (fs1 if primary == out1_name else float("nan"))
                    flow_split_col.append(val)
                X = np.column_stack([X, flow_split_col])
                feature_names = feature_names + ["flow_split"]
                # flow_split_inv = 1 / flow_split (NaN for zero or invalid)
                flow_split_arr = np.asarray(flow_split_col, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    flow_split_inv = np.where(
                        np.isfinite(flow_split_arr) & (flow_split_arr > 0),
                        100.0 / flow_split_arr,  # flow_split is in %, so inv is 100/%
                        np.nan,
                    )
                X = np.column_stack([X, flow_split_inv])
                feature_names = feature_names + ["flow_split_inv"]
                if args.verbose:
                    print(f"  Added flow_split, flow_split_inv from {geometric_results_path}")
            else:
                if args.verbose:
                    print(f"  Geometric results not found: {geometric_results_path}; skipping flow split")
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
                for jname, pout in zip(junction_names, outlet_primary_names):
                    writer.writerow([jname, pout])
            print(f"Saved geometric features meta to {meta_path}")

            # Load calibrated junction lumped parameters (targets) and save to a separate csv
            # Determine calibration output path based on geometry variant
            if geometry_variant == "bifurcations_EL":
                calib_output_filename = "bifurcations_EL_calibrated_output_BloodVesselJunction.json"
            else:
                calib_output_filename = "bifurcations_calibrated_output_BloodVesselJunction.json"
            
            calib_output_path = os.path.join(
                args.data_root,
                "zeroD",
                args.set_name,
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
                for idx, (jn, pn) in enumerate(zip(y_junction_names, y_primary_outlet_names)):
                    output_key_to_idx[(jn, pn)] = idx

                reorder = []
                for jn, pn in zip(junction_names, outlet_primary_names):
                    oi = output_key_to_idx.get((jn, pn))
                    if oi is None:
                        raise ValueError(
                            f"Output row for junction={jn}, outlet={pn} not found in "
                            f"calibration output for {geo}."
                        )
                    reorder.append(oi)

                Y = Y[reorder]
                y_junction_names = [y_junction_names[i] for i in reorder]
                y_primary_outlet_names = [y_primary_outlet_names[i] for i in reorder]
                print(f"  Reordered {len(reorder)} output rows to match input ordering")

            # Verify alignment after reordering
            assert y_junction_names == junction_names, (
                f"Junction name mismatch after reordering for {geo}: "
                f"{y_junction_names} != {junction_names}"
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
            print(f"  ✓ Consistency check passed: {len(X)} rows — junction names, primary outlets, and outlet_vessel_ids all match")

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
                for jname, pout in zip(y_junction_names, y_primary_outlet_names):
                    writer.writerow([jname, pout])
            print(f"Saved targets meta to {targets_meta_path}")

            # ---- Vessel CSVs (one row per non-connector vessel) ----
            try:
                X_v, feat_names_v, vessel_ids, vessel_names = load_vessel_geometric_features(
                    geometric_input_path, verbose=args.verbose
                )
                vessel_ids_t, vessel_names_t, targets_v = load_vessel_targets_from_config(calib_output_path)
            except Exception as e:
                if args.verbose:
                    print(f"  Skipping vessel CSVs for {geo}: {e}")
                X_v, feat_names_v, vessel_ids, vessel_names = None, None, None, None
                vessel_ids_t, vessel_names_t, targets_v = None, None, None
            if X_v is not None and len(X_v) > 0:
                # Align targets to feature order by vessel_id
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
            data_dict = build_data_dict_from_csvs(
                set_name=args.set_name,
                geometries=geometries,
                output_type=args.output_type,
                ml_inputs_root=os.path.join(args.data_root, "ml_inputs"),
                geometry_variant=geometry_variant,
                normalize=args.normalize,
            )

            norm_suffix = "_normalized" if args.normalize else ""
            jax_out_dir = os.path.join(args.data_root, "jax_arrays", args.set_name, geometry_variant, args.set_type)
            os.makedirs(jax_out_dir, exist_ok=True)
            jax_out_path = os.path.join(jax_out_dir, f"jax_arrays_num_geos_{num_geos}{norm_suffix}.pkl")
            save_dict(data_dict, jax_out_path)
            print(f"Wrote data_dict to {jax_out_path}")

            # ---- Build and save vessel data_dict ----
            vessel_data_dict = build_data_dict_from_vessel_csvs(
                set_name=args.set_name,
                geometries=geometries,
                ml_inputs_root=os.path.join(args.data_root, "ml_inputs"),
                geometry_variant=geometry_variant,
                normalize=args.normalize,
            )
            vessel_jax_path = os.path.join(jax_out_dir, f"jax_arrays_vessel_num_geos_{num_geos}{norm_suffix}.pkl")
            save_dict(vessel_data_dict, vessel_jax_path)
            n_vessel = vessel_data_dict["input"].shape[0]
            print(f"Wrote vessel data_dict to {vessel_jax_path} (n_vessel_rows={n_vessel})")

            # ---- Generate train/val split indices (by geometry: all rows from one geometry in same set) ----
            if "input" not in data_dict:
                raise ValueError("Expected 'input' in data_dict")
            num_pts = int(getattr(data_dict["input"], "shape")[0])
            ml_inputs_root = os.path.join(args.data_root, "ml_inputs")
            row_ranges, _, geometries_ordered = get_geometry_row_ranges(
                ml_inputs_root, args.set_name, geometry_variant, geometries=geometries
            )
            train_ind, val_ind, train_geo_idx, val_geo_idx = generate_split_indices(
                num_pts=num_pts,
                percent_train=args.percent_train,
                seed=args.seed,
                geometry_row_ranges=row_ranges,
            )

            train_geometries = [geometries_ordered[i] for i in train_geo_idx]
            val_geometries = [geometries_ordered[i] for i in val_geo_idx]

            split_dict = {
                "train_ind": train_ind,
                "val_ind": val_ind,
                "num_offsets": 1,
            }

            split_out_dir = os.path.join(args.data_root, "split_indices", args.set_name, geometry_variant, args.set_type)
            os.makedirs(split_out_dir, exist_ok=True)
            split_out_path = os.path.join(split_out_dir, f"train_val_ind_{args.set_name}_num_geos_{num_geos}")
            save_dict(split_dict, split_out_path)

            geometries_txt_path = split_out_path + "_geometries.txt"
            with open(geometries_txt_path, "w") as f:
                f.write("Train geometries:\n")
                for g in train_geometries:
                    f.write(f"  {g}\n")
                f.write("Validation geometries:\n")
                for g in val_geometries:
                    f.write(f"  {g}\n")

            print(f"Wrote split indices to {split_out_path} (n_train={len(train_ind)}, n_val={len(val_ind)})")
            print(f"Wrote geometry set assignment to {geometries_txt_path}")
            print("Train geometries:", train_geometries)
            print("Validation geometries:", val_geometries)
        except Exception as e:
            print(f"  Skipping data_dict / jax_arrays / split for {geometry_variant}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
        