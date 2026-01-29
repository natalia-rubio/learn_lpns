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

# Allow running as a script (python util/data_processing/run_data_processing.py ...)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.data_processing.inputs_from_0d_config import load_junction_geometric_features
from util.data_processing.outputs_from_config import load_junction_lumped_parameters
from util.data_processing.data_dict_from_csvs import build_data_dict_from_csvs
from util.data_processing.generate_split_indices import generate_split_indices
from util.tools.basic import save_dict

def main():
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline for NN training")
    parser.add_argument("--set-name", required=True, help="Set name (e.g., VMR)")
    parser.add_argument("--set-type", default="test", help="Dataset split name used by NN (default: test)")
    parser.add_argument("--geometries", nargs="+", required=True, help="List of geometries (e.g., 0063_1001 ...)")
    parser.add_argument("--output-type", default="rri", choices=["rri", "ri", "rr"], help="Target type (default: rri)")
    parser.add_argument("--percent-train", type=float, default=0.8, help="Fraction of points used for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for train/val split (default: 0)")
    parser.add_argument("--data-root", default="data", help="Repo data root (default: data)")
    parser.add_argument("--verbose", action="store_true", help="Verbose printing")
    args = parser.parse_args()

    print(f"Running data processing pipeline for {args.set_name} with geometries {args.geometries}")
    for geo in args.geometries:
        print(f"Processing geometry {geo}")

        geometric_input_path = os.path.join(
            args.data_root, "zeroD", args.set_name, geo, "bifurcations_geometric_input.json"
        )
        X, feature_names, junction_names = load_junction_geometric_features(
            geometric_input_path, verbose=args.verbose
        )
        print(f"Loaded {len(X)} junctions with {len(feature_names)} features")
        # Save the features to a csv file
        csv_path = os.path.join(args.data_root, "ml_inputs", args.set_name, geo, "geometric_features.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(feature_names)
            for row in X:
                writer.writerow(row)
        print(f"Saved features to {csv_path}")

        # Load calibrated junction lumped parameters (targets) and save to a separate csv
        calib_output_path = os.path.join(
            args.data_root,
            "zeroD",
            args.set_name,
            geo,
            "bifurcations_calibrated_output_BloodVesselJunction.json",
        )
        Y, target_names, y_junction_names = load_junction_lumped_parameters(
            calib_output_path,
            require_two_outlets=True,
            junction_names=junction_names,
            verbose=args.verbose,
        )
        if y_junction_names != junction_names:
            raise ValueError("Internal error: junction ordering mismatch between inputs and outputs.")
        print(f"Loaded {len(Y)} junctions with {len(target_names)} target values")

        targets_csv_path = os.path.join(args.data_root, "ml_inputs", args.set_name, geo, "junction_lumped_parameters.csv")
        os.makedirs(os.path.dirname(targets_csv_path), exist_ok=True)
        with open(targets_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(target_names)
            for row in Y:
                writer.writerow(row)
        print(f"Saved targets to {targets_csv_path}")

    # ---- Build concatenated data_dict for NN training (across all geometries) ----
    num_geos = len(args.geometries)
    data_dict = build_data_dict_from_csvs(
        set_name=args.set_name,
        geometries=args.geometries,
        output_type=args.output_type,
        ml_inputs_root=os.path.join(args.data_root, "ml_inputs"),
    )

    jax_out_dir = os.path.join(args.data_root, "jax_arrays", args.set_name, args.set_type)
    os.makedirs(jax_out_dir, exist_ok=True)
    jax_out_path = os.path.join(jax_out_dir, f"jax_arrays_num_geos_{num_geos}.pkl")
    save_dict(data_dict, jax_out_path)
    print(f"Wrote data_dict to {jax_out_path}")

    # ---- Generate train/val split indices ----
    if "input" not in data_dict:
        raise ValueError("Expected 'input' in data_dict")
    num_pts = int(getattr(data_dict["input"], "shape")[0])
    train_ind, val_ind = generate_split_indices(num_pts=num_pts, percent_train=args.percent_train, seed=args.seed)

    split_dict = {
        "train_ind": train_ind,
        "val_ind": val_ind,
        "num_offsets": 1,
    }

    split_out_dir = os.path.join(args.data_root, "split_indices", args.set_name, args.set_type)
    os.makedirs(split_out_dir, exist_ok=True)
    split_out_path = os.path.join(split_out_dir, f"train_val_ind_{args.set_name}_num_geos_{num_geos}")
    save_dict(split_dict, split_out_path)
    print(f"Wrote split indices to {split_out_path} (n_train={len(train_ind)}, n_val={len(val_ind)})")

if __name__ == "__main__":
    main()
        