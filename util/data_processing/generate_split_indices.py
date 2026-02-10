#!/usr/bin/env python3
"""
Generate train/validation split indices for the JAX-array dataset.

This writes the dictionary expected by `util/neural_network/launch_training.py`:
  - train_ind: 1D array/list of training indices
  - val_ind:   1D array/list of validation indices
  - num_offsets: int

For our current data format (one row per junction instance), we set:
  - num_offsets = 1
"""

import argparse
import os
import sys

import numpy as np

# Allow running as a script (python util/data_processing/generate_split_indices.py ...)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.tools.basic import load_dict, save_dict


def generate_split_indices(num_pts: int, percent_train: float, seed: int = 0):
    if not (0.0 < percent_train <= 1.0):
        raise ValueError(f"percent_train must be in (0, 1], got {percent_train}")
    if num_pts <= 0:
        raise ValueError(f"num_pts must be > 0, got {num_pts}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_pts)
    n_train = int(percent_train * num_pts)
    if n_train <= 0:
        raise ValueError(f"Invalid split: num_pts={num_pts}, percent_train={percent_train} -> n_train={n_train}")
    
    # Allow 100% train (n_train == num_pts)
    if n_train > num_pts:
        raise ValueError(f"Invalid split: num_pts={num_pts}, percent_train={percent_train} -> n_train={n_train}")

    train_indices = indices[:n_train]
    val_indices = indices[n_train:] if n_train < num_pts else np.array([], dtype=int)

    return train_indices, val_indices


def main():
    parser = argparse.ArgumentParser(description="Generate train/val split indices for NN training.")
    parser.add_argument("--set-name", required=True, help="e.g. VMR")
    parser.add_argument("--geometry-variant", default="bifurcations", 
                       choices=["bifurcations", "bifurcations_EL"],
                       help="Geometry variant (default: bifurcations)")
    parser.add_argument("--set-type", default="test", help="e.g. test")
    parser.add_argument("--num-geos", type=int, required=True, help="Number of geometries used to build the jax arrays")
    parser.add_argument("--percent-train", type=float, required=True, help="Fraction of points to use for training (0,1)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible split")
    parser.add_argument("--data-root", default="data", help="Repo data root (default: data)")
    args = parser.parse_args()

    jax_arrays_path = os.path.join(
        args.data_root,
        "jax_arrays",
        args.set_name,
        args.geometry_variant,
        args.set_type,
        f"jax_arrays_num_geos_{args.num_geos}.pkl",
    )
    data_dict = load_dict(jax_arrays_path)

    if "input" not in data_dict:
        raise ValueError(f"Expected 'input' in data_dict at {jax_arrays_path}")
    num_pts = int(np.asarray(data_dict["input"]).shape[0])

    train_ind, val_ind = generate_split_indices(num_pts=num_pts, percent_train=args.percent_train, seed=args.seed)

    split_dict = {
        "train_ind": np.asarray(train_ind, dtype=int),
        "val_ind": np.asarray(val_ind, dtype=int),
        "num_offsets": 1,
        "percent_train": float(args.percent_train),
        "seed": int(args.seed),
        "num_pts": int(num_pts),
    }

    out_dir = os.path.join(args.data_root, "split_indices", args.set_name, args.geometry_variant, args.set_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"train_val_ind_{args.set_name}_num_geos_{args.num_geos}")
    save_dict(split_dict, out_path)

    print(f"Wrote split indices to {out_path}")
    print(f"  num_pts={num_pts}  n_train={len(train_ind)}  n_val={len(val_ind)}  num_offsets={split_dict['num_offsets']}")


if __name__ == "__main__":
    main()


