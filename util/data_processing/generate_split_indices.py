#!/usr/bin/env python3
"""
Generate train/validation split indices for the JAX-array dataset.

This writes the dictionary expected by `util/neural_network/launch_training.py`:
  - train_ind: 1D array/list of training indices
  - val_ind:   1D array/list of validation indices
  - num_offsets: int

For our current data format (one row per junction instance), we set:
  - num_offsets = 1

When geometry_row_ranges is provided, all rows from a given geometry are kept
in the same set (either all in training or all in validation). In that case
percent_train is applied to the number of geometries, not points.
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# Allow running as a script (python util/data_processing/generate_split_indices.py ...)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.tools.basic import load_dict, save_dict


def get_geometry_row_ranges(
    ml_inputs_root: str,
    set_name: str,
    geometry_variant: str,
    geometries: Optional[List[str]] = None,
    run_config_suffix: Optional[str] = None,
) -> Tuple[List[Tuple[int, int]], int, List[str]]:
    """
    Return (row_ranges, total_rows, geometries) for each geometry in order.
    row_ranges[i] = (start, end) so geometry i has row indices [start, end).
    Geometries are in sorted order if discovered from disk.
    When run_config_suffix is set, ml_inputs path is .../set_name/run_config_suffix/geometry_variant/...
    """
    if run_config_suffix:
        ml_inputs_dir = os.path.join(ml_inputs_root, set_name, run_config_suffix, geometry_variant)
        geom_csv_template = os.path.join(ml_inputs_root, set_name, run_config_suffix, geometry_variant, "%s", "geometric_features.csv")
    else:
        ml_inputs_dir = os.path.join(ml_inputs_root, set_name, geometry_variant)
        geom_csv_template = os.path.join(ml_inputs_root, set_name, geometry_variant, "%s", "geometric_features.csv")
    if geometries is None:
        if not os.path.exists(ml_inputs_dir):
            raise FileNotFoundError(f"ML inputs dir not found: {ml_inputs_dir}")
        geometries = []
        for geo_dir in glob.glob(os.path.join(ml_inputs_dir, "*")):
            if not os.path.isdir(geo_dir):
                continue
            geo_name = os.path.basename(geo_dir)
            gf = os.path.join(geo_dir, "geometric_features.csv")
            if os.path.exists(gf):
                geometries.append(geo_name)
        geometries = sorted(geometries)

    row_ranges = []
    start = 0
    for geo in geometries:
        geom_csv = geom_csv_template % geo
        if not os.path.exists(geom_csv):
            raise FileNotFoundError(f"Expected CSV: {geom_csv}")
        with open(geom_csv) as f:
            n_rows = sum(1 for _ in f) - 1  # exclude header
        if n_rows < 0:
            n_rows = 0
        row_ranges.append((start, start + n_rows))
        start += n_rows
    return row_ranges, start, geometries


def generate_split_indices(
    num_pts: int,
    percent_train: float,
    seed: int = 0,
    geometry_row_ranges: Optional[List[Tuple[int, int]]] = None,
):
    if not (0.0 < percent_train <= 1.0):
        raise ValueError(f"percent_train must be in (0, 1], got {percent_train}")
    if num_pts <= 0:
        raise ValueError(f"num_pts must be > 0, got {num_pts}")

    if geometry_row_ranges is not None:
        total_from_ranges = sum(end - start for start, end in geometry_row_ranges)
        if total_from_ranges != num_pts:
            raise ValueError(
                f"Geometry row ranges sum to {total_from_ranges} but num_pts={num_pts}"
            )
        num_geos = len(geometry_row_ranges)
        rng = np.random.default_rng(seed)
        geo_order = rng.permutation(num_geos)
        n_train_geos = max(1, int(percent_train * num_geos))
        train_geo_idx_list = sorted(geo_order[:n_train_geos].tolist())
        val_geo_idx_list = sorted(geo_order[n_train_geos:].tolist())
        train_geo_indices = set(train_geo_idx_list)
        train_indices = []
        val_indices = []
        for g, (s, e) in enumerate(geometry_row_ranges):
            if g in train_geo_indices:
                train_indices.extend(range(s, e))
            else:
                val_indices.extend(range(s, e))
        return (
            np.array(train_indices, dtype=int),
            np.array(val_indices, dtype=int),
            train_geo_idx_list,
            val_geo_idx_list,
        )

    # Random split (legacy: point-level)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_pts)
    n_train = int(percent_train * num_pts)
    if n_train <= 0:
        raise ValueError(f"Invalid split: num_pts={num_pts}, percent_train={percent_train} -> n_train={n_train}")
    if n_train > num_pts:
        raise ValueError(f"Invalid split: num_pts={num_pts}, percent_train={percent_train} -> n_train={n_train}")
    train_indices = indices[:n_train]
    val_indices = indices[n_train:] if n_train < num_pts else np.array([], dtype=int)
    return train_indices, val_indices, None, None



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

    ml_inputs_root = os.path.join(args.data_root, "ml_inputs")
    row_ranges, total_rows, geometries = get_geometry_row_ranges(
        ml_inputs_root, args.set_name, args.geometry_variant
    )
    if len(row_ranges) != args.num_geos:
        raise ValueError(
            f"Geometry count mismatch: discovered {len(row_ranges)} geometries "
            f"but jax arrays were built with num_geos={args.num_geos}"
        )
    if total_rows != num_pts:
        raise ValueError(
            f"Row count mismatch: geometry row ranges sum to {total_rows} "
            f"but jax array has {num_pts} rows"
        )

    train_ind, val_ind, train_geo_idx, val_geo_idx = generate_split_indices(
        num_pts=num_pts,
        percent_train=args.percent_train,
        seed=args.seed,
        geometry_row_ranges=row_ranges,
    )

    train_geometries = [geometries[i] for i in train_geo_idx]
    val_geometries = [geometries[i] for i in val_geo_idx]

    split_dict = {
        "train_ind": np.asarray(train_ind, dtype=int),
        "val_ind": np.asarray(val_ind, dtype=int),
        "num_offsets": 1,
        "percent_train": float(args.percent_train),
        "seed": int(args.seed),
        "num_pts": int(num_pts),
        "split_by_geometry": True,
    }

    out_dir = os.path.join(args.data_root, "split_indices", args.set_name, args.geometry_variant, args.set_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"train_val_ind_{args.set_name}_num_geos_{args.num_geos}")
    save_dict(split_dict, out_path)

    geometries_txt_path = out_path + "_geometries.txt"
    with open(geometries_txt_path, "w") as f:
        f.write("Train geometries:\n")
        for g in train_geometries:
            f.write(f"  {g}\n")
        f.write("Validation geometries:\n")
        for g in val_geometries:
            f.write(f"  {g}\n")

    print(f"Wrote split indices to {out_path}")
    print(f"  num_pts={num_pts}  n_train={len(train_ind)}  n_val={len(val_ind)}  split_by_geometry=True  num_offsets={split_dict['num_offsets']}")
    print(f"Wrote geometry set assignment to {geometries_txt_path}")
    print("Train geometries:")
    for g in train_geometries:
        print(f"  {g}")
    print("Validation geometries:")
    for g in val_geometries:
        print(f"  {g}")


if __name__ == "__main__":
    main()


