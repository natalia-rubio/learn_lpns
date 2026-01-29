#!/usr/bin/env python3
"""
CLI to build and save `data_dict` files for util/neural_network from the ML CSVs.
"""

import argparse
import os
import sys

# Allow running as a script (python util/data_processing/build_jax_arrays.py ...)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.data_processing.data_dict_from_csvs import build_data_dict_from_csvs
from util.tools.basic import save_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-name", required=True)
    parser.add_argument("--set-type", default="test")
    parser.add_argument("--geometries", nargs="+", required=True)
    # parser.add_argument("--output-type", default="rri", choices=["rri", "ri", "rr"])
    # parser.add_argument("--ml-inputs-root", default="data/ml_inputs")
    # parser.add_argument("--out-root", default="data/jax_arrays")
    output_type = "rri"
    ml_inputs_root = "data/ml_inputs"
    out_root = "data/jax_arrays"

    args = parser.parse_args()

    data_dict = build_data_dict_from_csvs(
        set_name=args.set_name,
        geometries=args.geometries,
        output_type=output_type,
        ml_inputs_root=ml_inputs_root,
    )

    out_dir = os.path.join(out_root, args.set_name, args.set_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"jax_arrays_num_geos_{len(args.geometries)}.pkl")
    save_dict(data_dict, out_path)
    print(f"Wrote data_dict to {out_path}")


if __name__ == "__main__":
    main()

