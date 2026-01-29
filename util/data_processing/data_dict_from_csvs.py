#!/usr/bin/env python3
"""
Build `self.data_dict` compatible with `util/neural_network` from the ML CSVs.

Inputs:
  - `data/ml_inputs/<set_name>/<geo>/geometric_features.csv`
  - `data/ml_inputs/<set_name>/<geo>/junction_lumped_parameters.csv`

Outputs:
  - A Python dict containing JAX arrays with keys expected by `NeuralNet`:
      - input_o1, input_o2
      - output_o1_rri, output_o2_rri (and optionally _ri/_rr)
      - scaling_factors

This is intentionally "strict": missing columns raise ValueError.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    import jax.numpy as jnp  # type: ignore
except (ImportError, ModuleNotFoundError):
    jnp = None


def _read_csv_matrix(csv_path: str) -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        rows = []
        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])

    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")

    X = np.asarray(rows, dtype=float)
    if X.shape[1] != len(header):
        raise ValueError(
            f"CSV column mismatch in {csv_path}: header has {len(header)} columns but data has {X.shape[1]}"
        )
    return header, X


def build_data_dict_from_csvs(
    set_name: str,
    geometries: List[str],
    output_type: str = "rri",
    ml_inputs_root: str = "data/ml_inputs",
    require_same_rows: bool = True,
) -> Dict[str, "np.ndarray"]:
    """
    Concatenate multiple geometries' CSVs and build a `data_dict`.

    Each junction contributes 2 rows (swapped outlet order), so we simply concatenate
    all rows together into single `input` and `output` arrays.

    For output_type "rri": output columns are [R_poiseuille, stenosis_coefficient, L]
    per outlet, flattened as [R_outlet0, stenosis_outlet0, L_outlet0, R_outlet1, ...].
    """
    if output_type not in {"rri", "ri", "rr"}:
        raise ValueError(f"Unsupported output_type: {output_type}")

    all_inputs: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []

    def require_cols(col_to_idx: Dict[str, int], needed: List[str], ctx: str) -> List[int]:
        missing = [c for c in needed if c not in col_to_idx]
        if missing:
            raise ValueError(f"Missing required columns in {ctx}: {missing}")
        return [col_to_idx[c] for c in needed]

    for geo in geometries:
        geom_csv = os.path.join(ml_inputs_root, set_name, geo, "geometric_features.csv")
        out_csv = os.path.join(ml_inputs_root, set_name, geo, "junction_lumped_parameters.csv")

        geom_header, geom_X = _read_csv_matrix(geom_csv)
        out_header, out_Y = _read_csv_matrix(out_csv)

        if require_same_rows and geom_X.shape[0] != out_Y.shape[0]:
            raise ValueError(
                f"Row mismatch for geo {geo}: geometric_features has {geom_X.shape[0]} rows, "
                f"junction_lumped_parameters has {out_Y.shape[0]} rows"
            )

        # Use all columns from geometric_features (already includes both outlets in correct order)
        all_inputs.append(geom_X)
        # Use all columns from junction_lumped_parameters (already includes both outlets)
        all_outputs.append(out_Y)

    input_array = np.vstack(all_inputs)
    output_array = np.vstack(all_outputs)

    n = input_array.shape[0]
    scaling_factors = np.ones((n, 1), dtype=float)

    if jnp is not None:
        data_dict = {
            "input": jnp.asarray(input_array),
            f"output_{output_type}": jnp.asarray(output_array),
            "scaling_factors": jnp.asarray(scaling_factors),
        }
    else:
        data_dict = {
            "input": input_array,
            f"output_{output_type}": output_array,
            "scaling_factors": scaling_factors,
        }
    return data_dict


__all__ = [
    "build_data_dict_from_csvs",
]


