#!/usr/bin/env python3
"""
Utilities for extracting ML-ready outputs from calibration output files.

Current focus: build a junction-level target matrix from the calibrated
`junction_values` written into svZeroD calibration output JSONs, e.g.
`bifurcations_calibrated_output_BloodVesselJunction.json`.

Each row corresponds to one junction instance (typically after
bifurcation_splitting, i.e. each with exactly two outlets).
Each column is a scalar "lumped parameter" derived from `junction_values`.
"""

import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def load_junction_lumped_parameters(
    calibration_output_path: str,
    require_two_outlets: bool = True,
    junction_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract a junction-level target matrix from a calibration output JSON.

    This expects each junction to contain:
      - `junction_name`
      - `outlet_vessels` (list of vessel indices)
      - `junction_values` (dict of parameter_name -> list[float] per outlet)
      - `geometric_params.outlet_path_lengths` (dict vessel_name -> float), used
        to order outlets in the same way as `inputs_from_0d_config.py`.

    Args:
        calibration_output_path: path to a calibrated output JSON.
        require_two_outlets: if True, only use junctions with exactly two outlets.
        junction_names: if provided, return rows in exactly this order (and
            raise if any are missing).
        verbose: print debug information.

    Returns:
        Y: (n_junctions, n_targets) float array
        target_names: list of column names
        out_junction_names: list of junction names corresponding to Y rows
    """
    with open(calibration_output_path, "r") as f:
        cfg = json.load(f)

    junctions = cfg.get("junctions", [])
    vessels = cfg.get("vessels", [])
    if not isinstance(junctions, list):
        raise ValueError("Expected 'junctions' to be a list in calibration output.")
    if not isinstance(vessels, list):
        raise ValueError("Expected 'vessels' to be a list in calibration output.")

    rows: List[List[float]] = []
    out_junction_names: List[str] = []
    target_names: Optional[List[str]] = None

    for j in junctions:
        j_name = j.get("junction_name", "")
        outlet_vessel_ids = j.get("outlet_vessels", [])
        if require_two_outlets and len(outlet_vessel_ids) != 2:
            continue

        if not j_name:
            raise ValueError("Found junction with empty 'junction_name' in calibration output.")

        jv = j.get("junction_values", None)
        if not isinstance(jv, dict):
            raise ValueError(f"Junction {j_name} missing dict 'junction_values' in calibration output.")

        gp = j.get("geometric_params", {})
        outlet_path_lengths = gp.get("outlet_path_lengths", None)
        if not isinstance(outlet_path_lengths, dict):
            raise ValueError(
                f"Junction {j_name} missing dict geometric_params.outlet_path_lengths in calibration output."
            )

        # Map outlet vessel ids -> names in file order (assumed order of junction_values lists)
        outlet_vessel_names_in_file_order: List[str] = []
        for vid in outlet_vessel_ids:
            if not isinstance(vid, int):
                raise ValueError(f"Outlet vessel id {vid} is not an int for junction {j_name}")
            if vid < 0 or vid >= len(vessels):
                raise ValueError(f"Outlet vessel id {vid} out of bounds for junction {j_name}")
            vname = vessels[vid].get("vessel_name", "")
            if not vname:
                raise ValueError(f"Outlet vessel {vid} has empty vessel_name for junction {j_name}")
            outlet_vessel_names_in_file_order.append(vname)

        outlet_index = {vn: i for i, vn in enumerate(outlet_vessel_names_in_file_order)}
        if len(outlet_index) != len(outlet_vessel_names_in_file_order):
            raise ValueError(f"Duplicate outlet vessel names in junction {j_name}: {outlet_vessel_names_in_file_order}")

        # Sort outlets by descending path length, matching inputs_from_0d_config.py ordering
        # For connector vessels, use a default path length of 0.0 if not in outlet_path_lengths
        def get_path_length(vn):
            if vn in outlet_path_lengths:
                return float(outlet_path_lengths[vn])
            elif 'connector' in vn:
                # Connector vessels may not have path lengths (they have zero length)
                return 0.0
            else:
                raise ValueError(
                    f"Junction {j_name} missing outlet_path_lengths entry for outlet vessel {vn}"
                )
        
        outlet_names_sorted = sorted(
            outlet_vessel_names_in_file_order,
            key=get_path_length,
            reverse=True,
        )
        if require_two_outlets:
            outlet_names_sorted = outlet_names_sorted[:2]

        # Skip junctions where the primary outlet (outlet0) is a connector vessel
        primary_outlet = outlet_names_sorted[0]
        if 'connector' in primary_outlet and 'connectorEL' not in primary_outlet:
            if verbose:
                print(f"Skipping junction {j_name}: primary outlet {primary_outlet} is a connector vessel (not EL-adjusted)")
            continue

        # Build TWO rows per junction: one with outlet0 first, one with outlet1 first (swapped)
        # Row 1: param_outlet0, param_outlet1 (using sorted outlet order)
        row_0_first: List[float] = []
        local_target_names: List[str] = []
        for param_name in sorted(jv.keys()):
            val = jv[param_name]
            if isinstance(val, list):
                if require_two_outlets and len(val) != 2:
                    raise ValueError(
                        f"Junction {j_name} junction_values[{param_name}] expected length 2, got {len(val)}"
                    )
                for out_i, out_vn in enumerate(outlet_names_sorted):
                    idx = outlet_index[out_vn]
                    row_0_first.append(float(val[idx]))
                    import pdb; pdb.set_trace()
                    if len(local_target_names) < len(row_0_first):
                        local_target_names.append(f"{param_name}_outlet{out_i}")
            else:
                # Scalar parameter (rare, but supported)
                row_0_first.append(float(val))
                if len(local_target_names) < len(row_0_first):
                    local_target_names.append(param_name)

        if target_names is None:
            target_names = local_target_names
        elif local_target_names != target_names:
            raise ValueError(
                f"Inconsistent junction_values schema in calibration output. "
                f"Expected {target_names}, got {local_target_names} for junction {j_name}"
            )

        if verbose:
            print(f"Loaded junction_values for {j_name}: {dict(zip(local_target_names, row_0_first))}")

        rows.append(row_0_first)
        out_junction_names.append(j_name)

        # Row 2: param_outlet1, param_outlet0 (swapped order)
        # Skip swapped sample if outlet1 would be a connector (connector-as-primary exclusion)
        if 'connector' not in outlet_names_sorted[1] or 'connectorEL' in outlet_names_sorted[1]:
            row_1_first: List[float] = []
            for param_name in sorted(jv.keys()):
                val = jv[param_name]
                if isinstance(val, list):
                    # Swap: use outlet1 first, then outlet0
                    idx_1 = outlet_index[outlet_names_sorted[1]]
                    idx_0 = outlet_index[outlet_names_sorted[0]]
                    row_1_first.append(float(val[idx_1]))
                    row_1_first.append(float(val[idx_0]))
                else:
                    # Scalar parameter (same for both rows)
                    row_1_first.append(float(val))

            rows.append(row_1_first)
            out_junction_names.append(j_name)  # Same junction name for both rows
        else:
            if verbose:
                print(
                    f"Skipping swapped sample for junction {j_name}: "
                    f"primary outlet would be connector {outlet_names_sorted[1]} (not EL-adjusted)"
                )
    #import pdb; pdb.set_trace()
    if not rows:
        raise ValueError("No junctions with usable junction_values were found in calibration output.")
    if target_names is None:
        raise ValueError("Internal error: target_names not set.")

    Y = np.asarray(rows, dtype=float)

    if junction_names is not None:
        name_to_idx = {n: i for i, n in enumerate(out_junction_names)}
        missing = [n for n in junction_names if n not in name_to_idx]
        if missing:
            raise ValueError(f"Missing junction(s) in calibration output: {missing[:10]}")
        ordered = [Y[name_to_idx[n], :] for n in junction_names]
        Y = np.asarray(ordered, dtype=float)
        out_junction_names = list(junction_names)

    return Y, target_names, out_junction_names


__all__ = [
    "load_junction_lumped_parameters",
]


