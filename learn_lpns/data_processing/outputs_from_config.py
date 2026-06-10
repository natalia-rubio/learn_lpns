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

import numpy as np


def load_junction_lumped_parameters(
    calibration_output_path: str,
    require_two_outlets: bool = True,
    junction_names: list[str] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Extract a junction-level target matrix from a calibration output JSON.

    This expects each junction to contain:
      - `junction_name`
      - `outlet_vessels` (list of vessel indices)
      - `junction_values` (dict of parameter_name -> list[float] per outlet)

    Args:
        calibration_output_path: path to a calibrated output JSON.
        require_two_outlets: if True, only use junctions with exactly two outlets.
        junction_names: if provided, verify rows appear in exactly this order
            (and raise if any are missing).
        verbose: print debug information.

    Returns:
        Y: (n_junctions, n_targets) float array
        target_names: list of column names
        out_junction_names: list of junction names corresponding to Y rows
        out_primary_outlet_names: list of primary outlet vessel names per row
    """
    with open(calibration_output_path) as f:
        cfg = json.load(f)

    junctions = cfg.get("junctions", [])
    vessels = cfg.get("vessels", [])
    if not isinstance(junctions, list):
        raise ValueError("Expected 'junctions' to be a list in calibration output.")
    if not isinstance(vessels, list):
        raise ValueError("Expected 'vessels' to be a list in calibration output.")

    rows: list[list[float]] = []
    out_junction_names: list[str] = []
    out_primary_outlet_names: list[str] = []
    target_names: list[str] | None = None

    # Build vessel_id -> vessel_name mapping
    vessel_id_to_name = {}
    for v in vessels:
        vid = v.get("vessel_id")
        vname = v.get("vessel_name", "")
        if vid is not None and vname:
            vessel_id_to_name[vid] = vname

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
        assert len(outlet_vessel_ids) == 2, f"Junction {j_name} has unexpected number of outlets: {outlet_vessel_ids}"
        for i in range(len(outlet_vessel_ids)):
            vessel_name = vessel_id_to_name.get(outlet_vessel_ids[i], "")
            if not vessel_name:
                vessel_name = vessels[outlet_vessel_ids[i]].get("vessel_name", "")
            other_i = abs(i - 1)
            if verbose:
                print(f"Processing outlet {i} of junction {j_name}: {vessel_name} (vessel_id={outlet_vessel_ids[i]})")
            if "connector" in vessel_name and "connectorEL" not in vessel_name:
                if verbose:
                    print(
                        f"Skipping outlet {i} of junction {j_name}: {vessel_name} "
                        f"is a connector vessel (not EL-adjusted)"
                    )
                continue
            if verbose:
                print(
                    f"Adding outlet {i} of junction {j_name}: {vessel_name} "
                    f"with outlet vessel id: {outlet_vessel_ids[i]}"
                )
            row = [
                outlet_vessel_ids[i],
            ]
            for param_name in sorted(jv.keys()):
                val = jv[param_name]
                if isinstance(val, list):
                    row.append(float(val[i]))
                    row.append(float(val[other_i]))

            rows.append(row)
            out_junction_names.append(j_name)
            out_primary_outlet_names.append(vessel_name)

        target_names = ["outlet_vessel_id"] + [
            f"{param_name}_outlet{i}" for param_name in sorted(jv.keys()) for i in range(len(outlet_vessel_ids))
        ]
    if not rows:
        raise ValueError("No junctions with usable junction_values were found in calibration output.")
    if target_names is None:
        raise ValueError("Internal error: target_names not set.")

    Y = np.asarray(rows, dtype=float)

    return Y, target_names, out_junction_names, out_primary_outlet_names


__all__ = [
    "load_junction_lumped_parameters",
]
