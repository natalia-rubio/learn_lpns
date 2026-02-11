#!/usr/bin/env python3
"""
Utilities for extracting ML-ready inputs from 0D configuration files.

Current focus: build a junction-level feature matrix from the geometric
parameters written by `util.zerod_calibration.geometric_params`.

Each row in the returned array corresponds to one junction instance
(after bifurcation_splitting, i.e. each with exactly two outlets).
Each column is a scalar geometric feature derived from `geometric_params`.
"""

import json
from typing import Dict, List, Tuple, Any

import numpy as np


def _safe_get(d: Dict[str, Any], *keys, default=None):
    """Nested dict get with default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_junction_geometric_features(
    config_path: str,
    require_two_outlets: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract a junction-level geometric feature matrix from a 0D config JSON.

    Args:
        config_path:
            Path to a 0D configuration file that has already been processed
            by `geometric_params.extract_and_add_geometric_params`, so that
            each junction carries a `geometric_params` field.
        require_two_outlets:
            If True (default), only junctions with exactly two outlets are
            used (expected case after bifurcation_splitting). If False, all
            junctions are used but may raise if outlets < 2.

    Returns:
        X:
            NumPy array of shape (n_junctions, n_features).
        feature_names:
            List of length n_features describing each column of X.
        junction_names:
            List of length n_junctions; `junction_names[i]` corresponds to
            row `X[i, :]`.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    junctions = cfg.get("junctions", [])
    if not isinstance(junctions, list):
        raise ValueError("Expected 'junctions' to be a list in config.")

    rows: List[List[float]] = []
    junction_names: List[str] = []
    # Per-row primary outlet name (the outlet used as outlet0 for that row)
    outlet_primary_names: List[str] = []

    # We assume bifurcation geometry: each junction has exactly 1 inlet and 2 outlets.
    for j in junctions:
        j_name = j.get("junction_name", "")
        gp = j.get("geometric_params", {})

        outlet_vessels = j.get("outlet_vessels", [])

        if require_two_outlets and len(outlet_vessels) != 2:
            # Skip non-bifurcation junctions in this mode
            if verbose:
                print(f"Skipping junction {j_name}: not a bifurcation junction")
            continue
        if verbose:
            print(f"Processing junction {j_name}: {outlet_vessels}")


        # --- Per-junction scalars ---
        inlet_max_r = _safe_get(gp, "inlet_max_inscribed_radius", default=None)
        inlet_tangent = _safe_get(gp, "inlet_tangent", default=None) or [None, None, None]
        if len(inlet_tangent) != 3:
            inlet_tangent = [None, None, None]

        # --- Per-outlet features (we order outlets by descending path length) ---
        outlet_path_lengths = gp.get("outlet_path_lengths", {}) or {}
        outlet_tortuosities = gp.get("outlet_tortuosities", {}) or {}
        outlet_tangents = gp.get("outlet_tangents", {}) or {}
        outlet_max_r = gp.get("outlet_max_inscribed_radius", {}) or {}
        outlet_max_r_min_path = gp.get("max_inscribed_radius_min_on_path", {}) or {}
        outlet_max_r_max_path = gp.get("max_inscribed_radius_max_on_path", {}) or {}
        outlet_angle_diffs = gp.get("outlet_angle_diffs", {}) or {}
        # Map vessel_name -> metrics
        outlet_names = list(outlet_path_lengths.keys())
        if require_two_outlets and len(outlet_names) != 2:
            # Fall back to using junction's outlet list order (by vessel index) if needed
            outlet_names = []
            vessels = cfg.get("vessels", [])
            for vid in outlet_vessels:
                if 0 <= vid < len(vessels):
                    outlet_names.append(vessels[vid].get("vessel_name", f"v{vid}"))

        

        def _to_float(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                raise ValueError(f"Could not convert {x} to float")
                return None

        # Per-outlet features helper
        def get_outlet_features(outlet_name: str) -> List[float]:
            """Extract all features for a single outlet."""
            pl = outlet_path_lengths.get(outlet_name)
            tor = outlet_tortuosities.get(outlet_name)
            tan = outlet_tangents.get(outlet_name, [None, None, None]) or [None, None, None]
            if len(tan) != 3:
                print(f"Outlet tangent has wrong length: {tan}")
                tan = [None, None, None]
            r_loc = outlet_max_r.get(outlet_name)
            r_min_p = outlet_max_r_min_path.get(outlet_name)
            r_max_p = outlet_max_r_max_path.get(outlet_name)
            ang = outlet_angle_diffs.get(outlet_name)

            if verbose:
                print(f"Adding outlet path length: {pl}")
                print(f"Adding outlet tortuosity: {tor}")
                print(f"Adding outlet tangent features: {tan}")
                print(f"Adding outlet max inscribed radius local: {r_loc}")
                print(f"Adding outlet max inscribed radius min on path: {r_min_p}")
                print(f"Adding outlet max inscribed radius max on path: {r_max_p}")
                print(f"Adding outlet angle diff (inlet vs outlet tangent): {ang}")
            
            return [
                _to_float(pl),
                _to_float(tor),
                _to_float(tan[0]),
                _to_float(tan[1]),
                _to_float(tan[2]),
                _to_float(r_loc),
                _to_float(r_min_p),
                _to_float(r_max_p),
                _to_float(ang),
            ]

    # Build TWO rows per junction: one with outlet0 first, one with outlet1 first
        # Row 1: inlet + outlet0 + outlet1
        if 'connector' not in outlet_names[0] or 'connectorEL' in outlet_names[0]:
            feat_row_0_first: List[float] = [outlet_vessels[0]]
            if verbose:
                print(f"Adding outlet 0 features: {outlet_names[0]}, outlet vessel id: {outlet_vessels[0]}")
                print(f"Adding inlet max inscribed radius: {inlet_max_r}")
            feat_row_0_first.append(_to_float(inlet_max_r))
            feat_row_0_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_0_first.extend(get_outlet_features(outlet_names[0]))
            feat_row_0_first.extend(get_outlet_features(outlet_names[1]))
            rows.append(feat_row_0_first)
            junction_names.append(j_name)
            outlet_primary_names.append(outlet_names[0])

        # Row 2: inlet + outlet1 + outlet0 (swapped)
        # Skip swapped sample if outlet1 would be a connector (connector-as-primary exclusion)
        if 'connector' not in outlet_names[1] or 'connectorEL' in outlet_names[1]:
            print(f"Adding outlet 1 features: {outlet_names[1]}, outlet vessel id: {outlet_vessels[1]} of {outlet_vessels}")
            feat_row_1_first: List[float] = [outlet_vessels[1]]
            feat_row_1_first.append(_to_float(inlet_max_r))
            feat_row_1_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_1_first.extend(get_outlet_features(outlet_names[1]))
            feat_row_1_first.extend(get_outlet_features(outlet_names[0]))
            rows.append(feat_row_1_first)
            junction_names.append(j_name)  # Same junction name for both rows
            outlet_primary_names.append(outlet_names[1])
        else:
            if verbose:
                print(
                    f"Skipping swapped sample for junction {j_name}: "
                    f"primary outlet would be connector {outlet_names[1]}"
                )

    if not rows:
        raise ValueError("No junctions with usable geometric_params were found.")

    X = np.asarray(rows, dtype=float)

    # Feature names in the same order as feat_row construction above
    feature_names: List[str] = [
        "outlet_vessel_id",
        "inlet_max_inscribed_radius",
        "inlet_tangent_x",
        "inlet_tangent_y",
        "inlet_tangent_z",
        "outlet0_path_length",
        "outlet0_tortuosity",
        "outlet0_tangent_x",
        "outlet0_tangent_y",
        "outlet0_tangent_z",
        "outlet0_max_inscribed_radius_local",
        "outlet0_max_inscribed_radius_min_on_path",
        "outlet0_max_inscribed_radius_max_on_path",
        "outlet0_angle_diff",
        "outlet1_path_length",
        "outlet1_tortuosity",
        "outlet1_tangent_x",
        "outlet1_tangent_y",
        "outlet1_tangent_z",
        "outlet1_max_inscribed_radius_local",
        "outlet1_max_inscribed_radius_min_on_path",
        "outlet1_max_inscribed_radius_max_on_path",
        "outlet1_angle_diff",
    ]
    return X, feature_names, junction_names, outlet_primary_names


__all__ = [
    "load_junction_geometric_features",
]


