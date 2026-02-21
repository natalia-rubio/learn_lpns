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


def _safe_div(a, b):
    """Return a/b, or None if either operand is None or b is zero."""
    if a is None or b is None:
        return None
    try:
        return float(a) / float(b) if float(b) != 0.0 else None
    except (TypeError, ValueError):
        return None

def _safe_mult(a, b):
    """Return a*b, or None if either operand is None."""
    if a is None or b is None:
        return None
    try:
        return float(a) * float(b)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Computed per-outlet features
# ---------------------------------------------------------------------------
# Each entry is (suffix, func) where *suffix* becomes the column name
# "outlet{i}_{suffix}" and *func(outlet_raw, junction_raw)* returns a float.
#
# outlet_raw keys:  path_length, tortuosity, tangent, r_local, r_min_path,
#                   r_max_path, angle_diff
# junction_raw keys: inlet_max_r, inlet_tangent
#
# To add a new computed feature, just append a tuple here.
# ---------------------------------------------------------------------------
COMPUTED_OUTLET_FEATURES: List[Tuple[str, Any]] = [
    ("max_inscribed_radius_ratio",
     lambda out, junc: _safe_div(out["r_local"], junc["inlet_max_r"])),
     ("poiseuille_resistance_calc",
     lambda out, junc: _safe_div(8 * 0.04 * out["path_length"], np.pi * out["r_local"]**4)),
     ("inductance_calc",   
     lambda out, junc: _safe_mult(0.06*out["path_length"], out["r_local"]**2)),

]


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

    # Build vessel_id -> vessel_name mapping from the config
    vessels = cfg.get("vessels", [])
    vessel_id_to_name = {}
    vessel_name_to_id = {}
    for v in vessels:
        vid = v.get("vessel_id")
        vname = v.get("vessel_name", "")
        if vid is not None and vname:
            vessel_id_to_name[vid] = vname
            vessel_name_to_id[vname] = vid

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

        # Build authoritative outlet_name -> vessel_id mapping for this junction
        # using the junction's outlet_vessels list and the vessels array
        outlet_vessel_id_map = {}  # vessel_name -> vessel_id
        for vid in outlet_vessels:
            vname = vessel_id_to_name.get(vid, "")
            if vname:
                outlet_vessel_id_map[vname] = vid

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
        outlet_L = gp.get("outlet_L", {}) or {}
        outlet_R_poiseuille = gp.get("outlet_R_poiseuille", {}) or {}
        outlet_stenosis_coeff = gp.get("outlet_stenosis_coefficient", {}) or {}
        # Map vessel_name -> metrics (use outlet_path_lengths keys as canonical names)
        outlet_names = list(outlet_path_lengths.keys())
        if require_two_outlets and len(outlet_names) != 2:
            # Fall back to using junction's outlet list order (by vessel index) if needed
            outlet_names = []
            for vid in outlet_vessels:
                if 0 <= vid < len(vessels):
                    outlet_names.append(vessels[vid].get("vessel_name", f"v{vid}"))

        

        def _to_float(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                raise ValueError(f"Could not convert {x} to float")
                return None

        junction_raw = {"inlet_max_r": inlet_max_r, "inlet_tangent": inlet_tangent}

        def get_outlet_features(outlet_name: str) -> List[float]:
            """Extract raw + computed features for a single outlet."""
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
            L_val = outlet_L.get(outlet_name, 0.0)
            R_pois = outlet_R_poiseuille.get(outlet_name, 0.0)
            sten = outlet_stenosis_coeff.get(outlet_name, 0.0)

            outlet_raw = {
                "path_length": pl, "tortuosity": tor, "tangent": tan,
                "r_local": r_loc, "r_min_path": r_min_p, "r_max_path": r_max_p,
                "angle_diff": ang, "L": L_val, "R_poiseuille": R_pois,
                "stenosis_coefficient": sten,
            }

            features = [
                _to_float(pl),
                _to_float(tor),
                _to_float(tan[0]),
                _to_float(tan[1]),
                _to_float(tan[2]),
                _to_float(r_loc),
                _to_float(r_min_p),
                _to_float(r_max_p),
                _to_float(ang),
                _to_float(L_val),
                _to_float(R_pois),
                _to_float(sten),
            ]

            for _name, func in COMPUTED_OUTLET_FEATURES:
                features.append(_to_float(func(outlet_raw, junction_raw)))

            return features

    # Build TWO rows per junction: one with outlet0 first, one with outlet1 first
        # Look up vessel IDs from outlet_names via the authoritative mapping
        outlet0_name = outlet_names[0]
        outlet1_name = outlet_names[1]
        outlet0_vid = outlet_vessel_id_map.get(outlet0_name)
        outlet1_vid = outlet_vessel_id_map.get(outlet1_name)
        
        if outlet0_vid is None:
            raise ValueError(
                f"Junction {j_name}: outlet '{outlet0_name}' not found in outlet_vessels {outlet_vessels}. "
                f"Vessel name-to-id mapping: {outlet_vessel_id_map}"
            )
        if outlet1_vid is None:
            raise ValueError(
                f"Junction {j_name}: outlet '{outlet1_name}' not found in outlet_vessels {outlet_vessels}. "
                f"Vessel name-to-id mapping: {outlet_vessel_id_map}"
            )
        
        if verbose:
            print(f"  outlet0: {outlet0_name} (vessel_id={outlet0_vid}), outlet1: {outlet1_name} (vessel_id={outlet1_vid})")

        # Row 1: inlet + outlet0 + outlet1
        if 'connector' not in outlet0_name or 'connectorEL' in outlet0_name:
            feat_row_0_first: List[float] = [outlet0_vid]
            if verbose:
                print(f"Adding outlet 0 features: {outlet0_name}, outlet vessel id: {outlet0_vid}")
                print(f"Adding inlet max inscribed radius: {inlet_max_r}")
            feat_row_0_first.append(_to_float(inlet_max_r))
            feat_row_0_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_0_first.extend(get_outlet_features(outlet0_name))
            feat_row_0_first.extend(get_outlet_features(outlet1_name))
            rows.append(feat_row_0_first)
            junction_names.append(j_name)
            outlet_primary_names.append(outlet0_name)

        # Row 2: inlet + outlet1 + outlet0 (swapped)
        # Skip swapped sample if outlet1 would be a connector (connector-as-primary exclusion)
        if 'connector' not in outlet1_name or 'connectorEL' in outlet1_name:
            if verbose:
                print(f"Adding outlet 1 features: {outlet1_name}, outlet vessel id: {outlet1_vid}")
            feat_row_1_first: List[float] = [outlet1_vid]
            feat_row_1_first.append(_to_float(inlet_max_r))
            feat_row_1_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_1_first.extend(get_outlet_features(outlet1_name))
            feat_row_1_first.extend(get_outlet_features(outlet0_name))
            rows.append(feat_row_1_first)
            junction_names.append(j_name)  # Same junction name for both rows
            outlet_primary_names.append(outlet1_name)
        else:
            if verbose:
                print(
                    f"Skipping swapped sample for junction {j_name}: "
                    f"primary outlet would be connector {outlet1_name}"
                )

    if not rows:
        raise ValueError("No junctions with usable geometric_params were found.")

    X = np.asarray(rows, dtype=float)

    # Feature names in the same order as feat_row construction above
    _raw_outlet_suffixes = [
        "path_length",
        "tortuosity",
        "tangent_x",
        "tangent_y",
        "tangent_z",
        "max_inscribed_radius_local",
        "max_inscribed_radius_min_on_path",
        "max_inscribed_radius_max_on_path",
        "angle_diff",
        "absorbed_L",
        "absorbed_R_poiseuille",
        "absorbed_stenosis_coefficient",
    ]
    _computed_outlet_suffixes = [name for name, _ in COMPUTED_OUTLET_FEATURES]
    _all_outlet_suffixes = _raw_outlet_suffixes + _computed_outlet_suffixes

    feature_names: List[str] = [
        "outlet_vessel_id",
        "inlet_max_inscribed_radius",
        "inlet_tangent_x",
        "inlet_tangent_y",
        "inlet_tangent_z",
    ]
    for prefix in ("outlet0", "outlet1"):
        for suffix in _all_outlet_suffixes:
            feature_names.append(f"{prefix}_{suffix}")
    return X, feature_names, junction_names, outlet_primary_names


__all__ = [
    "load_junction_geometric_features",
]


