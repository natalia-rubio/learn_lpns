#!/usr/bin/env python3
"""
Utilities for extracting ML-ready inputs from 0D configuration files.

Current focus: build a junction-level feature matrix from the geometric
parameters written by `learn_lpns.zerod_calibration.geometric_params`.

Each row in the returned array corresponds to one junction instance
(after bifurcation_splitting, i.e. each with exactly two outlets).
Each column is a scalar geometric feature derived from `geometric_params`.

Includes **generation**: count of two-outlet junctions along the path from the root
inlet vessel (id-based ``inlet_vessels`` / ``outlet_vessels`` wiring). Root vessel
and first bifurcation junction have generation 0.
"""

import json
import os
import re
from collections import deque
from typing import Any

import numpy as np

from learn_lpns.zerod_calibration.tools.file_io import read_zerod_csv


def _safe_get(d: dict[str, Any], *keys, default=None):
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
COMPUTED_OUTLET_FEATURES: list[tuple[str, Any]] = [
    ("radius_ratio", lambda out, junc: _safe_div(out["r_local"], junc["inlet_max_r"])),
    (
        "poiseuille_resistance_calc",
        lambda out, junc: _safe_div(8 * 0.04 * out["path_length"], np.pi * out["r_local"] ** 4),
    ),
    (
        "inductance_calc",
        lambda out, junc: _safe_mult(1.06 * out["path_length"], out["r_local"] ** 2),
    ),
    ("rneg4", lambda out, junc: out["r_local"] ** -4),
    ("rneg2", lambda out, junc: out["r_local"] ** -2),
    ("rmin_rat", lambda out, junc: _safe_div(out["r_min_path"], out["r_local"])),
    ("rmax_rat", lambda out, junc: _safe_div(out["r_max_path"], out["r_local"])),
    ("nd_length", lambda out, junc: out["path_length"] / out["r_local"]),
]

# ---------------------------------------------------------------------------
# Computed per-vessel features (for vessel NN input)
# ---------------------------------------------------------------------------
# Each entry is (name, func) where func(vessel_raw) returns a float or None.
# vessel_raw has: inlet_area, outlet_area, path_length, area_ratio,
# r_local (sqrt(inlet_area/pi)), inlet_max_r (inlet_max_inscribed_radius).
# To add a new computed vessel feature, append a tuple here.
# ---------------------------------------------------------------------------
COMPUTED_VESSEL_FEATURES: list[tuple[str, Any]] = [
    ("radius_ratio", lambda d: _safe_div(d["outlet_max_r"], d["inlet_max_r"])),
    ("rmin_rat", lambda d: _safe_div(d["max_inscribed_radius_min"], d["outlet_max_r"])),
    ("rmax_rat", lambda d: _safe_div(d["max_inscribed_radius_max"], d["outlet_max_r"])),
    ("nd_length", lambda d: _safe_div(d["path_length"], d["outlet_max_r"])),
    (
        "poiseuille_resistance_calc",
        lambda d: _safe_div(8.0 * 0.04 * d["path_length"], np.pi * (d["r_local"] ** 4)),
    ),
    ("inductance_calc", lambda d: _safe_div(1.06 * d["path_length"], d["inlet_area"])),
    (
        "stenosis_calc",
        lambda d: max(0.0, 1.0 - d["area_ratio"]) if d.get("inlet_area", 0) > 0 else 0.0,
    ),
    ("rneg4", lambda d: d["outlet_max_r"] ** -4),
    ("rneg2", lambda d: d["outlet_max_r"] ** -2),
]


def _find_root_vessel_id_for_generation(cfg: dict[str, Any]):
    """
    Vessel_id of the tree root (inlet branch). Prefer vessel with inlet BC, then
    name containing branch0, else minimum vessel_id.
    """
    vessels = cfg.get("vessels", [])
    for v in vessels:
        vid = v.get("vessel_id")
        if vid is None:
            continue
        bc = v.get("boundary_conditions")
        if isinstance(bc, dict) and "inlet" in bc:
            return vid
    for v in vessels:
        name = v.get("vessel_name", "") or ""
        if "branch0" in name:
            vid = v.get("vessel_id")
            if vid is not None:
                return vid
    ids = [v.get("vessel_id") for v in vessels if v.get("vessel_id") is not None]
    return min(ids) if ids else None


def compute_bifurcation_generation_by_vessel(cfg: dict[str, Any]) -> dict[Any, float]:
    """
    Map vessel_id -> generation: number of 2-outlet junctions along the path from
    the root inlet vessel to this vessel. The root vessel has generation 0.

    Each time the path crosses a junction with exactly two elements in
    ``outlet_vessels``, generation increments by one for all downstream outlets.
    """
    root = _find_root_vessel_id_for_generation(cfg)
    if root is None:
        raise ValueError("Cannot compute bifurcation generation: no root inlet vessel found in config.")
    try:
        root = int(root)
    except (TypeError, ValueError):
        pass
    junctions = cfg.get("junctions", []) or []
    gen: dict[Any, float] = {root: 0.0}
    q = deque([root])
    while q:
        vid = q.popleft()
        g_here = gen[vid]
        for j in junctions:
            inlets_raw = j.get("inlet_vessels") or []
            inlets = []
            for iv in inlets_raw:
                try:
                    inlets.append(int(iv))
                except (TypeError, ValueError):
                    inlets.append(iv)
            if vid not in inlets:
                continue
            outs = j.get("outlet_vessels") or []
            inc = 1.0 if len(outs) == 2 else 0.0
            g_next = g_here + inc
            for oid in outs:
                if oid is None:
                    continue
                try:
                    oid_int = int(oid)
                except (TypeError, ValueError):
                    oid_int = oid
                if oid_int not in gen:
                    gen[oid_int] = float(g_next)
                    q.append(oid_int)
                else:
                    gen[oid_int] = min(gen[oid_int], float(g_next))
    out: dict[Any, float] = {}
    for k, v in gen.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            out[k] = float(v)
    return out


def load_junction_geometric_features(
    config_path: str,
    require_two_outlets: bool = True,
    verbose: bool = False,
    geometric_results_path: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
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
        geometric_results_path:
            Optional path to geometric simulation results CSV (e.g.
            ``bifurcations_EL_geometric_results.csv``). When present, appends
            ``flow_split``, ``flow_split_inv``, and ``speed_change`` columns.

    Returns:
        X:
            NumPy array of shape (n_junctions, n_features).
        feature_names:
            List of length n_features describing each column of X.
        junction_names:
            List of length n_junctions; ``junction_names[i]`` corresponds to
            row ``X[i, :]``.
        outlet_primary_names:
            Primary outlet vessel name for each row (used for flow_split).
    """
    with open(config_path) as f:
        cfg = json.load(f)

    gen_by_vessel = compute_bifurcation_generation_by_vessel(cfg)

    junctions = cfg.get("junctions", [])
    if not isinstance(junctions, list):
        raise ValueError("Expected 'junctions' to be a list in config.")

    rows: list[list[float]] = []
    junction_names: list[str] = []
    # Per-row primary outlet name (the outlet used as outlet0 for that row)
    outlet_primary_names: list[str] = []

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

        if not j_name:
            raise ValueError("Junction missing junction_name in config.")

        inlet_ids_list = j.get("inlet_vessels", []) or []
        if not inlet_ids_list:
            raise ValueError(f"Junction {j_name!r}: no inlet_vessels (expected one inlet for id-based wiring).")
        if len(inlet_ids_list) != 1:
            raise ValueError(
                f"Junction {j_name!r}: expected exactly one inlet vessel for id-based wiring, "
                f"got {len(inlet_ids_list)}: {inlet_ids_list!r}"
            )
        inlet_vid0 = inlet_ids_list[0]
        try:
            inlet_key = int(inlet_vid0)
        except (TypeError, ValueError):
            inlet_key = inlet_vid0
        if inlet_key not in gen_by_vessel:
            raise ValueError(
                f"Junction {j_name!r}: inlet vessel {inlet_key!r} has no bifurcation generation "
                f"(not reachable from root inlet vessel)."
            )
        generation_val = float(gen_by_vessel[inlet_key])

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
                raise ValueError(f"Could not convert {x} to float") from None

        junction_raw = {"inlet_max_r": inlet_max_r, "inlet_tangent": inlet_tangent}

        def get_outlet_features(
            outlet_name: str,
            *,
            _outlet_path_lengths=outlet_path_lengths,
            _outlet_tortuosities=outlet_tortuosities,
            _outlet_tangents=outlet_tangents,
            _outlet_max_r=outlet_max_r,
            _outlet_max_r_min_path=outlet_max_r_min_path,
            _outlet_max_r_max_path=outlet_max_r_max_path,
            _outlet_angle_diffs=outlet_angle_diffs,
            _outlet_L=outlet_L,
            _outlet_R_poiseuille=outlet_R_poiseuille,
            _outlet_stenosis_coeff=outlet_stenosis_coeff,
            _junction_raw=junction_raw,
        ) -> list[float]:
            """Extract raw + computed features for a single outlet."""
            pl = _outlet_path_lengths.get(outlet_name)
            tor = _outlet_tortuosities.get(outlet_name)
            tan = _outlet_tangents.get(outlet_name, [None, None, None]) or [None, None, None]
            if len(tan) != 3:
                print(f"Outlet tangent has wrong length: {tan}")
                tan = [None, None, None]
            r_loc = _outlet_max_r.get(outlet_name)
            r_min_p = _outlet_max_r_min_path.get(outlet_name)
            r_max_p = _outlet_max_r_max_path.get(outlet_name)
            ang = _outlet_angle_diffs.get(outlet_name)
            L_val = _outlet_L.get(outlet_name, 0.0)
            R_pois = _outlet_R_poiseuille.get(outlet_name, 0.0)
            sten = _outlet_stenosis_coeff.get(outlet_name, 0.0)

            outlet_raw = {
                "path_length": pl,
                "tortuosity": tor,
                "tangent": tan,
                "r_local": r_loc,
                "r_min_path": r_min_p,
                "r_max_path": r_max_p,
                "angle_diff": ang,
                "L": L_val,
                "R_poiseuille": R_pois,
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
                features.append(_to_float(func(outlet_raw, _junction_raw)))

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
            print(
                f"  outlet0: {outlet0_name} (vessel_id={outlet0_vid}), "
                f"outlet1: {outlet1_name} (vessel_id={outlet1_vid})"
            )

        # Row 1: inlet + outlet0 + outlet1
        if "connector" not in outlet0_name or "connectorEL" in outlet0_name:
            feat_row_0_first: list[float] = [outlet0_vid]
            if verbose:
                print(f"Adding outlet 0 features: {outlet0_name}, outlet vessel id: {outlet0_vid}")
                print(f"Adding inlet max inscribed radius: {inlet_max_r}")
            feat_row_0_first.append(_to_float(inlet_max_r))
            feat_row_0_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_0_first.append(generation_val)
            feat_row_0_first.extend(get_outlet_features(outlet0_name))
            feat_row_0_first.extend(get_outlet_features(outlet1_name))
            rows.append(feat_row_0_first)
            junction_names.append(j_name)
            outlet_primary_names.append(outlet0_name)

        # Row 2: inlet + outlet1 + outlet0 (swapped)
        # Skip swapped sample if outlet1 would be a connector (connector-as-primary exclusion)
        if "connector" not in outlet1_name or "connectorEL" in outlet1_name:
            if verbose:
                print(f"Adding outlet 1 features: {outlet1_name}, outlet vessel id: {outlet1_vid}")
            feat_row_1_first: list[float] = [outlet1_vid]
            feat_row_1_first.append(_to_float(inlet_max_r))
            feat_row_1_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_1_first.append(generation_val)
            feat_row_1_first.extend(get_outlet_features(outlet1_name))
            feat_row_1_first.extend(get_outlet_features(outlet0_name))
            rows.append(feat_row_1_first)
            junction_names.append(j_name)  # Same junction name for both rows
            outlet_primary_names.append(outlet1_name)
        else:
            if verbose:
                print(
                    f"Skipping swapped sample for junction {j_name}: primary outlet would be connector {outlet1_name}"
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

    feature_names: list[str] = [
        "outlet_vessel_id",
        "inlet_max_inscribed_radius",
        "inlet_tangent_x",
        "inlet_tangent_y",
        "inlet_tangent_z",
        "generation",
    ]
    for prefix in ("outlet0", "outlet1"):
        for suffix in _all_outlet_suffixes:
            feature_names.append(f"{prefix}_{suffix}")

    if geometric_results_path and os.path.exists(geometric_results_path):
        junction_by_name = {j.get("junction_name", ""): j for j in junctions}
        flow_splits = compute_junction_flow_splits(
            config_path,
            geometric_results_path,
            require_two_outlets=require_two_outlets,
        )
        flow_split_col: list[float] = []
        speed_change_col: list[float] = []
        for i, jname in enumerate(junction_names):
            primary = outlet_primary_names[i]
            if jname not in flow_splits:
                flow_split_col.append(np.nan)
                speed_change_col.append(np.nan)
                continue
            (out0_name, out1_name), (fs0, fs1) = flow_splits[jname]
            if primary == out0_name:
                flow_split = fs0
            elif primary == out1_name:
                flow_split = fs1
            else:
                flow_split = np.nan
            flow_split_col.append(float(flow_split))
            speed_change_col.append(_speed_change_for_junction_row(cfg, junction_by_name, jname, primary, flow_split))

        flow_split_arr = np.asarray(flow_split_col, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            flow_split_inv = np.where(
                np.isfinite(flow_split_arr) & (flow_split_arr > 0),
                100.0 / flow_split_arr,
                np.nan,
            )
        X = np.column_stack([X, flow_split_arr, flow_split_inv, np.asarray(speed_change_col, dtype=float)])
        feature_names = [*feature_names, "flow_split", "flow_split_inv", "speed_change"]
        if verbose:
            print(f"  Added flow_split, flow_split_inv, speed_change from {geometric_results_path}")
    elif geometric_results_path and verbose:
        print(f"  Geometric results not found: {geometric_results_path}; skipping simulation features")

    return X, feature_names, junction_names, outlet_primary_names


def _is_split_connector(vessel_name: str) -> bool:
    """True if vessel is a connector created by junction splitting (_connector0, _connector1, ...)."""
    return bool(re.search(r"_connector\d+$", vessel_name))


def _resolve_original_inlet_per_junction(cfg: dict[str, Any]) -> dict[str, str]:
    """
    For each junction (with two outlets), resolve the original inlet vessel name:
    the vessel that carries the total flow into the original (possibly multi-outlet) junction.
    When a multi-outlet junction was split into multiple bifurcations, trace back through
    connector inlets to the non-connector inlet of the first bifurcation in the chain.

    Returns:
        Dict mapping junction_name -> original_inlet_vessel_name.
    """
    vessels = cfg.get("vessels", [])
    junctions = cfg.get("junctions", [])
    vessel_id_to_name = {
        v.get("vessel_id"): v.get("vessel_name", "") for v in vessels if v.get("vessel_id") is not None
    }
    # vessel_id -> junction that has this vessel as an outlet (for tracing back)
    outlet_vessel_id_to_junction: dict[int, str] = {}
    for j in junctions:
        j_name = j.get("junction_name", "")
        for vid in j.get("outlet_vessels", []):
            outlet_vessel_id_to_junction[vid] = j_name

    junction_to_inlet_id: dict[str, int] = {}
    for j in junctions:
        inlets = j.get("inlet_vessels", [])
        if inlets:
            junction_to_inlet_id[j.get("junction_name", "")] = inlets[0]

    out: dict[str, str] = {}
    for j in junctions:
        j_name = j.get("junction_name", "")
        if not j_name or len(j.get("outlet_vessels", [])) != 2:
            continue
        current_inlet_id = junction_to_inlet_id.get(j_name)
        if current_inlet_id is None:
            continue
        # Walk back while the inlet is a split connector
        while True:
            inlet_name = vessel_id_to_name.get(current_inlet_id, "")
            if not inlet_name or not _is_split_connector(inlet_name):
                break
            # This inlet is a connector; find the junction that has it as outlet
            prev_junction = outlet_vessel_id_to_junction.get(current_inlet_id)
            if not prev_junction or prev_junction == j_name:
                break
            prev_inlet_id = junction_to_inlet_id.get(prev_junction)
            if prev_inlet_id is None:
                break
            current_inlet_id = prev_inlet_id
        out[j_name] = vessel_id_to_name.get(current_inlet_id, "")
    return out


def compute_junction_flow_splits(
    config_path: str,
    geometric_results_csv_path: str,
    require_two_outlets: bool = True,
) -> dict[str, tuple[float, float]]:
    """
    Compute flow split (percentage of inlet flow through each outlet) from the
    base geometric 0D simulation results. If multiple timepoints exist, the
    ratio is averaged over time.

    For junctions that came from splitting a multi-outlet junction into multiple
    bifurcations, the denominator is the flow through the *original* inlet (the
    vessel that fed the original multi-outlet junction), not the direct inlet
    of each split bifurcation.

    Args:
        config_path: Path to the 0D geometric config JSON (same as used for
            load_junction_geometric_features).
        geometric_results_csv_path: Path to the geometric simulation results CSV
            (e.g. bifurcations_EL_geometric_results.csv).
        require_two_outlets: If True, only junctions with exactly two outlets
            are included (same convention as load_junction_geometric_features).

    Returns:
        Dict mapping junction_name -> ((outlet0_name, outlet1_name), (flow_split0_pct, flow_split1_pct)).
        Flow splits are in [0, 100]. Missing/invalid data yields (( "", ""), (nan, nan)).
    """
    out: dict[str, tuple[tuple[str, str], tuple[float, float]]] = {}
    if not os.path.exists(geometric_results_csv_path):
        return out

    with open(config_path) as f:
        cfg = json.load(f)

    vessels = cfg.get("vessels", [])
    vessel_id_to_name = {
        v.get("vessel_id"): v.get("vessel_name", "") for v in vessels if v.get("vessel_id") is not None
    }
    original_inlet_by_junction = _resolve_original_inlet_per_junction(cfg)

    results, times = read_zerod_csv(geometric_results_csv_path)
    if not times:
        return out

    for j in cfg.get("junctions", []):
        j_name = j.get("junction_name", "")
        outlet_vessels = j.get("outlet_vessels", [])
        inlet_vessels = j.get("inlet_vessels", [])

        if require_two_outlets and len(outlet_vessels) != 2:
            continue
        if not inlet_vessels:
            raise ValueError(f"Junction {j_name!r} has no inlet vessels; cannot compute flow split.")

        # Use original inlet (trace back through connectors) for denominator
        original_inlet_name = original_inlet_by_junction.get(j_name, "")
        if not original_inlet_name:
            inlet_vid = inlet_vessels[0]
            original_inlet_name = vessel_id_to_name.get(inlet_vid, "")
        out0_name = vessel_id_to_name.get(outlet_vessels[0], "")
        out1_name = vessel_id_to_name.get(outlet_vessels[1], "")

        if not original_inlet_name or not out0_name or not out1_name:
            raise ValueError(
                f"Junction {j_name!r}: missing inlet or outlet names "
                f"(original_inlet={original_inlet_name!r}, out0={out0_name!r}, out1={out1_name!r})."
            )
        if original_inlet_name not in results or out0_name not in results or out1_name not in results:
            raise ValueError(
                f"Junction {j_name!r}: original inlet or outlets not found in geometric results. "
                f"Results keys include: {list(results.keys())[:5]}..."
            )

        ratios0: list[float] = []
        ratios1: list[float] = []
        for t in times:
            # Denominator: flow through original inlet (flow_out of that vessel at junction)
            inlet_data = results[original_inlet_name].get(t, {})
            out0_data = results[out0_name].get(t, {})
            out1_data = results[out1_name].get(t, {})
            q_in = inlet_data.get("flow_out")
            q0 = out0_data.get("flow_in")
            q1 = out1_data.get("flow_in")
            # Skip timesteps where inlet flow is missing, zero, or below threshold (avoids ratio blow-up)
            if q_in is None or q_in < 5.0:
                continue
            if q0 is not None:
                ratios0.append(float(q0) / float(q_in))
            if q1 is not None:
                ratios1.append(float(q1) / float(q_in))

        if not ratios0 and not ratios1:
            # All timesteps had inlet flow < 5; use default 50% / 50%
            fs0, fs1 = 50.0, 50.0
        elif not ratios0 or not ratios1:
            raise ValueError(
                f"Junction {j_name!r}: inconsistent flow split (one outlet has flow data, the other does not)."
            )
        else:
            fs0 = float(np.mean(ratios0)) * 100.0
            fs1 = float(np.mean(ratios1)) * 100.0
        out[j_name] = ((out0_name, out1_name), (fs0, fs1))

    return out


def load_vessel_geometric_features(
    config_path: str,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    """
    Extract a vessel-level geometric feature matrix from a 0D config JSON.

    One row per non-connector vessel. Connectors (vessel_name contains 'connector')
    are skipped. Config must have been processed by geometric_params so each
    vessel has geometric_params (inlet_area, outlet_area, path_length, tortuosity,
    angle_diff) and vessel_length.

    Returns:
        X: NumPy array of shape (n_vessels, n_features).
        feature_names: List of column names for X.
        vessel_ids: List of vessel_id; vessel_ids[i] corresponds to row X[i, :].
        vessel_names: List of vessel_name; vessel_names[i] corresponds to row X[i, :].
    """
    with open(config_path) as f:
        cfg = json.load(f)

    gen_by_vessel = compute_bifurcation_generation_by_vessel(cfg)

    vessels = cfg.get("vessels", [])
    if not isinstance(vessels, list):
        raise ValueError("Expected 'vessels' to be a list in config.")

    # Feature column names (order must match row construction below)
    # vessel_id, is_inlet, generation, base geometric params + all COMPUTED_VESSEL_FEATURES
    # + zero_d_element_values from config
    _computed_vessel_suffixes = [name for name, _ in COMPUTED_VESSEL_FEATURES]
    feature_names = [
        "vessel_id",
        "is_inlet",
        "generation",
        "vessel_length",
        "inlet_area",
        "outlet_area",
        "path_length",
        "tortuosity",
        "angle_diff",
        "area_ratio",
        "inlet_max_inscribed_radius",
        "outlet_max_inscribed_radius",
        "max_inscribed_radius_min",
        "max_inscribed_radius_max",
        *_computed_vessel_suffixes,
        "R_poiseuille_geometric",
        "L_geometric",
        "stenosis_coefficient_geometric",
    ]

    rows: list[list[float]] = []
    vessel_ids: list[int] = []
    vessel_names_out: list[str] = []

    for v in vessels:
        vessel_name = v.get("vessel_name", "")
        if not vessel_name:
            continue
        if "connector" in vessel_name.lower():
            if verbose:
                print(f"Skipping connector vessel: {vessel_name}")
            continue

        vessel_id = v.get("vessel_id")
        if vessel_id is None:
            continue
        is_inlet = 1.0 if "branch0" in vessel_name else 0.0
        try:
            vk = int(vessel_id)
        except (TypeError, ValueError):
            vk = vessel_id
        if vk not in gen_by_vessel:
            raise ValueError(
                f"Vessel {vessel_name!r} (id={vk}): no bifurcation generation (not reachable from root inlet vessel)."
            )
        gnum = float(gen_by_vessel[vk])
        vessel_length = float(v.get("vessel_length", 0.0) or 0.0)
        gp = v.get("geometric_params") or {}
        inlet_area = float(gp.get("inlet_area", 0.0) or 0.0)
        outlet_area = float(gp.get("outlet_area", 0.0) or 0.0)
        path_length = float(gp.get("path_length", 0.0) or 0.0)
        tortuosity = float(gp.get("tortuosity", 0.0) or 0.0)
        angle_diff = float(gp.get("angle_diff", 0.0) or 0.0)
        area_ratio = outlet_area / inlet_area if inlet_area > 0 else 0.0
        r_local = np.sqrt(inlet_area / np.pi) if inlet_area > 0 else 0.0

        # MISR: inlet, outlet, min and max along vessel (from geometric_params; 0 if missing)
        inlet_misr = float(gp.get("inlet_max_inscribed_radius", 0.0) or 0.0)
        outlet_misr = float(gp.get("outlet_max_inscribed_radius", 0.0) or 0.0)
        misr_min = float(gp.get("max_inscribed_radius_min", 0.0) or 0.0)
        misr_max = float(gp.get("max_inscribed_radius_max", 0.0) or 0.0)

        # Values from zero_d_element_values in the config (from centerline/oned_to_zerod or pipeline)
        z = v.get("zero_d_element_values") or {}
        R_poiseuille_geometric = float(z.get("R_poiseuille", 0.0) or 0.0)
        L_geometric = float(z.get("L", 0.0) or 0.0)
        stenosis_coefficient_geometric = float(z.get("stenosis_coefficient", 0.0) or 0.0)

        vessel_raw = {
            "inlet_area": inlet_area,
            "outlet_area": outlet_area,
            "path_length": path_length,
            "area_ratio": area_ratio,
            "r_local": r_local,
            "inlet_max_r": inlet_misr,
            "outlet_max_r": outlet_misr,
            "max_inscribed_radius_min": misr_min,
            "max_inscribed_radius_max": misr_max,
        }
        computed_vals = []
        for _name, func in COMPUTED_VESSEL_FEATURES:
            val = func(vessel_raw)
            computed_vals.append(float(val) if val is not None else 0.0)

        row = [
            float(vessel_id),
            is_inlet,
            gnum,
            vessel_length,
            inlet_area,
            outlet_area,
            path_length,
            tortuosity,
            angle_diff,
            area_ratio,
            inlet_misr,
            outlet_misr,
            misr_min,
            misr_max,
            *computed_vals,
            R_poiseuille_geometric,
            L_geometric,
            stenosis_coefficient_geometric,
        ]
        rows.append(row)
        vessel_ids.append(int(vessel_id))
        vessel_names_out.append(vessel_name)

    if not rows:
        raise ValueError("No non-connector vessels with geometric_params found in config.")

    X = np.asarray(rows, dtype=float)
    return X, feature_names, vessel_ids, vessel_names_out


def load_vessel_targets_from_config(
    calibrated_config_path: str,
) -> tuple[list[int], list[str], np.ndarray]:
    """
    Load vessel targets (R_poiseuille, stenosis_coefficient, L) from a calibrated
    0D config JSON. Only non-connector vessels are included; order matches
    config vessel order.

    Returns:
        vessel_ids: List of vessel_id.
        vessel_names: List of vessel_name.
        targets: Array of shape (n_vessels, 3) with columns [R_poiseuille, stenosis_coefficient, L].
    """
    with open(calibrated_config_path) as f:
        cfg = json.load(f)

    vessels = cfg.get("vessels", [])
    vessel_ids = []
    vessel_names = []
    rows = []

    for v in vessels:
        vessel_name = v.get("vessel_name", "")
        if not vessel_name or "connector" in vessel_name.lower():
            continue
        vessel_id = v.get("vessel_id")
        if vessel_id is None:
            continue
        z = v.get("zero_d_element_values") or {}
        R = float(z.get("R_poiseuille", 0.0) or 0.0)
        S = float(z.get("stenosis_coefficient", 0.0) or 0.0)
        L = float(z.get("L", 0.0) or 0.0)
        vessel_ids.append(int(vessel_id))
        vessel_names.append(vessel_name)
        rows.append([R, S, L])

    if not rows:
        return [], [], np.zeros((0, 3), dtype=float)
    return vessel_ids, vessel_names, np.asarray(rows, dtype=float)


def _original_inlet_area_for_junction(cfg: dict[str, Any], junction_name: str) -> float | None:
    """Cross-sectional area at the trunk inlet used as the flow_split reference."""
    original_inlet = _resolve_original_inlet_per_junction(cfg).get(junction_name, "")
    if not original_inlet:
        return None

    if "_bif" in junction_name:
        bif0_name = f"{junction_name.split('_bif')[0]}_bif0"
    else:
        bif0_name = junction_name

    junction_by_name = {j.get("junction_name", ""): j for j in cfg.get("junctions", [])}
    for candidate in (bif0_name, junction_name):
        j = junction_by_name.get(candidate)
        if not j:
            continue
        areas = _safe_get(j, "geometric_params", "inlet_vessel_areas", default={}) or {}
        if original_inlet in areas and areas[original_inlet] is not None:
            area = float(areas[original_inlet])
            return area if area > 0.0 else None
    return None


def _speed_change_from_areas(flow_split_pct: float, a_in: float, a_out: float) -> float:
    """``((100 / flow_split) / A_in)^2 - (1 / A_out)^2``; NaN when inputs are invalid."""
    if not np.isfinite(flow_split_pct) or flow_split_pct <= 0.0:
        return float("nan")
    if a_in <= 0.0 or a_out <= 0.0:
        return float("nan")
    inv_a_in_sq = 1.0 / (a_in * a_in)
    inv_a_out_sq = 1.0 / (a_out * a_out)
    scale = (100.0 / flow_split_pct) ** 2
    return scale * inv_a_in_sq - inv_a_out_sq


def _speed_change_for_junction_row(
    cfg: dict[str, Any],
    junction_by_name: dict[str, dict],
    junction_name: str,
    primary_outlet: str,
    flow_split_pct: float,
) -> float:
    """Per-row speed_change using trunk inlet area and primary-outlet junction area."""
    a_in = _original_inlet_area_for_junction(cfg, junction_name)
    j = junction_by_name.get(junction_name)
    if j is None or a_in is None:
        return float("nan")
    outlet_areas = _safe_get(j, "geometric_params", "outlet_vessel_areas", default={}) or {}
    a_out = outlet_areas.get(primary_outlet)
    if a_out is None:
        return float("nan")
    return _speed_change_from_areas(flow_split_pct, a_in, float(a_out))


__all__ = [
    "COMPUTED_VESSEL_FEATURES",
    "compute_bifurcation_generation_by_vessel",
    "load_junction_geometric_features",
    "load_vessel_geometric_features",
    "load_vessel_targets_from_config",
]
