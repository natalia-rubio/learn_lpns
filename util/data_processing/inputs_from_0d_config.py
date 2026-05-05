#!/usr/bin/env python3
"""
Utilities for extracting ML-ready inputs from 0D configuration files.

Current focus: build a junction-level feature matrix from the geometric
parameters written by `util.zerod_calibration.geometric_params`.

Each row in the returned array corresponds to one junction instance
(after bifurcation_splitting, i.e. each with exactly two outlets).
Each column is a scalar geometric feature derived from `geometric_params`.

Includes **generation**: count of junctions with exactly two outlets (two-outlet
bifurcations) along the path from the root inlet branch to the junction's inlet
vessel (0 at the inlet branch). Vessel-level features use the same count to the
vessel itself (including the bifurcation that feeds that vessel).
"""

import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np

from util.zerod_calibration.bifurcation_splitting import (
    junction_outlet_count,
    junction_uses_block_connectivity,
)
from util.zerod_calibration.post_processing import read_zerod_csv


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
    ("radius_ratio",
     lambda out, junc: _safe_div(out["r_local"], junc["inlet_max_r"])),
     ("poiseuille_resistance_calc",
     lambda out, junc: _safe_div(8 * 0.04 * out["path_length"], np.pi * out["r_local"]**4)),
     ("inductance_calc",   
     lambda out, junc: _safe_mult(1.06*out["path_length"], out["r_local"]**2)),
     ("rneg4",
     lambda out, junc: out["r_local"] ** -4),
     ("rneg2",
     lambda out, junc: out["r_local"] ** -2),
     ("rmin_rat",
     lambda out, junc: _safe_div(out["r_min_path"], out["r_local"])),
     ("rmax_rat",
     lambda out, junc: _safe_div(out["r_max_path"], out["r_local"])),
     ("nd_length",
     lambda out, junc: out["path_length"]/out["r_local"]),
]

# ---------------------------------------------------------------------------
# Computed per-vessel features (for vessel NN input)
# ---------------------------------------------------------------------------
# Each entry is (name, func) where func(vessel_raw) returns a float or None.
# vessel_raw has: inlet_area, outlet_area, path_length, area_ratio, r_local (sqrt(inlet_area/pi)), inlet_max_r (inlet_max_inscribed_radius).
# To add a new computed vessel feature, append a tuple here.
# ---------------------------------------------------------------------------
COMPUTED_VESSEL_FEATURES: List[Tuple[str, Any]] = [
    ("radius_ratio", lambda d: _safe_div(d["outlet_max_r"], d["inlet_max_r"])),
    ("rmin_rat", lambda d: _safe_div(d["max_inscribed_radius_min"], d["outlet_max_r"])),
    ("rmax_rat", lambda d: _safe_div(d["max_inscribed_radius_max"], d["outlet_max_r"])),
    ("nd_length", lambda d: _safe_div(d["path_length"], d["outlet_max_r"])),
    ("poiseuille_resistance_calc", lambda d: _safe_div(8.0 * 0.04 * d["path_length"], np.pi * (d["r_local"] ** 4))),
    ("inductance_calc", lambda d: _safe_div(1.06 * d["path_length"], d["inlet_area"])),
    ("stenosis_calc", lambda d: max(0.0, 1.0 - d["area_ratio"]) if d.get("inlet_area", 0) > 0 else 0.0),
    ("rneg4", lambda d: d["outlet_max_r"] ** -4),
    ("rneg2", lambda d: d["outlet_max_r"] ** -2),
]


def _find_root_vessel_id_for_generation(cfg: Dict[str, Any]):
    """
    Vessel_id of the tree root (inlet branch). Matches visualize_centerline_branches
    logic: prefer vessel with inlet boundary condition, then name containing branch0,
    else minimum vessel_id.
    """
    vessels = cfg.get("vessels", [])
    found = None
    for v in vessels:
        vid = v.get("vessel_id")
        if vid is None:
            continue
        bc = v.get("boundary_conditions")
        if isinstance(bc, dict) and "inlet" in bc:
            found = vid
            break
    if found is not None:
        return found
    for v in vessels:
        name = v.get("vessel_name", "") or ""
        if "branch0" in name:
            vid = v.get("vessel_id")
            if vid is not None:
                return vid
    ids = [v.get("vessel_id") for v in vessels if v.get("vessel_id") is not None]
    return min(ids) if ids else None


def _norm_vid(x: Any) -> Any:
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def compute_bifurcation_generation_by_vessel(cfg: Dict[str, Any]) -> Dict[Any, float]:
    """
    Map vessel_id -> generation: number of 2-outlet junctions along the path from
    the root inlet vessel to this vessel. The root vessel has generation 0.

    Counts a +1 each time the path crosses a junction with exactly two outlets
    (``outlet_vessels`` or ``outlet_blocks``). Cascaded ``BloodVesselJunction`` links
    (``inlet_blocks`` / ``outlet_blocks``) are traversed recursively.

    Vessels not reachable from the root are omitted from the map.
    """
    from collections import deque

    vessels = cfg.get("vessels", []) or []
    junctions = cfg.get("junctions", []) or []
    vessel_name_to_id: Dict[str, Any] = {}
    vessel_id_to_name: Dict[Any, str] = {}
    for v in vessels:
        vid = v.get("vessel_id")
        vname = v.get("vessel_name", "") or ""
        if vid is not None and vname:
            vessel_name_to_id[vname] = vid
            vessel_id_to_name[vid] = vname

    junc_names = {j.get("junction_name", "") for j in junctions if j.get("junction_name")}
    junc_by_name = {j.get("junction_name", ""): j for j in junctions if j.get("junction_name")}

    def _outlet_block_names(j: Dict[str, Any]) -> List[str]:
        ob = j.get("outlet_blocks")
        if ob:
            return list(ob)
        names: List[str] = []
        for vid in j.get("outlet_vessels") or []:
            vn = vessel_id_to_name.get(vid, "")
            if vn:
                names.append(vn)
        return names

    def _vessel_feeds_junction(vid: Any, j: Dict[str, Any]) -> bool:
        nv = _norm_vid(vid)
        if junction_uses_block_connectivity(j):
            for blk in j.get("inlet_blocks") or []:
                if blk in vessel_name_to_id and _norm_vid(vessel_name_to_id[blk]) == nv:
                    return True
            return False
        for iv in j.get("inlet_vessels") or []:
            if _norm_vid(iv) == nv:
                return True
        return False

    def _expand_junction_outputs(j_name: str, g_in: float, path: frozenset) -> List[Tuple[Any, float]]:
        """Return (vessel_id, generation) for every vessel outlet reachable from junction ``j_name``."""
        if not j_name or j_name not in junc_by_name or j_name in path:
            return []
        J = junc_by_name[j_name]
        nout = junction_outlet_count(J)
        inc = 1.0 if nout == 2 else 0.0
        g2 = g_in + inc
        path2 = path | {j_name}
        res: List[Tuple[Any, float]] = []
        for blk in _outlet_block_names(J):
            if blk in junc_names:
                res.extend(_expand_junction_outputs(blk, g2, path2))
            elif blk in vessel_name_to_id:
                oid = _norm_vid(vessel_name_to_id[blk])
                res.append((oid, g2))
        return res

    root = _find_root_vessel_id_for_generation(cfg)
    if root is None:
        return {}
    root = _norm_vid(root)
    gen: Dict[Any, float] = {root: 0.0}
    q = deque([root])
    while q:
        vid = q.popleft()
        nv = _norm_vid(vid)
        g_here = float(gen.get(nv, 0.0))
        for j in junctions:
            if not _vessel_feeds_junction(nv, j):
                continue
            nout = junction_outlet_count(j)
            inc = 1.0 if nout == 2 else 0.0
            g_next = g_here + inc
            for blk in _outlet_block_names(j):
                if blk in junc_names:
                    for oid, gdown in _expand_junction_outputs(blk, g_next, frozenset()):
                        oid = _norm_vid(oid)
                        if oid not in gen:
                            gen[oid] = float(gdown)
                            q.append(oid)
                        else:
                            gen[oid] = min(gen[oid], float(gdown))
                elif blk in vessel_name_to_id:
                    oid = _norm_vid(vessel_name_to_id[blk])
                    if oid not in gen:
                        gen[oid] = float(g_next)
                        q.append(oid)
                    else:
                        gen[oid] = min(gen[oid], float(g_next))

    out: Dict[Any, float] = {}
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

    gen_by_vessel = compute_bifurcation_generation_by_vessel(cfg)

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

    junc_names = {jn.get("junction_name", "") for jn in junctions if jn.get("junction_name")}

    # We assume bifurcation geometry: each junction has exactly 1 inlet and 2 outlets.
    for j in junctions:
        j_name = j.get("junction_name", "")
        gp = j.get("geometric_params", {})

        outlet_vessels = j.get("outlet_vessels", []) or []
        outlet_blocks = j.get("outlet_blocks") or []

        if require_two_outlets and junction_outlet_count(j) != 2:
            # Skip non-bifurcation junctions in this mode
            if verbose:
                print(f"Skipping junction {j_name}: not a bifurcation junction")
            continue
        if verbose:
            print(f"Processing junction {j_name}: outlet_vessels={outlet_vessels} outlet_blocks={outlet_blocks}")

        inlet_ids_list = j.get("inlet_vessels", []) or []
        inlet_blocks_list = j.get("inlet_blocks") or []
        inlet_vid0 = inlet_ids_list[0] if inlet_ids_list else None
        inlet_key = inlet_vid0
        if inlet_vid0 is not None:
            try:
                inlet_key = int(inlet_vid0)
            except (TypeError, ValueError):
                inlet_key = inlet_vid0
        elif inlet_blocks_list:
            blk0 = inlet_blocks_list[0]
            if blk0 in vessel_name_to_id:
                inlet_key = _norm_vid(vessel_name_to_id[blk0])
            else:
                inlet_key = None
        generation_val = float(gen_by_vessel[inlet_key]) if inlet_key is not None and inlet_key in gen_by_vessel else float("nan")

        # Build authoritative outlet_name -> vessel_id mapping for this junction
        # (junction-trunk outlets use sentinel id -1.0 for the ML outlet_vessel_id column)
        outlet_vessel_id_map: Dict[str, Any] = {}
        for vid in outlet_vessels:
            vname = vessel_id_to_name.get(vid, "")
            if vname:
                outlet_vessel_id_map[vname] = vid
        for blk in outlet_blocks:
            if blk in vessel_name_to_id:
                outlet_vessel_id_map[blk] = vessel_name_to_id[blk]
            elif blk in junc_names:
                outlet_vessel_id_map[blk] = -1

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
            if outlet_blocks:
                outlet_names = list(outlet_blocks)
            else:
                for vid in outlet_vessels:
                    vname = vessel_id_to_name.get(vid, "")
                    if vname:
                        outlet_names.append(vname)
                    elif isinstance(vid, int) and 0 <= vid < len(vessels):
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
                f"Junction {j_name}: outlet '{outlet0_name}' not in outlet_vessel_id_map {outlet_vessel_id_map}. "
                f"outlet_vessels={outlet_vessels} outlet_blocks={outlet_blocks}"
            )
        if outlet1_vid is None:
            raise ValueError(
                f"Junction {j_name}: outlet '{outlet1_name}' not in outlet_vessel_id_map {outlet_vessel_id_map}. "
                f"outlet_vessels={outlet_vessels} outlet_blocks={outlet_blocks}"
            )
        
        if verbose:
            print(f"  outlet0: {outlet0_name} (vessel_id={outlet0_vid}), outlet1: {outlet1_name} (vessel_id={outlet1_vid})")

        # Row 1: inlet + outlet0 + outlet1
        # Only emit rows whose primary outlet is a real vessel id. Split-phase
        # J-J trunk outlets are represented as sentinel -1 and should not become
        # supervised rows (no matching calibration target row exists).
        outlet0_is_real_vessel = (outlet0_vid is not None) and (float(outlet0_vid) >= 0.0)
        if outlet0_is_real_vessel and ('connector' not in outlet0_name or 'connectorEL' in outlet0_name):
            feat_row_0_first: List[float] = [outlet0_vid]
            if verbose:
                print(f"Adding outlet 0 features: {outlet0_name}, outlet vessel id: {outlet0_vid}")
                print(f"Adding inlet max inscribed radius: {inlet_max_r}")
            feat_row_0_first.append(_to_float(inlet_max_r))
            feat_row_0_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_0_first.append(generation_val if generation_val == generation_val else 0.0)
            feat_row_0_first.extend(get_outlet_features(outlet0_name))
            feat_row_0_first.extend(get_outlet_features(outlet1_name))
            rows.append(feat_row_0_first)
            junction_names.append(j_name)
            outlet_primary_names.append(outlet0_name)

        # Row 2: inlet + outlet1 + outlet0 (swapped)
        # Skip swapped sample if outlet1 would be a connector OR a J-J trunk
        # (connector-as-primary exclusion + non-vessel-primary exclusion).
        outlet1_is_real_vessel = (outlet1_vid is not None) and (float(outlet1_vid) >= 0.0)
        if outlet1_is_real_vessel and ('connector' not in outlet1_name or 'connectorEL' in outlet1_name):
            if verbose:
                print(f"Adding outlet 1 features: {outlet1_name}, outlet vessel id: {outlet1_vid}")
            feat_row_1_first: List[float] = [outlet1_vid]
            feat_row_1_first.append(_to_float(inlet_max_r))
            feat_row_1_first.extend(_to_float(c) for c in inlet_tangent)
            feat_row_1_first.append(generation_val if generation_val == generation_val else 0.0)
            feat_row_1_first.extend(get_outlet_features(outlet1_name))
            feat_row_1_first.extend(get_outlet_features(outlet0_name))
            rows.append(feat_row_1_first)
            junction_names.append(j_name)  # Same junction name for both rows
            outlet_primary_names.append(outlet1_name)
        else:
            if verbose:
                reason = "connector" if ("connector" in outlet1_name and "connectorEL" not in outlet1_name) else "non-vessel outlet block"
                print(
                    f"Skipping swapped sample for junction {j_name}: "
                    f"primary outlet would be {reason} {outlet1_name}"
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
        "generation",
    ]
    for prefix in ("outlet0", "outlet1"):
        for suffix in _all_outlet_suffixes:
            feature_names.append(f"{prefix}_{suffix}")
    return X, feature_names, junction_names, outlet_primary_names


def _resolve_original_inlet_per_junction(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    For each junction (with two outlets), resolve the carrier inlet vessel or upstream
    junction: trace ``inlet_blocks`` / upstream junction hops along J–J trunks introduced
    when multi-outlet junctions were cascaded split (no synthetic ``_*_connector{N}`` vessels).

    Returns:
        Dict mapping junction_name -> original_inlet_vessel_name.
    """
    vessels = cfg.get("vessels", [])
    junctions = cfg.get("junctions", [])
    vessel_name_to_id = {
        v.get("vessel_name", ""): v.get("vessel_id")
        for v in vessels
        if v.get("vessel_name") and v.get("vessel_id") is not None
    }
    vessel_id_to_name = {
        v.get("vessel_id"): v.get("vessel_name", "")
        for v in vessels
        if v.get("vessel_id") is not None
    }
    junc_names = {j.get("junction_name", "") for j in junctions if j.get("junction_name")}
    junc_by_name = {j.get("junction_name", ""): j for j in junctions if j.get("junction_name")}

    # vessel_id -> junction that has this vessel as an outlet (for tracing back)
    outlet_vessel_id_to_junction: Dict[Any, str] = {}
    for j in junctions:
        j_name = j.get("junction_name", "")
        for vid in j.get("outlet_vessels", []) or []:
            outlet_vessel_id_to_junction[_norm_vid(vid)] = j_name
        for blk in j.get("outlet_blocks") or []:
            if blk in vessel_name_to_id:
                outlet_vessel_id_to_junction[_norm_vid(vessel_name_to_id[blk])] = j_name

    def _first_inlet_token(j: Dict[str, Any]) -> Tuple[Any, Any]:
        """Return ('id', vessel_id) or ('junction', junc_name) for the primary inlet."""
        iv = j.get("inlet_vessels") or []
        if iv:
            return "id", _norm_vid(iv[0])
        ib = j.get("inlet_blocks") or []
        if not ib:
            return None, None
        blk = ib[0]
        if blk in vessel_name_to_id:
            return "id", _norm_vid(vessel_name_to_id[blk])
        if blk in junc_names:
            return "junction", blk
        return None, None

    junction_to_inlet_token: Dict[str, Tuple[Any, Any]] = {}
    for j in junctions:
        jn = j.get("junction_name", "")
        if jn:
            junction_to_inlet_token[jn] = _first_inlet_token(j)

    out: Dict[str, str] = {}
    for j in junctions:
        j_name = j.get("junction_name", "")
        if not j_name or junction_outlet_count(j) != 2:
            continue
        kind, cur = junction_to_inlet_token.get(j_name, (None, None))
        if cur is None:
            continue
        while True:
            if kind == "junction":
                prev_j = junc_by_name.get(str(cur))
                if not prev_j:
                    break
                kind, cur = _first_inlet_token(prev_j)
                if cur is None:
                    break
                continue
            # kind == "id": bifurcation inlet vessel (no legacy split *_connector{N} hops)
            break
        if kind == "id":
            out[j_name] = vessel_id_to_name.get(cur, "")
        elif kind == "junction":
            out[j_name] = str(cur)
    return out


def compute_junction_flow_splits(
    config_path: str,
    geometric_results_csv_path: str,
    require_two_outlets: bool = True,
) -> Dict[str, Tuple[float, float]]:
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
    out: Dict[str, Tuple[Tuple[str, str], Tuple[float, float]]] = {}
    if not os.path.exists(geometric_results_csv_path):
        return out

    with open(config_path, "r") as f:
        cfg = json.load(f)

    vessels = cfg.get("vessels", [])
    vessel_id_to_name = {
        v.get("vessel_id"): v.get("vessel_name", "")
        for v in vessels
        if v.get("vessel_id") is not None
    }
    original_inlet_by_junction = _resolve_original_inlet_per_junction(cfg)

    results, times = read_zerod_csv(geometric_results_csv_path)
    if not times:
        return out

    junc_names = {jn.get("junction_name", "") for jn in cfg.get("junctions", []) if jn.get("junction_name")}
    junc_by_name = {
        jn.get("junction_name", ""): jn
        for jn in cfg.get("junctions", [])
        if jn.get("junction_name")
    }
    vessel_name_to_id = {
        v.get("vessel_name", ""): v.get("vessel_id")
        for v in vessels
        if v.get("vessel_name") and v.get("vessel_id") is not None
    }

    def _vessel_name_for_flow_series(block_name: str, memo: Any) -> str:
        """Map an outlet block (vessel or downstream junction) to a vessel name present in CSV results."""
        if block_name in results:
            return block_name
        if block_name in vessel_name_to_id:
            return block_name
        if block_name not in junc_names:
            return ""
        if block_name in memo:
            return ""
        memo.add(block_name)
        jn = junc_by_name.get(block_name)
        if not jn:
            return ""
        for blk in jn.get("outlet_blocks") or []:
            vn = _vessel_name_for_flow_series(blk, memo)
            if vn:
                return vn
        for vid in jn.get("outlet_vessels") or []:
            nm = vessel_id_to_name.get(vid, "")
            if nm and nm in results:
                return nm
        return ""

    for j in cfg.get("junctions", []):
        j_name = j.get("junction_name", "")
        outlet_vessels = j.get("outlet_vessels", []) or []
        outlet_blocks = j.get("outlet_blocks") or []
        inlet_vessels = j.get("inlet_vessels", []) or []
        inlet_blocks = j.get("inlet_blocks") or []

        if require_two_outlets and junction_outlet_count(j) != 2:
            continue
        if not inlet_vessels and not inlet_blocks:
            raise ValueError(
                f"Junction {j_name!r} has no inlet_vessels or inlet_blocks; cannot compute flow split."
            )

        # Use original inlet (trace back through connectors) for denominator
        original_inlet_name = original_inlet_by_junction.get(j_name, "")
        if not original_inlet_name:
            if inlet_vessels:
                original_inlet_name = vessel_id_to_name.get(inlet_vessels[0], "")
            elif inlet_blocks:
                ib0 = inlet_blocks[0]
                if ib0 in vessel_name_to_id:
                    original_inlet_name = ib0
                elif ib0 in results:
                    original_inlet_name = ib0

        if outlet_blocks:
            memo0, memo1 = set(), set()
            out0_name = _vessel_name_for_flow_series(outlet_blocks[0], memo0)
            out1_name = _vessel_name_for_flow_series(outlet_blocks[1], memo1)
        else:
            out0_name = vessel_id_to_name.get(outlet_vessels[0], "")
            out1_name = vessel_id_to_name.get(outlet_vessels[1], "")

        if not original_inlet_name or not out0_name or not out1_name:
            # Block-wired EL/J-J cases can legitimately lack a direct vessel
            # series for one side. Keep junction present with NaN split so
            # downstream feature extraction can continue.
            out[j_name] = ((out0_name, out1_name), (float("nan"), float("nan")))
            continue
        if original_inlet_name not in results or out0_name not in results or out1_name not in results:
            # Same graceful fallback for missing geometric-results series.
            out[j_name] = ((out0_name, out1_name), (float("nan"), float("nan")))
            continue

        ratios0: List[float] = []
        ratios1: List[float] = []
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
            fs0, fs1 = float("nan"), float("nan")
        else:
            fs0 = float(np.mean(ratios0)) * 100.0
            fs1 = float(np.mean(ratios1)) * 100.0
        out[j_name] = ((out0_name, out1_name), (fs0, fs1))

    return out


def load_vessel_geometric_features(
    config_path: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
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
    with open(config_path, "r") as f:
        cfg = json.load(f)

    gen_by_vessel = compute_bifurcation_generation_by_vessel(cfg)

    vessels = cfg.get("vessels", [])
    if not isinstance(vessels, list):
        raise ValueError("Expected 'vessels' to be a list in config.")

    # Feature column names (order must match row construction below)
    # vessel_id, is_inlet, generation, base geometric params + all COMPUTED_VESSEL_FEATURES + zero_d_element_values from config
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
    ] + _computed_vessel_suffixes + [
        "R_poiseuille_geometric",
        "L_geometric",
        "stenosis_coefficient_geometric",
    ]

    rows: List[List[float]] = []
    vessel_ids: List[int] = []
    vessel_names_out: List[str] = []

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
        gnum = float(gen_by_vessel[vk]) if vk in gen_by_vessel else float("nan")
        if gnum != gnum:
            gnum = 0.0
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
        ] + computed_vals + [
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
) -> Tuple[List[int], List[str], np.ndarray]:
    """
    Load vessel targets (R_poiseuille, stenosis_coefficient, L) from a calibrated
    0D config JSON. Only non-connector vessels are included; order matches
    config vessel order.

    Returns:
        vessel_ids: List of vessel_id.
        vessel_names: List of vessel_name.
        targets: Array of shape (n_vessels, 3) with columns [R_poiseuille, stenosis_coefficient, L].
    """
    with open(calibrated_config_path, "r") as f:
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


__all__ = [
    "COMPUTED_VESSEL_FEATURES",
    "compute_bifurcation_generation_by_vessel",
    "load_junction_geometric_features",
    "load_vessel_geometric_features",
    "load_vessel_targets_from_config",
]


