"""
Normalize geometric 0D JSON junction connectivity to inlet_blocks / outlet_blocks
(Phase A), validate topology, and persist geometric_input_named.json beside geometric_input.json.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

# Dict-valued keys under junction["geometric_params"] whose keys are vessel identifiers
# (either vessel_name or legacy stringified vessel_id).
_VESSEL_KEYED_GEOMETRIC_PARAM_SUBDICTS = frozenset(
    {
        "outlet_L",
        "outlet_R_poiseuille",
        "outlet_stenosis_coefficient",
        "outlet_vessel_areas",
        "outlet_path_lengths",
        "outlet_tangents",
        "outlet_tortuosities",
        "outlet_max_inscribed_radius",
        "max_inscribed_radius_min_on_path",
        "max_inscribed_radius_max_on_path",
        "outlet_angle_diffs",
        "outlet_path_gids",
        "inlet_vessel_areas",
        "outlet_centerline_paths",
    }
)


def geometric_input_named_json_path(geometric_input_path: str) -> str:
    """
    Target path for the name-wired snapshot next to the given geometric JSON.

    Uses ``geometric_input_named.json`` when the input basename is ``geometric_input.json``;
    otherwise ``<stem>_named.json`` to avoid clobbering different variants in one folder.
    """
    d = os.path.dirname(os.path.abspath(geometric_input_path))
    base = os.path.basename(geometric_input_path)
    if base == "geometric_input.json":
        return os.path.join(d, "geometric_input_named.json")
    stem, _ext = os.path.splitext(base)
    return os.path.join(d, f"{stem}_named.json")


def write_geometric_input_named_json(geometric_input_path: str, geometric_input: Dict[str, Any]) -> str:
    """Write ``geometric_input`` to the sibling named-json path; returns that path."""
    out = geometric_input_named_json_path(geometric_input_path)
    with open(out, "w") as f:
        json.dump(geometric_input, f, indent=4)
    return out


def _remap_vessel_keyed_dict_keys(
    d: Dict[str, Any], vessel_id_to_name: Dict[int, str]
) -> None:
    """In-place: replace string keys that are decimal vessel ids with vessel_name."""
    if not isinstance(d, dict):
        return
    for k in list(d.keys()):
        if not isinstance(k, str) or not k.isdigit():
            continue
        vid = int(k)
        if vid not in vessel_id_to_name:
            continue
        new_k = vessel_id_to_name[vid]
        if new_k == k:
            continue
        if new_k in d and d[new_k] is not d[k]:
            raise ValueError(
                f"geometric_params key remap conflict: both {k!r} and {new_k!r} present with different values"
            )
        d[new_k] = d.pop(k)


def _remap_junction_geometric_params_keys(
    junc: Dict[str, Any], vessel_id_to_name: Dict[int, str]
) -> None:
    gp = junc.get("geometric_params")
    if not isinstance(gp, dict):
        return
    for subkey, subval in gp.items():
        if subkey not in _VESSEL_KEYED_GEOMETRIC_PARAM_SUBDICTS:
            continue
        if isinstance(subval, dict):
            _remap_vessel_keyed_dict_keys(subval, vessel_id_to_name)
        if subkey == "outlet_centerline_paths" and isinstance(subval, dict):
            for _vname, inner in list(subval.items()):
                if isinstance(inner, dict):
                    for _ik, iv in list(inner.items()):
                        if isinstance(iv, dict):
                            _remap_vessel_keyed_dict_keys(iv, vessel_id_to_name)


def outlet_blocks_with_kind(
    junc: Dict[str, Any],
    vessel_name_to_id: Dict[str, int],
    junction_names: Set[str],
) -> List[Tuple[Literal["vessel", "junction"], str]]:
    """
    For each ``outlet_blocks`` entry, yield ``('vessel', name)`` or ``('junction', name)``.
    Unknown names are skipped.
    """
    out: List[Tuple[Literal["vessel", "junction"], str]] = []
    for name in junc.get("outlet_blocks") or []:
        s = str(name)
        if s in vessel_name_to_id:
            out.append(("vessel", s))
        elif s in junction_names:
            out.append(("junction", s))
    return out


def resolve_junction_vessel_ids(
    junc: Dict[str, Any],
    vessel_name_to_id: Dict[str, int],
    junction_names: Set[str],
) -> Tuple[List[int], List[int]]:
    """
    Return (inlet_vessel_ids, outlet_vessel_ids) for algorithms that still index vessels by id.

    If ``inlet_vessels`` / ``outlet_vessels`` are present, they are returned as-is.
    Otherwise ids are derived from ``inlet_blocks`` / ``outlet_blocks`` **only for vessel
    names**. If a block lists a junction name, this function raises (use
    ``outlet_blocks_with_kind`` or block-aware GID logic instead).
    """
    iv = junc.get("inlet_vessels") or []
    ov = junc.get("outlet_vessels") or []
    if iv and ov:
        return [int(x) for x in iv], [int(x) for x in ov]

    ib = junc.get("inlet_blocks") or []
    ob = junc.get("outlet_blocks") or []
    if not ib and not ov and not ob:
        return [], []
    if not ib or not ob:
        jn = junc.get("junction_name", "?")
        raise ValueError(
            f"Junction {jn}: incomplete block connectivity (need non-empty inlet_blocks and outlet_blocks)"
        )

    def _block_to_vid(name: str) -> int:
        if name in vessel_name_to_id:
            return vessel_name_to_id[name]
        if name in junction_names:
            raise ValueError(
                f"Junction {junc.get('junction_name')}: block {name!r} refers to another junction; "
                "junction-to-junction resolution is not implemented in resolve_junction_vessel_ids."
            )
        raise ValueError(
            f"Junction {junc.get('junction_name')}: unknown block name {name!r} "
            f"(not a vessel_name and not a junction_name)"
        )

    return [_block_to_vid(str(x)) for x in ib], [_block_to_vid(str(x)) for x in ob]


def convert_junctions_to_block_connectivity(geometric_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutate ``geometric_input`` in place: every junction gets inlet_blocks / outlet_blocks
    from vessel ids where needed; remaps numeric-string keys in geometric_params; removes
    inlet_vessels / outlet_vessels when block lists are authoritative.

    Idempotent if junctions are already block-only with valid names.
    """
    vessels = geometric_input.get("vessels", [])
    if not vessels:
        return geometric_input

    for i, v in enumerate(vessels):
        if not v.get("vessel_name"):
            raise ValueError(f"vessels[{i}] is missing vessel_name")
        if "vessel_id" not in v:
            raise ValueError(f"vessel {v.get('vessel_name')!r} is missing vessel_id")

    vessel_id_to_name: Dict[int, str] = {}
    vessel_name_to_id: Dict[str, int] = {}
    for v in vessels:
        vid = int(v["vessel_id"])
        vname = str(v["vessel_name"])
        vessel_id_to_name[vid] = vname
        vessel_name_to_id[vname] = vid

    junctions = geometric_input.get("junctions", [])
    junction_names = {j.get("junction_name") for j in junctions if j.get("junction_name")}

    for junc in junctions:
        jn = junc.get("junction_name")
        if not jn:
            raise ValueError("junction missing junction_name")

        _remap_junction_geometric_params_keys(junc, vessel_id_to_name)

        has_ids = bool(junc.get("inlet_vessels")) or bool(junc.get("outlet_vessels"))
        ib_existing = junc.get("inlet_blocks") or []
        ob_existing = junc.get("outlet_blocks") or []

        if has_ids:
            inlet_ids = [int(x) for x in (junc.get("inlet_vessels") or [])]
            outlet_ids = [int(x) for x in (junc.get("outlet_vessels") or [])]
            ib_from_ids = [vessel_id_to_name[i] for i in inlet_ids]
            ob_from_ids = [vessel_id_to_name[i] for i in outlet_ids]
            if ib_existing and [str(x) for x in ib_existing] != ib_from_ids:
                raise ValueError(
                    f"Junction {jn}: inlet_blocks {ib_existing!r} disagrees with inlet_vessels -> {ib_from_ids!r}"
                )
            if ob_existing and [str(x) for x in ob_existing] != ob_from_ids:
                raise ValueError(
                    f"Junction {jn}: outlet_blocks {ob_existing!r} disagrees with outlet_vessels -> {ob_from_ids!r}"
                )
            junc["inlet_blocks"] = ib_from_ids
            junc["outlet_blocks"] = ob_from_ids
            junc.pop("inlet_vessels", None)
            junc.pop("outlet_vessels", None)
        elif ib_existing and ob_existing:
            junc["inlet_blocks"] = [str(x) for x in ib_existing]
            junc["outlet_blocks"] = [str(x) for x in ob_existing]
        else:
            jt = junc.get("junction_type", "")
            if jt in ("BloodVesselJunction", "NORMAL_JUNCTION", "internal_junction"):
                raise ValueError(
                    f"Junction {jn} (type {jt!r}) has no inlet_vessels/outlet_vessels or inlet_blocks/outlet_blocks"
                )

    return geometric_input


def validate_block_connectivity(geometric_input: Dict[str, Any]) -> None:
    """Raise ValueError if any block name does not resolve to a vessel or allowed junction ref."""
    vessels = geometric_input.get("vessels", [])
    vessel_name_to_id = {str(v["vessel_name"]): int(v["vessel_id"]) for v in vessels if v.get("vessel_name")}
    junction_names = {
        str(j["junction_name"]) for j in geometric_input.get("junctions", []) if j.get("junction_name")
    }

    for junc in geometric_input.get("junctions", []):
        jn = junc.get("junction_name", "?")
        if junc.get("inlet_vessels") or junc.get("outlet_vessels"):
            raise ValueError(
                f"Junction {jn}: still has inlet_vessels/outlet_vessels after block normalization"
            )
        ib = junc.get("inlet_blocks") or []
        ob = junc.get("outlet_blocks") or []
        if not ib and not ob:
            continue
        for lst, label in ((ib, "inlet_blocks"), (ob, "outlet_blocks")):
            for name in lst:
                s = str(name)
                if s in vessel_name_to_id:
                    continue
                if s in junction_names:
                    continue
                raise ValueError(f"Junction {jn}: {label} references unknown block {s!r}")

        jt = junc.get("junction_type", "")
        if jt == "BloodVesselJunction" and len(ib) != 1:
            raise ValueError(
                f"Junction {jn}: BloodVesselJunction expects exactly one inlet block, got {len(ib)}"
            )


def ensure_block_connectivity_from_ids(
    geometric_input: Dict[str, Any], *, validate: bool = True
) -> Dict[str, Any]:
    """Convert junctions to block connectivity in place; optionally validate."""
    convert_junctions_to_block_connectivity(geometric_input)
    if validate:
        validate_block_connectivity(geometric_input)
    return geometric_input


def apply_phase_a_geometry_files(
    geometric_input: Dict[str, Any],
    geometric_input_path: str,
    *,
    rewrite_source: bool = True,
    validate: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Run Phase A on an in-memory config, write ``geometric_input_named.json`` (or
    ``<stem>_named.json`` for other input basenames), and optionally rewrite the source
    JSON path with the same normalized content.

    Returns ``(named_json_path, source_path_if_rewritten)``.
    """
    ensure_block_connectivity_from_ids(geometric_input, validate=validate)
    named_path = write_geometric_input_named_json(geometric_input_path, geometric_input)
    src = None
    if rewrite_source:
        with open(geometric_input_path, "w") as f:
            json.dump(geometric_input, f, indent=4)
        src = geometric_input_path
    return named_path, src
