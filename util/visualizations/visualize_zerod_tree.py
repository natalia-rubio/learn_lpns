#!/usr/bin/env python3
"""
Plot a 2D schematic of the 0D lumped-parameter network from a zerod JSON.

Supports topology from either:

- Legacy ``inlet_vessels`` / ``outlet_vessels`` (integer ids), or
- Block wiring ``inlet_blocks`` / ``outlet_blocks`` (vessel / junction names),

including junction-to-junction (J–J) trunk links with no intervening vessel segment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.color": "black",
        "axes.labelcolor": "black",
    }
)

# Layout distances in data units (boxed vessel labels need generous separation).
# Vertical: distance between parent row and child row (y decreases with depth).
LEVEL_HEIGHT_DEFAULT = 23.0
# Horizontal: sibling subtree spread (orthogonal to LEVEL_HEIGHT).
LEAF_SPACE_DEFAULT = 26.5
HORIZONTAL_LAYOUT_BOOST = 1.35

FONT_CONFIG = {
    "vessel_label": 18,
    "node_label": 17,
    "title": 16,
    "axis_label": 13,
    "legend": 11,
}

ELEMENT_COLORS = {
    "inlet_bc": {"edge": "slategray", "linewidth": 3},
    "outlet_bc": {"edge": "slategray", "linewidth": 3},
    "junction": {"edge": "lightcoral", "linewidth": 2.5},
    "bifurcation_junction": {"edge": "crimson", "linewidth": 2.5},
    "junction_trunk": {"edge": "darkolivegreen", "linewidth": 2.8},
    "vessel": {"edge": "lightskyblue", "linewidth": 2},
    "connector_vessel": {"edge": "royalblue", "linewidth": 3},
    "line": {"color": "black", "linewidth": 2.5},
}

_JJ_DUMMY_VESSEL_ID = -1


def _norm_vid(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return -999999999
    return -999999999


def _uses_nonempty_vessel_lists(junc: Dict[str, Any]) -> bool:
    ov = junc.get("outlet_vessels")
    return isinstance(ov, list) and len(ov) > 0


def build_junction_graph_unified(cfg: Dict[str, Any]) -> Tuple[
    Dict[str, Dict[str, Any]],
    List[Tuple[str, str, str, int]],
    Optional[str],
]:
    """
    BC + junction nodes; edges = vessel segments plus J–J trunks (vessel_id==-1).
    """
    vessels = cfg.get("vessels", [])
    junctions = cfg.get("junctions", [])

    vessel_name_to_id: Dict[str, int] = {}
    vessel_by_id: Dict[int, Dict[str, Any]] = {}
    for v in vessels:
        vid = _norm_vid(v.get("vessel_id"))
        if vid <= -100000000:
            continue
        name = v.get("vessel_name", "")
        if name:
            vessel_name_to_id[name] = vid
            vessel_by_id[vid] = v

    junction_names: Set[str] = {
        j.get("junction_name", "") for j in junctions if j.get("junction_name")
    }
    junction_names.discard("")
    junction_name_to_node_id: Dict[str, str] = {}

    nodes: Dict[str, Dict[str, Any]] = {}
    for j in junctions:
        jnm = j.get("junction_name", "")
        if not jnm:
            continue
        nid = f"junc_{jnm}"
        junction_name_to_node_id[jnm] = nid
        nodes[nid] = {"name": jnm, "type": "junction"}

    inlet_bc_node: Optional[str] = None
    outlet_bc_nodes: Dict[int, str] = {}

    for v in vessels:
        vid = _norm_vid(v.get("vessel_id"))
        if vid <= -100000000:
            continue
        bcs = v.get("boundary_conditions") or {}
        if "inlet" in bcs:
            bc = bcs["inlet"]
            nid = f"bc_{bc}"
            nodes[nid] = {"name": bc, "type": "inlet_bc", "vessel_id": vid}
            inlet_bc_node = nid
        if "outlet" in bcs:
            bc = bcs["outlet"]
            nid = f"bc_{bc}"
            nodes[nid] = {"name": bc, "type": "outlet_bc", "vessel_id": vid}
            outlet_bc_nodes[vid] = nid

    vessel_downstream: Dict[int, str] = {}
    vessel_upstream: Dict[int, str] = {}
    jj_edges: Set[Tuple[str, str]] = set()

    for junc in junctions:
        jname = junc.get("junction_name", "")
        if not jname or jname not in junction_name_to_node_id:
            continue
        cur = junction_name_to_node_id[jname]

        if _uses_nonempty_vessel_lists(junc):
            inlet_list = [_norm_vid(x) for x in (junc.get("inlet_vessels") or [])]
            outlet_list = [_norm_vid(x) for x in (junc.get("outlet_vessels") or [])]
            inlet_list = [x for x in inlet_list if x >= 0]

            for inv in inlet_list:
                vessel_downstream[inv] = cur
            for outv in outlet_list:
                if outv >= 0:
                    vessel_upstream[outv] = cur
            continue

        ib = [str(x) for x in (junc.get("inlet_blocks") or []) if str(x).strip()]
        ob = [str(x) for x in (junc.get("outlet_blocks") or []) if str(x).strip()]

        for tok in ib:
            if tok in junction_names:
                pn = junction_name_to_node_id.get(tok)
                if pn:
                    jj_edges.add((pn, cur))
            elif tok in vessel_name_to_id:
                vessel_downstream[vessel_name_to_id[tok]] = cur

        for tok in ob:
            if tok in junction_names:
                dn = junction_name_to_node_id.get(tok)
                if dn:
                    jj_edges.add((cur, dn))
            elif tok in vessel_name_to_id:
                vessel_upstream[vessel_name_to_id[tok]] = cur

    edges: List[Tuple[str, str, str, int]] = []

    for v in vessels:
        vid = _norm_vid(v.get("vessel_id"))
        if vid <= -100000000:
            continue
        vname = v.get("vessel_name", f"vessel_{vid}")

        if vid in vessel_upstream:
            from_n = vessel_upstream[vid]
        elif "boundary_conditions" in v and "inlet" in v["boundary_conditions"]:
            from_n = inlet_bc_node
        else:
            from_n = f"implicit_inlet_{vid}"
            nodes[from_n] = {"name": "INLET?", "type": "inlet_bc", "vessel_id": vid}
            if inlet_bc_node is None:
                inlet_bc_node = from_n

        if vid in vessel_downstream:
            to_n = vessel_downstream[vid]
        elif vid in outlet_bc_nodes:
            to_n = outlet_bc_nodes[vid]
        else:
            to_n = f"implicit_outlet_{vid}"
            nodes[to_n] = {"name": f"OUT_{vid}", "type": "outlet_bc", "vessel_id": vid}

        edges.append((from_n, to_n, vname, vid))

    for a, b in sorted(jj_edges):
        la = nodes[a]["name"]
        lb = nodes[b]["name"]
        edges.append((a, b, f"{la} → {lb}", _JJ_DUMMY_VESSEL_ID))

    return nodes, edges, inlet_bc_node


def compute_junction_layout(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Tuple[str, str, str, int]],
    root_node: Optional[str],
    *,
    level_h: float = LEVEL_HEIGHT_DEFAULT,
    leaf_space: float = LEAF_SPACE_DEFAULT,
) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    if root_node is None:
        return positions

    # One edge per upstream→downstream (avoid duplicate children from parallel edges).
    children = defaultdict(list)
    seen_ft: Set[Tuple[str, str]] = set()
    for fr, to, _lbl, _vid in edges:
        if (fr, to) in seen_ft:
            continue
        seen_ft.add((fr, to))
        children[fr].append(to)

    queue = deque([root_node])
    levels = {root_node: 0}
    seen = set()
    max_lv = 0
    while queue:
        nid = queue.popleft()
        seen.add(nid)
        lv = levels[nid]
        max_lv = max(max_lv, lv)
        for cid in children.get(nid, []):
            if cid not in levels:
                levels[cid] = lv + 1
                queue.append(cid)

    for nid in nodes:
        if nid not in levels:
            levels[nid] = max_lv + 1
    max_lv = max(levels.values()) if levels else 0

    def leaf_count(nid):
        cs = children.get(nid, [])
        if not cs:
            return 1
        return sum(leaf_count(c) for c in cs)

    leaves = {n: leaf_count(n) for n in nodes}

    def assign(nid, xc, yp):
        positions[nid] = (xc, yp)
        cs = children.get(nid, [])
        if not cs:
            return
        yn = yp - level_h
        if len(cs) == 1:
            assign(cs[0], xc, yn)
            return
        tw = sum(leaves[c] for c in cs) * leaf_space
        x0 = xc - tw / 2
        for c in cs:
            w = leaves[c] * leaf_space
            cx = x0 + w / 2
            assign(c, cx, yn)
            x0 += w

    assign(root_node, 0.0, 0.0)
    return positions


def visualize_zerod_tree(
    json_path: str,
    output_path: str,
    title_suffix: Optional[str] = None,
    large_graph: bool = False,
    *,
    spacing_scale: float = 1.0,
) -> bool:
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    nodes, edges, root = build_junction_graph_unified(cfg)
    print(f"Reading {json_path}")
    print(f"  Nodes: {len(nodes)}, edges: {len(edges)}")

    ss = float(max(0.25, spacing_scale)) * (1.42 if large_graph else 1.0)
    lh = LEVEL_HEIGHT_DEFAULT * ss
    ls = LEAF_SPACE_DEFAULT * ss * HORIZONTAL_LAYOUT_BOOST
    pos = compute_junction_layout(
        nodes,
        edges,
        root,
        level_h=lh,
        leaf_space=ls,
    )
    if not pos:
        print("Error: no layout (missing inlet BC root?)")
        return False

    # Taller figure: wide trees need width; height keeps rows readable (avoid “zoomed out” look).
    fw, fh = (36, 30) if large_graph else (34, 26)
    fig, ax = plt.subplots(figsize=(fw, fh), layout="constrained")
    lc = ELEMENT_COLORS["line"]
    tc = ELEMENT_COLORS["junction_trunk"]

    for fr, to, lbl, ves_id in edges:
        if fr not in pos or to not in pos:
            continue
        p0, p1 = pos[fr], pos[to]
        is_jj = ves_id == _JJ_DUMMY_VESSEL_ID

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            "--" if is_jj else "-",
            color=tc["edge"] if is_jj else lc["color"],
            linewidth=tc["linewidth"] if is_jj else lc["linewidth"],
            alpha=0.85,
            zorder=1,
        )

        # J–J trunk: connector only — no boxed label (junction names are on nodes).
        if is_jj:
            continue

        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        h = float(np.hypot(dx, dy)) or 1.0
        # Small perpendicular nudge; scale with layout units (not raw ss).
        off = 0.11 * min(lh, ls)
        ox, oy = -dy / h * off, dx / h * off
        dl = lbl if len(lbl) < 72 else (lbl[:69] + "…")

        vcfg = (
            ELEMENT_COLORS["connector_vessel"]
            if "_connector" in lbl and "connectorEL" not in lbl
            else ELEMENT_COLORS["vessel"]
        )

        ax.text(
            mx + ox,
            my + oy,
            dl,
            fontsize=FONT_CONFIG["vessel_label"],
            ha="center",
            va="center",
            color="black",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor="white",
                alpha=0.94,
                edgecolor=vcfg["edge"],
                linewidth=vcfg.get("linewidth", 2),
            ),
        )

    for nid, xy in pos.items():
        ni = nodes.get(nid)
        if ni is None:
            continue
        nm, tp = ni["name"], ni["type"]
        if tp == "inlet_bc":
            cfg = ELEMENT_COLORS["inlet_bc"]
        elif tp == "outlet_bc":
            cfg = ELEMENT_COLORS["outlet_bc"]
        elif "_bif" in nm:
            cfg = ELEMENT_COLORS["bifurcation_junction"]
        else:
            cfg = ELEMENT_COLORS["junction"]

        ax.text(
            xy[0],
            xy[1],
            nm,
            fontsize=FONT_CONFIG["node_label"],
            ha="center",
            va="center",
            weight="bold",
            zorder=4,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor="white",
                alpha=0.94,
                edgecolor=cfg["edge"],
                linewidth=cfg["linewidth"],
            ),
        )

    ttl = f"0D network — {title_suffix}" if title_suffix else "0D network topology"
    ax.set_title(ttl, fontsize=FONT_CONFIG["title"], weight="bold")
    ax.set_xlabel("layout x", fontsize=FONT_CONFIG["axis_label"])
    ax.set_ylabel("layout depth", fontsize=FONT_CONFIG["axis_label"])
    ax.set_xticks([])
    ax.set_yticks([])

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    # Padding: avoid huge symmetric margins (they look “zoomed out”). Extra x for long vessel names.
    frac = 0.035
    ax.set_xlim(xmin - frac * dx - 1.65 * ls, xmax + frac * dx + 1.65 * ls)
    ax.set_ylim(ymin - frac * dy - 1.15 * lh, ymax + frac * dy + 1.15 * lh)
    # Independent x/y pixel scaling so wide graphs don’t squash vertical spacing (unlike aspect="equal").
    ax.set_aspect("auto")

    leg = [
        mpatches.Patch(
            facecolor="white",
            edgecolor=ELEMENT_COLORS["inlet_bc"]["edge"],
            linewidth=ELEMENT_COLORS["inlet_bc"]["linewidth"],
            label="Inlet BC",
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor=ELEMENT_COLORS["outlet_bc"]["edge"],
            linewidth=ELEMENT_COLORS["outlet_bc"]["linewidth"],
            label="Outlet BC",
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor=ELEMENT_COLORS["junction"]["edge"],
            linewidth=ELEMENT_COLORS["junction"]["linewidth"],
            label="Junction",
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor=ELEMENT_COLORS["junction_trunk"]["edge"],
            linewidth=ELEMENT_COLORS["junction_trunk"]["linewidth"],
            label="J–J trunk",
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor=ELEMENT_COLORS["vessel"]["edge"],
            linewidth=ELEMENT_COLORS["vessel"]["linewidth"],
            label="Vessel",
        ),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=FONT_CONFIG["legend"])

    outp = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=165, bbox_inches="tight")
    plt.close()
    print(f"Saved {outp}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot 0D zerod topology (vessel ids and/or block names)."
    )
    ap.add_argument("zerod_json", help="Path to zerod JSON")
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG (default: results/zerod_tree/<basename>_tree.png)",
    )
    ap.add_argument("--title", default="", help="Subtitle for figure title")
    ap.add_argument(
        "--large", action="store_true", help="Larger figure for dense graphs"
    )
    ap.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        metavar="S",
        help="Multiply node/edge spacing (>1 spreads layout more; default 1.0)",
    )
    ns = ap.parse_args()

    jpath = os.path.abspath(ns.zerod_json)
    if not os.path.isfile(jpath):
        print(f"Not found: {jpath}", file=sys.stderr)
        sys.exit(1)

    if ns.output:
        outp = os.path.abspath(ns.output)
    else:
        base = os.path.splitext(os.path.basename(jpath))[0]
        outp = os.path.join(os.getcwd(), "results", "zerod_tree", f"{base}_tree.png")

    sub = ns.title.strip() or os.path.basename(jpath)
    ok = visualize_zerod_tree(
        jpath,
        outp,
        title_suffix=sub,
        large_graph=ns.large,
        spacing_scale=ns.spacing,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
