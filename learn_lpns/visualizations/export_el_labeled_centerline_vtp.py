#!/usr/bin/env python3
"""
Write a centerline VTP whose BranchId / BifurcationId reflect the EL geometric model.

CLI: pass ``--set_name``, ``--geo_name``, and optionally ``--run_config`` to infer
paths under ``data/``, or pass explicit ``--geometric_input``, ``--centerline``,
and ``--output``.

Reads bifurcations_EL_geometric_input.json (or any geometric_input with EL-style
centerline_node_ids) plus the original centerline VTP. Junction regions are built
as the union of graph paths from each junction inlet to each outlet GID. Non-
connector vessels get BranchId = vessel_id on centerline points along the vessel
path minus junction points (junction wins at shared endpoints). Connector
vessels are not given a branch label; their points are covered only by junction
paths.

Original BranchId and BifurcationId are copied to BranchId_orig and
BifurcationId_orig before overwriting (skipped if *_orig already exist).

``NORMAL_JUNCTION`` entries are not treated as spatial bifurcation regions (no
paths, no ``BifurcationId``). Other junction types use sequential
``BifurcationId`` 0, 1, … in JSON ``junctions`` order among non-``NORMAL_JUNCTION``
junctions that have a valid inlet. Points on those junction paths get that
index; branch-only points use -1. Unused points remain -1 for both arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n
except ImportError as e:
    raise SystemExit("VTK is required: pip install vtk") from e

from learn_lpns.zerod_calibration.tools.file_io import read_centerline_vtp


def find_centerline_file(set_name: str, geo_name: str, data_dir: str = "data") -> Optional[str]:
    """
    Find centerline VTP under data/oneD or data/threeD for a geometry.

    VMR cohort set names (e.g. VMR_rigid_aorta_adults) store 1D solutions under
    data/oneD/VMR/{geo_name}/, matching ``tools.file_io.get_paths`` and
    ``generate_zerod_inputs.py``.
    """
    possible_paths = [
        os.path.join(data_dir, "oneD", set_name, geo_name, "unsteady_soln.vtp"),
        os.path.join(data_dir, "reduced_results", set_name, geo_name, "unsteady_soln.vtp"),
        os.path.join(data_dir, "oneD", set_name, geo_name, "centerlines_simVascular.vtp"),
        os.path.join(data_dir, "threeD", set_name, geo_name, "centerlines_simVascular.vtp"),
        os.path.join(data_dir, "threeD", set_name, geo_name, "centerlines", "centerlines.vtp"),
        os.path.join(data_dir, "threeD", set_name, geo_name, "centerlines.vtp"),
    ]
    if "VMR" in set_name:
        possible_paths.extend(
            [
                os.path.join(data_dir, "oneD", "VMR", geo_name, "unsteady_soln.vtp"),
                os.path.join(data_dir, "oneD", "VMR_rigid_aortas", geo_name, "unsteady_soln.vtp"),
            ]
        )
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def _centerline_search_paths(set_name: str, geo_name: str, data_dir: str = "data") -> List[str]:
    """Return candidate centerline paths (for error messages)."""
    paths = [
        os.path.join(data_dir, "oneD", set_name, geo_name, "unsteady_soln.vtp"),
        os.path.join(data_dir, "reduced_results", set_name, geo_name, "unsteady_soln.vtp"),
        os.path.join(data_dir, "oneD", set_name, geo_name, "centerlines_simVascular.vtp"),
        os.path.join(data_dir, "threeD", set_name, geo_name, "centerlines_simVascular.vtp"),
        os.path.join(data_dir, "threeD", set_name, geo_name, "centerlines", "centerlines.vtp"),
        os.path.join(data_dir, "threeD", set_name, geo_name, "centerlines.vtp"),
    ]
    if "VMR" in set_name:
        paths.extend(
            [
                os.path.join(data_dir, "oneD", "VMR", geo_name, "unsteady_soln.vtp"),
                os.path.join(data_dir, "oneD", "VMR_rigid_aortas", geo_name, "unsteady_soln.vtp"),
            ]
        )
    return paths


def infer_paths_from_set_run_geo(
    set_name: str,
    geo_name: str,
    run_config: Optional[str] = None,
    data_dir: str = "data",
    output_path: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Infer paths to EL geometric JSON, centerline VTP, and output VTP.

    - Geometric input: ``data/zeroD/<set>/<run>/<geo>/bifurcations_EL_geometric_input.json``
      (omit ``<run>/`` when ``run_config`` is None).
    - Centerline: first existing file from :func:`find_centerline_file`.
    - Output: ``centerlines_EL_labeled.vtp`` in the same directory as the
      centerline, unless ``output_path`` is given.

    Returns:
        (geometric_input_path, centerline_path, output_path)
    """
    if run_config:
        zerod_dir = os.path.join(data_dir, "zeroD", set_name, run_config, geo_name)
    else:
        zerod_dir = os.path.join(data_dir, "zeroD", set_name, geo_name)

    geometric_input_path = os.path.join(zerod_dir, "bifurcations_EL_geometric_input.json")

    centerline_path = find_centerline_file(set_name, geo_name, data_dir)
    if centerline_path is None:
        tried = "\n  ".join(_centerline_search_paths(set_name, geo_name, data_dir))
        raise FileNotFoundError(
            f"Could not find centerline VTP for set={set_name!r} geo={geo_name!r} "
            f"under data_dir={data_dir!r}. Tried:\n  {tried}"
        )

    if output_path is None:
        out_dir = os.path.dirname(os.path.abspath(centerline_path))
        output_path = os.path.join(out_dir, "centerlines_EL_labeled.vtp")
    else:
        output_path = os.path.abspath(output_path)

    return geometric_input_path, centerline_path, output_path


def _is_connector_vessel(vessel_name: str) -> bool:
    return "connector" in (vessel_name or "").lower()


def _find_gid_index(gid_arr: np.ndarray, gid: int) -> Optional[int]:
    matches = np.where(gid_arr == gid)[0]
    return int(matches[0]) if len(matches) > 0 else None


def _build_adjacency(n_points: int, cells: Sequence[Sequence[int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_points)]
    for edge in cells:
        if len(edge) != 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        if 0 <= a < n_points and 0 <= b < n_points:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def _bfs_path(adj: Sequence[Sequence[int]], start: int, goal: int) -> Optional[List[int]]:
    if start == goal:
        return [start]
    parent: Dict[int, Optional[int]] = {start: None}
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
                if v == goal:
                    path: List[int] = []
                    cur: Optional[int] = goal
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    return path[::-1]
    return None


def _path_along_branch(
    branch_id_arr: np.ndarray,
    path_arr: np.ndarray,
    idx_a: int,
    idx_b: int,
) -> Optional[np.ndarray]:
    """All point indices on the same BranchId with Path between the two endpoints (inclusive)."""
    ba = int(branch_id_arr[idx_a])
    bb = int(branch_id_arr[idx_b])
    if ba != bb or ba < 0:
        return None
    mask = branch_id_arr == ba
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    pa, pb = float(path_arr[idx_a]), float(path_arr[idx_b])
    lo, hi = min(pa, pb), max(pa, pb)
    sub = indices[(path_arr[indices] >= lo) & (path_arr[indices] <= hi)]
    order = np.argsort(path_arr[sub])
    return sub[order]


def _path_between_points(
    centerline_arrays: dict,
    idx_in: int,
    idx_out: int,
    adj: Sequence[Sequence[int]],
) -> Optional[List[int]]:
    """Unique path between two centerline vertices (tree graph or single-branch fallback)."""
    cells = centerline_arrays.get("Cells") or []
    if len(cells) > 0:
        p = _bfs_path(adj, idx_in, idx_out)
        if p is not None:
            return p
        # BFS failed (e.g. missing segments); try same-BranchId path

    branch_id_arr = centerline_arrays.get("BranchId")
    path_arr = centerline_arrays.get("Path")
    if branch_id_arr is None or path_arr is None:
        return None
    branch_id_arr = np.asarray(branch_id_arr)
    path_arr = np.asarray(path_arr)
    seg = _path_along_branch(branch_id_arr, path_arr, idx_in, idx_out)
    if seg is None or len(seg) == 0:
        return None
    return [int(x) for x in seg]


def _junction_outlet_gids(junc: dict) -> List[Tuple[str, int]]:
    """(vessel_name, gid) for each outlet with a valid GID."""
    cn = junc.get("centerline_node_ids") or {}
    outlets = cn.get("outlets")
    if not outlets:
        return []
    out: List[Tuple[str, int]] = []
    if isinstance(outlets, dict):
        for name, gid in outlets.items():
            if gid is None:
                continue
            out.append((str(name), int(gid)))
    return out


def _assign_el_labels(
    centerline_arrays: dict,
    geometric_input: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (branch_id_el, bifurcation_id_el) int32 arrays, length n_points.

    Junctions with ``junction_type == "NORMAL_JUNCTION"`` are skipped for spatial
    bifurcation labeling (0D pressure node only).
    """
    gid_arr = np.asarray(centerline_arrays["GlobalNodeId"])
    n_points = len(gid_arr)
    branch_el = np.full(n_points, -1, dtype=np.int32)
    bif_el = np.full(n_points, -1, dtype=np.int32)

    cells = centerline_arrays.get("Cells") or []
    adj = _build_adjacency(n_points, cells)

    vessels = geometric_input.get("vessels") or []
    junctions = geometric_input.get("junctions") or []

    junction_point_to_jidx: Dict[int, int] = {}

    # --- Junction regions (non-NORMAL only; bif_region_idx sequential among those) ---
    bif_region_idx = 0
    for junc in junctions:
        if junc.get("junction_type") == "NORMAL_JUNCTION":
            continue
        cn = junc.get("centerline_node_ids") or {}
        inlet_gid = cn.get("inlet")
        if inlet_gid is None:
            continue
        idx_in = _find_gid_index(gid_arr, int(inlet_gid))
        if idx_in is None:
            continue

        outlet_pairs = _junction_outlet_gids(junc)
        for _name, ogid in outlet_pairs:
            idx_out = _find_gid_index(gid_arr, ogid)
            if idx_out is None:
                continue
            path = _path_between_points(centerline_arrays, idx_in, idx_out, adj)
            if not path:
                continue
            for pt in path:
                if pt not in junction_point_to_jidx:
                    junction_point_to_jidx[pt] = bif_region_idx
        bif_region_idx += 1

    junction_indices: Set[int] = set(junction_point_to_jidx.keys())

    # --- Vessel branches: non-connector only; junction wins at overlaps ---
    for v in vessels:
        name = v.get("vessel_name", "")
        if _is_connector_vessel(name):
            continue
        vid = int(v.get("vessel_id", -1))
        cn = v.get("centerline_node_ids") or {}
        inlet_gid = cn.get("inlet")
        outlet_gid = cn.get("outlet")
        if inlet_gid is None or outlet_gid is None:
            continue
        idx_in = _find_gid_index(gid_arr, int(inlet_gid))
        idx_out = _find_gid_index(gid_arr, int(outlet_gid))
        if idx_in is None or idx_out is None:
            continue
        path = _path_between_points(centerline_arrays, idx_in, idx_out, adj)
        if not path:
            continue
        for pt in path:
            if pt in junction_indices:
                continue
            branch_el[pt] = vid

    # --- BifurcationId on junction points ---
    for pt, jidx in junction_point_to_jidx.items():
        bif_el[pt] = int(jidx)

    return branch_el, bif_el


def _numpy_to_vtk_int32(name: str, arr: np.ndarray) -> "vtk.vtkIntArray":
    vtk_arr = numpy_to_vtk(arr.astype(np.int32), deep=True)
    vtk_arr.SetName(name)
    return vtk_arr


def _remove_array_if_present(pd: "vtk.vtkPointData", name: str) -> None:
    if pd.HasArray(name):
        pd.RemoveArray(name)


def export_el_labeled_centerline_vtp(
    geometric_input_path: str,
    centerline_path: str,
    output_path: str,
) -> str:
    with open(geometric_input_path, "r") as f:
        geometric_input = json.load(f)

    centerline_arrays, poly = read_centerline_vtp(centerline_path)
    if "GlobalNodeId" not in centerline_arrays:
        raise ValueError("Centerline must contain GlobalNodeId")

    pd = poly.GetPointData()

    branch_el, bif_el = _assign_el_labels(centerline_arrays, geometric_input)
    if len(branch_el) != poly.GetNumberOfPoints():
        raise ValueError("EL label length mismatch with polydata (GlobalNodeId vs number of points)")

    # Preserve originals (only if not already present, so re-runs keep true SimVascular ids)
    if not pd.HasArray("BranchId_orig") and pd.HasArray("BranchId"):
        orig = np.asarray(v2n(pd.GetArray("BranchId")))
        pd.AddArray(_numpy_to_vtk_int32("BranchId_orig", orig.astype(np.int32)))
    if not pd.HasArray("BifurcationId_orig") and pd.HasArray("BifurcationId"):
        orig_b = np.asarray(v2n(pd.GetArray("BifurcationId")))
        _nan = np.isnan(orig_b.astype(float))
        orig_b_clean = orig_b.copy()
        if orig_b_clean.dtype.kind in "fc":
            orig_b_clean = np.where(_nan, -1, orig_b_clean)
        pd.AddArray(_numpy_to_vtk_int32("BifurcationId_orig", orig_b_clean.astype(np.int32)))

    _remove_array_if_present(pd, "BranchId")
    _remove_array_if_present(pd, "BifurcationId")
    pd.AddArray(_numpy_to_vtk_int32("BranchId", branch_el))
    pd.AddArray(_numpy_to_vtk_int32("BifurcationId", bif_el))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.Write()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write centerline VTP with BranchId/BifurcationId from EL geometric_input. "
            "BifurcationId is sequential among non-NORMAL_JUNCTION junctions only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Path modes:
  (1) Provide --set_name and --geo_name (and optionally --run_config): paths are
      inferred under --data_dir.
  (2) Provide --geometric_input, --centerline, and --output explicitly.

Examples:
  python -m learn_lpns.visualizations.export_el_labeled_centerline_vtp \\
      --set_name VMR_rigid_aorta_adults --geo_name 0094_0001 --run_config gen_loss

  python -m learn_lpns.visualizations.export_el_labeled_centerline_vtp \\
      --set_name VMR --geo_name 0063_1001 --output /tmp/out.vtp

  python -m learn_lpns.visualizations.export_el_labeled_centerline_vtp \\
      --geometric_input data/zeroD/VMR/g/bifurcations_EL_geometric_input.json \\
      --centerline data/oneD/VMR/g/unsteady_soln.vtp --output /tmp/labeled.vtp
""",
    )
    parser.add_argument(
        "--set_name",
        type=str,
        default=None,
        help="Dataset name (e.g. VMR_rigid_aorta_adults); use with --geo_name to infer paths",
    )
    parser.add_argument(
        "--geo_name",
        type=str,
        default=None,
        help="Geometry id (e.g. 0094_0001); use with --set_name",
    )
    parser.add_argument(
        "--run_config",
        type=str,
        default=None,
        help="ZeroD subfolder under set (e.g. gen_loss). If omitted, use data/zeroD/<set>/<geo>/",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory for inferred paths (default: data)",
    )
    parser.add_argument(
        "--geometric_input",
        type=str,
        default=None,
        help="Explicit path to bifurcations_EL_geometric_input.json",
    )
    parser.add_argument(
        "--centerline",
        type=str,
        default=None,
        help="Explicit path to original centerline VTP",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output VTP path (default next to centerline: centerlines_EL_labeled.vtp)",
    )
    args = parser.parse_args()

    use_inferred = args.set_name is not None or args.geo_name is not None
    if use_inferred:
        if not args.set_name or not args.geo_name:
            parser.error("--set_name and --geo_name are required together for inferred paths")
        try:
            geometric_input, centerline, output = infer_paths_from_set_run_geo(
                args.set_name,
                args.geo_name,
                run_config=args.run_config,
                data_dir=args.data_dir,
                output_path=args.output,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print("Inferred paths:")
        print(f"  geometric_input: {geometric_input}")
        print(f"  centerline:      {centerline}")
        print(f"  output:          {output}")
    else:
        if not args.geometric_input or not args.centerline or not args.output:
            parser.error(
                "Either pass --set_name and --geo_name, or all of --geometric_input, --centerline, and --output"
            )
        geometric_input = args.geometric_input
        centerline = args.centerline
        output = args.output

    if not os.path.isfile(geometric_input):
        print(f"Error: geometric input not found: {geometric_input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(centerline):
        print(f"Error: centerline not found: {centerline}", file=sys.stderr)
        sys.exit(1)

    out = export_el_labeled_centerline_vtp(geometric_input, centerline, output)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
