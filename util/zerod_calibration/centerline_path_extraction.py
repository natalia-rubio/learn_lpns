"""
Extract in-junction centerline paths using BranchIdTmp.

For each junction in a geometric_input.json, this module identifies the ordered
sequence of centerline GlobalNodeIds that form the path from the junction inlet
to each outlet.  It uses BranchIdTmp (an alternative branch labelling that
extends *through* junction regions) rather than BranchId (which is -1 inside
junctions).

Algorithm for each outlet:
  1.  Find the BranchIdTmp value at the outlet boundary (the first point of the
      outlet branch).
  2.  Collect all centerline points with that BranchIdTmp that lie inside the
      junction region (BifurcationId == junction_bif_id) or on the outlet branch
      up to the outlet GID.
  3.  Check whether the collected path reaches the junction inlet point.  If not,
      look for a *connector* BranchIdTmp — one whose points in the junction
      region bridge the gap between the inlet and the start of the outlet
      BranchIdTmp path.  Repeat until the path connects to the inlet.
  4.  Return the full ordered list of GIDs from inlet to outlet.
"""

import json
import os

import numpy as np

from util.zerod_calibration.tools.file_io import read_centerline_vtp


def get_path_length_from_gid_list(gid_list, centerline_data):
    """
    Compute the path length defined by an ordered list of GlobalNodeIds by
    summing the Euclidean distance between each pair of consecutive points.

    Parameters
    ----------
    gid_list : list[int]
        Ordered sequence of GlobalNodeIds defining the path.
    centerline_data : dict
        Centerline arrays (must contain 'GlobalNodeId' and 'Points').

    Returns
    -------
    float
        Total path length (sum of consecutive-point distances).
    """
    if len(gid_list) < 2:
        return 0.0

    gid_arr = np.asarray(centerline_data["GlobalNodeId"])
    points = np.asarray(centerline_data["Points"])

    # Build GID -> point-index lookup (once)
    gid_to_idx = {}
    for i, g in enumerate(gid_arr):
        gid_to_idx[int(g)] = i

    total = 0.0
    prev_idx = gid_to_idx.get(gid_list[0])
    if prev_idx is None:
        raise ValueError(f"GID {gid_list[0]} not found in centerline data")

    for g in gid_list[1:]:
        cur_idx = gid_to_idx.get(g)
        if cur_idx is None:
            raise ValueError(f"GID {g} not found in centerline data")
        total += float(np.linalg.norm(points[cur_idx] - points[prev_idx]))
        prev_idx = cur_idx

    return total


def _get_branch_id_from_name(vessel_name):
    """Extract integer branch ID from a vessel name like 'branch3_seg0'."""
    import re

    m = re.match(r"branch(\d+)", vessel_name)
    if m:
        return int(m.group(1))
    return None


def extract_junction_centerline_paths(centerline_data, geometric_input):
    """
    For every junction in *geometric_input*, build the ordered GID path from
    junction inlet to each outlet using BranchIdTmp.

    Returns
    -------
    dict
        junction_name -> {
            outlet_vessel_name -> {
                'outlet_branch_id_tmp': int,
                'connector_branch_id_tmps': [int, ...],
                'path_gids': [int, ...],   # ordered inlet -> outlet
            }
        }
    """
    gid = np.asarray(centerline_data["GlobalNodeId"])
    branch_id = np.asarray(centerline_data["BranchId"]).astype(int)
    bif_id = np.asarray(centerline_data["BifurcationId"])
    path_arr = np.asarray(centerline_data["Path"])
    points = np.asarray(centerline_data["Points"])

    if "BranchIdTmp" not in centerline_data:
        raise ValueError("BranchIdTmp array not found in centerline data")
    branch_id_tmp = np.asarray(centerline_data["BranchIdTmp"]).astype(int)

    mir = centerline_data.get("MaximumInscribedSphereRadius")
    if mir is not None:
        mir = np.asarray(mir)

    vessels = geometric_input.get("vessels", [])
    junctions = geometric_input.get("junctions", [])

    len(gid)

    # Pre-compute: for each BranchId, the inlet/outlet point indices (by Path)
    branch_inlet_idx = {}
    branch_outlet_idx = {}
    for b in np.unique(branch_id):
        if b < 0:
            continue
        mask = branch_id == b
        idx = np.where(mask)[0]
        bp = path_arr[idx]
        branch_inlet_idx[int(b)] = int(idx[np.argmin(bp)])
        branch_outlet_idx[int(b)] = int(idx[np.argmax(bp)])

    def find_point_from_gid(gid_value):
        matches = np.where(gid == gid_value)[0]
        return int(matches[0]) if len(matches) > 0 else None

    # ---------------------------------------------------------------
    result = {}

    for junc in junctions:
        junc_name = junc.get("junction_name", "")

        # Derive BifurcationId for this junction
        jid_part = junc_name[1:]  # strip leading 'J'
        if "_bif" in jid_part:
            jid_part = jid_part.split("_bif")[0]
        junction_bif_id = int(jid_part)

        inlet_vessel_ids = junc.get("inlet_vessels", [])
        outlet_vessel_ids = junc.get("outlet_vessels", [])
        if not inlet_vessel_ids or not outlet_vessel_ids:
            continue

        # Inlet vessel info
        inlet_id = inlet_vessel_ids[0]
        inlet_vessel = vessels[inlet_id]
        inlet_name = inlet_vessel.get("vessel_name", "")
        inlet_branch = _get_branch_id_from_name(inlet_name)

        # Junction inlet GID (from centerline_node_ids or branch outlet)
        junc_node_ids = junc.get("centerline_node_ids", {})
        inlet_gid_val = junc_node_ids.get("inlet")
        if inlet_gid_val is not None:
            inlet_pt = find_point_from_gid(inlet_gid_val)
        else:
            inlet_pt = branch_outlet_idx.get(inlet_branch)
            if inlet_pt is not None:
                inlet_gid_val = int(gid[inlet_pt])

        if inlet_pt is None:
            print(f"  Warning: could not locate inlet point for {junc_name}, skipping")
            continue

        inlet_bit = int(branch_id_tmp[inlet_pt])

        # Mask for junction region
        junc_mask = bif_id == junction_bif_id

        junc_result = {}

        for vessel_id in outlet_vessel_ids:
            vessel = vessels[vessel_id]
            vessel_name = vessel.get("vessel_name", "")
            b_id = _get_branch_id_from_name(vessel_name)

            # Outlet GID
            outlet_gids_dict = junc_node_ids.get("outlets", {})
            outlet_gid_val = outlet_gids_dict.get(vessel_name) if isinstance(outlet_gids_dict, dict) else None
            if outlet_gid_val is not None:
                outlet_pt = find_point_from_gid(outlet_gid_val)
            elif b_id is not None and b_id in branch_inlet_idx:
                outlet_pt = branch_inlet_idx[b_id]
                outlet_gid_val = int(gid[outlet_pt])
            else:
                print(f"    Warning: could not locate outlet point for {vessel_name}, skipping")
                junc_result[vessel_name] = {
                    "outlet_branch_id_tmp": None,
                    "connector_branch_id_tmps": [],
                    "path_gids": [],
                }
                continue

            # BranchIdTmp at the outlet point
            outlet_bit = int(branch_id_tmp[outlet_pt])

            # ----- collect points per BranchIdTmp in the junction region -----
            # For each BranchIdTmp present in the junction, store its points
            # sorted by Path.
            bits_in_junction = {}
            junc_indices = np.where(junc_mask)[0]
            for ji in junc_indices:
                bit = int(branch_id_tmp[ji])
                bits_in_junction.setdefault(bit, []).append(ji)

            # Also include points on the outlet branch up to the outlet GID
            # (for EL-adjusted connectors whose outlet is past the junction boundary)
            outlet_branch_pts = []
            if b_id is not None:
                b_mask = branch_id == b_id
                b_indices = np.where(b_mask)[0]
                outlet_path_val = float(path_arr[outlet_pt])
                for bi in b_indices:
                    if float(path_arr[bi]) <= outlet_path_val:
                        outlet_branch_pts.append(int(bi))

            # ----- trace path from outlet back to inlet using BranchIdTmp -----
            # Start with the outlet BranchIdTmp, then find connectors as needed.
            chain_bits = [outlet_bit]
            connector_bits = []
            visited_bits = {outlet_bit}

            MAX_DEPTH = 20
            for _ in range(MAX_DEPTH):
                current_bit = chain_bits[-1]
                pts_for_bit = bits_in_junction.get(current_bit, [])
                if not pts_for_bit:
                    break

                # Find the point with the smallest Path in this BranchIdTmp
                # within the junction region
                pts_sorted = sorted(pts_for_bit, key=lambda i: float(path_arr[i]))
                start_pt = pts_sorted[0]
                start_coord = points[start_pt]
                inlet_coord = points[inlet_pt]

                # Check if this start is close to the inlet
                dist_to_inlet = float(np.linalg.norm(start_coord - inlet_coord))
                if dist_to_inlet < 1e-6:
                    break  # path reaches inlet

                # Also check if BranchIdTmp at inlet_pt matches current
                if inlet_bit == current_bit:
                    break

                # Look for a connector BranchIdTmp whose end connects to this
                # start.  A connector's highest-Path point in the junction
                # should be close to start_pt.
                best_connector = None
                best_dist = float("inf")
                for cand_bit, cand_pts in bits_in_junction.items():
                    if cand_bit in visited_bits:
                        continue
                    cand_sorted = sorted(cand_pts, key=lambda i: float(path_arr[i]))
                    cand_end = points[cand_sorted[-1]]
                    d = float(np.linalg.norm(cand_end - start_coord))
                    if d < best_dist:
                        best_dist = d
                        best_connector = cand_bit

                if best_connector is None or best_dist > 1.0:
                    # Also try: check if inlet BranchIdTmp has points in the junction
                    # whose end connects
                    if inlet_bit not in visited_bits and inlet_bit in bits_in_junction:
                        ipts = sorted(bits_in_junction[inlet_bit], key=lambda i: float(path_arr[i]))
                        d = float(np.linalg.norm(points[ipts[-1]] - start_coord))
                        if d < 1.0:
                            best_connector = inlet_bit
                            best_dist = d
                    if best_connector is None or best_dist > 1.0:
                        break

                visited_bits.add(best_connector)
                connector_bits.append(best_connector)
                chain_bits.append(best_connector)

            # ----- assemble ordered GID list -----
            # chain_bits is [outlet_bit, connector1, connector2, ..., (inlet_bit maybe)]
            # Reverse so we go inlet -> outlet
            chain_bits_reversed = list(reversed(chain_bits))

            all_path_points = []  # (path_val, gid_val, pt_idx)

            # Add junction inlet point
            all_path_points.append((float(path_arr[inlet_pt]), int(gid[inlet_pt]), inlet_pt))

            for bit in chain_bits_reversed:
                pts_for_bit = bits_in_junction.get(bit, [])
                for pi in pts_for_bit:
                    all_path_points.append((float(path_arr[pi]), int(gid[pi]), pi))

            # Add outlet branch points (for EL extension)
            for bi in outlet_branch_pts:
                all_path_points.append((float(path_arr[bi]), int(gid[bi]), bi))

            # # Sort by path, deduplicate
            # all_path_points.sort(key=lambda x: x[0])
            # seen = set()
            # ordered_gids = []
            # for _, g, _ in all_path_points:
            #     if g not in seen:
            #         seen.add(g)
            #         ordered_gids.append(g)
            ## Sort by path, deduplicate
            all_path_points.sort(key=lambda x: x[1])
            seen = set()
            ordered_gids = []
            for _, g, _ in all_path_points:
                if g not in seen:
                    seen.add(g)
                    ordered_gids.append(g)

            # Check that for each point, the previous and next points are physically closer than all other points
            # if False:
            # for i in range(len(ordered_gids-1)):
            #     if i > 0:
            #         prev_gid = ordered_gids[i-1]
            #         curr_gid = ordered_gids[i]
            #         next_gid = ordered_gids[i+1]
            #         prev_pt = find_point_from_gid(prev_gid)
            #         curr_pt = find_point_from_gid(curr_gid)
            #         next_pt = find_point_from_gid(next_gid)
            #         prev_coord = points[prev_pt]
            #         curr_coord = points[curr_pt]
            #         next_coord = points[next_coord]
            #         dist_prev = float(np.linalg.norm(curr_coord - prev_coord))
            #         dist_next = float(np.linalg.norm(curr_coord - next_coord))
            # if dist > 1e-6:
            #     print(f"    Warning: Points {prev_gid} and {curr_gid} are not physically close, distance={dist}")
            #     import pdb; pdb.set_trace()

            # Remove the inlet_bit from connector list if it ended up there
            connector_bits_clean = [b for b in connector_bits if b != inlet_bit]

            junc_result[vessel_name] = {
                "outlet_branch_id_tmp": outlet_bit,
                "connector_branch_id_tmps": connector_bits_clean,
                "path_gids": ordered_gids,
            }

        result[junc_name] = junc_result

    return result


def add_centerline_paths_to_config(geometric_input, junction_paths):
    """
    Add 'geometric_params' -> 'centerline_path' info to each junction
    in the config dict (in-place).
    """
    for junc in geometric_input.get("junctions", []):
        junc_name = junc.get("junction_name", "")
        if junc_name not in junction_paths:
            continue
        if "geometric_params" not in junc:
            junc["geometric_params"] = {}
        junc["geometric_params"]["outlet_centerline_paths"] = junction_paths[junc_name]
    return geometric_input


def process_geometric_input(centerline_path, geometric_input_path, output_path=None):
    """
    Read a geometric_input.json and its centerline VTP, extract per-junction
    centerline paths using BranchIdTmp, and save the augmented config as
    geometric_centerline_input.json.

    Parameters
    ----------
    centerline_path : str
        Path to centerline VTP file.
    geometric_input_path : str
        Path to geometric_input.json.
    output_path : str or None
        Output path.  If None, replaces the extension with
        '_centerline_input.json' in the same directory.
    """
    print(f"Reading centerline from: {centerline_path}")
    centerline_data, _ = read_centerline_vtp(centerline_path)

    print(f"Reading geometric input from: {geometric_input_path}")
    with open(geometric_input_path, "r") as f:
        geometric_input = json.load(f)

    print("Extracting junction centerline paths using BranchIdTmp ...")
    junction_paths = extract_junction_centerline_paths(centerline_data, geometric_input)

    for jname, outlets in junction_paths.items():
        for vname, info in outlets.items():
            n_gids = len(info["path_gids"])
            conn = info["connector_branch_id_tmps"]
            print(f"  {jname} -> {vname}: BranchIdTmp={info['outlet_branch_id_tmp']}, connectors={conn}, {n_gids} GIDs")

    add_centerline_paths_to_config(geometric_input, junction_paths)

    if output_path is None:
        base, ext = os.path.splitext(geometric_input_path)
        output_path = base.replace("geometric_input", "geometric_centerline_input") + ext

    with open(output_path, "w") as f:
        json.dump(geometric_input, f, indent=4)
    print(f"Saved augmented config to: {output_path}")

    return output_path


# ---- CLI entry point ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract in-junction centerline paths using BranchIdTmp")
    parser.add_argument("centerline_path", help="Path to centerline VTP file")
    parser.add_argument("geometric_input_path", help="Path to geometric_input.json")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: geometric_centerline_input.json)",
    )
    args = parser.parse_args()

    process_geometric_input(args.centerline_path, args.geometric_input_path, args.output)
