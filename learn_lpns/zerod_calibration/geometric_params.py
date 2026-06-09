#!/usr/bin/env python3
"""
Functions to extract and add geometric parameters to 0D vessels and junctions.

For each vessel we currently store:
  - inlet_area, outlet_area                      (from CenterlineSectionArea at segment ends)
  - path_length                                  (0D vessel_length from geometric input)
  - tortuosity                                   (path_length / straight_distance between ends)
  - angle_diff                                   (angle between inlet and outlet tangents along the vessel)

For each junction we currently store:
  - inlet_vessel_areas, outlet_vessel_areas      (areas at the vessel/junction interfaces)
  - outlet_path_lengths                          (in‑junction path length from inlet to each outlet,
                                                  identical to the metric used in bifurcation_splitting)
  - inlet_tangent, outlet_tangents               (unit direction vectors at inlet / outlet sides)
  - outlet_tortuosities                          (in‑junction path_length / straight distance inlet→outlet)
  - inlet_max_inscribed_radius                   (MaximumInscribedSphereRadius at inlet branch outlet)
  - outlet_max_inscribed_radius                  (MaximumInscribedSphereRadius at each outlet branch inlet)
  - max_inscribed_radius_min_on_path, max_inscribed_radius_max_on_path
                                                  (min / max MaximumInscribedSphereRadius along the
                                                  path between inlet and each outlet; this uses the
                                                  same in‑junction segments as in bifurcation_splitting,
                                                  plus the inlet/outlet points themselves)
"""

import json
import re

import numpy as np

from learn_lpns.zerod_calibration.tools.file_io import read_centerline_vtp


def get_angle_diff(vec1, vec2):
    """
    Compute the angle difference between two 3D vectors (in radians).

    Both vectors are normalised before taking the arccos of their dot product.
    """
    v1 = np.asarray(vec1, dtype=float).reshape(-1)
    v2 = np.asarray(vec2, dtype=float).reshape(-1)

    if v1.size != 3 or v2.size != 3:
        raise ValueError(f"get_angle_diff expects 3D vectors, got shapes {v1.shape} and {v2.shape}")

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 0.0 or n2 <= 0.0:
        raise ValueError("Cannot compute angle between zero-length vectors")

    v1n = v1 / n1
    v2n = v2 / n2

    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return float(np.arccos(dot))


def extract_vessel_junction_areas(centerline_soln_path, geometric_input_path):
    """
    Extract inlet and outlet areas for all vessels and junctions from centerline solution.

    Similar to how observations are extracted, this function finds the appropriate
    centerline points for each vessel's inlet and outlet, and extracts the area
    at those points.

    Args:
        centerline_soln_path: Path to centerline solution VTP file (with area arrays)
        geometric_input_path: Path to geometric 0D input JSON (to understand vessel/junction structure)

    Returns:
        Dictionary with structure:
        {
            'vessels': {
                'vessel_name': {
                    'inlet_area': float,
                    'outlet_area': float
                },
                ...
            },
            'junctions': {
                'junction_name': {
                    'inlet_vessel_areas': {vessel_name: area, ...},
                    'outlet_vessel_areas': {vessel_name: area, ...}
                },
                ...
            }
        }
    """
    print(f"Reading centerline solution from: {centerline_soln_path}")
    centerline_data, _ = read_centerline_vtp(centerline_soln_path)
    verbose = False
    # Read geometric input to understand vessel/junction structure
    with open(geometric_input_path, "r") as f:
        geometric_input = json.load(f)

    vessels = geometric_input.get("vessels", [])
    junctions = geometric_input.get("junctions", [])

    # Get area array from centerline
    area = centerline_data.get("CenterlineSectionArea", None)
    if area is None:
        raise ValueError("CenterlineSectionArea array not found in centerline solution")

    branch_id = centerline_data.get("BranchId", None)
    if branch_id is None:
        raise ValueError("BranchId array not found in centerline solution")
    branch_id = np.asarray(branch_id)

    path_arr = centerline_data.get("Path", None)
    if path_arr is None:
        raise ValueError("Path array not found in centerline solution")
    path_arr = np.asarray(path_arr)

    gid = centerline_data.get("GlobalNodeId", None)
    points_array = centerline_data.get("Points", None)
    if points_array is None:
        raise ValueError("Points array not found in centerline solution")
    points_array = np.asarray(points_array)

    bifurcation_id_array = centerline_data.get("BifurcationId", None)
    if bifurcation_id_array is not None:
        bifurcation_id_array = np.asarray(bifurcation_id_array)
    else:
        raise ValueError("BifurcationId not found in centerline data")

    max_inscribed_radius = centerline_data.get("MaximumInscribedSphereRadius", None)
    if max_inscribed_radius is not None:
        max_inscribed_radius = np.asarray(max_inscribed_radius)
    else:
        raise ValueError("MaximumInscribedSphereRadius not found in centerline data")

    # Find inlet (GID == 0) and outlets
    outlet_indices = []
    if gid is not None:
        for i in range(len(gid)):
            if gid[i] == 0:
                pass
            elif gid[i] > 0:
                outlet_indices.append(i)

    # Helper to find centerline point index from GlobalNodeId
    def find_point_from_gid(gid_value):
        """
        Find centerline point index corresponding to a GlobalNodeId.

        Args:
            gid_value: GlobalNodeId (int)

        Returns:
            Point index (int) or None if not found
        """
        if gid is None:
            return None
        gid_np = np.atleast_1d(np.asarray(gid))
        matches = np.where(gid_np == gid_value)[0]
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            # If multiple matches, use the first one (shouldn't happen, but handle gracefully)
            print(f"    Warning: Multiple centerline points found with GID {gid_value}, using first match")
        return int(matches[0])

    # Helper to get inlet/outlet point indices from vessel's centerline_node_ids (GlobalNodeId) when present
    def find_point_indices_from_node_ids(v):
        """If vessel has centerline_node_ids with inlet and outlet, return (inlet_idx, outlet_idx) else (None, None)."""
        node_ids = v.get("centerline_node_ids") or {}
        inlet_gid = node_ids.get("inlet")
        outlet_gid = node_ids.get("outlet")
        if inlet_gid is None or outlet_gid is None:
            return None, None
        inlet_idx = find_point_from_gid(int(inlet_gid))
        outlet_idx = find_point_from_gid(int(outlet_gid))
        if inlet_idx is None or outlet_idx is None:
            return None, None
        return inlet_idx, outlet_idx

    # Helper to find a centerline point corresponding to a 0D vessel segment
    # (Same logic as in oned_to_zerod.py)
    def find_point_for_vessel_segment(vessel_name, prefer_end=True):
        """
        Map a 0D vessel (e.g. 'branch3_seg1') to a point index in the centerline arrays.
        prefer_end: if True, return a point near the downstream end of the segment;
                    if False, return a point near the upstream/start of the segment.
        """
        # Parse branch index from vessel_name
        parts = vessel_name.split("_")
        if not parts:
            raise ValueError(f"Invalid vessel name format: {vessel_name}")
        branch_part = parts[0]
        if not branch_part.startswith("branch"):
            raise ValueError(f"Vessel name {vessel_name} does not start with 'branch'")
        branch_str = branch_part.replace("branch", "")
        if not branch_str:
            raise ValueError(f"Could not extract branch number from vessel name: {vessel_name}")
        branch_idx = int(branch_str)

        # Indices on the centerline that belong to this branch
        branch_pts = [i for i, bid in enumerate(branch_id) if bid == branch_idx]
        if not branch_pts:
            raise ValueError(f"No centerline points found for branch {branch_idx} (vessel {vessel_name})")

        # Collect all 0D vessels from geometric input that belong to this branch
        branch_vessels = [v for v in vessels if v.get("vessel_name", "").startswith(f"branch{branch_idx}_")]
        if not branch_vessels:
            raise ValueError(f"No vessels found for branch {branch_idx} (vessel {vessel_name})")

        # Sort vessels by segment index
        def seg_index(v):
            name = v.get("vessel_name", "")
            if "_seg" in name and "connector" not in name:
                seg_str = name.split("_seg")[-1]
                if not seg_str:
                    raise ValueError(f"Invalid segment index in vessel name: {name}")
                return int(seg_str)
            return 0

        branch_vessels.sort(key=seg_index)

        # Build cumulative lengths along the branch from the 0D vessel lengths
        lengths = [float(v.get("vessel_length", 0.0) or 0.0) for v in branch_vessels]
        if sum(lengths) <= 0 and "connector" not in vessel_name:
            raise ValueError(f"Total vessel length is zero or negative for branch {branch_idx} (vessel {vessel_name})")

        cum_lengths = np.cumsum(lengths)

        # Determine which index in branch_vessels corresponds to the requested vessel
        idx_in_list = next((i for i, v in enumerate(branch_vessels) if v.get("vessel_name") == vessel_name), None)
        if idx_in_list is None:
            raise ValueError(f"Vessel {vessel_name} not found in branch {branch_idx} vessels")

        # Compute target path position measured from the branch start
        branch_start_path = float(path_arr[branch_pts[0]])
        if prefer_end:
            target_rel = float(cum_lengths[idx_in_list])
        else:
            seg_len = float(lengths[idx_in_list])
            if idx_in_list == 0:
                target_rel = 0.0
            else:
                target_rel = float(cum_lengths[idx_in_list] - seg_len)
        target_abs = branch_start_path + target_rel

        # Find nearest point
        branch_pts_sorted = sorted(branch_pts, key=lambda i: float(path_arr[i]))
        branch_paths = [float(path_arr[i]) for i in branch_pts_sorted]
        distances = [abs(p - target_abs) for p in branch_paths]
        nearest_idx = branch_pts_sorted[int(np.argmin(distances))]
        return nearest_idx

    # Extract areas for vessels
    vessel_areas = {}
    for vessel in vessels:
        vessel_name = vessel.get("vessel_name", "")
        if not vessel_name:
            raise ValueError("Vessel with empty vessel_name found")

        is_connector = "connector" in vessel_name
        angle_diff = None

        if is_connector:
            # For connector vessels: use inlet vessel outlet values and set tortuosity/path_length to 0
            # Find the junction this connector is an outlet of (could be any JX_bifY)
            connector_outlet_junction = None
            vessel_id = vessel.get("vessel_id")
            for junc in junctions:
                outlet_vessel_ids = junc.get("outlet_vessels", [])
                if vessel_id in outlet_vessel_ids:
                    connector_outlet_junction = junc
                    break

            if connector_outlet_junction is None:
                raise ValueError(f"Could not find outlet junction for connector vessel {vessel_name}")

            # Extract base junction name and find the appropriate junction to get inlet vessel
            outlet_junc_name = connector_outlet_junction.get("junction_name", "")

            # Determine which junction to use for getting the inlet vessel
            # If the junction is split (has _bif suffix), find the first bifurcation (_bif0)
            # Otherwise, use the junction directly
            if "_bif" in outlet_junc_name:
                # Junction is split - find the first bifurcation (JX_bif0)
                junction_base = outlet_junc_name.split("_bif")[0]
                first_bif_name = f"{junction_base}_bif0"
                target_junction = None
                for junc in junctions:
                    if junc.get("junction_name") == first_bif_name:
                        target_junction = junc
                        break

                if target_junction is None:
                    raise ValueError(
                        f"Could not find first bifurcation {first_bif_name} for connector vessel {vessel_name}"
                    )
            else:
                # Junction is not split - use it directly
                target_junction = connector_outlet_junction

            # Get the inlet vessel of the target junction (the physical inlet to the junction)
            inlet_vessel_ids = target_junction.get("inlet_vessels", [])
            if not inlet_vessel_ids:
                junction_name = target_junction.get("junction_name", "unknown")
                raise ValueError(f"Junction {junction_name} has no inlet vessels (for connector {vessel_name})")

            inlet_vessel_id = inlet_vessel_ids[0]
            if inlet_vessel_id >= len(vessels):
                raise ValueError(f"Inlet vessel ID {inlet_vessel_id} out of bounds for connector {vessel_name}")

            inlet_vessel = vessels[inlet_vessel_id]
            inlet_vessel_name = inlet_vessel.get("vessel_name", "")
            if not inlet_vessel_name:
                raise ValueError(f"Inlet vessel {inlet_vessel_id} has empty vessel_name for connector {vessel_name}")

            # Use inlet vessel's outlet point (where it connects to the junction)
            inlet_vessel_outlet_idx = find_point_for_vessel_segment(inlet_vessel_name, prefer_end=True)
            if inlet_vessel_outlet_idx is None:
                raise ValueError(
                    f"Could not find outlet point for inlet vessel {inlet_vessel_name} (for connector {vessel_name})"
                )
            if inlet_vessel_outlet_idx >= len(area):
                raise ValueError(
                    f"Point index {inlet_vessel_outlet_idx} out of bounds for area array (length {len(area)})"
                )

            # Use inlet vessel outlet values for connector vessel
            inlet_area = float(area[inlet_vessel_outlet_idx])
            outlet_area = float(area[inlet_vessel_outlet_idx])  # Same for connector
            path_length = 0.0
            tortuosity = 0.0
            angle_diff = 0.0

        else:
            # Regular vessel processing: prefer inlet/outlet from centerline_node_ids when present (e.g. EL geometry)
            inlet_point_idx, outlet_point_idx = find_point_indices_from_node_ids(vessel)
            if inlet_point_idx is None or outlet_point_idx is None:
                import pdb

                pdb.set_trace()
                raise ValueError(f"Could not find inlet or outlet point for vessel {vessel_name}")
                # inlet_point_idx = find_point_for_vessel_segment(vessel_name, prefer_end=False)
                # outlet_point_idx = find_point_for_vessel_segment(vessel_name, prefer_end=True)

            if inlet_point_idx is None:
                raise ValueError(f"Could not find inlet point for vessel {vessel_name}")
            if outlet_point_idx is None:
                raise ValueError(f"Could not find outlet point for vessel {vessel_name}")
            if inlet_point_idx >= len(area):
                raise ValueError(
                    f"Inlet point index {inlet_point_idx} out of bounds for area array "
                    f"(length {len(area)}) for vessel {vessel_name}"
                )
            if outlet_point_idx >= len(area):
                raise ValueError(
                    f"Outlet point index {outlet_point_idx} out of bounds for area array "
                    f"(length {len(area)}) for vessel {vessel_name}"
                )

            inlet_area = float(area[inlet_point_idx])
            outlet_area = float(area[outlet_point_idx])

            # Path length and tortuosity for this vessel
            # Use geometric 0D vessel_length as the path length
            path_length = float(vessel.get("vessel_length", 0.0) or 0.0)
            if path_length <= 0.0:
                raise ValueError(f"Vessel {vessel_name} has invalid path_length: {path_length}")

            if inlet_point_idx >= len(points_array):
                raise ValueError(
                    f"Inlet point index {inlet_point_idx} out of bounds for points array "
                    f"(length {len(points_array)}) for vessel {vessel_name}"
                )
            if outlet_point_idx >= len(points_array):
                raise ValueError(
                    f"Outlet point index {outlet_point_idx} out of bounds for points array "
                    f"(length {len(points_array)}) for vessel {vessel_name}"
                )

            p_in = points_array[inlet_point_idx]
            p_out = points_array[outlet_point_idx]
            straight_dist = float(np.linalg.norm(p_out - p_in))
            if straight_dist <= 0.0:
                tortuosity = 0.0
            else:
                tortuosity = path_length / straight_dist

            # Angle between inlet and outlet tangents along this vessel
            parts = vessel_name.split("_")
            if not parts:
                raise ValueError(f"Invalid vessel name format: {vessel_name}")
            branch_part = parts[0]
            if not branch_part.startswith("branch"):
                raise ValueError(f"Vessel name {vessel_name} does not start with 'branch'")
            branch_str = branch_part.replace("branch", "")
            if not branch_str:
                raise ValueError(f"Could not extract branch number from vessel name: {vessel_name}")
            branch_idx = int(branch_str)

            mask = branch_id == branch_idx
            idx = np.where(mask)[0]
            if idx.size < 2:
                raise ValueError(
                    f"Insufficient points ({idx.size}) on branch {branch_idx} "
                    f"to compute vessel tangents for {vessel_name}"
                )

            branch_paths = path_arr[idx]
            order = np.argsort(branch_paths)
            idx_sorted = idx[order]

            if inlet_point_idx not in idx_sorted:
                raise ValueError(
                    f"Inlet point index {inlet_point_idx} not found on branch {branch_idx} for vessel {vessel_name}"
                )
            if outlet_point_idx not in idx_sorted:
                raise ValueError(
                    f"Outlet point index {outlet_point_idx} not found on branch {branch_idx} for vessel {vessel_name}"
                )

            inlet_pos = int(np.where(idx_sorted == inlet_point_idx)[0][0])
            outlet_pos = int(np.where(idx_sorted == outlet_point_idx)[0][0])

            if inlet_pos >= len(idx_sorted) - 1:
                raise ValueError(f"No downstream neighbour available to compute inlet tangent for vessel {vessel_name}")
            if outlet_pos == 0:
                raise ValueError(f"No upstream neighbour available to compute outlet tangent for vessel {vessel_name}")

            inlet_next_idx = idx_sorted[inlet_pos + 1]
            outlet_prev_idx = idx_sorted[outlet_pos - 1]

            v_in = points_array[inlet_next_idx] - points_array[inlet_point_idx]
            v_out = points_array[outlet_point_idx] - points_array[outlet_prev_idx]

            n_in = float(np.linalg.norm(v_in))
            n_out = float(np.linalg.norm(v_out))
            if n_in <= 0.0 or n_out <= 0.0:
                raise ValueError(f"Zero-length vector encountered when computing vessel tangents for {vessel_name}")

            inlet_tangent_vessel = v_in / n_in
            outlet_tangent_vessel = v_out / n_out
            angle_diff = get_angle_diff(inlet_tangent_vessel, outlet_tangent_vessel)

            # Maximum inscribed sphere radius: inlet, outlet, min and max along the vessel segment
            inlet_misr = float(max_inscribed_radius[inlet_point_idx])
            outlet_misr = float(max_inscribed_radius[outlet_point_idx])
            segment_indices = idx_sorted[inlet_pos: outlet_pos + 1]
            if segment_indices.size > 0:
                segment_radii = max_inscribed_radius[segment_indices]
                misr_min = float(np.min(segment_radii))
                misr_max = float(np.max(segment_radii))
            else:
                misr_min = min(inlet_misr, outlet_misr)
                misr_max = max(inlet_misr, outlet_misr)

        if is_connector:
            vessel_areas[vessel_name] = {
                "inlet_area": inlet_area,
                "outlet_area": outlet_area,
                "path_length": path_length,
                "tortuosity": tortuosity,
                "angle_diff": angle_diff,
            }
        else:
            vessel_areas[vessel_name] = {
                "inlet_area": inlet_area,
                "outlet_area": outlet_area,
                "path_length": path_length,
                "tortuosity": tortuosity,
                "angle_diff": angle_diff,
                "inlet_max_inscribed_radius": inlet_misr,
                "outlet_max_inscribed_radius": outlet_misr,
                "max_inscribed_radius_min": misr_min,
                "max_inscribed_radius_max": misr_max,
            }

        print(
            f"  {vessel_name}: inlet_area={inlet_area:.6f}, outlet_area={outlet_area:.6f}, "
            f"path_length={path_length:.6f}, tortuosity={tortuosity:.6f}, "
            f"angle_diff={angle_diff:.6f}"
            + (
                f", MISR inlet={inlet_misr:.6f} outlet={outlet_misr:.6f} min={misr_min:.6f} max={misr_max:.6f}"
                if not is_connector
                else ""
            )
        )

    # Pre-compute inlet/outlet points and indices for each branch (used by junction metrics)
    branch_inlet_point = {}
    branch_outlet_point = {}
    branch_inlet_idx = {}
    branch_outlet_idx = {}
    path_arr_np = path_arr
    unique_branches = np.unique(branch_id)
    for b in unique_branches:
        mask = branch_id == b
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"No centerline points found for branch {b}")
        branch_paths = path_arr_np[idx]
        branch_points = points_array[idx]
        # Inlet is min Path, outlet is max Path on this branch
        min_local = int(np.argmin(branch_paths))
        max_local = int(np.argmax(branch_paths))
        start_idx = int(idx[min_local])
        end_idx = int(idx[max_local])
        branch_inlet_idx[int(b)] = start_idx
        branch_outlet_idx[int(b)] = end_idx
        branch_inlet_point[int(b)] = branch_points[min_local].copy()
        branch_outlet_point[int(b)] = branch_points[max_local].copy()

    # Helper to get branch id from a vessel name like "branch3_seg0" or "branch0_seg0_connector0"
    def get_branch_id_from_name(vessel_name):
        parts = vessel_name.split("_")
        if not parts:
            raise ValueError(f"Invalid vessel name format: {vessel_name}")
        branch_part = parts[0]
        if not branch_part.startswith("branch"):
            raise ValueError(f"Vessel name {vessel_name} does not start with 'branch'")
        branch_str = branch_part.replace("branch", "")
        if not branch_str:
            raise ValueError(f"Could not extract branch number from vessel name: {vessel_name}")
        return int(branch_str)

    # Helper mirroring bifurcation_splitting.compute_in_junction_path_lengths,
    # extended to also track MaximumInscribedSphereRadius along each outlet path.
    def compute_junction_outlet_metrics(inlet_branch_id, outlet_branch_ids, junction_bif_id):
        """
        Returns:
            dict: outlet_branch_id -> {
                'path_length': float,
                'radius_min': float,
                'radius_max': float,
            }
        """
        metrics = {}

        if inlet_branch_id not in branch_outlet_point:
            raise ValueError(f"No outlet point found for inlet branch {inlet_branch_id}")

        inlet_endpoint = branch_outlet_point[inlet_branch_id]

        if junction_bif_id is None:
            # If BifurcationId is None, we can't compute junction metrics
            return metrics

        junction_mask = bifurcation_id_array == junction_bif_id
        if not np.any(junction_mask):
            # No junction region found - this can happen for simple pass-through junctions
            # Return empty metrics dict (caller will handle gracefully)
            print(f"    Debug: No centerline points found with BifurcationId={junction_bif_id}")
            if bifurcation_id_array is not None:
                avail_ids = np.unique(bifurcation_id_array[~np.isnan(bifurcation_id_array)])
            else:
                avail_ids = "None"
            print(f"    Debug: Available BifurcationIds in centerline: {avail_ids}")
            return metrics

        num_junction_points = np.sum(junction_mask)
        print(f"    Debug: Found {num_junction_points} centerline points with BifurcationId={junction_bif_id}")
        # Additional debug context for matching failures
        print(
            f"    Debug: compute_junction_outlet_metrics context: "
            f"inlet_branch_id={inlet_branch_id}, outlet_branch_ids={outlet_branch_ids}"
        )

        junc_indices = np.where(junction_mask)[0]
        junction_paths = path_arr_np[junction_mask]
        junction_points = points_array[junction_mask]
        junction_radii = max_inscribed_radius[junction_mask]

        sort_order = np.argsort(junction_paths)
        sorted_paths = junction_paths[sort_order]
        sorted_points = junction_points[sort_order]
        sorted_indices = junc_indices[sort_order]
        sorted_radii = junction_radii[sort_order]

        # Identify distinct path segments by finding discontinuities
        segments = []
        current_segment_start = 0
        for i in range(1, len(sorted_paths)):
            path_diff = sorted_paths[i] - sorted_paths[i - 1]
            if i > 1:
                prev_diff = sorted_paths[i - 1] - sorted_paths[i - 2]
                if abs(path_diff) > 10 * abs(prev_diff) + 0.01:
                    segments.append((current_segment_start, i))
                    current_segment_start = i
            elif path_diff < -0.001:
                segments.append((current_segment_start, i))
                current_segment_start = i
        segments.append((current_segment_start, len(sorted_paths)))

        # For each segment, associate it with the closest outlet branch inlet,
        # and compute path length and radius range along that segment.
        for seg_start, seg_end in segments:
            seg_paths = sorted_paths[seg_start:seg_end]
            seg_points = sorted_points[seg_start:seg_end]
            seg_indices = sorted_indices[seg_start:seg_end]
            if len(seg_paths) == 0:
                raise ValueError(f"Empty path segment found in junction region (BifurcationId={junction_bif_id})")

            segment_path_length = float(np.max(seg_paths) - np.min(seg_paths))

            min_path_idx = int(np.argmin(seg_paths))
            startpoint = seg_points[min_path_idx]

            distance_to_inlet = float(np.linalg.norm(startpoint - inlet_endpoint))
            path_length_val = segment_path_length + distance_to_inlet

            max_path_idx = int(np.argmax(seg_paths))
            endpoint = seg_points[max_path_idx]
            best_outlet = None
            best_distance = float("inf")
            # Debug: print segment info
            try:
                if verbose:
                    print(f"    Debug: segment {seg_start}-{seg_end}, segment_path_length={segment_path_length:.6f}")
                    print(f"      startpoint={startpoint}, endpoint={endpoint}")
            except Exception:
                pass
            for outlet_branch_id in outlet_branch_ids:
                if outlet_branch_id not in branch_inlet_point:
                    raise ValueError(
                        f"Outlet branch {outlet_branch_id} not found in branch_inlet_point "
                        f"for junction BifurcationId={junction_bif_id}"
                    )
                outlet_inlet = branch_inlet_point[outlet_branch_id]
                distance = float(np.linalg.norm(endpoint - outlet_inlet))
                print(
                    f"      Debug: distance from segment endpoint to outlet branch "
                    f"{outlet_branch_id} inlet: {distance:.6f}"
                )
                if distance < best_distance:
                    best_distance = distance
                    best_outlet = outlet_branch_id

            if best_outlet is None:
                if verbose:
                    print(
                        f"    Debug: Could not match segment to any outlet branch (best_distance={best_distance:.6f})"
                    )
                    print(f"    Debug: Available outlet branch IDs: {outlet_branch_ids}")
                    print(f"    Debug: Segment endpoint: {endpoint}")
                continue  # Skip this segment instead of raising error
            threshold = 10
            if best_distance >= threshold:
                if verbose:
                    print(
                        f"    Debug: Best outlet match distance {best_distance:.6f} exceeds "
                        f"threshold of {threshold} for outlet {best_outlet}"
                    )
                if best_outlet is not None and best_outlet in branch_inlet_point:
                    if verbose:
                        print(
                            f"    Debug: Segment endpoint: {endpoint}, outlet inlet: {branch_inlet_point[best_outlet]}"
                        )
                else:
                    if verbose:
                        print(f"    Debug: Segment endpoint: {endpoint}, no valid best_outlet to show inlet coords")
                if verbose:
                    print(f"    Debug: Skipping segment {seg_start}-{seg_end} (no reliable match)")
                continue  # Skip this segment instead of raising error

            # Collect radius extrema for this segment (we'll add inlet/outlet points below)
            seg_radii = sorted_radii[seg_start:seg_end]
            seg_rad_min = float(np.min(seg_radii))
            seg_rad_max = float(np.max(seg_radii))

            # Keep only the longest path per outlet
            if best_outlet not in metrics or path_length_val > metrics[best_outlet]["path_length"]:
                metrics[best_outlet] = {
                    "path_length": path_length_val,
                    "segment_indices": seg_indices,
                    "seg_rad_min": seg_rad_min,
                    "seg_rad_max": seg_rad_max,
                }

        # Augment with inlet / outlet radii
        for outlet_branch_id, data in metrics.items():
            if inlet_branch_id not in branch_outlet_idx:
                raise ValueError(f"Inlet branch {inlet_branch_id} not found in branch_outlet_idx")
            if outlet_branch_id not in branch_inlet_idx:
                raise ValueError(f"Outlet branch {outlet_branch_id} not found in branch_inlet_idx")

            inlet_idx = branch_outlet_idx[inlet_branch_id]
            outlet_idx = branch_inlet_idx[outlet_branch_id]

            vals = [data["seg_rad_min"], data["seg_rad_max"]]
            vals.append(float(max_inscribed_radius[inlet_idx]))
            vals.append(float(max_inscribed_radius[outlet_idx]))

            data["radius_min"] = float(np.min(vals))
            data["radius_max"] = float(np.max(vals))

        return metrics

    # Extract areas and geometric metrics for junctions
    junction_areas = {}
    for junc in junctions:
        junc_name = junc.get("junction_name", "")
        if not junc_name:
            raise ValueError("Junction with empty junction_name found")

        inlet_vessel_ids = junc.get("inlet_vessels", [])
        outlet_vessel_ids = junc.get("outlet_vessels", [])

        inlet_vessel_areas = {}
        outlet_vessel_areas = {}
        outlet_path_lengths = {}
        outlet_tortuosities = {}
        outlet_tangents = {}
        outlet_max_inscribed_radius_min_on_path = {}
        outlet_max_inscribed_radius_max_on_path = {}
        outlet_angle_diffs = {}
        outlet_path_gids = {}

        inlet_tangent = None
        inlet_radius_val = None
        inlet_branch_id = None

        # Get junction inlet GID (from junction's centerline_node_ids)
        junc_node_ids = junc.get("centerline_node_ids", {})
        inlet_gid = junc_node_ids.get("inlet")

        if inlet_gid is not None:
            # Use GID directly to find the junction inlet point
            inlet_pt_idx = find_point_from_gid(inlet_gid)
            if inlet_pt_idx is None:
                print(
                    f"    Warning: Could not find centerline point with GID {inlet_gid} "
                    f"for junction {junc_name} inlet, falling back to vessel lookup"
                )
                inlet_gid = None

        if inlet_gid is None:
            # Fallback: use vessel-based lookup (for backward compatibility)
            if not inlet_vessel_ids:
                raise ValueError(f"Junction {junc_name} has no inlet vessels and no inlet GID")

            inlet_id = inlet_vessel_ids[0]
            if inlet_id >= len(vessels):
                raise ValueError(f"Inlet vessel ID {inlet_id} out of bounds for junction {junc_name}")

            inlet_vessel = vessels[inlet_id]
            inlet_name = inlet_vessel.get("vessel_name", "")
            if not inlet_name:
                raise ValueError(f"Inlet vessel {inlet_id} has empty vessel_name for junction {junc_name}")

            inlet_branch_id = get_branch_id_from_name(inlet_name)
            inlet_pt_idx = find_point_for_vessel_segment(inlet_name, prefer_end=True)
            if inlet_pt_idx is None:
                raise ValueError(f"Could not find outlet point for inlet vessel {inlet_name} in junction {junc_name}")
        else:
            # Get vessel name for area dictionary key
            if not inlet_vessel_ids:
                raise ValueError(f"Junction {junc_name} has no inlet vessels")
            inlet_id = inlet_vessel_ids[0]
            if inlet_id >= len(vessels):
                raise ValueError(f"Inlet vessel ID {inlet_id} out of bounds for junction {junc_name}")
            inlet_vessel = vessels[inlet_id]
            inlet_name = inlet_vessel.get("vessel_name", "")
            if not inlet_name:
                raise ValueError(f"Inlet vessel {inlet_id} has empty vessel_name for junction {junc_name}")
            inlet_branch_id = get_branch_id_from_name(inlet_name)

        if inlet_pt_idx >= len(area):
            raise ValueError(
                f"Point index {inlet_pt_idx} out of bounds for area array (length {len(area)}) "
                f"for junction {junc_name} inlet"
            )

        inlet_vessel_areas[inlet_name] = float(area[inlet_pt_idx])

        # Tangent at inlet side: use points near the inlet GID point
        # Find nearby points on the same branch to compute tangent
        if inlet_branch_id is not None:
            mask = branch_id == inlet_branch_id
            idx = np.where(mask)[0]
            if idx.size >= 2:
                # Find the inlet point in the branch indices
                branch_paths = path_arr_np[idx]
                order = np.argsort(branch_paths)
                idx_sorted = idx[order]

                # Find inlet point position in sorted list
                inlet_pos = np.where(idx_sorted == inlet_pt_idx)[0]
                if len(inlet_pos) > 0:
                    inlet_pos = inlet_pos[0]
                    # Use point before inlet (upstream) to compute tangent
                    if inlet_pos > 0:
                        prev_idx = idx_sorted[inlet_pos - 1]
                        v = points_array[inlet_pt_idx] - points_array[prev_idx]
                    else:
                        # If inlet is first point, use next point
                        next_idx = idx_sorted[inlet_pos + 1] if inlet_pos + 1 < len(idx_sorted) else inlet_pt_idx
                        v = points_array[next_idx] - points_array[inlet_pt_idx]

                    nrm = np.linalg.norm(v)
                    if nrm > 0.0:
                        inlet_tangent = (v / nrm).tolist()

        if inlet_tangent is None:
            raise ValueError(f"Could not compute inlet tangent for junction {junc_name}")

        # MaximumInscribedSphereRadius at junction inlet point
        inlet_radius_val = float(max_inscribed_radius[inlet_pt_idx])

        # For outlet vessels: get area at inlet (start) of the vessel (where it connects to junction)
        if not outlet_vessel_ids:
            raise ValueError(f"Junction {junc_name} has no outlet vessels")

        # Get outlet GIDs from junction (if available) - dict of vessel_name -> gid
        outlet_gids = junc_node_ids.get("outlets", {})

        # Create mapping from outlet vessel ID to outlet GID using vessel names
        outlet_id_to_gid = {}
        if outlet_gids and isinstance(outlet_gids, dict):
            for vessel_id in outlet_vessel_ids:
                if vessel_id < len(vessels):
                    vessel_name = vessels[vessel_id].get("vessel_name", "")
                    if vessel_name in outlet_gids and outlet_gids[vessel_name] is not None:
                        outlet_id_to_gid[vessel_id] = outlet_gids[vessel_name]

        outlet_branch_ids = []
        for vessel_id in outlet_vessel_ids:
            if vessel_id >= len(vessels):
                raise ValueError(f"Outlet vessel ID {vessel_id} out of bounds for junction {junc_name}")

            vessel = vessels[vessel_id]
            vessel_name = vessel.get("vessel_name", "")
            if not vessel_name:
                raise ValueError(f"Outlet vessel {vessel_id} has empty vessel_name for junction {junc_name}")

            # Check if this is a connector vessel
            is_connector = "connector" in vessel_name

            # Extract branch ID even for connectors (needed for junction-level metrics)
            b_id = get_branch_id_from_name(vessel_name)
            print(f"Outlet vessel {vessel_name} branch ID: {b_id}")

            # Get outlet GID for this vessel (if available)
            outlet_gid = outlet_id_to_gid.get(vessel_id)
            print(f"Outlet GID for vessel {vessel_name}: {outlet_gid}")
            outlet_pt_idx = None

            if outlet_gid is not None:
                # Use GID directly to find the outlet point
                outlet_pt_idx = find_point_from_gid(outlet_gid)
                print(f"Outlet point index for vessel {vessel_name}: {outlet_pt_idx} from GID {outlet_gid}")
                if outlet_pt_idx is None:
                    print(
                        f"    Warning: Could not find centerline point with GID {outlet_gid} "
                        f"for outlet vessel {vessel_name} in junction {junc_name}, "
                        f"falling back to vessel lookup"
                    )
                    outlet_gid = None

            if outlet_gid is None:
                # Fallback: use vessel-based lookup
                outlet_pt_idx = find_point_for_vessel_segment(vessel_name, prefer_end=False)
                if outlet_pt_idx is None:
                    raise ValueError(
                        f"Could not find inlet point for outlet vessel {vessel_name} in junction {junc_name}"
                    )

            if outlet_pt_idx >= len(area):
                raise ValueError(
                    f"Point index {outlet_pt_idx} out of bounds for area array (length {len(area)}) "
                    f"for outlet vessel {vessel_name}"
                )

            if is_connector and "connectorEL" not in vessel_name:
                # For outlets to connector vessels: these are artificial constructs.
                # Treat the junction->connector outlet path as zero-length and inherit
                # inlet properties (tangent, local radius). Do NOT include the
                # connector's branch id in outlet_branch_ids so the in-junction
                # path segmentation doesn't try to match segments to connector outlets.
                # Find the first bifurcation (JX_bif0) to get the original inlet
                if junc_name.endswith("_bif0"):
                    # This is the first bifurcation, use its inlet vessel outlet
                    inlet_vessel_outlet_area = inlet_vessel_areas.get(inlet_name)
                    if inlet_vessel_outlet_area is None:
                        raise ValueError(
                            f"Inlet vessel outlet area not found for connector outlet "
                            f"{vessel_name} in junction {junc_name}"
                        )
                    outlet_vessel_areas[vessel_name] = inlet_vessel_outlet_area
                    outlet_tangents[vessel_name] = inlet_tangent.copy() if inlet_tangent else None
                    # For connector outlets, set junction-level metrics to inlet values / zero
                    outlet_path_lengths[vessel_name] = 0.0
                    outlet_tortuosities[vessel_name] = 0.0
                    outlet_max_inscribed_radius_min_on_path[vessel_name] = inlet_radius_val
                    outlet_max_inscribed_radius_max_on_path[vessel_name] = inlet_radius_val
                    # Angle diff between inlet and connector outlet is zero (same tangent)
                    outlet_angle_diffs[vessel_name] = 0.0
                    # Path is just the inlet point (connector is at same location)
                    outlet_path_gids[vessel_name] = [int(gid[inlet_pt_idx])] if gid is not None else []
                else:
                    # Not the first bifurcation, find the appropriate junction to get inlet vessel
                    # If junction is split, find JX_bif0; otherwise use the junction directly
                    if "_bif" in junc_name:
                        # Junction is split - find the first bifurcation (JX_bif0)
                        junction_base = junc_name.split("_bif")[0]
                        target_junc_name = f"{junction_base}_bif0"
                        target_junc = None
                        for j in junctions:
                            if j.get("junction_name") == target_junc_name:
                                target_junc = j
                                break

                        if target_junc is None:
                            raise ValueError(
                                f"Could not find first bifurcation {target_junc_name} "
                                f"for connector outlet {vessel_name}"
                            )
                    else:
                        # Junction is not split - use it directly
                        target_junc = junc

                    target_inlet_ids = target_junc.get("inlet_vessels", [])
                    if not target_inlet_ids:
                        target_junc_name = target_junc.get("junction_name", "unknown")
                        raise ValueError(
                            f"Junction {target_junc_name} has no inlet vessels (for connector outlet {vessel_name})"
                        )

                    target_inlet_id = target_inlet_ids[0]
                    if target_inlet_id >= len(vessels):
                        raise ValueError(f"Junction inlet vessel ID {target_inlet_id} out of bounds")

                    target_inlet_vessel = vessels[target_inlet_id]
                    target_inlet_name = target_inlet_vessel.get("vessel_name", "")
                    if not target_inlet_name:
                        raise ValueError(f"Junction inlet vessel {target_inlet_id} has empty vessel_name")

                    # Use target junction inlet vessel outlet point
                    target_pt_idx = find_point_for_vessel_segment(target_inlet_name, prefer_end=True)
                    if target_pt_idx is None:
                        raise ValueError(f"Could not find outlet point for junction inlet vessel {target_inlet_name}")
                    if target_pt_idx >= len(area):
                        raise ValueError(f"Point index {target_pt_idx} out of bounds for area array")

                    outlet_vessel_areas[vessel_name] = float(area[target_pt_idx])

                    # Use tangent from target junction inlet
                    target_branch_id = get_branch_id_from_name(target_inlet_name)
                    mask = branch_id == target_branch_id
                    idx = np.where(mask)[0]
                    if idx.size >= 2:
                        branch_paths = path_arr_np[idx]
                        order = np.argsort(branch_paths)
                        idx_sorted = idx[order]
                        end_idx = idx_sorted[-1]
                        prev_idx = idx_sorted[-2]
                        v = points_array[end_idx] - points_array[prev_idx]
                        nrm = np.linalg.norm(v)
                        if nrm > 0.0:
                            outlet_tangents[vessel_name] = (v / nrm).tolist()
                        else:
                            outlet_tangents[vessel_name] = None
                    else:
                        outlet_tangents[vessel_name] = None

                    # For connector outlets (non-first bifurcations), inherit inlet radius;
                    # set path length/tortuosity to zero
                    # Find the original junction outlet idx for that branch
                    if target_branch_id not in branch_outlet_idx:
                        raise ValueError(f"Junction inlet branch {target_branch_id} not found in branch_outlet_idx")
                    target_idx_out = branch_outlet_idx[target_branch_id]
                    connector_inlet_radius = float(max_inscribed_radius[target_idx_out])
                    outlet_path_lengths[vessel_name] = 0.0
                    outlet_tortuosities[vessel_name] = 0.0
                    outlet_max_inscribed_radius_min_on_path[vessel_name] = connector_inlet_radius
                    outlet_max_inscribed_radius_max_on_path[vessel_name] = connector_inlet_radius
                    outlet_angle_diffs[vessel_name] = 0.0
                    outlet_path_gids[vessel_name] = [int(gid[inlet_pt_idx])] if gid is not None else []
                # Connectors are artificial — skip the outlet_metrics section below
                continue
            else:
                # Regular outlet vessel processing - use outlet_pt_idx computed from GID (or fallback)
                outlet_vessel_areas[vessel_name] = float(area[outlet_pt_idx])

                if b_id is not None:
                    outlet_branch_ids.append(b_id)

                # Tangent at outlet side: use points near the outlet GID point
                if b_id is not None:
                    mask = branch_id == b_id
                    idx = np.where(mask)[0]
                    if idx.size >= 2:
                        # Find the outlet point in the branch indices
                        branch_paths = path_arr_np[idx]
                        order = np.argsort(branch_paths)
                        idx_sorted = idx[order]

                        # Find outlet point position in sorted list
                        outlet_pos = np.where(idx_sorted == outlet_pt_idx)[0]
                        if len(outlet_pos) > 0:
                            outlet_pos = outlet_pos[0]
                            # Use point after outlet (downstream) to compute tangent
                            if outlet_pos + 1 < len(idx_sorted):
                                next_idx = idx_sorted[outlet_pos + 1]
                                v = points_array[next_idx] - points_array[outlet_pt_idx]
                            else:
                                # If outlet is last point, use previous point
                                prev_idx = idx_sorted[outlet_pos - 1] if outlet_pos > 0 else outlet_pt_idx
                                v = points_array[outlet_pt_idx] - points_array[prev_idx]

                            nrm = np.linalg.norm(v)
                            if nrm > 0.0:
                                outlet_tangents[vessel_name] = (v / nrm).tolist()

                if vessel_name not in outlet_tangents:
                    raise ValueError(
                        f"Could not compute outlet tangent for vessel {vessel_name} in junction {junc_name}"
                    )

        # In-junction path lengths and radius extrema, following bifurcation_splitting logic
        # Handle both regular junctions (J0) and bifurcated junctions (J0_bif0)
        # Extract the original junction ID from the name
        if not junc_name.startswith("J"):
            raise ValueError(f"Junction name {junc_name} does not start with 'J'")
        if len(junc_name) < 2:
            raise ValueError(f"Junction name {junc_name} is too short to extract BifurcationId")

        # For bifurcated junctions like "J0_bif0", extract "0" (the original junction ID)
        # For regular junctions like "J0", extract "0"
        junction_id_part = junc_name[1:]  # Remove 'J' prefix
        if "_bif" in junction_id_part:
            # Split at '_bif' and take the first part (the original junction ID)
            junction_id_part = junction_id_part.split("_bif")[0]

        if not junction_id_part:
            raise ValueError(f"Could not extract junction ID from junction name: {junc_name}")
        junction_bif_id = int(junction_id_part)

        if outlet_branch_ids:
            if verbose:
                print(
                    f"  Computing in-junction metrics for {junc_name} "
                    f"(inlet branch {inlet_branch_id}, outlets {outlet_branch_ids})"
                )

            outlet_metrics = compute_junction_outlet_metrics(inlet_branch_id, outlet_branch_ids, junction_bif_id)
            # Debug: show computed outlet_metrics keys and a short summary
            try:
                keys = list(outlet_metrics.keys())
                if verbose:
                    print(f"    Debug: outlet_metrics keys: {keys}")
                for k in keys:
                    v = outlet_metrics.get(k, {})
                    pl = v.get("path_length", None)
                    print(f"      Debug: outlet_metrics[{k}] -> path_length={pl}")
            except Exception:
                print(f"    Debug: outlet_metrics (raw): {outlet_metrics}")
        else:
            if verbose:
                print(f"  Warning: {junc_name} has no outlet branch IDs (all outlets may be connectors or invalid)")
                print(f"    Outlet vessel IDs: {outlet_vessel_ids}")
                outlet_names = [
                    vessels[vid].get("vessel_name", "unknown")
                    for vid in outlet_vessel_ids
                    if vid < len(vessels)
                ]
                print(f"    Outlet vessel names: {outlet_names}")
            outlet_metrics = {}  # Initialize to empty dict when no outlet branch IDs

            # If no metrics were computed (e.g., no BifurcationId region in centerline),
            # we skip junction-level metrics but still have basic metrics (areas, tangents, local radius)
            if not outlet_metrics:
                print(
                    f"    Warning: No junction region found for {junc_name} (BifurcationId={junction_bif_id}), "
                    f"skipping junction-level metrics (path lengths, tortuosities, radius min/max on path)"
                )

        # Map outlet_metrics (if any) and connectors into junction-level dictionaries
        for vessel_id in outlet_vessel_ids:
            if vessel_id >= len(vessels):
                raise ValueError(f"Outlet vessel ID {vessel_id} out of bounds for junction {junc_name}")

            vessel = vessels[vessel_id]
            vessel_name = vessel.get("vessel_name", "")
            if not vessel_name:
                raise ValueError(f"Outlet vessel {vessel_id} has empty vessel_name for junction {junc_name}")

            # Handle connector vessels - set path_length and tortuosity to 0
            b_id = get_branch_id_from_name(vessel_name)
            if verbose:
                print(f"Got branch id for vessel {vessel_name}: {b_id}")

            outlet_gid = outlet_id_to_gid.get(vessel_id)
            if verbose:
                print(f"Outlet GID for vessel {vessel_name}: {outlet_gid}")
            outlet_pt_idx = find_point_from_gid(outlet_gid)

            is_connector = "connector" in vessel_name
            if is_connector:
                # Non-EL connectors (e.g. _connector0) were fully handled in the
                # first loop — skip them here so we don't overwrite their values.
                if "connectorEL" not in vessel_name:
                    continue

                # Among connectorEL vessels, distinguish splitting-created ones
                # (_connectorEL0, _connectorEL1) from EL-adjusted ones (_connectorEL).
                numbered_conn = re.search(r"_connectorEL(\d+)$", vessel_name)
                if numbered_conn:
                    # Splitting-created connector: keep previous behavior (inherit inlet/tangent, zero-length)
                    if verbose:
                        print(f"Splitting-created connector: {vessel_name}")
                    outlet_path_lengths[vessel_name] = 0.0
                    outlet_tortuosities[vessel_name] = 0.0

                    # Use junction inlet radius for radius min/max on path (connectors inherit from original inlet)
                    if junc_name.endswith("_bif0"):
                        connector_inlet_radius = inlet_radius_val
                    elif "_bif" in junc_name:
                        junction_base = junc_name.split("_bif")[0]
                        target_junc_name = f"{junction_base}_bif0"
                        target_junc = None
                        for j in junctions:
                            if j.get("junction_name") == target_junc_name:
                                target_junc = j
                                break

                        if target_junc is None:
                            raise ValueError(
                                f"Could not find first bifurcation {target_junc_name} "
                                f"for connector outlet {vessel_name}"
                            )

                        target_inlet_ids = target_junc.get("inlet_vessels", [])
                        if not target_inlet_ids:
                            raise ValueError(
                                f"Junction {target_junc_name} has no inlet vessels (for connector outlet {vessel_name})"
                            )

                        target_inlet_id = target_inlet_ids[0]
                        if target_inlet_id >= len(vessels):
                            raise ValueError(f"Junction inlet vessel ID {target_inlet_id} out of bounds")

                        target_inlet_vessel = vessels[target_inlet_id]
                        target_inlet_name = target_inlet_vessel.get("vessel_name", "")
                        if not target_inlet_name:
                            raise ValueError(f"Junction inlet vessel {target_inlet_id} has empty vessel_name")

                        target_branch_id = get_branch_id_from_name(target_inlet_name)
                        if target_branch_id not in branch_outlet_idx:
                            raise ValueError(f"Junction inlet branch {target_branch_id} not found in branch_outlet_idx")
                        target_idx_out = branch_outlet_idx[target_branch_id]
                        connector_inlet_radius = float(max_inscribed_radius[target_idx_out])
                    else:
                        connector_inlet_radius = inlet_radius_val

                    outlet_max_inscribed_radius_min_on_path[vessel_name] = connector_inlet_radius
                    outlet_max_inscribed_radius_max_on_path[vessel_name] = connector_inlet_radius

                    if inlet_tangent is None:
                        raise ValueError(
                            f"Inlet tangent is None when computing angle_diff for connector outlet {vessel_name} "
                            f"in junction {junc_name}"
                        )
                    out_tan = outlet_tangents.get(vessel_name)
                    if out_tan is None:
                        raise ValueError(
                            f"Outlet tangent is None when computing angle_diff for connector outlet {vessel_name} "
                            f"in junction {junc_name}"
                        )
                    outlet_angle_diffs[vessel_name] = get_angle_diff(inlet_tangent, out_tan)
                    continue
                else:
                    # EL-adjusted connector (no numeric suffix): compute junction-level metrics
                    # based on the connector endpoint (where the EL extension ended).
                    # The outlet_pt_idx points to the centerline index at the end of the extension
                    conn_idx = outlet_pt_idx
                    if conn_idx is None and b_id is not None and b_id in branch_inlet_idx:
                        conn_idx = branch_inlet_idx[b_id]
                        print(
                            f"    Warning: GID lookup failed for connectorEL {vessel_name}, "
                            f"falling back to branch_inlet_idx[{b_id}] = {conn_idx}"
                        )
                    print(f"Adjustment created connector: {vessel_name}, conn_idx: {conn_idx}")

                    # Path length = in-junction portion + EL extension along the outlet branch.
                    # The in-junction portion is already computed by compute_junction_outlet_metrics
                    # (connectorEL branches were included in outlet_branch_ids).
                    in_junction_path = 0.0
                    if b_id is not None and b_id in outlet_metrics:
                        in_junction_path = outlet_metrics[b_id]["path_length"]

                    # EL extension: distance along the outlet branch from its inlet
                    # point to the connector endpoint
                    el_extension = 0.0
                    if conn_idx is not None and b_id is not None and b_id in branch_inlet_idx:
                        branch_inlet_path = float(path_arr_np[branch_inlet_idx[b_id]])
                        conn_path = float(path_arr_np[conn_idx])
                        el_extension = conn_path - branch_inlet_path
                        if el_extension < 0:
                            el_extension = float(
                                np.linalg.norm(points_array[conn_idx] - points_array[branch_inlet_idx[b_id]])
                            )

                    outlet_path_lengths[vessel_name] = float(in_junction_path + el_extension)
                    if verbose:
                        print(
                            f"    connectorEL {vessel_name}: in_junction_path={in_junction_path:.4f}, "
                            f"el_extension={el_extension:.4f}, total={in_junction_path + el_extension:.4f}"
                        )

                    # Tangent at the connector endpoint: compute using branch neighbours if possible
                    out_tan = None
                    if b_id is not None:
                        mask = branch_id == b_id
                        idx = np.where(mask)[0]
                        if idx.size >= 2:
                            branch_paths = path_arr_np[idx]
                            order = np.argsort(branch_paths)
                            idx_sorted = idx[order]
                            # Find position of conn_idx within idx_sorted
                            pos = np.where(idx_sorted == conn_idx)[0]
                            if len(pos) > 0:
                                pos = pos[0]
                                if pos + 1 < len(idx_sorted):
                                    next_idx = idx_sorted[pos + 1]
                                    v = points_array[next_idx] - points_array[conn_idx]
                                elif pos > 0:
                                    prev_idx = idx_sorted[pos - 1]
                                    v = points_array[conn_idx] - points_array[prev_idx]
                                else:
                                    v = None
                                if v is not None:
                                    nrm = np.linalg.norm(v)
                                    if nrm > 0.0:
                                        out_tan = (v / nrm).tolist()
                    if out_tan is None:
                        print(
                            f"    Warning: Could not compute tangent for connectorEL {vessel_name} "
                            f"in junction {junc_name}, using inlet tangent as fallback"
                        )
                        out_tan = inlet_tangent.copy() if inlet_tangent else None
                    outlet_tangents[vessel_name] = out_tan

                    # Tortuosity: total path length / straight-line distance (inlet to connector)
                    total_path = outlet_path_lengths[vessel_name]
                    if inlet_branch_id in branch_outlet_point and conn_idx is not None:
                        p_in = branch_outlet_point[inlet_branch_id]
                        p_out = points_array[conn_idx]
                        straight = float(np.linalg.norm(p_out - p_in))
                        if straight > 0.0:
                            outlet_tortuosities[vessel_name] = float(total_path / straight)
                        else:
                            outlet_tortuosities[vessel_name] = 0.0
                    else:
                        outlet_tortuosities[vessel_name] = 0.0

                    # Radius min/max on the path from junction inlet to connector endpoint.
                    # Walk actual centerline points rather than using outlet_metrics
                    # radius_min/max (which augments with branch_outlet_idx that may
                    # differ from the junction inlet GID point).
                    path_radii = []
                    # 1) Junction inlet point
                    r_inlet = float(max_inscribed_radius[inlet_pt_idx])
                    path_radii.append(r_inlet)
                    if verbose:
                        print(f"    MIR_on_path debug for {vessel_name}:")
                        print(f"      inlet_pt_idx={inlet_pt_idx}, MIR={r_inlet:.6f}")
                    # 2) Points in the BifurcationId junction region on the
                    #    segment matched to this outlet branch, filtered to only
                    #    include points on the inlet or outlet branch centerline
                    #    (excludes points on other branches' paths through the junction)
                    if b_id is not None and b_id in outlet_metrics:
                        seg_indices = outlet_metrics[b_id].get("segment_indices", np.array([]))
                        for si in seg_indices:
                            si_branch = int(branch_id[si])
                            if si_branch != inlet_branch_id and si_branch != b_id:
                                continue
                            r_si = float(max_inscribed_radius[si])
                            path_radii.append(r_si)
                            if verbose:
                                print(
                                    f"      junction seg idx={si}, GID={int(gid[si]) if gid is not None else '?'}, "
                                    f"BranchId={si_branch}, Path={float(path_arr_np[si]):.4f}, MIR={r_si:.6f}"
                                )
                    # 3) Points on the outlet branch from its inlet up to the
                    #    connector endpoint (includes branch inlet + EL extension)
                    if conn_idx is not None:
                        r_conn = float(max_inscribed_radius[conn_idx])
                        path_radii.append(r_conn)
                        if verbose:
                            print(f"      conn_idx={conn_idx}, MIR={r_conn:.6f}")
                        if b_id is not None and b_id in branch_inlet_idx:
                            branch_mask = branch_id == b_id
                            b_indices = np.where(branch_mask)[0]
                            conn_path_val = float(path_arr_np[conn_idx])
                            branch_inlet_path_val = float(path_arr_np[branch_inlet_idx[b_id]])
                            for bi in b_indices:
                                pt_path = float(path_arr_np[bi])
                                if branch_inlet_path_val <= pt_path <= conn_path_val:
                                    r_bi = float(max_inscribed_radius[bi])
                                    path_radii.append(r_bi)
                                    if verbose:
                                        gid_val = int(gid[bi]) if gid is not None else "?"
                                        print(
                                            f"      branch pt idx={bi}, GID={gid_val}, "
                                            f"Path={pt_path:.4f}, MIR={r_bi:.6f}"
                                        )
                    if path_radii:
                        outlet_max_inscribed_radius_min_on_path[vessel_name] = min(path_radii)
                        outlet_max_inscribed_radius_max_on_path[vessel_name] = max(path_radii)
                        if verbose:
                            print(f"      => min={min(path_radii):.6f}, max={max(path_radii):.6f}")
                    else:
                        outlet_max_inscribed_radius_min_on_path[vessel_name] = inlet_radius_val
                        outlet_max_inscribed_radius_max_on_path[vessel_name] = inlet_radius_val

                    # Angle diff between inlet and connector outlet tangent
                    if inlet_tangent is not None and out_tan is not None:
                        outlet_angle_diffs[vessel_name] = get_angle_diff(inlet_tangent, out_tan)
                    else:
                        outlet_angle_diffs[vessel_name] = 0.0
                    # print(f"Got outlet angle diff for connector: {vessel_name}: {outlet_angle_diffs[vessel_name]}")

                    # Build ordered list of GIDs on the path from inlet to connector
                    if gid is not None:
                        path_point_entries = []
                        # Inlet point
                        path_point_entries.append((float(path_arr_np[inlet_pt_idx]), int(gid[inlet_pt_idx])))
                        # Junction region segment points (filtered by BranchId)
                        if b_id is not None and b_id in outlet_metrics:
                            seg_indices = outlet_metrics[b_id].get("segment_indices", np.array([]))
                            for si in seg_indices:
                                si_branch = int(branch_id[si])
                                if si_branch == inlet_branch_id or si_branch == b_id:
                                    path_point_entries.append((float(path_arr_np[si]), int(gid[si])))
                        # Outlet branch points up to connector
                        if conn_idx is not None and b_id is not None and b_id in branch_inlet_idx:
                            branch_mask = branch_id == b_id
                            b_indices = np.where(branch_mask)[0]
                            conn_path_val = float(path_arr_np[conn_idx])
                            branch_inlet_path_val = float(path_arr_np[branch_inlet_idx[b_id]])
                            for bi in b_indices:
                                pt_path = float(path_arr_np[bi])
                                if branch_inlet_path_val <= pt_path <= conn_path_val:
                                    path_point_entries.append((pt_path, int(gid[bi])))
                        # Sort by path and deduplicate
                        path_point_entries.sort(key=lambda x: x[0])
                        seen = set()
                        ordered_gids = []
                        for _, g in path_point_entries:
                            if g not in seen:
                                seen.add(g)
                                ordered_gids.append(g)
                        outlet_path_gids[vessel_name] = ordered_gids
                    else:
                        outlet_path_gids[vessel_name] = []
                continue

            # Only add junction-level metrics if they were computed (outlet_metrics is not empty)
            # import pdb; pdb.set_trace()
            if b_id not in outlet_metrics:
                # If outlet_metrics is empty or this branch wasn't matched, emit debug info
                if verbose:
                    print(
                        f"    Warning: Branch {b_id} (vessel {vessel_name}) not found in "
                        f"outlet_metrics for {junc_name}, setting default path length to 0.0"
                    )

                if verbose:
                    print(f"      Debug: outlet_metrics keys: {list(outlet_metrics.keys())}")
                if verbose:
                    print(f"      Debug: branch_inlet_idx contains b_id? {b_id in branch_inlet_idx}")
                if b_id in branch_inlet_idx:
                    bi = branch_inlet_idx[b_id]
                    if verbose:
                        print(f"      Debug: branch_inlet_idx[{b_id}] = {bi}")
                    if b_id in branch_inlet_point:
                        if verbose:
                            print(f"      Debug: branch_inlet_point[{b_id}] = {branch_inlet_point[b_id]}")

                outlet_path_lengths[vessel_name] = 0.0
                outlet_tortuosities[vessel_name] = 0.0
                # Use local radius values for min/max
                if b_id in branch_inlet_idx:
                    outlet_idx = branch_inlet_idx[b_id]
                    outlet_rad = float(max_inscribed_radius[outlet_idx])
                    outlet_max_inscribed_radius_min_on_path[vessel_name] = outlet_rad
                    outlet_max_inscribed_radius_max_on_path[vessel_name] = outlet_rad
                else:
                    # Fallback to inlet radius
                    outlet_max_inscribed_radius_min_on_path[vessel_name] = inlet_radius_val
                    outlet_max_inscribed_radius_max_on_path[vessel_name] = inlet_radius_val

                # Angle difference
                if inlet_tangent is not None:
                    out_tan = outlet_tangents.get(vessel_name)
                    if out_tan is not None:
                        outlet_angle_diffs[vessel_name] = get_angle_diff(inlet_tangent, out_tan)
                    else:
                        outlet_angle_diffs[vessel_name] = 0.0
                else:
                    outlet_angle_diffs[vessel_name] = 0.0
                # No outlet_metrics — path is just inlet + outlet branch inlet
                if gid is not None:
                    path_gid_list = [int(gid[inlet_pt_idx])]
                    if b_id in branch_inlet_idx:
                        outlet_g = int(gid[branch_inlet_idx[b_id]])
                        if outlet_g != path_gid_list[-1]:
                            path_gid_list.append(outlet_g)
                    outlet_path_gids[vessel_name] = path_gid_list
                else:
                    outlet_path_gids[vessel_name] = []
                continue
            print(f"Got outlet metrics for branch {b_id}: {outlet_metrics[b_id]}")
            m = outlet_metrics[b_id]
            path_len_val = m["path_length"]

            # For EL-adjusted geometries, add the EL extension distance
            # The original path length is within the BifurcationId region.
            # After EL adjustment, the junction extends EL distance down the outlet vessel.
            # We need to add this EL extension to get the total path length.
            el_extension = 0.0

            # Check if this is an EL-adjusted geometry by looking at the vessel's centerline_node_ids
            # In EL-adjusted geometries, the vessel's inlet is at the new junction boundary (EL distance down)
            print(f"Checking vessel {vessel_name} for EL extension")
            vessel = next((v for v in vessels if v.get("vessel_name") == vessel_name), None)
            print(f"Checking vessel {vessel_name}: {vessel}")
            if vessel is not None:
                centerline_node_ids = vessel.get("centerline_node_ids", {})
                vessel_inlet_gid = centerline_node_ids.get("inlet")

                # Find the original junction boundary (outlet branch inlet point)
                # This is where the BifurcationId region ends
                original_boundary_idx = branch_inlet_idx.get(b_id)

                if vessel_inlet_gid is not None and original_boundary_idx is not None:
                    # Find the centerline point corresponding to the vessel's new inlet (EL extension point)
                    if gid is not None:
                        gid_np = np.asarray(gid)
                        new_inlet_mask = gid_np == vessel_inlet_gid
                        if np.any(new_inlet_mask):
                            new_inlet_idx = np.where(new_inlet_mask)[0][0]

                            # Calculate path distance from original boundary to new inlet along the centerline
                            # This is the EL extension distance
                            original_boundary_path = float(path_arr_np[original_boundary_idx])
                            new_inlet_path = float(path_arr_np[new_inlet_idx])

                            # EL extension is the path distance along the centerline
                            # The new inlet should be downstream of the original boundary (higher path value)
                            # For EL-adjusted geometries, the junction extends downstream,
                            # so new_inlet_path > original_boundary_path
                            path_diff = new_inlet_path - original_boundary_path
                            if path_diff > 0:
                                el_extension = path_diff
                            else:
                                # If path_diff is negative or zero, the vessel inlet hasn't moved (not EL-adjusted)
                                # or there's an issue with the node IDs
                                el_extension = 0.0

                            if el_extension > 0.0:
                                print(
                                    f"    Adding EL extension {el_extension:.6f} cm to path length "
                                    f"for {vessel_name} in {junc_name}"
                                )

            # Total path length = original junction path length + EL extension
            total_path_length = path_len_val + el_extension
            outlet_path_lengths[vessel_name] = total_path_length

            # Tortuosity between inlet and this outlet (junction-level)
            if inlet_branch_id not in branch_outlet_point:
                raise ValueError(
                    f"Inlet branch {inlet_branch_id} not found in branch_outlet_point for junction {junc_name}"
                )
            if b_id not in branch_inlet_point:
                raise ValueError(f"Outlet branch {b_id} not found in branch_inlet_point for junction {junc_name}")

            p_in = branch_outlet_point[inlet_branch_id]
            p_out = branch_inlet_point[b_id]
            straight_dist = float(np.linalg.norm(p_out - p_in))
            if straight_dist <= 0.0:
                raise ValueError(
                    f"Straight-line distance is zero between inlet and outlet branch {b_id} in junction {junc_name}"
                )

            outlet_tortuosities[vessel_name] = float(path_len_val / straight_dist)

            outlet_max_inscribed_radius_min_on_path[vessel_name] = m["radius_min"]
            outlet_max_inscribed_radius_max_on_path[vessel_name] = m["radius_max"]

            # Angle between inlet and outlet tangents for this outlet
            if inlet_tangent is None:
                raise ValueError(
                    f"Inlet tangent is None when computing angle_diff for outlet {vessel_name} in junction {junc_name}"
                )
            out_tan = outlet_tangents.get(vessel_name)
            if out_tan is None:
                import pdb

                pdb.set_trace()
                raise ValueError(
                    f"Outlet tangent is None when computing angle_diff for outlet {vessel_name} in junction {junc_name}"
                )
            outlet_angle_diffs[vessel_name] = get_angle_diff(inlet_tangent, out_tan)

            # Build ordered list of GIDs on the path from inlet to this outlet
            if gid is not None:
                path_point_entries = []
                # Inlet point
                path_point_entries.append((float(path_arr_np[inlet_pt_idx]), int(gid[inlet_pt_idx])))
                # Junction region segment points
                seg_indices = m.get("segment_indices", np.array([]))
                for si in seg_indices:
                    path_point_entries.append((float(path_arr_np[si]), int(gid[si])))
                # Outlet branch inlet point
                if b_id in branch_inlet_idx:
                    oi = branch_inlet_idx[b_id]
                    path_point_entries.append((float(path_arr_np[oi]), int(gid[oi])))
                # If there's an EL extension, include branch points up to the
                # vessel's new inlet GID
                if el_extension > 0.0 and vessel is not None:
                    cni = vessel.get("centerline_node_ids", {})
                    v_inlet_gid = cni.get("inlet")
                    if v_inlet_gid is not None and b_id in branch_inlet_idx:
                        branch_mask = branch_id == b_id
                        b_indices = np.where(branch_mask)[0]
                        new_inlet_idx = find_point_from_gid(v_inlet_gid)
                        if new_inlet_idx is not None:
                            new_inlet_path_val = float(path_arr_np[new_inlet_idx])
                            branch_inlet_path_val = float(path_arr_np[branch_inlet_idx[b_id]])
                            for bi in b_indices:
                                pt_path = float(path_arr_np[bi])
                                if branch_inlet_path_val <= pt_path <= new_inlet_path_val:
                                    path_point_entries.append((pt_path, int(gid[bi])))
                # Sort by path and deduplicate
                path_point_entries.sort(key=lambda x: x[0])
                seen = set()
                ordered_gids = []
                for _, g in path_point_entries:
                    if g not in seen:
                        seen.add(g)
                        ordered_gids.append(g)
                outlet_path_gids[vessel_name] = ordered_gids
            else:
                outlet_path_gids[vessel_name] = []

        # MaximumInscribedSphereRadius at each outlet point (local value).
        # For regular outlets: radius at the branch inlet point.
        # For connectors: radius at the connector's outlet GID (the adjusted outlet point).
        outlet_radius_val = {}
        for vessel_id in outlet_vessel_ids:
            if vessel_id >= len(vessels):
                raise ValueError(f"Outlet vessel ID {vessel_id} out of bounds for junction {junc_name}")

            vessel = vessels[vessel_id]
            vessel_name = vessel.get("vessel_name", "")
            if not vessel_name:
                raise ValueError(f"Outlet vessel {vessel_id} has empty vessel_name for junction {junc_name}")

            is_connector = "connector" in vessel_name

            if is_connector:
                # Use the outlet GID to look up the radius at the actual adjusted outlet point
                connector_gid = outlet_id_to_gid.get(vessel_id)
                if connector_gid is not None:
                    pt_idx = find_point_from_gid(connector_gid)
                    if pt_idx is not None:
                        outlet_radius_val[vessel_name] = float(max_inscribed_radius[pt_idx])
                    else:
                        print(
                            f"    Warning: Could not find point for connector {vessel_name} "
                            f"GID {connector_gid}, using inlet_radius_val"
                        )
                        outlet_radius_val[vessel_name] = inlet_radius_val
                else:
                    # No GID available — fall back to inlet radius
                    outlet_radius_val[vessel_name] = inlet_radius_val
            else:
                # Regular outlet vessel: radius at the branch inlet point
                b_id = get_branch_id_from_name(vessel_name)
                if b_id not in branch_inlet_idx:
                    raise ValueError(
                        f"Branch {b_id} (vessel {vessel_name}) not found in branch_inlet_idx for junction {junc_name}"
                    )

                idx_in = branch_inlet_idx[b_id]
                outlet_radius_val[vessel_name] = float(max_inscribed_radius[idx_in])

        junction_areas[junc_name] = {
            "inlet_vessel_areas": inlet_vessel_areas,
            "outlet_vessel_areas": outlet_vessel_areas,
            "outlet_path_lengths": outlet_path_lengths,
            "inlet_tangent": inlet_tangent,
            "outlet_tangents": outlet_tangents,
            "outlet_tortuosities": outlet_tortuosities,
            "inlet_max_inscribed_radius": inlet_radius_val,
            "outlet_max_inscribed_radius": outlet_radius_val,
            "max_inscribed_radius_min_on_path": outlet_max_inscribed_radius_min_on_path,
            "max_inscribed_radius_max_on_path": outlet_max_inscribed_radius_max_on_path,
            "outlet_angle_diffs": outlet_angle_diffs,
            "outlet_path_gids": outlet_path_gids,
        }

        print(
            f"  {junc_name}: {len(inlet_vessel_areas)} inlet vessels, "
            f"{len(outlet_vessel_areas)} outlet vessels, "
            f"{len(outlet_path_lengths)} outlet path-length entries"
        )

    return {"vessels": vessel_areas, "junctions": junction_areas}


def add_geometric_params_to_config(zerod_config_path, geometric_areas_dict, output_path=None):
    """
    Add geometric_params field to each vessel and junction in the 0D config file.

    Args:
        zerod_config_path: Path to 0D configuration JSON file
        geometric_areas_dict: Dictionary returned by extract_vessel_junction_areas()
        output_path: Optional output path. If None, overwrites input file.

    Returns:
        Modified config dictionary
    """
    print(f"Reading 0D config from: {zerod_config_path}")
    with open(zerod_config_path, "r") as f:
        config = json.load(f)

    vessels = config.get("vessels", [])
    junctions = config.get("junctions", [])

    vessel_areas = geometric_areas_dict.get("vessels", {})
    junction_areas = geometric_areas_dict.get("junctions", {})

    # Add geometric_params to vessels
    for vessel in vessels:
        vessel_name = vessel.get("vessel_name", "")
        if not vessel_name:
            raise ValueError("Vessel with empty vessel_name found in config")

        if vessel_name not in vessel_areas:
            raise ValueError(f"No geometric data found for vessel {vessel_name}")

        areas = vessel_areas[vessel_name]

        # Use 0.0 for missing scalar numerics so the 0D solver never sees null
        # (nlohmann throws type_error.305 otherwise)
        def _num(v, default=0.0):
            return default if v is None else v

        vessel["geometric_params"] = {
            "inlet_area": _num(areas.get("inlet_area")),
            "outlet_area": _num(areas.get("outlet_area")),
            "path_length": _num(areas.get("path_length")),
            "tortuosity": _num(areas.get("tortuosity")),
            "angle_diff": _num(areas.get("angle_diff")),
            "inlet_max_inscribed_radius": _num(areas.get("inlet_max_inscribed_radius")),
            "outlet_max_inscribed_radius": _num(areas.get("outlet_max_inscribed_radius")),
            "max_inscribed_radius_min": _num(areas.get("max_inscribed_radius_min")),
            "max_inscribed_radius_max": _num(areas.get("max_inscribed_radius_max")),
        }
        print(f"  Added geometric_params to vessel {vessel_name}")

    # Add geometric_params to junctions
    for junc in junctions:
        junc_name = junc.get("junction_name", "")
        if not junc_name:
            raise ValueError("Junction with empty junction_name found in config")

        if junc_name not in junction_areas:
            raise ValueError(f"No geometric data found for junction {junc_name}")

        areas = junction_areas[junc_name]
        existing_gp = junc.get("geometric_params", {})

        # Avoid null for scalars/dicts so the 0D solver never does operator[] on null (nlohmann type_error.305)
        def _num(v, default=0.0):
            return default if v is None else v

        def _obj(v, default=None):
            if default is None:
                default = {}
            return default if v is None else v

        def _tangent(v):
            if v is None or not isinstance(v, (list, tuple)) or len(v) != 3:
                return [0.0, 0.0, 0.0]
            return list(v)

        new_gp = {
            "inlet_vessel_areas": _obj(areas.get("inlet_vessel_areas")),
            "outlet_vessel_areas": _obj(areas.get("outlet_vessel_areas")),
            "outlet_path_lengths": _obj(areas.get("outlet_path_lengths")),
            "inlet_tangent": _tangent(areas.get("inlet_tangent")),
            "outlet_tangents": _obj(areas.get("outlet_tangents")),
            "outlet_tortuosities": _obj(areas.get("outlet_tortuosities")),
            "inlet_max_inscribed_radius": _num(areas.get("inlet_max_inscribed_radius")),
            "outlet_max_inscribed_radius": _obj(areas.get("outlet_max_inscribed_radius")),
            "max_inscribed_radius_min_on_path": _obj(areas.get("max_inscribed_radius_min_on_path")),
            "max_inscribed_radius_max_on_path": _obj(areas.get("max_inscribed_radius_max_on_path")),
            "outlet_angle_diffs": _obj(areas.get("outlet_angle_diffs")),
            "outlet_path_gids": _obj(areas.get("outlet_path_gids")),
        }
        existing_gp.update(new_gp)
        junc["geometric_params"] = existing_gp
        print(f"  Added geometric_params to junction {junc_name}")

    # Write output: ensure no JSON null so the 0D solver (nlohmann) never sees type_error.305
    if output_path is None:
        output_path = zerod_config_path

    def _no_none(obj):
        if obj is None:
            return 0.0
        if isinstance(obj, dict):
            return {k: _no_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_no_none(v) for v in obj]
        return obj

    config_clean = _no_none(config)
    print(f"Writing updated config to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(config_clean, f, indent=2)

    return config


def extract_and_add_geometric_params(centerline_soln_path, config_path, output_path=None):
    """
    Extract centerline areas and write geometric_params onto a 0D geometric config.

    Args:
        centerline_soln_path: Path to centerline solution VTP file
        config_path: Geometric 0D input JSON whose vessel/junction topology defines
            centerline sampling (e.g. bifurcations or bifurcations_EL variant).
        output_path: Optional output path. If None, overwrites config_path.

    Returns:
        Modified config dictionary
    """
    print("=" * 60)
    print("Extracting geometric parameters (inlet/outlet areas)")
    print("=" * 60)

    geometric_areas_dict = extract_vessel_junction_areas(centerline_soln_path, config_path)

    print("\n" + "=" * 60)
    print("Adding geometric parameters to 0D config")
    print("=" * 60)

    config = add_geometric_params_to_config(config_path, geometric_areas_dict, output_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return config
