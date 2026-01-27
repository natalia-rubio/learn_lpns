#!/usr/bin/env python3
"""
Functions to extract and add geometric parameters to 0D vessels and junctions.

For each vessel we currently store:
  - inlet_area, outlet_area                      (from CenterlineSectionArea at segment ends)
  - path_length                                  (0D vessel_length from geometric input)
  - tortuosity                                   (path_length / straight_distance between ends)

For each junction we currently store:
  - inlet_vessel_areas, outlet_vessel_areas      (areas at the vessel/junction interfaces)
  - outlet_path_lengths                          (in‑junction path length from inlet to each outlet,
                                                  identical to the metric used in bifurcation_splitting)
  - inlet_tangent, outlet_tangents               (unit direction vectors at inlet / outlet sides)
  - outlet_tortuosities                          (in‑junction path_length / straight distance inlet→outlet)
  - inlet_max_inscribed_radius                   (MaximumInscribedSphereRadius at inlet branch outlet)
  - outlet_max_inscribed_radius                  (MaximumInscribedSphereRadius at each outlet branch inlet)
  - radius_min_on_path, radius_max_on_path       (min / max MaximumInscribedSphereRadius along the
                                                  path between inlet and each outlet; this uses the
                                                  same in‑junction segments as in bifurcation_splitting,
                                                  plus the inlet/outlet points themselves)
"""

import json
import numpy as np
from util.zerod_calibration.file_io import read_centerline_vtp


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
    
    # Read geometric input to understand vessel/junction structure
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    
    # Get area array from centerline
    area = centerline_data.get('CenterlineSectionArea', None)
    if area is None:
        raise ValueError("CenterlineSectionArea array not found in centerline solution")
    
    branch_id = centerline_data.get('BranchId', None)
    if branch_id is None:
        raise ValueError("BranchId array not found in centerline solution")
    branch_id = np.asarray(branch_id)
    
    path_arr = centerline_data.get('Path', None)
    gid = centerline_data.get('GlobalNodeId', None)
    points_array = centerline_data.get('Points', None)
    if points_array is None:
        raise ValueError("Points array not found in centerline solution")
    points_array = np.asarray(points_array)

    bifurcation_id_array = centerline_data.get('BifurcationId', None)
    if bifurcation_id_array is not None:
        bifurcation_id_array = np.asarray(bifurcation_id_array)
    else:
        print("  Warning: BifurcationId not found in centerline data; "
              "junction path-length metrics will be limited.")

    max_inscribed_radius = centerline_data.get('MaximumInscribedSphereRadius', None)
    if max_inscribed_radius is not None:
        max_inscribed_radius = np.asarray(max_inscribed_radius)
    else:
        print("  Warning: MaximumInscribedSphereRadius not found in centerline data; "
              "radius-based metrics will be None.")
    
    # Find inlet (GID == 0) and outlets
    inlet_idx = None
    outlet_indices = []
    if gid is not None:
        for i in range(len(gid)):
            if gid[i] == 0:
                inlet_idx = i
            elif gid[i] > 0:
                outlet_indices.append(i)
    
    # Helper to find a centerline point corresponding to a 0D vessel segment
    # (Same logic as in oned_to_zerod.py)
    def find_point_for_vessel_segment(vessel_name, prefer_end=True):
        """
        Map a 0D vessel (e.g. 'branch3_seg1') to a point index in the centerline arrays.
        prefer_end: if True, return a point near the downstream end of the segment;
                    if False, return a point near the upstream/start of the segment.
        """
        # Parse branch index from vessel_name
        try:
            parts = vessel_name.split('_')
            branch_part = parts[0]
            branch_idx = int(branch_part.replace('branch', ''))
        except Exception:
            try:
                branch_idx = int(vessel_name.split('_')[0].replace('branch', ''))
            except Exception:
                return None
        
        # Indices on the centerline that belong to this branch
        branch_pts = [i for i, bid in enumerate(branch_id) if bid == branch_idx]
        if not branch_pts:
            return None
        
        # If we don't have a path array, fall back to simple heuristic
        if path_arr is None:
            if prefer_end:
                for i in range(len(branch_id) - 1, -1, -1):
                    if branch_id[i] == branch_idx:
                        if i in outlet_indices or gid is None:
                            return i
                for i in range(len(branch_id) - 1, -1, -1):
                    if branch_id[i] == branch_idx:
                        return i
            else:
                for i in range(len(branch_id)):
                    if branch_id[i] == branch_idx:
                        return i
            return None
        
        # Collect all 0D vessels from geometric input that belong to this branch
        branch_vessels = [v for v in vessels if v.get('vessel_name', '').startswith(f'branch{branch_idx}_')]
        if not branch_vessels:
            return branch_pts[-1] if prefer_end else branch_pts[0]
        
        # Sort vessels by segment index
        def seg_index(v):
            name = v.get('vessel_name', '')
            if '_seg' in name:
                try:
                    return int(name.split('_seg')[-1])
                except Exception:
                    return 0
            return 0
        
        branch_vessels.sort(key=seg_index)
        
        # Build cumulative lengths along the branch from the 0D vessel lengths
        lengths = [float(v.get('vessel_length', 0.0) or 0.0) for v in branch_vessels]
        if sum(lengths) <= 0:
            return branch_pts[-1] if prefer_end else branch_pts[0]
        
        cum_lengths = np.cumsum(lengths)
        
        # Determine which index in branch_vessels corresponds to the requested vessel
        idx_in_list = next((i for i, v in enumerate(branch_vessels) if v.get('vessel_name') == vessel_name), None)
        if idx_in_list is None:
            idx_in_list = len(branch_vessels) - 1 if prefer_end else 0
        
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
        vessel_name = vessel.get('vessel_name', '')
        if not vessel_name:
            continue
        
        # Find inlet and outlet points for this vessel
        inlet_point_idx = find_point_for_vessel_segment(vessel_name, prefer_end=False)
        outlet_point_idx = find_point_for_vessel_segment(vessel_name, prefer_end=True)
        
        inlet_area = None
        outlet_area = None
        
        if inlet_point_idx is not None and inlet_point_idx < len(area):
            inlet_area = float(area[inlet_point_idx])
        
        if outlet_point_idx is not None and outlet_point_idx < len(area):
            outlet_area = float(area[outlet_point_idx])
        
        # Path length and tortuosity for this vessel
        # Use geometric 0D vessel_length as the path length
        path_length = float(vessel.get('vessel_length', 0.0) or 0.0)
        tortuosity = None
        if path_length > 0.0 and inlet_point_idx is not None and outlet_point_idx is not None:
            if 0 <= inlet_point_idx < len(points_array) and 0 <= outlet_point_idx < len(points_array):
                p_in = points_array[inlet_point_idx]
                p_out = points_array[outlet_point_idx]
                straight_dist = float(np.linalg.norm(p_out - p_in))
                if straight_dist > 0.0:
                    tortuosity = path_length / straight_dist
                else:
                    tortuosity = 1.0
        
        vessel_areas[vessel_name] = {
            'inlet_area': inlet_area,
            'outlet_area': outlet_area,
            'path_length': path_length,
            'tortuosity': tortuosity
        }
        
        inlet_str = f"{inlet_area:.6f}" if inlet_area is not None else "None"
        outlet_str = f"{outlet_area:.6f}" if outlet_area is not None else "None"
        path_str = f"{path_length:.6f}" if path_length is not None else "None"
        tort_str = f"{tortuosity:.6f}" if tortuosity is not None else "None"
        print(f"  {vessel_name}: inlet_area={inlet_str}, outlet_area={outlet_str}, "
              f"path_length={path_str}, tortuosity={tort_str}")
    
    # Pre-compute inlet/outlet points and indices for each branch (used by junction metrics)
    branch_inlet_point = {}
    branch_outlet_point = {}
    branch_inlet_idx = {}
    branch_outlet_idx = {}
    if path_arr is not None:
        path_arr_np = np.asarray(path_arr)
        unique_branches = np.unique(branch_id)
        for b in unique_branches:
            mask = branch_id == b
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
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
    else:
        path_arr_np = None

    # Helper to get branch id from a vessel name like "branch3_seg0"
    def get_branch_id_from_name(vessel_name):
        try:
            branch_part = vessel_name.split('_')[0]
            return int(branch_part.replace('branch', ''))
        except Exception:
            return None

    # Helper mirroring bifurcation_splitting.compute_in_junction_path_lengths,
    # extended to also track MaximumInscribedSphereRadius along each outlet path.
    def compute_junction_outlet_metrics(inlet_branch_id, outlet_branch_ids, junction_bif_id):
        """
        Returns:
            dict: outlet_branch_id -> {
                'path_length': float,
                'radius_min': float or None,
                'radius_max': float or None,
            }
        """
        metrics = {}

        if bifurcation_id_array is None or path_arr_np is None:
            return metrics

        if inlet_branch_id not in branch_outlet_point:
            print(f"    Warning: No outlet point found for inlet branch {inlet_branch_id}")
            return metrics

        inlet_endpoint = branch_outlet_point[inlet_branch_id]

        if junction_bif_id is None:
            return metrics

        junction_mask = bifurcation_id_array == junction_bif_id
        if not np.any(junction_mask):
            print(f"    Warning: No junction region found with BifurcationId == {junction_bif_id}")
            return metrics

        junc_indices = np.where(junction_mask)[0]
        junction_paths = path_arr_np[junction_mask]
        junction_points = points_array[junction_mask]
        if max_inscribed_radius is not None:
            junction_radii = max_inscribed_radius[junction_mask]
        else:
            junction_radii = None

        sort_order = np.argsort(junction_paths)
        sorted_paths = junction_paths[sort_order]
        sorted_points = junction_points[sort_order]
        sorted_indices = junc_indices[sort_order]
        if junction_radii is not None:
            sorted_radii = junction_radii[sort_order]
        else:
            sorted_radii = None

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
                continue

            segment_path_length = float(np.max(seg_paths) - np.min(seg_paths))

            min_path_idx = int(np.argmin(seg_paths))
            startpoint = seg_points[min_path_idx]

            distance_to_inlet = float(np.linalg.norm(startpoint - inlet_endpoint))
            path_length_val = segment_path_length + distance_to_inlet

            max_path_idx = int(np.argmax(seg_paths))
            endpoint = seg_points[max_path_idx]

            best_outlet = None
            best_distance = float('inf')
            for outlet_branch_id in outlet_branch_ids:
                if outlet_branch_id not in branch_inlet_point:
                    continue
                outlet_inlet = branch_inlet_point[outlet_branch_id]
                distance = float(np.linalg.norm(endpoint - outlet_inlet))
                if distance < best_distance:
                    best_distance = distance
                    best_outlet = outlet_branch_id

            if best_outlet is None or best_distance >= 2.0:
                continue

            # Collect radius extrema for this segment (we'll add inlet/outlet points below)
            if sorted_radii is not None:
                seg_radii = sorted_radii[seg_start:seg_end]
                seg_rad_min = float(np.min(seg_radii))
                seg_rad_max = float(np.max(seg_radii))
            else:
                seg_rad_min = None
                seg_rad_max = None

            # Keep only the longest path per outlet
            if best_outlet not in metrics or path_length_val > metrics[best_outlet]['path_length']:
                metrics[best_outlet] = {
                    'path_length': path_length_val,
                    'segment_indices': seg_indices,
                    'seg_rad_min': seg_rad_min,
                    'seg_rad_max': seg_rad_max,
                }

        # Augment with inlet / outlet radii if available
        if max_inscribed_radius is not None:
            for outlet_branch_id, data in metrics.items():
                inlet_idx = branch_outlet_idx.get(inlet_branch_id)
                outlet_idx = branch_inlet_idx.get(outlet_branch_id)
                vals = []
                if data['seg_rad_min'] is not None and data['seg_rad_max'] is not None:
                    vals.extend([data['seg_rad_min'], data['seg_rad_max']])
                if inlet_idx is not None:
                    vals.append(float(max_inscribed_radius[inlet_idx]))
                if outlet_idx is not None:
                    vals.append(float(max_inscribed_radius[outlet_idx]))
                if vals:
                    data['radius_min'] = float(np.min(vals))
                    data['radius_max'] = float(np.max(vals))
                else:
                    data['radius_min'] = None
                    data['radius_max'] = None
        else:
            for data in metrics.values():
                data['radius_min'] = None
                data['radius_max'] = None

        return metrics
    
    # Extract areas and geometric metrics for junctions
    junction_areas = {}
    for junc in junctions:
        junc_name = junc.get('junction_name', '')
        if not junc_name:
            continue
        
        inlet_vessel_ids = junc.get('inlet_vessels', [])
        outlet_vessel_ids = junc.get('outlet_vessels', [])
        
        inlet_vessel_areas = {}
        outlet_vessel_areas = {}
        outlet_path_lengths = {}
        outlet_tortuosities = {}
        outlet_tangents = {}
        outlet_radius_min_on_path = {}
        outlet_radius_max_on_path = {}

        inlet_tangent = None
        inlet_radius_val = None
        inlet_branch_id = None

        # For inlet vessels: get area at outlet (end) of the vessel (where it connects to junction)
        if inlet_vessel_ids:
            inlet_id = inlet_vessel_ids[0]
            if inlet_id < len(vessels):
                inlet_vessel = vessels[inlet_id]
                inlet_name = inlet_vessel.get('vessel_name', '')
                if inlet_name:
                    inlet_branch_id = get_branch_id_from_name(inlet_name)
                    pt_idx = find_point_for_vessel_segment(inlet_name, prefer_end=True)
                    if pt_idx is not None and pt_idx < len(area):
                        inlet_vessel_areas[inlet_name] = float(area[pt_idx])

                    # Tangent at inlet side (along branch towards junction)
                    if inlet_branch_id is not None and path_arr_np is not None:
                        mask = branch_id == inlet_branch_id
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
                                inlet_tangent = (v / nrm).tolist()

                    # MaximumInscribedSphereRadius at inlet branch outlet
                    if max_inscribed_radius is not None and inlet_branch_id in branch_outlet_idx:
                        idx_out = branch_outlet_idx[inlet_branch_id]
                        inlet_radius_val = float(max_inscribed_radius[idx_out])

        # For outlet vessels: get area at inlet (start) of the vessel (where it connects to junction)
        outlet_branch_ids = []
        for vessel_id in outlet_vessel_ids:
            if vessel_id < len(vessels):
                vessel = vessels[vessel_id]
                vessel_name = vessel.get('vessel_name', '')
                if not vessel_name:
                    continue

                pt_idx = find_point_for_vessel_segment(vessel_name, prefer_end=False)
                if pt_idx is not None and pt_idx < len(area):
                    outlet_vessel_areas[vessel_name] = float(area[pt_idx])

                b_id = get_branch_id_from_name(vessel_name)
                if b_id is not None:
                    outlet_branch_ids.append(b_id)

                # Tangent at outlet side (from junction into branch)
                if b_id is not None and path_arr_np is not None:
                    mask = branch_id == b_id
                    idx = np.where(mask)[0]
                    if idx.size >= 2:
                        branch_paths = path_arr_np[idx]
                        order = np.argsort(branch_paths)
                        idx_sorted = idx[order]
                        first_idx = idx_sorted[0]
                        next_idx = idx_sorted[1]
                        v = points_array[next_idx] - points_array[first_idx]
                        nrm = np.linalg.norm(v)
                        if nrm > 0.0:
                            outlet_tangents[vessel_name] = (v / nrm).tolist()

        # In-junction path lengths and radius extrema, following bifurcation_splitting logic
        try:
            junction_bif_id = int(junc_name[1:]) if junc_name.startswith('J') else None
        except (ValueError, IndexError):
            junction_bif_id = None

        if inlet_branch_id is not None and outlet_branch_ids:
            print(f"  Computing in-junction metrics for {junc_name} "
                  f"(inlet branch {inlet_branch_id}, outlets {outlet_branch_ids})")
            outlet_metrics = compute_junction_outlet_metrics(inlet_branch_id, outlet_branch_ids, junction_bif_id)
            for vessel_id in outlet_vessel_ids:
                if vessel_id >= len(vessels):
                    continue
                vessel = vessels[vessel_id]
                vessel_name = vessel.get('vessel_name', '')
                if not vessel_name:
                    continue
                b_id = get_branch_id_from_name(vessel_name)
                if b_id is None or b_id not in outlet_metrics:
                    continue
                m = outlet_metrics[b_id]
                path_len_val = m.get('path_length')
                outlet_path_lengths[vessel_name] = path_len_val

                # Tortuosity between inlet and this outlet (junction-level)
                if inlet_branch_id in branch_outlet_point and b_id in branch_inlet_point and path_len_val is not None:
                    p_in = branch_outlet_point[inlet_branch_id]
                    p_out = branch_inlet_point[b_id]
                    straight_dist = float(np.linalg.norm(p_out - p_in))
                    if straight_dist > 0.0:
                        outlet_tortuosities[vessel_name] = float(path_len_val / straight_dist)
                    else:
                        outlet_tortuosities[vessel_name] = None
                else:
                    outlet_tortuosities[vessel_name] = None

                outlet_radius_min_on_path[vessel_name] = m.get('radius_min')
                outlet_radius_max_on_path[vessel_name] = m.get('radius_max')

        # MaximumInscribedSphereRadius at each outlet inlet point (local value)
        outlet_radius_val = {}
        if max_inscribed_radius is not None:
            for vessel_id in outlet_vessel_ids:
                if vessel_id >= len(vessels):
                    continue
                vessel = vessels[vessel_id]
                vessel_name = vessel.get('vessel_name', '')
                if not vessel_name:
                    continue
                b_id = get_branch_id_from_name(vessel_name)
                if b_id is None or b_id not in branch_inlet_idx:
                    continue
                idx_in = branch_inlet_idx[b_id]
                outlet_radius_val[vessel_name] = float(max_inscribed_radius[idx_in])

        junction_areas[junc_name] = {
            'inlet_vessel_areas': inlet_vessel_areas,
            'outlet_vessel_areas': outlet_vessel_areas,
            'outlet_path_lengths': outlet_path_lengths,
            'inlet_tangent': inlet_tangent,
            'outlet_tangents': outlet_tangents,
            'outlet_tortuosities': outlet_tortuosities,
            'inlet_max_inscribed_radius': inlet_radius_val,
            'outlet_max_inscribed_radius': outlet_radius_val,
            'radius_min_on_path': outlet_radius_min_on_path,
            'radius_max_on_path': outlet_radius_max_on_path,
        }
        
        print(f"  {junc_name}: {len(inlet_vessel_areas)} inlet vessels, "
              f"{len(outlet_vessel_areas)} outlet vessels, "
              f"{len(outlet_path_lengths)} outlet path-length entries")
    
    return {
        'vessels': vessel_areas,
        'junctions': junction_areas
    }


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
    with open(zerod_config_path, 'r') as f:
        config = json.load(f)
    
    vessels = config.get('vessels', [])
    junctions = config.get('junctions', [])
    
    vessel_areas = geometric_areas_dict.get('vessels', {})
    junction_areas = geometric_areas_dict.get('junctions', {})
    
    # Add geometric_params to vessels
    for vessel in vessels:
        vessel_name = vessel.get('vessel_name', '')
        if vessel_name in vessel_areas:
            areas = vessel_areas[vessel_name]
            vessel['geometric_params'] = {
                'inlet_area': areas.get('inlet_area'),
                'outlet_area': areas.get('outlet_area'),
                'path_length': areas.get('path_length'),
                'tortuosity': areas.get('tortuosity'),
            }
            print(f"  Added geometric_params to vessel {vessel_name}")
        else:
            # Add empty geometric_params if not found
            vessel['geometric_params'] = {
                'inlet_area': None,
                'outlet_area': None,
                'path_length': None,
                'tortuosity': None
            }
            print(f"  Warning: No area data found for vessel {vessel_name}, added empty geometric_params")
    
    # Add geometric_params to junctions
    for junc in junctions:
        junc_name = junc.get('junction_name', '')
        if junc_name in junction_areas:
            areas = junction_areas[junc_name]
            junc['geometric_params'] = {
                'inlet_vessel_areas': areas.get('inlet_vessel_areas', {}),
                'outlet_vessel_areas': areas.get('outlet_vessel_areas', {}),
                'outlet_path_lengths': areas.get('outlet_path_lengths', {}),
                'inlet_tangent': areas.get('inlet_tangent'),
                'outlet_tangents': areas.get('outlet_tangents', {}),
                'outlet_tortuosities': areas.get('outlet_tortuosities', {}),
                'inlet_max_inscribed_radius': areas.get('inlet_max_inscribed_radius'),
                'outlet_max_inscribed_radius': areas.get('outlet_max_inscribed_radius', {}),
                'radius_min_on_path': areas.get('radius_min_on_path', {}),
                'radius_max_on_path': areas.get('radius_max_on_path', {}),
            }
            print(f"  Added geometric_params to junction {junc_name}")
        else:
            # Add empty geometric_params if not found
            junc['geometric_params'] = {
                'inlet_vessel_areas': {},
                'outlet_vessel_areas': {},
                'outlet_path_lengths': {},
                'inlet_tangent': None,
                'outlet_tangents': {},
                'outlet_tortuosities': {},
                'inlet_max_inscribed_radius': None,
                'outlet_max_inscribed_radius': {},
                'radius_min_on_path': {},
                'radius_max_on_path': {},
            }
            print(f"  Warning: No area data found for junction {junc_name}, added empty geometric_params")
    
    # Write output
    if output_path is None:
        output_path = zerod_config_path
    
    print(f"Writing updated config to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


def extract_and_add_geometric_params(centerline_soln_path, geometric_input_path, zerod_config_path, output_path=None):
    """
    Convenience function that combines extract_vessel_junction_areas and add_geometric_params_to_config.
    
    Args:
        centerline_soln_path: Path to centerline solution VTP file
        geometric_input_path: Path to geometric 0D input JSON
        zerod_config_path: Path to 0D configuration JSON file to update
        output_path: Optional output path for updated config. If None, overwrites zerod_config_path.
        
    Returns:
        Modified config dictionary
    """
    print("=" * 60)
    print("Extracting geometric parameters (inlet/outlet areas)")
    print("=" * 60)
    
    # Extract areas
    geometric_areas_dict = extract_vessel_junction_areas(centerline_soln_path, geometric_input_path)
    
    print("\n" + "=" * 60)
    print("Adding geometric parameters to 0D config")
    print("=" * 60)
    
    # Add to config
    config = add_geometric_params_to_config(zerod_config_path, geometric_areas_dict, output_path)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return config

