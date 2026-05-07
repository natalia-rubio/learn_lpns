import re
import numpy as np
import copy
import json
import csv
from util.zerod_calibration.file_io import read_centerline_vtp
from util.zerod_calibration.centerline_path_extraction import get_path_length_from_gid_list
from util.zerod_calibration.junction_block_connectivity import (
    ensure_block_connectivity_from_ids,
    resolve_junction_vessel_ids,
)


def junction_uses_block_connectivity(junc):
    """
    True if this junction uses ``inlet_blocks`` / ``outlet_blocks`` (svZeroDPlus: mutually
    exclusive with ``inlet_vessels`` / ``outlet_vessels`` for the same junction).
    """
    ib = junc.get("inlet_blocks")
    ob = junc.get("outlet_blocks")
    return bool(ib) and bool(ob)


def junction_outlet_count(junc):
    ov = junc.get("outlet_vessels")
    if ov:
        return len(ov)
    ob = junc.get("outlet_blocks")
    return len(ob) if ob else 0


def _junction_name_set(junctions):
    return {str(j["junction_name"]) for j in junctions if j.get("junction_name")}


def _find_downstream_junction_for_vessel(merged_vessel_id, merged_vessel_name, junctions):
    """First junction whose inlet references this vessel (by id or vessel name in inlet_blocks)."""
    for scan_junc in junctions:
        if merged_vessel_id in scan_junc.get("inlet_vessels", []):
            return scan_junc
        for ib in scan_junc.get("inlet_blocks") or []:
            if str(ib) == merged_vessel_name:
                return scan_junc
    return None


def split_junctions(geometric_input, centerline_data):
    """
    Split junctions with more than 2 outlets into cascading bifurcations.
    
    For each multi-outlet junction:
    - The main outlet is chosen from pre-computed in-junction path lengths when available,
      otherwise from branchId = inlet branchId + 1 (fallback).
    - Side outlets are ordered by in-junction path length; the cascade uses
      junction-to-junction trunk wiring (``{J}_bif{i}``) with no synthetic connector vessels.
    - Order of branching is determined by centerline geometry (Path coordinate).
    
    Args:
        geometric_input: Dictionary with 0D model structure (vessels, junctions, boundary_conditions)
        centerline_data: Dictionary with centerline arrays (Points, BranchId, Path, etc.)
    
    Returns:
        Modified geometric_input with only bifurcations (2-outlet junctions)
    """
    import copy
    
    # Deep copy to avoid modifying original
    result = copy.deepcopy(geometric_input)
    ensure_block_connectivity_from_ids(result, validate=True)

    vessels = result.get('vessels', [])
    junctions = result.get('junctions', [])
    boundary_conditions = result.get('boundary_conditions', [])
    
    # Create vessel lookup by index and name
    vessel_by_id = {v['vessel_id']: v for v in vessels}
    vessel_by_name = {v['vessel_name']: v for v in vessels}
    vessel_name_to_id = {v['vessel_name']: v['vessel_id'] for v in vessels}
    all_junction_names = {j.get('junction_name') for j in junctions if j.get('junction_name')}
    
    # Extract branchId from vessel name (e.g., "branch3_seg0" -> 3)
    def get_branch_id(vessel_name):
        try:
            branch_part = vessel_name.split('_')[0]  # "branch3"
            return int(branch_part.replace('branch', ''))
        except (ValueError, IndexError):
            return None
    
    # Get centerline data arrays
    branch_id_array = centerline_data.get('BranchId', None)
    path_array = centerline_data.get('Path', None)
    points_array = centerline_data.get('Points', None)
    bifurcation_id_array = centerline_data.get('BifurcationId', None)
    gid_array = centerline_data.get('GlobalNodeId', None)
    
    if branch_id_array is None or path_array is None or points_array is None:
        print("  Warning: BranchId, Path, or Points not found in centerline data, cannot determine bifurcation order")
        return result
    
    if bifurcation_id_array is None:
        print("  Warning: BifurcationId not found in centerline data, using branch-based ordering")
        bifurcation_id_array = np.full_like(branch_id_array, -1)
    
    # Convert to numpy arrays for efficient processing
    branch_id_array = np.asarray(branch_id_array)
    path_array = np.asarray(path_array)
    points_array = np.asarray(points_array)
    if gid_array is not None:
        gid_array = np.asarray(gid_array)
    
    # Helper to find centerline inlet/outlet indices for a vessel segment
    def find_vessel_centerline_points(vessel_name):
        """
        Find centerline point indices for a vessel segment based on cumulative vessel lengths.
        This properly handles cases where a branch is split into multiple segments (seg0, seg1, seg2, etc.).
        
        Returns:
            (inlet_idx, outlet_idx): Tuple of centerline point indices, or (None, None) if not found
        """
        branch_id = get_branch_id(vessel_name)
        if branch_id is None:
            return None, None
        
        # Find all centerline points belonging to this branch
        branch_mask = branch_id_array == branch_id
        branch_indices = np.where(branch_mask)[0]
        
        if len(branch_indices) == 0:
            return None, None
        
        # Sort by path coordinate (distance along centerline)
        branch_paths = path_array[branch_indices]
        sorted_idx = np.argsort(branch_paths)
        sorted_indices = branch_indices[sorted_idx]
        
        # If path_array is not available, fall back to first/last point
        if path_array is None:
            return sorted_indices[0], sorted_indices[-1]
        
        # Find all vessels on the same branch
        branch_vessels = [v for v in vessels if get_branch_id(v.get('vessel_name', '')) == branch_id]
        if not branch_vessels:
            # Fallback to first/last point of branch
            return sorted_indices[0], sorted_indices[-1]
        
        # Sort vessels by segment index (segN in the name)
        def seg_index(v):
            name = v.get('vessel_name', '')
            if '_seg' in name:
                try:
                    return int(name.split('_seg')[-1])
                except Exception:
                    return 0
            return 0
        
        branch_vessels.sort(key=seg_index)
        
        # Build cumulative lengths along the branch from vessel lengths
        lengths = [float(v.get('vessel_length', 0.0) or 0.0) for v in branch_vessels]
        if sum(lengths) <= 0:
            # Fallback to first/last point of branch
            return sorted_indices[0], sorted_indices[-1]
        
        cum_lengths = np.cumsum(lengths)
        
        # Find the index of the requested vessel in the sorted list
        vessel_idx = next((i for i, v in enumerate(branch_vessels) if v.get('vessel_name') == vessel_name), None)
        if vessel_idx is None:
            # Fallback to first/last point of branch
            return sorted_indices[0], sorted_indices[-1]
        
        # Compute target path positions
        branch_start_path = float(path_array[sorted_indices[0]])
        
        # Inlet: start of this segment
        if vessel_idx == 0:
            inlet_target_path = branch_start_path
        else:
            inlet_target_path = branch_start_path + float(cum_lengths[vessel_idx - 1])
        
        # Outlet: end of this segment
        outlet_target_path = branch_start_path + float(cum_lengths[vessel_idx])
        
        # Find nearest centerline points to target paths
        branch_paths_sorted = [float(path_array[i]) for i in sorted_indices]
        
        # Find inlet point (nearest to inlet_target_path)
        inlet_distances = [abs(p - inlet_target_path) for p in branch_paths_sorted]
        inlet_idx_in_sorted = int(np.argmin(inlet_distances))
        inlet_idx = sorted_indices[inlet_idx_in_sorted]
        
        # Find outlet point (nearest to outlet_target_path)
        outlet_distances = [abs(p - outlet_target_path) for p in branch_paths_sorted]
        outlet_idx_in_sorted = int(np.argmin(outlet_distances))
        outlet_idx = sorted_indices[outlet_idx_in_sorted]
        
        return inlet_idx, outlet_idx
    
    def annotate_vessel_node_ids(vessel):
        """
        Annotate a vessel with the GlobalNodeId of its inlet and outlet centerline points.
        """
        if gid_array is None:
            return
        vessel_name = vessel.get('vessel_name', '')
        if not vessel_name:
            return
        inlet_idx, outlet_idx = find_vessel_centerline_points(vessel_name)
        if inlet_idx is None or outlet_idx is None:
            return
        inlet_gid = int(gid_array[inlet_idx])
        outlet_gid = int(gid_array[outlet_idx])
        vessel['centerline_node_ids'] = {
            'inlet': inlet_gid,
            'outlet': outlet_gid,
        }
    
    def get_vessel_gid(vessel, which='outlet'):
        """
        Get GlobalNodeId for a vessel's inlet or outlet.
        
        Args:
            vessel: Vessel dictionary
            which: 'inlet' or 'outlet'
        
        Returns:
            GID (int) or None if not available
        """
        if gid_array is None:
            return None
        node_ids = vessel.get('centerline_node_ids', {})
        if not node_ids:
            # Try to annotate if not already done
            annotate_vessel_node_ids(vessel)
            node_ids = vessel.get('centerline_node_ids', {})
        return node_ids.get(which)
    
    # Find the inlet point (start) of each branch
    branch_inlet_point = {}
    branch_inlet_path = {}
    unique_branches = np.unique(branch_id_array)
    
    for branch in unique_branches:
        branch_mask = branch_id_array == branch
        branch_paths = path_array[branch_mask]
        branch_points = points_array[branch_mask]
        
        if len(branch_paths) > 0:
            # Inlet is at minimum path value for this branch
            min_idx = np.argmin(branch_paths)
            branch_inlet_path[int(branch)] = float(branch_paths[min_idx])
            branch_inlet_point[int(branch)] = branch_points[min_idx].copy()
    
    # Find the outlet point (end) of each branch - where it connects to downstream junction
    branch_outlet_point = {}
    for branch in unique_branches:
        branch_mask = branch_id_array == branch
        branch_paths = path_array[branch_mask]
        branch_points = points_array[branch_mask]
        
        if len(branch_paths) > 0:
            # Outlet is at maximum path value for this branch
            max_idx = np.argmax(branch_paths)
            branch_outlet_point[int(branch)] = branch_points[max_idx].copy()
    
    # Legacy: also keep branch bifurcation path for fallback
    branch_bifurcation_path = branch_inlet_path.copy()
    
    # Process junctions
    new_junctions = []
    new_vessels = list(vessels)  # Start with existing vessels (no split-time connector vessels)
    next_junction_id = 0
    
    # First pass: find max junction ID
    for junc in junctions:
        junc_name = junc.get('junction_name', '')
        if junc_name.startswith('J'):
            try:
                jid = int(junc_name[1:])
                next_junction_id = max(next_junction_id, jid + 1)
            except ValueError:
                pass

# Annotate existing vessels with centerline node ids so GIDs are available
    if gid_array is not None:
        for v in vessels:
            annotate_vessel_node_ids(v)
    
    for junc in junctions:
        inlet_vessels, outlet_vessels = resolve_junction_vessel_ids(
            junc, vessel_name_to_id, all_junction_names
        )
        junc_name = junc.get('junction_name', '')
        junc_type = junc.get('junction_type', 'NORMAL_JUNCTION')
        
        # Skip junctions that are already bifurcations or simpler
        if len(outlet_vessels) <= 2:
            new_junctions.append(junc)
            continue
        
        if len(inlet_vessels) != 1:
            print(f"  Warning: Junction {junc_name} has {len(inlet_vessels)} inlets, keeping as-is")
            new_junctions.append(junc)
            continue
        
        inlet_vessel_id = inlet_vessels[0]
        inlet_vessel = vessel_by_id.get(inlet_vessel_id)
        inlet_gid = get_vessel_gid(inlet_vessel, 'outlet')

        if inlet_vessel is None:
            print(f"  Warning: Inlet vessel {inlet_vessel_id} not found for junction {junc_name}")
            new_junctions.append(junc)
            continue
        
        inlet_branch_id = get_branch_id(inlet_vessel['vessel_name'])
        if inlet_branch_id is None:
            print(f"  Warning: Could not parse branch ID from {inlet_vessel['vessel_name']}")
            new_junctions.append(junc)
            continue
        
        print(f"  Splitting junction {junc_name}: inlet={inlet_vessel['vessel_name']} (branch {inlet_branch_id}), {len(outlet_vessels)} outlets")
        
        # Get branch IDs for all outlets
        outlet_branch_ids = []
        outlet_id_to_branch = {}
        for outlet_id in outlet_vessels:
            outlet_vessel = vessel_by_id.get(outlet_id)
            if outlet_vessel is None:
                continue
            outlet_branch_id = get_branch_id(outlet_vessel['vessel_name'])
            if outlet_branch_id is not None:
                outlet_branch_ids.append(outlet_branch_id)
                outlet_id_to_branch[outlet_id] = outlet_branch_id
        
        # Parse junction BifurcationId from junction name (e.g., "J0" -> 0, "J1" -> 1)
        try:
            junction_bif_id = int(junc_name[1:])  # Remove 'J' prefix and convert to int
        except (ValueError, IndexError):
            print(f"    Warning: Could not parse BifurcationId from junction name {junc_name}")
            junction_bif_id = None
        
        # Compute in-junction path lengths to determine main outlet and ordering.
        # Prefer pre-computed paths from geometric_params.outlet_centerline_paths
        # (populated by centerline_path_extraction using BranchIdTmp).
        in_junction_path_lengths = {}
        geo_params = junc.get('geometric_params', {})
        outlet_cl_paths = geo_params.get('outlet_centerline_paths', {})

        if outlet_cl_paths:
            print(f"    Using pre-computed centerline paths for path lengths")
            for outlet_id in outlet_vessels:
                ov = vessel_by_id.get(outlet_id)
                if ov is None:
                    continue
                vname = ov['vessel_name']
                info = outlet_cl_paths.get(vname, {})
                gid_list = info.get('path_gids', [])
                if len(gid_list) >= 2:
                    pl = get_path_length_from_gid_list(gid_list, centerline_data)
                    ob = outlet_id_to_branch.get(outlet_id)
                    if ob is not None:
                        in_junction_path_lengths[ob] = pl

        else:
            print(
                f"    No pre-computed centerline paths found for junction {junc_name}; "
                f"using branchId fallback for main/side ordering."
            )
        # elif junction_bif_id is not None:
            # in_junction_path_lengths = compute_in_junction_path_lengths(inlet_branch_id, outlet_branch_ids, junction_bif_id)

        if in_junction_path_lengths:
            print(f"    In-junction path lengths:")
            for branch_id, path_len in sorted(in_junction_path_lengths.items(), key=lambda x: -x[1]):
                vessel_name = next((v['vessel_name'] for v in vessels if get_branch_id(v['vessel_name']) == branch_id), f"branch{branch_id}")
                print(f"      {vessel_name}: {path_len:.4f}")
        
        # Identify main outlet (longest in-junction path length)
        main_outlet_id = None
        side_outlets = []
        
        if in_junction_path_lengths:
            # Find outlet with longest path length
            max_path_length = -1
            for outlet_id in outlet_vessels:
                outlet_branch_id = outlet_id_to_branch.get(outlet_id)
                if outlet_branch_id is None:
                    continue
                path_length = in_junction_path_lengths.get(outlet_branch_id, 0)
                if path_length > max_path_length:
                    max_path_length = path_length
                    main_outlet_id = outlet_id
            
            # Remaining outlets are side outlets
            for outlet_id in outlet_vessels:
                if outlet_id != main_outlet_id:
                    side_outlets.append(outlet_id)
            
            if main_outlet_id is not None:
                main_vessel = vessel_by_id.get(main_outlet_id)
                main_branch_id = outlet_id_to_branch.get(main_outlet_id)
                print(f"    Main outlet (longest path): {main_vessel['vessel_name']} (branch {main_branch_id}, path length: {max_path_length:.4f})")
        else:
            # Fallback: use branchId = inlet_branch_id + 1
            for outlet_id in outlet_vessels:
                outlet_vessel = vessel_by_id.get(outlet_id)
                if outlet_vessel is None:
                    continue
                
                outlet_branch_id = get_branch_id(outlet_vessel['vessel_name'])
                if outlet_branch_id == inlet_branch_id + 1:
                    main_outlet_id = outlet_id
                    print(f"    Main outlet (by branchId): {outlet_vessel['vessel_name']} (branch {outlet_branch_id})")
                else:
                    side_outlets.append(outlet_id)
        # If no main outlet found, use the first outlet as main
        if main_outlet_id is None:
            main_outlet_id = outlet_vessels[0]
            side_outlets = outlet_vessels[1:]
            main_vessel = vessel_by_id.get(main_outlet_id)
            print(f"    Warning: Could not determine main outlet, using {main_vessel['vessel_name']} as main outlet")
        
        # Sort side outlets by their in-junction path length (shortest first = branches off first)
        def get_in_junction_path_length(outlet_id):
            outlet_branch_id = outlet_id_to_branch.get(outlet_id)
            if outlet_branch_id is None:
                return float('inf')
            return in_junction_path_lengths.get(outlet_branch_id, float('inf'))
        
        side_outlets.sort(key=get_in_junction_path_length)
        
        for i, outlet_id in enumerate(side_outlets):
            outlet_vessel = vessel_by_id.get(outlet_id)
            outlet_branch_id = outlet_id_to_branch.get(outlet_id) if outlet_vessel else None
            path_len = in_junction_path_lengths.get(outlet_branch_id, float('inf')) if outlet_branch_id else float('inf')
            print(f"    Side outlet {i+1}: {outlet_vessel['vessel_name'] if outlet_vessel else outlet_id} (in-junction path length: {path_len:.4f})")
        
        # Create cascading bifurcations (junction-to-junction trunk; no connector vessels).
        # New junctions: {junc_name}_bif{i}; trunk outlet is the child junction name until the last hop.
        inlet_vessel_name = inlet_vessel['vessel_name']
        orig_junc_inlet_gid = junc.get('centerline_node_ids', {}).get('inlet')
        if orig_junc_inlet_gid is None:
            orig_junc_inlet_gid = inlet_gid

        for i, side_outlet_id in enumerate(side_outlets):
            is_last = (i == len(side_outlets) - 1)
            new_junc_name = f"{junc_name}_bif{i}"
            side_outlet_vessel_obj = vessel_by_id.get(side_outlet_id)
            side_name_for_gp = (
                side_outlet_vessel_obj['vessel_name']
                if side_outlet_vessel_obj
                else str(side_outlet_id)
            )

            if is_last:
                main_outlet_vessel_obj = vessel_by_id.get(main_outlet_id)
                main_name_for_gp = (
                    main_outlet_vessel_obj['vessel_name']
                    if main_outlet_vessel_obj
                    else str(main_outlet_id)
                )
                if i == 0:
                    inlet_blocks = [inlet_vessel_name]
                else:
                    inlet_blocks = [f"{junc_name}_bif{i - 1}"]
                outlet_blocks = [side_name_for_gp, main_name_for_gp]
                new_junc = {
                    "junction_name": new_junc_name,
                    "junction_type": junc_type,
                    "inlet_blocks": inlet_blocks,
                    "outlet_blocks": outlet_blocks,
                    "geometric_params": {
                        "outlet_L": {
                            side_name_for_gp: 0.0,
                            main_name_for_gp: 0.0,
                        },
                        "outlet_R_poiseuille": {
                            side_name_for_gp: 0.0,
                            main_name_for_gp: 0.0,
                        },
                        "outlet_stenosis_coefficient": {
                            side_name_for_gp: 0.0,
                            main_name_for_gp: 0.0,
                        },
                    },
                }
                if gid_array is not None:
                    centerline_node_ids = {}
                    if i == 0:
                        ig = get_vessel_gid(inlet_vessel, 'outlet')
                        if ig is not None:
                            centerline_node_ids['inlet'] = ig
                    else:
                        centerline_node_ids['inlet'] = orig_junc_inlet_gid
                    outlet_gids = {}
                    outlet_vid_map = {}
                    if side_outlet_vessel_obj:
                        sn = side_outlet_vessel_obj['vessel_name']
                        sg = get_vessel_gid(side_outlet_vessel_obj, 'inlet')
                        if sg is not None:
                            outlet_gids[sn] = sg
                        outlet_vid_map[sn] = side_outlet_id
                    if main_outlet_vessel_obj:
                        mn = main_outlet_vessel_obj['vessel_name']
                        mg = get_vessel_gid(main_outlet_vessel_obj, 'inlet')
                        if mg is not None:
                            outlet_gids[mn] = mg
                        outlet_vid_map[mn] = main_outlet_id
                    if centerline_node_ids or outlet_gids:
                        new_junc['centerline_node_ids'] = centerline_node_ids
                        if outlet_gids:
                            new_junc['centerline_node_ids']['outlets'] = outlet_gids
                        if outlet_vid_map:
                            new_junc['centerline_node_ids']['outlet_vessel_ids'] = outlet_vid_map

                print(
                    f"    Created {new_junc_name}: inlet_blocks={inlet_blocks}, "
                    f"outlet_blocks={outlet_blocks} (final)"
                )
            else:
                child_junc_name = f"{junc_name}_bif{i + 1}"
                if i == 0:
                    inlet_blocks = [inlet_vessel_name]
                else:
                    inlet_blocks = [f"{junc_name}_bif{i - 1}"]
                outlet_blocks = [side_name_for_gp, child_junc_name]
                new_junc = {
                    "junction_name": new_junc_name,
                    "junction_type": junc_type,
                    "inlet_blocks": inlet_blocks,
                    "outlet_blocks": outlet_blocks,
                    "geometric_params": {
                        "outlet_L": {side_name_for_gp: 0.0, child_junc_name: 0.0},
                        "outlet_R_poiseuille": {
                            side_name_for_gp: 0.0,
                            child_junc_name: 0.0,
                        },
                        "outlet_stenosis_coefficient": {
                            side_name_for_gp: 0.0,
                            child_junc_name: 0.0,
                        },
                    },
                }
                if gid_array is not None:
                    centerline_node_ids = {}
                    if i == 0:
                        ig = get_vessel_gid(inlet_vessel, 'outlet')
                        if ig is not None:
                            centerline_node_ids['inlet'] = ig
                    else:
                        centerline_node_ids['inlet'] = orig_junc_inlet_gid
                    outlet_gids = {}
                    outlet_vid_map = {}
                    if side_outlet_vessel_obj:
                        sn = side_outlet_vessel_obj['vessel_name']
                        sg = get_vessel_gid(side_outlet_vessel_obj, 'inlet')
                        if sg is not None:
                            outlet_gids[sn] = sg
                        outlet_vid_map[sn] = side_outlet_id
                    outlet_gids[child_junc_name] = orig_junc_inlet_gid
                    if centerline_node_ids or outlet_gids:
                        new_junc['centerline_node_ids'] = centerline_node_ids
                        if outlet_gids:
                            new_junc['centerline_node_ids']['outlets'] = outlet_gids
                        if outlet_vid_map:
                            new_junc['centerline_node_ids']['outlet_vessel_ids'] = outlet_vid_map

                print(
                    f"    Created {new_junc_name}: inlet_blocks={inlet_blocks}, "
                    f"outlet_blocks={outlet_blocks} (trunk -> {child_junc_name})"
                )

            new_junctions.append(new_junc)
    
    # Update result
    result['vessels'] = new_vessels
    result['junctions'] = new_junctions
    
    # Update vessel_id in vessels to ensure they're sequential
    for i, vessel in enumerate(result['vessels']):
        vessel['vessel_id'] = i
    
    # Annotate vessels with centerline inlet/outlet node IDs (if available)
    if gid_array is not None:
        for vessel in result["vessels"]:
            annotate_vessel_node_ids(vessel)
        
        vessel_by_name_gid = {v['vessel_name']: v for v in result['vessels']}
        junction_by_name_gid = {
            j['junction_name']: j for j in result['junctions'] if j.get('junction_name')
        }
        # Refresh junction GIDs from block topology (vessel and junction-to-junction).
        for junc in result['junctions']:
            inlet_blocks = [str(x) for x in (junc.get('inlet_blocks') or [])]
            outlet_blocks = [str(x) for x in (junc.get('outlet_blocks') or [])]
            centerline_node_ids = dict(junc.get('centerline_node_ids') or {})

            if inlet_blocks:
                ib0 = inlet_blocks[0]
                if ib0 in vessel_by_name_gid:
                    inlet_v = vessel_by_name_gid[ib0]
                    ig = get_vessel_gid(inlet_v, 'outlet')
                    if ig is not None:
                        centerline_node_ids['inlet'] = ig
                elif ib0 in junction_by_name_gid:
                    up = junction_by_name_gid[ib0]
                    ig = (up.get('centerline_node_ids') or {}).get('inlet')
                    if ig is not None:
                        centerline_node_ids['inlet'] = ig

            outlet_gids = {}
            outlet_vid_map = {}
            for ob in outlet_blocks:
                if ob in vessel_by_name_gid:
                    out_v = vessel_by_name_gid[ob]
                    out_gid = get_vessel_gid(out_v, 'inlet')
                    if out_gid is not None:
                        outlet_gids[ob] = out_gid
                    outlet_vid_map[ob] = out_v['vessel_id']
                elif ob in junction_by_name_gid:
                    cj = junction_by_name_gid[ob]
                    cj_in = (cj.get('centerline_node_ids') or {}).get('inlet')
                    if cj_in is not None:
                        outlet_gids[ob] = cj_in

            if centerline_node_ids or outlet_gids:
                junc['centerline_node_ids'] = centerline_node_ids
                if outlet_gids:
                    junc['centerline_node_ids']['outlets'] = outlet_gids
                if outlet_vid_map:
                    junc['centerline_node_ids']['outlet_vessel_ids'] = outlet_vid_map
    
    # Update vessel references in junctions to match new IDs
    vessel_name_to_new_id = {v['vessel_name']: v['vessel_id'] for v in result['vessels']}
    for junc in result['junctions']:
        # Convert vessel IDs if needed (they might reference by old ID)
        # Since we kept original vessels and only added new ones, this should be OK
        pass
    
    print(f"  Split complete: {len(junctions)} junctions -> {len(new_junctions)} junctions")
    print(f"  Vessels: {len(vessels)} (unchanged; no split-time connector vessels)")
    
    return result


def split_junctions_from_files(
    geometric_input_path, centerline_path, output_path=None, verbose=False
):
    """
    Load geometry and centerline files, split multi-outlet junctions, and save result.
    
    Args:
        geometric_input_path: Path to geometric input JSON
        centerline_path: Path to centerline VTP file
        output_path: Path to save modified geometry (default: overwrite input)
        verbose: Reserved for parity with callers (e.g. ``generate_zerod_inputs``); the
            legacy ``split_junctions`` implementation on this branch does not yet emit verbose logs.
    
    Returns:
        Modified geometric input dictionary
    """
    print(f"\nSplitting multi-outlet junctions...")
    print(f"  Geometric input: {geometric_input_path}")
    print(f"  Centerline: {centerline_path}")
    
    # Load geometric input
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    # Load centerline
    centerline_data, _ = read_centerline_vtp(centerline_path)
    
    # Split junctions
    result = split_junctions(geometric_input, centerline_data)
    
    # Save result
    if output_path is None:
        output_path = geometric_input_path
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"  Saved to: {output_path}")
    
    return result


def convert_el_normal_junctions_to_blood_vessel_junction(config):
    """
    Convert NORMAL_JUNCTIONs to BloodVesselJunction in an EL-adjusted config and assign
    junction_values from geometric_params (outlet_L, outlet_R_poiseuille, outlet_stenosis_coefficient).
    Only multi-outlet junctions (2+ outlets) are converted; single-outlet remain NORMAL_JUNCTION.
    Modifies config in place.
    """
    vessels = config.get("vessels", [])
    vessel_id_to_name = {v["vessel_id"]: v["vessel_name"] for v in vessels}

    for junc in config.get("junctions", []):
        # Order must match outlet_blocks / outlet_vessels; keys in geometric_params use the same
        # strings (vessel_name or child junction_name for J–J). Do not use resolve_junction_vessel_ids
        # here — it is vessel-only and raises on junction block names.
        if junc.get("outlet_blocks"):
            outlet_names = [str(x) for x in junc["outlet_blocks"]]
        elif junc.get("outlet_vessels"):
            outlet_names = [vessel_id_to_name.get(int(vid), "") for vid in junc["outlet_vessels"]]
        else:
            if junc.get("junction_type") == "internal_junction":
                junc["junction_type"] = "NORMAL_JUNCTION"
            continue

        if len(outlet_names) < 2:
            if junc.get("junction_type") == "internal_junction":
                junc["junction_type"] = "NORMAL_JUNCTION"
            continue

        gp = junc.get("geometric_params", {})
        outlet_L = gp.get("outlet_L", {})
        outlet_R = gp.get("outlet_R_poiseuille", {})
        outlet_S = gp.get("outlet_stenosis_coefficient", {})
        L_vals = [outlet_L.get(name, 0.0) for name in outlet_names]
        R_vals = [outlet_R.get(name, 0.0) for name in outlet_names]
        S_vals = [outlet_S.get(name, 0.0) for name in outlet_names]

        junc["junction_type"] = "BloodVesselJunction"
        junc["junction_values"] = {
            "L": L_vals,
            "R_poiseuille": R_vals,
            "stenosis_coefficient": S_vals,
        }


def adjust_junction_boundaries_by_entrance_length_from_files(geometric_input_path, centerline_path, output_path=None, verbose=False):
    """
    Load geometry and centerline files, adjust junction boundaries by entrance length, and save result.
    
    This is an alternative to the standard junction splitting method. Instead of ending junctions
    at the centerline definition, junctions extend a distance EL (entrance length) down each
    outlet vessel, where EL = 10 * MaximumInscribedSphereRadius.
    
    The saved file (bifurcations_EL_geometric_input.json) has NORMAL_JUNCTIONs converted to
    BloodVesselJunction with junction_values taken from geometric_params (outlet_L,
    outlet_R_poiseuille, outlet_stenosis_coefficient) so the geometric input can be run directly
    as a forward simulation.
    
    Args:
        geometric_input_path: Path to geometric input JSON
        centerline_path: Path to centerline VTP file
        output_path: Path to save modified geometry (default: overwrite input)
        verbose: If True, print detailed information about the adjustment process
    
    Returns:
        Modified geometric input dictionary
    """
    if not verbose:
        print(f"\nAdjusting junction boundaries by entrance length...")
        print(f"  Geometric input: {geometric_input_path}")
        print(f"  Centerline: {centerline_path}")
    
    # Load geometric input
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    # Load centerline
    centerline_data, _ = read_centerline_vtp(centerline_path)
    
    # Adjust junction boundaries (pass verbose flag through)
    result = adjust_junction_boundaries_by_entrance_length(geometric_input, centerline_data, verbose=verbose)

    # Convert NORMAL_JUNCTIONs to BloodVesselJunction and set junction_values from geometric_params
    #import pdb; pdb.set_trace()
    convert_el_normal_junctions_to_blood_vessel_junction(result)
    
    # Save result
    if output_path is None:
        raise ValueError("Output path is required")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"  Saved to: {output_path}")
    
    return result


def apply_mass_conserving_jj_trunk_observations_inplace(
    renamed_y_dict,
    renamed_dy_dict,
    bifurcated_geometric_input,
    *,
    overwrite=False,
):
    """
    Set internal junction-to-junction ``flow:J_parent:J_child`` from the sum of
    terminal descendant vessel flows under ``child``, and align ``pressure:*`` when a
    child outlet observation exists.

    EL 1D extraction can disagree with subtree mass balance; ``overwrite=True`` reapplies
    the same summed-leaf rule used after bifurcation rename.

    Args:
        renamed_y_dict, renamed_dy_dict: mutated in place.
        overwrite: Only fill missing keys when False; recompute whenever descendants exist when True.

    Returns:
        (n_jj_flow_set, n_jj_pressure_set)
    """
    bif_vessels = bifurcated_geometric_input.get("vessels", [])
    bif_junctions = bifurcated_geometric_input.get("junctions", [])
    bif_vessel_by_id = {v["vessel_id"]: v for v in bif_vessels}
    bif_vessel_by_name = {v["vessel_name"]: v for v in bif_vessels}

    vessel_to_inlet_junction = {}
    for junc in bif_junctions:
        for blk in junc.get("outlet_blocks") or []:
            s = str(blk)
            if s in bif_vessel_by_name:
                vessel_to_inlet_junction[s] = junc["junction_name"]
        for outlet_id in junc.get("outlet_vessels", []):
            outlet_vessel = bif_vessel_by_id.get(outlet_id)
            if outlet_vessel:
                vessel_to_inlet_junction[outlet_vessel["vessel_name"]] = junc[
                    "junction_name"
                ]

    junc_by_name = {
        str(j.get("junction_name")): j
        for j in bif_junctions
        if j.get("junction_name")
    }

    def _descendant_terminal_vessels(start_junc_name, seen=None):
        if seen is None:
            seen = set()
        if start_junc_name in seen:
            return []
        seen.add(start_junc_name)
        j = junc_by_name.get(start_junc_name)
        if j is None:
            return []
        out = []
        for blk in j.get("outlet_blocks") or []:
            s = str(blk)
            if s in bif_vessel_by_name:
                out.append(s)
            elif s in junc_by_name:
                out.extend(_descendant_terminal_vessels(s, seen))
        if not out:
            for oid in j.get("outlet_vessels", []) or []:
                v = bif_vessel_by_id.get(oid)
                if v:
                    out.append(v.get("vessel_name", ""))
        return [x for x in out if x]

    jj_flow_set = 0
    jj_pressure_set = 0
    for parent_j in junc_by_name.values():
        parent_name = str(parent_j.get("junction_name", "") or "")
        if not parent_name:
            continue
        for blk in parent_j.get("outlet_blocks") or []:
            child_name = str(blk)
            if child_name not in junc_by_name:
                continue
            jj_key = f"flow:{parent_name}:{child_name}"
            if not overwrite:
                if jj_key in renamed_y_dict and jj_key in renamed_dy_dict:
                    continue

            terminal_vessels = _descendant_terminal_vessels(child_name)
            y_terms = []
            dy_terms = []
            for vnm in terminal_vessels:
                inlet_j = vessel_to_inlet_junction.get(vnm)
                if not inlet_j:
                    continue
                leaf_key = f"flow:{inlet_j}:{vnm}"
                if leaf_key in renamed_y_dict and leaf_key in renamed_dy_dict:
                    y_terms.append(np.asarray(renamed_y_dict[leaf_key], dtype=float))
                    dy_terms.append(np.asarray(renamed_dy_dict[leaf_key], dtype=float))

            if not y_terms:
                continue

            y_sum = np.sum(np.stack(y_terms, axis=0), axis=0)
            dy_quad = np.sqrt(np.sum(np.stack([d**2 for d in dy_terms], axis=0), axis=0))
            renamed_y_dict[jj_key] = y_sum.tolist()
            renamed_dy_dict[jj_key] = dy_quad.tolist()
            jj_flow_set += 1

            p_jj_key = f"pressure:{parent_name}:{child_name}"
            apply_p = overwrite or (
                p_jj_key not in renamed_y_dict or p_jj_key not in renamed_dy_dict
            )
            if apply_p:
                child_j = junc_by_name.get(child_name, {})
                child_pressure_key = None
                for cblk in child_j.get("outlet_blocks") or []:
                    cbs = str(cblk)
                    if cbs in bif_vessel_by_name:
                        cand = f"pressure:{child_name}:{cbs}"
                        if cand in renamed_y_dict and cand in renamed_dy_dict:
                            child_pressure_key = cand
                            break
                if child_pressure_key is None:
                    for oid in child_j.get("outlet_vessels", []) or []:
                        v = bif_vessel_by_id.get(oid)
                        if not v:
                            continue
                        cand = f"pressure:{child_name}:{v.get('vessel_name', '')}"
                        if cand in renamed_y_dict and cand in renamed_dy_dict:
                            child_pressure_key = cand
                            break
                if child_pressure_key is not None:
                    renamed_y_dict[p_jj_key] = list(renamed_y_dict[child_pressure_key])
                    renamed_dy_dict[p_jj_key] = list(renamed_dy_dict[child_pressure_key])
                    jj_pressure_set += 1

    return jj_flow_set, jj_pressure_set


def rename_observations_for_bifurcations(original_observations, bifurcated_geometric_input):
    """
    Rename existing observation keys to match bifurcations-only junction naming.

    Reflects cascading split junction names (e.g. ``J0_bif0``). At the end,
    ``apply_mass_conserving_jj_trunk_observations_inplace`` fills internal J–J trunk
    flow/pressure keys where needed.

    Args:
        original_observations: Dictionary with 'y' and 'dy' observations
        bifurcated_geometric_input: Bifurcated geometric input (after splitting)

    Returns:
        New observations dict with renamed keys (deep-copied).
    """
    import copy

    new_observations = copy.deepcopy(original_observations)

    bif_vessels = bifurcated_geometric_input.get('vessels', [])
    bif_junctions = bifurcated_geometric_input.get('junctions', [])
    bif_vessel_by_id = {v['vessel_id']: v for v in bif_vessels}
    bif_vessel_by_name = {v['vessel_name']: v for v in bif_vessels}

    y_dict = new_observations.get('y', {})
    dy_dict = new_observations.get('dy', {})

    # Build mapping from vessel names to their junction connections in bifurcated geometry
    vessel_to_outlet_junction = {}  # vessel_name -> junction_name (where vessel is inlet)
    for junc in bif_junctions:
        for blk in junc.get('inlet_blocks') or []:
            s = str(blk)
            if s in bif_vessel_by_name:
                vessel_to_outlet_junction[s] = junc['junction_name']
        for inlet_id in junc.get('inlet_vessels', []):
            inlet_vessel = bif_vessel_by_id.get(inlet_id)
            if inlet_vessel:
                vessel_to_outlet_junction[inlet_vessel['vessel_name']] = junc['junction_name']

    # Build mapping from outlet vessel names to their inlet junction in bifurcated geometry
    vessel_to_inlet_junction = {}  # vessel_name -> junction_name (where vessel is outlet)
    for junc in bif_junctions:
        for blk in junc.get('outlet_blocks') or []:
            s = str(blk)
            if s in bif_vessel_by_name:
                vessel_to_inlet_junction[s] = junc['junction_name']
        for outlet_id in junc.get('outlet_vessels', []):
            outlet_vessel = bif_vessel_by_id.get(outlet_id)
            if outlet_vessel:
                vessel_to_inlet_junction[outlet_vessel['vessel_name']] = junc['junction_name']

    # Rename keys in y/dy
    print(f"  Renaming observation keys to match bifurcated junction names...")
    renamed_y_dict = {}
    renamed_dy_dict = {}

    for key, value in y_dict.items():
        parts = key.split(':')
        if len(parts) == 3:
            obs_type, first, second = parts
            new_key = key

            # Case 1: "type:vessel:junction" - vessel as inlet to junction
            if first in vessel_to_outlet_junction:
                new_junction = vessel_to_outlet_junction[first]
                if second.startswith('J') and '_bif' not in second:
                    new_key = f"{obs_type}:{first}:{new_junction}"

            # Case 2: "type:junction:vessel" - junction to outlet vessel
            elif first.startswith('J') and '_bif' not in first:
                if second in vessel_to_inlet_junction:
                    new_junction = vessel_to_inlet_junction[second]
                    new_key = f"{obs_type}:{new_junction}:{second}"

            renamed_y_dict[new_key] = value
            if key in dy_dict:
                renamed_dy_dict[new_key] = dy_dict[key]
        else:
            renamed_y_dict[key] = value
            if key in dy_dict:
                renamed_dy_dict[key] = dy_dict[key]

    # Fill missing internal J-J trunk observations (sum of descendant leaf flows). See
    # apply_mass_conserving_jj_trunk_observations_inplace — EL path uses overwrite=True.
    jj_flow_added, jj_pressure_added = apply_mass_conserving_jj_trunk_observations_inplace(
        renamed_y_dict,
        renamed_dy_dict,
        bifurcated_geometric_input,
        overwrite=False,
    )

    if jj_flow_added > 0:
        print(f"  Added {jj_flow_added} internal J-J flow observation key(s)")
    if jj_pressure_added > 0:
        print(f"  Added {jj_pressure_added} internal J-J pressure observation key(s)")

    new_observations['y'] = renamed_y_dict
    new_observations['dy'] = renamed_dy_dict
    return new_observations


def adjust_junction_boundaries_by_entrance_length(geometric_input, centerline_data, verbose=False):
    """
    Adjust junction boundaries to extend a distance EL (entrance length) down each outlet vessel.
    
    EL = 10 * MaximumInscribedSphereRadius of the outlet vessel at its inlet.

    **Bridging vs** ``*_connectorEL``:

    When the EL chain stops because the downstream hop is another **junction** (multi-outlet
    bifurcation, or sole outlet listed as a child junction block), the implementation **rewires**
    the topology junction-to-junction (removes the bridge vessel, adjusts ``inlet_blocks`` /
    ``outlet_blocks``) — **no** zero-length ``connectorEL`` bridge vessel in that situation.

    A zero-length ``*_connectorEL`` element is introduced when the graph still needs a lumped
    outlet stub and J–J rewire does not apply, including: (1) the chain reaches a **boundary
    condition** before EL is satisfied (merged segment shorter than EL); (2) a **single** outlet
    vessel shorter than EL with no J–J anchor after the short-EL branch; (3) **Case 2** (vessel
    longer than EL) where moving the junction boundary by EL consumes the **entire** vessel
    remainder (negligible new length), so the outlet is fully absorbed into the junction.

    Args:
        geometric_input: Dictionary with 0D model structure (vessels, junctions, boundary_conditions)
        centerline_data: Dictionary with centerline arrays (Points, BranchId, Path, MaximumInscribedSphereRadius, etc.)
        verbose: If True, print detailed information about the adjustment process
    
    Returns:
        Modified geometric_input with adjusted junction boundaries and connector vessels where needed
    """
    import copy
    if verbose:
        print("\n" + "="*60)
        print("Adjusting junction boundaries by entrance length (EL)")
        print("="*60)
        print("  EL = 10 * MaximumInscribedSphereRadius at outlet vessel inlet")
        print("  Strategy:")
        print("    - If vessel length >= EL: Extend junction boundary by EL, reduce vessel length")
        print("    - If vessel length < EL: Include full vessel in junction, create connector")
    
    # Deep copy to avoid modifying original
    result = copy.deepcopy(geometric_input)
    
    vessels = result.get('vessels', [])
    junctions = result.get('junctions', [])
    boundary_conditions = result.get('boundary_conditions', [])
    
    if verbose:
        print(f"\n  Found {len(vessels)} vessels and {len(junctions)} junctions")
    
    # Get centerline data arrays
    branch_id_array = centerline_data.get('BranchId', None)
    path_array = centerline_data.get('Path', None)
    points_array = centerline_data.get('Points', None)
    max_inscribed_radius = centerline_data.get('MaximumInscribedSphereRadius', None)
    bifurcation_id_array = centerline_data.get('BifurcationId', None)
    gid_array = centerline_data.get('GlobalNodeId', None)
    
    if branch_id_array is None or path_array is None or points_array is None:
        print("  Warning: Required centerline arrays not found, skipping entrance length adjustment")
        return result
    
    if max_inscribed_radius is None:
        print("  Warning: MaximumInscribedSphereRadius not found in centerline data, skipping entrance length adjustment")
        return result
    
    if verbose:
        print(f"  Centerline data: {len(branch_id_array)} points")
    
    # Convert to numpy arrays for efficient processing
    branch_id_array = np.asarray(branch_id_array)
    path_array = np.asarray(path_array)
    points_array = np.asarray(points_array)
    max_inscribed_radius = np.asarray(max_inscribed_radius)
    if gid_array is not None:
        gid_array = np.asarray(gid_array)
    
    # Create vessel lookup dictionaries for fast access
    vessel_by_name = {v['vessel_name']: v for v in vessels}
    vessel_by_id = {v['vessel_id']: v for v in vessels}
    junction_names_set = _junction_name_set(junctions)
    # Create backup mapping for summary (before any vessels are removed)
    original_vessel_by_id = vessel_by_id.copy()
    
    # Extract branchId from vessel name (e.g., "branch3_seg0" -> 3)
    def get_branch_id(vessel_name):
        """Extract branch ID from vessel name."""
        try:
            branch_part = vessel_name.split('_')[0]  # "branch3"
            return int(branch_part.replace('branch', ''))
        except (ValueError, IndexError):
            return None
    
    # Helper to find centerline points for a vessel segment
    def find_vessel_centerline_points(vessel_name):
        """
        Find centerline point indices for a vessel segment based on cumulative vessel lengths.
        This properly handles cases where a branch is split into multiple segments (seg0, seg1, seg2, etc.).
        
        Returns:
            (inlet_idx, outlet_idx): Tuple of centerline point indices, or (None, None) if not found
        """
        branch_id = get_branch_id(vessel_name)
        if branch_id is None:
            return None, None
        
        # Find all centerline points belonging to this branch
        branch_mask = branch_id_array == branch_id
        branch_indices = np.where(branch_mask)[0]
        
        if len(branch_indices) == 0:
            return None, None
        
        # Sort by path coordinate (distance along centerline)
        branch_paths = path_array[branch_indices]
        sorted_idx = np.argsort(branch_paths)
        sorted_indices = branch_indices[sorted_idx]
        
        # If path_array is not available, fall back to first/last point
        if path_array is None:
            return sorted_indices[0], sorted_indices[-1]
        
        # Find all vessels on the same branch
        branch_vessels = [v for v in vessels if get_branch_id(v.get('vessel_name', '')) == branch_id]
        if not branch_vessels:
            # Fallback to first/last point of branch
            return sorted_indices[0], sorted_indices[-1]
        
        # Sort vessels by segment index (segN in the name)
        def seg_index(v):
            name = v.get('vessel_name', '')
            if '_seg' in name:
                try:
                    return int(name.split('_seg')[-1])
                except Exception:
                    return 0
            return 0
        
        branch_vessels.sort(key=seg_index)
        
        # Build cumulative lengths along the branch from vessel lengths
        lengths = [float(v.get('vessel_length', 0.0) or 0.0) for v in branch_vessels]
        if sum(lengths) <= 0:
            # Fallback to first/last point of branch
            return sorted_indices[0], sorted_indices[-1]
        
        cum_lengths = np.cumsum(lengths)
        
        # Find the index of the requested vessel in the sorted list
        vessel_idx = next((i for i, v in enumerate(branch_vessels) if v.get('vessel_name') == vessel_name), None)
        if vessel_idx is None:
            # Fallback to first/last point of branch
            return sorted_indices[0], sorted_indices[-1]
        
        # Compute target path positions
        branch_start_path = float(path_array[sorted_indices[0]])
        
        # Inlet: start of this segment
        if vessel_idx == 0:
            inlet_target_path = branch_start_path
        else:
            inlet_target_path = branch_start_path + float(cum_lengths[vessel_idx - 1])
        
        # Outlet: end of this segment
        outlet_target_path = branch_start_path + float(cum_lengths[vessel_idx])
        
        # Find nearest centerline points to target paths
        branch_paths_sorted = [float(path_array[i]) for i in sorted_indices]
        
        # Find inlet point (nearest to inlet_target_path)
        inlet_distances = [abs(p - inlet_target_path) for p in branch_paths_sorted]
        inlet_idx_in_sorted = int(np.argmin(inlet_distances))
        inlet_idx = sorted_indices[inlet_idx_in_sorted]
        
        # Find outlet point (nearest to outlet_target_path)
        outlet_distances = [abs(p - outlet_target_path) for p in branch_paths_sorted]
        outlet_idx_in_sorted = int(np.argmin(outlet_distances))
        outlet_idx = sorted_indices[outlet_idx_in_sorted]
        
        return inlet_idx, outlet_idx
    
    def set_vessel_node_ids(vessel, inlet_idx, outlet_idx):
        """
        Store GlobalNodeId of inlet and outlet centerline points for a vessel.
        """
        if gid_array is None:
            return
        if inlet_idx is None or outlet_idx is None:
            return
        inlet_gid = int(gid_array[inlet_idx])
        outlet_gid = int(gid_array[outlet_idx])
        vessel['centerline_node_ids'] = {
            'inlet': inlet_gid,
            'outlet': outlet_gid,
        }
    
    def get_vessel_gid(vessel, which='outlet'):
        """
        Get GlobalNodeId for a vessel's inlet or outlet.
        
        Args:
            vessel: Vessel dictionary
            which: 'inlet' or 'outlet'
        
        Returns:
            GID (int) or None if not available
        """
        if gid_array is None:
            return None
        node_ids = vessel.get('centerline_node_ids', {})
        if not node_ids:
            # Try to annotate if not already done
            vessel_name = vessel.get('vessel_name', '')
            if vessel_name:
                inlet_idx, outlet_idx = find_vessel_centerline_points(vessel_name)
                if inlet_idx is not None and outlet_idx is not None:
                    set_vessel_node_ids(vessel, inlet_idx, outlet_idx)
                    node_ids = vessel.get('centerline_node_ids', {})
        return node_ids.get(which)
    
    _EL_PARAM_KEYS = ('L', 'R_poiseuille', 'stenosis_coefficient')

    def _absorb_vessel_params(junc, outlet_vessel_name, vessel, fraction=1.0):
        """Add (a fraction of) a vessel's L/R_poiseuille/stenosis_coefficient to the junction's geometric_params."""
        gp = junc.setdefault('geometric_params', {})
        zvals = vessel.get('zero_d_element_values', {})
        jname = junc.get('junction_name', '?')
        if verbose:
            print(
                f"    _absorb_vessel_params: junc={jname}, outlet={outlet_vessel_name}, "
                f"fraction={fraction}, zvals_L={zvals.get('L', 'MISSING')}, "
                f"zvals_R={zvals.get('R_poiseuille', 'MISSING')}, "
                f"zvals_S={zvals.get('stenosis_coefficient', 'MISSING')}"
            )
        for key in _EL_PARAM_KEYS:
            outlet_dict = gp.setdefault(f'outlet_{key}', {})
            old_val = outlet_dict.get(outlet_vessel_name, 0.0)
            new_val = old_val + fraction * zvals.get(key, 0.0)
            outlet_dict[outlet_vessel_name] = new_val
            if verbose:
                print(f"      outlet_{key}[{outlet_vessel_name}]: {old_val} -> {new_val}")

    def _reduce_vessel_params(vessel, fraction_remaining):
        """Scale a vessel's L/R_poiseuille/stenosis_coefficient by the remaining fraction after absorption."""
        zvals = vessel.get('zero_d_element_values', {})
        for key in _EL_PARAM_KEYS:
            zvals[key] = zvals.get(key, 0.0) * fraction_remaining

    def _rename_vessel_topology_and_gp_everywhere(old_name, new_name):
        """
        When EL renames a vessel (connectorEL), merges segments, or J–J rewiring renames an
        outlet to a downstream junction block, svZeroDPlus needs the same identifier in
        inlet_blocks/outlet_blocks and all geometric_params.centerline lookups — not only
        outlet_* poiseuille/L/S dicts.

        Keeps vessel_by_name coherent with renamed vessel dicts after in-place edits.
        """
        nonlocal vessel_by_name
        if not old_name or not new_name or old_name == new_name:
            return
        if verbose:
            print(f"    _rename_vessel_topology_and_gp_everywhere: {old_name} -> {new_name}")
        for jc in junctions:
            jnm = jc.get('junction_name', '?')
            for blk_key in ('inlet_blocks', 'outlet_blocks'):
                bl = jc.get(blk_key)
                if isinstance(bl, list):
                    jc[blk_key] = [
                        new_name if str(x) == old_name else x for x in bl
                    ]
            gp = jc.get('geometric_params')
            if isinstance(gp, dict):
                ocp = gp.get('outlet_centerline_paths')
                if isinstance(ocp, dict) and old_name in ocp:
                    ocp[new_name] = ocp.pop(old_name)
                    if verbose:
                        print(f"      {jnm}: outlet_centerline_paths[{old_name!r}] -> {new_name!r}")
                for key in _EL_PARAM_KEYS:
                    outlet_dict = gp.get(f'outlet_{key}')
                    if isinstance(outlet_dict, dict) and old_name in outlet_dict:
                        outlet_dict[new_name] = outlet_dict.pop(old_name)
                        if verbose:
                            print(f"      {jnm}: outlet_{key} key {old_name!r} -> {new_name!r}")
            cnd = jc.get('centerline_node_ids')
            if isinstance(cnd, dict):
                for nest in ('outlets', 'outlet_vessel_ids'):
                    sub = cnd.get(nest)
                    if isinstance(sub, dict) and old_name in sub:
                        sub[new_name] = sub.pop(old_name)
                        if verbose:
                            print(
                                f"      {jnm}: centerline_node_ids[{nest}] {old_name!r} -> {new_name!r}"
                            )
        vessel_by_name = {v['vessel_name']: v for v in vessels}

    def _el_rewire_jj_skip_connector(adjusting_junc, outlet_orig_vessel_name, bridge_vessel, downstream_junc):
        """Remove bridge vessel; wire adjusting_junc outlet to downstream junction (no connectorEL)."""
        jj = str(downstream_junc.get("junction_name", ""))
        if not jj:
            return False
        bridge_name = bridge_vessel.get("vessel_name", "")
        bridge_id = bridge_vessel.get("vessel_id")

        _absorb_vessel_params(adjusting_junc, outlet_orig_vessel_name, bridge_vessel, fraction=1.0)

        if bridge_vessel in vessels:
            vessels.remove(bridge_vessel)
        if bridge_id is not None and bridge_id in vessel_by_id:
            del vessel_by_id[bridge_id]

        parent_nm = str(adjusting_junc.get("junction_name", ""))

        if adjusting_junc.get("outlet_blocks"):
            ob = [str(x) for x in adjusting_junc["outlet_blocks"]]
            new_ob = []
            for x in ob:
                if x == outlet_orig_vessel_name or x == bridge_name:
                    new_ob.append(jj)
                else:
                    new_ob.append(x)
            adjusting_junc["outlet_blocks"] = new_ob
            adjusting_junc.pop("outlet_vessels", None)
        elif adjusting_junc.get("outlet_vessels"):
            ov_ids = list(adjusting_junc["outlet_vessels"])
            new_blocks = []
            for vid in ov_ids:
                if vid == bridge_id:
                    new_blocks.append(jj)
                else:
                    vrec = vessel_by_id.get(vid)
                    if vrec:
                        new_blocks.append(str(vrec["vessel_name"]))
            adjusting_junc["outlet_blocks"] = new_blocks
            adjusting_junc.pop("outlet_vessels", None)

        if downstream_junc.get("inlet_blocks"):
            ib = [str(x) for x in downstream_junc["inlet_blocks"]]
            new_ib = []
            for x in ib:
                if x == bridge_name or x == outlet_orig_vessel_name:
                    if parent_nm not in new_ib:
                        new_ib.append(parent_nm)
                else:
                    new_ib.append(x)
            downstream_junc["inlet_blocks"] = new_ib
            downstream_junc.pop("inlet_vessels", None)
        elif downstream_junc.get("inlet_vessels"):
            downstream_junc["inlet_blocks"] = [parent_nm] if parent_nm else []
            downstream_junc.pop("inlet_vessels", None)

        _rename_vessel_topology_and_gp_everywhere(outlet_orig_vessel_name, jj)
        if bridge_name and bridge_name != outlet_orig_vessel_name:
            _rename_vessel_topology_and_gp_everywhere(bridge_name, jj)

        din = (downstream_junc.get("centerline_node_ids") or {}).get("inlet")
        if din is not None:
            adjusting_junc.setdefault("centerline_node_ids", {}).setdefault("outlets", {})[jj] = din
        cnd = adjusting_junc.get("centerline_node_ids")
        if isinstance(cnd, dict):
            cnd.pop("outlet_vessel_ids", None)

        if verbose:
            print(
                f"      → EL J–J rewire: removed bridge vessel {bridge_name!r}; "
                f"{adjusting_junc.get('junction_name')} outlet -> {jj!r} (no connectorEL)"
            )
        return True

    # Track vessels that need to be removed (if any - currently not used for EL adjustment)
    vessels_to_remove = []
    # Track vessels converted to connectors (renamed and modified in-place)
    new_connector_vessels = []
    
    if verbose:
        print(f"\n  Processing {len(junctions)} junctions...")
    
    # Process each junction
    for junc in junctions:
        junction_name = junc.get('junction_name', '')

        if junction_uses_block_connectivity(junc):
            ib = junc.get('inlet_blocks') or []
            ob = junc.get('outlet_blocks') or []
            if not ib or not ob:
                if verbose:
                    print(f"  Skipping {junction_name}: missing inlet or outlet blocks")
                continue
            if len(ob) < 2:
                if verbose:
                    print(
                        f"  Skipping {junction_name}: only {len(ob)} outlet block(s), "
                        f"need at least 2 for entrance length adjustment"
                    )
                continue
            inlet0 = str(ib[0])
            if inlet0 in junction_names_set:
                if verbose:
                    print(
                        f"  Skipping {junction_name}: inlet block {inlet0!r} is a junction "
                        f"(EL radius requires a vessel inlet)"
                    )
                continue
            inlet_vessel = vessel_by_name.get(inlet0)
            if inlet_vessel is None:
                if verbose:
                    print(f"  Skipping {junction_name}: unknown inlet vessel block {inlet0!r}")
                continue
            inlet_vessel_id = inlet_vessel['vessel_id']
            vessel_outlets = []
            for blk in ob:
                s = str(blk)
                if s in junction_names_set:
                    continue
                vout = vessel_by_name.get(s)
                if vout is not None:
                    vessel_outlets.append((vout['vessel_id'], vout.get('vessel_name', ''), vout))
        else:
            inlet_vessels = junc.get('inlet_vessels', [])
            outlet_vessels = junc.get('outlet_vessels', [])
            if len(inlet_vessels) == 0 or len(outlet_vessels) == 0:
                if verbose:
                    print(f"  Skipping {junction_name}: missing inlet or outlet vessels")
                continue
            if len(outlet_vessels) < 2:
                if verbose:
                    print(
                        f"  Skipping {junction_name}: only {len(outlet_vessels)} outlet(s), "
                        f"need at least 2 for entrance length adjustment"
                    )
                continue
            inlet_vessel_id = inlet_vessels[0]
            inlet_vessel = vessel_by_id.get(inlet_vessel_id)
            if inlet_vessel is None:
                if verbose:
                    print(f"    Warning: Inlet vessel ID {inlet_vessel_id} not found")
                continue
            vessel_outlets = []
            for outlet_vessel_id in outlet_vessels:
                outlet_vessel = vessel_by_id.get(outlet_vessel_id)
                if outlet_vessel is None:
                    if verbose:
                        print(f"    Warning: Outlet vessel ID {outlet_vessel_id} not found")
                    continue
                vessel_outlets.append(
                    (outlet_vessel_id, outlet_vessel.get('vessel_name', ''), outlet_vessel)
                )

        if verbose:
            print(f"\n  Processing junction: {junction_name}")
            print(f"    Inlets: 1 (vessel {inlet_vessel.get('vessel_name', '?')}), vessel outlets: {len(vessel_outlets)}")

        for outlet_vessel_id, outlet_vessel_name, outlet_vessel in vessel_outlets:

            if verbose:
                print(f"\n    Processing outlet vessel: {outlet_vessel_name}")
            
            # Find centerline points for outlet vessel
            # These indices correspond to the inlet and outlet of the vessel along the centerline
            outlet_inlet_idx, outlet_outlet_idx = find_vessel_centerline_points(outlet_vessel_name)
            if outlet_inlet_idx is None:
                print(f"    Warning: Could not find centerline points for {outlet_vessel_name}, skipping")
                continue
            
            # Store current inlet/outlet node IDs for this vessel (may be updated below if we move the boundary)
            set_vessel_node_ids(outlet_vessel, outlet_inlet_idx, outlet_outlet_idx)
            
            # Get MaximumInscribedSphereRadius at the outlet vessel inlet (where it connects to junction)
            # This is the radius at the junction boundary
            radius_at_inlet = float(max_inscribed_radius[outlet_inlet_idx])
            
            # Calculate entrance length: EL = 10 * radius
            # This is the distance the junction should extend down the outlet vessel
            EL = 10.0 * radius_at_inlet
            
            # Get path coordinates (cumulative distance along centerline) for outlet vessel
            outlet_vessel_path_start = float(path_array[outlet_inlet_idx])
            outlet_vessel_path_end = float(path_array[outlet_outlet_idx])
            outlet_vessel_length = outlet_vessel_path_end - outlet_vessel_path_start
            
            if verbose:
                print(f"      Radius at inlet: {radius_at_inlet:.6f} cm")
                print(f"      Entrance length (EL): {EL:.6f} cm (10 * radius)")
                print(f"      Vessel length: {outlet_vessel_length:.6f} cm")
            
            # Check if vessel is shorter than EL
            if outlet_vessel_length < EL:
                if verbose:
                    print(f"      → Vessel is SHORTER than EL ({outlet_vessel_length:.6f} < {EL:.6f})")
                
                # Try to extend through a chain of vessels connected by single-outlet junctions
                # Merge vessels and remove intermediate junctions, then check if merged vessel needs further extension
                merged_vessel = outlet_vessel
                merged_vessel_id = outlet_vessel_id
                merged_vessel_name = outlet_vessel_name
                merged_path_start = outlet_vessel_path_start
                merged_path_end = outlet_vessel_path_end
                merged_length = outlet_vessel_length
                merged_vessel_names = [outlet_vessel_name]
                merged_outlet_idx = outlet_outlet_idx  # Track outlet index for connector conversion
                junctions_to_remove = []
                extension_successful = False
                vessels_merged = False  # Track if any merging occurred
                # Downstream junction when chain stops at multi-outlet or sole outlet is another junction (J–J trunk)
                el_jj_anchor = None

                # Build chain by merging vessels connected by single-outlet junctions
                while merged_length < EL:
                    downstream_junction = _find_downstream_junction_for_vessel(
                        merged_vessel_id, merged_vessel_name, junctions
                    )
                    if downstream_junction is None:
                        if verbose:
                            print(f"      → No downstream junction found for {merged_vessel_name}, stopping chain extension")
                        break

                    n_down_out = junction_outlet_count(downstream_junction)
                    if n_down_out != 1:
                        el_jj_anchor = downstream_junction
                        if verbose:
                            print(
                                f"      → Downstream junction {downstream_junction.get('junction_name', 'unknown')} "
                                f"has {n_down_out} outlet(s); stop chain (prefer J–J vs connectorEL)"
                            )
                        break

                    next_vessel = None
                    next_vessel_id = None
                    dov = downstream_junction.get('outlet_vessels') or []
                    dob = downstream_junction.get('outlet_blocks') or []
                    if len(dov) == 1:
                        next_vessel_id = dov[0]
                        next_vessel = vessel_by_id.get(next_vessel_id)
                    elif len(dob) == 1:
                        b0 = str(dob[0])
                        if b0 in junction_names_set:
                            el_jj_anchor = downstream_junction
                            if verbose:
                                print(
                                    f"      → Downstream {downstream_junction.get('junction_name')} sole outlet "
                                    f"is junction {b0!r}; stop chain (J–J)"
                                )
                            break
                        next_vessel = vessel_by_name.get(b0)
                        if next_vessel is not None:
                            next_vessel_id = next_vessel.get('vessel_id')
                    else:
                        if verbose:
                            print(
                                f"      → Downstream junction {downstream_junction.get('junction_name')} "
                                f"has no single resolved outlet, stopping chain extension"
                            )
                        break

                    if next_vessel is None:
                        if verbose:
                            print(
                                f"      → Next vessel not found after downstream "
                                f"{downstream_junction.get('junction_name', 'unknown')}, stopping chain extension"
                            )
                        break

                    next_vessel_name = next_vessel.get('vessel_name', '')
                    
                    # Check if next vessel has an outlet boundary condition
                    # If so, stop chain extension and transfer BC to merged vessel
                    next_vessel_bc = next_vessel.get('boundary_conditions', {})
                    next_vessel_outlet_bc = next_vessel_bc.get('outlet', None)
                    
                    if next_vessel_outlet_bc is not None:
                        if verbose:
                            print(f"      → Next vessel {next_vessel_name} has outlet boundary condition: {next_vessel_outlet_bc}")
                            print(f"      → Stopping chain extension and transferring BC to merged vessel")
                        
                        # Merge the next vessel into the merged vessel
                        next_inlet_idx, next_outlet_idx = find_vessel_centerline_points(next_vessel_name)
                        if next_inlet_idx is not None:
                            next_vessel_path_start = float(path_array[next_inlet_idx])
                            next_vessel_path_end = float(path_array[next_outlet_idx])
                            next_vessel_length = next_vessel_path_end - next_vessel_path_start
                            
                            # Merge vessels: combine lengths
                            merged_length += next_vessel_length
                            merged_path_end = next_vessel_path_end
                            
                            # Create merged vessel name
                            base_name = merged_vessel_name.split('_seg')[0] if '_seg' in merged_vessel_name else merged_vessel_name.split('_')[0]
                            seg_parts = []
                            for name in merged_vessel_names + [next_vessel_name]:
                                if '_seg' in name:
                                    seg_parts.append(name.split('_seg')[-1])
                            merged_vessel_name = f"{base_name}_seg{'_'.join(seg_parts)}"
                            
                            # Update merged vessel properties
                            merged_vessel['vessel_length'] = float(merged_length)
                            merged_vessel['vessel_name'] = merged_vessel_name
                            
                            # Merge zero_d_element_values
                            if 'zero_d_element_values' not in merged_vessel:
                                merged_vessel['zero_d_element_values'] = {}
                            if 'zero_d_element_values' in next_vessel:
                                merged_vessel['zero_d_element_values']['R_poiseuille'] = (
                                    merged_vessel['zero_d_element_values'].get('R_poiseuille', 0.0) +
                                    next_vessel['zero_d_element_values'].get('R_poiseuille', 0.0)
                                )
                                merged_vessel['zero_d_element_values']['L'] = (
                                    merged_vessel['zero_d_element_values'].get('L', 0.0) +
                                    next_vessel['zero_d_element_values'].get('L', 0.0)
                                )
                                merged_vessel['zero_d_element_values']['stenosis_coefficient'] = (
                                    merged_vessel['zero_d_element_values'].get('stenosis_coefficient', 0.0) +
                                    next_vessel['zero_d_element_values'].get('stenosis_coefficient', 0.0)
                                )
                                merged_vessel['zero_d_element_values']['C'] = (
                                    merged_vessel['zero_d_element_values'].get('C', 1e-10) +
                                    next_vessel['zero_d_element_values'].get('C', 1e-10)
                                )
                            
                            # Transfer boundary condition to merged vessel
                            if 'boundary_conditions' not in merged_vessel:
                                merged_vessel['boundary_conditions'] = {}
                            merged_vessel['boundary_conditions']['outlet'] = next_vessel_outlet_bc
                            
                            # Update node IDs: inlet stays at original inlet, outlet moves to next vessel's outlet
                            set_vessel_node_ids(merged_vessel, outlet_inlet_idx, next_outlet_idx)
                            merged_outlet_idx = next_outlet_idx  # Update tracked outlet index
                            
                            # Remove next vessel and junction
                            if next_vessel in vessels:
                                vessels.remove(next_vessel)
                            if next_vessel_id in vessel_by_id:
                                del vessel_by_id[next_vessel_id]
                            vessels_to_remove.append(next_vessel_id)
                            
                            if downstream_junction in junctions:
                                junctions.remove(downstream_junction)
                                if verbose:
                                    print(f"      → Removed junction: {downstream_junction.get('junction_name', 'unknown')}")
                            
                            merged_vessel_names.append(next_vessel_name)
                            vessels_merged = True
                            
                            if verbose:
                                print(f"      → Merged vessel: {merged_vessel_name} (ID: {merged_vessel_id}), total length: {merged_length:.6f} cm")
                                print(f"      → Applied outlet boundary condition: {next_vessel_outlet_bc}")
                            
                            # Check if merged vessel is now long enough (or exceeds EL)
                            if merged_length >= EL:
                                # Case 2 logic applied to the merged vessel:
                                # extend the junction boundary EL down the merged vessel,
                                # keeping the outlet at the original (BC) outlet.
                                target_path = merged_path_start + EL

                                merged_branch_id = get_branch_id(merged_vessel_name)
                                if merged_branch_id is not None:
                                    branch_mask = branch_id_array == merged_branch_id
                                    branch_indices = np.where(branch_mask)[0]
                                    branch_paths = path_array[branch_indices]

                                    # Find the first centerline point at or beyond target_path
                                    valid_mask = branch_paths >= target_path
                                    if np.any(valid_mask):
                                        target_idx_in_branch = np.where(valid_mask)[0][0]
                                        target_centerline_idx = branch_indices[target_idx_in_branch]
                                        new_junction_boundary_path = float(path_array[target_centerline_idx])

                                        # New merged vessel length: from new boundary to original outlet
                                        new_merged_length = merged_path_end - new_junction_boundary_path
                                        length_epsilon = 1e-10

                                        # If the remainder length is zero or negligible, convert to connector (same as single-vessel path)
                                        if new_merged_length <= length_epsilon:
                                            if verbose:
                                                print(f"      → New merged vessel length is ~0, converting to connector (fully absorbed into junction)")
                                            old_name = merged_vessel.get('vessel_name', '')
                                            connector_name = old_name if 'connector' in old_name.lower() else f"{old_name}_connectorEL"
                                            merged_vessel['vessel_name'] = connector_name
                                            _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=1.0)
                                            if connector_name != outlet_vessel_name:
                                                _rename_vessel_topology_and_gp_everywhere(outlet_vessel_name, connector_name)
                                            merged_vessel['vessel_length'] = 0.0
                                            if 'zero_d_element_values' not in merged_vessel:
                                                merged_vessel['zero_d_element_values'] = {}
                                            merged_vessel['zero_d_element_values']['R_poiseuille'] = 0.0
                                            merged_vessel['zero_d_element_values']['C'] = 1e-10
                                            merged_vessel['zero_d_element_values']['L'] = 0.0
                                            merged_vessel['zero_d_element_values']['stenosis_coefficient'] = 0.0
                                            set_vessel_node_ids(merged_vessel, merged_outlet_idx, merged_outlet_idx)
                                            new_connector_vessels.append(merged_vessel)
                                            if verbose:
                                                print(
                                                    f"  Junction {junction_name}: Merged vessels "
                                                    f"{', '.join(merged_vessel_names)} "
                                                    f"fully absorbed by EL, converted to {connector_name}"
                                                )
                                            extension_successful = True
                                            break
                                        # Else: remainder length is positive, shorten merged vessel by EL
                                        # Absorb proportional params from the merged vessel
                                        consumed_length = new_junction_boundary_path - merged_path_start
                                        fraction_consumed = consumed_length / merged_length if merged_length > 0 else 0.0
                                        _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=fraction_consumed)
                                        _reduce_vessel_params(merged_vessel, 1.0 - fraction_consumed)
                                        if merged_vessel_name != outlet_vessel_name:
                                            _rename_vessel_topology_and_gp_everywhere(outlet_vessel_name, merged_vessel_name)

                                        merged_vessel['vessel_length'] = float(new_merged_length)

                                        # Update node IDs:
                                        # - inlet moves to new boundary (target_centerline_idx)
                                        # - outlet stays at original merged_outlet_idx (BC location)
                                        set_vessel_node_ids(merged_vessel, target_centerline_idx, merged_outlet_idx)

                                        if verbose:
                                            print(
                                                f"      → Shortened merged vessel to reach EL: "
                                                f"{merged_length:.6f} → {new_merged_length:.6f} cm"
                                            )
                                            print(
                                                f"  Junction {junction_name}: Merged vessels "
                                                f"{', '.join(merged_vessel_names)} "
                                                f"into {merged_vessel_name} and extended to reach EL={EL:.4f} "
                                                f"with outlet BC {next_vessel_outlet_bc}"
                                            )
                                        extension_successful = True
                                        break
                                # If we couldn't find a suitable centerline point, fall through and
                                # let step 4a convert to a connector.
                                if verbose and not extension_successful:
                                    print(f"      → Could not find suitable centerline point to shorten merged vessel to EL, will fall back to connector conversion")
                            else:
                                # Merged length is still less than EL, but we hit a BC so we stop
                                # This will trigger step 4a (convert to connector)
                                if verbose:
                                    print(f"      → Merged length ({merged_length:.6f} cm) still less than EL ({EL:.6f} cm), but hit BC, stopping")
                                break
                        else:
                            if verbose:
                                print(f"      → Could not find centerline points for {next_vessel_name}, stopping chain extension")
                            break
                    
                    if verbose:
                        print(f"      → Chain extension: merging {merged_vessel_name} with {next_vessel_name} (via {downstream_junction.get('junction_name', 'unknown')})")
                    
                    # Find centerline points for next vessel
                    next_inlet_idx, next_outlet_idx = find_vessel_centerline_points(next_vessel_name)
                    if next_inlet_idx is None:
                        if verbose:
                            print(f"      → Could not find centerline points for {next_vessel_name}, stopping chain extension")
                        break
                    
                    # Get path coordinates for next vessel
                    next_vessel_path_start = float(path_array[next_inlet_idx])
                    next_vessel_path_end = float(path_array[next_outlet_idx])
                    next_vessel_length = next_vessel_path_end - next_vessel_path_start
                    
                    # Merge vessels: combine lengths and update merged vessel
                    merged_length += next_vessel_length
                    merged_path_end = next_vessel_path_end
                    
                    # Create merged vessel name (e.g., branch4_seg0_1)
                    base_name = merged_vessel_name.split('_seg')[0] if '_seg' in merged_vessel_name else merged_vessel_name.split('_')[0]
                    seg_parts = []
                    for name in merged_vessel_names + [next_vessel_name]:
                        if '_seg' in name:
                            seg_parts.append(name.split('_seg')[-1])
                    merged_vessel_name = f"{base_name}_seg{'_'.join(seg_parts)}"
                    
                    # Update merged vessel properties
                    merged_vessel['vessel_length'] = float(merged_length)
                    merged_vessel['vessel_name'] = merged_vessel_name
                    
                    # Merge zero_d_element_values (use average or sum as appropriate)
                    if 'zero_d_element_values' not in merged_vessel:
                        merged_vessel['zero_d_element_values'] = {}
                    if 'zero_d_element_values' in next_vessel:
                        # For R, L, stenosis: use sum (resistances in series)
                        merged_vessel['zero_d_element_values']['R_poiseuille'] = (
                            merged_vessel['zero_d_element_values'].get('R_poiseuille', 0.0) +
                            next_vessel['zero_d_element_values'].get('R_poiseuille', 0.0)
                        )
                        merged_vessel['zero_d_element_values']['L'] = (
                            merged_vessel['zero_d_element_values'].get('L', 0.0) +
                            next_vessel['zero_d_element_values'].get('L', 0.0)
                        )
                        merged_vessel['zero_d_element_values']['stenosis_coefficient'] = (
                            merged_vessel['zero_d_element_values'].get('stenosis_coefficient', 0.0) +
                            next_vessel['zero_d_element_values'].get('stenosis_coefficient', 0.0)
                        )
                        # For C: use parallel capacitance (sum)
                        merged_vessel['zero_d_element_values']['C'] = (
                            merged_vessel['zero_d_element_values'].get('C', 1e-10) +
                            next_vessel['zero_d_element_values'].get('C', 1e-10)
                        )
                    
                    # Transfer boundary conditions from next vessel to merged vessel (if any)
                    # This ensures BCs are preserved when vessels are merged
                    next_vessel_bc = next_vessel.get('boundary_conditions', {})
                    if next_vessel_bc:
                        if 'boundary_conditions' not in merged_vessel:
                            merged_vessel['boundary_conditions'] = {}
                        # Transfer outlet BC (inlet BC should stay with original vessel)
                        if 'outlet' in next_vessel_bc:
                            merged_vessel['boundary_conditions']['outlet'] = next_vessel_bc['outlet']
                            if verbose:
                                print(f"      → Transferred outlet BC: {next_vessel_bc['outlet']}")
                    
                    # Update node IDs: inlet stays at original inlet, outlet moves to next vessel's outlet
                    set_vessel_node_ids(merged_vessel, outlet_inlet_idx, next_outlet_idx)
                    merged_outlet_idx = next_outlet_idx  # Update tracked outlet index
                    
                    # Mark junction for removal
                    junctions_to_remove.append(downstream_junction)
                    
                    # Update connections: merged vessel now connects directly to what next_vessel was connected to
                    # Find what next_vessel connects to downstream
                    next_vessel_downstream_junction = _find_downstream_junction_for_vessel(
                        next_vessel_id, next_vessel_name, junctions
                    )
                    
                    # Remove next_vessel immediately from vessels list and lookup
                    # This prevents it from being found again in future iterations
                    if next_vessel in vessels:
                        vessels.remove(next_vessel)
                    if next_vessel_id in vessel_by_id:
                        del vessel_by_id[next_vessel_id]
                    vessels_to_remove.append(next_vessel_id)
                    
                    # Remove downstream junction immediately and update connections
                    if downstream_junction in junctions:
                        junctions.remove(downstream_junction)
                        if verbose:
                            print(f"      → Removed junction: {downstream_junction.get('junction_name', 'unknown')}")
                    
                    if next_vessel_downstream_junction is not None:
                        # Update the downstream junction to use merged_vessel_id / merged name instead of next_vessel
                        if next_vessel_downstream_junction.get('inlet_vessels'):
                            inlet_vessel_ids = next_vessel_downstream_junction.get('inlet_vessels', [])
                            if next_vessel_id in inlet_vessel_ids:
                                inlet_vessel_ids.remove(next_vessel_id)
                                if merged_vessel_id not in inlet_vessel_ids:
                                    inlet_vessel_ids.append(merged_vessel_id)
                            if verbose:
                                print(f"      → Updated downstream junction {next_vessel_downstream_junction.get('junction_name', 'unknown')}: "
                                      f"replaced vessel ID {next_vessel_id} with merged vessel ID {merged_vessel_id}")
                        elif next_vessel_downstream_junction.get('inlet_blocks'):
                            ib = [str(x) for x in next_vessel_downstream_junction['inlet_blocks']]
                            mnm = str(merged_vessel.get('vessel_name', merged_vessel_name))
                            next_vessel_downstream_junction['inlet_blocks'] = [
                                mnm if str(x) == next_vessel_name else str(x) for x in ib
                            ]
                            if verbose:
                                print(
                                    f"      → Updated downstream junction "
                                    f"{next_vessel_downstream_junction.get('junction_name', 'unknown')} inlet_blocks: "
                                    f"{next_vessel_name!r} -> {mnm!r}"
                                )
                    
                    if verbose:
                        print(f"      → Removed merged vessel: {next_vessel_name} (ID: {next_vessel_id})")
                    
                    merged_vessel_names.append(next_vessel_name)
                    vessels_merged = True  # Mark that merging occurred
                    
                    if verbose:
                        print(f"      → Merged vessel: {merged_vessel_name} (ID: {merged_vessel_id}), total length: {merged_length:.6f} cm")
                    
                    # Check if merged vessel is now long enough
                    if merged_length >= EL:
                        # Calculate how much to shorten the merged vessel to reach exactly EL
                        length_to_remove = merged_length - EL
                        if length_to_remove > 0:
                            # Need to shorten the merged vessel
                            new_merged_path_end = merged_path_end - length_to_remove
                            
                            # Find the centerline point closest to new_merged_path_end
                            merged_branch_id = get_branch_id(merged_vessel_name)
                            if merged_branch_id is not None:
                                branch_mask = branch_id_array == merged_branch_id
                                branch_indices = np.where(branch_mask)[0]
                                branch_paths = path_array[branch_indices]
                                
                                # Find the first centerline point at or before new_merged_path_end
                                valid_mask = branch_paths <= new_merged_path_end
                                if np.any(valid_mask):
                                    target_idx_in_branch = np.where(valid_mask)[0][-1]  # Last point <= target
                                    target_centerline_idx = branch_indices[target_idx_in_branch]
                                    new_junction_boundary_path = float(path_array[target_centerline_idx])
                                    
                                    # Update merged vessel length
                                    new_merged_length = new_junction_boundary_path - merged_path_start

                                    # Absorb proportional params from the merged vessel
                                    fraction_consumed = new_merged_length / merged_length if merged_length > 0 else 0.0
                                    _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=fraction_consumed)
                                    _reduce_vessel_params(merged_vessel, 1.0 - fraction_consumed)
                                    if merged_vessel_name != outlet_vessel_name:
                                        _rename_vessel_topology_and_gp_everywhere(outlet_vessel_name, merged_vessel_name)

                                    merged_vessel['vessel_length'] = float(new_merged_length)
                                    
                                    # Update node IDs: outlet moves to new boundary
                                    set_vessel_node_ids(merged_vessel, outlet_inlet_idx, target_centerline_idx)
                                    merged_outlet_idx = target_centerline_idx  # Update tracked outlet index
                                    
                                    if verbose:
                                        print(
                                            f"      → Shortened merged vessel to reach EL: "
                                            f"{merged_length:.6f} → {new_merged_length:.6f} cm"
                                        )
                                        print(
                                            f"  Junction {junction_name}: Merged vessels "
                                            f"{', '.join(merged_vessel_names)} "
                                            f"into {merged_vessel_name} and extended to reach EL={EL:.4f}"
                                        )
                                    extension_successful = True
                                    break
                        else:
                            # Merged vessel length exactly equals EL — absorb everything
                            _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=1.0)
                            _reduce_vessel_params(merged_vessel, 0.0)
                            if merged_vessel_name != outlet_vessel_name:
                                _rename_vessel_topology_and_gp_everywhere(outlet_vessel_name, merged_vessel_name)

                            if verbose:
                                print(f"      → Merged vessel length exactly equals EL: {merged_length:.6f} cm")
                                print(
                                    f"  Junction {junction_name}: Merged vessels "
                                    f"{', '.join(merged_vessel_names)} "
                                    f"into {merged_vessel_name} (length={merged_length:.4f} = EL={EL:.4f})"
                                )
                            extension_successful = True
                            break
                    
                    # Continue with merged vessel as current (same ID, but updated properties)
                    # The next iteration will look for downstream junction of merged_vessel_id,
                    # which should now find the updated downstream junction (e.g., J5)
                
                # Update vessel_by_id lookup after removals (junctions already removed immediately)
                if vessels_to_remove:
                    vessel_by_id = {v['vessel_id']: v for v in vessels if v['vessel_id'] not in vessels_to_remove}
                
                if extension_successful:
                    continue  # Skip the default connector logic below

                # Stopped before EL at a true multi-outlet junction or sole outlet is another junction: J–J, no connectorEL
                if el_jj_anchor is not None:
                    if verbose:
                        print(
                            f"      → Chain stopped at junction {el_jj_anchor.get('junction_name')}; "
                            f"EL J–J rewire (no connectorEL bridge vessel)"
                        )
                    if _el_rewire_jj_skip_connector(junc, outlet_vessel_name, merged_vessel, el_jj_anchor):
                        vessel_by_name = {v["vessel_name"]: v for v in vessels}
                        vessel_by_id = {v["vessel_id"]: v for v in vessels}
                        continue

                # If vessels were merged but we didn't reach EL, convert the merged vessel to a connector
                # This handles cases where chain extension stopped due to boundary condition, etc.
                if vessels_merged:
                    if verbose:
                        print(f"      → Vessels were merged but didn't reach EL, converting merged vessel to connector")
                    
                    # Convert the merged vessel to a connector
                    old_name = merged_vessel.get('vessel_name', '')
                    if 'connector' in old_name.lower():
                        connector_name = old_name
                        if verbose:
                            print(f"      → Merged vessel already contains 'connector', keeping name: {connector_name}")
                    else:
                        connector_name = f"{old_name}_connectorEL"
                        merged_vessel['vessel_name'] = connector_name
                        if verbose:
                            print(f"      → Renamed merged vessel: {old_name} → {connector_name}")
                    
                    # Absorb the merged vessel's full params into the junction before zeroing
                    _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=1.0)
                    if connector_name != outlet_vessel_name:
                        _rename_vessel_topology_and_gp_everywhere(outlet_vessel_name, connector_name)

                    # Set length to zero
                    merged_vessel['vessel_length'] = 0.0
                    
                    # Set parameters to minimal values
                    if 'zero_d_element_values' not in merged_vessel:
                        merged_vessel['zero_d_element_values'] = {}
                    merged_vessel['zero_d_element_values']['R_poiseuille'] = 0.0
                    merged_vessel['zero_d_element_values']['C'] = 1e-10
                    merged_vessel['zero_d_element_values']['L'] = 0.0
                    merged_vessel['zero_d_element_values']['stenosis_coefficient'] = 0.0
                    
                    # Update node IDs: both inlet and outlet at endpoint of extended junction (merged vessel outlet)
                    set_vessel_node_ids(merged_vessel, merged_outlet_idx, merged_outlet_idx)
                    
                    if verbose:
                        print(f"      → Set merged vessel length to 0.0 cm")
                        print(f"      → Set parameters: R=0, C=1e-10, L=0 (minimal resistance)")
                        print(
                            f"  Junction {junction_name}: Merged vessels "
                            f"{', '.join(merged_vessel_names)} "
                            f"into {connector_name} but didn't reach EL={EL:.4f}, converted to connector"
                        )

                    new_connector_vessels.append(merged_vessel)
                    continue  # Skip the default connector logic below
                
                # Default case: Vessel is shorter than EL and doesn't connect to single-outlet junction
                # Rename to connector and set length to zero
                if verbose:
                    print(f"      → Converting vessel to connector (standard case)")
                    print(
                        f"  Junction {junction_name}: Outlet vessel {outlet_vessel_name} "
                        f"(length={outlet_vessel_length:.4f}) "
                        f"is shorter than EL={EL:.4f}, converting to connector vessel"
                    )

                # Append "_connector" to vessel name (if not already a connector)
                old_name = outlet_vessel.get('vessel_name', '')
                if 'connector' in old_name.lower():
                    # Already a connector, don't rename
                    connector_name = old_name
                    if verbose:
                        print(f"      → Vessel already contains 'connector', keeping name: {connector_name}")
                else:
                    # Append "_connectorEL" to the end for connectors created by EL adjustment
                    connector_name = f"{old_name}_connectorEL"
                    outlet_vessel['vessel_name'] = connector_name
                    if verbose:
                        print(f"      → Renamed: {old_name} → {connector_name}")
                
                # Absorb the vessel's full params into the junction before zeroing
                _absorb_vessel_params(junc, outlet_vessel_name, outlet_vessel, fraction=1.0)
                if connector_name != old_name:
                    _rename_vessel_topology_and_gp_everywhere(old_name, connector_name)

                # Set length to zero (vessel is now just a connection point)
                outlet_vessel['vessel_length'] = 0.0
                
                # Set parameters to minimal values (no resistance, minimal capacitance)
                if 'zero_d_element_values' not in outlet_vessel:
                    outlet_vessel['zero_d_element_values'] = {}
                outlet_vessel['zero_d_element_values']['R_poiseuille'] = 0.0
                outlet_vessel['zero_d_element_values']['C'] = 1e-10
                outlet_vessel['zero_d_element_values']['L'] = 0.0
                outlet_vessel['zero_d_element_values']['stenosis_coefficient'] = 0.0
                
                if verbose:
                    print(f"      → Set length: {outlet_vessel_length:.6f} → 0.0 cm")
                    print(f"      → Set parameters: R=0, C=1e-10, L=0 (minimal resistance)")
                    print(f"      → No rewiring needed - connections remain unchanged")
                
                # Track that we converted this vessel (for summary)
                # Note: We keep the vessel (just renamed and modified), so don't add to vessels_to_remove
                new_connector_vessels.append(outlet_vessel)  # Track as converted connector
                
                # For connector created here: both inlet and outlet at vessel endpoint (endpoint of extended junction)
                set_vessel_node_ids(outlet_vessel, outlet_outlet_idx, outlet_outlet_idx)

                if verbose:
                    print(f"    Converted vessel to connector: {connector_name} (ID: {outlet_vessel_id})")
            else:
                # Case 2: Vessel is longer than EL - extend junction boundary by EL distance
                if verbose:
                    print(f"      → Vessel is LONGER than EL ({outlet_vessel_length:.6f} >= {EL:.6f})")
                    print(f"      → Extending junction boundary by EL={EL:.6f} cm down the vessel")
                
                # Calculate target path coordinate: start of vessel + EL distance
                # This is where the new junction boundary should be
                target_path = outlet_vessel_path_start + EL
                
                if verbose:
                    print(f"      → Target path coordinate: {target_path:.6f} cm (start={outlet_vessel_path_start:.6f} + EL={EL:.6f})")
                
                # Find the centerline point closest to target_path
                outlet_branch_id = get_branch_id(outlet_vessel_name)
                if outlet_branch_id is None:
                    if verbose:
                        print(f"      → Warning: Could not extract branch ID from {outlet_vessel_name}")
                    continue
                
                # Get all centerline points for this branch
                branch_mask = branch_id_array == outlet_branch_id
                branch_indices = np.where(branch_mask)[0]
                branch_paths = path_array[branch_indices]
                
                if verbose:
                    print(f"      → Found {len(branch_indices)} centerline points for branch {outlet_branch_id}")
                
                # Find the first centerline point at or beyond the target path
                # This point marks the new junction boundary
                valid_mask = branch_paths >= target_path
                if np.any(valid_mask):
                    # Use the first point at or beyond target_path
                    target_idx_in_branch = np.where(valid_mask)[0][0]
                    target_centerline_idx = branch_indices[target_idx_in_branch]
                    new_junction_boundary_path = float(path_array[target_centerline_idx])
                    
                    if verbose:
                        print(f"      → Found boundary point at path={new_junction_boundary_path:.6f} cm")
                        print(f"      → Centerline point index: {target_centerline_idx}")
                    
                    # Calculate new vessel length: from new boundary to original outlet
                    # The part from vessel start to new boundary is now part of the junction
                    new_vessel_length = outlet_vessel_path_end - new_junction_boundary_path
                    length_epsilon = 1e-10  # Treat as zero if within floating-point noise
                    
                    if verbose:
                        print(f"      → Original vessel length: {outlet_vessel_length:.6f} cm")
                        print(f"      → New vessel length: {new_vessel_length:.6f} cm")
                        print(f"      → Length included in junction: {new_junction_boundary_path - outlet_vessel_path_start:.6f} cm")
                    
                    # If the new length is zero or negligible, the whole vessel was absorbed into the junction:
                    # convert to connectorEL (same as "vessel shorter than EL" case) so downstream code treats it as a connector.
                    if new_vessel_length <= length_epsilon:
                        if verbose:
                            print(f"      → New vessel length is ~0, converting to connector (fully absorbed into junction)")
                        old_name = outlet_vessel.get('vessel_name', '')
                        connector_name = old_name if 'connector' in old_name.lower() else f"{old_name}_connectorEL"
                        outlet_vessel['vessel_name'] = connector_name
                        _absorb_vessel_params(junc, outlet_vessel_name, outlet_vessel, fraction=1.0)
                        if connector_name != outlet_vessel_name:
                            _rename_vessel_topology_and_gp_everywhere(outlet_vessel_name, connector_name)
                        outlet_vessel['vessel_length'] = 0.0
                        if 'zero_d_element_values' not in outlet_vessel:
                            outlet_vessel['zero_d_element_values'] = {}
                        outlet_vessel['zero_d_element_values']['R_poiseuille'] = 0.0
                        outlet_vessel['zero_d_element_values']['C'] = 1e-10
                        outlet_vessel['zero_d_element_values']['L'] = 0.0
                        outlet_vessel['zero_d_element_values']['stenosis_coefficient'] = 0.0
                        set_vessel_node_ids(outlet_vessel, outlet_outlet_idx, outlet_outlet_idx)
                        new_connector_vessels.append(outlet_vessel)
                        if verbose:
                            print(
                                f"  Junction {junction_name}: Outlet {outlet_vessel_name} fully absorbed "
                                f"by EL, converted to {connector_name}"
                            )
                        continue
                    
                    # Absorb proportional params into the junction
                    consumed_length = new_junction_boundary_path - outlet_vessel_path_start
                    fraction_consumed = consumed_length / outlet_vessel_length if outlet_vessel_length > 0 else 0.0
                    _absorb_vessel_params(junc, outlet_vessel_name, outlet_vessel, fraction=fraction_consumed)
                    _reduce_vessel_params(outlet_vessel, 1.0 - fraction_consumed)

                    # Update vessel length in the geometric input
                    # The vessel now starts at the new boundary point
                    old_length = outlet_vessel.get('vessel_length', 0.0)
                    outlet_vessel['vessel_length'] = float(new_vessel_length)
                    
                    if verbose:
                        print(f"      → Updated vessel length: {old_length:.6f} → {new_vessel_length:.6f} cm")
                    
                    # Update node IDs for this vessel:
                    # - New inlet is at target_centerline_idx
                    # - Outlet remains at original outlet_outlet_idx
                    set_vessel_node_ids(outlet_vessel, target_centerline_idx, outlet_outlet_idx)

                    if verbose:
                        print(
                            f"  Junction {junction_name}: Extended boundary by EL={EL:.4f} for outlet "
                            f"{outlet_vessel_name}, new vessel length={new_vessel_length:.4f}"
                        )
                else:
                    # Edge case: EL extends beyond the vessel (shouldn't happen if logic is correct)
                    # This means target_path > outlet_vessel_path_end
                    if verbose:
                        print(f"      → Warning: EL={EL:.6f} extends beyond vessel end (path={outlet_vessel_path_end:.6f})")
                        print(f"      → This should not happen if vessel length >= EL")
                        print(f"      → Treating as full vessel inclusion")
                    
                    if verbose:
                        print(
                            f"  Warning: EL={EL:.4f} extends beyond vessel {outlet_vessel_name} "
                            f"(length={outlet_vessel_length:.4f}), treating as full vessel inclusion"
                        )
                    _absorb_vessel_params(junc, outlet_vessel_name, outlet_vessel, fraction=1.0)
                    _reduce_vessel_params(outlet_vessel, 0.0)
                    # Set vessel length to very small value (effectively removing it)
                    outlet_vessel['vessel_length'] = 0.01
    
    # Remove vessels that were merged or fully included in junctions (if any)
    # Note: Vessels shorter than EL are converted in-place (renamed and modified),
    # so they remain in the vessels list and don't need to be removed or re-added
    result['vessels'] = [v for v in vessels if v['vessel_id'] not in vessels_to_remove]
    
    # Create mapping from old vessel ID to new sequential ID (0, 1, 2, ...)
    # This is necessary because extract_vessel_junction_areas uses vessel IDs as array indices
    old_id_to_new_id = {}
    for new_id, vessel in enumerate(result['vessels']):
        old_id = vessel['vessel_id']
        old_id_to_new_id[old_id] = new_id
        vessel['vessel_id'] = new_id  # Renumber to be sequential

    # Rebuild vessel_by_id mapping to reflect renumbered vessels
    vessel_by_id = {v['vessel_id']: v for v in result['vessels']}
    
    # Update junctions: remove references to deleted vessels and remap remaining IDs
    cleaned_junctions = []
    for junc in junctions:
        # Clean up and remap inlet_vessels list
        cleaned_inlets = []
        if 'inlet_vessels' in junc:
            cleaned_inlets = [old_id_to_new_id[vid] for vid in junc['inlet_vessels'] if vid in old_id_to_new_id]
            junc['inlet_vessels'] = cleaned_inlets
        
        # Clean up and remap outlet_vessels list
        cleaned_outlets = []
        if 'outlet_vessels' in junc:
            cleaned_outlets = [old_id_to_new_id[vid] for vid in junc['outlet_vessels'] if vid in old_id_to_new_id]
            junc['outlet_vessels'] = cleaned_outlets

        vessel_ids_ok = len(cleaned_inlets) > 0 and len(cleaned_outlets) > 0
        # Connector-free / J-J bifurcation JSON often has only inlet_blocks/outlet_blocks
        # (no inlet_vessels/outlet_vessels). Do not drop those when vessel-ID cleanup is empty.
        ib = junc.get('inlet_blocks')
        ob = junc.get('outlet_blocks')
        block_topology_ok = (
            isinstance(ib, list)
            and isinstance(ob, list)
            and len(ib) > 0
            and len(ob) > 0
        )

        # Only keep junctions that still describe a consistent topology after cleanup
        if vessel_ids_ok or block_topology_ok:
            cleaned_junctions.append(junc)
        elif verbose:
            print(
                f"  Warning: Removed junction {junc.get('junction_name', 'unknown')} - "
                f"no valid vessels or block lists after cleanup"
            )
    
    result['junctions'] = cleaned_junctions
    
    # Ensure all vessels have centerline inlet/outlet node IDs if possible
    if gid_array is not None:
        for vessel in result['vessels']:
            if 'centerline_node_ids' not in vessel:
                vessel_name = vessel.get('vessel_name', '')
                inlet_idx, outlet_idx = find_vessel_centerline_points(vessel_name)
                if inlet_idx is not None and outlet_idx is not None:
                    set_vessel_node_ids(vessel, inlet_idx, outlet_idx)
        
        # Add/update GIDs for all junctions: inlet GID from inlet vessel outlet, outlet GIDs from outlet vessel inlets
        for junc in result['junctions']:
            inlet_vessels = junc.get('inlet_vessels', [])
            outlet_vessels = junc.get('outlet_vessels', [])
            
            centerline_node_ids = {}
            
            # Get inlet GID from inlet vessel outlet
            if inlet_vessels:
                inlet_vessel_id = inlet_vessels[0]
                inlet_vessel = vessel_by_id.get(inlet_vessel_id)
                if inlet_vessel:
                    inlet_gid = get_vessel_gid(inlet_vessel, 'outlet')
                    if inlet_gid is not None:
                        centerline_node_ids['inlet'] = inlet_gid
            
            # Get outlet GIDs and vessel ID mapping from outlet vessel inlets
            outlet_gids = {}
            outlet_vid_map = {}
            for outlet_id in outlet_vessels:
                outlet_vessel = vessel_by_id.get(outlet_id)
                if outlet_vessel:
                    outlet_vessel_name = outlet_vessel.get('vessel_name', '')
                    outlet_gid = get_vessel_gid(outlet_vessel, 'inlet')
                    if outlet_gid is not None:
                        outlet_gids[outlet_vessel_name] = outlet_gid
                    outlet_vid_map[outlet_vessel_name] = outlet_id
            
            if centerline_node_ids or outlet_gids:
                if 'centerline_node_ids' not in junc:
                    junc['centerline_node_ids'] = {}
                if 'inlet' in centerline_node_ids:
                    junc['centerline_node_ids']['inlet'] = centerline_node_ids['inlet']
                if outlet_gids:
                    junc['centerline_node_ids']['outlets'] = outlet_gids
                if outlet_vid_map:
                    junc['centerline_node_ids']['outlet_vessel_ids'] = outlet_vid_map
    
    # Note: Converted connector vessels are already in result['vessels'] since they were
    # modified in-place (not removed and recreated)
    
    if verbose:
        print("\n" + "="*60)
        print("Summary of adjustments:")
        print("="*60)
        print(f"  Junctions processed: {len(junctions)}")
        print(f"  Vessels converted to connectors: {len(new_connector_vessels)}")
        if new_connector_vessels:
            connector_names = [v.get('vessel_name', 'unknown') for v in new_connector_vessels]
            print(f"    Converted vessels: {', '.join(connector_names)}")
        if vessels_to_remove:
            print(f"  Vessels removed: {len(vessels_to_remove)}")
            # Get names from original vessel mapping (before removals)
            removed_names = []
            for vid in vessels_to_remove:
                if vid in original_vessel_by_id:
                    removed_names.append(original_vessel_by_id[vid].get('vessel_name', f'ID_{vid}'))
                else:
                    removed_names.append(f'ID_{vid}')
            print(f"    Removed vessels: {', '.join(removed_names)}")
        print(f"  Total vessels after adjustment: {len(result['vessels'])} (was {len(vessels)})")
        print("="*60)
    else:
        print(f"  Adjusted {len(junctions)} junctions based on entrance length")
        if new_connector_vessels:
            print(f"  Converted {len(new_connector_vessels)} vessels to connectors (length=0)")
        if vessels_to_remove:
            print(f"  Removed {len(vessels_to_remove)} vessels")
    
    return result
