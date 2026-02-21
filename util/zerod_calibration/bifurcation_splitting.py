import numpy as np
import copy
import json
import csv
from util.zerod_calibration.file_io import read_centerline_vtp
from util.zerod_calibration.centerline_path_extraction import get_path_length_from_gid_list



def split_junctions(geometric_input, centerline_data):
    """
    Split junctions with more than 2 outlets into cascading bifurcations.
    
    For each multi-outlet junction:
    - The main outlet is the one where branchID = inlet branchID + 1
    - A connecting vessel runs from inlet to main outlet
    - Other outlets branch off along this connecting vessel
    - Order of branching is determined by centerline geometry (Path coordinate)
    
    Args:
        geometric_input: Dictionary with 0D model structure (vessels, junctions, boundary_conditions)
        centerline_data: Dictionary with centerline arrays (Points, BranchId, Path, etc.)
    
    Returns:
        Modified geometric_input with only bifurcations (2-outlet junctions)
    """
    import copy
    
    # Deep copy to avoid modifying original
    result = copy.deepcopy(geometric_input)
    
    vessels = result.get('vessels', [])
    junctions = result.get('junctions', [])
    boundary_conditions = result.get('boundary_conditions', [])
    
    # Create vessel lookup by index and name
    vessel_by_id = {v['vessel_id']: v for v in vessels}
    vessel_by_name = {v['vessel_name']: v for v in vessels}
    
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
    
    def compute_in_junction_path_lengths(inlet_branch_id, outlet_branch_ids, junction_bif_id):
        """
        Compute in-junction path lengths for each outlet.
        
        Within a junction region (BifurcationId == junction_bif_id), there are multiple path segments,
        one for each route from inlet to outlet. The path length is max(Path) - min(Path)
        for each segment.
        
        Args:
            inlet_branch_id: BranchId of the inlet vessel
            outlet_branch_ids: List of BranchIds for outlet vessels
            junction_bif_id: The BifurcationId for this specific junction
        
        Returns:
            Dictionary mapping outlet_branch_id -> in_junction_path_length
        """
        # Find the inlet branch outlet point (where it connects to this junction)
        if inlet_branch_id not in branch_outlet_point:
            print(f"    Warning: No outlet point found for inlet branch {inlet_branch_id}")
            return {}
        
        inlet_endpoint = branch_outlet_point[inlet_branch_id]
        
        # Find all points in THIS specific junction region (BifurcationId == junction_bif_id)
        junction_mask = bifurcation_id_array == junction_bif_id
        
        if not np.any(junction_mask):
            print(f"    Warning: No junction region found with BifurcationId == {junction_bif_id}")
            return {}
        
        junction_paths = path_array[junction_mask]
        junction_points = points_array[junction_mask]
        
        print(f"    Found {len(junction_paths)} points in junction region (BifurcationId={junction_bif_id})")
        
        # Sort by Path to identify segments
        sort_order = np.argsort(junction_paths)
        sorted_paths = junction_paths[sort_order]
        sorted_points = junction_points[sort_order]
        
        # Identify distinct path segments by finding discontinuities
        # A discontinuity is where Path jumps (either backward or by a large amount)
        segments = []
        current_segment_start = 0
        
        for i in range(1, len(sorted_paths)):
            path_diff = sorted_paths[i] - sorted_paths[i-1]
            # Detect segment boundary: path jumps significantly (relative to typical increment)
            if i > 1:
                prev_diff = sorted_paths[i-1] - sorted_paths[i-2]
                if abs(path_diff) > 10 * abs(prev_diff) + 0.01:  # Significant jump
                    segments.append((current_segment_start, i))
                    current_segment_start = i
            elif path_diff < -0.001:  # Path decreased - new segment
                segments.append((current_segment_start, i))
                current_segment_start = i
        
        # Add final segment
        segments.append((current_segment_start, len(sorted_paths)))
        
        # For each segment, find its endpoint and match to outlet branch
        outlet_path_lengths = {}
        
        for seg_start, seg_end in segments:
            seg_paths = sorted_paths[seg_start:seg_end]
            seg_points = sorted_points[seg_start:seg_end]
            
            if len(seg_paths) == 0:
                print(f"    Warning: No path segments found for segment {seg_start}-{seg_end}")
                
                continue
            
            # Path length along the segment (max - min of Path values)
            segment_path_length = float(np.max(seg_paths) - np.min(seg_paths))
            
            # Start point of segment (where path is minimum)
            min_path_idx = np.argmin(seg_paths)
            startpoint = seg_points[min_path_idx]
            
            # Distance from segment start to junction inlet
            # (some segments don't start at the inlet, so we need to add this distance)
            distance_to_inlet = float(np.linalg.norm(startpoint - inlet_endpoint))
            
            # Total in-junction path length = segment path length + distance to inlet
            path_length = segment_path_length + distance_to_inlet
            
            # Endpoint of segment (where path is maximum)
            max_path_idx = np.argmax(seg_paths)
            endpoint = seg_points[max_path_idx]
            
            # Match endpoint to closest outlet branch inlet
            best_outlet = None
            best_distance = float('inf')
            
            for outlet_branch_id in outlet_branch_ids:
                if outlet_branch_id not in branch_inlet_point:
                    continue
                outlet_inlet = branch_inlet_point[outlet_branch_id]
                
                distance = np.linalg.norm(endpoint - outlet_inlet)
                print(f"    Outlet inlet: {outlet_inlet},  Endpoint: {endpoint}, Distance: {distance}")
                if distance < best_distance:
                    best_distance = distance
                    best_outlet = outlet_branch_id
            
            if best_outlet is not None and best_distance < 2.0:  # Matching threshold
                # Only keep the longest path if we already have one for this outlet
                print(f"    Best outlet: {best_outlet}, path length: {path_length:.4f} (segment: {segment_path_length:.4f} + inlet dist: {distance_to_inlet:.4f})")
                if best_outlet not in outlet_path_lengths or path_length > outlet_path_lengths[best_outlet]:
                    outlet_path_lengths[best_outlet] = path_length
        
        print(f"    Outlet path lengths: {outlet_path_lengths}")
        return outlet_path_lengths
    
    # Legacy: also keep branch bifurcation path for fallback
    branch_bifurcation_path = branch_inlet_path.copy()
    
    # Process junctions
    new_junctions = []
    new_vessels = list(vessels)  # Start with existing vessels
    next_vessel_id = max(v['vessel_id'] for v in vessels) + 1
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
        inlet_vessels = junc.get('inlet_vessels', [])
        outlet_vessels = junc.get('outlet_vessels', [])
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
            print(f"    No pre-computed centerline paths found for outlet {outlet_vessels}.")
            import pdb; pdb.set_trace()
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
        
        # Create cascading bifurcations
        # Each bifurcation has:
        # - Inlet from previous connector (or original inlet for first)
        # - One side outlet
        # - One outlet to next connector (or main outlet for last)
        #
        # Naming convention:
        # - New junctions: {original_junction}_bif{i} (e.g., J0_bif0, J0_bif1)
        # - Connector vessels: {inlet_vessel}_connector{i} (e.g., branch0_seg0_connector0)
        
        current_inlet_id = inlet_vessel_id
        inlet_vessel_name = inlet_vessel['vessel_name']
        
        for i, side_outlet_id in enumerate(side_outlets):
            is_last = (i == len(side_outlets) - 1)
            
            # Create new junction with name derived from original junction
            new_junc_name = f"{junc_name}_bif{i}"
            
            if is_last:
                # Last bifurcation: connects to main outlet
                side_outlet_vessel_obj = vessel_by_id.get(side_outlet_id)
                main_outlet_vessel_obj = vessel_by_id.get(main_outlet_id)
                side_name_for_gp = side_outlet_vessel_obj['vessel_name'] if side_outlet_vessel_obj else str(side_outlet_id)
                main_name_for_gp = main_outlet_vessel_obj['vessel_name'] if main_outlet_vessel_obj else str(main_outlet_id)

                new_junc = {
                    "inlet_vessels": [current_inlet_id],
                    "junction_name": new_junc_name,
                    "junction_type": junc_type,
                    "outlet_vessels": [side_outlet_id, main_outlet_id],
                    "geometric_params": {
                        "outlet_L": {side_name_for_gp: 0.0, main_name_for_gp: 0.0},
                        "outlet_R_poiseuille": {side_name_for_gp: 0.0, main_name_for_gp: 0.0},
                        "outlet_stenosis_coefficient": {side_name_for_gp: 0.0, main_name_for_gp: 0.0},
                    },
                }
                
                # Add GIDs to junction: inlet GID from inlet vessel outlet, outlet GIDs from outlet vessel inlets,
                
                if gid_array is not None:
                    inlet_vessel = vessel_by_id.get(current_inlet_id)
                    side_outlet_vessel = vessel_by_id.get(side_outlet_id)
                    main_outlet_vessel = vessel_by_id.get(main_outlet_id)
                    
                    centerline_node_ids = {}
                    if inlet_vessel:
                        inlet_gid = get_vessel_gid(inlet_vessel, 'outlet')
                        if inlet_gid is not None:
                            centerline_node_ids['inlet'] = inlet_gid
                    
                    outlet_gids = {}
                    outlet_vid_map = {}
                    if side_outlet_vessel:
                        side_name = side_outlet_vessel['vessel_name']
                        side_gid = get_vessel_gid(side_outlet_vessel, 'inlet')
                        if side_gid is not None:
                            outlet_gids[side_name] = side_gid
                        outlet_vid_map[side_name] = side_outlet_id
                    
                    if main_outlet_vessel:
                        main_name = main_outlet_vessel['vessel_name']
                        main_gid = get_vessel_gid(main_outlet_vessel, 'inlet')
                        if main_gid is not None:
                            outlet_gids[main_name] = main_gid
                        outlet_vid_map[main_name] = main_outlet_id
                    
                    if centerline_node_ids or outlet_gids:
                        new_junc['centerline_node_ids'] = centerline_node_ids
                        if outlet_gids:
                            new_junc['centerline_node_ids']['outlets'] = outlet_gids
                        if outlet_vid_map:
                            new_junc['centerline_node_ids']['outlet_vessel_ids'] = outlet_vid_map
                
                print(f"    Created {new_junc_name}: inlet={current_inlet_id}, outlets=[{side_outlet_id}, {main_outlet_id}] (final)")
            else:
                # Create connector vessel to next bifurcation
                # Name derived from inlet vessel name
                connector_name = f"{inlet_vessel_name}_connector{i}"
                
                # Connector vessels are artificial constructs - set R, L, stenosis to 0
                # Keep small non-zero capacitance for numerical stability
                connector_vessel = {
                    "vessel_id": next_vessel_id,
                    "vessel_length": inlet_vessel['vessel_length'] * 0.01,  # Small connector
                    "vessel_name": connector_name,
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "C": inlet_vessel['zero_d_element_values'].get('C', 1e-10) * 0.01,
                        "L": 0.0,  # No inductance for artificial connector
                        "R_poiseuille": 0.0,  # No resistance for artificial connector
                        "stenosis_coefficient": 0.0  # No stenosis for artificial connector
                    },
                    "centerline_node_ids": {
                        "inlet": inlet_gid,
                        "outlet": inlet_gid
                    }

                }
                
                new_vessels.append(connector_vessel)
                vessel_by_id[next_vessel_id] = connector_vessel
                vessel_by_name[connector_name] = connector_vessel
                
                # Create bifurcation junction
                side_outlet_vessel_obj2 = vessel_by_id.get(side_outlet_id)
                side_name_for_gp = side_outlet_vessel_obj2['vessel_name'] if side_outlet_vessel_obj2 else str(side_outlet_id)

                new_junc = {
                    "inlet_vessels": [current_inlet_id],
                    "junction_name": new_junc_name,
                    "junction_type": junc_type,
                    "outlet_vessels": [side_outlet_id, next_vessel_id],
                    "geometric_params": {
                        "outlet_L": {side_name_for_gp: 0.0, connector_name: 0.0},
                        "outlet_R_poiseuille": {side_name_for_gp: 0.0, connector_name: 0.0},
                        "outlet_stenosis_coefficient": {side_name_for_gp: 0.0, connector_name: 0.0},
                    },
                }
                
                # Add GIDs to junction: inlet GID from inlet vessel outlet, outlet GIDs from outlet vessel inlets
                if gid_array is not None:
                    inlet_vessel_obj = vessel_by_id.get(current_inlet_id)
                    side_outlet_vessel = vessel_by_id.get(side_outlet_id)
                    
                    centerline_node_ids = {}
                    if inlet_vessel_obj:
                        inlet_gid = get_vessel_gid(inlet_vessel_obj, 'outlet')
                        if inlet_gid is not None:
                            centerline_node_ids['inlet'] = inlet_gid
                    
                    outlet_gids = {}
                    outlet_vid_map = {}
                    if side_outlet_vessel:
                        side_name = side_outlet_vessel['vessel_name']
                        side_gid = get_vessel_gid(side_outlet_vessel, 'inlet')
                        if side_gid is not None:
                            outlet_gids[side_name] = side_gid
                        outlet_vid_map[side_name] = side_outlet_id
                    # Connector vessel doesn't have GIDs yet, will be set after annotation
                    outlet_gids[connector_name] = None  # Placeholder, updated after annotation
                    outlet_vid_map[connector_name] = next_vessel_id
                    
                    if centerline_node_ids or outlet_gids:
                        new_junc['centerline_node_ids'] = centerline_node_ids
                        if outlet_gids:
                            new_junc['centerline_node_ids']['outlets'] = outlet_gids
                        if outlet_vid_map:
                            new_junc['centerline_node_ids']['outlet_vessel_ids'] = outlet_vid_map
                
                print(f"    Created {new_junc_name}: inlet={current_inlet_id}, outlets=[{side_outlet_id}, {next_vessel_id}] (connector: {connector_name})")
                
                current_inlet_id = next_vessel_id
                next_vessel_id += 1
            
            new_junctions.append(new_junc)
    
    # Update result
    result['vessels'] = new_vessels
    result['junctions'] = new_junctions
    
    # Update vessel_id in vessels to ensure they're sequential
    for i, vessel in enumerate(result['vessels']):
        vessel['vessel_id'] = i
    
    # Annotate vessels with centerline inlet/outlet node IDs (if available)
    if gid_array is not None:
        for vessel in result['vessels']:
            if "connector" not in vessel['vessel_name']:
                annotate_vessel_node_ids(vessel)
        
        # Update junction GIDs now that all vessels (including connectors) are annotated
        for junc in result['junctions']:
            outlet_vessels = junc.get('outlet_vessels', [])
            inlet_vessels = junc.get('inlet_vessels', [])

            centerline_node_ids = junc.get('centerline_node_ids', {}) or {}

            # Get inlet GID from inlet vessel outlet (first inlet)
            if inlet_vessels:
                inlet_vid = inlet_vessels[0]
                inlet_v = next((v for v in result['vessels'] if v['vessel_id'] == inlet_vid), None)
                if inlet_v is not None:
                    inlet_gid = get_vessel_gid(inlet_v, 'outlet')
                    if inlet_gid is not None:
                        centerline_node_ids['inlet'] = inlet_gid

            # Get outlet GIDs and vessel ID mapping from outlet vessel inlets
            outlet_gids = {}
            outlet_vid_map = {}
            for outlet_id in outlet_vessels:
                out_v = next((v for v in result['vessels'] if v['vessel_id'] == outlet_id), None)
                if out_v is not None:
                    out_name = out_v['vessel_name']
                    out_gid = get_vessel_gid(out_v, 'inlet')
                    if out_gid is not None:
                        outlet_gids[out_name] = out_gid
                    outlet_vid_map[out_name] = outlet_id

            if outlet_gids:
                centerline_node_ids['outlets'] = outlet_gids
            if outlet_vid_map:
                centerline_node_ids['outlet_vessel_ids'] = outlet_vid_map

            if centerline_node_ids:
                junc['centerline_node_ids'] = centerline_node_ids
    
    # Update vessel references in junctions to match new IDs
    vessel_name_to_new_id = {v['vessel_name']: v['vessel_id'] for v in result['vessels']}
    for junc in result['junctions']:
        # Convert vessel IDs if needed (they might reference by old ID)
        # Since we kept original vessels and only added new ones, this should be OK
        pass
    
    print(f"  Split complete: {len(junctions)} junctions -> {len(new_junctions)} junctions")
    print(f"  Vessels: {len(vessels)} -> {len(new_vessels)}")
    
    return result


def split_junctions_from_files(geometric_input_path, centerline_path, output_path=None):
    """
    Load geometry and centerline files, split multi-outlet junctions, and save result.
    
    Args:
        geometric_input_path: Path to geometric input JSON
        centerline_path: Path to centerline VTP file
        output_path: Path to save modified geometry (default: overwrite input)
    
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


def adjust_junction_boundaries_by_entrance_length_from_files(geometric_input_path, centerline_path, output_path=None, verbose=False):
    """
    Load geometry and centerline files, adjust junction boundaries by entrance length, and save result.
    
    This is an alternative to the standard junction splitting method. Instead of ending junctions
    at the centerline definition, junctions extend a distance EL (entrance length) down each
    outlet vessel, where EL = 10 * MaximumInscribedSphereRadius.
    
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
    
    # Save result
    if output_path is None:
        raise ValueError("Output path is required")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"  Saved to: {output_path}")
    
    return result



def rename_observations_for_bifurcations(original_observations, bifurcated_geometric_input):
    """
    Rename existing observation keys to match bifurcations-only junction naming.

    This is used when we want the bifurcations-only calibration input to reference
    the new junction names (e.g. J0_bif0) even if we are not generating synthetic
    observations for connector vessels.

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

    y_dict = new_observations.get('y', {})
    dy_dict = new_observations.get('dy', {})

    # Build mapping from vessel names to their junction connections in bifurcated geometry
    vessel_to_outlet_junction = {}  # vessel_name -> junction_name (where vessel is inlet)
    for junc in bif_junctions:
        for inlet_id in junc.get('inlet_vessels', []):
            inlet_vessel = bif_vessel_by_id.get(inlet_id)
            if inlet_vessel:
                vessel_to_outlet_junction[inlet_vessel['vessel_name']] = junc['junction_name']

    # Build mapping from outlet vessel names to their inlet junction in bifurcated geometry
    vessel_to_inlet_junction = {}  # vessel_name -> junction_name (where vessel is outlet)
    for junc in bif_junctions:
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

    new_observations['y'] = renamed_y_dict
    new_observations['dy'] = renamed_dy_dict
    return new_observations


def generate_connector_observations(original_observations, original_geometric_input, 
                                     bifurcated_geometric_input, centerline_data):
    """
    Generate synthetic observations for connector vessels created during junction splitting.
    
    For connector vessels:
    - Flow: inlet_flow - sum(flows of side outlets that have already branched off)
    - Pressure: linear interpolation between inlet pressure and main outlet pressure
    
    Args:
        original_observations: Dictionary with 'y' and 'dy' observations from original geometry
        original_geometric_input: Original geometric input (before splitting)
        bifurcated_geometric_input: Bifurcated geometric input (after splitting)
        centerline_data: Centerline data with BranchId and Path arrays
    
    Returns:
        Updated observations dictionary with connector vessel observations
    """
    import copy
    
    # Deep copy observations
    new_observations = copy.deepcopy(original_observations)
    
    # Get vessel and junction info from both geometries
    orig_vessels = original_geometric_input.get('vessels', [])
    orig_junctions = original_geometric_input.get('junctions', [])
    bif_vessels = bifurcated_geometric_input.get('vessels', [])
    bif_junctions = bifurcated_geometric_input.get('junctions', [])
    
    # Build lookup tables
    orig_vessel_by_id = {v['vessel_id']: v for v in orig_vessels}
    orig_vessel_by_name = {v['vessel_name']: v for v in orig_vessels}
    bif_vessel_by_id = {v['vessel_id']: v for v in bif_vessels}
    bif_vessel_by_name = {v['vessel_name']: v for v in bif_vessels}
    
    # Rename existing observation keys to use new junction names (always)
    new_observations = rename_observations_for_bifurcations(new_observations, bifurcated_geometric_input)

    # Get y and dy dicts (post-rename)
    y_dict = new_observations.get('y', {})
    dy_dict = new_observations.get('dy', {})
    
    # Helper to extract branchId from vessel name
    def get_branch_id(vessel_name):
        try:
            branch_part = vessel_name.split('_')[0]
            return int(branch_part.replace('branch', ''))
        except (ValueError, IndexError):
            return None
    
    # NOTE: The mapping logic for renaming is now handled by rename_observations_for_bifurcations()
    
    # Get bifurcation positions from centerline
    branch_id_array = centerline_data.get('BranchId', None)
    path_array = centerline_data.get('Path', None)
    
    branch_bifurcation_path = {}
    if branch_id_array is not None and path_array is not None:
        for branch in np.unique(branch_id_array):
            branch_mask = branch_id_array == branch
            branch_paths = path_array[branch_mask]
            if len(branch_paths) > 0:
                branch_bifurcation_path[int(branch)] = float(np.min(branch_paths))
    
    # Find connector vessels (vessels that exist in bifurcated but not in original)
    orig_vessel_names = set(v['vessel_name'] for v in orig_vessels)
    connector_vessels = [v for v in bif_vessels if v['vessel_name'] not in orig_vessel_names]
    
    if not connector_vessels:
        print("  No connector vessels found, no new observations needed")
        return new_observations
    
    print(f"  Generating observations for {len(connector_vessels)} connector vessel(s)...")
    
    # For each connector vessel, find its context (what junction it's part of, what flows through it)
    for connector in connector_vessels:
        connector_name = connector['vessel_name']
        connector_id = connector['vessel_id']
        
        # Parse connector name to get inlet vessel name and connector index/variant
        # Support both splitting-created connectors ("..._connector{N}") and
        # EL-created connectors ("..._connectorEL").
        import re
        connector_match = re.match(r'(.+)_connector(?:(\d+)|EL)?$', connector_name)
        if not connector_match:
            print(f"    Warning: Could not parse connector name: {connector_name}")
            continue

        inlet_vessel_name_from_connector = connector_match.group(1)
        connector_idx_str = connector_match.group(2)
        connector_idx = int(connector_idx_str) if connector_idx_str is not None else None
        inlet_branch_id = get_branch_id(inlet_vessel_name_from_connector)
        
        # Find the junction where this connector is an outlet
        inlet_junction = None
        for junc in bif_junctions:
            if connector_id in junc.get('outlet_vessels', []):
                inlet_junction = junc
                break
        
        # Find the junction where this connector is an inlet
        outlet_junction = None
        for junc in bif_junctions:
            if connector_id in junc.get('inlet_vessels', []):
                outlet_junction = junc
                break
        
        if inlet_junction is None or outlet_junction is None:
            print(f"    Warning: Could not find junctions for connector {connector_name}")
            continue
        
        # Get the inlet vessel of the inlet junction
        inlet_vessel_id = inlet_junction['inlet_vessels'][0]
        inlet_vessel = bif_vessel_by_id.get(inlet_vessel_id)
        if inlet_vessel is None:
            print(f"    Warning: Could not find inlet vessel {inlet_vessel_id}")
            continue
        
        # Get the side outlet vessel that branches off at this junction
        side_outlet_id = None
        for out_id in inlet_junction['outlet_vessels']:
            if out_id != connector_id:
                side_outlet_id = out_id
                break
        
        side_outlet = bif_vessel_by_id.get(side_outlet_id)
        if side_outlet is None:
            print(f"    Warning: Could not find side outlet vessel {side_outlet_id}")
            continue
        
        # Find the main outlet vessel (at the end of the connector chain)
        # This is the vessel with branchId = inlet_branch_id + 1
        main_outlet = None
        for v in orig_vessels:
            if get_branch_id(v['vessel_name']) == inlet_branch_id + 1:
                main_outlet = v
                break
        
        if main_outlet is None:
            # Fall back to finding main outlet from bifurcated junctions
            # Look for the outlet vessel at the end of the connector chain
            for junc in bif_junctions:
                if connector_id in junc.get('inlet_vessels', []) or any(
                    '_connector' in bif_vessel_by_id.get(vid, {}).get('vessel_name', '')
                    for vid in junc.get('inlet_vessels', [])
                ):
                    for out_id in junc.get('outlet_vessels', []):
                        out_vessel = bif_vessel_by_id.get(out_id)
                        if out_vessel and get_branch_id(out_vessel['vessel_name']) == inlet_branch_id + 1:
                            main_outlet = out_vessel
                            break
        
        # Get inlet vessel name (could be the original inlet or a previous connector)
        inlet_vessel_name = inlet_vessel['vessel_name']
        
        # Get observation keys for flow calculation
        # Flow into connector = inlet flow - side outlet flow
        # We need to find the flow observation for the inlet vessel at its outlet
        
        # Find inlet flow observation key
        # This could be at a junction or BC
        inlet_flow_key = None
        for key in y_dict.keys():
            if key.startswith(f'flow:{inlet_vessel_name}:'):
                inlet_flow_key = key
                break
        
        # If inlet is a connector, we need to use the connector's calculated flow
        if inlet_flow_key is None and '_connector' in inlet_vessel_name:
            # This connector's flow should have been calculated already
            # Look for it in our newly added observations
            for key in y_dict.keys():
                if key.startswith(f'flow:{inlet_vessel_name}:'):
                    inlet_flow_key = key
                    break
        
        # Find side outlet flow observation key
        side_outlet_name = side_outlet['vessel_name']
        side_outlet_flow_key = None
        for key in y_dict.keys():
            if key.startswith(f'flow:{side_outlet_name}:'):
                side_outlet_flow_key = key
                break
        
        # Calculate connector flow: inlet_flow - side_outlet_flow
        connector_flow = None
        if inlet_flow_key and side_outlet_flow_key:
            inlet_flow = np.array(y_dict[inlet_flow_key])
            side_outlet_flow = np.array(y_dict[side_outlet_flow_key])
            connector_flow = inlet_flow - side_outlet_flow
            print(f"    {connector_name}: flow = {inlet_vessel_name} - {side_outlet_name}")
        elif inlet_flow_key:
            # If we don't have side outlet flow, use inlet flow directly
            connector_flow = np.array(y_dict[inlet_flow_key])
            print(f"    {connector_name}: flow = {inlet_vessel_name} (no side outlet flow found)")
        else:
            print(f"    Warning: Could not find inlet flow for {connector_name}")
        
        # Find inlet pressure observation (pressure at bifurcation inlet = inlet vessel's outlet pressure)
        inlet_pressure_key = None
        for key in y_dict.keys():
            if key.startswith(f'pressure:{inlet_vessel_name}:'):
                inlet_pressure_key = key
                break
        
        # For connectors with R=0, L=0: no pressure drop, so pressure is constant
        # Pressure at connector inlet = pressure at connector outlet = inlet vessel outlet pressure
        connector_pressure = None
        if inlet_pressure_key:
            connector_pressure = np.array(y_dict[inlet_pressure_key])
            print(f"    {connector_name}: pressure = {inlet_vessel_name} (no drop, R=L=0)")
        else:
            print(f"    Warning: Could not find inlet pressure for {connector_name}")
        
        # Add observations for the connector at both its inlet and outlet junctions
        inlet_junction_name = inlet_junction['junction_name']
        outlet_junction_name = outlet_junction['junction_name']
        
        if connector_flow is not None:
            # Observation at connector's outlet (connector -> outlet_junction)
            flow_key_outlet = f"flow:{connector_name}:{outlet_junction_name}"
            y_dict[flow_key_outlet] = connector_flow.tolist()
            if len(connector_flow) > 2:
                dy = np.gradient(connector_flow)
                dy_dict[flow_key_outlet] = dy.tolist()
            else:
                dy_dict[flow_key_outlet] = [0.0] * len(connector_flow)
            
            # Observation at connector's inlet (inlet_junction -> connector)
            flow_key_inlet = f"flow:{inlet_junction_name}:{connector_name}"
            y_dict[flow_key_inlet] = connector_flow.tolist()  # Same flow at inlet and outlet
            if len(connector_flow) > 2:
                dy = np.gradient(connector_flow)
                dy_dict[flow_key_inlet] = dy.tolist()
            else:
                dy_dict[flow_key_inlet] = [0.0] * len(connector_flow)
            
            print(f"    Added flow observations: {flow_key_inlet}, {flow_key_outlet}")
        
        if connector_pressure is not None:
            # Pressure at connector's outlet (connector -> outlet_junction)
            # Same as inlet pressure since R=0, L=0 means no pressure drop
            pressure_key_outlet = f"pressure:{connector_name}:{outlet_junction_name}"
            y_dict[pressure_key_outlet] = connector_pressure.tolist()
            if len(connector_pressure) > 2:
                dy = np.gradient(connector_pressure)
                dy_dict[pressure_key_outlet] = dy.tolist()
            else:
                dy_dict[pressure_key_outlet] = [0.0] * len(connector_pressure)
            
            # Pressure at connector's inlet (inlet_junction -> connector)
            # Same pressure as outlet (no drop)
            pressure_key_inlet = f"pressure:{inlet_junction_name}:{connector_name}"
            y_dict[pressure_key_inlet] = connector_pressure.tolist()
            if len(connector_pressure) > 2:
                dy = np.gradient(connector_pressure)
                dy_dict[pressure_key_inlet] = dy.tolist()
            else:
                dy_dict[pressure_key_inlet] = [0.0] * len(connector_pressure)
            
            print(f"    Added pressure observations: {pressure_key_inlet}, {pressure_key_outlet}")
    
    new_observations['y'] = y_dict
    new_observations['dy'] = dy_dict
    
    print(f"  Added observations for connector vessels")
    
    return new_observations


def adjust_junction_boundaries_by_entrance_length(geometric_input, centerline_data, verbose=False):
    """
    Adjust junction boundaries to extend a distance EL (entrance length) down each outlet vessel.
    
    EL = 10 * MaximumInscribedSphereRadius of the outlet vessel at its inlet.
    If an outlet vessel is shorter than EL, the full vessel is included in the junction,
    and an artificial connector_vessel is created to connect the junction outlet to the
    next junction or boundary condition.
    
    Args:
        geometric_input: Dictionary with 0D model structure (vessels, junctions, boundary_conditions)
        centerline_data: Dictionary with centerline arrays (Points, BranchId, Path, MaximumInscribedSphereRadius, etc.)
        verbose: If True, print detailed information about the adjustment process
    
    Returns:
        Modified geometric_input with adjusted junction boundaries and connector vessels where needed
    """
    import copy
    verbose = True
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
        print(f"    _absorb_vessel_params: junc={jname}, outlet={outlet_vessel_name}, "
              f"fraction={fraction}, zvals_L={zvals.get('L', 'MISSING')}, "
              f"zvals_R={zvals.get('R_poiseuille', 'MISSING')}, "
              f"zvals_S={zvals.get('stenosis_coefficient', 'MISSING')}")
        for key in _EL_PARAM_KEYS:
            outlet_dict = gp.setdefault(f'outlet_{key}', {})
            old_val = outlet_dict.get(outlet_vessel_name, 0.0)
            new_val = old_val + fraction * zvals.get(key, 0.0)
            outlet_dict[outlet_vessel_name] = new_val
            print(f"      outlet_{key}[{outlet_vessel_name}]: {old_val} -> {new_val}")

    def _reduce_vessel_params(vessel, fraction_remaining):
        """Scale a vessel's L/R_poiseuille/stenosis_coefficient by the remaining fraction after absorption."""
        zvals = vessel.get('zero_d_element_values', {})
        for key in _EL_PARAM_KEYS:
            zvals[key] = zvals.get(key, 0.0) * fraction_remaining

    def _rename_outlet_in_gp(junc, old_name, new_name):
        """Rename an outlet vessel key in the junction's geometric_params dicts."""
        gp = junc.get('geometric_params', {})
        jname = junc.get('junction_name', '?')
        print(f"    _rename_outlet_in_gp: junc={jname}, {old_name} -> {new_name}")
        for key in _EL_PARAM_KEYS:
            outlet_dict = gp.get(f'outlet_{key}', {})
            if old_name in outlet_dict:
                outlet_dict[new_name] = outlet_dict.pop(old_name)
                print(f"      Renamed outlet_{key}[{old_name}] -> outlet_{key}[{new_name}]")
            else:
                print(f"      WARNING: outlet_{key} has no key '{old_name}', keys={list(outlet_dict.keys())}")

    # Track vessels that need to be removed (if any - currently not used for EL adjustment)
    vessels_to_remove = []
    # Track vessels converted to connectors (renamed and modified in-place)
    new_connector_vessels = []
    
    if verbose:
        print(f"\n  Processing {len(junctions)} junctions...")
    
    # Process each junction
    for junc in junctions:
        junction_name = junc.get('junction_name', '')
        inlet_vessels = junc.get('inlet_vessels', [])
        outlet_vessels = junc.get('outlet_vessels', [])
        
        if len(inlet_vessels) == 0 or len(outlet_vessels) == 0:
            if verbose:
                print(f"  Skipping {junction_name}: missing inlet or outlet vessels")
            continue
        
        # Only adjust junctions with at least 2 outlet vessels
        # Junctions with 1 outlet are just straight connections and don't need entrance length adjustment
        if len(outlet_vessels) < 2:
            if verbose:
                print(f"  Skipping {junction_name}: only {len(outlet_vessels)} outlet(s), need at least 2 for entrance length adjustment")
            continue
        
        if verbose:
            print(f"\n  Processing junction: {junction_name}")
            print(f"    Inlets: {len(inlet_vessels)}, Outlets: {len(outlet_vessels)}")
        
        # Get inlet vessel (assuming single inlet)
        inlet_vessel_id = inlet_vessels[0]
        inlet_vessel = vessel_by_id.get(inlet_vessel_id)
        if inlet_vessel is None:
            if verbose:
                print(f"    Warning: Inlet vessel ID {inlet_vessel_id} not found")
            continue
        
        # Process each outlet vessel
        for outlet_vessel_id in outlet_vessels:
            outlet_vessel = vessel_by_id.get(outlet_vessel_id)
            if outlet_vessel is None:
                if verbose:
                    print(f"    Warning: Outlet vessel ID {outlet_vessel_id} not found")
                continue
            
            outlet_vessel_name = outlet_vessel.get('vessel_name', '')
            
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
                
                # Build chain by merging vessels connected by single-outlet junctions
                while merged_length < EL:
                    # Find downstream junction for merged vessel
                    downstream_junction = None
                    for scan_junc in junctions:
                        inlet_vessel_ids = scan_junc.get('inlet_vessels', [])
                        if merged_vessel_id in inlet_vessel_ids:
                            downstream_junction = scan_junc
                            break
                    
                    # Check if downstream junction exists and has only 1 outlet
                    if downstream_junction is None:
                        if verbose:
                            print(f"      → No downstream junction found for {merged_vessel_name}, stopping chain extension")
                        break
                    
                    downstream_outlet_vessels = downstream_junction.get('outlet_vessels', [])
                    if len(downstream_outlet_vessels) != 1:
                        if verbose:
                            print(f"      → Downstream junction {downstream_junction.get('junction_name', 'unknown')} has {len(downstream_outlet_vessels)} outlets, stopping chain extension")
                        break
                    
                    # Get next vessel in chain
                    next_vessel_id = downstream_outlet_vessels[0]
                    next_vessel = vessel_by_id.get(next_vessel_id)
                    
                    if next_vessel is None:
                        if verbose:
                            print(f"      → Next vessel ID {next_vessel_id} not found, stopping chain extension")
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

                                        # Absorb proportional params from the merged vessel
                                        consumed_length = new_junction_boundary_path - merged_path_start
                                        fraction_consumed = consumed_length / merged_length if merged_length > 0 else 0.0
                                        _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=fraction_consumed)
                                        _reduce_vessel_params(merged_vessel, 1.0 - fraction_consumed)

                                        merged_vessel['vessel_length'] = float(new_merged_length)

                                        # Update node IDs:
                                        # - inlet moves to new boundary (target_centerline_idx)
                                        # - outlet stays at original merged_outlet_idx (BC location)
                                        set_vessel_node_ids(merged_vessel, target_centerline_idx, merged_outlet_idx)

                                        if verbose:
                                            print(f"      → Shortened merged vessel to reach EL: {merged_length:.6f} → {new_merged_length:.6f} cm")

                                        print(f"  Junction {junction_name}: Merged vessels {', '.join(merged_vessel_names)} "
                                              f"into {merged_vessel_name} and extended to reach EL={EL:.4f} with outlet BC {next_vessel_outlet_bc}")
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
                    next_vessel_downstream_junction = None
                    for scan_junc in junctions:
                        inlet_vessel_ids = scan_junc.get('inlet_vessels', [])
                        if next_vessel_id in inlet_vessel_ids:
                            next_vessel_downstream_junction = scan_junc
                            break
                    
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
                        # Update the downstream junction to use merged_vessel_id instead of next_vessel_id
                        inlet_vessel_ids = next_vessel_downstream_junction.get('inlet_vessels', [])
                        if next_vessel_id in inlet_vessel_ids:
                            inlet_vessel_ids.remove(next_vessel_id)
                            if merged_vessel_id not in inlet_vessel_ids:
                                inlet_vessel_ids.append(merged_vessel_id)
                            if verbose:
                                print(f"      → Updated downstream junction {next_vessel_downstream_junction.get('junction_name', 'unknown')}: "
                                      f"replaced vessel ID {next_vessel_id} with merged vessel ID {merged_vessel_id}")
                    
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

                                    merged_vessel['vessel_length'] = float(new_merged_length)
                                    
                                    # Update node IDs: outlet moves to new boundary
                                    set_vessel_node_ids(merged_vessel, outlet_inlet_idx, target_centerline_idx)
                                    merged_outlet_idx = target_centerline_idx  # Update tracked outlet index
                                    
                                    if verbose:
                                        print(f"      → Shortened merged vessel to reach EL: {merged_length:.6f} → {new_merged_length:.6f} cm")
                                    
                                    print(f"  Junction {junction_name}: Merged vessels {', '.join(merged_vessel_names)} "
                                          f"into {merged_vessel_name} and extended to reach EL={EL:.4f}")
                                    extension_successful = True
                                    break
                        else:
                            # Merged vessel length exactly equals EL — absorb everything
                            _absorb_vessel_params(junc, outlet_vessel_name, merged_vessel, fraction=1.0)
                            _reduce_vessel_params(merged_vessel, 0.0)

                            if verbose:
                                print(f"      → Merged vessel length exactly equals EL: {merged_length:.6f} cm")
                            print(f"  Junction {junction_name}: Merged vessels {', '.join(merged_vessel_names)} "
                                  f"into {merged_vessel_name} (length={merged_length:.4f} = EL={EL:.4f})")
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
                
                # If vessels were merged but we didn't reach EL, convert the merged vessel to a connector
                # This handles cases where chain extension stopped due to multi-outlet junction, boundary condition, etc.
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
                        _rename_outlet_in_gp(junc, outlet_vessel_name, connector_name)

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
                    
                    print(f"  Junction {junction_name}: Merged vessels {', '.join(merged_vessel_names)} "
                          f"into {connector_name} but didn't reach EL={EL:.4f}, converted to connector")
                    
                    new_connector_vessels.append(merged_vessel)
                    continue  # Skip the default connector logic below
                
                # Default case: Vessel is shorter than EL and doesn't connect to single-outlet junction
                # Rename to connector and set length to zero
                if verbose:
                    print(f"      → Converting vessel to connector (standard case)")
                
                print(f"  Junction {junction_name}: Outlet vessel {outlet_vessel_name} (length={outlet_vessel_length:.4f}) "
                      f"is shorter than EL={EL:.4f}, converting to connector vessel")
                
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
                    _rename_outlet_in_gp(junc, old_name, connector_name)

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
                    
                    if verbose:
                        print(f"      → Original vessel length: {outlet_vessel_length:.6f} cm")
                        print(f"      → New vessel length: {new_vessel_length:.6f} cm")
                        print(f"      → Length included in junction: {new_junction_boundary_path - outlet_vessel_path_start:.6f} cm")
                    
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
                    
                    print(f"  Junction {junction_name}: Extended boundary by EL={EL:.4f} for outlet {outlet_vessel_name}, "
                          f"new vessel length={new_vessel_length:.4f}")
                else:
                    # Edge case: EL extends beyond the vessel (shouldn't happen if logic is correct)
                    # This means target_path > outlet_vessel_path_end
                    if verbose:
                        print(f"      → Warning: EL={EL:.6f} extends beyond vessel end (path={outlet_vessel_path_end:.6f})")
                        print(f"      → This should not happen if vessel length >= EL")
                        print(f"      → Treating as full vessel inclusion")
                    
                    print(f"  Warning: EL={EL:.4f} extends beyond vessel {outlet_vessel_name} "
                          f"(length={outlet_vessel_length:.4f}), treating as full vessel inclusion")
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
        
        # Only keep junctions that have at least one inlet and one outlet after cleanup
        if len(cleaned_inlets) > 0 and len(cleaned_outlets) > 0:
            cleaned_junctions.append(junc)
        elif verbose:
            print(f"  Warning: Removed junction {junc.get('junction_name', 'unknown')} - no valid vessels after cleanup")
    
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
