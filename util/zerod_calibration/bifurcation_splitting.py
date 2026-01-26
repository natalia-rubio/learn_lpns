import numpy as np
import copy
import json
import csv
from util.zerod_calibration.file_io import read_centerline_vtp



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
    
    if branch_id_array is None or path_array is None or points_array is None:
        print("  Warning: BranchId, Path, or Points not found in centerline data, cannot determine bifurcation order")
        return result
    
    if bifurcation_id_array is None:
        print("  Warning: BifurcationId not found in centerline data, using branch-based ordering")
        bifurcation_id_array = np.full_like(branch_id_array, -1)
    
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
        
        # Compute in-junction path lengths to determine main outlet and ordering
        if junction_bif_id is not None:
            in_junction_path_lengths = compute_in_junction_path_lengths(inlet_branch_id, outlet_branch_ids, junction_bif_id)
        else:
            in_junction_path_lengths = {}

        
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
                new_junc = {
                    "inlet_vessels": [current_inlet_id],
                    "junction_name": new_junc_name,
                    "junction_type": junc_type,
                    "outlet_vessels": [side_outlet_id, main_outlet_id]
                }
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
                    }
                }
                
                new_vessels.append(connector_vessel)
                vessel_by_id[next_vessel_id] = connector_vessel
                vessel_by_name[connector_name] = connector_vessel
                
                # Create bifurcation junction
                new_junc = {
                    "inlet_vessels": [current_inlet_id],
                    "junction_name": new_junc_name,
                    "junction_type": junc_type,
                    "outlet_vessels": [side_outlet_id, next_vessel_id]
                }
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
        
        # Parse connector name to get inlet vessel name and connector index
        # Format: "{inlet_vessel_name}_connector{i}" (e.g., "branch0_seg0_connector0")
        import re
        connector_match = re.match(r'(.+)_connector(\d+)$', connector_name)
        if not connector_match:
            print(f"    Warning: Could not parse connector name: {connector_name}")
            continue
        
        inlet_vessel_name_from_connector = connector_match.group(1)
        connector_idx = int(connector_match.group(2))
        inlet_branch_id = get_branch_id(connector_name)
        
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
