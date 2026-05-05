#!/usr/bin/env python3
"""
Visualize 1D centerline representation as a 2D tree with branch labels matching 0D representation.

Shows the centerline geometry as a tree diagram maintaining connectivity and labeling
each branch as it appears in the 0D model (e.g., branch0_seg0, branch1_seg0, etc.).
"""

import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict, deque

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Enable LaTeX rendering with Computer Modern font
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })
except ImportError:
    print("Error: Matplotlib is required.")
    sys.exit(1)


# =============================================================================
# FONT CONFIGURATION
# =============================================================================
# Define font sizes for different elements here.

FONT_CONFIG = {
    'vessel_label': 24,
    'node_label': 24,
    'title': 18,
    'axis_label': 14,
    'legend': 12,
}
# =============================================================================


# =============================================================================
# COLOR CONFIGURATION
# =============================================================================
# Define colors for different element types here.
# Each element type has an 'edge' color (outline) and optional 'linewidth'.
# All labels use white background with black text.

ELEMENT_COLORS = {
    'inlet_bc': {
        'edge': 'slategray',
        'linewidth': 3,
    },
    'outlet_bc': {
        'edge': 'slategray',
        'linewidth': 3,
    },
    'junction': {
        'edge': 'lightcoral',
        'linewidth': 2.5,
    },
    'bifurcation_junction': {
        'edge': 'crimson',
        'linewidth': 2.5,
    },
    'vessel': {
        'edge': 'lightskyblue',
        'linewidth': 2,
    },
    'connector_vessel': {
        'edge': 'royalblue',
        'linewidth': 3,
    },
    'line': {
        'color': 'black',
        'linewidth': 2.5,
    },
}
# =============================================================================


def find_centerline_file(set_name, geo_name):
    """Find centerline file in various possible locations."""
    possible_paths = [
        os.path.join('data', 'oneD', set_name, geo_name, 'unsteady_soln.vtp'),
        os.path.join('data', 'oneD', set_name, geo_name, 'centerlines_simVascular.vtp'),
        os.path.join('data', 'threeD', set_name, geo_name, 'centerlines_simVascular.vtp'),
        os.path.join('data', 'threeD', set_name, geo_name, 'centerlines', 'centerlines.vtp'),
        os.path.join('data', 'reduced_results', set_name, geo_name, 'unsteady_soln.vtp'),
        os.path.join('/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees', 
                     set_name, geo_name, 'unsteady_soln.vtp'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_geometric_input(set_name, geo_name, geometry_type='original'):
    """
    Find geometric 0D input JSON file.
    
    Args:
        set_name: Set name (e.g., 'VMR')
        geo_name: Geometry name (e.g., '0063_1001')
        geometry_type: 'original' or 'bifurcations'
    
    Returns:
        Path to geometric input file, or None if not found
    """
    if geometry_type == 'bifurcations':
        possible_paths = [
            os.path.join('data', 'zeroD', set_name, geo_name, 'bifurcations_geometric_input.json'),
        ]
    elif geometry_type == 'bifurcations_EL':
        possible_paths = [
            os.path.join('data', 'zeroD', set_name, geo_name, 'bifurcations_EL_geometric_input.json'),
        ]
    else:
        possible_paths = [
            os.path.join('data', 'zeroD', set_name, geo_name, 'geometric_input.json'),
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def build_tree_structure(geometric_input_path):
    """
    Build tree structure from geometric input JSON.
    
    Returns:
        vessels: Dictionary mapping vessel_id -> vessel_name
        tree: Dictionary mapping vessel_id -> list of child vessel_ids
        root_vessel_id: ID of root vessel (inlet)
        terminal_vessels: Set of vessel IDs that are terminal (have outlet BCs)
    """
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    
    # Create mapping from vessel_id to vessel name (use actual vessel_id from JSON)
    vessel_id_to_name = {}
    for vessel in vessels:
        vessel_id = vessel.get('vessel_id', None)
        if vessel_id is not None:
            vessel_id_to_name[vessel_id] = vessel.get('vessel_name', f'vessel_{vessel_id}')
    
    # Find root vessel (has inlet BC)
    root_vessel_id = None
    for vessel in vessels:
        vessel_id = vessel.get('vessel_id', None)
        if vessel_id is not None:
            if 'boundary_conditions' in vessel and 'inlet' in vessel['boundary_conditions']:
                root_vessel_id = vessel_id
                break
    
    # If no explicit inlet BC, use first vessel_id (if any)
    if root_vessel_id is None and vessel_id_to_name:
        root_vessel_id = min(vessel_id_to_name.keys())
    
    # Build tree structure from junctions
    # tree[vessel_id] = list of child vessel_ids
    tree = defaultdict(list)
    vessel_to_parent = {}  # For reverse lookup
    
    for junc in junctions:
        inlet_vessel_ids = junc.get('inlet_vessels', [])
        outlet_vessel_ids = junc.get('outlet_vessels', [])
        
        # Each inlet vessel connects to all outlet vessels through this junction
        for inlet_id in inlet_vessel_ids:
            for outlet_id in outlet_vessel_ids:
                tree[inlet_id].append(outlet_id)
                vessel_to_parent[outlet_id] = inlet_id
    
    # Find terminal vessels (have outlet BCs or no children)
    terminal_vessels = set()
    for vessel in vessels:
        vessel_id = vessel.get('vessel_id', None)
        if vessel_id is not None:
            if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
                terminal_vessels.add(vessel_id)
            elif vessel_id not in tree or len(tree[vessel_id]) == 0:
                # No children means it's terminal
                terminal_vessels.add(vessel_id)
    
    return vessel_id_to_name, tree, root_vessel_id, terminal_vessels


def build_junction_graph(geometric_input_path):
    """
    Build a graph structure where junctions are nodes and vessels are edges.
    
    Returns:
        nodes: Dictionary mapping node_id -> node_info dict
               node_info contains 'name', 'type' ('inlet_bc', 'outlet_bc', 'junction')
        edges: List of (from_node, to_node, vessel_name) tuples
        root_node: ID of the root node (inlet BC)
    """
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    
    nodes = {}
    edges = []
    
    # Create mapping from vessel_id to vessel info (use actual vessel_id from JSON, not index)
    vessel_by_id = {}
    for vessel in vessels:
        vessel_id = vessel.get('vessel_id', len(vessel_by_id))
        vessel_by_id[vessel_id] = vessel
    
    # Create junction nodes
    junction_name_to_node_id = {}
    for junc in junctions:
        junc_name = junc.get('junction_name', f"J{len(junction_name_to_node_id)}")
        node_id = f"junc_{junc_name}"
        nodes[node_id] = {
            'name': junc_name,
            'type': 'junction',
            'inlet_vessels': junc.get('inlet_vessels', []),
            'outlet_vessels': junc.get('outlet_vessels', [])
        }
        junction_name_to_node_id[junc_name] = node_id
    
    # Find vessels with inlet/outlet BCs and create BC nodes
    inlet_bc_node = None
    outlet_bc_nodes = {}
    
    for vessel in vessels:
        vessel_id = vessel.get('vessel_id', None)
        if vessel_id is None:
            continue  # Skip vessels without vessel_id
        vessel_name = vessel.get('vessel_name', f'vessel_{vessel_id}')
        
        if 'boundary_conditions' in vessel:
            if 'inlet' in vessel['boundary_conditions']:
                bc_name = vessel['boundary_conditions']['inlet']
                node_id = f"bc_{bc_name}"
                nodes[node_id] = {
                    'name': bc_name,
                    'type': 'inlet_bc',
                    'vessel_id': vessel_id
                }
                inlet_bc_node = node_id
            
            if 'outlet' in vessel['boundary_conditions']:
                bc_name = vessel['boundary_conditions']['outlet']
                node_id = f"bc_{bc_name}"
                nodes[node_id] = {
                    'name': bc_name,
                    'type': 'outlet_bc',
                    'vessel_id': vessel_id
                }
                outlet_bc_nodes[vessel_id] = node_id
    
    # Build mapping: vessel_id -> (upstream_junction, downstream_junction/bc)
    vessel_upstream = {}  # vessel_id -> node_id where vessel is outlet
    vessel_downstream = {}  # vessel_id -> node_id where vessel is inlet
    
    for junc in junctions:
        junc_name = junc.get('junction_name')
        node_id = junction_name_to_node_id[junc_name]
        
        for inlet_vessel_id in junc.get('inlet_vessels', []):
            vessel_downstream[inlet_vessel_id] = node_id
        
        for outlet_vessel_id in junc.get('outlet_vessels', []):
            vessel_upstream[outlet_vessel_id] = node_id
    
    # Create edges (vessels connect nodes)
    for vessel in vessels:
        vessel_id = vessel.get('vessel_id', None)
        if vessel_id is None:
            continue  # Skip vessels without vessel_id
        vessel_name = vessel.get('vessel_name', f'vessel_{vessel_id}')
        
        # Determine from_node (upstream end of vessel)
        if vessel_id in vessel_upstream:
            from_node = vessel_upstream[vessel_id]
        elif 'boundary_conditions' in vessel and 'inlet' in vessel['boundary_conditions']:
            from_node = inlet_bc_node
        else:
            # Vessel has no upstream - create implicit inlet node
            from_node = f"implicit_inlet_{vessel_id}"
            nodes[from_node] = {'name': 'INLET', 'type': 'inlet_bc', 'vessel_id': vessel_id}
            if inlet_bc_node is None:
                inlet_bc_node = from_node
        
        # Determine to_node (downstream end of vessel)
        if vessel_id in vessel_downstream:
            to_node = vessel_downstream[vessel_id]
        elif vessel_id in outlet_bc_nodes:
            to_node = outlet_bc_nodes[vessel_id]
        else:
            # Vessel has no downstream - create implicit outlet node
            to_node = f"implicit_outlet_{vessel_id}"
            nodes[to_node] = {'name': f'OUT_{vessel_id}', 'type': 'outlet_bc', 'vessel_id': vessel_id}
        
        edges.append((from_node, to_node, vessel_name, vessel_id))
    
    return nodes, edges, inlet_bc_node


def compute_tree_layout(tree, root_vessel_id, vessel_id_to_name):
    """
    Compute 2D positions for tree nodes using hierarchical layout.
    
    Returns:
        positions: Dictionary mapping vessel_id -> (x, y) tuple
    """
    positions = {}
    
    if root_vessel_id is None:
        return positions
    
    # BFS to assign levels (depth from root)
    levels = {}
    queue = deque([(root_vessel_id, 0)])
    levels[root_vessel_id] = 0
    max_level = 0
    
    while queue:
        vessel_id, level = queue.popleft()
        max_level = max(max_level, level)
        
        for child_id in tree.get(vessel_id, []):
            if child_id not in levels:
                levels[child_id] = level + 1
                queue.append((child_id, level + 1))
    
    # Group vessels by level
    vessels_by_level = defaultdict(list)
    for vessel_id, level in levels.items():
        vessels_by_level[level].append(vessel_id)
    
    # Assign y positions based on level
    level_height = 1.5
    y_start = 0.0
    
    # Assign x positions within each level (distribute evenly)
    for level in range(max_level + 1):
        vessels_in_level = vessels_by_level[level]
        num_vessels = len(vessels_in_level)
        
        if num_vessels == 1:
            x_positions = [0.0]
        else:
            x_spacing = max(2.0, num_vessels * 0.8)
            x_start = -x_spacing / 2
            x_positions = np.linspace(x_start, x_spacing / 2, num_vessels)
        
        y_pos = y_start - level * level_height
        
        for i, vessel_id in enumerate(vessels_in_level):
            if num_vessels == 1:
                positions[vessel_id] = (0.0, y_pos)
            else:
                positions[vessel_id] = (x_positions[i], y_pos)
    
    return positions


def compute_junction_layout(nodes, edges, root_node):
    """
    Compute 2D positions for junction nodes using hierarchical layout.
    Single-child nodes place child directly below (vertically aligned).
    
    Returns:
        positions: Dictionary mapping node_id -> (x, y) tuple
    """
    positions = {}
    
    if root_node is None:
        return positions
    
    # Build adjacency list from edges (downstream direction)
    children = defaultdict(list)
    for from_node, to_node, vessel_name, vessel_id in edges:
        children[from_node].append(to_node)
    
    # BFS to assign levels (depth from root)
    levels = {}
    parent_of = {}  # Track parent of each node
    queue = deque([(root_node, 0, None)])
    levels[root_node] = 0
    max_level = 0
    
    while queue:
        node_id, level, parent = queue.popleft()
        max_level = max(max_level, level)
        parent_of[node_id] = parent
        
        for child_id in children.get(node_id, []):
            if child_id not in levels:
                levels[child_id] = level + 1
                queue.append((child_id, level + 1, node_id))
    
    # Handle any nodes not connected to root
    for node_id in nodes:
        if node_id not in levels:
            levels[node_id] = max_level + 1
    
    max_level = max(levels.values()) if levels else 0
    
    # Count leaf nodes in subtree for each node (for width allocation)
    def count_leaves(node_id):
        child_list = children.get(node_id, [])
        if not child_list:
            return 1
        return sum(count_leaves(c) for c in child_list)
    
    leaf_counts = {node_id: count_leaves(node_id) for node_id in nodes}
    
    # Assign positions using recursive approach
    level_height = 3.0
    leaf_spacing = 3.0  # Space per leaf node
    
    def assign_positions(node_id, x_center, y_pos):
        positions[node_id] = (x_center, y_pos)
        
        child_list = children.get(node_id, [])
        if not child_list:
            return
        
        child_y = y_pos - level_height
        
        if len(child_list) == 1:
            # Single child: place directly below (vertical alignment)
            assign_positions(child_list[0], x_center, child_y)
        else:
            # Multiple children: distribute based on their subtree widths
            total_width = sum(leaf_counts[c] for c in child_list) * leaf_spacing
            current_x = x_center - total_width / 2
            
            for child_id in child_list:
                child_width = leaf_counts[child_id] * leaf_spacing
                child_x = current_x + child_width / 2
                assign_positions(child_id, child_x, child_y)
                current_x += child_width
    
    # Start layout from root
    assign_positions(root_node, 0.0, 0.0)
    
    return positions


def visualize_centerline_as_tree(geometric_input_path, output_path, geometry_type='original'):
    """
    Create 2D tree visualization with junction names on nodes and vessel names on edges.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        output_path: Path to save output image
        geometry_type: 'original' or 'bifurcations'
    """
    print(f"Reading geometric input from: {geometric_input_path}")
    
    # Build junction-based graph
    nodes, edges, root_node = build_junction_graph(geometric_input_path)
    
    if root_node is None:
        print("Error: Could not find root node (inlet)")
        return False
    
    num_junctions = sum(1 for n in nodes.values() if n['type'] == 'junction')
    num_inlet_bcs = sum(1 for n in nodes.values() if n['type'] == 'inlet_bc')
    num_outlet_bcs = sum(1 for n in nodes.values() if n['type'] == 'outlet_bc')
    
    print(f"Found {len(nodes)} nodes: {num_junctions} junctions, {num_inlet_bcs} inlet BC(s), {num_outlet_bcs} outlet BC(s)")
    print(f"Found {len(edges)} vessels (edges)")
    
    # Compute positions for junction nodes
    positions = compute_junction_layout(nodes, edges, root_node)
    
    if not positions:
        print("Error: Could not compute tree layout")
        return False
    
    # Create figure - larger for bifurcations geometry which may have more nodes
    fig_width = 26 if geometry_type == 'bifurcations' else 22
    fig_height = 20 if geometry_type == 'bifurcations' else 18
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Draw edges (vessels) with labels
    for i, (from_node, to_node, vessel_name, vessel_id) in enumerate(edges):
        if from_node not in positions or to_node not in positions:
            continue
        
        from_pos = positions[from_node]
        to_pos = positions[to_node]
        
        # Get line style from config
        line_config = ELEMENT_COLORS['line']
        
        # Draw line from parent to child
        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
               '-', color=line_config['color'], linewidth=line_config['linewidth'], 
               alpha=0.7, zorder=1)
        
        # Add vessel name label at midpoint of edge
        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] + to_pos[1]) / 2
        
        # Offset label slightly to avoid overlap with line
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Perpendicular offset
            offset_x = -dy / length * 0.2
            offset_y = dx / length * 0.2
        else:
            offset_x, offset_y = 0.1, 0
        
        # Get label style from config based on vessel type
        is_connector = '_connector' in vessel_name
        if is_connector:
            vessel_config = ELEMENT_COLORS['connector_vessel']
        else:
            vessel_config = ELEMENT_COLORS['vessel']
        
        # Horizontal labels (no rotation), white background
        ax.text(mid_x + offset_x, mid_y + offset_y, vessel_name, 
               fontsize=FONT_CONFIG['vessel_label'], ha='center', va='center',
               color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.95, edgecolor=vessel_config['edge'], 
                        linewidth=vessel_config['linewidth']))
    
    # Draw nodes (junctions and BCs) as rectangular labels with white background
    for node_id, (x, y) in positions.items():
        node_info = nodes[node_id]
        node_name = node_info['name']
        node_type = node_info['type']
        
        # Get style from config based on node type
        if node_type == 'inlet_bc':
            node_config = ELEMENT_COLORS['inlet_bc']
        elif node_type == 'outlet_bc':
            node_config = ELEMENT_COLORS['outlet_bc']
        else:  # junction
            # Check if it's a bifurcation junction (from split)
            is_bif_junction = '_bif' in node_name
            if is_bif_junction:
                node_config = ELEMENT_COLORS['bifurcation_junction']
            else:
                node_config = ELEMENT_COLORS['junction']
        
        # Add junction/BC name as rectangular label with white background
        ax.text(x, y, node_name, fontsize=FONT_CONFIG['node_label'], ha='center', va='center',
               weight='bold', zorder=4, color='black',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        alpha=0.95, edgecolor=node_config['edge'], 
                        linewidth=node_config['linewidth']))
    
    # Set labels and title
    ax.set_xlabel('Branch Position', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Tree Level', fontsize=FONT_CONFIG['axis_label'])
    
    if geometry_type == 'bifurcations':
        title = r'0D Network Structure (Bifurcations-Only)'
    else:
        title = r'0D Network Structure (Original)'
    ax.set_title(title, fontsize=FONT_CONFIG['title'], weight='bold')
    
    # Add legend - white backgrounds with different outline colors (from config)
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor=ELEMENT_COLORS['inlet_bc']['edge'], 
                      linewidth=ELEMENT_COLORS['inlet_bc']['linewidth'], label='Inlet BC'),
        mpatches.Patch(facecolor='white', edgecolor=ELEMENT_COLORS['outlet_bc']['edge'], 
                      linewidth=ELEMENT_COLORS['outlet_bc']['linewidth'], label='Outlet BC'),
        mpatches.Patch(facecolor='white', edgecolor=ELEMENT_COLORS['junction']['edge'], 
                      linewidth=ELEMENT_COLORS['junction']['linewidth'], label='Junction'),
        mpatches.Patch(facecolor='white', edgecolor=ELEMENT_COLORS['vessel']['edge'], 
                      linewidth=ELEMENT_COLORS['vessel']['linewidth'], label='Vessel'),
    ]
    if geometry_type == 'bifurcations':
        legend_elements.insert(3, mpatches.Patch(facecolor='white', 
                      edgecolor=ELEMENT_COLORS['bifurcation_junction']['edge'], 
                      linewidth=ELEMENT_COLORS['bifurcation_junction']['linewidth'], 
                      label='Bifurcation Junction'))
        legend_elements.append(mpatches.Patch(facecolor='white', 
                      edgecolor=ELEMENT_COLORS['connector_vessel']['edge'], 
                      linewidth=ELEMENT_COLORS['connector_vessel']['linewidth'], 
                      label='Connector Vessel'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_CONFIG['legend'])
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add some padding
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    if all_x and all_y:
        x_range = max(all_x) - min(all_x) if max(all_x) != min(all_x) else 1
        y_range = max(all_y) - min(all_y) if max(all_y) != min(all_y) else 1
        padding = 0.15
        ax.set_xlim(min(all_x) - padding * x_range - 0.5, max(all_x) + padding * x_range + 0.5)
        ax.set_ylim(min(all_y) - padding * y_range - 0.5, max(all_y) + padding * y_range + 0.5)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 1D centerline as 2D tree with branch labels matching 0D representation'
    )
    parser.add_argument('--set-name', type=str, required=True,
                       help='Set name (e.g., set_3)')
    parser.add_argument('--geo-name', type=str, required=True,
                       help='Geometry name (e.g., tree_002)')
    parser.add_argument('--geometry-type', type=str, default='original',
                       choices=['original', 'bifurcations', 'bifurcations_EL'],
                       help='Geometry type: original or bifurcations (default: original)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for image (default: results/centerline_labels/{set_name}/{geo_name}/centerline_labels_{geometry_type}.png)')
    
    args = parser.parse_args()
    
    # Find geometric input (centerline file not needed for tree visualization)
    geometric_input_path = find_geometric_input(args.set_name, args.geo_name, args.geometry_type)
    if geometric_input_path is None:
        if args.geometry_type == 'bifurcations':
            expected_file = 'bifurcations_geometric_input.json'
        elif args.geometry_type == 'bifurcations_EL':
            expected_file = 'bifurcations_EL_geometric_input.json'
        else:
            expected_file = 'geometric_input.json'
        print(f"Error: Could not find geometric input file for {args.set_name}/{args.geo_name}")
        print(f"  Expected: data/zeroD/{args.set_name}/{args.geo_name}/{expected_file}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join('results', 'centerline_labels', args.set_name, 
                                   args.geo_name, f'centerline_labels_{args.geometry_type}.png')
    
    # Create visualization
    success = visualize_centerline_as_tree(geometric_input_path, output_path, args.geometry_type)
    
    if success:
        print(f"\n✓ Successfully created visualization")
        print(f"  Output: {output_path}")
    else:
        print("\n✗ Failed to create visualization")
        sys.exit(1)


if __name__ == '__main__':
    main()
