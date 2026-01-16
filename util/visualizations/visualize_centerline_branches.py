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
except ImportError:
    print("Error: Matplotlib is required.")
    sys.exit(1)


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
    
    # Create mapping from vessel index to vessel name
    vessel_id_to_name = {}
    for i, vessel in enumerate(vessels):
        vessel_id_to_name[i] = vessel.get('vessel_name', f'vessel_{i}')
    
    # Find root vessel (has inlet BC)
    root_vessel_id = None
    for i, vessel in enumerate(vessels):
        if 'boundary_conditions' in vessel and 'inlet' in vessel['boundary_conditions']:
            root_vessel_id = i
            break
    
    # If no explicit inlet BC, assume first vessel is root
    if root_vessel_id is None and vessels:
        root_vessel_id = 0
    
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
    for i, vessel in enumerate(vessels):
        if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
            terminal_vessels.add(i)
        elif i not in tree or len(tree[i]) == 0:
            # No children means it's terminal
            terminal_vessels.add(i)
    
    return vessel_id_to_name, tree, root_vessel_id, terminal_vessels


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


def visualize_centerline_as_tree(geometric_input_path, output_path, geometry_type='original'):
    """
    Create 2D tree visualization of centerline with branch labels.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON
        output_path: Path to save output image
        geometry_type: 'original' or 'bifurcations'
    """
    print(f"Reading geometric input from: {geometric_input_path}")
    vessel_id_to_name, tree, root_vessel_id, terminal_vessels = build_tree_structure(geometric_input_path)
    
    if root_vessel_id is None:
        print("Error: Could not find root vessel (inlet)")
        return False
    
    print(f"Found {len(vessel_id_to_name)} vessels")
    print(f"Root vessel: {vessel_id_to_name[root_vessel_id]}")
    print(f"Terminal vessels: {[vessel_id_to_name[v] for v in terminal_vessels]}")
    
    # Compute positions
    positions = compute_tree_layout(tree, root_vessel_id, vessel_id_to_name)
    
    if not positions:
        print("Error: Could not compute tree layout")
        return False
    
    # Create figure - larger for bifurcations geometry which may have more nodes
    fig_width = 18 if geometry_type == 'bifurcations' else 14
    fig_height = 12 if geometry_type == 'bifurcations' else 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Color map for branches
    num_vessels = len(vessel_id_to_name)
    colors = plt.cm.tab20(np.linspace(0, 1, num_vessels))
    
    # Draw edges (connections)
    for parent_id, child_ids in tree.items():
        if parent_id not in positions:
            continue
        parent_pos = positions[parent_id]
        
        for child_id in child_ids:
            if child_id not in positions:
                continue
            child_pos = positions[child_id]
            
            # Draw line from parent to child
            ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]],
                   'k-', linewidth=2, alpha=0.5, zorder=1)
    
    # Draw nodes (vessels)
    for vessel_id, (x, y) in positions.items():
        vessel_name = vessel_id_to_name[vessel_id]
        color = colors[vessel_id % len(colors)]
        
        # Determine node style
        is_connector = '_connector' in vessel_name
        
        if vessel_id == root_vessel_id:
            # Root (inlet) - larger circle
            node_size = 300
            node_color = 'green'
            node_shape = 'o'
        elif vessel_id in terminal_vessels:
            # Terminal (outlet) - square
            node_size = 250
            node_color = 'red'
            node_shape = 's'
        elif is_connector:
            # Connector vessel (bifurcations geometry) - diamond
            node_size = 200
            node_color = 'orange'
            node_shape = 'D'
        else:
            # Internal - circle
            node_size = 200
            node_color = color
            node_shape = 'o'
        
        # Draw node
        ax.scatter(x, y, s=node_size, c=node_color, marker=node_shape, 
                  edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        
        # Add label - smaller font for bifurcations geometry
        fontsize = 8 if geometry_type == 'bifurcations' else 10
        ax.text(x, y, vessel_name, fontsize=fontsize, ha='center', va='center',
               weight='bold', zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor=node_color, linewidth=1.5))
    
    # Set labels and title
    ax.set_xlabel('Branch Position', fontsize=14)
    ax.set_ylabel('Tree Level', fontsize=14)
    
    if geometry_type == 'bifurcations':
        title = '1D Centerline Tree Structure (Bifurcations-Only 0D Representation)'
    else:
        title = '1D Centerline Tree Structure (Original 0D Representation)'
    ax.set_title(title, fontsize=16, weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='Inlet (Root)'),
        mpatches.Patch(color='red', label='Terminal Outlet (BC)'),
        mpatches.Patch(color='blue', label='Internal Branch'),
    ]
    if geometry_type == 'bifurcations':
        legend_elements.append(mpatches.Patch(color='orange', label='Connector Vessel'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add some padding
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    if all_x and all_y:
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        padding = 0.1
        ax.set_xlim(min(all_x) - padding * x_range, max(all_x) + padding * x_range)
        ax.set_ylim(min(all_y) - padding * y_range, max(all_y) + padding * y_range)
    
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
                       choices=['original', 'bifurcations'],
                       help='Geometry type: original or bifurcations (default: original)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for image (default: results/centerline_labels/{set_name}/{geo_name}/centerline_labels_{geometry_type}.png)')
    
    args = parser.parse_args()
    
    # Find geometric input (centerline file not needed for tree visualization)
    geometric_input_path = find_geometric_input(args.set_name, args.geo_name, args.geometry_type)
    if geometric_input_path is None:
        if args.geometry_type == 'bifurcations':
            expected_file = 'bifurcations_geometric_input.json'
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
