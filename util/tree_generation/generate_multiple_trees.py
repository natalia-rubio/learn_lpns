#!/usr/bin/env python3
"""
Script to generate multiple vascular trees using the workflow from test.py
Trees are named tree_000, tree_001, tree_002, etc.
"""

import os
import sys
import shutil
import pyvista as pv
from svv.domain.domain import Domain
from svv.tree.tree import Tree
from svv.simulation.simulation import Simulation
from svv.simulation.fluid.rom.zero_d.zerod_tree import export_0d_simulation
from write_sherlock_shell import write_sherlock_shell
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_single_tree(tree_id, set_name='test_set', domain_size=45.0, n_branches=5, 
                        root_pressure=2*1333.22, terminal_pressure=1*1333.22, 
                        terminal_flow=300, hsize=0.3, reynolds_number=5000, 
                        time_steps=200, crease_angle=60.0):
    """
    Generate a single vascular tree with the given parameters.
    
    Args:
        tree_id (int): ID number for the tree (will be formatted as tree_XXX)
        set_name (str): Name of the set/directory to save the tree
        domain_size (float): Size of the cubic domain
        n_branches (int): Number of branches to add to the tree
        root_pressure (float): Root pressure in dyn/cm^2
        terminal_pressure (float): Terminal pressure in dyn/cm^2
        terminal_flow (float): Terminal flow in cm^3/s
        hsize (float): Mesh size parameter
        reynolds_number (float): Target Reynolds number
        time_steps (int): Number of time steps for simulation
        crease_angle (float): Crease angle for face extraction
    
    Returns:
        bool: True if successful, False otherwise
    """
    geo_name = f'tree_{tree_id:03d}'
    threeD_dir = os.path.join(os.getcwd(), 'data', 'threeD', set_name, geo_name)
    zeroD_dir = os.path.join(os.getcwd(), 'data', 'zeroD', set_name, geo_name)
    
    try:
        logger.info(f"Generating tree: {geo_name}")
        
        # Create domain
        logger.info(f"Creating domain cube with size {domain_size}")
        cube = Domain(pv.Cube(center=(0.0, 0.0, 0.0), 
                             x_length=domain_size, 
                             y_length=domain_size, 
                             z_length=domain_size))
        cube.create()
        cube.solve()
        cube.build()
        
        # Create tree
        logger.info(f"Creating vascular tree with {n_branches} branches")
        tree = Tree()
        tree.set_domain(cube)
        
        # Set tree parameters
        tree.parameters.set("root_pressure", root_pressure)
        tree.parameters.set("terminal_pressure", terminal_pressure)
        tree.parameters.set("terminal_flow", terminal_flow)
        tree.set_root()
        tree.n_add(n_branches)
        
        # Create simulation
        logger.info("Creating simulation container")
        sim = Simulation(tree, name=geo_name, set_name=set_name)
        
        # Build meshes
        logger.info("Building CFD-ready meshes")
        sim.build_meshes(fluid=True, tissue=False, boundary_layer=True, 
                        smooth_junctions=True, hsize=hsize)
        sim.extract_faces(crease_angle=crease_angle)
        
        # Construct 3D fluid simulation
        logger.info("Constructing 3D fluid simulation")

        sim.construct_3d_fluid_simulation(number_of_time_steps=time_steps, 
                                         target_reynolds_number=reynolds_number,
                                         )
        
        sim.write_3d_fluid_simulation()
        
        # Export 0D model
        logger.info("Exporting 0D fluid simulation")
        sim.write_0d_fluid_simulation()
        
        # Generate centerlines using SimVascular
        logger.info("Generating centerlines using SimVascular")
        simvascular_cmd = f"/Applications/SimVascular.app/Contents/Resources/simvascular --python -- util/tree_generation/simVascularCenterlineExtraction.py {geo_name} {set_name}"
        logger.info(f"Running: {simvascular_cmd}")
        result = os.system(simvascular_cmd)
        
        if result != 0:
            logger.warning(f"SimVascular centerline extraction returned non-zero exit code: {result}")
        
        # Verification of generated artifacts
        base_dir = os.path.join(os.getcwd(), 'data', 'threeD', set_name, geo_name)
        mesh_dir = os.path.join(base_dir, 'mesh')
        vtu_path = os.path.join(mesh_dir, 'fluid_msh_0', 'fluid_msh_0.vtu')
        mesh_surfaces_dir = os.path.join(mesh_dir, 'fluid_msh_0', 'mesh-surfaces')
        centerlines_path = os.path.join(base_dir, 'centerlines_simVascular.vtp')

        # Check fluid volume mesh
        if not os.path.isfile(vtu_path):
            raise FileNotFoundError(f"Missing volume mesh: {vtu_path}")

        # Check mesh-surfaces contents: 6 caps and 1 wall
        if not os.path.isdir(mesh_surfaces_dir):
            raise FileNotFoundError(f"Missing mesh-surfaces directory: {mesh_surfaces_dir}")
        surface_files = [f for f in os.listdir(mesh_surfaces_dir) if f.endswith('.vtp')]
        cap_files = [f for f in surface_files if f.startswith('cap_')]
        wall_files = [f for f in surface_files if f.startswith('wall_')]
        if len(cap_files) != 7:
            raise ValueError(f"Expected 7 cap files in mesh-surfaces, found {len(cap_files)} (dir: {mesh_surfaces_dir})")
        if len(wall_files) != 1:
            raise ValueError(f"Expected at least 1 wall file in mesh-surfaces, found {len(wall_files)} (dir: {mesh_surfaces_dir})")

        # Check centerlines file
        if not os.path.isfile(centerlines_path):
            raise FileNotFoundError(f"Missing SimVascular centerlines: {centerlines_path}")
        
        # Check that centerlines contain BranchId field
        try:
            centerlines = pv.read(centerlines_path)
            # Check both point data and cell data for BranchId
            has_branch_id = (
                'BranchId' in centerlines.point_data.keys() or 
                'BranchId' in centerlines.cell_data.keys() or
                'BranchId' in centerlines.field_data.keys()
            )
            if not has_branch_id:
                raise ValueError(f"Centerlines file {centerlines_path} does not contain 'BranchId' field. "
                               f"Available point data: {list(centerlines.point_data.keys())}, "
                               f"Available cell data: {list(centerlines.cell_data.keys())}, "
                               f"Available field data: {list(centerlines.field_data.keys())}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise ValueError about missing BranchId
            else:
                raise RuntimeError(f"Failed to read centerlines file {centerlines_path}: {str(e)}")

        logger.info("Output verification passed: volume mesh, mesh-surfaces (7 caps + wall), and centerlines present.")
        
        # Write Sherlock shell script
        logger.info("Writing Sherlock shell script")
        write_sherlock_shell(geo_name=geo_name, set_name=set_name)
        
        logger.info(f"Successfully generated tree: {geo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate tree {geo_name}: {str(e)}")
        
        # Clean up geometry folders if they were created
        logger.info(f"Cleaning up failed tree geometry folders...")
        if os.path.exists(threeD_dir):
            try:
                shutil.rmtree(threeD_dir)
                logger.info(f"Removed threeD directory: {threeD_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove threeD directory {threeD_dir}: {cleanup_error}")
        
        if os.path.exists(zeroD_dir):
            try:
                shutil.rmtree(zeroD_dir)
                logger.info(f"Removed zeroD directory: {zeroD_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove zeroD directory {zeroD_dir}: {cleanup_error}")
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate multiple vascular trees')

    parser.add_argument('--num-trees', type=int, default=5, 
                       help='Number of trees to generate (default: 5)')
    parser.add_argument('--set-name', type=str, default='test_set',
                       help='Name of the set/directory (default: test_set)')
    parser.add_argument('--domain-size', type=float, default=45.0,
                       help='Size of the cubic domain (default: 45.0)')
    parser.add_argument('--n-branches', type=int, default=5,
                       help='Number of branches per tree (default: 5)')
    parser.add_argument('--root-pressure', type=float, default=3*1333.22,
                       help='Root pressure in dyn/cm^2 (default: 2666.44)')
    parser.add_argument('--terminal-pressure', type=float, default=1*1333.22,
                       help='Terminal pressure in dyn/cm^2 (default: 1333.22)')
    parser.add_argument('--terminal-flow', type=float, default=300,
                       help='Terminal flow in cm^3/s (default: 300)')
    parser.add_argument('--hsize', type=float, default=0.3,
                       help='Mesh size parameter (default: 0.3)')
    parser.add_argument('--reynolds-number', type=float, default=5000,
                       help='Target Reynolds number (default: 5000)')
    parser.add_argument('--time-steps', type=int, default=200,
                       help='Number of time steps (default: 100)')
    parser.add_argument('--crease-angle', type=float, default=60.0,
                       help='Crease angle for face extraction (default: 60.0)')
    parser.add_argument('--start-id', type=int, default=0,
                       help='Starting tree ID (default: 0)')
    parser.add_argument('--max-retries', type=int, default=10,
                       help='Maximum retries per tree before giving up (default: 10)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting generation of {args.num_trees} trees")
    logger.info(f"Set name: {args.set_name}")
    logger.info(f"Tree IDs: {args.start_id} to {args.start_id + args.num_trees - 1}")
    logger.info(f"Max retries per tree: {args.max_retries}")
    
    successful_trees = 0
    total_attempts = 0
    current_tree_id = args.start_id
    
    while successful_trees < args.num_trees:
        logger.info(f"\n{'='*50}")
        logger.info(f"Generating tree {successful_trees + 1}/{args.num_trees} (ID: {current_tree_id})")
        logger.info(f"{'='*50}")
        
        attempt = 1
        success = False
        
        while not success and attempt <= args.max_retries:
            if attempt > 1:
                logger.info(f"Retry attempt {attempt}/{args.max_retries} for tree ID {current_tree_id}")
            
            total_attempts += 1
            success = generate_single_tree(
                tree_id=current_tree_id,
                set_name=args.set_name,
                domain_size=args.domain_size,
                n_branches=args.n_branches,
                root_pressure=args.root_pressure,
                terminal_pressure=args.terminal_pressure,
                terminal_flow=args.terminal_flow,
                hsize=args.hsize,
                reynolds_number=args.reynolds_number,
                time_steps=args.time_steps,
                crease_angle=args.crease_angle
            )
            
            if success:
                successful_trees += 1
                logger.info(f"Successfully generated tree {successful_trees}/{args.num_trees} (ID: {current_tree_id})")
                current_tree_id += 1
            else:
                attempt += 1
                if attempt <= args.max_retries:
                    logger.warning(f"Tree ID {current_tree_id} failed, will retry...")
                else:
                    logger.error(f"Tree ID {current_tree_id} failed after {args.max_retries} attempts. Moving to next ID.")
                    current_tree_id += 1
    
    logger.info(f"\n{'='*50}")
    logger.info("GENERATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total trees requested: {args.num_trees}")
    logger.info(f"Successfully generated: {successful_trees}")
    logger.info(f"Total attempts: {total_attempts}")
    logger.info(f"Success rate: {successful_trees/total_attempts*100:.1f}%")
    
    if successful_trees == args.num_trees:
        logger.info("All requested trees generated successfully!")
        sys.exit(0)
    else:
        logger.error(f"Only {successful_trees}/{args.num_trees} trees generated successfully.")
        sys.exit(1)

if __name__ == "__main__":
    main()
