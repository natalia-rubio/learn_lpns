#!/usr/bin/env python3
"""
Generate shell scripts for launching svMultiPhysics jobs on Sherlock cluster.
"""

import os
from pathlib import Path


def write_sherlock_shell(geo_name, set_name):
    """
    Write a shell script for launching svMultiPhysics jobs on Sherlock cluster.
    
    Parameters:
    -----------
    geo_name : str
        Name of the geometry (used as job name)
    set_name : str
        Name of the set
    partition : str, optional
        SLURM partition to use (default: "amarsden")
    output_dir : str, optional
        Directory for output and error files (default: "/scratch/users/nrubio/job_scripts")
    time_limit : str, optional
        Time limit in HH:MM:SS format (default: "10:00:00")
    memory : str, optional
        Memory requirement in MB (default: "50000")
    nodes : int, optional
        Number of nodes (default: 2)
    tasks_per_node : int, optional
        Tasks per node (default: 24)
    image_path : str, optional
        Path to Singularity image (default: "/home/users/nrubio/SV_scripts/solver_latest.sif")
    svmultiphysics_path : str, optional
        Path to svMultiPhysics executable (default: "/home/users/nrubio/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics")
    output_file : str, optional
        Output shell script filename. If None, uses "{geo_name}.sh"
    
    Returns:
    --------
    str : Path to the generated shell script
    """
    nodes = 2
    tasks_per_node = 24
    partition = "amarsden"
    time_limit = "3:00:00"
    memory = "50000"
    image_path = "/home/users/nrubio/SV_scripts/solver_latest.sif"
    svmultiphysics_path = "/home/users/nrubio/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics"
    output_dir = "/scratch/users/nrubio/job_scripts"
    output_file = f"{geo_name}.sh"
    # Set default simulation directory
    simulation_dir = f'/scratch/users/nrubio/synthetic_junctions/CCO_trees/{set_name}/{geo_name}'
    
    # Set default output filename
    output_file = f"{geo_name}.sh"
    
    # Calculate total tasks
    total_tasks = nodes * tasks_per_node
    
    # Generate the shell script content
    shell_content = f"""#!/bin/bash 
#SBATCH --job-name={geo_name}
#SBATCH --partition={partition}
#SBATCH --output={output_dir}/{geo_name}.o%j
#SBATCH --error={output_dir}/{geo_name}.e%j
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={tasks_per_node}

export UCX_TLS=ib
export PMIX_MCA_gds=hash
export OMPI_MCA_btl_tcp_if_include=ib0
export F1='{simulation_dir}'
export IMAGE_PATH='{image_path}'

# Load Modules
module purge
module load openmpi
# Name of the executable you want to run
mpirun --mca mpi_cuda_support 0 -n {total_tasks} singularity run $IMAGE_PATH {svmultiphysics_path} $F1/fluid_simulation_0-0.xml
"""
    
    # Create output directory if it doesn't exist

    
    # Write the shell script
    threeD_dir = f"/Users/natalia/cursor_access/learn_lpns/data/threeD/{set_name}/{geo_name}"
    output_path = os.path.join(threeD_dir, output_file)
    with open(output_path, 'w') as f:
        f.write(shell_content)
    
    # Make the script executable

    
    print(f"Generated shell script: {output_file}")

    return
