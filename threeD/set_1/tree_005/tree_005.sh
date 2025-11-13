#!/bin/bash 
#SBATCH --job-name=tree_005
#SBATCH --partition=amarsden
#SBATCH --output=/scratch/users/nrubio/job_scripts/tree_005.o%j
#SBATCH --error=/scratch/users/nrubio/job_scripts/tree_005.e%j
#SBATCH --time=3:00:00
#SBATCH --mem=50000
#SBATCH --nodes=2
#SBATCH --tasks-per-node=24

export UCX_TLS=ib
export PMIX_MCA_gds=hash
export OMPI_MCA_btl_tcp_if_include=ib0
export F1='/scratch/users/nrubio/synthetic_junctions/CCO_trees/set_1/tree_005'
export IMAGE_PATH='/home/users/nrubio/SV_scripts/solver_latest.sif'

# Load Modules
module purge
module load openmpi
# Name of the executable you want to run
mpirun --mca mpi_cuda_support 0 -n 48 singularity run $IMAGE_PATH /home/users/nrubio/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics $F1/fluid_simulation_0-0.xml
