#!/bin/bash
#SBATCH --job-name=tree_centerline_proj
#SBATCH --partition=amarsden
#SBATCH --output=/scratch/users/nrubio/job_scripts/tree_centerline_proj
#SBATCH --error=/scratch/users/nrubio/job_scripts/tree_centerline_proj
#SBATCH --time=006:00:00
#SBATCH --mem=50000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24

module purge
module load openmpi
module load openblas
module load system
module load x11
module load mesa
module load viz
module load gcc
module load valgrind
module load python/3.9.0
module load py-numpy/1.20.3_py39
module load py-scipy/1.6.3_py39
module load py-scikit-learn/1.0.2_py39
module load gcc/10.1.0

python3 unsteady_tree_centerline_proj.py tree_10 24
