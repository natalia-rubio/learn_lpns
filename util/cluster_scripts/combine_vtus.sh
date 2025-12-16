#!/bin/bash
#SBATCH --job-name=batch_centerline_proj
#SBATCH --partition=amarsden
#SBATCH --output=/scratch/users/nrubio/job_scripts/batch_centerline_proj_%j.out
#SBATCH --error=/scratch/users/nrubio/job_scripts/batch_centerline_proj_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Load modules
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

# Set working directory
cd ~/SV_scripts/cluster_scripts

# Get set name from command line argument
SET_NAME=${1:-set_3}
NUM_PROCS=${2:-48}
NUM_THREADS=${3:-}

# Use Python method directly (svSlicer has issues)
export SVSLICER_USE_PYTHON_ONLY=1

# Set number of workers for parallel processing (use available CPUs)
export PYTHON_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "Starting batch centerline projection for set: $SET_NAME"
echo "Using $NUM_PROCS-procs for simulation files"
echo "Using $PYTHON_NUM_WORKERS parallel workers"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run the script
if [ -z "$NUM_THREADS" ]; then
    python3 combine_vtus.py "$SET_NAME" "$NUM_PROCS"
else
    python3 combine_vtus.py "$SET_NAME" "$NUM_PROCS" "$NUM_THREADS"
fi

echo "End time: $(date)"
echo "Job completed"

