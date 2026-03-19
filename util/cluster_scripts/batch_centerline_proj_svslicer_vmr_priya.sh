#!/bin/bash
#SBATCH --job-name=vmr_priya_centerline
#SBATCH --partition=amarsden
#SBATCH --output=/scratch/users/nrubio/job_scripts/vmr_priya_centerline_%j.out
#SBATCH --error=/scratch/users/nrubio/job_scripts/vmr_priya_centerline_%j.err
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

# Base and output on Sherlock
export VMR_PRIYA_BASE_DIR=/scratch/users/nrubio/VMR_priya_coa
export VMR_PRIYA_OUTPUT_DIR=/scratch/users/nrubio/VMR_priya_coa_reduced_results

# Parallelism
export PYTHON_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

NUM_PROCS=${1:-48}
SIMULATION=${2:-}

echo "Starting VMR_priya_coa centerline projection"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "Base: $VMR_PRIYA_BASE_DIR"
echo "Output: $VMR_PRIYA_OUTPUT_DIR"
echo "Num procs: $NUM_PROCS"
echo "Workers: $PYTHON_NUM_WORKERS"
echo "Start time: $(date)"

if [ -n "$SIMULATION" ]; then
    echo "Simulation filter: $SIMULATION"
    python3 batch_centerline_proj_svslicer_vmr_priya.py --num-procs "$NUM_PROCS" --simulation "$SIMULATION"
else
    python3 batch_centerline_proj_svslicer_vmr_priya.py --num-procs "$NUM_PROCS"
fi

echo "End time: $(date)"
echo "Job completed"
