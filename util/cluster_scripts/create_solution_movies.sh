#!/bin/bash
#SBATCH --job-name=create_movies
#SBATCH --partition=amarsden
#SBATCH --output=/scratch/users/nrubio/job_scripts/create_movies_%j.out
#SBATCH --error=/scratch/users/nrubio/job_scripts/create_movies_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=50000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
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
module load python/3.9.0
module load py-numpy/1.20.3_py39
module load py-scipy/1.6.3_py39
module load py-scikit-learn/1.0.2_py39
module load gcc/10.1.0

# Note: pyvista and moviepy may need to be installed via pip
# pip install --user pyvista moviepy

# Set working directory
cd ~/SV_scripts/cluster_scripts

echo "Starting movie generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Example usage - modify as needed:
# For 3D movies:
# python3 create_solution_movies.py --3d /scratch/users/nrubio/synthetic_junctions/CCO_trees/set_3/tree_006/48-procs --output /scratch/users/nrubio/movies/tree_006_pressure_3d.mp4 --field pressure

# For 1D movies:
# python3 create_solution_movies.py --1d /scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees/set_3/tree_006/unsteady_soln.vtp --output /scratch/users/nrubio/movies/tree_006_pressure_1d.mp4 --field pressure

# Get arguments from command line
INPUT_TYPE=$1  # "--3d" or "--1d"
INPUT_PATH=$2  # Directory or file path
OUTPUT_PATH=$3  # Output movie path
FIELD=${4:-pressure}  # Field name (default: pressure)

if [ -z "$INPUT_TYPE" ] || [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: sbatch create_solution_movies.sh <--3d|--1d> <input_path> <output_path> [field]"
    echo "Example: sbatch create_solution_movies.sh --3d /path/to/48-procs /path/to/output.mp4 pressure"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
if [ ! -z "$OUTPUT_DIR" ] && [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Check if xvfb-run is available (recommended for headless rendering)
if command -v xvfb-run &> /dev/null; then
    echo "Using xvfb-run for headless rendering"
    xvfb-run -a python3 create_solution_movies_vtk.py "$INPUT_TYPE" "$INPUT_PATH" --output "$OUTPUT_PATH" --field "$FIELD"
else
    echo "xvfb-run not found, trying direct execution (may fail without X display)"
    echo "To install: module load x11 or install xorg-x11-server-Xvfb"
    python3 create_solution_movies_vtk.py "$INPUT_TYPE" "$INPUT_PATH" --output "$OUTPUT_PATH" --field "$FIELD"
fi

echo "End time: $(date)"
echo "Job completed"

