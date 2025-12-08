#!/bin/bash
#SBATCH --job-name=gen_zerod
#SBATCH --partition=normal
#SBATCH --output=gen_zerod_%j.out
#SBATCH --error=gen_zerod_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

# Load necessary modules
module load openmpi
module load openblas
module load system
module load gcc/10.1.0
module load python/3.9.0
module load py-numpy/1.20.3_py39
module load py-scipy/1.6.3_py39

# Set working directory
cd ~/SV_scripts/cluster_scripts

# Run batch script
# Usage: sbatch batch_generate_zerod_inputs.sh set_1 [--skip-calibration]
python3 batch_generate_zerod_inputs.py "$@"

