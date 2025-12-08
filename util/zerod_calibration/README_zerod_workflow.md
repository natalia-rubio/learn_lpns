# svZeroDSolver Input Generation and Calibration Workflow

This workflow generates standard svZeroDSolver input files from centerline geometry and simulation results, similar to the workflow in `richter2024-paper-tools`.

## Overview

The workflow consists of three main steps:

1. **Generate Geometric 0D Input Files**: Extract vessel geometry from centerline VTP files and create initial svZeroDSolver input JSON files with geometric parameters (resistance, capacitance, inductance).

2. **Generate Calibration Input Files**: Extract observation data (pressure and flow) from 3D or 1D simulation results and create calibration input files with all elements set to zero.

3. **Run Calibration**: Execute svZeroDCalibrator to optimize 0D element values to match the observed data.

## Scripts

### `generate_zerod_inputs.py`

Main script to process a single geometry.

**Usage:**
```bash
python3 generate_zerod_inputs.py \
  --set-name set_1 \
  --geo-name tree_000 \
  [--centerline path/to/centerline.vtp] \
  [--one-d-soln path/to/unsteady_soln.vtp] \
  [--output-dir data/zeroD] \
  [--skip-calibration]
```

**Arguments:**
- `--set-name`: Set name (required, e.g., `set_1`)
- `--geo-name`: Geometry name (required, e.g., `tree_000`)
- `--centerline`: Path to centerline VTP file (optional, auto-detected if not provided)
- `--one-d-soln`: Path to 1D centerline solution VTP file (optional, auto-detected if not provided)
- `--output-dir`: Output directory for 0D files (default: `data/zeroD`)
- `--skip-calibration`: Skip calibration step (only generate geometric input)

**Output Files:**
- `{output_dir}/{set_name}/{geo_name}/geometric_input.json`: Geometric 0D input file
- `{output_dir}/{set_name}/{geo_name}/calibration_input.json`: Calibration input file (if calibration enabled)
- `{output_dir}/{set_name}/{geo_name}/calibrated_output.json`: Calibrated output file (if calibration enabled)

### `batch_generate_zerod_inputs.py`

Batch script to process all geometries in a set.

**Usage:**
```bash
python3 batch_generate_zerod_inputs.py set_1 \
  [--geo-name tree_000] \
  [--skip-calibration] \
  [--output-dir data/zeroD]
```

**Arguments:**
- `set_name`: Set name (required)
- `--geo-name`: Process only this geometry (optional)
- `--skip-calibration`: Skip calibration step
- `--output-dir`: Output directory for 0D files (default: `data/zeroD`)

### `batch_generate_zerod_inputs.sh`

SLURM job script for batch processing on the cluster.

**Usage:**
```bash
sbatch batch_generate_zerod_inputs.sh set_1 [--skip-calibration]
```

## Workflow Details

### Step 1: Geometric 0D Input Generation

The script reads the centerline VTP file and:

1. **Extracts vessel segments**: Groups centerline points by `BranchId` and creates vessel segments
2. **Calculates geometric parameters**:
   - **Length**: Sum of distances between consecutive points in each branch
   - **Area**: Average of `CenterlineSectionArea` along the segment
   - **Radius**: Calculated from area: `r = sqrt(area / π)`
   - **Resistance (R_poiseuille)**: `R = 8*μ*L / (π*r^4)`
   - **Capacitance (C)**: `C = 3*π*r^3*L / (2*E*h)` (using typical wall stiffness)
   - **Inductance (L)**: `L = ρ*L / A`

3. **Identifies junctions**: Detects branch connections (simplified tree structure)
4. **Creates boundary conditions**:
   - Inlet BC: `INFLOW` (FLOW type)
   - Outlet BCs: `OUT1`, `OUT2`, etc. (RESISTANCE type)

### Step 2: Calibration Input Generation

The script extracts observation data from 1D centerline solution VTP files:

1. **Finds timestep arrays**: Identifies `pressure_*` and `velocity_*` (or `flow_*`) arrays
2. **Extracts data at boundaries**: Finds inlet (GID=0) and outlet points
3. **Refines curves**: Uses cubic spline interpolation to create `NUM_OBS` (100) observation points
4. **Creates calibration input**: 
   - Copies geometric input structure
   - Sets all element values to zero
   - Adds observation data (`y` and `dy` arrays)
   - Adds calibration parameters

### Step 3: Calibration

Runs `pysvzerod.calibrate()` to optimize element values:

- **Calibration parameters**:
  - `tolerance_gradient`: 1e-5
  - `tolerance_increment`: 1e-10
  - `maximum_iterations`: 100
  - `calibrate_stenosis_coefficient`: True
  - `set_capacitance_to_zero`: False

## Dependencies

- **Python packages**:
  - `vtk` (for reading VTP files)
  - `numpy` (for array operations)
  - `scipy` (for cubic spline interpolation)
  - `pysvzerod` (for running simulations and calibration)
    - Install with: `pip install svzerodsolver`

- **Input files**:
  - Centerline VTP file with:
    - `BranchId` array
    - `CenterlineSectionArea` array
    - `Path` array (or will be calculated from points)
    - `GlobalNodeId` array (for identifying inlet/outlets)
  - 1D solution VTP file with:
    - `pressure_*` arrays (one per timestep)
    - `velocity_*` or `flow_*` arrays (one per timestep)

## File Structure

```
data/
├── threeD/
│   └── {set_name}/
│       └── {geo_name}/
│           └── centerlines_simVascular.vtp
├── oneD/
│   └── {set_name}/
│       └── {geo_name}/
│           └── unsteady_soln.vtp
├── reduced_results/  (alternative location, or /scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees/)
│   └── {set_name}/
│       └── {geo_name}/
│           └── unsteady_soln.vtp
└── zeroD/
    └── {set_name}/
        └── {geo_name}/
            ├── geometric_input.json
            ├── calibration_input.json
            └── calibrated_output.json
```

**Note:** The script looks for 1D solutions in this order:
1. `data/oneD/{set_name}/{geo_name}/unsteady_soln.vtp` (primary location)
2. `data/reduced_results/{set_name}/{geo_name}/unsteady_soln.vtp`
3. `/scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees/{set_name}/{geo_name}/unsteady_soln.vtp`

## Example

Process a single geometry:
```bash
python3 generate_zerod_inputs.py \
  --set-name set_1 \
  --geo-name tree_000 \
  --output-dir data/zeroD
```

Process all geometries in a set:
```bash
python3 batch_generate_zerod_inputs.py set_1
```

Or submit as a SLURM job:
```bash
sbatch batch_generate_zerod_inputs.sh set_1
```

## Notes

- The junction detection is simplified and assumes a tree structure. For complex geometries, you may need to manually adjust the junction definitions.
- The calibration requires `pysvzerod` to be installed. If not available, use `--skip-calibration` to only generate geometric inputs.
- The script automatically searches for centerline and solution files in common locations if not explicitly provided.
- 3D solution extraction is not yet implemented (only 1D centerline solutions are supported).

## Troubleshooting

1. **"BranchId array not found"**: Ensure the centerline VTP file contains the `BranchId` array (generated by SimVascular centerline extraction).

2. **"No 1D or 3D solution found"**: The script will skip calibration if no solution file is found. Check that:
   - The 1D solution file exists at the expected location
   - The file contains `pressure_*` and `velocity_*` arrays

3. **"pysvzerod not available"**: Install with `pip install svzerodsolver` or use `--skip-calibration`.

4. **Calibration fails**: Check that:
   - The observation data is valid (non-zero, reasonable values)
   - The geometric input file is valid
   - The calibration parameters are appropriate

