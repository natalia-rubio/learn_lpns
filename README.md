# Learned Lumped-Parameter Networks

This repository contains functionality to train and deploy neural networks that predict lumped parameters (e.g. resistances, inductances) for 0D "electric circuit" models of cardiovascular flows.  The neural networks predict lumped parameters from the vascular geometry and are trained on high-fidelity 3D data.  This work is described in greater detail in this [paper](https://arxiv.org/abs/2604.01549).  A second, more lightweight repo, [learnedZeroD](https://github.com/natalia-rubio/learnedZeroD), provides functionality to convert a standard 0D model of a vasculature into the more accurate learned representation using pre-trained neural networks.

** Documentation + production readiness in progress **

<img src="figures/github_figures.png" alt="Comparison of 3D finite element and 0D electric circuit models of blood flow" width="85%">

## Requirements

- **Python** 3.10 or newer
- **Python packages** (see Setup): core scientific stack plus **JAX**, **Optax**, and **dill** for training; **VTK** (`vtk` on PyPI) for geometric processing; **SciPy**,**matplotlib**, **pandas**.
- **`svzerodsolver`** and **`svzerodcalibrator`** binaries on `PATH`.  These applications are both supported by the repo [svZeroDSolver](https://github.com/SimVascular/svZeroDSolver).  Currently, this workflow is only compatible with [my fork](https://github.com/natalia-rubio/svZeroDPlus/tree/base_w_feat).


## Setup

Clone the repository and work from the repo root so imports like `util.*` resolve (scripts assume `REPO_ROOT` on `sys.path`).

```bash
git clone <YOUR_FORK_OR_REMOTE_URL> learn_lpns
cd learn_lpns

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
# JAX first (CPU/GPU variants): https://docs.jax.dev/en/latest/installation.html
pip install -U "jax[cpu]"
pip install -r requirements.txt

# Run modules from repo root, e.g.:
python -m util.zerod_calibration.batch_generate_zerod_inputs_vmr --help
```

**Notes**

- **`requirements.txt`** lists packages other than JAX; install **`jax[cpu]`** or a CUDA variant before `-r requirements.txt` so jax/jaxlib stay matched.
- Point **`PATH`** at your **svZeroDPlus** / solver install when running calibration or forward simulation steps.


## Functionality

The central workflow of this repo is as follows:

1. Pre-process 0D geometry and extract geometric features,
2. Find "ground truth" lumped paramters by calibration to 3D simulation data,
3. Train neural networks (NNs) to predict lumped parameters from geometric features,
4. Test the forward-simulation performace of 0D models with neural-network-predicted parameters.

$k$-fold cross-validation of this workflow is implemented, where $k$ different train and test sets are considered.  Scripts for visualizations are also provided.  

## Repository layout

| Path | Purpose |
|------|---------|
| `util/zerod_calibration/` | Core 0D pipeline: geometry pre-processing (split junctions with 3+ outlets into bifurcations, adjust junction-vessel boundaries based on a pseudo entrance length), calibration, forward simulation, CV orchestration (`run_cross_validation.py`), and run-config handling. |
| `util/neural_network/` | JAX/Optax model definitions, utilities, training launcher, and training loop for predicting lumped parameters from geometric features. |
| `util/data_processing/` | Builds ML-ready tables and JAX arrays from generated 0D outputs; creates split indices and helper summaries. |
| `util/visualizations/` | Plotting and reporting scripts for CV metrics, calibration diagnostics, and geometry/result inspection. |
| `util/cluster_scripts/` | Cluster helpers for centerline projection, VTU processing, and large-scale data generation workflows. |
| `util/tools/` | Shared lightweight utilities used across modules (e.g., dictionary save/load wrappers). |
| `data/` | Generated/working datasets (`zeroD`, `ml_inputs`, `jax_arrays`, `split_indices`) organized by set name and optional run-config suffix. |
| `results/` | Model artifacts and evaluation outputs (`results/models/...`, `results/cross_validation/...`). |
| `requirements.txt` | Python dependency list (install JAX separately first to match your platform/CUDA stack). |
| `README.template.md` | Project documentation template and onboarding notes for local setup/workflow. |

## Usage

Run from the **repository root** so imports and `--data-root data` resolve as expected.

### k-fold cross-validation (`run_cross_validation.py`)

Each trial randomly splits geometries into train vs validation sets, trains the junction NN (and vessel NN unless disabled), deploys with `generate_zerod_inputs.py --NN-only` on validation geometries, and aggregates MSE from each geometry’s `mse_comparison.csv`.

```bash
# Positional: <set_name> [geometry_variant] [num_trials]
# Defaults: geometry_variant=bifurcations_EL (EL refers to the entrance-length-adjusted vessel-junction definition), num_trials (k) =5
python util/zerod_calibration/run_cross_validation.py \
  VMR_abdo \
  bifurcations_EL \
  5 \
  --run-config symmetric_penalty_off_gen_loss \
```

Useful options: `--trial N` (re-run only trial `N`, 0-based), `--skip-training-if-exists`, `--no-redo`, `--no-NN-vessel`, `--metrics-only`, `--plots-only`. See `python util/zerod_calibration/run_cross_validation.py --help`.

**Outputs:** per-trial models under `results/models/<set_name>/<run_config>/…_trial_<k>/`, split pickles under `data/split_indices/…`, and a summary CSV under `results/cross_validation/<set_name>/<run_config>/<geometry_variant>_cv_summary.csv` (plus `_normalized` in the filename when normalization is enabled).


## Data

The `data/` directory is organized by `set_name` (the cohort of vascular geometries use for training and testing, e.g. VMR_abdo - abdominal aortas from the Vascular Model Repository), `geometry_id` (each geometry in the cohort), and (optionally) `run_config` suffix.


- `zeroD/`: per-geometry simulation workspace and outputs (generated geometric inputs, calibrated configs, forward-simulation result CSVs, downsampled CSVs, and MSE comparison files).
- `oneD/`: per-geometry 3D solutions projected onto 1D centerlines by integration over vessel cross-sections.  Used to calibrate the 0D model to find the ground truth lumped parameter values.  (Needs to be provided by user.  Sample script to project a 3D solution onto centerline in `util/cluster_scripts`.)
- `ml_inputs/`: per-geometry tabular features/targets extracted from `zeroD/` outputs for machine-learning training (e.g., geometric features and lumped-parameter labels).
- `jax_arrays/`: stacked, serialized arrays built from `ml_inputs/` for fast JAX training/inference; also stores vessel-mode arrays when enabled.
- `split_indices/`: saved train/validation geometry splits for reproducible CV trials (`trial_k` split files and matching geometry name lists).

## Results

The `results/` directory holds **trained models, cross-validation summaries, and derived plots or tables** produced by the pipeline and reporting utilities—the complement to `data/`, which stores inputs and intermediate datasets.  Paths usually follow `set_name`, an optional `run_config` suffix (matching `data/` and the CLI default), and sometimes `geometry_variant` or a CV trial index.

- `models/`: saved junction and vessel neural network checkpoints and metadata, commonly laid out as `…/<set_name>/<run_config>/…_trial_<k>/` for k-fold cross-validation.
- `cross_validation/`: aggregated metrics across trials.
- Script-specific outputs: visualization and post-processing tools write under predictable roots such as `location_comparison/` (plots of pressure and flow over time), `feature_histograms/` (histograms of geometrics features and lumped parameters in different sets), `param_comparison/` (comparison of standard, neural-net, and optimal lumped parameters)

