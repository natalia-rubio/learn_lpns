# Learned Lumped-Parameter Networks

This repository contains functionality to train and deploy neural networks that predict lumped parameters (e.g. resistances, inductances) for 0D "electric circuit" models of cardiovascular flows.  The neural networks predict lumped parameters from the vascular geometry and are trained on high-fidelity 3D data.  This work is described in greater detail in this [paper](https://arxiv.org/abs/2604.01549).  A second, more lightweight repo, [learnedZeroD](https://github.com/natalia-rubio/learnedZeroD), provides functionality to convert a standard 0D model of a vasculature into the more accurate learned representation using pre-trained neural networks.

## Requirements

- **Python** 3.10 or newer
- **Python packages** (see Setup): core scientific stack plus **JAX**, **Optax**, and **dill** for training; **VTK** (`vtk` on PyPI) for geometric processing; **SciPy**,**matplotlib**, **pandas**.
- `**svzerodsolver`** and `**svzerodcalibrator**` binaries (`SVZEROD_INSTALL_DIR` or `PATH`).  These applications are both supported by the repo [svZeroDSolver](https://github.com/SimVascular/svZeroDSolver).  Currently, this workflow is only compatible with [my fork](https://github.com/natalia-rubio/svZeroDPlus/tree/J-J_wiring) on branch `**J-J_wiring**` (built by `scripts/setup_cross_validation.sh`).

## Setup

Clone the repository and work from the repo root so imports like `util.*` resolve (scripts assume `REPO_ROOT` on `sys.path`).

**One-shot setup** (clone/build svZeroDPlus, Python venv, verify sample data):

```bash
# macOS: install cmake + Python 3.10+ first (Xcode python3 is often 3.9)
brew install cmake python@3.12
PYTHON=$(brew --prefix python@3.12)/bin/python3.12 ./scripts/setup_cross_validation.sh
source scripts/cv_env.sh
```

**Manual setup** (if you already cloned the repo and prefer to manage Python yourself):

Prerequisites for the solver build: `git`, `cmake`, and a C++ compiler (macOS: `xcode-select --install`; `brew install cmake`).

```bash
git clone <YOUR_FORK_OR_REMOTE_URL> learn_lpns
cd learn_lpns

# Python 3.10+ required (macOS: brew install python@3.12)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
# JAX first (CPU/GPU variants): https://docs.jax.dev/en/latest/installation.html
pip install -U "jax[cpu]"
pip install -r requirements.txt

# Clone/build svZeroDPlus fork (sibling ../svZeroDPlus/Release) — required for calibration/forward sim
./scripts/setup_cross_validation.sh --skip-clone --skip-python
source scripts/cv_env.sh

# Run modules from repo root, e.g.:
python -m util.zerod_calibration.batch_generate_zerod_inputs_vmr --help
python util/zerod_calibration/run_cross_validation.py --help
```

**Notes**

- `requirements.txt` lists packages other than JAX; install `jax[cpu]` or a CUDA variant before `-r requirements.txt` so jax/jaxlib stay matched.
- `setup_cross_validation.sh` clones [svZeroDPlus](https://github.com/natalia-rubio/svZeroDPlus) branch `J-J_wiring` into `../svZeroDPlus` and builds `svzerodsolver` + `svzerodcalibrator`. Use `--skip-python` if you already created `.venv`; use `--skip-clone` if you only want to rebuild the solver.
- `scripts/cv_env.sh` exports `SVZEROD_INSTALL_DIR` (default: `../svZeroDPlus/Release`). Source it in each shell, or set `SVZEROD_INSTALL_DIR` / add the binaries to `PATH` yourself.

## Functionality

The central workflow of this repo is as follows:

1. Pre-process 0D geometry and extract geometric features,
2. Find "ground truth" lumped parameters by calibration to 3D simulation data,
3. Train neural networks (NNs) to predict lumped parameters from geometric features,
4. Test the forward-simulation performance of 0D models with neural-network-predicted parameters.

$k$-fold cross-validation of this workflow is implemented, where $k$ different train and test sets are considered.  Scripts for visualizations are also provided.  

## Repository layout


| Path                      | Purpose                                                                                                                                                                                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `util/zerod_calibration/` | Core 0D pipeline: geometry pre-processing (split junctions with 3+ outlets into bifurcations, adjust junction-vessel boundaries based on a pseudo entrance length), calibration, forward simulation, CV orchestration (`run_cross_validation.py`), and run-config handling. |
| `util/neural_network/`    | JAX/Optax model definitions, utilities, and training loop. Trains three single-output networks (R, S, L) via `launch_training.py`.                                                                                                                                          |
| `util/data_processing/`   | Builds ML-ready tables and JAX arrays from generated 0D outputs; creates split indices and helper summaries.                                                                                                                                                                |
| `util/visualizations/`    | Plotting and reporting scripts for CV metrics, calibration diagnostics, and geometry/result inspection.                                                                                                                                                                     |
| `util/cluster_scripts/`   | Cluster helpers for centerline projection, VTU processing, and large-scale data generation workflows.                                                                                                                                                                       |
| `util/tools/`             | Shared lightweight utilities used across modules (e.g., dictionary save/load wrappers).                                                                                                                                                                                     |
| `data/`                   | Bundled sample inputs plus generated working datasets (`zeroD`, `ml_inputs`, `jax_arrays`, `split_indices`). Only seed inputs are tracked in Git; see `data/README.md`.                                                                                                     |
| `results/`                | Model artifacts and evaluation outputs (`results/models/...`, `results/cross_validation/...`).                                                                                                                                                                              |
| `requirements.txt`        | Python dependency list (install JAX separately first to match your platform/CUDA stack).                                                                                                                                                                                    |
| `README.template.md`      | Project documentation template and onboarding notes for local setup/workflow.                                                                                                                                                                                               |


## Usage

Run from the **repository root** so imports and `--data_root data` resolve as expected.

A **sample VMR cohort** (`VMR_aortas`, 5 geometries) is included under `data/`; see [data/README.md](data/README.md). All commands below use `--set_name VMR_aortas` so they run on bundled data without extra downloads.

### Quick start (cross-validation)

This command exercises most of the pipeline: prerequisite checks, k-fold training (junction + vessel NNs), NN deploy on validation geometries, MSE aggregation, summary CSVs, and pressure-error barcharts.

```bash
python util/zerod_calibration/run_cross_validation.py \
  --set_name VMR_aortas \
  --geometry_variant bifurcations_EL \
  --num_trials 2 \
  --run_config gen_loss
```

All arguments are keyword flags (e.g. `--set_name`, `--geometry_variant`, `--num_trials`). Defaults: `geometry_variant=bifurcations_EL`, `num_trials=5`, `run_config=gen_loss`.

### Run config (`--run_config`)

Physics and training variants are selected with a single `**--run_config**` flag on the zerod CLIs, CV, batch, data processing, and NN training. Tokens are **underscore-separated** and **order-independent**; they resolve to a **canonical path suffix** under `data/` and `results/`.

**Tokens** (see `util/zerod_calibration/run_config_canonical.py`):


| Token                | Effect                                                                             |
| -------------------- | ---------------------------------------------------------------------------------- |
| *(none)*             | `**base`** — RI junction model, symmetric NN loss, no generation-weighted training |
| `quadratic_resistor` | RRI junction model (R + stenosis + L); **off by default**                          |
| `penalty_on`         | Enable L2 calibration penalties (requires `quadratic_resistor`)                    |
| `asymmetric_loss`    | Asymmetric NN loss (per-coefficient overestimate weights)                          |
| `gen_loss`           | Generation-weighted NN training loss (`weight = scale / 2^generation`)             |


**Alias:** `generation_weighted_loss` parses as `gen_loss` (on-disk suffix remains `_gen_loss`).

**Default CLI:** `gen_loss` (`DEFAULT_CLI_RUN_CONFIG`).

**Canonical suffix composition order:** `quadratic_resistor` → `penalty_on` → `asymmetric_loss` → `gen_loss`.

Examples (token order does not matter):

```text
gen_loss
  →  gen_loss

quadratic_resistor_gen_loss
  →  quadratic_resistor_gen_loss

gen_loss_quadratic_resistor_asymmetric_loss
  →  quadratic_resistor_asymmetric_loss_gen_loss

base
  →  base   (explicit no optional tokens)
```

**Migration from older folder names** (re-process or rename data trees; old names are not accepted as passthrough):


| Old layout                         | New equivalent                           |
| ---------------------------------- | ---------------------------------------- |
| `stenosis_off_*_gen_loss`          | `gen_loss`                               |
| Default RRI with penalties         | `quadratic_resistor_penalty_on_gen_loss` |
| `penalty_off_quadratic_resistor_`* | `quadratic_resistor_gen_loss`            |


On-disk layout (when run-config is set):

```text
data/zeroD/<set_name>/<run_config>/<geo_id>/...
data/ml_inputs/<set_name>/<run_config>/<geometry_variant>/<geo_id>/...
data/jax_arrays/<set_name>/<run_config>/<geometry_variant>/<set_type>/...
data/split_indices/<set_name>/<run_config>/...
results/models/<set_name>/<run_config>/...
results/cross_validation/<set_name>/<run_config>/...
```

Implementation: `util/zerod_calibration/run_config_canonical.py` (token parsing, composition, flag derivation) and `util/zerod_calibration/generate_zerod_inputs_cli.py` (shared argparse for single-geo and batch entry points).

**CLI convention:** pipeline entry points use **keyword-only** flags with **underscores** (e.g. `--set_name`, `--geometry_variant`, `--run_config`, `--num_trials`, `--no_redo`).

### Batch 0D generation (`batch_generate_zerod_inputs_vmr.py`)

Runs `generate_zerod_inputs.py` over VMR geometries discovered from `data/zeroD/<set_name>/standard-0d/`.

```bash
python -m util.zerod_calibration.batch_generate_zerod_inputs_vmr \
  --set_name VMR_aortas \
  --run_config gen_loss \
  --skip_steps calibration_forward \
  --geometries 0076_1001
```

`**--skip_steps**` — skip pipeline stages without separate flags. Tokens (order-independent): `base_generation`, `observation`, `calibration`, `forward`, `nn_inference`, `mse`, `plots`. Example: `base_generation_observation_calibration` or `calibration_forward`.

Other useful flags: `--no_redo` (skip recreating files that already exist), `--NN_only`, `--NN_vessel`, `--skip_existing` (skip geometries that already have full outputs including NN forward results).

### k-fold cross-validation (`run_cross_validation.py`)

Each trial randomly splits geometries into train vs validation sets, trains the junction NN (and vessel NN unless disabled), deploys with `generate_zerod_inputs.py --NN_only` on validation geometries, and aggregates MSE from each geometry’s `mse_comparison.csv`. When CV finishes, `**cv_pressure_max_pct_error_barchart**` runs automatically (all three pressure metrics) unless you pass `**--skip_barchart**`.

**What CV regenerates**


| Stage                 | When                              | What runs                                                                 |
| --------------------- | --------------------------------- | ------------------------------------------------------------------------- |
| **Bootstrap** (start) | Only if prerequisites are missing | `batch_generate_zerod_inputs_vmr` + `run_data_processing`                 |
| **Per trial**         | Always (val geometries)           | `generate_zerod_inputs --NN_only` (NN inference, forward sim, MSE, plots) |


Bootstrap runs only when **any** of the following is true for the run-config:

- ml_inputs missing (`geometric_features.csv` / `junction_lumped_parameters.csv`) for some geometry
- jax pickle missing for the cohort size
- **calibrated** zeroD incomplete (missing calibrated JSON/CSV) — **not** missing NN forward files (those are created per trial at deploy)

When bootstrap runs, batch processes **only the missing geometries**, not the whole cohort. Data processing uses all geometries if jax must be rebuilt, otherwise only the batch subset.

- `**--no_redo`**: passed to bootstrap batch only; skips recreating zeroD files that already exist. Does not affect per-trial NN deploy (deploy does not pass `--no_redo`, so NN forward outputs are refreshed each trial).
- `**--skip_training_if_exists**`: skip training for a trial if model checkpoints already exist.
- `**--trial N**`: re-run only trial `N` (0-based); merges into existing summary CSV.
- `**--metrics_only**`: rebuild summary CSVs from existing per-geometry MSE files (no train/deploy); still runs barcharts unless `--skip_barchart`.
- `**--plots_only**`: regenerate location comparison plots from existing zeroD data.

**Outputs:** per-trial models under `results/models/<set_name>/<run_config>/…_trial_<k>/`, split pickles under `data/split_indices/…`, summary CSVs under `results/cross_validation/<set_name>/<run_config>/` (including pressure/flow MSE and max-error variants), and barchart PDFs in the same directory (e.g. `bifurcations_EL_max_pct_error.pdf`).

Saved junction/vessel checkpoints use the short names `rri_{set_name}_pred_{0,1,2}_model` (one scalar-output network per R/S/L coefficient).

### Neural network training (`launch_training.py`)

Training lives in `util/neural_network/`. Each RRI coefficient (linear R, stenosis S, inductor L) is a **separate single-output network** trained against one column of `output_rri` (`target_output_column` 0/1/2). Hyperparameters per coefficient are defined in `RRI_COEF_TRAIN_SPECS` inside `launch_training.py`.

```bash
python util/neural_network/launch_training.py \
  VMR_aortas 5 bifurcations_EL \
  --run_config gen_loss \
  --model_dir results/models/VMR_aortas/bifurcations_EL_trial_0
```

Positional args: `set_name`, `num_geos`, optional `geometry_variant` (default `all` → trains both `bifurcations` and `bifurcations_EL`). Use `--geometry_variant` when combined with `--vessel` so flag order does not matter.

Useful flags:


| Flag                               | Purpose                                                                                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `--run_config`                     | Path suffix for `jax_arrays` and `split_indices` (default: `gen_loss`); `_gen_loss` in the suffix also enables generation-weighted loss |
| `--asymmetric_loss`                | Per-coefficient asymmetric overestimate weights (also derived from run-config suffix)                                                   |
| `--generation_weighted_loss`       | Explicitly enable generation-weighted loss                                                                                              |
| `--generation_weighted_loss_scale` | Overall multiplier for generation weights (default: 1.0)                                                                                |
| `--vessel`                         | Train vessel NNs (separate jax arrays and `_vessel` model suffix)                                                                       |
| `--leaky_relu`                     | Leaky ReLU activations                                                                                                                  |
| `--quiet_epochs`                   | Suppress per-epoch loss logging (verbose by default)                                                                                    |
| `--split_path` / `--model_dir`     | Override split pickle or output directory (CV sets `--model_dir` per trial)                                                             |


Bifurcation **generation** is stored in the jax pickle (not as an NN input feature) and used only when generation-weighted loss is enabled.

### MSE modalities and summary tables

Forward-simulation and CV summaries compare several **modalities** (0D model variants). Display names for console tables, CSV headers, and LaTeX exports are centralized in `util/zerod_calibration/modality_paths.py`:


| Modality key                            | Display name                  |
| --------------------------------------- | ----------------------------- |
| `geometric`                             | Standard                      |
| `BloodVesselJunction`                   | Calibrated                    |
| `BloodVesselJunction_NN`                | Learned Junctions             |
| `NN_vessel`                             | Learned Vessels               |
| `BloodVesselJunction_NN_plus_Vessel_NN` | Learned Junctions and Vessels |


### Data processing

After zeroD outputs exist, build ml_inputs and jax stacks:

```bash
python util/data_processing/run_data_processing.py \
  --set_name VMR_aortas \
  --geometry_variant bifurcations_EL \
  --run_config gen_loss
```

Feature histograms and related paths follow the same `set_name / run_config / geometry_variant` layout as jax arrays.

### Visualizations

Reporting scripts under `util/visualizations/` (run as modules from repo root):


| Script                                  | Purpose                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| `cv_cross_set_summary_barchart`         | Grouped bar chart comparing CV metrics across cohorts (e.g. Aortic, Pulmonary) |
| `cv_max_pct_error_by_config_barchart`   | Horizontal bar chart of max pressure error by run-config                       |
| `cv_pressure_max_pct_error_barchart`    | Per-trial CV bar chart with modality comparison                                |
| `cv_pressure_errors_to_latex`           | CV pressure metrics → LaTeX table                                              |
| `cv_geometric_vs_calibrated_histograms` | Histograms of geometric vs calibrated parameter errors                         |
| `plot_location_comparison`              | Pressure/flow vs time at observation locations                                 |
| `plot_zero_d_parameter_bars`            | Bar charts of 0D parameters by modality                                        |
| `run_zerod_comparison_plots`            | Wrapper for common comparison plot workflows                                   |


Most accept `--run_config`, `--set_name`, and `--geometry_variant` (underscore keyword flags). See each module’s `--help` for defaults.

## Data

The `data/` directory is organized by `set_name` (the cohort of vascular geometries used for training and testing), `geometry_id` (each geometry in the cohort), and (optionally) `run_config` suffix.

### Bundled sample cohort (`VMR_aortas`)

Five adult aortic geometries from the [Vascular Model Repository](https://www.vascularmodel.com/) are included for local testing. Geometries: `0075_1001`, `0076_1001`, `0094_0001`, `0095_0001`, `0105_0001`. See [data/README.md](data/README.md) for layout and `scripts/git-freeze-sample-data.sh` if you want Git to ignore local edits to tracked seed files.

- `zeroD/`: per-geometry simulation workspace and outputs (generated geometric inputs, calibrated configs, forward-simulation result CSVs, downsampled CSVs, and MSE comparison files). VMR cohorts use `data/zeroD/<set_name>/standard-0d/` for reference solver input JSONs; centerlines for VMR live under `data/oneD/VMR/<geo_id>/unsteady_soln.vtp`.
- `oneD/`: per-geometry 3D solutions projected onto 1D centerlines by integration over vessel cross-sections.  Used to calibrate the 0D model to find the ground truth lumped parameter values.  (Needs to be provided by user.  Sample script to project a 3D solution onto centerline in `util/cluster_scripts`.)
- `ml_inputs/`: per-geometry tabular features/targets extracted from `zeroD/` outputs for machine-learning training (e.g., geometric features and lumped-parameter labels).
- `jax_arrays/`: stacked, serialized arrays built from `ml_inputs/` for fast JAX training/inference; also stores vessel-mode arrays when enabled.
- `split_indices/`: saved train/validation geometry splits for reproducible CV trials (`trial_k` split files and matching geometry name lists).

## Results

The `results/` directory holds **trained models, cross-validation summaries, and derived plots or tables** produced by the pipeline and reporting utilities—the complement to `data/`, which stores inputs and intermediate datasets.  Paths usually follow `set_name`, an optional `run_config` suffix (matching `data/` and the CLI default), and sometimes `geometry_variant` or a CV trial index.

- `models/`: saved junction and vessel neural network checkpoints and metadata, commonly laid out as `…/<set_name>/<run_config>/…_trial_<k>/` for k-fold cross-validation.
- `cross_validation/`: aggregated metrics across trials and barchart PDFs per run-config.
- Script-specific outputs: `location_comparison/` (pressure/flow vs time), `param_comparison/` (geometric vs calibrated vs NN parameters), and `feature_histograms/` under `data/feature_histograms/<set_name>/<run_config>/<geometry_variant>/<set_type>/`.

