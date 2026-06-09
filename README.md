# Learned Lumped-Parameter Networks

This repository contains functionality to train and deploy neural networks that predict lumped parameters (e.g. resistances, inductances) for 0D "electric circuit" models of cardiovascular flows.  The neural networks predict lumped parameters from the vascular geometry and are trained on high-fidelity 3D data.  This work is described in greater detail in this [paper](https://arxiv.org/abs/2604.01549).  A second, more lightweight repo, [learnedZeroD](https://github.com/natalia-rubio/learnedZeroD), provides functionality to convert a standard 0D model of a vasculature into the more accurate learned representation using pre-trained neural networks.


!<p align="center">
  <img src="assets/github_figures.png" alt="3D-0D schematic" width="85%">
</p>


### Inputs:
* **1D centerline solution file (vtp)**: This file contains a 1D centerline representation of the geometry and the 3D simulation results projected onto the 1D centerline.  The 1D geometry vtp file for a patient-specific anatomy can be generated with the [SimVascular ROM Simulation Tool](https://simvascular.github.io/documentation/rom_simulation.html).  3D simulation results can be projected onto the centerline by integration over centerline-normal cross sections, example code [here](https://github.com/natalia-rubio/projection_scripts_3d_1d).

* **Standard 0D input file (json)**: This file contains the standard SimVascular 0D representation of the vasculature and the simulation boundary conditions.  The 0D input file can also be generated with the [SimVascular ROM Simulation Tool](https://simvascular.github.io/documentation/rom_simulation.html).

### Outputs:
* **`learn-lpns-batch-zerod`**:
  * Augmented 0D input files: contain extra geometric information and modified junction-vessel discretization (saved in **data/zeroD**) 
  * Calibrated 0D input files: contains the optimal (ground truth) resistances and inductances (saved in **data/zeroD**)
  * Tabulated geometry (neural network features) and lumped parameter (neural network target) data, saved in human-readable csvs (**data/ml_inputs**)and pickled dictionaries of jax arrays (**data/jax_arrays**).  Train-validation splits also generated.  (Can be generated independently with `learn-lpns-data-processing`, run `learn-lpns-batch-zerod` first.)
  * Learned 0D input files: contains neural-network predicted resistances and inductances (saved in **data/zeroD**).  Generated only if trained neural networks are available (``learn-lpns-train`` has been run).
  * Forward 0D simulation results:  Flow and pressure results generated for standard, calibrated, and learned input files, as available.  csv files containing results for each 0D node at each timestep, plots of each solution (compared to the 3D solution) in time at a specified (generally inlet) node.  Printed table listing inlet pressure MSE with respect to 3D simulation.

* **``learn-lpns-train``**: Trained neural networks.

* **`learn-lpns-cv`**: Results of k-fold validation in csv and barchart visualization (saved to **results/cross-validation**).  Also generates all the above outputs in the proceess.


## Requirements

- **Python** 3.10+
- **Packages:** JAX, Optax, NumPy, pandas, SciPy, matplotlib, VTK, dill (see Setup)
- **Binaries:** `svzerodsolver` and `svzerodcalibrator` — built from [svZeroDPlus fork](https://github.com/natalia-rubio/svZeroDPlus/tree/J-J_wiring) (`J-J_wiring` branch) via `scripts/setup_cross_validation.sh`


## Setup

```bash
# One-shot (macOS: brew install cmake python@3.12 first)
PYTHON=$(brew --prefix python@3.12)/bin/python3.12 ./scripts/setup_cross_validation.sh
source scripts/cv_env.sh
```

Manual install:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -U "jax[cpu]"          # or CUDA variant — install JAX first
pip install -e ".[dev]"
./scripts/setup_cross_validation.sh --skip-clone --skip-python
source scripts/cv_env.sh
```

Install JAX before `pip install -e ".[dev]"` so jax/jaxlib stay matched. See [docs/usage.md](docs/usage.md) for all CLI commands.

## Quick start


```bash
./scripts/setup_cross_validation.sh
source scripts/cv_env.sh
learn-lpns-cv --set_name VMR_aortas --geometry_variant bifurcations_EL --num_trials 2 --run_config gen_loss
```

Bundled sample: **`VMR_aortas`** (5 geometries, ~30–60 min on CPU). Outputs: `results/cross_validation/VMR_aortas/gen_loss/bifurcations_EL_cv_summary.csv` and barchart PDFs.

```bash
pytest -v    # unit tests, no C++ solver required
```


## Documentation

| Guide | Contents |
| ----- | -------- |
| [docs/usage.md](docs/usage.md) | Run config, batch generation, CV, training, visualizations |
| [docs/architecture.md](docs/architecture.md) | Pipeline diagram, design decisions, tradeoffs |
| [docs/data_and_results.md](docs/data_and_results.md) | Data layout, sample cohort, output paths |
| [data/README.md](data/README.md) | Bundled VMR seed inputs |

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `learn_lpns/zerod_calibration/` | 0D pipeline: preprocessing, calibration, forward sim, CV, run-config |
| `learn_lpns/neural_network/` | JAX/Optax models and training |
| `learn_lpns/data_processing/` | ML tables, jax arrays, train/val splits |
| `learn_lpns/visualizations/` | CV metrics, diagnostics, comparison plots |
| `data/` | Bundled sample inputs + generated datasets (see [data/README.md](data/README.md)) |
| `results/` | Models, CV summaries, plots (gitignored) |
| `tests/` | Unit tests (no solver required) |

## Development

- Editable install with dev tools: `pip install -e ".[dev]"`
- Run unit tests (no C++ solver): `pytest -v`
- Lint source and tests: `ruff check learn_lpns tests`
- CI: GitHub Actions runs `ruff check` and `pytest` on Python 3.10 and 3.12 for every push/PR

## Console entry points

| Command | Purpose |
| ------- | ------- |
| `learn-lpns-cv` | k-fold cross-validation |
| `learn-lpns-batch-zerod` | Batch 0D generation |
| `learn-lpns-data-processing` | Build ml_inputs and jax arrays |
| `learn-lpns-train` | Train junction/vessel NNs |
