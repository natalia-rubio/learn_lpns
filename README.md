# Learned Lumped-Parameter Networks

This repository contains functionality to train and deploy neural networks that predict lumped parameters (e.g. resistances, inductances) for 0D "electric circuit" models of cardiovascular flows.  The neural networks predict lumped parameters from the vascular geometry and are trained on high-fidelity 3D data.  This work is described in greater detail in this [paper](https://arxiv.org/abs/2604.01549).  A second, more lightweight repo, [learnedZeroD](https://github.com/natalia-rubio/learnedZeroD), provides functionality to convert a standard 0D model of a vasculature into the more accurate learned representation using pre-trained neural networks.


![Cross-validation pressure errors by modality](figures/github_figures.png)

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
| `util/zerod_calibration/` | 0D pipeline: preprocessing, calibration, forward sim, CV, run-config |
| `util/neural_network/` | JAX/Optax models and training |
| `util/data_processing/` | ML tables, jax arrays, train/val splits |
| `util/visualizations/` | CV metrics, diagnostics, comparison plots |
| `data/` | Bundled sample inputs + generated datasets (see [data/README.md](data/README.md)) |
| `results/` | Models, CV summaries, plots (gitignored) |
| `tests/` | Unit tests (no solver required) |

## Development

- Editable install with dev tools: `pip install -e ".[dev]"`
- Run unit tests (no C++ solver): `pytest -v`
- Lint source and tests: `ruff check util tests`
- CI: GitHub Actions runs `ruff check` and `pytest` on Python 3.10 and 3.12 for every push/PR

## Console entry points

| Command | Purpose |
| ------- | ------- |
| `learn-lpns-cv` | k-fold cross-validation |
| `learn-lpns-batch-zerod` | Batch 0D generation |
| `learn-lpns-data-processing` | Build ml_inputs and jax arrays |
| `learn-lpns-train` | Train junction/vessel NNs |
