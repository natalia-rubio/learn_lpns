# Architecture

## Design decisions

- **Separate R / S / L networks** — Each RRI coefficient (linear resistor, stenosis resistor, inductor) is a single-output network with its own learning rate and architecture (`training.rri_coefficients` in `config/defaults.yaml`). This allows per-coefficient asymmetric loss weights and avoids one output dominating training.
- **Geometry-level CV splits** — Train/validation splits assign whole geometries, not individual junction rows, so validation measures generalization to unseen vasculatures.
- **Canonical `--run_config` paths** — Underscore tokens (e.g. `gen_loss`, `quadratic_resistor`) compose a single on-disk suffix under `data/` and `results/`, keeping physics variants and training modes reproducible without ad-hoc folder names (`run_config_canonical.py`).
- **Generate only missing prerequisites** — Cross-validation checks for absent cohort files (calibrated zeroD, ml_inputs, jax pickles) and generates only what is missing before trials run, reducing rerun cost.



## Public API

The package exposes a small set of functions for downstream tools and notebooks:

```python
from learn_lpns import (
    get_pipeline_config,
    load_pipeline_config,
    resolve_run_config_suffix,
    run_config_suffix_to_flags,
    modality_table_header,
    repo_root,
)
```

Also available from submodules (see `learn_lpns/__init__.py` and `learn_lpns/config/__init__.py`):

- **Config** — YAML loading, Pydantic models, calibration/solver parameter builders
- **Run config** — `run_config_canonical.py` token parsing and path suffix composition
- **Modalities** — `modality_paths.py` CSV column names and forward-sim file paths

CLI modules under `learn_lpns/zerod_calibration/`, `data_processing/`, and `neural_network/` are orchestration scripts. Run them via console entry points (`learn-lpns-cv`, etc.) after `pip install -e ".[dev]"`.

Repository root resolution for data paths lives in `learn_lpns.tools.paths.repo_root()` (shared by CLIs and config loading).
