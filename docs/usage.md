# Usage

Run from the **repository root** (or use console entry points after `pip install -e ".[dev]"`).

The provided sample cohort is **`VMR_aorta_starter`** (5 geometries for notebook/CV demos). The full aortic cohort is **`VMR_aorta`**. See [data/README.md](../data/README.md).

All four console entry points use **keyword flags** with underscores (e.g. `--set_name`, `--run_config`, `--geometry_variant`).

## Quick start (cross-validation)

Exercises prerequisite checks, k-fold training (junction + vessel NNs), NN deploy on validation geometries, MSE aggregation, summary CSVs, and pressure-error barcharts.

```bash
learn-lpns-cv \
  --set_name VMR_aorta_starter \
  --geometry_variant bifurcations_EL \
  --num_trials 2 \
  --run_config gen_loss
```

Defaults: `geometry_variant=bifurcations_EL`, `num_trials=5`, `run_config=gen_loss`.

## Run config (`--run_config`)

Physics and training variants are selected with a single `--run_config` flag on zerod CLIs, CV, batch, data processing, and NN training. Tokens are **underscore-separated** and **order-independent**; they resolve to a **canonical path suffix** under `data/` and `results/`.

### Tokens

See `learn_lpns/zerod_calibration/run_config_canonical.py`.


| Token                | Effect                                                                         |
| -------------------- | ------------------------------------------------------------------------------ |
| *(none)*             | `base` — RI junction model, symmetric NN loss, no generation-weighted training |
| `quadratic_resistor` | RRI junction model (R + stenosis + L); off by default                          |
| `penalty_on`         | Enable L2 calibration penalties (requires `quadratic_resistor`)                |
| `asymmetric_loss`    | Asymmetric NN loss (per-coefficient overestimate weights)                      |
| `gen_loss`           | Generation-weighted NN training loss (`weight = 1 / B^generation`; B from config) |


**Alias:** `generation_weighted_loss` → `gen_loss` (on-disk suffix remains `_gen_loss`).

**Default CLI:** `gen_loss`.

**Canonical suffix order:** `quadratic_resistor` → `penalty_on` → `asymmetric_loss` → `gen_loss`.

Examples (token order does not matter):

```text
gen_loss                          →  gen_loss
quadratic_resistor_gen_loss       →  quadratic_resistor_gen_loss
gen_loss_quadratic_resistor_asymmetric_loss  →  quadratic_resistor_asymmetric_loss_gen_loss
base                              →  base
```

### On-disk layout

```text
data/zeroD/<set_name>/<run_config>/<geo_id>/...
data/ml_inputs/<set_name>/<run_config>/<geometry_variant>/<geo_id>/...
data/jax_arrays/<set_name>/<run_config>/<geometry_variant>/<set_type>/...
data/split_indices/<set_name>/<run_config>/...
results/models/<set_name>/<run_config>/...
results/cross_validation/<set_name>/<run_config>/...
```

Implementation: `run_config_canonical.py` (token parsing, composition, flag derivation) and `generate_zerod_inputs_cli.py` (shared argparse).

## Batch 0D generation

Runs `generate_zerod_inputs.py` over VMR geometries from `data/zeroD/<set_name>/standard-0d/`.

```bash
learn-lpns-batch-zerod \
  --set_name VMR_aorta_starter \
  --run_config gen_loss \
  --skip_steps calibration_forward \
  --geometries 0076_1001
```

`--skip_steps` tokens (order-independent): `base_generation`, `observation`, `calibration`, `forward`, `nn_inference`, `mse`, `plots`.

Other flags: `--no_redo`, `--NN_only`, `--Vessel_NN`, `--skip_existing`.

### Calibration backend (`--calibration_backend`)

Step 3 (calibration) uses a **pure-Python decoupled least-squares fit** by default. Each vessel segment and each `BloodVesselJunction` outlet is fit independently from local 3D pressure/flow observations (`R`, `L`, and optionally stenosis `S`). Capacitance `C` stays from the geometric JSON.

```bash
learn-lpns-batch-zerod \
  --set_name VMR_aorta_starter \
  --run_config gen_loss \
  --calibration_backend decoupled_ls
```

| Backend        | Requirement                         | Notes |
| -------------- | ----------------------------------- | ----- |
| `decoupled_ls` | NumPy only (**default**)            | Fast; no `svzerodcalibrator` binary |
| `svzerod`      | `SVZEROD_INSTALL_DIR` + calibrator  | C++ Levenberg–Marquardt on equation residuals |

**Tradeoffs (decoupled vs C++):**

- Decoupled fitting ignores global network coupling; calibrated `R`/`L`/`S` generally **differ** from `svzerodcalibrator`.
- Stenosis uses observed `Q` in the `\|Q\|Q` term (linear in `S` given `Q`).
- If a local fit’s relative RMS pressure-drop error exceeds **10%**, a `UserWarning` is emitted (calibration still completes).
- Optional **`--plot_rsl_fits`**: per-element \(\Delta P\) vs \(Q\) scatter + fit line under `results/RSL_fits/<set_name>/<geo_name>/` (decoupled_ls only; default off).
- Forward simulation and MSE steps still use **svzerodsolver** unless replaced separately.

## k-fold cross-validation

Each trial randomly splits geometries into train vs validation sets, trains junction (and vessel) NNs, deploys with `generate_zerod_inputs.py --NN_only` on validation geometries, and aggregates MSE from each geometry’s `mse_comparison.csv`. When CV finishes, `cv_pressure_max_pct_error_barchart` runs automatically unless `--skip_barchart`.

### What CV regenerates


| Stage                       | When                      | What runs                                                                 |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| **Prerequisite generation** | Only if files are missing | `batch_generate_zerod_inputs_vmr` + `run_data_processing`                 |
| **Per trial**               | Always (val geometries)   | `generate_zerod_inputs --NN_only` (NN inference, forward sim, MSE, plots) |


Prerequisite generation runs when any of the following is true:

- ml_inputs missing for some geometry
- jax pickle missing for the cohort size
- calibrated zeroD incomplete (not missing NN forward files—those are created per trial)

When that step runs, batch processing generates only missing geometries. Data processing uses all geometries if jax must be rebuilt, otherwise only the batch subset.

### CV flags


| Flag                        | Purpose                                                                          |
| --------------------------- | -------------------------------------------------------------------------------- |
| `--no_redo`                 | During prerequisite batch generation only; skips recreating existing zeroD files |
| `--skip_training_if_exists` | Skip training if pre-trained models exist for a trial                            |
| `--trial N`                 | Re-run only trial `N` (0-based); merges into existing summary CSV                |
| `--metrics_only`            | Rebuild summary CSVs from existing MSE files (no train/deploy)                   |
| `--plots_only`              | Regenerate location comparison plots from existing zeroD data                    |
| `--skip_barchart`           | Skip automatic barchart generation                                               |


### CV outputs

- Models: `results/models/<set_name>/<run_config>/…_trial_<k>/`
- Splits: `data/split_indices/…`
- Summaries: `results/cross_validation/<set_name>/<run_config>/` (MSE and max-error CSVs, barchart PDFs)

Checkpoints: `rri_{set_name}_pred_{0,1,2}_model` (one scalar-output network per R/S/L coefficient).

## Neural network training

Each RRI coefficient (R, S, L) is a **separate single-output network** against one column of `output_rri`. Hyperparameters: `training.rri_coefficients` in `config/defaults.yaml`.

All arguments are **keyword flags** (e.g. `--set_name`, `--run_config`), consistent with the other pipeline CLIs.

```bash
# Minimal: defaults to bifurcations_EL; num_geos inferred from jax_arrays + split_indices
learn-lpns-train --set_name VMR_aorta_starter --run_config gen_loss

# Override output directory (e.g. CV trial or notebook demo)
learn-lpns-train --set_name VMR_aorta_starter \
  --run_config gen_loss \
  --model_dir results/models/VMR_aorta_starter/bifurcations_EL_trial_0

# Explicit cohort size and/or geometry variant
learn-lpns-train --set_name VMR_aorta_starter --num_geos 5
learn-lpns-train --set_name VMR_aorta_starter --geometry_variant all
```

**Defaults:** `geometry_variant=bifurcations_EL`, `run_config=gen_loss`. If `--num_geos` is omitted, it is inferred from `data/jax_arrays/{set_name}/{run_config}/bifurcations_EL/all/jax_arrays_num_geos_*.pkl`, choosing the **largest** cohort size that also has a matching file under `data/split_indices/...`. Pass `--split_path` (as CV does for trial splits) to take `num_geos` from that path instead.

**Note on `all`:** The string `all` appears in two unrelated places today. As `--geometry_variant all`, it means train **both** `bifurcations` and `bifurcations_EL`. As the **`set_type`** path segment under `jax_arrays/` and `split_indices/` (e.g. `.../bifurcations_EL/all/...`), it labels a cohort folder tier—the default tier used by training and data processing (other tiers such as `forward` exist for per-geometry inference pickles). This is confusing; naming will be revised in a future release to separate geometry-variant “train both” from path-tier labels.


| Flag                               | Purpose                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `--set_name`                       | Cohort name (required)                                                  |
| `--run_config`                     | Path suffix for jax_arrays and split_indices                            |
| `--num_geos`                       | Cohort size (overrides inference from jax_arrays / split_indices)       |
| `--geometry_variant`               | `bifurcations`, `bifurcations_EL`, or `all` (default: `bifurcations_EL`) |
| `--asymmetric_loss`                | Per-coefficient overestimate weights                                    |
| `--generation_weighted_loss`       | Enable generation-weighted loss                                         |
| `--generation_weighted_loss_decay_base` | Per-generation decay base B (weight = 1 / B^generation; default from `config/defaults.yaml`) |
| `--oracle_inputs`                  | Append R/S/L targets to inputs for training sanity check (not for deploy) |
| `--vessel`                         | Train vessel NNs                                                        |
| `--leaky_relu` / `--no-leaky_relu` | Leaky ReLU activations (default from `training.leaky_relu` in config)   |
| `--quiet_epochs`                   | Suppress per-epoch loss logging                                         |
| `--split_path` / `--model_dir`     | Override split pickle or output directory                               |


Bifurcation **generation** is stored in the jax pickle (not an NN input) and used only with generation-weighted loss.

## Notebook example (no C++ solver)

[examples/nn_parameter_comparison.ipynb](../examples/nn_parameter_comparison.ipynb) walks through a five-geometry demo (`VMR_aorta_starter`: `0129_0000`, `0154_0001`, `0174_0000`, `0175_0000`, `0176_0000`) without running calibration or forward simulation:

1. **Setup** — verify provided inputs per geometry (`standard-0d/<geo>.json`, `gen_loss/<geo>/bifurcations_EL_calibrated_output_BloodVesselJunction.json`, `oneD/VMR/<geo>/unsteady_soln.vtp`); set `SEED` for an 80% geometry-level train/val split (4 train / 1 val).
2. **Geometric pre-processing** — for all five geometries, build `bifurcations_EL_geometric_input.json` in Python via `materialize_bifurcations_el_from_base` (bifurcation split + entrance-length adjustment from standard-0d + VTP; no svZeroDPlus).
3. **Calibration** — skipped; uses the provided calibrated JSON as ground truth.
4. **Training data** — run `learn-lpns-data-processing` to build `ml_inputs`, `jax_arrays`, and geometry-level split indices.
5. **Training** — train junction and vessel NNs with `learn-lpns-train` (six models: R, S, L for junctions and vessels). After step 4, `learn-lpns-train --set_name VMR_aorta_starter --run_config gen_loss` is enough (`bifurcations_EL` and `num_geos=5` are inferred from the generated data).
6. **Inference** — predict R, S, L on the held-out geometry and write learned 0D JSON configs (`bifurcations_EL_NN_JunctionOnly.json`, etc.).
7. **Parameter comparison plot** — bar chart of standard, calibrated, and learned R/S/L per element on the validation geometry.
8. **Forward simulation** — skipped; notebook shows a reference inlet comparison figure (requires `svzerodsolver` to reproduce).

```bash
make notebook    # one-shot: venv, JAX, [dev,notebook], open Jupyter
```

Manual install:

```bash
pip install -e ".[dev,notebook]"
jupyter notebook examples/nn_parameter_comparison.ipynb
```

Provided inputs per geometry: standard-0d JSON, calibrated bifurcations_EL JSON, and 1D centerline VTP. The geometric bifurcations_EL JSON is generated in step 2. See [data/README.md](../data/README.md).

## MSE modalities

Display names for console tables, CSV headers, and LaTeX exports: `learn_lpns/zerod_calibration/modality_paths.py`.


| Modality key                            | Display name                  |
| --------------------------------------- | ----------------------------- |
| `geometric`                             | Standard                      |
| `BloodVesselJunction`                   | Calibrated                    |
| `BloodVesselJunction_NN`                | Learned Junctions             |
| `Vessel_NN`                             | Learned Vessels               |
| `BloodVesselJunction_NN_plus_Vessel_NN` | Learned Junctions and Vessels |


## Data processing

After zeroD outputs exist, build ml_inputs and jax stacks:

```bash
learn-lpns-data-processing \
  --set_name VMR_aorta_starter \
  --geometry_variant bifurcations_EL \
  --run_config gen_loss
```

### Stenosis clipping (`data_processing.stenosis_clipping`)

When enabled (default), data processing writes **unclipped** and **clipped** jax pickles:

- `jax_arrays_num_geos_{N}_unclipped.pkl` — full cohort reference
- `jax_arrays_num_geos_{N}_clipped.pkl` — default train/val split; S clipped in labels
- `jax_arrays_num_geos_{N}_trial_{T}_clipped.pkl` — per CV trial (written by `learn-lpns-cv`)

Bounds are computed from **train rows only** with `generation <= clip_generation_number` (default 1). Junction and vessel each get separate S bounds. Inspection CSVs are exported under `data/ml_inputs/.../clipped_labels/default/` or `trial_{T}/`.

Config keys in `config/defaults.yaml`:

```yaml
data_processing:
  stenosis_clipping:
    enabled: true
    clip_generation_number: 1
```

Training loads the clipped pickle matching `--split_path` (trial splits load `_trial_{T}_clipped`). Inference applies the same S bounds from model checkpoints (`stenosis_clip_bounds`). Re-run data processing after changing `clip_generation_number`.

## Visualizations

Reporting scripts under `learn_lpns/visualizations/` (run as modules from repo root):


| Script                                  | Purpose                                        |
| --------------------------------------- | ---------------------------------------------- |
| `cv_cross_set_summary_barchart`         | CV metrics across cohorts                      |
| `cv_max_pct_error_by_config_barchart`   | Max pressure error by run-config               |
| `cv_pressure_max_pct_error_barchart`    | Per-trial CV bar chart                         |
| `cv_pressure_errors_to_latex`           | CV metrics → LaTeX table                       |
| `cv_geometric_vs_calibrated_histograms` | Geometric vs calibrated parameter errors       |
| `plot_location_comparison`              | Pressure/flow vs time at observation locations |
| `plot_zero_d_parameter_bars`            | 0D parameters by modality                      |
| `run_zerod_comparison_plots`            | Common comparison plot workflows               |


Most accept `--run_config`, `--set_name`, and `--geometry_variant`. See each module’s `--help` for defaults.

## Console entry points


| Command                      | Script                                                            | Notes                                                                 |
| ---------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| `learn-lpns-cv`              | `learn_lpns/zerod_calibration/run_cross_validation.py`            | `--set_name` required; default `geometry_variant=bifurcations_EL`     |
| `learn-lpns-batch-zerod`     | `learn_lpns/zerod_calibration/batch_generate_zerod_inputs_vmr.py` | Per-geometry 0D pipeline over a cohort                                |
| `learn-lpns-data-processing` | `learn_lpns/data_processing/run_data_processing.py`               | Builds `ml_inputs`, `jax_arrays`, `split_indices`                     |
| `learn-lpns-train`           | `learn_lpns/neural_network/launch_training.py`                    | `--set_name` required; infers `--num_geos` when omitted               |
