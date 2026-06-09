# Usage

Run from the **repository root** (or use console entry points after `pip install -e ".[dev]"`).

The bundled sample cohort is **`VMR_aortas`** (5 geometries). See [data/README.md](../data/README.md).

## Quick start (cross-validation)

Exercises prerequisite checks, k-fold training (junction + vessel NNs), NN deploy on validation geometries, MSE aggregation, summary CSVs, and pressure-error barcharts.

```bash
learn-lpns-cv \
  --set_name VMR_aortas \
  --geometry_variant bifurcations_EL \
  --num_trials 2 \
  --run_config gen_loss
```

Defaults: `geometry_variant=bifurcations_EL`, `num_trials=5`, `run_config=gen_loss`. All pipeline flags use **keyword-only** underscores (e.g. `--set_name`, `--no_redo`).

## Run config (`--run_config`)

Physics and training variants are selected with a single `--run_config` flag on zerod CLIs, CV, batch, data processing, and NN training. Tokens are **underscore-separated** and **order-independent**; they resolve to a **canonical path suffix** under `data/` and `results/`.

### Tokens

See `learn_lpns/zerod_calibration/run_config_canonical.py`.

| Token | Effect |
| ----- | ------ |
| *(none)* | `base` — RI junction model, symmetric NN loss, no generation-weighted training |
| `quadratic_resistor` | RRI junction model (R + stenosis + L); off by default |
| `penalty_on` | Enable L2 calibration penalties (requires `quadratic_resistor`) |
| `asymmetric_loss` | Asymmetric NN loss (per-coefficient overestimate weights) |
| `gen_loss` | Generation-weighted NN training loss (`weight = scale / 2^generation`) |

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

### Migration from older folder names

| Old layout | New equivalent |
| ---------- | -------------- |
| `stenosis_off_*_gen_loss` | `gen_loss` |
| Default RRI with penalties | `quadratic_resistor_penalty_on_gen_loss` |
| `penalty_off_quadratic_resistor_*` | `quadratic_resistor_gen_loss` |

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
  --set_name VMR_aortas \
  --run_config gen_loss \
  --skip_steps calibration_forward \
  --geometries 0076_1001
```

`--skip_steps` tokens (order-independent): `base_generation`, `observation`, `calibration`, `forward`, `nn_inference`, `mse`, `plots`.

Other flags: `--no_redo`, `--NN_only`, `--NN_vessel`, `--skip_existing`.

## k-fold cross-validation

Each trial randomly splits geometries into train vs validation sets, trains junction (and vessel) NNs, deploys with `generate_zerod_inputs.py --NN_only` on validation geometries, and aggregates MSE from each geometry’s `mse_comparison.csv`. When CV finishes, `cv_pressure_max_pct_error_barchart` runs automatically unless `--skip_barchart`.

### What CV regenerates

| Stage | When | What runs |
| ----- | ---- | --------- |
| **Bootstrap** | Only if prerequisites missing | `batch_generate_zerod_inputs_vmr` + `run_data_processing` |
| **Per trial** | Always (val geometries) | `generate_zerod_inputs --NN_only` (NN inference, forward sim, MSE, plots) |

Bootstrap triggers when any of the following is true:

- ml_inputs missing for some geometry
- jax pickle missing for the cohort size
- calibrated zeroD incomplete (not missing NN forward files—those are created per trial)

When bootstrap runs, batch processes only missing geometries. Data processing uses all geometries if jax must be rebuilt, otherwise only the batch subset.

### CV flags

| Flag | Purpose |
| ---- | ------- |
| `--no_redo` | Passed to bootstrap batch only; skips recreating existing zeroD files |
| `--skip_training_if_exists` | Skip training if checkpoints exist for a trial |
| `--trial N` | Re-run only trial `N` (0-based); merges into existing summary CSV |
| `--metrics_only` | Rebuild summary CSVs from existing MSE files (no train/deploy) |
| `--plots_only` | Regenerate location comparison plots from existing zeroD data |
| `--skip_barchart` | Skip automatic barchart generation |

### CV outputs

- Models: `results/models/<set_name>/<run_config>/…_trial_<k>/`
- Splits: `data/split_indices/…`
- Summaries: `results/cross_validation/<set_name>/<run_config>/` (MSE and max-error CSVs, barchart PDFs)

Checkpoints: `rri_{set_name}_pred_{0,1,2}_model` (one scalar-output network per R/S/L coefficient).

## Neural network training

Each RRI coefficient (R, S, L) is a **separate single-output network** against one column of `output_rri`. Hyperparameters: `RRI_COEF_TRAIN_SPECS` in `launch_training.py`.

```bash
learn-lpns-train VMR_aortas 5 bifurcations_EL \
  --run_config gen_loss \
  --model_dir results/models/VMR_aortas/bifurcations_EL_trial_0
```

Positional args: `set_name`, `num_geos`, optional `geometry_variant` (default `all` → both `bifurcations` and `bifurcations_EL`).

| Flag | Purpose |
| ---- | ------- |
| `--run_config` | Path suffix for jax_arrays and split_indices |
| `--asymmetric_loss` | Per-coefficient overestimate weights |
| `--generation_weighted_loss` | Enable generation-weighted loss |
| `--generation_weighted_loss_scale` | Multiplier for generation weights (default 1.0) |
| `--vessel` | Train vessel NNs |
| `--leaky_relu` | Leaky ReLU activations |
| `--quiet_epochs` | Suppress per-epoch loss logging |
| `--split_path` / `--model_dir` | Override split pickle or output directory |

Bifurcation **generation** is stored in the jax pickle (not an NN input) and used only with generation-weighted loss.

## MSE modalities

Display names for console tables, CSV headers, and LaTeX exports: `learn_lpns/zerod_calibration/modality_paths.py`.

| Modality key | Display name |
| ------------ | ------------ |
| `geometric` | Standard |
| `BloodVesselJunction` | Calibrated |
| `BloodVesselJunction_NN` | Learned Junctions |
| `NN_vessel` | Learned Vessels |
| `BloodVesselJunction_NN_plus_Vessel_NN` | Learned Junctions and Vessels |

## Data processing

After zeroD outputs exist, build ml_inputs and jax stacks:

```bash
learn-lpns-data-processing \
  --set_name VMR_aortas \
  --geometry_variant bifurcations_EL \
  --run_config gen_loss
```

## Visualizations

Reporting scripts under `learn_lpns/visualizations/` (run as modules from repo root):

| Script | Purpose |
| ------ | ------- |
| `cv_cross_set_summary_barchart` | CV metrics across cohorts |
| `cv_max_pct_error_by_config_barchart` | Max pressure error by run-config |
| `cv_pressure_max_pct_error_barchart` | Per-trial CV bar chart |
| `cv_pressure_errors_to_latex` | CV metrics → LaTeX table |
| `cv_geometric_vs_calibrated_histograms` | Geometric vs calibrated parameter errors |
| `plot_location_comparison` | Pressure/flow vs time at observation locations |
| `plot_zero_d_parameter_bars` | 0D parameters by modality |
| `run_zerod_comparison_plots` | Common comparison plot workflows |

Most accept `--run_config`, `--set_name`, and `--geometry_variant`. See each module’s `--help` for defaults.

## Console entry points

| Command | Script |
| ------- | ------ |
| `learn-lpns-cv` | `learn_lpns/zerod_calibration/run_cross_validation.py` |
| `learn-lpns-batch-zerod` | `learn_lpns/zerod_calibration/batch_generate_zerod_inputs_vmr.py` |
| `learn-lpns-data-processing` | `learn_lpns/data_processing/run_data_processing.py` |
| `learn-lpns-train` | `learn_lpns/neural_network/launch_training.py` |
