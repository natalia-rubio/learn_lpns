# Data and results

## Data layout

The `data/` directory is organized by `set_name` (cohort of vascular geometries), `geometry_id` (each geometry in the cohort), and an optional `run_config` suffix.

| Subdirectory | Purpose |
| ------------ | ------- |
| `zeroD/` | Per-geometry simulation workspace and outputs (geometric inputs, calibrated configs, forward-simulation CSVs, MSE comparisons). VMR cohorts use `data/zeroD/<set_name>/standard-0d/` for reference solver JSONs. |
| `oneD/` | 3D solutions projected onto 1D centerlines for calibration. Example projection code: [projection_scripts_3d_1d](https://github.com/natalia-rubio/projection_scripts_3d_1d). |
| `ml_inputs/` | Tabular features and lumped-parameter labels extracted from `zeroD/` outputs. |
| `jax_arrays/` | Stacked arrays for JAX training/inference; includes vessel-mode arrays when enabled. Junction pickles are named `jax_arrays_num_geos_{N}.pkl` (vessel: `jax_arrays_vessel_num_geos_{N}.pkl`) where `N` is the cohort size. |
| `split_indices/` | Train/validation geometry splits for reproducible CV trials. Files are named `train_val_ind_{set_name}_num_geos_{N}` (CV trials append `_trial_{k}`). |

## Provided sample cohort (`VMR_aorta_starter`)

Five adult aortic geometries from the [Vascular Model Repository](https://www.vascularmodel.com/) ship with the repo for local testing and the example notebook:

`0129_0000`, `0154_0001`, `0174_0000`, `0175_0000`, `0176_0000`

This is separate from the full **`VMR_aorta`** cohort (more geometries), used for cross-validation and production experiments. See [data/README.md](../data/README.md) for paths.

## Results layout

The `results/` directory holds trained models, cross-validation summaries, and derived plots—the complement to `data/`.

| Subdirectory | Purpose |
| ------------ | ------- |
| `models/` | Junction and vessel NN checkpoints, commonly `…/<set_name>/<run_config>/…_trial_<k>/` for CV. |
| `cross_validation/` | Aggregated metrics across trials and barchart PDFs per run-config. |

Script-specific outputs also appear under `results/location_comparison/`, `results/param_comparison/`, and `data/feature_histograms/<set_name>/<run_config>/<geometry_variant>/<set_type>/`.

Paths follow `set_name`, optional `run_config` suffix (matching `data/` and CLI defaults), and sometimes `geometry_variant` or a CV trial index.
