# Data and results

## Data layout

The `data/` directory is organized by `set_name` (cohort of vascular geometries), `geometry_id` (each geometry in the cohort), and an optional `run_config` suffix.

| Subdirectory | Purpose |
| ------------ | ------- |
| `zeroD/` | Per-geometry simulation workspace and outputs (geometric inputs, calibrated configs, forward-simulation CSVs, MSE comparisons). VMR cohorts use `data/zeroD/<set_name>/standard-0d/` for reference solver JSONs. |
| `oneD/` | 3D solutions projected onto 1D centerlines for calibration. Sample projection scripts: `util/cluster_scripts/`. |
| `ml_inputs/` | Tabular features and lumped-parameter labels extracted from `zeroD/` outputs. |
| `jax_arrays/` | Stacked arrays for JAX training/inference; includes vessel-mode arrays when enabled. |
| `split_indices/` | Train/validation geometry splits for reproducible CV trials. |

## Bundled sample cohort (`VMR_aortas`)

Five adult aortic geometries from the [Vascular Model Repository](https://www.vascularmodel.com/) ship with the repo for local testing:

`0075_1001`, `0076_1001`, `0094_0001`, `0095_0001`, `0105_0001`

See [data/README.md](../data/README.md) for paths and `scripts/git-freeze-sample-data.sh` to ignore local edits to tracked seed files.

## Results layout

The `results/` directory holds trained models, cross-validation summaries, and derived plots—the complement to `data/`.

| Subdirectory | Purpose |
| ------------ | ------- |
| `models/` | Junction and vessel NN checkpoints, commonly `…/<set_name>/<run_config>/…_trial_<k>/` for CV. |
| `cross_validation/` | Aggregated metrics across trials and barchart PDFs per run-config. |

Script-specific outputs also appear under `results/location_comparison/`, `results/param_comparison/`, and `data/feature_histograms/<set_name>/<run_config>/<geometry_variant>/<set_type>/`.

Paths follow `set_name`, optional `run_config` suffix (matching `data/` and CLI defaults), and sometimes `geometry_variant` or a CV trial index.
