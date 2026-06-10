# Sample data (bundled with the repository)

This tree ships a **minimal VMR demo cohort** so you can run the pipeline without sourcing inputs elsewhere.

Large files (1D VTPs and 0D JSONs) are stored with **[Git LFS](https://git-lfs.com)**. After clone:

```bash
brew install git-lfs   # once per machine (macOS)
git lfs install        # once per user
git lfs pull           # download LFS objects for this repo
# or:
./scripts/setup_git_lfs.sh
```

`make notebook` and `scripts/setup_cross_validation.sh` run the LFS pull/check automatically when possible.

## Layout

| Path | Contents |
|------|----------|
| `zeroD/VMR_aortas/standard-0d/*.json` | Reference 0D solver input JSONs (one per geometry) |
| `zeroD/VMR_aortas/gen_loss/<geo>/bifurcations_EL_calibrated_output_BloodVesselJunction.json` | Calibrated ground truth on bifurcations_EL topology (notebook demo, tracked) |
| `oneD/VMR/<geo_id>/unsteady_soln.vtp` | 1D centerline solutions (3D projected onto centerlines) used for topology steps |
| `oneD/VMR/<geo_id>/centerlines_EL_labeled.vtp` | Optional labeled centerline (where present) |

**Active notebook cohort (5):** `0129_0000`, `0154_0001`, `0174_0000`, `0175_0000`, `0176_0000`

Per geometry, the notebook expects (tracked in git where noted):

| Path | Purpose |
|------|---------|
| `zeroD/VMR_aortas/standard-0d/<geo>.json` | Reference standard 0D solver input (tracked) |
| `zeroD/VMR_aortas/gen_loss/<geo>/bifurcations_EL_calibrated_output_BloodVesselJunction.json` | Calibrated ground truth / NN training labels (tracked) |
| `oneD/VMR/<geo>/unsteady_soln.vtp` | Centerline geometry for split + entrance-length steps (tracked) |

Section 1b of [examples/nn_parameter_comparison.ipynb](../examples/nn_parameter_comparison.ipynb) builds `bifurcations_EL_geometric_input.json` locally from standard-0d + VTP (no svZeroDPlus). The calibrated bifurcations_EL JSON is bundled as-is.

**Archived original demo cohort (5):** `0075_1001`, … — under `data2/` (gitignored).

## CLI `set_name`

Use **`VMR_aortas`** for scripts that take `--set_name` (batch generation, cross-validation, data processing).
Projected solutions live under `data/oneD/VMR/`; the code resolves that path automatically when `set_name` contains `VMR`.

Example:

```bash
python -m learn_lpns.zerod_calibration.batch_generate_zerod_inputs_vmr \
  --set_name VMR_aortas \
  --run_config gen_loss \
  --geometries 0076_1001
```

## Not included (generated locally)

These stay **gitignored** — the pipeline or notebook writes them under `data/` on your machine:

- `bifurcations_EL_geometric_input.json` and other files under `zeroD/VMR_aortas/gen_loss/` (beyond the calibrated JSON above)
- `ml_inputs/`, `jax_arrays/`, `split_indices/`
- `results/`

## Local edits vs Git

Only the seed inputs above are tracked. After clone, if you want Git to **ignore local modifications** to those tracked files (so pipeline reruns never show up as commits), run from repo root:

```bash
./scripts/git-freeze-sample-data.sh
```

That sets `skip-worktree` on tracked paths under `data/`. To undo: `./scripts/git-freeze-sample-data.sh --unfreeze`.

## License / attribution

Confirm you have rights to redistribute VMR-derived geometry and simulation data before publishing forks. Add citation and VMR/SimVascular attribution as required by your data sources.
