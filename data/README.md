# Sample data (provided with the repository)

  I provide a **minimal aortic demo cohort** from the [Vascular Model Repository](https://www.vascularmodel.com/) so the pipeline can be run without sourcing inputs elsewhere.  The specific files come from the work [Pfaller et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/cnm.3639) and the repo [richter2024-paper-tools](https://github.com/StanfordCBCL/richter2024-paper-tools/tree/main/data/geometric_pfaller22/input).

The larger files (1D VTPs and 0D JSONs) are stored with **[Git LFS](https://git-lfs.com)**. After clone:

```bash
brew install git-lfs   # once per machine (macOS)
git lfs install        # once per user
git lfs pull           # download LFS objects for this repo
# or:
./scripts/setup_git_lfs.sh
```

`make notebook` and `scripts/setup_cross_validation.sh` run the LFS pull/check automatically when possible.

## Layout


| Path                                                                                         | Contents                                                                                                                 |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `zeroD/VMR_aortas/standard-0d/*.json`                                                        | Standard 0D solver input JSONs (one per geometry)                                                                        |
| `zeroD/VMR_aortas/gen_loss/<geo>/bifurcations_EL_calibrated_output_BloodVesselJunction.json` | Calibrated ground truth on pre-processed bifurcations_EL 0D input file (used in notebook demo, tracked)                  |
| `oneD/VMR/<geo_id>/unsteady_soln.vtp`                                                        | 1D centerline solutions (3D solutions projected onto centerlines) used for geometric feature extraction and calibration. |


**Active notebook cohort (5 geometries):** `0129_0000`, `0154_0001`, `0174_0000`, `0175_0000`, `0176_0000`

## Not included (generated locally)

These stay **gitignored** — the pipeline or notebook writes them under `data/` on your machine:

- `bifurcations_EL_geometric_input.json` and other files under `zeroD/VMR_aortas/gen_loss/` (beyond the calibrated JSON above)
- `ml_inputs/`, `jax_arrays/`, `split_indices/` — run `learn-lpns-data-processing`, then train with `learn-lpns-train --set_name VMR_aortas --run_config gen_loss` (`num_geos` inferred from the jax pickle names)
- `results/`

## Local edits vs Git

Only the seed inputs above are tracked. After clone, if you want Git to **ignore local modifications** to those tracked files (so pipeline reruns never show up as commits), run from repo root:

```bash
./scripts/git-freeze-sample-data.sh
```

That sets `skip-worktree` on tracked paths under `data/`. To undo: `./scripts/git-freeze-sample-data.sh --unfreeze`.
