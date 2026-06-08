# Sample data (bundled with the repository)

This tree ships a **minimal VMR demo cohort** so you can run the pipeline without sourcing inputs elsewhere.

## Layout

| Path | Contents |
|------|----------|
| `zeroD/VMR_aortas/standard-0d/*.json` | Reference 0D solver input JSONs (one per geometry) |
| `oneD/VMR/<geo_id>/unsteady_soln.vtp` | 1D centerline solutions (3D projected onto centerlines) used for calibration |
| `oneD/VMR/<geo_id>/centerlines_EL_labeled.vtp` | Optional labeled centerline (where present) |

**Geometries (5):** `0075_1001`, `0076_1001`, `0094_0001`, `0095_0001`, `0105_0001`

## CLI `set_name`

Use **`VMR_aortas`** for scripts that take `--set_name` (batch generation, cross-validation, data processing).  
Projected solutions live under `data/oneD/VMR/`; the code resolves that path automatically when `set_name` contains `VMR`.

Example:

```bash
python -m util.zerod_calibration.batch_generate_zerod_inputs_vmr \
  --set_name VMR_aortas \
  --run_config gen_loss \
  --geometries 0076_1001
```

## Not included (generated locally)

These stay **gitignored** — the pipeline writes them under `data/` on your machine:

- `zeroD/VMR_aortas/<run_config>/…` — per-geometry 0D outputs
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
