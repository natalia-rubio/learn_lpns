# Setup scripts

Shell helpers for environment setup and sample data. Run from the **repository root** unless noted.

| Script | Purpose | Typical use |
| ------ | ------- | ----------- |
| [`setup_cross_validation.sh`](setup_cross_validation.sh) | Full CV stack: clone/update `svZeroDPlus`, build `svzerodsolver` + `svzerodcalibrator`, create `.venv`, `pip install` JAX + editable `[dev]`, verify sample data, write `cv_env.sh` | First-time setup for `learn-lpns-cv` |
| [`setup_notebook.sh`](setup_notebook.sh) | Notebook stack: `.venv`, JAX + editable `[dev,notebook]`, Git LFS pull/check, write `notebook_env.sh`; optional `--launch` opens Jupyter | `make notebook` or `make notebook-setup` |
| [`setup_git_lfs.sh`](setup_git_lfs.sh) | Install Git LFS hooks, `git lfs pull`, verify bundled VTP/JSON are materialized (not pointer stubs) | After clone; also `make lfs` |
| [`cv_env.sh`](cv_env.sh) | **Generated** — sets `SVZEROD_INSTALL_DIR`, prepends solver to `PATH`, activates `.venv`, `cd` to repo | `source scripts/cv_env.sh` before CV |
| [`notebook_env.sh`](notebook_env.sh) | **Generated** — activates `.venv`, `cd` to repo | `source scripts/notebook_env.sh` before Jupyter |
| [`git-freeze-sample-data.sh`](git-freeze-sample-data.sh) | `git update-index --skip-worktree` on tracked files under `data/` so local pipeline edits do not show in `git status` | Optional after clone; `--unfreeze` to undo |

## Makefile targets

| Target | Runs |
| ------ | ---- |
| `make lfs` | `./scripts/setup_git_lfs.sh` |
| `make notebook` | `./scripts/setup_notebook.sh --launch` |
| `make notebook-setup` | `./scripts/setup_notebook.sh` (no Jupyter launch) |

## Dependency source of truth

Python packages are installed from **`pyproject.toml`** (`pip install -e ".[dev]"` or `[dev,notebook]`).

## Regenerating env snippets

`cv_env.sh` and `notebook_env.sh` contain machine-specific paths. Re-run the corresponding setup script after moving the repo or rebuilding solvers:

```bash
./scripts/setup_cross_validation.sh --skip-clone --skip-solver-build   # refresh cv_env.sh only
./scripts/setup_notebook.sh --skip-python                            # refresh notebook_env.sh only
```
