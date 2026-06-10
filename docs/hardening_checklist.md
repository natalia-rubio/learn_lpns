# Hardening checklist (tesseract-jax style)

A ~1–2 day plan to raise **learn_lpns** toward library-grade engineering practice ([tesseract-jax](https://github.com/pasteurlabs/tesseract-jax) as reference) **without changing science behavior** (same outputs, same CLI flags, same default hyperparameters).

Use this as a sprint checklist. Each item has an effort estimate and clear “done when” criteria.

---

## Already in good shape

- [x] `pyproject.toml` packaging + console entry points
- [x] Editable install (`pip install -e ".[dev]"`)
- [x] Unit tests (55+) with no C++ solver required
- [x] CI: `ruff format --check`, `ruff check`, unit tests with coverage on Python 3.10 / 3.12
- [x] pre-commit (ruff lint/format, yaml/toml hygiene)
- [x] Expanded Ruff rules (`UP`, `B`, `RUF`; typographic unicode ignored for plot labels)
- [x] pytest integration marker (`pytest -m "not integration"` in CI)
- [x] Centralized YAML config (`config/defaults.yaml` + Pydantic)
- [x] Split user docs (`docs/usage.md`, `architecture.md`, `data_and_results.md`)

---

## Day 1 — Tooling & CI (~4–6 hours)

Goal: catch style/regression issues automatically; match tesseract’s “lint + format on every commit” habit.

### 1.1 Add `ruff format` (~30 min)

**Do:**
- Add `[tool.ruff.format]` to `pyproject.toml` (line-length 120 to match lint).
- Run `ruff format learn_lpns tests` once (large diff, no logic changes).
- CI: add step `ruff format --check learn_lpns tests` before `ruff check`.

**Done when:** CI fails if code is not formatted.

**Files:** `pyproject.toml`, `.github/workflows/ci.yml`, all Python under `learn_lpns/` + `tests/`.

---

### 1.2 Add pre-commit (~45 min)

**Do:**
- Add `.pre-commit-config.yaml` with:
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`
  - `ruff` (lint + fix) and `ruff-format`
- Document in README Development: `pip install pre-commit && pre-commit install`
- Optional dev extra: `pre-commit` in `[project.optional-dependencies].dev`

**Done when:** `pre-commit run --all-files` passes locally.

**Reference:** [tesseract-jax `.pre-commit-config.yaml`](https://github.com/pasteurlabs/tesseract-jax/blob/main/.pre-commit-config.yaml)

---

### 1.3 Expand Ruff rules incrementally (~1–2 hours)

**Do (phase A only — avoid docstring churn on legacy files):**
- Add to `extend-select`: `UP` (pyupgrade), `B` (bugbear), `I` (already have), `RUF`
- Keep `D` (docstrings) **off** for now, or enable only on `learn_lpns/config/**` and `tests/**`
- Fix any new violations in `config/`, `tests/`, and small modules first
- Leave per-file ignores for large legacy scripts until Day 2

**Done when:** `ruff check learn_lpns tests` passes with expanded rules; no changes to numerical outputs.

**Do not:** Enable `ANN` (full typing) repo-wide in Day 1 — too much churn.

---

### 1.4 pytest-cov in CI (~30 min)

**Do:**
- Add `pytest-cov` to dev dependencies
- CI: `pytest --cov=learn_lpns --cov-report=term-missing --cov-fail-under=25` (start low; raise later)
- Add `[tool.coverage.run]` omit patterns for `if __name__ == "__main__"` blocks if needed

**Done when:** CI prints coverage summary; threshold is met.

---

### 1.5 Mark integration tests (~30 min)

**Do:**
- Add to `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  markers = ["integration: requires svZeroDPlus binaries"]
  ```
- Any future solver tests use `@pytest.mark.integration`
- CI runs `pytest -m "not integration"` (default unit suite only)

**Done when:** Marker registered; CI command explicit about unit-only scope.

---

## Day 2 — Boundaries & public API (~4–6 hours)

Goal: look like a library with CLIs on top, not a folder of scripts (tesseract’s thin `__all__` pattern).

- [x] Remove `sys.path` hacks; shared `learn_lpns.tools.paths.repo_root()`
- [x] Public API in `learn_lpns/__init__.py` + typed `modality_paths.py`
- [x] `CONTRIBUTING.md` and architecture docs for supported imports

### 2.1 Remove `sys.path` hacks from entry-point modules (~2–3 hours)

**Current offenders (grep `REPO_ROOT` / `sys.path.insert`):**
- `learn_lpns/zerod_calibration/generate_zerod_inputs.py`
- `learn_lpns/zerod_calibration/run_cross_validation.py`
- `learn_lpns/zerod_calibration/run_cv_all_configs.py`
- `learn_lpns/zerod_calibration/batch_generate_zerod_inputs_vmr.py`
- `learn_lpns/neural_network/launch_training.py`
- `learn_lpns/data_processing/run_data_processing.py`
- `learn_lpns/data_processing/generate_split_indices.py`
- (+ a few visualization / report scripts)

**Do:**
- Delete `REPO_ROOT` + `sys.path.insert` blocks from modules that are installed via entry points
- Keep `if __name__ == "__main__": main()` only as thin shim (or remove if entry point exists)
- Remove corresponding `E402` per-file ignores from `pyproject.toml`
- README note: always `pip install -e ".[dev]"` before running CLIs

**Done when:** `grep -r "sys.path.insert" learn_lpns/` returns empty (or only HPC cluster scripts if any remain outside package).

**Science check:** `pytest -v` and one manual `learn-lpns-cv --help` smoke test.

---

### 2.2 Define a public API surface (~1 hour)

**Do:**
- Add `learn_lpns/__init__.py` exports, e.g.:
  ```python
  __all__ = [
      "get_pipeline_config",
      "load_pipeline_config",
      # run-config helpers used by downstream tools
  ]
  ```
- Or add `learn_lpns/api.py` re-exporting:
  - `learn_lpns.config` (config loader)
  - `learn_lpns.zerod_calibration.run_config_canonical` (token parsing)
  - `learn_lpns.zerod_calibration.modality_paths` (CV CSV column names)
- Document “Supported imports” in README or `docs/architecture.md`

**Done when:** A reviewer can answer “what am I allowed to import?” in one paragraph.

**Reference:** [tesseract_jax `__init__.py`](https://github.com/pasteurlabs/tesseract-jax/blob/main/tesseract_jax/__init__.py) — three symbols in `__all__`.

---

### 2.3 Type the config + run-config boundaries (~1–2 hours)

**Do (no full ANN rollout):**
- Config models already typed via Pydantic ✅
- Add return types to public functions in:
  - `run_config_canonical.py` (already mostly typed)
  - `modality_paths.py`
  - `config/load.py`
- Optional: small `@dataclass` for `TrainingRunContext` instead of loose `network_params` dict in **new** code only

**Done when:** `learn_lpns/config/` and `run_config_canonical.py` have complete function signatures; mypy not required yet.

---

### 2.4 Add `CONTRIBUTING.md` (~30 min)

**Do:**
- Short file covering:
  - `pip install -e ".[dev]"` + `pre-commit install`
  - `ruff check` + `ruff format` + `pytest`
  - Conventional commit types (`feat`, `fix`, `docs`, `ci`, `refactor`, `test`)
  - PR checklist: tests pass, no solver required for unit tests
- Link from README Development section

**Done when:** New contributor (or interviewer) can onboard from CONTRIBUTING alone.

**Reference:** [tesseract-jax CONTRIBUTING.md](https://github.com/pasteurlabs/tesseract-jax/blob/main/CONTRIBUTING.md)

---

## Verification gate (end of Day 2)

Run locally before pushing:

```bash
pip install -e ".[dev]"
pre-commit run --all-files
pytest -v --cov=learn_lpns --cov-report=term-missing
learn-lpns-cv --help
learn-lpns-train --help
```

**Science regression (optional, ~30–60 min CPU):** one geometry through existing outputs, compare checksums of a summary CSV — only if you changed entry-point import paths.

---

## Explicitly out of scope (this sprint)

These are tesseract-jax patterns worth **later**, not this 1–2 day pass:

| Item | Why defer |
|------|-----------|
| Sphinx / ReadTheDocs | Hand-written `docs/` is enough for now |
| Full `ANN` typing repo-wide | High churn on 1k+ line modules |
| Google docstrings (`D` rules) everywhere | Legacy scripts; start with `config/` only |
| versioneer / dynamic versioning | Static `0.1.0` is fine for research repo |
| Docker-based integration tests | Heavy; mark `@integration` first |
| Splitting `generate_zerod_inputs.py` | Science risk; separate refactor sprint |
| CLA / Discourse / issue templates | Optional for solo / interview repo |

---

## Suggested commit sequence

Keep PRs reviewable (tesseract-style small commits):

1. `ci: add ruff format check`
2. `chore: add pre-commit hooks`
3. `ci: add pytest-cov with baseline threshold`
4. `refactor: remove sys.path hacks from entry points`
5. `docs: add CONTRIBUTING and public API note`
6. `chore: expand ruff rules (UP, B, RUF)`

---

## Success criteria (what “done” looks like)

After 1–2 days, a reviewer should see:

- Formatting enforced locally (pre-commit) and in CI
- Coverage reported in CI with a floor threshold
- No `sys.path.insert` in installable package entry points
- Documented public import surface
- CONTRIBUTING with conventional commits
- **Same** default physics, training, and CV behavior as before

That closes most of the gap with tesseract-jax’s **process** quality while keeping learn_lpns honestly a **pipeline** repo.
