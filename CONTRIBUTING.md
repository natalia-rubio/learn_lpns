# Contributing

Thanks for improving **learn_lpns**. This repo is a research pipeline with library-style packaging: install the package, use the console CLIs, and import the small public API documented below.

## Setup

```bash
git clone <repo-url>
cd learn_lpns
pip install -e ".[dev]"
pre-commit install
```

Always use an editable install before running CLIs or `python -m learn_lpns...` modules. Do not rely on `sys.path` hacks.

## Checks before opening a PR

```bash
pre-commit run --all-files
pytest -m "not integration" -v
```

CI runs the same lint/format/unit-test suite on Python 3.10 and 3.12. Unit tests do **not** require building svZeroDPlus.

Optional local smoke tests:

```bash
learn-lpns-cv --help
learn-lpns-train --help
```

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:

| Prefix | Use for |
|--------|---------|
| `feat` | New user-facing behavior |
| `fix` | Bug fixes |
| `docs` | Documentation only |
| `ci` | CI / workflow changes |
| `refactor` | Code changes without behavior change |
| `test` | Tests only |
| `chore` | Tooling, deps, formatting |

Example: `refactor: remove sys.path hacks from entry points`

## PR checklist

- [ ] `pre-commit run --all-files` passes
- [ ] `pytest -m "not integration"` passes
- [ ] No change to default physics/training hyperparameters unless intentional
- [ ] New CLI flags or config keys documented in `docs/usage.md` or `config/defaults.yaml` comments
- [ ] Integration tests that need svZeroDPlus marked with `@pytest.mark.integration` (not run in default CI)

## Supported imports

Prefer the top-level API:

```python
from learn_lpns import get_pipeline_config, resolve_run_config_suffix, modality_table_header
```

Deeper imports (`learn_lpns.zerod_calibration.*`, etc.) are fine for in-repo scripts but may change without a semver guarantee. See `docs/architecture.md` for details.
