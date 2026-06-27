"""Smoke tests that editable install + JAX setup works (mirrors CI and README quick start).

CI runs:
  pip install "jax[cpu]"
  pip install -e ".[dev]"
  pytest -m "not integration" -v

These tests verify imports, config, console entry points, and setup script presence.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from importlib.metadata import entry_points, version

import pytest

from learn_lpns.tools.paths import repo_root

CLI_SCRIPT_NAMES = (
    "learn-lpns-cv",
    "learn-lpns-cv-all-sets",
    "learn-lpns-batch-zerod",
    "learn-lpns-data-processing",
    "learn-lpns-train",
)

README_SETUP_PATHS = (
    "scripts/setup_cross_validation.sh",
    "scripts/setup_git_lfs.sh",
    "scripts/setup_notebook.sh",
    "Makefile",
)


def test_learn_lpns_package_version():
    assert version("learn-lpns") == "0.1.0"


@pytest.mark.parametrize(
    "module_name",
    [
        "numpy",
        "jax",
        "jax.numpy",
        "optax",
        "pydantic",
        "yaml",
        "pandas",
        "scipy",
        "matplotlib",
        "vtk",
        "dill",
    ],
)
def test_core_dependency_imports(module_name: str):
    importlib.import_module(module_name)


def test_public_api_import():
    from learn_lpns import (
        MODALITY_DISPLAY,
        get_pipeline_config,
        modality_table_header,
        resolve_run_config_suffix,
    )
    from learn_lpns import (
        repo_root as package_repo_root,
    )

    cfg = get_pipeline_config()
    assert cfg.training.num_epochs > 0
    assert modality_table_header("geometric") == "Standard"
    assert MODALITY_DISPLAY["Vessel_NN"] == "Learned Vessels"
    assert package_repo_root().is_dir()
    assert resolve_run_config_suffix("gen_loss")


def test_console_scripts_registered():
    registered = {ep.name for ep in entry_points(group="console_scripts")}
    for name in CLI_SCRIPT_NAMES:
        assert name in registered, f"Missing console script {name!r} — run: pip install -e '.[dev]'"


@pytest.mark.parametrize("script_name", CLI_SCRIPT_NAMES)
def test_console_script_help(script_name: str):
    exe = shutil.which(script_name)
    if exe is None:
        pytest.fail(
            f"{script_name} not on PATH. Install editable package: pip install -e '.[dev]' "
            "(after pip install 'jax[cpu]')"
        )
    result = subprocess.run(
        [exe, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "usage:" in (result.stdout + result.stderr).lower()


def test_readme_setup_artifacts_exist():
    root = repo_root()
    for rel in README_SETUP_PATHS:
        path = root / rel
        assert path.is_file(), f"Missing README setup artifact: {rel}"


def test_standard_zero_d_seed_json_present():
    """Bundled JSON seeds ship in git (no LFS); required for notebook/CV demos."""
    root = repo_root()
    seed_dir = root / "data" / "zeroD" / "VMR_aortas" / "standard-0d"
    assert seed_dir.is_dir(), f"Missing {seed_dir.relative_to(root)}"
    for geo in ("0129_0000", "0154_0001", "0174_0000", "0175_0000", "0176_0000"):
        assert (seed_dir / f"{geo}.json").is_file()
