"""Example integration test — not run in default CI.

Run locally after building svZeroDPlus:

    pytest -m integration -v tests/test_integration_forward_sim_example.py

Default PR/CI checks use ``pytest -m "not integration"`` and skip this file.
"""

from __future__ import annotations

import subprocess

import pytest

from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.forward_simulation import run_forward_simulation
from learn_lpns.zerod_calibration.tools import svzerod_binaries

# Mark every test in this module as integration (svZeroDPlus + bundled data).
pytestmark = pytest.mark.integration

BUNDLED_CALIBRATED_JSON = (
    repo_root()
    / "data"
    / "zeroD"
    / "VMR_aorta_starter"
    / "gen_loss"
    / "0176_0000"
    / "bifurcations_EL_calibrated_output_BloodVesselJunction.json"
)


def _svzerodsolver_path() -> str | None:
    try:
        return svzerod_binaries.svzerod_binary("svzerodsolver")
    except RuntimeError:
        return None


@pytest.fixture(scope="module")
def svzerodsolver_path():
    path = _svzerodsolver_path()
    if path is None:
        pytest.skip("svzerodsolver not found; set SVZEROD_INSTALL_DIR or run scripts/setup_cross_validation.sh")
    return path


def test_svzerodsolver_executable_smoke(svzerodsolver_path):
    """Minimal check: binary exists and runs (no Python pipeline logic)."""
    result = subprocess.run(
        [svzerodsolver_path, "/__svzerodsolver_smoke_nonexistent__.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "[svzerodsolver]" in combined
    assert "cannot be opened" in combined


@pytest.fixture(scope="module")
def bundled_calibrated_input():
    if not BUNDLED_CALIBRATED_JSON.is_file():
        pytest.skip(f"Bundled sample JSON not found: {BUNDLED_CALIBRATED_JSON}")
    return BUNDLED_CALIBRATED_JSON


def test_forward_simulation_on_bundled_calibrated_json(bundled_calibrated_input, tmp_path_factory):
    """End-to-end: Python wrapper invokes svzerodsolver on repo sample data."""
    if _svzerodsolver_path() is None:
        pytest.skip("svzerodsolver not found")

    output_csv = tmp_path_factory.mktemp("forward_sim") / "results.csv"
    run_forward_simulation(str(bundled_calibrated_input), str(output_csv))

    assert output_csv.is_file()
    assert output_csv.stat().st_size > 0
