"""Tests for scripts/clean_zerod_generated.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "clean_zerod_generated.py"
STANDARD_0D_SUBDIR = "standard-0d"


def test_dry_run_includes_derived_artifacts_by_default(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    set_name = "VMR_pulmo"
    run_config = "quadratic_resistor_gen_loss"

    (data_root / "zeroD" / set_name / STANDARD_0D_SUBDIR).mkdir(parents=True)
    (data_root / "zeroD" / set_name / run_config).mkdir(parents=True)
    (data_root / "jax_arrays" / set_name / run_config).mkdir(parents=True)
    (results_root / "models" / set_name / run_config).mkdir(parents=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--set_name",
            set_name,
            "--run_config",
            run_config,
            "--data_root",
            str(data_root),
            "--results_root",
            str(results_root),
            "--dry_run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "[zeroD]" in proc.stdout
    assert "[jax_arrays]" in proc.stdout
    assert "[models]" in proc.stdout
    assert (data_root / "zeroD" / set_name / run_config).exists()
    assert (data_root / "jax_arrays" / set_name / run_config).exists()


def test_no_also_derived_skips_jax_arrays(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    set_name = "VMR_pulmo"
    run_config = "gen_loss"

    (data_root / "zeroD" / set_name / STANDARD_0D_SUBDIR).mkdir(parents=True)
    (data_root / "zeroD" / set_name / run_config).mkdir(parents=True)
    (data_root / "jax_arrays" / set_name / run_config).mkdir(parents=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--set_name",
            set_name,
            "--run_config",
            run_config,
            "--data_root",
            str(data_root),
            "--results_root",
            str(results_root),
            "--no-also_derived",
            "--dry_run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "[zeroD]" in proc.stdout
    assert "[jax_arrays]" not in proc.stdout
