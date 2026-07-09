"""Tests for CV trial parallelization config and subprocess env helpers."""

import os
from pathlib import Path

import pytest
import yaml

from learn_lpns.config import load_pipeline_config
from learn_lpns.config.load import _cached_pipeline_config
from learn_lpns.zerod_calibration.run_cross_validation import _cv_subprocess_env


@pytest.fixture(autouse=True)
def clear_config_cache():
    _cached_pipeline_config.cache_clear()
    yield
    _cached_pipeline_config.cache_clear()


def test_cv_max_parallel_trials_defaults_to_one():
    cfg = load_pipeline_config()
    assert cfg.split.cv_max_parallel_trials == 1
    assert cfg.split.cv_worker_cpu_threads is None


def test_cv_parallel_config_yaml_override(tmp_path: Path):
    defaults_path = Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"
    with open(defaults_path) as f:
        data = yaml.safe_load(f)
    data["split"]["cv_max_parallel_trials"] = 4
    data["split"]["cv_worker_cpu_threads"] = 2
    override = tmp_path / "overrides.yaml"
    override.write_text(yaml.dump(data))
    cfg = load_pipeline_config(override)
    assert cfg.split.cv_max_parallel_trials == 4
    assert cfg.split.cv_worker_cpu_threads == 2


def test_cv_subprocess_env_none_when_threads_unset():
    assert _cv_subprocess_env(None) is None


def test_cv_subprocess_env_sets_omp_and_xla_flags():
    env = _cv_subprocess_env(2)
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "2"
    assert "intra_op_parallelism_threads=2" in env["XLA_FLAGS"]
    assert env["XLA_FLAGS"].startswith("--xla_cpu_multi_thread_eigen=false")


def test_cv_subprocess_env_copies_parent_environ(monkeypatch):
    monkeypatch.setenv("CUSTOM_TEST_VAR", "keep_me")
    env = _cv_subprocess_env(1)
    assert env is not None
    assert os.environ.get("CUSTOM_TEST_VAR") == "keep_me"
    assert env["CUSTOM_TEST_VAR"] == "keep_me"


def test_cv_max_parallel_trials_rejects_zero():
    from learn_lpns.config.models import SplitConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SplitConfig(cv_max_parallel_trials=0)
