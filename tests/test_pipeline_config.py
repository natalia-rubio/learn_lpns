"""Tests for centralized YAML pipeline configuration."""

import pytest
import yaml
from pydantic import ValidationError

from learn_lpns.config import (
    apply_solver_parameters,
    build_calibration_parameters,
    get_pipeline_config,
    load_pipeline_config,
    resolve_config_path,
)
from learn_lpns.config.load import _cached_pipeline_config


@pytest.fixture(autouse=True)
def clear_config_cache():
    _cached_pipeline_config.cache_clear()
    yield
    _cached_pipeline_config.cache_clear()


def test_resolve_config_path_finds_repo_defaults():
    path = resolve_config_path()
    assert path.name == "defaults.yaml"
    assert path.is_file()


def test_load_pipeline_config_defaults():
    cfg = load_pipeline_config()
    assert cfg.physics.rho == pytest.approx(1.06)
    assert cfg.physics.mu == pytest.approx(0.04)
    assert cfg.solver.absolute_tolerance == pytest.approx(1e-5)
    assert cfg.solver.maximum_nonlinear_iterations == 50
    assert cfg.solver.number_of_cardiac_cycles == 1
    assert cfg.solver.steady_initial is False
    assert cfg.calibration.tolerance_gradient == pytest.approx(1e-4)
    assert cfg.calibration.default_l2_r == pytest.approx(1e5)
    assert cfg.split.percent_train == pytest.approx(0.9)
    assert cfg.split.data_processing_percent_train == pytest.approx(0.8)
    assert cfg.split.cv_num_trials == 5
    assert cfg.split.cv_trial_seed_stride == 1000
    # assert cfg.training.num_epochs == 500
    assert len(cfg.training.rri_coefficients) == 3
    assert cfg.training.rri_coefficients[0].name == "R"
    assert cfg.training.optimizer.transition_steps == 1000


def test_training_batch_size_for_n_train():
    cfg = load_pipeline_config()
    assert cfg.training.batch_size_for_n_train(25) == 3
    assert cfg.training.batch_size_for_n_train(1) == 1


def test_build_calibration_parameters_penalty_on():
    cfg = load_pipeline_config()
    params = build_calibration_parameters(
        cfg.calibration,
        quadratic_resistor=True,
        penalty_on=True,
        cohorts=cfg.cohorts,
    )
    assert params["tolerance_gradient"] == pytest.approx(1e-4)
    assert params["maximum_iterations"] == 20
    assert params["calibrate_stenosis_coefficient"] is True
    assert params["L2_penalty_R_poiseuille"] == pytest.approx(1e5)
    assert params["L2_penalty_stenosis_coefficient"] == pytest.approx(1e10)


def test_build_calibration_parameters_penalty_off():
    cfg = load_pipeline_config()
    params = build_calibration_parameters(
        cfg.calibration,
        quadratic_resistor=True,
        penalty_on=False,
    )
    assert params["L2_penalty_R_poiseuille"] == 0.0
    assert params["L2_penalty_stenosis_coefficient"] == 0.0


def test_load_pipeline_config_rejects_invalid_physics(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump({"physics": {"rho": -1.0, "mu": 0.04}}))
    with pytest.raises(ValidationError):
        load_pipeline_config(bad)


def test_load_pipeline_config_rejects_invalid_split(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump({"split": {"percent_train": 1.5}}))
    with pytest.raises(ValidationError):
        load_pipeline_config(bad)


def test_load_pipeline_config_deep_merge_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.dump({"physics": {"rho": 1.06, "mu": 0.04}, "solver": {"absolute_tolerance": 1e-5}}))
    cfg = load_pipeline_config(base, overrides={"physics": {"mu": 0.05}})
    assert cfg.physics.mu == pytest.approx(0.05)
    assert cfg.physics.rho == pytest.approx(1.06)


def test_load_pipeline_config_set_name_layer(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base.write_text(
        yaml.dump(
            {
                "calibration": {"default_l2_r": 1.0e5, "default_l2_stenosis": 1.0e10},
                "cohorts": {
                    "sets": {
                        "custom_cohort": {
                            "display_label": "Custom",
                            "calibration": {"l2_r": 42, "l2_stenosis": 99},
                        }
                    }
                },
            }
        )
    )
    sets_dir = tmp_path / "config" / "sets"
    sets_dir.mkdir(parents=True)
    (sets_dir / "custom_cohort.yaml").write_text(
        yaml.dump(
            {
                "cohorts": {
                    "sets": {
                        "custom_cohort": {
                            "display_label": "Custom merged",
                            "calibration": {"l2_r": 42, "l2_stenosis": 99},
                        }
                    }
                }
            }
        )
    )

    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr(
        "learn_lpns.config.load.repo_root",
        lambda: tmp_path,
    )
    cfg = load_pipeline_config(set_name="custom_cohort")
    assert cfg.cohorts.sets["custom_cohort"].display_label == "Custom merged"
    assert cfg.cohorts.l2_penalties_for_set("custom_cohort", cfg.calibration) == (pytest.approx(42), pytest.approx(99))


def test_apply_solver_parameters_writes_expected_fields():
    cfg = load_pipeline_config()
    sim = {"existing": True}
    apply_solver_parameters(sim, cfg.solver)
    assert sim["absolute_tolerance"] == pytest.approx(1e-5)
    assert sim["maximum_nonlinear_iterations"] == 50
    assert sim["number_of_cardiac_cycles"] == 1
    assert sim["steady_initial"] is False
    assert sim["existing"] is True


def test_get_pipeline_config_cached(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.dump({"physics": {"mu": 0.03}}))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(custom))
    cfg1 = get_pipeline_config()
    cfg2 = get_pipeline_config()
    assert cfg1 is cfg2
    assert cfg1.physics.mu == pytest.approx(0.03)


def test_get_pipeline_config_cached_per_set_name(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.dump({"calibration": {"default_l2_r": 1.0e5, "default_l2_stenosis": 1.0e10}}))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(custom))
    cfg_a = get_pipeline_config(set_name="cohort_a")
    cfg_b = get_pipeline_config(set_name="cohort_b")
    assert cfg_a is not cfg_b


def test_get_pipeline_config_reload(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.dump({"physics": {"mu": 0.03}}))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(custom))
    cfg1 = get_pipeline_config()
    custom.write_text(yaml.dump({"physics": {"mu": 0.035}}))
    cfg2 = get_pipeline_config(reload=True)
    assert cfg1.physics.mu == pytest.approx(0.03)
    assert cfg2.physics.mu == pytest.approx(0.035)


def test_learn_lpns_config_env_must_exist(tmp_path, monkeypatch):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(missing))
    with pytest.raises(FileNotFoundError):
        resolve_config_path()
