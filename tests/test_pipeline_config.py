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


def test_defaults_include_nondim_rsl_config():
    cfg = load_pipeline_config()
    assert cfg.physics.reference_reynolds == pytest.approx(5000.0)
    assert cfg.training.nondimensionalize_rsl is False


def test_training_quiet_epochs_defaults_true():
    cfg = load_pipeline_config()
    assert cfg.training.quiet_epochs is True


def test_rri_coefficient_effective_restore_and_threshold():
    from learn_lpns.config.models import RriCoefficientConfig

    bare = RriCoefficientConfig(
        name="R",
        label="R",
        target_output_column=0,
        lr_init=0.01,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
    )
    assert bare.effective_restore_best_weights(True) is True
    assert bare.effective_restore_best_weights(False) is False
    assert bare.effective_early_stop_loss_threshold(1e-7) == pytest.approx(1e-7)

    custom = RriCoefficientConfig(
        name="S",
        label="S",
        target_output_column=1,
        lr_init=0.1,
        junction_num_layers=1,
        junction_layer_width=3,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
        restore_best_weights=False,
        early_stop_loss_threshold=1e-5,
    )
    assert custom.effective_restore_best_weights(True) is False
    assert custom.effective_early_stop_loss_threshold(1e-7) == pytest.approx(1e-5)


def test_rri_coefficient_per_modality_epochs():
    from learn_lpns.config.models import RriCoefficientConfig

    explicit = RriCoefficientConfig(
        name="R",
        label="R",
        target_output_column=0,
        lr_init=0.01,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
        num_epochs=4000,
        vessel_num_epochs=2000,
    )
    assert explicit.training_epochs(vessel=False, default=500) == 4000
    assert explicit.training_epochs(vessel=True, default=500) == 2000

    bare = RriCoefficientConfig(
        name="R",
        label="R",
        target_output_column=0,
        lr_init=0.01,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
    )
    assert bare.training_epochs(vessel=False, default=500) == 500
    assert bare.training_epochs(vessel=True, default=500) == 500


def test_rri_coefficient_effective_generation_weighted_loss_decay_base():
    from learn_lpns.config.models import RriCoefficientConfig

    bare = RriCoefficientConfig(
        name="R",
        label="R",
        target_output_column=0,
        lr_init=0.01,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
    )
    assert bare.effective_generation_weighted_loss_decay_base(3.0) == pytest.approx(3.0)

    custom = RriCoefficientConfig(
        name="S",
        label="S",
        target_output_column=1,
        lr_init=0.1,
        junction_num_layers=1,
        junction_layer_width=3,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
        generation_weighted_loss_decay_base=2.5,
    )
    assert custom.effective_generation_weighted_loss_decay_base(3.0) == pytest.approx(2.5)


def test_rri_coefficient_effective_activation():
    from learn_lpns.config.models import RriCoefficientConfig

    bare = RriCoefficientConfig(
        name="R",
        label="R",
        target_output_column=0,
        lr_init=0.01,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
    )
    assert bare.effective_activation(default="tanh") == "tanh"

    legacy = RriCoefficientConfig(
        name="S",
        label="S",
        target_output_column=1,
        lr_init=0.1,
        junction_num_layers=1,
        junction_layer_width=3,
        junction_asymmetric_overestimate_weight=1.0,
        vessel_asymmetric_overestimate_weight=1.0,
        leaky_relu=True,
    )
    assert legacy.effective_activation(default="relu") == "leaky_relu"


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


def test_rri_coefficients_merged_by_name_from_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.dump(
            {
                "training": {
                    "rri_coefficients": [
                        {
                            "name": "R",
                            "label": "Linear Resistor (R)",
                            "target_output_column": 0,
                            "lr_init": 0.01,
                            "junction_num_layers": 4,
                            "junction_layer_width": 10,
                            "junction_asymmetric_overestimate_weight": 2000,
                            "vessel_asymmetric_overestimate_weight": 10,
                        },
                        {
                            "name": "S",
                            "label": "Stenosis Resistor (S)",
                            "target_output_column": 1,
                            "lr_init": 0.1,
                            "junction_num_layers": 1,
                            "junction_layer_width": 10,
                            "junction_asymmetric_overestimate_weight": 2000,
                            "vessel_asymmetric_overestimate_weight": 10,
                        },
                    ]
                }
            }
        )
    )
    cfg = load_pipeline_config(
        base,
        overrides={"training": {"rri_coefficients": [{"name": "S", "junction_layer_width": 20}]}},
    )
    r_spec, s_spec = cfg.training.rri_coefficients
    assert r_spec.name == "R"
    assert r_spec.junction_layer_width == 10  # untouched
    assert s_spec.name == "S"
    assert s_spec.junction_layer_width == 20  # overridden
    assert s_spec.lr_init == pytest.approx(0.1)  # other S fields preserved


def test_set_overrides_in_main_config_merges_single_coefficient(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base_data = yaml.safe_load(resolve_config_path().read_text())
    base_data["set_overrides"] = {
        "VMR_widetest": {
            "training": {"rri_coefficients": [{"name": "S", "junction_layer_width": 44}]}
        }
    }
    base.write_text(yaml.dump(base_data))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    default_cfg = load_pipeline_config(base)
    set_cfg = load_pipeline_config(set_name="VMR_widetest")

    default_by_name = {c.name: c for c in default_cfg.training.rri_coefficients}
    set_by_name = {c.name: c for c in set_cfg.training.rri_coefficients}
    assert set_by_name["S"].junction_layer_width == 44
    assert set_by_name["R"].junction_layer_width == default_by_name["R"].junction_layer_width
    assert set_by_name["L"].junction_layer_width == default_by_name["L"].junction_layer_width
    # set_overrides is loader-only and must not leak into the validated config tree.
    assert not hasattr(set_cfg, "set_overrides")


def test_set_overrides_ignored_for_other_set(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base_data = yaml.safe_load(resolve_config_path().read_text())
    base_data["set_overrides"] = {
        "VMR_widetest": {
            "training": {"rri_coefficients": [{"name": "S", "junction_layer_width": 44}]}
        }
    }
    base.write_text(yaml.dump(base_data))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    other_cfg = load_pipeline_config(set_name="VMR_other")
    default_cfg = load_pipeline_config(base)
    other_s = {c.name: c for c in other_cfg.training.rri_coefficients}["S"]
    default_s = {c.name: c for c in default_cfg.training.rri_coefficients}["S"]
    assert other_s.junction_layer_width == default_s.junction_layer_width


def test_inline_rri_set_overrides_under_coefficient(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base_data = yaml.safe_load(resolve_config_path().read_text())
    training = base_data["training"]
    s_entry = next(c for c in training["rri_coefficients"] if c["name"] == "S")
    s_entry["VMR_widetest"] = {"junction_layer_width": 44}
    base.write_text(yaml.dump(base_data))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    default_cfg = load_pipeline_config(base)
    set_cfg = load_pipeline_config(set_name="VMR_widetest")

    default_by_name = {c.name: c for c in default_cfg.training.rri_coefficients}
    set_by_name = {c.name: c for c in set_cfg.training.rri_coefficients}
    assert default_by_name["S"].junction_layer_width == 10
    assert set_by_name["S"].junction_layer_width == 44
    assert set_by_name["R"].junction_layer_width == default_by_name["R"].junction_layer_width
    assert set_by_name["L"].junction_layer_width == default_by_name["L"].junction_layer_width


def test_inline_stenosis_set_overrides_under_limit(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base_data = yaml.safe_load(resolve_config_path().read_text())
    limit = base_data["data_processing"]["stenosis_generation_limit"]
    limit["junction_max_generation"] = 1.0
    limit["VMR_widetest"] = {"junction_max_generation": 0}
    base.write_text(yaml.dump(base_data))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    default_cfg = load_pipeline_config(base)
    set_cfg = load_pipeline_config(set_name="VMR_widetest")

    default_limit = default_cfg.data_processing.stenosis_generation_limit
    set_limit = set_cfg.data_processing.stenosis_generation_limit
    assert default_limit.junction_max_generation == pytest.approx(1.0)
    assert set_limit.junction_max_generation == pytest.approx(0.0)
    assert set_limit.vessel_max_generation == default_limit.vessel_max_generation
    assert set_limit.enabled == default_limit.enabled


def test_inline_stenosis_set_overrides_ignored_for_other_set(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base_data = yaml.safe_load(resolve_config_path().read_text())
    base_data["data_processing"]["stenosis_generation_limit"]["VMR_widetest"] = {
        "junction_max_generation": 0
    }
    base.write_text(yaml.dump(base_data))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    other_cfg = load_pipeline_config(set_name="VMR_other")
    default_cfg = load_pipeline_config(base)
    assert other_cfg.data_processing.stenosis_generation_limit.junction_max_generation == pytest.approx(
        default_cfg.data_processing.stenosis_generation_limit.junction_max_generation
    )


def test_inline_stenosis_set_override_invalid_type_raises(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base_data = yaml.safe_load(resolve_config_path().read_text())
    base_data["data_processing"]["stenosis_generation_limit"]["VMR_widetest"] = 0
    base.write_text(yaml.dump(base_data))
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="stenosis_generation_limit.VMR_widetest"):
        load_pipeline_config(base)


def test_rri_coefficients_set_layer_merges_single_coefficient(tmp_path, monkeypatch):
    base = tmp_path / "defaults.yaml"
    base.write_text(resolve_config_path().read_text())
    sets_dir = tmp_path / "config" / "sets"
    sets_dir.mkdir(parents=True)
    (sets_dir / "VMR_widetest.yaml").write_text(
        yaml.dump({"training": {"rri_coefficients": [{"name": "S", "junction_layer_width": 33}]}})
    )
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(base))
    monkeypatch.setattr("learn_lpns.config.load.repo_root", lambda: tmp_path)

    default_cfg = load_pipeline_config(base)
    set_cfg = load_pipeline_config(set_name="VMR_widetest")

    default_by_name = {c.name: c for c in default_cfg.training.rri_coefficients}
    set_by_name = {c.name: c for c in set_cfg.training.rri_coefficients}
    assert set_by_name["S"].junction_layer_width == 33
    assert set_by_name["R"].junction_layer_width == default_by_name["R"].junction_layer_width
    assert set_by_name["L"].junction_layer_width == default_by_name["L"].junction_layer_width


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


def test_training_run_configs_overlay_by_run_config(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.dump(
            {
                "training": {
                    "include_speed_change": False,
                    "early_stop_loss_threshold": None,
                    "optimizer": {"init": 0.02, "transition_steps": 1000, "decay_rate": 0.8},
                    "vessel": {"num_layers": 2, "layer_width": 10},
                    "rri_coefficients": [
                        {
                            "name": "L",
                            "label": "Inductor (L)",
                            "target_output_column": 2,
                            "lr_init": 0.01,
                            "junction_num_layers": 4,
                            "junction_layer_width": 20,
                            "junction_asymmetric_overestimate_weight": 1,
                            "vessel_asymmetric_overestimate_weight": 1,
                        }
                    ],
                },
                "training_run_configs": {
                    "gen_loss": {},
                    "quadratic_resistor_gen_loss": {
                        "include_speed_change": True,
                        "early_stop_loss_threshold": 1.0e-7,
                        "optimizer": {"decay_rate": 0.99},
                        "vessel": {"num_layers": 1},
                        "rri_coefficients": [{"name": "L", "junction_layer_width": 10}],
                    },
                },
                "run_config_overrides": {
                    "quadratic_resistor_gen_loss": {
                        "data_processing": {"flow_split_method": "peak_inlet_flow"}
                    }
                },
            }
        )
    )
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(custom))

    base = load_pipeline_config()
    assert base.training.optimizer.decay_rate == pytest.approx(0.8)
    assert base.training.include_speed_change is False
    assert base.training.early_stop_loss_threshold is None
    assert base.training.vessel.num_layers == 2
    assert base.training.rri_coefficients[0].junction_layer_width == 20

    gen = load_pipeline_config(run_config="gen_loss")
    assert gen.training.optimizer.decay_rate == pytest.approx(0.8)
    assert gen.training.include_speed_change is False

    qr = load_pipeline_config(run_config="quadratic_resistor_gen_loss")
    assert qr.training.optimizer.decay_rate == pytest.approx(0.99)
    assert qr.training.include_speed_change is True
    assert qr.training.early_stop_loss_threshold == pytest.approx(1.0e-7)
    assert qr.training.vessel.num_layers == 1
    assert qr.training.rri_coefficients[0].junction_layer_width == 10
    assert qr.data_processing.flow_split_method == "peak_inlet_flow"

    # Unordered tokens canonicalize to the same overlay key.
    qr_alias = load_pipeline_config(run_config="gen_loss_quadratic_resistor")
    assert qr_alias.training.optimizer.decay_rate == pytest.approx(0.99)


def test_get_pipeline_config_cached_per_run_config(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.dump(
            {
                "training": {"optimizer": {"init": 0.02, "transition_steps": 1000, "decay_rate": 0.8}},
                "training_run_configs": {
                    "quadratic_resistor_gen_loss": {"optimizer": {"decay_rate": 0.99}},
                },
            }
        )
    )
    monkeypatch.setenv("LEARN_LPNS_CONFIG", str(custom))
    cfg_gen = get_pipeline_config(run_config="gen_loss")
    cfg_qr = get_pipeline_config(run_config="quadratic_resistor_gen_loss")
    assert cfg_gen is not cfg_qr
    assert cfg_gen.training.optimizer.decay_rate == pytest.approx(0.8)
    assert cfg_qr.training.optimizer.decay_rate == pytest.approx(0.99)


def test_defaults_gen_loss_vs_quadratic_resistor_profiles():
    gen = load_pipeline_config(run_config="gen_loss")
    qr = load_pipeline_config(run_config="quadratic_resistor_gen_loss")

    # Recovered RI/gen_loss defaults remain isolated from the QR profile.
    assert gen.training.optimizer.decay_rate == pytest.approx(0.8)
    assert gen.training.include_speed_change is False
    assert gen.training.clip_predictions is False
    assert gen.training.clip_input_features is False
    assert gen.training.resolved_activation() == "relu"
    assert gen.training.generation_weighted_loss_decay_base == pytest.approx(2.0)
    assert gen.training.vessel.num_layers == 2
    assert gen.data_processing.flow_split_method == "mean_over_time"

    # QR reproduces the complete operational package from 197c6c while sharing
    # the corrected BC pipeline.
    assert qr.training.optimizer.decay_rate == pytest.approx(0.99)
    assert qr.training.include_speed_change is True
    assert qr.training.clip_predictions is True
    assert qr.training.clip_input_features is True
    assert qr.training.resolved_activation() == "leaky_relu"
    assert qr.training.generation_weighted_loss_decay_base == pytest.approx(2.0)
    assert qr.training.vessel.num_layers == 1
    assert qr.training.vessel.layer_width == 10
    assert qr.data_processing.flow_split_method == "mean_over_time"

    r_gen = next(c for c in gen.training.rri_coefficients if c.name == "R")
    r_qr = next(c for c in qr.training.rri_coefficients if c.name == "R")
    s_qr = next(c for c in qr.training.rri_coefficients if c.name == "S")
    l_gen = next(c for c in gen.training.rri_coefficients if c.name == "L")
    l_qr = next(c for c in qr.training.rri_coefficients if c.name == "L")
    # QR currently inherits base R/S/L architecture (QR coefficient overlay commented out).
    assert r_gen.junction_num_layers == 2
    assert r_qr.junction_num_layers == 2
    assert s_qr.junction_num_layers == 1
    assert s_qr.generation_weighted_loss_decay_base is None
    assert l_gen.junction_num_layers == 4
    assert l_gen.junction_layer_width == 20
    assert l_qr.junction_num_layers == 4
    assert l_qr.junction_layer_width == 20

    assert qr.data_processing.stenosis_generation_limit.enabled is True
    assert qr.data_processing.stenosis_generation_limit.junction_max_generation == pytest.approx(0.0)
    assert qr.data_processing.stenosis_generation_limit.vessel_max_generation == pytest.approx(0.0)
    assert qr.calibration.decoupled_nonneg_r is True
