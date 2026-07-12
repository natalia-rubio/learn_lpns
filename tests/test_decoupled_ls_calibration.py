"""Unit tests for decoupled least-squares calibration."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.calibration import _postprocess_calibrated_config, run_calibration
from learn_lpns.zerod_calibration.decoupled_ls_calibration import (
    _fit_rlc,
    _is_connector_vessel,
    calibrate_decoupled_ls,
    infer_set_geo_from_zerod_path,
    rsl_fits_output_dir,
)

BUNDLED_CALIBRATED_JSON = (
    repo_root()
    / "data"
    / "zeroD"
    / "VMR_aorta_starter"
    / "gen_loss"
    / "0129_0000"
    / "bifurcations_EL_calibrated_output_BloodVesselJunction.json"
)


def _bundled_calibrated_json_available() -> bool:
    """True when the bundled sample exists and Git LFS has materialized it."""
    if not BUNDLED_CALIBRATED_JSON.is_file():
        return False
    with open(BUNDLED_CALIBRATED_JSON, encoding="utf-8") as f:
        first_line = f.readline()
    return not first_line.startswith("version https://git-lfs.github.com/spec/v1")


def _synthetic_vessel_config(
    *,
    r_true: float,
    l_true: float,
    s_true: float,
    fit_stenosis: bool,
    n: int = 50,
) -> dict:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    q_in = 2.0 + np.sin(2 * np.pi * t)
    dq_in = np.gradient(q_in)
    delta_p = r_true * q_in + s_true * np.abs(q_in) * q_in + l_true * dq_in

    vessel_name = "branch0_seg0"
    outlet_bc = "OUT"
    return {
        "calibration_parameters": {
            "calibrate_stenosis_coefficient": fit_stenosis,
            "freeze_connector_segments": True,
        },
        "y": {
            f"pressure:INFLOW:{vessel_name}": (1000.0 + delta_p).tolist(),
            f"flow:INFLOW:{vessel_name}": q_in.tolist(),
            f"pressure:{vessel_name}:{outlet_bc}": np.full(n, 1000.0).tolist(),
            f"flow:{vessel_name}:{outlet_bc}": q_in.tolist(),
        },
        "dy": {
            f"pressure:INFLOW:{vessel_name}": np.zeros(n).tolist(),
            f"flow:INFLOW:{vessel_name}": dq_in.tolist(),
            f"pressure:{vessel_name}:{outlet_bc}": np.zeros(n).tolist(),
            f"flow:{vessel_name}:{outlet_bc}": dq_in.tolist(),
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": vessel_name,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": "INFLOW", "outlet": outlet_bc},
                "zero_d_element_values": {
                    "R_poiseuille": 1.0,
                    "L": 1.0,
                    "C": 1e-6,
                    "stenosis_coefficient": 0.0,
                },
            }
        ],
        "junctions": [],
    }


def test_fit_rlc_recovers_known_parameters_without_stenosis():
    r_true, l_true = 0.05, 0.3
    config = _synthetic_vessel_config(r_true=r_true, l_true=l_true, s_true=0.0, fit_stenosis=False)
    y = config["y"]
    vessel_name = "branch0_seg0"
    q_in = np.asarray(y[f"flow:INFLOW:{vessel_name}"])
    dq_in = np.asarray(config["dy"][f"flow:INFLOW:{vessel_name}"])
    delta_p = np.asarray(y[f"pressure:INFLOW:{vessel_name}"]) - np.asarray(y["pressure:branch0_seg0:OUT"])

    r_fit, s_fit, l_fit, rel_err = _fit_rlc(delta_p, q_in, dq_in, fit_stenosis=False)

    assert s_fit == 0.0
    assert rel_err < 1e-10
    assert r_fit == pytest.approx(r_true, rel=1e-6)
    assert l_fit == pytest.approx(l_true, rel=1e-6)


def test_fit_rlc_l2_shrinks_coefficients_toward_zero():
    r_true, l_true = 0.5, 0.4
    config = _synthetic_vessel_config(r_true=r_true, l_true=l_true, s_true=0.0, fit_stenosis=False)
    y = config["y"]
    vessel_name = "branch0_seg0"
    q_in = np.asarray(y[f"flow:INFLOW:{vessel_name}"])
    dq_in = np.asarray(config["dy"][f"flow:INFLOW:{vessel_name}"])
    delta_p = np.asarray(y[f"pressure:INFLOW:{vessel_name}"]) - np.asarray(y["pressure:branch0_seg0:OUT"])

    r_unreg, _, l_unreg, _ = _fit_rlc(delta_p, q_in, dq_in, fit_stenosis=False)
    r_reg, _, l_reg, _ = _fit_rlc(
        delta_p,
        q_in,
        dq_in,
        fit_stenosis=False,
        l2_r=10.0,
        l2_l=10.0,
    )

    assert r_unreg == pytest.approx(r_true, rel=1e-5)
    assert l_unreg == pytest.approx(l_true, rel=1e-5)
    assert r_reg < r_unreg
    assert l_reg < l_unreg


def test_fit_rlc_enforces_nonnegative_r_and_l():
    r_true, l_true, s_true = 0.05, 0.2, -0.002
    config = _synthetic_vessel_config(
        r_true=r_true,
        l_true=l_true,
        s_true=s_true,
        fit_stenosis=True,
    )
    y = config["y"]
    vessel_name = "branch0_seg0"
    q_in = np.asarray(y[f"flow:INFLOW:{vessel_name}"])
    dq_in = np.asarray(config["dy"][f"flow:INFLOW:{vessel_name}"])
    delta_p = np.asarray(y[f"pressure:INFLOW:{vessel_name}"]) - np.asarray(y["pressure:branch0_seg0:OUT"])

    r_fit, s_fit, l_fit, _rel_err = _fit_rlc(
        delta_p, q_in, dq_in, fit_stenosis=True, nonneg_r=True, nonneg_l=True
    )

    assert r_fit >= 0.0
    assert l_fit >= 0.0
    assert s_fit == pytest.approx(s_true, rel=1e-4)


def test_fit_rlc_allows_negative_r_and_l_when_nonneg_disabled():
    r_true, l_true = -0.05, -0.2
    config = _synthetic_vessel_config(r_true=r_true, l_true=l_true, s_true=0.0, fit_stenosis=False)
    y = config["y"]
    vessel_name = "branch0_seg0"
    q_in = np.asarray(y[f"flow:INFLOW:{vessel_name}"])
    dq_in = np.asarray(config["dy"][f"flow:INFLOW:{vessel_name}"])
    delta_p = np.asarray(y[f"pressure:INFLOW:{vessel_name}"]) - np.asarray(y["pressure:branch0_seg0:OUT"])

    r_fit, s_fit, l_fit, rel_err = _fit_rlc(
        delta_p,
        q_in,
        dq_in,
        fit_stenosis=False,
        nonneg_r=False,
        nonneg_l=False,
    )

    assert rel_err < 1e-10
    assert r_fit == pytest.approx(r_true, rel=1e-5)
    assert l_fit == pytest.approx(l_true, rel=1e-5)
    assert s_fit == pytest.approx(0.0)


def test_fit_rlc_recovers_known_parameters_with_stenosis():
    r_true, l_true, s_true = 0.04, 0.25, 1e-4
    config = _synthetic_vessel_config(
        r_true=r_true,
        l_true=l_true,
        s_true=s_true,
        fit_stenosis=True,
    )
    y = config["y"]
    vessel_name = "branch0_seg0"
    q_in = np.asarray(y[f"flow:INFLOW:{vessel_name}"])
    dq_in = np.asarray(config["dy"][f"flow:INFLOW:{vessel_name}"])
    delta_p = np.asarray(y[f"pressure:INFLOW:{vessel_name}"]) - np.asarray(y["pressure:branch0_seg0:OUT"])

    r_fit, s_fit, l_fit, rel_err = _fit_rlc(delta_p, q_in, dq_in, fit_stenosis=True)

    assert rel_err < 1e-10
    assert r_fit == pytest.approx(r_true, rel=1e-5)
    assert l_fit == pytest.approx(l_true, rel=1e-5)
    assert s_fit == pytest.approx(s_true, rel=1e-5)


def test_calibrate_decoupled_ls_skips_nonneg_r_without_stenosis():
    """decoupled_nonneg_r applies only when calibrate_stenosis_coefficient is true."""
    config = _synthetic_vessel_config(r_true=-0.05, l_true=0.15, s_true=0.0, fit_stenosis=False)
    out = calibrate_decoupled_ls(config, nonneg_r=True, nonneg_l=False)
    values = out["vessels"][0]["zero_d_element_values"]
    assert values["R_poiseuille"] == pytest.approx(-0.05, rel=1e-4)
    assert values["L"] == pytest.approx(0.15, rel=1e-6)


def test_calibrate_decoupled_ls_uses_inlet_flow_derivative():
    """Vessel fits use dQ from the inlet flow series, not the outlet."""
    r_true, l_true = 0.07, 0.15
    config = _synthetic_vessel_config(r_true=r_true, l_true=l_true, s_true=0.0, fit_stenosis=False)
    n = len(config["y"]["flow:INFLOW:branch0_seg0"])
    # Inlet dy matches the generating model; outlet dy is deliberately wrong.
    config["dy"]["flow:branch0_seg0:OUT"] = np.full(n, 999.0).tolist()

    out = calibrate_decoupled_ls(config)
    values = out["vessels"][0]["zero_d_element_values"]
    assert values["R_poiseuille"] == pytest.approx(r_true, rel=1e-6)
    assert values["L"] == pytest.approx(l_true, rel=1e-6)


def test_calibrate_decoupled_ls_end_to_end_synthetic():
    config = _synthetic_vessel_config(r_true=0.07, l_true=0.15, s_true=0.0, fit_stenosis=False)
    out = calibrate_decoupled_ls(config)
    values = out["vessels"][0]["zero_d_element_values"]
    assert values["R_poiseuille"] == pytest.approx(0.07, rel=1e-6)
    assert values["L"] == pytest.approx(0.15, rel=1e-6)
    assert values["C"] == pytest.approx(1e-6)


def test_poor_fit_emits_warning():
    config = _synthetic_vessel_config(r_true=0.05, l_true=0.2, s_true=0.0, fit_stenosis=False)
    n = len(config["y"]["flow:INFLOW:branch0_seg0"])
    config["y"]["pressure:branch0_seg0:OUT"] = np.linspace(900.0, 1100.0, n).tolist()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrate_decoupled_ls(config)

    assert any(
        issubclass(w.category, UserWarning)
        and "branch0_seg0" in str(w.message)
        and "exceeds 10% threshold" in str(w.message)
        for w in caught
    )


def test_good_fit_does_not_warn():
    config = _synthetic_vessel_config(r_true=0.05, l_true=0.2, s_true=0.0, fit_stenosis=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrate_decoupled_ls(config)

    assert not any("exceeds" in str(w.message) for w in caught)


def test_connector_vessel_freeze():
    config = _synthetic_vessel_config(r_true=0.05, l_true=0.2, s_true=0.01, fit_stenosis=True)
    config["vessels"][0]["vessel_name"] = "branch0_seg2_connector0"
    vessel_name = config["vessels"][0]["vessel_name"]
    for obs_dict in (config["y"], config["dy"]):
        obs_dict[f"pressure:INFLOW:{vessel_name}"] = obs_dict.pop("pressure:INFLOW:branch0_seg0")
        obs_dict[f"flow:INFLOW:{vessel_name}"] = obs_dict.pop("flow:INFLOW:branch0_seg0")
        obs_dict[f"pressure:{vessel_name}:OUT"] = obs_dict.pop("pressure:branch0_seg0:OUT")
        obs_dict[f"flow:{vessel_name}:OUT"] = obs_dict.pop("flow:branch0_seg0:OUT")

    assert _is_connector_vessel(vessel_name)
    out = calibrate_decoupled_ls(config)
    values = out["vessels"][0]["zero_d_element_values"]
    assert values["R_poiseuille"] == 0.0
    assert values["L"] == 0.0
    assert values["stenosis_coefficient"] == 0.0


@pytest.mark.skipif(
    not _bundled_calibrated_json_available(),
    reason="bundled sample not present or Git LFS pointer (run git lfs pull)",
)
def test_smoke_on_bundled_sample():
    with open(BUNDLED_CALIBRATED_JSON) as f:
        config = json.load(f)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Decoupled LS fit .* exceeds .* threshold",
        )
        out = calibrate_decoupled_ls(config)
    out = _postprocess_calibrated_config(out)

    assert "vessels" in out
    assert "junctions" in out
    for vessel in out["vessels"]:
        if vessel.get("zero_d_element_type") not in {"BloodVessel", "BloodVesselCRL", "BloodVesselFC"}:
            continue
        values = vessel["zero_d_element_values"]
        assert np.isfinite(values["R_poiseuille"])
        assert np.isfinite(values["L"])


def test_run_calibration_decoupled_backend(tmp_path):
    config = _synthetic_vessel_config(r_true=0.03, l_true=0.1, s_true=0.0, fit_stenosis=False)
    input_path = tmp_path / "calibration_input.json"
    output_path = tmp_path / "calibrated_output.json"
    with open(input_path, "w") as f:
        json.dump(config, f)

    out = run_calibration(str(input_path), str(output_path), backend="decoupled_ls")
    assert output_path.is_file()
    assert out["vessels"][0]["zero_d_element_values"]["R_poiseuille"] == pytest.approx(0.03, rel=1e-4)


def test_infer_set_geo_from_zerod_path():
    path = (
        "/repo/data/zeroD/VMR_pulmo/quadratic_resistor_gen_loss/0080_0001/"
        "bifurcations_EL_calibration_input_BloodVesselJunction.json"
    )
    assert infer_set_geo_from_zerod_path(path) == ("VMR_pulmo", "0080_0001")


def test_plot_rsl_fits_writes_png(tmp_path):
    config = _synthetic_vessel_config(r_true=0.05, l_true=0.2, s_true=0.0, fit_stenosis=False)
    results_root = tmp_path / "results"
    calibrate_decoupled_ls(
        config,
        plot_rsl_fits=True,
        set_name="TEST_SET",
        geo_name="geo_001",
        results_root=results_root,
    )
    plot_dir = rsl_fits_output_dir("TEST_SET", "geo_001", results_root=results_root)
    plots = list(plot_dir.glob("*.png"))
    assert len(plots) == 1
    assert plots[0].stat().st_size > 0
