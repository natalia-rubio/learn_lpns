"""Tests for generation-gated stenosis (RSL vs RL)."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from learn_lpns.neural_network.nn_model import S_OUTPUT_COLUMN, NeuralNet
from learn_lpns.zerod_calibration.decoupled_ls_calibration import calibrate_decoupled_ls
from learn_lpns.zerod_calibration.stenosis_generation import (
    apply_stenosis_generation_gate,
    generation_allows_stenosis,
    zero_stenosis_by_generation,
)


def _synthetic_junction_outlet_config(
    *,
    r_true: float,
    l_true: float,
    s_true: float,
    n: int = 50,
) -> dict:
    """Bifurcation junction J0 at generation 0 (inlet vessel is root)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    q_in = 2.0 + np.sin(2 * np.pi * t)
    dq_out = np.gradient(q_in)
    delta_p = r_true * q_in + s_true * np.abs(q_in) * q_in + l_true * dq_out

    junc_name = "J0"
    root = "branch0_seg0"
    outlet0 = "branch1_seg0"
    outlet1 = "branch2_seg0"
    return {
        "calibration_parameters": {
            "calibrate_stenosis_coefficient": True,
            "freeze_connector_segments": True,
        },
        "y": {
            f"pressure:{root}:{junc_name}": (1000.0 + delta_p).tolist(),
            f"flow:{junc_name}:{outlet0}": q_in.tolist(),
            f"pressure:{junc_name}:{outlet0}": np.full(n, 1000.0).tolist(),
            f"flow:{junc_name}:{outlet1}": (q_in * 0.6).tolist(),
            f"pressure:{junc_name}:{outlet1}": np.full(n, 1000.0).tolist(),
            f"pressure:INFLOW:{root}": np.full(n, 1000.0).tolist(),
            f"flow:INFLOW:{root}": (q_in * 2).tolist(),
        },
        "dy": {
            f"pressure:{root}:{junc_name}": np.zeros(n).tolist(),
            f"flow:{junc_name}:{outlet0}": dq_out.tolist(),
            f"pressure:{junc_name}:{outlet0}": np.zeros(n).tolist(),
            f"flow:{junc_name}:{outlet1}": np.gradient(q_in * 0.6).tolist(),
            f"pressure:{junc_name}:{outlet1}": np.zeros(n).tolist(),
            f"pressure:INFLOW:{root}": np.zeros(n).tolist(),
            f"flow:INFLOW:{root}": np.gradient(q_in * 2).tolist(),
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": root,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": "INFLOW", "outlet": junc_name},
                "zero_d_element_values": {"R_poiseuille": 1.0, "L": 1.0, "C": 1e-6, "stenosis_coefficient": 0.0},
            },
            {
                "vessel_id": 1,
                "vessel_name": outlet0,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": junc_name, "outlet": "OUT0"},
                "zero_d_element_values": {"R_poiseuille": 1.0, "L": 1.0, "C": 1e-6, "stenosis_coefficient": 0.0},
            },
            {
                "vessel_id": 2,
                "vessel_name": outlet1,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": junc_name, "outlet": "OUT1"},
                "zero_d_element_values": {"R_poiseuille": 1.0, "L": 1.0, "C": 1e-6, "stenosis_coefficient": 0.0},
            },
        ],
        "junctions": [
            {
                "junction_name": junc_name,
                "junction_type": "BloodVesselJunction",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "outlet_blocks": [outlet0, outlet1],
                "junction_values": {
                    "R_poiseuille": [0.0, 0.0],
                    "L": [0.0, 0.0],
                    "stenosis_coefficient": [0.0, 0.0],
                },
            }
        ],
    }


def _synthetic_downstream_vessel_config(
    *,
    r_true: float,
    l_true: float,
    s_true: float,
    n: int = 50,
) -> dict:
    """Root bifurcation (gen 0) with one downstream vessel branch1_seg0 at generation 1."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    q_in = 2.0 + np.sin(2 * np.pi * t)
    dq_out = np.gradient(q_in)
    delta_p = r_true * q_in + s_true * np.abs(q_in) * q_in + l_true * dq_out

    junc_name = "J0"
    downstream = "branch1_seg0"
    root = "branch0_seg0"
    return {
        "calibration_parameters": {
            "calibrate_stenosis_coefficient": True,
            "freeze_connector_segments": True,
        },
        "y": {
            f"pressure:{junc_name}:{downstream}": (1000.0 + delta_p).tolist(),
            f"flow:{junc_name}:{downstream}": q_in.tolist(),
            f"pressure:{downstream}:OUT": np.full(n, 1000.0).tolist(),
            f"flow:{downstream}:OUT": q_in.tolist(),
            f"pressure:INFLOW:{root}": np.full(n, 1000.0).tolist(),
            f"flow:INFLOW:{root}": (q_in * 2).tolist(),
        },
        "dy": {
            f"pressure:{junc_name}:{downstream}": np.zeros(n).tolist(),
            f"flow:{junc_name}:{downstream}": dq_out.tolist(),
            f"pressure:{downstream}:OUT": np.zeros(n).tolist(),
            f"flow:{downstream}:OUT": dq_out.tolist(),
            f"pressure:INFLOW:{root}": np.zeros(n).tolist(),
            f"flow:INFLOW:{root}": np.gradient(q_in * 2).tolist(),
        },
        "vessels": [
            {
                "vessel_id": 0,
                "vessel_name": root,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": "INFLOW", "outlet": junc_name},
                "zero_d_element_values": {"R_poiseuille": 1.0, "L": 1.0, "C": 1e-6, "stenosis_coefficient": 0.0},
            },
            {
                "vessel_id": 1,
                "vessel_name": downstream,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": junc_name, "outlet": "OUT"},
                "zero_d_element_values": {"R_poiseuille": 1.0, "L": 1.0, "C": 1e-6, "stenosis_coefficient": 0.0},
            },
            {
                "vessel_id": 2,
                "vessel_name": "branch2_seg0",
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": junc_name, "outlet": "OUT2"},
                "zero_d_element_values": {"R_poiseuille": 1.0, "L": 1.0, "C": 1e-6, "stenosis_coefficient": 0.0},
            },
        ],
        "junctions": [
            {
                "junction_name": junc_name,
                "junction_type": "BloodVesselJunction",
                "inlet_vessels": [0],
                "outlet_vessels": [1, 2],
                "junction_values": {
                    "R_poiseuille": [0.0, 0.0],
                    "L": [0.0, 0.0],
                    "stenosis_coefficient": [0.0, 0.0],
                },
            }
        ],
    }


def test_legacy_max_generation_overrides_both_limits():
    from learn_lpns.config.models import StenosisGenerationLimitConfig

    cfg = StenosisGenerationLimitConfig(
        enabled=True,
        junction_max_generation=2.0,
        vessel_max_generation=1.0,
        max_generation=3.0,
    )
    assert cfg.junction_limit() == pytest.approx(3.0)
    assert cfg.vessel_limit() == pytest.approx(3.0)


def test_generation_allows_stenosis_inclusive():
    assert generation_allows_stenosis(0.0, 1.0)
    assert generation_allows_stenosis(1.0, 1.0)
    assert not generation_allows_stenosis(1.01, 1.0)


def test_zero_stenosis_by_generation():
    pred = np.array([0.5, -0.2, 1.0, 3.0])
    gen = np.array([0.0, 1.0, 2.0, 0.0])
    clipped = zero_stenosis_by_generation(pred, gen, max_generation=1.0)
    assert clipped[0] == pytest.approx(0.5)
    assert clipped[1] == pytest.approx(-0.2)
    assert clipped[2] == pytest.approx(0.0)
    assert clipped[3] == pytest.approx(3.0)


def test_resolve_stenosis_generation_limits_separate_modalities():
    from learn_lpns.zerod_calibration.stenosis_generation import resolve_stenosis_generation_limits

    enabled, j_max, v_max = resolve_stenosis_generation_limits(None)
    # Root config/defaults.yaml may enable gating; assert split limits are loaded.
    assert j_max == pytest.approx(2.0)
    assert v_max == pytest.approx(1.0)
    assert enabled is True


def test_calibrate_decoupled_ls_rl_for_high_generation_vessel():
    s_true = 0.002
    config = _synthetic_downstream_vessel_config(r_true=0.05, l_true=0.2, s_true=s_true)
    out = calibrate_decoupled_ls(
        config,
        stenosis_generation_limit_enabled=True,
        vessel_stenosis_generation_max=0.0,
        junction_stenosis_generation_max=1.0,
    )
    downstream = next(v for v in out["vessels"] if v["vessel_name"] == "branch1_seg0")
    values = downstream["zero_d_element_values"]
    assert values["stenosis_coefficient"] == 0.0
    assert np.isfinite(values["R_poiseuille"])
    assert values["R_poiseuille"] >= 0.0


def test_calibrate_decoupled_ls_rsl_for_low_generation_vessel():
    s_true = 0.002
    config = _synthetic_downstream_vessel_config(r_true=0.05, l_true=0.2, s_true=s_true)
    out = calibrate_decoupled_ls(
        config,
        stenosis_generation_limit_enabled=True,
        vessel_stenosis_generation_max=1.0,
        junction_stenosis_generation_max=1.0,
    )
    downstream = next(v for v in out["vessels"] if v["vessel_name"] == "branch1_seg0")
    values = downstream["zero_d_element_values"]
    assert values["stenosis_coefficient"] == pytest.approx(s_true, rel=1e-3)


def test_nn_s_sample_weights_zero_for_high_generation():
    n_rows = 4
    network_params = {
        "num_input_features": 3,
        "num_layers": 1,
        "layer_width": 4,
        "num_output_features": 1,
        "target_output_column": S_OUTPUT_COLUMN,
        "output_type": "rri",
        "set_name": "test",
        "set_type": "test",
        "num_geos": 1,
        "geometry_variant": "bifurcations_EL",
        "data_dict": {
            "input": jnp.ones((n_rows, 3), dtype=jnp.float32),
            "output_rri": jnp.ones((n_rows, 3), dtype=jnp.float32),
            "generation": jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32),
        },
        "use_leaky_relu": False,
        "asymmetric_loss": False,
        "asymmetric_loss_overestimate_weight": 1.0,
        "generation_weighted_loss": False,
        "generation_weighted_loss_decay_base": 2.0,
        "stenosis_generation_limit_enabled": True,
        "stenosis_generation_max": 1.0,
    }
    optimizer_params = {"init": 0.01, "transition_steps": 100, "decay_rate": 0.99}
    model = NeuralNet(network_params, optimizer_params)
    weights = np.array(model._sample_weights_for_indices(jnp.arange(n_rows)))
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)
    assert weights[3] == pytest.approx(0.0)


def test_calibrate_junction_rsl_when_inlet_gen_within_limit():
    s_true = 0.0015
    config = _synthetic_junction_outlet_config(r_true=0.04, l_true=0.18, s_true=s_true)
    out = calibrate_decoupled_ls(
        config,
        stenosis_generation_limit_enabled=True,
        junction_stenosis_generation_max=0.0,
        vessel_stenosis_generation_max=1.0,
    )
    junc = out["junctions"][0]["junction_values"]
    assert junc["stenosis_coefficient"][0] == pytest.approx(s_true, rel=1e-3)
    assert junc["stenosis_coefficient"][1] != 0.0


def test_calibrate_junction_rl_when_inlet_gen_exceeds_limit():
    """Downstream junction J1 has inlet generation 1; max_generation=0 forces RL on both outlets."""
    s_true = 0.0015
    base = _synthetic_downstream_vessel_config(r_true=0.04, l_true=0.18, s_true=s_true)
    n = 50
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    q_in = 2.0 + np.sin(2 * np.pi * t)
    dq_out = np.gradient(q_in)
    delta_p = 0.04 * q_in + s_true * np.abs(q_in) * q_in + 0.18 * dq_out

    junc_name = "J1"
    inlet_vessel = "branch1_seg0"
    out0 = "branch3_seg0"
    out1 = "branch4_seg0"
    base["y"].update(
        {
            f"pressure:{inlet_vessel}:{junc_name}": (1000.0 + delta_p).tolist(),
            f"flow:{junc_name}:{out0}": q_in.tolist(),
            f"pressure:{junc_name}:{out0}": np.full(n, 1000.0).tolist(),
            f"flow:{junc_name}:{out1}": (q_in * 0.7).tolist(),
            f"pressure:{junc_name}:{out1}": np.full(n, 1000.0).tolist(),
        }
    )
    base["dy"].update(
        {
            f"pressure:{inlet_vessel}:{junc_name}": np.zeros(n).tolist(),
            f"flow:{junc_name}:{out0}": dq_out.tolist(),
            f"pressure:{junc_name}:{out0}": np.zeros(n).tolist(),
            f"flow:{junc_name}:{out1}": np.gradient(q_in * 0.7).tolist(),
            f"pressure:{junc_name}:{out1}": np.zeros(n).tolist(),
        }
    )
    base["vessels"].extend(
        [
            {
                "vessel_id": 3,
                "vessel_name": out0,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": junc_name, "outlet": "OUT3"},
                "zero_d_element_values": {
                    "R_poiseuille": 1.0,
                    "L": 1.0,
                    "C": 1e-6,
                    "stenosis_coefficient": 0.0,
                },
            },
            {
                "vessel_id": 4,
                "vessel_name": out1,
                "zero_d_element_type": "BloodVessel",
                "boundary_conditions": {"inlet": junc_name, "outlet": "OUT4"},
                "zero_d_element_values": {
                    "R_poiseuille": 1.0,
                    "L": 1.0,
                    "C": 1e-6,
                    "stenosis_coefficient": 0.0,
                },
            },
        ]
    )
    base["junctions"].append(
        {
            "junction_name": junc_name,
            "junction_type": "BloodVesselJunction",
            "inlet_vessels": [1],
            "outlet_vessels": [3, 4],
            "outlet_blocks": [out0, out1],
            "junction_values": {
                "R_poiseuille": [0.0, 0.0],
                "L": [0.0, 0.0],
                "stenosis_coefficient": [0.0, 0.0],
            },
        }
    )

    out = calibrate_decoupled_ls(
        base,
        stenosis_generation_limit_enabled=True,
        junction_stenosis_generation_max=0.0,
        vessel_stenosis_generation_max=1.0,
    )
    j1 = next(j for j in out["junctions"] if j["junction_name"] == junc_name)
    s_vals = j1["junction_values"]["stenosis_coefficient"]
    assert s_vals[0] == 0.0
    assert s_vals[1] == 0.0


def test_apply_stenosis_generation_gate_zeros_high_gen():
    pred = np.array([1.5, 2.5, 3.5])
    gen = np.array([0.0, 1.0, 2.0])
    out = apply_stenosis_generation_gate(pred, gen, enabled=True, max_generation=1.0)
    assert out[0] == pytest.approx(1.5)
    assert out[1] == pytest.approx(2.5)
    assert out[2] == pytest.approx(0.0)


def test_apply_stenosis_generation_gate_disabled():
    pred = np.array([1.0, 2.0])
    gen = np.array([0.0, 5.0])
    out = apply_stenosis_generation_gate(pred, gen, enabled=False, max_generation=1.0)
    np.testing.assert_array_equal(out, pred)
