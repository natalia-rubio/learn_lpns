"""Tests for junction flow_split aggregation methods."""

from __future__ import annotations

import pytest

from learn_lpns.data_processing.inputs_from_0d_config import (
    FLOW_SPLIT_METHOD_MEAN_OVER_TIME,
    FLOW_SPLIT_METHOD_PEAK_INLET_FLOW,
    _flow_split_percentages_for_junction,
)


def _make_results(
    *,
    inlet_flows: dict[float, float],
    out0_flows: dict[float, float],
    out1_flows: dict[float, float],
) -> tuple[dict, list[float]]:
    results = {
        "inlet": {t: {"flow_out": q} for t, q in inlet_flows.items()},
        "out0": {t: {"flow_in": q} for t, q in out0_flows.items()},
        "out1": {t: {"flow_in": q} for t, q in out1_flows.items()},
    }
    times = sorted(inlet_flows.keys())
    return results, times


def test_flow_split_mean_over_time_averages_ratios():
    results, times = _make_results(
        inlet_flows={0.0: 10.0, 1.0: 20.0},
        out0_flows={0.0: 3.0, 1.0: 16.0},
        out1_flows={0.0: 7.0, 1.0: 4.0},
    )
    fs0, fs1 = _flow_split_percentages_for_junction(
        results,
        times,
        original_inlet_name="inlet",
        out0_name="out0",
        out1_name="out1",
        junction_name="j0",
        flow_split_method=FLOW_SPLIT_METHOD_MEAN_OVER_TIME,
    )
    assert fs0 == pytest.approx(55.0)
    assert fs1 == pytest.approx(45.0)


def test_flow_split_peak_inlet_flow_uses_max_inlet_time():
    results, times = _make_results(
        inlet_flows={0.0: 10.0, 1.0: 20.0},
        out0_flows={0.0: 3.0, 1.0: 16.0},
        out1_flows={0.0: 7.0, 1.0: 4.0},
    )
    fs0, fs1 = _flow_split_percentages_for_junction(
        results,
        times,
        original_inlet_name="inlet",
        out0_name="out0",
        out1_name="out1",
        junction_name="j0",
        flow_split_method=FLOW_SPLIT_METHOD_PEAK_INLET_FLOW,
    )
    assert fs0 == pytest.approx(80.0)
    assert fs1 == pytest.approx(20.0)


def test_flow_split_defaults_to_fifty_fifty_when_no_valid_inlet_flow():
    results, times = _make_results(
        inlet_flows={0.0: 1.0, 1.0: 2.0},
        out0_flows={0.0: 0.5, 1.0: 1.0},
        out1_flows={0.0: 0.5, 1.0: 1.0},
    )
    fs0, fs1 = _flow_split_percentages_for_junction(
        results,
        times,
        original_inlet_name="inlet",
        out0_name="out0",
        out1_name="out1",
        junction_name="j0",
        flow_split_method=FLOW_SPLIT_METHOD_PEAK_INLET_FLOW,
    )
    assert fs0 == pytest.approx(50.0)
    assert fs1 == pytest.approx(50.0)
