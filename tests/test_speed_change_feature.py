"""Tests for junction speed_change feature."""

from __future__ import annotations

import numpy as np
import pytest

from learn_lpns.data_processing.inputs_from_0d_config import _speed_change_from_areas


def test_speed_change_from_areas_formula():
    a_in, a_out, flow_split = 2.0, 0.5, 25.0
    speed_change = _speed_change_from_areas(flow_split, a_in, a_out)
    expected = (100.0 / flow_split) ** 2 / (a_in**2) - 1.0 / (a_out**2)
    assert speed_change == pytest.approx(expected)


def test_speed_change_pairs_with_outlet_q_squared():
    a_in, a_out, flow_split = 2.5, 0.4, 40.0
    speed_change = _speed_change_from_areas(flow_split, a_in, a_out)
    q_out = flow_split / 100.0
    v_in_sq = 1.0 / (a_in**2)
    v_out_sq = (q_out / a_out) ** 2
    assert q_out * q_out * speed_change == pytest.approx(v_in_sq - v_out_sq)


def test_speed_change_from_areas_invalid_inputs_are_nan():
    assert np.isnan(_speed_change_from_areas(np.nan, 2.0, 0.5))
    assert np.isnan(_speed_change_from_areas(0.0, 2.0, 0.5))
    assert np.isnan(_speed_change_from_areas(50.0, 0.0, 0.5))
    assert np.isnan(_speed_change_from_areas(50.0, 2.0, 0.0))
