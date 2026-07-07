"""Tests for MAPE (mean relative error) in forward MSE metrics."""

from __future__ import annotations

import numpy as np

from learn_lpns.zerod_calibration.forward_mse import _compute_aligned_mse


def test_compute_aligned_mse_mean_rel_error_is_mape():
    p3d = np.array([100.0, 110.0, 90.0])
    p0d = np.array([95.0, 121.0, 81.0])
    metrics = _compute_aligned_mse(p0d, p3d, "pressure")
    assert metrics is not None
    expected_mape = np.mean(np.abs(p0d - p3d) / np.abs(p3d))
    assert metrics["mean_rel_error"] == expected_mape
    assert metrics["mean_rel_error"] < metrics["rel_error_at_max"]


def test_compute_aligned_mse_mean_rel_error_skips_zero_reference():
    p3d = np.array([0.0, 100.0, 100.0])
    p0d = np.array([1.0, 90.0, 110.0])
    metrics = _compute_aligned_mse(p0d, p3d, "pressure")
    assert metrics is not None
    assert metrics["mean_rel_error"] == np.mean([0.1, 0.1])


def test_compute_aligned_mse_mean_rel_error_nan_when_all_reference_zero():
    p3d = np.array([0.0, 0.0])
    p0d = np.array([1.0, 2.0])
    metrics = _compute_aligned_mse(p0d, p3d, "pressure")
    assert metrics is not None
    assert np.isnan(metrics["mean_rel_error"])
