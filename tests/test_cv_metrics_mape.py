"""Tests for MAPE columns in CV summary aggregation."""

from __future__ import annotations

import numpy as np

from learn_lpns.zerod_calibration.cv_metrics import trial_metrics_to_row, write_metric_summary_csv


def test_trial_metrics_to_row_includes_pressure_mean_rel_error():
    trial_metrics = {
        "geometric": {
            "overall_mse": [1.0],
            "mean_pressure_mse": [1.0],
            "mean_flow_mse": [],
            "overall_max_error": [2.0],
            "mean_pressure_max_error": [2.0],
            "mean_flow_max_error": [],
            "overall_max_rel_error": [0.1],
            "mean_pressure_max_rel_error": [0.1],
            "mean_flow_max_rel_error": [],
            "overall_mean_rel_error": [0.05],
            "mean_pressure_mean_rel_error": [0.05],
            "mean_flow_mean_rel_error": [],
        }
    }
    row = trial_metrics_to_row(0, "geo_a", trial_metrics)
    assert row["PressureMeanRelError_geometric"] == 0.05
    assert row["MeanRelError_geometric"] == 0.05


def test_write_metric_summary_csv_pressure_mean_rel_error(tmp_path):
    rows = [
        {
            "trial_id": 0,
            "val_geometries": "g1",
            "PressureMeanRelError_geometric": 0.04,
        },
        {
            "trial_id": 1,
            "val_geometries": "g2",
            "PressureMeanRelError_geometric": 0.06,
        },
    ]
    out = tmp_path / "summary_pressure_mean_rel_error.csv"
    write_metric_summary_csv(str(out), "PressureMeanRelError_", ["geometric"], rows)
    text = out.read_text()
    assert "PressureMeanRelError_geometric" in text
    assert "0.05" in text or "0.050" in text
