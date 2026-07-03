"""Tests for input-vs-output correlation scatter plots."""

from pathlib import Path

import numpy as np

from learn_lpns.visualizations.plot_input_output_correlations import (
    _resolve_input_feature_names,
    plot_input_output_correlations,
)


def test_resolve_input_feature_names_from_dict():
    data_dict = {
        "input": np.zeros((3, 2)),
        "input_feature_names": ["a", "b"],
    }
    assert _resolve_input_feature_names(data_dict, vessel=False) == ["a", "b"]


def test_plot_input_output_correlations_writes_one_figure_per_input(tmp_path):
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    outputs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=float)
    feature_names = ["feat_a", "feat_b"]

    saved = plot_input_output_correlations(
        inputs,
        outputs,
        feature_names,
        set_name="test_set",
        geometry_variant="bifurcations_EL",
        modality="junction",
        output_dir=str(tmp_path),
        run_config_suffix="gen_loss",
    )

    assert len(saved) == 2
    assert all(path.endswith("_vs_rsl.png") for path in saved)
    for path in saved:
        assert Path(path).is_file()
