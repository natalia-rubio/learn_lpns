"""Tests for train-set R/S/L prediction clipping."""

from __future__ import annotations

import numpy as np
import pytest

from learn_lpns.neural_network.nn_util import (
    attach_train_output_bounds,
    clip_rsl_predictions,
    compute_train_output_bounds_from_split,
    resolve_train_output_bounds,
)
from learn_lpns.tools.basic import save_dict


class _FakeModel:
    def __init__(self, output: np.ndarray, **attrs):
        self.output = output
        for key, value in attrs.items():
            setattr(self, key, value)


def test_attach_train_output_bounds_stores_train_min_max():
    output = np.array(
        [
            [1.0, 10.0, 0.1],
            [5.0, 0.0, 0.2],
            [3.0, 5.0, 0.5],
        ],
        dtype=float,
    )
    model = _FakeModel(output)
    attach_train_output_bounds(model, [0, 2])
    assert model.train_output_min == pytest.approx(np.array([1.0, 5.0, 0.1]))
    assert model.train_output_max == pytest.approx(np.array([3.0, 10.0, 0.5]))


def test_clip_rsl_predictions():
    pred_R = np.array([-1.0, 2.0, 10.0])
    pred_S = np.array([0.0, 3.0, 4.0])
    pred_L = np.array([0.05, 0.2, 0.9])
    bounds_min = np.array([0.0, 1.0, 0.1])
    bounds_max = np.array([5.0, 3.5, 0.8])
    cR, cS, cL = clip_rsl_predictions(pred_R, pred_S, pred_L, bounds_min, bounds_max)
    assert cR == pytest.approx(np.array([0.0, 2.0, 5.0]))
    assert cS == pytest.approx(np.array([1.0, 3.0, 3.5]))
    assert cL == pytest.approx(np.array([0.1, 0.2, 0.8]))


def test_resolve_train_output_bounds_prefers_train_bounds():
    model = _FakeModel(np.zeros((2, 3)))
    model.train_output_min = np.array([0.1, 0.2, 0.3])
    model.train_output_max = np.array([1.1, 1.2, 1.3])
    mins, maxs = resolve_train_output_bounds(model)
    assert mins == pytest.approx(model.train_output_min)
    assert maxs == pytest.approx(model.train_output_max)


def test_compute_train_output_bounds_from_split(tmp_path):
    output = np.array(
        [
            [1.0, 10.0, 0.1],
            [5.0, 0.0, 0.2],
            [3.0, 5.0, 0.5],
            [9.0, 1.0, 0.9],
        ],
        dtype=float,
    )
    model = _FakeModel(
        output,
        set_name="VMR_test",
        geometry_variant="bifurcations_EL",
        set_type="all",
        num_geos=2,
        model_name_suffix="",
    )
    split_path = (
        tmp_path
        / "data"
        / "split_indices"
        / "VMR_test"
        / "gen_loss"
        / "bifurcations_EL"
        / "all"
        / "train_val_ind_VMR_test_num_geos_2"
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict(
        {
            "train_geometries": ["g0"],
            "val_geometries": ["g1"],
            "geometry_indices": {
                "g0": {"junction": (0, 2), "vessel": (0, 1)},
                "g1": {"junction": (2, 4), "vessel": (1, 2)},
            },
        },
        str(split_path),
    )
    mins, maxs = compute_train_output_bounds_from_split(
        model,
        data_root=str(tmp_path / "data"),
        run_config_suffix="gen_loss",
    )
    assert mins == pytest.approx(np.array([1.0, 0.0, 0.1]))
    assert maxs == pytest.approx(np.array([5.0, 10.0, 0.2]))
