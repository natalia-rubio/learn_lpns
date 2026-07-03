"""Tests for generation-filtered stenosis clipping."""

from __future__ import annotations

import os

import numpy as np
import pytest

from learn_lpns.data_processing.jax_arrays_paths import (
    jax_arrays_filename,
    parse_trial_id_from_split_path,
    resolve_jax_arrays_path,
)
from learn_lpns.data_processing.stenosis_clipping import (
    clip_stenosis_in_data_dict,
    clip_stenosis_values,
    compute_stenosis_clip_bounds,
    write_clipped_jax_pickles,
)
from learn_lpns.neural_network.launch_training import _jax_arrays_path
from learn_lpns.tools.basic import load_dict, save_dict


def test_compute_stenosis_clip_bounds_train_and_generation_filter():
    output_rri = np.array(
        [
            [1.0, 100.0, 0.1],
            [1.0, -50.0, 0.1],
            [1.0, 500.0, 0.1],
            [1.0, 10.0, 0.1],
        ]
    )
    generation = np.array([0.0, 1.0, 2.0, 0.0])
    train_inds = np.array([0, 1, 2, 3])
    s_min, s_max = compute_stenosis_clip_bounds(output_rri, generation, train_inds, clip_generation_number=1)
    assert s_min == pytest.approx(-50.0)
    assert s_max == pytest.approx(100.0)

    val_only_train = np.array([3])
    s_min_v, s_max_v = compute_stenosis_clip_bounds(output_rri, generation, val_only_train, clip_generation_number=1)
    assert s_max_v == pytest.approx(10.0)
    assert 500.0 not in (s_min_v, s_max_v)


def test_clip_stenosis_in_data_dict_only_touches_s_column():
    data_dict = {
        "output_rri": np.array([[1.0, 100.0, 0.5], [2.0, -20.0, 0.6]]),
        "generation": np.array([0.0, 1.0]),
    }
    clipped = clip_stenosis_in_data_dict(data_dict, -10.0, 50.0)
    out = np.asarray(clipped["output_rri"])
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(50.0)
    assert out[1, 1] == pytest.approx(-10.0)


def test_parse_trial_id_and_jax_filenames():
    split = "/data/split_indices/VMR/all/train_val_ind_VMR_num_geos_5_trial_2"
    assert parse_trial_id_from_split_path(split) == 2
    assert jax_arrays_filename(5, vessel=False, stenosis_clipping_enabled=True, split_path=split) == (
        "jax_arrays_num_geos_5_trial_2_clipped.pkl"
    )
    assert jax_arrays_filename(5, vessel=False, stenosis_clipping_enabled=False) == (
        "jax_arrays_num_geos_5_unclipped.pkl"
    )
    assert jax_arrays_filename(5, vessel=False, stenosis_clipping_enabled=True, split_path=None) == (
        "jax_arrays_num_geos_5_clipped.pkl"
    )


def test_resolve_jax_arrays_path_trial_aware():
    path = resolve_jax_arrays_path(
        "data",
        "VMR_test",
        "bifurcations_EL",
        "all",
        5,
        "gen_loss",
        vessel=False,
        split_path="data/split_indices/VMR_test/gen_loss/bifurcations_EL/all/train_val_ind_VMR_test_num_geos_5_trial_1",
        stenosis_clipping_enabled=True,
    )
    assert path.endswith("jax_arrays_num_geos_5_trial_1_clipped.pkl")


def test_write_clipped_jax_pickles_round_trip(tmp_path):
    junction = {
        "output_rri": np.array([[1.0, 100.0, 0.1], [1.0, -50.0, 0.1], [1.0, 500.0, 0.1]]),
        "generation": np.array([0.0, 1.0, 0.0]),
        "geometry_names_order": ["g0", "g1"],
        "geometry_row_ranges": [(0, 2), (2, 3)],
    }
    vessel = {
        "output_rri": np.array([[1.0, 20.0, 0.1], [1.0, 30.0, 0.1]]),
        "generation": np.array([0.0, 1.0]),
        "geometry_names_order": ["g0", "g1"],
        "geometry_row_ranges": [(0, 1), (1, 2)],
    }
    split_dict = {
        "train_geometries": ["g0"],
        "val_geometries": ["g1"],
        "geometry_indices": {
            "g0": {"junction": (0, 2), "vessel": (0, 1)},
            "g1": {"junction": (2, 3), "vessel": (1, 2)},
        },
    }
    j_path = tmp_path / "junction_clipped.pkl"
    v_path = tmp_path / "vessel_clipped.pkl"
    metadata = write_clipped_jax_pickles(
        junction_unclipped=junction,
        vessel_unclipped=vessel,
        split_dict=split_dict,
        clip_generation_number=1,
        junction_out_path=str(j_path),
        vessel_out_path=str(v_path),
    )
    loaded = load_dict(str(j_path))
    assert "stenosis_clip_bounds" in loaded
    assert loaded["stenosis_clip_bounds"]["junction"]["s_max"] == pytest.approx(100.0)
    assert np.asarray(loaded["output_rri"])[2, 1] <= metadata["junction"]["s_max"]


def test_launch_training_jax_path_uses_split_path(tmp_path):
    split_path = str(tmp_path / "train_val_ind_VMR_num_geos_3_trial_2")
    path = _jax_arrays_path(
        str(tmp_path),
        "VMR",
        "bifurcations_EL",
        "all",
        3,
        "gen_loss",
        vessel=False,
        split_path=split_path,
        stenosis_clipping_enabled=True,
    )
    assert path.endswith("jax_arrays_num_geos_3_trial_2_clipped.pkl")


def test_clip_stenosis_values():
    values = np.array([-100.0, 0.0, 200.0])
    clipped = clip_stenosis_values(values, -10.0, 50.0)
    assert clipped == pytest.approx(np.array([-10.0, 0.0, 50.0]))
