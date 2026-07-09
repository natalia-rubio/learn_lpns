"""Tests for per-geometry forward jax/split path helpers."""

import pytest

from learn_lpns.zerod_calibration.nn_inference import forward_jax_pickle_path, forward_split_indices_path


def test_forward_jax_pickle_path_includes_geo_name():
    path = forward_jax_pickle_path(
        "data",
        "VMR_all",
        "gen_loss",
        "bifurcations_EL",
        "0129_0000",
        num_geos=1,
    )
    assert path.endswith("forward/0129_0000/jax_arrays_num_geos_1.pkl")
    assert "/jax_arrays/VMR_all/gen_loss/bifurcations_EL/forward/0129_0000/" in path.replace("\\", "/")


def test_forward_jax_pickle_path_vessel_variant():
    path = forward_jax_pickle_path(
        "data",
        "VMR_all",
        "gen_loss",
        "bifurcations_EL",
        "0129_0000",
        vessel=True,
    )
    assert "jax_arrays_vessel_num_geos_1.pkl" in path
    assert "/forward/0129_0000/" in path.replace("\\", "/")


def test_forward_jax_pickle_path_requires_geo_name():
    with pytest.raises(ValueError, match="geo_name"):
        forward_jax_pickle_path("data", "S", "base", "bifurcations", "")


def test_forward_split_indices_path_includes_geo_name():
    path = forward_split_indices_path(
        "data",
        "VMR_all",
        "gen_loss",
        "bifurcations_EL",
        "0129_0000",
    )
    assert path.endswith("forward/0129_0000/train_val_ind_VMR_all_num_geos_1")
    assert "/split_indices/VMR_all/gen_loss/bifurcations_EL/forward/0129_0000/" in path.replace("\\", "/")


def test_forward_split_indices_path_requires_geo_name():
    with pytest.raises(ValueError, match="geo_name"):
        forward_split_indices_path("data", "S", "base", "bifurcations", "")
