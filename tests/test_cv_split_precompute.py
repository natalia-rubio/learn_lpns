"""Tests for CV split pre-computation before parallel trial dispatch."""

import os
from pathlib import Path

import pytest

from learn_lpns.data_processing.generate_split_indices import load_split_for_training
from learn_lpns.zerod_calibration.run_cross_validation import _precompute_cv_splits


GEOMETRIES = ["g0", "g1", "g2", "g3", "g4"]


@pytest.fixture
def minimal_data_dict():
    return {
        "geometry_row_ranges": [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)],
        "geometry_names_order": GEOMETRIES,
        "input": [[0.0] * 3] * 10,
    }


def test_precompute_cv_splits_writes_distinct_trial_files(tmp_path: Path, minimal_data_dict):
    split_dir = tmp_path / "split_indices"
    specs = _precompute_cv_splits(
        trials_to_run=[0, 1, 2],
        num_trials=3,
        trial_index=None,
        set_name="VMR_test",
        num_geos=len(GEOMETRIES),
        geometries=GEOMETRIES,
        num_pts=10,
        data_dict=minimal_data_dict,
        vessel_jax_path=str(tmp_path / "missing_vessel.pkl"),
        split_indices_dir=str(split_dir),
        percent_train=0.8,
    )
    assert len(specs) == 3
    val_sets = [frozenset(s.val_geometries) for s in specs]
    assert len(set(val_sets)) == 3

    for spec in specs:
        assert os.path.isfile(spec.split_path)
        split_dict = load_split_for_training(spec.split_path)
        assert list(split_dict["val_geometries"]) == list(spec.val_geometries)
        assert spec.trial_id == int(spec.split_path.rsplit("_trial_", 1)[-1])


def test_precompute_cv_splits_reuses_existing_split_for_single_trial_rerun(
    tmp_path: Path, minimal_data_dict
):
    split_dir = tmp_path / "split_indices"
    first = _precompute_cv_splits(
        trials_to_run=[0],
        num_trials=5,
        trial_index=None,
        set_name="VMR_test",
        num_geos=len(GEOMETRIES),
        geometries=GEOMETRIES,
        num_pts=10,
        data_dict=minimal_data_dict,
        vessel_jax_path=str(tmp_path / "missing_vessel.pkl"),
        split_indices_dir=str(split_dir),
        percent_train=0.8,
    )
    assert len(first) == 1
    split_path = first[0].split_path
    mtime_before = os.path.getmtime(split_path)

    rerun = _precompute_cv_splits(
        trials_to_run=[0],
        num_trials=5,
        trial_index=0,
        set_name="VMR_test",
        num_geos=len(GEOMETRIES),
        geometries=GEOMETRIES,
        num_pts=10,
        data_dict=minimal_data_dict,
        vessel_jax_path=str(tmp_path / "missing_vessel.pkl"),
        split_indices_dir=str(split_dir),
        percent_train=0.8,
    )
    assert len(rerun) == 1
    assert rerun[0].split_path == split_path
    assert rerun[0].val_geometries == first[0].val_geometries
    assert os.path.getmtime(split_path) == mtime_before
