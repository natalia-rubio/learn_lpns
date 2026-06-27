from __future__ import annotations

import os

import pytest

from learn_lpns.neural_network.launch_training import (
    _default_split_path,
    _infer_num_geos,
)


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


def test_infer_num_geos_prefers_largest_with_matching_split(tmp_path):
    data_root = tmp_path / "data"
    set_name = "VMR_aorta_starter"
    geometry_variant = "bifurcations_EL"
    set_type = "all"
    run_config = "gen_loss"

    for num_geos in (3, 5):
        _touch(
            data_root
            / "jax_arrays"
            / set_name
            / run_config
            / geometry_variant
            / set_type
            / f"jax_arrays_num_geos_{num_geos}.pkl"
        )
        _touch(_default_split_path(str(data_root), set_name, geometry_variant, set_type, num_geos, run_config))

    inferred = _infer_num_geos(
        data_root=str(data_root),
        set_name=set_name,
        geometry_variant=geometry_variant,
        set_type=set_type,
        run_config_suffix=run_config,
        vessel=False,
        explicit_num_geos=None,
        split_path=None,
    )
    assert inferred == 5


def test_infer_num_geos_from_split_path(tmp_path):
    split_path = tmp_path / "train_val_ind_VMR_aorta_starter_num_geos_7_trial_0"
    _touch(str(split_path))

    inferred = _infer_num_geos(
        data_root=str(tmp_path),
        set_name="VMR_aorta_starter",
        geometry_variant="bifurcations_EL",
        set_type="all",
        run_config_suffix="gen_loss",
        vessel=False,
        explicit_num_geos=None,
        split_path=str(split_path),
    )
    assert inferred == 7


def test_infer_num_geos_requires_matching_split_when_not_explicit(tmp_path):
    data_root = tmp_path / "data"
    _touch(
        data_root / "jax_arrays" / "VMR_aorta_starter" / "gen_loss" / "bifurcations_EL" / "all" / "jax_arrays_num_geos_5.pkl"
    )

    with pytest.raises(SystemExit, match="no matching split_indices"):
        _infer_num_geos(
            data_root=str(data_root),
            set_name="VMR_aorta_starter",
            geometry_variant="bifurcations_EL",
            set_type="all",
            run_config_suffix="gen_loss",
            vessel=False,
            explicit_num_geos=None,
            split_path=None,
        )
