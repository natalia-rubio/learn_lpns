"""Tests for R_S_hist visualization."""

import csv

import numpy as np

from learn_lpns.visualizations.R_S_hist import (
    _collect_r_s_for_run_config,
    _read_csv_columns,
    _shared_bins,
)
from learn_lpns.zerod_calibration.run_config_canonical import resolve_run_config_suffix


def test_resolve_gen_loss_quadratic_resistor_suffix():
    assert resolve_run_config_suffix("gen_loss_quadratic_resistor") == "quadratic_resistor_gen_loss"


def test_read_csv_columns(tmp_path):
    csv_path = tmp_path / "junction_lumped_parameters.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "outlet_vessel_id",
                "R_poiseuille_outlet0",
                "R_poiseuille_outlet1",
                "stenosis_coefficient_outlet0",
                "stenosis_coefficient_outlet1",
            ]
        )
        writer.writerow([1, 10.0, 20.0, 0.1, 0.2])
        writer.writerow([2, 30.0, 40.0, 0.3, 0.4])
    r_vals = _read_csv_columns(
        str(csv_path),
        ("R_poiseuille_outlet0", "R_poiseuille_outlet1"),
    )
    s_vals = _read_csv_columns(
        str(csv_path),
        ("stenosis_coefficient_outlet0", "stenosis_coefficient_outlet1"),
    )
    assert r_vals == [10.0, 20.0, 30.0, 40.0]
    assert s_vals == [0.1, 0.2, 0.3, 0.4]


def test_collect_r_s_for_run_config(tmp_path):
    data_root = tmp_path
    run_config = "gen_loss"
    geometry_variant = "bifurcations_EL"
    geo_dir = data_root / "ml_inputs" / "VMR_test" / run_config / geometry_variant / "0001_0001"
    geo_dir.mkdir(parents=True)

    junction_csv = geo_dir / "junction_lumped_parameters.csv"
    with open(junction_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R_poiseuille_outlet0", "stenosis_coefficient_outlet0"])
        writer.writerow([5.0, 0.5])

    vessel_csv = geo_dir / "vessel_lumped_parameters.csv"
    with open(vessel_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R_poiseuille", "stenosis_coefficient"])
        writer.writerow([7.0, 0.7])

    r_vals, s_vals = _collect_r_s_for_run_config(
        data_root=str(data_root),
        set_name="VMR_test",
        run_config_suffix=run_config,
        geometry_variant=geometry_variant,
    )
    assert np.array_equal(r_vals, np.array([5.0, 7.0]))
    assert np.array_equal(s_vals, np.array([0.5, 0.7]))


def test_collect_r_s_skips_stenosis_when_disabled(tmp_path):
    data_root = tmp_path
    geo_dir = data_root / "ml_inputs" / "VMR_test" / "gen_loss" / "bifurcations_EL" / "0001_0001"
    geo_dir.mkdir(parents=True)
    junction_csv = geo_dir / "junction_lumped_parameters.csv"
    with open(junction_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R_poiseuille_outlet0", "stenosis_coefficient_outlet0"])
        writer.writerow([5.0, 0.5])

    r_vals, s_vals = _collect_r_s_for_run_config(
        data_root=str(data_root),
        set_name="VMR_test",
        run_config_suffix="gen_loss",
        geometry_variant="bifurcations_EL",
        include_stenosis=False,
    )
    assert np.array_equal(r_vals, np.array([5.0]))
    assert s_vals.size == 0


def test_shared_bins():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([1.0, 3.0])
    bins = _shared_bins(a, b, 10)
    assert bins[0] == 0.0
    assert bins[-1] == 3.0
    assert len(bins) == 11
