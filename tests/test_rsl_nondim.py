"""Tests for physics-based R/S/L non-dimensionalization."""

import numpy as np
import pytest

from learn_lpns.data_processing.data_dict_from_csvs import (
    _finalize_stacked_ml_data,
    get_default_include_features,
)
from learn_lpns.data_processing.rsl_nondim import (
    characteristic_scales,
    nondimensionalize_rsl,
    redimensionalize_rsl,
    redimensionalize_rsl_predictions,
    rsl_scale_factors,
)
from learn_lpns.neural_network.nn_util import attach_nondim_metadata_from_data_dict


RHO = 1.06
MU = 0.04
RE_C = 4500.0


def test_rsl_scale_factors_match_characteristic_scales():
    lc = 0.25
    q_c, t_c, p_c, _ = characteristic_scales(lc, rho=RHO, mu=MU, reference_reynolds=RE_C)
    f_r, f_s, f_l = rsl_scale_factors(lc, rho=RHO, mu=MU, reference_reynolds=RE_C)
    assert f_r == pytest.approx(q_c / p_c)
    assert f_s == pytest.approx((q_c**2) / p_c)
    assert f_l == pytest.approx(q_c / (t_c * p_c))


def test_nondimensionalize_redimensionalize_round_trip():
    lc = np.array([0.2, 0.35])
    physical = np.array([[12.0, 0.5, 0.2], [30.0, 1.5, 0.8]])
    nondim = nondimensionalize_rsl(physical, lc, rho=RHO, mu=MU, reference_reynolds=RE_C)
    restored = redimensionalize_rsl(nondim, lc, rho=RHO, mu=MU, reference_reynolds=RE_C)
    np.testing.assert_allclose(restored, physical, rtol=1e-12, atol=1e-12)


def test_redimensionalize_rsl_predictions_from_features():
    feature_names = ["inlet_max_inscribed_radius", "other"]
    X = np.array([[0.2, 1.0], [0.3, 2.0]])
    physical = np.array([[5.0, 0.1, 0.05], [8.0, 0.2, 0.1]])
    nondim = nondimensionalize_rsl(physical, X[:, 0], rho=RHO, mu=MU, reference_reynolds=RE_C)
    pred_r, pred_s, pred_l = redimensionalize_rsl_predictions(
        nondim[:, 0],
        nondim[:, 1],
        nondim[:, 2],
        X,
        feature_names,
        rho=RHO,
        mu=MU,
        reference_reynolds=RE_C,
    )
    np.testing.assert_allclose(pred_r, physical[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pred_s, physical[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pred_l, physical[:, 2], rtol=1e-12, atol=1e-12)


def test_characteristic_scales_rejects_nonpositive_radius():
    with pytest.raises(ValueError, match="positive"):
        characteristic_scales(0.0, rho=RHO, mu=MU, reference_reynolds=RE_C)


def test_finalize_stacked_ml_data_nondimensionalize_metadata_and_round_trip():
    feature_order = get_default_include_features()
    lc_idx = feature_order.index("inlet_max_inscribed_radius")
    n_rows = 2
    input_array = np.zeros((n_rows, len(feature_order)))
    input_array[:, lc_idx] = [0.2, 0.35]
    physical = np.array([[10.0, 1.0, 0.5], [20.0, 2.0, 1.0]])

    result = _finalize_stacked_ml_data(
        input_array=input_array,
        output_array=physical.copy(),
        generation_array=np.zeros(n_rows),
        feature_order=feature_order,
        output_order=["R_poiseuille_outlet0", "stenosis_coefficient_outlet0", "L_outlet0"],
        geometries=["geo_a"],
        geometry_row_ranges=[(0, n_rows)],
        row_geo_names=["geo_a"] * n_rows,
        row_instance_names=["j0", "j1"],
        set_name="test_set",
        geometry_variant="bifurcations",
        input_nan_msg="input nan",
        output_nan_msg="output nan",
        summary_filename="summary.csv",
        plot_histograms=False,
        nondimensionalize=True,
        rho=RHO,
        mu=MU,
        reference_reynolds=RE_C,
    )

    assert result["nondim_rsl"] is True
    assert result["reference_reynolds"] == pytest.approx(RE_C)
    assert result["nondim_rho"] == pytest.approx(RHO)
    assert result["nondim_mu"] == pytest.approx(MU)

    stored = np.asarray(result["output_rri"])
    restored = redimensionalize_rsl(
        stored,
        input_array[:, lc_idx],
        rho=RHO,
        mu=MU,
        reference_reynolds=RE_C,
    )
    np.testing.assert_allclose(restored, physical, rtol=1e-6, atol=1e-6)


def test_attach_nondim_metadata_from_data_dict():
    class _Model:
        pass

    model = _Model()
    attach_nondim_metadata_from_data_dict(
        model,
        {
            "nondim_rsl": True,
            "reference_reynolds": RE_C,
            "nondim_rho": RHO,
            "nondim_mu": MU,
        },
    )
    assert model.nondim_rsl is True
    assert model.reference_reynolds == RE_C
    assert model.nondim_rho == RHO
    assert model.nondim_mu == MU

    model2 = _Model()
    attach_nondim_metadata_from_data_dict(model2, {})
    assert model2.nondim_rsl is False
