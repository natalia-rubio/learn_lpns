"""Tests for modality display names and forward-simulation path resolution."""

from util.zerod_calibration.modality_paths import (
    modality_csv_paths,
    modality_key_from_table_header,
    modality_table_header,
    nn_forward_sim_specs,
    read_cv_metric_from_row,
    sort_modalities,
)


def test_modality_table_header_known_and_unknown():
    assert modality_table_header("geometric") == "Standard"
    assert modality_table_header("custom_key") == "custom_key"


def test_modality_key_from_table_header_roundtrip():
    assert modality_key_from_table_header("Calibrated") == "BloodVesselJunction"
    assert modality_key_from_table_header("BloodVesselJunction") == "BloodVesselJunction"


def test_sort_modalities_default_order():
    unordered = ["NN_vessel", "geometric", "BloodVesselJunction_NN"]
    assert sort_modalities(unordered) == [
        "geometric",
        "BloodVesselJunction_NN",
        "NN_vessel",
    ]


def test_read_cv_metric_prefers_internal_key_then_display_label():
    row = {
        "PressureMaxError_BloodVesselJunction": "1.2",
        "PressureMaxError_Calibrated": "9.9",
    }
    assert read_cv_metric_from_row(row, "PressureMaxError_", "BloodVesselJunction") == "1.2"

    row_display_only = {"PressureMaxError_Calibrated": "3.4"}
    assert read_cv_metric_from_row(row_display_only, "PressureMaxError_", "BloodVesselJunction") == "3.4"


def test_nn_forward_sim_specs_without_vessel():
    specs = nn_forward_sim_specs("/out", "bifurcations_EL", "BloodVesselJunction", nn_vessel=False)
    assert len(specs) == 1
    assert specs[0][0] == "BloodVesselJunction_NN"
    assert specs[0][1].endswith("bifurcations_EL_NN_BloodVesselJunction.json")


def test_nn_forward_sim_specs_with_vessel():
    specs = nn_forward_sim_specs("/out", "bifurcations_EL", "BloodVesselJunction", nn_vessel=True)
    keys = [s[0] for s in specs]
    assert keys == [
        "BloodVesselJunction_NN",
        "BloodVesselJunction_NN_plus_Vessel_NN",
        "NN_vessel",
    ]


def test_modality_csv_paths_includes_existing_files(tmp_path):
    base_dir = tmp_path / "geo"
    base_dir.mkdir()
    geo_variant_name = "bifurcations_EL"
    junction_type = "BloodVesselJunction"

    geometric_csv = base_dir / "geometric_results.csv"
    geometric_csv.write_text("t,p\n")
    calibrated_csv = base_dir / f"{geo_variant_name}_calibrated_results.csv"
    calibrated_csv.write_text("t,p\n")
    nn_csv = base_dir / f"{geo_variant_name}_NN_{junction_type}_results.csv"
    nn_csv.write_text("t,p\n")

    geo_variant_paths = {
        "geometric_results": str(geometric_csv),
        "junction_types": {
            junction_type: {"calibrated_results": str(calibrated_csv)},
        },
    }

    paths = modality_csv_paths(
        geo_variant_paths,
        str(base_dir),
        geo_variant_name,
        junction_type,
        nn_vessel=False,
    )

    assert paths["geometric"] == str(geometric_csv)
    assert paths[junction_type] == str(calibrated_csv)
    assert paths["BloodVesselJunction_NN"] == str(nn_csv)
    assert "NN_vessel" not in paths
