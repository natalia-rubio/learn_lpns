"""Shared paths for forward-sim CSV/JSON outputs keyed by plot/MSE modality name."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

# Filename suffixes for NN forward-sim configs (paired with JunctionAndVessel / VesselOnly).
NN_JUNCTION_ONLY_SUFFIX = "JunctionOnly"
NN_JUNCTION_AND_VESSEL_SUFFIX = "JunctionAndVessel"
NN_VESSEL_ONLY_SUFFIX = "VesselOnly"

# Human-readable column headers for MSE summary tables and LaTeX exports.
MODALITY_DISPLAY: dict[str, str] = {
    "geometric": "Standard",
    "BloodVesselJunction": "Calibrated",
    "BloodVesselJunction_NN": "Learned Junctions",
    "BloodVesselJunction_NN_plus_Vessel_NN": "Learned Junctions and Vessels",
    "Vessel_NN": "Learned Vessel",
}

DEFAULT_MODALITY_ORDER: tuple[str, ...] = (
    "geometric",
    "BloodVesselJunction",
    "BloodVesselJunction_NN",
    "BloodVesselJunction_NN_plus_Vessel_NN",
    "Vessel_NN",
)

__all__ = [
    "DEFAULT_MODALITY_ORDER",
    "MODALITY_DISPLAY",
    "NN_JUNCTION_AND_VESSEL_SUFFIX",
    "NN_JUNCTION_ONLY_SUFFIX",
    "NN_VESSEL_ONLY_SUFFIX",
    "modality_csv_paths",
    "modality_json_paths",
    "modality_key_from_table_header",
    "modality_table_header",
    "nn_forward_sim_specs",
    "read_cv_metric_from_row",
    "sort_modalities",
    "split_location_plot_csv_paths",
]


def modality_table_header(modality_key: str) -> str:
    """Return display label for a modality key (falls back to the key itself)."""
    return MODALITY_DISPLAY.get(modality_key, modality_key)


_DISPLAY_TO_MODALITY = {label: key for key, label in MODALITY_DISPLAY.items()}


def modality_key_from_table_header(label: str) -> str:
    """Map a human-readable table/CSV header back to the internal modality key."""
    if label in MODALITY_DISPLAY:
        return label
    return _DISPLAY_TO_MODALITY.get(label, label)


def read_cv_metric_from_row(row: Mapping[str, Any], prefix: str, modality_key: str) -> Any:
    """
    Read one metric cell from a CV summary CSV row.

    Older CSVs used display labels in column headers (e.g. PressureMaxError_Calibrated);
    newer rows use internal keys (e.g. PressureMaxError_BloodVesselJunction).
    """
    key_col = f"{prefix}{modality_key}"
    display_col = f"{prefix}{modality_table_header(modality_key)}"
    for col in (key_col, display_col):
        val = row.get(col)
        if val is not None and str(val).strip() not in ("", "N/A"):
            return val
    return ""


def sort_modalities(modality_keys: Sequence[str]) -> list[str]:
    """Sort modality keys in DEFAULT_MODALITY_ORDER; unknown keys trail alphabetically."""
    order = {k: i for i, k in enumerate(DEFAULT_MODALITY_ORDER)}
    return sorted(modality_keys, key=lambda k: (order.get(k, len(order)), k))


def nn_forward_sim_specs(
    base_dir: str,
    geo_variant_name: str,
    junction_type: str,
    nn_vessel: bool,
) -> list[tuple[str, str, str]]:
    """Return (modality_key, sim_input_json, results_csv) for each NN forward sim."""
    specs = [
        (
            "BloodVesselJunction_NN",
            os.path.join(base_dir, f"{geo_variant_name}_NN_{NN_JUNCTION_ONLY_SUFFIX}.json"),
            os.path.join(base_dir, f"{geo_variant_name}_NN_{NN_JUNCTION_ONLY_SUFFIX}_results.csv"),
        ),
    ]
    if nn_vessel:
        specs.extend(
            [
                (
                    "BloodVesselJunction_NN_plus_Vessel_NN",
                    os.path.join(
                        base_dir,
                        f"{geo_variant_name}_NN_{NN_JUNCTION_AND_VESSEL_SUFFIX}.json",
                    ),
                    os.path.join(
                        base_dir,
                        f"{geo_variant_name}_NN_{NN_JUNCTION_AND_VESSEL_SUFFIX}_results.csv",
                    ),
                ),
                (
                    "Vessel_NN",
                    os.path.join(base_dir, f"{geo_variant_name}_NN_{NN_VESSEL_ONLY_SUFFIX}.json"),
                    os.path.join(base_dir, f"{geo_variant_name}_NN_{NN_VESSEL_ONLY_SUFFIX}_results.csv"),
                ),
            ]
        )
    return specs


def modality_csv_paths(
    geo_variant_paths: Mapping[str, Any],
    base_dir: str,
    geo_variant_name: str,
    junction_type: str,
    nn_vessel: bool,
    extra_junction_types: Sequence[str] = (),
) -> dict[str, str]:
    """Modality name -> forward results CSV (for MSE and location comparison plots)."""
    results: dict[str, str] = {}
    geometric_results = geo_variant_paths["geometric_results"]
    if os.path.exists(geometric_results):
        results["geometric"] = str(geometric_results)
    calibrated_results = geo_variant_paths["junction_types"][junction_type]["calibrated_results"]
    if os.path.exists(calibrated_results):
        results[junction_type] = str(calibrated_results)
    for extra_jtype in extra_junction_types:
        if extra_jtype == junction_type:
            continue
        extra_results = geo_variant_paths["junction_types"][extra_jtype]["calibrated_results"]
        if os.path.exists(extra_results):
            results[extra_jtype] = str(extra_results)
    if geo_variant_name in ("bifurcations", "bifurcations_EL"):
        for modality_key, _, results_csv in nn_forward_sim_specs(
            base_dir,
            geo_variant_name,
            junction_type,
            nn_vessel,
        ):
            if os.path.exists(results_csv):
                results[modality_key] = str(results_csv)
    return results


def modality_json_paths(
    geo_variant_paths: Mapping[str, Any],
    base_dir: str,
    geo_variant_name: str,
    junction_type: str,
    nn_vessel: bool,
) -> dict[str, str]:
    """Modality name -> calibrated/geometric JSON (for zero-D parameter bar charts)."""
    modality_jsons: dict[str, str] = {}
    geom_json = geo_variant_paths.get("geometric_input")
    if geom_json and os.path.exists(geom_json):
        modality_jsons["geometric"] = str(geom_json)
    calib_json = geo_variant_paths["junction_types"][junction_type]["calibrated_output"]
    if calib_json and os.path.exists(calib_json):
        modality_jsons[junction_type] = str(calib_json)
    if geo_variant_name in ("bifurcations", "bifurcations_EL"):
        for modality_key, sim_input, _ in nn_forward_sim_specs(
            base_dir,
            geo_variant_name,
            junction_type,
            nn_vessel,
        ):
            if os.path.exists(sim_input):
                modality_jsons[modality_key] = str(sim_input)
    return modality_jsons


def split_location_plot_csv_paths(
    all_csv_paths: Mapping[str, str],
    geo_variant_name: str,
    geo_variant_paths: Mapping[str, Any],
) -> tuple[str | None, dict[str, str], dict[str, str]]:
    """Split modality CSV dict into plot_location_comparison arguments."""
    geometric_csv_path = geo_variant_paths["geometric_results"]
    if not os.path.exists(geometric_csv_path):
        geometric_csv_path = None
    calibrated_csv_paths = {k: v for k, v in all_csv_paths.items() if k != "geometric"}
    geometric_csv_paths: dict[str, str] = {}
    if "geometric" in all_csv_paths:
        geometric_csv_paths[geo_variant_name] = all_csv_paths["geometric"]
    return geometric_csv_path, calibrated_csv_paths, geometric_csv_paths
