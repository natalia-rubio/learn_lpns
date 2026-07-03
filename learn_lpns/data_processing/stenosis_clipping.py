"""Train-only, generation-filtered stenosis (S) clipping for jax cohorts and inference."""

from __future__ import annotations

import copy
import csv
import json
import os
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from learn_lpns.data_processing.generate_split_indices import resolve_flat_indices
from learn_lpns.tools.basic import save_dict

S_OUTPUT_COLUMN = 1
OUTPUT_RRI_KEY = "output_rri"


def compute_stenosis_clip_bounds(
    output_rri: np.ndarray,
    generation: np.ndarray,
    train_row_indices: np.ndarray | list[int],
    clip_generation_number: int,
) -> tuple[float, float]:
    """Return (s_min, s_max) from train rows with generation <= clip_generation_number."""
    train_inds = np.asarray(train_row_indices, dtype=int)
    if train_inds.size == 0:
        raise ValueError("Cannot compute stenosis bounds: no training row indices.")
    out = np.asarray(output_rri, dtype=float)
    gen = np.asarray(generation, dtype=float).reshape(-1)
    if out.shape[0] != gen.shape[0]:
        raise ValueError(
            f"output_rri rows ({out.shape[0]}) must match generation rows ({gen.shape[0]})"
        )
    if out.shape[1] <= S_OUTPUT_COLUMN:
        raise ValueError(f"output_rri must have at least {S_OUTPUT_COLUMN + 1} columns")

    train_s = out[train_inds, S_OUTPUT_COLUMN]
    train_gen = gen[train_inds]
    mask = train_gen <= float(clip_generation_number)
    if not np.any(mask):
        raise ValueError(
            f"No train rows with generation <= {clip_generation_number} "
            f"(train generations: min={train_gen.min():.4g}, max={train_gen.max():.4g})"
        )
    filtered = train_s[mask]
    return float(np.min(filtered)), float(np.max(filtered))


def clip_stenosis_values(
    values: np.ndarray,
    s_min: float,
    s_max: float,
) -> np.ndarray:
    """Clip 1D stenosis predictions to [s_min, s_max]."""
    return np.clip(np.asarray(values, dtype=float), s_min, s_max)


def clip_stenosis_in_data_dict(
    data_dict: dict[str, Any],
    s_min: float,
    s_max: float,
) -> dict[str, Any]:
    """Return a copy of data_dict with output_rri[:, 1] clipped; R/L unchanged."""
    out = copy.deepcopy(data_dict)
    rri = np.asarray(out[OUTPUT_RRI_KEY], dtype=float).copy()
    rri[:, S_OUTPUT_COLUMN] = np.clip(rri[:, S_OUTPUT_COLUMN], s_min, s_max)
    out[OUTPUT_RRI_KEY] = jnp.asarray(rri)
    return out


def build_stenosis_clip_bounds_metadata(
    *,
    clip_generation_number: int,
    train_geometries: list[str],
    junction_bounds: tuple[float, float],
    vessel_bounds: tuple[float, float],
) -> dict[str, Any]:
    """Build metadata dict stored on clipped jax pickles."""
    return {
        "clip_generation_number": int(clip_generation_number),
        "train_geometries": list(train_geometries),
        "junction": {"s_min": float(junction_bounds[0]), "s_max": float(junction_bounds[1])},
        "vessel": {"s_min": float(vessel_bounds[0]), "s_max": float(vessel_bounds[1])},
    }


def compute_modality_stenosis_bounds(
    data_dict: dict[str, Any],
    split_dict: dict[str, Any],
    modality: Literal["junction", "vessel"],
    clip_generation_number: int,
    *,
    split_path: str | None = None,
) -> tuple[float, float]:
    """Compute S bounds for one modality from train rows in split_dict."""
    train_inds = resolve_flat_indices(split_dict, modality, "train", split_path=split_path)
    return compute_stenosis_clip_bounds(
        np.asarray(data_dict[OUTPUT_RRI_KEY]),
        np.asarray(data_dict["generation"]),
        train_inds,
        clip_generation_number,
    )


def write_clipped_jax_pickles(
    *,
    junction_unclipped: dict[str, Any],
    vessel_unclipped: dict[str, Any],
    split_dict: dict[str, Any],
    clip_generation_number: int,
    junction_out_path: str,
    vessel_out_path: str,
    split_path: str | None = None,
) -> dict[str, Any]:
    """Clip junction and vessel dicts, attach metadata, and save clipped pickles."""
    junction_bounds = compute_modality_stenosis_bounds(
        junction_unclipped,
        split_dict,
        "junction",
        clip_generation_number,
        split_path=split_path,
    )
    vessel_bounds = compute_modality_stenosis_bounds(
        vessel_unclipped,
        split_dict,
        "vessel",
        clip_generation_number,
        split_path=split_path,
    )
    metadata = build_stenosis_clip_bounds_metadata(
        clip_generation_number=clip_generation_number,
        train_geometries=list(split_dict.get("train_geometries", [])),
        junction_bounds=junction_bounds,
        vessel_bounds=vessel_bounds,
    )

    junction_clipped = clip_stenosis_in_data_dict(junction_unclipped, *junction_bounds)
    vessel_clipped = clip_stenosis_in_data_dict(vessel_unclipped, *vessel_bounds)
    junction_clipped["stenosis_clip_bounds"] = metadata
    vessel_clipped["stenosis_clip_bounds"] = metadata

    os.makedirs(os.path.dirname(junction_out_path), exist_ok=True)
    save_dict(junction_clipped, junction_out_path)
    save_dict(vessel_clipped, vessel_out_path)
    print(
        f"  Stenosis clip bounds: junction S [{junction_bounds[0]:.4g}, {junction_bounds[1]:.4g}], "
        f"vessel S [{vessel_bounds[0]:.4g}, {vessel_bounds[1]:.4g}] "
        f"(gen <= {clip_generation_number}, train geos={len(metadata['train_geometries'])})"
    )
    print(f"  Wrote clipped junction jax to {junction_out_path}")
    print(f"  Wrote clipped vessel jax to {vessel_out_path}")
    return metadata


def _ml_inputs_geo_dir(
    data_root: str,
    set_name: str,
    run_config_suffix: str | None,
    geometry_variant: str,
    geo: str,
) -> str:
    parts = [data_root, "ml_inputs", set_name]
    if run_config_suffix:
        parts.append(run_config_suffix)
    parts.extend([geometry_variant, geo])
    return os.path.join(*parts)


def _clipped_labels_root(
    data_root: str,
    set_name: str,
    run_config_suffix: str | None,
    geometry_variant: str,
    label_variant: str,
) -> str:
    """Return ``.../clipped_labels/{default|trial_T}/`` directory."""
    parts = [data_root, "ml_inputs", set_name]
    if run_config_suffix:
        parts.append(run_config_suffix)
    parts.extend([geometry_variant, "clipped_labels", label_variant])
    return os.path.join(*parts)


def _clip_stenosis_columns_in_csv_row(
    row: list[str],
    header: list[str],
    s_min: float,
    s_max: float,
    *,
    junction: bool,
    primary_s_value: float | None = None,
) -> list[str]:
    """Clip stenosis columns in a CSV row for inspection export."""
    out = list(row)
    for col_idx, name in enumerate(header):
        if junction:
            if not name.startswith("stenosis_coefficient_outlet"):
                continue
            if name == "stenosis_coefficient_outlet0" and primary_s_value is not None:
                out[col_idx] = f"{primary_s_value:.12g}"
            else:
                out[col_idx] = f"{np.clip(float(row[col_idx]), s_min, s_max):.12g}"
        elif name == "stenosis_coefficient":
            if primary_s_value is not None:
                out[col_idx] = f"{primary_s_value:.12g}"
            else:
                out[col_idx] = f"{np.clip(float(row[col_idx]), s_min, s_max):.12g}"
    return out


def export_clipped_lumped_parameter_csvs(
    *,
    junction_clipped: dict[str, Any],
    vessel_clipped: dict[str, Any],
    metadata: dict[str, Any],
    data_root: str,
    set_name: str,
    run_config_suffix: str | None,
    geometry_variant: str,
    label_variant: str,
) -> str:
    """Export per-geo clipped junction/vessel CSVs under clipped_labels/{variant}/."""
    out_root = _clipped_labels_root(data_root, set_name, run_config_suffix, geometry_variant, label_variant)
    os.makedirs(out_root, exist_ok=True)

    j_bounds = metadata["junction"]
    v_bounds = metadata["vessel"]
    j_s_min, j_s_max = j_bounds["s_min"], j_bounds["s_max"]
    v_s_min, v_s_max = v_bounds["s_min"], v_bounds["s_max"]

    geometries = list(junction_clipped.get("geometry_names_order", []))
    j_ranges = junction_clipped.get("geometry_row_ranges", [])
    v_ranges = vessel_clipped.get("geometry_row_ranges", [])
    j_rri = np.asarray(junction_clipped[OUTPUT_RRI_KEY])
    v_rri = np.asarray(vessel_clipped[OUTPUT_RRI_KEY])

    for geo_idx, geo in enumerate(geometries):
        src_dir = _ml_inputs_geo_dir(data_root, set_name, run_config_suffix, geometry_variant, geo)
        dst_dir = os.path.join(out_root, geo)
        os.makedirs(dst_dir, exist_ok=True)

        j_src = os.path.join(src_dir, "junction_lumped_parameters.csv")
        j_dst = os.path.join(dst_dir, "junction_lumped_parameters.csv")
        if os.path.isfile(j_src):
            j_start, j_end = j_ranges[geo_idx]
            with open(j_src) as fin, open(j_dst, "w", newline="") as fout:
                reader = csv.reader(fin)
                header = next(reader)
                writer = csv.writer(fout)
                writer.writerow(header)
                for local_row_idx, row in enumerate(reader):
                    global_row = j_start + local_row_idx
                    primary_s = float(j_rri[global_row, S_OUTPUT_COLUMN]) if global_row < j_end else None
                    writer.writerow(
                        _clip_stenosis_columns_in_csv_row(
                            row,
                            header,
                            j_s_min,
                            j_s_max,
                            junction=True,
                            primary_s_value=primary_s,
                        )
                    )

        v_src = os.path.join(src_dir, "vessel_lumped_parameters.csv")
        v_dst = os.path.join(dst_dir, "vessel_lumped_parameters.csv")
        if os.path.isfile(v_src):
            v_start, v_end = v_ranges[geo_idx]
            with open(v_src) as fin, open(v_dst, "w", newline="") as fout:
                reader = csv.reader(fin)
                header = next(reader)
                writer = csv.writer(fout)
                writer.writerow(header)
                for local_row_idx, row in enumerate(reader):
                    global_row = v_start + local_row_idx
                    primary_s = float(v_rri[global_row, S_OUTPUT_COLUMN]) if global_row < v_end else None
                    writer.writerow(
                        _clip_stenosis_columns_in_csv_row(
                            row,
                            header,
                            v_s_min,
                            v_s_max,
                            junction=False,
                            primary_s_value=primary_s,
                        )
                    )

    bounds_path = os.path.join(out_root, "bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Exported clipped_labels to {out_root} (bounds.json)")
    return out_root


def modality_stenosis_bounds_from_metadata(
    metadata: dict[str, Any],
    *,
    vessel: bool,
) -> tuple[float, float]:
    """Return (s_min, s_max) for junction or vessel from stenosis_clip_bounds metadata."""
    key = "vessel" if vessel else "junction"
    block = metadata[key]
    return float(block["s_min"]), float(block["s_max"])


def attach_stenosis_bounds_to_model(
    model,
    metadata: dict[str, Any],
    *,
    vessel: bool,
) -> None:
    """Store modality-specific stenosis clip bounds on a model checkpoint object."""
    s_min, s_max = modality_stenosis_bounds_from_metadata(metadata, vessel=vessel)
    model.stenosis_clip_bounds = {
        "s_min": s_min,
        "s_max": s_max,
        "clip_generation_number": int(metadata.get("clip_generation_number", 0)),
        "train_geometries": list(metadata.get("train_geometries", [])),
    }


def resolve_stenosis_bounds_from_model(model, *, vessel: bool) -> tuple[float, float]:
    """Return (s_min, s_max) from checkpoint; raise if missing."""
    bounds = getattr(model, "stenosis_clip_bounds", None)
    if bounds is None:
        raise ValueError(
            "stenosis_clipping is enabled but the loaded model has no stenosis_clip_bounds. "
            "Retrain with stenosis clipping enabled or disable stenosis_clipping in config."
        )
    return float(bounds["s_min"]), float(bounds["s_max"])


__all__ = [
    "S_OUTPUT_COLUMN",
    "attach_stenosis_bounds_to_model",
    "build_stenosis_clip_bounds_metadata",
    "clip_stenosis_in_data_dict",
    "clip_stenosis_values",
    "compute_modality_stenosis_bounds",
    "compute_stenosis_clip_bounds",
    "export_clipped_lumped_parameter_csvs",
    "modality_stenosis_bounds_from_metadata",
    "resolve_stenosis_bounds_from_model",
    "write_clipped_jax_pickles",
]
