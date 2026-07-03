#!/usr/bin/env python3
"""
Generate geometry-based train/validation splits for JAX-array datasets.

Split pickles store train/val geometry lists and per-geometry half-open row ranges
in junction and vessel stacked arrays. Use ``resolve_flat_indices()`` at training time.
"""

import argparse
import glob
import os
from typing import Any, Literal

import numpy as np

from learn_lpns.config import get_pipeline_config
from learn_lpns.data_processing.jax_arrays_paths import resolve_jax_arrays_path
from learn_lpns.tools.basic import load_dict, save_dict


def _ml_inputs_dir(
    ml_inputs_root: str,
    set_name: str,
    geometry_variant: str,
    run_config_suffix: str | None = None,
) -> str:
    if run_config_suffix:
        return os.path.join(ml_inputs_root, set_name, run_config_suffix, geometry_variant)
    return os.path.join(ml_inputs_root, set_name, geometry_variant)


def list_ml_input_geometries(
    ml_inputs_root: str,
    set_name: str,
    geometry_variant: str,
    run_config_suffix: str | None = None,
) -> list[str]:
    """Return sorted geometry folder names under ml_inputs that have geometric_features.csv."""
    ml_inputs_dir = _ml_inputs_dir(ml_inputs_root, set_name, geometry_variant, run_config_suffix)
    if not os.path.exists(ml_inputs_dir):
        raise FileNotFoundError(f"ML inputs dir not found: {ml_inputs_dir}")
    geometries = []
    for geo_dir in glob.glob(os.path.join(ml_inputs_dir, "*")):
        if not os.path.isdir(geo_dir):
            continue
        geo_name = os.path.basename(geo_dir)
        if os.path.exists(os.path.join(geo_dir, "geometric_features.csv")):
            geometries.append(geo_name)
    return sorted(geometries)


def get_geometry_row_ranges(
    ml_inputs_root: str,
    set_name: str,
    geometry_variant: str,
    geometries: list[str] | None = None,
    run_config_suffix: str | None = None,
) -> tuple[list[tuple[int, int]], int, list[str]]:
    """
    Return (row_ranges, total_rows, geometries) for each geometry in order.
    row_ranges[i] = (start, end) so geometry i has row indices [start, end).
    Geometries are in sorted order if discovered from disk.
    When run_config_suffix is set, ml_inputs path is .../set_name/run_config_suffix/geometry_variant/...
    """
    ml_inputs_dir = _ml_inputs_dir(ml_inputs_root, set_name, geometry_variant, run_config_suffix)
    geom_csv_template = os.path.join(ml_inputs_dir, "%s", "geometric_features.csv")
    if geometries is None:
        geometries = list_ml_input_geometries(ml_inputs_root, set_name, geometry_variant, run_config_suffix)

    row_ranges = []
    start = 0
    for geo in geometries:
        geom_csv = geom_csv_template % geo
        if not os.path.exists(geom_csv):
            raise FileNotFoundError(f"Expected CSV: {geom_csv}")
        with open(geom_csv) as f:
            n_rows = sum(1 for _ in f) - 1  # exclude header
        if n_rows < 0:
            n_rows = 0
        row_ranges.append((start, start + n_rows))
        start += n_rows
    return row_ranges, start, geometries


def resolve_geometry_row_ranges_from_jax_dict(
    data_dict: dict[str, Any],
) -> tuple[list[tuple[int, int]], int, list[str]]:
    """
    Return ``geometry_row_ranges`` and ``geometry_names_order`` stored in the junction jax pickle.

    Raises if metadata is missing or does not match the stacked ``input`` array.
    Re-run data processing to rebuild pickles written by the current pipeline.
    """
    inp = data_dict.get("input")
    if inp is None:
        raise ValueError("Jax pickle missing 'input' array.")
    num_pts = int(np.asarray(inp).shape[0])

    stored_ranges = data_dict.get("geometry_row_ranges")
    stored_geoms = data_dict.get("geometry_names_order")
    if not isinstance(stored_ranges, list) or not stored_ranges:
        raise ValueError("Jax pickle missing geometry_row_ranges. Re-run data processing to rebuild the pickle.")
    if not isinstance(stored_geoms, list) or not stored_geoms:
        raise ValueError("Jax pickle missing geometry_names_order. Re-run data processing to rebuild the pickle.")
    if len(stored_ranges) != len(stored_geoms):
        raise ValueError(
            f"Jax pickle geometry_row_ranges length ({len(stored_ranges)}) "
            f"does not match geometry_names_order ({len(stored_geoms)})."
        )

    row_ranges = [(int(s), int(e)) for s, e in stored_ranges]
    geometries = [str(g) for g in stored_geoms]
    total = sum(e - s for s, e in row_ranges)
    if total != num_pts:
        raise ValueError(
            f"Jax pickle row count mismatch: geometry_row_ranges sum to {total} "
            f"but input has {num_pts} rows. Delete the pickle and re-run data processing."
        )
    return row_ranges, total, geometries


def generate_split_indices(
    num_pts: int,
    percent_train: float,
    seed: int = 0,
    geometry_row_ranges: list[tuple[int, int]] | None = None,
):
    if not (0.0 < percent_train <= 1.0):
        raise ValueError(f"percent_train must be in (0, 1], got {percent_train}")
    if num_pts <= 0:
        raise ValueError(f"num_pts must be > 0, got {num_pts}")

    if geometry_row_ranges is not None:
        total_from_ranges = sum(end - start for start, end in geometry_row_ranges)
        if total_from_ranges != num_pts:
            raise ValueError(f"Geometry row ranges sum to {total_from_ranges} but num_pts={num_pts}")
        num_geos = len(geometry_row_ranges)
        rng = np.random.default_rng(seed)
        geo_order = rng.permutation(num_geos)
        n_train_geos = max(1, int(percent_train * num_geos))
        train_geo_idx_list = sorted(geo_order[:n_train_geos].tolist())
        val_geo_idx_list = sorted(geo_order[n_train_geos:].tolist())
        train_geo_indices = set(train_geo_idx_list)
        train_indices = []
        val_indices = []
        for g, (s, e) in enumerate(geometry_row_ranges):
            if g in train_geo_indices:
                train_indices.extend(range(s, e))
            else:
                val_indices.extend(range(s, e))
        return (
            np.array(train_indices, dtype=int),
            np.array(val_indices, dtype=int),
            train_geo_idx_list,
            val_geo_idx_list,
        )

    # Random split (legacy: point-level)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_pts)
    n_train = int(percent_train * num_pts)
    if n_train <= 0:
        raise ValueError(f"Invalid split: num_pts={num_pts}, percent_train={percent_train} -> n_train={n_train}")
    if n_train > num_pts:
        raise ValueError(f"Invalid split: num_pts={num_pts}, percent_train={percent_train} -> n_train={n_train}")
    train_indices = indices[:n_train]
    val_indices = indices[n_train:] if n_train < num_pts else np.array([], dtype=int)
    return train_indices, val_indices, None, None


Modality = Literal["junction", "vessel"]
SplitName = Literal["train", "val"]
RangeTuple = tuple[int, int]


def build_geometry_index_map(
    junction_dict: dict[str, Any],
    vessel_dict: dict[str, Any],
) -> dict[str, dict[str, RangeTuple]]:
    """Map each geometry name to junction and/or vessel row ranges in stacked jax arrays."""
    geometry_indices: dict[str, dict[str, RangeTuple]] = {}
    j_ranges = junction_dict.get("geometry_row_ranges") or []
    j_geos = junction_dict.get("geometry_names_order") or []
    for idx, geo in enumerate(j_geos):
        if idx < len(j_ranges):
            s, e = j_ranges[idx]
            geometry_indices.setdefault(str(geo), {})["junction"] = (int(s), int(e))
    v_ranges = vessel_dict.get("geometry_row_ranges") or vessel_dict.get("row_ranges") or []
    v_geos = vessel_dict.get("geometry_names_order") or vessel_dict.get("geometries") or []
    for idx, geo in enumerate(v_geos):
        if idx < len(v_ranges):
            s, e = v_ranges[idx]
            geometry_indices.setdefault(str(geo), {})["vessel"] = (int(s), int(e))
    return geometry_indices


def generate_geometry_split(
    percent_train: float,
    seed: int,
    geometry_names: list[str],
) -> tuple[list[str], list[str]]:
    """Randomly assign whole geometries to train or validation."""
    if not (0.0 < percent_train <= 1.0):
        raise ValueError(f"percent_train must be in (0, 1], got {percent_train}")
    if not geometry_names:
        raise ValueError("geometry_names must be non-empty")
    num_geos = len(geometry_names)
    rng = np.random.default_rng(seed)
    geo_order = rng.permutation(num_geos)
    n_train_geos = max(1, int(percent_train * num_geos))
    train_geometries = [geometry_names[i] for i in sorted(geo_order[:n_train_geos].tolist())]
    val_geometries = [geometry_names[i] for i in sorted(geo_order[n_train_geos:].tolist())]
    return train_geometries, val_geometries


def build_split_dict(
    train_geometries: list[str],
    val_geometries: list[str],
    geometry_indices: dict[str, dict[str, RangeTuple]],
    *,
    num_offsets: int = 1,
    **meta: Any,
) -> dict[str, Any]:
    """Assemble the split pickle dict; validates geometry names exist in geometry_indices."""
    for geo in train_geometries + val_geometries:
        if geo not in geometry_indices:
            raise KeyError(f"Geometry {geo!r} not in geometry_indices (available: {sorted(geometry_indices)})")
    return {
        "split_by_geometry": True,
        "num_offsets": int(num_offsets),
        "train_geometries": list(train_geometries),
        "val_geometries": list(val_geometries),
        "geometry_indices": {geo: {k: tuple(v) for k, v in ranges.items()} for geo, ranges in geometry_indices.items()},
        **meta,
    }


def require_geometry_indices(split_dict: dict[str, Any], split_path: str = "") -> None:
    if "geometry_indices" not in split_dict:
        label = split_path or "split pickle"
        if "train_ind" in split_dict:
            raise ValueError(
                f"{label} uses legacy train_ind/val_ind format. "
                "Regenerate splits with run_data_processing (geometry_indices required)."
            )
        raise ValueError(f"{label} missing geometry_indices.")


def resolve_flat_indices(
    split_dict: dict[str, Any],
    modality: Modality,
    split: SplitName,
    *,
    split_path: str = "",
) -> np.ndarray:
    """Flatten stored per-geometry ranges into a 1D index array for junction or vessel arrays."""
    require_geometry_indices(split_dict, split_path)
    geo_list = split_dict["train_geometries"] if split == "train" else split_dict["val_geometries"]
    geometry_indices = split_dict["geometry_indices"]
    flat: list[int] = []
    for geo in geo_list:
        ranges = geometry_indices.get(geo)
        if not ranges or modality not in ranges:
            continue
        start, end = ranges[modality]
        flat.extend(range(int(start), int(end)))
    return np.asarray(flat, dtype=int)


def load_split_for_training(split_path: str) -> dict[str, Any]:
    """Load a split pickle and require the geometry_indices schema."""
    split_dict = load_dict(split_path)
    require_geometry_indices(split_dict, split_path)
    return split_dict


def write_geometries_txt(
    path: str,
    train_geometries: list[str],
    val_geometries: list[str],
) -> None:
    with open(path, "w") as f:
        f.write("Train geometries:\n")
        for geo in train_geometries:
            f.write(f"  {geo}\n")
        f.write("Validation geometries:\n")
        for geo in val_geometries:
            f.write(f"  {geo}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val split indices for NN training.")
    parser.add_argument("--set_name", required=True, help="e.g. VMR")
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations",
        choices=["bifurcations", "bifurcations_EL"],
        help="Geometry variant (default: bifurcations)",
    )
    parser.add_argument(
        "--set_type",
        default="all",
        help="Cohort folder tier under jax_arrays/split_indices (default: all)",
    )
    parser.add_argument(
        "--num_geos",
        type=int,
        required=True,
        help="Number of geometries used to build the jax arrays",
    )
    parser.add_argument(
        "--percent_train",
        type=float,
        required=True,
        help="Fraction of points to use for training (0,1)",
    )
    split_seed_default = get_pipeline_config().split.seed
    parser.add_argument(
        "--seed",
        type=int,
        default=split_seed_default,
        help=f"RNG seed for reproducible split (default: {split_seed_default} from config)",
    )
    parser.add_argument("--data_root", default="data", help="Repo data root (default: data)")
    args = parser.parse_args()

    jax_arrays_path = resolve_jax_arrays_path(
        args.data_root,
        args.set_name,
        args.geometry_variant,
        args.set_type,
        args.num_geos,
        None,
        vessel=False,
        split_path=None,
        stenosis_clipping_enabled=False,
    )
    data_dict = load_dict(jax_arrays_path)

    if "input" not in data_dict:
        raise ValueError(f"Expected 'input' in data_dict at {jax_arrays_path}")
    num_pts = int(np.asarray(data_dict["input"]).shape[0])

    row_ranges, _total_rows, geometries = resolve_geometry_row_ranges_from_jax_dict(data_dict)
    if len(row_ranges) != args.num_geos:
        raise ValueError(
            f"Geometry count mismatch: pickle lists {len(row_ranges)} geometries "
            f"but jax arrays were built with num_geos={args.num_geos}"
        )

    train_geometries, val_geometries = generate_geometry_split(args.percent_train, args.seed, geometries)

    vessel_jax_path = resolve_jax_arrays_path(
        args.data_root,
        args.set_name,
        args.geometry_variant,
        args.set_type,
        args.num_geos,
        None,
        vessel=True,
        split_path=None,
        stenosis_clipping_enabled=False,
    )
    vessel_dict: dict[str, Any] = load_dict(vessel_jax_path) if os.path.exists(vessel_jax_path) else {}
    geometry_indices = build_geometry_index_map(data_dict, vessel_dict)
    split_dict = build_split_dict(
        train_geometries,
        val_geometries,
        geometry_indices,
        num_offsets=1,
        percent_train=float(args.percent_train),
        seed=int(args.seed),
        num_pts=int(num_pts),
    )

    train_ind = resolve_flat_indices(split_dict, "junction", "train")
    val_ind = resolve_flat_indices(split_dict, "junction", "val")

    out_dir = os.path.join(args.data_root, "split_indices", args.set_name, args.geometry_variant, args.set_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"train_val_ind_{args.set_name}_num_geos_{args.num_geos}")
    save_dict(split_dict, out_path)

    geometries_txt_path = out_path + "_geometries.txt"
    write_geometries_txt(geometries_txt_path, train_geometries, val_geometries)

    print(f"Wrote split indices to {out_path}")
    print(
        f"  num_pts={num_pts}  n_train={len(train_ind)}  n_val={len(val_ind)}  "
        f"split_by_geometry=True  num_offsets={split_dict['num_offsets']}"
    )
    print(f"Wrote geometry set assignment to {geometries_txt_path}")
    print("Train geometries:")
    for g in train_geometries:
        print(f"  {g}")
    print("Validation geometries:")
    for g in val_geometries:
        print(f"  {g}")


if __name__ == "__main__":
    main()
