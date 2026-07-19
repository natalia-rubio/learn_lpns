"""Generation-gated stenosis: RSL for proximal elements, RL (S=0) for distal ones."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from learn_lpns.config import get_pipeline_config
from learn_lpns.data_processing.inputs_from_0d_config import compute_bifurcation_generation_by_vessel

StenosisModality = Literal["junction", "vessel"]


def resolve_stenosis_generation_limits(set_name: str | None) -> tuple[bool, float, float]:
    """Return (enabled, junction_max_generation, vessel_max_generation) from pipeline config."""
    cfg = get_pipeline_config(set_name=set_name).data_processing.stenosis_generation_limit
    return bool(cfg.enabled), cfg.junction_limit(), cfg.vessel_limit()


def resolve_stenosis_generation_limit(
    set_name: str | None,
    *,
    vessel: bool = False,
) -> tuple[bool, float]:
    """Return (enabled, max_generation) for one modality (junction or vessel)."""
    enabled, junction_max, vessel_max = resolve_stenosis_generation_limits(set_name)
    max_gen = vessel_max if vessel else junction_max
    return enabled, max_gen


def generation_allows_stenosis(generation: float, max_generation: float) -> bool:
    """True when stenosis should be fit, trained, or predicted (inclusive bound).

    Generations are non-negative, so ``max_generation=-1`` disables stenosis for all elements.
    """
    return float(generation) <= float(max_generation)


def junction_inlet_generation(junction: dict[str, Any], gen_by_vessel: dict[Any, float]) -> float:
    """Bifurcation generation at a junction (inlet vessel generation)."""
    inlets = junction.get("inlet_vessels") or []
    if not inlets:
        raise ValueError(
            f"Junction {junction.get('junction_name', '?')!r}: no inlet_vessels for generation lookup."
        )
    inlet_key = inlets[0]
    try:
        inlet_key = int(inlet_key)
    except (TypeError, ValueError):
        pass
    if inlet_key not in gen_by_vessel:
        raise ValueError(
            f"Junction {junction.get('junction_name', '?')!r}: inlet vessel {inlet_key!r} "
            f"has no bifurcation generation."
        )
    return float(gen_by_vessel[inlet_key])


def vessel_generation(vessel_id: int, gen_by_vessel: dict[Any, float]) -> float:
    """Bifurcation generation for a vessel element."""
    key = vessel_id
    try:
        key = int(vessel_id)
    except (TypeError, ValueError):
        pass
    if key not in gen_by_vessel:
        raise ValueError(f"Vessel id {vessel_id!r} has no bifurcation generation.")
    return float(gen_by_vessel[key])


def zero_stenosis_by_generation(
    pred_S: np.ndarray,
    generation: np.ndarray,
    *,
    max_generation: float,
) -> np.ndarray:
    """Zero stenosis predictions where generation exceeds max_generation."""
    out = np.asarray(pred_S, dtype=float).copy()
    gen = np.asarray(generation, dtype=float).reshape(-1)
    if gen.shape[0] != out.shape[0]:
        raise ValueError(
            f"generation length ({gen.shape[0]}) must match pred_S length ({out.shape[0]})"
        )
    out[gen > float(max_generation)] = 0.0
    return out


def load_vessel_generation_from_geometric_csv(geometric_input_path: str) -> tuple[list[int], np.ndarray]:
    """Load vessel_id list and generation column from a geometric features CSV."""
    import csv

    with open(geometric_input_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            vid_idx = header.index("vessel_id")
            gen_idx = header.index("generation")
        except ValueError as exc:
            raise ValueError(
                f"Geometric CSV {geometric_input_path!r} must include vessel_id and generation columns."
            ) from exc
        vessel_ids: list[int] = []
        generations: list[float] = []
        for row in reader:
            if not row:
                continue
            vessel_ids.append(int(row[vid_idx]))
            generations.append(float(row[gen_idx]))
    return vessel_ids, np.asarray(generations, dtype=float)


def slice_jax_generation_for_geo(
    data_dict: dict[str, Any],
    *,
    geo_name: str | None,
    n_expected: int,
) -> np.ndarray:
    """Return per-row generation aligned with ``load_junction_rows_from_jax_dict`` slicing."""
    if "generation" not in data_dict:
        raise ValueError("Jax pickle missing 'generation' array. Re-run data processing.")
    gen_full = np.asarray(data_dict["generation"], dtype=float).reshape(-1)
    start, end = 0, gen_full.shape[0]
    if geo_name is not None:
        geoms = [str(g) for g in data_dict["geometry_names_order"]]
        ranges = data_dict["geometry_row_ranges"]
        if geo_name not in geoms:
            raise ValueError(f"Geometry {geo_name!r} not in jax pickle geometry_names_order: {geoms}")
        gi = geoms.index(geo_name)
        start, end = int(ranges[gi][0]), int(ranges[gi][1])
    gen_slice = gen_full[start:end]
    if gen_slice.shape[0] != n_expected:
        raise ValueError(
            f"Generation slice length ({gen_slice.shape[0]}) != expected rows ({n_expected})"
        )
    return gen_slice


def resolve_effective_stenosis_generation_limit(
    *,
    set_name: str,
    quadratic_resistor: bool,
    vessel: bool,
    bounds_model: Any | None = None,
) -> tuple[bool, float]:
    """Checkpoint metadata overrides config when present (per modality)."""
    if not quadratic_resistor:
        return False, 1.0
    enabled, max_gen = resolve_stenosis_generation_limit(set_name, vessel=vessel)
    if bounds_model is not None and getattr(bounds_model, "stenosis_generation_limit_enabled", False):
        enabled = True
        max_gen = float(getattr(bounds_model, "stenosis_generation_max", max_gen))
    return enabled, max_gen


def apply_stenosis_generation_gate(
    pred_S: np.ndarray,
    generation: np.ndarray,
    *,
    enabled: bool,
    max_generation: float,
    modality: StenosisModality = "junction",
) -> np.ndarray:
    """Zero pred_S where generation exceeds max_generation when gating is enabled."""
    if not enabled:
        return np.asarray(pred_S, dtype=float)
    out = zero_stenosis_by_generation(pred_S, generation, max_generation=max_generation)
    n_zeroed = int(np.sum(np.asarray(generation, dtype=float).reshape(-1) > float(max_generation)))
    if n_zeroed:
        print(
            f"      stenosis_generation_limit ({modality}): zeroed S on {n_zeroed}/{len(out)} rows "
            f"(generation > {max_generation:g})"
        )
    return out


__all__ = [
    "apply_stenosis_generation_gate",
    "compute_bifurcation_generation_by_vessel",
    "generation_allows_stenosis",
    "junction_inlet_generation",
    "load_vessel_generation_from_geometric_csv",
    "resolve_effective_stenosis_generation_limit",
    "resolve_stenosis_generation_limit",
    "resolve_stenosis_generation_limits",
    "slice_jax_generation_for_geo",
    "StenosisModality",
    "vessel_generation",
    "zero_stenosis_by_generation",
]
