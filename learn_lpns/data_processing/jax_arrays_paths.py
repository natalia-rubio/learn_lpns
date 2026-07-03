"""JAX pickle path resolution for unclipped / clipped / trial-specific cohorts."""

from __future__ import annotations

import os
import re

_TRIAL_SPLIT_RE = re.compile(r"_trial_(\d+)$")


def parse_trial_id_from_split_path(split_path: str | None) -> int | None:
    """Return trial index from split path ending in ``_trial_{T}``, else None."""
    if not split_path:
        return None
    base = os.path.basename(split_path)
    match = _TRIAL_SPLIT_RE.search(base)
    if match is None:
        return None
    return int(match.group(1))


def jax_arrays_filename(
    num_geos: int,
    *,
    vessel: bool,
    stenosis_clipping_enabled: bool,
    split_path: str | None = None,
) -> str:
    """Return jax pickle basename for the given split and clipping mode."""
    prefix = "jax_arrays_vessel" if vessel else "jax_arrays"
    trial = parse_trial_id_from_split_path(split_path)
    if trial is not None:
        suffix = f"_trial_{trial}_clipped" if stenosis_clipping_enabled else "_unclipped"
    elif stenosis_clipping_enabled:
        suffix = "_clipped"
    else:
        suffix = "_unclipped"
    return f"{prefix}_num_geos_{num_geos}{suffix}.pkl"


def resolve_jax_arrays_path(
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    num_geos: int,
    run_config_suffix: str | None,
    *,
    vessel: bool,
    split_path: str | None,
    stenosis_clipping_enabled: bool,
) -> str:
    """Resolve jax pickle path from split path and stenosis_clipping config."""
    filename = jax_arrays_filename(
        num_geos,
        vessel=vessel,
        stenosis_clipping_enabled=stenosis_clipping_enabled,
        split_path=split_path,
    )
    parts = [data_root, "jax_arrays", set_name]
    if run_config_suffix:
        parts.append(run_config_suffix)
    parts.extend([geometry_variant, set_type, filename])
    return os.path.join(*parts)


__all__ = [
    "jax_arrays_filename",
    "parse_trial_id_from_split_path",
    "resolve_jax_arrays_path",
]
