"""
Load and cache pipeline configuration from YAML.

Layering (each step deep-merges into the previous):

  1. defaults.yaml base sections (repo config/ or bundled learn_lpns/config/)
  2. defaults.yaml ``set_overrides.<set_name>`` (optional, when set_name is passed)
  3. inline ``<set_name>:`` blocks under ``training.rri_coefficients`` entries (optional)
  4. inline ``<set_name>:`` blocks under ``data_processing.stenosis_generation_limit`` (optional)
  5. config/sets/<set_name>.yaml (optional external file, when set_name is passed)
  6. overrides dict (programmatic patches, tests)

Merge rule: nested mappings deep-merge; scalars/lists are replaced, except
``training.rri_coefficients`` whose entries are merged by their ``name`` field. That lets
a per-set override change a single coefficient (e.g. S) without redefining R/S/L.

Inline per-set overrides may live under a coefficient entry::

  rri_coefficients:
    - name: S
      junction_layer_width: 10
      VMR_all:
        junction_layer_width: 20

or under ``data_processing.stenosis_generation_limit``::

  stenosis_generation_limit:
    junction_max_generation: 1
    VMR_aorta:
      junction_max_generation: 0

``set_overrides`` is a loader-only top-level section (stripped before validation).
Inline ``<set_name>`` blocks are also stripped before validation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from learn_lpns.config.models import (
    PipelineConfig,
    RriCoefficientConfig,
    StenosisGenerationLimitConfig,
)
from learn_lpns.tools.paths import repo_root

_PACKAGE_DIR = Path(__file__).resolve().parent

# Keys allowed on each ``training.rri_coefficients`` entry; any other mapping child is treated
# as an inline per-set override (e.g. ``VMR_all: { junction_layer_width: 20 }`` under ``S``).
_RRI_COEFFICIENT_FIELD_KEYS = frozenset(RriCoefficientConfig.model_fields.keys())

# Keys allowed on ``data_processing.stenosis_generation_limit``; any other child is an inline set override.
_STENOSIS_LIMIT_FIELD_KEYS = frozenset(StenosisGenerationLimitConfig.model_fields.keys())


def _collect_inline_rri_set_overrides(training: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """
    Scan coefficient entries for inline ``<set_name>: { ... }`` blocks.

    Returns ``set_name -> partial config`` trees suitable for :func:`_deep_merge`.
    """
    if not training or not isinstance(training.get("rri_coefficients"), list):
        return {}

    by_set: dict[str, list[dict[str, Any]]] = {}
    for entry in training["rri_coefficients"]:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        coef_name = entry["name"]
        for key, value in entry.items():
            if key in _RRI_COEFFICIENT_FIELD_KEYS:
                continue
            if not isinstance(value, dict):
                raise ValueError(
                    f"Inline set override under rri_coefficients.{coef_name}.{key} must be a mapping, "
                    f"got {type(value).__name__}."
                )
            by_set.setdefault(key, []).append({"name": coef_name, **value})

    return {
        set_name: {"training": {"rri_coefficients": coef_overrides}}
        for set_name, coef_overrides in by_set.items()
    }


def _strip_inline_rri_set_overrides(training: dict[str, Any] | None) -> None:
    """Remove inline ``<set_name>`` blocks from coefficient entries in place."""
    if not training or not isinstance(training.get("rri_coefficients"), list):
        return

    cleaned: list[Any] = []
    for entry in training["rri_coefficients"]:
        if not isinstance(entry, dict):
            cleaned.append(entry)
            continue
        cleaned.append(
            {key: value for key, value in entry.items() if key in _RRI_COEFFICIENT_FIELD_KEYS}
        )
    training["rri_coefficients"] = cleaned


def _collect_inline_stenosis_set_overrides(
    data_processing: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """
    Scan stenosis_generation_limit for inline ``<set_name>: { ... }`` blocks.

    Returns ``set_name -> partial config`` trees suitable for :func:`_deep_merge`.
    """
    if not data_processing or not isinstance(data_processing.get("stenosis_generation_limit"), dict):
        return {}

    limit = data_processing["stenosis_generation_limit"]
    by_set: dict[str, dict[str, Any]] = {}
    for key, value in limit.items():
        if key in _STENOSIS_LIMIT_FIELD_KEYS:
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Inline set override under stenosis_generation_limit.{key} must be a mapping, "
                f"got {type(value).__name__}."
            )
        by_set[key] = {"data_processing": {"stenosis_generation_limit": dict(value)}}

    return by_set


def _strip_inline_stenosis_set_overrides(data_processing: dict[str, Any] | None) -> None:
    """Remove inline ``<set_name>`` blocks from stenosis_generation_limit in place."""
    if not data_processing or not isinstance(data_processing.get("stenosis_generation_limit"), dict):
        return

    limit = data_processing["stenosis_generation_limit"]
    data_processing["stenosis_generation_limit"] = {
        key: value for key, value in limit.items() if key in _STENOSIS_LIMIT_FIELD_KEYS
    }


def _merge_named_list(base: list[Any], override: list[Any]) -> list[Any]:
    """
    Merge two lists of mappings keyed by their ``name`` field (override fields win).

    Entries in ``override`` that share a ``name`` with a base entry are deep-merged
    into it; new names are appended in order. If either list contains an entry that is
    not a mapping with a ``name`` key, fall back to full replacement by ``override``.
    """
    if not all(isinstance(e, dict) and "name" in e for e in base):
        return override
    if not all(isinstance(e, dict) and "name" in e for e in override):
        return override

    by_name: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for entry in base:
        by_name[entry["name"]] = dict(entry)
        order.append(entry["name"])
    for entry in override:
        name = entry["name"]
        if name in by_name:
            by_name[name] = _deep_merge(by_name[name], entry)
        else:
            by_name[name] = dict(entry)
            order.append(name)
    return [by_name[name] for name in order]


# Config keys whose list values are merged element-wise by ``name`` rather than replaced.
_NAMED_LIST_MERGE_KEYS = frozenset({"rri_coefficients"})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (override wins on leaf conflicts)."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in _NAMED_LIST_MERGE_KEYS
            and isinstance(merged.get(key), list)
            and isinstance(value, list)
        ):
            merged[key] = _merge_named_list(merged[key], value)
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__} in {path}")
    return data


def resolve_set_config_path(set_name: str) -> Path | None:
    """Return config/sets/<set_name>.yaml if it exists, else None."""
    path = repo_root() / "config" / "sets" / f"{set_name}.yaml"
    return path if path.is_file() else None


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """
    Resolve base defaults YAML path.

    Priority: explicit argument > LEARN_LPNS_CONFIG env > repo config/defaults.yaml
    > bundled learn_lpns/config/defaults.yaml.
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    env_path = os.environ.get("LEARN_LPNS_CONFIG")
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"LEARN_LPNS_CONFIG points to missing file: {path}")
        return path

    repo_default = repo_root() / "config" / "defaults.yaml"
    if repo_default.is_file():
        return repo_default

    bundled = _PACKAGE_DIR / "defaults.yaml"
    if bundled.is_file():
        return bundled

    raise FileNotFoundError(
        "No pipeline config found. Expected config/defaults.yaml at repo root "
        "or bundled learn_lpns/config/defaults.yaml."
    )


def load_pipeline_config(
    config_path: str | Path | None = None,
    *,
    set_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load YAML, merge optional per-set layer and overrides, validate with Pydantic."""
    path = resolve_config_path(config_path)
    data = _load_yaml_mapping(path)

    # Loader-only top-level section (not part of PipelineConfig).
    set_overrides = data.pop("set_overrides", None) or {}

    # Inline per-set blocks under rri_coefficients (e.g. VMR_all: { junction_layer_width: 20 }).
    inline_set_overrides = _collect_inline_rri_set_overrides(data.get("training"))
    _strip_inline_rri_set_overrides(data.get("training"))

    # Inline per-set blocks under stenosis_generation_limit (e.g. VMR_aorta: { junction_max_generation: 0 }).
    inline_stenosis_overrides = _collect_inline_stenosis_set_overrides(data.get("data_processing"))
    _strip_inline_stenosis_set_overrides(data.get("data_processing"))

    if set_name:
        cohort_override = set_overrides.get(set_name)
        if isinstance(cohort_override, dict):
            data = _deep_merge(data, cohort_override)
        inline_override = inline_set_overrides.get(set_name)
        if isinstance(inline_override, dict):
            data = _deep_merge(data, inline_override)
        stenosis_override = inline_stenosis_overrides.get(set_name)
        if isinstance(stenosis_override, dict):
            data = _deep_merge(data, stenosis_override)
        set_path = resolve_set_config_path(set_name)
        if set_path is not None:
            data = _deep_merge(data, _load_yaml_mapping(set_path))

    if overrides:
        data = _deep_merge(data, overrides)

    return PipelineConfig.model_validate(data)


@lru_cache(maxsize=16)
def _cached_pipeline_config(resolved_path: str, set_name: str) -> PipelineConfig:
    # set_name is normalized to "" when not provided so lru_cache stays hashable
    return load_pipeline_config(resolved_path, set_name=set_name or None)


def get_pipeline_config(
    config_path: str | Path | None = None,
    *,
    set_name: str | None = None,
    reload: bool = False,
) -> PipelineConfig:
    """
    Return cached pipeline config (keyed by defaults path + set_name).

    Pass set_name when loading cohort-specific overrides from config/sets/.
    Pass reload=True after editing YAML on disk.
    """
    path = resolve_config_path(config_path)
    if reload:
        _cached_pipeline_config.cache_clear()
    return _cached_pipeline_config(str(path), set_name or "")
