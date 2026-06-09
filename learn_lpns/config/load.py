"""
Load and cache pipeline configuration from YAML.

Layering (each step deep-merges into the previous):

  1. defaults.yaml (repo config/ or bundled learn_lpns/config/)
  2. config/sets/<set_name>.yaml (optional, when set_name is passed)
  3. overrides dict (programmatic patches, tests)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from learn_lpns.config.models import PipelineConfig

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (override wins on leaf conflicts)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
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
    path = _REPO_ROOT / "config" / "sets" / f"{set_name}.yaml"
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

    repo_default = _REPO_ROOT / "config" / "defaults.yaml"
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

    if set_name:
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
