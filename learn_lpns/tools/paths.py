"""Filesystem paths shared across CLIs and library code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Absolute path to the repository root (parent of the ``learn_lpns`` package)."""
    return Path(__file__).resolve().parents[2]
