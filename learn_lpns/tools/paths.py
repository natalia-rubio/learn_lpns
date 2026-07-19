"""Filesystem paths shared across CLIs and library code."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Absolute path to the repository root (parent of the ``learn_lpns`` package).

    If ``LEARN_LPNS_ROOT`` is set, that directory is used. This lets experiment
    sandboxes redirect training/deploy I/O away from the real checkout when the
    package is imported via a symlink (``Path(__file__).resolve()`` would otherwise
    escape the sandbox).
    """
    override = (os.environ.get("LEARN_LPNS_ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]
