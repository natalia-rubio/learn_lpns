#!/usr/bin/env python3
"""
Minimal utilities used across this repo.

Note: Some modules were originally written in a different project and expect
`learn_lpns.tools.basic.load_dict/save_dict`. We provide lightweight implementations
here using pickle (and dill if available).
"""

from __future__ import annotations

import os
import pickle
from typing import Any


def save_dict(obj: Any, path: str) -> None:
    """Serialize `obj` to `path` using dill (if available) else pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import dill  # type: ignore

        with open(path, "wb") as f:
            dill.dump(obj, f)
        return
    except (ImportError, ModuleNotFoundError):
        pass

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(path: str) -> Any:
    """Load an object from `path` that was saved with `save_dict`."""
    # Try dill first (it can load pickle too in many cases), then pickle.
    try:
        import dill  # type: ignore

        with open(path, "rb") as f:
            return dill.load(f)
    except (ImportError, ModuleNotFoundError):
        pass

    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "save_dict",
    "load_dict",
]
