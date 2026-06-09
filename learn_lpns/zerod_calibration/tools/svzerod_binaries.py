"""Resolve svZeroDSolver / svZeroDCalibrator executable paths."""

from __future__ import annotations

import os
import shutil


def svzerod_install_dir() -> str:
    """Directory containing svzerodsolver and svzerodcalibrator binaries."""
    env = os.environ.get("SVZEROD_INSTALL_DIR")
    if env:
        return os.path.abspath(env)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sibling = os.path.join(os.path.dirname(repo_root), "svZeroDPlus", "Release")
    return sibling


def svzerod_binary(name: str) -> str:
    """Return path to a svZeroD executable, checking install dir then PATH."""
    install_path = os.path.join(svzerod_install_dir(), name)
    if os.path.exists(install_path):
        return install_path

    found = shutil.which(name)
    if found:
        return found

    raise RuntimeError(
        f"{name} not found. Set SVZEROD_INSTALL_DIR to the directory containing "
        f"the binaries (expected {install_path}) or add {name} to PATH. "
        "Run scripts/setup_cross_validation.sh to clone and build svZeroDPlus."
    )
