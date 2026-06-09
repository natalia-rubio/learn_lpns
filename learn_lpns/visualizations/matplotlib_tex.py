"""Matplotlib LaTeX detection and label helpers for visualization scripts."""

from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def latex_is_available() -> bool:
    """Return True if a ``latex`` executable is on PATH."""
    return shutil.which("latex") is not None


def configure_matplotlib_latex(plt, *, force: bool | None = None) -> bool:
    """
    Enable LaTeX text rendering when available.

    Returns whether ``text.usetex`` was enabled.
    """
    use_latex = latex_is_available() if force is None else bool(force)
    if use_latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "cm"
    else:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "cm"
    return use_latex


def plot_label(text: str, *, use_latex: bool | None = None) -> str:
    """
    Adapt axis/legend strings for the active matplotlib text mode.

    With ``text.usetex=True``, percent signs and spaces may need LaTeX escaping.
    With mathtext only, strip those escapes so labels render correctly.
    """
    if use_latex is None:
        use_latex = latex_is_available()
    if use_latex:
        return text
    return text.replace(r"\%", "%").replace(r"\ ", " ")
