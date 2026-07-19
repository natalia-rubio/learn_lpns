"""Matplotlib LaTeX detection and label helpers for visualization scripts."""

from __future__ import annotations

import io
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

# Common macOS / Linux TeX locations when ``latex`` is not on PATH.
_LATEX_FALLBACK_PATHS = (
    "/Library/TeX/texbin/latex",
    "/usr/texbin/latex",
    "/usr/local/bin/latex",
)


@lru_cache(maxsize=1)
def latex_executable() -> str | None:
    """Return path to a ``latex`` binary, or None if not found."""
    found = shutil.which("latex")
    if found:
        return found
    for path in _LATEX_FALLBACK_PATHS:
        if Path(path).is_file():
            return path
    return None


@lru_cache(maxsize=1)
def latex_is_available() -> bool:
    """Return True if a ``latex`` executable is available."""
    return latex_executable() is not None


def ensure_tex_bin_on_path() -> str | None:
    """
    Ensure matplotlib can invoke ``latex`` / ``dvipng`` via PATH.

    Matplotlib's usetex subprocess looks up tools by name, so finding an
    absolute TeX binary is not enough — its directory must be on PATH.
    """
    exe = latex_executable()
    if not exe:
        return None
    texbin = str(Path(exe).resolve().parent)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if texbin not in parts:
        os.environ["PATH"] = texbin + (os.pathsep + path if path else "")
    return texbin


def _apply_mathtext_fallback(plt) -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"


def _apply_latex_style(plt) -> None:
    ensure_tex_bin_on_path()
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"


def probe_latex_available(plt) -> tuple[bool, str | None]:
    """
    Probe whether system LaTeX works with matplotlib ``text.usetex``.

    Returns ``(ok, error_message)``. On failure, mathtext fallback is applied.
    """
    if not latex_is_available():
        _apply_mathtext_fallback(plt)
        return False, "latex executable not found on PATH or in common TeX locations"
    try:
        _apply_latex_style(plt)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_title(r"Test $\alpha$")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return True, None
    except Exception as e:
        _apply_mathtext_fallback(plt)
        return False, f"{type(e).__name__}: {e}"


def configure_matplotlib_latex(plt, *, force: bool | None = None) -> bool:
    """
    Enable LaTeX text rendering when available.

    When ``force`` is None, probe that matplotlib can actually render with
    ``text.usetex`` (not only that ``latex`` is on PATH).

    Returns whether ``text.usetex`` was enabled.
    """
    if force is False:
        _apply_mathtext_fallback(plt)
        return False
    if force is True:
        _apply_latex_style(plt)
        return True
    ok, err = probe_latex_available(plt)
    if not ok and err:
        # Stash for callers that want a clearer warning.
        configure_matplotlib_latex.last_error = err  # type: ignore[attr-defined]
    else:
        configure_matplotlib_latex.last_error = None  # type: ignore[attr-defined]
    return ok


configure_matplotlib_latex.last_error = None  # type: ignore[attr-defined]


def savefig_with_latex_fallback(fig, path: str, plt, **kwargs: Any) -> None:
    """Save figure; disable usetex and retry if LaTeX rendering fails."""
    try:
        fig.savefig(path, **kwargs)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        msg = str(e).lower()
        if "latex" in msg or "tex" in msg or "dvipng" in msg:
            _apply_mathtext_fallback(plt)
            fig.savefig(path, **kwargs)
        else:
            raise


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
