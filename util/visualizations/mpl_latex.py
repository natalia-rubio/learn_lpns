"""Matplotlib LaTeX helpers with automatic fallback when latex is not installed."""

from __future__ import annotations

import io
from typing import Any


def _apply_mathtext_fallback(plt) -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"


def _apply_latex_style(plt) -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"


def probe_latex_available(plt) -> bool:
    """Return True if system LaTeX works with matplotlib usetex."""
    try:
        _apply_latex_style(plt)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_title(r"Test $\alpha$")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return True
    except Exception:
        _apply_mathtext_fallback(plt)
        return False


def savefig_with_latex_fallback(fig, path: str, plt, **kwargs: Any) -> None:
    """Save figure; disable usetex and retry if LaTeX rendering fails."""
    try:
        fig.savefig(path, **kwargs)
    except (RuntimeError, FileNotFoundError) as e:
        if "latex" in str(e).lower() or "tex" in str(e).lower():
            _apply_mathtext_fallback(plt)
            fig.savefig(path, **kwargs)
        else:
            raise
