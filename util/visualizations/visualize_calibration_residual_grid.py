#!/usr/bin/env python3
"""
Heatmap of stacked calibration residual L2 norms across cohort geometries and geometry variants.

Rows: per-case folders under a cohort directory (e.g. TST-1, TST-3, TST-5).
Columns: modalities original (base graph), bifurcations, bifurcations_EL — each using
BloodVesselJunction calibration residual CSV from svZeroDCalibrator.

Cell **color** encodes ``log₁₀`` of the residual Euclidean norm (blue = smallest, red =
largest within the grid). Cell annotations show the linear norm ‖r‖₂.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _stem_calibration_input(geometry_variant: str, junction_type: str) -> str:
    """JSON basename stem (no extension) for calibration_input_*."""
    if geometry_variant == "original":
        return f"calibration_input_{junction_type}"
    return f"{geometry_variant}_calibration_input_{junction_type}"


def residual_csv_basename_for_variant(
    geometry_variant: str, junction_type: str = "BloodVesselJunction"
) -> str:
    """
    Basename of the stacked residual CSV next to calibration inputs, matching
    :func:`util.zerod_calibration.calibration.calibration_residual_csv_basename`
    without importing that module (avoids heavy deps like vtk).
    """
    stem = _stem_calibration_input(geometry_variant, junction_type)
    if "_calibration_input_" in stem:
        return stem.replace("_calibration_input_", "_calibration_residual_", 1) + ".csv"
    if stem.endswith("_calibration_input"):
        return stem[: -len("_calibration_input")] + "_calibration_residual.csv"
    if stem == "calibration_input":
        return "calibration_residual.csv"
    return stem + "_calibration_residual.csv"


def read_residual_l2_norm(residual_csv_path: str) -> Optional[float]:
    """Euclidean norm of the ``residual`` column (one stacked vector)."""
    if not os.path.isfile(residual_csv_path):
        return None
    values: List[float] = []
    try:
        with open(residual_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "residual" not in reader.fieldnames:
                return None
            for row in reader:
                r = row.get("residual")
                if r is None or r.strip() == "":
                    continue
                values.append(float(r))
    except (OSError, ValueError):
        return None
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.linalg.norm(arr))


def discover_geometry_case_dirs(cohort_dir: str) -> List[str]:
    """Sorted list of immediate subdirectories (case names like TST-1)."""
    if not os.path.isdir(cohort_dir):
        return []
    names: List[str] = []
    for name in sorted(os.listdir(cohort_dir)):
        p = os.path.join(cohort_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            names.append(name)
    return names


DEFAULT_MODALITIES: Sequence[Tuple[str, str]] = (
    ("original", "original"),
    ("bifurcations", "bifurcations"),
    ("bifurcations_EL", "bifurcations_EL"),
)


def build_residual_norm_matrix(
    cohort_dir: str,
    geometry_case_dirs: Optional[Sequence[str]] = None,
    junction_type: str = "BloodVesselJunction",
    modalities: Sequence[Tuple[str, str]] = DEFAULT_MODALITIES,
) -> Tuple[np.ndarray, List[str], List[str], Dict[Tuple[int, int], str]]:
    """
    Returns:
        matrix (n_rows, n_cols) with NaN where file missing / invalid
        row_labels, col_labels (short names for modalities)
        paths: map (i,j) -> csv path attempted (for debugging)
    """
    cases = (
        list(geometry_case_dirs)
        if geometry_case_dirs is not None
        else discover_geometry_case_dirs(cohort_dir)
    )
    col_labels = [m[1] for m in modalities]
    n_rows, n_cols = len(cases), len(modalities)
    mat = np.full((n_rows, n_cols), np.nan, dtype=float)
    paths: Dict[Tuple[int, int], str] = {}

    for i, case in enumerate(cases):
        case_dir = os.path.join(cohort_dir, case)
        for j, (_, variant_key) in enumerate(modalities):
            bn = residual_csv_basename_for_variant(variant_key, junction_type)
            path = os.path.join(case_dir, bn)
            paths[(i, j)] = path
            nrm = read_residual_l2_norm(path)
            mat[i, j] = np.nan if nrm is None else nrm

    return mat, cases, col_labels, paths


def plot_calibration_residual_l2_grid(
    cohort_dir: str,
    geometry_case_dirs: Optional[Sequence[str]] = None,
    junction_type: str = "BloodVesselJunction",
    modalities: Sequence[Tuple[str, str]] = DEFAULT_MODALITIES,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Write a PNG heatmap; color scale is ``log₁₀‖r‖₂`` (linear norms shown in cells).

    ``cohort_dir`` is the directory containing per-geometry case folders (e.g.
    ``data/zeroD/TST-cohort/<run_config>/`` with ``TST-1``, ``TST-3``, … inside).
    """
    if not HAS_MPL:
        print("  ⊘ Skipping residual L2 grid (matplotlib not installed).")
        return None

    mat, row_labels, col_labels, _ = build_residual_norm_matrix(
        cohort_dir,
        geometry_case_dirs=geometry_case_dirs,
        junction_type=junction_type,
        modalities=modalities,
    )

    if mat.size == 0 or len(row_labels) == 0:
        print("  ⊘ Skipping residual L2 grid (no geometry case directories found).")
        return None

    valid = mat[np.isfinite(mat)]
    if valid.size == 0:
        print("  ⊘ Skipping residual L2 grid (no residual CSVs found or all invalid).")
        return None

    # Colors use log₁₀(||r||₂); floor zeros to avoid -inf (should be rare for physics residuals).
    eps = 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mat = np.where(np.isfinite(mat), np.log10(np.maximum(mat, eps)), np.nan)

    valid_log = log_mat[np.isfinite(log_mat)]
    vmin, vmax = float(np.min(valid_log)), float(np.max(valid_log))
    if vmin == vmax:
        vmax = vmin + 1e-15

    # Blue (small log norm) -> red (large log norm): coolwarm
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))

    masked = np.ma.masked_invalid(log_mat)
    fig_w = max(7.0, 1.3 * len(col_labels) + 4)
    fig_h = max(4.0, 0.55 * len(row_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Geometry variant (BloodVesselJunction)")
    ax.set_ylabel("Case")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Calibration residual ‖·‖₂ (stacked equations), colors = log₁₀")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}$(residual L2 norm)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            lv = log_mat[i, j]
            if np.isfinite(v):
                txt = f"{v:.4g}"
                # contrast on diverging map: scale by displayed (log) values
                rel = (lv - vmin) / (vmax - vmin) if np.isfinite(lv) and vmax > vmin else 0.5
                tcol = "white" if 0.25 < rel < 0.75 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=tcol, fontsize=9)

    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            os.getcwd(),
            "results",
            "zerod_calibration",
            "residual_l2_grid.png",
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Wrote calibration residual L2 grid: {output_path}")
    return output_path


def plot_residual_grid_after_calibration(
    *,
    repo_root: str,
    set_name: str,
    run_config_suffix: Optional[str],
    junction_types: Sequence[str],
) -> Optional[str]:
    """
    Called from ``generate_zerod_inputs`` after calibration. The cohort directory is
    ``<repo>/data/zeroD/<set_name>/[<run_config>/]`` (sibling case folders: TST-1, …).
    Only runs when ``BloodVesselJunction`` is in ``junction_types`` and that directory exists.
    """
    if "BloodVesselJunction" not in junction_types:
        return None

    if run_config_suffix:
        cohort_dir = os.path.join(repo_root, "data", "zeroD", set_name, run_config_suffix)
    else:
        cohort_dir = os.path.join(repo_root, "data", "zeroD", set_name)

    if not os.path.isdir(cohort_dir):
        return None

    out_name = f"residual_l2_grid_{set_name}"
    if run_config_suffix:
        out_name += f"_{run_config_suffix}"
    out_name += ".png"
    out_path = os.path.join(repo_root, "results", "zerod_calibration", out_name)

    title = f"{set_name}"
    if run_config_suffix:
        title += f" ({run_config_suffix})"

    return plot_calibration_residual_l2_grid(
        cohort_dir,
        junction_type="BloodVesselJunction",
        output_path=out_path,
        title=title,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "cohort_dir",
        help="Directory containing case subfolders (e.g. data/zeroD/TST-cohort/<run_config>)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--junction-type",
        default="BloodVesselJunction",
        help="Junction type used in residual CSV names (default: BloodVesselJunction)",
    )
    args = parser.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if args.output is None:
        args.output = os.path.join(
            REPO_ROOT,
            "results",
            "zerod_calibration",
            "residual_l2_grid_cli.png",
        )

    path = plot_calibration_residual_l2_grid(
        os.path.abspath(args.cohort_dir),
        junction_type=args.junction_type,
        output_path=args.output,
    )
    if path:
        print(path)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
