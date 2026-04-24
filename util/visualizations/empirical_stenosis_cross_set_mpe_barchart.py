#!/usr/bin/env python3
"""
Grouped bar chart: mean MPE (mean pressure max relative error, %) across cohorts
from empirical stenosis comparison outputs.

For each set under ``results/empirical_stenosis_comp/<set>/``, averages
``Mean Pressure Max Rel Error`` from ``empirical_stenosis_mse_comparison.csv`` over
all geometries that have both ``geometric`` and ``stenosis_zero`` columns filled.

95%% CI on each bar uses geometry-to-geometry spread:
    CI95 = 1.96 * std / sqrt(n_geometries)
(same construction as :mod:`cv_cross_set_summary_barchart` for trial-level spread).

Usage:
  python -m util.visualizations.empirical_stenosis_cross_set_mpe_barchart
  python -m util.visualizations.empirical_stenosis_cross_set_mpe_barchart \\
      --set-names VMR_rigid_aorta_adults VMR_abdo VMR_pulmo_healthy \\
      -o results/empirical_stenosis_comp/cross_set_mpe_baseline_vs_stenosis_zero.pdf
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from util.visualizations.plot_location_comparison import (
    _load_mean_pressure_max_rel_pct_from_mse_csv,
    get_line_style,
)

DEFAULT_SET_NAMES = [
    "VMR_rigid_aorta_adults",
    "VMR_abdo",
    "VMR_pulmo_healthy",
]

SET_DISPLAY_NAME = {
    "VMR_rigid_aorta_adults_all": "Aortic",
    "VMR_rigid_aorta_adults": "Aortic",
    "VMR_abdo": "Aortofemoral ",
    "VMR_pulmo": "Pulmonary",
    "VMR_pulmo_healthy": "Pulmonary",
    "VMR_all": "All",
}

MODALITY_ORDER = ["geometric", "stenosis_zero"]

MODALITY_LABEL = {
    "geometric": "With Empirical \n Stenosis Correction",
    "stenosis_zero": "Without Empirical \n Stenosis Correction",
}


def _isnan(x) -> bool:
    return x != x


def _mean_std_n(values: list) -> tuple[float, float, int]:
    clean = [v for v in values if not _isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean, ddof=1)) if n >= 2 else 0.0
    return mean_val, std_val, n


def _ci95_from_std(std_val: float, n: int) -> float:
    if n <= 1 or _isnan(std_val):
        return 0.0
    return 1.96 * std_val / math.sqrt(n)


def _format_set_label(set_name: str) -> str:
    if set_name in SET_DISPLAY_NAME:
        return SET_DISPLAY_NAME[set_name]
    if set_name.startswith("VMR_"):
        return set_name.replace("VMR_", "VMR\n", 1)
    return set_name


def _modality_style_key(modality: str) -> str:
    if modality == "geometric":
        return "geometric_0d"
    return modality


def discover_geometries(empirical_root: str, set_name: str) -> list[str]:
    p = os.path.join(empirical_root, set_name)
    if not os.path.isdir(p):
        return []
    out = []
    for name in sorted(os.listdir(p)):
        if os.path.isdir(os.path.join(p, name)):
            out.append(name)
    return out


def collect_paired_mpe_per_set(
    empirical_root: str, set_name: str
) -> tuple[list[float], list[float]]:
    """Per-geometry MPE %% for geometric and stenosis_zero (paired rows)."""
    geom_vals: list[float] = []
    zero_vals: list[float] = []
    for geo in discover_geometries(empirical_root, set_name):
        mse_csv = os.path.join(
            empirical_root,
            set_name,
            geo,
            "empirical_stenosis_mse_comparison.csv",
        )
        d = _load_mean_pressure_max_rel_pct_from_mse_csv(mse_csv)
        if not d:
            continue
        g = d.get("geometric")
        z = d.get("stenosis_zero")
        if g is None or z is None or _isnan(g) or _isnan(z):
            continue
        geom_vals.append(float(g))
        zero_vals.append(float(z))
    return geom_vals, zero_vals


def _modality_color(modality: str) -> str:
    return get_line_style(_modality_style_key(modality)).get("color", "gray")


def _format_bar_label(mean: float) -> str:
    if not np.isfinite(mean):
        return ""
    return f"{mean:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-set MPE bar chart from empirical_stenosis_comp MSE CSVs."
    )
    parser.add_argument(
        "--set-names",
        nargs="+",
        default=DEFAULT_SET_NAMES,
        help="Cohort folder names under empirical root (default: VMR_rigid_aorta_adults VMR_abdo VMR_pulmo_healthy)",
    )
    parser.add_argument(
        "--empirical-root",
        default="results/empirical_stenosis_comp",
        help="Root containing <set_name>/<geo_name>/ (default: results/empirical_stenosis_comp)",
    )
    parser.add_argument(
        "--geometry-variant-tag",
        default="original",
        help="Tag for default output filename only (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output PDF path (default: under empirical root)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI (default: 180)")
    parser.add_argument(
        "--ylim-max",
        type=float,
        default=None,
        help="Optional fixed y-axis max (%%). Default: auto from data.",
    )
    args = parser.parse_args()

    empirical_root = (
        args.empirical_root
        if os.path.isabs(args.empirical_root)
        else os.path.join(os.getcwd(), args.empirical_root)
    )

    means: dict[str, list[float]] = {}
    cis: dict[str, list[float]] = {}
    missing: list[str] = []
    ns: dict[str, list[int]] = {}

    for set_name in args.set_names:
        g_list, z_list = collect_paired_mpe_per_set(empirical_root, set_name)
        if not g_list:
            missing.append(os.path.join(empirical_root, set_name))
            continue
        means[set_name] = []
        cis[set_name] = []
        ns[set_name] = []
        for vals in (g_list, z_list):
            mean_val, std_val, n = _mean_std_n(vals)
            ci_val = _ci95_from_std(std_val, n)
            if _isnan(mean_val):
                mean_val = 0.0
                ci_val = 0.0
            means[set_name].append(mean_val)
            cis[set_name].append(ci_val)
            ns[set_name].append(n)

    valid_sets = [s for s in args.set_names if s in means]
    if not valid_sets:
        raise SystemExit(
            "No empirical MSE CSVs found for the requested sets (need "
            "empirical_stenosis_mse_comparison.csv with geometric and stenosis_zero).\n"
            + "\n".join(missing)
        )

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    x = np.arange(len(valid_sets))
    n_mod = len(MODALITY_ORDER)
    width = 0.85 / n_mod
    offsets = np.linspace(-0.425 + width / 2, 0.425 - width / 2, n_mod)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, modality in enumerate(MODALITY_ORDER):
        y = [means[s][i] for s in valid_sets]
        yerr = [cis[s][i] for s in valid_sets]
        if modality == "stenosis_zero":
            # Red fill; hatch lines use edgecolor (pink stripes).
            bar_kwargs = {
                "facecolor": _modality_color("geometric"), #"#c62828",
                "edgecolor": "black",
                "linewidth": 0.9,
                "hatch": "//",
            }
        else:
            bar_kwargs = {
                "color": _modality_color(modality),
                "edgecolor": "black",
                "linewidth": 0.6,
            }
        ax.bar(
            x + offsets[i],
            y,
            width,
            yerr=yerr,
            capsize=2.5,
            error_kw={"color": "black", "linewidth": 0.9},
            label=MODALITY_LABEL[modality],
            **bar_kwargs,
        )

    max_top = 0.0
    for set_name in valid_sets:
        for i in range(n_mod):
            yv = means[set_name][i]
            ye = cis[set_name][i]
            max_top = max(max_top, yv + ye)

    label_fs = 14
    label_base_offset_frac = 0.015
    label_y = label_base_offset_frac * max_top if max_top > 0 else 0.0
    label_bbox = {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.5,
    }
    for set_idx, set_name in enumerate(valid_sets):
        for i, _modality in enumerate(MODALITY_ORDER):
            yv = means[set_name][i]
            txt = _format_bar_label(yv)
            if not txt or yv <= 0:
                continue
            x_pos = x[set_idx] + offsets[i]
            ax.text(
                x_pos,
                label_y,
                txt,
                ha="center",
                va="bottom",
                fontsize=label_fs,
                clip_on=True,
                bbox=label_bbox,
            )

    y_hi = max_top * 1.08 if max_top > 0 else 1.0
    if args.ylim_max is not None:
        ax.set_ylim(0.0, float(args.ylim_max))
    else:
        ax.set_ylim(0.0, max(y_hi, ax.get_ylim()[1]))
    ax.set_xticks(x)
    ax.set_xticklabels([_format_set_label(s) for s in valid_sets], fontsize=18)
    ax.set_ylabel(
        "Max. Inlet Pressure Error \n over Cardiac Cycle (MPE) (\%)",
        fontsize=18,
    )
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Leave room at top: title (figure-level) above legend above axes.
    plt.tight_layout(rect=(0, 0, 1, 0.82))
    fig.suptitle(
        "Baseline (Poiseuille) Error",
        fontsize=18,
        y=0.995,
        va="top",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        bbox_transform=fig.transFigure,
        ncol=n_mod,
        frameon=False,
        fontsize=16,
    )

    if args.output:
        out_path = (
            args.output
            if os.path.isabs(args.output)
            else os.path.join(os.getcwd(), args.output)
        )
    else:
        os.makedirs(empirical_root, exist_ok=True)
        out_name = (
            f"empirical_stenosis_cross_set_mpe_{args.geometry_variant_tag}.pdf"
        )
        out_path = os.path.join(empirical_root, out_name)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    for set_name in valid_sets:
        n0 = ns[set_name][0] if ns[set_name] else 0
        print(f"  {set_name}: n_geometries={n0} (paired geometric / stenosis_zero)")
    if missing:
        print("Skipped sets (no paired MPE data):")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
