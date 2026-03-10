#!/usr/bin/env python3
"""
Bar chart of maximum percent error (pressure) per CV trial and mean across trials.
One bar per modality per trial (and per mean); modalities use the same set as the LaTeX table.
Requires matplotlib.

Usage:
  python -m util.visualizations.cv_pressure_max_pct_error_barchart VMR_rigid_aorta_adults bifurcations_EL [--output chart.png]
"""

import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from util.visualizations.plot_location_comparison import get_line_style
from util.visualizations.cv_pressure_errors_to_latex import MODALITY_DISPLAY, VAL_GEOMETRY_DISPLAY

# Same modality order as LaTeX table
MODALITY_KEYS = [
    "geometric",
    "BloodVesselJunction_NN",
    "NN_vessel",
    "BloodVesselJunction_NN_plus_Vessel_NN",
    "BloodVesselJunction",
]

# Map bar-chart modality key -> style key in plot_location_comparison (for color only)
MODALITY_STYLE_KEY = {
    "geometric": "geometric_0d",
    "BloodVesselJunction_NN": "BloodVesselJunction_NN",
    "NN_vessel": "NN_vessel",
    "BloodVesselJunction_NN_plus_Vessel_NN": "BloodVesselJunction_NN_plus_Vessel_NN",
    "BloodVesselJunction": "BloodVesselJunction",
}


def _display_to_legend_multiline(display_spec):
    """Convert MODALITY_DISPLAY entry (str or list of str) to legend label; list -> newline-separated."""
    if isinstance(display_spec, list):
        return "\n".join(display_spec)
    return display_spec


def _val_geo_to_label(val_geo):
    """Convert validation geometry id to display string (one or more lines)."""
    display = VAL_GEOMETRY_DISPLAY.get(val_geo, val_geo)
    if isinstance(display, list):
        return "\n".join(str(x) for x in display)
    return str(display)


PREFIX_REL = "PressureMaxRelError_"


def _isnan(x):
    return x != x


def load_csv_column(path, prefix, modality):
    """Load one metric column from a CV summary CSV. Returns list of values (one per trial)."""
    col = f"{prefix}{modality}"
    out = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_id = row.get("trial_id", "").strip()
            if trial_id in ("", "mean", "std"):
                break
            val = row.get(col, "")
            try:
                out.append(float(val))
            except (ValueError, TypeError):
                out.append(float("nan"))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Bar chart of max percent error per trial and mean, by modality."
    )
    parser.add_argument("set_name", help="Set name (e.g., VMR_rigid_aorta_adults)")
    parser.add_argument(
        "geometry_variant",
        nargs="?",
        default="bifurcations_EL",
        help="Geometry variant (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file (default: <set_name>_<geometry_variant>_max_pct_error.png)",
    )
    parser.add_argument(
        "--data-root",
        default="results",
        help="Root for results/cross_validation (default: results)",
    )
    parser.add_argument(
        "--run-config",
        default="base",
        help="Run config subfolder (default: base). Use e.g. stenosis_off or normalized_clip for other configs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    args = parser.parse_args()

    run_config = (args.run_config or "base").strip()
    out_dir = os.path.join(args.data_root, "cross_validation", args.set_name, run_config)
    base = os.path.join(out_dir, f"{args.geometry_variant}_cv_summary")
    path_rel = base + "_pressure_max_rel_error.csv"
    if not os.path.exists(path_rel):
        raise FileNotFoundError(f"Missing CSV: {path_rel}")

    # Load trial ids, validation geometry ids, and per-modality rel max error (fraction)
    trial_rows = []  # (trial_id, val_geometries)
    with open(path_rel, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if tid in ("", "mean", "std"):
                break
            val_geo = row.get("val_geometries", "").strip()
            trial_rows.append((tid, val_geo))
    trial_ids = [tid for tid, _ in trial_rows]

    by_mod = {}
    for mod in MODALITY_KEYS:
        by_mod[mod] = load_csv_column(path_rel, PREFIX_REL, mod)

    # X labels: "Trial N" with validation geometry display name(s) below; last is "Mean"
    x_labels = []
    for tid, val_geo in trial_rows:
        geo_label = _val_geo_to_label(val_geo)
        x_labels.append(f"Trial {tid}\n{geo_label}")
    x_labels.append("Mean")
    n_groups = len(x_labels)
    x = np.arange(n_groups)
    n_mods = len(MODALITY_KEYS)
    width = 0.8 / n_mods  # bar width so all bars fit in 0.8
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_mods)

    # Use LaTeX for text
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.2), 5))
    for i, mod in enumerate(MODALITY_KEYS):
        vals_frac = by_mod[mod]
        mean_val = statistics.mean([v for v in vals_frac if not _isnan(v)]) if vals_frac else float("nan")
        vals_frac = list(vals_frac) + [mean_val]
        vals_pct = [v * 100 if not _isnan(v) else 0 for v in vals_frac]
        style_key = MODALITY_STYLE_KEY.get(mod, mod)
        style = get_line_style(style_key)
        color = style["color"]
        label = _display_to_legend_multiline(MODALITY_DISPLAY.get(mod, mod))
        ax.bar(
            x + offsets[i],
            vals_pct,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_ylabel(r"Max. Relative Error over Cardiac Cycle (\%)", fontsize=12)
    ax.set_xlabel(r"Cross Validation\ Trial", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5, ha="center")
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)
    # No border (spines) around the chart
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Legend: larger font, multi-line entries (as in table), one row at top
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        ncol=len(MODALITY_KEYS),
        bbox_to_anchor=(0.5, 0.98),
        fontsize=12,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    #plt.tight_layout(rect=(0, 0.1, 1, 0.88))
    out_path = args.output
    if not out_path:
        out_path = os.path.join(out_dir, f"{args.geometry_variant}_max_pct_error.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
