#!/usr/bin/env python3
"""
Horizontal bar chart of maximum percent error (pressure) for the NN junction+vessel modality,
one bar per run config. Configs are subfolders under results/cross_validation/<set_name>/
(e.g. base, stenosis_off, penalty_off). Uses a display-name dictionary for axis labels.

Usage:
  python -m util.visualizations.cv_max_pct_error_by_config_barchart VMR_rigid_aorta_adults bifurcations_EL
  python -m util.visualizations.cv_max_pct_error_by_config_barchart VMR_rigid_aorta_adults bifurcations_EL --configs base stenosis_off penalty_off --output configs_comparison.pdf
"""

import argparse
import csv
import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats


# Display name for each run config (config subfolder name -> label on y-axis).
# Value can be a string (single line) or a list/tuple of strings (multiple lines, joined by newline).
# Add or override entries to customize; configs not listed use the folder name.
CONFIG_DISPLAY_NAME = {
    "base": [r"$R_{\mathrm{quad}}$ with calibrator penalty,", "asymmetric loss,", "entrance-length adjustment"],
    "stenosis_off": [r"No $R_{\mathrm{quad}}$,", "asymmetric loss,", "entrance-length adjustment"],
    "penalty_off": [r"$R_{\mathrm{quad}}$ without calibrator penalty,", "asymmetric loss", "entrance-length adjustment"],
    #"symmetric": ["No stenosis coefficient, symmetric loss"],
    "stenosis_off_symmetric": [r"No $R_{\mathrm{quad}}$,", "symmetric loss,", "entrance-length adjustment"],
    # Config with different geometry variant: "config_suffix:variant" -> data from that config folder, that variant's CSV
    "stenosis_off:bifurcations": [r"No $R_{\mathrm{quad}}$,", "asymmetric loss,", "no entrance-length adjustment"],
    "normalized": "Normalized",
    "normalized_clip": "Normalized + clip",
}


def _display_name_to_label(display_spec):
    """Convert CONFIG_DISPLAY_NAME entry to a single string; list/tuple -> newline-separated."""
    if isinstance(display_spec, (list, tuple)):
        return "\n".join(str(line) for line in display_spec)
    return str(display_spec)


PREFIX_REL = "PressureMaxRelError_"
# Modality for the metric: NN junction + vessel (Learned Junctions and Vessels)
MODALITY_COLUMN = "BloodVesselJunction_NN_plus_Vessel_NN"


def _isnan(x):
    return x != x


def _discover_configs(data_root, set_name, geometry_variant):
    """Find subfolders under results/cross_validation/<set_name>/ that contain the geometry's pressure_max_rel_error CSV."""
    base_dir = os.path.join(data_root, "cross_validation", set_name)
    if not os.path.isdir(base_dir):
        return []
    configs = []
    for name in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, name)
        if not os.path.isdir(sub):
            continue
        path = os.path.join(sub, f"{geometry_variant}_cv_summary_pressure_max_rel_error.csv")
        if os.path.isfile(path):
            configs.append(name)
    return configs


def _load_mean_max_rel_error(path, modality=MODALITY_COLUMN):
    """
    Load the 'mean' row from the pressure max rel error CSV and return the value
    for the given modality column (fraction, 0–1). Default: NN junction+vessel.
    """
    mean_frac, _, _ = _load_mean_std_n(path, modality)
    return mean_frac


def _load_mean_std_n(path, modality=MODALITY_COLUMN):
    """
    Load mean, std, and number of trials from the pressure max rel error CSV for the
    given modality. Returns (mean_frac, std_frac, n); std_frac is nan if n < 2.
    """
    col = PREFIX_REL + modality
    n = 0
    mean_frac = float("nan")
    std_frac = float("nan")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if tid in ("", "mean", "std"):
                if tid == "mean":
                    try:
                        mean_frac = float(row.get(col, float("nan")))
                    except (ValueError, TypeError):
                        pass
                elif tid == "std":
                    try:
                        std_frac = float(row.get(col, float("nan")))
                    except (ValueError, TypeError):
                        pass
                continue
            try:
                int(tid)
                n += 1
            except ValueError:
                continue
    return mean_frac, std_frac, n


def _ci95_half_width_frac(std_frac, n):
    """95% CI half-width (fraction) from std and n; uses t-distribution. Returns 0 if n < 2."""
    if n < 2 or _isnan(std_frac) or std_frac < 0:
        return 0.0
    t_crit = stats.t.ppf(0.975, n - 1)
    return t_crit * (std_frac / math.sqrt(n))


def main():
    parser = argparse.ArgumentParser(
        description="Horizontal bar chart of max percent error (pressure) by run config."
    )
    parser.add_argument("set_name", help="Set name (e.g., VMR_rigid_aorta_adults)")
    parser.add_argument(
        "geometry_variant",
        nargs="?",
        default="bifurcations_EL",
        help="Geometry variant (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Run config subfolders to include. Use 'config:variant' for a different geometry variant (e.g. stenosis_off:bifurcations). Default: auto-discover.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file (default: <set_name>_<geometry_variant>_max_pct_error_by_config.pdf)",
    )
    parser.add_argument(
        "--data-root",
        default="results",
        help="Root for results/cross_validation (default: results)",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        metavar="PCT",
        help="Upper limit for x-axis (max percent error). Labels for bars exceeding this are placed at the limit (default: auto).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    parser.add_argument(
        "--display-names",
        nargs="*",
        default=None,
        metavar="CONFIG=Label",
        help="Override display names, e.g. base='Default' stenosis_off='No stenosis'. Use __ for line break (e.g. a='Line1__Line2').",
    )
    args = parser.parse_args()

    data_root = args.data_root.rstrip(os.sep)
    set_name = args.set_name
    geometry_variant = args.geometry_variant

    # Resolve list of configs: each entry is "config_suffix" or "config_suffix:variant" (variant override for that bar)
    if args.configs:
        config_entries = [c.strip() for c in args.configs if c.strip()]
        configs = []  # list of (config_suffix, variant_to_use, display_key)
        for c in config_entries:
            if ":" in c:
                cfg, var = c.split(":", 1)
                cfg, var = cfg.strip(), var.strip()
                configs.append((cfg, var, c))  # display_key = full string for display name lookup
            else:
                configs.append((c, geometry_variant, c))
    else:
        discovered = _discover_configs(data_root, set_name, geometry_variant)
        configs = [(c, geometry_variant, c) for c in discovered]
    if not configs:
        raise SystemExit("No configs found. Specify --configs or ensure results/cross_validation/<set_name>/<config>/ exist with the geometry CSV.")

    # Display name dict: start from module default, then apply --display-names
    display_name = dict(CONFIG_DISPLAY_NAME)
    if args.display_names:
        for pair in args.display_names:
            if "=" in pair:
                k, v = pair.split("=", 1)
                # __ in value -> newline (multi-line label); single _ -> space
                v = v.strip().replace("__", "\n").replace("_", " ")
                display_name[k.strip()] = v
            else:
                display_name[pair.strip()] = pair.strip().replace("__", "\n").replace("_", " ")

    def _label_for_key(display_key):
        raw = display_name.get(display_key, display_key)
        return _display_name_to_label(raw)

    # Load mean, std, n per config; compute value (mean in %) and 95% CI half-width (in %)
    values = []
    labels = []
    ci_half_widths = []
    for config_suffix, variant_to_use, display_key in configs:
        path = os.path.join(
            data_root, "cross_validation", set_name, config_suffix,
            f"{variant_to_use}_cv_summary_pressure_max_rel_error.csv",
        )
        if not os.path.isfile(path):
            warnings.warn(f"Skipping {display_key!r}: missing {path}", stacklevel=1)
            continue

        mean_frac, std_frac, n = _load_mean_std_n(path)

        val_pct = mean_frac * 100.0 if not _isnan(mean_frac) else 0.0
        ci_half_frac = _ci95_half_width_frac(std_frac, n)
        ci_half_pct = ci_half_frac * 100.0
        values.append(val_pct)
        labels.append(_label_for_key(display_key))
        ci_half_widths.append(ci_half_pct)

    if not values:
        raise SystemExit("No config CSVs found. Check paths or run cross-validation for the requested geometry variants.")

    # Order bars from smallest to largest max percent error (smallest at top)
    triples = sorted(zip(values, labels, ci_half_widths), key=lambda p: p[0])
    values = [t[0] for t in triples]
    labels = [t[1] for t in triples]
    ci_half_widths = [t[2] for t in triples]

    # LaTeX formatting for text
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    # Horizontal bar chart: y = config labels, x = max percent error (NN junction+vessel)
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.5)))
    bars = ax.barh(
        y_pos, values, height=0.65, align="center", color="#b8b8b8", edgecolor="black", linewidth=0.5,
        xerr=ci_half_widths, capsize=2.5, error_kw={"color": "black", "linewidth": 1},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r"Max. Inlet Pressure Error over Cardiac Cycle (\%)", fontsize=12)
    x_max = args.xmax
    ax.set_xlim(0, x_max if x_max is not None else None)
    ax.set_title(r"Max. Inlet Pressure Error over Cardiac Cycle (\%) by Pipeline Configuration", fontsize=12)
    ax.invert_yaxis()  # smallest (best) at top
    ax.grid(axis="x", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Fade out clipped bars on the right when xmax is set (white overlay, alpha 0 -> 1 left to right)
    fade_n = 40
    fade_width_frac = 0.45
    if x_max is not None:
        fade_width = x_max * fade_width_frac
        strip_width = fade_width / fade_n
        for bar, val, ci in zip(bars, values, ci_half_widths):
            if bar.get_width() + ci <= x_max:
                continue
            y_lo = bar.get_y()
            bar_h = bar.get_height()
            for i in range(fade_n):
                x_left = x_max - fade_width + i * strip_width
                alpha = (i + 0.5) / fade_n  # 0 at left, 1 at right
                if alpha <= 0:
                    continue
                rect = mpatches.Rectangle(
                    (x_left, y_lo), strip_width, bar_h,
                    facecolor="white",
                    edgecolor="none",
                    alpha=alpha,
                    zorder=1.5,
                )
                ax.add_patch(rect)

    # Value at end of each bar (past the error bar); when bar exceeds x_max, place label just right of cap.
    # When xmax is set, don't clip any labels so text extending past the axis isn't cut off.
    label_offset = 0.3
    cap_label_offset = 0.8
    clip_labels = x_max is None
    for bar, val, ci in zip(bars, values, ci_half_widths):
        w = bar.get_width()
        x_text = w + ci + label_offset
        if x_max is not None and x_text > x_max:
            x_text = x_max + cap_label_offset
            ha = "left"
        else:
            ha = "left"
        label_text = rf"{val:.1f}\% $\pm$ {ci:.1f}\%"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, label_text,
                ha=ha, va="center", fontsize=12, clip_on=clip_labels)

    plt.tight_layout()
    out_path = args.output
    if not out_path:
        out_path = os.path.join(
            data_root, "cross_validation", set_name,
            f"{geometry_variant}_max_pct_error_by_config.pdf",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
