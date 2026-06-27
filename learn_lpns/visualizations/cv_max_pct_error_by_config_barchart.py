#!/usr/bin/env python3
"""
Horizontal bar chart of maximum percent error (pressure) for the NN junction+vessel modality.
For each run config (subfolder under results/cross_validation/<set_name>/), plots one or more
horizontal bars: one bar per set name when multiple sets are given (grouped by config on the y-axis).

Usage:
  python -m learn_lpns.visualizations.cv_max_pct_error_by_config_barchart VMR_rigid_aorta_adults
  python -m learn_lpns.visualizations.cv_max_pct_error_by_config_barchart VMR_rigid_aorta_adults \\
    --geometry bifurcations_EL
  python -m learn_lpns.visualizations.cv_max_pct_error_by_config_barchart \\
    VMR_rigid_aorta_adults VMR_abdo VMR_pulmo_healthy
  python -m learn_lpns.visualizations.cv_max_pct_error_by_config_barchart VMR_rigid_aorta_adults \\
    --configs base gen_loss --output configs_comparison.pdf
  python -m learn_lpns.visualizations.cv_max_pct_error_by_config_barchart --bar_thickness_scale 1.3

Set DEFAULT_SET_NAMES / DEFAULT_RUN_CONFIGS below to avoid repeating long CLI lists.
"""

import argparse
import csv
import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from learn_lpns.config import get_pipeline_config
from learn_lpns.visualizations.matplotlib_tex import configure_matplotlib_latex, plot_label
from learn_lpns.zerod_calibration.modality_paths import read_cv_metric_from_row

# Display name for each run config (config subfolder name -> label on y-axis).
# Value can be a string (single line) or a list/tuple of strings (multiple lines, joined by newline).
# Add or override entries to customize; configs not listed use the folder name.
CONFIG_DISPLAY_NAME = {
    "base": [r"No $R_{\mathrm{quad}}$ (RI),", "symmetric loss,", "entrance-length adjustment"],
    "gen_loss": [
        r"No $R_{\mathrm{quad}}$ (RI),",
        "symmetric loss,",
        "proximity-weighted NN loss,",
        "entrance-length adjustment",
    ],
    "gen_loss:bifurcations": [
        r"No $R_{\mathrm{quad}}$ (RI),",
        "proximity-weighted NN loss,",
        "no entrance-length adjustment",
    ],
    "quadratic_resistor_gen_loss": [
        r"$R_{\mathrm{quad}}$ (RRI),",
        "no calibrator penalty,",
        "symmetric loss,",
        "proximity-weighted NN loss,",
        "entrance-length adjustment",
    ],
    "quadratic_resistor_penalty_on_gen_loss": [
        r"$R_{\mathrm{quad}}$ (RRI),",
        "calibrator penalty on,",
        "symmetric loss,",
        "proximity-weighted NN loss,",
        "entrance-length adjustment",
    ],
    "asymmetric_loss": [
        r"No $R_{\mathrm{quad}}$ (RI),",
        "asymmetric loss,",
        "entrance-length adjustment",
    ],
    "asymmetric_loss_gen_loss": [
        r"No $R_{\mathrm{quad}}$ (RI),",
        "asymmetric loss,",
        "proximity-weighted NN loss,",
        "entrance-length adjustment",
    ],
}

# Optional: CV set names when no set names are passed on the command line.
# None = require at least one set name as a positional argument.
DEFAULT_SET_NAMES = None

# Optional: run config subfolders to plot when --configs is not passed on the command line.
# None = auto-discover every config under results/cross_validation/<first_set_name>/ that has the geometry CSV.
# Non-None = use this list in order (same syntax as --configs: "config" or "config:variant").
DEFAULT_RUN_CONFIGS = None
# Example:
DEFAULT_RUN_CONFIGS = [
    "gen_loss",
    "base",
    "quadratic_resistor_gen_loss",
    "quadratic_resistor_penalty_on_gen_loss",
    "gen_loss:bifurcations",
]

def _format_set_label(set_name):
    """Same rules as learn_lpns.visualizations.cv_cross_set_summary_barchart."""
    return get_pipeline_config().cohorts.format_display_label(set_name)


# Bar colors per CV set when multiple sets are plotted (cycles if more than four).
SET_SERIES_COLORS = ["hotpink", "turquoise", "darkorange", "silver"]

# Matplotlib text size for this figure (ticks, labels, legend, bar-end annotations).
PLOT_FONT_SIZE = 18


def _display_name_to_label(display_spec):
    """Convert CONFIG_DISPLAY_NAME entry to a single string; list/tuple -> newline-separated."""
    if isinstance(display_spec, list | tuple):
        return "\n".join(str(line) for line in display_spec)
    return str(display_spec)


PREFIX_REL = "PressureMaxRelError_"
# Modality for the metric: NN junction + vessel (Learned Junctions and Vessels)
MODALITY_COLUMN = "BloodVesselJunction_NN_plus_Vessel_NN"


def _isnan(x):
    return x != x


def _discover_configs(data_root, set_name, geometry_variant):
    """Find subfolders under results/cross_validation/<set_name>/ with pressure_max_rel_error CSV."""
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
    mean_frac = float("nan")
    std_frac = float("nan")
    n = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if tid in ("", "mean", "std"):
                if tid == "mean":
                    try:
                        mean_frac = float(read_cv_metric_from_row(row, PREFIX_REL, modality) or float("nan"))
                    except (ValueError, TypeError):
                        pass
                elif tid == "std":
                    try:
                        std_frac = float(read_cv_metric_from_row(row, PREFIX_REL, modality) or float("nan"))
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
        description=(
            "Horizontal bar chart of max percent error (pressure) by run config "
            "(one bar per CV set when multiple sets are given)."
        )
    )
    parser.add_argument(
        "set_names",
        nargs="*",
        default=None,
        metavar="SET_NAME",
        help=(
            "CV set name(s) under results/cross_validation/ (e.g. VMR_rigid_aorta_adults VMR_abdo). "
            "If omitted, uses cohorts.default_cv_set_names from config/defaults.yaml."
        ),
    )
    parser.add_argument(
        "--geometry",
        "-g",
        default="bifurcations_EL",
        help="Geometry variant for CSV filenames (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help=(
            "Run config subfolders to include. Use 'config:variant' for a different geometry variant "
            "(e.g. gen_loss:bifurcations). If omitted, uses DEFAULT_RUN_CONFIGS in this module "
            "when set, else auto-discover."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file (default: under first set's cross_validation folder, name includes geometry and set count)",
    )
    parser.add_argument(
        "--data_root",
        default="results",
        help="Root for results/cross_validation (default: results)",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Upper limit for x-axis (max percent error). "
            "Labels for bars exceeding this are placed at the limit (default: auto)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    parser.add_argument(
        "--display_names",
        nargs="*",
        default=None,
        metavar="CONFIG=Label",
        help=(
            "Override display names, e.g. base='Default' gen_loss='RI + gen loss'. "
            "Use __ for line break (e.g. a='Line1__Line2')."
        ),
    )
    parser.add_argument(
        "--bar_thickness_scale",
        type=float,
        default=3.0,
        metavar="S",
        help=(
            "Multiply bar thickness (default: 1). E.g. 1.3 for thicker bars. "
            "Very large values can make grouped bars overlap between rows."
        ),
    )
    args = parser.parse_args()

    data_root = args.data_root.rstrip(os.sep)
    if args.set_names:
        set_names = [s.strip() for s in args.set_names if s.strip()]
    elif DEFAULT_SET_NAMES is not None:
        set_names = [str(s).strip() for s in DEFAULT_SET_NAMES if str(s).strip()]
    else:
        set_names = list(get_pipeline_config().cohorts.default_cv_set_names)
    if not set_names:
        raise SystemExit("Provide at least one set name (positional), or configure cohorts.default_cv_set_names.")
    geometry_variant = args.geometry

    # Resolve list of configs: each entry is "config_suffix" or "config_suffix:variant" (variant override for that bar)
    if args.configs:
        config_entries = [c.strip() for c in args.configs if c.strip()]
    elif DEFAULT_RUN_CONFIGS is not None:
        config_entries = [str(c).strip() for c in DEFAULT_RUN_CONFIGS if str(c).strip()]
    else:
        config_entries = None

    if config_entries:
        configs = []  # list of (config_suffix, variant_to_use, display_key)
        for c in config_entries:
            if ":" in c:
                cfg, var = c.split(":", 1)
                cfg, var = cfg.strip(), var.strip()
                configs.append((cfg, var, c))  # display_key = full string for display name lookup
            else:
                configs.append((c, geometry_variant, c))
    else:
        discovered = _discover_configs(data_root, set_names[0], geometry_variant)
        configs = [(c, geometry_variant, c) for c in discovered]
    if not configs:
        raise SystemExit(
            "No configs found. Specify --configs or ensure results/cross_validation/<set_name>/<config>/ exist "
            "with the geometry CSV (discovery uses the first set name)."
        )

    # Display name dict: start from module default, then apply --display_names
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

    # Load mean, std, n per (config, set); values[ci, si], ci_half_widths[ci, si]
    n_sets = len(set_names)
    rows = []
    for config_suffix, variant_to_use, display_key in configs:
        vals = []
        cis = []
        for sn in set_names:
            path = os.path.join(
                data_root,
                "cross_validation",
                sn,
                config_suffix,
                f"{variant_to_use}_cv_summary_pressure_max_rel_error.csv",
            )
            if not os.path.isfile(path):
                warnings.warn(
                    f"Missing {path!r} for config {display_key!r}; treating as NaN for that set.",
                    stacklevel=1,
                )
                vals.append(float("nan"))
                cis.append(float("nan"))
                continue
            mean_frac, std_frac, n = _load_mean_std_n(path)
            val_pct = mean_frac * 100.0 if not _isnan(mean_frac) else float("nan")
            ci_half_frac = _ci95_half_width_frac(std_frac, n)
            ci_half_pct = ci_half_frac * 100.0 if not _isnan(mean_frac) else float("nan")
            vals.append(val_pct)
            cis.append(ci_half_pct)
        if all(_isnan(v) for v in vals):
            warnings.warn(f"Skipping config {display_key!r}: no data for any set.", stacklevel=1)
            continue
        rows.append(
            {
                "display_key": display_key,
                "label": _label_for_key(display_key),
                "values": vals,
                "ci_half": cis,
            }
        )

    if not rows:
        raise SystemExit(
            "No config CSVs found. Check paths or run cross-validation for the requested geometry variants."
        )

    def _row_sort_key(r):
        xs = [v for v in r["values"] if not _isnan(v)]
        if not xs:
            return float("inf")
        return float(np.nanmean(xs))

    rows.sort(key=_row_sort_key)
    config_labels = [r["label"] for r in rows]
    values = np.array([r["values"] for r in rows], dtype=float)
    ci_half_widths = np.array([r["ci_half"] for r in rows], dtype=float)

    use_latex = configure_matplotlib_latex(plt)
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
        }
    )

    n_configs = len(config_labels)
    # Horizontal bar chart: y = config labels, x = max percent error (NN junction+vessel); grouped by set.
    # bar_height: nominal step between bar centers within a group; patch height = bar_height * BAR_HEIGHT_INSET.
    th = max(0.05, float(args.bar_thickness_scale))
    BAR_HEIGHT_SINGLE_BASE = 0.65
    BAR_HEIGHT_MULTI_CAP = 0.22
    BAR_GAP_MULT = 1.08  # spacing between bar centers within a config group
    BAR_HEIGHT_INSET = 0.92
    ROW_GROUP_GAP = 0.08  # extra y-units between config groups (scaled with bar_height)

    if n_sets == 1:
        bar_height = BAR_HEIGHT_SINGLE_BASE * th
        offsets = np.array([0.0])
        colors = ["silver"]
    else:
        bar_height = min(BAR_HEIGHT_SINGLE_BASE / max(n_sets, 1), BAR_HEIGHT_MULTI_CAP) * th
        offsets = (np.arange(n_sets) - (n_sets - 1) / 2.0) * (bar_height * BAR_GAP_MULT)
        n_palette = len(SET_SERIES_COLORS)
        colors = [SET_SERIES_COLORS[i % n_palette] for i in range(n_sets)]

    patch_h = bar_height * BAR_HEIGHT_INSET
    off_min = float(offsets.min())
    off_max = float(offsets.max())
    # Vertical span of one config row (all sets): offset spread + drawn bar height
    group_extent = (off_max - off_min) + patch_h
    row_pitch = group_extent + max(0.12 * bar_height, ROW_GROUP_GAP * th)

    y_centers = np.arange(n_configs, dtype=float) * row_pitch

    # Figure height scales with total y span in data coordinates
    y_span = (n_configs - 1) * row_pitch + group_extent if n_configs else group_extent
    fig_w = 12 if n_sets > 1 else 7.0
    fig_h = max(4.0, 0.5 * y_span + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # (patch, value, ci half-width) for fade overlay when xmax is set
    bar_value_ci = []
    for s in range(n_sets):
        y = y_centers + offsets[s]
        v_col = values[:, s]
        ci_col = ci_half_widths[:, s]
        valid = ~np.isnan(v_col)
        if not np.any(valid):
            continue
        bars = ax.barh(
            y[valid],
            v_col[valid],
            height=patch_h,
            align="center",
            color=colors[s],
            edgecolor="black",
            linewidth=0.5,
            xerr=ci_col[valid],
            capsize=2.5 if n_sets == 1 else 2.0,
            error_kw={"color": "black", "linewidth": 1},
            label=_format_set_label(set_names[s]),
        )
        for rect, vv, cih in zip(bars, v_col[valid], ci_col[valid], strict=False):
            ci_plot = 0.0 if _isnan(cih) else cih
            bar_value_ci.append((rect, vv, ci_plot))

    ax.set_yticks(y_centers)
    ax.set_yticklabels(config_labels, fontsize=PLOT_FONT_SIZE)
    # Y limits: include full bar geometry so thickened bars do not overlap between groups
    y_mins = [c * row_pitch + off_min - patch_h / 2.0 for c in range(n_configs)]
    y_maxs = [c * row_pitch + off_max + patch_h / 2.0 for c in range(n_configs)]
    y_lim_lo = min(y_mins) - 0.15 * row_pitch
    y_lim_hi = max(y_maxs) + 0.15 * row_pitch
    ax.set_ylim(y_lim_lo, y_lim_hi)

    ax.set_xlabel(
        plot_label(
            r"Max. Inlet Pressure Error over Cardiac Cycle (MPE) (\%)",
            use_latex=use_latex,
        ),
        fontsize=PLOT_FONT_SIZE,
    )
    x_max = args.xmax
    ax.set_xlim(0, x_max if x_max is not None else None)
    # ax.set_title(
    #     r"MPE by Pipeline Configuration",
    #     fontsize=PLOT_FONT_SIZE,
    # )
    ax.invert_yaxis()  # smallest (best) at top
    ax.grid(axis="x", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n_sets > 1:
        ax.legend(loc="upper right", fontsize=PLOT_FONT_SIZE, framealpha=0)

    # Fade out clipped bars on the right when xmax is set (white overlay, alpha 0 -> 1 left to right)
    fade_n = 40
    fade_width_frac = 0.45
    if x_max is not None:
        fade_width = x_max * fade_width_frac
        strip_width = fade_width / fade_n
        for bar, val, ci in bar_value_ci:
            if val + ci <= x_max:
                continue
            y_lo = bar.get_y()
            bar_h = bar.get_height()
            for i in range(fade_n):
                x_left = x_max - fade_width + i * strip_width
                alpha = (i + 0.5) / fade_n  # 0 at left, 1 at right
                if alpha <= 0:
                    continue
                rect = mpatches.Rectangle(
                    (x_left, y_lo),
                    strip_width,
                    bar_h,
                    facecolor="white",
                    edgecolor="none",
                    alpha=alpha,
                    zorder=1.5,
                )
                ax.add_patch(rect)

    # Value at end of each bar (past the error bar); when bar exceeds x_max, place label just right of cap.
    label_offset = 0.3
    cap_label_offset = 0.8
    clip_labels = x_max is None
    for s in range(n_sets):
        y = y_centers + offsets[s]
        for i in range(n_configs):
            val = values[i, s]
            ci = ci_half_widths[i, s]
            if _isnan(val):
                continue
            x_text = val + (0.0 if _isnan(ci) else ci) + label_offset
            if x_max is not None and x_text > x_max:
                x_text = x_max + cap_label_offset
            ha = "left"
            ci_show = 0.0 if _isnan(ci) else ci
            label_text = rf"{val:.1f}\% $\pm$ {ci_show:.1f}\%"
            ax.text(
                x_text,
                y[i],
                label_text,
                ha=ha,
                va="center",
                fontsize=PLOT_FONT_SIZE,
                clip_on=clip_labels,
            )

    plt.tight_layout()
    out_path = args.output
    if not out_path:
        if len(set_names) == 1:
            out_path = os.path.join(
                data_root,
                "cross_validation",
                set_names[0],
                f"{geometry_variant}_max_pct_error_by_config.pdf",
            )
        else:
            set_slug = "__".join(set_names)
            out_path = os.path.join(
                data_root,
                "cross_validation",
                set_names[0],
                f"{geometry_variant}_max_pct_error_by_config_{len(set_names)}sets__{set_slug}.pdf",
            )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
