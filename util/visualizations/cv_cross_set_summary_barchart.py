#!/usr/bin/env python3
"""
Grouped bar chart of cross-validation averages across multiple set names.

For each set_name, plots the average CV metric for:
  - baseline / poiseuille
  - learned junctions
  - learned vessels
  - learned vessels and junctions
  - optimal

Also shows a 95% confidence interval computed from standard deviation:
    CI95 = 1.96 * std / sqrt(n_trials)

Usage:
  python -m util.visualizations.cv_cross_set_summary_barchart
  python -m util.visualizations.cv_cross_set_summary_barchart --run_config gen_loss
  python -m util.visualizations.cv_cross_set_summary_barchart --metric pressure_max_error
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from util.visualizations.plot_location_comparison import get_line_style
from util.visualizations.matplotlib_tex import configure_matplotlib_latex, plot_label


SET_NAMES_DEFAULT = [
    "VMR_rigid_aorta_adults_all",
    "VMR_abdo",
    "VMR_pulmo_healthy",
    "VMR_all"
]

# X-axis labels for each set_name (internal folder name -> plot text). Use "\n" for a line break.
# Names not listed here fall back to _format_set_label default (VMR_… split or raw set_name).
SET_DISPLAY_NAME = {
    "VMR_rigid_aorta_adults_all": "Aortic",
    #"VMR_rigid_aorta_adults": "Rigid aorta\n(adults)",
    "VMR_abdo": "Aortofemoral ",
    "VMR_pulmo": "Pulmonary",
    "VMR_pulmo_healthy": "Pulmonary",
    "VMR_all": "All",
    "VMR_all_balanced": "Mixed"
}

RUN_CONFIG_FALLBACK_ORDER = [
    "gen_loss",
    "base",
    "quadratic_resistor_gen_loss",
    "quadratic_resistor_penalty_on_gen_loss",
    "asymmetric_loss_gen_loss",
    "asymmetric_loss",
]

MODALITY_ORDER = [
    "geometric",
    "NN_vessel",
    "BloodVesselJunction_NN",
    "BloodVesselJunction_NN_plus_Vessel_NN",
    "BloodVesselJunction",
]

MODALITY_LABEL = {
    "geometric": ["Baseline", "(Poiseuille)"],
    "NN_vessel": ["Learned", "Vessels"],
    "BloodVesselJunction_NN": ["Learned", "Junctions"],
    "BloodVesselJunction_NN_plus_Vessel_NN": ["Learned", "Junctions", "and Vessels"],
    "BloodVesselJunction": ["Optimal", "(Fit to 3D)"],
}


def _multiline_label(display_spec):
    """Convert a label spec (str or sequence of lines) to a newline-separated string."""
    if isinstance(display_spec, (list, tuple)):
        return "\n".join(str(line) for line in display_spec)
    return str(display_spec)

METRIC_CONFIG = {
    "pressure_max_rel_error": {
        "csv_suffix": "_pressure_max_rel_error.csv",
        "col_prefix": "PressureMaxRelError_",
        "scale": 100.0,
        "ylabel": r"Max. Inlet Pressure Error over Cardiac Cycle (MPE) (\%)",
        "out_suffix": "pressure_max_rel_error",
    },
    "pressure_max_error": {
        "csv_suffix": "_pressure_max_error.csv",
        "col_prefix": "PressureMaxError_",
        "scale": 1.0,
        "ylabel": r"Max. Inlet Pressure Error over Cardiac Cycle (mmHg)",
        "out_suffix": "pressure_max_error",
    },
    "pressure_mse": {
        "csv_suffix": "_pressure_mse.csv",
        "col_prefix": "PressureMSE_",
        "scale": 1.0,
        "ylabel": r"Inlet Pressure MSE over Cardiac Cycle (mmHg$^2$)",
        "out_suffix": "pressure_mse",
    },
}


def _isnan(x):
    return x != x


def _load_trial_values(csv_path, col_name):
    """Return list of trial-level metric values from the requested column."""
    values = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_id = (row.get("trial_id") or "").strip()
            if trial_id in ("", "mean", "std"):
                break
            try:
                values.append(float(row.get(col_name, "")))
            except (TypeError, ValueError):
                values.append(float("nan"))
    return values


def _mean_std_n(values):
    clean = [v for v in values if not _isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean, ddof=1)) if n >= 2 else 0.0
    return mean_val, std_val, n


def _ci95_from_std(std_val, n):
    if n <= 1 or _isnan(std_val):
        return 0.0
    return 1.96 * std_val / math.sqrt(n)


def _modality_color(modality):
    style_key = "geometric_0d" if modality == "geometric" else modality
    return get_line_style(style_key).get("color", "gray")


def _format_set_label(set_name):
    if set_name in SET_DISPLAY_NAME:
        return SET_DISPLAY_NAME[set_name]
    if set_name.startswith("VMR_"):
        return set_name.replace("VMR_", "VMR\n", 1)
    return set_name


def _format_bar_label(mean, metric_key):
    """Text above each bar: mean only (95% CI remains on the error bars)."""
    if not np.isfinite(mean):
        return ""
    # text.usetex=True: plain % is a LaTeX comment; use \\%
    if metric_key == "pressure_max_rel_error":
        return f"{mean:.1f}"
    if metric_key == "pressure_max_error":
        return f"{mean:.1f}"
    return f"{mean:.3f}"


def _resolve_csv_path(data_root, set_name, run_config, geometry_variant, csv_suffix, allow_fallback):
    requested_path = os.path.join(
        data_root,
        "cross_validation",
        set_name,
        run_config,
        f"{geometry_variant}_cv_summary{csv_suffix}",
    )
    if os.path.isfile(requested_path):
        return requested_path, run_config
    if not allow_fallback:
        return None, None

    candidates = [run_config] + [c for c in RUN_CONFIG_FALLBACK_ORDER if c != run_config]
    for cfg in candidates:
        p = os.path.join(
            data_root,
            "cross_validation",
            set_name,
            cfg,
            f"{geometry_variant}_cv_summary{csv_suffix}",
        )
        if os.path.isfile(p):
            return p, cfg
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Grouped bar chart: CV averages across set names, with 95% CI."
    )
    parser.add_argument(
        "--set_names",
        nargs="+",
        default=SET_NAMES_DEFAULT,
        help="Set names to include (default: VMR_rigid_aorta_adults_all VMR_abdo VMR_pulmo)",
    )
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations_EL",
        help="Geometry variant in CV summary filename (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--run_config",
        default="gen_loss",
        help="Run config subfolder under results/cross_validation/<set_name>/ (default: gen_loss)",
    )
    parser.add_argument(
        "--allow_config_fallback",
        action="store_true",
        help="If a set is missing --run_config, try common config fallbacks for that set.",
    )
    parser.add_argument(
        "--metric",
        default="pressure_max_rel_error",
        choices=list(METRIC_CONFIG.keys()),
        help="Metric to plot (default: pressure_max_rel_error)",
    )
    parser.add_argument(
        "--data_root",
        default="results",
        help="Root containing cross_validation folder (default: results)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: auto path in results/cross_validation/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI (default: 180)",
    )
    parser.add_argument(
        "--ytick_fontsize",
        type=float,
        default=14,
        metavar="PT",
        help="Font size (pt) for y-axis tick labels (default: 9)",
    )
    args = parser.parse_args()

    mcfg = METRIC_CONFIG[args.metric]
    means = {}
    cis = {}
    missing = []

    for set_name in args.set_names:
        csv_path, used_config = _resolve_csv_path(
            args.data_root,
            set_name,
            args.run_config,
            args.geometry_variant,
            mcfg["csv_suffix"],
            args.allow_config_fallback,
        )
        if not csv_path:
            missing.append(
                os.path.join(
                    args.data_root,
                    "cross_validation",
                    set_name,
                    args.run_config,
                    f"{args.geometry_variant}_cv_summary{mcfg['csv_suffix']}",
                )
            )
            continue
        print(f"Using {set_name}: run-config={used_config} ({csv_path})")

        means[set_name] = []
        cis[set_name] = []
        for modality in MODALITY_ORDER:
            col_name = f"{mcfg['col_prefix']}{modality}"
            values = _load_trial_values(csv_path, col_name)
            mean_val, std_val, n = _mean_std_n(values)
            ci_val = _ci95_from_std(std_val, n)
            if _isnan(mean_val):
                mean_val = 0.0
                ci_val = 0.0
            means[set_name].append(mean_val * mcfg["scale"])
            cis[set_name].append(ci_val * mcfg["scale"])

    valid_sets = [s for s in args.set_names if s in means]
    if not valid_sets:
        raise SystemExit(
            "No valid CSV files were found for the requested set names.\n"
            + "\n".join(missing)
        )

    use_latex = configure_matplotlib_latex(plt)

    x = np.arange(len(valid_sets))
    n_mod = len(MODALITY_ORDER)
    width = 0.85 / n_mod
    offsets = np.linspace(-0.425 + width / 2, 0.425 - width / 2, n_mod)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, modality in enumerate(MODALITY_ORDER):
        y = [means[s][i] for s in valid_sets]
        yerr = [cis[s][i] for s in valid_sets]
        ax.bar(
            x + offsets[i],
            y,
            width,
            yerr=yerr,
            capsize=2.5,
            color=_modality_color(modality),
            edgecolor="black",
            linewidth=0.6,
            error_kw={"color": "black", "linewidth": 0.9},
            label=_multiline_label(MODALITY_LABEL[modality]),
        )

    # Max height (bar + error) for y-axis margin
    max_top = 0.0
    for set_name in valid_sets:
        for i in range(len(MODALITY_ORDER)):
            yv = means[set_name][i]
            ye = cis[set_name][i]
            max_top = max(max_top, yv + ye)

    # Labels: same data-y for every bar — small fixed offset above y=0, scaled only by chart range
    label_fs = 14
    label_base_offset_frac = 0.015  # fraction of max(bar+error) used as y offset from axis base
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
            txt = _format_bar_label(yv, args.metric)
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

    y_hi_auto = ax.get_ylim()[1]
    #ax.set_ylim(0.0, max(y_hi_auto, max_top * 1.06))
    ax.set_ylim(0, 37)

    ax.set_xticks(x)
    ax.set_xticklabels([_format_set_label(s) for s in valid_sets], fontsize=18)
    ax.set_ylabel(plot_label(mcfg["ylabel"], use_latex=use_latex), fontsize=18)
    #ax.set_xlabel("Set Name", fontsize=16)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Column headers aligned with bar positions (same x offsets as bar labels below).
    header_y = 1.04
    header_fs = 10
    x_center = float(np.mean(x))
    for i, modality in enumerate(MODALITY_ORDER):
        ax.text(
            x_center + offsets[i],
            header_y,
            _multiline_label(MODALITY_LABEL[modality]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=header_fs,
            color=_modality_color(modality),
            linespacing=0.85,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.86))

    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(args.data_root, "cross_validation")
        os.makedirs(out_dir, exist_ok=True)
        out_name = (
            f"cv_cross_set_summary_{args.run_config}_{args.geometry_variant}_{mcfg['out_suffix']}.pdf"
        )
        out_path = os.path.join(out_dir, out_name)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    if missing:
        print("Skipped missing CSV(s):")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
