#!/usr/bin/env python3
"""
Bar chart of a pressure error metric per CV trial and mean across trials.
One bar per modality per trial (and per mean); modalities use the same set as the LaTeX table.
Requires matplotlib.

Supported metrics (--metric):
  all                     (default) – generate all three plots below
  pressure_max_rel_error  – max relative error, displayed as %
  pressure_max_error      – max absolute error (mmHg)
  pressure_mse            – mean squared error (mmHg²)

Usage:
  python -m learn_lpns.visualizations.cv_pressure_max_pct_error_barchart VMR_rigid_aorta_adults bifurcations_EL
  python -m learn_lpns.visualizations.cv_pressure_max_pct_error_barchart \\
    VMR_rigid_aorta_adults bifurcations_EL --metric pressure_max_error
  python -m learn_lpns.visualizations.cv_pressure_max_pct_error_barchart --all_sets
  python -m learn_lpns.visualizations.cv_pressure_max_pct_error_barchart --all_sets bifurcations_EL
"""

import argparse
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from learn_lpns.visualizations.cv_pressure_errors_to_latex import MODALITY_DISPLAY, VAL_GEOMETRY_DISPLAY
from learn_lpns.visualizations.matplotlib_tex import configure_matplotlib_latex, plot_label
from learn_lpns.visualizations.plot_location_comparison import get_line_style
from learn_lpns.zerod_calibration.modality_paths import (
    DEFAULT_MODALITY_ORDER,
    read_cv_metric_from_row,
)
from learn_lpns.zerod_calibration.run_config_canonical import (
    discover_run_config_suffixes,
    resolve_run_config_suffix,
)

# Bar / legend order (matches MSE / LaTeX table column order)
MODALITY_KEYS = list(DEFAULT_MODALITY_ORDER)

# Map bar-chart modality key -> style key in plot_location_comparison (for color only)
MODALITY_STYLE_KEY = {
    "geometric": "geometric_0d",
    "BloodVesselJunction_NN": "BloodVesselJunction_NN",
    "Vessel_NN": "Vessel_NN",
    "BloodVesselJunction_NN_plus_Vessel_NN": "BloodVesselJunction_NN_plus_Vessel_NN",
    "BloodVesselJunction": "BloodVesselJunction",
}

METRIC_CONFIG = {
    "pressure_max_rel_error": {
        "csv_suffix": "_pressure_max_rel_error.csv",
        "col_prefix": "PressureMaxRelError_",
        "scale": 100.0,
        "ylabel": r"Max. Inlet Pressure Error over Cardiac Cycle (\%)",
        "out_suffix": "_max_pct_error.pdf",
    },
    "pressure_max_error": {
        "csv_suffix": "_pressure_max_error.csv",
        "col_prefix": "PressureMaxError_",
        "scale": 1.0,
        "ylabel": r"Max. Inlet Pressure Error over Cardiac Cycle (mmHg)",
        "out_suffix": "_max_error.pdf",
    },
    "pressure_mse": {
        "csv_suffix": "_pressure_mse.csv",
        "col_prefix": "PressureMSE_",
        "scale": 1.0,
        "ylabel": r"Inlet Pressure MSE over Cardiac Cycle (mmHg$^2$)",
        "out_suffix": "_mse.pdf",
    },
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


def _isnan(x):
    return x != x


def discover_cross_validation_set_names(data_root):
    """List subdirectory names under ``<data_root>/cross_validation`` (one per dataset)."""
    cv_root = os.path.join(data_root, "cross_validation")
    if not os.path.isdir(cv_root):
        return []
    return sorted(d for d in os.listdir(cv_root) if os.path.isdir(os.path.join(cv_root, d)) and not d.startswith("."))


def load_csv_column(path, prefix, modality):
    """Load one metric column from a CV summary CSV. Returns list of values (one per trial)."""
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_id = row.get("trial_id", "").strip()
            if trial_id in ("", "mean", "std"):
                break
            val = read_cv_metric_from_row(row, prefix, modality)
            try:
                out.append(float(val))
            except (ValueError, TypeError):
                out.append(float("nan"))
    return out


def generate_bar_chart(metric_key, out_dir, geometry_variant, output_path=None, dpi=150):
    """Generate a single bar chart for the given metric and save it to disk."""
    mcfg = METRIC_CONFIG[metric_key]
    scale = mcfg["scale"]

    base = os.path.join(out_dir, f"{geometry_variant}_cv_summary")
    csv_path = base + mcfg["csv_suffix"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    trial_rows = []  # (trial_id, val_geometries)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if tid in ("", "mean", "std"):
                break
            val_geo = row.get("val_geometries", "").strip()
            trial_rows.append((tid, val_geo))

    by_mod = {}
    for mod in MODALITY_KEYS:
        by_mod[mod] = load_csv_column(csv_path, mcfg["col_prefix"], mod)

    x_labels = []
    for tid, val_geo in trial_rows:
        geo_label = _val_geo_to_label(val_geo)
        x_labels.append(f"Trial {tid}\n{geo_label}")
    x_labels.append("Average")
    n_groups = len(x_labels)
    x = np.arange(n_groups)
    n_mods = len(MODALITY_KEYS)
    width = 0.8 / n_mods
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_mods)

    use_latex = configure_matplotlib_latex(plt)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.2), 5))
    for i, mod in enumerate(MODALITY_KEYS):
        raw_vals = by_mod[mod]
        clean = [v for v in raw_vals if not _isnan(v)]
        mean_val = statistics.mean(clean) if clean else float("nan")
        std_val = statistics.stdev(clean) if len(clean) >= 2 else (0.0 if clean else float("nan"))
        raw_vals = [*list(raw_vals), mean_val]
        plot_vals = [v * scale if not _isnan(v) else 0 for v in raw_vals]

        n_trials = len(clean)
        if n_trials >= 2 and not _isnan(std_val):
            t_crit = scipy_stats.t.ppf(0.975, df=n_trials - 1)
            sem = (std_val * scale) / (n_trials**0.5)
            ci_half = t_crit * sem
        else:
            ci_half = 0.0
        yerr = [0.0] * (n_groups - 1) + [ci_half]

        style_key = MODALITY_STYLE_KEY.get(mod, mod)
        style = get_line_style(style_key)
        color = style["color"]
        label = _display_to_legend_multiline(MODALITY_DISPLAY.get(mod, mod))
        ax.bar(
            x + offsets[i],
            plot_vals,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            yerr=yerr,
            capsize=2,
            error_kw={"color": "black", "linewidth": 0.8},
        )

    ax.set_ylabel(plot_label(mcfg["ylabel"], use_latex=use_latex), fontsize=12)
    ax.set_xlabel(plot_label(r"Cross Validation\ Trial", use_latex=use_latex), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10.5, ha="center")
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        ncol=len(MODALITY_KEYS),
        bbox_to_anchor=(0.5, 0.98),
        fontsize=12,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    out_path = output_path
    if not out_path:
        out_path = os.path.join(out_dir, f"{geometry_variant}{mcfg['out_suffix']}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bar chart of a pressure error metric per trial and mean, by modality."
    )
    parser.add_argument(
        "--all_sets",
        action="store_true",
        help="Generate plots for every set_name under <data-root>/cross_validation/ (ignores set_name positional).",
    )
    parser.add_argument(
        "set_name",
        nargs="?",
        default=None,
        help="Set name (e.g., VMR_rigid_aorta_adults). Omit when using --all_sets.",
    )
    parser.add_argument(
        "geometry_variant",
        nargs="?",
        default="bifurcations_EL",
        help="Geometry variant (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        default="all",
        choices=["all", *list(METRIC_CONFIG.keys())],
        help="Which pressure metric to plot (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file (only when a single set, single run-config, and single metric; not with --all_sets)",
    )
    parser.add_argument(
        "--data_root",
        default="results",
        help="Root for results/cross_validation (default: results)",
    )
    parser.add_argument(
        "--run_config",
        default="all",
        help="Run config subfolder (default: all = discover under each set). "
        "Pass a token string or canonical suffix (e.g. gen_loss, quadratic_resistor_gen_loss).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    args = parser.parse_args()

    if args.all_sets:
        if args.set_name is not None:
            parser.error("Do not pass set_name when using --all_sets")
        set_names = discover_cross_validation_set_names(args.data_root)
        if not set_names:
            parser.error(f"No subdirectories found under {os.path.join(args.data_root, 'cross_validation')}")
        print(f"--all_sets: found {len(set_names)} set(s): {', '.join(set_names)}")
    elif args.set_name is None:
        parser.error("set_name is required unless you pass --all_sets")
    else:
        set_names = [args.set_name]

    resolved_run_config = None
    if args.run_config != "all":
        try:
            resolved_run_config = resolve_run_config_suffix(args.run_config)
        except ValueError as e:
            parser.error(str(e))

    metrics = list(METRIC_CONFIG.keys()) if args.metric == "all" else [args.metric]
    single_output = len(set_names) == 1 and args.run_config != "all" and len(metrics) == 1 and not args.all_sets

    for set_name in set_names:
        if args.run_config == "all":
            run_configs = discover_run_config_suffixes(os.path.join(args.data_root, "cross_validation", set_name))
        else:
            run_configs = [resolved_run_config]
        for rc in run_configs:
            out_dir = os.path.join(args.data_root, "cross_validation", set_name, rc)
            if not os.path.isdir(out_dir):
                continue
            for metric_key in metrics:
                output_path = args.output if single_output else None
                try:
                    generate_bar_chart(metric_key, out_dir, args.geometry_variant, output_path, args.dpi)
                except FileNotFoundError as e:
                    print(f"Skipping {set_name}/{rc}/{metric_key}: {e}")


if __name__ == "__main__":
    main()
