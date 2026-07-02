#!/usr/bin/env python3
"""
Overlaid histograms of fitted R and S values from two run configs.

Two subplots: R_poiseuille (left), stenosis coefficient S (right).
Default overlays:
  - gen_loss (cornflowerblue): R only (no stenosis fit in this run config)
  - quadratic_resistor + gen_loss (deeppink; path suffix quadratic_resistor_gen_loss): R and S

Values are pooled from junction_lumped_parameters.csv and vessel_lumped_parameters.csv
across all geometries under data/ml_inputs/<set>/<run_config>/<geometry_variant>/.

Usage:
  python -m learn_lpns.visualizations.R_S_hist VMR_pulmo
  python -m learn_lpns.visualizations.R_S_hist VMR_pulmo VMR_aorta -o results/plots/R_S_hist.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from learn_lpns.visualizations.matplotlib_tex import configure_matplotlib_latex
from learn_lpns.zerod_calibration.run_config_canonical import (
    resolve_run_config_suffix,
    run_config_suffix_has_quadratic_resistor,
)

COLOR_GEN_LOSS = "cornflowerblue"
COLOR_QUAD_GEN_LOSS = "deeppink"
ALPHA = 0.5

JUNCTION_R_COLS = ("R_poiseuille_outlet0", "R_poiseuille_outlet1")
JUNCTION_S_COLS = ("stenosis_coefficient_outlet0", "stenosis_coefficient_outlet1")
VESSEL_R_COL = "R_poiseuille"
VESSEL_S_COL = "stenosis_coefficient"


def _discover_geometries(ml_inputs_dir: str) -> list[str]:
    if not os.path.isdir(ml_inputs_dir):
        return []
    geos: list[str] = []
    for name in sorted(os.listdir(ml_inputs_dir)):
        geo_dir = os.path.join(ml_inputs_dir, name)
        if not os.path.isdir(geo_dir):
            continue
        has_junction = os.path.isfile(os.path.join(geo_dir, "junction_lumped_parameters.csv"))
        has_vessel = os.path.isfile(os.path.join(geo_dir, "vessel_lumped_parameters.csv"))
        if has_junction or has_vessel:
            geos.append(name)
    return geos


def _read_csv_columns(csv_path: str, columns: tuple[str, ...]) -> list[float]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        present = [c for c in columns if c in reader.fieldnames]
        if not present:
            return []
        values: list[float] = []
        for row in reader:
            for col in present:
                raw = row.get(col)
                if raw is None or raw == "":
                    continue
                values.append(float(raw))
        return values


def _collect_r_s_for_run_config(
    *,
    data_root: str,
    set_name: str,
    run_config_suffix: str,
    geometry_variant: str,
    include_stenosis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ml_inputs_dir = os.path.join(
        data_root,
        "ml_inputs",
        set_name,
        run_config_suffix,
        geometry_variant,
    )
    geos = _discover_geometries(ml_inputs_dir)
    r_vals: list[float] = []
    s_vals: list[float] = []
    for geo in geos:
        geo_dir = os.path.join(ml_inputs_dir, geo)
        junction_csv = os.path.join(geo_dir, "junction_lumped_parameters.csv")
        vessel_csv = os.path.join(geo_dir, "vessel_lumped_parameters.csv")
        r_vals.extend(_read_csv_columns(junction_csv, JUNCTION_R_COLS))
        r_vals.extend(_read_csv_columns(vessel_csv, (VESSEL_R_COL,)))
        if include_stenosis:
            s_vals.extend(_read_csv_columns(junction_csv, JUNCTION_S_COLS))
            s_vals.extend(_read_csv_columns(vessel_csv, (VESSEL_S_COL,)))
    return np.asarray(r_vals, dtype=float), np.asarray(s_vals, dtype=float)


def _finite_values(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _shared_bins(a: np.ndarray, b: np.ndarray, nbins: int) -> np.ndarray:
    combined = np.concatenate([a, b]) if a.size and b.size else (a if a.size else b)
    if combined.size == 0:
        return np.linspace(0.0, 1.0, nbins + 1)
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, nbins + 1)


def _plot_histograms(
    series: list[tuple[str, str, np.ndarray, np.ndarray]],
    *,
    output_path: str,
    nbins: int,
    dpi: int,
) -> None:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required")

    configure_matplotlib_latex(plt)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    r_a = _finite_values(series[0][2])
    r_b = _finite_values(series[1][2])
    s_a = _finite_values(series[0][3])
    s_b = _finite_values(series[1][3])

    r_bins = _shared_bins(r_a, r_b, nbins)
    s_bins = _shared_bins(s_a, s_b, nbins) if s_a.size or s_b.size else np.linspace(0.0, 1.0, nbins + 1)

    ax_r, ax_s = axes
    for label, color, r_vals, s_vals in series:
        r_vals = _finite_values(r_vals)
        s_vals = _finite_values(s_vals)
        if r_vals.size:
            ax_r.hist(
                r_vals,
                bins=r_bins,
                color=color,
                alpha=ALPHA,
                label=label,
                edgecolor="none",
            )
        if s_vals.size:
            ax_s.hist(
                s_vals,
                bins=s_bins,
                color=color,
                alpha=ALPHA,
                label=label,
                edgecolor="none",
            )

    ax_r.set_title(r"$R_{\mathrm{Poiseuille}}$")
    ax_s.set_title(r"$S$ (stenosis coefficient)")
    ax_r.set_xlabel(r"$R_{\mathrm{Poiseuille}}$")
    ax_s.set_xlabel(r"$S$")
    ax_r.set_ylabel("Count")
    ax_s.set_ylabel("Count")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
    ax_r.legend()
    ax_s.legend()

    fig.suptitle("Fitted R and S values")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histograms of fitted R and S for gen_loss vs quadratic_resistor+gen_loss."
    )
    parser.add_argument(
        "set_names",
        nargs="+",
        help="One or more cohort set names (e.g. VMR_pulmo)",
    )
    parser.add_argument(
        "--run_config_a",
        default="gen_loss",
        help="First run config (default: gen_loss)",
    )
    parser.add_argument(
        "--run_config_b",
        default="gen_loss_quadratic_resistor",
        help=(
            "Second run config (default: gen_loss_quadratic_resistor; resolves to quadratic_resistor_gen_loss on disk)"
        ),
    )
    parser.add_argument(
        "--label_a",
        default=None,
        help="Legend label for run_config_a (default: run_config_a string)",
    )
    parser.add_argument(
        "--label_b",
        default=None,
        help="Legend label for run_config_b (default: run_config_b string)",
    )
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations_EL",
        help="Geometry variant under ml_inputs (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Data root containing ml_inputs (default: data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output figure path (default: results/plots/R_S_hist/<sets>.pdf)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("Error: matplotlib is required.", file=sys.stderr)
        sys.exit(1)

    data_root = os.path.abspath(args.data_root)
    run_config_a = resolve_run_config_suffix(args.run_config_a)
    run_config_b = resolve_run_config_suffix(args.run_config_b)
    label_a = args.label_a or args.run_config_a
    label_b = args.label_b or args.run_config_b

    pooled: dict[str, tuple[list[float], list[float]]] = {
        run_config_a: ([], []),
        run_config_b: ([], []),
    }
    for set_name in args.set_names:
        for run_config in (run_config_a, run_config_b):
            include_stenosis = run_config_suffix_has_quadratic_resistor(run_config)
            r_vals, s_vals = _collect_r_s_for_run_config(
                data_root=data_root,
                set_name=set_name,
                run_config_suffix=run_config,
                geometry_variant=args.geometry_variant,
                include_stenosis=include_stenosis,
            )
            pooled[run_config][0].extend(r_vals.tolist())
            pooled[run_config][1].extend(s_vals.tolist())
            s_note = f"S n={s_vals.size}" if include_stenosis else "S skipped (no quadratic_resistor)"
            print(
                f"  [{set_name} / {run_config}] "
                f"R n={r_vals.size}, {s_note} "
                f"(ml_inputs/{set_name}/{run_config}/{args.geometry_variant})"
            )

    r_a = np.asarray(pooled[run_config_a][0], dtype=float)
    s_a = np.asarray(pooled[run_config_a][1], dtype=float)
    r_b = np.asarray(pooled[run_config_b][0], dtype=float)
    s_b = np.asarray(pooled[run_config_b][1], dtype=float)

    if r_a.size == 0 and r_b.size == 0 and s_b.size == 0:
        print(
            "Error: no R/S values found. Run data processing for both run configs first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        set_slug = args.set_names[0] if len(args.set_names) == 1 else "__".join(args.set_names)
        out_dir = os.path.join("results", "plots", "R_S_hist")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{set_slug}_{args.geometry_variant}.pdf")

    series = [
        (label_a, COLOR_GEN_LOSS, r_a, s_a),
        (label_b, COLOR_QUAD_GEN_LOSS, r_b, s_b),
    ]
    _plot_histograms(series, output_path=output_path, nbins=args.bins, dpi=args.dpi)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
