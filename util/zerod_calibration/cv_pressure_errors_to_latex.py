#!/usr/bin/env python3
"""
Write cross-validation pressure error CSVs as a LaTeX table.

Rows = CV trials; columns = modalities, each with 3 sub-columns:
relative_max_error, max_error, MSE.

Usage:
  python -m util.zerod_calibration.cv_pressure_errors_to_latex VMR_rigid_aorta_adults bifurcations_EL [--output table.tex]
"""

import argparse
import csv
import os


# Modality order and display names for the table
MODALITY_ORDER = [
    ("geometric", "Poiseuille (Baseline)"),
    ("BloodVesselJunction_NN", r"Learned Junction Parameters"),
    ("NN_vessel", r"Learned Vessel Parameters"),
    ("BloodVesselJunction_NN_plus_Vessel_NN", r"Learned Junction and Vessel Parameters"),
    ("BloodVesselJunction", "Optimal Parameters"),
]

PREFIX_REL = "PressureMaxRelError_"
PREFIX_MAX = "PressureMaxError_"
PREFIX_MSE = "PressureMSE_"


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


def _isnan(x):
    return x != x


def main():
    parser = argparse.ArgumentParser(
        description="Convert CV pressure error CSVs to a LaTeX table."
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
        help="Output .tex file (default: print to stdout)",
    )
    parser.add_argument(
        "--data-root",
        default="results",
        help="Root for results/cross_validation (default: results)",
    )
    args = parser.parse_args()

    base = os.path.join(
        args.data_root, "cross_validation", args.set_name,
        f"{args.geometry_variant}_cv_summary"
    )
    path_rel = base + "_pressure_max_rel_error.csv"
    path_max = base + "_pressure_max_error.csv"
    path_mse = base + "_pressure_mse.csv"

    for p in (path_rel, path_max, path_mse):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CSV: {p}")

    # Load by modality: for each modality, lists of (rel, max, mse) per trial
    by_mod = {}
    for mod, display in MODALITY_ORDER:
        rel = load_csv_column(path_rel, PREFIX_REL, mod)
        max_ = load_csv_column(path_max, PREFIX_MAX, mod)
        mse = load_csv_column(path_mse, PREFIX_MSE, mod)
        by_mod[mod] = {"rel": rel, "max": max_, "mse": mse}

    # Trial IDs from first file
    trial_ids = []
    with open(path_rel, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if tid in ("", "mean", "std"):
                break
            trial_ids.append(tid)

    # Build LaTeX
    num_cols = 1 + 3 * len(MODALITY_ORDER)
    lines = []
    lines.append(r"\begin{tabular}{l|" + "c" * (3 * len(MODALITY_ORDER)) + "}")
    lines.append(r"\hline")
    # Header row 1: Trial & \multicolumn{3}{c|}{geometric} & ...
    header1 = "Trial"
    for mod, display in MODALITY_ORDER:
        header1 += f" & \\multicolumn{{3}}{{c|}}{{{display}}}"
    lines.append(header1 + r" \\")
    lines.append(r"\hline")
    # Sub-header: empty under Trial, then rel_max & max_err & MSE per modality
    sub_parts = ["rel\\_max", "max\\_err", "MSE"] * len(MODALITY_ORDER)
    lines.append(" & " + " & ".join(sub_parts) + r" \\")
    lines.append(r"\hline")

    for idx, tid in enumerate(trial_ids):
        row_parts = [tid]
        for mod, _ in MODALITY_ORDER:
            r = by_mod[mod]["rel"][idx] if idx < len(by_mod[mod]["rel"]) else float("nan")
            m = by_mod[mod]["max"][idx] if idx < len(by_mod[mod]["max"]) else float("nan")
            s = by_mod[mod]["mse"][idx] if idx < len(by_mod[mod]["mse"]) else float("nan")
            r_s = f"{r:.4f}" if not _isnan(r) else "---"
            m_s = f"{m:.3f}" if not _isnan(m) else "---"
            s_s = f"{s:.4f}" if not _isnan(s) else "---"
            row_parts.extend([r_s, m_s, s_s])
        lines.append(" & ".join(row_parts) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out = "\n".join(lines)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
