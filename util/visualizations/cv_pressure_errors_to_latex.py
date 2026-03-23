#!/usr/bin/env python3
"""
Write cross-validation pressure error CSVs as a LaTeX table.

Rows = CV trials, each with 3 sub-rows (relative_max_error, max_error, RMSE).
Each trial is introduced by a multirow cell: "Trial X: val_geometry".
Columns = modalities (geometric, BloodVesselJunction_NN, etc.).
Requires \\usepackage{multirow} in your LaTeX document.

Usage:
  python -m util.visualizations.cv_pressure_errors_to_latex VMR_rigid_aorta_adults bifurcations_EL [--output table.tex]
"""

import argparse
import csv
import math
import os
import statistics


# Modality keys (column order in CSV/code)
MODALITY_KEYS = [
    "geometric",
    "BloodVesselJunction_NN",
    "NN_vessel",
    "BloodVesselJunction_NN_plus_Vessel_NN",
    "BloodVesselJunction",
]

# Display names for the table header: modality key (code) -> display.
# Value = string (single-line header) or list of strings (multi-line via \\shortstack).
MODALITY_DISPLAY = {
    # Reference time series (location comparison plots only; not a CV summary column)
    "3d_model": [r"3D", r"Solution"],
    "geometric": [r"Poiseuille", r"(Baseline)"],
    "BloodVesselJunction_NN": [r"Learned", r"Junctions"],
    "NN_vessel": [r"Learned", r"Vessels"],
    "BloodVesselJunction_NN_plus_Vessel_NN": [
        r"Learned",  r"Junctions", r"and Vessels"
    ],
    "BloodVesselJunction": [r"Optimal", r"Fit to 3D"],
    # Calibrated $\Delta P = 0$ junction (not in default CV bar chart columns)
    "NORMAL_JUNCTION": [r"$\Delta P = 0$", r"Junction"],
}


def format_modality_display_for_legend(modality_key, extra_lines=None):
    """
    Turn a MODALITY_DISPLAY entry into a matplotlib legend string (one row per line).

    ``extra_lines`` optional suffix rows, e.g. ["(orig)"] when both original and bifurcations
    optimal curves appear on the same figure.
    """
    spec = MODALITY_DISPLAY.get(modality_key)
    if spec is None:
        return None
    lines = list(spec) if isinstance(spec, list) else [spec]
    if extra_lines:
        lines = lines + [str(x) for x in extra_lines]
    return "\n".join(lines)


# Display names for the Metric column (sub-rows). Key = metric key in code.
# Value = string or list of strings (multi-line via \\shortstack).
METRIC_DISPLAY = {
    "rel_max": r"Max. Err. (\%)",
    "max_err": r"Max. Err. (mmHg)",
    "mse": "Cycle RMSE (mmHg)",
}

# Order of the three metric sub-rows
METRIC_ROW_ORDER = ["rel_max", "max_err", "mse"]

# Display names for validation geometries (val_geometries column value -> display).
# Key = geometry id as in CSV (e.g. "0094_0001"). Value = string or list of strings (multi-line).
# If missing, the raw id is used (with underscores escaped).
VAL_GEOMETRY_DISPLAY = {
    # Example: "0094_0001": "Patient 94, Case 1",
    "0076_1001": ["0005_H_AO_SVD"],
    "0094_0001": ["0011_H_AO_H"],
    "0095_0001": ["0012_H_AO_H"],
    "0129_0000": ["0021_H_AO_MFS"],
    "0176_0000": ["0027_H_AO_MFS"],
}


def _latex_escape(s):
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_")


def _header_cell(display_spec):
    """Convert a display spec (str or list of str) to LaTeX header cell content."""
    if isinstance(display_spec, list):
        return r"\shortstack{" + " \\\\ ".join(display_spec) + "}"
    return display_spec


def _trial_cell(tid, val_geo):
    """Build multi-line multirow cell: Trial X on first line, validation geometry display on next line(s)."""
    display = VAL_GEOMETRY_DISPLAY.get(val_geo, val_geo)
    if isinstance(display, list):
        lines_tex = [_latex_escape(str(x)) for x in display]
    else:
        lines_tex = [_latex_escape(str(display))]
    trial_line = f"Trial {tid}"
    content = " \\\\ ".join([trial_line] + ["Validation Geometry:"] + lines_tex)
    return rf"\multirow{{3}}{{*}}{{\shortstack[l]{{{content}}}}}"


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
    parser.add_argument(
        "--run-config",
        default="base",
        help="Run-config subfolder (default: base). Use e.g. stenosis_off or normalized_clip when CV was run with that config.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, "cross_validation", args.set_name, (args.run_config or "base").strip())
    base = os.path.join(out_dir, f"{args.geometry_variant}_cv_summary")
    path_rel = base + "_pressure_max_rel_error.csv"
    path_max = base + "_pressure_max_error.csv"
    path_mse = base + "_pressure_mse.csv"

    for p in (path_rel, path_max, path_mse):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CSV: {p}")

    # Load by modality: for each modality, lists of (rel, max, mse) per trial
    by_mod = {}
    for mod in MODALITY_KEYS:
        rel = load_csv_column(path_rel, PREFIX_REL, mod)
        max_ = load_csv_column(path_max, PREFIX_MAX, mod)
        mse = load_csv_column(path_mse, PREFIX_MSE, mod)
        by_mod[mod] = {"rel": rel, "max": max_, "mse": mse}

    # Trial IDs and validation geometry names from first file
    trial_rows = []  # list of (trial_id, val_geometries)
    with open(path_rel, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if tid in ("", "mean", "std"):
                break
            val_geo = row.get("val_geometries", "").strip()
            trial_rows.append((tid, val_geo))

    # Build LaTeX: first column = multirow "Trial X: val_geometry" spanning 3 rows, then Metric | modalities
    # Requires \usepackage{multirow} in the document.
    n_mods = len(MODALITY_KEYS)
    lines = []
    lines.append(r"\begin{tabular}{l r " + "c" * n_mods + "}")
    
    header = r"\underline{Trial \#} & \underline{Metric}"
    for mod in MODALITY_KEYS:
        header += " & \\underline{" + _header_cell(MODALITY_DISPLAY[mod]) + "}"
    lines.append(header + r" \\")
    lines.append(r"\\[0.8em]")

    for idx, (tid, val_geo) in enumerate(trial_rows):
        r_vals = []
        m_vals = []
        s_vals = []
        for mod in MODALITY_KEYS:
            r = by_mod[mod]["rel"][idx] if idx < len(by_mod[mod]["rel"]) else float("nan")
            m = by_mod[mod]["max"][idx] if idx < len(by_mod[mod]["max"]) else float("nan")
            s = by_mod[mod]["mse"][idx] if idx < len(by_mod[mod]["mse"]) else float("nan")
            r_vals.append(f"{r:.3f}" if not _isnan(r) else "---")
            m_vals.append(f"{m:.2f}" if not _isnan(m) else "---")
            s_display = math.sqrt(s) if not _isnan(s) and s >= 0 else s
            s_vals.append(f"{s_display:.2f}" if not _isnan(s_display) else "---")
        val_lists = {"rel_max": r_vals, "max_err": m_vals, "mse": s_vals}
        trial_cell = _trial_cell(tid, val_geo)
        for i, metric_key in enumerate(METRIC_ROW_ORDER):
            first_col = trial_cell if i == 0 else ""
            trial_cell = ""  # only first of the 3 rows gets the multirow
            # Add extra vertical space after last row of each trial to separate blocks
            row_ending = r" \\[0.8em]" if i == len(METRIC_ROW_ORDER) - 1 else r" \\"
            lines.append(
                (first_col + " & " if first_col else " & ")
                + _header_cell(METRIC_DISPLAY[metric_key])
                + " & "
                + " & ".join(val_lists[metric_key])
                + row_ending
            )

    # Mean and Std row sets (across trials, per modality and metric)
    def _safe_mean(vals):
        finite = [x for x in vals if not _isnan(x)]
        return statistics.mean(finite) if len(finite) > 0 else float("nan")

    def _safe_stdev(vals):
        finite = [x for x in vals if not _isnan(x)]
        return statistics.stdev(finite) if len(finite) > 1 else (0.0 if len(finite) == 1 else float("nan"))

    for label, first_col_text in [("Mean", r"\multirow{3}{*}{\shortstack[l]{Average}}"), ("Std", r"\multirow{3}{*}{\shortstack[l]{Standard Deviation}}")]:
        r_vals = []
        m_vals = []
        s_vals = []
        for mod in MODALITY_KEYS:
            rel_list = by_mod[mod]["rel"]
            max_list = by_mod[mod]["max"]
            mse_list = by_mod[mod]["mse"]
            rmse_list = [math.sqrt(x) for x in mse_list if not _isnan(x) and x >= 0]
            if label == "Mean":
                rv = _safe_mean(rel_list)
                mv = _safe_mean(max_list)
                sv = _safe_mean(rmse_list)
                r_vals.append(f"{rv:.3f}" if not _isnan(rv) else "---")
                m_vals.append(f"{mv:.2f}" if not _isnan(mv) else "---")
                s_vals.append(f"{sv:.2f}" if not _isnan(sv) else "---")
            else:
                rv = _safe_stdev(rel_list)
                mv = _safe_stdev(max_list)
                sv = _safe_stdev(rmse_list)
                r_vals.append(f"{rv:.3f}" if not _isnan(rv) else "---")
                m_vals.append(f"{mv:.2f}" if not _isnan(mv) else "---")
                s_vals.append(f"{sv:.2f}" if not _isnan(sv) else "---")
        val_lists = {"rel_max": r_vals, "max_err": m_vals, "mse": s_vals}
        for i, metric_key in enumerate(METRIC_ROW_ORDER):
            first_col = first_col_text if i == 0 else ""
            if i == 0:
                first_col_text = ""  # only first row of the 3 gets the multirow
            row_ending = r" \\[0.8em]" if i == len(METRIC_ROW_ORDER) - 1 else r" \\"
            lines.append(
                (first_col + " & " if first_col else " & ")
                + _header_cell(METRIC_DISPLAY[metric_key])
                + " & "
                + " & ".join(val_lists[metric_key])
                + row_ending
            )
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
