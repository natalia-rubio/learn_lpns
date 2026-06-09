#!/usr/bin/env python3
"""
Plot 4 overlaid histograms for R_poiseuille and L across CV test geometries.

Two series per subplot (alpha=0.5):
  - geometric − calibrated (indian red)
  - NN − calibrated (lime green); NN from bifurcations_EL_NN_JunctionAndVessel.json

Layout:
  Row 1: Junctions — R_poiseuille, L
  Row 2: Vessels  — R_poiseuille, L

Uses geometric_input, NN_JunctionAndVessel, and calibrated_output_BloodVesselJunction
from data/zeroD/{set_name}/{run_config}/{geo}/. Only geometries with calibrated
output are included. Elements whose calibrated parameter is exactly 0 are excluded.
Test geometry list from CV summary.

Usage:
  python -m util.visualizations.cv_geometric_vs_calibrated_histograms VMR_rigid_aorta_adults --run_config gen_loss -o histograms.pdf

  Multiple sets (same --run_config and --geometry_variant; pooled histograms and stats):
  python -m util.visualizations.cv_geometric_vs_calibrated_histograms \\
    VMR_rigid_aorta_adults_all VMR_abdo VMR_pulmo --run_config gen_loss
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# Matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _extract_element_values(json_path):
    """
    Extract R_poiseuille and L from a 0D JSON (vessels and junctions).
    Returns (vessel_R, vessel_L, junction_R, junction_L,
             vessel_names_R, vessel_names_L, junction_names_R, junction_names_L).
    Names: vessel_name per sample; junctions use junction_name, with outlet key
    (junction_name:outlet_key) when values are per-outlet lists aligned with outlet_vessel_ids.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    vessels = data.get("vessels", [])
    junctions = data.get("junctions", [])

    vessel_R, vessel_L = [], []
    vessel_names_R, vessel_names_L = [], []
    for v in vessels:
        z = v.get("zero_d_element_values") or {}
        vname = (v.get("vessel_name") or "").strip() or str(v.get("vessel_id", "?"))
        if "R_poiseuille" in z:
            vessel_R.append(float(z["R_poiseuille"]))
            vessel_names_R.append(vname)
        if "L" in z:
            vessel_L.append(float(z["L"]))
            vessel_names_L.append(vname)

    junction_R, junction_L = [], []
    junction_names_R, junction_names_L = [], []
    for j in junctions:
        jv = j.get("junction_values") or {}
        jname = (j.get("junction_name") or "").strip() or str(j.get("junction_id", "?"))
        outlet_ids = (j.get("centerline_node_ids") or {}).get("outlet_vessel_ids")
        if isinstance(outlet_ids, dict):
            outlet_keys = list(outlet_ids.keys())
        else:
            outlet_keys = []

        for key, container, name_container in [
            ("R_poiseuille", junction_R, junction_names_R),
            ("L", junction_L, junction_names_L),
        ]:
            if key not in jv:
                continue
            vals = jv[key]
            if isinstance(vals, list):
                if outlet_keys and len(outlet_keys) == len(vals):
                    for ok, x in zip(outlet_keys, vals):
                        container.append(float(x))
                        name_container.append(f"{jname}:{ok}")
                else:
                    for i, x in enumerate(vals):
                        container.append(float(x))
                        name_container.append(f"{jname}[{i}]")
            else:
                container.append(float(vals))
                name_container.append(jname)

    return (
        vessel_R,
        vessel_L,
        junction_R,
        junction_L,
        vessel_names_R,
        vessel_names_L,
        junction_names_R,
        junction_names_L,
    )


def _get_validation_geometries(set_name, run_config, geometry_variant, results_root):
    """Read CV summary CSV and return unique list of validation geometry names."""
    if run_config:
        out_dir = os.path.join(results_root, "cross_validation", set_name, run_config)
    else:
        out_dir = os.path.join(results_root, "cross_validation", set_name)
    out_dir = os.path.abspath(out_dir)
    summary_path = os.path.join(out_dir, f"{geometry_variant}_cv_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"CV summary not found: {summary_path}. Run cross-validation first."
        )
    val_geos = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("trial_id") or "").strip()
            if tid in ("", "mean", "std"):
                break
            vg = (row.get("val_geometries") or "").strip()
            for g in vg.split(","):
                g = g.strip()
                if g and g not in val_geos:
                    val_geos.append(g)
    return val_geos


def _collect_differences(set_name, run_config, geometry_variant, data_root, results_root, min_abs_cal=5.0):
    """
    For each validation geometry that has calibrated output, load geometric, NN, and
    calibrated JSONs. Compute (geometric - calibrated) and (NN - calibrated) for
    R_poiseuille and L (vessels and junctions).
    Excludes elements where |calibrated| < min_abs_cal (so plots and reports use the same subset).
    Returns dict with keys junction_R, junction_L, vessel_R, vessel_L; each value is
    a dict {"geo_minus_cal": [...], "nn_minus_cal": [...], "geo_pct_err": [...], "nn_pct_err": [...],
    "cal_values": [...], "geometry": [...], "element_name": [...]}
    (geometry = validation geometry id; element_name = vessel_name or junction outlet label from cal JSON).
    Percent error = (value - calibrated) / |calibrated|; stored as fraction (multiply by 100 for %).
    """
    prefix = "" if geometry_variant == "original" else f"{geometry_variant}_"
    geo_name = f"{prefix}geometric_input.json"
    nn_name = f"{prefix}NN_JunctionAndVessel.json"
    cal_name = f"{prefix}calibrated_output_BloodVesselJunction.json"
    data_root = os.path.abspath(data_root)
    if run_config:
        zero_d_base = os.path.join(data_root, "zeroD", set_name, run_config)
    else:
        zero_d_base = os.path.join(data_root, "zeroD", set_name)

    val_geos = _get_validation_geometries(
        set_name, run_config, geometry_variant, results_root
    )
    out = {
        k: {
            "geo_minus_cal": [],
            "nn_minus_cal": [],
            "geo_pct_err": [],
            "nn_pct_err": [],
            "cal_values": [],
            "geometry": [],
            "element_name": [],
        }
        for k in ("junction_R", "junction_L", "vessel_R", "vessel_L")
    }

    for geo in val_geos:
        base = os.path.join(zero_d_base, geo)
        geo_path = os.path.join(base, geo_name)
        nn_path = os.path.join(base, nn_name)
        cal_path = os.path.join(base, cal_name)
        if not all(os.path.exists(p) for p in (geo_path, nn_path, cal_path)):
            continue
        vR_g, vL_g, jR_g, jL_g, _, _, _, _ = _extract_element_values(geo_path)
        vR_n, vL_n, jR_n, jL_n, _, _, _, _ = _extract_element_values(nn_path)
        (
            vR_c,
            vL_c,
            jR_c,
            jL_c,
            vnR_c,
            vnL_c,
            jnR_c,
            jnL_c,
        ) = _extract_element_values(cal_path)
        for g_list, n_list, c_list, names_c, key in [
            (jR_g, jR_n, jR_c, jnR_c, "junction_R"),
            (jL_g, jL_n, jL_c, jnL_c, "junction_L"),
            (vR_g, vR_n, vR_c, vnR_c, "vessel_R"),
            (vL_g, vL_n, vL_c, vnL_c, "vessel_L"),
        ]:
            n = min(len(g_list), len(n_list), len(c_list), len(names_c))
            for i in range(n):
                c = c_list[i]
                if c == 0 or abs(c) < min_abs_cal:
                    continue
                abs_c = abs(c)
                out[key]["geo_minus_cal"].append(g_list[i] - c)
                out[key]["nn_minus_cal"].append(n_list[i] - c)
                out[key]["geo_pct_err"].append((g_list[i] - c) / abs_c)
                out[key]["nn_pct_err"].append((n_list[i] - c) / abs_c)
                out[key]["cal_values"].append(c)
                out[key]["geometry"].append(geo)
                out[key]["element_name"].append(names_c[i])
    return out


def _merge_diffs_into(accum, new_diffs):
    """Extend accum's lists in-place with values from new_diffs (same structure as _collect_differences)."""
    for key in accum:
        for subkey in accum[key]:
            accum[key][subkey].extend(new_diffs[key][subkey])


def _report_average_pct_error(diffs, file=None, min_abs_cal=5.0):
    """
    Compute and print average percent error (geo − cal)/|cal| and (nn − cal)/|cal|
    per parameter (R_poiseuille, L), element type (junction, vessel), and source (geometric, NN).
    Percent errors are reported as percentage (× 100).
    Excludes elements where |calibrated| < min_abs_cal.
    """
    file = file or sys.stdout
    param_names = {"junction_R": "Junction R_poiseuille", "junction_L": "Junction L",
                   "vessel_R": "Vessel R_poiseuille", "vessel_L": "Vessel L"}
    print("Average percent error (value − calibrated) / |calibrated| × 100 (excluding |calibrated| < 5):", file=file)
    print(file=file)
    for key in ("junction_R", "junction_L", "vessel_R", "vessel_L"):
        geo_pct = np.array(diffs[key].get("geo_pct_err", []), dtype=float)
        nn_pct = np.array(diffs[key].get("nn_pct_err", []), dtype=float)
        cal_vals = np.array(diffs[key].get("cal_values", []), dtype=float)
        mask = np.isfinite(geo_pct) & np.isfinite(nn_pct)
        if len(cal_vals) == len(geo_pct):
            mask = mask & (np.abs(cal_vals) >= min_abs_cal)
        geo_pct = geo_pct[mask]
        nn_pct = nn_pct[mask]
        mean_geo = np.mean(geo_pct) * 100 if geo_pct.size else float("nan")
        mean_nn = np.mean(nn_pct) * 100 if nn_pct.size else float("nan")
        n_geo = len(geo_pct)
        n_nn = len(nn_pct)
        print(f"  {param_names[key]:25s}  geometric: {mean_geo:8.2f}% (n={n_geo})   NN: {mean_nn:8.2f}% (n={n_nn})", file=file)
    print(file=file)


def _report_average_error(diffs, file=None):
    """
    Compute and print average error (value − calibrated), no division by |calibrated|,
    per parameter (R_poiseuille, L), element type (junction, vessel), and source (geometric, NN).
    """
    file = file or sys.stdout
    param_names = {"junction_R": "Junction R_poiseuille", "junction_L": "Junction L",
                   "vessel_R": "Vessel R_poiseuille", "vessel_L": "Vessel L"}
    print("Average error (value − calibrated), no scaling:", file=file)
    print(file=file)
    for key in ("junction_R", "junction_L", "vessel_R", "vessel_L"):
        geo_err = np.array(diffs[key].get("geo_minus_cal", []), dtype=float)
        nn_err = np.array(diffs[key].get("nn_minus_cal", []), dtype=float)
        geo_err = geo_err[np.isfinite(geo_err)]
        nn_err = nn_err[np.isfinite(nn_err)]
        mean_geo = np.mean(geo_err) if geo_err.size else float("nan")
        mean_nn = np.mean(nn_err) if nn_err.size else float("nan")
        n_geo = len(geo_err)
        n_nn = len(nn_err)
        print(f"  {param_names[key]:25s}  geometric: {mean_geo:12.4e} (n={n_geo})   NN: {mean_nn:12.4e} (n={n_nn})", file=file)
    print(file=file)


def _report_extreme_errors(diffs, file=None, min_abs_cal=5.0):
    """
    For each parameter group, print extremes of NN − cal and geometric − cal
    (largest overprediction, largest underprediction, largest |error|), with validation
    geometry id and element name (junction outlet or vessel from calibrated JSON).
    Uses the same |calibrated| ≥ min_abs_cal subset as the histograms.
    """
    file = file or sys.stdout
    param_names = {
        "junction_R": "Junction R_poiseuille",
        "junction_L": "Junction L",
        "vessel_R": "Vessel R_poiseuille",
        "vessel_L": "Vessel L",
    }
    print(
        "Extreme errors (value − calibrated), with validation geometry "
        f"(excluding |calibrated| < {min_abs_cal}):",
        file=file,
    )
    print(file=file)

    def _one_tail(label, err_arr, geo_list, cal_arr, elem_list, arg_idx_fn):
        if err_arr.size == 0:
            print(f"    {label}: (no data)", file=file)
            return
        idx = arg_idx_fn(err_arr)
        e = float(err_arr[idx])
        g = geo_list[idx]
        el = elem_list[idx] if idx < len(elem_list) else "?"
        c = float(cal_arr[idx]) if cal_arr.size == len(err_arr) else float("nan")
        print(
            f"    {label}: {e:+.6e}   geometry={g}   element={el}   cal={c:.6e}",
            file=file,
        )

    for key in ("junction_R", "junction_L", "vessel_R", "vessel_L"):
        nn_err = np.asarray(diffs[key].get("nn_minus_cal", []), dtype=float)
        geo_err = np.asarray(diffs[key].get("geo_minus_cal", []), dtype=float)
        cal_vals = np.asarray(diffs[key].get("cal_values", []), dtype=float)
        geos_raw = diffs[key].get("geometry", [])
        elems_raw = diffs[key].get("element_name", [])

        n = nn_err.size
        mask = np.isfinite(nn_err) & np.isfinite(geo_err)
        if cal_vals.size == n:
            mask = mask & (np.abs(cal_vals) >= min_abs_cal)
        nn_err = nn_err[mask]
        geo_err = geo_err[mask]
        cal_vals = cal_vals[mask] if cal_vals.size == n else np.array([], dtype=float)
        idx_keep = np.nonzero(mask)[0]
        if len(geos_raw) == n:
            geos = [geos_raw[i] for i in idx_keep]
        else:
            geos = ["?"] * int(nn_err.size)
        if len(elems_raw) == n:
            elems = [elems_raw[i] for i in idx_keep]
        else:
            elems = ["?"] * int(nn_err.size)

        name = param_names[key]
        print(f"  {name}:", file=file)
        if nn_err.size == 0:
            print("    NN: (no data)", file=file)
        else:
            cal_m = cal_vals if cal_vals.size == nn_err.size else np.array([], dtype=float)
            _one_tail(
                "NN  max (largest overprediction)",
                nn_err,
                geos,
                cal_m,
                elems,
                np.argmax,
            )
            _one_tail(
                "NN  min (largest underprediction)",
                nn_err,
                geos,
                cal_m,
                elems,
                np.argmin,
            )
            idx_abs = int(np.argmax(np.abs(nn_err)))
            c_abs = float(cal_m[idx_abs]) if cal_m.size == nn_err.size else float("nan")
            el_abs = elems[idx_abs] if idx_abs < len(elems) else "?"
            print(
                f"    NN  max |error|: {float(nn_err[idx_abs]):+.6e}   geometry={geos[idx_abs]}   "
                f"element={el_abs}   cal={c_abs:.6e}",
                file=file,
            )
        if geo_err.size == 0:
            print("    Geo (baseline) max |error|: (no data)", file=file)
        else:
            cal_g = cal_vals if cal_vals.size == geo_err.size else np.array([], dtype=float)
            idx_g = int(np.argmax(np.abs(geo_err)))
            c_g = float(cal_g[idx_g]) if cal_g.size == geo_err.size else float("nan")
            el_g = elems[idx_g] if idx_g < len(elems) else "?"
            print(
                f"    Geo max |error|: {float(geo_err[idx_g]):+.6e}   geometry={geos[idx_g]}   "
                f"element={el_g}   cal={c_g:.6e}",
                file=file,
            )
        print(file=file)


def _format_latex_num(x, is_pct=False):
    """Format a number for LaTeX: percent as decimal, else scientific."""
    if np.isnan(x) or (is_pct and np.isnan(x)):
        return "---"
    if is_pct:
        return f"{x:.2f}"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(np.abs(x))))
    mant = x / (10 ** exp)
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def _print_latex_tables(diffs, file=None, min_abs_cal=5.0):
    """
    Print one LaTeX table with average percent error and average error (geometric vs NN)
    per parameter (junction/vessel R_poiseuille, L). Percent error excludes |calibrated| < 5;
    average error uses the same subset so both are over the same n.
    """
    file = file or sys.stdout
    row_keys = ("junction_R", "junction_L", "vessel_R", "vessel_L")
    row_labels = [
        r"Junction $R_{\mathrm{Poiseuille}}$",
        r"Junction $L$",
        r"Vessel $R_{\mathrm{Poiseuille}}$",
        r"Vessel $L$",
    ]
    # Compute percent error and average error over the same subset (|cal| >= 5)
    pct_geo, pct_nn, err_geo, err_nn, n_vals = [], [], [], [], []
    for key in row_keys:
        geo_pct = np.array(diffs[key].get("geo_pct_err", []), dtype=float)
        nn_pct = np.array(diffs[key].get("nn_pct_err", []), dtype=float)
        geo_err = np.array(diffs[key].get("geo_minus_cal", []), dtype=float)
        nn_err = np.array(diffs[key].get("nn_minus_cal", []), dtype=float)
        cal_vals = np.array(diffs[key].get("cal_values", []), dtype=float)
        mask = np.isfinite(geo_pct) & np.isfinite(nn_pct)
        if len(cal_vals) == len(geo_pct):
            mask = mask & (np.abs(cal_vals) >= min_abs_cal)
        geo_pct = geo_pct[mask]
        nn_pct = nn_pct[mask]
        geo_err = geo_err[mask]
        nn_err = nn_err[mask]
        pct_geo.append(np.mean(geo_pct) * 100 if geo_pct.size else float("nan"))
        pct_nn.append(np.mean(nn_pct) * 100 if nn_pct.size else float("nan"))
        err_geo.append(np.mean(geo_err) if geo_err.size else float("nan"))
        err_nn.append(np.mean(nn_err) if nn_err.size else float("nan"))
        n_vals.append(len(geo_pct))
    # Print single LaTeX table
    print("% LaTeX table: average percent error and average error (geometric vs NN vs calibrated)", file=file)
    print(file=file)
    print("\\begin{table}[htbp]", file=file)
    print("\\centering", file=file)
    print("\\begin{tabular}{l r r r r}", file=file)
    print("\\toprule", file=file)
    print("Parameter & \\multicolumn{2}{c}{Relative Error (\\%)} & \\multicolumn{2}{c}{Absolute Error} \\\\", file=file)
    print(" & \\underline{Baseline} & \\underline{Neural Network} & \\underline{Baseline} & \\underline{Neural Network} \\\\", file=file)
    print("\\midrule", file=file)
    for i, label in enumerate(row_labels):
        pg = _format_latex_num(pct_geo[i], is_pct=True)
        pn = _format_latex_num(pct_nn[i], is_pct=True)
        eg = _format_latex_num(err_geo[i], is_pct=False)
        en = _format_latex_num(err_nn[i], is_pct=False)
        print(f"{label} & {pg} & {pn} & ${eg}$ & ${en}$ \\\\", file=file)
    print("\\bottomrule", file=file)
    print("\\end{tabular}", file=file)
    print("\\end{table}", file=file)


# Histogram colors: geometric − calibrated (indian red), NN − calibrated (lime green)
COLOR_GEO_CAL = "indianred"
COLOR_NN_CAL = "limegreen"
ALPHA = 0.5


def _plot_histograms(diffs, output_path, nbins=25, dpi=150):
    """
    Create 2x2 figure: row0 = junctions (R_poiseuille, L), row1 = vessels (R_poiseuille, L).
    Each subplot has two overlaid histograms: geometric − calibrated (indian red),
    NN − calibrated (lime green), both with alpha=0.5. Y-axis is log-scaled (count).
    """
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required")

    from util.visualizations.matplotlib_tex import configure_matplotlib_latex

    configure_matplotlib_latex(plt)

    fig, axes = plt.subplots(2, 2, figsize=(6, 4))
    labels = [
        (r"Junctions: $\Delta R_{\mathrm{Poiseuille}}$", r"Junctions: $\Delta L$"),
        (r"Vessels: $\Delta R_{\mathrm{Poiseuille}}$", r"Vessels: $\Delta L$"),
    ]
    keys_grid = [
        ("junction_R", "junction_L"),
        ("vessel_R", "vessel_L"),
    ]
    for row, (key_R, key_L) in enumerate(keys_grid):
        for col, (key, title) in enumerate([(key_R, labels[row][0]), (key_L, labels[row][1])]):
            ax = axes[row, col]
            bucket = diffs.get(key, {"geo_minus_cal": [], "nn_minus_cal": []})
            geo_data = np.array(bucket.get("geo_minus_cal", []), dtype=float)
            nn_data = np.array(bucket.get("nn_minus_cal", []), dtype=float)
            geo_data = geo_data[np.isfinite(geo_data)]
            nn_data = nn_data[np.isfinite(nn_data)]
            if geo_data.size == 0 and nn_data.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                ax.set_xlabel("Error (geometric − cal, NN − cal)")
                continue
            all_vals = np.concatenate([geo_data, nn_data]) if geo_data.size and nn_data.size else (geo_data if geo_data.size else nn_data)
            x_min, x_max = np.nanmin(all_vals), np.nanmax(all_vals)
            if x_max <= x_min:
                x_max = x_min + 1.0
            bins = np.linspace(x_min, x_max, nbins + 1)
            if geo_data.size > 0:
                ax.hist(geo_data, bins=bins, color=COLOR_GEO_CAL, alpha=ALPHA, label="Baseline (Poiseuille)", edgecolor="none")
            if nn_data.size > 0:
                ax.hist(nn_data, bins=bins, color=COLOR_NN_CAL, alpha=ALPHA, label="Neural Network", edgecolor="none")
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
            #ax.set_title(title)
            #ax.set_xlabel("Error (geometric − cal, NN − cal)")
            #ax.set_ylabel("Count")
    axes[1,0].legend(loc="upper left", fontsize=8)
    axes[0, 0].set_title(r"$R_{\mathrm{lin}}$ Error (dyne s cm$^{-3}$)")
    axes[0,1].set_title("$L$ Error (dyne s$^2$ cm$^{-3}$)")
    axes[0,0].set_ylabel("Junctions")
    axes[1,0].set_ylabel("Vessels")

    # set figure title
    fig.suptitle("Error in 0D Parameters")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Histograms of geometric vs calibrated R_poiseuille and L for CV test geometries."
    )
    parser.add_argument(
        "set_names",
        nargs="*",
        default=["VMR_rigid_aorta_adults"],
        help="One or more set names (default: VMR_rigid_aorta_adults). Same --run_config for all.",
    )
    parser.add_argument(
        "--run_config",
        default="gen_loss",
        help="Run config subfolder (default: gen_loss)",
    )
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations_EL",
        help="Geometry variant (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root containing zeroD directory (default: data)",
    )
    parser.add_argument(
        "--results_root",
        default="results",
        help="Root containing cross_validation directory (default: results)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output figure path (default: under results/cross_validation/; multi-set dir joins names with __)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="Number of histogram bins (default: 25)",
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
    results_root = os.path.abspath(args.results_root)
    set_names = list(args.set_names)
    if not set_names:
        set_names = ["VMR_rigid_aorta_adults"]

    if len(set_names) == 1:
        print(f"Set: {set_names[0]}")
    else:
        print(f"Combining {len(set_names)} sets: {', '.join(set_names)}")

    diffs = None
    for sn in set_names:
        part = _collect_differences(
            sn,
            args.run_config,
            args.geometry_variant,
            data_root,
            results_root,
        )
        if diffs is None:
            diffs = part
        else:
            _merge_diffs_into(diffs, part)
        n_jR = len(part["junction_R"]["geo_minus_cal"])
        n_jL = len(part["junction_L"]["geo_minus_cal"])
        n_vR = len(part["vessel_R"]["geo_minus_cal"])
        n_vL = len(part["vessel_L"]["geo_minus_cal"])
        print(
            f"  [{sn}] geo−cal counts: junctions R={n_jR} L={n_jL}, vessels R={n_vR} L={n_vL}"
        )

    n_jR_geo = len(diffs["junction_R"]["geo_minus_cal"])
    n_jL_geo = len(diffs["junction_L"]["geo_minus_cal"])
    n_vR_geo = len(diffs["vessel_R"]["geo_minus_cal"])
    n_vL_geo = len(diffs["vessel_L"]["geo_minus_cal"])
    print(
        f"Collected (geo−cal / NN−cal): junctions R={n_jR_geo}/{len(diffs['junction_R']['nn_minus_cal'])} L={n_jL_geo}/{len(diffs['junction_L']['nn_minus_cal'])}, "
        f"vessels R={n_vR_geo}/{len(diffs['vessel_R']['nn_minus_cal'])} L={n_vL_geo}/{len(diffs['vessel_L']['nn_minus_cal'])}"
    )
    _report_average_pct_error(diffs)
    _report_average_error(diffs)
    _report_extreme_errors(diffs)
    _print_latex_tables(diffs)

    if args.output:
        output_path = args.output
    else:
        cross_dir = (
            set_names[0]
            if len(set_names) == 1
            else "__".join(set_names)
        )
        out_dir = os.path.join(
            results_root, "cross_validation", cross_dir, args.run_config
        )
        os.makedirs(out_dir, exist_ok=True)
        base_name = "cv_geometric_vs_calibrated_histograms"
        if len(set_names) > 1:
            base_name = f"{base_name}_{'__'.join(set_names)}"
        output_path = os.path.join(out_dir, f"{base_name}.pdf")

    _plot_histograms(diffs, output_path, nbins=args.bins, dpi=args.dpi)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
