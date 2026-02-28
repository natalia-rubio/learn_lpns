#!/usr/bin/env python3
"""
Scatter plots of ML input features (x) vs lumped parameters (y) using all vessel
data from a given set (e.g. VMR_rigid_aorta_adults).

Edit FEATURES_TO_PLOT and LUMPED_PARAMS below to choose which input features
and which target parameters to plot. All data from the set is combined.

Usage:
  python util/visualizations/plot_feature_vs_lumped_params.py VMR_rigid_aorta_adults
  python util/visualizations/plot_feature_vs_lumped_params.py VMR_rigid_aorta_adults --geometry-variant bifurcations --output-dir results/plots/feature_vs_params
"""
import os
import re
import argparse
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Options: specify which input features (x) and lumped parameters (y) to plot.
# Use exact column names from vessel_geometric_features.csv and vessel_lumped_parameters.csv.
# -----------------------------------------------------------------------------
FEATURES_TO_PLOT = [
    "inlet_max_inscribed_radius",
    "vessel_length",
    "path_length",
    "area_ratio",
]
LUMPED_PARAMS = [
    "R_poiseuille",
    "stenosis_coefficient",
    "L",
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _sanitize_filename(s):
    """Replace characters unsuitable for filenames with underscore."""
    return re.sub(r"[^\w\-.]", "_", s)


def discover_geometries_with_vessel_csvs(ml_inputs_root, set_name, geometry_variant):
    """Return sorted list of geometry names that have vessel feature and target CSVs."""
    base = os.path.join(ml_inputs_root, set_name, geometry_variant)
    if not os.path.isdir(base):
        return []
    geos = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "vessel_geometric_features.csv")) and os.path.isfile(
            os.path.join(path, "vessel_lumped_parameters.csv")
        ):
            geos.append(name)
    return sorted(geos)


def load_vessel_data(ml_inputs_root, set_name, geometry_variant, geometries):
    """
    Load and concatenate vessel features and lumped parameters for all geometries.
    Returns a single DataFrame with feature columns + lumped param columns + 'geometry'.
    """
    feature_cols = None
    rows = []
    for geo in geometries:
        feat_path = os.path.join(
            ml_inputs_root, set_name, geometry_variant, geo, "vessel_geometric_features.csv"
        )
        tgt_path = os.path.join(
            ml_inputs_root, set_name, geometry_variant, geo, "vessel_lumped_parameters.csv"
        )
        df_f = pd.read_csv(feat_path)
        df_t = pd.read_csv(tgt_path)
        # Target CSV may have vessel_name (string); keep only numeric target columns we need
        tgt_numeric = [c for c in LUMPED_PARAMS if c in df_t.columns]
        if len(tgt_numeric) == 0:
            raise ValueError(f"No target columns {LUMPED_PARAMS} in {tgt_path}")
        if feature_cols is None:
            feature_cols = list(df_f.columns)
        if len(df_f) != len(df_t):
            raise ValueError(
                f"Row count mismatch for {geo}: features {len(df_f)}, targets {len(df_t)}"
            )
        combined = df_f.copy()
        for c in tgt_numeric:
            combined[c] = df_t[c].values
        combined["geometry"] = geo
        rows.append(combined)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Plot input features vs lumped parameters (vessel data) for a set."
    )
    parser.add_argument(
        "set_name",
        help="Set name (e.g. VMR_rigid_aorta_adults)",
    )
    parser.add_argument(
        "--geometry-variant",
        default="bifurcations_EL",
        help="Geometry variant under ml_inputs (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root containing ml_inputs (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots (default: results/plots/feature_vs_lumped_params/<set_name>_<variant>)",
    )
    parser.add_argument(
        "--no-geometry-legend",
        action="store_true",
        help="Do not color points by geometry / omit geometry legend",
    )
    args = parser.parse_args()

    ml_inputs_root = os.path.join(REPO_ROOT, args.data_root, "ml_inputs")
    if args.output_dir is None:
        args.output_dir = os.path.join(
            REPO_ROOT,
            "results",
            "plots",
            "feature_vs_lumped_params",
            f"{args.set_name}_{args.geometry_variant}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    geometries = discover_geometries_with_vessel_csvs(
        ml_inputs_root, args.set_name, args.geometry_variant
    )
    if not geometries:
        print(
            f"No geometries with vessel CSVs found under {ml_inputs_root}/{args.set_name}/{args.geometry_variant}"
        )
        sys.exit(1)
    print(f"Found {len(geometries)} geometries: {geometries}")

    df = load_vessel_data(ml_inputs_root, args.set_name, args.geometry_variant, geometries)
    print(f"Total vessel rows: {len(df)}")

    # Restrict to features/params that exist
    feats = [f for f in FEATURES_TO_PLOT if f in df.columns]
    params = [p for p in LUMPED_PARAMS if p in df.columns]
    missing_f = set(FEATURES_TO_PLOT) - set(feats)
    missing_p = set(LUMPED_PARAMS) - set(params)
    if missing_f:
        print(f"Warning: feature columns not in CSV (skipped): {missing_f}")
    if missing_p:
        print(f"Warning: lumped param columns not in CSV (skipped): {missing_p}")
    if not feats or not params:
        print("No feature/param columns to plot. Exiting.")
        sys.exit(1)

    color_by_geometry = not args.no_geometry_legend and df["geometry"].nunique() > 1
    if color_by_geometry:
        geos = df["geometry"].unique()
        # Fixed colors so each geometry is distinct (avoid colormap API differences across mpl versions)
        try:
            colors = getattr(plt.cm.tab10, "colors", None) or getattr(plt.cm.get_cmap("tab10"), "colors", None)
        except Exception:
            colors = None
        if colors is None:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        geo_to_color = {g: colors[i % len(colors)] for i, g in enumerate(geos)}

    for feat in feats:
        for param in params:
            x = df[feat].values
            y = df[param].values
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                print(f"Skipping {feat} vs {param}: no finite pairs")
                continue
            x, y = x[valid], y[valid]

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            if color_by_geometry:
                geo_vals = df.loc[valid, "geometry"].values
                for g in np.unique(geo_vals):
                    mask = geo_vals == g
                    ax.scatter(
                        x[mask],
                        y[mask],
                        c=[geo_to_color[g]],
                        label=g,
                        alpha=0.7,
                        s=24,
                        edgecolors="none",
                    )
                ax.legend(loc="best", fontsize=7)
            else:
                ax.scatter(x, y, alpha=0.7, s=24, c="steelblue", edgecolors="none")

            ax.set_xlabel(feat, fontsize=10)
            ax.set_ylabel(param, fontsize=10)
            ax.set_title(f"{feat} vs {param}\n{args.set_name} / {args.geometry_variant} (n={len(x)})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_name = f"feature_{_sanitize_filename(feat)}_vs_{_sanitize_filename(param)}.png"
            out_path = os.path.join(args.output_dir, out_name)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")

    print(f"Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
