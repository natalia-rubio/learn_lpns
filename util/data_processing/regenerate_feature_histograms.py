#!/usr/bin/env python3
"""
Regenerate feature histogram figures for a given set and run config.

Reads ML input CSVs from data/ml_inputs/{set_name}/{run_config}/{geometry_variant}/
and saves histograms to results/feature_histograms/{set_name}/{geometry_variant}/.

Example (base run config for VMR_rigid_aorta_adults, both geometry variants):
  python -m util.data_processing.regenerate_feature_histograms VMR_rigid_aorta_adults --run-config base

Example (single geometry variant, custom output dir):
  python -m util.data_processing.regenerate_feature_histograms VMR_rigid_aorta_adults --run-config base --geometry-variant bifurcations -o results/feature_histograms/VMR_rigid_aorta_adults
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.data_processing.run_data_processing import discover_geometries_with_csvs
from util.data_processing.data_dict_from_csvs import build_data_dict_from_csvs


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate feature histogram figures from ML input CSVs."
    )
    parser.add_argument(
        "set_name",
        help="Set name; must match directory under data/ml_inputs/ (e.g., VMR_rigid_aorta_adults)",
    )
    parser.add_argument(
        "--run-config",
        default="base",
        help="Run config suffix; CSVs read from data/ml_inputs/{set_name}/{run_config}/... (default: base)",
    )
    parser.add_argument(
        "--geometry-variant",
        default="all",
        choices=["bifurcations", "bifurcations_EL", "all"],
        help="Geometry variant(s) to process (default: all)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Repo data root (default: data)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Base directory for histograms (default: results/feature_histograms/{set_name})",
    )
    args = parser.parse_args()

    run_config = (args.run_config or "base").strip()
    if args.geometry_variant == "all":
        variants = ["bifurcations", "bifurcations_EL"]
    else:
        variants = [args.geometry_variant]

    base_out = args.output_dir or os.path.join("results", "feature_histograms", args.set_name)
    ml_inputs_root = os.path.join(args.data_root, "ml_inputs", args.set_name, run_config)
    # build_data_dict_from_csvs expects set_name="" when ml_inputs_root already includes set_name + run_config
    set_name_for_path = ""

    for geometry_variant in variants:
        geometries = discover_geometries_with_csvs(
            args.set_name,
            geometry_variant=geometry_variant,
            data_root=args.data_root,
            run_config_suffix=run_config,
        )
        if not geometries:
            print(f"No geometries found for {args.set_name}/{run_config}/{geometry_variant}; skipping.")
            continue

        histogram_dir = os.path.join(base_out, geometry_variant)
        print(f"Regenerating histograms for {args.set_name} run_config={run_config} {geometry_variant} ({len(geometries)} geos) -> {histogram_dir}")

        build_data_dict_from_csvs(
            set_name=set_name_for_path,
            geometries=geometries,
            output_type="rri",
            ml_inputs_root=ml_inputs_root,
            plot_histograms=True,
            histogram_output_dir=histogram_dir,
            geometry_variant=geometry_variant,
            normalize=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()
