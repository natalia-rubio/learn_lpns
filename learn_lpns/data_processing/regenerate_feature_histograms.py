#!/usr/bin/env python3
"""
Regenerate feature histogram figures for a given set and run config.

Reads ML input CSVs from data/ml_inputs/{set_name}/{run_config}/{geometry_variant}/
and saves histograms to data/feature_histograms/{set_name}/{run_config}/{geometry_variant}/{set_type}/.

Example (quadratic_resistor_penalty_on_gen_loss for VMR_rigid_aorta_adults, both geometry variants):
  python -m learn_lpns.data_processing.regenerate_feature_histograms VMR_rigid_aorta_adults \\
      --run_config quadratic_resistor_penalty_on_gen_loss

Example (single geometry variant):
  python -m learn_lpns.data_processing.regenerate_feature_histograms VMR_rigid_aorta_adults \\
      --run_config base --geometry_variant bifurcations
"""

import argparse
import os

from learn_lpns.data_processing.data_dict_from_csvs import (
    build_data_dict_from_csvs,
    feature_histograms_dir,
)
from learn_lpns.data_processing.run_data_processing import discover_geometries_with_csvs


def main():
    parser = argparse.ArgumentParser(description="Regenerate feature histogram figures from ML input CSVs.")
    parser.add_argument(
        "set_name",
        help="Set name; must match directory under data/ml_inputs/ (e.g., VMR_rigid_aorta_adults)",
    )
    parser.add_argument(
        "--run_config",
        default=None,
        help="Run config suffix; CSVs read from data/ml_inputs/{set_name}/{run_config}/... "
        "(omit for legacy layouts without a config subfolder)",
    )
    parser.add_argument(
        "--geometry_variant",
        default="all",
        choices=["bifurcations", "bifurcations_EL", "all"],
        help="Geometry variant(s) to process (default: all)",
    )
    parser.add_argument(
        "--set_type",
        default="all",
        help="Cohort folder tier (default: all; matches jax_arrays layout)",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Repo data root (default: data)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Override histogram output directory (default: data/feature_histograms/... mirroring jax_arrays path)",
    )
    args = parser.parse_args()

    run_config = (args.run_config or "").strip() or None
    if args.geometry_variant == "all":
        variants = ["bifurcations", "bifurcations_EL"]
    else:
        variants = [args.geometry_variant]

    ml_inputs_root = (
        os.path.join(args.data_root, "ml_inputs", args.set_name, run_config)
        if run_config
        else os.path.join(args.data_root, "ml_inputs", args.set_name)
    )
    set_name_for_path = "" if run_config else args.set_name

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

        histogram_dir = args.output_dir or feature_histograms_dir(
            args.set_name,
            geometry_variant,
            set_type=args.set_type,
            run_config_suffix=run_config,
            data_root=args.data_root,
        )
        print(
            f"Regenerating histograms for {args.set_name} run_config={run_config} "
            f"{geometry_variant} ({len(geometries)} geos) -> {histogram_dir}"
        )

        build_data_dict_from_csvs(
            set_name=set_name_for_path,
            geometries=geometries,
            ml_inputs_root=ml_inputs_root,
            plot_histograms=True,
            histogram_output_dir=histogram_dir,
            geometry_variant=geometry_variant,
            cohort_set_name=args.set_name,
            run_config_suffix=run_config,
            set_type=args.set_type,
            data_root=args.data_root,
        )

    print("Done.")


if __name__ == "__main__":
    main()
