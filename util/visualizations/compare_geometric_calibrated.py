#!/usr/bin/env python3
"""
Script to compare geometric vs calibrated 0D element values using violin plots.
Shows distributions of resistance, inductance, capacitance, and stenosis coefficient.
"""

import argparse
import json
import os
import sys

import numpy as np

# Try to import matplotlib with compatibility handling
HAS_MATPLOTLIB = False
plt = None
sns = None

try:
    from collections import namedtuple

    try:
        import matplotlib
    except AttributeError as attr_err:
        error_str = str(attr_err)
        if "cbook" in error_str.lower() and ("_ExceptionInfo" in error_str or "_is_pandas_dataframe" in error_str):
            raise attr_err
        else:
            raise

    matplotlib.use("Agg")
    import matplotlib.cbook as cbook

    if not hasattr(cbook, "_is_pandas_dataframe"):
        cbook._is_pandas_dataframe = lambda x: False
    if not hasattr(cbook, "_ExceptionInfo"):
        _ExceptionInfo = namedtuple("_ExceptionInfo", ["type", "value", "traceback"])
        cbook._ExceptionInfo = _ExceptionInfo

    try:
        import matplotlib.pyplot as plt

        from util.visualizations.matplotlib_tex import configure_matplotlib_latex

        configure_matplotlib_latex(plt)

        plt.rcParams["axes.labelsize"] = 32
        plt.rcParams["axes.titlesize"] = 36
        plt.rcParams["xtick.labelsize"] = 28
        plt.rcParams["ytick.labelsize"] = 28
        plt.rcParams["legend.fontsize"] = 28
        plt.rcParams["figure.titlesize"] = 40

        try:
            import seaborn as sns

            sns.set_style("whitegrid")
        except ImportError:
            sns = None
        HAS_MATPLOTLIB = True
    except AttributeError as attr_err:
        error_str = str(attr_err)
        if "matplotlib.cbook" in error_str or "cbook" in error_str:
            if "_ExceptionInfo" in error_str:
                from collections import namedtuple

                cbook._ExceptionInfo = namedtuple("_ExceptionInfo", ["type", "value", "traceback"])
            elif "_is_pandas_dataframe" in error_str:
                cbook._is_pandas_dataframe = lambda x: False
            import matplotlib.pyplot as plt

            try:
                import seaborn as sns

                sns.set_style("whitegrid")
            except ImportError:
                sns = None
            HAS_MATPLOTLIB = True
        else:
            raise
except (ImportError, AttributeError) as e:
    error_str = str(e)
    is_cbook_error = "cbook" in error_str.lower() and (
        "_ExceptionInfo" in error_str
        or "_is_pandas_dataframe" in error_str
        or "cannot import name" in error_str
        or "has no attribute" in error_str
    )
    if not is_cbook_error:
        print(f"Warning: matplotlib not available: {e}")
        print("  Install with: pip install matplotlib seaborn")


def extract_element_values(json_path, label):
    """
    Extract 0D element values from JSON file (both vessels and junctions).

    Args:
        json_path: Path to JSON file
        label: Label for the data source ('geometric' or 'calibrated')

    Returns:
        Tuple of (vessel_values, junction_values) dictionaries
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    vessels = data.get("vessels", [])
    junctions = data.get("junctions", [])

    # Initialize dictionaries for vessel elements
    vessel_values = {"R_poiseuille": [], "L": [], "stenosis_coefficient": []}

    # Initialize dictionaries for junction elements
    junction_values = {"R_poiseuille": [], "L": [], "stenosis_coefficient": []}

    # Extract vessel values
    for vessel in vessels:
        if "zero_d_element_values" in vessel:
            values = vessel["zero_d_element_values"]
            for element_type in vessel_values.keys():
                if element_type in values:
                    vessel_values[element_type].append(values[element_type])

    # Extract junction values (junctions have arrays of values, one per outlet vessel)
    for junction in junctions:
        if "junction_values" in junction:
            values = junction["junction_values"]
            for element_type in junction_values.keys():
                if element_type in values:
                    # Values are arrays, flatten them
                    if isinstance(values[element_type], list):
                        junction_values[element_type].extend(values[element_type])
                    else:
                        junction_values[element_type].append(values[element_type])

    return vessel_values, junction_values


def create_violin_plots(
    geometric_vessel_values,
    calibrated_vessel_values,
    geometric_junction_values,
    calibrated_junction_values,
    output_path,
    geo_name=None,
    log_scale=None,
):
    """
    Create violin plots comparing geometric vs calibrated values for vessels and junctions.

    Args:
        geometric_vessel_values: Dictionary of vessel element values from geometric input
        calibrated_vessel_values: Dictionary of vessel element values from calibrated output
        geometric_junction_values: Dictionary of junction element values from geometric input
        calibrated_junction_values: Dictionary of junction element values from calibrated output
        output_path: Path to save the plot
        geo_name: Geometry name (for title)
        log_scale: List of element types to use log scale (default: None, no log scale)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        print("  Install with: pip install matplotlib seaborn")
        return False

    if log_scale is None:
        log_scale = []

    # Element labels for display (using LaTeX formatting)
    element_labels = {
        "R_poiseuille": r"Resistance ($R$)",
        "L": r"Inductance ($L$)",
        "stenosis_coefficient": r"Stenosis Coefficient",
    }

    # Create figure with subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    element_types = ["R_poiseuille", "L", "stenosis_coefficient"]

    # First row: Vessel elements
    for idx, element_type in enumerate(element_types):
        ax = axes[0, idx]
        element_label = element_labels[element_type]

        # Get vessel data for this element
        geo_vals = np.array(geometric_vessel_values.get(element_type, []))
        cal_vals = np.array(calibrated_vessel_values.get(element_type, []))

        # Filter zeros for log scale
        if element_type in log_scale:
            geo_vals = geo_vals[geo_vals > 0]
            cal_vals = cal_vals[cal_vals > 0]

        if len(geo_vals) == 0 and len(cal_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(element_label)
            continue

        # Prepare data for this element
        data_list = []
        labels_list = []

        if len(geo_vals) > 0:
            data_list.append(geo_vals)
            labels_list.append("Geometric")

        if len(cal_vals) > 0:
            data_list.append(cal_vals)
            labels_list.append("Calibrated")

        # Create violin plot
        # Color mapping: Geometric (blue), Calibrated (red)
        color_map = {"Geometric": "#3498db", "Calibrated": "#e74c3c"}

        if sns is not None:
            # Use seaborn if available
            import pandas as pd

            df = pd.DataFrame(
                {
                    "Value": np.concatenate(data_list),
                    "Type": np.repeat(labels_list, [len(d) for d in data_list]),
                }
            )
            # Create palette based on actual labels in data
            palette = [color_map[label] for label in labels_list]
            sns.violinplot(
                data=df,
                x="Type",
                y="Value",
                ax=ax,
                palette=palette,
                order=["Geometric", "Calibrated"],
            )
        else:
            # Fallback to matplotlib
            parts = ax.violinplot(data_list, positions=range(len(data_list)), showmeans=True, showmedians=True)
            ax.set_xticks(range(len(data_list)))
            ax.set_xticklabels(labels_list)

            # Color the violins based on label, not position
            colors = [color_map[label] for label in labels_list]
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

        # Set log scale if needed
        if element_type in log_scale and len(geo_vals) > 0 and len(cal_vals) > 0:
            ax.set_yscale("log")

        ax.set_title(element_label, fontsize=36, fontweight="bold")
        ax.set_ylabel(r"Value", fontsize=32)
        ax.grid(True, alpha=0.3)

    # Add row label for vessels (moved further left)
    fig.text(
        0.01,
        0.75,
        r"Vessels",
        fontsize=32,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
    )

    # Second row: Junction elements
    for idx, element_type in enumerate(element_types):
        ax = axes[1, idx]
        element_label = element_labels[element_type]

        # Get junction data for this element
        geo_vals = np.array(geometric_junction_values.get(element_type, []))
        cal_vals = np.array(calibrated_junction_values.get(element_type, []))

        # Filter zeros for log scale
        if element_type in log_scale:
            geo_vals = geo_vals[geo_vals > 0]
            cal_vals = cal_vals[cal_vals > 0]

        if len(geo_vals) == 0 and len(cal_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(element_label)
            continue

        # Prepare data for this element
        data_list = []
        labels_list = []

        if len(geo_vals) > 0:
            data_list.append(geo_vals)
            labels_list.append("Geometric")

        if len(cal_vals) > 0:
            data_list.append(cal_vals)
            labels_list.append("Calibrated")

        # Create violin plot
        # Color mapping: Geometric (blue), Calibrated (red)
        color_map = {"Geometric": "#3498db", "Calibrated": "#e74c3c"}

        if sns is not None:
            # Use seaborn if available
            import pandas as pd

            df = pd.DataFrame(
                {
                    "Value": np.concatenate(data_list),
                    "Type": np.repeat(labels_list, [len(d) for d in data_list]),
                }
            )
            # Create palette based on actual labels in data
            palette = [color_map[label] for label in labels_list]
            sns.violinplot(
                data=df,
                x="Type",
                y="Value",
                ax=ax,
                palette=palette,
                order=["Geometric", "Calibrated"],
            )
        else:
            # Fallback to matplotlib
            parts = ax.violinplot(data_list, positions=range(len(data_list)), showmeans=True, showmedians=True)
            ax.set_xticks(range(len(data_list)))
            ax.set_xticklabels(labels_list)

            # Color the violins based on label, not position
            colors = [color_map[label] for label in labels_list]
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

        # Set log scale if needed
        if element_type in log_scale and len(geo_vals) > 0 and len(cal_vals) > 0:
            ax.set_yscale("log")

        ax.set_title(element_label, fontsize=36, fontweight="bold")
        ax.set_ylabel(r"Value", fontsize=32)
        ax.grid(True, alpha=0.3)

    # Add row label for junctions (moved further left)
    fig.text(
        0.01,
        0.25,
        r"Junctions",
        fontsize=32,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
    )

    # Add overall title
    title = r"Geometric vs Calibrated 0D Element Values"
    if geo_name:
        title += f" --- {geo_name}"
    fig.suptitle(title, fontsize=40, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0.03, 0, 1, 1])  # Leave space on left for row labels
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Violin plots saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare geometric vs calibrated 0D element values using violin plots")
    parser.add_argument("--set_name", required=True, help="Set name (e.g., set_1)")
    parser.add_argument("--geo_name", required=True, help="Geometry name (e.g., tree_000)")
    parser.add_argument("--geometric_input", help="Path to geometric input JSON (default: auto-detect)")
    parser.add_argument("--calibrated_output", help="Path to calibrated output JSON (default: auto-detect)")
    parser.add_argument("--output", help="Output plot path (default: auto-generate)")
    parser.add_argument(
        "--output_dir",
        default="results/calibration_difference",
        help="Output directory (default: results/calibration_difference)",
    )
    parser.add_argument(
        "--data_dir",
        default="data/zeroD",
        help="Data directory for input files (default: data/zeroD)",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Enable log scale for resistance and stenosis coefficient",
    )

    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        print("  Install with: pip install matplotlib seaborn")
        sys.exit(1)

    # Auto-detect file paths
    data_dir = os.path.join(args.data_dir, args.set_name, args.geo_name)
    output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)

    if args.geometric_input:
        geometric_path = args.geometric_input
    else:
        geometric_path = os.path.join(data_dir, "geometric_input.json")

    if args.calibrated_output:
        calibrated_path = args.calibrated_output
    else:
        calibrated_path = os.path.join(data_dir, "calibrated_output.json")

    # Check files exist
    if not os.path.exists(geometric_path):
        print(f"Error: Geometric input file not found: {geometric_path}")
        sys.exit(1)

    if not os.path.exists(calibrated_path):
        print(f"Error: Calibrated output file not found: {calibrated_path}")
        sys.exit(1)

    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.geo_name}_element_comparison.png")

    print("=" * 60)
    print("Comparing Geometric vs Calibrated Element Values")
    print("=" * 60)
    print(f"  Geometric input: {geometric_path}")
    print(f"  Calibrated output: {calibrated_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Extract values
    print("\nExtracting element values...")
    geometric_vessel_values, geometric_junction_values = extract_element_values(geometric_path, "geometric")
    calibrated_vessel_values, calibrated_junction_values = extract_element_values(calibrated_path, "calibrated")

    # If geometric junction values are missing, create zeros to match calibrated junction outlets
    for element_type in ["R_poiseuille", "L", "stenosis_coefficient"]:
        if len(geometric_junction_values[element_type]) == 0 and len(calibrated_junction_values[element_type]) > 0:
            # Create zeros matching the number of calibrated junction outlets
            num_outlets = len(calibrated_junction_values[element_type])
            geometric_junction_values[element_type] = [0.0] * num_outlets
            print(f"  Note: Geometric {element_type} junction values missing, assuming {num_outlets} zeros")

    print("  Geometric vessel values:")
    for element_type, values in geometric_vessel_values.items():
        print(f"    {element_type}: {len(values)} vessels")

    print("  Geometric junction values:")
    for element_type, values in geometric_junction_values.items():
        print(f"    {element_type}: {len(values)} junction outlets")

    print("  Calibrated vessel values:")
    for element_type, values in calibrated_vessel_values.items():
        print(f"    {element_type}: {len(values)} vessels")

    print("  Calibrated junction values:")
    for element_type, values in calibrated_junction_values.items():
        print(f"    {element_type}: {len(values)} junction outlets")

    # Determine log scale
    log_scale = ["R_poiseuille", "stenosis_coefficient"] if args.log_scale else []

    # Create plots
    print("\nCreating violin plots...")
    success = create_violin_plots(
        geometric_vessel_values,
        calibrated_vessel_values,
        geometric_junction_values,
        calibrated_junction_values,
        output_path,
        geo_name=args.geo_name,
        log_scale=log_scale,
    )

    if success:
        print(f"\n✓ Successfully created comparison plot: {output_path}")
        sys.exit(0)
    else:
        print("\n✗ Failed to create plot")
        sys.exit(1)


if __name__ == "__main__":
    main()
