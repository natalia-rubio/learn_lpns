#!/usr/bin/env python3
"""
Build `self.data_dict` compatible with `learn_lpns/neural_network` from the ML CSVs.

Inputs:
  - `data/ml_inputs/<set_name>/<geo>/geometric_features.csv`
  - `data/ml_inputs/<set_name>/<geo>/junction_lumped_parameters.csv`

Outputs:
  - A Python dict containing JAX arrays with keys expected by `NeuralNet`:
      - input, output_rri
      - input_mean, input_std, output_mean, output_std (dataset stats; arrays stay raw)
      - generation, output_min, output_max
      - geometry_row_ranges, geometry_names_order
      - row_junction_names, row_primary_outlet_names, outlet_vessel_ids, input_feature_names
        (junction pickles only; used for deploy-time NN inference)

This is intentionally "strict": missing columns raise ValueError.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

# Try to import matplotlib for histogram generation
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def _setup_latex_histograms() -> bool:
    """Enable LaTeX rendering for matplotlib if available.

    Called by: :func:`_plot_column_histograms`.
    """
    if not HAS_MATPLOTLIB:
        return False
    try:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "cm"
        return True
    except Exception:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif"]
        return False


def _feature_name_to_latex(name: str) -> str:
    """Return LaTeX-safe string for a feature name (escape underscores).

    Called by: :func:`_plot_column_histograms`.
    """
    escaped = name.replace("_", r"\_")
    return rf"\texttt{{{escaped}}}"


def get_default_include_features() -> list[str]:
    """
    Default junction NN input column names (primary-outlet feature set).

    Called by: :func:`filter_features_from_array`, :func:`build_data_dict_from_csvs`,
    ``generate_zerod_inputs.py``.
    """
    return [
        "inlet_max_inscribed_radius",
        "outlet0_max_inscribed_radius_local",
        "outlet0_max_inscribed_radius_min_on_path",
        "outlet0_max_inscribed_radius_max_on_path",
        "outlet0_rmin_rat",
        "outlet0_rmax_rat",
        "outlet0_path_length",
        "outlet0_tortuosity",
        "outlet0_angle_diff",
        "outlet0_radius_ratio",
        "outlet0_poiseuille_resistance_calc",
        "outlet0_inductance_calc",
        "outlet0_absorbed_R_poiseuille",
        "outlet0_absorbed_L",
        "outlet0_absorbed_stenosis_coefficient",
        "flow_split_inv",
        "outlet0_rneg4",
        "outlet0_rneg2",
        "outlet0_nd_length",
    ]
    # return [
    #     "inlet_max_inscribed_radius",
    #     "outlet0_max_inscribed_radius_local",
    #     "outlet1_max_inscribed_radius_local",
    #     "outlet0_max_inscribed_radius_min_on_path",
    #     "outlet1_max_inscribed_radius_min_on_path",
    #     "outlet0_max_inscribed_radius_max_on_path",
    #     "outlet1_max_inscribed_radius_max_on_path",
    #     "outlet0_path_length",
    #     "outlet1_path_length",
    #     "outlet0_tortuosity",
    #     "outlet1_tortuosity",
    #     "outlet0_angle_diff",
    #     "outlet1_angle_diff",
    #     "outlet0_radius_ratio",
    #     "outlet1_radius_ratio",
    #     "outlet0_poiseuille_resistance_calc",
    #     "outlet1_poiseuille_resistance_calc",
    #     "outlet0_inductance_calc",
    #     "outlet1_inductance_calc",
    #     "outlet0_absorbed_R_poiseuille",
    #     "outlet1_absorbed_R_poiseuille",
    #     "outlet0_absorbed_L",
    #     "outlet1_absorbed_L",
    #     "outlet0_absorbed_stenosis_coefficient",
    #     "outlet1_absorbed_stenosis_coefficient",
    #     "flow_split"
    # ]


def get_default_include_outputs() -> list[str]:
    """
    Default junction NN output column names (R, stenosis, L for primary outlet).

    Called by: :func:`filter_outputs_from_array`, :func:`build_data_dict_from_csvs`.
    """
    return ["R_poiseuille_outlet0", "stenosis_coefficient_outlet0", "L_outlet0"]


def get_default_include_features_vessel() -> list[str]:
    """
    Default vessel NN input column names.

    Order matches ``inputs_from_0d_config.load_vessel_geometric_features``.

    Called by: :func:`filter_features_from_array`, :func:`build_data_dict_from_vessel_csvs`,
    ``generate_zerod_inputs.py``.
    """
    return [
        "is_inlet",
        "vessel_length",
        # "inlet_area",
        # "outlet_area",
        "path_length",
        "tortuosity",
        "angle_diff",
        # "area_ratio",
        "inlet_max_inscribed_radius",
        "outlet_max_inscribed_radius",
        "max_inscribed_radius_min",
        "max_inscribed_radius_max",
        "radius_ratio",
        "rmin_rat",
        "rmax_rat",
        "nd_length",
        "poiseuille_resistance_calc",
        "inductance_calc",
        "stenosis_calc",
        "rneg4",
        "rneg2",
        "R_poiseuille_geometric",
        "L_geometric",
        "stenosis_coefficient_geometric",
    ]


def get_default_include_outputs_vessel() -> list[str]:
    """
    Default vessel NN output column names (R_poiseuille, stenosis_coefficient, L).

    Called by: :func:`build_data_dict_from_vessel_csvs`, :func:`_read_vessel_lumped_parameters_csv`.
    """
    return [
        "R_poiseuille",
        "stenosis_coefficient",
        "L",
    ]


def filter_features_from_array(
    X: np.ndarray,
    feature_names: list[str],
    include_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Select and reorder feature columns from a numeric matrix.

    Calls: :func:`get_default_include_features` (when ``include_features`` is omitted).
    Called by: :func:`build_data_dict_from_csvs`, :func:`build_data_dict_from_vessel_csvs`,
    ``generate_zerod_inputs.py``. Tortuosity clamping is done later via :func:`_clamp_tortuosity`.
    """
    if include_features is None:
        include_features = get_default_include_features()

    missing_features = [f for f in include_features if f not in feature_names]
    if missing_features:
        raise ValueError(f"Missing requested features: {missing_features}. Available features: {feature_names}")

    col_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    feature_indices = [col_to_idx[f] for f in include_features]
    X_filtered = X[:, feature_indices]

    return X_filtered, include_features


def filter_outputs_from_array(
    Y: np.ndarray,
    output_names: list[str],
    include_outputs: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Select and reorder output columns from a numeric matrix.

    Calls: :func:`get_default_include_outputs` (when ``include_outputs`` is omitted).
    Called by: :func:`build_data_dict_from_csvs`.
    """
    if include_outputs is None:
        include_outputs = get_default_include_outputs()

    # Validate that all requested outputs exist
    missing_outputs = [o for o in include_outputs if o not in output_names]
    if missing_outputs:
        raise ValueError(f"Missing requested outputs: {missing_outputs}. Available outputs: {output_names}")

    # Get column indices for requested outputs in the specified order
    col_to_idx = {name: idx for idx, name in enumerate(output_names)}
    output_indices = [col_to_idx[o] for o in include_outputs]

    # Extract only the requested outputs in the specified order
    Y_filtered = Y[:, output_indices]

    return Y_filtered, include_outputs


def _parse_float_cell(x: str) -> float:
    """Parse a CSV cell as float; empty or non-finite values raise ValueError.

    Called by: :func:`_read_csv_matrix`, :func:`_read_csv_numeric_columns`,
    :func:`_read_vessel_lumped_parameters_csv`.
    """
    if x is None:
        raise ValueError("Empty numeric cell in CSV")
    s = str(x).strip()
    if s == "":
        raise ValueError("Empty numeric cell in CSV")
    val = float(s)
    if not np.isfinite(val):
        raise ValueError(f"Non-finite numeric value in CSV cell: {s!r}")
    return val


def _read_csv_matrix(csv_path: str) -> tuple[list[str], np.ndarray]:
    """Load an all-numeric CSV as (header, float matrix).

    Calls: :func:`_parse_float_cell`.
    Called by: :func:`build_data_dict_from_csvs`, :func:`build_data_dict_from_vessel_csvs`.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        rows = []
        for row_idx, r in enumerate(reader, start=2):
            if not r:
                continue
            try:
                row_vals = [_parse_float_cell(x) for x in r]
            except ValueError as exc:
                raise ValueError(f"{csv_path} row {row_idx}: {exc}") from exc
            rows.append(row_vals)

    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")

    X = np.asarray(rows, dtype=float)
    if X.shape[1] != len(header):
        raise ValueError(
            f"CSV column mismatch in {csv_path}: header has {len(header)} columns but data has {X.shape[1]}"
        )
    return header, X


def _read_csv_numeric_columns(csv_path: str, column_names: list[str]) -> tuple[list[str], np.ndarray]:
    """Read selected numeric columns from a CSV (full header, subset array).

    Calls: :func:`_parse_float_cell`.
    Called by: (none in-repo; kept for mixed-type CSV helpers).
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        col_indices = [header.index(c) for c in column_names]
        rows = []
        for row_idx, r in enumerate(reader, start=2):
            if not r:
                continue
            row_vals = []
            for col_name, i in zip(column_names, col_indices, strict=False):
                if i >= len(r):
                    raise ValueError(f"{csv_path} row {row_idx}: missing value for column '{col_name}'")
                try:
                    row_vals.append(_parse_float_cell(r[i]))
                except ValueError as exc:
                    raise ValueError(f"{csv_path} row {row_idx}, column '{col_name}': {exc}") from exc
            rows.append(row_vals)
    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    return header, np.asarray(rows, dtype=float)


def _plot_column_histograms(
    data_array: np.ndarray,
    column_names: list[str],
    output_dir: str,
    set_name: str,
    num_geos: int,
) -> int:
    """
    Plot one histogram per column (PNG + PDF).

    Calls: :func:`_setup_latex_histograms`, :func:`_feature_name_to_latex`.
    Called by: :func:`plot_feature_histograms`, :func:`plot_lumped_param_histograms`.
    """
    if not HAS_MATPLOTLIB:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    n_cols = len(column_names)
    if data_array.shape[1] != n_cols:
        raise ValueError(
            f"Column count mismatch: data_array has {data_array.shape[1]} columns, but {n_cols} names provided"
        )

    use_latex = _setup_latex_histograms()
    if use_latex:
        xlabel, ylabel = r"Value", r"Frequency"
    else:
        xlabel, ylabel = "Value", "Frequency"

    n_samples = data_array.shape[0]
    set_label = set_name if set_name else ""

    for idx, col_name in enumerate(column_names):
        col_data = data_array[:, idx]
        col_data_clean = col_data[np.isfinite(col_data)]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if len(col_data_clean) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            title_str = col_name
        else:
            n_bins = min(50, max(10, int(np.sqrt(len(col_data_clean)))))
            ax.hist(col_data_clean, bins=n_bins, edgecolor="black", alpha=0.7)
            mean_val = np.mean(col_data_clean)
            std_val = np.std(col_data_clean)
            n_val = len(col_data_clean)
            if use_latex:
                stats_text = rf"Mean: ${mean_val:.4f}$" + "\n" + rf"Std: ${std_val:.4f}$" + "\n" + rf"$N = {n_val}$"
                title_str = _feature_name_to_latex(col_name)
            else:
                stats_text = f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nN: {n_val}"
                title_str = col_name
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_str, fontsize=12)
        ax.grid(True, alpha=0.3)

        if set_label:
            fig.suptitle(
                rf"{set_label} ($n_{{\mathrm{{geos}}}} = {num_geos}$, $n_{{\mathrm{{samples}}}} = {n_samples}$)",
                fontsize=11,
                fontweight="bold",
                y=1.02,
            )
        else:
            fig.suptitle(
                rf"$n_{{\mathrm{{geos}}}} = {num_geos}$, $n_{{\mathrm{{samples}}}} = {n_samples}$",
                fontsize=11,
                fontweight="bold",
                y=1.02,
            )
        plt.tight_layout()

        safe_name = col_name.replace(os.sep, "_").strip() or f"col_{idx}"
        base_path = os.path.join(output_dir, safe_name)
        png_path = base_path + ".png"
        pdf_path = base_path + ".pdf"
        try:
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"  Saved {safe_name} -> {png_path}, {pdf_path}")
        except Exception as e:
            if use_latex and "usetex" in str(e).lower():
                plt.rcParams["text.usetex"] = False
                plt.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.savefig(pdf_path, bbox_inches="tight")
                print(f"  Saved {safe_name} (no LaTeX) -> {png_path}, {pdf_path}")
            else:
                raise
        plt.close(fig)

    return n_cols


def plot_feature_histograms(
    input_array: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    set_name: str,
    num_geos: int,
) -> None:
    """
    Plot one histogram per input feature.

    Calls: :func:`_plot_column_histograms`.
    Called by: :func:`_maybe_plot_histograms`, :func:`regenerate_feature_histograms` (via
    :func:`build_data_dict_from_csvs` with ``plot_histograms=True``).
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping histogram generation")
        return
    n = _plot_column_histograms(input_array, feature_names, output_dir, set_name, num_geos)
    print(f"Saved {n} feature histograms (PNG + PDF) to: {output_dir}")


def plot_lumped_param_histograms(
    output_array: np.ndarray,
    output_names: list[str],
    output_dir: str,
    set_name: str,
    num_geos: int,
) -> None:
    """
    Plot one histogram per lumped-parameter output column.

    Calls: :func:`_plot_column_histograms`.
    Called by: :func:`_maybe_plot_histograms`.
    """
    if not HAS_MATPLOTLIB:
        return
    lumped_dir = os.path.join(output_dir, "lumped_parameters")
    n = _plot_column_histograms(output_array, output_names, lumped_dir, set_name, num_geos)
    if n:
        print(f"Saved {n} junction lumped parameter histograms (PNG + PDF) to: {lumped_dir}")


OUTPUT_RRI_KEY = "output_rri"


def feature_histograms_dir(
    set_name: str,
    geometry_variant: str,
    set_type: str = "all",
    run_config_suffix: str | None = None,
    data_root: str = "data",
) -> str:
    """
    Default output directory for histograms and data-summary CSVs.

    Mirrors ``jax_arrays`` layout::
        {data_root}/feature_histograms/{set_name}/[{run_config}/]{geometry_variant}/{set_type}/

    Called by: :func:`_finalize_stacked_ml_data`, ``regenerate_feature_histograms.py``.
    """
    parts = [data_root, "feature_histograms", set_name]
    if run_config_suffix:
        parts.append(run_config_suffix)
    parts.extend([geometry_variant, set_type])
    return os.path.join(*parts)


def _require_consistent_column_order(
    selected: list[str],
    canonical: list[str] | None,
    *,
    kind: str,
    geo: str,
) -> list[str]:
    """Return canonical column-name order; error if ``selected`` differs from a prior geometry.

    Called by: :func:`build_data_dict_from_csvs`, :func:`build_data_dict_from_vessel_csvs`.
    """
    if canonical is None:
        return list(selected)
    if selected != canonical:
        raise ValueError(f"{kind} list mismatch: first geometry had {canonical}, but {geo} has {selected}")
    return canonical


def _clamp_tortuosity(input_array: np.ndarray, feature_order: list[str] | None) -> None:
    """In-place: clamp tortuosity feature columns to be at least 1.0.

    Called by: :func:`_finalize_stacked_ml_data`, ``generate_zerod_inputs.py``.
    """
    if not feature_order:
        return
    for idx, name in enumerate(feature_order):
        if "tortuosity" in name.lower():
            input_array[:, idx] = np.maximum(input_array[:, idx], 1.0)


def _read_vessel_lumped_parameters_csv(
    csv_path: str,
    include_outputs: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Read ``vessel_lumped_parameters.csv`` (string ``vessel_name`` + numeric R/S/L columns).

    Returns (lumped_params_array, output_column_names, vessel_names_per_row).

    Calls: :func:`_parse_float_cell`.
    Called by: :func:`build_data_dict_from_vessel_csvs`.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        if "vessel_name" not in header:
            raise ValueError(f"Missing required column 'vessel_name' in {csv_path}")
        vn_idx = header.index("vessel_name")
        try:
            out_indices = [header.index(c) for c in include_outputs]
        except ValueError as exc:
            raise ValueError(f"Missing required output column in {csv_path}: {exc}") from exc
        rows_y: list[list[float]] = []
        vessel_names: list[str] = []
        for row_idx, r in enumerate(reader, start=2):
            if not r:
                continue
            if vn_idx >= len(r) or not str(r[vn_idx]).strip():
                raise ValueError(f"{csv_path} row {row_idx}: missing vessel_name")
            vessel_names.append(str(r[vn_idx]).strip())
            row_y: list[float] = []
            for col_name, i in zip(include_outputs, out_indices, strict=False):
                if i >= len(r):
                    raise ValueError(f"{csv_path} row {row_idx}: missing value for column '{col_name}'")
                try:
                    row_y.append(_parse_float_cell(r[i]))
                except ValueError as exc:
                    raise ValueError(f"{csv_path} row {row_idx}, column '{col_name}': {exc}") from exc
            rows_y.append(row_y)
    if not rows_y:
        raise ValueError(f"No data rows in CSV: {csv_path}")
    return np.asarray(rows_y, dtype=float), list(include_outputs), vessel_names


def _validate_no_nans(
    input_array: np.ndarray,
    output_array: np.ndarray,
    *,
    input_msg: str,
    output_msg: str,
) -> None:
    """Raise if ``input_array`` or ``output_array`` contains NaN.

    Called by: :func:`_finalize_stacked_ml_data`.
    """
    if np.any(np.isnan(input_array)):
        raise ValueError(input_msg)
    if np.any(np.isnan(output_array)):
        raise ValueError(output_msg)


def _maybe_plot_histograms(
    *,
    input_array: np.ndarray,
    output_array: np.ndarray,
    feature_order: list[str],
    output_order: list[str],
    set_name: str,
    num_geos: int,
    histogram_output_dir: str,
    lumped_param_label: str = "junction",
) -> None:
    """Optionally write feature and lumped-parameter histograms.

    Calls: :func:`plot_feature_histograms`, :func:`plot_lumped_param_histograms`.
    Called by: :func:`_finalize_stacked_ml_data`.
    """
    print("Plotting feature histograms...")
    plot_feature_histograms(
        input_array=input_array,
        feature_names=feature_order,
        output_dir=histogram_output_dir,
        set_name=set_name,
        num_geos=num_geos,
    )
    print(f"Plotting {lumped_param_label} lumped parameter histograms...")
    plot_lumped_param_histograms(
        output_array=output_array,
        output_names=output_order,
        output_dir=histogram_output_dir,
        set_name=set_name,
        num_geos=num_geos,
    )


def _compute_dataset_stats(
    input_array: np.ndarray,
    output_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean/std/min/max stats for stacked input and output arrays.

    Called by: :func:`_finalize_stacked_ml_data`.
    """
    input_mean = np.mean(input_array, axis=0)
    input_std = np.std(input_array, axis=0)
    input_std_safe = input_std.copy()
    input_std_safe[input_std_safe == 0] = 1.0

    output_mean = np.mean(output_array, axis=0)
    output_std = np.std(output_array, axis=0)
    output_std_safe = output_std.copy()
    output_std_safe[output_std_safe == 0] = 1.0
    output_min = np.min(output_array, axis=0)
    output_max = np.max(output_array, axis=0)
    return input_mean, input_std_safe, output_mean, output_std_safe, output_min, output_max


def _write_data_summary_csv(
    *,
    input_array: np.ndarray,
    output_array: np.ndarray,
    feature_order: list[str],
    output_order: list[str],
    row_geo_names: list[str],
    row_instance_names: list[str],
    summary_path: str,
) -> None:
    """Write per-column summary statistics CSV with min/max row provenance.

    Called by: :func:`_finalize_stacked_ml_data`.
    """
    summary_dir = os.path.dirname(summary_path)
    os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(
            [
                "variable",
                "type",
                "mean",
                "std",
                "min",
                "max",
                "min_geometry",
                "min_outlet",
                "max_geometry",
                "max_outlet",
            ]
        )

        def _summary_row_for_col(col: np.ndarray, kind: str, name: str) -> None:
            min_idx = int(np.argmin(col))
            max_idx = int(np.argmax(col))
            writer.writerow(
                [
                    name,
                    kind,
                    f"{np.mean(col):.6g}",
                    f"{np.std(col):.6g}",
                    f"{np.min(col):.6g}",
                    f"{np.max(col):.6g}",
                    row_geo_names[min_idx],
                    row_instance_names[min_idx],
                    row_geo_names[max_idx],
                    row_instance_names[max_idx],
                ]
            )

        for col_idx, fname in enumerate(feature_order):
            _summary_row_for_col(input_array[:, col_idx], "input", fname)
        for col_idx, oname in enumerate(output_order):
            _summary_row_for_col(output_array[:, col_idx], "output", oname)
    print(f"  Saved data summary to {summary_path}")


def _finalize_stacked_ml_data(
    *,
    input_array: np.ndarray,
    output_array: np.ndarray,
    generation_array: np.ndarray,
    feature_order: list[str],
    output_order: list[str],
    geometries: list[str],
    geometry_row_ranges: list[tuple[int, int]],
    row_geo_names: list[str],
    row_instance_names: list[str],
    set_name: str,
    geometry_variant: str,
    input_nan_msg: str,
    output_nan_msg: str,
    summary_filename: str,
    plot_histograms: bool = False,
    histogram_output_dir: str | None = None,
    lumped_param_label: str = "junction",
    cohort_set_name: str | None = None,
    run_config_suffix: str | None = None,
    set_type: str = "all",
    data_root: str = "data",
    row_junction_names: list[str] | None = None,
    row_primary_outlet_names: list[str] | None = None,
    outlet_vessel_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Shared post-stack pipeline: clamp, validate, optional plots, summary CSV, JAX dict.

    Calls: :func:`_clamp_tortuosity`, :func:`_validate_no_nans`, :func:`_maybe_plot_histograms`,
    :func:`_compute_dataset_stats`, :func:`_write_data_summary_csv`.
    Called by: :func:`build_data_dict_from_csvs`, :func:`build_data_dict_from_vessel_csvs`.

    When ``row_junction_names`` is provided (junction cohort only), per-row inference metadata
    is validated and stored in the returned dict.
    """
    _clamp_tortuosity(input_array, feature_order)
    _validate_no_nans(input_array, output_array, input_msg=input_nan_msg, output_msg=output_nan_msg)

    label_set_name = cohort_set_name if cohort_set_name is not None else set_name
    summary_dir = histogram_output_dir or feature_histograms_dir(
        label_set_name,
        geometry_variant,
        set_type=set_type,
        run_config_suffix=run_config_suffix,
        data_root=data_root,
    )
    if plot_histograms:
        _maybe_plot_histograms(
            input_array=input_array,
            output_array=output_array,
            feature_order=feature_order,
            output_order=output_order,
            set_name=label_set_name,
            num_geos=len(geometries),
            histogram_output_dir=summary_dir,
            lumped_param_label=lumped_param_label,
        )

    (
        input_mean,
        input_std_safe,
        output_mean,
        output_std_safe,
        output_min,
        output_max,
    ) = _compute_dataset_stats(input_array, output_array)

    summary_path = os.path.join(summary_dir, summary_filename)
    _write_data_summary_csv(
        input_array=input_array,
        output_array=output_array,
        feature_order=feature_order,
        output_order=output_order,
        row_geo_names=row_geo_names,
        row_instance_names=row_instance_names,
        summary_path=summary_path,
    )

    n_rows = int(input_array.shape[0])
    if row_junction_names is not None:
        if row_primary_outlet_names is None or outlet_vessel_ids is None:
            raise ValueError("row_junction_names requires row_primary_outlet_names and outlet_vessel_ids")
        if not (len(row_junction_names) == len(row_primary_outlet_names) == len(outlet_vessel_ids) == n_rows):
            raise ValueError(
                f"Junction row metadata length mismatch: junctions={len(row_junction_names)}, "
                f"outlets={len(row_primary_outlet_names)}, vessel_ids={len(outlet_vessel_ids)}, "
                f"input rows={n_rows}"
            )

    result: dict[str, Any] = {
        "input": jnp.asarray(input_array),
        OUTPUT_RRI_KEY: jnp.asarray(output_array),
        "generation": jnp.asarray(generation_array, dtype=jnp.float32),
        "input_mean": jnp.asarray(input_mean),
        "input_std": jnp.asarray(input_std_safe),
        "output_mean": jnp.asarray(output_mean),
        "output_std": jnp.asarray(output_std_safe),
        "output_min": jnp.asarray(output_min),
        "output_max": jnp.asarray(output_max),
        "geometry_row_ranges": geometry_row_ranges,
        "geometry_names_order": list(geometries),
    }
    if row_junction_names is not None:
        result["row_junction_names"] = list(row_junction_names)
        result["row_primary_outlet_names"] = list(row_primary_outlet_names)
        result["outlet_vessel_ids"] = [int(v) for v in outlet_vessel_ids]
        result["input_feature_names"] = list(feature_order)
    return result


def build_data_dict_from_csvs(
    set_name: str,
    geometries: list[str],
    ml_inputs_root: str = "data/ml_inputs",
    require_same_rows: bool = True,
    plot_histograms: bool = False,
    histogram_output_dir: str | None = None,
    geometry_variant: str = "bifurcations",
    cohort_set_name: str | None = None,
    run_config_suffix: str | None = None,
    set_type: str = "all",
    data_root: str = "data",
) -> dict[str, Any]:
    """
    Concatenate junction CSVs across geometries and build a JAX ``data_dict``.

    Each junction contributes two rows (swapped outlet order). Outputs are R, stenosis, L
    for the primary outlet (``output_rri``).

    Calls: :func:`_read_csv_matrix`, :func:`filter_features_from_array`,
    :func:`filter_outputs_from_array`, :func:`_require_consistent_column_order`,
    :func:`_finalize_stacked_ml_data`.
    Called by: ``run_data_processing.py``, :func:`build_jax_arrays` CLI,
    :func:`regenerate_feature_histograms`.
    """
    if not geometries:
        raise ValueError("geometries must be non-empty")

    all_inputs: list[np.ndarray] = []
    all_outputs: list[np.ndarray] = []
    all_generation: list[np.ndarray] = []
    geometry_row_ranges: list[tuple[int, int]] = []
    row_offset = 0
    row_geo_names: list[str] = []
    row_instance_names: list[str] = []
    row_junction_names: list[str] = []
    row_primary_outlet_names: list[str] = []
    all_outlet_vessel_ids: list[int] = []

    include_features = get_default_include_features()
    include_outputs = get_default_include_outputs()
    feature_order: list[str] | None = None
    output_order: list[str] | None = None

    for geo in geometries:
        features_csv_path = os.path.join(ml_inputs_root, set_name, geometry_variant, geo, "geometric_features.csv")
        lumped_parameters_csv_path = os.path.join(
            ml_inputs_root, set_name, geometry_variant, geo, "junction_lumped_parameters.csv"
        )

        features_header, features_array = _read_csv_matrix(features_csv_path)
        lumped_params_header, lumped_params_array = _read_csv_matrix(lumped_parameters_csv_path)

        col_to_idx = {name: i for i, name in enumerate(features_header)}
        gi = col_to_idx.get("generation")
        if gi is None:
            raise ValueError(f"Missing required column 'generation' in {features_csv_path}")
        vid_idx = col_to_idx.get("outlet_vessel_id")
        if vid_idx is None:
            raise ValueError(f"Missing required column 'outlet_vessel_id' in {features_csv_path}")
        geo_gen = np.asarray(features_array[:, gi], dtype=float).ravel()
        n_feature_rows = int(features_array.shape[0])
        geo_outlet_vessel_ids = [int(features_array[i, vid_idx]) for i in range(n_feature_rows)]

        if require_same_rows and n_feature_rows != lumped_params_array.shape[0]:
            raise ValueError(
                f"Row mismatch for geo {geo}: geometric_features has {n_feature_rows} rows, "
                f"junction_lumped_parameters has {lumped_params_array.shape[0]} rows"
            )

        meta_csv = os.path.join(ml_inputs_root, set_name, geometry_variant, geo, "geometric_features_meta.csv")
        if not os.path.exists(meta_csv):
            raise FileNotFoundError(f"Missing geometric_features_meta.csv: {meta_csv}. Run data processing first.")
        geo_junction_names: list[str] = []
        geo_primary_outlets: list[str] = []
        with open(meta_csv, newline="") as fm:
            reader = csv.reader(fm)
            next(reader, None)
            for r in reader:
                if len(r) >= 2:
                    geo_junction_names.append(r[0])
                    geo_primary_outlets.append(r[1])
                    row_geo_names.append(geo)
                    row_instance_names.append(r[1])
        if len(geo_junction_names) != n_feature_rows:
            raise ValueError(
                f"Meta row count mismatch for geo {geo}: geometric_features_meta has "
                f"{len(geo_junction_names)} rows, geometric_features has {n_feature_rows}"
            )
        row_junction_names.extend(geo_junction_names)
        row_primary_outlet_names.extend(geo_primary_outlets)
        all_outlet_vessel_ids.extend(geo_outlet_vessel_ids)

        filtered_features_array, selected_features = filter_features_from_array(
            features_array, features_header, include_features=include_features
        )
        feature_order = _require_consistent_column_order(selected_features, feature_order, kind="Feature", geo=geo)

        lumped_params_array, selected_outputs = filter_outputs_from_array(
            lumped_params_array, lumped_params_header, include_outputs=include_outputs
        )
        output_order = _require_consistent_column_order(selected_outputs, output_order, kind="Output", geo=geo)

        n_stack = int(filtered_features_array.shape[0])
        geometry_row_ranges.append((row_offset, row_offset + n_stack))
        row_offset += n_stack
        all_inputs.append(filtered_features_array)
        all_outputs.append(lumped_params_array)
        all_generation.append(geo_gen)

    input_array = np.vstack(all_inputs)
    output_array = np.vstack(all_outputs)
    generation_array = np.concatenate(all_generation, axis=0)
    if row_offset != input_array.shape[0]:
        raise ValueError(f"internal row offset {row_offset} != stacked input rows {input_array.shape[0]}")
    if generation_array.shape[0] != input_array.shape[0]:
        raise ValueError(f"generation row count {generation_array.shape[0]} != input rows {input_array.shape[0]}")
    assert feature_order is not None and output_order is not None

    label_set_name = cohort_set_name if cohort_set_name is not None else set_name
    return _finalize_stacked_ml_data(
        input_array=input_array,
        output_array=output_array,
        generation_array=generation_array,
        feature_order=feature_order,
        output_order=output_order,
        geometries=geometries,
        geometry_row_ranges=geometry_row_ranges,
        row_geo_names=row_geo_names,
        row_instance_names=row_instance_names,
        row_junction_names=row_junction_names,
        row_primary_outlet_names=row_primary_outlet_names,
        outlet_vessel_ids=all_outlet_vessel_ids,
        set_name=set_name,
        geometry_variant=geometry_variant,
        input_nan_msg=(
            "NaN found in junction input array (geometric features). "
            "Check geometric_features.csv for all geometries (e.g. flow_split, flow_split_inv)."
        ),
        output_nan_msg=(
            "NaN found in junction output array (lumped parameters). "
            "Check junction_lumped_parameters.csv for all geometries."
        ),
        summary_filename=f"data_summary_{label_set_name}_num_geos_{len(geometries)}.csv",
        plot_histograms=plot_histograms,
        histogram_output_dir=histogram_output_dir,
        lumped_param_label="junction",
        cohort_set_name=cohort_set_name,
        run_config_suffix=run_config_suffix,
        set_type=set_type,
        data_root=data_root,
    )


def build_data_dict_from_vessel_csvs(
    set_name: str,
    geometries: list[str],
    ml_inputs_root: str = "data/ml_inputs",
    require_same_rows: bool = True,
    plot_histograms: bool = False,
    histogram_output_dir: str | None = None,
    geometry_variant: str = "bifurcations",
    cohort_set_name: str | None = None,
    run_config_suffix: str | None = None,
    set_type: str = "all",
    data_root: str = "data",
) -> dict[str, Any]:
    """
    Concatenate vessel CSVs across geometries and build a JAX ``data_dict``.

    Reads ``vessel_geometric_features.csv`` and ``vessel_lumped_parameters.csv`` per geometry.
    Outputs are R_poiseuille, stenosis_coefficient, L (``output_rri``).

    Calls: :func:`_read_csv_matrix`, :func:`filter_features_from_array`,
    :func:`_read_vessel_lumped_parameters_csv`, :func:`_require_consistent_column_order`,
    :func:`_finalize_stacked_ml_data`.
    Called by: ``run_data_processing.py``.
    """
    if not geometries:
        raise ValueError("geometries must be non-empty")

    include_features = get_default_include_features_vessel()
    include_outputs = get_default_include_outputs_vessel()
    all_inputs: list[np.ndarray] = []
    all_outputs: list[np.ndarray] = []
    all_generation: list[np.ndarray] = []
    geometry_row_ranges: list[tuple[int, int]] = []
    row_offset = 0
    row_geo_names: list[str] = []
    row_instance_names: list[str] = []
    feature_order: list[str] | None = None
    output_order: list[str] | None = None

    for geo in geometries:
        features_csv_path = os.path.join(
            ml_inputs_root, set_name, geometry_variant, geo, "vessel_geometric_features.csv"
        )
        lumped_parameters_csv_path = os.path.join(
            ml_inputs_root, set_name, geometry_variant, geo, "vessel_lumped_parameters.csv"
        )

        features_header, features_array = _read_csv_matrix(features_csv_path)
        feature_name_to_idx = {name: i for i, name in enumerate(features_header)}
        gi = feature_name_to_idx.get("generation")
        if gi is None:
            raise ValueError(f"Missing required column 'generation' in {features_csv_path}")
        vessel_gen = np.asarray(features_array[:, gi], dtype=float).ravel()

        filtered_features_array, selected_features = filter_features_from_array(
            features_array, features_header, include_features=include_features
        )
        feature_order = _require_consistent_column_order(selected_features, feature_order, kind="Feature", geo=geo)

        lumped_params_array, selected_outputs, vessel_names = _read_vessel_lumped_parameters_csv(
            lumped_parameters_csv_path, include_outputs
        )
        output_order = _require_consistent_column_order(selected_outputs, output_order, kind="Output", geo=geo)

        if require_same_rows and filtered_features_array.shape[0] != lumped_params_array.shape[0]:
            raise ValueError(
                f"Vessel row mismatch for {geo}: features has {filtered_features_array.shape[0]} rows, "
                f"lumped parameters has {lumped_params_array.shape[0]} rows"
            )

        n_stack = int(filtered_features_array.shape[0])
        geometry_row_ranges.append((row_offset, row_offset + n_stack))
        row_offset += n_stack
        all_inputs.append(filtered_features_array)
        all_outputs.append(lumped_params_array)
        all_generation.append(vessel_gen)
        row_geo_names.extend([geo] * n_stack)
        row_instance_names.extend(vessel_names)

    input_array = np.vstack(all_inputs)
    output_array = np.vstack(all_outputs)
    generation_array = np.concatenate(all_generation, axis=0)
    if row_offset != input_array.shape[0]:
        raise ValueError(f"internal row offset {row_offset} != stacked input rows {input_array.shape[0]}")
    if generation_array.shape[0] != input_array.shape[0]:
        raise ValueError(f"generation row count {generation_array.shape[0]} != input rows {input_array.shape[0]}")
    assert feature_order is not None and output_order is not None

    label_set_name = cohort_set_name if cohort_set_name is not None else set_name
    return _finalize_stacked_ml_data(
        input_array=input_array,
        output_array=output_array,
        generation_array=generation_array,
        feature_order=feature_order,
        output_order=output_order,
        geometries=geometries,
        geometry_row_ranges=geometry_row_ranges,
        row_geo_names=row_geo_names,
        row_instance_names=row_instance_names,
        set_name=set_name,
        geometry_variant=geometry_variant,
        input_nan_msg=(
            "NaN found in vessel input array (vessel geometric features). "
            "Check vessel_geometric_features.csv for all geometries."
        ),
        output_nan_msg=(
            "NaN found in vessel output array (vessel lumped parameters). "
            "Check vessel_lumped_parameters.csv for all geometries."
        ),
        summary_filename=f"data_summary_vessel_{label_set_name}_num_geos_{len(geometries)}.csv",
        plot_histograms=plot_histograms,
        histogram_output_dir=histogram_output_dir,
        lumped_param_label="vessel",
        cohort_set_name=cohort_set_name,
        run_config_suffix=run_config_suffix,
        set_type=set_type,
        data_root=data_root,
    )


def load_junction_rows_from_jax_dict(
    data_dict: dict[str, Any],
    geo_name: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[int], list[str]]:
    """
    Load junction NN inputs and per-row metadata from a junction jax pickle.

    When ``geo_name`` is set, slice rows for that geometry using ``geometry_row_ranges``.
    Raises if required metadata keys are missing (re-run data processing to rebuild pickle).
    """
    inp = data_dict.get("input")
    if inp is None:
        raise ValueError("Jax pickle missing 'input' array.")

    for key in (
        "row_junction_names",
        "row_primary_outlet_names",
        "outlet_vessel_ids",
        "input_feature_names",
        "geometry_row_ranges",
        "geometry_names_order",
    ):
        if key not in data_dict:
            raise ValueError(f"Jax pickle missing {key!r}. Re-run data processing to rebuild the pickle.")

    n_rows = int(np.asarray(inp).shape[0])
    junction_names = [str(g) for g in data_dict["row_junction_names"]]
    outlet_primary_names = [str(g) for g in data_dict["row_primary_outlet_names"]]
    outlet_vessel_ids = [int(v) for v in data_dict["outlet_vessel_ids"]]
    feature_names = [str(f) for f in data_dict["input_feature_names"]]

    if not (len(junction_names) == len(outlet_primary_names) == len(outlet_vessel_ids) == n_rows):
        raise ValueError(
            f"Jax pickle row metadata length mismatch: junctions={len(junction_names)}, "
            f"outlets={len(outlet_primary_names)}, vessel_ids={len(outlet_vessel_ids)}, "
            f"input rows={n_rows}. Re-run data processing."
        )

    start, end = 0, n_rows
    if geo_name is not None:
        geoms = [str(g) for g in data_dict["geometry_names_order"]]
        ranges = data_dict["geometry_row_ranges"]
        if geo_name not in geoms:
            raise ValueError(f"Geometry {geo_name!r} not in jax pickle geometry_names_order: {geoms}")
        gi = geoms.index(geo_name)
        start, end = int(ranges[gi][0]), int(ranges[gi][1])

    X = np.asarray(inp, dtype=float)[start:end]
    return (
        X,
        junction_names[start:end],
        outlet_primary_names[start:end],
        outlet_vessel_ids[start:end],
        feature_names,
    )


__all__ = [
    "_read_csv_matrix",
    "build_data_dict_from_csvs",
    "build_data_dict_from_vessel_csvs",
    "feature_histograms_dir",
    "filter_features_from_array",
    "filter_outputs_from_array",
    "get_default_include_features",
    "get_default_include_features_vessel",
    "get_default_include_outputs",
    "get_default_include_outputs_vessel",
    "load_junction_rows_from_jax_dict",
    "plot_feature_histograms",
    "plot_lumped_param_histograms",
]
