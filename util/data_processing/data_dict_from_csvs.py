#!/usr/bin/env python3
"""
Build `self.data_dict` compatible with `util/neural_network` from the ML CSVs.

Inputs:
  - `data/ml_inputs/<set_name>/<geo>/geometric_features.csv`
  - `data/ml_inputs/<set_name>/<geo>/junction_lumped_parameters.csv`

Outputs:
  - A Python dict containing JAX arrays with keys expected by `NeuralNet`:
      - input_o1, input_o2
      - output_o1_rri, output_o2_rri (and optionally _ri/_rr)
      - scaling_factors

This is intentionally "strict": missing columns raise ValueError.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp  # type: ignore
except (ImportError, ModuleNotFoundError):
    jnp = None

# Try to import matplotlib for histogram generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def get_default_include_features() -> List[str]:
    """
    Get the default list of features to include in the neural network input.
    
    Returns:
        List of feature names to include (13 features total)
    """
    return [
        "inlet_max_inscribed_radius",
        "outlet0_max_inscribed_radius_local",
        "outlet1_max_inscribed_radius_local",
        "outlet0_max_inscribed_radius_min_on_path",
        "outlet1_max_inscribed_radius_min_on_path",
        "outlet0_max_inscribed_radius_max_on_path",
        "outlet1_max_inscribed_radius_max_on_path",
        "outlet0_path_length",
        "outlet1_path_length",
        "outlet0_tortuosity",
        "outlet1_tortuosity",
        "outlet0_angle_diff",
        "outlet1_angle_diff",
        "outlet0_max_inscribed_radius_ratio",
        "outlet1_max_inscribed_radius_ratio",
        "outlet0_poiseuille_resistance_calc",
        "outlet1_poiseuille_resistance_calc",
        "outlet0_inductance_calc",
        "outlet1_inductance_calc",
        "outlet0_absorbed_R_poiseuille",
        "outlet1_absorbed_R_poiseuille",
        "outlet0_absorbed_L",
        "outlet1_absorbed_L",
        "outlet0_absorbed_stenosis_coefficient",
        "outlet1_absorbed_stenosis_coefficient",
    ]

def get_default_include_outputs() -> List[str]:
    """
    Get the default list of outputs to include in the neural network output.
    
    Returns:
        List of output names to include (3 outputs total)
    """
    return [
        "R_poiseuille_outlet0",
        "stenosis_coefficient_outlet0",
        "L_outlet0"
    ]


def filter_features_from_array(
    X: np.ndarray,
    feature_names: List[str],
    include_features: Optional[List[str]] = None,
    remap_tortuosity: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter features from an array to match the specified feature list.
    
    Args:
        X: Input array of shape (n_samples, n_features)
        feature_names: List of feature names corresponding to columns of X
        include_features: List of feature names to include. If None, uses get_default_include_features()
        remap_tortuosity: If True, remap tortuosity values < 1 to 1
    
    Returns:
        Filtered array and list of selected feature names
    """
    if include_features is None:
        include_features = get_default_include_features()
    
    # Validate that all requested features exist
    missing_features = [f for f in include_features if f not in feature_names]
    if missing_features:
        raise ValueError(
            f"Missing requested features: {missing_features}. "
            f"Available features: {feature_names}"
        )
    
    # Get column indices for requested features in the specified order
    col_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    feature_indices = [col_to_idx[f] for f in include_features]
    
    # Extract only the requested features in the specified order
    X_filtered = X[:, feature_indices]
    
    # Apply tortuosity remapping if requested
    if remap_tortuosity:
        tortuosity_indices = [
            idx for idx, name in enumerate(include_features)
            if 'tortuosity' in name.lower()
        ]
        if tortuosity_indices:
            for idx in tortuosity_indices:
                X_filtered[:, idx] = np.maximum(X_filtered[:, idx], 1.0)
    
    return X_filtered, include_features


def filter_outputs_from_array(
    Y: np.ndarray,
    output_names: List[str],
    include_outputs: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter outputs from an array to match the specified output list.
    
    Args:
        Y: Output array of shape (n_samples, n_outputs)
        output_names: List of output names corresponding to columns of Y
        include_outputs: List of output names to include. If None, uses get_default_include_outputs()
    
    Returns:
        Filtered array and list of selected output names
    """
    if include_outputs is None:
        include_outputs = get_default_include_outputs()
    
    # Validate that all requested outputs exist
    missing_outputs = [o for o in include_outputs if o not in output_names]
    if missing_outputs:
        raise ValueError(
            f"Missing requested outputs: {missing_outputs}. "
            f"Available outputs: {output_names}"
        )
    
    # Get column indices for requested outputs in the specified order
    col_to_idx = {name: idx for idx, name in enumerate(output_names)}
    output_indices = [col_to_idx[o] for o in include_outputs]
    
    # Extract only the requested outputs in the specified order
    Y_filtered = Y[:, output_indices]
    
    return Y_filtered, include_outputs


def _read_csv_matrix(csv_path: str) -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        rows = []
        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])

    if not rows:
        raise ValueError(f"No data rows in CSV: {csv_path}")

    X = np.asarray(rows, dtype=float)
    if X.shape[1] != len(header):
        raise ValueError(
            f"CSV column mismatch in {csv_path}: header has {len(header)} columns but data has {X.shape[1]}"
        )
    return header, X


def plot_feature_histograms(
    input_array: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    set_name: str,
    num_geos: int,
) -> None:
    """
    Generate histograms showing the frequency distribution for each feature.
    
    Args:
        input_array: NumPy array of shape (n_samples, n_features)
        feature_names: List of feature names corresponding to columns
        output_dir: Directory to save histogram plots
        set_name: Set name (e.g., "VMR")
        num_geos: Number of geometries processed
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping histogram generation")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_features = len(feature_names)
    if input_array.shape[1] != n_features:
        raise ValueError(
            f"Feature count mismatch: input_array has {input_array.shape[1]} columns, "
            f"but {n_features} feature names provided"
        )
    
    # Create a grid of subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx]
        feature_data = input_array[:, idx]
        
        # Remove NaN and infinite values
        feature_data_clean = feature_data[np.isfinite(feature_data)]
        
        if len(feature_data_clean) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_name, fontsize=12)
            continue
        
        # Create histogram
        n_bins = min(50, max(10, int(np.sqrt(len(feature_data_clean)))))
        ax.hist(feature_data_clean, bins=n_bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(feature_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(feature_data_clean)
        std_val = np.std(feature_data_clean)
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nN: {len(feature_data_clean)}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'Feature Distributions: {set_name} (n_geos={num_geos}, n_samples={input_array.shape[0]})',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'feature_histograms_{set_name}_num_geos_{num_geos}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature histograms to: {output_path}")


def build_data_dict_from_csvs(
    set_name: str,
    geometries: List[str],
    output_type: str = "rri",
    ml_inputs_root: str = "data/ml_inputs",
    require_same_rows: bool = True,
    plot_histograms: bool = True,
    histogram_output_dir: Optional[str] = None,
    geometry_variant: str = "bifurcations",
    normalize: bool = False,
) -> Dict[str, "np.ndarray"]:
    """
    Concatenate multiple geometries' CSVs and build a `data_dict`.

    Each junction contributes 2 rows (swapped outlet order), so we simply concatenate
    all rows together into single `input` and `output` arrays.

    For output_type "rri": output columns are [R_poiseuille, stenosis_coefficient, L]
    per outlet, flattened as [R_outlet0, stenosis_outlet0, L_outlet0, R_outlet1, ...].

    Args:
        set_name: Set name (e.g., "VMR")
        geometries: List of geometry names to process
        output_type: Output type ("rri", "ri", or "rr")
        ml_inputs_root: Root directory for ML inputs
        require_same_rows: Whether to require same number of rows in input and output CSVs
        plot_histograms: Whether to generate histograms for selected features (default: True)
        histogram_output_dir: Directory to save histogram plots. If None, saves to 
                             "data/feature_histograms/{set_name}/{geometry_variant}"
        geometry_variant: Geometry variant name (e.g., "bifurcations" or "bifurcations_EL")

    Returns:
        Dictionary with keys "input", "output_{output_type}", and "scaling_factors"
    """
    if output_type not in {"rri", "ri", "rr"}:
        raise ValueError(f"Unsupported output_type: {output_type}")

    all_inputs: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []
    # Per-row provenance: (geometry_name, primary_outlet_name) for every row
    row_geo_names: List[str] = []
    row_outlet_names: List[str] = []

    # Use default feature and output selection
    include_features = get_default_include_features()
    include_outputs = get_default_include_outputs()
    # Track feature and output order for consistency across geometries
    feature_order: Optional[List[str]] = None
    output_order: Optional[List[str]] = None

    def require_cols(col_to_idx: Dict[str, int], needed: List[str], ctx: str) -> List[int]:
        missing = [c for c in needed if c not in col_to_idx]
        if missing:
            raise ValueError(f"Missing required columns in {ctx}: {missing}")
        return [col_to_idx[c] for c in needed]

    for geo in geometries:
        geom_csv = os.path.join(ml_inputs_root, set_name, geometry_variant, geo, "geometric_features.csv")
        out_csv = os.path.join(ml_inputs_root, set_name, geometry_variant, geo, "junction_lumped_parameters.csv")

        geom_header, geom_X = _read_csv_matrix(geom_csv)
        out_header, out_Y = _read_csv_matrix(out_csv)

        if require_same_rows and geom_X.shape[0] != out_Y.shape[0]:
            raise ValueError(
                f"Row mismatch for geo {geo}: geometric_features has {geom_X.shape[0]} rows, "
                f"junction_lumped_parameters has {out_Y.shape[0]} rows"
            )

        # Load per-row metadata (geometry + primary outlet name)
        meta_csv = os.path.join(ml_inputs_root, set_name, geometry_variant, geo, "geometric_features_meta.csv")
        if os.path.exists(meta_csv):
            with open(meta_csv, "r", newline="") as fm:
                reader = csv.reader(fm)
                meta_header = next(reader, None)
                for r in reader:
                    if len(r) >= 2:
                        row_geo_names.append(geo)
                        row_outlet_names.append(r[1])  # primary_outlet_name
        else:
            # No meta file — fill with placeholder names
            for _ in range(geom_X.shape[0]):
                row_geo_names.append(geo)
                row_outlet_names.append("unknown")

        # Filter features using the reusable function
        geom_X, selected_features = filter_features_from_array(
            geom_X, geom_header, include_features=include_features, remap_tortuosity=False
        )
        # Note: remap_tortuosity=False here because we'll do it after stacking all arrays
        
        # Validate feature order consistency across geometries
        if feature_order is None:
            feature_order = selected_features
        else:
            if selected_features != feature_order:
                raise ValueError(
                    f"Feature list mismatch: first geometry had {feature_order}, "
                    f"but {geo} has {selected_features}"
                )

        # Filter outputs using the reusable function
        out_Y, selected_outputs = filter_outputs_from_array(
            out_Y, out_header, include_outputs=include_outputs
        )
        
        # Validate output order consistency across geometries
        if output_order is None:
            output_order = selected_outputs
        else:
            if selected_outputs != output_order:
                raise ValueError(
                    f"Output list mismatch: first geometry had {output_order}, "
                    f"but {geo} has {selected_outputs}"
                )

        # Use filtered columns from geometric_features and junction_lumped_parameters
        all_inputs.append(geom_X)
        all_outputs.append(out_Y)

    input_array = np.vstack(all_inputs)
    output_array = np.vstack(all_outputs)

    # Remap tortuosity values less than 1 to 1 (after stacking)
    # Tortuosity should be >= 1 (straight line = 1, curved paths > 1)
    if feature_order is not None:
        # Find indices of tortuosity columns
        tortuosity_indices = [
            idx for idx, name in enumerate(feature_order)
            if 'tortuosity' in name.lower()
        ]
        
        if tortuosity_indices:
            # Clamp tortuosity values to be at least 1.0
            for idx in tortuosity_indices:
                input_array[:, idx] = np.maximum(input_array[:, idx], 1.0)

    # Generate histograms if requested
    if plot_histograms and include_features is not None and feature_order is not None:
        print("Plotting histograms...")
        if histogram_output_dir is None:
            # Default output directory includes geometry variant
            histogram_output_dir = os.path.join("data", "feature_histograms", set_name, geometry_variant)
        
        plot_feature_histograms(
            input_array=input_array,
            feature_names=feature_order,
            output_dir=histogram_output_dir,
            set_name=set_name,
            num_geos=len(geometries),
        )

    # --- Compute stats (always, for the summary CSV) ---
    input_mean = np.mean(input_array, axis=0)
    input_std = np.std(input_array, axis=0)
    input_std_safe = input_std.copy()
    input_std_safe[input_std_safe == 0] = 1.0

    output_mean = np.mean(output_array, axis=0)
    output_std = np.std(output_array, axis=0)
    output_std_safe = output_std.copy()
    output_std_safe[output_std_safe == 0] = 1.0

    # --- Write summary CSV (always uses pre-normalization values) ---
    summary_dir = histogram_output_dir or os.path.join("data", "feature_histograms", set_name, geometry_variant)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"data_summary_{set_name}_num_geos_{len(geometries)}.csv")

    with open(summary_path, "w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow([
            "variable", "type", "mean", "std", "min", "max",
            "min_geometry", "min_outlet", "max_geometry", "max_outlet",
        ])

        # Input features
        for col_idx, fname in enumerate(feature_order or []):
            col = input_array[:, col_idx]
            min_idx = int(np.argmin(col))
            max_idx = int(np.argmax(col))
            writer.writerow([
                fname, "input",
                f"{np.mean(col):.6g}", f"{np.std(col):.6g}",
                f"{np.min(col):.6g}", f"{np.max(col):.6g}",
                row_geo_names[min_idx], row_outlet_names[min_idx],
                row_geo_names[max_idx], row_outlet_names[max_idx],
            ])

        # Output targets — include min/max instance provenance
        for col_idx, oname in enumerate(output_order or []):
            col = output_array[:, col_idx]
            min_idx = int(np.argmin(col))
            max_idx = int(np.argmax(col))
            writer.writerow([
                oname, "output",
                f"{np.mean(col):.6g}", f"{np.std(col):.6g}",
                f"{np.min(col):.6g}", f"{np.max(col):.6g}",
                row_geo_names[min_idx], row_outlet_names[min_idx],
                row_geo_names[max_idx], row_outlet_names[max_idx],
            ])

    print(f"  Saved data summary to {summary_path}")

    # --- Conditionally apply z-normalization ---
    if normalize:
        input_array = (input_array - input_mean) / input_std_safe
        output_array = (output_array - output_mean) / output_std_safe

        print(f"  Z-normalization applied:")
        print(f"    Input  mean range: [{input_mean.min():.4f}, {input_mean.max():.4f}]")
        print(f"    Input  std  range: [{input_std_safe.min():.4f}, {input_std_safe.max():.4f}]")
        print(f"    Output mean range: [{output_mean.min():.4f}, {output_mean.max():.4f}]")
        print(f"    Output std  range: [{output_std_safe.min():.4f}, {output_std_safe.max():.4f}]")
    else:
        print(f"  Z-normalization: OFF (raw values used)")

    n = input_array.shape[0]
    scaling_factors = np.ones((n, 1), dtype=float)

    if jnp is not None:
        data_dict = {
            "input": jnp.asarray(input_array),
            f"output_{output_type}": jnp.asarray(output_array),
            "scaling_factors": jnp.asarray(scaling_factors),
            "normalized": normalize,
        }
        if normalize:
            data_dict["input_mean"] = jnp.asarray(input_mean)
            data_dict["input_std"] = jnp.asarray(input_std_safe)
            data_dict["output_mean"] = jnp.asarray(output_mean)
            data_dict["output_std"] = jnp.asarray(output_std_safe)
    else:
        data_dict = {
            "input": input_array,
            f"output_{output_type}": output_array,
            "scaling_factors": scaling_factors,
            "normalized": normalize,
        }
        if normalize:
            data_dict["input_mean"] = input_mean
            data_dict["input_std"] = input_std_safe
            data_dict["output_mean"] = output_mean
            data_dict["output_std"] = output_std_safe
    return data_dict


__all__ = [
    "build_data_dict_from_csvs",
    "get_default_include_features",
    "get_default_include_outputs",
    "filter_features_from_array",
    "filter_outputs_from_array",
    "_read_csv_matrix",
]


