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
                             "data/feature_histograms/{set_name}"

    Returns:
        Dictionary with keys "input", "output_{output_type}", and "scaling_factors"
    """
    if output_type not in {"rri", "ri", "rr"}:
        raise ValueError(f"Unsupported output_type: {output_type}")

    all_inputs: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []


    include_features = [
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
    ]
    #include_features = None
    # Track feature order for consistency across geometries
    feature_order: Optional[List[str]] = None

    def require_cols(col_to_idx: Dict[str, int], needed: List[str], ctx: str) -> List[int]:
        missing = [c for c in needed if c not in col_to_idx]
        if missing:
            raise ValueError(f"Missing required columns in {ctx}: {missing}")
        return [col_to_idx[c] for c in needed]

    for geo in geometries:
        geom_csv = os.path.join(ml_inputs_root, set_name, geo, "geometric_features.csv")
        out_csv = os.path.join(ml_inputs_root, set_name, geo, "junction_lumped_parameters.csv")

        geom_header, geom_X = _read_csv_matrix(geom_csv)
        out_header, out_Y = _read_csv_matrix(out_csv)

        if require_same_rows and geom_X.shape[0] != out_Y.shape[0]:
            raise ValueError(
                f"Row mismatch for geo {geo}: geometric_features has {geom_X.shape[0]} rows, "
                f"junction_lumped_parameters has {out_Y.shape[0]} rows"
            )

        # Filter features if include_features is specified
        if include_features is not None:
            # Validate that all requested features exist
            missing_features = [f for f in include_features if f not in geom_header]
            if missing_features:
                raise ValueError(
                    f"Missing requested features in {geo}: {missing_features}. "
                    f"Available features: {geom_header}"
                )
            
            # Set feature order on first geometry, then validate consistency
            if feature_order is None:
                feature_order = include_features.copy()
            else:
                # Ensure feature order is consistent
                if set(include_features) != set(feature_order):
                    raise ValueError(
                        f"Feature list mismatch: first geometry had {feature_order}, "
                        f"but {geo} has different features requested"
                    )
            
            # Get column indices for requested features in the specified order
            col_to_idx = {name: idx for idx, name in enumerate(geom_header)}
            feature_indices = require_cols(col_to_idx, feature_order, f"geometric_features for {geo}")
            
            # Extract only the requested features in the specified order
            geom_X = geom_X[:, feature_indices]
        else:
            # Use all features, set feature_order on first geometry for consistency check
            if feature_order is None:
                feature_order = geom_header.copy()
            else:
                # Validate that all geometries have the same features in the same order
                if geom_header != feature_order:
                    raise ValueError(
                        f"Feature mismatch for {geo}: expected {feature_order}, got {geom_header}"
                    )

        # Use filtered columns from geometric_features
        all_inputs.append(geom_X)
        # Use all columns from junction_lumped_parameters (already includes both outlets)
        all_outputs.append(out_Y)

    input_array = np.vstack(all_inputs)
    output_array = np.vstack(all_outputs)

    # Remap tortuosity values less than 1 to 1
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
        if histogram_output_dir is None:
            # Default output directory
            histogram_output_dir = os.path.join("results", "feature_histograms", set_name)
        
        plot_feature_histograms(
            input_array=input_array,
            feature_names=feature_order,
            output_dir=histogram_output_dir,
            set_name=set_name,
            num_geos=len(geometries),
        )

    n = input_array.shape[0]
    scaling_factors = np.ones((n, 1), dtype=float)

    if jnp is not None:
        data_dict = {
            "input": jnp.asarray(input_array),
            f"output_{output_type}": jnp.asarray(output_array),
            "scaling_factors": jnp.asarray(scaling_factors),
        }
    else:
        data_dict = {
            "input": input_array,
            f"output_{output_type}": output_array,
            "scaling_factors": scaling_factors,
        }
    return data_dict


__all__ = [
    "build_data_dict_from_csvs",
]


