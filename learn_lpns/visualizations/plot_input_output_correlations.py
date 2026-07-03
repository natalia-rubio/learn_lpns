#!/usr/bin/env python3
"""
Scatter plots of each NN input feature vs R, S, and L for a cohort.

One figure per input feature with three horizontally arranged subplots (R, S, L).
All rows from the cohort jax pickle are pooled into each scatter.

Usage:
  python -m learn_lpns.visualizations.plot_input_output_correlations VMR_pulmo
  python -m learn_lpns.visualizations.plot_input_output_correlations VMR_pulmo \\
    --geometry_variant bifurcations_EL --run_config gen_loss
  python -m learn_lpns.visualizations.plot_input_output_correlations VMR_pulmo --vessel
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learn_lpns.data_processing.data_dict_from_csvs import (
    get_default_include_features,
    get_default_include_features_vessel,
)
from learn_lpns.neural_network.launch_training import (
    DEFAULT_GEOMETRY_VARIANT,
    _infer_num_geos,
    _jax_arrays_path,
)
from learn_lpns.neural_network.nn_model import L_OUTPUT_COLUMN, R_OUTPUT_COLUMN, S_OUTPUT_COLUMN
from learn_lpns.neural_network.nn_util import ORACLE_OUTPUT_FEATURE_NAMES, OUTPUT_RRI_KEY
from learn_lpns.tools.basic import load_dict
from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.run_config_canonical import DEFAULT_CLI_RUN_CONFIG

OUTPUT_COLUMNS = (R_OUTPUT_COLUMN, S_OUTPUT_COLUMN, L_OUTPUT_COLUMN)
OUTPUT_SHORT_LABELS = ("R", "S", "L")
DEFAULT_OUTPUT_DIR = "results/in_out_correlations"


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def _resolve_input_feature_names(data_dict: dict, *, vessel: bool) -> list[str]:
    names = data_dict.get("input_feature_names")
    if names is not None:
        return [str(n) for n in names]
    n_features = int(np.asarray(data_dict["input"]).shape[1])
    defaults = get_default_include_features_vessel() if vessel else get_default_include_features()
    if len(defaults) == n_features:
        return list(defaults)
    return [f"feature_{i}" for i in range(n_features)]


def load_cohort_from_jax(
    *,
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    num_geos: int,
    run_config_suffix: str | None,
    vessel: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    jax_path = _jax_arrays_path(
        data_root,
        set_name,
        geometry_variant,
        set_type,
        num_geos,
        run_config_suffix,
        vessel=vessel,
        stenosis_clipping_enabled=False,
    )
    if not os.path.isfile(jax_path):
        raise FileNotFoundError(f"Jax pickle not found: {jax_path}")

    data_dict = load_dict(jax_path)
    inputs = np.asarray(data_dict["input"], dtype=float)
    if OUTPUT_RRI_KEY not in data_dict:
        raise KeyError(f"Jax pickle missing {OUTPUT_RRI_KEY!r}: {jax_path}")
    outputs = np.asarray(data_dict[OUTPUT_RRI_KEY], dtype=float)
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError(
            f"Row count mismatch in {jax_path}: input rows={inputs.shape[0]}, "
            f"output rows={outputs.shape[0]}"
        )
    if outputs.shape[1] < len(OUTPUT_COLUMNS):
        raise ValueError(f"Expected at least 3 output columns in {jax_path}, got {outputs.shape[1]}")

    feature_names = _resolve_input_feature_names(data_dict, vessel=vessel)
    if len(feature_names) != inputs.shape[1]:
        raise ValueError(
            f"Feature name count ({len(feature_names)}) != input columns ({inputs.shape[1]})"
        )
    return inputs, outputs, feature_names


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def plot_input_output_correlations(
    inputs: np.ndarray,
    outputs: np.ndarray,
    feature_names: list[str],
    *,
    set_name: str,
    geometry_variant: str,
    modality: str,
    output_dir: str,
    run_config_suffix: str | None = None,
) -> list[str]:
    """Write one figure per input feature; return paths of saved PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []
    n_rows = inputs.shape[0]
    run_label = run_config_suffix or "default"

    for feat_idx, feat_name in enumerate(feature_names):
        x_all = inputs[:, feat_idx]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

        for ax, out_col, short_label, full_label in zip(
            axes,
            OUTPUT_COLUMNS,
            OUTPUT_SHORT_LABELS,
            ORACLE_OUTPUT_FEATURE_NAMES,
            strict=True,
        ):
            y_all = outputs[:, out_col]
            valid = np.isfinite(x_all) & np.isfinite(y_all)
            n_valid = int(np.sum(valid))
            if n_valid == 0:
                ax.set_title(f"{short_label}: no finite data")
                ax.set_xlabel(feat_name)
                ax.set_ylabel(full_label)
                continue

            x = x_all[valid]
            y = y_all[valid]
            ax.scatter(x, y, alpha=0.45, s=16, c="steelblue", edgecolors="none")
            r = _pearson_r(x, y)
            ax.set_title(f"{short_label}  (r={r:.3f}, n={n_valid})")
            ax.set_xlabel(feat_name)
            ax.set_ylabel(full_label)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{feat_name} vs R/S/L — {set_name} / {geometry_variant} ({modality}, n={n_rows}, {run_label})",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"input_{_sanitize_filename(feat_name)}_vs_rsl.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scatter each input feature vs R, S, L (one figure per input, 3 subplots)."
    )
    parser.add_argument("set_name", help="Cohort name (e.g. VMR_pulmo)")
    parser.add_argument(
        "--geometry_variant",
        default=DEFAULT_GEOMETRY_VARIANT,
        help=f"Geometry variant (default: {DEFAULT_GEOMETRY_VARIANT})",
    )
    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        help="Run-config suffix for jax_arrays path (default: gen_loss)",
    )
    parser.add_argument(
        "--set_type",
        default="all",
        help="Set type tier under jax_arrays (default: all)",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Data root containing jax_arrays (default: data)",
    )
    parser.add_argument(
        "--num_geos",
        type=int,
        default=None,
        help="Number of geometries (default: infer from jax_arrays / split_indices)",
    )
    parser.add_argument(
        "--vessel",
        action="store_true",
        help="Use vessel jax pickle instead of junction",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}/<set>_<variant>_<modality>)",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    data_root = str(root / args.data_root) if not os.path.isabs(args.data_root) else args.data_root
    run_config_suffix = (args.run_config or "").strip() or None
    modality = "vessel" if args.vessel else "junction"

    num_geos = _infer_num_geos(
        data_root=data_root,
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        set_type=args.set_type,
        run_config_suffix=run_config_suffix,
        vessel=bool(args.vessel),
        explicit_num_geos=args.num_geos,
        split_path=None,
    )

    inputs, outputs, feature_names = load_cohort_from_jax(
        data_root=data_root,
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        set_type=args.set_type,
        num_geos=num_geos,
        run_config_suffix=run_config_suffix,
        vessel=bool(args.vessel),
    )

    if args.output_dir is None:
        run_tag = run_config_suffix or "default"
        output_dir = os.path.join(
            str(root),
            DEFAULT_OUTPUT_DIR,
            f"{args.set_name}_{args.geometry_variant}_{modality}_{run_tag}",
        )
    else:
        output_dir = args.output_dir if os.path.isabs(args.output_dir) else str(root / args.output_dir)

    print(
        f"Loaded {inputs.shape[0]} rows, {len(feature_names)} input features "
        f"({modality}, num_geos={num_geos})"
    )
    saved = plot_input_output_correlations(
        inputs,
        outputs,
        feature_names,
        set_name=args.set_name,
        geometry_variant=args.geometry_variant,
        modality=modality,
        output_dir=output_dir,
        run_config_suffix=run_config_suffix,
    )
    for path in saved:
        print(f"Saved {path}")
    print(f"Wrote {len(saved)} figures to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
