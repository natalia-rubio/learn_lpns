#!/usr/bin/env python3
"""Diagnose how often decoupled LS R>=0 / L>=0 lower bounds are active.

Re-fits existing calibration-input JSONs in memory (does not write calibrated outputs).

Usage:
  python -m learn_lpns.zerod_calibration.diagnose_bound_activation \\
    --set_name VMR_pulmo --run_config quadratic_resistor_gen_loss
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from learn_lpns.config import get_pipeline_config
from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.decoupled_ls_calibration import collect_bound_activation_rows
from learn_lpns.zerod_calibration.run_config_canonical import resolve_run_config_suffix
from learn_lpns.zerod_calibration.tools.file_io import get_vmr_geometries, standard_0d_dir

DEFAULT_RUN_CONFIG = "quadratic_resistor_gen_loss"
DEFAULT_GEOMETRY_VARIANT = "bifurcations_EL"
JUNCTION_TYPE = "BloodVesselJunction"

CSV_FIELDS = [
    "set_name",
    "geo_name",
    "element_name",
    "element_kind",
    "fit_stenosis",
    "nonneg_r_enabled",
    "nonneg_l_enabled",
    "R",
    "S",
    "L",
    "r_lower_active",
    "l_lower_active",
    "rel_err",
]


def calibration_input_path(
    data_root: Path,
    set_name: str,
    run_config: str,
    geo_name: str,
    geometry_variant: str,
) -> Path:
    prefix = "" if geometry_variant == "original" else f"{geometry_variant}_"
    return (
        data_root
        / "zeroD"
        / set_name
        / run_config
        / geo_name
        / f"{prefix}calibration_input_{JUNCTION_TYPE}.json"
    )


def bound_activation_output_dir(
    results_root: Path,
    set_name: str,
    run_config: str,
) -> Path:
    return results_root / "bound_activation" / set_name / run_config


def _rate(n_active: int, n_total: int) -> str:
    if n_total == 0:
        return "n/a (0 fits)"
    return f"{100.0 * n_active / n_total:.2f}% ({n_active}/{n_total})"


def _subset_rates(rows: list[dict[str, Any]], key: str, value: Any) -> tuple[str, str]:
    subset = [r for r in rows if r.get(key) == value]
    n = len(subset)
    r_active = sum(1 for r in subset if r["r_lower_active"])
    l_active = sum(1 for r in subset if r["l_lower_active"])
    return _rate(r_active, n), _rate(l_active, n)


def format_summary(
    *,
    set_name: str,
    run_config: str,
    geometry_variant: str,
    rows: list[dict[str, Any]],
    n_geos_fitted: int,
    n_geos_skipped: int,
    skipped_geos: list[str],
) -> str:
    n = len(rows)
    r_active = sum(1 for r in rows if r["r_lower_active"])
    l_active = sum(1 for r in rows if r["l_lower_active"])
    n_r_enabled = sum(1 for r in rows if r["nonneg_r_enabled"])
    n_l_enabled = sum(1 for r in rows if r["nonneg_l_enabled"])

    lines = [
        f"set_name: {set_name}",
        f"run_config: {run_config}",
        f"geometry_variant: {geometry_variant}",
        f"geometries fitted: {n_geos_fitted}",
        f"geometries skipped (missing calibration input): {n_geos_skipped}",
        f"fitted elements: {n}",
        f"fits with R bound enabled: {n_r_enabled}",
        f"fits with L bound enabled: {n_l_enabled}",
        f"overall R lower-bound active: {_rate(r_active, n)}",
        f"overall L lower-bound active: {_rate(l_active, n)}",
    ]

    for kind in ("vessel", "junction_outlet"):
        r_rate, l_rate = _subset_rates(rows, "element_kind", kind)
        lines.append(f"  by element_kind={kind}: R={r_rate}; L={l_rate}")

    for fit_stenosis in (True, False):
        r_rate, l_rate = _subset_rates(rows, "fit_stenosis", fit_stenosis)
        label = "RSL" if fit_stenosis else "RL"
        lines.append(f"  by fit_stenosis={fit_stenosis} ({label}): R={r_rate}; L={l_rate}")

    if skipped_geos:
        preview = ", ".join(skipped_geos[:10])
        extra = f" ... (+{len(skipped_geos) - 10} more)" if len(skipped_geos) > 10 else ""
        lines.append(f"skipped geos: {preview}{extra}")

    return "\n".join(lines) + "\n"


def diagnose_set(
    *,
    set_name: str,
    run_config: str,
    geometry_variant: str,
    data_root: Path,
    results_root: Path,
    geometries: list[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    if geometries is None:
        std_dir = standard_0d_dir(str(data_root), set_name)
        if not os.path.isdir(std_dir):
            raise FileNotFoundError(f"standard-0d directory not found for {set_name}: {std_dir}")
        geo_names = get_vmr_geometries(std_dir, data_root=str(data_root), set_name=set_name)
    else:
        geo_names = list(geometries)

    pipe = get_pipeline_config(set_name=set_name)
    cal_cfg = pipe.calibration
    dp_cfg = pipe.data_processing.stenosis_generation_limit

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    fitted_geos = 0

    for geo_name in geo_names:
        path = calibration_input_path(
            data_root, set_name, run_config, geo_name, geometry_variant
        )
        if not path.is_file():
            skipped.append(geo_name)
            continue

        with open(path, encoding="utf-8") as f:
            config = json.load(f)

        element_rows = collect_bound_activation_rows(
            config,
            l2_r=cal_cfg.decoupled_l2_r,
            l2_stenosis=cal_cfg.decoupled_l2_stenosis,
            l2_l=cal_cfg.decoupled_l2_l,
            nonneg_r=cal_cfg.decoupled_nonneg_r,
            nonneg_l=cal_cfg.decoupled_nonneg_l,
            stenosis_generation_limit_enabled=dp_cfg.enabled,
            junction_stenosis_generation_max=dp_cfg.junction_limit(),
            vessel_stenosis_generation_max=dp_cfg.vessel_limit(),
        )
        for row in element_rows:
            out = {"set_name": set_name, "geo_name": geo_name, **row}
            rows.append(out)
        fitted_geos += 1

    summary = format_summary(
        set_name=set_name,
        run_config=run_config,
        geometry_variant=geometry_variant,
        rows=rows,
        n_geos_fitted=fitted_geos,
        n_geos_skipped=len(skipped),
        skipped_geos=skipped,
    )

    out_dir = bound_activation_output_dir(results_root, set_name, run_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_element.csv"
    summary_path = out_dir / "summary.txt"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})

    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run decoupled LS fits on existing calibration inputs and report "
            "how often R>=0 and L>=0 lower bounds are active (lsq_linear active_mask)."
        )
    )
    parser.add_argument(
        "--set_name",
        action="append",
        required=True,
        dest="set_names",
        help="Cohort name under data/zeroD/ (repeatable)",
    )
    parser.add_argument(
        "--run_config",
        default=DEFAULT_RUN_CONFIG,
        help=f"Run-config tokens (default: {DEFAULT_RUN_CONFIG})",
    )
    parser.add_argument(
        "--geometry_variant",
        default=DEFAULT_GEOMETRY_VARIANT,
        help=f"Geometry variant prefix for calibration input (default: {DEFAULT_GEOMETRY_VARIANT})",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Data root containing zeroD/ (default: data)",
    )
    parser.add_argument(
        "--results_root",
        default=None,
        help="Results root for bound_activation/ output (default: <repo>/results)",
    )
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="Optional geometry name subset (default: all in standard-0d)",
    )
    args = parser.parse_args(argv)

    run_config = resolve_run_config_suffix(args.run_config)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root() / data_root
    results_root = Path(args.results_root) if args.results_root else repo_root() / "results"
    if not results_root.is_absolute():
        results_root = repo_root() / results_root

    for set_name in args.set_names:
        diagnose_set(
            set_name=set_name,
            run_config=run_config,
            geometry_variant=args.geometry_variant,
            data_root=data_root,
            results_root=results_root,
            geometries=args.geometries,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
