"""Shared helpers for multi-config / multi-set cross-validation batch scripts."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from learn_lpns.tools.paths import repo_root

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configs to run: (entry_key, list of run_cross_validation CLI flags)
# Order is execution order; by-config barchart discovers and sorts by value.
DEFAULT_CONFIGS: list[tuple[str, list[str]]] = [
    ("gen_loss", ["--run_config", "gen_loss"]),
    ("quadratic_resistor", ["--run_config", "quadratic_resistor"]),
    ("quadratic_resistor_gen_loss", ["--run_config", "quadratic_resistor_gen_loss"]),
    ("base", ["--run_config", "base"]),
    ("gen_loss:bifurcations", ["--run_config", "gen_loss"]),
    ("quadratic_resistor_gen_loss:bifurcations", ["--run_config", "quadratic_resistor_gen_loss"]),
]

# Valid for `--configs` but not part of the default batch unless listed explicitly.
OPTIONAL_CONFIGS: dict[str, list[str]] = {
    "quadratic_resistor_gen_loss": ["--run_config", "quadratic_resistor_gen_loss"],
    "base": ["--run_config", "base"],
    "gen_loss": ["--run_config", "gen_loss"],
    "quadratic_resistor": ["--run_config", "quadratic_resistor"],
    "quadratic_resistor_penalty_on": ["--run_config", "quadratic_resistor_penalty_on"],
}

ConfigWithFlags = tuple[str, str, str, list[str]]


def config_flag_map() -> dict[str, list[str]]:
    return {**dict(DEFAULT_CONFIGS), **OPTIONAL_CONFIGS}


def default_config_entry_keys() -> list[str]:
    return [entry_key for entry_key, _ in DEFAULT_CONFIGS]


def parse_config_entry(entry: str, default_geometry_variant: str) -> tuple[str, str, str]:
    """
    Parse config entry syntax:
      - "config_suffix" -> (entry_key, run_config_suffix, geometry_variant)
      - "config_suffix:geometry_variant" -> (entry_key, run_config_suffix, geometry_variant)
    """
    raw = (entry or "").strip()
    if not raw:
        raise ValueError("Empty config entry.")
    if ":" in raw:
        run_cfg, geom_var = raw.split(":", 1)
        run_cfg = run_cfg.strip()
        geom_var = geom_var.strip()
        if not run_cfg or not geom_var:
            raise ValueError(f"Invalid config entry {entry!r}. Use 'config' or 'config:geometry_variant'.")
        return raw, run_cfg, geom_var
    return raw, raw, default_geometry_variant


def resolve_configs_with_flags(
    config_list: Sequence[str] | None,
    default_geometry_variant: str,
) -> list[ConfigWithFlags]:
    """Resolve config CLI entries to (entry_key, run_config_suffix, geometry_variant, cv_flags)."""
    entries = list(config_list) if config_list is not None else default_config_entry_keys()
    flag_map = config_flag_map()
    resolved: list[ConfigWithFlags] = []
    for entry in entries:
        entry_key, run_config_suffix, cfg_geometry_variant = parse_config_entry(
            entry, default_geometry_variant
        )
        if run_config_suffix not in flag_map:
            known = sorted(flag_map)
            raise ValueError(f"Unknown config: {run_config_suffix}. Known: {known}")
        resolved.append(
            (entry_key, run_config_suffix, cfg_geometry_variant, flag_map[run_config_suffix])
        )
    return resolved


def by_config_chart_entry_keys(configs_with_flags: Sequence[ConfigWithFlags]) -> list[str]:
    """Entry keys for cv_max_pct_error_by_config_barchart, including gen_loss:bifurcations alias."""
    entry_keys = [entry_key for entry_key, _, _, _ in configs_with_flags]
    if "gen_loss" in entry_keys and "gen_loss:bifurcations" not in entry_keys:
        entry_keys.append("gen_loss:bifurcations")
    return entry_keys


def unique_run_config_suffixes(configs_with_flags: Sequence[ConfigWithFlags]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for _, run_config_suffix, _, _ in configs_with_flags:
        if run_config_suffix not in seen:
            seen.add(run_config_suffix)
            out.append(run_config_suffix)
    return out


def run_by_config_comparison_barchart(
    set_name: str,
    *,
    geometry_variant: str,
    configs_with_flags: Sequence[ConfigWithFlags],
    results_root: str = "results",
    xmax: str = "40",
) -> int:
    """Run cv_max_pct_error_by_config_barchart for one set; return subprocess exit code."""
    cmd = [
        sys.executable,
        "-m",
        "learn_lpns.visualizations.cv_max_pct_error_by_config_barchart",
        set_name,
        "--geometry",
        geometry_variant,
        "--configs",
        *by_config_chart_entry_keys(configs_with_flags),
        "--data_root",
        results_root,
        "--xmax",
        xmax,
    ]
    return subprocess.run(cmd, cwd=repo_root()).returncode


def run_cross_set_summary_barchart(
    set_names: Sequence[str],
    *,
    run_config_suffix: str,
    geometry_variant: str = "bifurcations_EL",
    metrics: Sequence[str] | None = None,
) -> int:
    """Run cv_cross_set_summary_barchart for one run config across sets; return worst exit code."""
    metric_list = list(metrics) if metrics is not None else (
        "pressure_max_rel_error",
        "pressure_mean_rel_error",
    )
    worst = 0
    for metric in metric_list:
        cmd = [
            sys.executable,
            "-m",
            "learn_lpns.visualizations.cv_cross_set_summary_barchart",
            "--set_names",
            *set_names,
            "--geometry_variant",
            geometry_variant,
            "--run_config",
            run_config_suffix,
            "--metric",
            metric,
        ]
        ret = subprocess.run(cmd, cwd=repo_root()).returncode
        worst = max(worst, ret)
    return worst
