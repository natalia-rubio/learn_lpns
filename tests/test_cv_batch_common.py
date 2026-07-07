"""Unit tests for cv_batch_common config resolution."""

from __future__ import annotations

import pytest

from learn_lpns.zerod_calibration.cv_batch_common import (
    by_config_chart_entry_keys,
    default_config_entry_keys,
    parse_config_entry,
    resolve_configs_with_flags,
    unique_run_config_suffixes,
)


def test_default_config_entry_keys_matches_default_configs():
    keys = default_config_entry_keys()
    assert "gen_loss" in keys
    assert "gen_loss:bifurcations" in keys
    assert "quadratic_resistor_penalty_on_gen_loss" not in keys


def test_parse_config_entry_plain_and_override():
    assert parse_config_entry("gen_loss", "bifurcations_EL") == (
        "gen_loss",
        "gen_loss",
        "bifurcations_EL",
    )
    assert parse_config_entry("gen_loss:bifurcations", "bifurcations_EL") == (
        "gen_loss:bifurcations",
        "gen_loss",
        "bifurcations",
    )


def test_resolve_configs_with_flags_unknown_config():
    with pytest.raises(ValueError, match="Unknown config"):
        resolve_configs_with_flags(["not_a_config"], "bifurcations_EL")


def test_by_config_chart_entry_keys_adds_bifurcations_alias():
    configs = resolve_configs_with_flags(["gen_loss"], "bifurcations_EL")
    keys = by_config_chart_entry_keys(configs)
    assert "gen_loss" in keys
    assert "gen_loss:bifurcations" in keys


def test_unique_run_config_suffixes_deduplicates():
    configs = resolve_configs_with_flags(["gen_loss", "gen_loss:bifurcations"], "bifurcations_EL")
    assert unique_run_config_suffixes(configs) == ["gen_loss"]
