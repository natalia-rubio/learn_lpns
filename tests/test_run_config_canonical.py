"""Tests for run-config token parsing and canonical suffix composition."""

import pytest

from util.zerod_calibration.run_config_canonical import (
    RUN_CONFIG_TOKENS,
    canonical_run_config_for_data_paths,
    compose_run_config_suffix,
    discover_run_config_suffixes,
    parse_underscore_tokens,
    resolve_run_config_suffix,
    run_config_suffix_has_asymmetric_loss,
    run_config_suffix_has_generation_weighted_loss,
    run_config_suffix_has_penalty_on,
    run_config_suffix_has_quadratic_resistor,
    run_config_suffix_to_flags,
)


class TestParseUnderscoreTokens:
    def test_empty_spec_returns_empty_set(self):
        assert parse_underscore_tokens("", RUN_CONFIG_TOKENS) == frozenset()
        assert parse_underscore_tokens("   ", RUN_CONFIG_TOKENS) == frozenset()

    def test_single_token(self):
        assert parse_underscore_tokens("gen_loss", RUN_CONFIG_TOKENS) == frozenset({"gen_loss"})

    def test_multi_word_token_not_split(self):
        tokens = parse_underscore_tokens("quadratic_resistor_gen_loss", RUN_CONFIG_TOKENS)
        assert tokens == frozenset({"quadratic_resistor", "gen_loss"})

    def test_order_independent(self):
        a = parse_underscore_tokens("gen_loss_quadratic_resistor", RUN_CONFIG_TOKENS)
        b = parse_underscore_tokens("quadratic_resistor_gen_loss", RUN_CONFIG_TOKENS)
        assert a == b == frozenset({"gen_loss", "quadratic_resistor"})

    def test_alias_generation_weighted_loss(self):
        tokens = parse_underscore_tokens(
            "generation_weighted_loss",
            RUN_CONFIG_TOKENS,
            {"generation_weighted_loss": "gen_loss"},
        )
        assert tokens == frozenset({"gen_loss"})

    def test_unrecognized_token_raises(self):
        with pytest.raises(ValueError, match="Unrecognized token"):
            parse_underscore_tokens("gen_loss_unknown", RUN_CONFIG_TOKENS)


class TestComposeRunConfigSuffix:
    def test_no_tokens_returns_base(self):
        assert compose_run_config_suffix(frozenset()) == "base"

    def test_canonical_order(self):
        tokens = frozenset({"gen_loss", "asymmetric_loss", "quadratic_resistor"})
        assert compose_run_config_suffix(tokens) == "quadratic_resistor_asymmetric_loss_gen_loss"

    def test_penalty_on_requires_quadratic_resistor(self):
        with pytest.raises(ValueError, match="penalty_on requires quadratic_resistor"):
            compose_run_config_suffix(frozenset({"penalty_on"}))


class TestResolveRunConfigSuffix:
    def test_default_is_gen_loss(self):
        assert resolve_run_config_suffix(None) == "gen_loss"
        assert resolve_run_config_suffix("") == "gen_loss"

    def test_base_explicit(self):
        assert resolve_run_config_suffix("base") == "base"

    def test_reorders_user_tokens(self):
        resolved = resolve_run_config_suffix("gen_loss_quadratic_resistor_asymmetric_loss")
        assert resolved == "quadratic_resistor_asymmetric_loss_gen_loss"


class TestRunConfigSuffixHelpers:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("gen_loss", True),
            ("quadratic_resistor_gen_loss", True),
            ("base", False),
            (None, False),
        ],
    )
    def test_generation_weighted_loss(self, suffix, expected):
        assert run_config_suffix_has_generation_weighted_loss(suffix) is expected

    def test_canonical_for_data_paths_strips_gen_loss(self):
        assert canonical_run_config_for_data_paths("quadratic_resistor_gen_loss") == "quadratic_resistor"
        assert canonical_run_config_for_data_paths("gen_loss") == "gen_loss"

    def test_suffix_to_flags(self):
        flags = run_config_suffix_to_flags("quadratic_resistor_penalty_on_gen_loss")
        assert flags == {
            "quadratic_resistor": True,
            "asymmetric_loss": False,
            "penalty_on": True,
            "generation_weighted_loss": True,
        }

    def test_suffix_flag_detectors(self):
        suffix = "quadratic_resistor_asymmetric_loss_gen_loss"
        assert run_config_suffix_has_quadratic_resistor(suffix)
        assert run_config_suffix_has_asymmetric_loss(suffix)
        assert not run_config_suffix_has_penalty_on(suffix)


class TestDiscoverRunConfigSuffixes:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert discover_run_config_suffixes(str(tmp_path / "missing")) == []

    def test_lists_subdirectories(self, tmp_path):
        (tmp_path / "gen_loss").mkdir()
        (tmp_path / "base").mkdir()
        (tmp_path / "readme.txt").write_text("not a dir")
        assert discover_run_config_suffixes(str(tmp_path)) == ["base", "gen_loss"]
