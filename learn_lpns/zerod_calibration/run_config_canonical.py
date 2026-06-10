"""Run-config helpers: path suffixes, token parsing, composition, and flag derivation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

_GEN_LOSS_SUFFIX = "_gen_loss"  # on-disk path fragment; unchanged for existing data trees
_ASYMMETRIC_LOSS_SUFFIX = "_asymmetric_loss"
_QUADRATIC_RESISTOR_SUFFIX = "_quadratic_resistor"
_PENALTY_ON_SUFFIX = "_penalty_on"

# Default ``--run_config`` path suffix for repo CLIs when omitted (under ``set_name``).
DEFAULT_CLI_RUN_CONFIG = "gen_loss"

RUN_CONFIG_TOKENS: frozenset[str] = frozenset(
    {
        "penalty_on",
        "gen_loss",
        "asymmetric_loss",
        "quadratic_resistor",
    }
)

RUN_CONFIG_ALIASES: dict[str, str] = {
    # User token ``generation_weighted_loss`` → path token ``gen_loss`` (suffix ``_gen_loss``).
    "generation_weighted_loss": "gen_loss",
}


def parse_underscore_tokens(
    spec: str,
    vocabulary: frozenset[str],
    aliases: dict[str, str] | None = None,
) -> frozenset[str]:
    """
    Parse an underscore-separated token string (order-independent).

    Uses longest-token-first greedy matching so multi-word tokens like
    ``quadratic_resistor`` parse correctly.
    """
    aliases = aliases or {}
    remaining = (spec or "").strip()
    if not remaining:
        return frozenset()

    ordered_vocab = sorted(vocabulary | frozenset(aliases.keys()), key=len, reverse=True)
    found: set[str] = set()

    while remaining:
        matched = False
        for token in ordered_vocab:
            if remaining == token or remaining.startswith(token + "_"):
                canonical = aliases.get(token, token)
                if canonical not in vocabulary:
                    raise ValueError(f"Alias {token!r} maps to unknown token {canonical!r}")
                found.add(canonical)
                remaining = remaining[len(token) :].lstrip("_")
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Unrecognized token in {spec!r} (remainder: {remaining!r}). "
                f"Expected tokens: {', '.join(sorted(vocabulary))}"
            )
    return frozenset(found)


def compose_run_config_suffix(tokens: frozenset[str]) -> str:
    """Build canonical path suffix from a set of run-config tokens."""
    if "penalty_on" in tokens and "quadratic_resistor" not in tokens:
        raise ValueError("penalty_on requires quadratic_resistor (L2 calibration penalties apply only in RRI mode).")

    has_quadratic_resistor = "quadratic_resistor" in tokens
    has_penalty_on = "penalty_on" in tokens
    has_asymmetric_loss = "asymmetric_loss" in tokens
    has_gen_loss = "gen_loss" in tokens

    if not has_quadratic_resistor and not has_penalty_on and not has_asymmetric_loss and not has_gen_loss:
        return "base"

    parts: list[str] = []
    if has_quadratic_resistor:
        parts.append("quadratic_resistor")
    if has_penalty_on:
        parts.append("penalty_on")
    if has_asymmetric_loss:
        parts.append("asymmetric_loss")
    if has_gen_loss:
        parts.append("gen_loss")
    return "_".join(parts)


def iter_composed_run_config_suffixes() -> Iterator[str]:
    """Yield every canonical suffix composable from :data:`RUN_CONFIG_TOKENS` (including ``base``)."""
    from itertools import combinations

    yield "base"
    token_list = sorted(RUN_CONFIG_TOKENS)
    for r in range(1, len(token_list) + 1):
        for combo in combinations(token_list, r):
            try:
                yield compose_run_config_suffix(frozenset(combo))
            except ValueError:
                continue


def discover_run_config_suffixes(parent_dir: str) -> list[str]:
    """List run-config subfolder names under ``parent_dir`` (e.g. a CV set directory)."""
    if not os.path.isdir(parent_dir):
        return []
    return sorted(name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name)))


def resolve_run_config_suffix(spec: str | None) -> str:
    """
    Resolve user ``--run_config`` input to a canonical path suffix.

    Accepts ``base``, a canonical suffix (e.g. ``gen_loss``), or an unordered
    token string (e.g. ``gen_loss_quadratic_resistor_asymmetric_loss``).
    """
    raw = (spec or DEFAULT_CLI_RUN_CONFIG).strip()
    if not raw:
        raw = DEFAULT_CLI_RUN_CONFIG
    if raw == "base":
        return "base"

    tokens = parse_underscore_tokens(raw, RUN_CONFIG_TOKENS, RUN_CONFIG_ALIASES)
    return compose_run_config_suffix(tokens)


def canonical_run_config_for_data_paths(run_config_suffix: str | None) -> str | None:
    """
    Strip a trailing ``_gen_loss`` for comparisons that ignore gen-loss variant.

    The full suffix (including ``_gen_loss``) is still used on disk everywhere.
    """
    if run_config_suffix is None:
        return None
    s = str(run_config_suffix).strip()
    if not s:
        return None
    if s.endswith(_GEN_LOSS_SUFFIX) and len(s) > len(_GEN_LOSS_SUFFIX):
        return s[: -len(_GEN_LOSS_SUFFIX)]
    return s


def run_config_suffix_has_generation_weighted_loss(run_config_suffix: str | None) -> bool:
    """True if the run-config path suffix denotes generation-weighted training."""
    if not run_config_suffix:
        return False
    s = str(run_config_suffix).strip()
    return s == "gen_loss" or s.endswith(_GEN_LOSS_SUFFIX)


def run_config_suffix_has_asymmetric_loss(run_config_suffix: str | None) -> bool:
    """True if the run-config suffix includes the ``asymmetric_loss`` path fragment."""
    if not run_config_suffix:
        return False
    s = str(run_config_suffix).strip()
    return s == "asymmetric_loss" or _ASYMMETRIC_LOSS_SUFFIX in s or s.startswith("asymmetric_loss_")


def run_config_suffix_has_quadratic_resistor(run_config_suffix: str | None) -> bool:
    """True if the run-config suffix includes the ``quadratic_resistor`` path fragment."""
    if not run_config_suffix:
        return False
    s = str(run_config_suffix).strip()
    return s == "quadratic_resistor" or _QUADRATIC_RESISTOR_SUFFIX in s or s.startswith("quadratic_resistor_")


def run_config_suffix_has_penalty_on(run_config_suffix: str | None) -> bool:
    """True if the run-config suffix includes the ``penalty_on`` path fragment."""
    if not run_config_suffix:
        return False
    s = str(run_config_suffix).strip()
    return s == "penalty_on" or s == "quadratic_resistor_penalty_on" or _PENALTY_ON_SUFFIX in s


def run_config_includes_gen_loss(run_config_suffix: str | None) -> bool:
    """Deprecated alias for :func:`run_config_suffix_has_generation_weighted_loss`."""
    return run_config_suffix_has_generation_weighted_loss(run_config_suffix)


def run_config_suffix_to_flags(run_config_suffix: Any) -> dict[str, bool]:
    """Derive boolean flags from a canonical run_config suffix string."""
    s_full = (run_config_suffix or "base").strip()
    generation_weighted_loss = run_config_suffix_has_generation_weighted_loss(s_full)
    return {
        "quadratic_resistor": run_config_suffix_has_quadratic_resistor(s_full),
        "asymmetric_loss": run_config_suffix_has_asymmetric_loss(s_full),
        "penalty_on": run_config_suffix_has_penalty_on(s_full),
        "generation_weighted_loss": generation_weighted_loss,
    }


__all__ = [
    "DEFAULT_CLI_RUN_CONFIG",
    "RUN_CONFIG_ALIASES",
    "RUN_CONFIG_TOKENS",
    "canonical_run_config_for_data_paths",
    "compose_run_config_suffix",
    "discover_run_config_suffixes",
    "iter_composed_run_config_suffixes",
    "parse_underscore_tokens",
    "resolve_run_config_suffix",
    "run_config_includes_gen_loss",
    "run_config_suffix_has_asymmetric_loss",
    "run_config_suffix_has_generation_weighted_loss",
    "run_config_suffix_has_penalty_on",
    "run_config_suffix_has_quadratic_resistor",
    "run_config_suffix_to_flags",
]
