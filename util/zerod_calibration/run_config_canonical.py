"""Run-config helpers: path suffixes, token parsing, composition, and flag derivation."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

_GEN_LOSS_SUFFIX = "_gen_loss"

# Default ``--run-config`` path suffix for repo CLIs when omitted (under ``set_name``).
DEFAULT_CLI_RUN_CONFIG = "stenosis_off_symmetric_gen_loss"

ALLOWED_RUN_CONFIGS: FrozenSet[str] = frozenset({
    "base",
    "penalty_off",
    "penalty_off_gen_loss",
    "symmetric_penalty_off_gen_loss",
    "symmetric_penalty_off",
    "stenosis_off",
    "stenosis_off_symmetric",
    "stenosis_off_symmetric_gen_loss",
    "normalized",
    "normalized_penalty_off",
    "normalized_stenosis_off",
    "symmetric",
    "symmetric_gen_loss",
})

RUN_CONFIG_TOKENS: FrozenSet[str] = frozenset({
    "normalized",
    "stenosis_off",
    "symmetric",
    "penalty_off",
    "gen_loss",
})

RUN_CONFIG_ALIASES: Dict[str, str] = {
    "symmetric_loss": "symmetric",
}

CANONICAL_PHYSICS_ORDER: Tuple[str, ...] = (
    "normalized",
    "stenosis_off",
    "symmetric",
    "penalty_off",
)


def parse_underscore_tokens(
    spec: str,
    vocabulary: FrozenSet[str],
    aliases: Optional[Dict[str, str]] = None,
) -> FrozenSet[str]:
    """
    Parse an underscore-separated token string (order-independent).

    Uses longest-token-first greedy matching so multi-word tokens like
    ``base_generation`` or ``stenosis_off`` parse correctly.
    """
    aliases = aliases or {}
    remaining = (spec or "").strip()
    if not remaining:
        return frozenset()

    # Longest tokens first (e.g. stenosis_off before stenosis).
    ordered_vocab = sorted(vocabulary | frozenset(aliases.keys()), key=len, reverse=True)
    found: set[str] = set()

    while remaining:
        matched = False
        for token in ordered_vocab:
            if remaining == token or remaining.startswith(token + "_"):
                canonical = aliases.get(token, token)
                if canonical not in vocabulary:
                    raise ValueError(
                        f"Alias {token!r} maps to unknown token {canonical!r}"
                    )
                found.add(canonical)
                remaining = remaining[len(token):].lstrip("_")
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Unrecognized token in {spec!r} (remainder: {remaining!r}). "
                f"Expected tokens: {', '.join(sorted(vocabulary))}"
            )
    return frozenset(found)


def compose_run_config_suffix(tokens: FrozenSet[str]) -> str:
    """Build canonical path suffix from a set of run-config tokens."""
    if "stenosis_off" in tokens and "penalty_off" in tokens:
        raise ValueError("stenosis_off and penalty_off are incompatible")

    physics = [t for t in CANONICAL_PHYSICS_ORDER if t in tokens]
    has_gen_loss = "gen_loss" in tokens

    if not physics and not has_gen_loss:
        return "base"
    body = "_".join(physics)
    if has_gen_loss:
        return f"{body}_gen_loss" if body else "gen_loss"
    return body


def resolve_run_config_suffix(spec: Optional[str]) -> str:
    """
    Resolve user ``--run-config`` input to a canonical allowed suffix.

    Accepts full canonical suffixes (passthrough) or unordered token strings
    (e.g. ``gen_loss_penalty_off_symmetric``).
    """
    raw = (spec or DEFAULT_CLI_RUN_CONFIG).strip()
    if not raw:
        raw = DEFAULT_CLI_RUN_CONFIG

    if raw in ALLOWED_RUN_CONFIGS:
        return raw

    tokens = parse_underscore_tokens(raw, RUN_CONFIG_TOKENS, RUN_CONFIG_ALIASES)
    suffix = compose_run_config_suffix(tokens)
    if suffix not in ALLOWED_RUN_CONFIGS:
        raise ValueError(
            f"Invalid run-config combination {suffix!r} from tokens {raw!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_RUN_CONFIGS))}."
        )
    return suffix


def canonical_run_config_for_data_paths(run_config_suffix: Optional[str]) -> Optional[str]:
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


def run_config_includes_gen_loss(run_config_suffix: Optional[str]) -> bool:
    """True if the run-config suffix ends with ``_gen_loss``."""
    if not run_config_suffix:
        return False
    return str(run_config_suffix).strip().endswith(_GEN_LOSS_SUFFIX)


def run_config_suffix_to_flags(run_config_suffix: Any) -> Dict[str, bool]:
    """Derive boolean flags from a canonical run_config suffix string."""
    s_full = (run_config_suffix or "base").strip()
    gen_loss = run_config_includes_gen_loss(s_full)
    s = canonical_run_config_for_data_paths(s_full) or s_full
    return {
        "normalize": "normalized" in s,
        "stenosis_off": "stenosis_off" in s,
        "symmetric_loss": "symmetric" in s,
        "penalty_off": "penalty_off" in s,
        "gen_loss": gen_loss,
    }
