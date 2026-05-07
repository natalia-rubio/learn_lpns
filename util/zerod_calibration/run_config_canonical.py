"""Run-config helpers: path suffixes, ``_gen_loss`` detection, and flag parsing."""

from __future__ import annotations

from typing import Any, Dict, Optional

_GEN_LOSS_SUFFIX = "_gen_loss"

# Default ``--run-config`` path suffix for repo CLIs when omitted (under ``set_name``).
DEFAULT_CLI_RUN_CONFIG = "stenosis_off_symmetric_gen_loss"


def full_run_config_path_suffix(physics_suffix: str, gen_loss: bool) -> str:
    """
    Directory suffix under ``data/zeroD/<set>/`` / ``ml_inputs``: physics-only name from
    ``get_run_config_suffix`` plus optional ``_gen_loss``, matching ``--run-config`` on disk.
    """
    if not gen_loss:
        return physics_suffix
    if not physics_suffix or physics_suffix == "base":
        return f"base{_GEN_LOSS_SUFFIX}"
    return f"{physics_suffix}{_GEN_LOSS_SUFFIX}"


def canonical_run_config_for_data_paths(run_config_suffix: Optional[str]) -> Optional[str]:
    """
    Strip a trailing ``_gen_loss`` for *flag-consistency checks only*.

    ``--run-config`` is still the full string everywhere on disk (jax_arrays, splits, models,
    zeroD, etc.); that full string is what selects those artifacts and enables generation-
    weighted loss when it ends with ``_gen_loss``.

    Boolean CLI flags (:func:`get_run_config_suffix`) do not encode ``_gen_loss``. To verify
    that ``--run-config some_name_gen_loss`` matches ``--penalty-off`` / ``--stenosis-off`` / …,
    this returns ``some_name`` so it can be compared to the flag-derived suffix (e.g.
    ``symmetric_gen_loss`` → ``symmetric``). It does not
    remove ``_gen_loss`` from paths or from training.
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
    """True if the run-config suffix ends with ``_gen_loss`` (gen-weighted loss + separate artifact tree)."""
    if not run_config_suffix:
        return False
    return str(run_config_suffix).strip().endswith(_GEN_LOSS_SUFFIX)


def run_config_suffix_to_flags(run_config_suffix: Any) -> Dict[str, bool]:
    """
    Derive boolean flags from a run_config suffix string (inverse of get_run_config_suffix).

    The ``gen_loss`` flag comes from the full suffix (``…_gen_loss``). Substrings like
    ``stenosis_off`` are matched after stripping ``_gen_loss`` so physics tokens are read
    correctly.
    """
    s_full = (run_config_suffix or "base").strip()
    gen_loss = run_config_includes_gen_loss(s_full)
    s = canonical_run_config_for_data_paths(s_full) or s_full
    return {
        "normalize": "normalized" in s,
        "stenosis_off": "stenosis_off" in s,
        "symmetric_loss": "symmetric" in s,
        "clip_predictions": "clip" in s,
        "penalty_off": "penalty_off" in s,
        "gen_loss": gen_loss,
    }
