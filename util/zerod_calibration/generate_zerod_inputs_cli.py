"""Shared CLI definitions and argv building for generate_zerod_inputs / batch VMR."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from util.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    parse_underscore_tokens,
    resolve_run_config_suffix,
    run_config_suffix_to_flags,
)

SKIP_STEP_TOKENS = frozenset({
    "base_generation",
    "observation",
    "calibration",
    "forward",
    "mse",
    "plots",
})

_SKIP_STEP_TO_ATTR = {
    "base_generation": "skip_base_generation",
    "observation": "skip_observation",
    "calibration": "skip_calibration",
    "forward": "skip_forward",
    "mse": "skip_mse_calculation",
    "plots": "skip_plots",
}


def parse_skip_steps(spec: str) -> frozenset[str]:
    """Parse ``--skip-steps`` token string into canonical step names."""
    return parse_underscore_tokens(spec, SKIP_STEP_TOKENS)


def apply_skip_steps_to_namespace(ns: argparse.Namespace) -> None:
    """Set ``skip_*`` attrs on *ns* from ``ns.skip_steps``."""
    for step in parse_skip_steps(getattr(ns, "skip_steps", "") or ""):
        setattr(ns, _SKIP_STEP_TO_ATTR[step], True)


def resolve_namespace_run_config(ns: argparse.Namespace) -> str:
    """Resolve ``ns.run_config`` and store canonical suffix on ``ns.run_config_suffix``."""
    try:
        suffix = resolve_run_config_suffix(getattr(ns, "run_config", None))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ns.run_config_suffix = suffix
    return suffix


DEFAULT_JUNCTION_TYPES = ["BloodVesselJunction"]
DEFAULT_GEOMETRY_VARIANT = "bifurcations_EL"
BIFURCATION_GEOMETRY_VARIANTS = frozenset({"bifurcations", "bifurcations_EL"})


def add_generate_zerod_inputs_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_set_geo: bool,
) -> None:
    """Register generate_zerod_inputs flags (shared by single-geo and batch entry points)."""
    if require_set_geo:
        parser.add_argument("--set-name", required=True, help="Set name (e.g., VMR_rigid_aorta_adults)")
        parser.add_argument("--geo-name", required=True, help="Geometry name (e.g., 0076_1001)")
    else:
        parser.add_argument(
            "--set-name",
            default="VMR",
            help="Set name (e.g., VMR_rigid_aorta_adults; default: VMR)",
        )

    parser.add_argument(
        "--run-config",
        default=DEFAULT_CLI_RUN_CONFIG,
        metavar="TOKENS",
        help=(
            "Run-config tokens in any order, underscore-separated "
            f"(default: {DEFAULT_CLI_RUN_CONFIG}). "
            "Examples: stenosis_off_symmetric_gen_loss, gen_loss_penalty_off_symmetric. "
            "Full canonical suffixes are also accepted."
        ),
    )
    parser.add_argument(
        "--geometry-variant",
        default=DEFAULT_GEOMETRY_VARIANT,
        choices=sorted(BIFURCATION_GEOMETRY_VARIANTS),
        help=(
            "Bifurcation geometry variant for ML prep and NN inference "
            f"(default: {DEFAULT_GEOMETRY_VARIANT}). "
            "Steps 3.5/3.7 use this variant only; Step 1 still builds both variants "
            "for comparison when not skipped."
        ),
    )
    parser.add_argument(
        "--skip-steps",
        default="",
        metavar="STEPS",
        help=(
            "Pipeline steps to skip, underscore-separated (order-independent). "
            "Tokens: base_generation, observation, calibration, forward, mse, plots. "
            "Example: base_generation_observation_calibration"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-redo",
        action="store_true",
        help="Skip recreating files that already exist (check at each step)",
    )
    parser.add_argument(
        "--NN-only",
        action="store_true",
        help="Only NN inference + forward sim (skips observation and calibration)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        dest="plots_only",
        help="Only Step 6 comparison plots (location + zero-D parameter bars); requires existing zeroD outputs",
    )
    parser.add_argument(
        "--NN-vessel",
        action="store_true",
        dest="NN_vessel",
        help="Also run vessel NN inference and forward sim (*_NN_JunctionAndVessel)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory with rri_{set}_pred_{0,1,2}_model files (CV / custom models)",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=None,
        help="Append _trial_{id} to plot output paths (cross-validation)",
    )


def add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    """Register batch-only flags for batch_generate_zerod_inputs_vmr."""
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="Specific geometry names (default: all from richter-0d)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="Timeout in seconds per geometry (default: 1000)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Stop after N failures (default: continue all)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write batch results JSON to this path",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip geometries that already have required output files",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--only-failed",
        action="store_true",
        help="Only rerun geometries that failed (requires --log-file)",
    )
    filter_group.add_argument(
        "--only-timed-out",
        action="store_true",
        help="Only rerun timed-out geometries (requires --log-file)",
    )
    filter_group.add_argument(
        "--only-successful",
        action="store_true",
        help="Only rerun successful geometries (requires --log-file)",
    )


def _init_skip_flags(ns: argparse.Namespace) -> None:
    for attr in _SKIP_STEP_TO_ATTR.values():
        if not hasattr(ns, attr):
            setattr(ns, attr, False)


def prepare_generate_zerod_namespace(ns: argparse.Namespace) -> str:
    """
    Apply NN-only mode, skip-steps, and run-config resolution.

    Returns the resolved run-config suffix.
    """
    _init_skip_flags(ns)
    ns.junction_types = list(DEFAULT_JUNCTION_TYPES)
    if getattr(ns, "plots_only", False):
        if getattr(ns, "NN_only", False):
            raise SystemExit("--plots-only and --NN-only are mutually exclusive")
        ns.skip_base_generation = True
        ns.skip_observation = True
        ns.skip_calibration = True
        ns.skip_forward = True
        ns.skip_mse_calculation = True
    elif getattr(ns, "NN_only", False):
        ns.skip_observation = True
        ns.skip_calibration = True
    apply_skip_steps_to_namespace(ns)
    suffix = resolve_namespace_run_config(ns)
    flags = run_config_suffix_to_flags(suffix)
    ns.stenosis_off = flags["stenosis_off"]
    ns.penalty_off = flags["penalty_off"]
    ns.symmetric_loss = flags["symmetric_loss"]
    ns.normalize = flags["normalize"]
    return suffix


def namespace_to_generate_zerod_argv(
    ns: argparse.Namespace,
    *,
    set_name: str,
    geo_name: str,
) -> List[str]:
    """Build argv tail for generate_zerod_inputs.py from a prepared namespace."""
    script_path = os.path.join(
        os.path.dirname(__file__), "generate_zerod_inputs.py"
    )
    cmd: List[str] = [
        sys.executable,
        script_path,
        "--set-name",
        set_name,
        "--geo-name",
        geo_name,
    ]

    suffix = getattr(ns, "run_config_suffix", None) or resolve_namespace_run_config(ns)
    cmd.extend(["--run-config", suffix])
    cmd.extend(["--geometry-variant", getattr(ns, "geometry_variant", DEFAULT_GEOMETRY_VARIANT)])

    skip_steps_spec = (getattr(ns, "skip_steps", "") or "").strip()
    if skip_steps_spec:
        cmd.extend(["--skip-steps", skip_steps_spec])

    bool_flags = [
        ("verbose", "--verbose"),
        ("no_redo", "--no-redo"),
        ("NN_only", "--NN-only"),
        ("plots_only", "--plots-only"),
        ("NN_vessel", "--NN-vessel"),
    ]
    for attr, flag in bool_flags:
        if getattr(ns, attr, False):
            cmd.append(flag)

    if getattr(ns, "model_dir", None):
        cmd.extend(["--model-dir", ns.model_dir])
    if getattr(ns, "trial_id", None) is not None:
        cmd.extend(["--trial-id", str(ns.trial_id)])

    return cmd
