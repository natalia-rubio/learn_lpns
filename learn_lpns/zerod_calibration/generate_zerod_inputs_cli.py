"""Shared CLI definitions and argv building for generate_zerod_inputs / batch VMR."""

from __future__ import annotations

import argparse
import os
import sys

from learn_lpns.config import get_pipeline_config
from learn_lpns.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    parse_underscore_tokens,
    resolve_run_config_suffix,
    run_config_suffix_to_flags,
)

SKIP_STEP_TOKENS = frozenset(
    {
        "base_generation",
        "observation",
        "calibration",
        "forward",
        "nn_inference",
        "mse",
        "plots",
    }
)

_SKIP_STEP_TO_ATTR = {
    "base_generation": "skip_base_generation",
    "observation": "skip_observation",
    "calibration": "skip_calibration",
    "forward": "skip_forward",
    "nn_inference": "skip_nn_inference",
    "mse": "skip_mse_calculation",
    "plots": "skip_plots",
}


def parse_skip_steps(spec: str) -> frozenset[str]:
    """Parse ``--skip_steps`` token string into canonical step names."""
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
DEFAULT_CALIBRATION_BACKEND = "decoupled_ls"
CALIBRATION_BACKEND_CHOICES = ("decoupled_ls", "svzerod")
BIFURCATION_GEOMETRY_VARIANTS = frozenset({"bifurcations", "bifurcations_EL"})


def add_generate_zerod_inputs_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_set_geo: bool,
) -> None:
    """Register generate_zerod_inputs flags (shared by single-geo and batch entry points)."""
    if require_set_geo:
        parser.add_argument("--set_name", required=True, help="Set name (e.g., VMR_rigid_aorta_adults)")
        parser.add_argument("--geo_name", required=True, help="Geometry name (e.g., 0076_1001)")
    else:
        parser.add_argument(
            "--set_name",
            default="VMR",
            help="Set name (e.g., VMR_rigid_aorta_adults; default: VMR)",
        )

    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        metavar="TOKENS",
        help=(
            "Run-config tokens in any order, underscore-separated "
            f"(default: {DEFAULT_CLI_RUN_CONFIG}). "
            "Examples: gen_loss, gen_loss_quadratic_resistor, quadratic_resistor_penalty_on_gen_loss. "
            "Use base for the default physics config with no optional tokens. "
            "penalty_on requires quadratic_resistor."
        ),
    )
    parser.add_argument(
        "--geometry_variant",
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
        "--calibration_backend",
        default=DEFAULT_CALIBRATION_BACKEND,
        choices=CALIBRATION_BACKEND_CHOICES,
        help=(
            "Calibration backend for Step 3 (default: decoupled_ls). "
            "Use svzerod for C++ svzerodcalibrator (requires SVZEROD_INSTALL_DIR)."
        ),
    )
    parser.add_argument(
        "--plot_rsl_fits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save per-element dP vs Q least-squares fit plots under results/RSL_fits/ "
            "(decoupled_ls only; default from config calibration.plot_rsl_fits, usually false)"
        ),
    )
    parser.add_argument(
        "--skip_steps",
        default="",
        metavar="STEPS",
        help=(
            "Pipeline steps to skip, underscore-separated (order-independent). "
            "Tokens: base_generation, observation, calibration, forward, nn_inference, mse, plots. "
            "Example: base_generation_observation_calibration"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no_redo",
        action="store_true",
        help="Skip recreating files that already exist (check at each step)",
    )
    parser.add_argument(
        "--NN_only",
        action="store_true",
        help="Only NN inference + forward sim (skips observation and calibration)",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        dest="plots_only",
        help="Only Step 6 comparison plots (location + zero-D parameter bars); requires existing zeroD outputs",
    )
    parser.add_argument(
        "--Vessel_NN",
        action="store_true",
        dest="Vessel_NN",
        help="Also run vessel NN inference and forward sim (*_NN_JunctionAndVessel)",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Directory with rri_{set}_pred_{0,1,2}_model files (CV / custom models)",
    )
    parser.add_argument(
        "--multi_output_rri",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use single multi-output R/S/L checkpoint at inference "
            "(default: training.multi_output_rri from config, usually false)."
        ),
    )
    parser.add_argument(
        "--clip_predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Clip R/S/L NN predictions to train-set min/max "
            "(default: training.clip_predictions from config, usually false)."
        ),
    )
    parser.add_argument(
        "--trial_id",
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
        help="Specific geometry names (default: all from standard-0d)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="Timeout in seconds per geometry (default: 1000)",
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=None,
        help="Stop after N failures (default: continue all)",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Write batch results JSON to this path",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip geometries that already have required output files",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--only_failed",
        action="store_true",
        help="Only rerun geometries that failed (requires --log_file)",
    )
    filter_group.add_argument(
        "--only_timed_out",
        action="store_true",
        help="Only rerun timed-out geometries (requires --log_file)",
    )
    filter_group.add_argument(
        "--only_successful",
        action="store_true",
        help="Only rerun successful geometries (requires --log_file)",
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
            raise SystemExit("--plots_only and --NN_only are mutually exclusive")
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
    ns.quadratic_resistor = flags["quadratic_resistor"]
    ns.penalty_on = flags["penalty_on"]
    ns.asymmetric_loss = flags["asymmetric_loss"]
    if getattr(ns, "multi_output_rri", None) is None:
        ns.multi_output_rri = get_pipeline_config(set_name=getattr(ns, "set_name", None)).training.multi_output_rri
    if getattr(ns, "clip_predictions", None) is None:
        ns.clip_predictions = get_pipeline_config(set_name=getattr(ns, "set_name", None)).training.clip_predictions
    if getattr(ns, "plot_rsl_fits", None) is None:
        ns.plot_rsl_fits = get_pipeline_config().calibration.plot_rsl_fits
    return suffix


def namespace_to_generate_zerod_argv(
    ns: argparse.Namespace,
    *,
    set_name: str,
    geo_name: str,
) -> list[str]:
    """Build argv tail for generate_zerod_inputs.py from a prepared namespace."""
    script_path = os.path.join(os.path.dirname(__file__), "generate_zerod_inputs.py")
    cmd: list[str] = [
        sys.executable,
        script_path,
        "--set_name",
        set_name,
        "--geo_name",
        geo_name,
    ]

    suffix = getattr(ns, "run_config_suffix", None) or resolve_namespace_run_config(ns)
    cmd.extend(["--run_config", suffix])
    cmd.extend(["--geometry_variant", getattr(ns, "geometry_variant", DEFAULT_GEOMETRY_VARIANT)])
    cmd.extend(["--calibration_backend", getattr(ns, "calibration_backend", DEFAULT_CALIBRATION_BACKEND)])
    if getattr(ns, "plot_rsl_fits", False):
        cmd.append("--plot_rsl_fits")

    skip_steps_spec = (getattr(ns, "skip_steps", "") or "").strip()
    if skip_steps_spec:
        cmd.extend(["--skip_steps", skip_steps_spec])

    bool_flags = [
        ("verbose", "--verbose"),
        ("no_redo", "--no_redo"),
        ("NN_only", "--NN_only"),
        ("plots_only", "--plots_only"),
        ("Vessel_NN", "--Vessel_NN"),
    ]
    for attr, flag in bool_flags:
        if getattr(ns, attr, False):
            cmd.append(flag)

    if getattr(ns, "multi_output_rri", False):
        cmd.append("--multi_output_rri")
    else:
        cmd.append("--no-multi_output_rri")

    if getattr(ns, "clip_predictions", False):
        cmd.append("--clip_predictions")
    else:
        cmd.append("--no-clip_predictions")

    if getattr(ns, "model_dir", None):
        cmd.extend(["--model_dir", ns.model_dir])
    if getattr(ns, "trial_id", None) is not None:
        cmd.extend(["--trial_id", str(ns.trial_id)])

    return cmd
