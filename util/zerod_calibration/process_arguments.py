"""
CLI and subprocess flag handling for batch_generate_zerod_inputs_vmr.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, List, Sequence

from util.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    full_run_config_path_suffix,
    run_config_suffix_to_flags,
)

# Timeout in seconds for each generate_zerod_inputs.py run. Increase for slow/large geometries.
DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS = 1000

# Mutually exclusive with --run-config (VMR batch): use one style only.
INDIVIDUAL_RUN_CONFIG_KEYS = (
    "normalize",
    "stenosis_off",
    "penalty_off",
    "symmetric_loss",
    "clip_predictions",
    "gen_loss",
)


def _load_previous_log(log_file_path: str) -> dict[str, Any] | None:
    """Load a previous batch log file to extract failed/timed-out geometries."""
    if not os.path.exists(log_file_path):
        return None
    try:
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
        failed_geos = [
            item["geometry"]
            for item in log_data.get("failed", [])
            if not item.get("timed_out", False)
        ]
        timed_out_geos = log_data.get("timed_out", [])
        return {
            "failed": failed_geos,
            "timed_out": timed_out_geos,
            "all_failed": [item["geometry"] for item in log_data.get("failed", [])],
        }
    except Exception as e:
        print(f"  Warning: Could not load log file {log_file_path}: {e}")
        return None


def build_vmr_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch run generate_zerod_inputs.py for all VMR geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow for all geometries
  python3 batch_generate_zerod_inputs_vmr.py

  # Skip calibration and forward simulation (faster)
  python3 batch_generate_zerod_inputs_vmr.py --skip-calibration --skip-forward

  # Only process specific geometries
  python3 batch_generate_zerod_inputs_vmr.py --geometries 0063_1001 0155_0001

  # Rerun only failed geometries from a previous run
  python3 batch_generate_zerod_inputs_vmr.py --only-failed --log-file results/vmr_batch_log.json

  # Rerun only timed-out geometries from a previous run
  python3 batch_generate_zerod_inputs_vmr.py --only-timed-out --log-file results/vmr_batch_log.json

  # Rerun only successful geometries from a previous run
  python3 batch_generate_zerod_inputs_vmr.py --only-successful --log-file results/vmr_batch_log.json

  # Verbose output
  python3 batch_generate_zerod_inputs_vmr.py --verbose

  # Also run vessel NN inference (junction + vessel params)
  python3 batch_generate_zerod_inputs_vmr.py --NN-vessel
  python3 batch_generate_zerod_inputs_vmr.py --normalize

  # Either --run-config alone (implies all booleans) or individual flags alone:
  python3 batch_generate_zerod_inputs_vmr.py --run-config stenosis_off_symmetric_gen_loss
  python3 batch_generate_zerod_inputs_vmr.py --stenosis-off --symmetric-loss --gen-loss
        """,
    )
    parser.add_argument("--set-name", default="VMR", help="Set name (e.g., set_1)")
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="Specific geometry names to process (default: all)",
    )
    parser.add_argument(
        "--junction-types",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default="BloodVesselJunction",
        help="Comma-separated junction types to process (default: BloodVesselJunction only)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output for each geometry"
    )
    parser.add_argument(
        "--skip-base-generation",
        action="store_true",
        help="Skip generating base geometric input",
    )
    parser.add_argument(
        "--skip-observation",
        action="store_true",
        help="Skip observation extraction step",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true", help="Skip calibration step"
    )
    parser.add_argument(
        "--skip-forward", action="store_true", help="Skip forward simulation step"
    )
    parser.add_argument(
        "--skip-mse-calculation",
        action="store_true",
        help="Skip MSE calculation step",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating comparison plots",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip geometries that already have necessary output files",
    )
    parser.add_argument(
        "--no-redo",
        action="store_true",
        help="Skip recreating files if they already exist (check at each step)",
    )
    parser.add_argument(
        "--NN-only",
        action="store_true",
        help="Only run NN inference and forward simulation on NN inputs (skip calibration)",
    )
    parser.add_argument(
        "--NN-vessel",
        action="store_true",
        dest="NN_vessel",
        help="Also run vessel NN inference and forward sim (write *_NN_JunctionAndVessel.json/results)",
    )
    parser.add_argument(
        "--stenosis-off",
        action="store_true",
        dest="stenosis_off",
        help="Turn off stenosis: calibrate_stenosis_coefficient=False, set all stenosis to 0, do not use NN to predict stenosis",
    )
    parser.add_argument(
        "--penalty-off",
        action="store_true",
        dest="penalty_off",
        help="Zero L2 penalties on R and stenosis when stenosis is included (incompatible with --stenosis-off)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use normalized NN models and unnormalize predictions (pass --normalize to generate_zerod_inputs)",
    )
    parser.add_argument(
        "--symmetric-loss",
        action="store_true",
        dest="symmetric_loss",
        help="Symmetric loss run-config: overestimate weight 1.0 for all models (for path naming)",
    )
    parser.add_argument(
        "--clip-predictions",
        action="store_true",
        dest="clip_predictions",
        help="Clip R/S/L to training set min/max (run-config)",
    )
    parser.add_argument(
        "--gen-loss",
        action="store_true",
        dest="gen_loss",
        help="Path variant ending in _gen_loss (must match --run-config when using --run-config)",
    )
    parser.add_argument(
        "--strict-forward",
        action="store_true",
        dest="strict_forward",
        help="Pass to generate_zerod_inputs: abort on svzerodsolver failure (no all-zeros CSV).",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        metavar="SUFFIX",
        help="Path suffix for zeroD/ml_inputs. Mutually exclusive with individual run-config "
        "flags (--normalize, --stenosis-off, --gen-loss, ...); when set, those are inferred. "
        f"If omitted and no individual flags are set, defaults to {DEFAULT_CLI_RUN_CONFIG}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for each geometry (default: {DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file to write results (default: no log file)",
    )

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--only-failed",
        action="store_true",
        help="Only run geometries that failed in a previous run (requires --log-file)",
    )
    filter_group.add_argument(
        "--only-timed-out",
        action="store_true",
        help="Only run geometries that timed out in a previous run (requires --log-file)",
    )
    filter_group.add_argument(
        "--only-successful",
        action="store_true",
        help="Only run geometries that succeeded in a previous run (requires --log-file)",
    )

    return parser


def apply_default_run_config_if_unspecified(args: argparse.Namespace) -> None:
    """Set ``--run-config`` when omitted and no individual run-config flags are set."""
    rc = (getattr(args, "run_config", None) or "").strip()
    if rc:
        return
    if any(getattr(args, k, False) for k in INDIVIDUAL_RUN_CONFIG_KEYS):
        return
    args.run_config = DEFAULT_CLI_RUN_CONFIG


def resolve_vmr_run_config_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Either ``--run-config`` or individual run-config flags, not both. Expands --run-config into flags."""
    rc = (getattr(args, "run_config", None) or "").strip()
    individual_used = any(getattr(args, k, False) for k in INDIVIDUAL_RUN_CONFIG_KEYS)
    if rc and individual_used:
        parser.error(
            "Provide either --run-config SUFFIX or individual flags "
            "(--normalize, --stenosis-off, --penalty-off, --symmetric-loss, "
            "--clip-predictions, --gen-loss), not both."
        )
    if rc:
        fl = run_config_suffix_to_flags(rc)
        if fl["stenosis_off"] and fl["penalty_off"]:
            parser.error(
                "Run config implies both stenosis_off and penalty_off; use a valid suffix."
            )
        args.normalize = fl["normalize"]
        args.stenosis_off = fl["stenosis_off"]
        args.penalty_off = fl["penalty_off"]
        args.symmetric_loss = fl["symmetric_loss"]
        args.clip_predictions = fl["clip_predictions"]
        args.gen_loss = fl["gen_loss"]


def zero_d_run_config_subdir(args: argparse.Namespace) -> str:
    """Subfolder under data/zeroD/<set>/ for outputs (matches generate_zerod_inputs)."""
    rc = (getattr(args, "run_config", None) or "").strip()
    if rc:
        return rc
    from util.zerod_calibration.generate_zerod_inputs import get_run_config_suffix

    physics = get_run_config_suffix(
        normalize=getattr(args, "normalize", False),
        stenosis_off=getattr(args, "stenosis_off", False),
        symmetric_loss=getattr(args, "symmetric_loss", False),
        clip_predictions=getattr(args, "clip_predictions", False),
        penalty_off=getattr(args, "penalty_off", False),
    )
    return full_run_config_path_suffix(physics, getattr(args, "gen_loss", False))


def validate_vmr_batch_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if getattr(args, "stenosis_off", False) and getattr(args, "penalty_off", False):
        parser.error("Cannot use both --stenosis-off and --penalty-off.")


def namespace_to_subprocess_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "junction_types": args.junction_types,
        "verbose": args.verbose,
        "skip_base_generation": args.skip_base_generation,
        "skip_observation": args.skip_observation,
        "skip_calibration": args.skip_calibration,
        "skip_forward": args.skip_forward,
        "skip_mse_calculation": args.skip_mse_calculation,
        "skip_plots": args.skip_plots,
        "NN_only": args.NN_only,
        "NN_vessel": args.NN_vessel,
        "no_redo": args.no_redo,
        "normalize": getattr(args, "normalize", False),
        "stenosis_off": getattr(args, "stenosis_off", False),
        "penalty_off": getattr(args, "penalty_off", False),
        "symmetric_loss": getattr(args, "symmetric_loss", False),
        "clip_predictions": getattr(args, "clip_predictions", False),
        "gen_loss": getattr(args, "gen_loss", False),
        "run_config": (getattr(args, "run_config", None) or "").strip() or None,
        "strict_forward": getattr(args, "strict_forward", False),
    }


def append_generate_zerod_flags_to_cmd(cmd: List[str], args_dict: dict[str, Any]) -> None:
    """Mutate cmd to append flags passed through to generate_zerod_inputs.py."""
    if args_dict.get("junction_types"):
        cmd.extend(["--junction-types", ",".join(args_dict["junction_types"])])
    if args_dict.get("verbose", False):
        cmd.append("--verbose")
    if args_dict.get("skip_base_generation", False):
        cmd.append("--skip-base-generation")
    if args_dict.get("skip_observation", False):
        cmd.append("--skip-observation")
    if args_dict.get("skip_calibration", False):
        cmd.append("--skip-calibration")
    if args_dict.get("skip_forward", False):
        cmd.append("--skip-forward")
    if args_dict.get("skip_mse_calculation", False):
        cmd.append("--skip-mse-calculation")
    if args_dict.get("skip_plots", False):
        cmd.append("--skip-plots")
    if args_dict.get("NN_only", False):
        cmd.append("--NN-only")
    if args_dict.get("NN_vessel", False):
        cmd.append("--NN-vessel")
    if args_dict.get("no_redo", False):
        cmd.append("--no-redo")
    if args_dict.get("normalize", False):
        cmd.append("--normalize")
    if args_dict.get("stenosis_off", False):
        cmd.append("--stenosis-off")
    if args_dict.get("penalty_off", False):
        cmd.append("--penalty-off")
    if args_dict.get("symmetric_loss", False):
        cmd.append("--symmetric-loss")
    if args_dict.get("clip_predictions", False):
        cmd.append("--clip-predictions")
    if args_dict.get("gen_loss", False):
        cmd.append("--gen-loss")
    if args_dict.get("strict_forward", False):
        cmd.append("--strict-forward")
    rc = args_dict.get("run_config")
    if rc:
        cmd.extend(["--run-config", rc])


def discover_geometry_names(
    args: argparse.Namespace,
    set_name: str,
    repo_root: str,
    get_vmr_geometries: Callable[..., Sequence[str]],
) -> List[str]:
    if args.geometries:
        geo_names = list(args.geometries)
        print(f"Processing {len(geo_names)} specified geometries")
        return geo_names
    print("Discovering VMR geometries...")
    richter_dir = os.path.join(repo_root, "data", "zeroD", set_name, "richter-0d")
    geo_names = list(get_vmr_geometries(richter_dir=richter_dir))
    print(f"Found {len(geo_names)} VMR geometries")
    return geo_names


def apply_log_file_geometry_filter(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    repo_root: str,
    geo_names: List[str],
) -> List[str]:
    """Restrict geo_names using --only-failed / --only-timed-out / --only-successful."""
    if not (args.only_failed or args.only_timed_out or args.only_successful):
        return geo_names

    if not args.log_file:
        parser.error(
            "--only-failed, --only-timed-out, and --only-successful require --log-file to be specified"
        )

    log_path = os.path.join(repo_root, args.log_file)
    previous_log = _load_previous_log(log_path)
    if previous_log is None:
        parser.error(f"Could not load log file: {log_path}")

    if args.only_failed:
        target_geos = previous_log["failed"]
        print(f"\nFiltering to {len(target_geos)} failed geometries from log file")
    elif args.only_timed_out:
        target_geos = previous_log["timed_out"]
        print(f"\nFiltering to {len(target_geos)} timed-out geometries from log file")
    else:
        assert args.only_successful
        try:
            with open(log_path, "r") as f:
                log_data = json.load(f)
            target_geos = log_data.get("success", [])
            print(f"\nFiltering to {len(target_geos)} successful geometries from log file")
        except Exception as e:
            parser.error(f"Could not load successful geometries from log file: {e}")

    filtered = [geo for geo in geo_names if geo in target_geos]
    if len(filtered) == 0:
        print("No matching geometries found in log file. Exiting.")
        return filtered
    print(f"Will process {len(filtered)} geometries")
    return filtered


def print_batch_configuration(
    args: argparse.Namespace, set_name: str, geo_names: Sequence[str]
) -> None:
    print("\n" + "=" * 70)
    print("Batch Configuration")
    print("=" * 70)
    print(f"  Set name: {set_name}")
    print(f"  Number of geometries: {len(geo_names)}")
    print(f"  Junction types: {', '.join(args.junction_types)}")
    print(f"  Skip base generation: {args.skip_base_generation}")
    print(f"  Skip observation: {args.skip_observation}")
    print(f"  Skip calibration: {args.skip_calibration}")
    print(f"  Skip forward: {args.skip_forward}")
    print(f"  Skip MSE calculation: {args.skip_mse_calculation}")
    print(f"  Skip plots: {args.skip_plots}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  No-redo mode: {args.no_redo}")
    print(f"  NN-only mode: {args.NN_only}")
    print(f"  NN-vessel mode: {args.NN_vessel}")
    print(f"  Normalize: {getattr(args, 'normalize', False)}")
    print(f"  Stenosis-off: {getattr(args, 'stenosis_off', False)}")
    print(f"  Penalty-off: {getattr(args, 'penalty_off', False)}")
    print(f"  Symmetric-loss: {getattr(args, 'symmetric_loss', False)}")
    print(f"  Clip-predictions: {getattr(args, 'clip_predictions', False)}")
    print(f"  Gen-loss: {getattr(args, 'gen_loss', False)}")
    print(f"  Strict-forward: {getattr(args, 'strict_forward', False)}")
    rc_line = (getattr(args, "run_config", None) or "").strip()
    print(
        f"  Run-config (path suffix): {rc_line or '(derived from flags above)'}"
    )
    print(f"  Timeout per geometry: {args.timeout}s ({args.timeout / 60:.1f} minutes)")
    if args.only_failed:
        print("  Filter mode: Only failed geometries")
    elif args.only_timed_out:
        print("  Filter mode: Only timed-out geometries")
    elif args.only_successful:
        print("  Filter mode: Only successful geometries")
    else:
        print("  Filter mode: All geometries (default)")
    print("=" * 70 + "\n")
