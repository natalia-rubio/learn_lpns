#!/usr/bin/env python3
"""
For each VMR cohort geometry: copy or reproduce standard geometric 0D results, and run a
second forward simulation with all stenosis coefficients set to zero.

Outputs live under results/empirical_stenosis_comp/<set_name>/<geo_name>/ by default.
Does not run calibration or neural network inference.

Geometry variant ``original`` uses geometric_input.json / geometric_results.csv (no bifurcation
splitting or entrance-length adjustment). ``bifurcations`` / ``bifurcations_EL`` use the
prefixed files from generate_zerod_inputs.

If the geometric input is missing under ``data/zeroD/<set>/<run_config>/<geo>/`` (default
``run_config`` is ``base``), the script runs ``generate_zerod_inputs`` there (NORMAL_JUNCTION,
skips calibration and NN) to create it before forward runs.

After both forward CSVs exist, the script writes ``empirical_stenosis_mse_comparison.csv``,
``mse_comparison_plots/``, ``mse_debug_plots/``, ``location_comparison_plots/`` (INFLOW
locations), and ``param_comparison/zero_d_parameter_bars.(png|pdf)`` under each geometry folder
in ``--output-root``, unless ``--skip-mse-plots``.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from util.zerod_calibration.zerod_handling import zero_all_stenosis_coefficients

DEFAULT_SET_NAMES = [
    "VMR_rigid_aorta_adults",
    "VMR_abdo",
    "VMR_pulmo_healthy",
]

SET_NAME_ALIASES = {
    "vmr_pulmonary_healthy": "VMR_pulmo_healthy",
    "vmr_pulmo_healthy": "VMR_pulmo_healthy",
}

DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS = 1000


def canonical_set_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return SET_NAME_ALIASES.get(key, name.strip())


def get_geometry_names(repo_root: str, set_name: str) -> List[str]:
    richter_dir = os.path.join(repo_root, "data", "zeroD", set_name, "richter-0d")
    if not os.path.isdir(richter_dir):
        raise FileNotFoundError(f"Richter-0d directory not found: {richter_dir}")
    json_files = sorted(glob.glob(os.path.join(richter_dir, "*.json")))
    out = []
    for path in json_files:
        base = os.path.basename(path).replace(".json", "")
        try:
            with open(path) as f:
                json.load(f)
            out.append(base)
        except Exception as e:
            print(f"  Warning: skipping invalid JSON {base}: {e}")
    return out


def source_zero_d_geo_dir(repo_root: str, set_name: str, run_config: str, geo_name: str) -> str:
    """Absolute path to data/zeroD/<set>/<run_config>/<geo_name>/."""
    return os.path.join(repo_root, "data", "zeroD", set_name, run_config, geo_name)


def _absolute_output_root(output_root: str) -> str:
    if os.path.isabs(output_root):
        return output_root
    return os.path.join(REPO_ROOT, output_root)


def variant_filenames(variant: str) -> Tuple[str, str, str]:
    """geometric input basename, geometric results basename, calibration input basename."""
    if variant == "original":
        return (
            "geometric_input.json",
            "geometric_results.csv",
            "calibration_input.json",
        )
    prefix = f"{variant}_"
    return (
        f"{prefix}geometric_input.json",
        f"{prefix}geometric_results.csv",
        f"{prefix}calibration_input.json",
    )


def variant_zero_stenosis_basenames(variant: str) -> Tuple[str, str]:
    """Basenames for zero-stenosis input JSON and results CSV under the output geo dir."""
    if variant == "original":
        return (
            "geometric_stenosis_zero_input.json",
            "geometric_stenosis_zero_results.csv",
        )
    return (
        f"{variant}_geometric_stenosis_zero_input.json",
        f"{variant}_geometric_stenosis_zero_results.csv",
    )


def variant_std_work_basename(variant: str) -> str:
    """Temp JSON basename when re-running standard geometric forward."""
    if variant == "original":
        return "geometric_forward_work.json"
    return f"{variant}_geometric_forward_work.json"


def resolve_bc_source_path(geometric_src: str, calibration_src: str) -> str:
    """Prefer EL calibration input for inlet refinement; else geometric JSON (VMR Richter BC)."""
    if os.path.isfile(calibration_src):
        return calibration_src
    return geometric_src


def run_generate_zerod_inputs(
    set_name: str,
    geo_name: str,
    run_config: str,
    timeout_seconds: int,
    no_redo: bool,
    verbose: bool,
) -> Tuple[bool, str]:
    """Run generate_zerod_inputs to populate data/zeroD/<set>/<run_config>/<geo>/ (no calib/NN)."""
    script_path = os.path.join(REPO_ROOT, "util", "zerod_calibration", "generate_zerod_inputs.py")
    cmd = [
        sys.executable,
        script_path,
        "--set-name",
        set_name,
        "--geo-name",
        geo_name,
        "--run-config",
        run_config,
        "--junction-types",
        "NORMAL_JUNCTION",
        "--skip-calibration",
        "--skip-mse-calculation",
        "--skip-plots",
    ]
    if no_redo:
        cmd.append("--no-redo")
    if verbose:
        cmd.append("--verbose")
    try:
        r = subprocess.run(cmd, cwd=REPO_ROOT, text=True, timeout=timeout_seconds)
        if r.returncode != 0:
            return False, f"generate_zerod_inputs exited {r.returncode}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"
    except Exception as e:
        return False, str(e)


def process_one_geometry(
    set_name: str,
    geo_name: str,
    src_dir: str,
    variant: str,
    output_root: str,
    no_redo: bool,
    verbose: bool,
) -> Tuple[bool, str]:
    from util.zerod_calibration.forward_simulation import run_forward_simulation
    from util.zerod_calibration.inflow_handling import refine_inlet_bc_for_forward_simulation

    geo_in_name, geo_res_name, calib_name = variant_filenames(variant)
    geometric_src = os.path.join(src_dir, geo_in_name)
    results_src = os.path.join(src_dir, geo_res_name)
    calibration_src = os.path.join(src_dir, calib_name)

    if not os.path.isfile(geometric_src):
        return False, f"missing geometric input: {geometric_src}"

    out_dir = _absolute_output_root(os.path.join(output_root, set_name, geo_name))
    os.makedirs(out_dir, exist_ok=True)

    out_standard_csv = os.path.join(out_dir, geo_res_name)
    zero_in_name, zero_out_name = variant_zero_stenosis_basenames(variant)
    zero_out_csv = os.path.join(out_dir, zero_out_name)
    bc_source = resolve_bc_source_path(geometric_src, calibration_src)

    # --- Standard geometric results ---
    if no_redo and os.path.isfile(out_standard_csv):
        if verbose:
            print(f"    skip standard (exists): {out_standard_csv}")
    elif os.path.isfile(results_src):
        shutil.copy2(results_src, out_standard_csv)
        if verbose:
            print(f"    copied standard results from {results_src}")
    else:
        std_work = os.path.join(out_dir, variant_std_work_basename(variant))
        with open(geometric_src) as f:
            cfg = json.load(f)
        with open(std_work, "w") as f:
            json.dump(cfg, f, indent=4)
        try:
            refine_inlet_bc_for_forward_simulation(std_work, calibration_input_path=bc_source)
            run_forward_simulation(std_work, out_standard_csv)
        except Exception as e:
            if os.path.isfile(std_work):
                try:
                    os.remove(std_work)
                except OSError:
                    pass
            return False, f"standard forward failed: {e}"
        if os.path.isfile(std_work):
            try:
                os.remove(std_work)
            except OSError:
                pass
        if verbose:
            print(f"    wrote standard results {out_standard_csv}")

    # --- Zero-stenosis forward ---
    if no_redo and os.path.isfile(zero_out_csv):
        if verbose:
            print(f"    skip zero-stenosis (exists): {zero_out_csv}")
        return True, ""

    zero_path = os.path.join(out_dir, zero_in_name)
    with open(geometric_src) as f:
        cfg0 = json.load(f)
    cfg_z = copy.deepcopy(cfg0)
    zero_all_stenosis_coefficients(cfg_z)
    with open(zero_path, "w") as f:
        json.dump(cfg_z, f, indent=4)
    try:
        refine_inlet_bc_for_forward_simulation(zero_path, calibration_input_path=bc_source)
        run_forward_simulation(zero_path, zero_out_csv)
    except Exception as e:
        return False, f"zero-stenosis forward failed: {e}"
    if verbose:
        print(f"    wrote zero-stenosis results {zero_out_csv}")
    return True, ""


def run_empirical_mse_and_location_plots(
    set_name: str,
    geo_name: str,
    src_dir: str,
    variant: str,
    output_root: str,
    zoom_start_idx,
    zoom_end_idx,
    no_redo: bool,
    verbose: bool,
) -> Tuple[bool, str]:
    """
    MSE vs 1D/3D observations (from calibration input) for geometric vs stenosis-zero CSVs,
    plus INFLOW location comparison plots (same layout as generate_zerod_inputs step 6),
    and zero-D parameter bar charts (geometric vs stenosis-zero JSONs).
    """
    from util.zerod_calibration.post_processing import (
        calculate_mse_between_3d_and_0d,
        plot_zero_d_parameter_bars,
    )

    geo_in_name, geo_res_name, calib_name = variant_filenames(variant)
    calib_src = os.path.join(src_dir, calib_name)
    geom_input_src = os.path.join(src_dir, geo_in_name)
    out_dir = _absolute_output_root(os.path.join(output_root, set_name, geo_name))
    zero_in_name, zero_out_name = variant_zero_stenosis_basenames(variant)
    out_standard_csv = os.path.join(out_dir, geo_res_name)
    zero_out_csv = os.path.join(out_dir, zero_out_name)
    mse_csv = os.path.join(out_dir, "empirical_stenosis_mse_comparison.csv")

    if not os.path.isfile(calib_src):
        return False, f"calibration input missing for MSE/plots: {calib_src}"
    if not os.path.isfile(out_standard_csv) or not os.path.isfile(zero_out_csv):
        return False, f"missing result CSVs under {out_dir}"

    if no_redo and os.path.isfile(mse_csv):
        if verbose:
            print(f"    skip MSE (exists): {mse_csv}")
    else:
        csv_results_dict = {
            "geometric": out_standard_csv,
            "stenosis_zero": zero_out_csv,
        }
        try:
            calculate_mse_between_3d_and_0d(
                calib_src,
                csv_results_dict,
                geometric_input_path=geom_input_src,
                zoom_start_idx=zoom_start_idx,
                zoom_end_idx=zoom_end_idx,
                output_csv_path=mse_csv,
                verbose=verbose,
                set_name=set_name,
                artifact_base_dir=out_dir,
            )
        except Exception as e:
            return False, f"MSE calculation failed: {e}"

    vis_path = os.path.join(REPO_ROOT, "util", "visualizations")
    if vis_path not in sys.path:
        sys.path.insert(0, vis_path)
    from plot_location_comparison import (
        build_vessel_name_mapping,
        get_all_locations_from_calibration_input,
        get_time_period as get_time_period_func,
        plot_location_comparison,
    )

    time_period = None
    try:
        time_period = get_time_period_func(set_name, geo_name)
    except Exception:
        pass

    try:
        all_locs = get_all_locations_from_calibration_input(str(calib_src))
    except Exception as e:
        return False, f"location list failed: {e}"

    variant_locations = [loc for loc in all_locs if loc.startswith("INFLOW:")]
    if variant_locations:
        plots_dir = os.path.join(out_dir, "location_comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)

        vessel_name_mapping = None
        if variant == "bifurcations_EL":
            bif_in = os.path.join(src_dir, "bifurcations_geometric_input.json")
            el_in = os.path.join(src_dir, "bifurcations_EL_geometric_input.json")
            if os.path.isfile(bif_in) and os.path.isfile(el_in):
                vessel_name_mapping = build_vessel_name_mapping(bif_in, el_in)

        geo_csv_key = "original" if variant == "original" else variant
        variant_geometric_csv_paths = {geo_csv_key: out_standard_csv}
        variant_csv_paths = {"stenosis_zero": zero_out_csv}

        for location in variant_locations:
            safe = location.replace(":", "_").replace("/", "_")
            plot_path = os.path.join(plots_dir, f"{safe}_comparison.png")
            if no_redo and os.path.isfile(plot_path):
                continue
            try:
                plot_ok = plot_location_comparison(
                    str(calib_src),
                    out_standard_csv,
                    variant_csv_paths,
                    location,
                    plot_path,
                    set_name=set_name,
                    geo_name=geo_name,
                    time_period=time_period,
                    geometric_input_path=str(geom_input_src),
                    zoom_start_idx=zoom_start_idx,
                    zoom_end_idx=zoom_end_idx,
                    verbose=False,
                    geometric_csv_paths=variant_geometric_csv_paths,
                    vessel_name_mapping=vessel_name_mapping,
                )
                if not plot_ok:
                    return False, f"location plot failed for {location!r}"
            except Exception as e:
                return False, f"location plot {location!r}: {e}"
    else:
        print("  Warning: no INFLOW: locations; skipping location comparison plots")

    param_dir = os.path.join(out_dir, "param_comparison")
    os.makedirs(param_dir, exist_ok=True)
    param_png = os.path.join(param_dir, "zero_d_parameter_bars.png")
    if no_redo and os.path.isfile(param_png):
        if verbose:
            print(f"    skip param comparison (exists): {param_png}")
    else:
        zero_json = os.path.join(out_dir, zero_in_name)
        if not os.path.isfile(zero_json):
            return False, f"missing stenosis-zero input JSON: {zero_json}"
        if not os.path.isfile(geom_input_src):
            return False, f"missing geometric input JSON: {geom_input_src}"
        modality_jsons = {
            "geometric": str(geom_input_src),
            "stenosis_zero": str(zero_json),
        }
        try:
            out_path = plot_zero_d_parameter_bars(
                modality_jsons,
                output_dir=param_dir,
                output_name="zero_d_parameter_bars.png",
                verbose=verbose,
            )
            if not out_path:
                return False, "plot_zero_d_parameter_bars returned no path"
        except Exception as e:
            return False, f"param comparison plot failed: {e}"

    if verbose:
        print(f"    MSE CSV: {mse_csv}")
        if variant_locations:
            print(f"    location plots: {plots_dir}")
        print(f"    param comparison: {param_dir}")
    return True, ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standard vs zero-stenosis geometric 0D runs into results/empirical_stenosis_comp."
    )
    p.add_argument(
        "sets",
        nargs="*",
        default=[],
        help="Set names (default: VMR_rigid_aorta_adults VMR_abdo VMR_pulmo_healthy). "
        "Aliases: VMR_pulmonary_healthy -> VMR_pulmo_healthy",
    )
    p.add_argument(
        "--sets",
        dest="sets_flag",
        nargs="+",
        default=None,
        help="Alternative way to pass set names (overrides positional sets if both given).",
    )
    p.add_argument(
        "--run-config",
        default="base",
        metavar="NAME",
        help="Subfolder under data/zeroD/<set>/ (default: base). Missing geometric inputs there "
        "trigger generate_zerod_inputs for that path before forward runs.",
    )
    p.add_argument(
        "--geometry-variant",
        choices=("original", "bifurcations_EL", "bifurcations"),
        default="bifurcations_EL",
        help="Geometric input variant: original (no bifurcation split / EL), bifurcations, or "
        "bifurcations_EL (default: bifurcations_EL)",
    )
    p.add_argument(
        "--output-root",
        default="results/empirical_stenosis_comp",
        help="Output directory under repo root (default: results/empirical_stenosis_comp)",
    )
    p.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="If set, only these geometry names (per set)",
    )
    p.add_argument("--no-redo", action="store_true", help="Skip steps when output CSVs already exist")
    p.add_argument(
        "--skip-mse-plots",
        action="store_true",
        help="Skip empirical MSE CSV, MSE plots, INFLOW location plots, and param bar charts",
    )
    p.add_argument(
        "--zoom-start",
        type=int,
        default=None,
        help="Optional zoom window start index for MSE and location plots (default: auto)",
    )
    p.add_argument(
        "--zoom-end",
        type=int,
        default=None,
        help="Optional zoom window end index for MSE and location plots (default: auto)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS,
        help=(
            "Timeout seconds for generate_zerod_inputs when regenerating missing inputs "
            f"(default: {DEFAULT_GENERATE_ZEROD_INPUTS_TIMEOUT_SECONDS})"
        ),
    )
    p.add_argument("--log-file", default=None, help="Write JSON summary to this path (relative to repo root ok)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Stop after this many failures (default: no limit)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sets_flag is not None:
        set_list = args.sets_flag
    elif args.sets:
        set_list = args.sets
    else:
        set_list = DEFAULT_SET_NAMES
    set_list = [canonical_set_name(s) for s in set_list]

    results = {
        "success": [],
        "failed": [],
        "start_time": datetime.now().isoformat(),
        "sets": set_list,
        "run_config": args.run_config,
        "geometry_variant": args.geometry_variant,
        "output_root": args.output_root,
        "skip_mse_plots": args.skip_mse_plots,
    }
    failure_count = 0
    stop_all = False

    for set_name in set_list:
        if stop_all:
            break
        try:
            all_geos = get_geometry_names(REPO_ROOT, set_name)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            results["failed"].append({"set": set_name, "geometry": None, "error": str(e)})
            failure_count += 1
            if args.max_failures and failure_count >= args.max_failures:
                stop_all = True
            continue

        geo_list = all_geos if not args.geometries else [g for g in args.geometries if g in all_geos]
        if args.geometries:
            missing = [g for g in args.geometries if g not in all_geos]
            for g in missing:
                err = f"geometry {g} not found under richter-0d for {set_name}"
                print(f"  Warning: {err}")
                results["failed"].append({"set": set_name, "geometry": g, "error": err})
                failure_count += 1
                if args.max_failures and failure_count >= args.max_failures:
                    stop_all = True
                    break
            if stop_all:
                break

        for geo_name in geo_list:
            if stop_all:
                break
            print(f"\n[{set_name}] {geo_name}")
            src_dir = source_zero_d_geo_dir(REPO_ROOT, set_name, args.run_config, geo_name)
            geo_in_name, _, _ = variant_filenames(args.geometry_variant)
            geometric_src = os.path.join(src_dir, geo_in_name)

            if not os.path.isfile(geometric_src):
                print(
                    f"  missing {geo_in_name}; running generate_zerod_inputs "
                    f"(data/zeroD/{set_name}/{args.run_config}/{geo_name}/)..."
                )
                ok, msg = run_generate_zerod_inputs(
                    set_name,
                    geo_name,
                    args.run_config,
                    args.timeout,
                    args.no_redo,
                    args.verbose,
                )
                if not ok:
                    print(f"  generate_zerod_inputs failed: {msg}")
                    results["failed"].append(
                        {
                            "set": set_name,
                            "geometry": geo_name,
                            "error": f"generate_zerod_inputs: {msg}",
                        }
                    )
                    failure_count += 1
                    if args.max_failures and failure_count >= args.max_failures:
                        stop_all = True
                    continue
                if not os.path.isfile(geometric_src):
                    err = f"still missing after generate: {geometric_src}"
                    print(f"  failed: {err}")
                    results["failed"].append(
                        {"set": set_name, "geometry": geo_name, "error": err}
                    )
                    failure_count += 1
                    if args.max_failures and failure_count >= args.max_failures:
                        stop_all = True
                    continue

            ok, msg = process_one_geometry(
                set_name,
                geo_name,
                src_dir,
                args.geometry_variant,
                args.output_root,
                args.no_redo,
                args.verbose,
            )
            if not ok:
                print(f"  failed: {msg}")
                results["failed"].append({"set": set_name, "geometry": geo_name, "error": msg})
                failure_count += 1
                if args.max_failures and failure_count >= args.max_failures:
                    stop_all = True
                continue

            if not args.skip_mse_plots:
                print("  MSE and location comparison plots...")
                ok_m, msg_m = run_empirical_mse_and_location_plots(
                    set_name,
                    geo_name,
                    src_dir,
                    args.geometry_variant,
                    args.output_root,
                    args.zoom_start,
                    args.zoom_end,
                    args.no_redo,
                    args.verbose,
                )
                if not ok_m:
                    print(f"  failed: {msg_m}")
                    results["failed"].append(
                        {"set": set_name, "geometry": geo_name, "error": msg_m}
                    )
                    failure_count += 1
                    if args.max_failures and failure_count >= args.max_failures:
                        stop_all = True
                    continue

            results["success"].append(
                {
                    "set": set_name,
                    "geometry": geo_name,
                    "run_config": args.run_config,
                }
            )
            print(f"  ok")

    results["end_time"] = datetime.now().isoformat()
    results["success_count"] = len(results["success"])
    results["failed_count"] = len(results["failed"])

    if args.log_file:
        log_path = os.path.join(REPO_ROOT, args.log_file) if not os.path.isabs(args.log_file) else args.log_file
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote log: {log_path}")

    print(
        f"\nDone: {results['success_count']} succeeded, {results['failed_count']} failed"
    )
    return 0 if results["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
