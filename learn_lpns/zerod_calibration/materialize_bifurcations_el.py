"""Build bifurcations_EL 0D JSONs from a base model + centerline (no C++ solver)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from learn_lpns.config import get_pipeline_config
from learn_lpns.zerod_calibration.bifurcation_splitting import (
    adjust_junction_boundaries_by_entrance_length_from_files,
    convert_el_normal_junctions_to_blood_vessel_junction,
    split_junctions_from_files,
)
from learn_lpns.zerod_calibration.centerline_path_extraction import process_geometric_input
from learn_lpns.zerod_calibration.geometric_params import extract_and_add_geometric_params


def _prepare_base_config(base_json_path: str, *, apply_standard_sim_params: bool) -> dict[str, Any]:
    with open(base_json_path) as f:
        cfg = json.load(f)
    if apply_standard_sim_params:
        solver = get_pipeline_config().solver
        sim = cfg.setdefault("simulation_parameters", {})
        sim["output_all_cycles"] = True
        sim["number_of_cardiac_cycles"] = solver.number_of_cardiac_cycles
    return cfg


def _convert_bifurcations_junctions_in_place(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    convert_el_normal_junctions_to_blood_vessel_junction(cfg)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)


def materialize_bifurcations_el_from_base(
    *,
    base_json_path: str,
    centerline_path: str,
    output_bifurcations_el_path: str,
    apply_standard_sim_params: bool = False,
    extract_geometric_params: bool = True,
    convert_bifurcations_junctions: bool = False,
    verbose: bool = False,
) -> str:
    """
    Run centerline wiring, bifurcation splitting, and entrance-length adjustment.

    Mirrors the geometry-variant steps in ``generate_zerod_inputs.py`` (no calibration
    or forward simulation). Used by the NN comparison notebook to derive
    ``bifurcations_EL_geometric_input.json`` from ``standard-0d`` + centerline VTP.
    """
    base_json_path = os.path.abspath(base_json_path)
    centerline_path = os.path.abspath(centerline_path)
    output_bifurcations_el_path = os.path.abspath(output_bifurcations_el_path)

    if not os.path.isfile(base_json_path):
        raise FileNotFoundError(base_json_path)
    if not os.path.isfile(centerline_path):
        raise FileNotFoundError(centerline_path)

    out_dir = os.path.dirname(output_bifurcations_el_path)
    work_dir = os.path.join(out_dir, "_materialize_work")
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    geometric_input_path = os.path.join(work_dir, "geometric_input.json")
    geometric_centerline_path = os.path.join(work_dir, "geometric_centerline_input.json")
    bifurcations_path = os.path.join(work_dir, "bifurcations_geometric_input.json")

    cfg = _prepare_base_config(base_json_path, apply_standard_sim_params=apply_standard_sim_params)
    with open(geometric_input_path, "w") as f:
        json.dump(cfg, f, indent=4)

    geo_label = os.path.basename(out_dir)

    process_geometric_input(centerline_path, geometric_input_path, geometric_centerline_path, verbose=verbose)
    split_junctions_from_files(geometric_centerline_path, centerline_path, bifurcations_path, verbose=verbose)

    if extract_geometric_params:
        extract_and_add_geometric_params(centerline_path, bifurcations_path, verbose=verbose)
    if convert_bifurcations_junctions:
        _convert_bifurcations_junctions_in_place(bifurcations_path)

    if not verbose:
        print(f"  {geo_label}: wrote bifurcations_geometric_input.json")

    os.makedirs(out_dir, exist_ok=True)
    adjust_junction_boundaries_by_entrance_length_from_files(
        bifurcations_path,
        centerline_path,
        output_bifurcations_el_path,
        verbose=verbose,
    )
    if extract_geometric_params:
        extract_and_add_geometric_params(centerline_path, output_bifurcations_el_path, verbose=verbose)

    if not verbose:
        print(f"  {geo_label}: wrote bifurcations_EL_geometric_input.json")

    if not verbose:
        shutil.rmtree(work_dir, ignore_errors=True)
    return output_bifurcations_el_path


def default_centerline_path(data_root: Path | str, geo_name: str) -> Path:
    """VMR cohort: projected 1D solution used as centerline geometry."""
    return Path(data_root) / "oneD" / "VMR" / geo_name / "unsteady_soln.vtp"
