#!/usr/bin/env python3
"""
Report total NN dataset row counts per set_name from saved JAX pickles.

Reads the concatenated junction and vessel ``data_dict`` files under::

    data/jax_arrays/<set_name>/<run_config>/<geometry_variant>/<set_type>/

Default: ``gen_loss / bifurcations_EL / all``.

- **Non-connector vessels**: ``input.shape[0]`` in ``jax_arrays_vessel_num_geos_*.pkl``
  (one row per non-connector vessel; same construction as
  ``build_data_dict_from_vessel_csvs``).
- **Bifurcations**: ``input.shape[0] // 2`` in ``jax_arrays_num_geos_*.pkl``, because each
  bifurcation contributes two training rows (swapped outlet order); see
  ``build_data_dict_from_csvs``.

**Environment:** pickles contain JAX arrays. Run with the **svVasc_clean** conda env::

    conda activate svVasc_clean
    python -m util.data_processing.report_jax_test_split_counts

Or::

    conda run -n svVasc_clean python -m util.data_processing.report_jax_test_split_counts

Usage::
  python -m util.data_processing.report_jax_test_split_counts
  python -m util.data_processing.report_jax_test_split_counts VMR_abdo VMR_pulmo_healthy
  python -m util.data_processing.report_jax_test_split_counts --data_root /path/to/data
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.tools.basic import load_dict
from util.zerod_calibration.run_config_canonical import DEFAULT_CLI_RUN_CONFIG

_JAX_ENV_HINT = (
    "JAX is required to load these pickles. Activate the project conda env, e.g.:\n"
    "  conda activate svVasc_clean\n"
    "or:\n"
    "  conda run -n svVasc_clean python -m util.data_processing.report_jax_test_split_counts"
)


def _ensure_jax() -> None:
    try:
        import jax  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"{_JAX_ENV_HINT}\n\n(ImportError: {e})") from e


def _num_geos_from_basename(path: str) -> int:
    m = re.search(r"num_geos_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _select_pkl(directory: str, glob_pattern: str) -> str:
    paths = sorted(glob.glob(os.path.join(directory, glob_pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {glob_pattern!r} under {directory}"
        )

    return max(paths, key=_num_geos_from_basename)


def _input_nrows(data_dict: dict) -> int:
    inp = data_dict["input"]
    return int(getattr(inp, "shape", (0,))[0])


def report_one_set(
    data_root: str,
    set_name: str,
    run_config: str,
    geometry_variant: str,
    set_type: str,
) -> Tuple[int, int, int, int, str, str]:
    """
    Returns:
        n_vessel_rows, n_junction_rows, n_bifurcations, num_geos, junction_pkl, vessel_pkl
    """
    base = os.path.join(
        data_root,
        "jax_arrays",
        set_name,
        run_config,
        geometry_variant,
        set_type,
    )
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Missing directory: {base}")

    j_path = _select_pkl(base, "jax_arrays_num_geos_*.pkl")
    v_path = _select_pkl(base, "jax_arrays_vessel_num_geos_*.pkl")

    j_dict = load_dict(j_path)
    v_dict = load_dict(v_path)

    n_j = _input_nrows(j_dict)
    n_v = _input_nrows(v_dict)
    if n_j % 2 != 0:
        print(
            f"WARNING {set_name}: junction_rows={n_j} is odd "
            f"(typically even: 2 rows per bifurcation swap). "
            f"bifurcations reported as floor(rows/2).",
            file=sys.stderr,
        )
    n_bif = n_j // 2
    n_geo = _num_geos_from_basename(j_path)
    return n_v, n_j, n_bif, n_geo, j_path, v_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report vessel and bifurcation counts from jax_arrays test pickles.",
        epilog=_JAX_ENV_HINT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "set_names",
        nargs="*",
        default=None,
        help="Set names (e.g. VMR_abdo). Default: all subdirs of data/jax_arrays with the expected path.",
    )
    parser.add_argument(
        "--data_root",
        default=os.path.join(REPO_ROOT, "data"),
        help="Repo data root (default: <repo>/data)",
    )
    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        help=f"Run-config directory under jax_arrays/<set>/ (default: {DEFAULT_CLI_RUN_CONFIG})",
    )
    parser.add_argument(
        "--geometry_variant",
        default="bifurcations_EL",
        help="Geometry variant folder (default: bifurcations_EL)",
    )
    parser.add_argument(
        "--set_type",
        default="all",
        help="Cohort folder tier under jax_arrays (default: all)",
    )
    args = parser.parse_args()

    _ensure_jax()

    jax_root = os.path.join(args.data_root, "jax_arrays")
    if args.set_names:
        sets: List[str] = list(args.set_names)
    else:
        if not os.path.isdir(jax_root):
            raise SystemExit(f"Not a directory: {jax_root}")
        sets = sorted(
            name
            for name in os.listdir(jax_root)
            if os.path.isdir(
                os.path.join(
                    jax_root,
                    name,
                    args.run_config,
                    args.geometry_variant,
                    args.set_type,
                )
            )
        )

    if not sets:
        raise SystemExit(
            f"No sets found under {jax_root} with "
            f"{args.run_config}/{args.geometry_variant}/{args.set_type}"
        )

    rows_out: List[Tuple[str, int, int, int, int]] = []
    for s in sets:
        n_v, n_j, n_bif, n_geo, jp, vp = report_one_set(
            args.data_root,
            s,
            args.run_config,
            args.geometry_variant,
            args.set_type,
        )
        rows_out.append((s, n_geo, n_bif, n_v, n_j))
        print(
            f"{s}:  non_connector_vessels={n_v}  bifurcations={n_bif}  "
            f"(junction_rows={n_j}, num_geos={n_geo})"
        )
        print(f"    junction: {jp}")
        print(f"    vessel:   {vp}")

    print()
    print(
        f"{'set_name':<32} {'num_geos':>10} {'bifurcations':>14} {'vessels':>10} {'junction_rows':>14}"
    )
    print("-" * 86)
    for s, n_geo, n_bif, n_v, n_j in rows_out:
        print(f"{s:<32} {n_geo:>10} {n_bif:>14} {n_v:>10} {n_j:>14}")


if __name__ == "__main__":
    main()
