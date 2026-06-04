#!/usr/bin/env python3
"""
learn_lpns entry point: project 3D VTU(s) onto centerlines for TST-cohort (Python only, no svSlicer).

Reuses project_results_python_fallback from batch_centerline_proj_svslicer.py.

Expected TST-cohort layout:

  data/threeD/TST-cohort/<case>/<stem>-result-<timestep>.vtu
  data/oneD/TST-cohort/<case>/<stem>-centerlines.vtp

Writes (fixed filename per case):

  data/oneD/TST-cohort/<case>/unsteady_soln.vtp

Point arrays on the result use the same timestep padding as
``batch_centerline_proj_svslicer.project_results_python_fallback``
(e.g. timestep 1000 -> pressure_01000, velocity_01000).

The centerline VTP must include: CenterlineSectionNormal, GlobalNodeId, BranchId.

Usage (from repo root; use the **svVasc_clean** conda env for VTK, meshio, numpy, tqdm):

  conda activate svVasc_clean
  cd /path/to/learn_lpns
  PYTHONPATH=. python3 util/cluster_scripts/project_tst_cohort_centerline_python.py --dry-run
  PYTHONPATH=. python3 util/cluster_scripts/project_tst_cohort_centerline_python.py --case TST-5

One-liner without activating (sets ``PYTHONPATH`` in the subprocess):

  conda run -n svVasc_clean --no-capture-output bash -lc \\
    'cd /path/to/learn_lpns && export PYTHONPATH=. && python3 util/cluster_scripts/project_tst_cohort_centerline_python.py --case TST-5'
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLUSTER_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# batch_centerline_proj_svslicer imports ``util.vtk_functions`` from the nested
# ``cluster_scripts/util/`` package — that ``util`` must appear before repo ``util``.
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if _CLUSTER_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _CLUSTER_SCRIPTS_DIR)

import importlib.util  # noqa: E402

_bcc_path = os.path.join(_CLUSTER_SCRIPTS_DIR, "batch_centerline_proj_svslicer.py")
_bcc_spec = importlib.util.spec_from_file_location(
    "_batch_centerline_proj_svslicer", _bcc_path
)
assert _bcc_spec and _bcc_spec.loader
_bcc = importlib.util.module_from_spec(_bcc_spec)
_bcc_spec.loader.exec_module(_bcc)
pad_timestep = _bcc.pad_timestep
project_results_python_fallback = _bcc.project_results_python_fallback

OUTPUT_FILENAME = "unsteady_soln.vtp"


def _default_three_d_root(repo: str) -> str:
    return os.path.join(repo, "data", "threeD", "TST-cohort")


def _default_one_d_root(repo: str) -> str:
    return os.path.join(repo, "data", "oneD", "TST-cohort")


def discover_jobs(
    three_d_root: str,
    one_d_root: str,
    timestep: str,
    case_filter: str | None,
) -> list[dict]:
    """Each job: case, stem, vtu, vtp, output (unsteady_soln.vtp under 1D case dir)."""
    jobs: list[dict] = []
    if not os.path.isdir(three_d_root):
        return jobs

    suffix = f"-result-{timestep}.vtu"
    for name in sorted(os.listdir(three_d_root)):
        if case_filter is not None and name != case_filter:
            continue
        case_dir = os.path.join(three_d_root, name)
        if not os.path.isdir(case_dir):
            continue

        vtus = sorted(
            p
            for p in glob.glob(os.path.join(case_dir, f"*{suffix}"))
            if os.path.isfile(p)
        )
        if not vtus:
            continue
        if len(vtus) > 1:
            print(
                f"  Warning: multiple VTUs matching *{suffix} in {case_dir}; using {vtus[0]}"
            )
        vtu = vtus[0]
        base = os.path.basename(vtu)
        stem = base[: -len(suffix)]
        vtp = os.path.join(one_d_root, name, f"{stem}-centerlines.vtp")
        out = os.path.join(one_d_root, name, OUTPUT_FILENAME)
        jobs.append(
            {
                "case": name,
                "stem": stem,
                "vtu": vtu,
                "vtp": vtp,
                "output": out,
            }
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project TST-cohort 3D VTU(s) onto centerlines (Python only); "
        f"writes {OUTPUT_FILENAME} under each case in the 1D root.",
        epilog=(
            "Run inside conda env **svVasc_clean** (VTK, meshio, numpy, tqdm). "
            "Example: `conda activate svVasc_clean` then "
            "`PYTHONPATH=. python3 util/cluster_scripts/project_tst_cohort_centerline_python.py` "
            "from the repo root."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--three-d-root",
        default=None,
        help="Case folders with VTUs (default: <repo>/data/threeD/TST-cohort).",
    )
    parser.add_argument(
        "--one-d-root",
        default=None,
        help="Case folders with centerlines / output (default: <repo>/data/oneD/TST-cohort).",
    )
    parser.add_argument(
        "--timestep",
        default="1000",
        help="Suffix in *-result-<timestep>.vtu (default 1000 -> arrays pressure_01000 etc.).",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Process only this case directory name (e.g. TST-5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: env PYTHON_NUM_WORKERS or cpu_count-1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List jobs only; do not run projection.",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo_root)
    three_d = os.path.abspath(args.three_d_root or _default_three_d_root(repo))
    one_d = os.path.abspath(args.one_d_root or _default_one_d_root(repo))

    ts = str(args.timestep).strip()
    jobs = discover_jobs(three_d, one_d, ts, args.case)

    if not jobs:
        print(f"No jobs found under:\n  3D: {three_d}\n  1D: {one_d}\n")
        print(
            f"Expected VTU per case: *-result-{ts}.vtu\n"
            f"Expected centerline: <1D>/<case>/<stem>-centerlines.vtp\n"
            f"  where <stem> is the VTU basename without '-result-{ts}.vtu'.\n"
            f"Output: <1D>/<case>/{OUTPUT_FILENAME}"
        )
        raise SystemExit(1)

    print(f"Repo: {repo}")
    print(f"3D root: {three_d}")
    print(f"1D root: {one_d}")
    print(f"Timestep: {ts} (array suffix _{pad_timestep(ts)})\n")

    ok_n = 0
    for job in jobs:
        print(f"Case {job['case']}  stem={job['stem']}")
        print(f"  VTU: {job['vtu']}")
        print(f"  VTP: {job['vtp']}")
        print(f"  Out: {job['output']}")
        if not os.path.isfile(job["vtu"]):
            print("  Skip: VTU missing\n")
            continue
        if not os.path.isfile(job["vtp"]):
            print("  Skip: centerline VTP missing\n")
            continue
        if args.dry_run:
            ok_n += 1
            print("  (dry-run)\n")
            continue

        os.makedirs(os.path.dirname(job["output"]), exist_ok=True)
        time_files = [(ts, job["vtu"])]
        success = project_results_python_fallback(
            geo_dir=repo,
            sim_dir=repo,
            centerline_path=job["vtp"],
            num_procs="unused",
            output_path=job["output"],
            time_files=time_files,
            num_workers=args.workers,
        )
        if success:
            ok_n += 1
            print("  OK\n")
        else:
            print("  FAILED\n")

    print(f"Done: {ok_n}/{len(jobs)} succeeded.")
    if ok_n < len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
