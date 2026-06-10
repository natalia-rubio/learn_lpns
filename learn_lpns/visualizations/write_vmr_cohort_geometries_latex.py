#!/usr/bin/env python3
"""
Build a LaTeX table of VMR cohort geometry names for selected sets.

Geometries are discovered under ``data/zeroD/<set_name>/<run_config>/`` (subdirectories).
Legacy folder names under ``data/zeroD/`` are mapped to the ``Name`` column in
``data/dataset-svprojects.csv`` (via ``Legacy Name``); the table lists only those names,
one cohort per column (Aortic, Aortofemoral, Pulmonary).

Display labels for cohorts match ``SET_DISPLAY_NAME`` in
``learn_lpns.visualizations.cv_cross_set_summary_barchart.py`` (internal set_name -> short label).

Requires in the LaTeX preamble::
    \\usepackage{booktabs}

Usage::
  python -m learn_lpns.visualization.write_vmr_cohort_geometries_latex
  python -m learn_lpns.visualization.write_vmr_cohort_geometries_latex -o results/vmr_cohort_geometries.tex
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence

from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.tools.file_io import STANDARD_0D_SUBDIR

# Internal folder names used under data/zeroD/ and results/
DEFAULT_SET_ORDER: Sequence[str] = (
    "VMR_rigid_aorta_adults_all",
    "VMR_abdo",
    "VMR_pulmo_healthy",
)

# Display names (aligned with cv_cross_set_summary_barchart.SET_DISPLAY_NAME); strip() applied when writing.
SET_DISPLAY_NAME: dict[str, str] = {
    "VMR_rigid_aorta_adults_all": "Aortic",
    "VMR_abdo": "Aortofemoral ",
    "VMR_pulmo_healthy": "Pulmonary",
}

# Alternate spellings -> canonical set_name under data/zeroD/
SET_NAME_ALIASES: dict[str, str] = {
    "VMR_pulmonary_healthy": "VMR_pulmo_healthy",
}

# Prefer run-config subfolders that contain per-geometry directories (not only standard-0d JSON).
RUN_CONFIG_DIR_PREFERENCE: Sequence[str] = (
    "gen_loss",
    "quadratic_resistor_penalty_on_gen_loss",
    "quadratic_resistor_gen_loss",
    "base",
)


def _latex_escape_cell(text: str) -> str:
    """Escape underscores for LaTeX table cells (IDs and VMR names)."""
    return text.replace("_", r"\_")


def _resolve_set_name(name: str) -> str:
    return SET_NAME_ALIASES.get(name, name)


def _load_legacy_to_name_from_csv(csv_path: str) -> dict[str, str]:
    """Map Legacy Name -> Name (first CSV column). Raises on duplicate legacy keys with conflicting names."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "Legacy Name" not in fields or "Name" not in fields:
            raise KeyError(f"CSV must have 'Name' and 'Legacy Name'; got fields: {fields!r}")
        out: dict[str, str] = {}
        for row in reader:
            leg = (row.get("Legacy Name") or "").strip()
            name = (row.get("Name") or "").strip()
            if not leg:
                continue
            if not name:
                raise ValueError(f"CSV row with Legacy Name {leg!r} has empty Name")
            if leg in out and out[leg] != name:
                raise ValueError(f"Duplicate Legacy Name {leg!r} with different Name: {out[leg]!r} vs {name!r}")
            out[leg] = name
        return out


def discover_geometry_legacy_names(set_name: str, zero_d_root: str) -> list[str]:
    """
    Return sorted legacy-style geometry directory names for ``set_name``.
    """
    set_dir = os.path.join(zero_d_root, set_name)
    if not os.path.isdir(set_dir):
        raise FileNotFoundError(f"Set directory not found: {set_dir}")

    for rc in RUN_CONFIG_DIR_PREFERENCE:
        rc_path = os.path.join(set_dir, rc)
        if not os.path.isdir(rc_path):
            continue
        names = sorted(
            x for x in os.listdir(rc_path) if os.path.isdir(os.path.join(rc_path, x)) and not x.startswith(".")
        )
        if names:
            return names

    std_0d = os.path.join(set_dir, STANDARD_0D_SUBDIR)
    if os.path.isdir(std_0d):
        out = sorted(
            os.path.splitext(x)[0] for x in os.listdir(std_0d) if x.endswith(".json") and not x.startswith(".")
        )
        if out:
            return out

    raise FileNotFoundError(
        f"No geometry directories found under {set_dir} "
        f"(tried run configs {list(RUN_CONFIG_DIR_PREFERENCE)} and {STANDARD_0D_SUBDIR}/*.json)"
    )


def build_latex_table(
    set_order: Sequence[str],
    zero_d_root: str,
    csv_path: str,
    label: str = "tab:vmr-cohort-geometries",
) -> str:
    legacy_to_name = _load_legacy_to_name_from_csv(csv_path)
    columns: list[list[str]] = []
    headers: list[str] = []

    for set_name in set_order:
        canonical = _resolve_set_name(set_name)
        display = SET_DISPLAY_NAME.get(canonical, canonical).strip()
        headers.append(display)
        geos = discover_geometry_legacy_names(canonical, zero_d_root)
        missing = [g for g in geos if g not in legacy_to_name]
        if missing:
            raise ValueError(f"Set {canonical}: legacy IDs not in CSV Legacy Name column: {missing}")
        columns.append([legacy_to_name[g] for g in geos])

    ncols = len(headers)
    nrows = max((len(c) for c in columns), default=0)

    hdr_cells = " & ".join(f"\\textbf{{{h}}}" for h in headers)
    blocks: list[str] = [
        r"% Requires: \usepackage{booktabs}",
        r"% One column per cohort; geometry labels are CSV ``Name`` (via Legacy Name lookup).",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{VMR cohort geometries: dataset entry name (\texttt{Name} in "
        r"\texttt{data/dataset-svprojects.csv}) for each case in the cohort.}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{@{{}}{'l' * ncols}@{{}}}}",
        r"    \toprule",
        f"    {hdr_cells} \\\\",
        r"    \midrule",
    ]

    for i in range(nrows):
        cells = []
        for col in columns:
            if i < len(col):
                cells.append(_latex_escape_cell(col[i]))
            else:
                cells.append("")
        blocks.append("    " + " & ".join(cells) + r" \\")

    blocks.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write LaTeX table of VMR cohort geometries (CSV Name, one column per cohort)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(str(repo_root()), "results", "vmr_cohort_geometries.tex"),
        help="Output .tex path (default: results/vmr_cohort_geometries.tex under repo root)",
    )
    parser.add_argument(
        "--zero_d_root",
        default=os.path.join(str(repo_root()), "data", "zeroD"),
        help="Root directory containing VMR_* set folders",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(str(repo_root()), "data", "dataset-svprojects.csv"),
        help="Path to dataset-svprojects.csv",
    )
    parser.add_argument(
        "--label",
        default="tab:vmr-cohort-geometries",
        help="LaTeX \\label{...} for the table",
    )
    parser.add_argument(
        "sets",
        nargs="*",
        default=list(DEFAULT_SET_ORDER),
        help="Set names (default: VMR_rigid_aorta_adults_all VMR_abdo VMR_pulmo_healthy). "
        "VMR_pulmonary_healthy is accepted as an alias for VMR_pulmo_healthy.",
    )
    args = parser.parse_args()
    resolved = [_resolve_set_name(s) for s in args.sets]
    tex = build_latex_table(
        resolved,
        zero_d_root=args.zero_d_root,
        csv_path=args.csv,
        label=args.label,
    )
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
