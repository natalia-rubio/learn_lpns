#!/usr/bin/env python3
"""Remove generated run-config trees under data/zeroD/<set_name>/, keeping standard-0d.

By default also removes derived training artifacts (jax_arrays, split_indices, models,
cross_validation) for the same cohort and run-config suffixes so config changes such as
``training.nondimensionalize_rsl`` take effect on the next CV run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.tools.file_io import STANDARD_0D_SUBDIR

# Bundled demo cohort — keep all run-config outputs (notebook/CV samples).
PROTECTED_ZEROD_SET_NAMES = frozenset({"VMR_aorta_starter"})

# (label, path under data_root or results_root)
_DERIVED_DATA_CATEGORIES = (
    ("jax_arrays", "jax_arrays"),
    ("split_indices", "split_indices"),
)
_DERIVED_RESULTS_CATEGORIES = (
    ("models", "models"),
    ("cross_validation", "cross_validation"),
)


def _zerod_set_dir(data_root: Path, set_name: str) -> Path:
    return data_root / "zeroD" / set_name


def _generated_subdirs(set_dir: Path, *, exclude: frozenset[str] = frozenset()) -> list[Path]:
    if not set_dir.is_dir():
        return []
    return sorted(
        p for p in set_dir.iterdir() if p.is_dir() and p.name not in exclude
    )


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MiB"
    return f"{num_bytes / 1024**3:.2f} GiB"


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _discover_run_config_names(
    set_name: str,
    *,
    data_root: Path,
    results_root: Path,
    also_derived: bool,
) -> list[str]:
    names: set[str] = set()
    zerod_set_dir = _zerod_set_dir(data_root, set_name)
    for path in _generated_subdirs(zerod_set_dir, exclude=frozenset({STANDARD_0D_SUBDIR})):
        names.add(path.name)

    if also_derived:
        for _, category in _DERIVED_DATA_CATEGORIES:
            for path in _generated_subdirs(data_root / category / set_name):
                names.add(path.name)
        for _, category in _DERIVED_RESULTS_CATEGORIES:
            for path in _generated_subdirs(results_root / category / set_name):
                names.add(path.name)

    return sorted(names)


def _resolve_run_configs(
    set_name: str,
    *,
    data_root: Path,
    results_root: Path,
    run_configs: list[str] | None,
    also_derived: bool,
) -> list[str]:
    if run_configs is not None:
        return sorted({name.strip() for name in run_configs if name.strip()})
    return _discover_run_config_names(
        set_name,
        data_root=data_root,
        results_root=results_root,
        also_derived=also_derived,
    )


def _collect_removal_targets(
    set_name: str,
    run_config_names: list[str],
    *,
    data_root: Path,
    results_root: Path,
    also_derived: bool,
    include_zerod: bool,
) -> list[tuple[str, Path]]:
    """Return (category_label, path) pairs to remove."""
    targets: list[tuple[str, Path]] = []

    if include_zerod:
        zerod_set_dir = _zerod_set_dir(data_root, set_name)
        for name in run_config_names:
            path = zerod_set_dir / name
            if path.is_dir():
                targets.append(("zeroD", path))

    if also_derived:
        for label, category in _DERIVED_DATA_CATEGORIES:
            base = data_root / category / set_name
            for name in run_config_names:
                path = base / name
                if path.is_dir():
                    targets.append((label, path))
        for label, category in _DERIVED_RESULTS_CATEGORIES:
            base = results_root / category / set_name
            for name in run_config_names:
                path = base / name
                if path.is_dir():
                    targets.append((label, path))

    return targets


def clean_set(
    set_name: str,
    *,
    data_root: Path,
    results_root: Path,
    run_configs: list[str] | None,
    also_derived: bool,
    dry_run: bool,
    yes: bool,
) -> int:
    """Remove generated subdirs for one cohort. Returns count of removed directories."""
    if set_name in PROTECTED_ZEROD_SET_NAMES:
        print(f"  ⊘ {set_name}: protected bundled cohort (skipped)")
        return 0

    zerod_set_dir = _zerod_set_dir(data_root, set_name)
    standard_dir = zerod_set_dir / STANDARD_0D_SUBDIR
    include_zerod = standard_dir.is_dir()
    if not include_zerod:
        print(
            f"  ⚠ {set_name}: no {STANDARD_0D_SUBDIR}/ under {zerod_set_dir} "
            f"(skipping zeroD cleanup)",
            file=sys.stderr,
        )

    run_config_names = _resolve_run_configs(
        set_name,
        data_root=data_root,
        results_root=results_root,
        run_configs=run_configs,
        also_derived=also_derived,
    )
    if not run_config_names:
        print(f"  ⊘ {set_name}: no run-config folders found to remove")
        return 0

    targets = _collect_removal_targets(
        set_name,
        run_config_names,
        data_root=data_root,
        results_root=results_root,
        also_derived=also_derived,
        include_zerod=include_zerod,
    )
    if not targets:
        scope = ", ".join(run_config_names)
        print(f"  ⊘ {set_name}: no existing folders for run-config(s) {scope}")
        return 0

    if run_configs is None:
        config_note = f"discovered run-config(s): {', '.join(run_config_names)}"
    else:
        config_note = f"run-config filter: {', '.join(run_config_names)}"
    derived_note = "zeroD + derived artifacts" if also_derived else "zeroD only"
    print(f"  {set_name} — {derived_note}; {config_note}")
    print(f"    will remove {len(targets)} folder(s), keep {STANDARD_0D_SUBDIR}/:")
    display_root = repo_root()
    for label, path in targets:
        size = _dir_size(path)
        try:
            rel = path.relative_to(display_root)
        except ValueError:
            rel = path
        print(f"    - [{label}] {rel}  ({_format_size(size)})")

    if dry_run:
        print("  (dry run — no files deleted)")
        return len(targets)

    if not yes:
        answer = input(f"Delete {len(targets)} folder(s) for {set_name}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("  Aborted.")
            return 0

    for _label, path in targets:
        shutil.rmtree(path)
    print(f"  ✓ Removed {len(targets)} folder(s) for {set_name}")
    return len(targets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete generated pipeline output under data/zeroD/<set_name>/ "
            f"(all subfolders except {STANDARD_0D_SUBDIR}/). "
            "By default also removes jax_arrays, split_indices, trained models, "
            "and cross_validation results for the same run-config suffixes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./scripts/clean_zerod_generated.py --set_name VMR_aorta --dry_run\n"
            "  ./scripts/clean_zerod_generated.py --set_name VMR_pulmo "
            "--run_config quadratic_resistor_gen_loss --yes\n"
            "  ./scripts/clean_zerod_generated.py --set_name VMR_aorta --no-also_derived --yes\n"
            "  ./scripts/clean_zerod_generated.py --all_sets --yes\n"
            "\n"
            "After changing training.nondimensionalize_rsl, clean and re-run CV so "
            "jax_arrays pickles and models are rebuilt with the new setting.\n"
            "\n"
            f"Protected (never cleaned): {', '.join(sorted(PROTECTED_ZEROD_SET_NAMES))}\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--set_name",
        nargs="+",
        metavar="NAME",
        help="One or more cohort names (e.g. VMR_aorta)",
    )
    group.add_argument(
        "--all_sets",
        action="store_true",
        help=f"Clean every directory under data/zeroD/ that contains {STANDARD_0D_SUBDIR}/",
    )
    parser.add_argument(
        "--run_config",
        nargs="+",
        metavar="SUFFIX",
        default=None,
        help=(
            "Limit cleanup to these run-config path suffixes (e.g. gen_loss "
            "quadratic_resistor_gen_loss). Default: all discovered suffixes for the cohort."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Data root directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=None,
        help="Results root directory (default: <repo>/results)",
    )
    parser.add_argument(
        "--also_derived",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also remove jax_arrays, split_indices, results/models, and "
            "results/cross_validation for the same run-config suffixes (default: on). "
            "Use --no-also_derived for zeroD-only cleanup."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print folders that would be removed without deleting",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    data_root = (args.data_root or (root / "data")).resolve()
    results_root = (args.results_root or (root / "results")).resolve()

    if args.all_sets:
        zero_d_root = data_root / "zeroD"
        if not zero_d_root.is_dir():
            print(f"No zeroD directory at {zero_d_root}", file=sys.stderr)
            return 1
        set_names = sorted(
            p.name
            for p in zero_d_root.iterdir()
            if p.is_dir()
            and (p / STANDARD_0D_SUBDIR).is_dir()
            and p.name not in PROTECTED_ZEROD_SET_NAMES
        )
        if not set_names:
            print(f"No cohorts with {STANDARD_0D_SUBDIR}/ under {zero_d_root}")
            return 0
    else:
        set_names = list(args.set_name)

    print(f"Data root: {data_root}")
    print(f"Results root: {results_root}")
    removed = 0
    for set_name in set_names:
        removed += clean_set(
            set_name,
            data_root=data_root,
            results_root=results_root,
            run_configs=args.run_config,
            also_derived=args.also_derived,
            dry_run=args.dry_run,
            yes=args.yes,
        )

    if removed and not args.dry_run:
        print(f"Done — removed {removed} folder(s) across {len(set_names)} cohort(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
