#!/usr/bin/env python3
"""Remove generated run-config trees under data/zeroD/<set_name>/, keeping standard-0d."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from learn_lpns.tools.paths import repo_root
from learn_lpns.zerod_calibration.tools.file_io import STANDARD_0D_SUBDIR

# Bundled demo cohort — keep all run-config outputs (notebook/CV samples).
PROTECTED_ZEROD_SET_NAMES = frozenset({"VMR_aorta_starter"})


def _zerod_set_dir(data_root: Path, set_name: str) -> Path:
    return data_root / "zeroD" / set_name


def _generated_subdirs(set_dir: Path) -> list[Path]:
    if not set_dir.is_dir():
        return []
    return sorted(
        p for p in set_dir.iterdir() if p.is_dir() and p.name != STANDARD_0D_SUBDIR
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


def clean_set(
    set_name: str,
    *,
    data_root: Path,
    dry_run: bool,
    yes: bool,
) -> int:
    """Remove generated subdirs for one cohort. Returns count of removed directories."""
    if set_name in PROTECTED_ZEROD_SET_NAMES:
        print(f"  ⊘ {set_name}: protected bundled cohort (skipped)")
        return 0

    set_dir = _zerod_set_dir(data_root, set_name)
    if not set_dir.is_dir():
        print(f"  ✗ Not found: {set_dir}", file=sys.stderr)
        return 0

    standard_dir = set_dir / STANDARD_0D_SUBDIR
    if not standard_dir.is_dir():
        print(
            f"  ⚠ Skipping {set_name}: no {STANDARD_0D_SUBDIR}/ "
            f"(expected seed JSONs at {standard_dir})",
            file=sys.stderr,
        )
        return 0

    targets = _generated_subdirs(set_dir)
    if not targets:
        print(f"  ⊘ {set_name}: nothing to remove (only {STANDARD_0D_SUBDIR}/ present)")
        return 0

    print(f"  {set_name} — will remove {len(targets)} folder(s), keep {STANDARD_0D_SUBDIR}/:")
    for path in targets:
        size = _dir_size(path)
        print(f"    - {path.relative_to(data_root)}  ({_format_size(size)})")

    if dry_run:
        print("  (dry run — no files deleted)")
        return len(targets)

    if not yes:
        answer = input(f"Delete {len(targets)} folder(s) under {set_dir}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("  Aborted.")
            return 0

    for path in targets:
        shutil.rmtree(path)
    print(f"  ✓ Removed {len(targets)} folder(s) from {set_dir}")
    return len(targets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete generated pipeline output under data/zeroD/<set_name>/ "
            f"(all subfolders except {STANDARD_0D_SUBDIR}/)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./scripts/clean_zerod_generated.py --set_name VMR_aorta --dry_run\n"
            "  ./scripts/clean_zerod_generated.py --set_name VMR_aorta VMR_abdo --yes\n"
            "  ./scripts/clean_zerod_generated.py --all_sets --yes\n"
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
        "--data_root",
        type=Path,
        default=None,
        help="Data root directory (default: <repo>/data)",
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
    data_root = args.data_root or (repo_root() / "data")
    data_root = data_root.resolve()

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
    removed = 0
    for set_name in set_names:
        removed += clean_set(
            set_name,
            data_root=data_root,
            dry_run=args.dry_run,
            yes=args.yes,
        )

    if removed and not args.dry_run:
        print(f"Done — removed {removed} top-level folder(s) across {len(set_names)} cohort(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
