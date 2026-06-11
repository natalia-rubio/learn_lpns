#!/usr/bin/env bash
# Ensure Git LFS is installed and sample data files are materialized (not pointer stubs).
#
# Usage (from repo root):
#   ./scripts/setup_git_lfs.sh          # install hooks + git lfs pull
#   ./scripts/setup_git_lfs.sh --check  # verify only (no pull)
#
# Fresh clone:
#   git clone <repo>
#   cd learn_lpns
#   ./scripts/setup_git_lfs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
fi

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

is_lfs_pointer() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  head -n 1 "$path" 2>/dev/null | grep -q '^version https://git-lfs.github.com/spec/v1'
}

require_git_lfs() {
  if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
    die "Git LFS is not installed.

  macOS:  brew install git-lfs && git lfs install
  Linux:  see https://git-lfs.com"
  fi
}

pull_lfs_objects() {
  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    die "Not inside a git repository ($REPO_ROOT)"
  fi
  if ! git lfs install --local >/dev/null 2>&1; then
    git lfs install
  fi
  if $CHECK_ONLY; then
    return 0
  fi
  if git lfs pull 2>/dev/null; then
    log "git lfs pull complete"
  else
    log "git lfs pull skipped or nothing to fetch (OK for a fresh working tree)"
  fi
}

verify_sample_files() {
  local missing=0
  local pointer=0
  local geo path

  for geo in 0129_0000 0154_0001 0174_0000 0175_0000 0176_0000; do
    for path in \
      "data/zeroD/VMR_aortas/standard-0d/${geo}.json" \
      "data/zeroD/VMR_aortas/gen_loss/${geo}/bifurcations_EL_calibrated_output_BloodVesselJunction.json" \
      "data/oneD/VMR/${geo}/unsteady_soln.vtp"
    do
      if [[ ! -f "$path" ]]; then
        printf '  missing: %s\n' "$path" >&2
        missing=1
        continue
      fi
      if is_lfs_pointer "$path"; then
        printf '  LFS pointer (not downloaded): %s\n' "$path" >&2
        pointer=1
      fi
    done
  done

  if [[ "$missing" -ne 0 ]]; then
    die "Bundled sample files missing. Clone the repo and run: ./scripts/setup_git_lfs.sh"
  fi
  if [[ "$pointer" -ne 0 ]]; then
    die "Some files are still Git LFS pointers. Run: git lfs pull"
  fi
  log "Sample data files present and materialized (5 geos × 3 files)"
}

require_git_lfs
pull_lfs_objects
verify_sample_files
