#!/usr/bin/env bash
# Hide local edits to tracked seed files under data/ from git status.
#
# Sets git skip-worktree on every tracked path under data/ (bundled LFS samples).
# Pipeline outputs under data/ are usually gitignored and are not affected.
# Use after clone if you rerun batch/CV and do not want modified seeds in status.
#
# Usage (from repo root):
#   ./scripts/git-freeze-sample-data.sh
#   ./scripts/git-freeze-sample-data.sh --unfreeze
#
# See data/README.md and scripts/README.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

unfreeze=false
if [[ "${1:-}" == "--unfreeze" ]]; then
  unfreeze=true
fi

mapfile -t tracked < <(git ls-files 'data/')

if [[ ${#tracked[@]} -eq 0 ]]; then
  echo "No tracked files under data/."
  exit 0
fi

flag=skip-worktree
verb="Frozen"
if $unfreeze; then
  flag=no-skip-worktree
  verb="Unfrozen"
fi

for f in "${tracked[@]}"; do
  git update-index "--$flag" "$f"
done

echo "$verb ${#tracked[@]} tracked file(s) under data/."
