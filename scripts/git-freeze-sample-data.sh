#!/usr/bin/env bash
# Mark tracked files under data/ as skip-worktree so local pipeline reruns do not
# appear in git status. Safe to run after clone; does not affect untracked files.
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
