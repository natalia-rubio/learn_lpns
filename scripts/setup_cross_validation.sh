#!/usr/bin/env bash
# Clone/build dependencies for learn_lpns/zerod_calibration/run_cross_validation.py:
#   - learn_lpns (this repo + bundled sample data under data/)
#   - svZeroDPlus fork (svzerodsolver + svzerodcalibrator)
#   - Python venv with JAX and pipeline packages
#
# Usage (from anywhere):
#   ./scripts/setup_cross_validation.sh
#
# Override defaults with environment variables (see --help).

set -euo pipefail

LEARN_LPNS_REPO="${LEARN_LPNS_REPO:-https://github.com/natalia-rubio/learn_lpns.git}"
LEARN_LPNS_BRANCH="${LEARN_LPNS_BRANCH:-id_based_wiring}"
SVZEROD_REPO="${SVZEROD_REPO:-https://github.com/natalia-rubio/svZeroDPlus.git}"
SVZEROD_BRANCH="${SVZEROD_BRANCH:-J-J_wiring}"
JAX_VARIANT="${JAX_VARIANT:-jax[cpu]}"
PYTHON="${PYTHON:-}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
MIN_PYTHON_VERSION="3.10"

SKIP_CLONE=false
SKIP_SOLVER_BUILD=false
SKIP_PYTHON=false

usage() {
  cat <<'EOF'
setup_cross_validation.sh — prepare environment for run_cross_validation.py

Install layout (default):
  <workspace>/
    learn_lpns/          this repo
    svZeroDPlus/         fork with svzerodsolver + svzerodcalibrator
    svZeroDPlus/Release/ build output (SVZEROD_INSTALL_DIR)

Steps:
  1. Clone or update learn_lpns and svZeroDPlus (unless --skip-clone)
  2. cmake build svzerodsolver and svzerodcalibrator (unless --skip-solver-build)
  3. Create .venv and pip install JAX + requirements.txt (unless --skip-python)
  4. Verify bundled sample data (data/zeroD, data/oneD/VMR)
  5. Write scripts/cv_env.sh for easy activation

Options:
  --skip-clone           Do not git clone/fetch; use existing directories
  --skip-solver-build    Skip cmake build of svZeroDPlus
  --skip-python          Skip venv creation and pip installs
  -h, --help             Show this message

Environment overrides:
  WORKSPACE_DIR          Parent of learn_lpns and svZeroDPlus (default: parent of this repo)
  LEARN_LPNS_DIR         Path to learn_lpns checkout (default: <workspace>/learn_lpns)
  SVZEROD_DIR            Path to svZeroDPlus checkout (default: <workspace>/svZeroDPlus)
  LEARN_LPNS_REPO        Git remote for learn_lpns
  LEARN_LPNS_BRANCH      Git branch for learn_lpns (default: id_based_wiring)
  SVZEROD_REPO           Git remote for svZeroDPlus fork
  SVZEROD_BRANCH         Git branch for svZeroDPlus (default: J-J_wiring)
  JAX_VARIANT            pip JAX extra, e.g. jax[cpu] or jax[cuda12] (default: jax[cpu])
  PYTHON                 Python 3.10+ interpreter (auto-detected if unset)

Prerequisites (checked before clone/build):
  - git, cmake, a C++ compiler (Xcode CLT on macOS: xcode-select --install)
  - Python 3.10+ (macOS: brew install python@3.12)
  - On macOS, plain python3 is often 3.9 from Xcode — the script searches for
    python3.12, python3.11, etc., or set PYTHON explicitly.

After setup:
  source scripts/cv_env.sh
  learn-lpns-cv --set_name VMR_aortas --geometry_variant bifurcations_EL --num_trials 2

Sample data ships in learn_lpns (5 VMR geometries). Pipeline outputs under data/
and results/ are generated on first run.
EOF
}

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  case "$1" in
    cmake)
      if [[ "$(uname -s)" == "Darwin" ]]; then
        die "Missing cmake. Install with: brew install cmake"
      fi
      ;;
    git)
      if [[ "$(uname -s)" == "Darwin" ]]; then
        die "Missing git. Install Xcode Command Line Tools: xcode-select --install"
      fi
      ;;
  esac
  die "Missing required command: $1"
}

python_ok() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

python_version_label() {
  "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
}

find_python() {
  if [[ -n "$PYTHON" ]]; then
    command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON not found: $PYTHON"
    python_ok "$PYTHON" || die "PYTHON=$PYTHON is older than $MIN_PYTHON_VERSION (JAX requires 3.10+)"
    printf '%s' "$PYTHON"
    return 0
  fi

  local candidates=()
  local ver
  for ver in 13 12 11 10; do
    candidates+=("python3.$ver")
    if [[ "$(uname -s)" == "Darwin" ]]; then
      candidates+=("/opt/homebrew/bin/python3.$ver" "/usr/local/bin/python3.$ver")
    fi
  done
  candidates+=("python3")

  local py
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1 && python_ok "$py"; then
      printf '%s' "$py"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    local found
    found="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo unknown)"
    die "Python $MIN_PYTHON_VERSION+ required (found python3 = $found). On macOS install a newer Python, e.g.:
  brew install python@3.12
  PYTHON=\$(brew --prefix python@3.12)/bin/python3.12 ./scripts/setup_cross_validation.sh"
  fi
  die "Python $MIN_PYTHON_VERSION+ not found. On macOS: brew install python@3.12"
}

check_prerequisites() {
  need_cmd git
  need_cmd cmake
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! xcrun --find clang >/dev/null 2>&1; then
      die "C++ compiler not found. Install Xcode Command Line Tools: xcode-select --install"
    fi
  fi
  PYTHON="$(find_python)"
  log "Using Python: $PYTHON ($(python_version_label "$PYTHON"))"
}

clone_or_update() {
  local dir="$1" repo="$2" branch="$3" label="$4"
  if [[ -d "$dir/.git" ]]; then
    log "Updating $label at $dir (branch $branch)"
    git -C "$dir" fetch origin
    git -C "$dir" checkout "$branch"
    git -C "$dir" pull --ff-only origin "$branch" || true
  else
    log "Cloning $label into $dir (branch $branch)"
    git clone --branch "$branch" --depth 1 "$repo" "$dir"
  fi
}

build_svzerod() {
  local src="$1"
  local build_dir="$src/$CMAKE_BUILD_TYPE"

  log "Configuring svZeroDPlus in $build_dir"
  cmake -S "$src" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DPython_EXECUTABLE="$PYTHON"

  log "Building svzerodsolver and svzerodcalibrator"
  cmake --build "$build_dir" --target svzerodsolver svzerodcalibrator -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
}

verify_sample_data() {
  local root="$1"
  local missing=0
  local check
  for check in \
    "$root/data/README.md" \
    "$root/data/zeroD/VMR_aortas/standard-0d/0076_1001.json" \
    "$root/data/oneD/VMR/0076_1001/unsteady_soln.vtp"
  do
    if [[ ! -f "$check" ]]; then
      printf '  missing: %s\n' "$check" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    die "Bundled sample data not found under $root/data/. Ensure learn_lpns was cloned with tracked data/ files (branch $LEARN_LPNS_BRANCH)."
  fi
  log "Sample data present under $root/data/"
}

write_cv_env() {
  local learn_lpns_dir="$1"
  local svzerod_install="$2"
  local env_file="$learn_lpns_dir/scripts/cv_env.sh"
  cat >"$env_file" <<EOF
# Generated by scripts/setup_cross_validation.sh — source before running CV.
#   source scripts/cv_env.sh

export SVZEROD_INSTALL_DIR="$svzerod_install"
export PATH="\$SVZEROD_INSTALL_DIR:\$PATH"

if [[ -f "$learn_lpns_dir/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$learn_lpns_dir/.venv/bin/activate"
fi

cd "$learn_lpns_dir"
EOF
  chmod +x "$env_file"
  log "Wrote $env_file"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-clone) SKIP_CLONE=true ;;
    --skip-solver-build) SKIP_SOLVER_BUILD=true ;;
    --skip-python) SKIP_PYTHON=true ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_LPNS_DIR="${LEARN_LPNS_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "$LEARN_LPNS_DIR")}"
SVZEROD_DIR="${SVZEROD_DIR:-$WORKSPACE_DIR/svZeroDPlus}"
SVZEROD_INSTALL_DIR="$SVZEROD_DIR/$CMAKE_BUILD_TYPE"

log "Workspace: $WORKSPACE_DIR"
log "learn_lpns: $LEARN_LPNS_DIR"
log "svZeroDPlus: $SVZEROD_DIR"
log "SVZEROD_INSTALL_DIR: $SVZEROD_INSTALL_DIR"

check_prerequisites

if ! $SKIP_CLONE; then
  if [[ ! -d "$LEARN_LPNS_DIR/.git" ]]; then
    clone_or_update "$LEARN_LPNS_DIR" "$LEARN_LPNS_REPO" "$LEARN_LPNS_BRANCH" "learn_lpns"
  else
    log "Using existing learn_lpns at $LEARN_LPNS_DIR"
    if [[ -n "${LEARN_LPNS_BRANCH:-}" ]]; then
      git -C "$LEARN_LPNS_DIR" fetch origin 2>/dev/null || true
      git -C "$LEARN_LPNS_DIR" checkout "$LEARN_LPNS_BRANCH" 2>/dev/null || true
    fi
  fi
  clone_or_update "$SVZEROD_DIR" "$SVZEROD_REPO" "$SVZEROD_BRANCH" "svZeroDPlus"
else
  log "Skipping git clone/update (--skip-clone)"
  [[ -d "$LEARN_LPNS_DIR" ]] || die "learn_lpns not found at $LEARN_LPNS_DIR"
  [[ -d "$SVZEROD_DIR" ]] || die "svZeroDPlus not found at $SVZEROD_DIR"
fi

if ! $SKIP_SOLVER_BUILD; then
  build_svzerod "$SVZEROD_DIR"
else
  log "Skipping svZeroDPlus build (--skip-solver-build)"
fi

for bin in svzerodsolver svzerodcalibrator; do
  [[ -x "$SVZEROD_INSTALL_DIR/$bin" ]] || die "$bin not found at $SVZEROD_INSTALL_DIR/$bin (build failed or wrong CMAKE_BUILD_TYPE?)"
done
log "svZeroD binaries OK: $SVZEROD_INSTALL_DIR"

if ! $SKIP_PYTHON; then
  VENV_DIR="$LEARN_LPNS_DIR/.venv"
  if [[ -d "$VENV_DIR" ]]; then
    if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      log "Removing existing .venv (Python < 3.10 or incomplete install)"
      rm -rf "$VENV_DIR"
    fi
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating Python venv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools
  log "Installing $JAX_VARIANT (install JAX before other deps)"
  python -m pip install -U "$JAX_VARIANT"
  log "Installing package (editable) and dev extras"
  python -m pip install -e "$LEARN_LPNS_DIR[dev]"
else
  log "Skipping Python setup (--skip-python)"
fi

verify_sample_data "$LEARN_LPNS_DIR"
write_cv_env "$LEARN_LPNS_DIR" "$SVZEROD_INSTALL_DIR"

cat <<EOF

Setup complete.

  source $LEARN_LPNS_DIR/scripts/cv_env.sh

  learn-lpns-cv \\
    --set_name VMR_aortas \\
    --geometry_variant bifurcations_EL \\
    --num_trials 2 \\
    --run_config gen_loss

EOF
