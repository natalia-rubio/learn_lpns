#!/usr/bin/env python3
"""Forward-selection ablation for quadratic_resistor_gen_loss training overrides.

Preserves config/defaults.yaml. Uses full YAML copies via LEARN_LPNS_CONFIG and an
isolated sandbox so existing QR models/CV outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SET_NAME = "VMR_pulmo"
GEOMETRY = "bifurcations_EL"
RUN_CONFIG = "quadratic_resistor_gen_loss"
SCREEN_TRIALS = (2, 3)
PRIMARY_COL = "PressureMeanRelError_BloodVesselJunction_NN_plus_Vessel_NN"
CONTROL_GUARDRAIL_PP = 2.0  # reject if trial-3 MAPE worsens by more than this many percentage points
COHORT_GEOMETRIES = ("0080_0001", "0081_0001", "0082_0001", "0086_0001", "0155_0001")

# Overrides identical to shared training defaults (not screened as experiments).
NOOP_OVERRIDES = {
    "generation_weighted_loss_decay_base": 2.0,
    "early_stop_loss_threshold": 1.0e-7,
    "vessel.layer_width": 10,
    "S.junction_num_layers": 1,
    "S.junction_layer_width": 10,
}

# One-setting (or one-group) candidates relative to shared defaults.
CANDIDATES: dict[str, dict[str, Any]] = {
    "01_include_speed_change": {"include_speed_change": True},
    "02_clip_predictions": {"clip_predictions": True},
    "03_clip_input_features": {"clip_input_features": True},
    "04_activation_leaky_relu": {"activation": "leaky_relu"},
    "05_optimizer_decay_0p99": {"optimizer": {"decay_rate": 0.99}},
    "06_vessel_num_layers_1": {"vessel": {"num_layers": 1}},
    "07_r_arch_4x10": {
        "rri_coefficients": [
            {"name": "R", "junction_num_layers": 4, "junction_layer_width": 10},
        ]
    },
    "08_s_gen_decay_3": {
        "rri_coefficients": [
            {"name": "S", "generation_weighted_loss_decay_base": 3.0},
        ]
    },
    "09_l_arch_2x10": {
        "rri_coefficients": [
            {"name": "L", "junction_num_layers": 2, "junction_layer_width": 10},
        ]
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif key == "rri_coefficients" and isinstance(value, list) and isinstance(out.get(key), list):
            by_name = {e["name"]: deepcopy(e) for e in out[key] if isinstance(e, dict) and "name" in e}
            for entry in value:
                name = entry["name"]
                if name in by_name:
                    by_name[name] = _deep_merge(by_name[name], entry)
                else:
                    by_name[name] = deepcopy(entry)
            # Preserve original order, append new names at end.
            names = [e["name"] for e in out[key]]
            for name in by_name:
                if name not in names:
                    names.append(name)
            out[key] = [by_name[n] for n in names]
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _campaign_dir_from_args(args: argparse.Namespace) -> Path:
    if args.campaign_dir:
        return Path(args.campaign_dir).resolve()
    matches = sorted((REPO_ROOT / "results" / "experiments").glob("qr_ablation_*"))
    if not matches:
        raise SystemExit("No campaign directory found under results/experiments/qr_ablation_*")
    return matches[-1].resolve()


def make_variant_configs(campaign: Path) -> dict[str, Path]:
    """Write full immutable YAML variants; return name -> path."""
    source = campaign / "configs" / "source_defaults.yaml"
    if not source.is_file():
        source = REPO_ROOT / "config" / "defaults.yaml"
    base = _load_yaml(source)

    # Shared-default QR: empty training overlay.
    shared = deepcopy(base)
    shared["training_run_configs"] = {
        "gen_loss": {},
        "quadratic_resistor_gen_loss": {},
    }
    # Keep physics/data QR overlay as in source (flow_split etc.).
    if "run_config_overrides" not in shared:
        shared["run_config_overrides"] = {}

    config_dir = campaign / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    mirror = REPO_ROOT / "config" / "experiments" / campaign.name
    mirror.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    def write_named(name: str, data: dict[str, Any]) -> Path:
        p = config_dir / f"{name}.yaml"
        _write_yaml(p, data)
        _write_yaml(mirror / f"{name}.yaml", data)
        paths[name] = p
        return p

    write_named("00_shared_defaults", shared)
    write_named("current_qr_profile", base)

    noop_note = {
        "identical_to_shared_default": NOOP_OVERRIDES,
        "reason": "These QR overlay keys match shared training defaults and are not screened.",
    }
    (campaign / "noop_overrides.json").write_text(json.dumps(noop_note, indent=2))

    for name, overlay in CANDIDATES.items():
        data = deepcopy(shared)
        data["training_run_configs"]["quadratic_resistor_gen_loss"] = deepcopy(overlay)
        write_named(name, data)

    # Diff summary
    diffs = []
    for name, path in paths.items():
        data = _load_yaml(path)
        overlay = data.get("training_run_configs", {}).get("quadratic_resistor_gen_loss", {})
        diffs.append({"name": name, "training_overlay": overlay})
    (campaign / "resolved_configs.json").write_text(json.dumps(diffs, indent=2))
    return paths


def setup_sandbox(campaign: Path) -> Path:
    """Create sandbox with symlinked code/data/venv and writable QR output trees."""
    campaign = campaign.resolve()
    sandbox = (campaign / "sandbox").resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dest: Path) -> None:
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        dest.symlink_to(src)

    link(REPO_ROOT / "learn_lpns", sandbox / "learn_lpns")
    link(REPO_ROOT / ".venv", sandbox / ".venv")
    link(REPO_ROOT / "scripts", sandbox / "scripts")
    link(REPO_ROOT / "pyproject.toml", sandbox / "pyproject.toml")

    # Build a hybrid data tree: symlink all top-level entries, but copy the QR zeroD
    # tree so NN deploy outputs do not mutate the main repository.
    data_sb = sandbox / "data"
    data_sb.mkdir(exist_ok=True)
    main_data = REPO_ROOT / "data"
    # Writable local copies for trees that experiments may rebuild or mutate.
    writable_copies = {
        "zeroD": main_data / "zeroD" / SET_NAME / RUN_CONFIG,
        "jax_arrays": main_data / "jax_arrays" / SET_NAME / RUN_CONFIG,
        "split_indices": main_data / "split_indices" / SET_NAME / RUN_CONFIG,
        "ml_inputs": main_data / "ml_inputs" / SET_NAME / RUN_CONFIG,
        "feature_histograms": main_data / "feature_histograms" / SET_NAME / RUN_CONFIG,
    }
    for child in main_data.iterdir():
        dest = data_sb / child.name
        if child.name in writable_copies:
            continue
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(child)

    def ensure_copied_subtree(kind: str, src: Path) -> Path:
        if kind == "zeroD":
            dest = data_sb / "zeroD" / SET_NAME / RUN_CONFIG
            parent_root = data_sb / "zeroD"
        else:
            dest = data_sb / kind / SET_NAME / RUN_CONFIG
            parent_root = data_sb / kind
        parent_root.mkdir(exist_ok=True)
        (parent_root / SET_NAME).mkdir(exist_ok=True)
        # Symlink sibling sets/configs
        main_kind = main_data / kind
        if main_kind.is_dir():
            for set_dir in main_kind.iterdir():
                if not set_dir.is_dir():
                    continue
                set_dest = parent_root / set_dir.name
                set_dest.mkdir(exist_ok=True)
                for cfg_dir in set_dir.iterdir():
                    cfg_dest = set_dest / cfg_dir.name
                    if set_dir.name == SET_NAME and cfg_dir.name == RUN_CONFIG:
                        continue
                    if cfg_dest.exists() or cfg_dest.is_symlink():
                        continue
                    cfg_dest.symlink_to(cfg_dir)
        if src.is_dir() and not dest.exists():
            print(f"  Copying {kind}/{SET_NAME}/{RUN_CONFIG} into sandbox", flush=True)
            shutil.copytree(src, dest, symlinks=True)
        return dest

    for kind, src in writable_copies.items():
        if src.is_dir():
            ensure_copied_subtree(kind, src)

    (sandbox / "results").mkdir(exist_ok=True)
    (sandbox / "config").mkdir(exist_ok=True)
    link(campaign / "configs", sandbox / "config" / "experiments_configs")
    return sandbox


def clear_sandbox_qr_outputs(sandbox: Path) -> None:
    """Remove QR models/CV outputs so the next variant cannot reuse prior models."""
    targets = [
        sandbox / "results" / "models" / SET_NAME / RUN_CONFIG,
        sandbox / "results" / "cross_validation" / SET_NAME / RUN_CONFIG,
        sandbox / "results" / "location_comparison" / RUN_CONFIG / SET_NAME,
        sandbox / "results" / "param_comparison" / RUN_CONFIG / SET_NAME,
    ]
    for path in targets:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
    # Also clear NN deploy products in the sandbox QR zeroD copy (keep calibrated/geometric).
    qr_zero = sandbox / "data" / "zeroD" / SET_NAME / RUN_CONFIG
    if qr_zero.is_dir():
        for geo_dir in qr_zero.iterdir():
            if not geo_dir.is_dir():
                continue
            for p in geo_dir.iterdir():
                name = p.name
                if any(
                    tok in name
                    for tok in (
                        "_NN_",
                        "NN_Junction",
                        "NN_Vessel",
                        "mse_comparison",
                        "downsampled_to_3d",
                    )
                ):
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()


def archive_sandbox_outputs(sandbox: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for rel in (
        Path("results") / "models" / SET_NAME / RUN_CONFIG,
        Path("results") / "cross_validation" / SET_NAME / RUN_CONFIG,
    ):
        src = sandbox / rel
        if src.exists():
            target = dest / rel
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, target)


def _variant_wants_speed_change(config_yaml: Path) -> bool:
    data = _load_yaml(config_yaml)
    base = bool(data.get("training", {}).get("include_speed_change", False))
    overlay = (
        data.get("training_run_configs", {})
        .get(RUN_CONFIG, {})
    )
    if isinstance(overlay, dict) and "include_speed_change" in overlay:
        return bool(overlay["include_speed_change"])
    return base


def _sandbox_python(sandbox: Path) -> str:
    """Absolute venv interpreter path without following symlinks to the base CPython binary.

    Using Path.resolve() collapses `.venv/bin/python` to Homebrew's python3.12 and drops
    the venv site-packages (ModuleNotFoundError: yaml). Prefer abspath of the venv entrypoint.
    """
    candidates = [
        (sandbox / ".venv" / "bin" / "python").absolute(),
        (REPO_ROOT / ".venv" / "bin" / "python").absolute(),
        Path(sys.executable).absolute(),
    ]
    for py in candidates:
        if py.is_file():
            return str(py)
    return str(Path(sys.executable).absolute())


def _jax_has_speed_change(sandbox: Path) -> bool | None:
    sandbox = sandbox.resolve()
    pkl = (
        sandbox
        / "data"
        / "jax_arrays"
        / SET_NAME
        / RUN_CONFIG
        / GEOMETRY
        / "all"
        / "jax_arrays_num_geos_5.pkl"
    )
    if not pkl.is_file():
        return None
    # Avoid importing jax in the orchestrator if possible; use dill via venv python.
    code = (
        "from learn_lpns.tools.basic import load_dict; "
        f"d=load_dict({str(pkl)!r}); "
        "print('speed_change' in (d.get('input_feature_names') or []))"
    )
    proc = subprocess.run([_sandbox_python(sandbox), "-c", code], cwd=str(sandbox), capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip().endswith("True")


def _frozen_split_source(campaign_dir: Path) -> Path | None:
    """Resolve snapshot root. Accept either .../snapshot/{all,forward} or .../snapshot/<geometry>/{all,forward}."""
    src = campaign_dir / "split_indices_snapshot"
    if not src.is_dir():
        return None
    if (src / "all").is_dir() or (src / "forward").is_dir():
        return src
    nested = src / GEOMETRY
    if nested.is_dir() and ((nested / "all").is_dir() or (nested / "forward").is_dir()):
        return nested
    return src


def _restore_frozen_split_indices(sandbox: Path, log_path: Path) -> None:
    """Keep CV folds identical to the campaign snapshot after data_processing regenerates splits."""
    campaign_dir = sandbox.parent
    src = _frozen_split_source(campaign_dir)
    dst = sandbox / "data" / "split_indices" / SET_NAME / RUN_CONFIG / GEOMETRY
    if src is None:
        with log_path.open("a") as log:
            log.write(f"# WARNING: missing split snapshot under {campaign_dir / 'split_indices_snapshot'}; leaving regenerated splits\n")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    with log_path.open("a") as log:
        log.write(f"# restored frozen split indices from {src} -> {dst}\n")


def ensure_jax_features_match_config(sandbox: Path, config_yaml: Path, log_path: Path) -> None:
    """Rebuild sandbox jax arrays when include_speed_change disagrees with the pickle."""
    sandbox = sandbox.resolve()
    want = _variant_wants_speed_change(config_yaml)
    have = _jax_has_speed_change(sandbox)
    if have is not None and have == want:
        return
    env = os.environ.copy()
    env["LEARN_LPNS_CONFIG"] = str(config_yaml.resolve())
    env["LEARN_LPNS_ROOT"] = str(sandbox)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str((sandbox / ".mplconfig").resolve())
    env["PYTHONUNBUFFERED"] = "1"
    (sandbox / ".mplconfig").mkdir(exist_ok=True)
    cmd = [
        _sandbox_python(sandbox),
        "-u",
        "-m",
        "learn_lpns.data_processing.run_data_processing",
        "--set_name",
        SET_NAME,
        "--geometry_variant",
        GEOMETRY,
        "--run_config",
        RUN_CONFIG,
        "--geometries",
        *COHORT_GEOMETRIES,
    ]
    with log_path.open("a") as log:
        log.write(f"\n# rebuilding jax arrays want_speed_change={want} had={have}\n")
        log.write(f"# {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(sandbox), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"# data_processing exit={proc.returncode}\n")
        if proc.returncode != 0:
            raise RuntimeError(f"data_processing failed while aligning features for {config_yaml}")
    _restore_frozen_split_indices(sandbox, log_path)


def run_cv(
    *,
    sandbox: Path,
    config_yaml: Path,
    trials: list[int] | None,
    num_trials: int,
    log_path: Path,
    skip_barchart: bool = True,
) -> int:
    sandbox = sandbox.resolve()
    env = os.environ.copy()
    env["LEARN_LPNS_CONFIG"] = str(config_yaml.resolve())
    # Force all repo_root()-based I/O (models, zeroD deploy, CV CSVs) into the sandbox.
    # Required because sandbox/learn_lpns is a symlink and Path.resolve() escapes otherwise.
    env["LEARN_LPNS_ROOT"] = str(sandbox)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str((sandbox / ".mplconfig").resolve())
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONUNBUFFERED"] = "1"
    (sandbox / ".mplconfig").mkdir(exist_ok=True)

    cmd = [
        _sandbox_python(sandbox),
        "-u",
        "-m",
        "learn_lpns.zerod_calibration.run_cross_validation",
        "--set_name",
        SET_NAME,
        "--geometry_variant",
        GEOMETRY,
        "--num_trials",
        str(num_trials),
        "--run_config",
        RUN_CONFIG,
        "--max_parallel_trials",
        "1",
        # Do NOT pass --no_redo: the sandbox zeroD tree is seeded from the main QR
        # copy and may contain stale NN/MSE artifacts. Ablation trials must always
        # regenerate deploy products from the models just trained.
    ]
    if skip_barchart:
        cmd.append("--skip_barchart")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # Guardrail: confirm the child interpreter will resolve repo_root() into the sandbox.
    probe = subprocess.run(
        [
            _sandbox_python(sandbox),
            "-c",
            "from learn_lpns.tools.paths import repo_root; print(repo_root())",
        ],
        cwd=str(sandbox),
        env=env,
        capture_output=True,
        text=True,
    )
    resolved_root = (probe.stdout or "").strip()
    if probe.returncode != 0 or Path(resolved_root).resolve() != sandbox:
        raise RuntimeError(
            "Sandbox isolation check failed: child repo_root()="
            f"{resolved_root!r} (stderr={probe.stderr!r}). "
            "Expected LEARN_LPNS_ROOT to point at the campaign sandbox."
        )
    with log_path.open("w") as log:
        log.write(f"# cmd: {' '.join(cmd)}\n")
        log.write(f"# LEARN_LPNS_CONFIG={env['LEARN_LPNS_CONFIG']}\n")
        log.write(f"# LEARN_LPNS_ROOT={env['LEARN_LPNS_ROOT']}\n")
        log.write(f"# repo_root_probe={resolved_root}\n")
        log.write(f"# cwd={sandbox}\n")
        log.write(f"# start={_utc_now()}\n\n")
        log.flush()
    ensure_jax_features_match_config(sandbox, config_yaml, log_path)
    with log_path.open("a") as log:
        if trials is None:
            proc = subprocess.run(cmd, cwd=str(sandbox), env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.returncode
        else:
            # Run selected trials in one interpreter so companion-metric merge across
            # --trial invocations cannot drop MAPE columns if an older CV build is used.
            driver = (
                "import sys\n"
                "from learn_lpns.zerod_calibration.run_cross_validation import run_cross_validation\n"
                f"trials = {list(trials)!r}\n"
                "rc = 0\n"
                "for trial in trials:\n"
                "    print(f'\\n# --- trial {trial} ---', flush=True)\n"
                "    try:\n"
                "        run_cross_validation(\n"
                f"            set_name={SET_NAME!r},\n"
                f"            geometry_variant={GEOMETRY!r},\n"
                f"            num_trials={int(num_trials)},\n"
                f"            run_config_suffix={RUN_CONFIG!r},\n"
                "            trial_index=trial,\n"
                "            skip_barchart=True,\n"
                "            no_redo=False,\n"
                "            max_parallel_trials=1,\n"
                "        )\n"
                "    except SystemExit as exc:\n"
                "        code = int(getattr(exc, 'code', 1) or 1)\n"
                "        rc = code or rc\n"
                "        print(f'# trial {trial} SystemExit {code}', flush=True)\n"
                "        break\n"
                "    except Exception as exc:\n"
                "        rc = 1\n"
                "        print(f'# trial {trial} failed: {exc!r}', flush=True)\n"
                "        break\n"
                "sys.exit(rc)\n"
            )
            driver_cmd = [_sandbox_python(sandbox), "-u", "-c", driver]
            log.write(f"\n# multi-trial driver trials={list(trials)}\n")
            log.flush()
            proc = subprocess.run(driver_cmd, cwd=str(sandbox), env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.returncode
        log.write(f"\n# end={_utc_now()} elapsed_s={time.time()-start:.1f} exit={rc}\n")
    return rc


def _read_metric_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def extract_metrics(artifact_dir: Path) -> dict[str, Any]:
    cv_dir = artifact_dir / "results" / "cross_validation" / SET_NAME / RUN_CONFIG
    out: dict[str, Any] = {"artifact_dir": str(artifact_dir), "trials": {}}
    metric_files = {
        "pressure_mean_rel_error": cv_dir / f"{GEOMETRY}_cv_summary_pressure_mean_rel_error.csv",
        "pressure_max_rel_error": cv_dir / f"{GEOMETRY}_cv_summary_pressure_max_rel_error.csv",
        "pressure_mse": cv_dir / f"{GEOMETRY}_cv_summary_pressure_mse.csv",
        "mse": cv_dir / f"{GEOMETRY}_cv_summary.csv",
    }
    for metric_name, path in metric_files.items():
        rows = _read_metric_rows(path)
        out[metric_name] = {}
        for row in rows:
            tid = (row.get("trial_id") or "").strip()
            if tid in {"mean", "std", ""}:
                out[metric_name][tid or "blank"] = row
                continue
            try:
                t = int(tid)
            except ValueError:
                continue
            out["trials"].setdefault(str(t), {})[metric_name] = row
            out[metric_name][str(t)] = row
    # Convenience primary values
    primary = {}
    for t in ("2", "3", "0", "1", "4", "mean", "std"):
        row = out.get("pressure_mean_rel_error", {}).get(t)
        if row and PRIMARY_COL in row and row[PRIMARY_COL] not in ("", None):
            try:
                primary[t] = float(row[PRIMARY_COL])
            except ValueError:
                pass
    out["primary_nn_vessel_mape"] = primary
    return out


def _mape_pp(frac: float | None) -> float | None:
    if frac is None or (isinstance(frac, float) and math.isnan(frac)):
        return None
    return 100.0 * float(frac)


def passes_guards(metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> tuple[bool, str]:
    """Require trial-2 improvement vs shared baseline; reject if trial-3 worsens > guardrail."""
    cur = metrics.get("primary_nn_vessel_mape", {})
    base = baseline_metrics.get("primary_nn_vessel_mape", {})
    if "2" not in cur or "3" not in cur:
        return False, "missing trial 2/3 primary metrics"
    if "2" not in base or "3" not in base:
        return False, "missing baseline trial metrics"
    t2_pp = _mape_pp(cur["2"])
    t3_pp = _mape_pp(cur["3"])
    b2_pp = _mape_pp(base["2"])
    b3_pp = _mape_pp(base["3"])
    assert t2_pp is not None and t3_pp is not None and b2_pp is not None and b3_pp is not None
    if t2_pp >= b2_pp - 1e-9:
        return False, f"trial2 MAPE {t2_pp:.2f}% not improved vs baseline {b2_pp:.2f}%"
    if t3_pp > b3_pp + CONTROL_GUARDRAIL_PP:
        return False, f"trial3 MAPE {t3_pp:.2f}% worsened >{CONTROL_GUARDRAIL_PP}pp vs {b3_pp:.2f}%"
    return True, f"trial2 {b2_pp:.2f}->{t2_pp:.2f}; trial3 {b3_pp:.2f}->{t3_pp:.2f}"


def aggregate_score(metrics: dict[str, Any]) -> float:
    """Lower is better: mean of trial-2 and trial-3 primary MAPE fractions."""
    p = metrics.get("primary_nn_vessel_mape", {})
    vals = [p[k] for k in ("2", "3") if k in p]
    if not vals:
        return float("inf")
    return float(sum(vals) / len(vals))


def write_summary_csv(campaign: Path, records: list[dict[str, Any]]) -> None:
    path = campaign / "summary.csv"
    fields = [
        "name",
        "phase",
        "exit_code",
        "trial2_mape_pct",
        "trial3_mape_pct",
        "mean_t2_t3_mape_pct",
        "trial2_mpe_pct",
        "trial3_mpe_pct",
        "kept",
        "rationale",
        "config",
        "artifact_dir",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def run_variant(
    *,
    campaign: Path,
    sandbox: Path,
    name: str,
    config_path: Path,
    trials: list[int] | None,
    num_trials: int,
    phase: str,
) -> dict[str, Any]:
    art = campaign / "artifacts" / name
    log = campaign / "logs" / f"{name}.log"
    clear_sandbox_qr_outputs(sandbox)
    print(f"\n=== [{phase}] {name} ===", flush=True)
    print(f"  config: {config_path}", flush=True)
    rc = run_cv(
        sandbox=sandbox,
        config_yaml=config_path,
        trials=trials,
        num_trials=num_trials,
        log_path=log,
    )
    archive_sandbox_outputs(sandbox, art)
    metrics = extract_metrics(art)
    primary = metrics.get("primary_nn_vessel_mape", {})
    if rc == 0 and trials is not None:
        missing = [str(t) for t in trials if str(t) not in primary]
        if missing:
            rc = 2
            print(
                f"  ERROR: missing primary MAPE for trials {missing} after CV "
                f"(likely stale/empty mse_comparison). Marking exit_code={rc}.",
                flush=True,
            )
    maxrel = {}
    for t, row in metrics.get("trials", {}).items():
        r = row.get("pressure_max_rel_error") or {}
        col = "PressureMaxRelError_BloodVesselJunction_NN_plus_Vessel_NN"
        if col in r and r[col] not in ("", None):
            try:
                maxrel[t] = float(r[col])
            except ValueError:
                pass
    rec = {
        "name": name,
        "phase": phase,
        "exit_code": rc,
        "trial2_mape_pct": None if "2" not in primary else 100 * primary["2"],
        "trial3_mape_pct": None if "3" not in primary else 100 * primary["3"],
        "mean_t2_t3_mape_pct": None
        if not {"2", "3"} <= set(primary)
        else 50 * (primary["2"] + primary["3"]),
        "trial2_mpe_pct": None if "2" not in maxrel else 100 * maxrel["2"],
        "trial3_mpe_pct": None if "3" not in maxrel else 100 * maxrel["3"],
        "kept": "",
        "rationale": "",
        "config": str(config_path),
        "artifact_dir": str(art),
        "metrics": metrics,
        "score": aggregate_score(metrics) if rc == 0 else float("inf"),
    }
    # Persist per-variant metrics json
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"  exit={rc} MAPE t2={rec['trial2_mape_pct']} t3={rec['trial3_mape_pct']} "
        f"score={rec['score']}",
        flush=True,
    )
    return rec


def merge_overlays(overlays: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ov in overlays:
        out = _deep_merge(out, ov)
    return out


def build_combined_config(
    shared_path: Path,
    overlays: list[dict[str, Any]],
    dest: Path,
) -> Path:
    data = _load_yaml(shared_path)
    data["training_run_configs"]["quadratic_resistor_gen_loss"] = merge_overlays(overlays)
    _write_yaml(dest, data)
    return dest


def write_report(
    campaign: Path,
    records: list[dict[str, Any]],
    retained: list[str],
    recommended_overlay: dict[str, Any],
) -> None:
    lines = [
        "# QR Training-Override Ablation Report",
        "",
        f"- Campaign: `{campaign.name}`",
        f"- Generated: `{_utc_now()}`",
        f"- Primary metric: NN+vessel pressure MAPE (`{PRIMARY_COL}`)",
        f"- Screen folds: trial 2 (`0155_0001`), trial 3 (`0082_0001`)",
        f"- Control guardrail: reject if trial-3 MAPE worsens by >{CONTROL_GUARDRAIL_PP} pp",
        "",
        "## No-op overlays (not screened)",
        "",
        "```json",
        json.dumps(NOOP_OVERRIDES, indent=2),
        "```",
        "",
        "## Screening results",
        "",
        "| name | t2 MAPE % | t3 MAPE % | mean | kept | rationale |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in records:
        if r["phase"] != "screen" and r["phase"] != "baseline":
            continue
        lines.append(
            f"| {r['name']} | {_fmt(r['trial2_mape_pct'])} | {_fmt(r['trial3_mape_pct'])} | "
            f"{_fmt(r['mean_t2_t3_mape_pct'])} | {r.get('kept','')} | {r.get('rationale','')} |"
        )
    lines += [
        "",
        "## Forward selection / reverse ablation",
        "",
        "| name | phase | t2 MAPE % | t3 MAPE % | kept | rationale |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in records:
        if r["phase"] in {"combine", "reverse", "confirm"}:
            lines.append(
                f"| {r['name']} | {r['phase']} | {_fmt(r['trial2_mape_pct'])} | "
                f"{_fmt(r['trial3_mape_pct'])} | {r.get('kept','')} | {r.get('rationale','')} |"
            )
    lines += [
        "",
        "## Retained special settings",
        "",
        "```yaml",
        yaml.safe_dump({"quadratic_resistor_gen_loss": recommended_overlay}, sort_keys=False),
        "```",
        "",
        f"Retained candidate IDs: {', '.join(retained) if retained else '(none)'}",
        "",
        "## Notes",
        "",
        "- `config/defaults.yaml` was not modified.",
        "- All variants used full YAML copies via `LEARN_LPNS_CONFIG`.",
        "- Outputs live under this campaign directory only.",
        "",
    ]
    (campaign / "REPORT.md").write_text("\n".join(lines))


def _fmt(x: Any) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return str(x)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign_dir", default=None, help="Existing campaign directory")
    p.add_argument(
        "--phase",
        choices=["all", "configs", "screen", "combine", "confirm", "report"],
        default="all",
    )
    p.add_argument("--skip_existing", action="store_true", help="Skip variants with metrics.json present")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = _campaign_dir_from_args(args)
    print(f"Campaign: {campaign}")

    configs = make_variant_configs(campaign)
    if args.phase == "configs":
        print(f"Wrote {len(configs)} configs")
        return 0

    sandbox = setup_sandbox(campaign)
    records: list[dict[str, Any]] = []
    records_path = campaign / "records.json"
    if records_path.is_file() and args.skip_existing:
        records = json.loads(records_path.read_text())

    def save_records() -> None:
        # metrics blobs are large; store slim records + separate metrics.json already written
        slim = []
        for r in records:
            s = {k: v for k, v in r.items() if k != "metrics"}
            slim.append(s)
        records_path.write_text(json.dumps(slim, indent=2))
        write_summary_csv(campaign, slim)

    def already_done(name: str) -> bool:
        if not args.skip_existing:
            return False
        mpath = campaign / "artifacts" / name / "metrics.json"
        if not mpath.is_file():
            return False
        try:
            payload = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            return False
        primary = payload.get("primary_nn_vessel_mape") or {}
        # Screen/combine: trials 2+3. Confirm five-fold: trials 0..4.
        if name.startswith("confirm_"):
            return all(str(t) in primary for t in range(5))
        return "2" in primary and "3" in primary

    # --- baseline + screen ---
    if args.phase in {"all", "screen"}:
        # Shared defaults baseline (trials 2+3)
        if not already_done("00_shared_defaults"):
            rec = run_variant(
                campaign=campaign,
                sandbox=sandbox,
                name="00_shared_defaults",
                config_path=configs["00_shared_defaults"],
                trials=list(SCREEN_TRIALS),
                num_trials=5,
                phase="baseline",
            )
            rec["kept"] = "baseline"
            rec["rationale"] = "shared-default QR training baseline"
            records = [r for r in records if r["name"] != "00_shared_defaults"] + [rec]
            save_records()
        else:
            metrics = json.loads((campaign / "artifacts" / "00_shared_defaults" / "metrics.json").read_text())
            primary = metrics.get("primary_nn_vessel_mape", {})
            records = [r for r in records if r["name"] != "00_shared_defaults"] + [
                {
                    "name": "00_shared_defaults",
                    "phase": "baseline",
                    "exit_code": 0,
                    "trial2_mape_pct": None if "2" not in primary else 100 * primary["2"],
                    "trial3_mape_pct": None if "3" not in primary else 100 * primary["3"],
                    "mean_t2_t3_mape_pct": None
                    if not {"2", "3"} <= set(primary)
                    else 50 * (primary["2"] + primary["3"]),
                    "trial2_mpe_pct": "",
                    "trial3_mpe_pct": "",
                    "kept": "baseline",
                    "rationale": "shared-default QR training baseline (cached)",
                    "config": str(configs["00_shared_defaults"]),
                    "artifact_dir": str(campaign / "artifacts" / "00_shared_defaults"),
                    "metrics": metrics,
                    "score": aggregate_score(metrics),
                }
            ]

        baseline = next(r for r in records if r["name"] == "00_shared_defaults")

        for name in CANDIDATES:
            if already_done(name):
                metrics = json.loads((campaign / "artifacts" / name / "metrics.json").read_text())
                rec = {
                    "name": name,
                    "phase": "screen",
                    "exit_code": 0,
                    "trial2_mape_pct": _mape_pp(metrics.get("primary_nn_vessel_mape", {}).get("2")),
                    "trial3_mape_pct": _mape_pp(metrics.get("primary_nn_vessel_mape", {}).get("3")),
                    "mean_t2_t3_mape_pct": None,
                    "trial2_mpe_pct": "",
                    "trial3_mpe_pct": "",
                    "kept": "",
                    "rationale": "cached",
                    "config": str(configs[name]),
                    "artifact_dir": str(campaign / "artifacts" / name),
                    "metrics": metrics,
                    "score": aggregate_score(metrics),
                }
                if rec["trial2_mape_pct"] is not None and rec["trial3_mape_pct"] is not None:
                    rec["mean_t2_t3_mape_pct"] = 0.5 * (rec["trial2_mape_pct"] + rec["trial3_mape_pct"])
            else:
                rec = run_variant(
                    campaign=campaign,
                    sandbox=sandbox,
                    name=name,
                    config_path=configs[name],
                    trials=list(SCREEN_TRIALS),
                    num_trials=5,
                    phase="screen",
                )
            ok, why = passes_guards(rec["metrics"], baseline["metrics"])
            rec["kept"] = "yes" if ok and rec["exit_code"] == 0 else "no"
            rec["rationale"] = why if rec["exit_code"] == 0 else f"cv failed exit={rec['exit_code']}"
            records = [r for r in records if r["name"] != name] + [rec]
            save_records()

    def screen_complete() -> bool:
        needed = ["00_shared_defaults", *CANDIDATES.keys()]
        return all(already_done(n) for n in needed)

    # --- combine (forward selection) ---
    retained: list[str] = []
    recommended_overlay: dict[str, Any] = {}
    if args.phase in {"all", "combine"}:
        if not screen_complete():
            missing = [n for n in ["00_shared_defaults", *CANDIDATES.keys()] if not already_done(n)]
            print(
                "Screening incomplete; refusing combine/confirm until trials 2+3 MAPE exist for: "
                + ", ".join(missing),
                flush=True,
            )
            save_records()
            return 2
        baseline = next(r for r in records if r["name"] == "00_shared_defaults")
        screen = [r for r in records if r["phase"] == "screen" and r.get("kept") == "yes"]
        screen.sort(key=lambda r: r["score"])
        current_overlays: list[dict[str, Any]] = []
        current_metrics = baseline["metrics"]
        current_names: list[str] = []

        for cand in screen:
            trial_overlays = current_overlays + [CANDIDATES[cand["name"]]]
            combo_name = "combo_" + "_plus_".join(current_names + [cand["name"].split("_", 1)[0]])
            # Shorter stable name
            combo_name = f"combo_{len(current_names)+1:02d}_add_{cand['name']}"
            dest = campaign / "configs" / f"{combo_name}.yaml"
            build_combined_config(configs["00_shared_defaults"], trial_overlays, dest)
            mirror = REPO_ROOT / "config" / "experiments" / campaign.name / f"{combo_name}.yaml"
            shutil.copy2(dest, mirror)

            if already_done(combo_name):
                metrics = json.loads((campaign / "artifacts" / combo_name / "metrics.json").read_text())
                rec = {
                    "name": combo_name,
                    "phase": "combine",
                    "exit_code": 0,
                    "trial2_mape_pct": _mape_pp(metrics.get("primary_nn_vessel_mape", {}).get("2")),
                    "trial3_mape_pct": _mape_pp(metrics.get("primary_nn_vessel_mape", {}).get("3")),
                    "mean_t2_t3_mape_pct": None,
                    "trial2_mpe_pct": "",
                    "trial3_mpe_pct": "",
                    "kept": "",
                    "rationale": "cached",
                    "config": str(dest),
                    "artifact_dir": str(campaign / "artifacts" / combo_name),
                    "metrics": metrics,
                    "score": aggregate_score(metrics),
                }
                if rec["trial2_mape_pct"] is not None and rec["trial3_mape_pct"] is not None:
                    rec["mean_t2_t3_mape_pct"] = 0.5 * (rec["trial2_mape_pct"] + rec["trial3_mape_pct"])
            else:
                rec = run_variant(
                    campaign=campaign,
                    sandbox=sandbox,
                    name=combo_name,
                    config_path=dest,
                    trials=list(SCREEN_TRIALS),
                    num_trials=5,
                    phase="combine",
                )
            # Keep if improves aggregate vs current and passes guards vs shared baseline
            ok_base, why_base = passes_guards(rec["metrics"], baseline["metrics"])
            improved = rec["score"] < aggregate_score(current_metrics) - 1e-12
            if rec["exit_code"] == 0 and ok_base and improved:
                rec["kept"] = "yes"
                rec["rationale"] = f"accepted: {why_base}; improved aggregate vs prior combo"
                current_overlays = trial_overlays
                current_metrics = rec["metrics"]
                current_names.append(cand["name"])
                retained.append(cand["name"])
            else:
                rec["kept"] = "no"
                if rec["exit_code"] != 0:
                    rec["rationale"] = f"rejected: cv failed exit={rec['exit_code']}"
                elif not ok_base:
                    rec["rationale"] = f"rejected vs baseline: {why_base}"
                else:
                    rec["rationale"] = (
                        f"rejected: aggregate score {rec['score']:.5f} "
                        f"not better than current {aggregate_score(current_metrics):.5f}"
                    )
            records = [r for r in records if r["name"] != combo_name] + [rec]
            save_records()

        recommended_overlay = merge_overlays([CANDIDATES[n] for n in retained]) if retained else {}
        # If no combination accepted but some singles kept, take the single best
        if not retained and screen:
            best = screen[0]
            retained = [best["name"]]
            recommended_overlay = deepcopy(CANDIDATES[best["name"]])

        # Reverse ablation
        if len(retained) >= 2:
            for drop in list(retained):
                keep = [n for n in retained if n != drop]
                name = f"reverse_drop_{drop}"
                dest = campaign / "configs" / f"{name}.yaml"
                build_combined_config(
                    configs["00_shared_defaults"],
                    [CANDIDATES[n] for n in keep],
                    dest,
                )
                shutil.copy2(dest, REPO_ROOT / "config" / "experiments" / campaign.name / f"{name}.yaml")
                if already_done(name):
                    metrics = json.loads((campaign / "artifacts" / name / "metrics.json").read_text())
                    rec = {
                        "name": name,
                        "phase": "reverse",
                        "exit_code": 0,
                        "trial2_mape_pct": _mape_pp(metrics.get("primary_nn_vessel_mape", {}).get("2")),
                        "trial3_mape_pct": _mape_pp(metrics.get("primary_nn_vessel_mape", {}).get("3")),
                        "mean_t2_t3_mape_pct": None,
                        "trial2_mpe_pct": "",
                        "trial3_mpe_pct": "",
                        "kept": "",
                        "rationale": "cached",
                        "config": str(dest),
                        "artifact_dir": str(campaign / "artifacts" / name),
                        "metrics": metrics,
                        "score": aggregate_score(metrics),
                    }
                    if rec["trial2_mape_pct"] is not None and rec["trial3_mape_pct"] is not None:
                        rec["mean_t2_t3_mape_pct"] = 0.5 * (rec["trial2_mape_pct"] + rec["trial3_mape_pct"])
                else:
                    rec = run_variant(
                        campaign=campaign,
                        sandbox=sandbox,
                        name=name,
                        config_path=dest,
                        trials=list(SCREEN_TRIALS),
                        num_trials=5,
                        phase="reverse",
                    )
                full_score = aggregate_score(current_metrics)
                # If dropping does not hurt (score <= full + tiny eps), drop is unnecessary
                if rec["exit_code"] == 0 and rec["score"] <= full_score + 1e-6:
                    rec["kept"] = "drop_ok"
                    rec["rationale"] = (
                        f"dropping {drop} did not hurt (score {rec['score']:.5f} vs full {full_score:.5f}); "
                        "remove from retained"
                    )
                    retained = keep
                    recommended_overlay = merge_overlays([CANDIDATES[n] for n in retained]) if retained else {}
                    current_metrics = rec["metrics"]
                else:
                    rec["kept"] = "necessary"
                    rec["rationale"] = (
                        f"{drop} necessary; drop score {rec['score']:.5f} worse than full {full_score:.5f}"
                    )
                records = [r for r in records if r["name"] != name] + [rec]
                save_records()

        (campaign / "recommended_overlay.yaml").write_text(
            yaml.safe_dump(
                {"training_run_configs": {"quadratic_resistor_gen_loss": recommended_overlay}},
                sort_keys=False,
            )
        )
        (campaign / "retained_candidates.json").write_text(json.dumps(retained, indent=2))

    # --- confirm five-fold ---
    if args.phase in {"all", "confirm"}:
        if args.phase == "all" and not screen_complete():
            print("Skipping confirm because screening is incomplete.", flush=True)
            return 2
        # Always reload retained/overlay when confirm runs standalone (or after combine
        # in a prior process). Local `recommended_overlay` may still be {} here.
        if (campaign / "retained_candidates.json").is_file() and not retained:
            retained = json.loads((campaign / "retained_candidates.json").read_text())
        overlay_path = campaign / "recommended_overlay.yaml"
        if overlay_path.is_file():
            loaded = yaml.safe_load(overlay_path.read_text()) or {}
            recommended_overlay = (
                loaded.get("training_run_configs", {}).get("quadratic_resistor_gen_loss") or {}
            )
        elif retained:
            recommended_overlay = merge_overlays([CANDIDATES[n] for n in retained])

        confirm_specs = [
            ("confirm_shared_defaults", configs["00_shared_defaults"]),
            ("confirm_current_qr_profile", configs["current_qr_profile"]),
        ]
        dest = campaign / "configs" / "confirm_recommended_minimal.yaml"
        build_combined_config(configs["00_shared_defaults"], [recommended_overlay], dest)
        shutil.copy2(
            dest,
            REPO_ROOT / "config" / "experiments" / campaign.name / "confirm_recommended_minimal.yaml",
        )
        confirm_specs.append(("confirm_recommended_minimal", dest))
        # If the recommended overlay changed since a prior confirm, force a redo.
        prev_overlay = campaign / "artifacts" / "confirm_recommended_minimal" / "resolved_overlay.yaml"
        new_overlay_text = yaml.safe_dump(
            {"training_run_configs": {"quadratic_resistor_gen_loss": recommended_overlay}},
            sort_keys=True,
        )
        if prev_overlay.is_file() and prev_overlay.read_text() != new_overlay_text:
            stale = campaign / "artifacts" / "confirm_recommended_minimal" / "metrics.json"
            if stale.is_file():
                stale.unlink()
                print("Invalidated confirm_recommended_minimal (overlay changed).", flush=True)
        (campaign / "artifacts" / "confirm_recommended_minimal").mkdir(parents=True, exist_ok=True)
        prev_overlay.write_text(new_overlay_text)

        for name, cfg in confirm_specs:
            if already_done(name):
                metrics = json.loads((campaign / "artifacts" / name / "metrics.json").read_text())
                primary = metrics.get("primary_nn_vessel_mape", {})
                rec = {
                    "name": name,
                    "phase": "confirm",
                    "exit_code": 0,
                    "trial2_mape_pct": _mape_pp(primary.get("2")),
                    "trial3_mape_pct": _mape_pp(primary.get("3")),
                    "mean_t2_t3_mape_pct": _mape_pp(primary.get("mean")),
                    "trial2_mpe_pct": "",
                    "trial3_mpe_pct": "",
                    "kept": "confirm",
                    "rationale": "five-fold confirmation (cached)",
                    "config": str(cfg),
                    "artifact_dir": str(campaign / "artifacts" / name),
                    "metrics": metrics,
                    "score": aggregate_score(metrics),
                }
            else:
                rec = run_variant(
                    campaign=campaign,
                    sandbox=sandbox,
                    name=name,
                    config_path=cfg,
                    trials=None,
                    num_trials=5,
                    phase="confirm",
                )
                # For five-fold, also record mean if present
                mean = rec["metrics"].get("primary_nn_vessel_mape", {}).get("mean")
                if mean is not None:
                    rec["mean_t2_t3_mape_pct"] = 100 * mean
                rec["kept"] = "confirm"
                rec["rationale"] = "five-fold confirmation"
            records = [r for r in records if r["name"] != name] + [rec]
            save_records()

            # Optional barcharts inside artifact tree
            try:
                py = sandbox / ".venv" / "bin" / "python"
                env = os.environ.copy()
                env["LEARN_LPNS_CONFIG"] = str(Path(cfg).resolve())
                env["MPLBACKEND"] = "Agg"
                art_cv = campaign / "artifacts" / name / "results"
                # Run barchart against archived results by temporarily pointing data_root
                # Module expects results/cross_validation under --data_root
                if art_cv.is_dir():
                    subprocess.run(
                        [
                            str(py),
                            "-m",
                            "learn_lpns.visualizations.cv_pressure_max_pct_error_barchart",
                            SET_NAME,
                            GEOMETRY,
                            "--run_config",
                            RUN_CONFIG,
                            "--data_root",
                            str(art_cv),
                            "--metric",
                            "pressure_mean_rel_error",
                        ],
                        cwd=sandbox,
                        env=env,
                        check=False,
                    )
            except Exception as exc:
                print(f"  barchart skipped: {exc}", flush=True)

    if args.phase in {"all", "combine", "confirm", "report"}:
        if (campaign / "retained_candidates.json").is_file():
            retained = json.loads((campaign / "retained_candidates.json").read_text())
        if (campaign / "recommended_overlay.yaml").is_file():
            rec_yaml = _load_yaml(campaign / "recommended_overlay.yaml")
            recommended_overlay = (
                rec_yaml.get("training_run_configs", {}).get("quadratic_resistor_gen_loss", {}) or {}
            )
        write_report(campaign, records, retained, recommended_overlay)
        save_records()
        print(f"\nDone. Report: {campaign / 'REPORT.md'}")
        print(f"Recommended overlay: {campaign / 'recommended_overlay.yaml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
