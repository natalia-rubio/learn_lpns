"""
Drop calibration timesteps where 1D-extracted flows violate mass balance.

Each vessel in/out pair and each junction's sum(in) vs sum(out) must satisfy **either**
a relative bound (default 1% of max(|a|,|b|)) **or** an absolute difference bound
(default 0.1 in the same flow units as ``observations['y']``).

For each vessel: flow in must match flow out (same keys as ``extract_observations_from_1d*``).

For each junction: sum of inlet flows must match sum of outlet flows.

When ``verbose`` is True, each dropped timestep prints the first failing check
(vessel in/out pair or junction mass imbalance).

Strict validation: all observation series must share one length; every vessel must
resolve in/out flow keys; every junction inlet/outlet flow key implied by the
geometric graph must exist in ``observations['y']``. Otherwise :class:`ValueError`.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from util.zerod_calibration.bifurcation_splitting import junction_uses_block_connectivity

GeometricInput = Union[Dict[str, Any], str]

DEFAULT_RELATIVE_TOL = 0.01
DEFAULT_ABSOLUTE_TOL = 0.1


def _flow_balance_passes(
    a: float, b: float, *, relative_tol: float, absolute_tol: float
) -> bool:
    """True if |a-b| is within ``relative_tol * max(|a|,|b|,1e-12)`` OR |a-b| <= ``absolute_tol``."""
    diff = abs(a - b)
    denom = max(abs(a), abs(b), 1e-12)
    return diff <= relative_tol * denom or diff <= absolute_tol


def _load_geo(geometric_input: GeometricInput) -> Dict[str, Any]:
    if isinstance(geometric_input, dict):
        return geometric_input
    path = str(geometric_input)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Geometric input not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _series_len(series: Any) -> int:
    if isinstance(series, (list, tuple)):
        return len(series)
    arr = np.asarray(series)
    return int(arr.shape[0])


def _assert_aligned_lengths(observations: Dict[str, Any], label: str) -> int:
    """Return common length T or raise ValueError."""
    y = observations.get("y") or {}
    lengths: Dict[str, int] = {}
    for k, series in y.items():
        lengths[f"y[{k!r}]"] = _series_len(series)
    dy = observations.get("dy") or {}
    for k, series in dy.items():
        lengths[f"dy[{k!r}]"] = _series_len(series)
    if not lengths:
        raise ValueError(f"{label}: observations have no 'y' series.")
    vals = list(lengths.values())
    T = vals[0]
    for name, n in lengths.items():
        if n != T:
            raise ValueError(
                f"{label}: observation length mismatch: {name} has length {n}, "
                f"expected {T} (all y and dy series must align)."
            )
    return T


def _resolve_vessel_inflow_key(
    vessel: Dict[str, Any],
    junctions: Sequence[Dict[str, Any]],
    y: Dict[str, Any],
) -> Optional[str]:
    vname = vessel.get("vessel_name")
    if not vname:
        return None
    bc = vessel.get("boundary_conditions") or {}
    if "inlet" in bc:
        k = f"flow:INFLOW:{vname}"
        if k in y:
            return k
        if vname == "branch0_seg0":
            legacy = "flow:INFLOW:branch0_seg0"
            if legacy in y:
                return legacy
        return None
    for j in junctions:
        jn = j.get("junction_name")
        if not jn:
            continue
        k = f"flow:{jn}:{vname}"
        if k in y:
            return k
    return None


def _resolve_vessel_outflow_key(
    vessel: Dict[str, Any],
    junctions: Sequence[Dict[str, Any]],
    y: Dict[str, Any],
) -> Optional[str]:
    vname = vessel.get("vessel_name")
    if not vname:
        return None
    bc = vessel.get("boundary_conditions") or {}
    out = bc.get("outlet")
    if out:
        k = f"flow:{vname}:{out}"
        if k in y:
            return k
        return None
    for j in junctions:
        jn = j.get("junction_name")
        if not jn:
            continue
        k = f"flow:{vname}:{jn}"
        if k in y:
            return k
    return None


def _build_mass_checks(
    geometric_input: Dict[str, Any],
    y: Dict[str, Any],
    *,
    label: str,
) -> Tuple[
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, List[str], List[str]]],
]:
    """
    Returns:
        vessel_specs: (kind, vessel_name, in_key, out_key)
        junction_specs: (kind, junction_name, in_keys, out_keys)
    """
    vessels = list(geometric_input.get("vessels") or [])
    junctions = list(geometric_input.get("junctions") or [])

    vessel_specs: List[Tuple[str, str, str, str]] = []
    for v in vessels:
        vname = v.get("vessel_name")
        if not vname:
            raise ValueError(f"{label}: vessel entry missing vessel_name.")
        ik = _resolve_vessel_inflow_key(v, junctions, y)
        ok = _resolve_vessel_outflow_key(v, junctions, y)
        if ik is None or ok is None:
            raise ValueError(
                f"{label}: vessel {vname!r} missing flow observation key(s) "
                f"(in_key={ik!r}, out_key={ok!r}); every vessel must have resolvable "
                f"INFLOW/junction-in and outlet/junction-out keys in observations['y']."
            )
        if ik not in y or ok not in y:
            raise ValueError(
                f"{label}: vessel {vname!r} resolved keys not in y: in={ik!r}, out={ok!r}."
            )
        vessel_specs.append(("vessel", str(vname), ik, ok))

    junction_specs: List[Tuple[str, str, List[str], List[str]]] = []
    for j in junctions:
        jn = j.get("junction_name")
        if not jn:
            continue
        if junction_uses_block_connectivity(j):
            ibs = list(j.get("inlet_blocks") or [])
            obs = list(j.get("outlet_blocks") or [])
            if not ibs or not obs:
                raise ValueError(
                    f"{label}: junction {jn!r} uses block connectivity but has "
                    f"inlet_blocks={ibs!r}, outlet_blocks={obs!r} (need both non-empty)."
                )
            ins = [f"flow:{ib}:{jn}" for ib in ibs]
            outs = [f"flow:{jn}:{ob}" for ob in obs]
        else:
            iv = list(j.get("inlet_vessels") or [])
            ov = list(j.get("outlet_vessels") or [])
            if not iv or not ov:
                # No mass-balance check for junctions without both sides (e.g. unused schema)
                continue
            ins = []
            for vid in iv:
                if not isinstance(vid, int) or not (0 <= vid < len(vessels)):
                    raise ValueError(
                        f"{label}: junction {jn!r} has invalid inlet_vessel id {vid!r}."
                    )
                vn = vessels[vid].get("vessel_name")
                if not vn:
                    raise ValueError(
                        f"{label}: junction {jn!r} inlet vessel id {vid} has no vessel_name."
                    )
                ins.append(f"flow:{vn}:{jn}")
            outs = []
            for vid in ov:
                if not isinstance(vid, int) or not (0 <= vid < len(vessels)):
                    raise ValueError(
                        f"{label}: junction {jn!r} has invalid outlet_vessel id {vid!r}."
                    )
                vn = vessels[vid].get("vessel_name")
                if not vn:
                    raise ValueError(
                        f"{label}: junction {jn!r} outlet vessel id {vid} has no vessel_name."
                    )
                outs.append(f"flow:{jn}:{vn}")
        for k in ins + outs:
            if k not in y:
                raise ValueError(
                    f"{label}: junction {jn!r} requires flow observation {k!r} in observations['y']."
                )
        junction_specs.append(("junction", str(jn), ins, outs))

    return vessel_specs, junction_specs


def flow_mass_balance_keep_mask(
    observations: Dict[str, Any],
    geometric_input: GeometricInput,
    *,
    relative_tol: float = DEFAULT_RELATIVE_TOL,
    absolute_tol: float = DEFAULT_ABSOLUTE_TOL,
    min_timesteps: int = 1,
    verbose: bool = True,
    label: str = "Flow mass-balance filter",
) -> np.ndarray:
    """
    Return a boolean keep-mask for timesteps passing flow mass-balance checks.

    A pair of flows ``(a, b)`` passes if ``|a-b| <= relative_tol * max(|a|,|b|,1e-12)``
    **or** ``|a-b| <= absolute_tol``.

    Raises:
        ValueError: length mismatch across y/dy or missing vessel/junction keys.

    If fewer than ``min_timesteps`` would remain after filtering, logs a warning and
    returns an all-True mask (does not raise).

    With ``verbose=True``, logs one line per dropped timestep naming the vessel or
    junction check that failed first at that index.
    """
    geo = _load_geo(geometric_input)
    y = observations.get("y") or {}
    if not y:
        raise ValueError(f"{label}: observations['y'] is empty.")

    T = _assert_aligned_lengths(observations, label)

    vessel_specs, junction_specs = _build_mass_checks(geo, y, label=label)

    keep = np.ones(T, dtype=bool)
    for t in range(T):
        fail_detail = None
        for _kind, vname, ik, ok in vessel_specs:
            qi = float(y[ik][t])
            qo = float(y[ok][t])
            if not _flow_balance_passes(
                qi, qo, relative_tol=relative_tol, absolute_tol=absolute_tol
            ):
                fail_detail = (
                    f"vessel {vname!r}: flow_in {qi:.6g} ({ik}) vs flow_out {qo:.6g} ({ok}) "
                    f"(|Δ|={abs(qi - qo):.6g}; need rel≤{relative_tol:g}·scale or |Δ|≤{absolute_tol:g})"
                )
                break
        if fail_detail is None:
            for _kind, jn, ins, outs in junction_specs:
                s_in = sum(float(y[k][t]) for k in ins)
                s_out = sum(float(y[k][t]) for k in outs)
                if not _flow_balance_passes(
                    s_in, s_out, relative_tol=relative_tol, absolute_tol=absolute_tol
                ):
                    fail_detail = (
                        f"junction {jn!r}: sum(in)={s_in:.6g} vs sum(out)={s_out:.6g} "
                        f"(|Δ|={abs(s_in - s_out):.6g}; need rel≤{relative_tol:g}·scale or |Δ|≤{absolute_tol:g}; "
                        f"{len(ins)} inlet(s), {len(outs)} outlet(s))"
                    )
                    break
        if fail_detail is not None:
            keep[t] = False
            if verbose:
                print(f"  {label}: drop timestep index {t}: {fail_detail}")

    n_keep = int(np.sum(keep))
    if n_keep < min_timesteps:
        if verbose:
            print(
                f"  Warning: {label} would keep only {n_keep} timestep(s) "
                f"(min {min_timesteps}); keeping all {T} timesteps unchanged."
            )
        return np.ones(T, dtype=bool)

    if n_keep < T and verbose:
        print(
            f"  {label}: kept {n_keep}/{T} timesteps "
            f"(pass if |Δ|≤{relative_tol:.4g}·max(|a|,|b|,1e-12) or |Δ|≤{absolute_tol:g}; "
            f"{len(vessel_specs)} vessel pair(s), {len(junction_specs)} junction(s))."
        )

    return keep


def apply_nan_mask_to_observations(
    observations: Dict[str, Any],
    keep_mask: Sequence[bool],
) -> Dict[str, Any]:
    """
    Return deep copy with failing timesteps set to NaN in both y and dy.
    """
    out = copy.deepcopy(observations)
    T = _assert_aligned_lengths(out, "NaN mask observations")
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.shape[0] != T:
        raise ValueError(
            f"NaN mask length mismatch: keep_mask has length {mask.shape[0]}, expected {T}."
        )
    for bucket in ("y", "dy"):
        d = out.get(bucket)
        if not isinstance(d, dict):
            continue
        for k, series in list(d.items()):
            arr = np.asarray(series, dtype=float).copy()
            arr[~mask] = np.nan
            out[bucket][k] = arr.tolist()
    return out


def filter_observations_by_flow_mass_balance(
    observations: Dict[str, Any],
    geometric_input: GeometricInput,
    *,
    relative_tol: float = DEFAULT_RELATIVE_TOL,
    absolute_tol: float = DEFAULT_ABSOLUTE_TOL,
    min_timesteps: int = 1,
    verbose: bool = True,
    label: str = "Flow mass-balance filter",
) -> Dict[str, Any]:
    """
    Legacy helper: deep copy with y/dy sliced to kept timesteps.
    """
    keep = flow_mass_balance_keep_mask(
        observations,
        geometric_input,
        relative_tol=relative_tol,
        absolute_tol=absolute_tol,
        min_timesteps=min_timesteps,
        verbose=verbose,
        label=label,
    )
    T = _assert_aligned_lengths(observations, label)
    idx = np.flatnonzero(keep)
    out = copy.deepcopy(observations)
    for bucket in ("y", "dy"):
        d = out.get(bucket)
        if not isinstance(d, dict):
            continue
        for k, series in list(d.items()):
            arr = np.asarray(series)
            if arr.shape[0] == T:
                out[bucket][k] = arr[idx].tolist()
    return out
