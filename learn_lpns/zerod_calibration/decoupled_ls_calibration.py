"""Decoupled least-squares calibration of R/L/(S) from local P/Q observations."""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from learn_lpns.tools.paths import repo_root

VESSEL_ELEMENT_TYPES = frozenset({"BloodVessel", "BloodVesselCRL", "BloodVesselFC"})
FIT_QUALITY_THRESHOLD = 0.10
Q_MIN_MASK = 1e-12
RMS_DELTA_P_MIN = 1e-12
RSL_FITS_RESULTS_SUBDIR = Path("results") / "RSL_fits"


def _is_connector_vessel(vessel_name: str) -> bool:
    """Match svzerodcalibrator connector freeze rule (exclude connectorEL)."""
    return "connector" in vessel_name and "connectorEL" not in vessel_name


def _get_series(
    y: dict[str, Any],
    dy: dict[str, Any],
    *keys: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    for key in keys:
        if key in y and key in dy:
            return np.asarray(y[key], dtype=float), np.asarray(dy[key], dtype=float)
    return None


def _build_vessel_id_map(vessels: list[dict]) -> dict[int, dict]:
    return {int(v["vessel_id"]): v for v in vessels}


def _junction_at_vessel_inlet(
    vessel_id: int,
    vessel_name: str,
    junctions: list[dict],
) -> str | None:
    """Junction upstream of this vessel's inlet (vessel is an outlet branch of J)."""
    for junction in junctions:
        if vessel_id in junction.get("outlet_vessels", []):
            return junction["junction_name"]
        if vessel_name in junction.get("outlet_blocks", []):
            return junction["junction_name"]
    return None


def _junction_at_vessel_outlet(
    vessel_id: int,
    vessel_name: str,
    junctions: list[dict],
) -> str | None:
    """Junction downstream of this vessel's outlet (vessel feeds J as an inlet branch)."""
    for junction in junctions:
        if vessel_id in junction.get("inlet_vessels", []):
            return junction["junction_name"]
        if vessel_name in junction.get("inlet_blocks", []):
            return junction["junction_name"]
    return None


def _resolve_vessel_observations(
    config: dict,
    vessel: dict,
    junctions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (delta_p, q_in, dq_out, dq_out_full) or None if observations are missing."""
    y = config["y"]
    dy = config["dy"]
    vessel_name = vessel["vessel_name"]
    vessel_id = int(vessel["vessel_id"])
    bc = vessel.get("boundary_conditions", {})

    inlet_bc = bc.get("inlet")
    outlet_bc = bc.get("outlet")
    inlet_junc = _junction_at_vessel_inlet(vessel_id, vessel_name, junctions)
    outlet_junc = _junction_at_vessel_outlet(vessel_id, vessel_name, junctions)

    if inlet_bc == "INFLOW":
        inlet_p = _get_series(
            y,
            dy,
            f"pressure:INFLOW:{vessel_name}",
            "pressure:INFLOW:branch0_seg0",
        )
        inlet_q = _get_series(
            y,
            dy,
            f"flow:INFLOW:{vessel_name}",
            "flow:INFLOW:branch0_seg0",
        )
    elif inlet_junc is not None:
        inlet_p = _get_series(y, dy, f"pressure:{inlet_junc}:{vessel_name}")
        inlet_q = _get_series(y, dy, f"flow:{inlet_junc}:{vessel_name}")
    else:
        return None

    if outlet_bc is not None:
        outlet_p = _get_series(y, dy, f"pressure:{vessel_name}:{outlet_bc}")
        outlet_q = _get_series(y, dy, f"flow:{vessel_name}:{outlet_bc}")
    elif outlet_junc is not None:
        outlet_p = _get_series(y, dy, f"pressure:{vessel_name}:{outlet_junc}")
        outlet_q = _get_series(y, dy, f"flow:{vessel_name}:{outlet_junc}")
    else:
        return None

    if inlet_p is None or inlet_q is None or outlet_p is None or outlet_q is None:
        return None

    p_in, _ = inlet_p
    q_in, _ = inlet_q
    p_out, _ = outlet_p
    _, dq_out = outlet_q

    delta_p = p_in - p_out
    return delta_p, q_in, dq_out, dq_out


def _resolve_junction_outlet_observations(
    config: dict,
    junction: dict,
    outlet_index: int,
    outlet_name: str,
    vessel_id_map: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (delta_p, q_in, dq_out) for one junction outlet leg."""
    y = config["y"]
    dy = config["dy"]
    junc_name = junction["junction_name"]

    inlet_pressures: list[np.ndarray] = []
    if junction.get("inlet_vessels"):
        for inlet_id in junction["inlet_vessels"]:
            inlet_vessel = vessel_id_map[int(inlet_id)]
            series = _get_series(
                y,
                dy,
                f"pressure:{inlet_vessel['vessel_name']}:{junc_name}",
            )
            if series is not None:
                inlet_pressures.append(series[0])
    elif junction.get("inlet_blocks"):
        for block_name in junction["inlet_blocks"]:
            series = _get_series(y, dy, f"pressure:{block_name}:{junc_name}")
            if series is not None:
                inlet_pressures.append(series[0])

    outlet_p = _get_series(y, dy, f"pressure:{junc_name}:{outlet_name}")
    outlet_q = _get_series(y, dy, f"flow:{junc_name}:{outlet_name}")
    if not inlet_pressures or outlet_p is None or outlet_q is None:
        return None

    p_upstream = np.mean(np.column_stack(inlet_pressures), axis=1)
    p_out, _ = outlet_p
    q_in, dq_out = outlet_q
    delta_p = p_upstream - p_out
    return delta_p, q_in, dq_out


def _predict_delta_p(
    q_in: np.ndarray,
    dq_out: np.ndarray,
    r_poiseuille: float,
    stenosis: float,
    inductance: float,
) -> np.ndarray:
    q_in = np.asarray(q_in, dtype=float)
    dq_out = np.asarray(dq_out, dtype=float)
    return r_poiseuille * q_in + stenosis * np.abs(q_in) * q_in + inductance * dq_out


def _relative_fit_error(
    delta_p: np.ndarray,
    q_in: np.ndarray,
    dq_out: np.ndarray,
    r_poiseuille: float,
    stenosis: float,
    inductance: float,
) -> float:
    delta_p = np.asarray(delta_p, dtype=float)
    q_in = np.asarray(q_in, dtype=float)
    dq_out = np.asarray(dq_out, dtype=float)
    predicted = _predict_delta_p(q_in, dq_out, r_poiseuille, stenosis, inductance)
    residual = delta_p - predicted
    rms_delta_p = float(np.sqrt(np.mean(delta_p**2)))
    if rms_delta_p < RMS_DELTA_P_MIN:
        return 0.0
    rms_residual = float(np.sqrt(np.mean(residual**2)))
    return rms_residual / rms_delta_p


def _fit_rlc(
    delta_p: np.ndarray,
    q_in: np.ndarray,
    dq_out: np.ndarray,
    *,
    fit_stenosis: bool,
    l2_r: float = 0.0,
    l2_stenosis: float = 0.0,
    l2_l: float = 0.0,
) -> tuple[float, float, float, float]:
    """Fit R, S, L from local pressure drop; R >= 0, L >= 0; optional L2 toward 0."""
    delta_p = np.asarray(delta_p, dtype=float)
    q_in = np.asarray(q_in, dtype=float)
    dq_out = np.asarray(dq_out, dtype=float)

    mask = np.abs(q_in) >= Q_MIN_MASK
    if not np.any(mask):
        mask = np.ones_like(q_in, dtype=bool)

    q = q_in[mask]
    dq = dq_out[mask]
    dp = delta_p[mask]

    if fit_stenosis:
        design = np.column_stack([q, np.abs(q) * q, dq])
        lower = np.array([0.0, -np.inf, 0.0])
        upper = np.array([np.inf, np.inf, np.inf])
        l2_weights = np.array([l2_r, l2_stenosis, l2_l], dtype=float)
    else:
        design = np.column_stack([q, dq])
        lower = np.array([0.0, 0.0])
        upper = np.array([np.inf, np.inf])
        l2_weights = np.array([l2_r, l2_l], dtype=float)

    if np.any(l2_weights > 0.0):
        reg_rows = np.diag(np.sqrt(l2_weights))
        design = np.vstack([design, reg_rows])
        dp = np.concatenate([dp, np.zeros(reg_rows.shape[0], dtype=float)])

    coeffs = lsq_linear(design, dp, bounds=(lower, upper)).x

    if fit_stenosis:
        r_poiseuille, stenosis, inductance = (float(c) for c in coeffs)
    else:
        r_poiseuille, inductance = (float(c) for c in coeffs)
        stenosis = 0.0

    rel_err = _relative_fit_error(delta_p, q_in, dq_out, r_poiseuille, stenosis, inductance)
    return r_poiseuille, stenosis, inductance, rel_err


def _warn_if_poor_fit(element_name: str, rel_err: float) -> None:
    if rel_err > FIT_QUALITY_THRESHOLD:
        warnings.warn(
            f"Decoupled LS fit for {element_name}: relative RMS error "
            f"{rel_err * 100:.1f}% exceeds {FIT_QUALITY_THRESHOLD * 100:.0f}% threshold",
            stacklevel=3,
        )


def _junction_outlet_names(
    junction: dict,
    vessel_id_map: dict[int, dict],
) -> list[str]:
    if junction.get("outlet_blocks"):
        return [str(name) for name in junction["outlet_blocks"]]
    if junction.get("outlet_vessels"):
        return [vessel_id_map[int(vid)]["vessel_name"] for vid in junction["outlet_vessels"]]
    return []


def _num_junction_outlets(junction: dict) -> int:
    if junction.get("outlet_vessels"):
        return len(junction["outlet_vessels"])
    if junction.get("outlet_blocks"):
        return len(junction["outlet_blocks"])
    return 0


def infer_set_geo_from_zerod_path(path: str | Path) -> tuple[str | None, str | None]:
    """Parse ``data/zeroD/<set_name>/<run_config>/<geo_name>/...`` from a calibration path."""
    parts = Path(path).resolve().parts
    try:
        idx = parts.index("zeroD")
        set_name = parts[idx + 1]
        geo_name = parts[idx + 3]
        return set_name, geo_name
    except (ValueError, IndexError):
        return None, None


def rsl_fits_output_dir(
    set_name: str,
    geo_name: str,
    *,
    results_root: Path | None = None,
) -> Path:
    root = results_root or repo_root()
    return root / RSL_FITS_RESULTS_SUBDIR / set_name / geo_name


def _safe_plot_stem(element_name: str) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", element_name.strip())
    return stem.strip("_") or "element"


def _save_rsl_fit_plot(
    element_name: str,
    q_in: np.ndarray,
    delta_p: np.ndarray,
    r_poiseuille: float,
    stenosis: float,
    inductance: float,
    dq_out: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_in = np.asarray(q_in, dtype=float)
    delta_p = np.asarray(delta_p, dtype=float)
    dq_out = np.asarray(dq_out, dtype=float)
    delta_p_fit = _predict_delta_p(q_in, dq_out, r_poiseuille, stenosis, inductance)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(q_in, delta_p, s=12, c="0.35", alpha=0.65, label="observations", zorder=2)
    ax.plot(q_in, delta_p_fit, color="blue", linewidth=1.5, label="fit", zorder=3)
    ax.set_xlabel(r"Q")
    ax.set_ylabel(r"$\Delta P$")
    ax.set_title(element_name)
    ax.grid(True, alpha=0.3)
    param_text = f"R = {r_poiseuille:.6g}\nS = {stenosis:.6g}\nL = {inductance:.6g}"
    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def calibrate_decoupled_ls(
    config: dict,
    *,
    plot_rsl_fits: bool = False,
    set_name: str | None = None,
    geo_name: str | None = None,
    calibration_input_path: str | Path | None = None,
    results_root: Path | None = None,
    l2_r: float = 0.0,
    l2_stenosis: float = 0.0,
    l2_l: float = 0.0,
) -> dict:
    """
    Calibrate R/L/(S) per vessel and BloodVesselJunction outlet using decoupled lstsq.

    Uses the same calibration JSON contract as svzerodcalibrator (y, dy, vessels, junctions).
    Capacitance C is kept from the input geometry.
    """
    cali = copy.deepcopy(config)
    cal_params = cali.get("calibration_parameters", {})
    fit_stenosis = bool(cal_params.get("calibrate_stenosis_coefficient", False))
    freeze_connector = bool(cal_params.get("freeze_connector_segments", True))

    if not cali.get("y") or not cali.get("dy"):
        full_obs = cali.get("_full_observations")
        if full_obs and full_obs.get("y") and full_obs.get("dy"):
            cali["y"] = full_obs["y"]
            cali["dy"] = full_obs["dy"]

    y = cali.get("y")
    dy = cali.get("dy")
    if not y or not dy:
        raise ValueError("Calibration config must contain observation dictionaries 'y' and 'dy'")

    vessels = cali.get("vessels", [])
    junctions = cali.get("junctions", [])
    vessel_id_map = _build_vessel_id_map(vessels)

    plot_dir: Path | None = None
    if plot_rsl_fits:
        if not set_name or not geo_name:
            inferred_set, inferred_geo = infer_set_geo_from_zerod_path(calibration_input_path or "")
            set_name = set_name or inferred_set
            geo_name = geo_name or inferred_geo
        if set_name and geo_name:
            plot_dir = rsl_fits_output_dir(set_name, geo_name, results_root=results_root)
            print(f"  RSL fit plots -> {plot_dir}")
        else:
            warnings.warn(
                "plot_rsl_fits enabled but set_name/geo_name could not be resolved; skipping plots",
                stacklevel=2,
            )

    def _maybe_plot(
        element_name: str,
        delta_p: np.ndarray,
        q_in: np.ndarray,
        dq_out: np.ndarray,
        r_poiseuille: float,
        stenosis: float,
        inductance: float,
    ) -> None:
        if plot_dir is None:
            return
        out_path = plot_dir / f"{_safe_plot_stem(element_name)}.png"
        _save_rsl_fit_plot(
            element_name,
            q_in,
            delta_p,
            r_poiseuille,
            stenosis,
            inductance,
            dq_out,
            out_path,
        )

    for vessel in vessels:
        element_type = vessel.get("zero_d_element_type")
        if element_type not in VESSEL_ELEMENT_TYPES:
            continue

        vessel_name = vessel["vessel_name"]
        values = vessel.setdefault("zero_d_element_values", {})

        if freeze_connector and _is_connector_vessel(vessel_name):
            values["R_poiseuille"] = 0.0
            values["L"] = 0.0
            values["stenosis_coefficient"] = 0.0
            continue

        resolved = _resolve_vessel_observations(cali, vessel, junctions)
        if resolved is None:
            continue

        delta_p, q_in, dq_out, _ = resolved
        r_poiseuille, stenosis, inductance, rel_err = _fit_rlc(
            delta_p,
            q_in,
            dq_out,
            fit_stenosis=fit_stenosis,
            l2_r=l2_r,
            l2_stenosis=l2_stenosis,
            l2_l=l2_l,
        )
        if not fit_stenosis:
            stenosis = 0.0

        values["R_poiseuille"] = r_poiseuille
        values["L"] = inductance
        values["stenosis_coefficient"] = stenosis
        _warn_if_poor_fit(vessel_name, rel_err)
        _maybe_plot(vessel_name, delta_p, q_in, dq_out, r_poiseuille, stenosis, inductance)

    for junction in junctions:
        if junction.get("junction_type") != "BloodVesselJunction":
            continue
        num_outlets = _num_junction_outlets(junction)
        if num_outlets < 2:
            continue

        junc_name = junction["junction_name"]
        outlet_names = _junction_outlet_names(junction, vessel_id_map)
        junc_values = junction.setdefault("junction_values", {})

        r_list = list(junc_values.get("R_poiseuille", [0.0] * num_outlets))
        l_list = list(junc_values.get("L", [0.0] * num_outlets))
        s_list = list(junc_values.get("stenosis_coefficient", [0.0] * num_outlets))
        while len(r_list) < num_outlets:
            r_list.append(0.0)
        while len(l_list) < num_outlets:
            l_list.append(0.0)
        while len(s_list) < num_outlets:
            s_list.append(0.0)

        for outlet_index, outlet_name in enumerate(outlet_names):
            element_label = f"{junc_name}/outlet_{outlet_index} ({outlet_name})"

            if freeze_connector and _is_connector_vessel(outlet_name):
                r_list[outlet_index] = 0.0
                l_list[outlet_index] = 0.0
                s_list[outlet_index] = 0.0
                continue

            resolved = _resolve_junction_outlet_observations(
                cali,
                junction,
                outlet_index,
                outlet_name,
                vessel_id_map,
            )
            if resolved is None:
                continue

            delta_p, q_in, dq_out = resolved
            r_poiseuille, stenosis, inductance, rel_err = _fit_rlc(
                delta_p,
                q_in,
                dq_out,
                fit_stenosis=fit_stenosis,
                l2_r=l2_r,
                l2_stenosis=l2_stenosis,
                l2_l=l2_l,
            )
            if not fit_stenosis:
                stenosis = 0.0

            r_list[outlet_index] = r_poiseuille
            l_list[outlet_index] = inductance
            s_list[outlet_index] = stenosis
            _warn_if_poor_fit(element_label, rel_err)
            _maybe_plot(element_label, delta_p, q_in, dq_out, r_poiseuille, stenosis, inductance)

        junc_values["R_poiseuille"] = r_list
        junc_values["L"] = l_list
        junc_values["stenosis_coefficient"] = s_list

    return cali
