"""
Physics-based non-dimensionalization of R/S/L (RRI) coefficients.

Follows arXiv:2508.21165 (Rubio et al.) Eqs. 9-12. Characteristic length l_c is the
inlet radius (``inlet_max_inscribed_radius``). Non-dimensional targets are:

  R* = R * Q_c / P_c
  S* = S * Q_c^2 / P_c
  L* = L * Q_c / (t_c * P_c)

where Q_c = pi * l_c^2 * U_c, U_c = Re_c * mu / (2 * rho * l_c),
t_c = l_c / U_c, P_c = rho * U_c^2.
"""

from __future__ import annotations

import numpy as np

INLET_RADIUS_FEATURE = "inlet_max_inscribed_radius"


def characteristic_scales(
    inlet_radius: np.ndarray | float,
    *,
    rho: float,
    mu: float,
    reference_reynolds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (Q_c, t_c, P_c, U_c) for each inlet radius l_c.

    Raises ValueError if any inlet_radius <= 0.
    """
    lc = np.asarray(inlet_radius, dtype=np.float64)
    if np.any(lc <= 0):
        raise ValueError(f"inlet_radius must be positive, got min={np.min(lc)}")

    u_c = reference_reynolds * mu / (2.0 * rho * lc)
    q_c = np.pi * lc**2 * u_c
    t_c = lc / u_c
    p_c = rho * u_c**2
    return q_c, t_c, p_c, u_c


def rsl_scale_factors(
    inlet_radius: np.ndarray | float,
    *,
    rho: float,
    mu: float,
    reference_reynolds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f_R, f_S, f_L) such that R* = R * f_R, S* = S * f_S, L* = L * f_L."""
    q_c, t_c, p_c, _ = characteristic_scales(
        inlet_radius, rho=rho, mu=mu, reference_reynolds=reference_reynolds
    )
    f_r = q_c / p_c
    f_s = (q_c**2) / p_c
    f_l = q_c / (t_c * p_c)
    return f_r, f_s, f_l


def nondimensionalize_rsl(
    rsl: np.ndarray,
    inlet_radius: np.ndarray | float,
    *,
    rho: float,
    mu: float,
    reference_reynolds: float,
) -> np.ndarray:
    """
    Convert physical R/S/L to non-dimensional R*/S*/L*.

    ``rsl`` shape (n, 3) or (3,); columns are R, S, L.
    """
    arr = np.asarray(rsl, dtype=np.float64)
    single_row = arr.ndim == 1
    if single_row:
        arr = arr.reshape(1, 3)
    if arr.shape[1] != 3:
        raise ValueError(f"rsl must have 3 columns (R, S, L), got shape {arr.shape}")

    lc = np.asarray(inlet_radius, dtype=np.float64)
    if lc.ndim == 0:
        lc = np.full(arr.shape[0], float(lc))
    elif lc.shape[0] != arr.shape[0]:
        raise ValueError(f"inlet_radius length {lc.shape[0]} != rsl rows {arr.shape[0]}")

    f_r, f_s, f_l = rsl_scale_factors(
        lc, rho=rho, mu=mu, reference_reynolds=reference_reynolds
    )
    out = np.column_stack([arr[:, 0] * f_r, arr[:, 1] * f_s, arr[:, 2] * f_l])
    return out[0] if single_row else out


def redimensionalize_rsl(
    rsl_star: np.ndarray,
    inlet_radius: np.ndarray | float,
    *,
    rho: float,
    mu: float,
    reference_reynolds: float,
) -> np.ndarray:
    """
    Convert non-dimensional R*/S*/L* back to physical R/S/L.

    ``rsl_star`` shape (n, 3) or (3,); columns are R*, S*, L*.
    """
    arr = np.asarray(rsl_star, dtype=np.float64)
    single_row = arr.ndim == 1
    if single_row:
        arr = arr.reshape(1, 3)
    if arr.shape[1] != 3:
        raise ValueError(f"rsl_star must have 3 columns (R*, S*, L*), got shape {arr.shape}")

    lc = np.asarray(inlet_radius, dtype=np.float64)
    if lc.ndim == 0:
        lc = np.full(arr.shape[0], float(lc))
    elif lc.shape[0] != arr.shape[0]:
        raise ValueError(f"inlet_radius length {lc.shape[0]} != rsl_star rows {arr.shape[0]}")

    f_r, f_s, f_l = rsl_scale_factors(
        lc, rho=rho, mu=mu, reference_reynolds=reference_reynolds
    )
    out = np.column_stack([arr[:, 0] / f_r, arr[:, 1] / f_s, arr[:, 2] / f_l])
    return out[0] if single_row else out


def redimensionalize_rsl_predictions(
    pred_R: np.ndarray,
    pred_S: np.ndarray,
    pred_L: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    *,
    rho: float,
    mu: float,
    reference_reynolds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map non-dimensional R*/S*/L* network outputs back to physical R, S, L."""
    if INLET_RADIUS_FEATURE not in feature_names:
        raise ValueError(f"Missing {INLET_RADIUS_FEATURE!r} in feature_names for re-dimensionalization")
    lc_idx = feature_names.index(INLET_RADIUS_FEATURE)
    inlet_radius = np.asarray(feature_matrix[:, lc_idx], dtype=np.float64)
    out = redimensionalize_rsl(
        np.column_stack([pred_R, pred_S, pred_L]),
        inlet_radius,
        rho=rho,
        mu=mu,
        reference_reynolds=reference_reynolds,
    )
    return out[:, 0], out[:, 1], out[:, 2]
