"""Unsteady 0D forward simulation using CasADi Opti + IPOPT.

Ported from cco_bifurcations ``svzerod_to_casadi_unsteady.py`` with RCR outlet
support. Supports BloodVesselJunction and NORMAL_JUNCTION only.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import pandas as pd

from learn_lpns.config.models import SolverConfig

JUNCTION_PRESSURE_SCALE = 1333.0 * 20.0
SUPPORTED_JUNCTION_TYPES = frozenset({"BloodVesselJunction", "NORMAL_JUNCTION"})


def _require_casadi():
    try:
        import casadi as ca
    except ImportError as e:
        raise ImportError(
            "CasADi forward backend requires the casadi package. "
            'Install with: pip install -e ".[casadi]"'
        ) from e
    return ca


def _inflow_bc(input_data: dict) -> dict:
    for bc in input_data.get("boundary_conditions", []):
        if bc.get("bc_name") == "INFLOW" and bc.get("bc_type") == "FLOW":
            return bc
    raise ValueError("INFLOW FLOW boundary condition not found in input JSON")


def _bc_index(input_data: dict) -> dict[str, int]:
    return {bc["bc_name"]: i for i, bc in enumerate(input_data.get("boundary_conditions", []))}


def _vessel_resistance(Q_in, R_lin, R_sten, R_quad, ca):
    return R_lin + R_sten * ca.sqrt(1e-2 + Q_in**2) + R_quad * Q_in


def build_casadi_problem(
    input_data: dict,
    *,
    is_initial_step: bool,
    dt: float,
) -> tuple[Any, dict[str, Any]]:
    """Build a CasADi Opti problem for one time step.

    Returns (opti, context) where context holds variables and metadata for extraction.
    """
    ca = _require_casadi()

    vessels = input_data["vessels"]
    junctions = input_data["junctions"]
    num_vessels = len(vessels)
    num_junctions = len(junctions)
    bc_ind = _bc_index(input_data)

    opti = ca.Opti()
    objective = 0

    Q_in = opti.variable(num_vessels)
    Q_out = opti.variable(num_vessels)
    Q_in_dt = opti.variable(num_vessels)
    Q_out_dt = opti.variable(num_vessels)
    P_in = opti.variable(num_vessels)
    P_out = opti.variable(num_vessels)
    P_in_dt = opti.variable(num_vessels)
    P_out_dt = opti.variable(num_vessels)

    inlet_flow = opti.parameter()
    inflow_extractors = opti.parameter(num_vessels, num_junctions)
    outflow_extractors = opti.parameter(num_vessels, num_junctions)
    opti.set_value(inflow_extractors, 0)
    opti.set_value(outflow_extractors, 0)

    Q_in_prev = opti.parameter(num_vessels)
    Q_out_prev = opti.parameter(num_vessels)
    P_in_prev = opti.parameter(num_vessels)
    P_out_prev = opti.parameter(num_vessels)

    vessel_dict: dict[int, dict[str, Any]] = defaultdict(dict)
    vessel_order: list[int] = []
    Pc: dict[int, Any] = {}
    Pc_dt: dict[int, Any] = {}
    Pc_prev: dict[int, Any] = {}

    for i, vessel in enumerate(vessels):
        vid = vessel["vessel_id"]
        vessel_dict[vid]["v_ind"] = i
        vessel_dict[vid]["v_name"] = vessel["vessel_name"]
        vessel_order.append(vid)

        vals = vessel["zero_d_element_values"]
        R_lin = float(vals["R_poiseuille"])
        R_sten = float(vals.get("stenosis_coefficient", 0.0))
        R_quad = float(vals.get("pressure_recovery_coefficient", 0.0))
        C = float(vals.get("C", 0.0))
        L = float(vals.get("L", 0.0))

        R_total = _vessel_resistance(Q_in[i], R_lin, R_sten, R_quad, ca)
        objective += (P_in[i] - P_out[i] - R_total * Q_in[i] - L * Q_out_dt[i]) ** 2

        dR_dQ = 2 * R_sten * Q_in[i] / ca.sqrt(1e-10 + Q_in[i] ** 2) + R_quad
        opti.subject_to(
            Q_in[i]
            - Q_out[i]
            - C * P_in_dt[i]
            + C * (R_lin + dR_dQ * Q_in[i]) * Q_in_dt[i]
            == 0
        )

        bcs = vessel.get("boundary_conditions", {})
        if "outlet" in bcs:
            bc_name = bcs["outlet"]
            bc = input_data["boundary_conditions"][bc_ind[bc_name]]
            bc_type = bc.get("bc_type")
            bc_values = bc.get("bc_values", {})
            if bc_type == "RESISTANCE":
                R = float(bc_values.get("R", 0.0))
                Pd = float(bc_values.get("Pd", 0.0))
                opti.subject_to(P_out[i] - Q_out[i] * R == Pd)
            elif bc_type == "RCR":
                Rp = float(bc_values["Rp"])
                C_rcr = float(bc_values["C"])
                Rd = float(bc_values["Rd"])
                Pd = float(bc_values.get("Pd", 0.0))
                Pc[i] = opti.variable()
                Pc_dt[i] = opti.variable()
                Pc_prev[i] = opti.parameter()
                opti.subject_to(P_out[i] - Pc[i] - Rp * Q_out[i] == 0)
                opti.subject_to(C_rcr * Pc_dt[i] - Q_out[i] + (Pc[i] - Pd) / Rd == 0)
            else:
                raise ValueError(
                    f"Unsupported outlet bc_type {bc_type!r} on vessel {vessel['vessel_name']!r}; "
                    "CasADi forward solver supports RESISTANCE and RCR only."
                )

        if "inlet" in bcs:
            opti.subject_to(Q_in[i] == inlet_flow)

    for j_idx, junction in enumerate(junctions):
        j_type = junction.get("junction_type")
        if j_type not in SUPPORTED_JUNCTION_TYPES:
            raise ValueError(
                f"Unsupported junction_type {j_type!r} on {junction.get('junction_name')!r}; "
                f"expected one of {sorted(SUPPORTED_JUNCTION_TYPES)}"
            )

        inlet_vessel_id = junction["inlet_vessels"][0]
        inlet_vessel_ind = vessel_dict[inlet_vessel_id]["v_ind"]
        outlet_vessel_inds = [vessel_dict[ov]["v_ind"] for ov in junction["outlet_vessels"]]

        for j, outlet_vessel_ind in enumerate(outlet_vessel_inds):
            if j_type == "BloodVesselJunction":
                jvals = junction["junction_values"]
                R_lin = float(jvals["R_poiseuille"][j])
                R_sten = float(jvals["stenosis_coefficient"][j])
                R_quad = float(jvals.get("pressure_recovery_coefficient", [0.0] * len(outlet_vessel_inds))[j])
                L = float(jvals["L"][j])

                R_j = _vessel_resistance(Q_in[outlet_vessel_ind], R_lin, R_sten, R_quad, ca)
                objective += (
                    (
                        P_out[inlet_vessel_ind]
                        - P_in[outlet_vessel_ind]
                        - R_j * Q_in[outlet_vessel_ind]
                        - L * Q_in_dt[outlet_vessel_ind]
                    )
                    / JUNCTION_PRESSURE_SCALE
                ) ** 2

                opti.subject_to(P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] >= 0)

                flow_splits = jvals.get("flow_split")
                if flow_splits is not None:
                    objective += (
                        (Q_in[outlet_vessel_ind] - float(flow_splits[j]) * Q_out[inlet_vessel_ind]) * 1000.0
                    ) ** 2

            elif j_type == "NORMAL_JUNCTION":
                opti.subject_to(P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] == 0)

            opti.set_value(outflow_extractors[outlet_vessel_ind, j_idx], -1)
            opti.set_value(inflow_extractors[inlet_vessel_ind, j_idx], 1)
            opti.subject_to(Q_out.T @ inflow_extractors[:, j_idx] + Q_in.T @ outflow_extractors[:, j_idx] == 0)

    if is_initial_step:
        opti.subject_to(ca.vec(Q_in_dt) == 0)
        opti.subject_to(ca.vec(Q_out_dt) == 0)
        opti.subject_to(ca.vec(P_in_dt) == 0)
        opti.subject_to(ca.vec(P_out_dt) == 0)
        for idx in Pc_dt:
            opti.subject_to(Pc_dt[idx] == 0)
    else:
        opti.subject_to(Q_in_dt == (Q_in - Q_in_prev) / dt)
        opti.subject_to(Q_out_dt == (Q_out - Q_out_prev) / dt)
        opti.subject_to(P_in_dt == (P_in - P_in_prev) / dt)
        opti.subject_to(P_out_dt == (P_out - P_out_prev) / dt)
        for idx, pc_dt in Pc_dt.items():
            opti.subject_to(pc_dt == (Pc[idx] - Pc_prev[idx]) / dt)

    opti.minimize(objective)
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
    opti.solver("ipopt", opts)

    context = {
        "objective": objective,
        "vessel_dict": dict(vessel_dict),
        "vessel_order": vessel_order,
        "Q_in": Q_in,
        "Q_out": Q_out,
        "Q_in_dt": Q_in_dt,
        "Q_out_dt": Q_out_dt,
        "P_in": P_in,
        "P_out": P_out,
        "P_in_dt": P_in_dt,
        "P_out_dt": P_out_dt,
        "Pc": Pc,
        "Pc_dt": Pc_dt,
        "inlet_flow": inlet_flow,
        "Q_in_prev": Q_in_prev,
        "Q_out_prev": Q_out_prev,
        "P_in_prev": P_in_prev,
        "P_out_prev": P_out_prev,
        "Pc_prev": Pc_prev,
    }
    return opti, context


def solve_timestep(
    input_data: dict,
    *,
    inlet_Q: float,
    sol_prev: dict[str, Any] | None,
    dt: float,
) -> dict[str, Any]:
    """Solve one time step and return state dict for the next step."""
    is_initial = sol_prev is None
    opti, ctx = build_casadi_problem(input_data, is_initial_step=is_initial, dt=dt)
    opti.set_value(ctx["inlet_flow"], inlet_Q)

    if is_initial:
        opti.set_value(ctx["Q_in_prev"], 0)
        opti.set_value(ctx["Q_out_prev"], 0)
        opti.set_value(ctx["P_in_prev"], 0)
        opti.set_value(ctx["P_out_prev"], 0)
        for idx, pc_prev in ctx["Pc_prev"].items():
            opti.set_value(pc_prev, 0)
    else:
        opti.set_value(ctx["Q_in_prev"], sol_prev["Q_in"])
        opti.set_value(ctx["Q_out_prev"], sol_prev["Q_out"])
        opti.set_value(ctx["P_in_prev"], sol_prev["P_in"])
        opti.set_value(ctx["P_out_prev"], sol_prev["P_out"])
        for idx, pc_prev in ctx["Pc_prev"].items():
            opti.set_value(pc_prev, sol_prev["Pc"][idx])

        opti.set_initial(ctx["Q_in"], sol_prev["Q_in"] + dt * sol_prev["Q_in_dt"])
        opti.set_initial(ctx["Q_out"], sol_prev["Q_out"] + dt * sol_prev["Q_out_dt"])
        opti.set_initial(ctx["Q_in_dt"], sol_prev["Q_in_dt"])
        opti.set_initial(ctx["Q_out_dt"], sol_prev["Q_out_dt"])
        opti.set_initial(ctx["P_in"], sol_prev["P_in"] + dt * sol_prev["P_in_dt"])
        opti.set_initial(ctx["P_out"], sol_prev["P_out"] + dt * sol_prev["P_out_dt"])
        opti.set_initial(ctx["P_in_dt"], sol_prev["P_in_dt"])
        opti.set_initial(ctx["P_out_dt"], sol_prev["P_out_dt"])
        for idx, pc in ctx["Pc"].items():
            opti.set_initial(pc, sol_prev["Pc"][idx] + dt * sol_prev["Pc_dt"][idx])
            opti.set_initial(ctx["Pc_dt"][idx], sol_prev["Pc_dt"][idx])

    solver_ok = True
    try:
        sol = opti.solve()
    except Exception:
        sol = opti.debug
        solver_ok = False

    objective_val = float(opti.debug.value(ctx["objective"]))

    Pc_vals = {idx: float(sol.value(ctx["Pc"][idx])) for idx in ctx["Pc"]}
    Pc_dt_vals = {idx: float(sol.value(ctx["Pc_dt"][idx])) for idx in ctx["Pc_dt"]}

    return {
        "Q_in": sol.value(ctx["Q_in"]),
        "Q_out": sol.value(ctx["Q_out"]),
        "P_in": sol.value(ctx["P_in"]),
        "P_out": sol.value(ctx["P_out"]),
        "Q_in_dt": sol.value(ctx["Q_in_dt"]),
        "Q_out_dt": sol.value(ctx["Q_out_dt"]),
        "P_in_dt": sol.value(ctx["P_in_dt"]),
        "P_out_dt": sol.value(ctx["P_out_dt"]),
        "Pc": Pc_vals,
        "Pc_dt": Pc_dt_vals,
        "_vessel_order": ctx["vessel_order"],
        "_vessel_dict": ctx["vessel_dict"],
        "_objective": objective_val,
        "_solver_ok": solver_ok,
    }


def _simulation_schedule(input_data: dict) -> tuple[list[float], list[float], float]:
    """Return (times, inlet_flows, dt) to simulate and write to CSV."""
    inflow = _inflow_bc(input_data)
    t_cycle = list(inflow["bc_values"]["t"])
    q_cycle = list(inflow["bc_values"]["Q"])
    sim = input_data.get("simulation_parameters", {})

    n_pts = int(sim.get("number_of_time_pts_per_cardiac_cycle", len(t_cycle)))
    n_pts = min(n_pts, len(t_cycle), len(q_cycle))
    t_cycle = t_cycle[:n_pts]
    q_cycle = q_cycle[:n_pts]

    if n_pts < 2:
        raise ValueError("INFLOW boundary condition must have at least 2 time points")

    dt = t_cycle[1] - t_cycle[0]
    if t_cycle[0] == 0.0:
        period = t_cycle[-1]
    else:
        period = t_cycle[-1] - t_cycle[0] + dt

    n_cycles = int(sim.get("number_of_cardiac_cycles", 1))
    output_all = bool(sim.get("output_all_cycles", True))

    all_times: list[float] = []
    all_flows: list[float] = []
    for cycle in range(n_cycles):
        offset = cycle * period
        for i in range(n_pts):
            all_times.append(t_cycle[i] + offset)
            all_flows.append(q_cycle[i])

    if not output_all and n_cycles > 1:
        start = (n_cycles - 1) * n_pts
        return all_times[start:], all_flows[start:], dt

    return all_times, all_flows, dt


def _solution_to_rows(sol: dict[str, Any], time: float) -> list[dict[str, float | str]]:
    rows = []
    for vid in sol["_vessel_order"]:
        info = sol["_vessel_dict"][vid]
        idx = info["v_ind"]
        rows.append(
            {
                "name": info["v_name"],
                "time": time,
                "flow_in": float(sol["Q_in"][idx]),
                "flow_out": float(sol["Q_out"][idx]),
                "pressure_in": float(sol["P_in"][idx]),
                "pressure_out": float(sol["P_out"][idx]),
            }
        )
    return rows


def run_casadi_forward_simulation(
    input_data: dict,
    output_csv_path: str,
    *,
    solver_config: SolverConfig | None = None,
) -> None:
    """Run unsteady CasADi forward simulation and write results CSV."""
    _require_casadi()

    times, flows, dt = _simulation_schedule(input_data)
    n_steps = len(times)
    print(f"  CasADi forward simulation: {n_steps} time steps (dt={dt:.6f}s)")

    rows: list[dict[str, float | str]] = []
    sol_prev: dict[str, Any] | None = None
    sim_t0 = time.perf_counter()

    for step_idx, (time_val, inlet_Q) in enumerate(zip(times, flows, strict=True)):
        step_t0 = time.perf_counter()
        sol_prev = solve_timestep(input_data, inlet_Q=inlet_Q, sol_prev=sol_prev, dt=dt)
        step_elapsed = time.perf_counter() - step_t0

        objective = sol_prev["_objective"]
        solver_ok = sol_prev["_solver_ok"]
        status = "ok" if solver_ok else "debug"
        print(
            f"    step {step_idx + 1}/{n_steps}: t={time_val:.6f}, Q_in={inlet_Q:.3f}, "
            f"residual={objective:.6e}, elapsed={step_elapsed:.3f}s ({status})"
        )
        rows.extend(_solution_to_rows(sol_prev, time_val))

    total_elapsed = time.perf_counter() - sim_t0
    print(f"  CasADi forward simulation finished in {total_elapsed:.3f}s")

    result_df = pd.DataFrame(rows, columns=["name", "time", "flow_in", "flow_out", "pressure_in", "pressure_out"])
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
