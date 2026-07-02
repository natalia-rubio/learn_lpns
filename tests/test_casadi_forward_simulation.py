"""Tests for CasADi forward simulation."""

from __future__ import annotations

import json

import pytest

from learn_lpns.tools.paths import repo_root

pytest.importorskip("casadi")

from learn_lpns.zerod_calibration.casadi_forward_simulation import (  # noqa: E402
    _simulation_schedule,
    build_casadi_problem,
    run_casadi_forward_simulation,
    solve_timestep,
)

MINIMAL_FIXTURE = repo_root() / "tests" / "fixtures" / "minimal_zerod_network.json"


@pytest.fixture
def minimal_input():
    with open(MINIMAL_FIXTURE) as f:
        return json.load(f)


def test_simulation_schedule_single_cycle(minimal_input):
    times, flows, dt = _simulation_schedule(minimal_input)
    assert dt == pytest.approx(0.01)
    assert len(times) == 4
    assert len(flows) == 4
    assert times[0] == pytest.approx(0.0)
    assert flows[0] == pytest.approx(100.0)


def test_build_casadi_problem_minimal_network(minimal_input):
    opti, ctx = build_casadi_problem(minimal_input, is_initial_step=True, dt=0.01)
    assert opti is not None
    assert len(ctx["vessel_order"]) == 2
    assert 1 in ctx["Pc"]


def test_solve_timestep_two_steps(minimal_input):
    sol0 = solve_timestep(minimal_input, inlet_Q=100.0, sol_prev=None, dt=0.01)
    assert len(sol0["Q_in"]) == 2
    assert sol0["Q_in"][0] == pytest.approx(100.0, rel=1e-2)

    sol1 = solve_timestep(minimal_input, inlet_Q=110.0, sol_prev=sol0, dt=0.01)
    assert sol1["Q_in"][0] == pytest.approx(110.0, rel=1e-2)


def test_run_casadi_forward_simulation_writes_csv(minimal_input, tmp_path):
    output_csv = tmp_path / "results_casadi.csv"
    run_casadi_forward_simulation(minimal_input, str(output_csv))

    assert output_csv.is_file()
    text = output_csv.read_text()
    assert "name,time,flow_in,flow_out,pressure_in,pressure_out" in text
    assert "inlet_seg" in text
    assert "outlet_seg" in text
    # 4 time steps x 2 vessels
    assert text.count("\n") >= 8


def test_unsupported_junction_type_raises(minimal_input):
    bad = json.loads(json.dumps(minimal_input))
    bad["junctions"][0]["junction_type"] = "HybridJunction"
    with pytest.raises(ValueError, match="Unsupported junction_type"):
        build_casadi_problem(bad, is_initial_step=True, dt=0.01)
