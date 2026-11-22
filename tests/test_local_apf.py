from __future__ import annotations

import math

import numpy as np
import pytest

from config import DT, ROBOT_RADIUS, V_MAX, W_MAX
from planners.base import LocalPlanner
from planners.local_apf import APFPlanner


def _step_unicycle(state: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    nx = float(state[0]) + v * math.cos(float(state[2])) * dt
    ny = float(state[1]) + v * math.sin(float(state[2])) * dt
    ntheta = (float(state[2]) + w * dt + math.pi) % (2.0 * math.pi) - math.pi
    return np.array([nx, ny, ntheta, v, w], dtype=float)


def test_apf_interface_and_reset_reproducibility() -> None:
    planner = APFPlanner()
    assert isinstance(planner, LocalPlanner)
    assert planner.name == "APFPlanner"

    state = np.array([10.0, 10.0, 0.0, 0.2, 0.0], dtype=float)
    goal = np.array([15.0, 10.2], dtype=float)
    pts = np.array([[11.8, 10.3], [12.5, 9.4]], dtype=float)

    planner.reset()
    v1, w1, traj1 = planner.compute_control(state, goal, pts, DT)
    v2, w2, traj2 = planner.compute_control(_step_unicycle(state, v1, w1, DT), goal, pts, DT)

    planner.reset()
    r_v1, r_w1, r_traj1 = planner.compute_control(state, goal, pts, DT)
    r_v2, r_w2, r_traj2 = planner.compute_control(_step_unicycle(state, r_v1, r_w1, DT), goal, pts, DT)

    assert v1 == pytest.approx(r_v1)
    assert w1 == pytest.approx(r_w1)
    assert v2 == pytest.approx(r_v2)
    assert w2 == pytest.approx(r_w2)
    assert np.allclose(traj1, r_traj1)
    assert np.allclose(traj2, r_traj2)


def test_apf_near_goal_stops_cleanly() -> None:
    planner = APFPlanner()
    planner.reset()

    v, w, traj = planner.compute_control(
        [12.0, 10.0, 0.1, 0.4, 0.2],
        [12.02, 10.0],
        np.zeros((0, 2), dtype=float),
        DT,
    )

    assert v == 0.0
    assert w == 0.0
    assert traj.shape == (planner.pred_steps, 2)
    assert np.all(np.isfinite(traj))


def test_apf_handles_empty_or_none_points_and_invalid_inputs() -> None:
    planner = APFPlanner()
    planner.reset()

    v_none, w_none, traj_none = planner.compute_control([5.0, 5.0, 0.0], [8.0, 5.0], None, DT)
    assert math.isfinite(v_none) and math.isfinite(w_none)
    assert traj_none.shape == (planner.pred_steps, 2)
    assert np.all(np.isfinite(traj_none))

    v_empty, w_empty, traj_empty = planner.compute_control(
        [5.0, 5.0, 0.0],
        [8.0, 5.0],
        np.zeros((0, 2), dtype=float),
        DT,
    )
    assert math.isfinite(v_empty) and math.isfinite(w_empty)
    assert traj_empty.shape == (planner.pred_steps, 2)

    with pytest.raises(ValueError):
        planner.compute_control([1.0, 2.0], [5.0, 5.0], None, DT)

    with pytest.raises(ValueError):
        planner.compute_control([1.0, 2.0, 0.0], [5.0], None, DT)


def test_apf_moves_toward_goal_and_respects_limits() -> None:
    planner = APFPlanner()
    planner.reset()

    state = np.array([13.0, 10.0, 0.0, 0.0, 0.0], dtype=float)
    goal = np.array([22.0, 10.0], dtype=float)

    for _ in range(250):
        v, w, traj = planner.compute_control(state, goal, np.zeros((0, 2), dtype=float), DT)
        assert math.isfinite(v) and math.isfinite(w)
        assert traj.shape == (planner.pred_steps, 2)
        assert np.all(np.isfinite(traj))
        assert abs(w) <= W_MAX + 1e-6
        assert v <= V_MAX + 1e-6

        state = _step_unicycle(state, v, w, DT)
        if math.hypot(float(goal[0] - state[0]), float(goal[1] - state[1])) < 0.2:
            break

    assert float(state[0]) > 16.0


def test_apf_brakes_for_obstacle_ahead() -> None:
    planner = APFPlanner()
    state = np.array([10.0, 10.0, 0.0, 0.6, 0.0], dtype=float)
    goal = np.array([18.0, 10.0], dtype=float)

    planner.reset()
    v_far, _, _ = planner.compute_control(state, goal, np.array([[14.5, 10.0]]), DT)

    planner.reset()
    v_near, _, _ = planner.compute_control(state, goal, np.array([[10.0 + ROBOT_RADIUS + 0.14, 10.0]]), DT)

    assert v_near < v_far


def test_apf_locks_side_and_escapes_blockage() -> None:
    planner = APFPlanner(stuck_window=0.5, stuck_dist=0.05, escape_time=0.6)
    planner.reset()

    state = np.array([10.0, 10.0, 0.0, 0.0, 0.0], dtype=float)
    goal = np.array([16.0, 10.0], dtype=float)

    blocker = np.array([[11.1, 10.0], [11.1, 10.12], [11.1, 9.88]])

    locked_sides: set[int] = set()
    for _ in range(80):
        v, w, traj = planner.compute_control(state, goal, blocker, DT)
        assert math.isfinite(v) and math.isfinite(w)
        assert traj.shape == (planner.pred_steps, 2)
        if planner._side != 0:
            locked_sides.add(planner._side)
        state = _step_unicycle(state, v, w, DT)

    assert len(locked_sides) == 1
    assert planner._esc_until > 0.0 or abs(float(state[1]) - 10.0) > 0.05
