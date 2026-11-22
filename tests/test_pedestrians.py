"""tests/test_pedestrians.py -- Kiem chung module vat can dong env/pedestrians.py.

Chay (tu THU MUC GOC cua du an):
    python -m tests.test_pedestrians
    python -m pytest tests/test_pedestrians.py -v
"""
from __future__ import annotations

import math
import time

import numpy as np

from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from env.pedestrians import (
    Pedestrian,
    PedestrianManager,
    circles_from_pedestrians,
    default_hospital_pedestrians,
    is_circle_free,
)
from sensors.lidar import Lidar2D

TOL = 1e-9


class SimpleGrid:
    """Luoi toi thieu de test doc lap."""

    def __init__(self, width: float = 10.0, height: float = 10.0, res: float = 0.2):
        self.width = width
        self.height = height
        self.res = res
        self.n_cols = int(round(width / res))
        self.n_rows = int(round(height / res))
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=bool)

    def fill_rect(self, x1: float, y1: float, x2: float, y2: float) -> "SimpleGrid":
        c1 = max(0, int(math.floor(x1 / self.res)))
        c2 = min(self.n_cols, int(math.ceil(x2 / self.res)))
        r1 = max(0, int(math.floor(y1 / self.res)))
        r2 = min(self.n_rows, int(math.ceil(y2 / self.res)))
        self.grid[r1:r2, c1:c2] = True
        return self


def test_circle_free_and_boundary_checks():
    g = SimpleGrid(width=10.0, height=10.0, res=0.5)
    g.fill_rect(4.0, 4.0, 5.0, 5.0)

    assert is_circle_free(None, 2.0, 2.0, 0.3)
    assert is_circle_free(g, 2.0, 2.0, 0.3)
    assert not is_circle_free(g, 0.2, 5.0, 0.3)
    assert not is_circle_free(g, 4.5, 4.5, 0.25)
    assert not is_circle_free(g, 3.85, 4.5, 0.25)
    assert is_circle_free(g, 3.60, 4.5, 0.25)


def test_input_validation_errors():
    for bad_kw in (
        dict(x=1.0, y=1.0, radius=0.0),
        dict(x=1.0, y=1.0, radius=-0.2),
        dict(x=1.0, y=1.0, radius=float("nan")),
        dict(x=1.0, y=1.0, speed=-0.5),
        dict(x=1.0, y=1.0, mode="invalid"),
    ):
        try:
            Pedestrian(**bad_kw)
        except ValueError:
            continue
        raise AssertionError(f"Pedestrian({bad_kw}) phai bao ValueError")

    p = Pedestrian(1.0, 1.0)
    for bad_dt in (-0.1, float("nan"), float("inf")):
        try:
            p.step(bad_dt)
        except ValueError:
            continue
        raise AssertionError(f"Pedestrian.step({bad_dt}) phai bao ValueError")


def test_single_pedestrian_waypoint_loop_and_ping_pong():
    p_loop = Pedestrian(x=1.0, y=2.0, radius=0.25, speed=1.0, waypoints=[(1.0, 2.0), (4.0, 2.0)], loop=True)
    p_ping = Pedestrian(x=1.0, y=5.0, radius=0.25, speed=1.0, waypoints=[(1.0, 5.0), (3.0, 5.0)], loop=False)

    max_x_loop = p_loop.x
    max_x_ping = p_ping.x
    reversed_ping = False

    for _ in range(120):
        p_loop.step(0.05, world=None)
        p_ping.step(0.05, world=None)
        max_x_loop = max(max_x_loop, p_loop.x)
        if p_ping.x > max_x_ping:
            max_x_ping = p_ping.x
        elif max_x_ping > 2.6 and p_ping.x < max_x_ping - 0.1:
            reversed_ping = True

    arr = circles_from_pedestrians([p_loop, p_ping])
    assert arr.shape == (2, 3)
    assert max_x_loop > 3.5
    assert p_loop.x < max_x_loop - 0.5
    assert max_x_ping > 2.6
    assert reversed_ping or p_ping.x < max_x_ping - 0.3


def test_never_penetrates_static_wall_or_unmapped_cart():
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    ped = Pedestrian(x=15.5, y=10.8, radius=0.28, speed=0.8, waypoints=[(15.5, 10.8), (19.0, 10.8)], loop=False)

    for _ in range(120):
        cx, cy, r = ped.step(0.05, world=world)
        assert is_circle_free(world, cx, cy, r, min_clearance=-TOL)


def test_manager_seed_reset_and_lidar_integration():
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    mgr = default_hospital_pedestrians(seed=42)
    lidar = Lidar2D(rng=np.random.default_rng(42))

    first_run = [mgr.step(0.05, world=world).copy() for _ in range(20)]
    mgr.reset(seed=42)
    second_run = [mgr.step(0.05, world=world).copy() for _ in range(20)]
    for a, b in zip(first_run, second_run):
        assert np.allclose(a, b, atol=TOL)

    mgr.reset(seed=42)
    saw_dynamic = False
    for _ in range(80):
        circles = mgr.step(0.05, world=world)
        scan = lidar.scan(world, [12.0, 10.0, 0.0], circles=circles)
        if scan.dynamic.any():
            saw_dynamic = True
            assert scan.points_dynamic.shape[0] == int(scan.dynamic.sum())
    assert saw_dynamic


def test_pedestrian_manager_custom_list():
    m = PedestrianManager([Pedestrian(2.0, 2.0, speed=0.5, mode="wander")], seed=9)
    c = m.step(0.05)
    assert c.shape == (1, 3)


def test_step_performance():
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    mgr = default_hospital_pedestrians(seed=42)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        mgr.step(0.05, world=world)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) * 1000.0 < 10.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test pass")
    raise SystemExit(1 if failed else 0)