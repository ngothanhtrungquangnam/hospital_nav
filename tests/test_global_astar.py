"""tests/test_global_astar.py -- Kiem tra planners/global_astar.py.

Chay (tu THU MUC GOC cua du an, da kich hoat .venv):
    python -m tests.test_global_astar          # in bang so sanh heuristic + ve hinh
    python -m pytest tests/test_global_astar.py -v
"""
import math

import numpy as np

from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from planners.global_astar import (GlobalAStarPlanner, SubGoalSelector, polyline_length)

H_LIST = ("dijkstra", "manhattan", "euclid", "octile")


class ToyMap:
    """Ban do nho tu tay (res = 1 m) co cung giao dien toi thieu voi HospitalMap."""
    xy_to_rc = HospitalMap.xy_to_rc
    rc_to_xy = HospitalMap.rc_to_xy

    def __init__(self, blocked):
        self.inflated = np.asarray(blocked, dtype=bool)
        self.res = 1.0
        self.n_rows, self.n_cols = self.inflated.shape
        self.width, self.height = float(self.n_cols), float(self.n_rows)


def center(r, c):
    return (c + 0.5, r + 0.5)                                   # tam o (r, c) khi res = 1


def check_path(blocked, rc):
    """Duong hop le: o tu do, hai o lien tiep la hang xom, buoc cheo khong cat goc."""
    rc = np.asarray(rc)
    if len(rc) == 0 or blocked[rc[:, 0], rc[:, 1]].any():
        return False
    for a, b in zip(rc[:-1], rc[1:]):
        d = b - a
        if np.abs(d).max() != 1:
            return False
        if d[0] != 0 and d[1] != 0 and (blocked[a[0] + d[0], a[1]] or blocked[a[0], a[1] + d[1]]):
            return False
    return True


# ---------------------------------------------------------------- A. ban do nho
def test_open_grid_diagonal():
    m = ToyMap(np.zeros((10, 10), bool))
    res = GlobalAStarPlanner(m).plan(center(0, 0), center(9, 9))
    assert res.found and len(res.path_xy) == 10
    assert abs(res.length - 9 * math.sqrt(2)) < 1e-9


def test_no_corner_cutting():
    blk = np.zeros((4, 4), bool)
    blk[1, 2] = blk[2, 1] = True                                # hai o chan cheo nhau
    res = GlobalAStarPlanner(ToyMap(blk)).plan(center(1, 1), center(2, 2))
    assert res.found and res.length > 1.5                       # phai di vong, khong luon qua khe
    assert check_path(blk, res.path_rc)


def test_unreachable_returns_not_found():
    blk = np.zeros((5, 5), bool)
    blk[:, 2] = True
    res = GlobalAStarPlanner(ToyMap(blk)).plan(center(2, 0), center(2, 4))
    assert not res.found and res.msg and len(res.path_xy) == 0


def test_start_equals_goal():
    res = GlobalAStarPlanner(ToyMap(np.zeros((5, 5), bool))).plan(center(2, 2), center(2, 2))
    assert res.found and len(res.path_xy) == 1 and res.length == 0.0


# ---------------------------------------------------------------- B. ban do benh vien
def test_invalid_endpoints_do_not_raise():
    m = HospitalMap()
    p = GlobalAStarPlanner(m)
    r1 = p.plan(m.start, (10.1, 15.0))                          # Goal trong tuong ngan x = 10
    r2 = p.plan(m.start, (45.0, 5.0))                           # ngoai ban do
    r3 = p.plan((0.1, 0.1), m.goals["G1"])                      # Start trong tuong
    for r in (r1, r2, r3):
        assert not r.found and r.msg


def test_heuristics_on_hospital_map():
    m = HospitalMap()
    for name, goal in m.goals.items():
        res = {h: GlobalAStarPlanner(m, h).plan(m.start, goal) for h in H_LIST}
        for h, r in res.items():
            assert r.found and check_path(m.inflated, r.path_rc), f"{name}/{h}: duong khong hop le"
        L = {h: r.length for h, r in res.items()}
        N = {h: r.n_expanded for h, r in res.items()}
        assert abs(L["euclid"] - L["dijkstra"]) < 1e-9 and abs(L["octile"] - L["dijkstra"]) < 1e-9, \
            f"{name}: heuristic chap nhan duoc phai cho duong toi uu"
        assert L["manhattan"] >= L["dijkstra"] - 1e-9
        assert N["octile"] <= N["euclid"] <= N["dijkstra"], f"{name}: so o mo rong sai thu tu {N}"


def test_simplify_keeps_endpoints_and_is_collision_free():
    m = HospitalMap()
    p = GlobalAStarPlanner(m)
    for name, goal in m.goals.items():
        r = p.plan(m.start, goal)
        wp = p.simplify(r.path_xy)
        assert len(wp) < len(r.path_xy)
        assert np.allclose(wp[0], m.start) and np.allclose(wp[-1], goal)
        assert all(p.segment_free(a, b) for a, b in zip(wp[:-1], wp[1:])), f"{name}: doan cat vat can"
        assert polyline_length(wp) <= polyline_length(r.path_xy) + 1e-9


def test_resample_spacing():
    pts = GlobalAStarPlanner.resample(np.array([[0, 0], [10, 0], [10, 5]]), 1.0)
    assert np.allclose(pts[0], [0, 0]) and np.allclose(pts[-1], [10, 5])
    assert np.allclose(np.hypot(*np.diff(pts, axis=0).T), 1.0)


def test_subgoal_selector_follows_path():
    m = HospitalMap()
    p = GlobalAStarPlanner(m)
    wp = p.simplify(p.plan(m.start, m.goals["G3"]).path_xy)
    sel = SubGoalSelector(wp, lookahead=1.5, goal_tol=0.3)
    robot_path = GlobalAStarPlanner.resample(wp, 0.1)           # gia lap robot di dung theo duong
    last_s, reached = -1.0, False
    for q in robot_path:
        sg, reached = sel.update(q)
        assert sel.s >= last_s - 1e-12, "hinh chieu bi lui"
        assert np.hypot(*(sg - q)) <= 1.5 + 1e-6, "sub-goal xa hon look-ahead"
        last_s = sel.s
    assert reached and np.allclose(sg, wp[-1])                  # cuoi duong: sub-goal = dich
    sel.reset()
    sg0, _ = sel.update(m.start)
    assert 0.5 < np.hypot(*(sg0 - np.asarray(m.start))) <= 1.5 + 1e-6


def test_plan_route_through_stops():
    m = HospitalMap()
    p = GlobalAStarPlanner(m)
    route = p.plan_route([m.start, m.goals["G1"], m.goals["G2"]])
    a, b = p.plan(m.start, m.goals["G1"]), p.plan(m.goals["G1"], m.goals["G2"])
    assert route.found and abs(route.length - (a.length + b.length)) < 1e-9
    assert np.allclose(route.path_xy[0], m.start) and np.allclose(route.path_xy[-1], m.goals["G2"])


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t()
        print(f"[OK] {t.__name__}")

    m = HospitalMap()
    world = m.with_obstacles(UNMAPPED_CARTS)
    print(f"\n{'Dich':<5}{'Heuristic':<11}{'Dai [m]':>9}{'So o mo rong':>14}{'Thoi gian':>11}")
    for name, goal in m.goals.items():
        for h in H_LIST:
            r = GlobalAStarPlanner(m, h).plan(m.start, goal)
            print(f"{name:<5}{h:<11}{r.length:>9.3f}{r.n_expanded:>14d}{r.time_s:>10.2f}s")
        print()

    print("Duong A* (octile) co di qua cac xe day phat sinh khong? (de dat xe day dung cho trong kich ban 1)")
    p = GlobalAStarPlanner(m)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    world.plot(ax, title="A* (octile): duong luoi (net manh) va waypoint sau khi rut gon (cham)")
    for (name, goal), col in zip(m.goals.items(), colors):
        r = p.plan(m.start, goal)
        wp = p.simplify(r.path_xy)
        hit = bool(world.inflated[r.path_rc[:, 0], r.path_rc[:, 1]].any())
        print(f"  Start -> {name}: {len(r.path_xy)} o -> {len(wp)} waypoint, "
              f"{'CAT xe day (A* khong biet -> robot phai tranh)' if hit else 'khong cat xe day'}")
        ax.plot(r.path_xy[:, 0], r.path_xy[:, 1], "-", color=col, lw=2.5)
        ax.plot(wp[:, 0], wp[:, 1], "w.", ms=7, mec="k")
    print("\n>>> GLOBAL_ASTAR DAT: tat ca kiem tra deu qua.")
    if matplotlib.get_backend().lower() != "agg":
        plt.tight_layout()
        plt.show()