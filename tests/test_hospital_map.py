"""tests/test_hospital_map.py -- Kiem tra env/hospital_map.py.

Chay (tu THU MUC GOC cua du an, da kich hoat .venv):
    python -m tests.test_hospital_map        # in bao cao chi tiet + luu hinh
    python -m pytest tests/test_hospital_map.py -v
"""
import numpy as np

import config
from env.hospital_map import (CORRIDOR_WIDTH, DOOR_WIDTH, UNMAPPED_CARTS, HospitalMap)


def test_shape_and_boundary():
    m = HospitalMap()
    assert m.grid.shape == (200, 400)                       # (hang, cot) = (y, x)
    assert m.grid[0, :].all() and m.grid[-1, :].all() and m.grid[:, 0].all() and m.grid[:, -1].all()
    assert (m.inflated >= m.grid).all()                     # inflated bao ham grid


def test_inflation_radius_matches_config():
    m = HospitalMap()
    assert m.inflate_radius >= config.ROBOT_RADIUS          # phong it nhat bang ban kinh robot
    assert m.inflate_cells * m.res >= m.inflate_radius - 1e-9


def test_start_and_goals_free_and_reachable():
    m = HospitalMap()
    for name, p in [("START", m.start)] + list(m.goals.items()):
        assert m.is_free(*p), f"{name} nam trong vung bi chan"
    for name, p in m.goals.items():
        assert m.reachable(m.start, p), f"Khong co duong tu START toi {name}"


def test_door_and_corridor_clearance():
    m = HospitalMap()
    k = m.inflate_cells
    door_cells = round(DOOR_WIDTH / m.res)
    for (xd, yd) in m.doors:
        r, c0 = m.xy_to_rc(xd - 1.0, yd)
        _, c1 = m.xy_to_rc(xd + 1.0, yd)
        n_free = int((~m.inflated[r, c0:c1 + 1]).sum())
        assert n_free == door_cells - 2 * k, f"Cua x={xd}: thong {n_free} o, ky vong {door_cells - 2 * k}"
        assert n_free >= 2                                   # robot (coi la diem) phai di lot
    # hanh lang tai x = 8 m, khong co vat can
    ys, yn = m.corridor_y
    r0, c = m.xy_to_rc(8.0, ys + 0.05)
    r1, _ = m.xy_to_rc(8.0, yn - 0.05)
    n_cor = int((~m.inflated[r0:r1 + 1, c]).sum())
    assert n_cor == round(CORRIDOR_WIDTH / m.res) - 2 * k


def test_with_obstacles_returns_copy_and_keeps_connectivity():
    known = HospitalMap()
    n_before = int(known.grid.sum())
    world = known.with_obstacles(UNMAPPED_CARTS)
    assert int(known.grid.sum()) == n_before                 # ban do da biet KHONG bi sua
    assert int(world.grid.sum()) > n_before                  # the gioi that co them vat can
    assert not known.inflated[world.grid & ~known.grid].all()  # A* (ban do biet) khong thay xe day
    for p in world.goals.values():
        assert world.reachable(world.start, p), "Xe day lam bit duong toi mot G"


def test_coordinate_roundtrip():
    m = HospitalMap()
    r, c = m.xy_to_rc(11.5, 11.45)
    assert (r, c) == (114, 115)                              # 0-based: 11.45/0.1 -> hang 114; 11.5/0.1 -> cot 115
    x, y = m.rc_to_xy(r, c)
    assert abs(x - 11.55) < 1e-9 and abs(y - 11.45) < 1e-9
    rr, cc = m.xy_to_rc(np.array([0.0, 39.99, 100.0]), np.array([0.0, 19.99, -3.0]))
    assert list(cc) == [0, 399, 399] and list(rr) == [0, 199, 0]   # ep vao bien


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[OK] {t.__name__}")

    m = HospitalMap()
    k = m.inflate_cells
    print(f"\nLuoi {m.n_rows} hang x {m.n_cols} cot, res = {m.res} m")
    print(f"Phong vat can: {m.inflate_radius:.2f} m = {k} o "
          f"(ROBOT_RADIUS {config.ROBOT_RADIUS} + bien an toan {m.inflate_radius - config.ROBOT_RADIUS:.2f})")
    print(f"Cua {DOOR_WIDTH} m -> sau phong con thong {(round(DOOR_WIDTH / m.res) - 2 * k) * m.res:.2f} m")
    print(f"Hanh lang {CORRIDOR_WIDTH} m -> sau phong con thong {(round(CORRIDOR_WIDTH / m.res) - 2 * k) * m.res:.2f} m")
    free = int((~m.inflated).sum())
    print(f"O di duoc: {free}/{m.grid.size} ({100 * free / m.grid.size:.1f}%)")
    print("\n>>> HOSPITAL_MAP DAT: tat ca kiem tra deu qua.")

    if matplotlib.get_backend().lower() != "agg":
        world = m.with_obstacles(UNMAPPED_CARTS)
        fig, axes = plt.subplots(2, 1, figsize=(11, 10))
        m.plot(axes[0], title="Ban do DA BIET (A* dung)")
        world.plot(axes[1], title="The gioi THAT (them vat can phat sinh - kich ban 1)")
        plt.tight_layout()
        plt.show()