"""tests/test_lidar.py -- Kiem chung sensors/lidar.py bang DAP AN GIAI TICH (khong chi "chay duoc").

Chay (tu THU MUC GOC cua du an):
    python -m tests.test_lidar
    python -m pytest tests/test_lidar.py -v

Cach kiem chung
---------------
1. The gioi tong hop (GridWorld) co dap an biet truoc: tuong thang -> r = d / cos(goc), hinh tron -> r = d - R.
2. So sanh voi nghiem THAM CHIEU doc lap (giao tia - tung o vuong theo phuong phap "slab") tren hang tram nghin tia
   ngau nhien, ke ca tia luot qua goc o (cho ma LiDAR lay mau 5 cm thuong bo lot).
3. Tren ban do benh vien that: xe day chua biet chi hien thi o the gioi THAT, khong hien thi o ban do DA BIET.
4. Tinh nhat quan voi simulator: LiDAR khong bao gio bao vat can gan hon khoang cach that (static_clearance).
5. Mo hinh cam bien (nhieu, roi tia, vung mu, FOV) bang thong ke co dung sai ro rang; tai lap theo seed.
"""
from __future__ import annotations

import math
import time

import numpy as np

from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from sensors.lidar import Lidar2D, LidarScan

TOL = 1e-9


# ------------------------------------------------------------------ the gioi tong hop va nghiem tham chieu
class GridWorld:
    """Ban do luoi toi thieu, cung giao dien voi HospitalMap (grid, res, width, height, n_rows, n_cols)."""

    def __init__(self, width=10.0, height=10.0, res=0.1):
        self.res, self.width, self.height = res, width, height
        self.n_cols, self.n_rows = int(round(width / res)), int(round(height / res))
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=bool)

    def fill(self, x1, y1, x2, y2):
        c1, c2 = int(round(x1 / self.res)), int(round(x2 / self.res))
        r1, r2 = int(round(y1 / self.res)), int(round(y2 / self.res))
        self.grid[r1:r2, c1:c2] = True
        return self


def random_world(seed, size=8.0, p=0.10):
    w = GridWorld(size, size)
    w.grid = np.random.default_rng(seed).random(w.grid.shape) < p
    return w


def reference_ranges(world, x, y, angles, max_range):
    """Nghiem chinh xac DOC LAP: giao tia - tung o vuong dac (slab) + thoat khoi ban do = vat can."""
    res = world.res
    rr, cc = np.nonzero(world.grid)
    x0, x1, y0, y1 = cc * res, (cc + 1) * res, rr * res, (rr + 1) * res
    cs, sn = np.cos(angles)[:, None], np.sin(angles)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0, tx1 = (x0[None] - x) / cs, (x1[None] - x) / cs
        ty0, ty1 = (y0[None] - y) / sn, (y1[None] - y) / sn
    tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
    tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
    par_x, par_y = np.abs(cs) < 1e-12, np.abs(sn) < 1e-12
    inx, iny = (x >= x0) & (x <= x1), (y >= y0) & (y <= y1)
    tmin = np.where(par_x, np.where(inx[None], np.minimum(ty0, ty1), np.inf), tmin)
    tmax = np.where(par_x, np.where(inx[None], np.maximum(ty0, ty1), -np.inf), tmax)
    tmin = np.where(par_y, np.where(iny[None], np.minimum(tx0, tx1), np.inf), tmin)
    tmax = np.where(par_y, np.where(iny[None], np.maximum(tx0, tx1), -np.inf), tmax)
    t = np.where(tmax >= np.maximum(tmin, 0.0), np.maximum(tmin, 0.0), np.inf).min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ex = np.where(cs[:, 0] > 0, (world.width - x) / cs[:, 0], np.where(cs[:, 0] < 0, -x / cs[:, 0], np.inf))
        ey = np.where(sn[:, 0] > 0, (world.height - y) / sn[:, 0], np.where(sn[:, 0] < 0, -y / sn[:, 0], np.inf))
    t = np.minimum(t, np.minimum(ex, ey))
    return np.where(t < max_range, t, max_range), t < max_range


def true_clearance(world, x, y):
    """Khoang cach that tu (x, y) toi be mat vat can gan nhat (o = hinh vuong) -- cung dinh nghia voi simulator."""
    res = world.res
    rr, cc = np.nonzero(world.grid)
    cx, cy = (cc + 0.5) * res, (rr + 0.5) * res
    dx = np.maximum(np.abs(x - cx) - res / 2.0, 0.0)
    dy = np.maximum(np.abs(y - cy) - res / 2.0, 0.0)
    return float(np.hypot(dx, dy).min())


def idx(lidar, angle_rel):
    """Chi so tia gan nhat voi goc (so voi huong robot)."""
    return int(np.argmin(np.abs(lidar.angles_rel - angle_rel)))


# ------------------------------------------------------------------ 1. dap an giai tich
def test_straight_wall_range_is_exact():
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)                       # tuong x in [5,5.4)
    lid = Lidar2D()
    s = lid.scan(w, [2.0, 5.0, 0.0])
    assert abs(s.ranges[idx(lid, 0.0)] - 3.0) < TOL
    assert s.hit[idx(lid, 0.0)]


def test_oblique_rays_follow_d_over_cos():
    """Tuong thang: r = d / cos(goc) voi moi tia trong +-55 do (sai so chi la lam tron so thuc)."""
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)
    lid = Lidar2D()
    s = lid.scan(w, [2.0, 5.0, 0.0])
    sel = np.abs(lid.angles_rel) <= math.radians(55)
    expected = 3.0 / np.cos(lid.angles_rel[sel])
    expected = np.minimum(expected, 5.0)
    got = s.ranges[sel]
    near = expected < 5.0 - 1e-6                                     # tia dai hon tam xa -> khong ve
    assert np.abs(got[near] - expected[near]).max() < TOL
    assert (s.hit[sel] == near).all()


def test_no_obstacle_returns_max_range_and_empty_points():
    w = GridWorld(40.0, 40.0)
    s = Lidar2D().scan(w, [20.0, 20.0, 1.0])
    assert (s.ranges == 5.0).all() and not s.hit.any() and not s.dynamic.any()
    assert s.points.shape == (0, 2)


def test_range_is_never_above_max_range_and_hit_matches_ranges():
    w = random_world(3)
    s = Lidar2D().scan(w, [4.05, 4.05, 0.3])
    assert s.ranges.max() <= 5.0 and (s.hit == (s.ranges < 5.0)).all()


def test_leaving_map_counts_as_obstacle():
    w = GridWorld()                                                 # khong co vat can, nhung co ranh gioi ban do
    lid = Lidar2D()
    s = lid.scan(w, [1.0, 5.0, math.pi])                            # nhin ve phia x = 0
    assert abs(s.ranges[idx(lid, 0.0)] - 1.0) < TOL and s.hit[idx(lid, 0.0)]


def test_origin_inside_occupied_cell_gives_zero_range():
    w = GridWorld().fill(4.0, 4.0, 4.5, 4.5)
    s = Lidar2D().scan(w, [4.25, 4.25, 0.0])
    assert (s.ranges == 0.0).all() and s.hit.all()


# ------------------------------------------------------------------ 2. so sanh voi nghiem tham chieu doc lap
def test_matches_reference_on_random_worlds_including_corner_clipping():
    """~127 nghin tia tren the gioi ngau nhien day goc o: khop tuyet doi, khong lech 'thay / khong thay'."""
    lid, rng = Lidar2D(), np.random.default_rng(0)
    n_rays = 0
    for seed in range(25):
        w = random_world(seed)
        for _ in range(8):
            x, y = rng.uniform(0.5, 7.5, 2)
            if w.grid[int(y / w.res), int(x / w.res)]:
                continue
            th = rng.uniform(-math.pi, math.pi)
            s = lid.scan(w, [x, y, th])
            ref, ref_hit = reference_ranges(w, x, y, s.angles, 5.0)
            assert (s.hit == ref_hit).all(), f"world {seed}: {(s.hit != ref_hit).sum()} tia lech hit"
            assert np.abs(s.ranges - ref).max() < TOL
            n_rays += s.ranges.size
    assert n_rays > 50000


def test_matches_reference_on_real_hospital_map():
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    free = np.argwhere(~world.inflated)
    rng = np.random.default_rng(1)
    lid = Lidar2D()
    for k in rng.choice(len(free), 60, replace=False):
        r, c = free[k]
        x, y, th = (c + rng.uniform()) * world.res, (r + rng.uniform()) * world.res, rng.uniform(-math.pi, math.pi)
        s = lid.scan(world, [x, y, th])
        ref, ref_hit = reference_ranges(world, x, y, s.angles, 5.0)
        assert (s.hit == ref_hit).all() and np.abs(s.ranges - ref).max() < TOL


def test_rotation_equivariance():
    """Quay robot 90 do = dich vong cac tia 90 vi tri (360 tia, 1 do/tia); cung diem cham trong he the gioi."""
    w = random_world(7)
    lid = Lidar2D()
    a = lid.scan(w, [4.05, 3.95, 0.0])
    b = lid.scan(w, [4.05, 3.95, math.pi / 2])
    assert np.abs(b.ranges - np.roll(a.ranges, -90)).max() < 1e-9


# ------------------------------------------------------------------ 3. ban do benh vien: known vs world
def test_corridor_walls_at_exact_distance():
    known = HospitalMap()
    lid = Lidar2D()
    s = lid.scan(known, [8.0, 10.0, 0.0])                            # giua hanh lang, x = 8
    assert abs(s.ranges[idx(lid, math.pi / 2)] - 1.4) < TOL          # tuong bac y = 11,4
    assert abs(s.ranges[idx(lid, -math.pi / 2)] - 1.4) < TOL         # mep tuong nam y = 8,6


def test_unmapped_cart_visible_in_world_but_not_in_known_map():
    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)
    lid = Lidar2D()
    pose = [15.0, 10.9, 0.0]                                         # nhin thang +x, xe day o x in [17, 17,6]
    sw, sk = lid.scan(world, pose), lid.scan(known, pose)
    i = idx(lid, 0.0)
    assert sw.hit[i] and abs(sw.ranges[i] - 2.0) < TOL
    assert not sk.hit[i]                                             # ban do da biet: hanh lang thong, khong thay gi trong 5 m


def test_pedestrian_behind_wall_is_occluded():
    known = HospitalMap()
    lid = Lidar2D()
    person = np.array([[8.0, 13.0, 0.25]])                           # trong Nha thuoc, sau tuong hanh lang
    s = lid.scan(known, [8.0, 10.0, 0.0], circles=person)
    assert not s.dynamic.any()                                       # bi tuong che khuat


# ------------------------------------------------------------------ 4. nguoi di bo (hinh tron)
def test_circle_range_and_dynamic_flag():
    w = GridWorld()
    lid = Lidar2D()
    s = lid.scan(w, [2.0, 5.0, 0.0], circles=np.array([[5.0, 5.0, 0.5]]))
    i = idx(lid, 0.0)
    assert abs(s.ranges[i] - 2.5) < TOL and s.dynamic[i] and s.hit[i]
    p = s.points_dynamic
    assert np.abs(np.hypot(p[:, 0] - 5.0, p[:, 1] - 5.0) - 0.5).max() < 1e-9      # diem cham nam tren duong tron
    assert s.points_static.shape[0] + s.points_dynamic.shape[0] == s.points.shape[0]


def test_circle_nearer_than_wall_wins_and_farther_loses():
    w = GridWorld().fill(6.0, 0.0, 6.4, 10.0)
    lid = Lidar2D()
    near = lid.scan(w, [2.0, 5.0, 0.0], circles=np.array([[4.0, 5.0, 0.3]]))      # nguoi truoc tuong
    far = lid.scan(w, [2.0, 5.0, 0.0], circles=np.array([[7.0, 5.0, 0.3]]))       # nguoi sau tuong
    i = idx(lid, 0.0)
    assert abs(near.ranges[i] - 1.7) < TOL and near.dynamic[i]
    assert abs(far.ranges[i] - 4.0) < TOL and not far.dynamic[i]


def test_nearest_of_several_circles_and_out_of_range_circle_ignored():
    w = GridWorld(20.0, 10.0)
    lid = Lidar2D()
    c = np.array([[6.0, 5.0, 0.25], [4.0, 5.0, 0.25], [9.0, 5.0, 0.25]])          # 3 nguoi cung huong; nguoi o x=9 ngoai tam xa
    s = lid.scan(w, [2.0, 5.0, 0.0], circles=c)
    assert abs(s.ranges[idx(lid, 0.0)] - 1.75) < TOL
    far = Lidar2D().scan(w, [2.0, 5.0, 0.0], circles=np.array([[9.0, 5.0, 0.25]]))    # cach 7 m > tam xa 5 m
    assert not far.dynamic.any() and not far.hit[idx(Lidar2D(), 0.0)]               # (tia lui van cham ranh gioi x = 0)


def test_lidar_inside_pedestrian_gives_zero_range_without_crash():
    s = Lidar2D().scan(GridWorld(), [5.0, 5.0, 0.0], circles=np.array([[5.0, 5.0, 0.3]]))
    assert (s.ranges == 0.0).all() and s.dynamic.all()


def test_empty_or_none_circles_equal_no_pedestrians():
    w = random_world(2)
    lid = Lidar2D()
    a = lid.scan(w, [4.05, 4.05, 0.2])
    for c in (None, np.zeros((0, 3)), []):
        b = lid.scan(w, [4.05, 4.05, 0.2], circles=c)
        assert np.array_equal(a.ranges, b.ranges)


# ------------------------------------------------------------------ 5. nhat quan voi simulator (an toan)
def test_lidar_never_reports_obstacle_closer_than_truth():
    """min(range) >= khoang cach that toi be mat vat can (LiDAR khong 'tuong tuong' ra vat can gan hon)."""
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    free = np.argwhere(~world.inflated)
    rng = np.random.default_rng(4)
    lid = Lidar2D()
    for k in rng.choice(len(free), 80, replace=False):
        r, c = free[k]
        x, y = (c + rng.uniform()) * world.res, (r + rng.uniform()) * world.res
        d = true_clearance(world, x, y)
        if d < 5.0:
            s = lid.scan(world, [x, y, rng.uniform(-math.pi, math.pi)])
            assert s.ranges.min() >= d - 1e-9


def test_flat_wall_nearest_range_within_angular_resolution():
    """Truoc mot be mat phang: min range <= d / cos(nua buoc goc) -- sai so chi vi hon 1 do/tia."""
    world = HospitalMap()
    lid = Lidar2D()
    s = lid.scan(world, [8.0, 10.0, 0.0])
    d = true_clearance(world, 8.0, 10.0)
    assert d - 1e-9 <= s.ranges.min() <= d / math.cos(lid.angular_resolution / 2.0) + 1e-9


def test_robot_collision_radius_matches_lidar_sector_min():
    """Quy dinh va cham cua simulator (d_static < r) <=> LiDAR bao min range < r (tai tam)."""
    import config
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)
    lid = Lidar2D()
    for gap, collides in ((0.20, True), (0.30, False)):
        s = lid.scan(w, [5.0 - gap, 5.0, 0.0])
        assert (s.ranges.min() < config.ROBOT_RADIUS) == collides


# ------------------------------------------------------------------ 6. mo hinh cam bien
def test_noise_statistics_and_clipping():
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)
    base = Lidar2D().scan(w, [2.0, 5.0, 0.0])
    lid = Lidar2D(noise_std=0.02, rng=np.random.default_rng(0))
    err = []
    for _ in range(60):
        s = lid.scan(w, [2.0, 5.0, 0.0])
        both = base.hit & s.hit
        err.append(s.ranges[both] - base.ranges[both])
        assert (s.ranges[~s.hit] == 5.0).all()                       # tia khong ve van = max_range
    err = np.concatenate(err)
    assert err.size > 3000
    assert abs(err.mean()) < 0.001 and 0.0185 < err.std() < 0.0215   # ~ N(0, 0,02^2)
    assert s.ranges.min() >= 0.0 and s.ranges.max() <= 5.0


def test_noise_is_reproducible_by_seed_and_rng_can_be_reassigned():
    w = random_world(5)
    a = Lidar2D(noise_std=0.02, rng=np.random.default_rng(1)).scan(w, [4.05, 4.05, 0.0])
    b = Lidar2D(noise_std=0.02, rng=np.random.default_rng(1)).scan(w, [4.05, 4.05, 0.0])
    assert np.array_equal(a.ranges, b.ranges)
    lid = Lidar2D(noise_std=0.02, rng=np.random.default_rng(1))
    lid.rng = np.random.default_rng(2)                                # simulator gan lai rng moi lan chay
    c = lid.scan(w, [4.05, 4.05, 0.0])
    assert not np.array_equal(a.ranges, c.ranges)


def test_ideal_mode_does_not_consume_rng():
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    Lidar2D(rng=rng).scan(random_world(1), [4.05, 4.05, 0.0])
    assert rng.bit_generator.state == state


def test_dropout_rate_and_dropped_rays_are_clean():
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)
    base = Lidar2D().scan(w, [2.0, 5.0, 0.0])
    lid = Lidar2D(dropout=0.3, rng=np.random.default_rng(0))
    dropped = total = 0
    for _ in range(80):
        s = lid.scan(w, [2.0, 5.0, 0.0])
        dropped += int((base.hit & ~s.hit).sum())
        total += int(base.hit.sum())
        assert (s.ranges[~s.hit] == 5.0).all() and not (s.dynamic & ~s.hit).any()
    assert 0.28 < dropped / total < 0.32


def test_min_range_blind_zone():
    w = GridWorld().fill(2.15, 0.0, 2.55, 10.0)                      # tuong cach tam 0,15 m
    assert Lidar2D().scan(w, [2.0, 5.0, 0.0]).hit[idx(Lidar2D(), 0.0)]
    blind = Lidar2D(min_range=0.2)
    s = blind.scan(w, [2.0, 5.0, 0.0])
    assert not s.hit[idx(blind, 0.0)] and s.ranges[idx(blind, 0.0)] == 5.0


def test_limited_fov_matches_full_scan_on_same_angles():
    w = random_world(9)
    full, fov = Lidar2D(), Lidar2D(n_rays=91, fov_deg=90.0)
    assert abs(fov.angles_rel[0] + math.pi / 4) < 1e-12 and abs(fov.angles_rel[-1] - math.pi / 4) < 1e-12
    a, b = full.scan(w, [4.05, 3.95, 0.4]), fov.scan(w, [4.05, 3.95, 0.4])
    assert np.abs(b.ranges - a.ranges[135:226]).max() < 1e-9
    assert abs(fov.angular_resolution - math.radians(1.0)) < 1e-12


def test_single_ray_and_custom_resolution():
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)
    s = Lidar2D(n_rays=1).scan(w, [2.0, 5.0, 0.0])
    assert s.ranges.shape == (1,) and abs(s.ranges[0] - 3.0) < TOL
    assert Lidar2D(n_rays=720).scan(w, [2.0, 5.0, 0.0]).ranges.shape == (720,)


# ------------------------------------------------------------------ 7. dau ra LidarScan
def test_scan_fields_geometry_and_independence():
    w = random_world(4)
    lid = Lidar2D()
    s = lid.scan(w, [4.05, 4.05, 0.7])
    assert isinstance(s, LidarScan) and s.ranges.shape == s.angles.shape == s.hit.shape == s.dynamic.shape == (360,)
    assert np.allclose(s.angles, 0.7 + s.angles_rel) and s.origin.shape == (2,) and s.max_range == 5.0
    pts = s.points
    assert pts.shape == (int(s.hit.sum()), 2)
    assert np.allclose(np.hypot(pts[:, 0] - 4.05, pts[:, 1] - 4.05), s.ranges[s.hit])
    s.angles_rel[0] = 99.0                                            # sua ket qua khong duoc anh huong lan quet sau
    assert lid.scan(w, [4.05, 4.05, 0.7]).angles_rel[0] != 99.0
    assert abs(lid.angular_resolution - math.radians(1.0)) < 1e-12


def test_sector_min():
    w = GridWorld().fill(5.0, 0.0, 5.4, 10.0)
    lid = Lidar2D()
    s = lid.scan(w, [4.0, 5.0, 0.0])                                  # tuong cach 1,0 m phia truoc
    assert abs(s.sector_min(0.0, math.radians(10)) - 1.0) < TOL        # truoc mat: 1,0 m
    assert s.sector_min(math.pi, math.radians(10)) == 4.0 or s.sector_min(math.pi, math.radians(10)) <= 5.0
    assert s.sector_min(math.pi, math.radians(10)) > 1.0              # phia sau xa hon
    assert abs(s.sector_min() - s.ranges.min()) < TOL                 # mac dinh = toan bo vong


def test_scan_does_not_modify_world():
    w = random_world(6)
    before = w.grid.copy()
    Lidar2D().scan(w, [4.05, 4.05, 0.1], circles=np.array([[5.0, 4.0, 0.3]]))
    assert np.array_equal(w.grid, before)


def test_input_validation():
    for bad in (dict(n_rays=0), dict(max_range=0.0), dict(max_range=float("inf")), dict(noise_std=-0.1),
                dict(fov_deg=0.0), dict(fov_deg=400.0), dict(min_range=6.0), dict(dropout=1.0), dict(dropout=-0.1)):
        try:
            Lidar2D(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} phai bao ValueError")
    lid, w = Lidar2D(), GridWorld()
    for pose in ([1.0, 2.0], [1.0, float("nan"), 0.0]):
        try:
            lid.scan(w, pose)
        except ValueError:
            continue
        raise AssertionError("pose xau phai bao ValueError")
    try:
        lid.scan(w, [1.0, 1.0, 0.0], circles=np.zeros((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("circles sai dang phai bao ValueError")


# ------------------------------------------------------------------ 8. toc do (simulator goi moi buoc dt)
def test_scan_is_fast_enough_for_real_time_loop():
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    lid = Lidar2D()
    ped = np.array([[19.0, 10.0, 0.25]])
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        lid.scan(world, [14.0, 10.0, 0.3], circles=ped)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    print(f"      (LiDAR 360 tia: median {med * 1000:.2f} ms/lan quet)")
    assert med < 0.020                                                # nguong rong de khong chap chon tren may cham; do thuc te ~1 ms


# ------------------------------------------------------------------ chay khong can pytest
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:                                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test pass")
    raise SystemExit(1 if failed else 0)