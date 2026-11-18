"""tests/test_lidar.py -- Kiem tra sensors/lidar.py.

Chay (tu THU MUC GOC cua du an, da kich hoat .venv):
    python -m tests.test_lidar           # in bao cao + ve hinh
    python -m pytest tests/test_lidar.py -v
"""
import time

import numpy as np

from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from sensors.lidar import Lidar2D


def idx_of(lidar, rel_angle):
    """Chi so tia gan nhat voi goc tuong doi rel_angle (rad)."""
    d = np.abs(np.angle(np.exp(1j * (lidar.angles_rel - rel_angle))))
    return int(np.argmin(d))


def make():
    return HospitalMap(with_furniture=False), Lidar2D(n_rays=360, max_range=5.0, noise_std=0.0)


def test_basic_geometry():
    world, lidar = make()
    # Robot trong Nha thuoc tai (7, 16), huong +x.  Tuong ngan phia dong: x = 10.
    s = lidar.scan(world, [7.0, 16.0, 0.0])
    assert abs(s.ranges[idx_of(lidar, 0.0)] - 3.0) < 0.01                 # toi tuong x = 10
    assert abs(s.ranges[idx_of(lidar, np.pi / 2)] - 3.8) < 0.01           # toi tuong bac y = 19.8
    assert abs(s.ranges[idx_of(lidar, -np.pi / 2)] - 4.4) < 0.01           # huong -y: tuong hanh lang y = 11.6
    assert s.ranges[idx_of(lidar, np.pi)] == lidar.max_range               # huong -x: tuong o xa 6.8 m > tam 5 m
    assert not s.hit[idx_of(lidar, np.pi)] and s.hit[idx_of(lidar, 0.0)]


def test_rotation_with_heading():
    world, lidar = make()
    a = lidar.scan(world, [7.0, 16.0, 0.0])
    b = lidar.scan(world, [7.0, 16.0, np.pi / 2])                          # robot quay len phia bac
    assert abs(b.ranges[idx_of(lidar, 0.0)] - 3.8) < 0.01                  # tia truoc mat gio huong bac
    assert abs(a.ranges[idx_of(lidar, 0.0)] - 3.0) < 0.01
    # goc the gioi: tia truoc mat luon huong theo theta
    assert abs(np.angle(np.exp(1j * (b.angles[idx_of(lidar, 0.0)] - np.pi / 2)))) < 1e-9


def test_no_hit_gives_max_range_and_no_points():
    world, lidar = make()
    s = lidar.scan(world, [20.0, 10.0, 0.0])                               # giua hanh lang: tuong gan nhat 1.4 m
    assert s.hit.any()
    s2 = Lidar2D(n_rays=360, max_range=1.0).scan(world, [20.0, 10.0, 0.0])  # tam 1 m < 1.4 m
    assert not s2.hit.any() and (s2.ranges == 1.0).all() and s2.points.shape == (0, 2)


def test_points_lie_on_obstacles():
    world, lidar = make()
    s = lidar.scan(world, [7.0, 16.0, 0.3])
    p, ang = s.points, s.angles[s.hit]
    assert p.shape == (int(s.hit.sum()), 2) and len(p) > 0
    nudge = p + 0.01 * np.column_stack([np.cos(ang), np.sin(ang)])         # day nhe ra sau diem cham
    r, c = world.xy_to_rc(nudge[:, 0], nudge[:, 1])
    assert world.grid[r, c].all(), "co diem cham khong nam tren vat can"
    back = p - 0.01 * np.column_stack([np.cos(ang), np.sin(ang)])          # lui nhe ve truoc diem cham
    r, c = world.xy_to_rc(back[:, 0], back[:, 1])
    assert (~world.grid[r, c]).all(), "co diem cham nam sau vat can (xuyen qua)"


def test_circles_pedestrians():
    world, lidar = make()
    ped = np.array([[9.0, 16.0, 0.5]])                                     # nguoi tai (9,16), r = 0.5
    s = lidar.scan(world, [7.0, 16.0, 0.0], circles=ped)
    assert abs(s.ranges[idx_of(lidar, 0.0)] - 1.5) < 0.01                  # cham mat truoc cua nguoi, khong phai tuong
    # nguoi dung sau tuong -> bi tuong che
    behind = np.array([[12.0, 16.0, 0.5]])
    s2 = lidar.scan(world, [7.0, 16.0, 0.0], circles=behind)
    assert abs(s2.ranges[idx_of(lidar, 0.0)] - 3.0) < 0.01
    # nguoi o ngoai tam xa
    far = np.array([[7.0, 30.0, 0.5]])                                     # nguoi o xa, ngoai tam LiDAR
    s3 = lidar.scan(world, [7.0, 16.0, 0.0], circles=far)
    assert np.array_equal(s3.ranges, lidar.scan(world, [7.0, 16.0, 0.0]).ranges)
    # tam hinh tron nam sau lung robot -> khong anh huong tia truoc
    back = np.array([[5.0, 16.0, 0.5]])
    s4 = lidar.scan(world, [7.0, 16.0, 0.0], circles=back)
    assert abs(s4.ranges[idx_of(lidar, np.pi)] - 1.5) < 0.01
    assert abs(s4.ranges[idx_of(lidar, 0.0)] - 3.0) < 0.01


def test_unmapped_cart_visible_only_in_true_world():
    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)                           # xe day (17.0..17.6, 10.4..11.4)
    lidar = Lidar2D(noise_std=0.0)
    pose = [17.3, 9.5, np.pi / 2]                                          # robot huong bac, ngay duoi xe day
    r_world = lidar.scan(world, pose).ranges[idx_of(lidar, 0.0)]
    r_known = lidar.scan(known, pose).ranges[idx_of(lidar, 0.0)]
    assert abs(r_world - 0.9) < 0.01                                       # thay xe day o y = 10.4
    assert abs(r_known - 1.9) < 0.01                                       # ban do da biet: toi tuong bac y = 11.4


def test_noise_reproducible_and_only_on_hits():
    world = HospitalMap()
    pose = [7.0, 16.0, 0.0]
    clean = Lidar2D(noise_std=0.0).scan(world, pose)
    a = Lidar2D(noise_std=0.02, rng=np.random.default_rng(7)).scan(world, pose)
    b = Lidar2D(noise_std=0.02, rng=np.random.default_rng(7)).scan(world, pose)
    c = Lidar2D(noise_std=0.02, rng=np.random.default_rng(8)).scan(world, pose)
    assert np.array_equal(a.ranges, b.ranges) and not np.array_equal(a.ranges, c.ranges)
    assert (a.ranges[~clean.hit] == 5.0).all()                              # tia khong cham: khong them nhieu
    assert 0.01 < np.std((a.ranges - clean.ranges)[clean.hit]) < 0.04


def test_speed():
    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    lidar = Lidar2D()
    ped = np.array([[10.0 + i, 10.0, 0.25] for i in range(6)])
    t0 = time.perf_counter()
    for _ in range(100):
        lidar.scan(world, [20.0, 10.0, 0.5], circles=ped)
    ms = (time.perf_counter() - t0) / 100 * 1000
    assert ms < 100, f"1 lan quet mat {ms:.1f} ms -- qua cham"
    return ms


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    for t in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        out = t()
        print(f"[OK] {t.__name__}" + (f"   ({out:.2f} ms / lan quet)" if out is not None else ""))

    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)
    lidar = Lidar2D()
    pose = [14.5, 9.6, 0.0]
    ped = np.array([[19.0, 10.0, 0.25], [21.0, 9.3, 0.25]])
    s = lidar.scan(world, pose, circles=ped)
    print(f"\nQuet tai pose = {pose}: {int(s.hit.sum())}/{lidar.n_rays} tia cham, "
          f"khoang cach gan nhat {s.ranges.min():.2f} m")
    print("\n>>> LIDAR DAT: tat ca kiem tra deu qua.")

    if matplotlib.get_backend().lower() != "agg":
        fig, ax = plt.subplots(figsize=(11, 5.5))
        world.plot(ax, title="LiDAR (360 tia, 5 m) tren the gioi that: xe day + 2 nguoi di bo")
        for a, r in zip(s.angles, s.ranges):
            ax.plot([pose[0], pose[0] + r * np.cos(a)], [pose[1], pose[1] + r * np.sin(a)],
                    "-", color="tab:blue", lw=0.3, alpha=0.5)
        ax.plot(s.points[:, 0], s.points[:, 1], "r.", ms=3)
        for cx, cy, r in ped:
            ax.add_patch(plt.Circle((cx, cy), r, color="tab:orange", alpha=0.8))
        ax.plot(pose[0], pose[1], "k^", ms=10)
        ax.set_xlim(5, 35)
        ax.set_ylim(5, 15)
        plt.tight_layout()
        plt.show()