"""sensors/lidar.py -- LiDAR 2D gia lap bang raycasting (vector hoa bang numpy).

Nguoi phu trach: TV1.

LiDAR do tren THE GIOI THAT (``world.grid`` = vat can that, KHONG phai ``inflated``), gom:
  * vat can tinh tren luoi (tuong, noi that, xe day phat sinh...),
  * vat can dong dang hinh tron (nguoi di bo): mang (M, 3) = [cx, cy, ban_kinh].

Dau ra ``LidarScan``:
  ranges      (n,)    khoang cach moi tia [m]; tia khong cham gi = max_range
  angles_rel  (n,)    goc cua tia SO VOI huong robot [rad], tu -pi (khong gom +pi)
  angles      (n,)    goc cua tia trong he the gioi [rad]  (= theta + angles_rel)
  hit         (n,)    True neu tia cham vat can (ranges < max_range)
  points      (K, 2)  toa do THE GIOI cua cac diem cham (K = so tia cham) -- day la mang
                      ``points`` trong hop dong ``LocalPlanner.compute_control``.

Do chinh xac: lay mau doc tia moi res/2 (tuong day >= 0.2 m nen khong "xuyen" qua), roi chia doi
6 lan de tinh diem cham chinh xac toi ~1.5 mm.

Quy uoc: toa do met, goc radian, theta = 0 la huong +x, duong nguoc chieu kim dong ho.
Tham so doc tu config.py neu co (LIDAR_N_RAYS, LIDAR_RANGE, LIDAR_NOISE_STD), khong thi dung
mac dinh 360 tia, 5.0 m, 0.0 m (khong nhieu).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config

N_RAYS = getattr(config, "LIDAR_N_RAYS", 360)
RANGE = getattr(config, "LIDAR_RANGE", 5.0)              # [m] tam xa toi da
NOISE_STD = getattr(config, "LIDAR_NOISE_STD", 0.0)      # [m] do lech chuan nhieu do (0 = khong nhieu)


@dataclass
class LidarScan:
    ranges: np.ndarray
    angles_rel: np.ndarray
    angles: np.ndarray
    hit: np.ndarray
    origin: np.ndarray                                    # (2,) vi tri robot luc quet

    @property
    def points(self) -> np.ndarray:
        """(K, 2) toa do the gioi cua cac diem cham. K co the bang 0."""
        a = self.angles[self.hit]
        r = self.ranges[self.hit]
        return np.column_stack([self.origin[0] + r * np.cos(a), self.origin[1] + r * np.sin(a)])


class Lidar2D:
    """LiDAR 2D 360 do.

    Dung::

        lidar = Lidar2D(rng=np.random.default_rng(seed))
        scan = lidar.scan(world, pose=[x, y, theta], circles=pedestrians_array)
        scan.points   # (K, 2) -> dua cho LocalPlanner
        scan.ranges   # (n,)   -> VFH co the dung truc tiep
    """

    def __init__(self, n_rays: int | None = None, max_range: float | None = None,
                 noise_std: float | None = None, rng: np.random.Generator | None = None):
        self.n_rays = int(N_RAYS if n_rays is None else n_rays)
        self.max_range = float(RANGE if max_range is None else max_range)
        self.noise_std = float(NOISE_STD if noise_std is None else noise_std)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.angles_rel = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)
        self._n_bisect = 6

    # ------------------------------------------------------------------ noi bo
    @staticmethod
    def _occupied(world, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """True neu diem (px, py) nam trong vat can that (ngoai ban do cung tinh la vat can)."""
        inside = (px >= 0) & (px < world.width) & (py >= 0) & (py < world.height)
        c = np.clip(np.floor(px / world.res).astype(int), 0, world.n_cols - 1)
        r = np.clip(np.floor(py / world.res).astype(int), 0, world.n_rows - 1)
        return world.grid[r, c] | ~inside

    def _ranges_to_grid(self, world, x: float, y: float, cs: np.ndarray, sn: np.ndarray) -> np.ndarray:
        step = world.res / 2.0
        d = np.arange(step, self.max_range + 1e-9, step)                     # (m,)
        d[-1] = self.max_range
        occ = self._occupied(world, x + d[None, :] * cs[:, None], y + d[None, :] * sn[:, None])  # (n, m)
        any_hit = occ.any(axis=1)
        first = np.argmax(occ, axis=1)                                        # chi so mau dau tien bi chan
        hi = d[first]
        lo = np.where(first > 0, d[np.maximum(first - 1, 0)], 0.0)
        for _ in range(self._n_bisect):                                       # chia doi de tinh diem cham
            mid = 0.5 * (lo + hi)
            blocked = self._occupied(world, x + mid * cs, y + mid * sn)
            hi = np.where(blocked, mid, hi)
            lo = np.where(blocked, lo, mid)
        return np.where(any_hit, hi, self.max_range)

    def _ranges_to_circles(self, x: float, y: float, cs: np.ndarray, sn: np.ndarray, circles) -> np.ndarray:
        """Khoang cach tia -> hinh tron gan nhat (giai tich). Khong cham: +inf."""
        c = np.asarray(circles, dtype=float).reshape(-1, 3)
        if len(c) == 0:
            return np.full(len(cs), np.inf)
        vx, vy = c[None, :, 0] - x, c[None, :, 1] - y                         # (1, M)
        t_c = vx * cs[:, None] + vy * sn[:, None]                             # (n, M) hinh chieu len tia
        d2 = (vx ** 2 + vy ** 2) - t_c ** 2                                   # binh phuong khoang cach tia-tam
        r2 = c[None, :, 2] ** 2
        t = t_c - np.sqrt(np.maximum(r2 - d2, 0.0))
        valid = (d2 <= r2) & (t >= 0.0) & (t <= self.max_range)
        return np.where(valid, t, np.inf).min(axis=1)

    # ------------------------------------------------------------------ giao dien chinh
    def scan(self, world, pose, circles=None) -> LidarScan:
        """Quet 360 do tai pose = [x, y, theta] tren `world` (HospitalMap THAT) + cac hinh tron."""
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        ang = th + self.angles_rel
        cs, sn = np.cos(ang), np.sin(ang)

        rng = self._ranges_to_grid(world, x, y, cs, sn)
        if circles is not None and len(circles):
            rng = np.minimum(rng, self._ranges_to_circles(x, y, cs, sn, circles))
        hit = rng < self.max_range - 1e-9
        rng = np.minimum(rng, self.max_range)
        if self.noise_std > 0.0:
            noisy = rng + self.rng.normal(0.0, self.noise_std, rng.shape)
            rng = np.where(hit, np.clip(noisy, 0.0, self.max_range), rng)     # chi them nhieu cho tia co cham
        return LidarScan(ranges=rng, angles_rel=self.angles_rel.copy(), angles=ang,
                         hit=hit, origin=np.array([x, y]))