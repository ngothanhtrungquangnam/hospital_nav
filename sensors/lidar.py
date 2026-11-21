"""sensors/lidar.py -- LiDAR 2D gia lap: raycasting CHINH XAC tren luoi + giao tia-duong tron + mo hinh cam bien.

Nguoi phu trach: TV1.

Vi tri trong he thong
---------------------
LiDAR do tren THE GIOI THAT (``world.grid`` = vat can that, KHONG phai ``inflated``), gom:
  * vat can tinh tren luoi (tuong, noi that, xe day phat sinh...)   -> giao tia - o vuong,
  * vat can dong dang hinh tron (nguoi di bo): mang (M, 3) = [cx, cy, ban_kinh]   -> giao tia - duong tron.
Planner chi nhan ``scan.points``; no KHONG BAO GIO duoc cham vao ``known_map`` (xem simulator.py).

Thuat toan (vi sao chinh xac)
-----------------------------
Moi o luoi la mot HINH VUONG DAC [c*res, (c+1)*res) x [r*res, (r+1)*res) -- cung quy uoc voi simulator
(``static_clearance`` do toi be mat hinh vuong) nen "LiDAR thay vat can" <=> "robot se cham vat can".

Tia  p(t) = o + t*d  (|d| = 1).  Moi tia cat cac duong luoi doc  x = i*res  tai  t = (i*res - ox)/dx  va cac duong
luoi ngang  y = j*res  tai  t = (j*res - oy)/dy.  Sap xep toan bo cac t  ->  tia di qua lan luot cac o; diem GIUA moi
doan [t_k, t_k+1] nam gon trong dung mot o.  Tra cuu o do trong ``grid``; doan dau tien co o dac cho diem cham
t = t_k (tuc la thoi diem tia VAO o).  Day la duyet luoi kieu Amanatides-Woo, vector hoa theo tat ca cac tia:
  * khong lay mau  ->  khong the "xuyen" qua goc o mong (ban cu lay mau 5 cm bo lot ~0,2% tia tren ban do that),
  * sai so chi la lam tron so thuc (~1e-15 m), khong phai sai so thuat toan,
  * chi phi O(n_tia * (max_range/res)) khong vong lap Python.
Ngoai ban do coi nhu vat can (tia di ra ngoai se "cham" ranh gioi ban do).  Neu tam LiDAR nam trong o dac -> range 0.

Giao tia - duong tron la nghiem giai tich cua phuong trinh bac hai; nguoi di bo che khuat va bi che khuat boi
tuong dung (lay vat gan hon).  Neu tam LiDAR nam trong nguoi di bo -> range 0 (da cham).

Mo hinh cam bien (mac dinh LY TUONG de giu nguyen ket qua baseline; bat tung thanh phan khi can kich ban thuc te)
  fov_deg     goc quet (360 = vong tron day du; nho hon thi doi xung quanh huong robot)
  min_range   vung mu gan: vat can gan hon khong duoc bao ve (nhu RPLidar A1 ~0,15 m)
  noise_std   nhieu do Gaussian cong tinh [m], chi cho cac tia co cham
  dropout     xac suat mot tia co cham bi roi (khong ve) -- mo phong be mat toi/bong
Thu tu: lay nghiem hinh hoc -> vung mu -> roi tia -> nhieu -> cat [0, max_range] (qua tam xa => khong ve).
Cac nguon ngau nhien chi duoc rut khi tham so > 0, nen che do ly tuong khong tieu thu rng.

Dau ra ``LidarScan`` (tuong thich ban cu: draw_scan, planner, simulator khong phai sua)
  ranges      (n,)   khoang cach moi tia [m]; tia khong ve = max_range
  angles_rel  (n,)   goc tia so voi huong robot [rad]; 360 do: tu -pi (khong gom +pi)
  angles      (n,)   goc tia trong he the gioi [rad] (= theta + angles_rel)
  hit         (n,)   True neu tia co ve (ranges < max_range)
  origin      (2,)   vi tri LiDAR (= tam robot)
  max_range   float  tam xa toi da
  dynamic     (n,)   True neu tia ve vao NGUOI DI BO (False: vat can tinh / khong ve)
  points      (K,2)  toa do THE GIOI cua cac diem cham -- day la mang ``points`` trong hop dong LocalPlanner
  points_static / points_dynamic   tach theo loai vat can;   sector_min(...)  khoang cach nho nhat trong mot cung

Quy uoc: met, radian, theta = 0 la huong +x, quay nguoc chieu kim dong ho.
Tham so doc tu config.py neu co (LIDAR_N_RAYS, LIDAR_RANGE, LIDAR_NOISE_STD, LIDAR_FOV_DEG, LIDAR_MIN_RANGE,
LIDAR_DROPOUT), khong thi dung mac dinh: 360 tia, 5,0 m, 360 do, khong nhieu, khong vung mu, khong roi tia.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config

N_RAYS = int(getattr(config, "LIDAR_N_RAYS", 360))
RANGE = float(getattr(config, "LIDAR_RANGE", 5.0))                 # [m] tam xa toi da
NOISE_STD = float(getattr(config, "LIDAR_NOISE_STD", 0.0))         # [m] do lech chuan nhieu do (0 = khong nhieu)
FOV_DEG = float(getattr(config, "LIDAR_FOV_DEG", 360.0))           # [do]
MIN_RANGE = float(getattr(config, "LIDAR_MIN_RANGE", 0.0))         # [m] vung mu gan (0 = khong co)
DROPOUT = float(getattr(config, "LIDAR_DROPOUT", 0.0))             # xac suat roi tia, [0, 1)

_EPS_T = 1e-12                                                      # doan tia ngan hon nay coi nhu do dai 0


# --------------------------------------------------------------------------------------------
@dataclass
class LidarScan:
    """Ket qua mot lan quet. Cac mang cung do dai n (so tia)."""

    ranges: np.ndarray
    angles_rel: np.ndarray
    angles: np.ndarray
    hit: np.ndarray
    origin: np.ndarray                                    # (2,) vi tri LiDAR luc quet
    max_range: float = RANGE
    dynamic: np.ndarray | None = None                     # (n,) bool: tia ve vao nguoi di bo

    def __post_init__(self):
        if self.dynamic is None:
            self.dynamic = np.zeros(np.shape(self.hit), dtype=bool)

    def _points(self, mask: np.ndarray) -> np.ndarray:
        a, r = self.angles[mask], self.ranges[mask]
        return np.column_stack([self.origin[0] + r * np.cos(a), self.origin[1] + r * np.sin(a)])

    @property
    def points(self) -> np.ndarray:
        """(K, 2) toa do the gioi cua cac diem cham (moi loai). K co the bang 0."""
        return self._points(self.hit)

    @property
    def points_static(self) -> np.ndarray:
        """(K, 2) diem cham vat can tinh (tuong, noi that, xe day)."""
        return self._points(self.hit & ~self.dynamic)

    @property
    def points_dynamic(self) -> np.ndarray:
        """(K, 2) diem cham nguoi di bo."""
        return self._points(self.hit & self.dynamic)

    def sector_min(self, center_rel: float = 0.0, half_width: float = math.pi) -> float:
        """Khoang cach nho nhat trong cung [center_rel +- half_width] (goc SO VOI huong robot).

        Tia khong ve tinh bang max_range. Cung rong (half_width >= pi) = toan bo vong quet.
        Vd phanh khan cap:  scan.sector_min(0.0, math.radians(30)) < 0.8
        """
        diff = (self.angles_rel - center_rel + math.pi) % (2.0 * math.pi) - math.pi
        sel = np.abs(diff) <= half_width + 1e-12
        return float(self.ranges[sel].min()) if sel.any() else float(self.max_range)


# --------------------------------------------------------------------------------------------
class Lidar2D:
    """LiDAR 2D (mac dinh 360 tia, 360 do, 5 m).

    Dung::

        lidar = Lidar2D(rng=np.random.default_rng(seed))
        scan = lidar.scan(world, pose=[x, y, theta], circles=pedestrians_array)
        scan.points   # (K, 2) -> dua cho LocalPlanner
        scan.ranges   # (n,)   -> VFH co the dung truc tiep

    ``rng`` co the gan lai sau khi tao (simulator lam vay moi lan chay de tai lap theo seed).
    """

    def __init__(self, n_rays: int | None = None, max_range: float | None = None,
                 noise_std: float | None = None, rng: np.random.Generator | None = None,
                 fov_deg: float | None = None, min_range: float | None = None, dropout: float | None = None):
        self.n_rays = int(N_RAYS if n_rays is None else n_rays)
        self.max_range = float(RANGE if max_range is None else max_range)
        self.noise_std = float(NOISE_STD if noise_std is None else noise_std)
        self.fov_deg = float(FOV_DEG if fov_deg is None else fov_deg)
        self.min_range = float(MIN_RANGE if min_range is None else min_range)
        self.dropout = float(DROPOUT if dropout is None else dropout)
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.n_rays < 1:
            raise ValueError("n_rays phai >= 1")
        if not (math.isfinite(self.max_range) and self.max_range > 0.0):
            raise ValueError("max_range phai > 0 va huu han")
        if self.noise_std < 0.0 or not math.isfinite(self.noise_std):
            raise ValueError("noise_std phai >= 0")
        if not (0.0 < self.fov_deg <= 360.0):
            raise ValueError("fov_deg phai nam trong (0, 360]")
        if not (0.0 <= self.min_range < self.max_range):
            raise ValueError("min_range phai nam trong [0, max_range)")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout phai nam trong [0, 1)")

        fov = math.radians(self.fov_deg)
        if self.n_rays == 1:
            self.angles_rel = np.zeros(1)
        elif self.fov_deg >= 360.0 - 1e-9:
            self.angles_rel = np.linspace(-math.pi, math.pi, self.n_rays, endpoint=False)
        else:
            self.angles_rel = np.linspace(-fov / 2.0, fov / 2.0, self.n_rays)

    @property
    def angular_resolution(self) -> float:
        """Khoang cach goc giua hai tia ke nhau [rad]."""
        if self.n_rays == 1:
            return 0.0
        full = self.fov_deg >= 360.0 - 1e-9
        return math.radians(self.fov_deg) / (self.n_rays if full else self.n_rays - 1)

    # ------------------------------------------------------------------ noi bo
    def _cast_grid(self, world, x: float, y: float, cs: np.ndarray, sn: np.ndarray):
        """Duyet luoi chinh xac. Tra ve (range [n], hit [n]); khong ve -> range = max_range."""
        res, R = world.res, self.max_range
        n = cs.shape[0]
        K = int(math.ceil(R / res)) + 2                                       # so duong luoi toi da moi truc trong tam xa
        k = np.arange(K)[None, :]
        ix0, iy0 = math.floor(x / res + 1e-9), math.floor(y / res + 1e-9)     # cung quy uoc voi HospitalMap.xy_to_rc

        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.where((cs > 0.0)[:, None], ix0 + 1 + k, ix0 - k)          # chi so duong luoi doc ma tia cat lan luot
            ly = np.where((sn > 0.0)[:, None], iy0 + 1 + k, iy0 - k)
            tx = np.where(np.abs(cs)[:, None] < _EPS_T, np.inf, (lx * res - x) / cs[:, None])
            ty = np.where(np.abs(sn)[:, None] < _EPS_T, np.inf, (ly * res - y) / sn[:, None])

        crossings = np.minimum(np.maximum(np.concatenate([tx, ty], axis=1), 0.0), R)   # [0, R]; qua tam xa -> R
        t = np.concatenate([np.zeros((n, 1)), np.sort(crossings, axis=1), np.full((n, 1), R)], axis=1)
        seg = t[:, 1:] - t[:, :-1]
        mid = 0.5 * (t[:, 1:] + t[:, :-1])                                    # diem giua doan: nam gon trong mot o

        c = np.floor((x + mid * cs[:, None]) / res).astype(int)
        r = np.floor((y + mid * sn[:, None]) / res).astype(int)
        inside = (c >= 0) & (c < world.n_cols) & (r >= 0) & (r < world.n_rows)
        occ = np.ones(c.shape, dtype=bool)                                    # ngoai ban do = vat can
        occ[inside] = world.grid[r[inside], c[inside]]
        occ &= seg > _EPS_T                                                   # bo doan do dai 0 (tia di qua dinh o)

        hit = occ.any(axis=1)
        first = occ.argmax(axis=1)                                            # doan dac dau tien doc tia
        rng = np.where(hit, t[np.arange(n), first], R)
        return rng, hit

    def _cast_circles(self, x: float, y: float, cs: np.ndarray, sn: np.ndarray, circles) -> np.ndarray:
        """Khoang cach tia -> hinh tron gan nhat (giai tich); khong cham hoac qua tam xa: +inf."""
        n = cs.shape[0]
        if circles is None:
            return np.full(n, np.inf)
        c = np.asarray(circles, dtype=float)
        if c.size == 0:
            return np.full(n, np.inf)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError("circles phai co dang (M, 3) = [cx, cy, ban_kinh]")
        vx, vy = c[None, :, 0] - x, c[None, :, 1] - y                         # (1, M)
        t_c = vx * cs[:, None] + vy * sn[:, None]                             # (n, M) hinh chieu tam len tia
        dist2 = vx ** 2 + vy ** 2
        r2 = c[None, :, 2] ** 2
        disc = r2 - (dist2 - t_c ** 2)                                        # binh phuong nua day cung
        inside = dist2 < r2                                                   # LiDAR nam trong nguoi: da cham
        t = np.where(inside, 0.0, t_c - np.sqrt(np.maximum(disc, 0.0)))
        valid = inside | ((disc >= 0.0) & (t >= 0.0) & (t <= self.max_range))
        return np.where(valid, t, np.inf).min(axis=1)

    # ------------------------------------------------------------------ giao dien chinh
    def scan(self, world, pose, circles=None) -> LidarScan:
        """Quet tai pose = [x, y, theta] tren `world` (HospitalMap THAT) + cac hinh tron (nguoi di bo)."""
        p = np.asarray(pose, dtype=float).ravel()
        if p.size < 3 or not np.all(np.isfinite(p[:3])):
            raise ValueError("pose phai co it nhat 3 so huu han [x, y, theta]")
        x, y, th = float(p[0]), float(p[1]), float(p[2])
        R, n = self.max_range, self.n_rays

        ang = th + self.angles_rel
        cs, sn = np.cos(ang), np.sin(ang)

        r_grid, hit_grid = self._cast_grid(world, x, y, cs, sn)
        t_circ = self._cast_circles(x, y, cs, sn, circles)
        hit_circ = np.isfinite(t_circ)
        dyn = hit_circ & (~hit_grid | (t_circ < r_grid))                      # nguoi gan hon vat tinh (hoa: vat tinh)
        rng = np.where(dyn, t_circ, r_grid)
        hit = hit_grid | hit_circ

        if self.min_range > 0.0:                                              # vung mu gan
            hit &= ~(rng < self.min_range)
        if self.dropout > 0.0:                                                # rut dung n so cho moi lan quet -> tai lap duoc
            hit &= ~(self.rng.random(n) < self.dropout)
        if self.noise_std > 0.0:
            noisy = rng + self.rng.normal(0.0, self.noise_std, n)
            rng = np.where(hit, np.maximum(noisy, 0.0), rng)
            hit &= rng < R                                                    # nhieu day tia ra ngoai tam xa -> khong ve

        rng = np.where(hit, rng, R)
        dyn &= hit
        return LidarScan(ranges=rng, angles_rel=self.angles_rel.copy(), angles=ang, hit=hit,
                         origin=np.array([x, y]), max_range=R, dynamic=dyn)