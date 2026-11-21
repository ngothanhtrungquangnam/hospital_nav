"""env/hospital_map.py -- Ban do luoi tang benh vien 40 m x 20 m.

Nguoi phu trach: TV1.

Quy uoc toa do (ca nhom dung chung)
-----------------------------------
- Don vi: met, radian, giay.
- He toa do the gioi: goc (0, 0) o goc duoi-trai, x sang phai, y len tren.
- grid la numpy.ndarray bool, shape = (n_rows, n_cols) = (200, 400):
    + Hang r tuong ung truc y (hang 0 o phia duoi).
    + Cot c tuong ung truc x.
    + O (r, c) phu vung x in [c*res, (c+1)*res),
      y in [r*res, (r+1)*res).
    + c = floor(x/res), r = floor(y/res).
    + Tam o: x = (c+0.5)*res, y = (r+0.5)*res.
- Khi ve grid bang imshow, dung origin="lower".

Hai ban do khac nhau (kich ban 1)
---------------------------------
- Ban do DA BIET: HospitalMap(). A* chi duoc su dung ban do nay.
- THE GIOI THAT: known.with_obstacles(UNMAPPED_CARTS). Ban sao nay co them
  vat can phat sinh ma A* khong biet. LiDAR do tren the gioi that.

Quy uoc bo cuc
--------------
- Kich thuoc: 40 m x 20 m.
- Hanh lang chinh: y = 8.6 m den y = 11.4 m.
- Nguon: (5.0, 14.0).
- Dich: G1, G2, G3, G4.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

import config


# ---------------------------------------------------------------------------
# Thong so rieng cua ban do
# ---------------------------------------------------------------------------
DOOR_WIDTH: float = getattr(config, "DOOR_WIDTH", 1.2)
CORRIDOR_WIDTH: float = getattr(config, "CORRIDOR_WIDTH", 2.8)
SAFETY_MARGIN: float = getattr(config, "SAFETY_MARGIN", 0.10)
WALL_THICKNESS: float = 0.2

# Hinh chu nhat (x1, y1, x2, y2), don vi met.
Rect = tuple[float, float, float, float]
# Mot noi that gom (loai noi that, hinh chu nhat).
Furniture = tuple[str, Rect]


# Noi that co dinh: A* BIET.
# Cac vat chua xac dinh ro loai duoc gan kind="furniture" de renderer
# van tach chung khoi tuong, nhung khong tu suy doan chung la giuong/tu.
FIXED_FURNITURE: list[Furniture] = [
    # Ghe cho trong sanh
    ("bench", (15.0, 3.0, 17.0, 3.5)),
    ("bench", (19.0, 3.0, 21.0, 3.5)),
    ("bench", (23.0, 3.0, 25.0, 3.5)),
    ("bench", (15.0, 6.0, 17.0, 6.5)),
    ("bench", (19.0, 6.0, 21.0, 6.5)),
    ("bench", (23.0, 6.0, 25.0, 6.5)),

    # Thung rac
    ("bin", (26.5, 1.0, 26.9, 1.4)),

    # Ke thuoc trong nha thuoc
    ("shelf", (2.0, 18.0, 8.0, 19.2)),

    # Noi that tai cac phong (phan loai chi tiet, giu nguyen toa do rect va grid)
    ("bed", (11.0, 17.5, 13.0, 19.5)),
    ("bed", (17.0, 17.0, 19.0, 19.0)),
    ("bed", (21.0, 17.5, 23.0, 19.5)),
    ("lab", (27.0, 16.0, 29.0, 18.0)),
    ("bed", (31.0, 17.5, 33.0, 19.5)),
    ("lab", (37.0, 15.0, 38.8, 17.0)),
    ("bed", (1.0, 6.0, 3.0, 8.0)),
    ("bed", (9.0, 1.0, 11.0, 3.0)),
]

# Vat can PHAT SINH mac dinh cho kich ban 1: A* KHONG BIET cac vat can nay.
UNMAPPED_CARTS: list[Rect] = [
    (17.0, 10.4, 17.6, 11.4),  # Xe day sat tuong bac hanh lang
    (22.0, 8.6, 22.6, 9.6),   # Xe day phia nam
    (32.0, 10.4, 32.6, 11.4), # Xe day sat tuong bac
]


@dataclass(frozen=True)
class Room:
    """Thong tin phong dung de hien thi nhan tren ban do."""

    name: str
    label_xy: tuple[float, float]
    rect: tuple[float, float, float, float] | None = None
    kind: str = "room"

    @property
    def x1(self) -> float:
        return self.rect[0] if self.rect else 0.0

    @property
    def y1(self) -> float:
        return self.rect[1] if self.rect else 0.0

    @property
    def x2(self) -> float:
        return self.rect[2] if self.rect else 0.0

    @property
    def y2(self) -> float:
        return self.rect[3] if self.rect else 0.0


def _rnd(value: float) -> int:
    """Lam tron nua len; phu hop voi toa do la boi so cua RESOLUTION."""
    return int(math.floor(value + 0.5))


class HospitalMap:
    """Ban do occupancy grid cua tang benh vien.

    Cac thuoc tinh chinh
    -------------------
    grid            : ndarray bool; True = o co vat can thuc te tren ban do nay.
    inflated        : ndarray bool; True = tam robot khong duoc dat vao.
    inflate_radius  : ROBOT_RADIUS + SAFETY_MARGIN, don vi met.
    start           : (x, y) diem xuat phat tai Nha thuoc.
    goals           : dict {"G1": (x, y), ...}.
    doors           : list cac tam cua (x, y).
    rooms           : list[Room] de ve nhan phong.
    furniture       : list[(kind, rect)] noi that co dinh, dung cho map_view.
    unmapped        : list[rect] vat can phat sinh, chi co tren the gioi that.

    Luu y: with_obstacles() tao ban sao; khong sua doi doi tuong ban dau.
    """

    def __init__(self, with_furniture: bool = True):
        self.res: float = float(config.RESOLUTION)
        self.width: float = float(config.MAP_WIDTH)
        self.height: float = float(config.MAP_HEIGHT)
        self.n_rows: int = _rnd(self.height / self.res)
        self.n_cols: int = _rnd(self.width / self.res)

        self.inflate_radius: float = float(config.ROBOT_RADIUS) + SAFETY_MARGIN
        self.inflate_cells: int = int(
            math.ceil(self.inflate_radius / self.res - 1e-9)
        )

        self.grid: np.ndarray = np.zeros(
            (self.n_rows, self.n_cols), dtype=bool
        )
        self._inflated: np.ndarray | None = None

        self.corridor_y: tuple[float, float] = (
            self.height / 2 - CORRIDOR_WIDTH / 2,
            self.height / 2 + CORRIDOR_WIDTH / 2,
        )

        self.start: tuple[float, float] = (5.0, 14.0)
        self.goals: dict[str, tuple[float, float]] = {
            "G1": (15.0, 15.0),
            "G2": (25.0, 15.0),
            "G3": (35.0, 15.0),
            "G4": (6.0, 4.0),
        }
        self.goal_rooms: dict[str, str] = {
            "G1": "Khoa Noi",
            "G2": "Khoa Ngoai",
            "G3": "Xet nghiem",
            "G4": "Cap cuu",
        }

        # Metadata dung chung voi env/map_view.py.
        # Phai khoi tao truoc khi goi _build().
        self.wall_thickness: float = WALL_THICKNESS
        self.door_width: float = DOOR_WIDTH
        self.doors: list[tuple[float, float]] = []
        self.rooms: list[Room] = []
        self.furniture: list[Furniture] = []
        self.unmapped: list[Rect] = []

        self._build(with_furniture)

    # ------------------------------------------------------------------
    # Dung ban do
    # ------------------------------------------------------------------
    def _build(self, with_furniture: bool) -> None:
        W, H = self.width, self.height
        t = self.wall_thickness
        ys, yn = self.corridor_y
        dw = self.door_width

        # 1. Tuong bao quanh
        boundary_walls: list[Rect] = [
            (0.0, 0.0, W, t),
            (0.0, H - t, W, H),
            (0.0, 0.0, t, H),
            (W - t, 0.0, W, H),
        ]
        for rect in boundary_walls:
            self._fill(rect, True)

        # 2. Tuong ngan hanh lang voi cac hang phong
        self._fill((0.0, yn, W, yn + t), True)                   # Phia bac
        self._fill((0.0, ys - t, 12.0 + t, ys), True)           # Phia nam, ben trai
        self._fill((28.0 - t, ys - t, W, ys), True)             # Phia nam, ben phai

        # 3. Tuong ngan giua cac phong
        for xp in (10.0, 20.0, 30.0):
            self._fill((xp, yn, xp + t, H), True)
        self._fill((12.0, 0.0, 12.0 + t, ys), True)
        self._fill((28.0 - t, 0.0, 28.0, ys), True)

        # 4. Khoet cua
        top_doors = (5.0, 15.0, 25.0, 35.0)
        bottom_doors = (6.0, 34.0)

        for xd in top_doors:
            self._fill((xd - dw / 2, yn, xd + dw / 2, yn + t), False)
            self.doors.append((float(xd), yn + t / 2))

        for xd in bottom_doors:
            self._fill((xd - dw / 2, ys - t, xd + dw / 2, ys), False)
            self.doors.append((float(xd), ys - t / 2))

        # 5. Noi that co dinh
        # Luu metadata de map_view.py tach noi that khoi tuong khi render.
        self.furniture = list(FIXED_FURNITURE) if with_furniture else []
        for _kind, rect in self.furniture:
            self._fill(rect, True)

        # 6. Nhan phong (de ve)
        labels = [
            ("Nha thuoc", (5.0, 12.3), (0.0, 11.6, 10.0, 20.0), "pharmacy"),
            ("Khoa Noi", (15.0, 12.3), (10.0, 11.6, 20.0, 20.0), "ward"),
            ("Khoa Ngoai", (25.0, 12.3), (20.0, 11.6, 30.0, 20.0), "ward"),
            ("Xet nghiem", (35.0, 12.3), (30.0, 11.6, 40.0, 20.0), "lab"),
            ("Cap cuu", (3.5, 4.3), (0.0, 0.0, 12.0, 8.6), "emergency"),
            ("Sanh cho", (20.0, 7.6), (12.0, 0.0, 28.0, 8.6), "waiting"),
            ("Thang may", (34.0, 7.6), (28.0, 0.0, 40.0, 8.6), "service"),
            ("Hanh lang chinh", (20.0, 10.0), (0.0, 8.6, 40.0, 11.6), "corridor"),
        ]
        self.rooms = [Room(name, label_xy, rect, kind)
                      for name, label_xy, rect, kind in labels]

    def _fill(self, rect: Rect, value: bool) -> None:
        """To value vao cac o thuoc hinh chu nhat (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = rect

        c1 = max(0, _rnd(x1 / self.res))
        c2 = min(self.n_cols, _rnd(x2 / self.res))
        r1 = max(0, _rnd(y1 / self.res))
        r2 = min(self.n_rows, _rnd(y2 / self.res))

        if c2 > c1 and r2 > r1:
            self.grid[r1:r2, c1:c2] = value
            # Vo hieu hoa cache inflation khi grid thay doi.
            self._inflated = None

    # ------------------------------------------------------------------
    # Vat can phat sinh: tao ban sao, khong sua ban do goc
    # ------------------------------------------------------------------
    def with_obstacles(self, rects: Iterable[Rect]) -> HospitalMap:
        """Tra ve ban sao co them cac vat can hinh chu nhat.

        Args:
            rects: Cac hinh chu nhat (x1, y1, x2, y2), don vi met.

        Returns:
            Ban sao HospitalMap co them vat can trong grid va danh sach unmapped.

        Ban do hien tai (self) duoc giu nguyen. Vi vay A* co the tiep tuc dung
        known, trong khi simulator/LiDAR su dung world = known.with_obstacles(...).
        """
        new = copy.deepcopy(self)
        added_rects = list(rects)

        for rect in added_rects:
            new._fill(rect, True)

        # Luu rieng danh sach vat can phat sinh de map_view.py ve mau rieng.
        new.unmapped = [*self.unmapped, *added_rects]
        return new

    # ------------------------------------------------------------------
    # Phong vat can
    # ------------------------------------------------------------------
    @property
    def inflated(self) -> np.ndarray:
        """Ban do phong vat can theo ban kinh inflate_cells (tinh mot lan)."""
        if self._inflated is None:
            k = self.inflate_cells
            yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
            kernel = (xx ** 2 + yy ** 2) <= k * k
            self._inflated = ndimage.binary_dilation(
                self.grid,
                structure=kernel,
            )
        return self._inflated

    # ------------------------------------------------------------------
    # Doi toa do
    # ------------------------------------------------------------------
    def xy_to_rc(self, x, y):
        """Chuyen (x, y) [m] sang (r, c) 0-based; toa do duoc kep vao bien."""
        c = np.clip(
            np.floor(np.asarray(x, dtype=float) / self.res + 1e-9).astype(int),
            0,
            self.n_cols - 1,
        )
        r = np.clip(
            np.floor(np.asarray(y, dtype=float) / self.res + 1e-9).astype(int),
            0,
            self.n_rows - 1,
        )

        if r.ndim == 0:
            return int(r), int(c)
        return r, c

    def rc_to_xy(self, r, c):
        """Chuyen (r, c) 0-based sang toa do tam o (x, y) [m]."""
        return (np.asarray(c) + 0.5) * self.res, (np.asarray(r) + 0.5) * self.res

    # ------------------------------------------------------------------
    # Truy van ban do
    # ------------------------------------------------------------------
    def is_free(self, x: float, y: float, inflated: bool = True) -> bool:
        """True neu diem (x, y) tu do; inflated=True kiem tra vung phong robot."""
        if not (0.0 <= x < self.width and 0.0 <= y < self.height):
            return False

        r, c = self.xy_to_rc(x, y)
        occupancy = self.inflated if inflated else self.grid
        return not bool(occupancy[r, c])

    def reachable(self, a, b, inflated: bool = True) -> bool:
        """Kiem tra lien thong 4 huong giua hai diem a=(x,y), b=(x,y)."""
        blocked = self.inflated if inflated else self.grid
        labels, _ = ndimage.label(~blocked)

        ra, ca = self.xy_to_rc(*a)
        rb, cb = self.xy_to_rc(*b)

        return bool(
            labels[ra, ca] != 0
            and labels[ra, ca] == labels[rb, cb]
        )

    # ------------------------------------------------------------------
    # Ve ban do occupancy grid co ban
    # ------------------------------------------------------------------
    def plot(self, ax=None, show_inflated: bool = True, title: str | None = None):
        """Ve grid: xam = vat can, hong = inflation, xanh = Start, do = Goal."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 5.5))

        img = np.ones((self.n_rows, self.n_cols, 3), dtype=float)

        if show_inflated:
            img[self.inflated & ~self.grid] = (1.0, 0.88, 0.88)

        img[self.grid] = (0.25, 0.25, 0.25)

        ax.imshow(
            img,
            origin="lower",
            extent=[0.0, self.width, 0.0, self.height],
            interpolation="nearest",
        )

        for room in self.rooms:
            ax.text(
                *room.label_xy,
                room.name,
                ha="center",
                fontsize=8,
                color=(0.1, 0.2, 0.6),
            )

        ax.plot(*self.start, "o", ms=10, mfc="lime", mec="k")
        ax.text(
            self.start[0], self.start[1] + 0.9, "START",
            ha="center", fontsize=8, weight="bold",
        )

        for name, (gx, gy) in self.goals.items():
            ax.plot(gx, gy, "o", ms=10, mfc="red", mec="k")
            ax.text(gx, gy + 0.9, name, ha="center", fontsize=8, weight="bold")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(0.0, self.height)
        ax.set_aspect("equal")

        if title:
            ax.set_title(title)

        return ax


if __name__ == "__main__":
    # Chay tu thu muc goc du an: python -m env.hospital_map
    import matplotlib.pyplot as plt

    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    known.plot(axes[0], title="Ban do DA BIET (A* dung ban do nay)")
    world.plot(axes[1], title="THE GIOI THAT (them vat can phat sinh)")

    plt.tight_layout()
    plt.show()
