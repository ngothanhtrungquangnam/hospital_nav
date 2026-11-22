"""env/pedestrians.py -- Quan ly vat can dong (nguoi di bo, xe lan, y ta + hop thuoc).

Nguoi phu trach: TV1.

Quy uoc du lieu
---------------
- Don vi: met [m], radian [rad], giay [s].
- He toa do the gioi: goc (0, 0) o duoi-trai, x sang phai, y len tren (khop HospitalMap).
- Mang hinh tron tra ve cho LiDAR / renderer: np.ndarray shape (M, 3) = [cx, cy, radius].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

import config

Vec2 = tuple[float, float]
Circle3 = tuple[float, float, float]

DEFAULT_PED_RADIUS: float = float(getattr(config, "PEDESTRIAN_RADIUS", 0.28))
DEFAULT_PED_SPEED: float = float(getattr(config, "PEDESTRIAN_SPEED", 0.70))
DEFAULT_ARRIVE_TOL: float = float(getattr(config, "PEDESTRIAN_ARRIVE_TOL", 0.25))
DEFAULT_TURN_RATE: float = float(getattr(config, "PEDESTRIAN_TURN_RATE", 6.5))

VALID_PED_KINDS: tuple[str, ...] = ("visitor", "patient", "wheelchair", "nurse_cart")

_EPS = 1e-12


def _wrap_angle(angle: float) -> float:
    """Chuan hoa goc ve khoang [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def circle_grid_clearance(world, x: float, y: float, radius: float) -> float:
    """Khoang cach nho nhat tu vien hinh tron (x, y, radius) toi vat can tinh / bien.

    > 0 : khong va cham (khe ho toi thieu).
    <= 0: da cham / nam trong vat can / ngoai bien.
    """
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(radius)) or radius <= 0.0:
        return -math.inf
    if world is None:
        return math.inf

    grid = getattr(world, "grid", None)
    res = float(getattr(world, "res", getattr(config, "RESOLUTION", 0.1)))
    width = float(
        getattr(world, "width", grid.shape[1] * res if grid is not None else getattr(config, "MAP_WIDTH", 40.0))
    )
    height = float(
        getattr(world, "height", grid.shape[0] * res if grid is not None else getattr(config, "MAP_HEIGHT", 20.0))
    )

    edge_clear = min(x - radius, width - (x + radius), y - radius, height - (y + radius))
    if edge_clear <= 0.0 or grid is None or grid.size == 0:
        return edge_clear

    n_rows, n_cols = grid.shape
    pad = radius + res
    c0 = max(0, int(math.floor((x - pad) / res)))
    c1_idx = min(n_cols - 1, int(math.floor((x + pad) / res)))
    r0 = max(0, int(math.floor((y - pad) / res)))
    r1_idx = min(n_rows - 1, int(math.floor((y + pad) / res)))
    if c0 > c1_idx or r0 > r1_idx:
        return edge_clear

    sub = grid[r0 : r1_idx + 1, c0 : c1_idx + 1]
    if not np.any(sub):
        return edge_clear

    rr, cc = np.nonzero(sub)
    x_left = (cc + c0) * res
    x_right = x_left + res
    y_bot = (rr + r0) * res
    y_top = y_bot + res

    dx = np.maximum.reduce([x_left - x, np.zeros_like(x_left), x - x_right])
    dy = np.maximum.reduce([y_bot - y, np.zeros_like(y_bot), y - y_top])
    nearest_cell_dist = float(np.min(np.hypot(dx, dy)))
    return min(edge_clear, nearest_cell_dist - radius)


def is_circle_free(world, x: float, y: float, radius: float, min_clearance: float = 0.0) -> bool:
    """True neu hinh tron (x, y, radius) khong cham vat can tinh voi khe ho >= min_clearance."""
    return circle_grid_clearance(world, x, y, radius) >= float(min_clearance)


def circles_for_pose(
    x: float,
    y: float,
    radius: float,
    heading: float,
    kind: str = "visitor",
) -> list[Circle3]:
    """Sinh danh sach 1 hoac 2 hinh tron va cham khop voi tung loai tac nhan."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    if kind == "wheelchair":
        # Gia dinh mo hinh: 2 vong tron doc theo truc di chuyen (sau + truoc)
        r_body = max(0.22, radius * 0.92)
        r_front = max(0.20, radius * 0.85)
        offset = max(0.28, radius * 1.15)
        return [
            (x - 0.35 * offset * cos_h, y - 0.35 * offset * sin_h, r_body),
            (x + 0.65 * offset * cos_h, y + 0.65 * offset * sin_h, r_front),
        ]

    if kind == "nurse_cart":
        # Gia dinh mo hinh: 1 vong cho y ta + 1 vong cho hop/xe thuoc phia truoc
        r_nurse = max(0.20, radius * 0.90)
        r_cart = max(0.18, radius * 0.78)
        cart_dist = max(0.45, radius * 1.85)
        return [
            (x, y, r_nurse),
            (x + cart_dist * cos_h, y + cart_dist * sin_h, r_cart),
        ]

    # "visitor" hoac "patient": 1 hinh tron chuan
    return [(float(x), float(y), float(radius))]


def circles_from_pedestrians(pedestrians: Iterable["Pedestrian"]) -> np.ndarray:
    """Chuyen danh sach Pedestrian sang mang (M, 3) = [cx, cy, radius] cho Lidar2D."""
    items: list[Circle3] = []
    for p in pedestrians:
        items.extend(p.as_circles())
    if not items:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(items, dtype=float)


@dataclass
class Pedestrian:
    """Tac nhan dong trong moi truong benh vien tu cap nhat trang thai theo buoc dt."""

    x: float
    y: float
    radius: float = DEFAULT_PED_RADIUS
    speed: float = DEFAULT_PED_SPEED
    waypoints: list[Vec2] = field(default_factory=list)
    loop: bool = True
    mode: str = "waypoint"          # "waypoint" | "wander"
    kind: str = "visitor"           # "visitor" | "patient" | "wheelchair" | "nurse_cart"
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    arrive_tol: float = DEFAULT_ARRIVE_TOL
    turn_rate: float = DEFAULT_TURN_RATE
    ped_id: int = 0
    _wp_idx: int = field(default=0, repr=False)
    _wp_dir: int = field(default=1, repr=False)

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.radius = float(self.radius)
        self.speed = float(self.speed)
        self.kind = str(self.kind)
        self.heading = _wrap_angle(float(self.heading))
        self.vx = float(self.vx)
        self.vy = float(self.vy)
        self.arrive_tol = float(self.arrive_tol)
        self.turn_rate = float(self.turn_rate)
        self.waypoints = [(float(px), float(py)) for px, py in self.waypoints]

        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError("Toa do (x, y) cua tac nhan phai huu han")
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("radius phai > 0 va huu han")
        if not (math.isfinite(self.speed) and self.speed >= 0.0):
            raise ValueError("speed phai >= 0 va huu han")
        if not (math.isfinite(self.arrive_tol) and self.arrive_tol > 0.0):
            raise ValueError("arrive_tol phai > 0")
        if not (math.isfinite(self.turn_rate) and self.turn_rate >= 0.0):
            raise ValueError("turn_rate phai >= 0")
        if self.mode not in ("waypoint", "wander"):
            raise ValueError(f"Che do di chuyen khong hop le: {self.mode!r}")
        if self.kind not in VALID_PED_KINDS:
            raise ValueError(f"Loai tac nhan khong hop le: {self.kind!r}")

        if math.hypot(self.vx, self.vy) > _EPS:
            self.heading = math.atan2(self.vy, self.vx)
        elif self.waypoints:
            start_target_idx = (
                1
                if (
                    len(self.waypoints) > 1
                    and math.hypot(self.waypoints[0][0] - self.x, self.waypoints[0][1] - self.y) <= self.arrive_tol
                )
                else 0
            )
            tx, ty = self.waypoints[start_target_idx]
            dx, dy = tx - self.x, ty - self.y
            if math.hypot(dx, dy) > _EPS:
                self.heading = math.atan2(dy, dx)
                self.vx = self.speed * math.cos(self.heading)
                self.vy = self.speed * math.sin(self.heading)
                self._wp_idx = start_target_idx

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    def as_circle(self) -> Circle3:
        """Tra ve 1 hinh tron dai dien chinh (tuong thich giao dien cu)."""
        return (self.x, self.y, self.radius)

    def as_circles(self) -> list[Circle3]:
        """Tra ve danh sach 1 hoac 2 hinh tron va cham khop voi `self.kind`."""
        return circles_for_pose(self.x, self.y, self.radius, self.heading, kind=self.kind)

    def _select_waypoint(self) -> Vec2 | None:
        if not self.waypoints:
            return None
        tx, ty = self.waypoints[self._wp_idx]
        if math.hypot(tx - self.x, ty - self.y) <= self.arrive_tol:
            n_wp = len(self.waypoints)
            if n_wp == 1:
                return (tx, ty)
            if self.loop:
                self._wp_idx = (self._wp_idx + 1) % n_wp
            else:
                if self._wp_idx >= n_wp - 1:
                    self._wp_dir = -1
                elif self._wp_idx <= 0:
                    self._wp_dir = 1
                self._wp_idx = int(np.clip(self._wp_idx + self._wp_dir, 0, n_wp - 1))
        return self.waypoints[self._wp_idx]

    def step(
        self,
        dt: float,
        world=None,
        rng: np.random.Generator | None = None,
        others: Sequence[Circle3] | None = None,
    ) -> Circle3:
        """Cap nhat vi tri, huong va van toc tac nhan sau khoang thoi gian dt [s]."""
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt phai >= 0 va huu han")
        if dt == 0.0:
            return self.as_circle()

        if self.speed <= _EPS:
            self.vx = 0.0
            self.vy = 0.0
            return self.as_circle()

        if self.mode == "waypoint":
            target = self._select_waypoint()
            if target is not None:
                dx, dy = target[0] - self.x, target[1] - self.y
                if math.hypot(dx, dy) > _EPS:
                    desired_heading = math.atan2(dy, dx)
                    delta = _wrap_angle(desired_heading - self.heading)
                    max_turn = self.turn_rate * dt
                    self.heading = _wrap_angle(
                        self.heading + float(np.clip(delta, -max_turn, max_turn))
                    )
        else:
            gen = rng if rng is not None else np.random.default_rng()
            jitter = float(gen.normal(0.0, 0.75)) * math.sqrt(max(dt, 1e-6))
            self.heading = _wrap_angle(self.heading + jitter)

        step_dist = self.speed * dt
        offsets = (
            0.0,
            0.25, -0.25,
            0.50, -0.50,
            0.85, -0.85,
            1.25, -1.25,
            math.pi / 2.0, -math.pi / 2.0,
            2.35, -2.35,
            math.pi,
        )

        own_circles = self.as_circles()

        def _pose_free_static(nx: float, ny: float, ang: float) -> bool:
            for cx, cy, cr in circles_for_pose(nx, ny, self.radius, ang, kind=self.kind):
                if not is_circle_free(world, cx, cy, cr, min_clearance=1e-6):
                    return False
            return True

        def _hits_other(nx: float, ny: float, ang: float) -> bool:
            if not others:
                return False
            cand_circles = circles_for_pose(nx, ny, self.radius, ang, kind=self.kind)
            for ocx, ocy, ocr in others:
                # Bo qua chinh cac hinh tron cua ban than tac nhan hien tai
                if any(
                    abs(ocx - scx) < _EPS and abs(ocy - scy) < _EPS and abs(ocr - scr) < _EPS
                    for scx, scy, scr in own_circles
                ):
                    continue
                cur_d2 = min((scx - ocx) ** 2 + (scy - ocy) ** 2 for scx, scy, _ in own_circles)
                for ccx, ccy, ccr in cand_circles:
                    min_d = ccr + float(ocr) + 0.04
                    new_d2 = (ccx - ocx) ** 2 + (ccy - ocy) ** 2
                    if new_d2 < min_d * min_d and new_d2 <= cur_d2 + _EPS:
                        return True
            return False

        moved = False
        for strict_others in (True, False):
            for off in offsets:
                ang = _wrap_angle(self.heading + off)
                nx = self.x + step_dist * math.cos(ang)
                ny = self.y + step_dist * math.sin(ang)
                if not _pose_free_static(nx, ny, ang):
                    continue
                if strict_others and _hits_other(nx, ny, ang):
                    continue
                self.heading = ang
                self.vx = (nx - self.x) / dt
                self.vy = (ny - self.y) / dt
                self.x = nx
                self.y = ny
                moved = True
                break
            if moved:
                break

        if not moved:
            self.heading = _wrap_angle(self.heading + math.pi)
            self.vx = 0.0
            self.vy = 0.0

        return self.as_circle()


class PedestrianManager:
    """Lop tien ich gom nhieu Pedestrian, ho tro reset theo seed va xuat mang (M, 3)."""

    def __init__(self, pedestrians: Iterable[Pedestrian] = (), seed: int | None = None) -> None:
        self._specs: list[Pedestrian] = [
            Pedestrian(
                x=p.x,
                y=p.y,
                radius=p.radius,
                speed=p.speed,
                waypoints=list(p.waypoints),
                loop=p.loop,
                mode=p.mode,
                kind=p.kind,
                heading=p.heading,
                vx=p.vx,
                vy=p.vy,
                arrive_tol=p.arrive_tol,
                turn_rate=p.turn_rate,
                ped_id=p.ped_id if p.ped_id else idx + 1,
            )
            for idx, p in enumerate(pedestrians)
        ]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pedestrians: list[Pedestrian] = []
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.pedestrians = [
            Pedestrian(
                x=s.x,
                y=s.y,
                radius=s.radius,
                speed=s.speed,
                waypoints=list(s.waypoints),
                loop=s.loop,
                mode=s.mode,
                kind=s.kind,
                heading=s.heading,
                vx=s.vx,
                vy=s.vy,
                arrive_tol=s.arrive_tol,
                turn_rate=s.turn_rate,
                ped_id=s.ped_id,
            )
            for s in self._specs
        ]

    def __len__(self) -> int:
        return len(self.pedestrians)

    def as_circles(self) -> np.ndarray:
        return circles_from_pedestrians(self.pedestrians)

    def step(self, dt: float, world=None) -> np.ndarray:
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt phai >= 0 va huu han")
        if dt == 0.0 or not self.pedestrians:
            return self.as_circles()

        snapshot_all: list[Circle3] = []
        per_ped_circles: list[list[Circle3]] = []
        for p in self.pedestrians:
            cs = p.as_circles()
            per_ped_circles.append(cs)
            snapshot_all.extend(cs)

        for idx, ped in enumerate(self.pedestrians):
            own_set = per_ped_circles[idx]
            others = [c for c in snapshot_all if c not in own_set]
            ped.step(dt, world=world, rng=self.rng, others=others)
        return self.as_circles()


def default_hospital_pedestrians(seed: int | None = 42) -> PedestrianManager:
    """Tao kich ban nguoi di bo, xe lan va y ta + hop thuoc doc lap tren ban do benh vien."""
    specs = [
        Pedestrian(
            6.5, 10.4,
            radius=0.28,
            speed=0.72,
            waypoints=[(6.5, 10.4), (13.5, 10.4), (6.5, 10.4)],
            loop=True,
            kind="visitor",
            ped_id=1,
        ),
        Pedestrian(
            25.5, 9.2,
            radius=0.28,
            speed=0.62,
            waypoints=[(25.5, 9.2), (33.0, 9.2), (25.5, 9.2)],
            loop=True,
            kind="nurse_cart",
            ped_id=2,
        ),
        Pedestrian(
            9.0, 10.7,
            radius=0.28,
            speed=0.58,
            waypoints=[(9.0, 10.7), (16.5, 10.7), (9.0, 10.7)],
            loop=True,
            kind="wheelchair",
            ped_id=3,
        ),
        Pedestrian(
            18.0, 4.8,
            radius=0.28,
            speed=0.78,
            waypoints=[(18.0, 4.8), (24.5, 4.8), (24.5, 7.0), (18.0, 4.8)],
            loop=True,
            kind="patient",
            ped_id=4,
        ),
    ]
    return PedestrianManager(specs, seed=seed)