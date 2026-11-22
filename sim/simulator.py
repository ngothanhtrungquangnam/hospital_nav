"""sim/simulator.py -- Vong lap mo phong: A* toan cuc + bo dieu huong cuc bo + robot + LiDAR.

Nguoi phu trach: TV1.

Mot lan chay ``Simulator.run(local_planner, goal, ...)``:
  1. A* lap duong tren ban do DA BIET (known_map), rut gon thanh waypoint.
  2. Dat robot tai Start, quay san ve huong waypoint dau tien (khong phat thuong/phat ai lai dau).
  3. Moi buoc dt:  LiDAR quet tren the gioi THAT (world_map + nguoi di bo)
                   -> chon sub-goal (look-ahead doc duong A*)
                   -> local_planner.compute_control(state, sub_goal, points, dt)
                   -> Robot.step(v_cmd, w_cmd, dt)  -> cap nhat nguoi di bo.
  4. Ket thuc o mot trong cac trang thai (``SimResult.status``):

        'success'   robot den trong goal_tol cua dich
        'collision' robot cham vat can (tuong/xe day/noi that) hoac nguoi di bo  -> DUNG NGAY
        'stuck'     trong stuck_window giay robot di chuyen < stuck_dist  ("freezing robot")
        'timeout'   het t_max giay ma chua den dich
        'no_path'   A* khong tim duoc duong tren ban do da biet

Hop dong voi bo dieu huong cuc bo (planners/base.py):
    compute_control(state, sub_goal, points, dt) -> (v_cmd, w_cmd, pred_traj)
      state    : ndarray (5,) = [x, y, theta, v, w]
      sub_goal : ndarray (2,)  [gx, gy]
      points   : ndarray (N, 2) cac diem LiDAR cham vat can, he toa do THE GIOI (N co the = 0)
      Co the co ``reset()``: simulator goi truoc moi lan chay neu co.
      v_cmd, w_cmd la so thuc (mang 1 phan tu cung duoc chap nhan).

Hop dong voi nguoi di bo (env/pedestrians.py, sap viet) -- chi can 3 thu nay:
    reset(rng)      dat lai ve trang thai ban dau (rng: np.random.Generator de ngau nhien hoa)
    step(dt)        tien them dt giay
    circles         ndarray (M, 3) = [cx, cy, ban_kinh] cua tung nguoi o thoi diem hien tai

Tham so (doc tu config.py neu co, khong thi dung mac dinh):
    SIM_T_MAX=120 s, GOAL_TOL=0.3 m, LOOKAHEAD=1.5 m, STUCK_WINDOW=10 s, STUCK_DIST=0.3 m
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import config
from models.robot import Robot
from planners.global_astar import GlobalAStarPlanner, PlanResult, SubGoalSelector, polyline_length
from sensors.lidar import Lidar2D

T_MAX = getattr(config, "SIM_T_MAX", 120.0)
GOAL_TOL = getattr(config, "GOAL_TOL", 0.3)
LOOKAHEAD = getattr(config, "LOOKAHEAD", 1.5)
STUCK_WINDOW = getattr(config, "STUCK_WINDOW", 10.0)
STUCK_DIST = getattr(config, "STUCK_DIST", 0.3)
CLEAR_CAP = 1.0                                   # [m] khoang cach toi da duoc do/ghi khi do khoang trong voi vat can tinh


class NoPedestrians:
    """Khong co nguoi di bo (kich ban S1 va cac phep thu ban dau)."""

    def reset(self, rng=None) -> None:
        pass

    def step(self, dt: float) -> None:
        pass

    @property
    def circles(self) -> np.ndarray:
        return np.zeros((0, 3))


@dataclass
class SimResult:
    status: str
    goal: tuple
    time_s: float                                  # thoi diem ket thuc [s]
    n_steps: int
    travelled: float                               # quang duong robot thuc di [m]
    global_plan: PlanResult
    waypoints: np.ndarray
    log: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success"

    def summary(self) -> dict:
        """Cac so lieu vo huong cua mot lan chay (dau vao cho evaluation/metrics)."""
        lg = self.log
        # Mốc tham chiếu độ dài: đường gấp khúc của waypoint (đường ngắn nhất "gần như tự do góc").
        # Độ dài lưới của A* (global_plan.length) dài hơn ~2-4% vì đi chéo theo bậc thang.
        g_len = polyline_length(self.waypoints) if len(self.waypoints) > 1 else float("nan")
        w = lg.get("w", np.zeros(0))
        return {
            "status": self.status,
            "success": self.success,
            "collision": self.status == "collision",
            "time_s": self.time_s,
            "travelled_m": self.travelled,
            "global_length_m": g_len,
            "astar_grid_length_m": self.global_plan.length if self.global_plan.found else float("nan"),
            "path_ratio": self.travelled / g_len if g_len and g_len > 0 else float("nan"),
            "min_clearance_static_m": float(np.min(lg["d_static"])) if len(lg.get("d_static", [])) else float("nan"),
            "min_gap_ped_m": float(np.min(lg["d_ped"])) if len(lg.get("d_ped", [])) else float("nan"),
            "smoothness_w": float(np.abs(np.diff(w)).sum()) if len(w) > 1 else 0.0,
            "mean_compute_ms": 1000.0 * float(np.mean(lg["t_compute"])) if len(lg.get("t_compute", [])) else float("nan"),
        }


class Simulator:
    """Mo phong mot robot giao hang di tu Start toi mot diem dich.

    known_map : HospitalMap A* BIET (tuong, cua, noi that co dinh).
    world_map : HospitalMap THAT dung de do LiDAR va kiem tra va cham
                (mac dinh = known_map; kich ban 1 truyen ``known.with_obstacles(...)``).

    Vi du::

        sim = Simulator(known, world)
        res = sim.run(DWAPlanner(), "G2", pedestrians=peds, seed=1001)
        print(res.status, res.summary())
    """

    def __init__(self, known_map, world_map=None, lidar: Lidar2D | None = None,
                 heuristic: str = "octile", lookahead: float | None = None,
                 goal_tol: float | None = None, t_max: float | None = None,
                 stuck_window: float | None = None, stuck_dist: float | None = None,
                 dt: float | None = None, robot: Robot | None = None):
        self.known = known_map
        self.world = world_map if world_map is not None else known_map
        self.lidar = lidar if lidar is not None else Lidar2D()
        self.global_planner = GlobalAStarPlanner(known_map, heuristic)
        self.lookahead = LOOKAHEAD if lookahead is None else float(lookahead)
        self.goal_tol = GOAL_TOL if goal_tol is None else float(goal_tol)
        self.t_max = T_MAX if t_max is None else float(t_max)
        self.stuck_window = STUCK_WINDOW if stuck_window is None else float(stuck_window)
        self.stuck_dist = STUCK_DIST if stuck_dist is None else float(stuck_dist)
        self.dt = config.DT if dt is None else float(dt)
        self.robot = robot if robot is not None else Robot()
        self._plan_cache: dict = {}
        self._win = int(math.ceil(CLEAR_CAP / self.world.res)) + 1

    # ------------------------------------------------------------------ tien ich
    def _resolve_goal(self, goal) -> tuple:
        if isinstance(goal, str):
            return tuple(self.known.goals[goal])
        return (float(goal[0]), float(goal[1]))

    def _global_plan(self, start, goal):
        """A* + rut gon, co nho de 360 lan chay khong lap lai A* nhieu lan."""
        key = (round(start[0], 3), round(start[1], 3), round(goal[0], 3), round(goal[1], 3))
        if key not in self._plan_cache:
            plan = self.global_planner.plan(start, goal)
            wps = self.global_planner.simplify(plan.path_xy) if plan.found else np.zeros((0, 2))
            self._plan_cache[key] = (plan, wps)
        return self._plan_cache[key]

    def static_clearance(self, x: float, y: float) -> float:
        """Khoang cach tu TAM robot toi be mat vat can tinh gan nhat tren the gioi that [m].

        Chinh xac (do toi tung o vat can, coi o la hinh vuong), toi da CLEAR_CAP. Ngoai ban do tra -1.
        Va cham khi gia tri nay < ban kinh robot.
        """
        w = self.world
        if not (0.0 <= x < w.width and 0.0 <= y < w.height):
            return -1.0
        res, k = w.res, self._win
        c0, r0 = int(x // res), int(y // res)
        c1, c2 = max(0, c0 - k), min(w.n_cols, c0 + k + 1)
        r1, r2 = max(0, r0 - k), min(w.n_rows, r0 + k + 1)
        sub = w.grid[r1:r2, c1:c2]
        if not sub.any():
            return CLEAR_CAP
        rr, cc = np.nonzero(sub)
        cx, cy = (cc + c1 + 0.5) * res, (rr + r1 + 0.5) * res
        dx = np.maximum(np.abs(x - cx) - res / 2.0, 0.0)
        dy = np.maximum(np.abs(y - cy) - res / 2.0, 0.0)
        return float(min(np.hypot(dx, dy).min(), CLEAR_CAP))

    @staticmethod
    def pedestrian_gap(x: float, y: float, radius: float, circles: np.ndarray) -> float:
        """Khoang trong be mat-be mat giua robot va nguoi gan nhat [m] (< 0 la da cham). Khong co nguoi: inf."""
        if circles is None or len(circles) == 0:
            return float("inf")
        c = np.asarray(circles, dtype=float).reshape(-1, 3)
        return float((np.hypot(c[:, 0] - x, c[:, 1] - y) - c[:, 2] - radius).min())

    # ------------------------------------------------------------------ chay
    def run(self, local_planner, goal, pedestrians=None, seed: int | None = None,
            start=None, record_pred: bool = False, callback=None) -> SimResult:
        """Chay mot lan mo phong. Khong bao loi khi that bai (xem ``status``).

        goal      : ten ('G1'...) hoac toa do (x, y).
        seed      : int -> cac nguon ngau nhien (nhieu LiDAR, nguoi di bo) co dinh va doc lap; None -> ngau nhien.
        callback  : ham goi moi buoc voi mot dict {k, t, state, scan, sub_goal, circles, pred_traj} (de ve hoat hinh).
        record_pred: luu pred_traj cua bo dieu huong moi buoc vao log['pred_traj'] (ton bo nho).
        """
        goal_xy = self._resolve_goal(goal)
        start_xy = tuple(self.known.start if start is None else start)
        peds = pedestrians if pedestrians is not None else NoPedestrians()

        plan, wps = self._global_plan(start_xy, goal_xy)
        empty = SimResult("no_path", goal_xy, 0.0, 0, 0.0, plan, wps)
        if not plan.found:
            return empty

        # ---- chuan bi lan chay ----
        seq = np.random.SeedSequence(seed)
        rng_lidar, rng_ped = (np.random.default_rng(s) for s in seq.spawn(2))
        self.lidar.rng = rng_lidar
        peds.reset(rng_ped)
        if hasattr(local_planner, "reset"):
            local_planner.reset()

        selector = SubGoalSelector(wps, lookahead=self.lookahead, goal_tol=self.goal_tol)
        sub0, _ = selector.update(start_xy)
        theta0 = math.atan2(sub0[1] - start_xy[1], sub0[0] - start_xy[0]) if np.hypot(*(sub0 - start_xy)) > 1e-9 else 0.0
        robot = self.robot
        robot.reset(start_xy[0], start_xy[1], theta0, 0.0, 0.0)
        r_robot = robot.radius
        dt = self.dt
        n_max = int(round(self.t_max / dt))
        n_stuck = max(1, int(round(self.stuck_window / dt)))

        rows = {k: [] for k in ("t", "x", "y", "theta", "v", "w", "d_static", "d_ped", "t_compute")}
        sub_goals, preds = [], []
        status, travelled, k = "timeout", 0.0, 0
        state = robot.get_state()

        for k in range(n_max + 1):
            t = k * dt
            circles = peds.circles
            x, y = float(state[0]), float(state[1])

            # ---- 1. do, kiem tra ----
            d_static = self.static_clearance(x, y)
            d_ped = self.pedestrian_gap(x, y, r_robot, circles)
            sub_goal, reached = selector.update((x, y))

            rows["t"].append(t)
            rows["x"].append(x)
            rows["y"].append(y)
            rows["theta"].append(float(state[2]))
            rows["v"].append(float(state[3]))
            rows["w"].append(float(state[4]))
            rows["d_static"].append(d_static)
            rows["d_ped"].append(d_ped)
            sub_goals.append(sub_goal)

            if d_static < r_robot or d_ped < 0.0:                       # va cham: uu tien hon thanh cong
                status = "collision"
                break
            if reached:
                status = "success"
                break
            if k >= n_stuck:
                moved = math.hypot(x - rows["x"][k - n_stuck], y - rows["y"][k - n_stuck])
                if moved < self.stuck_dist:
                    status = "stuck"
                    break
            if k == n_max:
                status = "timeout"
                break

            # ---- 2. cam bien + bo dieu huong ----
            scan = self.lidar.scan(self.world, state[:3], circles=circles)
            t0 = time.perf_counter()
            out = local_planner.compute_control(state.copy(), sub_goal.copy(), scan.points, dt)
            rows["t_compute"].append(time.perf_counter() - t0)
            v_cmd = float(np.asarray(out[0]).item())
            w_cmd = float(np.asarray(out[1]).item())
            pred = out[2] if len(out) > 2 else None
            if record_pred:
                preds.append(None if pred is None else np.asarray(pred, dtype=float).copy())

            if callback is not None:
                callback({"k": k, "t": t, "state": state.copy(), "scan": scan, "sub_goal": sub_goal,
                          "circles": np.asarray(circles).copy(), "pred_traj": pred})

            # ---- 3. tien robot va nguoi di bo ----
            prev = (x, y)
            state = robot.step(v_cmd, w_cmd, dt)
            peds.step(dt)
            travelled += math.hypot(state[0] - prev[0], state[1] - prev[1])

        log = {key: np.asarray(val, dtype=float) for key, val in rows.items()}
        log["sub_goal"] = np.asarray(sub_goals, dtype=float).reshape(-1, 2)
        if record_pred:
            log["pred_traj"] = preds
        return SimResult(status=status, goal=goal_xy, time_s=k * dt, n_steps=k, travelled=travelled,
                         global_plan=plan, waypoints=wps, log=log)


                         