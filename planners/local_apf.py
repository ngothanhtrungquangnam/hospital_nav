"""planners/local_apf.py -- Bo dieu khien cuc bo theo Truong the nang nhan tao (APF), ban on dinh.

Nguoi phu trach: TV1.   Giao dien giu nguyen (LocalPlanner):  compute_control(state, sub_goal, points, dt)
                         -> (v_cmd, w_cmd, pred_traj),  reset(),  name.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np

from config import (
    DT,
    MAX_ACCEL,
    MAX_W_ACCEL,
    ROBOT_RADIUS,
    V_MAX,
    V_MIN,
    W_MAX,
)
from planners.base import LocalPlanner

_EPS = 1e-9
_CAP = 5.0  # [m] khoang cach gan dung cho "khong thay vat can"


def _wrap_angle(angle: float) -> float:
    """Chuan hoa goc ve khoang [-pi, pi)."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class APFPlanner(LocalPlanner):
    """Bo dieu khien cuc bo APF: hut ve sub-goal, day khoi vat can, co phanh va thoat ket."""

    name = "APFPlanner"

    def __init__(
        self,
        k_att: float = 1.65,
        k_rep: float = 0.52,
        influence_dist: float = 1.45,
        robot_radius: float = ROBOT_RADIUS,
        clearance_margin: float = 0.24,
        goal_slow_radius: float = 0.75,
        v_max: float = V_MAX,
        v_min: float = V_MIN,
        w_max: float = W_MAX,
        max_accel: float = MAX_ACCEL,
        max_w_accel: float = MAX_W_ACCEL,
        pred_steps: int = 12,
        pred_dt: float | None = None,
        # ---- tham so mo rong (co mac dinh -> code goi cu van chay) ----
        n_sectors: int = 72,
        k_heading: float = 1.30,
        force_smooth: float = 0.52,
        goal_slow_floor: float = 0.35,
        lane_margin: float = 0.14,
        stop_gap: float = 0.22,
        brake_ratio: float = 0.85,
        block_dist: float = 1.35,
        k_tangent: float = 1.25,
        side_hold: float = 1.5,
        ttc_horizon: float = 1.45,
        stuck_window: float = 2.2,
        stuck_dist: float = 0.22,
        escape_time: float = 1.8,
        traj_check_steps: int = 8,
        traj_min_scale: float = 0.25,
        desired_smooth: float = 0.0,
        angle_deadband: float = 0.025,
    ) -> None:
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.influence_dist = float(max(influence_dist, robot_radius + 0.2))
        self.robot_radius = float(max(robot_radius, 0.0))
        self.clearance_margin = float(max(clearance_margin, 0.05))
        self.goal_slow_radius = float(max(goal_slow_radius, 0.1))
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.w_max = float(abs(w_max))
        self.max_accel = float(max(max_accel, 0.0))
        self.max_w_accel = float(max(max_w_accel, 0.0))
        self.pred_steps = int(max(pred_steps, 1))
        self.pred_dt = float(pred_dt) if pred_dt is not None else None

        self.n_sectors = int(max(n_sectors, 12))
        self.k_heading = float(k_heading)
        self.force_smooth = float(np.clip(force_smooth, 0.0, 0.95))
        self.goal_slow_floor = float(np.clip(goal_slow_floor, 0.0, 1.0))
        self.lane_margin = float(max(lane_margin, 0.0))
        self.stop_gap = float(max(stop_gap, 0.0))
        self.brake_ratio = float(np.clip(brake_ratio, 0.1, 1.0))
        self.block_dist = float(max(block_dist, 0.4))
        self.k_tangent = float(max(k_tangent, 0.0))
        self.side_hold = float(max(side_hold, 0.0))
        self.ttc_horizon = float(max(ttc_horizon, 0.0))
        self.stuck_window = float(max(stuck_window, 0.5))
        self.stuck_dist = float(max(stuck_dist, 0.05))
        self.escape_time = float(max(escape_time, 0.5))
        self.traj_check_steps = int(max(traj_check_steps, 1))
        self.traj_min_scale = float(np.clip(traj_min_scale, 0.0, 1.0))
        self.desired_smooth = float(np.clip(desired_smooth, 0.0, 0.9))
        self.angle_deadband = float(max(angle_deadband, 0.0))

        self.reset()

    # ------------------------------------------------------------------ trang thai
    def reset(self) -> None:
        """Khoi tao lai trang thai truoc moi lan chay mo phong."""
        self._prev_v = 0.0
        self._prev_w = 0.0
        self._t = 0.0
        self._f: np.ndarray | None = None
        self._desired_prev: float | None = None
        self._side = 0
        self._free_t = 0.0
        self._dlane_prev: float | None = None
        self._closing = 0.0
        self._hist: deque[tuple[float, float, float, float]] = deque()
        self._arc_total = 0.0
        self._last_xy: tuple[float, float] | None = None
        self._esc_until = -1.0
        self._esc_side = 0

    # ------------------------------------------------------------------ vat can
    @staticmethod
    def _clean_points(points: Sequence[Sequence[float]] | np.ndarray | None) -> np.ndarray:
        if points is None:
            return np.zeros((0, 2), dtype=float)
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=float)
        return pts[np.all(np.isfinite(pts), axis=1)]

    def _sector_indices(self, rel: np.ndarray) -> np.ndarray:
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        idx = ((angles + math.pi) / (2.0 * math.pi) * self.n_sectors).astype(int)
        return np.clip(idx, 0, self.n_sectors - 1)

    def _profile(self, rel: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Khoang cach be mat gan nhat theo tung o goc (n_sectors), mac dinh _CAP khi khong thay gi."""
        prof = np.full(self.n_sectors, _CAP, dtype=float)
        if len(s) > 0:
            idx = self._sector_indices(rel)
            np.minimum.at(prof, idx, np.minimum(s, _CAP))
        return prof

    def _side_score(self, prof: np.ndarray, ug: np.ndarray, side: int) -> float:
        """Do thoang phia ben `side` (+1 trai, -1 phai) cua huong muc tieu."""
        centers = (np.arange(self.n_sectors, dtype=float) + 0.5) * (2.0 * math.pi / self.n_sectors) - math.pi
        goal_ang = math.atan2(float(ug[1]), float(ug[0]))
        rel_brg = (centers - goal_ang + math.pi) % (2.0 * math.pi) - math.pi
        lo, hi = math.radians(15.0), math.radians(105.0)
        mask = (rel_brg * side > lo) & (rel_brg * side < hi)
        if not np.any(mask):
            return _CAP
        vals = prof[mask]
        return float(0.6 * np.min(vals) + 0.4 * np.mean(vals))

    def _choose_side(self, prof: np.ndarray, ug: np.ndarray) -> int:
        left = self._side_score(prof, ug, +1)
        right = self._side_score(prof, ug, -1)
        if abs(left - right) < 0.04:
            return -1
        return +1 if left > right else -1

    def _lane_gap(self, along: np.ndarray, lat: np.ndarray, half_width: float) -> float:
        """Quang duong robot (hinh tron ban kinh r) di duoc doc truc truoc khi cham mot diem trong lan."""
        r = self.robot_radius
        mask = (along > 0.0) & (lat < half_width)
        if not np.any(mask):
            return float("inf")
        lat_m = np.minimum(lat[mask], r)
        cut = np.sqrt(np.maximum(r * r - lat_m * lat_m, 0.0))
        return float(np.min(along[mask] - cut))

    # ------------------------------------------------------------------ luc
    def _repulsion(self, rel: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Luc day tu cac o goc (moi o chi lay diem gan nhat): do lon dang duong dong muc co chan."""
        d0 = max(self.influence_dist - self.robot_radius, 0.05)
        mask = s < d0
        if not np.any(mask):
            return np.zeros(2, dtype=float)

        rel_m = rel[mask]
        s_m = s[mask]
        idx = self._sector_indices(rel_m)

        order = np.argsort(s_m)
        _, first = np.unique(idx[order], return_index=True)
        sel = order[first]

        rs = rel_m[sel]
        ss = s_m[sel]
        s_min = 0.05
        q = np.clip((d0 - ss) / max(d0 - s_min, _EPS), 0.0, 1.0)
        mag = self.k_rep * 6.0 * q * q

        dc = np.maximum(np.hypot(rs[:, 0], rs[:, 1]), _EPS)
        u = -rs / dc[:, None]
        return (mag[:, None] * u).sum(axis=0)

    # ------------------------------------------------------------------ du doan & an toan quy dao
    def _rollout_trajectory(
        self,
        x: float,
        y: float,
        theta: float,
        v: float,
        w: float,
        step_dt: float,
    ) -> np.ndarray:
        """Du doan quy dao ngan (M, 2) tu trang thai hien tai theo (v, w)."""
        pts = np.empty((self.pred_steps, 2), dtype=float)
        cx, cy, cth = x, y, theta
        for i in range(self.pred_steps):
            cx += v * math.cos(cth) * step_dt
            cy += v * math.sin(cth) * step_dt
            cth = _wrap_angle(cth + w * step_dt)
            pts[i, 0] = cx
            pts[i, 1] = cy
        return pts

    def _traj_safety_scale(
        self,
        traj: np.ndarray,
        obs_pts: np.ndarray,
        is_maneuvering: bool,
        closing_fast: bool,
    ) -> tuple[float, bool]:
        """Danh gia do an toan tren quy dao du doan."""
        if len(obs_pts) == 0 or len(traj) == 0:
            return 1.0, False

        k = min(self.traj_check_steps, len(traj))
        check_pts = traj[:k]
        dx = check_pts[:, 0:1] - obs_pts[:, 0][None, :]
        dy = check_pts[:, 1:2] - obs_pts[:, 1][None, :]
        clearances = np.hypot(dx, dy) - self.robot_radius
        step_min = clearances.min(axis=1)

        early_horizon = min(6 if closing_fast else 3, k)
        early_min = float(step_min[:early_horizon].min())
        stop_thresh = 0.30 if closing_fast else 0.05
        if early_min <= stop_thresh:
            return 0.0, True

        min_clearance = float(step_min.min())
        safe_band = max(self.clearance_margin + self.stop_gap, 0.16)
        if closing_fast:
            safe_band = max(safe_band, 0.75)

        if min_clearance >= safe_band:
            return 1.0, False

        if closing_fast and min_clearance < 0.48:
            return 0.0, True

        ratio = float(np.clip(min_clearance / safe_band, 0.0, 1.0))
        floor = 0.0 if closing_fast else (0.30 if is_maneuvering else self.traj_min_scale)
        scale = max(floor, ratio ** 1.3)
        return scale, False

    def _apply_rate_limits(
        self,
        raw_v: float,
        raw_w: float,
        cur_v: float,
        cur_w: float,
        step_dt: float,
        allow_stop: bool = False,
        emergency_brake: bool = False,
    ) -> tuple[float, float]:
        if self.max_accel > 0.0:
            accel_mult = 2.0 if (emergency_brake and raw_v <= 1e-6) else 1.0
            dv = self.max_accel * accel_mult * step_dt
            v_cmd = float(np.clip(raw_v, cur_v - dv, cur_v + dv))
        else:
            v_cmd = raw_v

        if self.max_w_accel > 0.0:
            dw = self.max_w_accel * step_dt
            w_cmd = float(np.clip(raw_w, cur_w - dw, cur_w + dw))
        else:
            w_cmd = raw_w

        v_floor = 0.0 if allow_stop else self.v_min
        v_cmd = float(np.clip(v_cmd, v_floor, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))
        return v_cmd, w_cmd

    # ------------------------------------------------------------------ chinh
    def compute_control(
        self,
        state: Sequence[float] | np.ndarray,
        sub_goal: Sequence[float] | np.ndarray,
        points: np.ndarray | None,
        dt: float,
    ) -> tuple[float, float, np.ndarray]:
        """Tinh (v_cmd, w_cmd, pred_traj) tu state, sub_goal, points va dt."""
        st = np.asarray(state, dtype=float).reshape(-1)
        if st.size < 3:
            raise ValueError("state phai co it nhat 3 phan tu [x, y, theta, ...]")
        x, y = float(st[0]), float(st[1])
        theta = _wrap_angle(float(st[2]))
        cur_v = float(st[3]) if st.size >= 4 and math.isfinite(st[3]) else self._prev_v
        cur_w = float(st[4]) if st.size >= 5 and math.isfinite(st[4]) else self._prev_w

        sg = np.asarray(sub_goal, dtype=float).reshape(-1)
        if sg.size < 2:
            raise ValueError("sub_goal phai co dang [gx, gy]")
        gx, gy = float(sg[0]), float(sg[1])

        step_dt = float(dt) if dt is not None and math.isfinite(float(dt)) and float(dt) > 0.0 else DT
        horizon_dt = self.pred_dt if self.pred_dt is not None else step_dt
        self._t += step_dt

        if self._last_xy is not None:
            self._arc_total += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)

        pts = self._clean_points(points)
        dist_goal = math.hypot(gx - x, gy - y)
        if dist_goal <= 0.05:
            self._prev_v = 0.0
            self._prev_w = 0.0
            self._desired_prev = None
            return 0.0, 0.0, self._rollout_trajectory(x, y, theta, 0.0, 0.0, horizon_dt)

        ug = np.array([(gx - x) / dist_goal, (gy - y) / dist_goal], dtype=float)
        goal_heading = math.atan2(float(ug[1]), float(ug[0]))
        r = self.robot_radius

        # ---- vat can trong he robot / he muc tieu ----
        if len(pts) > 0:
            rel = pts - np.array([x, y], dtype=float)
            dc = np.hypot(rel[:, 0], rel[:, 1])
            s = np.maximum(dc - r, 1e-3)
            c_t, s_t = math.cos(theta), math.sin(theta)
            xr = rel[:, 0] * c_t + rel[:, 1] * s_t
            yr = -rel[:, 0] * s_t + rel[:, 1] * c_t
            along_g = rel[:, 0] * ug[0] + rel[:, 1] * ug[1]
            lat_g = np.abs(rel[:, 0] * ug[1] - rel[:, 1] * ug[0])
            s_min_all = float(s.min())
        else:
            rel = np.zeros((0, 2), dtype=float)
            s = np.zeros(0, dtype=float)
            xr = yr = along_g = lat_g = np.zeros(0, dtype=float)
            s_min_all = float("inf")

        prof = self._profile(rel, s)
        half_w = r + self.lane_margin

        # ---- lan phia truoc (theo huong robot va huong muc tieu) + toc do tien lai gan ----
        d_lane_robot = self._lane_gap(xr, np.abs(yr), half_w)
        d_lane_goal = self._lane_gap(along_g, lat_g, half_w + 0.06)
        d_lane = min(d_lane_robot, d_lane_goal)
        if self._dlane_prev is not None and math.isfinite(d_lane):
            raw_closing = (self._dlane_prev - d_lane) / step_dt
            blend = 0.80 if raw_closing > self._closing else 0.45
            self._closing = (1.0 - blend) * self._closing + blend * float(np.clip(raw_closing, -3.0, 5.0))
        else:
            self._closing *= 0.5
        self._dlane_prev = d_lane if math.isfinite(d_lane) else None
        obst_closing = max(0.0, self._closing - max(cur_v, 0.0))
        near_closing = (obst_closing > 0.12 or self._closing > 0.65) and (d_lane < 1.15 or s_min_all < 0.85)
        closing_fast = (obst_closing > 0.12 or self._closing > 0.65) and d_lane < 2.40

        # ---- lan theo huong muc tieu: co bi chan khong? ----
        d_goal_lane = self._lane_gap(along_g, lat_g, half_w)
        if dist_goal > self.block_dist:
            d_goal_lane = min(d_goal_lane, float("inf"))
        blocked = d_goal_lane < self.block_dist

        if blocked:
            self._free_t = 0.0
            if self._side == 0:
                self._side = self._choose_side(prof, ug)
        elif self._side != 0:
            self._free_t += step_dt
            if self._free_t > self.side_hold:
                self._side = 0
                self._free_t = 0.0

        # ---- phat hien ket -> thoat ----
        self._hist.append((self._t, x, y, self._arc_total))
        while self._hist and self._t - self._hist[0][0] > self.stuck_window:
            self._hist.popleft()

        has_nearby_obs = (len(pts) > 0) and (s_min_all < min(self.influence_dist, 1.10))
        escaping = self._t < self._esc_until
        if (
            not escaping
            and has_nearby_obs
            and len(self._hist) > 1
            and (self._t - self._hist[0][0]) >= (self.stuck_window - 1e-9)
            and dist_goal > 0.4
        ):
            net_moved = math.hypot(x - self._hist[0][1], y - self._hist[0][2])
            arc_moved = self._arc_total - self._hist[0][3]
            if net_moved < self.stuck_dist and arc_moved < max(self.stuck_dist * 1.4, 0.28):
                self._esc_side = self._choose_side(prof, ug)
                self._esc_until = self._t + self.escape_time
                self._side = self._esc_side
                self._free_t = 0.0
                self._hist.clear()
                escaping = True

        # ---- tong hop luc ----
        att_mag = self.k_att * max(0.38, min(1.0, dist_goal / self.goal_slow_radius))
        f_att = att_mag * ug
        f_rep = self._repulsion(rel, s) if len(pts) > 0 else np.zeros(2, dtype=float)

        if not blocked and not escaping:
            par = float(f_rep @ ug)
            if par < 0.0:
                f_rep = f_rep - par * ug

        is_maneuvering = escaping or (self._side != 0 and blocked)

        f_tan = np.zeros(2, dtype=float)
        if self._side != 0 and (blocked or escaping):
            urgency = (
                1.0
                if escaping
                else float(np.clip((self.block_dist - d_goal_lane) / max(self.block_dist - 0.2, _EPS), 0.0, 1.0))
            )
            tvec = np.array([-ug[1], ug[0]], dtype=float) * self._side
            tan_boost = 1.25 if closing_fast else 1.0
            f_tan = self.k_tangent * self.k_att * max(urgency, 0.45) * tan_boost * tvec

        if escaping:
            f = 0.45 * f_att + f_rep + 1.3 * self.k_att * np.array([-ug[1], ug[0]], dtype=float) * self._esc_side
        else:
            f = f_att + f_rep + f_tan

        raw_angle_err = (
            _wrap_angle(math.atan2(float(f[1]), float(f[0])) - theta)
            if math.hypot(float(f[0]), float(f[1])) > _EPS
            else _wrap_angle(goal_heading - theta)
        )
        sharp_turn_near_obs = abs(raw_angle_err) > 0.45 and s_min_all < 0.90
        alpha = 0.25 if sharp_turn_near_obs else self.force_smooth
        self._f = f if self._f is None else alpha * self._f + (1.0 - alpha) * f
        f = self._f

        # ---- huong mong muon va omega ----
        if math.hypot(float(f[0]), float(f[1])) <= _EPS:
            desired = goal_heading
        else:
            diff = _wrap_angle(math.atan2(float(f[1]), float(f[0])) - goal_heading)
            max_bias = math.radians(85.0 if is_maneuvering else 95.0)
            clamped_diff = float(np.clip(diff, -max_bias, max_bias))
            desired = _wrap_angle(goal_heading + clamped_diff)

        if self.desired_smooth > 0.0:
            if self._desired_prev is None:
                desired = desired
            else:
                d_err = _wrap_angle(desired - self._desired_prev)
                desired = _wrap_angle(self._desired_prev + (1.0 - self.desired_smooth) * d_err)
            self._desired_prev = desired

        angle_err = _wrap_angle(desired - theta)
        if not is_maneuvering and s_min_all > 1.2 and abs(angle_err) < self.angle_deadband:
            eff_err = 0.0
        else:
            eff_err = angle_err

        raw_w = float(np.clip(self.k_heading * eff_err, -self.w_max, self.w_max))

        # ---- toc do: quay -> gan sub-goal -> khoang trong -> quang duong dung ----
        turn = max(0.22 if sharp_turn_near_obs else 0.35, math.cos(angle_err)) ** 0.85
        goal_f = min(1.0, max(self.goal_slow_floor, dist_goal / self.goal_slow_radius))
        prox = 1.0 if s_min_all >= self.clearance_margin else max(0.40, s_min_all / self.clearance_margin)
        target_v = self.v_max * turn * goal_f * prox

        must_stop = False
        if math.isfinite(d_lane):
            a_dec = self.brake_ratio * (self.max_accel if self.max_accel > 0.0 else 1.0)
            extra_margin = self.stop_gap + (0.35 if near_closing else 0.0)
            d_eff = d_lane - extra_margin - obst_closing * (self.ttc_horizon * (1.10 if near_closing else 1.0))
            stop_cut = 0.25 if near_closing else 0.03
            if d_eff <= stop_cut:
                target_v = 0.0
                must_stop = True
            else:
                target_v = min(target_v, math.sqrt(2.0 * a_dec * d_eff))

        allow_stop_now = must_stop or (near_closing and d_lane < 0.85)
        raw_v = float(np.clip(target_v, 0.0 if allow_stop_now else self.v_min, self.v_max))
        v_cmd, w_cmd = self._apply_rate_limits(
            raw_v,
            raw_w,
            cur_v,
            cur_w,
            step_dt,
            allow_stop=allow_stop_now,
            emergency_brake=near_closing,
        )

        # ---- danh gia an toan tren quy dao du doan ----
        pred_traj = self._rollout_trajectory(x, y, theta, v_cmd, w_cmd, horizon_dt)
        traj_scale, traj_stop = self._traj_safety_scale(pred_traj, pts, is_maneuvering, near_closing)
        if traj_stop or traj_scale < 1.0:
            adj_raw_v = 0.0 if traj_stop else float(np.clip(raw_v * traj_scale, 0.0, self.v_max))
            stop_flag = allow_stop_now or traj_stop or adj_raw_v <= 1e-6
            v_cmd, w_cmd = self._apply_rate_limits(
                adj_raw_v,
                raw_w,
                cur_v,
                cur_w,
                step_dt,
                allow_stop=stop_flag,
                emergency_brake=(near_closing or traj_stop),
            )
            pred_traj = self._rollout_trajectory(x, y, theta, v_cmd, w_cmd, horizon_dt)

        self._prev_v = v_cmd
        self._prev_w = w_cmd
        return v_cmd, w_cmd, pred_traj