"""tests/stub_planner.py -- Bo dieu huong cuc bo "gia" de thu simulator khi chua co DWA/APF/VFH.

Nguoi phu trach: TV1.

StubPlanner chi quay ve phia sub-goal roi chay toi (dieu khien ti le), KHONG tranh vat can.
  * brake_distance = None : mu hoan toan -> dam vao vat can chua biet (dung de thu 'collision').
  * brake_distance = d    : dung lai neu LiDAR thay diem cham trong d [m] trong non hep phia truoc
                            (+- brake_cone_deg, mac dinh 15 do) -> khong dam nhung se dung mai o truoc
                            vat can (dung de thu 'stuck'). Non hep de KHONG phanh vi khung cua / tuong ben.
"""
from __future__ import annotations

import math

import numpy as np

import config
from planners.base import LocalPlanner


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class StubPlanner(LocalPlanner):
    name = "Stub"

    def __init__(self, brake_distance: float | None = None, v_nominal: float = 0.8, k_w: float = 2.0,
                 brake_cone_deg: float = 15.0):
        self.brake_distance = brake_distance
        self.brake_cone = math.radians(brake_cone_deg)
        self.v_nominal = v_nominal
        self.k_w = k_w

    def reset(self) -> None:
        pass

    def compute_control(self, state, sub_goal, points, dt):
        x, y, theta = state[0], state[1], state[2]
        err = _wrap(math.atan2(sub_goal[1] - y, sub_goal[0] - x) - theta)
        w_cmd = float(np.clip(self.k_w * err, -config.W_MAX, config.W_MAX))
        v_cmd = self.v_nominal * max(0.0, math.cos(err))             # lech huong nhieu thi cham lai

        if self.brake_distance is not None and len(points):
            rel = np.asarray(points) - np.array([x, y])
            dist = np.hypot(rel[:, 0], rel[:, 1])
            ang = np.abs((np.arctan2(rel[:, 1], rel[:, 0]) - theta + math.pi) % (2.0 * math.pi) - math.pi)
            if np.any((dist < self.brake_distance) & (ang < self.brake_cone)):
                v_cmd = 0.0
        return float(v_cmd), w_cmd, None