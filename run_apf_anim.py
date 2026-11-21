import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from config import DT, V_MAX, WHEEL_RADIUS, W_MAX
from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from env.map_view import draw_map, draw_pedestrians, draw_robot, draw_scan
from env.pedestrians import Pedestrian, PedestrianManager
from planners.local_apf import APFPlanner
from sensors.lidar import Lidar2D

world = HospitalMap().with_obstacles(UNMAPPED_CARTS)

# Các tác nhân động dọc hành lang để tăng mức tương tác & né tránh với robot
ped_specs = [
    Pedestrian(15.0, 10.0, radius=0.28, speed=0.75, waypoints=[(15.0, 10.0), (25.5, 10.0), (15.0, 10.0)], loop=True, kind="visitor", ped_id=1),
    Pedestrian(25.0, 9.6, radius=0.28, speed=0.68, waypoints=[(25.0, 9.6), (14.5, 9.6), (25.0, 9.6)], loop=True, kind="nurse_cart", ped_id=2),
    Pedestrian(18.0, 10.3, radius=0.28, speed=0.62, waypoints=[(18.0, 10.3), (26.0, 10.3), (18.0, 10.3)], loop=True, kind="wheelchair", ped_id=3),
    Pedestrian(21.5, 9.2, radius=0.28, speed=0.80, waypoints=[(21.5, 9.2), (21.5, 10.6), (16.5, 9.8), (21.5, 9.2)], loop=True, kind="patient", ped_id=4),
]
mgr = PedestrianManager(ped_specs, seed=42)
lidar = Lidar2D(rng=np.random.default_rng(42))

planner = APFPlanner(
    max_accel=0.85,
    max_w_accel=3.0,
    k_heading=1.45,
    force_smooth=0.45,
    goal_slow_floor=0.45,
    desired_smooth=0.12,
)
planner.reset()

state = np.array([13.0, 10.0, 0.0, 0.0, 0.0], dtype=float)
# Điểm đích tại khu vực Khoa Ngoại
sub_goal = np.array([25.0, 14.5], dtype=float)

path_x = [float(state[0])]
path_y = [float(state[1])]
wheel_angle = 0.0
wr = float(WHEEL_RADIUS) if float(WHEEL_RADIUS) > 0.0 else 0.075

fig, ax = plt.subplots(figsize=(15, 7.5))
fig.subplots_adjust(left=0.055, right=0.965, top=0.92, bottom=0.085)

draw_map(world, ax=ax, style="blueprint")
ax.set_xlim(-0.5, 40.5)
ax.set_ylim(-0.5, 20.5)
ax.set_aspect("equal", adjustable="datalim")

ax.scatter([sub_goal[0]], [sub_goal[1]], c="#76FF03", s=105, marker="*", zorder=6, label="Sub-goal (Khoa Ngoai)")

(path_line,) = ax.plot(path_x, path_y, color="#00E5FF", linewidth=2.2, zorder=10, label="Quy dao robot")
(pred_line,) = ax.plot([], [], "--", color="#FF9100", linewidth=1.7, zorder=11, label="Du doan APF")
ax.legend(loc="upper right", framealpha=0.92)

base_patches = set(ax.patches)
base_collections = set(ax.collections)
base_lines = set(ax.lines)
base_texts = set(ax.texts)


def _clear_dynamic_artists():
    for p in list(ax.patches):
        if p not in base_patches:
            p.remove()
    for c in list(ax.collections):
        if c not in base_collections:
            c.remove()
    for ln in list(ax.lines):
        if ln not in base_lines:
            ln.remove()
    for tx in list(ax.texts):
        if tx not in base_texts:
            tx.remove()


def update(frame):
    global state, wheel_angle
    _clear_dynamic_artists()

    if frame > 0:
        mgr.step(DT, world=world)
        scan = lidar.scan(world, state[:3], circles=mgr.as_circles())
        obs_pts = (
            np.vstack([p for p in (scan.points_static, scan.points_dynamic) if len(p) > 0])
            if (len(scan.points_static) > 0 or len(scan.points_dynamic) > 0)
            else np.zeros((0, 2), dtype=float)
        )

        v_cmd, w_cmd, pred_traj = planner.compute_control(state, sub_goal, obs_pts, DT)
        v_cmd = float(np.clip(v_cmd, -0.2, V_MAX))
        w_cmd = float(np.clip(w_cmd, -W_MAX, W_MAX))

        nx = float(state[0]) + v_cmd * float(np.cos(state[2])) * DT
        ny = float(state[1]) + v_cmd * float(np.sin(state[2])) * DT
        ntheta = float((state[2] + w_cmd * DT + math.pi) % (2.0 * math.pi) - math.pi)
        ds = math.hypot(nx - float(state[0]), ny - float(state[1]))
        direction = 1.0 if v_cmd >= -1e-6 else -1.0
        wheel_angle = float((wheel_angle + direction * (ds / wr) + math.pi) % (2.0 * math.pi) - math.pi)

        state = np.array([nx, ny, ntheta, v_cmd, w_cmd], dtype=float)
        path_x.append(nx)
        path_y.append(ny)
    else:
        scan = lidar.scan(world, state[:3], circles=mgr.as_circles())
        pred_traj = np.zeros((0, 2), dtype=float)

    draw_pedestrians(
        ax,
        mgr.as_circles(),
        style="blueprint",
        pedestrians=getattr(mgr, "pedestrians", None),
        visual_scale=1.25,
    )
    draw_scan(ax, scan, style="blueprint", show_rays=True, ray_alpha=0.55, hit_size=26.0, ray_step=1)
    draw_robot(
        ax,
        float(state[0]),
        float(state[1]),
        float(state[2]),
        style="blueprint",
        wheel_angle=wheel_angle,
        scale=1.7,
    )

    path_line.set_data(path_x, path_y)
    if pred_traj is not None and len(pred_traj) > 0:
        pred_line.set_data(pred_traj[:, 0], pred_traj[:, 1])
    else:
        pred_line.set_data([], [])

    dist_g = float(np.hypot(sub_goal[0] - state[0], sub_goal[1] - state[1]))
    ax.set_title(
        f"APFPlanner | t={frame * DT:.1f}s | v={state[3]:.2f} m/s | w={state[4]:.2f} rad/s | dist={dist_g:.2f}m"
    )
    return [path_line, pred_line]


anim = FuncAnimation(fig, update, frames=300, interval=30, blit=False)
plt.show()