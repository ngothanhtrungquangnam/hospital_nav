"""run_dynamic_anim.py -- Mo phong hoat hoa robot AGV dieu huong thoi gian thuc.

Ket hop:
  * Thuat toan toan cuc: Global A* tren ban do benh vien (HospitalMap)
  * Bo dieu huong cuc bo: DWA, APF hoac VFH (chon qua command line)
  * Cam bien LiDAR 2D 360 do quét vat can tinh (tuong, xe day chua biet) va vat can dong
  * Cac tac nhan dong (Pedestrians): nguoi di kham, y ta day xe thuoc, xe lan, benh nhan

Cach chay (PowerShell):
  python run_dynamic_anim.py               # Mac dinh: DWA + A*, dich: G2
  python run_dynamic_anim.py dwa G2        # DWA toi Khoa Ngoai (G2)
  python run_dynamic_anim.py apf G1        # APF toi Khoa Noi (G1)
  python run_dynamic_anim.py vfh G3        # VFH toi Phong xet nghiem (G3)
  python run_dynamic_anim.py --planner vfh --goal G4
"""
from __future__ import annotations

import argparse
import math
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import config
from config import DT, V_MAX, WHEEL_RADIUS, W_MAX
from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from env.map_view import draw_map, draw_pedestrians, draw_robot, draw_scan
from env.pedestrians import Pedestrian, PedestrianManager
from planners.global_astar import GlobalAStarPlanner, SubGoalSelector
from planners.local_apf import APFPlanner
from planners.local_dwa import DWAPlanner
from planners.local_vfh import VFHPlanner
from sensors.lidar import Lidar2D

PLANNERS = {
    "dwa": DWAPlanner,
    "apf": APFPlanner,
    "vfh": VFHPlanner,
}

GOAL_NAMES = {
    "G1": "Khoa Nội",
    "G2": "Khoa Ngoại",
    "G3": "Xét nghiệm / Cấp cứu",
    "G4": "Thang máy / Kho",
}


def build_corridor_pedestrians(seed: int = 42) -> PedestrianManager:
    """Tao cac tac nhan dong di chuyen doc hanh lang benh vien."""
    ped_specs = [
        Pedestrian(
            15.0, 10.0,
            radius=0.28, speed=0.72,
            waypoints=[(15.0, 10.0), (26.0, 10.0), (15.0, 10.0)],
            loop=True, kind="visitor", ped_id=1,
        ),
        Pedestrian(
            25.5, 9.6,
            radius=0.28, speed=0.65,
            waypoints=[(25.5, 9.6), (14.5, 9.6), (25.5, 9.6)],
            loop=True, kind="nurse_cart", ped_id=2,
        ),
        Pedestrian(
            18.5, 10.3,
            radius=0.28, speed=0.58,
            waypoints=[(18.5, 10.3), (27.0, 10.3), (18.5, 10.3)],
            loop=True, kind="wheelchair", ped_id=3,
        ),
        Pedestrian(
            22.0, 9.2,
            radius=0.28, speed=0.75,
            waypoints=[(22.0, 9.2), (22.0, 10.6), (16.5, 9.8), (22.0, 9.2)],
            loop=True, kind="patient", ped_id=4,
        ),
    ]
    return PedestrianManager(ped_specs, seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description="Animation mô phỏng điều hướng AGV (Global A* + Local DWA/APF/VFH) né vật cản động."
    )
    parser.add_argument(
        "planner",
        nargs="?",
        default="dwa",
        choices=["dwa", "apf", "vfh"],
        help="Thuật toán cục bộ: dwa, apf, vfh (mặc định: dwa)",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="G2",
        choices=["G1", "G2", "G3", "G4"],
        help="Điểm đích: G1, G2, G3, G4 (mặc định: G2)",
    )
    parser.add_argument(
        "--planner", "-p",
        dest="opt_planner",
        choices=["dwa", "apf", "vfh"],
        help="Cờ chỉ định thuật toán",
    )
    parser.add_argument(
        "--goal", "-g",
        dest="opt_goal",
        choices=["G1", "G2", "G3", "G4"],
        help="Cờ chỉ định điểm đích",
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=600,
        help="Số khung hình tối đa (mặc định: 600)",
    )

    args = parser.parse_args()
    planner_key = (args.opt_planner or args.planner).lower().strip()
    goal_key = (args.opt_goal or args.goal).upper().strip()

    if planner_key not in PLANNERS:
        print(f"Lỗi: Planner '{planner_key}' không hợp lệ. Chọn: {list(PLANNERS.keys())}")
        sys.exit(1)

    # 1. Khởi tạo bản đồ & thuật toán toàn cục A*
    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)

    start_xy = tuple(known.start)
    if goal_key not in known.goals:
        print(f"Lỗi: Đích '{goal_key}' không tồn tại. Chọn: {list(known.goals.keys())}")
        sys.exit(1)
    goal_xy = tuple(known.goals[goal_key])

    print(f" Quy hoạch đường đi toàn cục A* từ {start_xy} -> {goal_key} {goal_xy}...")
    astar = GlobalAStarPlanner(known, heuristic="octile")
    plan = astar.plan(start_xy, goal_xy)
    if not plan.found:
        print(f"Lỗi: Không tìm được đường đi A* tới đích {goal_key}!")
        sys.exit(1)

    wps = astar.simplify(plan.path_xy)
    selector = SubGoalSelector(wps, lookahead=1.5, goal_tol=0.25)
    sub_goal, reached_init = selector.update(start_xy)

    # 2. Khởi tạo bộ điều hướng cục bộ & Cảm biến
    planner_cls = PLANNERS[planner_key]
    planner = planner_cls()
    if hasattr(planner, "reset"):
        planner.reset()

    mgr = build_corridor_pedestrians(seed=42)
    lidar = Lidar2D(rng=np.random.default_rng(42))

    # 3. Trạng thái robot ban đầu
    dx_init = sub_goal[0] - start_xy[0]
    dy_init = sub_goal[1] - start_xy[1]
    theta0 = math.atan2(dy_init, dx_init) if math.hypot(dx_init, dy_init) > 1e-6 else 0.0
    state = np.array([start_xy[0], start_xy[1], theta0, 0.0, 0.0], dtype=float)

    path_x = [float(state[0])]
    path_y = [float(state[1])]
    wheel_angle = 0.0
    wr = float(WHEEL_RADIUS) if float(WHEEL_RADIUS) > 0.0 else 0.075

    # 4. Thiết lập giao diện đồ họa Blueprint
    fig, ax = plt.subplots(figsize=(15.0, 8.0), facecolor="#0A1F3F")
    fig.subplots_adjust(left=0.045, right=0.975, top=0.93, bottom=0.07)

    goal_desc = GOAL_NAMES.get(goal_key, goal_key)
    draw_map(
        world, ax=ax,
        title=f"AGV DYNAMIC NAVIGATION [{planner.name} + A*] — Đích: {goal_key} ({goal_desc})",
        style="blueprint",
        show_inflated=True, show_doors=True, show_dims=True,
    )
    ax.set_xlim(-0.5, 40.5)
    ax.set_ylim(-0.5, 20.5)
    ax.set_aspect("equal", adjustable="datalim")

    # Đường A* toàn cục tham chiếu (nét đứt xanh lá mạ)
    ax.plot(
        plan.path_xy[:, 0], plan.path_xy[:, 1],
        "--", color="#10B981", linewidth=2.0, alpha=0.85,
        label=f"Đường toàn cục A* ({len(wps)} waypoints)", zorder=6
    )

    # Đích đến cuối cùng
    ax.scatter(
        [goal_xy[0]], [goal_xy[1]],
        c="#F43F5E", s=130, marker="X", edgecolors="#FFFFFF", linewidth=1.5,
        zorder=16, label=f"Đích ({goal_key})"
    )

    # Sub-goal động dọc đường đi A*
    (sub_goal_marker,) = ax.plot(
        [sub_goal[0]], [sub_goal[1]],
        marker="*", markersize=13, color="#FACC15", markeredgecolor="#FFFFFF",
        linestyle="None", zorder=15, label="Sub-goal A*"
    )

    # Quỹ đạo robot đã đi (xanh ngọc sáng)
    (path_line,) = ax.plot([], [], color="#00E5FF", linewidth=2.5, zorder=10, label="Quỹ đạo AGV")

    # Quỹ đạo cục bộ dự đoán (cam nổi bật)
    (pred_line,) = ax.plot([], [], "--", color="#FB923C", linewidth=2.0, zorder=11, label=f"Dự đoán {planner.name}")

    # Bảng Telemetry HUD góc trên bên phải
    hud_box = ax.text(
        0.985, 0.965, "", transform=ax.transAxes,
        fontsize=8.5, fontweight="bold",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#020617", edgecolor="#38BDF8", alpha=0.92),
        color="#E2E8F0", zorder=25,
    )

    ax.legend(loc="lower left", framealpha=0.90, fontsize=8.5)

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

    mission_complete = False

    def update(frame):
        nonlocal state, wheel_angle, mission_complete, sub_goal
        _clear_dynamic_artists()

        if frame > 0 and not mission_complete:
            # 1. Cập nhật vị trí các tác nhân động
            mgr.step(DT, world=world)

            # 2. Quét LiDAR 360 độ (nhìn thấy cả bản đồ, xe đẩy vàng và người)
            scan = lidar.scan(world, state[:3], circles=mgr.as_circles())
            obs_pts = (
                np.vstack([p for p in (scan.points_static, scan.points_dynamic) if len(p) > 0])
                if (len(scan.points_static) > 0 or len(scan.points_dynamic) > 0)
                else np.zeros((0, 2), dtype=float)
            )

            # 3. Cập nhật sub-goal từ thuật toán toàn cục A*
            current_xy = (float(state[0]), float(state[1]))
            sub_goal, reached = selector.update(current_xy)

            dist_to_final = math.hypot(goal_xy[0] - state[0], goal_xy[1] - state[1])
            if reached or dist_to_final <= 0.35:
                mission_complete = True
                v_cmd, w_cmd = 0.0, 0.0
                pred_traj = np.zeros((0, 2), dtype=float)
            else:
                # 4. Tính toán điều khiển cục bộ (DWA / APF / VFH)
                v_cmd, w_cmd, pred_traj = planner.compute_control(state, sub_goal, obs_pts, DT)
                v_cmd = float(np.clip(v_cmd, -0.2, V_MAX))
                w_cmd = float(np.clip(w_cmd, -W_MAX, W_MAX))

            # 5. Tích phân động học vi sai (Differential Drive)
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

        # Vẽ vật cản động với hình người, xe lăn, xe thuốc chuyên nghiệp
        draw_pedestrians(
            ax,
            mgr.as_circles(),
            style="blueprint",
            pedestrians=getattr(mgr, "pedestrians", None),
            visual_scale=1.20,
        )

        # Vẽ tia quét LiDAR
        draw_scan(ax, scan, style="blueprint", show_rays=True, ray_alpha=0.45, hit_size=20.0, ray_step=2)

        # Vẽ robot AGV
        draw_robot(
            ax,
            float(state[0]),
            float(state[1]),
            float(state[2]),
            style="blueprint",
            wheel_angle=wheel_angle,
            scale=1.5,
        )

        # Cập nhật đường đi và sub-goal
        path_line.set_data(path_x, path_y)
        sub_goal_marker.set_data([sub_goal[0]], [sub_goal[1]])

        if pred_traj is not None and len(pred_traj) > 0:
            pred_line.set_data(pred_traj[:, 0], pred_traj[:, 1])
        else:
            pred_line.set_data([], [])

        # Cập nhật bảng Telemetry HUD
        dist_to_final = math.hypot(goal_xy[0] - state[0], goal_xy[1] - state[1])
        dist_travelled = np.sum(np.hypot(np.diff(path_x), np.diff(path_y))) if len(path_x) > 1 else 0.0
        status_str = "ĐÃ ĐẾN ĐÍCH (GOAL REACHED)" if mission_complete else "ACTIVE / SCANNING & AVOIDING"
        status_color = "#4ADE80" if mission_complete else "#38BDF8"

        hud_box.set_text(
            f"  AGV DYNAMIC TELEMETRY  \n"
            f"-------------------------\n"
            f"• Planner    : {planner.name} + A*\n"
            f"• Goal Target: {goal_key} ({goal_desc})\n"
            f"• Position X : {state[0]:.2f} m\n"
            f"• Position Y : {state[1]:.2f} m\n"
            f"• Heading (θ): {math.degrees(state[2]):.1f}°\n"
            f"• Linear V   : {state[3]:.2f} m/s\n"
            f"• Angular W  : {state[4]:.2f} rad/s\n"
            f"• Odometry   : {dist_travelled:.2f} m\n"
            f"• Dist Target: {dist_to_final:.2f} m\n"
            f"• Sim Time   : {frame * DT:.1f} s\n"
            f"• Status     : {status_str}"
        )

        return [path_line, pred_line, sub_goal_marker]

    print(f"🎬 Bắt đầu mô phỏng hoạt họa [{planner.name} + A*] tới {goal_key} ({goal_desc}). Đang mở cửa sổ...")
    anim = FuncAnimation(fig, update, frames=args.frames, interval=30, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
