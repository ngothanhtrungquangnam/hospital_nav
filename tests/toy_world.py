import math
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Đảm bảo import được thư mục gốc hospital_nav
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from planners.local_dwa import DWAPlanner
from planners.local_vfh import VFHPlanner

PLANNERS = {
    "dwa": DWAPlanner,
    "vfh": VFHPlanner,
}

ROBOT_RADIUS = 0.25
GOAL_TOLERANCE = 0.25
LIDAR_RANGE = 6.0


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def make_circle_pts(cx: float, cy: float, r: float, n: int = 36) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def make_wall_pts(x1: float, y1: float, x2: float, y2: float, step: float = 0.1) -> np.ndarray:
    dist = math.hypot(x2 - x1, y2 - y1)
    n = max(2, int(dist / step) + 1)
    return np.column_stack([np.linspace(x1, x2, n), np.linspace(y1, y2, n)])


def build_scenario(name: str):
    """Trả về (start_state, goal, obstacle_points)."""
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    goal = np.array([6.0, 0.0], dtype=float)

    if name == "single":
        # 1 vật cản tròn chắn ngay giữa đường
        pts = make_circle_pts(3.0, 0.05, 0.45)
    elif name == "gap":
        # 2 vật cản tạo khe hẹp rộng ~1.0m ở giữa
        c1 = make_circle_pts(3.0, 0.90, 0.40)
        c2 = make_circle_pts(3.0, -0.90, 0.40)
        pts = np.vstack([c1, c2])
    elif name == "u_trap":
        # Bẫy hình chữ U chắn trước mặt đích
        back = make_wall_pts(3.6, -1.0, 3.6, 1.0)
        top = make_wall_pts(2.4, 1.0, 3.6, 1.0)
        bot = make_wall_pts(2.4, -1.0, 3.6, -1.0)
        pts = np.vstack([back, top, bot])
    else:
        raise ValueError(f"Kịch bản không hợp lệ: {name}")

    return start, goal, pts


def run_simulation(
    scenario: str = "single",
    planner_key: str = "dwa",
    dt: float = 0.1,
    max_steps: int = 250,
):
    key = planner_key.lower()
    if key not in PLANNERS:
        raise ValueError(f"Planner không hợp lệ: {planner_key}. Chọn: {list(PLANNERS.keys())}")

    planner = PLANNERS[key]()
    print(f"=== Đang test Planner: {planner.name} | Kịch bản: {scenario} ===")

    state, goal, all_pts = build_scenario(scenario)
    history = [state[:2].copy()]

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))

    for step in range(max_steps):
        # Giả lập LiDAR chỉ thấy các điểm trong tầm quét
        dists = np.linalg.norm(all_pts - state[:2], axis=1)
        visible_pts = all_pts[dists <= LIDAR_RANGE]

        # Kiểm tra va chạm thực tế
        if len(all_pts) > 0 and float(np.min(dists)) <= ROBOT_RADIUS:
            print(f"[STEP {step}] VA CHẠM! Khoảng cách nhỏ nhất: {np.min(dists):.3f} m")
            break

        # Kiểm tra tới đích
        dist_goal = float(np.linalg.norm(goal - state[:2]))
        if dist_goal <= GOAL_TOLERANCE:
            print(f"[STEP {step}] ĐÃ TỚI ĐÍCH sau {step * dt:.1f}s!")
            break

        # Tính lệnh điều khiển từ planner đang chọn
        v_cmd, w_cmd, pred_traj = planner.compute_control(state, goal, visible_pts, dt)

        # Cập nhật động học robot
        x, y, theta, _, _ = state
        x += v_cmd * math.cos(theta) * dt
        y += v_cmd * math.sin(theta) * dt
        theta = wrap_angle(theta + w_cmd * dt)
        state = np.array([x, y, theta, v_cmd, w_cmd], dtype=float)
        history.append(state[:2].copy())

        # Vẽ trực quan hóa
        ax.clear()
        ax.scatter(all_pts[:, 0], all_pts[:, 1], c="black", s=10, label="Vật cản")
        ax.plot(goal[0], goal[1], "r*", markersize=14, label="Đích")

        hist_arr = np.array(history)
        ax.plot(hist_arr[:, 0], hist_arr[:, 1], "b-", linewidth=2, label="Quỹ đạo đã đi")

        if pred_traj is not None and len(pred_traj) > 0:
            ax.plot(
                pred_traj[:, 0],
                pred_traj[:, 1],
                "g--",
                linewidth=1.5,
                label=f"{planner.name} dự đoán",
            )

        # Vẽ thân robot và hướng mũi
        ax.add_patch(Circle((x, y), ROBOT_RADIUS, fill=False, edgecolor="blue", linewidth=2))
        ax.arrow(
            x,
            y,
            0.35 * math.cos(theta),
            0.35 * math.sin(theta),
            head_width=0.1,
            color="blue",
        )

        dbg = getattr(planner, "last_debug", {})
        status = dbg.get("status", "ok")
        ax.set_title(
            f"[{planner.name}] Scenario: {scenario} | t={step*dt:.1f}s | v={v_cmd:.2f} w={w_cmd:.2f} | {status}"
        )
        ax.set_xlim(-0.8, 7.0)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        plt.pause(0.02)
    else:
        print(f"Hết {max_steps} bước nhưng robot chưa tới đích (còn cách {dist_goal:.2f} m).")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # Cú pháp: python tests/toy_world.py <scenario> <planner>
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "single"
    planner_name = sys.argv[2] if len(sys.argv) > 2 else "dwa"
    run_simulation(scenario_name, planner_name)