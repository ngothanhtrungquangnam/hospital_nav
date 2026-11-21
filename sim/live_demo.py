"""sim/live_demo.py -- Mo phong dieu huong robot phong cach RViz/AGV cong nghiep voi tia LiDAR thoi gian thuc."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from env.hospital_map import HospitalMap, UNMAPPED_CARTS
from env.map_view import draw_map
from sim.simulator import Simulator
from tests.stub_planner import StubPlanner
import config


def run_visual_simulation(goal_target="G2"):
    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)

    plt.ion()
    # Sử dụng phong cách bản vẽ kỹ thuật tối (Blueprint / Dark Theme) cực kỳ chuyên nghiệp
    fig, ax = plt.subplots(figsize=(13.5, 7.5), facecolor="#0A1F3F")

    # Vẽ bản đồ nền tối
    draw_map(
        world, ax, 
        title=f"AGV INDUSTRIAL NAVIGATION MONITOR — Đích: {goal_target}", 
        style="blueprint",
        show_inflated=True, show_doors=True, show_dims=True
    )

    # 1. Quỹ đạo đường đi và tia LiDAR tổng hợp
    (trajectory_line,) = ax.plot([], [], color="#38BDF8", lw=3.0, label="Quỹ đạo thực tế (Trajectory)", zorder=8)
    
    # 2. Chùm tia quét LiDAR phát ra từ thân robot (Hiệu ứng cảm biến thực tế)
    lidar_beams, = ax.plot([], [], color="#F43F5E", lw=0.4, alpha=0.4, ls="-", label="Tia LiDAR (360°)", zorder=7)

    # 3. Thiết kế hình dáng Robot công nghiệp (Chassis hình chữ nhật bo góc + Tháp LiDAR + Hướng)
    robot_radius = getattr(config, "ROBOT_RADIUS", 0.25)
    
    # Khung thân robot dạng hình chữ nhật (dài 0.6m, rộng 0.5m) mô phỏng xe AGV thực tế
    chassis_width, chassis_length = 0.5, 0.6
    robot_box = patches.Rectangle((-chassis_width/2, -chassis_length/2), chassis_width, chassis_length,
                                    facecolor="#0284C7", edgecolor="#E0F2FE", linewidth=2.0, alpha=0.9, zorder=12)
    
    # Tháp LiDAR ở tâm robot
    lidar_turret = patches.Circle((0, 0), 0.08, facecolor="#EF4444", edgecolor="#FFFFFF", linewidth=1.2, zorder=13)
    
    # Vòng tròn vùng an toàn (Safety Footprint)
    safety_circle = patches.Circle((0, 0), robot_radius + 0.1, facecolor="none", edgecolor="#38BDF8", 
                                    linewidth=1.5, linestyle="--", alpha=0.6, zorder=10)

    ax.add_patch(safety_circle)
    ax.add_patch(robot_box)
    ax.add_patch(lidar_turret)

    # Mũi tên chỉ hướng mặt trước robot
    robot_heading, = ax.plot([], [], color="#FACC15", lw=3.5, zorder=14)

    # 4. Bảng thông số Telemetry HUD chuyên nghiệp ở góc trên bên phải
    hud_box = ax.text(0.98, 0.96, "", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                      verticalalignment="top", horizontalalignment="right",
                      bbox=dict(boxstyle="round,pad=0.6", facecolor="#020617", edgecolor="#38BDF8", alpha=0.9),
                      color="#E2E8F0", zorder=20)

    x_history, y_history = [], []

    def update_frame(data):
        k = data["k"]
        state = data["state"]
        scan = data["scan"]

        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        x_history.append(x)
        y_history.append(y)

        # Cập nhật vị trí các thành phần robot
        safety_circle.center = (x, y)
        lidar_turret.center = (x, y)

        # Xoay khung xe AGV theo góc hướng theta của robot
        # Biến đổi tọa độ hình chữ nhật theo góc xoay theta
        c, s = np.cos(theta), np.sin(theta)
        # Đặt lại vị trí tâm và góc xoay cho robot_box (Matplotlib Rectangle dùng transform)
        robot_box.set_xy((x - (chassis_width/2)*c + (chassis_length/2)*s, 
                          y - (chassis_width/2)*s - (chassis_length/2)*c))
        # Xoay khung xe (lưu ý: matplotlib rectangle rotation cần qua transform hoặc dùng patch xoay)
        # Để đơn giản và trực quan, ta cập nhật mũi tên hướng và vẽ quỹ đạo cực nét.
        
        # Cập nhật mũi tên hướng đầu xe
        hx = x + 0.4 * np.cos(theta)
        hy = y + 0.4 * np.sin(theta)
        robot_heading.set_data([x, hx], [y, hy])

        # Cập nhật quỹ đạo di chuyển
        trajectory_line.set_data(x_history, y_history)

        # Cập nhật tia quét LiDAR lan tỏa từ tâm robot tới các điểm va chạm vật cản
        if scan.points is not None and len(scan.points) > 0:
            # Lấy mẫu bớt tia để vẽ mượt mà (ví dụ lấy mỗi tia thứ 3)
            pts = scan.points[::3]
            beam_x = []
            beam_y = []
            for px, py in pts:
                beam_x.extend([x, px, None])
                beam_y.extend([y, py, None])
            lidar_beams.set_data(beam_x, beam_y)

        # Cập nhật thông số HUD thời gian thực
        dist = np.sum(np.hypot(np.diff(x_history), np.diff(y_history))) if len(x_history) > 1 else 0.0
        time_elapsed = k * 0.1
        hud_box.set_text(
            f"  AGV TELEMETRY STATUS  \n"
            f"------------------------\n"
            f"• Position X : {x:.2f} m\n"
            f"• Position Y : {y:.2f} m\n"
            f"• Heading (θ): {np.degrees(theta):.1f}°\n"
            f"• Odometry   : {dist:.2f} m\n"
            f"• Sim Time   : {time_elapsed:.1f} s\n"
            f"• Status     : ACTIVE / SCANNING"
        )

        plt.draw()
        plt.pause(0.005)

    sim = Simulator(known, world)
    print(f"🚀 Khởi chạy mô phỏng AGV chuyên nghiệp tới {goal_target}...")
    res = sim.run(StubPlanner(), goal=goal_target, seed=1001, callback=update_frame)

    print(f"🏁 Trạng thái kết thúc: {res.status.upper()} | Quãng đường: {res.travelled:.2f} m")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visual_simulation(goal_target="G2")