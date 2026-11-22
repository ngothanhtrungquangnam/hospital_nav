"""tests/test_draw_robot.py -- Kiem tra truc quan hinh dang robot (Footprint + Heading).

Chay:
    python -m tests.test_draw_robot
"""
import matplotlib.pyplot as plt
import numpy as np

from env.hospital_map import HospitalMap
from env.map_view import draw_map, draw_robot

if __name__ == "__main__":
    print("===== ĐANG KIỂM TRA HÌNH DẠNG ROBOT (FOOTPRINT & HEADING) =====")
    
    # 1. Khởi tạo bản đồ bệnh viện
    m = HospitalMap()
    
    # 2. Tạo khung hình với style Blueprint chuyên nghiệp
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    draw_map(
        m, 
        ax=ax, 
        style="blueprint", 
        title="KIỂM TRA HÌNH DẠNG ROBOT: Footprint (r = 0.35m) & Mũi tên hướng (Heading)"
    )
    
    # 3. Thử vẽ robot tại 3 vị trí với các góc quay khác nhau để kiểm tra hình hiển thị:
    
    # Vị trí 1: Tại điểm xuất phát Start (Đang quay mặt sang phải, theta = 0)
    sx, sy = m.start
    draw_robot(ax, sx, sy, theta=0.0, style="blueprint", color="#34D399")
    ax.text(sx, sy - 0.6, "Robot @ Start (theta=0)", color="#34D399", fontsize=7.5, fontweight="bold", ha="center")

    # Vị trí 2: Giữa hành lang chính (Đang di chuyển ngang sang phải, theta = 0)
    draw_robot(ax, 20.0, 11.0, theta=0.0, style="blueprint", color="#38BDF8")
    ax.text(20.0, 10.4, "Hành lang (theta=0)", color="#38BDF8", fontsize=7.5, fontweight="bold", ha="center")

    # Vị trí 3: Trước cửa phòng Khoa Ngoại G2 (Đang quay đầu đi lên phòng, theta = pi/2)
    gx, gy = m.goals["G2"]
    draw_robot(ax, gx, gy - 2.0, theta=1.57, style="blueprint", color="#F87171")
    ax.text(gx, gy - 2.6, "Rẽ vào phòng (theta=90°)", color="#F87171", fontsize=7.5, fontweight="bold", ha="center")

    print(">>> Đã render xong cửa sổ đồ họa. Hãy kiểm tra hình dạng robot!")
    plt.tight_layout()
    plt.show()