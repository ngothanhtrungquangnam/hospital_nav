    # planners/base.py - Hợp đồng giao tiếp chuẩn cho các Local Planner
    import numpy as np

    class LocalPlanner:
        name = "BasePlanner"

        def reset(self):
            """Khởi tạo lại trạng thái trước mỗi lần chạy mô phỏng"""
            pass

        def compute_control(self, state, sub_goal, points, dt):
            """
            Hợp đồng giao tiếp bắt buộc cho mọi thuật toán điều khiển (DWA, APF, VFH, Stub)
            
            Đầu vào:
            - state    : list/array 5 phần tử [x, y, theta, v, w] (m, m, rad, m/s, rad/s)
            - sub_goal : list/array 2 phần tử [gx, gy] (tọa độ đích cục bộ, mét)
            - points   : numpy ndarray (N, 2) chứa tọa độ [x, y] điểm LiDAR chạm vật cản (hệ thế giới). N có thể = 0.
            - dt       : float, bước thời gian mô phỏng (giây)

            Đầu ra:
            - v_cmd     : float, vận tốc dài mong muốn (m/s)
            - w_cmd     : float, vận tốc góc mong muốn (rad/s)
            - pred_traj : numpy ndarray (M, 2) chứa quỹ đạo dự đoán để vẽ mô phỏng, hoặc None.
            """
            raise NotImplementedError("Lớp con bắt buộc phải viết đè hàm này!")