# config.py - Cấu hình chung cho toàn bộ dự án robot điều hướng bệnh viện

# Kích thước bản đồ (40m x 20m)
MAP_WIDTH = 40.0   # mét
MAP_HEIGHT = 20.0  # mét
RESOLUTION = 0.1   # Độ phân giải lưới (mét/ô)

# Thông số vật lý của Robot
ROBOT_RADIUS = 0.25 # Bán kính an toàn của robot (m)
V_MAX = 1.0         # Vận tốc dài tối đa (m/s)
V_MIN = -0.2        # Vận tốc lùi tối đa (m/s)
W_MAX = 1.5         # Vận tốc góc tối đa (rad/s)
MAX_ACCEL = 0.5     # Gia tốc dài tối đa (m/s^2)
MAX_W_ACCEL = 2.0   # Gia tốc góc tối đa (rad/s^2)

# Chu kỳ thời gian mô phỏng
DT = 0.1            # Bước thời gian cho mỗi vòng lặp (giây)