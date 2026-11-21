# config.py - Cấu hình chung cho toàn bộ dự án robot điều hướng bệnh viện

# Kích thước bản đồ (40m x 20m)
MAP_WIDTH = 40.0   # mét
MAP_HEIGHT = 20.0  # mét
RESOLUTION = 0.1   # Độ phân giải lưới (mét/ô)

# Thong so ban do va phong vat can bo sung
DOOR_WIDTH = 1.2        # [m] bề rộng cửa phòng
CORRIDOR_WIDTH = 2.8    # [m] bề rộng hành lang chính
SAFETY_MARGIN = 0.10    # [m] biên an toàn khi phồng vật cản
# Thông số LiDAR 2D  
LIDAR_N_RAYS = 360       # số tia
LIDAR_RANGE = 5.0        # [m] tầm xa
LIDAR_FOV_DEG = 360.0    # [độ]
LIDAR_MIN_RANGE = 0.0    # [m] vùng mù gần
LIDAR_NOISE_STD = 0.0    # [m] nhiễu Gaussian
LIDAR_DROPOUT = 0.0      # xác suất rớt tia

# Thông số vật lý của Robot
ROBOT_RADIUS = 0.25 # Bán kính an toàn của robot (m)
V_MAX = 1.0         # Vận tốc dài tối đa (m/s)
V_MIN = -0.2        # Vận tốc lùi tối đa (m/s)
W_MAX = 1.5         # Vận tốc góc tối đa (rad/s)
MAX_ACCEL = 0.5     # Gia tốc dài tối đa (m/s^2)
MAX_W_ACCEL = 2.0   # Gia tốc góc tối đa (rad/s^2)
WHEEL_RADIUS = 0.075  # Bán kính bánh xe (m)
WHEEL_BASE = 0.36     # Khoảng cách giữa tâm hai bánh (m)
# Chu kỳ thời gian mô phỏng
DT = 0.1            # Bước thời gian cho mỗi vòng lặp (giây)