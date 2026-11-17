
"""
models/robot.py

Differential Drive Robot Model.

State convention:
    state = [x, y, theta, v, w]

Units:
    x, y       : meter (m)
    theta      : radian (rad)
    v          : meter/second (m/s)
    w          : radian/second (rad/s)
"""

import math
import numpy as np

import config


class Robot:
    """
    Mô hình động học robot vi sai hai bánh.

    Robot được điều khiển bởi vận tốc dài v và vận tốc góc w.

    Các giới hạn vật lý được lấy trực tiếp từ config.py.
    """

    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        v=0.0,
        w=0.0
    ):
        self.radius = config.ROBOT_RADIUS

        # Giới hạn vận tốc
        self.v_max = config.V_MAX
        self.v_min = config.V_MIN
        self.w_max = config.W_MAX

        # Giới hạn gia tốc
        self.max_accel = config.MAX_ACCEL
        self.max_w_accel = config.MAX_W_ACCEL

        self.reset(x, y, theta, v, w)

    @staticmethod
    def _wrap_angle(angle):
        """
        Chuẩn hóa góc về khoảng [-pi, pi).
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _move_towards(current, target, max_change):
        """
        Giới hạn mức thay đổi từ current đến target.

        max_change là mức thay đổi tối đa trong một bước.
        """
        difference = target - current

        if abs(difference) <= max_change:
            return target

        return current + math.copysign(max_change, difference)

    @staticmethod
    def _validate_finite(value, name):
        """
        Kiểm tra giá trị đầu vào có phải số hữu hạn không.
        """
        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

        return value

    def reset(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        v=0.0,
        w=0.0
    ):
        """
        Đưa robot về trạng thái ban đầu.

        Vận tốc ban đầu được giới hạn theo config.py.
        """

        self.x = self._validate_finite(x, "x")
        self.y = self._validate_finite(y, "y")
        theta = self._validate_finite(theta, "theta")

        v = self._validate_finite(v, "v")
        w = self._validate_finite(w, "w")

        self.theta = self._wrap_angle(theta)

        self.v = float(np.clip(
            v,
            self.v_min,
            self.v_max
        ))

        self.w = float(np.clip(
            w,
            -self.w_max,
            self.w_max
        ))

        return self.get_state()

    def get_state(self):
        """
        Trả về trạng thái robot theo đúng contract của nhóm.

        Output:
            numpy ndarray (5,)
            [x, y, theta, v, w]
        """

        return np.array(
            [
                self.x,
                self.y,
                self.theta,
                self.v,
                self.w
            ],
            dtype=np.float64
        )

    def step(self, v_cmd, w_cmd, dt=None):
        """
        Cập nhật trạng thái robot sau một bước mô phỏng.

        Input:
            v_cmd : vận tốc dài yêu cầu (m/s)
            w_cmd : vận tốc góc yêu cầu (rad/s)
            dt    : bước thời gian (s)

        Quy trình:
            1. Giới hạn vận tốc yêu cầu.
            2. Áp dụng giới hạn gia tốc.
            3. Cập nhật vị trí và góc hướng.
            4. Trả về trạng thái mới.
        """

        if dt is None:
            dt = config.DT

        dt = self._validate_finite(dt, "dt")
        v_cmd = self._validate_finite(v_cmd, "v_cmd")
        w_cmd = self._validate_finite(w_cmd, "w_cmd")

        if dt <= 0.0:
            raise ValueError("dt must be greater than zero.")

        # --------------------------------------------------
        # 1. Giới hạn vận tốc yêu cầu
        # --------------------------------------------------

        v_target = float(np.clip(
            v_cmd,
            self.v_min,
            self.v_max
        ))

        w_target = float(np.clip(
            w_cmd,
            -self.w_max,
            self.w_max
        ))

        # --------------------------------------------------
        # 2. Giới hạn gia tốc
        # --------------------------------------------------

        max_dv = self.max_accel * dt
        max_dw = self.max_w_accel * dt

        self.v = self._move_towards(
            self.v,
            v_target,
            max_dv
        )

        self.w = self._move_towards(
            self.w,
            w_target,
            max_dw
        )

        # --------------------------------------------------
        # 3. Cập nhật vị trí bằng mô hình động học
        # --------------------------------------------------

        theta_old = self.theta
        theta_new = theta_old + self.w * dt

        # Tích phân chính xác cho vận tốc v, w không đổi
        # trong một bước dt.

        if abs(self.w) < 1e-8:

            # Trường hợp chuyển động gần như đường thẳng
            self.x += self.v * math.cos(theta_old) * dt
            self.y += self.v * math.sin(theta_old) * dt

        else:

            # Trường hợp chuyển động theo cung tròn
            self.x += (
                self.v / self.w
                * (math.sin(theta_new) - math.sin(theta_old))
            )

            self.y -= (
                self.v / self.w
                * (math.cos(theta_new) - math.cos(theta_old))
            )

        self.theta = self._wrap_angle(theta_new)

        return self.get_state()

    def wheel_speeds(self, wheel_radius, wheel_base):
        """
        Tính vận tốc góc của hai bánh xe.

        Input:
            wheel_radius : bán kính bánh xe (m)
            wheel_base   : khoảng cách hai bánh xe (m)

        Output:
            (omega_left, omega_right) [rad/s]

        Chưa sử dụng trực tiếp trong vòng lặp điều khiển
        vì config.py hiện chưa quy định hình học bánh xe.
        """

        wheel_radius = self._validate_finite(
            wheel_radius,
            "wheel_radius"
        )

        wheel_base = self._validate_finite(
            wheel_base,
            "wheel_base"
        )

        if wheel_radius <= 0.0 or wheel_base <= 0.0:
            raise ValueError(
                "wheel_radius and wheel_base must be positive."
            )

        omega_left = (
            self.v - self.w * wheel_base / 2.0
        ) / wheel_radius

        omega_right = (
            self.v + self.w * wheel_base / 2.0
        ) / wheel_radius

        return omega_left, omega_right

    def __repr__(self):
        return (
            f"Robot("
            f"x={self.x:.3f}, "
            f"y={self.y:.3f}, "
            f"theta={self.theta:.3f}, "
            f"v={self.v:.3f}, "
            f"w={self.w:.3f})"
        )
