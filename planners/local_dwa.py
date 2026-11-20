
"""
planners/local_dwa.py

Dynamic Window Approach (DWA) for Differential Drive Robot.

Interface contract:
    compute_control(self, state, sub_goal, points, dt)

Input:
    state:
        [x, y, theta, v, w]
        Shape: (5,)

    sub_goal:
        [gx, gy]
        Shape: (2,)

    points:
        LiDAR obstacle points in world coordinates.
        Shape: (N, 2)

    dt:
        Simulation time step (s).

Output:
    v_cmd:
        Desired linear velocity (m/s).

    w_cmd:
        Desired angular velocity (rad/s).

    pred_traj:
        Predicted trajectory.
        Shape: (M, 2)
"""

import math
import numpy as np

import config
from planners.base import LocalPlanner


class DWAPlanner(LocalPlanner):
    """
    Dynamic Window Approach for a differential drive robot.

    Objective function:

        G = alpha * heading
          + beta  * clearance
          + gamma * velocity

    The three objective components are normalized to [0, 1].
    """

    name = "DWA"

    def __init__(
        self,
        predict_time=1.5,
        v_resolution=0.05,
        w_resolution=0.1,
        alpha=0.2,
        beta=0.3,
        gamma=0.5,
        safety_margin=0.1,
        clearance_distance=1.0,
        goal_tolerance=0.25
    ):
        # -------------------------------------------------
        # 1. Robot physical parameters from shared config
        # -------------------------------------------------

        self.robot_radius = config.ROBOT_RADIUS

        self.v_max = config.V_MAX
        self.v_min = config.V_MIN
        self.w_max = config.W_MAX

        self.max_accel = config.MAX_ACCEL
        self.max_w_accel = config.MAX_W_ACCEL

        # -------------------------------------------------
        # 2. DWA parameters
        # -------------------------------------------------

        self.predict_time = self._validate_positive(
            predict_time, "predict_time"
        )

        self.v_resolution = self._validate_positive(
            v_resolution, "v_resolution"
        )

        self.w_resolution = self._validate_positive(
            w_resolution, "w_resolution"
        )

        self.safety_margin = self._validate_non_negative(
            safety_margin, "safety_margin"
        )

        self.clearance_distance = self._validate_positive(
            clearance_distance, "clearance_distance"
        )

        self.goal_tolerance = self._validate_positive(
            goal_tolerance, "goal_tolerance"
        )

        # Normalize objective weights
        weights = np.array(
            [alpha, beta, gamma],
            dtype=np.float64
        )

        if not np.all(np.isfinite(weights)):
            raise ValueError("DWA weights must be finite.")

        if np.any(weights < 0.0) or np.sum(weights) <= 0.0:
            raise ValueError(
                "DWA weights must be non-negative "
                "and at least one weight must be positive."
            )

        weights /= np.sum(weights)

        self.alpha = float(weights[0])
        self.beta = float(weights[1])
        self.gamma = float(weights[2])

        # Debug information for testing and tuning
        self.last_debug = {}

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _validate_positive(value, name):
        value = float(value)

        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")

        return value

    @staticmethod
    def _validate_non_negative(value, name):
        value = float(value)

        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be a non-negative finite number."
            )

        return value

    @staticmethod
    def _wrap_angle(angle):
        """Normalize angle to [-pi, pi)."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """Reset planner information before a simulation run."""
        self.last_debug = {}

    # =====================================================
    # 1. DYNAMIC WINDOW
    # =====================================================

    def calc_dynamic_window(self, state, dt):
        """
        Calculate the feasible velocity window.

        Returns:
            (v_low, v_high, w_low, w_high)
        """

        state = np.asarray(state, dtype=np.float64)

        if state.shape != (5,) or not np.all(np.isfinite(state)):
            raise ValueError(
                "state must be a finite array with shape (5,)."
            )

        dt = self._validate_positive(dt, "dt")

        # Current velocities
        v_current = float(np.clip(
            state[3], self.v_min, self.v_max
        ))

        w_current = float(np.clip(
            state[4], -self.w_max, self.w_max
        ))

        # Physical velocity limits (Vs)
        vs = (
            self.v_min,
            self.v_max,
            -self.w_max,
            self.w_max
        )

        # Acceleration limits (Va)
        va = (
            v_current - self.max_accel * dt,
            v_current + self.max_accel * dt,

            w_current - self.max_w_accel * dt,
            w_current + self.max_w_accel * dt
        )

        # Intersection: Vd = Vs ∩ Va
        v_low = max(vs[0], va[0])
        v_high = min(vs[1], va[1])

        w_low = max(vs[2], va[2])
        w_high = min(vs[3], va[3])

        return v_low, v_high, w_low, w_high

    # =====================================================
    # 2. VELOCITY SAMPLING
    # =====================================================

    @staticmethod
    def _sample_range(low, high, resolution, current=None):
        """
        Sample candidate values inside [low, high].

        Always include both boundaries and zero when feasible.
        """

        if high < low:
            return np.array([], dtype=np.float64)

        values = [
            low + i * resolution
            for i in range(
                int(math.floor((high - low) / resolution)) + 1
            )
        ]

        values.append(high)

        if low <= 0.0 <= high:
            values.append(0.0)

        if current is not None:
            values.append(float(np.clip(current, low, high)))

        return np.unique(
            np.round(np.asarray(values), decimals=10)
        )

    # =====================================================
    # 3. ROBOT TRAJECTORY PREDICTION
    # =====================================================

    @staticmethod
    def _integrate_pose(x, y, theta, v, w, dt):
        """
        Integrate robot pose for constant v and w.

        Uses exact circular-arc integration.
        """

        theta_new = theta + w * dt

        if abs(w) < 1e-8:

            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt

        else:

            x_new = x + (
                v / w
                * (math.sin(theta_new) - math.sin(theta))
            )

            y_new = y - (
                v / w
                * (math.cos(theta_new) - math.cos(theta))
            )

        theta_new = DWAPlanner._wrap_angle(theta_new)

        return x_new, y_new, theta_new

    def predict_trajectory(self, state, v, w, dt):
        """
        Predict the robot trajectory using a candidate (v, w).

        Assumption:
            The candidate velocity is reachable within the current
            control step due to the Dynamic Window constraint.

            After that, the candidate velocity is kept constant
            throughout the prediction horizon.

        Returns:
            NumPy array with shape (M, 2).
        """

        x, y, theta = map(float, state[:3])

        trajectory = [[x, y]]

        elapsed = 0.0

        while elapsed < self.predict_time - 1e-10:

            step_dt = min(
                dt,
                self.predict_time - elapsed
            )

            x, y, theta = self._integrate_pose(
                x, y, theta,
                v, w,
                step_dt
            )

            trajectory.append([x, y])

            elapsed += step_dt

        return np.asarray(
            trajectory,
            dtype=np.float64
        )

   
    # =====================================================
    # 4. HEADING SCORE
    # =====================================================

    def calc_heading_score(self, state, trajectory, w, sub_goal):
        """
        Evaluate how well the final robot orientation
        points toward the local goal.

        Output is normalized to [0, 1].
        """

        final_x, final_y = trajectory[-1]

        # Final robot orientation
        final_theta = self._wrap_angle(
            float(state[2]) + w * self.predict_time
        )

        # Direction from final position to local goal
        dx = sub_goal[0] - final_x
        dy = sub_goal[1] - final_y

        if math.hypot(dx, dy) < 1e-8:
            return 1.0

        goal_angle = math.atan2(dy, dx)

        heading_error = self._wrap_angle(
            goal_angle - final_theta
        )

        return float(
            (1.0 + math.cos(heading_error)) / 2.0
        )


    # =====================================================
    # 5. CLEARANCE SCORE AND COLLISION CHECK
    # =====================================================

    def calc_clearance_score(self, trajectory, points):
        """
        Calculate minimum clearance between the robot
        and LiDAR obstacle points.

        points are in world coordinates.

        Returns:
            safe          : bool
            clearance     : minimum distance from robot
                            boundary to obstacle points
            clearance_score : normalized score [0, 1]
        """

        # Empty obstacle cloud
        if points.shape[0] == 0:

            return True, float("inf"), 1.0

        min_distance = float("inf")

        # Check every predicted robot position
        for position in trajectory:

            distances = np.linalg.norm(
                points - position,
                axis=1
            )

            local_min = float(np.min(distances))

            min_distance = min(
                min_distance,
                local_min
            )

        # Collision threshold includes robot radius
        # and an additional safety margin.
        collision_distance = (
            self.robot_radius + self.safety_margin
        )

        # No valid trajectory if collision is predicted
        if min_distance <= collision_distance:

            clearance = min_distance - self.robot_radius

            return False, clearance, 0.0

        # Distance from robot boundary to nearest obstacle
        clearance = (
            min_distance - self.robot_radius
        )

        clearance_score = np.clip(
            clearance / self.clearance_distance,
            0.0,
            1.0
        )

        return (
            True,
            float(clearance),
            float(clearance_score)
        )

    # =====================================================
    # 6. VELOCITY SCORE
    # =====================================================

    def calc_velocity_score(self, v):
        """
        Ưu tiên vận tốc tiến dương.
        v <= 0 không nhận điểm thưởng để tránh robot đứng im ăn điểm.
        """
        if v <= 0.0:
            return 0.0
        return float(min(v / self.v_max, 1.0))

    # =====================================================
    # 7. EMERGENCY BRAKING
    # =====================================================

    @staticmethod
    def _brake_to_zero(current, max_change):
        """
        Calculate a reachable velocity command that moves
        the current velocity toward zero.
        """

        if current > 0.0:
            return max(0.0, current - max_change)

        if current < 0.0:
            return min(0.0, current + max_change)

        return 0.0

    # =====================================================
    # 8. MAIN CONTROL FUNCTION
    # =====================================================

    def compute_control(self, state, sub_goal, points, dt):
        """
        Main DWA control function.

        This signature MUST remain unchanged to comply
        with planners/base.py.
        """

        # -------------------------------------------------
        # Validate input
        # -------------------------------------------------

        state = np.asarray(
            state,
            dtype=np.float64
        )

        sub_goal = np.asarray(
            sub_goal,
            dtype=np.float64
        )

        if state.shape != (5,):
            raise ValueError(
                "state must have shape (5,): [x,y,theta,v,w]."
            )

        if sub_goal.shape != (2,):
            raise ValueError(
                "sub_goal must have shape (2,): [gx,gy]."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError("state contains non-finite values.")

        if not np.all(np.isfinite(sub_goal)):
            raise ValueError("sub_goal contains non-finite values.")

        if points is None:
            points = np.empty((0, 2), dtype=np.float64)
        else:
            points = np.asarray(
                points,
                dtype=np.float64
            )

        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "points must have shape (N, 2)."
            )

        if not np.all(np.isfinite(points)):
            raise ValueError("points contains non-finite values.")

        dt = self._validate_positive(dt, "dt")

        x, y, theta, v_current, w_current = state

        # -------------------------------------------------
        # Check whether local goal has been reached
        # -------------------------------------------------

        goal_distance = math.hypot(
            sub_goal[0] - x,
            sub_goal[1] - y
        )

        if goal_distance <= self.goal_tolerance:

            # Brake instead of stopping instantaneously
            v_cmd = self._brake_to_zero(
                v_current,
                self.max_accel * dt
            )

            w_cmd = self._brake_to_zero(
                w_current,
                self.max_w_accel * dt
            )

            pred_traj = self.predict_trajectory(
                state,
                v_cmd,
                w_cmd,
                dt
            )

            self.last_debug = {
                "status": "goal_reached",
                "candidate_count": 0
            }

            return float(v_cmd), float(w_cmd), pred_traj

        # -------------------------------------------------
        # Step 1: Calculate Dynamic Window
        # -------------------------------------------------

        v_low, v_high, w_low, w_high = (
            self.calc_dynamic_window(state, dt)
        )

        # -------------------------------------------------
        # Step 2: Sample candidate velocities
        # -------------------------------------------------

        v_samples = self._sample_range(
            v_low,
            v_high,
            self.v_resolution,
            current=v_current
        )

        w_samples = self._sample_range(
            w_low,
            w_high,
            self.w_resolution,
            current=w_current
        )

        # -------------------------------------------------
        # Step 3: Evaluate candidate trajectories
        # -------------------------------------------------

        best_score = -float("inf")

        best_v = None
        best_w = None
        best_traj = None

        best_components = None

        candidate_count = 0
        safe_count = 0

        for v in v_samples:

            for w in w_samples:

                candidate_count += 1

                # 3.1 Predict trajectory
                trajectory = self.predict_trajectory(
                    state,
                    float(v),
                    float(w),
                    dt
                )

                # 3.2 Check obstacle clearance
                safe, clearance, clearance_score = (
                    self.calc_clearance_score(
                        trajectory,
                        points
                    )
                )

                # Reject colliding trajectories
                if not safe:
                    continue

                safe_count += 1

                heading_score = self.calc_heading_score(
                    state,
                    trajectory,
                    float(w),
                    sub_goal
                )

                # 3.4 Calculate velocity score
                velocity_score = (
                    self.calc_velocity_score(float(v))
                )

                # 3.5 Overall objective function
                total_score = (
                    self.alpha * heading_score
                    + self.beta * clearance_score
                    + self.gamma * velocity_score
                )

                # 3.6 Keep the best candidate
                if total_score > best_score:

                    best_score = total_score

                    best_v = float(v)
                    best_w = float(w)

                    best_traj = trajectory

                    best_components = {
                        "heading": heading_score,
                        "clearance": clearance_score,
                        "velocity": velocity_score,
                        "min_clearance_m": clearance
                    }

        # -------------------------------------------------
        # Step 4: No safe trajectory -> brake
        # -------------------------------------------------

        if best_v is None:

            v_cmd = self._brake_to_zero(
                v_current,
                self.max_accel * dt
            )

            w_cmd = self._brake_to_zero(
                w_current,
                self.max_w_accel * dt
            )

            pred_traj = self.predict_trajectory(
                state,
                v_cmd,
                w_cmd,
                dt
            )

            self.last_debug = {
                "status": "emergency_braking",
                "candidate_count": candidate_count,
                "safe_count": safe_count
            }

            return float(v_cmd), float(w_cmd), pred_traj

        # -------------------------------------------------
        # Step 5: Return the best control command
        # -------------------------------------------------

        self.last_debug = {
            "status": "normal",
            "candidate_count": candidate_count,
            "safe_count": safe_count,
            "best_score": best_score,
            "best_components": best_components
        }

        return best_v, best_w, best_traj
