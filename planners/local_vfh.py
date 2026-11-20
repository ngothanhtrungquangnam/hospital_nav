
import math
import numpy as np

from config import (
    ROBOT_RADIUS,
    V_MAX,
    V_MIN,
    W_MAX,
    MAX_ACCEL,
    MAX_W_ACCEL,
)

from planners.base import LocalPlanner


class VFHPlanner(LocalPlanner):
    name = "VFH"

    def __init__(
        self,
        grid_resolution=0.1,
        max_range=3.0,
        num_sectors=72,
        histogram_threshold=0.12,
        safety_margin=0.25,
        clearance_distance=1.0,
        goal_tolerance=0.25,
        predict_time=2.0,
        k_omega=2.4,
        goal_weight=0.85,
        continuity_weight=0.15,
    ):
        # Cartesian histogram grid
        self.grid_resolution = float(grid_resolution)
        self.max_range = float(max_range)

        # Polar histogram
        self.num_sectors = int(num_sectors)
        self.histogram_threshold = float(histogram_threshold)

        # Robot geometry and safety
        self.robot_radius = ROBOT_RADIUS
        self.safety_margin = float(safety_margin)
        self.influence_radius = self.robot_radius + self.safety_margin

        # Robot velocity limits
        self.v_max = V_MAX
        self.v_min = V_MIN
        self.w_max = W_MAX

        self.max_accel = MAX_ACCEL
        self.max_w_accel = MAX_W_ACCEL

        # Controller parameters
        self.clearance_distance = float(clearance_distance)
        self.goal_tolerance = float(goal_tolerance)
        self.predict_time = float(predict_time)
        self.k_omega = float(k_omega)

        self.goal_weight = float(goal_weight)
        self.continuity_weight = float(continuity_weight)

        self.sector_width = 2.0 * math.pi / self.num_sectors

        # Last selected heading, expressed in robot coordinates
        self.last_direction = None

        self.last_debug = {}

        # Pre-compute the obstacle inflation kernel
        self.inflation_offsets = self._create_inflation_kernel()

    def reset(self):
        self.last_direction = None
        self.last_debug = {}

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _create_inflation_kernel(self):
        """
        Generate grid offsets used to inflate LiDAR obstacle cells.

        Inflation radius = robot radius + safety margin.
        """
        radius_cells = int(
            math.ceil(self.influence_radius / self.grid_resolution)
        )

        offsets = []

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                distance = math.hypot(dx, dy) * self.grid_resolution

                if distance <= self.influence_radius + 1e-9:
                    offsets.append((dy, dx, distance))

        return offsets

    def _world_to_robot(self, state, points):
        """
        Transform world-coordinate LiDAR points into robot coordinates.

        Robot frame:
            X: forward
            Y: left
        """
        x, y, theta = state[:3]

        if points is None:
            return np.empty((0, 2), dtype=float)

        points = np.asarray(points, dtype=float)

        if points.size == 0:
            return np.empty((0, 2), dtype=float)

        points = points.reshape(-1, 2)

        # Remove invalid values
        points = points[np.all(np.isfinite(points), axis=1)]

        if len(points) == 0:
            return np.empty((0, 2), dtype=float)

        dx = points[:, 0] - x
        dy = points[:, 1] - y

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy

        return np.column_stack((local_x, local_y))

    def _build_cartesian_grid(self, local_points):
        """
        Build a local Cartesian histogram grid.

        Each LiDAR return marks an obstacle cell.
        Nearby cells are inflated to account for robot size
        and the safety margin.
        """
        grid_size = int(
            math.ceil(2.0 * self.max_range / self.grid_resolution)
        )

        grid = np.zeros((grid_size, grid_size), dtype=float)

        if len(local_points) == 0:
            return grid

        px = local_points[:, 0]
        py = local_points[:, 1]

        ranges = np.hypot(px, py)

        valid = (
            (ranges > 0.0)
            & (ranges <= self.max_range)
            & (px >= -self.max_range)
            & (px < self.max_range)
            & (py >= -self.max_range)
            & (py < self.max_range)
        )

        px = px[valid]
        py = py[valid]
        ranges = ranges[valid]

        if len(px) == 0:
            return grid

        # Convert local coordinates to grid indices
        ix = np.floor(
            (px + self.max_range) / self.grid_resolution
        ).astype(int)

        iy = np.floor(
            (py + self.max_range) / self.grid_resolution
        ).astype(int)

        # A nearby obstacle has a greater magnitude
        strengths = 1.0 / (0.2 + ranges)

        # Keep the strongest observation in each cell
        np.maximum.at(grid, (iy, ix), strengths)

        # Save the original occupied cells before inflation
        source_y, source_x = np.nonzero(grid > 0.0)
        source_values = grid[source_y, source_x].copy()

        # Inflate the obstacle cells
        for dy, dx, distance in self.inflation_offsets:

            target_y = source_y + dy
            target_x = source_x + dx

            inside = (
                (target_y >= 0)
                & (target_y < grid_size)
                & (target_x >= 0)
                & (target_x < grid_size)
            )

            target_y = target_y[inside]
            target_x = target_x[inside]

            values = source_values[inside]

            # Reduce influence with distance from the obstacle cell
            falloff = max(
                0.0,
                1.0 - distance / (
                    self.influence_radius + self.grid_resolution
                )
            )

            values = values * falloff

            np.maximum.at(
                grid,
                (target_y, target_x),
                values,
            )

        return grid

    def _build_polar_histogram(self, grid):
        """
        Project the Cartesian histogram grid onto angular sectors.

        Returns:
            normalized_histogram: obstacle density per sector
        """
        histogram = np.zeros(self.num_sectors, dtype=float)

        rows, cols = np.nonzero(grid > 0.0)

        if len(rows) == 0:
            return histogram

        # Get the center coordinates of occupied cells
        local_x = (
            -self.max_range
            + (cols + 0.5) * self.grid_resolution
        )

        local_y = (
            -self.max_range
            + (rows + 0.5) * self.grid_resolution
        )

        ranges = np.hypot(local_x, local_y)

        valid = (
            (ranges > 0.0)
            & (ranges <= self.max_range)
        )

        local_x = local_x[valid]
        local_y = local_y[valid]
        ranges = ranges[valid]

        cell_values = grid[rows[valid], cols[valid]]

        angles = np.arctan2(local_y, local_x)

        # Convert angle from [-pi, pi] into a sector index
        sector_indices = np.floor(
            (angles + math.pi)
            / (2.0 * math.pi)
            * self.num_sectors
        ).astype(int)

        sector_indices %= self.num_sectors

        # Nearby cells contribute more to obstacle density
        contributions = cell_values / (0.2 + ranges)

        np.add.at(
            histogram,
            sector_indices,
            contributions,
        )

        # Normalize histogram
        maximum = float(np.max(histogram))

        if maximum > 1e-12:
            histogram /= maximum

        # Circular smoothing to reduce sudden sector changes
        histogram = (
            0.25 * np.roll(histogram, 1)
            + 0.50 * histogram
            + 0.25 * np.roll(histogram, -1)
        )

        maximum = float(np.max(histogram))

        if maximum > 1e-12:
            histogram /= maximum

        return histogram

    def _select_direction(self, histogram, goal_angle):
        """
        Select a free direction closest to the target direction.

        If every sector is blocked, return the least occupied
        sector and indicate that the robot must stop.
        """
        sector_angles = (
            -math.pi
            + (np.arange(self.num_sectors) + 0.5)
            * self.sector_width
        )

        blocked = histogram >= self.histogram_threshold
        open_indices = np.where(~blocked)[0]

        if len(open_indices) == 0:
            # No open sector: choose the least occupied sector
            best_index = int(np.argmin(histogram))

            return (
                float(sector_angles[best_index]),
                False,
            )

        # Evaluate all available open directions
        candidates = sector_angles[open_indices]

        goal_errors = np.abs(
            np.array([
                self._wrap_angle(angle - goal_angle)
                for angle in candidates
            ])
        )

        if self.last_direction is None:
            continuity_errors = np.zeros_like(goal_errors)
        else:
            continuity_errors = np.abs(
                np.array([
                    self._wrap_angle(angle - self.last_direction)
                    for angle in candidates
                ])
            )

        # Lower cost means a better candidate direction
        costs = (
            self.goal_weight * goal_errors
            + self.continuity_weight * continuity_errors
        )

        best_local_index = int(np.argmin(costs))
        best_index = int(open_indices[best_local_index])

        return (
            float(sector_angles[best_index]),
            True,
        )

    def _get_direction_clearance(
        self,
        local_points,
        direction,
    ):
        """
        Find the nearest LiDAR obstacle in the selected direction.
        """
        if len(local_points) == 0:
            return self.max_range

        ranges = np.hypot(
            local_points[:, 0],
            local_points[:, 1],
        )

        angles = np.arctan2(
            local_points[:, 1],
            local_points[:, 0],
        )

        angle_errors = np.abs(
            np.array([
                self._wrap_angle(angle - direction)
                for angle in angles
            ])
        )

        # Consider obstacles in a small angular neighborhood
        angular_limit = max(
            1.5 * self.sector_width,
            math.radians(8.0),
        )

        relevant = angle_errors <= angular_limit

        if not np.any(relevant):
            return self.max_range

        return float(np.min(ranges[relevant]))

    def _limit_acceleration(self, state, v_target, w_target, dt):
        """
        Apply linear and angular acceleration limits.
        """
        current_v = float(state[3])
        current_w = float(state[4])

        v_low = max(
            self.v_min,
            current_v - self.max_accel * dt,
        )

        v_high = min(
            self.v_max,
            current_v + self.max_accel * dt,
        )

        w_low = max(
            -self.w_max,
            current_w - self.max_w_accel * dt,
        )

        w_high = min(
            self.w_max,
            current_w + self.max_w_accel * dt,
        )

        v_cmd = float(np.clip(v_target, v_low, v_high))
        w_cmd = float(np.clip(w_target, w_low, w_high))

        return v_cmd, w_cmd

    def _predict_trajectory(self, state, v_cmd, w_cmd, dt):
        """
        Predict robot trajectory under constant v_cmd and w_cmd.
        """
        x = float(state[0])
        y = float(state[1])
        theta = float(state[2])

        trajectory = [[x, y]]

        elapsed = 0.0

        while elapsed < self.predict_time - 1e-9:
            step_dt = min(dt, self.predict_time - elapsed)

            next_theta = theta + w_cmd * step_dt

            if abs(w_cmd) < 1e-8:
                x += v_cmd * math.cos(theta) * step_dt
                y += v_cmd * math.sin(theta) * step_dt
            else:
                x += (
                    v_cmd / w_cmd
                    * (math.sin(next_theta) - math.sin(theta))
                )

                y -= (
                    v_cmd / w_cmd
                    * (math.cos(next_theta) - math.cos(theta))
                )

            theta = next_theta
            elapsed += step_dt

            trajectory.append([x, y])

        return np.asarray(trajectory, dtype=float)

    def compute_control(self, state, sub_goal, points, dt):
        """
        LocalPlanner interface.

        state: [x, y, theta, v, w]
        sub_goal: [gx, gy]
        points: ndarray (N, 2), world-coordinate LiDAR points
        dt: control time step

        Returns:
            v_cmd, w_cmd, pred_traj
        """
        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        state = np.asarray(state, dtype=float)
        sub_goal = np.asarray(sub_goal, dtype=float).reshape(2)

        if state.size != 5:
            raise ValueError("state must contain [x, y, theta, v, w]")

        # --------------------------------------------------
        # 1. Transform LiDAR points into the robot frame
        # --------------------------------------------------
        local_points = self._world_to_robot(state, points)

        # --------------------------------------------------
        # 2. Build Cartesian and polar histograms
        # --------------------------------------------------
        cartesian_grid = self._build_cartesian_grid(local_points)

        polar_histogram = self._build_polar_histogram(
            cartesian_grid
        )

        # --------------------------------------------------
        # 3. Calculate target direction in the robot frame
        # --------------------------------------------------
        dx = float(sub_goal[0] - state[0])
        dy = float(sub_goal[1] - state[1])

        goal_distance = math.hypot(dx, dy)

        cos_theta = math.cos(float(state[2]))
        sin_theta = math.sin(float(state[2]))

        goal_local_x = cos_theta * dx + sin_theta * dy
        goal_local_y = -sin_theta * dx + cos_theta * dy

        goal_angle = math.atan2(
            goal_local_y,
            goal_local_x,
        )

        # --------------------------------------------------
        # 4. Stop when the local goal has been reached
        # --------------------------------------------------
        if goal_distance <= self.goal_tolerance:
            v_cmd, w_cmd = self._limit_acceleration(
                state,
                0.0,
                0.0,
                dt,
            )

            pred_traj = self._predict_trajectory(
                state,
                v_cmd,
                w_cmd,
                dt,
            )

            self.last_debug = {
                "status": "goal_reached",
                "goal_distance": goal_distance,
                "polar_histogram": polar_histogram,
            }

            return v_cmd, w_cmd, pred_traj

        # --------------------------------------------------
        # 5. Select an available direction
        # --------------------------------------------------
        selected_direction, has_open_sector = (
            self._select_direction(
                polar_histogram,
                goal_angle,
            )
        )

        self.last_direction = selected_direction

        # --------------------------------------------------
        # 6. Convert selected direction into velocity commands
        # --------------------------------------------------
        clearance = self._get_direction_clearance(
            local_points,
            selected_direction,
        )

        free_clearance = max(
            0.0,
            clearance - self.influence_radius,
        )

        clearance_factor = float(
            np.clip(
                free_clearance / self.clearance_distance,
                0.0,
                1.0,
            )
        )

        # Prefer forward movement.
        # If the selected direction is behind the robot,
        # rotate first instead of driving backward.
        alignment = max(
            0.0,
            math.cos(selected_direction),
        )

        if has_open_sector:
            v_target = (
                self.v_max
                * alignment
                * clearance_factor
            )
        else:
            # All sectors are blocked: stop and rotate
            v_target = 0.0

        w_target = float(
            np.clip(
                self.k_omega * selected_direction,
                -self.w_max,
                self.w_max,
            )
        )

        # --------------------------------------------------
        # 7. Respect robot acceleration constraints
        # --------------------------------------------------
        v_cmd, w_cmd = self._limit_acceleration(
            state,
            v_target,
            w_target,
            dt,
        )

        # --------------------------------------------------
        # 8. Predict the resulting trajectory
        # --------------------------------------------------
        pred_traj = self._predict_trajectory(
            state,
            v_cmd,
            w_cmd,
            dt,
        )

        self.last_debug = {
            "status": "normal" if has_open_sector else "no_open_sector",
            "goal_distance": goal_distance,
            "goal_angle": goal_angle,
            "selected_direction": selected_direction,
            "clearance": clearance,
            "clearance_factor": clearance_factor,
            "v_target": v_target,
            "w_target": w_target,
            "polar_histogram": polar_histogram,
            "cartesian_grid": cartesian_grid,
        }

        return v_cmd, w_cmd, pred_traj
