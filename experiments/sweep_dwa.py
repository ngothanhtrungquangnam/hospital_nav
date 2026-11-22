"""Parameter sweep for the Dynamic Window Approach (DWA).

Run from the project root:
    python experiments/sweep-dwa.py

The script reuses the scenarios in tests/toy_world.py but runs them without
Matplotlib so that hundreds of trials can be evaluated automatically.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# Make the project root and tests directory importable regardless of the
# directory from which this script is launched.
ROOT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT_DIR / "tests"

for path in (ROOT_DIR, TESTS_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from planners.local_dwa import DWAPlanner  # noqa: E402
from toy_world import (  # noqa: E402
    GOAL_TOLERANCE,
    LIDAR_RANGE,
    ROBOT_RADIUS,
    build_scenario,
)


SCENARIOS = ("single", "gap", "u_trap")

RAW_FIELDS = (
    "config_id",
    "alpha",
    "beta",
    "gamma",
    "scenario",
    "status",
    "success",
    "collision",
    "steps",
    "simulation_time_s",
    "completion_time_s",
    "path_length_m",
    "final_goal_distance_m",
    "minimum_clearance_m",
    "mean_abs_linear_acceleration_mps2",
    "mean_abs_angular_acceleration_radps2",
)

SUMMARY_FIELDS = (
    "rank",
    "config_id",
    "alpha",
    "beta",
    "gamma",
    "scenario_count",
    "success_count",
    "success_rate_pct",
    "collision_count",
    "collision_rate_pct",
    "mean_completion_time_success_s",
    "mean_path_length_success_m",
    "mean_path_length_all_m",
    "mean_minimum_clearance_m",
    "mean_abs_linear_acceleration_mps2",
    "mean_abs_angular_acceleration_radps2",
)


def build_weight_configs(weight_step: float) -> list[dict[str, float | str]]:
    """Create all non-negative weight triples summing to one.

    The current DWA defaults from planners/local_dwa.py are also included
    explicitly, even when they do not lie on the selected grid.
    """
    if not math.isfinite(weight_step) or weight_step <= 0.0:
        raise ValueError("weight_step must be a positive finite number")

    units_float = 1.0 / weight_step
    units = int(round(units_float))
    if not math.isclose(units * weight_step, 1.0, abs_tol=1e-9):
        raise ValueError("weight_step must divide 1.0 evenly, e.g. 0.1 or 0.2")

    configs: list[dict[str, float | str]] = []
    seen: set[tuple[float, float, float]] = set()

    for alpha_units in range(units + 1):
        for beta_units in range(units - alpha_units + 1):
            gamma_units = units - alpha_units - beta_units
            alpha = alpha_units / units
            beta = beta_units / units
            gamma = gamma_units / units
            weights = (round(alpha, 8), round(beta, 8), round(gamma, 8))

            config_id = (
                f"grid_a{weights[0]:.2f}_"
                f"b{weights[1]:.2f}_g{weights[2]:.2f}"
            )
            configs.append(
                {
                    "config_id": config_id,
                    "alpha": weights[0],
                    "beta": weights[1],
                    "gamma": weights[2],
                }
            )
            seen.add(weights)

    # These are the defaults in the uploaded/current local_dwa.py.
    baseline = (0.30, 0.15, 0.55)
    if baseline not in seen:
        configs.append(
            {
                "config_id": "current_default",
                "alpha": baseline[0],
                "beta": baseline[1],
                "gamma": baseline[2],
            }
        )

    return configs


def _nearest_obstacle_distance(state: np.ndarray, obstacle_points: np.ndarray) -> float:
    if obstacle_points.size == 0:
        return math.inf
    distances = np.linalg.norm(obstacle_points - state[:2], axis=1)
    return float(np.min(distances))


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def run_trial(
    scenario: str,
    weights: dict[str, float | str],
    dt: float = 0.1,
    max_steps: int = 250,
) -> dict[str, float | int | str]:
    """Run one non-visual trial using the same update/collision rules as toy_world."""
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")

    planner = DWAPlanner(
        alpha=float(weights["alpha"]),
        beta=float(weights["beta"]),
        gamma=float(weights["gamma"]),
    )

    state, goal, all_points = build_scenario(scenario)
    state = np.asarray(state, dtype=float).copy()
    goal = np.asarray(goal, dtype=float)
    all_points = np.asarray(all_points, dtype=float).reshape(-1, 2)

    history = [state[:2].copy()]
    # Include the initial velocity so the first control transition is counted.
    command_history = [state[3:5].copy()]

    minimum_clearance = math.inf
    steps = 0
    status = "timeout"

    for step in range(max_steps):
        obstacle_distances = np.linalg.norm(all_points - state[:2], axis=1)
        nearest_distance = (
            float(np.min(obstacle_distances))
            if len(obstacle_distances)
            else math.inf
        )
        minimum_clearance = min(
            minimum_clearance,
            nearest_distance - ROBOT_RADIUS,
        )

        # Match the discrete collision threshold used by toy_world.py.
        if nearest_distance <= ROBOT_RADIUS:
            status = "collision"
            break

        goal_distance = float(np.linalg.norm(goal - state[:2]))
        if goal_distance <= GOAL_TOLERANCE:
            status = "success"
            break

        visible_points = all_points[obstacle_distances <= LIDAR_RANGE]
        v_cmd, w_cmd, _predicted_trajectory = planner.compute_control(
            state,
            goal,
            visible_points,
            dt,
        )

        x, y, theta, _, _ = state
        x += float(v_cmd) * math.cos(float(theta)) * dt
        y += float(v_cmd) * math.sin(float(theta)) * dt
        theta = (float(theta) + float(w_cmd) * dt + math.pi) % (2.0 * math.pi) - math.pi

        state = np.array([x, y, theta, v_cmd, w_cmd], dtype=float)
        history.append(state[:2].copy())
        command_history.append(state[3:5].copy())
        steps = step + 1

    # Check the final pose as well, including when the last allowed step moved
    # the robot into the goal or into an obstacle.
    final_nearest_distance = _nearest_obstacle_distance(state, all_points)
    minimum_clearance = min(
        minimum_clearance,
        final_nearest_distance - ROBOT_RADIUS,
    )
    final_goal_distance = float(np.linalg.norm(goal - state[:2]))

    if status == "timeout":
        if final_nearest_distance <= ROBOT_RADIUS:
            status = "collision"
        elif final_goal_distance <= GOAL_TOLERANCE:
            status = "success"

    positions = np.asarray(history, dtype=float)
    path_length = float(
        np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    ) if len(positions) > 1 else 0.0

    commands = np.asarray(command_history, dtype=float)
    if len(commands) > 1:
        mean_abs_linear_acceleration = float(
            np.mean(np.abs(np.diff(commands[:, 0]))) / dt
        )
        mean_abs_angular_acceleration = float(
            np.mean(np.abs(np.diff(commands[:, 1]))) / dt
        )
    else:
        mean_abs_linear_acceleration = 0.0
        mean_abs_angular_acceleration = 0.0

    success = status == "success"
    return {
        "config_id": str(weights["config_id"]),
        "alpha": float(weights["alpha"]),
        "beta": float(weights["beta"]),
        "gamma": float(weights["gamma"]),
        "scenario": scenario,
        "status": status,
        "success": int(success),
        "collision": int(status == "collision"),
        "steps": steps,
        "simulation_time_s": steps * dt,
        "completion_time_s": steps * dt if success else "",
        "path_length_m": path_length,
        "final_goal_distance_m": final_goal_distance,
        "minimum_clearance_m": minimum_clearance,
        "mean_abs_linear_acceleration_mps2": mean_abs_linear_acceleration,
        "mean_abs_angular_acceleration_radps2": mean_abs_angular_acceleration,
    }


def summarize_results(
    raw_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate trial metrics by weight configuration."""
    config_ids = list(dict.fromkeys(str(row["config_id"]) for row in raw_rows))
    summary_rows = []

    for config_id in config_ids:
        group = [row for row in raw_rows if row["config_id"] == config_id]
        successful = [row for row in group if row["success"] == 1]

        success_count = len(successful)
        collision_count = sum(int(row["collision"]) for row in group)
        scenario_count = len(group)

        first = group[0]
        summary_rows.append(
            {
                "rank": "",
                "config_id": config_id,
                "alpha": first["alpha"],
                "beta": first["beta"],
                "gamma": first["gamma"],
                "scenario_count": scenario_count,
                "success_count": success_count,
                "success_rate_pct": 100.0 * success_count / scenario_count,
                "collision_count": collision_count,
                "collision_rate_pct": 100.0 * collision_count / scenario_count,
                "mean_completion_time_success_s": _mean_or_none(
                    float(row["completion_time_s"]) for row in successful
                ),
                "mean_path_length_success_m": _mean_or_none(
                    float(row["path_length_m"]) for row in successful
                ),
                "mean_path_length_all_m": _mean_or_none(
                    float(row["path_length_m"]) for row in group
                ),
                "mean_minimum_clearance_m": _mean_or_none(
                    float(row["minimum_clearance_m"])
                    for row in group
                    if math.isfinite(float(row["minimum_clearance_m"]))
                ),
                "mean_abs_linear_acceleration_mps2": _mean_or_none(
                    float(row["mean_abs_linear_acceleration_mps2"])
                    for row in group
                ),
                "mean_abs_angular_acceleration_radps2": _mean_or_none(
                    float(row["mean_abs_angular_acceleration_radps2"])
                    for row in group
                ),
            }
        )

    # Rank primarily by success rate, then successful completion time,
    # then mean clearance. This is a sorting aid, not a universal optimum.
    summary_rows.sort(
        key=lambda row: (
            -float(row["success_rate_pct"]),
            (
                float(row["mean_completion_time_success_s"])
                if row["mean_completion_time_success_s"] is not None
                else math.inf
            ),
            -(
                float(row["mean_minimum_clearance_m"])
                if row["mean_minimum_clearance_m"] is not None
                else -math.inf
            ),
        )
    )

    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    return summary_rows


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep alpha, beta and gamma weights for the DWA planner."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help="Scenarios to evaluate (default: all three).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation time step in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Maximum steps per trial (default: 250).",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.1,
        help="Weight grid interval; must divide 1.0 evenly (default: 0.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "experiments" / "results",
        help="Directory for CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_weight_configs(args.weight_step)
    total_trials = len(configs) * len(args.scenarios)

    print("=== DWA parameter sweep ===")
    print(f"Weight configurations : {len(configs)}")
    print(f"Scenarios             : {', '.join(args.scenarios)}")
    print(f"Total trials          : {total_trials}")
    print(f"dt / max_steps        : {args.dt} s / {args.max_steps}")
    print("Running without Matplotlib visualization...\n")

    raw_rows = []
    for config_index, weights in enumerate(configs, start=1):
        for scenario in args.scenarios:
            result = run_trial(
                scenario=scenario,
                weights=weights,
                dt=args.dt,
                max_steps=args.max_steps,
            )
            raw_rows.append(result)
            print(
                f"[{len(raw_rows):>3}/{total_trials}] "
                f"{str(weights['config_id']):<26} "
                f"{scenario:<7} {result['status']:<9} "
                f"time={result['simulation_time_s']:.1f}s "
                f"path={result['path_length_m']:.2f}m"
            )

    summary_rows = summarize_results(raw_rows)

    raw_path = args.output_dir / "dwa_sweep_raw.csv"
    summary_path = args.output_dir / "dwa_sweep_summary.csv"
    write_csv(raw_path, RAW_FIELDS, raw_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    print("\n=== Top 10 configurations (ranking aid) ===")
    print("Rank  alpha  beta   gamma  Success  Mean time  Mean clearance")
    for row in summary_rows[:10]:
        mean_time = row["mean_completion_time_success_s"]
        mean_clearance = row["mean_minimum_clearance_m"]
        time_text = f"{mean_time:.2f}s" if mean_time is not None else "N/A"
        clearance_text = (
            f"{mean_clearance:.2f}m"
            if mean_clearance is not None
            else "N/A"
        )
        print(
            f"{int(row['rank']):>4}  "
            f"{float(row['alpha']):.2f}   "
            f"{float(row['beta']):.2f}   "
            f"{float(row['gamma']):.2f}    "
            f"{float(row['success_rate_pct']):5.1f}%  "
            f"{time_text:>9}  {clearance_text:>14}"
        )

    print("\nCSV files saved:")
    print(f"  Raw results : {raw_path}")
    print(f"  Summary     : {summary_path}")


if __name__ == "__main__":
    main()
