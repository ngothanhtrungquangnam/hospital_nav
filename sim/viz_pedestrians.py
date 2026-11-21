"""sim/viz_pedestrians.py -- Mo phong dang nguoi, xe lan, hop thuoc & benh nhan tren toan benh vien.

Chay (tu THU MUC GOC cua du an):
    python -m sim.viz_pedestrians --animate
    python -m sim.viz_pedestrians --animate --full-map
    python -m sim.viz_pedestrians --animate --show-collision-circles
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D

from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from env.pedestrians import (
    VALID_PED_KINDS,
    Pedestrian,
    PedestrianManager,
    default_hospital_pedestrians,
)
from sensors.lidar import Lidar2D

ROLE_BY_PED_ID: dict[int, str] = {
    1: "visitor",     # Nguoi di kham tren hanh lang chinh
    2: "nurse_cart",  # Y ta day hop thuoc / xe thuoc
    3: "wheelchair",  # Nguoi di xe lan
    4: "patient",     # Benh nhan o sanh cho
    5: "patient",     # Benh nhan di cham doc hanh lang chinh
    6: "visitor",     # Nguoi di kham di doc hanh lang chinh
    7: "patient",     # Benh nhan o sanh cho
    8: "nurse_cart",  # Y ta day hop thuoc o khu vuc phia dong / gan thang may
    9: "wheelchair",  # Nguoi di xe lan o khu vuc phia tay / gan cap cuu
    10: "visitor",    # Nguoi di lai khu cua vao Khoa Noi / hanh lang
}


def all_areas_pedestrians(seed: int | None = 42) -> PedestrianManager:
    """Tao kich ban nguoi di bo, benh nhan, xe lan va y ta day hop thuoc o nhieu khu vuc."""
    base_mgr = default_hospital_pedestrians(seed=seed)
    extra_specs = [
        # P5: Benh nhan di cham doc hanh lang chinh
        Pedestrian(
            x=9.2,
            y=10.6,
            radius=0.27,
            speed=0.45,
            waypoints=[(9.2, 10.6), (23.5, 10.6), (9.2, 10.6)],
            loop=True,
            kind="patient",
            ped_id=5,
        ),
        # P6: Nguoi di kham di nguoc chieu tren hanh lang chinh
        Pedestrian(
            x=26.5,
            y=9.1,
            radius=0.27,
            speed=0.62,
            waypoints=[(26.5, 9.1), (14.5, 9.1), (26.5, 9.1)],
            loop=True,
            kind="visitor",
            ped_id=6,
        ),
        # P7: Benh nhan di lai o Sanh cho (di qua luong trong giua cac hang ghe)
        Pedestrian(
            x=15.0,
            y=5.3,
            radius=0.27,
            speed=0.48,
            waypoints=[(15.0, 5.3), (25.5, 5.3), (25.5, 3.6), (15.0, 3.6)],
            loop=True,
            kind="patient",
            ped_id=7,
        ),
        # P8: Y ta day hop thuoc o khu hanh lang phia dong / gan Thang may
        Pedestrian(
            x=29.5,
            y=10.0,
            radius=0.28,
            speed=0.60,
            waypoints=[(29.5, 10.0), (36.5, 10.0), (29.5, 10.0)],
            loop=True,
            kind="nurse_cart",
            ped_id=8,
        ),
        # P9: Nguoi di xe lan o khu vuc phia tay / gan Cap cuu & cua vao hanh lang
        Pedestrian(
            x=5.5,
            y=10.0,
            radius=0.29,
            speed=0.52,
            waypoints=[(5.5, 10.0), (11.5, 10.0), (5.5, 10.0)],
            loop=True,
            kind="wheelchair",
            ped_id=9,
        ),
        # P10: Nguoi di kham di qua khu vuc truoc Khoa Noi
        Pedestrian(
            x=15.2,
            y=10.2,
            radius=0.26,
            speed=0.58,
            waypoints=[(15.2, 10.2), (15.2, 13.0), (18.0, 13.0), (15.2, 10.2)],
            loop=True,
            kind="visitor",
            ped_id=10,
        ),
    ]
    return PedestrianManager(list(base_mgr._specs) + extra_specs, seed=seed)


def _clean_texts_outside_limits(ax, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
    """An cac nhan van ban nam ngoai khung nhin cua truc ax."""
    for txt in list(ax.texts):
        tx, ty = txt.get_position()
        if not (x_lim[0] <= tx <= x_lim[1] and y_lim[0] <= ty <= y_lim[1]):
            txt.set_visible(False)


def _transform_to_ax(ax, cx: float, cy: float, heading_rad: float):
    """Xoay quanh (0, 0) theo `heading_rad - pi/2` (truc +y la huong tien) roi dich toi (cx, cy)."""
    return Affine2D().rotate(heading_rad - math.pi / 2.0).translate(cx, cy) + ax.transData


def _draw_person_figure(
    ax,
    cx: float,
    cy: float,
    r: float,
    heading_rad: float,
    face_color: str = "#f59e0b",
    head_color: str = "#fef3c7",
    edge_color: str = "#111827",
    zorder: int = 5,
) -> list:
    """Ve bieu tuong con nguoi dang top-down (vai, dau, mui ten huong di)."""
    tf = _transform_to_ax(ax, cx, cy, heading_rad)
    artists = []

    torso = Ellipse(
        (0.0, 0.0),
        width=r * 2.15,
        height=r * 1.15,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=1.5,
        zorder=zorder,
    )
    torso.set_transform(tf)
    ax.add_patch(torso)
    artists.append(torso)

    head = Circle(
        (0.0, r * 0.22),
        radius=r * 0.46,
        facecolor=head_color,
        edgecolor=edge_color,
        linewidth=1.3,
        zorder=zorder + 1,
    )
    head.set_transform(tf)
    ax.add_patch(head)
    artists.append(head)

    nose = Polygon(
        [(-r * 0.24, r * 0.72), (r * 0.24, r * 0.72), (0.0, r * 1.12)],
        closed=True,
        facecolor=edge_color,
        edgecolor=edge_color,
        zorder=zorder + 1,
    )
    nose.set_transform(tf)
    ax.add_patch(nose)
    artists.append(nose)

    return artists


def _draw_wheelchair_figure(
    ax,
    cx: float,
    cy: float,
    r: float,
    heading_rad: float,
    zorder: int = 5,
) -> list:
    """Ve bieu tuong xe lan (2 banh xe hai ben + ghe + nguoi ngoi)."""
    tf = _transform_to_ax(ax, cx, cy, heading_rad)
    artists = []

    for side in (-1.0, 1.0):
        wheel = FancyBboxPatch(
            (side * r * 0.92 - r * 0.14, -r * 0.78),
            width=r * 0.28,
            height=r * 1.56,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#334155",
            edgecolor="#0f172a",
            linewidth=1.3,
            zorder=zorder,
        )
        wheel.set_transform(tf)
        ax.add_patch(wheel)
        artists.append(wheel)

    seat = FancyBboxPatch(
        (-r * 0.72, -r * 0.58),
        width=r * 1.44,
        height=r * 1.16,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#38bdf8",
        edgecolor="#0f172a",
        linewidth=1.4,
        zorder=zorder + 1,
    )
    seat.set_transform(tf)
    ax.add_patch(seat)
    artists.append(seat)

    rider_head = Circle(
        (0.0, r * 0.08),
        radius=r * 0.44,
        facecolor="#fde68a",
        edgecolor="#0f172a",
        linewidth=1.3,
        zorder=zorder + 2,
    )
    rider_head.set_transform(tf)
    ax.add_patch(rider_head)
    artists.append(rider_head)

    return artists


def _draw_med_box_prop(
    ax,
    cx: float,
    cy: float,
    r: float,
    heading_rad: float,
    zorder: int = 5,
) -> list:
    """Ve hop thuoc / xe thuoc nam phia truoc nguoi day (y ta)."""
    tf = _transform_to_ax(ax, cx, cy, heading_rad)
    artists = []

    box_w = r * 1.45
    box_h = r * 1.05
    box_y0 = r * 1.18

    (handle,) = ax.plot(
        *zip(*[
            tf.transform((0.0, r * 0.75)),
            tf.transform((0.0, box_y0)),
        ]),
        "-",
        color="#475569",
        lw=1.8,
        zorder=zorder,
    )
    artists.append(handle)

    box = FancyBboxPatch(
        (-box_w / 2.0, box_y0),
        width=box_w,
        height=box_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#ecfeff",
        edgecolor="#0f766e",
        linewidth=1.5,
        zorder=zorder + 1,
    )
    box.set_transform(tf)
    ax.add_patch(box)
    artists.append(box)

    cross_size = r * 0.24
    box_yc = box_y0 + box_h / 2.0
    (h_bar,) = ax.plot(
        *zip(*[
            tf.transform((-cross_size, box_yc)),
            tf.transform((cross_size, box_yc)),
        ]),
        "-",
        color="#dc2626",
        lw=1.6,
        zorder=zorder + 2,
    )
    (v_bar,) = ax.plot(
        *zip(*[
            tf.transform((0.0, box_yc - cross_size)),
            tf.transform((0.0, box_yc + cross_size)),
        ]),
        "-",
        color="#dc2626",
        lw=1.6,
        zorder=zorder + 2,
    )
    artists.extend([h_bar, v_bar])

    return artists


def _draw_collision_circles(ax, ped: Pedestrian, zorder: int = 4) -> list:
    """Ve cac hinh tron va cham (1 hoac 2 vong) tu `ped.as_circles()` de doi chieu."""
    artists = []
    for ccx, ccy, ccr in ped.as_circles():
        circ = Circle(
            (ccx, ccy),
            radius=ccr,
            fill=True,
            facecolor="#ef4444",
            alpha=0.10,
            edgecolor="#dc2626",
            linestyle="--",
            linewidth=1.0,
            zorder=zorder,
        )
        ax.add_patch(circ)
        artists.append(circ)
    return artists


def _entity_role(ped: Pedestrian, idx: int) -> str:
    kind = getattr(ped, "kind", None)
    if kind in VALID_PED_KINDS:
        return str(kind)
    pid = ped.ped_id if ped.ped_id else (idx + 1)
    return ROLE_BY_PED_ID.get(pid, "visitor")


def _entity_label(role: str, idx: int) -> str:
    if role == "nurse_cart":
        return f"Y ta P{idx + 1} + hop thuoc"
    if role == "wheelchair":
        return f"Xe lan P{idx + 1}"
    if role == "patient":
        return f"Benh nhan P{idx + 1}"
    return f"Nguoi kham P{idx + 1}"


def _draw_entities_layer(
    ax,
    pedestrians: list[Pedestrian],
    trails: list[list[tuple[float, float]]],
    robot_xy: tuple[float, float] | None = None,
    show_info: bool = True,
    show_collision_circles: bool = False,
    z_base: int = 5,
) -> list:
    """Ve benh nhan, nguoi di kham, xe lan va hop thuoc kem quy dao & nhan thong tin."""
    artists = []
    for i, ped in enumerate(pedestrians):
        cx, cy, r = ped.x, ped.y, ped.radius
        speed = math.hypot(ped.vx, ped.vy)
        heading = math.atan2(ped.vy, ped.vx) if speed > 1e-4 else ped.heading
        role = _entity_role(ped, i)

        if i < len(trails) and len(trails[i]) > 1:
            xs, ys = zip(*trails[i])
            if role == "nurse_cart":
                trail_color = "#0f766e"
            elif role == "wheelchair":
                trail_color = "#0284c7"
            elif role == "patient":
                trail_color = "#2563eb"
            else:
                trail_color = "#d97706"
            (line,) = ax.plot(
                xs,
                ys,
                "--",
                lw=1.4,
                color=trail_color,
                alpha=0.82,
                zorder=z_base - 1,
                label="Quy dao di chuyen" if (show_info and i == 0) else "",
            )
            artists.append(line)

        if show_collision_circles:
            artists.extend(_draw_collision_circles(ax, ped, zorder=z_base - 1))

        if role == "wheelchair":
            artists.extend(_draw_wheelchair_figure(ax, cx, cy, r, heading, zorder=z_base))
        elif role == "nurse_cart":
            artists.extend(_draw_person_figure(ax, cx, cy, r, heading, face_color="#14b8a6", zorder=z_base))
            artists.extend(_draw_med_box_prop(ax, cx, cy, r, heading, zorder=z_base))
        elif role == "patient":
            artists.extend(
                _draw_person_figure(
                    ax, cx, cy, r, heading, face_color="#60a5fa", head_color="#eff6ff", zorder=z_base
                )
            )
        else:
            artists.extend(_draw_person_figure(ax, cx, cy, r, heading, face_color="#f59e0b", zorder=z_base))

        if show_info:
            parts = [_entity_label(role, i), f"v={speed:.2f}m/s"]
            if robot_xy is not None:
                d_rob = math.hypot(cx - robot_xy[0], cy - robot_xy[1])
                parts.append(f"d={d_rob:.1f}m")
            txt = ax.text(
                cx,
                cy + r + 0.22,
                " | ".join(parts),
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="#94a3b8", lw=0.7, alpha=0.94),
                zorder=z_base + 4,
            )
            artists.append(txt)

    return artists


def render_snapshot(
    world: HospitalMap,
    mgr: PedestrianManager,
    lidar: Lidar2D,
    robot_pose: tuple[float, float, float] = (13.0, 10.0, 0.0),
    steps: int = 140,
    dt: float = 0.05,
    x_lim: tuple[float, float] = (3.5, 37.5),
    y_lim: tuple[float, float] = (2.5, 14.5),
    show_collision_circles: bool = False,
    save_png: Path | None = Path("pedestrians_corridor.png"),
    show: bool = True,
) -> plt.Figure:
    """Xuat anh mo phong 2 khung: toan canh benh vien va khu vuc hanh lang + sanh."""
    mgr.reset(seed=mgr.seed)
    trails: list[list[tuple[float, float]]] = [[(p.x, p.y)] for p in mgr.pedestrians]

    for _ in range(steps):
        circles = mgr.step(dt, world=world)
        for idx, p in enumerate(mgr.pedestrians):
            trails[idx].append((p.x, p.y))

    circles = mgr.as_circles()
    scan = lidar.scan(world, list(robot_pose), circles=circles)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.2, 9.0),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    world.plot(
        ax=ax1,
        show_inflated=False,
        title=f"Toan canh mat bang benh vien (t = {steps * dt:.1f}s, {len(mgr)} tac nhan dong)",
    )
    _draw_entities_layer(
        ax1,
        mgr.pedestrians,
        trails,
        robot_xy=None,
        show_info=False,
        show_collision_circles=show_collision_circles,
        z_base=5,
    )

    world.plot(
        ax=ax2,
        show_inflated=False,
        title="Hanh lang, sanh cho & cac khu lan can: benh nhan, nguoi kham, xe lan, hop thuoc & LiDAR",
    )
    ax2.set_xlim(*x_lim)
    ax2.set_ylim(*y_lim)
    _clean_texts_outside_limits(ax2, x_lim, y_lim)

    rx, ry, _ = robot_pose
    for ang, r_val, is_dyn in zip(scan.angles[::2], scan.ranges[::2], scan.dynamic[::2]):
        ex = rx + r_val * math.cos(float(ang))
        ey = ry + r_val * math.sin(float(ang))
        ax2.plot(
            [rx, ex],
            [ry, ey],
            "-",
            lw=0.45,
            alpha=0.22,
            color="#f59e0b" if is_dyn else "#ef4444",
            zorder=2,
        )

    if len(scan.points_static):
        ax2.scatter(
            scan.points_static[:, 0],
            scan.points_static[:, 1],
            s=14,
            c="#ef4444",
            zorder=4,
            label="Diem cham vat can tinh",
        )
    if len(scan.points_dynamic):
        ax2.scatter(
            scan.points_dynamic[:, 0],
            scan.points_dynamic[:, 1],
            s=28,
            c="#facc15",
            edgecolors="#111827",
            lw=0.65,
            zorder=6,
            label="Diem cham vat can dong",
        )

    ax2.add_patch(Circle((rx, ry), 0.30, fc="#1d4ed8", ec="#0f172a", lw=1.6, zorder=6))
    ax2.text(rx, ry - 0.56, "Robot", ha="center", va="top", fontsize=8, weight="bold", color="#1e3a8a", zorder=7)

    _draw_entities_layer(
        ax2,
        mgr.pedestrians,
        trails,
        robot_xy=(rx, ry),
        show_info=True,
        show_collision_circles=show_collision_circles,
        z_base=5,
    )
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.92)

    plt.tight_layout()
    if save_png is not None:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=180)
        print(f"Da luu anh: {save_png}")
    if show:
        plt.show()
    return fig


def run_animation(
    world: HospitalMap,
    mgr: PedestrianManager,
    lidar: Lidar2D,
    robot_pose: tuple[float, float, float] = (13.0, 10.0, 0.0),
    frames: int = 240,
    dt: float = 0.05,
    x_lim: tuple[float, float] = (3.5, 37.5),
    y_lim: tuple[float, float] = (2.5, 14.5),
    show_collision_circles: bool = False,
    save_gif: Path | None = None,
) -> None:
    """Chay hoat hinh chuyen dong benh nhan, nguoi di kham, xe lan va hop thuoc."""
    mgr.reset(seed=mgr.seed)
    trails: list[list[tuple[float, float]]] = [[(p.x, p.y)] for p in mgr.pedestrians]

    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    world.plot(ax=ax, show_inflated=False, title="Hoat hinh chuyen dong tren hanh lang, sanh cho & khu lan can")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    _clean_texts_outside_limits(ax, x_lim, y_lim)

    rx, ry, _ = robot_pose
    ax.add_patch(Circle((rx, ry), 0.30, fc="#1d4ed8", ec="#0f172a", lw=1.6, zorder=6))
    ax.text(rx, ry - 0.55, "Robot", ha="center", va="top", fontsize=8, weight="bold", color="#1e3a8a", zorder=7)

    dyn_scat = ax.scatter([], [], s=26, c="#facc15", edgecolors="#111827", lw=0.6, zorder=6, label="LiDAR dynamic")
    ax.legend(loc="upper right", fontsize=8)

    dynamic_artists: list = []

    def _update(_frame: int):
        nonlocal dynamic_artists
        for art in dynamic_artists:
            try:
                art.remove()
            except Exception:  # noqa: BLE001
                pass
        dynamic_artists = []

        circles = mgr.step(dt, world=world)
        for idx, p in enumerate(mgr.pedestrians):
            trails[idx].append((p.x, p.y))
            trails[idx][:] = trails[idx][-70:]

        scan = lidar.scan(world, list(robot_pose), circles=circles)
        if len(scan.points_dynamic):
            dyn_scat.set_offsets(scan.points_dynamic)
        else:
            dyn_scat.set_offsets(np.empty((0, 2)))

        dynamic_artists = _draw_entities_layer(
            ax,
            mgr.pedestrians,
            trails,
            robot_xy=(rx, ry),
            show_info=True,
            show_collision_circles=show_collision_circles,
            z_base=5,
        )
        return [dyn_scat] + dynamic_artists

    ani = FuncAnimation(fig, _update, frames=frames, interval=int(dt * 1000), blit=False)
    plt.tight_layout()
    if save_gif is not None:
        Path(save_gif).parent.mkdir(parents=True, exist_ok=True)
        ani.save(save_gif, writer="pillow", fps=max(1, int(round(1.0 / dt))))
        print(f"Da luu GIF: {save_gif}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mo phong benh nhan, nguoi di kham, xe lan va hop thuoc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--full-map", action="store_true", help="Mo rong khung nhin ra toan tang 40m x 20m.")
    parser.add_argument("--show-collision-circles", action="store_true", help="Ve vong tron va cham tu as_circles().")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--save-png", type=Path, default=Path("pedestrians_corridor.png"))
    parser.add_argument("--save-gif", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    world = HospitalMap().with_obstacles(UNMAPPED_CARTS)
    mgr = all_areas_pedestrians(seed=args.seed)
    lidar = Lidar2D(rng=np.random.default_rng(args.seed))

    x_lim = (0.0, 40.0) if args.full_map else (3.5, 37.5)
    y_lim = (0.0, 20.0) if args.full_map else (2.5, 14.5)

    if args.animate or args.save_gif is not None:
        run_animation(
            world=world,
            mgr=mgr,
            lidar=lidar,
            frames=args.steps if args.animate else max(args.steps, 120),
            dt=args.dt,
            x_lim=x_lim,
            y_lim=y_lim,
            show_collision_circles=args.show_collision_circles,
            save_gif=args.save_gif,
        )
    else:
        render_snapshot(
            world=world,
            mgr=mgr,
            lidar=lidar,
            steps=args.steps,
            dt=args.dt,
            x_lim=x_lim,
            y_lim=y_lim,
            show_collision_circles=args.show_collision_circles,
            save_png=args.save_png,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()