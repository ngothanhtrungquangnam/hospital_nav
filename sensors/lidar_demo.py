"""sensors/lidar_demo.py -- Hinh minh hoa LiDAR cho bao cao / slide (dung API cua env/map_view.py).

Chay (tu thu muc goc):  python -m sensors.lidar_demo      -> ghi figures/lidar_demo_{report,blueprint}.{png,pdf}

Hinh gom hai phan cung mot lan quet:
  tren : ban do phong to quanh robot (the gioi THAT): tia quet, diem cham tinh / nguoi di bo, vung bong sau nguoi
  duoi : do thi khoang cach theo goc (range profile) -- cach cac planner dang thuc su "nhin" du lieu LiDAR
"""
from __future__ import annotations

import math

import numpy as np

from env import map_view as mv
from env.hospital_map import UNMAPPED_CARTS, HospitalMap
from sensors.lidar import Lidar2D


def lidar_figure(world, pose, circles=None, style: str = "report", lidar: Lidar2D | None = None,
                 window: tuple = (10.5, 19.5, 7.9, 12.3), title: str = "Quét LiDAR 2D trên thế giới thật"):
    """Ve mot lan quet: ban do phong to (tren) + do thi range-theo-goc (duoi). Tra ve figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    C = mv.STYLES[style]
    lid = lidar if lidar is not None else Lidar2D()
    circles = np.zeros((0, 3)) if circles is None else np.asarray(circles, dtype=float)
    s = lid.scan(world, pose, circles=circles)

    fig = plt.figure(figsize=(9.0, 7.4), dpi=110)
    fig.patch.set_facecolor(C["figure_bg"])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.78], hspace=0.38, left=0.075, right=0.975, top=0.90, bottom=0.085)
    ax = fig.add_subplot(gs[0])
    mv.draw_map(world, ax, style=style, legend=False, show_labels=False, show_dims=False, show_door_tags=False,
                show_north=False, show_scalebar=False, show_room_codes=False, set_figure_bg=True)
    x0, x1, y0, y1 = window
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks(np.arange(math.ceil(x0), x1 + 1e-9, 1.0))
    ax.set_yticks(np.arange(math.ceil(y0), y1 + 1e-9, 1.0))
    ax.set_xticks(np.arange(math.ceil(x0 * 2) / 2, x1 + 1e-9, 0.5), minor=True)
    ax.set_yticks(np.arange(math.ceil(y0 * 2) / 2, y1 + 1e-9, 0.5), minor=True)
    mv._vn_axes(ax)
    for t_ in ax.texts:                                    # nhan nam ngoai khung phong to khong duoc lam phinh hinh
        t_.set_clip_on(True)

    shadow_pts = s.points                                  # tia + diem cham
    mv.draw_scan(ax, s, style, ray_alpha=0.22, hit_size=14.0, ray_step=1)
    if len(s.points_dynamic):
        pd = s.points_dynamic
        ax.scatter(pd[:, 0], pd[:, 1], s=16, c=C["ped"], edgecolors=mv._darker(C["ped"], 0.6), linewidths=0.5, zorder=9)
    mv.draw_pedestrians(ax, circles, style)
    mv.draw_robot(ax, pose[0], pose[1], pose[2], style, lidar_range=lid.max_range, zorder=12)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=C["text"], pad=22)
    ax.text(0.0, 1.012, f"{lid.n_rays} tia · {mv._vn(lid.angular_resolution * 180 / math.pi, 1)}°/tia · tầm {mv._vn(lid.max_range, 1)} m"
            f" · FOV {lid.fov_deg:.0f}° · nhiễu σ = {mv._vn(lid.noise_std * 1000, 0)} mm · robot đúng tỷ lệ",
            transform=ax.transAxes, fontsize=7.5, color=C["subtitle"], va="bottom")

    # ---- do thi range theo goc ----
    bx = fig.add_subplot(gs[1])
    bx.set_facecolor(C["axes_bg"])
    deg = np.degrees(s.angles_rel)
    st, dy, nr = s.hit & ~s.dynamic, s.hit & s.dynamic, ~s.hit
    bx.scatter(deg[nr], s.ranges[nr], s=5, c=C["spine"], alpha=0.8, linewidths=0, label="không có phản hồi (= tầm xa)")
    bx.scatter(deg[st], s.ranges[st], s=8, c=C["lidar"], linewidths=0, label="vật cản tĩnh (tường, nội thất, xe đẩy)")
    bx.scatter(deg[dy], s.ranges[dy], s=10, c=C["ped"], edgecolors=mv._darker(C["ped"], 0.6), linewidths=0.4,
               label="người đi bộ")
    bx.axhline(lid.max_range, color=C["spine"], lw=0.8, ls=":")
    bx.axhline(world.inflate_cells * world.res, color=C["inflate"], lw=0.9, ls="--")
    bx.text(178, world.inflate_cells * world.res + 0.07, f"vùng cấm {mv._vn(world.inflate_cells * world.res, 2)} m", ha="right",
            va="bottom", fontsize=7, color=C["inflate"])
    bx.set_xlim(-180, 180)
    bx.set_ylim(0, lid.max_range * 1.08)
    bx.set_xticks(np.arange(-180, 181, 45))
    bx.set_xlabel("góc tia so với hướng robot [độ]  (0° = phía trước)", color=C["room_text"], fontsize=9)
    bx.set_ylabel("khoảng cách [m]", color=C["room_text"], fontsize=9)
    bx.grid(color=C["grid"], lw=0.5, alpha=0.8)
    bx.tick_params(colors=C["room_text"], labelsize=8)
    from matplotlib.ticker import FuncFormatter
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    bx.yaxis.set_major_formatter(FuncFormatter(mv._vn_tick))
    for sp in bx.spines.values():
        sp.set_color(C["spine"])
    leg = bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False, fontsize=7.5, markerscale=2.0)
    for t_ in leg.get_texts():
        t_.set_color(C["text"])
    return fig


if __name__ == "__main__":
    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)
    pose = [13.0, 10.0, 0.0]
    people = np.array([[15.6, 10.1, 0.25]])
    for style in mv.STYLES:
        fig = lidar_figure(world, pose, people, style=style, lidar=Lidar2D(noise_std=0.0))
        for ext in ("png", "pdf"):
            print("da ghi:", mv.save_figure(fig, f"figures/lidar_demo_{style}.{ext}", 300))