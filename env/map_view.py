"""env/map_view.py -- Ve ban do benh vien kieu ban ve ky thuat (vector), toi uu cho bao cao va slide.

Chi lo HIEN THI, khong thay doi du lieu ban do.  Moi hinh la vector (xuat PDF/SVG khong vo hat).

Ham chinh
---------
draw_map(m, ax, ...)            ban do: san phong, tuong, cua, noi that, vat can chua biet, kich thuoc, thuoc ti le, chu thich
draw_robot(ax, x, y, theta)     robot DUNG KICH THUOC mo hinh (hinh tron r = ROBOT_RADIUS, banh o +-WHEEL_BASE/2)
draw_trajectory(ax, xy, ...)    quy dao to mau theo van toc, moc thoi gian moi N giay
draw_scan(ax, scan)             tia LiDAR + diem cham (nhan Lidar2D.scan)
draw_pedestrians(ax, circles)   nguoi di bo
draw_door_zoom(m, door_idx)     hinh phong to tai cua: robot that so voi khung cua
draw_pair(known, world)         ban do DA BIET va THE GIOI THAT xep doc, chung mot chu thich
save_figure(fig, path)          luu hinh dung chuan bao cao (cat le trang, font PDF nhung duoc)
save_map_figures(...)           xuat PNG + PDF cho ca hai style

Quy tac quan trong
------------------
- Ve robot luon theo DUNG ban kinh va khoang cach banh trong config.py, de hinh khop mo hinh va cham.
- Khong tron hai style (report / blueprint) trong cung mot figure: mau chu va mau nen se xung dot.
- Khi truyen ``ax`` co san, draw_map KHONG doi mau nen cua figure (tru khi set_figure_bg=True).
- Chu thich chi liet ke nhung gi THUC SU co tren hinh, dung mau voi mau ve.
- Dung dau phay thap phan (kieu Viet Nam) o moi nhan so, ke ca tick truc.
- ``target``: "report" (hinh ~9 in, chen vao Word/LaTeX rong ~16 cm) hoac "slide" (16:9, chu to hon).

Mau nhan phong: vi tri nhan trong ROOM_LABEL_POS la lop HIEN THI (da kiem tra khong de len noi that / cua /
diem G); ``Room.label_xy`` cua HospitalMap chi dung cho ``HospitalMap.plot`` co ban.
"""
from __future__ import annotations

import math
import os
import unicodedata

import numpy as np

import config

STYLES = {
    "report": {
        "figure_bg": "#FFFFFF", "axes_bg": "#FFFFFF",
        "floor": {"pharmacy": "#E3F3E8", "ward": "#E6EEFB", "lab": "#EFE7F8", "emergency": "#FCE9E7",
                  "waiting": "#FFF4DC", "service": "#ECEEF2", "corridor": "#F6F7F9", "room": "#F6F7F9"},
        "wall": "#1E293B",
        "furniture": {"furniture": "#9DB2CA", "shelf": "#7DB8A6", "cabinet": "#8FA9C7", "bed": "#B6C5DC",
                      "lab": "#B7A3D4", "bench": "#CBA97A", "bin": "#8F9AA6"},
        "unmapped_fill": "#F59E0B", "unmapped_edge": "#9A4B07",
        "inflate": "#DC2626", "door": "#64748B", "grid": "#E2E8F0", "dim": "#475569",
        "text": "#0F172A", "room_text": "#1E293B", "subtitle": "#475569",
        "start": "#16A34A", "goal": "#DC2626", "marker_edge": "#FFFFFF", "spine": "#64748B",
        "robot_body": "#334155", "robot_accent": "#0284C7", "lidar": "#EF4444", "ped": "#F59E0B",
        "track_cmap": "viridis",
    },
    "blueprint": {
        "figure_bg": "#0A1F3F", "axes_bg": "#0A1F3F",
        "floor": {"pharmacy": "#0F3052", "ward": "#0E2D5A", "lab": "#16295C", "emergency": "#1A2A55",
                  "waiting": "#102F55", "service": "#0D2548", "corridor": "#0C2A4F", "room": "#0C2A4F"},
        "wall": "#DCEBFF",
        "furniture": {"furniture": "#5B86BA", "shelf": "#2A9AA8", "cabinet": "#3A6EA5", "bed": "#4A7FC0",
                      "lab": "#7B6BC4", "bench": "#B38D4F", "bin": "#9FB3C8"},
        "unmapped_fill": "#F59E0B", "unmapped_edge": "#FDE68A",
        "inflate": "#F87171", "door": "#7FB2E5", "grid": "#1F4C8A", "dim": "#8CC2F5",
        "text": "#E6F1FF", "room_text": "#C4D9FF", "subtitle": "#7FB2E5",
        "start": "#34D399", "goal": "#F87171", "marker_edge": "#0A1F3F", "spine": "#3B6AA8",
        "robot_body": "#1E293B", "robot_accent": "#38BDF8", "lidar": "#F87171", "ped": "#FBBF24",
        "track_cmap": "cool",
    },
}

# target -> (figsize map don, figsize cap doc, he so chu).  "report": chen vao tai lieu rong ~16 cm; "slide": 16:9.
TARGETS = {
    "report": {"figsize": (9.0, 5.6), "pair_figsize": (9.0, 9.0), "fs": 1.0},
    "slide": {"figsize": (13.33, 7.5), "pair_figsize": (13.33, 11.5), "fs": 1.45},
}

ROOM_NAMES = {
    "Nha thuoc": "NHÀ THUỐC", "Khoa Noi": "KHOA NỘI", "Khoa Ngoai": "KHOA NGOẠI",
    "Xet nghiem": "XÉT NGHIỆM", "Cap cuu": "CẤP CỨU", "Sanh cho": "SẢNH CHỜ TRUNG TÂM",
    "Thang may": "THANG MÁY / KHO", "Hanh lang chinh": "HÀNH LANG CHÍNH",
}
# Vi tri nhan phong co dinh (da kiem tra khong de len noi that / cua / diem G).
ROOM_LABEL_POS = {
    "Nha thuoc": (5.0, 16.5), "Khoa Noi": (15.0, 16.5), "Khoa Ngoai": (24.0, 16.5), "Xet nghiem": (34.2, 16.5),
    "Cap cuu": (6.5, 6.5), "Sanh cho": (20.0, 7.5), "Thang may": (34.0, 5.5), "Hanh lang chinh": (27.5, 10.3),
}
FURNITURE_NAMES = {"shelf": "Kệ thuốc"}          # chi nhan chu cho ke thuoc; cac loai khac nhan biet bang bieu tuong
ROOM_CODE_DY = {"Nha thuoc": 0.9, "Khoa Noi": 0.9, "Khoa Ngoai": 0.9, "Xet nghiem": 0.9}   # ma phong: tren ten (tranh diem G)
ROOM_CODES = {"Nha thuoc": "P.101", "Khoa Noi": "P.102", "Khoa Ngoai": "P.103", "Xet nghiem": "P.104",
              "Cap cuu": "P.001", "Sanh cho": "SC.01", "Thang may": "K.01", "Hanh lang chinh": "HL.01"}
# Ten trong chu thich + thu tu hien thi (chi loai nao CO tren ban do moi duoc liet ke).
FURNITURE_LEGEND = {"furniture": "Thiết bị cố định", "cabinet": "Tủ", "bed": "Giường bệnh", "lab": "Bàn xét nghiệm",
                    "shelf": "Kệ thuốc", "bench": "Ghế chờ", "bin": "Thùng rác"}
FURNITURE_ORDER = {k: i for i, k in enumerate(FURNITURE_LEGEND)}


# --------------------------------------------------------------------------------------------
# Tien ich
# --------------------------------------------------------------------------------------------
def _strip(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _vn(x: float, nd: int = 1) -> str:
    """So thuc -> chuoi dung dau phay thap phan (kieu Viet Nam)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _vn_tick(v: float, _pos=None) -> str:
    """Dinh dang tick: so nguyen khong co phan le, con lai 1 chu so, dau phay."""
    return _vn(v, 0) if abs(v - round(v)) < 1e-9 else _vn(v, 1)


def _vn_axes(ax) -> None:
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(_vn_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_vn_tick))


def _darker(color: str, k: float = 0.7) -> tuple:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r * k, g * k, b * k)


def _target(target: str) -> dict:
    if target not in TARGETS:
        raise ValueError(f"target phai la mot trong {list(TARGETS)}")
    return TARGETS[target]


def _rasterize(rects, shape, res) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in rects:
        c1, c2 = max(0, int(math.floor(x1 / res + 0.5))), min(shape[1], int(math.floor(x2 / res + 0.5)))
        r1, r2 = max(0, int(math.floor(y1 / res + 0.5))), min(shape[0], int(math.floor(y2 / res + 0.5)))
        if c2 > c1 and r2 > r1:
            mask[r1:r2, c1:c2] = True
    return mask


def wall_mask(m) -> np.ndarray:
    """Cac o la TUONG = grid tru noi that tru vat can chua biet (dung cho ve va cho test)."""
    shape, res = m.grid.shape, m.res
    furn = [r for _, r in getattr(m, "furniture", [])]
    return m.grid & ~_rasterize(furn, shape, res) & ~_rasterize(list(getattr(m, "unmapped", [])), shape, res)


def _mask_to_rects(mask: np.ndarray, res: float) -> list:
    """Gop cac o True thanh it hinh chu nhat nhat co the (run-length theo hang, gop theo chieu doc)."""
    n_rows = mask.shape[0]
    open_runs: dict = {}
    rects = []
    for r in range(n_rows + 1):
        runs = set()
        if r < n_rows:
            row = np.concatenate(([0], mask[r].astype(np.int8), [0]))
            d = np.diff(row)
            runs = set(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))
        for key in [k for k in open_runs if k not in runs]:
            r0 = open_runs.pop(key)
            rects.append((key[0] * res, r0 * res, (key[1] - key[0]) * res, (r - r0) * res))
        for key in runs:
            open_runs.setdefault(key, r)
    return rects


def _inflation_text(m) -> str:
    """Cau mo ta phong vat can, dua tren config (khong suy nguoc tu inflate_radius)."""
    r, sm = float(config.ROBOT_RADIUS), float(getattr(config, "SAFETY_MARGIN", 0.1))
    return (f"phồng vật cản: {_vn(r, 2)} m (robot) + {_vn(sm, 2)} m (biên an toàn) = {_vn(r + sm, 2)} m "
            f"→ làm tròn lên {m.inflate_cells} ô = {_vn(m.inflate_cells * m.res, 2)} m")


def _grid_text(m) -> str:
    return f"Lưới {m.n_cols}×{m.n_rows} ô · {_vn(m.res * 100, 0)} cm/ô · {_inflation_text(m)}"


def _furniture_detail(ax, kind, rect, face, fs, zorder=3.4):
    """Ve chi tiet BEN TRONG khoi noi that (goi/chan giuong, cac o ban xet nghiem, ngan ke, cho ngoi...).

    Chi la lop trang tri nam gon trong hinh chu nhat cua noi that, nen vung chiem dung / va cham khong doi.
    """
    from matplotlib.patches import Circle, FancyBboxPatch

    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    ec, lw = _darker(face, 0.6), 0.6

    def rr(x, y, ww, hh, fc="none", a=1.0, r=0.05):
        ax.add_patch(FancyBboxPatch((x, y), ww, hh, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
                                    edgecolor=ec, linewidth=lw, alpha=a, zorder=zorder))

    if kind == "bed":                                                  # cum giuong (moi giuong ~1,0 x 2,0 m), dau giuong sat tuong
        n = max(1, int(round(w / 1.0)))
        bw = w / n
        head_up = 0.5 * (y1 + y2) > 10.0
        for i in range(n):
            bx = x1 + i * bw
            rr(bx + 0.07, y1 + 0.07, bw - 0.14, h - 0.14, r=0.06)
            py = y2 - 0.07 - 0.42 if head_up else y1 + 0.07
            rr(bx + 0.17, py + 0.04, bw - 0.34, 0.34, fc="#FFFFFF", a=0.9, r=0.08)           # goi
            fy = (y2 - 0.07 - 0.42 - 0.85) if head_up else (y1 + 0.07 + 0.42 + 0.85)        # mep chan
            ax.plot([bx + 0.07, bx + bw - 0.07], [fy, fy], color=ec, lw=lw, zorder=zorder)
    elif kind == "lab":                                                # luoi o ban + thiet bi (kinh hien vi / may)
        nx, ny = max(1, int(round(w / 1.0))), max(1, int(round(h / 1.0)))
        cw, ch = w / nx, h / ny
        for i in range(nx):
            for j in range(ny):
                rr(x1 + i * cw + 0.08, y1 + j * ch + 0.08, cw - 0.16, ch - 0.16, r=0.04)
                ax.add_patch(Circle((x1 + (i + 0.5) * cw, y1 + (j + 0.5) * ch), 0.14, facecolor=ec, edgecolor="none",
                                    zorder=zorder))
    elif kind == "shelf":                                              # ngan ke thuoc moi 1 m
        for k in range(1, int(w // 1.0) + 1):
            if x1 + k * 1.0 < x2 - 0.05:
                ax.plot([x1 + k * 1.0] * 2, [y1 + 0.08, y2 - 0.08], color=ec, lw=lw, zorder=zorder)
    elif kind == "bench":                                              # cho ngoi moi 0,5 m
        for k in range(1, int(round(w / 0.5))):
            ax.plot([x1 + k * 0.5] * 2, [y1 + 0.05, y2 - 0.05], color=ec, lw=lw, zorder=zorder)
    elif kind == "bin":
        ax.add_patch(Circle(((x1 + x2) / 2, (y1 + y2) / 2), min(w, h) * 0.32, facecolor="none", edgecolor=ec,
                            linewidth=lw, zorder=zorder))


def _legend_handles(m, C, tr, show_inflated: bool, extra=None) -> list:
    """Chu thich: CHI nhung loai co that tren hinh, mau dung bang mau ve."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    h = [Patch(facecolor=C["wall"], label=tr("Tường cố định"))]
    kinds = sorted(dict.fromkeys(k for k, _ in getattr(m, "furniture", [])), key=lambda k: FURNITURE_ORDER.get(k, 99))
    for k in kinds:
        face = C["furniture"].get(k, C["furniture"]["furniture"])
        h.append(Patch(facecolor=face, edgecolor=_darker(face), label=tr(FURNITURE_LEGEND.get(k, k))))
    if getattr(m, "unmapped", None):
        h.append(Patch(facecolor=C["unmapped_fill"], edgecolor=C["unmapped_edge"], hatch="////",
                       label=tr("Vật cản chưa biết (chỉ LiDAR thấy)")))
    h.append(Line2D([0], [0], color=C["door"], lw=1.0, ls=":",
                    label=tr(f"Cửa rộng {_vn(getattr(m, 'door_width', 1.2))} m (cung mở)")))
    if show_inflated:
        h.append(Line2D([0], [0], color=C["inflate"], lw=1.0, ls="--",
                        label=tr(f"Biên vùng cấm tâm robot ({_vn(m.inflate_cells * m.res, 2)} m)")))
    h += [Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor=C["start"],
                 markeredgecolor=C["marker_edge"], label=tr("Xuất phát (S)")),
          Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor=C["goal"],
                 markeredgecolor=C["marker_edge"], label=tr("Điểm giao hàng G1–G4"))]
    return h + list(extra or [])


def save_figure(fig, path: str, dpi: int = 300) -> str:
    """Luu hinh dung chuan bao cao: cat le trang thua, font PDF/PS nhung duoc (Type 42)."""
    import matplotlib as mpl

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    return path


# --------------------------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------------------------
def draw_robot(ax, x: float, y: float, theta: float, style: str = "report", color: str | None = None,
               label: str | None = None, label_offset: tuple = (0.0, 0.75), ring_radius: float | None = None,
               lidar_range: float | None = None, fov_deg: float | None = None, wheel_angle: float = 0.0,
               zorder: float = 12.0, alpha: float = 1.0, scale: float = 1.0):
    """Ve robot vi sai DUNG KICH THUOC mo hinh (khi scale=1.0) hoac phong to theo he so ``scale``."""
    from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle, Wedge
    from matplotlib.transforms import Affine2D

    C = STYLES[style]
    r = float(config.ROBOT_RADIUS)
    wb = float(getattr(config, "WHEEL_BASE", 0.36))
    wr = float(getattr(config, "WHEEL_RADIUS", 0.075))
    accent = color or C["robot_accent"]
    T = Affine2D().scale(scale).rotate(theta).translate(x, y) + ax.transData

    def add(patch, dz=0.0):
        patch.set_transform(T)
        patch.set_zorder(zorder + dz)
        ax.add_patch(patch)

    if fov_deg:
        r_fov = float(lidar_range) if lidar_range else 1.2
        half_fov = 0.5 * float(fov_deg)
        deg = math.degrees(theta)
        ax.add_patch(Wedge((x, y), r_fov, deg - half_fov, deg + half_fov,
                           facecolor=accent, alpha=0.06, edgecolor="none", zorder=zorder - 2))
    if lidar_range:
        ax.add_patch(Circle((x, y), lidar_range, facecolor="none", edgecolor=C["lidar"], alpha=0.45,
                            linewidth=0.75, linestyle=(0, (1, 3)), zorder=zorder - 2))
    if ring_radius:
        ax.add_patch(Circle((x, y), ring_radius, facecolor="none", edgecolor=C["inflate"], linewidth=0.95,
                            linestyle=(0, (4, 3)), zorder=zorder - 1))

    # 1. Vo ngoai + vong chia ky thuat + vach chia goc moi 15 do
    add(Circle((0, 0), r, facecolor=C["robot_body"], edgecolor=accent, linewidth=1.7, alpha=alpha), dz=0.0)
    add(Circle((0, 0), 0.87 * r, facecolor="none", edgecolor=accent, linewidth=0.6,
               alpha=0.35 * alpha), dz=0.03)

    for deg_tick in range(0, 360, 15):
        rad_t = math.radians(deg_tick)
        if deg_tick % 90 == 0:
            r_in, r_out, thick_y, a_tick = 0.85 * r, 0.99 * r, 0.011, 0.78
        elif deg_tick % 45 == 0:
            r_in, r_out, thick_y, a_tick = 0.88 * r, 0.99 * r, 0.008, 0.62
        else:
            r_in, r_out, thick_y, a_tick = 0.93 * r, 0.99 * r, 0.005, 0.38
        seg_len = max(0.008, r_out - r_in)
        mid_r = 0.5 * (r_in + r_out)
        tick_patch = Rectangle((-0.5 * seg_len, -0.5 * thick_y), seg_len, thick_y,
                               facecolor=accent, edgecolor="none", alpha=a_tick * alpha)
        tick_patch.set_transform(Affine2D().rotate(rad_t).translate(mid_r * math.cos(rad_t), mid_r * math.sin(rad_t)) + T)
        tick_patch.set_zorder(zorder + 0.05)
        ax.add_patch(tick_patch)

    # 2. Khung chassis ben trong tao cam giac ket cau co khi
    deck_w = min(1.36 * r, max(0.24, 1.15 * r))
    deck_h = min(1.12 * r, max(0.18, 0.82 * wb))
    add(FancyBboxPatch(
        (-0.52 * deck_w, -0.5 * deck_h),
        deck_w,
        deck_h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor="#132238",
        edgecolor=accent,
        linewidth=0.75,
        alpha=0.55 * alpha,
    ), dz=0.07)

    # 3. Truc vi sai + cang noi + 2 banh xe co ranh quay theo wheel_angle
    half_wb = 0.5 * wb
    wheel_len = 2.0 * wr
    wheel_thick = max(0.038, 0.44 * wr)
    add(Rectangle((-0.006, -half_wb), 0.012, wb,
                  facecolor="#94A3B8", edgecolor="none", alpha=0.5 * alpha), dz=0.09)

    for side in (+1, -1):
        wy = side * half_wb
        # Cang/truc ngang giua khung va banh
        add(Rectangle((-0.25 * wr, wy - 0.006), 0.5 * wr, 0.012,
                      facecolor="#CBD5E1", edgecolor="none", alpha=0.65 * alpha), dz=0.11)
        # Than banh xe
        add(Rectangle((-wr, wy - 0.5 * wheel_thick), wheel_len, wheel_thick,
                      facecolor="#090E1A", edgecolor="#E2E8F0", linewidth=0.75, alpha=alpha), dz=0.14)
        # Vach chi bao goc lan cua banh xe
        spoke_x = math.cos(wheel_angle) * (0.72 * wr)
        add(Rectangle((spoke_x - 0.007, wy - 0.5 * wheel_thick), 0.014, wheel_thick,
                      facecolor=accent, edgecolor="none", alpha=0.95 * alpha), dz=0.18)

    # 4. Cung can truoc + mui chi huong +x gon, sac net
    add(Arc((0, 0), 2 * 0.94 * r, 2 * 0.94 * r, theta1=-35, theta2=35,
            color=accent, linewidth=2.1, alpha=alpha), dz=0.21)

    tip_x = min(0.72 * r, max(0.20, 0.48 * r + wr))
    tail_x = 0.25 * r
    wing_y = min(0.26 * r, max(0.06, 0.85 * wr))
    notch_x = 0.33 * r
    add(Polygon([[tip_x, 0.0], [tail_x, wing_y], [notch_x, 0.0], [tail_x, -wing_y]],
                closed=True, facecolor=accent, edgecolor="#090E1A", linewidth=0.55, alpha=alpha), dz=0.25)

    # 5. Cum LiDAR o tam (0, 0)
    add(Circle((0, 0), max(0.052, 0.20 * r), facecolor="#090E1A", edgecolor=accent,
               linewidth=1.1, alpha=alpha), dz=0.35)
    add(Circle((0, 0), max(0.032, 0.12 * r), facecolor="none", edgecolor="#94A3B8",
               linewidth=0.55, alpha=0.75 * alpha), dz=0.40)
    add(Circle((0, 0), max(0.016, 0.06 * r), facecolor=C["lidar"], edgecolor="none",
               alpha=alpha), dz=0.45)

    if label:
        ax.text(x + label_offset[0], y + label_offset[1], label, ha="center", va="center", fontsize=7,
                fontweight="bold", color="#FFFFFF", zorder=zorder + 2,
                bbox=dict(boxstyle="circle,pad=0.22", facecolor=accent, edgecolor=C["marker_edge"], linewidth=0.8))


def draw_scan(ax, scan, style: str = "report", show_rays: bool = True, ray_alpha: float = 0.12, hit_size: float = 7.0,
              ray_step: int = 2):
    """Ve mot lan quet LiDAR (``Lidar2D.scan``): tia mong + diem cham."""
    from matplotlib.collections import LineCollection

    C = STYLES[style]
    o = np.asarray(scan.origin, dtype=float).ravel()[:2]
    rng = np.asarray(scan.ranges, dtype=float)
    rng = np.where(np.isfinite(rng), rng, 0.0)                          # tia khong cham (inf) khong ve
    end = np.column_stack([o[0] + rng * np.cos(scan.angles), o[1] + rng * np.sin(scan.angles)])
    if show_rays:
        end_rays = end[::max(1, ray_step)]                                  # ve thua hon cho do ron
        segs = np.stack([np.broadcast_to(o, end_rays.shape), end_rays], axis=1)
        ax.add_collection(LineCollection(segs, colors=C["lidar"], linewidths=0.35, alpha=ray_alpha, zorder=8))
    pts = scan.points
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=hit_size, c=C["lidar"], linewidths=0, zorder=8.5)


def draw_pedestrians(ax, circles, style: str = "report", pedestrians=None, visual_scale: float = 1.0):
    """Ve tac nhan dong (nguoi di bo, xe lan, y ta + hop thuoc) dang nhin tu tren xuong."""
    from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
    from matplotlib.transforms import Affine2D

    C = STYLES[style]
    arr = np.asarray(circles, dtype=float).reshape(-1, 3) if circles is not None else np.zeros((0, 3), dtype=float)
    ped_list = list(pedestrians) if pedestrians is not None else []

    circle_meta: list[tuple[float, str, int | None]] = []
    if len(ped_list) == len(arr):
        for p in ped_list:
            circle_meta.append((
                float(getattr(p, "heading", 0.0)),
                str(getattr(p, "kind", "visitor")),
                getattr(p, "ped_id", None),
            ))
    else:
        for _ in range(len(arr)):
            circle_meta.append((0.0, "visitor", None))

    body_fill = C["ped"]
    body_edge = _darker(body_fill, 0.55)
    head_fill = "#FFFBEB" if style == "blueprint" else "#FFFFFF"
    text_color = "#0F172A"

    for idx, (cx, cy, rad) in enumerate(arr):
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(rad)) or rad <= 0.0:
            continue
        heading, kind, ped_id = circle_meta[idx]
        r_vis = float(rad) * float(visual_scale)
        T = Affine2D().rotate(heading).translate(float(cx), float(cy)) + ax.transData

        def _add_ped(patch, dz=0.0):
            patch.set_transform(T)
            patch.set_zorder(11.0 + dz)
            ax.add_patch(patch)

        # Vong ranh gioi va cham nhe de doi chieu kich thuoc thuc te
        _add_ped(
            Circle((0.0, 0.0), r_vis, facecolor=body_fill, edgecolor=body_edge, linewidth=1.0, alpha=0.32),
            dz=0.0,
        )

        # Than/vai nguoi (elip hoi det theo chieu di chuyen + duong vien ao/vai)
        shoulder_w = 1.55 * r_vis
        chest_h = 0.88 * r_vis
        _add_ped(
            Ellipse((0.0, 0.0), width=chest_h, height=shoulder_w,
                    facecolor=body_fill, edgecolor=body_edge, linewidth=1.3, alpha=0.95),
            dz=0.1,
        )
        _add_ped(
            Ellipse((-0.06 * r_vis, 0.0), width=0.38 * r_vis, height=1.15 * r_vis,
                    facecolor="none", edgecolor=body_edge, linewidth=0.75, alpha=0.55),
            dz=0.15,
        )

        # Phan dau hoi lech ve phia truoc (+x theo huong di chuyen) + vach chi huong
        head_r = max(0.06, 0.34 * r_vis)
        _add_ped(
            Circle((0.26 * r_vis, 0.0), head_r, facecolor=head_fill, edgecolor=body_edge, linewidth=1.05, alpha=0.98),
            dz=0.25,
        )
        _add_ped(
            Rectangle((0.26 * r_vis, -0.015 * r_vis), 0.42 * r_vis, 0.03 * r_vis,
                      facecolor=body_edge, edgecolor="none", alpha=0.85),
            dz=0.28,
        )

        # Chi tiet rieng cho xe lan va y ta + xe thuoc
        if kind == "wheelchair":
            for side in (-1.0, 1.0):
                _add_ped(
                    Rectangle((-0.45 * r_vis, side * 0.72 * r_vis - 0.04 * r_vis),
                              0.90 * r_vis, 0.08 * r_vis,
                              facecolor="#1E293B", edgecolor="#E2E8F0", linewidth=0.6, alpha=0.9),
                    dz=0.2,
                )
        elif kind == "nurse_cart":
            _add_ped(
                FancyBboxPatch(
                    (0.55 * r_vis, -0.32 * r_vis),
                    0.48 * r_vis,
                    0.64 * r_vis,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor="#38BDF8",
                    edgecolor="#0F172A",
                    linewidth=0.8,
                    alpha=0.9,
                ),
                dz=0.2,
            )

        tag = "P" if ped_id is None else f"P{ped_id}"
        ax.text(
            float(cx),
            float(cy) - 0.02 * r_vis,
            tag,
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=text_color,
            zorder=11.6,
        )


def draw_trajectory(ax, xy, speed=None, t=None, tick_every: float = 5.0, style: str = "report", lw: float = 2.6,
                    vmax: float | None = None, colorbar: bool = True, color: str | None = None, tick_side: int = -1):
    """Ve quy dao robot.

    speed     : neu co -> to mau theo van toc (kem thanh mau, dai mau tranh vang nhat de khong chim tren nen sang).
    t         : neu co -> cham moc thoi gian moi ``tick_every`` giay, nhan co nen de doc duoc tren tuong/cua.
    tick_side : -1 = nhan moc nam ben PHAI huong di (mac dinh), +1 = ben TRAI. Doi dau neu nhan roi vao tuong.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.ticker import FuncFormatter

    C = STYLES[style]
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 0:
        return
    xy = xy.reshape(-1, 2)

    edge, edge_a = (C["wall"], 0.55) if style == "report" else (C["axes_bg"], 0.85)
    ax.plot(xy[:, 0], xy[:, 1], "-", color=edge, lw=lw + 2.0, alpha=edge_a, zorder=7, solid_capstyle="round")
    if speed is not None and len(xy) > 1:
        sp = np.asarray(speed, dtype=float)
        top = 0.88 if style == "report" else 1.0
        cmap = ListedColormap(plt.get_cmap(C["track_cmap"])(np.linspace(0.0, top, 256)))
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=Normalize(0.0, vmax if vmax else max(float(sp.max()), 1e-6)),
                            linewidths=lw, zorder=7.5, capstyle="round")
        lc.set_array(0.5 * (sp[:-1] + sp[1:]))
        ax.add_collection(lc)
        if colorbar:
            cb = ax.figure.colorbar(lc, ax=ax, fraction=0.022, pad=0.012, shrink=0.75)
            cb.set_label("v [m/s]", color=C["room_text"], fontsize=8)
            cb.ax.tick_params(colors=C["room_text"], labelsize=7)
            cb.ax.yaxis.set_major_formatter(FuncFormatter(_vn_tick))
            cb.outline.set_edgecolor(C["spine"])
    else:
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color or C["robot_accent"], lw=lw, zorder=7.5, solid_capstyle="round")
    if t is not None and len(np.asarray(t)) > 1:
        t = np.asarray(t, dtype=float)
        for tk in np.arange(tick_every, t[-1] + 1e-9, tick_every):
            i = int(np.argmin(np.abs(t - tk)))
            d = xy[min(i + 1, len(xy) - 1)] - xy[max(i - 1, 0)]
            u = d / (np.hypot(*d) + 1e-12)
            nx, ny = u[1], -u[0]                                    # phap tuyen ben phai huong di
            px, py = xy[i, 0] + (-tick_side) * 0.55 * nx, xy[i, 1] + (-tick_side) * 0.55 * ny
            ax.scatter([xy[i, 0]], [xy[i, 1]], s=26, facecolor="#FFFFFF", edgecolor=C["text"], linewidths=1.0,
                       zorder=8.2)
            ax.text(px, py, f"{tk:.0f} s", ha="center", va="center", fontsize=6.5, color=C["text"], zorder=8.3,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=C["axes_bg"], edgecolor="none", alpha=0.85))


# --------------------------------------------------------------------------------------------
# Ban do
# --------------------------------------------------------------------------------------------
def draw_map(m, ax=None, show_inflated: bool | str = False, title: str | None = None, style: str = "report",
             show_doors: bool = True, show_dims: bool = True, show_labels: bool = True,
             legend: bool = True, ascii_labels: bool = False, subtitle: str | None = None,
             set_figure_bg: bool | None = None, legend_extra=None, target: str = "report",
             show_scalebar: bool = True, corridor_label: bool = True, legend_ncol: int = 4,
             show_door_tags: bool = True, show_north: bool = True, show_room_codes: bool = True,
             title_block: tuple | None = None):
    """Ve HospitalMap ``m`` len ``ax`` (tao figure moi neu ax=None). Tra ve ``ax``.

    show_inflated : False (mac dinh: hinh sach) | True / 'both' | 'fill' | 'outline'  -- vung cam tam robot.
                    Chi bat cho hinh minh hoa quy hoach A* / giai thich phong vat can.
    set_figure_bg : None = chi doi mau nen figure khi draw_map tu tao figure.
    legend_extra  : them muc vao chu thich (vd handle cua robot, quy dao).
    target        : "report" | "slide" -- kich thuoc figure (khi tu tao) va he so chu.
    subtitle      : None = dong mo ta luoi + phong vat can; "" = khong ve dong phu de.
    corridor_label: False de bo nhan HANH LANG CHINH (tranh de len quy dao).
    show_door_tags: ma so cua D1..Dn canh moi cua (de bao cao tham chieu "robot qua cua D3").
    show_room_codes: ma phong (P.101...) duoi ten phong.   show_north: mui ten huong Bac (+y).
    title_block   : (ten do an, ma ban ve, ghi chu) -> khung ten ban ve o goc duoi (chi khi draw_map tu tao figure).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Arc, FancyBboxPatch, Rectangle

    if style not in STYLES:
        raise ValueError(f"style phai la mot trong {list(STYLES)}")
    C = STYLES[style]
    P = _target(target)
    fs = P["fs"]
    tr = _strip if ascii_labels else (lambda s: s)

    if ax is not None:                                           # canh bao khi tron hai style trong mot figure
        prev = getattr(ax.figure, "_map_view_style", None)
        if prev is not None and prev != style:
            import warnings
            warnings.warn(f"draw_map: figure nay da ve bang style '{prev}', nay lai ve '{style}'. Mau chu va mau nen "
                          f"se xung dot (chu khong doc duoc). Dung cung mot style, hoac dung draw_pair().",
                          stacklevel=2)

    if ax is None:
        fig, ax = plt.subplots(figsize=P["figsize"], dpi=110)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=(0.25 if title_block else 0.2) if legend else 0.1)
        set_figure_bg = True if set_figure_bg is None else set_figure_bg
    if set_figure_bg:
        ax.figure.patch.set_facecolor(C["figure_bg"])
    ax.figure._map_view_style = style
    ax.set_facecolor(C["axes_bg"])

    W, H, res = m.width, m.height, m.res
    wt = getattr(m, "wall_thickness", 0.2)
    dw = getattr(m, "door_width", 1.2)
    ys, yn = m.corridor_y

    # 1. san phong
    for room in m.rooms:
        if room.rect is not None:
            x1, y1, x2, y2 = room.rect
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=C["floor"].get(room.kind, C["floor"]["room"]),
                                   edgecolor="none", zorder=0))

    # 2. luoi
    ax.set_xticks(np.arange(0, W + 0.1, 5))
    ax.set_yticks(np.arange(0, H + 0.1, 5))
    ax.set_xticks(np.arange(0, W + 0.1, 1), minor=True)
    ax.set_yticks(np.arange(0, H + 0.1, 1), minor=True)
    ax.grid(which="minor", color=C["grid"], lw=0.3, alpha=0.4)
    ax.grid(which="major", color=C["grid"], lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", colors=C["room_text"], labelsize=8 * fs, length=3)
    _vn_axes(ax)

    # 3. vung cam tam robot
    mode = "both" if show_inflated is True else show_inflated
    if mode:
        xs, yv = (np.arange(m.n_cols) + 0.5) * res, (np.arange(m.n_rows) + 0.5) * res
        z = m.inflated.astype(float)
        if mode in ("both", "fill"):
            ax.contourf(xs, yv, z, levels=[0.5, 1.5], colors=[C["inflate"]], alpha=0.07, zorder=1)
        if mode in ("both", "outline"):
            ax.contour(xs, yv, z, levels=[0.5], colors=[C["inflate"]], linewidths=0.7, linestyles="--", alpha=0.75,
                       zorder=6)

    # 4. tuong
    walls = [Rectangle((x, y), w, h) for x, y, w, h in _mask_to_rects(wall_mask(m), res)]
    ax.add_collection(PatchCollection(walls, facecolor=C["wall"], edgecolor=C["wall"], linewidths=0.2, zorder=4))

    # 5. noi that
    fdef = C["furniture"]["furniture"]
    for kind, (x1, y1, x2, y2) in getattr(m, "furniture", []):
        face = C["furniture"].get(kind, fdef)
        ax.add_patch(FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=face, edgecolor=_darker(face), linewidth=0.7, zorder=3))
        _furniture_detail(ax, kind, (x1, y1, x2, y2), face, fs)
        if show_labels and kind in FURNITURE_NAMES and min(x2 - x1, y2 - y1) >= 1.1:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, tr(FURNITURE_NAMES[kind]), ha="center", va="center",
                    fontsize=6.5 * fs, color="#FFFFFF" if style == "blueprint" else C["text"], zorder=3.5)

    # 6. vat can chua biet
    for (x1, y1, x2, y2) in getattr(m, "unmapped", []):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=C["unmapped_fill"], edgecolor=C["unmapped_edge"],
                               hatch="////", linewidth=1.0, zorder=3.2))

    # 7. cua
    if show_doors:
        for (xd, yd) in m.doors:
            up = yd > H / 2
            y_edge = yd + (wt / 2 if up else -wt / 2)
            hx, sgn = xd - dw / 2, (1 if up else -1)
            ax.plot([hx, hx], [y_edge, y_edge + sgn * dw], color=C["door"], lw=1.0, zorder=5)
            ax.add_patch(Arc((hx, y_edge), 2 * dw, 2 * dw, theta1=0 if up else -90, theta2=90 if up else 0,
                             color=C["door"], lw=0.7, linestyle=":", zorder=5))

    if show_doors and show_door_tags:
        for i, (xd, yd) in enumerate(m.doors, 1):
            ax.text(xd + dw / 2 + 0.55, yd, f"D{i}", ha="center", va="center", fontsize=5.8 * fs, fontweight="bold",
                    color=C["text"], zorder=5.5,
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor=C["axes_bg"], edgecolor=C["door"], linewidth=0.7))

    # 8. kich thuoc
    if show_dims:
        dim_kw = dict(arrowstyle="<->", color=C["dim"], lw=0.8, shrinkA=0, shrinkB=0)
        xc = W - 0.85
        ax.annotate("", xy=(xc, ys), xytext=(xc, yn), arrowprops=dim_kw, zorder=7)
        ax.text(xc - 0.25, (ys + yn) / 2, f"{_vn(yn - ys)} m", rotation=90, ha="right", va="center",
                fontsize=7.5 * fs, color=C["dim"], zorder=7, fontweight="bold")
        low = [d for d in m.doors if d[1] < H / 2]
        if low:
            xd = low[-1][0]
            yd_dim = ys + 0.35
            ax.annotate("", xy=(xd - dw / 2, yd_dim), xytext=(xd + dw / 2, yd_dim), arrowprops=dim_kw, zorder=7)
            ax.text(xd, yd_dim + 0.22, f"{_vn(dw)} m", ha="center", va="bottom", fontsize=7.5 * fs, color=C["dim"],
                    zorder=7, fontweight="bold")

    # 9. nhan phong
    if show_labels:
        for room in m.rooms:
            if room.rect is None or (room.name == "Hanh lang chinh" and not corridor_label):
                continue
            x1, y1, x2, y2 = room.rect
            lx, ly = ROOM_LABEL_POS.get(room.name, ((x1 + x2) / 2, (y1 + y2) / 2))
            ax.text(lx, ly, tr(ROOM_NAMES.get(room.name, room.name)), ha="center", va="center", fontsize=8 * fs,
                    color=C["room_text"], fontweight="bold", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=C["axes_bg"], edgecolor=C["spine"], alpha=0.8))
            if show_room_codes and room.name in ROOM_CODES:
                ax.text(lx, ly + ROOM_CODE_DY.get(room.name, -0.72), ROOM_CODES[room.name], ha="center", va="center", fontsize=6.3 * fs,
                        color=C["room_text"], alpha=0.85, zorder=3)

    # 10. Start va G1-G4
    ms = 170 * fs * fs
    sx, sy = m.start
    ax.scatter([sx], [sy], s=ms, c=C["start"], edgecolors=C["marker_edge"], linewidths=1.5, zorder=9)
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=7.5 * fs, fontweight="bold", color="#FFFFFF", zorder=10)
    for name, (gx, gy) in m.goals.items():
        ax.scatter([gx], [gy], s=ms, c=C["goal"], edgecolors=C["marker_edge"], linewidths=1.5, zorder=9)
        ax.text(gx, gy, name, ha="center", va="center", fontsize=6.8 * fs, fontweight="bold", color="#FFFFFF", zorder=10)

    # 10b. thuoc ti le 5 m (trong phong Thang may / Kho, noi trong)
    if show_scalebar:
        x0, y0 = W - 8.0, 1.4
        ax.plot([x0, x0 + 5.0], [y0, y0], color=C["text"], lw=2.2, zorder=7, solid_capstyle="butt")
        for xx in (x0, x0 + 2.5, x0 + 5.0):
            ax.plot([xx, xx], [y0 - 0.18, y0 + 0.18], color=C["text"], lw=1.0, zorder=7)
        ax.text(x0, y0 + 0.35, "0", ha="center", va="bottom", fontsize=7 * fs, color=C["text"], zorder=7)
        ax.text(x0 + 5.0, y0 + 0.35, "5 m", ha="center", va="bottom", fontsize=7 * fs, color=C["text"], zorder=7)

    if show_north:
        nx_, ny_ = W - 1.3, 2.9
        ax.annotate("", xy=(nx_, ny_ + 1.6), xytext=(nx_, ny_), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=C["text"], lw=1.3, shrinkA=0, shrinkB=0))
        ax.text(nx_, ny_ + 1.85, "N", ha="center", va="bottom", fontsize=8 * fs, fontweight="bold", color=C["text"],
                zorder=7)

    # 11. khung, tieu de
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel(tr("x [m]"), color=C["room_text"], fontsize=9 * fs)
    ax.set_ylabel(tr("y [m]"), color=C["room_text"], fontsize=9 * fs)
    for sp in ax.spines.values():
        sp.set_color(C["spine"])
        sp.set_linewidth(0.8)
    if title:
        ax.set_title(tr(title), loc="left", fontsize=11.5 * fs, fontweight="bold", color=C["text"],
                     pad=(26 if subtitle != "" else 8) * fs)
        if subtitle is None:
            subtitle = _grid_text(m)
        if subtitle:
            ax.text(0.0, 1.012, tr(subtitle), transform=ax.transAxes, fontsize=7.5 * fs, color=C["subtitle"],
                    va="bottom")

    # 11b. khung ten ban ve
    if title_block and ax.figure is not None and getattr(ax.figure, "_map_view_style", None) == style:
        from matplotlib.patches import Rectangle as _R
        tb = ax.figure.add_axes([0.40, 0.006, 0.585, 0.052])
        tb.set_xlim(0, 3)
        tb.set_ylim(0, 1)
        tb.set_xticks([])
        tb.set_yticks([])
        tb.set_facecolor(C["axes_bg"])
        for sp in tb.spines.values():
            sp.set_color(C["spine"])
        for xv, (txt, wt_) in zip((0.08, 1.58, 2.32), zip(title_block, ("bold", "bold", "normal"))):   # 3 o: ten do an | ma ban ve | ghi chu ngan
            tb.text(xv, 0.5, tr(txt), va="center", ha="left", fontsize=7.5 * fs, color=C["text"], fontweight=wt_)
        for xv in (1.5, 2.25):
            tb.plot([xv, xv], [0, 1], color=C["spine"], lw=0.8)

    # 12. chu thich (chi liet ke nhung gi THUC SU co tren hinh)
    if legend:
        h = _legend_handles(m, C, tr, bool(mode), legend_extra)
        leg = ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(len(h), legend_ncol),
                        frameon=False, fontsize=7.5 * fs, columnspacing=1.6, handlelength=1.8)
        for t_ in leg.get_texts():
            t_.set_color(C["text"])
    return ax


def draw_door_zoom(m, door_idx: int = 0, style: str = "report", half_size: float = 2.3):
    """Hinh phong to tai mot cua: robot DUNG KICH THUOC o giua cua, vong phong vat can, khoang ho moi ben."""
    import matplotlib.pyplot as plt

    C = STYLES[style]
    xd, yd = m.doors[door_idx]
    up = yd > m.height / 2
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
    fig.patch.set_facecolor(C["figure_bg"])
    draw_map(m, ax, show_inflated="outline", style=style, show_labels=False, show_dims=False, legend=False,
             show_scalebar=False, set_figure_bg=True)
    theta = math.pi / 2 if up else -math.pi / 2
    ry = yd
    r, rr = float(config.ROBOT_RADIUS), m.inflate_cells * m.res          # rr = ban kinh phong THUC TE tren luoi
    draw_robot(ax, xd, ry, theta, style, ring_radius=rr, zorder=12)
    # khoang ho moi ben giua than robot va khung cua
    gap = m.door_width / 2 - r
    y_dim = ry + (0.62 if up else -0.62)
    kw = dict(arrowstyle="<->", color=C["dim"], lw=1.0, shrinkA=0, shrinkB=0)
    for sgn in (-1, +1):
        ax.annotate("", xy=(xd + sgn * r, y_dim), xytext=(xd + sgn * m.door_width / 2, y_dim), arrowprops=kw, zorder=13)
        ax.text(xd + sgn * (r + gap / 2), y_dim + (0.12 if up else -0.12), f"{_vn(gap * 100, 0)} cm", ha="center",
                va="bottom" if up else "top", fontsize=8, color=C["dim"], fontweight="bold", zorder=13)
    y_door = ry - (0.7 if up else -0.7)
    ax.annotate("", xy=(xd - m.door_width / 2, y_door), xytext=(xd + m.door_width / 2, y_door), arrowprops=kw, zorder=13)
    ax.text(xd, y_door - (0.12 if up else -0.12), f"cửa {_vn(m.door_width * 100, 0)} cm", ha="center",
            va="top" if up else "bottom", fontsize=8, color=C["dim"], fontweight="bold", zorder=13)
    # dai tam robot duoc phep (sau khi phong): quyet dinh robot co qua duoc cua hay khong
    band = m.door_width - 2.0 * rr
    if band > 0:
        y_band = ry - (1.15 if up else -1.15)                 # phia hanh lang, khong de len cung mo cua
        kw2 = dict(arrowstyle="<->", color=C["inflate"], lw=1.2, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(xd - band / 2, y_band), xytext=(xd + band / 2, y_band), arrowprops=kw2, zorder=13)
        ax.text(xd, y_band - (0.12 if up else -0.12), f"dải tâm cho phép {_vn(band * 100, 0)} cm", ha="center",
                va="top" if up else "bottom", fontsize=8, color=C["inflate"], fontweight="bold", zorder=13)
    ax.set_xlim(xd - half_size, xd + half_size)
    ax.set_ylim(ry - half_size, ry + half_size)
    for t_ in ax.texts:                       # nhan nam ngoai khung phong to (S, G1..G4...) khong duoc lam phinh hinh khi cat le
        t_.set_clip_on(True)
    ax.set_xticks(np.arange(math.ceil((xd - half_size) * 2) / 2, xd + half_size, 0.5))
    ax.set_yticks(np.arange(math.ceil((ry - half_size) * 2) / 2, ry + half_size, 0.5))
    _vn_axes(ax)
    ax.set_title(f"Chi tiết cửa: robot r = {_vn(r * 100, 0)} cm đi qua cửa {_vn(m.door_width * 100, 0)} cm",
                 loc="left", fontsize=10.5, fontweight="bold", color=C["text"])
    ax.text(0.0, -0.09, f"Vòng đứt đỏ: vùng cấm tâm robot = {m.inflate_cells} ô × {_vn(m.res * 100, 0)} cm = "
                        f"{_vn(rr * 100, 0)} cm (robot {_vn(r * 100, 0)} cm + biên an toàn, làm tròn lên theo ô lưới).",
            transform=ax.transAxes, fontsize=7.5, color=C["subtitle"], va="top")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.15)
    return fig


def draw_pair(known, world, style: str = "report", target: str = "report", **kwargs):
    """Ban do DA BIET va THE GIOI THAT xep doc, dung CUNG mot style va MOT chu thich chung."""
    import matplotlib.pyplot as plt

    C = STYLES[style]
    P = _target(target)
    tr = _strip if kwargs.get("ascii_labels") else (lambda s: s)
    fig, axes = plt.subplots(2, 1, figsize=P["pair_figsize"], dpi=110)
    fig.patch.set_facecolor(C["figure_bg"])
    kw = dict(style=style, legend=False, target=target, show_scalebar=False, **kwargs)
    draw_map(known, axes[0], title="Bản đồ ĐÃ BIẾT — A* lập đường trên bản đồ này", **kw)
    n_unk = len(getattr(world, "unmapped", []))
    draw_map(world, axes[1], title="Thế giới THẬT — LiDAR đo và kiểm tra va chạm trên bản đồ này",
             subtitle=f"như bản đồ đã biết, cộng {n_unk} vật cản chưa biết mà A* không thấy", **kw)
    handles = _legend_handles(world, C, tr, bool(kwargs.get("show_inflated", False)))
    leg = fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False,
                     fontsize=8 * P["fs"], columnspacing=1.5, handlelength=1.8)
    for t_ in leg.get_texts():
        t_.set_color(C["text"])
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.30, left=0.07, right=0.985)
    return fig


def save_map_figures(known, world=None, outdir: str = "figures", dpi: int = 300) -> list:
    """Xuat hinh cho bao cao: PNG (dpi cao) + PDF vector, ca hai style. Tra ve danh sach file da ghi."""
    import matplotlib.pyplot as plt

    written = []
    for style in STYLES:
        fig = draw_map(known, None, title="Bản đồ tầng bệnh viện (40 m × 20 m)", style=style).figure
        for ext in ("png", "pdf"):
            written.append(save_figure(fig, os.path.join(outdir, f"hospital_map_{style}.{ext}"), dpi))
        plt.close(fig)
        if world is not None:
            fig = draw_pair(known, world, style=style)
            for ext in ("png", "pdf"):
                written.append(save_figure(fig, os.path.join(outdir, f"hospital_map_known_vs_world_{style}.{ext}"), dpi))
            plt.close(fig)
    return written


if __name__ == "__main__":
    # Chay:  python -m env.map_view     (tu thu muc goc cua du an)  -> ghi hinh vao figures/
    from env.hospital_map import UNMAPPED_CARTS, HospitalMap

    known = HospitalMap()
    world = known.with_obstacles(UNMAPPED_CARTS)
    for f in save_map_figures(known, world, "figures"):
        print("da ghi:", f)