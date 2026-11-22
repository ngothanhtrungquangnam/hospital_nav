"""tests/test_map_view.py -- Kiem chung env/map_view.py (hinh hoc, chu thich, robot, dinh dang, xuat file).

Chay:  python -m tests.test_map_view      (tu thu muc goc cua du an)
Hoac:  pytest tests/test_map_view.py
"""
from __future__ import annotations

import math
import os
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")                                   # khong can man hinh
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib.patches import Circle                   # noqa: E402

import config                                           # noqa: E402
from env import map_view as mv                          # noqa: E402
from env.hospital_map import UNMAPPED_CARTS, HospitalMap  # noqa: E402
from sensors.lidar import Lidar2D                       # noqa: E402

KNOWN = HospitalMap()
WORLD = KNOWN.with_obstacles(UNMAPPED_CARTS)


def _free_run(line: np.ndarray) -> int:
    """Do dai (so o) cua day o trong lien tiep dai nhat cua mot vecto bool (True = vat can)."""
    best = cur = 0
    for blocked in line:
        cur = 0 if blocked else cur + 1
        best = max(best, cur)
    return best


# ---------------------------------------------------------------- hinh hoc ban do (nen tang cua hinh ve)
def test_grid_shape_and_inflation_cells():
    assert KNOWN.grid.shape == (200, 400)
    assert KNOWN.inflate_cells == 4                                     # 0,25 + 0,10 = 0,35 -> 4 o = 0,40 m


def test_door_is_12_cells_and_center_band_is_4_cells():
    """Cua 1,2 m = 12 o; sau khi phong 4 o moi ben con dai tam 4 o = 0,40 m -> robot qua duoc."""
    xd, yd = KNOWN.doors[1]                                              # cua Cap cuu (x = 6)
    r, c0 = KNOWN.xy_to_rc(xd, yd)
    row = KNOWN.grid[r]
    c_mid = c0
    left = c_mid
    while left > 0 and not row[left - 1]:
        left -= 1
    right = c_mid
    while right < KNOWN.n_cols - 1 and not row[right + 1]:
        right += 1
    assert right - left + 1 == 12
    assert _free_run(KNOWN.inflated[r]) >= 4
    band = KNOWN.inflated[r, left:right + 1]
    assert int((~band).sum()) == 4


def test_corridor_is_28_cells_and_center_band_is_20_cells():
    c = KNOWN.xy_to_rc(20.0, 10.0)[1]
    # cot x = 12,5 m nam trong doan thong voi Sanh cho -> dung cot x = 5 m (tuong hang phong hai ben)
    c = KNOWN.xy_to_rc(8.0, 10.0)[1]
    col = KNOWN.grid[:, c]
    ys, yn = KNOWN.corridor_y
    r1, r2 = KNOWN.xy_to_rc(8.0, ys + 0.05)[0], KNOWN.xy_to_rc(8.0, yn - 0.05)[0]
    assert not col[r1:r2 + 1].any()
    assert (r2 + 1 - r1) >= 27                                          # ~28 o (2,8 m)
    assert int((~KNOWN.inflated[r1:r2 + 1, c]).sum()) >= 19            # dai tam ~2,0 m


def test_wall_furniture_unmapped_layers_are_disjoint_and_cover_grid():
    wm = mv.wall_mask(WORLD)
    furn = mv._rasterize([r for _, r in WORLD.furniture], WORLD.grid.shape, WORLD.res)
    unk = mv._rasterize(WORLD.unmapped, WORLD.grid.shape, WORLD.res)
    assert not (wm & furn).any() and not (wm & unk).any()
    assert (wm | furn | unk).sum() == WORLD.grid.sum()                  # ba lop hop lai = toan bo vat can
    assert unk.sum() == sum(
        (round(x2 / WORLD.res) - round(x1 / WORLD.res)) * (round(y2 / WORLD.res) - round(y1 / WORLD.res))
        for x1, y1, x2, y2 in UNMAPPED_CARTS)


def test_mask_to_rects_is_lossless():
    wm = mv.wall_mask(KNOWN)
    rects = mv._mask_to_rects(wm, KNOWN.res)
    back = mv._rasterize([(x, y, x + w, y + h) for x, y, w, h in rects], wm.shape, KNOWN.res)
    assert (back == wm).all()


# ---------------------------------------------------------------- chu thich va mau
def test_all_furniture_kinds_have_color_in_both_styles():
    kinds = {k for k, _ in KNOWN.furniture}
    for style, C in mv.STYLES.items():
        for k in kinds:
            assert k in C["furniture"], f"style {style} thieu mau cho loai '{k}'"
    assert kinds <= set(mv.FURNITURE_LEGEND)


def test_legend_lists_exactly_what_is_on_the_map():
    for style in mv.STYLES:
        for m, has_unk in ((KNOWN, False), (WORLD, True)):
            ax = mv.draw_map(m, None, title="t", style=style)
            texts = [t.get_text() for t in ax.get_legend().get_texts()]
            for k in {k for k, _ in m.furniture}:
                assert mv.FURNITURE_LEGEND[k] in texts
            assert "Tủ" not in texts and "Thiết bị cố định" not in texts   # loai khong co tren ban do -> khong liet ke
            assert "Giường bệnh" in texts and "Bàn xét nghiệm" in texts
            assert ("Vật cản chưa biết (chỉ LiDAR thấy)" in texts) == has_unk
            plt.close(ax.figure)


def test_legend_swatch_color_matches_drawn_color():
    ax = mv.draw_map(KNOWN, None, title="t", style="blueprint")
    C = mv.STYLES["blueprint"]
    by_text = {t.get_text(): h for t, h in zip(ax.get_legend().get_texts(), ax.get_legend().legend_handles)}
    h = by_text[mv.FURNITURE_LEGEND["bed"]]
    assert matplotlib.colors.to_hex(h.get_facecolor()) == C["furniture"]["bed"].lower()
    plt.close(ax.figure)


def test_furniture_reclassification_keeps_occupancy_grid():
    """Doi 'kind' (furniture -> bed/lab) chi doi cach ve; luoi chiem dung phai giu nguyen de test simulator con dung."""
    import hashlib
    h = hashlib.md5(np.packbits(KNOWN.grid).tobytes()).hexdigest()
    assert h == "12cb6fac5e88eac5454896ddf2be0ed8" and int(KNOWN.grid.sum()) == 8828


def test_door_tags_room_codes_north_and_title_block():
    ax = mv.draw_map(KNOWN, None, title="t", title_block=("Do an", "MB-01", "Don vi: m"))
    texts = [t.get_text() for t in ax.texts]
    for i in range(1, len(KNOWN.doors) + 1):
        assert f"D{i}" in texts
    for room in KNOWN.rooms:
        assert mv.ROOM_CODES[room.name] in texts
    assert "N" in texts
    assert len(ax.figure.axes) == 2                                     # truc ban do + khung ten
    ax2 = mv.draw_map(KNOWN, None, title="t", show_door_tags=False, show_north=False, show_room_codes=False)
    texts2 = [t.get_text() for t in ax2.texts]
    assert "D1" not in texts2 and "N" not in texts2 and "P.101" not in texts2
    plt.close("all")


def test_furniture_detail_stays_inside_its_rectangle():
    """Bieu tuong ben trong noi that khong duoc tran ra ngoai hinh chu nhat (vung chiem dung khong doi)."""
    for kind, rect in KNOWN.furniture:
        fig, ax = plt.subplots()
        n0 = len(ax.patches) + len(ax.lines)
        mv._furniture_detail(ax, kind, rect, "#999999", 1.0)
        x1, y1, x2, y2 = rect
        for ln in ax.lines:
            xs, ys = ln.get_xdata(), ln.get_ydata()
            assert min(xs) >= x1 - 1e-9 and max(xs) <= x2 + 1e-9 and min(ys) >= y1 - 1e-9 and max(ys) <= y2 + 1e-9
        for p in ax.patches:
            bb = p.get_path().get_extents().transformed(p.get_patch_transform())
            assert bb.x0 >= x1 - 1e-6 and bb.x1 <= x2 + 1e-6 and bb.y0 >= y1 - 1e-6 and bb.y1 <= y2 + 1e-6
        plt.close(fig)


def test_subtitle_inflation_text_uses_config_values():
    s = mv._grid_text(KNOWN)
    assert "0,25 m (robot)" in s and "0,10 m (biên an toàn)" in s and "0,35 m" in s and "4 ô = 0,40 m" in s
    assert "." not in s.replace("...", "")                              # dau phay thap phan


def test_vn_number_formatting():
    assert mv._vn(2.8) == "2,8" and mv._vn(0.4, 2) == "0,40"
    assert mv._vn_tick(13.0) == "13" and mv._vn_tick(13.5) == "13,5"


# ---------------------------------------------------------------- ve chung
def test_draw_map_runs_all_options():
    for style in mv.STYLES:
        for kw in ({}, {"show_inflated": True}, {"show_inflated": "fill"}, {"show_inflated": "outline"},
                   {"show_labels": False, "show_dims": False, "show_doors": False, "legend": False},
                   {"ascii_labels": True, "corridor_label": False, "target": "slide"}):
            ax = mv.draw_map(WORLD, None, title="t", style=style, **kw)
            plt.close(ax.figure)


def test_axes_tick_labels_use_decimal_comma():
    fig = mv.draw_door_zoom(KNOWN, 0)
    ax = fig.axes[0]
    fig.canvas.draw()
    labels = [t.get_text() for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) if t.get_text()]
    assert labels and not any("." in s for s in labels)
    assert any("," in s for s in labels)
    plt.close(fig)


def test_style_mixing_warns():
    fig, ax = plt.subplots()
    mv.draw_map(KNOWN, ax, style="report")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mv.draw_map(KNOWN, ax, style="blueprint")
    assert any("style" in str(x.message) for x in w)
    plt.close(fig)


def test_invalid_style_and_target_raise():
    for bad in ({"style": "x"}, {"target": "x"}):
        try:
            mv.draw_map(KNOWN, None, **bad)
        except ValueError:
            continue
        raise AssertionError("phai bao ValueError")


def test_map_without_lower_doors_does_not_crash():
    m = HospitalMap()
    m.doors = [d for d in m.doors if d[1] > m.height / 2]
    ax = mv.draw_map(m, None, title="t")
    plt.close(ax.figure)


# ---------------------------------------------------------------- robot, scan, nguoi, quy dao
def test_robot_is_drawn_with_model_radius_and_wheels_fit_inside():
    fig, ax = plt.subplots()
    mv.draw_robot(ax, 5.0, 5.0, 0.3)
    radii = [p.get_radius() for p in ax.patches if isinstance(p, Circle)]
    assert math.isclose(max(radii), config.ROBOT_RADIUS)
    wheel_corner = math.hypot(config.WHEEL_RADIUS, config.WHEEL_BASE / 2 + 0.02)
    assert wheel_corner < config.ROBOT_RADIUS                           # banh nam tron trong than
    plt.close(fig)


def test_robot_lidar_dot_is_at_center():
    fig, ax = plt.subplots()
    mv.draw_robot(ax, 5.0, 5.0, 0.0)
    small = [p for p in ax.patches if isinstance(p, Circle) and p.get_radius() < 0.06]
    assert small and all(p.center == (0, 0) for p in small)             # tam trong he robot -> trung tam than
    plt.close(fig)


def test_scan_pedestrian_trajectory_draw_with_real_lidar():
    fig = mv.draw_map(WORLD, None, title="t").figure
    ax = fig.axes[0]
    ped = np.array([[19.0, 10.0, 0.25]])
    scan = Lidar2D().scan(WORLD, [17.0, 9.6, 0.0], circles=ped)
    mv.draw_scan(ax, scan)
    mv.draw_pedestrians(ax, ped)
    mv.draw_robot(ax, 17.0, 9.6, 0.0, scale=2.0, lidar_range=5.0, fov_deg=90.0)
    t = np.linspace(0, 12, 60)
    xy = np.column_stack([5 + t, np.full_like(t, 10.0)])
    mv.draw_trajectory(ax, xy, speed=np.full_like(t, 0.8), t=t)
    mv.draw_trajectory(ax, xy, t=None)
    mv.draw_pedestrians(ax, np.zeros((0, 3)))
    plt.close(fig)


def test_draw_scan_handles_infinite_ranges():
    class S:                                                            # tia khong cham tra inf
        origin = np.array([5.0, 10.0, 0.7])                              # pose 3 phan tu van duoc chap nhan
        angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        ranges = np.array([1, np.inf, 2, np.inf, 1, 1, 1, 1.0])
        points = np.zeros((0, 2))
    fig, ax = plt.subplots()
    mv.draw_scan(ax, S())
    plt.close(fig)


# ---------------------------------------------------------------- hinh phong to cua, cap doi, xuat file
def test_door_zoom_and_pair_run():
    for style in mv.STYLES:
        for i in range(len(KNOWN.doors)):
            plt.close(mv.draw_door_zoom(KNOWN, i, style))
        plt.close(mv.draw_pair(KNOWN, WORLD, style))
        plt.close(mv.draw_pair(KNOWN, WORLD, style, target="slide"))


def test_door_zoom_texts_do_not_inflate_saved_figure():
    """Nhan nam ngoai khung phong to (S, G1..G4) khong duoc lam phinh anh khi cat le."""
    fig = mv.draw_door_zoom(KNOWN, 1)
    with tempfile.TemporaryDirectory() as d:
        p = mv.save_figure(fig, os.path.join(d, "z.png"), dpi=100)
        img = plt.imread(p)
    plt.close(fig)
    assert img.shape[1] < 1000 and img.shape[0] < 1000


def test_save_map_figures_writes_png_and_pdf():
    with tempfile.TemporaryDirectory() as d:
        files = mv.save_map_figures(KNOWN, WORLD, d, dpi=60)
        assert len(files) == 8
        for f in files:
            assert os.path.getsize(f) > 2000
        pdfs = [f for f in files if f.endswith(".pdf")]
        for f in pdfs:                                                  # font nhung duoc (khong dung Type 3)
            with open(f, "rb") as fh:
                assert b"/Type3" not in fh.read()


# ---------------------------------------------------------------- chay khong can pytest
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:                                          # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        plt.close("all")
    print(f"\n{len(tests) - failed}/{len(tests)} test pass")
    raise SystemExit(1 if failed else 0)