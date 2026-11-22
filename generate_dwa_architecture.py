"""generate_dwa_architecture.py -- Tao so do khoi kien truc he thong dieu huong phan tang Global A* va Local DWA."""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Cau hinh font chu tieng Viet ho tro day du
plt.rcParams["font.sans-serif"] = [
    "Segoe UI",
    "Arial",
    "DejaVu Sans",
    "Tahoma",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def draw_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[tuple[str, bool, str]],  # (text, is_bold, color)
    border_color: str,
    fill_color: str,
    title_color: str = "#111827",
    rounding: float = 0.10,
    align_left: bool = False,
    left_margin: float = 0.22,
):
    """Ve mot the khoi chu nhat bo goc hien dai voi typography sac net."""
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        edgecolor=border_color,
        facecolor=fill_color,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(box)

    # Tieu de khoi
    title_y = y + h - 0.28
    ax.text(
        x + w / 2.0,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )

    # Duong gach phan cach nhe duoi tieu de
    ax.plot(
        [x + 0.25, x + w - 0.25],
        [title_y - 0.18, title_y - 0.18],
        color=border_color,
        lw=0.8,
        alpha=0.45,
        zorder=3,
    )

    # Noi dung chi tiet ben trong the
    if lines:
        n_lines = len(lines)
        avail_h = h - 0.60
        dy = avail_h / max(n_lines, 1)
        start_y = title_y - 0.32
        for i, (text, is_bold, color) in enumerate(lines):
            line_y = start_y - i * dy
            weight = "bold" if is_bold else "normal"
            if align_left:
                ax.text(
                    x + left_margin,
                    line_y,
                    text,
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    fontweight=weight,
                    color=color,
                    zorder=3,
                )
            else:
                ax.text(
                    x + w / 2.0,
                    line_y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight=weight,
                    color=color,
                    zorder=3,
                )


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#1e293b",
    lw: float = 1.6,
    rad: float = 0.0,
    dashed: bool = False,
    label: str | None = None,
    label_pos: tuple[float, float] | None = None,
    label_color: str = "#1e293b",
    fontsize: float = 8.2,
):
    """Ve mui ten ket noi ro rang giua cac khoi."""
    arrowprops = dict(
        arrowstyle="-|>",
        color=color,
        lw=lw,
        mutation_scale=14,
        connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-",
        shrinkA=3,
        shrinkB=3,
    )
    ax.annotate("", xy=end, xytext=start, arrowprops=arrowprops, zorder=4)
    if label and label_pos:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="semibold",
            color=label_color,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, lw=0.6, alpha=0.95),
            zorder=5,
        )


def generate_diagram(out_paths: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 7.5), dpi=240)
    ax.set_xlim(0.0, 16.5)
    ax.set_ylim(0.0, 7.5)
    ax.axis("off")

    # Tieu de tong the
    ax.text(
        8.25,
        7.15,
        "Kiến trúc Hệ thống Điều hướng Phân tầng: Global A* & Local DWAPlanner Control Loop",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        8.25,
        6.85,
        "(Hospital Navigation System Architecture: Hierarchical Global A* Path Planning & Dynamic Window Approach)",
        ha="center",
        va="center",
        fontsize=9.2,
        fontstyle="italic",
        color="#475569",
    )

    # 1. COT 1: MOI TRUONG & TAC NHAN
    draw_card(
        ax,
        x=0.5,
        y=3.2,
        w=2.7,
        h=2.4,
        title="HospitalMap",
        lines=[
            ("Bản đồ bệnh viện 40×20 m", False, "#1e293b"),
            ("Lưới tĩnh (Static Grid Map)", False, "#1e293b"),
            ("Tường cố định, 6 phòng & 6 cửa", False, "#1e293b"),
            ("Vật cản đã biết & vùng lạm phát", False, "#1e293b"),
            ("Độ phân giải: 0.1 m / ô (400×200)", False, "#1e293b"),
        ],
        border_color="#1d4ed8",  # Blue
        fill_color="#eff6ff",
        title_color="#1e40af",
    )

    draw_card(
        ax,
        x=0.5,
        y=0.6,
        w=2.7,
        h=2.4,
        title="PedestrianManager",
        lines=[
            ("Tác nhân ngoại vi & Động", False, "#1e293b"),
            ("(Dynamic & Unmapped Agents)", False, "#475569"),
            ("Người đi bộ: 0.5 - 1.4 m/s", False, "#1e293b"),
            ("(Ngược chiều, cùng chiều, cắt ngang)", False, "#475569"),
            ("Xe đẩy lạ phát sinh chưa lập bản đồ", False, "#1e293b"),
            ("Mô hình chuyển động thực tế", False, "#1e293b"),
        ],
        border_color="#d97706",  # Amber/Gold
        fill_color="#fffbeb",
        title_color="#b45309",
    )

    # 2. COT 2: GLOBAL PLANNING & SENSING
    draw_card(
        ax,
        x=3.8,
        y=4.0,
        w=2.9,
        h=1.6,
        title="Global A* Planner",
        lines=[
            ("Quy hoạch đường đi toàn cục", False, "#1e293b"),
            ("Thuật toán A* đồ thị 8 hướng", False, "#1e293b"),
            ("Hàm Heuristic Euclidean tối ưu", False, "#1e293b"),
            ("Đầu ra: Global Path {(xk, yk)}", True, "#0369a1"),
        ],
        border_color="#0284c7",  # Sky Blue
        fill_color="#f0f9ff",
        title_color="#0369a1",
    )

    draw_card(
        ax,
        x=3.8,
        y=2.3,
        w=2.9,
        h=1.5,
        title="Sub-Goal Selector",
        lines=[
            ("Bộ chọn đích phụ thích ứng", False, "#1e293b"),
            ("Lookahead Distance L = 4.2 m", False, "#1e293b"),
            ("Cắt tỉa waypoint đã vượt qua", False, "#1e293b"),
            ("Đầu ra: Sub-goal (gx, gy)", True, "#0f766e"),
        ],
        border_color="#0d9488",  # Teal/Cyan
        fill_color="#f0fdfa",
        title_color="#0f766e",
    )

    draw_card(
        ax,
        x=3.8,
        y=0.6,
        w=2.9,
        h=1.5,
        title="Lidar2D Sensor",
        lines=[
            ("Quét Laser 2D thời gian thực", False, "#1e293b"),
            ("Tầm quét: 15 m / 360 tia (1°)", False, "#1e293b"),
            ("Dò tìm tường, xe đẩy & người", False, "#1e293b"),
            ("Đầu ra: Điểm vật cản dwa_pts", True, "#15803d"),
        ],
        border_color="#16a34a",  # Green
        fill_color="#f0fdf4",
        title_color="#15803d",
    )

    # 3. COT 3: LOCAL PLANNER (DWA - CORE)
    draw_card(
        ax,
        x=7.3,
        y=0.6,
        w=4.4,
        h=5.0,
        title="Local DWAPlanner (Dynamic Window Approach)",
        lines=[
            ("ĐẦU VÀO: Robot State (x, y, θ, v, ω), Sub-goal (gx, gy), dwa_pts", True, "#c2410c"),
            ("", False, "#000000"),
            ("1. Không gian vận tốc khả thi (Cửa sổ động Vd):", True, "#9a3412"),
            ("   Vd = [v-av·Δt, v+av·Δt] ∩ [vmin, vmax]", False, "#1e293b"),
            ("        × [ω-aω·Δt, ω+aω·Δt] ∩ [ωmin, ωmax]", False, "#1e293b"),
            ("   Ràng buộc gia tốc động cơ (av, aω) và tốc độ an toàn", False, "#475569"),
            ("", False, "#000000"),
            ("2. Mô phỏng quỹ đạo dự đoán (Forward Rollout):", True, "#9a3412"),
            ("   Mô phỏng vi sai với thời gian dự đoán Tpred = 1.2 s", False, "#1e293b"),
            ("", False, "#000000"),
            ("3. Hàm mục tiêu đánh giá tối ưu G(v, ω):", True, "#9a3412"),
            ("   G(v,ω) = α·Heading + β·Clearance + γ·Velocity", False, "#1e293b"),
            ("   Tối ưu hướng tới đích, né vật cản & tối đa tốc độ tiến", False, "#475569"),
            ("", False, "#000000"),
            ("ĐẦU RA: Vận tốc điều khiển tối ưu (v_cmd, ω_cmd)", True, "#c2410c"),
            ("             và Quỹ đạo dự đoán tương lai (pred_traj)", True, "#c2410c"),
        ],
        border_color="#ea580c",  # Vibrant Orange
        fill_color="#fff7ed",
        title_color="#c2410c",
        align_left=True,
        left_margin=0.22,
    )

    # 4. COT 4: KINEMATICS & EVALUATION
    draw_card(
        ax,
        x=12.4,
        y=3.2,
        w=3.6,
        h=2.4,
        title="Robot Kinematics",
        lines=[
            ("Mô hình AGV dẫn động vi sai (Diff-Drive)", False, "#1e293b"),
            ("Tích phân Euler chu kỳ Δt = 0.1 s:", False, "#1e293b"),
            ("ẋ = v·cos(θ),  ẏ = v·sin(θ),  θ̇ = ω", True, "#5b21b6"),
            ("Ràng buộc vận tốc bánh & gia tốc động cơ", False, "#1e293b"),
            ("Cập nhật vị thế thực (x, y, θ, v, ω)", False, "#1e293b"),
            ("Ghi nhận quỹ đạo thực tế (traveled_path)", False, "#1e293b"),
        ],
        border_color="#7c3aed",  # Purple
        fill_color="#faf5ff",
        title_color="#6d28d9",
    )

    draw_card(
        ax,
        x=12.4,
        y=0.6,
        w=3.6,
        h=2.4,
        title="Logging & Evaluation",
        lines=[
            ("Lưu trữ & Đánh giá định lượng hiệu năng", False, "#1e293b"),
            ("Kiểm thử chuẩn 12 kịch bản (A - H)", False, "#1e293b"),
            ("5 chỉ tiêu cốt lõi: Tỷ lệ thành công (100%),", True, "#0f172a"),
            ("Thời gian [s], Tỉ lệ quãng đường, Khoảng trống", False, "#1e293b"),
            ("an toàn nhỏ nhất (tĩnh & động), Độ mượt ω", False, "#1e293b"),
            ("Xuất kết quả: runs/metrics CSV, JSON & Charts", False, "#475569"),
        ],
        border_color="#334155",  # Slate
        fill_color="#f8fafc",
        title_color="#1e293b",
    )

    # CAC MUI TEN VA LUONG DU LIEU TIEN (FORWARD ARROWS)
    # HospitalMap -> Global A*
    draw_arrow(ax, (3.2, 4.8), (3.8, 4.8), color="#1d4ed8", lw=1.8, label="Lưới tĩnh", label_pos=(3.5, 5.05), label_color="#1e40af")
    # HospitalMap -> Lidar2D
    draw_arrow(ax, (3.2, 3.8), (3.8, 1.8), color="#1d4ed8", rad=-0.12, lw=1.4)
    # PedestrianManager -> Lidar2D
    draw_arrow(ax, (3.2, 1.35), (3.8, 1.35), color="#d97706", lw=1.8, label="Tác nhân động", label_pos=(3.5, 1.05), label_color="#b45309")

    # Global A* -> Sub-Goal Selector
    draw_arrow(ax, (5.25, 4.0), (5.25, 3.8), color="#0284c7", lw=2.0, label="Global Path", label_pos=(5.9, 3.9), label_color="#0369a1")

    # Sub-Goal Selector -> DWAPlanner
    draw_arrow(ax, (6.7, 3.05), (7.3, 3.05), color="#0d9488", lw=2.0, label="Sub-goal (gx, gy)", label_pos=(7.0, 3.28), label_color="#0f766e")

    # Lidar2D -> DWAPlanner
    draw_arrow(ax, (6.7, 1.35), (7.3, 1.35), color="#16a34a", lw=2.0, label="Điểm quét dwa_pts", label_pos=(7.0, 1.58), label_color="#15803d")

    # DWAPlanner -> Robot Kinematics
    draw_arrow(
        ax,
        (11.7, 4.4),
        (12.4, 4.4),
        color="#ea580c",
        lw=2.2,
        label="v_cmd, ω_cmd",
        label_pos=(12.05, 4.68),
        label_color="#c2410c",
        fontsize=8.8,
    )

    # Robot Kinematics -> Logging & Evaluation
    draw_arrow(ax, (14.2, 3.2), (14.2, 3.0), color="#64748b", lw=1.8, dashed=True, label="Ghi nhận nhật ký", label_pos=(14.2, 3.1), label_color="#475569")

    # VONG PHAN HOI (FEEDBACK LOOPS - DUONG CONG TREN CAO, KHONG CAT QUA NHA CHU)
    # 1. Tu Robot Kinematics (x=14.2, y=5.6) vong tren cao ve dinh DWAPlanner (x=9.5, y=5.6)
    draw_arrow(
        ax,
        (14.2, 5.6),
        (9.5, 5.6),
        color="#4338ca",
        lw=1.8,
        rad=0.22,
        dashed=True,
        label="Phản hồi trạng thái Robot {x, y, θ, v, ω} mỗi chu kỳ Δt = 0.1s",
        label_pos=(11.85, 6.38),
        label_color="#3730a3",
        fontsize=8.8,
    )

    # 2. Tu DWAPlanner re ve Sub-Goal Selector qua khoang trong giua 2 cot
    draw_arrow(
        ax,
        (7.3, 3.8),
        (6.7, 3.4),
        color="#4338ca",
        lw=1.5,
        rad=-0.12,
        dashed=True,
        label="Vị thế (x, y, θ)",
        label_pos=(7.0, 4.05),
        label_color="#3730a3",
        fontsize=7.8,
    )

    # Ghi chu cac tang dieu khien (Layer annotations)
    ax.text(1.85, 0.22, "TẦNG MÔ PHỎNG & TÁC NHÂN", ha="center", fontsize=8.8, fontweight="bold", color="#64748b")
    ax.text(5.25, 0.22, "TẦNG TOÀN CỤC & CẢM BIẾN", ha="center", fontsize=8.8, fontweight="bold", color="#64748b")
    ax.text(9.5, 0.22, "TẦNG ĐIỀU HƯỚNG CỤC BỘ (CORE)", ha="center", fontsize=8.8, fontweight="bold", color="#64748b")
    ax.text(14.2, 0.22, "TẦNG THỰC THI & ĐÁNH GIÁ", ha="center", fontsize=8.8, fontweight="bold", color="#64748b")

    fig.tight_layout()
    for path in out_paths:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print(f"Da xuat anh so do kien truc: {path}")
    plt.close(fig)


if __name__ == "__main__":
    targets = [
        "outputs/eval_dwa/fig3_dwa_architecture.png",
        "outputs/compare_all/fig3_dwa_architecture.png",
    ]
    generate_diagram(targets)
