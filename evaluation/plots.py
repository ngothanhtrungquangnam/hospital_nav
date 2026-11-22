"""evaluation/plots.py -- Vẽ và lưu biểu đồ tổng hợp 5 chỉ tiêu với tiếng Việt có dấu chuẩn."""
from __future__ import annotations

import os
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import FIVE_METRICS, aggregate_metrics

# Cấu hình font chữ hỗ trợ tiếng Việt có dấu đầy đủ trên Windows & đa nền tảng
plt.rcParams["font.sans-serif"] = [
    "Segoe UI",
    "Arial",
    "DejaVu Sans",
    "Tahoma",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# Bảng chuyển đổi tên tình huống sang tiếng Việt có dấu đẹp mắt, súc tích
SCENARIO_DISPLAY: dict[str, str] = {
    "A. the gioi = ban do biet -> G1": "A. Bản đồ biết -> G1",
    "A. the gioi = ban do biet -> G2": "A. Bản đồ biết -> G2",
    "A. the gioi = ban do biet -> G3": "A. Bản đồ biết -> G3",
    "A. the gioi = ban do biet -> G4": "A. Bản đồ biết -> G4",
    "B. xe day chua biet -> G2": "B. Xe đẩy chưa biết -> G2",
    "B. xe day chua biet -> G3": "B. Xe đẩy chưa biết -> G3",
    "C. nguoi dung yen tren duong -> G3": "C. Người đứng yên -> G3",
    "D. nguoi di nguoc chieu 1,0 m/s -> G3": "D. Ngược chiều 1,0 m/s -> G3",
    "E. nguoi di nguoc chieu 1,4 m/s -> G3": "E. Ngược chiều 1,4 m/s -> G3",
    "F. nguoi cat ngang tu sanh cho -> G3": "F. Cắt ngang sảnh -> G3",
    "G. xe day chua biet + nguoi nguoc chieu -> G3": "G. Xe đẩy + ngược chiều -> G3",
    "H. 2 nguoi cung chieu cham 0,5 m/s -> G3": "H. 2 người cùng chiều 0,5 m/s -> G3",
    # Ánh xạ dự phòng nếu đã có dấu
    "A. Bản đồ biết -> G1": "A. Bản đồ biết -> G1",
    "A. Bản đồ biết -> G2": "A. Bản đồ biết -> G2",
    "A. Bản đồ biết -> G3": "A. Bản đồ biết -> G3",
    "A. Bản đồ biết -> G4": "A. Bản đồ biết -> G4",
    "B. Xe đẩy chưa biết -> G2": "B. Xe đẩy chưa biết -> G2",
    "B. Xe đẩy chưa biết -> G3": "B. Xe đẩy chưa biết -> G3",
    "C. Người đứng yên -> G3": "C. Người đứng yên -> G3",
    "D. Ngược chiều 1,0 m/s -> G3": "D. Ngược chiều 1,0 m/s -> G3",
    "E. Ngược chiều 1,4 m/s -> G3": "E. Ngược chiều 1,4 m/s -> G3",
    "F. Cắt ngang sảnh -> G3": "F. Cắt ngang sảnh -> G3",
    "G. Xe đẩy + ngược chiều -> G3": "G. Xe đẩy + ngược chiều -> G3",
    "H. 2 người cùng chiều 0,5 m/s -> G3": "H. 2 người cùng chiều 0,5 m/s -> G3",
}


def plot_metric_by_scenario(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: str,
    title: str | None = None,
    ylabel: str | None = None,
) -> str:
    """Vẽ biểu đồ cột so sánh theo từng kịch bản (và theo planner nếu có)."""
    agg = aggregate_metrics(rows, group_by=("planner", "scenario"))
    if not agg:
        raise ValueError("No evaluation rows to plot.")

    planners = list(dict.fromkeys(str(r["planner"]) for r in agg))
    scenarios = list(dict.fromkeys(str(r["scenario"]) for r in agg))

    lookup = {(str(r["planner"]), str(r["scenario"])): float(r.get(metric, np.nan)) for r in agg}
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(planners), 1)

    fig, ax = plt.subplots(figsize=(max(11.0, 0.95 * len(scenarios)), 5.5), dpi=150)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, p_name in enumerate(planners):
        vals = [lookup.get((p_name, sc), np.nan) for sc in scenarios]
        offset = (idx - (len(planners) - 1) / 2.0) * width
        color = palette[idx % len(palette)] if len(planners) > 1 else "#1f77b4"
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=p_name,
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_xticks(x)
    display_scenarios = [SCENARIO_DISPLAY.get(sc, sc) for sc in scenarios]
    ax.set_xticklabels(display_scenarios, rotation=25, ha="right", fontsize=9)

    if len(planners) == 1 and planners[0] != "default":
        full_title = f"{title or metric} [{planners[0]}]"
    else:
        full_title = title or f"So sánh chỉ tiêu: {metric}"

    ax.set_title(full_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel or metric, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if len(planners) > 1:
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all_metrics(
    rows: list[dict[str, Any]],
    out_dir: str = "outputs/eval_apf",
    metrics: list[str] | None = None,
) -> list[str]:
    """Xuất trọn bộ biểu đồ cho các chỉ tiêu (mặc định 5 chỉ tiêu chính) có dấu tiếng Việt."""
    os.makedirs(out_dir, exist_ok=True)
    labels = {
        "success_rate": ("Tỷ lệ hoàn thành thành công theo kịch bản", "Tỷ lệ thành công (%)"),
        "mean_time_s": ("Thời gian hoàn thành trung bình theo kịch bản", "Thời gian [giây]"),
        "mean_path_ratio": ("Tỉ lệ quãng đường thực tế / tham chiếu", "Tỉ lệ quãng đường (thực/tham chiếu)"),
        "min_clearance_m": ("Khoảng cách an toàn nhỏ nhất theo kịch bản", "Khoảng cách an toàn nhỏ nhất [m]"),
        "mean_smoothness_w": ("Độ mượt vận tốc góc trung bình (tổng |Δω|)", "Độ mượt ω [rad/s]"),
    }
    saved: list[str] = []
    for m in (metrics or FIVE_METRICS):
        t, yl = labels.get(m, (f"Chỉ tiêu: {m}", m))
        p = os.path.join(out_dir, f"{m}.png")
        plot_metric_by_scenario(rows, m, out_path=p, title=t, ylabel=yl)
        saved.append(p)
    return saved
