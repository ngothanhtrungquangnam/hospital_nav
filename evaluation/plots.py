"""evaluation/plots.py -- Ve va luu bieu do tong hop 5 chi tieu."""
from __future__ import annotations

import os
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import FIVE_METRICS, aggregate_metrics


def plot_metric_by_scenario(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: str,
    title: str | None = None,
    ylabel: str | None = None,
) -> str:
    """Ve bieu do cot so sanh theo tung kich ban (va theo planner neu co)."""
    agg = aggregate_metrics(rows, group_by=("planner", "scenario"))
    if not agg:
        raise ValueError("No evaluation rows to plot.")

    planners = list(dict.fromkeys(str(r["planner"]) for r in agg))
    scenarios = list(dict.fromkeys(str(r["scenario"]) for r in agg))

    lookup = {(str(r["planner"]), str(r["scenario"])): float(r.get(metric, np.nan)) for r in agg}
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(planners), 1)

    fig, ax = plt.subplots(figsize=(max(10.0, 0.95 * len(scenarios)), 5.2), dpi=150)
    for idx, p_name in enumerate(planners):
        vals = [lookup.get((p_name, sc), np.nan) for sc in scenarios]
        offset = (idx - (len(planners) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=p_name)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right", fontsize=8.5)
    ax.set_title(title or f"So sanh chi tieu: {metric}", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel or metric)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if len(planners) > 1 or planners[0] != "default":
        ax.legend(frameon=False)

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all_metrics(
    rows: list[dict[str, Any]],
    out_dir: str = "outputs/eval",
    metrics: list[str] | None = None,
) -> list[str]:
    """Xuat tron bo bieu do cho cac chi tieu (mac dinh 5 chi tieu chinh)."""
    os.makedirs(out_dir, exist_ok=True)
    labels = {
        "success_rate": ("Ty le hoan thanh thanh cong theo tinh huong", "Success rate [%]"),
        "mean_time_s": ("Thoi gian hoan thanh trung binh theo tinh huong", "Time [s]"),
        "mean_path_ratio": ("Ti le quang duong thuc / tham chieu", "Path ratio"),
        "min_clearance_m": ("Khoang trong an toan nho nhat", "Min clearance [m]"),
        "mean_smoothness_w": ("Do muot van toc goc (sum |dw|)", "Smoothness w [rad/s]"),
    }
    saved: list[str] = []
    for m in (metrics or FIVE_METRICS):
        t, yl = labels.get(m, (f"Chi tieu: {m}", m))
        p = os.path.join(out_dir, f"{m}.png")
        plot_metric_by_scenario(rows, m, out_path=p, title=t, ylabel=yl)
        saved.append(p)
    return saved
