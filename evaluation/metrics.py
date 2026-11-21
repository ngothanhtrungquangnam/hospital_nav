"""evaluation/metrics.py -- Tong hop 5 chi tieu danh gia tu ket qua Simulator."""
from __future__ import annotations

import math
from typing import Any
import numpy as np

# 5 chi tieu danh gia chinh
FIVE_METRICS: list[str] = [
    "success_rate",           # 1. Ty le hoan thanh thanh cong (%)
    "mean_time_s",            # 2. Thoi gian di chuyen trung binh [s]
    "mean_path_ratio",        # 3. Ti le quang duong thuc / tham chieu
    "min_clearance_m",        # 4. Khoang trong an toan nho nhat (tuong & nguoi) [m]
    "mean_smoothness_w",      # 5. Do muot van toc goc trung binh (sum |dw|)
]


def _to_summary_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        d = dict(item)
    elif hasattr(item, "summary") and callable(item.summary):
        d = dict(item.summary())
    else:
        raise TypeError(f"Unsupported record type: {type(item)!r}")

    if "success" not in d:
        d["success"] = bool(d.get("status") == "success")
    if "collision" not in d:
        d["collision"] = bool(d.get("status") == "collision")
    return d


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chuan hoa danh sach summary va bo sung khoang trong an toan gop min_clearance_m."""
    out: list[dict[str, Any]] = []
    for r in records:
        d = _to_summary_dict(r)
        c_stat = d.get("min_clearance_static_m", float("nan"))
        c_ped = d.get("min_gap_ped_m", float("nan"))
        vals = [float(v) for v in (c_stat, c_ped) if v is not None and not math.isnan(float(v))]
        d["min_clearance_m"] = min(vals) if vals else float("nan")
        d["success_pct"] = 100.0 * (1.0 if d.get("success") else 0.0)
        out.append(d)
    return out


def _nanmean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanmin(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def aggregate_metrics(
    rows: list[dict[str, Any]],
    group_by: tuple[str, ...] = ("planner", "scenario"),
) -> list[dict[str, Any]]:
    """Gop nhom theo group_by va tinh 5 chi tieu + mean_compute_ms."""
    norm_rows = summarize_records(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in norm_rows:
        key = tuple(r.get(k, "default") for k in group_by)
        groups.setdefault(key, []).append(r)

    summary_rows: list[dict[str, Any]] = []
    for key, items in groups.items():
        n = len(items)
        n_succ = sum(1 for x in items if bool(x.get("success")))
        n_coll = sum(1 for x in items if bool(x.get("collision")))
        entry: dict[str, Any] = {k: v for k, v in zip(group_by, key)}
        entry.update({
            "n_runs": n,
            "n_success": n_succ,
            "n_collision": n_coll,
            "success_rate": 100.0 * n_succ / n if n else float("nan"),
            "mean_time_s": _nanmean([float(x.get("time_s", float("nan"))) for x in items]),
            "mean_path_ratio": _nanmean([float(x.get("path_ratio", float("nan"))) for x in items]),
            "min_clearance_m": _nanmin([float(x.get("min_clearance_m", float("nan"))) for x in items]),
            "min_clearance_static_m": _nanmin([float(x.get("min_clearance_static_m", float("nan"))) for x in items]),
            "min_gap_ped_m": _nanmin([float(x.get("min_gap_ped_m", float("nan"))) for x in items]),
            "mean_smoothness_w": _nanmean([float(x.get("smoothness_w", float("nan"))) for x in items]),
            "mean_compute_ms": _nanmean([float(x.get("mean_compute_ms", float("nan"))) for x in items]),
        })
        summary_rows.append(entry)
    return summary_rows
