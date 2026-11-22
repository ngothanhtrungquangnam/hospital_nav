"""run_batch_eval.py -- Chay danh gia hang loat cho TV1 (xuat CSV, JSON va bieu do)."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from typing import Any

from evaluation.metrics import FIVE_METRICS, aggregate_metrics, summarize_records
from evaluation.plots import plot_all_metrics
from sim.simulator import Simulator
from tests.eval_local_planner import scenarios


def _load_planner_cls(spec: str):
    if ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
    else:
        parts = spec.rsplit(".", 1)
        mod_name, cls_name = parts[0], parts[1]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation for hospital_nav")
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["planners.local_apf:APFPlanner"],
        help="Danh sach planner (vd: planners.local_apf:APFPlanner planners.local_dwa:DWAPlanner)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--out-dir", default="outputs/eval_apf")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    known, scen_list = scenarios()

    raw_rows: list[dict[str, Any]] = []
    for p_spec in args.planners:
        p_cls = _load_planner_cls(p_spec)
        p_label = p_cls.__name__
        for seed in args.seeds:
            for scen_name, goal, mk in scen_list:
                world, peds = mk()
                if peds is not None and hasattr(peds, "reset"):
                    peds.reset(seed)
                planner = p_cls()
                if hasattr(planner, "reset"):
                    planner.reset()
                sim = Simulator(known, world)
                res = sim.run(planner, goal, pedestrians=peds, seed=seed)
                s = res.summary()
                s.update({"planner": p_label, "scenario": scen_name, "seed": seed})
                raw_rows.append(s)

    norm_rows = summarize_records(raw_rows)
    agg_rows = aggregate_metrics(norm_rows, group_by=("planner", "scenario"))

    runs_csv = os.path.join(args.out_dir, "runs_summary.csv")
    agg_csv = os.path.join(args.out_dir, "metrics_5_summary.csv")
    agg_json = os.path.join(args.out_dir, "metrics_5_summary.json")

    if norm_rows:
        with open(runs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
            w.writeheader()
            w.writerows(norm_rows)

    if agg_rows:
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)
        with open(agg_json, "w", encoding="utf-8") as f:
            json.dump(agg_rows, f, indent=2, ensure_ascii=False)

    charts = plot_all_metrics(norm_rows, out_dir=args.out_dir)
    print(f"Saved {len(norm_rows)} run summaries -> {runs_csv}")
    print(f"Saved {len(agg_rows)} aggregated rows (5 metrics: {FIVE_METRICS}) -> {agg_csv}")
    print(f"Saved {len(charts)} charts -> {args.out_dir}")


if __name__ == "__main__":
    main()
