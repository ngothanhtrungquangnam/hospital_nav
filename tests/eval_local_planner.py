"""tests/eval_local_planner.py -- Bo tinh huong danh gia bo dieu huong cuc bo (dung chung cho APF / DWA / VFH).

12 tinh huong: A (the gioi = ban do biet, 4 dich), B (xe day chua biet), C (nguoi dung yen), D/E (nguoi di nguoc
chieu 1,0 va 1,4 m/s), F (nguoi cat ngang), G (xe day + nguoi nguoc chieu), H (2 nguoi cham cung chieu).
Nguoi di bo o day la "Walker" di thang deu va KHONG ne robot (kho hon nguoi that); thay bang PedestrianManager
cua ban de do tren hanh vi that.

Cot ket qua:  ti le = quang duong robot di / duong tham chieu | cach tuong, cach nguoi = khoang trong be mat nho nhat [m]
              dao w = so lan omega doi dau (|omega| > 0,3) -> do giang co | quay = tong goc quay [rad]

Chay (PowerShell, tu thu muc goc du an, da kich hoat .venv):
    python -m tests.eval_local_planner                                  # APFPlanner trong planners.local_apf
    python -m tests.eval_local_planner planners.local_dwa DWAPlanner    # bo dieu huong khac
Khong ghi file nao; muon don bo nho dem:  Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force
"""
from __future__ import annotations
import math, importlib, sys, time
import numpy as np
import config
from env.hospital_map import HospitalMap, UNMAPPED_CARTS
from sim.simulator import Simulator


class Walker:
    """Nguoi di thang deu (khong ne robot)."""
    def __init__(self, x0, y0, vx=0.0, vy=0.0, radius=0.25, t0=0.0):
        self.p0, self.v, self.r, self.t0 = np.array([x0, y0], float), np.array([vx, vy], float), radius, t0
        self.reset()
    def reset(self, rng=None): self.p, self.t = self.p0.copy(), 0.0
    def step(self, dt):
        self.t += dt
        if self.t >= self.t0: self.p = self.p + self.v * dt
    @property
    def c(self): return [self.p[0], self.p[1], self.r]


class Peds:
    def __init__(self, walkers): self.w = walkers
    def reset(self, rng=None): [w.reset(rng) for w in self.w]
    def step(self, dt): [w.step(dt) for w in self.w]
    @property
    def circles(self): return np.array([w.c for w in self.w]).reshape(-1, 3)


def path_y_at(sim, goal, x):
    plan, _ = sim._global_plan(sim.known.start, sim.known.goals[goal])
    p = plan.path_xy
    return float(p[np.argmin(np.abs(p[:, 0] - x)), 1])


def scenarios():
    """Tra ve list (ten, dich, ham tao(world_known) -> (world, peds))."""
    known = HospitalMap()
    carts = known.with_obstacles(UNMAPPED_CARTS)
    sim0 = Simulator(known)
    yc = lambda g, x: path_y_at(sim0, g, x)
    S = []
    for g in ("G1", "G2", "G3", "G4"):
        S.append((f"A. the gioi = ban do biet -> {g}", g, lambda: (known, None)))
    for g in ("G2", "G3"):
        S.append((f"B. xe day chua biet -> {g}", g, lambda: (carts, None)))
    S.append(("C. nguoi dung yen tren duong -> G3", "G3", lambda: (known, Peds([Walker(20.0, yc("G3", 20.0))]))))
    S.append(("D. nguoi di nguoc chieu 1,0 m/s -> G3", "G3", lambda: (known, Peds([Walker(30.0, yc("G3", 25.0), vx=-1.0)]))))
    S.append(("E. nguoi di nguoc chieu 1,4 m/s -> G3", "G3", lambda: (known, Peds([Walker(32.0, yc("G3", 25.0), vx=-1.4)]))))
    S.append(("F. nguoi cat ngang tu sanh cho -> G3", "G3", lambda: (known, Peds([Walker(18.0, 7.0, vy=+1.0, t0=2.0)]))))
    S.append(("G. xe day chua biet + nguoi nguoc chieu -> G3", "G3", lambda: (carts, Peds([Walker(31.0, 9.8, vx=-1.0)]))))
    S.append(("H. 2 nguoi cung chieu cham 0,5 m/s -> G3", "G3", lambda: (known, Peds([Walker(12.0, yc("G3", 12.0), vx=0.5), Walker(14.5, 9.6, vx=0.4)]))))
    return known, S


def wflips(w, thr=0.3):
    s = np.sign(w[np.abs(w) > thr]); return int((np.diff(s) != 0).sum()) if len(s) > 1 else 0


def run_all(planner_factory, label, seeds=(1,), verbose=True):
    known, S = scenarios()
    rows = []
    for name, goal, mk in S:
        world, peds = mk()
        sim = Simulator(known, world)
        res = sim.run(planner_factory(), goal, pedestrians=peds, seed=seeds[0])
        L, s = res.log, res.summary()
        th = L["theta"]; turn = float(np.abs(np.angle(np.exp(1j * np.diff(th)))).sum()) if len(th) > 1 else 0.0
        rows.append(dict(name=name, status=res.status, t=res.time_s, ratio=s["path_ratio"], clear=s["min_clearance_static_m"],
                         gap=s["min_gap_ped_m"], flips=wflips(L["w"]), turn_rad=turn,
                         end=(L["x"][-1], L["y"][-1]), compute_ms=s["mean_compute_ms"]))
    if verbose:
        print(f"\n===== {label} =====")
        print(f"{'Tinh huong':<48}{'Ket qua':<11}{'t[s]':>6}{'ti le':>7}{'cach tuong':>11}{'cach nguoi':>11}{'dao w':>7}{'quay[rad]':>10}")
        for r in rows:
            gap = "-" if not np.isfinite(r["gap"]) else f"{r['gap']:.2f}"
            print(f"{r['name']:<48}{r['status']:<11}{r['t']:>6.1f}{r['ratio']:>7.2f}{r['clear']:>11.2f}{gap:>11}{r['flips']:>7d}{r['turn_rad']:>10.1f}")
        ok = sum(r["status"] == "success" for r in rows)
        print(f"-> thanh cong {ok}/{len(rows)}")
    return rows


if __name__ == "__main__":
    mod, cls = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("planners.local_apf", "APFPlanner")
    C = getattr(importlib.import_module(mod), cls)
    run_all(lambda: C(), f"{mod}.{cls}")
