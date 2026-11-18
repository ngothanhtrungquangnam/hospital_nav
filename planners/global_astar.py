"""planners/global_astar.py -- Quy hoach toan cuc A* tren luoi 8 huong + chon sub-goal.

Nguoi phu trach: TV1.

Gom 3 phan:
  * GlobalAStarPlanner : tim duong tren ``map.inflated`` (ban do DA BIET, da phong vat can).
  * PlanResult         : ket qua (duong di, do dai, so o mo rong, thoi gian...).
  * SubGoalSelector    : tu duong di + vi tri robot -> sub-goal cho bo dieu huong cuc bo
                         (DWA / APF / VFH), theo kieu "look-ahead" doc duong di.

Quy tac di chuyen: 8 huong, chi phi 1 (ngang/doc) va sqrt(2) (cheo).
KHONG cho cat goc: buoc cheo bi cam neu mot trong hai o ke ngang/doc bi chan.
Vi tim tren ban do da phong vat can nen robot (coi nhu mot diem) khong cham tuong.

Heuristic ('heuristic=' khi tao planner):
  'octile'    : chinh xac nhat cho luoi 8 huong, chap nhan duoc (admissible)  [mac dinh]
  'euclid'    : chap nhan duoc nhung "long leo" hon octile
  'manhattan' : KHONG chap nhan duoc khi cho di cheo -> khong dam bao toi uu
  'dijkstra'  : h = 0 (khong dung heuristic), moc so sanh

Toa do: met, he toa do the gioi cua HospitalMap (xem env/hospital_map.py).
Chi so luoi (r, c) la 0-based, r <-> y, c <-> x.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

_SQRT2 = math.sqrt(2.0)
_DR = (-1, 1, 0, 0, -1, -1, 1, 1)          # 4 huong dau: ngang/doc; 4 huong sau: cheo
_DC = (0, 0, -1, 1, -1, 1, -1, 1)
_COST = (1.0, 1.0, 1.0, 1.0, _SQRT2, _SQRT2, _SQRT2, _SQRT2)

HEURISTICS = ("octile", "euclid", "manhattan", "dijkstra")


@dataclass
class PlanResult:
    """Ket qua mot lan quy hoach."""
    found: bool = False
    msg: str = ""                                   # ly do that bai ('' neu thanh cong)
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    path_rc: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    length: float = float("nan")                    # [m] do dai duong luoi (ngang/doc=res, cheo=res*sqrt2)
    n_expanded: int = 0                             # so o da mo rong (chi so danh gia heuristic)
    n_pushed: int = 0                               # so lan day vao hang doi uu tien
    time_s: float = float("nan")
    h_type: str = "octile"
    closed: np.ndarray | None = None                # bool (n_rows, n_cols): cac o da mo rong (de ve minh hoa)


def polyline_length(points: np.ndarray) -> float:
    """Do dai duong gap khuc di qua cac diem (N, 2) [m]."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


class GlobalAStarPlanner:
    """Quy hoach toan cuc A* tren ban do da biet.

    Dung::

        planner = GlobalAStarPlanner(known_map)                 # known_map: HospitalMap
        res = planner.plan(known_map.start, known_map.goals["G1"])
        if res.found:
            waypoints = planner.simplify(res.path_xy)           # rut gon bang line-of-sight
            selector = SubGoalSelector(waypoints, lookahead=1.5)
    """

    def __init__(self, grid_map, heuristic: str = "octile"):
        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic phai la mot trong {HEURISTICS}, nhan duoc '{heuristic}'")
        self.map = grid_map
        self.heuristic = heuristic

    # ------------------------------------------------------------------ heuristic
    def _make_h(self, gr: int, gc: int):
        t = self.heuristic
        if t == "dijkstra":
            return lambda r, c: 0.0
        if t == "manhattan":
            return lambda r, c: float(abs(r - gr) + abs(c - gc))
        if t == "euclid":
            return lambda r, c: math.hypot(r - gr, c - gc)
        k = _SQRT2 - 2.0                                              # octile
        return lambda r, c: (abs(r - gr) + abs(c - gc)) + k * min(abs(r - gr), abs(c - gc))

    # ------------------------------------------------------------------ A*
    def plan(self, start_xy, goal_xy) -> PlanResult:
        """Tim duong tu start_xy toi goal_xy (met). Khong bao loi: that bai -> found=False + msg."""
        t0 = time.perf_counter()
        m = self.map
        out = PlanResult(h_type=self.heuristic)
        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)

        # ---- kiem tra dau vao ----
        for name, p in (("Start", start_xy), ("Goal", goal_xy)):
            if not (0.0 <= p[0] < m.width and 0.0 <= p[1] < m.height):
                out.msg = f"{name} nam ngoai ban do"
                out.time_s = time.perf_counter() - t0
                return out
        blocked = m.inflated
        n_rows, n_cols = blocked.shape
        sr, sc = m.xy_to_rc(*start_xy)
        gr, gc = m.xy_to_rc(*goal_xy)
        for name, (r, c) in (("Start", (sr, sc)), ("Goal", (gr, gc))):
            if blocked[r, c]:
                out.msg = f"{name} nam trong vat can (hoac vung phong vat can)"
                out.time_s = time.perf_counter() - t0
                return out

        # ---- du lieu thuat toan (danh chi so phang de chay nhanh trong Python thuan) ----
        n = n_rows * n_cols
        blk = blocked.ravel().tolist()
        g = [math.inf] * n
        parent = [-1] * n
        closed = bytearray(n)
        h = self._make_h(gr, gc)

        s_idx, g_idx = sr * n_cols + sc, gr * n_cols + gc
        g[s_idx] = 0.0
        hs = h(sr, sc)
        heap = [(hs, hs, s_idx)]                    # (f, h, chi so): hoa nhau thi uu tien o gan dich
        n_push, n_exp, found = 1, 0, False

        while heap:
            _, _, cur = heapq.heappop(heap)
            if closed[cur]:                         # ban ghi cu (xoa "lazy")
                continue
            closed[cur] = 1
            n_exp += 1
            if cur == g_idx:
                found = True
                break
            r, c = divmod(cur, n_cols)
            g_cur = g[cur]
            for k in range(8):
                rr, cc = r + _DR[k], c + _DC[k]
                if rr < 0 or rr >= n_rows or cc < 0 or cc >= n_cols:
                    continue
                nb = rr * n_cols + cc
                if blk[nb] or closed[nb]:
                    continue
                if k >= 4 and (blk[rr * n_cols + c] or blk[r * n_cols + cc]):   # cam cat goc
                    continue
                tg = g_cur + _COST[k]
                if tg < g[nb]:
                    g[nb] = tg
                    parent[nb] = cur
                    hv = h(rr, cc)
                    heapq.heappush(heap, (tg + hv, hv, nb))
                    n_push += 1

        out.n_expanded, out.n_pushed = n_exp, n_push
        out.closed = np.frombuffer(bytes(closed), dtype=np.uint8).reshape(n_rows, n_cols).astype(bool)
        if not found:
            out.msg = "Khong co duong di tu Start toi Goal"
            out.time_s = time.perf_counter() - t0
            return out

        # ---- truy vet ----
        idx, chain = g_idx, []
        while idx != -1:
            chain.append(idx)
            idx = parent[idx]
        chain.reverse()
        rows, cols = np.divmod(np.asarray(chain), n_cols)
        x, y = m.rc_to_xy(rows, cols)
        path_xy = np.column_stack([x, y]).astype(float)
        path_xy[0], path_xy[-1] = start_xy, goal_xy         # dau/cuoi dung toa do that, khong phai tam o

        out.found = True
        out.path_rc = np.column_stack([rows, cols])
        out.path_xy = path_xy
        out.length = g[g_idx] * m.res
        out.time_s = time.perf_counter() - t0
        return out

    # ------------------------------------------------------------------ tuyen nhieu diem
    def plan_route(self, stops) -> PlanResult:
        """Quy hoach qua nhieu diem lien tiep: stops = [p0, p1, p2, ...]. Ghep cac doan thanh mot duong."""
        stops = [np.asarray(p, dtype=float) for p in stops]
        total = PlanResult(h_type=self.heuristic, found=True, length=0.0, time_s=0.0)
        pieces = []
        for a, b in zip(stops[:-1], stops[1:]):
            r = self.plan(a, b)
            total.n_expanded += r.n_expanded
            total.n_pushed += r.n_pushed
            total.time_s += r.time_s
            if not r.found:
                total.found, total.msg = False, f"Doan ({a[0]:.1f},{a[1]:.1f}) -> ({b[0]:.1f},{b[1]:.1f}): {r.msg}"
                return total
            total.length += r.length
            pieces.append(r.path_xy if not pieces else r.path_xy[1:])
        if pieces:
            total.path_xy = np.vstack(pieces)
        return total

    # ------------------------------------------------------------------ hau xu ly
    def segment_free(self, a, b) -> bool:
        """True neu doan thang a->b khong cat vung bi chan (lay mau moi res/4)."""
        m = self.map
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = float(np.hypot(*(b - a)))
        n = max(2, int(math.ceil(d / (m.res / 4.0))) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        r, c = m.xy_to_rc(pts[:, 0], pts[:, 1])
        return not bool(m.inflated[r, c].any())

    def simplify(self, path_xy) -> np.ndarray:
        """Rut gon duong luoi thanh cac waypoint bang kiem tra tam nhin thang (line-of-sight).

        Giu nguyen diem dau/cuoi. Moi doan thang giua hai waypoint lien tiep deu nam ngoai vung phong
        vat can. Tra ve (K, 2) voi K <= N.
        """
        p = np.asarray(path_xy, dtype=float)
        n = len(p)
        if n <= 2:
            return p.copy()
        keep, i = [0], 0
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not self.segment_free(p[i], p[j]):
                j -= 1
            keep.append(j)
            i = j
        return p[keep]

    @staticmethod
    def resample(path_xy, spacing: float) -> np.ndarray:
        """Lay mau lai duong gap khuc thanh cac diem cach deu nhau `spacing` [m] (co ca diem cuoi)."""
        p = np.asarray(path_xy, dtype=float)
        if len(p) < 2:
            return p.copy()
        seg = np.hypot(*np.diff(p, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(0.0, cum[-1], spacing)
        s = np.append(s, cum[-1])
        return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


class SubGoalSelector:
    """Chon sub-goal doc duong di theo kieu look-ahead, tien dan ve phia truoc (khong lui).

    Moi lan goi ``update(xy)``:
      1. chieu vi tri robot len duong gap khuc (chi xet cac doan tu doan hien tai tro di,
         trong cua so ``window`` doan -> khong "nhay tat" sang phan duong xa hon),
      2. sub-goal = diem tren duong cach hinh chieu `lookahead` [m] ve phia truoc
         (bi chan o diem cuoi cua duong).

    Dung trong vong lap mo phong::

        sel = SubGoalSelector(waypoints, lookahead=1.5, goal_tol=0.3)
        sub_goal, reached = sel.update(robot_xy)
        v, w, traj = local_planner.compute_control(state, sub_goal, points, dt)
    """

    def __init__(self, waypoints, lookahead: float = 1.5, goal_tol: float = 0.3, window: int = 20):
        w = np.asarray(waypoints, dtype=float).reshape(-1, 2)
        if len(w) == 0:
            raise ValueError("waypoints rong")
        keep = [0] + [i for i in range(1, len(w)) if np.hypot(*(w[i] - w[i - 1])) > 1e-9]
        self.wp = w[keep]                                           # bo diem trung nhau
        self.lookahead, self.goal_tol, self.window = float(lookahead), float(goal_tol), int(window)
        d = np.diff(self.wp, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1]) if len(d) else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])
        self.reset()

    def reset(self) -> None:
        self.seg = 0                                                # doan hien tai
        self.s = 0.0                                                # cung dai [m] cua hinh chieu tu dau duong

    def _point_at(self, s: float) -> np.ndarray:
        if len(self.seg_len) == 0:
            return self.wp[0].copy()
        s = min(max(s, 0.0), self.total)
        j = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        t = (s - self.cum[j]) / self.seg_len[j]
        return self.wp[j] + t * (self.wp[j + 1] - self.wp[j])

    def update(self, xy):
        """Tra ve (sub_goal [x, y], reached). reached=True khi robot den trong goal_tol cua diem cuoi."""
        p = np.asarray(xy, dtype=float)
        if len(self.seg_len):
            hi = min(self.seg + self.window, len(self.seg_len))
            a, d, L = self.wp[self.seg:hi], np.diff(self.wp, axis=0)[self.seg:hi], self.seg_len[self.seg:hi]
            t = np.clip(((p - a) * d).sum(axis=1) / (L * L), 0.0, 1.0)
            proj = a + t[:, None] * d
            k = int(np.argmin(np.hypot(*(proj - p).T)))
            self.seg += k
            self.s = max(self.s, float(self.cum[self.seg] + t[k] * L[k]))   # khong cho lui
        sub_goal = self._point_at(self.s + self.lookahead)
        reached = bool(np.hypot(*(p - self.wp[-1])) <= self.goal_tol)
        return sub_goal, reached

    @property
    def remaining(self) -> float:
        """Do dai con lai doc duong tu hinh chieu hien tai toi dich [m]."""
        return self.total - self.s