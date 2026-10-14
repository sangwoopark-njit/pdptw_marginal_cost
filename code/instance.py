"""
Instance handling for the bi-level clustering experiments.

Li & Lim PDPTW files are read and adapted to the setting of the paper:
  * uniform loads   - every request occupies one unit of courier capacity
  * per-vehicle origins and ready times
  * soft time windows - violations are priced, never forbidden
  * open routes - a courier ends at its last delivery

Li & Lim format
  line 1 : n_vehicles  capacity  speed
  line i : id  x  y  demand  earliest  latest  service  pickup_idx  delivery_idx
  node 0 is the depot; demand > 0 marks a pickup, demand < 0 a delivery.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Instance:
    name: str
    n_req: int                 # number of requests
    xp: np.ndarray             # (n_req, 2) pickup coordinates
    xd: np.ndarray             # (n_req, 2) delivery coordinates
    ep: np.ndarray             # pickup release times
    lp: np.ndarray             # pickup latest start
    ed: np.ndarray             # delivery release times
    ld: np.ndarray             # delivery latest start
    sp: np.ndarray             # pickup service durations
    sd: np.ndarray             # delivery service durations
    origins: np.ndarray        # (n_veh, 2) courier start positions
    theta: np.ndarray          # (n_veh,)   courier ready times
    cap: np.ndarray            # (n_veh,)   capacity, in requests
    horizon: float

    @property
    def n_veh(self) -> int:
        return len(self.theta)

    def stops(self):
        """All stop coordinates, pickups first then deliveries."""
        return np.vstack([self.xp, self.xd])


def read_lilim(path: str):
    """Parse a Li & Lim file into raw node arrays."""
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    hdr = lines[0].split()
    n_veh, cap = int(hdr[0]), float(hdr[1])
    rows = [list(map(float, ln.split())) for ln in lines[1:]]
    arr = np.array(rows)
    return n_veh, cap, arr


def build_instance(path: str, name: str, n_req: int, n_veh: int,
                   cap_per_veh: int, rng: np.random.Generator,
                   stagger: float = 0.0, spread: float = 0.0,
                   slice_frac: float | None = None) -> Instance:
    """
    Adapt a Li & Lim file.

    n_req        requests to retain (sampled without replacement)
    n_veh        couriers
    cap_per_veh  capacity in requests (uniform loads)
    stagger      couriers become ready at U[0, stagger * horizon]
    spread       couriers start at the depot perturbed by N(0, spread) per axis;
                 spread = 0 puts every courier at the depot
    slice_frac   sample a dispatch batch: requests released inside one window
                 covering this fraction of the horizon
    """
    _, _, arr = read_lilim(path)
    depot = arr[0]
    horizon = float(depot[5])

    # pair pickups with their deliveries
    ids = arr[:, 0].astype(int)
    idx = {int(i): k for k, i in enumerate(ids)}
    pickups = [k for k in range(len(arr)) if arr[k, 3] > 0]

    if slice_frac is not None and slice_frac < 1.0:
        # a dispatch batch: keep requests released within one window of the day
        rel = arr[pickups, 4]
        lo_pool = rel[rel <= horizon * (1.0 - slice_frac)]
        base = float(rng.choice(lo_pool)) if len(lo_pool) else float(rel.min())
        hi = base + slice_frac * horizon
        pool = [k for k in pickups if base <= arr[k, 4] <= hi]
        if len(pool) < n_req:                      # widen until enough
            order = np.argsort(np.abs(arr[pickups, 4] - (base + hi) / 2))
            pool = [pickups[i] for i in order[:max(n_req, len(pool))]]
        pickups = pool

    chosen = rng.choice(pickups, size=min(n_req, len(pickups)), replace=False)
    chosen = np.sort(chosen)

    xp, xd, ep, lp, ed, ld, sp, sd = ([] for _ in range(8))
    for k in chosen:
        d_row = idx[int(arr[k, 8])]           # delivery index of this pickup
        xp.append(arr[k, 1:3]);  xd.append(arr[d_row, 1:3])
        ep.append(arr[k, 4]);    lp.append(arr[k, 5])
        ed.append(arr[d_row, 4]); ld.append(arr[d_row, 5])
        sp.append(arr[k, 6]);    sd.append(arr[d_row, 6])

    origins = np.tile(depot[1:3], (n_veh, 1)).astype(float)
    if spread > 0:
        origins = origins + rng.normal(0.0, spread, size=origins.shape)
    theta = (rng.uniform(0.0, stagger * horizon, size=n_veh)
             if stagger > 0 else np.zeros(n_veh))

    return Instance(
        name=name, n_req=len(chosen),
        xp=np.array(xp), xd=np.array(xd),
        ep=np.array(ep), lp=np.array(lp),
        ed=np.array(ed), ld=np.array(ld),
        sp=np.array(sp), sd=np.array(sd),
        origins=origins, theta=theta,
        cap=np.full(n_veh, cap_per_veh, dtype=float),
        horizon=horizon,
    )


def travel_matrix(pts: np.ndarray) -> np.ndarray:
    """Euclidean travel times; Li & Lim uses unit speed."""
    d = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def tighten(inst: Instance):
    """
    Window tightening of the paper:
        e_hat_p = e_p                       l_hat_p = min(l_p, l_d - S_p - T_pd)
        e_hat_d = max(e_d, e_p + S_p + T_pd) l_hat_d = l_d
    """
    tpd = np.sqrt(((inst.xp - inst.xd) ** 2).sum(1))
    ehp = inst.ep.copy()
    lhp = np.minimum(inst.lp, inst.ld - inst.sp - tpd)
    ehd = np.maximum(inst.ed, inst.ep + inst.sp + tpd)
    lhd = inst.ld.copy()
    # a tightened window can invert on pathological data; clamp to a point
    lhp = np.maximum(lhp, ehp)
    ehd = np.minimum(ehd, lhd)
    return ehp, lhp, ehd, lhd, tpd
