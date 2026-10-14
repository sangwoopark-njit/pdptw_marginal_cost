"""
Clustering baselines from the PDP/VRP literature.

Every method here produces a request-to-vehicle assignment and is then routed
with the SAME exact single-vehicle solver (`routing.solve_route`), so any
difference in the reported metrics is attributable to the clustering alone.
That is the comparison our contribution actually needs: ALNS searches the whole
solution space and belongs to a different class.

  sweep        Gillett & Miller (1974).  Sort requests by polar angle about the
               depot and cut into groups when capacity is reached.  The oldest
               and still the most common cluster-first heuristic.
  fisher       Fisher & Jaikumar (1981).  Pick |K| seed requests, then solve a
               generalised assignment problem with the classical insertion cost
               a_ik = c(o_k, i) + c(i, s_k) - c(o_k, s_k).
  kmeans       Plain k-means on request midpoints, |K| clusters, then a
               capacity-repair pass.  The naive baseline.
  st_kmeans    k-means on (x, y, t): the same, with a scaled time coordinate.
               Isolates "temporal awareness without the bi-level structure".
  cap_kmeans   Bradley et al. (2000) capacity-constrained k-means, solved as
               a transportation problem.  Our direct predecessor, and the
               single-level version of our own method.
  agglom       Ward agglomerative clustering cut at |K| groups, capacity-repaired.

A request is represented by the midpoint of its pickup and delivery, so a pair
is never split across couriers.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
import routing


# ------------------------------------------------------------- helpers ----
def _mid(inst):
    return 0.5 * (inst.xp + inst.xd)


def _evaluate(inst, ehp, lhp, ehd, lhd, Tall, assign, rho=1.0, phi=0.0):
    """Route every courier exactly and return the four metrics."""
    n = inst.n_req
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    routing.set_nhalf(n)
    travel = tard = 0.0; used = 0; utils = []
    for k, req in assign.items():
        req = list(req)
        if not req:
            continue
        used += 1
        tr, pen, seq = routing.solve_route(req, Tall, e, l, S, None, None,
                                           inst.theta[k], 2 * n + k, rho)
        travel += tr; tard += pen
        load = peak = 0
        for row in seq:
            load += 1 if row < n else -1
            peak = max(peak, load)
        utils.append(peak / max(inst.cap[k], 1e-9))
    return dict(cost=travel + rho * tard + phi * used, travel=travel,
                tardiness=tard, couriers=used,
                peak_util=float(np.mean(utils)) if utils else 0.0)


def violations(labels, inst):
    """Requests over capacity before any repair, and couriers over the cap."""
    over = 0; bad = 0
    for k in range(inst.n_veh):
        c = int((labels == k).sum())
        if c > inst.cap[k]:
            over += c - int(inst.cap[k]); bad += 1
    return over, bad


def _capacity_repair(labels, inst):
    """Move requests out of over-full couriers into the nearest courier with room."""
    K = inst.n_veh
    mid = _mid(inst)
    centres = np.array([mid[labels == k].mean(0) if (labels == k).any()
                        else inst.origins[k] for k in range(K)])
    labels = labels.copy()
    for k in range(K):
        while (labels == k).sum() > inst.cap[k]:
            members = np.where(labels == k)[0]
            far = members[np.argmax(((mid[members] - centres[k]) ** 2).sum(1))]
            room = [j for j in range(K) if (labels == j).sum() < inst.cap[j]]
            if not room:
                break
            d = [((mid[far] - centres[j]) ** 2).sum() for j in room]
            labels[far] = room[int(np.argmin(d))]
    return labels


def _to_assign(labels, K):
    return {k: np.where(labels == k)[0].tolist() for k in range(K)}


# ---------------------------------------------------------------- sweep ---
def sweep(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0):
    mid = _mid(inst)
    centre = inst.origins.mean(0)
    ang = np.arctan2(mid[:, 1] - centre[1], mid[:, 0] - centre[0])
    order = np.argsort(ang)
    labels = np.zeros(inst.n_req, int)
    k = 0; load = 0
    for r in order:
        if load >= inst.cap[k] and k < inst.n_veh - 1:
            k += 1; load = 0
        labels[r] = k; load += 1
    m = _evaluate(inst, ehp, lhp, ehd, lhd, Tall, _to_assign(labels, inst.n_veh), rho, phi)
    m["labels"] = labels
    return m


# -------------------------------------------------------------- fisher ----
def fisher(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0):
    """Fisher & Jaikumar: seeds by max-spread, then a transportation problem."""
    rng = np.random.default_rng(seed)
    mid = _mid(inst); n = inst.n_req; K = inst.n_veh
    seeds = [int(rng.integers(n))]
    for _ in range(K - 1):
        d = np.min([((mid - mid[s]) ** 2).sum(1) for s in seeds], axis=0)
        seeds.append(int(np.argmax(d)))
    o = inst.origins
    a = np.zeros((n, K))
    for k in range(K):
        s = mid[seeds[k]]
        c_o_i = np.sqrt(((mid - o[k]) ** 2).sum(1))
        c_i_s = np.sqrt(((mid - s) ** 2).sum(1))
        c_o_s = np.sqrt(((s - o[k]) ** 2).sum())
        a[:, k] = c_o_i + c_i_s - c_o_s
    # transportation problem: each request to one courier, capacity respected
    A_eq = np.zeros((n, n * K))
    for r in range(n):
        A_eq[r, r * K:(r + 1) * K] = 1.0
    A_ub = np.zeros((K, n * K))
    for k in range(K):
        A_ub[k, k::K] = 1.0
    res = linprog(a.ravel(), A_ub=A_ub, b_ub=inst.cap, A_eq=A_eq,
                  b_eq=np.ones(n), bounds=(0, 1), method="highs")
    labels = (res.x.reshape(n, K).argmax(1) if res.success
              else _capacity_repair(a.argmin(1), inst))
    m = _evaluate(inst, ehp, lhp, ehd, lhd, Tall, _to_assign(labels, K), rho, phi)
    m["labels"] = labels
    return m


# -------------------------------------------------------------- k-means ---
def _lloyd(X, K, rng, iters=50):
    idx = [int(rng.integers(len(X)))]
    for _ in range(K - 1):
        d = np.min([((X - X[i]) ** 2).sum(1) for i in idx], axis=0)
        p = d / d.sum() if d.sum() > 0 else None
        idx.append(int(rng.choice(len(X), p=p)))
    C = X[idx].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        new = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        if (new == lab).all():
            break
        lab = new
        for k in range(K):
            if (lab == k).any():
                C[k] = X[lab == k].mean(0)
    return lab


def kmeans(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lab = _lloyd(_mid(inst), inst.n_veh, rng)
    over, bad = violations(lab, inst)
    rep = _capacity_repair(lab, inst)
    m = _evaluate(inst, ehp, lhp, ehd, lhd, Tall, _to_assign(rep, inst.n_veh), rho, phi)
    m.update(cap_over=over, cap_bad_veh=bad, moves=int((lab != rep).sum()))
    m["labels"] = rep
    return m


def st_kmeans(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0, alpha=1.0):
    """k-means on (x, y, t), the time axis scaled to the spatial spread."""
    rng = np.random.default_rng(seed)
    mid = _mid(inst); t = 0.5 * (ehp + lhd)
    sx = mid.std(0).mean(); st = t.std()
    X = np.column_stack([mid, alpha * (sx / max(st, 1e-9)) * t])
    lab = _lloyd(X, inst.n_veh, rng)
    over, bad = violations(lab, inst)
    rep = _capacity_repair(lab, inst)
    m = _evaluate(inst, ehp, lhp, ehd, lhd, Tall, _to_assign(rep, inst.n_veh), rho, phi)
    m.update(cap_over=over, cap_bad_veh=bad, moves=int((lab != rep).sum()))
    m["labels"] = rep
    return m


def cap_kmeans(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0, iters=30):
    """Bradley et al.: alternate a transportation assignment with centroid update."""
    rng = np.random.default_rng(seed)
    mid = _mid(inst); n = inst.n_req; K = inst.n_veh
    idx = rng.choice(n, size=K, replace=False)
    C = mid[idx].copy()
    lab = np.zeros(n, int)
    A_eq = np.zeros((n, n * K))
    for r in range(n):
        A_eq[r, r * K:(r + 1) * K] = 1.0
    A_ub = np.zeros((K, n * K))
    for k in range(K):
        A_ub[k, k::K] = 1.0
    for _ in range(iters):
        d = ((mid[:, None, :] - C[None]) ** 2).sum(-1)
        res = linprog(d.ravel(), A_ub=A_ub, b_ub=inst.cap, A_eq=A_eq,
                      b_eq=np.ones(n), bounds=(0, 1), method="highs")
        if not res.success:
            break
        new = res.x.reshape(n, K).argmax(1)
        if (new == lab).all():
            break
        lab = new
        for k in range(K):
            if (lab == k).any():
                C[k] = mid[lab == k].mean(0)
    m = _evaluate(inst, ehp, lhp, ehd, lhd, Tall, _to_assign(lab, K), rho, phi)
    m["labels"] = lab
    return m


def agglom(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0):
    from scipy.cluster.hierarchy import linkage, fcluster
    mid = _mid(inst)
    Z = linkage(mid, method="ward")
    lab = fcluster(Z, t=inst.n_veh, criterion="maxclust") - 1
    over, bad = violations(lab, inst)
    rep = _capacity_repair(lab, inst)
    m = _evaluate(inst, ehp, lhp, ehd, lhd, Tall, _to_assign(rep, inst.n_veh), rho, phi)
    m.update(cap_over=over, cap_bad_veh=bad, moves=int((lab != rep).sum()))
    m["labels"] = rep
    return m


METHODS = {"sweep": sweep, "fisher": fisher, "kmeans": kmeans,
           "st_kmeans": st_kmeans, "cap_kmeans": cap_kmeans, "agglom": agglom}
