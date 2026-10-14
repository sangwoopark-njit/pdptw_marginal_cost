"""
Baselines for the experimental campaign.

B1  monolithic MIP     the corrected model of Section II, on HiGHS
B2  OR-Tools           Google's routing library, guided local search
B4  single-level       capacity-constrained k-means straight to vehicles,
                       |C| = |K|, followed by the same exact routing

B3 (ALNS), B5 and B6 are configurations of cluster.run and live in campaign.py.
"""
from __future__ import annotations
import numpy as np, time
import routing


# ============================================================ B1: MIP ====
def monolithic_mip(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, time_limit=600,
                   gap=1e-6):
    """
    Corrected monolithic formulation.  Open routes via a dummy terminal per
    vehicle, per-vehicle origins and ready times, soft windows, uniform loads,
    dispatch cost phi.  Returns (cost, bound, status, seconds).

    Variables per vehicle k:
      x_ijk  arc use          b_ik  service start     y_ik  load on leaving
      t+_ik  tardiness        u_k   dispatched
    """
    import highspy
    n = inst.n_req; K = inst.n_veh
    N = 2 * n                                   # stops: pickups 0..n-1, deliveries n..2n-1
    O = lambda k: N + k                         # origin row in Tall
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    q = np.concatenate([np.ones(n), -np.ones(n)])
    Q = inst.cap; th = inst.theta

    # reachability bounds (Prop. 5), per vehicle
    Tbar = Tall[:N, :N].max(axis=0)
    def bounds(k):
        lo = np.empty(N); hi = np.empty(N)
        for i in range(N):
            if i < n: lo[i] = th[k] + Tall[O(k), i]
            else:     lo[i] = th[k] + Tall[O(k), i - n] + S[i - n] + Tall[i - n, i]
        top = max(th[k], e.max()) + (S + Tbar).sum()
        hi[:] = top
        lo = np.maximum(lo, e)
        return lo, hi

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", gap)
    inf = highspy.kHighsInf
    var = {}
    def add(name, lo, hi, cost, integer):
        idx = h.getNumCol()
        h.addVar(lo, hi)
        h.changeColCost(idx, cost)
        if integer: h.changeColIntegrality(idx, highspy.HighsVarType.kInteger)
        var[name] = idx
        return idx

    # arcs: origin->pickup, stop->stop, delivery->terminal (terminal = virtual, no row)
    arcs = {}
    for k in range(K):
        for j in range(n):                                   # o_k -> pickup
            arcs[('o', j, k)] = add(('x', 'o', j, k), 0, 1, Tall[O(k), j], True)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                if j == i + n:                 # pickup -> own delivery is fine
                    pass
                if i >= n and j == i - n:      # delivery -> own pickup: never
                    continue
                arcs[(i, j, k)] = add(('x', i, j, k), 0, 1, Tall[i, j], True)
        for i in range(n, N):                                # delivery -> terminal, zero cost
            arcs[(i, 'e', k)] = add(('x', i, 'e', k), 0, 1, 0.0, True)
        add(('u', k), 0, 1, phi, True)
        lo, hi = bounds(k)
        for i in range(N):
            add(('b', i, k), lo[i], hi[i], 0.0, False)
            add(('y', i, k), max(0, q[i]), min(Q[k], Q[k] + q[i]), 0.0, False)
            add(('tp', i, k), 0, inf, rho, False)
        add(('bo', k), th[k], hi.max(), 0.0, False)

    def row(coefs, lo, hi):
        idx = [var[c] for c, _ in coefs]; val = [v for _, v in coefs]
        h.addRow(lo, hi, len(idx), idx, val)

    # each stop served exactly once across all vehicles (outgoing)
    for i in range(N):
        cs = []
        for k in range(K):
            for j in range(N):
                if (i, j, k) in arcs: cs.append((('x', i, j, k), 1.0))
            if i >= n: cs.append((('x', i, 'e', k), 1.0))
        row(cs, 1, 1)
    # flow conservation per stop per vehicle
    for k in range(K):
        for i in range(N):
            ins = [(('x', j, i, k), 1.0) for j in range(N) if (j, i, k) in arcs]
            if i < n: ins.append((('x', 'o', i, k), 1.0))
            outs = [(('x', i, j, k), -1.0) for j in range(N) if (i, j, k) in arcs]
            if i >= n: outs.append((('x', i, 'e', k), -1.0))
            row(ins + outs, 0, 0)
        # origin leaves at most once, and only if dispatched
        row([(('x', 'o', j, k), 1.0) for j in range(n)] + [(('u', k), -1.0)], -inf, 0)
        # same vehicle serves pickup and its delivery
        for r in range(n):
            pk = [(('x', r, j, k), 1.0) for j in range(N) if (r, j, k) in arcs]
            dl = [(('x', r + n, j, k), -1.0) for j in range(N) if (r + n, j, k) in arcs]
            dl.append((('x', r + n, 'e', k), -1.0))
            row(pk + dl, 0, 0)
    # timing, load, precedence, penalties
    for k in range(K):
        lo, hi = bounds(k)
        for j in range(n):                                   # origin -> pickup
            M = hi.max() + Tall[O(k), j]                 # bo <= hi.max(), b_j >= lo[j] >= 0
            # b_j >= bo + T_oj - M(1 - x)   <=>   b_j - bo - M x >= T_oj - M
            row([(('b', j, k), 1.0), (('bo', k), -1.0), (('x', 'o', j, k), -M)],
                Tall[O(k), j] - M, inf)
        for (i, j, kk), _ in arcs.items():
            if kk != k or i == 'o' or j == 'e': continue
            M = max(0, hi[i] + S[i] + Tall[i, j] - lo[j])
            # b_j >= b_i + S_i + T_ij - M(1 - x)
            row([(('b', j, k), 1.0), (('b', i, k), -1.0), (('x', i, j, k), -M)],
                S[i] + Tall[i, j] - M, inf)
            W = min(Q[k], Q[k] + q[j])
            row([(('y', j, k), 1.0), (('y', i, k), -1.0), (('x', i, j, k), -W)],
                q[j] - W, inf)
            row([(('y', j, k), 1.0), (('y', i, k), -1.0), (('x', i, j, k), W)],
                -inf, q[j] + W)
        for j in range(n):                                   # first pickup leaves with load 1
            # y_j <= 1 + Q(1 - x_oj)   <=>   y_j + Q x_oj <= 1 + Q
            row([(('y', j, k), 1.0), (('x', 'o', j, k), Q[k])], -inf, 1 + Q[k])
        for r in range(n):                                   # precedence
            row([(('b', r + n, k), 1.0), (('b', r, k), -1.0)],
                S[r] + Tall[r, r + n], inf)
        for i in range(N):                                   # tardiness
            row([(('tp', i, k), 1.0), (('b', i, k), -1.0)], -l[i], inf)

    t0 = time.time()
    h.run()
    secs = time.time() - t0
    st = h.getModelStatus()
    info = h.getInfo()
    cost = info.objective_function_value
    bound = info.mip_dual_bound
    status = h.modelStatusToString(st)
    return cost, bound, status, secs


# ======================================================== B2: OR-Tools ====
def ortools_solve(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, time_limit=30):
    """
    OR-Tools routing in a subprocess (its bundled HiGHS conflicts with highspy).
    Routes are re-evaluated with our exact scheduler so costs are comparable.
    Returns (cost, travel, penalty, used, seconds) or None.
    """
    import subprocess, json, tempfile, os, sys
    n = inst.n_req; K = inst.n_veh; N = 2 * n
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, xp=inst.xp, xd=inst.xd, ehp=ehp, lhp=lhp, ehd=ehd, lhd=lhd,
                 sp=inst.sp, sd=inst.sd, origins=inst.origins, theta=inst.theta,
                 cap=inst.cap, Tall=Tall, rho=rho, phi=phi, time_limit=time_limit)
        path = f.name
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, os.path.join(here, "ortools_worker.py"), path],
                         capture_output=True, text=True, timeout=time_limit + 60)
    os.unlink(path)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    if not res.get("ok"):
        return None
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    routing.set_nhalf(n)
    travel = pen = 0.0; used = 0
    for k, seq in enumerate(res["routes"]):
        if not seq: continue
        used += 1
        tr, pn, _ = routing._evaluate(seq, Tall, e, l, S, inst.theta[k], N + k, rho)
        travel += tr; pen += pn
    return travel + rho * pen + phi * used, travel, pen, used, res["secs"]


# ===================================================== B4: single-level ====
def single_level(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, seed=0, iters=30):
    """
    Bradley-style capacity-constrained k-means directly to vehicles: one
    centroid per vehicle, requests assigned by the same network-flow (P2) with
    s = identity, plain (unanchored) centroid update, then exact routing.
    Isolates the value of the intermediate cluster layer.
    """
    import cluster
    from scipy.optimize import linprog
    rng = np.random.default_rng(seed)
    n = inst.n_req; K = inst.n_veh
    routing.set_nhalf(n)
    psi = np.hstack([inst.xp, inst.xd])
    iv = np.stack([ehp, lhd], 1)
    width = iv[:, 1] - iv[:, 0]; dbar = np.median(width) if np.median(width) > 0 else 1.0
    w = dbar / (dbar + width)
    C_s, C_t = cluster._seed(psi, iv, K, rng)
    s = np.arange(K)
    a = cluster._greedy_assign(psi, iv, w, C_s, C_t, s, inst.cap, K, 1.0, 1.0)
    for _ in range(iters):
        Crc = cluster.request_cost(psi, iv, w, C_s, C_t, 1.0, 1.0)
        a_new = cluster.solve_p2(Crc, s, inst.cap, K, set())
        if a_new is None or np.array_equal(a_new, a):
            break
        a = a_new
        for c in range(K):                                  # plain mean
            m = np.where(a == c)[0]
            if len(m):
                C_s[c] = psi[m].mean(0); C_t[c] = iv[m].mean()
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    travel = pen = 0.0; used = 0
    for k in range(K):
        req = np.where(a == k)[0]
        if len(req) == 0: continue
        used += 1
        tr, pn, _ = routing.solve_route(list(req), Tall, e, l, S, None, None,
                                        inst.theta[k], 2 * n + k, rho)
        travel += tr; pen += pn
    return travel + rho * pen + phi * used, travel, pen, used
