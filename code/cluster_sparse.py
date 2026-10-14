"""
The single-level method with candidate lists, so it scales.

WHY.  `cluster_flat.py` builds a dense |R| x |K| assignment: every request may go
to every courier.  Measured growth is ~|R|^2.5 for the marginal cost matrix and
worse for the integer program, whose size is |R||K| binaries.  At |R| = 1000 with
|K| = 300 that is 300,000 binaries and a 1000 x 300,300 constraint matrix, which
is hopeless whatever hardware is available.

Nothing is lost by refusing most of those variables.  A request is never assigned
to a courier on the other side of the map, so for each request we keep only the
`cand` most promising couriers.  That makes

    binaries            |R| |K|      ->  cand |R| + |K|
    marginal cost calls |R| |K|      ->  cand |R|
    relocate candidates |R| |K|      ->  cand |R|
    swap candidates     |R|^2 / 2    ->  requests sharing a candidate courier

all linear in |R| for fixed `cand`, and the constraint matrix is built sparse so
it is never materialised densely.

The candidate set is rebuilt every outer iteration from the current solution: a
courier's position proxy is the mean midpoint of the requests it currently holds,
or its own origin when it holds none, and a request keeps the `cand` nearest.  So the
list follows the solution rather than being fixed in advance.

WHAT IS PARALLELISABLE, with one core per courier:
    marginal costs   independent per courier   -> divides
    exact routing    independent per courier   -> divides
    descent screen   independent per candidate -> divides
    the MIP          not parallel this way     -> the serial floor
`profile` on the returned dict reports each, so the projection can be made from
measurement rather than assumption.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

import routing
from cluster2 import make_evaluator, chain_eval, chain_order, _route_all


def candidate_lists(inst, lab, cand):
    """For each request, the `cand` couriers whose current work is nearest.

    A courier holding nothing has no current work to measure from, so its own
    origin stands in -- otherwise it would enter no candidate list, could never
    be opened, and the fleet could only ever shrink.  (There is no depot in this
    model: every courier starts wherever its last assignment left it.)
    """
    n, K = inst.n_req, inst.n_veh
    mid = 0.5 * (inst.xp + inst.xd)
    pos = np.empty((K, 2))
    for k in range(K):
        m = np.where(lab == k)[0]
        pos[k] = mid[m].mean(0) if len(m) else inst.origins[k]
    d = ((mid[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    c = min(cand, K)
    idx = np.argpartition(d, c - 1, axis=1)[:, :c]
    # always keep the incumbent, so the current solution stays feasible
    for r in range(n):
        if lab[r] not in idx[r]:
            idx[r, int(np.argmax(d[r, idx[r]]))] = lab[r]
    return np.sort(idx, axis=1)


def sparse_marginals(lab, idx, inst, ev):
    """Marginal cost of each allowed (request, courier) pair only."""
    n = inst.n_req
    members = {k: [int(x) for x in np.where(lab == k)[0]]
               for k in range(inst.n_veh)}
    base = {k: ev(k, members[k]) for k in members}
    out = np.empty(idx.shape)
    for r in range(n):
        for j, k in enumerate(idx[r]):
            k = int(k)
            if lab[r] == k:
                out[r, j] = base[k] - ev(k, [x for x in members[k] if x != r])
            else:
                out[r, j] = ev(k, members[k] + [r]) - base[k]
    return out


def solve_sparse(C, idx, cap, phi, K, time_limit=30.0):
    """
    min sum_(r,k in candidates) C_rk x_rk + phi sum_k u_k
    s.t. sum_k x_rk = 1 for every r,  sum_r x_rk <= Q_k u_k for every k

    Built with sparse matrices: the dense form has |R| x (cand|R| + |K|) entries
    and is the reason the dense version cannot be scaled.
    """
    n, c = idx.shape
    nx = n * c
    cost = np.concatenate([C.ravel(), np.full(K, float(phi))])

    rows = np.repeat(np.arange(n), c)
    cols = np.arange(nx)
    A_eq = sparse.csr_matrix((np.ones(nx), (rows, cols)), shape=(n, nx + K))

    kr = idx.ravel()
    A_cap = sparse.csr_matrix((np.ones(nx), (kr, cols)), shape=(K, nx + K))
    A_u = sparse.csr_matrix((-np.asarray(cap, float), (np.arange(K), np.arange(K))),
                            shape=(K, K))
    A_ub = sparse.hstack([A_cap[:, :nx], A_u], format="csr")

    res = milp(c=cost,
               constraints=[LinearConstraint(A_eq, 1, 1),
                            LinearConstraint(A_ub, -np.inf, 0)],
               integrality=np.ones(nx + K),
               bounds=Bounds(0, 1),
               options=dict(time_limit=time_limit))
    if res.x is None:
        return None, False
    x = np.asarray(res.x[:nx]).reshape(n, c)
    return idx[np.arange(n), x.argmax(axis=1)], bool(res.status == 0)


def sparse_descent(lab, idx, inst, ev_scr, ev_dec, top=8, max_moves=100000):
    """Relocate and swap, restricted to candidate couriers.

    Swaps are only considered between two requests when each is a candidate for
    the other's courier, which is what keeps this linear in |R| rather than
    quadratic.
    """
    n, K = inst.n_req, inst.n_veh
    lab = lab.copy()
    load = np.bincount(lab, minlength=K)
    members = {k: set(np.where(lab == k)[0].tolist()) for k in range(K)}
    allowed = [set(int(k) for k in idx[r]) for r in range(n)]

    for _ in range(max_moves):
        cands = []
        for r in range(n):
            k0 = int(lab[r])
            rest0 = sorted(members[k0] - {r})
            for k1 in allowed[r]:
                if k1 == k0 or load[k1] + 1 > inst.cap[k1]:
                    continue
                m1 = sorted(members[k1])
                d = ((ev_scr(k0, rest0) + ev_scr(k1, m1 + [r]))
                     - (ev_scr(k0, sorted(members[k0])) + ev_scr(k1, m1)))
                cands.append((d, "move", (r, k0, k1)))
        for r1 in range(n):
            k1 = int(lab[r1])
            for k2 in allowed[r1]:
                if k2 == k1:
                    continue
                for r2 in members[k2]:
                    if k1 not in allowed[r2] or r2 <= r1:
                        continue
                    l1 = sorted((members[k1] - {r1}) | {r2})
                    l2 = sorted((members[k2] - {r2}) | {r1})
                    d = ((ev_scr(k1, l1) + ev_scr(k2, l2))
                         - (ev_scr(k1, sorted(members[k1]))
                            + ev_scr(k2, sorted(members[k2]))))
                    cands.append((d, "swap", (r1, r2, k1, k2)))

        cands.sort(key=lambda z: z[0])
        best = None
        for d, kind, pay in cands[:top]:
            if kind == "move":
                r, k0, k1 = pay
                new = (ev_dec(k0, sorted(members[k0] - {r}))
                       + ev_dec(k1, sorted(members[k1] | {r})))
                cur = ev_dec(k0, sorted(members[k0])) + ev_dec(k1, sorted(members[k1]))
            else:
                r1, r2, k1, k2 = pay
                new = (ev_dec(k1, sorted((members[k1] - {r1}) | {r2}))
                       + ev_dec(k2, sorted((members[k2] - {r2}) | {r1})))
                cur = ev_dec(k1, sorted(members[k1])) + ev_dec(k2, sorted(members[k2]))
            delta = new - cur
            if delta < -1e-9 and (best is None or delta < best[0]):
                best = (delta, kind, pay)

        if best is None:
            break
        if best[1] == "move":
            r, k0, k1 = best[2]
            members[k0].discard(r); members[k1].add(r)
            load[k0] -= 1; load[k1] += 1; lab[r] = k1
        else:
            r1, r2, k1, k2 = best[2]
            members[k1].discard(r1); members[k1].add(r2)
            members[k2].discard(r2); members[k2].add(r1)
            lab[r1], lab[r2] = k2, k1
    return lab


def run(inst, ehp, lhp, ehd, lhd, Tall, Top, *, rho=1.0, phi=0.0, T_out=4,
        seed=0, cand=8, polish=True, mip_limit=30.0, top=8, assign='mip',
        lag_iters=150, cache=None, trace=None, **_ignored):
    """`assign='mip'` solves the coupled assignment exactly (one integer program
    per outer iteration); `assign='lagrangian'` prices the coupling constraint and
    lets each courier solve its own subproblem, which is the parallel form.

    Pass a list as `trace` to record the assignment after every stage of every
    outer iteration; the figure script uses it to draw the method working."""
    rng = np.random.default_rng(seed)
    n, K = inst.n_req, inst.n_veh
    routing.set_nhalf(n)
    if cache is None:
        cache = {}

    e_all = np.concatenate([ehp, ehd])
    l_all = np.concatenate([lhp, lhd])
    S_all = np.concatenate([inst.sp, inst.sd])
    ev = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                        'insert', 0.0)
    ev_scr = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                            'chain', phi)
    ev_dec = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                            'insert', phi)

    # construction: nearest courier with room, in release order
    mid = 0.5 * (inst.xp + inst.xd)
    lab = np.full(n, -1)
    load = np.zeros(K)
    for r in np.argsort(ehp + 1e-6 * rng.random(n)):
        d = ((mid[r] - inst.origins) ** 2).sum(1)
        for k in np.argsort(d):
            if load[k] + 1 <= inst.cap[k]:
                lab[r] = k; load[k] += 1
                break
    s_id = np.arange(K)

    prof = dict(candidates=0.0, marginal=0.0, mip=0.0, descent=0.0, routing=0.0)
    best = None
    mip_calls = mip_closed = 0
    nx = 0

    def _snap(stage, labels, it):
        if trace is None:
            return
        tr_, pen_, used_, _ = _route_all(labels, s_id, K, inst, Tall, e_all,
                                         l_all, S_all, rho, phi, cache)
        trace.append(dict(stage=stage, iter=it, labels=labels.copy(),
                          travel=tr_, tardiness=pen_, couriers=used_,
                          cost=tr_ + rho * pen_ + phi * used_))

    _snap("construction", lab, -1)

    for it in range(T_out):
        t0 = time.time(); idx = candidate_lists(inst, lab, cand)
        prof["candidates"] += time.time() - t0
        t0 = time.time(); C = sparse_marginals(lab, idx, inst, ev)
        prof["marginal"] += time.time() - t0
        nx = idx.size
        t0 = time.time()
        if assign in ('lagrangian', 'lag_lp', 'lag_mip'):
            import lagrangian as LG
            fn = {'lagrangian': LG.solve, 'lag_lp': LG.solve_lag_lp,
                  'lag_mip': LG.solve_lag_mip}[assign]
            new, info = fn(C, idx, inst.cap, phi, K, iters=lag_iters)
            closed = bool(info.get("closed", False))
            lag_gap = info["gap"]
        else:
            new, closed = solve_sparse(C, idx, inst.cap, phi, K, mip_limit)
            lag_gap = np.nan
        prof["mip"] += time.time() - t0
        mip_calls += 1; mip_closed += int(closed)
        if new is not None:
            lab = new
        _snap("assignment", lab, it)
        if polish:
            t0 = time.time()
            lab = sparse_descent(lab, idx, inst, ev_scr, ev_dec, top)
            prof["descent"] += time.time() - t0
        _snap("descent", lab, it)
        t0 = time.time()
        tr, pen, used, _pi = _route_all(lab, s_id, K, inst, Tall, e_all, l_all,
                                        S_all, rho, phi, cache)
        prof["routing"] += time.time() - t0
        cost = tr + rho * pen + phi * used
        if best is None or cost < best["cost"] - 1e-9:
            best = dict(cost=cost, travel=tr, penalty=pen, used=used,
                        a=lab.copy(), s=s_id.copy(), iter=it, n_cl_final=K)

    best["profile"] = prof
    best["n_binaries"] = nx + K
    best["mip_calls"] = mip_calls
    best["mip_closed"] = mip_closed
    best["cand"] = cand
    best["assign"] = assign
    return best
