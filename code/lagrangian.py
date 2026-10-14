"""
The assignment step as K independent per-courier subproblems, coupled only by
prices.

The exact assignment solved in cluster_sparse is one integer program:

    min  sum_rk C_rk x_rk + phi sum_k u_k
    s.t. sum_k x_rk = 1          for every request   <- the only coupling
         sum_r x_rk <= Q_k u_k   for every courier

Dualising the coupling constraint with a price lambda_r per request gives

    L(lambda) = sum_r lambda_r
              + sum_k [ min over x,u of  sum_r (C_rk - lambda_r) x_rk + phi u_k ]

and the bracket involves only courier k's own variables.  Each courier therefore
solves its own problem: take the Q_k requests whose REDUCED cost C_rk - lambda_r
is most negative, and open if their total beats -phi.  That is a sort, it is
independent of every other courier, and it is what makes the step parallel.

WHAT THIS DOES NOT DO.  It does not enforce that each request is taken exactly
once -- that constraint is gone from the feasible set and lives in the objective
instead.  Couriers may claim the same request or none may claim it, so:

  * L(lambda) is a LOWER BOUND on the assignment optimum, not a solution;
  * prices are adjusted by subgradient steps, lambda_r += step * (1 - claims_r),
    which cheapens a request that two couriers wanted and dearens one nobody
    wanted -- an auction;
  * a repair pass turns the claims into a feasible assignment, and the best
    repaired solution over the iterations is what is returned.

Because this is an integer program there is a duality gap, so the repaired
solution can be worse than the exact one.  `solve` reports the bound, the primal
value and the gap so the loss is measured rather than assumed.
"""
from __future__ import annotations
import numpy as np


def _courier_index(idx, K):
    """For each courier, the positions in the flattened candidate list it owns."""
    flat_k = idx.ravel()
    order = np.argsort(flat_k, kind="stable")
    bounds = np.searchsorted(flat_k[order], np.arange(K + 1))
    return order, bounds


def subproblems(rc, order, bounds, cap, phi, K, n_cand):
    """
    Every courier's own problem, given reduced costs.  Independent across k:
    this is the loop that a machine with one core per courier would run in
    parallel.  Returns (total value, chosen positions, claims per request).
    """
    total = 0.0
    chosen = []
    for k in range(K):
        lo, hi = bounds[k], bounds[k + 1]
        if lo == hi:
            continue
        pos = order[lo:hi]
        vals = rc[pos]
        neg = np.where(vals < 0)[0]
        if neg.size == 0:
            continue                       # nothing worth taking: stay closed
        q = int(cap[k])
        if neg.size > q:                   # keep the q most negative
            keep = neg[np.argpartition(vals[neg], q - 1)[:q]]
        else:
            keep = neg
        val = float(vals[keep].sum()) + phi
        if val < 0.0:                      # opening pays for itself
            total += val
            chosen.append(pos[keep])
    return total, (np.concatenate(chosen) if chosen else np.empty(0, int))


def repair(claim_count, C, idx, cap, phi, K, n):
    """
    Turn claims into a feasible assignment.

    Requests wanted by exactly one courier go there if it has room; contested and
    orphaned requests go to their cheapest candidate with room.  Requests are
    processed in order of how much their best and second-best options differ, so
    the ones with the most to lose choose first.
    """
    load = np.zeros(K)
    lab = np.full(n, -1)
    order_c = np.argsort(C, axis=1)
    regret = np.where(C.shape[1] > 1,
                      C[np.arange(n), order_c[:, min(1, C.shape[1] - 1)]]
                      - C[np.arange(n), order_c[:, 0]], 0.0)
    for r in np.argsort(-regret):
        for j in order_c[r]:
            k = int(idx[r, j])
            if load[k] + 1 <= cap[k]:
                lab[r] = k
                load[k] += 1
                break
        if lab[r] < 0:
            # every candidate is full.  Eject one of their members that has a
            # free candidate of its own, rather than assigning r off-list, which
            # would leave it with no C_rk and corrupt the objective.
            done = False
            for j in order_c[r]:
                k = int(idx[r, j])
                for r2 in np.where(lab == k)[0]:
                    for j2 in order_c[r2]:
                        k2 = int(idx[r2, j2])
                        if k2 != k and load[k2] + 1 <= cap[k2]:
                            lab[r2] = k2
                            load[k2] += 1
                            lab[r] = k
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if not done:                   # genuinely nowhere to go
                k = int(idx[r, int(np.argmin(C[r]))])
                lab[r] = k
                load[k] += 1
    return lab


def primal_value(lab, C, idx, phi):
    """Objective of a feasible assignment, on the same costs the MIP minimises.

    A request assigned outside its candidate list has no C_rk.  Scoring it as
    zero would make such a solution look BETTER the more it breaks the candidate
    structure, so it is charged the worst cost in its row instead -- conservative,
    and it keeps the comparison with the exact program honest.
    """
    n, c = idx.shape
    tot = 0.0
    off = 0
    for r in range(n):
        j = np.where(idx[r] == lab[r])[0]
        if j.size:
            tot += float(C[r, j[0]])
        else:
            tot += float(C[r].max())
            off += 1
    return tot + phi * len(set(lab.tolist()))


def solve(C, idx, cap, phi, K, iters=150, alpha=2.0, patience=20, seed=0):
    """
    Lagrangian assignment.  Returns (labels, info) where info carries the best
    bound, the best primal value and the gap between them.

    `iters` subgradient steps; each one runs K independent subproblems and then a
    single serial price update.
    """
    n, c = idx.shape
    order, bounds = _courier_index(idx, K)
    flat_C = C.ravel()
    flat_r = np.repeat(np.arange(n), c)

    lam = np.zeros(n)
    best_bound = -np.inf
    best_lab, best_primal = None, np.inf
    stale = 0

    for t in range(iters):
        rc = flat_C - lam[flat_r]
        val, picked = subproblems(rc, order, bounds, cap, phi, K, c)
        bound = float(lam.sum()) + val
        if bound > best_bound:
            best_bound = bound

        claims = np.bincount(flat_r[picked], minlength=n) if picked.size else np.zeros(n, int)

        lab = repair(claims, C, idx, cap, phi, K, n)
        pv = primal_value(lab, C, idx, phi)
        if pv < best_primal:
            best_primal, best_lab, stale = pv, lab, 0
        else:
            stale += 1

        g = 1.0 - claims
        gn = float((g ** 2).sum())
        if gn < 1e-12:
            break                          # every request claimed exactly once
        step = alpha * max(best_primal - bound, 1e-6) / gn
        lam = lam + step * g

        if stale >= patience:
            alpha *= 0.5
            stale = 0
            if alpha < 1e-3:
                break

    gap = (100 * (best_primal - best_bound) / abs(best_primal)
           if np.isfinite(best_primal) and best_primal != 0 else np.nan)
    return best_lab, dict(bound=best_bound, primal=best_primal, gap=gap,
                          iters=t + 1)


# ---------------------------------------------------------------------------
# Two ways to spend the Lagrangian solution, instead of accepting its 11% loss.
# ---------------------------------------------------------------------------
from scipy import sparse as _sp
from scipy.optimize import linprog as _linprog


def fleet_from(lab, K):
    """The set of couriers a solution opens."""
    return sorted(set(int(x) for x in lab.tolist()))


def assignment_given_fleet(C, idx, cap, open_k, n, K):
    """
    Optimal assignment with the OPEN SET FIXED.

    Once u is fixed the dispatch charge is a constant, and what remains

        min sum_rk C_rk x_rk   s.t.  sum_k x_rk = 1,  sum_r x_rk <= Q_k

    is a transportation problem: totally unimodular, so the LP relaxation has an
    integral optimum and no branching is needed.  This is exactly the structure
    Proposition 2 relies on, recovered by taking the fixed charge out of the
    program rather than by adding a cluster layer.

    Couriers are added to `open_k` if some request has no open candidate, or if
    the open capacity cannot hold every request, so the LP is always feasible.
    """
    n_, c = idx.shape
    openset = set(int(k) for k in open_k)

    for r in range(n):
        if not any(int(k) in openset for k in idx[r]):
            openset.add(int(idx[r, int(np.argmin(C[r]))]))
    while sum(cap[k] for k in openset) < n:
        rest = [k for k in range(K) if k not in openset]
        if not rest:
            break
        openset.add(int(rest[int(np.argmax([cap[k] for k in rest]))]))

    mask = np.array([[int(idx[r, j]) in openset for j in range(c)]
                     for r in range(n)])
    rr, jj = np.where(mask)
    nx = rr.size
    kk = idx[rr, jj]
    cost = C[rr, jj]

    A_eq = _sp.csr_matrix((np.ones(nx), (rr, np.arange(nx))), shape=(n, nx))
    klist = sorted(openset)
    kpos = {k: i for i, k in enumerate(klist)}
    rows = np.array([kpos[int(k)] for k in kk])
    A_ub = _sp.csr_matrix((np.ones(nx), (rows, np.arange(nx))),
                          shape=(len(klist), nx))
    b_ub = np.array([cap[k] for k in klist], float)

    res = _linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(n),
                   bounds=(0, 1), method="highs")
    if not res.success:
        return None, len(openset)
    x = np.asarray(res.x)
    lab = np.full(n, -1)
    for r in range(n):
        sel = np.where(rr == r)[0]
        lab[r] = int(kk[sel[int(np.argmax(x[sel]))]])
    return lab, len(openset)


def solve_lag_lp(C, idx, cap, phi, K, iters=150, widen=2):
    """
    Lagrangian chooses the fleet, an LP then assigns optimally to it.

    The fleet size is also tried a little smaller and larger than the Lagrangian
    suggested, since that is the decision its duality gap is worst on, and each
    trial is only an LP.
    """
    lab0, info = solve(C, idx, cap, phi, K, iters=iters)
    n = idx.shape[0]
    base = fleet_from(lab0, K)
    best, best_val, best_k = lab0, primal_value(lab0, C, idx, phi), len(base)

    loads = np.bincount(lab0, minlength=K)
    ranked = sorted(range(K), key=lambda k: -loads[k])
    for delta in range(-widen, widen + 1):
        size = len(base) + delta
        if size < 1 or size > K:
            continue
        trial = ranked[:size]
        lab, nopen = assignment_given_fleet(C, idx, cap, trial, n, K)
        if lab is None:
            continue
        val = primal_value(lab, C, idx, phi)
        if val < best_val:
            best, best_val, best_k = lab, val, nopen
    info.update(primal=best_val, fleet=best_k)
    return best, info


def solve_lag_mip(C, idx, cap, phi, K, iters=150, keep=3, time_limit=5.0):
    """
    Lagrangian reduced costs narrow the candidate list, then the exact program is
    solved on what is left.  Each request keeps only the `keep` couriers with the
    best reduced cost, so the integer program shrinks by a factor of cand/keep.
    This is a heuristic fixing: it can remove an optimal pair, so it is measured
    rather than claimed to be exact.
    """
    from cluster_sparse import solve_sparse
    lab0, info = solve(C, idx, cap, phi, K, iters=iters)
    n, c = idx.shape
    k_keep = min(keep, c)
    sub = np.argsort(C, axis=1)[:, :k_keep]
    for r in range(n):                       # never drop the incumbent
        if lab0[r] not in idx[r, sub[r]]:
            where = np.where(idx[r] == lab0[r])[0]
            if where.size:                   # it may be off-list after a repair
                sub[r, -1] = int(where[0])
    idx2 = np.take_along_axis(idx, sub, axis=1)
    C2 = np.take_along_axis(C, sub, axis=1)
    lab, closed = solve_sparse(C2, idx2, cap, phi, K, time_limit)
    if lab is None:
        return lab0, info
    info.update(primal=primal_value(lab, C2, idx2, phi), closed=closed)
    return lab, info
