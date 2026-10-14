"""
B3: Adaptive Large Neighbourhood Search, Ropke & Pisinger (2006) style,
on the `alns` package.  Solutions are per-vehicle sequences of stop rows
(pickup r is row r, delivery r is row r + n), evaluated by the same exact
scheduler as every other method so costs are comparable.

Destroy: random removal, worst removal, Shaw (relatedness) removal.
Repair : greedy insertion, regret-2 insertion.
Acceptance: simulated annealing.  Stop: time limit.
"""
from __future__ import annotations
import numpy as np, time, copy
import routing


class PDPState:
    def __init__(self, routes, unassigned, ctx):
        self.routes = routes            # list of lists of stop rows, one per vehicle
        self.unassigned = unassigned    # set of request ids
        self.ctx = ctx

    def copy(self):
        return PDPState([r[:] for r in self.routes], set(self.unassigned), self.ctx)

    def objective(self):
        c = self.ctx
        tot = 0.0
        for k, seq in enumerate(self.routes):
            if not seq:
                continue
            tr, pn, _ = routing._evaluate(seq, c.T, c.e, c.l, c.S, c.theta[k], c.N + k, c.rho)
            tot += tr + c.rho * pn + c.phi
        return tot + 1e6 * len(self.unassigned)       # unassigned is never acceptable


class Ctx:
    pass


def _route_cost(ctx, k, seq):
    if not seq:
        return 0.0
    tr, pn, _ = routing._evaluate(seq, ctx.T, ctx.e, ctx.l, ctx.S, ctx.theta[k], ctx.N + k, ctx.rho)
    return tr + ctx.rho * pn + ctx.phi


def _load_ok(ctx, seq, k):
    """Capacity feasibility of a route.

    The paper's model caps SIMULTANEOUS load, which is what this checks by
    default and what every earlier experiment used.  Set `ctx.count_cap = True`
    to cap the TOTAL number of requests on a courier instead, which is the
    stricter rule every cluster-first method is bound by: a cluster is assigned
    as a whole, before any sequence exists to count concurrency on.  The two
    rules only diverge when it pays to pile requests onto few couriers, i.e.
    exactly when a dispatch charge is present -- without the flag ALNS can close
    couriers the clustering formulation is not allowed to close, and a
    fleet-sizing comparison would measure that difference instead of the
    methods.
    """
    load = 0
    for s in seq:
        load += 1 if s < ctx.n else -1
        if load > ctx.cap[k]:
            return False
    if getattr(ctx, "count_cap", False):
        if sum(1 for s in seq if s < ctx.n) > ctx.cap[k]:
            return False
    return True


def _best_insertion(ctx, state, r):
    """Cheapest feasible insertion of request r over all vehicles.  Returns
    (delta, k, i, j) or None."""
    best = None
    p, d = r, r + ctx.n
    for k, seq in enumerate(state.routes):
        base = _route_cost(ctx, k, seq)
        for i in range(len(seq) + 1):
            for j in range(i + 1, len(seq) + 2):
                cand = seq[:i] + [p] + seq[i:j - 1] + [d] + seq[j - 1:]
                if not _load_ok(ctx, cand, k):
                    continue
                delta = _route_cost(ctx, k, cand) - base
                if best is None or delta < best[0]:
                    best = (delta, k, i, j)
    return best


# ------------------------------------------------------------ destroy ----
def random_removal(state, rng, frac=0.25):
    s = state.copy(); ctx = s.ctx
    served = [r for r in range(ctx.n) if r not in s.unassigned]
    k = max(1, int(frac * len(served)))
    for r in rng.choice(served, size=min(k, len(served)), replace=False):
        _remove(s, int(r))
    return s


def worst_removal(state, rng, frac=0.25):
    s = state.copy(); ctx = s.ctx
    served = [r for r in range(ctx.n) if r not in s.unassigned]
    gains = []
    for r in served:
        t = s.copy(); _remove(t, r)
        gains.append((state.objective() - t.objective(), r))
    gains.sort(reverse=True)
    k = max(1, int(frac * len(served)))
    for _, r in gains[:k]:
        _remove(s, r)
    return s


def shaw_removal(state, rng, frac=0.25):
    s = state.copy(); ctx = s.ctx
    served = [r for r in range(ctx.n) if r not in s.unassigned]
    if not served:
        return s
    seed = int(rng.choice(served))
    def rel(a, b):
        return (ctx.T[a, b] + ctx.T[a + ctx.n, b + ctx.n]
                + 0.1 * abs(ctx.e[a] - ctx.e[b]))
    order = sorted(served, key=lambda r: rel(seed, r))
    k = max(1, int(frac * len(served)))
    for r in order[:k]:
        _remove(s, r)
    return s


def _remove(state, r):
    for seq in state.routes:
        if r in seq:
            seq.remove(r); seq.remove(r + state.ctx.n)
            break
    state.unassigned.add(r)


# ------------------------------------------------------------- repair ----
def greedy_repair(state, rng):
    s = state.copy()
    while s.unassigned:
        r = int(rng.choice(sorted(s.unassigned)))
        ins = _best_insertion(s.ctx, s, r)
        if ins is None:
            break
        _, k, i, j = ins
        seq = s.routes[k]
        s.routes[k] = seq[:i] + [r] + seq[i:j - 1] + [r + s.ctx.n] + seq[j - 1:]
        s.unassigned.discard(r)
    return s


def regret_repair(state, rng):
    s = state.copy(); ctx = s.ctx
    while s.unassigned:
        best = None
        for r in sorted(s.unassigned):
            opts = []
            p, d = r, r + ctx.n
            for k, seq in enumerate(s.routes):
                base = _route_cost(ctx, k, seq)
                for i in range(len(seq) + 1):
                    for j in range(i + 1, len(seq) + 2):
                        cand = seq[:i] + [p] + seq[i:j - 1] + [d] + seq[j - 1:]
                        if _load_ok(ctx, cand, k):
                            opts.append((_route_cost(ctx, k, cand) - base, k, i, j))
            if not opts:
                continue
            opts.sort()
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else 1e9
            if best is None or regret > best[0]:
                best = (regret, r, opts[0])
        if best is None:
            break
        _, r, (_, k, i, j) = best
        seq = s.routes[k]
        s.routes[k] = seq[:i] + [r] + seq[i:j - 1] + [r + ctx.n] + seq[j - 1:]
        s.unassigned.discard(r)
    return s


# --------------------------------------------------------------- driver ----
def alns_solve(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, time_limit=30, seed=0,
               count_cap=False):
    """Returns (cost, travel, penalty, used, seconds).

    `count_cap=True` makes ALNS obey the same total-count capacity the
    clustering methods obey; see `_load_ok`.  Default False preserves every
    earlier result.
    """
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    from alns.stop import MaxRuntime

    ctx = Ctx()
    ctx.n = inst.n_req; ctx.N = 2 * inst.n_req; ctx.T = Tall
    ctx.e = np.concatenate([ehp, ehd]); ctx.l = np.concatenate([lhp, lhd])
    ctx.S = np.concatenate([inst.sp, inst.sd]); ctx.theta = inst.theta
    ctx.cap = inst.cap; ctx.rho = rho; ctx.phi = phi
    ctx.count_cap = count_cap
    routing.set_nhalf(ctx.n)

    rng = np.random.default_rng(seed)
    init = greedy_repair(PDPState([[] for _ in range(inst.n_veh)], set(range(ctx.n)), ctx), rng)

    alns = ALNS(rng)
    for f in (random_removal, worst_removal, shaw_removal):
        alns.add_destroy_operator(f)
    for f in (greedy_repair, regret_repair):
        alns.add_repair_operator(f)
    select = RouletteWheel([3, 2, 1, 0.5], 0.8, 3, 2)
    accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.5, 2000)
    t0 = time.time()
    res = alns.iterate(init, select, accept, MaxRuntime(time_limit))
    secs = time.time() - t0
    best = res.best_state
    travel = pen = 0.0; used = 0
    for k, seq in enumerate(best.routes):
        if not seq:
            continue
        used += 1
        tr, pn, _ = routing._evaluate(seq, ctx.T, ctx.e, ctx.l, ctx.S, ctx.theta[k], ctx.N + k, rho)
        travel += tr; pen += pn
    return travel + rho * pen + phi * used, travel, pen, used, secs
