"""
Bi-level clustering, second generation.

WHY THIS EXISTS.  The first version priced the temporal part of the
request-to-cluster cost C_rc in two ways, and both are wrong in the same
direction:

  D_t = dist^2(C_t^c, I_r)   pulls requests whose windows coincide into one
                             cluster, because it is a distance to a temporal
                             CENTROID.
  incompat (summed)          `sym = neg + neg.T` charges the negative slack of
                             BOTH orders of a pair.  A courier picks the order,
                             so this charges most for the pairs that chain most
                             easily: measured on the generated instances it
                             correlates +0.97 with |t_r - t_r'|.

A courier serves its cluster one stop after another.  Requests that must be
served at the same moment are therefore in CONTENTION, not in agreement, and
both terms above reward exactly that contention.  That is why dropping the
temporal term (omega_t = 0) beat both of them in every test.

WHAT REPLACES THEM.  A cluster is good when its requests CHAIN: when the
courier can leave one and reach the next inside its window.  That is a property
of a path, not of a ball, so the cost here is a marginal path cost

    Delta_chain(r, c) = J(members(c) + r, k) - J(members(c), k)

where J is travel + rho * tardiness of the time-ordered route of that request
set on the courier k = s(c), scheduled by the same earliest-feasible rule the
exact router uses.  It is computed from the CURRENT assignment and held fixed
while (P2) is solved, exactly as C_ck and the old incompat term were, so (P2)
remains a fixed-cost network flow and Proposition 2 is untouched.  Unlike D_t
it is also vehicle-aware: it knows where the courier is and when it is free.

Three further changes, each switchable so it can be ablated:

  init='chain'   seed the clusters by a time-ordered chaining pass instead of
                 k-means++.  The measured restart spread of the old method was
                 64.6% of cost on one instance -- larger than any effect any
                 cost term had -- so the seeding, not the cost function, was the
                 dominant quantity.
  restarts=R     keep the best of R runs by exact routed cost.
  polish=True    a descent on the request-to-vehicle assignment (relocate and
                 swap) scored by the same chain cost, before the final exact
                 routing.

`cluster.py` is left untouched so the earlier results stay reproducible.
"""
from __future__ import annotations
import numpy as np

import routing
from cluster import (solve_p1, solve_p2, centroid_update, cluster_cost,
                     request_cost, time_buckets, _per_request_penalty,
                     _solo_penalty, _split, _seed, _greedy_assign)


# ------------------------------------------------------------ chain cost ---
def chain_order(members, ehp, ehd, n_req):
    """Stop rows of `members` in time-window order, pickups before deliveries.

    Window tightening guarantees e_hat_d >= e_hat_p + S_p + T_pd > e_hat_p, so
    ordering the stops by their release time and breaking ties toward the
    pickup always yields a precedence-feasible sequence.  `check_chain_order`
    in the validation script tests that on random instances rather than
    trusting the argument.
    """
    rows, keys = [], []
    for r in members:
        rows.append(int(r)); keys.append((ehp[r], 0))
        rows.append(int(r) + n_req); keys.append((ehd[r], 1))
    o = sorted(range(len(rows)), key=lambda i: keys[i])
    return [rows[i] for i in o]


def chain_eval(seq, Tall, e_all, l_all, S_all, theta_k, o_row, rho):
    """travel + rho * tardiness of a fixed stop sequence, earliest-feasible."""
    t = theta_k
    prev = o_row
    travel = pen = 0.0
    for row in seq:
        d = Tall[prev, row]
        travel += d
        t = max(t + d, e_all[row])
        if t > l_all[row]:
            pen += t - l_all[row]
        t += S_all[row]
        prev = row
    return travel + rho * pen


def chain_matrix(a, s, n_cl, inst, ehp, ehd, e_all, l_all, S_all, Tall, rho):
    """
    Delta_chain[r, c] = marginal cost of adding r to cluster c, on c's courier.

    Computed from the current assignment and held fixed for (P2), so the
    subproblem stays a fixed-cost network flow.  A request already in c is
    priced by what it would cost to put it back, i.e. against the cluster
    WITHOUT it, so members and non-members are on the same scale -- otherwise
    (P2) would see a systematic discount for staying put.
    """
    n = inst.n_req
    out = np.zeros((n, n_cl))
    base = np.zeros(n_cl)
    members = [np.where(a == c)[0] for c in range(n_cl)]
    for c in range(n_cl):
        k = int(s[c])
        seq = chain_order(members[c], ehp, ehd, n)
        base[c] = chain_eval(seq, Tall, e_all, l_all, S_all, inst.theta[k],
                             2 * n + k, rho) if seq else 0.0
    for c in range(n_cl):
        k = int(s[c])
        mem = members[c]
        memset = set(int(x) for x in mem)
        for r in range(n):
            if r in memset:
                without = [x for x in mem if x != r]
                sq = chain_order(without, ehp, ehd, n)
                b = chain_eval(sq, Tall, e_all, l_all, S_all, inst.theta[k],
                               2 * n + k, rho) if sq else 0.0
                out[r, c] = base[c] - b
            else:
                sq = chain_order(list(mem) + [r], ehp, ehd, n)
                out[r, c] = chain_eval(sq, Tall, e_all, l_all, S_all,
                                       inst.theta[k], 2 * n + k, rho) - base[c]
    return out


def marginal_matrix(a, s, n_cl, n, ev):
    """
    Delta[r, c] = marginal cost of adding r to cluster c, under evaluator `ev`.

    Same object as chain_matrix but with the route cost supplied by `ev`, so the
    accurate insertion evaluator can be used for C_rc while the descent screens
    with the cheap one.  Members are priced against the cluster WITHOUT them, so
    members and non-members are on one scale.
    """
    out = np.zeros((n, n_cl))
    members = [[int(x) for x in np.where(a == c)[0]] for c in range(n_cl)]
    for c in range(n_cl):
        k = int(s[c])
        mem = members[c]
        base = ev(k, mem)
        memset = set(mem)
        for r in range(n):
            if r in memset:
                out[r, c] = base - ev(k, [x for x in mem if x != r])
            else:
                out[r, c] = ev(k, mem + [r]) - base
    return out


def insert_eval(mem, ehp, ehd, Tall, e_all, l_all, S_all, theta_k, o_row, rho,
                n):
    """
    Cost of serving `mem` on one courier, by cheapest insertion in time order.

    The time-ordered chain is a very loose stand-in for what the exact router
    will charge -- measured over 120 random request sets it is 2.42x the exact
    cost on average, because ordering by window ignores geography entirely and
    the route zig-zags.  Inserting each request at its cheapest position instead
    costs 709 us against the exact DP's 30 900 us and lands within 4.3% of it,
    with a rank correlation of 0.990.  Rank is what a descent needs.
    """
    order = sorted((int(r) for r in mem), key=lambda r: ehp[r])
    seq = []
    for r in order:
        best, bseq = None, None
        m = len(seq)
        for i in range(m + 1):
            for j in range(i, m + 1):
                cand = seq[:i] + [r] + seq[i:j] + [r + n] + seq[j:]
                v = chain_eval(cand, Tall, e_all, l_all, S_all, theta_k,
                               o_row, rho)
                if best is None or v < best:
                    best, bseq = v, cand
        seq = bseq
    return chain_eval(seq, Tall, e_all, l_all, S_all, theta_k, o_row, rho)


def insert_seq(mem, ehp, ehd, Tall, e_all, l_all, S_all, theta_k, o_row, rho,
               n):
    """The sequence cheapest insertion builds, so the repairs can charge
    tardiness to individual requests without calling the exact router."""
    order = sorted((int(r) for r in mem), key=lambda r: ehp[r])
    seq = []
    for r in order:
        best, bseq = None, None
        m = len(seq)
        for i in range(m + 1):
            for j in range(i, m + 1):
                cand = seq[:i] + [r] + seq[i:j] + [r + n] + seq[j:]
                v = chain_eval(cand, Tall, e_all, l_all, S_all, theta_k,
                               o_row, rho)
                if best is None or v < best:
                    best, bseq = v, cand
        seq = bseq
    return seq


def make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                   mode='insert', phi=0.0):
    """f(k, members) -> cost of that request set on courier k, memoised.

    mode='chain'  time-ordered sequence        ~1 us, rank 0.949
    mode='insert' cheapest insertion in time order ~709 us, rank 0.990
    Both use the same earliest-feasible schedule as the exact router.

    A NON-EMPTY courier also costs `phi`, the dispatch charge.  Without this the
    descent cannot see the saving from emptying a courier, and the whole
    fleet-sizing decision -- which is what (P1) exists to make -- would be
    invisible to every step after it.  With phi = 0 this is a no-op, which is
    why it did not matter in the earlier experiments.
    """
    n = inst.n_req
    memo = {}

    def f(k, members):
        if len(members) == 0:
            return 0.0
        key = (k, tuple(sorted(int(x) for x in members)))
        v = memo.get(key)
        if v is None:
            if mode == 'chain':
                v = chain_eval(chain_order(key[1], ehp, ehd, n), Tall, e_all,
                               l_all, S_all, inst.theta[k], 2 * n + k, rho)
            else:
                v = insert_eval(key[1], ehp, ehd, Tall, e_all, l_all, S_all,
                                inst.theta[k], 2 * n + k, rho, n)
            v += phi                      # dispatch charge for using courier k
            memo[key] = v
        return v

    f.memo = memo
    return f


def incompat_min(a, n_cl, ehp, lhd, Tall, n):
    """The compatibility term with the order bug removed.

        slack(r -> r') = (e_p(r') - l_d(r)) - T(d_r, p_r')

    A courier chooses the order, so the pair is incompatible only if BOTH
    orders fail: the penalty is min, not the sum of the two.
    """
    D = Tall[n:2 * n, 0:n]
    gap = ehp[None, :] - lhd[:, None]
    neg = np.maximum(0.0, D - gap)
    np.fill_diagonal(neg, 0.0)
    sym = np.minimum(neg, neg.T)
    out = np.zeros((n, n_cl))
    for c in range(n_cl):
        mem = np.where(a == c)[0]
        if len(mem):
            out[:, c] = sym[:, mem].sum(1)
    return out


# --------------------------------------------------------- initialisation ---
def init_chain(inst, ehp, ehd, e_all, l_all, S_all, Tall, n_cl, cap, n_veh,
               rho, rng, ev=None):
    """
    Seed by chaining rather than by k-means++.

    Sweep the requests in release-time order and put each one on the cluster
    whose current chain it extends most cheaply, subject to the courier
    capacity.  This produces clusters that are sequences from the start, which
    is what the router will make of them anyway; k-means++ seeding produces
    balls, and its draw was responsible for a 64.6% mean spread in cost.
    """
    n = inst.n_req
    s = np.arange(n_cl) % n_veh
    a = np.full(n, -1)
    members = [[] for _ in range(n_cl)]
    load = np.zeros(n_veh)
    order = np.argsort(ehp + 1e-6 * rng.random(n))
    for r in order:
        best, bestc = None, None
        for c in range(n_cl):
            k = int(s[c])
            if load[k] + 1 > cap[k]:
                continue
            b0 = chain_eval(chain_order(members[c], ehp, ehd, n), Tall, e_all,
                            l_all, S_all, inst.theta[k], 2 * n + k, rho) \
                if members[c] else 0.0
            b1 = chain_eval(chain_order(members[c] + [int(r)], ehp, ehd, n),
                            Tall, e_all, l_all, S_all, inst.theta[k],
                            2 * n + k, rho)
            if best is None or b1 - b0 < best:
                best, bestc = b1 - b0, c
        if bestc is None:                      # every courier full: least loaded
            bestc = int(np.argmin([load[int(s[c])] for c in range(n_cl)]))
        a[r] = bestc
        members[bestc].append(int(r))
        load[int(s[bestc])] += 1
    return a, s


# ------------------------------------------------------------- the polish ---
def polish_assignment(a, s, n_cl, inst, ehp, ehd, e_all, l_all, S_all, Tall,
                      rho, max_moves=200, cache=None, screen=None, decide=None,
                      top=8):
    """
    Descent on the request-to-cluster assignment.  Relocate and swap, best
    improvement, capacity respected, repeated until no improving move remains.

    TWO STAGE.  Every candidate move is screened with the ~1 us chain cost, and
    only the `top` most promising are re-scored with the 709 us insertion cost,
    which is what actually decides the move.  Screening alone misranks (rank
    correlation 0.949 against the exact router, and a level 2.4x too high);
    deciding with the accurate evaluator on everything would cost 40x more.

    Both evaluators use the same earliest-feasible schedule as the exact router,
    and the result is re-routed exactly afterwards, so this step only proposes.
    """
    n = inst.n_req
    n_veh = inst.n_veh
    if screen is None:
        screen = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                                'chain')
    if decide is None:
        decide = screen

    a = a.copy()
    cl_of_veh = {}
    for c in range(n_cl):
        cl_of_veh.setdefault(int(s[c]), c)

    req_of = {k: [int(r) for r in range(n) if s[a[r]] == k] for k in range(n_veh)}

    for _ in range(max_moves):
        cand = []
        for r in range(n):
            k0 = int(s[a[r]])
            rest0 = [x for x in req_of[k0] if x != r]
            for k1, c1 in cl_of_veh.items():
                if k1 == k0 or len(req_of[k1]) + 1 > inst.cap[k1]:
                    continue
                d = ((screen(k0, rest0) + screen(k1, req_of[k1] + [r]))
                     - (screen(k0, req_of[k0]) + screen(k1, req_of[k1])))
                cand.append((d, "move", (r, k0, k1, c1)))
        for r1 in range(n):
            k1 = int(s[a[r1]])
            for r2 in range(r1 + 1, n):
                k2 = int(s[a[r2]])
                if k1 == k2:
                    continue
                l1 = [x for x in req_of[k1] if x != r1] + [r2]
                l2 = [x for x in req_of[k2] if x != r2] + [r1]
                d = ((screen(k1, l1) + screen(k2, l2))
                     - (screen(k1, req_of[k1]) + screen(k2, req_of[k2])))
                cand.append((d, "swap", (r1, r2, k1, k2)))

        cand.sort(key=lambda z: z[0])
        best = None
        for d, kind, pay in cand[:top]:
            if kind == "move":
                r, k0, k1, c1 = pay
                rest0 = [x for x in req_of[k0] if x != r]
                new = decide(k0, rest0) + decide(k1, req_of[k1] + [r])
                cur = decide(k0, req_of[k0]) + decide(k1, req_of[k1])
            else:
                r1, r2, k1, k2 = pay
                l1 = [x for x in req_of[k1] if x != r1] + [r2]
                l2 = [x for x in req_of[k2] if x != r2] + [r1]
                new = decide(k1, l1) + decide(k2, l2)
                cur = decide(k1, req_of[k1]) + decide(k2, req_of[k2])
            delta = new - cur
            if delta < -1e-9 and (best is None or delta < best[0]):
                best = (delta, kind, pay)

        if best is None:
            break
        if best[1] == "move":
            r, k0, k1, c1 = best[2]
            req_of[k0] = [x for x in req_of[k0] if x != r]
            req_of[k1] = req_of[k1] + [r]
            a[r] = c1
        else:
            r1, r2, k1, k2 = best[2]
            l1 = [x for x in req_of[k1] if x != r1] + [r2]
            l2 = [x for x in req_of[k2] if x != r2] + [r1]
            a[r1], a[r2] = a[r2], a[r1]
            req_of[k1], req_of[k2] = l1, l2
    return a


def polish_labels(labels, inst, ehp, lhp, ehd, lhd, Tall, rho=1.0,
                  max_moves=200, phi=0.0):
    """
    Run the same descent on an EXTERNAL baseline's request-to-vehicle labels.

    Our method is an optimisation over assignments; sweep, Fisher, k-means and
    the rest are construction rules.  Comparing the two directly would credit
    our clustering with whatever the descent contributes, so every baseline is
    also reported with this applied.  |C| = |K| and s is the identity, which is
    the single-level special case of the same code.  Returns the new labels.
    """
    K = inst.n_veh
    e_all = np.concatenate([ehp, ehd])
    l_all = np.concatenate([lhp, lhd])
    S_all = np.concatenate([inst.sp, inst.sd])
    a = np.asarray(labels, int).copy()
    s = np.arange(K)
    ev = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                        'insert', phi)
    ev_scr = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                            'chain', phi)
    a2 = polish_assignment(a, s, K, inst, ehp, ehd, e_all, l_all, S_all, Tall,
                           rho, max_moves=max_moves, screen=ev_scr, decide=ev)
    return a2


def ils(inst, ehp, lhp, ehd, lhd, Tall, Top, *, nu=3, rho=1.0, phi=0.0,
        budget=5.0, seed=0, pool=6, kick=3, cache=None, **kw):
    """
    Anytime version: iterated local search in the ASSIGNMENT space.

    Plain restarts buy nothing once the seeding is chain-based, because it is
    near-deterministic and every restart lands in the same basin -- measured, not
    assumed.  What the budget can buy instead is perturbation: kick a few
    requests onto other couriers, re-run the descent, keep it if the chain cost
    improved.

    The search runs entirely on the chain cost, which is an upper bound on the
    exact routed cost and costs microseconds; only the `pool` best DISTINCT
    assignments found are routed exactly at the end, and the best of those by
    exact cost is returned.  So the exact router still decides the answer, and
    the cheap surrogate only decides what it is asked about.
    """
    import time as _time
    t0 = _time.time()
    if cache is None:
        cache = {}
    n, K = inst.n_req, inst.n_veh
    e_all = np.concatenate([ehp, ehd])
    l_all = np.concatenate([lhp, lhd])
    S_all = np.concatenate([inst.sp, inst.sd])
    rng = np.random.default_rng(seed)

    start = run(inst, ehp, lhp, ehd, lhd, Tall, Top, nu=nu, rho=rho, phi=phi,
                seed=seed, polish=True, cache=cache, **{**CHAIN_DEFAULTS, **kw})
    ev = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                        'insert', phi)
    ev_scr = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                            'chain', phi)
    a, s, n_cl = start["a"].copy(), start["s"].copy(), start["n_cl_final"]

    memo = {}

    def chain_total(aa):
        return sum(ev(k, [int(r) for r in range(n) if s[aa[r]] == k])
                   for k in range(K))

    cl_of_veh = {}
    for c in range(n_cl):
        cl_of_veh.setdefault(int(s[c]), c)

    best_a, best_v = a.copy(), chain_total(a)
    seen = {tuple(best_a.tolist()): best_v}
    iters = 0
    while _time.time() - t0 < budget * 0.80:
        cand = best_a.copy()
        for _ in range(int(rng.integers(1, kick + 1))):
            r = int(rng.integers(n))
            ks = [k for k in cl_of_veh
                  if k != int(s[cand[r]])
                  and sum(1 for x in range(n) if s[cand[x]] == k) < inst.cap[k]]
            if ks:
                cand[r] = cl_of_veh[ks[int(rng.integers(len(ks)))]]
        cand = polish_assignment(cand, s, n_cl, inst, ehp, ehd, e_all, l_all,
                                 S_all, Tall, rho, screen=ev_scr, decide=ev)
        v = chain_total(cand)
        seen[tuple(cand.tolist())] = v
        if v < best_v - 1e-9:
            best_a, best_v = cand.copy(), v
        iters += 1

    # the incumbent is ranked by the surrogate like everything else, so it can
    # fall outside the pool and be discarded; put it back explicitly, otherwise
    # spending more time could report a worse answer than spending none.
    top = sorted(seen.items(), key=lambda kv: kv[1])[:pool]
    for key in (tuple(start["a"].tolist()), tuple(best_a.tolist())):
        if key not in dict(top):
            top.append((key, seen.get(key, float("inf"))))
    best = None
    for key, _v in top:
        aa = np.array(key, int)
        tr, pen, used, _pi = _route_all(aa, s, n_cl, inst, Tall, e_all, l_all,
                                        S_all, rho, phi, cache)
        c = tr + rho * pen + phi * used
        if best is None or c < best["cost"] - 1e-9:
            best = dict(cost=c, travel=tr, penalty=pen, used=used, a=aa.copy(),
                        s=s.copy(), n_cl_final=n_cl)
    best["iters"] = iters
    best["pool"] = len(top)
    return best


CHAIN_DEFAULTS = dict(ws=0.0, wt=0.0, wchain=1.0, init='chain',
                      chain_refresh='outer')


# ---------------------------------------------------------------- the run ---
def _route_all(a, s, n_cl, inst, Tall, e_all, l_all, S_all, rho, phi, cache):
    """Exact routing of every courier, with a cache over request sets."""
    n = inst.n_req
    total_tr = total_pen = 0.0
    used = 0
    pi = np.zeros(n)
    for k in range(inst.n_veh):
        cl = [c for c in range(n_cl) if s[c] == k]
        req = np.concatenate([np.where(a == c)[0] for c in cl]) if cl else np.array([], int)
        req = np.unique(req).astype(int)
        if len(req) == 0:
            continue
        used += 1
        key = (k, tuple(req.tolist()))
        hit = cache.get(key) if cache is not None else None
        if hit is None:
            tr, pen, seq = routing.solve_route(
                list(req), Tall, e_all, l_all, S_all, None, None,
                inst.theta[k], 2 * n + k, rho)
            if cache is not None:
                cache[key] = (tr, pen, seq)
        else:
            tr, pen, seq = hit
        total_tr += tr
        total_pen += pen
        pi[req] += _per_request_penalty(list(req), Tall, e_all, l_all, S_all,
                                        inst.theta[k], 2 * n + k, n, seq)
    return total_tr, total_pen, used, pi


def _score_surrogate(a, s, n_cl, inst, ehp, ehd, e_all, l_all, S_all, Tall,
                     rho, ev):
    # note: `ev` already includes the dispatch charge per non-empty courier
    """Cost of an assignment under the insertion surrogate, plus the per-request
    tardiness it implies.  Used to rank the outer iterates cheaply so the exact
    router is spent only on the best of them."""
    n = inst.n_req
    tot = 0.0
    pi = np.zeros(n)
    used = 0
    for k in range(inst.n_veh):
        req = [int(r) for r in range(n) if s[a[r]] == k]
        if not req:
            continue
        used += 1
        tot += ev(k, req)
        seq = insert_seq(req, ehp, ehd, Tall, e_all, l_all, S_all,
                         inst.theta[k], 2 * n + k, rho, n)
        pi[req] += _per_request_penalty(req, Tall, e_all, l_all, S_all,
                                        inst.theta[k], 2 * n + k, n, seq)
    return tot, pi, used


def _one_run(inst, ehp, lhp, ehd, lhd, Tall, Top, *, nu, mu, beta, rho, phi,
             ws, wt, wc, wchain, kappa, T_out, use_exclusion, use_splitting,
             iv_mode, lmode, n_buckets, init, polish, seed, max_inner, cache,
             compat_mode, chain_refresh, eval_mode, screen_mode, top,
             route_mode, route_top, n_cl_fixed):
    rng = np.random.default_rng(seed)
    n_req, n_veh = inst.n_req, inst.n_veh
    routing.set_nhalf(n_req)

    psi_s = np.hstack([inst.xp, inst.xd])
    iv = (np.stack([ehp, lhp], axis=1) if iv_mode == 'pickup'
          else np.stack([ehp, lhd], axis=1))
    width = iv[:, 1] - iv[:, 0]
    dbar = np.median(width) if np.median(width) > 0 else 1.0
    w = dbar / (dbar + width)

    # |C| may be given directly: deriving it as ceil(|R|/nu) with integer nu
    # cannot express, say, 15 clusters for 40 requests, and |C| < |K| silently
    # caps the number of couriers that can ever be opened (see design.py).
    n_cl = int(n_cl_fixed) if n_cl_fixed else max(1, int(np.ceil(n_req / nu)))

    e_all = np.concatenate([ehp, ehd])
    l_all = np.concatenate([lhp, lhd])
    S_all = np.concatenate([inst.sp, inst.sd])

    ev_decide = make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                               eval_mode, phi)
    ev_screen = (ev_decide if screen_mode == eval_mode else
                 make_evaluator(inst, ehp, ehd, e_all, l_all, S_all, Tall, rho,
                                screen_mode, phi))

    travel_hat = float(np.median(Tall[:2 * n_req, :2 * n_req][
        np.triu_indices(2 * n_req, 1)])) if n_req > 1 else 0.0
    service_hat = float(np.concatenate([inst.sp, inst.sd]).mean())
    bucket, n_b, per_cap = time_buckets(ehp, lhp, n_buckets, service_hat, travel_hat)

    if init == 'chain':
        a, s = init_chain(inst, ehp, ehd, e_all, l_all, S_all, Tall, n_cl,
                          inst.cap, n_veh, rho, rng, ev=ev_decide)
        C_s, C_t = centroid_update(a, s, psi_s, iv, w, inst.origins,
                                   inst.theta, n_cl, mu, wt)
    else:
        C_s, C_t = _seed(psi_s, iv, n_cl, rng)
        s = np.arange(n_cl) % n_veh
        a = _greedy_assign(psi_s, iv, w, C_s, C_t, s, inst.cap, n_veh, ws, wt)

    best = None
    Z = {}
    hist = []
    iterates = []

    chain_cached = [None]

    for it in range(T_out):
        members = [np.where(a == c)[0] for c in range(n_cl)]
        Cck = cluster_cost(members, inst.xp, inst.xd, ehp, inst.origins,
                           inst.theta, Top, beta, lmode, ehd)
        chain_cached[0] = None          # recomputed once per outer iteration

        for _ in range(max_inner):
            sizes = np.array([len(np.where(a == c)[0]) for c in range(n_cl)], float)
            ncb = np.zeros((n_cl, n_b))
            for c in range(n_cl):
                m = np.where(a == c)[0]
                if len(m):
                    np.add.at(ncb[c], bucket[m], 1.0)
            s_new = solve_p1(Cck, sizes, inst.cap, phi, ncb, per_cap)
            if s_new is None:
                s_new = s
            forbid = {(r, c) for (r, c), exp in Z.items() if exp > it}

            Crc = request_cost(psi_s, iv, w, C_s, C_t, ws, wt)
            if wchain > 0:
                if chain_refresh == 'outer':
                    # computed once per outer iteration and held fixed for the
                    # whole inner loop, exactly as C_ck is -- this is what keeps
                    # the inner loop a sequence of fixed-cost subproblems.
                    if chain_cached[0] is None:
                        chain_cached[0] = marginal_matrix(a, s_new, n_cl,
                                                          n_req, ev_decide)
                    Ch = chain_cached[0]
                else:
                    Ch = marginal_matrix(a, s_new, n_cl, n_req, ev_decide)
                Crc = Crc + wchain * Ch
            elif wc > 0:
                inc = (incompat_min(a, n_cl, ehp, lhd, Tall, n_req)
                       if compat_mode == 'min'
                       else __import__('cluster').incompat_matrix(
                           a, n_cl, ehp, lhd, Tall, n_req))
                Crc = request_cost(psi_s, iv, w, C_s, C_t, ws, 0.0, inc, wc)

            a_new = solve_p2(Crc, s_new, inst.cap, n_veh, forbid,
                             bucket, n_b, per_cap)
            if a_new is None:
                a_new = a
            C_s, C_t = centroid_update(a_new, s_new, psi_s, iv, w,
                                       inst.origins, inst.theta, n_cl, mu, wt)
            done = np.array_equal(a_new, a) and np.array_equal(s_new, s)
            a, s = a_new, s_new
            if done:
                break

        if polish:
            a = polish_assignment(a, s, n_cl, inst, ehp, ehd, e_all, l_all,
                                  S_all, Tall, rho, screen=ev_screen,
                                  decide=ev_decide, top=top)

        if route_mode == 'final':
            # rank this iterate with the surrogate; the exact router is called
            # once, on the best few, after the loop.  The surrogate's rank
            # correlation with the exact cost is 0.990, and the exact router
            # still decides the answer.
            sc, pi, used = _score_surrogate(a, s, n_cl, inst, ehp, ehd, e_all,
                                            l_all, S_all, Tall, rho, ev_decide)
            iterates.append((sc, a.copy(), s.copy(), n_cl, it))
            hist.append(sc)
        else:
            tr, pen, used, pi = _route_all(a, s, n_cl, inst, Tall, e_all, l_all,
                                           S_all, rho, phi, cache)
            cost = tr + rho * pen + phi * used
            hist.append(cost)
            if best is None or cost < best["cost"] - 1e-9:
                best = dict(cost=cost, travel=tr, penalty=pen, used=used,
                            a=a.copy(), s=s.copy(), iter=it, n_cl_final=n_cl)

        if it == T_out - 1:
            break

        if use_exclusion:
            k_ex = max(1, int(np.ceil(kappa * n_req)))
            worst = np.argsort(-pi)[:k_ex]
            for r in worst:
                Z[(int(r), int(a[r]))] = it + 1 + T_out

        if use_splitting:
            n_cl, a, s, C_s, C_t = _split(a, s, iv, pi, n_cl, C_s, C_t,
                                          psi_s, w, inst, mu, rng)

    if route_mode == 'final':
        for sc, aa, ss, ncl, it in sorted(iterates, key=lambda z: z[0])[:route_top]:
            tr, pen, used, _pi = _route_all(aa, ss, ncl, inst, Tall, e_all,
                                            l_all, S_all, rho, phi, cache)
            cost = tr + rho * pen + phi * used
            if best is None or cost < best["cost"] - 1e-9:
                best = dict(cost=cost, travel=tr, penalty=pen, used=used,
                            a=aa.copy(), s=ss.copy(), iter=it, n_cl_final=ncl)
    best["history"] = hist
    return best


def run(inst, ehp, lhp, ehd, lhd, Tall, Top, *, nu=3, mu=0.05, beta=0.3,
        rho=1.0, phi=0.0, ws=1.0, wt=0.0, wc=0.0, wchain=0.0, kappa=0.15,
        T_out=6, use_exclusion=True, use_splitting=True, iv_mode='request',
        lmode='mst', n_buckets=1, init='kmeanspp', polish=False, seed=0,
        max_inner=30, restarts=1, compat_mode='min', chain_refresh='inner',
        eval_mode='insert', screen_mode='chain', top=8, route_mode='every',
        route_top=2, n_cl=None, cache=None, **_ignored):
    """Best of `restarts` runs, selected by exact routed cost."""
    if cache is None:
        cache = {}
    best = None
    for i in range(restarts):
        r = _one_run(inst, ehp, lhp, ehd, lhd, Tall, Top, nu=nu, mu=mu,
                     beta=beta, rho=rho, phi=phi, ws=ws, wt=wt, wc=wc,
                     wchain=wchain, kappa=kappa, T_out=T_out,
                     use_exclusion=use_exclusion, use_splitting=use_splitting,
                     iv_mode=iv_mode, lmode=lmode, n_buckets=n_buckets,
                     init=init, polish=polish, seed=seed + 997 * i,
                     max_inner=max_inner, cache=cache, compat_mode=compat_mode,
                     chain_refresh=chain_refresh, eval_mode=eval_mode,
                     screen_mode=screen_mode, top=top, route_mode=route_mode,
                     route_top=route_top, n_cl_fixed=n_cl)
        if best is None or r["cost"] < best["cost"] - 1e-9:
            best = r
    best["restarts"] = restarts
    return best
