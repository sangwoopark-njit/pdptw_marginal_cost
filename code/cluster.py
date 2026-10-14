"""
Bi-level capacity- and time-aware clustering.

Inner loop   (P1) clusters -> vehicles, (P2) requests -> clusters,
             anchored centroid update, repeated to convergence.
Outer loop   recompute the cluster costs, run the inner loop, route each
             courier exactly, then optionally repair.

The two repairs are independently switchable, which is the point of the
experiment:
    use_exclusion  forbid the worst-served requests from their current cluster
    use_splitting  split temporally incoherent clusters in two
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import routing


# --------------------------------------------------------------- costs ----
def incompat_matrix(a, n_cl, ehp, lhd, Tall, n):
    """
    Chaining incompatibility of request r with cluster c, computed from the
    CURRENT assignment and then held fixed while (P2) is solved -- the same
    device already used for C_ck, so (P2) stays a fixed-cost network flow.

        slack(r -> r') = (e_p(r') - l_d(r)) - T(d_r, p_r')

    is the spare time between finishing r and starting r'.  Negative slack is
    travel the courier cannot physically make, and is what turns into
    tardiness; positive slack is only idling and is not charged.
    """
    D = Tall[n:2 * n, 0:n]                     # delivery of r -> pickup of r'
    gap = ehp[None, :] - lhd[:, None]          # gap[r, r'] = e_p(r') - l_d(r)
    neg = np.maximum(0.0, D - gap)             # infeasible travel, both orders
    sym = neg + neg.T
    np.fill_diagonal(sym, 0.0)
    out = np.zeros((n, n_cl))
    for c in range(n_cl):
        mem = np.where(a == c)[0]
        if len(mem):
            out[:, c] = sym[:, mem].sum(1)
    return out


def request_cost(psi_s, iv, w, C_s, C_t, ws, wt, incompat=None, wc=0.0):
    """C_rc of the paper: weighted spatial squared distance + interval distance."""
    d2 = ((psi_s[:, None, :] - C_s[None, :, :]) ** 2).sum(-1)      # (n_req, n_cl)
    lo, hi = iv[:, 0][:, None], iv[:, 1][:, None]
    dt = np.maximum(0.0, np.maximum(lo - C_t[None, :], C_t[None, :] - hi))
    base = ws ** 2 * d2 + wt ** 2 * dt ** 2
    if incompat is not None and wc > 0:
        base = ws ** 2 * d2 + wc * incompat      # compatibility replaces D_t
    return w[:, None] * base


def tour_estimate(pts, mode="mst"):
    """
    Internal cost L(c): an estimate of the travel needed to serve a cluster.

    mst    minimum spanning tree.  A lower bound on the optimal tour, and a
           loose one: it ignores that a tour must return through its stops.
    nn     nearest-neighbour tour.  A feasible tour, hence an upper bound.
    exact  optimal open tour by held-karp.  Only for small clusters, which is
           the regime the method should run in anyway (nu small).
    bhh    Beardwood-Halton-Hammersley asymptotic, 0.7 sqrt(n A).
    """
    n = len(pts)
    if n <= 1:
        return 0.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    if mode == "mst":
        return mst_weight(pts)
    if mode == "nn":
        seen = np.zeros(n, bool); seen[0] = True; cur = 0; tot = 0.0
        for _ in range(n - 1):
            j = int(np.argmin(np.where(seen, np.inf, d[cur])))
            tot += d[cur, j]; seen[j] = True; cur = j
        return float(tot)
    if mode == "bhh":
        area = max((pts[:, 0].max() - pts[:, 0].min()) *
                   (pts[:, 1].max() - pts[:, 1].min()), 1e-9)
        return float(0.7 * np.sqrt(n * area))
    if mode == "tpath":
        # travel along the stops in TIME-WINDOW order.  The router serves a
        # cluster roughly in time order, so this estimates what it will pay --
        # unlike the MST, which ignores time, or D_t, which ignores geography.
        return None          # handled by cluster_cost, which has the windows
    if mode == "exact":
        if n > 9:
            return mst_weight(pts)
        best = {(1 << i, i): d[0, i] if i else 0.0 for i in range(n)}
        best = {(1, 0): 0.0}
        for _ in range(n - 1):
            nxt = {}
            for (mask, last), c in best.items():
                for j in range(n):
                    if mask >> j & 1:
                        continue
                    k = (mask | (1 << j), j); v = c + d[last, j]
                    if k not in nxt or v < nxt[k]:
                        nxt[k] = v
            best = nxt
        return float(min(best.values())) if best else 0.0
    raise ValueError(mode)


def mst_weight(pts):
    """Prim, on the complete Euclidean graph."""
    n = len(pts)
    if n <= 1:
        return 0.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    seen = np.zeros(n, bool); seen[0] = True
    best = d[0].copy(); best[0] = np.inf
    tot = 0.0
    for _ in range(n - 1):
        j = int(np.argmin(np.where(seen, np.inf, best)))
        tot += best[j]; seen[j] = True
        best = np.minimum(best, d[j])
    return float(tot)


def time_ordered_path(pts, times):
    o = np.argsort(times)
    q = pts[o]
    return float(np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1)).sum())


def cluster_cost(members, xp, xd, ehp, origins, theta, Top, beta, lmode="mst",
                 ehd=None):
    """
    C_ck = A(c,k) + beta * L(c).
    A(c,k) = min over pickups of [ max(theta_k + T, e_hat_p) - theta_k ]
    L(c)   = MST over the cluster's stops
    """
    n_cl = len(members)
    n_veh = len(theta)
    A = np.zeros((n_cl, n_veh))
    L = np.zeros(n_cl)
    for c, mem in enumerate(members):
        if len(mem) == 0:
            continue
        arr = np.maximum(theta[None, :] + Top[np.ix_(mem, range(n_veh))],
                         ehp[mem][:, None])                  # (|mem|, n_veh)
        A[c] = (arr - theta[None, :]).min(axis=0)
        if lmode == "tpath":
            pts = np.vstack([xp[mem], xd[mem]])
            tt = np.concatenate([ehp[mem], ehd[mem] if ehd is not None else ehp[mem]])
            L[c] = time_ordered_path(pts, tt)
        else:
            L[c] = tour_estimate(np.vstack([xp[mem], xd[mem]]), lmode)
    return A + beta * L[:, None]


def time_buckets(ehp, lhp, n_buckets, service, travel_hat):
    """
    Partition the batch horizon into `n_buckets` equal periods and assign each
    request to the one containing the midpoint of its tightened pickup window.
    Each request belongs to exactly one bucket, which is what keeps (P2) a pure
    network flow.

    The per-period capacity is derived, not tuned: within a period of width
    Delta a courier can complete at most Delta / (2 S + That) requests, where S
    is the mean service time and That the mean inter-stop travel.
    """
    mid = 0.5 * (ehp + lhp)
    lo, hi = mid.min(), mid.max()
    if n_buckets <= 1 or hi - lo < 1e-9:
        return np.zeros(len(mid), int), 1, np.inf
    edges = np.linspace(lo, hi + 1e-9, n_buckets + 1)
    b = np.clip(np.searchsorted(edges, mid, side="right") - 1, 0, n_buckets - 1)
    delta = (hi - lo) / n_buckets
    per = max(1.0, np.floor(delta / max(2 * service + travel_hat, 1e-9)))
    return b, n_buckets, per


# ------------------------------------------------------- (P1) and (P2) ----
def solve_p1(Cck, ncl_size, cap, phi, ncb=None, per_cap=None):
    """
    min  sum_ck C_ck z_ck + phi sum_k u_k
    s.t. sum_k z_ck = 1                          for every cluster
         sum_c n_c z_ck        <= Q_k u_k        aggregate capacity
         sum_c n_cb z_ck       <= kappa_k u_k    for every period b   (if ncb given)

    `ncb[c, b]` counts the requests of cluster c falling in period b.  The
    aggregate row is the single-period special case.  Still a linear integer
    program with |C||K| binaries; only the number of rows grows.
    """
    n_cl, n_veh = Cck.shape
    nz = n_cl * n_veh
    c = np.concatenate([Cck.ravel(), np.full(n_veh, phi)])
    A_eq = np.zeros((n_cl, nz + n_veh))
    for i in range(n_cl):
        A_eq[i, i * n_veh:(i + 1) * n_veh] = 1.0
    A_ub = np.zeros((n_veh, nz + n_veh))
    for k in range(n_veh):
        for i in range(n_cl):
            A_ub[k, i * n_veh + k] = ncl_size[i]
        A_ub[k, nz + k] = -cap[k]
    cons = [LinearConstraint(A_eq, 1, 1), LinearConstraint(A_ub, -np.inf, 0)]
    if ncb is not None and per_cap is not None and np.isfinite(per_cap):
        nb = ncb.shape[1]
        A_per = np.zeros((n_veh * nb, nz + n_veh))
        for k in range(n_veh):
            for b in range(nb):
                row = k * nb + b
                for i in range(n_cl):
                    A_per[row, i * n_veh + k] = ncb[i, b]
                A_per[row, nz + k] = -per_cap
        cons.append(LinearConstraint(A_per, -np.inf, 0))
    res = milp(c=c, constraints=cons, integrality=np.ones(nz + n_veh),
               bounds=Bounds(0, 1))
    if not res.success:
        return None
    z = np.asarray(res.x[:nz]).reshape(n_cl, n_veh)
    return z.argmax(axis=1)


def solve_p2(Crc, s, cap, n_veh, forbidden, bucket=None, n_b=1, per_cap=None):
    """
    min  sum_rc C_rc z_rc
    s.t. sum_c z_rc = 1,  sum_{c in C(k)} sum_r z_rc <= Q_k
    A pure network flow under uniform loads; the LP relaxation is integral,
    so linprog suffices and the optimum is attained at a vertex.
    """
    n_req, n_cl = Crc.shape
    cost = Crc.copy()
    for (r, c) in forbidden:
        if c < n_cl:
            cost[r, c] = 1e9                     # effectively z_rc = 0
    nz = n_req * n_cl
    A_eq = np.zeros((n_req, nz))
    for r in range(n_req):
        A_eq[r, r * n_cl:(r + 1) * n_cl] = 1.0
    A_ub = np.zeros((n_veh, nz))
    for k in range(n_veh):
        cols = [c for c in range(n_cl) if s[c] == k]
        for r in range(n_req):
            for c in cols:
                A_ub[k, r * n_cl + c] = 1.0
    if bucket is not None and per_cap is not None and np.isfinite(per_cap):
        A_per = np.zeros((n_veh * n_b, nz))
        for k in range(n_veh):
            cols = [c for c in range(n_cl) if s[c] == k]
            for b in range(n_b):
                row = k * n_b + b
                for r in np.where(bucket == b)[0]:
                    for c in cols:
                        A_per[row, r * n_cl + c] = 1.0
        A_ub = np.vstack([A_ub, A_per])
        cap = np.concatenate([cap, np.full(n_veh * n_b, per_cap)])
    res = linprog(cost.ravel(), A_ub=A_ub, b_ub=cap,
                  A_eq=A_eq, b_eq=np.ones(n_req), bounds=(0, 1),
                  method="highs")
    if not res.success:
        return None
    return np.asarray(res.x).reshape(n_req, n_cl).argmax(axis=1)


def centroid_update(a, s, psi_s, iv, w, origins, theta, n_cl, mu, wt):
    """
    Anchored update: spatial coordinates are the anchored weighted mean,
    the temporal coordinate minimises the piecewise quadratic by a sweep
    over the sorted interval endpoints.
    """
    C_s = np.zeros((n_cl, psi_s.shape[1]))
    C_t = np.zeros(n_cl)
    for c in range(n_cl):
        mem = np.where(a == c)[0]
        v = origins[s[c]]
        anchor = np.concatenate([v, v])
        if len(mem) == 0:
            C_s[c] = anchor
            C_t[c] = theta[s[c]]
            continue
        W = w[mem].sum()
        C_s[c] = (w[mem][:, None] * psi_s[mem]).sum(0) + mu * anchor
        C_s[c] /= (W + mu)
        C_t[c] = _temporal_min(iv[mem], w[mem], mu, theta[s[c]])
    return C_s, C_t


def _temporal_min(iv, w, mu, th):
    """argmin  sum_r w_r dist^2(t, I_r) + mu (t - theta)^2  by breakpoint sweep."""
    bps = np.unique(np.concatenate([iv[:, 0], iv[:, 1], [th]]))
    def g(t):                                   # derivative / 2
        p = np.clip(t, iv[:, 0], iv[:, 1])
        return (w * (t - p)).sum() + mu * (t - th)
    lo, hi = bps[0] - 1.0, bps[-1] + 1.0
    if g(lo) > 0:
        return lo
    if g(hi) < 0:
        return hi
    for _ in range(200):                        # bisection; g is non-decreasing
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------ algorithm ---
def run(inst, ehp, lhp, ehd, lhd, Tall, Top, *, nu=6, mu=0.05, beta=0.3,
        rho=1.0, phi=0.0, ws=1.0, wt=1.0, kappa=0.15,
        T_out=8, use_exclusion=False, use_splitting=False, iv_mode='request',
        n_buckets=1, lmode='mst', wc=0.0,
        tenure=None, excess=True, seed=0, max_inner=30):
    """
    Returns a dict with the best cost found and diagnostics.
    Cost = total travel + rho * total penalty + phi * (couriers dispatched).
    """
    rng = np.random.default_rng(seed)
    n_req, n_veh = inst.n_req, inst.n_veh
    routing.set_nhalf(n_req)

    # request representation
    psi_s = np.hstack([inst.xp, inst.xd])                  # raw spatial coords
    if iv_mode == 'pickup':
        iv = np.stack([ehp, lhp], axis=1)                  # pickup window only
    else:
        iv = np.stack([ehp, lhd], axis=1)                  # release to deadline
    width = iv[:, 1] - iv[:, 0]
    dbar = np.median(width) if np.median(width) > 0 else 1.0
    w = dbar / (dbar + width)                              # tightness weights

    # |C| = ceil(|R| / nu).  No floor at |K|: allowing |C| < |K| is what makes
    # courier dispatch a real decision, since at most |C| couriers can be used.
    n_cl = max(1, int(np.ceil(n_req / nu)))

    # per-period (throughput) capacity
    travel_hat = float(np.median(Tall[:2 * n_req, :2 * n_req][
        np.triu_indices(2 * n_req, 1)])) if n_req > 1 else 0.0
    service_hat = float(np.concatenate([inst.sp, inst.sd]).mean())
    bucket, n_b, per_cap = time_buckets(ehp, lhp, n_buckets, service_hat, travel_hat)

    # ---- initialisation: k-means++ style seeding then a capacity-aware pass
    seeds = _seed(psi_s, iv, n_cl, rng)
    C_s, C_t = seeds
    s = np.arange(n_cl) % n_veh
    a = _greedy_assign(psi_s, iv, w, C_s, C_t, s, inst.cap, n_veh, ws, wt)

    # stop-level arrays for routing
    e_all = np.concatenate([ehp, ehd])
    l_all = np.concatenate([lhp, lhd])
    S_all = np.concatenate([inst.sp, inst.sd])

    best = None
    Z = {}                     # (r, c) -> outer iteration when it expires
    hist = []

    for it in range(T_out):
        members = [np.where(a == c)[0] for c in range(n_cl)]
        Cck = cluster_cost(members, inst.xp, inst.xd, ehp,
                           inst.origins, inst.theta, Top, beta, lmode, ehd)

        # ---- inner loop, C_ck held fixed
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
            inc = (incompat_matrix(a, n_cl, ehp, lhd, Tall, n_req)
                   if wc > 0 else None)
            Crc = request_cost(psi_s, iv, w, C_s, C_t, ws, wt, inc, wc)
            a_new = solve_p2(Crc, s_new, inst.cap, n_veh, forbid,
                             bucket, n_b, per_cap)
            if a_new is None:
                a_new = a
            C_s, C_t = centroid_update(a_new, s_new, psi_s, iv, w,
                                       inst.origins, inst.theta, n_cl, mu, wt)
            if np.array_equal(a_new, a) and np.array_equal(s_new, s):
                a, s = a_new, s_new
                break
            a, s = a_new, s_new

        # ---- route each courier exactly (clusters merged per vehicle)
        total_tr = total_pen = 0.0
        pi = np.zeros(n_req)
        used = 0
        for k in range(n_veh):
            cl = [c for c in range(n_cl) if s[c] == k]
            req = np.concatenate([np.where(a == c)[0] for c in cl]) if cl else np.array([], int)
            req = np.unique(req).astype(int)
            if len(req) == 0:
                continue
            used += 1
            tr, pen, seq = routing.solve_route(
                list(req), Tall, e_all, l_all, S_all, None, None,
                inst.theta[k], 2 * n_req + k, rho)
            total_tr += tr
            total_pen += pen
            pi[req] += _per_request_penalty(list(req), Tall, e_all, l_all, S_all,
                                            inst.theta[k], 2 * n_req + k, n_req, seq)
        cost = total_tr + rho * total_pen + phi * used
        hist.append(cost)
        if best is None or cost < best["cost"] - 1e-9:
            best = dict(cost=cost, travel=total_tr, penalty=total_pen,
                        used=used, a=a.copy(), s=s.copy(), iter=it)

        if it == T_out - 1:
            break

        # ---- repairs
        if use_exclusion:
            score = pi - _solo_penalty(inst, ehp, lhp, ehd, lhd, Tall, s, a,
                                       n_req, rho) if excess else pi
            k_ex = max(1, int(np.ceil(kappa * n_req)))
            worst = np.argsort(-score)[:k_ex]
            exp = it + 1 + (tenure if tenure else T_out)
            for r in worst:
                Z[(int(r), int(a[r]))] = exp

        if use_splitting:
            n_cl, a, s, C_s, C_t = _split(a, s, iv, pi, n_cl, C_s, C_t,
                                          psi_s, w, inst, mu, rng)

    best["history"] = hist
    best["n_cl_final"] = n_cl
    return best


# ------------------------------------------------------------- helpers ----
def _seed(psi_s, iv, n_cl, rng):
    mid = iv.mean(1)
    pts = np.hstack([psi_s, mid[:, None]])
    n = len(pts)
    first = rng.integers(n)
    idx = [first]
    d2 = ((pts - pts[first]) ** 2).sum(1)
    for _ in range(n_cl - 1):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        j = rng.choice(n, p=p)
        idx.append(int(j))
        d2 = np.minimum(d2, ((pts - pts[j]) ** 2).sum(1))
    return psi_s[idx].copy(), mid[idx].copy()


def _greedy_assign(psi_s, iv, w, C_s, C_t, s, cap, n_veh, ws, wt):
    """Capacity-aware nearest-centroid pass; exact under uniform loads."""
    Crc = request_cost(psi_s, iv, w, C_s, C_t, ws, wt)
    n_req, n_cl = Crc.shape
    load = np.zeros(n_veh)
    a = np.full(n_req, -1)
    for r in np.argsort(Crc.min(1)):
        for c in np.argsort(Crc[r]):
            if load[s[c]] + 1 <= cap[s[c]]:
                a[r] = c
                load[s[c]] += 1
                break
        if a[r] < 0:
            a[r] = int(np.argmin(Crc[r]))
    return a


def _per_request_penalty(req, T, e, l, S, o_time, o_row, n_req, seq):
    """Split the route penalty back onto individual requests."""
    t = o_time; prev = o_row
    out = np.zeros(len(req))
    pos = {r: i for i, r in enumerate(req)}
    for row in seq:
        t = max(t + T[prev, row], e[row])
        pen = max(0.0, t - l[row])
        r = row if row < n_req else row - n_req
        if r in pos:
            out[pos[r]] += pen
        t += S[row]
        prev = row
    return out


def _solo_penalty(inst, ehp, lhp, ehd, lhd, T, s, a, n_req, rho):
    """Penalty each request would incur served alone by its current courier."""
    out = np.zeros(n_req)
    for r in range(n_req):
        k = s[a[r]]
        o = 2 * n_req + k
        t = max(inst.theta[k] + T[o, r], ehp[r])
        out[r] += max(0.0, t - lhp[r])
        t += inst.sp[r] + T[r, r + n_req]
        t = max(t, ehd[r])
        out[r] += max(0.0, t - lhd[r])
    return out


def _split(a, s, iv, pi, n_cl, C_s, C_t, psi_s, w, inst, mu, rng):
    """Split the most penalised cluster in two by 1-D k-means on time."""
    per_cl = np.zeros(n_cl)
    for c in range(n_cl):
        mem = np.where(a == c)[0]
        per_cl[c] = pi[mem].sum() if len(mem) else 0.0
    if per_cl.max() <= 1e-9:
        return n_cl, a, s, C_s, C_t
    c = int(np.argmax(per_cl))
    mem = np.where(a == c)[0]
    if len(mem) < 2:
        return n_cl, a, s, C_s, C_t
    mid = iv[mem].mean(1)
    lo, hi = mid.min(), mid.max()
    if hi - lo < 1e-9:
        return n_cl, a, s, C_s, C_t
    cen = np.array([lo, hi])
    for _ in range(20):                       # 1-D Lloyd
        lab = np.argmin(np.abs(mid[:, None] - cen[None, :]), axis=1)
        for j in (0, 1):
            if (lab == j).any():
                cen[j] = mid[lab == j].mean()
    if len(set(lab)) < 2:
        return n_cl, a, s, C_s, C_t
    a = a.copy()
    a[mem[lab == 1]] = n_cl                   # the new cluster
    s = np.append(s, s[c])                    # both parts keep the parent vehicle
    C_s = np.vstack([C_s, C_s[c]])
    C_t = np.append(C_t, cen[1])
    C_t[c] = cen[0]
    return n_cl + 1, a, s, C_s, C_t
