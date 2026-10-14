"""Grouping by pairwise chaining slack, in both symmetrisations.

Section III-B of the paper argues that a natural repair to spatio-temporal
proximity --- penalise pairs a courier cannot chain --- is still the wrong
object, and that the usual symmetrisation makes it worse.  Writing

    slack(r -> r') = (e_{p_{r'}} - l_{d_r}) - T(d_r, p_{r'})

and charging the negative part, a courier CHOOSES the order, so a pair is
incompatible only when BOTH orders fail: the penalty is the min of the two
directions, not their sum.

The paper supported this with a correlation, which a referee reasonably called
thin.  This module turns both versions into actual grouping rules so they can be
measured on the same footing as the six published ones: same capacity repair,
same exact router, and each reported twice, as published and with our descent.

    slack_sum   the summed symmetrisation (what the earlier formulation used)
    slack_min   the corrected one, incompatible only if both orders fail
"""
from __future__ import annotations
import numpy as np

import clustering_baselines as CB


def pair_penalty(inst, ehp, lhd, Tall, mode):
    """Pairwise chaining penalty, (n x n), under either symmetrisation."""
    n = inst.n_req
    D = Tall[n:2 * n, 0:n]                 # delivery of r -> pickup of r'
    gap = ehp[None, :] - lhd[:, None]      # time available between them
    neg = np.maximum(0.0, D - gap)         # how far short of feasible
    np.fill_diagonal(neg, 0.0)
    if mode == "sum":
        return neg + neg.T
    if mode == "min":
        return np.minimum(neg, neg.T)
    raise ValueError(mode)


def _group(P, n_veh, cap, rng, iters=25):
    """Assign requests to couriers to minimise total within-group penalty.

    Alternates exactly as a k-means-type rule does: given the current groups,
    each request is re-assigned to the group it is cheapest to join, taking
    requests in order of how much the choice matters (the regret between best
    and second best) so that capacity goes to the requests that care most.
    """
    n = P.shape[0]
    lab = np.array([i % n_veh for i in rng.permutation(n)], dtype=int)
    order0 = rng.permutation(n)
    lab_full = np.empty(n, dtype=int)
    lab_full[order0] = lab
    lab = lab_full

    for _ in range(iters):
        cost = np.zeros((n, n_veh))
        for k in range(n_veh):
            mem = np.where(lab == k)[0]
            if len(mem):
                cost[:, k] = P[:, mem].sum(1)
        # a request already in k should not be charged against itself
        cost[np.arange(n), lab] -= 0.0

        srt = np.sort(cost, axis=1)
        regret = (srt[:, 1] - srt[:, 0]) if n_veh > 1 else np.zeros(n)
        new = np.full(n, -1, dtype=int)
        load = np.zeros(n_veh, dtype=int)
        for r in np.argsort(-regret):
            for k in np.argsort(cost[r]):
                if load[k] < cap[k]:
                    new[r] = k; load[k] += 1
                    break
            if new[r] < 0:                       # every courier full
                new[r] = int(np.argmin(load))
                load[new[r]] += 1
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _rule(mode):
    def f(inst, ehp, lhp, ehd, lhd, Tall, rho=1.0, phi=0.0, seed=0):
        rng = np.random.default_rng(seed)
        P = pair_penalty(inst, ehp, lhd, Tall, mode)
        lab = _group(P, inst.n_veh, inst.cap.astype(int), rng)
        over, bad = CB.violations(lab, inst)
        rep = CB._capacity_repair(lab, inst)
        m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                         CB._to_assign(rep, inst.n_veh), rho, phi)
        m.update(cap_over=over, cap_bad_veh=bad,
                 moves=int((lab != rep).sum()))
        m["labels"] = rep
        return m
    f.__name__ = f"slack_{mode}"
    f.__doc__ = f"Group by pairwise chaining slack, {mode} symmetrisation."
    return f


slack_sum = _rule("sum")
slack_min = _rule("min")

METHODS = {"slack_sum": slack_sum, "slack_min": slack_min}


def correlation_with_separation(inst, ehp, lhd, Tall):
    """How much does each symmetrisation just measure time separation?

    The claim in Section III-B is that the summed version is, to a good
    approximation, a measure of how far apart in time two requests are --- which
    is the very quantity the paper argues is the wrong one to minimise.
    """
    n = inst.n_req
    t = 0.5 * (ehp + lhd)
    sep = np.abs(t[:, None] - t[None, :])
    iu = np.triu_indices(n, 1)
    out = {}
    for mode in ("sum", "min"):
        P = pair_penalty(inst, ehp, lhd, Tall, mode)
        a, b = P[iu], sep[iu]
        out[mode] = (float(np.corrcoef(a, b)[0, 1])
                     if a.std() > 1e-12 else float("nan"))
    return out
