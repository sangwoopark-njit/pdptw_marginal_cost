"""
Instance-level structural metrics, computed from the data alone.

`pressure` (tardiness/travel) was a property of the SOLUTION, so two methods
on one instance could report different values.  These are properties of the
INSTANCE, so they stratify honestly.

  spatial_scatter   mean pairwise distance among all stops          (as in the paper)
  mixing_degree     fraction of requests whose pickup and delivery
                    lie within delta                                (as in the paper)
  temporal_conflict fraction of request pairs whose intervals OVERLAP.
                    Replaces `max l - min e`, which measured the horizon
                    rather than the difficulty.  High conflict means many
                    requests compete for the same courier-time.
  slack_ratio       available courier-time / required work.  < 1 means the
                    fleet provably cannot meet every window, so the instance
                    measures who minimises a forced penalty.
  cap_tightness     |R| / total capacity.  Near 1 means capacity binds.
"""
from __future__ import annotations
import numpy as np


def describe(inst, ehp, lhp, ehd, lhd, Tall, delta=10.0):
    n = inst.n_req
    pts = np.vstack([inst.xp, inst.xd])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    spatial = float(d[np.triu_indices(len(pts), 1)].mean())
    mixing = float((np.sqrt(((inst.xp - inst.xd) ** 2).sum(1)) <= delta).mean())

    a, b = ehp, lhd                                   # request interval I_r
    overlap = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            if not (b[i] < a[j] or b[j] < a[i]):
                overlap += 1
    conflict = overlap / max(tot, 1)

    span = float(b.max() - min(a.min(), inst.theta.min()))
    avail = span * inst.n_veh
    svc = float(np.concatenate([inst.sp, inst.sd]).sum())
    hop = float(np.median(d[np.triu_indices(len(pts), 1)]))
    work = svc + 2 * n * hop
    slack = avail / max(work, 1e-9)

    return dict(spatial_scatter=spatial, mixing_degree=mixing,
                temporal_conflict=conflict, slack_ratio=slack,
                cap_tightness=n / max(inst.cap.sum(), 1e-9),
                mean_interval=float((b - a).mean()))
