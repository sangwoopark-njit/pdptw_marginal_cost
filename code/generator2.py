"""
Spatio-temporal MISALIGNMENT generator.

Motivation.  Well-designed dispatch data has time windows that do not all
conflict -- a batch whose requests every one overlap in time offers nothing to
separate temporally, and every method reduces to spatial clustering.  Holding
conflict low, two axes remain:

  spatial clusteredness   how tightly requests group in space
  MISALIGNMENT            whether a request's spatial group predicts its
                          temporal group

Misalignment is the axis that should separate methods.  When spatial and
temporal structure agree, a purely spatial clustering (k-means, Fisher, sweep,
Bradley) already finds temporally coherent groups and nothing more is needed.
When they disagree, a spatial method groups requests that must be served hours
apart, and pays for it in tardiness; only a method clustering in the joint
space can trade the two off.

`alignment` in [0,1]:  1.0 = a request's temporal band equals its spatial hub
                       0.0 = band drawn independently of hub
The realised misalignment is measured, not assumed, and returned.
"""
from __future__ import annotations
import numpy as np
from instance import Instance


def generate(n_req=24, n_veh=4, seed=0, *, n_hubs=4, hub_sigma=5.0, extent=100.0,
             alignment=1.0, n_bands=None, band_gap=140.0, window_width=45.0,
             service=8.0, pd_gap=35.0, cap_slack=1.3, courier_spread=8.0,
             stagger=0.05, delivery_mode="local"):
    rng = np.random.default_rng(seed)
    n_bands = n_bands or n_hubs

    hubs = rng.uniform(0.12 * extent, 0.88 * extent, size=(n_hubs, 2))
    hub_of = rng.integers(0, n_hubs, size=n_req)

    # temporal band: equals the spatial hub with probability `alignment`
    keep = rng.random(n_req) < alignment
    rand_band = rng.integers(0, n_bands, size=n_req)
    band_of = np.where(keep, hub_of % n_bands, rand_band)

    xp = np.clip(hubs[hub_of] + rng.normal(0, hub_sigma, (n_req, 2)), 0, extent)
    if delivery_mode == "local":
        xd = np.clip(xp + rng.normal(0, hub_sigma, (n_req, 2)), 0, extent)
    else:
        other = (hub_of + rng.integers(1, max(2, n_hubs), n_req)) % n_hubs
        xd = np.clip(hubs[other] + rng.normal(0, hub_sigma, (n_req, 2)), 0, extent)

    # bands are separated by band_gap, so conflict stays low by construction
    centre = band_of * band_gap + rng.uniform(0, window_width, n_req)
    ep = centre
    lp = centre + window_width
    ed = centre + pd_gap
    ld = centre + pd_gap + window_width
    horizon = float(ld.max())

    origins = np.clip(hubs[rng.integers(0, n_hubs, n_veh)]
                      + rng.normal(0, courier_spread, (n_veh, 2)), 0, extent)
    theta = rng.uniform(0, stagger * horizon, n_veh)
    cap = np.full(n_veh, max(1, int(np.ceil(cap_slack * n_req / n_veh))), float)

    inst = Instance(name=f"mis_a{alignment:g}_h{n_hubs}",
                    n_req=n_req, xp=xp, xd=xd, ep=ep, lp=lp, ed=ed, ld=ld,
                    sp=np.full(n_req, service), sd=np.full(n_req, service),
                    origins=origins, theta=theta, cap=cap, horizon=horizon)
    return inst, measure_misalignment(inst, ep, ld, n_hubs)


def measure_misalignment(inst, ep, ld, k):
    """1 - (best agreement between the spatial and temporal k-way partitions).
    0 means the two structures coincide; near 1 means they are unrelated."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from itertools import permutations
    mid = 0.5 * (inst.xp + inst.xd)
    t = 0.5 * (ep + ld)
    sl = fcluster(linkage(mid, "ward"), t=k, criterion="maxclust")
    tl = fcluster(linkage(t.reshape(-1, 1), "ward"), t=k, criterion="maxclust")
    n = len(sl)
    best = max(sum(sl == np.array([p[x - 1] for x in tl]))
               for p in permutations(range(1, k + 1)))
    return 1.0 - best / n
