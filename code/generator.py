"""
A structured instance generator, to complement Li & Lim rather than replace it.

WHY THIS EXISTS, AND THE HONESTY CONSTRAINT
-------------------------------------------
Li & Lim instances were designed for a different problem: a single depot, hard
windows, closed routes, heterogeneous loads, and no notion of a dispatch batch.
They cannot vary the structural properties this paper's theory says should
matter, and two of those properties cannot be reached from that data at all
(see `_rescale_windows` in campaign.py for one such dead end).

The purpose here is therefore to *characterise the regime* in which the
decomposition helps, not to manufacture a favourable comparison.  The
generator exposes knobs in both directions, and every experiment that uses it
must report the adversarial settings alongside the favourable ones.  An
instance family that only ever flatters the method is worthless as evidence.

Knobs, and what the theory predicts:

  n_hubs          number of spatial clusters of pickups.  Many well-separated
                  hubs should favour a clustering decomposition; n_hubs = 0
                  (uniform) should not.
  hub_sigma       spread within a hub.  Small = tight, natural clusters.
  temporal_press  scales window widths down, so tardiness matters more.
                  Prior runs show the method gains as this rises.
  disjoint_target target fraction of temporally disjoint request pairs, hit by
                  spreading requests apart in time while holding each
                  request's own pickup->delivery gap fixed.
  cap_slack       total capacity / |R|.  Near 1 makes capacity bind, which is
                  what the capacity-constrained clustering exists for.
  courier_spread  how far couriers start from the hubs.  0 = all at one depot.
  delivery_mode   'local'  deliveries near their pickup   (high mixing degree)
                  'cross'  deliveries in a different hub  (low mixing degree)
                  'uniform' deliveries anywhere

An ADVERSARIAL preset is provided and should always be run.
"""
from __future__ import annotations
import numpy as np
from instance import Instance


def generate(n_req=40, n_veh=8, seed=0, *,
             n_hubs=4, hub_sigma=4.0, extent=100.0,
             window_width=60.0, temporal_press=1.0,
             disjoint_target=0.4, service=10.0,
             pd_gap=40.0, cap_slack=1.3, courier_spread=0.0,
             delivery_mode="cross", stagger=0.05) -> Instance:
    """Build one structured instance.  Returns an `Instance` in the same form
    as `instance.build_instance`, so every downstream function accepts it."""
    rng = np.random.default_rng(seed)

    # ---- spatial layout
    if n_hubs <= 0:
        xp = rng.uniform(0, extent, size=(n_req, 2))
        hub_of = np.zeros(n_req, int)
        hubs = np.array([[extent / 2, extent / 2]])
    else:
        hubs = rng.uniform(0.1 * extent, 0.9 * extent, size=(n_hubs, 2))
        hub_of = rng.integers(0, n_hubs, size=n_req)
        xp = hubs[hub_of] + rng.normal(0, hub_sigma, size=(n_req, 2))
    xp = np.clip(xp, 0, extent)

    if delivery_mode == "local":
        xd = xp + rng.normal(0, hub_sigma, size=(n_req, 2))
    elif delivery_mode == "cross" and n_hubs > 1:
        other = (hub_of + rng.integers(1, max(2, n_hubs), size=n_req)) % n_hubs
        xd = hubs[other] + rng.normal(0, hub_sigma, size=(n_req, 2))
    else:
        xd = rng.uniform(0, extent, size=(n_req, 2))
    xd = np.clip(xd, 0, extent)

    # ---- temporal layout
    # each request has an internal pickup->delivery gap; requests are spread
    # apart in time by a factor chosen to hit the disjoint-pair target.
    width = max(2.0, window_width / max(temporal_press, 1e-9))
    base = rng.uniform(0, 1, size=n_req)                 # relative position in the batch
    gap = pd_gap * rng.uniform(0.7, 1.3, size=n_req)
    horizon_guess = (width + gap).mean()

    def build(spread):
        c = base * spread * horizon_guess
        ep = c
        lp = c + width
        ed = c + gap
        ld = c + gap + width
        return ep, lp, ed, ld

    def disjoint(spread):
        ep, _, _, ld = build(spread)
        n = len(ep); cnt = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1
                if ld[i] < ep[j] or ld[j] < ep[i]:
                    cnt += 1
        return cnt / max(tot, 1)

    lo, hi = 0.05, 60.0
    if disjoint(hi) < disjoint_target:
        spread = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if disjoint(mid) < disjoint_target:
                lo = mid
            else:
                hi = mid
        spread = 0.5 * (lo + hi)

    # SERVABILITY.  If the batch is squeezed into less time than the fleet needs
    # to do the work, tardiness becomes unavoidable and every method is merely
    # minimising a forced penalty -- the instance then measures nothing.  We
    # therefore widen the spread until it covers the per-courier workload.
    mean_leg = float(np.abs(xp[:, None, :] - xd[None, :, :]).sum(-1).mean()) / 2 + 1e-9
    work_per_courier = (n_req / max(n_veh, 1)) * (2 * service + mean_leg)
    min_spread = work_per_courier / max(horizon_guess, 1e-9)
    spread = max(spread, min_spread)
    ep, lp, ed, ld = build(spread)
    horizon = float(ld.max())

    # ---- couriers
    if courier_spread <= 0:
        origins = np.tile(hubs.mean(0), (n_veh, 1))
    else:
        base_pts = hubs[rng.integers(0, len(hubs), size=n_veh)]
        origins = base_pts + rng.normal(0, courier_spread, size=(n_veh, 2))
    origins = np.clip(origins, 0, extent)
    theta = rng.uniform(0, stagger * horizon, size=n_veh)

    cap_total = max(n_veh, int(np.ceil(cap_slack * n_req)))
    cap = np.full(n_veh, max(1, int(np.ceil(cap_total / n_veh))), dtype=float)

    return Instance(
        name=f"gen_h{n_hubs}_p{temporal_press:g}_d{disjoint_target:g}",
        n_req=n_req, xp=xp, xd=xd,
        ep=ep, lp=lp, ed=ed, ld=ld,
        sp=np.full(n_req, service), sd=np.full(n_req, service),
        origins=origins, theta=theta, cap=cap, horizon=horizon,
    )


# ---------------------------------------------------------------- presets --
PRESETS = {
    # the theory says these should favour a clustering decomposition
    "favourable": dict(n_hubs=6, hub_sigma=3.0, temporal_press=3.0,
                       disjoint_target=0.55, cap_slack=1.15,
                       courier_spread=8.0, delivery_mode="cross"),
    # ... and these should not: no spatial structure, slack time, slack capacity
    "adversarial": dict(n_hubs=0, hub_sigma=25.0, temporal_press=0.4,
                        disjoint_target=0.05, cap_slack=3.0,
                        courier_spread=0.0, delivery_mode="uniform"),
    # a middle setting
    "neutral": dict(n_hubs=3, hub_sigma=10.0, temporal_press=1.0,
                    disjoint_target=0.3, cap_slack=1.6,
                    courier_spread=4.0, delivery_mode="uniform"),
}


def describe(inst):
    """Structural summary, using the paper's metrics plus overlap."""
    import numpy as np
    pts = np.vstack([inst.xp, inst.xd])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    spatial = d[np.triu_indices(len(pts), 1)].mean()
    mix = float((np.sqrt(((inst.xp - inst.xd) ** 2).sum(1)) <= 10.0).mean())
    a, b = inst.ep, inst.ld
    n = len(a); cnt = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            if b[i] < a[j] or b[j] < a[i]:
                cnt += 1
    return dict(spatial_scatter=spatial, mixing_degree=mix,
                disjoint_frac=cnt / max(tot, 1),
                mean_interval=float((b - a).mean()),
                cap_slack=float(inst.cap.sum() / inst.n_req))


# ------------------------------------------------ theory-driven presets ----
# Derived from what the method actually does, not tuned to flatter it.  Each
# knob is set by a mechanism the algorithm exploits; the ADVERSARIAL preset
# negates every one and MUST be reported alongside.
#
#   1. capacity binds but leaves choices        cap_slack ~ 1.3
#      -> the capacity-constrained assignment has work to do but is not forced
#   2. spatial and temporal structure DISAGREE  hubs + moderate disjointness
#      -> the spatio-temporal embedding has a genuine trade-off to resolve;
#         if requests that are close in space are also close in time, any
#         spatial method finds the same answer
#   3. requests separate in time                disjoint_target ~ 0.4-0.6
#      -> D_t is nonzero for a real fraction of pairs.  At conflict = 1.0
#         (everything overlaps) the temporal term is pure noise
#   4. couriers start apart, at different times courier_spread > 0, stagger > 0
#      -> A(c,k) discriminates, so cluster-to-vehicle assignment matters
#   5. fleet moderately loaded                  slack_ratio ~ 0.8-1.5
#      -> tardiness is avoidable but not free, so service quality separates
#         methods instead of being 0 or forced
#   6. many more clusters than couriers         nu << |R|/|K|
#      -> the bi-level structure is non-degenerate

PRESETS.update({
    "designed": dict(n_hubs=5, hub_sigma=4.0, temporal_press=1.2,
                     disjoint_target=0.50, cap_slack=1.3,
                     courier_spread=10.0, delivery_mode="cross", stagger=0.10),
    # every mechanism above negated
    "designed_adv": dict(n_hubs=0, hub_sigma=30.0, temporal_press=0.3,
                         disjoint_target=0.02, cap_slack=3.0,
                         courier_spread=0.0, delivery_mode="local", stagger=0.0),
})
