"""
Instance and parameter design, as a checked rule rather than ad-hoc numbers.

WHY THIS EXISTS.  An earlier setup used |R| = 24 requests, |K| = 12 couriers and
nu = 3, which makes |C| = ceil(24/3) = 8 clusters.  Since (P1) assigns whole
clusters, AT MOST |C| couriers can ever be opened -- so 4 of the 12 couriers were
unusable by construction, and the method looked bad at phi = 0 for a reason that
had nothing to do with the method.  The rule below makes that impossible to
express.

THE PARAMETERS.  Choose the fleet |K|, the reduction factor f (the fraction of
the fleet we EXPECT a good solution to dispatch), and nu (requests per dispatched
courier).  Everything else follows:

    K_eff = f |K|                  couriers we expect to be dispatched
    |R|   = nu K_eff               requests
    |C|   = 1.5 K_eff              clusters: 1.5 per dispatched courier, so a
                                   courier takes one or two of them
    Q     = floor(4 nu / 3)        capacity: two clusters' worth, and a little
                                   more than one courier's intended load nu

THE THREE CONSTRAINTS, each of which has bitten us:

    Q >= nu    capacity must hold the load we intend a courier to carry  <=> nu >= 3
    Q <= 8     the exact router is exact only to 8 requests per courier  <=> nu <= 6
    |C| >= |K| else the cluster count caps the fleet and the dispatch
               decision is not free                                      <=> f >= 2/3

The fleet decision then ranges over ceil(|R|/Q) .. |K| couriers, with K_eff
sitting inside it, so both directions are available: a method may open more
couriers to cut travel, or fewer to cut the dispatch charge.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

import generator2 as G2


@dataclass
class Design:
    n_veh: int          # |K|, couriers available
    f: float            # reduction factor: fraction we expect to dispatch
    nu: float           # requests per dispatched courier
    n_req: int          # |R|
    n_cl: int           # |C|
    cap: int            # Q_k
    k_eff: int          # couriers we expect to dispatch
    min_couriers: int   # forced by capacity
    max_couriers: int   # = |K|

    def describe(self) -> str:
        return (f"|K|={self.n_veh} available, |R|={self.n_req}, |C|={self.n_cl}, "
                f"Q_k={self.cap}; fleet decision over "
                f"{self.min_couriers}..{self.max_couriers} couriers "
                f"(f={self.f:g}, nu={self.nu:g}, expected {self.k_eff})")


def design(n_veh: int, f: float = 5 / 6, nu: int = 4,
           clusters_per_courier: float = 1.5) -> Design:
    """Apply the rule and check it, rather than trusting the arithmetic."""
    k_eff = int(round(f * n_veh))
    n_req = int(nu * k_eff)
    n_cl = int(round(clusters_per_courier * k_eff))
    cap = int(math.floor(4 * nu / 3))
    min_c = math.ceil(n_req / cap)

    if cap < nu:
        raise ValueError(f"capacity {cap} < intended load per courier {nu}: "
                         "raise nu to at least 3")
    if cap > 8:
        raise ValueError(f"capacity {cap} > 8: the router stops being exact, "
                         "so lower nu to at most 6")
    if n_cl < n_veh:
        raise ValueError(
            f"|C|={n_cl} < |K|={n_veh}: at most |C| couriers can ever be opened, "
            f"so the fleet decision is not free.  Raise f to at least "
            f"{2 / 3:.3f} (currently {f:g})")
    if min_c >= n_veh:
        raise ValueError(f"capacity forces {min_c} of {n_veh} couriers: "
                         "there is no dispatch decision left to make")

    return Design(n_veh=n_veh, f=f, nu=nu, n_req=n_req, n_cl=n_cl, cap=cap,
                  k_eff=k_eff, min_couriers=min_c, max_couriers=n_veh)


def make_instance(d: Design, seed: int, alignment: float):
    """Generate an instance to the design, with the capacity set exactly."""
    inst, mis = G2.generate(n_req=d.n_req, n_veh=d.n_veh, seed=seed,
                            alignment=alignment)
    inst.cap = np.full(d.n_veh, float(d.cap))
    return inst, mis


if __name__ == "__main__":
    print("the rule, over a range of settings\n")
    print(f"{'|K|':>4} {'f':>5} {'nu':>3} {'|R|':>4} {'|C|':>4} {'Q':>3} "
          f"{'couriers':>10}  note")
    for n_veh in (6, 12, 18):
        for f in (2 / 3, 5 / 6, 1.0):
            for nu in (3, 4, 6):
                try:
                    d = design(n_veh, f, nu)
                    print(f"{d.n_veh:4d} {f:5.2f} {nu:3d} {d.n_req:4d} "
                          f"{d.n_cl:4d} {d.cap:3d} "
                          f"{str(d.min_couriers) + '..' + str(d.max_couriers):>10}  ok")
                except ValueError as e:
                    print(f"{n_veh:4d} {f:5.2f} {nu:3d} {'':4} {'':4} {'':3} "
                          f"{'':>10}  rejected: {str(e)[:60]}")
