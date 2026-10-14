"""Is our tardiness deficit a rational trade, or a real weakness?

Referee point 1: the paper motivates the method by punctuality --- temporal
proximity concentrates requests a courier cannot serve on time --- yet the
measured gains are travel, and against the strongest baseline our tardiness is
worse.  Section V-D calls that a rational trade at rho = 1, which is defensible
but untested.

It is testable.  rho prices lateness against distance in the objective

    travel + rho * (earliness + tardiness) + phi * couriers

and it enters everywhere: the exact router sequences with it, the marginal costs
C_rk are computed with it, and the descent decides with it.  If our tardiness is
worse only because rho = 1 makes it worth trading away, then raising rho should
move our solutions towards punctuality and the deficit should close.  If it
persists at rho = 10, the mechanism claim needs moderating rather than
defending.

    python3 sweep_rho.py --seeds 4
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import design as D, budget as B, metrics_instance as M
import cluster_sparse as CS, cluster2 as K2, visualise as V
import clustering_baselines as CB, alns_baseline as A

PROD = dict(assign="mip", mip_limit=5.0, cand=8, T_out=4)
RULES = ["sweep", "fisher", "kmeans", "st_kmeans", "cap_kmeans", "agglom"]
RHOS = [0.5, 1.0, 2.0, 5.0, 10.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_veh", type=int, default=12)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--alns_seeds", type=int, default=3)
    ap.add_argument("--out", default="results_paper_rho.csv")
    a = ap.parse_args()

    d = D.design(a.n_veh, a.f, a.nu)
    rows, t00 = [], time.time()
    for rho in RHOS:
        for align in (1.0, 0.0):
            for s in range(a.seeds):
                inst, mis = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                st = M.describe(inst, ehp, lhp, ehd, lhd, Tall)
                base = dict(n_req=d.n_req, n_veh=d.n_veh, cap=float(d.cap),
                            phi=a.phi, rho=rho, alignment=align, seed=s,
                            misalign=mis, **st)

                def rec(name, m, secs=np.nan):
                    rows.append(dict(base, method=name, travel=m["travel"],
                                     tardiness=m["tardiness"],
                                     couriers=m["couriers"], cost=m["cost"],
                                     secs=secs))

                t0 = time.time()
                r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, rho=rho,
                           phi=a.phi, seed=s, **PROD)
                secs = time.time() - t0
                rec("ours", CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                         V.assignment_from_cluster(r, d.n_veh),
                                         rho, a.phi), secs)

                for rule in RULES:
                    m0 = CB.METHODS[rule](inst, ehp, lhp, ehd, lhd, Tall, rho,
                                          a.phi, seed=s)
                    lab0 = np.asarray(m0["labels"], int)
                    rec(rule, m0)
                    lp = K2.polish_labels(lab0, inst, ehp, lhp, ehd, lhd, Tall,
                                          rho, phi=a.phi)
                    rec(rule + "+ls",
                        CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                     {k: np.where(lp == k)[0].tolist()
                                      for k in range(d.n_veh)}, rho, a.phi))

                bud = max(10, int(np.ceil(secs)))
                costs = [A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, rho,
                                      a.phi, time_limit=bud, seed=sd,
                                      count_cap=True)
                         for sd in range(a.alns_seeds)]
                med = sorted(costs)[len(costs) // 2]
                rec("ALNS", dict(cost=med[0], travel=med[1], tardiness=med[2],
                                 couriers=med[3]), med[4])

                pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"  rho={rho:4.1f} align={align:g} done "
                  f"({time.time() - t00:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({len(rows)} rows, {time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
