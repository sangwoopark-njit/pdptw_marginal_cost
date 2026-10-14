"""What does the pairwise-slack criterion cost, in both symmetrisations?

Referee point 4: Section III-B dismisses the min-slack repair on a correlation
argument and forward-references a table that does not exist.  This produces it.

Two things are measured on the same instances:

  1. the correlation of each symmetrisation with temporal separation
     |t_r - t_r'|, which is what the paper claims the summed version secretly
     measures;
  2. the cost of actually grouping by each one, as published and with our
     descent, against our marginal cost, all routed by the same exact router.

    python3 experiment_slack.py --fleets 12,30 --seeds 3
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import design as D, budget as B, slack_rules as SR
import cluster_sparse as CS, cluster2 as K2, visualise as V
import clustering_baselines as CB

PROD = dict(assign="mip", mip_limit=5.0, cand=8, T_out=4)
RULES = ["sweep", "fisher", "kmeans", "st_kmeans", "cap_kmeans", "agglom"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleets", default="12,30")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--out", default="results_paper_slack.csv")
    a = ap.parse_args()

    rows, t00 = [], time.time()
    for K in [int(x) for x in a.fleets.split(",")]:
        d = D.design(K, a.f, a.nu)
        for align in (1.0, 0.0):
            for s in range(a.seeds):
                inst, mis = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                cor = SR.correlation_with_separation(inst, ehp, lhd, Tall)
                base = dict(n_req=d.n_req, n_veh=K, cap=float(d.cap),
                            phi=a.phi, alignment=align, seed=s, misalign=mis,
                            corr_sum=cor["sum"], corr_min=cor["min"])

                def rec(name, m):
                    rows.append(dict(base, method=name, travel=m["travel"],
                                     tardiness=m["tardiness"],
                                     couriers=m["couriers"],
                                     cost=m["travel"] + m["tardiness"]
                                          + a.phi * m["couriers"]))

                r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=a.phi,
                           seed=s, **PROD)
                rec("ours", CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                         V.assignment_from_cluster(r, K),
                                         1.0, 0.0))

                allm = dict(SR.METHODS)
                allm.update({k: CB.METHODS[k] for k in RULES})
                for name, fn in allm.items():
                    m0 = fn(inst, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0, seed=s)
                    lab0 = np.asarray(m0["labels"], int)
                    rec(name, m0)
                    lp = K2.polish_labels(lab0, inst, ehp, lhp, ehd, lhd, Tall,
                                          1.0, phi=a.phi)
                    rec(name + "+ls",
                        CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                     {k: np.where(lp == k)[0].tolist()
                                      for k in range(K)}, 1.0, 0.0))
                pd.DataFrame(rows).to_csv(a.out, index=False)
                print(f"  |R|={d.n_req} align={align:g} seed={s} "
                      f"({time.time() - t00:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({len(rows)} rows, {time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
