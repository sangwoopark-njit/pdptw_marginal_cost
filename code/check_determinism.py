"""Is the production method's answer independent of its random seed?

The paper claims the method returns the same answer every time, which matters
for auditable dispatch: a dispatcher asked why a courier got a request should
not have to answer "a random number".  The claim was only ever measured on the
superseded two-level method, so it is measured here on `cluster_sparse`, and
against the stochastic reference on the same instances for contrast.

    python3 check_determinism.py --fleets 12,30 --seeds 3 --restarts 5
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import design as D, budget as B, cluster_sparse as CS, visualise as V
import clustering_baselines as CB, alns_baseline as A

PROD = dict(assign="mip", mip_limit=5.0, cand=8, T_out=4)


def spread(v):
    v = np.asarray(v, float)
    return 100.0 * (v.max() - v.min()) / max(v.min(), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleets", default="12,30")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--out", default="results_paper_determinism.csv")
    a = ap.parse_args()

    rows = []
    for K in [int(x) for x in a.fleets.split(",")]:
        d = D.design(K, a.f, a.nu)
        for align in (1.0, 0.0):
            for s in range(a.seeds):
                inst, _ = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)

                ours, t_ours = [], []
                for rs in range(a.restarts):
                    t0 = time.time()
                    r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top,
                               phi=a.phi, seed=1000 + rs, **PROD)
                    t_ours.append(time.time() - t0)
                    m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                     V.assignment_from_cluster(r, d.n_veh),
                                     1.0, 0.0)
                    ours.append(m["travel"] + m["tardiness"]
                                + a.phi * m["couriers"])

                bud = max(1, int(np.ceil(np.mean(t_ours))))
                ref = [A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0, a.phi,
                                    time_limit=bud, seed=1000 + rs,
                                    count_cap=True)[0]
                       for rs in range(a.restarts)]

                rows.append(dict(n_req=d.n_req, n_veh=K, alignment=align,
                                 seed=s, restarts=a.restarts,
                                 ours_spread=spread(ours),
                                 ours_identical=int(len(set(np.round(ours, 6)))
                                                    == 1),
                                 ours_mean=float(np.mean(ours)),
                                 ref_spread=spread(ref),
                                 ref_mean=float(np.mean(ref)),
                                 # what P cores buy a stochastic method: run it
                                 # P times and keep the cheapest, against the
                                 # median a single-core comparison reports
                                 ref_best=float(np.min(ref)),
                                 ref_median=float(np.median(ref)),
                                 ours_best=float(np.min(ours)),
                                 ours_median=float(np.median(ours))))
                pd.DataFrame(rows).to_csv(a.out, index=False)
                print(f"  |R|={d.n_req} align={align:g} seed={s}: "
                      f"ours spread {spread(ours):5.2f}%  "
                      f"reference spread {spread(ref):5.2f}%", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nours: mean spread {df.ours_spread.mean():.3f}%, "
          f"max {df.ours_spread.max():.3f}%, "
          f"bit-identical on {df.ours_identical.sum()}/{len(df)} instances")
    print(f"reference: mean spread {df.ref_spread.mean():.2f}%, "
          f"max {df.ref_spread.max():.2f}%")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
