"""Does ANY weight on the time axis help a spatio-temporal clustering?

Referee point 3: the paper reports that adding a time coordinate to k-means
makes tardiness worse, but never says how the time axis was scaled.  If the
scaling were arbitrary the result would measure a mis-scaled implementation
rather than the concept.

`clustering_baselines.st_kmeans` already takes `alpha`, the weight on the time
coordinate after it has been normalised to the spatial spread:

    X = [x, y, alpha * (sigma_x / sigma_t) * t]

so alpha = 0 is plain k-means and alpha = 1 gives time the same spread as space.
This sweeps alpha so the claim can be stated as "no weight helps" rather than
"this weight hurts".

    python3 sweep_st_weight.py --fleets 12,30 --seeds 3
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import design as D, budget as B, clustering_baselines as CB
import cluster2 as K2

ALPHAS = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleets", default="12,30")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--out", default="results_paper_stweight.csv")
    a = ap.parse_args()

    rows = []
    t0 = time.time()
    for K in [int(x) for x in a.fleets.split(",")]:
        d = D.design(K, a.f, a.nu)
        for align in (1.0, 0.0):
            for s in range(a.seeds):
                inst, mis = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                for al in ALPHAS:
                    m = CB.st_kmeans(inst, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0,
                                     seed=s, alpha=al)
                    lab = np.asarray(m["labels"], int)
                    base = dict(n_req=d.n_req, n_veh=K, cap=float(d.cap),
                                phi=a.phi, alignment=align, seed=s,
                                misalign=mis, alpha=al)
                    rows.append(dict(base, variant="as published",
                                     travel=m["travel"],
                                     tardiness=m["tardiness"],
                                     couriers=m["couriers"],
                                     cost=m["travel"] + m["tardiness"]
                                          + a.phi * m["couriers"]))
                    # and with our descent, so the comparison is the fair one
                    lp = K2.polish_labels(lab, inst, ehp, lhp, ehd, lhd, Tall,
                                          1.0, phi=a.phi)
                    mp = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                      {k: np.where(lp == k)[0].tolist()
                                       for k in range(K)}, 1.0, 0.0)
                    rows.append(dict(base, variant="+ descent",
                                     travel=mp["travel"],
                                     tardiness=mp["tardiness"],
                                     couriers=mp["couriers"],
                                     cost=mp["travel"] + mp["tardiness"]
                                          + a.phi * mp["couriers"]))
                    pd.DataFrame(rows).to_csv(a.out, index=False)
                print(f"  |R|={d.n_req} align={align:g} seed={s} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({len(rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
