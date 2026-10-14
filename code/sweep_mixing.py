"""Does spatial mixing explain the smaller margin on Li & Lim?

Referee point 8, and the fair version of the objection behind it: the headline
margin is measured on instances our own generator produced, and those instances
are structurally unlike the Li & Lim benchmark.  Measured, the two do not
overlap on the axis that matters:

    metric              Li & Lim            our design point
    mean interval       69 -- 1227          80 (fixed)
    temporal conflict   0.59 -- 0.99        0.23 -- 0.30
    slack ratio         0.57 -- 3.95        1.57 -- 1.99

So "our instances are like Li & Lim but harder" is NOT true as the generator
stands, and claiming it would be the kind of thing a referee checks.

`generator2.generate` takes `window_width`, and the request interval is exactly
`window_width + pd_gap`, so that one knob moves the instance along precisely the
axis where the two datasets differ.  This sweeps it from our design point into
the benchmark's regime and reports the margin as a function of it, so the claim
becomes "the advantage declines smoothly with slack and reaches the benchmark's
value where the benchmark sits" rather than "it holds on instances we chose".

    python3 sweep_window.py --seeds 4
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import design as D, budget as B, generator2 as G2, metrics_instance as M
import cluster_sparse as CS, cluster2 as K2, visualise as V
import clustering_baselines as CB, alns_baseline as A

PROD = dict(assign="mip", mip_limit=5.0, cand=8, T_out=4)
RULES = ["sweep", "fisher", "kmeans", "st_kmeans", "cap_kmeans", "agglom"]
# our design point is 45; Li & Lim's mean interval of 464 corresponds to ~430
MODES = ["local", "cross"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_veh", type=int, default=12)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--alns_seeds", type=int, default=3)
    ap.add_argument("--out", default="results_paper_mixing.csv")
    a = ap.parse_args()

    d = D.design(a.n_veh, a.f, a.nu)
    rows, t00 = [], time.time()
    for mode in MODES:
        for align in (1.0, 0.0):
            for s in range(a.seeds):
                inst, mis = G2.generate(n_req=d.n_req, n_veh=d.n_veh, seed=s,
                                        alignment=align, delivery_mode=mode)
                inst.cap = np.full(d.n_veh, float(d.cap))
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                st = M.describe(inst, ehp, lhp, ehd, lhd, Tall)
                base = dict(n_req=d.n_req, n_veh=d.n_veh, cap=float(d.cap),
                            phi=a.phi, alignment=align, seed=s, misalign=mis,
                            delivery_mode=mode, **st)

                t0 = time.time()
                r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=a.phi,
                           seed=s, **PROD)
                secs = time.time() - t0
                m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                 V.assignment_from_cluster(r, d.n_veh), 1.0, 0.0)
                rows.append(dict(base, method="ours", travel=m["travel"],
                                 tardiness=m["tardiness"],
                                 couriers=m["couriers"],
                                 cost=m["travel"] + m["tardiness"]
                                      + a.phi * m["couriers"], secs=secs))

                for rule in RULES:
                    m0 = CB.METHODS[rule](inst, ehp, lhp, ehd, lhd, Tall, 1.0,
                                          0.0, seed=s)
                    lab0 = np.asarray(m0["labels"], int)
                    for tag, lab in (("", lab0),
                                     ("+ls", K2.polish_labels(
                                         lab0, inst, ehp, lhp, ehd, lhd, Tall,
                                         1.0, phi=a.phi))):
                        mm = CB._evaluate(
                            inst, ehp, lhp, ehd, lhd, Tall,
                            {k: np.where(lab == k)[0].tolist()
                             for k in range(d.n_veh)}, 1.0, 0.0)
                        rows.append(dict(
                            base, method=rule + tag, travel=mm["travel"],
                            tardiness=mm["tardiness"], couriers=mm["couriers"],
                            cost=mm["travel"] + mm["tardiness"]
                                 + a.phi * mm["couriers"], secs=np.nan))

                bud = max(10, int(np.ceil(secs)))
                costs = [A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0,
                                      a.phi, time_limit=bud, seed=sd,
                                      count_cap=True)
                         for sd in range(a.alns_seeds)]
                med = sorted(costs)[len(costs) // 2]
                rows.append(dict(base, method="ALNS", travel=med[1],
                                 tardiness=med[2], couriers=med[3],
                                 cost=med[0], secs=med[4]))

                pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"  mode={mode:6s} align={align:g} done "
                  f"({time.time() - t00:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({len(rows)} rows, {time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
