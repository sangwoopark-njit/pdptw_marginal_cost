"""What the parallel (Lagrangian) assignment costs, in components not percentages.

The paper's claim about the Lagrangian variant was carried in prose as a single
quality percentage.  That hides where the loss is, and the loss is not spread
evenly: prices coordinate which courier takes a request well and whether a
courier opens at all badly, so the damage should land in the fleet term.  This
measures it, reporting travel, tardiness, couriers, total cost and the time
split for every variant on identical instances.

    python3 experiment_lagrangian.py --fleets 12,30 --seeds 3
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, pandas as pd

import design as D, budget as B, cluster_sparse as CS, visualise as V
import clustering_baselines as CB

VARIANTS = [
    ("exact MIP", dict(assign="mip", mip_limit=5.0)),
    ("Lagrangian", dict(assign="lagrangian")),
    ("Lagr.\\ + fleet fixed, LP", dict(assign="lag_lp")),
    ("Lagr.\\ + fleet fixed, MIP", dict(assign="lag_mip")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleets", default="12,30")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--aligns", default="1.0,0.0")
    ap.add_argument("--out", default="results_paper_lagrangian.csv")
    a = ap.parse_args()

    rows = []
    t00 = time.time()
    for K in [int(x) for x in a.fleets.split(",")]:
        d = D.design(K, a.f, a.nu)
        for align in [float(x) for x in a.aligns.split(",")]:
            for s in range(a.seeds):
                inst, mis = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                for name, kw in VARIANTS:
                    t0 = time.time()
                    r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top,
                               phi=a.phi, seed=s, cand=8, T_out=4, **kw)
                    secs = time.time() - t0
                    m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall,
                                     V.assignment_from_cluster(r, d.n_veh),
                                     1.0, 0.0)
                    p = r["profile"]
                    rows.append(dict(
                        n_req=d.n_req, n_veh=d.n_veh, cap=float(d.cap),
                        phi=a.phi, alignment=align, seed=s, misalign=mis,
                        variant=name, travel=m["travel"],
                        tardiness=m["tardiness"], couriers=m["couriers"],
                        cost=m["travel"] + m["tardiness"]
                             + a.phi * m["couriers"],
                        secs=secs, t_assign=p["mip"],
                        t_par=p["marginal"] + p["descent"] + p["routing"]))
                    pd.DataFrame(rows).to_csv(a.out, index=False)
                print(f"  |R|={d.n_req} |K|={K} align={align:g} seed={s} "
                      f"({time.time() - t00:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({len(rows)} rows, {time.time() - t00:.0f}s)")


if __name__ == "__main__":
    main()
