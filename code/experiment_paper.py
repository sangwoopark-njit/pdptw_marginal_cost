"""
The paper's experiments, run on the PRODUCTION algorithm.

    python3 experiment_paper.py --part main
    python3 experiment_paper.py --part scale

Production algorithm: `cluster_sparse.run` with candidate lists (cand = 8) and
the exact assignment (one integer program per outer iteration, 5 s limit),
T_out = 4, exact per-courier routing.  Instances are built by `design.py` with
f = 5/6 and nu = 4, so |R| = nu f |K|, Q_k = floor(4 nu / 3), and the fleet
decision is free over ceil(|R|/Q_k) .. |K| couriers.

PART "main"  the comparison table: six published clustering rules as published
             (tier 1) and with our descent (tier 2), ours, and ALNS given the
             same wall-clock our method used.
PART "scale" the scaling table: ours, the strongest baseline, and ALNS at
             |R| = 100 .. 1000, with the runtime split into the part that is
             independent per courier and the part that is not.

Cost = travel + rho * tardiness + phi * (couriers dispatched), with rho = 1.
Every solution is evaluated by the same exact router, so differences come from
the assignment alone.
"""
from __future__ import annotations
import argparse, csv, math, os, time
import numpy as np

import budget as B, metrics_instance as M
import cluster2 as K2, cluster_sparse as CS, visualise as V
import clustering_baselines as CB, alns_baseline as A
import design as D

RULES = ["sweep", "fisher", "kmeans", "st_kmeans", "cap_kmeans", "agglom"]
PROD = dict(assign='mip', mip_limit=5.0, cand=8, T_out=4)


def score(inst, ehp, lhp, ehd, lhd, Tall, assign, phi):
    m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall, assign, 1.0, 0.0)
    m["cost"] = m["travel"] + m["tardiness"] + phi * m["couriers"]
    return m


def _write(path, rows, hdr):
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not hdr:
            w.writeheader()
            hdr = True
        for r in rows:
            w.writerow(r)
    return hdr


def run_main(a):
    hdr = os.path.exists(a.out)
    t00 = time.time()
    done = 0
    for K in [int(x) for x in a.fleets.split(",")]:
        d = D.design(K, a.f, a.nu)
        for align in [float(x) for x in a.aligns.split(",")]:
            for s in range(a.seeds):
                inst, mis = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                st = M.describe(inst, ehp, lhp, ehd, lhd, Tall)
                base = dict(n_req=d.n_req, n_veh=K, cap=float(d.cap),
                            phi=a.phi, alignment=align, seed=s, misalign=mis,
                            min_couriers=d.min_couriers, **st)
                rows = []

                t0 = time.time()
                r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=a.phi,
                           seed=s, **PROD)
                ours_secs = time.time() - t0
                m = score(inst, ehp, lhp, ehd, lhd, Tall,
                          V.assignment_from_cluster(r, K), a.phi)
                rows.append(dict(base, method="ours", tier=4, travel=m["travel"],
                                 tardiness=m["tardiness"], couriers=m["couriers"],
                                 cost=m["cost"], secs=ours_secs,
                                 par_secs=r["profile"]["marginal"]
                                 + r["profile"]["routing"] + r["profile"]["descent"],
                                 ser_secs=r["profile"]["mip"]
                                 + r["profile"]["candidates"]))

                for rule in RULES:
                    t0 = time.time()
                    m0 = CB.METHODS[rule](inst, ehp, lhp, ehd, lhd, Tall, 1.0,
                                          0.0, seed=s)
                    lab0 = np.asarray(m0["labels"], int)
                    m = score(inst, ehp, lhp, ehd, lhd, Tall,
                              {k: np.where(lab0 == k)[0].tolist() for k in range(K)},
                              a.phi)
                    rows.append(dict(base, method=rule, tier=1, travel=m["travel"],
                                     tardiness=m["tardiness"],
                                     couriers=m["couriers"], cost=m["cost"],
                                     secs=time.time() - t0, par_secs=np.nan,
                                     ser_secs=np.nan))
                    t0 = time.time()
                    lab = K2.polish_labels(lab0, inst, ehp, lhp, ehd, lhd, Tall,
                                           1.0, phi=a.phi)
                    m = score(inst, ehp, lhp, ehd, lhd, Tall,
                              {k: np.where(lab == k)[0].tolist() for k in range(K)},
                              a.phi)
                    rows.append(dict(base, method=rule + "+ls", tier=2,
                                     travel=m["travel"], tardiness=m["tardiness"],
                                     couriers=m["couriers"], cost=m["cost"],
                                     secs=time.time() - t0, par_secs=np.nan,
                                     ser_secs=np.nan))

                bud = max(a.floor, math.ceil(ours_secs))
                costs = []
                for sd in range(a.alns_seeds):
                    al = A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0, a.phi,
                                      time_limit=bud, seed=sd, count_cap=True)
                    costs.append(al)
                med = sorted(costs)[len(costs) // 2]
                rows.append(dict(base, method="ALNS", tier=9, travel=med[1],
                                 tardiness=med[2], couriers=med[3], cost=med[0],
                                 secs=med[4], par_secs=np.nan, ser_secs=np.nan))

                hdr = _write(a.out, rows, hdr)
                done += 1
                print(f"  [{done}] |R|={d.n_req} |K|={K} align={align:g} seed={s} "
                      f"({time.time() - t00:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({done} instances, {time.time() - t00:.0f}s)")


def run_scale(a):
    hdr = os.path.exists(a.out)
    t00 = time.time()
    done = 0
    for K in [int(x) for x in a.fleets.split(",")]:
        d = D.design(K, a.f, a.nu)
        for s in range(a.seeds):
            inst, mis = D.make_instance(d, s, 0.0)
            ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
            base = dict(n_req=d.n_req, n_veh=K, cap=float(d.cap), phi=a.phi,
                        seed=s, min_couriers=d.min_couriers)
            rows = []

            t0 = time.time()
            r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=a.phi, seed=s,
                       **PROD)
            ours_secs = time.time() - t0
            p = r["profile"]
            m = score(inst, ehp, lhp, ehd, lhd, Tall,
                      V.assignment_from_cluster(r, K), a.phi)
            rows.append(dict(base, method="ours", travel=m["travel"],
                             tardiness=m["tardiness"], couriers=m["couriers"],
                             cost=m["cost"], secs=ours_secs,
                             t_marginal=p["marginal"], t_assign=p["mip"],
                             t_descent=p["descent"], t_routing=p["routing"]))

            t0 = time.time()
            m0 = CB.METHODS["cap_kmeans"](inst, ehp, lhp, ehd, lhd, Tall, 1.0,
                                          0.0, seed=s)
            lab = K2.polish_labels(np.asarray(m0["labels"], int), inst, ehp, lhp,
                                   ehd, lhd, Tall, 1.0, phi=a.phi)
            m = score(inst, ehp, lhp, ehd, lhd, Tall,
                      {k: np.where(lab == k)[0].tolist() for k in range(K)}, a.phi)
            rows.append(dict(base, method="cap_kmeans+ls", travel=m["travel"],
                             tardiness=m["tardiness"], couriers=m["couriers"],
                             cost=m["cost"], secs=time.time() - t0,
                             t_marginal=np.nan, t_assign=np.nan,
                             t_descent=np.nan, t_routing=np.nan))

            bud = max(a.floor, math.ceil(ours_secs))
            costs = [A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0, a.phi,
                                  time_limit=bud, seed=sd, count_cap=True)
                     for sd in range(a.alns_seeds)]
            med = sorted(costs)[len(costs) // 2]
            rows.append(dict(base, method="ALNS", travel=med[1], tardiness=med[2],
                             couriers=med[3], cost=med[0], secs=med[4],
                             t_marginal=np.nan, t_assign=np.nan,
                             t_descent=np.nan, t_routing=np.nan))

            hdr = _write(a.out, rows, hdr)
            done += 1
            print(f"  [{done}] |R|={d.n_req} |K|={K} seed={s} "
                  f"ours {ours_secs:.1f}s ({time.time() - t00:.0f}s)", flush=True)
    print(f"\nwrote {a.out} ({done} instances, {time.time() - t00:.0f}s)")


def run_lilim(a):
    """External validity: the same comparison on adapted Li & Lim instances.

    The benchmark's model differs from ours, so instances are adapted exactly as
    the problem statement describes: unit demands with capacity counted in
    requests, couriers at perturbed positions with staggered ready times, open
    routes, soft windows, and a batch sampled from a release-time window.  The
    fleet, capacity and request count are set to the same design point used for
    the generated instances, so the two tables are directly comparable.
    """
    import campaign as C
    hdr = os.path.exists(a.out)
    t00 = time.time()
    done = 0
    K = int(a.fleets.split(",")[0])
    d = D.design(K, a.f, a.nu)
    fams = a.fams.split(",")
    for fam in fams:
        for s in range(a.seeds):
            try:
                inst, ehp, lhp, ehd, lhd, Tall, Top = C.load(
                    fam, s, d.n_req, d.n_veh, cap=d.cap)
            except Exception as e:
                print("  skip", fam, s, e)
                continue
            st = M.describe(inst, ehp, lhp, ehd, lhd, Tall)
            base = dict(n_req=d.n_req, n_veh=d.n_veh, cap=float(d.cap),
                        phi=a.phi, alignment=np.nan, seed=s, fam=fam,
                        misalign=np.nan, min_couriers=d.min_couriers, **st)
            rows = []

            t0 = time.time()
            r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=a.phi, seed=s,
                       **PROD)
            ours_secs = time.time() - t0
            m = score(inst, ehp, lhp, ehd, lhd, Tall,
                      V.assignment_from_cluster(r, d.n_veh), a.phi)
            rows.append(dict(base, method="ours", tier=4, travel=m["travel"],
                             tardiness=m["tardiness"], couriers=m["couriers"],
                             cost=m["cost"], secs=ours_secs))

            for rule in RULES:
                t0 = time.time()
                m0 = CB.METHODS[rule](inst, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0,
                                      seed=s)
                lab0 = np.asarray(m0["labels"], int)
                m = score(inst, ehp, lhp, ehd, lhd, Tall,
                          {k: np.where(lab0 == k)[0].tolist()
                           for k in range(d.n_veh)}, a.phi)
                rows.append(dict(base, method=rule, tier=1, travel=m["travel"],
                                 tardiness=m["tardiness"], couriers=m["couriers"],
                                 cost=m["cost"], secs=time.time() - t0))
                t0 = time.time()
                lab = K2.polish_labels(lab0, inst, ehp, lhp, ehd, lhd, Tall, 1.0,
                                       phi=a.phi)
                m = score(inst, ehp, lhp, ehd, lhd, Tall,
                          {k: np.where(lab == k)[0].tolist()
                           for k in range(d.n_veh)}, a.phi)
                rows.append(dict(base, method=rule + "+ls", tier=2,
                                 travel=m["travel"], tardiness=m["tardiness"],
                                 couriers=m["couriers"], cost=m["cost"],
                                 secs=time.time() - t0))

            bud = max(a.floor, math.ceil(ours_secs))
            costs = [A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0, a.phi,
                                  time_limit=bud, seed=sd, count_cap=True)
                     for sd in range(a.alns_seeds)]
            med = sorted(costs)[len(costs) // 2]
            rows.append(dict(base, method="ALNS", tier=9, travel=med[1],
                             tardiness=med[2], couriers=med[3], cost=med[0],
                             secs=med[4]))

            hdr = _write(a.out, rows, hdr)
            done += 1
            print(f"  [{done}] {fam} seed={s} ({time.time() - t00:.0f}s)",
                  flush=True)
    print(f"\nwrote {a.out} ({done} instances, {time.time() - t00:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="main",
                    choices=["main", "scale", "lilim"])
    ap.add_argument("--fams",
                    default="lrc101,lrc102,lrc201,lrc202,lr201,lr202,lc201,lc202")
    ap.add_argument("--fleets", default="12,30,60")
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--aligns", default="1.0,0.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alns_seeds", type=int, default=3)
    ap.add_argument("--floor", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"results_paper_{a.part}.csv"
    {"main": run_main, "scale": run_scale, "lilim": run_lilim}[a.part](a)


if __name__ == "__main__":
    main()
