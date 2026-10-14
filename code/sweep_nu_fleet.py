"""
JOINT SWEEP over nu (requests per dispatched courier) and |K| (fleet size).

    python3 sweep_nu_fleet.py --hours 6

The reduction factor f is held at 5/6 throughout, so every cell is built by the
same rule (design.py) and only the two axes of interest move:

    nu  in {3, 4, 5, 6}   how much work one courier is meant to carry.
                          Bounded below by Q >= nu and above by Q <= 8, which is
                          where the router stops being exact.
    |K| in {6, 9, 12, 18} how many couriers exist.  |R| = nu * f * |K| grows with
                          both axes, so the top corner is |R| = 90.

WHAT THE SWEEP IS FOR.  Every scaling claim so far was measured on one cell.
Two questions need the grid:
  1. does our advantage over the published rules hold as the fleet grows, or was
     it a small-fleet effect?
  2. how does the cost of ENUMERATING fleet sizes -- the obvious way to give a
     published rule a dispatch decision -- grow against our single solve?  The
     number of legal fleet sizes grows with |K|, so this is the axis on which the
     enumeration argument stands or falls.

PROTOCOL NOTES.
  * ALNS gets the same wall-clock OUR method took on that instance, with a floor
    of `--floor` seconds.  A fixed budget would quietly favour whichever method
    happens to suit the size.
  * Tier 3 (fleet enumeration) is run for ONE published rule, named by
    --fleet_rule, because running it for all six costs six times as much and
    measures the same thing.
  * Cells run in ascending order of |R| * |K|, so if the time budget runs out the
    completed part of the grid is the small and medium cells rather than a
    random scatter.
  * Rows are flushed as they are produced; re-running appends.
"""
from __future__ import annotations
import argparse, csv, math, os, time
import numpy as np

import budget as B, metrics_instance as M
import cluster2 as K2, visualise as V
import clustering_baselines as CB, alns_baseline as A
import design as D
from instance import Instance

RULES = ["sweep", "fisher", "kmeans", "st_kmeans", "cap_kmeans", "agglom"]


def reduced_fleet(inst, k):
    return Instance(name=inst.name, n_req=inst.n_req, xp=inst.xp, xd=inst.xd,
                    ep=inst.ep, lp=inst.lp, ed=inst.ed, ld=inst.ld,
                    sp=inst.sp, sd=inst.sd, origins=inst.origins[:k],
                    theta=inst.theta[:k], cap=inst.cap[:k], horizon=inst.horizon)


def score(inst, ehp, lhp, ehd, lhd, Tall, assign, phi):
    m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall, assign, 1.0, 0.0)
    m["cost"] = m["travel"] + m["tardiness"] + phi * m["couriers"]
    return m


def one_cell(d, inst, ehp, lhp, ehd, lhd, Tall, Top, phi, seed, base,
             fleet_rule, alns_seeds, floor, out, hdr):
    K = d.n_veh
    rows = []

    # ours first, so ALNS can be given the same wall-clock
    our_secs = None
    for tag, ncl in [("ours_bilevel", d.n_cl), ("ours_single", K)]:
        t0 = time.time()
        r = K2.run(inst, ehp, lhp, ehd, lhd, Tall, Top, n_cl=ncl, seed=seed,
                   phi=phi, polish=True, route_mode='final', route_top=2,
                   **K2.CHAIN_DEFAULTS)
        sec = time.time() - t0
        m = score(inst, ehp, lhp, ehd, lhd, Tall,
                  V.assignment_from_cluster(r, K), phi)
        if tag == "ours_bilevel":
            our_secs = sec
        rows.append(dict(base, method=tag, tier=4, travel=m["travel"],
                         tardiness=m["tardiness"], couriers=m["couriers"],
                         cost=m["cost"], secs=sec, budget=np.nan))

    for rule in RULES:
        fn = CB.METHODS[rule]
        t0 = time.time()
        m0 = fn(inst, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0, seed=seed)
        lab0 = np.asarray(m0["labels"], int)
        m = score(inst, ehp, lhp, ehd, lhd, Tall,
                  {k: np.where(lab0 == k)[0].tolist() for k in range(K)}, phi)
        rows.append(dict(base, method=rule, tier=1, travel=m["travel"],
                         tardiness=m["tardiness"], couriers=m["couriers"],
                         cost=m["cost"], secs=time.time() - t0, budget=np.nan))

        t0 = time.time()
        lab = K2.polish_labels(lab0, inst, ehp, lhp, ehd, lhd, Tall, 1.0, phi=phi)
        m = score(inst, ehp, lhp, ehd, lhd, Tall,
                  {k: np.where(lab == k)[0].tolist() for k in range(K)}, phi)
        rows.append(dict(base, method=rule + "+pol", tier=2, travel=m["travel"],
                         tardiness=m["tardiness"], couriers=m["couriers"],
                         cost=m["cost"], secs=time.time() - t0, budget=np.nan))

    # tier 3 for one rule: the enumeration cost is the thing being measured
    fn = CB.METHODS[fleet_rule]
    t0 = time.time()
    best = None
    for k in range(d.min_couriers, K + 1):
        sub = reduced_fleet(inst, k)
        mk = fn(sub, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0, seed=seed)
        lk = K2.polish_labels(np.asarray(mk["labels"], int), sub, ehp, lhp, ehd,
                              lhd, Tall, 1.0, phi=phi)
        mm = score(sub, ehp, lhp, ehd, lhd, Tall,
                   {j: np.where(lk == j)[0].tolist() for j in range(k)}, phi)
        if best is None or mm["cost"] < best["cost"]:
            best = mm
    rows.append(dict(base, method=fleet_rule + "+pol+fleet", tier=3,
                     travel=best["travel"], tardiness=best["tardiness"],
                     couriers=best["couriers"], cost=best["cost"],
                     secs=time.time() - t0, budget=np.nan))

    bud = max(floor, math.ceil(our_secs))
    costs = []
    for sd in range(alns_seeds):
        al = A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0, phi,
                          time_limit=bud, seed=sd, count_cap=True)
        costs.append((al[0], al[1], al[2], al[3], al[4]))
    med = sorted(costs)[len(costs) // 2]
    rows.append(dict(base, method="ALNS(ref)", tier=9, travel=med[1],
                     tardiness=med[2], couriers=med[3], cost=med[0],
                     secs=med[4], budget=bud))
    return _write(out, rows, hdr)


def _write(path, rows, hdr):
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not hdr:
            w.writeheader()
            hdr = True
        for r in rows:
            w.writerow(r)
    return hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nus", default="3,4,5,6")
    ap.add_argument("--fleets", default="6,9,12,18")
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--phis", default="100,0",
                    help="dispatch charges, OUTERMOST loop so each grid "
                         "completes before the next charge starts")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--aligns", default="1.0,0.0")
    ap.add_argument("--alns_seeds", type=int, default=3)
    ap.add_argument("--floor", type=float, default=10.0)
    ap.add_argument("--fleet_rule", default="agglom")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--out", default="results_nu_fleet.csv")
    a = ap.parse_args()

    phis = [float(x) for x in a.phis.split(",")]
    nus = [int(x) for x in a.nus.split(",")]
    fleets = [int(x) for x in a.fleets.split(",")]
    aligns = [float(x) for x in a.aligns.split(",")]

    grid = []
    for nu in nus:
        for K in fleets:
            try:
                d = D.design(K, a.f, nu)
            except ValueError as e:
                print(f"  skip nu={nu} |K|={K}: {e}", flush=True)
                continue
            grid.append((d.n_req * d.n_veh, nu, K, d))
    grid.sort()                      # smallest cells first

    print(f"grid of {len(grid)} (nu, |K|) cells, f = {a.f:g}, "
          f"phi in {phis}, {len(aligns)} alignment levels x {a.seeds} seeds each",
          flush=True)
    for _, nu, K, d in grid:
        print(f"    nu={nu} {d.describe()}", flush=True)

    hdr = os.path.exists(a.out)
    t0 = time.time()
    done = 0
    deadline = a.hours * 3600
    for phi in phis:
      for _, nu, K, d in grid:
        for align in aligns:
            for s in range(a.seeds):
                if time.time() - t0 > deadline:
                    print(f"\nstopping: {a.hours} h budget reached with "
                          f"{done} cells done", flush=True)
                    return
                inst, mis = D.make_instance(d, s, align)
                ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
                st = M.describe(inst, ehp, lhp, ehd, lhd, Tall)
                base = dict(source="generated", fam=f"align{align}", seed=s,
                            nu=nu, n_req=d.n_req, n_veh=d.n_veh, n_cl=d.n_cl,
                            cap=float(d.cap), phi=phi, alignment=align,
                            misalign=mis, min_couriers=d.min_couriers,
                            k_eff=d.k_eff, n_fleet_sizes=d.n_veh - d.min_couriers + 1,
                            **st)
                hdr = one_cell(d, inst, ehp, lhp, ehd, lhd, Tall, Top, phi, s,
                               base, a.fleet_rule, a.alns_seeds, a.floor,
                               a.out, hdr)
                done += 1
                el = time.time() - t0
                print(f"  [{done}] phi={phi:g} nu={nu} |K|={K} |R|={d.n_req} "
                      f"align={align:g} seed={s}  ({el / 60:.1f} min elapsed)",
                      flush=True)
    print(f"\nwrote {a.out}  ({done} cells, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
