"""
Three questions that the earlier experiments cannot answer:

  1. Does our method SCALE better than ALNS?  The tables so far give ALNS a fixed
     time BUDGET, so its "seconds" column is an input, not a measurement, and no
     scaling claim can be read off it.  The answerable version is: HOW LONG DOES
     ALNS NEED TO REACH THE QUALITY OUR METHOD PRODUCES?  That is measured here
     with a stopping criterion that halts ALNS the moment it matches our cost.

  2. How much of our runtime is parallelisable?  The per-courier routing
     subproblems are independent, which is the decomposition's structural claim.
     This measures what fraction of the wall-clock that actually is.

  3. Does ALNS benefit from trivial parallelism too?  It is stochastic, so P
     cores buy it best-of-P independent runs.  Our method is deterministic, so
     P cores buy it nothing on that axis.  This measures the size of that effect,
     because a parallelism argument has to be made against a parallel opponent.

    python3 check_scaling_parallel.py [--reps 3] [--cap 30]
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import generator2 as G2, budget as B
import cluster2 as K2, visualise as V
import alns_baseline as A
import routing

pd.set_option("display.width", 220)


class StopAtTargetOrTime:
    """Stop as soon as ALNS reaches `target`, or when `cap` seconds have passed.

    Same interface as alns.stop.MaxRuntime.  `hit` records the time at which the
    target was reached, or None if it never was.
    """

    def __init__(self, target: float, cap: float):
        self.target = target
        self.cap = cap
        self._start = None
        self.hit = None

    def __call__(self, rng, best, current) -> bool:
        if self._start is None:
            self._start = time.perf_counter()
        elapsed = time.perf_counter() - self._start
        if best.objective() <= self.target + 1e-9:
            if self.hit is None:
                self.hit = elapsed
            return True
        return elapsed > self.cap


def alns_time_to_reach(inst, ehp, lhp, ehd, lhd, Tall, target, cap, seed):
    """Seconds ALNS needs to match `target`, or None if it never does in `cap`."""
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel

    ctx = A.Ctx()
    ctx.n = inst.n_req; ctx.N = 2 * inst.n_req; ctx.T = Tall
    ctx.e = np.concatenate([ehp, ehd]); ctx.l = np.concatenate([lhp, lhd])
    ctx.S = np.concatenate([inst.sp, inst.sd]); ctx.theta = inst.theta
    ctx.cap = inst.cap; ctx.rho = 1.0; ctx.phi = 0.0
    routing.set_nhalf(ctx.n)

    rng = np.random.default_rng(seed)
    init = A.greedy_repair(
        A.PDPState([[] for _ in range(inst.n_veh)], set(range(ctx.n)), ctx), rng)
    alns = ALNS(rng)
    for f in (A.random_removal, A.worst_removal, A.shaw_removal):
        alns.add_destroy_operator(f)
    for f in (A.greedy_repair, A.regret_repair):
        alns.add_repair_operator(f)
    stop = StopAtTargetOrTime(target, cap)
    t0 = time.perf_counter()
    alns.iterate(init, RouletteWheel([3, 2, 1, 0.5], 0.8, 3, 2),
                 SimulatedAnnealing.autofit(init.objective(), 0.05, 0.5, 2000),
                 stop)
    return stop.hit, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="instances per fleet size")
    ap.add_argument("--seeds", type=int, default=3, help="ALNS seeds for time-to-match")
    ap.add_argument("--pseeds", type=int, default=5, help="ALNS seeds for best-of-P")
    ap.add_argument("--cap", type=float, default=30.0)
    ap.add_argument("--pbudget", type=float, default=10.0)
    ap.add_argument("--ratio", type=int, default=6)
    a = ap.parse_args()

    rows = []
    for K in [2, 3, 4, 6, 8]:
        R = a.ratio * K
        for s in range(a.reps):
            align = [1.0, 0.0][s % 2]
            inst, _ = G2.generate(n_req=R, n_veh=K, seed=s, alignment=align)
            ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)

            t0 = time.perf_counter()
            r = K2.run(inst, ehp, lhp, ehd, lhd, Tall, Top, nu=3, seed=s,
                       polish=True, route_mode='final', route_top=2,
                       **K2.CHAIN_DEFAULTS)
            m = V.metrics(inst, ehp, lhp, ehd, lhd, Tall,
                          V.assignment_from_cluster(r, K))
            ours = m["travel"] + m["tardiness"]
            t_ours = time.perf_counter() - t0

            hits = []
            for sd in range(a.seeds):
                hit, _tot = alns_time_to_reach(inst, ehp, lhp, ehd, lhd, Tall,
                                               ours, a.cap, sd)
                hits.append(hit if hit is not None else np.nan)

            costs = [A.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0,
                                  time_limit=a.pbudget, seed=sd)[0]
                     for sd in range(a.pseeds)]

            rows.append(dict(K=K, R=R, seed=s, align=align,
                             ours=ours, t_ours=t_ours,
                             alns_match_med=np.nanmedian(hits),
                             alns_match_n=int(np.sum(~np.isnan(hits))),
                             alns_med=float(np.median(costs)),
                             alns_best=float(np.min(costs))))
            print(f"  K={K} s={s}: ours {ours:8.1f} in {t_ours:5.2f}s | "
                  f"ALNS matched by {np.nanmedian(hits) if np.any(~np.isnan(hits)) else float('nan'):6.2f}s "
                  f"({int(np.sum(~np.isnan(hits)))}/{a.seeds} seeds) | "
                  f"ALNS@{a.pbudget:g}s med {np.median(costs):8.1f} best {np.min(costs):8.1f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results_scaling_parallel.csv", index=False)

    print("\n=== 1. TIME ALNS NEEDS TO REACH OUR ONE-PASS QUALITY ===")
    print("    (workload per courier fixed at 6 requests; NaN = never within the cap)\n")
    t = df.groupby("K").agg(our_secs=("t_ours", "mean"),
                            alns_secs_to_match=("alns_match_med", "mean"),
                            seeds_that_matched=("alns_match_n", "sum"))
    t["ALNS/ours"] = t.alns_secs_to_match / t.our_secs
    print(t.round(2).to_string())

    print("\n=== 3. WHAT PARALLELISM BUYS ALNS (best of P runs vs its median) ===\n")
    df["best_of_P_gain"] = 100 * (df.alns_med - df.alns_best) / df.alns_med
    g = df.groupby("K").agg(gain_pct=("best_of_P_gain", "mean"))
    print(g.round(2).to_string())
    print(f"\n  mean over all sizes: {df.best_of_P_gain.mean():.2f}%  "
          f"(P = {a.pseeds} independent runs)")
    print("  Our method is deterministic (restart spread 0.0%), so the same P")
    print("  cores buy it nothing on this axis.")


if __name__ == "__main__":
    main()
