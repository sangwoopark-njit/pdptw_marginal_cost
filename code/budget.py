"""
E9: fixed compute budget, and E10: regime characterisation.

E9 is the axis the paper never tests.  Two readings are reported, because they
answer different questions:

  anytime   every method gets the SAME wall-clock budget B.  Ours is run with
            as many outer iterations / restarts as fit in B.  This is the fair
            "who wins in B seconds" comparison.
  native    ours runs to its natural stopping point, the others get the full B.
            This is the "we win despite using a fraction of the time"
            comparison, and the one that matters operationally when dispatch
            decisions are made every few minutes.

E10 runs the same comparison across `generator.PRESETS`, adversarial included.
A method characterised only on instances built to suit it has not been
characterised.
"""
from __future__ import annotations
import numpy as np, pandas as pd, time, argparse, sys
import instance as I, cluster, baselines, alns_baseline, generator as G

CENTRAL = dict(nu=5, mu=0.05, beta=0.3, rho=1.0, phi=0.0, kappa=0.15)


def prep(inst):
    ehp, lhp, ehd, lhd, _ = I.tighten(inst)
    Tall = I.travel_matrix(np.vstack([inst.xp, inst.xd, inst.origins]))
    Top = Tall[:inst.n_req, 2 * inst.n_req:]
    return ehp, lhp, ehd, lhd, Tall, Top


def ours_anytime(inst, ehp, lhp, ehd, lhd, Tall, Top, budget, seed, **kw):
    """Restart our method with fresh seeds until the budget is spent, keeping
    the best.  Gives the method an honest way to use a larger budget."""
    t0 = time.time(); best = None; restarts = 0
    while True:
        r = cluster.run(inst, ehp, lhp, ehd, lhd, Tall, Top,
                        seed=seed + 1000 * restarts, T_out=6,
                        use_exclusion=True, use_splitting=True, **kw)
        restarts += 1
        if best is None or r["cost"] < best["cost"]:
            best = r
        if time.time() - t0 > budget * 0.85 or restarts >= 50:
            break
    best["restarts"] = restarts
    best["secs"] = time.time() - t0
    return best


def run_one(inst, tag, seed, budget, with_mip=False, mip_limit=120):
    ehp, lhp, ehd, lhd, Tall, Top = prep(inst)
    rho, phi = CENTRAL["rho"], CENTRAL["phi"]
    out = dict(instance=tag, seed=seed, budget=budget,
               n_req=inst.n_req, n_veh=inst.n_veh)
    out.update(G.describe(inst))

    t0 = time.time()
    nat = cluster.run(inst, ehp, lhp, ehd, lhd, Tall, Top, seed=seed, T_out=6,
                      use_exclusion=True, use_splitting=True, **CENTRAL)
    out["ours_native"] = nat["cost"]; out["ours_native_secs"] = time.time() - t0

    at = ours_anytime(inst, ehp, lhp, ehd, lhd, Tall, Top, budget, seed, **CENTRAL)
    out["ours_anytime"] = at["cost"]; out["ours_restarts"] = at["restarts"]
    out["ours_anytime_secs"] = at["secs"]

    ort = baselines.ortools_solve(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, budget)
    out["ortools"] = ort[0] if ort else np.nan
    out["ortools_secs"] = ort[4] if ort else np.nan

    aln = alns_baseline.alns_solve(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, budget, seed=seed)
    out["alns"] = aln[0]; out["alns_secs"] = aln[4]

    sl = baselines.single_level(inst, ehp, lhp, ehd, lhd, Tall, rho, phi, seed=seed)
    out["single_level"] = sl[0]

    if with_mip:
        m, b, st, ms = baselines.monolithic_mip(inst, ehp, lhp, ehd, lhd, Tall,
                                                rho, phi, mip_limit)
        out.update(mip=m, mip_bound=b, mip_status=st, mip_secs=ms)
    return out


def E9(seeds=5, budgets=(10,), n_req=40, n_veh=8, preset="neutral"):
    """Fixed-budget comparison on generated instances of one regime."""
    rows = []; t0 = time.time()
    for B in budgets:
        for s in range(seeds):
            inst = G.generate(n_req=n_req, n_veh=n_veh, seed=s, **G.PRESETS[preset])
            rows.append(run_one(inst, preset, s, B))
            r = rows[-1]
            print(f"  {preset} s{s} B={B}s: ours_nat={r['ours_native']:.1f}"
                  f"({r['ours_native_secs']:.2f}s) any={r['ours_anytime']:.1f}"
                  f"[{r['ours_restarts']}x] ort={r['ortools']:.1f} alns={r['alns']:.1f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"results_E9_{preset}.csv", index=False)
    print(f"wrote results_E9_{preset}.csv")


def E10(seeds=5, budget=10, n_req=40, n_veh=8):
    """The same comparison across all three regimes, adversarial included."""
    rows = []; t0 = time.time()
    for preset in ("favourable", "neutral", "adversarial"):
        for s in range(seeds):
            inst = G.generate(n_req=n_req, n_veh=n_veh, seed=s, **G.PRESETS[preset])
            rows.append(run_one(inst, preset, s, budget))
            r = rows[-1]
            print(f"  {preset:11s} s{s}: ours={r['ours_anytime']:.1f} "
                  f"ort={r['ortools']:.1f} alns={r['alns']:.1f} sl={r['single_level']:.1f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("results_E10.csv", index=False)
    print("wrote results_E10.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["E9", "E10"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--n_req", type=int, default=40)
    ap.add_argument("--n_veh", type=int, default=8)
    ap.add_argument("--preset", default="neutral")
    a = ap.parse_args()
    if a.exp == "E9":
        E9(a.seeds, (a.budget,), a.n_req, a.n_veh, a.preset)
    else:
        E10(a.seeds, a.budget, a.n_req, a.n_veh)
