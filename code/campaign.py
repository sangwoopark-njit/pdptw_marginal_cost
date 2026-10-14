"""
Campaign runner.  Each experiment in experimental_design.md is a function here;
`python3 campaign.py E1 --seeds 5` runs one.  Results go to results_<E>.csv.

Pilot sizes are small so that the exact MIP proves optimality; the scaling
experiments (E2, E3 at large |K|) need a bigger machine and are parameterised
for it.
"""
import numpy as np, pandas as pd, time, argparse, sys
import instance as I, cluster, baselines, alns_baseline

FAMS = ["lr201", "lrc201", "lrc101"]
CENTRAL = dict(nu=5, mu=0.05, beta=0.3, rho=1.0, phi=0.0, kappa=0.15, T_out=6)
ALPHA = 0.5          # dimensionless temporal weight; omega_t = ALPHA * sigma_x / sigma_t


def temporal_scale(inst, ehp, lhd):
    """omega_t that puts the temporal term of C_rc on the same scale as the
    spatial one.  sigma_x is the mean per-axis spread of the stops, sigma_t the
    spread of the request-interval midpoints.  Leaving omega_t = 1 over-weights
    time by an order of magnitude or more and wrecks the clustering."""
    import numpy as _np
    sx = _np.vstack([inst.xp, inst.xd]).std(0).mean()
    st = ((ehp + lhd) / 2).std()
    return float(sx / max(st, 1e-9))


def load(fam, seed, n_req, n_veh, cap=None, slice_frac=0.25, slack=1.3):
    rng = np.random.default_rng(seed)
    if cap is None:                       # binding, but with room to choose
        cap = max(1, int(np.ceil(slack * n_req / n_veh)))
    inst = I.build_instance(f"data/{fam}.txt", fam, n_req, n_veh, cap, rng,
                            stagger=0.03, spread=3.0, slice_frac=slice_frac)
    ehp, lhp, ehd, lhd, _ = I.tighten(inst)
    Tall = I.travel_matrix(np.vstack([inst.xp, inst.xd, inst.origins]))
    Top = Tall[:inst.n_req, 2 * inst.n_req:]
    return inst, ehp, lhp, ehd, lhd, Tall, Top


def ours_kwargs(inst, ehp, lhd, **over):
    """CENTRAL plus a correctly scaled omega_t for this instance."""
    kw = dict(CENTRAL); kw.update(over)
    kw["ws"] = 1.0
    kw["wt"] = ALPHA * temporal_scale(inst, ehp, lhd)
    return kw


def E1_E3_pilot(seeds, n_req=8, n_veh=2, cap=5, mip_limit=45, ort_limit=5):
    """Ground truth (E1) and single-level comparison (E3) on instances small
    enough for HiGHS to prove optimality."""
    rows = []; t0 = time.time()
    for fam in FAMS:
        for s in range(seeds):
            inst, ehp, lhp, ehd, lhd, Tall, Top = load(fam, s, n_req, n_veh, cap)
            mip, bnd, st, msec = baselines.monolithic_mip(inst, ehp, lhp, ehd, lhd, Tall,
                                                          CENTRAL["rho"], CENTRAL["phi"], mip_limit)
            t1 = time.time()
            ours = cluster.run(inst, ehp, lhp, ehd, lhd, Tall, Top, seed=s,
                               use_exclusion=True, use_splitting=True, **CENTRAL)
            osec = time.time() - t1
            ort = baselines.ortools_solve(inst, ehp, lhp, ehd, lhd, Tall,
                                          CENTRAL["rho"], CENTRAL["phi"], ort_limit)
            sl = baselines.single_level(inst, ehp, lhp, ehd, lhd, Tall,
                                        CENTRAL["rho"], CENTRAL["phi"], seed=s)
            press = ours["penalty"] / max(ours["travel"], 1e-9)
            rows.append(dict(family=fam, seed=s, mip=mip, mip_bound=bnd, mip_status=st,
                             mip_secs=msec, ours=ours["cost"], ours_secs=osec,
                             ortools=ort[0] if ort else np.nan, ortools_secs=ort[4] if ort else np.nan,
                             single_level=sl[0], pressure=press))
            print(f"  {fam} s{s}: mip={mip:.1f}[{st}] ours={ours['cost']:.1f} "
                  f"ort={rows[-1]['ortools']:.1f} sl={sl[0]:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results_E1_E3.csv", index=False)
    return df




# ---------------------------------------------------------------- E2 ----
def E2_scaling(seeds=5, sizes=((20,4),(40,8),(80,16),(160,32)), cap=6,
               mip_limit=600, heur_limit=30):
    """Runtime and quality as |R| and |K| grow together (H2)."""
    rows=[]; t0=time.time()
    for (n_req,n_veh) in sizes:
        for fam in FAMS:
            for s in range(seeds):
                inst,ehp,lhp,ehd,lhd,Tall,Top = load(fam,s,n_req,n_veh,cap)
                kw = dict(CENTRAL); kw["nu"] = max(3, n_req//n_veh)
                t1=time.time()
                ours = cluster.run(inst,ehp,lhp,ehd,lhd,Tall,Top,seed=s,
                                   use_exclusion=True,use_splitting=True,**kw)
                osec=time.time()-t1
                ort = baselines.ortools_solve(inst,ehp,lhp,ehd,lhd,Tall,
                                              CENTRAL["rho"],CENTRAL["phi"],heur_limit)
                aln = alns_baseline.alns_solve(inst,ehp,lhp,ehd,lhd,Tall,
                                               CENTRAL["rho"],CENTRAL["phi"],heur_limit,seed=s)
                mip=bnd=np.nan; st="skipped"; msec=np.nan
                if n_req<=20:
                    mip,bnd,st,msec = baselines.monolithic_mip(inst,ehp,lhp,ehd,lhd,Tall,
                                          CENTRAL["rho"],CENTRAL["phi"],mip_limit)
                rows.append(dict(family=fam,seed=s,n_req=n_req,n_veh=n_veh,
                    ours=ours["cost"],ours_secs=osec,
                    ortools=ort[0] if ort else np.nan, ortools_secs=ort[4] if ort else np.nan,
                    alns=aln[0],alns_secs=aln[4],
                    mip=mip,mip_bound=bnd,mip_status=st,mip_secs=msec))
                print(f"  |R|={n_req} |K|={n_veh} {fam} s{s}: ours={ours['cost']:.1f}/{osec:.1f}s "
                      f"ort={rows[-1]['ortools']:.1f} alns={aln[0]:.1f}  ({time.time()-t0:.0f}s)",flush=True)
    pd.DataFrame(rows).to_csv("results_E2.csv",index=False); print("wrote results_E2.csv")


# ---------------------------------------------------------------- E3 ----
def E3_bilevel(seeds=10, fleets=(4,8,16), ratios=(3,5,8), cap=6):
    """Full method vs single-level clustering (H7)."""
    rows=[]; t0=time.time()
    for n_veh in fleets:
        for ratio in ratios:
            n_req=n_veh*ratio
            for fam in FAMS:
                for s in range(seeds):
                    inst,ehp,lhp,ehd,lhd,Tall,Top = load(fam,s,n_req,n_veh,cap)
                    kw=dict(CENTRAL); kw["nu"]=max(3,ratio)
                    ours=cluster.run(inst,ehp,lhp,ehd,lhd,Tall,Top,seed=s,
                                     use_exclusion=True,use_splitting=True,**kw)
                    sl=baselines.single_level(inst,ehp,lhp,ehd,lhd,Tall,
                                              CENTRAL["rho"],CENTRAL["phi"],seed=s)
                    rows.append(dict(family=fam,seed=s,n_veh=n_veh,ratio=ratio,n_req=n_req,
                                     ours=ours["cost"],single_level=sl[0],
                                     ours_travel=ours["travel"],ours_pen=ours["penalty"]))
            print(f"  |K|={n_veh} ratio={ratio} done ({time.time()-t0:.0f}s)",flush=True)
    pd.DataFrame(rows).to_csv("results_E3.csv",index=False); print("wrote results_E3.csv")


# ---------------------------------------------------------------- E4 ----
def E4_temporal(seeds=6, n_req=24, n_veh=4, cap=6,
                targets=(0.1,0.3,0.5,0.7), alphas=(0.0,0.25,0.5,1.0,2.0)):
    """Temporal awareness under CONTROLLED interval overlap (H4).

    Window widths are rescaled about their midpoints until the fraction of
    temporally disjoint request pairs hits the target, so the temporal axis
    means what the paper says it means."""
    rows=[]; t0=time.time()
    for target in targets:
        for fam in FAMS:
            for s in range(seeds):
                inst,ehp,lhp,ehd,lhd,Tall,Top = load(fam,s,n_req,n_veh,cap)
                ehp,lhp,ehd,lhd,frac = _rescale_windows(ehp,lhp,ehd,lhd,target)
                sx=np.vstack([inst.xp,inst.xd]).std(0).mean()
                for mode in ("request","pickup"):
                    hi = lhd if mode=="request" else lhp
                    st_=((ehp+hi)/2).std(); scale=sx/max(st_,1e-9)
                    for al in alphas:
                        r=cluster.run(inst,ehp,lhp,ehd,lhd,Tall,Top,seed=s,
                                      ws=1.0,wt=al*scale,iv_mode=mode,
                                      use_exclusion=True,use_splitting=True,**CENTRAL)
                        rows.append(dict(family=fam,seed=s,target=target,disjoint=frac,
                                         iv=mode,alpha=al,cost=r["cost"],
                                         travel=r["travel"],penalty=r["penalty"]))
        print(f"  overlap target {target} done ({time.time()-t0:.0f}s)",flush=True)
    pd.DataFrame(rows).to_csv("results_E4.csv",index=False); print("wrote results_E4.csv")


def _rescale_windows(ehp, lhp, ehd, lhd, target, tol=0.02, iters=60):
    """
    Control the fraction of temporally DISJOINT request pairs.

    Shrinking window widths alone saturates near 0.31, because the request
    interval I_r = [e_p, l_d] spans the pickup-to-delivery gap however narrow
    the windows are.  We therefore stretch the batch along the time axis about
    its mean, keeping each window's width fixed, so intervals move apart
    without becoming degenerate.  Returns the new windows and the achieved
    disjoint fraction.
    """
    mp, md = (ehp + lhp) / 2, (ehd + lhd) / 2
    hp, hd = (lhp - ehp) / 2, (lhd - ehd) / 2
    M = np.concatenate([mp, md]).mean()

    ctr = (mp + md) / 2                        # each request's own centre
    dmp, dmd = mp - ctr, md - ctr              # its internal pickup->delivery offsets
    M = ctr.mean()

    def build(g):
        # stretch requests apart from EACH OTHER; keep each request's own
        # internal gap and window widths fixed, so |I_r| does not grow with g
        ctr2 = M + g * (ctr - M)
        mp2, md2 = ctr2 + dmp, ctr2 + dmd
        return mp2 - hp, mp2 + hp, md2 - hd, md2 + hd

    def frac(g):
        a, _, _, b = build(g)                  # I_r = [e_p, l_d]
        n = len(a); c = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1
                if b[i] < a[j] or b[j] < a[i]: c += 1
        return c / max(tot, 1)

    lo, hi = 0.2, 40.0
    if frac(hi) < target:                      # cannot reach it; return the widest spread
        e1, l1, e2, l2 = build(hi); return e1, l1, e2, l2, frac(hi)
    for _ in range(iters):
        g = 0.5 * (lo + hi); got = frac(g)
        if abs(got - target) < tol: break
        if got < target: lo = g
        else: hi = g
    g = 0.5 * (lo + hi)
    e1, l1, e2, l2 = build(g)
    return e1, l1, e2, l2, frac(g)


# ---------------------------------------------------------------- E6 ----
def E6_nu(seeds=5, n_req=80, n_veh=8, cap=8, nus=(3,5,8,12)):
    """The nu trade-off (H6): subproblem size against assignment freedom."""
    rows=[]; t0=time.time()
    for nu in nus:
        for fam in FAMS:
            for s in range(seeds):
                inst,ehp,lhp,ehd,lhd,Tall,Top = load(fam,s,n_req,n_veh,cap)
                kw=dict(CENTRAL); kw["nu"]=nu
                t1=time.time()
                r=cluster.run(inst,ehp,lhp,ehd,lhd,Tall,Top,seed=s,
                              use_exclusion=True,use_splitting=True,**kw)
                rows.append(dict(family=fam,seed=s,nu=nu,cost=r["cost"],
                                 travel=r["travel"],penalty=r["penalty"],
                                 secs=time.time()-t1,n_cl=r["n_cl_final"]))
        print(f"  nu={nu} done ({time.time()-t0:.0f}s)",flush=True)
    pd.DataFrame(rows).to_csv("results_E6.csv",index=False); print("wrote results_E6.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    fns = {"E1": E1_E3_pilot, "E2": E2_scaling, "E3": E3_bilevel,
           "E4": E4_temporal, "E6": E6_nu}
    if a.exp not in fns:
        print("choose one of:", ", ".join(fns)); sys.exit(1)
    fns[a.exp](a.seeds)
