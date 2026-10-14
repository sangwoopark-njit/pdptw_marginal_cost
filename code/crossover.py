"""How long does ALNS need to match the decomposition's answer?

The comparison table gives ALNS the same wall-clock our method used.  That is the
fair budget-matched comparison, but it invites the obvious question: what happens
if the metaheuristic is simply given more time?  This measures it, by stopping
ALNS the moment its incumbent reaches our cost.
"""
import argparse, time, numpy as np, pandas as pd
import design as D, budget as B, cluster_sparse as CS, visualise as V
import clustering_baselines as CB, alns_baseline as A, routing
from check_scaling_parallel import StopAtTargetOrTime

def alns_to_target(inst, ehp, lhp, ehd, lhd, Tall, target, phi, cap_s, seed):
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    ctx = A.Ctx()
    ctx.n=inst.n_req; ctx.N=2*inst.n_req; ctx.T=Tall
    ctx.e=np.concatenate([ehp,ehd]); ctx.l=np.concatenate([lhp,lhd])
    ctx.S=np.concatenate([inst.sp,inst.sd]); ctx.theta=inst.theta
    ctx.cap=inst.cap; ctx.rho=1.0; ctx.phi=phi; ctx.count_cap=True
    routing.set_nhalf(ctx.n)
    rng=np.random.default_rng(seed)
    init=A.greedy_repair(A.PDPState([[] for _ in range(inst.n_veh)], set(range(ctx.n)), ctx), rng)
    al=ALNS(rng)
    for f in (A.random_removal,A.worst_removal,A.shaw_removal): al.add_destroy_operator(f)
    for f in (A.greedy_repair,A.regret_repair): al.add_repair_operator(f)
    stop=StopAtTargetOrTime(target, cap_s)
    t0=time.perf_counter()
    res=al.iterate(init, RouletteWheel([3,2,1,0.5],0.8,3,2),
                   SimulatedAnnealing.autofit(init.objective(),0.05,0.5,2000), stop)
    return stop.hit, time.perf_counter()-t0, res.best_state.objective(), \
        components(res.best_state)


def components(state):
    """Split an ALNS state into travel, tardiness and couriers dispatched.

    The comparison must report what the cost is made of, not only its total, so
    every route is re-evaluated with the same exact scheduler used everywhere
    else.
    """
    c = state.ctx
    tr_t = pen_t = 0.0
    used = 0
    for k, seq in enumerate(state.routes):
        if not seq:
            continue
        tr, pn, _ = routing._evaluate(seq, c.T, c.e, c.l, c.S, c.theta[k],
                                      c.N + k, c.rho)
        tr_t += tr; pen_t += pn; used += 1
    return dict(travel=tr_t, tardiness=pen_t, couriers=used)

ap=argparse.ArgumentParser()
ap.add_argument("--fleets", default="30,60,120")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--cap", type=float, default=300.0)
ap.add_argument("--phi", type=float, default=100.0)
a=ap.parse_args()
rows=[]
for K in [int(x) for x in a.fleets.split(",")]:
    d=D.design(K,5/6,4)
    for s in range(a.seeds):
        inst,_=D.make_instance(d,s,0.0)
        ehp,lhp,ehd,lhd,Tall,Top=B.prep(inst)
        t0=time.time()
        r=CS.run(inst,ehp,lhp,ehd,lhd,Tall,Top,phi=a.phi,seed=s,assign='mip',
                 mip_limit=5.0,cand=8,T_out=4)
        ours_t=time.time()-t0
        m=CB._evaluate(inst,ehp,lhp,ehd,lhd,Tall,V.assignment_from_cluster(r,K),1.0,0.0)
        target=m["travel"]+m["tardiness"]+a.phi*m["couriers"]
        hits=[]
        for sd in range(2):
            hit,tot,fin,comp=alns_to_target(inst,ehp,lhp,ehd,lhd,Tall,target,
                                            a.phi,a.cap,sd)
            hits.append((hit,fin,comp))
        matched=[h for h,_,_ in hits if h is not None]
        # with an even number of seeds there is no median seed: order[len//2]
        # is the upper middle (the WORSE of two), while alns_best below is the
        # median cost, i.e. the mean of the two.  The alns_* components are
        # therefore not the components of alns_best; per-seed columns follow.
        order=sorted(range(len(hits)), key=lambda i: hits[i][1])
        med=order[len(order)//2]
        per_seed={}
        for i,(_,fin,comp) in enumerate(hits):
            per_seed.update({f"alns_cost_s{i}":fin,
                             f"alns_travel_s{i}":comp["travel"],
                             f"alns_tard_s{i}":comp["tardiness"],
                             f"alns_cour_s{i}":comp["couriers"]})
        rows.append(dict(R=d.n_req,K=K,seed=s,ours=target,ours_t=ours_t,
                         ours_travel=m["travel"], ours_tard=m["tardiness"],
                         ours_cour=m["couriers"],
                         matched=len(matched), n=len(hits),
                         alns_t=np.median(matched) if matched else np.nan,
                         alns_best=np.median([f for _,f,_ in hits]),
                         alns_travel=hits[med][2]["travel"],
                         alns_tard=hits[med][2]["tardiness"],
                         alns_cour=hits[med][2]["couriers"], **per_seed))
        print(f"  |R|={d.n_req:4d} s={s}: ours {target:8.0f} in {ours_t:5.1f}s | "
              f"ALNS matched {len(matched)}/{len(hits)}"
              + (f" in {np.median(matched):6.1f}s ({np.median(matched)/ours_t:4.1f}x ours)"
                 if matched else f", best after {a.cap:.0f}s = {np.median([f for _,f,_ in hits]):8.0f} "
                 f"({100*(np.median([f for _,f,_ in hits])-target)/target:+.0f}%)"), flush=True)
pd.DataFrame(rows).to_csv("results_crossover.csv",index=False)
