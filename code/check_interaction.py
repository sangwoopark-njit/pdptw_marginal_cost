"""How far from modular is the routing cost?

Section 3.7 bounds the error of the frozen assignment costs by the pairwise
interaction of J(.,k): if J were modular the assignment program would be an
exact reformulation, and otherwise the objective error on one courier is at most
(3Q_k^2 - Q_k)/2 times the largest second difference

    I(S; r, r') = J(S u {r,r'}) - J(S u {r}) - J(S u {r'}) + J(S).

A bound nobody has evaluated is not worth much, so this measures it with the
exact router, on the instances the paper uses.  Interactions are reported
relative to the cost of the set they are measured on, since an absolute number
in travel-time units is not interpretable on its own.

    python3 check_interaction.py --seeds 3
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd

import design as D, budget as B, routing
import clustering_baselines as CB


def J(req, inst, ehp, lhp, ehd, lhd, Tall, k, rho):
    """Exact cost of serving request set `req` on courier k."""
    if not len(req):
        return 0.0
    m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall, {k: list(req)}, rho, 0.0)
    return m["travel"] + rho * m["tardiness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_veh", type=int, default=12)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--out", default="results_paper_interaction.csv")
    a = ap.parse_args()

    d = D.design(a.n_veh, a.f, a.nu)
    rows, t0 = [], time.time()
    for align in (1.0, 0.0):
        for s in range(a.seeds):
            inst, mis = D.make_instance(d, s, align)
            ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
            routing.set_nhalf(inst.n_req)
            rng = np.random.default_rng(1000 + s)
            n, Q = inst.n_req, int(d.cap)

            for _ in range(a.samples):
                k = int(rng.integers(inst.n_veh))
                # a base set S of size 0..Q-2, plus two further requests
                size = int(rng.integers(0, max(1, Q - 1)))
                pick = rng.choice(n, size=size + 2, replace=False)
                S, r, rp = list(pick[:size]), int(pick[size]), int(pick[size + 1])

                jS = J(S, inst, ehp, lhp, ehd, lhd, Tall, k, a.rho)
                jr = J(S + [r], inst, ehp, lhp, ehd, lhd, Tall, k, a.rho)
                jp = J(S + [rp], inst, ehp, lhp, ehd, lhd, Tall, k, a.rho)
                jb = J(S + [r, rp], inst, ehp, lhp, ehd, lhd, Tall, k, a.rho)
                inter = jb - jr - jp + jS
                # the marginal of r measured at S, and at S u {r'}: the two the
                # frozen cost would confuse
                marg_S = jr - jS
                rows.append(dict(alignment=align, seed=s, misalign=mis,
                                 courier=k, set_size=size,
                                 J_base=jS, J_both=jb, interaction=inter,
                                 marginal=marg_S,
                                 rel_to_both=inter / max(jb, 1e-9),
                                 rel_to_marg=(abs(inter)
                                              / max(abs(marg_S), 1e-9))))
            print(f"  align={align:g} seed={s} done ({time.time()-t0:.0f}s)",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    print(f"\n{len(df)} sampled second differences, Q_k = {int(d.cap)}")
    print(f"  interaction / cost of the pair set : "
          f"mean {df.rel_to_both.mean():+.4f}, "
          f"median {df.rel_to_both.median():+.4f}, "
          f"95th pct |.| {df.rel_to_both.abs().quantile(0.95):.4f}")
    print(f"  |interaction| / |marginal|         : "
          f"mean {df.rel_to_marg.mean():.4f}, "
          f"median {df.rel_to_marg.median():.4f}")
    print(f"  exactly modular (interaction = 0)  : "
          f"{100 * (df.interaction.abs() < 1e-9).mean():.1f}% of samples")
    print(f"  bound Q_k * max|interaction| / J   : "
          f"{int(d.cap) * df.interaction.abs().max() / df.J_both.mean():.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
