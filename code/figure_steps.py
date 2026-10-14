"""The method working, step by step, on one real instance.

The pipeline figure says what the steps are; this one shows them happening, in
the way a k-means figure shows centroids moving.  Every panel is the same
instance at a different point in the outer loop, drawn from a trace the solver
emits rather than from a reconstruction, so what is plotted is what ran.

    python3 figure_steps.py                 # greyscale, for print
    python3 figure_steps.py --colour

Panels
    (a) the instance as posed, pickup -> delivery
    (b) construction: nearest courier with room, in release order
    (c) after the first assignment program
    (d) after the first descent
    (e--g) the same pair at later outer iterations
    (h) cost and fleet against stage, so the two axes show what each step buys
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import budget as B, design as D
import cluster_sparse as CS
import clustering_baselines as CB
import figure_paper as FP
from figure_paper import (pal, square_limits, route_of, draw_instance,
                          _finish)

# this figure is reduced by about 0.6 into the text block, so the
# drawing needs type near 11pt for roughly 7pt on the page

PROD = dict(assign="mip", mip_limit=5.0, cand=8, T_out=4)


def draw_groups(ax, inst, lab, title, colour, lims, show_routes=False,
                ehp=None, lhp=None, ehd=None, lhd=None, Tall=None):
    """Each request tied to the courier that holds it.

    Before routing there is no sequence to draw, so a group is shown as the
    requests that belong to it joined to their courier: that is exactly what the
    assignment decides, and it keeps the panels comparable.
    """
    P, n = pal(colour), inst.n_req
    mid = 0.5 * (inst.xp + inst.xd)
    for k in range(inst.n_veh):
        mem = np.where(lab == k)[0]
        if not len(mem):
            continue
        c = P[k % len(P)]
        if show_routes:
            seq, _ = route_of(inst, ehp, lhp, ehd, lhd, Tall, mem.tolist(), k)
            xs = [inst.origins[k][0]] + [(inst.xp[i, 0] if i < n
                                          else inst.xd[i - n, 0]) for i in seq]
            ys = [inst.origins[k][1]] + [(inst.xp[i, 1] if i < n
                                          else inst.xd[i - n, 1]) for i in seq]
            ax.plot(xs, ys, color=c, lw=1.1, zorder=2)
            # the route visits pickups and deliveries, so those are the nodes to
            # draw.  Marking group midpoints here instead -- as the grouping
            # panels do -- left the lines passing through nothing.
            ax.scatter(xs[1:], ys[1:], color=c, s=16, zorder=3,
                       edgecolors="white", linewidths=0.4)
            ax.scatter(*inst.origins[k], color=c, s=90, marker="*", zorder=4,
                       edgecolors="white", linewidths=0.5)
            continue
        else:
            # the group as a region rather than a bundle of spokes: with eight
            # couriers the spokes cross each other and hide the partition, which
            # is the thing these panels exist to show
            pts = np.vstack([mid[mem], inst.origins[k][None, :]])
            if len(pts) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    h = ConvexHull(pts)
                    poly = pts[h.vertices]
                    ax.fill(poly[:, 0], poly[:, 1], color=c, alpha=0.13,
                            zorder=1, linewidth=0)
                    ax.plot(np.append(poly[:, 0], poly[0, 0]),
                            np.append(poly[:, 1], poly[0, 1]),
                            color=c, lw=0.7, alpha=0.75, zorder=2)
                except Exception:
                    pass
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=c, lw=0.7, alpha=0.75,
                        zorder=2)
        ax.scatter(mid[mem, 0], mid[mem, 1], color=c, s=14, zorder=3,
                   edgecolors="white", linewidths=0.3)
        ax.scatter(*inst.origins[k], color=c, s=70, marker="*", zorder=4,
                   edgecolors="white", linewidths=0.4)
    idle = [k for k in range(inst.n_veh) if not len(np.where(lab == k)[0])]
    if idle:
        ax.scatter(inst.origins[idle, 0], inst.origins[idle, 1], s=55,
                   marker="*", facecolor="none", edgecolors="0.6",
                   linewidths=0.8, zorder=4)
    _finish(ax, title, lims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_veh", type=int, default=12)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--alignment", type=float, default=1.0)
    ap.add_argument("--colour", action="store_true")
    ap.add_argument("--out", default="../figures/fig_steps")
    a = ap.parse_args()

    d = D.design(a.n_veh, a.f, a.nu)
    inst, mis = D.make_instance(d, a.seed, a.alignment)
    ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
    lims = square_limits(inst)

    tr = []
    CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=a.phi, seed=a.seed,
           trace=tr, **PROD)
    print(f"{len(tr)} traced stages")
    for t in tr:
        print(f"  it={t['iter']:2d} {t['stage']:12s} cost {t['cost']:8.1f} "
              f"couriers {t['couriers']:3d}")

    # Laid out in inches rather than by tight_layout.  The figure is scaled to
    # \textwidth in the document, so canvas and type shrink together: growing
    # the canvas to cure an overlap shrinks the panels on the page as well.
    # Fixing every margin in inches and solving for the canvas keeps the panels
    # as large as the fixed overheads allow, at a known on-page type size.
    PANEL = 1.72            # each square panel, inches
    WGAP = 0.30             # between columns
    L, R = 0.06, 0.60       # margins; the right one holds panel (h)'s labels
    TITLE = 0.70            # three lines at TITLE_FS, plus axes.titlepad
    SUP = 0.30              # the running head
    # panel (h)'s rotated tick labels, measured: at 0.92 they used 27pt of the
    # 66 reserved and the other 39 came out as blank canvas under the artwork,
    # which on the page is a 30pt hole between the figure and its caption
    BOT = 0.44
    VGAP = TITLE + 0.18     # between the rows: row 2's titles live here

    W = L + 4 * PANEL + 3 * WGAP + R
    FP.set_sizes(W)          # one on-page type size for every figure
    H = SUP + TITLE + 2 * PANEL + VGAP + BOT
    fig, AX = plt.subplots(
        2, 4, figsize=(W, H),
        gridspec_kw=dict(left=L / W, right=1 - R / W,
                         bottom=BOT / H, top=1 - (SUP + TITLE) / H,
                         wspace=WGAP / PANEL, hspace=VGAP / PANEL))
    print(f"canvas {W:.2f} x {H:.2f} in; at 6.5in text width the panels are "
          f"{PANEL * 6.5 / W:.2f} in and the titles "
          f"{FP.TITLE_FS * 6.5 / W:.1f} pt on the page")

    def cap(lbl, t):
        # A panel is about 2.4in wide.  Spelling out "travel"/"late" made the
        # line wider than that and it ran into the neighbouring panel, so the
        # three terms are given bare and named once in the caption instead.
        return (f"{lbl}\ncost {t['cost']:.0f}\n"
                f"{t['travel']:.0f} $+$ {t['tardiness']:.0f} "
                f"$+$ {a.phi:g}$\\times${t['couriers']}")

    # What the filled and open circles mean holds for every map panel, not
    # just this one, so it reads as a legend rather than a title and lives in
    # the caption instead.
    draw_instance(AX[0, 0], inst,
                  f"(a) the instance\n"
                  f"misalignment {mis:.2f}", lims)

    con = tr[0]
    draw_groups(AX[0, 1], inst, con["labels"],
                cap("(b) construction", con), a.colour, lims)

    a0 = [t for t in tr if t["stage"] == "assignment" and t["iter"] == 0][0]
    d0 = [t for t in tr if t["stage"] == "descent" and t["iter"] == 0][0]
    draw_groups(AX[0, 2], inst, a0["labels"],
                cap("(c) iter. 1: assignment", a0), a.colour, lims)
    draw_groups(AX[0, 3], inst, d0["labels"],
                cap("(d) iter. 1: descent", d0), a.colour, lims)

    later = [t for t in tr if t["stage"] == "descent" and t["iter"] >= 1]
    for j, t in enumerate(later[:3]):
        last = (j == len(later[:3]) - 1)
        # "after descent" set at 9pt is wider than a 1.32in panel, so the
        # titles of neighbouring panels ran into each other.  The step a panel
        # shows is named in one word and the caption carries the sentence.
        head = f"({'efg'[j]}) iter. {t['iter'] + 1}: descent"
        if last:
            head = f"({'efg'[j]}) iter. {t['iter'] + 1}: routed"
        draw_groups(AX[1, j], inst, t["labels"], cap(head, t),
                    a.colour, lims, show_routes=last, ehp=ehp, lhp=lhp,
                    ehd=ehd, lhd=lhd, Tall=Tall)

    # ---- the trajectory, on two axes: what each step buys, and at what fleet
    ax = AX[1, 3]
    labels, costs, cours = [], [], []
    for t in tr:
        tag = ("constr." if t["stage"] == "construction"
               else ("asg" if t["stage"] == "assignment" else "desc")
               + f"{t['iter'] + 1}")
        labels.append(tag); costs.append(t["cost"]); cours.append(t["couriers"])
    x = np.arange(len(tr))
    ax.plot(x, costs, marker="o", ms=4, lw=1.4, color="0.15", zorder=3,
            label="cost")
    # the map panels are squared by set_aspect("equal") and so shrink inside
    # their cell; without this the trajectory filled its cell, came out taller
    # than every other panel, and pushed its y-label over panel (g)
    ax.set_box_aspect(1)
    ax.set_ylabel("cost", fontsize=9, labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=60, ha="right")
    # a coarse grid: five ticks read as well as ten at this size
    ax.locator_params(axis="y", nbins=4)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(alpha=0.25, lw=0.35)
    ax2 = ax.twinx()
    ax2.set_box_aspect(1)
    ax2.bar(x, cours, color="0.75", width=0.55, zorder=1, label="couriers")
    ax2.set_ylabel("couriers", fontsize=9, labelpad=1)
    ax2.locator_params(axis="y", nbins=4)
    ax2.tick_params(axis="y", labelsize=8, pad=1)
    ax2.set_ylim(0, max(cours) * 1.9)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    # which series is which axis is a legend too, and the caption already
    # says it; the title keeps only what names the panel
    ax.set_title("(h) what each step buys", fontsize=FP.TITLE_FS)
    for sp in list(ax.spines.values()) + list(ax2.spines.values()):
        sp.set_linewidth(0.5)

    # the explanation belongs in the LaTeX caption, where it can wrap; stretched
    # across the artwork it ran off both edges
    fig.suptitle(f"One instance: $|\\mathcal{{R}}|={d.n_req}$, "
                 f"$|\\mathcal{{K}}|={d.n_veh}$, $Q_k={d.cap}$, "
                 f"$\\varphi={a.phi:g}$, $\\rho=1$, seed {a.seed}",
                 fontsize=12)
    fig.savefig(f"{a.out}.pdf")
    fig.savefig(f"{a.out}.png", dpi=150)
    plt.close(fig)
    print(f"   wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
