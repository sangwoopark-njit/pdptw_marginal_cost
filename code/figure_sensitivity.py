"""The sweeps as plots rather than tables.

Four of the paper's tables are trends over one parameter, and a trend is easier
to read as a line than as five rows of numbers.  Each panel overlays the margin
against the baselines (left axis, lines) on the quantity that explains it (right
axis, bars), which is also how the journal's own figures present sensitivity
analyses.

    python3 figure_sensitivity.py

Sources
    results_paper_phisweep.csv + main + phi0   the dispatch charge
    results_paper_alignsweep.csv               the alignment axis
    results_paper_stweight.csv                 the temporal weight
    results_paper_rho.csv                      the penalty rate
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_paper as FP  # publisher-approved fonts, set on import

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ["sweep", "fisher", "kmeans", "st_kmeans", "cap_kmeans", "agglom"]


def rel(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 100.0 * np.mean((x - y) / y)


def margins(frame, key, idx=("alignment", "seed")):
    """Margin against published rules, strengthened rules and ALNS, per level."""
    out = []
    for v in sorted(frame[key].unique()):
        sub = frame[frame[key] == v]
        piv = sub.pivot_table(index=list(idx), columns="method", values="cost")
        out.append(dict(
            level=v,
            pub=rel(piv["ours"], piv[RULES].min(axis=1)),
            desc=rel(piv["ours"], piv[[r + "+ls" for r in RULES]].min(axis=1)),
            alns=rel(piv["ours"], piv["ALNS"]),
            cour=sub[sub.method == "ours"].couriers.mean()))
    return pd.DataFrame(out)


def style(ax, xlabel, title):
    ax.axhline(0, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel(xlabel, fontsize=FP.AXLAB_FS)
    ax.set_title(title, fontsize=FP.TITLE_FS)
    ax.tick_params(labelsize=FP.TICK_FS)
    ax.grid(alpha=0.25, lw=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def legend_in_headroom(room_ax, leg_ax, fig, loc, rows, ncol=1):
    """Put the legend above the data instead of on top of it.

    A fixed `loc` collides with whatever happens to occupy that corner: in (a)
    the box sat on the tops of the bars, in (d) the ALNS line ran straight
    through it.  Hunting for an empty corner is fragile, because which corner
    is free changes with the data.  Instead the y-axis is extended to open a
    band at the top that no series reaches, and the legend goes in it.

    The band is sized from the legend rather than guessed: `rows` lines of type
    plus the frame needs `need` points, and an axes `h` points tall showing a
    span `s` must be extended by d = s*need/(h-need) for that band to be free.

    `room_ax` is the axes whose limits are stretched -- the one carrying the
    series that would be overlapped -- which is not always the axes the legend
    belongs to, since in (d) the labels describe the bars on the twin.
    """
    fs = FP.TICK_FS - 0.8
    h = room_ax.get_position().height * fig.get_figheight() * 72.0
    need = rows * 1.45 * fs + 9.0          # lines of type, plus the frame
    lo, hi = room_ax.get_ylim()
    room_ax.set_ylim(lo, hi + (hi - lo) * need / max(h - need, 1.0))

    def place(size):
        return leg_ax.legend(fontsize=size, loc=loc, framealpha=0.9, ncol=ncol,
                             borderpad=0.3, labelspacing=0.3, handlelength=1.8)

    leg = place(fs)
    # The row count sizes the headroom, but the WIDTH depends on how long the
    # labels happen to be, and a legend wider than its axes spills over the
    # spine into the neighbouring panel.  Measured and shrunk until it fits,
    # rather than trusted.
    for _ in range(8):
        fig.canvas.draw()
        if (leg.get_window_extent().width
                <= room_ax.get_window_extent().width * 0.97):
            break
        fs *= 0.93
        leg.remove()
        leg = place(fs)
    return leg


def lines(ax, x, df):
    ax.plot(x, df.pub, marker="s", ms=4, lw=1.3, color="0.55",
            label="vs. published rules")
    ax.plot(x, df.desc, marker="o", ms=4, lw=1.5, color="0.05",
            label="vs. rules + our descent")
    ax.plot(x, df.alns, marker="^", ms=4, lw=1.2, color="0.35", ls="--",
            label="vs. ALNS")
    ax.set_ylabel("cost difference (%)", fontsize=FP.AXLAB_FS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figures/fig_sensitivity")
    a = ap.parse_args()

    # one canvas width for the whole paper's figures, so the type
    # arrives at the same size on the page as in the others
    W, H = 8.6, 6.2
    FP.set_sizes(W)
    print(f'canvas {W} x {H} in; titles '
          f'{FP.TITLE_FS * FP.TEXTWIDTH_IN / W:.1f} pt on the page')
    fig, AX = plt.subplots(2, 2, figsize=(W, H))

    # ---------------------------------------------------- (a) dispatch charge
    fr = [pd.read_csv(os.path.join(HERE, f)) for f in
          ("results_paper_main_phi0.csv", "results_paper_phisweep.csv",
           "results_paper_main.csv")]
    phi = pd.concat([f[f.n_req == 40] for f in fr], ignore_index=True)
    dphi = margins(phi, "phi")
    ax = AX[0, 0]
    xp = np.arange(len(dphi))
    lines(ax, xp, dphi)
    ax2 = ax.twinx()
    ax2.bar(xp, dphi.cour, width=0.55, color="0.85", zorder=0)
    ax2.set_ylabel("couriers dispatched", fontsize=FP.AXLAB_FS)
    ax2.set_ylim(0, dphi.cour.max() * 2.2)
    ax2.tick_params(labelsize=FP.TICK_FS)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xticks(xp)
    ax.set_xticklabels([f"{v:g}" for v in dphi.level])
    style(ax, "dispatch charge $\\varphi$",
          "(a)")
    legend_in_headroom(ax, ax, fig, "upper center", rows=3)

    # ------------------------------------------------------- (b) the alignment
    al = pd.read_csv(os.path.join(HERE, "results_paper_alignsweep.csv"))
    dal = margins(al, "alignment", idx=("seed",))
    mis = al.groupby("alignment").misalign.mean()
    tw = al.pivot_table(index=["alignment", "seed"], columns="method",
                        values="tardiness")
    pen = [tw.xs(v, level="alignment")["st_kmeans"].mean()
           / max(tw.xs(v, level="alignment")["kmeans"].mean(), 1e-9)
           for v in dal.level]
    ax = AX[0, 1]
    lines(ax, mis.values, dal)
    ax2 = ax.twinx()
    ax2.bar(mis.values, pen, width=0.07, color="0.85", zorder=0)
    ax2.set_ylabel("s.-t. $k$-means tardiness\nas a multiple of plain $k$-means",
                   fontsize=FP.AXLAB_FS)
    ax2.set_ylim(0, max(pen) * 2.2)
    ax2.tick_params(labelsize=FP.TICK_FS)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    style(ax, "realised misalignment",
          "(b)")

    # -------------------------------------------------- (c) the temporal weight
    st = pd.read_csv(os.path.join(HERE, "results_paper_stweight.csv"))
    ax = AX[1, 0]
    for lbl, var, ls_, col in [("published", "as published", "-", "0.05"),
                               ("$+$ descent", "+ descent", "--", "0.5")]:
        for alg, mk in [(0.0, "o"), (1.0, "s")]:
            sub = st[(st.variant == var) & (st.alignment == alg)]
            g = sub.groupby("alpha").tardiness.mean()
            base = g.loc[0.0]
            ax.plot(np.arange(len(g)), g / max(base, 1e-9), marker=mk, ms=4, lw=1.3,
                    ls=ls_, color=col,
                    label=f"{lbl}, {'misaligned' if alg == 0 else 'aligned'}")
    ax.axhline(1.0, color="0.5", lw=0.6, ls=":")
    ax.set_xticks(np.arange(len(sorted(st.alpha.unique()))))
    ax.set_xticklabels([f"{v:g}" for v in sorted(st.alpha.unique())])
    ax.set_ylabel("tardiness as a multiple of\nthe same rule at $\\omega=0$",
                  fontsize=FP.AXLAB_FS)
    style(ax, "temporal weight $\\omega$",
          "(c)")
    legend_in_headroom(ax, ax, fig, "upper left", rows=2, ncol=2)

    # ------------------------------------------------------- (d) the penalty rate
    rr = pd.read_csv(os.path.join(HERE, "results_paper_rho.csv"))
    drh = margins(rr, "rho")
    piv_t = rr.pivot_table(index=["rho", "alignment", "seed"], columns="method",
                           values="tardiness")
    piv_c = rr.pivot_table(index=["rho", "alignment", "seed"], columns="method",
                           values="cost")
    ours_t, base_t = [], []
    for v in drh.level:
        st_ = piv_t.xs(v, level="rho"); sc_ = piv_c.xs(v, level="rho")
        pick = sc_[[r + "+ls" for r in RULES]].idxmin(axis=1)
        ours_t.append(st_["ours"].mean())
        base_t.append(np.mean([st_.loc[i, pick.loc[i]] for i in st_.index]))
    ax = AX[1, 1]
    xr = np.arange(len(drh))
    lines(ax, xr, drh)
    ax2 = ax.twinx()
    ax2.bar(xr - 0.19, ours_t, width=0.34, color="0.35", zorder=0,
            label="ours")
    ax2.bar(xr + 0.19, base_t, width=0.34, color="0.85", zorder=0,
            label="best rule + descent")
    ax2.set_ylabel("tardiness", fontsize=FP.AXLAB_FS)
    ax2.set_ylim(0, max(ours_t) * 2.4)
    ax2.tick_params(labelsize=FP.TICK_FS)
    # the labels name the bars on the twin, but it is the lines on the left
    # axis that the box would cover, so the room is made there
    legend_in_headroom(ax, ax2, fig, "upper right", rows=2)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xticks(xr)
    ax.set_xticklabels([f"{v:g}" for v in drh.level])
    style(ax, "penalty rate $\\rho$",
          "(d)")

    fig.tight_layout()
    fig.savefig(f"{a.out}.pdf")
    fig.savefig(f"{a.out}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
