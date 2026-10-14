"""
The paper's mechanism figure, on the production pipeline and the current design.

    python3 figure_paper.py                 # greyscale, for print
    python3 figure_paper.py --colour

Two instances at |R| = 40, |K| = 12, Q_k = 5, phi = 100, built by design.py with
f = 5/6 and nu = 4, so 8 to 12 couriers may be dispatched and the charge decides.
One instance is spatially/temporally ALIGNED (requests close together are also due
at similar times), the other MISALIGNED.

Eight panels per instance:

  top row, maps    the instance as posed; the published rule; the same rule with
                   our descent; the proposed method
  bottom row, time each dispatched courier's assigned requests on a time axis, so
                   that temporal contention within one courier is visible, with a
                   mark where the exact route actually serves each stop

The instance shown is the one whose result is the MEDIAN over the seeds, not the
best; every seed's gap is printed when the script runs.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import budget as B, design as D
import cluster2 as K2, cluster_sparse as CS, visualise as V
import clustering_baselines as CB
import routing

GREY = ["0.0", "0.40", "0.62", "0.25", "0.52", "0.72", "0.15", "0.47",
        "0.68", "0.32", "0.57", "0.10"]
COLOUR = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
          "#17becf", "#e377c2", "#bcbd22", "#7f7f7f", "#393b79", "#637939"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
LINE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)),
        (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1)), "-", "--", "-.", ":"]
PROD = dict(assign='mip', mip_limit=5.0, cand=8, T_out=4)

#: what every figure's type should measure ON THE PAGE, in points, against a
#: 10pt body.  Elsevier's floor for artwork is 7pt at final size.
#:
#: The three are set independently rather than derived from the title size.
#: The tick size fixes how wide a courier label is, and that in turn fixes the
#: gap a figure has to leave between columns -- so deriving ticks from titles
#: means enlarging a title silently shrinks the panels.
ON_PAGE_TITLE = 9.0
ON_PAGE_AXLAB = 7.3
ON_PAGE_TICK = 6.8
TEXTWIDTH_IN = 6.5          # the elsarticle 3p text block


def canvas_pt(on_page, width_in):
    """Point size to draw at so it arrives at `on_page` once scaled to the page.

    A figure included at \\textwidth is scaled by TEXTWIDTH_IN / width_in, and
    that scaling hits the type as well as the artwork.  Two figures drawn at the
    same nominal size but on different canvases therefore print at different
    sizes, which is why the panel titles used to disagree from figure to figure.
    """
    return on_page * width_in / TEXTWIDTH_IN


# point sizes BEFORE the figure is scaled into the page; each script sets these
# from canvas_pt() once it knows how wide its canvas is
TITLE_FS = 10.5
TICK_FS = 8
AXLAB_FS = 9


def set_sizes(width_in, title=None):
    """Set the shared sizes for a canvas of this width, and report them."""
    global TITLE_FS, TICK_FS, AXLAB_FS
    TITLE_FS = canvas_pt(title or ON_PAGE_TITLE, width_in)
    AXLAB_FS = canvas_pt(ON_PAGE_AXLAB, width_in)
    TICK_FS = canvas_pt(ON_PAGE_TICK, width_in)
    return TITLE_FS


#: XCharter-Roman.otf, shipped with the TeX xcharter package; the same face the
#: manuscript sets.  Point the XCHARTER_OTF environment variable at it; without
#: it the figures fall back to DejaVu Serif.
XCHARTER = os.environ.get("XCHARTER_OTF", "")


def use_publisher_fonts():
    """Type the figures in the same face as the manuscript.

    The published TR-E article is set in Charis SIL with STIX mathematics, both
    Charter-derived.  Charis SIL has no Type1 files here so pdflatex cannot use
    it; XCharter is the same Charter lineage, does work with pdflatex, and is
    what the manuscript loads -- so the figures use it too and panel titles are
    literally the running text's font.  Mathematics stays STIX, as in the
    published article.

    This also avoids a substitution: matplotlib's default DejaVu Sans is not on
    Elsevier's artwork list (Arial, Courier, Times New Roman, Symbol), and they
    warn that other fonts are replaced in production, which can drop symbols or
    overlap type.  Their floor is 7pt at final size; each script prints what its
    type will measure once the figure is scaled into the text block.
    """
    fam = "DejaVu Serif"
    if os.path.exists(XCHARTER):
        matplotlib.font_manager.fontManager.addfont(XCHARTER)
        fam = "XCharter"
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [fam, "Charis SIL", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })
    return fam


use_publisher_fonts()


def pal(colour):
    return COLOUR if colour else GREY


def route_of(inst, ehp, lhp, ehd, lhd, Tall, req, k):
    n = inst.n_req
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    routing.set_nhalf(n)
    if not len(req):
        return [], {}
    _tr, _pen, seq = routing.solve_route(list(req), Tall, e, l, S, None, None,
                                         inst.theta[k], 2 * n + k, 1.0)
    t, prev, when = inst.theta[k], 2 * n + k, {}
    for row in seq:
        t = max(t + Tall[prev, row], e[row])
        when[row] = (t, max(0.0, t - l[row]))
        t += S[row]
        prev = row
    return seq, when


def draw_instance(ax, inst, title, lims=None):
    n = inst.n_req
    for r in range(n):
        ax.plot([inst.xp[r, 0], inst.xd[r, 0]], [inst.xp[r, 1], inst.xd[r, 1]],
                color="0.85", lw=0.6, zorder=1)
    ax.scatter(inst.xp[:, 0], inst.xp[:, 1], facecolor="0.15", s=18, marker="o",
               zorder=3, edgecolors="white", linewidths=0.4)
    ax.scatter(inst.xd[:, 0], inst.xd[:, 1], facecolor="white", s=18, marker="o",
               zorder=3, edgecolors="0.15", linewidths=0.7)
    ax.scatter(inst.origins[:, 0], inst.origins[:, 1], color="black", s=60,
               marker="*", zorder=4)
    _finish(ax, title, lims)


def draw_routes(ax, inst, ehp, lhp, ehd, lhd, Tall, assign, title,
                colour, lims=None):
    P, n = pal(colour), inst.n_req
    for k, req in sorted(assign.items()):
        if not len(req):
            continue
        c = P[k % len(P)]
        seq, _ = route_of(inst, ehp, lhp, ehd, lhd, Tall, req, k)
        xs = [inst.origins[k][0]] + [(inst.xp[i, 0] if i < n else inst.xd[i - n, 0])
                                     for i in seq]
        ys = [inst.origins[k][1]] + [(inst.xp[i, 1] if i < n else inst.xd[i - n, 1])
                                     for i in seq]
        ax.plot(xs, ys, color=c, lw=0.9, ls=LINE[k % len(LINE)], zorder=2)
        ax.scatter(xs[1:], ys[1:], facecolor=c, s=11, zorder=3,
                   marker=MARK[k % len(MARK)], edgecolors="white", linewidths=0.3)
        ax.scatter(*inst.origins[k], color=c, s=60, marker="*", zorder=4)
    _finish(ax, title, lims)


def square_limits(inst, pad=0.06):
    """A single square window covering every point of an instance.

    Without this each map panel sized itself to its own data under
    `set_aspect("equal")`, so a tall point cloud produced a narrow axes and the
    rows of the figure came out different widths.  Fixing one square window per
    instance makes every panel the same shape and fills the space.
    """
    xs = np.concatenate([inst.xp[:, 0], inst.xd[:, 0], inst.origins[:, 0]])
    ys = np.concatenate([inst.xp[:, 1], inst.xd[:, 1], inst.origins[:, 1]])
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def _finish(ax, title, lims=None):
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_aspect("equal")
    if lims is not None:
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def draw_courier_time(ax, inst, ehp, lhp, ehd, lhd, Tall, assign, title,
                      colour, xlabel=True, max_labels=None):
    P, n = pal(colour), inst.n_req
    row, yt, yl = 0, [], []
    for k in sorted(assign):
        req = list(assign[k])
        if not req:
            continue
        c = P[k % len(P)]
        _seq, when = route_of(inst, ehp, lhp, ehd, lhd, Tall, req, k)
        for j, r in enumerate(sorted(req, key=lambda x: ehp[x])):
            y = row + 0.14 * (j % 3)
            ax.plot([ehp[r], lhd[r]], [y, y], color=c, lw=1.7,
                    solid_capstyle="butt", alpha=0.8)
            for rid in (r, r + n):
                if rid in when:
                    t, late = when[rid]
                    ax.plot([t], [y], marker="|", color="black", ms=3, mew=0.7)
                    if late > 1e-6:
                        ax.plot([t], [y], marker="x", color="black", ms=3.5, mew=1.0)
        yt.append(row + 0.14); yl.append(f"k{k}")
        row += 1
    # A dozen couriers in a one-inch panel leaves less room per label than the
    # label is tall, so they overprinted each other.  Thinning keeps the one
    # thing the axis is read for -- how many bands are occupied, which is the
    # band count, not the label count -- while the labels stay legible.
    if max_labels and len(yt) > max_labels:
        step = -(-len(yt) // max_labels)
        keep = set(range(0, len(yt), step))
        yl = [lab if i in keep else "" for i, lab in enumerate(yl)]
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=TICK_FS)
    if xlabel:
        ax.set_xlabel("time", fontsize=AXLAB_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.grid(axis="x", alpha=0.25, lw=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def score(inst, ehp, lhp, ehd, lhd, Tall, assign, phi):
    m = CB._evaluate(inst, ehp, lhp, ehd, lhd, Tall, assign, 1.0, 0.0)
    m["cost"] = m["travel"] + m["tardiness"] + phi * m["couriers"]
    return m


def solve_all(inst, ehp, lhp, ehd, lhd, Tall, Top, K, phi, seed, rule):
    m0 = CB.METHODS[rule](inst, ehp, lhp, ehd, lhd, Tall, 1.0, 0.0, seed=seed)
    lab0 = np.asarray(m0["labels"], int)
    raw = {k: np.where(lab0 == k)[0].tolist() for k in range(K)}
    lab = K2.polish_labels(lab0, inst, ehp, lhp, ehd, lhd, Tall, 1.0, phi=phi)
    pol = {k: np.where(lab == k)[0].tolist() for k in range(K)}
    r = CS.run(inst, ehp, lhp, ehd, lhd, Tall, Top, phi=phi, seed=seed, **PROD)
    return raw, pol, V.assignment_from_cluster(r, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_veh", type=int, default=12)
    ap.add_argument("--nu", type=int, default=4)
    ap.add_argument("--f", type=float, default=5 / 6)
    ap.add_argument("--phi", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rule", default="cap_kmeans")
    ap.add_argument("--colour", action="store_true")
    ap.add_argument("--out", default="../figures/fig_mechanism")
    a = ap.parse_args()

    d = D.design(a.n_veh, a.f, a.nu)
    print(d.describe())

    # Every axes is placed in inches.  This used to be a plain subplots grid
    # finished with tight_layout, and that wasted a fifth of the width: the map
    # panels are set_aspect("equal"), so each square was centred inside a wider
    # grid cell and the surplus came out as canvas margin -- 1.04in blank on
    # the left and 0.83in on the right of an 8.32in canvas, with the panels
    # 1.44in rather than the 1.72in the arithmetic above them claimed.  The
    # gaps that were left could not hold the courier labels either, so each
    # timeline panel's labels sat against the frame of the panel before it.
    #
    # The canvas is now exactly the text width, so canvas inches are page
    # inches and type is set at its on-page size directly.
    MECH_W = TEXTWIDTH_IN
    L, R, WGAP = 0.10, 0.14, 0.30   # WGAP holds one "k11" plus its tick
    PANEL = (MECH_W - L - R - 3 * WGAP) / 4
    TIMEH = 0.95
    # A title is its lines plus axes.titlepad (6pt): three lines at TITLE_FS is
    # 3*1.2*8 + 6 = 34.8pt, one line 15.6pt.  Tick labels below an axes are the
    # tick (3.5pt) plus its pad (3.5pt) plus the label (1.2*TICK_FS), about
    # 15pt, and the "time" label adds a line of AXLAB_FS under that.  Sized
    # short, the second block's titles printed over the first block's tick
    # labels, so each allowance carries a little slack.
    MAPTITLE, TIMETITLE = 0.56, 0.26    # three lines and one, at TITLE_FS
    XTICK, XTIME = 0.24, 0.40           # tick labels, and the "time" label
    PAD = 0.04
    MECH_H = (PAD + MAPTITLE + PANEL + TIMETITLE + TIMEH + XTICK
              + MAPTITLE + PANEL + TIMETITLE + TIMEH + XTIME + PAD)
    set_sizes(MECH_W)
    # how many courier labels a timeline panel can carry without them touching
    NLAB = max(2, int(TIMEH * 72 / (TICK_FS * 1.15)))
    print(f'canvas {MECH_W:.2f} x {MECH_H:.2f} in; panels '
          f'{PANEL:.2f} in and titles {TITLE_FS:.1f} pt on the page; '
          f'up to {NLAB} courier labels per timeline panel')

    # one figure, four rows: maps and time axis for the aligned instance, then
    # the same for the misaligned one, so the two regimes can be read together.
    # Row heights and the space above each row differ -- a map row carries a
    # three-line title, a timeline row one line -- which a single hspace cannot
    # express, so the rows are stacked by hand from the top down.
    fig = plt.figure(figsize=(MECH_W, MECH_H))
    AX = np.empty((4, 4), dtype=object)
    top = MECH_H - PAD
    for r, (lead, height) in enumerate([(MAPTITLE, PANEL), (TIMETITLE, TIMEH),
                                        (MAPTITLE, PANEL), (TIMETITLE, TIMEH)]):
        top -= lead
        for c in range(4):
            left = L + c * (PANEL + WGAP)
            AX[r, c] = fig.add_axes([left / MECH_W, (top - height) / MECH_H,
                                     PANEL / MECH_W, height / MECH_H])
        top -= height + (XTICK if r == 1 else 0)

    for block, (align, tag) in enumerate([(1.0, "aligned"), (0.0, "misaligned")]):
        gaps = []
        for s in range(a.seeds):
            inst, _ = D.make_instance(d, s, align)
            ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
            _raw, pol, ours = solve_all(inst, ehp, lhp, ehd, lhd, Tall, Top,
                                        d.n_veh, a.phi, s, a.rule)
            cb = score(inst, ehp, lhp, ehd, lhd, Tall, pol, a.phi)["cost"]
            co = score(inst, ehp, lhp, ehd, lhd, Tall, ours, a.phi)["cost"]
            gaps.append((100 * (co - cb) / cb, s))
        gaps.sort()
        gap, seed = gaps[len(gaps) // 2]
        print(f"{tag}: gaps {[(s, round(g,1)) for g,s in gaps]} -> median seed "
              f"{seed} ({gap:+.1f}%)")

        inst, mis = D.make_instance(d, seed, align)
        ehp, lhp, ehd, lhd, Tall, Top = B.prep(inst)
        raw, pol, ours = solve_all(inst, ehp, lhp, ehd, lhd, Tall, Top, d.n_veh,
                                   a.phi, seed, a.rule)
        mr = score(inst, ehp, lhp, ehd, lhd, Tall, raw, a.phi)
        mp = score(inst, ehp, lhp, ehd, lhd, Tall, pol, a.phi)
        mo = score(inst, ehp, lhp, ehd, lhd, Tall, ours, a.phi)

        def cap(lbl, m):
            # bare terms: spelling out "travel"/"tard" made the line wider than
            # a panel and it ran into the next one.  Named once in the caption.
            return (f"{lbl}\ncost {m['cost']:.0f}\n"
                    f"{m['travel']:.0f} $+$ {m['tardiness']:.0f} "
                    f"$+$ {a.phi:g}$\\times${m['couriers']}")

        lims = square_limits(inst)
        rm, rt = 2 * block, 2 * block + 1          # map row, time row
        letters = "abcdefgh" if block == 0 else "ijklmnop"
        ax = AX
        # At 8pt on the page a 1.7in panel holds roughly twenty characters, so
        # the wording lives in the caption and the titles carry only what
        # distinguishes one panel from the next.
        draw_instance(ax[rm, 0], inst,
                      f"({letters[0]}) {tag.upper()}\n"
                      f"misalign. {mis:.2f}", lims)
        draw_routes(ax[rm, 1], inst, ehp, lhp, ehd, lhd, Tall, raw,
                    cap(f"({letters[1]}) rule", mr), a.colour, lims)
        draw_routes(ax[rm, 2], inst, ehp, lhp, ehd, lhd, Tall, pol,
                    cap(f"({letters[2]}) rule $+$ desc.", mp), a.colour,
                    lims)
        draw_routes(ax[rm, 3], inst, ehp, lhp, ehd, lhd, Tall, ours,
                    cap(f"({letters[3]}) proposed", mo), a.colour, lims)

        axd = ax[rt, 0]
        for i, r in enumerate(np.argsort(ehp)):
            axd.plot([ehp[r], lhd[r]], [i, i], color="0.45", lw=1.3,
                     solid_capstyle="butt")
        axd.set_title(f"({letters[4]}) all windows", fontsize=TITLE_FS)
        if block == 1:
            axd.set_xlabel("time", fontsize=AXLAB_FS)
        axd.set_yticks([])
        axd.tick_params(axis="x", labelsize=TICK_FS)
        axd.grid(axis="x", alpha=0.25, lw=0.35)
        for sp in axd.spines.values():
            sp.set_linewidth(0.5)

        draw_courier_time(ax[rt, 1], inst, ehp, lhp, ehd, lhd, Tall, raw,
                          f"({letters[5]}) rule", a.colour,
                          xlabel=(block == 1), max_labels=NLAB)
        draw_courier_time(ax[rt, 2], inst, ehp, lhp, ehd, lhd, Tall, pol,
                          f"({letters[6]}) rule $+$ desc.", a.colour,
                          xlabel=(block == 1), max_labels=NLAB)
        draw_courier_time(ax[rt, 3], inst, ehp, lhp, ehd, lhd, Tall, ours,
                          f"({letters[7]}) proposed", a.colour,
                          xlabel=(block == 1), max_labels=NLAB)

    # no tight_layout: it is what centred each square panel in a wider cell
    # and turned the surplus into canvas margin.  The axes are placed above.
    fig.savefig(f"{a.out}.pdf")
    fig.savefig(f"{a.out}.png", dpi=150)
    plt.close(fig)
    print(f"   wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
