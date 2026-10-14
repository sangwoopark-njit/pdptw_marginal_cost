"""
Visual inspection of a small instance and the solutions two methods find.

Panel 1  the instance: pickups and deliveries on the plane, each labelled with
         its service window and its demand sign, pickups joined to their
         deliveries by a faint line, couriers shown as squares.
Panel 2  a timeline: each request as a horizontal bar spanning its pickup and
         delivery windows, so temporal overlap is visible directly.
Panel 3+ one panel per method, drawing the routes actually taken.

The point is to see *why* one method beats another rather than inferring it
from a cost number.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import routing

PCOL, DCOL = "#1f77b4", "#d62728"          # pickup blue, delivery red
VCOL = ["#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]


def draw_instance(ax, inst, ehp, lhp, ehd, lhd, annotate=True):
    n = inst.n_req
    for r in range(n):
        ax.plot([inst.xp[r, 0], inst.xd[r, 0]], [inst.xp[r, 1], inst.xd[r, 1]],
                color="0.85", lw=0.8, zorder=1)
    ax.scatter(inst.xp[:, 0], inst.xp[:, 1], c=PCOL, s=70, zorder=3,
               edgecolors="white", linewidths=0.8)
    ax.scatter(inst.xd[:, 0], inst.xd[:, 1], c=DCOL, s=70, marker="s", zorder=3,
               edgecolors="white", linewidths=0.8)
    ax.scatter(inst.origins[:, 0], inst.origins[:, 1], c="black", s=130,
               marker="*", zorder=4)
    if annotate:
        for r in range(n):
            ax.annotate(f"P{r}\n[{ehp[r]:.0f},{lhp[r]:.0f}]\n+1",
                        inst.xp[r], textcoords="offset points", xytext=(6, 4),
                        fontsize=6, color=PCOL)
            ax.annotate(f"D{r}\n[{ehd[r]:.0f},{lhd[r]:.0f}]\n-1",
                        inst.xd[r], textcoords="offset points", xytext=(6, 4),
                        fontsize=6, color=DCOL)
        for k in range(inst.n_veh):
            ax.annotate(f"V{k}\n$\\theta$={inst.theta[k]:.0f}", inst.origins[k],
                        textcoords="offset points", xytext=(6, -14), fontsize=6)
    ax.set_title("instance", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)


def draw_timeline(ax, inst, ehp, lhp, ehd, lhd):
    n = inst.n_req
    order = np.argsort(ehp)
    for i, r in enumerate(order):
        ax.plot([ehp[r], lhp[r]], [i, i], color=PCOL, lw=5, solid_capstyle="butt")
        ax.plot([ehd[r], lhd[r]], [i, i], color=DCOL, lw=5, solid_capstyle="butt")
        ax.plot([lhp[r], ehd[r]], [i, i], color="0.8", lw=1)
    for k in range(inst.n_veh):
        ax.axvline(inst.theta[k], color="0.4", ls=":", lw=1)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"r{r}" for r in order], fontsize=6)
    ax.set_xlabel("time", fontsize=8)
    ax.set_title("windows (blue = pickup, red = delivery)", fontsize=9)
    ax.grid(alpha=0.2, axis="x")


def draw_routes(ax, inst, ehp, lhp, ehd, lhd, Tall, assign, rho, title):
    """`assign` maps vehicle -> list of request ids."""
    n = inst.n_req
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    routing.set_nhalf(n)
    ax.plot([inst.xp[:, 0], inst.xd[:, 0]], [inst.xp[:, 1], inst.xd[:, 1]],
            color="0.9", lw=0.6, zorder=1)
    ax.scatter(inst.xp[:, 0], inst.xp[:, 1], c=PCOL, s=45, zorder=3)
    ax.scatter(inst.xd[:, 0], inst.xd[:, 1], c=DCOL, s=45, marker="s", zorder=3)
    ax.scatter(inst.origins[:, 0], inst.origins[:, 1], c="black", s=110,
               marker="*", zorder=4)
    tot_tr = tot_pen = 0.0
    pts = np.vstack([inst.xp, inst.xd])
    for k, req in assign.items():
        if not len(req):
            continue
        tr, pen, seq = routing.solve_route(list(req), Tall, e, l, S, None, None,
                                           inst.theta[k], 2 * n + k, rho)
        tot_tr += tr; tot_pen += pen
        col = VCOL[k % len(VCOL)]
        path = np.vstack([inst.origins[k], pts[seq]])
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.6, alpha=0.9, zorder=2,
                label=f"V{k}: {len(req)} req")
        for i, row in enumerate(seq):
            ax.annotate(str(i + 1), pts[row], textcoords="offset points",
                        xytext=(-8, -9), fontsize=6, color=col)
    ax.set_title(f"{title}\ntravel {tot_tr:.0f} + penalty {tot_pen:.0f} "
                 f"= {tot_tr + rho * tot_pen:.0f}", fontsize=9)
    ax.legend(fontsize=6, loc="best"); ax.set_aspect("equal"); ax.grid(alpha=0.2)


def compare(inst, ehp, lhp, ehd, lhd, Tall, solutions, rho=1.0, path="instance.png"):
    """`solutions` is {name: {vehicle: [request ids]}}."""
    ncols = 2 + len(solutions)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.4))
    draw_instance(axes[0], inst, ehp, lhp, ehd, lhd)
    draw_timeline(axes[1], inst, ehp, lhp, ehd, lhd)
    for ax, (name, asg) in zip(axes[2:], solutions.items()):
        draw_routes(ax, inst, ehp, lhp, ehd, lhd, Tall, asg, rho, name)
    handles = [Line2D([], [], marker="o", ls="", color=PCOL, label="pickup (+1)"),
               Line2D([], [], marker="s", ls="", color=DCOL, label="delivery (-1)"),
               Line2D([], [], marker="*", ls="", color="black", label="courier origin")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def metrics(inst, ehp, lhp, ehd, lhd, Tall, assign, rho=1.0):
    """Report the four quantities separately rather than as one blended cost.

      tardiness   sum_i (t_i^+ + t_i^-); earliness is zero in an optimal
                  schedule, so this is realised lateness  -- SERVICE QUALITY
      travel      sum T_ij x_ij over dispatched couriers   -- OPERATING COST
      peak_util   mean over dispatched couriers of (max simultaneous load)/Q_k
      late_reqs   fraction of requests served after their window closes
    """
    n = inst.n_req
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd])
    S = np.concatenate([inst.sp, inst.sd])
    routing.set_nhalf(n)
    travel = tard = 0.0; used = 0; utils = []; late = 0
    for k, req in assign.items():
        req = list(req)
        if not req:
            continue
        used += 1
        tr, pen, seq = routing.solve_route(req, Tall, e, l, S, None, None,
                                           inst.theta[k], 2 * n + k, rho)
        travel += tr; tard += pen
        load = 0; peak = 0
        tcur = inst.theta[k]; prev = 2 * n + k
        for row in seq:
            tcur = max(tcur + Tall[prev, row], e[row])
            if tcur > l[row] + 1e-9:
                late += 1
            tcur += S[row]; prev = row
            load += 1 if row < n else -1
            peak = max(peak, load)
        utils.append(peak / max(inst.cap[k], 1e-9))
    return dict(tardiness=tard, travel=travel, couriers=used,
                peak_util=float(np.mean(utils)) if utils else 0.0,
                late_frac=late / max(2 * n, 1))


def assignment_from_cluster(res, n_veh):
    """Extract {vehicle: [requests]} from a cluster.run result."""
    a, s = res["a"], res["s"]
    out = {}
    for k in range(n_veh):
        cl = [c for c in range(len(s)) if s[c] == k]
        req = np.unique(np.concatenate([np.where(a == c)[0] for c in cl])) if cl else np.array([], int)
        out[k] = req.astype(int).tolist()
    return out
