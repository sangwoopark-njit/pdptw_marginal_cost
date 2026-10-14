"""
The induced routing subproblem: one courier, a fixed set of requests,
an open route, and soft time windows.

Scheduling.  For a fixed sequence the optimal schedule is the earliest
feasible one, b_i = max(arrival_i, e_i).  Earliness is then zero at every
stop, and tardiness is non-decreasing in b_i, so no delay can help.  The
sequencing problem therefore reduces to a shortest-path problem over
subsets, which we solve exactly by label setting with dominance.

Labels carry (cost, time) at a node for a given visited set; a label is
dominated when another at the same state has both no larger.  Both
components matter because a cheaper label may arrive later and incur more
tardiness downstream.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations


def solve_route(idx, T, e, l, S, is_pickup, mate, o_time, o_row, rho,
                exact_limit=8):
    """
    idx        request indices served by this courier
    T          (m+1, m+1) travel times among [stops..., origin]; origin is last
    e, l, S    per-stop release, latest start, service duration
    is_pickup  bool per stop
    mate       for a pickup, the index of its delivery, and vice versa
    o_time     courier ready time
    o_row      index of the origin row in T
    rho        penalty weight on earliness + tardiness

    Returns (travel, penalty, sequence).  Exact when the number of requests
    is at most exact_limit, otherwise a cheapest-insertion start improved by
    or-opt and 2-opt, with the same exact scheduling rule.
    """
    n = len(idx)
    if n == 0:
        return 0.0, 0.0, []
    if n <= exact_limit:
        return _exact(idx, T, e, l, S, is_pickup, mate, o_time, o_row, rho)
    return _heuristic(idx, T, e, l, S, is_pickup, mate, o_time, o_row, rho)


# ----------------------------------------------------------------- exact ---
def _exact(idx, T, e, l, S, is_pickup, mate, o_time, o_row, rho):
    stops = []
    for r in idx:
        stops.append(('p', r))
        stops.append(('d', r))
    m = len(stops)
    # map each stop to its row in T and its pickup partner position
    row = {}
    pos_of = {}
    for k, (kind, r) in enumerate(stops):
        row[k] = r if kind == 'p' else r + _NHALF[0]
        pos_of[(kind, r)] = k

    # precedence: delivery k requires its pickup served
    need = {}
    for k, (kind, r) in enumerate(stops):
        need[k] = pos_of[('p', r)] if kind == 'd' else None

    full = (1 << m) - 1
    # state -> dict last -> list of (cost, time)  (Pareto set)
    best = {}
    start = {}
    for k in range(m):
        if need[k] is not None:
            continue
        t = max(o_time + T[o_row, row[k]], e[row[k]])
        c = T[o_row, row[k]] + rho * max(0.0, t - l[row[k]])
        start.setdefault(1 << k, {}).setdefault(k, []).append((c, t, [k]))
    best[0] = None
    frontier = start

    final = None
    for _ in range(m - 1):
        nxt = {}
        for mask, byl in frontier.items():
            for last, labels in byl.items():
                for (c, t, seq) in labels:
                    dep = t + S[row[last]]
                    for k in range(m):
                        if mask >> k & 1:
                            continue
                        nd = need[k]
                        if nd is not None and not (mask >> nd & 1):
                            continue
                        arr = dep + T[row[last], row[k]]
                        tk = max(arr, e[row[k]])
                        ck = c + T[row[last], row[k]] + rho * max(0.0, tk - l[row[k]])
                        nm = mask | (1 << k)
                        lst = nxt.setdefault(nm, {}).setdefault(k, [])
                        if not _dominated(lst, ck, tk):
                            lst.append((ck, tk, seq + [k]))
        frontier = {mk: {kk: _prune(v) for kk, v in by.items()}
                    for mk, by in nxt.items()}
        if not frontier:
            break

    bylast = frontier.get(full, {})
    bestlab = None
    for last, labels in bylast.items():
        for lab in labels:
            if bestlab is None or lab[0] < bestlab[0]:
                bestlab = lab
    if bestlab is None:
        return _heuristic(idx, T, e, l, S, is_pickup, mate, o_time, o_row, rho)
    _, _, seq = bestlab
    return _evaluate([row[k] for k in seq], T, e, l, S, o_time, o_row, rho)


def _dominated(lst, c, t):
    for (cc, tt, _) in lst:
        if cc <= c and tt <= t:
            return True
    return False


def _prune(lst):
    lst = sorted(lst, key=lambda z: (z[0], z[1]))
    out = []
    for c, t, s in lst:
        if all(not (cc <= c and tt <= t) for cc, tt, _ in out):
            out.append((c, t, s))
    return out


# ------------------------------------------------------------- heuristic ---
def _heuristic(idx, T, e, l, S, is_pickup, mate, o_time, o_row, rho):
    seq = []
    for r in idx:                       # cheapest insertion, pickup then delivery
        bestc, bestseq = None, None
        for i in range(len(seq) + 1):
            for j in range(i + 1, len(seq) + 2):
                cand = seq[:i] + [r] + seq[i:j - 1] + [r + _NHALF[0]] + seq[j - 1:]
                tr, pen, _ = _evaluate(cand, T, e, l, S, o_time, o_row, rho)
                c = tr + rho * pen
                if bestc is None or c < bestc:
                    bestc, bestseq = c, cand
        seq = bestseq
    improved = True
    while improved:                     # or-opt on adjacent pairs
        improved = False
        base = _cost(seq, T, e, l, S, o_time, o_row, rho)
        for i in range(len(seq) - 1):
            for j in range(len(seq)):
                if abs(i - j) < 2:
                    continue
                cand = seq[:i] + seq[i + 2:]
                cand = cand[:j] + seq[i:i + 2] + cand[j:]
                if _valid(cand) and _cost(cand, T, e, l, S, o_time, o_row, rho) < base - 1e-9:
                    seq, improved = cand, True
                    break
            if improved:
                break
    return _evaluate(seq, T, e, l, S, o_time, o_row, rho)


def _valid(seq):
    seen = set()
    for s in seq:
        if s >= _NHALF[0] and (s - _NHALF[0]) not in seen:
            return False
        seen.add(s)
    return True


def _cost(seq, T, e, l, S, o_time, o_row, rho):
    tr, pen, _ = _evaluate(seq, T, e, l, S, o_time, o_row, rho)
    return tr + rho * pen


def _evaluate(rows, T, e, l, S, o_time, o_row, rho):
    """Earliest-feasible schedule along a fixed sequence of T-rows."""
    t = o_time
    prev = o_row
    travel = 0.0
    pen = 0.0
    for r in rows:
        travel += T[prev, r]
        arr = t + T[prev, r]
        t = max(arr, e[r])
        pen += max(0.0, t - l[r])          # earliness is zero by construction
        t += S[r]
        prev = r
    return travel, pen, rows


# number of requests in the current instance; set by the caller
_NHALF = [0]


def set_nhalf(n):
    _NHALF[0] = n
