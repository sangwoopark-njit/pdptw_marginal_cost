"""OR-Tools baseline, run as a separate process because its bundled HiGHS
conflicts with highspy in the same interpreter.  Usage: python3 ortools_worker.py in.npz"""
import sys, json, time, numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

def main(path):
    d = np.load(path)
    xp, xd, ehp, lhp, ehd, lhd, sp, sd = (d[k] for k in ["xp","xd","ehp","lhp","ehd","lhd","sp","sd"])
    origins, theta, cap, Tall = d["origins"], d["theta"], d["cap"], d["Tall"]
    rho, phi, tl, scale = float(d["rho"]), float(d["phi"]), int(d["time_limit"]), 100
    n = len(xp); K = len(theta); N = 2 * n; end = N + K
    e = np.concatenate([ehp, ehd]); l = np.concatenate([lhp, lhd]); S = np.concatenate([sp, sd])
    def T(i, j): return 0 if (i == end or j == end) else int(round(Tall[i, j] * scale))
    mgr = pywrapcp.RoutingIndexManager(N + K + 1, K, list(range(N, N + K)), [end] * K)
    rt = pywrapcp.RoutingModel(mgr)
    dcb = rt.RegisterTransitCallback(lambda a, b: T(mgr.IndexToNode(a), mgr.IndexToNode(b)))
    rt.SetArcCostEvaluatorOfAllVehicles(dcb)
    def tcb_f(a, b):
        i, j = mgr.IndexToNode(a), mgr.IndexToNode(b)
        return T(i, j) + (int(round(S[i] * scale)) if i < N else 0)
    tcb = rt.RegisterTransitCallback(tcb_f)
    horizon = int(round((max(theta.max(), l.max()) + (S + Tall[:N, :N].max(0)).sum()) * scale))
    rt.AddDimension(tcb, horizon, horizon, False, "Time")
    tdim = rt.GetDimensionOrDie("Time")
    for i in range(N):
        idx = mgr.NodeToIndex(i)
        tdim.CumulVar(idx).SetMin(int(round(e[i] * scale)))
        tdim.SetCumulVarSoftUpperBound(idx, int(round(l[i] * scale)), int(round(rho * scale)))
    for k in range(K):
        tdim.CumulVar(rt.Start(k)).SetMin(int(round(theta[k] * scale)))
        rt.SetFixedCostOfVehicle(int(round(phi * scale)), k)
    q = [1] * n + [-1] * n + [0] * (K + 1)
    qcb = rt.RegisterUnaryTransitCallback(lambda a: q[mgr.IndexToNode(a)])
    rt.AddDimensionWithVehicleCapacity(qcb, 0, [int(c) for c in cap], True, "Load")
    for r in range(n):
        p, dd = mgr.NodeToIndex(r), mgr.NodeToIndex(r + n)
        rt.AddPickupAndDelivery(p, dd)
        rt.solver().Add(rt.VehicleVar(p) == rt.VehicleVar(dd))
        rt.solver().Add(tdim.CumulVar(p) <= tdim.CumulVar(dd))
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.FromSeconds(tl)
    t0 = time.time(); sol = rt.SolveWithParameters(prm); secs = time.time() - t0
    if sol is None:
        print(json.dumps({"ok": False})); return
    routes = []
    for k in range(K):
        idx = sol.Value(rt.NextVar(rt.Start(k))); seq = []
        while not rt.IsEnd(idx):
            seq.append(int(mgr.IndexToNode(idx))); idx = sol.Value(rt.NextVar(idx))
        routes.append(seq)
    print(json.dumps({"ok": True, "routes": routes, "secs": secs}))

if __name__ == "__main__":
    main(sys.argv[1])
