"""Run a numerical check and save its measurements."""


import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import trsk, grid


def main(out, M, beta, iters):
    X = grid.torus_X_beta(M, beta)
    I, J = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    TI, TJ = (-1 - I) % M, (-1 - J) % M
    rep = (I + J * M) < (TI + TJ * M)
    P = X[:, rep].T.copy()
    sites = np.stack([np.stack([I[rep], J[rep]], 1), np.stack([TI[rep], TJ[rep]], 1)], 1)
    if iters:
        P = trsk.lloyd(P, iters)
    m = trsk.VoronoiMesh(P)
    nsides = np.bincount(np.bincount(m.cellsOnEdge.ravel(), minlength=m.N))
    def lat_dist(a, b):
        di = (a[0] - b[0] + M // 2) % M - M // 2; dj = (a[1] - b[1] + M // 2) % M - M // 2
        return (abs(di) + abs(dj) + abs(di + dj)) // 2
    reach = max(min(lat_dist(sites[c1][s1], sites[c2][s2]) for s1 in (0, 1) for s2 in (0, 1)) for c1, c2 in m.cellsOnEdge)
    res = dict(M=M, beta=beta, lloyd=iters, ncells=int(m.N), sides={int(k): int(v) for k, v in enumerate(nsides) if v},
               defects=int(m.N - nsides[6]) if len(nsides) > 6 else int(m.N), neighbour_reach=int(reach))
    orig = trsk.mesh_points
    trsk.mesh_points = lambda spec: P
    try:
        for case in ("tc2_a0.7", "tc5"):
            r = trsk.run(("torus", M), case, dict(solver="trsk", integrator="rk4", courant=0.5))
            res[case] = r.get("l2"); res[case + "_rms_m"] = r.get("rms_err_m"); res[case + "_stable"] = r["stable"]
    finally:
        trsk.mesh_points = orig
    print(json.dumps(res), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
