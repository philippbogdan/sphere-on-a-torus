"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, tfd, testcases as tc

H = 3000.0


class Rest(tc.Case):
    def coriolis(self, X):
        return tc.coriolis(X)


def growth(M, form, hv_order=4, lap_kind="metric", kappa=0.0):
    X = grid.torus_X(M); n = M * M; N = 3 * n
    t1 = np.cross(X, np.array([0, 0, 1.0])[:, None, None] * np.ones_like(X), axis=0)
    bad = np.linalg.norm(t1, axis=0) < 1e-6
    t1[:, bad] = np.cross(X[:, bad], np.array([1.0, 0, 0])[:, None], axis=0)
    t1 /= np.linalg.norm(t1, axis=0); t2 = np.cross(X, t1, axis=0)
    S = tfd.TFD(X, Rest(), order=4, form=form, hv_order=hv_order, lap_kind=lap_kind)
    h = math.sqrt(S.weight.mean()); c = math.sqrt(tc.G * H)
    nu = kappa * h ** hv_order / (0.5 * h / c)
    A = np.zeros((N, N)); eps = 1e-3
    D0 = H * np.ones((M, M)); v0 = np.zeros_like(X)
    for j in range(N):
        D = D0.copy(); v = v0.copy(); k, p = divmod(j, n); i0, j0 = divmod(p, M)
        if k == 0: D[i0, j0] += eps
        elif k == 1: v[:, i0, j0] += eps * t1[:, i0, j0]
        else: v[:, i0, j0] += eps * t2[:, i0, j0]
        dD, dv = S.rhs(D, v)
        if nu:
            dD = dD - nu * S.hyper(D - D0)
            dv = dv - nu * np.stack([S.hyper(v[q]) for q in range(3)]); dv = dv - (dv * X).sum(0) * X
        A[:n, j] = dD.ravel() / eps; A[n:2*n, j] = (dv * t1).sum(0).ravel() / eps; A[2*n:, j] = (dv * t2).sum(0).ravel() / eps
    ev = np.linalg.eigvals(A)
    return dict(M=M, form=form, filter=f"del{hv_order}-{lap_kind}" if kappa else "none", kappa=kappa,
                max_re=float(ev.real.max()), max_re_times_h_over_c=float(ev.real.max() * h / c),
                n_growing=int((ev.real > 1e-9).sum()))


if __name__ == "__main__":
    out = []
    for M in (16, 24, 32):
        out.append(growth(M, "flux"))
        out.append(growth(M, "split"))
        for hvo, lk in ((4, "metric"), (6, "flat")):
            for kappa in (0.0005, 0.002):
                out.append(growth(M, "split", hvo, lk, kappa))
        print(out[-6:], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
