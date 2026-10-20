"""Run a numerical check and save its measurements."""


import sys, os, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.interpolate import RBFInterpolator
from polarswe import grid, tfd, testcases as tc, reference as ref, configs, experiments as ex


def main(M, nlat, reffn, out):
    X = grid.torus_X(M); case = tc.TC6()
    F, S0, h, cmax, dt = ex.build(X, case, configs.MAIN)
    D = np.ascontiguousarray(case.height(X) - S0.hs); v = np.ascontiguousarray(case.velocity(X))
    n = int(np.ceil(tc.DAY / dt)); dt = tc.DAY / n
    for k in range(n):
        D, v = F.step(D, v, dt)
    _, _, g, recs = ref.read(reffn)
    I = ref.SphereInterp(g, [r for r in recs if abs(r[0] - 24.0) < 1e-6][0][3])
    err2 = D + S0.hs - I(X)
    err = err2.ravel()

    Ii, Jj = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    keep = (Ii + Jj * M) < ((-1 - Ii) % M + ((-1 - Jj) % M) * M)
    pts = X[:, keep].T
    rbf = RBFInterpolator(pts, err2[keep], neighbors=30, kernel="thin_plate_spline", degree=1)
    x, wq = np.polynomial.legendre.leggauss(nlat)
    lat = np.arcsin(x)[::-1]; lon = 2 * np.pi * np.arange(2 * nlat) / (2 * nlat)
    L, P = np.meshgrid(lon, lat)
    Xg = np.stack([np.cos(P) * np.cos(L), np.cos(P) * np.sin(L), np.sin(P)])
    dh = rbf(Xg.reshape(3, -1).T).reshape(P.shape)

    C = tc.TC6(); v3 = C.velocity(Xg); e, nn = tc.east_north(Xg)
    w_, K, R = C.w, C.K, C.R
    zeta = 2 * w_ * np.sin(P) - K * np.sin(P) * np.cos(P) ** R * (R * R + 3 * R + 2) * np.cos(R * L) + 2 * tc.OMEGA * np.sin(P)
    zero = np.zeros_like(P)
    with open(out, "wb") as f:
        for Fld in ((v3 * e).sum(0), (v3 * nn).sum(0), C.height(Xg) + dh, zero, zeta):
            np.ascontiguousarray(Fld, dtype=np.float64).tofile(f)
    W = np.broadcast_to(wq[::-1][:, None], P.shape)
    print(f"seed: torus M={M} day-1 height error, RMS {math.sqrt((W * dh**2).sum() / W.sum()):.3f} m on the Gaussian grid "
          f"(on the torus {math.sqrt((S0.weight * err.reshape(M, M)**2).sum() / S0.weight.sum()):.3f} m)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
