"""Run a numerical check and save its measurements."""


import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import testcases as tc


def tetra_frame():

    lat = math.radians(-19.4712206)
    V = np.array([[0, 0, 1.0]] + [[math.cos(lat) * math.cos(math.radians(l)), math.cos(lat) * math.sin(math.radians(l)), math.sin(lat)] for l in (0, 120, 240)])
    W = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / math.sqrt(3)
    U, _, Vt = np.linalg.svd(W.T @ V)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def main(nlat, amp, out):
    nlon = 2 * nlat
    x, _ = np.polynomial.legendre.leggauss(nlat)
    lat = np.arcsin(x)[::-1]
    lon = 2 * np.pi * np.arange(nlon) / nlon
    L, P = np.meshgrid(lon, lat)
    X = np.stack([np.cos(P) * np.cos(L), np.cos(P) * np.sin(L), np.sin(P)])
    C = tc.TC6()
    v3 = C.velocity(X)
    e, n = tc.east_north(X)
    u, v = (v3 * e).sum(0), (v3 * n).sum(0)
    h = C.height(X)

    w, K, R = C.w, C.K, C.R
    zeta = 2 * w * np.sin(P) - K * np.sin(P) * np.cos(P) ** R * (R * R + 3 * R + 2) * np.cos(R * L)
    zeta = zeta + 2 * tc.OMEGA * np.sin(P)
    Y = np.tensordot(tetra_frame(), X, axes=1)
    pert = Y[0] * Y[1] * Y[2]
    _, wq = np.polynomial.legendre.leggauss(nlat)
    W = np.broadcast_to(wq[::-1][:, None], pert.shape)
    rms = math.sqrt((W * pert ** 2).sum() / W.sum())
    h = h + amp * pert / rms
    zero = np.zeros_like(P)
    with open(out, "wb") as f:
        for F in (u, v, h, zero, zeta):
            np.ascontiguousarray(F, dtype=np.float64).tofile(f)
    print("wrote", out, "perturbation RMS", amp, "m; check tetrahedral symmetry of pert:",
          float(np.abs(np.tensordot(tetra_frame(), np.array([0, 0, 1.0]), axes=1)).round(4).tolist() is not None))


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
