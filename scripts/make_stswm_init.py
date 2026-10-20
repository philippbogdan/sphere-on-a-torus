"""Run a numerical check and save its measurements."""


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import testcases as tc


def gauss_lat(nlat):
    x, _ = np.polynomial.legendre.leggauss(nlat)
    return np.arcsin(x)[::-1]


def main(case, nlat, out):
    nlon = 2 * nlat
    lat = gauss_lat(nlat)
    lon = 2 * np.pi * np.arange(nlon) / nlon
    L, P = np.meshgrid(lon, lat)
    X = np.stack([np.cos(P) * np.cos(L), np.cos(P) * np.sin(L), np.sin(P)])
    C = tc.Galewsky(perturb=(case == "galewsky"))
    u = C.u(P)
    h = C.height(X)

    ph0, ph1 = C.phi0, C.phi1
    inside = (P > ph0) & (P < ph1)
    du = np.zeros_like(P)
    q = (P[inside] - ph0) * (P[inside] - ph1)
    du[inside] = -u[inside] * (2 * P[inside] - ph0 - ph1) / q ** 2
    zeta = -du / tc.A + u * np.tan(P) / tc.A

    zeta = zeta + 2 * tc.OMEGA * np.sin(P)
    zero = np.zeros_like(P)
    with open(out, "wb") as f:
        for F in (u, zero, h, zero, zeta):
            np.ascontiguousarray(F, dtype=np.float64).tofile(f)
    print("wrote", out, "h range", h.min(), h.max(), "max |zeta|", np.abs(zeta).max())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
