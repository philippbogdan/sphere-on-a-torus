"""Run a numerical check and save its measurements."""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import testcases as tc
nlat, amp, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
nlon = 2 * nlat; x, wq = np.polynomial.legendre.leggauss(nlat); lat = np.arcsin(x)[::-1]
lon = 2 * np.pi * np.arange(nlon) / nlon; L, P = np.meshgrid(lon, lat)
X = np.stack([np.cos(P) * np.cos(L), np.cos(P) * np.sin(L), np.sin(P)])
C = tc.TC6(); v3 = C.velocity(X); e, n = tc.east_north(X)
w_, K, R = C.w, C.K, C.R
zeta = 2 * w_ * np.sin(P) - K * np.sin(P) * np.cos(P) ** R * (R * R + 3 * R + 2) * np.cos(R * L) + 2 * tc.OMEGA * np.sin(P)
p2 = 1.5 * np.sin(P) ** 2 - 0.5; W = np.broadcast_to(wq[::-1][:, None], P.shape); p2 = p2 / math.sqrt((W * p2 ** 2).sum() / W.sum())
with open(out, "wb") as f:
    for F in ((v3 * e).sum(0), (v3 * n).sum(0), C.height(X) + amp * p2, np.zeros_like(P), zeta):
        np.ascontiguousarray(F, dtype=np.float64).tofile(f)
