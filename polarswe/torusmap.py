"""Numerical routines for the shallow-water studies."""


import numpy as np
from . import ellip

E_ROOTS = ellip.roots()
if ellip.LATTICE == "hex":
    SCALE = np.abs(E_ROOTS[0]) * np.sqrt(2.0)
    ORIENT = np.eye(3)
else:


    SCALE = float(np.abs(E_ROOTS[0]))
    ORIENT = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
C_BRANCH = 2.0 * SCALE


def nearest_branch(zeta):
    best = None
    for b in ellip.BRANCH_POINTS:
        d = ellip.reduce_lattice(zeta - b)
        if best is None:
            best = d.copy()
        else:
            m = np.abs(d) < np.abs(best)
            best[m] = d[m]
    return best


def radial_g(rho, a, r0, k):
    t = np.clip(1.0 - rho / r0, 0.0, None)
    return np.sqrt(rho ** 2 + a * rho * t ** k)


def correct(zeta, a, r0, k):
    if a == 0.0:
        return zeta
    d = nearest_branch(zeta)
    rho = np.abs(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        fac = np.where(rho > 0, radial_g(rho, a, r0, k) / rho, 1.0)
    return zeta + d * (fac - 1.0)


def stereo(w):
    r2 = np.abs(w) ** 2
    inf = ~np.isfinite(r2)
    with np.errstate(invalid="ignore"):
        X = np.stack([2 * w.real, 2 * w.imag, r2 - 1.0]) / (r2 + 1.0)
    X[:, inf] = np.array([0.0, 0.0, 1.0])[:, None]
    if ellip.LATTICE != "hex":
        X = np.tensordot(ORIENT, X, axes=1)
    return X


def sphere_point(zeta, a=0.0, r0=0.25, k=3, rotation=None):
    zc = correct(np.asarray(zeta, dtype=complex), a, r0, k)
    X = stereo(ellip.wp(zc) / SCALE)
    if rotation is not None:
        X = np.tensordot(rotation, X, axes=1)
    return X


def grid(M, a=0.0, r0=0.25, k=3, rotation=None):


    i = np.arange(M)
    I, J = np.meshgrid(i, i, indexing="ij")
    zeta = ((I + 0.5) + (J + 0.5) * ellip.EPS) / M
    return zeta, sphere_point(zeta, a, r0, k, rotation)
