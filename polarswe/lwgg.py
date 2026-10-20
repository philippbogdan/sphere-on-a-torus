"""Numerical routines for the shallow-water studies."""


import numpy as np
from . import testcases as tc

AX_I, AX_J = -2, -1


def _r(F, si, sj):
    if si:
        F = np.roll(F, -si, axis=AX_I)
    if sj:
        F = np.roll(F, -sj, axis=AX_J)
    return F


def _unit(V):
    return V / np.linalg.norm(V, axis=0)


def _dot(a, b):
    return (a * b).sum(0)


class CellGeom:

    def __init__(self, C00, C10, C11, C01, n):
        corners = [C00, C10, C11, C01]
        self.n = n
        s = np.sign(_dot(np.cross(C10 - C00, C01 - C00, axis=0), n))
        if not (np.all(s == s.flat[0])):
            raise ValueError("cells with mixed orientation")
        self.s = s.flat[0]
        self.T, self.N = [], []
        for a, b in zip(corners, corners[1:] + corners[:1]):
            d = self.s * (b - a)
            ne = _unit(a + b)
            self.T.append(d)
            self.N.append(np.cross(d, ne, axis=0))
        self.area = self.s * _dot(np.cross(C11 - C00, C01 - C10, axis=0), n) / 2


class LWGG:
    def __init__(self, X, case, a=tc.A):
        self.X = X
        self.case = case
        self.a = a
        P00, P10, P11, P01 = X, _r(X, 1, 0), _r(X, 1, 1), _r(X, 0, 1)
        self.Xc = _unit(P00 + P10 + P11 + P01)
        self.prim = CellGeom(P00, P10, P11, P01, self.Xc)
        Q00, Q10, Q11, Q01 = _r(self.Xc, -1, -1), _r(self.Xc, 0, -1), self.Xc, _r(self.Xc, -1, 0)
        self.dual = CellGeom(Q00, Q10, Q11, Q01, X)

        for g in (self.prim, self.dual):
            g.T = [t * a for t in g.T]
            g.N = [nn * a for nn in g.N]
            g.area = g.area * a * a
        self.f = case.coriolis(X)
        self.fc = case.coriolis(self.Xc)
        self.hs = case.topography(X)
        self.hsc = case.topography(self.Xc)


    @staticmethod
    def _prim_corners(F):
        return [F, _r(F, 1, 0), _r(F, 1, 1), _r(F, 0, 1)]

    @staticmethod
    def _dual_corners(Fc):
        return [_r(Fc, -1, -1), _r(Fc, 0, -1), Fc, _r(Fc, -1, 0)]

    @staticmethod
    def _ops(geom, cs_scalar=None, cs_vec=None, cs_flux=None):
        out = {}
        pairs = lambda cs: list(zip(cs, cs[1:] + cs[:1]))
        if cs_scalar is not None:
            g = sum(0.5 * (p + q) * N for (p, q), N in zip(pairs(cs_scalar), geom.N))
            out["grad"] = g / geom.area
        if cs_vec is not None:
            c = sum(_dot(0.5 * (p + q), T) for (p, q), T in zip(pairs(cs_vec), geom.T))
            out["curl"] = c / geom.area
        if cs_flux is not None:
            d = sum(_dot(0.5 * (p + q), N) for (p, q), N in zip(pairs(cs_flux), geom.N))
            out["div"] = d / geom.area
        return out

    def step(self, D, v, dt):
        g = tc.G

        cD = self._prim_corners(D)
        cv = self._prim_corners(v)
        K = g * (D + self.hs) + 0.5 * _dot(v, v)
        ops = self._ops(self.prim, self._prim_corners(K), cv, [d * w for d, w in zip(cD, cv)])
        Dbar = 0.25 * sum(cD)
        vbar = 0.25 * sum(cv)
        n = self.Xc
        vbar = vbar - _dot(vbar, n) * n
        Dh = Dbar - 0.5 * dt * ops["div"]
        rhs = (self.fc + ops["curl"]) * np.cross(n, vbar, axis=0) + ops["grad"]
        vh = vbar - 0.5 * dt * rhs
        vh = vh - _dot(vh, n) * n

        Kh = g * (Dh + self.hsc) + 0.5 * _dot(vh, vh)
        dD = self._dual_corners(Dh)
        dv = self._dual_corners(vh)
        ops = self._ops(self.dual, self._dual_corners(Kh), dv, [d * w for d, w in zip(dD, dv)])
        vmid = 0.25 * sum(dv)
        n = self.X
        Dn = D - dt * ops["div"]
        vn = v - dt * ((self.f + ops["curl"]) * np.cross(n, vmid, axis=0) + ops["grad"])
        vn = vn - _dot(vn, n) * n
        return Dn, vn

    def mass(self, D):
        return (self.dual.area * D).sum()


def norms(h, href, w):
    I = lambda f: (w * f).sum()
    e = h - href
    return (I(np.abs(e)) / I(np.abs(href)), np.sqrt(I(e * e) / I(href * href)),
            np.abs(e).max() / np.abs(href).max())
