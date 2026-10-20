"""Numerical routines for the shallow-water studies."""


import numpy as np
from . import testcases as tc
from .lwgg import _r, _unit, _dot
from . import lsops

G = tc.G


class Geom:
    def __init__(self, C00, C10, C11, C01, n, a):
        corners = [C00, C10, C11, C01]
        area = _dot(np.cross(C11 - C00, C01 - C10, axis=0), n) / 2
        sgn = np.sign(np.median(area))
        self.sign = sgn
        self.area = sgn * area * a * a
        self.T, self.N = [], []
        for p, q in zip(corners, corners[1:] + corners[:1]):
            d = sgn * (q - p) * a
            self.T.append(d)
            self.N.append(np.cross(d, _unit(p + q), axis=0))
        self.n = n


def _ops(geom, cs_scalar, cs_vec, cs_flux, area):
    pairs = lambda cs: list(zip(cs, cs[1:] + cs[:1]))
    grad = sum(0.5 * (p + q) * N for (p, q), N in zip(pairs(cs_scalar), geom.N)) / area
    curl = sum(_dot(0.5 * (p + q), T) for (p, q), T in zip(pairs(cs_vec), geom.T)) / area
    div = sum(_dot(0.5 * (p + q), N) for (p, q), N in zip(pairs(cs_flux), geom.N)) / area
    return grad, curl, div


class LWTorus:
    def __init__(self, X, Xc, case, bcells, a=tc.A, ls_radius=2.6, q=2, nu4=0.0):
        self.X, self.Xc, self.case, self.a = X, Xc, case, a
        self.nu4 = nu4
        M0, M1 = X.shape[1:]
        P00, P10, P11, P01 = X, _r(X, 1, 0), _r(X, 1, 1), _r(X, 0, 1)
        self.prim = Geom(P00, P10, P11, P01, Xc, a)
        Q00, Q10, Q11, Q01 = _r(Xc, -1, -1), _r(Xc, 0, -1), Xc, _r(Xc, -1, 0)
        self.dual = Geom(Q00, Q10, Q11, Q01, X, a)
        self.bcells = bcells
        bi = np.array(bcells)
        self.bmask = np.zeros((M0, M1), bool); self.bmask[bi[:, 0], bi[:, 1]] = True
        pa = self.prim.area.copy()
        if (pa[~self.bmask] <= 0).any() or (self.dual.area <= 0).any():
            raise ValueError("non-positive cell area outside branch cells")
        pa[self.bmask] = 1.0
        self.prim_area = pa
        self.f, self.fc = case.coriolis(X), case.coriolis(Xc)
        self.hs, self.hsc = case.topography(X), case.topography(Xc)
        self.weight = self.dual.area

        smean2 = self.weight.mean() * 2.0 / np.sqrt(3.0)
        self.flat_scale = (2.0 / 3.0) / smean2 * self.weight.mean() / self.weight

        self.ls = []
        for (i, j) in bcells:
            offs, idx, Xs = [], [], []
            for di in range(-2, 4):
                for dj in range(-2, 4):
                    dz = (di - 0.5) + (dj - 0.5) * np.exp(1j * np.pi / 3)
                    if abs(dz) <= ls_radius:
                        offs.append((di, dj))
                        idx.append(((i + di) % M0, (j + dj) % M1))
            idx = np.array(idx)
            Xs = X[:, idx[:, 0], idx[:, 1]]
            Xb = Xc[:, i, j]
            t1, t2 = lsops.frame(Xb[:, None, None])
            t1, t2 = t1[:, 0, 0], t2[:, 0, 0]
            Y = np.stack([Xs.T @ t1, Xs.T @ t2], -1)
            hsc = np.sqrt((Y ** 2).sum(-1)).max()
            Ys = Y / hsc
            mons = lsops.monomials(q)
            V = np.stack([Ys[:, 0] ** p * Ys[:, 1] ** r for p, r in mons], -1)
            w = 1.0 / (1.0 + np.sqrt((Ys ** 2).sum(-1))) ** 2
            L = np.linalg.pinv(V * w[:, None]) * w[None, :]
            w0 = L[mons.index((0, 0))]
            gx = L[mons.index((1, 0))] / (hsc * a)
            gy = L[mons.index((0, 1))] / (hsc * a)
            W = t1[:, None] * gx[None] + t2[:, None] * gy[None]
            self.ls.append(dict(cell=(i, j), idx=idx, w0=w0, W=W, Xb=Xb,
                                fb=case.coriolis(Xb[:, None])[0], hsb=float(case.topography(Xb[:, None])[0]),
                                hs_vals=case.topography(Xs)))

    def _branch_predictor(self, D, v, dt, Dh, vh):
        for L in self.ls:
            I, J = L["idx"][:, 0], L["idx"][:, 1]
            Ds, vs = D[I, J], v[:, I, J]
            w0, W, Xb = L["w0"], L["W"], L["Xb"]
            Db = w0 @ Ds
            gD = W @ Ds
            gHs = W @ L["hs_vals"]
            vb = vs @ w0
            gv = np.stack([W @ vs[m] for m in range(3)])
            divv = sum(gv[m][m] for m in range(3))
            zeta = Xb @ sum(np.cross(gv[m], np.eye(3)[m]) for m in range(3))
            vb = vb - (vb @ Xb) * Xb
            gK = G * (gD + gHs) + sum(vs[m] @ w0 * gv[m] for m in range(3))
            i, j = L["cell"]
            Dh[i, j] = Db - 0.5 * dt * (vb @ gD + Db * divv)
            rhs = (L["fb"] + zeta) * np.cross(Xb, vb) + gK
            vv = vb - 0.5 * dt * rhs
            vh[:, i, j] = vv - (vv @ Xb) * Xb

    def step(self, D, v, dt):

        cD = [D, _r(D, 1, 0), _r(D, 1, 1), _r(D, 0, 1)]
        cv = [v, _r(v, 1, 0), _r(v, 1, 1), _r(v, 0, 1)]
        K = G * (D + self.hs) + 0.5 * _dot(v, v)
        cK = [K, _r(K, 1, 0), _r(K, 1, 1), _r(K, 0, 1)]
        grad, curl, div = _ops(self.prim, cK, cv, [d * w for d, w in zip(cD, cv)], self.prim_area)
        n = self.Xc
        Dbar = 0.25 * sum(cD)
        vbar = 0.25 * sum(cv)
        vbar = vbar - _dot(vbar, n) * n
        Dh = Dbar - 0.5 * dt * div
        vh = vbar - 0.5 * dt * ((self.fc + curl) * np.cross(n, vbar, axis=0) + grad)
        vh = vh - _dot(vh, n) * n
        self._branch_predictor(D, v, dt, Dh, vh)

        Kh = G * (Dh + self.hsc) + 0.5 * _dot(vh, vh)
        dD = [_r(Dh, -1, -1), _r(Dh, 0, -1), Dh, _r(Dh, -1, 0)]
        dv = [_r(vh, -1, -1), _r(vh, 0, -1), vh, _r(vh, -1, 0)]
        dK = [_r(Kh, -1, -1), _r(Kh, 0, -1), Kh, _r(Kh, -1, 0)]
        grad, curl, div = _ops(self.dual, dK, dv, [d * w for d, w in zip(dD, dv)], self.dual.area)
        vmid = 0.25 * sum(dv)
        n = self.X
        Dn = D - dt * div
        vn = v - dt * ((self.f + curl) * np.cross(n, vmid, axis=0) + grad)
        vn = vn - _dot(vn, n) * n
        if self.nu4:
            Dn = Dn - dt * self.nu4 * self.lap(self.lap(Dn))
            vn = vn - dt * self.nu4 * np.stack([self.lap(self.lap(vn[k])) for k in range(3)])
            vn = vn - _dot(vn, n) * n
        return Dn, vn

    def lap(self, phi):
        nb = (_r(phi, 1, 0) + _r(phi, -1, 0) + _r(phi, 0, 1) + _r(phi, 0, -1)
              + _r(phi, 1, -1) + _r(phi, -1, 1))
        return self.flat_scale * (nb - 6.0 * phi)

    def mass(self, D):
        return (self.weight * D).sum()
