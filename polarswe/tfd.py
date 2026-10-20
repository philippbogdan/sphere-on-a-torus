"""Solve shallow water with local differences on a periodic index grid."""


import numpy as np
from . import testcases as tc

AXI, AXJ = -2, -1


def xp_of(a):
    if isinstance(a, np.ndarray):
        return np
    import cupy
    return cupy


def sh(F, s, ax):
    return xp_of(F).roll(F, -s, axis=ax)


def D(F, ax, order):
    if order == 2:
        return 0.5 * (sh(F, 1, ax) - sh(F, -1, ax))
    if order == 4:
        return (8.0 * (sh(F, 1, ax) - sh(F, -1, ax)) - (sh(F, 2, ax) - sh(F, -2, ax))) / 12.0
    if order == 6:
        return (45.0 * (sh(F, 1, ax) - sh(F, -1, ax)) - 9.0 * (sh(F, 2, ax) - sh(F, -2, ax))
                + (sh(F, 3, ax) - sh(F, -3, ax))) / 60.0
    if order == 8:
        return (672.0 * (sh(F, 1, ax) - sh(F, -1, ax)) - 168.0 * (sh(F, 2, ax) - sh(F, -2, ax))
                + 32.0 * (sh(F, 3, ax) - sh(F, -3, ax)) - 3.0 * (sh(F, 4, ax) - sh(F, -4, ax))) / 840.0
    raise ValueError(order)


def dot(a, b):
    return (a * b).sum(0)


def cross(a, b):
    return xp_of(a).stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class Metric:
    def __init__(self, X, order, a=tc.A, allow_sign_change=False):
        self.X = X
        n = X
        A1 = a * D(X, AXI, order)
        A2 = a * D(X, AXJ, order)
        A1 = A1 - dot(A1, n) * n
        A2 = A2 - dot(A2, n) * n
        self.A1, self.A2 = A1, A2
        g11, g12, g22 = dot(A1, A1), dot(A1, A2), dot(A2, A2)
        det = g11 * g22 - g12 ** 2
        self.sqrtG = dot(np.cross(A1, A2, axis=0), n)
        if not allow_sign_change and not (np.all(self.sqrtG > 0) or np.all(self.sqrtG < 0)):
            raise ValueError("Jacobian changes sign")

        self.a1 = (g22 * A1 - g12 * A2) / det
        self.a2 = (-g12 * A1 + g11 * A2) / det

        U1 = a * D(X, AXI, order)
        U2 = a * D(X, AXJ, order)
        self.B1, self.B2_ = U2, U1
        self.sqrtG_s = dot(cross(U1, U2), n)
        self.c1 = cross(U2, n) / self.sqrtG_s
        self.c2 = cross(n, U1) / self.sqrtG_s

        B1 = a * D(X, AXI, 2); B2 = a * D(X, AXJ, 2)
        B1 = B1 - dot(B1, n) * n; B2 = B2 - dot(B2, n) * n
        h11, h12, h22 = dot(B1, B1), dot(B1, B2), dot(B2, B2)
        hdet = h11 * h22 - h12 ** 2
        sg = np.sqrt(hdet)
        self.lap_sg = sg

        self.L11 = sg * h22 / hdet
        self.L22 = sg * h11 / hdet
        self.L12 = -sg * h12 / hdet

    def grad(self, phi, order):
        return self.a1 * D(phi, AXI, order) + self.a2 * D(phi, AXJ, order)

    def div(self, F, order):
        sg = self.sqrtG
        return (D(sg * dot(F, self.a1), AXI, order) + D(sg * dot(F, self.a2), AXJ, order)) / sg

    def curl(self, v, order):
        return (D(dot(v, self.A2), AXI, order) - D(dot(v, self.A1), AXJ, order)) / self.sqrtG

    def div_split(self, F, order):


        G = cross(self.X, F)
        return (dot(D(G, AXI, order), self.B1) - dot(D(G, AXJ, order), self.B2_)) / self.sqrtG_s

    def curl_split(self, F, order):
        return (dot(D(F, AXI, order), self.B1) - dot(D(F, AXJ, order), self.B2_)) / self.sqrtG_s

    def grad_split(self, phi, order):
        return self.c1 * D(phi, AXI, order) + self.c2 * D(phi, AXJ, order)

    def div_cart(self, F, order):
        return dot(self.a1, D(F, AXI, order)) + dot(self.a2, D(F, AXJ, order))

    def curl_cart(self, F, order):
        X = self.X
        return dot(np.cross(self.a1, D(F, AXI, order), axis=0) + np.cross(self.a2, D(F, AXJ, order), axis=0), X)

    def lap(self, phi):
        L11, L22, L12 = self.L11, self.L22, self.L12

        a_ip = 0.5 * (L11 + sh(L11, 1, AXI)); a_im = 0.5 * (L11 + sh(L11, -1, AXI))
        b_jp = 0.5 * (L22 + sh(L22, 1, AXJ)); b_jm = 0.5 * (L22 + sh(L22, -1, AXJ))
        out = a_ip * (sh(phi, 1, AXI) - phi) - a_im * (phi - sh(phi, -1, AXI))
        out = out + b_jp * (sh(phi, 1, AXJ) - phi) - b_jm * (phi - sh(phi, -1, AXJ))

        out = out + 0.5 * (sh(L12 * D(phi, AXJ, 2), 1, AXI) - sh(L12 * D(phi, AXJ, 2), -1, AXI))
        out = out + 0.5 * (sh(L12 * D(phi, AXI, 2), 1, AXJ) - sh(L12 * D(phi, AXI, 2), -1, AXJ))
        return out / (self.sqrtG if getattr(self, "lap_w", None) is None else self.lap_w)


class TFD:
    def __init__(self, X, case, order=4, nu4=0.0, integrator="rk4", a=tc.A, form="flux",
                 special_rings=0, ls_m=2, ls_q=3, allow_sign_change=False, hv_mode="stage", lap_kind="metric",
                 hv_order=4):
        self.X = X
        self.case = case
        self.order = order
        self.m = Metric(X, order, a, allow_sign_change)
        self.f = case.coriolis(X)
        self.hs = case.topography(X)
        self.nu4 = nu4
        self.integrator = integrator
        self.form = form
        self.hv_mode = hv_mode
        self.hv_order = hv_order
        self.mass_weight = np.abs(self.m.sqrtG_s) if form == "split" else np.abs(self.m.sqrtG)
        self.lap_kind = lap_kind
        if lap_kind == "flat":
            from . import ellip
            if ellip.LATTICE != "hex":
                raise ValueError("the flat Laplacian is the hexagonal lattice's")
        self.weight = np.abs(self.m.sqrtG)

        smean2 = self.weight.mean() * 2.0 / np.sqrt(3.0)
        self.flat_scale = (2.0 / 3.0) / smean2 * self.weight.mean() / self.weight
        self.special = None
        self.cone = None
        if special_rings > 0:
            from . import lsops
            mask = branch_mask(X.shape[1], special_rings)
            offs, idx, W = lsops.ls_weights(X, mask, m=ls_m, q=ls_q)
            self.special = (offs, idx, W / a)
            self.Xs = X[:, idx[:, 0], idx[:, 1]]

    def _gather(self, F, di, dj):
        offs, idx, W = self.special
        M0, M1 = F.shape[-2:]
        return F[..., (idx[:, 0] + di) % M0, (idx[:, 1] + dj) % M1]

    def _ls_grad(self, phi):
        offs, idx, W = self.special
        return sum(W[:, :, k] * self._gather(phi, di, dj) for k, (di, dj) in enumerate(offs))

    def _ls_div(self, F):
        offs, idx, W = self.special
        return sum((W[:, :, k] * self._gather(F, di, dj)).sum(0) for k, (di, dj) in enumerate(offs))

    def _ls_curl(self, F):
        offs, idx, W = self.special
        c = sum(np.cross(W[:, :, k], self._gather(F, di, dj), axis=0) for k, (di, dj) in enumerate(offs))
        return (c * self.Xs).sum(0)

    def rhs(self, Dp, v):
        m, o = self.m, self.order
        X = self.X
        K = tc.G * (Dp + self.hs) + 0.5 * dot(v, v)
        if self.form == "flux":
            zeta = m.curl(v, o)
            dD = -m.div(Dp * v, o)
            gK = m.grad(K, o)
        elif self.form == "fluxsc":


            zeta = m.curl_split(v, o)
            dD = -m.div(Dp * v, o)
            gK = m.grad(K, o)
        elif self.form == "split":
            zeta = m.curl_split(v, o)
            dD = -m.div_split(Dp * v, o)
            gK = m.grad_split(K, o)
        else:
            zeta = m.curl_cart(v, o)
            dD = -(dot(v, m.grad(Dp, o)) + Dp * m.div_cart(v, o))
            gK = m.grad(K, o)
        if self.cone is not None:
            C = self.cone
            zeta = zeta + C.curl(v)
            dD = dD - C.div(Dp * v)
            gK = gK + C.grad(K)
        if self.special is not None:
            idx = self.special[1]
            I, J = idx[:, 0], idx[:, 1]
            zeta[I, J] = self._ls_curl(v)
            dD[I, J] = -self._ls_div(Dp * v)
            gK[:, I, J] = self._ls_grad(K)
        dv = -(self.f + zeta) * cross(X, v) - gK
        if np.any(self.nu4) and self.hv_mode == "stage":
            dD = dD - self.filtered(Dp)
            dv = dv - xp_of(v).stack([self.filtered(v[k]) for k in range(3)])
        dv = dv - dot(dv, X) * X
        return dD, dv

    def lap(self, phi):
        if self.lap_kind == "metric":
            return self.m.lap(phi)
        nb = (sh(phi, 1, AXI) + sh(phi, -1, AXI) + sh(phi, 1, AXJ) + sh(phi, -1, AXJ)
              + sh(sh(phi, 1, AXI), -1, AXJ) + sh(sh(phi, -1, AXI), 1, AXJ))
        return self.flat_scale * (nb - 6.0 * phi)

    def hyper(self, f):

        if self.hv_order not in (4, 6, 8, 10):
            raise ValueError(self.hv_order)
        n = self.hv_order // 2
        for k in range(n):
            f = self.lap(f)
        return f if n % 2 == 0 else -f

    def filtered(self, f):

        if np.ndim(self.nu4) == 0:
            return self.nu4 * self.hyper(f)
        if self.hv_order != 4:
            raise ValueError("a filter strength varying in space is implemented for del^4")
        return self.lap(self.nu4 * self.lap(f))

    def step(self, Dp, v, dt):
        """Advance depth in metres and tangent Cartesian velocity by dt seconds."""
        Dn, vn = self._step(Dp, v, dt)
        if np.any(self.nu4) and self.hv_mode == "step":
            Dn = Dn - dt * self.filtered(Dn)
            vn = vn - dt * xp_of(vn).stack([self.filtered(vn[k]) for k in range(3)])
            vn = vn - dot(vn, self.X) * self.X
        return Dn, vn

    def _step(self, Dp, v, dt):
        if self.integrator == "rk4":
            k1 = self.rhs(Dp, v)
            k2 = self.rhs(Dp + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
            k3 = self.rhs(Dp + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
            k4 = self.rhs(Dp + dt * k3[0], v + dt * k3[1])
            Dn = Dp + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            vn = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        elif self.integrator.startswith("kg"):

            c = kg_coeffs(self.integrator[2:])
            Dn, vn = Dp, v
            for ck in c:
                dD, dv = self.rhs(Dn, vn)
                Dn, vn = Dp + ck * dt * dD, v + ck * dt * dv
        elif self.integrator.startswith("ls"):

            A, B = ls_coeffs(self.integrator[2:])
            Dn, vn = Dp, v
            qD, qv = 0.0 * Dp, 0.0 * v
            for Ak, Bk in zip(A, B):
                dD, dv = self.rhs(Dn, vn)
                qD, qv = Ak * qD + dt * dD, Ak * qv + dt * dv
                Dn, vn = Dn + Bk * qD, vn + Bk * qv
        elif self.integrator == "ssprk3":
            k1 = self.rhs(Dp, v)
            D1, v1 = Dp + dt * k1[0], v + dt * k1[1]
            k2 = self.rhs(D1, v1)
            D2, v2 = 0.75 * Dp + 0.25 * (D1 + dt * k2[0]), 0.75 * v + 0.25 * (v1 + dt * k2[1])
            k3 = self.rhs(D2, v2)
            Dn = Dp / 3 + 2 / 3 * (D2 + dt * k3[0])
            vn = v / 3 + 2 / 3 * (v2 + dt * k3[1])
        else:
            raise ValueError(self.integrator)
        vn = vn - dot(vn, self.X) * self.X
        return Dn, vn

    def mass(self, Dp):
        """Integrate depth using the corrected area weights of the double cover."""
        return (self.mass_weight * Dp).sum()

    def set_cone(self, C):


        if self.form != "flux":
            raise ValueError("cone operators correct the flux form")
        self.cone = C
        self.mass_weight = C.weight
        self.m.lap_w = np.sign(self.m.sqrtG) * C.weight
        return self


def branch_mask(M, rings):
    from . import ellip, torusmap
    i = np.arange(M)
    I, J = np.meshgrid(i, i, indexing="ij")
    zeta = ((I + 0.5) + (J + 0.5) * ellip.EPS) / M
    d = torusmap.nearest_branch(zeta) * M
    return np.abs(d) <= rings + 0.5


def to_device(S):
    import cupy as cp
    m = S.m
    for name in ("X", "A1", "A2", "a1", "a2", "sqrtG", "L11", "L22", "L12", "lap_sg"):
        setattr(m, name, cp.asarray(getattr(m, name)))
    for name in ("X", "f", "hs", "weight", "flat_scale", "mass_weight"):
        setattr(S, name, cp.asarray(getattr(S, name)))
    if np.ndim(S.nu4):
        S.nu4 = cp.asarray(S.nu4)
    if getattr(m, "lap_w", None) is not None:
        m.lap_w = cp.asarray(m.lap_w)
    if getattr(S, "cone", None) is not None:
        S.cone = S.cone.to_device()
    return S


_KG = {}


def ls_coeffs(s):
    import json, os
    tab = json.load(open(os.path.join(os.path.dirname(__file__), "kg_coeffs.json")))
    e = tab["ls" + str(s)]
    return e["A"], e["B"]


def kg_coeffs(s):
    if s not in _KG:
        import json, os
        tab = json.load(open(os.path.join(os.path.dirname(__file__), "kg_coeffs.json")))
        _KG[s] = tab[str(s)]["c"]
    return _KG[s]
