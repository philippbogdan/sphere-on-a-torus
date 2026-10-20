"""Compile the shallow-water kernel with Numba."""


import numpy as np
import numba as nb
from . import testcases as tc

G = tc.G


@nb.njit(inline="always")
def _d(F, i, j, ax, order, M0, M1):
    if ax == 0:
        ip, im = (i + 1) % M0, (i - 1) % M0
        if order == 2:
            return 0.5 * (F[ip, j] - F[im, j])
        ip2, im2 = (i + 2) % M0, (i - 2) % M0
        if order == 4:
            return (8.0 * (F[ip, j] - F[im, j]) - (F[ip2, j] - F[im2, j])) / 12.0
        ip3, im3 = (i + 3) % M0, (i - 3) % M0
        if order == 6:
            return (45.0 * (F[ip, j] - F[im, j]) - 9.0 * (F[ip2, j] - F[im2, j]) + (F[ip3, j] - F[im3, j])) / 60.0
        ip4, im4 = (i + 4) % M0, (i - 4) % M0
        return (672.0 * (F[ip, j] - F[im, j]) - 168.0 * (F[ip2, j] - F[im2, j]) + 32.0 * (F[ip3, j] - F[im3, j])
                - 3.0 * (F[ip4, j] - F[im4, j])) / 840.0
    else:
        jp, jm = (j + 1) % M1, (j - 1) % M1
        if order == 2:
            return 0.5 * (F[i, jp] - F[i, jm])
        jp2, jm2 = (j + 2) % M1, (j - 2) % M1
        if order == 4:
            return (8.0 * (F[i, jp] - F[i, jm]) - (F[i, jp2] - F[i, jm2])) / 12.0
        jp3, jm3 = (j + 3) % M1, (j - 3) % M1
        if order == 6:
            return (45.0 * (F[i, jp] - F[i, jm]) - 9.0 * (F[i, jp2] - F[i, jm2]) + (F[i, jp3] - F[i, jm3])) / 60.0
        jp4, jm4 = (j + 4) % M1, (j - 4) % M1
        return (672.0 * (F[i, jp] - F[i, jm]) - 168.0 * (F[i, jp2] - F[i, jm2]) + 32.0 * (F[i, jp3] - F[i, jm3])
                - 3.0 * (F[i, jp4] - F[i, jm4])) / 840.0


@nb.njit(parallel=True)
def _pass1(Dp, v, hs, A1, A2, a1, a2, sg, K, vA1, vA2, F1, F2):
    M0, M1 = Dp.shape
    for i in nb.prange(M0):
        for j in range(M1):
            vx, vy, vz = v[0, i, j], v[1, i, j], v[2, i, j]
            d = Dp[i, j]
            K[i, j] = G * (d + hs[i, j]) + 0.5 * (vx * vx + vy * vy + vz * vz)
            vA1[i, j] = vx * A1[0, i, j] + vy * A1[1, i, j] + vz * A1[2, i, j]
            vA2[i, j] = vx * A2[0, i, j] + vy * A2[1, i, j] + vz * A2[2, i, j]
            s = sg[i, j] * d
            F1[i, j] = s * (vx * a1[0, i, j] + vy * a1[1, i, j] + vz * a1[2, i, j])
            F2[i, j] = s * (vx * a2[0, i, j] + vy * a2[1, i, j] + vz * a2[2, i, j])


@nb.njit(parallel=True)
def _pass2(v, X, a1, a2, isg, f, K, vA1, vA2, F1, F2, order, dD, dv):
    M0, M1 = K.shape
    for i in nb.prange(M0):
        for j in range(M1):
            zeta = (_d(vA2, i, j, 0, order, M0, M1) - _d(vA1, i, j, 1, order, M0, M1)) * isg[i, j]
            div = (_d(F1, i, j, 0, order, M0, M1) + _d(F2, i, j, 1, order, M0, M1)) * isg[i, j]
            dKi = _d(K, i, j, 0, order, M0, M1)
            dKj = _d(K, i, j, 1, order, M0, M1)
            x0, x1, x2 = X[0, i, j], X[1, i, j], X[2, i, j]
            vx, vy, vz = v[0, i, j], v[1, i, j], v[2, i, j]
            fz = f[i, j] + zeta
            c0 = x1 * vz - x2 * vy
            c1 = x2 * vx - x0 * vz
            c2 = x0 * vy - x1 * vx
            g0 = -(fz * c0) - (a1[0, i, j] * dKi + a2[0, i, j] * dKj)
            g1 = -(fz * c1) - (a1[1, i, j] * dKi + a2[1, i, j] * dKj)
            g2 = -(fz * c2) - (a1[2, i, j] * dKi + a2[2, i, j] * dKj)
            s = g0 * x0 + g1 * x1 + g2 * x2
            dv[0, i, j] = g0 - s * x0
            dv[1, i, j] = g1 - s * x1
            dv[2, i, j] = g2 - s * x2
            dD[i, j] = -div


@nb.njit(parallel=True)
def _lap_metric(phi, L11, L22, L12, isg, out):
    M0, M1 = phi.shape
    for i in nb.prange(M0):
        ip, im = (i + 1) % M0, (i - 1) % M0
        for j in range(M1):
            jp, jm = (j + 1) % M1, (j - 1) % M1
            p = phi[i, j]
            a_ip = 0.5 * (L11[i, j] + L11[ip, j]); a_im = 0.5 * (L11[i, j] + L11[im, j])
            b_jp = 0.5 * (L22[i, j] + L22[i, jp]); b_jm = 0.5 * (L22[i, j] + L22[i, jm])
            o = a_ip * (phi[ip, j] - p) - a_im * (p - phi[im, j]) + b_jp * (phi[i, jp] - p) - b_jm * (p - phi[i, jm])

            tip = L12[ip, j] * 0.5 * (phi[ip, (j + 1) % M1] - phi[ip, (j - 1) % M1])
            tim = L12[im, j] * 0.5 * (phi[im, (j + 1) % M1] - phi[im, (j - 1) % M1])
            tjp = L12[i, jp] * 0.5 * (phi[(i + 1) % M0, jp] - phi[(i - 1) % M0, jp])
            tjm = L12[i, jm] * 0.5 * (phi[(i + 1) % M0, jm] - phi[(i - 1) % M0, jm])
            o += 0.5 * (tip - tim) + 0.5 * (tjp - tjm)
            out[i, j] = o * isg[i, j]


@nb.njit(parallel=True)
def _lap_flat(phi, scale, out):
    M0, M1 = phi.shape
    for i in nb.prange(M0):
        ip, im = (i + 1) % M0, (i - 1) % M0
        for j in range(M1):
            jp, jm = (j + 1) % M1, (j - 1) % M1
            out[i, j] = scale[i, j] * (phi[ip, j] + phi[im, j] + phi[i, jp] + phi[i, jm]
                                       + phi[ip, jm] + phi[im, jp] - 6.0 * phi[i, j])


@nb.njit(parallel=True)
def _axpy_state(Dp, v, c, dD, dv, Do, vo):
    M0, M1 = Dp.shape
    for i in nb.prange(M0):
        for j in range(M1):
            Do[i, j] = Dp[i, j] + c * dD[i, j]
            for k in range(3):
                vo[k, i, j] = v[k, i, j] + c * dv[k, i, j]


@nb.njit(parallel=True)
def _project(v, X):
    M0, M1 = X.shape[1], X.shape[2]
    for i in nb.prange(M0):
        for j in range(M1):
            s = v[0, i, j] * X[0, i, j] + v[1, i, j] * X[1, i, j] + v[2, i, j] * X[2, i, j]
            for k in range(3):
                v[k, i, j] -= s * X[k, i, j]


class FastTFD:
    def __init__(self, S):
        if S.form != "flux" or S.special is not None:
            raise ValueError("fast path supports form='flux' without special stencils")
        self.cone = getattr(S, "cone", None)
        if S.order not in (2, 4, 6, 8):
            raise ValueError(f"fast path supports difference orders 2, 4, 6, 8, not {S.order}")
        if getattr(S, "hv_order", 4) not in (4, 6, 8, 10):
            raise ValueError("fast path implements filters of order 4, 6, 8, 10")
        self.hv_order = getattr(S, "hv_order", 4)
        m = S.m
        self.S = S
        c = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        self.X, self.A1, self.A2, self.a1, self.a2 = c(S.X), c(m.A1), c(m.A2), c(m.a1), c(m.a2)
        self.sg, self.isg = c(m.sqrtG), c(1.0 / m.sqrtG)

        self.isg_lap = c(1.0 / m.lap_w) if getattr(m, "lap_w", None) is not None else self.isg
        self.f, self.hs = c(S.f), c(S.hs)
        self.L11, self.L22, self.L12 = c(m.L11), c(m.L22), c(m.L12)
        self.flat_scale = c(S.flat_scale)
        self.order, self.nu4 = S.order, S.nu4
        self.hv_mode, self.lap_kind, self.integrator = S.hv_mode, S.lap_kind, S.integrator
        shp = self.f.shape
        self.tmp = [np.empty(shp) for _ in range(7)]
        self.k = [(np.empty(shp), np.empty((3,) + shp)) for _ in range(4)]
        self.stage = (np.empty(shp), np.empty((3,) + shp))

    def lap(self, phi, out):
        if self.lap_kind == "metric":
            _lap_metric(phi, self.L11, self.L22, self.L12, self.isg_lap, out)
        else:
            _lap_flat(phi, self.flat_scale, out)

    def _hv(self, Dp, v, dD, dv, coef):

        n = self.hv_order // 2
        sgn = 1.0 if n % 2 == 0 else -1.0
        t1, t2 = self.tmp[5], self.tmp[6]
        if np.ndim(coef) > 0:

            if n != 2:
                raise ValueError("a filter strength varying in space is implemented for del^4")
            cf = np.ascontiguousarray(coef)
            def apply_var(phi):
                self.lap(phi, t1)
                np.multiply(t1, cf, out=t1)
                self.lap(t1, t2)
                return t2
            dD += apply_var(Dp)
            for kk in range(3):
                dv[kk] += apply_var(np.ascontiguousarray(v[kk]))
            return
        def apply(phi):
            src = phi
            for k in range(n):
                dst = t1 if k % 2 == 0 else t2
                self.lap(src, dst); src = dst
            return src
        dD += (sgn * coef) * apply(Dp)
        for kk in range(3):
            dv[kk] += (sgn * coef) * apply(np.ascontiguousarray(v[kk]))

    def rhs(self, Dp, v, dD, dv):
        K, vA1, vA2, F1, F2 = self.tmp[:5]
        _pass1(Dp, v, self.hs, self.A1, self.A2, self.a1, self.a2, self.sg, K, vA1, vA2, F1, F2)
        _pass2(v, self.X, self.a1, self.a2, self.isg, self.f, K, vA1, vA2, F1, F2, self.order, dD, dv)
        if self.cone is not None:
            C = self.cone
            dD -= C.div(Dp * v)
            dv -= C.curl(v) * np.cross(self.X, v, axis=0) + C.grad(K)
            _project(dv, self.X)
        if np.any(self.nu4) and self.hv_mode == "stage":
            self._hv(Dp, v, dD, dv, -self.nu4)
            _project(dv, self.X)

    def step(self, Dp, v, dt):
        k = self.k
        Ds, vs = self.stage
        if self.integrator == "rk4":
            self.rhs(Dp, v, *k[0])
            _axpy_state(Dp, v, 0.5 * dt, *k[0], Ds, vs); self.rhs(Ds, vs, *k[1])
            _axpy_state(Dp, v, 0.5 * dt, *k[1], Ds, vs); self.rhs(Ds, vs, *k[2])
            _axpy_state(Dp, v, dt, *k[2], Ds, vs); self.rhs(Ds, vs, *k[3])
            Dn = Dp + dt / 6 * (k[0][0] + 2 * k[1][0] + 2 * k[2][0] + k[3][0])
            vn = v + dt / 6 * (k[0][1] + 2 * k[1][1] + 2 * k[2][1] + k[3][1])
        elif self.integrator.startswith("kg"):
            from .tfd import kg_coeffs
            Dn, vn = Dp, v
            for ck in kg_coeffs(self.integrator[2:]):
                self.rhs(Dn, vn, *k[0])
                Dn = Dp + ck * dt * k[0][0]
                vn = v + ck * dt * k[0][1]
        elif self.integrator.startswith("ls"):
            from .tfd import ls_coeffs
            A, B = ls_coeffs(self.integrator[2:])
            Dn, vn = Dp.copy(), v.copy()
            qD, qv = k[1]
            qD[...] = 0.0; qv[...] = 0.0
            for Ak, Bk in zip(A, B):
                self.rhs(Dn, vn, *k[0])
                qD *= Ak; qD += dt * k[0][0]
                qv *= Ak; qv += dt * k[0][1]
                Dn += Bk * qD; vn += Bk * qv
        elif self.integrator == "ssprk3":
            self.rhs(Dp, v, *k[0])
            D1, v1 = Dp + dt * k[0][0], v + dt * k[0][1]
            self.rhs(D1, v1, *k[1])
            D2 = 0.75 * Dp + 0.25 * (D1 + dt * k[1][0]); v2 = 0.75 * v + 0.25 * (v1 + dt * k[1][1])
            self.rhs(D2, v2, *k[2])
            Dn = Dp / 3 + 2 / 3 * (D2 + dt * k[2][0]); vn = v / 3 + 2 / 3 * (v2 + dt * k[2][1])
        else:
            raise ValueError(self.integrator)
        _project(vn, self.X)
        if np.any(self.nu4) and self.hv_mode == "step":
            dD = np.zeros_like(Dn); dv = np.zeros_like(vn)
            self._hv(Dn, vn, dD, dv, -dt * self.nu4)
            Dn += dD; vn += dv
            _project(vn, self.X)
        return Dn, vn

    def mass(self, Dp):
        return self.S.mass(Dp)
