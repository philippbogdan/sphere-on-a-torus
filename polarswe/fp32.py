"""Compare plain FP32 and perturbation-form FP32 with FP64."""


import numpy as np
from . import tfd, testcases as tc

F32 = np.float32


def _cast_solver(S, dt):
    m = S.m
    for name in ("X", "A1", "A2", "a1", "a2", "sqrtG", "L11", "L22", "L12", "lap_sg", "B1", "B2_", "sqrtG_s", "c1", "c2"):
        setattr(m, name, getattr(m, name).astype(dt))
    if getattr(m, "lap_w", None) is not None:
        m.lap_w = m.lap_w.astype(dt)
    for name in ("X", "f", "hs", "weight", "flat_scale", "mass_weight"):
        setattr(S, name, getattr(S, name).astype(dt))
    S.nu4 = S.nu4.astype(dt) if np.ndim(S.nu4) else dt(S.nu4)
    if getattr(S, "cone", None) is not None:
        S.cone = S.cone.cast(dt)
    return S


def full_ops(S):
    m, o, C = S.m, S.order, getattr(S, "cone", None)
    grad = (lambda f: m.grad(f, o) + C.grad(f)) if C is not None else (lambda f: m.grad(f, o))
    div = (lambda F: m.div(F, o) + C.div(F)) if C is not None else (lambda F: m.div(F, o))
    curl = (lambda v: m.curl(v, o) + C.curl(v)) if C is not None else (lambda v: m.curl(v, o))
    return grad, div, curl


class Plain32:
    def __init__(self, S):
        self.S = _cast_solver(S, F32)

    def init(self, D, v):
        return D.astype(F32), v.astype(F32)

    def step(self, D, v, dt):
        return self.S.step(D, v, F32(dt))

    def full(self, D, v):
        return D.astype(np.float64), v.astype(np.float64)

    def to_device(self):
        tfd.to_device(self.S)
        return self


class Reference32:

    def __init__(self, S, D_r, v_r):
        g = tc.G
        X = S.X
        grad, div, curl = full_ops(S)
        K_r = g * (D_r + S.hs) + 0.5 * tfd.dot(v_r, v_r)
        self.gK_r = grad(K_r).astype(F32)
        self.div_r = div(D_r * v_r).astype(F32)
        self.curl_r = curl(v_r).astype(F32)

        self.filt_D_r = S.filtered(D_r).astype(F32) if np.any(S.nu4) else None
        self.filt_v_r = np.stack([S.filtered(v_r[k]) for k in range(3)]).astype(F32) if np.any(S.nu4) else None
        self.D_r, self.v_r = D_r.astype(F32), v_r.astype(F32)
        self.S = _cast_solver(S, F32)
        self.ops = full_ops(self.S)

    def to_device(self):
        import cupy as cp
        for name in ("gK_r", "div_r", "curl_r", "filt_D_r", "filt_v_r", "D_r", "v_r"):
            if getattr(self, name) is not None:
                setattr(self, name, cp.asarray(getattr(self, name)))
        tfd.to_device(self.S)
        self.ops = full_ops(self.S)
        return self

    def init(self, D, v):
        return (D - self.D_r.astype(np.float64)).astype(F32), (v - self.v_r.astype(np.float64)).astype(F32)

    def full(self, Dp, vp):
        return Dp.astype(np.float64) + self.D_r.astype(np.float64), vp.astype(np.float64) + self.v_r.astype(np.float64)

    def rhs(self, Dp, vp):
        S = self.S
        grad, div, curl = self.ops
        X, vr, Dr = S.X, self.v_r, self.D_r
        Kp = F32(tc.G) * Dp + tfd.dot(vp, vr + F32(0.5) * vp)
        flux_p = Dr * vp + Dp * vr + Dp * vp
        zeta = self.curl_r + curl(vp)
        v = vr + vp
        dD = -(self.div_r + div(flux_p))
        dv = -(S.f + zeta) * tfd.cross(X, v) - (self.gK_r + grad(Kp))
        dv = dv - tfd.dot(dv, X) * X
        return dD, dv

    def step(self, Dp, vp, dt):
        S = self.S
        dt = F32(dt)
        if S.integrator.startswith("kg"):
            Dn, vn = Dp, vp
            for ck in tfd.kg_coeffs(S.integrator[2:]):
                dD, dv = self.rhs(Dn, vn)
                Dn, vn = Dp + F32(ck) * dt * dD, vp + F32(ck) * dt * dv
        elif S.integrator.startswith("ls"):
            A, B = tfd.ls_coeffs(S.integrator[2:])
            Dn, vn = Dp, vp
            qD, qv = 0 * Dp, 0 * vp
            for Ak, Bk in zip(A, B):
                dD, dv = self.rhs(Dn, vn)
                qD, qv = F32(Ak) * qD + dt * dD, F32(Ak) * qv + dt * dv
                Dn, vn = Dn + F32(Bk) * qD, vn + F32(Bk) * qv
        else:
            raise ValueError("reference form implemented for the KG and 2N-storage integrators")
        X = S.X
        vn = vn - tfd.dot(vr_full := (vn + self.v_r), X) * X
        if self.filt_D_r is not None:
            Dn = Dn - dt * (S.filtered(Dn) + self.filt_D_r)
            vn = vn - dt * (tfd.xp_of(vn).stack([S.filtered(vn[k]) for k in range(3)]) + self.filt_v_r)
            vn = vn - tfd.dot(vn + self.v_r, X) * X
        return Dn, vn
