"""Count arithmetic and dependency reach for the solver kernels."""


import re
import numpy as np


class C:
    n = 0
    def __init__(self, v):
        self.v = float(v.v) if isinstance(v, C) else float(v)
    @staticmethod
    def _w(x):
        return x if isinstance(x, C) else C(x)
    def __add__(self, o): C.n += 1; return C(self.v + C._w(o).v)
    __radd__ = __add__
    def __sub__(self, o): C.n += 1; return C(self.v - C._w(o).v)
    def __rsub__(self, o): C.n += 1; return C(C._w(o).v - self.v)
    def __mul__(self, o): C.n += 1; return C(self.v * C._w(o).v)
    __rmul__ = __mul__
    def __truediv__(self, o): C.n += 1; return C(self.v / C._w(o).v)
    def __rtruediv__(self, o): C.n += 1; return C(C._w(o).v / self.v)
    def __neg__(self): return C(-self.v)


def sqrt(x):
    C.n += 1
    return C(np.sqrt(x.v))


def vec(*a):
    return [C(x) for x in a]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


ORDER = [4]


def d4(fm2, fm1, fp1, fp2):

    if ORDER[0] == 2:
        return 0.5 * (fp1 - fm1)
    if ORDER[0] == 6:
        r = np.random.default_rng(9)
        fm3, fp3 = C(r.normal()), C(r.normal())
        return (45 * (fp1 - fm1) - 9 * (fp2 - fm2) + (fp3 - fm3)) * (1.0 / 60)
    if ORDER[0] == 8:
        r = np.random.default_rng(9)
        fm3, fp3, fm4, fp4 = C(r.normal()), C(r.normal()), C(r.normal()), C(r.normal())
        return (672 * (fp1 - fm1) - 168 * (fp2 - fm2) + 32 * (fp3 - fm3) - 3 * (fp4 - fm4)) * (1.0 / 840)
    return (8 * (fp1 - fm1) - (fp2 - fm2)) * (1.0 / 12)


def count(fn):
    C.n = 0
    fn()
    return C.n


def rhs_point(order=4, with_hv=False):
    r = np.random.default_rng(0)
    R = lambda k=1: [C(x) for x in r.normal(size=k)]

    X, A1, A2, a1, a2 = R(3), R(3), R(3), R(3), R(3)
    isg, sg, f, hs = R(1)[0], R(1)[0], R(1)[0], R(1)[0]
    D = lambda: d4(*R(4))
    def body():
        Dp = R(1)[0]; v = R(3)
        K = 9.80616 * (Dp + hs) + 0.5 * dot(v, v)


        zeta = (D() - D()) * isg

        dKi, dKj = D(), D()
        gK = [a1[k] * dKi + a2[k] * dKj for k in range(3)]

        div = (D() + D()) * isg
        fz = f + zeta
        Xv = cross(X, v)
        dv = [-(fz * Xv[k]) - gK[k] for k in range(3)]
        dD = -div

        s = dot(dv, X)
        dv = [dv[k] - s * X[k] for k in range(3)]

        vA1, vA2 = dot(v, A1), dot(v, A2)
        Dv = [Dp * v[k] for k in range(3)]
        F1, F2 = sg * dot(Dv, a1), sg * dot(Dv, a2)
        if with_hv:
            hv_point_inline()
        return dD, dv
    return body


def lap_point():

    r = np.random.default_rng(1)
    R = lambda: C(r.normal())
    L11, L11p, L11m, L22, L22p, L22m = R(), R(), R(), R(), R(), R()
    L12 = R(); isg = R()
    phi, pip, pim, pjp, pjm = R(), R(), R(), R(), R()
    a_ip = 0.5 * (L11 + L11p); a_im = 0.5 * (L11 + L11m)
    b_jp = 0.5 * (L22 + L22p); b_jm = 0.5 * (L22 + L22m)
    out = a_ip * (pip - phi) - a_im * (phi - pim) + b_jp * (pjp - phi) - b_jm * (phi - pjm)

    dj = 0.5 * (pjp - pjm); di = 0.5 * (pip - pim)
    t1 = L12 * dj; t2 = L12 * di
    out = out + 0.5 * (R() - R()) + 0.5 * (R() - R())
    return out * isg


def hv_point_inline():
    for _ in range(4):
        lap_point(); lap_point()


def metric_point():
    r = np.random.default_rng(2)
    R = lambda k=1: [C(x) for x in r.normal(size=k)]
    X = R(3)
    A1 = [d4(*R(4)) * 6.37122e6 for _ in range(3)]
    A2 = [d4(*R(4)) * 6.37122e6 for _ in range(3)]
    s1, s2 = dot(A1, X), dot(A2, X)
    A1 = [A1[k] - s1 * X[k] for k in range(3)]
    A2 = [A2[k] - s2 * X[k] for k in range(3)]
    g11, g12, g22 = dot(A1, A1), dot(A1, A2), dot(A2, A2)
    det = g11 * g22 - g12 * g12
    idet = 1.0 / det
    a1 = [(g22 * A1[k] - g12 * A2[k]) * idet for k in range(3)]
    a2 = [(g11 * A2[k] - g12 * A1[k]) * idet for k in range(3)]
    sg = dot(cross(A1, A2), X)
    isg = 1.0 / sg
    f = 2 * 7.292e-5 * X[2]

    B1 = [0.5 * (R()[0] - R()[0]) * 6.37122e6 for _ in range(3)]
    B2 = [0.5 * (R()[0] - R()[0]) * 6.37122e6 for _ in range(3)]
    t1, t2 = dot(B1, X), dot(B2, X)
    B1 = [B1[k] - t1 * X[k] for k in range(3)]
    B2 = [B2[k] - t2 * X[k] for k in range(3)]
    h11, h12, h22 = dot(B1, B1), dot(B1, B2), dot(B2, B2)
    hdet = h11 * h22 - h12 * h12
    sgl = sqrt(hdet); ih = 1.0 / hdet
    L11, L22, L12 = sgl * h22 * ih, sgl * h11 * ih, -(sgl * h12 * ih)


def rk4_combine_point():
    r = np.random.default_rng(3)
    R = lambda: C(r.normal())
    for _ in range(4):
        y, k1, k2, k3, k4, dt = R(), R(), R(), R(), R(), R()

        s1 = y + 0.5 * dt * k1; s2 = y + 0.5 * dt * k2; s3 = y + dt * k3
        yn = y + dt * (1.0 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    X = [R(), R(), R()]; v = [R(), R(), R()]
    s = dot(v, X); v = [v[k] - s * X[k] for k in range(3)]


def lap_flat_point():
    r = np.random.default_rng(4)
    R = lambda: C(r.normal())
    nbs = [R() for _ in range(6)]
    p, scale = R(), R()
    acc = nbs[0]
    for x in nbs[1:]:
        acc = acc + x
    return scale * (acc - 6.0 * p)


def ssprk3_combine_point():
    r = np.random.default_rng(5)
    R = lambda: C(r.normal())
    for _ in range(4):
        y, k1, k2, k3, dt = R(), R(), R(), R(), R()
        y1 = y + dt * k1
        y2 = 0.75 * y + 0.25 * (y1 + dt * k2)
        yn = (1.0 / 3) * y + (2.0 / 3) * (y2 + dt * k3)
    X = [R(), R(), R()]; v = [R(), R(), R()]
    s = dot(v, X); v = [v[k] - s * X[k] for k in range(3)]


def kg_combine_point(s):
    r = np.random.default_rng(6)
    R = lambda: C(r.normal())
    for _ in range(s):
        for _ in range(4):
            y, k, cdt = R(), R(), R()
            yn = y + cdt * k
    X = [R(), R(), R()]; v = [R(), R(), R()]
    sp = dot(v, X); v = [v[k] - sp * X[k] for k in range(3)]


def ls_combine_point(s):
    r = np.random.default_rng(8)
    R = lambda: C(r.normal())
    for _ in range(s):
        for _ in range(4):
            q, k, y, A_, B_, dt = R(), R(), R(), R(), R(), R()
            q = A_ * q + dt * k
            y = y + B_ * q
    X = [R(), R(), R()]; v = [R(), R(), R()]
    sp = dot(v, X); v = [v[k] - sp * X[k] for k in range(3)]


def step_cost(order=4, integrator="rk4", hv_mode="stage", lap_kind="metric", metric="stored"):
    ORDER[0] = order
    rhs = count(rhs_point(with_hv=False))
    ORDER[0] = 4
    lap = count(lap_point) if lap_kind == "metric" else count(lap_flat_point)
    hv = 8 * lap
    if integrator.startswith("kg"):
        stages = int(re.match(r"\d+", integrator[2:]).group())
        comb = count(lambda: kg_combine_point(stages))
    elif integrator.startswith("ls"):
        stages = int(re.match(r"\d+", integrator[2:]).group())
        comb = count(lambda: ls_combine_point(stages))
    else:
        stages = 4 if integrator == "rk4" else 3
        comb = count(rk4_combine_point) if integrator == "rk4" else count(ssprk3_combine_point)
    flops = stages * rhs + comb + (stages * hv if hv_mode == "stage" else hv + 11)
    ORDER[0] = order
    met = count(metric_point)
    ORDER[0] = 4
    if metric == "recomputed":
        flops += met
    r_stage = order // 2
    lap_reach = 2
    reach = stages * max(r_stage, lap_reach if hv_mode == "stage" else 0)
    if hv_mode == "step":
        reach += lap_reach
    return dict(flops=flops, reach=reach, rhs=rhs, hv=hv, combine=comb, metric_flops=met)


def summary():
    rhs = count(rhs_point(with_hv=False))
    rhs_hv = count(rhs_point(with_hv=True))
    lap = count(lap_point)
    met = count(metric_point)
    comb = count(rk4_combine_point)
    return dict(rhs=rhs, rhs_with_hv=rhs_hv, laplacian=lap, hv_per_eval=8 * lap, metric=met,
                rk4_combine=comb,
                step_hv_in_stages=4 * rhs_hv + comb,
                step_hv_once=4 * rhs + comb + 8 * lap,
                step_hv_once_metric_recomputed=4 * rhs + comb + 8 * lap + met)


if __name__ == "__main__":
    for k, v in summary().items():
        print(f"{k:32s} {v}")


def lwt_point():


    r = np.random.default_rng(7)
    R = lambda k=1: [C(x) for x in r.normal(size=k)]
    def cell(with_avg):
        corners = [R(3) for _ in range(4)]
        n = R(3)
        D = R(4); V = [R(3) for _ in range(4)]; K = []
        for i in range(4):
            K.append(9.8 * (D[i] + R()[0]) + 0.5 * dot(V[i], V[i]))
        area = dot(cross([corners[2][k] - corners[0][k] for k in range(3)],
                         [corners[3][k] - corners[1][k] for k in range(3)]), n) * 0.5
        ia = 1.0 / area
        g = [C(0), C(0), C(0)]; curl = C(0); div = C(0)
        for e in range(4):
            p, q = e, (e + 1) % 4
            d = [corners[q][k] - corners[p][k] for k in range(3)]
            mid = [corners[q][k] + corners[p][k] for k in range(3)]
            ne = cross(d, mid)
            inv = 1.0 / sqrt(dot(mid, mid))
            ne = [x * inv for x in ne]
            Ke = 0.5 * (K[p] + K[q])
            g = [g[k] + Ke * ne[k] for k in range(3)]
            ve = [0.5 * (V[p][k] + V[q][k]) for k in range(3)]
            curl = curl + dot(ve, d)
            fe = [0.5 * (D[p] * V[p][k] + D[q] * V[q][k]) for k in range(3)]
            div = div + dot(fe, ne)
        g = [x * ia for x in g]; curl = curl * ia; div = div * ia
        vbar = [0.25 * (V[0][k] + V[1][k] + V[2][k] + V[3][k]) for k in range(3)]
        s = dot(vbar, n); vbar = [vbar[k] - s * n[k] for k in range(3)]
        Dbar = 0.25 * (D[0] + D[1] + D[2] + D[3])
        hdt = R()[0]; f = R()[0]
        Dn = Dbar - hdt * div
        fz = f + curl; nv = cross(n, vbar)
        vn = [vbar[k] - hdt * (fz * nv[k] + g[k]) for k in range(3)]
        s = dot(vn, n); vn = [vn[k] - s * n[k] for k in range(3)]
    cell(True)
    cell(False)
    for _ in range(4):
        lap_flat_point(); lap_flat_point()


def lwt_cost():
    return dict(flops=count(lwt_point), reach=3)


def cone_point(n_entries):


    def body():
        r = np.random.default_rng(7)
        R = lambda: C(r.normal())

        g = [C(0.0)] * 3; dv = C(0.0); z = C(0.0)
        for j in range(n_entries):
            K = R(); F = [R(), R(), R()]; v = [R(), R(), R()]
            Eg = [R(), R(), R()]; Ed = [R(), R(), R()]; Ec = [R(), R(), R()]
            g = [g[k] + Eg[k] * K for k in range(3)]
            dv = dv + dot(Ed, F)
            z = z + dot(Ec, v)

        X = [R(), R(), R()]; v = [R(), R(), R()]; f = R()
        a = f + z
        xv = cross(X, v)
        out = [-(a * xv[k]) - g[k] for k in range(3)]
        s = dot(out, X); out = [out[k] - s * X[k] for k in range(3)]
        dD = -dv
    return body


def cone_patch_cost(order=8, n_entries=61, integrator="ls10o4"):
    interior = step_cost(order=order, integrator=integrator, hv_mode="step", metric="recomputed")
    stages = int(re.match(r"\d+", integrator[2:]).group()) if integrator[:2] in ("kg", "ls") else 4
    ORDER[0] = order
    rhs_int = count(rhs_point(with_hv=False))
    ORDER[0] = 4
    rhs_cone = count(cone_point(n_entries))
    patch = interior["flops"] - interior["metric_flops"] + stages * (rhs_cone - rhs_int)
    return dict(interior=interior["flops"], patch=patch, ratio=patch / interior["flops"],
                rhs_interior=rhs_int, rhs_cone=rhs_cone)
