"""Run the longer cone, precision and configuration checks explicitly."""
import json, os, math
import numpy as np
import pytest
from polarswe import ellip, moser, grid, tfd, fast, testcases as tc, flopcount, dataflow, lsops, conehp, experiments, configs

def test_near_cone_filter_confined_to_patches_and_conservative():

    M = 64
    X = grid.torus_X(M); case = tc.TC2(1.5)
    scheme = configs.FRONTIER[configs.SEPCURL_NAME][0]
    sc = experiments.scheme_of(scheme)
    S, S0, h, cmax, dt = experiments.build(X, case, scheme)
    nu = S.nu4
    assert np.ndim(nu) == 2
    d = experiments.branch_distance_cells(M)
    R = float(str(sc["cone"]).split("_")[0])
    far = nu[d > R]
    assert np.ptp(far) == 0 and nu[d <= R].min() >= far[0] and nu.max() > 1.9 * far[0]
    assert (nu > far[0]).sum() == (d <= R).sum() - ((nu == far[0]) & (d <= R)).sum()
    D = np.ascontiguousarray(case.height(X) - S0.hs); v = np.ascontiguousarray(case.velocity(X)); m0 = S.mass(D)
    for _ in range(10):
        D, v = S.step(D, v, dt)
    assert abs(S.mass(D) - m0) / m0 < 1e-13


def test_fp32_reference_form_with_varying_filter():
    from polarswe import fp32
    M = 64
    X = grid.torus_X(M); case = tc.TC2(0.7)
    scheme = configs.FRONTIER[configs.SEPCURL_NAME][0]
    S, S0, h, cmax, dt = experiments.build(X, case, scheme, fast=False)
    D0 = case.height(X) - S0.hs; v0 = case.velocity(X)
    W = fp32.Reference32(experiments.build(X, case, scheme, fast=False)[0], D0, v0)
    D, v = D0.copy(), v0.copy(); Dp, vp = W.init(D0, v0)
    for _ in range(5):
        D, v = S.step(D, v, dt); Dp, vp = W.step(Dp, vp, dt)
    Df, _ = W.full(Dp, vp)
    assert np.abs(Df - D).max() < 1e-3


def test_compatible_curl_and_reproducible_build():


    X = grid.torus_X(64)
    S = tfd.TFD(X, tc.TC2(0.7), order=8)
    C = conehp.ConeHP(S, R=8, p=3, rs=4, shape="lines", curl="compat", degw=10.0)
    C2 = conehp.ConeHP(S, R=8, p=3, rs=4, shape="lines", curl="compat", degw=10.0)
    m = S.m
    v = np.random.default_rng(3).standard_normal(X.shape); v = v - (v * X).sum(0) * X
    nv = tfd.cross(S.X, v)
    z = m.curl(v, 8) + C.curl(v)
    assert np.abs(z + m.div(nv, 8) + C.div(nv)).max() < 1e-12 * np.abs(z).max()
    assert all(abs(g - g2).max() == 0 for g, g2 in zip(C.Gc, C2.Gc)) and np.array_equal(C.weight, C2.weight)


    q = np.random.default_rng(4).standard_normal(X.shape[1:])
    lhs = m.curl(q * nv, 8) + C.curl(q * nv)
    rhs = m.div(q * v, 8) + C.div(q * v)
    assert np.abs(lhs - rhs).max() < 1e-12 * np.abs(rhs).max()


def test_cone_operators_adjoint_conservative_consistent():
    X = grid.torus_X(64)
    S = tfd.TFD(X, tc.TC2(0.7), order=8)
    C = conehp.ConeHP(S, R=6, p=2, rs=4, shape="lines")
    m, a = S.m, tc.A
    x, y, z = X
    B = np.array([[0.3, -1.1, 0.4], [0.7, 0.2, -0.5], [-0.2, 0.9, 0.1]])
    BX = np.tensordot(B, X, axes=1); F = BX - (BX * X).sum(0) * X
    div_ex = np.trace(B) - 3 * (X * BX).sum(0)
    G = lambda f: m.grad(f, 8) + C.grad(f); Dv = lambda Fv: m.div(Fv, 8) + C.div(Fv)
    psi = np.cos(3 * x + y) * z
    w = C.weight
    assert abs((w * (psi * Dv(F) + (F * G(psi)).sum(0))).sum()) / (w * np.abs(psi * Dv(F))).sum() < 1e-13
    assert abs((w * Dv(F)).sum()) / (w * np.abs(Dv(F))).sum() < 1e-13
    d = grid.branch_distance_deg(X)
    near = d < 6
    e_fd = np.abs(m.div(F, 8) * a - div_ex)[near].max(); e_c = np.abs(Dv(F) * a - div_ex)[near].max()
    assert e_c < 0.05 * e_fd
    with pytest.raises(ValueError):
        conehp.ConeHP(tfd.TFD(grid.torus_X(32), tc.TC2(0.7), order=8), R=8, p=3, rs=4)


def test_main_filter_off_in_cone_patches_and_conservative():

    M = 64
    X = grid.torus_X(M); case = tc.TC2(1.5)
    sc = experiments.scheme_of(configs.MAIN)
    S, S0, h, cmax, dt = experiments.build(X, case, configs.MAIN)
    nu = S.nu4
    d = experiments.branch_distance_cells(M)
    R = experiments.cone_radius(sc)
    assert np.ndim(nu) == 2 and np.all(nu[d <= R] == 0) and np.ptp(nu[d > R]) == 0 and nu[d > R][0] > 0
    Sr = experiments.build(X, case, configs.MAIN, fast=False)[0]
    f = np.random.default_rng(3).standard_normal((M, M))
    f = 0.5 * (f + f[::-1, ::-1][np.r_[-1, :M - 1]][:, np.r_[-1, :M - 1]])
    assert abs((Sr.mass_weight * Sr.filtered(f)).sum()) < 1e-12 * (Sr.mass_weight * np.abs(Sr.filtered(f))).sum()
    D = np.ascontiguousarray(case.height(X) - S0.hs); v = np.ascontiguousarray(case.velocity(X)); m0 = S.mass(D)
    for _ in range(10):
        D, v = S.step(D, v, dt)
    assert abs(S.mass(D) - m0) / m0 < 1e-13


def test_skew_tracer_form_conserves_mass_and_variance():

    M = 64
    X = grid.torus_X(M); case = tc.TC1(math.pi / 2)
    sc = experiments.scheme_of(configs.MAIN)
    S = tfd.TFD(X, case, order=sc["order"]); S.set_cone(experiments.cone_of(X, sc))
    m, o, C, w = S.m, sc["order"], S.cone, S.mass_weight
    v = case.velocity(X)
    rng = np.random.default_rng(5)
    v = v + 3.0 * np.cross(X, rng.standard_normal((3, 1, 1)), axis=0) * np.cos(2 * X[2])
    D = lambda F: m.div(F, o) + C.div(F); G = lambda f: m.grad(f, o) + C.grad(f)
    q = case.height(X) + 10 * np.sin(3 * X[0]) * X[1]
    Dv = D(v)
    r = -0.5 * (D(q * v) + (v * G(q)).sum(0) + q * Dv)
    scale = (w * np.abs(q * D(q * v))).sum()
    assert abs((w * r).sum()) < 1e-12 * (w * np.abs(D(q * v))).sum()
    assert abs((w * q * r).sum() + 0.5 * (w * q * q * Dv).sum()) < 1e-12 * scale


def test_every_pipeline_scheme_builds_and_steps():


    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import tc6_week2 as w2
    from polarswe import reproduce as R
    X = grid.torus_X(64)
    schemes = [sc for name, (sc, kind) in configs.FRONTIER.items() if kind not in ("trsk", "lwt") and "lattice=square" not in sc]
    schemes += [configs.MAIN + ("," + ov if ov else "") for _, ov in R.CONE_STAB]
    schemes += [configs.MAIN + extra for _, extra in w2.FILTERS]
    for sc in schemes:
        S, S0, h, cmax, dt = experiments.build(X, tc.TC6(), sc)
        D = np.ascontiguousarray(tc.TC6().height(X) - S0.hs); v = np.ascontiguousarray(tc.TC6().velocity(X))
        D, v = S.step(D, v, dt)
        assert np.isfinite(D).all(), sc
    for spec in R.TRACER_SPECS:
        g, n, case = spec.split(":")[:3]
        sch = spec.split(":", 3)[3]
        r = experiments.run(("torus", 64), case, days=0.3, scheme=sch)
        assert r["stable"], spec

