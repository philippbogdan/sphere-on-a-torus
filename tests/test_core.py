"""Check geometry, conservation, time integration and the compiled solver."""
import json, os, math
import numpy as np
import pytest

from polarswe import ellip, moser, grid, tfd, fast, testcases as tc, flopcount, dataflow, lsops, conehp, experiments, configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_weierstrass_identities():
    e = ellip.roots()
    assert abs(e.sum()) < 1e-12
    assert abs(e[0] * e[1] + e[1] * e[2] + e[2] * e[0]) < 1e-10
    z = np.array([0.13 + 0.21j, 0.31 + 0.05j, 0.4 + 0.6j])
    assert np.abs(ellip.wp(z + 1) - ellip.wp(z)).max() < 1e-10
    assert np.abs(ellip.wp(z + ellip.EPS) - ellip.wp(z)).max() < 1e-10
    assert np.abs(ellip.wp(-z) - ellip.wp(z)).max() < 1e-12
    p, dp = ellip.wp_and_derivative(z)
    g3 = 4 * np.prod(e)
    assert np.abs(dp ** 2 - (4 * p ** 3 - g3)).max() / np.abs(dp ** 2).max() < 1e-10


def test_branch_points_are_tetrahedron():
    M = 32
    Xc = grid.torus_X_offset(M, 1.0)
    V = np.array([Xc[:, i, j] for i, j in grid.branch_cells(M)])
    G = V @ V.T
    off = G[~np.eye(4, dtype=bool)]
    assert np.allclose(off, -1.0 / 3.0, atol=1e-9)


def test_equal_area_and_bounded_cells():
    M = 64
    X = grid.torus_X(M)

    I, J = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    assert np.abs(X - X[:, (-1 - I) % M, (-1 - J) % M]).max() < 1e-12
    S = tfd.TFD(X, tc.TC2(0.0))
    w = S.weight
    assert abs(w.sum() / (8 * np.pi * tc.A ** 2) - 1) < 2e-3
    chords = [np.linalg.norm(np.roll(np.roll(X, -a, 1), -b, 2) - X, axis=0) for a, b in ((1, 0), (0, 1), (1, -1))]
    c = np.concatenate([x.ravel() for x in chords]); c = c[c > 1e-12]
    assert c.min() / c.mean() > 0.6


def test_fd_order_away_from_branch_points():
    errs = []
    for M in (64, 128):
        X = grid.torus_X(M)
        case = tc.TC2(0.7)
        S = tfd.TFD(X, case, order=4)
        dD, dv = S.rhs(case.height(X) - S.hs, case.velocity(X))
        verts = np.array([[0, 0, 1.0]] + [[math.cos(math.radians(-19.4712)) * math.cos(math.radians(l)),
                                            math.cos(math.radians(-19.4712)) * math.sin(math.radians(l)),
                                            math.sin(math.radians(-19.4712))] for l in (0, 120, 240)])
        ang = np.degrees(np.arccos(np.clip(np.tensordot(verts, X, axes=1), -1, 1))).min(0)
        errs.append(np.abs(dD[ang > 48]).max())
    assert errs[0] / errs[1] > 12


@pytest.mark.parametrize("integ,order", [("rk4", 4), ("kg9", 4), ("kg9", 6), ("kg9", 8), ("kg10o4", 6), ("ls10o4", 8), ("ssprk3", 2)])
def test_compiled_equals_reference_and_conserves_mass(integ, order):
    X = grid.torus_X(32)
    case = tc.TC5()
    S = tfd.TFD(X, case, nu4=1e17, integrator=integ, hv_mode="step", order=order)
    F = fast.FastTFD(tfd.TFD(X, case, nu4=1e17, integrator=integ, hv_mode="step", order=order))
    D0 = case.height(X) - S.hs; v0 = case.velocity(X)
    Da, va = D0.copy(), v0.copy(); Db, vb = np.ascontiguousarray(D0), np.ascontiguousarray(v0)
    m0 = S.mass(Da)
    for _ in range(10):
        Da, va = S.step(Da, va, 2000.0)
        Db, vb = F.step(Db, vb, 2000.0)
    assert np.abs(Da - Db).max() < 1e-8
    assert abs(S.mass(Da) - m0) / m0 < 1e-13


def test_flop_counts():
    assert flopcount.step_cost(hv_mode="step")["flops"] == 758
    c = flopcount.step_cost(integrator="kg9", hv_mode="step")
    assert c["flops"] == 1286 and c["reach"] == 20


def test_dataflow_matches_paper():

    k = dataflow.paper_kernel("swe")
    util = 155 * 24 * 24 / (2 * k.cycles(24))
    assert abs(util - 0.53) < 0.005

    he = dataflow.paper_kernel("he5"); m = dataflow.Machine(cores_w=720, cores_h=720)
    assert abs(dataflow.step_time(he, 32, m)["rate"] / 1.05e5 - 1) < 0.02
    assert abs(dataflow.step_time(he, 64, m)["rate"] / 2.7e4 - 1) < 0.02


def test_stability_interval():
    tab = json.load(open(os.path.join(ROOT, "polarswe/kg_coeffs.json")))
    a = np.array(tab["9"]["a"])
    beta = tab["9"]["beta"]
    assert beta > 7.999
    y = np.linspace(0, beta, 4000)
    assert np.abs(np.polyval(a[::-1], 1j * y)).max() < 1 + 1e-9
    assert abs(tab["9"]["c"][-1] - 1) < 1e-12 and abs(tab["9"]["c"][-2] - 0.5) < 1e-6


def test_least_squares_weights_exact_for_quadratics():
    X = grid.torus_X(64)
    offs, idx, W = lsops.ls_weights(X, np.ones(X.shape[1:], bool), m=2, q=3)
    f = X[0] * X[1] + X[2]
    g3 = np.stack([X[1], X[0], np.ones_like(X[2])]); grad = g3 - (g3 * X).sum(0) * X
    gl = sum(W[:, :, k] * f[(idx[:, 0] + di) % 64, (idx[:, 1] + dj) % 64] for k, (di, dj) in enumerate(offs))
    err = np.linalg.norm(gl - grad[:, idx[:, 0], idx[:, 1]], axis=0)
    assert err.max() < 5e-4


def test_kg10o4_fourth_order_on_linear_problems():
    from polarswe import tfd
    c = tfd.kg_coeffs("10o4")
    def run(dt, T=10.0, w=1.0):
        y = 1.0 + 0j
        for _ in range(int(round(T / dt))):
            ys = y
            for ck in c:
                ys = y + ck * dt * (1j * w * ys)
            y = ys
        return abs(y - np.exp(1j * w * T))
    e1, e2 = run(0.5), run(0.25)
    assert 12 < e1 / e2 < 20


def test_trsk_reconstruction_properties():

    from polarswe import trsk
    for P in (trsk.torus_mesh(16)[0], trsk.icosahedral(2)):
        m = trsk.VoronoiMesh(P)
        assert abs(m.areaCell.sum() / (4 * np.pi) - 1) < 1e-12
        assert abs(m.W + m.W.T).max() < 1e-12
        psi = np.random.default_rng(1).standard_normal(m.V)
        U = psi[m.verticesOnEdge[:, 0]] - psi[m.verticesOnEdge[:, 1]]
        voc = m.verticesOnCell
        pc = np.where(voc >= 0, m.Rkite * psi[np.where(voc >= 0, voc, 0)], 0).sum(1)
        assert np.abs(m.W @ U - (pc[m.cellsOnEdge[:, 1]] - pc[m.cellsOnEdge[:, 0]])).max() < 1e-12


def test_trsk_conserves_mass_and_reach_is_fixed():
    from polarswe import trsk, testcases as tc
    P, _ = trsk.torus_mesh(32)
    sol = trsk.Solver(trsk.VoronoiMesh(P), tc.TC5(), "kg10o4")
    h, u = sol.S.initial()
    m0 = (sol.S.A * h).sum()
    for _ in range(20):
        h, u = sol.step(h, u, 600.0)
    assert abs((sol.S.A * h).sum() / m0 - 1) < 1e-13
    assert trsk.torus_layout(32, sol.S)["reach_stage"] == trsk.torus_layout(64)["reach_stage"]


def test_ls10o4_fourth_order_nonlinear():
    A, B = tfd.ls_coeffs("10o4")
    f = lambda u: -u * u + np.sin(u)
    def run(n):
        u = 1.0
        for k in range(n):
            q = 0.0
            for Ak, Bk in zip(A, B):
                q = Ak * q + (2.0 / n) * f(u); u = u + Bk * q
        return u
    ref = run(3200)
    e1, e2 = abs(run(40) - ref), abs(run(80) - ref)
    assert 3.8 < math.log2(e1 / e2) < 4.3


def test_orientations_and_branch_points():
    for o in ("vertex", "edge", "generic", "generic2"):
        R = grid.orientation(o)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        B = grid.branch_positions(o)
        G = B @ B.T
        assert np.allclose(G[~np.eye(4, dtype=bool)], -1.0 / 3.0, atol=1e-9)
    assert np.allclose(grid.branch_positions("vertex")[0], [0, 0, 1], atol=1e-12)


def test_varying_filter_conserves_mass():
    X = grid.torus_X(32); case = tc.TC5()
    sc = "order=8,integrator=ls10o4,courant=1.252,hv_mode=step,kappa=0.0001,kappa_near=0.002,rnear=20"
    S, S0, h, cmax, dt = experiments.build(X, case, sc)
    assert np.ndim(S.nu4) == 2
    D = np.ascontiguousarray(case.height(X) - S0.hs); v = np.ascontiguousarray(case.velocity(X)); m0 = S.mass(D)
    for _ in range(10):
        D, v = S.step(D, v, dt)
    assert abs(S.mass(D) - m0) / m0 < 1e-13














