"""Check the cone algebra on a small patch before a long run."""
import numpy as np
from polarswe import grid, tfd, testcases as tc, experiments, configs


def test_cone_pair_and_steps():
    X = grid.torus_X(64)
    case = tc.TC2(0.7)
    S, S0, h, cmax, dt = experiments.build(X,case,configs.MAIN + ",cone=4_1_4_lines_cc_dw10",fast=False)
    C, m = S.cone, S.m
    rng = np.random.default_rng(9)
    q = rng.standard_normal((64,64))
    v = rng.standard_normal(X.shape)
    v -= (v*X).sum(0)*X
    grad = m.grad(q,8)+C.grad(q)
    div = m.div(v,8)+C.div(v)
    pairing = (C.weight*(q*div+(v*grad).sum(0))).sum()
    assert abs(pairing) < 1e-12*(C.weight*np.abs(q*div)).sum()
    curl = m.curl(v,8)+C.curl(v)
    rotated = tfd.cross(S.X,v)
    assert np.allclose(curl,-m.div(rotated,8)-C.div(rotated),rtol=1e-11,atol=1e-18)
    d = experiments.branch_distance_cells(64)
    assert np.all(S.nu4[d<=4] == 0)
    assert np.all(S.nu4[d>4] > 0)
    D = case.height(X)-S.hs
    v = case.velocity(X)
    mass = S.mass(D)
    for _ in range(5):
        D,v=S.step(D,v,dt)
    assert np.isfinite(D).all() and np.isfinite(v).all()
    assert abs(S.mass(D)/mass-1) < 1e-13
