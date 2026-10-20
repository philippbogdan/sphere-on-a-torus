"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, tfd, testcases as tc, lwgg, configs, experiments as ex

VERTS = np.array([[0, 0, 1.0]] + [[math.cos(math.radians(-19.4712206)) * math.cos(math.radians(l)),
                                    math.cos(math.radians(-19.4712206)) * math.sin(math.radians(l)),
                                    math.sin(math.radians(-19.4712206))] for l in (0, 120, 240)])


def run(M, scheme, mode):
    sc = configs.parse(scheme)
    X = grid.torus_X(M); case = tc.TC2(0.7)
    S, S0, h, cmax, dt0 = ex.build(X, case, scheme, fast=False)
    S.nu4 = float(S.nu4) if np.ndim(S.nu4) == 0 else S.nu4
    D0 = case.height(X) - S.hs; v0 = case.velocity(X)
    R = {"none": -1, "fixed8": 8, "scaled": M // 16}[mode]
    mask = tfd.branch_mask(M, R) if R >= 0 else np.zeros((M, M), bool)
    tauD, tauv = S.rhs(D0, v0)
    tauD, tauv = tauD * mask, tauv * mask


    filt = S.filtered
    hD0 = filt(D0) * mask if np.any(S.nu4) else 0.0
    hv0 = np.stack([filt(v0[q]) for q in range(3)]) * mask if np.any(S.nu4) else 0.0
    base = S.rhs
    S.rhs = lambda Dp, v: (base(Dp, v)[0] - tauD, base(Dp, v)[1] - tauv)
    def step(Dp, v, dt):
        Dn, vn = S._step(Dp, v, dt)
        if np.any(S.nu4):
            Dn = Dn - dt * (filt(Dn) - hD0)
            vn = vn - dt * (np.stack([filt(vn[q]) for q in range(3)]) - hv0)
        return Dn, vn - (vn * S.X).sum(0) * S.X
    D, v = D0.copy(), v0.copy()
    dt = sc["courant"] * h / cmax; T = 5 * tc.DAY; n = int(np.ceil(T / dt)); dt = T / n
    for k in range(n):
        D, v = step(D, v, dt)
    e = D + S.hs - case.height(X); w = S.weight; href = case.height(X)
    ang = np.degrees(np.arccos(np.clip(np.tensordot(VERTS, X, axes=1), -1, 1))).min(0)
    far = ang >= 40
    return dict(M=M, scheme=scheme, mode=mode, R_cells=R, masked_points=int(mask.sum()),
                l2=float(lwgg.norms(D + S.hs, href, w)[1]), linf=float(np.abs(e).max() / np.abs(href).max()),
                l2_far=float(math.sqrt((w[far] * e[far] ** 2).sum() / (w[far] * href[far] ** 2).sum())))


if __name__ == "__main__":
    out = sys.argv[1]
    for scheme in (configs.MAIN, configs.MAIN + ",kappa=0", configs.PREV_MAIN):
        for mode in ("none", "fixed8", "scaled"):
            for M in (64, 128, 256):
                r = run(M, scheme, mode)
                print(json.dumps(r), flush=True)
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
