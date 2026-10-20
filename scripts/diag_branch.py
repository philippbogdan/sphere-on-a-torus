"""Run a numerical check and save its measurements."""


import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, tfd, lsops, testcases as tc, torusmap, ellip

VERTS = np.array([[0, 0, 1.0]] + [[np.cos(np.radians(-19.4712)) * np.cos(np.radians(l)),
                                    np.cos(np.radians(-19.4712)) * np.sin(np.radians(l)),
                                    np.sin(np.radians(-19.4712))] for l in (0, 120, 240)])
BANDS = [(0, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 180)]


def bdist(X):
    return np.degrees(np.arccos(np.clip(np.tensordot(VERTS, X, axes=1), -1, 1))).min(0)


def truncation():
    out = []
    for order in (2, 4, 6):
        for M in (64, 128, 256):
            X = grid.torus_X(M); case = tc.TC2(0.7)
            S = tfd.TFD(X, case, order=order)
            dD, dv = S.rhs(case.height(X) - S.hs, case.velocity(X))
            ang = bdist(X)
            out.append(dict(order=order, M=M, max_abs_dDdt_by_band={f"{lo}-{hi}": float(np.abs(dD[(ang >= lo) & (ang < hi)]).max()) for lo, hi in BANDS}))
    return out


B = np.array([[0.3, -1.0, 0.2], [0.7, 0.1, -0.5], [0.4, 0.9, -0.2]])


def exact(X):
    x, y, z = X
    phi = x * y + z * z + 0.5 * x
    g3 = np.stack([y + 0.5, x, 2 * z]); grad = g3 - (g3 * X).sum(0) * X
    BX = np.tensordot(B, X, axes=1); F = BX - (BX * X).sum(0) * X
    XBX = (X * BX).sum(0); div = np.trace(B) - 3 * XBX
    Wv = np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]]); curl = np.tensordot(Wv, X, axes=1)
    return phi, grad, F, div, curl


def operators():
    out = []
    bands = [(0, 3), (3, 6), (6, 12), (12, 180)]
    for M in (64, 128, 256):
        X = grid.torus_X(M); ang = bdist(X)
        phi, grad, F, div, curl = exact(X)
        m = tfd.Metric(X, 4, a=1.0)
        ed = np.abs(m.div(F, 4) - div); ec = np.abs(m.curl(F, 4) - curl)
        eds = np.abs(m.div_split(F, 4) - div); ecs = np.abs(m.curl_split(F, 4) - curl)
        edc = np.abs(m.div_cart(F, 4) - div); ecc = np.abs(m.curl_cart(F, 4) - curl)
        offs, idx, W = lsops.ls_weights(X, np.ones(X.shape[1:], bool), m=2, q=3)
        dls = np.zeros(len(idx)); cls = np.zeros(len(idx))
        for k, (di, dj) in enumerate(offs):
            ii, jj = (idx[:, 0] + di) % M, (idx[:, 1] + dj) % M
            Fq = F[:, ii, jj]
            dls += (W[:, :, k] * Fq).sum(0)
            cls += (np.cross(W[:, :, k], Fq, axis=0) * X[:, idx[:, 0], idx[:, 1]]).sum(0)
        edl = np.abs(dls - div[idx[:, 0], idx[:, 1]]).reshape(M, M); ecl = np.abs(cls - curl[idx[:, 0], idx[:, 1]]).reshape(M, M)
        row = dict(M=M)
        for name, e in (("div_flux", ed), ("curl_flux", ec), ("div_split", eds), ("curl_split", ecs), ("div_cart", edc), ("curl_cart", ecc), ("div_ls3", edl), ("curl_ls3", ecl)):
            row[name] = {f"{lo}-{hi}": float(e[(ang >= lo) & (ang < hi)].max()) for lo, hi in bands}
        out.append(row)
    return out


def map_quality(M=256):
    i = np.arange(M); I, J = np.meshgrid(i, i, indexing="ij")
    zeta = ((I + 0.5) + (J + 0.5) * ellip.EPS) / M
    def q(X):
        h = 1.0 / M
        A1 = (np.roll(X, -1, 1) - np.roll(X, 1, 1)) / 2; A2 = (np.roll(X, -1, 2) - np.roll(X, 1, 2)) / 2
        g11, g22, g12 = (A1 * A1).sum(0), (A2 * A2).sum(0), (A1 * A2).sum(0)
        E = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]]) * h; Ei = np.linalg.inv(E)
        Gm = np.stack([np.stack([g11, g12]), np.stack([g12, g22])]); Gf = np.einsum('ai,abxy,bj->ijxy', Ei, Gm, Ei)
        tr = Gf[0, 0] + Gf[1, 1]; det = Gf[0, 0] * Gf[1, 1] - Gf[0, 1] ** 2; s = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        l1 = np.sqrt(tr / 2 + s); l2 = np.sqrt(np.maximum(tr / 2 - s, 0)); mm = np.sqrt((l1 * l2).mean())
        return dict(sigma_min_over_mean=float(l2.min() / mm), sigma_max_over_mean=float(l1.max() / mm))
    res = {"conformal": q(torusmap.sphere_point(zeta))}
    best = None
    for k in (2, 3, 4):
        for r0 in (0.15, 0.2, 0.25):
            for a in (0.05, 0.1, 0.15, 0.2, 0.3):
                v = q(torusmap.sphere_point(zeta, a=a, r0=r0, k=k))
                if best is None or v["sigma_min_over_mean"] > best[0]["sigma_min_over_mean"]:
                    best = (v, dict(a=a, r0=r0, k=k))
    res["best_polynomial_radial_correction"] = dict(best[0], params=best[1])
    res["equal_area_moser"] = q(grid.torus_X(M))
    return res


if __name__ == "__main__":
    out = dict(truncation=truncation(), operators=operators(), maps=map_quality())
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out["maps"]))
