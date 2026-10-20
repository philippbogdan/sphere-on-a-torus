"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from scipy.optimize import minimize_scalar
from polarswe import grid, tfd, fast, testcases as tc, reference as ref, lwgg, configs, experiments as ex

TRUTH_2H = os.path.join(ROOT, "refs/stswm/runs/T85_tc6_2h/grid_out.bin")


def solver(X, case):
    F, S0, h, cmax, dt = ex.build(X, case, configs.MAIN)
    D0 = case.height(X) - S0.hs; v0 = case.velocity(X)
    return F, S0, np.ascontiguousarray(D0), np.ascontiguousarray(v0), dt


def advance(F, D, v, t, T, dt):
    n = int(np.ceil((T - t) / dt)); hh = (T - t) / n
    for k in range(n):
        D, v = F.step(D, v, hh)
    return D, v


def twins(out):
    import make_tc6_perturbed as mk
    M = 128; X = grid.torus_X(M); case = tc.TC6()
    for shape in ("tetrahedral_l3", "zonal_P2"):
        Fa, S0, D0, v0, dt = solver(X, case); Fb = solver(X, case)[0]; w = S0.weight
        if shape == "tetrahedral_l3":
            Y = np.tensordot(mk.tetra_frame(), X, axes=1); p = Y[0] * Y[1] * Y[2]
        else:
            p = 1.5 * X[2] ** 2 - 0.5
        p = p / math.sqrt((w * p ** 2).sum() / w.sum())
        Da, va = D0.copy(), v0.copy(); Db, vb = np.ascontiguousarray(D0 + p), v0.copy(); t = 0.0; res = {}
        for day in (1, 3, 5, 7, 10, 14):
            T = day * tc.DAY
            Da, va = advance(Fa, Da, va, t, T, dt); Db, vb = advance(Fb, Db, vb, t, T, dt); t = T
            res[str(day)] = float(math.sqrt((w * (Da - Db) ** 2).sum() / w.sum()))
        rec = dict(mode="twins", M=M, shape=shape, initial_rms_m=1.0, rms_difference_m_by_day=res, scheme=configs.MAIN)
        print(json.dumps(rec), flush=True)
        open(out, "a").write(json.dumps(rec) + "\n")


def tilt(out):
    M = 128; X = grid.torus_X(M); case = tc.TC6()
    F, S0, D, v, dt = solver(X, case); w = S0.weight
    lat, lon = tc.latlon(X); latd = np.degrees(lat)
    nlon, nlat, glat, recs = ref.read(ex.REFS["tc6"])
    bands = [(10, 35), (35, 60), (60, 80)]
    t = 0.0
    for day in (3, 5, 7, 10, 14):
        T = day * tc.DAY; D, v = advance(F, D, v, t, T, dt); t = T
        I = ref.SphereInterp(glat, [r for r in recs if abs(r[0] - 24 * day) < 1e-6][0][3]); hm = D + S0.hs
        e = hm - I(X)
        zmean = np.zeros_like(e)
        for lo in range(-90, 90, 10):
            m = (latd >= lo) & (latd < lo + 10); zmean[m] = (w[m] * e[m]).sum() / w[m].sum()
        rec = dict(mode="tilt", M=M, day=day, scheme=configs.MAIN, rms_err_m=float(math.sqrt((w * e * e).sum() / w.sum())),
                   zonal_mean_share=float((w * zmean ** 2).sum() / (w * e ** 2).sum()),
                   polar_cap_mean_err_m={"N": float(zmean[latd > 70].mean()), "S": float(zmean[latd < -70].mean())}, bands={})
        for lo, hi in bands:
            for sgn, tag in ((1, "N"), (-1, "S")):
                m = (sgn * latd >= lo) & (sgn * latd < hi); Xm = X[:, m]
                def err(dl):
                    Xr = np.stack([np.cos(dl) * Xm[0] + np.sin(dl) * Xm[1], -np.sin(dl) * Xm[0] + np.cos(dl) * Xm[1], Xm[2]])
                    return float((w[m] * (hm[m] - I(Xr)) ** 2).sum())
                b = minimize_scalar(err, bounds=(-0.15, 0.15), method="bounded", options={"xatol": 1e-7})
                am = abs((w[m] * hm[m] * np.exp(-4j * lon[m])).sum()); ar = abs((w[m] * I(Xm) * np.exp(-4j * lon[m])).sum())
                rec["bands"][f"{tag}{lo}-{hi}"] = dict(shift_deg=float(np.degrees(b.x)), wave4_amplitude_ratio=float(am / ar))
        print(json.dumps(rec), flush=True)
        open(out, "a").write(json.dumps(rec) + "\n")


def nudge(out):
    M = 64; X = grid.torus_X(M); case = tc.TC6()
    nlon, nlat, g85, rec85 = ref.read(TRUTH_2H)
    times = np.array([r[0] for r in rec85]) * 3600.0
    _, _, glat, recs = ref.read(ex.REFS["tc6"])
    for R in (-1, 2, 6):
        F, S0, D, v, dt = solver(X, case); w = S0.weight
        mask = tfd.branch_mask(M, R) if R >= 0 else np.zeros((M, M), bool)
        Xm = X[:, mask]; e_, n_ = tc.east_north(Xm)
        truth = [(ref.SphereInterp(g85, hh)(Xm), ref.SphereInterp(g85, u, vector=True)(Xm) * e_ + ref.SphereInterp(g85, vv, vector=True)(Xm) * n_)
                 for t, u, vv, hh in rec85] if R >= 0 else None
        t = 0.0; res = {}
        for day in (3, 7, 14):
            T = day * tc.DAY; n = int(np.ceil((T - t) / dt)); hh = (T - t) / n
            for k in range(n):
                D, v = F.step(D, v, hh); t += hh
                if R >= 0:
                    j = min(int(t // 7200.0), len(times) - 2); a = (t - times[j]) / (times[j + 1] - times[j])
                    D[mask] = (1 - a) * truth[j][0] + a * truth[j + 1][0] - S0.hs[mask]
                    v[:, mask] = (1 - a) * truth[j][1] + a * truth[j + 1][1]
            t = T
            I = ref.SphereInterp(glat, [r for r in recs if abs(r[0] - 24 * day) < 1e-6][0][3])
            res[str(day)] = float(lwgg.norms(D + S0.hs, I(X), w)[1])
        rec = dict(mode="nudge", M=M, R_cells=R, imposed_points=int(mask.sum()), l2_by_day=res, scheme=configs.MAIN)
        print(json.dumps(rec), flush=True)
        open(out, "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    dict(twins=twins, tilt=tilt, nudge=nudge)[sys.argv[1]](sys.argv[2])
