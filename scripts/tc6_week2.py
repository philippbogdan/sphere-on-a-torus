"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from polarswe import grid, tfd, fast, testcases as tc, reference as ref, lwgg, configs, experiments as ex, ellip

TRUTH_2H = os.path.join(ROOT, "refs/stswm/runs/T85_tc6_2h/grid_out.bin")
VERTS = np.array([[0, 0, 1.0]] + [[math.cos(math.radians(-19.4712206)) * math.cos(math.radians(l)),
                                    math.cos(math.radians(-19.4712206)) * math.sin(math.radians(l)),
                                    math.sin(math.radians(-19.4712206))] for l in (0, 120, -120)])
BANDS = ((-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90))


def emit(out, rec):
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")


def reference():
    nlon, nlat, glat, recs = ref.read(ex.REFS["tc6"])
    return glat, {r[0]: r for r in recs}


def setup(M, scheme, near_bp_only=False):
    X, _ = ex.make_grid(("torus", M), ex.scheme_of(scheme).get("orient")); case = tc.TC6()
    F, S0, h, cmax, dt = ex.build(X, case, scheme)
    if near_bp_only:
        dist = np.degrees(np.arccos(np.clip(np.tensordot(VERTS, X, axes=1), -1, 1))).min(0)
        F.nu4 = np.ascontiguousarray(F.nu4 * np.exp(-(dist / 10.0) ** 2))
    return F, S0, X, dt


def state_from(glat, rec, X, S0):
    e, n = tc.east_north(X)
    _, u, v, hh = rec
    D = ref.SphereInterp(glat, hh)(X) - S0.hs
    V = ref.SphereInterp(glat, u, vector=True)(X) * e + ref.SphereInterp(glat, v, vector=True)(X) * n
    return np.ascontiguousarray(D), np.ascontiguousarray(V)


def advance(F, D, v, t, T, dt):
    n = int(np.ceil((T - t) / dt)); hh = (T - t) / n
    for k in range(n):
        D, v = F.step(D, v, hh)
    return D, v


def restart(out, square=False):
    glat, R = reference()
    main = configs.MAIN
    q = f"{main},courant={configs.parse(main)['courant'] / 4:g}"
    if square:
        runs = [(128, 7, "square cover", configs.FRONTIER["square cover"][0], False)]
    else:
        runs = [(M, 7, "main", main, False) for M in (64, 128, 256, 512)] + [
        (128, 0, "main", main, False),
        (128, 7, configs.PREV_NAME, configs.PREV_MAIN, False),
        (128, 7, configs.FD4_NAME, configs.FRONTIER[configs.FD4_NAME][0], False),
        (128, 7, configs.FD8F_NAME, configs.FRONTIER[configs.FD8F_NAME][0], False),
        (128, 7, "main, kappa x 5", main + f",kappa={5 * configs.parse(main)['kappa']:g}", False),
        (128, 7, "main, kappa / 5", main + f",kappa={configs.parse(main)['kappa'] / 5:g}", False),
        (128, 7, "main, quarter step", q, False),
        (128, 7, "main, filter only near the branch points", main, True),
        (128, 7, configs.EDGE_NAME, configs.FRONTIER[configs.EDGE_NAME][0], False)]
    for M, d0, label, scheme, nb in runs:
        if square and ellip.LATTICE != "square":
            raise RuntimeError("restart_square needs POLARSWE_LATTICE=square")
        F, S0, X, dt = setup(M, scheme, nb); w = S0.weight
        lat, _ = tc.latlon(X); latd = np.degrees(lat)
        if d0 == 0:
            D = np.ascontiguousarray(tc.TC6().height(X) - S0.hs); v = np.ascontiguousarray(tc.TC6().velocity(X))
        else:
            D, v = state_from(glat, R[24.0 * d0], X, S0)
        t = d0 * tc.DAY; l2 = {}; extra = {}
        for day in range(d0 + 1, min(d0 + 7, 14) + 1):
            D, v = advance(F, D, v, t, day * tc.DAY, dt); t = day * tc.DAY
            href = ref.SphereInterp(glat, R[24.0 * day][3])(X); e = D + S0.hs - href
            l2[str(day)] = float(lwgg.norms(D + S0.hs, href, w)[1])
            if day == d0 + 3:
                zm = np.zeros_like(e)
                for lo in range(-90, 90, 5):
                    m = (latd >= lo) & (latd < lo + 5); zm[m] = (w[m] * e[m]).sum() / w[m].sum()
                extra = dict(day=day, zonal_mean_share=float((w * zm ** 2).sum() / (w * e ** 2).sum()),
                             rms_by_lat_m={f"{lo}..{hi}": float(math.sqrt((w[(latd >= lo) & (latd < hi)] * e[(latd >= lo) & (latd < hi)] ** 2).sum()
                                                                  / w[(latd >= lo) & (latd < hi)].sum())) for lo, hi in BANDS})
        emit(out, dict(mode="restart", M=M, start_day=d0, label=label, scheme=scheme, l2_by_day=l2, after_3_days=extra))


def sharpness(out):
    glat, R = reference()
    M = 512; X = grid.torus_X(M); S = tfd.TFD(X, tc.TC6())
    lat, _ = tc.latlon(X); latd = np.degrees(lat); e, n = tc.east_north(X)
    far = np.degrees(np.arccos(np.clip(np.tensordot(VERTS, X, axes=1), -1, 1))).min(0) > 6.0
    for day in (0, 3, 7, 10, 14):
        _, u, vv, hh = R[24.0 * day]
        V = ref.SphereInterp(glat, u, vector=True)(X) * e + ref.SphereInterp(glat, vv, vector=True)(X) * n
        zeta = S.m.curl(V, 6); gz = np.linalg.norm(S.m.grad(zeta, 6), axis=0)
        rec = dict(mode="sharpness", day=day, bands={})
        for lo, hi in ((0, 30), (30, 60), (60, 80)):
            m = (np.abs(latd) >= lo) & (np.abs(latd) < hi) & far
            rec["bands"][f"|lat| {lo}-{hi}"] = dict(max_abs_zeta=float(np.abs(zeta[m]).max()), max_grad_zeta=float(gz[m].max()),
                                                   length_m=float(np.abs(zeta[m]).max() / gz[m].max()))
        emit(out, rec)


def twins7(out):
    import make_tc6_perturbed as mk
    glat, R = reference()
    M = 128
    for shape in ("random", "zonal_P2", "tetrahedral_l3"):
        Fa, S0, X, dt = setup(M, configs.MAIN); Fb = setup(M, configs.MAIN)[0]; w = S0.weight
        if shape == "tetrahedral_l3":
            Y = np.tensordot(mk.tetra_frame(), X, axes=1); p = Y[0] * Y[1] * Y[2]
        elif shape == "zonal_P2":
            p = 1.5 * X[2] ** 2 - 0.5
        else:
            rng = np.random.default_rng(1); p = np.zeros(X.shape[1:])
            for k in range(40):
                a = rng.normal(size=3); a /= np.linalg.norm(a)
                p += rng.normal() * np.cos(rng.integers(1, 8) * np.arccos(np.clip((a[:, None, None] * X).sum(0), -1, 1)) + rng.uniform(0, 6.28))
        p = p / math.sqrt((w * p ** 2).sum() / w.sum())
        Da, va = state_from(glat, R[24.0 * 7], X, S0); Db, vb = np.ascontiguousarray(Da + p), va.copy()
        t = 7 * tc.DAY; res = {}
        for day in range(8, 15):
            Da, va = advance(Fa, Da, va, t, day * tc.DAY, dt); Db, vb = advance(Fb, Db, vb, t, day * tc.DAY, dt); t = day * tc.DAY
            res[str(day)] = float(math.sqrt((w * (Da - Db) ** 2).sum() / w.sum()))
        emit(out, dict(mode="twins7", M=M, shape=shape, initial_rms_m=1.0, rms_difference_m_by_day=res, scheme=configs.MAIN))


RELAX = [("none", 0, 0), ("bp", 10, 0), ("bp", 20, 0), ("disc", 5, 0), ("disc", 10, 0), ("disc", 20, 0),
         ("annulus", 10, 0), ("polar", 60, 0), ("bp", 20, 60), ("disc", 20, 60), ("annulus", 10, 60), ("disc", 10, 60)]


def relax(out, k=None):


    nlon, nlat, g, recs = ref.read(TRUTH_2H)
    times = np.array([r[0] for r in recs]) * 3600.0
    M = 128; X = grid.torus_X(M); lat, _ = tc.latlon(X); latd = np.degrees(lat); e, n = tc.east_north(X)
    for kk, (region, par, rot) in enumerate(RELAX):
        if k is not None and kk != k:
            continue
        F, S0, X_, dt = setup(M, configs.MAIN); w = S0.weight
        V = VERTS
        if rot:
            c_, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            Rz = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
            V = np.array([[0, 0, -1.0]] + [Rz @ v_ for v_ in VERTS[1:]])
        dist = np.degrees(np.arccos(np.clip(np.tensordot(V, X, axes=1), -1, 1))).min(0)
        W = {"none": lambda: np.zeros_like(dist), "bp": lambda: np.exp(-(dist / max(par, 1e-9)) ** 2),
             "disc": lambda: 0.5 * (1 - np.tanh((dist - par) / 2.0)),
             "annulus": lambda: 0.25 * (1 + np.tanh((dist - par) / 2.0)) * (1 - np.tanh((dist - 2 * par) / 2.0)),
             "polar": lambda: 0.5 * (1 + np.tanh((np.abs(latd) - par) / 3.0))}[region]()
        sel = W > 1e-3; Ws = W[sel]; tau = 3600.0
        cache = {}
        def truth_at(t):
            j = min(int(t // 7200.0), len(times) - 2); a = (t - times[j]) / (times[j + 1] - times[j])
            for i_ in (j, j + 1):
                if i_ not in cache:
                    _, u, vv, hh = recs[i_]
                    cache[i_] = (ref.SphereInterp(g, hh)(X[:, sel]) - S0.hs[sel],
                                 ref.SphereInterp(g, u, vector=True)(X[:, sel]) * e[:, sel] + ref.SphereInterp(g, vv, vector=True)(X[:, sel]) * n[:, sel])
            for i_ in [i_ for i_ in cache if i_ < j]:
                del cache[i_]
            return (1 - a) * cache[j][0] + a * cache[j + 1][0], (1 - a) * cache[j][1] + a * cache[j + 1][1]
        D = np.ascontiguousarray(tc.TC6().height(X) - S0.hs); v = np.ascontiguousarray(tc.TC6().velocity(X))
        t = 0.0; res = {}
        for day in (1, 3, 5, 7, 10):
            T = day * tc.DAY; nn = int(np.ceil((T - t) / dt)); hh = (T - t) / nn
            for i_ in range(nn):
                D, v = F.step(D, v, hh); t += hh
                if sel.any():
                    Dt, vt = truth_at(t); c = (hh / tau) * Ws
                    D[sel] += c * (Dt - D[sel]); v[:, sel] += c * (vt - v[:, sel])
            t = T
            hr = ref.SphereInterp(g, [r for r in recs if abs(r[0] - 24 * day) < 1e-6][0][3])(X)
            res[str(day)] = float(lwgg.norms(D + S0.hs, hr, w)[1])
        emit(out, dict(mode="relax", M=M, region=region, par=par, control_rotation_deg=rot,
                       weighted_area=float((w * W).sum() / w.sum()), l2_vs_T85_by_day=res, scheme=configs.MAIN))


_K = configs.parse(configs.MAIN)["kappa"]
LATLON = [(64, 0.0), (90, 0.0), (128, 0.0), (64, _K), (90, _K), (128, _K)]


def latlon(out, k=None):


    glat, R = reference()
    sc = ex.scheme_of(configs.MAIN)
    for kk, (N, kappa) in enumerate(LATLON):
        if k is not None and kk != k:
            continue
        X = grid.dfs_latlon_X(2 * N, N); case = tc.TC6()
        S0 = tfd.TFD(X, case, order=sc["order"], allow_sign_change=True); w = S0.weight
        D = case.height(X) - S0.hs; v = case.velocity(X)
        cmax = np.sqrt(tc.G * D.max()) + np.linalg.norm(v, axis=0).max()
        h = math.sqrt(w.mean()); nunique = 2 * N * N
        dx_u = tc.A * math.pi / math.sqrt(nunique / 2.0)
        A1 = (np.roll(X, -1, 1) - np.roll(X, 1, 1)) / 2 * tc.A; A2 = (np.roll(X, -1, 2) - np.roll(X, 1, 2)) / 2 * tc.A
        smin = np.minimum(np.linalg.norm(A1, axis=0), np.linalg.norm(A2, axis=0))
        S = tfd.TFD(X, case, order=sc["order"], nu4=1.0 if kappa else 0.0, integrator=sc["integrator"], hv_mode="step",
                    allow_sign_change=True)
        F = fast.FastTFD(S)
        if kappa:
            F.nu4 = np.ascontiguousarray(kappa * np.minimum(smin, h) ** 4 / (0.5 * h / cmax))
        kmax = {2: 1.0, 4: 1.372, 6: 1.586, 8: 1.73}[sc["order"]]
        dt = 0.5 * 3.32 * math.sin(math.pi / (2 * N)) * (8.0 / 2.83) / kmax * dx_u / cmax
        D = np.ascontiguousarray(D); v = np.ascontiguousarray(v); t = 0.0; res = {}
        for day in (1, 3, 5, 7, 10, 14):
            D, v = advance(F, D, v, t, day * tc.DAY, dt); t = day * tc.DAY
            if not np.isfinite(D).all() or np.abs(D).max() > 1e5:
                res[str(day)] = None; break
            res[str(day)] = float(lwgg.norms(D + S0.hs, ref.SphereInterp(glat, R[24.0 * day][3])(X), w)[1])
        emit(out, dict(mode="latlon", N=N, nunique=nunique, kappa=kappa, dt=dt, l2_by_day=res, scheme=configs.MAIN))


_f = configs.parse(configs.MAIN)["kappa"] / 0.0005


_U = ",cone_filter=1"
FILTERS = [("del4 metric (main)", ""), ("del4 flat", ",lap_kind=flat"), ("del4 metric, also in the cone patches", _U),
           ("del6 flat, also in the cone patches", f",hv_order=6,lap_kind=flat,kappa={9.6e-5 * _f:.3g}" + _U),
           ("del8 flat, also in the cone patches", f",hv_order=8,lap_kind=flat,kappa={1.85e-5 * _f:.3g}" + _U),
           ("del10 flat, also in the cone patches", f",hv_order=10,lap_kind=flat,kappa={3.55e-6 * _f:.3g}" + _U),
           ("del8 flat, 4x, also in the cone patches", f",hv_order=8,lap_kind=flat,kappa={7.4e-5 * _f:.3g}" + _U),
           ("del8 metric, also in the cone patches", f",hv_order=8,kappa={1.85e-5 * _f:.3g}" + _U),
           ("del4 metric, kappa/5", f",kappa={configs.parse(configs.MAIN)['kappa'] / 5:g}")]


def filters(out, k=None):

    glat, R = reference()
    for kk, (label, extra) in enumerate(FILTERS):
        if k is not None and kk != k:
            continue
        M = 128; X = grid.torus_X(M); case = tc.TC6()
        F, S0, h, cmax, dt = ex.build(X, case, configs.MAIN + extra); w = S0.weight
        D = np.ascontiguousarray(case.height(X) - S0.hs); v = np.ascontiguousarray(case.velocity(X)); t = 0.0; res = {}
        for day in (1, 3, 5, 7, 10, 14):
            D, v = advance(F, D, v, t, day * tc.DAY, dt); t = day * tc.DAY
            if not np.isfinite(D).all():
                res[str(day)] = None; break
            res[str(day)] = float(lwgg.norms(D + S0.hs, ref.SphereInterp(glat, R[24.0 * day][3])(X), w)[1])
        emit(out, dict(mode="filters", M=M, label=label, scheme=configs.MAIN + extra, l2_by_day=res))


if __name__ == "__main__":
    k = int(sys.argv[3]) if len(sys.argv) > 3 else None
    modes = dict(restart=lambda o: restart(o), restart_square=lambda o: restart(o, square=True), sharpness=sharpness,
                 twins7=twins7, relax=lambda o: relax(o, k), latlon=lambda o: latlon(o, k), filters=lambda o: filters(o, k))
    modes[sys.argv[1]](sys.argv[2])
