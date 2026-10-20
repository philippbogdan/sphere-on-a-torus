"""Run a numerical check and save its measurements."""


import sys, os, json, subprocess, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np
from polarswe import experiments as ex, tfd, testcases as tc


def git_commit():
    from polarswe.revision import current
    return current()


def setup(spec, name, scheme):
    from polarswe import ellip
    if spec[0] == "torus" and scheme.get("lattice", "hex") != ellip.LATTICE:
        raise RuntimeError("lattice mismatch: set POLARSWE_LATTICE")
    sc = ex.scheme_of({k: v for k, v in scheme.items() if k != "solver"})
    case = ex.make_case(name)
    X, info = ex.make_grid(spec, sc.get("orient"))
    S0 = tfd.TFD(X, case, order=sc["order"], allow_sign_change=info["allow_sign_change"])
    D0 = case.height(X) - S0.hs
    v0 = case.velocity(X)
    cmax = np.sqrt(tc.G * D0.max()) + np.linalg.norm(v0, axis=0).max()
    dx_comp = tc.A * np.pi / np.sqrt(info["npts"] / 2.0)
    dx_uniq = tc.A * np.pi / np.sqrt(info["nunique"] / 2.0)
    return sc, case, X, info, S0, D0, v0, cmax, dx_comp, dx_uniq


def trial_lwt(r, spec, name, days, scheme, ctx):
    from polarswe import lwt, grid as gr
    sc, case, X, info, S0, D0, v0, cmax, dx_comp, dx_uniq = ctx
    M = spec[1]
    S = lwt.LWTorus(X, gr.torus_X_offset(M, 1.0), case, gr.branch_cells(M))
    hmean = np.sqrt(S.weight.mean())
    S.nu4 = ex.filter_nu(sc, hmean, cmax)
    dt0 = r * dx_comp / cmax
    T = days * tc.DAY; n = int(np.ceil(T / dt0)); dt = T / n
    Dp, v = D0.copy(), v0.copy(); bound = 3 * np.abs(D0).max()
    for k in range(n):
        Dp, v = S.step(Dp, v, dt)
        if k % 20 == 0 and not (np.abs(Dp).max() < bound):
            return False, k * dt / tc.DAY
    return bool(np.abs(Dp).max() < bound), days


_TRSK = {}


def trial_trsk(r, spec, name, days, scheme, ctx):
    from polarswe import trsk
    sc, case, X, info, S0, D0, v0, cmax, dx_comp, dx_uniq = ctx
    if spec not in _TRSK:
        _TRSK[spec] = trsk.VoronoiMesh(trsk.mesh_points(spec))
    sol = trsk.Solver(_TRSK[spec], case, scheme.get("integrator", "rk4"))
    h, u = sol.S.initial()
    dt0 = r * dx_comp / cmax
    T = days * tc.DAY; n = int(np.ceil(T / dt0)); dt = T / n
    bound = 3 * np.abs(h).max()
    for k in range(n):
        h, u = sol.step(h, u, dt)
        if k % 20 == 0 and not (np.abs(h).max() < bound):
            return False, k * dt / tc.DAY
    return bool(np.abs(h).max() < bound), days


def trial(r, spec, name, days, scheme, ctx):
    if scheme.get("solver") == "lwt":
        return trial_lwt(r, spec, name, days, scheme, ctx)
    if scheme.get("solver") == "trsk":
        return trial_trsk(r, spec, name, days, scheme, ctx)
    sc, case, X, info, S0, D0, v0, cmax, dx_comp, dx_uniq = ctx
    dt0 = r * dx_comp / cmax
    T = days * tc.DAY
    n = int(np.ceil(T / dt0)); dt = T / n

    gpu = os.environ.get("POLARSWE_DEVICE", "cpu") == "gpu"
    S = ex.build(X, case, sc, allow_sign_change=info["allow_sign_change"], device="gpu" if gpu else "cpu")[0]
    if gpu:
        import cupy as cp
        Dp, v = cp.asarray(D0), cp.asarray(v0)
    else:
        Dp, v = np.ascontiguousarray(D0.copy()), np.ascontiguousarray(v0.copy())
    bound = 3 * np.abs(D0).max()
    for k in range(n):
        Dp, v = S.step(Dp, v, dt)
        if k % 20 == 0:
            mx = max(abs(float(Dp.max())), abs(float(Dp.min())))
            if not (mx < bound):
                return False, k * dt / tc.DAY
    mx = max(abs(float(Dp.max())), abs(float(Dp.min())))
    return bool(mx < bound), days


def main():
    out, g, n, name, days, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    from polarswe import configs
    scheme_str = sys.argv[8] if len(sys.argv) > 8 else ""
    scheme = configs.parse(scheme_str)
    spec = (g, n)
    ctx = setup(spec, name, scheme)
    t0 = time.time(); log = []
    ok_lo, _ = trial(lo, spec, name, days, scheme, ctx); log.append((lo, ok_lo))
    ok_hi, t_hi = trial(hi, spec, name, days, scheme, ctx); log.append((hi, ok_hi))
    if not ok_lo or ok_hi:
        res = dict(error="bracket invalid", log=log)
    else:
        while (hi - lo) / lo > 0.02:
            mid = np.sqrt(lo * hi)
            ok, tf = trial(mid, spec, name, days, scheme, ctx)
            log.append((mid, ok, tf))
            if ok: lo = mid
            else: hi = mid
        sc, case, X, info, S0, D0, v0, cmax, dx_comp, dx_uniq = ctx
        res = dict(r_max_computed=lo, r_max_unique=lo * dx_comp / dx_uniq,
                   dt_max_s=lo * dx_comp / cmax, dx_eq_computed_m=dx_comp, dx_eq_unique_m=dx_uniq,
                   cmax=cmax, npts=info["npts"], nunique=info["nunique"], log=log)
    res.update(grid=spec, case=name, days=days, scheme=scheme, scheme_str=scheme_str, wall_s=time.time() - t0, commit=git_commit(),
               scheme_full={k: v for k, v in ctx[0].items()})
    with open(out, "a") as f:
        f.write(json.dumps(res, default=float) + "\n")
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
