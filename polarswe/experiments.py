"""Run shallow-water cases with one shared solver constructor."""


import os, json, time
import numpy as np
from . import grid, tfd, lwgg, testcases as tc, reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = {

    "tc5": os.path.join(ROOT, "refs/stswm/runs/T213dt22_tc5/grid_out.bin"),
    "tc6": os.path.join(ROOT, "refs/stswm/runs/T213dt22_tc6/grid_out.bin"),
    "galewsky": os.path.join(ROOT, "refs/stswm/runs/T341dt30_galewsky/grid_out.bin"),
}


REFS_ALT = {
    "tc5": {"T341dt90": os.path.join(ROOT, "refs/stswm/runs/T341dt90_tc5/grid_out.bin"),
            "T213archived": os.path.join(ROOT, "refs/stswm/runs/T213_tc5/grid_out.bin")},
    "tc6": {"T341dt90": os.path.join(ROOT, "refs/stswm/runs/T341dt90_tc6/grid_out.bin"),
            "T213archived": os.path.join(ROOT, "refs/stswm/runs/T213_tc6/grid_out.bin")},
    "galewsky": {"T341dt60": os.path.join(ROOT, "refs/stswm/runs/T341dt60_galewsky/grid_out.bin")},
}


DEFAULT = dict(order=4, kappa=0.002, integrator="rk4", form="flux", rings=0, courant=0.5,
               hv_mode="step", lap_kind="metric", hv_order=4, cone="", orient="vertex", kappa_near=-1.0, rnear=5.0,
               rnear_cells=-1.0, cone_filter=1.0, tracer="flux")


def scheme_of(scheme):
    from . import configs
    sc = dict(DEFAULT)
    sc.update(configs.parse(scheme) if isinstance(scheme, str) else (scheme or {}))
    return sc


def branch_distance_cells(M):

    from . import ellip, torusmap
    i = np.arange(M)
    I, J = np.meshgrid(i, i, indexing="ij")
    return np.abs(torusmap.nearest_branch(((I + 0.5) + (J + 0.5) * ellip.EPS) / M)) * M


def cone_radius(sc):
    return float(str(sc["cone"]).replace(":", "_").split("_")[0])


def filter_nu(sc, h, c, X=None):


    nu = sc["kappa"] * h ** sc["hv_order"] / (0.5 * h / c) if sc["kappa"] else 0.0
    if sc.get("cone") and float(sc.get("cone_filter", 1.0)) != 1.0 and X is not None and nu:
        out = np.full(X.shape[1:], nu)
        out[branch_distance_cells(X.shape[1]) <= cone_radius(sc)] = nu * float(sc["cone_filter"])
        return out
    if sc.get("kappa_near", -1.0) >= 0 and X is not None:
        nn = sc["kappa_near"] * h ** sc["hv_order"] / (0.5 * h / c)
        if sc.get("rnear_cells", -1.0) > 0:
            d = branch_distance_cells(X.shape[1])
            wgt = np.exp(-(d / sc["rnear_cells"]) ** 2)
            if sc.get("cone"):
                wgt[d > cone_radius(sc)] = 0.0
            return nu + (nn - nu) * wgt
        d = grid.branch_distance_deg(X, sc.get("orient"))
        return nu + (nn - nu) * np.exp(-(d / sc["rnear"]) ** 2)
    return nu


def build(X, case, scheme, allow_sign_change=False, fast=True, device="cpu"):
    """Return (solver, metric_solver, spacing_m, wave_speed_m_per_s, step_s).

    X is a unit-sphere position array of shape (3, M, M).
    The step is the configured Courant number times mean spacing over initial wave speed.
    """


    sc = scheme_of(scheme)
    from . import ellip
    if not allow_sign_change and sc.get("lattice", "hex") != ellip.LATTICE:
        raise RuntimeError(f"scheme asks for lattice {sc.get('lattice', 'hex')} but POLARSWE_LATTICE is {ellip.LATTICE}")
    S0 = tfd.TFD(X, case, order=sc["order"], allow_sign_change=allow_sign_change)
    h = float(np.sqrt(S0.weight.mean()))
    D0 = case.height(X) - S0.hs
    cmax = float(np.sqrt(tc.G * D0.max()) + np.linalg.norm(case.velocity(X), axis=0).max())
    S = tfd.TFD(X, case, order=sc["order"], nu4=filter_nu(sc, h, cmax, X), integrator=sc["integrator"], form=sc["form"],
                special_rings=sc["rings"], allow_sign_change=allow_sign_change, hv_mode=sc["hv_mode"],
                lap_kind=sc["lap_kind"], hv_order=sc["hv_order"])
    if sc.get("cone"):
        S.set_cone(cone_of(X, sc))
    if device == "gpu":
        S = tfd.to_device(S)
    elif fast and sc["form"] == "flux" and not sc["rings"]:
        from .fast import FastTFD
        S = FastTFD(S)
    return S, S0, h, cmax, sc["courant"] * h / cmax


def cone_of(X, sc):


    from . import conehp
    parts = str(sc["cone"]).replace(":", "_").split("_")
    R, p, rs = float(parts[0]), int(parts[1]), int(parts[2])
    shape = parts[3] if len(parts) > 3 else "disc"
    curl = "compat" if (len(parts) > 4 and parts[4] == "cc") else True
    degw = float(parts[5][2:]) if (len(parts) > 5 and parts[5].startswith("dw")) else 1.0
    return conehp.cached(tfd.TFD(X, tc.TC2(0.0), order=sc["order"]), R=R, p=p, rs=rs, shape=shape, curl=curl, degw=degw)


def make_case(name):
    name = name.lower()
    if name.startswith("tc2"):
        alpha = float(name.split("_a")[1]) if "_a" in name else 0.0
        return tc.TC2(alpha)
    if name.startswith("tc1"):
        alpha = float(name.split("_a")[1]) if "_a" in name else 0.0
        return tc.TC1(alpha)
    if name == "galewsky_nopert":
        return tc.Galewsky(perturb=False)
    return {"tc5": tc.TC5, "tc6": tc.TC6, "galewsky": tc.Galewsky}[name]()


def make_grid(spec, orient=None):
    kind, n = spec
    if kind == "torus":
        X = grid.oriented(grid.torus_X(n), orient)
        return X, dict(allow_sign_change=False, npts=n * n, nunique=n * n // 2)
    if kind == "dfs":
        X = grid.dfs_latlon_X(2 * n, n)
        return X, dict(allow_sign_change=True, npts=2 * (2 * n) * n, nunique=2 * n * n)
    raise ValueError(kind)


def reference_height(name, case, X, t):
    if name.startswith("tc2") or name.startswith("tc1"):
        return case.height(X, t)
    if name == "galewsky_nopert":
        return case.height(X)
    fn = REFS[name.lower()]
    return reference.height_at(fn, X, t / 3600.0)


def run(spec, name, days=None, scheme=None, dt=None, verbose=False, max_steps=None, device=None):
    device = device or os.environ.get("POLARSWE_DEVICE", "cpu")
    sc = scheme_of(scheme)
    if sc.get("solver") == "trsk":
        from . import trsk
        return trsk.run(tuple(spec), name, {k: sc[k] for k in ("solver", "integrator", "courant")}, days=days,
                        by_day=[1, 3, 5, 7, 10] if name == "tc6" else None, verbose=verbose)
    case = make_case(name)
    days = case.days if days is None else days
    X, info = make_grid(spec, sc.get("orient"))
    T = days * tc.DAY
    if name.startswith("tc1"):
        if sc.get("solver") == "lwt":
            raise NotImplementedError("advection-only case for the Lax-Wendroff torus")
        hmean = float(np.sqrt(tfd.TFD(X, case, order=sc["order"], allow_sign_change=info["allow_sign_change"]).weight.mean()))
        return run_tc1(X, info, case, sc, days, dt, hmean, verbose, spec, name)
    if sc.get("solver") == "lwt":
        S0 = tfd.TFD(X, case, order=sc["order"]); hmean = np.sqrt(S0.weight.mean())
        D0 = case.height(X) - S0.hs
        cmax = np.sqrt(tc.G * D0.max()) + np.linalg.norm(case.velocity(X), axis=0).max()
        dt = sc["courant"] * hmean / cmax if dt is None else dt
        n = int(np.ceil(T / dt - 1e-9))
        return run_lwt(X, info, case, sc, days, T / n, hmean, cmax, verbose, spec, name)
    S, S0, hmean, cmax, dt0 = build(X, case, sc, allow_sign_change=info["allow_sign_change"], device=device)
    nu4 = filter_nu(sc, hmean, cmax, X)
    if dt is None:
        dt = dt0
    n = int(np.ceil(T / dt - 1e-9)); dt = T / n
    Dp = np.ascontiguousarray(case.height(X) - S0.hs)
    v = np.ascontiguousarray(case.velocity(X))
    if device == "gpu":
        import cupy as cp
        Dp, v = cp.asarray(Dp), cp.asarray(v)
    m0 = float(S.mass(Dp))
    t0 = time.time(); ok = True
    for k in range(n):
        Dp, v = S.step(Dp, v, dt)
        if k % 50 == 0 and not (abs(float(Dp.max())) < 1e6 and abs(float(Dp.min())) < 1e6):
            ok = False; break
        if max_steps and k + 1 >= max_steps:
            break
    if device == "gpu":
        mass_end = float(S.mass(Dp))
        Dp, v = Dp.get(), v.get()
    else:
        mass_end = float(S.mass(Dp))
    out = dict(grid=spec, case=name, days=days, scheme=sc, dt=dt, nsteps=n, nu4=float(np.max(nu4)), device=device,
               hmean_m=hmean, cmax=cmax, wall_s=time.time() - t0, stable=bool(ok and np.isfinite(Dp).all()),
               npts=info["npts"], nunique=info["nunique"])
    if out["stable"] and not max_steps:
        h = Dp + case.topography(X)
        href = reference_height(name, case, X, T)
        w = S0.weight
        l1, l2, li = lwgg.norms(h, href, w)
        out.update(l1=l1, l2=l2, linf=li, mass_rel=(mass_end - m0) / m0,
                   rms_err_m=float(np.sqrt((w * (h - href) ** 2).sum() / w.sum())))
        for tag, alt in REFS_ALT.get(name.lower(), {}).items():
            if os.path.exists(alt):
                try:
                    ha = reference.height_at(alt, X, T / 3600.0)
                    out[f"l2_vs_{tag}"] = float(lwgg.norms(h, ha, w)[1])
                    out[f"rms_err_vs_{tag}_m"] = float(np.sqrt((w * (h - ha) ** 2).sum() / w.sum()))
                except KeyError:
                    pass
    if verbose:
        print(json.dumps({k: out[k] for k in out if k not in ("scheme",)}, default=float))
    return out


def run_tc1(X, info, case, sc, days, dt, hmean, verbose, spec, name):


    S = tfd.TFD(X, case, order=sc["order"], allow_sign_change=info["allow_sign_change"], lap_kind=sc["lap_kind"],
                hv_order=sc["hv_order"])
    v = case.velocity(X)
    q = case.height(X)
    umax = np.linalg.norm(v, axis=0).max()
    if dt is None:
        dt = sc["courant"] * hmean / umax
    T = days * tc.DAY; n = int(np.ceil(T / dt - 1e-9)); dt = T / n
    nu4 = filter_nu(sc, hmean, umax, X)
    m, o = S.m, sc["order"]
    if sc.get("cone"):
        S.set_cone(cone_of(X, sc))
    C = S.cone
    if sc.get("tracer", "flux") == "skew":
        Dv = m.div(v, o) + (C.div(v) if C is not None else 0.0)
        def rhs_nohv(q):
            Gq = m.grad(q, o) + (C.grad(q) if C is not None else 0.0)
            return -0.5 * (m.div(q * v, o) + (C.div(q * v) if C is not None else 0.0) + (v * Gq).sum(0) + q * Dv)
    elif sc.get("tracer", "flux") == "flux":
        def rhs_nohv(q):
            return -(m.div(q * v, o) + (C.div(q * v) if C is not None else 0.0))
    else:
        raise ValueError(sc["tracer"])
    def rhs(q):
        r = rhs_nohv(q)
        if np.any(nu4):
            r = r - (nu4 * S.hyper(q) if np.ndim(nu4) == 0 else S.lap(nu4 * S.lap(q)))
        return r
    t0 = time.time(); m0 = (S.mass_weight * q).sum()
    for k in range(n):
        if sc["integrator"] == "rk4":
            k1 = rhs(q); k2 = rhs(q + 0.5 * dt * k1); k3 = rhs(q + 0.5 * dt * k2); k4 = rhs(q + dt * k3)
            q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif sc["integrator"].startswith("kg"):
            qs = q
            for ck in tfd.kg_coeffs(sc["integrator"][2:]):
                qs = q + ck * dt * rhs_nohv(qs)
            q = qs - dt * (nu4 * S.hyper(qs) if np.ndim(nu4) == 0 else S.lap(nu4 * S.lap(qs))) if np.any(nu4) else qs
        elif sc["integrator"].startswith("ls"):
            A_, B_ = tfd.ls_coeffs(sc["integrator"][2:])
            qs, qq = q, 0.0 * q
            for Ak, Bk in zip(A_, B_):
                qq = Ak * qq + dt * rhs_nohv(qs)
                qs = qs + Bk * qq
            q = qs - dt * (nu4 * S.hyper(qs) if np.ndim(nu4) == 0 else S.lap(nu4 * S.lap(qs))) if np.any(nu4) else qs
        else:
            k1 = rhs(q); q1 = q + dt * k1; q2 = 0.75 * q + 0.25 * (q1 + dt * rhs(q1))
            q = q / 3 + 2 / 3 * (q2 + dt * rhs(q2))
    qref = case.height(X, T)
    l1, l2, li = lwgg.norms(q, qref, S.weight)
    out = dict(grid=spec, case=name, days=days, scheme=sc, dt=dt, nsteps=n, nu4=float(np.max(nu4)), device="cpu", hmean_m=hmean,
               cmax=umax, wall_s=time.time() - t0, stable=bool(np.isfinite(q).all()), npts=info["npts"],
               nunique=info["nunique"], l1=l1, l2=l2, linf=li, mass_rel=((S.mass_weight * q).sum() - m0) / m0,
               qmin=float(q.min()), qmax_rel=float(q.max() / qref.max()))
    if verbose:
        print(json.dumps({k: out[k] for k in out if k != "scheme"}, default=float))
    return out


def run_lwt(X, info, case, sc, days, dt, hmean, cmax, verbose, spec, name):
    from . import lwt
    M = spec[1]
    S = lwt.LWTorus(X, grid.torus_X_offset(M, 1.0), case, grid.branch_cells(M))
    S.nu4 = sc["kappa"] * hmean ** 4 / (0.5 * hmean / cmax) if sc["kappa"] else 0.0
    Dp = case.height(X) - S.hs; v = case.velocity(X)
    T = days * tc.DAY
    if dt is None:
        dt = sc["courant"] * hmean / cmax
    n = int(np.ceil(T / dt - 1e-9)); dt = T / n
    m0 = S.mass(Dp); t0 = time.time(); ok = True
    for k in range(n):
        Dp, v = S.step(Dp, v, dt)
        if k % 50 == 0 and not (np.abs(Dp).max() < 1e6):
            ok = False; break
    out = dict(grid=spec, case=name, days=days, scheme=sc, dt=dt, nsteps=n, nu4=S.nu4, device="cpu", hmean_m=hmean,
               cmax=cmax, wall_s=time.time() - t0, stable=bool(ok and np.isfinite(Dp).all()), npts=info["npts"],
               nunique=info["nunique"])
    if out["stable"]:
        h = Dp + S.hs
        href = reference_height(name, case, X, T)
        w = S.weight
        l1, l2, li = lwgg.norms(h, href, w)
        out.update(l1=l1, l2=l2, linf=li, mass_rel=(S.mass(Dp) - m0) / m0,
                   rms_err_m=float(np.sqrt((w * (h - href) ** 2).sum() / w.sum())))
    if verbose:
        print(json.dumps({k: out[k] for k in out if k != "scheme"}, default=float))
    return out
