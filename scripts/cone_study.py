"""Run a numerical check and save its measurements."""


import sys, os, json, math, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, tfd, testcases as tc, lwgg, configs, experiments as ex


def emit(out, rec):
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def smooth_fields(X):

    x, y, z = X
    rng = np.random.default_rng(0); B = rng.standard_normal((3, 3))
    phi = np.exp(x) * np.sin(2 * y + z)
    g = np.stack([phi, np.exp(x) * 2 * np.cos(2 * y + z), np.exp(x) * np.cos(2 * y + z)]); g = g - (g * X).sum(0) * X
    BX = np.tensordot(B, X, axes=1); PBX = BX - (BX * X).sum(0) * X
    ez = np.exp(z); gez = np.stack([0 * z, 0 * z, ez]); gez = gez - (gez * X).sum(0) * X
    F = PBX * ez
    div = ez * (np.trace(B) - 3 * (X * BX).sum(0)) + (gez * PBX).sum(0)
    ax = np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]])
    curl = ez * np.tensordot(ax, X, axes=1) + (np.cross(gez, PBX, axis=0) * X).sum(0)
    return phi, g, F, div, curl


def ops(out, M, cone=None):
    sc = ex.scheme_of(configs.MAIN + (f",cone={cone}" if cone else ""))
    X = grid.torus_X(M)
    S = tfd.TFD(X, tc.TC2(0.7), order=sc["order"])
    t0 = time.time(); C = ex.cone_of(X, sc); tb = time.time() - t0
    a = tc.A; m = S.m; o = sc["order"]
    phi, g, F, div, curl = smooth_fields(X)
    h_deg = math.degrees(math.sqrt(4 * math.pi / (M * M / 2)))
    dcell = grid.branch_distance_deg(X) / h_deg
    bins = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 1e9))
    rec = dict(mode="ops", M=M, cone=sc["cone"], build_s=tb, solve=C.info, svd=C.svd,
               patch_points=int(sum(i["npts"] for i in C.info)))
    for name, num, ex_ in (("grad_fd", m.grad(phi, o) * a, g), ("grad", (m.grad(phi, o) + C.grad(phi)) * a, g),
                           ("div_fd", m.div(F, o) * a, div), ("div", (m.div(F, o) + C.div(F)) * a, div),
                           ("curl_fd", m.curl(F, o) * a, curl), ("curl", (m.curl(F, o) + C.curl(F)) * a, curl)):
        e = np.abs(num - ex_); e = e.max(0) if e.ndim == 3 else e
        rec[name] = {f"{lo}-{hi}": float(e[(dcell >= lo) & (dcell < hi)].max()) for lo, hi in bins
                     if ((dcell >= lo) & (dcell < hi)).any()}
    w = C.weight
    G = lambda f: m.grad(f, o) + C.grad(f); Dv = lambda Fv: m.div(Fv, o) + C.div(Fv)
    psi = np.cos(3 * X[0] + X[1]) * X[2]
    rec["adjointness"] = float((w * (psi * Dv(F) + (F * G(psi)).sum(0))).sum() / (w * np.abs(psi * Dv(F))).sum())
    rec["conservation"] = float((w * Dv(F)).sum() / (w * np.abs(Dv(F))).sum())
    emit(out, rec)


def tc2(out, M, cone, kappa, R=None, label=None, extra=""):
    sc = ex.scheme_of(configs.MAIN)

    scheme = configs.MAIN + (f",kappa={kappa}" if kappa is not None else "") + (",kappa_near=-1" if kappa == 0 else "") + ("" if cone else ",cone=")
    if cone and R is not None:
        parts = str(sc["cone"]).split("_"); parts[0] = f"{R:g}"
        scheme += ",cone=" + "_".join(parts)
    if extra:
        scheme += "," + extra.replace(";", ",")
    r = ex.run(("torus", M), "tc2_a0.7", scheme=scheme)
    emit(out, dict(mode="tc2", M=M, cone_operators=bool(cone), kappa=ex.scheme_of(scheme)["kappa"], scheme=scheme,
                   label=label or ("main" if cone else "no cone operators"),
                   cone_R=float(str(ex.scheme_of(scheme)["cone"]).split("_")[0]) if cone else None,
                   l2=r.get("l2"), linf=r.get("linf"), mass_rel=r.get("mass_rel"), stable=r["stable"]))


if __name__ == "__main__":
    mode, out, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
    opt = dict(a.split("=", 1) for a in sys.argv[4:])
    if mode == "ops":
        ops(out, M, opt.get("cone"))
    else:
        tc2(out, M, int(opt.get("cone", 1)), float(opt["kappa"]) if "kappa" in opt else None,
            float(opt["R"]) if "R" in opt else None, opt.get("label"), opt.get("extra", ""))
