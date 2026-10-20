"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, tfd, testcases as tc, lwgg, fp32, configs, experiments as ex


def main(out, case_name, Ms):
    sc = dict(ex.DEFAULT); sc.update(configs.parse(configs.MAIN))
    gpu = os.environ.get("POLARSWE_DEVICE", "cpu") == "gpu"
    if gpu:
        import cupy as cp
        dev = lambda a: cp.asarray(a)
        host = lambda a: cp.asnumpy(a)
    else:
        dev = host = lambda a: a
    for M in Ms:
        X = grid.torus_X(M); case = ex.make_case(case_name)
        S0 = tfd.TFD(X, case); h = math.sqrt(S0.weight.mean()); w = S0.weight
        D0 = case.height(X) - S0.hs; v0 = case.velocity(X)
        cmax = np.sqrt(tc.G * D0.max()) + np.linalg.norm(v0, axis=0).max()
        days = case.days
        dt = sc["courant"] * h / cmax; T = days * tc.DAY; n = int(np.ceil(T / dt)); dt = T / n
        mk = lambda: ex.build(X, case, configs.MAIN, fast=False)[0]
        res = {}
        S = mk()
        if gpu:
            tfd.to_device(S)
        D, v = dev(D0.copy()), dev(v0.copy())
        for k in range(n):
            D, v = S.step(D, v, dt)
        res["fp64"] = host(D)
        for name, wrap in (("fp32_plain", lambda: fp32.Plain32(mk())), ("fp32_reference", lambda: fp32.Reference32(mk(), D0, v0))):
            W = wrap()
            if gpu:
                W.to_device()
            Dp, vp = W.init(dev(D0), dev(v0))
            for k in range(n):
                Dp, vp = W.step(Dp, vp, dt)
            res[name] = host(W.full(Dp, vp)[0])
        href = ex.reference_height(case_name, case, X, T)
        hnorm = math.sqrt((w * href ** 2).sum() / w.sum())
        rec = dict(case=case_name, M=M, days=days, scheme=configs.MAIN,
                   fp64_error_l2=float(lwgg.norms(res["fp64"] + S0.hs, href, w)[1]))
        for name in ("fp32_plain", "fp32_reference"):
            d = res[name] - res["fp64"]
            rec[f"{name}_minus_fp64_l2"] = float(math.sqrt((w * d * d).sum() / w.sum()) / hnorm)
            rec[f"{name}_minus_fp64_max_m"] = float(np.abs(d).max())
        print(json.dumps(rec), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(m) for m in sys.argv[3:]])
