"""Run a numerical check and save its measurements."""


import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, tfd, testcases as tc, reference as ref, lwgg, configs, experiments as ex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(out, Ms, scheme=None):
    scheme = scheme or configs.MAIN
    nlon, nlat, glat, recs = ref.read(ex.REFS["tc6"])
    refs = {r[0]: ref.SphereInterp(glat, r[3]) for r in recs}
    for M in Ms:
        sc = ex.scheme_of(scheme)
        X, _ = ex.make_grid(("torus", M), sc.get("orient")); case = tc.TC6()
        S, S0, hmean, cmax, dt = ex.build(X, case, scheme)
        D = np.ascontiguousarray(case.height(X) - S0.hs); v = np.ascontiguousarray(case.velocity(X))
        w = S0.weight
        aam = lambda D, v: float((w * D * (X[0] * v[1] - X[1] * v[0])).sum())
        a0 = aam(D, v); am = {}
        t = 0.0; res = {}
        for day in (1, 3, 5, 7, 10, 14):
            T = day * tc.DAY; n = int(np.ceil((T - t) / dt)); h = (T - t) / n
            for k in range(n):
                D, v = S.step(D, v, h)
            t = T
            l1, l2, li = lwgg.norms(D + S0.hs, refs[day * 24.0](X), S0.weight)
            res[day] = l2
            am[day] = (aam(D, v) - a0) / a0
        rec = dict(M=M, nunique=M * M // 2, l2_by_day=res, aam_rel_by_day=am, scheme=scheme)
        print(json.dumps(rec))
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    args = [a for a in sys.argv[2:] if not a.startswith("scheme=")]
    sch = [a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("scheme=")]
    main(sys.argv[1], [int(m) for m in args], sch[0] if sch else None)
