"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import experiments as ex, configs, testcases as tc

BANDS = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 1e9))


def main(out, M, alpha, label, extra):
    scheme = configs.MAIN + ("," + extra if extra else "")
    sc = ex.scheme_of(scheme)
    X, info = ex.make_grid(("torus", M), sc.get("orient"))
    case = tc.TC2(alpha)
    S, S0, hmean, cmax, dt0 = ex.build(X, case, sc, fast=False)
    D0 = case.height(X) - S0.hs; v0 = case.velocity(X)
    d = ex.branch_distance_cells(M)
    tD, tv = S.rhs(D0, v0)
    fD = -S.filtered(D0) if np.any(S.nu4) else 0.0 * D0
    fv = -np.stack([S.filtered(v0[q]) for q in range(3)]) if np.any(S.nu4) else 0.0 * v0
    rec = dict(M=M, alpha=alpha, label=label, scheme=scheme, bands={})
    for lo, hi in BANDS:
        b = (d >= lo) & (d < hi)
        if b.any():
            rec["bands"][f"{lo}-{hi:g}"] = dict(scheme_D=float(np.abs(tD[b]).max()), filter_D=float(np.abs(fD[b]).max()),
                                                scheme_v=float(np.linalg.norm(tv[:, b], axis=0).max()),
                                                filter_v=float(np.linalg.norm(fv[:, b], axis=0).max()))
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "")
