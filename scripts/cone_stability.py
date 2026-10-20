"""Run a numerical check and save its measurements."""


import sys, os, json, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import experiments as ex, configs, testcases as tc, grid


def main(out, M, alpha, label, extra):
    scheme = configs.MAIN + ("," + extra if extra else "")
    sc = ex.scheme_of(scheme)
    X, info = ex.make_grid(("torus", M), sc.get("orient"))
    case = tc.TC2(alpha)
    S, S0, hmean, cmax, dt0 = ex.build(X, case, sc)
    D = np.ascontiguousarray(case.height(X) - S0.hs)
    v = np.ascontiguousarray(case.velocity(X))
    T = 5 * tc.DAY; n = int(np.ceil(T / dt0 - 1e-9)); dt = T / n
    Dex, w = D.copy(), S0.weight
    hdeg = math.degrees(math.sqrt(4 * math.pi / (M * M / 2)))
    dcell = grid.branch_distance_deg(X, sc.get("orient")) / hdeg
    k3 = int(round(0.6 * n)); lin = {}; at = {}
    ok = True
    for k in range(1, n + 1):
        D, v = S.step(D, v, dt)
        if k % 50 == 0 and not np.isfinite(D).all():
            ok = False; break
        if k in (k3, n):
            a = np.abs(D - Dex); i = np.unravel_index(a.argmax(), a.shape)
            lin[k], at[k] = float(a.max()), float(dcell[i])
    rec = dict(M=M, alpha=alpha, label=label, scheme=scheme, stable=ok)
    if ok:
        e = D - Dex
        rec.update(l2=math.sqrt((w * e * e).sum() / (w * Dex * Dex).sum()), linf_m_day3=lin[k3], linf_m_day5=lin[n],
                   growth_day3_to_5=lin[n] / lin[k3], linf_cells_from_cone_day5=at[n],
                   flow_at_cones=[flow_at(alpha, b) for b in grid.branch_positions(sc.get("orient"))])
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def case_axis(alpha):
    return np.array([-math.sin(alpha), 0.0, math.cos(alpha)])


def flow_at(alpha, b):
    u = np.cross(case_axis(alpha), b)
    e = np.cross([0.0, 0.0, 1.0], b)
    if np.linalg.norm(e) < 1e-9 or np.linalg.norm(u) < 1e-9:
        return dict(speed_over_u0=float(np.linalg.norm(u)), direction_deg=None)
    e /= np.linalg.norm(e)
    return dict(speed_over_u0=float(np.linalg.norm(u)), direction_deg=float(math.degrees(math.atan2(u @ np.cross(b, e), u @ e))))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "")
