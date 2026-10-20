"""Numerical routines for the shallow-water studies."""


import numpy as np
import cvxpy as cp


def feasible(s, beta, ny=400, margin=1.0, order=2):
    import math
    y = np.linspace(0, beta, ny)
    p = order
    a = cp.Variable(s - p)
    fixed = np.array([1.0 / math.factorial(k) for k in range(p + 1)])
    cons = []
    k = np.arange(s + 1)
    for yy in y:
        z = (1j * yy) ** k
        re = float(np.real(z[:p + 1] @ fixed)); im = float(np.imag(z[:p + 1] @ fixed))
        Re = re + np.real(z[p + 1:]) @ a
        Im = im + np.imag(z[p + 1:]) @ a
        cons.append(cp.norm(cp.hstack([Re, Im])) <= margin)
    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return np.concatenate([fixed, a.value])


def optimal(s, tol=1e-3, margin=1.0, order=2):
    lo, hi = 0.0, float(s)
    best = None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        a = feasible(s, mid, margin=margin, order=order)
        if a is not None:
            lo, best = mid, a
        else:
            hi = mid
    return lo, best


def horner_coeffs(a):
    s = len(a) - 1
    c = np.zeros(s)
    c[s - 1] = 1.0
    for k in range(1, s):
        c[s - 1 - k] = a[k + 1] / a[k]
    return c


def check(a, beta):
    y = np.linspace(0, beta, 4000)
    P = np.polyval(a[::-1], 1j * y)
    return np.abs(P).max()


if __name__ == "__main__":


    import sys, json, os
    if len(sys.argv) > 2:
        s, p = int(sys.argv[1]), int(sys.argv[2])
        beta, a = optimal(s, order=p)
        c = horner_coeffs(a)
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_coeffs.json")
        tab = json.load(open(fn))
        tab[str(s) if p == 2 else f"{s}o{p}"] = dict(beta=beta, a=list(map(float, a)), c=list(map(float, c)), order_linear=p)
        json.dump(tab, open(fn, "w"), indent=1)
        print(f"s={s} p={p} beta={beta:.4f} max|P| on [0,beta]={check(a, beta):.6f}")
    else:
        for s in range(3, 13):
            beta, a = optimal(s)
            c = horner_coeffs(a)
            print(f"s={s:2d} beta={beta:.3f} (s-1={s-1}) beta/s={beta/s:.3f} max|P| on [0,beta]={check(a, beta):.6f}  c={np.round(c, 4).tolist()}")
