"""Run a numerical check and save its measurements."""


import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe.tfd import ls_coeffs, kg_coeffs


def butcher(A, B):
    s = len(A)
    d = np.zeros((s, s))
    for i in range(s):
        d[i, i] = 1.0
        if i > 0:
            d[i, :i] = A[i] * d[i - 1, :i]
    a = np.zeros((s, s))
    for i in range(1, s):
        for j in range(i):
            a[i, j] = sum(B[k] * d[k, j] for k in range(j, i))
    b = np.array([sum(B[k] * d[k, j] for k in range(j, s)) for j in range(s)])
    return a, b, a.sum(1)


def main(out):
    A, B = ls_coeffs("10o4")
    a, b, c = butcher(np.array(A), np.array(B))
    oc = [b.sum() - 1, b @ c - 1 / 2, b @ c ** 2 - 1 / 3, b @ a @ c - 1 / 6, b @ c ** 3 - 1 / 4, (b * c) @ a @ c - 1 / 8,
          b @ a @ c ** 2 - 1 / 12, b @ a @ a @ c - 1 / 24]
    tab = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "polarswe/kg_coeffs.json")))
    poly = np.array(tab["10o4"]["a"]); s = len(A)
    v = np.ones(s); coef = [1.0]
    for n in range(1, s + 1):
        coef.append(b @ v); v = a @ v
    dpoly = float(np.max(np.abs(np.array(coef) - poly) / np.maximum(np.abs(poly), 1e-300)))
    y = np.linspace(0, 12, 120001); P = np.abs(np.polyval(np.array(coef)[::-1], 1j * y))
    first = lambda tol: float(y[np.argmax(P > 1 + tol)]) if np.any(P > 1 + tol) else float(y[-1])
    beta = first(1e-12); beta4 = first(1e-4); pmax = float(P[y <= tab["10o4"]["beta"]].max())
    f = lambda u: -u * u + np.sin(u)
    def ls(u, dt):
        q = 0.0
        for Ak, Bk in zip(A, B):
            q = Ak * q + dt * f(u); u = u + Bk * q
        return u
    def kg(u, dt):
        u0 = u
        for ck in kg_coeffs("10o4"):
            u = u0 + ck * dt * f(u)
        return u
    def run(st, n):
        u = 1.0
        for k in range(n):
            u = st(u, 2.0 / n)
        return u
    ref = run(ls, 6400)
    order = lambda st: float(np.log2(abs(run(st, 40) - ref) / abs(run(st, 80) - ref)))
    res = dict(scheme="ls10o4", polynomial_of="kg10o4", max_order_residual=float(np.max(np.abs(oc))),
               max_polynomial_difference=dpoly, imaginary_interval=beta, imaginary_interval_1e4=beta4,
               design_interval=tab["10o4"]["beta"], max_abs_P_on_design_interval=pmax, observed_order_nonlinear=order(ls),
               observed_order_nonlinear_kg=order(kg), A=list(A), B=list(B))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("A", "B")}))


if __name__ == "__main__":
    main(sys.argv[1])
