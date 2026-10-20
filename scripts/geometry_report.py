"""Run a numerical check and save its measurements."""


import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import grid, torusmap, ellip


def stats(X, M):
    h = 1.0 / M
    A1 = (np.roll(X, -1, 1) - np.roll(X, 1, 1)) / 2
    A2 = (np.roll(X, -1, 2) - np.roll(X, 1, 2)) / 2
    g11, g22, g12 = (A1 * A1).sum(0), (A2 * A2).sum(0), (A1 * A2).sum(0)
    E = np.array([[1.0, ellip.EPS.real], [0.0, ellip.EPS.imag]]) * h
    Ei = np.linalg.inv(E)
    Gm = np.stack([np.stack([g11, g12]), np.stack([g12, g22])])
    Gf = np.einsum('ai,abxy,bj->ijxy', Ei, Gm, Ei)
    tr = Gf[0, 0] + Gf[1, 1]; det = Gf[0, 0] * Gf[1, 1] - Gf[0, 1] ** 2
    s = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1 = np.sqrt(tr / 2 + s); l2 = np.sqrt(np.maximum(tr / 2 - s, 0))
    area = l1 * l2
    mean_lin = np.sqrt(area.mean())

    chords = []
    for di, dj in ((1, 0), (0, 1), (1, -1)) if ellip.LATTICE == "hex" else ((1, 0), (0, 1)):
        Y = np.roll(np.roll(X, -di, 1), -dj, 2)
        d = np.linalg.norm(Y - X, axis=0)
        chords.append(d[d > 1e-12])
    chords = np.concatenate([c.ravel() for c in chords])
    flat_nb = h
    return dict(sigma_min_over_mean=float(l2.min() / mean_lin), sigma_max_over_mean=float(l1.max() / mean_lin),
                anisotropy_max=float((l1 / np.maximum(l2, 1e-300)).max()),
                anisotropy_median=float(np.median(l1 / np.maximum(l2, 1e-300))),
                area_max_over_min=float(area.max() / area.min()),
                min_chord_over_mean_chord=float(chords.min() / chords.mean()),
                total_area_over_8pi=float(area.sum() * ellip.EPS.imag * h * h / (8 * np.pi)))


def main(out):
    res = {}
    for M in (128, 256):
        i = np.arange(M); I, J = np.meshgrid(i, i, indexing="ij")
        zeta = ((I + 0.5) + (J + 0.5) * ellip.EPS) / M
        res[f"conformal_M{M}"] = stats(torusmap.sphere_point(zeta), M)
        res[f"equal_area_M{M}"] = stats(grid.torus_X(M), M)
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        print(k, {a: round(b, 4) for a, b in v.items()})


if __name__ == "__main__":
    main(sys.argv[1])
