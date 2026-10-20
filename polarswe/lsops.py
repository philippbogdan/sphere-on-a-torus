"""Numerical routines for the shallow-water studies."""


import numpy as np


def hex_offsets(m):
    out = []
    for di in range(-m, m + 1):
        for dj in range(-m, m + 1):
            if (abs(di) + abs(dj) + abs(di + dj)) // 2 <= m:
                out.append((di, dj))
    return out


def monomials(q):
    return [(a, b) for d in range(q + 1) for a in range(d, -1, -1) for b in [d - a]]


def frame(X):
    ref = np.where(np.abs(X[2:3]) < 0.9, np.array([0, 0, 1.0])[:, None, None] * np.ones_like(X), np.array([1.0, 0, 0])[:, None, None] * np.ones_like(X))
    t1 = np.cross(ref, X, axis=0)
    t1 /= np.linalg.norm(t1, axis=0)
    t2 = np.cross(X, t1, axis=0)
    return t1, t2


def ls_weights(X, mask, m=2, q=3, power=2.0):
    offs = hex_offsets(m)
    mons = monomials(q)
    idx = np.argwhere(mask)
    Xp = X[:, idx[:, 0], idx[:, 1]]
    t1, t2 = frame(Xp[:, :, None])
    t1, t2 = t1[:, :, 0], t2[:, :, 0]
    M0, M1 = X.shape[1:]
    Y = np.empty((len(idx), len(offs), 2))
    for k, (di, dj) in enumerate(offs):
        Xq = X[:, (idx[:, 0] + di) % M0, (idx[:, 1] + dj) % M1]
        Y[:, k, 0] = (Xq * t1).sum(0)
        Y[:, k, 1] = (Xq * t2).sum(0)

    hs = np.sqrt((Y ** 2).sum(-1)).max(1)[:, None, None] / m
    Ys = Y / hs
    V = np.stack([Ys[..., 0] ** a * Ys[..., 1] ** b for a, b in mons], -1)
    r = np.sqrt((Ys ** 2).sum(-1))
    w = 1.0 / (1.0 + r) ** power
    Vw = V * w[..., None]

    pinv = np.linalg.pinv(Vw)
    L = pinv * w[:, None, :]
    i10 = mons.index((1, 0)); i01 = mons.index((0, 1))
    gx = L[:, i10, :] / hs[:, :, 0]
    gy = L[:, i01, :] / hs[:, :, 0]
    W = t1[:, :, None] * gx[None] + t2[:, :, None] * gy[None]
    return offs, idx, W
