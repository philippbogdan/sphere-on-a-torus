"""Read locally generated NCAR reference fields."""


import numpy as np
from scipy.interpolate import RectBivariateSpline


def read(fn):
    with open(fn, "rb") as f:
        nlon, nlat = np.fromfile(f, dtype=np.int32, count=2)
        glat = np.fromfile(f, dtype=np.float64, count=nlat)
        recs = []
        while True:
            t = np.fromfile(f, dtype=np.float64, count=1)
            if t.size == 0:
                break
            arr = np.fromfile(f, dtype=np.float64, count=3 * nlat * nlon)
            if arr.size < 3 * nlat * nlon:
                break
            u, v, h = arr.reshape(3, nlat, nlon)
            recs.append((float(t[0]), u, v, h))
    return nlon, nlat, glat, recs


class SphereInterp:


    def __init__(self, glat, field, vector=False, ext=6, k=5):
        nlat, nlon = field.shape
        order = np.argsort(glat)
        lat = glat[order]
        F = field[order]
        lon = 2 * np.pi * np.arange(nlon) / nlon
        sgn = -1.0 if vector else 1.0
        half = nlon // 2
        Fs = sgn * np.roll(F[:ext][::-1], -half, axis=1)
        Fn = sgn * np.roll(F[-ext:][::-1], -half, axis=1)
        lat_s = -np.pi - lat[:ext][::-1]
        lat_n = np.pi - lat[-ext:][::-1]
        lat_e = np.concatenate([lat_s, lat, lat_n])
        F_e = np.concatenate([Fs, F, Fn], axis=0)
        lon_e = np.concatenate([lon[-ext:] - 2 * np.pi, lon, lon[:ext] + 2 * np.pi])
        F_e = np.concatenate([F_e[:, -ext:], F_e, F_e[:, :ext]], axis=1)
        self.it = RectBivariateSpline(lat_e, lon_e, F_e, kx=k, ky=k, s=0)

    def __call__(self, X):
        lat = np.arcsin(np.clip(X[2], -1, 1))
        lon = np.arctan2(X[1], X[0]) % (2 * np.pi)
        return self.it(lat.ravel(), lon.ravel(), grid=False).reshape(lat.shape)


def height_at(fn, X, tau_hours):
    nlon, nlat, glat, recs = read(fn)
    for t, u, v, h in recs:
        if abs(t - tau_hours) < 1e-6:
            return SphereInterp(glat, h)(X)
    raise KeyError(f"no output at tau={tau_hours} h in {fn}: {[r[0] for r in recs]}")
