"""Numerical routines for the shallow-water studies."""


import numpy as np
from scipy.integrate import quad

A = 6.37122e6
OMEGA = 7.292e-5
G = 9.80616
DAY = 86400.0


def latlon(X):
    lat = np.arcsin(np.clip(X[2], -1, 1))
    lon = np.arctan2(X[1], X[0]) % (2 * np.pi)
    return lat, lon


def east_north(X):
    r = np.sqrt(X[0] ** 2 + X[1] ** 2)
    safe = r > 1e-14
    rs = np.where(safe, r, 1.0)
    e = np.stack([np.where(safe, -X[1] / rs, 0.0), np.where(safe, X[0] / rs, 1.0), np.zeros_like(r)])
    n = np.cross(X, e, axis=0)
    return e, n


def axis(alpha):
    return np.array([-np.sin(alpha), 0.0, np.cos(alpha)])


def coriolis(X, alpha=0.0):

    return 2 * OMEGA * np.tensordot(axis(alpha), X, axes=1)


def solid_body(X, u0, alpha):
    k = axis(alpha)
    return (u0 / 1.0) * np.cross(np.broadcast_to(k[:, None], (3, X[0].size)).reshape((3,) + X.shape[1:]), X, axis=0)


class Case:
    alpha = 0.0
    hs = None
    days = None
    def coriolis(self, X):
        return coriolis(X, self.alpha)
    def topography(self, X):
        return np.zeros(X.shape[1:])


class TC1(Case):
    days = 12.0
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.u0 = 2 * np.pi * A / (12 * DAY)
    def velocity(self, X, t=0.0):
        return solid_body(X, self.u0, self.alpha)
    def height(self, X, t=0.0):
        h0, R = 1000.0, A / 3

        c0 = np.array([0.0, -1.0, 0.0])
        k = axis(self.alpha); th = self.u0 * t / A
        c = c0 * np.cos(th) + np.cross(k, c0) * np.sin(th) + k * np.dot(k, c0) * (1 - np.cos(th))
        r = A * np.arccos(np.clip(np.tensordot(c, X, axes=1), -1, 1))
        return np.where(r < R, 0.5 * h0 * (1 + np.cos(np.pi * r / R)), 0.0)


class TC2(Case):
    days = 5.0
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.u0 = 2 * np.pi * A / (12 * DAY)
        self.gh0 = 2.94e4
    def velocity(self, X, t=0.0):
        return solid_body(X, self.u0, self.alpha)
    def height(self, X, t=0.0):
        s = np.tensordot(axis(self.alpha), X, axes=1)
        return (self.gh0 - (A * OMEGA * self.u0 + 0.5 * self.u0 ** 2) * s ** 2) / G


class TC5(Case):
    days = 15.0
    def __init__(self):
        self.u0, self.h0 = 20.0, 5960.0
    def velocity(self, X, t=0.0):
        return solid_body(X, self.u0, 0.0)
    def height(self, X, t=0.0):
        lat, _ = latlon(X)
        return self.h0 - (A * OMEGA * self.u0 + 0.5 * self.u0 ** 2) * np.sin(lat) ** 2 / G
    def topography(self, X):
        lat, lon = latlon(X)
        R = np.pi / 9
        dl = lon - 1.5 * np.pi
        r = np.minimum(R, np.sqrt(dl ** 2 + (lat - np.pi / 6) ** 2))
        return 2000.0 * (1 - r / R)


class TC6(Case):
    days = 14.0
    w = K = 7.848e-6
    R = 4
    h0 = 8000.0
    def velocity(self, X, t=0.0):
        lat, lon = latlon(X)
        w, K, R = self.w, self.K, self.R
        c, s = np.cos(lat), np.sin(lat)
        u = A * w * c + A * K * c ** (R - 1) * (R * s ** 2 - c ** 2) * np.cos(R * lon)
        v = -A * K * R * c ** (R - 1) * s * np.sin(R * lon)
        e, n = east_north(X)
        return u * e + v * n
    def height(self, X, t=0.0):
        lat, lon = latlon(X)
        w, K, R = self.w, self.K, self.R
        c = np.cos(lat)
        Aa = 0.5 * w * (2 * OMEGA + w) * c ** 2 + 0.25 * K ** 2 * c ** (2 * R) * (
            (R + 1) * c ** 2 + (2 * R ** 2 - R - 2) - 2 * R ** 2 * c ** (-2.0))
        Bb = 2 * (OMEGA + w) * K / ((R + 1) * (R + 2)) * c ** R * ((R ** 2 + 2 * R + 2) - (R + 1) ** 2 * c ** 2)
        Cc = 0.25 * K ** 2 * c ** (2 * R) * ((R + 1) * c ** 2 - (R + 2))
        gh = G * self.h0 + A ** 2 * (Aa + Bb * np.cos(R * lon) + Cc * np.cos(2 * R * lon))
        return gh / G


class Galewsky(Case):
    days = 6.0
    umax, phi0, phi1 = 80.0, np.pi / 7, np.pi / 2 - np.pi / 7
    hbar = 10000.0
    def __init__(self, perturb=True):
        self.perturb = perturb
        self.en = np.exp(-4.0 / (self.phi1 - self.phi0) ** 2)

        ph = np.linspace(-np.pi / 2, np.pi / 2, 20001)
        integrand = -A * self.u(ph) * (2 * OMEGA * np.sin(ph) + np.tan(np.clip(ph, -1.57, 1.57)) * self.u(ph) / A) / G
        hh = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ph))])

        mean = 0.5 * np.trapezoid(hh * np.cos(ph), ph)
        self._ph, self._h = ph, hh - mean + self.hbar
    def u(self, ph):
        ph = np.asarray(ph, dtype=float)
        inside = (ph > self.phi0) & (ph < self.phi1)
        out = np.zeros_like(ph)
        p = ph[inside]
        out[inside] = self.umax / self.en * np.exp(1.0 / ((p - self.phi0) * (p - self.phi1)))
        return out
    def velocity(self, X, t=0.0):
        lat, _ = latlon(X)
        e, _ = east_north(X)
        return self.u(lat) * e
    def height(self, X, t=0.0):
        lat, lon = latlon(X)
        h = np.interp(lat, self._ph, self._h)
        if self.perturb:
            lam = np.where(lon > np.pi, lon - 2 * np.pi, lon)
            h = h + 120.0 * np.cos(lat) * np.exp(-(lam / (1.0 / 3)) ** 2) * np.exp(-((np.pi / 4 - lat) / (1.0 / 15)) ** 2)
        return h
