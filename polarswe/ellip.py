"""Evaluate the Weierstrass map on the periodic lattice."""


import os
import numpy as np

LATTICE = os.environ.get("POLARSWE_LATTICE", "hex")
if LATTICE == "hex":
    EPS = np.exp(1j * np.pi / 3.0)
elif LATTICE == "square":
    EPS = 1j
else:
    raise ValueError(f"unknown lattice {LATTICE}")
TAU = EPS
NTERMS = 12


def _qpow(n):
    return np.exp(1j * np.pi * TAU * (n + 0.5) ** 2)


def theta1_derivs(v):
    t0 = np.zeros_like(v, dtype=complex)
    t1 = np.zeros_like(v, dtype=complex)
    t2 = np.zeros_like(v, dtype=complex)
    for n in range(NTERMS):
        c = 2.0 * (-1) ** n * _qpow(n)
        k = 2 * n + 1
        s, co = np.sin(k * v), np.cos(k * v)
        t0 += c * s
        t1 += c * k * co
        t2 -= c * k * k * s
    return t0, t1, t2


def _const():
    d1 = sum(2.0 * (-1) ** n * _qpow(n) * (2 * n + 1) for n in range(NTERMS))
    d3 = sum(-2.0 * (-1) ** n * _qpow(n) * (2 * n + 1) ** 3 for n in range(NTERMS))
    return d3 / (3.0 * d1)


_C3 = _const()


def reduce_lattice(z):

    b = z.imag / EPS.imag
    a = z.real - b * EPS.real
    a -= np.round(a)
    b -= np.round(b)
    return a + b * EPS


def wp(z):
    z = reduce_lattice(np.asarray(z, dtype=complex))
    v = np.pi * z
    t0, t1, t2 = theta1_derivs(v)
    return np.pi ** 2 * ((t1 / t0) ** 2 - t2 / t0 + _C3)


HALF_PERIODS = np.array([0.5, 0.5 * EPS, 0.5 * (1 + EPS)])
BRANCH_POINTS = np.array([0.0, 0.5, 0.5 * EPS, 0.5 * (1 + EPS)])


def roots():
    return wp(HALF_PERIODS)


def wp_and_derivative(z):
    z = reduce_lattice(np.asarray(z, dtype=complex))
    v = np.pi * z
    t0 = np.zeros_like(v); t1 = np.zeros_like(v); t2 = np.zeros_like(v); t3 = np.zeros_like(v)
    for n in range(NTERMS):
        c = 2.0 * (-1) ** n * _qpow(n)
        k = 2 * n + 1
        s, co = np.sin(k * v), np.cos(k * v)
        t0 += c * s
        t1 += c * k * co
        t2 -= c * k ** 2 * s
        t3 -= c * k ** 3 * co
    L1, L2, L3 = t1 / t0, t2 / t0, t3 / t0
    p = np.pi ** 2 * (L1 ** 2 - L2 + _C3)
    dp = np.pi ** 3 * (3 * L1 * L2 - 2 * L1 ** 3 - L3)
    return p, dp
