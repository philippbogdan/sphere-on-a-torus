"""Move the conformal grid towards equal area."""


import numpy as np
from . import ellip

SQ3 = np.sqrt(3.0)


def lattice_coords(z):
    b = z.imag / ellip.EPS.imag
    a = z.real - b * ellip.EPS.real
    return a, b


class MoserField:
    def __init__(self, Nf=512, scale=None, beta=0.0):
        from .torusmap import SCALE
        s = SCALE if scale is None else scale
        self.Nf = Nf
        i = np.arange(Nf)
        I, J = np.meshgrid(i, i, indexing="ij")
        z = ((I + 0.5) + (J + 0.5) * ellip.EPS) / Nf
        p, dp = ellip.wp_and_derivative(z)
        w = p / s
        lam = 2.0 * np.abs(dp) / s / (1.0 + np.abs(w) ** 2)
        lam2 = lam ** 2
        self.area_check = lam2.mean() * ellip.EPS.imag / (8 * np.pi)
        mu = lam2 / lam2.mean()
        m = np.fft.fftfreq(Nf, 1.0 / Nf)
        Mm, Nn = np.meshgrid(m, m, indexing="ij")

        Kx = 2 * np.pi * Mm
        Ky = 2 * np.pi * (Nn - Mm * ellip.EPS.real) / ellip.EPS.imag
        K2 = Kx ** 2 + Ky ** 2
        K2[0, 0] = 1.0
        f = np.fft.fft2(mu - 1.0)
        psi_hat = -f / K2
        psi_hat[0, 0] = 0.0
        self.gx = np.real(np.fft.ifft2(1j * Kx * psi_hat))
        self.gy = np.real(np.fft.ifft2(1j * Ky * psi_hat))
        self.mu = mu

        self.beta = beta
        if beta:
            self.gx = self.gx * (1 - beta); self.gy = self.gy * (1 - beta)
        self.spectral_tail = np.abs(f[Nf // 2 - 4:Nf // 2 + 4, :]).max() / np.abs(f).max()

    def interp(self, z, fields, order=6):
        a, b = lattice_coords(z)
        u = a * self.Nf - 0.5
        v = b * self.Nf - 0.5
        iu = np.floor(u).astype(int); fu = u - iu
        iv = np.floor(v).astype(int); fv = v - iv
        offs = np.arange(-(order // 2) + 1, order // 2 + 1)
        def weights(f):
            W = []
            for k in offs:
                wk = np.ones_like(f)
                for l in offs:
                    if l != k:
                        wk = wk * (f - l) / (k - l)
                W.append(wk)
            return W
        Wu, Wv = weights(fu), weights(fv)
        out = [np.zeros_like(u) for _ in fields]
        Nf = self.Nf
        for ku, wu in zip(offs, Wu):
            iu_k = (iu + ku) % Nf
            for kv, wv in zip(offs, Wv):
                iv_k = (iv + kv) % Nf
                w = wu * wv
                for o, F in zip(out, fields):
                    o += w * F[iu_k, iv_k]
        return out

    def velocity(self, z, t):
        gx, gy, mu = self.interp(z, (self.gx, self.gy, self.mu))
        if self.beta:
            d = mu * (1.0 - t + t * self.beta) + t * (1.0 - self.beta)
        else:
            d = (1.0 - t) * mu + t
        return (gx + 1j * gy) / d

    def inverse_map(self, zeta, nsteps=400):
        z = np.array(zeta, dtype=complex)

        sk = np.linspace(1.0, 0.0, nsteps + 1) ** 2
        def f(z, s):
            return 2.0 * s * self.velocity(z, s * s)
        for k in range(nsteps):
            s0, s1 = sk[k], sk[k + 1]
            hs = s1 - s0
            k1 = f(z, s0)
            k2 = f(z + 0.5 * hs * k1, s0 + 0.5 * hs)
            k3 = f(z + 0.5 * hs * k2, s0 + 0.5 * hs)
            k4 = f(z + hs * k3, s1)
            z = z + hs / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z
