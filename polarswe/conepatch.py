"""Numerical routines for the shallow-water studies."""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from . import testcases as tc


FD = {2: {1: 0.5}, 4: {1: 8 / 12, 2: -1 / 12}, 6: {1: 45 / 60, 2: -9 / 60, 3: 1 / 60},
      8: {1: 672 / 840, 2: -168 / 840, 3: 32 / 840, 4: -3 / 840}}


def stencil(order):
    out = [(0, 0, 0.0)]
    for s, c in FD[order].items():
        for ax in (0, 1):
            out += [(ax, s, c), (ax, -s, -c)]
    return out


def axial(B):
    return np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]])


class _System:
    def __init__(self, nu):
        self.I, self.J, self.V, self.b, self.k, self.nu = [], [], [], [], 0, nu

    def eq(self, entries, rhs):
        for j, v in entries:
            self.I.append(self.k); self.J.append(j); self.V.append(v)
        self.b.append(rhs); self.k += 1

    def solve(self, colscale):
        A = sp.csr_matrix((self.V, (self.I, self.J)), shape=(self.k, self.nu)); b = np.array(self.b)
        As = A @ sp.diags(colscale)
        rn = np.sqrt(np.asarray(As.multiply(As).sum(axis=1)).ravel()) + 1e-300
        As = sp.diags(1 / rn) @ As; bs = b / rn
        y = spla.lsqr(As, bs, atol=1e-15, btol=1e-15, iter_lim=500000)[0]
        return y * colscale, float(np.abs(As @ y - bs).max())


class ConePatch:
    def __init__(self, S, R, a=tc.A, wreg=1.0, diag=False, quad=False, wconst=False, taper=0.0):


        from .tfd import branch_mask
        m = S.m
        X = S.X
        M0, M1 = X.shape[1:]
        self.shape = (M0, M1)
        N = M0 * M1
        flat = lambda i, j: (i % M0) * M1 + (j % M1)
        mask = branch_mask(M0, R).ravel()
        pts = np.nonzero(mask)[0]
        pidx = {q: n for n, q in enumerate(pts)}
        Xf = X.reshape(3, N).T
        w0 = np.abs(m.sqrtG).ravel(); wbar = w0.mean(); wn = w0 / wbar
        ak = (m.a1.reshape(3, N).T * a, m.a2.reshape(3, N).T * a)
        Ak = (m.A1.reshape(3, N).T / a, m.A2.reshape(3, N).T / a)
        wu = m.sqrtG.ravel() / a ** 2
        ref = np.where(np.abs(Xf[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        t1 = np.cross(ref, Xf); t1 /= np.linalg.norm(t1, axis=1, keepdims=True); t2 = np.cross(Xf, t1)
        T = (t1, t2)
        st = stencil(S.order)
        extra = [(2, s_, 0.0) for s_ in range(-max(FD[S.order]), max(FD[S.order]) + 1) if s_ != 0] if diag else []
        reach = max(abs(s_) for ax_, s_, c_ in st)
        def nb(i, j, ax, s_):
            return flat(i + s_, j) if ax == 0 else flat(i, j + s_) if ax == 1 else flat(i + s_, j - s_)
        from . import torusmap, ellip
        I_, J_ = np.meshgrid(np.arange(M0), np.arange(M1), indexing="ij")
        dcell = np.abs(torusmap.nearest_branch(((I_ + 0.5) + (J_ + 0.5) * ellip.EPS) / M0) * M0).ravel()
        near = np.zeros(N, bool)
        for r in pts:
            ri, rj = divmod(r, M1)
            for d_ in range(-reach, reach + 1):
                near[flat(ri + d_, rj)] = True; near[flat(ri, rj + d_)] = True; near[flat(ri + d_, rj - d_)] = True
        wn0 = wn.copy()
        if wconst:
            wn = wn.copy(); wn[pts] = 1.0

        H0, K0 = {}, {}
        for q in np.nonzero(near)[0]:
            qi, qj = divmod(q, M1)
            for ax, s_, c in st + (extra if mask[q] else []):
                p = nb(qi, qj, ax, s_)
                if ax == 2:
                    if mask[p]:
                        H0.setdefault((q, p), np.zeros(3)); K0.setdefault((q, p), np.zeros(3))
                    continue
                if mask[q] or mask[p]:

                    H0[(q, p)] = H0.get((q, p), 0.0) + wn0[q] * c * ak[ax][q]
                if mask[q] and c != 0.0:

                    K0[(q, p)] = K0.get((q, p), 0.0) + (c * Ak[1][p] if ax == 0 else -c * Ak[0][p]) / wu[q]
        pairs = sorted(pq for pq in H0 if mask[pq[0]] and mask[pq[1]])
        npair = len(pairs)
        rowsof, colsof = {}, {}
        for n, (q, p) in enumerate(pairs):
            rowsof.setdefault(q, []).append((n, p)); colsof.setdefault(p, []).append((n, q))
        fixrow, fixcol = {}, {}
        for (q, p), h in H0.items():
            if mask[q] and not mask[p]:
                fixrow.setdefault(q, []).append((p, h))
            if mask[p] and not mask[q]:
                fixcol.setdefault(p, []).append((q, h))
        E = [np.outer(np.eye(3)[i], np.eye(3)[j]) for i in range(3) for j in range(3)]

        nu = 2 * npair + len(pts); iw = lambda q: 2 * npair + pidx[q]
        sysH = _System(nu)
        for q in pts:
            free = rowsof.get(q, []); fixed = fixrow.get(q, [])
            Hsum = sum((H0[(q, p)] for n, p in free), np.zeros(3)) + sum((h for p, h in fixed), np.zeros(3))
            for tb in T:
                tq = tb[q]; c1, c2 = tq @ t1[q], tq @ t2[q]
                sysH.eq([(2 * n, c1) for n, p in free] + [(2 * n + 1, c2) for n, p in free], -tq @ Hsum)
                for cmp in range(3):
                    lhs0 = sum((H0[(q, p)] * Xf[p, cmp] for n, p in free), np.zeros(3)) + sum((h * Xf[p, cmp] for p, h in fixed), np.zeros(3))
                    sysH.eq([(2 * n, c1 * Xf[p, cmp]) for n, p in free] + [(2 * n + 1, c2 * Xf[p, cmp]) for n, p in free]
                            + [(iw(q), -tq[cmp])], wn[q] * tq[cmp] - tq @ lhs0)
                if quad:
                    for ca, cb in ((0, 0), (1, 1), (0, 1), (0, 2), (1, 2)):
                        fv = lambda p_: Xf[p_, ca] * Xf[p_, cb]
                        g = np.eye(3)[ca] * Xf[q, cb] + np.eye(3)[cb] * Xf[q, ca]
                        g = g - (g @ Xf[q]) * Xf[q]
                        lhs0 = sum((H0[(q, p)] * fv(p) for n, p in free), np.zeros(3)) + sum((h * fv(p) for p, h in fixed), np.zeros(3))
                        sysH.eq([(2 * n, c1 * fv(p)) for n, p in free] + [(2 * n + 1, c2 * fv(p)) for n, p in free]
                                + [(iw(q), -(tq @ g))], wn[q] * (tq @ g) - tq @ lhs0)
        for r in pts:
            free = colsof.get(r, []); fixed = fixcol.get(r, [])
            s1 = sum((H0[(q, r)] for n, q in free), np.zeros(3)) + sum((h for q, h in fixed), np.zeros(3))
            for cmp in range(3):
                sysH.eq([(2 * n, t1[q, cmp]) for n, q in free] + [(2 * n + 1, t2[q, cmp]) for n, q in free]
                        + [(iw(r), -2 * Xf[r, cmp])], 2 * wn[r] * Xf[r, cmp] - s1[cmp])
            Xr = Xf[r]
            for B in E:
                tgt = np.trace(B) - 3 * Xr @ B @ Xr
                s0 = sum((H0[(q, r)] @ (B @ Xf[q]) for n, q in free), 0.0) + sum((h @ (B @ Xf[q]) for q, h in fixed), 0.0)
                sysH.eq([(2 * n, -(t1[q] @ (B @ Xf[q]))) for n, q in free] + [(2 * n + 1, -(t2[q] @ (B @ Xf[q]))) for n, q in free]
                        + [(iw(r), -tgt)], wn[r] * tgt + s0)
        gscale = max(np.abs(h).max() for h in H0.values())
        cs = np.ones(nu); cs[:2 * npair] = gscale; cs[2 * npair:] = 1.0 / wreg
        if taper:
            for n, (q, p) in enumerate(pairs):
                f_ = 1.0 / (1.0 + taper * (max(dcell[q], dcell[p]) / R) ** 4)
                cs[2 * n] *= f_; cs[2 * n + 1] *= f_
            for q in pts:
                cs[iw(q)] /= (1.0 + taper * (dcell[q] / R) ** 4)
        x, self.residual_GD = sysH.solve(cs)
        dH = np.array([x[2 * n] * t1[q] + x[2 * n + 1] * t2[q] for n, (q, p) in enumerate(pairs)]).reshape(-1, 3)

        for q in pts:
            free = rowsof.get(q, [])
            tot = sum((H0[(q, p)] + dH[n] for n, p in free), np.zeros(3)) + sum((h for p, h in fixrow.get(q, [])), np.zeros(3))
            nself = [n for n, p in free if p == q]
            if nself:
                dH[nself[0]] -= tot - (tot @ Xf[q]) * Xf[q]
        w = wn.copy(); w[pts] += x[2 * npair:]
        if (w <= 0).any():
            raise ValueError("corrected mass weights not positive")
        self.w_range = (float((w[pts] / wn[pts]).min()), float((w[pts] / wn[pts]).max()))

        kp = sorted(pq for pq in K0 if mask[pq[1]])
        kidx = {pq: n for n, pq in enumerate(kp)}
        sysK = _System(2 * len(kp))
        krow = {}
        for (r, q) in K0:
            krow.setdefault(r, []).append(q)
        for r in pts:
            qs = krow.get(r, [])
            for c in range(3):
                lhs0 = sum(K0[(r, q)] @ (np.eye(3)[c] - Xf[q, c] * Xf[q]) for q in qs)
                ent = []
                for q in qs:
                    if (r, q) in kidx:
                        n = kidx[(r, q)]; vq = np.eye(3)[c] - Xf[q, c] * Xf[q]
                        ent += [(2 * n, t1[q] @ vq), (2 * n + 1, t2[q] @ vq)]
                sysK.eq(ent, -lhs0)
            for B in E:
                tgt = axial(B) @ Xf[r]
                vq_ = {q: B @ Xf[q] - (Xf[q] @ B @ Xf[q]) * Xf[q] for q in qs}
                lhs0 = sum(K0[(r, q)] @ vq_[q] for q in qs)
                ent = []
                for q in qs:
                    if (r, q) in kidx:
                        n = kidx[(r, q)]
                        ent += [(2 * n, t1[q] @ vq_[q]), (2 * n + 1, t2[q] @ vq_[q])]
                sysK.eq(ent, tgt - lhs0)
        kscale = max(np.abs(k).max() for k in K0.values())
        y, self.residual_C = sysK.solve(np.full(2 * len(kp), kscale))
        dK = np.array([y[2 * n] * t1[q] + y[2 * n + 1] * t2[q] for n, (r, q) in enumerate(kp)]).reshape(-1, 3)

        Hn = {pq: H0[pq] + dH[n] for n, pq in enumerate(pairs)}
        gi, gj, gv = [], [], []
        for (q, p), h in H0.items():
            if mask[q]:
                gi.append(q); gj.append(p); gv.append(Hn.get((q, p), h) / w[q] - h / wn0[q])
        gv = np.array(gv).reshape(-1, 3) / a
        self.Gc = [sp.csr_matrix((gv[:, c], (gi, gj)), shape=(N, N)) for c in range(3)]
        di, dj, dv = [], [], []
        for (q, r), h in H0.items():
            if mask[r]:
                base = h / wn0[r]
                di.append(r); dj.append(q); dv.append(-(Hn.get((q, r), h) / w[r]) + base)
        dv = np.array(dv).reshape(-1, 3) / a
        self.Dc = [sp.csr_matrix((dv[:, c], (di, dj)), shape=(N, N)) for c in range(3)]
        ki = [r for r, q in kp]; kj = [q for r, q in kp]
        self.Cc = [sp.csr_matrix((dK[:, c] / a, (ki, kj)), shape=(N, N)) for c in range(3)]
        self.X = X; self.N = N; self.npatch = len(pts)
        self.weight = (w * wbar).reshape(self.shape)
        self.n_unknowns = (nu, 2 * len(kp)); self.n_constraints = (sysH.k, sysK.k)
        self.dH_max_rel = float(np.abs(dH).max() / gscale); self.dK_max_rel = float(np.abs(dK).max() / kscale)

    def grad(self, phi):
        f = phi.ravel()
        return np.stack([(self.Gc[k] @ f) for k in range(3)]).reshape((3,) + self.shape)

    def div(self, F):
        Ff = F.reshape(3, self.N)
        return sum(self.Dc[k] @ Ff[k] for k in range(3)).reshape(self.shape)

    def curl(self, v):
        vf = v.reshape(3, self.N)
        return sum(self.Cc[k] @ vf[k] for k in range(3)).reshape(self.shape)
