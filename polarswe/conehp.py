"""Fit local cone corrections to an adjoint gradient and divergence pair."""


import numpy as np
import scipy.sparse as sp
from . import testcases as tc, ellip


class RowOp:

    def __init__(self, A, xp=np):
        A = sp.csr_matrix(A)
        rows = np.nonzero(np.diff(A.indptr))[0]
        L = int(max(np.diff(A.indptr)[rows].max(), 1)) if len(rows) else 1
        cols = np.zeros((len(rows), L), dtype=np.int64); vals = np.zeros((len(rows), L), dtype=A.dtype)
        for n, r in enumerate(rows):
            a, b = A.indptr[r], A.indptr[r + 1]
            cols[n, :b - a] = A.indices[a:b]; vals[n, :b - a] = A.data[a:b]
        self.N = A.shape[0]; self.xp = xp
        self.rows, self.cols, self.vals = xp.asarray(rows), xp.asarray(cols), xp.asarray(vals)

    def __matmul__(self, f):
        out = self.xp.zeros(self.N, dtype=self.vals.dtype if self.vals.dtype == f.dtype else np.result_type(self.vals.dtype, f.dtype))
        if len(self.rows):
            out[self.rows] = (self.vals * f[self.cols]).sum(1)
        return out

    def astype(self, dtype):
        import copy
        o = copy.copy(self); o.vals = self.vals.astype(dtype)
        return o

FD = {2: {1: 0.5}, 4: {1: 8 / 12, 2: -1 / 12}, 6: {1: 45 / 60, 2: -9 / 60, 3: 1 / 60},
      8: {1: 672 / 840, 2: -168 / 840, 3: 32 / 840, 4: -3 / 840}}


def hexdist(di, dj):
    return (abs(di) + abs(dj) + abs(di + dj)) // 2


def branch_index_points(M):
    h = M / 2
    return [(-0.5, -0.5), (h - 0.5, -0.5), (-0.5, h - 0.5), (h - 0.5, h - 0.5)]


def monomials(p):
    return [(a, b) for d in range(p + 1) for a in range(d + 1) for b in [d - a]]


class ConeHP:
    def __init__(self, S, R=8.0, p=4, rs=4, a=tc.A, curl=True, verbose=False, rcond=1e-10, Rc=None, shape="disc", pD=None, orth=False, degw=1.0,
                 solver="svd"):
        if ellip.LATTICE != "hex":
            raise ValueError("hexagonal lattice only")
        m = S.m
        X = np.asarray(S.X)
        M0, M1 = X.shape[1:]
        assert M0 == M1
        M = M0; N = M * M
        self.shape = (M, M); self.N = N
        o = S.order
        cs = FD[o]
        Xf = X.reshape(3, N).T
        sgn = 1.0 if np.all(m.sqrtG > 0) else -1.0
        w0 = np.abs(m.sqrtG).ravel() / a ** 2
        ak = (m.a1.reshape(3, N).T * a, m.a2.reshape(3, N).T * a)
        Ak = (m.A1.reshape(3, N).T / a, m.A2.reshape(3, N).T / a)
        sgu = m.sqrtG.ravel() / a ** 2
        flat = lambda i, j: (i % M) * M + (j % M)
        mirror = lambda q: flat(-1 - q // M, -1 - q % M)

        def E0(i, j):
            ii, ij_ = divmod(i, M); ji, jj = divmod(j, M)
            di = (ji - ii + M // 2) % M - M // 2; dj = (jj - ij_ + M // 2) % M - M // 2
            out = np.zeros(3)
            if dj == 0 and di != 0 and abs(di) in cs:
                out += w0[i] * np.sign(di) * cs[abs(di)] * ak[0][i]
            if di == 0 and dj != 0 and abs(dj) in cs:
                out += w0[i] * np.sign(dj) * cs[abs(dj)] * ak[1][i]
            return out

        def K0(i, j):
            ii, ij_ = divmod(i, M); ji, jj = divmod(j, M)
            di = (ji - ii + M // 2) % M - M // 2; dj = (jj - ij_ + M // 2) % M - M // 2
            out = np.zeros(3)
            if dj == 0 and di != 0 and abs(di) in cs:
                out += np.sign(di) * cs[abs(di)] * Ak[1][j] / sgu[i]
            if di == 0 and dj != 0 and abs(dj) in cs:
                out -= np.sign(dj) * cs[abs(dj)] * Ak[0][j] / sgu[i]
            return out

        if shape == "disc":
            offs = [(di, dj) for di in range(-rs, rs + 1) for dj in range(-rs, rs + 1) if hexdist(di, dj) <= rs]
        elif shape == "cross":
            offs = [(0, 0)] + [(s_, 0) for s_ in range(-rs, rs + 1) if s_] + [(0, s_) for s_ in range(-rs, rs + 1) if s_]
        elif shape == "lines":
            offs = [(0, 0)] + [(s_ * a_, s_ * b_) for a_, b_ in ((1, 0), (0, 1), (1, -1)) for s_ in range(-rs, rs + 1) if s_]
        else:
            raise ValueError(shape)
        self.n_entries = len(offs)
        fdoffs = [(s, 0) for s in range(-max(cs), max(cs) + 1) if s] + [(0, s) for s in range(-max(cs), max(cs) + 1) if s]
        mons = monomials(p)
        monsD = monomials(p if pD is None else pD) if (pD is None or pD >= 0) else []
        seen = set()
        for (bi, bj) in branch_index_points(M):
            for di in range(-int(R) - 2 - rs, int(R) + 3 + rs):
                for dj in range(-int(R) - 2 - rs, int(R) + 3 + rs):
                    i, j = int(np.floor(bi)) + 1 + di, int(np.floor(bj)) + 1 + dj
                    ui, uj = i - bi, j - bj
                    if ui * ui + uj * uj + ui * uj <= (R + rs) ** 2:
                        q = flat(i, j)
                        if q in seen:
                            raise ValueError(f"cone patches of R = {R} cells (stencil reach {rs}) overlap on the M = {M} torus")
                        seen.add(q)
        E_corr = {}
        K_corr = {}
        w = w0.copy()
        self.info = []
        for (bi, bj) in branch_index_points(M):

            pts = []
            for di in range(-int(R) - 2, int(R) + 3):
                for dj in range(-int(R) - 2, int(R) + 3):
                    i, j = int(np.floor(bi)) + 1 + di, int(np.floor(bj)) + 1 + dj
                    ui, uj = i - bi, j - bj
                    if ui * ui + uj * uj + ui * uj <= R * R:
                        pts.append(flat(i, j))
            pts = sorted(set(pts))
            inP = set(pts)
            Rcon = R if Rc is None else Rc
            def bdist(q):
                qi, qj = divmod(q, M)
                ui = (qi - bi + M / 2) % M - M / 2; uj = (qj - bj + M / 2) % M - M / 2
                return float(np.sqrt(ui * ui + uj * uj + ui * uj))

            rep = {q: min(q, mirror(q)) for q in pts}
            phys = sorted(set(rep.values()))
            pid = {P: n for n, P in enumerate(phys)}
            nP = len(phys)
            B = Xf[phys].mean(0); B /= np.linalg.norm(B)
            e1 = np.cross([0.0, 0.0, 1.0], B) if abs(B[2]) < 0.9 else np.cross([1.0, 0.0, 0.0], B)
            e1 /= np.linalg.norm(e1); e2 = np.cross(B, e1)
            rho = max(np.linalg.norm(Xf[q] - B) for q in pts)

            def lc(Y):
                return (Y @ e1) / rho, (Y @ e2) / rho

            def mval(Y, ab):
                x, y = lc(Y); return x ** ab[0] * y ** ab[1]

            def mgrad(Y, ab):
                x, y = lc(Y); A_, B_ = ab
                gx = A_ * x ** (A_ - 1) * y ** B_ if A_ else 0.0
                gy = B_ * x ** A_ * y ** (B_ - 1) if B_ else 0.0
                g = (gx * e1 + gy * e2) / rho
                return g - (g @ Y) * Y

            if orth:

                Vm = np.array([[mval(Xf[P], ab) for ab in mons] for P in phys]) * np.sqrt(w0[phys])[:, None]
                _, Rq = np.linalg.qr(Vm)
                Rinv = np.linalg.inv(Rq)
                mval0, mgrad0 = mval, mgrad
                def mval(Y, k, _m=mons, _R=Rinv):
                    return sum(mval0(Y, ab) * _R[n, k] for n, ab in enumerate(_m) if _R[n, k] != 0.0)
                def mgrad(Y, k, _m=mons, _R=Rinv):
                    return sum((mgrad0(Y, ab) * _R[n, k] for n, ab in enumerate(_m) if _R[n, k] != 0.0), np.zeros(3))
                monsB = list(range(len(mons)))

            T = {}
            for P in phys:
                Y = Xf[P]
                t1 = e1 - (e1 @ Y) * Y; t1 /= np.linalg.norm(t1); t2 = np.cross(Y, t1)
                T[P] = (t1, t2)

            def stencil_of(q):
                qi, qj = divmod(q, M)
                return [flat(qi + di, qj + dj) for di, dj in offs]


            free = {}
            col_copy = {}
            for P in phys:
                for j in stencil_of(P):
                    if j in inP:
                        Q = rep[j]
                        if (P, Q) not in free:
                            free[(P, Q)] = len(free)
                            col_copy[(P, Q)] = j
            nfree = len(free)

            fixed_row = {P: [] for P in phys}
            for P in phys:
                pi_, pj = divmod(P, M)
                for di, dj in fdoffs:
                    j = flat(pi_ + di, pj + dj)
                    if j not in inP:
                        fixed_row[P].append((Xf[j], E0(P, j)))
            fixed_col = {Q: [] for Q in phys}
            for Q in phys:
                for Qc in (Q,):
                    qi, qj = divmod(Qc, M)
                    for di, dj in fdoffs:
                        i = flat(qi + di, qj + dj)
                        if i not in inP:

                            fixed_col[Q].append((Xf[i], E0(i, Qc), i))

            x0 = np.zeros(2 * nfree + nP)
            E_init = {}
            for P in phys:
                pi_, pj = divmod(P, M)
                for di, dj in fdoffs:
                    j = flat(pi_ + di, pj + dj)
                    if j in inP:
                        E_init[(P, rep[j])] = E_init.get((P, rep[j]), 0.0) + E0(P, j)
            for (P, Q), n in free.items():
                v = E_init.get((P, Q), np.zeros(3))
                x0[2 * n] = v @ T[P][0]; x0[2 * n + 1] = v @ T[P][1]
            x0[2 * nfree:] = [w0[P] for P in phys]

            rows, cols, vals, rhs = [], [], [], []
            cw = []
            k = 0
            byrow = {}; bycol = {}
            for (P, Q), n in free.items():
                byrow.setdefault(P, []).append((Q, n)); bycol.setdefault(Q, []).append((P, n))
            for P in phys:
                if bdist(P) > Rcon:
                    continue
                Y = Xf[P]; t1, t2 = T[P]
                for ab in (monsB if orth else mons):
                    g = mgrad(Y, ab)
                    fixed_sum = sum((E * mval(Xc, ab) for Xc, E in fixed_row[P]), np.zeros(3))
                    for tb, comp in ((t1, 0), (t2, 1)):

                        for Q, n in byrow.get(P, []):
                            mv = mval(Xf[Q], ab)
                            if mv != 0.0:
                                rows += [k, k]; cols += [2 * n, 2 * n + 1]; vals += [mv * (T[P][0] @ tb), mv * (T[P][1] @ tb)]
                        rows.append(k); cols.append(2 * nfree + pid[P]); vals.append(-(g @ tb))
                        dg = sum(ab) if isinstance(ab, tuple) else 0
                        rhs.append(-(fixed_sum @ tb)); cw.append(1e3 if ab in ((0, 0), 0) else degw ** (p - dg)); k += 1
            nG = k

            for Q in phys:
                if bdist(Q) > Rcon:
                    continue
                YQ = Xf[Q]
                for ab in ((monsB[:len(monsD)]) if orth else monsD):
                    for c in (e1, e2):
                        divF = mgrad(YQ, ab) @ c - 2.0 * mval(YQ, ab) * (c @ YQ)
                        fsum = 0.0
                        for Xr, E, i in fixed_col[Q]:
                            F = (c - (c @ Xr) * Xr) * mval(Xr, ab)
                            fsum += E @ F
                        for P, n in bycol.get(Q, []):
                            YP = Xf[P]; F = (c - (c @ YP) * YP) * mval(YP, ab)
                            rows += [k, k]; cols += [2 * n, 2 * n + 1]; vals += [-(T[P][0] @ F), -(T[P][1] @ F)]
                        rows.append(k); cols.append(2 * nfree + pid[Q]); vals.append(-divF)
                        dg = sum(ab) if isinstance(ab, tuple) else 0
                        rhs.append(fsum); cw.append(degw ** (p - dg)); k += 1
            A = sp.csr_matrix((vals, (rows, cols)), shape=(k, 2 * nfree + nP))
            b = np.array(rhs); cweight = np.array(cw)

            hs = np.median(np.abs(x0[:2 * nfree][x0[:2 * nfree] != 0])) if np.any(x0[:2 * nfree]) else 1.0
            colscale = np.concatenate([np.full(2 * nfree, hs), np.full(nP, np.mean(x0[2 * nfree:]))])
            As = A @ sp.diags(colscale)
            rn = np.sqrt(np.asarray(As.multiply(As).sum(axis=1)).ravel()) + 1e-300

            rn = rn / np.where(cweight > 0, cweight, 1.0)
            As = (sp.diags(1.0 / rn) @ As).tocsr()
            r0 = (b - A @ x0) / rn


            if solver == "svd":


                Ad = As.toarray()
                Us, sv, Vt = np.linalg.svd(Ad, full_matrices=False)
                del Ad
                keep = sv > max(rcond, 10 ** -6.5) * sv[0]
                lam = sv ** 2
                self.lam_rel = lam / lam[0]
                coef = Us.T @ r0
                dy = Vt[keep].T @ (coef[keep] / sv[keep])
                self.svd = dict(smax=float(sv[0]), smin=float(sv[-1]), nkeep=int(keep.sum()), n=len(sv),
                                incons=float(np.linalg.norm(coef[~keep])), rnorm=float(np.linalg.norm(r0)), solver="svd")
                del Us, Vt
            else:
                Nm = (As @ As.T).toarray()
                lam, U = np.linalg.eigh(Nm)
                del Nm
                keep = lam > max(rcond ** 2, 1e-13) * lam[-1]
                self.lam_rel = lam / lam[-1]
                coef = U.T @ r0
                y = U[:, keep] @ (coef[keep] / lam[keep])
                dy = As.T @ y
                self.svd = dict(smax=float(np.sqrt(lam[-1])), smin=float(np.sqrt(max(lam[0], 0.0))), nkeep=int(keep.sum()), n=len(lam),
                                incons=float(np.linalg.norm(coef[~keep])), rnorm=float(np.linalg.norm(r0)))
                del U
            x = x0 + dy * colscale
            for P in phys:
                nself = free.get((P, P))
                if nself is None:
                    continue
                tot = sum((x[2 * n] * T[P][0] + x[2 * n + 1] * T[P][1] for Q, n in byrow.get(P, [])), np.zeros(3))
                tot = tot + sum((E for Xc, E in fixed_row[P]), np.zeros(3))
                x[2 * nself] -= tot @ T[P][0]; x[2 * nself + 1] -= tot @ T[P][1]
            res = np.abs(A @ x - b) / (rn * cweight)
            wP = x[2 * nfree:]
            self.info.append(dict(npts=len(pts), nphys=nP, nfree=nfree, ncons=k, residual=float(res.max()),
                                  residual_G=float(res[:nG].max()), residual_D=float(res[nG:].max()),
                                  w_ratio=(float((wP / x0[2 * nfree:]).min()), float((wP / x0[2 * nfree:]).max())),
                                  dE_rel=float(np.abs(dy[:2 * nfree]).max())))
            if verbose:
                print(self.info[-1], self.svd, flush=True)
            if np.any(wP <= 0):
                raise ValueError("cone weights not positive")

            for P in phys:
                for Pc in (P, mirror(P)):
                    w[Pc] = wP[pid[P]]
            for (P, Q), n in free.items():
                vec = x[2 * n] * T[P][0] + x[2 * n + 1] * T[P][1]
                j = col_copy[(P, Q)]
                E_corr[(P, j)] = vec
                Pm, jm = mirror(P), mirror(j)
                if Pm != P:
                    E_corr[(Pm, jm)] = vec

            for P in phys:
                for Pc in (P, mirror(P)):
                    pi_, pj = divmod(Pc, M)
                    for di, dj in fdoffs:
                        j = flat(pi_ + di, pj + dj)
                        if j in inP and (Pc, j) not in E_corr:
                            E_corr[(Pc, j)] = np.zeros(3)

            if curl is True:
                for P in phys:
                    YP = Xf[P]
                    pi_, pj = divmod(P, M)
                    cand = {}
                    for di, dj in offs:
                        j = flat(pi_ + di, pj + dj)
                        Q = rep.get(j, j)
                        cand.setdefault(Q, j)
                    Qs = list(cand)
                    k0 = {}
                    for di, dj in fdoffs:
                        j = flat(pi_ + di, pj + dj)
                        Q = rep.get(j, j)
                        k0[Q] = k0.get(Q, 0.0) + K0(P, j)
                    TQ = {}
                    for Q in Qs:
                        Y = Xf[Q]; t1 = e1 - (e1 @ Y) * Y; t1 /= np.linalg.norm(t1); TQ[Q] = (t1, np.cross(Y, t1))
                    y0 = np.zeros(2 * len(Qs))
                    for n, Q in enumerate(Qs):
                        v = k0.get(Q, np.zeros(3)); y0[2 * n] = v @ TQ[Q][0]; y0[2 * n + 1] = v @ TQ[Q][1]
                    Ac, bc = [], []
                    for ab in (monsB if orth else mons):
                        for c in (e1, e2):
                            row = np.zeros(2 * len(Qs))
                            for n, Q in enumerate(Qs):
                                Y = Xf[Q]; F = (c - (c @ Y) * Y) * mval(Y, ab)
                                row[2 * n] = TQ[Q][0] @ F; row[2 * n + 1] = TQ[Q][1] @ F
                            Ac.append(row); bc.append(-(mgrad(YP, ab) @ np.cross(YP, c)))
                    Ac = np.array(Ac); bc = np.array(bc)
                    ks = np.median(np.abs(y0[y0 != 0])) if np.any(y0) else 1.0
                    Acs = Ac * ks; rn = np.linalg.norm(Acs, axis=1) + 1e-300
                    dy, *_ = np.linalg.lstsq(Acs / rn[:, None], (bc - Ac @ y0) / rn, rcond=1e-13)
                    y = y0 + dy * ks
                    self.info[-1].setdefault("residual_C", 0.0)
                    self.info[-1]["residual_C"] = max(self.info[-1]["residual_C"], float(np.abs((Ac @ y - bc) / rn).max()))
                    for Pc, sgn_ in ((P, 1), (mirror(P), 1)):
                        for n, Q in enumerate(Qs):
                            vec = y[2 * n] * TQ[Q][0] + y[2 * n + 1] * TQ[Q][1]
                            j = cand[Q] if Pc == P else mirror(cand[Q])
                            K_corr[(Pc, j)] = vec

                        pci, pcj = divmod(Pc, M)
                        for di, dj in fdoffs:
                            j = flat(pci + di, pcj + dj)
                            if (Pc, j) not in K_corr:
                                K_corr[(Pc, j)] = np.zeros(3)

        gi, gj, gv = [], [], []
        for (i, j), e in E_corr.items():
            gi.append(i); gj.append(j); gv.append(e / w[i] - E0(i, j) / w0[i])

        for i in np.nonzero(w != w0)[0]:
            ii, ij_ = divmod(i, M)
            for dd in fdoffs:
                j = flat(ii + dd[0], ij_ + dd[1])
                if (i, j) not in E_corr:
                    e = E0(i, j)
                    if np.any(e):
                        gi.append(i); gj.append(j); gv.append(e / w[i] - e / w0[i])
        gv = np.array(gv).reshape(-1, 3) / a
        self.Gc = [sp.csr_matrix((gv[:, c], (gi, gj)), shape=(N, N)) for c in range(3)]

        di_, dj_, dv_ = [], [], []
        cols_changed = set(j for (i, j) in E_corr) | set(np.nonzero(w != w0)[0])
        for (i, j), e in E_corr.items():
            di_.append(j); dj_.append(i); dv_.append(-(e / w[j]) + E0(i, j) / w0[j])

        for j in np.nonzero(w != w0)[0]:
            ji, jj = divmod(j, M)
            for dd in fdoffs:
                i = flat(ji - dd[0], jj - dd[1])
                if (i, j) not in E_corr:
                    e = E0(i, j)
                    if np.any(e):
                        di_.append(j); dj_.append(i); dv_.append(-(e / w[j]) + e / w0[j])
        dv_ = np.array(dv_).reshape(-1, 3) / a
        self.Dc = [sp.csr_matrix((dv_[:, c], (di_, dj_)), shape=(N, N)) for c in range(3)]
        if curl == "compat":


            ki, kj = list(di_), list(dj_)
            kv = np.cross(Xf[np.array(dj_, dtype=int)], dv_) if len(dj_) else np.zeros((0, 3))
        else:
            ki, kj, kv = [], [], []
            for (i, j), e in K_corr.items():
                ki.append(i); kj.append(j); kv.append(e - K0(i, j))
            kv = np.array(kv).reshape(-1, 3) / a if kv else np.zeros((0, 3))
        self.curl_mode = curl
        self.Cc = [sp.csr_matrix((kv[:, c], (ki, kj)), shape=(N, N)) for c in range(3)]
        self.w_unit = w
        self.weight = (w * a ** 2).reshape(self.shape)
        self.rows = np.array(sorted(set(gi) | set(di_) | set(ki)))


    def grad(self, phi):
        from .tfd import xp_of
        f = phi.reshape(-1)
        return xp_of(phi).stack([(self.Gc[k] @ f) for k in range(3)]).reshape((3,) + self.shape)

    def div(self, F):
        Ff = F.reshape(3, self.N)
        return sum(self.Dc[k] @ Ff[k] for k in range(3)).reshape(self.shape)

    def curl(self, v):
        vf = v.reshape(3, self.N)
        return sum(self.Cc[k] @ vf[k] for k in range(3)).reshape(self.shape)

    def cast(self, dtype):
        import copy
        C = copy.copy(self)
        C.Gc = [g.astype(dtype) for g in self.Gc]; C.Dc = [d.astype(dtype) for d in self.Dc]
        C.Cc = [c.astype(dtype) for c in self.Cc]; C.weight = self.weight.astype(dtype)
        return C

    def to_device(self):
        import copy, cupy as cp
        C = copy.copy(self)
        C.Gc = [RowOp(g, cp) for g in self.Gc]; C.Dc = [RowOp(d, cp) for d in self.Dc]
        C.Cc = [RowOp(c, cp) for c in self.Cc]; C.weight = cp.asarray(self.weight)
        return C


def cached(S, R=8.0, p=4, rs=4, Rc=None, tag="", shape="disc", pD=None, degw=1.0, curl=True):

    import os, hashlib, pickle
    from . import grid as _grid
    here = os.path.dirname(os.path.abspath(__file__))
    code = hashlib.sha1(open(os.path.join(here, "conehp.py"), "rb").read()).hexdigest()[:10]
    X = np.ascontiguousarray(np.asarray(S.X))
    xh = hashlib.sha1(X.tobytes()).hexdigest()[:12]
    cc = "" if curl is True else f"_curl{curl}"
    fn = os.path.join(_grid.CACHE, f"conehp_{xh}_o{S.order}_R{R:g}_p{p}_pD{pD}_rs{rs}_Rc{Rc}_{shape}_dw{degw:g}{cc}_{code}{tag}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    C = ConeHP(S, R=R, p=p, rs=rs, Rc=Rc, shape=shape, pD=pD, degw=degw, curl=curl)
    os.makedirs(_grid.CACHE, exist_ok=True)
    tmp = f"{fn}.{os.getpid()}.tmp"
    pickle.dump(C, open(tmp, "wb")); os.replace(tmp, fn)
    return C
