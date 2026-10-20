"""Implement the published TRSK scheme on locally generated Voronoi meshes."""


import os, json, time, hashlib
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull

from . import testcases as tc

_HERE = os.path.dirname(os.path.abspath(__file__))


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _tri_area(a, b, c):
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2.0 * np.arctan2(num, den)


def _arc(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("ij,ij->i", a, b))


class VoronoiMesh:

    def __init__(self, P):
        P = _normalize(np.asarray(P, dtype=float))
        N = len(P)
        hull = ConvexHull(P)
        tri = hull.simplices.copy()
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        nrm = np.cross(b - a, c - a)
        flip = np.einsum("ij,ij->i", nrm, a + b + c) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        V = _normalize(np.cross(b - a, c - a))
        T = len(tri)

        d0 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
        d1 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
        dt = np.tile(np.arange(T), 3)
        lo, hi = np.minimum(d0, d1), np.maximum(d0, d1)
        key = lo.astype(np.int64) * N + hi
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        assert np.all(key_s[0::2] == key_s[1::2]), "every edge must be shared by two triangles"
        first, second = order[0::2], order[1::2]
        E = len(first)
        c0 = lo[first]; c1 = hi[first]

        fwd_first = d0[first] == c0
        tl = np.where(fwd_first, dt[first], dt[second])
        tr = np.where(fwd_first, dt[second], dt[first])
        self.N, self.E, self.V = N, E, T
        self.P, self.Pv, self.tri = P, V, tri
        self.cellsOnEdge = np.stack([c0, c1], 1)
        self.verticesOnEdge = np.stack([tr, tl], 1)
        xe = _normalize(P[c0] + P[c1])
        ne = _normalize(P[c1] - P[c0])
        te = np.cross(xe, ne)
        self.xe, self.ne, self.te = xe, ne, te
        self.dc = _arc(P[c0], P[c1])
        sgn = np.sign(np.einsum("ij,ij->i", V[tl] - V[tr], te))
        self.dv = sgn * _arc(V[tr], V[tl])

        A = np.zeros(N)
        np.add.at(A, c0, _tri_area(P[c0], V[tr], V[tl]))
        np.add.at(A, c1, _tri_area(P[c1], V[tl], V[tr]))
        self.areaCell = A

        kite = np.zeros((T, 3))
        for k in range(3):
            pk, pn, pp = P[tri[:, k]], P[tri[:, (k + 1) % 3]], P[tri[:, (k - 1) % 3]]
            m1, m2 = _normalize(pk + pn), _normalize(pk + pp)
            kite[:, k] = _tri_area(pk, m1, V) + _tri_area(pk, V, m2)
        self.kite = kite
        self.areaTriangle = kite.sum(1)
        self._cell_lists()
        self._weights()

    def _cell_lists(self):
        N, E, P = self.N, self.E, self.P
        c0, c1 = self.cellsOnEdge.T
        cell = np.concatenate([c0, c1])
        edge = np.concatenate([np.arange(E), np.arange(E)])

        e_hat, n_hat = tc.east_north(P[cell].T)
        d = self.xe[edge] - P[cell]
        ang = np.arctan2(np.einsum("ij,ji->i", d, n_hat), np.einsum("ij,ji->i", d, e_hat))
        order = np.lexsort((ang, cell))
        cell_s, edge_s = cell[order], edge[order]
        nE = np.bincount(cell_s, minlength=N)
        maxE = nE.max()
        start = np.concatenate([[0], np.cumsum(nE)[:-1]])
        pos = np.arange(len(cell_s)) - start[cell_s]
        eoc = -np.ones((N, maxE), dtype=np.int64)
        eoc[cell_s, pos] = edge_s
        self.nEdgesOnCell, self.edgesOnCell, self.maxEdges = nE, eoc, maxE

        sig = np.zeros((N, maxE))
        valid = eoc >= 0
        ee = np.where(valid, eoc, 0)
        sig[valid] = np.where(self.cellsOnEdge[ee[valid], 0] == np.nonzero(valid)[0], 1.0, -1.0)
        self.edgeSignOnCell = sig

        voc = -np.ones((N, maxE), dtype=np.int64)
        for k in range(maxE):
            kn = np.where(k + 1 < nE, k + 1, 0)
            has = k < nE
            ea, eb = eoc[has, k], eoc[np.nonzero(has)[0], kn[has]]
            va, vb = self.verticesOnEdge[ea], self.verticesOnEdge[eb]
            common = np.where((va[:, 0] == vb[:, 0]) | (va[:, 0] == vb[:, 1]), va[:, 0], va[:, 1])
            ok = (common == vb[:, 0]) | (common == vb[:, 1])
            assert ok.all(), "adjacent edges of a cell must share a vertex"
            voc[has, k] = common
        self.verticesOnCell = voc

        R = np.zeros((N, maxE))
        for k in range(maxE):
            has = k < nE
            v = voc[has, k]
            cidx = np.nonzero(has)[0]
            which = np.argmax(self.tri[v] == cidx[:, None], axis=1)
            assert np.all(self.tri[v, which] == cidx)
            R[has, k] = self.kite[v, which] / self.areaCell[cidx]
        self.Rkite = R

    def _weights(self):


        rows, cols, vals = [], [], []
        N, maxE = self.N, self.maxEdges
        nE, eoc, sig, R = self.nEdgesOnCell, self.edgesOnCell, self.edgeSignOnCell, self.Rkite
        for k0 in range(maxE):
            has = k0 < nE
            cells = np.nonzero(has)[0]
            n = nE[cells]
            e = eoc[cells, k0]
            s_e = sig[cells, k0]
            acc = np.zeros(len(cells))
            for j in range(1, maxE):
                act = j < n
                kk = (k0 + j) % np.maximum(n, 1)
                kv = (k0 + j - 1) % np.maximum(n, 1)
                acc = acc + np.where(act, R[cells, kv], 0.0)
                ep = eoc[cells, kk]
                w = s_e * sig[cells, kk] * (0.5 - acc)
                rows.append(e[act]); cols.append(ep[act]); vals.append(w[act])
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        self.W = sp.csr_matrix((vals, (rows, cols)), shape=(self.E, self.E))


class TRSK:

    def __init__(self, mesh, case, a=tc.A):
        m = self.m = mesh
        self.a = a
        N, E, V = m.N, m.E, m.V
        c0, c1 = m.cellsOnEdge.T
        v0, v1 = m.verticesOnEdge.T
        dc, dv = m.dc * a, m.dv * a
        A, Av = m.areaCell * a * a, m.areaTriangle * a * a
        self.dc, self.dv, self.A, self.Av = dc, dv, A, Av
        eidx = np.arange(E)

        self.Div = sp.csr_matrix((np.concatenate([dv / A[c0], -dv / A[c1]]),
                                  (np.concatenate([c0, c1]), np.concatenate([eidx, eidx]))), shape=(N, E))

        self.Grad = sp.csr_matrix((np.concatenate([-1 / dc, 1 / dc]), (np.concatenate([eidx, eidx]),
                                  np.concatenate([c0, c1]))), shape=(E, N))

        self.Curl = sp.csr_matrix((np.concatenate([dc / Av[v1], -dc / Av[v0]]),
                                   (np.concatenate([v1, v0]), np.concatenate([eidx, eidx]))), shape=(V, E))

        self.Hv = sp.csr_matrix(((m.kite * (a * a) / Av[:, None]).ravel(),
                                 (np.repeat(np.arange(V), 3), m.tri.ravel())), shape=(V, N))
        self.He = sp.csr_matrix((np.full(2 * E, 0.5), (np.concatenate([eidx, eidx]), np.concatenate([c0, c1]))),
                                shape=(E, N))
        self.Qe = sp.csr_matrix((np.full(2 * E, 0.5), (np.concatenate([eidx, eidx]), np.concatenate([v0, v1]))),
                                shape=(E, V))
        self.Ke = sp.csr_matrix((np.concatenate([dc * dv / (4 * A[c0]), dc * dv / (4 * A[c1])]),
                                 (np.concatenate([c0, c1]), np.concatenate([eidx, eidx]))), shape=(N, E))

        self.Wq = (sp.diags(1 / dc) @ m.W @ sp.diags(dv)).tocsr()
        self.fv = case.coriolis(m.Pv.T)
        self.b = case.topography(m.P.T)
        self.case = case

    def initial(self):
        m = self.m
        h = self.case.height(m.P.T) - self.b
        u = np.einsum("ij,ji->i", m.ne, self.case.velocity(m.xe.T))
        return h, u

    def tendency(self, h, u):
        he = self.He @ h
        F = he * u
        dh = -(self.Div @ F)
        zeta = self.Curl @ u
        q = (zeta + self.fv) / (self.Hv @ h)
        qe = self.Qe @ q
        perp = 0.5 * (qe * (self.Wq @ F) + self.Wq @ (qe * F))
        K = self.Ke @ (u * u)
        du = perp - self.Grad @ (K + tc.G * (h + self.b))
        return dh, du

    def energy(self, h, u):
        return float((self.A * tc.G * h * (0.5 * h + self.b)).sum() + 0.5 * (self.dc * self.dv * (self.He @ h) * u * u).sum())


def icosahedral(level):
    t = (1 + 5 ** 0.5) / 2
    P = _normalize(np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                             [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float))
    F = ConvexHull(P).simplices
    for _ in range(level):
        pts = list(P)
        cache = {}
        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                cache[k] = len(pts)
                pts.append(_normalize(P[i] + P[j]))
            return cache[k]
        newF = []
        for i, j, k in F:
            a, b, c = mid(i, j), mid(j, k), mid(k, i)
            newF += [[i, a, c], [j, b, a], [k, c, b], [a, b, c]]
        P = np.array(pts); F = np.array(newF)
    return P


def lloyd(P, iters, fixed=None, verbose=False, tol=None, check=50):

    P = _normalize(P.copy())
    h = np.sqrt(4 * np.pi / len(P))
    Pc = P.copy()
    for it in range(iters):
        hull = ConvexHull(P)
        tri = hull.simplices.copy()
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        nrm = np.cross(b - a, c - a)
        flip = np.einsum("ij,ij->i", nrm, a + b + c) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        V = _normalize(np.cross(b - a, c - a))

        C = np.zeros_like(P)
        for k in range(3):
            pk, pn, pp = P[tri[:, k]], P[tri[:, (k + 1) % 3]], P[tri[:, (k - 1) % 3]]
            m1, m2 = _normalize(pk + pn), _normalize(pk + pp)
            a1 = _tri_area(pk, m1, V); a2 = _tri_area(pk, V, m2)
            np.add.at(C, tri[:, k], a1[:, None] * (pk + m1 + V) / 3 + a2[:, None] * (pk + V + m2) / 3)
        Pn = _normalize(C)
        if fixed is not None:
            Pn[fixed] = P[fixed]
        P = Pn
        if (it + 1) % check == 0:
            mv = _arc(Pc, P).max() / h
            if verbose:
                print(it + 1, mv, flush=True)
            if tol is not None and mv < tol:
                break
            Pc = P.copy()
    return P


def refine(P):
    tri = ConvexHull(P).simplices
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([P, _normalize(P[e[:, 0]] + P[e[:, 1]])])


LLOYD_TOL = 1e-4
LLOYD_MAX = 6000


def icos_mesh(level):

    from .grid import CACHE
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"icos_scvt_L{level}_{MESH_TAG}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    if level <= 2:
        P = icosahedral(level)
    else:
        P = refine(icos_mesh(level - 1))
    P = lloyd(P, LLOYD_MAX, tol=LLOYD_TOL)
    tmp = fn + f".{os.getpid()}.tmp.npy"
    np.save(tmp, P); os.replace(tmp, fn)
    return P


def torus_mesh(M):

    from . import grid
    X = grid.torus_X(M)
    I, J = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    TI, TJ = (-1 - I) % M, (-1 - J) % M
    rep = (I + J * M) < (TI + TJ * M)
    P = X[:, rep].T
    sites = np.stack([np.stack([I[rep], J[rep]], 1), np.stack([TI[rep], TJ[rep]], 1)], 1)
    return P, sites


def mesh_points(spec):
    kind, n = spec
    if kind == "torus":
        return torus_mesh(n)[0]
    if kind == "icos":
        return icos_mesh(n)
    raise ValueError(kind)


def _mesh_tag():
    import inspect
    src = "".join(inspect.getsource(f) for f in (_normalize, _tri_area, _arc, icosahedral, lloyd, refine))
    return hashlib.sha1((src + repr((LLOYD_TOL, LLOYD_MAX))).encode()).hexdigest()[:10]


MESH_TAG = _mesh_tag()


class Solver:

    def __init__(self, mesh, case, integrator="rk4", advect_only=False):
        self.S = TRSK(mesh, case)
        self.integrator = integrator
        self.advect_only = advect_only
        if advect_only:
            m = mesh
            self.u_adv = np.einsum("ij,ji->i", m.ne, case.velocity(m.xe.T))

    def tendency(self, h, u):
        if self.advect_only:
            return -(self.S.Div @ ((self.S.He @ h) * self.u_adv)), np.zeros_like(u)
        return self.S.tendency(h, u)

    def step(self, h, u, dt):
        f = self.tendency
        if self.integrator == "rk4":
            a1 = f(h, u); a2 = f(h + 0.5 * dt * a1[0], u + 0.5 * dt * a1[1])
            a3 = f(h + 0.5 * dt * a2[0], u + 0.5 * dt * a2[1]); a4 = f(h + dt * a3[0], u + dt * a3[1])
            return (h + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
                    u + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]))
        if self.integrator.startswith("kg"):
            from .tfd import kg_coeffs
            hs, us = h, u
            for ck in kg_coeffs(self.integrator[2:]):
                dh, du = f(hs, us)
                hs, us = h + ck * dt * dh, u + ck * dt * du
            return hs, us
        raise ValueError(self.integrator)

    def stages(self):
        if self.integrator == "rk4":
            return 4
        import re
        return int(re.match(r"\d+", self.integrator[2:]).group())


def cmax_of(S, h, u, advect_only=False):
    vmax = float(np.linalg.norm(S.case.velocity(S.m.P.T), axis=0).max())
    return vmax if advect_only else float(np.sqrt(tc.G * h.max()) + vmax)


def run(spec, name, scheme, days=None, by_day=None, verbose=False):


    from . import experiments as ex, lwgg, reference
    sc = dict(integrator="rk4", courant=0.5)
    sc.update(scheme or {})
    case = ex.make_case(name)
    days = case.days if days is None else days
    P = mesh_points(spec)
    m = VoronoiMesh(P)
    adv = name.startswith("tc1")
    sol = Solver(m, case, sc["integrator"], advect_only=adv)
    S = sol.S
    h, u = S.initial()
    if adv:
        h = case.height(m.P.T)
    hmean = float(S.dc.mean())
    cmax = cmax_of(S, h, u, adv)
    dt0 = sc["courant"] * hmean / cmax
    T = days * tc.DAY
    want = sorted(set(by_day or []) | {days})
    rec, t, ok = {}, 0.0, True
    t0 = time.time()
    m0, e0 = float((S.A * h).sum()), (None if adv else S.energy(h, u))
    nsteps = 0
    for d in want:
        Td = d * tc.DAY
        n = int(np.ceil((Td - t) / dt0 - 1e-9))
        dt = (Td - t) / n
        for k in range(n):
            h, u = sol.step(h, u, dt)
            if k % 50 == 0 and not np.isfinite(h).all():
                ok = False; break
        nsteps += n
        t = Td
        if not ok:
            break
        if d != days:
            href = ex.reference_height(name, case, m.P.T, Td)
            rec[d] = float(lwgg.norms(h + (0 if adv else S.b), href, S.A)[1])
    out = dict(solver="trsk", mesh=spec[0], grid=list(spec), case=name, days=days, scheme=sc, dt=T / max(nsteps, 1),
               nsteps=nsteps, hmean_m=hmean, cmax=cmax, wall_s=time.time() - t0, stable=bool(ok and np.isfinite(h).all()),
               npts=(2 * m.N if spec[0] == "torus" else m.N), nunique=m.N, ncells=m.N, nedges=m.E)
    if out["stable"]:
        hs = h + (0 if adv else S.b)
        href = ex.reference_height(name, case, m.P.T, T)
        l1, l2, li = lwgg.norms(hs, href, S.A)
        w = S.A
        out.update(l1=l1, l2=l2, linf=li, mass_rel=(float((S.A * h).sum()) - m0) / m0,
                   rms_err_m=float(np.sqrt((w * (hs - href) ** 2).sum() / w.sum())))
        if e0 is not None:
            out["energy_rel"] = (S.energy(h, u) - e0) / e0
        if by_day:
            rec[days] = l2
            out["l2_by_day"] = {f"{k:g}": v for k, v in sorted(rec.items())}
        for tag, alt in ex.REFS_ALT.get(name.lower(), {}).items():
            if os.path.exists(alt):
                try:
                    ha = reference.height_at(alt, m.P.T, T / 3600.0)
                    out[f"l2_vs_{tag}"] = float(lwgg.norms(hs, ha, w)[1])
                except KeyError:
                    pass
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "scheme"}, default=float))
    return out


def _hexdist(di, dj):
    return np.where(di * dj >= 0, np.abs(di) + np.abs(dj), np.maximum(np.abs(di), np.abs(dj)))


def balanced_owner(m, cap=3):

    from scipy.sparse.csgraph import maximum_flow
    N, E = m.N, m.E
    src, snk = 0, 1 + E + N
    eidx = np.arange(E)
    rows = np.concatenate([np.zeros(E, int), 1 + eidx, 1 + eidx, 1 + E + np.arange(N)])
    cols = np.concatenate([1 + eidx, 1 + E + m.cellsOnEdge[:, 0], 1 + E + m.cellsOnEdge[:, 1], np.full(N, snk)])
    caps = np.concatenate([np.ones(3 * E, np.int32), np.full(N, cap, np.int32)])
    G = sp.csr_matrix((caps, (rows, cols)), shape=(snk + 1, snk + 1))
    r = maximum_flow(G, src, snk, method="dinic")
    if r.flow_value != E:
        raise RuntimeError("no orientation with at most %d edges per cell" % cap)
    F = r.flow.tocsr()
    to0 = np.asarray(F[1 + eidx, 1 + E + m.cellsOnEdge[:, 0]]).ravel() > 0
    return np.where(to0, m.cellsOnEdge[:, 0], m.cellsOnEdge[:, 1])


def torus_layout(M, S=None):


    P, sites = torus_mesh(M)
    if S is None:
        S = TRSK(VoronoiMesh(P), __import__("polarswe.testcases", fromlist=["x"]).TC5())
    m = S.m
    N, E = m.N, m.E
    B = lambda A: (abs(A) > 0).astype(np.int8).tocsr()
    Div, He, Curl, Hv, Qe, Wq, Ke, Grad = map(B, (S.Div, S.He, S.Curl, S.Hv, S.Qe, S.Wq, S.Ke, S.Grad))
    Qu, Qh = B(Qe @ Curl), B(Qe @ Hv)
    dep_hu, dep_hh = Div, B(Div @ He)
    dep_uu = B(Wq + Qu + Wq @ Qu + Grad @ Ke)
    dep_uh = B(Wq @ He + Qh + Wq @ Qh + Grad)
    s0 = sites[:, 0, :]
    owner = balanced_owner(m)
    cell_site = s0
    edge_site = s0[owner]
    def offsets(rows_site, cols_sites_both, A):
        A = A.tocoo()
        r = rows_site[A.row]
        best = None
        for c in range(2):
            cs = cols_sites_both[A.col, c, :]
            d = (cs - r + M // 2) % M - M // 2
            hd = _hexdist(d[:, 0], d[:, 1])
            if best is None:
                best, bd = hd, d
            else:
                take = hd < best
                best = np.where(take, hd, best); bd = np.where(take[:, None], d, bd)
        return bd, best
    edge_sites_both = sites[owner]
    res = dict(di=0, dj=0, hex=0)
    for rows_site, cols, A in ((cell_site, edge_sites_both, dep_hu), (cell_site, sites, dep_hh),
                               (edge_site, edge_sites_both, dep_uu), (edge_site, sites, dep_uh)):
        d, hd = offsets(rows_site, cols, A)
        res["di"] = max(res["di"], int(np.abs(d[:, 0]).max())); res["dj"] = max(res["dj"], int(np.abs(d[:, 1]).max()))
        res["hex"] = max(res["hex"], int(hd.max()))
    owned = np.bincount(owner, minlength=N)
    res.update(reach_stage=max(res["di"], res["dj"]), owned_edges_max=int(owned.max()), owned_edges_mean=float(owned.mean()),
               max_edges_on_cell=int(m.maxEdges), cell_edges_hist=np.bincount(m.nEdgesOnCell).tolist())
    return res


def flops_per_cell_stage(S):

    N, E, V = S.m.N, S.m.E, S.m.V
    nnz = lambda A: A.nnz
    f = (2 * nnz(S.He) + E
         + 2 * nnz(S.Div)
         + 2 * nnz(S.Curl) + V + 2 * nnz(S.Hv) + V
         + 2 * nnz(S.Qe)
         + 2 * 2 * nnz(S.Wq) + 2 * E + 2 * E
         + E + 2 * nnz(S.Ke)
         + 3 * N + 2 * nnz(S.Grad) + E)
    return f / N


def static_words_per_site(S):


    m = S.m
    maxE = m.maxEdges
    wrow = np.diff(S.Wq.indptr).max()
    return int(2 * maxE + 1 + 3 * (1 + wrow) + 2 * (3 + 3 + 1))
