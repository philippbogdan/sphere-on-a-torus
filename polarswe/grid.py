"""Build the double-cover grid. X has shape (3, M, M) on the unit sphere."""


import os, hashlib
import numpy as np
from . import ellip, moser, torusmap

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_TAG = hashlib.sha1(b"".join(open(os.path.join(_HERE, f), "rb").read()
                                 for f in ("ellip.py", "moser.py", "torusmap.py", "grid.py"))).hexdigest()[:10]


def _save(fn, X):
    tmp = f"{fn[:-4]}.{os.getpid()}.tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, fn)


def torus_X(M, nsteps=200, Nf=256, verbose=False):
    """Map an even M by M lattice to unit-sphere positions, shape (3, M, M).

    Sites (i, j) and (-1-i, -1-j), modulo M, share a position.
    Cached maps are keyed by the source files that build them.
    """
    os.makedirs(CACHE, exist_ok=True)
    tag = "" if ellip.LATTICE == "hex" else f"_{ellip.LATTICE}"
    fn = os.path.join(CACHE, f"torusX_ea{tag}_M{M}_{CODE_TAG}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    i = np.arange(M)
    I, J = np.meshgrid(i, i, indexing="ij")
    zeta = ((I + 0.5) + (J + 0.5) * ellip.EPS) / M
    F = moser.MoserField(Nf=Nf)

    half = (I + J * M) < ((-1 - I) % M + ((-1 - J) % M) * M)
    selfsym = (I == (-1 - I) % M) & (J == (-1 - J) % M)
    todo = half | selfsym
    z = np.empty_like(zeta)
    z[todo] = F.inverse_map(zeta[todo], nsteps)
    other = ~todo
    z[other] = -z[(-1 - I[other]) % M, (-1 - J[other]) % M]
    X = torusmap.stereo(ellip.wp(z) / torusmap.SCALE)
    X /= np.linalg.norm(X, axis=0)
    _save(fn, X)
    return X


def torus_X_offset(M, offset, nsteps=200, Nf=256):

    if offset == 0.5:
        return torus_X(M, nsteps, Nf)
    os.makedirs(CACHE, exist_ok=True)
    tag = "" if ellip.LATTICE == "hex" else f"_{ellip.LATTICE}"
    fn = os.path.join(CACHE, f"torusX_ea{tag}_M{M}_off{offset:g}_{CODE_TAG}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    i = np.arange(M)
    I, J = np.meshgrid(i, i, indexing="ij")
    zeta = ((I + offset) + (J + offset) * ellip.EPS) / M
    F = moser.MoserField(Nf=Nf)
    z = F.inverse_map(zeta, nsteps)
    w = ellip.wp(z) / torusmap.SCALE
    X = torusmap.stereo(w)
    X /= np.linalg.norm(X, axis=0)
    _save(fn, X)
    return X


def branch_cells(M):

    h = M // 2
    return [(M - 1, M - 1), (h - 1, M - 1), (M - 1, h - 1), (h - 1, h - 1)]


def latlon_spacing_eq(npoints):
    nlat = np.sqrt(npoints / 2.0)
    return np.pi / nlat


def dfs_latlon_X(nlon, nlat):


    lam = (np.arange(nlon) + 0.5) * 2 * np.pi / nlon
    th = -np.pi + (np.arange(2 * nlat) + 0.5) * np.pi / nlat
    L, T = np.meshgrid(lam, th, indexing="ij")
    return np.stack([np.sin(T) * np.cos(L), np.sin(T) * np.sin(L), np.cos(T)])


def _rot_to_z(m):
    m = m / np.linalg.norm(m); z = np.array([0, 0, 1.0]); ax = np.cross(m, z); s_ = np.linalg.norm(ax); c = m @ z
    if s_ < 1e-14:
        return np.eye(3)
    ax /= s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s_ * K + (1 - c) * K @ K


def orientation(name):


    if name in (None, "", "vertex"):
        return np.eye(3)
    import math
    from scipy.spatial.transform import Rotation
    lat = math.radians(-19.4712206)
    V = np.array([[0, 0, 1.0]] + [[math.cos(lat) * math.cos(math.radians(l)), math.cos(lat) * math.sin(math.radians(l)),
                                   math.sin(lat)] for l in (0, 120, -120)])
    if name == "edge":
        return _rot_to_z(V[0] + V[1])
    if name == "generic":
        return Rotation.from_euler("zyx", [17, 43, 29], degrees=True).as_matrix()
    if name == "generic2":
        return Rotation.from_euler("zyx", [-71, 23, 58], degrees=True).as_matrix()
    raise ValueError(name)


def oriented(X, name):
    R = orientation(name)
    return X if name in (None, "", "vertex") else np.ascontiguousarray(np.tensordot(R, X, axes=1))


def branch_positions(orient=None):

    with np.errstate(all="ignore"):
        B = torusmap.stereo(ellip.wp(np.array(ellip.BRANCH_POINTS, dtype=complex)) / torusmap.SCALE)
    B = B / np.linalg.norm(B, axis=0)
    return (orientation(orient) @ B).T


def branch_distance_deg(X, orient=None):
    B = branch_positions(orient)
    return np.degrees(np.arccos(np.clip(np.tensordot(B, X, axes=1).max(0), -1.0, 1.0)))


def torus_X_beta(M, beta, nsteps=200, Nf=256):

    if beta == 0:
        return torus_X(M, nsteps, Nf)
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"torusX_beta{beta:g}_M{M}_{CODE_TAG}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    i = np.arange(M)
    I, J = np.meshgrid(i, i, indexing="ij")
    zeta = ((I + 0.5) + (J + 0.5) * ellip.EPS) / M
    F = moser.MoserField(Nf=Nf, beta=beta)
    half = (I + J * M) < ((-1 - I) % M + ((-1 - J) % M) * M)
    selfsym = (I == (-1 - I) % M) & (J == (-1 - J) % M)
    todo = half | selfsym
    z = np.empty_like(zeta)
    z[todo] = F.inverse_map(zeta[todo], nsteps)
    other = ~todo
    z[other] = -z[(-1 - I[other]) % M, (-1 - J[other]) % M]
    X = torusmap.stereo(ellip.wp(z) / torusmap.SCALE)
    X /= np.linalg.norm(X, axis=0)
    _save(fn, X)
    return X
