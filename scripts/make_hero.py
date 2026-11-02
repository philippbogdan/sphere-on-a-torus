"""Render the saved Galewsky vorticity as a sphere beside its periodic torus.

Write GIF, H.264 MP4 and provenance JSON to figures/. The input must contain
the full M = 512, six-day case described in figure-pack/hero-input.json.
Requires the figures extra, SciPy, ffmpeg, ffprobe and gifsicle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle, Polygon
from matplotlib.tri import Triangulation
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

from make_figures import audit

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "figure-pack"
OUT = ROOT / "figures"

DAY = 86400.0
DAYS = 6.0
VMAX = 1.6e-4
TICKS = [-1.5e-4, -0.75e-4, 0.0, 0.75e-4, 1.5e-4]
LEVELS = 199
SHIFT = 15 / 32
VIEW_LON, VIEW_LAT = 0.0, 35.0

W, H = 1000, 560
SS = 2
FONTS = ["Arial", "DejaVu Sans"]
INK = "#1A1A1A"
SOFT = "#3D3D3D"
GUIDE = "#8C8C8C"
CHARCOAL = "#333333"
GOLD = "#F2C12E"
RIM = "#1A1A1A"
BLUE = "#0072B2"
RUST = "#D55E00"

SPHERE_C = (208.0, 244.0)
SPHERE_R = 196.0
TORUS_S = 372.0
TORUS_X0 = 424.0
TORUS_Y0 = 405.0
CBAR = (48.0, 472.0, 320.0, 12.0)
TAXIS = (500.0, 478.0, 440.0)
LEGEND = (882.0, 396.0)

WORDS = {
    "sphere": "Sphere",
    "torus": "Periodic torus",
    "colour_bar": "Relative vorticity (1/s)",
    "legend": "Cone point",
    "time_axis": "Simulated time (days; variable playback speed)",
}
SEC3 = math.sqrt(3.0)


def px(value):
    """Convert design pixels to Matplotlib points at the reference resolution."""
    return value * 72 / 96


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def srgb_from_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def linear_from_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def lab_from_hex(colour):
    rgb = np.array([int(colour[k:k + 2], 16) / 255 for k in (1, 3, 5)])
    xyz = linear_from_srgb(rgb) @ XYZ.T / WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def linear_from_lab(L, a, b):
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * WHITE
    return xyz @ np.linalg.inv(XYZ).T


def half_ramp(colour, s, light=95.0, dark=24.0, peak=70.0, at=0.55, end=45.0):
    """Build one colour-map arm, reducing chroma where needed to stay in sRGB."""
    lab = lab_from_hex(colour)
    hue = math.atan2(lab[2], lab[1])
    L = light - (light - dark) * s
    target = np.where(s <= at, peak * s / at, peak + (end - peak) * (s - at) / (1 - at))
    full = linear_from_lab(L, target * math.cos(hue), target * math.sin(hue))
    ok = ((full >= -1e-9) & (full <= 1 + 1e-9)).all(-1)
    lo, hi = np.where(ok, target, 0.0), target.copy()
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        lin = linear_from_lab(L, mid * math.cos(hue), mid * math.sin(hue))
        inside = ((lin >= -1e-9) & (lin <= 1 + 1e-9)).all(-1)
        lo = np.where(ok | ~inside, lo, mid)
        hi = np.where(ok | inside, hi, mid)
    rgb = srgb_from_linear(linear_from_lab(L, lo * math.cos(hue), lo * math.sin(hue)))
    return rgb, L, lo


def diverging(levels):
    """Return a blue-to-rust palette with matching lightness on both sides."""
    s = np.linspace(-1.0, 1.0, levels)
    rgb = np.zeros((levels, 3))
    lightness = np.zeros(levels)
    neg, pos = s < 0, s >= 0
    rgb[neg], lightness[neg], _ = half_ramp(BLUE, -s[neg])
    rgb[pos], lightness[pos], _ = half_ramp(RUST, s[pos])
    return rgb, lightness


def view_basis(lon, lat):
    """Return east, screen-up and outward view directions for the fixed globe."""
    lon, lat = math.radians(lon), math.radians(lat)
    e3 = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    e1 = np.array([-math.sin(lon), math.cos(lon), 0.0])
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def load(archive):
    """Read the saved geometry, fields, physical times and numerical metadata."""
    data = np.load(archive)
    meta = json.loads(str(data["meta"]))
    X = data["X"]
    t = data["t_s"].astype(np.float64)
    branch = data["branch_distance_cells"]
    vort = data["vorticity"]
    height = data["height"]
    return data, meta, X, t, branch, vort, height


def verify(meta, X, t, branch, vort, height, expected):
    """Check the full case, snapshot timing, finite fields and fixed colour range.

    Numerical metadata must match hero-input.json. Archive-digest agreement is
    recorded separately in the provenance; it is not enforced here.
    """
    M = int(meta["M"])
    for key in ("case", "M", "scheme", "dt_s", "nsteps", "mass_rel", "mean_spacing_m", "branch_points_ij"):
        if meta[key] != expected[key]:
            raise SystemExit(f"meta.{key} {meta[key]!r} != figure-pack {expected[key]!r}")
    if meta["case"] != "galewsky" or M != 512:
        raise SystemExit("archive is not the M = 512 Galewsky case")
    F = len(t)
    if X.shape != (3, M, M) or vort.shape != (F, M, M) or height.shape != (F, M, M) or branch.shape != (M, M):
        raise SystemExit("array shapes do not match")
    stride = max(1, int(round(1800.0 / meta["dt_s"])))
    nsteps = int(meta["nsteps"])
    steps = list(range(0, nsteps + 1, stride))
    if steps[-1] != nsteps:
        steps.append(nsteps)
    if F != len(steps) or F != expected["frames"]:
        raise SystemExit(f"archive has {F} frames, expected {len(steps)}")
    if not np.allclose(t, np.array(steps) * meta["dt_s"], rtol=1e-12, atol=1e-6):
        raise SystemExit("t_s does not match the solver steps")
    if t[0] != 0.0 or not (np.diff(t) > 0).all():
        raise SystemExit("t_s must start at zero and increase")
    if not (math.isclose(t[-1], nsteps * meta["dt_s"], rel_tol=1e-12) and math.isclose(t[-1], DAYS * DAY, rel_tol=1e-12)):
        raise SystemExit("final time is not six days")
    for name, arr in (("X", X), ("vorticity", vort), ("height", height), ("branch_distance_cells", branch)):
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name} is not finite")
    vmin, vmax = float(vort.min()), float(vort.max())
    if max(abs(vmin), abs(vmax)) >= VMAX:
        raise SystemExit("vorticity exceeds the fixed colour scale")
    return {
        "frames": F,
        "first_t_s": float(t[0]),
        "last_t_s": float(t[-1]),
        "last_t_days": float(t[-1] / DAY),
        "dt_s": meta["dt_s"],
        "nsteps": nsteps,
        "regular_snapshot_step_stride": stride,
        "last_snapshot_step_stride": steps[-1] - steps[-2],
        "regular_snapshot_interval_s": float(stride * meta["dt_s"]),
        "last_snapshot_interval_s": float(t[-1] - t[-2]),
        "mass_rel": meta["mass_rel"],
        "mean_spacing_m": meta["mean_spacing_m"],
        "vorticity_min_per_s": vmin,
        "vorticity_max_per_s": vmax,
        "vorticity_max_abs_per_s": max(abs(vmin), abs(vmax)),
        "vorticity_max_abs_frame": int(np.maximum(vort.max(axis=(1, 2)), -vort.min(axis=(1, 2))).argmax()),
        "height_min_m": float(height.min()),
        "height_max_m": float(height.max()),
        "height_frames": int(height.shape[0]),
        "all_fields_finite": True,
    }


def cones(meta, X, branch):
    """Locate cone markers in both views and check their neighbouring distances."""
    M = int(meta["M"])
    e1, e2, e3 = view_basis(VIEW_LON, VIEW_LAT)
    rows = []
    near = set()
    for bi, bj in meta["branch_points_ij"]:
        ii = [int(math.floor(bi)) % M, int(math.ceil(bi)) % M]
        jj = [int(math.floor(bj)) % M, int(math.ceil(bj)) % M]
        q = sum(X[:, a, b].astype(np.float64) for a in ii for b in jj)
        q = q / np.linalg.norm(q)
        u = ((bj + 0.5) / M + SHIFT) % 1.0
        v = ((bi + 0.5) / M + SHIFT) % 1.0
        d = [float(branch[a, b]) for a in ii for b in jj]
        near.update((a, b) for a in ii for b in jj)
        rows.append({
            "ij": [bi, bj],
            "u": u,
            "v": v,
            "torus_x": u + v / 2,
            "torus_y": SEC3 * v / 2,
            "sphere_xyz": [float(c) for c in q],
            "lat_deg": math.degrees(math.asin(max(-1.0, min(1.0, q[2])))),
            "lon_deg": math.degrees(math.atan2(q[1], q[0])),
            "view_x": float(q @ e1),
            "view_y": float(q @ e2),
            "front_facing": bool(q @ e3 > 0),
            "neighbour_branch_distance_cells": d,
        })
    small = set(zip(*np.nonzero(branch < 1.0)))
    small = {(int(a), int(b)) for a, b in small}
    check = {
        "cells_with_branch_distance_below_1": len(small),
        "all_at_cone_neighbours": small == near,
        "neighbour_distance_max": max(max(r["neighbour_branch_distance_cells"]) for r in rows),
        "global_min_distance": float(branch.min()),
    }
    if not check["all_at_cone_neighbours"]:
        raise SystemExit("cone points do not match the branch-distance minima")
    north = max(rows, key=lambda r: r["lat_deg"])
    if not (math.isclose(north["u"], SHIFT) and math.isclose(north["v"], SHIFT)):
        raise SystemExit("the north-pole cone is not at (15/32, 15/32)")
    return rows, check


def sphere_weights(X, size):
    """Precompute interpolation on visible triangles of the unique sphere samples."""
    M = X.shape[1]
    # The other half of the index grid contains the paired copy of each sample.
    unique = X[:, : M // 2, :].reshape(3, -1).T.astype(np.float64)
    e1, e2, e3 = view_basis(VIEW_LON, VIEW_LAT)
    hull = ConvexHull(unique)
    simp = hull.simplices.copy()
    p0, p1, p2 = unique[simp[:, 0]], unique[simp[:, 1]], unique[simp[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    flip = np.einsum("ij,ij->i", normal, p0 + p1 + p2) < 0
    simp[flip] = simp[flip][:, [0, 2, 1]]
    normal[flip] *= -1
    a, b, c = unique @ e1, unique @ e2, unique @ e3
    keep = (c[simp] > 0).all(1) & (normal @ e3 > 0)
    tri = simp[keep]
    used = np.unique(tri)
    remap = -np.ones(len(unique), dtype=np.int64)
    remap[used] = np.arange(len(used))
    triangulation = Triangulation(a[used], b[used], remap[tri])
    finder = triangulation.get_trifinder()
    centre = (np.arange(size) + 0.5) / (size / 2) - 1.0
    qx, qy = np.meshgrid(centre, -centre)
    disc = qx ** 2 + qy ** 2 < 1.0
    xs, ys = qx[disc], qy[disc]
    found = finder(xs, ys)
    idx = np.zeros((len(xs), 3), dtype=np.int64)
    wts = np.zeros((len(xs), 3), dtype=np.float32)
    hit = found >= 0
    t = tri[found[hit]]
    ax, ay = a[t], b[t]
    det = (ay[:, 1] - ay[:, 2]) * (ax[:, 0] - ax[:, 2]) + (ax[:, 2] - ax[:, 1]) * (ay[:, 0] - ay[:, 2])
    l0 = ((ay[:, 1] - ay[:, 2]) * (xs[hit] - ax[:, 2]) + (ax[:, 2] - ax[:, 1]) * (ys[hit] - ay[:, 2])) / det
    l1 = ((ay[:, 2] - ay[:, 0]) * (xs[hit] - ax[:, 2]) + (ax[:, 0] - ax[:, 2]) * (ys[hit] - ay[:, 2])) / det
    idx[hit] = t
    wts[hit] = np.stack([l0, l1, 1 - l0 - l1], -1)
    miss = ~hit
    if miss.any():
        # Fill the narrow projected rim from the nearest front-facing sample.
        front = np.nonzero(c > 0)[0]
        tree = cKDTree(np.stack([a[front], b[front]], -1))
        _, near = tree.query(np.stack([xs[miss], ys[miss]], -1))
        idx[miss] = front[near][:, None]
        wts[miss] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rep = np.arange(M // 2)[:, None] * M + np.arange(M)[None, :]
    flat = rep.reshape(-1)[idx]
    info = {
        "unique_samples": int(len(unique)),
        "hull_triangles": int(len(simp)),
        "front_triangles": int(len(tri)),
        "pixels": int(disc.sum()),
        "pixels_outside_front_triangles": int(miss.sum()),
        "weights_min": float(wts.min()),
        "weights_max": float(wts.max()),
    }
    return disc, flat, wts, info


def torus_lookup(u, v, M):
    """Return sample indices and linear weights on the periodic lattice triangles."""
    # Swapping the index axes preserves the sphere's outward orientation.
    fi = M * ((v - SHIFT) % 1.0) - 0.5
    fj = M * ((u - SHIFT) % 1.0) - 0.5
    i0, j0 = np.floor(fi), np.floor(fj)
    a, b = fi - i0, fj - j0
    i0 = i0.astype(np.int64) % M
    j0 = j0.astype(np.int64) % M
    i1, j1 = (i0 + 1) % M, (j0 + 1) % M
    low = a + b <= 1.0
    idx = np.where(
        low[:, None],
        np.stack([i0 * M + j0, i1 * M + j0, i0 * M + j1], -1),
        np.stack([i1 * M + j1, i1 * M + j0, i0 * M + j1], -1),
    )
    wts = np.where(low[:, None], np.stack([1 - a - b, a, b], -1), np.stack([a + b - 1, 1 - b, 1 - a], -1))
    return idx, wts.astype(np.float32)


def torus_weights(width, height, scale, M):
    """Map pixels inside the rhombus to the saved torus field."""
    x = (np.arange(width) + 0.5) / scale
    y = (height - np.arange(height) - 0.5) / scale
    qx, qy = np.meshgrid(x, y)
    v = 2 * qy / SEC3
    u = qx - v / 2
    inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    idx, wts = torus_lookup(u[inside], v[inside], M)
    return inside, idx, wts


def ring_check(field, cone, M):
    """Check whether every sampled ray meets the jet before leaving the rhombus.

    This threshold check describes the displayed band, not flow accuracy.
    """
    xc, yc = cone["torus_x"], cone["torus_y"]
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radii = np.linspace(0.0, 0.6, 1201)
    closed, inner, outer = [], [], []
    for theta in angles:
        x = xc + radii * math.cos(theta)
        y = yc + radii * math.sin(theta)
        v = 2 * y / SEC3
        u = x - v / 2
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        stop = np.argmin(inside) if not inside.all() else len(radii)
        idx, wts = torus_lookup(u[:stop], v[:stop], M)
        val = (field.reshape(-1)[idx] * wts).sum(1)
        band = np.nonzero(np.abs(val) >= 2e-5)[0]
        closed.append(len(band) > 0)
        if len(band):
            inner.append(float(radii[band[0]]))
        outer.append(float(radii[stop - 1]))
    return {
        "threshold_per_s": 2e-5,
        "rays": len(angles),
        "rays_meeting_jet_inside_domain": int(sum(closed)),
        "closed": bool(all(closed)),
        "jet_inner_radius_min": min(inner) if inner else None,
        "jet_inner_radius_max": max(inner) if inner else None,
        "distance_to_domain_edge_min": min(outer),
    }


def half_turn_check(field, cone, M):
    """Compare interpolated vorticity at paired points around the north cone."""
    rng = np.random.default_rng(0)
    r = 0.35 * np.sqrt(rng.random(20000))
    th = 2 * np.pi * rng.random(20000)
    u0, v0 = cone["u"], cone["v"]
    du, dv = r * np.cos(th), r * np.sin(th)
    a_idx, a_w = torus_lookup((u0 + du) % 1, (v0 + dv) % 1, M)
    b_idx, b_w = torus_lookup((u0 - du) % 1, (v0 - dv) % 1, M)
    flat = field.reshape(-1)
    fa = (flat[a_idx] * a_w).sum(1)
    fb = (flat[b_idx] * b_w).sum(1)
    return float(np.abs(fa - fb).max())


def schedule(args):
    """Return playback intervals in seconds with target times in simulated days.

    Initial and final holds bracket two linear advances, meeting at split_day.
    The total duration includes both holds.
    """
    b0 = args.initial_hold
    b1 = b0 + args.early_seconds
    b2 = b1 + args.late_seconds
    b3 = b2 + args.final_hold
    return [
        {"start_s": 0.0, "end_s": b0, "start_day": 0.0, "end_day": 0.0, "treatment": "hold first snapshot"},
        {"start_s": b0, "end_s": b1, "start_day": 0.0, "end_day": args.split_day, "treatment": "linear"},
        {"start_s": b1, "end_s": b2, "start_day": args.split_day, "end_day": DAYS, "treatment": "linear"},
        {"start_s": b2, "end_s": b3, "start_day": DAYS, "end_day": DAYS, "treatment": "hold last snapshot"},
    ]


def select_slots(t, plan, fps):
    """Select saved snapshots for display slots sampled at their midpoints.

    t contains chronological snapshot times in simulated seconds. Each
    advancing slot selects the nearest saved time, with exact ties going to
    the earlier snapshot; holds select the exact first or last snapshot.
    Return slot records and their snapshot indices, preserving repeats.
    """
    total = plan[-1]["end_s"]
    count = int(round(total * fps))
    if not math.isclose(count / fps, total, rel_tol=0, abs_tol=1e-9):
        raise SystemExit("playback length is not a whole number of display slots")
    slots = []
    for n in range(count):
        start = n / fps
        mid = (n + 0.5) / fps
        part = next(k for k, seg in enumerate(plan) if seg["start_s"] <= mid < seg["end_s"])
        seg = plan[part]
        if part == 0:
            k, day = 0, None
        elif part == len(plan) - 1:
            k, day = len(t) - 1, None
        else:
            day = seg["start_day"] + (seg["end_day"] - seg["start_day"]) * (mid - seg["start_s"]) / (seg["end_s"] - seg["start_s"])
            gap = np.abs(t - day * DAY)
            k = int(np.flatnonzero(gap == gap.min())[0])
        slots.append({"slot": n, "start_s": start, "duration_s": 1 / fps, "midpoint_s": mid, "interval": part,
                      "target_day": day, "index": k, "t_s": float(t[k]), "t_days": float(t[k] / DAY)})
    pick = np.array([row["index"] for row in slots])
    if (np.diff(pick) < 0).any():
        raise SystemExit("selected snapshots are not in time order")
    if pick[0] != 0 or pick[-1] != len(t) - 1:
        raise SystemExit("playback does not start and end on the saved endpoints")
    return slots, pick


def box(x, y, w, h):
    return [x / W, 1 - (y + h) / H, w / W, h / H]


def build_figure(lut, cone_rows):
    """Build fixed views, cone markers, a shared colour bar and the time axis."""
    plt.rcParams.update({
        "font.family": FONTS,
        "font.size": px(16),
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONTS[0],
        "mathtext.it": FONTS[0] + ":italic",
        "mathtext.bf": FONTS[0] + ":bold",
        "mathtext.sf": FONTS[0],
        "mathtext.tt": FONTS[0],
        "mathtext.cal": FONTS[0],
        "mathtext.bfit": FONTS[0] + ":italic:bold",
        "text.color": INK,
        "path.simplify": False,
    })
    fig = plt.figure(figsize=(W / 96, H / 96), dpi=96 * SS, facecolor="white")
    cx, cy = SPHERE_C
    r = SPHERE_R
    sphere_px = int(round(2 * r * SS))
    sx0 = int(round((cx - r) * SS))
    sy0 = int(round(H * SS - (cy + r) * SS))
    sphere_img = fig.figimage(np.zeros((sphere_px, sphere_px, 4), np.uint8), xo=sx0, yo=sy0, origin="upper", zorder=1)
    sax = fig.add_axes([sx0 / (W * SS), sy0 / (H * SS), sphere_px / (W * SS), sphere_px / (H * SS)])
    sax.set_xlim(-1, 1)
    sax.set_ylim(-1, 1)
    sax.set_axis_off()
    sax.set_zorder(3)
    sax.patch.set_alpha(0)
    sax.add_patch(Circle((0, 0), 1.0, fill=False, lw=px(1.0), ec=GUIDE, zorder=3))

    s = TORUS_S * SS
    tw = int(math.ceil(1.5 * s))
    th = int(math.ceil(SEC3 / 2 * s))
    tx0 = int(round(TORUS_X0 * SS))
    ty0 = int(round(H * SS - TORUS_Y0 * SS))
    torus_img = fig.figimage(np.zeros((th, tw, 4), np.uint8), xo=tx0, yo=ty0, origin="upper", zorder=1)
    tax = fig.add_axes([tx0 / (W * SS), ty0 / (H * SS), tw / (W * SS), th / (H * SS)])
    tax.set_xlim(0, tw / s)
    tax.set_ylim(0, th / s)
    tax.set_axis_off()
    tax.set_zorder(3)
    tax.patch.set_alpha(0)
    top = SEC3 / 2
    tax.add_patch(Polygon([(0, 0), (1, 0), (1.5, top), (0.5, top)], closed=True, fill=False, lw=px(1.0), ec=GUIDE, zorder=3))
    tick = 0.032
    tick_kw = dict(color=GUIDE, lw=px(1.6), solid_capstyle="butt", zorder=3, clip_on=False)
    for (mx, my) in ((0.5, 0.0), (1.0, top)):
        tax.plot([mx, mx], [my - tick / 2, my + tick / 2], **tick_kw)
    along = np.array([0.5, top])
    across = np.array([top, -0.5])
    for base in (np.array([0.25, top / 2]), np.array([1.25, top / 2])):
        for off in (-0.011, 0.011):
            c0 = base + off * along
            p, q = c0 - across * tick / 2, c0 + across * tick / 2
            tax.plot([p[0], q[0]], [p[1], q[1]], **tick_kw)

    disc_r = 5.5 / TORUS_S
    for row in cone_rows:
        tax.add_patch(Circle((row["torus_x"], row["torus_y"]), disc_r, fc=GOLD, ec=RIM, lw=px(1.4), zorder=5, clip_on=False))
        if row["front_facing"]:
            sax.add_patch(Circle((row["view_x"], row["view_y"]), 5.5 / SPHERE_R, fc=GOLD, ec=RIM, lw=px(1.4), zorder=5, clip_on=False))

    fig.text(cx / W, 1 - 24 / H, WORDS["sphere"], ha="center", va="center", fontsize=px(18), color=INK)
    fig.text((TORUS_X0 + 0.75 * TORUS_S) / W, 1 - 24 / H, WORDS["torus"], ha="center", va="center", fontsize=px(18), color=INK)

    lx, ly = LEGEND
    lax = fig.add_axes(box(lx - 8, ly - 8, 16, 16))
    lax.set_xlim(-8, 8)
    lax.set_ylim(-8, 8)
    lax.set_axis_off()
    lax.add_patch(Circle((0, 0), 5.5, fc=GOLD, ec=RIM, lw=px(1.4), clip_on=False))
    fig.text((lx + 12) / W, 1 - ly / H, WORDS["legend"], ha="left", va="center_baseline", fontsize=px(16), color=INK)

    x, y, w, h = CBAR
    cax = fig.add_axes(box(x, y, w, h))
    cmap = ListedColormap(lut)
    bar = fig.colorbar(ScalarMappable(Normalize(-VMAX, VMAX), cmap), cax=cax, orientation="horizontal", ticks=TICKS)
    bar.outline.set_visible(False)
    bar.ax.set_xticklabels([f"{v / 1e-4:g}" for v in TICKS])
    bar.ax.tick_params(length=5, width=px(1.0), color=SOFT, labelcolor=SOFT, labelsize=px(16), pad=4)
    bar.set_label(WORDS["colour_bar"], fontsize=px(16), color=INK, labelpad=6)
    fig.text((x + w + 8) / W, 1 - (y + h / 2) / H, r"$\times10^{-4}$", ha="left", va="center", fontsize=px(16), color=SOFT)

    ax0, ay0, aw = TAXIS
    taxis = fig.add_axes(box(ax0, ay0 - 10, aw, 20))
    taxis.set_xlim(0, DAYS)
    taxis.set_ylim(-1, 1)
    for side in ("left", "right", "top"):
        taxis.spines[side].set_visible(False)
    taxis.spines["bottom"].set_position(("data", 0))
    taxis.spines["bottom"].set_color(GUIDE)
    taxis.spines["bottom"].set_linewidth(px(1.4))
    taxis.set_yticks([])
    taxis.set_xticks(range(0, 7))
    taxis.set_xticklabels([str(k) for k in range(0, 7)])
    taxis.tick_params(axis="x", length=5, width=px(1.0), color=SOFT, labelcolor=SOFT, labelsize=px(16), pad=4)
    taxis.set_xlabel(WORDS["time_axis"], fontsize=px(16), color=INK, labelpad=6)
    taxis.patch.set_alpha(0)
    (dot,) = taxis.plot([0], [0], "o", ms=px(12), mfc=CHARCOAL, mec="white", mew=px(1.5), clip_on=False, zorder=6)
    return fig, sphere_img, torus_img, dot, (sphere_px, tw, th, s)


def colour(values, lut_rgba):
    """Map interpolated vorticity to the shared fixed palette."""
    q = np.rint((np.clip(values, -VMAX, VMAX) / VMAX + 1) * 0.5 * (LEVELS - 1)).astype(np.int64)
    return lut_rgba[q]


def render(args):
    """Validate the archive, render selected fields and record media provenance.

    Standard output is JSON containing media sizes and hashes, the display
    slot count (including holds and repeats), and geometric checks.
    Intermediates stay in the chosen directory.
    """
    archive = Path(args.archive)
    expected = json.loads((PACK / "hero-input.json").read_text())
    digest = sha256(archive)
    nbytes = archive.stat().st_size
    data, meta, X, t, branch, vort, height = load(archive)
    summary = verify(meta, X, t, branch, vort, height, expected)
    del height
    M = int(meta["M"])
    I = (-1 - np.arange(M)) % M
    pair_diff = max(float(np.abs(f - f[I][:, I]).max()) for f in vort)
    x_pair_diff = float(np.abs(X - X[:, I][:, :, I]).max())
    cone_rows, cone_check = cones(meta, X, branch)
    north = max(cone_rows, key=lambda r: r["lat_deg"])

    rgb, lightness = diverging(LEVELS)
    lut = np.concatenate([(rgb * 255).round(), np.full((LEVELS, 1), 255)], 1).astype(np.uint8)

    fig, sphere_img, torus_img, dot, (sphere_px, tw, th, s) = build_figure(rgb, cone_rows)
    text_audit = audit(fig, list(WORDS.values()))
    disc, s_idx, s_w, sphere_info = sphere_weights(X, sphere_px)
    inside, t_idx, t_w = torus_weights(tw, th, s, M)

    plan = schedule(args)
    slots, pick = select_slots(t, plan, args.fps)
    distinct = sorted(set(int(k) for k in pick))
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="hero-"))
    frames_dir = work / "frames"
    if frames_dir.exists():
        # Reusing a work directory replaces its previous frame sequence.
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    sphere_rgba = np.zeros((sphere_px, sphere_px, 4), np.uint8)
    torus_rgba = np.zeros((th, tw, 4), np.uint8)
    checks = {}
    for n, k in enumerate(distinct):
        field = vort[k]
        flat = field.reshape(-1)
        sphere_rgba[disc] = colour((flat[s_idx] * s_w).sum(1), lut)
        torus_rgba[inside] = colour((flat[t_idx] * t_w).sum(1), lut)
        sphere_img.set_data(sphere_rgba)
        torus_img.set_data(torus_rgba)
        dot.set_xdata([t[k] / DAY])
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).reduce(SS)
        frame.save(frames_dir / f"snap_{k:04d}.png", optimize=False, compress_level=1)
        if n in (0, len(distinct) // 2, len(distinct) - 1):
            checks[str(int(k))] = {
                "t_s": float(t[k]),
                "ring": ring_check(field, north, M),
                "half_turn_max_abs_diff_per_s": half_turn_check(field, north, M),
            }
    plt.close(fig)

    gif_path, mp4_path = OUT / "galewsky.gif", OUT / "galewsky.mp4"
    OUT.mkdir(exist_ok=True)
    count = len(pick)
    for n, k in enumerate(pick):
        os.symlink(f"snap_{int(k):04d}.png", frames_dir / f"slot_{n:04d}.png")
    encode_mp4(frames_dir, count, args.fps, mp4_path, args.crf)
    palette, runs = encode_gif(frames_dir, pick, args.fps, gif_path, lut[:, :3], args.colours)
    gif_info = decode_gif(gif_path)
    mp4_frames = count_mp4_frames(mp4_path)
    if gif_info["images"] != len(runs) or gif_info["delays_cs"] != [r["delay_cs"] for r in runs]:
        raise SystemExit("encoded GIF images or delays differ from the selected runs")
    if mp4_frames != count:
        raise SystemExit("encoded MP4 frame count differs from the display slots")

    outputs = {}
    for path in (gif_path, mp4_path):
        outputs[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    probe = ffprobe(mp4_path)
    font = font_manager.findfont(font_manager.FontProperties(family=FONTS))
    provenance = {
        "figure": "galewsky",
        "labels": WORDS,
        "text_audit": {"labels": text_audit["labels"], "numeric": text_audit["ticks"], "min_text_px": text_audit["min_text_px"]},
        "colour_bar_ticks_per_s": TICKS,
        "colour_bar_tick_labels": [f"{v / 1e-4:g}" for v in TICKS],
        "colour_bar_multiplier": "x10^-4",
        "time_axis_ticks_days": list(range(0, 7)),
        "input": {
            "archive": archive_label(archive),
            "sha256": digest,
            "bytes": nbytes,
            "sha256_matches_figure_pack": digest == expected["archive_sha256"],
            "figure_pack_record": {"path": "figure-pack/hero-input.json", "sha256": sha256(PACK / "hero-input.json")},
            "arrays_used": ["X", "vorticity", "t_s", "branch_distance_cells", "meta"],
            "arrays_checked": ["height"],
            "meta": {k: meta[k] for k in ("case", "M", "scheme", "dt_s", "nsteps", "mass_rel", "mean_spacing_m", "branch_points_ij")},
        },
        "archive_summary": summary,
        "pair_symmetry": {
            "pairing": "(i, j) ~ (-1-i, -1-j) mod M",
            "X_max_abs_diff": x_pair_diff,
            "vorticity_max_abs_diff_per_s": pair_diff,
        },
        "colour_scale": {
            "kind": "linear, symmetric, fixed for every frame and both views",
            "range_per_s": [-VMAX, VMAX],
            "levels": LEVELS,
            "zero_colour": "#%02x%02x%02x" % tuple(lut[LEVELS // 2, :3]),
            "negative_end": "#%02x%02x%02x" % tuple(lut[0, :3]),
            "positive_end": "#%02x%02x%02x" % tuple(lut[-1, :3]),
            "negative_hue_from": BLUE,
            "positive_hue_from": RUST,
            "cielab_lightness_zero_to_end": [float(lightness[LEVELS // 2]), float(lightness[-1])],
            "clipped_values": 0,
        },
        "sphere_view": {
            "projection": "orthographic",
            "centre_lon_deg": VIEW_LON,
            "centre_lat_deg": VIEW_LAT,
            "north_up": True,
            "deduplication": "samples with i < M/2 represent each pair",
            "interpolation": "linear on the spherical Delaunay triangulation (convex hull) of the unique samples, front-facing triangles only",
            **sphere_info,
        },
        "torus_view": {
            "u": "((j + 1/2)/M + 15/32) mod 1",
            "v": "((i + 1/2)/M + 15/32) mod 1",
            "x": "u + v/2",
            "y": "sqrt(3) v / 2",
            "paired_location": "(15/16 - u, 15/16 - v) mod 1",
            "interpolation": "linear on the triangles of the hexagonal index lattice, wrapped at both periodic boundaries",
            "edge_ticks": "one tick on the v = 0 and v = 1 edges, two ticks on the u = 0 and u = 1 edges",
            "orientation_relative_to_sphere_seen_from_outside": "preserved",
        },
        "cone_points": cone_rows,
        "cone_check": cone_check,
        "north_pole_cone_ring": checks,
        "playback": {
            "schedule": plan,
            "sampling": "target evaluated at each display slot midpoint; nearest saved t_s, exact ties to the earlier snapshot; holds use the first and last snapshots",
            "fps": args.fps,
            "seconds": plan[-1]["end_s"],
            "display_slots": len(slots),
            "distinct_snapshots": len(distinct),
            "slots": slots,
            "selected_index": [int(k) for k in pick],
            "selected_t_s": [float(t[k]) for k in pick],
            "max_abs_offset_from_target_s": max(abs(r["t_s"] - r["target_day"] * DAY) for r in slots if r["target_day"] is not None),
            "same_snapshots_in_gif_and_mp4": True,
            "gif": {
                "images": gif_info["images"],
                "image_delays_cs": gif_info["delays_cs"],
                "image_snapshot_index": [r["index"] for r in runs],
                "image_slot_count": [r["slots"] for r in runs],
                "decoded_duration_s": gif_info["duration_s"],
                "loop": gif_info["loop"],
            },
            "mp4": {
                "frames": mp4_frames,
                "decoded_duration_s": float(probe["format"]["duration"]),
            },
        },
        "render": {
            "canvas_px": [W, H],
            "supersampling": SS,
            "font": Path(font).name,
            "gif_palette_colours": palette,
            "mp4": probe,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
                "pillow": Image.__version__,
                "ffmpeg": tool_version(["ffmpeg", "-version"]),
                "gifsicle": tool_version(["gifsicle", "--version"]),
            },
        },
        "outputs": outputs,
    }
    (OUT / "galewsky.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "frames": count, "checks": checks}, indent=2))


def archive_label(archive):
    """Record a repository-relative input name without exposing an external path."""
    try:
        return str(archive.absolute().relative_to(ROOT))
    except ValueError:
        return archive.name


def tool_version(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def ffprobe(path):
    """Read video format and duration, and check that metadata precedes media."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_frames,codec_type",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    with open(path, "rb") as handle:
        head = handle.read(1 << 16)
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    info["faststart"] = moov >= 0 and (mdat < 0 or moov < mdat)
    return info


def encode_mp4(frames_dir, count, fps, path, crf):
    """Encode all display slots, including both holds, as H.264 without audio."""
    tmp = path.with_suffix(".tmp.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:g}", "-i", str(frames_dir / "slot_%04d.png"),
        "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int+full_chroma_inp,format=yuv420p",
        "-r", f"{fps:g}", "-frames:v", str(count),
        "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", "-an", "-map_metadata", "-1", "-metadata", "creation_time=", str(tmp),
    ], check=True)
    os.replace(tmp, path)


def encode_gif(frames_dir, pick, fps, path, lut_rgb, colours):
    """Encode a looping GIF with a fixed palette and no dithering."""
    greys = np.linspace(0, 255, 40).round()
    fixed = [tuple(int(c) for c in row) for row in lut_rgb]
    extra = [(int(g), int(g), int(g)) for g in greys]
    for hexc in (GOLD, RIM, CHARCOAL, GUIDE, SOFT, INK):
        extra.append(tuple(int(hexc[k:k + 2], 16) for k in (1, 3, 5)))
    gold = np.array([int(GOLD[k:k + 2], 16) for k in (1, 3, 5)], float)
    rim = np.array([int(RIM[k:k + 2], 16) for k in (1, 3, 5)], float)
    for f in (0.25, 0.5, 0.75):
        extra.append(tuple(int(c) for c in (gold * f + rim * (1 - f)).round()))
        extra.append(tuple(int(c) for c in (gold * f + 255 * (1 - f)).round()))
    step = max(1, len(fixed) // max(1, colours - len(extra)))
    field = fixed[::step]
    if fixed[len(fixed) // 2] not in field:
        field.append(fixed[len(fixed) // 2])
    entries = list(dict.fromkeys(field + extra))[:256]
    flat = [c for rgb in entries for c in rgb] + [0, 0, 0] * (256 - len(entries))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(flat)
    delay = 100 / fps
    if not math.isclose(delay, round(delay), abs_tol=1e-9):
        raise SystemExit("frame rate does not give whole-centisecond GIF delays")
    delay = int(round(delay))
    runs = []
    for k in pick:
        if runs and runs[-1]["index"] == int(k):
            runs[-1]["slots"] += 1
        else:
            runs.append({"index": int(k), "slots": 1})
    for run in runs:
        run["delay_cs"] = run["slots"] * delay
    images = []
    for run in runs:
        with Image.open(frames_dir / f"snap_{run['index']:04d}.png") as im:
            images.append(im.convert("RGB").quantize(palette=palette_image, dither=Image.Dither.NONE))
    durations = [run["delay_cs"] * 10 for run in runs]
    raw = frames_dir / "raw.gif"
    images[0].save(raw, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False, disposal=1)
    tmp = path.with_suffix(".tmp.gif")
    subprocess.run(["gifsicle", "-O3", "--loopcount=forever", "--no-comments", "--no-names", "-o", str(tmp), str(raw)], check=True)
    os.replace(tmp, path)
    return len(entries), runs


def decode_gif(path):
    out = subprocess.run(["gifsicle", "--info", str(path)], capture_output=True, text=True, check=True).stdout
    delays = []
    for line in out.splitlines():
        if "delay" in line:
            delays.append(int(round(float(line.split("delay")[1].split("s")[0]) * 100)))
    with Image.open(path) as im:
        images = im.n_frames
        pil = []
        for n in range(images):
            im.seek(n)
            pil.append(int(im.info.get("duration", 0)))
    if [d * 10 for d in delays] != pil:
        raise SystemExit("GIF delays disagree between decoders")
    return {"images": images, "delays_cs": delays, "duration_s": sum(delays) / 100, "loop": "forever" if "loop forever" in out else out}


def count_mp4_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(ROOT / "build" / "hero.npz"), help="Saved Galewsky archive (default: build/hero.npz in the repository).")
    parser.add_argument("--work", default=None, help="Directory for retained intermediate files; replaces its frames/ subdirectory (default: a temporary directory).")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback frames per second (default: %(default)s).")
    parser.add_argument("--initial-hold", type=float, default=0.5, help="Seconds to hold the initial snapshot before advancing (default: %(default)s).")
    parser.add_argument("--early-seconds", type=float, default=2.5, help="Playback seconds from day 0 to --split-day, excluding the initial hold (default: %(default)s).")
    parser.add_argument("--split-day", type=float, default=4.0, help="Simulated day at which playback changes from the early to the late interval (default: %(default)s).")
    parser.add_argument("--late-seconds", type=float, default=12.0, help="Playback seconds from --split-day to day 6, excluding the final hold (default: %(default)s).")
    parser.add_argument("--final-hold", type=float, default=1.0, help="Seconds to hold the final snapshot after advancing (default: %(default)s).")
    parser.add_argument("--crf", type=int, default=12, help="H.264 constant rate factor; lower values preserve more detail (default: %(default)s).")
    parser.add_argument("--colours", type=int, default=256, help="Palette-size target for the GIF (default: %(default)s; actual count is recorded).")
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    sys.exit(main())
