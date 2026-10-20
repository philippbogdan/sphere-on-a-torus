"""Estimate step rates and link traffic. These are models, not hardware timings."""


from dataclasses import dataclass, field
import math

WORD_BITS = 32


@dataclass
class Machine:
    nodes_x: int = 8
    nodes_y: int = 8
    cores_w: int = 744
    cores_h: int = 1116
    clock_hz: float = 0.75e9
    peak_flop_per_cycle: float = 2.0
    side_bw_bits: float = 300e9
    node_latency_s: float = 10e-6
    core_link_latency_s: float = 12e-9
    sram_bytes_per_core: int = 48 * 1024
    balanced_io: bool = False

    @property
    def nodes(self):
        return self.nodes_x * self.nodes_y

    @property
    def side_words_per_s(self):
        return self.side_bw_bits / WORD_BITS


PAPER_MODELS = {
    "he5": dict(f=(105.0, 3.74, 6.72), flops=9, io=(4, 4)),
    "he9": dict(f=(97.0, 3.5, 9.37), flops=17, io=(4, 4)),
    "swe": dict(f=(1026.0, 183.2, 137.6), flops=155, io=(64, 80)),
}


@dataclass
class Kernel:


    name: str
    c0: float
    c1: float
    c2: float
    flops_per_point: float
    wh1: float
    wh0: float
    wv1: float
    wv0: float
    reach: float = 1.0
    words_per_point_resident: float = 0.0
    notes: str = ""

    def cycles(self, n):
        return self.c0 + self.c1 * n + self.c2 * n * n


def paper_kernel(key):
    m = PAPER_MODELS[key]
    c0, c1, c2 = m["f"]
    a, b = m["io"]


    return Kernel(key, c0, c1, c2, m["flops"], a / 2, b / 2, a / 2, b / 2)


def dt_kernel(name, flops_per_point, efficiency, state_words, static_words, reach, translations=1,
              c0=None, c1=None, resident_words=0.0, notes=""):


    words_strip = translations * (2 * reach * state_words + reach * static_words)


    corner = 0.0
    node_corner = translations * (4 * reach * reach * state_words + reach * reach * static_words)
    c2 = flops_per_point / (2.0 * efficiency)
    swe = PAPER_MODELS["swe"]
    ratio = words_strip / 32.0
    if c1 is None:
        c1 = swe["f"][1] * ratio
    if c0 is None:
        c0 = swe["f"][0] * ratio
    k = Kernel(name, c0, c1, c2, flops_per_point, words_strip, corner, words_strip, corner,
               reach=reach * translations, words_per_point_resident=resident_words, notes=notes)
    k.node_corner_words = node_corner
    return k


def step_time(kernel, n, mach=Machine(), nx=None, ny=None):
    """Return seconds, rate and the largest of compute, I/O and latency limits.

    n is the width of a square window per core. nx and ny allow a rectangle.
    Bandwidth is per wafer side and direction. Corner traffic is counted once per node.
    """
    nx = n if nx is None else nx
    ny = n if ny is None else ny
    neff = math.sqrt(nx * ny)
    t_comp = kernel.cycles(neff) / mach.clock_hz

    nc = getattr(kernel, "node_corner_words", 0.0)
    H = mach.cores_h * (kernel.wh1 * ny + kernel.wh0) + nc
    V = mach.cores_w * (kernel.wv1 * nx + kernel.wv0) + nc
    if mach.balanced_io:
        t_io = (H + V) / (2 * mach.side_words_per_s)
    else:
        t_io = max(H, V) / mach.side_words_per_s


    c_point = kernel.cycles(neff) / (nx * ny) / mach.clock_hz
    t_lat = 2 * kernel.reach * mach.core_link_latency_s / max(neff, 1e-9)
    limits = dict(compute=t_comp, io=t_io, latency=t_lat)
    lim = max(limits, key=limits.get)
    return dict(t=limits[lim], limit=lim, t_compute=t_comp, t_io=t_io, t_latency=t_lat,
                rate=1.0 / limits[lim], H_words=H, V_words=V,
                latency_hidden=(nx * ny) > 2 * kernel.reach * mach.core_link_latency_s / c_point)


A_EARTH = 6.37122e6
AREA = 4 * math.pi * A_EARTH ** 2
EFFICIENCY_SWE = 0.53


def torus_kernel(name, flops, reach, state_words=4, static_words=4, resident_words=50.0):

    return dt_kernel(name, flops, EFFICIENCY_SWE, state_words, static_words, reach,
                     resident_words=resident_words)


def latlon_points(arcsec):
    nlon = int(round(360 * 3600 / arcsec))
    return nlon * (nlon // 2)


def config_rate(kernel, npts, mach=Machine()):
    """Spread npts computed points over all cores and evaluate the step model."""
    per_core = npts / (mach.nodes * mach.cores_w * mach.cores_h)
    n = math.sqrt(per_core)
    s = step_time(kernel, n, mach)
    s.update(points_per_core=per_core, n=n)
    return s


def link_report(kernel, s, mach=Machine()):


    H, V = s["H_words"], s["V_words"]
    bw = mach.side_words_per_s
    n = s["n"]
    core_link_words = kernel.wh1 * n + kernel.wh0
    return dict(node_edge_words_H=H, node_edge_words_V=V,
                link_utilisation_H=H / bw / s["t"], link_utilisation_V=V / bw / s["t"],
                core_link_words=core_link_words,
                core_link_utilisation=core_link_words / (s["t"] * mach.clock_hz))


def dx_eq(npts):
    return math.pi * A_EARTH / math.sqrt(npts / 2)


def approach_throughput(approach, n_unique, kernel, r_unique, C_polar, mach=Machine(), c_wave=1.0):


    computed = 2 * n_unique if approach == "torus" else n_unique
    s = config_rate(kernel, computed, mach)
    t = s["t"]
    cap = {"single_cover": mach.node_latency_s, "latlon_filter": 3 * mach.node_latency_s}.get(approach, 0.0)
    limit = s["limit"]
    if cap > t:
        t, limit = cap, "latency"
    if approach == "latlon_poles":
        nlat = math.sqrt(n_unique / 2)
        r = C_polar * math.sin(math.pi / (2 * nlat))
    else:
        r = r_unique
    dt = r * dx_eq(n_unique) / c_wave
    return dict(computed=computed, n=s["n"], rate=1.0 / t, dt=dt, r=r, throughput=dt / t, limit=limit)


def cone_patch_points(R):

    pts = []
    r = int(R) + 2
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            ui, uj = di - 0.5, dj - 0.5
            if ui * ui + uj * uj + ui * uj <= R * R:
                pts.append((di, dj))
    return pts


def cone_patch_block_max(R, bx=12, by=8):

    pts = cone_patch_points(R)
    best = 0
    for oi in range(bx):
        for oj in range(by):
            counts = {}
            for di, dj in pts:
                key = ((di + oi) // bx, (dj + oj) // by)
                counts[key] = counts.get(key, 0) + 1
            best = max(best, max(counts.values()))
    return best


def cone_patch_cost(kernel, interior_flops, patch_flops, R, static_words_patch, npts, bx=12, by=8,
                    mach=Machine()):


    s = config_rate(kernel, npts, mach)
    k = cone_patch_block_max(R, bx, by)
    nblock = bx * by
    ratio = (nblock - k + k * patch_flops / interior_flops) / nblock
    t_comp_worst = s["t_compute"] * ratio
    npatch = len(cone_patch_points(R))
    extra_words = npatch * static_words_patch
    t_io = s["t_io"] + extra_words / mach.side_words_per_s
    t_noredist = max(t_comp_worst, t_io, s["t_latency"])

    nshare = len({((di) // bx + a, (dj) // by + b) for di, dj in cone_patch_points(R) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)})
    extra = npatch * (patch_flops - interior_flops) / interior_flops
    ratio_shared = 1.0 + extra / nshare / nblock
    t_redist = max(s["t_compute"] * ratio_shared, t_io, s["t_latency"])
    return dict(patch_points=npatch, block_max=k, block=nblock, worst_core_ratio=ratio, t_compute=s["t_compute"],
                t_compute_worst=t_comp_worst, t_io=s["t_io"], t_io_with_patch=t_io, extra_link_words=extra_words,
                steps_per_s_no_redistribution=1.0 / t_noredist, steps_per_s=1.0 / s["t"], nshare=nshare,
                worst_core_ratio_shared=ratio_shared, steps_per_s_redistributed=1.0 / t_redist)
