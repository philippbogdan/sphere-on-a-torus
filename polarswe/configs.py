"""Set the main scheme and the alternatives used in the saved runs."""

from . import flopcount as fc


MAIN = "order=8,integrator=ls10o4,courant=1.252,hv_mode=step,kappa=0.0001,cone=8_3_4_lines_cc_dw10,cone_filter=0,tracer=skew"
MAIN_NAME = "FD8+cone operators (main)"
PREV_MAIN = "order=6,integrator=kg10o4,courant=1.366,hv_mode=step,kappa=0.0005"
PREV_NAME = "FD6+KG10 (previous main)"
FD4_NAME = "FD4+KG10"
FD8_NAME = "FD8+KG10"
FD8F_NAME = "FD8, filter strong near the cones"
EDGE_NAME = "main, edge orientation"
SEPCURL_NAME = "cone operators with a separate curl, filter doubled in the patches"
FILTIN_NAME = "main with the filter in the cone patches, flux-form tracers (f5df93d)"
TRSK_NAME = "TRSK C-grid torus"

CONE = dict(R=8, p=3, rs=4, shape="lines")


TRSK = "solver=trsk,integrator=kg10o4,courant=TRSK_COURANT"
TRSK_COURANT = 1.7

FRONTIER = {

    MAIN_NAME: (MAIN, dict(order=8, integrator="ls10o4", hv_mode="step")),
    EDGE_NAME: (MAIN + ",orient=edge", dict(order=8, integrator="ls10o4", hv_mode="step")),
    SEPCURL_NAME: ("order=8,integrator=ls10o4,courant=1.252,hv_mode=step,kappa=0.0001,cone=8_3_4_lines,kappa_near=0.0002,rnear_cells=3",
                   dict(order=8, integrator="ls10o4", hv_mode="step")),
    FILTIN_NAME: (MAIN + ",cone_filter=1,tracer=flux", dict(order=8, integrator="ls10o4", hv_mode="step")),
    FD8F_NAME: ("order=8,integrator=ls10o4,courant=1.252,hv_mode=step,kappa=0.0001,kappa_near=0.002,rnear=5",
                dict(order=8, integrator="ls10o4", hv_mode="step")),
    PREV_NAME: (PREV_MAIN, dict(order=6, integrator="kg10o4", hv_mode="step")),
    "FD6+KG9": ("order=6,integrator=kg9,courant=1.222,hv_mode=step,kappa=0.0005", dict(order=6, integrator="kg9", hv_mode="step")),
    FD4_NAME: ("order=4,integrator=kg10o4,courant=1.58,hv_mode=step,kappa=0.002", dict(order=4, integrator="kg10o4", hv_mode="step")),
    "FD4+KG10 kappa 0.0005": ("order=4,integrator=kg10o4,courant=1.58,hv_mode=step,kappa=0.0005", dict(order=4, integrator="kg10o4", hv_mode="step")),
    "FD6+KG10 kappa 0.002": ("order=6,integrator=kg10o4,courant=1.366,hv_mode=step,kappa=0.002", dict(order=6, integrator="kg10o4", hv_mode="step")),
    FD8_NAME: ("order=8,integrator=kg10o4,courant=1.252,hv_mode=step,kappa=0.0005", dict(order=8, integrator="kg10o4", hv_mode="step")),
    "FD6+RK4": ("order=6,integrator=rk4,courant=0.43,hv_mode=step,kappa=0.0005", dict(order=6, integrator="rk4", hv_mode="step")),
    "FD4+RK4": ("order=4,integrator=rk4,courant=0.5,hv_mode=step,kappa=0.002", dict(order=4, integrator="rk4", hv_mode="step")),
    "FD2+SSPRK3 flat filter": ("order=2,integrator=ssprk3,courant=0.4,hv_mode=step,lap_kind=flat,kappa=0.002",
                               dict(order=2, integrator="ssprk3", hv_mode="step", lap_kind="flat")),
    "Lax-Wendroff torus": ("solver=lwt,courant=0.35,kappa=0.002", "lwt"),
    "square cover": (PREV_MAIN + ",lattice=square", dict(order=6, integrator="kg10o4", hv_mode="step")),
    TRSK_NAME: (TRSK.replace("TRSK_COURANT", repr(TRSK_COURANT)), "trsk"),
}


STATE_WORDS = 3
STATIC_WORDS = {"positions": 4, "basis": 7}
DERIVED_METRIC_FLOPS = 50
STATIC = "positions"


def cost(name, static=STATIC):
    scheme, ck = FRONTIER[name]
    if ck == "trsk":
        return trsk_cost(scheme)
    if ck == "lwt":
        c = fc.lwt_cost()
        return dict(flops=c["flops"], reach=c["reach"], static_words=4, state_words=STATE_WORDS)
    if static == "positions":
        c = fc.step_cost(**ck, metric="recomputed")
        out = dict(flops=c["flops"], reach=c["reach"], static_words=STATIC_WORDS["positions"], state_words=STATE_WORDS)
        if "cone=" in scheme:
            out.update(cone_cost(scheme, ck))
        return out
    c = fc.step_cost(**ck)
    return dict(flops=c["flops"] + DERIVED_METRIC_FLOPS, reach=c["reach"], static_words=STATIC_WORDS["basis"],
                state_words=STATE_WORDS)


def cone_entries(shape, rs):
    from .conehp import hexdist
    if shape == "disc":
        return sum(1 for di in range(-rs, rs + 1) for dj in range(-rs, rs + 1) if hexdist(di, dj) <= rs)
    if shape == "lines":
        return 1 + 6 * rs
    return 1 + 4 * rs


def cone_cost(scheme, ck):


    sc = parse(scheme)
    parts = str(sc["cone"]).split("_")
    R, p, rs, shape = (parts + ["disc"])[:4]
    cc = len(parts) > 4 and parts[4] == "cc"
    ne = cone_entries(shape, int(rs))
    c = fc.cone_patch_cost(order=ck["order"], n_entries=ne, integrator=ck["integrator"])
    near = float(sc.get("kappa_near", -1)) >= 0 and float(sc.get("rnear_cells", -1)) > 0
    return dict(cone_R=float(R), cone_entries=ne, cone_flops=c["patch"], cone_static_words=(6 if cc else 9) * ne + 1 + int(near),
                cone_flop_ratio=c["ratio"], cone_curl="compatible" if cc else "separate")


def trsk_cost(scheme, M=128):


    from . import trsk, testcases as tc
    sc = parse(scheme)
    P, _ = trsk.torus_mesh(M)
    S = trsk.TRSK(trsk.VoronoiMesh(P), tc.TC5())
    lay = trsk.torus_layout(M, S)
    stages = trsk.Solver.__new__(trsk.Solver)
    stages.integrator = sc.get("integrator", "rk4")
    s = stages.stages()
    flops = s * (trsk.flops_per_cell_stage(S) + 2 * 4)
    return dict(flops=float(flops), reach=int(s * lay["reach_stage"]), static_words=trsk.static_words_per_site(S),
                state_words=4, reach_stage=lay["reach_stage"], stages=s)


def parse(scheme):
    d = {}
    for item in filter(None, scheme.split(",")):
        k, v = item.split("=")
        try:
            d[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            d[k] = v
    return d
