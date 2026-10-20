"""Run the numerical studies. The full study needs cluster time."""


import os, sys, json, subprocess, argparse, glob, shutil, time, math
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data")
REFD = os.path.join(ROOT, "refs/stswm")
PY = sys.executable
sys.path.insert(0, ROOT)
from polarswe import configs


def current_commit():
    from polarswe.revision import current
    return current()


COMMIT = current_commit()

MAIN = configs.MAIN
PREV = configs.PREV_MAIN
FD4 = configs.FRONTIER[configs.FD4_NAME][0]
TRSK = configs.FRONTIER[configs.TRSK_NAME][0]

MPAS = "solver=trsk,integrator=rk4,courant=0.5"
HALF_PI = "1.5707963267948966"


def accuracy_specs(highres):
    s = []
    for M in (64, 128, 256):
        s += [f"torus:{M}:tc1_a{a}" for a in ("0.0", "0.7853981633974483", HALF_PI)]
        s += [f"torus:{M}:tc2_a{a}" for a in ("0.0", "0.05", "0.7", "1.5207963267948966", HALF_PI)]
        s += [f"torus:{M}:tc5", f"torus:{M}:tc6"]
    s += [f"torus:{M}:galewsky_nopert" for M in (128, 256)] + [f"torus:{M}:galewsky" for M in (128, 256)]

    s += [f"torus:110:tc1_a{a}" for a in ("0.7853981633974483", HALF_PI)]


    s += [f"torus:{M}:{c}" for M in (144, 288) for c in MPAS_CASES]
    if highres:
        s += ["torus:512:tc5", "torus:512:tc6", "torus:512:tc2_a0.7", "torus:512:galewsky", "torus:512:galewsky_nopert",
              "torus:512:tc1_a" + HALF_PI, "torus:512:tc2_a" + HALF_PI, "torus:512:tc2_a1.5207963267948966"]
        s += [f"torus:576:{c}" for c in MPAS_CASES]
    return s


def needs_refs(spec):
    case = spec.split(":")[2]
    return case in ("tc5", "tc6", "galewsky")


FRONTIER_CASES = ("tc1_a" + HALF_PI, "tc2_a0.7", "tc5", "tc6", "galewsky_nopert", "galewsky")
MPAS_CASES = ("tc2_a0.7", "tc5", "tc6", "galewsky", "galewsky_nopert")
ORIENTATIONS = ("vertex", "edge", "generic", "generic2")

TRSK_BETA = [(128, 0.0, 0), (128, 0.5, 0), (128, 0.75, 0), (128, 0.0, 30), (128, 0.75, 30), (256, 0.0, 0), (256, 0.75, 0)]


CONE_STAB = (("main", ""), ("separate curl, filter in the patches", "cone=8_3_4_lines,cone_filter=1"),
             ("constraints unweighted", "cone=8_3_4_lines_cc"), ("filter in the patches", "cone_filter=1"),
             ("separate curl, filter doubled in the patches", "cone=8_3_4_lines,cone_filter=1,kappa_near=0.0002,rnear_cells=3"),
             ("no filter", "kappa=0"))


TRACER_SPECS = ([f"torus:{M}:tc1_a{HALF_PI}:{MAIN},tracer=flux" for M in (128, 256, 512)]
                + [f"torus:{M}:tc1_a{HALF_PI}:{MAIN},tracer=flux,cone_filter=1" for M in (128, 256, 512)]
                + [f"torus:{M}:tc1_a{HALF_PI}:{MAIN},kappa=0,kappa_near=-1" for M in (128, 256)])


TIMESTEP = (
    [("torus", M, "tc5", 50, 1.0, 6.0, MAIN) for M in (64, 128, 256)]
    + [("torus", M, "tc5", 50, 1.0, 6.0, FD4) for M in (64, 128)]
    + [("torus", 64, "tc5", 50, 0.2, 6.0, configs.FRONTIER[k][0]) for k in configs.FRONTIER
       if k not in (configs.MAIN_NAME, configs.FD4_NAME, "square cover")]
    + [("torus", 128, "tc5", 50, 1.0, 8.0, TRSK)]
    + [("dfs", 45, "tc5", 5, 0.005, 0.5, "kappa=0,order=2,integrator=rk4"),
       ("dfs", 90, "tc5", 3, 0.005, 0.3, "kappa=0,order=2,integrator=rk4"),
       ("dfs", 180, "tc5", 1, 0.002, 0.2, "kappa=0,order=2,integrator=rk4"),
       ("dfs", 45, "tc5", 5, 0.005, 0.5, "kappa=0,order=4,integrator=rk4")]
)

FLOOR_SPECS = ["dfs:32:tc5:order=2,kappa=0,integrator=rk4,dt=90", "dfs:64:tc5:order=2,kappa=0,integrator=rk4,dt=45",
               "dfs:32:tc2_a0.7:order=2,kappa=0,integrator=rk4,dt=90", "dfs:64:tc2_a0.7:order=2,kappa=0,integrator=rk4,dt=45"]

DIAG_SPECS = [
    "torus:64:tc2_a0.7:order=4,kappa=0,integrator=rk4,courant=0.5",
    "torus:128:tc2_a0.7:order=4,kappa=0,integrator=rk4,courant=0.5",
    "torus:64:tc2_a0.7:order=4,kappa=0.002,integrator=rk4,courant=0.5",
    "torus:128:tc2_a0.7:order=4,kappa=0.002,integrator=rk4,courant=0.5",
    "torus:64:tc2_a0.7:order=4,form=cart,kappa=0,integrator=rk4,courant=0.5",
    "torus:64:tc2_a0.7:order=4,form=split,kappa=0,integrator=rk4,courant=0.5",
    "torus:64:tc2_a0.7:order=4,form=split,kappa=0.002,integrator=rk4,courant=0.5",
    "torus:128:tc2_a0.7:order=4,form=split,kappa=0.002,integrator=rk4,courant=0.5",
    "torus:64:tc2_a0.7:order=4,rings=4,kappa=0.008,integrator=rk4,courant=0.5,hv_mode=stage",
    "torus:128:tc2_a0.7:order=4,rings=4,kappa=0.008,integrator=rk4,courant=0.5,hv_mode=stage",
    "torus:64:tc2_a0.7:solver=lwt,kappa=0,courant=0.3",
    "torus:64:tc2_a0.7:solver=lwt,kappa=0.002,courant=0.3",
]


def done_specs(fn):
    out = set()
    if os.path.exists(fn):
        for l in open(fn):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("stable") is not None and r.get("commit") == COMMIT:
                out.add(r.get("spec"))
    return out


def run_suite(fn, specs, workers, force, env=None):
    todo = specs if force else [s for s in specs if s not in done_specs(fn)]
    if not todo:
        print(f"[{os.path.basename(fn)}] nothing to do", flush=True); return
    subprocess.run([PY, os.path.join(ROOT, "scripts/suite.py"), fn, str(workers)] + todo, check=True,
                   env=dict(os.environ, **(env or {})))


def run_parallel(cmds, workers, env=None):
    procs = []
    for cmd in cmds:
        while sum(p.poll() is None for p in procs) >= workers:
            time.sleep(2)
        print("running:", " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd[1:]), flush=True)
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, NUMBA_NUM_THREADS="1", **(env or {}))))
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"{sum(1 for r in rc if r)} commands failed")


STAMPS = os.path.join(RES, ".stamps")


def done(name):
    p = os.path.join(STAMPS, name)
    return os.path.exists(p) and open(p).read().strip() == COMMIT


def run_outputs(jobs, workers, force, env=None):

    names = sorted({n for n, _ in jobs if force or not done(n)})
    for n in names:
        for p in (os.path.join(RES, n), os.path.join(STAMPS, n)):
            if os.path.exists(p):
                os.remove(p)
    run_parallel([c for n, c in jobs if n in names], workers, env)
    os.makedirs(STAMPS, exist_ok=True)
    for n in names:
        open(os.path.join(STAMPS, n), "w").write(COMMIT + "\n")


def script(name, *args):
    return [PY, os.path.join(ROOT, "scripts", name)] + [str(a) for a in args]


def exp_namelist(dt, hdc, taue, icond, alpha=0.0, sits=True, tauo=24.0):
    return f""" $PHYVAR
 ALPHA  =  {alpha!r},
 HDC    =  {hdc:.4E},
 $END
 $EXPDEF
 DT     =   {dt},
 EGYFRQ =   24.0,
 ERRFRQ =   999.0,
 SPCFRQ =   999.0,
 TAUE   =   {taue},
 TAUO   =   {tauo},
 GPHFRQ =   999.0,
 SITS   =   {'.TRUE.' if sits else '.FALSE.'},
 FORCED =   .FALSE.,
 MOMENT =   .FALSE.,
 ICOND  =   {icond},
 $END
 $PLTDEF
 LGPHS  = .FALSE.,
 $END
 $FNAMES
 $END
"""


def _records(fn):
    try:
        sys.path.insert(0, ROOT)
        from polarswe import reference as ref
        return len(ref.read(fn)[3])
    except Exception:
        return 0


def _stswm_job(spec):
    T, name, dt, hdc, taue, icond, alpha, sits, tauo, init, force = spec
    run = os.path.join(REFD, "runs", name)
    out = os.path.join(run, "grid_out.bin")
    need = int(round(taue / tauo)) + 1
    stamp = os.path.join(run, "commit.txt")
    if not force and os.path.exists(out) and _records(out) >= need and os.path.exists(stamp) and open(stamp).read().strip() == COMMIT:
        return run + " (kept)"
    if os.path.exists(stamp):
        os.remove(stamp)
    os.makedirs(run, exist_ok=True)
    if init is not None:
        subprocess.run([PY, os.path.join(ROOT, init[0])] + [str(x) for x in init[1:]] + [os.path.join(run, "init_grid.bin")], check=True)
    open(os.path.join(run, "exp"), "w").write(exp_namelist(dt, hdc, taue, icond, alpha, sits, tauo))
    if os.path.exists(out):
        os.remove(out)
    exe = os.path.join(REFD, f"build_T{T}_ref", "stswm")
    subprocess.run(f"{exe} < exp > stdout 2> stderr", shell=True, cwd=run, check=True)
    if _records(out) < need:
        raise RuntimeError(f"{name}: incomplete output")
    open(stamp, "w").write(COMMIT + "\n")
    return run


SPECTRAL = {21: (32, 64), 42: (64, 128), 85: (128, 256), 170: (256, 512), 213: (320, 640), 341: (512, 1024)}
TC1_DT = {21: 900.0, 42: 900.0, 85: 450.0, 170: 225.0}


def hdc_for(T):
    return 8e12 * (213 * 214 / (T * (T + 1))) ** 2


def stage_refs(args):
    subprocess.run(script("validate_stswm.py", os.path.join(RES, "stswm_validation.json")), check=True)
    for T, (nlat, nlon) in SPECTRAL.items():
        subprocess.run(["./build.sh", str(T), str(nlat), str(nlon), "_ref"], cwd=REFD, check=True, capture_output=True)
    f = args.force
    w1 = [


        (213, "T213dt22_tc5", 22.5, 8.0e12, 360.0, 5, 0.0, True, 24.0, None, f),
        (213, "T213dt22_tc6", 22.5, 8.0e12, 336.0, 6, 0.0, True, 24.0, None, f),
        (341, "T341dt30_galewsky", 30.0, hdc_for(341), 144.0, 8, 0.0, True, 24.0, ("scripts/make_stswm_init.py", "galewsky", 512), f),
        (213, "T213dt45_tc5", 45.0, 8.0e12, 360.0, 5, 0.0, True, 24.0, None, f),
        (213, "T213dt45_tc6", 45.0, 8.0e12, 336.0, 6, 0.0, True, 24.0, None, f),
        (341, "T341dt60_galewsky", 60.0, hdc_for(341), 144.0, 8, 0.0, True, 24.0, ("scripts/make_stswm_init.py", "galewsky", 512), f),
        (341, "T341dt90_tc5", 90.0, hdc_for(341), 360.0, 5, 0.0, True, 24.0, None, f),
        (341, "T341dt90_tc6", 90.0, hdc_for(341), 336.0, 6, 0.0, True, 24.0, None, f),
        (213, "T213dt90_tc5", 90.0, 8.0e12, 360.0, 5, 0.0, True, 24.0, None, f),
        (213, "T213dt90_tc6", 90.0, 8.0e12, 336.0, 6, 0.0, True, 24.0, None, f),

        (213, "T213_tc5", 360.0, 8.0e12, 360.0, 5, 0.0, True, 24.0, None, f),
        (213, "T213_tc6", 180.0, 8.0e12, 336.0, 6, 0.0, True, 24.0, None, f),

        (42, "T42_case6", 600.0, 0.0, 336.0, 6, 0.0, True, 24.0, None, f),
        (42, "T42_case8_tc6", 600.0, 0.0, 336.0, 8, 0.0, True, 24.0, ("scripts/make_tc6_perturbed.py", 64, 0.0), f),
        (42, "T42_case8_jet", 600.0, 0.0, 144.0, 8, 0.0, True, 24.0, ("scripts/make_stswm_init.py", "galewsky_nopert", 64), f),

        (213, "T213_tc6_pert1m", 180.0, 8.0e12, 336.0, 8, 0.0, True, 24.0, ("scripts/make_tc6_perturbed.py", 320, 1.0), f),
        (85, "T85_tc6_zonal0", 150.0, hdc_for(85), 336.0, 8, 0.0, True, 24.0, ("scripts/make_tc6_zonal.py", 128, 0.0), f),
        (85, "T85_tc6_zonal1", 150.0, hdc_for(85), 336.0, 8, 0.0, True, 24.0, ("scripts/make_tc6_zonal.py", 128, 1.0), f),
        (85, "T85_tc6_2h", 45.0, hdc_for(85), 336.0, 6, 0.0, True, 2.0, None, f),

        (85, "T85dt45_tc6_hdc0.1", 45.0, 0.1 * hdc_for(85), 336.0, 6, 0.0, True, 24.0, None, f),
        (85, "T85dt45_tc6_hdc0.01", 45.0, 0.01 * hdc_for(85), 336.0, 6, 0.0, True, 24.0, None, f),
    ]
    for T, dt in ((42, 90.0), (85, 90.0)):
        w1 += [(T, f"T{T}dt90_tc5", dt, hdc_for(T), 360.0, 5, 0.0, True, 24.0, None, f),
               (T, f"T{T}dt90_tc6", dt, hdc_for(T), 336.0, 6, 0.0, True, 24.0, None, f)]
    for T, dt in ((42, 900.0), (85, 450.0), (170, 225.0)):
        w1 += [(T, f"T{T}_galewsky", dt, hdc_for(T), 144.0, 8, 0.0, True, 24.0, ("scripts/make_stswm_init.py", "galewsky", SPECTRAL[T][0]), f)]
    for T, dt in ((21, 1800.0), (42, 900.0), (85, 450.0), (170, 225.0)):
        w1 += [(T, f"T{T}_tc5", dt, hdc_for(T), 360.0, 5, 0.0, True, 24.0, None, f),
               (T, f"T{T}_tc6", dt, hdc_for(T), 336.0, 6, 0.0, True, 24.0, None, f),
               (T, f"T{T}_tc2", dt, 0.0, 120.0, 2, 0.7, True, 24.0, None, f),
               (T, f"T{T}_tc1", TC1_DT[T], 0.0, 288.0, 1, float(HALF_PI), False, 24.0, None, f)]
    w1.sort(key=lambda j: -(j[0] / 213.0) ** 3 * j[4] / j[2])
    with mp.Pool(max(1, args.ref_workers)) as pool:
        for run in pool.imap_unordered(_stswm_job, w1):
            print("STSWM:", os.path.relpath(run.split(" (")[0], ROOT), run.split(" ")[-1] if "(kept)" in run else "done", flush=True)

    seed = (213, "T213_tc6_seed", 180.0, 8.0e12, 336.0, 8, 0.0, True, 24.0,
            ("scripts/tc6_seed.py", 128, 320, os.path.join(REFD, "runs/T213dt22_tc6/grid_out.bin")), f)
    print("STSWM:", _stswm_job(seed), flush=True)
    fn = os.path.join(RES, "stswm_errors.jsonl")
    if os.path.exists(fn):
        os.remove(fn)
    subprocess.run(script("stswm_errors.py", fn), check=True)
    subprocess.run(script("check_interp.py", os.path.join(RES, "interp_check.json")), check=True)
    lr = lambda n: os.path.join(REFD, "runs", n, "grid_out.bin")
    subprocess.run(script("tc6_instability.py", lr("T42_case8_tc6"), lr("T213_tc6_pert1m"), lr("T213_tc6"),
                          os.path.join(RES, "tc6_instability.json"), lr("T42_case6"), lr("T42_case8_jet"),
                          lr("T85_tc6_zonal0"), lr("T85_tc6_zonal1"), lr("T213_tc6_seed")), check=True)


def stage_geometry(args):
    subprocess.run(script("geometry_report.py", os.path.join(RES, "geometry.json")), check=True)
    subprocess.run(script("integrator_check.py", os.path.join(RES, "integrator_check.json")), check=True)
    subprocess.run(script("geometry_report.py", os.path.join(RES, "geometry_square.json")), check=True,
                   env=dict(os.environ, POLARSWE_LATTICE="square"))


def stage_cone_grow(args):


    kap = configs.parse(MAIN)["kappa"]
    grow = [("cone_grow.jsonl", script("cone_study.py", "tc2", os.path.join(RES, "cone_grow.jsonl"), M, "cone=1", f"kappa={k}", f"R={R}"))
            for M, R in ((256, 11), (512, 11)) for k in (kap, 0.0)]
    run_outputs(grow, 1, args.force)


def stage_diagnostics(args, after_refs=False, grow=True):
    if not after_refs:
        run_outputs([("diagnostics.json", script("diag_branch.py", os.path.join(RES, "diagnostics.json"))),
                     ("linstab.json", script("linstab.py", os.path.join(RES, "linstab.json"))),
                     ("oracle.jsonl", script("oracle.py", os.path.join(RES, "oracle.jsonl"))),
                     ("tc6_twins.jsonl", script("tc6_mechanism.py", "twins", os.path.join(RES, "tc6_twins.jsonl")))],
                    args.workers, args.force)
        run_suite(os.path.join(RES, "diagnostics.jsonl"), DIAG_SPECS, args.workers, args.force)
        cone = [("cone_study.jsonl", script("cone_study.py", "ops", os.path.join(RES, "cone_study.jsonl"), M)) for M in (128, 256, 512)]
        cone += [("cone_study.jsonl", script("cone_study.py", "ops", os.path.join(RES, "cone_study.jsonl"), M, "cone=8_3_4_lines_cc")) for M in (128, 256, 512)]
        kap = configs.parse(MAIN)["kappa"]
        cone += [("cone_study.jsonl", script("cone_study.py", "tc2", os.path.join(RES, "cone_study.jsonl"), M, f"cone={c}", f"kappa={k}"))
                 for M in (64, 128, 256, 512) for c in (1, 0) for k in (kap, 0.0)]
        cone += [("cone_study.jsonl", script("cone_study.py", "tc2", os.path.join(RES, "cone_study.jsonl"), M, "cone=1", f"kappa={kap}",
                                             "label=filter in the patches", "extra=cone_filter=1")) for M in (64, 128, 256, 512)]

        cone += [("cone_tau.jsonl", script("cone_tau.py", os.path.join(RES, "cone_tau.jsonl"), M, a, lab, ov))
                 for M in (128, 256, 512) for a in (0.7, float(HALF_PI)) for lab, ov in (("main", ""), ("filter in the patches", "cone_filter=1"))]

        fn = os.path.join(RES, "cone_stability.jsonl")
        cone += [("cone_stability.jsonl", script("cone_stability.py", fn, M, a, lab, ov)) for M, alphas, labs in (
                    (256, (0.7, 1.4, 1.5207963267948966, 1.5707963267948966), CONE_STAB),
                    (512, (1.4, 1.5207963267948966), CONE_STAB[:4])) for a in alphas for lab, ov in labs]
        cone += [("cone_stability.jsonl", script("cone_stability.py", fn, 256, 1.5207963267948966, f"main, {o} orientation", f"orient={o}"))
                 for o in ("edge", "generic", "generic2")]
        run_outputs(cone, args.workers, args.force)
        run_suite(os.path.join(RES, "tracer.jsonl"), TRACER_SPECS, args.workers, args.force)
        if grow:
            stage_cone_grow(args)
    else:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import tc6_week2 as w2
        per_config = {"relax": len(w2.RELAX), "latlon": len(w2.LATLON), "filters": len(w2.FILTERS)}
        jobs = [("tc6_tilt.jsonl", script("tc6_mechanism.py", "tilt", os.path.join(RES, "tc6_tilt.jsonl")))]
        jobs += [(f"tc6_week2_{m}.jsonl", script("tc6_week2.py", m, os.path.join(RES, f"tc6_week2_{m}.jsonl")))
                 for m in ("restart", "sharpness", "twins7")]
        jobs += [(f"tc6_week2_{m}.jsonl", script("tc6_week2.py", m, os.path.join(RES, f"tc6_week2_{m}.jsonl"), k))
                 for m, n in per_config.items() for k in range(n)]

        jobs += [("trsk_beta.jsonl", script("trsk_beta.py", os.path.join(RES, "trsk_beta.jsonl"), M, b, it)) for M, b, it in TRSK_BETA]
        run_outputs(jobs, args.workers, args.force)
        run_outputs([("tc6_week2_restart_square.jsonl", script("tc6_week2.py", "restart_square", os.path.join(RES, "tc6_week2_restart_square.jsonl")))],
                    1, args.force, env={"POLARSWE_LATTICE": "square"})


def stage_timestep(args):
    fn = os.path.join(RES, "timestep.jsonl")
    have = set()
    if os.path.exists(fn) and not args.force:
        for l in open(fn):
            r = json.loads(l)
            if r.get("commit") == COMMIT:
                have.add((r["grid"][0], r["grid"][1], r["case"], r.get("scheme_str", "")))
    cmds = []
    for g, n, case, days, lo, hi, sc in TIMESTEP:
        if (g, n, case, sc) in have:
            continue
        cmds.append(script("maxdt.py", fn, g, n, case, days, lo, hi, sc))
    sq = configs.FRONTIER["square cover"][0]
    if ("torus", 64, "tc5", sq) not in have:
        cmds.append(script("maxdt.py", fn, "torus", 64, "tc5", 50, 1.0, 6.0, sq))
    sq_env = {"POLARSWE_LATTICE": "square"}
    normal = [c for c in cmds if c[-1] != sq]
    run_parallel(normal, args.workers)
    run_parallel([c for c in cmds if c[-1] == sq], 1, env=sq_env)


def sfx(refs_ready):

    return "_ref" if refs_ready else ""


def stage_accuracy(args, refs_ready):
    for tag, sc in (("main", MAIN), ("fd4", FD4)):
        specs = [f"{s}:{sc}" for s in accuracy_specs(args.highres and tag == "main")]
        specs += [f"torus:{M}:tc2_a0.7:{sc},days=15" for M in (64, 128, 256)]
        if tag == "main":
            specs += [f"torus:{M}:{c}:{sc},courant={configs.parse(sc)['courant'] / 4:g}" for M in (128, 256) for c in ("tc2_a0.7", "tc5", "tc6")]
        specs = [s for s in specs if needs_refs(s) == refs_ready]
        run_suite(os.path.join(RES, f"accuracy_{tag}{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force)


def stage_variants(args, refs_ready):
    specs = []
    for name, (sc, _) in configs.FRONTIER.items():
        if name in (configs.MAIN_NAME, "square cover"):
            continue
        for M in (64, 128):
            for c in FRONTIER_CASES:
                if c.startswith("tc1") and "solver=lwt" in sc:
                    continue
                s = f"torus:{M}:{c}:{sc}"
                if needs_refs(s) == refs_ready:
                    specs.append(s)
    run_suite(os.path.join(RES, f"variants{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force)
    sq = configs.FRONTIER["square cover"][0]
    specs = [f"torus:{M}:{c}:{sq}" for M in (64, 128) for c in FRONTIER_CASES]
    specs = [s for s in specs if needs_refs(s) == refs_ready]
    run_suite(os.path.join(RES, f"variants_square{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force, env={"POLARSWE_LATTICE": "square"})


def stage_mpas(args, refs_ready):


    levels = list(zip((4, 5, 6) + ((7,) if args.highres else ()), (64, 128, 256) + ((512,) if args.highres else ())))
    specs = []
    for L, M in levels[::-1]:
        specs += [f"icos:{L}:{c}:{MPAS}" for c in MPAS_CASES] + [f"torus:{M}:{c}:{MPAS}" for c in MPAS_CASES]
    specs = [s for s in specs if needs_refs(s) == refs_ready]
    run_suite(os.path.join(RES, f"mpas{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force)


def stage_orientation(args, refs_ready):

    specs = [f"torus:128:{c}:{MAIN},orient={o}" for o in ORIENTATIONS for c in FRONTIER_CASES]
    specs = [x for x in specs if needs_refs(x) == refs_ready]
    run_suite(os.path.join(RES, f"orientation{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force)
    sq = configs.FRONTIER["square cover"][0]
    specs = [f"torus:128:{c}:{sq},orient=generic" for c in FRONTIER_CASES]
    specs = [x for x in specs if needs_refs(x) == refs_ready]
    run_suite(os.path.join(RES, f"orientation_square{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force, env={"POLARSWE_LATTICE": "square"})


def stage_floor(args, refs_ready):
    specs = [s for s in FLOOR_SPECS if needs_refs(s) == refs_ready]
    run_suite(os.path.join(RES, f"floor{sfx(refs_ready)}.jsonl"), specs, args.workers, args.force)


def stage_series(args):
    fn = os.path.join(RES, "tc6_series.jsonl")
    Ms = (64, 128, 144, 160, 180, 256, 288) + ((512, 576) if args.highres else ())
    jobs = [("tc6_series.jsonl", script("tc6_series.py", fn, M)) for M in Ms]
    jobs += [("tc6_series.jsonl", script("tc6_series.py", fn, M, f"scheme={sc}")) for M in (128, 256)
             for sc in (configs.FRONTIER[configs.EDGE_NAME][0], PREV, configs.FRONTIER[configs.FD8F_NAME][0])]
    run_outputs(jobs, args.workers, args.force)
    runs = ["T213dt22_tc6", "T341dt90_tc6", "T170_tc6", "T85_tc6", "T85dt90_tc6", "T42_tc6", "T42dt90_tc6", "T21_tc6"]
    run_outputs([("aam_spectral.json", script("aam_spectral.py", os.path.join(RES, "aam_spectral.json"), *runs))], 1, args.force)


def stage_fp32(args):
    fn = os.path.join(RES, "fp32.jsonl")
    run_outputs([("fp32.jsonl", script("fp32_study.py", fn, "tc2_a0.7", M)) for M in (64, 128, 256) + ((512,) if args.highres else ())],
                args.workers, args.force)


FP32_GPU_M = (512, 1024, 2048)


def stage_fp32gpu(args, Ms=FP32_GPU_M):

    run_outputs([(f"fp32_gpu_M{M}.jsonl", script("fp32_study.py", os.path.join(RES, f"fp32_gpu_M{M}.jsonl"), "tc2_a0.7", M))
                 for M in Ms], 1, args.force, env={"POLARSWE_DEVICE": "gpu"})


def stage_tables(args):
    subprocess.run(script("tables.py"), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ref-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="archive existing result files first (data/previous/)")
    ap.add_argument("--highres", action="store_true", help="also run M = 512")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    if args.gpu:
        os.environ["POLARSWE_DEVICE"] = "gpu"
    os.makedirs(RES, exist_ok=True)
    if args.fresh:
        dst = os.path.join(RES, "previous", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(RES, "*.json*")):
            shutil.move(f, dst)
    single = dict(refs=stage_refs, geometry=stage_geometry, timestep=stage_timestep, series=stage_series,
                  fp32=stage_fp32, fp32gpu=stage_fp32gpu, tables=stage_tables,
                  orientation=lambda a: (stage_orientation(a, False), stage_orientation(a, True)),
                  diagnostics=lambda a: (stage_diagnostics(a), stage_diagnostics(a, True)),
                  accuracy=lambda a: (stage_accuracy(a, False), stage_accuracy(a, True)),
                  variants=lambda a: (stage_variants(a, False), stage_variants(a, True)),
                  floor=lambda a: (stage_floor(a, False), stage_floor(a, True)),
                  mpas=lambda a: (stage_mpas(a, False), stage_mpas(a, True)))

    NOREF = (("geometry", stage_geometry), ("diagnostics (part 1)", stage_diagnostics),
             ("accuracy (analytic references)", lambda a: stage_accuracy(a, False)),
             ("variants (analytic references)", lambda a: stage_variants(a, False)),
             ("floor (analytic references)", lambda a: stage_floor(a, False)),
             ("MPAS's scheme (analytic references)", lambda a: stage_mpas(a, False)),
             ("orientation (analytic references)", lambda a: stage_orientation(a, False)),
             ("fp32", stage_fp32), ("timestep", stage_timestep))
    WITHREF = (("accuracy (STSWM references)", lambda a: stage_accuracy(a, True)),
               ("variants (STSWM references)", lambda a: stage_variants(a, True)),
               ("floor (STSWM references)", lambda a: stage_floor(a, True)),
               ("MPAS's scheme (STSWM references)", lambda a: stage_mpas(a, True)),
               ("orientation (STSWM references)", lambda a: stage_orientation(a, True)),
               ("series", stage_series), ("diagnostics (part 2)", lambda a: stage_diagnostics(a, True)))
    def run_list(lst):
        for name, fn in lst:
            print(f"=== stage {name}", flush=True)
            fn(args)
    single.update(noref=lambda a: run_list(NOREF), withref=lambda a: run_list(WITHREF))
    if args.stages:
        for s in args.stages:
            print(f"=== stage {s}", flush=True)
            single[s](args)
        return

    print("=== stage refs (in parallel)", flush=True)
    ref_proc = mp.Process(target=stage_refs, args=(args,))
    ref_proc.start()
    run_list(NOREF)
    ref_proc.join()
    if ref_proc.exitcode != 0:
        raise SystemExit("reference stage failed")
    run_list(WITHREF + (("tables", stage_tables),))


if __name__ == "__main__":
    main()
