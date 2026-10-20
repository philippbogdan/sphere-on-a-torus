"""Run a numerical check and save its measurements."""


import sys, os, json, subprocess, multiprocessing as mp, traceback, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_commit():
    from polarswe.revision import current
    return current()


def parse(spec):
    parts = spec.split(":")
    g, n, case = parts[0], int(parts[1]), parts[2]
    kw = {}
    if len(parts) > 3 and parts[3]:
        for item in parts[3].split(","):
            k, v = item.split("=")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            kw[k] = v
    return (g, n), case, kw


def work(spec):
    warnings.filterwarnings("ignore")
    from polarswe import experiments as ex
    try:
        g, case, kw = parse(spec)
        days = kw.pop("days", None)
        dt = kw.pop("dt", None)
        r = ex.run(g, case, days=days, scheme=kw, dt=dt)
        r["spec"] = spec
        r["commit"] = git_commit()
        return r
    except Exception:
        return dict(spec=spec, error=traceback.format_exc())


if __name__ == "__main__":
    out, nproc, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]

    from polarswe import grid, trsk, experiments as ex
    built = set()
    for s in specs:
        (g, n), _, kw = parse(s)
        if g == "torus":
            X = grid.torus_X(n)
            sc = ex.scheme_of({k: v for k, v in kw.items() if k not in ("days", "dt")})
            key = (n, sc.get("orient"), sc.get("cone"), sc.get("order"))
            if sc.get("cone") and sc.get("solver") is None and key not in built:
                ex.cone_of(grid.oriented(X, sc.get("orient")), sc); built.add(key)
        elif g == "icos":
            trsk.icos_mesh(n)
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(work, specs):
            with open(out, "a") as f:
                f.write(json.dumps(r, default=float) + "\n")
            print(r.get("spec"), {k: r.get(k) for k in ("stable", "l1", "l2", "linf", "wall_s", "error")}, flush=True)
