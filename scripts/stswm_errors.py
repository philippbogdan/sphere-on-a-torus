"""Run a numerical check and save its measurements."""


import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import reference as ref, testcases as tc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "refs/stswm/runs")
run = lambda name: os.path.join(RUNS, name, "grid_out.bin")


def grid_X(g, nlon):
    lon = 2 * np.pi * np.arange(nlon) / nlon
    L, P = np.meshgrid(lon, g)
    return np.stack([np.cos(P) * np.cos(L), np.cos(P) * np.sin(L), np.sin(P)]), P


def norms(h, href, nlat):
    _, wq = np.polynomial.legendre.leggauss(nlat)
    Wt = np.broadcast_to(wq[::-1][:, None], h.shape)
    I = lambda f: (Wt * f).sum()
    e = h - href
    return dict(l1=float(I(np.abs(e)) / I(np.abs(href))), l2=float(np.sqrt(I(e * e) / I(href * href))),
                linf=float(np.abs(e).max() / np.abs(href).max()), rms_err_m=float(np.sqrt(I(e * e) / I(np.ones_like(e)))))


def against(fn, reffn, tau):
    nlon, nlat, g, recs = ref.read(fn)
    h = [r[3] for r in recs if abs(r[0] - tau) < 1e-6][0]
    _, _, gr, rr = ref.read(reffn)
    X, _ = grid_X(g, nlon)
    href = ref.SphereInterp(gr, [r[3] for r in rr if abs(r[0] - tau) < 1e-6][0])(X)
    return dict(norms(h, href, nlat), npts=int(nlon * nlat))


def analytic(fn, case, tau):
    nlon, nlat, g, recs = ref.read(fn)
    h = [r[3] for r in recs if abs(r[0] - tau) < 1e-6][0]
    X, _ = grid_X(g, nlon)
    return dict(norms(h, case.height(X, tau * 3600.0), nlat), npts=int(nlon * nlat))


def emit(out, rec):
    print(json.dumps(rec))
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    out = sys.argv[1]
    DT = {21: 1800.0, 42: 900.0, 85: 450.0, 170: 225.0}
    TC6_DAYS = (24.0, 72.0, 120.0, 168.0, 240.0, 336.0)
    def by_day(fn, reffn):
        return {f"{t / 24:.0f}": against(fn, reffn, t)["l2"] for t in TC6_DAYS}
    for T in (21, 42, 85, 170):
        for c, tau in (("tc5", 360.0), ("tc6", 336.0)):
            for tag, reffn in (("T213dt22", run(f"T213dt22_{c}")), ("T213archived", run(f"T213_{c}"))):
                rec = dict(against(run(f"T{T}_{c}"), reffn, tau), model="STSWM", truncation=T, dt=DT[T], case=c, against=tag)
                if c == "tc6" and tag == "T213dt22":
                    rec["l2_by_day"] = by_day(run(f"T{T}_{c}"), reffn)
                emit(out, rec)
            if T in (42, 85):
                rec = dict(against(run(f"T{T}dt90_{c}"), run(f"T213dt22_{c}"), tau), model="STSWM", truncation=T, dt=90.0, case=c,
                           against="T213dt22")
                if c == "tc6":
                    rec["l2_by_day"] = by_day(run(f"T{T}dt90_{c}"), run(f"T213dt22_{c}"))
                emit(out, rec)
        if T >= 42:
            emit(out, dict(against(run(f"T{T}_galewsky"), run("T341dt30_galewsky"), 144.0), model="STSWM", truncation=T, dt=DT[T],
                           case="galewsky", against="T341dt30"))
        emit(out, dict(analytic(run(f"T{T}_tc1"), tc.TC1(np.pi / 2), 288.0), model="STSWM", truncation=T, dt={21: 900.0, 42: 900.0, 85: 450.0, 170: 225.0}[T],
                       case="tc1_a1.5707963267948966", against="analytic"))
        emit(out, dict(analytic(run(f"T{T}_tc2"), tc.TC2(0.7), 120.0), model="STSWM", truncation=T, dt=DT[T],
                       case="tc2_a0.7", against="analytic"))

    for name in ("T85_tc6_2h", "T85dt45_tc6_hdc0.1", "T85dt45_tc6_hdc0.01"):
        rec = dict(against(run(name), run("T213dt22_tc6"), 336.0), model="STSWM", truncation=85, dt=45.0, case="tc6",
                   against="T213dt22", kind="dissipation_sensitivity", run=name,
                   hdc_factor={"T85_tc6_2h": 1.0, "T85dt45_tc6_hdc0.1": 0.1, "T85dt45_tc6_hdc0.01": 0.01}[name])
        rec["l2_by_day"] = by_day(run(name), run("T213dt22_tc6"))
        emit(out, rec)


    for c, taus in (("tc5", (120.0, 240.0, 360.0)), ("tc6", (24.0, 72.0, 120.0, 168.0, 240.0, 336.0)),
                    ("galewsky", (48.0, 96.0, 144.0))):
        pairs = (((f"T341dt60_{c}", f"T341dt30_{c}"),) if c == "galewsky" else
                 ((f"T213dt90_{c}", f"T341dt90_{c}"), (f"T213dt45_{c}", f"T213dt22_{c}"),
                  (f"T213_{c}", f"T213dt22_{c}"), (f"T341dt90_{c}", f"T213dt22_{c}"), (f"T170_{c}", f"T213dt22_{c}")))
        for name, refname in pairs:
            byday = {f"{tau / 24:.0f}": against(run(name), run(refname), tau) for tau in taus}
            last = byday[f"{taus[-1] / 24:.0f}"]
            emit(out, dict(last, model="STSWM", run=name, case=c, against=refname.replace(f"_{c}", ""),
                           kind="reference_uncertainty", l2_by_day={d: v["l2"] for d, v in byday.items()},
                           rms_by_day_m={d: v["rms_err_m"] for d, v in byday.items()}))
