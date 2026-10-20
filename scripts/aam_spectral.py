"""Run a numerical check and save its measurements."""


import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import reference as ref, testcases as tc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(out, runs):
    res = {}
    for name in runs:
        nlon, nlat, glat, recs = ref.read(os.path.join(ROOT, "refs/stswm/runs", name, "grid_out.bin"))
        x, wq = np.polynomial.legendre.leggauss(nlat)
        W = np.broadcast_to(wq[::-1][:, None], (nlat, nlon))
        cg = np.cos(glat)[:, None]
        aam = lambda r: float((W * r[3] * r[1] * cg).sum())
        a0 = aam(recs[0])
        res[name] = {str(int(r[0] / 24)): (aam(r) - a0) / a0 for r in recs if r[0] % 24 == 0}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
