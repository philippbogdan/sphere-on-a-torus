"""Run a numerical check and save its measurements."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import reference as ref, testcases as tc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = {}
rng = np.random.default_rng(2)
X = rng.normal(size=(3, 50000)); X /= np.linalg.norm(X, axis=0)
for name, C in (("tc5", tc.TC5()), ("tc6", tc.TC6())):
    n, l, g, r = ref.read(os.path.join(ROOT, f"refs/stswm/runs/T213_{name}/grid_out.bin"))
    res[name] = float(np.abs(ref.SphereInterp(g, r[0][3])(X) - C.height(X)).max())
json.dump(res, open(sys.argv[1], "w")); print(res)
