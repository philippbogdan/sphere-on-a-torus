"""Run a numerical check and save its measurements."""


import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from polarswe import reference as ref


def W(nlat, shape):
    _, wq = np.polynomial.legendre.leggauss(nlat)
    return np.broadcast_to(wq[::-1][:, None], shape)


def rms_diff(fa, fb):
    _, nlat, _, a = ref.read(fa); _, _, _, b = ref.read(fb)
    w = W(nlat, a[0][3].shape)
    return {f"{x[0]/24:.0f}": float(np.sqrt((w * (x[3] - y[3]) ** 2).sum() / w.sum())) for x, y in zip(a, b)}


def main(c8, per, rfn, out, c6, jet, z0, z1, seed):
    res = {}
    _, _, _, a = ref.read(c8); _, _, _, b = ref.read(c6)
    res["case8_vs_case6_T42_max_abs_m"] = {f"{x[0]/24:.0f}": float(np.abs(x[3] - y[3]).max()) for x, y in zip(a, b)}
    _, _, _, j = ref.read(jet)
    res["jet_T42_max_abs_change_m"] = {f"{t/24:.0f}": float(np.abs(hj - j[0][3]).max()) for t, _, _, hj in j}
    res["T213_tetrahedral_twin_rms_m"] = rms_diff(per, rfn)
    res["T85_zonal_twin_rms_m"] = rms_diff(z1, z0)
    res["T213_seeded_with_torus_day1_error_rms_m"] = rms_diff(seed, rfn)
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        print(k, " ".join(f"d{d}:{x:.2e}" for d, x in v.items()))


if __name__ == "__main__":
    main(*sys.argv[1:10])
