"""Save a Galewsky jet time series on the periodic double cover.

The compressed archive contains unit-sphere positions X, relative vorticity
in 1/s, surface height in metres, snapshot times in seconds and cone geometry.
The requested snapshot interval is rounded to a whole number of solver steps;
use t_s for playback. The archive includes the initial and final states."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from polarswe import configs, experiments, testcases as tc


def vorticity(S, v):
    """Return relative vorticity in 1/s, including the cone curl correction."""
    base = S.S if hasattr(S, 'S') else S
    zeta = base.m.curl(v, base.order)
    if base.cone is not None:
        zeta = zeta + base.cone.curl(v)
    return zeta


def branch_points(M):
    """Return the periodic half-index coordinates of the four cone points."""
    h = M / 2 - 0.5
    return [[-0.5, -0.5], [h, -0.5], [-0.5, h], [h, h]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--M', type=int, default=512, help='Even torus width; each physical point is represented twice.')
    ap.add_argument('--days', type=float, default=6.0, help='Simulated duration in days.')
    ap.add_argument('--every-hours', type=float, default=1.0, help='Requested snapshot interval in simulated hours, rounded to solver steps.')
    ap.add_argument('--case', default='galewsky', help='Case name accepted by experiments.make_case.')
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'hero.npz', help='Destination for the compressed field archive.')
    args = ap.parse_args()
    case = experiments.make_case(args.case)
    sc = experiments.scheme_of(configs.MAIN)
    X, info = experiments.make_grid(('torus', args.M), sc.get('orient'))
    S, S0, h, cmax, dt = experiments.build(X, case, sc, allow_sign_change=info['allow_sign_change'])
    T = args.days * tc.DAY
    n = int(np.ceil(T / dt - 1e-9))
    # Adjust the step to land exactly on the requested final time.
    dt = T / n
    # The saved timestamps, rather than the requested interval, define playback.
    every = max(1, int(round(args.every_hours * 3600 / dt)))
    D = np.ascontiguousarray(case.height(X) - S0.hs)
    v = np.ascontiguousarray(case.velocity(X))
    m0 = float(S.mass(D))
    # Store display fields in FP32 without changing the stepping state.
    frames, heights, times = [vorticity(S, v).astype(np.float32)], [(D + S0.hs).astype(np.float32)], [0.0]
    t0 = time.time()
    for k in range(1, n + 1):
        D, v = S.step(D, v, dt)
        if k % every == 0 or k == n:
            frames.append(vorticity(S, v).astype(np.float32))
            heights.append((D + S0.hs).astype(np.float32))
            times.append(k * dt)
            if not np.isfinite(D).all():
                raise RuntimeError('The depth field contains non-finite values.')
            print(f'Step {k}/{n}; simulated day {k * dt / tc.DAY:.2f}; elapsed {time.time() - t0:.0f} s', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(case=args.case, M=args.M, scheme=configs.MAIN, dt_s=dt, nsteps=n, mean_spacing_m=h,
                mass_rel=float(S.mass(D)) / m0 - 1, wall_s=time.time() - t0, branch_points_ij=branch_points(args.M))
    np.savez_compressed(args.out, X=X.astype(np.float32), vorticity=np.array(frames), height=np.array(heights),
                        t_s=np.array(times), branch_distance_cells=experiments.branch_distance_cells(args.M).astype(np.float32),
                        meta=json.dumps(meta))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
