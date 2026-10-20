# Use or port the solver

`polarswe/tfd.py` is the CPU update; `fast.py` compiles it with Numba. Start with `examples/steady_flow.py` after [installation](REPRODUCE.md).

## Fields and stepping

`grid.torus_X(M)` returns unit-sphere Cartesian positions of shape `(3, M, M)`. M must be even and at least 64 for separate main-scheme cone patches. Both axes are periodic. Paired physical points must have equal initial fields; see [the map](METHOD.md).

Cases supply `height(X)`, `velocity(X)` and topography through `testcases.py`. Height and topography are metres; velocity is metres per second. Depth `D` is free-surface height minus topography. The three Cartesian velocity components are tangent to the sphere.

`experiments.build(X, case, configs.MAIN)` returns the solver, unfiltered metric, mean spacing in metres, initial maximum wave speed and proposed step in seconds. `S.step(D, v, dt)` returns new depth and velocity; copy the old state if needed. `S.mass(D)` uses corrected area weights. TC1 uses the separate skew tracer equation in `experiments.run_tc1`, not the continuity flux.

## Machine implementation

Preserve the full composed dependency reach or use an equivalent staged schedule. Preserve gradient/divergence adjointness, compatible curl, corrected mass weights and the patch dissipation mask.

The model packs depth and two tangent velocity components into three moving words. Python stores three Cartesian velocity components. A port must cost tangent-basis packing and reconstruction, including temporary storage. Positions, topography and cone coefficients move with the logical grid. `dataflow.dt_kernel` includes four static words, metric reconstruction and corner traffic; larger payloads need a new estimate.

Resolve the [coefficient-memory shortfall](PERFORMANCE.md#machine-and-kernel) by distribution or explicit streaming. Implement boundary crossings and test every cone alignment. Check SRAM use and time complete steps before claiming hardware speed. For FP32, use `fp32.py`'s perturbation form with FP64 reference-state preparation, then rerun the numerical cases in target arithmetic.

`dataflow.Machine` holds node/core dimensions, clock, directional bandwidth and latencies. `step_time` separates compute, I/O and latency limits; `cone_patch_cost` adds patch work and storage. `scripts/check_claims.py` connects these to measured stability.
