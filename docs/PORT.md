# Port the local update

Start with the CPU implementation in `polarswe/tfd.py`. `fast.py` compiles the same update with Numba. The tests compare their outputs and conservation. `examples/steady_flow.py` builds the main scheme, takes a short run and prints its score.

## Fields and units

`grid.torus_X(M)` returns unit-sphere Cartesian positions with shape `(3, M, M)`. M is even. The main cone patches need M at least 64 to stay separate. Both grid directions are periodic. Physical points are paired by `(i, j) -> (-1-i, -1-j)` modulo M. Initial physical fields must agree on paired positions.

The case object supplies `height(X)`, `velocity(X)` and topography through the interfaces in `testcases.py`. Height and topography are in metres, velocity in metres per second. The solver's `D` is depth, equal to free-surface height minus topography. Velocity has three Cartesian components tangent to the sphere. `experiments.build` returns the solver, an unfiltered metric instance, mean spacing in metres, initial maximum wave speed and a proposed step in seconds. The main configuration fixes the orientation, integrator, local dissipation and cone operators.

For direct stepping, call `S.step(D, v, dt)`. It returns new depth and velocity arrays. `S.mass(D)` uses the corrected area weights. Keep a copy if the caller needs the old state. The main configuration's tracer equation has a separate skew form in `experiments.run_tc1`; do not substitute the continuity flux equation for it.

## Carry these costs into a machine port

- Ten stages of four-point reach and a two-point dissipation pass give reach 42 per full step. Translation must cover the full composed dependency or use an equivalent staged schedule.
- The performance model encodes three moving state words, two tangent velocity components and depth. The Python solver stores three Cartesian velocity components. A three-word hardware implementation must pack velocity into a tangent basis and reconstruct it as needed. Its extra work and temporary storage need checking on the target.
- Positions and topography travel with the translated points. `dataflow.dt_kernel` budgets four static words and recomputes the metric from positions. It also counts the corner payload. Do not transmit a larger Python object and retain the same traffic estimate.
- Each cone point has 151 stored coefficient words in the main model. Coefficients must move with the logical grid. The worst core cannot keep all 96 patch points' entries in its assumed 48 kB. Distribute entries and work over nearby cores or account for streaming explicitly.
- Preserve the gradient/divergence adjoint relation and the compatible curl. Use the same corrected mass weights. Keep local dissipation off inside the eight-cell cone patches.
- In FP32, use the perturbation form in `fp32.py`. Its initial reference state and tendency are prepared in FP64. Measure its error on your own flows.

## Change the machine model

`dataflow.Machine` holds node counts, core dimensions, clock, per-side bandwidth and latencies. `step_time` returns compute, I/O and latency limits separately. `cone_patch_cost` adds the worst-core work and coefficient payload. `scripts/check_claims.py` shows the complete calculation from kernel counts and measured stability records to throughput.

Before a hardware claim, implement the translated schedule, packing, coefficient storage and boundary crossings. Check SRAM use, test all cone alignments, time complete steps and rerun the numerical cases in the target arithmetic. The published ratios are design estimates until that work is done.
