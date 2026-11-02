# Use the solver or port its local update

`polarswe` supplies the spherical geometry, cone operators, time stepping and comparison cases. Start with [`examples/steady_flow.py`](../examples/steady_flow.py) after [installation](../REPRODUCE.md). The CPU implementation is executable; the CS-3 throughput remains a model.

## Fields and stepping

[`grid.torus_X(M)`](../polarswe/grid.py) returns unit-sphere Cartesian positions with shape `(3, M, M)`. Both axes are periodic. M must be even; use at least M = 64 to keep the main scheme's cone patches separate. Initialise identical physical fields at paired positions `(i, j)` and `(-1-i, -1-j)`, modulo M.

[`testcases.py`](../polarswe/testcases.py) supplies `height(X)`, `velocity(X)` and `topography(X)`. Height and topography are in metres. Depth `D` is height minus topography; velocity has three Cartesian components in metres per second, tangent to the sphere.

[`experiments.build(X, case, configs.MAIN)`](../polarswe/experiments.py) returns `(S, S0, h, cmax, dt)`: the stepping solver, unfiltered metric solver, mean spacing in metres, initial maximum wave speed in metres per second and proposed step in seconds. `S.step(D, v, dt)` advances depth and velocity. Copy a state before stepping if it must be retained. `S.mass(D)` uses the corrected area weights; on the double cover this integrates both copies.

[`tfd.py`](../polarswe/tfd.py) defines the update and [`fast.py`](../polarswe/fast.py) compiles it with Numba. TC1 tracer transport uses the separate skew equation in `experiments.run_tc1`, rather than the continuity flux. Preserve that distinction when adding a transported field.

## GPU arithmetic

`experiments.build` accepts `device="gpu"` with a compatible CuPy installation. Geometry and cone coefficients are prepared on the CPU and transferred; depth and velocity must also be device arrays. [`fp32.py`](../polarswe/fp32.py) provides plain and perturbation-form FP32, with FP64 preparation of the reference state and tendency. The supplied precision evidence covers TC2 only.

The [reproduction guide](../REPRODUCE.md#gpu-precision) gives the optional GPU precision command. The measured CPU/GPU timings are not predictions for Cerebras hardware.

## Porting to a local dataflow machine

Preserve the full composed dependency reach, or use an equivalent staged schedule. Gradient/divergence adjointness, compatible curl, corrected mass weights and the cone dissipation mask are numerical requirements, not optional optimisations.

The model packs depth and two tangent velocity components into three moving words. Python stores three Cartesian velocity components. A port must account for tangent-basis packing, reconstruction and temporary storage. Positions, topography and cone coefficients move with the logical grid. `dataflow.dt_kernel` includes metric reconstruction, corner traffic and four static words; a larger payload requires a new estimate.

Resolve the [coefficient-memory shortfall](PERFORMANCE.md#kernel-and-memory) by distributing coefficients or explicitly streaming them. Implement boundary crossings and check every cone alignment. Measure SRAM use and complete-step time before reporting hardware throughput. The proposed sharing of patch work remains unimplemented.

[`dataflow.Machine`](../polarswe/dataflow.py) holds the CS-3 system dimensions, core dimensions, clock, bandwidth and latency assumptions. `step_time` separates compute, I/O and latency limits; `cone_patch_cost` adds work and storage. [`check_claims.py`](../scripts/check_claims.py) combines these estimates with measured stability limits.
