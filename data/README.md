# Numerical evidence

These records support the measured stability and accuracy comparisons and the modelled throughput. [`check_claims.py`](../scripts/check_claims.py) joins the relevant records, verifies their digests and recalculates the headline numbers; [REPRODUCE.md](../REPRODUCE.md) gives the commands.

| Files | What they establish |
|---|---|
| `accuracy_main.jsonl`, `accuracy_main_ref.jsonl` | Main-scheme errors, stability, steps and mass drift, with analytic or numerical references |
| `mpas.jsonl`, `mpas_ref.jsonl` | TRSK errors on generated centroidal icosahedral and torus-point meshes |
| `tc6_series.jsonl`, `stswm_errors.jsonl` | TC6 error over time, spectral comparisons and reference uncertainty |
| `timestep.jsonl` | Stability brackets, test durations and polar step-size scaling |
| `fp32.jsonl`, `fp32_gpu_M*.jsonl` | TC2 precision differences between FP32 and FP64 |
| `geometry.json`, `geometry_square.json` | Cell-area variation, distortion and the conformal-map comparison |
| `cone_*.jsonl`, `tracer.jsonl`, `oracle.jsonl` | Local consistency, dissipation, tracer behaviour and cone-error sensitivity |
| `accuracy_fd4*.jsonl`, `variants*.jsonl`, `orientation*.jsonl`, `floor*.jsonl` | Alternative schemes, map orientations and latitude-longitude comparisons |
| `diagnostics.json`, `diagnostics.jsonl`, `linstab.json`, `trsk_beta.jsonl` | Operator errors, linear stability and mesh sensitivity |
| `tc6_week2_*.jsonl`, `tc6_twins.jsonl`, `tc6_tilt.jsonl`, `tc6_instability.json` | Late-time wave behaviour, perturbations and reference-based restarts |
| `aam_spectral.json`, `integrator_check.json`, `interp_check.json`, `stswm_validation.json` | Momentum diagnostics, time integrator, interpolation and reference-build checks |

JSONL files contain one record per line. `l2` is normalised height error, `linf` is normalised maximum height error, and `rms_err_m` is height RMS error in metres. `nunique` counts physical sphere points; the torus computes each twice. `r_max_computed` and `r_max_unique` use different spacing conventions, defined in [PERFORMANCE.md](../docs/PERFORMANCE.md#stable-step).

The `mpas` filenames identify this repository's TRSK comparison, not measurements from the MPAS executable. `wall_s` is CPU/GPU execution time, not Cerebras timing. `spec`, `scheme` and `scheme_full` identify numerical configurations. Missing values remain missing and must not be plotted as zero.

[`manifest.json`](manifest.json) lists score-file hashes. [`source-stamps.json`](source-stamps.json) and record-level `commit` fields retain consistency identifiers used by the checks. They do not make a fresh computation equivalent to the included evidence.

NCAR reference fields are fetched and computed separately under their [own terms](../THIRD_PARTY.md). The scores here are measurements of the solver comparisons, not redistributed NCAR fields.
