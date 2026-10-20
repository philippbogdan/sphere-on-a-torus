# Saved numerical evidence

The paired comparisons give lower height l2 error for the torus in all 18 cases and resolutions. The smallest advantage is 1.16x on TC6 at day 14. The dataflow calculation gives about 856x throughput against the polar-limited baseline, or 1,263x with an additional patch-sharing estimate. Those speeds are model outputs, not timings.

The measurements come from source revision `27793ec4662ca401fba9c9fa4da8cdaedf97d21e`. Suite and timestep records carry that revision. `source-stamps.json` preserves the run's revision stamps for the diagnostic and precision files that do not carry it in each row. `manifest.json` gives the SHA-256 of every original score file. Numerical bodies are retained unchanged.

| Files | What they establish |
|---|---|
| `accuracy_main*.jsonl` | Main scheme errors, stability, steps, mass drift and alternate reference scores |
| `mpas*.jsonl` | This TRSK implementation on generated icosahedral and torus-point meshes |
| `tc6_series.jsonl`, `stswm_errors.jsonl` | TC6 day-5 and day-14 comparisons and NCAR spectral scores |
| `timestep.jsonl` | Stable-step bisections and the polar scaling used in the model |
| `fp32*.jsonl` | TC2 FP32/FP64 comparisons, including GPU measurements through M=2048 |
| `geometry*.json` | Cell sizes, distortion and the conformal-map comparison |
| `cone_*.jsonl`, `tracer.jsonl` | Local cone consistency, dissipation and tracer checks |
| Remaining score files | Alternative schemes, reference validation, orientation and second-week diagnostics |

`l2` is the normalised height error, not an elapsed time. `wall_s` is a CPU or GPU run time and is not used as Cerebras performance evidence. The short original file name `mpas` is retained for traceability; it does not mean the MPAS executable was run. `spec` and `scheme` identify the exact configuration. Missing scores remain missing.

`scripts/check_claims.py` reads the original records and recalculates the main ratios. `scripts/prepare_figures.py` turns those calculations into the small CSV pack used for the figures. Neither needs NCAR source or a long solver run.

Full reference fields are not bundled. They must be regenerated with the separately fetched NCAR model to independently rescore the simulations. The short checks verify numerical properties and saved-score arithmetic. They do not rerun the full study.
