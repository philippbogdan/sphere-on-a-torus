# Saved evidence

Measurements come from revision `27793ec4662ca401fba9c9fa4da8cdaedf97d21e`. `source-stamps.json` retains diagnostic/precision revision stamps; `manifest.json` hashes every original score file. Numerical bodies are unchanged.

| Files | Contents |
|---|---|
| `accuracy_main*.jsonl` | Main errors, stability, steps, mass drift, alternate references |
| `mpas*.jsonl` | Our TRSK on generated icosahedral and torus-point meshes |
| `tc6_series.jsonl`, `stswm_errors.jsonl` | TC6 time series and spectral scores |
| `timestep.jsonl` | Stability bisections and polar scaling |
| `fp32*.jsonl` | TC2 precision, including GPU through M=2048 |
| `geometry*.json` | Cell sizes, distortion, conformal-map comparison |
| `cone_*.jsonl`, `tracer.jsonl` | Cone consistency, dissipation, tracer checks |
| Other scores | Alternative schemes, reference validation, orientation, second-week diagnostics |

`l2` is normalised height error. `wall_s` is CPU/GPU runtime, not Cerebras timing. The historical name `mpas` does not mean MPAS was run. `spec` and `scheme` identify configurations. Missing scores remain missing.

[Reproduction](../docs/REPRODUCE.md) explains checking these records and regenerating unbundled reference fields.
