# Check the numbers

The release includes the saved score records, an independent claim calculator and all solver and reference-run code. The short checks do not rerun the numerical study. Full independent rescoring needs the long NCAR reference runs and solver studies below.

The commands shown here were run from a clean source copy with Python 3.14.7 on macOS. The default test set has 23 checks. A fresh source copy with no caches took 9.39 seconds using the clean installation. Package downloads, compilation and slower machines can add time. The longer cone and configuration checks are in `tests/extended.py`; the source suite's 28 checks also passed during packaging in about five minutes.

## Install and check the saved evidence

From the repository root:

```sh
./run.sh
.venv/bin/python scripts/check_claims.py --out build/claims.json
.venv/bin/python scripts/prepare_figures.py
.venv/bin/python examples/steady_flow.py
```

The calculator verifies the original score-file hashes and recorded revisions. It recalculates the 855.80x and 1262.69x modelled speed ratios, all 18 TRSK l2 ratios, the spectral comparison and the polar-filter trade. It also checks the FP32 study's scope. The figure pack contains those calculated numbers, not transcribed plot positions.

The example runs two steps of TC2 on 2,048 unique points. Its measured l2 was 2.89025e-6, with zero reported mass drift. It is a usage check, not the five-day convergence result. Building the main cone operators for a new grid is slower than stepping it; the build is cached.

## Build the independent reference model

The fetched source has separate research and education terms. Read [the notice](THIRD_PARTY.md) before running it. The validation needs a local `gfortran` compiler and `python3` in PATH.

```sh
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python scripts/validate_stswm.py build/stswm-validation.json
```

This downloads the exact checked archive, preserves its notices, builds the modified T42 model and compares 100 diagnostics with NCAR's original six-hour TC5 log. The largest relative difference is 1.27e-4, in potential enstrophy. The original log used 32-bit arithmetic; this build uses 64-bit arithmetic. This check does not validate every long reference field.

## Rerun the numerical study

```sh
.venv/bin/python -m polarswe.reproduce --help
```

The driver supplies stages for the references, geometry, timestep bisections, accuracy, alternative schemes, generated-mesh TRSK comparisons, orientation, diagnostics, TC6 series and FP32. The full no-stage invocation runs the CPU stages in dependency order. Its `--highres` option adds the high-resolution CPU cases through M=576. Individual named stages allow the work to be split into jobs. The `noref` and `withref` groups separate cases with analytic references from those that need NCAR fields.

The precision study through M=2048 is the separate `fp32gpu` stage. It needs a CUDA GPU with adequate FP64 performance and the matching CuPy distribution. The driver stops at M=2048. No M=4096 result is claimed. Records include a hash of the numerical code and scripts. A local Git revision or an optional `POLARSWE_COMMIT` label is included when available. A parent checkout is never used as the run identity.

These full stages need cluster time. The T213/T341 references, high-resolution TRSK runs, stability bisections, cone studies and GPU precision study were not rerun for this release. Some cone studies need about 4 GB for a dense solve alone. The saved run used CPU jobs and GPU jobs. There is no Cerebras job in this pipeline.

The driver writes new score records into `data/`. Its `--fresh` option archives existing JSON files before starting. Keep the release's saved evidence in a separate copy. Repeating a stage resumes records tagged with the same run identifier. The final table stage recalculates the available comparisons; smaller runs will have fewer pairs. Use the claim calculator's `--rerun` mode for a new dataset, since the release's hash and expected-range checks are specific to the saved run. Full reference fields must be present to independently rescore TC5, TC6 and the perturbed jet.

Only the short commands above are presented as release-tested commands. The long study is supplied as code and a description of its stages, rather than an untested launch recipe.
