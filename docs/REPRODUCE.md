# Check the numbers

Use Python 3.11 or newer. These short commands were tested from a clean copy on macOS with Python 3.14.7. The 23 default checks took 9.39 seconds without caches; installation and compilation add time. They check code properties, saved-data integrity and claim arithmetic, not a fresh numerical study.

## Install and inspect

From the repository root:

```sh
./run.sh
.venv/bin/python scripts/check_claims.py --out build/claims.json
.venv/bin/python scripts/prepare_figures.py
.venv/bin/python examples/steady_flow.py
```

`run.sh` creates an environment and installs pinned packages. The calculator checks score hashes/revisions and rebuilds the speed, accuracy, spectral, filter and precision comparisons. The example takes two TC2 steps on 2,048 unique points: l2 was 2.89025e-6, with zero reported mass drift. This is not the five-day convergence test. New cone-operator builds are cached.

Longer cone/configuration checks are in `tests/extended.py`. The 28-check source suite passed in about five minutes during packaging.

## Build NCAR references

NCAR STSWM is restricted to scientific research and education. Its terms ask that resulting publications acknowledge NCAR and substantial modifications be identified as derivatives. Read the [full notice](THIRD_PARTY.md). With `gfortran` and `python3` in PATH:

```sh
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python scripts/validate_stswm.py build/stswm-validation.json
```

The fetch verifies the archive and preserves its notices. The modified FP64 T42 build matched 100 diagnostics from NCAR's original FP32 six-hour TC5 log. Maximum relative differences were 1.22e-6 for mass, 9.45e-6 for energy, 1.27e-4 for potential enstrophy and 2.11e-5 for grid height. This does not validate every long reference field.

## Rerun the study

```sh
.venv/bin/python -m polarswe.reproduce --help
```

The driver covers references, geometry, stability bisections, accuracy, alternative schemes, TRSK, orientation, diagnostics, TC6 series and precision. With no stage it runs CPU stages in dependency order. `--highres` adds CPU cases through M=576; named stages split jobs. `noref` and `withref` separate analytic and NCAR references.

The separate `fp32gpu` stage needs CUDA, adequate FP64 performance and matching CuPy. It stops at M=2048; no M=4096 result is claimed. Some cone studies need about 4 GB for a dense solve alone.

Full stages need cluster time. T213/T341 references, high-resolution TRSK, bisections, cone studies and GPU precision were not rerun for this release. Only the short commands above were release-tested. Independent rescoring of TC5, TC6 and the perturbed jet needs regenerated reference fields.

Keep saved evidence in a separate copy: the driver writes to `data/`, and `--fresh` archives existing JSON files. Records carry a code/script hash plus an available local Git revision or optional `POLARSWE_COMMIT` label. Matching run identifiers resume records. The table stage compares available pairs; smaller runs have fewer. Use the claim calculator's `--rerun` mode for new data, bypassing release-specific hashes and expected ranges.

Figure rebuild commands are in [FIGURES.md](FIGURES.md); saved-file meanings are in [data/](../data/).
