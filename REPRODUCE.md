# Check the results and repeat the calculations

The included scores reproduce the throughput estimates, accuracy ratios, stable steps, spectral comparison and precision bounds without downloading reference fields. From the repository root, with Python 3.11 or newer:

```sh
./run.sh
```

This creates `.venv`, installs the pinned dependencies, runs the default tests and recalculates the claims. The score calculation itself takes under a second after installation; it does not repeat the fluid simulations. Installation needs package access. With an existing environment, the checks are local.

## Inspect the numbers

```sh
.venv/bin/python scripts/check_claims.py --out build/claims.json
.venv/bin/python scripts/prepare_figures.py
```

The calculator checks every file listed in `data/manifest.json` against its SHA-256 digest, checks consistency of the stored revision labels, and asserts the headline ranges. The JSON output includes all paired errors and model inputs. The second command writes the [figure tables](figure-pack/README.md) from the same calculation.

Try the solver on a short analytic case, or request the more expensive cone and configuration tests separately:

```sh
.venv/bin/python examples/steady_flow.py
.venv/bin/python -m pytest -q tests/extended.py
```

The example reports stability, unique-point count, step count, height error and mass drift. It is a short stepping example, not a convergence measurement.

## Recompute the numerical comparisons

Use a separate working copy for new measurements: the reproduction driver writes into `data/` and can replace existing results. An ordinary CPU can execute the stages, but high-resolution reference fields, TRSK meshes and dense cone fits are substantial calculations. Select worker counts to fit available memory. The full command below archives existing scores with `--fresh` and recomputes them with `--force`, avoiding duplicate records from different code fingerprints.

NCAR STSWM is needed for the non-analytic references. Read its [research and education terms](THIRD_PARTY.md#ncar-stswm-20), then fetch the archive. Building also requires `gfortran` and `python3` in PATH.

```sh
.venv/bin/python scripts/fetch_reference.py
.venv/bin/python scripts/validate_stswm.py build/stswm-validation.json
.venv/bin/python -m polarswe.reproduce --fresh --force --highres --workers 1 --ref-workers 1
```

The fetch verifies the archive hash. The validation compares the modified FP64 build with the original FP32 TC5 diagnostics supplied by NCAR. The historical reference in the saved validation record describes NCAR's original 1992 calculation, not a timing measurement of this solver. The validation does not check every long reference field. No NCAR source or reference-field archive is bundled here.

With no stage name, the driver builds references and executes the CPU stages in dependency order. `--highres` includes the finest paired accuracy cases. Once the full calculation has produced compatible records, select individual stages as needed:

```sh
.venv/bin/python -m polarswe.reproduce --help
.venv/bin/python -m polarswe.reproduce refs --ref-workers 1
.venv/bin/python -m polarswe.reproduce accuracy mpas series --highres --workers 1
.venv/bin/python -m polarswe.reproduce timestep --workers 1
```

Build references before stages that use them. `noref` selects analytic-reference calculations and `withref` selects calculations needing NCAR fields. Other stages cover geometry, alternative schemes, orientation, cone diagnostics and precision. The `mpas` stage uses this repository's TRSK implementation and generated meshes.

Matching specifications and code fingerprints allow existing records to be reused. `--force` recomputes eligible outputs; `--fresh` first moves existing JSON files into `data/previous/`. Recompute all inputs used by a comparison before interpreting it. New code fingerprints include changes to source text as well as numerical code.

For a fresh set of measurements, bypass the bundled-data digests and expected headline ranges:

```sh
.venv/bin/python scripts/check_claims.py --rerun --out build/new-claims.json
```

This still needs compatible input records. Missing resolution pairs are omitted where supported; `--rerun` does not fill in missing results or establish agreement with the bundled scores.

## GPU precision

Install CuPy for the local CUDA environment separately, then use one GPU:

```sh
.venv/bin/python -m polarswe.reproduce fp32gpu --workers 1
```

This stage compares plain and perturbation-form FP32 with FP64 through M = 2048. It needs sufficient memory and FP64 capability. The ordinary solver and saved-data checks need no GPU.

## Movie data and figure inputs

Install the plotting dependencies into the same environment. SciPy is already a required solver dependency; the `figures` extra adds Matplotlib and Pillow. The movie also needs `ffmpeg` with H.264 encoding, `ffprobe` and `gifsicle` in PATH.

```sh
.venv/bin/python -m pip install -e '.[figures]'
```

The static plots can be rebuilt from the included scores without any fluid simulation:

```sh
.venv/bin/python scripts/prepare_figures.py
.venv/bin/python scripts/make_figures.py
```

This writes four SVG/PNG pairs and their provenance JSON files into `figures/`. The renderer checks plotted ratios, source-record matches and label placement. It prints output sizes and hashes as JSON.

The Galewsky movie uses solver fields, not an illustrative animation. Generate its input archive with:

```sh
.venv/bin/python scripts/hero_run.py --M 512 --days 6 --every-hours 0.5 --out build/hero.npz
```

This is a full simulation, separate from the quick checks. The snapshot interval is rounded to an integer number of steps, so use the saved `t_s` array for playback. The archive includes the initial and final states, sphere positions, relative vorticity, surface height and cone locations. Keep it outside the published file set.

Once `build/hero.npz` exists, render both movie formats with:

```sh
.venv/bin/python scripts/make_hero.py
```

Use `--archive PATH` for an archive stored elsewhere, or `--work DIR` to keep intermediate frames at a chosen location. The renderer replaces `DIR/frames`; choose a directory reserved for these frames. Without `--work`, it uses a temporary directory. It checks the archive against the numerical metadata in `figure-pack/hero-input.json`, records whether its digest matches, and writes `figures/galewsky.gif`, `figures/galewsky.mp4` and `figures/galewsky.provenance.json`.

By default, the renderer produces a 16-second loop at 10 frames per second. It holds day 0 for 0.5 seconds, advances linearly to day 4 over 2.5 seconds, spends 12 seconds advancing linearly through days 4 to 6, then holds day 6 for one second. These times include both endpoint holds, with no extra endpoint frame. Each 0.1-second display slot uses the saved snapshot nearest to the target time at its midpoint, with exact ties going to the earlier snapshot; the two holds use the exact initial and final snapshots. No intermediate fluid states are synthesised. The linear day axis shows the selected snapshot's actual time and is labelled `Simulated time (days; variable playback speed)`.

The [provenance](figures/galewsky.provenance.json) records the schedule, exact selection and physical times, display durations, encoded frame counts, decoded durations, interpolation, colour scale, input and output hashes, and encoder versions. Use that record for exact counts; the GIF merges adjacent repeats while preserving their duration, and the MP4 retains every display slot.

The supplied figures use Arial, with text outlined in SVG. On systems without Arial, the renderer falls back to DejaVu Sans and output bytes differ. Encoder and library versions also matter for exact reproduction; each provenance file records them. Fonts and encoders are installed separately.

The [figure data guide](figure-pack/README.md) identifies the plotted columns and static rendering checks. The method guide describes the movie's [coordinates and interpolation](docs/METHOD.md#coordinates-and-interpolation) and [input arrays, checks and media formats](docs/METHOD.md#movie-inputs-and-checks). Large reference fields and movie inputs are regenerated locally rather than bundled.
