# Data behind the figures

The tables separate measured accuracy and stability from modelled throughput. [`scripts/prepare_figures.py`](../scripts/prepare_figures.py) derives them from the included scores and the cost model after checking the data digests.

| File | Columns and use |
|---|---|
| `throughput.csv` | `layout`, `relative`, `steps_per_s`: modelled throughput against the unfiltered polar-limited baseline |
| `stable_step_vs_spacing.csv` | `grid` and `width` identify the measurements; `dx_eq_computed_m / 1000` and `r_max_computed` give spacing in km and the normalised stable step; `dt_max_s` links to the source record |
| `stable_step.csv` | `m`, `r`: the torus subset of the stable-step comparison |
| `accuracy.csv` | `case`, `m`, `torus_unique`, `trsk_cells`, raw `torus_l2` and `trsk_l2`, their `ratio`, and available maximum errors |
| `spectral.csv` | `m`, `torus_l2`, `spectral_l2`, `torus_over_spectral`: the separate TC6 day-14 comparison |
| `filter_trade.csv` | `arcsec`, `filter_over_torus`, step rates and `filter_limit`: the same-kernel polar-filter model |
| `hero-input.json` | Archive digest, dimensions, physical times and metadata for the supplied Galewsky field archive |
| `model.json` | Extrapolated step ratio, kernel operations and reach, patch cost and model calibration |

`relative` is simulated time per wall time, not just steps per second. `ratio` in the TRSK table is TRSK error divided by torus error; `torus_over_spectral` uses the opposite numerator. The filtered comparison has its own denominator and must not be combined with the polar-limited headline.

[`scripts/make_figures.py`](../scripts/make_figures.py) renders four SVG/PNG pairs from these tables; [`scripts/make_hero.py`](../scripts/make_hero.py) renders the movie from `build/hero.npz`. Each figure occupies a full README row. [REPRODUCE.md](../REPRODUCE.md#movie-data-and-figure-inputs) gives the dependencies and commands.

The static renderer checks throughput against the step-rate ratio times the extrapolated step ratio, stable-step values against `data/timestep.jsonl`, accuracy ratios against `trsk_l2 / torus_l2` and the paired point counts, and filtered throughput against `filter_steps_per_s / torus_steps_per_s` under the equal-step assumption. It also checks the expected cases, spacings and rounded labels. Maximum errors are not plotted; blank TC6 day-5 entries are not zero. The spectral comparison remains separate, with its own resolutions and opposite ratio convention.

Throughput and accuracy use horizontal dot plots on logarithmic axes. Stable step uses logarithmic axes with spacing decreasing to the right; the filtered comparison uses a base-two logarithmic spacing axis and a linear ratio axis. Connecting segments are guides, not fits or evidence of an exact filter crossover. The tables contain no interpolated missing scores or statistical confidence intervals. Exact axis limits, labels and colours are recorded beside each figure in its `.provenance.json` file.

The static figures use a 1,000 px design width, PNGs at twice that resolution and outlined SVG text. The shared text audit checks required and allowed labels, a 16 px minimum size at design width, canvas bounds, text overlaps and sampled intersections with lines. The static renderer also checks that data points lie inside the axes. These geometric checks support visual inspection; they do not replace it.

The movie shows the same saved vorticity field on the sphere and torus, with a shared fixed colour scale and variable playback speed. Its [coordinates and interpolation](../docs/METHOD.md#coordinates-and-interpolation), [input arrays and renderer checks](../docs/METHOD.md#movie-inputs-and-checks), and [playback](../REPRODUCE.md#movie-data-and-figure-inputs) are documented separately. The large field archive is not bundled.

Each provenance file records source hashes, plotted values or snapshot selection, transformations, rendering versions and output hashes. The [movie record](../figures/galewsky.provenance.json) includes the full display-slot schedule and encoding details. The supplied text uses Arial; a DejaVu Sans fallback changes the bytes. Fonts are not bundled.
