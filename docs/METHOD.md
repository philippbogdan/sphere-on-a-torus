# A periodic grid for the whole sphere

The torus replaces polar crowding with four local cone points and two copies of each physical point. Both index directions wrap around, so the same bounded neighbourhood works everywhere.

## Map and local operators

On an even $`M \times M`$ lattice, the following positions represent the same sphere point:

```math
(i,j) \sim (-1-i,-1-j) \pmod M.
```

There are $`M^2`$ computed points and $`M^2/2`$ unique points. A Weierstrass map places the branch points at a tetrahedron's vertices. Moser flow redistributes area so the cells beside them do not shrink relative to the rest of the grid as resolution increases. The map is only approximately equal-area: at M = 128 its largest-to-smallest cell-area ratio is 2.73, compared with 2,773 for the conformal map. See [geometry.json](../data/geometry.json).

```text
periodic torus grid  --->  sphere
 two copies of flow       one physical flow
 four branch points       four cone points
```

The main scheme in [`configs.MAIN`](../polarswe/configs.py) uses eighth-order centred differences and a ten-stage, fourth-order low-storage Runge-Kutta integrator. A stage reaches four lattice points; a two-point dissipation pass makes the full-step reach 42. This is a dependency distance, not a count of all samples contributing to a step.

Within eight cells of each cone, [`conehp.py`](../polarswe/conehp.py) fits a gradient/divergence pair along the interior stencil's lattice lines. The pair is adjoint under corrected area weights, conserving mass to round-off. Low-degree consistency constraints retain small residuals, so the fit is not exactly cubic. Curl is evaluated as minus the divergence of the rotated tangent velocity. Dissipation is off inside the patches; tracer transport uses a skew form to control variance growth.

The complete scheme has observed global order about three on smooth TC2, despite the higher-order interior differences. Cone treatment and dissipation remain part of the accuracy cost. The [companion error study](https://github.com/philippbogdan/sphere-grid-errors) examines the cone regions in more detail.

For a locally oriented branched cover from torus to sphere, Riemann-Hurwitz gives:

```math
0 = 2d - B,
```

where $`d`$ is the cover degree and $`B`$ is total branching. Degree one would require a homeomorphism between torus and sphere. Degree two, attained here, needs four simple branch points. This establishes the minimum within this class of maps; it is not a bound on grids with seams or moving boundaries.

## Reading the movie

The README movie shows relative vorticity from the M = 512 perturbed Galewsky jet over six simulated days. The initial smooth ring shows clear undulations by day 4 and rolled-up vortices and filaments by day 6. Each physical structure appears twice on the unwrapped torus, related by a half turn about its north-pole cone point. Gold dots identify the map's cone points, not flow features.

Both views use the same saved snapshot and a fixed, symmetric linear colour scale from -1.6e-4 to +1.6e-4 1/s. Negative vorticity is blue, zero is near-white and positive vorticity is rust. The shared bar is labelled `Relative vorticity (1/s)`, with ticks `-1.5`, `-0.75`, `0`, `0.75` and `1.5` under a $`\times10^{-4}`$ multiplier. The torus chart preserves the sphere's orientation as seen from outside; its [coordinates and interpolation](#coordinates-and-interpolation) are described below. The [figure provenance](../figures/galewsky.provenance.json) records the selected indices and physical times.

Playback gives the quiet early days less screen time: day 0 is held for 0.5 seconds, day 4 is reached over the next 2.5 seconds, then days 4 to 6 take 12 seconds before a one-second final hold. The loop totals 16 seconds at 10 frames per second. Only saved snapshots are selected, in chronological order; there is no interpolation between fluid states.

```text
playback:   0 s    0.5 s          3 s                         15 s   16 s
            | hold | fast advance |      slower advance       | hold |
simulation: day 0  day 0          day 4                       day 6  day 6
```

The axis is labelled `Simulated time (days; variable playback speed)`. Its ticks remain evenly spaced in physical days, and its dot follows the displayed snapshot's actual time, including the holds. Read elapsed simulated time from that axis; apparent motion per second of playback is not a measure of physical flow speed. At the GIF loop boundary, the field and time dot reset to day 0.

Fine grid-scale stripes inside the late vorticity bands are present in the saved field, not introduced by rendering. They remain visible without smoothing. The movie illustrates the computed evolution; it does not establish that the smallest structures are resolved or converged. The quantitative accuracy comparisons below use their own resolution pairs and reference fields.

### Coordinates and interpolation

The left view is a fixed orthographic projection centred on longitude zero and latitude 35 degrees north, with north up. It uses the saved `X` positions, retaining `i < M/2` to represent each paired sphere point once. Vorticity is interpolated linearly on front-facing triangles of these samples' convex hull. Pixels in the narrow rim outside those triangles take the nearest front-facing sample; triangles never connect across the horizon. Colour has no shaded relief or frame-dependent lighting.

The right view is the flat torus unwrapped as its rhombic fundamental domain, with dimensionless coordinates

```math
u = ((j+1/2)/M + 15/32) \bmod 1,
\qquad
v = ((i+1/2)/M + 15/32) \bmod 1,
```

drawn at $`x=u+v/2`$ and $`y=\sqrt{3}v/2`$. Taking `u` from `j` and `v` from `i` preserves outward orientation; paired locations are `(15/16-u, 15/16-v)` modulo one. The field and cone markers use the same transformation. Vorticity is interpolated linearly on the lattice triangles, wrapping both periodic boundaries. Matching edge ticks identify opposite sides: one on each `v = 0, 1` edge and two on each `u = 0, 1` edge.

Gold discs with dark rims mark `meta.branch_points_ij`. All four remain inside the torus image at combinations of $`u,v\in\{15/32,31/32\}`$; the north-pole cone is at `(15/32, 15/32)`, keeping its closed jet ring away from the cut. On the sphere, each marker lies at the normalised mean of the four neighbouring `X` samples around its half-index position. Only front-facing sphere markers are drawn. The `Cone point` legend identifies geometry, not vortices or instability sites.

### Movie inputs and checks

The input archive contains the full M = 512 perturbed Galewsky case. [`hero-input.json`](../figure-pack/hero-input.json) records its digest, dimensions and numerical metadata. With `F` saved snapshots, the arrays are:

| Array | Shape | Use |
|---|---|---|
| `X` | `(3, 512, 512)` | Cartesian unit-sphere positions for the left view and sphere cone markers |
| `vorticity` | `(F, 512, 512)` | Relative vorticity in 1/s, shown in both views |
| `t_s` | `(F,)` | Actual physical times in seconds, used for snapshot selection and the time-axis dot |
| `branch_distance_cells` | `(512, 512)` | Distances used to check the cone neighbours |
| `height` | `(F, 512, 512)` | Surface height in metres, checked for shape and finite values but not plotted |
| `meta` | JSON scalar | Case, grid, scheme, timing, mass drift, spacing and cone locations |

[`make_hero.py`](../scripts/make_hero.py) requires `case`, `M`, `scheme`, `dt_s`, `nsteps`, `mass_rel`, `mean_spacing_m` and `branch_points_ij` to match the included metadata. It checks array shapes, the recorded frame count and exact solver-step times within floating-point tolerance. Times must start at zero, increase strictly and end at both `nsteps * dt_s` and six days. The supplied snapshots are nine solver steps apart, about 29.6 minutes, with a final interval of seven steps. `X`, vorticity, height and branch distances must be finite; every vorticity magnitude must be strictly below the fixed scale limit. The archive digest is recorded with a match flag, rather than enforced.

The renderer requires the cells with branch distance below one to be exactly the four neighbours of each cone, and checks the north cone's chart position. It records paired-sample symmetry differences, interpolation statistics and sampled ring and half-turn diagnostics without imposing rejection thresholds on those diagnostics. The ring diagnostic describes the displayed band, not flow accuracy. Selected snapshots must be chronological and include the exact endpoints. Both views use each selected snapshot together, without temporal interpolation or blending.

The outputs are 1,000 by 560 px: a looping GIF with a fixed palette and no dithering, and an H.264 MP4 with `yuv420p`, fast-start metadata and no audio. The GIF merges adjacent repeated selections by summing their delays; the MP4 retains every display slot. The renderer compares GIF delays from two decoders, requires the GIF image count and delays to match the selected sequence, and checks the MP4 frame count against the display slots. It also applies the [shared text audit](../figure-pack/README.md). Decoded durations, format details, palette size, field extrema, mass drift and input/output hashes are recorded in provenance. File sizes are recorded, not enforced; the supplied GIF and MP4 are each below 10,000,000 bytes. The [reproduction guide](../REPRODUCE.md#movie-data-and-figure-inputs) gives the playback-selection rule and rendering commands; the [movie record](../figures/galewsky.provenance.json) holds the full slot schedule and exact counts.

## Accuracy comparisons

[`trsk.py`](../polarswe/trsk.py) implements energy-conserving TRSK without dissipation, using RK4 at Courant number 0.5. It generates centroidal icosahedral Voronoi meshes by refinement and Lloyd iteration. The comparisons use this implementation of the published scheme, not the MPAS executable or downloaded meshes.

| Resolution | Torus width M | Torus unique points | TRSK cells |
|---|---:|---:|---:|
| Coarse | 144 | 10,368 | 10,242 |
| Middle | 288 | 41,472 | 40,962 |
| Fine | 576 | 165,888 | 163,842 |

The torus has about 1.2% more unique points and computes each twice. These counts do not match work, storage or degrees of freedom. For height $`h`$, reference height $`h_{ref}`$ and area weights $`w`$, the normalised norms are:

```math
l_2 = \sqrt{\frac{\sum w(h-h_{ref})^2}{\sum w h_{ref}^2}},
\qquad
l_\infty = \frac{\max |h-h_{ref}|}{\max |h_{ref}|}.
```

The [paired table](../figure-pack/accuracy.csv) divides TRSK $`l_2`$ by torus $`l_2`$. Every one of its 18 ratios exceeds one.

| Test | Comparison time | Reference |
|---|---|---|
| TC2, steady rotating flow | Day 5 | Analytic |
| TC5, flow over a mountain | Day 15 | NCAR STSWM T213, step 22.5 s |
| TC6, Rossby-Haurwitz wave | Days 5 and 14 | NCAR STSWM T213, step 22.5 s |
| Steady Galewsky jet | Day 6 | Analytic |
| Perturbed Galewsky jet | Day 6 | NCAR STSWM T341, step 30 s |

The numerical reference fields are from NCAR STSWM. At the finest TC5 pair, torus error approaches reference uncertainty, about 0.24 to 0.37 m RMS. Reference resolution and step-size comparisons are retained in [`stswm_errors.jsonl`](../data/stswm_errors.jsonl).

TRSK has lower $`l_\infty`$ in two comparisons: TC5 at 40,962 cells, by about 13%, and TC6 day 14 at 10,242 cells, by about 16%. The day-5 TC6 pairs contain only $`l_2`$ in the paired table; blank maximum-error entries are not zero.

NCAR's spectral model also has lower TC6 day-14 $`l_2`$ at matched point counts. The [torus-to-spectral error ratios](../figure-pack/spectral.csv) are 1.863, 1.943 and 1.918 for M = 128, 256 and 512 against T42, T85 and T170 respectively. The TRSK comparison therefore does not establish a general ranking of methods.

The tracer scheme is not monotone and can leave small ripples. The tests do not cover full atmosphere physics, wetting and drying, shorelines or operational forecast skill.

## Stability and precision

TC5 stability limits on the main torus scheme are 2.449, 2.316 and 2.252 equatorial wave-crossing times for M = 64, 128 and 256. Each bisection spans 50 simulated days and stops at 2% relative bracket width. These limits use the computed-point spacing convention; the corresponding unique-point values are 1.73, 1.64 and 1.59. Definitions and the latitude-longitude comparison are in [PERFORMANCE.md](PERFORMANCE.md#stable-step).

For TC2, perturbation-form FP32 differs from FP64 by $`2.6\times10^{-8}`$ to $`2.9\times10^{-8}`$ in relative $`l_2`$ through M = 2048 across the supplied CPU and GPU measurements. Geometry and the reference-state tendency are prepared in FP64 before rounding. This measures precision error on steady flow, not FP32 accuracy or stability for other flows. See [`fp32.py`](../polarswe/fp32.py) and the [precision records](../data/README.md).
