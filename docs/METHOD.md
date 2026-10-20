# Method and numerical limits

An M by M periodic lattice maps `(i, j)` and `(-1-i, -1-j)`, modulo M, to the same sphere point. A Weierstrass map gives the tetrahedral double cover. Moser's flow spreads its area so cells beside the four branch points do not shrink with resolution.

The main scheme uses eighth-order centred differences and a ten-stage, fourth-order low-storage integrator. Smooth TC2 runs show global order about three, set by local fourth-order dissipation and the cone treatment. Each stage reaches four points; a two-point dissipation pass brings the full-step reach to 42.

Within eight cells of each branch point, `conehp.py` fits an adjoint gradient/divergence pair on the interior stencil's lattice lines. Low-degree consistency constraints have small residuals: the operators are not exactly cubic. Adjointness and mass conservation hold to round-off. Curl is `-div(n x v)`. Tracer transport uses a skew form to avoid cone variance growth. Dissipation is off inside these patches.

For a locally oriented branched cover, Riemann-Hurwitz gives `0 = 2d - B`, with degree d and total branching B. Degree one would require a homeomorphism between torus and sphere. Degree two, attained here, needs four simple branch points. This argument says nothing about the cost of arbitrary folded or seamed processor mappings.

## Accuracy comparisons

`trsk.py` implements energy-conserving TRSK without dissipation, using RK4 at Courant number 0.5. It generates centroidal icosahedral Voronoi meshes by refinement and Lloyd iteration. No MPAS executable or downloaded mesh is used. See [literature and software terms](THIRD_PARTY.md).

| Resolution pair | TRSK cells | Torus unique points |
|---|---:|---:|
| Coarse | 10,242 | 10,368 |
| Middle | 40,962 | 41,472 |
| Fine | 163,842 | 165,888 |

Unique cells do not equate storage, work or degrees of freedom across schemes. Both schemes use the same normalised height l2 and reference:

- TC2 and steady jet: analytic solutions.
- TC5 and TC6: NCAR STSWM T213, step 22.5 s.
- Perturbed Galewsky jet: NCAR STSWM T341, step 30 s.

We acknowledge NCAR STSWM for the numerical reference fields. The records also retain l1, linf, mass drift and alternative-reference scores where available.

TRSK's two linf wins are TC5 at 40,962 cells and TC6 day 14 at 10,242 cells. At the finest TC5 resolutions, torus error approaches reference uncertainty, about 0.24 to 0.37 m RMS. The TC1 tracer is not monotone and leaves small ripples. These tests do not establish performance for full atmosphere physics, wetting and drying, shorelines or operational forecasts.

Spectral's TC6 day-14 advantage applies at matched point counts for M=128 to 512. Its basis also represents TC2's degree-two steady field exactly; coarse M=64 results have other spectral advantages. The paired TRSK result is not a universal ranking of schemes.

## Precision

The TC2 FP32 study uses angle 0.7 and M=64 to 2048. Geometry and initial tendency are prepared in FP64 before rounding. At roughly M=500, round-off overtakes TC2's FP64 truncation error. This CPU/GPU study does not establish FP32 accuracy or stability for other flows.
