# A sphere with local dependencies

The index grid is an M by M periodic lattice. The map sends each pair (i, j) and (-1-i, -1-j), modulo M, to the same sphere point. There are M squared computed points and half as many unique points. A Weierstrass map gives the tetrahedral double cover. Moser's flow spreads its area to stop the cells next to the four branch points from shrinking with resolution.

The main scheme uses eighth-order centred differences and a ten-stage, fourth-order low-storage integrator. Its measured global order on smooth TC2 flows is about three. Local fourth-order dissipation and the cone treatment set that order. There is no whole-latitude polar filter. The local dissipation is off inside the cone patches.

Within eight cells of each branch point, `conehp.py` fits an adjoint gradient and divergence pair on the same lattice lines as the interior stencil. The fit gives more weight to low-degree consistency constraints. These constraints have a small residual, so the operators should not be described as exactly cubic. Their adjoint relation and mass conservation hold to round-off. Curl uses the same divergence, as `-div(n x v)`. Tracer transport uses the skew form to avoid variance growth at a cone.

Each stage reaches four lattice points. Ten stages plus the two-point local dissipation give a full-step reach of 42. The source and tests retain alternatives so the effects of the cone treatment, dissipation and integrator can be checked separately.

## Why two copies

For a locally oriented branched cover from the torus to the sphere, Riemann-Hurwitz gives `0 = 2d - B`, where d is the degree and B is total branching. Degree one would be a homeomorphism and is impossible. Degree two needs four simple branch points. This grid attains degree two.

That statement concerns branched covers with consistent local neighbourhoods. It is not a lower bound on every possible processor mapping, grid with seams, moving boundary or algorithm for a sphere. A folded latitude mapping may instead pay in bandwidth and boundary handling. This release does not prove the cost of every such implementation.

## Accuracy comparisons

`trsk.py` implements the published energy-conserving TRSK scheme, with no dissipation, RK4 and Courant number 0.5. It generates centroidal icosahedral Voronoi meshes by refinement and Lloyd iteration. These are MPAS-type meshes. They are not downloaded MPAS meshes, and this comparison does not run the MPAS software.

The paired meshes have 10,242, 40,962 and 163,842 cells. The torus has 10,368, 41,472 and 165,888 unique points. Thus its unique-point count is about 1.2% higher. Its computed-point count is twice that, a cost included in the throughput model. Equal unique cells are not equal storage, work or degrees of freedom between these schemes.

All paired height errors use the same normalised l2 score and reference. TC2 uses its analytic steady solution. TC5 and TC6 use locally generated NCAR STSWM T213 fields with a 22.5 s step. The perturbed Galewsky jet uses T341 with a 30 s step. The steady jet uses its analytic initial state. The records also retain l1, linf, mass drift and alternative-reference scores where available.

The torus has lower l2 in all 18 paired comparisons, including TC6 at days 5 and 14. It does not win every norm: TRSK has lower linf for TC5 at 40,962 cells and TC6 day 14 at 10,242 cells. At the finest TC5 resolutions, the torus error is near the reference uncertainty, about 0.24 to 0.37 m RMS. The TC1 tracer is not monotone and leaves small ripples. No claim about full atmosphere physics, wetting and drying, shorelines or operational forecasts follows from these shallow-water tests.

NCAR's spectral model has lower TC6 day-14 l2 by about 1.9x at matched point counts. TC2's degree-two steady field is represented exactly in its spatial basis. It would be wrong to say that spectral is ahead only on TC6. TC6 at day 14 is the material spectral advantage on the nontrivial evolved cases at M=128 to 512. The coarse M=64 results have other spectral advantages too.

## Precision

For TC2 at angle 0.7, perturbation-form FP32 differs from FP64 by 2.6 to 2.9e-8 in relative l2 from M=64 to 2048. Geometry and the initial tendency are computed in FP64 before rounding. This is a CPU/GPU arithmetic study, not a Cerebras run. At roughly M=500 and above, that round-off exceeds TC2's FP64 truncation error. It does not establish stability or accuracy for every case in FP32.
