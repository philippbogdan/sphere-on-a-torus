# Sphere on a torus

A whole-sphere shallow-water solver with a fixed 42-point reach and no whole-latitude polar filter. It avoids the polar timestep collapse by computing each sphere point twice.

## Modelled throughput on 64 CS-3 systems

At 15 arcseconds, the model gives **856x more simulated time per wall time** than an unfiltered, polar-limited lat-lon baseline. Sharing cone-patch work raises the estimate to 1,263x.

<a href="figures/throughput.svg"><img src="figures/throughput.png" width="100%" alt="Modelled throughput relative to the polar-limited baseline: 1, 856 with counted patch work, 1,263 with estimated sharing."></a>

No Cerebras hardware was used. The baseline extends the [Domain Translation paper's](https://arxiv.org/abs/2511.11542) cost model with an extrapolated polar timestep limit. The paper does not disclose its pole treatment. Both estimates need a coefficient layout that fits core memory; the shared schedule is unimplemented. [Model and memory limits](docs/PERFORMANCE.md).

## Local work, poles included

The tetrahedral double cover has four local cone patches. Measured TC5 stable steps at widths 64, 128 and 256 are 2.45, 2.32 and 2.25 equatorial wave-crossing times.

<a href="figures/locality.svg"><img src="figures/locality.png" width="100%" alt="Line chart titled 'The poles no longer collapse the stable step'. The horizontal axis is torus width M, with M squared computed points. The vertical axis is stable step divided by equatorial wave-crossing time. Values are 2.45, 2.32 and 2.25 at widths 64, 128 and 256 respectively."></a>

Two copies are the minimum for a locally oriented branched cover from torus to sphere. This is not a bound on all grids with seams or moving boundaries. Local dissipation remains outside the cone patches. [Method](docs/METHOD.md).

## Accuracy at nearly equal unique-cell counts

The torus has **lower normalised height l2 error in all 18 paired comparisons** with our TRSK implementation on generated MPAS-type meshes. It has about 1.2% more unique points and computes each twice.

<a href="figures/accuracy.svg"><img src="figures/accuracy.png" width="100%" alt="TRSK l2 divided by torus l2 ranges from 1.16 to about 3,200 across steady flow, mountain, wave and jet tests."></a>

TRSK wins two linf comparisons. NCAR spectral has about 1.9x lower TC6 day-14 l2. These are shallow-water tests, not complete atmosphere models. On TC2, perturbation-form FP32 differs from FP64 by 2.6 to 2.9e-8 in relative l2 through M=2048. [Reference uncertainty and precision limits](docs/METHOD.md#accuracy-comparisons).

## When a polar filter is worth it

An optimistic filtered lat-lon model is **1.41x faster at 15 arcseconds**, 1.33x at 240, but only 0.72x and 0.62x as fast at 480 and 960.

<a href="figures/filter-trade.svg"><img src="figures/filter-trade.png" width="100%" alt="Line chart titled 'A polar filter can win while bandwidth dominates'. The horizontal axis is lat-lon grid spacing in arcseconds. The vertical axis is filtered lat-lon throughput divided by torus throughput. Values are 1.41, 1.41, 1.40, 1.39, 1.33, 0.72 and 0.62 at spacings of 15, 30, 60, 120, 240, 480 and 960 arcseconds respectively. A dashed line marks equal throughput at 1.0."></a>

This separate comparison fixes the kernel and unique-cell count, omits patch-core overhead and charges at least three node latencies per filter step. The filter's accuracy cost was not tested.

## How to use it

Start with the [installation and checks](docs/REPRODUCE.md), then the [field and stepping interfaces](docs/PORT.md). The release supplies a CPU solver, an optional GPU arithmetic path and a dataflow model. A Cerebras kernel still needs to be built.

Our code, generated data, figures and text are [CC0 1.0](LICENSE). Separately fetched NCAR reference software has [research and education terms](docs/THIRD_PARTY.md).
