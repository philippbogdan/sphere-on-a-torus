# What the speed model says

At 15 arcseconds on an 8 by 8 torus of CS-3 nodes, the model gives about 856x the simulated-time throughput of an unfiltered, polar-limited latitude-longitude baseline. Sharing cone-patch work raises the estimate to about 1,263x. Neither figure is a hardware measurement or a comparison against a known production pole treatment.

The baseline uses the [Domain Translation paper's](https://arxiv.org/html/2511.11542v2) shallow-water cost model: `1026 + 183.2 n + 137.6 n squared` cycles per step and `64 n + 80` received words per core. Its grid is 86,400 by 43,200. The paper uses a Mercator projection and does not specify its pole treatment. Here a measured polar timestep scaling from a second-order centred/RK4 scheme class is extrapolated to that grid. It is not a measured stability limit of the paper's unpublished Lax-Wendroff implementation.

The torus is compared at the same nominal equatorial spacing, 463.3 m, using sphere area divided by spacing squared for its unique-point count. That gives about 2.38 billion unique points and 4.75 billion computed points. This comparison therefore differs from the same-unique-cell comparison used for the polar-filter trade below.

## Arithmetic and traffic

The model uses 744 by 1,116 cores per node, 0.75 GHz, two FP32 operations per cycle, 300 Gbit/s per wafer side and direction, 10 microseconds node latency, and 12 ns average core-link latency. It applies the paper's 53% compute utilisation to the new kernel. Directional bandwidth is separate. The baseline's directional traffic split, which is not published, is assumed equal.

The torus kernel counts 2,163 operations per point per step, including rebuilding metric terms from transmitted positions. It sends 420 words per edge point per step: 252 of state and 168 of position and topography. Each translation moves data by the 42-point dependency reach. The model scales the baseline's constant and linear cycle terms by the traffic ratio. That scaling and transfer of compute efficiency to this different kernel are assumptions.

The 920 cone-patch points carry 151 extra stored words each and take 2.43x the interior point work. Counting the worst core's work gives 1,413 steps/s. Sharing work over 42 nearby cores gives an estimated 2,085 steps/s. The baseline gives 27,258 steps/s. The extrapolated stable torus step is about 16,505x longer, producing the throughput ratios above.

The unshared arithmetic count is not a ready-to-run memory layout. A worst-aligned core with 96 patch points would need about 70.5 kB if all its coefficients were resident, exceeding the assumed 48 kB. Coefficients must be distributed or streamed. The shared model budgets that storage over nearby cores, but its schedule and communication have not been implemented or timed. Porting must settle this before treating either speed estimate as achievable.

The timestep extrapolation uses the measured TC5 stability limit at M=256. Limits were bisected to 2% over 50 simulated days at M=64, 128 and 256. The 15-arcsecond simulation itself was not run. As a check on the inherited model, compute-bound heat-equation rates match values read from the paper's figure within 1%. I/O-bound estimates are 9% to 35% optimistic. This checks the model calculation, not the new solver's hardware performance.

## A polar filter can be faster

This separate comparison holds the per-point kernel and number of unique points fixed. It omits cone-patch core overhead on both layouts. The filtered lat-lon model assumes a stable step as large as the torus and charges at least three node latencies per whole-latitude filter step. It does not count a specific FFT implementation or test the filter's accuracy. Treat it as an optimistic design comparison.

Filtered lat-lon throughput divided by torus throughput is 1.41 at 15 arcseconds, 1.33 at 240, 0.72 at 480 and 0.62 at 960. At fine resolution bandwidth dominates and computing one copy helps. At coarse resolution the assumed filter latency dominates. The large 856x headline concerns the unfiltered polar-limited baseline and does not apply to this filtered alternative.
