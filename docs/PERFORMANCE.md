# Speed-model assumptions

The headline comparison uses an 8 by 8 torus of CS-3 nodes at nominal equatorial spacing 463.3 m. Sphere area divided by spacing squared gives 2.38 billion unique torus points and 4.75 billion computed points. The lat-lon baseline has 86,400 by 43,200 points.

The baseline uses the [Domain Translation paper's](https://arxiv.org/html/2511.11542v2) cost model: `1026 + 183.2 n + 137.6 n squared` cycles per step and `64 n + 80` received words per core. Its polar timestep is extrapolated from second-order centred/RK4 measurements, not measured for the paper's Lax-Wendroff implementation.

## Machine and kernel

The model assumes 744 by 1,116 cores per node, 0.75 GHz, two FP32 operations per cycle, 300 Gbit/s per wafer side and direction, 10 microseconds node latency and 12 ns average core-link latency. It transfers the paper's 53% compute utilisation to this kernel and assumes an equal directional split of the baseline's unpublished traffic.

The torus counts 2,163 operations per point per step, including metric reconstruction. Each edge point sends 420 words: 252 state and 168 position/topography words. Translations cover 42 points. The baseline's constant and linear cycle terms scale with the traffic ratio, an assumption.

The 920 cone-patch points each need 151 extra coefficient words and 2.43x the interior work. Counting the worst core gives 1,413 steps/s; sharing over 42 nearby cores gives an estimated 2,085 steps/s. The baseline gives 27,258 steps/s. An extrapolated torus timestep 16,505x longer gives the README's throughput ratios.

A worst-aligned core holding all 96 patch points' coefficients would need about 70.5 kB against the assumed 48 kB. Both estimates require distribution or streaming. The shared model spreads storage over nearby cores, but neither its schedule nor communication has been implemented or timed. See the [porting requirements](PORT.md).

## Checks and scope

TC5 stability limits were bisected to 2% over 50 simulated days at M=64, 128 and 256. The extrapolation uses M=256; the 15-arcsecond simulation was not run. The inherited model matches compute-bound heat-equation rates read from the paper within 1%, but is 9% to 35% optimistic for I/O-bound rates. This checks the model, not hardware performance of the new solver.

The polar-filter comparison instead fixes the per-point kernel and unique-point count. It assumes the torus timestep and at least three node latencies for a whole-latitude filter, without costing a specific FFT. It omits cone-patch core overhead on both layouts. Bandwidth favours one copy at fine resolution; filter latency dominates at coarse resolution. Its ratios cannot be combined with the polar-limited headline.
