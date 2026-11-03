# What the throughput model predicts

At 15 arcseconds (463.3 m nominal equatorial spacing) on 64 CS-3 systems, the cost model gives the torus 856x the simulated time per wall time of latitude-longitude with no polar filter, with no hardware run. In a separate model at the same spacing, latitude-longitude with a polar filter is 1.41x as fast as the torus. The unfiltered comparison combines modelled step rates with extrapolated time steps.

| Layout | Modelled steps/s | Modelled simulated time per wall time, relative to baseline |
|---|---:|---:|
| Polar-limited latitude-longitude | 27,257.8 | 1 |
| Torus, patch work on the busiest core | 1,413.3 | 855.80 |
| Torus, shared patch work, unimplemented estimate | 2,085.3 | 1,262.69 |

Both torus estimates require coefficient distribution or streaming to fit core memory. The values are in [`throughput.csv`](../figure-pack/throughput.csv); [`check_claims.py`](../scripts/check_claims.py) reconstructs them from stability measurements and the dataflow model.

## The denominator matters

The baseline extends the Domain Translation cost model of [Oppelstrup et al., arXiv:2511.11542](https://arxiv.org/abs/2511.11542) with a polar time-step limit. The paper does not disclose its pole treatment. The limit is extrapolated from a second-order centred/RK4 solver, not measured for the paper's Lax-Wendroff implementation.

At this spacing the torus step is extrapolated to be 16,505x longer. Multiplying that ratio by the modelled step-rate ratio gives the cost-model throughput ratio against latitude-longitude with no polar filter:

```math
\frac{\text{torus simulated time / wall time}}{\text{baseline simulated time / wall time}}
=
\frac{\text{torus steps/s}}{\text{baseline steps/s}}
\frac{\Delta t_{torus}}{\Delta t_{baseline}}.
```

The 15-arcsecond torus case was not simulated. Its time step is extrapolated from M = 256. The layout has about 2.38 billion unique torus points and 4.75 billion computed points; the latitude-longitude baseline has 86,400 by 43,200 points. Nominal equatorial spacing sets these counts, rather than equal accuracy or equal work.

## Kernel and memory

The model arranges 64 CS-3 systems in an 8 by 8 torus. [`dataflow.py`](../polarswe/dataflow.py) contains the assumed core array, clock, directional bandwidth, latencies and cost coefficients. The solver inherits the paper's compute utilisation; baseline traffic is assumed to split equally between directions, and traffic ratios scale the baseline's constant and linear cycle costs.

[`model.json`](../figure-pack/model.json) records 2,163 operations per point per step and a full-step reach of 42. The payload packs depth and tangent velocity into three moving words, alongside four static position/topography words. Metric reconstruction and corner traffic are included.

The four cone patches contain 920 points in total. Each needs 151 extra coefficient words and about 2.43x the interior work. In the worst alignment, one core contains 96 patch points. Keeping coefficients and the assumed working arrays there needs about 70.5 kB against 48 kB of assumed core memory. The modelled rate with work counted on that core is therefore not yet an executable hardware result.

The shared estimate spreads patch work over 42 nearby cores. It raises the modelled rate to 2,085.3 steps/s, but the sharing schedule and its communication have not been implemented or timed. Both layouts need a concrete storage solution. [Porting requirements](PORT.md#porting-to-a-local-dataflow-machine).

Calibration against compute-bound heat-equation rates read from the Domain Translation paper agrees within 1%. The model is about 9% to 36% optimistic for I/O-bound rates. The six relative errors are retained in `calibration` in [`model.json`](../figure-pack/model.json). Those comparisons check the inherited model, not the hardware performance of this solver.

## Stable step

[`timestep.jsonl`](../data/timestep.jsonl) stores the stability brackets. The plotted quantity is the largest accepted step divided by an equatorial wave-crossing time:

```math
r = \frac{\Delta t_{max} c_{max}}{\Delta x_{eq}},
\qquad
\Delta x_{eq} = \frac{\pi a}{\sqrt{N_{computed}/2}}.
```

Here $`a`$ is sphere radius and $`c_{max}`$ is the initial maximum gravity-wave speed plus the maximum flow speed. This computed-point convention includes both copies of the sphere; it is not the minimum polar-cell width.

| Grid | Width | Equatorial spacing (km) | Stable step (wave-crossing times) | Simulated duration (days) |
|---|---:|---:|---:|---:|
| Torus | M = 64 | 442.3 | 2.449 | 50 |
| Torus | M = 128 | 221.1 | 2.316 | 50 |
| Torus | M = 256 | 110.6 | 2.252 | 50 |
| Unfiltered latitude-longitude | 45 latitudes | 314.5 | 0.164 | 5 |
| Unfiltered latitude-longitude | 90 latitudes | 157.3 | 0.082 | 3 |
| Unfiltered latitude-longitude | 180 latitudes | 78.6 | 0.041 | 1 |

All brackets were bisected to 2%. The latitude-longitude values use second-order centred differences and RK4. Their decreasing test durations matter: this is evidence of polar step-size scaling, not a matched-duration stability comparison. The [plot table](../figure-pack/stable_step_vs_spacing.csv) retains exact spacings, normalised limits and limits in seconds.

## A separate polar-filter model

A whole-latitude filter relaxes the polar restriction but adds non-local communication. Holding the kernel and unique-point count fixed, the model gives the following filtered latitude-longitude throughput divided by torus throughput:

| Spacing (arcseconds) | Modelled filter / torus throughput |
|---:|---:|
| 15 | 1.411 |
| 30 | 1.408 |
| 60 | 1.403 |
| 120 | 1.392 |
| 240 | 1.335 |
| 480 | 0.718 |
| 960 | 0.621 |

This comparison assumes the torus time step for both layouts, omits cone-patch overhead and charges at least three CS-3 system latencies per filter step. It does not cost a specific FFT or test the filter's effect on accuracy. One copy saves bandwidth at fine spacing; filter latency limits the coarser cases. The [ratios](../figure-pack/filter_trade.csv) have a different denominator from the cost-model comparison against latitude-longitude with no polar filter and cannot be combined with it.
