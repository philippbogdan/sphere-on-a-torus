US$100,000 to US$10 million

My estimate of the work's expected incremental value to weather, climate and scientific computing over the next ten years, with a central judgement of about US$1 million. Amounts are in roughly 2026 dollars, without discounting. The eventual realised benefit could be zero.

The opportunity is cheaper global simulation through a periodic grid with bounded local communication, including near the poles. The supplied implementation and benchmark data make this a credible research contribution: saved results show lower height l2 error in all 18 paired comparisons against the project's own TRSK implementation. The method computes each physical point twice, and these comparisons establish neither operational forecast quality nor savings at equal accuracy. See the [method and numerical limits](docs/METHOD.md).

I assign a 5% to 15% joint chance that further development establishes a useful advantage over available alternatives and leads to material adoption within the decade. The [performance model](docs/PERFORMANCE.md) reports an 856-fold speedup against an unfiltered, polar-limited baseline, with no Cerebras hardware run and an unresolved coefficient-memory constraint. Its separate filtered comparison actually favours latitude-longitude by 1.41 times at the finest spacing. Established alternatives include [NOAA's cubed-sphere FV3](https://www.gfdl.noaa.gov/fv3/fv3-grids/); [operational AI forecasting](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational) also competes for investment. These factors make modest eventual savings a more defensible basis than the headline multiplier.

The spending scale is substantial: NOAA's 2020 supercomputing agreement had an [estimated US$505 million value over up to ten years](https://www.gdit.com/about-gdit/press-releases/noaa-awards-general-dynamics-high-performance-computing-contract/), roughly US$50 million annually. That covers a whole service. I assume this work could eventually affect a subset of workloads across one or several organisations, with five useful operating years after development:

| Assumption | Cautious | Middle | Optimistic |
| --- | ---: | ---: | ---: |
| Annual spending on affected workloads | US$10m | US$30m | US$100m |
| Incremental saving over available alternatives | 10% | 10% | 15% |
| Further development and integration costs | US$3m | US$5m | US$15m |
| Benefit if successful: spending × saving × 5 years, less costs | US$2m | US$10m | US$60m |
| Probability of achieving that outcome | 5% | 10% | 15% |
| Expected value: probability × benefit | US$0.1m | US$1m | US$9m |

These are alternative judgemental scenarios, rounded to the opening range. Uptake, savings, costs and probabilities are assumptions, not measured commercial results. The estimate credits this mapping and solver's incremental contribution beyond the existing [Domain Translation method](https://arxiv.org/abs/2511.11542) and hardware. Better forecasts could have additional value, but the supplied shallow-water experiments do not establish that benefit.
