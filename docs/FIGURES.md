# Rebuild the figures

After [core installation](REPRODUCE.md), these release-tested commands rebuild all four figures:

```sh
.venv/bin/python -m pip install -q -e '.[figures]'
.venv/bin/python scripts/prepare_figures.py
.venv/bin/python scripts/make_figures.py
```

The renderer reads [figure-pack/](../figure-pack/). Beside each figure, provenance JSON records inputs, exact values, rounding and output hashes. Joined points are not fitted curves; these deterministic results have no confidence intervals.

The figure design used Kassis et al. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065), for visualisation guidance.
