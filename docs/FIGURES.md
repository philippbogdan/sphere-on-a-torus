# Read and rebuild the figures

All four figures use a full 830-pixel GitHub column. Titles are 25 CSS pixels and the smallest text is 17. The PNGs are exported at twice that pixel width. SVG text is converted to paths to preserve its appearance without an installed font. The CSVs and provenance JSON keep every plotted number available as text.

Throughput uses a log10 dot plot, with an open diamond for the additional patch-sharing estimate. Stable step uses measured bisection limits on a zero-based y axis. Accuracy shows all 18 paired l2 ratios, with a logarithmic colour scale and a number in every cell. The filter comparison has a zero-based y axis and a dashed parity line. The lines join tested conditions, not fitted curves. No confidence intervals exist for these deterministic values.

`scripts/prepare_figures.py` rebuilds the clean data pack from original score records and the model. `scripts/make_figures.py` checks wording, text size, bounds, overlap and matrix-label contrast before exporting. The input hashes, exact values, rounding and output hashes are recorded beside each figure.

The colours also have shapes, direct labels or numeric values. Text contrast is at least 10.86:1 on white; matrix labels are at least 4.74:1 against their cells. Grayscale previews were inspected. This does not certify accessibility or replace the underlying data and image descriptions.

## Rebuild

These commands were run in the clean copy after the core installation:

```sh
.venv/bin/python -m pip install -q -e '.[figures]'
.venv/bin/python scripts/prepare_figures.py
.venv/bin/python scripts/make_figures.py
```

The plotting script needs no external account, local skill installation or network connection. Its source data is in `figure-pack/`. The public figures use Matplotlib 3.11.1 and Pillow 12.2.0.

## Visualisation guidance

The figure design and checks used the scientific-visualization procedure described in Kassis, T., Agarwal, V., He, Y., Patel, D. and Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). This is an acknowledgement of visualisation guidance, not evidence for the numerical results.
