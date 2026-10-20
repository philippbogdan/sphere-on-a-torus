# Reuse and reference software

Our solver, generated meshes, scores, figures and text use [CC0 1.0](../LICENSE). Dependencies keep their own licences. No NCAR source, archive, binary or reference-field archive, MPAS source or downloaded mesh, paper PDF or original paper figure is bundled.

## NCAR STSWM 2.0

The [Netlib distribution](https://www.netlib.org/chammp/) identifies UCAR copyright 1992, all rights reserved. Source use is limited to scientific research and education. It is not public domain. Its terms ask that resulting publications acknowledge NCAR and substantial modifications be identified as derivatives of NCAR STSWM.

The reference fetch checks SHA-256 `385b001d0b116f86026e8df9f8c27f7749ed0533231c8b42541b1b8e2af2d9f1` and preserves upstream notices. Read them before use. Fetched code is outside this project's CC0 dedication.

Our build changes truncation/precision, replaces spectral netCDF output with a new grid writer and adds file-based jet/diagnostic initial conditions. The patches grant no rights over NCAR code. We acknowledge NCAR STSWM as the reference model for the numerical comparisons.

## Meshes and papers

`polarswe/trsk.py` generates the meshes and implements published mathematics: Thuburn et al. (2009), JCP 228, 8321-8335, and Ringler et al. (2010), JCP 229, 3065-3090. Comparison scores come from our runs; no published tables or digitised values are redistributed.

The [Domain Translation paper](https://arxiv.org/abs/2511.11542) supplies reported model parameters and retains its own terms.
