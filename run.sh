#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -m venv .venv
.venv/bin/python -m pip install -q -e '.[check]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_claims.py
