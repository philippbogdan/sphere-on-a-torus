"""Advance a short analytic steady flow and print its error and mass drift."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from polarswe import configs, experiments

if __name__ == '__main__':
    result = experiments.run(('torus',64),'tc2_a0.7',days=0.05,scheme=configs.MAIN)
    print(json.dumps({k:result[k] for k in ('stable','nunique','nsteps','l2','mass_rel')},indent=2))
