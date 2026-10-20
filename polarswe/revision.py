"""Identify the code used for a run without borrowing a parent checkout's revision."""
from pathlib import Path
import hashlib
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def current():
    digest = hashlib.sha256()
    paths = sorted(list((ROOT/'polarswe').glob('*.py')) + list((ROOT/'polarswe').glob('*.json'))
                   + list((ROOT/'scripts').glob('*.py')) + list((ROOT/'refs/stswm').glob('*.sh'))
                   + list((ROOT/'refs/stswm').glob('*.py')) + list((ROOT/'refs/stswm').glob('*.f')))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0' + path.read_bytes())
    fingerprint = digest.hexdigest()[:16]
    label = os.environ.get('POLARSWE_COMMIT')
    if not label and (ROOT/'.git').exists():
        label = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    return (label + ':' if label else 'source:') + fingerprint
