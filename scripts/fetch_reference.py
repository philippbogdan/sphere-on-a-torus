"""Download and hash-check NCAR STSWM for scientific research and education.

The fetched archive retains UCAR's terms and is outside the CC0 dedication.
Read THIRD_PARTY.md before use."""
from pathlib import Path
import hashlib
import urllib.request

URL = 'https://www.netlib.org/chammp/stswm.tgz'
SHA256 = '385b001d0b116f86026e8df9f8c27f7749ed0533231c8b42541b1b8e2af2d9f1'
DEST = Path(__file__).resolve().parents[1] / 'refs/stswm/stswm.tgz'

def main():
    with urllib.request.urlopen(URL) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('Archive hash mismatch. Verify the distribution and its terms before use.')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(data)
    print('Saved the hash-checked NCAR archive. See THIRD_PARTY.md for research and education terms.')

if __name__ == '__main__':
    main()
