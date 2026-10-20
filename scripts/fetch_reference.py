"""Fetch NCAR STSWM for local research use. It is not part of the CC0 release."""
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
        raise SystemExit('The upstream archive has changed. Check its source and terms before using it.')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(data)
    print('Fetched the checked NCAR archive. See docs/THIRD_PARTY.md for its research and education terms.')

if __name__ == '__main__':
    main()
