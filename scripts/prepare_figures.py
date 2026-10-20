"""Write the figure data from saved measurements. No plots are read or copied."""
from pathlib import Path
import csv
import json
import sys
from check_claims import calculate, verify

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'figure-pack'

def write_csv(name, rows):
    with (PACK / name).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    PACK.mkdir(exist_ok=True)
    s = calculate()
    verify(s)
    write_csv('throughput.csv',s['speed_model'])
    write_csv('filter_trade.csv',s['filter_trade_model'])
    write_csv('accuracy.csv',s['accuracy'])
    write_csv('spectral.csv',s['tc6_spectral'])
    write_csv('stable_step.csv',s['stable_step'])
    (PACK/'model.json').write_text(json.dumps({k:s[k] for k in ('step_ratio','cost','patch_model','calibration')},indent=2)+'\n')
    print('Wrote',PACK)

if __name__ == '__main__':
    main()
