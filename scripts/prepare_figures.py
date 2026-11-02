"""Export the verified comparison tables and model costs for plotting."""
from pathlib import Path
import csv
import json
import sys
from check_claims import calculate, verify, configs

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
    ts = [json.loads(l) for l in (ROOT/'data'/'timestep.jsonl').read_text().splitlines() if l.strip()]
    torus = [dict(grid='torus', width=r['grid'][1], dx_eq_computed_m=r['dx_eq_computed_m'], r_max_computed=r['r_max_computed'],
                  dt_max_s=r['dt_max_s']) for r in ts if r.get('grid', [''])[0] == 'torus' and r.get('scheme_str') == configs.MAIN
             and r['grid'][1] in (64, 128, 256)]
    latlon = [dict(grid='lat-lon', width=r['grid'][1], dx_eq_computed_m=r['dx_eq_computed_m'], r_max_computed=r['r_max_computed'],
                   dt_max_s=r['dt_max_s']) for r in ts if r.get('grid', [''])[0] == 'dfs'
              and r.get('scheme', {}).get('order') == 2 and 'r_max_unique' in r]
    write_csv('stable_step_vs_spacing.csv', torus + latlon)
    (PACK/'model.json').write_text(json.dumps({k:s[k] for k in ('step_ratio','cost','patch_model','calibration')},indent=2)+'\n')
    print('Saved figure tables to',PACK)

if __name__ == '__main__':
    main()
