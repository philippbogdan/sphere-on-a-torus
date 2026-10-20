"""Recompute the headline numbers from saved measurements and the dataflow model."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from polarswe import configs, dataflow as df


def rows(name):
    return [json.loads(line) for line in (ROOT / 'data' / name).read_text().splitlines() if line.strip()]


def combined(name):
    return rows(name + '.jsonl') + rows(name + '_ref.jsonl')


def main_record(records, case, m):
    hits = [r for r in records if r.get('grid') == ['torus', m] and r.get('case') == case
            and r.get('spec', '').split(':', 3)[-1] == configs.MAIN]
    if len(hits) != 1:
        raise ValueError(f'Expected one main record for {case}, M={m}; found {len(hits)}')
    return hits[0]


def calculate():
    ts = rows('timestep.jsonl')
    constants = [r['r_max_unique'] / math.sin(math.pi / (2 * r['grid'][1])) for r in ts
                 if r.get('grid', [''])[0] == 'dfs' and r.get('scheme', {}).get('order') == 2
                 and 'r_max_unique' in r]
    polar_constant = sum(constants) / len(constants)
    steps = {m: next(r for r in ts if r.get('grid') == ['torus', m]
                    and r.get('scheme_str') == configs.MAIN) for m in (64, 128, 256)}
    cost = configs.cost(configs.MAIN_NAME)
    kernel = df.dt_kernel('main', cost['flops'], df.EFFICIENCY_SWE, configs.STATE_WORDS,
                          cost['static_words'], cost['reach'])
    n_latlon = df.latlon_points(15)
    # Match the lat-lon equatorial spacing by mean unique-cell area on the torus.
    unique_torus = df.AREA / 463.3 ** 2
    floor = df.config_rate(df.paper_kernel('swe'), n_latlon)
    patch = df.cone_patch_cost(kernel, cost['flops'], cost['cone_flops'], cost['cone_R'],
                              cost['cone_static_words'], 2 * unique_torus)
    r_polar = polar_constant * math.sin(math.pi / (2 * 43200))
    dt_ratio = steps[256]['r_max_computed'] * df.dx_eq(2 * unique_torus) / (r_polar * df.dx_eq(n_latlon))
    speed = [dict(layout='Polar-limited lat-lon', relative=1.0, steps_per_s=floor['rate']),
             dict(layout='Torus, patch work counted', relative=dt_ratio * patch['steps_per_s_no_redistribution'] / floor['rate'],
                  steps_per_s=patch['steps_per_s_no_redistribution']),
             dict(layout='Torus, shared patch work', relative=dt_ratio * patch['steps_per_s_redistributed'] / floor['rate'],
                  steps_per_s=patch['steps_per_s_redistributed'])]
    trade = []
    # Table F holds the kernel and unique-point count fixed and omits patch-core work for all layouts.
    for arcsec in (15, 30, 60, 120, 240, 480, 960):
        n = df.latlon_points(arcsec)
        tor = df.approach_throughput('torus', n, kernel, steps[64]['r_max_unique'], polar_constant)
        filt = df.approach_throughput('latlon_filter', n, kernel, steps[64]['r_max_unique'], polar_constant)
        trade.append(dict(arcsec=arcsec, filter_over_torus=filt['throughput']/tor['throughput'],
                          torus_steps_per_s=tor['rate'], filter_steps_per_s=filt['rate'], filter_limit=filt['limit']))
    main = combined('accuracy_main')
    trsk = combined('mpas')
    series = rows('tc6_series.jsonl')
    pairs = []
    for m, level in ((144, 5), (288, 6), (576, 7)):
        if not any(r.get('grid') == ['torus',m] for r in main):
            continue
        for case, label in [('tc2_a0.7','TC2, day 5'), ('tc5','TC5, day 15'),
                            ('tc6','TC6, day 14'), ('galewsky_nopert','Steady jet, day 6'),
                            ('galewsky','Perturbed jet, day 6')]:
            a = main_record(main, case, m)
            b = next(r for r in trsk if r.get('grid') == ['icos',level] and r.get('case') == case)
            pairs.append(dict(case=label, m=m, torus_unique=a['nunique'], trsk_cells=b['ncells'],
                              torus_l2=a['l2'], trsk_l2=b['l2'], ratio=b['l2']/a['l2'],
                              torus_linf=a['linf'], trsk_linf=b['linf']))
            if case == 'tc6':
                ss = next(r for r in series if r['M'] == m and r.get('scheme') == configs.MAIN)
                av, bv = ss['l2_by_day']['5'], b['l2_by_day']['5']
                pairs.append(dict(case='TC6, day 5', m=m, torus_unique=a['nunique'], trsk_cells=b['ncells'],
                                  torus_l2=av,trsk_l2=bv,ratio=bv/av))
    spectral = []
    for m, trunc in ((128,42),(256,85),(512,170)):
        if not any(r.get('grid') == ['torus',m] and r.get('case') == 'tc6' for r in main):
            continue
        a = main_record(main,'tc6',m)
        b = next(r for r in rows('stswm_errors.jsonl') if r.get('truncation') == trunc
                 and r.get('case') == 'tc6' and r.get('against') == 'T213dt22' and not r.get('kind'))
        spectral.append(dict(m=m, torus_l2=a['l2'], spectral_l2=b['l2'], torus_over_spectral=a['l2']/b['l2']))
    fp32 = rows('fp32.jsonl')
    for m in (512,1024,2048):
        if (ROOT/'data'/f'fp32_gpu_M{m}.jsonl').exists():
            fp32 += rows(f'fp32_gpu_M{m}.jsonl')
    calibration = []
    for n, measured in {2:1.6e6,4:1.04e6,8:0.63e6,16:0.35e6,32:1.05e5,64:2.7e4}.items():
        model = df.step_time(df.paper_kernel('he5'),n,df.Machine(cores_w=720,cores_h=720))['rate']
        calibration.append(dict(n=n,model=model,paper_figure=measured,relative_error=model/measured-1))
    return dict(speed_model=speed, step_ratio=dt_ratio, cost=cost, patch_model=patch,
                polar_constant=polar_constant, stable_step=[dict(m=m,r=s['r_max_computed']) for m,s in steps.items()],
                filter_trade_model=trade, accuracy=pairs, tc6_spectral=spectral, fp32=fp32, calibration=calibration)


def verify(summary):
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Saved data changed: {name}')
    stamps = json.loads((ROOT/'data/source-stamps.json').read_text())
    assert set(stamps.values()) == {manifest['source_commit']}
    commits = {r['commit'] for p in (ROOT/'data').glob('*.jsonl') for r in rows(p.name) if r.get('commit')}
    assert commits == {manifest['source_commit']}, commits
    assert summary['cost']['reach'] == 42 and summary['cost']['flops'] == 2163
    assert 850 < summary['speed_model'][1]['relative'] < 860
    assert 1250 < summary['speed_model'][2]['relative'] < 1270
    assert len(summary['accuracy']) == 18
    assert len(summary['tc6_spectral']) == 3
    assert all(r['ratio'] > 1 for r in summary['accuracy'])
    assert all(1.8 < r['torus_over_spectral'] < 2.0 for r in summary['tc6_spectral'])
    assert max(r['M'] for r in summary['fp32']) == 2048
    assert all(2.5e-8 < r['fp32_reference_minus_fp64_l2'] < 3e-8 for r in summary['fp32'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, help='Write the calculated numbers as JSON.')
    ap.add_argument('--rerun', action='store_true', help='Use newly computed data without checking the saved-data hashes or expected ranges.')
    args = ap.parse_args()
    summary = calculate()
    if not args.rerun:
        verify(summary)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(summary,indent=2)+'\n')
    print('Modelled speed ratios:', ', '.join(f"{r['relative']:.2f}" for r in summary['speed_model']))
    print('TRSK comparisons:', len(summary['accuracy']), 'pairs; lowest l2 ratio:',f"{min(r['ratio'] for r in summary['accuracy']):.3f}")
    print('Filter / torus:', ', '.join(f"{r['arcsec']}: {r['filter_over_torus']:.3f}" for r in summary['filter_trade_model']))
    print('Saved evidence checks passed.' if not args.rerun else 'Calculated new results.')

if __name__ == '__main__':
    main()
