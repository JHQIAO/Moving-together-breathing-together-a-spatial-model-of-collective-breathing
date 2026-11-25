"""Read saved outcomes only; prepare the requested figure without changing TeX."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'study_out/rz75_paper'
DATA = HERE / 'source_data'
QA = HERE / 'qa'
for p in [DATA, QA]:
    p.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(9001, 9011))
TYPES = ['Fast', 'Medium', 'Slow']
HASHES = {}
CHECKS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def source(path):
    path = Path(path)
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    return path

def read_json(path):
    return json.loads(source(path).read_text())

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')

def summary(values):
    a = np.asarray(values, float)
    assert np.isfinite(a).all() and len(a) >= 2
    return dict(mean=a.mean(axis=0), sd=a.std(axis=0, ddof=1), n=len(a), per_run=a)

def floor_intervals(t, height):
    below = height < 1
    indices = np.flatnonzero(below[:-1] != below[1:])
    crossing = t[indices] + (1-height[indices]) / (height[indices+1]-height[indices]) * (t[indices+1]-t[indices])
    down = crossing[~below[indices]]
    up = crossing[below[indices]]
    if below[0]: down = np.r_[t[0], down]
    if below[-1]: up = np.r_[up, t[-1]]
    return np.column_stack([down, up])

def rest_and_drive(e, tr, total, burn=300):
    rests = floor_intervals(tr['time'], tr['median'])
    rows = []
    for j, (end, start) in enumerate(zip(e['end'][:-1], e['start'][1:])):
        overlap = np.maximum(0, np.minimum(rests[:, 1], start) - np.maximum(rests[:, 0], end))
        rows.append(dict(event=j, end=end, next_start=start, participation=e['part'][j],
                         cls=int(e['cls'][j]), gap=start-end, rest_duration=overlap.sum()))
    frame = pd.DataFrame(rows)
    class_means = [frame.loc[frame.cls == c, 'rest_duration'].mean() for c in range(1, 5)]
    lags = np.arange(41)
    curves, counts, uninterrupted = {}, {}, {}
    for label, mask in [('small', e['part'] < .2), ('large', e['part'] >= .5)]:
        ids = np.flatnonzero(mask & (e['end'] >= burn) & (e['end'] <= total-40))
        assert len(ids)
        arr = np.asarray([np.interp(e['end'][j]+lags, tr['time'], tr['mean_need']) for j in ids])
        curves[label] = arr.mean(axis=0)
        counts[label] = len(ids)
        next_start = np.r_[e['start'][1:], total]
        uninterrupted[label] = ((next_start[ids]-e['end'][ids])[:, None] > lags).mean(axis=0)
    return np.array(class_means), curves, counts, uninterrupted, frame

def rise_check(e, tr, seed):
    # Same sampled whole-shoal 1 BL upward-crossing definition as Fig. 3.
    t, h = tr['time'], tr['median']
    lift = t[np.flatnonzero((h[:-1] < 1) & (h[1:] >= 1)) + 1]
    lift = lift[lift >= 300]
    ix = np.searchsorted(lift, e['end'], side='right')
    valid = ix < len(lift)
    frame = pd.DataFrame(dict(seed=seed, event=np.flatnonzero(valid),
        event_end=e['end'][valid], participation=e['part'][valid], cls=e['cls'][valid],
        next_liftoff=lift[ix[valid]], time_to_next_liftoff=lift[ix[valid]]-e['end'][valid]))
    means = [frame.loc[frame.cls == c, 'time_to_next_liftoff'].mean() for c in range(1, 5)]
    result = dict(seed=seed, n_complete=len(frame), n_right_censored=int((~valid).sum()),
        class_next_liftoff_rho=stats.spearmanr(frame.cls, frame.time_to_next_liftoff).statistic,
        participation_next_liftoff_rho=stats.spearmanr(frame.participation, frame.time_to_next_liftoff).statistic)
    return frame, np.asarray(means), result

def main():
    snapshot = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'paper_draft').rglob('*.tex')}
    write_json(QA/'tex_hashes_before.json', snapshot)
    out = dict(metadata=dict(reference_seeds=SEEDS, reference_seconds=13440, burn_seconds=300,
        isolated_seeds=list(range(9601, 9611)), uncertainty='Mean ±1 sample SD across runs (ddof=1); record 95% block-bootstrap CI',
        no_new_simulations=True))
    old = ROOT/'study_out/local_work/figure4a_depth_distributions_20261013'
    obs = pd.read_csv(source(old/'depth_observations.csv.gz'))
    bins = np.asarray(read_json(old/'depth_distributions.json')['bin_edges_bl'])
    retained = obs[obs.reserve < .75]
    assert len(retained) == 544557
    means, densities, count_rows = [], [], []
    for seed in SEEDS:
        seed_means, seed_densities = [], []
        for group in [0, 1]:
            part = retained[(retained.seed == seed) & (retained.group == group)]
            event_hist, event_mean = [], []
            for _, rows in part.groupby('event', sort=True):
                counts = np.histogram(rows.depth_bl, bins=bins)[0]
                assert counts.sum() == len(rows)
                event_hist.append(counts / len(rows) / np.diff(bins))
                event_mean.append(rows.depth_bl.mean())
            seed_means.append(np.mean(event_mean))
            seed_densities.append(np.mean(event_hist, axis=0))
            count_rows.append(dict(seed=seed, group=group, fish_event_observations=len(part), nonempty_events=len(event_mean)))
        means.append(seed_means); densities.append(seed_densities)
    np.testing.assert_allclose((np.asarray(densities)*np.diff(bins)).sum(axis=2), 1)
    out['a'] = dict(**summary(means), bin_edges=bins, density=np.mean(densities, axis=0),
        density_by_run=densities, input_observations=len(obs), retained_observations=len(retained),
        weighting='Equal fish within event/category, equal nonempty events within run, equal runs')
    pd.DataFrame(count_rows).to_csv(DATA/'a_counts.csv', index=False)
    decisions = read_json(BASE/'W11_decisions/analysis/summary.json')
    runs = sorted([r for r in decisions['per_run'] if r['arm'] == 'Shoal'], key=lambda r: r['seed'])
    assert [r['seed'] for r in runs] == SEEDS
    need = 100*np.array([[r['types'][typ]['need_fraction'] for typ in TYPES] for r in runs])
    full = 100*np.array([[r['types'][typ]['full_fraction'] for typ in TYPES] for r in runs])
    np.testing.assert_allclose(need+full, 100)
    out['b'] = dict(below=summary(need), above=summary(full), reserve_time='accepted breathing decision', threshold=.75)
    models = pd.read_csv(source(BASE/'W1_reference/analysis/model_replicates.csv')).sort_values('seed')
    assert models.seed.tolist() == SEEDS
    alone = read_json(BASE/'W8_results_gaps/analysis/g1_model_alone.json')
    units = sorted(alone['units'], key=lambda r: (r['seed'], r['type']))
    for unit in units:
        np.testing.assert_allclose(unit['mean'], np.mean(unit['values']), atol=1e-12)
    isolated = []
    for typ in range(3):
        group = [u for u in units if u['type'] == typ]
        assert [u['seed'] for u in group] == list(range(9601, 9611))
        isolated.append([u['mean'] for u in group])
    out['c'] = dict(isolated=summary(np.array(isolated).T),
        shoal=summary(models[[f'individual_{t.lower()}' for t in TYPES]].values),
        within_run='Mean of eligible exit-to-next-release intervals pooled within breathing type')
    pd.DataFrame([{k:u[k] for k in ['seed', 'type', 'n', 'mean']} for u in units]).to_csv(DATA/'c_isolated_runs.csv', index=False)
    # Independent reproduction of the previous implementation on its original three runs.
    module = ast.parse(source(BASE/'W2_mechanism/analyze_w2.py').read_text())
    funcs = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in ['dwell_intervals', 'size_analysis']], type_ignores=[])
    namespace = dict(np=np, stats=stats, mean=lambda v: float(np.mean(v)), rho=lambda a,b: float(stats.spearmanr(a,b).statistic))
    exec(compile(funcs, '<read-only original analysis functions>', 'exec'), namespace)
    w2 = pd.read_csv(source(BASE/'W2_mechanism/analysis/per_run_metrics.csv'))
    rest_means, curve_values = [], {'small': [], 'large': []}
    events, curve_rows, curve_counts, rise_frames, rise_means, rise_summary = [], [], [], [], [], []
    checks = []
    for seed in SEEDS:
        record = read_json(BASE/f'W1_reference/records/reference_s{seed}.json')
        rawfile = source(BASE/f'W1_reference/trajectories/reference_s{seed}.npz')
        assert HASHES[str(rawfile.relative_to(ROOT))] == record['sha256']
        assert next(r for r in runs if r['seed'] == seed)['sha256'] == record['sha256']
        with np.load(source(BASE/f'W1_reference/analysis/reference_s{seed}_derived.npz')) as d:
            e = {dest:d[src].copy() for dest,src in [('end','end'),('start','start'),('part','parts'),('cls','classes'),('gaps','gaps')]}
            for typ in TYPES:
                np.testing.assert_allclose(np.mean(d[f'bytype__{typ.lower()}']), models.loc[models.seed == seed, f'individual_{typ.lower()}'].iloc[0])
        with np.load(rawfile) as z:
            t = np.asarray(z['time'], float)
            h = np.median(np.asarray(z['pos'][:,:,2], float), axis=1)
            reserve = np.asarray(z['tank'], float)
            school = z['status'] == 0
        p = record['params']
        assert p['need_setpoint'] == .75 and p['upward_strength'] == .35 and p['need_down'] == .1
        drive = np.where(reserve < .75, .35, .1)*(.75-reserve)
        tr = dict(time=t, median=h, mean_need=(drive*school).sum(axis=1)/school.sum(axis=1), all_mean_need=drive.mean(axis=1))
        assert np.isfinite(tr['mean_need']).all()
        total = record['seconds']
        cm, curves, counts, uninterrupted, frame = rest_and_drive(e, tr, total)
        original, _, cv = namespace['size_analysis'](e, tr, total, 300)
        np.testing.assert_allclose(cm, [original[f'class{c}_rest_mean'] for c in range(1,5)], rtol=0, atol=1e-12)
        for label in ['small', 'large']:
            np.testing.assert_allclose(curves[label], cv[label], rtol=0, atol=1e-12)
            curve_values[label].append(curves[label])
            for lag, value, fraction in zip(range(41), curves[label], uninterrupted[label]):
                curve_rows.append(dict(seed=seed, category=label, lag_s=lag, schooling_mean_drive=value, next_event_free_fraction=fraction))
        curve_counts.append(dict(seed=seed, **counts))
        frame.insert(0, 'seed', seed); events.append(frame); rest_means.append(cm)
        rframe, rmean, rsummary = rise_check(e, tr, seed)
        rise_frames.append(rframe); rise_means.append(rmean); rise_summary.append(rsummary)
        if seed <= 9003:
            with np.load(source(BASE/f'W2_mechanism/analysis/traces/A0_s{seed}.npz')) as oldtr:
                for key in ['median', 'mean_need', 'all_mean_need']:
                    np.testing.assert_allclose(tr[key][:len(oldtr['time'])], oldtr[key], rtol=0, atol=1e-12)
                shorttr = {k:oldtr[k].copy() for k in ['time','median','mean_need','all_mean_need']}
                mask = e['end'] <= 7498
                shorte = {k:v[mask] for k,v in e.items() if k != 'gaps'}
                shorte['gaps'] = shorte['start'][1:]-shorte['end'][:-1]
                np.testing.assert_allclose(shorte['start'], oldtr['event_start'], rtol=0, atol=1e-12)
            oldcm, oldcv, _, _, _ = rest_and_drive(shorte, shorttr, 7500)
            savedrow = w2[w2.name == f'A0_s{seed}'].iloc[0]
            np.testing.assert_allclose(oldcm, savedrow[[f'class{c}_rest_mean' for c in range(1,5)]].to_numpy(float), rtol=0, atol=1e-12)
            with np.load(source(BASE/f'W2_mechanism/analysis/A0_s{seed}_recovery.npz')) as saved:
                for label in ['small','large']:
                    np.testing.assert_allclose(oldcv[label], saved[label], rtol=0, atol=1e-12)
            checks.append(dict(seed=seed, original_7500_s_results='reproduced to 1e-12'))
        np.testing.assert_allclose(stats.spearmanr(e['cls'][:-1], e['gaps']).statistic, models.loc[models.seed == seed, 'class_next_rho'].iloc[0], atol=1e-12)
        print(f'Prepared seed {seed}: {len(e["start"])} events, rest and drive checked', flush=True)
    out['d'] = dict(**summary(rest_means), measure='Total time with whole-shoal median height below 1 BL within end-to-next-start interval; linearly interpolated threshold crossings')
    out['e'] = dict(lags=np.arange(41), small=summary(curve_values['small']), large=summary(curve_values['large']),
        event_counts=curve_counts, window='Complete 40 s windows; display 0–30 s; intervening events retained; schooling agents only')
    pd.concat(events).to_csv(DATA/'d_event_rest.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(DATA/'e_drive_per_run.csv', index=False)
    record = read_json(BASE/'W1_reference/analysis/record_uncertainty.json')
    no_rest = w2[(w2.arm == 'A1') & (w2.depth == 3.3) & w2['type'].isna()].sort_values('seed')
    weak = pd.read_csv(source(BASE/'W4_robustness/analysis/metrics_by_seed.csv'))
    weak = weak[(weak.arm == 'kappa_bottom_low') & (weak.phase == 'tier2') & (weak.burn == 300)].sort_values('seed')
    assert len(no_rest) == len(weak) == 3
    out['f'] = dict(record=dict(mean=record['point']['class_next_rho'], ci=record['methods']['block10']['ci']['class_next_rho']),
        full=summary(models.class_next_rho), no_rest=summary(no_rest.class_next_rho), weaker_rest=summary(weak.class_next_rho),
        full_seeds=SEEDS, no_rest_seeds=no_rest.seed.tolist(), weaker_rest_seeds=weak.seed.tolist(),
        correlation='Spearman rho between participation class (1–4) and next collective interval',
        variant_seconds=7500, variant_burn=300,
        variant_note='No-rest and weaker-rest each retain 3 saved runs. Historical elapsed times ~24 and 20 min/run: extending both to 10 needs ~5 CPU hours; no new simulations run. Weaker-rest uses its existing re-estimated isolated-fish depletion rates.')
    spatial = read_json(BASE/'W10_type_space/analysis/summary.json')
    spatial_runs = sorted(spatial['per_seed'], key=lambda r:r['seed'])
    assert [r['seed'] for r in spatial_runs] == SEEDS
    shares = 100*np.array([[r['events'][cat] for cat in ['schooling_top_share','first']] for r in spatial_runs])
    np.testing.assert_allclose(shares.sum(axis=2), 100)
    np.testing.assert_allclose(shares[:,1,:].mean(axis=0), 100*np.asarray(decisions['first_type_shares']['mean']))
    out['supplementary_type_shares'] = dict(**summary(shares), population_percent=100*np.array(spatial['metadata']['type_counts'])/200)
    out['supplementary_rise_check'] = dict(**summary(rise_means), per_run_correlations=rise_summary,
        rho=summary([r['class_next_liftoff_rho'] for r in rise_summary]),
        definition='Time from each completed event end to the next sampled upward crossing of 1 BL by whole-shoal median height. Events without an observed later crossing are excluded. Multiple events in one rise share the next crossing; no independent-event inference.')
    pd.concat(rise_frames).to_csv(DATA/'supplementary_event_to_next_liftoff.csv', index=False)
    pd.DataFrame(rise_summary).to_csv(DATA/'supplementary_rise_correlations.csv', index=False)
    rows = []
    for idx,seed in enumerate(SEEDS):
        for j,typ in enumerate(TYPES):
            rows += [dict(panel='b',seed=seed,category=typ,metric='reserve_below_percent',value=need[idx,j]),
                     dict(panel='c',seed=seed,category=typ,metric='shoal_interval_s',value=out['c']['shoal']['per_run'][idx,j])]
        for j,cat in enumerate(['<20','20–50','50–80','≥80']):
            rows.append(dict(panel='d',seed=seed,category=cat,metric='rest_s',value=rest_means[idx][j]))
    pd.DataFrame(rows).to_csv(DATA/'b_c_d_run_means.csv',index=False)
    write_json(DATA/'plotted_values.json', out)
    write_json(QA/'input_hashes.json', HASHES)
    write_json(QA/'numerical_checks.json', dict(original_three_run_reproduction=checks, all_ten_independent_function_checks=True,
        all_reference_trajectory_hashes_verified=True, accepted_decision_source_matches_reference=True,
        isolated_mean_checks=True, individual_interval_means_verified=True, correlation_checks=True,
        depth_density_normalization=True, uncertainty='Sample SD across runs, never range'))
    assert snapshot == {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper_draft').rglob('*.tex')}
    print('All numerical checks passed. TeX unchanged.')
    for key in ['d','supplementary_rise_check']:
        print(key, 'mean', out[key]['mean'], 'SD', out[key]['sd'])
    print('Rise-based correlation',out['supplementary_rise_check']['rho']['mean'],out['supplementary_rise_check']['rho']['sd'])

if __name__ == '__main__':
    main()
