"""Recover Figure 4a fish-event depths from saved trajectories; no model imports.

The reconstruction follows W6_behaviour_limits/analyse_behaviour.py exactly:
left-limit reserves, linearly interpolated positions, and saved complete events.
Every event count and mean is checked against that existing analysis.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'study_out/rz75_paper'
OUT = ROOT / 'study_out/local_work/figure4a_depth_distributions_20261013'
INPUTS = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source(path):
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def onset_reserves(data, query):
    """Same strict left-limit reconstruction as the saved W6 analysis."""
    assert np.all(np.diff(query) > 0)
    state = data['status'][0].copy()
    reserve = data['tank'][0].astype(float).copy()
    last = np.full(len(state), float(data['time'][0]))
    ev = data['events']
    ev = ev[np.argsort(ev['time'], kind='stable')]
    result = np.empty((len(query), len(state)))
    k = 0
    for j, when in enumerate(query):
        while k < len(ev) and ev['time'][k] < when - 1e-10:
            e = ev[k]
            i = e['agent']
            state[i], reserve[i], last[i] = e['to_state'], e['tank_after'], e['time']
            k += 1
        result[j] = np.maximum(reserve - data['agent_consumption_rates'] *
                               (when - last) * (state != 2), 0)
    return result


def extract_seed(seed, bins):
    tag = f'reference_s{seed}'
    record = json.loads(source(BASE/f'W1_reference/records/{tag}.json').read_text())
    trajectory = source(BASE/f'W1_reference/trajectories/{tag}.npz')
    assert INPUTS[str(trajectory.relative_to(ROOT))] == record['sha256']
    w6 = BASE/'W6_behaviour_limits/analysis'
    events = pd.read_csv(source(w6/f'{tag}_events.csv'), float_precision='round_trip')
    breathers = pd.read_csv(source(w6/f'{tag}_breathers.csv.gz'), float_precision='round_trip')
    reference = json.loads(source(w6/f'{tag}.json').read_text())
    assert reference['input_sha256'] == record['sha256']
    assert len(events) == reference['anatomy_events']
    with np.load(trajectory) as saved:
        data = {k: saved[k] for k in ['time', 'pos', 'status', 'tank', 'events', 'agent_consumption_rates']}
    query = events.onset.to_numpy()
    t = data['time'].astype(float)
    assert np.all(np.diff(t) > 0) and query.min() >= t[0] and query.max() <= t[-1]
    idx = np.clip(np.searchsorted(t, query, side='right') - 1, 0, len(t)-2)
    frac = (query - t[idx]) / (t[idx+1] - t[idx])
    z = data['pos'][:, :, 2]
    depth = record['params']['box'][2] - (z[idx] * (1-frac[:, None]) + z[idx+1] * frac[:, None])
    q = onset_reserves(data, query)
    assert np.isfinite(depth).all() and np.isfinite(q).all()
    assert np.all((depth >= 0) & (depth <= 3.3))
    members = breathers.groupby('event').agent.unique().to_dict()
    histograms = [[], []]
    means = [[], []]
    counts = [0, 0]
    rows = []
    max_mean_error = 0.
    for j, event in enumerate(events.itertuples(index=False)):
        participant = np.zeros(q.shape[1], dtype=bool)
        participant[members[event.event].astype(int)] = True
        ready = q[j] < .8
        assert int(ready.sum()) == event.ready_n
        assert int((ready & ~participant).sum()) == event.ready_missed_n
        for group, mask, expected in [(0, ready & participant, event.ready_participant_depth),
                                      (1, ready & ~participant, event.missed_depth)]:
            ids = np.flatnonzero(mask)
            values = depth[j, ids]
            if len(values) == 0:
                assert np.isnan(expected)
                continue
            mean = float(values.mean())
            np.testing.assert_allclose(mean, expected, rtol=0, atol=1e-10)
            max_mean_error = max(max_mean_error, abs(mean-expected))
            histogram = np.histogram(values, bins=bins)[0]
            assert histogram.sum() == len(values)
            histograms[group].append(histogram / len(values) / np.diff(bins))
            means[group].append(mean)
            counts[group] += len(values)
            rows.append(pd.DataFrame(dict(seed=seed, event=int(event.event), agent=ids,
                group=group, depth_bl=values, reserve=q[j, ids], within_event_weight=1/len(values))))
    density = np.array([np.mean(v, axis=0) for v in histograms])
    seed_means = np.array([np.mean(v) for v in means])
    for group, key in enumerate(['ready_participant_depth_event_mean', 'missed_depth_event_mean']):
        np.testing.assert_allclose(seed_means[group], reference['scalar'][key], rtol=0, atol=1e-10)
    observations = pd.concat(rows, ignore_index=True)
    event_counts = [len(v) for v in means]
    observations['weight'] = observations.within_event_weight / observations.group.map(dict(enumerate(event_counts))) / 10
    paired = events[['ready_participant_depth', 'missed_depth']].dropna()
    info = dict(seed=seed, complete_events=len(events), eligible_fish_event_counts=counts,
        nonempty_event_counts=event_counts,
        empty_event_counts=[len(events)-n for n in event_counts],
        excluded_reserve_ge_0_8=int(q.size-(q < .8).sum()),
        event_mean_max_abs_error=max_mean_error, means=seed_means.tolist(),
        paired_events=len(paired), paired_event_mean_difference=float((paired.missed_depth-paired.ready_participant_depth).mean()))
    print(f'{tag}: retained {counts} fish-event observations; event means verified', flush=True)
    return observations, density, seed_means, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = OUT/'depth_provenance.json'
    if args.check:
        saved = json.loads(manifest.read_text())
        for name, expected in {**saved['inputs'], **saved['outputs']}.items():
            assert digest(ROOT/name) == expected, name
        print('PASS: depth source and output hashes unchanged; no simulations.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    source(Path(__file__).resolve())
    source(BASE/'W6_behaviour_limits/analyse_behaviour.py')
    bins = np.linspace(0, 3.3, 34)
    extracted = [extract_seed(seed, bins) for seed in range(9001, 9011)]
    observations = pd.concat([item[0] for item in extracted], ignore_index=True)
    densities = np.array([item[1] for item in extracted])
    means = np.array([item[2] for item in extracted])
    np.testing.assert_allclose((densities*np.diff(bins)).sum(axis=2), 1, rtol=0, atol=1e-12)
    for group in range(2):
        rows = observations[observations.group == group]
        np.testing.assert_allclose(rows.weight.sum(), 1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.average(rows.depth_bl, weights=rows.weight), means[:, group].mean(), rtol=0, atol=1e-10)
    stats = dict(groups=['Take part', 'Miss event'], bin_edges_bl=bins.tolist(),
        density_by_simulation=densities.tolist(), density_mean=densities.mean(axis=0).tolist(),
        density_sd=densities.std(axis=0, ddof=1).tolist(), mean_by_simulation=means.tolist(),
        mean=means.mean(axis=0).tolist(), sd=means.std(axis=0, ddof=1).tolist(), n=10,
        weighting='Equal fish within event/category, equal nonempty events within simulation, equal simulations.',
        sampling='Repeated fish-event observations; ten independent simulation replicates.',
        simulations_run=0, per_simulation=[item[3] for item in extracted])
    (OUT/'depth_distributions.json').write_text(json.dumps(stats, indent=2)+'\n')
    observations.to_csv(OUT/'depth_observations.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    density_rows = [dict(seed=seed, group=stats['groups'][g], lower_bl=bins[b], upper_bl=bins[b+1], density=densities[s,g,b])
                    for s,seed in enumerate(range(9001,9011)) for g in range(2) for b in range(len(bins)-1)]
    pd.DataFrame(density_rows).to_csv(OUT/'depth_densities.csv', index=False)
    pd.DataFrame(dict(seed=np.arange(9001,9011), take_part_mean_bl=means[:,0], miss_event_mean_bl=means[:,1],
                      paired_difference_bl=means[:,1]-means[:,0])).to_csv(OUT/'depth_means.csv', index=False)
    outputs = {str((OUT/name).relative_to(ROOT)): digest(OUT/name) for name in
        ['depth_distributions.json','depth_observations.csv.gz','depth_densities.csv','depth_means.csv']}
    manifest.write_text(json.dumps(dict(inputs=INPUTS, outputs=outputs, simulations_run=0), indent=2)+'\n')
    print('Depth distributions recovered and verified; original means:', stats['mean'])


if __name__ == '__main__':
    main()
