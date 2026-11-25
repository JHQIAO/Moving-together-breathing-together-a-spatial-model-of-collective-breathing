"""Audit and analyse the ten-run ablation extension without running simulations.

Full-model W1 trajectories are truncated to the original 7,500-s W2 horizon.
Other horizons remain 7,500 s, except the original 3,600-s drive-off protocol.
--partial writes explicitly suffixed preliminary files while new runs finish.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W1 = ROOT / 'study_out/rz75_paper/W1_reference'
W2 = ROOT / 'study_out/rz75_paper/W2_mechanism'
DATA, QA = HERE/'source_data', HERE/'qa'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE/'cache/numba')
os.environ['MPLCONFIGDIR'] = str(HERE/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(HERE/'cache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(W2))
import numpy as np
import pandas as pd
from model_w2 import settings, code_pins, clean, SWITCHES
from metrics_w2 import canonical

ARMS = ['A0','A1','A2','A8','A6']
SEEDS = list(range(9001,9011))
LABELS = {'A0':'Full model','A1':'No rest zone','A2':'No oxygen drive',
          'A8':'No orientation or attraction','A6':'No surface avoidance'}
METRICS = ['gap_mean','small_fraction','individual_mean','polarization',
           'low_height_fraction','participation_mean','class_next_rho']
INPUTS = {}
DIGESTS = {}


def digest(path):
    path = Path(path)
    if path not in DIGESTS:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):
                h.update(block)
        DIGESTS[path] = h.hexdigest()
    return DIGESTS[path]


def source(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def atomic_json(path, values):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(values),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def atomic_csv(path, frame):
    temp = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp,index=False)
    temp.replace(path)


def horizon(arm):
    return 3600. if arm=='A2' else 7500.


def format_mean_sd(stats, places=2, scale=1.):
    mean, sd = stats['mean']*scale, stats['sd']*scale
    # Preserve a positive SD even when the default precision would show zero.
    while sd>0 and float(f'{sd:.{places}f}')==0 and places<10:
        places+=1
    return f'{mean:.{places}f} ± {sd:.{places}f}'


def job(arm,seed):
    return dict(name=f'{arm}_s{seed}',arm=arm,seed=seed,seconds=horizon(arm),
                burn=300.,depth=3.3,type=None)


def paths(arm,seed):
    if arm=='A0':
        return W1/f'records/reference_s{seed}.json',W1/f'trajectories/reference_s{seed}.npz'
    base = W2/'outputs' if seed<=9003 else HERE/'runs'
    return base/f'records/{arm}_s{seed}.json',base/f'trajectories/{arm}_s{seed}.npz'


def validate_record(arm,seed,record_path,trajectory_path,expected_pins):
    record = read_json(record_path)
    expected, changes = settings(job(arm,seed))
    assert record['params']==clean(asdict(expected)), (arm,seed,'parameters')
    assert record['seed']==seed
    assert digest(source(trajectory_path))==record['sha256'], (arm,seed,'trajectory digest')
    if arm=='A0':
        assert record['seconds']==13440. and record['burn']==300. and record['sample_dt']==1.
        saved_pins=record['source_sha256']
        audit={k:record[k] for k in ['budget_error','displacement_error','turn_violations','boundary_corrections']}
        audit.update(source='W1 full-duration numerical audit',checked_horizon_s=record['seconds'])
    else:
        assert record['job']==job(arm,seed), (arm,seed,'job')
        assert record['changes']==clean(changes)
        assert record['switches']==SWITCHES.get(arm,{})
        saved_pins=record['code_sha256']
        assert saved_pins==expected_pins, (arm,seed,'frozen code manifest')
        audit=dict(record['audit'])
        assert audit['reaction_count']==round(horizon(arm)/expected.reaction_interval)
        assert audit['initialization']=={'kind':'random'}
    for relative,expected_hash in saved_pins.items():
        assert digest(source(ROOT/relative))==expected_hash, (arm,seed,relative,'source changed')
    assert np.isfinite(audit['budget_error']) and audit['budget_error']<=1e-9
    assert np.isfinite(audit['displacement_error']) and audit['displacement_error']<=1e-10
    assert audit['turn_violations']==0
    assert audit['boundary_corrections']==0
    return record, audit


def analyse_one(arm,seed,record,trajectory_path,historical):
    T=horizon(arm)
    with np.load(trajectory_path) as f:
        times=f['time'].astype(float)
        chosen=times<=T
        t=times[chosen]
        z=f['pos'][chosen,:,2].astype(float)
        headings=f['heading'][chosen].astype(float)
        states=f['status'][chosen]
        types=f['type_id']
        episodes=f['episodes13']
        transitions=f['events']
        if arm!='A0':
            for key,dtype in record['storage_dtypes'].items():
                assert str(f[key].dtype)==dtype,(arm,seed,key,'storage dtype')
        else:
            for key in ['time','pos','heading','tank','status','type_id','episodes13','events']:
                assert str(f[key].dtype)==record['arrays'][key]['dtype'],(arm,seed,key,'storage dtype')
    # Only breaths that would have been logged by the chosen horizon exist in
    # this virtual prefix. This must precede event grouping or a later release
    # could extend the final event and alter retained inter-event intervals.
    ep=episodes[episodes['exit']<=T]
    ev=transitions[transitions['time']<=T]
    np.testing.assert_array_equal(t,np.arange(0,T+1,1.))
    assert z.shape==(len(t),200) and states.shape==z.shape
    assert np.isfinite(z).all() and np.isfinite(headings).all()
    assert z.min()>=-1e-6 and z.max()<=3.3+1e-6
    assert np.isin(states,[0,1,2,3,4]).all()
    assert np.all(ep['release']>=ep['depart']) and np.all(ep['exit']>=ep['release'])
    assert len(ep)>0 and len(types)==200
    school=states==0
    n_school=school.sum(axis=1)
    assert np.all(n_school>0)
    mean_heading=(headings*school[:,:,None]).sum(axis=1)/n_school[:,None]
    polarization=np.linalg.norm(mean_heading,axis=1)
    median_height=np.median(z,axis=1)
    valid=(t>=300.)&(t<T)
    # Keep the original canonical implementation and export only requested,
    # whole-population metrics (no historical observation-slab statistics).
    metrics,events=canonical(dict(episodes13=ep,type_id=types),200,T,300.)
    row=dict(arm=arm,label=LABELS[arm],seed=seed,seconds=T,burn=300.,
             gap_mean=float(metrics['gap_mean']),small_fraction=float(metrics['small_fraction']),
             individual_mean=float(metrics['individual_mean']),
             polarization=float(polarization[valid].mean()),
             low_height_fraction=float((median_height[valid]<1.).mean()),
             participation_mean=float(metrics['participation_mean']),
             class_next_rho=float(metrics['class_next_rho']),
             n_breaths=int(metrics['n_breaths']),n_events=int(metrics['n_bouts']),
             n_intervals=int(metrics['n_gaps']),n_individual_intervals=len(events['individual']),
             source_record=str(paths(arm,seed)[0].relative_to(ROOT)),
             source_trajectory=str(trajectory_path.relative_to(ROOT)),
             source_duration_s=record['seconds'],
             truncated_from_long_reference=arm=='A0')
    assert np.isfinite([row[k] for k in METRICS]).all(),(arm,seed,'nonfinite output')
    assert row['gap_mean']>0 and row['individual_mean']>0
    assert 0<=row['small_fraction']<=1 and 0<=row['low_height_fraction']<=1
    row['rhythm_flag']=bool(row['polarization']<.2 and row['low_height_fraction']>=.95)
    comparison=None
    if seed<=9003:
        original=historical[(historical.arm==arm)&(historical.seed==seed)&
                            (historical.depth==3.3)&historical['type'].isna()]
        assert len(original)==1
        reference=original.iloc[0]
        pairs={**{k:k for k in METRICS if k!='low_height_fraction'},
               'low_height_fraction':'bottom_fraction',
               'n_events':'n_bouts','n_intervals':'n_gaps','n_breaths':'n_breaths'}
        differences={}
        for key,old_key in pairs.items():
            np.testing.assert_allclose(row[key],reference[old_key],rtol=0,atol=1e-10,
                                       err_msg=f'{arm} s{seed} {key}')
            differences[key]=float(row[key]-reference[old_key])
        comparison=dict(arm=arm,seed=seed,all_metrics_reproduced=True,differences=differences)
        if arm=='A0':
            with np.load(source(W2/f'outputs/trajectories/A0_s{seed}.npz')) as f:
                np.testing.assert_array_equal(ep,f['episodes13'])
                np.testing.assert_array_equal(ev,f['events'])
                np.testing.assert_array_equal(t,f['time'])
            comparison['original_episode_transition_time_prefix_exact']=True
    return row,comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true',help='Audit available runs; write .partial files')
    args=parser.parse_args()
    for directory in [DATA,QA]:directory.mkdir(exist_ok=True)
    source(Path(__file__).resolve())
    source(W2/'metrics_w2.py')
    source(ROOT/'breathing_study.py')
    expected_pins=code_pins()
    protocol=read_json(HERE/'runs/protocol.json')
    assert protocol['code_sha256']==expected_pins
    assert digest(source(HERE/'run_missing.py'))==protocol['runner_sha256']
    inventory=[];missing=[]
    for arm in ARMS:
        for seed in SEEDS:
            rp,tp=paths(arm,seed)
            if rp.exists():
                assert tp.exists(),(arm,seed,'record without trajectory')
                inventory.append((arm,seed,rp,tp))
            else:missing.append(dict(arm=arm,seed=seed))
    if missing and not args.partial:
        raise SystemExit(f'{len(missing)} runs still missing; use --partial for preliminary validation.')
    history_path=source(W2/'analysis/per_run_metrics.csv')
    historical=pd.read_csv(history_path)
    rows=[];audits=[];comparisons=[]
    for arm,seed,rp,tp in inventory:
        record,audit=validate_record(arm,seed,rp,tp,expected_pins)
        row,comparison=analyse_one(arm,seed,record,tp,historical)
        rows.append(row)
        audits.append(dict(arm=arm,seed=seed,trajectory_hash_verified=True,
                           source_hashes_verified=True,numerical_audit=audit))
        if comparison is not None:comparisons.append(comparison)
        print(f'{arm} s{seed}: verified; interval {row["gap_mean"]:.5f} s; flag {row["rhythm_flag"]}',flush=True)
    frame=pd.DataFrame(rows)
    assert len(comparisons)==15,'Must reproduce all original five-arm, three-run results'
    conditions={};summary_rows=[]
    for arm in ARMS:
        v=frame[frame.arm==arm].sort_values('seed')
        if not args.partial:np.testing.assert_array_equal(v.seed,SEEDS)
        assert len(v)>=3 and v.seed.is_unique
        stats={key:dict(mean=float(v[key].mean()),sd=float(v[key].std(ddof=1))) for key in METRICS}
        flags=int(v.rhythm_flag.sum())
        conditions[arm]=dict(label=LABELS[arm],n=len(v),seeds=v.seed.tolist(),seconds=horizon(arm),
                             statistics=stats,flagged_runs=flags,all_runs_flagged=flags==len(v))
        row=dict(arm=arm,label=LABELS[arm],n=len(v),seconds=horizon(arm),flagged_runs=flags,
                 rhythm_flag_all=flags==len(v))
        for key,val in stats.items():row.update({key+'_mean':val['mean'],key+'_sd':val['sd']})
        summary_rows.append(row)
    rules=dict(burn_s=300,saved_trajectory_interval_s=1,
        episode_prefix='Keep completed episodes with exit <= analysis horizon before grouping.',
        collective_events='Original metrics_w2.canonical: release-to-exit chains with 1-s separation; unique participants; event start >=300 s and event end <=T-2 s.',
        individual_intervals='Consecutive completed episodes per fish; preceding exit >=300 s and next release <=T; pool intervals within each run.',
        trace_time_mask='300 <= saved time < T (final sample excluded).',
        polarization='Norm of mean 3D heading over current schooling fish, averaged across retained samples.',
        low_height_fraction='Fraction of retained samples with the whole-population median height <1 BL.',
        rhythm_flag='Descriptive post-hoc flag: mean schooling polarization <0.2 AND low_height_fraction >=0.95. Not a statistical test for periodicity.',
        uncertainty='Equal-weight run means; ±1 sample SD (ddof=1).',
        baseline='Ten W1 runs truncated to7500 s; do not substitute their full13440-s metrics.',
        horizons={arm:horizon(arm) for arm in ARMS})
    result=dict(complete=not missing,partial_mode=args.partial,n_runs_total=len(frame),
                n_runs_expected=50,missing=missing,conditions=conditions,rules=rules)
    suffix='.partial' if args.partial else ''
    atomic_csv(DATA/f'per_run_metrics{suffix}.csv',frame)
    atomic_csv(DATA/f'summary{suffix}.csv',pd.DataFrame(summary_rows))
    atomic_json(DATA/f'summary{suffix}.json',result)
    table=[]
    for arm in ARMS:
        c=conditions[arm];s=c['statistics']
        table.append(dict(condition=c['label'],n=c['n'],duration_s=c['seconds'],
            collective_interval_s=format_mean_sd(s['gap_mean']),
            individual_interval_s=format_mean_sd(s['individual_mean']),
            small_events_percent=format_mean_sd(s['small_fraction'],scale=100),
            schooling_polarization=format_mean_sd(s['polarization'],places=3),
            low_height_percent=format_mean_sd(s['low_height_fraction'],scale=100),
            flagged_runs=f"{c['flagged_runs']}/{c['n']}"))
    atomic_csv(DATA/f'table_S6{suffix}.csv',pd.DataFrame(table))
    atomic_json(DATA/f'caption_counts{suffix}.json',dict(complete=not missing,
        n_by_condition={arm:c['n'] for arm,c in conditions.items()},
        flags_by_condition={arm:c['flagged_runs'] for arm,c in conditions.items()},
        horizons_s=rules['horizons'],burn_s=300,uncertainty=rules['uncertainty'],
        matched_horizon_note='Full-model results use the first7500 s of each long reference run, matching the original ablation horizon.',
        rules=rules))
    outputs=[DATA/f'{name}{suffix}.{ext}' for name,ext in [
        ('per_run_metrics','csv'),('summary','csv'),('summary','json'),('table_S6','csv'),('caption_counts','json')]]
    atomic_json(QA/f'analysis_checks{suffix}.json',dict(passed=True,complete=not missing,
        old_results_reproduced=len(comparisons),historical_comparisons=comparisons,
        run_audits=audits,inputs=INPUTS,
        outputs={str(p.relative_to(ROOT)):digest(p) for p in outputs},
        rules=rules,simulations_run_by_analysis=0))
    print(json.dumps(dict(passed=True,complete=not missing,n_runs=len(frame),
                          counts={arm:c['n'] for arm,c in conditions.items()},missing=len(missing)),indent=2))


if __name__=='__main__':main()
