"""Rebuild every W4 table from checksum-verified run records and trajectories."""
from run_w4 import *
import csv,argparse
from collections import defaultdict
from scipy import stats
from breathing_study import load_empirical_fig3

def csv_out(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})

def empirical():
    from paper.results_pipeline.analysis import analyse_empirical
    ref=analyse_empirical()
    gap,duration,cls=load_empirical_fig3()
    starts=np.flatnonzero(~np.isfinite(gap));ends=np.r_[starts[1:],len(gap)]
    assert starts.tolist()==[0,16,133,267,376,517]
    segments=[(gap[a:b],duration[a:b],cls[a:b]) for a,b in zip(starts,ends)]
    def calc(chosen):
        g=np.concatenate([s[0][1:] for s in chosen]);c=np.concatenate([s[2] for s in chosen]);d=np.concatenate([s[1] for s in chosen])
        before=np.concatenate([s[2][:-1] for s in chosen]);after=np.concatenate([s[2][1:] for s in chosen])
        v=dict(gap_mean=np.mean(g),gap_sd=np.std(g,ddof=1),short_fraction=np.mean(g<5),long_fraction=np.mean(g>30),
          whole_duration_mean=np.mean(d),class_next_rho=stats.spearmanr(before,g).statistic,class_previous_rho=stats.spearmanr(after,g).statistic)
        for i,n in enumerate(['small','medium','large','huge'],1):v[n+'_fraction']=np.mean(c==i)
        return v
    point=calc(segments)
    assert np.isclose(point['gap_mean'],ref['gap_mean']) and np.isclose(point['gap_sd'],ref['gap_sd'])
    assert np.isclose(point['class_previous_rho'],ref['r_cls_prevgap'])
    rng=np.random.default_rng(493075);rep=[calc([segments[i] for i in rng.integers(0,6,6)]) for _ in range(2000)]
    ci={k:np.quantile([r[k] for r in rep],[.025,.975]).tolist() for k in point}
    result=dict(values=point,ci95=ci,ci_method='Percentile cluster bootstrap of six complete recording segments; 2000 replicates, seed 493075. No cross-segment gap pairs.',
      segment_events=[len(s[0]) for s in segments],n_intervals=int(np.isfinite(gap).sum()),n_events=len(cls),
      source_sha256=digest(ROOT/'arapaima-air-breathing/data/supplementary/Supplementary_Data_1.xlsx'),
      pipeline_values={k:v for k,v in ref.items() if not k.startswith('_')},
      caveat='Recorded participation is categorical; there is no exact recorded participation mean. Front-glass duration is not a whole-shoal event duration.',
      time_scale_scenario={k:1.5*v for k,v in point.items() if k in ['gap_mean','gap_sd','whole_duration_mean']})
    write_json(HERE/'analysis/empirical.json',result)
    return result

def aggregate(rows):
    result={}
    for k in rows[0]['metrics']:
        v=np.array([r['metrics'][k] for r in rows if r['metrics'][k] is not None],float)
        result[k]=dict(mean=float(v.mean()) if len(v) else None,min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None,
          sd=float(v.std(ddof=1)) if len(v)>1 else None,n_valid=len(v),n_seeds=len(rows))
    return result

def paired_statistics(values,controls):
    """Require all planned seed pairs for an aggregate; preserve undefineds."""
    values=np.asarray(values,float);controls=np.asarray(controls,float)
    delta=values-controls;ok=np.isfinite(delta)
    mu=float(np.mean(controls)) if np.isfinite(controls).all() else None
    spread=float(np.std(controls,ddof=1)) if np.isfinite(controls).all() and len(controls)>1 else None
    out=dict(n=len(delta),n_valid=int(ok.sum()),control_mean=mu,control_seed_sd=spread,
      paired_deltas=delta.tolist(),delta_mean=None,delta_min=None,delta_max=None,normalized_effect=None,spread_effect=None)
    if ok.all() and len(delta):
        change=float(delta.mean())
        out.update(delta_mean=change,delta_min=float(delta.min()),delta_max=float(delta.max()),
          normalized_effect=change/mu if mu is not None and abs(mu)>1e-12 else None,
          spread_effect=change/spread if spread is not None and spread>1e-12 else None)
    return clean(out)

def effects(groups,phase):
    base={r['seed']:r['metrics'] for r in groups[(phase,'control',300.)]};out=[]
    for (ph,arm,burn),rr in groups.items():
        if ph!=phase or burn!=300. or arm=='control':continue
        for key in RANK_METRICS:
            paired=[(r['seed'],r['metrics'][key],base[r['seed']][key]) for r in rr]
            out.append(dict(phase=ph,arm=arm,metric=key,**paired_statistics([x for s,x,y in paired],[y for s,x,y in paired])))
    return out

def check_control_prefix():
    jobs=shoal_jobs('control',{},'tier1')+shoal_jobs('control',{},'tier2',7500.)
    assert all(valid(j) for j in jobs)
    checks=[]
    for seed in SEEDS:
        with np.load(OUT/'trajectories'/f'tier1_control_s{seed}.npz') as short, np.load(OUT/'trajectories'/f'tier2_control_s{seed}.npz') as long:
            for key in ['time','pos','heading','tank','status']:
                a=short[key];b=long[key][:len(a)]
                assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),(seed,key)
            a=short['episodes13'];b=long['episodes13'];b=b[b['exit']<=1800.]
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),(seed,'episodes13')
        checks.append(dict(seed=seed,seconds=1800.,passed=True))
    write_json(HERE/'qa/control_prefix.json',checks)
    return checks

def analyse(partial=False):
    records=[];groups=defaultdict(list)
    for rp in sorted((OUT/'records').glob('*.json')):
        r=json.loads(rp.read_text());j=r['job'];assert valid(j)
        m=metric(j);row=dict(name=j['name'],phase=j['phase'],arm=j['arm'],seed=j['seed'],seconds=j['seconds'],burn=300.,type=j['type'],
          metrics=m,changed_fields=r['changed_fields'],elapsed_s=r['elapsed_s'],sha256=r['sha256'],audit=r['audit'],reused=r.get('reused'))
        records.append(row)
        if j['type'] is None:
            groups[(j['phase'],j['arm'],300.)].append(row)
            if j['phase']=='numerical':
                groups[('numerical',j['arm'],900.)].append(row|dict(burn=900.,metrics=metric(j,900.)))
    for rr in groups.values():rr.sort(key=lambda r:r['seed'])
    summary=[dict(phase=ph,arm=a,burn=b,seeds=[r['seed'] for r in rr],metrics=aggregate(rr)) for (ph,a,b),rr in groups.items()]
    write_json(HERE/'analysis/summary.json',summary)
    csv_out(HERE/'analysis/metrics_by_seed.csv',[{k:v for k,v in r.items() if k!='metrics'}|r['metrics'] for r in records])
    csv_out(HERE/'analysis/summary.csv',[dict(phase=r['phase'],arm=r['arm'],burn=r['burn'],metric=k,**v) for r in summary for k,v in r['metrics'].items()])
    empirical()
    if partial:return summary
    rank=json.loads((HERE/'analysis/ranking.json').read_text())
    assert len([r for r in records if r['phase']=='tier1'])==123
    assert len([r for r in records if r['phase']=='solo'])==369
    assert len([r for r in records if r['phase']=='tier2'])==51
    assert len([r for r in records if r['phase']=='numerical'])==9
    assert all(len(rr)==3 for rr in groups.values())
    eff=effects(groups,'tier1')+effects(groups,'tier2')
    write_json(HERE/'analysis/paired_effects.json',eff);csv_out(HERE/'analysis/paired_effects.csv',eff)
    numerical=[];base={r['seed']:r['metrics'] for r in groups[('numerical','control',300.)]}
    for (ph,arm,burn),rr in groups.items():
        if ph!='numerical' or (arm=='control' and burn==300.):continue
        for key in RANK_METRICS:
            numerical.append(dict(arm=arm,burn=burn,metric=key,**paired_statistics([r['metrics'][key] for r in rr],[base[r['seed']][key] for r in rr])))
    write_json(HERE/'analysis/numerical_effects.json',numerical);csv_out(HERE/'analysis/numerical_effects.csv',numerical)
    calibrations=[json.loads((HERE/'analysis'/f'calibration_{p}_{v}.json').read_text()) for p in rank['selected'] for v in ['low','high']]
    write_json(HERE/'analysis/calibrations.json',calibrations)
    csv_out(HERE/'analysis/calibrations.csv',[dict(arm=c['arm'],type=t,rate=c['rates'][t],training_error=c['history'][t][-1]['error_fraction'],
      updates=c['history'][t][-1]['iteration'],training_pass=c['training_pass'][t],heldout_error=c['heldout'][t]['error_fraction'],heldout_pass=c['heldout_pass'][t]) for c in calibrations for t in range(3)])
    # Provenance and physical audits are reported, never silently clipped.
    launches=[json.loads(line) for line in (OUT/'launches.jsonl').read_text().splitlines()]
    assert all(0<=r['active_before']<r['budget']<=4 for r in launches),'CPU launch budget violation'
    prefix_checks=check_control_prefix()
    qa=dict(n_records=len(records),n_shoal=sum(r['type'] is None for r in records),n_solo=sum(r['type'] is not None for r in records),
      launch_budget_verified=True,n_launches=len(launches),
      short_long_control_prefix_bitwise=prefix_checks,
      all_checksums_verified=True,max_budget_error=max(r['audit']['budget_error'] for r in records),
      max_displacement_error=max(r['audit']['displacement_error'] for r in records),
      nonzero_turn_violations=[dict(name=r['name'],values=r['audit']['turn_violations']) for r in records if np.any(r['audit']['turn_violations'])],
      nonzero_boundary_corrections=[dict(name=r['name'],values=r['audit']['boundary_corrections']) for r in records if np.any(r['audit']['boundary_corrections'])],
      reused=[dict(name=r['name'],**r['reused']) for r in records if r.get('reused')],
      not_converged=[dict(arm=c['arm'],type=t) for c in calibrations for t in range(3) if not c['training_pass'][t]],
      heldout_failures=[dict(arm=c['arm'],type=t,error=c['heldout'][t]['error_fraction']) for c in calibrations for t in range(3) if not c['heldout_pass'][t]],
      undefined_paired_statistics=[r for r in eff+numerical if r['delta_mean'] is None],
      source_pins={p.name:digest(p) for p in HERE.glob('*.py')})
    write_json(HERE/'qa/final_analysis.json',qa)
    write_json(OUT/'manifest.json',{rp.name:digest(rp) for rp in sorted((OUT/'records').glob('*.json'))})
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--prefix-only',action='store_true')
    args=ap.parse_args()
    if args.prefix_only:print(json.dumps(check_control_prefix()))
    else:analyse(args.partial)
