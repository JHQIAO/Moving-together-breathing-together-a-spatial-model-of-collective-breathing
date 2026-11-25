"""Reusable RZ75 measurements. Inputs are saved trajectories; no simulations run here.

Event membership follows breathing_study.bout_events_from on release-sorted breaths.
Independent simulation seeds, not events/fish, define model replication.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S25 = ROOT / 'study_out/local_work/rest_zone_step25'
sys.path.insert(0, str(ROOT))
from breathing_study import bout_events_from, fit_report, load_empirical_fig3, EMP_XLSX
TYPES = ('fast', 'medium', 'slow')
ALONE_TARGETS = np.array([55.630943, 85.5925, 134.24])
SEGMENT_SIZES = (16, 117, 134, 109, 141, 101)

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def rho(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum()<3 or len(np.unique(x[ok]))<2 or len(np.unique(y[ok]))<2: return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)

def class_codes(part): return np.digitize(part, [.2, .5, .8])+1

def distribution_stats(gaps):
    g = np.asarray(gaps, float); g = g[np.isfinite(g)]
    fit = fit_report(g) if len(g)>=20 else fit_small(g)
    return dict(gap_mean=g.mean(), gap_sd=g.std(ddof=1), gap_cv=g.std(ddof=1)/g.mean(),
                gap_median=np.median(g), gap_skew=stats.skew(g),
                gap_kurtosis=stats.kurtosis(g, fisher=False),
                gap_kurtosis_unbiased=stats.kurtosis(g, fisher=False, bias=False),
                short_fraction=np.mean(g<5), long_fraction=np.mean(g>30),
                aic=fit['aic'], aic_winner=fit['aic_winner'],
                laplace_gamma=fit['aic']['laplace']-fit['aic']['gamma'])

def fit_small(g):
    # Same MLE constraints/counts as fit_report, including free exponential location.
    fits = {'laplace':(stats.laplace,{}), 'gamma':(stats.gamma,{'floc':0}),
            'lognorm':(stats.lognorm,{'floc':0}), 'norm':(stats.norm,{}), 'expon':(stats.expon,{})}
    aic={}
    for name,(dist,kw) in fits.items():
        prm=dist.fit(g,**kw)
        aic[name]=float(2*(len(prm)-len(kw))-2*dist.logpdf(g,*prm).sum())
    return dict(aic=aic,aic_winner=min(aic,key=aic.get))

def peak_occupancy(ep):
    # Sweep distinct fish present in [release, exit); process exits first on ties.
    ticks = [(float(t), 1, int(i)) for t,i in zip(ep['release'],ep['agent'])]
    ticks += [(float(t), -1, int(i)) for t,i in zip(ep['exit'],ep['agent'])]
    counts={}; peak=0
    for _, delta, i in sorted(ticks):
        counts[i]=counts.get(i,0)+delta
        if counts[i]==0: del counts[i]
        peak=max(peak,len(counts))
    return peak

def event_arrays(data, seconds, burn=300., N=None):
    e=data['episodes13']; e=e[np.argsort(e['release'],kind='stable')]
    if N is None: N=len(data['type_id'])
    bouts=bout_events_from(e['release'],e['exit'],e['agent'],gap=1.)
    bouts=[b for b in bouts if b['t_start']>=burn and b['t_end']<=seconds-2.]
    starts=np.array([b['t_start'] for b in bouts]); ends=np.array([b['t_end'] for b in bouts])
    parts=np.array([b['n']/N for b in bouts]); cls=class_codes(parts)
    windows={str(w):[] for w in (.25,.5,1.,2.)}
    peak=[]; repeat=[]; burst=[]; burst_distinct=[]; breath_duration=[]; counts=[]
    for b in bouts:
        a=np.searchsorted(e['release'],b['t_start']-1e-10)
        z=np.searchsorted(e['release'],b['t_end']+1e-10,side='right'); ep=e[a:z]
        assert len(np.unique(ep['agent']))==b['n'] and len(ep)==b['n_breaths']
        peak.append(peak_occupancy(ep)/N); repeat.append(len(ep)-b['n']); counts.append(len(ep))
        breath_duration.extend(ep['exit']-ep['release'])
        burst.append(float(ep['release'][9]-ep['release'][0]) if len(ep)>=10 else np.nan)
        first=np.sort(np.unique(ep['agent'],return_index=True)[1])
        burst_distinct.append(float(ep['release'][first[9]]-ep['release'][first[0]]) if len(first)>=10 else np.nan)
        for w in windows:
            obs=ep[ep['release_y']<=float(w)]
            windows[w].append(float(obs['exit'].max()-obs['release'].min()) if len(obs) else np.nan)
    individual=[]; bytype=[[] for _ in TYPES]; fishmeans=[[] for _ in TYPES]; individual_duration=[]
    for i in range(N):
        ep=e[e['agent']==i]; valid=(ep['exit'][:-1]>=burn)&(ep['release'][1:]<=seconds)
        vals=(ep['release'][1:]-ep['exit'][:-1])[valid]
        individual.extend(vals); typ=int(data['type_id'][i]); bytype[typ].extend(vals)
        if len(vals): fishmeans[typ].append(vals.mean())
        individual_duration.extend((ep['exit'][:-1]-ep['release'][:-1])[valid])
    return dict(gaps=starts[1:]-ends[:-1], parts=parts, classes=cls, start=starts,end=ends,
                durations=ends-starts, slab={k:np.array(v) for k,v in windows.items()},
                peak=np.array(peak), repeats=np.array(repeat), breath_counts=np.array(counts),
                burst10=np.array(burst),burst10_distinct=np.array(burst_distinct),breath_duration=np.array(breath_duration),
                individual=np.array(individual), individual_duration=np.array(individual_duration),
                bytype={n:np.array(v) for n,v in zip(TYPES,bytype)},
                fishmeans={n:np.array(v) for n,v in zip(TYPES,fishmeans)})

def summarize_arrays(a, alone_mean=85.091):
    g=a['gaps']; c=a['classes']; out=distribution_stats(g)
    out.update(n_intervals=len(g),n_events=len(c),participation_mean=np.mean(a['parts']),
               class_next_rho=rho(c[:-1],g),class_previous_rho=rho(c[1:],g),
               size_next_rho=rho(a['parts'][:-1],g),size_previous_rho=rho(a['parts'][1:],g),
               whole_duration_mean=a['durations'].mean(),whole_duration_sd=a['durations'].std(ddof=1),
               whole_size_duration_rho=rho(a['parts'],a['durations']),
               whole_class_duration_rho=rho(c,a['durations']),
               individual_mean=a['individual'].mean(),individual_sd=a['individual'].std(ddof=1),
               effect_size=alone_mean/g.mean(),repeat_share=a['repeats'].sum()/a['breath_counts'].sum(),
               repeat_event_share=np.mean(a['repeats']>0),breath_duration_mean=a['breath_duration'].mean(),
               burst10_mean=np.nanmean(a['burst10']),burst10_median=np.nanmedian(a['burst10']),
               burst10_within_half=np.mean(a['burst10'][np.isfinite(a['burst10'])]<=.5+1e-10),
               burst10_eligible=int(np.isfinite(a['burst10']).sum()))
    d=a['burst10_distinct'];valid=d[np.isfinite(d)]
    out.update(burst10_distinct_mean=np.mean(valid),burst10_distinct_median=np.median(valid),
               burst10_distinct_within_half=np.mean(valid<=.5+1e-10),burst10_distinct_eligible=len(valid))
    for i in range(1,5):
        out[f'class_{i}']=np.mean(c==i);out[f'peak_class_{i}']=np.mean(class_codes(a['peak'])==i)
        out[f'next_class_{i}']=np.mean(g[c[:-1]==i]) if np.any(c[:-1]==i) else np.nan
        out[f'previous_class_{i}']=np.mean(g[c[1:]==i]) if np.any(c[1:]==i) else np.nan
    for w,v in a['slab'].items():
        out[f'slab_{w}_mean']=np.nanmean(v); out[f'slab_{w}_sd']=np.nanstd(v,ddof=1)
        out[f'slab_{w}_n']=int(np.isfinite(v).sum());out[f'slab_{w}_size_rho']=rho(a['parts'],v)
        out[f'slab_{w}_class_rho']=rho(c,v)
    for n in TYPES:
        out[f'individual_{n}']=np.mean(a['bytype'][n]);out[f'individual_{n}_sd']=np.std(a['bytype'][n],ddof=1)
        out[f'fish_equal_{n}']=np.mean(a['fishmeans'][n])
    out['cycle_mean']=np.mean(g+a['durations'][:-1])
    out['budget_lhs']=out['participation_mean']*out['individual_mean']
    out['budget_ratio']=out['budget_lhs']/out['cycle_mean']
    out['budget_with_breath_duration']=out['participation_mean']*(out['individual_mean']+np.mean(a['individual_duration']))/(1-out['repeat_share'])
    out['budget_adjusted_ratio']=out['budget_with_breath_duration']/out['cycle_mean']
    # Exact finite-window count identity, using all complete cycles except last event.
    total_time=a['start'][-1]-a['start'][0]
    n=len(c)-1;breaths=a['breath_counts'][:-1].sum()
    distinct=np.sum(a['parts'][:-1])
    # N cancels: counts/participation imply the population size exactly.
    N=np.sum(a['breath_counts']-a['repeats'])/np.sum(a['parts'])
    repeat=a['repeats'][:-1].sum()/breaths
    rate_interval=N*total_time/breaths
    out['budget_exact_cycle']=distinct/n*rate_interval/(1-repeat)
    out['budget_exact_error']=out['budget_exact_cycle']-total_time/n
    return out

def analyse_run(path, seconds=None, burn=300., alone_mean=85.091):
    path=Path(path)
    with np.load(path) as d:
        seconds=float(d['time'][-1]) if seconds is None else seconds
        a=event_arrays(d,seconds,burn);s=summarize_arrays(a,alone_mean)
        valid=(d['time']>=burn)&(d['time']<seconds)
        h=d['heading'][valid];st=d['status'][valid]
        s['polarization']=np.mean([np.linalg.norm(v[q==0].astype(float).mean(axis=0)) for v,q in zip(h,st) if np.any(q==0)])
    return dict(path=str(path),scalar=s,arrays=a)

def analyse_runs(paths, **kwargs): return [analyse_run(p,**kwargs) for p in paths]

def load_record():
    """One feature row per bout; lag pairs are formed BEFORE any resampling."""
    import pandas as pd
    gap,duration,cls=load_empirical_fig3()
    starts=np.flatnonzero(~np.isfinite(gap));bounds=np.r_[starts,len(gap)]
    assert starts.tolist()==[0,16,133,267,376,517]
    assert np.diff(bounds).tolist()==list(SEGMENT_SIZES)
    segments=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        prev=gap[a:b];nxt=np.r_[prev[1:],np.nan]
        segments.append(np.column_stack([prev,nxt,duration[a:b],cls[a:b]]))
    f2=pd.read_excel(EMP_XLSX,sheet_name='Fig2')
    alone_groups=[g.iloc[:,3].dropna().to_numpy(float) for _,g in f2.groupby('Fish_ID')]
    alone=np.concatenate(alone_groups)
    burst=pd.read_excel(EMP_XLSX,sheet_name='Fig3B-ind_in_shoal')
    burst3=burst[burst['RecNumber']==3]
    assert len(burst3)==10
    burst_span=float(np.ptp(burst3['Air_release_frame'])/30.)
    out=dict(segments=segments,alone=alone,alone_groups=alone_groups,
             burst10_record=burst_span,individual_breath_duration=burst.iloc[:,-1].to_numpy(float),
             source_sha256=digest(EMP_XLSX))
    # Cross-check canonical F13-basis record statistics without running its writers.
    from paper.results_pipeline.analysis import analyse_empirical
    ref=analyse_empirical();s=record_statistics(segments,alone.mean())
    mapping={'gap_mean':'gap_mean','gap_sd':'gap_sd','gap_cv':'gap_cv','gap_median':'gap_median',
             'gap_skew':'gap_skew','duration_mean':'duration_mean','duration_sd':'duration_sd',
             'class_previous_rho':'r_cls_prevgap','class_duration_rho':'r_cls_dur','effect_size':'effect_size'}
    for k,v in mapping.items(): np.testing.assert_allclose(s[k],ref[v],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(s['gap_kurtosis'],ref['gap_exkurt']+3)
    np.testing.assert_allclose([s[f'class_{i}'] for i in range(1,5)],ref['classes'])
    out['f13_crosscheck']=True
    return out

def record_statistics(segments, alone_mean, scale=1.):
    rows=np.concatenate(segments);prev,nxt,dur,cls=rows.T
    prev=prev*scale;nxt=nxt*scale;dur=dur*scale
    s=distribution_stats(prev)
    s.update(duration_mean=dur.mean(),duration_sd=dur.std(ddof=1),
             class_duration_rho=rho(cls,dur),class_next_rho=rho(cls,nxt),
             class_previous_rho=rho(cls,prev),effect_size=alone_mean/s['gap_mean'],
             alone_mean=alone_mean,n_events=len(rows),n_intervals=int(np.isfinite(prev).sum()))
    for i in range(1,5):
        s[f'class_{i}']=np.mean(cls==i)
        for name,vals in [('next',nxt),('previous',prev)]:
            sel=vals[(cls==i)&np.isfinite(vals)]
            s[f'{name}_class_{i}']=sel.mean() if len(sel) else np.nan
    return s

def bootstrap_record(record, repeats=5000, seed=20261009):
    """Overlapping noncircular moving blocks within each original segment.

    Blocks resample complete feature rows, including original paired next/previous
    gaps. No pair is created across bootstrap joins or original segment boundaries.
    Whole-segment bootstrap samples six original segments, with replacement.
    """
    rng=np.random.default_rng(seed);segs=record['segments'];alone_mean=record['alone'].mean()
    point=record_statistics(segs,alone_mean);methods={}
    for method in ['block5','block10','block20','segments']:
        values=[];winners=[];aics=[]
        for b in range(repeats):
            if method=='segments': sampled=[segs[i] for i in rng.integers(0,6,6)]
            else:
                block=int(method[5:]);sampled=[]
                for seg in segs:
                    n=len(seg);L=min(block,n)
                    starts=rng.integers(0,n-L+1,int(np.ceil(n/L)))
                    ix=np.concatenate([np.arange(a,a+L) for a in starts])[:n]
                    sampled.append(seg[ix])
            s=record_statistics(sampled,alone_mean)
            ids=rng.integers(0,len(record['alone_groups']),len(record['alone_groups']))
            alone_boot=np.concatenate([record['alone_groups'][i] for i in ids]).mean()
            s['effect_size_fixed_alone']=s['effect_size']
            s['effect_size_joint']=alone_boot/s['gap_mean'];s['alone_mean_cluster']=alone_boot
            winners.append(s.pop('aic_winner'));aics.append(s.pop('aic'));values.append(s)
        keys=list(values[0]);matrix=np.array([[v[k] for k in keys] for v in values])
        methods[method]=dict(ci={k:np.nanquantile(matrix[:,i],[.025,.975]) for i,k in enumerate(keys)},
                            aic_ci={k:np.quantile([a[k] for a in aics],[.025,.975]) for k in aics[0]},
                            winner_fraction={k:winners.count(k)/repeats for k in point['aic']},
                            replicates=repeats,keys=keys,samples=matrix)
    return dict(point=point,methods=methods,seed=seed)

def aggregate_scalars(runs):
    """Equal-run summaries, with both min/max and empirical quantile ranges."""
    out={}
    for key,value in runs[0]['scalar'].items():
        if value is not None and (not np.isscalar(value) or isinstance(value,str)): continue
        x=np.asarray([r['scalar'][key] for r in runs],float);valid=x[np.isfinite(x)]
        if not len(valid):continue
        out[key]=dict(mean=valid.mean(),sd=valid.std(ddof=1),minimum=valid.min(),maximum=valid.max(),
                      q025=np.quantile(valid,.025),q975=np.quantile(valid,.975),n=len(valid))
    return out

def distribution_yardstick(runs,record_gaps):
    pairs=[];comparisons=[]
    def distances(x,y):
        return dict(ks=float(stats.ks_2samp(x,y,method='asymp').statistic),
                    cvm=float(stats.cramervonmises_2samp(x,y,method='asymptotic').statistic))
    for i,r in enumerate(runs):
        comparisons.append(dict(run=i,**distances(r['arrays']['gaps'],record_gaps)))
        for j in range(i):pairs.append(dict(run1=j,run2=i,**distances(runs[j]['arrays']['gaps'],r['arrays']['gaps'])))
    summary={}
    for metric in ['ks','cvm']:
        x=np.array([v[metric] for v in pairs]);y=np.array([v[metric] for v in comparisons])
        summary[metric]=dict(pair_median=np.median(x),pair_q025=np.quantile(x,.025),pair_q975=np.quantile(x,.975),
                             pair_min=x.min(),pair_max=x.max(),record_mean=y.mean(),record_min=y.min(),record_max=y.max(),
                             above_pair_q975=int(np.sum(y>np.quantile(x,.975))))
    return dict(pairs=pairs,record_comparisons=comparisons,summary=summary)

def shape_segments(runs,record):
    rec=[];model=[]
    for i,seg in enumerate(record['segments']):
        rec.append(dict(segment=i+1,events=len(seg),**distribution_stats(seg[1:,0])))
    for ri,r in enumerate(runs):
        a=r['arrays'];start=0;k=0
        while start<len(a['classes']):
            target=SEGMENT_SIZES[k%6];stop=min(start+target,len(a['classes']))
            gaps=a['gaps'][start:stop-1]
            row=dict(run=ri,chunk=k+1,template_segment=k%6+1,target_events=target,events=stop-start,
                     complete=(stop-start==target),start_event=start,stop_event=stop)
            if len(gaps)>=4:row.update(distribution_stats(gaps))
            model.append(row);start=stop;k+=1
    return dict(record=rec,model=model)

def verify_reference():
    rows=[]
    for seed in (7301,7313,7319):
        stem=f'A_RZ75_s{seed}'; r=json.loads((S25/'outputs/records'/f'{stem}.json').read_text())
        path=S25/'outputs/trajectories'/f'{stem}.npz'
        assert digest(path)==r['sha256']
        a=analyse_run(path); s=a['scalar']; old=r['step25']['scalar']
        mapping={'gap_mean':'gap_mean','gap_sd':'gap_sd','short_fraction':'short_fraction',
                 'long_fraction':'long_fraction','class_1':'small_fraction','class_4':'huge_fraction',
                 'slab_0.5_mean':'slab_mean','individual_mean':'individual_mean',
                 'class_next_rho':'class_next_rho','class_previous_rho':'class_previous_rho',
                 'gap_kurtosis_unbiased':'kurtosis','polarization':'polarization'}
        errors={k:float(s[k]-old[v]) for k,v in mapping.items()}
        assert max(abs(x) for x in errors.values())<1e-10,errors
        rows.append(dict(seed=seed,sha256=digest(path),errors=errors,scalar=s))
    means={k:float(np.mean([r['scalar'][k] for r in rows])) for k in mapping}
    expected=dict(gap_mean=14.647,gap_sd=6.056,short_fraction=.0728,long_fraction=.0214,
                  class_1=.1271,class_4=.0085,**{'slab_0.5_mean':2.323},individual_mean=48.85,class_next_rho=.172)
    for k,v in expected.items(): assert abs(means[k]-v)<(.01 if k=='individual_mean' else .0006),(k,means[k],v)
    result=dict(passed=True,means=means,records=rows)
    write_json(HERE/'qa/reference_reproduction.json',result)
    return result

if __name__=='__main__': print(json.dumps(clean(verify_reference()['means']),indent=2))
