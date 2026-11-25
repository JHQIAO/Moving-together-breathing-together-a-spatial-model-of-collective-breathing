"""Saved-data-only W3 prediction analysis. Replication unit: independent seed."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import json,argparse,warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
sys.path.insert(0,str(HERE.parent/'W1_reference'))
import analysis_core as core
sys.path.insert(0,str(HERE))
from run_simulations import jobs,sha,valid,S25
TYPES=core.TYPES;TARGET=core.ALONE_TARGETS
OUT=HERE/'analysis';OUT.mkdir(exist_ok=True)
def mean(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];return float(x.mean()) if len(x) else np.nan
def sd(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];return float(x.std(ddof=1)) if len(x)>1 else np.nan
def cv(x):return sd(x)/mean(x) if mean(x)>0 else np.nan
def frac(x):return mean(x)
def stretches(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int));a=np.flatnonzero(edges==1);b=np.flatnonzero(edges==-1)
    return b-a,(b-a)[(a>0)&(b<len(mask))]
def canonical(a):
    g=a['gaps'];out={}
    if len(g)>1:
        try:out.update(core.distribution_stats(g))
        except (ValueError,RuntimeError):out.update(gap_mean=mean(g),gap_sd=sd(g),gap_cv=cv(g),short_fraction=mean(g<5),long_fraction=mean(g>30))
    else:out.update(gap_mean=mean(g),gap_sd=sd(g),gap_cv=cv(g),short_fraction=mean(g<5),long_fraction=mean(g>30))
    out.update(n_intervals=len(g),n_events=len(a['parts']),participation_mean=mean(a['parts']),whole_duration_mean=mean(a['durations']),slab_mean=mean(a['slab']['0.5']),individual_mean=mean(a['individual']),individual_sd=sd(a['individual']),class_next_rho=core.rho(a['classes'][:-1],g),class_previous_rho=core.rho(a['classes'][1:],g),cycle_mean=mean(np.diff(a['start'])),cycle_cv=cv(np.diff(a['start'])),peak_mean=mean(a['peak']),within_event_overlap=mean(a['peak']/a['parts']))
    for k in range(1,5):out[f'class_{k}']=mean(a['classes']==k)
    return out

def compute(path,meta,solo):
    name=path.stem;cache=OUT/f'{name}.json';arrpath=OUT/f'{name}_derived.npz';pin=sha(path);analysis_pin=sha(__file__);dependency_pins={str(Path(core.__file__).relative_to(ROOT)):sha(core.__file__),'breathing_study.py':sha(ROOT/'breathing_study.py')}
    if cache.exists() and arrpath.exists():
        old=json.loads(cache.read_text())
        if old.get('source_sha256')==pin and old.get('analysis_sha256')==analysis_pin and old.get('dependency_sha256')==dependency_pins:return old
    with np.load(path) as d:
        T=float(d['time'][-1]);N=len(d['type_id']);a=core.event_arrays(d,T,300,N);s=canonical(a)
        times=d['time'].astype(float);ok=(times>=300)&(times<T);t=times[ok];z=d['pos'][ok,:,2].astype(float);status=d['status'][ok];school=status==0;q=d['tank'][ok].astype(float);heading=d['heading'][ok].astype(float);typ=d['type_id'];counts=np.bincount(typ,minlength=3)
        school_z=np.array([np.median(row[mask]) if mask.any() else np.nan for row,mask in zip(z,school)])
        all_z=np.median(z,axis=1);missing=~np.isfinite(school_z)
        smooth=uniform_filter1d(all_z,3,mode='nearest')
        school_smooth=uniform_filter1d(np.where(missing,all_z,school_z),3,mode='nearest')
        school_crests=find_peaks(school_smooth,prominence=.2)[0]
        school_crests=school_crests[(~missing[school_crests])&(school_crests>=4)&(school_crests<len(t)-4)]
        crest=find_peaks(smooth,prominence=.2)[0];trough=find_peaks(-smooth,prominence=.2)[0]
        crest=crest[(crest>=4)&(crest<len(t)-4)];trough=trough[(trough>=4)&(trough<len(t)-4)]
        crest_gaps=np.diff(t[crest]);dwell,complete=stretches(all_z<1)
        coupling=np.array([np.min(np.abs(a['start']-tt))<=4 if len(a['start']) else False for tt in t[crest]])
        s.update(crest_mean=mean(crest_gaps),crest_cv=cv(crest_gaps),n_crests=len(crest),bottom_dwell_mean=mean(dwell),bottom_dwell_complete_mean=mean(complete),bottom_fraction=mean(all_z<1),height_amplitude=float(np.quantile(all_z,.95)-np.quantile(all_z,.05)),school_only_bottom_fraction=mean(school_z[np.isfinite(school_z)]<1),school_only_crest_mean=mean(np.diff(t[school_crests])),school_only_crest_cv=cv(np.diff(t[school_crests])),school_missing_fraction=mean(missing),crest_event_fraction=mean(coupling),polarization=mean([np.linalg.norm(h[mask].mean(axis=0)) for h,mask in zip(heading,school) if mask.any()]),empty_fish_seconds=float(d['empty_postburn_agent_seconds']),empty_fraction=float(d['empty_postburn_agent_seconds'])/(N*(T-300)))
        bc=a['breath_counts'];den=bc.sum();s['breaths_multifish_fraction']=float(bc[a['parts']*N>=2-1e-9].sum()/den) if den else np.nan;s['breaths_ge20_fraction']=float(bc[a['parts']>=.2].sum()/den) if den else np.nan
        s['rhythm_screen']=bool(s['n_crests']>=10 and s['crest_cv']<=.5 and .05<=s['bottom_fraction']<=.95 and s['height_amplitude']>=.5 and s['crest_event_fraction']>=.5 and s['breaths_multifish_fraction']>=.5)
        for label,cvl,low,couple in [('loose',.6,.02,.4),('strict',.4,.1,.6)]:s[f'rhythm_{label}']=bool(s['n_crests']>=10 and s['crest_cv']<=cvl and low<=s['bottom_fraction']<=1-low and s['height_amplitude']>=.5 and s['crest_event_fraction']>=couple and s['breaths_multifish_fraction']>=.5)
        e=d['episodes13'];e=e[np.argsort(e['release'],kind='stable')];first=np.zeros(3);early=np.zeros(3);particip=np.zeros(3);event_member={};leader_rows=[]
        for idx,(start,end) in enumerate(zip(a['start'],a['end'])):
            ep=e[(e['release']>=start-1e-9)&(e['release']<=end+1e-9)];ids=np.unique(ep['agent']);particip+=np.bincount(typ[ids],minlength=3)
            firstids=np.unique(ep['agent'][np.isclose(ep['release'],ep['release'][0],atol=1e-9,rtol=0)])
            first+=np.bincount(typ[firstids],minlength=3)/len(firstids)
            earlyids=np.unique(ep['agent'][ep['release']<=start+.5]);early+=np.bincount(typ[earlyids],minlength=3)
            leader_rows.append((start,*np.bincount(typ[firstids],minlength=3)))
            for agent in ids:event_member.setdefault(int(agent),[]).append(idx)
        needy=q<.75;school_needy=school&needy;school_needy_counts=np.array([school_needy[:,typ==k].sum() for k in range(3)],float);needy_counts=np.array([needy[:,typ==k].sum() for k in range(3)],float)
        lift_counts=np.array([needy[trough][:,typ==k].sum() for k in range(3)],float)
        event_samples=np.clip(np.searchsorted(t,a['start'],side='right')-1,0,len(t)-1)
        event_need=np.array([needy[event_samples][:,typ==k].sum() for k in range(3)],float)
        # First canonical event after each local trough, before the next trough; no event counted twice.
        lift_event=[]
        for i,tr in enumerate(trough):
            idx=np.searchsorted(a['start'],t[tr]);upper=t[trough[i+1]] if i+1<len(trough) else T-2
            if idx<len(a['start']) and a['start'][idx]<upper:lift_event.append(idx)
        lift_first=np.zeros(3)
        for idx in sorted(set(lift_event)):
            ep=e[np.isclose(e['release'],a['start'][idx],atol=1e-9,rtol=0)];ids=np.unique(ep['agent']);lift_first+=np.bincount(typ[ids],minlength=3)/len(ids)
        derived=dict(gaps=a['gaps'],parts=a['parts'],start=a['start'],end=a['end'],crest_gaps=crest_gaps,school_time=t,school_median=school_z,group_median=all_z,crest_times=t[crest],trough_times=t[trough],bottom_dwell=complete)
        type_rows=[]
        for k,name_t in enumerate(TYPES):
            intervals=a['bytype'][name_t];release_intervals=[];skips=[]
            for i in np.flatnonzero(typ==k):
                ep=e[e['agent']==i];mask=(ep['exit'][:-1]>=300)&(ep['release'][1:]<=T);release_intervals.extend(np.diff(ep['release'])[mask]);skips.extend(np.diff(event_member.get(int(i),[])))
            ri=np.asarray(release_intervals);cycle=s['cycle_mean'];rat=ri/cycle if cycle>0 else ri*np.nan
            tr=dict(type=name_t,n_fish=int(counts[k]),n_intervals=len(intervals),mean=mean(intervals),sd=sd(intervals),cv=cv(intervals),skew=float(stats.skew(intervals)) if len(intervals)>2 else np.nan,kurtosis=float(stats.kurtosis(intervals,fisher=False)) if len(intervals)>3 else np.nan,median=float(np.median(intervals)) if len(intervals) else np.nan,ratio_record=mean(intervals)/TARGET[k],ratio_model_solo=mean(intervals)/solo[name_t]['mean'],event_participation=particip[k]/(counts[k]*len(a['parts'])) if counts[k] and len(a['parts']) else np.nan,needy_share=needy_counts[k]/needy_counts.sum() if needy_counts.sum() else np.nan,school_needy_share=school_needy_counts[k]/school_needy_counts.sum() if school_needy_counts.sum() else np.nan,liftoff_needy_share=lift_counts[k]/lift_counts.sum() if lift_counts.sum() else np.nan,event_needy_share=event_need[k]/event_need.sum() if event_need.sum() else np.nan,first_share=first[k]/first.sum() if first.sum() else np.nan,early_share=early[k]/early.sum() if early.sum() else np.nan,liftoff_first_share=lift_first[k]/lift_first.sum() if lift_first.sum() else np.nan,integer_fraction=mean(np.abs(rat-np.rint(rat))<=.1) if np.isfinite(rat).any() else np.nan,phase_resultant=float(abs(np.mean(np.exp(2j*np.pi*rat)))) if len(rat) else np.nan,event_skip_mean=mean(skips),release_cycle_ratio=mean(rat))
            tr['needy_enrichment']=tr['needy_share']/(counts[k]/N) if counts[k] else np.nan;tr['first_enrichment']=tr['first_share']/(counts[k]/N) if counts[k] else np.nan
            for key,val in tr.items():
                if isinstance(val,(float,int,np.floating,np.integer)):s[f'{name_t}_{key}']=val
            type_rows.append(tr);derived[f'individual_{name_t}']=intervals;derived[f'release_intervals_{name_t}']=ri;derived[f'event_skips_{name_t}']=np.asarray(skips)
        # Raw event intervals preserve all tails; no truncation for display or summaries.
        np.savez_compressed(arrpath,**derived)
    result=dict(name=name,source=str(path.relative_to(ROOT)),source_sha256=pin,analysis_sha256=analysis_pin,dependency_sha256=dependency_pins,meta=meta,scalar=s,types=type_rows,counts=counts.tolist(),seconds=T)
    core.write_json(cache,result);return result

def solo_data():
    solo=json.loads((S25/'outputs/b_RZ75_solo_final.json').read_text());derived={};rows=[]
    for k,name in enumerate(TYPES):
        means=[]
        for stem in solo[name]['records']:
            rp=S25/'outputs/records'/f'{stem}.json';r=json.loads(rp.read_text());p=S25/'outputs/trajectories'/f'{stem}.npz';assert sha(p)==r['sha256']
            with np.load(p) as d:
                ep=d['episodes13'];ep=ep[np.argsort(ep['release'])];g=(ep['release'][1:]-ep['exit'][:-1])[ep['exit'][:-1]>=300]
                derived[stem]=g;means.append(mean(g));rows.append(dict(type=name,name=stem,mean=mean(g),cv=cv(g),n=len(g),sha256=sha(p)))
        np.testing.assert_allclose(mean(means),solo[name]['mean'],rtol=1e-10)
    np.savez_compressed(OUT/'solo_derived.npz',**derived);core.write_json(OUT/'solo_reference.json',dict(summary=solo,runs=rows));return solo

def aggregate(rows):
    groups={}
    for r in rows:groups.setdefault(r['meta']['arm'],[]).append(r)
    result={}
    for arm,rs in groups.items():
        scalars={}
        for key in sorted(set().union(*(r['scalar'].keys() for r in rs))):
            vs=[r['scalar'].get(key) for r in rs];vs=np.asarray([v for v in vs if isinstance(v,(int,float,bool)) and np.isfinite(v)],float)
            if len(vs):scalars[key]=dict(mean=mean(vs),min=float(vs.min()),max=float(vs.max()),sd=sd(vs),n=len(vs))
        result[arm]=dict(meta=rs[0]['meta'],n=len(rs),counts=rs[0]['counts'],metrics=scalars,seeds=[r['meta']['seed'] for r in rs])
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    core.HERE=HERE;core.verify_reference();solo=solo_data();rows=[]
    for j in jobs():
        if (HERE/'records'/f"{j['name']}.json").exists():
            assert valid(j);rows.append(compute(HERE/'trajectories'/f"{j['name']}.npz",j,solo));print('ANALYSED',j['name'],flush=True)
        elif not args.partial:raise RuntimeError(f"Missing {j['name']}")
    w1=[]
    if (HERE.parent/'W1_reference/REPORT.md').exists():
        for seed in range(9001,9011):
            path=HERE.parent/'W1_reference/trajectories'/f'reference_s{seed}.npz';record=json.loads((path.parent.parent/'records'/f'{path.stem}.json').read_text());assert sha(path)==record['sha256']
            w1.append(compute(path,dict(arm='W1_N200',protocol='reference',N=200,seed=seed),solo));print('ANALYSED',path.stem,flush=True)
    allrows=rows+w1;groups=aggregate(allrows)
    pooled=[dict(r,meta=dict(r['meta'],arm='combined_N200')) for r in allrows if r['meta']['arm'] in ('same_N200','W1_N200')]
    groups.update(aggregate(pooled))
    # Paired contrasts where the requested seed sets intersect.
    contrasts=[]
    for n in (50,100,400):
        for seed in range(9201,9204):
            aa=[r for r in rows if r['meta']['arm']==f'same_N{n}' and r['meta']['seed']==seed];bb=[r for r in rows if r['meta']['arm']==f'density_N{n}' and r['meta']['seed']==seed]
            if aa and bb:
                for key in ('gap_mean','participation_mean','individual_mean','crest_cv','polarization','bottom_fraction'):contrasts.append(dict(N=n,seed=seed,metric=key,delta_density_minus_same=bb[0]['scalar'][key]-aa[0]['scalar'][key]))
    pd.DataFrame(contrasts).to_csv(OUT/'density_paired.csv',index=False)
    flat=[];types=[]
    for r in allrows:
        flat.append(dict(name=r['name'],arm=r['meta']['arm'],seed=r['meta']['seed'],N=r['meta']['N'],**{k:v for k,v in r['scalar'].items() if not isinstance(v,dict)}))
        types.extend(dict(name=r['name'],arm=r['meta']['arm'],seed=r['meta']['seed'],**t) for t in r['types'])
    pd.DataFrame(flat).to_csv(OUT/'run_metrics.csv',index=False);pd.DataFrame(types).to_csv(OUT/'type_metrics.csv',index=False)
    core.write_json(OUT/'groups.json',groups)
    rec=core.load_record();record_stats=core.record_statistics(rec['segments'],rec['alone'].mean());unc=json.loads((HERE.parent/'W1_reference/analysis/record_uncertainty.json').read_text())
    core.write_json(OUT/'record_reference.json',dict(point=record_stats,uncertainty=unc,source_sha256=rec['source_sha256'],scenario_note='Record shoal times may require a factor 1.5; model/alone times unchanged.'))
    np.savez_compressed(OUT/'record_derived.npz',gaps=np.concatenate([x[1:,0] for x in rec['segments']]),classes=np.concatenate([x[:,3] for x in rec['segments']]))
    core.write_json(OUT/'completion.json',dict(w3_complete=len(rows),w3_expected=len(jobs()),w1_included=len(w1),partial=len(rows)!=len(jobs()),analysis_sha256=sha(__file__)))

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);main()
