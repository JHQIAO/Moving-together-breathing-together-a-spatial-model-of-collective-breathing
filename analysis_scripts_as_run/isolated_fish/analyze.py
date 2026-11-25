"""Reproducible W8 analyses; reads other workstreams, writes only HERE."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,argparse,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(BASE/'W1_reference'))
from analysis_core import load_record,distribution_stats,fit_small,rho,class_codes,bout_events_from,digest,write_json,clean,TYPES,SEGMENT_SIZES
sys.path.insert(0,str(BASE/'W5a_calibration'))
from calibration_core import fish_intervals
A=HERE/'analysis';QA=HERE/'qa';B=2000;SEED=20261011
FAMILIES={'gamma':(stats.gamma,{'floc':0}),'lognorm':(stats.lognorm,{'floc':0}),'norm':(stats.norm,{}),'laplace':(stats.laplace,{}),'expon':(stats.expon,{})}
def read(p):return json.loads(Path(p).read_text())
def save(n,x):write_json(A/(n+'.json'),x)
def dist(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];s=distribution_stats(x)
    s.update(n=len(x),gamma_shape=stats.gamma.fit(x,floc=0)[0],ranking=sorted(s['aic'],key=s['aic'].get),fit_params={k:list(d.fit(x,**kw)) for k,(d,kw) in FAMILIES.items()})
    return s
def agg(rows):
    out={}
    for k,v in rows[0].items():
        if isinstance(v,(int,float,np.number)):
            x=np.array([r[k] for r in rows],float);x=x[np.isfinite(x)]
            if len(x):out[k]=dict(mean=x.mean(),sd=x.std(ddof=1) if len(x)>1 else 0.,min=x.min(),max=x.max(),q025=np.quantile(x,.025),q975=np.quantile(x,.975),n=len(x))
    if 'aic_winner' in rows[0]:out['winner_fraction']={k:np.mean([r['aic_winner']==k for r in rows]) for k in FAMILIES}
    return out
def cis(rows):
    out={k:np.nanquantile([r[k] for r in rows],[.025,.975]) for k,v in rows[0].items() if isinstance(v,(int,float,np.number))}
    if 'aic' in rows[0]:out['aic']={k:np.quantile([r['aic'][k] for r in rows],[.025,.975]) for k in FAMILIES}
    return out
def blocks(seg,rng,L=10):
    n=len(seg);L=min(n,L);start=rng.integers(0,n-L+1,int(np.ceil(n/L)))
    return seg[np.concatenate([np.arange(s,s+L) for s in start])[:n]]
def corr(x,y,rank=True):
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
    if rank:x=stats.rankdata(x);y=stats.rankdata(y)
    return float(np.corrcoef(x,y)[0,1])
def detrend(x,mode):
    x=np.asarray(x,float)
    if mode=='raw':return x.copy()
    if mode=='center':return x-np.nanmean(x)
    return x-pd.Series(x).rolling(int(mode[3:]),center=True,min_periods=1).median().to_numpy()
def pair_rows(xs,maxlag=5):
    # Keep original row counts, including NaNs, so block10 matches W1 sampling.
    out=[]
    for x in xs:
        ar=np.full((len(x),1+maxlag),np.nan);ar[:,0]=x
        for l in range(1,maxlag+1):ar[:-l,l]=x[l:]
        out.append(ar)
    return out
def acf(rows,maxlag=5):
    ar=np.concatenate(rows);s={}
    for l in range(1,maxlag+1):
        s[f'spearman_{l}']=corr(ar[:,0],ar[:,l]);s[f'pearson_{l}']=corr(ar[:,0],ar[:,l],False)
        s[f'pairs_{l}']=int(np.sum(np.isfinite(ar[:,0])&np.isfinite(ar[:,l])))
    if maxlag>=2:s['spearman_lag2_minus_lag1']=s['spearman_2']-s['spearman_1']
    return s
def serial_record(record):
    rng=np.random.default_rng(SEED+2);out={}
    for mode in ['raw','center','med11','med21','classes']:
        xs=[s[:,3] if mode=='classes' else detrend(s[:,0],mode) for s in record['segments']];ml=3 if mode=='classes' else 5
        pairs=pair_rows(xs,ml);point=acf(pairs,ml);boot=[acf([blocks(s,rng) for s in pairs],ml) for _ in range(B)]
        out[mode]=dict(point=point,ci=cis(boot),replicates=B)
        print('Record serial',mode,point,flush=True)
    save('g2_record',out);return out
def load_w1():
    out=[]
    for seed in range(9001,9011):
        rp=BASE/f'W1_reference/records/reference_s{seed}.json';r=read(rp);ap=BASE/f'W1_reference/analysis/reference_s{seed}_derived.npz';ar=read(ap.with_name(ap.name.replace('_derived.npz','.json')))
        assert digest(ap)==ar['derived_sha256'] and ar['input_sha256']==r['sha256']
        with np.load(ap) as d:a={k:d[k] for k in d.files}
        out.append(dict(seed=seed,record=r,arrays=a,path=BASE/f'W1_reference/trajectories/reference_s{seed}.npz'))
    return out

def record_alone():
    f=pd.read_excel(ROOT/'arapaima-air-breathing/data/supplementary/Supplementary_Data_1.xlsx',sheet_name='Fig2');col=f.columns[3]
    # Rank-paired type assignment: fish 1/6 fast, 2/5 medium, 3/4 slow.
    mapping={1:0,6:0,2:1,5:1,3:2,4:2};units=[]
    for (t,fish),g in f.groupby(['trial','Fish_ID']):
        x=pd.to_numeric(g[col],errors='coerce').dropna().to_numpy(float)
        units.append(dict(trial=int(t),fish=int(fish),type=mapping[int(fish)],values=x,n=len(x),mean=x.mean()))
    assert len(units)==12 and sum(u['n'] for u in units)==213
    x=np.concatenate([u['values']/u['mean'] for u in units]);point=dist(x);rng=np.random.default_rng(SEED+1);bs=[]
    for _ in range(B):
        chosen=rng.integers(1,7,6);vals=np.concatenate([u['values']/u['mean'] for fish in chosen for u in units if u['fish']==fish]);bs.append(dist(vals))
    result=dict(point=point,ci=cis(bs),winner_fraction=agg(bs)['winner_fraction'],units=units,mapping=mapping,bootstrap=B)
    save('g1_record_alone',result);return result

def normalized_shoal(record,runs):
    rg=np.concatenate([s[:,0] for s in record['segments']]);rg=rg[np.isfinite(rg)];point=dist(rg/rg.mean());rng=np.random.default_rng(SEED+3);bs=[]
    for _ in range(B):
        g=np.concatenate([blocks(s,rng)[:,0] for s in record['segments']]);g=g[np.isfinite(g)];bs.append(dist(g/g.mean()))
    coll=[];ind=[];types={t:[] for t in TYPES}
    for r in runs:
        a=r['arrays'];g=a['gaps'];coll.append(dict(seed=r['seed'],**dist(g/g.mean())))
        vals=[]
        for t in TYPES:
            x=a['bytype__'+t];x=x/x.mean();vals.append(x);types[t].append(dict(seed=r['seed'],**dist(x)))
        assert sum(len(v) for v in vals)==len(a['individual'])
        ind.append(dict(seed=r['seed'],**dist(np.concatenate(vals))))
    out=dict(record=dict(point=point,ci=cis(bs),winner_fraction=agg(bs)['winner_fraction']),collective=dict(runs=coll,summary=agg(coll)),individual=dict(runs=ind,summary=agg(ind)),bytype={t:dict(runs=v,summary=agg(v)) for t,v in types.items()})
    save('g1_shoal',out);return out

def analyze_solo(alone):
    units=[];pins=[]
    for s in range(9601,9611):
        for t in range(3):
            n=f'solo_t{t}_s{s}';r=read(HRE:=HERE/'records'/f'{n}.json');p=HERE/'trajectories'/f'{n}.npz';assert digest(p)==r['sha256']
            with np.load(p) as d:
                assert d['episodes13'].dtype['release']==np.dtype('float64')
                x=fish_intervals(d['episodes13'],1,7200.,300.)[0]
            units.append(dict(seed=s,type=t,n=len(x),mean=x.mean(),values=x,stats=dist(x/x.mean())))
            pins.append(dict(name=n,sha256=r['sha256'],params=r['params'],elapsed_s=r['elapsed_s'],audit=r['audit']))
    full=dist(np.concatenate([u['values']/u['mean'] for u in units]));rng=np.random.default_rng(SEED+4);pseudo=[];draws=[]
    bootstrap=[]
    for _ in range(B):
        seeds=rng.integers(9601,9611,10)
        x=np.concatenate([u['values']/u['mean'] for seed in seeds for u in units if u['seed']==seed])
        bootstrap.append(dist(x))
    for b in range(B):
        xs=[];rr=[]
        for rec in alone['units']:
            choices=[u for u in units if u['type']==rec['type']];u=choices[int(rng.integers(len(choices)))];n=rec['n'];assert u['n']>=n
            start=int(rng.integers(0,u['n']-n+1));x=u['values'][start:start+n];xs.append(x/x.mean());rr.append([u['seed'],u['type'],start,n])
        s=dist(np.concatenate(xs));pseudo.append(s);draws.append(rr)
    out=dict(units=units,pooled=full,pooled_ci=cis(bootstrap),bootstrap_method='Resample the 10 seed triplets, retaining all three types; 2000 replicates.',run_summary=agg([u['stats'] for u in units]),bytype={t:dict(summary=agg([u['stats'] for u in units if u['type']==i]),pooled=dist(np.concatenate([u['values']/u['mean'] for u in units if u['type']==i]))) for i,t in enumerate(TYPES)},matched=dict(replicates=B,summary=agg(pseudo),delta=[s['laplace_gamma'] for s in pseudo],aics=[s['aic'] for s in pseudo],winner_fraction=agg(pseudo)['winner_fraction'],record_delta=alone['point']['laplace_gamma'],fraction_delta_le_record=np.mean([s['laplace_gamma']<=alone['point']['laplace_gamma'] for s in pseudo])),records=pins)
    np.savez_compressed(A/'matched_solo_draws.npz',draws=np.asarray(draws))
    save('g1_model_alone',out);return out

def serial_model(runs):
    modes={};chunks=[];pools={i:[] for i in range(6)}
    for mode in ['raw','center','med11','med21','classes']:
        rows=[]
        for r in runs:
            x=r['arrays']['classes'] if mode=='classes' else detrend(r['arrays']['gaps'],mode);ml=3 if mode=='classes' else 5
            rows.append(dict(seed=r['seed'],**acf(pair_rows([x],ml),ml)))
        modes[mode]=dict(runs=rows,summary=agg(rows))
    for r in runs:
        a=r['arrays'];start=0;k=0
        while start<len(a['classes']):
            idx=k%6;n=SEGMENT_SIZES[idx];stop=min(start+n,len(a['classes']));x=a['gaps'][start:stop-1]
            row=dict(seed=r['seed'],chunk=k+1,template=idx+1,start=start,stop=stop,complete=(stop-start==n),**acf(pair_rows([x])))
            chunks.append(row)
            if row['complete']:pools[idx].append(x)
            start=stop;k+=1
    rng=np.random.default_rng(SEED+5);matched=[]
    assert all(pools.values()),{k:len(v) for k,v in pools.items()}
    for _ in range(B):
        xs=[pools[i][rng.integers(len(pools[i]))] for i in range(6)];matched.append(acf(pair_rows(xs)))
    out=dict(modes=modes,chunks=chunks,complete_chunk_summary=agg([r for r in chunks if r['complete']]),matched=dict(summary=agg(matched),rows=matched,replicates=B,pool_counts={k:len(v) for k,v in pools.items()},method='Six complete contiguous W1 chunks, one per segment-size template, independently sampled from all seeds; no boundary pairs.'))
    save('g2_model',out);return out

def events(e,T,N=200,gap=1.,floor=1,w=3.3,visible=1):
    e=e[np.argsort(e['release'],kind='stable')];b=bout_events_from(e['release'],e['exit'],e['agent'],gap=gap);kept=[]
    for v in b:
        if v['t_start']<300 or v['t_end']>T-2 or v['n']<floor:continue
        ix=np.searchsorted(e['release'],[v['t_start']-1e-10,v['t_end']+1e-10],side='left');ep=e[ix[0]:ix[1]]
        assert len(ep)==v['n_breaths']
        if np.sum(ep['release_y']<=w)<visible:continue
        kept.append(v)
    start=np.array([v['t_start'] for v in kept]);end=np.array([v['t_end'] for v in kept]);part=np.array([v['n']/N for v in kept]);c=class_codes(part)
    return dict(gaps=start[1:]-end[:-1],classes=c,parts=part,start=start,end=end)
def metrics(a,T):
    s=distribution_stats(a['gaps']);s.update(class_next_rho=rho(a['classes'][:-1],a['gaps']),class_previous_rho=rho(a['classes'][1:],a['gaps']),events_per_hour=len(a['classes'])*3600/(T-302),n_events=len(a['classes']))
    s.update({f'class_{i}':np.mean(a['classes']==i) for i in range(1,5)})
    s.update({f'lag{i}':corr(a['gaps'][:-i],a['gaps'][i:]) for i in [1,2]})
    return s
def settings_grid():
    specs=[]
    for gap in [.5,.75,1.,1.5,2.]:specs.append(dict(name=f'gap_{gap:g}',family='merge',gap=gap))
    for m in [1,2,3,5,10,20]:specs.append(dict(name=f'floor_{m}',family='floor',floor=m))
    for v in [1,3]:
        for w in [.5,1.,1.65,3.3]:specs.append(dict(name=f'visible{v}_{w:g}',family=f'visible{v}',w=w,visible=v))
    return specs

def paths_extra(arm):
    if arm=='A1':return [(s,BASE/f'W2_mechanism/outputs/records/A1_s{s}.json',BASE/f'W2_mechanism/outputs/trajectories/A1_s{s}.npz') for s in range(9001,9004)]
    return [(s,BASE/f'W7_reselection/posthoc_k020_q078/records/k020_q078_posthoc_s{s}.json',BASE/f'W7_reselection/posthoc_k020_q078/trajectories/k020_q078_posthoc_s{s}.npz') for s in range(9501,9506)]

def mechanism(a,t,pos):
    g=a['gaps'];c=a['classes'];q=np.quantile(g[:-1],[1/3,2/3]);terc=np.digitize(g[:-1],q);res={}
    for j in range(3):
        x=g[1:][terc==j];res[f'next_given_tercile{j+1}']=x.mean();res[f'n_tercile{j+1}']=len(x)
    for j in range(1,5):
        x=g[1:][c[:-2]==j];res[f'next_given_class{j}']=x.mean() if len(x) else np.nan;res[f'n_class{j}']=len(x)
    z=np.median(pos[:,:,2],axis=1);rest=[]
    for lo,hi in zip(a['end'][:-1],a['start'][1:]):
        # Piecewise-constant 1 s trajectory intervals, clipped to exact event gap.
        weight=np.maximum(0,np.minimum(t[:-1]+1.,hi)-np.maximum(t[:-1],lo));rest.append(np.sum(weight*(z[:-1]<1.)))
    rest=np.array(rest);res.update(gap_rest_rho=rho(g,rest),rest_next_rho=rho(rest[:-1],g[1:]),rest_mean=rest.mean(),next_high_minus_low=res['next_given_tercile3']-res['next_given_tercile1'])
    return res

def grid_and_contrasts(runs,record,g2r):
    rec=read(BASE/'W1_reference/analysis/record_uncertainty.json');ci=rec['methods']['block10']['ci'];point=rec['point'];ci=dict(ci);point=dict(point)
    for i in [1,2]:ci[f'lag{i}']=g2r['raw']['ci'][f'spearman_{i}'];point[f'lag{i}']=g2r['raw']['point'][f'spearman_{i}']
    def rate(segs):
        rows=np.concatenate(segs);return len(rows)*3600/(np.nansum(rows[:,0])+np.sum(rows[:,2]))
    point['events_per_hour']=rate(record['segments']);rng=np.random.default_rng(SEED+6)
    ci['events_per_hour']=np.quantile([rate([blocks(s,rng) for s in record['segments']]) for _ in range(B)],[.025,.975])
    grid=[];contrast={};hashes=[]
    for arm in ['RZ75','A1','k020_q078']:
        paths=[(r['seed'],BASE/f'W1_reference/records/reference_s{r["seed"]}.json',r['path']) for r in runs] if arm=='RZ75' else paths_extra(arm)
        mrows=[];acrows=[]
        for seed,rp,tp in paths:
            r=read(rp);assert digest(tp)==r['sha256'];T=r.get('seconds',r['job']['seconds']);hashes.append(dict(arm=arm,seed=seed,npz=str(tp.relative_to(ROOT)),sha256=r['sha256'],seconds=T,params=r['params']))
            with np.load(tp) as d:
                e=d['episodes13'];a=events(e,T)
                if arm=='RZ75':
                    expected=next(x for x in runs if x['seed']==seed)['arrays'];np.testing.assert_allclose(a['gaps'],expected['gaps'],atol=1e-10,rtol=0);np.testing.assert_array_equal(a['classes'],expected['classes'])
                mrows.append(dict(seed=seed,**mechanism(a,d['time'].astype(float),d['pos'])))
                acrows.append(dict(seed=seed,**acf(pair_rows([a['gaps']]))))
                if arm!='A1':
                    for spec in settings_grid():
                        aa=events(e,T,**{k:v for k,v in spec.items() if k not in ['name','family']});ss=metrics(aa,T);grid.append(dict(arm=arm,seed=seed,setting=spec['name'],**ss))
            print('Grid/contrast complete',arm,seed,flush=True)
        contrast[arm]=dict(serial=dict(runs=acrows,summary=agg(acrows)),conditioning=dict(runs=mrows,summary=agg(mrows)))
    summarized=[]
    for arm in ['RZ75','k020_q078']:
        for spec in settings_grid():
            rows=[g for g in grid if g['arm']==arm and g['setting']==spec['name']];sm=agg(rows)
            flags={k:dict(mean_inside=bool(lo<=sm[k]['mean']<=hi),seeds_inside=sum(lo<=r[k]<=hi for r in rows),direction='below' if sm[k]['mean']<lo else 'above' if sm[k]['mean']>hi else 'inside') for k,(lo,hi) in ci.items() if k in sm}
            summarized.append(dict(arm=arm,setting=spec,summary=sm,flags=flags))
    out=dict(record=dict(point=point,ci=ci,events_per_hour_note='Reconstructed annotated occupancy span: bouts / (sum within-segment gaps + durations); not wall-clock recording rate.'),rows=grid,settings=summarized,contrasts=contrast,inputs=hashes)
    # Robustness and comparison claims computed, not hand-selected.
    robust={};comparison={}
    for arm in ['RZ75','k020_q078']:
        rows=[r for r in summarized if r['arm']==arm];keys=set.intersection(*[set(r['flags']) for r in rows]);rr={}
        for k in sorted(keys):
            dirs=[r['flags'][k]['direction'] for r in rows]
            rr[k]=dict(all_means_outside=all(v!='inside' for v in dirs),all_seed_values_outside=all(r['flags'][k]['seeds_inside']==0 for r in rows),direction=dirs[0] if len(set(dirs))==1 else 'mixed',inside_settings=[r['setting']['name'] for r in rows if r['flags'][k]['mean_inside']])
        robust[arm]=rr
    for k in point:
        if k not in summarized[0]['summary']:continue
        pairs=[]
        for spec in settings_grid():
            rz=next(r for r in summarized if r['arm']=='RZ75' and r['setting']['name']==spec['name'])['summary'][k]['mean'];kk=next(r for r in summarized if r['arm']=='k020_q078' and r['setting']['name']==spec['name'])['summary'][k]['mean']
            pairs.append(dict(setting=spec['name'],rz75=rz,k020_q078=kk,difference_k_minus_rz=kk-rz,closer='k020_q078' if abs(kk-point[k])<abs(rz-point[k]) else 'RZ75'))
        comparison[k]=pairs
    out.update(robustness=robust,comparison=comparison);save('g3_g4',out);return out

def main():
    args=argparse.ArgumentParser();args.add_argument('--phase',choices=['saved','solo','assemble'],required=True);arg=args.parse_args()
    if arg.phase=='saved':
        record=load_record();runs=load_w1();record_alone();normalized_shoal(record,runs);g2r=serial_record(record);serial_model(runs);grid_and_contrasts(runs,record,g2r)
    elif arg.phase=='solo':analyze_solo(read(A/'g1_record_alone.json'))
    else:
        names=['g1_record_alone','g1_model_alone','g1_shoal','g2_record','g2_model','g3_g4','supplement'];summary={n:read(A/(n+'.json')) for n in names}
        summary['protocol']=dict(bootstrap_replicates=B,seed=SEED,preflight=read(QA/'preflight.json'),run_state=read(HERE/'run_state.json'),analysis_sha256=digest(__file__),caveats=['All family rankings descriptive MLE AIC, not goodness-of-fit tests.','Alone fish type mapping: pooled-mean rank pairs 1/6 fast, 2/5 medium, 3/4 slow.','Running medians use truncated edges; lag pairs formed before within-segment block bootstrap.','G3 visibility retains whole-event boundaries and total-school participant classes.','Record frame rate and annotation rule unknown; only model definitions varied.'])
        save('summary',summary)
if __name__=='__main__':main()
