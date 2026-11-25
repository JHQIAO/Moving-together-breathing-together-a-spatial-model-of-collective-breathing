"""Build the new SI evidence from saved data; never changes original results."""
from pathlib import Path
import os,sys,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BASE=ROOT/'study_out/rz75_paper'
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy import stats
from breathing_study import load_empirical_fig3,bout_events_from
DATA=HERE/'source_data';QA=HERE/'qa'
for p in [DATA,QA,HERE/'figures',HERE/'tables'] :p.mkdir(exist_ok=True)
INPUTS={}
def source(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    INPUTS[str(p.relative_to(ROOT))]=h.hexdigest();return p
def js(p):return json.loads(source(p).read_text())
def csv(p):return pd.read_csv(source(p))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def agg(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(mean=np.mean(x) if len(x) else None,sd=np.std(x,ddof=1) if len(x)>1 else None,n=len(x))
FAMILIES={'Laplace':(stats.laplace,{}),'Gamma':(stats.gamma,{'floc':0}),'Normal':(stats.norm,{}),'Lognormal':(stats.lognorm,{'floc':0}),'Exponential':(stats.expon,{})}
def fits(g,label):
    rows=[]
    for name,(dist,kw) in FAMILIES.items():
        prm=dist.fit(g,**kw);k=len(prm)-len(kw);assert k==2
        rows.append(dict(dataset=label,n=len(g),family=name,aic=2*k-2*dist.logpdf(g,*prm).sum(),shape=prm[0] if len(prm)==3 else None,location=prm[-2],scale=prm[-1]))
    best=min(r['aic'] for r in rows)
    for r in rows:r['delta_aic']=r['aic']-best
    return rows
def event_metrics(path,N,T):
    with np.load(source(path)) as z:ep=np.sort(z['episodes13'],order='release',kind='stable')
    b=bout_events_from(ep['release'],ep['exit'],ep['agent'],gap=1)
    b=[v for v in b if v['t_start']>=300 and v['t_end']<=T-2]
    starts=np.array([v['t_start'] for v in b]);ends=np.array([v['t_end'] for v in b]);parts=np.array([v['n']/N for v in b]);cls=np.digitize(parts,[.2,.5,.8])+1;g=starts[1:]-ends[:-1]
    m=dict(gap_mean=np.mean(g),gap_sd=np.std(g,ddof=1),short_fraction=np.mean(g<5),long_fraction=np.mean(g>30),participation_mean=parts.mean(),
        class_previous_rho=stats.spearmanr(cls[1:],g).statistic,class_next_rho=stats.spearmanr(cls[:-1],g).statistic,
        aic_winner=min(fits(g,'variant'),key=lambda r:r['aic'])['family'])
    for i in range(1,5):m[f'class_{i}']=np.mean(cls==i)
    return m
def solo_values(path,T=3600):
    with np.load(source(path)) as z:ep=np.sort(z['episodes13'],order='release')
    ok=(ep['exit'][:-1]>=300)&(ep['release'][1:]<=T)
    g=(ep['release'][1:]-ep['exit'][:-1])[ok];return g/g.mean()
def shape_ci(units):
    seeds=sorted(set(s for s,t,g in units));pool=np.concatenate([g for s,t,g in units])
    def shape(x):return np.array([np.std(x,ddof=1)/np.mean(x),stats.skew(x)])
    rng=np.random.default_rng(20261109);boot=[]
    for _ in range(2000):
        chosen=rng.choice(seeds,len(seeds),replace=True)
        boot.append(shape(np.concatenate([g for s in chosen for ss,t,g in units if s==ss])))
    return dict(point=shape(pool),ci=np.quantile(boot,[.025,.975],axis=0).T,n_seeds=len(seeds),n_runs=len(units))

def main():
    write(QA/'main_hash_before.json',{'main.tex':hashlib.sha256((ROOT/'paper_draft/main.tex').read_bytes()).hexdigest()})
    rec=js(BASE/'W1_reference/analysis/record_uncertainty.json')
    full=csv(BASE/'W1_reference/analysis/model_replicates.csv').sort_values('seed')
    w2=csv(BASE/'W2_mechanism/analysis/per_run_metrics.csv')
    w4=csv(BASE/'W4_robustness/analysis/metrics_by_seed.csv')
    source(ROOT/'arapaima-air-breathing/data/supplementary/Supplementary_Data_1.xlsx')
    g,dur,cls=load_empirical_fig3();ok=np.isfinite(g);nextok=np.flatnonzero(ok)
    rprev=stats.spearmanr(cls[ok],g[ok]);rnext=stats.spearmanr(cls[nextok-1],g[nextok])
    np.testing.assert_allclose([rprev.statistic,rnext.statistic],[rec['point']['class_previous_rho'],rec['point']['class_next_rho']],atol=1e-12)
    out=dict(record=rec,reference_summary={k:agg(full[k]) for k in full.select_dtypes('number') if k!='seed'},
        empirical_correlations=dict(previous=dict(rho=rprev.statistic,p=rprev.pvalue,n=int(ok.sum())),
            next=dict(rho=rnext.statistic,p=rnext.pvalue,n=int(ok.sum())),published=dict(rho=.09,p=.021,n=612)),
        numerical={},ablations={},variants={},depth=[],groups=[])
    fitrows=fits(g[ok],'Record')
    for seed in range(9001,9011):
        with np.load(source(BASE/f'W1_reference/analysis/reference_s{seed}_derived.npz')) as z:fitrows.extend(fits(z['gaps'],str(seed)))
    for row in fitrows:
        if row['dataset']!='Record':
            key={'Normal':'norm','Lognormal':'lognorm','Exponential':'expon'}.get(row['family'],row['family'].lower())
            expected=full.loc[full.seed==int(row['dataset']),f'aic_{key}'].iloc[0]
            np.testing.assert_allclose(row['aic'],expected,atol=1e-7,rtol=0)
    pd.DataFrame(fitrows).to_csv(DATA/'S4_distribution_fits.csv',index=False)
    for arm in ['control','half_dt','reaction_010']:
        df=w4[(w4.phase=='numerical')&(w4.arm==arm)&(w4.burn==300)]
        assert len(df)==3
        out['numerical'][arm]={k:agg(df[k]) for k in ['gap_mean','participation_mean','small_fraction','medium_fraction','large_fraction','huge_fraction','long_fraction']}
    ndf=w4[(w4.phase=='numerical')&(w4.burn==300)];ndf.to_csv(DATA/'S5_numerical_runs.csv',index=False)
    for arm in ['A0','A1','A2','A8','A6','A3','A4','A5','A7','A9','A10']:
        df=w2[(w2.arm==arm)&(w2.depth==3.3)&w2['type'].isna()]
        assert len(df)==3
        # Descriptive post-hoc rule makes the existing Fig.4 hollow-symbol flag reproducible;
        # it is not a spectral or significance test of rhythm absence.
        flags=(df.polarization<.2)&(df.bottom_fraction>=.95)
        out['ablations'][arm]=dict(n=3,seconds=df.seconds.tolist(),seeds=df.seed.tolist(),
            no_clear_rhythm_runs=int(flags.sum()),statistics={k:agg(df[k]) for k in ['gap_mean','small_fraction','individual_mean','participation_mean','class_next_rho','polarization','bottom_fraction','surface_rise_rate_min']})
    assert out['ablations']['A8']['no_clear_rhythm_runs']==3
    assert all(v['no_clear_rhythm_runs']==0 for k,v in out['ablations'].items() if k!='A8')
    out['variants']['Full model']=dict(n=10,statistics={k:agg(full[k]) for k in ['gap_mean','gap_sd','short_fraction','long_fraction','participation_mean','class_previous_rho','class_next_rho','class_1','class_2','class_3','class_4']},aic_winners=full.aic_winner.tolist())
    for label,df,folder in [
        ('No rest zone',w2[(w2.arm=='A1')&(w2.depth==3.3)&w2['type'].isna()],BASE/'W2_mechanism/outputs'),
        ('Weaker rest',w4[(w4.arm=='kappa_bottom_low')&(w4.phase=='tier2')&(w4.burn==300)],BASE/'W4_robustness/outputs')]:
        rows=[]
        for _,row in df.iterrows():
            m=event_metrics(folder/'trajectories'/f'{row["name"]}.npz',200,row.seconds)
            for k in ['gap_mean','class_next_rho','class_previous_rho']:np.testing.assert_allclose(m[k],row[k],atol=1e-12)
            rows.append(m)
        out['variants'][label]=dict(n=len(rows),statistics={k:agg([r[k] for r in rows]) for k in rows[0] if k!='aic_winner'},aic_winners=[r['aic_winner'] for r in rows])
    for arm in ['A0','A1']:
        for depth in [2.2,3.3,5.,6.6]:
            df=w2[(w2.arm==arm)&(w2.depth==depth)&w2['type'].isna()];assert len(df)==3
            out['depth'].append(dict(arm=arm,depth=depth,n=len(df),statistics={k:agg(df[k]) for k in ['gap_mean','participation_mean','class_next_rho']}))
    groups=js(BASE/'W3_predictions/analysis/groups.json')
    for name,v in groups.items():
        if not (name.startswith('same_N') or name.startswith('density_N')):continue
        if name=='same_N200':continue
        out['groups'].append(dict(name=name,N=v['meta']['N'],protocol=v['meta']['protocol'],n=v['n'],statistics={k:v['metrics'][k] for k in ['gap_mean','participation_mean','class_next_rho']}))
    v=groups['combined_N200'];out['groups'].append(dict(name='combined_N200',N=200,protocol='shared reference',n=v['n'],statistics={k:v['metrics'][k] for k in ['gap_mean','participation_mean','class_next_rho']}))
    out['groups']=sorted(out['groups'],key=lambda r:(r['N'],r['protocol']))
    isolated_record=js(BASE/'W8_results_gaps/analysis/g1_record_alone.json')
    isolated_model=js(BASE/'W8_results_gaps/analysis/g1_model_alone.json')
    out['isolated']=dict(targets=[],record_shape={k:dict(point=isolated_record['point'][k],ci=isolated_record['ci'][k]) for k in ['gap_cv','gap_skew']},
        model_shape={k:dict(point=isolated_model['pooled'][k],ci=isolated_model['pooled_ci'][k]) for k in ['gap_cv','gap_skew']})
    rp=js(BASE/'W1_reference/records/reference_s9001.json');out['reference_params']=rp['params']
    for i,(name,fish) in enumerate(zip(['Fast','Medium','Slow'],[6,5,4])):
        units=[u for u in isolated_record['units'] if u['fish']==fish]
        values=np.concatenate([u['values'] for u in units]);model=[u for u in isolated_model['units'] if u['type']==i]
        out['isolated']['targets'].append(dict(type=name,fish=fish,n=len(values),target=values.mean(),rate=rp['params']['consumption_rates'][i],
            trial_means=[u['mean'] for u in units],trial_n=[u['n'] for u in units],model=agg([u['mean'] for u in model])))
    write(DATA/'isolated_distributions.json',dict(record=isolated_record['units'],model=isolated_model['units']))
    for label,df,folder in [
        ('No rest zone',w2[(w2.arm=='A1')&(w2.depth==3.3)&w2['type'].notna()],BASE/'W2_mechanism/outputs'),
        ('Weaker rest',w4[(w4.arm=='kappa_bottom_low')&(w4.phase=='heldout')],BASE/'W4_robustness/outputs')]:
        assert len(df)==9
        units=[(int(row.seed),int(row['type']),solo_values(folder/'trajectories'/f'{row["name"]}.npz')) for _,row in df.iterrows()]
        out['variants'][label]['isolated_shape']=shape_ci(units)
    # Preserve the complete selection evidence, including the historical failed guard.
    out['selection']=js(ROOT/'study_out/local_work/rest_zone_step25/outputs/selection.json')
    out['adopted_reference_note']='Step25 selected=null; the subsequent programme adopted and froze RZ75. Later seeds are new simulation realisations, not an independent empirical validation dataset.'
    out['source_fig5']=js(ROOT/'study_out/local_work/figure5_relayout_20261104/source_data/plotted_values.json')
    out['plateau']=js(ROOT/'study_out/local_work/figure5e_plateau_20261105/summary.json')
    for f in ['study_out/local_work/rest_zone_step25/PLAN.txt','study_out/local_work/section31_audit_20261104/audit_report.txt',
              'study_out/rz75_paper/COMMON.md','paper_draft/supplement/compact/tab_parameters.tex'] :source(ROOT/f)
    write(DATA/'supplement_values.json',out);write(QA/'input_hashes_saved.json',INPUTS)
    print('Prepared all saved-data tables. Empirical correlations:',out['empirical_correlations'])
    print('Isolated targets:',out['isolated']['targets'])
    print('Numerical means:',{k:v['gap_mean'] for k,v in out['numerical'].items()})

if __name__=='__main__':main()
