"""Run all W1 comparisons from finished, hashed trajectories; no simulation writes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,csv,argparse
from pathlib import Path
import numpy as np
from scipy import stats
from analysis_core import (HERE,ROOT,clean,write_json,digest,load_record,analyse_run,aggregate_scalars,
                           distribution_yardstick,shape_segments,record_statistics,TYPES,ALONE_TARGETS,EMP_XLSX)

def csv_write(path, rows):
    rows=clean(rows)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def array_flat(a):
    out={}
    for k,v in a.items():
        if isinstance(v,dict):
            for sub,value in v.items():out[k+'__'+sub]=value
        else:out[k]=v
    return out

def array_unflat(d):
    out={}
    for k in d.files:
        if '__' in k:
            base,sub=k.split('__');out.setdefault(base,{})[sub]=d[k]
        else:out[k]=d[k]
    return out

def load_runs(seeds=range(9001,9011)):
    record=load_record();runs=[]
    analysis_sha=digest(HERE/'analysis_core.py')
    for seed in seeds:
        name=f'reference_s{seed}';rp=HERE/'records'/f'{name}.json';tp=HERE/'trajectories'/f'{name}.npz'
        assert rp.exists() and tp.exists(),f'Run not complete: {name}'
        rec=json.loads(rp.read_text());assert digest(tp)==rec['sha256']
        cached=HERE/'analysis'/f'{name}.json';ap=HERE/'analysis'/f'{name}_derived.npz'
        if cached.exists() and ap.exists():
            old=json.loads(cached.read_text())
            if (old['analysis_sha256']==analysis_sha and old['input_sha256']==rec['sha256']
                    and old.get('empirical_sha256')==record['source_sha256'] and old['derived_sha256']==digest(ap)):
                with np.load(ap) as d:arrays=array_unflat(d)
                runs.append(dict(seed=seed,scalar=old['scalar'],arrays=arrays));continue
        row=analyse_run(tp,alone_mean=record['alone'].mean());row['seed']=seed
        np.savez_compressed(ap,**array_flat(row['arrays']))
        write_json(cached,dict(seed=seed,scalar=row['scalar'],input_sha256=rec['sha256'],
                              empirical_sha256=record['source_sha256'],analysis_sha256=analysis_sha,derived_sha256=digest(ap)))
        runs.append(row)
    return runs

TABLE_METRICS=[
 ('gap_mean','Collective interval mean (s)','gap_mean',True),
 ('gap_sd','Collective interval SD (s)','gap_sd',True),
 ('gap_cv','Collective interval CV','gap_cv',False),
 ('gap_median','Collective interval median (s)','gap_median',False),
 ('gap_skew','Collective interval skew','gap_skew',False),
 ('gap_kurtosis','Collective interval Pearson kurtosis','gap_kurtosis',False),
 ('short_fraction','Intervals <5 s (%)','short_fraction',True),
 ('long_fraction','Intervals >30 s (%)','long_fraction',True),
 ('class_1','Participation <20% (%)','class_1',True),
 ('class_2','Participation 20–50% (%)','class_2',False),
 ('class_3','Participation 50–80% (%)','class_3',False),
 ('class_4','Participation ≥80% (%)','class_4',True),
 ('class_next_rho','Class → next interval Spearman','class_next_rho',False),
 ('class_previous_rho','Class → previous interval Spearman','class_previous_rho',False),
 ('slab_0.5_mean','Front-slab duration mean (s)','duration_mean',True),
 ('slab_0.5_sd','Front-slab duration SD (s)','duration_sd',False),
 ('slab_0.5_class_rho','Class → front-slab duration Spearman','class_duration_rho',False),
 ('whole_duration_mean','Whole-event duration mean (s)',None,False),
 ('effect_size','Alone/collective mean ratio','effect_size',False),
 ('individual_fast','Fast per-fish interval (s)',None,False),
 ('individual_medium','Medium per-fish interval (s)',None,False),
 ('individual_slow','Slow per-fish interval (s)',None,False),
 ('individual_mean','All per-fish intervals (s)',None,False),
 ('polarization','School polarisation',None,True),
 ('repeat_share','Repeated breaths / all breaths (%)',None,False),
 ('burst10_mean','First ten releases mean span (s)',None,False)]

def comparison_table(summary,record_uncertainty):
    point=record_uncertainty['point'];ci=record_uncertainty['methods']['block10']['ci'];rows=[]
    for model_key,label,rec_key,selected in TABLE_METRICS:
        m=summary[model_key];v=point[rec_key] if rec_key else None
        c=ci[rec_key] if rec_key else None
        rows.append(dict(statistic=model_key,label=label,record=v,record_low=c[0] if c else None,
                         record_high=c[1] if c else None,model_mean=m['mean'],model_min=m['minimum'],
                         model_max=m['maximum'],model_q025=m['q025'],model_q975=m['q975'],
                         point_inside_minmax=m['minimum']<=v<=m['maximum'] if v is not None else None,
                         point_inside_quantiles=m['q025']<=v<=m['q975'] if v is not None else None,
                         ci_overlaps_minmax=c[0]<=m['maximum'] and c[1]>=m['minimum'] if c else None,
                         used_in_selection=selected))
    return rows

def main():
    (HERE/'analysis').mkdir(exist_ok=True)
    record=load_record();runs=load_runs()
    for r in runs:
        for field,key in [('burst10','burst10_within_scenario_B'),('burst10_distinct','burst10_distinct_within_scenario_B')]:
            x=r['arrays'][field];x=x[np.isfinite(x)]
            r['scalar'][key]=float(np.mean(x<=1.5*record['burst10_record']+1e-10))
    summary=aggregate_scalars(runs)
    uncertainty=json.loads((HERE/'analysis/record_uncertainty.json').read_text())
    import pandas as pd
    f2=pd.read_excel(EMP_XLSX,sheet_name='Fig2')
    fish=[dict(fish_id=int(i),mean=float(g.iloc[:,3].mean()),n=int(g.iloc[:,3].count())) for i,g in f2.groupby('Fish_ID')]
    matches=[]
    for typ,target in zip(TYPES,ALONE_TARGETS):
        match=min(fish,key=lambda r:abs(r['mean']-target))
        matches.append(dict(type=typ,frozen_target=float(target),**match,absolute_difference=abs(match['mean']-target)))
    assert all(r['absolute_difference']<1e-5 for r in matches)
    write_json(HERE/'analysis/alone_target_provenance.json',dict(fish=fish,numerical_matches=matches,
       note='Numerical matches only; no inference about why calibration targets were selected; frozen targets unchanged.'))
    gaps=np.concatenate([s[1:,0] for s in record['segments']])
    yard=distribution_yardstick(runs,gaps);yardB=distribution_yardstick(runs,gaps*1.5)
    shape=shape_segments(runs,record)
    tables={'A':comparison_table(summary,uncertainty),'B':comparison_table(summary,uncertainty['scenario_B'])}
    for sc,t in tables.items():csv_write(HERE/f'analysis/table1_scenario_{sc}.csv',t)
    csv_write(HERE/'analysis/model_replicates.csv',[dict(seed=r['seed'],**{k:v for k,v in r['scalar'].items() if k!='aic'},
       aic_ranking=' < '.join(sorted(r['scalar']['aic'],key=r['scalar']['aic'].get)),
       **{'aic_'+k:v for k,v in r['scalar']['aic'].items()}) for r in runs])
    csv_write(HERE/'analysis/model_summary.csv',[dict(statistic=k,**v) for k,v in summary.items()])
    csv_write(HERE/'analysis/record_segment_shape.csv',[{k:v for k,v in r.items() if k!='aic'} for r in shape['record']])
    csv_write(HERE/'analysis/model_chunk_shape.csv',[{k:v for k,v in r.items() if k!='aic'} for r in shape['model']])
    csv_write(HERE/'analysis/ks_cvm_record.csv',[dict(seed=runs[x['run']]['seed'],scenario=sc,**x) for sc,y in [('A',yard),('B',yardB)] for x in y['record_comparisons']])
    csv_write(HERE/'analysis/ks_cvm_pairs.csv',yard['pairs'])
    rng=np.random.default_rng(90019010);idx=rng.integers(0,len(runs),(10000,len(runs)))
    ci_mean={}
    for k in summary:
        vals=np.array([r['scalar'][k] for r in runs],float)
        ci_mean[k]=np.nanquantile(np.nanmean(vals[idx],axis=1),[.025,.975])
    shape_summary=[]
    for i,rec in enumerate(shape['record']):
        all_candidates=[x for x in shape['model'] if x['template_segment']==i+1 and x['complete']]
        candidates=[x for x in all_candidates if x['chunk']<=6]
        assert len({x['run'] for x in candidates})==len(candidates)
        assert candidates,'No complete model chunk for this recorded segment size'
        row=dict(segment=i+1,record_events=rec['events'],model_chunks=len(candidates),
                 all_complete_chunks=len(all_candidates),continuation_diagnostics={})
        for key in ['gap_skew','gap_kurtosis','laplace_gamma']:
            vals=np.array([x[key] for x in candidates]);row[key]=dict(record=rec[key],mean=vals.mean(),minimum=vals.min(),maximum=vals.max(),q025=np.quantile(vals,.025),q975=np.quantile(vals,.975))
            byseed=np.array([np.mean([x[key] for x in all_candidates if x['run']==j]) for j in sorted({x['run'] for x in all_candidates})])
            row['continuation_diagnostics'][key]=dict(equal_seed_mean=byseed.mean(),seed_min=byseed.min(),seed_max=byseed.max(),
                  chunk_min=min(x[key] for x in all_candidates),chunk_max=max(x[key] for x in all_candidates))
        shape_summary.append(row)
    # Within-segment centering removes mean drift; no pseudo chronological joins.
    rec_residual=np.concatenate([s[1:,0]-s[1:,0].mean() for s in record['segments']])
    shape_drift=dict(record_global_kurtosis=stats.kurtosis(gaps,fisher=False),
                     record_centered_kurtosis=stats.kurtosis(rec_residual,fisher=False),
                     record_between_segment_mean_sd=np.std([s[1:,0].mean() for s in record['segments']],ddof=1),
                     record_within_segment_skew=[r['gap_skew'] for r in shape['record']])
    winners={k:sum(r['scalar']['aic_winner']==k for r in runs) for k in uncertainty['point']['aic']}
    result=dict(summary=summary,replicates=[dict(seed=r['seed'],scalar=r['scalar']) for r in runs],
                model_mean_ci=ci_mean,yardstick_A=yard,yardstick_B=yardB,shape=shape,
                shape_summary=shape_summary,shape_drift=shape_drift,aic_winners=winners,tables=tables,
                record_annotated_gap_plus_duration_s=float(sum(np.nansum(s[:,0])+s[:,2].sum() for s in record['segments'])),
                record_source_sha256=record['source_sha256'],analysis_core_sha256=digest(HERE/'analysis_core.py'))
    write_json(HERE/'analysis/ensemble.json',result)
    records=[json.loads((HERE/'records'/f'reference_s{s}.json').read_text()) for s in range(9001,9011)]
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert all(r['source_sha256']==protocol['source_sha256'] for r in records),'Model sources changed across runs'
    accounting=dict(max_budget_error=max(r['budget_error'] for r in records),
                    max_displacement_error=max(r['displacement_error'] for r in records),
                    total_turn_violations=sum(r['turn_violations'] for r in records),
                    total_boundary_corrections=sum(r['boundary_corrections'] for r in records),
                    turn_violations=[dict(seed=r['seed'],values=r['turn_violations']) for r in records],
                    boundary_corrections=[dict(seed=r['seed'],values=r['boundary_corrections']) for r in records])
    assert accounting['max_budget_error']<1e-9 and accounting['max_displacement_error']<1e-9
    write_json(HERE/'qa/ensemble_integrity.json',dict(passed=True,n_runs=len(runs),
              launch_source_hashes_all_equal=True,
              frozen_params_all_equal=all(json.loads((HERE/'records'/f'reference_s{s}.json').read_text())['params']==json.loads((HERE/'protocol.json').read_text())['params'] for s in range(9001,9011)),
              exact_budget_max_error=max(abs(r['scalar']['budget_exact_error']) for r in runs),accounting=accounting))
    print(json.dumps(clean(dict(summary=summary,aic_winners=winners)),indent=2))

if __name__=='__main__':main()
