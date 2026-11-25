"""All W2 quantitative claims are derived here from saved trajectories."""
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
import argparse,csv,json,warnings,fcntl
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy import stats,optimize,special
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from model_w2 import *
from metrics_w2 import canonical,events,mean,sd,rho
OUT=HERE/'outputs';ANA=HERE/'analysis'

def quant(x,q):return float(np.quantile(x,q)) if len(x) else np.nan

def summarise(vals):
    a=np.array([np.nan if v is None else v for v in vals],float);v=a[np.isfinite(a)]
    return dict(mean=mean(v),min=float(min(v)) if len(v) else np.nan,max=float(max(v)) if len(v) else np.nan,sd=sd(v),n=len(v),n_total=len(a))

def csvwrite(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(clean(rows))

def traces(d,p):
    t=np.asarray(d['time'],float);z=np.asarray(d['pos'][:,:,2],float);q=np.asarray(d['tank'],float);h=np.asarray(d['heading'],float);st=d['status'];school=st==0
    num=school.sum(1);den=np.where(num>0,num,np.nan)
    need=np.where(q<p.need_setpoint,p.upward_strength,p.need_down)*(p.need_setpoint-q)
    fixed_need=np.where(q<.75,.35,.10)*(.75-q)
    hm=(h*school[:,:,None]).sum(1)/den[:,None];pol=np.linalg.norm(hm,axis=1)
    pitch=np.degrees(np.arctan2(hm[:,2],np.linalg.norm(hm[:,:2],axis=1)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        school_med=np.nanmedian(np.where(school,z,np.nan),axis=1)
    ep=d['episodes13']
    active_breaths=np.searchsorted(np.sort(ep['release']),t,side='right')-np.searchsorted(np.sort(ep['exit']),t,side='right')
    return dict(time=t,median=np.median(z,axis=1),school_median=school_med,
        mean_need=(need*school).sum(1)/den,all_mean_need=need.mean(1),
        diagnostic_need=(fixed_need*school).sum(1)/den,
        needy_share=((q<.75)&school).sum(1)/den,all_needy_share=(q<.75).mean(1),pitch=pitch,heading_z=hm[:,2],polarization=pol,
        breathing=active_breaths.astype(float),surface_contact_count=(st==2).sum(1).astype(float),n_school=num,
        floor_fraction=(z<.6).mean(1),empty_sample=((q<=1e-12)&(st!=2)).mean(1))

def dwell_intervals(t,z,threshold=1.):
    """Continuous threshold crossings by linear interpolation of the 1 s samples."""
    down=[];up=[]
    if z[0]<threshold:down.append(t[0])
    for i in np.flatnonzero((z[:-1]<threshold)!=(z[1:]<threshold)):
        cross=t[i]+(threshold-z[i])/(z[i+1]-z[i])*(t[i+1]-t[i])
        (up if z[i]<threshold else down).append(cross)
    if z[-1]<threshold:up.append(t[-1])
    return np.column_stack([down,up]) if down else np.empty((0,2))

def auc(y,p):
    a=np.sum(y);b=len(y)-a
    return float((stats.rankdata(p)[y==1].sum()-a*(a+1)/2)/(a*b)) if a and b else np.nan

def logistic(X,y):
    def fun(b):
        eta=X@b
        return np.sum(np.logaddexp(0,eta)-y*eta)+1e-8*np.dot(b,b),X.T@(special.expit(eta)-y)+2e-8*b
    fit=optimize.minimize(fun,np.zeros(X.shape[1]),jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':500})
    b=fit.x;phat=special.expit(X@b);ll=-float(np.sum(np.logaddexp(0,X@b)-y*(X@b)))
    grad=float(np.max(np.abs(fun(b)[1])))
    return dict(coef=b,converged=bool(fit.success or grad<1e-4),max_gradient=grad,n=len(y),n_lifts=int(y.sum()),loglik=ll,aic=2*len(b)-2*ll,auc=auc(y,phat),brier=float(np.mean((y-phat)**2)),predict=phat)

def breath_observation_end(d,T):
    """Conservative cutoff before any departure lacking a completed episode.

    Full-precision transitions retain unfinished motor episodes at the right edge.
    Their release time is unavailable, so exclude windows reaching their departure.
    """
    completed={(int(x['agent']),float(x['depart'])) for x in d['episodes13']}
    pending=[float(x['time']) for x in d['events'] if x['to_state']==1 and (int(x['agent']),float(x['time'])) not in completed]
    return float(np.nextafter(min(pending),-np.inf)) if pending else float(T)

def lift_analysis(d,p,r,tr,e):
    t=tr['time'];T=r['seconds'];burn=r['job']['burn'];med=tr['median']
    ix=np.flatnonzero((med[:-1]<1)&(med[1:]>=1))+1
    ix=ix[(t[ix]>=burn)&(t[ix]<=T)]
    observation_end=breath_observation_end(d,T)
    complete=ix[(t[ix]>=burn+15)&(t[ix]<=T-15)]
    valid=complete[t[complete]+15<=observation_end]
    lags=np.arange(-15,16);keys=['median','school_median','mean_need','all_mean_need','needy_share','all_needy_share','surface_contact_count','pitch','polarization','breathing']
    eta={k:np.mean(tr[k][valid[:,None]+lags],axis=0) if len(valid) else np.full(len(lags),np.nan) for k in keys}
    eta.update(lags=lags,n_windows=len(valid),breath_observation_end=observation_end,n_unfinished_breath_excluded=len(complete)-len(valid))
    # Prospective per-second hazard: state/predictors at t predict crossing at t+1.
    end=e['end'];idx=np.searchsorted(end,t,side='right')-1;age=t-np.where(idx>=0,end[np.maximum(idx,0)],np.nan) if len(end) else np.full_like(t,np.nan)
    risk=(t[:-1]>=burn)&(t[:-1]<T-1)&(med[:-1]<1)&np.isfinite(age[:-1])&np.isfinite(tr['mean_need'][:-1])
    ri=np.flatnonzero(risk);y=(med[ri+1]>=1).astype(float)
    predictors=np.column_stack([tr['mean_need'][ri]/.01,age[ri]/10])
    fits={}
    if len(y) and y.sum() and y.sum()<len(y):
        for label,col in [('time_only',[1]),('need_only',[0]),('both',[0,1])]:
            X=np.column_stack([np.ones(len(ri)),predictors[:,col]])
            f=logistic(X,y);f.pop('predict');fits[label]=f
    leader=[]
    for i in ix:
        order=np.argsort(d['pos'][i,:,2],kind='stable');n=max(1,int(np.ceil(.1*p.N)))
        top=order[-n:];other=order[:-n];typ=d['type_id'];q=d['tank'][i]
        row=dict(seed=r['seed'],time=t[i],top_q=float(q[top].mean()),rest_q=float(q[other].mean()),top_needy=float((q[top]<.75).mean()),rest_needy=float((q[other]<.75).mean()),top_school=float((d['status'][i,top]==0).mean()))
        for k,name in enumerate(['fast','medium','slow']):row['top_'+name]=float((typ[top]==k).mean());row['rest_'+name]=float((typ[other]==k).mean())
        leader.append(row)
    summary=dict(n_liftoffs=len(ix),n_eta_windows=len(valid),eta_unfinished_breath_excluded=len(complete)-len(valid),need_at_liftoff=mean(tr['mean_need'][ix]),
      all_need_at_liftoff=mean(tr['all_mean_need'][ix]),all_needy_at_liftoff=mean(tr['all_needy_share'][ix]),diagnostic_need_at_liftoff=mean(tr['diagnostic_need'][ix]),positive_need_liftoff_fraction=mean(tr['mean_need'][ix]>0),
      needy_at_liftoff=mean(tr['needy_share'][ix]),mean_pitch_liftoff=mean(tr['pitch'][ix]),
      hazard_predictor_r=rho(predictors[:,0],predictors[:,1]) if len(predictors) else np.nan)
    for key in leader[0] if leader else []:
        if key not in ['seed','time']:summary['leader_'+key]=mean([a[key] for a in leader])
    return summary,eta,fits,leader,dict(time=t[ri],y=y,need=predictors[:,0],age=predictors[:,1]),ix

def size_analysis(e,tr,T,burn):
    t=tr['time'];rests=dwell_intervals(t,tr['median']);rows=[];g=e['gaps'];c=e['cls'][:-1]
    for j,(a,b) in enumerate(zip(e['end'][:-1],e['start'][1:])):
        overlap=np.maximum(0,np.minimum(rests[:,1],b)-np.maximum(rests[:,0],a))
        rows.append(dict(event_end=a,next_start=b,participation=e['part'][j],cls=int(c[j]),gap=g[j],rest_duration=float(overlap.sum()),longest_rest=float(overlap.max()) if len(overlap) else 0.))
    d=np.array([a['rest_duration'] for a in rows]);x=stats.rankdata(c);y=stats.rankdata(g);z=stats.rankdata(d)
    total=rho(c,g);partial=np.nan;atten=np.nan;direct=np.nan;indirect=np.nan
    if len(g)>3 and np.std(x)>0 and np.std(y)>0 and np.std(z)>0:
        x=(x-x.mean())/x.std();y=(y-y.mean())/y.std();z=(z-z.mean())/z.std()
        Z=np.column_stack([np.ones(len(g)),z])
        rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0];ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
        partial=float(np.corrcoef(rx,ry)[0,1]);beta=np.linalg.lstsq(np.column_stack([np.ones(len(g)),x,z]),y,rcond=None)[0]
        direct=float(beta[1]);indirect=total-direct;atten=indirect/total if abs(total)>1e-6 else np.nan
    lags=np.arange(0,41);curves={};counts={}
    # Canonical small/large classes; large combines classes 3 and 4 (>=50%).
    # Fixed windows include intervening events, with contamination reported by lag.
    for label,mask in [('small',e['part']<.2),('large',e['part']>=.5)]:
        inds=np.flatnonzero(mask&(e['end']>=burn)&(e['end']<=T-40))
        curves[label]=np.mean([np.interp(e['end'][j]+lags,t,tr['mean_need']) for j in inds],axis=0) if len(inds) else np.full(len(lags),np.nan)
        curves[label+'_all']=np.mean([np.interp(e['end'][j]+lags,t,tr['all_mean_need']) for j in inds],axis=0) if len(inds) else np.full(len(lags),np.nan)
        nxt=np.array([e['start'][j+1]-e['end'][j] if j+1<len(e['start']) else T-e['end'][j] for j in inds])
        curves[label+'_uncontaminated_share']=np.mean(nxt[:,None]>lags,axis=0) if len(inds) else np.full(len(lags),np.nan)
        # Supplement: only observations before the next event; sample count falls by lag.
        arr=np.array([np.interp(e['end'][j]+lags,t,tr['mean_need']) for j in inds]) if len(inds) else np.empty((0,len(lags)))
        counts[label]=len(inds)
        valid=nxt[:,None]>lags if len(inds) else np.empty((0,len(lags)),bool)
        curves[label+'_at_risk_n']=valid.sum(0)
        curves[label+'_censored']=np.divide(np.where(valid,arr,0).sum(0),valid.sum(0),out=np.full(len(lags),np.nan),where=valid.sum(0)>0)
    summ=dict(class_next_rho=total,class_rest_rho=rho(c,d),size_rest_rho=rho(e['part'][:-1],d),rest_gap_rho=rho(d,g),class_nonrest_rho=rho(c,g-d),nonrest_mean=mean(g-d),
      partial_class_gap_given_rest=partial,rank_direct_beta=direct,rank_indirect_beta=indirect,rank_attenuation_fraction=atten,
      small_recovery_events=counts['small'],large_recovery_events=counts['large'])
    for label in ['small','large']:
        summ[label+'_need_at_event_end']=curves[label][0]
        summ[label+'_all_need_at_event_end']=curves[label+'_all'][0]
    for code in range(1,5):summ[f'class{code}_rest_mean']=mean(d[c==code]);summ[f'class{code}_n']=int(np.sum(c==code))
    curves['lags']=lags
    return summ,rows,curves

def run_analysis(path):
    r=json.loads(path.read_text());j=r['job'];p,_=settings(j);tp=OUT/'trajectories'/f'{r["name"]}.npz'
    assert digest(tp)==r['sha256']
    with np.load(tp) as zz:d={k:zz[k] for k in zz.files}
    m,e=canonical(d,p.N,j['seconds'],j['burn']);m['empty_fraction']=r['audit']['empty_fraction'];m['realized_consumption_mean']=float(np.mean(d['agent_consumption_rates']))
    result=dict(name=r['name'],job=j,metrics=m,sha256=r['sha256'])
    if p.N==1:
        q=d['tank'][d['time']>=j['burn']];m['never_breathed_postburn']=int(not np.any(d['episodes13']['release']>=j['burn']))
        m['min_tank']=float(q.min());result['observed_intervals']=len(e['individual'])
        write_json(ANA/'runs'/f'{r["name"]}.json',result);return result
    tr=traces(d,p);t=tr['time'];valid=(t>=j['burn'])&(t<j['seconds'])
    sm=uniform_filter1d(tr['median'],3,mode='nearest');pk=find_peaks(sm,prominence=.2)[0];trough=find_peaks(-sm,prominence=.2)[0]
    pk=pk[valid[pk]];trough=trough[valid[trough]];cycle=np.diff(t[pk])
    mask=tr['median'][valid]<1;edge=np.diff(np.r_[False,mask,False].astype(int));a=np.flatnonzero(edge==1);b=np.flatnonzero(edge==-1)
    durations=b-a;complete=(a>0)&(b<len(mask))
    # Test whether the proposed rest mechanism is literally a zero-need crossing.
    # Use complete sampled rests with a defined schooling need throughout.
    need=tr['mean_need'][valid];rest_rows=[]
    for aa,bb in zip(a[complete],b[complete]):
        v=need[aa:bb]
        if not np.all(np.isfinite(v)):continue
        rest_rows.append(dict(seed=r['seed'],start=t[valid][aa],end=t[valid][bb],duration=bb-aa,
            need_entry=v[0],need_min=v.min(),need_last=v[-1],
            positive_throughout=bool(np.all(v>0)),positive_at_entry=bool(v[0]>0),
            has_upward_zero_crossing=bool(np.any((v[:-1]<=0)&(v[1:]>0)))))
    for key in ['need_entry','need_min','need_last','positive_throughout','positive_at_entry','has_upward_zero_crossing']:
        m['rest_'+key]=mean([v[key] for v in rest_rows])
    m['rest_sign_n']=len(rest_rows)
    if j['arm'] in ['A0','A1'] and j['depth']==3.3:csvwrite(ANA/f'{r["name"]}_rest_need.csv',rest_rows)
    m.update(crest_cycle_mean=mean(cycle),crest_cycle_sd=sd(cycle),crest_cycle_median=quant(cycle,.5),crest_count=len(pk),
      bottom_dwell_mean=mean(durations),bottom_dwell_complete_mean=mean(durations[complete]),bottom_dwell_n=len(durations),bottom_dwell_censored=int(np.sum(~complete)),
      bottom_fraction=float(mask.mean()),within_rest_fraction=float(np.mean(tr['median'][valid]<p.rest_height)),median_height_mean=mean(tr['median'][valid]),floor_fraction=mean(tr['floor_fraction'][valid]),pitch_p05=quant(tr['pitch'][valid][np.isfinite(tr['pitch'][valid])],.05),pitch_p95=quant(tr['pitch'][valid][np.isfinite(tr['pitch'][valid])],.95),
      polarization=mean(tr['polarization'][valid][np.isfinite(tr['polarization'][valid])]),trough_height=mean(sm[trough]),trough_depth=mean(p.box[2]-sm[trough]),
      rise_rate_min=len(pk)*60/(j['seconds']-j['burn']),surface_rise_rate_min=int(np.sum(sm[pk]>=p.box[2]-1))*60/(j['seconds']-j['burn']))
    zone_exits=(tr['median'][:-1]<p.rest_height)&(tr['median'][1:]>=p.rest_height)&valid[1:]
    m['rest_height_exit_rate_min']=int(zone_exits.sum())*60/(j['seconds']-j['burn'])
    # Inherited metric: unique nearest crest within 4 s; event starts otherwise unassigned.
    full=pk[(t[pk]>=j['burn']+4)&(t[pk]<j['seconds']-4)];assigned=np.zeros(len(full),int);unassigned=0
    for start in e['start']:
        delta=np.abs(t[full]-start)
        if len(delta) and delta.min()<=4:assigned[np.argmin(delta)]+=1
        else:unassigned+=1
    for label,condition in [('0',assigned==0),('1',assigned==1),('2plus',assigned>=2)]:m['rises_'+label+'_fraction']=mean(condition)
    m['rise_bouts_unassigned']=unassigned;m['rises_scored']=len(full)
    # Broader rise-cycle assignment: midpoints between crests, all complete interior cycles.
    bounds=(t[pk[:-1]]+t[pk[1:]])/2;cycle_counts=np.histogram(e['start'],bins=bounds)[0] if len(bounds)>1 else np.array([])
    for label,condition in [('0',cycle_counts==0),('1',cycle_counts==1),('2plus',cycle_counts>=2)]:m['cycle_events_'+label+'_fraction']=mean(condition)
    lift,eta,hazard,leaders,risk,ix=lift_analysis(d,p,r,tr,e);m.update(lift);result['hazard']=hazard
    if j['arm']=='A0' and j['depth']==3.3:
        mechanism,evrows,curves=size_analysis(e,tr,j['seconds'],j['burn']);m.update(mechanism)
        csvwrite(ANA/f'{r["name"]}_event_rest.csv',[dict(seed=r['seed'],**x) for x in evrows]);csvwrite(ANA/f'{r["name"]}_leaders.csv',leaders)
        np.savez_compressed(ANA/f'{r["name"]}_recovery.npz',**curves)
        np.savez_compressed(ANA/f'{r["name"]}_risk.npz',**risk)
    if j['arm'] in ['A0','A1'] and j['depth']==3.3:
        np.savez_compressed(ANA/f'{r["name"]}_liftoff.npz',**eta)
    # Small derived one-dimensional series for reproducible plots and audits, no raw duplicates.
    np.savez_compressed(ANA/'traces'/f'{r["name"]}.npz',**tr,event_start=e['start'],event_end=e['end'],event_part=e['part'],crests=t[pk],troughs=t[trough],liftoffs=t[ix])
    write_json(ANA/'runs'/f'{r["name"]}.json',result);return result

def empirical():
    sys.path.insert(0,str(ROOT/'paper'))
    from results_pipeline.analysis import analyse_empirical
    raw=analyse_empirical()
    import pandas as pd
    x=ROOT/'arapaima-air-breathing/data/supplementary/Supplementary_Data_1.xlsx';f=pd.read_excel(x,sheet_name='Fig3')
    gap=f.iloc[:,2].astype(float).values;dur=f.iloc[:,1].astype(float).values;cls=f.iloc[:,3].astype(int).values
    starts=np.flatnonzero(~np.isfinite(gap));bounds=list(starts)+[len(gap)]
    segs=[dict(gap=gap[a+1:b],duration=dur[a:b],cls=cls[a:b],next_cls=cls[a:b-1],previous_cls=cls[a+1:b]) for a,b in zip(bounds[:-1],bounds[1:])]
    def calc(ss):
        g=np.concatenate([s['gap'] for s in ss]);c=np.concatenate([s['cls'] for s in ss]);du=np.concatenate([s['duration'] for s in ss]);nc=np.concatenate([s['next_cls'] for s in ss]);pc=np.concatenate([s['previous_cls'] for s in ss])
        result=dict(gap_mean=mean(g),gap_sd=sd(g),short_fraction=mean(g<5),long_fraction=mean(g>30),slab_mean=mean(du),class_next_rho=rho(nc,g),class_previous_rho=rho(pc,g))
        result.update({name+'_fraction':mean(c==i) for i,name in enumerate(['small','medium','large','huge'],1)})
        return result
    val=calc(segs);rng=np.random.default_rng(20902);boot=[calc([segs[k] for k in rng.integers(0,len(segs),len(segs))]) for _ in range(2000)]
    cis={k:np.quantile([b[k] for b in boot],[.025,.975]) for k in val}
    assert np.isclose(val['gap_mean'],raw['gap_mean']);assert np.isclose(val['class_previous_rho'],raw['r_cls_prevgap'])
    result=dict(values=val,ci95=cis,ci_method='2000 whole-recording-segment bootstrap resamples; six segments; no cross-boundary pairs',n_segments=len(segs),segment_bouts=[len(s['cls']) for s in segs],n_intervals=len(raw['_gaps']),
      alone_mean=raw['alone_mean'],alone_targets=[55.630943,85.5925,134.24],source_sha256=digest(x),frame_scenario_factor=1.5,
      frame_scenario_mean=1.5*val['gap_mean'],frame_scenario_sd=1.5*val['gap_sd'])
    write_json(ANA/'empirical.json',result);return result

def depth_summary(rows):
    out=[]
    for arm in ['A0','A1']:
        for seed in range(9001,9004):
            rr=sorted([r for r in rows if r['job']['type'] is None and r['job']['arm']==arm and r['job']['seed']==seed],key=lambda r:r['job']['depth'])
            if len(rr)!=4:continue
            depths=np.array([r['job']['depth'] for r in rr]);v=dict(arm=arm,seed=seed)
            for key in ['gap_mean','gap_sd','crest_cycle_mean','bottom_dwell_complete_mean','trough_depth']:
                yy=np.array([np.nan if r['metrics'][key] is None else r['metrics'][key] for r in rr],dtype=float);v[key+'_slope']=float(np.polyfit(depths,yy,1)[0]);v[key+'_deep_minus_shallow']=float(yy[-1]-yy[0]);v[key+'_deep_minus_reference']=float(yy[-1]-yy[1]);v[key+'_slope_reference_and_deeper']=float(np.polyfit(depths[1:],yy[1:],1)[0])
            out.append(v)
    csvwrite(ANA/'depth_slopes.csv',out)
    if not out:return {}
    result={arm:{k:summarise([r[k] for r in out if r['arm']==arm]) for k in out[0] if k not in ['arm','seed']} for arm in ['A0','A1']}
    pairs=[]
    for seed in range(9001,9004):
        rz=next((r for r in out if r['arm']=='A0' and r['seed']==seed),None)
        nr=next((r for r in out if r['arm']=='A1' and r['seed']==seed),None)
        if rz is None or nr is None:continue
        pairs.append(dict(seed=seed,**{k:rz[k]-nr[k] for k in rz if k not in ['arm','seed']}))
    csvwrite(ANA/'depth_paired_differences.csv',pairs)
    result['paired_A0_minus_A1']={k:summarise([r[k] for r in pairs]) for k in out[0] if k not in ['arm','seed']}
    return result

def pooled_hazard(rows):
    sets=[]
    for seed in range(9001,9004):
        path=ANA/f'A0_s{seed}_risk.npz'
        if path.exists():
            with np.load(path) as d:sets.append(dict(seed=seed,**{k:d[k] for k in d.files}))
    if len(sets)!=3:return {}
    X=np.concatenate([np.column_stack([np.ones(len(d['y'])),np.full(len(d['y']),i==1),np.full(len(d['y']),i==2),d['need'],d['age']]) for i,d in enumerate(sets)])
    y=np.concatenate([d['y'] for d in sets]);fit=logistic(X,y);fit.pop('predict');fit['columns']=['intercept','seed9002','seed9003','need_per_0.01','age_per_10s']
    cv=[]
    for i,test in enumerate(sets):
        train=[d for j,d in enumerate(sets) if i!=j]
        for label,cols in [('time_only',['age']),('need_only',['need']),('both',['need','age'])]:
            xx=np.concatenate([np.column_stack([np.ones(len(d['y']))]+[d[c] for c in cols]) for d in train]);yy=np.concatenate([d['y'] for d in train])
            f=logistic(xx,yy);xt=np.column_stack([np.ones(len(test['y']))]+[test[c] for c in cols]);pr=special.expit(xt@f['coef']);eta=xt@f['coef']
            cv.append(dict(seed=test['seed'],model=label,auc=auc(test['y'],pr),brier=float(np.mean((test['y']-pr)**2)),negative_loglik_per_s=float(np.mean(np.logaddexp(0,eta)-test['y']*eta)),n_risk=len(pr),n_lifts=int(test['y'].sum())))
    csvwrite(ANA/'hazard_cross_validation.csv',cv)
    return dict(pooled_seed_intercepts=fit,cross_validation=cv,note='Per-second prospective logistic hazard, risk set below 1 BL; leave-one-seed-out validation. No independence-based p-values.')

def w1_compare(force=False):
    cached=HERE/'qa/w1_comparison.json'
    if cached.exists() and not force:
        previous=json.loads(cached.read_text())
        if previous.get('status')=='complete':return previous
    folder=HERE.parent/'W1_reference';rows=[]
    if not folder.exists():return {'status':'W1 folder unavailable','runs':[]}
    for seed in range(9001,9004):
        ours=OUT/'trajectories'/f'A0_s{seed}.npz'
        # Request explicitly permits reading W1 to compare the matched prefix.
        paths=[p for p in folder.rglob(f'reference_s{seed}.npz') if '.tmp.' not in p.name]
        if not ours.exists() or not paths:rows.append(dict(seed=seed,status='not yet available'));continue
        chosen=sorted(paths,key=lambda p:p.stat().st_size,reverse=True)[0]
        try:
            with np.load(ours) as a,np.load(chosen) as b:
                checks={};notes={}
                for k in ['time','pos','heading','tank','status','type_id','agent_consumption_rates','episodes13','events','decision_trials','entries21']:
                    if k not in b.files:notes[k]='absent in W1';continue
                    aa=a[k];bb=b[k]
                    if k in ['time','pos','heading','tank','status']:bb=bb[:len(aa)]
                    elif k in ['episodes13','events','decision_trials','entries21']:
                        field='exit' if k=='episodes13' else 'time';bb=bb[bb[field]<=7500.]
                        if k in ['decision_trials','entries21']:bb=bb[bb[field]<7500.]
                    if bb.dtype!=aa.dtype:
                        if k in ['time','pos','heading','tank','status','type_id']:
                            notes[k]=f'W1 {bb.dtype} cast to required W2 storage {aa.dtype}'
                            bb=bb.astype(aa.dtype)
                        else:notes[k]=f'dtype differs {aa.dtype} vs {bb.dtype}'
                    checks[k]=bool(aa.shape==bb.shape and aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes())
                rows.append(dict(seed=seed,status='pass' if checks and all(checks.values()) else 'mismatch',source=str(chosen.relative_to(ROOT)),source_sha256=digest(chosen),w2_sha256=digest(ours),checks=checks,notes=notes))
        except Exception as ex:rows.append(dict(seed=seed,status='read error',error=repr(ex)))
    result=dict(status='complete' if len(rows)==3 and all(r['status']=='pass' for r in rows) else 'pending or mismatch',runs=rows,
        precision_note='Byte comparison of required stored float32/int8 trajectories and original-precision event/trial logs. Full-precision pre-cast field hashes are retained separately in W2 records.')
    write_json(HERE/'qa/w1_comparison.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--refresh',action='store_true');a=ap.parse_args()
    for f in ['runs','traces']: (ANA/f).mkdir(parents=True,exist_ok=True)
    analysis_lock=(ANA/'analysis.lock').open('w');fcntl.flock(analysis_lock,fcntl.LOCK_EX)
    records=sorted((OUT/'records').glob('*.json'));expected=len(jobs())
    if not a.partial:assert len(records)==expected,(len(records),expected)
    rows=[]
    for path in records:
        saved=ANA/'runs'/path.name
        if saved.exists() and not a.refresh:r=json.loads(saved.read_text())
        else:r=run_analysis(path)
        rows.append(r)
    groups={}
    for r in rows:
        j=r['job'];key=f'{j["arm"]}_z{j["depth"]:.1f}'+(f'_solo_t{j["type"]}' if j['type'] is not None else '')
        groups.setdefault(key,[]).append(r)
    agg={g:{k:summarise([r['metrics'].get(k) for r in rr]) for k in dict.fromkeys(k for r in rr for k in r['metrics'])} for g,rr in groups.items()}
    csvwrite(ANA/'per_run_metrics.csv',[dict(**r['job'],**r['metrics']) for r in rows])
    summary=dict(completed=len(rows),expected=expected,groups=agg,depth=depth_summary(rows),hazard=pooled_hazard(rows),w1=w1_compare(force=a.refresh))
    if not (ANA/'empirical.json').exists() or a.refresh:empirical()
    summary['empirical']=json.loads((ANA/'empirical.json').read_text())
    write_json(ANA/'summary.json',summary)
    print(json.dumps(dict(analysed=len(rows),expected=expected,groups=len(groups),w1=summary['w1']['status'])))

if __name__=='__main__':main()
