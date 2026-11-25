"""Saved-run decision analysis. No simulation calls; every writer stays in HERE."""
import os,sys,json,hashlib,warnings
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; BASE=HERE.parent
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
sys.path[:0]=[str(ROOT),str(BASE/'W1_reference'),str(BASE/'W5b_code')]
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from analysis_core import event_arrays,summarize_arrays,clean
from rz75_model import decision_probability
TYPES=['Fast','Medium','Slow']; CATS=['Deep need','Need','Low need','No need']; ATTR=['Need-driven','Carried','Other']
QB=np.linspace(0,1,41); TB=np.linspace(0,1,21); BURN=300.; DT=.025
INPUTS={}
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def source(p):
 p=Path(p);INPUTS[str(p.relative_to(ROOT))]=digest(p);return p
def write(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def agg(x):
 a=np.asarray(x,float);n=np.sum(np.isfinite(a),axis=0)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  return dict(mean=np.nanmean(a,axis=0),min=np.nanmin(a,axis=0),max=np.nanmax(a,axis=0),n=n)
def mean(x):return float(np.mean(x)) if len(x) else np.nan
def cats(q):return np.digitize(q,[.5,.75,.8])
def shares(a,n):return np.bincount(np.asarray(a,int),minlength=n)/len(a) if len(a) else np.full(n,np.nan)
def hist(q):return np.histogram(q,QB)[0]/len(q)/np.diff(QB) if len(q) else np.full(len(QB)-1,np.nan)
def reproduction():
 rows=[]
 for seed in [7301,7313,7319]:
  p=ROOT/f'study_out/local_work/rest_zone_step25/outputs/trajectories/A_RZ75_s{seed}.npz'
  r=json.loads(source(p.parent.parent/'records'/f'{p.stem}.json').read_text());assert digest(source(p))==r['sha256']
  with np.load(p) as d:rows.append(summarize_arrays(event_arrays(d,float(d['time'][-1]))))
 expected={'gap_mean':14.647,'gap_sd':6.056,'short_fraction':.0728,'long_fraction':.0214,'class_1':.1271,'class_4':.0085,'slab_0.5_mean':2.323,'individual_mean':48.854,'class_next_rho':.172}
 out={}
 for k,v in expected.items():
  actual=mean([r[k] for r in rows]);tol=.000051 if 'fraction' in k or k.startswith('class_') and k!='class_next_rho' else .00051
  assert abs(actual-v)<tol,(k,actual,v)
  out[k]=dict(observed=actual,expected=v,tolerance=tol)
 return dict(passed=True,metrics=out)
def rising_windows(t,z,ev):
 med=np.median(z,axis=1);up=np.flatnonzero((med[:-1]<1)&(med[1:]>=1))+1;down=np.flatnonzero((med[:-1]>=1)&(med[1:]<1))+1
 windows=[]
 for u in up:
  ds=down[down>u];end=t[ds[0]] if len(ds) else t[-1]+1
  es=np.flatnonzero((ev['start']>=t[u])&(ev['start']<end))
  if len(es):
   j=es[0];windows.append((t[u],ev['start'][j],ev['end'][j]))
 return np.asarray(windows).reshape(-1,3)
def in_windows(x,w,col):
 out=np.zeros(len(x),bool)
 for a,b,c in w:out|=(x>=a)&(x<=(b if col==1 else c))
 return out

def process(path):
 print('Analyzing',path.name,flush=True)
 rec=json.loads(source(path.parent.parent/'records'/f'{path.stem}.json').read_text());assert digest(source(path))==rec['sha256']
 with np.load(path) as f:d={k:f[k] for k in ['time','pos','status','type_id','agent_consumption_rates','episodes13','decision_trials','entries21']}
 p=rec['params'];seed=rec['seed'];t=d['time'].astype(float);T=t[-1];end=T-2;N=len(d['type_id']);arm='Shoal' if N>1 else 'Alone';typ=d['type_id'];tr=d['decision_trials'];en=d['entries21'];ep=d['episodes13']
 for k,v in {'dt':DT,'need_setpoint':.75,'fermi_midpoint':.8,'fermi_temperature':.1,'decision_timescale':.3,'reaction_interval':.05,'upward_strength':.35,'need_down':.1,'school_band_weight':0,'entry_decision':False}.items():assert p[k]==v,(k,p[k],v)
 prob=decision_probability(tr['tank'],DT,.8,.1,.3);maxerr=np.max(abs(prob-tr['probability']))
 assert maxerr<1e-14 and np.array_equal(tr['accepted'],tr['draw']<tr['probability'])
 assert np.all(tr['z']>=2.7-1e-12)
 # Exact trial-to-entry mapping: latest logged entry of the same fish.
 ti=np.empty(len(tr),int);ei=np.empty(len(ep),int);match=np.empty(len(ep),int)
 for fish in range(N):
  it=np.flatnonzero(tr['agent']==fish);ie=np.flatnonzero(en['agent']==fish);ib=np.flatnonzero(ep['agent']==fish)
  ix=np.searchsorted(en['time'][ie],tr['time'][it]+1e-10,side='right')-1;assert np.all(ix>=0);ti[it]=ie[ix]
  accepted=it[tr['accepted'][it]];a=tr['time'][accepted];ix=np.searchsorted(a,ep['depart'][ib]-1e-10)
  assert np.all(ix<len(a));ma=accepted[ix];assert np.allclose(tr['time'][ma],ep['depart'][ib],rtol=0,atol=1e-9)
  assert np.array_equal(tr['tank'][ma],ep['tank_depart'][ib]);match[ib]=ma;ei[ib]=ti[ma]
 ntrial=np.bincount(ti,minlength=len(en));nyes=np.bincount(ti,weights=tr['accepted'],minlength=len(en));assert np.all(nyes<=1) and np.all(ntrial>=1)
 first=np.full(len(en),np.inf);last=np.full(len(en),-np.inf);np.minimum.at(first,ti,tr['time']);np.maximum.at(last,ti,tr['time']);assert np.allclose(first,en['time'],rtol=0,atol=1e-9)
 # Repeated trials must be contiguous in each eligible visit, until acceptance or exit.
 assert np.max(abs((last-first)/DT+1-ntrial))<1e-7
 visit_closed=last+DT<=(end+1e-9);vk=(en['time']>=BURN)&visit_closed
 v=pd.DataFrame(dict(seed=seed,arm=arm,agent=en['agent'],type=typ[en['agent']],entry=en['entry'],time=en['time'],reserve=en['tank'],n_trials=ntrial,accepted=nyes.astype(bool),last_trial=last,complete=vk))
 bk=(ep['depart']>=BURN)&(ep['release']>=BURN)&(ep['exit']<=end)
 b=pd.DataFrame(dict(seed=seed,arm=arm,agent=ep['agent'][bk],type=typ[ep['agent'][bk]],release=ep['release'][bk],exit=ep['exit'][bk],decision=ep['depart'][bk],reserve=ep['tank_depart'][bk],entry=en['entry'][ei[bk]],entry_time=en['time'][ei[bk]],entry_reserve=en['tank'][ei[bk]],entry_z=en['z'][ei[bk]],entry_trials=ntrial[ei[bk]]))
 b['need_term']=np.where(b.entry_reserve<.75,.35,.1)*(.75-b.entry_reserve)
 b['category']=cats(b.reserve);b['no_upward_need']=b.need_term<=0
 ev=event_arrays(d,T) if N>1 else None
 w=rising_windows(t,d['pos'][:,:,2],ev) if N>1 else np.empty((0,3))
 b['rising_pre_onset']=in_windows(b.entry_time.to_numpy(),w,1);b['rising_through_event']=in_windows(b.entry_time.to_numpy(),w,2)
 b['attribution']=np.where(b.need_term>0,0,np.where(b.rising_through_event,1,2))
 ix=np.searchsorted(t,b.entry_time.to_numpy()-3,side='right')-1;assert np.all(ix>=0)
 b['rank_lag_seconds']=b.entry_time.to_numpy()-t[ix]
 if N>1:
  ranks=(rankdata(d['pos'][:,:,2],axis=1)-1)/(N-1);b['height_rank_pre3']=ranks[ix,b.agent]
  b['height_pre3']=d['pos'][ix,b.agent,2]
 else:b['height_rank_pre3']=np.nan;b['height_pre3']=d['pos'][ix,b.agent,2]
 b['event_id']=-1;b['event_class']=0;b['release_stage']=-1;b['first_tied']=False;b['first10']=False
 eventrows=[];nonrows=[];firstshares=[]
 tk=(tr['time']>=BURN)&(tr['time']<end)
 if N>1:
  for j,(s,e,c) in enumerate(zip(ev['start'],ev['end'],ev['classes'])):
   ix=b.index[(b.release>=s-1e-10)&(b.release<=e+1e-10)];bb=b.loc[ix].sort_values('release',kind='stable').drop_duplicates('agent');ids=bb.agent.to_numpy();assert len(ids)==round(ev['parts'][j]*N)
   rr=rankdata(bb.release,method='average');pct=(rr-.5)/len(rr);stage=np.minimum((pct*3).astype(int),2)
   b.loc[ix,'event_id']=j;b.loc[ix,'event_class']=c;b.loc[bb.index,'release_stage']=stage;b.loc[bb.index[bb.release<=s+1e-9],'first_tied']=True;b.loc[bb.index[:10],'first10']=True
   firstids=bb.agent[bb.release<=s+1e-9].to_numpy();firstshares.append(shares(typ[firstids],3))
   a=np.searchsorted(tr['time'],s-1e-10);z=np.searchsorted(tr['time'],e+1e-10,side='right');tt=tr[a:z]
   # Exact eligible presence (state zero), not interpolation of sparse positions.
   eligible=np.unique(tt['agent']);missing=eligible[~np.isin(eligible,ids)]
   ss=np.flatnonzero((t>=s)&(t<=e));anyband=np.unique(np.where(d['pos'][ss,:,2]>=2.7)[1]) if len(ss) else np.array([],int);samplemiss=anyband[~np.isin(anyband,ids)]
   qs=[];mts=[]
   for fish in missing:
    fq=tt['tank'][tt['agent']==fish][0];qs.append(fq);mts.append(typ[fish]);nonrows.append(dict(seed=seed,event_id=j,event_class=c,agent=fish,type=typ[fish],reserve=fq))
   qs=np.asarray(qs);mts=np.asarray(mts)
   for k in [-1,0,1,2]:
    sel=np.ones(len(bb),bool) if k<0 else bb.type.to_numpy()==k;sub=b.loc[bb.index[sel]];miss=qs if k<0 else qs[mts==k];el=eligible if k<0 else eligible[typ[eligible]==k]
    row=dict(seed=seed,event_id=j,event_class=int(c),type=k,n_participants=len(sub),n_eligible=len(el),n_nonbreathers=len(miss),fraction_nonbreathers=len(miss)/len(el) if len(el) else np.nan,nonbreather_reserve=mean(miss),sampled_any_state_nonbreathers=len(samplemiss) if k<0 else np.sum(typ[samplemiss]==k),reserve=mean(sub.reserve),need_fraction=mean(sub.reserve.to_numpy()<.75),full_fraction=mean(sub.reserve.to_numpy()>=.75),carried_fraction=mean(sub.attribution.to_numpy()==1),no_upward_fraction=mean(sub.no_upward_need))
    for ci,sh in enumerate(shares(sub.category,4)):row[f'category_{ci}']=sh
    eventrows.append(row)
 out=dict(seed=seed,arm=arm,seconds=T,N=N,sha256=rec['sha256'],counts=dict(trials_total=len(tr),entries_total=len(en),episodes_total=len(ep),trials_window=int(tk.sum()),visits_window=int(vk.sum()),breaths_window=len(b),visits_excluded=int((~vk).sum()),breaths_excluded=int((~bk).sum())),probability_max_error=maxerr,types={},first_type_shares=np.mean(firstshares,axis=0) if firstshares else None)
 trialcurves={}
 for k in ([-1,0,1,2] if N>1 else [int(typ[0])]):
  name='All' if k<0 else TYPES[k];bb=b if k<0 else b[b.type==k];vv=v[v.complete & ((v.type==k) if k>=0 else True)];tt=tr[tk & ((typ[tr['agent']]==k) if k>=0 else True)]
  rates=[];expect=[];ns=[];residual=[]
  for lo,hi in zip(TB[:-1],TB[1:]):
   sel=(tt['tank']>=lo)&(tt['tank']<(hi if hi<1 else 1.000001));ns.append(int(sel.sum()));rates.append(mean(tt['accepted'][sel]));expect.append(mean(tt['probability'][sel]));residual.append(mean(tt['accepted'][sel]-tt['probability'][sel]))
  visits_by_cat=[]
  for ci in range(4):
   sub=vv[cats(vv.reserve)==ci];visits_by_cat.append(dict(n=len(sub),refusal_fraction=mean(~sub.accepted),trials_mean=mean(sub.n_trials)))
  interval=[]
  for fish in (range(N) if k<0 else np.flatnonzero(typ==k)):
   ee=np.sort(ep[ep['agent']==fish],order='release');ok=(ee['exit'][:-1]>=BURN)&(ee['release'][1:]<=end);interval.extend((ee['release'][1:]-ee['exit'][:-1])[ok])
  out['types'][name]=dict(n_breaths=len(bb),reserve_mean=mean(bb.reserve),reserve_median=float(bb.reserve.median()),reserve_quantiles=np.quantile(bb.reserve,[.05,.25,.5,.75,.95]),categories=shares(bb.category,4),need_fraction=mean(bb.reserve.to_numpy()<.75),full_fraction=mean(bb.reserve.to_numpy()>=.75),attribution=shares(bb.attribution,3),no_upward_fraction=mean(bb.no_upward_need),strict_carried_fraction=mean(bb.no_upward_need & bb.rising_pre_onset),rising_fraction=mean(bb.rising_through_event),need_term_mean=mean(bb.need_term),need_term_quantiles=np.quantile(bb.need_term,[.05,.5,.95]),entry_to_accept_mean=mean(bb.decision-bb.entry_time),rank_mean=mean(bb.height_rank_pre3),rank_by_attribution=[mean(bb.height_rank_pre3[bb.attribution==j]) for j in range(3)],entry_to_rank_lag_mean=mean(bb.rank_lag_seconds),n_visits=len(vv),trials_per_visit=mean(vv.n_trials),trials_per_visit_quantiles=np.quantile(vv.n_trials,[.5,.9,.99]),refusal_fraction=mean(~vv.accepted),visits_by_category=visits_by_cat,acceptance_per_trial=mean(tt['accepted']),acceptance_expected=mean(tt['probability']),acceptance_residual=mean(tt['accepted']-tt['probability']),individual_interval_mean=mean(interval),density=hist(bb.reserve),trial_rates=rates,trial_expected=expect,trial_residual=residual,trial_n=ns)
 if N>1:
  er=pd.DataFrame(eventrows);out['events']={};out['order']={}
  for k in [-1,0,1,2]:
   name='All' if k<0 else TYPES[k];out['events'][name]={};out['order'][name]={}
   for c in [0,1,2,3,4]:
    sub=er[(er.type==k)&((er.event_class==c) if c else True)]
    out['events'][name][str(c)]={col:mean(sub[col].dropna()) for col in sub if col not in ['seed','event_id','event_class','type']};out['events'][name][str(c)]['n_events']=len(sub)
   for stage in ['First','Early','Middle','Late','First10']:
    sel=b.first_tied if stage=='First' else b.first10 if stage=='First10' else b.release_stage==['Early','Middle','Late'].index(stage)
    ss=b[sel & ((b.type==k) if k>=0 else True)];groups=list(ss.groupby('event_id'))
    # Equal event means within seed; absent type/stage is missing, never zero.
    out['order'][name][stage]=dict(n_events=len(groups),n_participants=len(ss),reserve=mean([mean(x.reserve) for _,x in groups]),categories=np.mean([shares(x.category,4) for _,x in groups],axis=0) if groups else np.full(4,np.nan),full_fraction=mean([mean(x.reserve>=.75) for _,x in groups]),carried_fraction=mean([mean(x.attribution==1) for _,x in groups]))
  er.to_csv(HERE/'analysis'/f'{path.stem}_events.csv.gz',index=False)
  pd.DataFrame(nonrows).to_csv(HERE/'analysis'/f'{path.stem}_nonbreathers.csv.gz',index=False)
 b.to_csv(HERE/'analysis'/f'{path.stem}_breaths.csv.gz',index=False)
 v[v.complete].to_csv(HERE/'analysis'/f'{path.stem}_visits.csv.gz',index=False)
 write(HERE/'analysis'/f'{path.stem}_summary.json',out)
 return out

def combine(rows,repro):
 def recurse(xs):
  if isinstance(xs[0],dict):return {k:recurse([x[k] for x in xs]) for k in xs[0]}
  if isinstance(xs[0],list) and xs[0] and isinstance(xs[0][0],dict):return [recurse([x[i] for x in xs]) for i in range(len(xs[0]))]
  return agg(xs)
 shoal=[r for r in rows if r['arm']=='Shoal'];solo=[r for r in rows if r['arm']=='Alone']
 out=dict(metadata=dict(new_simulations=0,burn=BURN,end_margin=2,dt=DT,reaction_interval=.05,decision_timescale=.3,fermi_midpoint=.8,fermi_width=.1,need_setpoint=.75,upward_strength=.35,need_down=.1,surface_strength=.3,surface_layer=1.5,dilution_min=.2,dilution_scale=6,vision_k=24,rest_strength=.3,rest_height=1,band_height=2.7,release_height=2.8,rank_target_lag=3,type_names=TYPES,category_names=CATS,category_edges=[0,.5,.75,.8,1],attribution_names=ATTR,reserve_bins=QB,trial_bins=TB,shoal_seeds=[r['seed'] for r in shoal],solo_seeds=sorted(set(r['seed'] for r in solo)),n_shoal=len(shoal),n_solo=len(solo),probability_examples={str(q):float(decision_probability(q,DT,.8,.1,.3)) for q in [0,.5,.75,.8,.95,1]}),reproduction=repro,shoal={},alone={},contrasts={},events={},order={},first_type_shares=agg([r['first_type_shares'] for r in shoal]),per_run=rows)
 for name in ['All']+TYPES:
  out['shoal'][name]=recurse([r['types'][name] for r in shoal])
  if name!='All':
   ss=[r['types'][name] for r in solo if name in r['types']];out['alone'][name]=recurse(ss)
   am=mean([r['reserve_mean'] for r in ss]);im=mean([r['individual_interval_mean'] for r in ss]);out['contrasts'][name]=dict(reserve_shoal_minus_alone_mean=agg([r['types'][name]['reserve_mean']-am for r in shoal]),interval_alone_over_shoal=agg([im/r['types'][name]['individual_interval_mean'] for r in shoal]))
  out['events'][name]=recurse([r['events'][name] for r in shoal]);out['order'][name]=recurse([r['order'][name] for r in shoal])
 out['qa']=dict(probability_max_error=max(r['probability_max_error'] for r in rows),total_trials_checked=sum(r['counts']['trials_total'] for r in rows),total_breaths_linked=sum(r['counts']['episodes_total'] for r in rows),shoal_breaths_retained=sum(r['counts']['breaths_window'] for r in shoal),solo_breaths_retained=sum(r['counts']['breaths_window'] for r in solo),records_hash_verified=len(rows),canonical_reproduction_passed=True)
 out['inputs']=INPUTS
 write(HERE/'analysis/summary.json',out)
 print('Complete',out['qa'],flush=True)
if __name__=='__main__':
 repro=reproduction();rows=[]
 for p in sorted((BASE/'W1_reference/trajectories').glob('reference_s90*.npz'))+sorted((BASE/'W8_results_gaps/trajectories').glob('solo_t*_s96*.npz')):rows.append(process(p))
 assert len(rows)==40
 for p in [Path(__file__),BASE/'W1_reference/analysis_core.py',BASE/'W5b_code/rz75_model.py',ROOT/'study_out/local_work/rest_zone_step25/model25.py']:source(p)
 combine(rows,repro)
