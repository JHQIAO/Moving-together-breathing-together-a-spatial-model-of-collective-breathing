"""Saved-data-only W10 analysis; all writers are confined to this folder."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
ROOT=HERE.parents[2];BASE=HERE.parent
sys.path[:0]=[str(ROOT),str(BASE/'W1_reference')]
import numpy as np
from scipy.stats import rankdata,spearmanr
from scipy.spatial import cKDTree
from analysis_core import event_arrays,summarize_arrays,write_json,digest
import argparse,time,warnings
warnings.filterwarnings('ignore',message='Mean of empty slice')
TYPES=['Fast','Medium','Slow'];PHASES=['overall','rest','rise','event','descent','unclassified'];SCOPES=['schooling','all']
B=499

def div(a,b):
 return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)>0)
def agg(v):
 a=np.asarray(v,float);return dict(mean=np.nanmean(a,axis=0),min=np.nanmin(a,axis=0),max=np.nanmax(a,axis=0),n=np.sum(np.isfinite(a),axis=0))
def reproduction():
 paths=sorted((ROOT/'study_out/local_work/rest_zone_step25/outputs/trajectories').glob('A_RZ75_s73*.npz'))
 assert len(paths)==3
 rows=[]
 for p in paths:
  with np.load(p) as d:rows.append(summarize_arrays(event_arrays(d,float(d['time'][-1]))))
 expected={'gap_mean':14.647,'gap_sd':6.056,'short_fraction':.0728,'long_fraction':.0214,'class_1':.1271,'class_4':.0085,'slab_0.5_mean':2.323,'individual_mean':48.854,'class_next_rho':.172}
 out={}
 for k,e in expected.items():
  v=float(np.mean([r[k] for r in rows]));tol=.00051 if k not in ['short_fraction','long_fraction','class_1','class_4'] else .000051
  assert abs(v-e)<tol,(k,v,e)
  out[k]=dict(observed=v,expected=e,tolerance=tol)
 write_json(HERE/'qa/reproduction.json',dict(passed=True,metrics=out,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in paths]))
 return out

def phases(t,z,ev):
 med=np.median(z,axis=1);up=np.flatnonzero((med[:-1]<1)&(med[1:]>=1))+1
 down=np.flatnonzero((med[:-1]>=1)&(med[1:]<1))+1
 ph=np.full(len(t),5,int);ph[med<1]=1
 # Each above-threshold excursion is a rise only if followed by an event
 # before it next returns below threshold. Event windows take precedence.
 for u in up:
  nxt=down[down>u];end=t[nxt[0]] if len(nxt) else t[-1]+1
  starts=ev['start'][(ev['start']>=t[u])&(ev['start']<end)]
  if len(starts):ph[(t>=t[u])&(t<starts[0])]=2
 for e in ev['end']:
  ix=np.searchsorted(t,e)
  if ix>=len(t) or med[ix]<1:continue
  nxt=down[down>=ix];end=t[nxt[0]] if len(nxt) else t[-1]+1
  ph[(t>e)&(t<end)]=4
 for s,e in zip(ev['start'],ev['end']):ph[(t>=s)&(t<=e)]=3
 return ph,up,med

def process(path,seed,highres=False):
 start_clock=time.monotonic();print('START',path.name,flush=True)
 with np.load(path) as f:d={k:f[k] for k in ['time','pos','heading','tank','status','type_id','episodes13','events','agent_consumption_rates']}
 t=d['time'].astype(float);T=float(t[-1]);typ=d['type_id'];N=len(typ);counts=np.bincount(typ,minlength=3)
 assert N==200 and counts.tolist()==[67,67,66]
 assert np.all(np.diff(t)>0) and np.all(np.isfinite(d['pos']))
 ev=event_arrays(d,T);ph,up,med=phases(t,d['pos'][:,:,2],ev)
 keep=(t>=300)&(t<T-2);up=up[(t[up]>=300)&(t[up]<T-2)]
 out=dict(seed=seed,seconds=T,sample_step=float(np.median(np.diff(t))),type_counts=counts,n_events=len(ev['start']),n_liftoffs=len(up),phase_counts={PHASES[p]:int(np.sum(keep&(ph==p))) for p in range(1,6)},states={str(s):int(np.sum(d['status'][keep]==s)) for s in range(5)},scopes={},events={})
 rng=np.random.default_rng(seed+100000)
 perms=np.vstack([typ]+[rng.permutation(typ) for _ in range(B)])
 P=np.stack([perms==k for k in range(3)],axis=1).astype(float) # draw,type,fish
 nulls={};curves={};ts={};weights={}
 def ratio_metric(key,num,den):
  vals=div(P@np.asarray(num,float),P@np.asarray(den,float))
  nulls[key]=vals
  return vals[0]
 z=d['pos'][:,:,2].astype(float);q=d['tank'].astype(float)
 # Whole-shoal crossings match W2; no event completeness dependence for positions.
 dt=float(np.median(np.diff(t)));lags=np.arange(-15,15+dt/2,dt);li=np.rint(lags/dt).astype(int)
 valid_up=up[(t[up]>=315)&(t[up]<=T-17)]
 hb=np.linspace(0,3.3,67);rb=np.linspace(0,1,21);qb=np.linspace(0,1,21)
 curves.update(height_edges=hb,rank_edges=rb,reserve_edges=qb,lags=lags)
 geometry={}
 # Schooling centroid and mean heading define the school's horizontal frame.
 school=d['status']==0;ns=school.sum(1)
 xy=d['pos'][:,:,:2].astype(float);box=np.array([10.,3.3])
 cm=div((np.exp(2j*np.pi*xy/box)*school[:,:,None]).sum(1),ns[:,None]) if False else (np.exp(2j*np.pi*xy/box)*school[:,:,None]).sum(1)/np.maximum(ns[:,None],1)
 center=(np.angle(cm)%(2*np.pi))*box/(2*np.pi)
 rel=xy-center[:,None,:];rel-=box*np.round(rel/box)
 spans=np.zeros((len(t),2))
 for i in np.flatnonzero(keep):
  for axis,L in enumerate(box):
   x=np.sort(xy[i,school[i],axis]);spans[i,axis]=L-np.max(np.diff(np.r_[x,x[0]+L])) if len(x)>1 else 0
 h=(d['heading'][:,:,:2]*school[:,:,None]).sum(1)/np.maximum(ns[:,None],1)
 hnorm=np.linalg.norm(h,axis=1);unit=div(h,hnorm[:,None])
 compact=(spans<=box/2).all(1)&(np.abs(cm)>.2).all(1)&(hnorm>=.2)&(ns>24)
 centroid_ok=(np.abs(cm)>.2).all(1)&(ns>24)
 front=np.sum(rel*unit[:,None,:],axis=2);side=rel[:,:,0]*(-unit[:,None,1])+rel[:,:,1]*unit[:,None,0]
 radius=np.linalg.norm(rel,axis=2)
 out['geometry']=dict(compact_fraction=float(compact[keep].mean()),centroid_defined_fraction=float(centroid_ok[keep].mean()),span_gt_half_fraction=np.mean(spans[keep]>box/2,axis=0),mean_span=np.mean(spans[keep],axis=0),circular_strength_mean=np.mean(np.abs(cm[keep]),axis=0),horizontal_heading_strength_mean=float(hnorm[keep].mean()))
 for scope in SCOPES:
  m=(school if scope=='schooling' else np.ones_like(school))&keep[:,None]
  ranks=rankdata(np.where(m,z,np.nan),axis=1,nan_policy='omit',method='average')
  ranks=div(ranks-1,m.sum(1)[:,None]-1)
  assert np.allclose(np.nanmean(ranks[keep],axis=1),.5)
  mets={'height':z,'rank':ranks,'reserve':q,'radius':np.where(centroid_ok[:,None],radius,np.nan),'compact_radius':np.where(compact[:,None],radius,np.nan),'front':np.where(compact[:,None],front,np.nan),'side':np.where(compact[:,None],side,np.nan),'abs_side':np.where(compact[:,None],abs(side),np.nan)}
  scopeout={};curve_h=[];curve_r=[]
  for phase,pid in zip(PHASES,range(6)):
   mm=m if pid==0 else m&(ph==pid)[:,None]
   ss={}
   for key,v in mets.items():
    good=mm&np.isfinite(v);num=np.where(good,v,0).sum(0);den=good.sum(0)
    ss[key]=ratio_metric(f'{scope}/{phase}/{key}',num,den)
   ss['fish_samples']=np.array([np.sum(mm[:,typ==k]) for k in range(3)])
   hs=[];rs=[]
   for k in range(3):
    sel=mm&(typ==k)[None,:];a=z[sel];b=ranks[sel]
    hc=np.histogram(a,hb)[0];rc=np.histogram(b,rb)[0]
    hs.append(div(hc,len(a)*np.diff(hb)));rs.append(div(rc,len(b)*np.diff(rb)))
    assert hc.sum()==len(a) and rc.sum()==len(b)
   curve_h.append(hs);curve_r.append(rs);scopeout[phase]=ss
  curves[scope+'_height_density']=curve_h;curves[scope+'_rank_density']=curve_r
  curves[scope+'_lift_height']=np.array([np.nanmean(div(np.where(m[:,typ==k],z[:,typ==k],0).sum(1),m[:,typ==k].sum(1))[valid_up[:,None]+li],axis=0) for k in range(3)])
  curves[scope+'_lift_rank']=np.array([np.nanmean(np.nanmean(np.where(m[:,typ==k],ranks[:,typ==k],np.nan),axis=1)[valid_up[:,None]+li],axis=0) for k in range(3)])
  qc=[];qn=[];corr=[]
  for k in range(3):
   mask=m&(typ==k)[None,:];a=q[mask];b=ranks[mask]
   ids=np.clip(np.digitize(a,qb)-1,0,len(qb)-2);nn=np.bincount(ids,minlength=len(qb)-1);vv=np.bincount(ids,weights=b,minlength=len(qb)-1)
   qc.append(div(vv,nn));qn.append(nn);corr.append(float(spearmanr(a,b).statistic))
  curves[scope+'_reserve_rank']=qc;curves[scope+'_reserve_count']=qn
  scopeout['reserve_rank_spearman']=corr
  # Regression via sufficient statistics; exact label permutations preserve all
  # trajectories and autocorrelation. Slow is reference; no fish-time SEs.
  base=np.stack([np.ones_like(q),.75-q,np.maximum(.75-q,0)]+[np.broadcast_to((ph==j)[:,None],q.shape) for j in range(2,6)],axis=2)
  X=base[m];y=ranks[m];xtx=X.T@X;xty=X.T@y
  fish_base=(base*m[:,:,None]).sum(0);fish_y=np.where(m,ranks,0).sum(0);fish_n=m.sum(0)
  fits=[]
  for pp in P:
   dummy=pp[:2].T;cross=fish_base.T@dummy;dn=dummy.T@fish_n
   mat=np.block([[xtx,cross],[cross.T,np.diag(dn)]])
   rhs=np.r_[xty,dummy.T@fish_y]
   fits.append(np.linalg.solve(mat,rhs))
  fits=np.array(fits);scopeout['regression']=dict(columns=['intercept','signed_deficit','positive_deficit','rise','event','descent','unclassified','Fast_vs_Slow','Medium_vs_Slow'],coef=fits[0],fish_samples=len(y))
  nulls[scope+'/adjusted_type']=fits[:,-2:]
  # Linear version requested; phase-adjusted hinge above checks a key nonlinearity.
  cols=[0,1,7,8] # selected from assembled full matrix only in the last loop: reassemble observed
  pp=P[0];dummy=pp[:2].T;cross=fish_base.T@dummy;dn=dummy.T@fish_n
  mat=np.block([[xtx,cross],[cross.T,np.diag(dn)]]);rhs=np.r_[xty,dummy.T@fish_y]
  scopeout['simple_regression']=dict(columns=['intercept','signed_deficit','Fast_vs_Slow','Medium_vs_Slow'],coef=np.linalg.solve(mat[np.ix_(cols,cols)],rhs[cols]))
  out['scopes'][scope]=scopeout
  del base,X,y
  # Full 1 s samples: periodic 3D nearest neighbours, local graph retained only as aggregates.
  # The high-resolution validation tests timing only, not another neighbour analysis.
  if not highres:
   adjacency=np.zeros((6,2,N,N));timevals=[];timeidx=[]
   for i in np.flatnonzero(keep):
    ids=np.flatnonzero(m[i]);p=d['pos'][i,ids].astype(float);p[:,:2]%=box
    assert len(ids)>24
    nn=cKDTree(p,boxsize=[10.,3.3,100.]).query(p,k=25,workers=1)[1][:,1:]
    assert not np.any(nn==np.arange(len(ids))[:,None])
    vals=[]
    for ki,k in enumerate([6,24]):
     a=np.repeat(ids,k);b=ids[nn[:,:k].ravel()]
     np.add.at(adjacency[0,ki],(a,b),1/(len(ids)*k))
     np.add.at(adjacency[ph[i],ki],(a,b),1/(len(ids)*k))
     observed=float(np.mean(typ[a]==typ[b]));nc=np.bincount(typ[ids],minlength=3)
     expected=float(np.sum(nc*(nc-1))/(len(ids)*(len(ids)-1)))
     vals.extend([observed,expected])
    timevals.append(vals);timeidx.append(i)
   vv=np.array(timevals);timeidx=np.array(timeidx)
   scopeout['neighbours']={}
   for pid,phase in enumerate(PHASES):
    ii=np.ones(len(timeidx),bool) if pid==0 else ph[timeidx]==pid
    count=int(ii.sum());nr={}
    for ki,k in enumerate([6,24]):
     a=adjacency[pid,ki]/max(count,1)
     # Fixed identity shuffle, marginally a uniform label shuffle at every sample.
     values=np.zeros(B+1)
     for j in range(3):values+=np.einsum('bi,ij,bj->b',P[:,j],a,P[:,j],optimize=True)
     nulls[f'{scope}/{phase}/nn{k}']=values
     nr[str(k)]=dict(observed=values[0],fixed_identity_null_mean=float(values[1:].mean()),same_sample_composition_null=float(vv[ii,ki*2+1].mean()) if count else np.nan,samples=count)
     assert not count or abs(values[0]-vv[ii,ki*2].mean())<1e-10
    scopeout['neighbours'][phase]=nr
   block=np.floor((t[timeidx]-300)/60).astype(int)
   ts[scope]=dict(time=[float(t[timeidx][block==b].mean()) for b in np.unique(block)],values=[vv[block==b].mean(0) for b in np.unique(block)],columns=['k6_observed','k6_sample_null','k24_observed','k24_sample_null'])
  # event weights make nulls exact for the event-equal shares.
  top=np.zeros(N);topden=np.zeros(N);top_count=0
  for i in up:
   ids=np.flatnonzero(m[i]);n=max(1,int(np.ceil(.1*len(ids))));sel=ids[np.argsort(z[i,ids],kind='stable')[-n:]]
   top[sel]+=1/n;topden[ids]+=1/len(ids);top_count+=1
  val=(P@top)/max(top_count,1);nulls[scope+'/top_share']=val
  out['events'][scope+'_top_share']=val[0]
  out['events'][scope+'_top_available_share']=(P[0]@topden)/max(top_count,1)
 ep=np.sort(d['episodes13'],order='release');event_weights={k:np.zeros(N) for k in ['first','first10','first_halfsecond']+[f'participants_class{c}' for c in range(1,5)]};nweight={k:0 for k in event_weights}
 missed={s:np.zeros(N) for s in SCOPES};ready={s:np.zeros(N) for s in SCOPES};matched=0;delay=[]
 for s,e,c in zip(ev['start'],ev['end'],ev['classes']):
  a=np.searchsorted(ep['release'],s-1e-10);b=np.searchsorted(ep['release'],e+1e-10,side='right');ee=ep[a:b]
  ids=ee['agent'];uniq=np.unique(ids);firstids=np.unique(ids[np.isclose(ee['release'],s,rtol=0,atol=1e-9)])
  _,order=np.unique(ids,return_index=True);firstten=ids[np.sort(order)[:10]];half=np.unique(ids[ee['release']<=s+.5+1e-10])
  for key,sel in [('first',firstids),('first10',firstten),('first_halfsecond',half),(f'participants_class{c}',uniq)]:
   event_weights[key][sel]+=1/len(sel);nweight[key]+=1
  i=np.searchsorted(t,s,side='right')-1;delay.append(s-t[i]);participates=np.isin(np.arange(N),uniq)
  for scope in SCOPES:
   eligible=(q[i]<.8)&((d['status'][i]==0) if scope=='schooling' else True)
   ready[scope]+=eligible;missed[scope]+=eligible&~participates
  before=up[t[up]<=s]
  if len(before) and not np.any((t>t[before[-1]])&(t<s)&(med<1)):matched+=1
 for key,v in event_weights.items():
  val=(P@v)/max(nweight[key],1);val[:]=np.nan if nweight[key]==0 else val
  nulls['events/'+key]=val;out['events'][key]=val[0];out['events'][key+'_n']=nweight[key]
 for scope in SCOPES:
  out['events'][scope+'_miss_ready']=ratio_metric('events/'+scope+'_miss_ready',missed[scope],ready[scope])
  out['events'][scope+'_ready_opportunities']=P[0]@ready[scope]
  out['events'][scope+'_missed_ready_counts']=P[0]@missed[scope]
 out['events']['matched_rise_events']=matched;out['events']['pre_onset_sample_delay_mean']=float(np.mean(delay))
 out['aligned_windows']=len(valid_up)
 prefix='highres' if highres else 'reference'
 write_json(HERE/f'analysis/{prefix}_s{seed}.json',out)
 np.savez_compressed(HERE/f'analysis/{prefix}_s{seed}_curves.npz',**{k:np.asarray(v) for k,v in curves.items()})
 np.savez_compressed(HERE/f'analysis/{prefix}_s{seed}_nulls.npz',**nulls)
 if ts:write_json(HERE/f'analysis/{prefix}_s{seed}_timeseries.json',ts)
 write_json(HERE/f'qa/{prefix}_s{seed}_source.json',dict(path=str(path.relative_to(ROOT)),sha256=digest(path),analysis_sha256=digest(Path(__file__)),elapsed_seconds=time.monotonic()-start_clock))
 print('DONE',path.name,round(time.monotonic()-start_clock,1),'s',flush=True)

def combine():
 import json
 rows=[json.loads((HERE/f'analysis/reference_s{s}.json').read_text()) for s in range(9001,9011)]
 def rec(items):
  if isinstance(items[0],dict):return {k:rec([v[k] for v in items]) for k in items[0]}
  if isinstance(items[0],str) or (isinstance(items[0],list) and items[0] and isinstance(items[0][0],str)):return items[0]
  return agg(items)
 ensemble=rec(rows)
 files=[np.load(HERE/f'analysis/reference_s{s}_nulls.npz') for s in range(9001,9011)];null={};tests=[]
 for k in files[0].files:
  vals=np.mean([f[k] for f in files],axis=0)
  null[k]=dict(observed=vals[0],null_mean=np.mean(vals[1:],axis=0),null_025=np.quantile(vals[1:],.025,axis=0),null_975=np.quantile(vals[1:],.975,axis=0))
  if vals.ndim==2 and vals.shape[1]==3:
   contrast=vals[:,0]-vals[:,2];observed=contrast[0];center=np.mean(contrast[1:]);p=(1+np.sum(abs(contrast[1:]-center)>=abs(observed-center)))/(B+1)
   null[k]['Fast_minus_Slow']=dict(observed=observed,null_mean=center,null_025=np.quantile(contrast[1:],.025),null_975=np.quantile(contrast[1:],.975),p_two_sided=p)
   tests.append((k,p))
  elif vals.ndim==1:
   center=np.mean(vals[1:]);p=(1+np.sum(abs(vals[1:]-center)>=abs(vals[0]-center)))/(B+1);null[k]['p_two_sided']=p;tests.append((k,p))
  else:
   p=(1+np.sum(abs(vals[1:]-np.mean(vals[1:],axis=0))>=abs(vals[0]-np.mean(vals[1:],axis=0)),axis=0))/(B+1);null[k]['p_two_sided']=p
 # exploratory multiplicity correction over scalar fast-slow and overall sorting contrasts.
 ps=np.array([v for _,v in tests]);order=np.argsort(ps);bh=np.empty_like(ps);bh[order]=np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1].clip(0,1)
 for (k,_),v in zip(tests,bh):null[k]['BH_q']=v
 for f in files:f.close()
 fc=[np.load(HERE/f'analysis/reference_s{s}_curves.npz') for s in range(9001,9011)]
 curves={k:(fc[0][k] if k in ['height_edges','rank_edges','reserve_edges','lags'] else agg([f[k] for f in fc])) for k in fc[0].files}
 for f in fc:f.close()
 metadata=dict(n_seeds=10,seeds=list(range(9001,9011)),burn_seconds=300,seconds=13440,analysis_end_margin_seconds=2,type_names=TYPES,phases=PHASES,scopes=SCOPES,school_state=0,setpoint=.75,ready_threshold=.8,k=[6,24],top_fraction=.1,shuffle_repeats=B,shuffle_seed_offset=100000,type_counts=[67,67,66],null_method='Uniform type permutation among all identities, same mapping at every sample; independent permutations per seed, equal-seed aggregate. Preserves trajectory autocorrelation. Per-sample schooling-composition analytic null also reported.',rank_definition='(midrank-1)/(number present-1), ties averaged; schooling rank among state-0 fish, all rank among all fish.',regression='OLS separately per seed, Slow reference. Primary simple signed deficit + type; sensitivity adds positive deficit hinge and cycle phase. Descriptive conditional association, no independent fish-time errors.',phase_definition='Sampled whole-shoal median upward crossing of 1 BL; rest below 1 BL outside events, rise from upward crossing to first onset before next downcrossing, descent after event end to next downcrossing; events take precedence; remaining above-threshold samples unclassified.',horizontal_definition='Schooling circular centroid, minimum-image xy offsets. Front/side and compact radius only if both minimal covering arcs <= half-period, both circular strengths >0.2, horizontal mean-heading norm >=0.2, and >24 schooling fish. Toroidal radius only requires circular strength >0.2 and >24 schooling fish.',event_definition='Canonical gap=1 s; start>=300 and end<=T-2. Shares averaged equally per event within seed. Earliest tied releasers split weight; first10 distinct and first0.5 s sensitivity. Ready misses use last saved sample at/before onset and completed event membership.',new_simulations=0)
 high=json.loads((HERE/'analysis/highres_s9001.json').read_text()) if (HERE/'analysis/highres_s9001.json').exists() else None
 write_json(HERE/'analysis/summary.json',dict(metadata=metadata,ensemble=ensemble,per_seed=rows,nulls=null,curves=curves,highres=high,reproduction=json.loads((HERE/'qa/reproduction.json').read_text())))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--highres',action='store_true');ap.add_argument('--combine',action='store_true');ap.add_argument('--reproduction',action='store_true');a=ap.parse_args()
 if a.reproduction:reproduction()
 elif a.combine:combine()
 elif a.highres:process(BASE/'W1_reference/trajectories/highres_s9001.npz',9001,True)
 else:
  for s in ([a.seed] if a.seed else range(9001,9011)):process(BASE/f'W1_reference/trajectories/reference_s{s}.npz',s)
