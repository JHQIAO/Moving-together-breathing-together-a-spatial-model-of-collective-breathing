"""Canonical release-to-exit bout statistics and saved-trajectory diagnostics."""
from config import *
from breathing_study import bout_events_from
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def mean(x):return float(np.mean(x)) if len(x) else np.nan
def sd(x):return float(np.std(x,ddof=1)) if len(x)>1 else np.nan
def rho(x,y):
    return float(stats.spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan

def canonical(d,N,T,burn=300.):
    e=d['episodes13'];e=e[np.argsort(e['release'],kind='stable')]
    b=bout_events_from(e['release'],e['exit'],e['agent'],gap=1.)
    b=[v for v in b if v['t_start']>=burn and v['t_end']<=T-2.]
    start=np.array([v['t_start'] for v in b]);end=np.array([v['t_end'] for v in b])
    part=np.array([v['n']/N for v in b]);cls=np.digitize(part,[.2,.5,.8])+1
    g=start[1:]-end[:-1];slab=[]
    for a,z in zip(start,end):
        v=e[(e['release']>=a-1e-10)&(e['release']<=z+1e-10)&(e['release_y']<=.5)]
        if len(v):slab.append(float(v['exit'].max()-v['release'].min()))
    individual=[];bytype=[[],[],[]]
    for i in range(N):
        v=e[e['agent']==i];ok=(v['exit'][:-1]>=burn)&(v['release'][1:]<=T)
        x=(v['release'][1:]-v['exit'][:-1])[ok];individual.extend(x);bytype[int(d['type_id'][i])].extend(x)
    m=dict(n_bouts=len(b),n_gaps=len(g),gap_mean=mean(g),gap_sd=sd(g),short_fraction=mean(g<5),long_fraction=mean(g>30),
      participation_mean=mean(part),slab_mean=mean(slab),whole_duration_mean=mean(end-start),individual_mean=mean(individual),
      individual_sd=sd(individual),class_next_rho=rho(cls[:-1],g),class_previous_rho=rho(cls[1:],g))
    for i,n in enumerate(['small','medium','large','huge'],1):m[n+'_fraction']=mean(cls==i)
    for n,v in zip(['fast','medium','slow'],bytype):m['individual_'+n]=mean(v);m['individual_'+n+'_n']=len(v)
    # Cast samples back to float64 for reductions; keep physical 1 s sampling.
    t=d['time'].astype(float);ok=(t>=burn)&(t<T)
    median=np.median(d['pos'][:,:,2].astype(float),axis=1)
    peaks=find_peaks(uniform_filter1d(median,3,mode='nearest'),prominence=.2)[0]
    cg=np.diff(t[peaks[ok[peaks]]])
    pol=[np.linalg.norm(h[s==0].astype(float).mean(axis=0)) for h,s in zip(d['heading'][ok],d['status'][ok]) if np.any(s==0)]
    m.update(crest_interval_mean=mean(cg),crest_interval_median=float(np.median(cg)) if len(cg) else np.nan,
      crest_interval_iqr=float(np.subtract(*np.quantile(cg,[.75,.25]))) if len(cg) else np.nan,
      crest_count=int(np.sum(ok[peaks])),polarization=mean(pol))
    return clean(m),dict(gaps=g,parts=part,classes=cls,slab=np.array(slab),individual=np.array(individual),crest_gaps=cg)
