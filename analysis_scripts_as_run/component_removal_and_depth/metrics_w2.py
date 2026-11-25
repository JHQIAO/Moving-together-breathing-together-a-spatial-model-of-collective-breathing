"""Canonical event metrics, using the step-13 bout definition and censoring."""
from model_w2 import *
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from breathing_study import bout_events_from

def mean(x):return float(np.mean(x)) if len(x) else np.nan
def sd(x):return float(np.std(x,ddof=1)) if len(x)>1 else np.nan
def rho(x,y):
    return float(stats.spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan

def events(data,N,T,burn=300.):
    ep=data['episodes13'];ep=ep[np.argsort(ep['release'],kind='stable')]
    bouts=bout_events_from(ep['release'],ep['exit'],ep['agent'],gap=1.)
    bouts=[b for b in bouts if b['t_start']>=burn and b['t_end']<=T-2.]
    start=np.array([b['t_start'] for b in bouts]);end=np.array([b['t_end'] for b in bouts])
    part=np.array([b['n']/N for b in bouts]);cls=np.digitize(part,[.2,.5,.8])+1
    slab=[]
    for a,b in zip(start,end):
        rows=ep[(ep['release']>=a-1e-10)&(ep['release']<=b+1e-10)&(ep['release_y']<=.5)]
        if len(rows):slab.append(float(rows['exit'].max()-rows['release'].min()))
    indiv=[];bytype=[[],[],[]]
    for i in range(N):
        e=ep[ep['agent']==i];ok=(e['exit'][:-1]>=burn)&(e['release'][1:]<=T)
        g=(e['release'][1:]-e['exit'][:-1])[ok]
        indiv.extend(g);bytype[int(data['type_id'][i])].extend(g)
    return dict(start=start,end=end,part=part,cls=cls,gaps=start[1:]-end[:-1],slab=np.array(slab),individual=np.array(indiv),bytype=[np.array(v) for v in bytype])

def canonical(data,N,T,burn=300.):
    e=events(data,N,T,burn);g=e['gaps'];part=e['part'];cls=e['cls']
    ep=data['episodes13'];n_breaths=int(np.sum((ep['release']>=burn)&(ep['exit']<=T)))
    m=dict(n_breaths=n_breaths,completed_breaths_per_fish_hour=n_breaths/N*3600/(T-burn),n_bouts=len(part),n_gaps=len(g),gap_mean=mean(g),gap_sd=sd(g),short_fraction=mean(g<5),long_fraction=mean(g>30),
      participation_mean=mean(part),slab_mean=mean(e['slab']),whole_duration_mean=mean(e['end']-e['start']),individual_mean=mean(e['individual']),
      individual_sd=sd(e['individual']),class_next_rho=rho(cls[:-1],g),class_previous_rho=rho(cls[1:],g))
    for code,name in enumerate(['small','medium','large','huge'],1):m[name+'_fraction']=mean(cls==code)
    for name,v in zip(['fast','medium','slow'],e['bytype']):m['individual_'+name]=mean(v);m['individual_'+name+'_n']=len(v)
    return m,e
