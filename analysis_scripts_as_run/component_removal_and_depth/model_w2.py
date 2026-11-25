"""W2-only interventions; frozen Sim25 is imported without modification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys, json, hashlib
sys.dont_write_bytecode=True
from pathlib import Path
from dataclasses import asdict
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
S25=ROOT/'study_out/local_work/rest_zone_step25'
sys.path.insert(0,str(S25))
from model25 import Sim25, parameters
from model23 import need_terms, _row_normalize, minimum_image
RATES=tuple(json.loads((S25/'outputs/b_RZ75_frozen.json').read_text())['rates'])
REFERENCE=json.loads((S25/'outputs/records/A_RZ75_s7301.json').read_text())['params']

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n');tmp.replace(path)

ARMS={
 'A0':{},'A1':{'kappa_bottom':0.},
 'A2':{'upward_strength':0.,'need_down':1e-12},
 'A3':{'upward_strength':0.,'need_down':1e-12,'kappa_bottom':0.},
 'A4':{'need_down':.35},'A5':{'dilution':False},'A6':{'surface_strength':0.},
 'A7':{'consumption_rates':(float(np.mean(RATES)),)*3},'A8':{},'A9':{},
 'A10':{'upward_strength':0.,'need_down':1e-12,'kappa_bottom':.6,'rest_height':1.5}}
SWITCHES={'A8':{'no_schooling':True},'A9':{'constant_rest':True}}
LABELS={'A0':'RZ75','A1':'No rest zone','A2':'Need off','A3':'Need off, no rest',
 'A4':'Symmetric need','A5':'Undiluted push','A6':'No surface push','A7':'Identical rates',
 'A8':'No schooling','A9':'Level everywhere','A10':'Need off, stronger hold'}

def frozen():
    p=parameters(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=RATES)
    assert clean(asdict(p))==REFERENCE,'Frozen parameter mismatch'
    return p

def settings(job):
    frozen()  # Required before every run, including interventions and solo controls.
    changes=dict(ARMS[job['arm']])
    if job.get('depth',3.3)!=3.3:changes['box']=(10.,3.3,job['depth'])
    if job.get('type') is not None:
        changes.update(N=1,type_ratio=tuple(float(i==job['type']) for i in range(3)))
    p=parameters(**dict(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=RATES,**{})) if not changes else parameters(**(dict(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=RATES)|changes))
    diff={k:{'from':REFERENCE[k],'to':v} for k,v in clean(asdict(p)).items() if v!=REFERENCE[k]}
    assert set(diff)=={k for k,v in changes.items() if clean(v)!=REFERENCE[k]}
    return p,diff

class W2Sim(Sim25):
    """Switches affect voluntary steering only; repulsion and motor states retained."""
    def __init__(self,*args,no_schooling=False,constant_rest=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.no_schooling=no_schooling;self.constant_rest=constant_rest
    def social_direction(self):
        if not self.no_schooling:return super().social_direction()
        p=self.p
        dif=minimum_image(self.pos[None,:,:]-self.pos[:,None,:],p.box)
        dist=np.linalg.norm(dif,axis=2)
        unit=np.divide(dif,dist[:,:,None],out=np.zeros_like(dif),where=dist[:,:,None]>0)
        rep=(dist>0)&(dist<p.rr);in_rep=rep.any(axis=1)
        dr=-_row_normalize(np.einsum('ijk,ij->ik',unit,rep))
        direction=self.dir.copy();direction[in_rep]=dr[in_rep]
        cancelled=np.linalg.norm(direction,axis=1)<1e-12
        direction[cancelled]=self.dir[cancelled]
        return direction,in_rep
    def reaction_target(self):
        if not self.constant_rest:return super().reaction_target()
        p=self.p
        if self.total_steps%self.reaction_stride==0:
            desired,in_rep=self.social_direction();self.cached_social=desired.copy()
            n=self.nearby_count() if p.dilution else np.zeros(p.N)
            need,push=need_terms(self.reserves,p.box[2]-self.pos[:,2],p,n)
            vertical=need+push-p.kappa_bottom*self.dir[:,2]
            desired[~in_rep,2]+=vertical[~in_rep]
            base=_row_normalize(desired.copy())
            boosted=self.follow_active&(self.status==0)&~in_rep
            desired[boosted,2]+=p.follow_weight
            desired=_row_normalize(desired)
            noise=self.rng.normal(0,p.eta*np.sqrt(p.reaction_interval/.1),(p.N,3))
            noise[:,2]*=p.vertical_noise_scale;desired+=noise
            self.cached_without_follow=base+noise
            self.cached_follow_mask=boosted.copy();self.cached_desired=desired;self.reaction_count+=1
            if self.t>=self.audit_burn:
                self.neighbour_sum+=float(n.sum());self.neighbour_observations+=p.N
                self.noise_sum+=noise.sum(axis=0);self.noise_squares+=(noise*noise).sum(axis=0);self.noise_n+=p.N
        else:
            ended=self.cached_follow_mask&~self.follow_active
            self.cached_desired[ended]=self.cached_without_follow[ended];self.cached_follow_mask[ended]=False
        return self.cached_desired.copy(),self.cached_social.copy()

def jobs():
    shoal=[]
    for arm in ARMS:
        for seed in range(9001,9004):
            shoal.append(dict(name=f'{arm}_s{seed}',arm=arm,seed=seed,seconds=3600. if arm=='A2' else 7500.,burn=300.,depth=3.3,type=None))
    for arm in ['A0','A1']:
        for depth in [2.2,5.,6.6]:
            for seed in range(9001,9004):
                shoal.append(dict(name=f'{arm}_z{depth:.1f}_s{seed}',arm=arm,seed=seed,seconds=7500.,burn=300.,depth=depth,type=None))
    solo=[]
    for arm,depth in [(a,3.3) for a in ['A0','A1','A2','A3','A4','A5','A6','A9','A10']]+[(a,z) for a in ['A0','A1'] for z in [2.2,5.,6.6]]:
        for typ in range(3):
            for seed in range(8501,8504):
                solo.append(dict(name=f'solo_{arm}_z{depth:.1f}_t{typ}_s{seed}',arm=arm,seed=seed,seconds=3600.,burn=300.,depth=depth,type=typ))
    return shoal+solo

def code_pins():
    paths={Path(__file__).resolve(),HERE/'run_w2.py'}
    for c in Sim25.__mro__:
        mod=sys.modules.get(c.__module__)
        if getattr(mod,'__file__',None):paths.add(Path(mod.__file__).resolve())
    paths.add(ROOT/'breathing_model2.py')
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}
