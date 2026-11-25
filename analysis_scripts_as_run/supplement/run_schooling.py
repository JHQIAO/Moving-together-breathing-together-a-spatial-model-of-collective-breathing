"""Supplement S2: bounded zone-sensitivity simulations, isolated from saved outcomes."""
from pathlib import Path
import os,sys,json,time,hashlib
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'study_out/rz75_paper/W5b_code'))
sys.path.insert(0,str(ROOT))
import numpy as np
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from rz75_model import RZ75Sim,parameters
from breathing_study import bout_events_from

OUT=HERE/'schooling'
for sub in ['records','trajectories','progress']:(OUT/sub).mkdir(parents=True,exist_ok=True)

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(obj),indent=2)+'\n');tmp.replace(path)

class MovementOnly(RZ75Sim):
    def transition(self,mask,to_state,when,reason,before=None):
        if to_state==1 and reason=='proximity':
            self.decision_rng.random(self.p.N)
            return
        return super().transition(mask,to_state,when,reason,before)

def movement_metrics(d,p,burn):
    keep=(d['time']>=burn)&(d['time']<d['time'][-1])
    pos=d['pos'][keep].astype(float);v=d['heading'][keep].astype(float)
    pol=np.linalg.norm(v.mean(axis=1),axis=1)
    centre=pos.mean(axis=1);strength=[]
    for axis,L in enumerate(p.box[:2]):
        phase=np.exp(2j*np.pi*pos[:,:,axis]/L).mean(axis=1)
        centre[:,axis]=np.mod(np.angle(phase),2*np.pi)*L/(2*np.pi)
        strength.append(np.abs(phase))
    offset=pos-centre[:,None,:]
    for axis,L in enumerate(p.box[:2]):offset[:,:,axis]-=L*np.rint(offset[:,:,axis]/L)
    radius=np.linalg.norm(offset,axis=2,keepdims=True)
    unit=np.divide(offset,radius,out=np.zeros_like(offset),where=radius>0)
    momentum=np.linalg.norm(np.cross(unit,v).mean(axis=1),axis=1)
    return dict(polarization=pol.mean(),angular_momentum=momentum.mean(),
                ambiguous_circular_centre_fraction=(np.min(strength,axis=0)<.2).mean()),dict(time=d['time'][keep],polarization=pol,angular_momentum=momentum)

def one(job):
    name=job['name'];rp=OUT/'records'/f'{name}.json';tp=OUT/'trajectories'/f'{name}.npz'
    if rp.exists():
        r=json.loads(rp.read_text());assert r['job']==job
        assert hashlib.sha256(tp.read_bytes()).hexdigest()==r['sha256']
        return r
    changes=dict(job['changes'])
    if job['mode']=='movement_only':changes.update(upward_strength=0.,need_down=0.,consumption_rates=(0.,0.,0.))
    p=parameters(**changes);start=time.monotonic()
    sim=(MovementOnly if job['mode']=='movement_only' else RZ75Sim)(p,job['seed'],audit_burn=300.)
    original=sim.step
    def step():
        original()
        if sim.total_steps%4000==0:
            write(OUT/'progress'/f'{name}.json',dict(seconds=sim.t,target=job['seconds'],elapsed_s=time.monotonic()-start,pid=os.getpid()))
    sim.step=step
    raw=sim.run(job['seconds'],sample_dt=1.)
    keys=['time','pos','heading','tank','status','type_id','agent_consumption_rates','episodes13','events']
    d={k:raw[k] for k in keys}
    for k in ['pos','heading','tank']:d[k]=d[k].astype(np.float32)
    if job['mode']=='movement_only':assert not len(d['episodes13']) and np.all(d['status']==0)
    metrics,traces=movement_metrics(d,p,300.)
    if job['mode']=='breathing':
        e=np.sort(d['episodes13'],order='release',kind='stable')
        bouts=bout_events_from(e['release'],e['exit'],e['agent'],gap=1.)
        bouts=[b for b in bouts if b['t_start']>=300 and b['t_end']<=job['seconds']-2]
        starts=np.array([b['t_start'] for b in bouts]);ends=np.array([b['t_end'] for b in bouts])
        parts=np.array([b['n']/p.N for b in bouts])
        metrics.update(gap_mean=np.mean(starts[1:]-ends[:-1]),participation_mean=parts.mean(),n_events=len(parts))
    np.savez_compressed(tp,**d,**{f'diagnostic_{k}':v for k,v in traces.items()})
    r=dict(job=job,params=asdict(p),metrics=metrics,elapsed_s=time.monotonic()-start,
        sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),engine_sha256=hashlib.sha256((ROOT/'study_out/rz75_paper/W5b_code/rz75_model.py').read_bytes()).hexdigest())
    write(rp,r);print('DONE',name,round(r['elapsed_s'],1),flush=True);return clean(r)

def main():
    # Replay preflight verifies the reused standalone engine against frozen samples.
    sim=RZ75Sim(parameters(),9001);d=sim.run(10.,sample_dt=1.)
    with np.load(ROOT/'study_out/rz75_paper/W1_reference/trajectories/reference_s9001.npz') as ref:
        for key in ['pos','heading','tank','status']:
            np.testing.assert_array_equal(d[key].astype(ref[key].dtype),ref[key][:11])
    jobs=[]
    # Fixed protocol, declared before simulation: 3 x 3 widths, three seeds each.
    for wo in [.15,.30,.60]:
        for wa in [1.25,2.50,5.00]:
            for seed in [9301,9302,9303]:
                jobs.append(dict(name=f'movement_wo{wo:.2f}_wa{wa:.2f}_s{seed}',mode='movement_only',seed=seed,
                    seconds=900.,changes=dict(ro=.15+wo,ra=.15+wo+wa)))
    for factor in [.8,1.2]:
        for seed in [9301,9302,9303]:
            jobs.append(dict(name=f'breathing_ro{factor:.1f}_s{seed}',mode='breathing',seed=seed,
                seconds=1800.,changes=dict(ro=.45*factor)))
    write(HERE/'schooling_protocol.json',dict(jobs=jobs,burn_seconds=300,sample_dt=1,workers=3,
        reference=dict(rr=.15,ro=.45,ra=2.95,orientation_width=.30,attraction_width=2.50),
        movement_only='Disable breathing acceptance and oxygen-drive strengths; keep reserves fixed by zero depletion. Keep social movement, surface avoidance, bottom damping, boundaries and movement noise.',
        angular_momentum='Norm of mean cross product of normalized minimum-image radial displacement from circular xy/arithmetic z centroid with unit heading. Report fraction of samples with either circular-centre strength below 0.2.',
        breathing_comparison='Orientation radius ±20% with all other values fixed; reuse W4 tier1 control and attraction-radius ±20% runs (1800 s, same seeds) for comparison. No rate re-estimation.',
        interpretation='Local sensitivity diagnostics; no changes to the selected reference and no parameter reselection.',preflight='10 s all-sample state/position/heading/reserve replay passed'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,j) for j in jobs]
        result=[future.result() for future in as_completed(futures)]
    write(OUT/'summary.json',dict(runs=sorted(result,key=lambda r:r['job']['name'])))
    print('Schooling analyses complete',flush=True)

if __name__=='__main__':main()
