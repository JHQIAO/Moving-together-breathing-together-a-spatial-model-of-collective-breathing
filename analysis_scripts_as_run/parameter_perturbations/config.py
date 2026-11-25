"""Frozen W4 protocol; no writes outside this workstream."""
import os, sys, json, hashlib
from pathlib import Path
from dataclasses import asdict
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
S25=ROOT/'study_out/local_work/rest_zone_step25'
OUT=HERE/'outputs'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(S25))
from model25 import parameters as legacy_parameters, Sim25
from rz75_frozen import parameters, RZ75Sim
import numpy as np
REFERENCE=json.loads((S25/'outputs/records/A_RZ75_s7301.json').read_text())['params']
RATES=tuple(json.loads((S25/'outputs/b_RZ75_frozen.json').read_text())['rates'])
TARGETS=[55.63,85.59,134.24]
SEEDS=[9301,9302,9303]
TRAIN=[8501,8502,8503]
HOLD=[8504,8505,8506]
LEVELS=dict(need_setpoint=(.70,.80),kappa_bottom=(.20,.40),rest_height=(.75,1.25),
 upward_strength=(.28,.42),need_down=(.07,.13),surface_strength=(.24,.36),surface_layer=(1.2,1.8),
 dilution_min=(.1,.3),dilution_scale=(4.5,7.5),speed=(.5,.7),ascent_speed=(1.7,2.5),
 fermi_midpoint=(.75,.85),fermi_temperature=(.07,.13),decision_timescale=(.2,.45),
 trigger_depth=(.5,.7),refill_level=(.90,1.),eta=(.24,.36),vertical_noise_scale=(.15,.25),
 turning_rate=(REFERENCE['turning_rate']*.8,REFERENCE['turning_rate']*1.2),
 ra=(REFERENCE['ra']*.8,REFERENCE['ra']*1.2))
RANK_METRICS=['gap_mean','gap_sd','long_fraction','participation_mean','individual_mean','slab_mean','class_next_rho']
KEEP=['time','pos','heading','tank','status','episodes13','events','decision_trials','entries21','type_id','agent_consumption_rates']

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def budget():
    b=json.loads((HERE.parent/'CPU_BUDGET.json').read_text())['W4_robustness']
    assert isinstance(b,int) and b>=0
    return min(b,4)

def settings(changes=None):
    """Assert the entire legacy dataclass, then the standalone active projection."""
    base=legacy_parameters(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=RATES)
    assert clean(asdict(base))==REFERENCE
    p=parameters(**(changes or {}))
    active=clean(asdict(parameters()))
    assert all(v==REFERENCE[k] for k,v in active.items())
    legacy=legacy_parameters(**(dict(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=RATES)|(changes or {})))
    full=clean(asdict(legacy))
    assert all(v==full[k] for k,v in clean(asdict(p)).items())
    diff={k:dict(old=REFERENCE[k],new=v) for k,v in full.items() if v!=REFERENCE[k]}
    allowed=set(changes or {})
    if 'speed' in allowed:allowed.add('descent_speed')
    if 'trigger_depth' in allowed or 'return_depth' in allowed:allowed.add('return_clearance')
    assert set(diff)<=allowed,(diff,allowed)
    return p,full,diff

def arms():
    out={'control':{}}
    for key,values in LEVELS.items():
        for label,value in zip(['low','high'],values):
            change={key:value}
            if key=='trigger_depth':change['return_depth']=value+.1
            out[key+'_'+label]=change
    return out

def job(name,arm,seed,seconds,changes,phase,typ=None):
    return dict(name=name,arm=arm,seed=seed,seconds=float(seconds),burn=300.,changes=changes,phase=phase,type=typ)

def shoal_jobs(arm,changes,phase,seconds=1800.,seeds=SEEDS):
    return [job(f'{phase}_{arm}_s{s}',arm,s,seconds,changes,phase) for s in seeds]

def solo_jobs(arm,changes,phase='solo',seeds=TRAIN,typ=None,rate=None):
    result=[]
    for t in range(3) if typ is None else [typ]:
        c=changes|dict(N=1,type_ratio=[float(i==t) for i in range(3)])
        # Match the step-24/25 single-type protocol, including redundant rates.
        c['consumption_rates']=[RATES[t] if rate is None else rate]*3
        for s in seeds:result.append(job(f'{phase}_{arm}_t{t}_s{s}',arm,s,3600.,c,phase,t))
    return result

def code_pins():
    return {n:digest(HERE/n) for n in ['rz75_frozen.py','config.py','worker.py']}
