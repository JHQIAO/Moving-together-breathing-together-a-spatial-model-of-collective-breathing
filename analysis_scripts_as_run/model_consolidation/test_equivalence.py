"""Sequential, resumable byte-equality suite; never more than one simulator runs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse,dataclasses,hashlib,json,platform,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import rz75_model as consolidated
S25=ROOT/'study_out/local_work/rest_zone_step25'
sys.path.insert(0,str(S25))
from model25 import Sim25,parameters as legacy_parameters

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.loads(json.dumps(value))
def jobs():
    out=[('rz75_7301',7301,1800.,{}),('rz75_9001',9001,600.,{})]
    for j,name in enumerate(('fast','medium','slow')):
        out.append((f'solo_{name}',8501+j,3600.,dict(N=1,type_ratio=tuple(float(i==j) for i in range(3)))))
    out += [('no_rest',9001,600.,dict(kappa_bottom=0.)),
            ('need_off',9001,600.,dict(upward_strength=0.,need_down=1e-12)),
            ('no_dilution',9001,600.,dict(dilution=False)),
            ('half_dt',9001,600.,dict(dt=.0125)),
            ('depth5',9001,600.,dict(box=(10.,3.3,5.)))]
    return out

def run_one(name,seed,seconds,changes):
    frozen=json.loads((S25/'outputs/records/A_RZ75_s7301.json').read_text())['params']
    rates=tuple(json.loads((S25/'outputs/b_RZ75_frozen.json').read_text())['rates'])
    base=legacy_parameters(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=rates)
    assert canonical(dataclasses.asdict(base))==frozen
    p=legacy_parameters(**dict(dict(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=rates),**changes))
    q=consolidated.parameters(**changes)
    for key,val in dataclasses.asdict(q).items():assert canonical(val)==canonical(getattr(p,key)),key
    pd=canonical(dataclasses.asdict(p)); changed={k:v for k,v in pd.items() if v!=frozen[k]}
    version=dict(module=digest(HERE/'rz75_model.py'),suite=digest(__file__))
    source_paths={Path(sys.modules[c.__module__].__file__).resolve() for c in Sim25.__mro__ if c is not object}
    source_paths.update(ROOT/x for x in ['breathing_model2.py','study_out/local_work/reaction_clock_step17/model17.py'])
    version['legacy']={str(x.relative_to(ROOT)):digest(x) for x in sorted(source_paths)}
    target=HERE/'outputs'/f'{name}.npz';record=target.with_suffix('.json')
    target.parent.mkdir(exist_ok=True)
    if record.exists() and target.exists():
        r=json.loads(record.read_text())
        if r['params']==pd and r['code_version']==version and r['sha256']==digest(target) and r['seconds']==seconds and r['seed']==seed and r['passed']:
            print('SKIP verified',name,flush=True);return r
        raise RuntimeError(f'{name}: stale existing outputs; move them into a named local archive before rerun')
    print('START',name,seed,seconds,changes,flush=True)
    timings={};data={}
    # Alternate order across cases to reduce systematic warm-up/order bias.
    order=['Sim25','standalone'] if seed%2 else ['standalone','Sim25']
    for label in order:
        t=time.perf_counter()
        sim=Sim25(p,seed,audit_burn=300.) if label=='Sim25' else consolidated.RZ75Sim(q,seed,audit_burn=300.)
        data[label]=sim.run(seconds,1.)
        timings[label]=time.perf_counter()-t
        print('FINISHED',name,label,round(timings[label],3),flush=True)
    a,b=data['Sim25'],data['standalone']
    assert set(a)==set(b),(set(a),set(b))
    comparisons={}
    for k in a:
        x,y=a[k],b[k]
        same=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
        comparisons[k]=dict(passed=bool(same),shape=list(x.shape),dtype=x.dtype.descr if x.dtype.names else str(x.dtype),reference_sha256=hashlib.sha256(x.tobytes()).hexdigest(),standalone_sha256=hashlib.sha256(y.tobytes()).hexdigest(),bytes=x.nbytes)
    passed=all(v['passed'] for v in comparisons.values())
    # Equality was checked at float64 before compressing trajectories for storage.
    saved={k:(v.astype(np.float32) if k in ('time','pos','heading','tank') else v) for k,v in b.items()}
    np.savez_compressed(target,**saved)
    result=dict(name=name,seed=seed,seconds=seconds,sample_dt=1.,audit_burn=300.,params=pd,standalone_params=canonical(dataclasses.asdict(q)),changed=changed,elapsed=timings,code_version=version,sha256=digest(target),passed=passed,comparisons=comparisons,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),OPENBLAS_NUM_THREADS=os.environ['OPENBLAS_NUM_THREADS'],OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS']))
    record.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS' if passed else 'FAIL',name,flush=True)
    if not passed:raise AssertionError([k for k in comparisons if not comparisons[k]['passed']])
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    selected=jobs()
    if args.smoke:selected=[('smoke_'+n,s,2.,c) for n,s,t,c in selected]
    rows=[]
    for job in selected:rows.append(run_one(*job))
    (HERE/'outputs'/('smoke_summary.json' if args.smoke else 'equality_summary.json')).write_text(json.dumps(dict(passed=all(r['passed'] for r in rows),runs=[r['name'] for r in rows]),indent=2)+'\n')
if __name__=='__main__':main()
