"""Frozen, resumable W1 runs; four simulation processes maximum."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR']=str(HERE/'.cache/numba')
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json,time,traceback,argparse,platform
import numpy as np
from analysis_core import ROOT,S25,clean,digest,write_json,verify_reference
sys.path.insert(0,str(S25))
from model25 import Sim25,parameters

def frozen_params():
    rates=tuple(json.loads((S25/'outputs/b_RZ75_frozen.json').read_text())['rates'])
    p=parameters(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=rates)
    ref=json.loads((S25/'outputs/records/A_RZ75_s7301.json').read_text())['params']
    assert clean(asdict(p))==ref,'Frozen parameter mismatch'
    return p

def sources():
    pins={}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f:
            path=Path(f).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT) and not path.is_relative_to(HERE):
                pins[str(path.relative_to(ROOT))]=digest(path)
    for f in [ROOT/'data/milling_init/combined.npz',S25/'outputs/b_RZ75_frozen.json',HERE/'run_simulations.py']:
        pins[str(f.relative_to(ROOT))]=digest(f)
    return pins

def one(j):
    p=frozen_params(); name=j['name'];rp=HERE/'records'/f'{name}.json';tp=HERE/'trajectories'/f'{name}.npz'
    pins=sources()
    if rp.exists() and tp.exists():
        r=json.loads(rp.read_text())
        assert r['job']==j and r['params']==clean(asdict(p)) and digest(tp)==r['sha256']
        assert r['source_sha256']==pins
        print('SKIP',name,flush=True);return r
    if tp.exists(): raise RuntimeError(f'Orphan npz: {tp}; inspect before rerun')
    write_json(HERE/'logs'/f'{name}.progress.json',dict(pid=os.getpid(),job=j,simulated_s=0,status='running'))
    start=time.monotonic();s=Sim25(p,j['seed'],audit_burn=300.);step=s.step
    def tracked():
        step()
        if s.total_steps%round(300/p.dt)==0:
            progress=dict(pid=os.getpid(),job=j,simulated_s=s.t,elapsed_s=time.monotonic()-start,status='running')
            write_json(HERE/'logs'/f'{name}.progress.json',progress)
            print(json.dumps(progress),flush=True)
    s.step=tracked;d=s.run(j['seconds'],j['sample_dt'])
    # Keep full-precision event logs; retain only required trajectory channels, once.
    keep=['time','pos','heading','tank','status','type_id','agent_consumption_rates',
          'episodes13','events','decision_trials','entries21','state_agent_seconds',
          'empty_postburn_agent_seconds','audit_burn']
    saved={k:d[k] for k in keep}
    for k in ['time','pos','heading','tank']: saved[k]=saved[k].astype(np.float32)
    for k in ['status','type_id']: saved[k]=saved[k].astype(np.int8)
    temp=tp.with_suffix('.tmp.npz');np.savez_compressed(temp,**saved);temp.replace(tp)
    assert sources()==pins,'Source changed during run'
    r=dict(job=j,seed=j['seed'],seconds=j['seconds'],sample_dt=j['sample_dt'],burn=300.,
           params=asdict(p),changed_parameters={},elapsed_s=time.monotonic()-start,sha256=digest(tp),
           source_sha256=pins,code_version='W1 frozen Sim25; source SHA256 manifest',
           python=platform.python_version(),numpy=np.__version__,pid=os.getpid(),
           budget_error=s.max_budget_error,displacement_error=s.max_free_displacement_error,
           turn_violations=s.postburn_state_turn_violations,boundary_corrections=s.postburn_corrections,
           arrays={k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in saved.items()})
    write_json(rp,r);write_json(HERE/'logs'/f'{name}.progress.json',dict(pid=os.getpid(),job=j,status='complete',elapsed_s=r['elapsed_s']))
    print('DONE',name,r['elapsed_s'],flush=True);return r

def verify_highres():
    with np.load(HERE/'trajectories/reference_s9001.npz') as lo,np.load(HERE/'trajectories/highres_s9001.npz') as hi:
        ix=np.arange(0,len(hi['time']),20);checks={}
        for k in ['time','pos','heading','tank','status']:
            a=hi[k][ix];b=lo[k][:len(ix)]
            checks[k]=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
            assert checks[k],k
    write_json(HERE/'qa/highres_bitwise.json',dict(passed=True,checks=checks,whole_second_samples=len(ix)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    assert 1<=args.workers<=4
    for d in ['trajectories','records','logs','qa','.cache/matplotlib','.cache/numba']: (HERE/d).mkdir(parents=True,exist_ok=True)
    verify_reference();frozen_params()
    jobs=[dict(name=f'reference_s{s}',seed=s,seconds=13440.,sample_dt=1.) for s in range(9001,9011)]
    jobs.append(dict(name='highres_s9001',seed=9001,seconds=1500.,sample_dt=.05))
    write_json(HERE/'protocol.json',dict(jobs=jobs,params=asdict(frozen_params()),max_workers=args.workers,source_sha256=sources()))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut=[pool.submit(one,j) for j in jobs]
        write_json(HERE/'active_processes.json',dict(runner_pid=os.getpid(),worker_pids=[p.pid for p in pool._processes.values()],status='running',max_workers=args.workers))
        for f in as_completed(fut): f.result()
    verify_highres()
    write_json(HERE/'active_processes.json',dict(runner_pid=os.getpid(),worker_pids=[],status='complete'))

if __name__=='__main__':
    try: main()
    except Exception:
        (HERE/'logs/failure.txt').write_text(traceback.format_exc());raise
