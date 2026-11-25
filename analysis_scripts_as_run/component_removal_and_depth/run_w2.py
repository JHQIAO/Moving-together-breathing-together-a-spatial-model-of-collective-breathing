"""Resumable four-worker queue. All writes and logs remain in W2."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,hashlib,traceback,fcntl,contextlib
sys.dont_write_bytecode=True
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from model_w2 import *
OUT=HERE/'outputs'
KEEP=['time','pos','heading','tank','status','episodes13','events','decision_trials','entries21','type_id','agent_consumption_rates']

def one(j):
    p,diff=settings(j);name=j['name'];rp=OUT/'records'/f'{name}.json';tp=OUT/'trajectories'/f'{name}.npz';pins=code_pins()
    if rp.exists() and tp.exists():
        r=json.loads(rp.read_text())
        assert r['job']==j and r['params']==clean(asdict(p)) and r['sha256']==digest(tp) and r['code_sha256']==pins,name
        return dict(name=name,status='verified-existing')
    start=time.monotonic()
    with (OUT/'logs'/f'{name}.log').open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        print(json.dumps(dict(start=name,pid=os.getpid(),job=j,changes=diff,switches=SWITCHES.get(j['arm'],{}))),flush=True)
        try:
            s=W2Sim(p,j['seed'],audit_burn=j['burn'],**SWITCHES.get(j['arm'],{}))
            original_step=s.step
            def tracked():
                original_step()
                if s.total_steps%12000==0:print(json.dumps(dict(simulated_s=s.t,elapsed_s=time.monotonic()-start)),flush=True)
            s.step=tracked;data=s.run(j['seconds'],1.)
            raw_hashes={k:hashlib.sha256(np.ascontiguousarray(data[k]).tobytes()).hexdigest() for k in KEEP}
            saved={k:data[k].astype(np.float32) if k in ['time','pos','heading','tank'] else data[k].astype(np.int8) if k in ['status','type_id'] else data[k] for k in KEEP}
            temp=tp.with_suffix('.tmp.npz');np.savez_compressed(temp,**saved);temp.replace(tp)
            audit=dict(budget_error=s.max_budget_error,displacement_error=s.max_free_displacement_error,
                turn_violations=s.postburn_state_turn_violations,boundary_corrections=s.postburn_corrections,
                empty_fraction=s.empty_postburn_agent_seconds/((j['seconds']-j['burn'])*p.N),
                reaction_count=s.reaction_count,initialization=s.initialization)
            record=dict(name=name,job=j,params=asdict(p),changes=diff,switches=SWITCHES.get(j['arm'],{}),
                seed=j['seed'],seconds=j['seconds'],elapsed_s=time.monotonic()-start,sha256=digest(tp),code_sha256=pins,
                raw_f64_sha256=raw_hashes,storage_dtypes={k:str(v.dtype) for k,v in saved.items()},audit=audit)
            write_json(rp,record);print(json.dumps(dict(done=name,elapsed_s=record['elapsed_s'],audit=audit)),flush=True)
        except Exception:
            traceback.print_exc();raise
    return dict(name=name,status='complete',elapsed_s=time.monotonic()-start)

def main():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=json.loads((HERE/'qa/preflight.json').read_text());assert gate['passed']
    work=jobs();protocol=dict(jobs=work,workers=4,code_sha256=code_pins())
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==clean(protocol),'Protocol changed'
    else:write_json(pp,protocol)
    errors=[];done=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(one,j):j for j in work}
        write_json(OUT/'active_processes.json',dict(runner_pid=os.getpid(),worker_pids=[p.pid for p in pool._processes.values()],status='running',total=len(work)))
        for f in as_completed(fs):
            try:r=f.result();done.append(r)
            except Exception as e:r=dict(name=fs[f]['name'],error=repr(e));errors.append(r);traceback.print_exc()
            print(json.dumps(dict(done=len(done),total=len(work),failed=len(errors),result=r)),flush=True)
            write_json(OUT/'progress.json',dict(done=len(done),total=len(work),failed=errors,last=r))
    write_json(OUT/'active_processes.json',dict(runner_pid=os.getpid(),worker_pids=[],status='failed' if errors else 'complete',total=len(work)))
    write_json(OUT/'manifest.json',dict(results=done,errors=errors,records={p.stem:digest(p) for p in sorted((OUT/'records').glob('*.json'))}))
    if errors:raise RuntimeError(errors)

if __name__=='__main__':main()
