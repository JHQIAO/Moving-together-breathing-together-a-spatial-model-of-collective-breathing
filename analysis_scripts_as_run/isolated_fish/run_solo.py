"""W8 frozen solo runs; dynamic CPU gate and atomic, hash-verified records."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,subprocess,fcntl,traceback
sys.dont_write_bytecode=True
from pathlib import Path
from dataclasses import asdict
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'W2_mechanism'))
from model_w2 import settings,Sim25,clean,digest,write_json,code_pins
JOBS=[dict(name=f'solo_t{t}_s{s}',arm='A0',depth=3.3,type=t,seed=s,seconds=7200.,burn=300.) for s in range(9601,9611) for t in range(3)]
KEEP=['time','pos','heading','tank','status','episodes13','events','decision_trials','entries21','type_id','agent_consumption_rates']
def pins():return code_pins()|{str(Path(__file__).relative_to(ROOT)):digest(__file__)}
def one(j):
    p,diff=settings(j);rp=HERE/'records'/f'{j["name"]}.json';tp=HERE/'trajectories'/f'{j["name"]}.npz';codes=pins()
    if rp.exists() and tp.exists():
        r=json.loads(rp.read_text());assert r['job']==j and r['params']==clean(asdict(p)) and r['sha256']==digest(tp) and r['code_sha256']==codes
        print('VERIFIED EXISTING',j['name'],flush=True);return
    start=time.monotonic();sim=Sim25(p,j['seed'],audit_burn=300.)
    original=sim.step
    def tracked():
        original()
        if sim.total_steps%12000==0:print(json.dumps(dict(t=sim.t,elapsed=time.monotonic()-start)),flush=True)
    sim.step=tracked
    print(json.dumps(dict(job=j,pid=os.getpid(),changes=diff)),flush=True)
    d=sim.run(j['seconds'],1.)
    saved={k:d[k].astype('float32') if k in ['time','pos','heading','tank'] else d[k].astype('int8') if k in ['status','type_id'] else d[k] for k in KEEP}
    tmp=tp.with_suffix('.tmp.npz');np.savez_compressed(tmp,**saved);tmp.replace(tp)
    r=dict(job=j,params=asdict(p),changes=diff,seed=j['seed'],seconds=j['seconds'],elapsed_s=time.monotonic()-start,sha256=digest(tp),code_sha256=codes,storage_dtypes={k:str(v.dtype) for k,v in saved.items()},audit=dict(budget_error=sim.max_budget_error,displacement_error=sim.max_free_displacement_error,turn_violations=sim.postburn_state_turn_violations,initialization=sim.initialization))
    write_json(rp,r);print('COMPLETE',j['name'],r['elapsed_s'],flush=True)
def queue():
    lock=(HERE/'logs/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((HERE/'qa/preflight.json').read_text())['passed']
    todo=JOBS.copy();active={};done=[];errors=[]
    while todo or active:
        for name,(proc,log,j) in list(active.items()):
            if proc.poll() is not None:
                log.close();(done if proc.returncode==0 else errors).append(name);del active[name]
        while todo:
            budget=min(4,int(json.loads((HERE.parent/'CPU_BUDGET.json').read_text())['W8_results_gaps']))
            if len(active)>=budget:break
            j=todo.pop(0);log=(HERE/'logs'/f'{j["name"]}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--job',j['name']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=HERE)
            active[j['name']]=(proc,log,j)
        state=dict(runner_pid=os.getpid(),active={n:dict(pid=p.pid,job=j,log=f'logs/{n}.log') for n,(p,l,j) in active.items()},done=done,errors=errors,pending=[j['name'] for j in todo],status='running' if todo or active else ('failed' if errors else 'complete'))
        write_json(HERE/'run_state.json',state)
        (HERE/'STATUS.md').write_text('# W8 current status\n\nSimulation queue PID: '+str(os.getpid())+'\n\n```json\n'+json.dumps(state,indent=2)+'\n```\n\nRemaining: finish solo runs; complete G1–G4 analyses, figures, QA and REPORT.md.\n')
        if todo or active:time.sleep(5)
    if errors:raise RuntimeError(errors)
if __name__=='__main__':
    if '--job' in sys.argv:one(next(j for j in JOBS if j['name']==sys.argv[-1]))
    else:queue()
