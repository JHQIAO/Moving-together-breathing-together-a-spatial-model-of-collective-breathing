"""Resumable W3 simulations; budget checked before every launch and resume."""
import os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S25=ROOT/'study_out/local_work/rest_zone_step25'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib');os.environ['NUMBA_CACHE_DIR']=str(HERE/'.cache/numba')
import json,time,hashlib,subprocess,signal,argparse,traceback,platform
from dataclasses import asdict
import numpy as np
sys.path.insert(0,str(S25));from model25 import parameters
sys.path.insert(0,str(HERE.parent/'W5b_code'));from rz75_model import RZ75Sim,type_counts

def clean(v):return json.loads(json.dumps(v,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(v),indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def jobs():
    out=[]
    for n in (2,5,10,25,50,100,200,400):
        for s in range(9201,9206 if n<=50 else 9204):out.append(dict(arm=f'same_N{n}',protocol='same',N=n,seed=s,changes={'N':n} if n!=200 else {}))
    for n in (50,100,400):
        for s in range(9201,9204):out.append(dict(arm=f'density_N{n}',protocol='density',N=n,seed=s,changes=dict(N=n,box=[10*n/200,3.3,3.3])))
    for name,ratio in [('all_fast',[1,0,0]),('all_medium',[0,1,0]),('all_slow',[0,0,1]),('fast211',[2,1,1]),('slow112',[1,1,2])]:
        for s in range(9201,9204):out.append(dict(arm=name,protocol='composition',N=200,seed=s,changes=dict(type_ratio=ratio)))
    for j in out:j.update(name=f"{j['arm']}_s{j['seed']}",seconds=7500.,burn=300.,sample_dt=1.)
    return out

def params(j):
    ref=json.loads((S25/'outputs/records/A_RZ75_s7301.json').read_text())['params']
    p0=parameters(kappa_bottom=.3,rest_height=1.,need_setpoint=.75,consumption_rates=tuple(json.loads((S25/'outputs/b_RZ75_frozen.json').read_text())['rates']))
    assert clean(asdict(p0))==ref
    from dataclasses import replace
    p=replace(p0,**{k:tuple(v) if isinstance(v,list) else v for k,v in j['changes'].items()})
    actual={k:v for k,v in clean(asdict(p)).items() if v!=ref[k]}
    assert actual==j['changes'],(actual,j['changes'])
    assert p.box[0]>2*p.glide_layer and p.box[2]>p.return_depth
    assert p.dt==.025 and p.reaction_interval==.05
    return p

def valid(j):
    rp=HERE/'records'/f"{j['name']}.json";tp=HERE/'trajectories'/f"{j['name']}.npz"
    if not rp.exists():return False
    r=json.loads(rp.read_text());assert tp.exists()
    assert r['job']==j and r['params']==clean(asdict(params(j)))
    assert r['sha256']==sha(tp)
    return True

def one(j):
    if valid(j):return
    p=params(j);tp=HERE/'trajectories'/f"{j['name']}.npz";rp=HERE/'records'/f"{j['name']}.json"
    assert not tp.exists(),f'Orphan output: {tp}'
    model=HERE.parent/'W5b_code/rz75_model.py';pin=sha(model);start=time.monotonic()
    sim=RZ75Sim(p,j['seed'],audit_burn=j['burn']);step=sim.step
    def tracked():
        step()
        if sim.total_steps%12000==0:
            v=dict(name=j['name'],pid=os.getpid(),simulated_s=sim.t,elapsed_s=time.monotonic()-start,status='running')
            write(HERE/'logs'/f"{j['name']}.progress.json",v);print(json.dumps(v),flush=True)
    sim.step=tracked;d=sim.run(j['seconds'],j['sample_dt'])
    keep=['time','pos','heading','tank','status','type_id','agent_consumption_rates','episodes13','events','decision_trials','entries21','state_agent_seconds','empty_postburn_agent_seconds','audit_burn']
    saved={k:d[k] for k in keep}
    for k in ('time','pos','heading','tank'):saved[k]=saved[k].astype(np.float32)
    for k in ('status','type_id'):saved[k]=saved[k].astype(np.int8)
    tmp=tp.with_suffix('.tmp.npz');np.savez_compressed(tmp,**saved);tmp.replace(tp)
    assert sha(model)==pin
    write(rp,dict(job=j,params=asdict(p),changed_parameters=j['changes'],seed=j['seed'],seconds=j['seconds'],burn=j['burn'],elapsed_s=time.monotonic()-start,sha256=sha(tp),code_version='W3 standalone RZ75; frozen full Sim25 parameter dataclass',model_sha256=pin,runner_sha256=sha(__file__),python=platform.python_version(),numpy=np.__version__,type_counts=type_counts(p.N,p.type_ratio),pid=os.getpid(),budget_error=sim.max_budget_error,turn_violations=sim.postburn_state_turn_violations,boundary_corrections=sim.postburn_corrections,arrays={k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in saved.items()}))
    write(HERE/'logs'/f"{j['name']}.progress.json",dict(name=j['name'],pid=os.getpid(),status='complete',elapsed_s=time.monotonic()-start))

def budget():return max(0,min(4,int(json.loads((HERE.parent/'CPU_BUDGET.json').read_text())['W3_predictions'])))
def main():
    for d in ('records','trajectories','logs','qa','analysis','figures','.cache/matplotlib'):(HERE/d).mkdir(parents=True,exist_ok=True)
    js=jobs()
    for j in js:params(j)
    write(HERE/'protocol.json',dict(jobs=js,n_jobs=len(js),baseline=asdict(params(next(j for j in js if j['arm']=='same_N200'))),max_workers=4,geometry_note='Minimum-image x/y torus; shortest Lx=2.5 BL > 2*glide_layer. Interaction radius may exceed half a periodic box, as in the frozen y geometry; each fish appears once, no image duplication.'))
    pending=[j for j in js if not valid(j)];running=[];failed=[];done=len(js)-len(pending)
    while pending or running:
        for r in running[:]:
            code=r['p'].poll()
            if code is not None:
                running.remove(r);r['log'].close()
                if code==0 and valid(r['job']):done+=1
                else:failed.append(dict(job=r['job'],returncode=code))
        cap=budget()
        active=[r for r in running if not r['paused']]
        for r in active[cap:]:os.kill(r['p'].pid,signal.SIGSTOP);r['paused']=True
        for r in running:
            if r['paused'] and sum(not x['paused'] for x in running)<budget():os.kill(r['p'].pid,signal.SIGCONT);r['paused']=False
        while pending and len(running)<budget():
            j=pending.pop(0);params(j)
            cap=budget() # immediately before every simulation launch
            if len(running)>=cap:pending.insert(0,j);break
            log=open(HERE/'logs'/f"{j['name']}.log",'a')
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--one',j['name']],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            running.append(dict(p=p,log=log,job=j,paused=False));print('LAUNCH',j['name'],p.pid,'budget',cap,flush=True)
            with open(HERE/'logs/launches.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),name=j['name'],pid=p.pid,budget=cap,active=len(running)))+'\n')
        state=dict(supervisor_pid=os.getpid(),budget=budget(),completed=done,total=len(js),pending=len(pending),failed=failed,running=[dict(name=r['job']['name'],pid=r['p'].pid,paused=r['paused']) for r in running])
        write(HERE/'active_processes.json',state)
        (HERE/'STATUS.md').write_text('# W3 execution status\n\n'+f"Completed {done}/{len(js)} runs. Supervisor PID {os.getpid()}. Dynamic budget {state['budget']}.\n\n"+'```json\n'+json.dumps(state,indent=2)+'\n```\n\nOutputs: records/, trajectories/, logs/. Remaining: finish queued simulations, run analyse.py, plot_figures.py, write_report.py and final QA.\n')
        if pending or running:time.sleep(15)
    assert not failed,failed
    for script in ('analyse.py','plot_figures.py','write_report.py'):
        path=HERE/script
        if path.exists():subprocess.run([sys.executable,'-B',str(path)],cwd=HERE,check=True)
    print('ALL SIMULATIONS COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--one');a=parser.parse_args()
    try:
        if a.one:one(next(j for j in jobs() if j['name']==a.one))
        else:main()
    except Exception:
        traceback.print_exc();raise
