"""One simulation in one process; invoked only by the dynamic-budget scheduler."""
from config import *
import time

def one(j):
    p,full,diff=settings(j['changes'])
    rp=OUT/'records'/f'{j["name"]}.json';tp=OUT/'trajectories'/f'{j["name"]}.npz'
    pins=code_pins()
    if rp.exists() and tp.exists():
        r=json.loads(rp.read_text())
        assert r['job']==j and r['params']==full and r['code_sha256']==pins and r['sha256']==digest(tp)
        return r
    start=time.monotonic();sim=RZ75Sim(p,j['seed'],audit_burn=j['burn']);step=sim.step
    def tracked():
        step()
        if sim.total_steps%round(300/p.dt)==0:print(json.dumps(dict(simulated_s=sim.t,elapsed_s=time.monotonic()-start)),flush=True)
    sim.step=tracked
    print(json.dumps(dict(start=j['name'],pid=os.getpid(),changes=diff)),flush=True)
    d=sim.run(j['seconds'],1.)
    raw={k:hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() for k in KEEP}
    saved={k:d[k].astype(np.float32) if k in ['time','pos','heading','tank'] else d[k].astype(np.int8) if k in ['status','type_id'] else d[k] for k in KEEP}
    reused=None
    if j['phase']=='numerical' and j['arm']=='control':
        wr=HERE.parent/'W2_mechanism/outputs/records'/f'A0_s{j["seed"]}.json'
        wt=HERE.parent/'W2_mechanism/outputs/trajectories'/f'A0_s{j["seed"]}.npz'
        if wr.exists() and wt.exists():
            w=json.loads(wr.read_text())
            assert w['params']==full and w['seconds']==j['seconds'] and digest(wt)==w['sha256']
            assert w['raw_f64_sha256']==raw,'Full-duration W2 equality failed'
            with np.load(wt) as old:
                for k in KEEP:assert old[k].dtype==saved[k].dtype and old[k].shape==saved[k].shape and old[k].tobytes()==saved[k].tobytes(),k
            reused=dict(source=str(wt),record=str(wr),record_sha256=digest(wr),full_duration_bitwise=True)
            tp=wt
    if reused is None:
        temp=tp.with_suffix('.tmp.npz');np.savez_compressed(temp,**saved);temp.replace(tp)
    audit=dict(budget_error=sim.max_budget_error,displacement_error=sim.max_free_displacement_error,
      turn_violations=sim.postburn_state_turn_violations,boundary_corrections=sim.postburn_corrections,
      empty_fraction=sim.empty_postburn_agent_seconds/((j['seconds']-j['burn'])*p.N),reaction_count=sim.reaction_count)
    r=dict(name=j['name'],job=j,params=full,active_params=asdict(p),changed_fields=diff,seed=j['seed'],seconds=j['seconds'],
      elapsed_s=time.monotonic()-start,sha256=digest(tp),code_sha256=pins,raw_f64_sha256=raw,
      storage_dtypes={k:str(v.dtype) for k,v in saved.items()},audit=audit,reused=reused)
    write_json(rp,r);print(json.dumps(dict(done=j['name'],elapsed_s=r['elapsed_s'],audit=audit)),flush=True)
    return r

if __name__=='__main__':one(json.loads(Path(sys.argv[1]).read_text()))
