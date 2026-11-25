"""Autonomous, resumable W4 pipeline with a dynamically polled CPU budget."""
from config import *
import subprocess,time,signal,fcntl,traceback
from metrics import canonical
PY='/opt/anaconda3/bin/python'

def trajectory(r):return Path(r['reused']['source']) if r.get('reused') else OUT/'trajectories'/f'{r["name"]}.npz'

def valid(j):
    rp=OUT/'records'/f'{j["name"]}.json'
    if not rp.exists():return False
    r=json.loads(rp.read_text());tp=trajectory(r)
    p,full,_=settings(j['changes'])
    assert r['job']==j and r['params']==full and r['code_sha256']==code_pins(),j['name']
    assert tp.exists() and digest(tp)==r['sha256'],j['name']
    return True

def metric(j,burn=300.):
    r=json.loads((OUT/'records'/f'{j["name"]}.json').read_text())
    cache=HERE/'analysis'/f'{j["name"]}_burn{burn:g}.json'
    pin=digest(HERE/'metrics.py')
    if cache.exists():
        a=json.loads(cache.read_text())
        if a['sha256']==r['sha256'] and a['metric_sha256']==pin:return a['metrics']
    with np.load(trajectory(r)) as d:m,_=canonical(d,r['params']['N'],j['seconds'],burn)
    write_json(cache,dict(name=j['name'],sha256=r['sha256'],metric_sha256=pin,burn=burn,metrics=m))
    return m

def status(phase,active,done,total):
    state=dict(runner_pid=os.getpid(),phase=phase,done=done,total=total,budget=budget(),
      active=[dict(pid=p.pid,name=j['name'],paused=paused,log=str(OUT/'logs'/f'{j["name"]}.log')) for p,j,log,paused in active],updated=time.time())
    write_json(OUT/'active_processes.json',state)
    text=['# W4 status','',f'Runner PID: {os.getpid()}. Phase: {phase}. Completed in this phase: {done}/{total}.',
      f'Dynamic budget: {state["budget"]}. Updated Unix time: {state["updated"]:.0f}.','',
      'Active simulations:',*['- '+json.dumps(a) for a in state['active']], '',
      'The pipeline runs screening shoals, all isolated-fish settings, ranked recalibration with held-out tests,',
      'long recalibrated shoals, and numerical checks. It then generates tables, draft figures and REPORT.md.',
      'Completed records are checksum-verified on resume. No final results are claimed while this file says running.',
      '', 'Resume: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python study_out/rz75_paper/W4_robustness/run_w4.py`.']
    (HERE/'STATUS.md').write_text('\n'.join(text)+'\n')

def run_jobs(jobs,phase):
    write_json(OUT/f'protocol_{phase}.json',dict(jobs=jobs,code_sha256=code_pins()))
    pending=[j for j in jobs if not valid(j)];done=len(jobs)-len(pending);active=[]
    while pending or active:
        # Re-read also during runs. SIGSTOP/SIGCONT only children we own when a
        # coordinator budget reduction would otherwise leave too many running.
        limit=budget()
        for i,(p,j,log,paused) in enumerate(active):
            should_pause=i>=limit
            if should_pause!=paused and p.poll() is None:
                os.kill(p.pid,signal.SIGSTOP if should_pause else signal.SIGCONT)
                active[i]=(p,j,log,should_pause)
        finished=[]
        for i,(p,j,log,paused) in enumerate(active):
            code=p.poll()
            if code is not None:
                log.close()
                if code:raise RuntimeError(f'{j["name"]} exited {code}; see outputs/logs')
                assert valid(j);metric(j);done+=1;finished.append(i)
                print(json.dumps(dict(phase=phase,done=done,total=len(jobs),name=j['name'])),flush=True)
        for i in reversed(finished):active.pop(i)
        while pending and len(active)<budget(): # mandatory fresh read before EVERY launch
            j=pending.pop(0);jp=OUT/'jobs'/f'{j["name"]}.json';write_json(jp,j)
            log=(OUT/'logs'/f'{j["name"]}.log').open('a',buffering=1)
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            launch_budget=budget()
            if len(active)>=launch_budget:pending.insert(0,j);log.close();break
            p=subprocess.Popen([PY,'-B',str(HERE/'worker.py'),str(jp)],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
            with (OUT/'launches.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),pid=p.pid,name=j['name'],budget=launch_budget,active_before=len(active)))+'\n')
            active.append((p,j,log,False))
        status(phase,active,done,len(jobs))
        if pending or active:time.sleep(5 if active else 30)
    return [json.loads((OUT/'records'/f'{j["name"]}.json').read_text()) for j in jobs]

def ranking(groups):
    base=np.array([[metric(j)[k] for k in RANK_METRICS] for j in groups['control']],float)
    mu=base.mean(axis=0);spread=base.std(axis=0,ddof=1);effects=[]
    for arm,jobs in groups.items():
        if arm=='control':continue
        data=np.array([[metric(j)[k] for k in RANK_METRICS] for j in jobs],float)
        delta=data-base
        for i,k in enumerate(RANK_METRICS):
            effects.append(dict(arm=arm,parameter=arm.rsplit('_',1)[0],level=arm.rsplit('_',1)[1],metric=k,
              control_mean=mu[i],control_seed_sd=spread[i],delta_mean=np.mean(delta[:,i]),delta_min=np.min(delta[:,i]),delta_max=np.max(delta[:,i]),
              delta_seed_sd=np.std(delta[:,i],ddof=1),normalized_effect=float(np.mean(delta[:,i])/mu[i]) if abs(mu[i])>1e-12 else None,
              spread_effect=float(np.mean(delta[:,i])/spread[i]) if spread[i]>1e-12 else None))
    # Average metric-wise rank of maximum absolute low/high control-SD effect.
    # Rank aggregation avoids a near-zero control observable dominating selection.
    scores={p:[] for p in LEVELS}
    detail={}
    from scipy.stats import rankdata
    for k in RANK_METRICS:
        maxima=[]
        for p in LEVELS:
            vals=[abs(e['spread_effect']) for e in effects if e['parameter']==p and e['metric']==k and e['spread_effect'] is not None]
            maxima.append(max(vals) if vals else np.nan)
        if not np.isfinite(maxima).all():raise ValueError(f'Undefined ranking metric {k}; needs explicit review')
        ranks=rankdata(-np.array(maxima),method='average')
        for p,v,r in zip(LEVELS,maxima,ranks):scores[p].append(float(r));detail.setdefault(p,{})[k]=dict(max_abs_spread_effect=v,rank=r)
    result=sorted([dict(parameter=p,mean_rank=np.mean(v),metric_ranks=detail[p]) for p,v in scores.items()],key=lambda x:(x['mean_rank'],x['parameter']))
    write_json(HERE/'analysis/tier1_effects.json',effects)
    write_json(HERE/'analysis/ranking.json',dict(rule='Mean rank across seven metrics of max(abs(low-control),abs(high-control))/control sample SD; alphabetic ties.',ranking=result,selected=[v['parameter'] for v in result[:8]]))
    return [v['parameter'] for v in result[:8]]

def calibrate(arm,changes):
    rates=list(RATES);history=[];train_jobs=[]
    for t,target in enumerate(TARGETS):
        entries=[]
        for iteration in range(6): # baseline plus at most five multiplicative updates
            phase='solo' if iteration==0 else f'cal{iteration}'
            jobs=solo_jobs(arm,changes,phase,TRAIN,t,rates[t])
            run_jobs(jobs,f'calibration_{arm}_t{t}_i{iteration}')
            values=[metric(j)['individual_mean'] for j in jobs]
            if any(v is None for v in values):raise ValueError(f'No isolated intervals: {arm}, type {t}')
            mu=float(np.mean(values));error=mu/target-1
            entries.append(dict(iteration=iteration,rate=rates[t],seed_means=values,mean=mu,error_fraction=error,jobs=[j['name'] for j in jobs]))
            if abs(error)<=.02 or iteration==5:break
            rates[t]*=float(np.clip(mu/target,.5,1.5))
        train_jobs.extend(jobs);history.append(entries)
    held=[]
    for t,rate in enumerate(rates):
        jobs=solo_jobs(arm,changes,'heldout',HOLD,t,rate)
        run_jobs(jobs,f'heldout_{arm}_t{t}')
        vals=[metric(j)['individual_mean'] for j in jobs]
        held.append(dict(type=t,seed_means=vals,mean=np.mean(vals),error_fraction=np.mean(vals)/TARGETS[t]-1,jobs=[j['name'] for j in jobs]))
    result=dict(arm=arm,rates=rates,targets=TARGETS,history=history,training_pass=[abs(h[-1]['error_fraction'])<=.02 for h in history],heldout=held,
      heldout_pass=[abs(h['error_fraction'])<=.05 for h in held])
    write_json(HERE/'analysis'/f'calibration_{arm}.json',result)
    return rates

def calibrate_all(selected):
    """Batch independent types/arms without changing any seed or rate update."""
    global run_jobs
    settings_by_arm={p+'_'+level:arms()[p+'_'+level] for p in selected for level in ['low','high']}
    state={arm:dict(rates=list(RATES),history=[[],[],[]]) for arm in settings_by_arm}
    def observe(arm,t,iteration,jobs):
        values=[metric(j)['individual_mean'] for j in jobs]
        if any(v is None for v in values):raise ValueError(f'No isolated intervals: {arm}, type {t}')
        mu=float(np.mean(values))
        state[arm]['history'][t].append(dict(iteration=iteration,rate=state[arm]['rates'][t],
          seed_means=values,mean=mu,error_fraction=mu/TARGETS[t]-1,jobs=[j['name'] for j in jobs]))
    for arm,changes in settings_by_arm.items():
        for t in range(3):
            jobs=solo_jobs(arm,changes,'solo',TRAIN,t,RATES[t])
            assert all(valid(j) for j in jobs),'Tier-1 isolated runs must already be complete'
            observe(arm,t,0,jobs)
    for iteration in range(1,6):
        todo=[];groups=[]
        for arm,changes in settings_by_arm.items():
            for t in range(3):
                last=state[arm]['history'][t][-1]
                if abs(last['error_fraction'])<=.02:continue
                state[arm]['rates'][t]*=float(np.clip(last['mean']/TARGETS[t],.5,1.5))
                jobs=solo_jobs(arm,changes,f'cal{iteration}',TRAIN,t,state[arm]['rates'][t])
                todo.extend(jobs);groups.append((arm,t,jobs))
        if not todo:break
        run_jobs(todo,f'calibration_round_{iteration}')
        for arm,t,jobs in groups:observe(arm,t,iteration,jobs)
        write_json(HERE/'analysis/calibration_progress.json',dict(iteration=iteration,arms=state))
    heldout={}
    for arm,changes in settings_by_arm.items():
        for t,rate in enumerate(state[arm]['rates']):heldout[(arm,t)]=solo_jobs(arm,changes,'heldout',HOLD,t,rate)
    run_jobs([j for jobs in heldout.values() for j in jobs],'heldout_all')
    expected={}
    for arm in settings_by_arm:
        held=[]
        for t in range(3):
            jobs=heldout[(arm,t)];vals=[metric(j)['individual_mean'] for j in jobs]
            if any(v is None for v in vals):raise ValueError(f'No held-out intervals: {arm}, type {t}')
            held.append(dict(type=t,seed_means=vals,mean=np.mean(vals),error_fraction=np.mean(vals)/TARGETS[t]-1,jobs=[j['name'] for j in jobs]))
        history=state[arm]['history']
        result=dict(arm=arm,rates=state[arm]['rates'],targets=TARGETS,history=history,
          training_pass=[abs(h[-1]['error_fraction'])<=.02 for h in history],heldout=held,
          heldout_pass=[abs(h['error_fraction'])<=.05 for h in held])
        expected[arm]=clean(result);write_json(HERE/'analysis'/f'calibration_{arm}.json',result)
    # Replay the original sequential controller against cached records only.
    # If even one rate/job differs, fail before any additional simulation launch.
    original_run_jobs=run_jobs
    def cached_only(jobs,phase):
        assert all(valid(j) for j in jobs),f'Batched/sequential job mismatch: {phase}'
        return [json.loads((OUT/'records'/f'{j["name"]}.json').read_text()) for j in jobs]
    run_jobs=cached_only
    try:
        for arm,changes in settings_by_arm.items():
            calibrate(arm,changes)
            actual=json.loads((HERE/'analysis'/f'calibration_{arm}.json').read_text())
            assert actual==expected[arm],f'Calibration scheduling changed the result: {arm}'
    finally:run_jobs=original_run_jobs
    write_json(HERE/'qa/calibration_batch_equivalence.json',dict(passed=True,arms=list(settings_by_arm),
      method='Original sequential controller replayed on cached records only; every requested job and entire calibration result matched exactly.'))
    return {arm:state[arm]['rates'] for arm in settings_by_arm}

def main():
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((HERE/'qa/preflight.json').read_text())['passed']
    groups={a:shoal_jobs(a,c,'tier1') for a,c in arms().items()}
    run_jobs([j for js in groups.values() for j in js],'tier1_shoals')
    selected=ranking(groups)
    run_jobs([j for a,c in arms().items() for j in solo_jobs(a,c)],'tier1_solos')
    # Calibration target control is frozen; compare long runs on the SAME seeds.
    tier2=shoal_jobs('control',{},'tier2',7500.)
    for parameter in selected:
        for level in ['low','high']:
            arm=parameter+'_'+level;c=arms()[arm];rates=calibrate(arm,c)
            tier2.extend(shoal_jobs(arm,c|dict(consumption_rates=rates),'tier2',7500.))
    run_jobs(tier2,'tier2_shoals')
    numerical=[]
    for arm,c in [('control',{}),('half_dt',{'dt':.0125}),('reaction_010',{'reaction_interval':.10})]:
        numerical.extend(shoal_jobs(arm,c,'numerical',7500.,[9001,9002,9003]))
    run_jobs(numerical,'numerical')
    for j in numerical:metric(j,900.)
    subprocess.run([PY,'-B',str(HERE/'analyse_w4.py')],cwd=HERE,check=True)
    subprocess.run([PY,'-B',str(HERE/'plot_w4.py')],cwd=HERE,check=True)
    subprocess.run([PY,'-B',str(HERE/'report_w4.py')],cwd=HERE,check=True)
    status('complete',[],len(list((OUT/'records').glob('*.json'))),len(list((OUT/'records').glob('*.json'))))
    (HERE/'STATUS.md').write_text('# W4 complete\n\nAll planned phases completed. See REPORT.md.\n')

if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        with (HERE/'STATUS.md').open('a') as f:f.write('\n## Pipeline stopped\n\n```\n'+error+'```\nOther owned workers may still be running; see outputs/active_processes.json.\n')
        raise
