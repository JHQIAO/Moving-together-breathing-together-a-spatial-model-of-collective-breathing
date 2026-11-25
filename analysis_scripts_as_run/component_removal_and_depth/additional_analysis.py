"""Paired contrasts, lift-off definition sensitivity and complete storage audit."""
from analyze_w2 import *

def main():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('*.json'))]
    rr={r['name']:r for r in records};paired=[];checks=[];sensitivity=[];solo_dilution=[];need_off=[]
    for r in records:
        j=r['job'];p,diff=settings(j);tp=OUT/'trajectories'/f'{r["name"]}.npz'
        assert r['params']==clean(asdict(p)) and r['changes']==clean(diff) and r['sha256']==digest(tp)
        assert r['code_sha256']==code_pins(),'simulation source changed'
        with np.load(tp) as d:
            assert all(k in d.files for k in ['episodes13','events','decision_trials','entries21','time','pos','heading','tank','status'])
            assert d['time'].dtype==np.float32 and d['pos'].dtype==np.float32 and d['heading'].dtype==np.float32 and d['tank'].dtype==np.float32 and d['status'].dtype==np.int8
            assert len(d['time'])==round(j['seconds'])+1 and np.all(np.diff(d['time'])==1)
            assert d['pos'].shape==(round(j['seconds'])+1,p.N,3)
            for key in ['episodes13','events','decision_trials','entries21']:
                for f in d[key].dtype.names:
                    if d[key].dtype[f].kind=='f':assert d[key].dtype[f]==np.dtype('f8')
            assert np.all(np.isfinite(d['tank'])) and d['tank'].min()>=0 and d['tank'].max()<=1
            assert np.isin(d['status'],[0,1,2,3,4]).all()
            assert np.allclose(np.linalg.norm(d['heading'],axis=2),1.,rtol=0,atol=2e-6)
            if j['type'] is not None and j['arm']=='A5':
                baseline=OUT/'trajectories'/f'solo_A0_z3.3_t{j["type"]}_s{j["seed"]}.npz'
                if baseline.exists():
                    with np.load(baseline) as base:
                        equality={k:d[k].dtype==base[k].dtype and d[k].shape==base[k].shape and d[k].tobytes()==base[k].tobytes() for k in d.files}
                        assert all(equality.values()),('solo dilution identity',r['name'],equality)
                    solo_dilution.append(dict(name=r['name'],all_fields_bitwise_equal=True))
            if j['type'] is None and j['arm']!='A0' and j['depth']==3.3:
                base=rr.get(f'A0_s{j["seed"]}')
                if base:
                    with np.load(OUT/'trajectories'/f'{base["name"]}.npz') as b:
                        T=min(j['seconds'],base['seconds']);mm,_=canonical(d,200,T);bm,_=canonical(b,200,T)
                        v=dict(arm=j['arm'],seed=j['seed'],common_seconds=T)
                        v.update({k+'_difference':mm[k]-bm[k] for k in mm if not k.endswith('_n') and k not in ['n_bouts','n_gaps']})
                        paired.append(v)
            if j['type'] is None and j['arm'] in ['A0','A1'] and j['depth']==3.3:
                tr=traces(d,p);t=tr['time'];sm=tr['school_median'];med=tr['median']
                for definition,z in [('all_fish',med),('schooling_only',sm)]:
                    ix=np.flatnonzero((z[:-1]<1)&(z[1:]>=1))+1;ix=ix[(t[ix]>=300)&(t[ix]<j['seconds'])]
                    # A three-second rest requirement guards noise recrossings.
                    steady=np.array([i for i in ix if i>=3 and np.all(z[i-3:i]<1)],int)
                    sensitivity.append(dict(arm=j['arm'],seed=j['seed'],definition=definition,n_lifts=len(ix),need_at_lift=mean(tr['mean_need'][ix]),positive_need_fraction=mean(tr['mean_need'][ix]>0),n_steady_lifts=len(steady),steady_need_at_lift=mean(tr['mean_need'][steady])))
            if j['type'] is None and j['arm'] in ['A2','A10']:
                horizon=3600.;tr=traces(d,p);keep=tr['time']<=horizon;t=tr['time'][keep]
                sm=uniform_filter1d(tr['median'][keep],3,mode='nearest');pk=find_peaks(sm,prominence=.2)[0]
                pk=pk[(t[pk]>=j['burn'])&(t[pk]<horizon)];sample=(t>=j['burn'])&(t<horizon)
                cm,_=canonical(d,p.N,horizon,j['burn'])
                need_off.append(dict(arm=j['arm'],seed=j['seed'],seconds=horizon,
                    rise_rate_min=len(pk)*60/(horizon-j['burn']),
                    surface_rise_rate_min=int(np.sum(sm[pk]>=p.box[2]-1))*60/(horizon-j['burn']),
                    empty_sample_fraction=mean(tr['empty_sample'][keep][sample]),
                    completed_breaths_per_fish_hour=cm['completed_breaths_per_fish_hour']))
        checks.append(dict(name=r['name'],sha256=r['sha256'],parameter_assertion=True,storage_assertion=True,**r['audit']))
    csvwrite(ANA/'paired_canonical_contrasts.csv',paired);csvwrite(ANA/'liftoff_definition_sensitivity.csv',sensitivity)
    csvwrite(ANA/'need_off_matched_horizon.csv',need_off)
    write_json(ANA/'need_off_matched_summary.json',{a:{k:summarise([r[k] for r in need_off if r['arm']==a]) for k in ['rise_rate_min','surface_rise_rate_min','empty_sample_fraction','completed_breaths_per_fish_hour']} for a in ['A2','A10']})
    result=dict(completed=len(records),expected=len(jobs()),all_complete=len(records)==len(jobs()),checks=checks,solo_dilution_identity=solo_dilution,
       max_budget_error=max([r['budget_error'] for r in checks],default=0),max_displacement_error=max([r['displacement_error'] for r in checks],default=0),
       total_turn_violations=sum(r['turn_violations'] for r in checks),total_boundary_corrections=sum(r['boundary_corrections'] for r in checks))
    write_json(HERE/'qa/storage_audit.json',result)
    write_json(ANA/'paired_summary.json',{a:{k:summarise([v[k] for v in paired if v['arm']==a]) for k in paired[0] if k.endswith('_difference')} for a in ARMS if any(v['arm']==a for v in paired)} if paired else {})
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
