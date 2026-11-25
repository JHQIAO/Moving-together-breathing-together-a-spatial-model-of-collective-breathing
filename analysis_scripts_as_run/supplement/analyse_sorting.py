"""Schooling-only reserve–height sorting and a temporally coupled permutation null."""
from pathlib import Path
import os,sys,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE=ROOT/'study_out/rz75_paper/W1_reference'
OUT=HERE/'source_data';OUT.mkdir(exist_ok=True)
B=499;LAGS=np.arange(-15,16)

def main():
    observed=[];nulls=[];counts=[];rows=[];pins={}
    for seed in range(9001,9011):
        path=BASE/f'trajectories/reference_s{seed}.npz'
        pins[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        record=json.loads((BASE/f'records/reference_s{seed}.json').read_text())
        assert pins[str(path.relative_to(ROOT))]==record['sha256']
        with np.load(path) as f:
            t=f['time'].astype(float);z=f['pos'][:,:,2].astype(float);q=f['tank'].astype(float);school=f['status']==0
            ep=f['episodes13'];ev=f['events']
        med=np.median(z,axis=1)
        lifts=np.flatnonzero((med[:-1]<1)&(med[1:]>=1))+1
        completed={(int(a['agent']),float(a['depart'])) for a in ep}
        pending=[float(a['time']) for a in ev if a['to_state']==1 and (int(a['agent']),float(a['time'])) not in completed]
        cutoff=np.nextafter(min(pending),-np.inf) if pending else t[-1]
        lifts=lifts[(t[lifts]-15>=300)&(t[lifts]+15<=cutoff)]
        idx=lifts[:,None]+LAGS;unique=np.unique(idx)
        x=rankdata(np.where(school,q,np.nan),axis=1,nan_policy='omit',method='average')
        y=rankdata(np.where(school,z,np.nan),axis=1,nan_policy='omit',method='average')
        n=school.sum(axis=1)
        x=np.where(school,x-(n[:,None]+1)/2,0)
        y=np.where(school,y-(n[:,None]+1)/2,0)
        x/=np.linalg.norm(x,axis=1)[:,None];y/=np.linalg.norm(y,axis=1)[:,None]
        rho=(x*y).sum(axis=1)
        assert np.isfinite(rho[unique]).all()
        for i in unique[::max(1,len(unique)//10)]:
            np.testing.assert_allclose(rho[i],spearmanr(q[i,school[i]],z[i,school[i]]).statistic,atol=1e-12)
        mean=rho[idx].mean(axis=0);observed.append(mean)
        rng=np.random.default_rng(np.random.SeedSequence([20261109,seed]))
        permutations=np.array([rng.permutation(q.shape[1]) for _ in range(B)])
        shuffled=np.zeros((len(t),B))
        # Each draw fixes a random ordering of fish IDs for the entire run. Restrict
        # that ordering to the current schooling set, then reassign reserve ranks
        # to ascending schooling IDs. This is a uniform shuffle within each pool,
        # with identical reassignment whenever its membership is unchanged.
        # It preserves contemporaneous margins and avoids independent new shuffles
        # for repeated windows covering the same time point.
        for i in unique:
            s=school[i];order=permutations[s[permutations]].reshape(B,n[i])
            shuffled[i]=(x[i,order]*y[i,s][None,:]).sum(axis=1)
        curves=shuffled[idx].mean(axis=0).T;nulls.append(curves)
        counts.append(dict(seed=seed,n_complete_liftoff_windows=len(lifts),n_unique_samples=len(unique),observation_end=cutoff))
        rows.extend(dict(seed=seed,lag_s=int(lag),rho=float(value)) for lag,value in zip(LAGS,mean))
        np.savez_compressed(OUT/f'S5_sorting_s{seed}.npz',lags=LAGS,observed=mean,null=curves,liftoffs=t[lifts])
        print('Sorting',seed,'windows',len(lifts),'rho at lift-off',round(mean[15],4),flush=True)
    observed=np.array(observed);nulls=np.array(nulls);ensemble_null=nulls.mean(axis=0)
    data=dict(lags=LAGS.tolist(),mean=observed.mean(0).tolist(),sd=observed.std(0,ddof=1).tolist(),
        per_run=observed.tolist(),null_mean=ensemble_null.mean(0).tolist(),null_lower=np.quantile(ensemble_null,.025,axis=0).tolist(),
        null_upper=np.quantile(ensemble_null,.975,axis=0).tolist(),permutations=B,counts=counts,n_runs=10,
        scope='Current schooling fish (state 0) at each saved sample.',
        relative_height='(midrank of height among current schooling fish - 1)/(n_schooling-1); 0 is bottom, 1 top.',
        null_method='499 random identity orderings per run, each fixed across time and restricted to the current schooling set; assign reserve ranks in that order to the observed height ranks in ascending fish-ID order. Average correlations within run, then equally across runs. Central 95% of shuffled ensemble means is a descriptive null envelope, not a confidence interval for biological shoals.',
        temporal_note='Reassignment stays fixed while schooling membership is fixed; when membership changes, restriction of the ordering changes the mapping. This is not an exact full-trajectory permutation across changing pools.')
    (OUT/'S5_sorting.json').write_text(json.dumps(data,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'S5_sorting_per_run.csv',index=False)
    (HERE/'qa/sorting_inputs.json').write_text(json.dumps(pins,indent=2)+'\n')

if __name__=='__main__':main()
