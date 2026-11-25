"""Empirical uncertainty and scenario targets, independent of running simulations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
import numpy as np
from analysis_core import HERE,load_record,record_statistics,bootstrap_record,write_json,digest,clean

def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=5000);args=p.parse_args()
    record=load_record();out=bootstrap_record(record,args.repeats)
    samples={}
    for method,result in out['methods'].items():
        samples[method]=result.pop('samples');samples[method+'_keys']=np.array(result['keys'])
    # A single set of resampled rows supplies the scenario too: scale-equivariance
    # of means/SD/medians and ratios, but threshold tails must be recomputed.
    # Run the same RNG schedule on scaled data to preserve exact paired resamples.
    scaled=dict(record,segments=[s*np.array([1.5,1.5,1.5,1.]) for s in record['segments']])
    scenario=bootstrap_record(scaled,args.repeats)
    for method,result in scenario['methods'].items():
        samples['B_'+method]=result.pop('samples');samples['B_'+method+'_keys']=np.array(result['keys'])
    (HERE/'analysis').mkdir(exist_ok=True)
    np.savez_compressed(HERE/'analysis/record_bootstrap_samples.npz',**samples)
    counts={k:{str(key):int(np.isfinite(samples[k][:,i]).sum()) for i,key in enumerate(samples[k+'_keys'])}
            for k in samples if not k.endswith('_keys')}
    write_json(HERE/'analysis/bootstrap_valid_counts.json',counts)
    out.update(source_sha256=record['source_sha256'],f13_crosscheck=record['f13_crosscheck'],
               analysis_source_sha256={name:digest(HERE/name) for name in ['analysis_core.py','analyse_record.py']},
               segment_sizes=[len(s) for s in record['segments']],
               alone_fish=len(record['alone_groups']),alone_n=len(record['alone']),
               alone_fish_means=[np.mean(g) for g in record['alone_groups']],
               burst10_record=record['burst10_record'],
               individual_breath_duration_mean=np.mean(record['individual_breath_duration']),
               scenario_B=scenario)
    write_json(HERE/'analysis/record_uncertainty.json',out)
    print(json.dumps(clean(out['point']),indent=2))
    print('Laplace win fractions:',{k:v['winner_fraction']['laplace'] for k,v in out['methods'].items()})

if __name__=='__main__':main()
