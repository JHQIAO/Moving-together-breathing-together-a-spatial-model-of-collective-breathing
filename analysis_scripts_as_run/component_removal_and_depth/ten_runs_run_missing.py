"""Extend existing ablation protocols to seeds 9004–9010 without changing the engine."""
from pathlib import Path
import os
import sys
import time
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE/'cache/numba')
os.environ['MPLCONFIGDIR'] = str(HERE/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(HERE/'cache')
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'study_out/rz75_paper/W2_mechanism'))
import run_w2
from model_w2 import code_pins, digest, clean, write_json

# Reuse the original audited writer, redirecting all new outputs to this task.
run_w2.OUT = HERE/'runs'


def main():
    jobs = [dict(name=f'{arm}_s{seed}',arm=arm,seed=seed,
                 seconds=3600. if arm=='A2' else 7500.,burn=300.,depth=3.3,type=None)
            for seed in range(9004,9011) for arm in ['A1','A6','A8','A2']]
    pins = code_pins()
    for arm in ['A1','A2','A6','A8']:
        original=ROOT/f'study_out/rz75_paper/W2_mechanism/outputs/records/{arm}_s9001.json'
        assert pins==json.loads(original.read_text())['code_sha256'], 'Historical source mismatch'
    protocol=dict(jobs=jobs,workers=4,code_sha256=pins,
                  runner_sha256=digest(Path(__file__)),new_runs=28,
                  reason='User-requested expansion to ten simulations per condition')
    pp=HERE/'runs/protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==clean(protocol)
    else:write_json(pp,protocol)
    start=time.monotonic();done=[];errors=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(run_w2.one,j):j for j in jobs}
        write_json(HERE/'runs/status.json',dict(status='running',total=28,done=0,
                   runner_pid=os.getpid(),started_unix=time.time()))
        for future in as_completed(fs):
            job=fs[future]
            try:
                result=future.result();done.append(result)
            except Exception as exc:
                result=dict(name=job['name'],error=repr(exc));errors.append(result)
                traceback.print_exc()
            status=dict(status='running',total=28,done=len(done),errors=errors,
                        elapsed_s=time.monotonic()-start,last=result)
            write_json(HERE/'runs/status.json',status)
            print(json.dumps(status),flush=True)
    assert code_pins()==pins,'Source changed during execution'
    write_json(HERE/'runs/status.json',dict(status='failed' if errors else 'complete',
               total=28,done=len(done),errors=errors,elapsed_s=time.monotonic()-start))
    if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
