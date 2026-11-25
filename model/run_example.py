"""Run the model once with the reference parameters and print the collective statistics.

    python run_example.py [seconds] [seed]      # defaults: 900 s, seed 9001

The first 300 s are discarded as burn-in. The article uses ten runs of 13,440 s
(seeds 9001-9010); a 900-s run takes a minute or two and gives only a rough estimate
(about 40 collective events), so expect values near, not equal to, the published ones.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import dataclasses
import json
import sys
import time
from pathlib import Path

from rz75_model import RZ75Sim, parameters
from collective_events import collective_statistics

HERE = Path(__file__).resolve().parent
BURN = 300.


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 900.
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 9001
    assert seconds > BURN + 60, 'run longer than the 300-s burn-in'
    p = parameters()
    # The defaults of parameters() are the reference parameters of the article. The saved
    # record also lists switches of mechanisms that this standalone model no longer contains
    # (all inactive), so compare only the parameters the model has.
    saved = json.loads((HERE / 'reference_parameters.json').read_text())
    current = json.loads(json.dumps(dataclasses.asdict(p)))
    assert all(saved[key] == value for key, value in current.items())
    start = time.time()
    sim = RZ75Sim(p, seed=seed, audit_burn=BURN)
    data = sim.run(seconds, sample_dt=1.)        # 1-s samples of positions, headings, reserves, states
    stats = collective_statistics(data['episodes13'], p.N, seconds, burn=BURN)['scalar']
    print(f'{seconds:.0f} s simulated in {time.time() - start:.0f} s (N = {p.N}, seed {seed})')
    print(f"collective events             {stats['n_events']}")
    print(f"mean collective interval (s)  {stats['gap_mean']:.2f}")
    print(f"mean participation (%)        {100 * stats['participation_mean']:.1f}")
    print('participation classes <20 / 20-50 / 50-80 / >=80 (%)  '
          + ' / '.join(f'{100 * f:.1f}' for f in stats['class_fractions']))
    print(f"mean individual inter-breath interval (s)  {stats['individual_mean']:.1f}")


if __name__ == '__main__':
    main()
