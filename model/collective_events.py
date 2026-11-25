"""Collective breathing events and their statistics, from the breaths of one simulation.

A breath is one row of the model output ``episodes13`` (air-release time ``release``,
dive-down time ``exit``, fish index ``agent``, release position ``release_x``/``release_y``).
The definitions below are the ones used for every model result in the article:

* a collective event opens with an air release and absorbs every later release that comes
  no more than ``gap`` seconds after the event's current end (the last dive-down so far);
* event start = first air release, event end = last dive-down;
* collective interval = next event start - previous event end;
* participation = number of distinct fish in the event / group size, with classes
  <20 %, 20-50 %, 50-80 % and >=80 %.

``cluster_episodes`` and ``bout_events_from`` are copied from the project's analysis module
(breathing_study.py) and ``collective_statistics`` from the canonical-metrics function of the
model consolidation (study_out/rz75_paper/W5b_code/analyse.py); the computations are unchanged.
"""
import numpy as np
from scipy.stats import spearmanr


def cluster_episodes(t0, t1, gap=2.0):
    """Index ranges (s, k, end) of collective events: a new event opens when
    the next air release comes more than `gap` s after the running event's
    current end (last dive-down so far)."""
    if len(t0) == 0:
        return []
    out = []
    s, end = 0, t1[0]
    for k in range(1, len(t0)):
        if t0[k] > end + gap:
            out.append((s, k, end))
            s, end = k, t1[k]
        else:
            end = max(end, t1[k])
    out.append((s, len(t0), end))
    return out


def bout_events_from(t0, t1, ag, gap=2.0, observe_k=None, rng=None,
                     observe_front=None, y0=None):
    """Collective events under the paper's definitions, from breath
    episodes (release time t0, dive-down time t1, agent).

    event start = first air release, event end = last dive-down.
    Observation windows mimic the scorer, who timed only the fish that could
    be seen through the front glass: `observe_k` takes the TIMING of each
    event over a random subset of k of its breaths; `observe_front` takes it
    over the breaths released within that many BL of the front wall (y0 <=
    observe_front), and drops events with no observed breath. Membership
    (n, n_breaths) always stays complete.
    """
    ranges = cluster_episodes(t0, t1, gap)
    if observe_k is not None and rng is None:
        rng = np.random.default_rng(0)
    events = []
    for s, k, e in ranges:
        idx = np.arange(s, k)
        if observe_front is not None:
            idx = idx[y0[idx] <= observe_front]
            if len(idx) == 0:
                continue                      # unobserved event
        elif observe_k is not None and len(idx) > observe_k:
            idx = rng.choice(idx, observe_k, replace=False)
        ts, te = float(t0[idx].min()), float(t1[idx].max())
        events.append(dict(t_start=ts, t_end=te, duration=te - ts,
                           n=int(len(np.unique(ag[s:k]))),
                           n_breaths=int(k - s), n_obs=int(len(idx)),
                           span=float(t0[idx].max() - t0[idx].min()),
                           rel_sd=float(np.std(t0[idx]))))
    return events


def _avg(x):
    return float(np.mean(x)) if len(x) else None


def _corr(x, y):
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def collective_statistics(episodes, N, T, burn=300., gap=1.):
    """Summary statistics of one run of length T seconds with N fish.

    Events are built with a separation `gap` of 1 s; only events that start at or after
    `burn` and end at least 2 s before T are kept. Returns ``scalar`` (means and
    fractions) and ``arrays`` (collective intervals, participation, individual
    inter-breath intervals).
    """
    e = np.sort(episodes, order='release', kind='stable')
    bouts = bout_events_from(e['release'], e['exit'], e['agent'], gap=gap)
    bouts = [b for b in bouts if b['t_start'] >= burn and b['t_end'] <= T - 2.]
    starts = np.array([b['t_start'] for b in bouts]); ends = np.array([b['t_end'] for b in bouts])
    parts = np.array([b['n'] / N for b in bouts]); gaps = starts[1:] - ends[:-1]
    cls = np.digitize(parts, [.2, .5, .8]) + 1
    individual = []
    for i in range(N):
        v = e[e['agent'] == i]; ok = (v['exit'][:-1] >= burn) & (v['release'][1:] <= T)
        individual.extend((v['release'][1:] - v['exit'][:-1])[ok])
    freq = np.histogram(parts, [0, .2, .5, .8, 1.01])[0] / len(parts) if len(parts) else np.full(4, np.nan)
    scalar = dict(
        gap_mean=_avg(gaps), gap_sd=float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None,
        short_fraction=_avg(gaps < 5), long_fraction=_avg(gaps > 30),
        participation_mean=_avg(parts),
        class_fractions=[float(f) for f in freq],
        whole_duration_mean=_avg(ends - starts), individual_mean=_avg(individual),
        class_next_rho=_corr(cls[:-1], gaps), class_previous_rho=_corr(cls[1:], gaps),
        n_events=len(parts), n_intervals=len(gaps), n_individual=len(individual))
    return dict(scalar=scalar, arrays=dict(gaps=gaps.tolist(), parts=parts.tolist(),
                                           individual=[float(x) for x in individual]))
