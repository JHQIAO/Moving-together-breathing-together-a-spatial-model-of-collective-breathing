"""Figure 3 from ten pinned reference records; no simulations or model changes."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'study_out/rz75_paper/W1_reference'
DATA, QA = HERE / 'source_data', HERE / 'qa'
os.environ['MPLCONFIGDIR'] = str(HERE / 'mpl_cache')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/junheqiao/.codex/skills/nature-figure/scripts')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

SEEDS = list(range(9001, 9011))
LAGS = np.arange(-15, 16)
TYPES = ['Fast', 'Medium', 'Slow']
TYPE_COLORS = ['#c97924', '#679ab6', '#8064a2']
SIGNALS = {'D': ('Mean oxygen drive', '#b8562f'),
           'B': ('Active breaths', '#a58a2a'),
           'H': ('Median height', '#1b6ca8')}
GREY = '#8c8c8c'
INPUTS = {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source(path):
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def peak(values):
    eligible = np.flatnonzero((LAGS >= -5) & (LAGS <= 12))
    i = eligible[np.argmax(values[eligible])]
    if -5 < LAGS[i] < 12:
        a, b, c = values[i-1:i+2]
        den = a - 2*b + c
        if den < 0:
            return float(LAGS[i] + np.clip(.5*(a-c)/den, -.5, .5))
    return float(LAGS[i])


def analyse():
    aligned, type_aligned, peaks, counts, examples = [], [], [], [], []
    for seed in SEEDS:
        rec = json.loads(source(BASE / f'records/reference_s{seed}.json').read_text())
        p = rec['params']
        path = source(BASE / f'trajectories/reference_s{seed}.npz')
        assert INPUTS[str(path.relative_to(ROOT))] == rec['sha256']
        assert p['N'] == 200 and rec['seconds'] == 13440 and rec['burn'] == 300
        with np.load(path) as f:
            t = f['time'].astype(float)
            z = f['pos'][:, :, 2].astype(float)
            q = f['tank'].astype(float)
            school = f['status'] == 0
            tid, ep, ev = f['type_id'], f['episodes13'], f['events']
        np.testing.assert_array_equal(np.diff(t), 1.)
        np.testing.assert_array_equal(np.unique(tid), [0, 1, 2])
        drive = np.where(q < p['need_setpoint'], p['upward_strength'],
                         p['need_down']) * (p['need_setpoint'] - q)
        n_school = school.sum(1)
        assert np.all(n_school > 0)
        school_drive = (drive * school).sum(1) / n_school
        school_z = np.where(school, z, np.nan)
        school_d = np.where(school, drive, np.nan)
        school_height = np.nanmedian(school_z, axis=1)
        median = np.median(z, axis=1)
        active = (np.searchsorted(np.sort(ep['release']), t, side='right') -
                  np.searchsorted(np.sort(ep['exit']), t, side='right')).astype(float)
        assert np.all(active >= 0)
        lifts = np.flatnonzero((median[:-1] < 1) & (median[1:] >= 1)) + 1
        lifts = lifts[(t[lifts] >= rec['burn']) & (t[lifts] <= rec['seconds'])]
        completed = {(int(e['agent']), float(e['depart'])) for e in ep}
        pending = [float(e['time']) for e in ev if e['to_state'] == 1 and
                   (int(e['agent']), float(e['time'])) not in completed]
        cutoff = np.nextafter(min(pending), -np.inf) if pending else rec['seconds']
        use = lifts[(t[lifts]-15 >= rec['burn']) & (t[lifts]+15 <= cutoff)]
        indices = use[:, None] + LAGS
        np.testing.assert_array_equal(t[indices], t[use, None]+LAGS)
        # Same windows as the independently computed ten-run sorting analysis.
        sorting = source(ROOT / f'study_out/local_work/supplement_rebuild_20261109/source_data/S5_sorting_s{seed}.npz')
        with np.load(sorting) as f:
            np.testing.assert_array_equal(t[use], f['liftoffs'])
        counts.append(dict(seed=seed,postburn_liftoffs=len(lifts),
                           complete_windows=len(use),excluded_edge_windows=len(lifts)-len(use),
                           observation_end=float(cutoff),run_duration_s=rec['seconds']))
        windows = {'D': school_drive[indices], 'B': active[indices], 'H': median[indices]}
        for signal, values in windows.items():
            assert np.isfinite(values).all()
            aligned.extend(dict(seed=seed,lag_s=int(lag),signal=signal,mean=float(value))
                           for lag, value in zip(LAGS, values.mean(0)))
        for i, onset in enumerate(t[use]):
            row = dict(seed=seed,liftoff_s=float(onset))
            row.update({key: peak(values[i]) for key,values in windows.items()})
            row.update(drive_to_breaths=row['B']-row['D'],
                       breaths_to_height=row['H']-row['B'],drive_to_height=row['H']-row['D'])
            peaks.append(row)
        height_by_type, drive_by_type, n_by_type = [], [], []
        for k, name in enumerate(TYPES):
            mask = school[:, tid == k]
            n = mask.sum(1)
            assert np.all(n > 0), (seed, name, 'empty schooling type')
            h = np.nanmedian(np.where(mask, z[:, tid == k], np.nan), axis=1)
            d = (drive[:, tid == k] * mask).sum(1) / n
            height_by_type.append(h); drive_by_type.append(d); n_by_type.append(n)
            for lag, hh, dd in zip(LAGS,h[indices].mean(0),d[indices].mean(0)):
                type_aligned.append(dict(seed=seed,lag_s=int(lag),type=name,
                                         median_height_bl=float(hh),mean_drive=float(dd)))
        # Weight type means by current membership to recover the schooling mean.
        np.testing.assert_allclose(np.sum(np.array(drive_by_type)*np.array(n_by_type),axis=0)
                                   / n_school, school_drive,rtol=1e-12,atol=1e-14)
        if seed == 9001:
            ix = (t >= 300) & (t <= 420)
            h10,h90 = np.nanpercentile(school_z[ix],[10,90],axis=1)
            d10,d90 = np.nanpercentile(school_d[ix],[10,90],axis=1)
            for j, i in enumerate(np.flatnonzero(ix)):
                row = dict(time_s=t[i],height_p10=h10[j],height_p90=h90[j],
                           drive_p10=d10[j],drive_p90=d90[j],
                           school_median_height=school_height[i],school_mean_drive=school_drive[i],
                           n_schooling=int(n_school[i]),whole_population_median_height=median[i])
                for k,name in enumerate(TYPES):
                    row.update({name.lower()+'_median_height':height_by_type[k][i],
                                name.lower()+'_mean_drive':drive_by_type[k][i],
                                name.lower()+'_n_schooling':int(n_by_type[k][i])})
                examples.append(row)
        print(f'{seed}: {len(use)} complete windows',flush=True)
    frames = {name:pd.DataFrame(rows) for name,rows in [
        ('aligned_per_run',aligned),('type_aligned_per_run',type_aligned),
        ('peak_times',peaks),('window_counts',counts),('example_type_traces',examples)]}
    assert len(frames['peak_times']) == 5733
    for name,frame in frames.items():
        frame.to_csv(DATA / f'{name}.csv', index=False)
    return frames


def letter(ax, value, dy=7):
    ax.set_gid(value)
    ax.annotate(value,(0,1),xycoords='axes fraction',xytext=(-32,dy),
                textcoords='offset points',fontsize=9,weight='bold',va='bottom')


def plot(frames):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'lines.linewidth':1.2,'legend.frameon':False,
        'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True,
        'svg.hashsalt':'figure3-types-ten-runs-20261112'})
    fig=plt.figure(figsize=(160/25.4,148/25.4))
    gs=fig.add_gridspec(3,2,height_ratios=[1,1,1.35],left=18/160,right=156/160,
                      top=129/148,bottom=15/148,hspace=.65,wspace=.62)
    a,b=fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:])
    c,d=fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])
    for ax,value in zip([a,b,c,d],'abcd'):letter(ax,value,0 if value in 'cd' else 7)
    ex=frames['example_type_traces']
    for ax,key,summary in [(a,'height','school_median_height'),(b,'drive','school_mean_drive')]:
        ax.fill_between(ex.time_s,ex[key+'_p10'],ex[key+'_p90'],color=GREY,alpha=.22,lw=0,
                        label='10–90% of schooling fish')
        for name,color in zip(TYPES,TYPE_COLORS):
            col=name.lower()+('_median_height' if key=='height' else '_mean_drive')
            ax.plot(ex.time_s,ex[col],color=color,lw=1.2,label=name)
        ax.plot(ex.time_s,ex[summary],color='black',lw=1.25,label='All schooling fish')
        ax.set_xlim(300,420)
    a.set(ylabel='Median height (BL)',ylim=(0,3.4))
    a.tick_params(labelbottom=False)
    b.axhline(0,color=GREY,ls=':',lw=.6)
    b.set(ylabel='Mean oxygen drive',xlabel='Time (s)')
    handles,labels=a.get_legend_handles_labels()
    order=[1,2,3,4,0]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],ncol=5,
               loc='upper center',bbox_to_anchor=(.54,.995),columnspacing=.85,
               handlelength=1.4,handletextpad=.45)
    aligned=[]
    for key,(name,color) in SIGNALS.items():
        vals=frames['aligned_per_run'].query('signal == @key').pivot(index='seed',columns='lag_s',values='mean').loc[SEEDS,LAGS].to_numpy()
        m,sd=vals.mean(0),vals.std(0,ddof=1)
        span=np.ptp(m);assert span>0
        scaled=(m-m.min())/span
        c.plot(LAGS,scaled,color=color,lw=1.2,label=name)
        c.fill_between(LAGS,scaled-sd/span,scaled+sd/span,color=color,alpha=.17,lw=0)
        aligned.extend(dict(signal=key,lag_s=int(lag),mean=mean,sd=ss,scaled_mean=scaled_mean,
                            scaled_sd=ss/span) for lag,mean,ss,scaled_mean in zip(LAGS,m,sd,scaled))
    pd.DataFrame(aligned).to_csv(DATA/'aligned_ensemble.csv',index=False)
    c.axvline(0,color=GREY,ls=':',lw=.6)
    c.set(xlabel='From lift-off (s)',ylabel='Scaled mean (0–1)',xlim=(-15,15),
          xticks=[-15,0,15],ylim=(-.05,1.12),yticks=[0,.5,1])
    fig.legend(*c.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.348),
               ncol=3,fontsize=7,handlelength=1.3,columnspacing=1.3,borderaxespad=0)
    pk=frames['peak_times']
    bp=d.boxplot([pk[k] for k in SIGNALS],vert=False,whis=(10,90),showfliers=True,
                 widths=.5,patch_artist=True,
                 flierprops=dict(marker='.',markersize=1.5,alpha=.25,markeredgewidth=0,markerfacecolor='#555555'),
                 medianprops=dict(color='black',lw=.9),whiskerprops=dict(lw=.8),
                 capprops=dict(lw=.8),boxprops=dict(lw=.8))
    for box,(_,color) in zip(bp['boxes'],SIGNALS.values()):box.set(facecolor=color,alpha=.5)
    d.set(yticks=[1,2,3],yticklabels=['Drive','Breaths','Height'],
          xlabel='Peak time from lift-off (s)',xlim=(-6,13),xticks=[-5,0,5,10])
    d.axvline(0,color=GREY,ls=':',lw=.6)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/'alignment.json',
        overlay_svg=QA/'alignment.svg',strict=True,require_panel_labels=True,
        row_groups=[['c','d']],column_groups=[['a','b']])
    for ext in ['pdf','svg']:fig.savefig(HERE/f'cycle.{ext}')
    fig.savefig(HERE/'cycle.png',dpi=600)
    plt.close(fig)


def report(frames):
    pk=frames['peak_times'];ts=frames['type_aligned_per_run']
    type_summary={}
    for name in TYPES:
        subset=ts[ts.type==name]
        prior=subset[subset.lag_s.between(-5,-1)].groupby('seed').median_height_bl.mean()
        at=subset[subset.lag_s==0].median_height_bl
        type_summary[name]=dict(pre_liftoff_minus5_to_minus1_height_mean=float(prior.mean()),
                                pre_liftoff_height_sd=float(prior.std(ddof=1)),
                                lift_off_height_mean=float(at.mean()),lift_off_height_sd=float(at.std(ddof=1)))
    values=dict(n_runs=10,seeds=SEEDS,example_seed=9001,example_window_s=[300,420],
        example_scope='Current schooling fish only (status 0), separately at every sample',
        type_colors=dict(zip(TYPES,TYPE_COLORS)),complete_windows=len(pk),
        counts=frames['window_counts'].to_dict('records'),type_height_check=type_summary,
        peak_search_s=[-5,12],peak_refinement='Interior parabola with shift clipped to ±0.5 s; first maximum for ties',
        paired_peak_lags={key:dict(median=float(pk[key].median()),mean=float(pk[key].mean()),
                                    positive_fraction=float((pk[key]>0).mean()))
                          for key in ['drive_to_breaths','breaths_to_height','drive_to_height']},
        c_uncertainty='±1 sample SD across ten equally weighted run-level mean curves',
        d_summary='Pooled window peak times: median, quartiles, 10th–90th percentile whiskers and outliers',
        simulations_run=0)
    (DATA/'summary.json').write_text(json.dumps(values,indent=2)+'\n')
    caption=r'''\textbf{Oxygen drive and the collective rest--rise cycle.} (a,b) Median height (a) and mean oxygen drive $F_\tau$ (b) of each breathing type among currently schooling fish, shown over 300--420 s of reference run 9001. Orange, blue and purple denote fast-, medium- and slow-depleting fish. Black lines show the corresponding summaries across all schooling fish; grey bands span their 10th--90th percentiles. Fish currently breathing are excluded from these summaries. (c) Mean schooling drive, active breathing episodes and median height of all fish, aligned to lift-off: the upward crossing of $1\,\mathrm{BL}$ by the whole-population median. Curves are averaged within each run and then equally across ten reference simulations (seeds 9001--9010), and independently scaled to $0$--$1$; shading shows $\pm1$ SD across run means. (d) Peak times relative to lift-off in 5,733 complete windows from the same ten simulations. Complete windows contain all 1-s samples from $-15$ to $+15\,\mathrm{s}$ after burn-in and before the observation cutoff. Peaks are sought from $-5$ to $+12\,\mathrm{s}$. Boxes show the interquartile range and median; whiskers span the 10th--90th percentiles, with points beyond them. BL, body length.'''
    (HERE/'caption.txt').write_text(caption+'\n')
    (HERE/'supplement_window_update.txt').write_text(
        'The ten 13,440-s reference runs (seeds 9001–9010) provide 587, 576, 576, 581, 571, 565, 583, 564, 567 and 563 complete windows, respectively (5,733 in total). Fourteen edge windows are excluded from 5,747 post-burn-in lift-offs. Completeness refers only to temporal coverage; it does not require a prominent, unique or interior peak. All other lift-off and peak definitions remain unchanged.\n')
    outputs={str(path.relative_to(ROOT)):digest(path) for path in
             [HERE/'cycle.pdf',HERE/'cycle.svg',HERE/'cycle.png',*sorted(DATA.glob('*.csv')),DATA/'summary.json']}
    (QA/'provenance.json').write_text(json.dumps(dict(inputs=INPUTS,outputs=outputs,
        simulations_run=0,scope='Figure only; manuscript sources not edited'),indent=2)+'\n')
    print(json.dumps(values['paired_peak_lags'],indent=2))


if __name__=='__main__':
    for folder in (DATA,QA):folder.mkdir(exist_ok=True)
    source(Path(__file__).resolve())
    result=analyse()
    plot(result)
    report(result)
