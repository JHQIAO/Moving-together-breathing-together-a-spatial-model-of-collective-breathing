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
    ex=frames['example_individual_traces']
    for ax,key,ylabel in [(a,'height_bl','Height (BL)'),(b,'oxygen_drive','Oxygen drive')]:
        rows=ex[ex.variable==key]
        ax.fill_between(rows.time,rows.p10,rows.p90,color=GREY,alpha=.25,lw=0,
                        label='10–90% of fish')
        for name,color,fish_id in zip(TYPES,TYPE_COLORS,[0,2,4]):
            ax.plot(rows.time,rows[f'fish_{fish_id}'],color=color,lw=.8,label=name+' fish')
        ax.plot(rows.time,rows.shoal_summary,color='black',lw=1.4,label='Shoal summary')
        ax.set(xlim=(300,420),ylabel=ylabel)
    a.set_ylim(0,3.4)
    a.tick_params(labelbottom=False)
    b.axhline(0,color=GREY,ls=':',lw=.6)
    b.set_xlabel('Time (s)')
    fig.legend(*a.get_legend_handles_labels(),ncol=5,loc='upper center',
               bbox_to_anchor=(.54,.995),columnspacing=.9,handlelength=1.5,handletextpad=.5)
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


def main():
    previous = ROOT / 'study_out/local_work/figure3_types_20261112'
    pins = json.loads((previous/'qa/provenance.json').read_text())
    frames = {}
    # Reuse the verified ten-run numerical results unchanged.
    for name in ['aligned_per_run', 'peak_times', 'window_counts']:
        original = previous/'source_data'/f'{name}.csv'
        assert digest(original) == pins['outputs'][str(original.relative_to(ROOT))]
        assert digest(DATA/f'{name}.csv') == digest(original)
        frames[name] = pd.read_csv(source(DATA/f'{name}.csv'))
    original = ROOT/'study_out/local_work/draft_figure_revision_20261019/source_data/cycle_individual_traces.csv'
    restored = DATA/'example_individual_traces.csv'
    assert digest(restored) == digest(source(original))
    frames['example_individual_traces'] = pd.read_csv(source(restored))
    ex = frames['example_individual_traces']
    # Check the restored curves directly against the saved reference trajectory.
    with np.load(BASE/'trajectories/reference_s9001.npz') as f:
        t, z, q, st, tid = [f[k] for k in ['time','pos','tank','status','type_id']]
    np.testing.assert_array_equal(tid[[0,2,4]], [0,1,2])
    ix = (t >= 300) & (t <= 420)
    drive = np.where(q.astype(float)<.75,.35,.1)*(.75-q.astype(float))
    for key,arr in [('height_bl',z[:,:,2].astype(float)),('oxygen_drive',drive)]:
        rows=ex[ex.variable==key]
        np.testing.assert_array_equal(rows.time,t[ix])
        for fish_id in [0,2,4]:
            np.testing.assert_allclose(rows[f'fish_{fish_id}'],arr[ix,fish_id],rtol=0,atol=1e-14)
        low,high=np.percentile(arr[ix],[10,90],axis=1)
        np.testing.assert_allclose(rows.p10,low,rtol=0,atol=1e-14)
        np.testing.assert_allclose(rows.p90,high,rtol=0,atol=1e-14)
        masked=np.where(st[ix]==0,arr[ix],np.nan)
        summary=np.nanmedian(masked,axis=1) if key=='height_bl' else np.nanmean(masked,axis=1)
        np.testing.assert_allclose(rows.shoal_summary,summary,rtol=0,atol=1e-12)
    assert len(frames['peak_times']) == 5733
    assert frames['aligned_per_run'].seed.nunique() == 10
    plot(frames)
    old=pd.read_csv(previous/'source_data/aligned_ensemble.csv')
    now=pd.read_csv(DATA/'aligned_ensemble.csv')
    pd.testing.assert_frame_equal(old,now,check_exact=False,rtol=0,atol=1e-14)
    values=json.loads((previous/'source_data/summary.json').read_text())
    values.update(example_fish_ids=[0,2,4],example_scope='Individual fish in all states; grey bands over all 200 fish; black summaries schooling-only')
    values.pop('type_height_check',None)
    (DATA/'summary.json').write_text(json.dumps(values,indent=2)+'\n')
    source(Path(__file__).resolve())
    outputs={str(f.relative_to(ROOT)):digest(f) for f in
             [HERE/'cycle.pdf',HERE/'cycle.svg',HERE/'cycle.png',*sorted(DATA.glob('*.csv')),DATA/'summary.json']}
    (QA/'provenance.json').write_text(json.dumps(dict(inputs=INPUTS,outputs=outputs,
        ten_run_provenance=str(previous/'qa/provenance.json'),simulations_run=0,
        restored_individuals_verified=True,ten_run_summaries_unchanged=True),indent=2)+'\n')
    print('Restored original fish 0/2/4; retained ten-run results and all 5,733 windows.')


if __name__ == '__main__':
    main()
