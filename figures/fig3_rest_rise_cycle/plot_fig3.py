"""Figure 3: oxygen drive and the collective rest-rise cycle.

Reads the saved values in source_data/ beside this file and writes cycle.pdf/.svg (and source_data/aligned_ensemble.csv, the plotted curves of panel c).
No simulation is run. Usage:

    python plot_fig3.py [output_directory]     # default: this folder

Adapted from the as-run script study_out/local_work/figure3_individuals_10runs_20261112/build_cycle.py
(kept in analysis_scripts_as_run/). The plotting code is unchanged; only the
file locations differ, and development-only layout audits and provenance
bookkeeping are removed.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / 'source_data'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

SEEDS = list(range(9001, 9011))
LAGS = np.arange(-15, 16)
TYPES = ['Fast', 'Medium', 'Slow']
TYPE_COLORS = ['#c97924', '#679ab6', '#8064a2']
SIGNALS = {'D': ('Mean oxygen drive', '#b8562f'),
           'B': ('Active breaths', '#a58a2a'),
           'H': ('Median height', '#1b6ca8')}
GREY = '#8c8c8c'


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
    for ext in ['pdf','svg']:fig.savefig(OUT/f'cycle.{ext}')
    plt.close(fig)


def main():
    # example_individual_traces: panels a,b (one reference simulation, seed 9001, 300-420 s).
    # aligned_per_run, peak_times, window_counts: panels c,d (ten reference simulations).
    frames = {name: pd.read_csv(DATA/f'{name}.csv') for name in
              ['aligned_per_run', 'peak_times', 'window_counts', 'example_individual_traces']}
    assert len(frames['peak_times']) == 5733
    assert frames['aligned_per_run'].seed.nunique() == 10
    plot(frames)
    print('Rendered Figure 3: fish 0/2/4 traces; ten-run summaries over 5,733 windows.')


if __name__ == '__main__':
    main()
