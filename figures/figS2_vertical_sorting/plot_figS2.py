"""Supplementary Figure S2: vertical sorting and breathing type.

Reads the saved values in source_data/ beside this file and writes figure_S2.pdf/.svg.
No simulation is run. Usage:

    python plot_figS2.py [output_directory]     # default: this folder

Adapted from the as-run script study_out/local_work/supplement_rebuild_20261109/build_figures.py (function fig5; the figure was renumbered S2 in the final supplement)
(kept in analysis_scripts_as_run/). The plotting code is unchanged; only the
file locations differ, and development-only layout audits and provenance
bookkeeping are removed.
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / 'source_data'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.8,'ytick.labelsize':7.8,
    'legend.fontsize':7.2,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
    'lines.linewidth':1.2,'lines.markersize':4,'legend.frameon':False,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.unicode_minus':True,'mathtext.fontset':'dejavusans'})
BLUE='#1b6ca8';GRAY='#9e9e9e'
def letter(ax,label):
    ax.set_label(label);ax.annotate(label,(0,1),xycoords='axes fraction',xytext=(-30,7),textcoords='offset points',weight='bold',fontsize=9,va='bottom')
def save(fig,n,**kw):
    fig.canvas.draw()
    fig.savefig(OUT/f'figure_S{n}.pdf');fig.savefig(OUT/f'figure_S{n}.svg');plt.close(fig)
# sorting.json: reserve-relative-height correlation aligned to lift-off (ten reference simulations,
#   with the shuffled null); sorting_per_run.csv and sorting_per_run_curves/ give the per-run values.
# type_shares_source.json: 'supplementary_type_shares' holds the breathing-type shares of panel b.
def fig2():
    s=json.loads((DATA/'sorting.json').read_text())
    f=json.loads((DATA/'type_shares_source.json').read_text())['supplementary_type_shares']
    fig,(a,b)=plt.subplots(1,2,figsize=(178/25.4,78/25.4));fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.83,wspace=.6)
    x=np.array(s['lags']);m=np.array(s['mean']);sd=np.array(s['sd'])
    a.fill_between(x,s['null_lower'],s['null_upper'],color='#BDBDBD',alpha=.65,label='Shuffled 95% envelope',zorder=2)
    a.plot(x,s['null_mean'],color='#777777',ls=':',lw=.6)
    a.plot(x,m,color=BLUE,label='Schooling fish');a.fill_between(x,m-sd,m+sd,color=BLUE,alpha=.2,lw=0)
    a.axvline(0,color=GRAY,ls=':',lw=.6);a.set(xlim=(-15,15),ylim=(-.36,.24),xticks=[-15,0,15],xlabel='Time from lift-off (s)',ylabel=r'Reserve–relative-height $\rho$');a.legend(loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=6.6)
    m=np.array(f['mean']);sd=np.array(f['sd']);colors=['#c97924','#679ab6','#8064a2'];x=np.arange(2)
    for i,(name,color) in enumerate(zip(['Fast','Medium','Slow'],colors)):
        b.bar(x+(i-1)*.23,m[:,i],width=.22,color=color,label=name)
        b.errorbar(x+(i-1)*.23,m[:,i],yerr=sd[:,i],fmt='none',ecolor='black',elinewidth=.8,capsize=2)
    b.set(xticks=x,xticklabels=['Top 10% at\nlift-off','First breathers'],ylabel='Breathing-type share (%)',ylim=(0,80));b.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,columnspacing=.7,handlelength=1)
    letter(a,'a');letter(b,'b');save(fig,2)
if __name__=='__main__':
    fig2();print('Rendered Supplementary Figure S2 from saved summaries.')
