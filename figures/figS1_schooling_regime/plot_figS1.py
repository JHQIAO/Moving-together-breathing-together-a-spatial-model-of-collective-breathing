"""Supplementary Figure S1: schooling regime and zone sensitivity.

Reads the saved values in source_data/ beside this file and writes figure_S1.pdf/.svg (and the plotted summaries in source_data/).
No simulation is run. Usage:

    python plot_figS1.py [output_directory]     # default: this folder

Adapted from the as-run script study_out/local_work/supplement_rebuild_20261109/build_figures.py (function fig1)
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
# schooling_runs.json: 27 movement-only zone-grid runs and 6 breathing runs with r_o changed by 20%.
# attraction_radius_runs.csv: the reference and the r_a +/-20% breathing runs (three seeds each).
def fig1():
    path=DATA/'schooling_runs.json'
    runs=json.loads(path.read_text())['runs'];s=[]
    fig=plt.figure(figsize=(178/25.4,140/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.88,bottom=.13,top=.94,wspace=.5,hspace=.85)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:]);ct=c.twinx()
    for ax,key,label in [(a,'polarization','Polarisation'),(b,'angular_momentum','Angular momentum')]:
        for j,wo in enumerate([.15,.3,.6]):
            means=[];sds=[]
            for wa in [1.25,2.5,5.]:
                subset=[r for r in runs if r['job']['mode']=='movement_only' and np.isclose(r['params']['ro']-.15,wo) and np.isclose(r['params']['ra']-r['params']['ro'],wa)]
                vals=[r['metrics'][key] for r in subset];assert len(vals)==3
                means.append(np.mean(vals));sds.append(np.std(vals,ddof=1))
                s.append(dict(metric=key,orientation_width=wo,attraction_width=wa,mean=means[-1],sd=sds[-1]))
            ax.errorbar([1.25,2.5,5],means,yerr=sds,color=['#333333',BLUE,'#92C5DE'][j],marker=['o','s','^'][j],capsize=2,elinewidth=.8,label=f'{wo:g} BL')
            if wo==.3:ax.plot(2.5,means[1],marker='*',ms=11,mfc='white',mec='black',mew=.8,zorder=5)
        ax.set(xlabel='Attraction-zone width (BL)',ylabel=label,xticks=[1.25,2.5,5],ylim=(0,1.05 if key=='polarization' else 1))
        ax.legend(title='Orientation-zone width',title_fontsize=7.2,loc='upper right' if key=='angular_momentum' else 'lower right')
    w=pd.read_csv(DATA/'attraction_radius_runs.csv');w=w[(w.phase=='tier1')&(w.burn==300)]
    series=[]
    for name,factor in [('ro −20%',.8),('Reference',1.),('ro +20%',1.2),('ra −20%',.8),('ra +20%',1.2)]:
        if name.startswith('ro'):
            v=[r['metrics'] for r in runs if r['job']['mode']=='breathing' and np.isclose(r['params']['ro'],.45*factor)]
        else:
            arm='control' if name=='Reference' else 'ra_low' if factor<1 else 'ra_high'
            v=w[w.arm==arm].to_dict('records')
        assert len(v)==3
        series.append(dict(setting=name,interval_mean=np.mean([x['gap_mean'] for x in v]),interval_sd=np.std([x['gap_mean'] for x in v],ddof=1),participation_mean=100*np.mean([x['participation_mean'] for x in v]),participation_sd=100*np.std([x['participation_mean'] for x in v],ddof=1)))
    x=np.arange(5)
    c.errorbar(x-.08,[v['interval_mean'] for v in series],yerr=[v['interval_sd'] for v in series],color=BLUE,marker='o',capsize=2,elinewidth=.8,label='Mean interval')
    ct.errorbar(x+.08,[v['participation_mean'] for v in series],yerr=[v['participation_sd'] for v in series],color='#333333',ls='--',marker='s',capsize=2,elinewidth=.8,label='Mean participation')
    c.set(xticks=x,xticklabels=[r'$r_o$ −20%','Reference',r'$r_o$ +20%',r'$r_a$ −20%',r'$r_a$ +20%'],xlabel='Zone-radius perturbation',ylabel='Mean interval (s)',ylim=(0,None));ct.set(ylabel='Mean participation (%)',ylim=(0,100))
    ct.spines['right'].set_visible(True);ct.spines['left'].set_visible(False);c.spines['right'].set_visible(False)
    handles,labels=c.get_legend_handles_labels();h,l=ct.get_legend_handles_labels();c.legend(handles+h,labels+l,loc='lower center',bbox_to_anchor=(.5,1.05),ncol=2)
    for ax,l in [(a,'a'),(b,'b'),(c,'c')]:letter(ax,l)
    save(fig,1,row_groups=[['a','b']]);pd.DataFrame(s).to_csv(DATA/'movement_summary.csv',index=False);pd.DataFrame(series).to_csv(DATA/'breathing_summary.csv',index=False)
if __name__=='__main__':
    fig1();print('Rendered Supplementary Figure S1 from saved run summaries.')
