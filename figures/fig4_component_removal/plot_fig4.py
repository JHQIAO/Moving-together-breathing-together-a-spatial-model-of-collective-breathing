"""Figure 4: effects of removing model components (ten simulations per condition).

Reads the saved values in source_data/ beside this file and writes mechanism.pdf/.svg (and source_data/plotted_summary.csv).
No simulation is run. Usage:

    python plot_fig4.py [output_directory]     # default: this folder

Adapted from the as-run script study_out/local_work/figure4_ten_runs_20261112/render_mechanism.py
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
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = HERE / 'source_data'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

BLUE,DARK,BAND='#1b6ca8','#4d4d4d','#e3e3e3'
ROWS=[('A0','Full model',0),('A1','No rest zone',1.5),('A2','No oxygen drive',2.5),
      ('A8','No orientation\nor attraction',4),('A6','No surface avoidance',5.5)]
COLUMNS=[('gap_mean',1.,'gap_mean','Collective\ninterval (s)',
          dict(xscale='log',xlim=(.9,160),xticks=[1,10,100],xticklabels=['1','10','100'])),
         ('small_fraction',100.,'class_1','Small events (%)',
          dict(xlim=(-4,104),xticks=[0,50,100])),
         ('individual_mean',1.,None,'Per-fish breathing\ninterval (s)',
          dict(xscale='log',xlim=(15,1000),xticks=[20,100,500],xticklabels=['20','100','500']))]


def main():
    data_path=DATA/'per_run_metrics.csv'
    record_path=DATA/'record_uncertainty.json'
    metrics=pd.read_csv(data_path)
    assert len(metrics)==50
    for arm,_,_ in ROWS:
        rows=metrics[metrics.arm==arm]
        assert sorted(rows.seed)==list(range(9001,9011)),arm
        assert set(rows.seconds)=={3600. if arm=='A2' else 7500.}
    # Accept only fully classified conditions; mixed flag results need an explicit symbol.
    flags={arm:int(metrics.loc[metrics.arm==arm,'rhythm_flag'].astype(bool).sum()) for arm,_,_ in ROWS}
    assert all(v in [0,10] for v in flags.values()),flags
    record=json.loads(record_path.read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.2,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.2,
        'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True,
        'svg.hashsalt':'mechanism-ten-runs-20261112'})
    W,LEFT,RIGHT,GAP=160.,31.,3.,5.
    pw=(W-LEFT-RIGHT-2*GAP)/3
    TOP,PH,BOTTOM=8.,42.,23.
    H=TOP+PH+BOTTOM
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    gs=GridSpec(1,5,figure=fig,left=LEFT/W,right=1-RIGHT/W,top=1-TOP/H,
                bottom=BOTTOM/H,width_ratios=[pw,GAP,pw,GAP,pw],wspace=0)
    axes=[fig.add_subplot(gs[0,2*j]) for j in range(3)]
    plotted=[];record_values={}
    for j,(ax,(key,scale,rkey,xlabel,settings)) in enumerate(zip(axes,COLUMNS)):
        ax.set_gid('abc'[j])
        if rkey:
            point=record['point'][rkey]*scale
            low,high=np.array(record['methods']['block10']['ci'][rkey])*scale
            assert low<=point<=high
            record_values[key]=dict(point=point,ci95=[low,high])
            ax.axvspan(low,high,color=BAND,lw=0,zorder=0)
            ax.axvline(point,color='black',lw=.8,zorder=1)
        for arm,name,y in ROWS:
            rows=metrics[metrics.arm==arm].sort_values('seed')
            v=rows[key].to_numpy(float)*scale
            assert np.isfinite(v).all()
            m,sd=float(v.mean()),float(v.std(ddof=1))
            if settings.get('xscale')=='log':assert m-sd>0
            assert settings['xlim'][0]<=m-sd and m+sd<=settings['xlim'][1],(arm,key,m,sd)
            plotted.append(dict(arm=arm,condition=name.replace('\n',' '),metric=key,
                                n=10,mean=m,sd=sd,scale=scale,flagged_runs=flags[arm]))
            if arm=='A0':ax.axvline(m,color=BLUE,lw=.8,ls='--',zorder=1)
            color=BLUE if arm=='A0' else DARK
            hollow=flags[arm]==10 and rkey is not None
            ax.errorbar(m,y,xerr=sd,fmt='o',ms=3.6,color=color,
                        mfc='white' if hollow else color,mew=.9,capsize=1.8,elinewidth=.8,zorder=3)
        ax.set(ylim=(6.1,-.6),xlabel=xlabel,**settings)
        ax.set_yticks([r[2] for r in ROWS])
        ax.set_yticklabels([r[1] for r in ROWS] if j==0 else [])
        ax.tick_params(axis='y',length=0);ax.minorticks_off()
        ax.annotate('abc'[j],(0,1),xycoords='axes fraction',xytext=(-3,4),
                    textcoords='offset points',fontsize=9,weight='bold',va='bottom',ha='right')
    axes[0].get_yticklabels()[0].set_color(BLUE)
    handles=[(Patch(color=BAND),Line2D([],[],color='black',lw=.8)),
             Line2D([],[],color=BLUE,ls='--',lw=.8),
             Line2D([],[],color=DARK,marker='o',mfc='white',ls='none',ms=3.6)]
    fig.legend(handles,['Record (95% CI)','Full model','No collective rhythm'],
               handler_map={tuple:HandlerTuple(ndivide=1)},loc='lower center',ncol=3,
               bbox_to_anchor=(.55,.005),handlelength=1.6,columnspacing=2)
    fig.canvas.draw()
    for ext in ['pdf','svg']:fig.savefig(OUT/f'mechanism.{ext}')
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(DATA/'plotted_summary.csv',index=False)
    print('Rendered Figure 4: ten runs per condition, mean +/- 1 SD.')


if __name__=='__main__':main()
