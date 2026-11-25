"""Render ten-run component-removal results without altering model or manuscript."""
from pathlib import Path
import os
import sys
import json
import hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ['MPLCONFIGDIR']=str(HERE/'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(HERE/'cache')
sys.dont_write_bytecode=True
sys.path.insert(0,'/Users/junheqiao/.codex/skills/nature-figure/scripts')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE,DARK,BAND='#1b6ca8','#4d4d4d','#e3e3e3'
ROWS=[('A0','Full model',0),('A1','No rest zone',1.5),('A2','No oxygen drive',2.5),
      ('A8','No orientation\nor attraction',4),('A6','No surface avoidance',5.5)]
COLUMNS=[('gap_mean',1.,'gap_mean','Collective\ninterval (s)',
          dict(xscale='log',xlim=(.9,160),xticks=[1,10,100],xticklabels=['1','10','100'])),
         ('small_fraction',100.,'class_1','Small events (%)',
          dict(xlim=(-4,104),xticks=[0,50,100])),
         ('individual_mean',1.,None,'Per-fish breathing\ninterval (s)',
          dict(xscale='log',xlim=(15,1000),xticks=[20,100,500],xticklabels=['20','100','500']))]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data_path=HERE/'source_data/per_run_metrics.csv'
    record_path=ROOT/'study_out/rz75_paper/W1_reference/analysis/record_uncertainty.json'
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
    require_matplotlib_panel_alignment(fig,json_out=HERE/'qa/alignment.json',
        overlay_svg=HERE/'qa/alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        strict=True,require_panel_labels=True)
    for ext in ['pdf','svg']:fig.savefig(HERE/f'mechanism.{ext}')
    fig.savefig(HERE/'mechanism.png',dpi=600)
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(HERE/'source_data/plotted_summary.csv',index=False)
    # A separate replacement snippet keeps the original manuscript untouched.
    def pm(values, digits=1, factor=1.):
        values=np.asarray(values,float)*factor
        sd=float(values.std(ddof=1))
        while sd>0 and round(sd,digits)==0:
            digits+=1
        return rf'${values.mean():.{digits}f}\pm {sd:.{digits}f}$'
    # This compact data snippet can replace the rows in the existing SI layout.
    table_rows=[]
    for arm,name,_ in ROWS:
        v=metrics[metrics.arm==arm]
        row=' & '.join([name.replace('\n',' '),str(int(v.seconds.iloc[0])),
                       pm(v.gap_mean),pm(v.individual_mean),pm(v.small_fraction,factor=100),
                       pm(v.polarization,digits=3),pm(v.low_height_fraction,factor=100),
                       f'{flags[arm]}/10'])+r'\\'
        table_rows.append(row)
    (HERE/'table_S6_rows.tex').write_text('\n'.join(table_rows)+'\n')
    (HERE/'table_S6_note.txt').write_text(
        'Means ±1 sample SD across seeds 9001–9010; burn-in 300 s. '
        'The full-model records are restricted to the first 7,500 s to match the original ablation horizon. '
        'Low height is the fraction of post-burn-in samples with whole-shoal median height below 1 BL. '
        'A run is flagged when mean schooling polarisation P<0.2 and low-height time is at least 95%. '
        'This is a post-hoc descriptive flag, not a statistical test for periodicity.\n')
    (HERE/'caption.txt').write_text(r'''\textbf{Effects of removing model components.} (a) Mean collective interval. (b) Fraction of events with $<20\%$ participation. (c) Mean individual inter-breath interval. Points and error bars show means $\pm1$ SD across ten simulations per condition (seeds 9001--9010). Runs lasted 7,500 s, except the no-oxygen-drive condition (3,600 s); all exclude a 300-s burn-in. The full-model reference runs are analysed over the same 7,500-s horizon. Grey bands and black lines indicate recorded $95\%$ CIs and point estimates; dashed lines mark the full model. Open symbols mark collective-event statistics for the condition meeting the descriptive no-clear-rhythm criterion in all ten runs (Supplementary Table~\ref{supp-tab:ablations}). Removing orientation and attraction retains repulsion.'''+'\n')
    provenance=dict(inputs={str(p.relative_to(ROOT)):digest(p) for p in [data_path,record_path,Path(__file__).resolve()]},
                    outputs={str(p.relative_to(ROOT)):digest(p) for p in [HERE/'mechanism.pdf',HERE/'mechanism.svg',HERE/'mechanism.png']},
                    record_values=record_values,simulations_per_condition=10,flags=flags,
                    uncertainty='Mean ±1 sample SD across independent seeds; recorded 95% block-bootstrap CI',
                    manuscript_modified=False)
    (HERE/'qa/render_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered Figure4: ten runs per condition, orientation label, mean ±1 SD.')


if __name__=='__main__':main()
