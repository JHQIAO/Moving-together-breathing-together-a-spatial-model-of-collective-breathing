"""Supplementary scientific figures from pinned saved data and new diagnostics."""
from pathlib import Path
import os,sys,json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['MPLCONFIGDIR']=str(HERE/'mpl_cache');os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,'/Users/junheqiao/.codex/skills/nature-figure/scripts')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.8,'ytick.labelsize':7.8,
    'legend.fontsize':7.2,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
    'lines.linewidth':1.2,'lines.markersize':4,'legend.frameon':False,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.unicode_minus':True,'mathtext.fontset':'dejavusans'})
BLUE='#1b6ca8';GRAY='#9e9e9e';A=json.loads((HERE/'source_data/supplement_values.json').read_text())
OUT=HERE/'figures';QA=HERE/'qa';DATA=HERE/'source_data';base=ROOT/'study_out/rz75_paper'
def letter(ax,label):
    ax.set_label(label);ax.annotate(label,(0,1),xycoords='axes fraction',xytext=(-30,7),textcoords='offset points',weight='bold',fontsize=9,va='bottom')
def save(fig,n,**kw):
    fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=QA/f'S{n}_alignment.json',strict=True,**kw)
    fig.savefig(OUT/f'figure_S{n}.pdf');fig.savefig(OUT/f'figure_S{n}.svg');fig.savefig(OUT/f'figure_S{n}.png',dpi=600);plt.close(fig)
def fig1():
    path=HERE/'schooling/summary.json'
    if not path.exists():print('S1 pending schooling runs');return
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
    w=pd.read_csv(base/'W4_robustness/analysis/metrics_by_seed.csv');w=w[(w.phase=='tier1')&(w.burn==300)]
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
    save(fig,1,row_groups=[['a','b']]);pd.DataFrame(s).to_csv(DATA/'S1_movement_summary.csv',index=False);pd.DataFrame(series).to_csv(DATA/'S1_breathing_summary.csv',index=False)
def fig2():
    d=json.loads((DATA/'isolated_distributions.json').read_text());fig,axes=plt.subplots(1,3,figsize=(178/25.4,73/25.4));fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.71,wspace=.44)
    for i,(ax,fish,name) in enumerate(zip(axes,[6,5,4],['Fast','Medium','Slow'])):
        record=np.concatenate([u['values'] for u in d['record'] if u['fish']==fish]);model=[np.array(u['values']) for u in d['model'] if u['type']==i]
        # Values are raw intervals (not the pooled unit-normalised shape diagnostic).
        for u in [u for u in d['model'] if u['type']==i]:np.testing.assert_allclose(np.mean(u['values']),u['mean'])
        edges=np.arange(0,np.ceil(max(record.max(),max(v.max() for v in model))/10)*10+10,10)
        rd=np.histogram(record,bins=edges,density=True)[0];md=np.array([np.histogram(v,bins=edges,density=True)[0] for v in model]);mean=md.mean(0);sd=md.std(0,ddof=1)
        ax.stairs(rd,edges,color='black',label='Recorded target fish');ax.stairs(mean,edges,color=BLUE,label='Model (10 runs)')
        xx=np.repeat(edges,2)[1:-1];ax.fill_between(xx,np.repeat(mean-sd,2),np.repeat(mean+sd,2),color=BLUE,alpha=.2,lw=0)
        ax.set(xlabel='Inter-breath interval (s)',ylabel='Probability density (per s)' if i==0 else '',ylim=(0,None),title=f'{name} (fish {fish})')
        letter(ax,'abc'[i])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.98),ncol=2)
    save(fig,2)
def fig3():
    # Existing independently audited log-axis figure, preserving all 2-s bins.
    src=ROOT/'study_out/local_work/section31_audit_20261104'
    shutil.copy2(src/'fig2a_log_preview.pdf',OUT/'figure_S3.pdf')
    shutil.copy2(src/'fig2a_log_preview.svg',OUT/'figure_S3.svg')
    import fitz
    doc=fitz.open(OUT/'figure_S3.pdf');doc[0].get_pixmap(dpi=600).save(OUT/'figure_S3.png')
    shutil.copy2(src/'log_histogram_source.csv',DATA/'S3_histogram.csv');shutil.copy2(src/'log_preview_alignment.json',QA/'S3_alignment.json')
def fig4():
    # Existing independently audited source-of-decline figure, same Fig.3 windows.
    src=ROOT/'study_out/local_work/drive_decline_20261104'
    shutil.copy2(src/'drive_comparison.pdf',OUT/'figure_S4.pdf')
    shutil.copy2(src/'drive_comparison.svg',OUT/'figure_S4.svg')
    import fitz
    doc=fitz.open(OUT/'figure_S4.pdf');doc[0].get_pixmap(dpi=600).save(OUT/'figure_S4.png')
    shutil.copy2(src/'plotted_values.csv',DATA/'S4_drive_curves.csv');shutil.copy2(src/'alignment.json',QA/'S4_alignment.json')
def fig5():
    s=json.loads((DATA/'S5_sorting.json').read_text());f=A['source_fig5']['supplementary_type_shares']
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
    letter(a,'a');letter(b,'b');save(fig,5)
def fig6():
    df=pd.read_csv(ROOT/'study_out/local_work/figure5_relayout_20261104/source_data/e_drive_per_run.csv');fig,ax=plt.subplots(figsize=(145/25.4,86/25.4));fig.subplots_adjust(left=.16,right=.96,bottom=.21,top=.93)
    rows=[]
    for category,ls,label in [('small','-','<20% events'),('large','--','≥50% events')]:
        v=df[df.category==category].pivot(index='seed',columns='lag_s',values='next_event_free_fraction');x=v.columns.to_numpy(float);v=100*(1-v);m=v.mean(0).to_numpy();s=v.std(0,ddof=1).to_numpy()
        ax.plot(x,m,color=BLUE,ls=ls,label=label);ax.fill_between(x,m-s,m+s,color=BLUE,alpha=.2,lw=0)
        rows.extend(dict(category=category,time=t,mean=a,sd=b) for t,a,b in zip(x,m,s))
    ax.set(xlim=(0,30),ylim=(0,100),xlabel='Time since event end (s)',ylabel='Windows reaching another event (%)');ax.legend(loc='lower right');save(fig,6);pd.DataFrame(rows).to_csv(DATA/'S6_next_event.csv',index=False)
def fig7():
    s=A['source_fig5']['supplementary_rise_check'];fig,ax=plt.subplots(figsize=(140/25.4,85/25.4));fig.subplots_adjust(left=.15,right=.97,bottom=.22,top=.93)
    ax.errorbar(range(4),s['mean'],yerr=s['sd'],color=BLUE,marker='o',elinewidth=.8,capsize=2)
    ax.set(xticks=range(4),xticklabels=['<20','20–50','50–80','≥80'],xlabel='Event participation (%)',ylabel='Time to next lift-off (s)',ylim=(0,30));save(fig,7)
if __name__=='__main__':
    todo=sys.argv[1:] or ['1','2','3','4','5','6','7']
    for n in todo:globals()['fig'+n]();print('Figure',n,flush=True)
