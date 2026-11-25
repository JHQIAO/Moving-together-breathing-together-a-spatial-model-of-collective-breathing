"""Render evidence tables with literal numbers, mean +/- sample SD by run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
A=json.loads((HERE/'source_data/supplement_values.json').read_text());OUT=HERE/'tables'
def ms(v,d=2,scale=1):
    if v.get('mean') is None:return '---'
    m=f"{v['mean']*scale:.{d}f}"
    return '$'+m+(r'\pm '+f"{v['sd']*scale:.{d}f}" if v.get('sd') is not None else '')+'$'
def rec(k,d=2,scale=1):
    v=A['record']['point'][k];lo,hi=A['record']['methods']['block10']['ci'][k]
    return f'{v*scale:.{d}f} [{lo*scale:.{d}f}, {hi*scale:.{d}f}]'
def table(n,title,label,heads,widths,rows,notes=''):
    # Keep the full table inside the 166-mm text block without shrinking type.
    available=165-len(widths)*6*25.4/72.27
    if sum(widths)>available:widths=[round(w*available/sum(widths),2) for w in widths]
    spec=''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+'mm}' for w in widths)
    h=' & '.join(heads)+r'\\'
    text=r'\begingroup\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.16}'+'\n'
    aliases={3:'tab:full-comparison',7:'tab:reselection',8:'tab:depth'}
    alias=r'\label{'+aliases[n]+'}' if n in aliases else ''
    text+=r'\begin{longtable}{'+spec+'}\n'+r'\caption{'+title+'}'+r'\label{'+label+'}'+alias+r'\\'+'\n'
    text+=r'\toprule'+'\n'+h+'\n'+r'\midrule\endfirsthead'+'\n'
    text+=r'\multicolumn{'+str(len(heads))+r'}{l}{\textit{Table S'+str(n)+r' continued}}\\\toprule'+'\n'+h+'\n'+r'\midrule\endhead'+'\n'+r'\bottomrule\endfoot'+'\n'
    text+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\end{longtable}'+'\n'
    if notes:text+=r'\noindent\footnotesize '+notes+'\n'
    text+=r'\endgroup'+'\n';(OUT/f'table_S{n}.tex').write_text(text)

def main():
    # Keep verified active parameter values; remove the old provenance column to
    # make room for legible descriptions, and add operational constants.
    old=(ROOT/'paper_draft/supplement/compact/tab_parameters.tex').read_text()
    rows=[]
    for line in old.splitlines():
        if ' & ' not in line or line.startswith('Symbol'):continue
        x=line.rstrip('\\').split(' & ')
        rows.append([x[0],x[1],x[2],x[3]])
    rows += [[r'$\tau_{\min}^{\rm init}$','Minimum initial reserve','0.5','1'],
        [r'$T_\eta$','Noise reference interval','0.1','s'],
        [r'$\epsilon_t$','Downward-turn threshold: $d_{i,z}<-\epsilon_t$','0.05','1'],
        [r'$\Delta_{\rm sep}$','Maximum gap merged into an event','1','s'],
        ['Sampling','Saved trajectory interval','1','s'],['Burn-in','Initial time excluded','300','s'],
        ['Run duration','Reference shoal simulation','13,440','s'],
        [r'$N_f,N_m,N_s$','Fast, medium, slow fish counts','67, 67, 66','fish']]
    table(1,'Reference parameters and implementation constants.','tab:parameters',['Symbol','Description','Value','Unit'],[23,65,45,13],rows,
      r'Reserve and steering weights are dimensionless. The radii are cumulative: orientation width $r_o-r_r=0.30$ BL and attraction width $r_a-r_o=2.50$ BL. Horizontal boundaries are periodic. Breathing types are denoted $b_i$, to distinguish them from position coordinates. Inactive legacy options are not model parameters.')
    rows=[]
    for v in A['isolated']['targets']:
        rows.append([v['type'],str(v['fish']),str(v['n']),f"{v['target']:.3f}",f"{v['rate']:.9f}",ms(v['model'],3)])
    table(2,'Isolated-fish targets, depletion rates and interval statistics.','tab:calibration',
        ['Type','Fish','Intervals','Recorded mean (s)',r'$\lambda_b$ (s$^{-1}$)','Model mean (s)'],[18,10,17,27,33,37],rows,
        r'The recorded target pools both trials of the indicated fish, weighting each interval equally. Model means are means $\pm1$ sample SD across ten runs per type (seeds 9601--9610). The following shape statistics instead use all six recorded fish and both trials, and all 30 isolated simulations. Each fish--trial or simulation is divided by its own mean before pooling; brackets denote 95\% cluster-bootstrap intervals (2,000 replicates).')
    s='\n'+r'\begin{center}\small\begin{tabular}{lll}\toprule Statistic & Record & Model\\\midrule'+'\n'
    for key,label in [('gap_cv','CV'),('gap_skew','Skewness')]:
        cells=[]
        for src in ['record_shape','model_shape']:
            v=A['isolated'][src][key];cells.append(f"{v['point']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
        s+=label+' & '+' & '.join(cells)+r'\\'+'\n'
    s+=r'\bottomrule\end{tabular}\end{center}'+'\n'
    with (OUT/'table_S2.tex').open('a') as f:f.write(s)
    rows=[];ref=A['reference_summary']
    for label,k,tol,d,scale in [
        ('Mean interval (s)','gap_mean',r'$|\bar I-15.3|<2.865$ s',2,1),
        ('Within-run interval SD (s)','gap_sd',r'$\leq8.5$ s',2,1),
        ('Intervals $>30$ s (\%)','long_fraction',r'$\leq6\%$',2,100),
        ('Intervals $<5$ s (\%)','short_fraction',r'$\leq10\%$',2,100),
        ('Events $\geq80\%$ (\%)','class_4',r'$\leq10\%$',2,100),
        ('Events $<20\%$ (\%)','class_1',r'$\leq30\%$',2,100)]:
        rows.append([label,'Final target',tol,rec(k,d,scale),ms(ref[k],d,scale)])
    rows += [[r'Historical slab duration','Final guard',r'$\leq2.7415$ s','Not comparable','Not retained'],
        ['Polarisation','Final guard',r'$\geq0.80$','---',ms(ref['polarization'],3)],
        ['Empty-reserve time','Final guard',r'$\leq0.5\%$','---',r'$0.0109\%$ (historical)'],
        ['Isolated mean intervals','Final guard',r'All errors $\leq5\%$','Table S2','Table S2'],
        ['Drive-off low-height time','Final guard',r'$>80\%$','---',r'$76.2\%$ (historical)'],
        [r'Events 20--50\% (\%)','Earlier target','No final tolerance',rec('class_2',2,100),ms(ref['class_2'],2,100)],
        [r'Events 50--80\% (\%)','Earlier target','No final tolerance',rec('class_3',2,100),ms(ref['class_3'],2,100)],
        ['Full interval distribution','Earlier target','Distance objectives','Fig. S3','Fig. S3'],
        ['Duration distribution','Earlier target','Distance objectives','Historical','Not retained'],
        ['Peak participation','Earlier target','Improvement check','Historical','Not retained'],
        ['Individual shoal intervals','Earlier target','30--45 s in Step 13','---',ms(ref['individual_mean'])],
        ['Surface/band occupancy and exposure','Earlier guard','Branch-specific','---','Not a final target'],
        ['Cohesion, event counts, numerical integrity','Earlier guard','Branch-specific','---','Not a final target'],
        ['AIC ranking','Diagnostic','No final tolerance','Laplace','Laplace: 10/10'],
        [r'Previous-interval $\rho$','Diagnostic','No final tolerance',rec('class_previous_rho',3),ms(ref['class_previous_rho'],3)],
        [r'Next-interval $\rho$','Diagnostic','No final tolerance',rec('class_next_rho',3),ms(ref['class_next_rho'],3)]]
    table(3,'Parameter-selection targets, development checks and later diagnostics.','tab:selection',
        ['Statistic','Role','Tolerance or rule','Record [95\% CI]','Frozen reference'],[36,21,34,30,28],rows,
        r'Final means and guards were applied to three development runs, not to the ten-run column shown here. The table reports the frozen reference on seeds 9001--9010 as mean $\pm$ SD. The drive-off cell is the original one-run guard, not a full-model statistic. RZ75 passed ten of eleven final checks; its drive-off result failed. It was subsequently adopted as the reference. Earlier distribution objectives used all four participation classes. Thus absence from the final checklist does not imply an independent empirical test. Historical observation-slab statistics are listed only to document selection; they are not current observation outputs.')
    df=pd.read_csv(HERE/'source_data/S4_distribution_fits.csv');rows=[]
    for _,v in df.iterrows():
        rows.append([v.dataset,str(int(v.n)),v.family,f'{v.aic:.2f}',f'{v.delta_aic:.2f}',
            '---' if pd.isna(v['shape']) else f"{v['shape']:.5f}",f'{v.location:.5f}',f'{v.scale:.5f}'])
    table(4,'Separate maximum-likelihood fits to the recording and each reference simulation.','tab:aic',
        ['Data','$n$','Family','AIC',r'$\Delta$AIC','Shape','Location','Scale'],[17,10,23,20,19,19,19,21],rows,
        r'All fits have two estimated parameters. Laplace and normal have free location and scale; exponential also has free location and scale. Gamma and lognormal have location fixed to zero, with shape and scale estimated. Shape is the gamma shape or lognormal log-standard-deviation. Location and scale use seconds (the lognormal scale is the median at zero location). $\Delta$AIC is relative to the lowest AIC within each dataset; raw AIC values are not compared between datasets. Laplace wins for the recording and all ten runs.')
    rows=[]
    for k,dt,dr in [('control','0.025','0.05'),('half_dt','0.0125','0.05'),('reaction_010','0.025','0.10')]:
        v=A['numerical'][k];rows.append([dt,dr,ms(v['gap_mean']),ms(v['participation_mean'],2,100)]+[ms(v[x],2,100) for x in ['small_fraction','medium_fraction','large_fraction','huge_fraction']])
    table(5,'Integration-step and reaction-interval sensitivity.','tab:numerics',
        [r'$\Delta t$ (s)',r'$\Delta_r$ (s)','Interval (s)','Mean part. (\%)',r'$<20\%$',r'20--50\%',r'50--80\%',r'$\geq80\%$'],[15,15,24,24,19,19,19,19],rows,
        r'Each cell is a mean $\pm1$ sample SD across three paired runs (seeds 9001--9003), each 7,500 s with 300 s discarded. The final four columns are percentages of events in each participation class. Depletion rates are unchanged. Changing $\Delta_r$ changes the behavioural update timescale and is distinct from integration-step sensitivity.')
    labels={'A0':'Full model','A1':'No rest zone','A2':'No oxygen drive','A8':'No social steering','A6':'No surface avoidance','A3':'No drive or rest','A4':'Symmetric drive','A5':'No neighbour dilution','A7':'Identical depletion rates','A9':'Rest damping everywhere','A10':'No drive, stronger rest'}
    rows=[]
    for arm,v in A['ablations'].items():
        z=v['statistics'];rows.append([labels[arm],str(int(v['seconds'][0]))]+[ms(z[k],1,100 if k=='small_fraction' else 1) for k in ['gap_mean','individual_mean','small_fraction']]+[ms(z['polarization'],3),ms(z['bottom_fraction'],1,100),f"{v['no_clear_rhythm_runs']}/3"])
    table(6,'Ablation statistics and the descriptive no-clear-rhythm flag.','tab:ablations',
        ['Condition','Run (s)','Collective interval (s)','Individual interval (s)',r'$<20\%$ events (\%)','$P$','Low height (\%)','Flagged runs'],[34,14,24,24,20,16,17,15],rows,
        r'Means $\pm$ SD across seeds 9001--9003; burn-in 300 s. The first five rows underlie main Fig. 4. Low height is the fraction of time with whole-shoal median height below 1 BL. A run is flagged when mean schooling polarisation $P<0.2$ and low-height time is at least 95\%. This post-hoc descriptive rule reproduces the original qualitative no-social-steering flag; it is not a statistical test for periodicity. Remaining arms are additional mechanism checks.')
    rows=[]
    for label,k,d,scale in [('Mean interval (s)','gap_mean',2,1),('Interval SD (s)','gap_sd',2,1),('Intervals $<5$ s (\%)','short_fraction',2,100),('Intervals $>30$ s (\%)','long_fraction',2,100),
        (r'$<20\%$ events (\%)','class_1',2,100),(r'20--50\% events (\%)','class_2',2,100),(r'50--80\% events (\%)','class_3',2,100),(r'$\geq80\%$ events (\%)','class_4',2,100),
        (r'Previous-interval $\rho$','class_previous_rho',3,1),(r'Next-interval $\rho$','class_next_rho',3,1)]:
        rows.append([label,rec(k,d,scale)]+[ms(v['statistics'][k],d,scale) for v in A['variants'].values()])
    rows.append(['AIC winner','Laplace']+[', '.join(f'{name}: {v["aic_winners"].count(name)}/{v["n"]}' for name in sorted(set(v['aic_winners']))) for v in A['variants'].values()])
    for i,(key,label) in enumerate([('gap_cv','Isolated CV'),('gap_skew','Isolated skewness')]):
        rr=[]
        for source in ['record_shape','model_shape']:
            v=A['isolated'][source][key];rr.append(f"{v['point']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
        for source in ['No rest zone','Weaker rest']:
            v=A['variants'][source]['isolated_shape'];rr.append(f"{v['point'][i]:.3f} [{v['ci'][i][0]:.3f}, {v['ci'][i][1]:.3f}]")
        rows.append([label]+rr)
    table(7,'Rest-zone variants compared with the recording.','tab:rest-variants',
        ['Statistic','Record [95\% CI]','Full model ($n=10$)','No rest ($n=3$)','Weaker rest ($n=3$)'],[35,30,29,29,29],rows,
        r'Collective model cells show means $\pm$ SD. Full-model runs use seeds 9001--9010 and 13,440 s; no-rest runs use 9001--9003 and 7,500 s; weaker-rest runs use 9301--9303 and 7,500 s. All discard 300 s. Rest strengths are 0.30, 0 and 0.20, respectively. The weaker-rest rate-estimation procedure retained the original three depletion rates (zero updates); its slow-type check error was $-5.42\%$, just outside the 5\% check tolerance. Isolated shape cells are pooled, unit-normalised estimates [95\% cluster-bootstrap CI], using 30 full-model runs or nine runs per variant (three per type), not across-run SD. Recorded class frequencies are observed categories; an exact recorded mean participation is unavailable.')
    rows=[]
    for v in A['depth']:
        rows.append(['Full' if v['arm']=='A0' else 'No rest',str(v['depth']),'200','10','3']+[ms(v['statistics'][k],3 if k=='class_next_rho' else 2,100 if k=='participation_mean' else 1) for k in ['gap_mean','participation_mean','class_next_rho']])
    for v in A['groups']:
        rho=v['statistics']['class_next_rho'];cell=ms(rho,3)
        if rho['n']!=v['n']:cell+=f" ({rho['n']}/{v['n']})"
        rows.append([{'same':'Fixed tank','density':'Fixed density','shared reference':'Shared reference'}[v['protocol']],'3.3',str(v['N']),f"{10*v['N']/200 if v['protocol']=='density' else 10:g}",str(v['n']),ms(v['statistics']['gap_mean']),ms(v['statistics']['participation_mean'],2,100),cell])
    table(8,'Depth and group-size sweeps.','tab:sweeps',
        ['Protocol','$L_z$ (BL)','$N$','$L_x$ (BL)','$n$','Interval (s)','Participation (\%)',r'Next-interval $\rho$'],[29,17,11,16,8,26,26,25],rows,
        r'Cells show mean $\pm$ SD; $n$ counts runs. All have $L_y=3.3$ BL and use fixed depletion rates. Depth sweeps use seeds 9001--9003, 7,500 s and 300 s burn-in. Fixed-tank group sweeps use seeds 9201--9205 for $N\leq50$ and 9201--9203 for larger groups; fixed-density runs use 9201--9203 at $N=50,100,400$. Fixed density changes $L_x=10N/200$ while keeping the other dimensions fixed. The shared $N=200$ point retains the 13-run ensemble plotted in the depth/group-size figure: ten 13,440-s reference runs and three 7,500-s group-sweep runs. Other group-size runs last 7,500 s. For correlations, parenthesised counts give runs with a defined value; a single valid run has no SD. Constant participation makes Spearman correlation undefined.')
    print('Wrote eight tables')
if __name__=='__main__':main()
