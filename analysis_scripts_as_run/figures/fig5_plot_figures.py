"""Render saved, checked scientific summaries using Matplotlib only."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['MPLCONFIGDIR'] = str(HERE/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(HERE/'cache')
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import fitz

SKILL = Path('/Users/junheqiao/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

QA = HERE/'qa'
D = json.loads((HERE/'source_data/plotted_values.json').read_text())
BLUE, LIGHT = '#1b6ca8', '#92C5DE'
ISOLATED, BAND, REF = '#8C8C8C', '#E0E0E0', '#9E9E9E'
ORANGE, OLIVE = '#b8562f', '#A6761D'
TYPES = ['Fast', 'Medium', 'Slow']
CLASSES = ['<20', '20–50', '50–80', '≥80']
FONT = font_manager.findfont('Arial', fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'axes.grid': False, 'axes.unicode_minus': True, 'lines.linewidth': 1.2, 'lines.markersize': 4,
    'errorbar.capsize': 2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.labelpad': 3, 'xtick.major.pad': 3, 'ytick.major.pad': 3,
})

def label(ax, letter):
    ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(-23/72, 5/72, ax.figure.dpi_scale_trans),
            fontsize=9, weight='bold', ha='left', va='bottom', clip_on=False)

def errorbar(ax, x, y, error, color, *, horizontal=False, fmt='o', **kwargs):
    return ax.errorbar(x, y, **({'xerr': error} if horizontal else {'yerr': error}),
        fmt=fmt, color=color, markersize=4, elinewidth=.8, capsize=2, capthick=.8, **kwargs)

def export(fig, name, panel_ids=None):
    fig.canvas.draw()
    if len(fig.axes) > 1:
        require_matplotlib_panel_alignment(fig, json_out=QA/f'{name}.alignment.json',
            overlay_svg=QA/f'{name}.alignment.svg', panel_ids=panel_ids,
            require_panel_labels=True)
    # Deliberately no tight bounding box: preserve the requested physical size.
    fig.savefig(HERE/f'{name}.pdf', metadata={'Creator':'Matplotlib; saved-data figure revision'})
    fig.savefig(HERE/f'{name}.svg')
    fig.savefig(HERE/f'{name}.png', dpi=600)
    fig.savefig(QA/f'{name}.preview.png', dpi=300)
    plt.close(fig)

def main_figure():
    fig = plt.figure(figsize=(180/25.4, 115/25.4))
    grid = fig.add_gridspec(2, 3, left=17/180, right=176/180,
        bottom=15/115, top=107/115, wspace=.45, hspace=.55)
    axes = [fig.add_subplot(grid[row,col]) for row in range(2) for col in range(3)]
    a,b,c,d,e,f = axes
    for ax, letter in zip(axes, 'abcdef'): label(ax, letter)

    edges = np.array(D['a']['bin_edges'])
    density = np.array(D['a']['density'])
    y = np.repeat(edges, 2)[1:-1]
    for i in range(2):
        widths = .36 * np.repeat(density[i], 2) / density.max()
        a.fill_betweenx(y, i-widths, i+widths, color=LIGHT, linewidth=0)
    errorbar(a, range(2), D['a']['mean'], D['a']['sd'], 'black', zorder=3)
    a.set(xticks=range(2), xticklabels=['Take part','Miss'], xlim=(-.5,1.5),
        ylim=(3.5,0), yticks=np.arange(0,3.51,.5),
        xlabel=r'Fish with $\tau<\tau_0$ at onset', ylabel='Depth below surface (BL)')
    a.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    low, high = D['b']['below'], D['b']['above']
    b.bar(range(3), low['mean'], width=.6, color=BLUE, linewidth=0, label=r'$\tau<\tau_0$')
    b.bar(range(3), high['mean'], bottom=low['mean'], width=.6, color=LIGHT, linewidth=0, label=r'$\tau\geq\tau_0$')
    errorbar(b, range(3), low['mean'], low['sd'], 'black', fmt='none')
    b.set(xticks=range(3), xticklabels=TYPES, xlabel='Breathing type',
          ylabel='Accepted breaths (%)', ylim=(0,100), yticks=[0,25,50,75,100], xlim=(-.55,2.55))
    b.legend(loc='lower center', bbox_to_anchor=(.5,1), ncol=2, frameon=False,
             handlelength=.8, columnspacing=1.3, handletextpad=.35, borderaxespad=.2)

    for key, color, offset, legend in [('isolated',ISOLATED,-.12,'Isolated'),('shoal',BLUE,.12,'Shoal')]:
        q = D['c'][key]
        errorbar(c, np.arange(3)+offset, q['mean'], q['sd'], color, label=legend)
    c.set(xticks=range(3), xticklabels=TYPES, xlabel='Breathing type', ylabel='Mean inter-breath interval (s)',
          ylim=(0,155), yticks=[0,50,100,150], xlim=(-.5,2.5))
    c.legend(loc='upper left', handlelength=1, borderaxespad=.3)

    errorbar(d, range(4), D['d']['mean'], D['d']['sd'], BLUE, fmt='o-', linewidth=1.2)
    d.set(xticks=range(4), xticklabels=CLASSES, xlabel='Participation (%)',
        ylabel='Floor rest in next interval (s)', ylim=(0,28), yticks=[0,5,10,15,20,25], xlim=(-.25,3.25))

    x = np.array(D['e']['lags']); visible = (x >= 0) & (x <= 30)
    for key, style, legend in [('small','-','<20% events'),('large','--','≥50% events')]:
        mean, sd = np.array(D['e'][key]['mean']), np.array(D['e'][key]['sd'])
        e.fill_between(x[visible], (mean-sd)[visible], (mean+sd)[visible], color=BLUE, alpha=.2, linewidth=0)
        e.plot(x[visible], mean[visible], color=BLUE, linestyle=style, label=legend)
    e.axhline(0, color=REF, linestyle=':', linewidth=.6, zorder=0)
    e.set(xlim=(0,30), xticks=[0,10,20,30], ylim=(-.01,.025), yticks=[-.01,0,.01,.02],
        xlabel='Time since event end (s)', ylabel='Mean oxygen drive')
    e.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    e.legend(loc='upper right', handlelength=1.8, borderaxespad=.3, labelspacing=.3)

    q = D['f']['record']
    f.axvspan(*q['ci'], color=BAND, linewidth=0, zorder=0)
    f.axvline(0, color=REF, linestyle=':', linewidth=.6, zorder=1)
    errorbar(f, q['mean'], 0, [[q['mean']-q['ci'][0]], [q['ci'][1]-q['mean']]], 'black', horizontal=True, zorder=3)
    for i,(key,color) in enumerate([('full',BLUE),('no_rest',ORANGE),('weaker_rest',OLIVE)],1):
        q = D['f'][key]
        errorbar(f, q['mean'], i, q['sd'], color, horizontal=True, zorder=3)
    f.set(yticks=range(4), yticklabels=['Record','Full model','No rest zone','Weaker rest'],
        ylim=(3.5,-.5), xlim=(-.3,.5), xticks=[-.2,0,.2,.4],
        xlabel=r'Participation–next interval $\rho$')
    export(fig, 'recruit', list('abcdef'))

def type_shares():
    fig, ax = plt.subplots(figsize=(100/25.4,75/25.4))
    fig.subplots_adjust(left=.16,right=.97,bottom=.22,top=.95)
    q=D['supplementary_type_shares']; mean=np.asarray(q['mean']); sd=np.asarray(q['sd'])
    for i,(typ,color,marker) in enumerate(zip(TYPES,['#c97924','#679ab6','#8064a2'],['o','s','^'])):
        x=np.arange(2)+(i-1)*.2
        errorbar(ax,x,mean[:,i],sd[:,i],color,fmt=marker,label=typ)
        ax.hlines([q['population_percent'][i]]*2,x-.07,x+.07,color=REF,linestyle=':',linewidth=.6)
    ax.set(xticks=range(2),xticklabels=['Top 10% at\nlift-off','First breathers'],
           ylabel='Type share (%)',ylim=(0,80),yticks=[0,20,40,60,80],xlim=(-.5,1.5))
    ax.legend(loc='upper left',ncol=3,handlelength=.7,columnspacing=.8,handletextpad=.3)
    export(fig,'supplementary_type_shares')

def rise_robustness():
    fig, axes=plt.subplots(1,2,figsize=(150/25.4,75/25.4))
    fig.subplots_adjust(left=16/150,right=146/150,bottom=17/75,top=66/75,wspace=.48)
    a,b=axes
    for ax,letter in zip(axes,'ab'):label(ax,letter)
    q=D['supplementary_rise_check']
    errorbar(a,range(4),q['mean'],q['sd'],BLUE,fmt='o-')
    a.set(xticks=range(4),xticklabels=CLASSES,xlabel='Event participation (%)',
          ylabel='Time to next lift-off (s)',ylim=(0,30),xlim=(-.25,3.25))
    for i,values in enumerate([D['f']['full'],q['rho']]):
        errorbar(b,values['mean'],i,values['sd'],BLUE,horizontal=True)
    b.axvline(0,color=REF,linestyle=':',linewidth=.6)
    b.set(yticks=range(2),yticklabels=['Next interval','Next lift-off'],ylim=(1.65,-.65),
          xlim=(-.05,.5),xticks=[0,.2,.4],xlabel=r'Participation-class correlation $\rho$')
    export(fig,'supplementary_rise_check',['a','b'])

def unchanged_figure6():
    original=ROOT/'paper_draft/figures/spatial_predictions.pdf'
    shutil.copyfile(original,HERE/'figure6.pdf')
    shutil.copyfile(original.with_suffix('.svg'),HERE/'figure6.svg')
    document=fitz.open(HERE/'figure6.pdf')
    document[0].get_pixmap(dpi=600,alpha=False).save(HERE/'figure6.png')
    document[0].get_pixmap(dpi=300,alpha=False).save(QA/'figure6.preview.png')
    assert original.read_bytes()==(HERE/'figure6.pdf').read_bytes()

def main():
    main_figure();type_shares();rise_robustness();unchanged_figure6()
    before=json.loads((QA/'tex_hashes_before.json').read_text())
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'paper_draft').rglob('*.tex')}
    assert before==after
    (QA/'export_metadata.json').write_text(json.dumps(dict(font_file=FONT,
        size_mm=[180,115],dpi=600,pdf_fonttype=42,tex_unchanged=True,
        figure6_pdf_identical=True,model_blue=BLUE,no_rest_orange=ORANGE),indent=2)+'\n')
    print('Exported Fig. 5, supplementary type shares, SI rise check, and unchanged Fig. 6. TeX unchanged.')

if __name__=='__main__':main_figure()
