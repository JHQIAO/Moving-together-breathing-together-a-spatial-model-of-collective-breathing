"""Figure 5: recruitment and reserve replenishment.

Reads the saved values in source_data/ beside this file and writes recruit.pdf/.svg.
No simulation is run. Usage:

    python plot_fig5.py [output_directory]     # default: this folder

Adapted from the as-run script study_out/local_work/figure5_label_20261113/plot_figures.py
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
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / 'source_data'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

# plotted_values.json holds every plotted number; the CSV files beside it give the
# run-level and event-level values behind panels a-e.
D = json.loads((DATA/'plotted_values.json').read_text())
BLUE, LIGHT = '#1b6ca8', '#92C5DE'
ISOLATED, BAND, REF = '#8C8C8C', '#E0E0E0', '#9E9E9E'
ORANGE, OLIVE = '#b8562f', '#A6761D'
TYPES = ['Fast', 'Medium', 'Slow']
CLASSES = ['<20', '20–50', '50–80', '≥80']
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
    # Deliberately no tight bounding box: preserve the requested physical size.
    fig.savefig(OUT/f'{name}.pdf', metadata={'Creator':'Matplotlib; saved-data figure revision'})
    fig.savefig(OUT/f'{name}.svg')
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

if __name__=='__main__':
    main_figure()
    print('Rendered Figure 5: six panels from saved values.')
