"""Figure 6: effects of tank depth and group size.

Reads the saved values in source_data/ beside this file and writes spatial_predictions.pdf/.svg (and source_data/plotted_values.json).
No simulation is run. Usage:

    python plot_fig6.py [output_directory]     # default: this folder

Adapted from the as-run script paper_draft/build_results_figures.py
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

BLUE, GREY, ORANGE, GOLD = '#1b6ca8', '#8c8c8c', '#b8562f', '#a58a2a'
TYPE_COLORS = ['#c97924', '#679ab6', '#8064a2']
plt.rcParams.update({'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.1,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
VALUES = {}


def grid(rows, cols, height, bottom=17, rowgap=26):
    width, left, right, top, gap = 160, 17, 4, 11, 21
    fig, axes = plt.subplots(rows, cols, figsize=(width/25.4, height/25.4), squeeze=False)
    pw = (width-left-right-gap*(cols-1))/cols
    ph = (height-top-bottom-rowgap*(rows-1))/rows
    fig.subplots_adjust(left=left/width, right=1-right/width, top=1-top/height,
        bottom=bottom/height, wspace=gap/pw, hspace=rowgap/ph)
    for i, ax in enumerate(axes.flat):
        ax.annotate(chr(97+i), (0, 1), xycoords='axes fraction',
            xytext=(-30, 7), textcoords='offset points', fontsize=9,
            weight='bold', va='bottom')
    return fig, axes.ravel()


def mean_sd(ax, x, rows, key, color=BLUE, fmt='o-', label=None, scale=1):
    values = np.asarray(rows, dtype=float)*scale
    assert np.isfinite(values).all() and len(values) > 1
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    VALUES[key] = {'replicate_values': values.tolist(), 'mean': mean.tolist(), 'sd': sd.tolist(), 'n': len(values)}
    ax.errorbar(x, mean, yerr=sd, fmt=fmt, color=color, capsize=2, ms=3, label=label)


def save(fig, name):
    fig.canvas.draw()
    fig.savefig(OUT/f'{name}.pdf', metadata={'Creator': 'Saved-results manuscript renderer'})
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


def predictions(rows, groups):
    fig, (a, b, c, d) = grid(2, 2, 118, bottom=20, rowgap=26)
    depths = [2.2, 3.3, 5, 6.6]
    for arm, color, ls, label in [('A0', BLUE, '-', 'Full model'), ('A1', ORANGE, '--', 'No rest zone')]:
        for ax, metric, scale, panel in [(a, 'gap_mean', 1, 'a'), (b, 'participation_mean', 100, 'b')]:
            vals = [rows[(rows.arm == arm) & (rows.depth == z) & rows['type'].isna()].sort_values('seed')[metric].values for z in depths]
            assert all(len(v) == 3 for v in vals)
            mean_sd(ax, depths, np.array(vals).T, f'predictions_{panel}_{arm}', color, 'o'+ls, label, scale)
    a.set(xlabel='Tank depth (BL)', ylabel='Collective interval (s)', xticks=depths, ylim=(0, 32))
    b.set(xlabel='Tank depth (BL)', ylabel='Participation (%)', xticks=depths, ylim=(0, 100))
    a.legend(loc='upper left')
    for prefix, color, ls, label, ns in [('same', BLUE, '-', 'Fixed tank', [2, 5, 10, 25, 50, 100, 200, 400]),
                                       ('density', ORANGE, '--', 'Fixed density', [50, 100, 200, 400])]:
        for ax, metric, scale, panel in [(c, 'gap_mean', 1, 'c'), (d, 'participation_mean', 100, 'd')]:
            qs = [groups[f'{prefix}_N{n}' if not (prefix == 'density' and n == 200) else 'same_N200']['metrics'][metric] for n in ns]
            means = np.array([q['mean'] for q in qs])*scale; sds = np.array([q['sd'] for q in qs])*scale
            VALUES[f'predictions_{panel}_{prefix}'] = dict(x=ns, mean=means.tolist(), sd=sds.tolist())
            assert min(ns) > 0
            ax.errorbar(ns, means, yerr=sds, color=color, ls=ls, marker='o', capsize=2, ms=3, label=label)
    for ax in (c, d):
        ax.set(xlabel='Group size N', xscale='log', xticks=[2, 10, 50, 400])
        ax.set_xticklabels(['2', '10', '50', '400'])
    c.set(ylabel='Collective interval (s)', ylim=(0, 60)); d.set(ylabel='Participation (%)', ylim=(0, 105))
    c.legend(loc='upper right')
    save(fig, 'spatial_predictions')


def main():
    # per_run_metrics.csv: depth runs with (A0) and without (A1) the rest zone, three runs each.
    # groups.json: group-size runs in the fixed tank ('same_N*') and at fixed density ('density_N*').
    rows = pd.read_csv(DATA/'per_run_metrics.csv')
    groups = json.loads((DATA/'groups.json').read_text())
    predictions(rows, groups)
    (DATA/'plotted_values.json').write_text(json.dumps(VALUES, indent=2)+'\n')
    print('Rendered Figure 6 from saved summaries; no simulations.')


if __name__ == '__main__':
    main()
