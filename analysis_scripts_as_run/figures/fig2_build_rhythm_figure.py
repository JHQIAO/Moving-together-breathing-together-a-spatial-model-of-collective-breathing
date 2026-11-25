"""Render the two-panel collective comparison from saved results; no simulations."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / 'paper_draft'
BASE = ROOT / 'study_out/rz75_paper'
WORK = ROOT / 'study_out/local_work/figure1_record_model_20261013'
QA = WORK / 'figure_qa'
OUT = HERE / 'figures'
os.environ.setdefault('MPLCONFIGDIR', str(WORK / 'mpl_cache'))
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/junheqiao/.codex/skills/nature-figure/scripts')
import numpy as np
import pandas as pd
from scipy.stats import laplace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE = '#1b6ca8'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.1,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
INPUTS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    provenance = QA / 'provenance.json'
    if args.check:
        saved = json.loads(provenance.read_text())
        for kind in ['inputs', 'outputs']:
            for path, expected in saved[kind].items():
                assert digest(ROOT/path) == expected, path
        print('PASS: two-panel Figure 1 inputs and outputs unchanged; no simulations.')
        return
    QA.mkdir(parents=True, exist_ok=True)
    source(Path(__file__).resolve())
    h = pd.read_csv(source(BASE/'W1_reference/analysis/histogram_source.csv'))
    u = json.loads(source(BASE/'W1_reference/analysis/record_uncertainty.json').read_text())
    e = json.loads(source(BASE/'W1_reference/analysis/ensemble.json').read_text())['summary']
    saved_fits = json.loads(source(BASE/'W8_results_gaps/analysis/summary.json').read_text())

    # Exactly the saved Figure 1a/b observations, bins and uncertainty summaries.
    rec = h[h.series == 'Recorded']
    edges = np.r_[rec.lower.values, rec.upper.values[-1]]
    assert np.allclose(np.diff(edges), 2)
    seeds = list(range(9001, 9011))
    densities = np.array([h[h.series == str(s)].density.values for s in seeds])
    assert densities.shape == (10, len(edges)-1)
    assert np.isfinite(densities).all()
    assert np.allclose(np.sum(densities*np.diff(edges), axis=1), 1)
    assert np.isclose(np.sum(rec.density.values*np.diff(edges)), 1)
    mean, sd = densities.mean(axis=0), densities.std(axis=0, ddof=1)
    # The saved fit used intervals normalised by their mean. Transform its
    # location and scale back to seconds; this does not estimate a new fit.
    norm_params = saved_fits['g1_shoal']['record']['point']['fit_params']['laplace']
    params = np.asarray(norm_params)*u['point']['gap_mean']
    assert np.isclose(params[0], u['point']['gap_median'])
    x = np.linspace(0, edges[-1], 600)
    fitted_density = laplace.pdf(x, *params)

    keys = [f'class_{i}' for i in range(1, 5)]
    record_bars = np.array([u['point'][k] for k in keys])*100
    record_ci = np.array([u['methods']['block10']['ci'][k] for k in keys])*100
    model_bars = np.array([e[k]['mean'] for k in keys])*100
    model_sd = np.array([e[k]['sd'] for k in keys])*100
    assert np.isclose(record_bars.sum(), 100) and np.isclose(model_bars.sum(), 100)

    width, height = 160, 82
    fig, (a, b) = plt.subplots(1, 2, figsize=(160/25.4, 82/25.4))
    panel_width = (width-17-4-21)/2
    fig.subplots_adjust(left=17/width, right=1-4/width, top=1-11/height,
                        bottom=28/height, wspace=21/panel_width)
    for label, ax in zip('ab', [a, b]):
        ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-30, 7),
                    textcoords='offset points', fontsize=9, weight='bold', va='bottom')
    a.stairs(rec.density.values, edges, color='black', fill=True, facecolor=(0, 0, 0, .07))
    a.stairs(rec.density.values, edges, color='black', label='Recorded data', zorder=4)
    a.stairs(mean, edges, color=BLUE, label='Model')
    a.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(np.maximum(0, mean-sd), 2),
                   np.repeat(mean+sd, 2), color=BLUE, alpha=.18, lw=0)
    a.plot(x, fitted_density, color='black', ls=':', label='Recorded Laplace fit')
    a.set(xlim=(0, edges[-1]), xlabel='Collective interval (s)', ylabel='Probability density (per s)')
    xx = np.arange(4)
    b.bar(xx-.18, record_bars, .36, color='black')
    b.errorbar(xx-.18, record_bars,
               yerr=[record_bars-record_ci[:, 0], record_ci[:, 1]-record_bars],
               fmt='none', color='black', capsize=2)
    b.bar(xx+.18, model_bars, .36, color=BLUE)
    b.errorbar(xx+.18, model_bars, yerr=model_sd, fmt='none', color=BLUE, capsize=2)
    b.set(xticks=xx, xticklabels=['<20', '20–50', '50–80', '≥80'],
          xlabel='Participation (%)', ylabel='Events (%)', ylim=(0, 65))
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(.53, .015), handlelength=2.4, columnspacing=2.0)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=QA/'alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT/'rhythm_collective.pdf')
    fig.savefig(OUT/'rhythm_collective.svg')
    fig.savefig(QA/'rhythm_collective.png', dpi=300)
    plt.close(fig)
    plotted = dict(edges=edges.tolist(), record_density=rec.density.tolist(),
        model_density_mean=mean.tolist(), model_density_sd=sd.tolist(),
        record_laplace_params_seconds=params.tolist(),
        record_participation=record_bars.tolist(), record_participation_ci=record_ci.tolist(),
        model_participation=model_bars.tolist(), model_participation_sd=model_sd.tolist())
    outputs = {str(p.relative_to(ROOT)): digest(p)
               for p in [OUT/'rhythm_collective.pdf', OUT/'rhythm_collective.svg']}
    provenance.write_text(json.dumps(dict(inputs=INPUTS, outputs=outputs, plotted_values=plotted,
        n_model_simulations=len(seeds), record_intervals=u['point']['n_intervals'],
        simulations_run=0, excluded_observations=0), indent=2)+'\n')
    print('Rendered Figure 1: two collective panels; all saved observations retained.')


if __name__ == '__main__':
    main()
