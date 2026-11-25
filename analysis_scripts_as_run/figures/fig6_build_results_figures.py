"""Reorganise manuscript figures using saved summaries only; never run a model.

Original pack assets stay unchanged. The output provenance records every input
hash and plotted value. --check verifies those inputs and exported files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'study_out/rz75_paper'
WORK = ROOT / 'study_out/local_work/figure4d_decision_reserve_20261013'
QA = WORK / 'figure_qa'
OUT = HERE / 'figures'
os.environ.setdefault('MPLCONFIGDIR', str(WORK / 'mpl_cache'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/Users/junheqiao/.codex/skills/nature-figure/scripts')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, GREY, ORANGE, GOLD = '#1b6ca8', '#8c8c8c', '#b8562f', '#a58a2a'
TYPE_COLORS = ['#c97924', '#679ab6', '#8064a2']
plt.rcParams.update({'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.1,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
INPUTS, VALUES = {}, {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def js(path):
    return json.loads(source(path).read_text())


def csv(path):
    return pd.read_csv(source(path))


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


def point_range(ax, x, q, key, color=BLUE, label=None, scale=1, horizontal=False):
    mean, low, high = [float(q[k])*scale for k in ('mean', 'min', 'max')]
    assert low <= mean <= high
    VALUES[key] = {'mean': mean, 'min': low, 'max': high}
    limits = [[mean-low], [high-mean]]
    if horizontal:
        ax.errorbar(mean, x, xerr=limits, fmt='o', color=color, capsize=2, ms=3, label=label)
    else:
        ax.errorbar(x, mean, yerr=limits, fmt='o', color=color, capsize=2, ms=3, label=label)


def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=QA/f'{name}.alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT/f'{name}.pdf', metadata={'Creator': 'Saved-results manuscript renderer'})
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(QA/f'{name}.png', dpi=300)
    plt.close(fig)


def participation(decisions, groups, solo):
    """Access, reserve state, recruitment order and individual consequence."""
    vals = [js(p)['scalar'] for p in sorted((BASE/'W6_behaviour_limits/analysis').glob('reference_s????.json'))]
    assert len(vals) == 10 and decisions['metadata']['n_shoal'] == 10
    assert decisions['metadata']['n_solo'] == 30
    width, height, left, right, top, bottom, gap, rowgap = 160, 145, 17, 4, 11, 20, 18, 30
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    cell_width = (width-left-right-5*gap)/6
    panel_height = (height-top-bottom-rowgap)/2
    gs = GridSpec(2, 6, figure=fig, left=left/width, right=1-right/width,
                  top=1-top/height, bottom=bottom/height,
                  wspace=gap/cell_width, hspace=rowgap/panel_height)
    axes = [fig.add_subplot(gs[0, 2*i:2*i+2]) for i in range(3)]
    axes.extend([fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])])
    a, b, c, d, e = axes
    for label, ax in zip('abcde', axes):
        ax.set_gid(label)
        ax.annotate(label, (0, 1), xycoords='axes fraction',
                    xytext=(-30, 7), textcoords='offset points', fontsize=9,
                    weight='bold', va='bottom')
    keys = ['ready_participant_depth_event_mean', 'missed_depth_event_mean']
    depth = js(ROOT/'study_out/local_work/figure4a_depth_distributions_20261013/depth_distributions.json')
    edges = np.asarray(depth['bin_edges_bl'])
    densities = np.asarray(depth['density_mean'])
    assert depth['n'] == len(vals) and np.all(np.diff(edges) > 0)
    np.testing.assert_allclose((densities * np.diff(edges)).sum(axis=1), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(depth['mean_by_simulation'], [[v[k] for k in keys] for v in vals], rtol=0, atol=1e-10)
    # Mirrored empirical histograms: common bins and a common width scale.
    # These show fish-event depths, not distributions of the event averages.
    y = np.repeat(edges, 2)[1:-1]
    width_scale = .36 / densities.max()
    for i, density in enumerate(densities):
        widths = width_scale * np.repeat(density, 2)
        a.fill_betweenx(y, i-widths, i+widths, color=BLUE, alpha=.25,
                        edgecolor=BLUE, linewidth=.7, zorder=1)
    VALUES['participation_a_distribution'] = dict(bin_edges_bl=edges.tolist(),
        density_mean=densities.tolist(), n=depth['n'], weighting=depth['weighting'],
        mean=depth['mean'], sd=depth['sd'])
    mean_sd(a, range(2), [[v[k] for k in keys] for v in vals], 'participation_a', color='#202020', fmt='o')
    a.set(xticks=range(2), xticklabels=['Take part', 'Miss event'], ylim=(0, 3.3),
          xlim=(-.5, 1.5), ylabel='Depth below surface (BL)', xlabel='Fish with low reserves')
    types, stages = ['Fast', 'Medium', 'Slow'], ['First', 'Early', 'Middle', 'Late']
    for i, typ in enumerate(types):
        for arm, offset, color, label in [('shoal', -.12, BLUE, 'Shoal'), ('alone', .12, GREY, 'Isolated')]:
            q = decisions[arm][typ]['full_fraction']
            point_range(b, i+offset, q, f'participation_b_{arm}_{typ}', color, label if i == 0 else None, 100)
    b.set(xticks=range(3), xticklabels=types, ylim=(-4, 100), yticks=[0, 25, 50, 75, 100],
          xlim=(-.5, 2.5), ylabel='Breaths with reserve ≥0.75 (%)')
    b.legend(loc='upper left')
    for typ, color in zip(types, TYPE_COLORS):
        qq = [decisions['order'][typ][s]['full_fraction'] for s in stages]
        m, lo, hi = [100*np.array([q[k] for q in qq]) for k in ('mean', 'min', 'max')]
        VALUES['participation_c_'+typ] = dict(mean=m.tolist(), min=lo.tolist(), max=hi.tolist())
        c.plot(range(4), m, color=color, marker='o', ms=3, label=typ)
        c.fill_between(range(4), lo, hi, color=color, alpha=.17, lw=0)
    c.set(xticks=range(4), xticklabels=stages, xlabel='Release order',
          ylabel='Participants with reserve ≥0.75 (%)', ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    c.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=3,
             handlelength=.7, columnspacing=.3, handletextpad=.25, fontsize=7)
    runs = sorted((r for r in decisions['per_run'] if r['arm'] == 'Shoal'), key=lambda r: r['seed'])
    assert [r['seed'] for r in runs] == decisions['metadata']['shoal_seeds']
    bottom = np.zeros(3)
    for metric, key, color, label in [('need_fraction', 'below', ORANGE, 'Reserve <0.75'),
                                      ('full_fraction', 'at_or_above', GOLD, 'Reserve ≥0.75')]:
        replicas = 100*np.array([[r['types'][t][metric] for t in types] for r in runs])
        mean, low, high = [100*np.array([decisions['shoal'][t][metric][k] for t in types])
                           for k in ('mean', 'min', 'max')]
        assert np.isfinite(replicas).all() and ((replicas >= 0) & (replicas <= 100)).all()
        np.testing.assert_allclose(mean, replicas.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(low, replicas.min(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(high, replicas.max(axis=0), rtol=0, atol=1e-12)
        VALUES['participation_d_'+key] = dict(mean=mean.tolist(), min=low.tolist(), max=high.tolist(),
            replicate_values=replicas.tolist(), n=len(runs), reserve_time='accepted breathing decision')
        d.bar(range(3), mean, bottom=bottom, width=.65, color=color, label=label)
        if key == 'below':
            d.errorbar(range(3), mean, yerr=[mean-low, high-mean], fmt='none',
                       color='#202020', capsize=2, lw=.9, zorder=3)
        bottom += mean
    assert np.allclose(bottom, 100)
    np.testing.assert_allclose(np.array(VALUES['participation_d_below']['replicate_values']) +
                               np.array(VALUES['participation_d_at_or_above']['replicate_values']),
                               100, rtol=0, atol=1e-12)
    d.set(xticks=range(3), xticklabels=types, ylabel='Breaths (%)', ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    d.legend(loc='upper center', bbox_to_anchor=(.5, -.13), fontsize=7, handlelength=1,
             ncol=2, columnspacing=1, title='At accepted breathing decision', title_fontsize=7)
    for i, typ in enumerate(['fast', 'medium', 'slow']):
        point_range(e, i-.1, solo['summary'][typ], 'participation_e_solo_'+typ, GREY,
                    'Isolated' if i == 0 else None)
        point_range(e, i+.1, groups['combined_N200']['metrics'][typ+'_mean'],
                    'participation_e_shoal_'+typ, BLUE, 'Shoal' if i == 0 else None)
    e.set(xticks=range(3), xticklabels=types, ylabel='Per-fish interval (s)', ylim=(0, 155), xlim=(-.5, 2.5))
    e.legend(loc='upper left')
    save(fig, 'participation')


def recovery(rows):
    base = rows[(rows.arm == 'A0') & (rows.depth == 3.3) & rows['type'].isna()]
    assert len(base) == 3
    fig, (a, b, c) = grid(1, 3, 78, bottom=24)
    mean_sd(a, range(4), base[[f'class{i}_rest_mean' for i in range(1, 5)]].values, 'recovery_a')
    a.set(xticks=range(4), xticklabels=['<20', '20–50', '50–80', '≥80'],
          xlabel='Participation (%)', ylabel='Rest within next interval (s)')
    ds = []
    for seed in range(9001, 9004):
        with np.load(source(BASE/f'W2_mechanism/analysis/A0_s{seed}_recovery.npz')) as z:
            ds.append({k: z[k] for k in z.files})
    for key, ls, label in [('small', '-', '<20%'), ('large', '--', '≥50%')]:
        v = np.array([z[key] for z in ds]); m, sd = v.mean(axis=0), v.std(axis=0, ddof=1)
        VALUES['recovery_b_'+key] = dict(x=ds[0]['lags'].tolist(), mean=m.tolist(), sd=sd.tolist(), n=3)
        # Retain the original figure's 0–30 s display window. Select its path
        # explicitly so PDF geometry audits do not count invisible clipped paths.
        visible = (ds[0]['lags'] >= 0) & (ds[0]['lags'] <= 30)
        x = ds[0]['lags'][visible]
        b.plot(x, m[visible], color=BLUE, ls=ls, label=label)
        b.fill_between(x, (m-sd)[visible], (m+sd)[visible], color=BLUE, alpha=.17, lw=0)
    b.axhline(0, color=GREY, ls=':', lw=.6)
    b.set(xlabel='From event end (s)', ylabel='Mean oxygen drive', xlim=(0, 30))
    b.legend(loc='upper left', bbox_to_anchor=(0, 1.16), ncol=2, handlelength=1.5)
    rec = js(BASE/'W1_reference/analysis/record_uncertainty.json')
    mean = rec['point']['class_next_rho']; low, high = rec['methods']['block10']['ci']['class_next_rho']
    point_range(c, 0, dict(mean=mean, min=low, max=high), 'recovery_c_record', 'black', horizontal=True)
    q = js(BASE/'W1_reference/analysis/ensemble.json')['summary']['class_next_rho']
    point_range(c, 1, dict(mean=q['mean'], min=q['minimum'], max=q['maximum']), 'recovery_c_reference', horizontal=True)
    other = rows[(rows.arm == 'A1') & (rows.depth == 3.3) & rows['type'].isna()]
    perturb = csv(BASE/'W4_robustness/analysis/metrics_by_seed.csv')
    perturb = perturb[(perturb.arm == 'kappa_bottom_low') & (perturb.phase == 'tier2')]
    if 'burn' in perturb: perturb = perturb[perturb.burn == 300]
    values = [('no_rest', other.class_next_rho.values, ORANGE),
              ('weaker_rest', perturb.class_next_rho.values, GOLD)]
    for i, (name, v, color) in enumerate(values, 2):
        assert len(v) == 3
        point_range(c, i, dict(mean=v.mean(), min=v.min(), max=v.max()), 'recovery_c_'+name, color, horizontal=True)
    c.axvline(0, color=GREY, ls=':', lw=.6)
    c.set(yticks=range(4), yticklabels=['Record', 'Model', 'No rest', 'Weaker rest'],
          xlabel='Size–next interval\ncorrelation', xlim=(-.3, .5), ylim=(3.45, -.45), xticks=[-.2, 0, .2, .4])
    c.tick_params(axis='y', labelsize=7)
    save(fig, 'recovery')


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


def diagnostics():
    refs = []
    for p in sorted((BASE/'W6_behaviour_limits/analysis').glob('reference_s????_derived.npz')):
        with np.load(source(p)) as z: refs.append({k: z[k] for k in ['rank_depart', 'random_depart', 'response', 'offsets']})
    assert len(refs) == 10
    fig, (a, b, c, d) = grid(2, 2, 130, bottom=21, rowgap=31)
    bins = np.linspace(0, 1, 21); x = (bins[1:]+bins[:-1])/2
    def band(ax, x, values, key, color=BLUE, ls='-', label=None):
        v = np.asarray(values); m, sd = np.nanmean(v, axis=0), np.nanstd(v, axis=0, ddof=1)
        VALUES[key] = dict(x=x.tolist(), mean=m.tolist(), sd=sd.tolist(), n=len(v))
        ax.plot(x, m, color=color, ls=ls, label=label)
        ax.fill_between(x, m-sd, m+sd, color=color, alpha=.17, lw=0)
    for key, color, ls, label in [('rank_depart', BLUE, '-', 'Departure'), ('random_depart', GREY, '--', 'Random')]:
        values = [np.histogram(r[key][np.isfinite(r[key])], bins, density=True)[0] for r in refs]
        band(a, x, values, 'diagnostics_a_'+key, color, ls, label)
    a.set(xlabel='Height rank within shoal', ylabel='Probability density', xlim=(0, 1)); a.legend(loc='upper left')
    response = [np.nanmean(r['response'], axis=0) for r in refs]
    for ax, idx, name, ylabel in [(b, 0, 'b', 'Non-participant Δheight (BL)'), (c, 2, 'c', 'School polarisation')]:
        band(ax, refs[0]['offsets'], [r[idx] for r in response], 'diagnostics_'+name)
        ax.axvline(0, color=GREY, ls=':', lw=.6)
        ax.set(xlabel='From event start (s)', ylabel=ylabel, xlim=(-10, 20))
    b.axhline(0, color=GREY, ls=':', lw=.6)
    ac = csv(BASE/'RESULTS_PACK/analysis/w8_acf.csv')
    for name, color, ls in [('Record raw', 'black', '-'), ('Record detrended', 'black', '--'), ('RZ75 raw', BLUE, '-'), ('RZ75 detrended', BLUE, '--')]:
        q = ac[ac.series == name]
        VALUES['diagnostics_d_'+name] = q[['lag', 'mean', 'low', 'high']].to_dict('records')
        d.plot(q.lag, q['mean'], color=color, ls=ls, label=name.replace('RZ75', 'Model'))
        d.fill_between(q.lag, q.low, q.high, color=color, alpha=.13, lw=0)
    d.axhline(0, color=GREY, ls=':', lw=.6)
    d.set(xlabel='Interval lag', ylabel='Spearman autocorrelation', xticks=range(1, 6), xlim=(.8, 5.2))
    d.legend(loc='upper center', bbox_to_anchor=(.5, 1.32), ncol=2, columnspacing=.7, handlelength=1.2, fontsize=7)
    save(fig, 'schooling_diagnostics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--participation-only', action='store_true',
                        help='Update the participation figure and retain the other saved figure exports.')
    selection.add_argument('--recovery-only', action='store_true',
                        help='Update the recovery figure and retain the other saved figure exports.')
    args = parser.parse_args()
    selected = 'participation' if args.participation_only else 'recovery' if args.recovery_only else None
    manifest = QA/'provenance.json'
    if args.check:
        saved = json.loads(manifest.read_text())
        for path, h in saved['inputs'].items(): assert digest(ROOT/path) == h, path
        for path, h in saved['outputs'].items(): assert digest(ROOT/path) == h, path
        print('PASS: saved input and figure output hashes unchanged; no simulations.'); return
    QA.mkdir(parents=True, exist_ok=True); OUT.mkdir(exist_ok=True)
    previous = None
    if selected:
        previous = json.loads(manifest.read_text())
        # Inputs to preserved panels must still match their recorded provenance.
        for path, h in previous['inputs'].items():
            if ROOT/path != Path(__file__).resolve():
                assert digest(ROOT/path) == h, path
        for path, h in previous['outputs'].items():
            assert digest(ROOT/path) == h, path
    source(__file__)
    decisions = js(BASE/'W11_decisions/analysis/summary.json')
    groups = js(BASE/'W3_predictions/analysis/groups.json')
    solo = js(BASE/'W3_predictions/analysis/solo_reference.json')
    rows = csv(BASE/'W2_mechanism/analysis/per_run_metrics.csv')
    if not args.recovery_only:
        participation(decisions, groups, solo)
    if not args.participation_only:
        recovery(rows)
    if not selected:
        predictions(rows, groups); diagnostics()
    outputs = {str((OUT/f'{name}.{ext}').relative_to(ROOT)): digest(OUT/f'{name}.{ext}')
        for name in ['participation', 'recovery', 'spatial_predictions', 'schooling_diagnostics'] for ext in ['pdf', 'svg']}
    inputs = {**previous['inputs'], **INPUTS} if previous else INPUTS
    preserved_values = {k: v for k, v in previous['plotted_values'].items()
                        if not k.startswith(selected+'_')} if previous else {}
    values = {**preserved_values, **VALUES}
    manifest.write_text(json.dumps(dict(inputs=inputs, outputs=outputs, plotted_values=values,
        author_requested_omissions={'participation': ['within-event timing spread (former panel e)'],
                                    'recovery': ['Given rest adjustment in panel c']},
        figure_definitions={'participation_d': 'Reserve <0.75 or >=0.75 at the accepted breathing decision; all retained breaths, equal simulation weights.'},
        simulations_run=0, note='All panels use saved summaries; Figure 4d now shows accepted-decision reserve classes.'), indent=2)+'\n')
    print(f'Rendered {selected} figure only; no simulations.' if selected else
          'Rendered four figures from saved summaries; no simulations.')


if __name__ == '__main__':
    main()
