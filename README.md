# Code and data: "Moving together, breathing together: a spatial model of collective breathing"

This folder holds the simulation model, the data behind every figure and table, and the
scripts that draw the figures. Lengths are in body lengths (BL) and times in seconds.

## Contents

| Folder | What it holds |
|---|---|
| `model/` | The model (`rz75_model.py`, NumPy only), its reference parameters, the event definitions, and a short example run. |
| `figures/` | One folder per figure: the figure as it appears in the article (PDF and editable SVG), a script that redraws it, and its `source_data/`. |
| `tables/` | The values behind Table 1 and the supplementary tables. |
| `data/empirical/` | The recorded data of Bartashevich et al. (2026), used for every comparison with the recording. |
| `data/reference_simulations/` | The ten reference simulations: run records, per-run statistics and event-level results. |
| `data/other_simulations/` | Run definitions and summaries of the other simulation sets (component removal, depth, group size, isolated fish, zone radii). |
| `analysis_scripts_as_run/` | The scripts that produced the simulations and summaries, exactly as they were run. Kept as a record; see the note at the end. |
| `COPIED_FILES.csv` | Every file with its size, SHA-256 and where it came from in the working project. |

## Requirements

Python 3.11 with NumPy, SciPy, pandas and Matplotlib (`requirements.txt` lists the versions used).
The model itself needs only NumPy. The figures use Arial or Helvetica when installed and
otherwise fall back to the Matplotlib default font.

## Running the model

```sh
cd model
python run_example.py            # 900 s, seed 9001; takes a minute or two
python run_example.py 13440 9001 # one full reference run (roughly half an hour)
```

In Python:

```python
from rz75_model import RZ75Sim, parameters
from collective_events import collective_statistics

p = parameters()                               # reference parameters of the article
sim = RZ75Sim(p, seed=9001, audit_burn=300.)
data = sim.run(13440., sample_dt=1.)           # positions, headings, reserves, states, breaths
stats = collective_statistics(data['episodes13'], p.N, 13440., burn=300.)
```

`parameters(name=value, ...)` changes parameters, for example `parameters(kappa_bottom=0.)`
removes the rest zone and `parameters(N=50)` changes the group size. `model/README.md` maps each
equation of the article to the code and describes the random streams and the order of operations.
`collective_statistics` reproduces the saved statistics of the reference runs exactly from their
breath tables.

## Redrawing the figures

Each script reads only the `source_data/` beside it and runs in a few seconds. No simulation is run.

```sh
python figures/fig2_intervals_and_participation/plot_fig2.py           # writes into the figure folder
python figures/fig2_intervals_and_participation/plot_fig2.py some/dir  # or into another folder
```

| Figure | Folder | Script | Simulations behind it |
|---|---|---|---|
| 1 Model schematic | `fig1_model_schematic` | `draw_fig1.py` | none (drawing) |
| 2 Intervals and participation | `fig2_intervals_and_participation` | `plot_fig2.py` | ten reference runs; recording |
| 3 Rest–rise cycle | `fig3_rest_rise_cycle` | `plot_fig3.py` | a,b: reference run 9001; c,d: ten reference runs, 5,733 lift-off windows |
| 4 Component removal | `fig4_component_removal` | `plot_fig4.py` | five conditions × ten runs |
| 5 Recruitment and replenishment | `fig5_recruitment_and_replenishment` | `plot_fig5.py` | ten reference runs; isolated fish; rest-zone variants |
| 6 Depth and group size | `fig6_depth_and_group_size` | `plot_fig6.py` | depth runs and group-size runs |
| S1 Schooling regime | `figS1_schooling_regime` | `plot_figS1.py` | zone-width grid without breathing; zone radii ±20 % |
| S2 Vertical sorting | `figS2_vertical_sorting` | `plot_figS2.py` | ten reference runs |

On the computer used for the article, all eight scripts reproduce the published PDFs pixel for
pixel. Another Matplotlib version or another font can shift text by small amounts.

## Tables

| Table | Source |
|---|---|
| Table 1, recorded and simulated statistics | `tables/table1_record_vs_model.csv`; best-fitting family from `tables/tableS_interval_distribution_fits.csv` |
| Supplementary parameters | `model/reference_parameters.json` |
| Supplementary isolated-fish targets and rates | `tables/supplementary_table_values.json` (key `isolated`); intervals in `tables/tableS_isolated_fish_intervals.json` |
| Supplementary interval-distribution fits | `tables/tableS_interval_distribution_fits.csv` |
| Supplementary component removal | `tables/tableS_component_removal.csv` |
| Supplementary rest-zone variants | `tables/supplementary_table_values.json` (key `variants`) |
| Supplementary depth and group-size sweeps | `tables/supplementary_table_values.json` (keys `depth`, `groups`) |

## Simulation sets

All runs use 200 fish in a 10 × 3.3 × 3.3 BL tank unless stated, a time step of 0.025 s, and
discard the first 300 s.

| Set | Runs | Where |
|---|---|---|
| Reference | seeds 9001–9010, 13,440 s each | `data/reference_simulations/` |
| Component removal (Fig. 4) | five conditions × seeds 9001–9010; 7,500 s (no oxygen drive: 3,600 s). The reference runs are analysed over their first 7,500 s. | `figures/fig4_component_removal/source_data/`, `data/other_simulations/component_removal_and_depth/` |
| Tank depth (Fig. 6a,b) | depths 2.2, 3.3, 5 and 6.6 BL, with and without the rest zone; seeds 9001–9003, 7,500 s | `data/other_simulations/component_removal_and_depth/` |
| Group size (Fig. 6c,d) | N = 2–400 in the fixed tank and N = 50–400 at fixed density; seeds 9201–9203 (9201–9205 for N ≤ 50 in the fixed tank), 7,500 s | `data/other_simulations/group_size/` |
| Isolated fish (Fig. 5c) | one fish per run, three breathing types × seeds 9601–9610, 7,200 s | `data/other_simulations/isolated_fish/` |
| Weaker rest zone (Fig. 5f) and attraction radius ±20 % (Fig. S1c) | seeds 9301–9303 | `data/other_simulations/parameter_perturbations/` |
| Schooling regime (Fig. S1) | 27 runs without breathing over a grid of zone widths; 6 breathing runs with the orientation radius ±20 %; seeds 9301–9303 | `data/other_simulations/schooling_regime/` |

In `data/reference_simulations/`:

- `run_records/` — seed, duration, all parameters and software versions of each run.
- `event_level/reference_s9001_derived.npz` … — per run: collective intervals (`gaps`), participation
  (`parts`, `classes`), event start and end times, event durations, and individual inter-breath
  intervals (all fish, and by breathing type).
- `per_run_statistics/`, `model_replicates.csv`, `model_summary.csv`, `ensemble.json` — statistics of each
  run and across the ten runs.
- `record_uncertainty.json` — the same statistics for the recording, with bootstrap confidence intervals.

## What is not included

The full simulation output (positions, headings and reserves of every fish each second, and the
tables of all breaths and decisions) is not included. It is about 100 MB per reference run and several
GB in total. Each run can be regenerated from its seed and parameters in the run records.

## Notes on provenance

- The reference, component-removal, depth, group-size and isolated-fish runs were made with the
  development version of the model, a chain of Python classes built up step by step.
  `model/rz75_model.py` is that model written as one file. It gave byte-identical output to the
  development version in all 16 test runs, which cover the reference shoal, isolated fish and
  several parameter changes (`analysis_scripts_as_run/model_consolidation/REPORT.md`).
  The schooling-regime runs were made with `rz75_model.py` directly.
- `model/reference_parameters.json` is the parameter record of the development version. It also lists
  switches of mechanisms that were tested and removed; they are all off and do not exist in `rz75_model.py`.
- The files in `analysis_scripts_as_run/` are unchanged copies. They read the full simulation output and
  contain file paths of the computer they ran on, so they do not run from this folder. They document how
  each summary was computed. The scripts in `figures/` and `model/` are the ones meant to be run.
- Some summary files hold more than the article shows (further model variants in `per_run_metrics*.csv`
  and `supplementary_table_values.json`, composition variants in `groups.json`).
- Two files are extracts rather than copies: `figures/figS1_schooling_regime/source_data/attraction_radius_runs.csv`
  and `data/other_simulations/parameter_perturbations/perturbation_runs_used.csv` contain only the rows of
  the perturbation table that the article uses. `COPIED_FILES.csv` marks them.

## Recorded data

`data/empirical/Supplementary_Data_1.xlsx` is Supplementary Data 1 of

> Bartashevich, P., Francisco, F. A., Escurra-Alegre, A., Schäfer, F., Wuertz, S., Krause, J., Kloas, W.
> & Bierbach, D. (2026). Air-breathing synchrony in juvenile *Arapaima gigas* reveals collective
> coordination under individual physiological constraints. *Communications Biology* 9, 831.
> https://doi.org/10.1038/s42003-026-10472-w

It is reproduced here unchanged under the article's CC BY 4.0 licence. Please cite that article when
using the recorded data.
