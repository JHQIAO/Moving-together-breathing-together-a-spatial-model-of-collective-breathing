# Data: "Moving together, breathing together: a spatial model of collective breathing"

This repository holds the data behind every figure and table of the article. It contains data
only; it holds no program code. Lengths are in body lengths (BL) and times in seconds.

## Contents

| Folder | What it holds |
|---|---|
| `figures/` | One folder per figure: the figure as it appears in the article (PDF and editable SVG) and the values it shows, in `source_data/`. |
| `tables/` | The values behind Table 1 and the supplementary tables. |
| `data/reference_simulations/` | The ten reference simulations: run records, per-run statistics and event-level results. |
| `data/other_simulations/` | Run definitions and summaries of the other simulation sets (component removal, depth, group size, isolated fish, zone radii). |
| `data/empirical/` | The recorded data of Bartashevich et al. (2026), used for every comparison with the recording. |
| `COPIED_FILES.csv` | Every file with its size and SHA-256. |

File formats: CSV and JSON are plain text. The `.npz` files are NumPy archives and open with
`numpy.load`.

## Figures and their source data

Each `figures/<figure>/source_data/` folder holds the values of that figure.

| Figure | Folder | Source data |
|---|---|---|
| 1 Model schematic | `fig1_model_schematic` | none (drawing) |
| 2 Intervals and participation | `fig2_intervals_and_participation` | `plotted_values.json`: all plotted values. `histogram_source.csv`: interval histograms in 2-s bins for the recording, each of the ten reference runs, and the published model of Bartashevich et al. (peak-to-peak intervals; not plotted). `record_uncertainty.json`: statistics of the recording with confidence intervals. `ensemble.json`: statistics across the ten runs. `g1_shoal.json`: interval-distribution fits, including the recorded Laplace fit. |
| 3 Rest–rise cycle | `fig3_rest_rise_cycle` | `example_individual_traces.csv`: panels a,b (reference run 9001, 300–420 s). `aligned_per_run.csv` and `aligned_ensemble.csv`: panel c, signals aligned to lift-off (D oxygen drive, B active breaths, H median height). `peak_times.csv`: panel d, one row per lift-off window (5,733 windows). `window_counts.csv`, `summary.json`: counts and lag values. |
| 4 Component removal | `fig4_component_removal` | `per_run_metrics.csv`: one row per run (five conditions × ten runs). `plotted_summary.csv`: plotted means and SDs. `summary.csv`, `summary.json`, `table_S6.csv`, `caption_counts.json`: condition summaries. `record_uncertainty.json`: recorded values and confidence intervals. |
| 5 Recruitment and replenishment | `fig5_recruitment_and_replenishment` | `plotted_values.json`: all plotted values, by panel (`a`–`f`). `a_counts.csv`, `b_c_d_run_means.csv`, `c_isolated_runs.csv`, `d_event_rest.csv`, `e_drive_per_run.csv`: run-level and event-level values behind panels a–e. |
| 6 Depth and group size | `fig6_depth_and_group_size` | `plotted_values.json`: all plotted values. `per_run_metrics.csv`: depth runs (column `depth`). `groups.json`: group-size runs in the fixed tank (`same_N…`) and at fixed density (`density_N…`). |
| S1 Schooling regime | `figS1_schooling_regime` | `movement_summary.csv`: panels a,b. `breathing_summary.csv`: panel c. `schooling_runs.json`, `attraction_radius_runs.csv`: the runs behind them. |
| S2 Vertical sorting | `figS2_vertical_sorting` | `sorting.json`: panel a, with the shuffled null. `sorting_per_run.csv`, `sorting_per_run_curves/`: per-run values. `type_shares_source.json`: panel b (key `supplementary_type_shares`). |

Condition codes in the `arm` column of `per_run_metrics.csv`: `A0` full model, `A1` no rest zone,
`A2` no oxygen drive, `A6` no surface avoidance, `A8` no orientation or attraction.

## Tables

| Table | Source |
|---|---|
| Table 1, recorded and simulated statistics | `tables/table1_record_vs_model.csv`; best-fitting family from `tables/tableS_interval_distribution_fits.csv` |
| Supplementary parameters | `tables/reference_parameters.json` |
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

- The program code of the model and of the analyses.
- The full simulation output (positions, headings and reserves of every fish each second, and the
  tables of all breaths and decisions). It is about 100 MB per reference run and several GB in total.

## Notes

- `tables/reference_parameters.json` is the parameter record of the simulations. It also lists switches
  of mechanisms that were tested and removed during development; they are all off.
- Some summary files hold more than the article shows (further model variants in `per_run_metrics*.csv`
  and `supplementary_table_values.json`, composition variants in `groups.json`).
- Run records and protocols name the program files that produced each run, with their checksums.
  Those files are not part of this repository.

## Recorded data

`data/empirical/Supplementary_Data_1.xlsx` is Supplementary Data 1 of

> Bartashevich, P., Francisco, F. A., Escurra-Alegre, A., Schäfer, F., Wuertz, S., Krause, J., Kloas, W.
> & Bierbach, D. (2026). Air-breathing synchrony in juvenile *Arapaima gigas* reveals collective
> coordination under individual physiological constraints. *Communications Biology* 9, 831.
> https://doi.org/10.1038/s42003-026-10472-w

It is reproduced here unchanged under the article's CC BY 4.0 licence. Please cite that article when
using the recorded data.
