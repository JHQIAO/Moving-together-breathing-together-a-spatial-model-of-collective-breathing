# W5b — Standalone RZ75 consolidation

Completed. The NumPy-only standalone module reproduces the frozen Sim25 implementation byte-for-byte in all 16 full-duration test pairs: 368 returned-array comparisons, comprising 140,258,908 reference-array bytes checked before trajectory downcasting. The full type-by-seed solo grid is included. No model parameters were recalibrated or scientifically retuned.

Deliverables: [`rz75_model.py`](rz75_model.py), [`test_equivalence.py`](test_equivalence.py) plus [`extra_solo.py`](extra_solo.py), [`README.md`](README.md), [`analyse.py`](analyse.py), and two PNG/PDF draft QA figures. All work and outputs are confined to this workstream. No commits, resets, stashes, shared-source edits or paper-build writers were used.

## 1. What was run

Every pair runs both implementations, sequentially, at a 1 s output cadence with a 300 s analysis burn-in. At most one simulation worker ran at a time; OPENBLAS_NUM_THREADS and OMP_NUM_THREADS were both 1. The default and every variant were checked against the saved full parameter dictionary before simulation. Rates remained the frozen RZ75 values in all arms.

| Arm | Changed fields from RZ75 | Seed | Length (s), per implementation | Status |
|---|---|---:|---:|---|
| rz75_7301 | none | 7301 | 1800 | byte-identical |
| rz75_9001 | none | 9001 | 600 | byte-identical |
| solo_fast | N=1, type_ratio=[1.0, 0.0, 0.0] | 8501 | 3600 | byte-identical |
| solo_medium | N=1, type_ratio=[0.0, 1.0, 0.0] | 8502 | 3600 | byte-identical |
| solo_slow | N=1, type_ratio=[0.0, 0.0, 1.0] | 8503 | 3600 | byte-identical |
| no_rest | kappa_bottom=0.0 | 9001 | 600 | byte-identical |
| need_off | upward_strength=0.0, need_down=1e-12 | 9001 | 600 | byte-identical |
| no_dilution | dilution=False | 9001 | 600 | byte-identical |
| half_dt | dt=0.0125 | 9001 | 600 | byte-identical |
| depth5 | box=[10.0, 3.3, 5.0] | 9001 | 600 | byte-identical |
| solo_fast_s8502 | N=1, type_ratio=[1.0, 0.0, 0.0] | 8502 | 3600 | byte-identical |
| solo_fast_s8503 | N=1, type_ratio=[1.0, 0.0, 0.0] | 8503 | 3600 | byte-identical |
| solo_medium_s8501 | N=1, type_ratio=[0.0, 1.0, 0.0] | 8501 | 3600 | byte-identical |
| solo_medium_s8503 | N=1, type_ratio=[0.0, 1.0, 0.0] | 8503 | 3600 | byte-identical |
| solo_slow_s8501 | N=1, type_ratio=[0.0, 0.0, 1.0] | 8501 | 3600 | byte-identical |
| solo_slow_s8502 | N=1, type_ratio=[0.0, 0.0, 1.0] | 8502 | 3600 | byte-identical |

The initial suite assigns one solo seed to each type; the supplemental suite completes every type at each of seeds 8501–8503. The no-rest, need-off and no-dilution lengths were unspecified in the request and were set to 600 s, paired on seed 9001. “Need off” follows the legacy convention upward_strength=0 and need_down=1e-12 because Sim25 requires need_down>0; it is explicitly a near-zero control. The standalone API also accepts both strengths exactly zero. Short smoke tests preceded the full tests and are saved separately; they are not substituted for requested durations.

**What it shows:** the simplified implementation preserves the frozen dynamics, decision draws and passive diagnostics across baseline, solo, ablation and numerical/geometric variants. **Paper value: strong** for code-release reproducibility; identity tests do not test biological adequacy.

## 2. Runtime

Run wall time includes simulator initialization and all returned logging arrays; comparison, compression and analysis are excluded. Ratios above one mean the standalone implementation was faster. Each case has one timed execution per implementation on the shared machine, so small differences are not reliable benchmark effects. Alternating execution order was used where seed parity differs; most listed seeds are odd, so most pairs ran Sim25 first.

| Arm | Sim25 wall s / simulated s | Standalone wall s / simulated s | Speed ratio (Sim25 / standalone) |
|---|---:|---:|---:|
| rz75_7301 | 0.21007 | 0.21836 | 0.962 |
| rz75_9001 | 0.21925 | 0.22140 | 0.990 |
| solo_fast | 0.02063 | 0.01898 | 1.087 |
| solo_medium | 0.01880 | 0.01678 | 1.120 |
| solo_slow | 0.01805 | 0.01734 | 1.041 |
| no_rest | 0.23081 | 0.21561 | 1.070 |
| need_off | 0.20066 | 0.19939 | 1.006 |
| no_dilution | 0.13984 | 0.13984 | 1.000 |
| half_dt | 0.24852 | 0.24191 | 1.027 |
| depth5 | 0.22051 | 0.22126 | 0.997 |
| solo_fast_s8502 | 0.01953 | 0.01884 | 1.037 |
| solo_fast_s8503 | 0.01867 | 0.01715 | 1.088 |
| solo_medium_s8501 | 0.01829 | 0.01785 | 1.024 |
| solo_medium_s8503 | 0.01845 | 0.01722 | 1.071 |
| solo_slow_s8501 | 0.01896 | 0.01786 | 1.062 |
| solo_slow_s8502 | 0.02018 | 0.01898 | 1.063 |

Total simulation wall time: Sim25 1751.50 s; standalone 1716.31 s; workload-weighted ratio 1.021. Per-case ratio mean 1.040 [0.962, 1.120]. Slower measured standalone cases: rz75_7301, rz75_9001, depth5. No numerical shortcut or altered random stream was used.

**What it shows:** consolidation removes dispatch and redundant inactive-branch calculations while retaining output compatibility. The table gives actual measured performance, including any regressions. **Paper value: supporting** as a computational-cost record; weak evidence for a precise speedup because timings share a contended machine and are not repeated benchmarks.

## 3. Canonical-observable reproduction gate

Before analysing new outcomes, `analyse.py --reproduce-only` reloaded the three saved Step25 A_RZ75 trajectories, verified their recorded SHA-256 hashes, reconstructed sorted-release collective bouts, and compared each statistic with each saved record. The maximum discrepancy was below the script's 1e-12 tolerance. This reproduces the programme's rounded targets, without rerunning or editing those saved results.

Equal-seed mean [minimum, maximum] below uses the three original training seeds. The record column uses the unchanged workbook. Its 95% percentile intervals resample the six entire recording segments, preserving within-segment order and never making cross-segment gap pairs. These are descriptive intervals from 2000 bootstrap draws, not an assertion that six segments are independent biological replicates. No frame-rate rescaling is applied.

| Statistic | Saved RZ75, equal-seed mean [min, max] | Recorded value [95% segment-bootstrap CI] |
|---|---:|---:|
| Collective gap mean (s) | 14.647 [14.523, 14.725] | 15.339 [14.002, 17.209] |
| Collective gap sample SD (s) | 6.056 [4.831, 6.681] | 6.499 [5.020, 7.660] |
| Gaps <5 s (%) | 7.276 [2.597, 10.256] | 3.105 [1.844, 4.239] |
| Gaps >30 s (%) | 2.137 [0.000, 5.128] | 2.451 [0.411, 4.965] |
| Participation <20% (%) | 12.712 [10.127, 15.190] | 10.356 [7.193, 14.260] |
| Participation ≥80% (%) | 0.849 [0.000, 1.282] | 3.722 [0.872, 8.720] |
| Front-slab duration (s) | 2.323 [2.225, 2.385] | 0.830 [0.764, 0.896] (front-glass duration) |
| Whole-event duration (s) | 4.355 [4.252, 4.459] | not directly comparable |
| Per-fish interval (s) | 48.854 [48.271, 49.402] | not directly comparable |
| Class versus next-gap Spearman | 0.172 [0.117, 0.258] | 0.160 [0.049, 0.257] |
| Class versus previous-gap Spearman | 0.074 [-0.036, 0.182] | 0.093 [-0.009, 0.212] |

**What it shows:** the endpoint and class definitions reproduce the historical table, including the class-to-next-gap association. Matching the metric pipeline is distinct from improving agreement with the record; the slab duration discrepancy remains visible. **Paper value: strong** as an analysis-consistency gate, supporting only for biological interpretation because these are reused selection-era runs.

## 4. Outcomes of the equality runs

These are implementation tests, not a new selection experiment. Solo rows aggregate three seeds equally. Each shoal row contains one seed, so its [min,max] is necessarily degenerate; the two baseline lengths are kept separate. Undefined correlations remain undefined for constant solo participation. No bouts were removed as outliers. The primary event window is start ≥300 s and end ≤T−2 s; individual gaps start from previous exits ≥300 s. Counts and full per-run metrics are in `outputs/analysis.json`.

| Arm | Seeds | Collective mean gap (s), mean [min,max] | Front slab (s), mean [min,max] | Individual gap (s), mean [min,max] |
|---|---:|---:|---:|---:|
| rz75_7301 | 1 | 14.725 [14.725, 14.725] | 2.225 [2.225, 2.225] | 49.402 [49.402, 49.402] |
| rz75_9001 | 1 | 13.940 [13.940, 13.940] | 1.858 [1.858, 1.858] | 47.405 [47.405, 47.405] |
| solo_fast | 3 | 58.041 [56.810, 59.708] | 0.668 [0.664, 0.674] | 58.041 [56.810, 59.708] |
| solo_medium | 3 | 85.051 [83.522, 87.380] | 0.676 [0.659, 0.691] | 85.051 [83.522, 87.380] |
| solo_slow | 3 | 130.530 [129.022, 132.039] | 0.659 [0.651, 0.668] | 129.817 [128.391, 132.039] |
| no_rest | 1 | 9.158 [9.158, 9.158] | 1.839 [1.839, 1.839] | 31.356 [31.356, 31.356] |
| need_off | 1 | 84.366 [84.366, 84.366] | 1.036 [1.036, 1.036] | 103.708 [103.708, 103.708] |
| no_dilution | 1 | 10.760 [10.760, 10.760] | 2.905 [2.905, 2.905] | 65.147 [65.147, 65.147] |
| half_dt | 1 | 14.228 [14.228, 14.228] | 2.508 [2.508, 2.508] | 47.477 [47.477, 47.477] |
| depth5 | 1 | 15.125 [15.125, 15.125] | 2.643 [2.643, 2.643] | 43.791 [43.791, 43.791] |

**What it shows:** identical arrays imply identical observables for both implementations, including the ablations and solo types. The shorter shoal records have fewer post-burn events and are not substitutes for the reference ensemble. **Paper value: supporting** for implementation coverage; **weak** as standalone mechanism or calibration evidence because the workstream was designed for equality and uses limited seeds per shoal arm.

## 5. Code, equation and provenance audit

The delivered module has one simulation class, 40 active/derived dataclass fields, and no project or SciPy imports. README maps all 26 labelled equations and all behavioural states to source lines. It separately accounts for 28 removed/fixed-route legacy fields. Parameter values come from the saved RZ75 record. Rates are classified as isolated-fish calibrated; rest-zone settings, cruise speed, need set point/downward strength and the evaluated dilution branch are classified as shoal-tuned choices, with assumed formula constants disclosed. Other inherited choices are not promoted to measurements without source evidence.

No substantive active dynamical equation/state mismatch was found. The text describes the normalised noisy target as cached, whereas the engine caches it before boundary normalisation; the steering direction is equivalent. Numerical tolerances, a bounded number of predictive boundary passes and exact operation order add implementation details beyond the equations. The text's event equations do not specify the analysis censoring/burn window. README records these points, the isolated-calibration selection history, the failed historical need-off guard and the absence of measured cruise-speed validation. No TeX was edited.

**What it shows:** the model can be released without the exploratory inheritance chain, and the parameter table distinguishes data information from model choices. **Paper value: strong** for Methods/code availability and accurate provenance; the audit does not confer independent validation on tuned settings.

## 6. QA and failures

- All full-duration equality comparisons passed for all returned keys, shapes, dtypes and raw bytes. Core trajectories, episodes13, events and decision_trials plus entries21, band22 and all inherited counters/scalars were checked.
- The fresh 1800 s seed-7301 replay also matches every historical Step25 returned-array full-precision fingerprint (`verify_history.py`). Historical float64 trajectory files and the required new float32 files have intentionally different raw storage bytes; conversion and full-precision fingerprints are checked separately.
- Comparisons occur before float32 trajectory storage; episode/decision/event tables remain full precision. Per-array reference and standalone hashes and versioned source pins are in each run JSON. Saved NPZ hashes were reverified during analysis.
- A short development smoke check initially exposed a missing empty `social_trials` compatibility field; it was fixed before the recorded smoke/full suites. An exploratory comparison initially used unequal default sample cadences; all recorded tests explicitly use identical sample_dt=1. No full-duration failing case was discarded or hidden.
- Figure source preflight initially requested an additional SVG export; SVG was added alongside the required PNG/PDF. Its remaining draft-export warnings concern 300 dpi PNG rather than 600 dpi and no TIFF; PNG/PDF were explicitly requested, so these are documented rather than represented as failed numerical checks.
- Standalone import isolation, returned-array partition, source dependency checks and final figure audits are recorded in `outputs/final_qa.json`. Figure font/collision audits passed; single-panel alignment is explicitly not applicable. Visual review is recorded separately.
- Generic project instructions mention shared engine/paper build commands. They were not run here because this task expressly prohibits writers outside W5b and does not alter those shared files; the requested scoped equality and metric-reproduction gates are complete.
- The first delivery-audit attempt occurred while Matplotlib was building its local font cache and stopped because the figure directory had not yet been written. After rendering completed, the entire audit was rerun successfully.
- Process inspection via `ps` was denied by the environment. Progress and completion were verified from worker logs and completed hash-checked records; no other session's process was inspected or stopped.
- No byte identity is claimed across different NumPy/BLAS/platform builds. Timings include shared-machine variability. The historic near-zero need-off convention and calibration caveats are reported above.

## 7. Draft figures

- [`equality_coverage.png`](figures/equality_coverage.png) / [`PDF`](figures/equality_coverage.pdf): blue cells indicate byte identity for every array in each exhaustive output group across all full-duration pairs; deterministic checks have no sampling error bars.
- [`runtime_ratio.png`](figures/runtime_ratio.png) / [`PDF`](figures/runtime_ratio.pdf): reference/standalone wall-time ratios for each complete test pair, with unity marking equal measured time; one timing per implementation per case.

Figures are draft implementation QA, with plain labels and no biological-distribution claims. Their source CSVs, alignment records, font/collision audits and PNG previews are in `figures/`. `FIGURE_CONTRACT.md` states the evidence and rendering choices.

## Reproduction and inventory

Follow the sequential commands in README. `consolidate.py` regenerates the standalone source from the preserved active routines; its resulting module is independently importable. `analyse.py`, `make_figures.py`, `document.py` and `report.py` regenerate all reported tables, figures and text from saved evidence. Every numerical result in this report is script-derived. `outputs/delivery_manifest.json` pins the final deliverables. `run.log` and `finish.log` preserve completion records.
