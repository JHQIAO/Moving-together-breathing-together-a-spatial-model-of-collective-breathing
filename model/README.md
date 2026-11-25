# Standalone RZ75 model

`rz75_model.py` is a single simulation class with NumPy as its only third-party dependency. It has no project imports, path manipulation, data-file dependency, SciPy dependency or subclass chain. Python standard-library dataclasses support immutable parameters. Default parameters are the adopted frozen RZ75 configuration, not the earlier exploratory defaults of `model25.parameters()`.

```python
from rz75_model import RZ75Sim, parameters
sim = RZ75Sim(parameters(), seed=9001, audit_burn=300.)
data = sim.run(600., sample_dt=1.)
```

Run this example only when your CPU allocation permits it. From this directory, reproduce the suite using:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python test_equivalence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python extra_solo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python verify_history.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python analyse.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python make_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python document.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python audit_delivery.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python report.py
```

The suite is strictly sequential and resumes only records with matching parameters, seed, duration, source versions and NPZ SHA-256. The two implementations run in the same process one after the other. Full float64 returned arrays are compared by dtype, shape and bytes, including NaN bit patterns, before storing float32 trajectories. Only one trajectory archive is saved per pair: there is no duplicate reference archive. Full-precision event, breath, decision and entry tables plus all extra outputs are retained. Every array has separate reference and standalone SHA-256 fingerprints. `events` contains individual state transitions; collective events are derived offline from `episodes13`.

The release supports rest strength/height, separate need strengths/set point, dilution on/off, surface strength, depletion rates, type ratios, population size, box, integration/reaction intervals, cruise/ascent/descent speeds, and the inherited active movement and decision parameters. `parameters(speed=...)` also sets descent speed unless explicitly supplied. Both need strengths can be exactly zero in this module; the legacy equivalence arm uses upward=0 and downward=1e-12 because the parent rejects an exact zero downward strength. This follows the Step 25 near-zero control convention, without changing its rates to the old C24 rates.

Inactive recruitment, partial refill, following, one-entry decisions, attraction quotas, alternative hazards and social decision weighting are absent. Empty `social_trials` remains solely as an output schema compatibility field; `band22` remains a passive all-state neighbour composition log. There is no option to turn these removed mechanisms on. `Sim25` is only a convenience alias to the standalone class.

## Numerical contract and random streams

Movement starts with `default_rng(seed)`, uniform positions, then normalised Gaussian headings; the same stream subsequently draws heading noise for every fish on reaction ticks, including motor-state fish. Initial reserves use `SeedSequence([seed,417])` and uniform [0.5,1]. Types use `[seed,618]`, largest-remainder allocation with stable fast/medium/slow ties, then a permutation. Decisions use `[seed,829]` and consume N draws on every integration step even if no fish is eligible. Removed cue/follow streams never influenced these streams.

The integration order is proximity decisions, state snapshot, reaction target/boundary steering, motor overrides, position and constraints, depletion, contact refill, dwell/turn/return transitions, then passive release and physical-descent logging. The target is cached between reaction ticks. Preserve NumPy operation order: algebraically equivalent rearrangements and different BLAS/NumPy implementations need not remain byte-identical over long chaotic trajectories. Equality is established in the recorded environment; cross-platform identity is not claimed. The public `run` API, like the original, advances the simulation for the requested number of seconds per call. Pass the duration and sample cadence explicitly: the standalone core retains TypedSim.run defaults (600 s, 0.1 s), whereas the old outer Sim25 wrappers default to 1800 s and 1 s. All programme examples and equality tests explicitly use the required 1 s cadence. This API-default difference does not change any tested dynamics or output array.

## Equation and state map

Line links below refer to the delivered source. Collective-event equations are observation rules rather than engine dynamics and are implemented in the analysis script.

| TeX equation label | Implementation | Numerical detail |
|---|---|---|
| `eq:visibility` | [`social_direction`](rz75_model.py#L301) | Heading-centred FOV; implementation includes a floating-point tolerance. |
| `eq:neighbour-sets` | [`social_direction`](rz75_model.py#L301) | Metric repulsion is uncapped; nearest visible selection precedes zone filtering. |
| `eq:steering` | [`social_direction`](rz75_model.py#L301) | Normalise repulsion, orientation and attraction sums, then the latter combined direction. |
| `eq:need` | [`reaction_target`](rz75_model.py#L435) | Separate upward/downward strengths around the reserve set point. |
| `eq:density-factor` | [`reaction_target`](rz75_model.py#L435) | Visible count from nearby_count, capped at vision_k. |
| `eq:surface` | [`reaction_target`](rz75_model.py#L435) | Clipped local surface layer multiplied by density factor. |
| `eq:rest-strength` | [`reaction_target`](rz75_model.py#L435) | Bottom damping varies linearly with height; zero-kappa branch retains parent arithmetic. |
| `eq:rest-zone` | [`reaction_target`](rz75_model.py#L435) | Subtract kappa times current vertical heading. |
| `eq:vertical-total` | [`reaction_target`](rz75_model.py#L435) | Add need and surface terms before subtracting bottom damping. |
| `eq:combined` | [`reaction_target`](rz75_model.py#L435) | Add vertical force only outside repulsion, then normalise. |
| `eq:noise` | [`reaction_target`](rz75_model.py#L435) | One full Cartesian draw per reaction; scale world z afterward. |
| `eq:noise-scale` | [`reaction_target`](rz75_model.py#L435) | Reference noise time is 0.1 s. |
| `eq:position` | [`_changed_step`](rz75_model.py#L505) | Updated heading, motor-specific speed, stationary states, periodic wrap. |
| `eq:boundary-proximity` | [`steer_boundary`](rz75_model.py#L354) | Distance to target-facing vertical boundary. |
| `eq:boundary-weight` | [`steer_boundary`](rz75_model.py#L354) | Smoothstep blend. |
| `eq:boundary-blend` | [`steer_boundary`](rz75_model.py#L354) | Horizontal projection uses tangent fallback. |
| `eq:boundary-clearance` | [`steer_boundary`](rz75_model.py#L354) | Predictive clearance revisited at most three times. |
| `eq:states` | [`_changed_step`](rz75_model.py#L505) | State transitions are logged by transition; physical release/exit by step. |
| `eq:reserve` | [`_changed_step`](rz75_model.py#L505) | Bounded per-step consumption; no drain during accumulation. |
| `eq:fermi` | [`scaled_need`](rz75_model.py#L143) | Logistic scaled reserve; hazard computed directly using logaddexp. |
| `eq:decision-probability` | [`decision_probability`](rz75_model.py#L152) | Stable -expm1 evaluation; N paired random draws every integration step. |
| `eq:refill` | [`_changed_step`](rz75_model.py#L505) | Contact refill after consumption; preserve subtract/add operation order. |
| `eq:event-times` | [`canonical`](analyse.py) | analyse.py:canonical calls breathing_study.bout_events_from on sorted releases. |
| `eq:event-duration` | [`canonical`](analyse.py) | analyse.py:canonical subtracts event starts from ends. |
| `eq:collective-interval` | [`canonical`](analyse.py) | analyse.py:canonical computes starts[1:] - ends[:-1]. |
| `eq:participation` | [`canonical`](analyse.py) | analyse.py:canonical divides distinct participants by actual N, including solo tests. |

| Code state | TeX state | Motion and exit condition |
|---|---|---|
| 0 | stay | Voluntary social/vertical target; Fermi acceptance in decision band enters leave. |
| 1 | leave | Turn upward and move at ascent_speed; surface contact refills and enters accumulation. |
| 2 | accumulation | Stationary, no depletion; dwell expiry enters turn. |
| 3 | turn | Stationary; rotate downward at return_turning_rate until heading_z < -0.05. |
| 4 | back | Move at descent_speed, ordinary turn rate; depth >= return_depth returns to stay. |

State code: [`_changed_step`](rz75_model.py#L505); air-release interpolation and first downward displacement: [`step`](rz75_model.py#L571); wrapping/clipping: [`constrain`](rz75_model.py#L400); robust Rodrigues turn cap: [`_cap_turn`](rz75_model.py#L247).

## Parameter provenance

Values below are generated from the frozen saved record and the standalone dataclass. “Tuned” includes declared model choices evaluated against shoal behaviour; it does not imply a statistically identified estimate or an automated optimiser. Mixed provenance is stated explicitly. The isolated-fish calibration targets come from recordings, but the fitted depletion rates are latent effective parameters. The real isolated-fish water depth is different from the model tank used for calibration, as flagged by COMMON.md; do not describe these as directly measured oxygen consumption rates.

| Active parameter | RZ75 value | Provenance class | Evidence / interpretation |
|---|---|---|---|
| `N` | `200` | measured (record context) | Recorded shoal population; fixed at 200 in programme conventions. |
| `box` | `(10.0, 3.3, 3.3)` | measured (record context) + modelling assumption | Record-scale BL tank dimensions; horizontal periodicity is assumed, not measured. |
| `dt` | `0.025` | assumed/numerical | Step21 fixed .025 s integration; half-step checks do not fit dt. |
| `speed` | `0.6` | tuned to shoal record | Step24 compared 1/.75/.60 BL/s and user retained .60; RESULTS explicitly says pixel-to-BL speed measurement was not performed. |
| `rr` | `0.15` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `ro` | `0.45` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `ra` | `2.95` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `vision_k` | `24` | assumed/inherited | k24 schooling fixed in Step21; no direct neighbour-count measurement or fresh estimate established by inspected steps21–25. |
| `fov_deg` | `335.0` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `turning_rate` | `3.141592653589793` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `eta` | `0.3` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `surface_strength` | `0.3` | assumed/inherited | Inherited .30 surface steering strength; not an empirical risk estimate. |
| `glide_layer` | `0.6` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `guard_margin` | `0.05` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `upward_strength` | `0.35` | assumed/inherited | Inherited reserve steering strength .35; retained through steps21–25, not estimated from direct force observations. |
| `consumption_rates` | `(0.00840887578389495, 0.005158809198668612, 0.003100327856481332)` | calibrated to isolated fish | Step25 frozen RZ75 solo calibration; same held-out seeds first flagged recalibration, so final old checks were not untouched validation. |
| `type_ratio` | `(1.0, 1.0, 1.0)` | assumed/inherited | Equal fast/medium/slow weights with fixed largest-remainder counts; not an estimated shoal type census. |
| `trigger_depth` | `0.6` | tuned to shoal record (inherited) | Inherited .6 BL decision band used in Step21 baseline. |
| `breath_time` | `0.15` | assumed/inherited motor setting | Step21 explicitly fixed .15 s; not equal to the recorded whole visible breath duration. |
| `refill_level` | `0.95` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `return_depth` | `0.7` | tuned to shoal record (inherited) | Inherited .7 BL return geometry; Step21 compared deeper alternatives. |
| `fermi_midpoint` | `0.8` | tuned to shoal record | Step11 confirm_q80_t0p3 record inherited through step13 and step21. |
| `fermi_temperature` | `0.1` | assumed/inherited | Inherited fixed movement/physiology specification; no direct estimation evidence in inspected steps21–25 PLAN/RESULTS. |
| `decision_timescale` | `0.3` | tuned to shoal record | Step11 confirm_q80_t0p3; physical hazard reference, not reaction or dwell time. |
| `ascent_speed` | `2.1` | assumed/inherited motor setting | Step13 MOTOR constant and Step21 fixed baseline; no calibrated video-speed evidence in the inspected provenance. |
| `return_turning_rate` | `7.853981633974483` | assumed/inherited motor setting | Step13 MOTOR constant 450 degrees/s; Step21 fixed baseline. |
| `release_depth` | `0.5` | assumed observation rule | Step13 release proxy .5 BL; changes measured breath timing, not motor dynamics. |
| `reaction_interval` | `0.05` | assumed/inherited biological clock | Step17/21 fixed .05 s target refresh; separated from integration and hazard clocks. |
| `need_setpoint` | `0.75` | tuned to shoal record | Steps21/23 used .70; Step25 compared .70/.75, adopted RZ75. |
| `surface_layer` | `1.5` | tuned to shoal record (mechanism choice) | Step21 proposed local 1.5 BL surface layer and compared with global push. |
| `dilution` | `True` | tuned to shoal record (mechanism choice) | Step21 compared diluted/localized controls; diluted retained as Step22 parent. |
| `dilution_radius` | `1.0` | tuned to shoal record (assumed formula) | Step21 declares 1 BL visible neighbourhood; explicitly not experimentally measured. |
| `dilution_min` | `0.2` | tuned to shoal record (assumed formula) | Step21 declared limiting factor .20; not a measured predation-risk function. |
| `dilution_scale` | `6.0` | tuned to shoal record (assumed formula) | Step21 declared count scale 6; not estimated from direct risk observations. |
| `vertical_noise_scale` | `0.2` | tuned to shoal record (mechanism choice) | Step22 compared world-z noise scales 1 and .2; reduced-noise branch inherited. |
| `need_down` | `0.1` | tuned to shoal record | Step23 asymmetric-need candidate .10 retained by Step24 request despite selection failure. |
| `descent_speed` | `0.6` | tuned to shoal record (tied) | Step24 ties descent to selected cruise speed. |
| `kappa_bottom` | `0.3` | tuned to shoal record | Step25 requested .30 after exploratory shoal tests; not a direct measurement. |
| `rest_height` | `1.0` | tuned to shoal record | Step25 requested 1 BL layer; tested alternatives descriptively. |
| `return_clearance` | `0.09999999999999998` | derived | return_depth - trigger_depth; subtraction preserved, including binary rounding. |

The following frozen-record fields are deliberately absent from the standalone parameter API. Listing them prevents inactive inherited settings being mistaken for fitted RZ75 parameters.

| Removed or fixed-route field | Frozen value | Provenance / treatment |
|---|---|---|
| `side_layer` | `1.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `vertical_layer` | `0.2` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `wall_strength` | `1.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `boundary_scope` | `vertical` | Assumed/inherited; fixed active route compiled into the implementation |
| `breathing_enabled` | `True` | Assumed/inherited; fixed active route compiled into the implementation |
| `decision_policy` | `fermi` | Assumed/inherited; fixed active route compiled into the implementation |
| `attraction_quota` | `0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `partial_amount` | `0.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `cue_mode` | `off` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `cue_k` | `50` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `join_midpoint` | `0.6` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `join_temperature` | `0.08` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `cue_reset_s` | `2.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `motor_social_weight` | `0.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `movement_cycle` | `True` | Assumed/inherited; fixed active route compiled into the implementation |
| `signed_need` | `True` | Assumed/inherited; fixed active route compiled into the implementation |
| `local_surface` | `True` | Assumed/inherited; fixed active route compiled into the implementation |
| `entry_decision` | `False` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `school_band_weight` | `0.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `lopsided_need` | `True` | Assumed/inherited; fixed active route compiled into the implementation |
| `follow_dasher` | `False` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_depth` | `1.5` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_radius` | `1.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_duration` | `0.5` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_weight` | `1.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_speed` | `2.0` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_stop_depth` | `0.2` | Assumed/inherited; inactive branch/unused inherited setting removed |
| `follow_refractory` | `2.0` | Assumed/inherited; inactive branch/unused inherited setting removed |

Additional fixed constants are part of the numerical/observation contract, not fitted parameters: reserve initial bound 0.5; noise reference time 0.1 s; downward heading threshold 0.05; burn-in 300 s; sample interval 1 s; merge separation 1 s; final-event guard 2 s; front slab width 0.5 BL; participation edges 0.2/0.5/0.8. The RNG stream offsets are reproducibility conventions. Geometric and angular tolerances are floating-point guards.

## TeX/code audit and interpretation

No substantive mismatch was found in the active RZ75 dynamical equations or state rules. The following qualifications make the prose precisely reproducible:

- The code caches the pre-normalised noisy target, then normalises it inside boundary steering at each integration step; the text describes the normalised target as cached. These are the same target direction in this implementation, but the stored array is not necessarily unit length.
- Visibility and geometry comparisons use tiny tolerances, the predictive boundary check makes at most three passes, and exact zero/antiparallel vectors have deterministic fallbacks. The text describes the principal rules rather than all floating-point safeguards.
- Depletion and refill use subtract/add arithmetic. The refill equation describes the mathematical value; preserving this arithmetic, rather than rewriting as a direct maximum assignment, is required for bitwise reproduction.
- Entry logs mark first eligible stay-state occupancy after geometric rearming; they are passive diagnostics, not a second breathing-decision mechanism. `events` are state transitions and must not be confused with collective bouts.
- The text states that rates were calibrated on isolated-fish means, which matches Step25. Its calibration history should nevertheless be disclosed: old final held-out seeds were also used to flag which types required recalibration. W5b equality tests do not establish calibration validity.
- The user adopted RZ75 for the programme; the historical Step25 selection protocol had stopped because its need-off guard failed. Adoption does not retroactively turn that selection outcome into a pass. The inspected Step24 record explicitly does not validate cruise speed from video.
- The text does not state the primary analysis burn-in and final-event censoring in its event equations. W5b uses the binding programme convention in `analyse.py`. The potential shoal frame-rate correction remains a separate scenario and is not applied here.

## Provenance inputs

The consolidation source selector is retained as `consolidate.py`; it copies active numerical routines and explicitly flattens initialization, transition dispatch and returned-array assembly. It is a build/audit aid; running the released module never reads the old model. Re-generating it intentionally rewrites only files in this workstream. The source hash inventory below is read-only and can detect later changes to the reference model, model text and tuning history.

- `study_out/local_work/rest_zone_step25/model_section/model_rz75.tex` — SHA-256 `1778a65021dc94cd497b8f32e187bc9f7734bf5fb878be137a097588668e11f7`
- `study_out/local_work/movement_cycle_step21/PLAN.md` — SHA-256 `96f9edd2ae787fce1ff7841e123a1da5535c69916a10a1a3751e098500e0d346`
- `study_out/local_work/movement_cycle_step21/RESULTS.txt` — SHA-256 `e0b5f6363a16d9043b4f98942339123a11497daac4a6abddb4e70d507a6c5a70`
- `study_out/local_work/vertical_noise_step22/PLAN.txt` — SHA-256 `57f8eafeabd38fc8998068342dcf9a82ec15453c2b590c93e9d7992c53693b10`
- `study_out/local_work/vertical_noise_step22/RESULTS.txt` — SHA-256 `65a52dd6687767e7059465d2c47c9047f13205595f78e6ccc44bee93230dc35c`
- `study_out/local_work/dasher_need_step23/PLAN.txt` — SHA-256 `6175e5a76fcfaaadd1e63d02c292a827c6d3f8f5f17eea70f9814578c53f2dbe`
- `study_out/local_work/dasher_need_step23/RESULTS.txt` — SHA-256 `c0c69f4aff755aff8271babd6248d1d792a9e773e4c192b21335347891a5a25d`
- `study_out/local_work/cruise_speed_step24/PLAN.txt` — SHA-256 `a8defa04469959d7641322520dbd11dc4e78b2c188cbe28b556314c0202190f7`
- `study_out/local_work/cruise_speed_step24/RESULTS.txt` — SHA-256 `036ed2b56f576b27fafe05a6366a32cfd147c0365c8f1fabc162c38935f5de9b`
- `study_out/local_work/rest_zone_step25/PLAN.txt` — SHA-256 `4e2bc0fcedaeff911ee94eacd7548431a40c4fb9bc4e5bfe6ae63c61ec0df992`
- `study_out/local_work/rest_zone_step25/RESULTS.txt` — SHA-256 `cdb237006f756f8aa00e2778dd2e17af7bbbd277161f14aae3730f8012b6db68`
