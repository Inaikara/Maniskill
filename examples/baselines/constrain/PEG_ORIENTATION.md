# Peg orientation environment foundation

## Retained best result and scene-0 videos (2026-11-23)

The best completed paired comparison is the nonblind random-500 update-25k
result: random 23/100 versus Base 3/100 and fixed 54/100 versus 52/100. Those
weights are hardlinked under `outputs/ckpt/retained_orientation/PegInsertionSide/`.
The original scene-0 recordings are copied to `outputs/video/PegInsertionSide_BaseDP/`
and `outputs/video/PegInsertionSide_ConstrainNet/`. This does not replace the
formal corrective retest's +12 pp qualification result or delete other checkpoints.

## Round 67 — fresh terminal-50k outside-entry observation

The authorized three-arm follow-up completed in
`outputs/exp/20261119_063749_peg_entry_50k_comparison/`: **180/180 paired
groups, 540/540 arm outcomes**, 155,029 physical controls including full arm
prefixes/initializations/suffixes, exit 0, 3,446.54 s. The existing v2
pose-window bank was read-only. It contributes six origins per profile reused
across five conditions and three inference draws; 90 draws/profile are
descriptive repeated-origin observations, not independent scenes.

Base / random-500@30k / random-500@50k successes were **73/180, 85/180,
84/180**. The terminal 50k minus 30k delta is −0.6 pp pooled descriptively,
**+3.3 pp random** (28/90 vs 25/90) and **−4.4 pp fixed** (56/90 vs 60/90).
Condition and per-seed deltas vary in direction; there is no consistent
monotone gain. This is terminal-checkpoint comparison only: the resumed 20k
uses the registered extension LR schedule and sampler cursor restart, so it is
not a pure update-count causal effect or scaling law. No intermediate
checkpoint evaluation/selection, training, or new data occurred. Independent
audit confirms source mappings, exact canonical handoffs, and unchanged
checkpoint/input/source hashes. The post-run 16-file focused suite passed
**141 tests**. Full tables, hashes, budget, and per-arm records are in
`results_summary.md` and `independent_audit.json` under the run directory.

Do not infer generalization from this reused diagnostic bank or launch further
evaluation/training without parent direction.

## Goal-loop round 64 — random-500 continuation to terminal update 50k

Completed the authorized continuation from the exact update-30,000 parent
`outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_30000.pt`
(SHA-256 `6e26647b5ee1ad356d57f06464f77a1884fcd76b1fad52c8ed181e9e0d1de1de`).
Exactly **20,000 new optimizer updates** completed, reaching cumulative step
**50,000** (exit 0; 625.432 s training elapsed). The preregistered terminal
checkpoint is `outputs/ckpt/20261119_135500_peg_orientation_random500_extend50k/update_50000.pt`
(SHA-256 `1e0b6246c7b01f7939f4e65d74255295cfd38e7c17982df0fb14ab7430bb060a`);
provenance checkpoints at 35k/40k/45k are retained alongside it. No best-checkpoint
selection or physics/policy evaluation occurred.

This restores the parent AdamW and EMA states, retains seed 1, batch 1,024 and
the same 512 random-500 + 512 fixed-1,000 examples per update, frozen Base, data,
model architecture and hint. The sampler has no saved cursor and restarts; this
is disclosed, not an exact trajectory resume. The original 30k cosine schedule
had reached zero LR. The continuation therefore uses a distinct 20k extension
phase (500-update warmup to `1e-4`, then cosine to zero); it is **not** presented
as an original 50k-from-scratch schedule. Terminal raw weights, EMA, optimizer
parameter steps and EMA optimization step all advanced to 50k and differ from
the parent. The 200 logged losses (every 100 steps, 30,100–50,000) are finite;
range `0.0011584–0.0072619`. These are training-loss diagnostics, not evidence
of policy-performance improvement.

All 14 registered data/base/catalog inputs and 19 archived source files match
their post-run hashes. The exact required 16-file suite passed **139 tests**
before training (13.39 s) and after (13.28 s; outer elapsed 14.56 s), with
180-second TERM / 10-second KILL bounds. GPU was an NVIDIA GeForce RTX 5070;
environment versions, exact commands, raw output, timing, input and checkpoint
hashes, and offline state audit are retained under
`outputs/exp/20261119_135500_peg_orientation_random500_extend50k/`.
The terminal model is preregistered for a subsequent fresh same-bank evaluation;
until then there is no observed policy improvement or selected checkpoint.
Suggested next comparison remains the registered three-arm fresh-Base/500@30k/
500@50k, same 60-bank × 3-seed design (540 arms), with 200-suffix follow-up;
do not evaluate intermediate 35k/40k/45k checkpoints for selection.

## Fix-only Round 63 — fresh-run seed ordering

Moved fresh-run-only Python/NumPy/Torch seeding ahead of model construction in
`peg_orientation_constrain.py`, so random initialization of the Base/CN model
(including the new hint encoder) is deterministic for seed 1. Resume still
constructs all components first, then restores parent checkpoint RNG/state
before the first loader iterator is created. No optimizer update or native
physics was run. A regression drives the actual `train()` seam twice with mocked
data/network plumbing and verifies identical randomly initialized states; the
existing resume RNG restoration test remains passing.

Round 63's expected-red log is newly captured at
`outputs/exp/20261119_180000_peg_entry_scaling_development/round63/expected_red.log`
(1 failed, showing different initial network state); two initial test-harness
failures are retained separately. Round 62's original expected-red log was
missing and is not reconstructed. The correct 16-file suite, including failure
analysis and entry-scaling tests, passed **139 tests in 13.45 s**; exact output,
outer timing and UTC bounds are retained in the same `round63/` directory. An
initial, incorrect 16-file list also completed (140 passed) and is retained
separately as `actual_review_suite.log`; it is not the required-suite result.
No full baselines suite, training, checkpoint/data mutation, ManiSkill source
edit or commit was made.

## Native round 61 — outside-entry checkpoint/data-size comparison

The preregistered fresh six-checkpoint run completed once at
`outputs/exp/20261119_025526_peg_entry_data_scaling_v2/`: **180/180 groups,
1,080/1,080 arms**, 306,628 native controls including replay, and 7,203.799 s
native elapsed (exit 0). It reused the complete 60-case bank, the same six
origins per profile across five conditions, and three inference seed offsets;
these are repeated-origin descriptive draws, not 180 independent scenes.

Primary (Base / 100@30k / 200@30k / 500@30k) all-condition totals were random
**11/90, 30/90, 27/90, 25/90** and fixed **62/90, 60/90, 57/90, 60/90**.
Secondary (Base / 100@15k / 200@30k / 500@25k) totals were random
**11/90, 33/90, 27/90, 27/90** and fixed **62/90, 60/90, 57/90, 57/90**.
Condition-level counts/rates, per-seed tables, raw-trace paired transitions, and
binary primary/secondary success-pattern monotonicity are in `results_summary.md`
and `results_summary.json` in that run directory. The result is mixed and does
not support a consistent monotonic response or a scaling-law claim. No exponent,
significance test, checkpoint selection, retuning, or full-task gate claim was
made. There is one training seed per data size; historical provenance for the
100-demo checkpoints remains incomplete, so data count is not isolated
causally.

All 21 preregistered input hashes and 38 source-closure hashes match post-run;
source-to-bank mapping was independently resolved via catalog seed → source
`episode_seed` → HDF5 `episode_id` for all 60 bank rows. All 180 handoffs had
exact matching saved histories and zero measured physical-state error against
Base (threshold 1e-5); exposed controller state keys were empty. No claim is
made about hidden environment state. The original failed validator record and
the earlier zero-native setup directory remain unchanged; the v2 cardinality
correction is documented in `parent_pre_native_validation.json`. The reused
135-test pre-suite is identified as reused evidence; the actual post-run 16-file
suite passed **135 tests in 13.32 s**. See `independent_audit.json`,
`posthashes.json`, `post_suite_timing.json`, and `final_artifact_hashes.json`.
No training, data/checkpoint/source mutation, ManiSkill edit, or video capture
occurred. Bank velocity is diagnostic only and is not treated as static state.

## Fix-only round 60 — scaling replay budgets

The scaling evaluator now passes its registered **669,600-control / 32,400 s**
limits into the existing paired replay preparation. `_prepare` accepts optional
limits while retaining the established two-arm defaults (**223,200 / 10,800**).
Regression tests call the real preparation path at each legacy cap and verify
that scaling proceeds in the fake physics environment while default calls
reject. No generic budget framework or rollout change was added.

The new regression was captured red before the fix: `_prepare()` rejected the
scaling limit kwargs (2 failed, 8 passed), raw output
`outputs/exp/20261119_180000_peg_entry_scaling_development/round60_expected_red.log`.
Post-fix, the full 16-file Peg suite passed **133 tests in 16.46 s**. Command
used activated `maniskill`, `PYTHONPATH=.`, single-thread OMP/MKL, a 180 s
timeout with TERM and 10 s KILL grace; raw output is `round60_required_suite.log`
in the same development directory. Round 59's original expected-red raw log
was not retained and is unavailable; it was not reconstructed. The first
Round 60 test invocation also had a collection-path error, preserved separately
as `round60_collection_failure.log`; the correctly configured red and green
runs are retained. No native rollout, training, or ManiSkill source change.

## Goal-loop round 59 — outside-entry training-volume scorer development (offline)

Added `peg_orientation_entry_scaling_eval.py` as a separate six-arm scorer; the
proven two-arm scorer is unchanged. Roster is the registered Base plus distinct
Constrain EMA checkpoints: random-100 at 15k and 30k, random-200 at 30k, and
random-500 at 25k and 30k. Primary curve: Base / 100@30k / 200@30k / 500@30k;
secondary: Base / 100@15k / 200@30k / 500@25k. Base and 200@30k are scored once
per draw and shared. All use the existing 60-row outside-entry pose-window bank,
not the older inside-hole bank, and independently replay the full expert action
prefix plus initializer before suffix inference. Source IDs are resolved by
catalog seed → source `episode_seed` → HDF5 `episode_id`.

Checkpoint audit is `outputs/exp/20261119_180000_peg_entry_scaling_development/`
(`checkpoint_comparability_audit.json`, `comparability_table.md`, raw
`checkpoint_configs.json`, `adapter_smoke.json`). All five
Constrain models use the exact Base EMA SHA, same 39-D observation / 7-D native
`pd_ee_delta_pose` contract, 2/8/16 horizons, 100-step DDPM, seed 1 and EMA power
0.75. Random coverage manifests show 100 ⊂ 200 ⊂ 500 by seed; their random-data
hashes differ, with fixed-1,000 replay also in the 100-data checkpoints and in
the later models. The 100-model historical provenance remains incompletely
attested; manifest nesting does not prove every historical training input. No
claim of full provenance or isolated data-count effect is made. Six real adapters
passed offline finite `(1,2,39) → (1,8,7)` smoke calls; Constrain adapter loading
verified frozen Base weights. No task/environment inference was used in smoke.

The scorer summarizes paired transitions vs Base and adjacent sizes plus
per-seed and aggregate binary monotonicity patterns; it does not fit a scaling
law or report inferential tests. The development fixture exercises 2 profiles × 5 conditions × 6 origins × 3
draws = 180 groups / 1,080 model arms through real preparation and suffix code.
It verifies all six arms are constructed/prepared before queries, <=1e-5 handoff
matching, per-arm finite shape checks, live task-bound Constrain hints, source
episode mapping, failure journaling and completed-arm preservation. Primary and
secondary paired transitions are descriptive; outputs disclose six unique
origins reused across conditions and one training seed per data size. This is not
a formal scaling-law study. Budgets: 669,600 physical controls maximum (including
replay), 32,400 s native stage, 43,200 s whole stage, 300 s per arm, native
TimeLimit 1,000. **No native rollout or training was run**; parent review is
required before any six-model execution. The correct 16-file suite passed **129
tests in 14.36 s** (15-file historical required set plus this new test module). Offline
adapter smoke outputs and test evidence are in the same development directory.

## Goal-loop round 58 — native outside-entry paired policy comparison

The frozen Base and Constrain checkpoints were evaluated once on the round-55
pose-window bank in `outputs/exp/20261118_174230_peg_entry_policy_comparison/`.
All **180/180 pairs and 360/360 arms** completed across random/fixed profiles,
five single conditions, six fixed origins per profile-condition, and three
repeated inference draws (seed offsets 1/1001/2001) per origin. The policy arms
independently replayed the full expert prefix plus initializer before inference;
the official native insertion success predicate and 200-action suffix limit were
unchanged. Both adapters used their actual frozen weights and native 39d
observations / 7d actions. The real source episode was mapped by catalog reset
seed to source metadata `episode_seed` and then `episode_id`; independent
read-only verification matched all 60 source prefixes and full action hashes.

The head began **19.8247–20.4765 mm outside the physical front entry plane**
`x=-peg_half_length`; it was not already inside the channel as in the older
near-hole bank. Measured perturbation relative to each same-origin aligned
handoff was hole-frame Y **+4.993…+4.998 mm / −4.991…−4.999 mm**, or world-Z yaw
**+4.9989…+5.0004° / −4.9985…−5.0000°**. The fixed bank origins are reused across
conditions; three inference draws do not add independent scenes.

| Profile | Condition | Base | Constrain | Both / Base-only / Constrain-only / neither | Δ pp |
|---|---|---:|---:|---:|---:|
| Random | aligned | 2/18 | 6/18 | 2/0/4/12 | +22.22 |
| Random | lateral hole-Y +5 mm | 2/18 | 5/18 | 2/0/3/13 | +16.67 |
| Random | lateral hole-Y −5 mm | 2/18 | 7/18 | 2/0/5/11 | +27.78 |
| Random | world-Z yaw +5° | 3/18 | 6/18 | 0/3/6/9 | +16.67 |
| Random | world-Z yaw −5° | 2/18 | 3/18 | 2/0/1/15 | +5.56 |
| Fixed | aligned | 10/18 | 8/18 | 8/2/0/8 | −11.11 |
| Fixed | lateral hole-Y +5 mm | 13/18 | 15/18 | 13/0/2/3 | +11.11 |
| Fixed | lateral hole-Y −5 mm | 17/18 | 16/18 | 16/1/0/1 | −5.56 |
| Fixed | world-Z yaw +5° | 13/18 | 12/18 | 10/3/2/3 | −5.56 |
| Fixed | world-Z yaw −5° | 9/18 | 6/18 | 3/6/3/6 | −16.67 |

Across conditions, the repeated-draw descriptive aggregate is random Base
**11/90 (12.22%)** vs Constrain **27/90 (30.00%), +17.78 pp**; fixed Base
**62/90 (68.89%)** vs Constrain **57/90 (63.33%), −5.56 pp**. The aggregate is
secondary only: the same six origins are reused across conditions and these are
not 90 independent scenes per profile. Per-seed-offset tables, paired transitions,
full traces, setup/runtime provenance and independent recomputation are in
`results_summary.md`, `results_summary.json`, `independent_audit.json`,
`source_prefix_mapping_audit.json` and `native_attempt_001/pairs.jsonl`.

This small conditional diagnostic does **not** show a general Constrain advantage:
random draws favor Constrain descriptively, while fixed results are mixed and
favor Base in aligned and both yaw conditions. It is not a formal gate, full-task
rate, causal explanation, checkpoint-selection result, or Plug evidence. The
pose-window admission requires an 11-sample low-drift pose window; it is not a
claim of zero instantaneous PhysX velocity. In the registered bank, instantaneous
pose-window sample diagnostics were peg linear speed **20.544–29.469 mm/s** and
angular speed **0.590–1.314 rad/s**; these were not admission limits. No videos
were preregistered or recorded. The historically executed 15-file suite passed
**130 tests** pre/post, but substituted
`test_constrain_adapter.py` for the required `test_peg_orientation_failure_analysis.py`.
The correct required 15-file suite was run after review: **123 passed in 11.42 s**,
with activated `maniskill`, single-thread settings and a 180-second TERM / 10-second
KILL bound. Exact command, raw logs and timing are in `post_review_required_suite/`.
This does not retroactively repair historical pre-run coverage. A historical
post-completion bounded rerun of the substituted list also passed 130. The
pre-suite raw output is retained, but its outer UTC/monotonic wrapper timing was
not captured and it used the tool timeout rather than the shell timeout wrapper;
this provenance limitation is explicitly recorded, not backfilled. Native outer
monotonic elapsed time was **2,318.40 s**, exit 0. All 94 registered input hashes
and 33 finite source closure hashes matched pre/post; the two checkpoint
identities remained the frozen Base (`067a4f5c…b62beea`) and Constrain
(`73834286…b82b7df`). No code, ManiSkill source, data, checkpoint, or formal
result changed. Official success checks peg-head hole-frame coordinates
(`x >= -0.015 m`, with Y/Z inside hole bounds); it does not require whole-peg
containment or an independent orientation threshold.

## Goal-loop round 55 — prospective pose-window initializer bank (model-free)

Fresh bank: `outputs/exp/20261119_164500_peg_entry_pose_window_initialization_v2/`.
Before this run's first native control, it froze the same 12 source IDs as rounds
52–53 (random `[0,2,3,4,6,7]`, fixed `[0,5,6,7,10,11]`), all five single
conditions, and continuous feedback only. New-bank admission is explicitly
`pose_window`; legacy `instant_velocity` remains the default and retains its
original five-sample plus instantaneous-velocity limits. The new rule requires
11 consecutive post-withdrawal control samples spanning 10 actual control-time
steps. Each sample must meet head target/norm/axis, true-entry clearance,
relative-yaw, grasp, non-insertion and signed lateral criteria. Across the
window, head and peg-center pairwise diameters must be ≤0.25 mm, full-quaternion
rotational diameter ≤0.25°, and each arm-joint qpos range ≤0.002 rad. Instant
peg/arm velocities remain recorded diagnostics, not pose-window hard gates.
This operational window is not a claim of zero substep motion or absence of
aliasing.

All **60/60** registered cases completed once and independently recompute as
admitted: random and fixed each **6/6** in aligned, hole-Y ±5 mm and world-Z yaw
±5° (no replacements). All window bounds independently pass. Across admitted
windows, maximum head-position diameter was **0.2464 mm**, peg-center diameter
**0.2264 mm**, quaternion rotational diameter **0.1498°**, and per-joint qpos
range **0.001969 rad**. Actual signed lateral delta ranges were **−4.327 to
−3.809 mm** for the negative condition and **+5.669 to +6.176 mm** for positive;
they do not overlap. Realized relative yaw ranges were **−5.556 to −5.200°** and
**+4.443 to +4.799°**, also nonoverlapping. These are selected, expert-reachable
validation-bank initializations, not a whole-scene rate or policy outcome.

Instantaneous diagnostics over the 660 admitted samples: peg linear speed
**20.544–29.469 mm/s** (median 25.050), angular speed **0.590–1.314 rad/s**
(median 0.848), maximum arm qvel **0.00102–0.05283 rad/s** (median 0.01730).
Thus this mode deliberately differs from strict instantaneous-velocity
stationarity; do not call the new bank “zero velocity” or interpret the API as
wrong. Native work was **8,225 prefix + 1,370 initializer = 9,595 controls**;
runner 59.52 s, outer 62.86 s. All registered 57 input/source hashes match
pre/post. Both checkpoint references were hashed before/after (unchanged) and
never loaded or queried. Exact 14-file suite: **113 tests** before and after
(12.08/12.14 s; timeout 180 s, TERM/KILL grace 10 s).

Transparency: the first fresh attempt at
`outputs/exp/20261119_164200_peg_entry_pose_window_initialization/` completed
60 cases, but its stop condition ended at the first 11 per-sample pose-eligible
controls without requiring the full window-diameter bounds. Its final admission
still applied those bounds and recorded 0/60; there were no false admissions.
It is preserved as a runner-defect incident, excluded and not reclassified. The stop
condition was fixed and the separate v2 registration/archive/run above completed
once. No expected-red raw file was retained before discovery; this gap is
explicit rather than backfilled. No learned policy was run, no checkpoint or
training/data artifact changed, and no formal-test or Plug work occurred. Rounds
52–53 historical 0/60 results and all formal-gate statuses remain unchanged.
See v2 `independent_audit.json`, `post_execution_audit.json`, raw logs, archived
runner/tests/wrapper and the v1 `runner_incident_v1.json`.

## Goal-loop round 54 — paired initializer settling calibration (no policies)

Fresh, bounded model-free calibration: `outputs/exp/20261119_000000_peg_entry_settling_calibration/`.
Before outcomes, it froze the first selected random and fixed source-bank scenes
(catalog ID **0** in each), all five original single perturbation conditions,
and paired strategy order: original continuous peg-to-TCP feedback, then a
single-latch fixed-world-TCP hold. The fixed strategy latched a copied actual
world TCP position/quaternion at the first pose-only-qualified state after
withdrawal and kept that target immutable. All pose, yaw, clearance, stability,
velocity limits and five-consecutive-control requirements remained unchanged.
The signed lateral bands were explicitly registered. No scene replacement,
teleport, weight load/query, training, data change, policy rollout, formal test,
or Plug work occurred.

All **20/20** cases completed once and all were rejected by the original
initializer admission rule (**continuous 0/10; fixed TCP 0/10**, also 0/5 per
strategy/profile). Native controls were **2,650 expert-prefix + 1,300 initializer
= 3,950**; runner elapsed 27.35 s and outer monotonic elapsed 29.70 s. Each
profile/condition pair began after withdrawal with exact physical-state,
observation-history, head-pose and prefix-hash agreement (maximum numeric
difference **0**). Prefix hashes matched all 20/20; positive/negative native
yaw preflights passed all 20/20. Inputs—including both referenced checkpoint
files—matched pre/post; hashes were recorded but weights were **not loaded**.

Every case reached the physical outside-entry clearance range **19.920–20.667
mm**. Continuous feedback retained final head error **1.256–1.776 mm**; fixed
TCP hold ended at **2.331–3.682 mm**. Both arms' last-five peg linear speeds
were **22.855–23.690 mm/s** (unchanged limit 5 mm/s), and angular speeds
**0.597–0.863 rad/s** (limit 0.05 rad/s); arm qvel remained ≤0.01229 rad/s
(limit 0.02). Thus fixed TCP did not improve admission or measured motion
stability in these ten paired branches and had larger position drift; this small
calibration does not establish a general cause or say anything about policy
behavior. Do not infer a model/controller root cause or relax thresholds.

The fixed target was latched once at perturb control 10 in all ten fixed-TCP
cases; per-command phase, action hash and canonical target hash are in
`independent_audit.json`. **Per-control timestamps were not emitted by the
runner**, so they are unavailable and were not imputed; only registered outer
and runner elapsed times are reported. The exact 14-file regression passed
**111 tests** pre and post (11.50 s / 11.44 s). Expected-red, raw logs, command,
source archive/closure, full journals, paired rows, registration, timing and
post-run audit are retained in the experiment directory. This is initializer
calibration only—not a new initialization bank, policy comparison, formal gate,
or learning result. Existing 0/60 rounds 52–53 and all formal counts/statuses
remain historical and unchanged.

## Goal-loop round 53 — repaired outside-entry perturbation initialization (no policies)

Fresh attempt: `outputs/exp/20261118_155806_peg_entry_perturb_initialization_v2/`.
It replayed the same first six previously admitted source IDs per profile as
round 52 (random `[0,2,3,4,6,7]`, fixed `[0,5,6,7,10,11]`), all five unchanged
single perturbation conditions, and no replacements. All **60/60** cases
completed once; **0/60** met the prospective initializer-only admission
criteria. No policies were queried and no weights were loaded.

The repaired cap-return rows retain the executed prefix and initializer actions,
action hashes, full actual observation history, per-control physical/controller
measurements and requested/realized commands, final physical state, positive /
negative native yaw calibration, and the actual final outside-entry geometry.
Independent recomputation confirms 60/60 prefix hashes and controller
calibrations, and all 55 registered sources/inputs match pre/post. Native work
was **8,225 prefix + 3,900 initializer = 12,125 controls**, 82.80 s runner / 85.90 s
outer, exit 0. The exact 14-file Peg suite passed **109 tests** before and after;
the bounded post suite also passed 109 (180 s TERM / 10 s KILL).

All cases reached true front-plane clearance **19.824–20.476 mm**, and final
head-target norm error was **1.256–2.690 mm**; the final five head-position and
yaw checks passed. The measured cap-path blocker was motion: final-sample peg
linear speed was **21.864–28.085 mm/s** (limit 5 mm/s), angular speed
**0.597–1.310 rad/s** (limit 0.05 rad/s); final-sample arm qvel reached
0.03673 rad/s (limit 0.02). Across all 300 final-five samples, linear speed
was 21.812–28.088 mm/s and maximum arm qvel was 0.036744 rad/s. Raw peg
velocity vectors were not retained; these are the recorded scalar norms. Every case
exhausted the registered 60-control settle allowance, with no quiet retry.
These data do not identify the cause or imply a policy/model defect. The
newly measured outcomes do not retroactively reclassify round 52's historical
0/60, and do not authorize policy scoring or a next-stage Base/CN comparison.

**Registration limitation:** although the runner applied the signed lateral
admission band `[3,7] mm` in the expected direction, the frozen JSON
preregistration omitted that exact band. It is disclosed in
`preregistration_deviation.json` and is not claimed as preregistered; all cases
independently fail recorded velocity limits, so it does not change the 0/60
outcome. The first expected-red test output was displayed only in the tool
console and was not retained as a raw file. Do not claim a saved red log. Raw
pre/post green logs, source/wrapper archive, per-case journals, hashes, timing,
and the unrounded per-case `independent_audit.json` are in the experiment
directory. Round-52 original artifacts remain immutable. Existing formal gates,
historical counts, weights, and Plug status are unchanged.

The referenced checkpoint files were not loaded; their current byte hashes
match the preregistered reference digests. They were not part of the registered
pre/post input-hash set, so no before/after checkpoint identity claim is made.

## Goal-loop round 52 — physical outside-entry perturbation bank (no policies)

The single frozen initialization attempt is
`outputs/exp/20261118_152038_peg_entry_perturb_initialization/`. It selected the
first six already-admitted validation-bank catalog IDs in each group, frozen
before outcomes: random `[0,2,3,4,6,7]`, fixed `[0,5,6,7,10,11]`. All 12 IDs
received exactly five registered conditions (aligned; hole-frame Y ±5 mm;
world-Z yaw ±5°), with no combined perturbations, replacements, policy
inference, checkpoint loading, or training.

**No condition met the frozen initialization admission criteria (0/60).** All
60 cases independently replayed their original expert EE prefixes and reached
the 20 mm outside-entry region, then exhausted the 60-control perturb/settle
allowance. Final true front-plane clearance was 19.824–20.476 mm. Final head
target error was 1.256–2.690 mm (median 2.068 mm), above the frozen 1 mm bound;
the original runner did not retain the terminal five-control velocity and yaw
window on its cap path, so those criteria cannot be audited retrospectively.
Do not read this as a policy failure, a Base-vs-Constrain result, or evidence
that the requested perturbations were admitted. Do not proceed to policy
scoring from this bank. A repaired initializer and any fresh native attempt
need parent review; this completed run is not retried or replaced.

Native accounting reconciles to **8,225 prefix replay + 3,900 scripted
initializer = 12,125 controls**, 60 complete condition journals, 72.10 s runner
elapsed / 74.58 s outer elapsed. Source-prefix action hashes reconstructed
offline match all 12 source-bank hashes; preregistered source/input hashes match
pre/post. The exact 14-file Peg suite passed **106 tests** before and after
(11.34 s / 11.23 s). Actual installed-controller calibration passed positive
and negative yaw checks using the native clip/scale path: frame
`root_translation:root_aligned_body_rotation`, `use_delta=True`,
`use_target=False`, position and rotation bounds ±0.1, public peg linear/angular
velocity APIs available.

The attempt's immutable cap-return rows omitted source-prefix action arrays,
last-five velocity measurements, per-control official-insertion flags and the
final physical-state snapshot. `initialization_reconciliation.json` additively
reconstructs exact expert-prefix and scripted actions, action hashes and final
head/entry geometry from the preregistered source files and preserved trace; it
does not infer unavailable qpos/qvel or insertion history. Raw pre/post suite
logs, commands, timing, registration, wrapper/source archive and all journals
are retained. The first expected-red collection output was only in the tool
console and not saved as a raw file; this provenance gap is disclosed rather
than backfilled.

**Important geometry correction:** the prior bank's head x near −0.04 to
−0.05 m was already inside the physical entrance: the source hole frame's
front plane is `x = -peg_half_length` (half-length 0.085–0.125 m). The official
success boundary `x >= -0.015 m` is near the box center, not the hole mouth.
Round-46 eligibility/counts and round-50 results remain historical and
unchanged; the new protocol deliberately withdraws to 20 mm outside the real
entry plane. No change to ManiSkill, datasets, weights, formal gates, or Plug
status occurred.

## Goal-loop round 51 — model-free near-hole hold attempt stopped before native execution

The authorized one-pass hold control did **not** begin. The attempt at
`outputs/exp/20261118_142454_peg_near_hole_hold_control/` exited 1 in runner
setup because its output directory had already been created to retain the exact
pre-run regression artifacts, while the runner requires `mkdir(exist_ok=False)`.
The traceback is preserved; native controls and completed arms are both **0**.
No environment was created, no prefix was replayed, and there was no retry.
The source archive, closure, inputs observed after the setup failure, wrapper,
timing, and raw logs are retained in that directory. The exact 13-file suite
passed 99 tests pre-attempt (11.41 s) and post-attempt (11.33 s).

At this point in the chronology, the intended control had no outcome: one deterministic
`[0,0,0,0,0,0,-1]` native `pd_ee_delta_pose` hold per each of the frozen 15
random and 11 fixed eligible round-50 prefixes, with the original exclusions
and no replacements. Historical Base/Candidate checkpoint identities are
references only and were not loaded. This setup failure supplies no evidence
about carryover, controller hold, or learned-action necessity; it does not alter
round 50's 45/45 and 33/33 results or any formal gate.

Round-51 completion addendum: following the parent's explicit setup-fix decision,
the first failed directory remains unchanged and a fresh attempt completed once
at `outputs/exp/20261118_143004_peg_near_hole_hold_control_v2/`. The minimal
workspace guard now accepts its precreated wrapper/test-log directory but rejects
any prior registration/result/journal/archive. All 26 frozen eligible branches
completed: random 0/15 and fixed 0/11 official success; all retained grasp and
used the full 200-control suffix. Native work was 3,537 prefix + 5,200 hold =
8,737 controls in 46.18 s runner / 48.19 s outer elapsed. Every outgoing action
was the exact zero-pose/closed-gripper vector; replay physical/history/head error
was 0. The two in-pass videos decode to 201 frames each. Frozen bank, catalog,
demo and runtime source identities match round 50. The 101-test exact suite
passed pre/post (11.39 s / 11.54 s). This only shows no official completion
under this exact controller-hold command on these moving, grasped, prealigned
branches; zero delta can actively hold the EE target. It does not prove all
velocity confounds removed or establish general learned-action necessity. No
checkpoint was loaded; round-50 draws remain unpooled and all other gate/status
results are unchanged. See the v2 `analysis_report.md` and `independent_audit.json`.

## Goal-loop round 50 — near-hole paired policy score completed

The fresh authorized run is
`outputs/exp/20261118_220128_peg_near_hole_policy_comparison_v2/`. It used the
frozen Base EMA (`067a4f5c…b62beea`) and random-500 update-25,000 EMA
(`73834286…b82b7df`), native 39d/7d adapters, DDPM-100, and identical
independently replayed expert EE prefixes. All **78 pairs / 156 arms** completed:
random eligible branches were **15/15** for each policy at each seed offset
`1,1001,2001` plus catalog index; fixed branches were **11/11** for each offset.
Pooled descriptive repeated draws: random **45/45 vs 45/45** over 15 unique
branches, fixed **33/33 vs 33/33** over 11. Paired transitions were 0
Base-only / 0 Constrain-only / N both / 0 neither. Three draws per branch are
not independent scenes; ineligible entries from the original 20+20 catalog are
not policy failures.

The expert prefixes already placed the peg head at random x `[-0.049854,
-0.040661] m`, yaw error `[-0.012903,-0.003340] rad`; fixed x `[-0.048915,
-0.041266] m`, yaw `[-0.010893,-0.003375] rad`. This is an already-grasped,
closely aligned conditional task slice with moving expert handoffs, not static
randomized near-hole resets. Maximum absolute arm-joint velocities were
0.280–0.386 rad/s (random) and 0.268–0.382 rad/s (fixed); success took only
3–4 controls. Residual expert motion and ceiling effects remain possible, so
these results do not establish that learned actions were necessary. They do not
establish equality on full-task 500-demo performance or identify an upstream
cause. Round 40's whole-task random 23/100 and fixed 54/100 use a different
protocol and cannot be directly equated. No inference about end-to-end gates or
Plug follows.

All 78 unique `(profile,index,replicate)` results reconcile; 21,222 prefix
replay + 486 suffix controls = **21,708 physical controls**, 181.72 s runner
elapsed / 183.94 s wrapper-monotonic elapsed, below the 78,000-control and
7,200/10,800-second bounds. Both commands exited 0; no failed attempt. Maximum
bank-vs-replay physical/history/head error was 0. Public controller state was
empty (`get_state()` extent only; hidden internal buffers are not claimed),
mode matched as `pd_ee_delta_pose`, and actual two-frame histories matched
before inference. Four videos were captured in-pass, decoded, and hashed. The
exact 12-file suite passed 97 tests pre/post; all registered sources and 55
input identities reconcile. The comparison retains the bank's retrospective
fixed-stage amendment and unavailable initial historical source bytes as
limitations. See the experiment's `analysis_report.md` and `independent_audit.json`.

No training, data collection, checkpoint mutation, full-task evaluation, or
Plug work occurred. Round 40's numerical gate outcomes remain unchanged.

## Goal-loop round 49 — near-hole policy input batching repair (no native)

Both near-hole adapters now receive the same latest-two 39d observations with
an explicit batch axis, `(1, 2, 39)`. The strict full-comparison fake-adapter
regression first reproduced the real `_run_suffix` call's old `(2, 39)` shape,
then passed after the one-line batching fix; it checks both adapters and the
latest two history values. The unchanged exact 12-file regression passed **97
tests**. Raw red/green output, exact command, timing, and file set are retained
in `outputs/exp/20261118_215301_peg_near_hole_batching_repair/`. No model,
normalization, controller, history values, native scoring, or old failed-run
artifact changed. Near-hole policy scoring remains pending.

Near-hole bank provenance clarification: see
`outputs/exp/20261118_122546_peg_near_hole_initialization/provenance_correction.json`.
The fixed-stage amendment was recorded after that stage and is retrospective;
initial historical source bytes were not archived. `commands.txt` was edited
after its historical audit, and its updated hash does not backfill provenance.
Before policy scoring, independently replay each prefix and check both arms'
public controller state, physical state and actual history within `1e-5`.
Bank mode/pose snapshots alone do not establish controller-target equality.

## Goal-loop round 48 — near-hole comparison runner repair (no native evaluation)

Round 47's failed attempt at
`outputs/exp/20261118_130407_peg_near_hole_policy_comparison/` remains unchanged.
The repair is limited to four confirmed defects in the new comparison runner:
use the actual task-level near-hole predicate after prefix replay; retain prefix
metadata in completed rows for final geometry; freeze/hash/archive the current
outer wrapper separately from runtime sources; and persist global physical
control-budget/failure evidence while preserving completed arm rows and the
original exception. NumPy 0-D values now normalize recursively. Native and whole
stage deadlines are explicitly 7,200 s and 10,800 s, respectively.

The fake-environment full-orchestration regressions call the real preparation,
suffix, predicate and finalization paths; they use no genuine model, policy
weights, or native physics. They verify six complete pairs across both groups,
retained geometry and paired counts, current-wrapper archive identity, and an
injected second-arm failure retaining Base's complete outcome and physical
controls, and a failure during prefix preparation retaining its replayed-control
count. The exact 12-file suite passed **97 tests** in final pre/post runs; raw
red/green command output, logs and deadline metadata are in
`outputs/exp/20261118_132711_peg_near_hole_eval_repair/`. No native controls,
policy inference, scoring, training, data changes, or Plug work occurred. The
round-47 142-control setup failure and its evidence are preserved; this repair
does not itself authorize a native retry. Round 40's reported **23/100 random**
and **54/100 fixed** comparison and all formal-gate statuses remain unchanged.

## Goal-loop round 47 — near-hole policy comparison stopped before inference

The new comparison is registered at
`outputs/exp/20261118_130407_peg_near_hole_policy_comparison/`. Its exact
12-file regression passed **91 tests** both before and after the native attempt
(11.93 s / 11.44 s). Frozen Base/Candidate identities, bank/catalog/source
hashes, runtime, archived runner/test/controller source, command and preregistration
are retained. The registration preserves the bank's retrospective fixed-stage
amendment and unavailable historical source bytes; it does not claim a fully
preregistered bank.

The sole attempt exited 1 after replaying the random validation index-0 prefix
for **142 native controls**, before either policy produced an inference/action.
The runner called `near_hole_eligible(head_xyz, grasped, inserted)`, but that
function accepts the task object; the three-argument geometry predicate is a
separate helper. The original traceback, exact start/end timing, empty first-pair
failure journal, and post-execution source audit are preserved in the experiment
directory. No Base-vs-Constrain outcome, success count, video, or completed pair
is available. Source closure and archived bytes still match; no runner/test was
edited after the first native action, and no retry or substitute branch was run.

This was a setup failure, not a policy failure. A new registered attempt needs a
parent decision because the defect was found after native controls had already
run. The 500-demo study's historical random **23/100** and fixed **54/100**
comparison and all earlier gate outcomes remain unchanged; this adds no gate,
end-to-end, or Plug evidence.

## Goal-loop round 46 — validation near-hole initialization bank

This slice built near-hole initialization branches only; it did not load either policy, perform policy inference, train, or evaluate a gate. The frozen random/fixed validation catalogs contributed indices 0–19 each. No scenario replacement was allowed. Registrations, reports, journals, timings and raw logs are in `outputs/exp/20261118_122546_peg_near_hole_initialization/`; diagnostic expert source/replay packages are in `demos/PegInsertionSide-near-hole-diagnostic-random_20261118_122546/` and `...-fixed_20261118_122546/` and are not training data.

Collection produced random 15/20 expert successes and fixed 12/20. Official EE replay accepted 15/15 random and 11/12 fixed (one fixed conversion failed). Eligible bank branches: random **15/20**, fixed **11/20**; all ineligible rows remain in the denominator. The first post-action eligible state had native grasp, was not officially inserted, and had official hole-frame peg-head x in `[-0.05,-0.02] m` with Euclidean norm ≤`0.05 m`. No source branch missed the window after a successful replay. Random head-x/radius ranges are `[-0.049854,-0.040661] m` / `[0.040805,0.049920] m`; fixed ranges are `[-0.048915,-0.041266] m` / `[0.041382,0.049018] m`. The experts already aligned the peg: relative peg-to-hole yaw error ranged `[-0.01290,-0.00334] rad` random and `[-0.01089,-0.00337] rad` fixed. This is an expert-reachable conditional benchmark, not an all-scene success rate.

Both policies in the proposed next stage must independently replay the exact recorded converted EE prefix from the same catalog reset, verify state/history/controller mode before inference, and receive identical two-observation history. These branch prefixes are not part of the 200-control policy suffix. The future comparison remains unrun. Interpret it only as near-hole conditional paired performance; it does not establish end-to-end comparison or qualify Peg. The source demonstrates expert-reachable, already-aligned initializations, so report that ease explicitly. No Plug work or changes to old gate outcomes.

Random native stepping completed, but its initial JSON summary finalizer failed on NumPy arrays after all 20 per-scene attempts. The full progress journal existed; recursive normalization and a journal completeness fix were made. The summary was recovered offline from the journal without repeating any native action. The fixed stage used the amended runner; amendment, hashes and chronology are recorded in the experiment directory. Collection reports explicitly account for the planned 20 attempts per profile and their expert failures. No ManiSkill source was modified. The eleven-file Peg regression passed **88 tests in 11.39 s**. See the experiment `preregistration.json`, `fixed_stage_amendment.json`, `random_bank/`, `fixed_bank/`, `attempt_*`, and source audit for detailed traceability.

## Goal-loop round 45 — bidirectional takeover diagnostic (not a gate retest)

Fresh registration `outputs/exp/20261118_102350_peg_orientation_bidirectional_takeover_v2/`
bound the correct `validation_random` catalog (scenes SHA-256
`cdc3f1ee5cc7fc36f4884158c21c86fd238037e87e07b64bcafd427af3ae5d89`, manifest
SHA-256 `a2d2b7c8a0f5981f9174bfe9c3ef1ef56469ca1edc61738fce70dcfb43d4f3bb`).
The prior fixed-catalog constant was removed from this runner; preparation now
guards the manifest split. All registered sources, inputs, result hash and 55
journal hashes reconcile post-run.

All eight selected training near-branches triggered: recorded suffix **8/8**,
live official expert **8/8**, frozen policy **7/8**. Of the 12 random validation
scenes, five had native policy success before 296 and seven were eligible for a
matched pair. At the handoff, unchanged policy continuation succeeded **0/7**;
live expert succeeded **4/7**, with three `plan_complete` non-success outcomes.
Paired counts (both/expert-only/policy-only/neither) are **0/4/0/3**. Replay
branches matched qpos/qvel/root/TCP/peg/hole, two observations and action hash
with maximum numeric error **0**. Native run: **10,365 controls**, 263.90 s.

This supports physical feasibility under this expert for four validation
branches, not a universal oracle, irreversibility, an OOD cause, or pure policy
versus servo attribution: expert mode switch and controller reset are explicit
confounds. Training and validation differ in geometry/stage; sample count is
small and each policy branch has one stochastic sample. This diagnostic does
not rescore or change round 40's met numerical gates (+20pp random, +2pp fixed)
or its nonblind, single-seed and provenance limitations; it does not authorize
Plug work. No video was preregistered or
captured. Full analysis and audit are in the experiment directory's
`analysis_report.md` and `attempt_native_actual_once/`.

The earlier round-43 run remains unchanged and its outcomes/claimed 8,021 total
remain unavailable. That run's fixed-catalog and `10000+index` pre-restore seed
deviations are historical only and are not pooled. The fresh run preserves two
separate setup errors that exited before native stepping; the corrected native
attempt completed once.

## Goal-loop round 44 — takeover runner repair (development only)

No native environment or planner was invoked. The runner repair normalizes
NumPy scalar booleans before final hashing and failure serialization; writes
unique full normalized progress snapshots for every completed training arm and
validation update before proceeding, with snapshot hashes linked in the ledger;
uses validation policy seed `1+catalog_index` only after restoring the initial
scene while keeping the post-prefix RNG stream across the control-296 handoff;
and recognizes the live expert arm's top-level `status="native_success"` in
paired validation analysis. Training policy seed remains `10000+episode_id`.
The registration now records training and validation seed/history rules
separately. Source closure includes the runner and its own regression tests.

The exact ten-file suite failed first (**77 passed, 5 failed**) and then passed
(**83 passed in 11.49 s**). Raw logs, command, timing, and source-closure
evidence are retained in
`outputs/exp/20261118_100309_peg_orientation_takeover_runner_repair/`.

Round-43 artifacts at
`outputs/exp/20261118_175052_peg_orientation_bidirectional_takeover/` remain
unchanged. Their registered run hit a NumPy-bool hashing error and the exception
handler then failed to serialize the failure record. The only retained execution
data are a thin ledger and raw traceback, without complete per-arm outcomes,
traces, or action/history rows. Exact native success outcomes and native-control
aggregate are unavailable; do not claim 8,021 controls or reconstruct outcomes
from the ledger's counts. Round-42's separate positive-control result remains
2/2 per arm, 571 controls in 6.21 s and is not pooled with round 43.

Round 43's validation seed rule/timing differed from the corrected intended
future protocol. A fresh registration must separately freeze validation seed
`1+index`, applied after scene restoration, and the actual observation/RNG
continuity through handoff. This is not a new preregistration or native result.
No experiment gate or Plug status changes.

## Goal-loop round 42 — live expert-continuation handoff preflight

The user-authorized groundwork registered a bounded two-episode, training-data
positive-control study at
`outputs/exp/20261118_092013_peg_orientation_takeover_preflight/`. The candidate
update-25,000 EMA (`73834286…b82b7df`) and random-500 data were unchanged; no
candidate inference/checkpoint load, training, data collection, validation/test/
retention access, or Plug work occurred. Episodes 0/1 were fixed before rollout
(seeds 2026112000/2026112002; target yaw 1.95233/1.35036 rad), not chosen by
outcome.

Both training demonstrations reached the frozen trigger: first postaction state
with native grasp, official peg-head-in-hole x in `[-0.10, -0.015)` m, and no
official insertion. Trigger head coordinates were `[-0.099280, 0.002830,
-0.002582]` m and `[-0.098871, 0.000406, -0.000846]` m. Recorded suffix replay
and stock live expert continuation each achieved official native success in
**2/2** branches: recorded suffixes used 8/9 controls; the live planner used
47/35 controls. Both independent prefix replays matched action hashes, two
observation histories, robot qpos/qvel/root, TCP, peg and hole poses exactly
(max absolute error 0). At live handoff, public `set_control_mode('pd_joint_pos')`
plus controller reset produced zero physical controls, unchanged grasp, and max
physical-state error 0. The planner's gripper state was set CLOSED without
separate close-gripper controls. All arms together used **571 controls in 6.21 s**.

This shows a stock official planner can finish these two physically matched
training branches under a controller-mode intervention; it does **not** estimate
general recoverability or show that the learned policy lacks recovery. The live
arm changes from `pd_ee_delta_pose` to `pd_joint_pos`, so controller/target reset
is a comparison confound. Internal controller target-state snapshots were not
serialized by this runner; their omission is disclosed in the post-execution
audit and the confound is retained rather than presented as a pure policy
comparison.

All four physical arms completed once, but the registered runner exited 1 after
rollout while canonical-hashing a result containing a NumPy array. Original raw
failure/log/ledger are immutable; a separate read-only
`attempt_001/results_posthoc_reconciled.json` reconciles the completed rows, and
`reporting_reconciliation.json` records the error. **No native retry** occurred.
Pre/post source closure and data hashes reconcile. The exact nine-file regression
passed 70 tests before and after native execution. No gate, formal result, or
Plug status changes. Full registration, hashes, commands, timing, branch records,
source snapshots and limitations are in the experiment directory.

## Goal-loop round 40 — random-500 candidate nonblind 600-episode comparison

The validation-selected random-500 update-25,000 candidate was compared once
against frozen Base in the six preregistered 100-episode cohorts at
`outputs/exp/20261118_061124_peg_orientation_random500_comparison/`. Candidate
SHA-256 is `73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df`;
Base SHA-256 is `067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea`.
In frozen order, random Base/correct/fixed-standard/shuffled scored **3/23/4/4**;
fixed-retention Base/correct scored **52/54**. Paired random correct-vs-Base
Base-only/correct-only/both/neither is `1/21/2/76`, exactly **+20pp**. Paired
fixed retention (Base-only/correct-only/both/neither) is `4/6/48/42`, **+2pp**.
Thus this comparison meets both frozen numeric gates (random gain ≥20pp; fixed
loss no worse than −5pp). Correct vs
fixed-standard and shuffled gains are +19pp each, diagnostics only.

This is an explicitly **nonblind** comparison because the random test was exposed
in prior rounds, and it remains a single-training-seed result. It does not erase
or pool the previous formal candidate's +18pp/+6pp outcome or round-37's
+12pp/−3pp retest. Validation scores for random-100/200/500 candidates
(15/38, 20/34, 18/34) are likewise separate descriptive references, not a causal
demo-count comparison. `not_reached` remains a proximity-trigger-only diagnostic.
No Plug work was performed or started.

All six cohorts completed once: **600 episodes, 158,647 native controls, and
7,015.80 s** aggregate evaluator time. Each recorded a same-pass scene-0 video;
all 554 registered source files and 22 inputs reconciled. The exact eight-file
regression passed **64 tests** both before (10.84 s) and after (10.98 s) the
cohorts. Results, paired transitions, five random-yaw bins, timings, per-scene
rows, video hashes, pre/post regression logs, and audit are in the
experiment directory's `results.json`, `results.md`, `execution_ledger.json`,
`post_execution_audit.json`, and `post_regression.*` files.

Two invocations failed before native rollout: the first Base attempt encountered
an evaluator-owned output-directory collision; the first fixed-standard attempt
passed unsupported CLI value `standard`. Neither produced episode records or a
summary, and each corrected cohort completed once. The ledger retains failed-
attempt exit metadata and historical stderr hashes, but retries overwrote the
shared stdout/stderr paths: the original failed raw log bytes are unavailable,
so the recorded causes cannot be independently authenticated from those bytes.
No reconstruction was made. Runner hashes changed during execution: initial
failed Base `bee8bb232e9203fffea9c99fbec45ad93f5486a64ac8baeba73625f712a22454`,
intermediate `4b969575382e4651fac5bcbb59e0efb317e12d35942ffd25e895ab17b031a5d1`,
and final `34a65147208783b21faadb07aed7c03e68437015160b6574d38f11ff8593c506`.
The registration records wrapper edits for output handling and CLI mapping, but
historical runner bytes are unavailable; registration and current closure hash
do not prove one immutable driver across cohorts or lock the full environment.
Evaluator, adapter, model, catalog and registered inputs reconcile in the audit,
but that does not authenticate wrapper immutability. Native outcomes, summaries,
videos, bins and numerical results remain unchanged. See experiment
`reporting_correction.json` and `reporting_artifact_audit.json`. Meeting these
numeric thresholds does not establish universal data effects or authorize Plug.

## Goal-loop round 39 — random-500 fresh training and validation completed

The authorized follow-through to round 38's admitted 500-demo package completed once under `outputs/exp/20261118_035943_peg_orientation_random500_study/`; six checkpoints are in the matching `outputs/ckpt/` directory. Training restarted from unchanged Base EMA SHA-256 `067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea` with a fresh zero-control branch, seed 1, exactly 30,000 updates, batch 1,024, and 512 random-500 / 512 fixed-1,000 windows per update. The datasets contain 75,088 and 146,948 training windows respectively. All six EMA checkpoints ran once on both frozen validation-50 catalogs (12 cohorts, 600 episodes, 132,933 controls); the per-update results are 15/27, 18/32, 19/33, 18/34, 18/34, 18/32 (random/fixed).

Frozen selection rule: fixed validation ≥34/50, maximize random successes, latest update on ties, no fallback. Updates 20k and 25k were feasible and tied at 18/50 random, 34/50 fixed; update 25k is selected, SHA-256 `73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df`. Against reused Base validation 4/50 and 36/50, its paired net changes are +14/50 random and −2/50 fixed. On aligned scenes vs the historical random-100 candidate (15/50, 38/50), the selected candidate is +3/50 random and −4/50 fixed; vs the historical random-200 candidate (20/50, 34/50), it is −2/50 random and 0/50 fixed. Historical candidates were not rerun. These are validation-only comparisons, not formal qualification or evidence that demo count alone caused the change.

Training elapsed 933.76s; validation elapsed 6,174.02s. All commands exited zero once, all 555 registered source paths and 13 registered inputs remained unchanged, and the post-training eight-file regression passed 64 tests. Full pre-registration, commands, runtime versions, hash/timing ledger, checkpoint identities, per-episode outcomes, paired transitions, and final audit are in the experiment directory. The unchanged historical formal gate remains failed (+12pp random in the nonblind retest); this stage ran no final-test, fixed-retention-100, additional data, or Plug evaluation. The single-seed, cumulative-budget limitations remain; do not claim a causal sample-count effect.

## Goal-loop round 38 — random-500 data expansion completed (data-only)

The user's forward objective is **“将随机演示扩充到500条 然后再观察是否有提升 继续研究”**. This supersedes the former random-300 cap and the round-37 no-further-data decision only for Peg random-demo research through 500; it does not rewrite the round-37 negative formal gate, pass any prior gate, or authorize Plug. The new admitted package is `demos/PegInsertionSide-orientation-random-500/`: the prior 200 HDF5 episode payloads are retained unchanged, followed by the first 300 official CPU replay acceptances from the new source in source order. No policy outcome was used to select rows.

Before collection, `outputs/exp/20261118_033057_peg_orientation_random500_data/preregistration.json` froze the objective/scope, source identities, collision-checked candidate interval `2026117000..2026117699`, exact commands/output paths, wall/attempt/replay/stage budgets, selection, and a prospective fresh training/validation plan. Collection reached **350 expert successes / 544 attempts** (218.30 s native elapsed). Official CPU replay accepted **325/350**; all 25 failures remain in the source audit. The first 300 accepted are selected; the extra 25 accepted remain materialized in the additional-source directory. Thus the combined package has exactly **200 retained + 300 new = 500**.

Independent audit: all 500 seeds and action trajectory hashes are unique; all actions retain 7 channels with nonzero rotation and all observations align T+1; replay acceptance and selected source order reconcile. Retained HDF5 group payloads 0–199 match the random-200 package byte-for-byte. Combined HDF5 SHA-256 is `1fe27c25346d1117442d14bdadaf6037717943ea0b86090bcece2556ee008044`, JSON `097bd41280296444db05df6a6030ac2f86bc9ccae799545fdd43e40d231dbf39`, and training coverage `ad7011bdf53f60fa4ac8dc61dd5ef281a456bfa7f30206976ccb66f568d7f3c3`. Overall target-yaw range is `1.178476–1.962855 rad`, 5-bin counts `116/121/106/82/75`; new-300 counts are `74/72/67/46/41`. Old random-100 and fixed-1000 source hashes are unchanged. Detailed independent checks, all hashes and geometry ranges are in `independent_dataset_audit.json`; exact commands/logs/times and bounded deviations are in `execution_record.json` and the experiment directory.

The initial collector wrapper failed before starting because Python's default `/bin/sh` does not implement `source`; direct bash execution then completed the single collector run. Coverage assembly also revealed that the existing writer did not accept a previously combined dataset as the retained source. A focused regression was added: expected red captured this `collection_report.json` lookup failure, then a minimal public writer branch composed existing `training_coverage.json` rows with the new primitive source, checking seed order. It passed green. The required exact eight-file regression passed **64 tests in 11.03 s** after the change (the same suite passed **63 tests in 11.14 s** before registration). No ManiSkill source was changed.

This round performed no optimizer updates, policy/native test rollouts, checkpoint selection, or Plug work. The prospective next stage is a separately staged fresh 30k-update Base+zero-control run with the same Base EMA, 512 random-500 / 512 fixed-1000 windows, and validation-only selection as frozen in the registration; it is not run or claimed as approved outcome. The historical formal gate remains failed (+12pp random in the round-37 nonblind retest); the next training/validation evidence will be reviewed before any final test decision.

## Goal-loop round 37 — one authorized corrective nonblind formal retest

The targeted diagnosis/correction path is now complete. Round 34's fresh
validation-motion probe found no live-hint or controller-frame implementation
defect; its traces instead showed low/variable requested motion and limited
realized motion under contact. Following the separately authorized random-200
training correction in round 36, the single permitted corrective retest ran all
six fresh 100-episode cohorts in the frozen order. Because the random-test
outcomes were historically viewed, this is explicitly a **nonblind corrective
retest**, not a new blind test. Fixed-retention also has prior Base exposure.

Registration, exact model/data/catalog/assignment identities, finite source
closure, commands, UTC and monotonic timing, logs, episode records and results
are in `outputs/exp/20261117_230000_peg_orientation_corrective_retest/`.
Candidate: round-36 update-30,000 EMA SHA-256
`6fa85080d06025ba9283e062e411acfc7fa7e8cc8e936f970180400b8677f1e7`; unchanged
Base EMA SHA-256 `067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea`.
The same frozen seed-1703 shuffled test assignment was reused exactly, including
its two self-matches and two identical target-yaw matches; it was not
rerandomized. Every profile used paired seeds `1+i`, native cap and TimeLimit
300, DDPM-100 and full native 7D actions. Stage geometry was opt-in. Each
cohort's catalog scene 0 video was captured in that same pass; these videos do
not retroactively satisfy the old round-32 video record.

| Cohort | Success | Controls | Contact-loss transitions | `not_reached` |
|---|---:|---:|---:|---:|
| Random Base | 3/100 | 29,551 | 61 | 100/100 |
| Random correct | 15/100 | 27,565 | 29 | 100/100 |
| Random fixed-standard | 3/100 | 29,703 | 38 | 100/100 |
| Random shuffled | 7/100 | 29,047 | 31 | 100/100 |
| Fixed-retention Base | 52/100 | 22,097 | 36 | 100/100 |
| Fixed-retention correct | 49/100 | 22,455 | 10 | 100/100 |

Paired random Base-only/candidate-only/both/neither is `2/14/1/83`, a **+12pp**
correct-hint gain, below the unchanged +20pp requirement. Paired fixed
Base-only/candidate-only/both/neither is `10/7/42/41`, a **−3pp** difference,
within the allowed maximum −5pp loss. Correct vs fixed-standard is +12pp and
correct vs shuffled +8pp, descriptive only. Random Base reproduces 3/100 and
fixed Base reproduces 52/100. Historical prior-candidate +18pp random / +6pp
fixed results remain immutable and are not pooled. Five equal-width target-yaw
bins and per-scene paired rows are in `results.json`.

All six commands exited 0 once; 600 episodes and 160,418 native controls took
6,884.48 s. Per-cohort times were 947.21, 1,327.59, 1,436.17, 1,380.82, 717.49,
and 1,075.20 s in registered order; no cleanup or retry was needed. The finite
553-file local source closure and all registered inputs stayed stable within
each cohort; this is not a claim of a fully immutable Python environment.
`not_reached` remains only a proximity-trigger status, not physical
non-insertion. Yaw diagnostics are not a hard gate.

**Final decision: Peg formal gate not passed; Ticket 06 is resolved with a
negative outcome and Plug remains blocked.** No additional training, data,
tuning, checkpoint selection, or Plug work is authorized by this result.

## Goal-loop round 36 — random-200 corrective training and validation

The authorized fresh correction completed in
`outputs/exp/20261117_211300_peg_orientation_random200_correction/`; six
checkpoints are in the matching `outputs/ckpt/` directory. Training restarted
from the unchanged Base update-100,000 EMA (`067a4f5c…b62beea`) with a fresh
zero-control branch, seed 1, exactly 30,000 updates, batch 1,024 and 512 random-
200 / 512 fixed-1,000 windows per batch. The six EMA checkpoints were each
evaluated once on the two frozen validation-50 cohorts, checkpoint order 5k to
30k and random then fixed, seed `1+i`, correct live hint and 300 native-control
cap. No final-test, fixed-retention, retest, new data, or Plug run occurred.

| Update | Random validation | Fixed validation | Fixed feasible (≥34) |
|---:|---:|---:|:---:|
| 5,000 | 19/50 | 31/50 | no |
| 10,000 | 21/50 | 33/50 | no |
| 15,000 | 17/50 | 37/50 | yes |
| 20,000 | 16/50 | 35/50 | yes |
| 25,000 | 14/50 | 33/50 | no |
| 30,000 | **20/50** | **34/50** | yes |

Frozen rule: fixed ≥ Base reference 36−2, then maximize random successes, ties
latest; no fallback. The selected update-30,000 EMA is
`outputs/ckpt/20261117_211300_peg_orientation_random200_correction/update_30000.pt`,
SHA-256 `6fa85080d06025ba9283e062e411acfc7fa7e8cc8e936f970180400b8677f1e7`.
Against the reused Base validation 4/50 random and 36/50 fixed, paired counts
are random Base-only/candidate-only/both/neither `0/16/4/30` (+16/50) and fixed
`5/3/31/11` (−2/50). Against the prior mixed random-100/fixed-1,000 candidate
(15/50, 38/50), this candidate's aligned paired deltas are +5/50 random and
−4/50 fixed. The prior Base/candidate scores are validation-only historical
references, not reruns.

Actual native rollout total was **131,670 controls / 600 episodes**; measured
rollout time 6,154.67 s, training time 931.41 s, and foreground stage span
7,334.96 s, all within their frozen budgets. All 13 commands exited 0 once;
per-stage source, data, checkpoint and validation-catalog identities reconciled.
The prior formal result remains 21/100 vs 3/100 random (+18pp, below +20pp) and
58/100 vs 52/100 fixed (+6pp, within the maximum-loss boundary). It is unchanged;
this validation result does not formally qualify the new candidate. All
diagnostics were `not_reached`, a trigger-only status; no yaw error is claimed.
Formal retest and retention are not included in this round, and any later retest
must disclose prior random-test exposure. Full registration, execution ledger,
checkpoint identities, pair tables and results are in the experiment directory;
`candidate_freeze.json` records the proposed next candidate without running it.

The previous round-35 200-demo data-only record below remains historical. This
training correction expands the training pool from 100 to 200 while keeping the
fixed pool, batch mixture, initialization, seed, update count, and schedule fixed;
it is an informative comparison, not proof that data quantity alone caused the
change. Total experiment budget also includes the prior runs documented below.

## Goal-loop round 35 — corrective random-data expansion (data-only)

The selected 100→200 correction is complete. The combined training package is
`demos/PegInsertionSide-orientation-random-200/`: the previous 100 trajectories
are retained first and byte-identical in HDF5, followed by the first 100
officially replay-accepted trajectories in source order. No policy-outcome
selection, training, checkpoint selection, final-scene access, or Plug work was
performed. Fixed training data remains unchanged.

Before collection, `outputs/exp/20261117_204638_peg_orientation_random200_data/preregistration.json`
froze source identities, the `2026116000..2026116239` candidate interval, exact
commands/output paths, 120 expert-success / 240-attempt / 1,800-s collector
limits, replay and stage limits, and the future fresh-training plan. The range
was disjoint from inspected prior Peg selected/attempt seeds, validation/test/
retention catalogs, and the fixed-training reserve ending at `2026115999`.
The collector reached **120 expert successes in 184 attempts** (70.61 s native
collector elapsed; 74.02 s measured driver elapsed). Official CPU replay accepted
**113/120**; seven native replay failures were retained in the replay audit and
excluded. `--materialize-all` preserved all 113 valid replays in the additional
source. The combined package selected exactly the first 100 accepted rows; the
remaining valid rows stay in their source package.

The new source is
`demos/PegInsertionSide-orientation-random-additional100_20261117_204638/`.
The 200-row package hashes are HDF5
`1e662966292e20549039fca8a65ad584a53b28be758e089a37ac2cc3cb8f1c96`, JSON
`0672c1b531b2fdfee0d03353098beb98ca83bc933eecbb135fcac73d567234c3`, and
coverage `b95a154a04a8a72c1382871d58836445c9699b20602d95e4cd8a36a377738027`.
Coverage reports **100 retained + 100 new**, target-yaw range
`1.183618–1.962855 rad`, and five-bin counts `42/49/39/36/34`. Independent audit
verified old HDF5 groups 0–99 byte-for-byte, 200 unique seeds and action hashes,
7D actions, T/T+1 state/observation/hint alignment and restore audits, valid
90°±22.5° target range, and unchanged preregistered old random/fixed data hashes.

Transparency: the collection preregistration initially omitted the exact
independent-auditor path/command and the collector's transitive ManiSkill source
hash list. Actual outer collection/replay wrappers were 2,050/750 s instead of
the registered 2,000/700 s (actual elapsed 74/134 s; no limit approached). The
post-collection `data_audit_registration_addendum.json` discloses these
deviations, freezes the auditor before its run, and records the `PROVENANCE_FILES`
hashes (the sources were unmodified); it is not represented as pre-collection
preregistration. Complete
timings, command/exit records, logs, audit report, and artifact identities are
in the experiment directory's `execution_record.json` and
`independent_dataset_audit.json`. The full required eight-file regression
(including failure analysis) passed **61 tests in 10.52 s**, bounded at 180 s
TERM/10 s KILL. No training or checkpoint was produced; only a future fresh
30k-update seed-1 Base + zero-control training plan is recorded for its own
stage. Formal random-test/fixed-retention gates and Plug status are unchanged.

## Round 34 validation motion probe (completed; diagnosis only)

Regression coverage correction after independent review: the historical 49-test
runs below did not establish the required full file list; their exact commands
were not retained. The omitted 11 tests correspond to `test_base_dp.py`.
The reviewer reran the original seven-file suite plus the failure-analysis test:
**60/60 passed in 10.75 s**, activated `maniskill`, single-threaded, bounded by
180 s TERM / 10 s KILL. The frozen registration is unchanged.

The authorized follow-up used random-validation catalog indices 6–11 (not the final test), frozen selected update-15,000 EMA, correct live hint, seeds `1+index`, six episodes, ≤300 controls each/1,800 total, zero updates, and a same-pass six-episode video. Registration, complete traces, analysis, video, and post-run source/input audit are in `outputs/exp/20261117_201647_peg_orientation_validation_motion_probe/`. Native result: 2/6 success, 1,472 controls; all six grasped, with two contact-loss transitions in scene 11.

All 187 actual sampler-call hints match the live peg-yaw/target-yaw recomputation (maximum component error `1.97e-7`); native observation histories also match. Actual controller config in every episode is `root_translation:root_aligned_body_rotation`, delta mode, positional bounds `[-0.1,0.1]m`. The analysis separately measures normalized action, scaled root/world/hole-frame request, and realized TCP/head motion. Requested hole-axis x and realized TCP x are positively associated (`r=.69` overall, `.50` while grasped), but this six-scene descriptive association is not a controller tracking metric. Four failures terminate with head x −0.112 to −0.132m; both successes pass official insertion near −0.010m. Late commands mix low/negative requests and positive requests that realize little motion under contact. **No controller-frame mapping or hint bug/correction is supported.**

The pre-existing trace field `commanded_translation_in_hole` is an instrumentation caveat: it transforms raw normalized action without native positional scaling. The post-run analysis does not rely on it; it recomputes the physical request from captured native bounds and quaternions. It does not affect the action passed to the controller and is not evidence of a policy/controller mapping bug. Source remained frozen after registration. The 1,472-frame video is diagnostic validation material, not the missing formal preregistered scene-0 video. Native exit was 0 under 900s TERM/20s KILL. Exact start/elapsed was missed; preregistration-to-evaluator completion bounds native execution below 176s and preregistration-to-final analysis/audit below 454s (<1,800s aggregate). The seven-file Peg suite plus failure-analysis test passed 49/49 pre-registration and post-run (9.14s / 9.15s). Formal random +18pp remains below +20pp; fixed +6pp remains within the loss bound. Ticket 06 is incomplete; Plug remains blocked.

The next corrective method slice requires a parent decision. A possible 100→200 random-training-demo expansion is within the existing 300-demo cap, but needs an explicit update/checkpoint-selection plan and validation-only selection; this probe does not authorize training.

## Round 33 retained formal-trace diagnosis (offline only)

Continuation amendment: the user's existing instruction authorizes the next
bounded probe on random-validation indices 6–11, seeds `1+i`, six episodes,
at most 300 controls each (1,800 total), and zero optimizer updates. This
supersedes the separate-authorization wording below and the original analysis
plan's test-index 0–5 proposal; no final-test scene will be used for this probe.

See `outputs/exp/20261117_164500_peg_orientation_failure_diagnosis/analysis_plan.json`, `summary.md`, and `full_analysis.json`; rerun with `python -m examples.baselines.constrain.peg_orientation_failure_analysis --output-dir <new-timestamped-output-dir>` after activating `maniskill`. The analysis uses the source-defined official head frame and insertion predicate, all pre/post/terminal samples, and the recorded live wrapped world-Z peg-axis minus hole-axis yaw. It makes no new threshold or phase inference from center XY 0.10 m. Random-correct failure closest-head medians are 0.1044 m 3D, x −0.1044 m, lateral yz 0.0092 m; success medians are 0.0102 m, −0.0097 m, and 0.0030 m. This points to approach/head geometry for a validation probe but does not prove cause. Contact losses (31 correct vs 61 Base on random; 10 vs 36 fixed) also remain descriptive. Only first-two and final hint samples were retained; no action/TCP/force trace exists, so no command mapping or hint-cadence correction is supported. Formal random +18pp still fails +20pp and fixed +6pp remains within the allowed loss. The proposed next probe is validation indices 6–11 (indices 0–5 were already used), correct-hint frozen checkpoint, 6 episodes / ≤300 controls each / zero updates, with per-control action/TCP/head/hint capture; any native execution requires separate authorization.

## Round 30 forward Base-gate amendment and selected-candidate freeze (2026-11-17)

The user said **“放宽要求 72可以了 继续推进”**. Narrow interpretation: accept the existing current Base fixed-validation result **36/50 (72%)** as the Base prerequisite for continuation, and waive the former independent fixed-retention **≥80/100** threshold only as a blocker. This does not invent a 72/100 measurement, does not claim the old threshold passed, and does not rewrite earlier history. The unchanged final gates are random-test correct-hint gain **≥20pp** and paired fixed-retention loss **≤5pp**; the final 100-scene fixed comparison remains required. Plug stays blocked until actual Peg gates pass.

The existing replay-trained candidate is selected by its already documented validation-only rule: metadata-complete update-15,000 EMA SHA-256 `522be233a68123e2a4afae882d84cf5e87cab26844054059585d74ffac820898`, from `outputs/ckpt/20261117_003640_peg_orientation_replay_exploratory/metadata_complete/update_15000.pt`. On validation, Base vs candidate is random 4/50 vs 15/50 (+22pp; paired Base-only/candidate-only/both/neither 1/12/3/34), fixed 36/50 vs 38/50 (candidate gain +4pp; paired 3/5/33/9). Candidate selection proof and hashes are retained in `outputs/exp/20261117_151022_peg_orientation_gate_preparation/`.

Training caveat: the selected run used 100 random demos **plus** the existing fixed-1000 dataset, 30,000 updates, seed 1, batch 1024 with 512/512 windows, from Base EMA `067a4f5c...b62beea`. It is neither pure 100-random-demo training nor a 1,000-demo same-information fine-tune control. Stop random data expansion at 100 based on the existing validation target and hint diagnostic; this is not a guarantee of final qualification.

Round 30 performed only read-only identity/selection auditing and planning. It did not open final scene contents or outcomes, train, collect data, replay, or call native evaluation. Existing diagnostic `not_reached` remains proximity-trigger-only, never a claim of no physical insertion. The frozen shuffled-hint seed 1703 must be applied to random **test** target yaws to create a fresh mapping before scene access; do not reuse the validation-50 assignment. Fixed-retention scenes have historical exposure from an earlier Base checkpoint; disclose it. Current selected weights have no retained outcome record on that fixed catalog; final random-test outcomes remain unseen.

Ticket 02 is accepted as the revised continuation prerequisite (36/50 validation, not 72/100). Ticket 04 and Ticket 05 are complete under the bounded revised acceptance summary; no numerical yaw availability is required and frozen `not_reached` remains a valid trigger-only diagnostic. Ticket 05 overfit assessment remains unknown, not a required pass criterion. Ticket 06 is ready for formal preregistration but not completed; the final ≥20pp random-test gain and ≤5pp paired fixed-retention loss remain mandatory. Plug remains blocked.

## Optional stage-geometry tracing correction (2026-11-17)

The evaluator's native geometry snapshot is opt-in via `record_stage_geometry`;
the Constrain CLI enables it only with `--record-stage-geometry`. Stage geometry
implies the phase trace because pre/post control rows are its carrier. With the
flag off, evaluator output and policy/action sampling remain on the prior path.
The trace reports live peg/hole center distance, official peg-head position and
insertion boolean, and wrapped peg-axis-minus-hole-axis yaw. These are descriptive
measurements, not thresholds or success definitions.

Regression extended the existing evaluator fixture to change pose across each
step and cross the ±π yaw representation boundary, checking pre-action,
post-action, and terminal snapshots, including the final control. The first test
run failed because the fixed-pose fixture expectation (0.05 m) no longer matched
the intentionally mutated terminal pose (0.08 m); this was a fixture expectation
mismatch, not evidence of a pre-existing production regression. The corrected
focused test passed. The required seven-file suite passed 58 tests (11.06 s on
the final correction run).

A separate fresh registration and bounded six-scene validation diagnosis is at
`outputs/exp/20261117_153000_peg_orientation_stage_diagnosis/`. Base scored 0/6;
selected Constrain scored 3/6 on the same scenes and per-scene seeds, with 3
Constrain-only outcomes. All 12 episodes acquired grasp; none reached the frozen
center-XY ≤0.10 m trigger. Three Constrain episodes nevertheless achieved
official insertion/success at terminal center distances 0.104–0.116 m. This
reinforces that diagnostic `not_reached` is not equivalent to physical
non-insertion; it does not establish a yaw cause or general performance claim.
Formal statuses remain unchanged: Base unqualified, Ticket 04 incomplete,
Tickets 05/06 and Plug blocked. The previously frozen 0.10 m proximity trigger
is unchanged. The additive `report_correction.json` documents the analysis-only
repair to minimum-measurement aggregation: minima include all pre/post stage
snapshots and XY minima use recorded observation XY measurements, not values
inferred from 3D distance. The terminal yaw in scene 0 is signed −0.039726 rad;
the former prose mislabeled the trajectory minimum 0.000103 rad as terminal.
Native episode/summary records and formal statuses remain unchanged. Corrected
`results.json` SHA-256: `940b003fb8ab4fe091621e7434fc0df5db5fe4ebd558256db0f1aef14661d676`.

## Dynamic-yaw hint intervention validation (2026-11-17; exploratory only)

The selected evaluated update-15,000 EMA was frozen for one preregistered,
paired 50-scene random-validation comparison: correct dynamic hint, fixed
standard 90° target (recomputed from live peg yaw), and an episode-constant
shuffled pseudo-target assigned by a seeded permutation of only the same 50
validation targets. Results were **15/50, 2/50, and 4/50**, respectively.
Correct-vs-fixed outcomes were 2 both / 13 correct-only / 0 fixed-only / 35
neither; correct-vs-shuffled were 2 / 13 / 2 / 33. Correct reproduced its
previous 15/50 score. The assignment has zero self-matches and zero identical
target-yaw matches. All three profiles used native policy seeds `1 + index`,
the same catalog, 300-control cap and exact checkpoint; every cohort's
pre/post checkpoint, catalog, assignment and finite 553-file source identities
matched. Full assignment data, sampler hint samples, per-scene tables, five-bin
counts and execution evidence are at
`outputs/exp/20261117_115044_peg_orientation_hint_interventions/`.

This is descriptive evidence on one checkpoint and one validation cohort, not a
formal test or a general causal/robustness result. All 150 pre-insertion
diagnostics remained `not_reached`; no yaw-performance or physical insertion
phase claim is available. No final test, fixed-retention evaluation, training,
data collection, checkpoint selection, or ManiSkill/official source edit was
performed. Base remains unqualified; Ticket 04 remains formally incomplete and
Tickets 05/06 and Plug remain blocked. See the local tracker README and Ticket
04 Execution for the status boundary.

## Balanced replay exploratory continuation (2026-11-17)

The autonomous follow-up is registered at
`outputs/exp/20261117_003640_peg_orientation_replay_exploratory/preregistration.json`.
It makes one fresh initialization from the current selected Base EMA
(`067a4f5c...b62beea`) and zero Constrain branch, then trains 30,000 updates
with an exact 512 random-100 / 512 fixed-1000 windows per batch. The existing
39d observation, live dynamic relative-yaw hint, native full 7d action, and
2/8/16 policy horizons are unchanged. No demos were added. This comparison
also adds the fixed 1,000-row dataset and halves random windows/update relative
to the prior pilot, so it does not isolate mixture ratio as the only changed
factor.

Fresh Base validation was **4/50 random, 36/50 fixed**. All six checkpoints
were evaluated once on each same frozen validation-50 catalog, seed `1+i`,
300 native/environment controls. Frozen rule: candidate fixed score must be at
least Base fixed score minus 2; among such candidates choose maximum random
score, ties latest. Feasible updates: 15k (15/50 random, 38/50 fixed), 20k
(12/50, 39/50), 25k (12/50, 35/50), 30k (10/50, 34/50). Selection is update
15,000 EMA, evaluated metadata-complete checkpoint SHA-256
`522be233a68123e2a4afae882d84cf5e87cab26844054059585d74ffac820898`; its raw
trained checkpoint SHA-256 is
`6d137172de8dfa4aefbbb1e44e830eb9e7df952515804a93f026347936f01111`.

Selected paired outcomes: random Base-only 1, Constrain-only 12, both 3,
neither 34 (**+11/50, +22 pp**); fixed Base-only 3, Constrain-only 5, both 33,
neither 9 (**+2/50, +4 pp**). The predeclared exploratory target was met, but
that is not formal qualification. The prior Base remains unqualified at 36/50;
its independent 100-scene fixed-retention test was not run. Ticket 04 remains
formally incomplete, Tickets 05/06 and Plug remain blocked. The earlier pilot's
partial provenance and episode rows were not changed or used as a baseline.

All 14 evaluation calls exited zero; aggregate rollout wall time was 7,387.86 s
under the 14,400 s cap. Actual per-episode control counts were positive and at
most 300; completed action-chunk sums reconcile. All 700 yaw diagnostics were
`not_reached`, which only means the frozen 0.10 m diagnostic trigger did not
fire; do not infer lack of physical insertion or report yaw-error performance.
No final-random or fixed-retention scene was accessed. Exact records, all six
checkpoint identities, pair tables, control totals, source/catalog/checkpoint
pre/post hashes, commands and bounded failures are in the experiment directory
`results.json`, `validation_commands.json`, `checkpoint_metadata_enrichment.json`
and `results.md`.

The model tensors in the raw checkpoints were retained unchanged. Their
metadata-complete copies were made after training to embed dataset paths,
coverage identities, per-episode seeds and sampler/RNG protocol; tensor hashes
match before/after. Runtime package versions were captured after training but
before validation, not in the pre-optimization registration. The pre-run source
set covers seven direct local runtime/test files, not the full transitive
ManiSkill and official Diffusion Policy source trees. Direct registered source
hashes were verified unchanged before/after every validation call; do not
present package or dependency identities as contemporaneously preregistered.

## Exploratory orientation-conditioned ConstrainNet (2026-11-16)

User-approved narrow exception: experiment from current Base update 100,000
(EMA SHA-256 `067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea`)
despite its **unqualified** 36/50 fixed-validation score. This does not change
the formal Base criterion (independent fixed retention >=80/100), and does not
complete the Base prerequisite, Ticket 04, or unlock Tickets 05/06/Plug.
Original thresholds and prior blocked history remain in force. Contract and
execution evidence: `.scratch/orientation-conditioned-insertion/README.md`,
Tickets 02/04, and `outputs/exp/20261116_164019_peg_orientation_constrain_exploratory/`.

The Peg-specific experiment reuses `ConstrainDenoiser` with 39d global
conditioning (2x39), full native 7d actions, and a 2d `[sin, cos]` current
world-Z yaw error. Training computes the current hint from the aligned latest
observation in each 2/16 window; inference refreshes `task.get_orientation_hint()`
before each native policy sample and repeats that live condition across the
16-action prediction horizon. It does not append target yaw to Base observation
or drop rotation actions. The frozen Base was bit-exact across all 30,000
updates; initialization and `control_enabled=False` match the Base denoiser,
and same-RNG full-sampler outputs match. Details, checkpoint identities, pair
tables, bins, and honest limits are recorded in `results.json`.

Exploratory outcomes: selected random validation 16/50 vs Base 4/50 (+24 pp,
12 Constrain-only and 0 Base-only); selected fixed validation 21/50 vs Base
36/50 (−30 pp, 18 Base-only and 3 Constrain-only). The yaw diagnostic was
unavailable: all rows report `not_reached`, which means the frozen 0.10 m
diagnostic trigger did not fire, **not** that physical insertion was never
reached. Some native-success episodes ended with peg-center XY distance above
0.10 m. Do not infer yaw performance or alter that trigger post hoc. This is a
feasibility result with a large fixed-validation regression, not a formal method
pass; no final test or fixed-retention scenes were accessed.

### Round 23 acceptance-gap recovery addendum

The original episode JSONs, preregistration, and `results.json` remain unchanged.
The read-only reconciliation is
`outputs/exp/20261116_164019_peg_orientation_constrain_exploratory/recovery_verification_report.json`
(generator `peg_orientation_recovery_audit.py`). It checks all nine retained
50-episode cohorts: indices 0–49, policy seeds 1–50, catalog seeds, actual
controls within the 300 cap, and episode-native outcomes against each saved
summary. The 4/50 and 36/50 Base counts, six random Constrain counts
5/8/13/14/16/14, selected random 16/50, fixed 21/50, and paired transition
tables reconcile without changing their source files.

**Historical provenance remains incomplete for every cohort.** Original
episode headers did not record checkpoint/source/catalog hashes, and the frozen
preregistration lacks pre-execution source hashes. Current checkpoint, data,
catalog, and evaluator hashes in the recovery report identify files as they
exist now; they do not authenticate what a historical process loaded. In
particular, the 5,000-update cohort's summary identity was added after rollout,
so its association to those outcomes is unverified and its checkpoint-selection
contribution is provisional. The retained native counts and selection among
verified retained cohorts agree independently, but the historical selection
claim cannot be fully authenticated. `not_reached` is retained as the recorded
diagnostic status, with yaw unavailable rather than a zero/failed insertion
claim.

Forward only: future evaluator episode headers now capture adapter checkpoint
metadata, catalog-manifest SHA-256, and evaluator-source SHA-256 before the
episode loop. This does not backfill historical records or alter evaluation
behavior.

Audit command (read-only, 180 s timeout, `maniskill` activated):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && timeout --signal=TERM --kill-after=10s 180s python -m examples.baselines.constrain.peg_orientation_recovery_audit
```

It reconciled 9/9 cohorts; report SHA-256 is
`9e08103fec2359f33c4b6dc6523cf2196e856190b87dfb1b6a0fa1387d6a9116`.
Required regression command (180 s timeout) was:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 180s python -m pytest examples/baselines/constrain/tests/test_peg_orientation.py examples/baselines/constrain/tests/test_base_dp.py examples/baselines/constrain/tests/test_peg_orientation_preflight.py examples/baselines/constrain/tests/test_peg_orientation_data.py examples/baselines/constrain/tests/test_peg_orientation_base_dp.py examples/baselines/constrain/tests/test_peg_official_dp.py examples/baselines/constrain/tests/test_peg_orientation_constrain.py -q
```

Result: 55 passed in 11.83 s; `git diff --check` passed. Audit generator SHA
`eee0297ef1945337a17d99d212a775d0b79e976eb97220cb093d0420ba9b6b78`, Peg
Constrain regression SHA
`c93826eda4dedd0d50105771210fe03f873b651d5074c20f8bd0d79f46c8aab5`, and
evaluator source SHA
`3ac4b24ef41d952e28454eea43db116f05ae8fb2d5721a4f74256cf3d786dcbb`.

Registered environments: `PegInsertionSideOrientation-v1` (target yaw fixed at
90 degrees) and `PegInsertionSideRandomOrientation-v1` (uniform target yaw in
90 degrees ±22.5 degrees). They support one CPU simulation environment and
reject `num_envs != 1` during construction. Both use the native `panda_wristcam`
task, reset, state restoration, the native success criterion, and native
`pd_ee_delta_pose` control. The flattened state observation is 39d: Panda
`qpos` 9d + `qvel` 9d; `tcp_pose` 7d; `peg_pose` 7d; `peg_half_size` 3d;
`hole_center` 3d; and `box_hole_radius` 1d, as measured from the runtime
observation space and field shapes. The native 7-channel action is
`[dx, dy, dz, dRx, dRy, dRz, gripper]` in normalized controller coordinates.
The arm controller is 6d (translation and all three rotation-vector channels);
the gripper controller is 1d. No zero-padding rotation special case is applied.

`_get_obs_extra` state schema audit:

| Field | Included | Dimensions | Meaning |
| --- | --- | ---: | --- |
| `tcp_pose` | yes | 7 | robot TCP world position and quaternion |
| `peg_pose` | yes | 7 | current peg world position and quaternion |
| `peg_half_size` | yes | 3 | peg half-length and cross-section geometry |
| `hole_center` | yes | 3 | world-space center of the hole |
| `box_hole_radius` | yes | 1 | hole clearance radius |
| `box_hole_pose` | no | 7 | omitted because quaternion reveals target yaw |
| `goal_pose` | no | 7 | omitted; derives target orientation and peg-head offset |

The base state observation also contains the native robot joint state under
`agent` (`qpos`, plus the native state fields); no target yaw or target-oriented
frame is included. Consumers should take `OBSERVATION_FIELDS` as the agreed
task-field selection and preserve the enclosing native robot state.

Yaw is rotation around world +Z, in radians, period 2π. The peg axis is its
local +X transformed to world; the target axis is the box local +X transformed
to world. `task.get_orientation_hint()` reads the current peg and box world
poses at call time, projects each local +X axis into world XY, derives both
yaws, and returns `[sin(error), cos(error)]` for wrapped
`peg_yaw - target_yaw`. Call it for each observation; it is deliberately not
inserted into the base observation. `build_orientation_hint(peg_yaw,
target_yaw)` remains the angle-pair encoding helper. Restoring state restores
the box pose, and the hint then reflects the restored peg/target pair.

Focused verification:

```sh
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
python -m pytest examples/baselines/constrain/tests/test_peg_orientation.py -q
```

## Ticket 01 collection and replay preflight

`peg_orientation_preflight.py` collects from the official
`solvePegInsertionSide` expert in native `pd_joint_pos`, then converts the
recorded joint targets with ManiSkill's official
`from_pd_joint_pos_to_ee` path through `replay_trajectory --target-control-mode
pd_ee_delta_pose`. Replay acceptance is the native `info["success"]` result.
The retained seven action channels are `[dx, dy, dz, dRx, dRy, dRz, gripper]`;
rotation is not zero-filled. `replay_audit.json` recomputes the `[sin, cos]` hint
from each of the T+1 restored states, audits T+1 observations and state/action
alignment, and checks hole-center/hint restoration at the first and last state.

MPLib 0.2.1's binding differs from the legacy shared motion-planner caller at
`set_base_pose` and `plan_screw`. The experiment process installs a narrow
adapter for the seven-vector/Pose arguments and drops the unsupported
`use_point_cloud=False` keyword. No ManiSkill source is changed. Smoke showed one
random-yaw expert success whose converted actions did not meet native success;
it remains explicitly recorded as a rejected candidate, not a valid demo.

### Reproduction

Run from the repository root. Collection has a 100-attempt ceiling and 1,500 s
internal wall limit; each command below has an outer 1,700 s limit. Conversion
and replay has a 700 s outer limit. The retained datasets are immutable; use new
explicit dataset and timestamped experiment paths to repeat the run.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
timeout --signal=TERM --kill-after=20s 1700s python -m examples.baselines.constrain.peg_orientation_preflight collect --distribution fixed --output-dir demos/PegInsertionSide-orientation-fixed --target-successes 80 --max-attempts 100 --wall-seconds 1500 --seed 2026112000
timeout --signal=TERM --kill-after=20s 1700s python -m examples.baselines.constrain.peg_orientation_preflight collect --distribution random --output-dir demos/PegInsertionSide-orientation-random --target-successes 80 --max-attempts 100 --wall-seconds 1500 --seed 2026112000
timeout --signal=TERM --kill-after=20s 700s python -m examples.baselines.constrain.peg_orientation_preflight replay --dataset-dir demos/PegInsertionSide-orientation-fixed --count 65 --min-successes 50
timeout --signal=TERM --kill-after=20s 700s python -m examples.baselines.constrain.peg_orientation_preflight replay --dataset-dir demos/PegInsertionSide-orientation-random --count 63 --min-successes 50
timeout --signal=TERM --kill-after=5s 30s python -m examples.baselines.constrain.peg_orientation_preflight summarize --fixed-dir demos/PegInsertionSide-orientation-fixed --random-dir demos/PegInsertionSide-orientation-random --output outputs/exp/20261111_164512_peg_orientation_preflight/summary.json
```

Each dataset contains exactly 50 replay-verified episodes at
`trajectory.state.pd_ee_delta_pose.physx_cpu.h5` plus its matching JSON,
`dataset_manifest.json`, the joint-space source HDF5/JSON, all per-attempt
records, collection report, and conversion/replay audit. The timestamped
experiment directory contains the commands' logs and paired coverage summary.

### Results (2026-11-11)

| Distribution | Expert successes / attempts | Native-success converted replays / candidates | Materialized |
|---|---:|---:|---:|
| Fixed target yaw 90° | 65/100 | 62/65 | 50 |
| Random target yaw 90° ±22.5° | 63/100 | 62/63 | 50 |

The expert-source target was 80 candidates per mode; the 100-attempt budget
closed collection at 65 fixed and 63 random expert successes. This is recorded
as bounded candidate shortfall; both candidate sets still yielded at least 50
valid converted/replayed episodes. Three fixed and one random conversion replay
failures are retained by seed in the audit. Across the paired 100-seed reset
catalog, maximum hole-center difference between fixed and random target yaw was
`1.49e-8 m`, and initial peg yaws matched exactly. Fixed target yaw remained
90°; sampled random target yaw ranged 1.18545–1.95233 rad, with 5-bin counts
17/18/24/20/21. Initial peg yaw ranged 0.53053–2.58749 rad. All 50 selected
episodes in both datasets have 7D actions with nonzero rotation controls,
T+1 observation/state/hint alignment, and passing first/final state restoration
checks. Full per-episode lengths, geometries, source hashes, reset seeds, and
replay failures are in the dataset manifests and
`outputs/exp/20261111_164512_peg_orientation_preflight/summary.json`.

## Ticket 02 fixed-yaw Base DP data prerequisite (2026-11-11)

The training-ready package is `demos/PegInsertionSide-orientation-fixed-1000/`:
exactly 1,000 official replay-verified trajectories, comprising the unchanged
original 50 plus 950 new rows. The original HDF5 remains SHA-256
`715f76393b661f70ba8958b9c87ace0f0a23192b216c6d4ebf1c1407115d1606`; combined
HDF5 SHA-256 is
`54b4a0dd7a62ef9cdd61ca5d5545051e3c769a3a179f0f7f965fddc8fe453631`, JSON
`cdf61ef88c7d5bf1acfa57b329a4d972af29c089a01ada7957d2f4ccbef04800`, and
coverage `cfe75bf570ec97c67409329424993d0c5e21c3431b021771a223f3e64980ffee`.
The manifest preserves each source dataset's trajectory/JSON/manifest hashes;
`training_coverage.json` contains per-seed target/peg yaw and geometry.

New fixed collection used five bounded batches of 200 expert successes, followed
by one 10-success recovery batch: total 1,615 attempts (maximum permitted
3,000). The five batches yielded 1,000 expert successes and 948 accepted
converted replays; recovery yielded 10/10 replay successes. 950 new trajectories
were selected in source order. All 1,000 selected rows have unique seeds and
action trajectory hashes, target yaw exactly 90°, 7D converted pose actions,
T/T+1 action/state/observation/hint alignment, nonzero rotation controls and
passing first/final state-restoration checks. Geometry and source attempt/replay
counts are in the coverage file. Command logs and per-batch candidate files are
retained in `outputs/exp/20261111_174907_peg_orientation_base_data/` and
`demos/PegInsertionSide-orientation-fixed-additional*`.

The original split and scene catalogs remain unchanged. Additive reservation
`outputs/exp/20261111_174907_peg_orientation_base_data/fixed_training_reservation_v2.json`
retains the prior frozen-split SHA-256, the four scene/manifest identities and
reserves `2026112500..2026115999`; collection was capped at 3,000 attempts, so
the candidate-seed attempt ceiling remained inside the reserve. This is a
versioned supplement, not a retroactive preregistration.

Reproduction uses the `maniskill` environment. Collection batch bounds were
350 attempts / 2,200 s (outer 2,400 s), recovery 20 attempts / 240 s (outer
300 s), and official replay outer 700 s per batch. The five fixed expert batch
starts are `2026112500`, `2026112850`, `2026113200`, `2026113550`, and
`2026113900`, each with `--target-successes 200 --max-attempts 350
--wall-seconds 2200`, followed by `--count 200 --min-successes 1
--materialize-all`. Recovery starts at `2026114250`. The final combine/audit
commands use the original fixed source and all six additions with
`--expected-count 1000`. No model training or policy evaluation was performed
in this data-prerequisite slice.

## Ticket 03 adaptation data and frozen scenes (2026-11-11)

### Additional random-yaw training data

The original `demos/PegInsertionSide-orientation-random/` is unchanged: its
50-row trajectory HDF5 remains SHA-256
`ab04a2ccd3d640e1aff40f1e2efd8b59888c26023bee3ba5a5851895bc034940`. The new
`demos/PegInsertionSide-orientation-random-100/` contains the original 50
episode identities followed by 50 newly replay-verified episodes. Its combined
trajectory HDF5 is SHA-256
`c4d6d2c9a8f85ebb39bcb003a987f7bd29baed24ff7a680d7eaf3726b5786f25`; companion
JSON SHA-256 is `fba19d4b69345309c5e859b7d120c6ca0f4aededc9c6655315bc3a9beff89874`.
`dataset_manifest.json` retains each source dataset's HDF5, JSON, and manifest
hashes and records distinct seeds and distinct action-trajectory hashes.
`training_coverage.json` records all 100 row identities, target and initial peg
yaws, hole centers, peg half-sizes, hole radii, and the 5-bin target-yaw counts.
Across all 100, target yaw spans `1.183618–1.952331 rad` and initial peg yaw
spans `0.523984–2.579496 rad`.

New collection used seeds `2026112100..2026112199`, wholly after Ticket 01's
`2026112000..2026112099`. The first bounded batch collected 50 expert successes
in 77 attempts; official conversion/replay accepted 49/50, rejecting seed
`2026112153` (native success failed after conversion). A separate 5-success
recovery batch used seeds starting at `2026112177`; it produced four native
successful replays and one selected valid episode (seed `2026112178`), while
recording replay rejection `2026112183`. Combined with the original 50, this
produces exactly 100 valid, unique episodes. All retained rows passed the same
official conversion/replay audits for 7D `pd_ee_delta_pose` actions,
nonzero rotation control, T actions versus T+1 state/observation/hint rows, and
first/final restored hint and hole-center checks.

The reserved catalogs are in
`outputs/exp/20261111_172604_peg_orientation_adaptation/frozen_splits.json`:
50 random and 50 fixed validation scenes, 100 random paired test scenes, and
100 independent fixed-yaw retention scenes. Scene seeds are respectively
`2026112200..2249`, `2026112250..2299`, `2026112300..2399`, and
`2026112400..2499`; the random test set is exactly 100. A 1,000-seed future
fixed-training range `2026112500..3499` is reserved disjointly. The explicit
fixed-training range was added to the frozen plan after the random additional
collection had run; those training samples remained in their separately
reserved random-training range, and no validation/test scenes or test outcomes
were used during collection. The frozen artifact preserves this chronology.

Each self-contained scene catalog in that experiment directory contains the
complete initial state, initial 39d observation, seed, target and initial peg
yaw, peg/hole geometry, stable scene-state identity, and HDF5/manifest hashes.
`PegSceneCatalog.load` verifies those identities, `restore_peg_scene` restores
and observation-checks one scene without a policy/model, and
`build_hint_for_mode` provides correct, fixed-standard, and episode-constant
shuffled targets. `hint_interventions.json` freezes shuffled test-target
permutation seed `1703` before any test rollout. Correct hints use live peg yaw
minus actual target yaw; fixed-standard hints recompute against 90°; shuffled
hints use one assigned pseudo-target yaw for the entire episode. All are
`[sin(error), cos(error)]` in the existing world-Z convention.

`preinsertion_yaw_diagnostic` samples the first pre-action state whose peg-center
XY distance to hole-center XY is at most `0.10 m`, records the step index and
wrapped yaw error, and records terminal native outcome/failure type if this
trigger was never reached. It is auxiliary only; native `info.success` remains
the primary metric. No model training or policy rollout was started in Ticket
03; test scenes remain unused for tuning.

### Reproduction commands

Each finite run used the `maniskill` conda environment and explicit artifact
paths. Collection is bounded by 100 attempts / 1,500 s (outer timeout 1,800 s);
the recovery collection is bounded by 10 attempts / 240 s (outer timeout 300
s); replay commands have 700 s or 300 s outer timeouts. The retained logs,
including the initial 49/50 replay shortfall, are in the timestamped experiment
directory.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
timeout --signal=TERM --kill-after=20s 1800s python -m examples.baselines.constrain.peg_orientation_preflight collect --distribution random --output-dir demos/PegInsertionSide-orientation-random-additional50 --target-successes 50 --max-attempts 100 --wall-seconds 1500 --seed 2026112100
timeout --signal=TERM --kill-after=20s 700s python -m examples.baselines.constrain.peg_orientation_preflight replay --dataset-dir demos/PegInsertionSide-orientation-random-additional50 --count 50 --min-successes 49
timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.peg_orientation_preflight collect --distribution random --output-dir demos/PegInsertionSide-orientation-random-recovery --target-successes 5 --max-attempts 10 --wall-seconds 240 --seed 2026112177
timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.peg_orientation_preflight replay --dataset-dir demos/PegInsertionSide-orientation-random-recovery --count 5 --min-successes 1
timeout --signal=TERM --kill-after=5s 30s python -m examples.baselines.constrain.peg_orientation_data combine-training --source-dirs demos/PegInsertionSide-orientation-random demos/PegInsertionSide-orientation-random-additional50 demos/PegInsertionSide-orientation-random-recovery --output-dir demos/PegInsertionSide-orientation-random-100
timeout --signal=TERM --kill-after=5s 30s python -m examples.baselines.constrain.peg_orientation_data audit-training --source-dirs demos/PegInsertionSide-orientation-random demos/PegInsertionSide-orientation-random-additional50 demos/PegInsertionSide-orientation-random-recovery --output-dir demos/PegInsertionSide-orientation-random-100
timeout --signal=TERM --kill-after=20s 1800s python -m examples.baselines.constrain.peg_orientation_data build-scenes --split random_test --output-dir outputs/exp/20261111_172604_peg_orientation_adaptation/random_test_scenes
timeout --signal=TERM --kill-after=20s 1800s python -m examples.baselines.constrain.peg_orientation_data build-scenes --split validation_random --output-dir outputs/exp/20261111_172604_peg_orientation_adaptation/validation_random_scenes
timeout --signal=TERM --kill-after=20s 1800s python -m examples.baselines.constrain.peg_orientation_data build-scenes --split validation_fixed --output-dir outputs/exp/20261111_172604_peg_orientation_adaptation/validation_fixed_scenes
timeout --signal=TERM --kill-after=20s 1800s python -m examples.baselines.constrain.peg_orientation_data build-scenes --split fixed_retention --output-dir outputs/exp/20261111_172604_peg_orientation_adaptation/fixed_retention_scenes
timeout --signal=TERM --kill-after=5s 30s python -m examples.baselines.constrain.peg_orientation_data freeze-hints --scenes outputs/exp/20261111_172604_peg_orientation_adaptation/random_test_scenes --output outputs/exp/20261111_172604_peg_orientation_adaptation/hint_interventions.json --seed 1703
```

The random-test, validation-random, validation-fixed, and fixed-retention scene
HDF5 hashes are respectively `ee659401e16a9533b83843a5147877d2881d0b5eae15f0654c1542a844eef9c7`,
`cdc3f1ee5cc7fc36f4884158c21c86fd238037e87e07b64bcafd427af3ae5d89`,
`bfd31085fb9ba2cc283c75b3c9d89342defab6ed06fb42452b493db10b894a37`, and
`81d9ae6d15a4aa0e87cb61cc9e40b35a27d41e2f772ded34d669a0b84759ffb3`. Each
catalog checksum and all 300 scene states were verified; a random-test scene
was restored through the public model-independent API with exact initial
observation agreement. The final data-prerequisite focused command passed **25
tests** under `maniskill`; no training was run in that data slice.

## Ticket 02 BaseDP training and gate (2026-11-11)

`peg_orientation_base_dp.py` reuses the official ManiSkill Diffusion Policy
`Agent` and its 2-frame observation history, 16-step prediction horizon,
8-action execution horizon, 100-step epsilon DDPM scheduler, squared-cosine
schedule, 4.37M-parameter U-Net, AdamW/cosine warmup and EMA. It trains from
scratch on all 1,000 demos with the target-hidden 39d state and complete native
7d `pd_ee_delta_pose` actions. Observation and action normalization are identity
(the native state/action values are already in the controller/data contract);
rotation channels are neither dropped nor zero-padded.

The preregistration was saved before optimizer updates at
`outputs/exp/20261111_184900_peg_orientation_base/preregistration.json`. It
freezes seed `2026111102`, 30,000 optimizer updates, checkpoints every 5,000,
the fixed-validation-50 catalog and checkpoint selection rule (maximum native
successes, ties to latest update), and forbids retention-set checkpoint
selection. Training used CUDA and took 375.23 seconds. Data HDF5 SHA-256 is
`54b4a0dd7a62ef9cdd61ca5d5545051e3c769a3a179f0f7f965fddc8fe453631`; the
companion JSON SHA-256 is
`cdf61ef88c7d5bf1acfa57b329a4d972af29c089a01ada7957d2f4ccbef04800`.

All validation checkpoints (5k through 30k) scored 0/50; every validation
episode was classified `preinsertion_not_reached`. The preregistered tie-break
selected update 30,000, EMA checkpoint SHA-256
`db6184ca4a467cc4a2163c82cc27e5f64e7dd17148757d64ec727539751597f4`. Its
one-time independent fixed-retention run scored **0/100**, with all failures
`preinsertion_not_reached` and final peg-hole XY distance 0.2356–0.6980 m
(median 0.4752 m). The run includes all 100 episode records and a representative
rollout video (SHA-256
`6a64fe44da3b4a2183b83b1ac89bafb36c17c82cb8517ddd95e9dabbbcd09189`).

One separate validation-scene diagnostic showed 100 policy steps, peg
displacement 0.2761 m, TCP displacement 0.0255 m, terminal XY distance 0.2288 m,
and mean absolute gripper command 0.9982. The aggregate gripper statistic is
phase-confounded and by itself does not indicate an initial grasp-command error;
the matched-phase follow-up below supersedes that interpretation. The first retention
video attempt touched scene 0 and issued one action before a tensor-to-ImageIO
conversion error stopped it without an outcome. Conversion was fixed before
running the full set; no test outcomes informed model selection or tuning. The
exposure is documented in `selection_and_gate.json`.

The Base is **unqualified** (required ≥80/100). Ticket 02 remains paused below
threshold and Ticket 04 remains blocked; no automatic training-budget extension
or qualified-base claim is made. Reproduction, smoke, regression, and full
command records are in `.scratch/orientation-conditioned-insertion/issues/02-peg-base.md`.

### Round 8 diagnostic correction

The original rollout code retained its first truthy `not_reached` dictionary,
so subsequent proximity states were never sampled. It also sampled after an
action and left terminal native outcome/failure fields unset. Thus the recorded
`preinsertion_not_reached` categories are invalid diagnostic metadata; the
independent native outcomes remain exactly 0/50 validation and 0/100 retention.
The evaluator now samples before each action until the first trigger and
finalizes terminal diagnostics with native outcome/failure information. A
zero-update checkpoint probe found held-out expert-window epsilon MSE 0.00599
(online weights 0.00592; shuffled observations 0.5833). The initial +0.9977
gripper output on four frozen validation states was phase-confounded: on 32
matched training starts the expert also requests +1/open and policy sign agrees
32/32. At 24 first-close training observations, seed 1907 agrees 24/24; across
four pinned inference seeds, gripper signs mismatch on 23/96 pre-close and 30/96
close actions. This establishes sampler-dependent timing at the expert close
boundary as a concrete candidate, but not as the cause of 0% success.

Native replay of successful training episode 0 restored the initial observation
exactly and stayed within `1.49e-4` max observation error through native success
at step 120. The recorded 7D actions reproduce the native success. At its
pre-close state, expert grasp occurred at chunk offset 1; four policy-noise
counterfactual chunks grasped at offsets 0, 2, 2, and 1 (all grasped). Thus
reset/observation ordering and gross control mapping errors are disfavored, while
this one-episode timing variation does not explain the historical 0/100 result.
Detailed measurements and commands are in
`outputs/exp/20261111_203000_peg_orientation_base_diagnosis/followup_diagnosis.json`.
The proposed noise-only follow-up was superseded by the training-horizon audit
below. No optimizer updates were used.

### Round 8 continuation: training-derived evaluation horizon

The fixed training HDF5 has 1,000 trajectories with lengths min99, median145,
p90=183, p95=195, p99=216, max248. Stored native first-success steps have median
134.5 and max236; **973/1000 succeed after step100**, and 999/1000 trajectories
have a `truncated` marker at step100. Episode0 records truncation at100 but
terminates successfully at120; independent replay of its recorded actions
confirms success at120. The previous evaluator inherited this 100-step task cap,
so the historical 0/50 validation was horizon-limited relative to the training
demonstrations.

Before another validation run, the training-only rule `ceil(max trajectory
length / 50) * 50` preregistered a 250-control budget. The evaluator now passes
the selected explicit budget to the native environment, also enforces it against
the action loop (including a partial final chunk), and records a fixed policy
sampling seed. The corrected run used the selected update-30000 EMA checkpoint,
validation-50 catalog, seed1907, and budget250 exactly once: **9/50 native
successes**, 41/50 ended at the action budget. The full preregistration, training
audit, result, and command are under
`outputs/exp/20261111_221500_peg_orientation_base_horizon/`.

The old 0/50 and fixed-retention 0/100 remain historical outcomes; retention was
not rerun. Because the earlier validation did not record a fixed sampling seed,
the 9/50 change cannot be attributed solely to the longer horizon. The horizon
mismatch is nevertheless confirmed by stored training success times and native
action replay. 9/50 is below the independent ≥80/100 qualification gate; Ticket
02 remains paused and Ticket 04 blocked. No training or test/retention evaluation
was run. Any further diagnosis should focus on the remaining 41/50 failures under
the corrected horizon, before a separate training-budget decision.

### Goal-loop round 9: paired extended horizons

With user authorization, the selected update-30000 EMA was evaluated on the same
50 fixed-validation states at 250, 500, and 1,000 native controls. Each scene's
Torch CPU/CUDA policy stream is seeded independently as `1907 + scene_index`
after state restoration, so each horizon has the same per-scene DDPM-noise prefix.
The environment TimeLimit is fixed at 1,000 controls for all three runs, while
the evaluator enforces each registered cap. Scores were **12/50, 13/50, 13/50**;
one new scene succeeded at control 367 and none succeeded after control 500.
At 1,000 controls, 37 failures exhausted the cap and all remained outside the
0.10 m preinsertion trigger (last measured distance median 0.2953 m). This is a
validation-only plateau for this checkpoint/seed, not qualification: fixed
retention was not accessed, Ticket 02 remains paused below 80%, and Ticket 04
remains blocked. The preregistration, complete paired per-scene records, and
provenance are in
`outputs/exp/20261111_210357_peg_orientation_extended_horizon/`; see Ticket 02
Execution for exact commands and artifact hashes.

### Round 10: bounded training-phase diagnosis

The preregistered 12-episode training-only teacher-forced probe records the
first pre-close and peg-hole XY≤0.10 m phases, pinned noise seeds, action-channel
errors, and actual phase geometry. At the preinsertion trigger, mean 8-action
rotation-Z MAE is 0.0427 versus 0.0162 for the first action; other channels and
the full sample are recorded in
`outputs/exp/20261111_214832_peg_orientation_base_diagnosis/training_phase_probe.json`.
This teacher-forced signal motivated a physical follow-up but did not by itself
establish closed-loop causality. Separate environments replayed each episode's
expert actions from reset until native grasp, then branched to expert-8,
policy-8, and policy-2 replanning for eight controls. Eight matched branches
across two training episodes started with real contact and exactly matching
recorded observations; all retained contact. Policy-2 replanning improved
peg-to-hole XY progress in one of three paired comparisons and regressed in the
other two. The preregistered chunk-drift criterion was not met, so no validation
horizon comparison or inference correction was run. The 500-control budget was
reached; a fourth episode stopped before grasp only because the budget was
exhausted, not due to task termination. Full results and hashes are in
`outputs/exp/20261111_214832_peg_orientation_base_diagnosis/closed_loop_differential_summary.json`
and Ticket 02 Execution. The independent full-rollout follow-up is summarized
below.

The six-scene full rollout follow-up then reset both profiles directly to the
stored training initial states (max observation error 0) and executed each with
a 500-control cap. Execute-8 achieved **5/6** native successes versus **1/6** for
replan-2. Both profiles grasped and reached the distance-only XY proximity
trigger in all six, but contact-loss transitions were **1** under execute-8 and
**18** under replan-2. The shorter replanner was earlier to first grasp in only
three of six pairs and did not increase preinsertion-trigger reach. This
moderately supports harm from the combined frequent-replanning strategy on these
training scenes; it does not isolate sampler RNG frequency from observation
cadence or establish the validation plateau's cause. The predeclared validation
gate favored only improvements from replan-2 and was not met, so validation was
not run and no inference correction was made. Full per-control evidence is in
`outputs/exp/20261111_214832_peg_orientation_base_diagnosis/full_rollout_training_summary.json`.
The remaining causal distinction is repeated diffusion-noise resampling versus
fresh-observation replanning at post-grasp states.

### Round 11: corrected validation-only checkpoint re-selection

The original six-way selection was made after all checkpoints scored 0/50 under
the incorrect 100-step environment cap. That ranking is invalidated: 973/1,000
training demonstrations reach native success after step100. Before any new
rollout, a preregistration bound all six unchanged EMA checkpoint hashes, the
fixed validation catalog, current evaluator hash, 500-control cap, 1000-control
environment TimeLimit, per-scene seed `1907 + index`, and native success rule.
The 500-control budget is based only on the already-observed validation plateau
for update30000 (13/50 at both 500 and 1000, with no new success after 500).
This comparison is validation-informed; no test or retention outcomes were used.

All six existing checkpoints were compared on the same 50 scenes at 500 native
controls using unchanged eight-action inference. The round-9 update30000 500-step
rollout was reused only after exact evaluator, checkpoint, catalog, seed, cap and
inference-horizon provenance verification. Results were 0/50 (update5000), 5/50
(10000), 6/50 (15000), 10/50 (20000), 10/50 (25000), and 13/50 (30000). The
selection rule (highest success count, latest update on ties) therefore selects
update30000 EMA, SHA-256
`db6184ca4a467cc4a2163c82cc27e5f64e7dd17148757d64ec727539751597f4`. Its 13/50
validation score remains below the contractual independent 80/100 gate; no
checkpoint reached the 40/50 validation marker. Ticket 02 remains paused below
qualification and Ticket 04 blocked. No fixed-retention/test scene or additional
training budget was used. Full preregistration, logs, per-scene records, hashes,
and selection rationale are under
`outputs/exp/20261111_223000_peg_orientation_base_reselection/` and in Ticket 02
Execution.

### Round 12: validation stage diagnosis

The selected update-30000 policy was diagnosed on fixed-validation indices 0..5
without new training. Index 0 first reproduced the known 500-control official
failure through the unchanged evaluate path. Its pose distance matched the earlier
validation record exactly, and it had already grasped at control 51 with no contact
loss. The observation-slice distance disagreed because slice 35 is the hole radius,
not the hole center. A zero-control restore confirmed the physical hole matches the
catalog hole and starts at index 36. The diagnostic index was corrected, with tests,
before the remaining scenes. Scene 0's pose, grasp, and official-success evidence
was reused; its old observation-slice distances were not.

Indices 1..5, measured with the corrected hole position, also failed officially at
500 controls. All six scenes grasp and keep contact. Peg-hole XY is unchanged until
grasp, then decreases by 0.227–0.387 m, but the closest approach is still
0.142–0.242 m. Five scenes end farther from the hole than that minimum while the
grasp remains. Corrected observation and pose XY agree to within `8.0e-8` m on
indices 1..5. XY≤0.1 was not observed and would not by itself be insertion or axis
alignment. The prior training execute-8 result (5/6 official success, 6/6 grasp)
used different seeds and resets, and its stored XY-proximity labels used the wrong
slice, so it is not a controlled comparison. The base remains unqualified. Full
records are in `outputs/exp/20261112_010814_peg_orientation_validation_phase/`.

### Round 13: scene 0 peg-head window

The round-12 closest approach, scene 0 control 191, was replayed for 200 controls
with the same checkpoint, restore, execute-8 profile, and seed 1907. The pose and
gripper anchors matched the prior 500-control trace exactly, so the run was not
extended. Official success stayed false. The new window records the executed 7D
action and the official peg-head position in the hole frame, not peg-center XY.

At control 191 the head is at hole-frame `[-0.0353, -0.0031, 0.0011]` m. Its y
and z are inside the hole radius; its x is 20 mm short of the official `-0.015`
threshold. The peg axis is 4.8° from the hole axis. Controls 191..199 command
less than 5 mm of root-frame translation each, and the head moves 3.8 mm farther
from insertion. There was no large insertion command, so contact blocking is
untested rather than shown. Rotation direction was not analyzed after controller
mapping; raw action sign is not physical rotation sign. The cause of the
small translation command is unresolved. Ticket 02 remains paused and Ticket 04
blocked. The preregistration and scored window are in
`outputs/exp/20261112_013447_peg_orientation_scene0_insertion_window/`.

### Round 14: scene 0 hole-+X intervention

The authorized arm replayed scene 0 through control 190 with the same checkpoint,
restore, execute-8 profile, and seed 1907. The branch state matched the round-13
window exactly, including the official head at `[-0.0353, -0.0031, 0.0011]` m.
Controls 191..198 then replaced only normalized translation with a 5 mm request
along the original hole +X axis. The stored baseline rotation and gripper were
held. The command maps to root/world +Y and normalized `[0, 0.05, 0]`, inside
the ±0.1 m controller bounds. No pose was teleported.

Official success stayed false. This is a scripted diagnostic, not model
performance. Grasp never dropped. The hole did not move. Official head x advanced
2.2 mm and remains 18.1 mm short of `-0.015`; y and z stayed inside the hole
radius. TCP displacement was 0.22–0.84 mm per control against the 5 mm request,
and the peg center moved with the TCP. The preregistered 1 mm-per-control
insertion prediction is falsified. The request was not transmitted at that
threshold. That does not identify contact, IK, or ordinary controller tracking.
The retained policy over the same 8 controls moved the head 3.1 mm farther from
insertion, so the scripted command reversed that retreat without closing the gap.
Ticket 02 remains paused and Ticket 04 blocked. Records are in
`outputs/exp/20261112_103851_peg_orientation_scene0_hole_x_intervention/`.

### Round 15: expert translation scale from stored demonstrations

This slice does not run the simulator, a policy, or an optimizer. It asks
whether successful fixed-1000 expert steps with normalized translation near
0.05 already move only a fraction of a millimeter, like the round-14
intervention, or whether that intervention is low relative to those experts.
The plan was written before the summary. It uses every `traj_0`..`traj_999`
row, the inclusive band `[0.04, 0.06]`, and no other band.

Stored layout checks passed before the comparison. Each episode has T 7D
actions and T+1 39d observations. Action `t` is paired with TCP position
`observation[t, 18:21]` and `observation[t+1, 18:21]`. Joint position and peg
pose in that observation match the stored articulation and peg actor exactly.
The robot root pose is the stored articulation pose and is constant at
`[-0.615, 0, 0]` with identity quaternion, so a root-frame delta and its world
delta are the same vector here; the formula still multiplies by the stored
quaternion. Native grasp is not stored. A negative gripper channel is only a
request label. The comparable stratum is that negative request, because the
intervention held one. The other stratum is reported and does not reclassify.

The controller scale is the Panda `pd_ee_delta_pose` bound, `±0.1 m`, already
recorded in round 14. Physical request is `0.1 * clip(action[:3], -1, 1)` m.
For this band that is 4–6 mm. Realized motion is the next TCP displacement.
Projection is that displacement along the requested world direction. Quantiles
use `numpy.quantile` linear interpolation. The frozen typicality rule compares
the intervention's largest TCP step, 0.843 mm, with the expert 10th and 90th
percentiles. The 5 mm-to-5 mm assumption is refuted only if the expert median
of realized TCP norm over requested physical norm is below 0.5.

Comparable rows: **8,168** across all **1,000** episodes. Realized TCP norm is
0.090 / 0.927 / 1.468 / 2.332 / 6.826 mm at the minimum, 10th, median, 90th,
and maximum. The median projection along the request is 1.351 mm, so the
shortfall is mostly magnitude, not a large sideways component. The median ratio
is **0.294**; the 90th percentile is 0.453. The broad assumption is therefore
refuted: a 5 mm-scale request does not produce about 5 mm of TCP motion on
these successful expert steps. The intervention maximum, 0.843 mm, is below the
expert 10th percentile, 0.927 mm, so the frozen rule calls it unusual-low. The
gap is 0.084 mm. Expert motion in this band is typically about 1.5 mm, not
submillimeter.

The nonnegative-request stratum has 4,304 rows in 996 episodes, median TCP
1.979 mm and median ratio 0.409. It is not the decision stratum. In this band
the operational close label adds nothing independent: every negative request is
after the first negative request, and every nonnegative request is before it.
Native grasp remains unavailable.

Insertion proximity was computed from stored peg pose, peg half-length, hole
center, hole radius, and box orientation, using the official head offset along
peg-local +X. The frozen window is head x in `[-0.035, -0.015)` m with y and z
inside the stored hole radius. None of the 8,168 comparable rows is in that
window. A separate label check, not a new band, reconstructed every one of the
11,232 stored success states as officially inserted and found 1,652 proximal
states elsewhere. The empty band phase is therefore real. This comparison is
not matched to the intervention's near-hole geometry and does not identify
contact, IK, or tracking. Ticket 02 remains paused and Ticket 04 blocked.
Plan SHA-256 is `97522fc7939313eb18d6cf10321e237e36a3eae2a2fcb9859ad94b493c6b3952`;
result SHA-256 is `7343280bf0d093a84b330511fb4764343c65394da82fa2a2156cc020210f27d7`.

### Round 16: proximal expert requests, not a new band

This slice does not run the simulator, a policy, or an optimizer. It summarizes
the 1,652 insertion-proximal states already found in round 15. The predicate is
unchanged: official head x in `[-0.035, -0.015)` m and y/z inside the stored
hole radius. Officially inserted states, head x at least `-0.015` m with the
same lateral test, stay excluded. The plan was written before the summary.

Action `t` is labeled by observation `t`. Observation `T` has no action and is
not paired with action `T-1`. The reconstruction again marks 1,652 of 148,948
states proximal and all 11,232 stored success post-states inserted. None of the
1,652 proximal states is terminal-only, so the actionable count is also 1,652,
across all 1,000 episodes, at most 5 rows in one episode. None has a normalized
translation in `[0.04, 0.06]`. The band was counted, not used as a filter.
Every selected gripper channel is negative. That is a request label, not a
stored native grasp. On these rows the hole center and box quaternion do not
change, so the head-x difference is in a fixed hole frame.

Stored request L2 is 0.257 / 0.283 / 0.303 / 0.324 / 0.592 at the minimum,
10th, median, 90th, and maximum. The ±0.1 m controller scale makes those
targets about 25.7–59.2 mm. No component exceeds 1, so the scale does not clip
them. The median hole-+X target is 30.3 mm. Following TCP L2 is 3.07 / 11.06 /
11.91 / 12.72 / 14.04 mm, and its projection on the request is almost the same
as the L2, so the shortfall is mostly magnitude. Official head-x advance has
the same shape, median 11.9 mm. Rotation L2 median is 0.0365 in stored
normalized units. `rot_lower=-0.1` flips physical rotation sign, and this norm
is not a realized angle.

The frozen rule compares the median request, 0.303, with the intervention
request 0.05 and the round-13 window maximum 0.0373341420915306. The median is
above 0.05, so the classification is **larger**. The minimum is also above both
anchors, so this is not a median-only crossing. Scene 0 at control 191 had
official head x `-0.035316` m, 0.316 mm outside this window, so the rows are
not that state. Experts also realize about 12 mm of a 30 mm hole-+X target.
A stored target is not a physical guarantee, and the policy was not queried on
these observations. The comparison does not identify contact, IK, tracking, or
the 13/50 plateau. Ticket 02 remains paused and Ticket 04 blocked.
Plan SHA-256 is `7cfb378415f806fff742b1f95465e845eaeec3ee8a47e3f03fbd830383782fc9`;
result SHA-256 is `5024044d7d6000c83bb9dbcabd95f3b1d9d54f58a910924a4b21cd4edb3097ee`.

### Round 17: proximal policy first actions, teacher-forced

This slice does not step the simulator, update an optimizer, or edit the dataset
or checkpoint. It queries the frozen update-30000 EMA once on the same 1,652
actionable proximal observations. The plan was written before inference. The
predicate is unchanged. History is the official 39d pair `observation[t-1],
observation[t]`, with observation 0 repeated at `t=0`. The first executed action
is index 0 of the chunk returned by `PegBaseAdapter.get_action`, which calls
official `Agent.get_action`. Horizons remain 2/16/8. Native 7d scaling is
unchanged; the physical target is the recorded ±0.1 m controller scale, not a
rescaled action and not realized motion. Rotation is not analyzed. Gripper sign
is not read and is not native grasp.

The noise choice, frozen before the run, is seed 1907 once and batch 64, not
row-wise reseeding. There were 26 calls. The last repeats the last real history
12 times and those outputs are discarded. Closed-loop evaluation seeds
`1907 + scene_index` and calls batch-1 `get_action` repeatedly. This batched
profile is not that RNG and is not an ablation of it.

Identity matched the frozen counts: 1,652 actionable rows, 0 terminal-only, all
1,000 episodes, at most 5 rows in one episode. The expert translation quantiles
match round 16, including median 0.3031500560873744. No policy translation
component has absolute value above 1.

Policy translation L2 is 0.152 / 0.277 / 0.302 / 0.327 / 0.404 at the minimum,
10th, median, 90th, and maximum. The median is 0.3017986107882178. The frozen
rule is strict: below 0.0373341420915306 would replicate the low commands;
above 0.05 means that profile is not reproduced; the closed interval between
those anchors, including both endpoints, is inconclusive. The median is above
0.05, so the classification is **low_command_profile_not_reproduced**. The
minimum is also above 0.05. Counts are 0 below the window maximum, 0 at or
below 0.05, 846 below the paired expert norm, 846 at or below it, and 806
above it. There is no tie. The median ratio is 0.998, and the median signed
norm error is -0.000530. The median vector error is 0.00986 normalized, about
0.986 mm as a physical target. Median hole-+X targets are 30.1 mm for the
policy and 30.3 mm for the expert; the median signed difference is -0.045 mm.

This is teacher forcing on stored expert states. It can miss closed-loop
distribution drift. Scene 0 control 191 remains outside this window. The result
does not identify contact, IK, tracking, or the 13/50 plateau, and it does not
qualify the base. Ticket 02 remains paused and Ticket 04 blocked. Plan SHA-256
is `f255f54b8ff29775cf970a1931be68a0e0e041ef972ee080701b600cd0c9fbee`; result
SHA-256 is `6960955dd312d62332d1edb7a9168071cb3ff6fdbf14e3d6207880525c488913`.

### Round 18: scene 0 actual sampler history

This slice is one unchanged policy replay, not an intervention and not a score.
The previously authorized 8 scripted steps stay used. No optimizer update, new
data, checkpoint, seed, cohort, retention scene, test scene, or ManiSkill source
edit. The plan was written before physics. It binds the update-30000 EMA, the
fixed-validation catalog, evaluator SHA-256
`0059b723fd6f182518612e84077c8b9594f53ea1ceea2565c707cae3a4800bf0`, diagnosis
SHA-256 `a3fe81f1332942fa64c9e327e2fdce80cebed224f23843b2a98056044be1b621`, scene
0, seed 1907, execute-8, environment TimeLimit 1000, and a 200-control cap. The
capture window is controls 185..200. The scheduled calls are preaction steps 184
and 192, covering 185..192 and 193..200. No extra `get_action` and no retry.

The mismatch prediction, frozen before the run, is confirmed only if a captured
current history disagrees with native qpos, TCP, peg, or hole center by more than
`1e-5`, the prior slot is not distinct, offline reproduction of the saved history
and pre-call RNG misses the executed chunk, the sampler schedule changes, or the
retained native profile changes. Otherwise the small commands are the official
sampler output on a coherent closed-loop history. That does not prove the state
is outside expert support. Nearest-expert comparison was not run: a new distance
would not be a matched-noise test. Round 17 remains that evidence.

The replay exited 0 inside 600 s. The environment was closed. There were 200
controls, 25 sampler calls, and 8 actions each. Anchors match the round-13
episode exactly: control 1 pose XY `0.46256815990730255`, control 191 pose XY
`0.1419355650506847`, control 191 gripper `-0.9911929368972778`, control 199
pose XY `0.1449914032255529`, first grasp 51, zero contact losses, and initial
observation error `5.960464477539063e-08`. Gripper actions for controls 1..200
and 7d actions for controls 185..199 match at max absolute error 0. Official
heads at those overlapping controls match exactly, including control 191
`[-0.03531639277935028, -0.003050796687602997, 0.0011345744132995605]`. Control
200 was outside the round-13 action window; its gripper matches, and its 7d
action is not claimed against a missing round-13 field.

Both captured histories are shape `[2, 39]`, prior then current. Current versus
native qpos, qvel, TCP, peg, peg half-size, hole radius, and hole center is 0.
Prior differs from current by 0.02935 and 0.02295. The hole pose does not move
between the calls. Neither call is officially inserted. Head in the hole frame
is `[-0.037661, -0.002515, 0.000737]` m before control 185 and
`[-0.035680, -0.002596, 0.002563]` m before control 193. Lateral coordinates
are inside the 0.018055 m radius. The second head x is 0.68 mm outside the
proximal window `[-0.035, -0.015)`. That proximity is not a support proof.

The first chunk, sampled before control 185, has translation L2
`0.1208, 0.1312, 0.1401, 0.1219, 0.1031, 0.0628, 0.0204, 0.0096`. Its first
action is not small. Controls 191 and 192 are later indices of that same chunk.
The second chunk, sampled before control 193, has translation L2
`0.0336, 0.0123, 0.0301, 0.0319, 0.0373, 0.0247, 0.0319, 0.0232`. Its first
action is 0.0336 and its maximum is the retained window maximum
`0.0373341420915306`. Offline reproduction with the frozen loader and the saved
CPU and CUDA RNG matches both chunks at error 0. Sampling noise is therefore the
noise that produced these chunks. It is not, by itself, an explanation of the
gap from round 17, because that run used different histories and a different RNG
profile. No matched-noise test was run.

The frozen mismatch prediction is falsified. No implementation bug was found, so
no fix was applied. The root cause of the small closed-loop request remains
unestablished. Round 17's teacher-forced median on proximal expert states is
0.3017986107882178. These two measurements are not a matched comparison and are
not an out-of-distribution proof. The 13/50 plateau is not explained. The base
remains unqualified. Ticket 02 remains paused and Ticket 04 blocked. The goal is
not complete.

Plan SHA-256 is `a55e8cbb1f2bae09b8973d26b668e28494d55330e51730a7a3924cacb4c149c4`.
Episode SHA-256 is `48a7eb0b3ab5945a1672ffbbb41aac0dbd1803e06bd999b042f38ade00734f08`.
Reproduction SHA-256 is `2c95a168cb570e6876c1b9d1e3a5bea383ce98e012869b29c7691df1e6ca3af6`.
Analysis SHA-256 is `f55f65c288983d07047cc9314e4d4ab92efb9db209af26568d15b414ec2ecd23`.

## Official original PegInsertionSide-v1 baseline (2026-11-12)

This section indexes a separate prerequisite. It does not change the fixed-yaw
contract above and does not qualify Ticket 02 or unblock Ticket 04.

The unmodified official `examples/baselines/diffusion_policy/train.py` trained
native `PegInsertionSide-v1` state Diffusion Policy from scratch for 100,000
optimizer updates. Data are the official motion-planning trajectories replayed
to `state` / `pd_ee_delta_pose` / `physx_cpu`; the official loader used the
first 100 sorted rows. Seed was the resolved `Args` default 1. Tracking was
omitted because no Weights & Biases credentials were present. Checkpoints are
in `outputs/ckpt/20261112_181319_peg_official_dp/`. The official
`success_once` selection is 36/100 at TensorBoard step 85000 (85,001 updates).
That number is not the custom fixed-yaw 80/100 retention result.

## Official-config fixed-yaw Base (2026-11-16)

The first-budget text above, including the 30,000-update run, stays historical.
This resumed run uses the resolved official Peg training hyperparameters on the
existing fixed-yaw 39d task and the verified 1,000-demo set. It is not the
official 43d baseline. Preregistration SHA-256 is
`520985d722495fe35e3cdbe5d937bb4a3171bf43fcabaec61a78b9b37159411a`, written
before optimizer updates, with a null initial checkpoint. Training seed is now
`Args` default 1, not `2026111102`. Batch size is 1024. The minibatch iterator
is the official iteration sampler. Evaluation uses the frozen 50 scenes, native
TimeLimit 300, executed cap 300, execute-8, and inference seed `1 + scene_index`.

Training completed 100,000 updates in 2,076.12 s. Validation scores from 5,000
through 100,000 were 6, 11, 13, 21, 22, 18, 25, 28, 31, 30, 32, 32, 35, 35, 35,
36, 35, 34, 34, and 36 out of 50. Selection keeps update 100,000 at 36/50.
SHA-256 is `067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea`.
Because 36/50 is below 40/50, the fixed-retention 100 was not evaluated again.
This is not an 80/100 qualification. Do not attribute the change from the old
13/50 only to the longer update budget: batch, seed, sampler, and cap also
changed. Ticket 04 remains blocked. Artifacts are under
`outputs/exp/20261116_123901_peg_orientation_base_official_config/`.

### Goal-loop round 21 — evaluator control-count correction

The retained evaluation outcomes and selected checkpoint are unchanged. In the
old per-episode `executed_chunk_lengths` field, each entry recorded the planned
chunk size before execution. Therefore 483/1,000 retained validation records
overstate completed native controls when native success stopped inside a chunk;
their chunk-length sums are not authoritative. `executed_controls` remains the
actual completed-control count used by the scores and checkpoint selection.
Current evaluator source records completed actions per chunk. This is an
accounting-only correction: no physics replay, training, or evaluation rerun was
performed, and no historical episode record or checkpoint was rewritten. Details,
the source hash, and regression log are in Ticket 02 Execution and
`outputs/exp/20261116_123901_peg_orientation_base_official_config/round21_accounting_review.md`.

## Ticket 06 formal paired gate (goal-loop round 32; 2026-11-17)

The formal six-cohort evaluation is retained at
`outputs/exp/20261117_153824_peg_orientation_formal_gate/`. Preregistration froze
both selected checkpoint identities, random-test/fixed-retention catalogs,
seed-1703 test-target shuffle, six-cohort order, paired `1+index` seeds, native
300-control cap/TimeLimit, 553-file source closure, analysis, deadlines, historical
exposure, and unchanged gates before native evaluation. The shuffle assignment
had two self-matches and two exact-yaw matches; it was preserved. No policy
selection used final outcomes.

| Cohort | Official native success | Controls |
|---|---:|---:|
| Random Base | 3/100 | 29,551 |
| Random correct live hint | 21/100 | 26,692 |
| Random fixed-standard-90° live hint | 7/100 | 29,085 |
| Random shuffled test-target hint | 5/100 | 29,174 |
| Fixed-retention Base | 52/100 | 22,097 |
| Fixed-retention correct live hint | 58/100 | 21,148 |

Random correct-vs-Base paired outcomes are Base-only 0 / Constrain-only 18 /
both 3 / neither 79 (**+18pp**, required ≥20pp). Fixed-retention outcomes are
Base-only 5 / Constrain-only 11 / both 47 / neither 37 (**+6pp**, within the
maximum 5pp loss). Correct vs fixed-standard is +14pp and correct vs shuffled
+16pp, descriptive with no hard hint-effect threshold. All six cohorts completed
100 scenes within the 300-control cap (157,747 controls total); no cohort timed
out. The random gate therefore remains **not passed**. Base 3/100 leaves 97pp
headroom, so the frozen headroom exception does not apply. The accepted Base
prerequisite remains validation-only 36/50, not 72/100; the old absolute 80/100
prerequisite remains waived, while fixed retention remains a required paired
measurement.

All 600 pre-insertion trigger records were `not_reached`; no yaw sample exists.
This is only the frozen 0.10m XY trigger and cannot be read as non-insertion:
official insertion occurred in all native successes. Phase traces show grasp in
all random episodes and 99/100 in each fixed cohort. The five frozen random yaw
bins, per-mode failures, full pair tables, and episode identities are in
`results.json`; commands and per-cohort hashes are in `execution_ledger.json` and
`post_execution_audit.json`.

Limitations: fixed-retention scenes were previously exposed to an older Base;
random catalog content/model-independent restore was historically accessed
without policy outcomes. Current selected weights have no retained result on the
fixed catalog. The first two foreground cohorts lack contemporaneous per-cohort
timing/pre-post source ledgers, but all per-episode checkpoint/catalog/evaluator
identities and the current 553-file closure match registration. A preregistered
scene-0 video was not captured because the existing CLI lacks a selected-scene
video switch; no cohort was rerun or source/protocol altered after outcomes. A
runner attempted to invoke already completed random-correct output once; it
exited on `FileExistsError` before scene restoration or native action, and the
original rows were reconciled without rerun. These are open deliverable/provenance
limitations. The formal status is **awaiting targeted diagnosis/correction
before final rejection**; no retuning, training, data expansion, retest, or Plug
work occurred. See the experiment `results.md` for exact regression commands,
individual elapsed times/bounds, identity hashes, incidents and full
interpretation.
