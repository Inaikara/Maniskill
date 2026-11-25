# PlugCharger orientation preflight (development contract)

## Retained best result and replay videos (2026-11-23)

The validation-selected random-500 30k checkpoint and frozen v1 80k Base are
hardlinked under `outputs/ckpt/retained_orientation/PlugCharger/`. Their heldout
results remain 41/100 versus 22/100 random and 54/100 versus 53/100 fixed.
Representative videos replay the recorded actions of the lowest successful scene
in each cohort; replay native success matches the audited result. They are in
`outputs/video/PlugCharger_BaseDP/` and `outputs/video/PlugCharger_ConstrainNet/`.
No historical checkpoint was deleted and no policy was resampled.

## Current cross-task frontier — Ticket 09 remains incomplete

**Peg:** the formal corrective retest (`outputs/exp/20261117_230000_peg_orientation_corrective_retest`)
scored random Base/correct/fixed-standard/shuffled **3/15/3/7 per 100**: correct
vs Base **+12 pp**, below the +20 pp target. Fixed retention was **52/49**, a
3 pp loss within the 5 pp allowance. This was nonblind because earlier random-test
outcomes had been viewed. The earlier +18 pp/+6 pp first formal test is historical,
not the qualification result. Base continuation was validation **36/50**, not an
80/100 pass. The random pool reached 500; selected studies used 30k updates.
Pose-trigger diagnostics were `not_reached` and provide no yaw-error samples.

**Plug:** frozen first Base v1 used 80k updates after 100k official-recipe
updates on 1,000 fixed demos; observations hide target orientation. Its absolute
Base gates were waived. Fresh 30k training runs used random pools of 100/200/300/500.
For the selected random-500 checkpoint
(`911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1`), validation
was Base **30/50 fixed, 18/50 random** and candidate **37/50, 23/50**: **+10 pp**
random gain, fixed retention passed, +20 pp target unmet. On the explicitly
authorized post-gate heldout, random Base/correct/fixed-standard/shuffled were
**22/41/28/20 per 100** and fixed retention **53/54**; deltas were **+19/+6/−2/+1
pp**. Correct exceeded the two hint controls, while shuffled was near Base: a
descriptive hint-sensitivity signal, not a separate hard gate. This one-seed
heldout estimate followed a missed validation gate. The run used 291,334 controls
and exactly 10,713.55349533 s outer time; wrapper exit 2 was a quoted-zero shell
artifact after complete journals.

Together, both tasks have positive correct-hint point estimates and fixed-skill
retention within 5 pp, but neither met its +20 pp random-gain target. They differ
in robot/task, success predicate, observations, data, budgets, and exposure; do
not average or add their success rates. These results support feasibility of
conditional adaptation only, not superiority to untested ordinary finetuning;
yaw sin/cos conditioning is not arbitrary 3D orientation control. Single-seed
feasibility is the present scope; formal multi-seed scope is undecided, not
authorized or completed. No representative videos were recorded, and independent
terminal full-pose recomputation is unavailable. The heldout was explicitly
authorized after the validation +20 pp miss; it does not convert that miss into
a pass. Ticket 09 Completion remains incomplete.

## Goal-loop Round 128 — random-500 heldout executed once

The separately authorized post-validation-gate roster ran once on CPU in the
registered order: random-final Base, correct, fixed-standard, shuffled; then
fixed-base-test Base and correct (**100 scenes per cohort, 600 total**). The
candidate remained the frozen random-500 update-30,000 checkpoint
`911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1`; frozen
Base SHA-256 is `ceeb68b4838ced6e6ffa43646bcc3cfc47538469e02264c4bc80a2542a33d130`.
The historical validation result remains **23/50 vs 18/50 (+10 pp)** and did
not meet the unchanged **+20 pp** gate. This heldout run does not rewrite that
failure, select another checkpoint, or authorize training.

| Heldout cohort | Successes | Delta vs paired Base | Native controls |
|---|---:|---:|---:|
| Random-final Base | 22/100 | — | 53,469 |
| Random-final correct hint | 41/100 | **+19 pp** | 47,896 |
| Random-final fixed-standard hint (current yaw − world-X zero) | 28/100 | +6 pp | 50,745 |
| Random-final shuffled hint | 20/100 | −2 pp | 53,583 |
| Fixed-base-test Base | 53/100 | — | 42,964 |
| Fixed-base-test correct hint | 54/100 | +1 pp | 42,677 |

Paired random-final transitions (wins/losses; both-success/both-failure):

| Comparison | Wins | Losses | Both pass | Both fail | Net / marginal delta |
|---|---:|---:|---:|---:|---:|
| Correct vs Base | 26 | 7 | 15 | 52 | **+19 pp** |
| Fixed-standard vs Base | 16 | 10 | 12 | 62 | +6 pp |
| Shuffled vs Base | 16 | 18 | 4 | 62 | −2 pp |
| Correct vs fixed-standard | 24 | 11 | 17 | 48 | +13 pp |
| Correct vs shuffled | 28 | 7 | 13 | 52 | +21 pp |
| Fixed retention: correct vs Base | 13 | 12 | 41 | 34 | +1 pp |

Correct-vs-Base randomized heldout gain is **+19 pp**, below +20 pp; do not
claim the numeric threshold was met. This is a one-seed post-gate heldout
estimate, not a significance claim. No routine-finetune comparison or Peg/Plug
pooling is included. Official outcomes use native `info.success`; journal
audits reconciled all 600 episodes to summaries, scene IDs, seeds, ≤600 controls,
finite 46-D observations/7-D actions, and the registered hints. Journals do not
retain terminal simulator physical states, so independent full-pose predicate
recomputation is unavailable.

The runner used **291,334 native controls**, taking **10,709.807 s** internally
and **10,713.55349533 s** outer monotonic time (within 21,600/25,200 s limits). The
exact canonical 25-file suite passed **285 before and 285 after**. An initial
pre-test collection attempt without `PYTHONPATH=.` failed before any policy
action (25 import errors); its raw logs are retained and the correctly configured
retry passed. All 15 directly sealed input identities, 261 archived current
source files, and 4,010 imported Python runtime-source hashes match pre/post.
The seed-1703 target-yaw permutation was written and hashed from manifest
metadata before the rollout; two self-matches were retained. Heldout scenes were
accessed only by this one runner.

The evaluator completed and wrote all 600 results. Its wrapper nevertheless
returned exit 2 because a quoting error passed literal `"0"` to the final shell
`exit`; the raw command/logs and incident are preserved, and no rerun occurred.
This wrapper bookkeeping issue does not change the complete evaluator summary.
Complete episode-level CSV/JSONL audit, cohort summaries, paired transitions,
preregistration, runtime/source archives, raw logs, exact pre/post tests and
hashes are under `outputs/exp/20261123_070724_plug_random500_heldout_envelope/`;
the 600 per-control journals and six arm summaries are under
`outputs/exp/20261123_070724_plug_random500_heldout/`. No source/dependency,
data, or checkpoint was changed; no commit/push. Local ticket notes remain
unstaged.

## Round 126 — independent heldout runner implemented (not executed)

The public `evaluate-heldout` CLI is implemented and bound to the existing
validation selection artifact, selected random-500 update-30,000 checkpoint,
frozen Base v1 identities, and both frozen heldout catalog manifests. Forward
authorization explicitly allows evaluating this already-selected candidate
despite its **+10 pp** randomized validation gain missing the unchanged **+20
pp** gate. This is not a gate pass or a passing-gate confirmatory test; heldout
outcomes remain unseen and cannot select or retrain checkpoints. Historical
validation artifacts and decisions are unchanged.

The fixed plan is 600 CPU episodes in this order: random-final Base/correct/
fixed-standard/shuffled (100 each), then fixed-base Base/correct (100 each).
Correct hints use scene yaw, fixed-standard means current charger yaw minus
world-X yaw zero, and the one seed-1703 permutation is derived from random
catalog metadata before rollout and retains self-matches. The Base adapter and
Constrain adapter are each instantiated once. Seed `1 + scene index` is applied
after scene restore for every arm; native `info.success` is authoritative.
Native official `obj_to_goal_dist` / `obj_to_goal_angle` values are journaled
when supplied by `info`. The runner journals every episode/control and aligned yaw-error diagnostics, writes
incremental progress, and preserves completed rows on exceptions. Journals do
not retain simulator terminal physical states, so independent full-pose success
recomputation is unavailable. No existing compatible per-arm video path was
established; video is an explicit gap, not synthesized. No Base absolute 40/80
qualification gates are applied.

The exact registered random-validation result remains 23/50 versus fresh Base
18/50 (+10 pp); fixed is 37/50 versus Base 30/50. The method gate remains unmet.
Runner development recorded a genuine missing-command RED (1 failed), followed
by focused GREEN (44 passed) and canonical 25-file GREEN (284 passed). Raw logs,
UTC/monotonic markers, exact roster, and command history are in
`outputs/exp/20261123_064041_plug_heldout_runner_development/`; the genuine RED
is at `outputs/exp/20261123_063021_plug_heldout_runner_development/`. **This implementation
round executed zero heldout episodes, zero native controls, and no policy action.**
Ticket 09 remains incomplete pending actual paired heldout results and the later
cross-task writeup/final conclusion.

Runnable command (use a fresh timestamped output path; execution is not part of
the implementation evidence):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=..:.

timeout --signal=TERM --kill-after=20s 25200s \
  python -m examples.baselines.constrain.plug_orientation_adaptation evaluate-heldout \
  --output-root outputs/exp/YYYYMMDD_HHMMSS_plug_heldout_600
```

The CLI pins the registered checkpoint and catalogs; its internal stage budget
is 21,600 seconds and its roster cannot be reduced through public flags.

## Round 125 — random-500 validation complete; method gate not met

The one authorized 700-episode CPU PhysX roster completed with exit 0 in
**12,816.473 s** under the 18,000 s TERM / 20 s KILL cap. It evaluated fresh
frozen Base once per catalog and all six 500-random checkpoints on both frozen
50-scene validation catalogs: **100 Base + 600 Constrain episodes**, totaling
**316,520 native controls** (below the 420,000 maximum). No early stopping,
retry, training, data change, or heldout access occurred. The canonical 25-file
suite passed **276 before and after**.

Fresh Base: **30/50 fixed, 18/50 random**. Candidate fixed/random scores:

| Update | Fixed | Random | Fixed loss vs Base | Random gain vs Base | Retention eligible |
|---:|---:|---:|---:|---:|:---:|
| 5k | 21/50 | 12/50 | +18 pp | −12 pp | no |
| 10k | 25/50 | 12/50 | +10 pp | −12 pp | no |
| 15k | 28/50 | 17/50 | +4 pp | −2 pp | yes |
| 20k | 33/50 | 17/50 | −6 pp | −2 pp | yes |
| 25k | 31/50 | 21/50 | −2 pp | +6 pp | yes |
| 30k | 37/50 | 23/50 | −14 pp | **+10 pp** | yes |

The frozen rule selects **30k**, checkpoint SHA-256
`911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1`. Fixed
retention passes; randomized gain **+10 pp** misses the unchanged **+20 pp**
method gate. This is a positive validation point estimate, not a gate pass,
significance result, or independent generalization claim. One training seed and
validation-based checkpoint selection limit interpretation. Historical selected
100/200/300/500 results are respectively **30/20, 38/21, 33/23, 37/23**
(fixed/random); the fresh Base fixed scores were **29/31/30/30** across those
rounds, so these are descriptive cross-round outcomes, not a causal data-scaling
comparison.

All 700 episode journals reconcile to the 14 cohort summaries on scene index,
seed, controls, finite 46-D observations/7-D actions and recorded native
`info.success`. Journals do not retain simulator terminal physical states, so
independent full-pose terminal-predicate recomputation is not claimed. **338
directly preregistered input/source hashes match pre/post**; this includes the
inherited 49-file data/ancestry/Base/catalog seal, six checkpoints and the
captured evaluator/runtime source closure. Nine catalog-bundle files were
identity-only; final/test scene contents were not opened. The Base absolute gate
remains waived only as a prerequisite; this result does not qualify Base.
Supplemental post-run checks also found all 18 random-100/200/300 checkpoints and
all five original Base v1 plus five Base v2 checkpoints match their retained
prior identity seals; these are not claimed as contemporaneous pre-run captures.

Evidence and complete provenance: validation episodes and summaries at
`outputs/exp/20261122_045531_plug_constrain_random500_validation/`; preregistration,
raw logs, 338-file pre/post hashes, 261-file source archive, 700-journal audit,
tests and artifact manifest at
`outputs/exp/20261122_045531_plug_constrain_random500_validation_envelope/`.
The 500-pool feasibility study stops here. Ticket 09 remains incomplete; both
heldout catalogs remain closed, with no further data, training, variants, or
heldout runner authorized by this validation result.

**Historical-state pointer:** this Round-125 “heldout closed” statement records
the validation-time decision only. Round 128's later explicitly authorized
heldout execution is the current execution state; see the current cross-task
frontier above. It does not change the missed validation gate.

## Round 124 — random-500 training complete; validation pending

The one separately authorized fresh seed-1 run completed **30,000/30,000 CUDA
updates** on 500 random / 1,000 fixed demonstrations. It started from the frozen
Base v1 EMA with a fresh zero-control branch, never from a prior Constrain
checkpoint. Runtime: **1,187.152 s trainer / 1,207.809 s outer monotonic**;
exit 0 on an RTX 5070. The exact canonical 25-file suite passed 276 before and
after. All 300 log rows are finite and final LR is zero.

The six 5k–30k candidates passed strict CPU payload checks: all 148 Base tensors
are exact, all 100 control EMA values are finite and match their EMA shadow
ordering, and optimizer/scheduler/EMA update steps agree. Checkpoint hashes and
full audit are in `outputs/exp/20261122_042359_plug_orientation_constrain_random500_envelope/`;
weights are in `outputs/ckpt/20261122_042359_plug_orientation_constrain_random500/`.
All 49 directly preregistered protected identities match pre/post, including
the 500 data and ancestors, Base, sources and all nine catalogs (identity-only).

No policy action, validation, native environment or heldout query was made. This
is training completion only, not an efficacy result. Stop for independent parent
review; if separately approved, the only next operation is the existing
700-episode validation roster, with no added data or training. Round 123's data
admission and Round 121's negative validation result remain unchanged.

## Forward amendment — Round 122 random-500 implementation (offline only)

User instruction (2026-11-22): **“增加到500条 继续相同实验”**. This one forward amendment supersedes Round 121's 300-data stop rule for the added 500-pool experiment only; the prior 300 data, contracts, checkpoint identities and negative +10 pp validation result remain historical and unchanged. Keep the exact 300 accepted rows first, append the first 200 eligible disjoint candidates (seeds 2026123450–2026123749; one `default_rng(1704)` stream draws 750 yaws and consumes indices 450–749), then train once from the same frozen Base v1 EMA for 30k updates and validate the same 700-scene roster. No 400-pool training, Base variant, method-gate change, expansion beyond 500, final/heldout access or paired heldout runner is authorized. Base absolute gates remain waived only as prerequisites; method gates are unchanged and a negative outcome is acceptable.

Offline implementation adds the 500 contract, actual-parent/ancestry preflight and lineage validation, `collect-random500`, `train-expanded500`, and count-bound checkpoints; existing validation handles the six-candidate 700-arm roster. The parent-300 validator was exercised against the real immutable parent before an intentional pre-native callback stop; a synthetic NativeEE source/replay expansion passed the real 500 lineage reader, followed by a two-update CPU toy training/checkpoint check. This round ran zero new native controls, production optimizer updates, trained-policy queries or heldout queries. The canonical historical 25-file suite passes **276 tests** (270 baseline + six additions). Independent review is required before any new collection. Full commands, scope and artifacts: `outputs/exp/20261122_031224_plug_random500_development/` and `OUTPUTS.md`.

After independent review and separate stage authorization, these are the frozen runnable forms (use new timestamped paths if the paths already exist):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.

timeout --signal=TERM --kill-after=20s 12000s \
  python -m examples.baselines.constrain.plug_orientation_adaptation collect-random500 \
  --parent-dataset demos/PlugCharger-orientation-random-300_20261120_173932 \
  --dataset-dir demos/PlugCharger-orientation-random-500_20261122_031224 \
  --exp-dir outputs/exp/20261122_031224_plug_random500_collection

timeout --signal=TERM --kill-after=20s 9000s \
  python -m examples.baselines.constrain.plug_orientation_adaptation train-expanded500 \
  --random-h5 demos/PlugCharger-orientation-random-500_20261122_031224/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --random-manifest demos/PlugCharger-orientation-random-500_20261122_031224/dataset_manifest.json \
  --fixed-h5 demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --fixed-manifest demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.json \
  --exp-dir outputs/exp/20261122_031224_plug_random500_train \
  --ckpt-dir outputs/ckpt/20261122_031224_plug_random500_train --device cuda

timeout --signal=TERM --kill-after=20s 21600s \
  python -m examples.baselines.constrain.plug_orientation_adaptation evaluate-validation \
  --ckpt-dir outputs/ckpt/20261122_031224_plug_random500_train \
  --random-catalog outputs/exp/20261119_184307_plug_orientation_catalogs/random_validation \
  --fixed-catalog outputs/exp/20261119_184307_plug_orientation_catalogs/fixed_validation \
  --output-root outputs/exp/20261122_031224_plug_random500_validation
```

The outer collection/train/validation timeouts are respectively **12,000 / 9,000 / 21,600 s**; code enforces internal collection/training budgets **10,800 / 7,200 s**. Collection may admit only the first 200 eligible of the 300 registered candidates. Training/validation should begin only after independently auditing exactly 500 rows and their provenance. These commands do not authorize use of final catalogs.

## Goal-loop Round 121 — random-300 Constrain validation complete (method gate failed)

The single preregistered validation-only roster completed **700/700 episodes**:
fresh frozen Base once per 50-scene profile, plus all six Round-119 random-300
checkpoints on both fixed and random validation catalogs. CPU PhysX, default
stride-8 native 7-D actions, seed `1 + scene index`, and the unchanged 600-control
episode cap were used. Total **322,364 native controls**, **13,079 s outer wall**,
exit 0 under the exact 18,000 s TERM / 20 s KILL cap. The canonical 25-file suite
passed **270 before and after** (52.92 s each).

Fresh Base was **30/50 fixed, 18/50 random**. All six candidates:

| Update | Fixed | Random | Fixed retention loss | Random gain | Eligible |
|---:|---:|---:|---:|---:|:---:|
| 5k | 14/50 | 7/50 | +32 pp | −22 pp | no |
| 10k | 26/50 | 12/50 | +8 pp | −12 pp | no |
| 15k | 33/50 | 15/50 | −6 pp | −6 pp | yes |
| 20k | 33/50 | 19/50 | −6 pp | +2 pp | yes |
| 25k | 33/50 | 23/50 | −6 pp | **+10 pp** | yes |
| 30k | 34/50 | 17/50 | −8 pp | −2 pp | yes |

Selection among fixed-retention-eligible candidates is 25k,
SHA-256 `75226bbea156ce833da0658d54f9d565990dc7ea602e05287d76ec15778e8cca`.
Its fixed retention passes, but random gain **+10 percentage points** misses the
unchanged **+20 pp** method gate. **A positive +10 pp validation point estimate
was observed, but the prespecified improvement target was not met.** The bounded
feasibility study closes; original Ticket 09 acceptance remains incomplete
because its independent paired heldout evaluation was not executed.
No further data collection, training/refit,
architecture change, or variant is authorized; do not start heldout/final paired
evaluation. Both heldout catalogs remain closed. The old Base absolute gates
remain waived only as forward-study prerequisites; this run does not qualify
Base. One-seed, validation-selection results are descriptive, not significance
or independent-generalization evidence.

All 700 event journals were audited against their summaries: scene IDs/seeds,
controls, finite 46-D observations and 7-D actions, and episode outcomes reconcile
to native `info.success`. Journals do not store simulator terminal physical
states, so independent full-pose terminal-predicate recomputation is not claimed.
All 32 registered input files (including all nine catalog files), ten experiment
source files, and 18 explicitly captured installed runtime source files match
pre/post hashes. The 12 older random-100/200 checkpoints and ten v1/v2 Base
checkpoints also match retained prior sealed identities; that comparison is not
represented as a new contemporaneous pre/post capture. Final/test catalog
payloads were hash-only and never opened as scenes.

Lineage boundary: the Round-121 preregistration did not separately enumerate the
random-300 attempt ledger/parent-lineage JSON, fixed-dataset manifest, or random-100
ancestor files. The random-300 ledger/parent lineage/manifest match Round-119
pre/post seals; the fixed manifest matches the original collection admission seal;
random-100 HDF5/JSON/manifest match hashes embedded in the frozen random-200
parent-lineage record. These are prior/transitive provenance checks, not direct
Round-121 pre-run file entries.

Preregistration SHA-256 `ff966a69f7ff8d508b551735a136074a4201b88723763af1434eb6e723b4800f`.
Complete run, raw logs, preregistration, input/source snapshots, audits, gate,
and artifact hashes: `outputs/exp/20261120_183731_plug_constrain_random300_validation_envelope/`;
700 journals and summaries: `outputs/exp/20261120_183731_plug_constrain_random300_validation/`.
No source/dependency/data/checkpoint edits, branch change, commit, or push.

Across the validation rounds, the selected results are: random-100 **30/50 fixed,
20/50 random** (Base 29/18; fixed loss −2 pp, random gain +4 pp); random-200
**38/50, 21/50** (Base 31/18; fixed loss −14 pp, gain +6 pp); random-300
**33/50, 23/50** (Base 30/18; fixed loss −6 pp, gain +10 pp). All miss the +20 pp
method gate. Fresh Base fixed scores differ (29/31/30), so these are historical
validation summaries, not direct paired pool-size effects. The 300-pool failure
closes the bounded study and leaves holdouts closed.

## Goal-loop Round 119 — random-300 Constrain training complete; validation pending (historical)

After the Round-118 data admission, a separate parent authorization permitted
exactly one fresh seed-1 30,000-update CUDA run on the 300-random / unchanged
1,000-fixed pools. Production loader checks passed (**84,891 / 279,625 windows**;
`(2,46)/(16,7)/(2,)` samples); balanced 512/512 windows, batch 1,024. The run
used the unchanged frozen Base v1 80k EMA and a fresh zero-control branch, not
prior Constrain weights. It completed in **1,187.25 s trainer / 1,207.03 s outer**,
exit 0. Six distinct 5k–30k candidates pass CPU strict-load/provenance audit;
all 148 Base tensors are bit-identical and 100 control EMA tensors are finite and
nonzero. The exact canonical 25-file suite passed **270 before and after**.

At Round 119 this was training completion only; Round 121 has since completed the
registered validation roster and failed the method gate. The bounded study stops
at the 300-demo ceiling; do not collect, tune, retrain Base, or open heldouts. Full hashes,
checkpoints and protocol evidence are in
`outputs/exp/20261120_175628_plug_random300_train_envelope/` and sibling run
directories documented in `OUTPUTS.md`. Incident accounting is corrected
additively: the path-normalization helper error is documented, while the separate
claimed pre-created-directory CLI failure has no retained `first_launch.*` raw
command/stderr/exit/time evidence. Its zero-update status is not independently
verified and it is not counted as a proven zero-update incident. The retained
successful-run evidence establishes training completion, not that missing
first-attempt detail; see the reconciliation receipt referenced in `OUTPUTS.md`.

## Goal-loop Round 118 — random-300 data admitted; no training or evaluation

The approved bounded continuation ran once from the immutable random-200 pool
and completed successfully. Collection began from that pool's actual identity,
using only the sealed new seed range **2026123300–2026123449** and the 150 yaw
draws **300–449** from `default_rng(1704)` uniformly over `[-pi/8,+pi/8]`.
The first 100 eligible new source/replay pairs were admitted in candidate order:
**100/107** candidates, seeds **2026123300–2026123406**. No earlier candidate
range was reused and no retry occurred.

The unchanged random-200 parent contributes 200 episodes; the new 100 bring the
effective pool to **300**. New-stage counters are separate from historical parent
attempts: source **29,882** controls and raw replay **29,882** controls (90,000
cap each); the maximum per-attempt side was **374** (each episode remained within
the 600-control contract). New yaw bins are **[19,25,18,14,24]** and combined
300-row bins **[55,65,50,64,66]**. The complete production `PlugAdaptationDataset`
loader and real 300-lineage validator passed: **84,891 windows**, finite native
data, action maximum 1.0, and first-item shapes observations `(2,46)`, actions
`(16,7)`, hint `(2,)`. This checks the actual random-200 lineage and its
random-100 ancestor, all parent HDF5 values, and every new source/replay/case
identity, action stream, initial state/observation and full-pose terminal gate.

Collection took **266.439 s** (exit 0), below the 5,400 s internal and external
TERM / 20 s KILL limits. The exact canonical 25-file regression suite passed
**270 tests both before and after**. Registered source files and the complete
immutable parent tree matched pre/post hashes. The other protected historical
datasets, checkpoint files, nine catalog files, and producer archives were
post-hashed against the retained prior identity bundle or their own producer
seals; the evidence explicitly does not represent those comparisons as a new
Round-118 contemporaneous pre/post snapshot. Dataset HDF5 SHA-256
`95ede016098066b535208ec2569d4c092186a242e4627206306d44137dc2fa0f`, JSON SHA
`129dcf74369eb95dad2e2960877024afe9f831a47bd36576dea4ce1411f1a42f`, and
manifest SHA `e32b6914a08d41adfb7cb2e87f0af458cd4d035880373a4b51cf3fdfc175b6dc`.

Data and native attempt records:
`demos/PlugCharger-orientation-random-300_20261120_173932/`; preregistration,
source archive, and exact collection run:
`outputs/exp/20261120_173932_plug_random300_collection/`; independent admission,
pre/post inputs, tests, timing, and artifact manifest:
`outputs/exp/20261120_173932_plug_random300_collection_envelope/`.

This is **data admission only**. No production training update, checkpoint,
policy action, validation episode, or heldout/test outcome was run. The prior
random-200 method result remains **+6 pp**, below the fixed **+20 pp** gate; this
data does not change that result or establish method improvement. The existing
stop rule permits parent review only: no next data expansion beyond 300, training
or validation is authorized by this collection result. Final/heldout catalogs
remain closed; Ticket 09 remains incomplete. No ManiSkill core/dependency change,
branch change, commit, or push.

## Goal-loop Round 115 — fresh random-200 Constrain validation complete (method gate failed)

The complete 700-arm validation roster ran once: fresh frozen v1 Base once per
50-scene profile and all six 5k–30k random-200 Constrain checkpoints on both
profiles. Fresh Base was **31/50 fixed, 18/50 random**. Candidate scores were
5k **26/50, 9/50**; 10k **22/50, 14/50**; 15k **36/50, 14/50**; 20k
**39/50, 19/50**; 25k **41/50, 17/50**; 30k **38/50, 21/50** (fixed, random).
The selector chooses 30k, SHA-256
`b0764215fb65d6c3b1bd7852788dadc39a0883d9a06ec391f3c5a73868e67167`, with
fixed retention loss −14 pp and random gain **+6 pp**. Retention passes, but
the +20 pp random-gain method gate fails. No candidate is frozen for final
evaluation; keep final/test and heldout outcomes closed. The old Base absolute
qualification gate remains waived only as a forward-study prerequisite; no
Base qualification is claimed.

Work was **700 episodes / 311,659 native controls / 12,517 s**, exit 0 under the
18,000 s cap. Exact 261-test roster passed before/after (28.64 s / 35.46 s).
All 700 event journals reconcile to summary controls and native `info.success`;
registered inputs/source and all nine catalog-file hashes match pre/post. Fifteen
captured installed source files match pre/post. No final/test outcome was opened.
The journal does not retain simulator terminal states, so independent
full-quaternion terminal-predicate recomputation is not claimed. This one-seed
validation is used for selection, not significance inference; Round 107 remains
an independent historical result. Full evidence:
`outputs/exp/20261120_131136_plug_constrain_random200_validation_envelope/` and
`outputs/exp/20261120_131136_plug_constrain_random200_validation/`.

## Goal-loop Round 114 — fresh random-200 Constrain training complete; validation pending

After the independently admitted 200-effective-episode pool and production-loader
lineage audit, one fresh seed-1 run completed **30,000/30,000 CUDA updates** from
the exact frozen original v1 80k Base EMA, with a zero-initialized Constrain branch
and balanced 512 random / 512 fixed windows per batch. The new run-owned contract
records both the parent random-100 forward-contract digest and the actual
random-200 manifest digest. Training took **1,188.47 s trainer time / 1,206 s
outer**, exit 0; six distinct checkpoints (5k–30k) were saved.

The exact 25-file suite passed **261 tests** before/after; CPU audit confirms all 148 Base
tensors unchanged in every checkpoint, optimizer/scheduler/EMA step alignment,
finite 300-row loss/LR log, and strict adapter restore without action sampling.
No trained-policy action, validation scene, final/test scene, or heldout outcome
was accessed. This is not a method-gate result: the random-100 validation gain
remains +4 pp and below the +20 pp gate, while the new random-200 candidate is
unvalidated. Continue only with parent review and then the frozen validation-only
roster; do not open final/heldout data.

Evidence and checkpoint SHA-256s:
`outputs/exp/20261120_123140_plug_orientation_constrain_random200_envelope/`;
training logs/config/source archive:
`outputs/exp/20261120_123140_plug_orientation_constrain_random200/`; weights:
`outputs/ckpt/20261120_123140_plug_orientation_constrain_random200/`. No commit.

## Goal-loop Round 107 — complete 700-arm Constrain validation (method gate failed)

The full registered validation-only roster completed once: fresh frozen Base
**29/50 fixed, 18/50 random**, and each of the six 5k–30k Constrain checkpoints
on both complete 50-scene validation catalogs. The evaluator used CPU, seed
`1 + catalog index`, the official stride-8 sampler/native 7-D actions, and the
unchanged 600-control per-episode cap. Total was **700 episodes / 324,694
controls / 13,431.92 s**, exit 0 under the 18,000-second outer cap. Exact 248-test
suite passed before and after (30.94 s / 30.29 s). All nine catalog-bundle files,
all six candidate checkpoints, the frozen Base, and experiment-side source hashes
match their pre-run identities. No final/test or holdout scene was accessed.

| Update | Fixed | Random | Fixed retention loss | Random gain | Retention eligible |
|---:|---:|---:|---:|---:|:---:|
| 5k | 19/50 | 9/50 | +20 pp | −18 pp | no |
| 10k | 28/50 | 8/50 | +2 pp | −20 pp | yes |
| 15k | 33/50 | 10/50 | −8 pp | −16 pp | yes |
| 20k | 34/50 | 11/50 | −10 pp | −14 pp | yes |
| 25k | 30/50 | 20/50 | −2 pp | **+4 pp** | yes |
| 30k | 36/50 | 18/50 | −14 pp | 0 pp | yes |

Frozen selection is **25k**, SHA-256
`c6b933083d19b68bbcb42ee4449f40fe95fcdc4efb7c0b1e89b836a25deb5735`, by max
random success among checkpoints meeting `fixed >= ceil(29 − 2.5) = 27/50`, with
latest-update tie-break. It passes fixed retention but gains only **+4 pp** on
random validation, below the required **+20 pp** (10/50). The method gate fails;
this candidate is validation-frontier-only, not qualified, and does not open any
heldout/test split. The old Base absolute 40/50 and 80/100 gates remain waived
only as forward-study prerequisites; historical Base remains 30/50 and
unqualified. No significance claim is made from the single training seed.

Audit caveat: per-control action/observation journals include native
`info.success` and reconcile with all 700 arm-summary outcomes, but simulator
terminal states were not saved, so terminal-state recomputation of full-quaternion
5 mm / 0.2 rad success was unavailable. Pre-run source closure sealed 14
experiment-side files and runtime versions but not installed ManiSkill/Diffusers
source bytes; no installed-source byte-identity claim is made. Full reports,
selection, raw episodes, preregistration, hashes, logs, timings, audits, and
artifact manifest are in `outputs/exp/20261120_050952_plug_constrain_random100_validation/`
and sibling `outputs/exp/20261120_050952_plug_orientation_constrain_random100_validation_envelope/`.

Next action is parent review/decision on validation-only diagnosis and whether to
expand the admitted randomized demonstrations (within the existing 300-episode
ceiling). Do not open heldout/test data or start the final paired runner unless a
candidate first meets the random-gain and fixed-retention method gates.

## Goal-loop Round 106 — authorized Constrain adaptation training complete

The amended forward study completed its one fresh seed-1 run at **30,000/30,000
updates** on CUDA, using 512 random-100 and 512 fixed-1,000 episode windows per
batch. All six registered checkpoints (5k–30k) have distinct file hashes and
pass CPU-only format/config/provenance, finite-weight, optimizer/scheduler/EMA
step, strict adapter-load, and exact frozen-Base audits. Every checkpoint
preserves all 148 tensors of the original v1 80k EMA Base bit-for-bit; the EMA
control branch is nonzero. The 300-row training log is finite and ends at LR 0.
The exact registered 248-test suite passed both before and after (22.93 s and
30.91 s). The run-seeded sequential yaw algorithm used for the admitted random
100 is bound to its unchanged collection preregistration and deviation record;
neither was rewritten.

This is **training completion only**. No trained-policy action, validation or
test scene, or heldout outcome was accessed, and there is no method-improvement
or qualification claim. The historical Base remains 30/50 and unqualified. Run
artifacts and the full audit are under
`outputs/exp/20261120_043914_plug_orientation_constrain_random100_envelope/`; run
logs and checkpoints are in the matching timestamped `outputs/exp/` and
`outputs/ckpt/` directories. After parent audit, the next authorized gate is
validation-only evaluation. Keep heldout outcomes closed until the candidate is
frozen; no Base retraining or v2/v3 continuation is part of this run.

## Goal-loop Round 105 — randomized adaptation data admitted

The authorized initial randomized-100 native EE collection completed once under
the unchanged additive adaptation contract. It admitted the first 100 eligible
source-plus-independent-raw-action-replay episodes after **118** ordered
candidates (seeds `2026123000–2026123117`); 18 were rejected by the unchanged
official full-pose predicate (17 `lift_tracking_failed`, one `align_failed`).
There were **31,925 source + 31,925 replay controls** across all attempts, within
the registered 90,000-per-side caps; collection took **274.29 s** of the 5,400 s
internal deadline. Accepted yaw-bin counts are **[18, 18, 20, 23, 21]**, with
actual target yaw range **−0.387467…+0.392201 rad**. The source/replay controls
are native `pd_ee_delta_pose` 7-D actions. The environment class is
`PlugChargerOrientation-v1`; random targets were applied as each case's actual
`target_yaw` override and recorded in raw native state and consolidated metadata.

Independent admission checked all 100 source and 100 replay HDF5s: finite bounded
7-D actions, finite T+1 states/46-D observations, unchanged official terminal
5 mm / 0.2 rad full-pose success, exact source/replay action equality and exact
initial state/observation equality. All 100 consolidated trajectories were
recursively value-compared with their raw source trajectories. Production
datasets admit **28,202 randomized** and **279,625 fixed** windows; first-item
shapes were observations `(2,46)`, actions `(16,7)`, hint `(2,)`. This was an
offline dataset check only (zero model queries/environment construction). The
random HDF5 SHA-256 is
`2df8ffa9b95c27e2ef9f7a471edefb68f73c8f6123769e9858eb9fa53f2fe26c`; JSON is
`77e9fed3070ad326b0323ace70b6b7015ad4394bf374a71193487685db4f6c0f`.

**Collection RNG deviation:** the request specified per-case
`default_rng(candidate_scene_seed + 1703)`, but production used the single
sequential stream `np.random.default_rng(collection_run_seed=1 + 1703)` for all
150 planned yaw draws. The full list was prospectively sealed before physics in
the unchanged preregistration. This did not affect the disjoint 118-scene
eligibility/coverage order, uniform `±pi/8` yaw support, or row eligibility;
there was no policy-outcome-driven selection. The 100 rows are admitted for
forward study using the actual run-seeded algorithm, not the requested per-scene
algorithm. No rerun, new data, reselection, preregistration alteration, or source
fitting occurred. Full deviation record:
`outputs/exp/20261120_041640_plug_random_adaptation_collection_envelope/collection_rng_deviation.md`.

The exact registered **248-test** suite passed before (**23.61 s**) and after
(**22.68 s**) collection. The frozen fixed dataset, adaptation contract, 80k
Base checkpoint, and all 32 archived source-closure files match their checked
identities. The four catalog bundle reconciles to the frozen Round 92 catalog-set
digest; individual catalog files were not separately captured in this round's
pre-run hash list, so the evidence does not claim a direct per-file before/after
comparison. No catalog scene was opened. The test subprocesses completed well
inside the 180 s cap; their wrapper used Python's timeout rather than the planned
TERM-then-KILL shell timeout (no timeout signal was needed). UTC starts and
elapsed times are retained; individual subprocess monotonic starts were not
captured or reconstructed. Evidence, exact commands/test list,
logs, preregistration, timings, hashes, and independent audit are in
`outputs/exp/20261120_041640_plug_random_adaptation_collection_envelope/`; raw
attempts/source/replay and consolidated data are in
`demos/PlugCharger-orientation-random-100_20261120_041640/`.

**Disposition:** 100 randomized episodes are independently data-admitted for
the authorized adaptation training slice. No optimizer step, checkpoint, trained
policy query, validation/test scene, or heldout outcome occurred. This does not
establish method improvement or change the original Base result (80k, 30/50);
its historical unqualified status remains unchanged. The sole next authorized
step is the registered 30k balanced 512/512 adaptation training on the unchanged
fixed 1,000 + admitted randomized 100 data, freezing the exact original Base.
Validation-only selection remains the next policy gate; final/test splits stay
closed. No commit or push was made.

## Forward amendment — frozen v1 Base accepted for method study

The user amended the next study to continue with the original v1 Base: “不用严格
限定base的成功率 只要是按照官方要求训练的就好 重点看我们的方法有没有优化
就按照第一版的base模型继续吧”. This waives the previous Base absolute-success
prerequisites for forward Constrain adaptation only. It does **not** change the
original Base result or claim qualification: the selected v1 checkpoint remains
80k, 30/50 fixed validation; its 40/50 validation and 80/100 test gates remain
historical and were not met. No Base retraining or v2/v3 continuation is planned.

The additive, forward-only machine contract is
[`plug_orientation_adaptation_contract.json`](plug_orientation_adaptation_contract.json).
It pins the original v1 checkpoint bytes and EMA tensor identity, 30k seed-1
fresh ControlNet-branch updates, randomized/fixed balanced windows, full-pose
evaluation, and the unchanged method comparisons (≤5 percentage-point fixed
retention loss and ≥20-point randomized gain). This contract does not edit the
historical Base contract or outcomes.

Offline implementation provides the bounded randomized-100 collection CLI,
live 46-D hint extraction, official Plug episode-window semantics, a frozen-v1
Constrain adapter and 30k training CLI, plus the validation-only 700-arm runner
and retention-constrained selection in `plug_orientation_adaptation.py`. The
final heldout paired runner remains deferred until a candidate is validated; it
is not a blocker for this initial chain. These CLIs do not imply permission to
run native collection or production training. No physical callbacks, policy
outcomes, or production optimizer updates occurred in this implementation slice.

Run commands (after review/authorization, with fresh UTC-stamped output paths):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
RUN_ID=$(date -u +%Y%m%d_%H%M%S)
timeout --signal=TERM --kill-after=10s 6000s python -u -m examples.baselines.constrain.plug_orientation_adaptation collect-random100 \
  --dataset-dir "demos/PlugCharger-orientation-random-100_${RUN_ID}" \
  --exp-dir "outputs/exp/${RUN_ID}_plug_orientation_random_collection"
timeout --signal=TERM --kill-after=10s 7200s python -u -m examples.baselines.constrain.plug_orientation_adaptation train \
  --random-h5 "demos/PlugCharger-orientation-random-100_${RUN_ID}/trajectory.state.pd_ee_delta_pose.physx_cpu.h5" \
  --random-manifest "demos/PlugCharger-orientation-random-100_${RUN_ID}/trajectory.state.pd_ee_delta_pose.physx_cpu.json" \
  --fixed-h5 demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --fixed-manifest demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.json \
  --exp-dir "outputs/exp/${RUN_ID}_plug_orientation_adaptation" \
  --ckpt-dir "outputs/ckpt/${RUN_ID}_plug_orientation_adaptation" --device cuda
timeout --signal=TERM --kill-after=10s 21600s python -u -m examples.baselines.constrain.plug_orientation_adaptation evaluate-validation \
  --ckpt-dir "outputs/ckpt/${RUN_ID}_plug_orientation_adaptation" \
  --random-catalog outputs/exp/20261119_184307_plug_orientation_catalogs/random_validation \
  --fixed-catalog outputs/exp/20261119_184307_plug_orientation_catalogs/fixed_validation \
  --output-root "outputs/exp/${RUN_ID}_plug_adaptation_validation"
```

## Goal-loop Round 101 — v2 full fixed-validation result (gate failed)

The complete authorized five-checkpoint v2 validation ran once on the unchanged
50-scene `fixed_validation` catalog, with execution stride 8, CPU inference,
seed `1 + catalog index`, a 600-control per-scene cap, and no retries or early
cohort stopping. Results were **20k 1/50, 40k 9/50, 60k 15/50, 80k 13/50,
100k 19/50**. The frozen max-success/latest-update-tie rule selects **100k,
19/50**. This is below the **40/50** fixed-test opening threshold, so no
fixed-test episode was opened and Base remains unqualified. The random-final
test was also untouched; no Constrain training followed.

All **250** episode records and per-control event traces were independently
checked for complete unique catalog indices, paired catalog/policy seeds,
trace/summary control-count agreement, finite 46-D observations and native 7-D
actions, native action bounds, and agreement with the official terminal success
flag and pose thresholds. Total native controls were **133,080 / 150,000**.
The 24-file required regression suite passed **239 tests before and after**.
Frozen data, v1 and v2 checkpoints, scale statistics, four catalog identities,
contracts, evaluator and test sources remained hash-identical pre/post.

This is the separately normalized, validation-informed v2 corrective family;
it does **not** alter the original v1 outcome (**30/50**) or represent a pass of
the unmodified v1 baseline. v1's test gate remains closed. No new weights/data,
normalization fit, source/dependency changes, branch/commit/push, random-final
test, or Constrain work occurred. Full preregistration, command logs, all raw
episode traces, test results, hashes, and audit are in
`outputs/exp/20261120_005452_plug_scaled_v2_validation_envelope/`; model
summaries and per-control traces are in
`outputs/exp/20261120_005452_plug_scaled_v2_validation/`.

The immutable v2 selection digest is recorded in that validation directory's
`validation_selection.json`; it is **not** a fixed-test authorization.
Qualification is not established; any future validation-only diagnosis or
continuation needs its own prospectively bounded step and cannot access either
closed test here.

## Goal-loop Round 100 — fresh v2 scaled-model training complete (evaluation pending)

The prospectively authorized corrective v2 family completed its one fresh
seed-1 production run: **100,000/100,000 CUDA updates** on the NVIDIA RTX 5070,
exit 0, in **3,314.40 s** outer wall time. TRAIN-only max-absolute scales were
refit from all 280,625 native action rows of the unchanged admitted 1,000-episode
dataset before the first optimizer update. The exact scale vector and full-row
action digest are retained with the run evidence below. The five distinct
20k/40k/60k/80k/100k checkpoints have the v2 format and scale buffer; CPU-only
audits confirm finite raw/EMA weights, exact buffer equality, optimizer and EMA
steps at each checkpoint update, and terminal cosine LR 0. The 1,000-row loss/LR
log is finite. The exact required 24-file suite passed **239 tests before and
after** training.

Experiment logs/config/source archive:
`outputs/exp/20261119_235008_plug_orientation_base_scaled_v2/`; checkpoints:
`outputs/ckpt/20261119_235008_plug_orientation_base_scaled_v2/`; immutable
registration, exact commands, timings, input/source hashes and checkpoint audit:
`outputs/exp/20261119_235008_plug_orientation_base_scaled_v2_envelope/`.
Admitted TRAIN data, frozen catalog identities, v1 contract, and all five v1
checkpoints were hash-identical pre/post. No trained-policy inference, validation
scene, fixed-test scene, or holdout outcome was run in Round 100. Round 101
subsequently completed v2 validation at 19/50; the v2 fixed-test gate remains
closed. The original v1 selection remains **30/50**, its fixed test remains
closed, and v1 is not replaced or qualified. Qualification still requires
≥80/100 on an authorized complete fixed test. No Constrain/09 training is
authorized.

## Goal-loop Round 99 — minimal v2 checkpoint/admission correction

Fixed the v2 checkpoint boundary: both raw `agent` and `ema_agent` persistent
`action_scale` buffers are checked for exact float32 shape-(7,) equality with the
TRAIN-derived scale before adapter construction or environment/backend access.
This prevents strict state loading from silently replacing the verified scale
with a different serialized value. Regression tests cover mismatches in each
buffer; strict EMA restore separately proves inference uses EMA parameters while
the raw checkpoint state remains unchanged. A genuine expected-red run failed
for both mismatched buffers before the fix.

The complete required 24-file suite passes **239 tests**. A bounded fake-catalog
integration exercises the public v2 fixed-test entry point with five distinct
complete candidate checkpoints/summaries, a valid ≥40/50 selection, and confirms
the shared rollout backend is called exactly once with the v2 test authorization.
Only the rollout backend is stubbed; v2 checkpoint, selection, catalog, summary,
and digest validation remain live. Evidence, test file list, hashes, raw red/green
logs, and source archive are in
`outputs/exp/20261119_234116_plug_scaled_minimal_review_fix/`.

Offline implementation/test work only: no trained checkpoint query, native
control, optimizer step, production run, test/holdout outcome, commit, or push.
The v1 contract hash remains `ef3f13b077f50605c9459e7b86fc2c3345d505237aa1f221427d61fa0af52517`;
all five existing v1 checkpoint file hashes are recorded in the evidence bundle.
No training or qualification claim/authorization is added.

## Goal-loop Round 98 — action-scaled v2 model family (implemented, offline only)

Added a distinct `plug_orientation_fixed_base_dp_action_scaled_v2` family and
prospective addendum. The v1 contract, checkpoints, 30/50 selection, summaries,
and closed fixed-test gate were not changed. The v2 family keeps observations,
native actions/controller, dataset windows, architecture, seed, 100k update
budget, optimizer/schedule, EMA, horizons, and stride 8 fixed; its only model
change is symmetric per-action-channel max-absolute TRAIN normalization and the
matching registered-buffer inverse transform at inference. Evaluation interfaces
and selection bind family, format, scale statistics, dataset, addendum, and frozen
catalog identities; v2 requires its own five complete validation summaries and
≥40/50 before the fixed-test interface can run. Qualification still requires
≥80/100. No fallback from v2 to v1 selection is accepted.

TRAIN-only scale fitting read every action row from the admitted 1,000-episode
HDF5 (280,625 rows; no validation/test fitting):
`[0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
0.283505916595459, 0.2955075800418854, 0.30000001192092896, 1.0]`.
Full-row action SHA-256 is
`fa8db7e570b15e475385a0398af66e142439cf9d7246d7ebe427b8400b1a44f3`; HDF5,
JSON, dataset-manifest, catalog-set, and addendum hashes are recorded in
`scale_statistics.json`. The fit has no offset, clipping, or inference gain.

Implementation and exact 24-file regression suite are in
`outputs/exp/20261119_232310_plug_action_scaled_development/`; **229 passed**.
The suite includes a synthetic CPU wrapper normalization/decode check and the
pre-existing official uninitialized-CPU sampler smoke, but queried no trained
checkpoint. There were zero optimizer steps, native controls, demo collection,
trained-policy queries, or fixed test/holdout access. This v2 correction is not
the original official identity-action baseline, has not been trained or qualified,
and authorizes no Constrain training.

## Goal-loop Round 97 — original gate repair and action-scale audit (offline)

The original Base selection and authorized-test gates now inspect the execution
cadence in every saved candidate summary. Explicit stride other than the frozen
stride 8 rejects; absent stride is treated as 8 only for complete pre-field
legacy summaries. Direct test evaluation likewise rejects non-8 stride before
policy/environment construction. Genuine RED and final GREEN evidence (exact
23-file suite, **221 passed**) are in
`outputs/exp/20261119_225137_plug_action_scale_audit/`. Historical summaries,
selection, checkpoint, traces and Round 96 outcomes remain untouched.
The failed stride-1 arm's additive Round 97 disposition is
`candidate_rejected` / `diagnostic_only` / `candidate_qualified: false`.

The same round performed offline analysis only: frozen TRAIN HDF5/JSON and the
existing 50 saved 80k fixed-validation per-control traces. It opened no test
catalog/outcome and performed no native controls, model query or optimizer step.
Across 280,625 TRAIN action rows, raw motion-channel standard deviations are
0.0189–0.1613, while gripper std is 0.9715. A TRAIN-only late-insertion band
(charger-center residual X −25…−5 mm and world-YZ radius ≤5 mm) contains 27,231
source rows. The 20 saved 80k failures' last-50 mean absolute X action is
0.00630 versus 0.02065 source +X action in that band; 19/20 have the known
~−16 mm X stall. This makes action representation precision a falsifiable
hypothesis, not a demonstrated root cause. Installed Pinocchio's exposed IK
default is `eps=1e-4`; the native controller call omits `eps` and sets
`max_iterations=100`, so a 1 mm dead-zone is not assumed.

The offline analysis itself made zero model calls. The separate 23-file suite
does include the existing uninitialized CPU DP Agent sampler smoke (one
synthetic random-agent forward call); no trained or candidate checkpoint was
queried. This test-scope detail is disclosed in the Round 97 evidence.

Proposed next candidate only (not trained or authorized in this round): a
separate `plug_orientation_fixed_base_dp_action_scaled_v2` format with
per-channel symmetric max-absolute scaling fit on TRAIN only, no clipping,
unchanged identity 46-D observations/native action inverse/controller, and one
fresh seed-1 100k run. Keep the v1 30/50 original baseline and closed test gate
permanent. If prospectively approved, evaluate all five checkpoints on the same
full 50-scene fixed-validation catalog with the original selector; only ≥40/50
can open the fixed 100-scene test, and qualification still requires ≥80/100.
This would be a bounded representation correction, not a retrospective original
baseline pass or hyperparameter sweep. Full tables, exact scale vector, hashes,
commands, test list/logs and roundtrip check are in the audit directory above.

## Goal-loop Round 95 — offline diagnosis of saved Base validation traces

Analyzed the existing 250 `episode_*.events.jsonl` files from the five frozen
fixed-validation cohorts; no native controls, model/checkpoint queries, training,
new demos, or test-catalog access occurred. The additive report, event hashes,
execution record, command template (containing placeholders), and 23-file test log are in
`outputs/exp/20261119_220651_plug_base_validation_failure_analysis/`.

The frozen 80k winner and validation gate are unchanged. Outcomes remain 2/50,
17/50, 27/50, 30/50, and 27/50 at 20k/40k/60k/80k/100k; fixed-test opening
still requires 40/50, and neither 100-scene test was opened. All 60k–100k
failures have a saved control with object-goal distance ≤50 mm and full
quaternion angle ≤0.2 rad, but miss the official 5 mm position gate. At the
closest-position control, a dominant cluster has about −16 mm world-X charger
residual (Y/Z within 2 mm): 20/23 failures at 60k, 19/20 at 80k, and 21/23 at
100k. This late insertion-position stall is consistent with contact/insertion
geometry or policy correction behavior, not proof of a unique cause. State-derived
position and sign-invariant full-quaternion errors were checked against native
info at every control. TCP-frame object-relative translation/quaternion and their
drift after first proximity are included as kinematic transport evidence only.
Saved traces lack force/contact or secure-grasp status; lift is treated only as
transport evidence. The last-50 gripper outputs are
bimodal near ±1, so soft gripper outputs are not supported as the leading cause.

On the same 50 scene seeds, 80k/100k both pass 20, only 80k passes 10, only 100k
passes 7, and both fail 13. This does not justify changing checkpoint selection
or interpreting training loss as quality. Proposed next falsifiable seam, not
executed: under the standing continuation authorization, preregister a six-scene
validation-only comparison of the current execution stride 8 with stride 1,
holding the 80k checkpoint, prediction chunk 8 and native action codec fixed.
Choose the first four failing and first two successful validation indices in
sorted order; advance only with at least two recoveries and no regressions, then
require a new complete 40/50 validation result before opening the fixed test.
This is a separately named validation-informed execution correction, not a
retroactive pass of the original baseline or a pure cadence-causality test:
the number of RNG-consuming sampler calls changes. Neither test is used here.

## Goal-loop Round 94 — Base fixed-validation gate (FAIL; fixed test unopened)

The registered five-checkpoint validation ran all five complete 50-scene cohorts
once on fixed-validation seeds 2026124100–2026124149: update 20k **2/50**, 40k
**17/50**, 60k **27/50**, 80k **30/50**, and 100k **27/50**. The registered
maximum-success/latest-update-tie selector locked update 80k at **30/50**, below
the required **40/50** gate. Thus the fixed Base test was **not opened**, Base is
not qualified, and no Constrain/09 training is authorized by this result. The
random-final-test catalog remains unopened. Total native execution was **115,851
controls** in **2,789.52 seconds**; the exact 22-file regression suite passed
**216 tests both before and after**. Saved terminal native states were independently
rechecked against the official ≤5 mm position and ≤0.2 rad full-quaternion
predicate. Full run, per-control traces and immutable validation selection:
`outputs/exp/20261119_205526_plug_orientation_base_validation/`; outer
preregistration, hashes, timing, independent audit, suite logs and artifact
inventory: `outputs/exp/20261119_205526_plug_orientation_base_validation_envelope/`.

## Goal-loop Round 93 — authorized from-scratch Base production training

Completed the frozen seed-1 official Base run on the admitted 1,000-episode
dataset: **100,000/100,000 updates**, batch 1,024, on the RTX 5070/CUDA, in
3,348.13 s outer wall time (3,333.997 s trainer time), exit 0. The run used the
contract-bound 46-D/7-D DPAgent, AdamW/cosine/EMA configuration and emitted the
required 1,000 finite loss/LR rows at 100-update intervals. The first/last 100
logged loss medians were 0.0097254 / 0.000912935 (diagnostic only; not a policy
quality or qualification gate). Pre- and post-run registered 22-file regression
suites both passed; the run's source archive, 22-file live closure, HDF5/JSON/
dataset manifest, contract, four catalog manifests and catalog-set identities
were checked. All four reset-only catalog `scenes.h5` files also match the
scene hashes in their frozen manifests. No evaluation or fixed-test outcome was
opened.

The five `validation_candidate` checkpoints are retained at
`outputs/ckpt/20261119_194656_plug_orientation_base/` (20k, 40k, 60k, 80k,
100k). Full checkpoint audit passed: expected format/config/provenance, finite
raw and EMA tensors, optimizer and EMA steps matching each checkpoint update,
terminal cosine LR 0, and five distinct full-file SHA-256 identities. Evidence,
exact command/wrapper, logs, preregistration, runtime, source/input hashes and
audit records are in the sibling
`outputs/exp/20261119_194656_plug_orientation_base_envelope/`; training logs and
resolved configuration are in `outputs/exp/20261119_194656_plug_orientation_base/`.

**Disposition:** the frozen Base training budget is complete, but Base is not
policy-qualified. Next execute the five-checkpoint **fixed-validation 50-scene**
evaluation using the existing frozen catalog and the registered selector. Keep
the fixed Base test unopened unless validation selects a checkpoint with at
least 40/50 successes; Base qualification still requires at least 80/100 there.
No Constrain/09 work is authorized by this result.

## Round 90 — fail-closed Base/data corrections (offline only)

Fixed four concrete public-seam defects without changing native control values:
training now re-iterates a nonempty loader until the frozen update target (and
rejects an empty loader); checkpoint evaluation verifies current contract, data,
dataset-manifest, and complete catalog-bundle identities before constructing an
environment or policy; validation requires five distinct checkpoint byte
identities and locks the fixed-validation catalog plus each complete hashed
evaluation summary (counts are not accepted as free-standing JSON); and the collector retains the
real 600-control-cap failure as a failed candidate with its partial source count
and continues at the next reserved seed. Unknown exceptions still abort.

Collection wall time is enforced with a main-thread `SIGALRM` around the complete
native callback plus monotonic guards before each expert/replay control, after the
callback, before episode admission, and before final consolidation. The existing
21600-second contract outer timeout remains an upper backstop; the collection
invocation below uses the stricter 18000-second contract deadline. No collection,
native control, replay, production training, or policy evaluation ran in this
round. The single-threaded 22-file regression suite passes **215 tests** in 20.79 s; the
official CPU sampler smoke is included and does not perform optimizer updates.

## Round 89 — Ticket 08 data/Base implementation (offline only)

Implemented the contract-bound pipeline in `plug_orientation_data.py`,
`plug_orientation_scenes.py`, and `plug_orientation_base_dp.py`. The collector
archives its source closure and registration before its first teacher callback,
uses only ordered candidates `2026121000..2026122499`, retains every attempt,
independently rechecks saved source/replay full-pose outcomes plus exact raw action
and initial-state/observation identity, and consolidates the first 1,000 eligible
episodes without rewriting source trajectories. Scene catalogs are reset-only,
checksum-bound and split-disjoint. The Base dataset keeps episode boundaries and
official 2/16/8 window semantics; the from-scratch Agent is 46-D/7-D and has
strict checkpoint identity, five-checkpoint validation selection, a 40/50 test
gate, and a separate fixed-test opening function. No Peg checkpoint or Constrain
weights are consumed.

The production orchestration is available from the repository root (choose fresh
UTC-stamped paths; all directories refuse overwrite). These are instructions,
not commands executed in Round 89:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate /home/xuwj/.miniconda3/envs/maniskill
RUN_ID=$(date -u +%Y%m%d_%H%M%S)
timeout --signal=TERM --kill-after=20s 18000s python -m examples.baselines.constrain.plug_orientation_base_dp collect-fixed \
  --dataset-dir "demos/PlugCharger-orientation-fixed-1000_${RUN_ID}" \
  --exp-dir "outputs/exp/${RUN_ID}_plug_orientation_collection"
python -m examples.baselines.constrain.plug_orientation_base_dp freeze-catalogs \
  --output-root "outputs/exp/${RUN_ID}_plug_orientation_catalogs"
timeout --signal=TERM --kill-after=20s 25200s python -m examples.baselines.constrain.plug_orientation_base_dp train \
  --data "demos/PlugCharger-orientation-fixed-1000_${RUN_ID}/trajectory.state.pd_ee_delta_pose.physx_cpu.h5" \
  --manifest "demos/PlugCharger-orientation-fixed-1000_${RUN_ID}/trajectory.state.pd_ee_delta_pose.physx_cpu.json" \
  --catalog-root "outputs/exp/${RUN_ID}_plug_orientation_catalogs" \
  --exp-dir "outputs/exp/${RUN_ID}_plug_orientation_base" \
  --ckpt-dir "outputs/ckpt/${RUN_ID}_plug_orientation_base"
timeout --signal=TERM --kill-after=20s 21600s python -m examples.baselines.constrain.plug_orientation_base_dp evaluate-validation \
  --ckpt-dir "outputs/ckpt/${RUN_ID}_plug_orientation_base" \
  --catalog "outputs/exp/${RUN_ID}_plug_orientation_catalogs/fixed_validation" \
  --output-root "outputs/exp/${RUN_ID}_plug_orientation_validation"
VALIDATION_SUCCESS=$(python -c 'import json; print(json.load(open("outputs/exp/'"${RUN_ID}"'_plug_orientation_validation/validation_selection.json"))["validation_successes"])')
if [ "$VALIDATION_SUCCESS" -ge 40 ]; then
  SELECTED_CHECKPOINT=$(python -c 'import json; print(json.load(open("outputs/exp/'"${RUN_ID}"'_plug_orientation_validation/validation_selection.json"))["selected_checkpoint"])')
  timeout --signal=TERM --kill-after=20s 21600s python -m examples.baselines.constrain.plug_orientation_base_dp evaluate-fixed-test \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --catalog "outputs/exp/${RUN_ID}_plug_orientation_catalogs/fixed_base_test" \
    --selection "outputs/exp/${RUN_ID}_plug_orientation_validation/validation_selection.json" \
    --output-dir "outputs/exp/${RUN_ID}_plug_orientation_fixed_base_test"
fi
```

The test command in this sequence is gated by the frozen ≥40/50 validation
selection and the selected checkpoint SHA in `validation_selection.json`.

The 100-scene randomized final-test catalog is frozen but has no Ticket 08
evaluation entry point. Reserved adaptation seeds are metadata only. No native
collection, production optimizer update, policy evaluation, or qualification ran
in Round 89. New data/Base modules passed **4/4** and **9/9** focused tests. The
20-file historical Peg+Plug/Base regression set passed **195 tests**; together
with the two new modules, the 22-file suite passed **208 tests** in 18.58 s (no
native simulator control). A genuine official DP Agent CPU
DDPM sampler smoke returned finite `(1, 8, 7)` output with zero optimizer steps.
The development evidence bundle, including its source archive, exact test list,
pytest logs, RED/green record, and sampler smoke, is retained at
`outputs/exp/20261119_180756_plug_base_development/`. No production collection,
training, checkpoint evaluation, or policy qualification artifact was created.

## Round 92 — fixed Base data admission and catalog freeze

Executed the frozen production collector once under `maniskill` on the unchanged
contract (`plug_orientation_contract.json` SHA-256
`ef3f13b077f50605c9459e7b86fc2c3345d505237aa1f221427d61fa0af52517`). The
dataset is `demos/PlugCharger-orientation-fixed-1000_20261119_184307/`; its
collection experiment, sibling envelope, and reset-only catalog set are
`outputs/exp/20261119_184307_plug_orientation_collection/`,
`outputs/exp/20261119_184307_plug_orientation_collection_envelope/`, and
`outputs/exp/20261119_184307_plug_orientation_catalogs/`. The source archive and
pre-registration were sealed before the first teacher callback. The command
completed once with exit 0 in **2,581.512 s** under the 18,000-s wall cap.

The ordered candidate range actually used was **2026121000–2026122065**:
1,066 attempts yielded the first **1,000 eligible** source+raw-action-replay
episodes (93.81%). All 66 non-admissions remain in their raw attempt folders:
61 failed the official full-pose success predicate and 5 were known MPLib
planning failures. Measured expert/replay controls were **294,461 / 294,230**;
maximum accepted source episode length was 400 controls. The independent audit
recomputed all 1,000 official position/quaternion gates, finite native 7-D
actions, finite T+1 46-D observations and states, source/replay action and
initial-state/observation agreement, source-HDF5 value identity, seed ordering,
and candidate/evaluation split disjointness. The official `PlugBaseDataset`
constructor admitted all 1,000 trajectories and **279,625** episode-local
windows. No filtering beyond the contract's first-eligible-in-seed-order rule
occurred.

Authoritative hashes: training HDF5
`d909986dd2319f41e5a3e098b13744cb6eda5e3caf2b779d66fe3340165d11c1`, training
JSON `190dd31f0787b58871fe4963e36931ab65c33ab7b3eb05a6dd160a29a8b8088f`, and
dataset manifest `a6744fd41d388dd2dce0d85f9f80a0bf2f74ce731600c5d837b492f2d172c850`.
All **30** registered source files match both their archived and current bytes
after collection; no ManiSkill source or dependency changed.

All four catalogs were frozen before optimization and contain reset-only scenes:
fixed validation 50, random validation 50, random final test 100, and fixed
Base test/retention 100. Catalog-set SHA-256 is
`3f75e6ec5d3729e142e87117e34bfd00a39f921283c21000486e30800f34481f`. An
independent reset/restore audit checked all **300** finite 46-D observations;
maximum restored-observation error was `1.1921e-7` (tolerance `1e-6`). No policy
was evaluated; frozen target-yaw roundtrip also passed all 300 scenes (maximum
wrapped error `1.45e-8` rad, tolerance `1e-6`). No optimizer update/checkpoint
was created. The exact 22-file
regression suite passed **216 tests** both before and after native collection;
post-run source hashes match pre-run hashes. Detailed admission, timing, source
and catalog identities are in the sibling collection envelope.

**Next execution:** the authorized next slice is the contract's from-scratch
seed-1 official Base run (100,000 updates, batch 1,024), followed by five
fixed-validation checkpoint evaluations. Do not open fixed Base test unless the
selected validation score is at least 40/50; Base qualification still requires
at least 80/100 on that test. No training has started in Round 92. From the repo
root, after choosing a fresh `TRAIN_ID=$(date -u +%Y%m%d_%H%M%S)`, the exact
production command is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
TRAIN_ID=$(date -u +%Y%m%d_%H%M%S)
timeout --signal=TERM --kill-after=20s 25200s python -u -m examples.baselines.constrain.plug_orientation_base_dp train \
  --data demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --manifest demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.json \
  --catalog-root outputs/exp/20261119_184307_plug_orientation_catalogs \
  --exp-dir outputs/exp/${TRAIN_ID}_plug_orientation_base \
  --ckpt-dir outputs/ckpt/${TRAIN_ID}_plug_orientation_base \
  --device cuda
```

The collector's `dataset_manifest.json` is adjacent to the training JSON and is
required by this command. Validation and any gated fixed-test run must use fresh
timestamped output directories and the frozen CLI gates; no test result is
available yet.

## Round 88 — independently audited confirmation and frozen Ticket 08 contract

The prospective Round 87 confirmation completed all 20 registered cases under
the 9,000-second outer timeout: fixed expert/replay **10/10**, random
expert/replay **10/10**, **12,194 controls** (6,097 expert + 6,097 replay), and
53.915 seconds outer elapsed. Independent read-only audit
`outputs/exp/20261119_164619_plug_nativeEE_v4_sealed_confirmation/independent_audit.json`
revalidated all 28 preregistered source SHA-256 identities against both live
and archived bytes, checked every source HDF5's finite native 7-D actions,
finite 46-D T+1 observations, and recomputed the official full-position/full-
quaternion success predicate from saved charger/receptacle states. All 20 source
episodes pass. All 20 source/replay HDF5 pairs independently confirm exact raw
action equality, successful outcomes, finite state/observation T+1 rows, and
exact full state/observation agreement at row 0 (max error 0); the full source
and replay trajectories also match exactly. All 6,097 recorded source hint rows
were independently
recomputed from the corresponding post-action charger state and registered goal
yaw; maximum absolute error was zero. Random targets are ten distinct values with bin occupancy
`[2,2,2,2,2]`, including endpoints; this is not ten occupied bins. Expert
trajectory lengths were 249/309/349 controls (min/median/max).

**Disposition: Ticket 07 confirmation PASS; user-conditional Ticket 08 start
authorization is satisfied.** This was a nonblind confirmation on the same
preplanned cases after historical numeric outcomes had been viewed. It is not a
blind generalization result or policy/Base qualification. Exclude these 20 cases
and all historical diagnostic/preflight data from future training. The frozen
machine-readable collection, Base training, split, evaluation, and future-gate
contract is [`plug_orientation_contract.json`](plug_orientation_contract.json).
It freezes 1,000 accepted fixed-yaw native EE demonstrations from at most 1,500
attempts, then a from-scratch official 46-D Base DP run for 100,000 updates;
Base qualification requires at least 80/100 on its untouched fixed test before
any Constrain/09 training. No data collection or training ran in Round 88.

The independent decision is `gate_decision.json`; this gate passes preflight
only and does not qualify the policy or method. The prior prospective protocol
and historical pauses below are retained as history, not current blockers.

## Round 87 — fail-closed v4 prospective sealing (offline implementation only)

The v4 runner now writes `preregistration.json` and archives its finite declared
source closure before its first `run_case` callback. It seals the exact planned
seeds/yaws, control/time envelopes, official 5 mm / 0.2 rad gate, acceptance
criteria, runtime versions, and source SHA-256 identities; it refuses existing
registration/result/archive outputs and checks registration plus live/archive
source identities before each case. This is explicitly a nonblind confirmation:
the prior Round 86 numeric outcomes had already been viewed. Ten distinct random
yaw targets span five bins with planned occupancy `[2,2,2,2,2]`; accepted
expert+replay occupancy and endpoint coverage are recorded separately. Round 86
remains quantitatively passing but **not prospectively registered**, and is not
retroactively certified. No native case, replay, training, or model operation is
part of this implementation round.

Offline verification: the two new fake-callback runner tests pass; the exact
19-file Peg+Plug subset passes **192 tests** in 22.46 s (23.66 s outer). Logs,
command, UTC timing, and test list are retained at
`outputs/exp/20261119_163707_plug_v4_preregistration_fix/`. These
exercise registration-before-callback, all 20 planned cases, archived source
hashes, coverage reporting, and refusing an existing registration without
changing it. No native physics or case callback into the real expert was run.

## Round 86 — v4 align-budget experiment (quantitative pass; protocol NOT certified)

The only teacher-logic change was raising `MAX_ALIGN_CONTROLS` from 100 to 180;
the 600-control episode cap, lift/insert limits, controller, targets, geometry,
physics, and official 5 mm / 0.2 rad full-pose predicate stayed unchanged. The
teacher and HDF5 source identity are labeled v4; the runner supports fixed-yaw
pair and the preplanned 20-case fixed-then-random order. The old v3 artifacts
were not modified.

The old 19-file regression set passed **193 tests** (23.45 s, then 25.14 s after
the final runner directory-guard correction). The budget assertion produced the
expected RED first (old expected value 100 vs actual 180), then passed GREEN.
Three fixed pair cases were not run: the first runner invocation aborted with
zero actions because a preregistered empty demo directory was rejected; this
setup-only attempt is preserved at `outputs/exp/20261119_161229_.../`. It was
corrected before any action; no physical case was retried.

The v4 pair ran seeds `2026113003`, `2026113001` once each, both source and exact
replay passing official evaluation: **658 expert + 658 replay = 1,316 controls**,
5.646 s. The preplanned full set then ran all fixed seeds 3000–3009 followed by
random seeds 3000–3009 at ten distinct evenly spaced yaw targets in [−π/8,+π/8],
covering the five-bin design with planned occupancy `[2,2,2,2,2]`, including both
endpoints. All
20 experts and all 20 independent raw-7D replays passed: fixed **10/10**, random
**10/10**, exact replay acceptance **10/10** in each profile, all ten random yaw
bins (including both endpoints), **6,097 expert + 6,097 replay = 12,194 controls**,
50.609 s. Independent HDF5 audit verified finite 46-D T+1 observations, finite
7-D exact action matches, T+1 states, and exact initial state/observation matches
(max absolute error 0 for every case); all replay outcomes meet the unchanged
full-pose gate. Raw traces and audit are under
`demos/PlugCharger-nativeEE-v4-preflight_20261119_161534/` and
`outputs/exp/20261119_161534_plug_nativeEE_v4_preflight/`.
Captured runner stdout is empty (the runner writes structured JSON); per-case
JSON/HDF5, aggregate JSON, and the independent audit JSON are the retained records.

**Critical execution-protocol exception:** the planned preregistration JSON was
not actually persisted before either physical stage (the shell heredoc used with
`conda run` did not execute). This was discovered during post-run audit. No
post-hoc preregistration is fabricated: `independent_audit.json` explicitly
marks `quantitative_gates_pass: true`, `protocol_gate_pass: false`, and pins source
identities only post-run. Therefore these excellent physical outcomes are not a
certified Ticket 07 PASS and do not authorize training under the frozen protocol.
No further physical attempt, training, dependency change, ManiSkill-source edit,
commit, or push was made. Continue by correcting the registration/logging path;
do not treat this round as a formal PASS.

**Round 88 annotation (no historical result rewrite):** the Round 86 phrase
“all ten random yaw bins” is a bin-count misstatement. The actual design has five
bins with planned/accepted occupancy `[2,2,2,2,2]` and ten distinct yaw targets;
Round 86 remains unregistered and is not retroactively certified. The fresh
Round 88 confirmation independently rechecked that same five-bin coverage.

## Round 85 — v3 object-pose lift gate (diagnostic mixed; no qualification)

The internal v3 teacher removes the pre-lift native `is_grasping` abort and
treats that classifier as per-control diagnostic only. It retains the existing
12-control closure, 80-control lift budget, unchanged EE targets/controller and
the actual charger full-pose tracking gate (≤1.5 mm and ≤0.03 rad). The lift
target is the live initial charger pose translated +30 mm in world Z; no contact,
physics, force, friction, stiffness, waypoint, or official-success threshold was
changed. Traces/results record the object start/end height, progress and full-pose
error. This teacher-semantic change is not an admission or threshold waiver.

The exact 19-file suite passed **193 tests** after a behavior RED. Both authorized
fixed cases and one independent replay of each exact raw 7-D action stream ran
once. Total: **1,098 controls** in 5.165 monotonic seconds (outer start UTC was
not captured and is not reconstructed). Seed `2026113003` did reach the lift:
16 lift controls moved the charger from **11.998 mm** to **40.761 mm** height
(**28.763 mm** progress; target **41.998 mm**); final lift error was **1.257 mm /
0.000684 rad**. It then exhausted 100 align controls and failed at 78.651 mm
official position error; exact replay also failed. Its first recorded native
`is_grasping=true` was the first lift control, after 121 pre-lift controls. The
mechanism for object motion is not identified.

Seed `2026113001` succeeded officially after reach/grasp/close/lift/align/insert/
settle counts **90/39/12/16/79/66/10**; final error was **0.294 mm / 0.000442
rad**. Its exact replay also succeeded. Both replays had identical recorded
actions, 46-D observations, T+1 alignment and exact initial observation match
(max error 0); source closure and wrapper hashes match post-run. The result is
**1/2**, not Ticket 07 PASS or training admission. Execution remains in progress:
standing authorization permits v4 pair validation, then the full 20-case gate if
both pairs pass, then training only after independently verified full PASS.
Preserve all failed traces and continue
physical diagnosis; do not pool or rewrite prior rounds. Evidence:
`outputs/exp/20261119_154809_plug_ee_lift_v3_pair/` and
`demos/PlugCharger-EE-lift-v3_20261119_154809/`.

## Round 84 — streamed native EE teacher v2 (diagnostic FAIL)

The experiment-only teacher streams each planned FK waypoint once instead of
requiring every intermediate waypoint to satisfy a tight tracking tolerance. It
then holds the final planned TCP target for 10 controls and closes for 12 while
continuing to target that same planned pose. Intermediate tracking errors are
diagnostic only; the actual `is_grasping` requirement and official 5 mm / 0.2 rad
gate are unchanged. The legacy strict trajectory-source identity is distinguished
from the new implementation by `experimental_online_ee_teacher_v2` and
`native_ee_stream_final_target_hold_v2` metadata.

The exact required 19-file suite passed **190 tests**. Both preregistered native
cases (seeds 2026113003 and 2026113001) plus independent exact-action replays ran
once, totaling 524 controls. Both reached `charger_not_grasped` before attempting
a lift. At the final closure sample, planned-target TCP position errors were
approximately 5.25/3.19 mm for seeds 2026113003/2026113001; full-rotation errors
were 0.0101/0.00644 rad. Native grasp remained false throughout closure; this
does not establish physical pickup impossibility. The source/replay initial
state and observation matched exactly; actions, 7-D shape and T+1 alignment
matched. Neither expert nor replay passed the unchanged official pose predicate.
This v2 candidate is diagnostic only, does not pass Ticket 07, does not admit
data, and is not ready for the full 20-case validation. Next diagnosis should
focus on grasp-pose/contact geometry and the final 10+12 hold behavior, not loosen
the tracking or official success gates. Complete records and hashes are in
`outputs/exp/20261119_153718_plug_ee_stream_v2_pair/` and
`demos/PlugCharger-EE-stream-v2-diagnostic_20261119_153718/`.

## Round 83 — corrected native EE teacher pair (diagnostic FAIL)

Fixed two implementation defects in the experiment-side teacher: bounded targets
now use the shared `PoseValue` accepted by the native action encoder, and recorded
46-D observations are validated as flat vectors (structured observations are
flattened without adding an extra batch dimension). Replay now checks source/replay
state and observation row 0 (≤1e-5) before acting and still replays failed experts
as explicitly diagnostic cases. A real native-controller test passes through
`_step` with a 7-D gripper command and nontrivial TCP roll/yaw; it verifies the
actual `compute_target_pose` result and no clipped command. Planner RNG seeding is
explicitly reset to `seed+1` before/after reset and before each planner.

The authorized fixed-yaw seeds 2026113003 and 2026113001 were each attempted once
on CPU with the unchanged 600-control cap. Both stopped at `grasp_tracking_failed`
(212 and 207 teacher controls; 419 total). The official final errors were
290.564 mm / 0.483186 rad and 193.191 mm / 0.939333 rad, respectively. Each
failure retained its full per-control action/state/phase trace. Independent exact
action diagnostic replay was then run on each saved source: the source and replay
initial states matched; initial observation max error was 0; all 7-D actions were
byte/value-identical; T+1 states and observations were retained (213 and 208
rows). Neither replay reached the native success predicate. These expert failures
are not converted into successes and do not pass Ticket 07 or admit training data.

The exact 19-file suite passes **188 tests**. Expected-red reproduced the real
SAPIEN-Pose/`PoseValue` API exception; native-controller calibration and fake
HDF5/torch observation tests pass. Traces, failed and corrected replay diagnostics,
logs, timing, test list and hashes are under
`outputs/exp/20261119_152051_plug_ee_teacher_native_pair/`; source and replay HDF5s
are under `demos/PlugCharger-EE-teacher-native-pair_20261119_152051/`. Ticket 07
remains blocked; no training, dependency change, ManiSkill source edit, commit, or
push occurred.

## Round 82 — bounded online EE-teacher implementation (incomplete native setup)

Added `plug_orientation_ee_expert.py` and tests for actual native controller
inversion, bounded TCP targets, full-quaternion charger-relative goals, and the
dry-run-only joint planner seam. The 19-file required suite passed **185 tests**.
The preregistered fixed two-case design retained the historical fail/pass pair,
unchanged target range and official full-pose predicate. Native execution stopped
on the first case before any physical step: the initial MPLib binding failure and
two subsequent pose-type errors each produced zero controls/replay. Both
diagnosed zero-action repair attempts are exhausted; no further native attempt
was run. Artifact and error details are in
`outputs/exp/20261119_145706_plug_ee_teacher_diagnostic/` and
`demos/PlugCharger-EE-expert-diagnostic_20261119_145706/`. This produces no
expert/replay success evidence and does not pass Ticket 07, admit training data,
or authorize training. Prior Round 81 remains FAIL and unchanged.

This is authorized groundwork for Ticket 07. Peg's negative formal result remains
unchanged. As of Round 80, the user has authorized proceeding to Ticket 08 training
only after Ticket 07 receives an independently verified PASS; no training or policy
evaluation is authorized before that gate. Ticket 09 remains unauthorized.

## Round 81 — deterministic official-expert preflight (gate failed)

The sole preregistered 20-case run used paired seeds `2026113000..2026113009`,
official solver defaults (15° grasp tilt; no experiment-side override), and the
unchanged native full-pose predicate (5 mm and 0.2 rad). All four bounded stages
completed. Independent audit found fixed expert **8/10**, random expert **5/10**;
among successful expert trajectories, EE replay accepted fixed **4/8** and random
**3/5**, below the required 95% in both profiles. Accepted random yaws occupy only
3/5 planned bins and include no endpoint. **Ticket 07 fails; no preflight data is
training-admitted and Ticket 08 must not start.** The six rejected EE replay traces
and all expert attempts, including planner failures, remain intact for the next
diagnosis slice. Full results, control counts, hashes, and audit are recorded in
`outputs/exp/20261119_143359_plug_orientation_preflight_deterministic_envelope/`
and `outputs/exp/20261119_143359_plug_orientation_preflight_deterministic/`;
source demonstrations/replay products are in
`demos/PlugCharger-orientation-preflight-deterministic_20261119_143359/`.
Official solver false outcomes were fixed seeds 2026113000/3005 and random seeds
2026113000/3001/3002/3005/3009 (including both random endpoints). EE replay false
outcomes were fixed source IDs 3/6/7/8 and random IDs 3/6. Exact terminal errors
and all per-action traces are retained in the audit and episode HDF5s.
The 18-file suite passed 180 tests before execution. No model training, policy
evaluation, dependency installation, ManiSkill source edit, or commit occurred.

Round 80 closes the replay provenance blocker: source replay selection accepts
only explicit boolean success declarations from attempt records or source
metadata and raises on missing or contradictory declarations. Replay's true
destination initial capture is gated on the environment's actual
`pd_ee_delta_pose` control mode. The 18-file offline suite passed 180 tests;
details and raw logs are recorded in `OUTPUTS.md`. This is not native
preflight evidence and does not pass Ticket 07 or authorize training.

## Environment boundary

`plug_orientation.py` registers `PlugChargerOrientation-v1` and
`PlugChargerRandomOrientation-v1` as experiment subclasses. It uses the native
`panda_wristcam`, `pd_ee_delta_pose` default for policy use, existing charger and
receptacle geometry, and the native full-pose success predicate unchanged:
position error ≤5 mm **and** full quaternion rotation angle ≤0.2 rad. The stock
expert is a separate `pd_joint_pos` solve.

The native receptacle is sampled with yaw near π; the native target pose
`receptacle.pose * RotZ(π)` therefore has yaw near zero. Fixed target yaw is 0;
random target yaw is stratified in [−π/8,+π/8], matching the official target
range. The sampled receptacle center is preserved, and the actual receptacle is
rotated to `target_yaw - π`; the native goal is reconstructed at that same center
with `target_yaw`. Charger initialization remains unchanged. Hint is live world
charger local +X yaw minus target-goal local +X yaw, wrapped and encoded
`[sin(error), cos(error)]` in radians.

The base state observation keeps agent state, TCP pose, current charger pose,
native goal position, and the task's constant geometry; it omits both
`receptacle_pose` and `goal_pose`. Cameras/render-only observations are not added.
Task state serialization includes target yaw and goal position so restoration
reconstructs a coherent goal orientation without putting target yaw in policy
observations. Native evaluation implementation is not changed.

## Proposed next preflight (not run in this development round)

`plug_orientation_preflight.py` freezes the same 10 seeds for fixed/random paired
scene initialization. Fixed uses 0; random uses 10 distinct evenly stratified yaw
values including both endpoints. The planned denominator is 10 per profile with
no replacement. Maximum 20 expert attempts/profile, 600 physical controls per
attempt, 3,600 seconds/profile stage and 9,000 seconds whole stage. Collection
uses the stock official motion-planning solver; its internal reset is intercepted
so the preplanned target yaw is applied and its true post-reset initial simulator
state is captured. Every attempt is journaled, including failures. Converted EE
actions must be 7D and replay every expert-success trajectory independently on
CPU; native full-pose success is mandatory. No data is training-admitted by this
preflight.

Provisional evidence gates for review before native use: at least 8/10 expert
successes in each profile; replay all expert successes with no contamination or
shape/observation leak; zero false replay successes; all five yaw bins populated
in the random plan and at least four populated among replay-accepted outcomes,
including an endpoint. These thresholds do not convert the planned 10 into a
success-selected denominator. Final data counts, training budgets, splits, and
policy gates remain provisional until the native preflight is independently
reviewed; no authorization for Ticket 08 follows automatically.

Suggested future base study: 1,000 fixed-yaw expert demonstrations, seed 1,
30,000 updates, followed by a separately isolated fixed validation/test; initial
adaptation set 100, extendable only by validation evidence to a hard cap of 300.
Candidate policy gates for discussion—not frozen or training-authorizing—are
fixed validation ≥80%, randomized paired gain ≥20 percentage points, and fixed
retention loss ≤5 points. The stricter Plug position+full-rotation predicate and
the still-unverified expert/replay feasibility mean these are proposals only.
Peg's historical +12 pp gate failure, nonblind status, and single-seed limitations
remain explicit context, not a reason to weaken Plug thresholds.

Round 70 adds `plug_orientation_pipeline.py`, a single four-child fixed/random
collection-then-replay sequence, immutable preregistration/source snapshots,
combined expert/replay control accounting, shared stage/whole deadlines, and
TERM→20-second grace→KILL process-group cleanup. The worker initializes shared
replay counters, applies shared control limits, derives yaw from the saved initial
task state (no NaN fallback), and rejects non-finite converted actions/states/
observations, misaligned observation rows, leaked world-goal fields, or changed
initial orientation state. Fake-child lifecycle/budget tests and actual
worker collection/replay seams run without native physics. This is still
implementation coverage, not native preflight evidence; independent review remains
required. No native planner,
simulator control, collection, replay, training, checkpoint load, or policy query
was run. Tickets 08/09 remain blocked.

Round 71 minimal offline corrections: replay's BaseEnv audit now calls the public
`get_obs(unflattened=True)` API (default state mode is a tensor, not a dict).
Timeout phases are journaled before launch and retain drained child output, exit
status, unknown controls when no report exists, and measured completed controls
separately. The prelaunch archive now includes a finite, explicitly enumerated
set of experiment, task, solver, replay, robot, controller and recorder sources,
with package-relative archive paths and pre-child hash validation, plus selected
runtime package versions. This is not a complete environment lock or native-run
source proof; the actual run must still validate its live archived bytes.

## Round 72 — native preflight quantitative pause

The user-authorized one-time preflight ran the four-stage bounded supervisor
under a 9,000-second outer timeout. The main pipeline preregistration/archive
is `outputs/exp/20261119_102856_plug_orientation_preflight/`; the separate
immutable outer protocol, wrapper record, and independent audit are under
`outputs/exp/20261119_102856_plug_orientation_preflight_envelope/`. Dataset
attempt/replay provenance is under
`demos/PlugCharger-orientation-preflight_20261119_102856/`.

The complete fixed and stratified-random planned denominators were attempted:
**0/10 fixed expert successes, 0/10 random expert successes**. All 20 solver
attempts stopped before any environment control with the identical native
`TypeError`: `set_base_pose()` expected `mplib.pymp.Pose` but received a
float32 NumPy array from the official PlugCharger planning solution. Physical
controls were 0/12,000 in the combined expert stage. Therefore no successful
expert HDF5 episodes existed to replay; official CPU replay received zero source
episodes per profile and used zero controls. The pipeline's `complete`/replay
`complete` fields indicate only complete attempt journaling and vacuous
zero-source replay, not a passing collection or replay gate.

Reset-level paired audit passed for all ten seeds: charger initialization, goal
center, robot articulation and table states matched between fixed/random
targets, while receptacle orientation changed. This does not establish live-hint
observation, action/observation dimensions, full-pose success, or accepted yaw
coverage. The official success predicate remains unchanged (position ≤5 mm and
full quaternion error ≤0.2 rad), but was not observed on an executed episode.
All 13 registered/archived source hashes matched.

**Ticket 07 status: paused; no training contract frozen.** Retain the proposed
future study values above only as candidates: 1,000 fixed demonstrations, 30k
Base updates, adaptation 100→300 maximum, 30k balanced adaptation if separately
authorized, disjoint future seed ranges, and the proposed ≥20 pp random gain /
≤5 pp fixed-retention loss. They are not validated or authorized by this run.
Next action is parent review of a minimal experiment-only compatibility fix
before any newly authorized diagnostic. No retry, training, 08/09 work, or
ManiSkill source edit occurred. Peg's negative Ticket-06 result remains
unchanged. Full failure detail and artifact hashes are in the envelope's
`results_summary.md` and `independent_audit.json`.

## Round 73 — MPLib compatibility development (native retry blocked)

The default Plug collector now installs the existing experimental
`install_mplib_base_pose_compat` adapter from Peg before importing/calling the
stock Plug solution. It converts the official planner's seven-value NumPy base
pose and screw goal to `mplib.Pose`, and removes the unsupported
`use_point_cloud=False` keyword. The prelaunch source closure now includes this
Peg helper and `base_motionplanner/motionplanner.py` alongside the Plug solution;
the development source archive labels the adapter experimental, not upstream.
Unexpected per-attempt exceptions now retain full traceback text in the attempt
journal. The expected-red default-collection seam failed on the float32 array
binding before the adapter and passed after it. The exact 17-file suite now
passes **166 tests in 13.53 s**. No native planner, physics/control, collection, replay,
training, policy/checkpoint query, or ManiSkill source edit occurred. Round 72's
zero-success evidence and artifacts are immutable; any fresh same-20-case
preflight remains blocked pending parent review/authorization.

## Round 74 — compatibility-v2 native preflight paused

After explicit authorization for a new same-20-case run, the exact pipeline at
commit `9a930b7983a2533ab45b5d31196cdd8c03821a8f` completed all four phases
under the approved limits. Main output:
`outputs/exp/20261119_105049_plug_orientation_preflight_compat_v2/`; dataset:
`demos/PlugCharger-orientation-preflight-compat_v2_20261119_105049/`; wrapper,
pre/post suite, timing and independent audit:
`outputs/exp/20261119_105049_plug_orientation_preflight_compat_v2_envelope/`.

All 10 fixed and 10 random preregistered cases were attempted exactly once;
fixed expert success was **8/10**, random expert success **5/10**. Official
isolated CPU EE replay attempted all 13 expert-success trajectories, accepting
**4/8 fixed** and **3/5 random** (**7/13 total**). Replay acceptance is below
the preregistered 95% requirement, and random accepted outcomes cover only 3/5
yaw strata. This fails both the random expert ≥8/10 gate and the replay gate;
complete replay journals do not mean accepted replay. Initial reset matching
passed for all 10 paired seeds. Actual observation width is 46 and converted
replay actions are finite 7D. The native success predicate remains position
error ≤5 mm and full quaternion error ≤0.2 rad.

Independent audit also confirms source expert actions are 8D, all 13 saved source
episodes satisfy final-state official success, and no converted failure was
accepted falsely. The six converted failures ended with 17.27–17.73 mm position
error; two additionally exceeded 0.2 rad. Their physical/conversion cause is not
established by this audit. Accepted random coverage includes no endpoint.

Controls were **3,792 expert** and **5,056 replay**, below each combined
12,000-control cap; all stage/whole deadlines were respected (28.860 s total).
The 16-file execution source closure matches both archive and live source.
Exact historical 17-file Peg+Plug regression passed **166 tests pre and post**.

**Disposition: PAUSE; no final Ticket 07 training contract is frozen and no
training data are admitted.** Preserve all 20 attempts and all source/replay
episodes for diagnosis; do not relabel or pool Round 72's 20 old failures.
Tickets 08/09 remain unauthorized, and no model/checkpoint was loaded or queried.
The exact outcomes, runtime, audit and artifact hashes are in the main output's
`results_summary.md`/`.json` and the envelope's `independent_audit.json` and
`produced_artifacts.sha256`. No source or dependency installation was performed.

## Round 75 — bounded native replay diagnosis (development only)

Replayed the retained fixed fail case (source episode 2, seed 2026113003) and
pass case (episode 0, seed 2026113001) through source-joint and official
joint-to-EE paths, with per-control state capture. Source-joint replay already
reproduces the exact fail/pass split: episode 2 ends at 17.737 mm / 0.18859 rad;
episode 0 at 1.350 mm / 0.000761 rad. Converted EE replay is nearly identical
(17.734 mm / 0.18865 rad; 1.401 mm / 0.002681 rad). Failure source-state replay
first diverges from its recorded charger path after control 97; its initial
qpos/qvel/charger pose match recorded state 0 exactly. Therefore a converter-only
rotation convention issue is not supported; source-action replay/contact/physics
divergence is the leading diagnosis, with mechanism not yet isolated.

On the failed case, all 203 source-FK targets were also compared directly with
the converted controller's native target pose. Maximum position difference was
**0.31 µm** and rotation difference was zero, strongly ruling out the proposed
rotation parameterization/order mismatch for this pair. Actual destination TCP
tracking error relative to the source target was **5.64 / 10.14 / 14.50 mm**
median/p90/max. Source and destination public grasp checks both became true at
control 97 and remained true for the remaining 107 controls; the mechanism is
therefore a path/pose divergence during an established grasp, not observed
grasp loss. Per-control commands, controller bounds/frame, root/TCP, source FK,
native target, hint, relative pose and grasp traces are in
`outputs/exp/20261119_203000_plug_replay_diagnosis_r5/`.

Separately, offline HDF5 inspection finds converted state row 0 equals source
state row 1 exactly in 13/13 existing converted trajectories and source state
row 0 in 0/13. ManiSkill replay restoration shifts the recorder buffer after
slicing the state list; this is a saved-initial-state alignment defect, not the
cause of the live source-joint replay failure. No ManiSkill source was edited.
The asserted two-case native loop used 1,287 controls in 6.45 s; including
instrumentation runs and the earlier misindexed harness attempt, total physical
controls stayed at 5,826/6,000. Required
17-file suite: 166 passed in 12.71 s. No training/data admission or contract
freeze follows. Next diagnostic/fix seam: use retained per-step source states
for experiment-side source FK target reconstruction and test the same fail/pass
pair, while separately covering the recorded initial-state row offset. See
`outputs/exp/20261119_203000_plug_replay_diagnosis_r3/` for traces and hashes.

## Round 76 — reconstructed source-state EE candidate (development only)

Implemented the experiment-side seam and physically tested three registered
timing variants on the retained fixed fail/pass pair. Source state row 0 is
restored once; rows 1..T are converted offline to world TCP targets by Pinocchio;
the physical simulation receives native EE delta actions and the source gripper
stream. New artifacts include the true initial observation/state row and retain
T+1/T alignment. No original data was rewritten.

All six new fail/pass outcomes miss official full-pose success: terminal position
error is 17.38–18.60 mm (threshold 5 mm), despite rotation error staying below
0.2 rad. Adaptive 1–3-control, exactly-1, and up-to-2-control variants used 2,367
controls total; all are under the 6,000 round budget and 600 per trajectory.
The adaptive run tracks source FK targets to median 0.53–0.83 mm until terminal
insertion, where target error rises to 14.85–15.43 mm; the exactly-one variant
diverges earlier. The exact 17-file Peg+Plug suite passes 171 tests. This
diagnostic does **not** identify a unique contact/grasp mechanism and is not
Ticket 07 PASS, training admission, or training authorization. Do not train.
Full traces, hashes, target streams, and reports are in
`outputs/exp/20261119_230000_plug_replay_reconstruction_r76/` and the three
`demos/PlugCharger-orientation-reconstruction_r76*20261119_230000/` directories.

## Round 78 — matched grasp-tilt diagnostic (not a gate pass)

Added an experiment-side copy of the official PlugCharger expert with one
explicit parameter, local grasp-Y tilt in degrees. Its default remains the
official 15°; the 0° candidate is selected only with
`--expert-grasp-tilt-deg 0`. The approach offset, joint velocity/acceleration
limits, close-gripper timing, preinsert/refinement, insertion, and native success
predicate are unchanged. The pipeline archives this candidate source and records
the selected tilt. No ManiSkill source or dependency was changed.

Native same-scene tests ran the 15° control and 0° candidate on fixed seeds
2026113003 and 2026113001 (same target yaw and captured reset state within each
pair). Results:

| Seed | Tilt | Expert | Controls | Official EE replay |
| --- | ---: | --- | ---: | --- |
| 2026113003 | 15° | pass | 205 | pass, 2.045 mm / 0.000898 rad |
| 2026113003 | 0° | fail | 177 | not attempted (no success episode) |
| 2026113001 | 15° | pass | 226 | pass, 1.406 mm / 0.002726 rad |
| 2026113001 | 0° | fail | 193 | not attempted (no success episode) |

These planner outcomes differ from the historical fail/pass labels for these
seeds, and the independent 15°/0° calls can follow different planner paths. This
is not evidence that tilt causally explains the earlier divergence. The 0°
candidate is rejected by this small matched screen; the control is non-inferior
only on these two freshly run scenes. The separate smoke seed 2026113000 had
15° expert failure and 0° expert success, but the latter's EE replay failed
(17.702 mm / 0.174813 rad); it is retained only as contradictory diagnostic
evidence, not a gate case.

The pre-existing replay recorder defect remains unresolved: converted row 0 is
known to be offset to source row 1 in historical conversions. Therefore even
the two passing EE endpoint results do not admit the converted files for
training. Source-joint physical replay and full 07 expert/replay denominators
were not completed; failed expert attempt action trajectories are not currently
saved by the collector. No training was started. The native artifacts and
attempt journals are retained under
`demos/PlugCharger-orientation-tilt-diagnostic_20261119_r78/` and
`demos/PlugCharger-orientation-matched-r78_20261119/` (matched summary:
`matched_results.json`). The passing 15° source HDF5 SHA-256 values are
`26e1b3ddddd3ceaa129c235eff10c23a51b17a30851ec6bb2c464cf09229f0e6` (seed
2026113003) and
`692c678f310f8563a18831152a0ff4142496c87be5de3281e0827d9f3eafabc8` (seed
2026113001). The two passing converted files retain separate provenance beside
their source episodes; their initial-row defect prevents training admission.
Ticket 07 remains blocked, and no training contract is frozen.

## Round 79 — deterministic replay/collection development

Collection seeds Python, NumPy, Torch and installed MPLib's global planner RNG
per planned case using `scene_seed + 1`. RNGs are set before solver reset and
reapplied after its environment reset but before planner creation; fixed/random
pairs therefore share the same planner seed. Attempt journals and source JSON
record that seed. This is repeatable planner RNG initialization, not a claim of
bitwise-identical physics. The official 15-degree solver remains the pipeline
default; an explicitly selected copy tilt is forwarded to both collection workers.

Every non-empty expert attempt is saved regardless of native success, with its
actual saved episode ID and expert status. Replay selects only expert-success
source IDs while reporting the saved-source denominator and retained failures.
Public replay captures destination state and observation immediately before its
first action. It aligns the converted initial row only after the capture matches
canonical source state 0; the raw conversion is retained, actions are unchanged,
and malformed captures fail before alignment writes. Each corrected trajectory has
an `.alignment.json` sidecar with raw/corrected file hashes, action hash and the
captured-initial-state fingerprint.

The canonical 16-file Peg suite plus both Plug test modules (18 files total)
passed **178 tests** in 17.41 s (18.61 s outer), under a 180 s TERM / 10 s KILL
limit. Expected-red and green logs,
file list and timing are in
`outputs/exp/20261119_235000_plug_round79_development/`. This is development
evidence only: no native planner, simulator control, collection, replay, training,
policy/checkpoint query, dependency change, or ManiSkill source edit occurred.
Ticket 07 remains **not passed**; the same-20-case native attempt is pending
independent review.
