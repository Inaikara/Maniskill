# Final retained artifacts — three obstacle tasks

## Orientation best results and videos (2026-11-23)

Following the obstacle-task pattern, one best result per task and method is
retained by hardlink. Historical checkpoints and evaluations were not deleted.
Peg uses the random-500 comparison that met both numeric gates; it is nonblind.
Plug uses the validation-selected random-500 30k candidate. Plug videos replay
the recorded actions of the lowest successful scene and match native success.
Peg videos are the original scene-0 recordings, not new policy samples.

| Method | Result | Weight | Video directory |
|---|---|---|---|
| Peg Base | random 3/100, fixed 52/100 | `outputs/ckpt/retained_orientation/PegInsertionSide/BaseDP/update_100000.pt` | `outputs/video/PegInsertionSide_BaseDP/` |
| Peg ConstrainNet | random 23/100, fixed 54/100 | `outputs/ckpt/retained_orientation/PegInsertionSide/ConstrainNet/update_25000.pt` | `outputs/video/PegInsertionSide_ConstrainNet/` |
| Plug Base | random 22/100, fixed 53/100 | `outputs/ckpt/retained_orientation/PlugCharger/BaseDP/checkpoint_080000.pt` | `outputs/video/PlugCharger_BaseDP/` |
| Plug ConstrainNet | random correct 41/100, fixed 54/100 | `outputs/ckpt/retained_orientation/PlugCharger/ConstrainNet/checkpoint_030000.pt` | `outputs/video/PlugCharger_ConstrainNet/` |

Selection, hashes, and replay identity: `outputs/exp/20261123_103016_retained_orientation_results/selection.json`.

Additional videos use the lowest recorded indices, not visual selection. BaseDP
failures: Peg random scenes 0–2 and fixed scenes 2/6/9; Plug random scenes
1/2/4/5 and fixed scenes 1/8/9. ConstrainNet successes: Peg random scenes
0/2/9 and fixed scenes 0/1/3; Plug random scenes 0/1/3 and fixed scenes 0/2/3.
Plug clips replay the saved actions and matched native success. Peg clips are
same-seed CUDA rerenders because per-control actions were not retained; each
rerender kept the original success or failure. One Plug fixed Base replay
disagreed with its recorded failure and was discarded.

## PlugCharger Goal-loop Round 128 — independent heldout complete

The authorized 600-episode CPU roster completed once in the frozen order
random-final Base/correct/fixed-standard/shuffled, then fixed-base-test
Base/correct. Counts were **22/100, 41/100, 28/100, 20/100, 53/100, and
54/100** respectively; native controls were **291,334**. Random correct vs Base
is **+19 pp** (paired 26 wins / 7 losses); fixed-standard vs Base +6 pp,
shuffled vs Base −2 pp, correct vs fixed-standard +13 pp, correct vs shuffled
+21 pp, and fixed retention correct vs Base +1 pp (13 wins / 12 losses).
The historical validation remains **+10 pp**, below the unchanged **+20 pp**
gate. Heldout +19 pp is below +20 pp as well; no gate pass, checkpoint
reselection, retraining, significance, routine-finetune comparison, or Peg/Plug
pooling is claimed.

All 600 journals reconcile to summaries, scene IDs, seeds, ≤600 controls,
finite 46-D observations/7-D actions and native `info.success`. Terminal
simulator states are absent, so independent full-pose recomputation is
unavailable. Runtime was **10,709.807 s stage / 10,713.55349533 s outer**; the exact
canonical 25-file suite passed **285 before and after**. A pre-test import-path
setup error and a post-completion wrapper exit-quoting error are fully recorded;
the evaluator itself wrote the complete result and was not rerun. All 15 direct
input hashes, 261 source-archive hashes, and 4,010 imported runtime-source hashes
match pre/post. The shuffled assignment was sealed before first policy action.

- Six cohorts and episode journals: `outputs/exp/20261123_070724_plug_random500_heldout/`.
- Preregistration, complete episode CSV/JSONL, paired audit, raw command logs,
  wrapper incident, source/runtime archives, tests, timings and pre/post hashes:
  `outputs/exp/20261123_070724_plug_random500_heldout_envelope/`.
- Frozen selected checkpoint SHA-256:
  `911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1`;
  frozen Base SHA-256:
  `ceeb68b4838ced6e6ffa43646bcc3cfc47538469e02264c4bc80a2542a33d130` (EMA
  tensor SHA-256 `ab97b69e3bf62d77de030dd745ed2f416691589a22b67c20efdefac791b2c083`).

## PlugCharger Round 126 — heldout runner implementation only

Implemented `evaluate-heldout` with production identity checks against the
registered random-500 validation selection, selected 30,000-update checkpoint,
frozen Base v1, and the two heldout catalog manifests. Contract:
`plug_orientation_heldout_contract.json`. The explicit forward authorization
does not rewrite the historical result: selected random validation is 23/50
versus Base 18/50 (+10 pp), and the unchanged +20 pp method gate is unmet.
This run is not a passing-gate confirmatory test.

The implementation includes a fixed 600-arm roster, one Base and one Constrain
adapter, seed-1703 shuffled targets, per-control/per-episode journals, alignment
diagnostics and native official position/rotation errors when supplied by info,
and exception-safe partial progress. Terminal simulator states are
not journaled, so independent full-pose recomputation is unavailable. Videos
remain an explicit gap because no compatible existing per-arm recording path
was established. The runner was tested with fake env/policy boundaries and real
selection/checkpoint/catalog identity preflight. RED/GREEN raw outputs and UTC/
monotonic markers are retained in `outputs/exp/20261123_064041_plug_heldout_runner_development/`;
the genuine missing-command RED is in `outputs/exp/20261123_063021_plug_heldout_runner_development/`.
Focused tests passed **44** and the exact canonical 25-file roster passed **284**.
No heldout scene was run: **0 heldout episodes / 0 native controls / 0 policy
actions**. Ticket 09 remains incomplete; actual paired heldout results and the
cross-task/final conclusion are future work.

Runnable command (execution requires the separately planned stage operation;
this implementation round did not execute it):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=..:.
timeout --signal=TERM --kill-after=20s 25200s \
  python -m examples.baselines.constrain.plug_orientation_adaptation evaluate-heldout \
  --output-root outputs/exp/YYYYMMDD_HHMMSS_plug_heldout_600
```

## PlugCharger Round 125 — random-500 validation complete; method gate failed

The complete validation roster ran once: fresh Base on both profiles plus six
500-random checkpoints on both frozen 50-scene validation catalogs (**700/700**).
CPU PhysX, seed `1 + scene index`, stride 8/native 7-D actions, 600-control cap;
**316,520 controls**, **12,816.473 s**, exit 0 under the 18,000 s TERM / 20 s KILL
cap. The exact canonical 25-file suite passed **276 pre/post**.

Fresh Base scored **30/50 fixed, 18/50 random**. Candidate fixed/random:
5k **21/12**, 10k **25/12**, 15k **28/17**, 20k **33/17**, 25k **31/21**,
30k **37/23**. The selector chooses 30k, SHA-256
`911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1`. Fixed
retention loss is −14 pp and passes; random gain is **+10 pp**, below the
unchanged **+20 pp** method gate. This positive point estimate does not pass the
gate. The one-seed, validation-selected result is descriptive only.

All 700 event journals reconcile with the 14 summaries for scene/seed, controls,
finite observations/actions and native `info.success`; simulator terminal states
are not stored, so independent full-pose predicate recomputation is not claimed.
All **338 directly registered input/source hashes** match pre/post. The nine
catalog-bundle files are identity-only; heldout/final scene contents remain
closed. Supplemental post-run identity checks found all 18 random-100/200/300
checkpoints and all ten Base v1/v2 checkpoints match their prior seals (not
contemporaneous pre-run captures). No new training/data, source/dependency edit,
commit, or push.

- Episode data and selection: `outputs/exp/20261122_045531_plug_constrain_random500_validation/`.
- Registration, raw logs, pre/post hashes, source archive, audit, tests and
  artifact manifest: `outputs/exp/20261122_045531_plug_constrain_random500_validation_envelope/`.
- Historical selected fixed/random results for 100/200/300/500 pools are
  30/20, 38/21, 33/23, 37/23. Fresh Base fixed results vary (29/31/30/30),
  so these are not causal evidence of a data-size effect. Ticket 09 remains
  incomplete; both heldouts remain closed.

## PlugCharger Round 124 — random-500 fresh Constrain training complete (validation pending)

Following admission and review/authorization, one fresh seed-1 CUDA run completed
**30,000/30,000 updates** on the exact 500-random / 1,000-fixed pools. It did not
resume any 100/200/300 checkpoint. Runtime was **1,187.152 s trainer** and
**1,207.809 s outer monotonic**, exit 0 on an NVIDIA GeForce RTX 5070. The
canonical 25-file suite passed **276 before and 276 after**. All 300 logged rows
are finite at the registered 100-update cadence; final LR is zero.

Six 5k–30k validation-candidate checkpoints were strict CPU-loaded and audited.
Each preserves all **148 frozen Base tensors bit-for-bit**; each has 100 finite
control EMA tensors matching EMA shadow ordering and matching optimizer,
scheduler and EMA update counts. Their hashes are:

| Update | SHA-256 |
|---:|---|
| 5k | `c22d79d60d91637dcf4a6f98fd3f3df64f95141df82357135a7548ef67ebe155` |
| 10k | `6415dc5cd4bae718beadcd10b2f978572e4700597c81a3cf1c6d6b4edbb34fa9` |
| 15k | `0a550af1798b8dfe59a0653ec75af356d699ee9874b73b04dfe9b8950880913f` |
| 20k | `49c9447beba008db2cd043429081803308368a8d4997f4c5c09e8f55ba1aee8e` |
| 25k | `ddf2b7765c15eaae9901566c57c60300949569436426cdb3ca13d09b211b2aac` |
| 30k | `911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1` |

Direct pre/post identities for **49 protected files** match, including all
current/ancestor data artifacts, the frozen Base checkpoint, relevant sources,
and all nine catalog bundle files (catalogs were identity-only; no scenes or
outcomes were accessed). No policy queries, validation episodes, or heldout
queries occurred. This is **training completion only**, not an efficacy claim.
Stop for independent parent review; the only prospective next action is the
already registered 700-episode validation roster if separately approved. No
additional data or training is authorized.

- Experiment/logs: `outputs/exp/20261122_042359_plug_orientation_constrain_random500/`
- Weights: `outputs/ckpt/20261122_042359_plug_orientation_constrain_random500/`
- Full raw command, stdout/stderr, UTC/monotonic times, preregistration,
  pre/post hashes, tests, audits and summaries:
  `outputs/exp/20261122_042359_plug_orientation_constrain_random500_envelope/`.
- No source-code, dependency, data, or prior-checkpoint edits; no commit/push.

## PlugCharger Round 123 — random-500 data admitted (data only)

User-approved extension of the same experiment completed once: the immutable
random-300 parent plus the first 200 fresh eligible NativeEE v4 expert/raw-replay
pairs now form the 500-example pool. The 300-candidate plan (seeds
`2026123450..2026123749`, `default_rng(1704)` draws 450–749) was sealed before
native callbacks. **200 were admitted after 222 attempts**: 21 ordinary native
eligibility rejections and one recorded MPLib screw-planning failure (zero
controls), then continuation with the next sealed seed. Attempt-stage controls
were **61,340 source + 61,340 replay**; admitted rows account for **56,914 +
56,914**. New yaw bins `[42,35,46,40,37]`; combined 500 bins
`[97,100,96,104,103]`.

The real `PlugAdaptationDataset(expected_episodes=500)` reader validated all 500
episodes, source/replay pairs and saved full-pose terminal predicates: **141,605
windows**; sample shapes `(2,46)/(16,7)/(2,)`, finite, max action magnitude 1.0.
Canonical 25-file tests passed **276 before and after**. Thirty-five protected
input identities and all 35 archived/current source files match; nine catalog
files were hash-only. Collection elapsed **559.530 s**, exit 0 within the
12,000 s outer deadline. No training, policy query, validation or heldout access.
At collection close this admitted data only and made no efficacy claim. The one
subsequently authorized 30k run is recorded in Round 124; do not exceed 500.
The 18 prior random-100/200/300 checkpoints match retained Round-122 identity
seals post-run; this is not claimed as direct contemporaneous pre/post hashing.

- Data: `demos/PlugCharger-orientation-random-500_20261122_035654/` — HDF5
  `121ff0905b737e999a0299e81a62e4bf142f4306d8c1f6b242e62f3d36052197`, JSON
  `06e568da0debbb2bb4465e8738990b4eec82cf02f613a17fc3a2fc8d3018bf05`, manifest
  `8f88ce6ced2f0b907f2b07c63cda37e12771e6b6a15ea5af334f43e5cf3bce68`.
- Raw logs, registrations, audits and summaries:
  `outputs/exp/20261122_035654_plug_random500_collection/` and sibling
  `outputs/exp/20261122_035654_plug_random500_collection_envelope/`.
- The historic random-300 parent and the Round-121 +10 pp negative gate remain
  unchanged. No training update or performance improvement is claimed.

## PlugCharger Round 122 — one random-500 forward experiment path implemented (offline)

User amendment (2026-11-22): **“增加到500条 继续相同实验”**. It prospectively supersedes Round 121's 300-example stop rule for this single 500-pool experiment only. The actual 300-row pool, prior contracts/checkpoints/results, and Round-121 +10 pp result below are historical and unchanged. Scope is immutable 300 + first 200 fresh eligible native expert/raw-replay pairs, then one fresh seed-1 30k run over 500 random / 1,000 fixed, then the same 700 validation episodes. No 400-pool training, Base variant, heldout access, final paired runner or automatic expansion beyond 500. Fixed retention and random gain gates are unchanged; old Base absolute qualification gates remain waived only as prerequisites.

Offline work adds the additive `plug_orientation_adaptation_expansion500_contract.json`, 500-pool actual lineage validator, and runnable `collect-random500` / `train-expanded500` commands; existing `evaluate-validation` retains the frozen two-catalog, six-checkpoint 700-episode roster. Actual random-300 data and ancestry were read-only audited before a callback deliberately stopped pre-native. A synthetic NativeEE source/replay pool passes the real 500 lineage reader, followed by a separate two-update CPU toy training/checkpoint test. No new native controls, production updates, trained policy queries, or holdout queries occurred. Exact historical 25-file suite plus six additions: **276 passed**. Expected-red assertion was observed before implementation, but its raw initial console output was not saved; no output has been reconstructed. Review is required before collection/training/validation.

- Development evidence: `outputs/exp/20261122_031224_plug_random500_development/`.
- Runnable bounded commands and contract details are recorded in `PLUG_ORIENTATION.md`.
- Data cap: 300 new candidates / first 200 admitted, 600 controls per teacher/replay, 180,000 controls per side, 10,800 s internal / 12,000 s outer. Training: 30,000 updates, 7,200 s internal / 9,000 s outer. Validation: 700 episodes, 21,600 s outer.

## PlugCharger Round 121 — random-300 validation complete (method gate failed)

The one registered 700-episode roster completed exit 0: fresh Base once per
profile and all six Round-119 candidates on both 50-scene validation catalogs.
Fresh Base scored **30/50 fixed, 18/50 random**. Total was **322,364 native
controls / 13,079 s outer wall** under the 18,000 s TERM / 20 s KILL cap. The
canonical 25-file suite passed **270 before and after** (52.92 s each).

| Update | Fixed | Random | Fixed retention loss | Random gain | Eligible |
|---:|---:|---:|---:|---:|:---:|
| 5k | 14/50 | 7/50 | +32 pp | −22 pp | no |
| 10k | 26/50 | 12/50 | +8 pp | −12 pp | no |
| 15k | 33/50 | 15/50 | −6 pp | −6 pp | yes |
| 20k | 33/50 | 19/50 | −6 pp | +2 pp | yes |
| 25k | 33/50 | 23/50 | −6 pp | **+10 pp** | yes |
| 30k | 34/50 | 17/50 | −8 pp | −2 pp | yes |

The selector picks 25k among candidates meeting the fixed-retention boundary;
checkpoint SHA-256 is `75226bbea156ce833da0658d54f9d565990dc7ea602e05287d76ec15778e8cca`.
Its fixed retention passes (−6 pp loss), but randomized gain is only **+10 pp**,
below the **+20 pp** requirement. **The bounded method study closes with no
demonstrated prespecified gain.** No more data, training/refit, architecture
changes, or variants; no heldout/final paired runner. Both heldout catalogs remain
closed. Old Base absolute gates remain waived only as forward prerequisites and
are not qualified by this run. One-seed, selection-set results are descriptive,
not statistical significance or independent generalization.

The independent audit reconciles all **700 episode journals** to scene IDs,
seeds, controls, finite 46-D observations/7-D actions, and native `info.success`;
the journal omits simulator terminal physical states, so independent full-pose
terminal recomputation is not claimed. **32 registered inputs** (including all
nine catalog files), ten experiment-side source files and 18 explicitly captured
installed runtime source files match pre/post hashes. The 12 random-100/200
checkpoints and ten v1/v2 Base checkpoints also match retained prior seals (not a
new contemporaneous pre/post capture). Final/test payloads were hash-only and
never opened as scenes.

Lineage coverage caveat: Round-121's direct preregistration did not separately
list the random-300 attempt ledger/parent-lineage JSON, fixed dataset manifest, or
random-100 ancestor files. The first three random-300 data artifacts match
Round-119 pre/post seals; the fixed manifest matches its original admission seal;
random-100 HDF5/JSON/manifest match the hashes embedded in the random-200
parent-lineage record. These prior/transitive checks are not direct Round-121
pre-run entries; see `lineage_provenance_reconciliation.json`.

- Preregistration SHA-256: `ff966a69f7ff8d508b551735a136074a4201b88723763af1434eb6e723b4800f`.
- Full protocol, raw logs, input/source audits, decision and artifact manifest:
  `outputs/exp/20261120_183731_plug_constrain_random300_validation_envelope/`.
- Episode journals and evaluator selection: `outputs/exp/20261120_183731_plug_constrain_random300_validation/`.
- No ManiSkill/dependency/source/data/checkpoint edit, checkout, commit, or push.

### Random-pool validation summary (historical Base result varies by run)

| Training pool | Fresh Base fixed/random | Selected checkpoint | Selected fixed/random | Fixed retention loss | Random gain | +20 pp method gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 100 | 29/50, 18/50 | 25k | 30/50, 20/50 | −2 pp | +4 pp | fail |
| 200 | 31/50, 18/50 | 30k | 38/50, 21/50 | −14 pp | +6 pp | fail |
| 300 | 30/50, 18/50 | 25k | 33/50, 23/50 | −6 pp | +10 pp | fail |

These are validation-set checkpoint-selection results from distinct one-seed
training runs, not paired across datasets or evidence of statistical significance.
The 300-pool bounded study therefore ends negative; no heldout evaluation follows.

## PlugCharger Round 119 — random-300 Constrain training complete (validation pending at the time)

Following the Round-118 data admission and a separate explicit parent authorization,
one fresh seed-1 CUDA run completed **30,000/30,000 updates** on the registered
300-random / 1,000-fixed window pools. The random pool has **84,891 windows** and
the unchanged fixed pool **279,625**; each batch is balanced 512/512. Trainer time
was **1,187.25 s**, outer monotonic elapsed **1,207.03 s**, exit 0 within the
7,200 s internal / 7,230 s external TERM/KILL bounds. All 300 logged rows are
finite and final LR is zero.

This is a fresh zero-control branch over exact frozen Base v1 80k EMA
(`ceeb68b4…a2a33d130`, EMA `ab97b69e…b2c083`), not a resume of prior Constrain
weights. All six 5k–30k `validation_candidate` checkpoints have distinct hashes
and pass strict CPU adapter/provenance audits without `get_action` or policy
forward. Every checkpoint preserves all **148** frozen Base tensors bit-for-bit;
optimizer/scheduler/EMA steps agree and 100 trainable control EMA tensors are
finite/nonzero. The exact 25-file suite passed **270 before and after**. Forty-eight
captured protected identities and the registered current-source closure match
pre/post; all nine catalog bundle files are identity-only seals, with no catalog
outcomes queried. Historical parent/source/checkpoint artifacts retain their
earlier seals; these are not represented as new contemporaneous pre/post hashes.

One setup incident is directly documented: the evidence helper's path-normalization
error, recorded in `pretraining_setup_incident.txt` as occurring before any
production loader/training call. A separate narrative says a first CLI attempt
encountered a pre-created experiment directory, but its referenced `first_launch.*`
command/stderr/exit/time files are absent; that failure's cause and zero-update
status are therefore not independently verified. Do not count it as a verified
zero-update incident. The successful run's own retained command, logs, empty
experiment directory before launch, 30,000-update report, and six audited
checkpoints establish the successful training run. No retrospective timestamp or
missing launch log is reconstructed.

At Round 119 training completed without policy actions or validation. Round 121
has since run the registered roster and failed the method gate; the bounded study
stops at the 300-data ceiling and final/heldout remain closed.

- Logs/config/source archive: `outputs/exp/20261120_175628_plug_random300_train_r1/`
- Weights: `outputs/ckpt/20261120_175628_plug_random300_train_r1/`
- Commands, actual timing, pre/post tests and identities, loader/checkpoint audits,
  results and artifact manifest:
  `outputs/exp/20261120_175628_plug_random300_train_envelope/`; additive
  reconciliation receipt: `outputs/exp/20261120_175628_plug_random300_train_reconciliation/receipt.json`.

## PlugCharger Round 118 — random-300 collection admitted (data only)

Completed one bounded continuation from the immutable random-200 pool. The
150 sealed candidates use seeds `2026123300..2026123449` and the 150 uniform
`default_rng(1704)` yaw draws 300–449. The first 100 source-plus-raw-replay
eligible episodes were admitted after 107 attempts, seeds
`2026123300..2026123406`; no retry or old candidate reuse. New-stage totals are
29,882 source controls and 29,882 replay controls, with a 90,000 cap per side.
The unchanged parent 200 plus new 100 yield exactly 300 episodes; new yaw bins
`[19,25,18,14,24]`, combined bins `[55,65,50,64,66]`.

The actual production loader and 300-lineage/200-parent ancestry validator pass:
84,891 windows, finite native values, first-item shapes `(2,46)/(16,7)/(2,)`,
action absolute maximum 1.0. The exact canonical 25-file suite passed 270 before
and after. Collection elapsed 266.439 s, exit 0 under the 5,400 s TERM / 20 s
KILL whole-stage timeout. Registered current source and complete parent-tree
hashes match pre/post. Other protected historical datasets, checkpoints, nine
catalog files, and producer archives were post-hashed against retained earlier
identity records/seals; this is not claimed as a new contemporaneous pre/post
snapshot for those files. HDF5 SHA-256
`95ede016098066b535208ec2569d4c092186a242e4627206306d44137dc2fa0f`, JSON
`129dcf74369eb95dad2e2960877024afe9f831a47bd36576dea4ce1411f1a42f`, manifest
`e32b6914a08d41adfb7cb2e87f0af458cd4d035880373a4b51cf3fdfc175b6dc`.

Data: `demos/PlugCharger-orientation-random-300_20261120_173932/`; collection
and sealed sources: `outputs/exp/20261120_173932_plug_random300_collection/`;
independent audit, pre/post source/tree identities, tests, timings and artifact
manifest: `outputs/exp/20261120_173932_plug_random300_collection_envelope/`.
This is data admission only: no training, policy query, validation or heldout
evaluation. The random-200 method gate remains failed (+6 pp vs +20 pp); no
method improvement is established. Ticket 09 remains incomplete and the 300
episode ceiling is unchanged. No further expansion, training or validation is
authorized by this collection result; final/heldout catalogs remain closed.

## PlugCharger Round 116 — bounded random-300 path implemented (offline; review required)

Added one registered expansion only: the immutable random-200 pool (manifest
`3f96a3ff…ac304b`, HDF5 `eeed1e01…c510c210`, JSON
`a5245c8d…f1ff269`) remains first; 100 new successes may be admitted from the
first eligible rows of 150 candidates, seeds `2026123300..2026123449`. Yaws use
one `default_rng(1704)` stream after skipping draws 0–299 and consume draws
300–449. The collector validates the real random-200 plus random-100 ancestry
before its first callback, archives the current source/contract, separates new
attempt/control totals from parent ledgers, and seals current preregistration,
journal, source/replay/case hashes and consolidated native HDF5 values. The
reader rechecks all 200 parent rows and all 100 new source/replay rows against
that lineage. The old random-100/200 entry points and expansion contracts remain
registered unchanged; only 100/200/300 pools are accepted.

The additive contract pins the same frozen Base v1 EMA, fresh zero-control
initialization, 30k updates, balanced 512/512 batches, six existing checkpoints,
700 validation episodes and unchanged method gates. It explicitly stops this
study if either method gate fails; it authorizes no further data/code variants
or final/heldout access. Runnable CLI stages are `collect-random300` and
`train-expanded300`; after independent review and stage authorization, use
timestamped paths and these bounded forms (activate the `maniskill` Conda
environment first):

```bash
timeout --signal=TERM --kill-after=10s 6000s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m examples.baselines.constrain.plug_orientation_adaptation collect-random300 \
  --parent-dataset demos/PlugCharger-orientation-random-200_20261120_121208 \
  --dataset-dir demos/PlugCharger-orientation-random-300_<UTC> \
  --exp-dir outputs/exp/<YYYYMMDD_HHMMSS>_plug_random300_collection

timeout --signal=TERM --kill-after=20s 9000s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m examples.baselines.constrain.plug_orientation_adaptation train-expanded300 \
  --random-h5 demos/PlugCharger-orientation-random-300_<UTC>/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --random-manifest demos/PlugCharger-orientation-random-300_<UTC>/dataset_manifest.json \
  --fixed-h5 demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --fixed-manifest demos/PlugCharger-orientation-fixed-1000_20261119_184307/trajectory.state.pd_ee_delta_pose.physx_cpu.json \
  --exp-dir outputs/exp/<YYYYMMDD_HHMMSS>_plug_random300_train \
  --ckpt-dir outputs/ckpt/<YYYYMMDD_HHMMSS>_plug_random300_train --device cuda

timeout --signal=TERM --kill-after=20s 21600s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python -m examples.baselines.constrain.plug_orientation_adaptation evaluate-validation \
  --ckpt-dir outputs/ckpt/<YYYYMMDD_HHMMSS>_plug_random300_train \
  --random-catalog outputs/exp/20261119_184307_plug_orientation_catalogs/random_validation \
  --fixed-catalog outputs/exp/20261119_184307_plug_orientation_catalogs/fixed_validation \
  --output-root outputs/exp/<YYYYMMDD_HHMMSS>_plug_random300_validation
```

This round ran no teacher/replay/native control, production optimizer update,
production policy action query, validation episode, or holdout query. A synthetic
HDF5 end-to-end collector test admitted 100 ordered fresh rows through the native
file/pair predicate without invoking a teacher or simulator. The exact historical
25-file suite plus round-116 tests passed **269 tests** in **53.1 s**; the new
300-row loader fixture and two-update CPU toy training use owned test bindings,
not a production provenance bypass. The initial expected-red assertion showed
the missing `collect_random300` API; raw output is retained, but its monotonic
start/end timestamps were not captured and are not reconstructed. Frozen parent,
Base/checkpoint, and random-200 contract hashes match before/after. Evidence,
source archive/closure, exact roster/command, logs and hashes are in
`outputs/exp/20261120_165132_plug_random300_development/`. The prior random-200
validation remains **+6 pp**, below +20 pp; no outcome or gate is rewritten.
Independent review remains the prerequisite to the bounded collect/admit/train/
validation chain. No ManiSkill source/dependency change, commit, or push.

## PlugCharger Round 115 — random-200 Constrain validation complete (method gate failed)

The full frozen validation-only roster completed **700/700 episodes**: fresh
Base once on each 50-scene profile and all six 5k–30k random-200 Constrain
checkpoints on both profiles. CPU `physx_cpu`, seed `1 + scene index`, native
stride-8 7-D actions, and the unchanged 600-control cap were used. Total work
was **311,659 native controls / 12,517 s**, exit 0 under the 18,000 s TERM/KILL
limit. All 700 per-control journals reconcile to the native `info.success`
summaries and exact control counts. The exact 261-test roster passed before and
after (**28.64 s / 35.46 s**). All 54 registered source/input identities,
including all nine catalog bundle files, match pre-run hashes; 15 archived
installed source-module files also match pre/post. No final/test catalog or
heldout outcome was opened.

Fresh frozen v1 Base: **31/50 fixed, 18/50 random**. Candidate results:

| Update | Fixed | Random | Fixed retention loss | Random gain |
|---:|---:|---:|---:|---:|
| 5k | 26/50 | 9/50 | +10 pp | −18 pp |
| 10k | 22/50 | 14/50 | +18 pp | −8 pp |
| 15k | 36/50 | 14/50 | −10 pp | −8 pp |
| 20k | 39/50 | 19/50 | −16 pp | +2 pp |
| 25k | 41/50 | 17/50 | −20 pp | −2 pp |
| 30k | 38/50 | 21/50 | −14 pp | **+6 pp** |

The validation selector chooses **30k**, SHA-256
`b0764215fb65d6c3b1bd7852788dadc39a0883d9a06ec391f3c5a73868e67167`.
Retention passes (−14 pp loss), but randomized gain **+6 pp** is below the
unchanged +20 pp gate. **Method gate failed; no checkpoint is frozen for final
evaluation and no heldout evaluation is authorized by this result.** The waived
Base absolute gate was not applied. Round 107's +4 pp result and this run are
separate validation samples/runs; historical outcomes are not rewritten.

Audit boundary: native `info.success` is retained and reconciled, but the
evaluator journal does not save simulator terminal states, so independent
recomputation of the full-quaternion success predicate is unavailable. This is
one training seed and validation used for selection, not a blind significance
claim. Full report, gate decision, source/input pre/post hashes, installed-source
archive, exact commands/tests, runtime, timings, and artifact manifest:
`outputs/exp/20261120_131136_plug_constrain_random200_validation_envelope/`;
the 700 episode journals and summaries are in
`outputs/exp/20261120_131136_plug_constrain_random200_validation/`.
No source/dependency/data/checkpoint edits, commit, or push.

## PlugCharger Round 114 — random-200 Constrain training complete (validation pending)

The fresh seed-1 Constrain run completed **30,000/30,000 CUDA updates** on the
RTX 5070 in **1,188.47 s trainer time / 1,206 s outer wall time**, exit 0. It
used the independently admitted 200-random-episode pool (**56,481 windows**) and
the unchanged fixed-1,000 pool (**279,625 windows**), with balanced 512/512
windows per 1,024-item batch. This is a fresh zero-control branch over the exact
frozen original v1 80k EMA—not a resume of random-100 weights. All 300 100-update
log rows are finite and terminal LR is zero.

All six 5k–30k checkpoints have distinct SHA-256 identities and pass CPU-only
strict adapter/provenance checks. Every checkpoint preserves all **148** frozen
Base tensors bit-for-bit; optimizer, scheduler, and EMA steps agree with the
checkpoint update; the 100 trainable EMA tensors and raw zero-convolution branch
are finite/nonzero. The run-owned contract copy binds the parent random-100
forward-contract SHA and actual random-200 manifest SHA. The exact 261-test suite
passed before (**28.13 s**) and after (**28.85 s**) the run; registered source and
input hashes match pre/post. Historical producer archives remain separately
verified against their own bytes.

This is **training completion only**. No trained action was sampled and no
validation, final/test, or heldout outcome was accessed. The prior random-100
candidate remains at **+4 pp**, below the frozen **+20 pp** randomized-gain gate;
that failure is unchanged. The new random-200 model has no performance result or
method-improvement claim. Validation-only evaluation is the next gate after
parent review; keep heldout closed.

- Logs/config/source archive: `outputs/exp/20261120_123140_plug_orientation_constrain_random200/`
- Weights: `outputs/ckpt/20261120_123140_plug_orientation_constrain_random200/`
- Registration, exact commands, versions, data/catalog hashes, loader audit,
  timings, test logs, input pre/post identities, six-checkpoint audit, and artifact
  manifest: `outputs/exp/20261120_123140_plug_orientation_constrain_random200_envelope/`
- Random data: `demos/PlugCharger-orientation-random-200_20261120_121208/`

No ManiSkill source/dependency, training data, old Base/random-100 checkpoint,
catalog, or historical result was changed. No commit or push.

## PlugCharger Round 113 — float32 yaw representation repair and offline admission

The expansion reader previously compared consolidated metadata yaw against the
sealed float64 plan at `1e-12`, although that row is sourced from the native
float32 target state. It now compares only that metadata row after float32
round-trip; candidate-plan and attempt-ledger comparisons remain at `1e-12`, and
sealed plan-stream verification, exact native source/replay checks, and official
full-pose admission remain unchanged. Historical expansion source closure is
verified against its archived bytes, not today's offline reader implementation.
The old Round 112 rejection remains an immutable record; the fix changes no
collection source, manifest, journal, HDF5, or replay.

The production `PlugAdaptationDataset` reader now independently accepts the
actual 200-row dataset `demos/PlugCharger-orientation-random-200_20261120_121208/`:
200 episodes, **56,481** production windows, first-item shapes observations
`(2,46)`, actions `(16,7)`, hint `(2,)`. Its existing validator traverses all
200 source/replay/case references and checks expansion lineage, exact parent
trajectory values, ordered first-eligible candidate ledger, and archived source
closure. Collection remains 107 attempts / 100 new accepted, seeds
2026123150–2026123256, with 30,027 source and replay controls; this is readmission
of completed data, not new physics. The HDF5 SHA-256 remains
`eeed1e0170702b89de1a5505f2ae863fccaf9c6c467eeec9a549ed14c510c210`; consolidated
JSON `a5245c8de32d103cffe7c60a1f9a409badfa42c87c0d6488931c6c951f1ff269`; manifest
`3f96a3ff54072d64023dc057271614cd09225deb5356dafb8d51785562ac304b`.

Regression tests cover genuine float64-plan → float32-row acceptance and reject
a `1e-4` yaw tamper. The exact unchanged Round 110 25-file suite plus new cases
passes **261 tests** (27.92 s); focused lineage cases pass 4/4. No native control,
policy/model query, optimizer update, production training, or evaluation ran.
No commit. Data is reader-admissible for independent review; fresh control or
training remains pending that review/authorization.

## PlugCharger Round 110 — random-200 provenance/deadline blocker fixes (offline)

The collector now checks the actual frozen random-100 manifest hash before any
callback; the loader binds all 100 parent HDF5 trajectories and source/replay/case
references to the actual frozen files, then checks the ordered 150-candidate plan,
first 100 eligible ledger rows, candidate source hashes, and native source/replay
pair audit before training. Toy training tests replace only this validator boundary;
they cannot present synthetic rows as production provenance. Unknown controls on
the recognized MPLib failure are retained as null with an auditable partial summary
before raising. Candidate completion and final publication both recheck the monotonic
deadline and sealed/current/archive parent inputs; late candidates are never admitted.

The exact historical 25-file regression roster passes **257 tests** (27.63 s); the
focused adaptation module passes **18 tests**. Raw suite stdout/stderr, exact command,
UTC and monotonic timing are retained at
`outputs/exp/20261120_110001_plug_random200_blockerfix/`. Evidence is offline only: no callback,
native controls, training update, model query, or evaluation was run. This is code
ready for independent review only, not a completed or authorized production dataset.
Parent data and frozen checkpoints are unchanged. No commit or push.

## PlugCharger Round 109 — bounded random-200 support (offline implementation)

Added the separate `collect-random200` and `train-expanded` entry points while
leaving the historical random-100 contract/checkpoints untouched. The 200-pool
registration is an additive amendment pinned to the original random-100 dataset,
its full source/replay tree, manifest, collection preregistration and archived
source closure. It retains the original 100 episodes first and admits only the
first 100 eligible new episodes from seeds `2026123150..2026123299`; one
`default_rng(1704)` stream records all 150 yaw draws after skipping original draws
0–149. No old unused candidate seeds are reused. Current-attempt counters exclude
all parent attempts/controls; the separate parent lineage ledger preserves source
metadata and attempts without merging those counters into this collection.

The new prospective contract fixes the same exact frozen Base v1 EMA, fresh
zero-control initialization, 30k updates, balanced 512/512 windows, six validation
checkpoints, frozen validation catalogs, 600-control episodes and unchanged method
gates. Validation accepts only legacy 100 or this exact registered 200 lineage;
it rejects unregistered sizes and inconsistent checkpoint rosters. This does not
implement/open a final heldout runner. The prior random-100 validation stays at
Base 18/50 vs selected Constrain 20/50 (+4 pp); the +20 pp method gate remains
failed. No collection, production optimizer update, production evaluation, test
catalog access or ManiSkill-source/dependency change occurred in Round 109.

Focused new tests first failed on the absent 200 entry points/count registration;
the exact unchanged 25-file historical command roster plus new cases then passed **253 tests**
in **25.58 s**. An offline audit traversed the real immutable 100-demo parent and
all archived source identities, then stopped at a fake callback before any teacher
action. Synthetic HDF5 tests verify parent-first exact payload values, immutable
parent hashes, and rejection of a bad parent hash before callback. A two-update CPU
toy train consumes a lineage-checked 200-row fixture and writes count/parent
provenance. Raw logs, test command and timing, source archive, contract and hash audit are in
`outputs/exp/20261120_100123_plug_random200_development/`. The original parent
dataset hashes remain manifest `5e084199…11198`, HDF5 `2df8ffa9…2fe26c`, and JSON
`77e9fed3…6c0f`; the frozen Base checkpoint hash is unchanged. No commit or push.

After independent review and explicit production authorization, collection should
use the bounded CLI with timestamped output paths and an external whole-stage
deadline (the code also enforces its internal 5,400 s collection deadline):
`timeout --signal=TERM --kill-after=10s 6000s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m examples.baselines.constrain.plug_orientation_adaptation collect-random200 --parent-dataset demos/PlugCharger-orientation-random-100_20261120_041640 --dataset-dir demos/PlugCharger-orientation-random-200_<UTC> --exp-dir outputs/exp/<YYYYMMDD_HHMMSS>_plug_random200_collection`.
No such collection has started. Training/evaluation remain subsequent separately
authorized stages; use `train-expanded` with the completed dataset manifest and
the unchanged fixed-1000 inputs only after review.

## PlugCharger Round 108 — Round 107 validation diagnosis (offline)

Read-only paired-journal analysis at `outputs/exp/20261120_091212_plug_random100_validation_diagnosis/` verified all 760 Round 107 artifact SHA-256s both before and after. Against fresh Base, 25k is 10 wins/9 losses fixed (29→30/50) and 9/7 random (18→20/50); random-bin net changes are `[+5,-3,-1,0,+1]` with occupancy `[13,13,14,7,3]`. Thus the +4 pp is concentrated in the lowest-yaw bin, not broad; the existing 100-demo TRAIN yaw-bin occupancy `[18,18,20,23,21]` is already balanced. The positive-edge validation sample is thin, but that does not prove a training coverage hole. Failure-phase findings are limited to an explicitly nonofficial position-only observation proxy; no terminal states/contact evidence or implementation bug was identified. Method gate still fails; no holdout access, qualification, source/data/checkpoint change, native controls, training, or commit. The diagnosis includes a prospective, not executed/authorized-here, bounded 200-effective-demo plan.

## PlugCharger Round 107 — Constrain random-100 validation (gain gate not met)

The complete validation-only roster finished **700/700 episodes**: fresh frozen
Base once on each 50-scene profile, plus all six Constrain checkpoints on all 50
fixed and 50 random validation scenes. CPU inference used the unchanged official
stride-8 sampler, seed `1 + catalog index`, native 7-D actions, and a 600-control
episode cap. Work was **324,694 controls** in **13,431.92 s** (outer timeout
18,000 s; exit 0). Exact 248-test suite passed before (**30.94 s**) and after
(**30.29 s**). Source, all nine catalog-bundle files, frozen Base, and all six
checkpoint hashes remained unchanged. Neither final/test catalog was opened.

Fresh validation Base: **29/50 fixed, 18/50 random** (the historical Base result
remains 30/50 and unqualified). The frozen retention-constrained selection is
25k, **30/50 fixed, 20/50 random**, checkpoint SHA-256
`c6b933083d19b68bbcb42ee4449f40fe95fcdc4efb7c0b1e89b836a25deb5735`. Its fixed
retention loss is −2 pp and passes the ≥27/50 rule. Random gain is only **+4 pp**
(2/50), below the unchanged **+20 pp** method gate. Thus this is a validation
frontier candidate, **not a method-gate pass or qualification**; no heldout run is
authorized/opened. One training seed does not support a statistical-significance
claim.

The evaluator's per-control journals retain actions, observations, and native
`info.success`; those flags reconcile with every arm summary. They do not retain
simulator terminal states, so an independent terminal-state recomputation of the
full-quaternion 5 mm / 0.2 rad predicate was not possible. The preregistration
sealed 14 experiment-side source files and runtime package versions, but did not
archive installed ManiSkill/Diffusers source bytes; no stronger installed-source
byte-identity claim is made. The full caveats, raw journals, exact CLI/test logs,
selection, audits, timings, and artifact hashes are retained at
`outputs/exp/20261120_050952_plug_constrain_random100_validation/` and its sibling
`outputs/exp/20261120_050952_plug_orientation_constrain_random100_validation_envelope/`.
No source, dependency, data, or checkpoint was changed; no commit or push.

## PlugCharger Constrain random-100 adaptation — 30k training complete

The authorized seed-1 run completed **30,000/30,000 CUDA updates** on the RTX
5070 (trainer 1,181.54 s; outer 1,196.62 s; exit 0). It used the admitted random
100 + frozen fixed 1,000 episodes with balanced 512/512 windows per batch. All
300 logged loss/LR rows are finite and terminal LR is 0. Six distinct
5k–30k `validation_candidate` checkpoints passed CPU strict-load, provenance,
finite-weight, optimizer/scheduler/EMA-step, and frozen Base checks. All 148
frozen v1 Base tensors match the selected 80k EMA bit-for-bit in every
checkpoint; EMA control parameters are nonzero. No policy action was sampled and
no validation, test, or heldout outcome was accessed. This is training completion
only—not method improvement or qualification; Base remains the historical 30/50
and unqualified.

Run logs/config/source closure: `outputs/exp/20261120_043914_plug_orientation_constrain_random100/`;
weights: `outputs/ckpt/20261120_043914_plug_orientation_constrain_random100/`;
sealed protocol, input/catalog hashes, exact commands, pre/post 248-test records,
checkpoint audit, timing, and artifact ledger:
`outputs/exp/20261120_043914_plug_orientation_constrain_random100_envelope/`.
Inputs were the admitted random data at
`demos/PlugCharger-orientation-random-100_20261120_041640/` and unchanged fixed
data at `demos/PlugCharger-orientation-fixed-1000_20261119_184307/`. The actual
run-seeded sequential random-yaw deviation is documented and was not retrofitted
into the old preregistration or contract. Next step is parent audit, then
validation-only evaluation; heldout outcomes remain closed until a candidate is
frozen. No commit or push was made.

## PlugCharger Round 105 — randomized adaptation-100 collection admitted

Completed the explicitly authorized first-100 randomized-yaw source + exact
native-EE replay collection once. **118 ordered attempts** yielded the first
**100 eligible** cases; 17 failures were `lift_tracking_failed`, one was
`align_failed`. Native controls were **31,925 source + 31,925 replay** across all
attempts in **274.29 s**. Accepted five-bin occupancy is `[18,18,20,23,21]`;
actual accepted yaw range is `−0.387467…+0.392201 rad`. The existing
`PlugChargerOrientation-v1` environment class was used with each random target
yaw actually overridden and preserved in case/native/consolidated metadata.

Independent audit confirms all 100 source/replay outcomes against the unchanged
native 5 mm / 0.2 rad full-pose predicate, finite T+1 observations/states and
bounded 7-D actions, exact raw action streams, exact initial states/observations,
and exact consolidated source trajectory values. Dataset construction yields
28,202 random and 279,625 fixed windows; the production first item has `(2,46)`
observations, `(16,7)` actions, and `(2,)` hint. Random training HDF5 SHA-256:
`2df8ffa9b95c27e2ef9f7a471edefb68f73c8f6123769e9858eb9fa53f2fe26c`; JSON:
`77e9fed3070ad326b0323ace70b6b7015ad4394bf374a71193487685db4f6c0f`.

The exact 25-file suite passed **248 tests before and after** (23.61 s / 22.68 s).
All 32 archived source closure files match current bytes; frozen data, contract,
Base checkpoint, and catalog-set bundle identity remain unchanged. The run
archive includes exact test list/logs, preregistration, wrapper command/exit/time,
post hashes, collection summary, and independent admission audit:
`outputs/exp/20261120_041640_plug_random_adaptation_collection_envelope/`;
source/replay attempts and consolidated admitted data:
`demos/PlugCharger-orientation-random-100_20261120_041640/`.

**Collection RNG deviation:** requested per-case
`default_rng(candidate_scene_seed + 1703)` was implemented as one sequential
`np.random.default_rng(collection_run_seed=1 + 1703)` stream for all 150 planned
yaws. The exact list predates physics in the unchanged preregistration. Uniform
`±pi/8` support, disjoint ordered 118-scene eligibility/coverage, and admission
rules are unchanged; selection was not based on policy outcomes. Admit the 100
rows for forward study only under the actual run-seeded algorithm. No rerun,
new data, reselection, old-preregistration adaptation, or source fitting was
performed. See the envelope's `collection_rng_deviation.md` for the full record.

Protocol caveat: pre/post tests completed within the 180 s cap using Python's
subprocess timeout rather than a TERM-then-KILL shell wrapper; no timeout signal
was needed. UTC starts/elapsed are recorded but individual monotonic subprocess
starts were not captured. The frozen catalog bundle was reconciled to its
historically pinned catalog-set SHA; individual catalog files were not separately
snapshotted in the pre-run registration, so no direct per-file pre/post claim is
made.

**No optimizer update, training checkpoint, trained-policy inference, validation
or test scene, or heldout outcome was run.** This admits data only; it is not a
method-improvement claim and does not alter the historical frozen Base 30/50
result or its unqualified status. Continue only with the authorized 30k balanced
512/512 adaptation training; leave final/test splits closed. No commit or push.

## PlugCharger Round 104 — frozen Base adapter alias fix (offline)

The adapter now snapshots the pinned Base tensors before loading a candidate
checkpoint, then verifies the loaded tensors against that independent snapshot.
The public-adapter regression changes one `base_net` weight while leaving frozen
Base metadata unchanged: genuine RED failed because the adapter accepted it, and
the fix rejects it. The existing real frozen-Base zero-control equivalence test
also passes unchanged. Exact 24-file historical suite plus adaptation tests:
**248 passed in 23.93 s**.

Evidence (raw RED/GREEN logs, exact test list, runtime, execution record and source
SHA-256s):
`outputs/exp/20261120_041213_plug_adapter_frozen_base_fix/`. Round 103's prior
missing historical RED archive remains disclosed and was not reconstructed. No
native/production collection, training, or evaluation outcome was run; no
ManiSkill source, checkpoint, or historical Base contract was changed.

## PlugCharger Round 103 — initial frozen-v1 Constrain chain (offline implementation)

The additive forward contract and initial collector/train/validation implementation
are in `plug_orientation_adaptation.py` and
`plug_orientation_adaptation_contract.json`. Exact 80k Base-v1 checkpoint and EMA
identities are pinned; the historical v1 contract, results, v2/v3 artifacts and
closed Base test are unchanged. The final heldout runner is deferred until a
candidate passes validation.

Development evidence and exact 24-file historical suite list plus adaptation tests:
`outputs/exp/20261120_035615_plug_adaptation_development/` (**247 passed** in
22.60 s). This includes fake-physics collector/validation fixtures, two toy
optimizer updates, and the real frozen Base CPU sampler equivalence smoke. No
production collection, native simulator controls, production optimizer updates,
or final/test outcome was run; no checkpoint was trained or saved for production.
Independent review is required before running the documented production CLIs.

## PlugCharger Goal-loop Round 100 — fresh v2 scaled-model training complete (validation pending)

One authorized fresh seed-1 v2 run completed **100,000/100,000 CUDA updates** on
the RTX 5070 in **3,314.40 s** outer wall time (exit 0). TRAIN-only max-absolute
scales were freshly fit before training from all 280,625 rows in the admitted
1,000-episode HDF5:
`[0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
0.283505916595459, 0.2955075800418854, 0.30000001192092896, 1.0]`;
full-row action SHA-256 is
`fa8db7e570b15e475385a0398af66e142439cf9d7246d7ebe427b8400b1a44f3`. All five
distinct v2 checkpoints (20k–100k) passed CPU-only format/config/provenance,
finite-weight, exact raw/EMA float32 scale-buffer, optimizer-step, EMA-step, and
SHA-256 audits. The 1,000-row finite training log ends at update 100,000 and LR
0. The exact required 24-file suite passed **239 tests both before and after**.

Logs/config/source archive:
`outputs/exp/20261119_235008_plug_orientation_base_scaled_v2/`; model weights:
`outputs/ckpt/20261119_235008_plug_orientation_base_scaled_v2/`; preregistration,
exact commands, scale fit, input/source hashes, timings, regression logs, and
checkpoint audit:
`outputs/exp/20261119_235008_plug_orientation_base_scaled_v2_envelope/`.
Admitted data and frozen catalog identities, old v1 contract/checkpoints, and
current source closure remained unchanged. No trained-policy inference,
validation/test scene, or holdout outcome was run. Original v1 remains unchanged
at 30/50 with its fixed test closed; v2 training completion is not validation or
qualification. Parent review precedes full v2 fixed validation (gate ≥40/50); its
fixed test requires that gate, and qualification requires ≥80/100. No
Constrain/09 training is authorized. No commit or push was made.

## PlugCharger Goal-loop Round 99 — v2 scaled-buffer and fixed-test admission correction

The v2 checkpoint validator now requires the raw and EMA state dictionaries to
contain exact, finite, positive float32 `(7,)` action-scale buffers matching the
validated TRAIN scale. This check runs before strict restore and before evaluation
backend/environment construction. Strict EMA restoration is covered with distinct
raw/EMA model parameters and confirms the source checkpoint's raw state is
unchanged. Genuine expected-red coverage reproduced both raw and EMA scale-buffer
mismatches before the fix.

The exact required 24-file suite passes **239 tests**. A bounded fake-backed
public fixed-test integration validates the complete v2 five-checkpoint selection
and full 50-scene summary/catalog identities, then records exactly one call to the
shared rollout backend; only that backend is stubbed. Raw logs, test list, source
archive, and SHA-256 records are in
`outputs/exp/20261119_234116_plug_scaled_minimal_review_fix/`. No physical/native
rollout, production model query, optimizer update, training, holdout outcome,
commit, or push occurred. v1 contract and checkpoint identities remain unchanged;
v2 remains offline-only, untrained, and unqualified.

## PlugCharger Goal-loop Round 98 — distinct action-scaled v2 implementation (offline only)

Implemented the prospectively authorized corrective model family
`plug_orientation_fixed_base_dp_action_scaled_v2`; the frozen v1 Base contract,
checkpoint bytes, 30/50 validation selection, summaries, and closed fixed-test
gate remain untouched. The additive v2 contract explicitly describes one fresh
seed-1/100k from-scratch run using the same admitted 1,000 TRAIN demonstrations.
Its sole intended model change is symmetric per-channel max-absolute action
normalization fit from all full TRAIN rows and the matching native-unit decode
using the same registered scale buffer. Observation representation, native
controller/actions, horizons, architecture, optimizer/schedule, EMA and stride 8
remain fixed. v2 checkpoints/adapters and validation/test selection are family-
and scale-bound; v1 selection cannot open v2 test evaluation. The ≥40/50 v2
validation gate and ≥80/100 qualification gate are unchanged.

Scale fit used all **280,625** rows from the hash-admitted TRAIN HDF5, and no
validation/test outcomes or fitted statistics. Exact scale is
`[0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
0.283505916595459, 0.2955075800418854, 0.30000001192092896, 1.0]`; full-row
action SHA-256 is
`fa8db7e570b15e475385a0398af66e142439cf9d7246d7ebe427b8400b1a44f3`.
All input/addendum and output identities are in `scale_statistics.json`.

The exact 24-file historical regression set plus the new scaled-family tests
passes **229 tests** in 19.92 s. Evidence and scale file:
`outputs/exp/20261119_232310_plug_action_scaled_development/`. Tests include a
synthetic CPU wrapper scale/decode and the longstanding uninitialized CPU sampler
smoke; there were zero training updates, native controls, collection, trained
checkpoint/model queries, or holdout/test access. This is a distinct corrective
method—not a pass or replacement of the original identity-action baseline—and
does not qualify Base or authorize Constrain/09. No commit was made.

## PlugCharger Goal-loop Round 97 — stride gate repair and offline action-scale audit

Repaired the original-v1 gate hole: fixed-validation selection and authorized
test summary verification now reject any explicitly recorded execution stride
other than 8; summaries written before the field existed remain readable as
stride 8. A direct authorized-test evaluation with stride 1 rejects before policy
or environment construction. Genuine expected-red output records both gate
regressions before the fix; the exact 23-file regression suite passes **221
tests** after the fix (180 s timeout, TERM then 10 s KILL grace). Original
Round-96 paired diagnostic artifacts, summaries, checkpoints and traces were not
rewritten.

Offline-only TRAIN action analysis used the frozen 1,000-episode HDF5/JSON
(280,625 action rows; 279,625 actual dataset windows) and the already-saved
80k fixed-validation events. The analysis made no model/policy query and used no
environment, native control, training update, test catalog, or test outcome.
Motion-channel raw
TRAIN action std is **0.0189–0.1613** versus gripper **0.9715**; TRAIN-only
late-insertion selection yields 27,231 rows. In 20 saved 80k failures, last-50
mean absolute +X policy action is **0.00630**, compared with **0.02065** source
mean +X action in the TRAIN late-insertion region; 19/20 failures have the
observed ~−16 mm X stall. This supports (but does not prove) an action-scale
representation hypothesis. Installed Pinocchio IK docs expose default
`eps=1e-4`; controller omits `eps` and supplies 100 iterations, so no 1 mm
dead-band is assumed.

The **proposed only, not trained or authorized** follow-up is a separate v2
format with per-channel symmetric max-absolute scales fit solely on the frozen
TRAIN actions, identity observation normalization and exact native action
inverse transform; then one fresh seed-1, 100k run with five checkpoints and the
same full 50-scene validation gate (≥40/50 to open fixed test; ≥80/100 to
qualify). The current v1 result remains 30/50 and unqualified; its test stays
closed. Full actual per-channel tables, failure-vs-source comparison, hashes,
commands, suite list/logs and bounded proposal are in
`outputs/exp/20261119_225137_plug_action_scale_audit/`. No commit was made.

That Round 97 directory additively marks the failed Round 96 stride-1 arm
`candidate_rejected` / `diagnostic_only` / `candidate_qualified: false`; no old
Round 96 artifact was edited.

Protocol clarification: the 23-file regression suite includes the longstanding
uninitialized CPU Agent sampler smoke (`test_official_agent_sampler_contract_smoke`),
which does make a synthetic random-agent forward call. No trained/candidate
checkpoint or policy was queried; the saved-trace analysis itself makes zero
model calls. This distinction is explicit in the round evidence.

## PlugCharger Goal-loop Round 96 — execution-cadence correction diagnostic (FAIL; test closed)

The prospectively registered six-scene paired diagnostic compared the original
8-control execution cadence with executing one predicted control before querying
the unchanged official CPU DP sampler again. Both arms used the locked 80k
checkpoint (`ceeb68b4…a2a33d130`), the same fixed-validation catalog scenes
`plug-fixed_validation-{000,001,002,003,005,009}`, and per-scene inference seeds
`1 + catalog index`. The model still receives `(1,2,46)` and predicts `(1,8,7)`;
only the number of committed actions changed. No training, checkpoint selection,
new data, test/holdout access, dependency change, or ManiSkill-source edit occurred.

The matched stride-8 rerun completed **2/6** (and reproduced both original
successes); stride 1 completed **0/6**, recovering **0/4** original failures and
regressing **both** original successes. The four failures still reached a closest
position error of about **16.0 mm**; at their best same-control pose points full
rotation error ranged **0.00075–0.00755 rad**. Thus this intervention is not a
candidate improvement and advances no gate. This bounded sample does not establish
a contact cause; the pending hypothesis is contact geometry/native IK dead zone
near the observed ~16 mm stall. The first policy input and first action matched
exactly in all six pairs. Later queries consume different DDPM noise draws, so the
comparison is cadence **plus** sampler-stream divergence, not an isolated causal
test of feedback cadence.

Native work was **6,471 controls** (2,871 stride 8 + 3,600 stride 1), within the
7,200 cap; elapsed evaluator time was 709.69 s. All 12 arm traces are complete.
The 23-file regression suite passed **219 tests** before and after. A separate
source-archive setup invocation failed before adapter/environment construction
and executed zero controls; its incident metadata is retained and disclosed
(the initial console traceback was not separately archived), then the corrected
pre-action archive verification passed. Locked checkpoint, contract, validation
catalog, original 30/50 results, source closure and checkpoint-embedded dataset
hashes reconcile unchanged. The fixed Base test remains unopened because the
registered gate is still **30/50 < 40/50**.

Registration, source archive, exact commands/logs, paired arm results, per-control
events, pose diagnostics and post-execution audit are in
`outputs/exp/20261119_221526_plug_base_execution_correction_v1/`. No commit was
created. Next step is parent review of this failed correction and the bounded
contact-geometry/native-IK-dead-zone hypothesis—not full-50 validation or test.

## PlugCharger Goal-loop Round 94 — Base fixed-validation gate (FAIL; test unopened)

All five registered from-scratch Base checkpoints completed their full 50-scene
fixed-validation cohorts once, on the same catalog identities (seeds
2026124100–2026124149). Independently recomputing the native success predicate
from each saved terminal simulator state confirms the recorded outcomes:
20k **2/50**, 40k **17/50**, 60k **27/50**, 80k **30/50**, 100k **27/50**.
Total work was **115,851 controls** in **2,789.52 s**. Selection was frozen by
the registered max-success/latest-update-tie rule: update 80k, **30/50**. Since
the required test-opening gate is 40/50, the fixed Base test was not opened;
neither fixed Base qualification nor Constrain/09 is established. The random
final-test catalog remains unopened.

The exact 22-file regression suite passed **216 tests before and after**. Run
outputs and all per-control actions, observations and native states are in
`outputs/exp/20261119_205526_plug_orientation_base_validation/`; preregistration,
independent terminal-state audit, source/input hashes, timing, logs and artifact
inventory are in sibling
`outputs/exp/20261119_205526_plug_orientation_base_validation_envelope/`.
The validation selection SHA-256 is
`80436f6f0c6c524837b0da496a21f69915236a0c824158dd8e4f6ce8b972655c`.
Continue diagnosis using validation evidence only; do not inspect or use final
test outcomes to choose or adapt a policy.

## PlugCharger Goal-loop Round 93 — from-scratch Base training complete

The authorized seed-1 production run completed **100,000/100,000** updates on
the NVIDIA RTX 5070/CUDA in **3,348.13 s** outer elapsed time (exit 0). The
frozen dataset and all four pretraining scene catalogs remained hash-identical;
the exact 22-file regression suite passed before and after the run. The 1,000-row
training log is finite, and all five registered checkpoints (20k–100k) passed
format, config, provenance, finite-weight, optimizer/EMA-step and SHA-256 audits.
No validation or test policy outcome was measured, so this is training completion,
not Base qualification. Training outputs:

- Logs/config/source archive: `outputs/exp/20261119_194656_plug_orientation_base/`
- Weights: `outputs/ckpt/20261119_194656_plug_orientation_base/`
- Preregistration, exact command/wrapper, run logs, hashes and audit evidence:
  `outputs/exp/20261119_194656_plug_orientation_base_envelope/`

Next: run the five-checkpoint fixed-validation cohort (50 scenes). The fixed Base
test remains unopened unless the registered validation gate is reached (≥40/50);
qualification requires ≥80/100 on that test. No Constrain/09 run is authorized.

## PlugCharger Goal-loop Round 90 — fail-closed Base/data corrections

Fixed the four reviewed Ticket 08 implementation failures: repeat a nonempty
training loader to the exact update target; bind evaluation to current contract,
training data, dataset manifest, and complete frozen catalog identities before
environment/policy construction; reject duplicate checkpoint bytes and bind the
five candidates to the fixed-validation catalog and complete hashed evaluation
summaries; and continue after only the
precisely recognized native 600-control-cap failure while retaining its partial
source count. Collection has an actual callback wall timer, per-control deadline
guards, and post-callback/admission/finalization deadline checks. Unknown failures
remain fail-fast. No native control, collection/replay, production update, policy
evaluation, or ManiSkill-source edit occurred. All frozen numeric controls and
budgets remain unchanged.

The exact historical 22-file single-threaded regression suite passes **215 tests**
in 20.79 s under a 180-second timeout with TERM and 10-second KILL grace. This includes the
official CPU DDPM sampler shape/finite smoke (zero optimizer steps), real public
multi-epoch training-loop coverage with optimizer/scheduler/EMA counts, dataset/
contract rejection before environment construction, the 600-cap partial-failure
continuation, and callback/per-control timeout checks. Logs and source archive are
retained at `outputs/exp/20261119_182600_plug_base_round90_fixonly/`.

## PlugCharger Goal-loop Round 89 — Ticket 08 implementation (offline only)

Added the fixed native-EE collection, scene-catalog, 46-D from-scratch Base DP,
and validation-gated fixed-test orchestration in `plug_orientation_data.py`,
`plug_orientation_scenes.py`, and `plug_orientation_base_dp.py`. Usage and
fresh-output/timeout commands are in [`PLUG_ORIENTATION.md`](PLUG_ORIENTATION.md).
The two new offline data/Base test modules pass **4/4** and **9/9**. The 20-file
historical Peg+Plug/Base regression set passed **195 tests**; with the new modules,
the 22-file suite passed **208 tests** (18.58 s, single-threaded). A real official
DP Agent CPU sampler smoke was finite at `(1, 8, 7)`, with no optimizer update.
No native collection, production training, checkpoint evaluation, or fixed-test
opening was run. Production 1,000-case collection and 100,000-update training
remain for the next reviewed round; no production experiment, checkpoint, or data
artifact was created. Offline development evidence (including the source archive,
test list/logs, RED/green record, and sampler smoke) is retained at
`outputs/exp/20261119_180756_plug_base_development/`.

## PlugCharger Goal-loop Round 87 — prospective-seal implementation (offline only)

The v4 runner now creates its prospective registration and archives a declared,
finite source closure before the first physical callback; it refuses prior
registration/archive/result output and verifies current/archive SHA-256 identities
before cases. Planned yaws and five-bin planned/accepted coverage are distinct
report fields. Because Round 86 results were already viewed, any subsequent run is
explicitly nonblind confirmation, not a new blind test. Round 86's quantitative
20/20 result remains intact, but its missing preregistration is not repaired or
certified retroactively. No physical execution or training was performed in this
implementation slice.

The exact 19-file offline Peg+Plug suite passes **192 tests** in 22.46 s pytest /
23.66 s outer, with 180 s timeout, TERM, then KILL after 10 s. Raw stdout,
timing, command and UTC timestamps are retained under
`outputs/exp/20261119_163707_plug_v4_preregistration_fix/`.

## PlugCharger Goal-loop Round 86 — v4 align-budget candidate (quantitative pass, protocol exception)

Only the align phase cap changed (100 → 180); the 600-control episode cap and
all controller, force/contact, IK/grasp, geometry, target and official success
logic remained fixed. The 19-file suite passed 193 tests (23.45 s; rerun 25.14 s
after the final setup guard). Pair: both preregistered seeds as specified by the
execution plan passed expert and exact replay, 1,316 controls / 5.646 s. Full 20:
fixed 10/10 and random 10/10 expert successes; replay 10/10 in both profiles;
ten distinct random target yaws covering five bins at `[2,2,2,2,2]` including
endpoints; 12,194 total controls / 50.609 s.
Independent HDF5 audit passed finite shape, exact-action, T+1 and initial-state /
observation checks with zero initial error in every case; official full-pose
evaluation passed throughout. Details: `outputs/exp/20261119_161534_plug_nativeEE_v4_preflight/`
and `demos/PlugCharger-nativeEE-v4-preflight_20261119_161534/`.
Captured runner stdout is empty; structured case JSON, aggregate JSON, HDF5 and
independent audit JSON are the retained records.

**Not a certified Ticket 07 PASS.** The pre-action immutable preregistration files
were not persisted before the pair/full stages because the shell heredoc piped
through `conda run` did not execute. The omission was identified after collection;
the audit marks quantitative gates PASS but protocol gate FAIL and records only
post-run source identities. No retroactive registration was created and no
training was started. The zero-action runner-setup attempt is retained under
`outputs/exp/20261119_161229_plug_nativeEE_v4_preflight/`; the corrected run used
fresh output locations. ManiSkill source and dependencies were not changed.

## PlugCharger Goal-loop Round 85 — v3 object-pose lift gate (diagnostic mixed)

Removed only the pre-lift native `is_grasping` abort and the object-loop
classifier-false abort. The classifier remains diagnostic. Existing 12-control
close, 80-control lift, native 7-D EE controller/targets, full charger-pose
tracking tolerances and official 5 mm / 0.2 rad predicate are unchanged. The
30 mm lift target is derived from the live initial charger pose. No contact or
physics intervention, controller tuning, dependency/model operation or
ManiSkill-source edit occurred.

The exact 19-file regression suite passed **193 tests**. Both fixed diagnostic
seeds ran once, followed by exact raw-action replay once per case: **1,098 total
controls**, **1/2 expert successes**, **1/2 replay successes**. Seed 2026113003
reached the +30 mm lift target to 1.257 mm / 0.000684 rad pose error, with
28.763 mm measured charger-height progress, but failed after the unchanged
100-control align cap (78.651 mm official position error); replay failed too.
Seed 2026113001 succeeded officially and its independent replay succeeded
(0.294 mm / 0.000442 rad final error). Both replays preserve exact 7-D action
arrays, T+1 state/observation alignment and exact initial observation match.
The first native `is_grasping=true` appears on the first lift control in both
traces; this and the observed object movement do not establish a contact
mechanism. Outer start UTC was not captured; monotonic elapsed is 5.165 seconds.

**Not a Ticket 07 PASS, no data are training-admitted, and no training started.**
Full registration, bounded logs, source archive/closure, hashes, phase-level
reconciliation and traces:
`outputs/exp/20261119_154809_plug_ee_lift_v3_pair/` and
`demos/PlugCharger-EE-lift-v3_20261119_154809/`. Historical rounds remain
immutable.

## PlugCharger Goal-loop Round 84 — streamed native EE teacher v2 (diagnostic FAIL)

Changed only the experiment-side planned-path execution: every FK target is
issued once, the final planned TCP target is held for 10 controls, then the
gripper closes for 12 controls while holding that same planned target. Intermediate
tracking errors are logged rather than used as a per-waypoint abort gate. The
actual native grasp predicate and official 5 mm / 0.2 rad success thresholds are
unchanged. Per-control traces include planned and bounded targets, measured TCP
before/after, waypoint/hold metadata, controller configuration, clipping, native
grasp state, hint and official errors. Teacher/replay provenance identifies
`native_ee_stream_final_target_hold_v2`.

The exact 19-file Peg+Plug suite passed **190 tests**. Both preregistered fixed-yaw
cases ran once and each exact action stream was independently replayed: historical
fail seed `2026113003` ended `charger_not_grasped` (121 expert + 121 replay
controls); historical pass seed `2026113001` also ended `charger_not_grasped`
(141 + 141). Both source/replay state-0 and observation-0 checks matched exactly;
7-D actions were identical and T+1 state/observation counts matched. Replay also
failed the unchanged official predicate. Combined work was **524 controls**.
Thus this two-case candidate does not pass and is not ready for the full 20-case
validation. The deepest common failure is grasp closure, not a tracking abort;
both traces show `native_is_grasping=false` throughout. For seed 3003, final
grasp-phase TCP tracking was about 1.05 mm / 0.0011 rad, while closing moved the
charger and ended at official errors 288.090 mm / 0.492774 rad. For seed 3001,
the final grasp-phase tracking error was about 1.26 mm / 0.0020 rad; closure ended
at 193.191 mm / 0.939330 rad. This is not a success, Ticket 07 pass, training
admission, or authorization to train.

Registration, archived source closure and matching post-run hashes, per-seed
expert/replay records and the native traces are retained in
`outputs/exp/20261119_153718_plug_ee_stream_v2_pair/` and
`demos/PlugCharger-EE-stream-v2-diagnostic_20261119_153718/`. No ManiSkill source,
dependency, model or checkpoint was changed; no full 20-case run, training,
commit, or push occurred. Historical Rounds 83 and earlier remain intact.

## PlugCharger Goal-loop Round 83 — bounded native EE teacher pair (diagnostic FAIL)

Fixed the bounded-target pose-type mismatch and flat HDF5 observation handling;
added source/replay state and initial-observation gates before the first replay
action, explicit per-seed planner RNG setup, native controller calibration through
the real `_step`, and fake HDF5/torch run-case coverage. The exact 19-file suite
passed **188 tests**. One authorized run of the two fixed-yaw cases used seeds
2026113003/2026113001 and the unchanged 600-control cap. Both expert attempts
ended in `grasp_tracking_failed` at 212/207 controls; neither passed the unchanged
full-pose predicate (terminal errors 290.564 mm / 0.483186 rad and 193.191 mm /
0.939333 rad). Exact diagnostic replay of both retained action streams matched
state 0 and observation 0 exactly, preserved all 7-D actions and T+1 alignment,
and also failed the native predicate. Total teacher/replay controls: 838. No
training admission or Ticket 07 pass is claimed.

Attempt logs, replay audit, test list/results and hashes:
`outputs/exp/20261119_152051_plug_ee_teacher_native_pair/`. Raw expert/replay HDF5s
and full step traces:
`demos/PlugCharger-EE-teacher-native-pair_20261119_152051/`. Prior Round 82 and
all historical counts/outcomes remain unchanged. No training, ManiSkill source
edit, dependency installation, commit, or push occurred.

## PlugCharger Goal-loop Round 82 — online native EE teacher (INCOMPLETE; setup cap reached)

Implemented `plug_orientation_ee_expert.py` as a privileged-goal online teacher
with planned reach/grasp motions executed only through native `pd_ee_delta_pose`
controls, followed by closed-loop lift/alignment/insertion. Added exact-action
independent replay capture/verification and native controller/full-pose regression
tests. The exact 19-file Peg+Plug selection passed **185 tests in 15.36 s**;
the new focused module passed 5 tests. Source RED/green, source hashes, runtime,
test list and native traces are retained under
`outputs/exp/20261119_145706_plug_ee_teacher_diagnostic/` and
`demos/PlugCharger-EE-expert-diagnostic_20261119_145706/`.

The bounded native pair did **not** reach a physical control: three setup-level
errors stopped the fixed historical-fail seed `2026113003` with zero controls
and zero replay. The first was MPLib's ndarray base-pose binding (fixed by the
existing experiment-side compatibility adapter); the next two were a mismatch
between SAPIEN `Pose` and the shared `PoseValue` interface in the new target
encoder. Both repair attempts were also zero-control. The preregistered maximum
of two diagnosed zero-action setup repairs is now exhausted; no further native
attempt was made. The seed `2026113001`, all independent replays and the overall
two-case candidate remain unrun. **No Ticket 07 PASS, training admission, or
training authorization is claimed.** No ManiSkill source, dependency, data, or
checkpoint was modified. Raw exceptions, immutable partial recordings, and
attempt accounting are preserved; this is not physical feasibility evidence.

## PlugCharger Ticket 07 Round 81 — deterministic native preflight (FAIL)

The exact registered 18-file suite passed **180 tests in 15.49 s wall time**.
One official-solver 20-case preflight then completed all four stages in 31.47 s
outer wall time (29.77 s pipeline-reported). It used paired fixed/random seeds
2026113000–2026113009, official solver default 15° grasp tilt, no threshold or
budget changes, and CPU EE replay. Results:

| Profile | Official expert | EE replay | EE replay rate | Expert controls | Replay controls |
|---|---:|---:|---:|---:|---:|
| Fixed | 8/10 | 4/8 | 50.0% | 1,887 | 3,112 |
| Random | 5/10 | 3/5 | 60.0% | 1,905 | 1,944 |

Combined expert/replay controls were **3,792 / 5,056**. The 3 accepted random
targets occupy **3/5** yaw bins and include no endpoint. The six native EE replay
failures are retained with full traces; fixed source IDs 3, 6, 7, 8 miss full
pose with position error 17.27–17.73 mm (rotation 0.173–0.224 rad), while
random source IDs 3 and 6 miss at 17.44/17.62 mm (0.186/0.194 rad). The strict
95% replay gate therefore fails in both profiles, and random expert feasibility
also misses the 8/10 gate. No threshold relaxation, retry, or candidate selection
was performed. **Ticket 07 FAIL; no data is training-admitted and Ticket 08 did
not start.**

Independent audit verified the 20-attempt denominator and saved episode IDs,
successful-source-only replay selection, finite native trajectories, T+1 state
alignment, 7-D EE actions, exact unchanged actions between raw and aligned replay
files, retained raw copies/alignment seals, destination initial-state capture,
46-D observation contract, and unchanged archived/current source hashes. The
same-seed fixed/random initial charger, robot, and goal-center fields matched.
Full machine-readable audit, per-episode outcomes, hashes, and summaries:
`outputs/exp/20261119_143359_plug_orientation_preflight_deterministic_envelope/`
(`independent_audit.json`, `gate_decision.json`, `results_summary.md`); native
pipeline record/source archive:
`outputs/exp/20261119_143359_plug_orientation_preflight_deterministic/`;
all source and replay HDF5s:
`demos/PlugCharger-orientation-preflight-deterministic_20261119_143359/`.
The next slice is parent review and expert/replay failure diagnosis, not training.

## PlugCharger Ticket 07 Round 80 — strict replay provenance blocker fix

Replay now requires explicit boolean success provenance from the saved-attempt
map or source episode metadata (`expert_success`, `final_success`, or `success`);
missing or contradictory declarations fail closed. Destination initial-state
capture is accepted only when the actual replay environment reports
`pd_ee_delta_pose`. The exact 18-file regression suite passed **180 tests**.
RED/green logs and reproducibility metadata are in
`outputs/exp/20261119_141853_round80-minimal-blockerfix/`. This is implementation
verification only: no native planner, simulator control, collection, replay,
training, checkpoint load, or policy query was run. No physical thresholds,
solver, tolerances, or control budgets changed; native preflight remains gated
on independent review.

## PlugCharger Ticket 07 Round 79 — deterministic replay/collection development

Planner RNGs (Python, NumPy, Torch and MPLib) are initialized from the paired
planned scene seed before reset and reapplied after solver reset before planner
creation. Every nonempty expert attempt is retained with success metadata and its
actual sparse episode ID; replay filters to expert-success IDs while preserving
the failure denominator. Public replay now captures the destination's true
pre-action state/observation, validates it against canonical source state 0, and
aligns only the new converted file while retaining the raw conversion and
unchanged actions. `.alignment.json` sidecars seal raw/corrected/action hashes and
the initial-capture fingerprint. The official 15° solver is still default and
explicit tilt is forwarded through the pipeline.

The canonical 16-file Peg suite plus both Plug test modules (18 files total)
passed **178 tests**; raw expected-red/green logs, file list and timing are under
`outputs/exp/20261119_235000_plug_round79_development/`. This is offline
implementation verification only: no native planner, control, collection, replay,
training, or policy query. Ticket 07 is **not passed**; independent review must
precede the same-20-case native preflight. User authorization permits Ticket 08
training only after that PASS; Ticket 09 remains unauthorized.

## PlugCharger Ticket 07 Round 76 — reconstructed source-state EE candidate

Three experiment-only native candidates derive world TCP targets from retained
source `env_states[t+1]` by Pinocchio FK and issue native EE controls, with source
gripper commands and a fresh T+1 recorder. All six fixed fail/pass outcomes fail
the official 5 mm position threshold (terminal error 17.38–18.60 mm); no method
preserved the expected fail/pass split. Controls: 2,367 total, 596 maximum per
trajectory, within 6,000/600 caps. The 17-file Peg+Plug suite passes 171 tests.
**No Ticket 07 pass, training data admission, or training authorization.** Full
run reports, immutable source snapshots, target hashes, before/after traces and
metrics: `outputs/exp/20261119_230000_plug_replay_reconstruction_r76/`. New
trajectory artifacts are in the siblings `demos/PlugCharger-orientation-
reconstruction_r76_20261119_230000/`, `..._one_control_20261119_230000/`, and
`..._two_controls_20261119_230000/`. No original HDF5/JSON or ManiSkill source
was edited.

## PlugCharger Ticket 07 Round 75 — native replay diagnosis

The bounded fixed fail/pass pair was run through source-joint and official EE
replay. Source episode 2 / seed 2026113003 fails both ways (17.737 mm / 0.18859
rad joint; 17.734 mm / 0.18865 rad converted); episode 0 / seed 2026113001
passes both (1.350 mm / 0.000761 rad; 1.401 mm / 0.002681 rad). Failure source
replay diverges from the recorded charger path at post-action 97, although the
initial qpos/qvel/charger pose match exactly. This points away from a
conversion-only cause and toward source trajectory replay/contact/physics.
All-13 offline audit also found the converted saved state row 0 equals source
row 1 in 13/13, an independent recorder-alignment defect that does not alter the
live first physical state. **No fix, feasibility pass, training admission, or
final contract freeze is claimed.** Native asserted run: 1,287 controls / 6.45
s; 3,930 cumulative including the invalid harness attempt, below 6,000. Exact
17-file suite: 166 passed. Additional fail-case instrumentation found source FK
and native converted target agreement within 0.31 µm / zero rotation error for
all 203 controls; source/destination grasp checks agree (grasp maintained), while
TCP target tracking error reaches 14.50 mm. Total controls including all
instrumentation and the initial misindexed run were 5,826/6,000. Full report and traces:
`outputs/exp/20261119_203000_plug_replay_diagnosis_r3/`.

## PlugCharger Ticket 07 Round 74 — compatibility-v2 preflight paused

Fresh authorized same-plan preflight completed 10 fixed + 10 stratified-random
expert attempts and isolated EE CPU replay of all 13 expert-success episodes.
Expert results: **8/10 fixed, 5/10 random**; replay accepts **4/8 fixed,
3/5 random**. Random accepted replay outcomes cover only three yaw strata.
All 10 reset-pair comparisons pass; observation width is 46, converted replay
actions are finite 7D, and the unchanged native full-pose threshold is 5 mm / 0.2
rad. Actual control counts are 3,792 expert + 5,056 replay, within the respective
12,000 caps; pipeline elapsed 28.860 s. The exact 17-file suite passed **166
tests both before and after**. All 16 source-closure archive/current hashes
match. This **fails** random-expert and replay acceptance gates: **Ticket 07 is
paused, no training contract is frozen, and no training data are admitted**.
Round 72's 0/10 + 0/10 failure remains historical and unchanged. Tickets 08/09
remain unauthorized; no model/checkpoint query or ManiSkill source edit occurred.
Full reports, native attempt journals, independent audit and hash list are in
`outputs/exp/20261119_105049_plug_orientation_preflight_compat_v2/`, its sibling
`..._envelope/`, and
`demos/PlugCharger-orientation-preflight-compat_v2_20261119_105049/`.

## PlugCharger Ticket 07 Round 73 — compatibility development only

Reused Peg's existing experimental MPLib adapter at the default Plug collector
boundary, before loading the official solver. A strict fake official planner
exercises actual `collect()` default setup with the binding's ndarray rejection
and `use_point_cloud` signature constraint: regression red before the change,
green after. Full unexpected per-attempt tracebacks are now journaled. The
prospective source closure adds the adapter and installed base motion-planner
source. Exact 17-file Peg+Plug regression suite: **166 passed in 13.53 s**. The development
archive, runtime versions, source hashes, raw suite output and timing are in
`outputs/exp/20261119_104118_plug_mplib_compat_development/`. No native planner
or physics/control, collection, replay,
training, policy/checkpoint query, or ManiSkill source edit occurred. Round 72
remains unchanged; fresh native preflight requires parent review and explicit
authorization.

## PlugCharger Ticket 07 Round 72 — native preflight paused

The single preregistered native pipeline completed its 10 fixed + 10 stratified
random expert attempts, but every attempt failed before the first control with
an `mplib.Pose` versus float32 NumPy pose `TypeError` in the official
PlugCharger solution. Expert successes: **0/10 fixed, 0/10 random**; controls:
0; isolated CPU replay episodes: 0; replay successes: 0. The supervisor's
zero-source replay completion is not a pass. Independent reset-pair checks
passed all 10 seed pairs for charger/robot/goal-center independence, and all 13
registered source hashes matched. No usable live hint, replay dimension, or
full-pose success evidence was produced.

**Disposition: Ticket 07 paused; no training contract frozen.** The historic
Peg negative result is unchanged. No retry, training, policy query, Ticket 08/09
work, or ManiSkill source edit occurred. Exact budgets, commands, partial/failed
attempt evidence, hashes, and the bounded next-review requirement are in
`outputs/exp/20261119_102856_plug_orientation_preflight_envelope/`; native
artifacts are in `outputs/exp/20261119_102856_plug_orientation_preflight/` and
`demos/PlugCharger-orientation-preflight_20261119_102856/`. Focused pre-run
tests passed **23**; the exact 17-file Peg+Plug regression suite passed **164**.

## PlugCharger Ticket 07 Round 71 — offline minimal corrections

Fixed the replay preflight's observation-shape error using ManiSkill's public
`BaseEnv.get_obs(unflattened=True)` API; its fake environment now returns the
default flattened tensor unless explicitly asked for the observation dict.
Timeouts now retain phase records and captured stdout after process-group
termination, with unknown controls for unreported work and explicit measured
completed controls. The prelaunch source archive has a finite enumerated source
closure with package-preserving paths, runtime version metadata and source/archive
hash validation before the first child. These hashes are only preparation for a
future native run; no native source archive/run evidence is claimed here.

Verification: the full fake Plug orchestration suite passes **23 tests**. No
expected-red run was retained for Round 71; this is not reconstructed. No native
physics, collection, replay, training, checkpoint load, or commit occurred.

## PlugCharger Ticket 07 Round 70 — offline supervisor implementation

Added `plug_orientation_pipeline.py`: one CLI runs four isolated child phases
(fixed/random expert collection, then fixed/random official CPU replay), creates
an immutable preregistration and source snapshot before starting children, shares
expert/replay control caps across both profiles, enforces combined 3,600-second
stage and 9,000-second whole-run deadlines, and terminates timed-out process
groups with TERM then KILL after 20 seconds. Worker fixes include initialized
cross-episode replay control accounting, initial saved target-yaw derivation,
finite converted action/state/observation checks, T+1 observation alignment,
and observation leak checks.

The 21-test Plug suite passed offline, including calls through the actual
`collect()` and `replay_successes()` worker functions with fake recorder,
Gymnasium environments and public replay API, plus four-stage fake-child budget,
nonzero-stop, timeout-cleanup and no-overwrite coverage. This is **not** native
coverage or execution evidence. The implementation awaits independent review. No native
expert, simulator, collection, replay, training, model/checkpoint load or query
ran. Ticket 07 remains unvalidated; Tickets 08/09 remain blocked. The green log,
current source archive, and SHA-256 list are retained under
`outputs/exp/20261119_231500_plug_round70_supervisor/`. No expected-red log was
captured for this round and none is reconstructed.

## PlugCharger Ticket 07 preflight foundation — development only

Added `plug_orientation.py`, a yaw-only subclass with native PlugCharger success
semantics and the strict state-observation boundary, plus
`plug_orientation_preflight.py` for paired 10-scene fixed/random plans, injected
target options on the stock expert's own reset, post-reset state capture,
per-attempt 600-control guarding, and failure journaling. The task-specific draft
contract and proposed (not frozen) acceptance/data/training gates are in
`PLUG_ORIENTATION.md`. Exact task source was inspected read-only. No ManiSkill
source was modified. No native expert, simulator, replay, training, or policy
query occurred. The eight new task/runner regressions pass; the exact 16-file Peg
suite plus this test passed **149 tests** (including failure-analysis and
excluding `test_constrain_adapter.py`). Red/green output, source snapshot and
UTC timing are retained in `outputs/exp/20261119_094707_plug_orientation_development/`.
Ticket 07's
forward user exception does not alter Ticket 06's negative Peg outcome or
authorize Tickets 08/09.

## Peg orientation random-500 update-30k vs terminal update-50k scorer development

Added `peg_orientation_entry_continuation_eval.py`, a frozen three-arm roster
(Base, random-500@30k, random-500@50k) over the existing 60-case outside-entry
bank and three paired inference seeds: 180 groups / 540 arms. Base and 30k→50k
are the only registered comparisons; the shared Base is evaluated once per
draw. The runner uses explicit 334,800-control / 16,200-second stage / 21,600-
second whole-stage caps for both prefix preparation and suffix execution. Existing
six-arm defaults remain unchanged. This is scorer implementation only: no
native physics, fresh policy outcomes, training, data, or ManiSkill-source edits.

The 16-file Peg regression suite passed **141 tests** (14.47 s pytest / 15.71 s
outer). All three real frozen adapters passed a one-row offline `(1,2,39)` →
finite `(1,8,7)` smoke using the bank's recorded live relative-yaw hint; the
50k adapter reports step 50,000. Checkpoint SHA-256 values match before/after.
Logs, timings and smoke record are under
`outputs/exp/20261119_164500_peg_entry_50k_eval_development/`. The first smoke
logging attempt stopped after successful queries while formatting a missing
Base metadata `step` field; the corrected offline smoke is separately retained.
The initial test collection also caught an indentation error, and a subsequent
three-arm assertions were corrected; an intermediate suite failure was observed
before final verification, but its raw log was overwritten by the final pass and is
not reconstructed. Earlier console-only collection failure was not captured.
No native evaluation was run.

### Round 66 continuation scorer minimal fixes (offline only)

The generic aggregator now retains its prior adjacent curve edges and adds
deduplicated Base-versus-every-Constrain edge counts; the full fake three-arm
run asserts the `base->cn500_50k` paired counts. The public continuation entry
point constructs its own three adapters when none are supplied; its regression
patches the continuation factory and fails if the six-arm factory is called.
The six-arm roster, defaults and budgets remain unchanged. The exact 16-file
suite passed **141 tests** (13.35 s pytest; 180 s timeout, TERM then KILL after
10 s). Archive of the final current runners and test, raw suite output, UTC
times, outer timing and SHA-256 hashes are retained in
`outputs/exp/20261119_063023_peg_entry_continuation_round66/`. The earlier
development directory does not contain the source archive previously claimed;
this round archives the current bytes and makes no retrospective claim about
that earlier smoke. No expected-red log for these two defects was retained or
reconstructed. No native physics, model query, training, data/checkpoint change,
or ManiSkill-source edit occurred; the continuation remains unapproved for
native rollout.

## Peg orientation random-500 continuation — goal-loop round 64

Completed the authorized update-30,000 → terminal update-50,000 continuation:
20,000 new updates, exit 0, **625.432 s**. Terminal checkpoint
`outputs/ckpt/20261119_135500_peg_orientation_random500_extend50k/update_50000.pt`
SHA-256 `1e0b6246c7b01f7939f4e65d74255295cfd38e7c17982df0fb14ab7430bb060a`;
35k/40k/45k provenance checkpoints share that `outputs/ckpt/` run directory.
Experiment registration, command, raw logs, 200-row loss series, source archive,
environment, pre/post 139-test suite records, input hashes, and offline
checkpoint/state audit are under
`outputs/exp/20261119_135500_peg_orientation_random500_extend50k/`.
Parent AdamW/EMA and the frozen Base/data/model/hint were retained; sampler
cursor restart is disclosed. Because the prior cosine schedule ended at zero,
the extension used its own 500-warmup/20k-cosine LR phase. It is not a 50k
from-scratch schedule, and no validation/native physics or policy query was run.
Terminal evaluation remains required before any policy-improvement claim.

## Peg orientation fix-only Round 63 — fresh initialization seed ordering

Moved fresh-only Python/NumPy/Torch seeding before Base/CN model construction,
making randomly initialized hint-encoder state reproducible for seed 1. Resume
still restores checkpoint RNG/state after all construction and before loader
iteration. The actual fresh `train()` seam regression was captured RED before
the fix and passes afterward; the existing parent-RNG restore regression also
passes. Round 62's original expected-red log was missing and is not reconstructed.
Its independent actual review result remains **138 tests**; this corrected Round
63 16-file suite (including failure-analysis and entry-scaling, plus existing
Peg/Constrain tests) passes **139 tests in 13.45 s**. Exact expected-red, harness
failures, suite logs and timing are retained at
`outputs/exp/20261119_180000_peg_entry_scaling_development/round63/`. A different
initial 16-file list's 140-test result is retained but is not represented as the
required suite. No training, native physics, checkpoint/data/source mutation,
full baselines suite, or commit occurred.

## Peg orientation native round 61 — outside-entry checkpoint/data-size comparison

Fresh run: `outputs/exp/20261119_025526_peg_entry_data_scaling_v2/`. All
**180/180 groups and 1,080/1,080 arms** completed once (exit 0), using **306,628
native controls including full replay** in **7,203.799 s**. It stayed below the
669,600-control, 32,400-s native, 43,200-s whole-stage, and 300-s/arm budgets.
No retries, new scenes, videos, training, checkpoint selection, or retuning.

Primary Base / 100@30k / 200@30k / 500@30k all-condition outcomes: random
**11/90 / 30/90 / 27/90 / 25/90**, fixed **62/90 / 60/90 / 57/90 / 60/90**.
Secondary Base / 100@15k / 200@30k / 500@25k: random
**11/90 / 33/90 / 27/90 / 27/90**, fixed **62/90 / 60/90 / 57/90 / 57/90**.
Per-condition counts/rates, per-seed results, primary and secondary binary
success patterns, monotonicity and paired model transitions are retained in
`results_summary.md`/`.json`; transitions were recomputed from the final raw
per-arm suffix traces, including Base-vs-200 and Base-vs-500 edges.

Interpretation is deliberately descriptive: condition/profile response is mixed,
not consistently monotonic, and does not establish a scaling law. Six origins
per profile recur across all conditions and seed offsets; one training seed per
size and incomplete retrospective provenance for the historical 100-demo
checkpoints preclude an isolated data-count effect or inference. No model was
selected as “best” and no full-task gate is claimed.

Independent audit confirms all 21 registered inputs and 38 source-closure files
match their sealed hashes; all 60 bank rows match raw pair action hashes and
prefix lengths, with source IDs resolved by seed-to-`episode_id` (not catalog
index). Handoff history is exactly matched, measured physical-state error is 0
(≤1e-5), and exposed controller state keys are empty; hidden environment state
is not claimed. Original zero-native setup and failed-validator records remain
unchanged; the fresh v2 validator cardinality correction is explicitly
documented. Pre-suite 135 tests were reused by hash from the zero-native setup;
the actual post-run suite independently passed **135 tests in 13.32 s**. Raw
execution records, exact command/timing, input/source posthashes and final
artifact hashes are in that experiment directory. No ManiSkill/source/data/
checkpoint edits or videos. See `PEG_ORIENTATION.md` for the fuller caveats.

## Peg orientation fix-only round 60 — scaling replay budget repair

The six-model scorer now applies its 669,600-control / 32,400 s limits to full
prefix replay, without changing the paired scorer's default two-arm limits of
223,200 / 10,800. Real `_prepare` boundary regressions pass, and the exact
16-file Peg suite passed **133 tests in 16.46 s**. Expected-red, collection
failure, and full-suite logs are retained in
`20261119_180000_peg_entry_scaling_development/round60_*.log`. No native
execution or ManiSkill source changes. The Round 59 expected-red raw log was
missing and was not backfilled; this round captured the newly found regression
red before implementation.

## Peg orientation round 59 — offline six-model scaling scorer development

Audit, source configs, offline six-adapter smoke logs and runner evidence are in
`outputs/exp/20261119_180000_peg_entry_scaling_development/` (`comparability_table.md`,
`checkpoint_comparability_audit.json`, raw configs and `adapter_smoke.json`). Six checkpoint
identities are distinct and unchanged. All adapters return finite `(1,8,7)` for
39-D observations; the Constrain adapters verify the frozen Base EMA. Coverage
manifests establish random seed nesting 100→200→500, but historical checkpoint
metadata does not establish complete provenance for every 100-demo input. No
full-provenance or isolated data-count claim is made.

The new `peg_orientation_entry_scaling_eval.py` uses all 60 bank rows for three
independent inference draws, giving 180 groups / 1,080 arms. Base and CN200@30k
are shared between the preregistered primary and validation-selected secondary
curves. Tests exercise full preparation/suffix orchestration with fake physics,
not mocked preparation/suffix: **129 passed in 14.36 s** in the exact required 16-file
suite. No native rollout or training was started. The six-model execution is
pending independent parent review. It reports paired transitions and descriptive
binary monotonicity patterns but no fitted scaling law or inferential tests. It is
explicitly a three-size, single-training-seed comparison, not a formal
scaling-law discovery.

## Peg orientation round 58 — outside-entry paired policy comparison

The fresh native run is `outputs/exp/20261118_174230_peg_entry_policy_comparison/`.
All **180 pairs / 360 arms** completed once; native exit 0, **104,383 physical
controls** including both full-prefix replays, and outer monotonic elapsed
**2,318.40 s**. Frozen Base and Constrain checkpoints are identified in the
registration and input hashes; neither was modified. All pair handoffs matched
their paired history/controller/physical replay checks at maximum error **0**.
Independent trace recomputation confirms every official-success outcome and all
180 unique `(profile, condition, catalog index, replicate)` keys. Read-only
seed-to-source-episode audit confirms all 60 bank prefixes and complete action
hashes exactly match the original source trajectories.

The bank starts 19.8247–20.4765 mm outside the true front entry plane
`x=-peg_half_length`, unlike the older near-hole bank already inside the channel.
Pooled repeated draws per condition (six origins × three draws) are random
Base/Constrain: aligned **2/6**, lateral +5 **2/5**, lateral −5 **2/7**, yaw +5
**3/6**, yaw −5 **2/3**; fixed: **10/8**, **13/15**, **17/16**, **13/12**,
**9/6** respectively (each denominator 18). Random aggregate across conditions
is **11/90 vs 27/90 (+17.78 pp)** and fixed **62/90 vs 57/90 (−5.56 pp)**, purely
descriptive because the same origins recur in all conditions. There is no general
Constrain advantage: random results lean Constrain, fixed results are mixed and
favor Base on aligned and both yaw conditions. This conditional study is not a
formal gate, end-to-end result, causal claim, or Plug evidence. Three inference
draws are not independent scenes; pose-window admission does not imply zero
instantaneous velocity (bank sample diagnostics: peg linear 20.544–29.469 mm/s,
angular 0.590–1.314 rad/s). No video was preregistered or captured.

The historical 15-file list passed **130 tests**, but substituted
`test_constrain_adapter.py` for the required failure-analysis test. The correct
required 15-file list passed **123 tests in 11.42 s** after review, with raw logs,
exact command and 180-second TERM / 10 s KILL timing retained under
`post_review_required_suite/`; this is not retroactive pre-run coverage.
The historically substituted list also has a bounded post-completion rerun.
Pre-suite raw logs are retained, but its outer UTC/monotonic timing
and shell timeout wrapper were not captured; this deviation is disclosed in
`preregistration_deviations.json`. All **94** registered input hashes and **33**
finite source-closure hashes match pre/post. Detailed per-seed/per-condition
tables, raw pair traces, native timing, logs, environment versions and independent audit are in `results_summary.md`,
`results_summary.json`, `native_attempt_001/`, `independent_audit.json`, and
`source_prefix_mapping_audit.json`.

## Peg orientation round 57 — outside-entry source episode mapping repair

Corrected `peg_orientation_entry_perturb_eval.py` to resolve each source HDF5
episode by matching the validation catalog scene seed to the original source
trajectory JSON's `episode_seed`. The numeric suffix in `source_scene_id` is a
catalog index, not a source episode ID. A real `run_comparison` regression first
reproduced the wrong loader arguments for random catalog 2 and fixed catalog 5
(2 and 5 instead of source episode 1); both pass after the fix. The unchanged
actual source JSON/HDF5 data were checked read-only: all 60 bank prefixes match
their resolved source trajectory exactly. Unique mappings and input hashes are
retained in `outputs/exp/20261118_173544_peg_entry_source_mapping_repair/`.
Its `migration_map.json` maps the original date-only directory while preserving
all original file bytes and historical internal paths.
The 15-file suite passed **123 tests**. No source bank or ManiSkill source was
changed; no native physics or policy inference was run.

## Peg orientation round 56 — outside-entry scorer implementation only

Added `peg_orientation_entry_perturb_eval.py` for the separate round-55
outside-entry pose-window bank at
`outputs/exp/20261119_164500_peg_entry_pose_window_initialization_v2/`; it does
not use the older inside-hole bank. The scorer validates all 60 admitted rows,
replays the complete recorded expert prefix plus initializer from each canonical
catalog scene, and checks physical state, actual latest-two observations, and
the public controller extent before policy inference. Its next-run design is 2
profiles × 5 conditions × 6 origins × 3 repeat draws, paired Base vs Constrain
(360 arms), with separate 223,200-control / 10,800-second native-stage and
14,400-second whole-stage caps. This round ran no native physics or policy
rollout. Both real frozen EMA adapters passed only an offline `(1,2,39)` to
`(1,8,7)` checkpoint contract smoke; checkpoint hashes were unchanged. The
actual `run_comparison` entrypoint was exercised across 30 pairs / 60 fake-physics
arms (2 profiles × 5 conditions × 3 seeds), using the real bank parser, prefix
replayer, suffix loop and result finalizer with strict adapter shape checks. The
tests also verify pre-inference paired-state checks, retained handoff geometry,
wrapper/archive setup, no-inference replay mismatch, and journaling/control
accounting after an injected Constrain failure. The exact 15-file suite passed
**121 tests**. This is implementation verification, not a performance result,
new gate, or authorization to run the scorer.

Round-55 v1 clarification: the runner stopped its candidate search too early,
but its final predicate still enforced the full 11-frame pairwise geometry
bounds. Its historical result was **0/60 admitted**, not false early admissions;
historical records and the separate v2 bank remain unchanged.

## Peg orientation round 55 — outside-entry pose-window initializer bank

The valid fresh attempt is
`outputs/exp/20261119_164500_peg_entry_pose_window_initialization_v2/`.
All **60/60** cases independently pass the prospective 11-control pose-window
admission (random/fixed each 6/6 for aligned, signed hole-Y ±5 mm, and world-Z
yaw ±5°). No replacement, policy inference, checkpoint load, training, data
change, formal test, or Plug work occurred. The legacy instantaneous-velocity
mode remains the unchanged default; new `pose_window` admission records those
velocities diagnostically and must not be described as zero-substep motion.

Across admitted windows: maximum head/peg-center diameter **0.2464/0.2264 mm**,
full-quaternion rotational diameter **0.1498°**, arm-joint qpos range **0.001969
rad**. Realized lateral deltas are −4.327…−3.809 mm and +5.669…+6.176 mm; realized
relative yaw is −5.556…−5.200° and +4.443…+4.799°. Instantaneous velocity ranges
are peg linear **20.544–29.469 mm/s**, angular **0.590–1.314 rad/s**, and maximum
arm qvel **0.00102–0.05283 rad/s**. They are reported as measured diagnostics,
not as evidence of stationary instantaneous physics.

Totals: **8,225 prefix + 1,370 initializer = 9,595 native controls**, 59.52 s
runner / 62.86 s wrapper. Pre/post input/source identities (57 entries) and both
checkpoint-reference hashes reconcile; weights were never loaded. Exact 14-file
suite passed **113 tests** pre/post (12.08/12.14 s). The earlier v1 run is
retained separately: its candidate search stopped too early, while its final
predicate still enforced full-window bounds, yielding historical 0/60 admission.
It is not pooled or reclassified; its expected-red raw log was unavailable. Full independent
per-case recomputation and provenance are in the v2 `independent_audit.json`,
`post_execution_audit.json`, journals, source archive and timing files. Historical
rounds 52–54 and gate outcomes remain unchanged.

## Peg orientation round 54 — paired initializer settling calibration

The fresh model-free run is `outputs/exp/20261119_000000_peg_entry_settling_calibration/`.
It paired the original continuous initializer against one immutable actual-world
TCP-pose latch on the first pose-only-qualified state, for the first selected
random/fixed source-bank scene (ID 0 each) and all five frozen conditions. The
case/condition/strategy order, signed lateral bands, unchanged pose/yaw/velocity
criteria and five-control stability rule were registered before native actions.
No model was loaded or queried; there was no training, data, formal-test, policy,
or Plug work.

All 20 cases completed once: continuous **0/10 admitted**, fixed-TCP **0/10**.
Native controls reconcile to **2,650 prefix + 1,300 initializer = 3,950**;
27.35 s runner / 29.70 s outer elapsed. Every paired post-withdraw state/history/
head and prefix hash matched exactly; all 20 prefixes and native yaw preflights
passed. Outside clearance was 19.920–20.667 mm. Last-five peg linear speed was
22.855–23.690 mm/s and angular speed 0.597–0.863 rad/s, both over unchanged
limits; arm qvel remained below its unchanged limit. Fixed TCP had higher final
head error (2.331–3.682 mm) than continuous feedback (1.256–1.776 mm), with no
observed velocity-stability improvement. This bounded sample supports no
policy/root-cause inference and does not authorize changing thresholds.

Checkpoint bytes were hashed before/after but never loaded; all registered
source/input hashes reconcile. Per-control target/phase/action hashes and the
explicit per-control-timestamp omission are documented in `independent_audit.json`.
The exact 14-file suite passed **111 tests** pre/post. Raw expected-red and
green logs, command, timing, source archive, 20 case journals and analysis are
retained in the run directory. This is calibration only, not a bank admission,
formal gate, or learning result; rounds 52–53 and historical gate outcomes stay
unchanged.

## Peg orientation round 52 — physical perturbed outside-entry initialization

Fresh bounded attempt: `outputs/exp/20261118_152038_peg_entry_perturb_initialization/`.
The fixed six-ID cohort per validation group was random `[0,2,3,4,6,7]` and
fixed `[0,5,6,7,10,11]`, selected from already-admitted source-bank IDs before
new outcomes. It registered 60 cases across aligned, lateral ±5 mm in hole-frame
Y, and world-Z yaw ±5° (one perturbation at a time). All 60 replayed the expert
prefix and withdrew to measured true entry clearance 19.824–20.476 mm, but all
exhausted the 60-control perturb/settle allowance. **Admitted: 0/60**; final
head-position error was 1.256–2.690 mm, above the frozen 1 mm bound. Terminal
five-control velocity/yaw metrics were omitted by the runner's cap-return path,
so admission is unavailable and no policy scoring may follow from this bank.
No rerun/replacement or policy inference occurred; any repaired fresh native
attempt needs parent review.

Native work reconciles as 8,225 prefix replay + 3,900 scripted controls =
12,125; 72.10 s runner / 74.58 s outer elapsed. Prefix action hashes and
registered source/input hashes reconcile. Exact 14-file tests passed 106/106
pre and post. Native controller scale/frame and positive/negative yaw were
calibrated against the installed public controller. See `PEG_ORIENTATION.md`,
`analysis_report.md`, `initialization_reconciliation.json`, and `journal.jsonl`
for detailed evidence and missing-trace disclosure.

This run corrects the former “near mouth” interpretation: old bank x values
−0.04 to −0.05 m were already inside the true entry plane
`x=-peg_half_length`; official success x≥−0.015 m is near box center. Historical
round-46 eligibility/counts and round-50 policy outcomes are unchanged. No
training, data collection, checkpoint change, formal gate, or Plug work.

## Peg near-hole round 51 — hold control setup failure (no native action)

The authorized model-free control at
`outputs/exp/20261118_142454_peg_near_hole_hold_control/` stopped before
environment creation: the runner attempted to create an output directory that
had already been initialized for regression artifacts. Exit was 1; **0 native
controls, 0 prefixes, 0 arms**; no retry occurred. Raw traceback, outer timing,
wrapper, source archive/closure, and post-failure input hashes are retained.
The exact 13-file test suite passed **99 tests** pre/post (11.41 s / 11.33 s).
At that initial-failure point, the 15 random + 11 fixed one-pass hold outcomes remained unmeasured; this failure
does not update round 50, the 500-demo 23/54 results, or formal statuses.

Round-51 completion addendum: the parent authorized only the minimal setup fix
and one fresh attempt, retained at
`outputs/exp/20261118_143004_peg_near_hole_hold_control_v2/`. All **26/26** frozen
branches completed once: **0/15 random, 0/11 fixed** official successes; all
retained grasp and exhausted the 200-control suffix. Totals: **3,537 prefix +
5,200 hold = 8,737 native controls**, 46.18 s runner / 48.19 s outer elapsed.
Actual actions and replay checks reconcile exactly (max error 0); both in-pass
videos decode to 201 frames. Bank/catalog/demo/source hashes match round 50; the
exact 13-file suite passed **101 tests** both pre and post. This is only a
controller-hold conditional observation, not proof that all velocity confounds
are removed or that learned commands are generally necessary. No checkpoints
were loaded; model outcomes and gate statuses remain unchanged. Details and
limitations are in `analysis_report.md` and `independent_audit.json`.

Near-hole interpretation note: round 50 used moving, grasped, expert-prealigned
handoffs, not stationary randomized resets. All arms succeeded within 3–4
controls; expert residual motion and ceiling effects have not been excluded.
Seed labels 1/1001/2001 are offsets added to the catalog index, not training
seeds. Conditional success does not establish that learned actions are needed.

## Peg orientation round 50 — near-hole Base vs Constrain scoring completed

The fresh, parent-authorized CUDA attempt completed all **78/78 pairs** (156
policy arms) in
`outputs/exp/20261118_220128_peg_near_hole_policy_comparison_v2/`. On each of
the 15 random and 11 fixed eligible expert-reachable validation branches, both
frozen policies succeeded for all three draws: per seed `1`, `1001`, and `2001`,
random Base/Constrain **15/15 vs 15/15**, fixed **11/11 vs 11/11**. Descriptive
pooled counts are random **45/45 vs 45/45** (45 draws over 15 scenes) and fixed
**33/33 vs 33/33** (33 draws over 11 scenes); every paired table is
Base-only/Constrain-only/both/neither = **0/0/N/0**. Repeated draws are not
independent scenes. The other 14 of the frozen 40-scene catalog were bank-
ineligible/unconstructed, not policy failures.

All 78 unique pair keys and full arm traces reconcile. Native work was **21,222
prefix-replay + 486 suffix = 21,708 controls** (78,000 maximum), 181.72 s runner
elapsed / 183.94 s wrapper-monotonic elapsed; native and wrapper exit 0, with no
failed/partial attempt. Replay physical/history/head error was 0 (≤1e-5); public
controller `get_state()` was empty on both arms, so equality covers only that
empty public extent, not hidden controller buffers. Both modes stayed
`pd_ee_delta_pose`; both real histories were equal before inference. Four
preregistered in-pass videos decode to four frames each; hashes and the full
audit are in `independent_audit.json`. The exact 12-file regression passed 97
tests both before and after. Registered source closure and 55 selected
checkpoint/bank/data/catalog hashes reconcile.

This is near-hole conditional evidence on already-grasped, closely aligned
expert prefixes, not end-to-end performance, a formal gate, or Plug evidence.
It is not directly comparable to round 40's separate random **23/100** / fixed
**54/100** whole-task results and does not establish why whole-task performance
is difficult. The bank's retrospective fixed-stage amendment and unavailable
initial historical source bytes remain disclosed; this run does not repair that
provenance limitation. Full row-level results, hashes, timing, command, videos,
and interpretation are in the experiment directory's `results.json`,
`pairs.jsonl`, `analysis_report.md`, `independent_audit.json`, and
`preregistration.json`. No training, new data, weight/source edits, full-task
test, or Plug work occurred; historical gate outcomes are unchanged.

## Peg orientation round 49 — near-hole policy input batching repair (no native)

`_run_suffix` explicitly batches the identical latest-two 39d observation
history for both Base and Constrain adapters as `(1, 2, 39)`. A strict full-run
fake-adapter test captured the old `(2, 39)` failure then passed after the
minimal fix. The exact 12-file suite passed **97 tests**. Red/green logs,
commands, timings, and test list are in
`outputs/exp/20261118_215301_peg_near_hole_batching_repair/`. This fixes policy
input shape only; native scoring remains pending and round 47's failed attempt
is unchanged.

Near-hole bank source/timing and command-document provenance corrections are
additively recorded in
`outputs/exp/20261118_122546_peg_near_hole_initialization/provenance_correction.json`.
The amendment is retrospective; historical source bytes are not authenticated.
Near-hole policy scoring still requires replay-based controller, physical and
actual-history equality checks; no policy performance is reported by the bank.

## Peg orientation round 48 — near-hole comparison runner repair (no native)

Round 47's failed attempt and artifacts at
`outputs/exp/20261118_130407_peg_near_hole_policy_comparison/` are immutable.
The bounded runner repair fixes the actual task predicate call, preserves prefix
metadata for final geometry, hashes/archives the current outer wrapper rather
than a historical runner, recursively normalizes 0-D NumPy values, and records
global physical controls plus completed arm results on exceptions. Native stage
and whole-stage wall limits are 7,200 s and 10,800 s. The exact 12-file suite
passed **97 tests** in final pre/post runs; raw red/green stdout/stderr, command, timing, file list and
fake-environment orchestration coverage are retained in
`outputs/exp/20261118_132711_peg_near_hole_eval_repair/`. It exercised real
runner preparation/suffix/finalization with fake envs/adapters, not native
physics, actual checkpoints, or policy scoring. No native action, training, data
mutation, or Plug work occurred. This repair does not authorize retrying round
47. The old failure evidence and formal / random-500 comparison outcomes are
unchanged.

## Peg orientation round 47 — near-hole comparison setup failure

The registered Base-vs-Constrain near-hole comparison is at
`outputs/exp/20261118_130407_peg_near_hole_policy_comparison/`. The exact
12-file suite passed 91 tests before and after. Its sole native attempt replayed
142 controls from random validation index 0, then exited before policy inference:
the runner called the task-level one-argument `near_hole_eligible` with three
geometry arguments. Failure logs, empty first-pair journal, timing, source
archive/closure and a post-execution failure audit are retained. No policy result
or video exists, and no native retry/source edit occurred. This setup defect is
not a policy failure; a new native attempt requires a parent decision. Round 40's
random 23/100 and fixed 54/100 and all formal gate outcomes are unchanged.

## Peg orientation round 46 — near-hole initialization bank (no policy evaluation)

Registration, per-attempt logs and reports are in `outputs/exp/20261118_122546_peg_near_hole_initialization/`; expert-only diagnostic data/replay artifacts are in `demos/PegInsertionSide-near-hole-diagnostic-random_20261118_122546/` and `demos/PegInsertionSide-near-hole-diagnostic-fixed_20261118_122546/`. The held-out validation catalog indices were frozen at 0–19 in each profile (40 scenes; no replacement). Expert collection was random 15/20 and fixed 12/20; official EE replay accepted random 15/15 and fixed 11/12. The near-hole bank has 15/20 random and 11/20 fixed eligible branches. Every ineligible scene remains represented; these are expert-reachable conditional denominators, not whole-catalog policy rates.

The first eligible post-action state used the official hole-frame peg-head geometry, native grasp and noninserted checks. Head-x/radius ranges are random `[-0.049854,-0.040661] m` / `[0.040805,0.049920] m`, fixed `[-0.048915,-0.041266] m` / `[0.041382,0.049018] m`. Expert prefixes are replayed in `pd_ee_delta_pose`; future policy suffix remains capped at 200 controls. Neither frozen policy has been scored yet. Any next comparison must replay and verify the same prefix/history/controller state for both arms and must be reported only as near-hole conditional paired evidence, not end-to-end or formal gate evidence. Experts have already aligned yaw closely (ranges recorded in geometry audits); disclose this initialization easiness.

Random branch native work was not repeated after a summary-finalizer error: all per-scene progress was journaled and the JSON bank was reconstructed offline after fixing NumPy normalization and ensuring no-source scenes are logged. The fixed-stage runner amendment and all setup/native errors are preserved. Eleven-file Peg suite passed **88 tests in 11.39 s**. Historical 500-demo data, checkpoints, and prior round 37/40/gate outcomes are unchanged. No training, test/retention access, policy inference, or Plug work occurred.

## Peg orientation round 45 — bidirectional takeover diagnostic

The fresh, immutable registration and completed native diagnostic are in
`outputs/exp/20261118_102350_peg_orientation_bidirectional_takeover_v2/`.
The study now binds `validation_random_scenes` directly and checks the catalog
manifest split before registration; source closure and all five frozen input
hashes reconcile after execution. Eight selected training branches triggered:
recorded suffix and live official expert each succeeded 8/8; the frozen policy
succeeded 7/8. On all 12 random-validation scenes, 5 succeeded before the
control-296 handoff and 7 formed matched pairs. Policy continuation succeeded
0/7; live expert succeeded 4/7 (the other 3 completed planning without native
success), yielding both/expert-only/policy-only/neither = 0/4/0/3. Maximum
matched-prefix numeric error was 0. The run used 10,365 native controls in
263.90 s and retained 55 hashed progress journals. Expert controller-mode
switch/reset is a confound; this is not a formal gate result, OOD-cause claim,
or recoverability theorem. Full row-level analysis, pre-native setup incidents,
limitations and audit are in `analysis_report.md` and `attempt_native_actual_once/`.
The exact ten-file suite passed **83 tests** before native execution (11.53 s)
and afterward (12.30 s); command and stdout/stderr captures are retained beside
the registration.

Round 43 remains unreconciled: its exact outcomes and aggregate controls are
unavailable; do not infer 8,021 from its thin ledger. Its fixed-catalog binding
and validation seed/timing deviation are historical and are not pooled with
this corrected fresh run. This diagnostic does not rescore or change round 40's
met numerical gates (+20pp random, +2pp fixed) or its nonblind, single-seed and
provenance limitations; it does not authorize Plug work.

## Peg orientation runner repair — development only, no native execution

Goal-loop round 44 fixes four diagnostic-runner defects in
`peg_orientation_takeover_study.py`: recursive NumPy scalar normalization (also
applied to the preceding takeover preflight finalizer), unique full-row progress
journals linked from the execution ledger, validation policy seed `1+index`
applied after scene restoration while preserving the policy RNG stream across
the 296-control handoff, and validation analysis of the expert arm's own
`status="native_success"` field. Training remains seeded `10000+episode_id`.
No native planner, environment step, or study attempt was run in this repair.

The focused ten-file regression reproduced five expected failures before the
fix (**77 passed, 5 failed**) and passed afterward (**83 passed**, 11.49 s).
Raw stdout, exact command, TERM/10-second-KILL deadline, timing, and source
closure check are retained in
`outputs/exp/20261118_100309_peg_orientation_takeover_runner_repair/`.
The source-closure regression includes the runtime and both takeover test files.

The failed round-43 study at
`outputs/exp/20261118_175052_peg_orientation_bidirectional_takeover/` is
unchanged. Its finalizer raised while hashing `np.bool_`, then failed again
serializing the failure record; it retained only a thin progress ledger and no
full per-arm result rows/traces. Therefore exact native outcomes, native
success totals, and a definitive aggregate control total (including a claimed
8,021) are unavailable; no rows or outcomes are reconstructed from control
counts. Round-42's separate, fully retained two-positive-control result remains
2/2 in each arm, 571 controls, 6.21 s; it is not pooled with round 43.

The failed preregistration recorded the validation seed as `10000+episode_id`
and the attempt seeded before scene restoration. Round 44 corrects the runner
for a future **newly registered** main study to use validation seed `1+index`
after restore, with actual restored observation history and an uninterrupted
policy RNG stream across control 296. This is a development note only, not a
new preregistration or native-run authorization/claim. Fresh registration is
required before any future native study. No formal gate or Plug status changes.

## Peg orientation round 42 — live expert-continuation handoff preflight

The frozen two-episode training-set positive-control study is registered at
`outputs/exp/20261118_092013_peg_orientation_takeover_preflight/`; runner and
registered source snapshots, the package lock, preregistration, exact commands,
raw logs, native ledger, post-execution audit and post-hoc reconciled results are
retained there. The existing random-500 dataset and update-25,000 candidate were
unchanged; the candidate checkpoint was not loaded and there was no training,
data collection, policy rollout, validation/test/retention access, or Plug work.

Both preregistered training demonstrations (IDs 0/1; seeds 2026112000/2026112002)
reached the frozen native-grasped, not-yet-inserted head-x trigger. Their recorded
suffix positive controls succeeded in **2/2** (8/9 suffix controls); stock live
official expert continuation succeeded in **2/2** (47/35 suffix controls). Each
paired branch replay matched its two-observation history, robot qpos/qvel/root,
TCP, peg and hole poses at max absolute error **0**; the public switch from
`pd_ee_delta_pose` to `pd_joint_pos` also used zero physical controls, preserved
grasp, and changed those physical snapshots by **0**. This is two selected
training-branch feasibility evidence, not a general recoverability estimate or a
policy effect: the live arm has a controller-mode/target-reset confound.

All four arms completed: **571 controls / 6.21 s**. The runner then exited 1 in
post-rollout result hashing because a planner result contained a NumPy array. The
original failure and log are preserved; a read-only post-hoc reconciliation
retains the already-complete arm rows at `attempt_001/results_posthoc_reconciled.json`
and documents the reporting fault in `reporting_reconciliation.json`. There was
no native retry. The exact nine-file regression passed **70 tests** before and
after. Internal controller target-state snapshots were not serialized; this
limitation is disclosed in the post-execution audit. See
`examples/baselines/constrain/PEG_ORIENTATION.md` and the topic README for details.

## Peg orientation round 40 — random-500 candidate nonblind comparison

The frozen random-500 update-25,000 candidate was compared with frozen Base in
all six registered 100-episode cohorts at
`outputs/exp/20261118_061124_peg_orientation_random500_comparison/`. Random
Base/correct/fixed-standard/shuffled scored **3/23/4/4**; fixed Base/correct
scored **52/54**. Paired correct-vs-Base random Base-only/correct-only/both/neither
is **1/21/2/76** (**+20pp**),
meeting the ≥20pp threshold exactly; fixed retention is **+2pp** (`4/6/48/42`),
within the −5pp loss limit. Both numerical gates are met on this comparison.
It is explicitly nonblind due to prior test exposure and single-seed; prior
formal +18pp/+6pp and round-37 +12pp/−3pp outcomes remain separate and unpooled.
No Plug work was started.

The six cohorts completed once: 600 episodes, 158,647 controls, 7,015.80 s;
scene-0 videos were captured in-pass. 554 registered source files and 22 inputs
reconciled. The exact eight-file regression passed 64 tests before and after.
Full per-scene results, hashes, paired tables, yaw bins, timing, and audit are in
the experiment directory's `results.json`, `results.md`,
`execution_ledger.json`, `post_execution_audit.json`, and regression artifacts.
Two pre-native failures (Base output-directory collision; fixed-standard CLI
label typo) are disclosed in the ledger. Their stderr hashes/exit metadata remain,
but retry log-path reuse overwrote their raw stdout/stderr bytes; they are not
retained. Each corrected cohort ran once. Current report corrections and artifact
stability hashes are in `reporting_correction.json` and
`reporting_artifact_audit.json`.

## Peg orientation round 39 — random-500 fresh training and validation

The admitted 500-demo package received the preauthorized fresh seed-1, 30,000-update run and all 12 validation cohorts at `outputs/exp/20261118_035943_peg_orientation_random500_study/`; six checkpoints are at `outputs/ckpt/20261118_035943_peg_orientation_random500_study/`. Training elapsed 933.76s; validation totaled 600 episodes / 132,933 controls / 6,174.02s. Random/fixed validation scores at updates 5k–30k were **15/27, 18/32, 19/33, 18/34, 18/34, 18/32**. The fixed≥34, maximize-random, latest-tie rule selects update 25k EMA (**18/50 random, 34/50 fixed**), SHA-256 `73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df`.

The candidate is validation-only: paired net vs reused Base is +14/50 random and −2/50 fixed; aligned raw-score changes vs historical random-100 and random-200 candidates are respectively (+3, −4) and (−2, 0) successes. Neither old candidate was rerun. This does not show a causal sample-count effect or alter the historical negative formal gate. All 555 source paths and 13 inputs reconciled; the eight-file regression passed 64 tests both before and after the run. No final-test, fixed-retention, data collection, or Plug work occurred. Full records: `preregistration.json`, `results.json`, `results.md`, `execution_ledger.json`, `candidate_freeze.json`, and `post_execution_audit.json`.

## Peg orientation round 38 — random-500 data expansion (data-only)

The forward user objective expanded Peg random demos to 500; this does not
rewrite the round-37 negative gate, qualify Peg, authorize Plug, or imply that
training has improved. `demos/PegInsertionSide-orientation-random-500/` contains
the original random-200 HDF5 episodes byte-identically followed by the first 300
accepted rows from `demos/PegInsertionSide-orientation-random-additional300_20261118_033057/`.
Collection obtained 350 expert successes in 544 attempts; official CPU replay
accepted 325/350 and rejected 25. The extra accepted 25 are retained in the
additional source. Package HDF5 SHA-256 is
`1fe27c25346d1117442d14bdadaf6037717943ea0b86090bcece2556ee008044`; JSON
`097bd41280296444db05df6a6030ac2f86bc9ccae799545fdd43e40d231dbf39`; coverage
`ad7011bdf53f60fa4ac8dc61dd5ef281a456bfa7f30206976ccb66f568d7f3c3`.
Registration, commands, logs, timing, independent audit and bounded deviations
are in `outputs/exp/20261118_033057_peg_orientation_random500_data/`. A minimal
coverage-composition seam was added after a failing audit and a red/green test;
the required eight-file regression passed 64 tests. This round did not train,
evaluate policies, access final-scene outcomes, select a checkpoint, or work on
Plug. The prospective fresh 30k-update training/validation protocol is recorded
in the registration only; its execution is a separate next stage.

## Peg orientation round 37 — corrective nonblind formal retest (gate not passed)

The one authorized post-diagnosis corrective retest completed all six fresh
cohorts in `outputs/exp/20261117_230000_peg_orientation_corrective_retest/`:
random Base/correct/fixed-standard/shuffled and fixed-retention Base/correct.
The frozen update-30,000 random-200 correction was used without further
selection; unchanged Base counts reproduced 3/100 random and 52/100 fixed.
Candidate scores are 15/100 random and 49/100 fixed retention: paired random
gain **+12pp (fails ≥+20pp)**; fixed delta **−3pp (passes ≤−5pp-loss bound)**.
Thus the final gate is **not passed**, Ticket 06 has a negative outcome and Plug
remains blocked. Correct-vs-fixed-standard is +12pp and vs shuffled +8pp,
diagnostic only. The prior candidate's +18pp/+6pp result is immutable and not
pooled. The retest is expressly nonblind due to historical random-test exposure.

All six cohorts ran once: 600 episodes, 160,418 controls, 6,884.48 s total;
all commands exited 0, with no cleanup or retry. Scene-0 video was captured
within each cohort's native pass. The old formal experiment remains unchanged;
new videos do not backfill its missing-video record. Registration, exact
identities, ledger, per-episode rows, video hashes, yaw bins and summaries are
in the experiment directory. No training, data collection, tuning, further
checkpoint selection or Plug work occurred.

## Peg orientation round 36 — random-200 corrective run

The authorized fresh 30,000-update correction is registered and fully reported
under `outputs/exp/20261117_211300_peg_orientation_random200_correction/`; six
EMA checkpoints are under the matching `outputs/ckpt/` path. Training uses the
unchanged Base EMA plus a fresh zero-control branch, random-200 + unchanged
fixed-1,000 data, seed 1, and exact 512/512 windows per batch. All six checkpoints
were evaluated once on random/fixed validation-50 (12 cohorts, 600 episodes,
131,670 controls); no final-test or retention cohort was run.

| Update | Random validation | Fixed validation |
|---:|---:|---:|
| 5k | 19/50 | 31/50 |
| 10k | 21/50 | 33/50 |
| 15k | 17/50 | 37/50 |
| 20k | 16/50 | 35/50 |
| 25k | 14/50 | 33/50 |
| 30k | **20/50** | **34/50** |

Frozen fixed-feasibility rule (≥34/50; maximize random, ties latest) selects
update 30,000 EMA, SHA-256
`6fa85080d06025ba9283e062e411acfc7fa7e8cc8e936f970180400b8677f1e7`. Paired
against reused Base validation 4/50 random and 36/50 fixed, the result is
random +16/50 and fixed −2/50; against the old mixed random-100/fixed-1,000
candidate it is +5/50 and −4/50 on aligned validation scenes. These are
validation results only. The historical formal gate stays unchanged at +18pp
random (below +20pp) and +6pp fixed; no formal outcome for the new candidate is
claimed. No Plug work. See `PEG_ORIENTATION.md` for budgets and limitations,
and the experiment's `results.json`, `execution_ledger.json`, and
`candidate_freeze.json` for complete provenance and selection evidence.

## Peg orientation round 34 validation motion probe (diagnosis only)

The already-authorized validation-only probe completed once on existing random-validation indices 6–11 with the frozen update-15,000 EMA and correct hint: 2/6 native successes, 1,472 controls, all six grasped. All 187 actual sampler-call hints match live yaw geometry (maximum absolute error `1.97e-7`); all six native controller frames/bounds match the inspected root-translation delta contract. Per-control normalized action, physical root/world/hole request, realized TCP/head motion, video, and source/input audits are in `outputs/exp/20261117_201647_peg_orientation_validation_motion_probe/`.

The data do not support a controller-frame or hint implementation patch: action requests are often small, and late behavior varies between small/backtracking requests and positive requests that realize little head/TCP progress under contact. The existing trace label `commanded_translation_in_hole` omits positional scaling; the offline analysis recomputes physical commands independently and does not treat this measurement-field caveat as a runtime mapping bug. The validation video is not the missing formal preregistered scene-0 video. Formal random +18pp remains below +20pp; fixed retention +6pp remains within its bound. Ticket 06 is still incomplete and Plug blocked. A model/data corrective slice requires an explicit parent decision and training/checkpoint plan.

## Peg orientation random-200 corrective data

`demos/PegInsertionSide-orientation-random-200/` retains the original 100
trajectories unchanged and adds the first 100 official replay acceptances from
`demos/PegInsertionSide-orientation-random-additional100_20261117_204638/`.
Audits and collection registration are under
`outputs/exp/20261117_204638_peg_orientation_random200_data/`.
184 attempts yielded 120 expert successes and 113 replay-valid trajectories;
the combined HDF5 SHA-256 is
`1e662966292e20549039fca8a65ad584a53b28be758e089a37ac2cc3cb8f1c96`.
The saved registration and execution both use 2050/750-second outer caps;
2000/700 were parent task limits, not a different retained registration.
See `timeout_provenance_clarification.json`. Data admission passed review;
training is a separate stage and the formal +18pp result is unchanged.

## Peg orientation Ticket 06 formal paired gate (2026-11-17; not passed)

Round 32 completed all six preregistered native cohorts (600 episodes, 157,747
controls) under `outputs/exp/20261117_153824_peg_orientation_formal_gate/`.
Frozen Base / correct-hint Constrain results were **3/100 / 21/100** on random
test (+18pp, below the required +20pp), and **52/100 / 58/100** on fixed
retention (+6pp, within the maximum −5pp loss). Fixed-standard and shuffled
random-test hints scored 7/100 and 5/100, diagnostic only. Ticket 06 is
**awaiting targeted diagnosis/correction before final rejection**, not completed;
Plug remains blocked. The accepted Base prerequisite is 36/50 fixed validation
(not 72/100); the old Base ≥80/100 threshold remains waived only as a blocker.
Full registration, results, paired tables/bins, source and episode audit, command
ledger, regression evidence and limitations are in `preregistration.json`,
`results.json`, `post_execution_audit.json`, `execution_ledger.json`, and
`results.md`. No video was captured despite its preregistered plan; first-two
foreground cohorts lack contemporaneous per-cohort timing/source ledgers. Both
limitations are disclosed. No outcome-based rerun or retuning followed.

### Round 33 — retained-trace diagnosis (offline; no correction established)

The reproducible runner is `examples/baselines/constrain/peg_orientation_failure_analysis.py`; its TDD fixture is `tests/test_peg_orientation_failure_analysis.py`. Frozen analysis plan, complete per-episode derived measurements/raw episode SHA-256 identities, and concise report are in `outputs/exp/20261117_164500_peg_orientation_failure_diagnosis/`. It reads all 600 retained round-32 rows, includes every preaction and postaction official peg-head-in-hole sample plus terminal exactly once, and verifies source row hashes unchanged. The source-defined geometry is `(box_hole_pose.inv() * peg_head_pose).p`; official insertion is x ≥ −0.015 m with y/z each within the actual hole radius. Reported 3D/lateral distances and wrapped peg local +X minus box local +X yaw are descriptive only; no yaw threshold or XY≤0.10 m phase is introduced.

Random-correct failures (79) have median closest head distance 0.1044 m, head x −0.1044 m, lateral yz 0.0092 m, and absolute wrapped yaw 0.0601 rad at that closest sample. Correct-hint successes (21) have corresponding medians 0.0102 m, −0.0097 m, 0.0030 m, and 0.0187 rad. The recorded native insertion boolean appears in 21/21 successes and 0/79 failures. All random profiles grasped in all 100 rows; contact-loss transitions were 61 Base / 31 correct / 48 fixed-standard / 38 shuffled. This suggests approach/head geometry as the main next probe, but is exploratory outcome association, not causal attribution. Fixed retention shows 36 Base vs 10 correct contact-loss transitions, consistent with no measured fixed-skill regression in this trace summary, not proof of mechanism.

Hint trace coverage is only first-two plus final sampler call, not every hint/action. Requested action vectors, TCP pose and force are not present in these retained rows, so translation command-vs-motion and contact-blocking cannot be quantified here. No implementation bug or correction was established. Next smallest validation-only probe, if separately authorized: existing random-validation catalog indices **6–11**, one frozen correct-hint checkpoint, seeds `1+i`, max 300 controls/episode, zero optimizer updates, and per-control requested 7D action, TCP/peg/hole pose, official head geometry, and live hint capture. The falsifiable candidate is a late-approach frame-mapping error: it predicts persistent grasp plus a consistent requested-to-realized head-axial direction/scale mismatch; absence falsifies a mapping correction. If requested motion itself is small, do not change the controller mapping; a later training-budget decision would require authorization. Indices 0–5 are excluded from this next probe because the earlier six-scene stage diagnosis already used them. No native action was run in round 33. Formal +18pp random result still misses +20pp; fixed +6pp remains within its loss bound; Ticket 06 and Plug status do not change.

The continuation instruction already authorizes the bounded validation-only
6–11 probe above (six episodes, ≤1,800 controls, zero updates); no additional
approval is needed. This supersedes its separate-authorization wording and the
original analysis plan's test-index 0–5 proposal. Training is outside this probe.

## Peg orientation gate preparation and candidate freeze (2026-11-17; no native evaluation)

Round 30 records the user-authorized Base prerequisite amendment and a read-only audit/freeze at `outputs/exp/20261117_151022_peg_orientation_gate_preparation/`. The accepted Base is the existing fixed-validation **36/50 (72%)**, not an independent 72/100 measurement. The old Base ≥80/100 requirement is waived only as a blocker; the final paired fixed-retention ≤5pp and random-test ≥20pp gates remain unchanged.

The selected validation-only candidate is mixed-replay update 15,000 EMA, SHA-256 `522be233a68123e2a4afae882d84cf5e87cab26844054059585d74ffac820898`: random validation 15/50 vs Base 4/50 (+22pp), fixed validation 38/50 vs Base 36/50 (+4pp). Training combined random-100 with fixed-1000 at 512 windows each/update for 30,000 updates; do not describe it as pure random-100 training or same-information fine-tuning. No final scene content/outcome was loaded and no native command was run. The final random test and fixed-retention tests remain pending; historical fixed-catalog exposure from an earlier Base is disclosed in the freeze record. Round 31 reconciles Tickets 04/05 as complete under revised bounded criteria; Ticket 05 overfit assessment is unknown, not proven absent. Ticket 06 is ready for formal preregistration, not completed; Plug remains blocked.

### Round 31 status reconciliation (docs-only)

No artifact identity, quote, historical outcome, or experiment record was rewritten. Ticket 04 acceptance is supported by retained paired validation scores/five angle-bin summaries, correct/fixed/shuffled hint intervention, and six-scene grasp/insertion traces. A numeric yaw sample is not required: frozen `not_reached` remains a valid proximity-trigger-only value and must not be read as no physical insertion. Ticket 05 diagnosis is bounded by angle coverage, hint interventions, frozen-Base/sampler equivalence, action/replay contract audits, and stage traces; overfit assessment is explicitly unknown because single-seed validation selection cannot prove absence, and is not treated as a required acceptance gate. Ticket 02 accepted value is 36/50 validation only, not 72/100. The unchanged formal gates are random-test gain ≥20pp and paired fixed-retention loss ≤5pp. Ticket 06 is preregistration-ready but still unrun; Plug is blocked. Additive machine-readable status: `outputs/exp/20261117_151022_peg_orientation_gate_preparation/readiness_round31.json`.

## Peg evaluator stage-geometry trace (2026-11-17)

The evaluator now supports opt-in pre/post native geometry snapshots through
`--record-stage-geometry`; without the flag the normal Constrain evaluation does
not add phase/geometry trace output. Stage tracing implies phase tracing so each
executed control receives both snapshots. The focused fixture verifies mutation
across controls and wrapped yaw across ±π; the required seven-file regression
passed 58 tests. The fresh six-scene stage diagnosis is retained at
`outputs/exp/20261117_153000_peg_orientation_stage_diagnosis/`: Base 0/6 and
selected Constrain 3/6, descriptive only. No episode reached the frozen XY
center-distance ≤0.10 m trigger; three Constrain episodes nevertheless achieved
official insertion, illustrating that the diagnostic label is not a physical
non-insertion claim. No yaw causality or general performance claim follows. See
`PEG_ORIENTATION.md` and the experiment `results.md` for details.

## Peg dynamic-yaw hint interventions (2026-11-17; exploratory only)

The selected replay-trained 15,000-update EMA was frozen and evaluated once on
each of three paired 50-scene **random validation** cohorts. The run is
`outputs/exp/20261117_115044_peg_orientation_hint_interventions/`; it records a
pre-registration, validation-only shuffled-target assignment, 150 episode
rows, identity checks, and full paired/bin summaries. Correct dynamic hint
reproduced **15/50**; live fixed-standard-90° scored **2/50**; live shuffled
validation targets scored **4/50**. Correct vs fixed paired outcomes were 2
both / 13 correct-only / 0 fixed-only / 35 neither (+26 pp); correct vs
shuffled were 2 / 13 / 2 / 33 (+22 pp). The assignment used seed 1703, with no
self-matches or identical target-yaw matches. Correct also reproduced its
prior Base comparison (3 both / 12 correct-only / 1 Base-only / 34 neither,
+22 pp).

The third pair table is retained in `fixed_shuffled_pair_addendum.json`: fixed
vs shuffled has 0 both / 2 fixed-only / 4 shuffled-only / 44 neither (−4 pp).
It was aggregated from the existing 50 aligned rows per mode after review;
no scene was rerun and original result files remain unchanged.

This is validation-only, single-checkpoint exploratory evidence; it does not
establish general causality or method-level stability. All 150 frozen
pre-insertion diagnostics were `not_reached`, so no yaw-performance or physical
insertion-phase claim is made. No final random-test/fixed-retention scenes,
training, data collection, or checkpoint selection occurred. Base and Ticket
04 remain unqualified/incomplete; Tickets 05/06 and Plug remain blocked. See
the run's `results.md` and `.scratch/orientation-conditioned-insertion/issues/04-peg-constrain-pilot.md`.

## Peg orientation balanced fixed/random replay continuation (2026-11-17; exploratory only)

The user authorized autonomous bounded exploration without changing the formal
qualification contract. Run artifacts are under
`outputs/exp/20261117_003640_peg_orientation_replay_exploratory/`; trained raw
weights and metadata-complete evaluation copies are under
`outputs/ckpt/20261117_003640_peg_orientation_replay_exploratory/`. The complete
pre-registration, per-cohort audit, paired transitions, checkpoint hashes,
commands, runtime versions and limitations are in that experiment's
`preregistration.json`, `results.json`, `validation_commands.json`,
`checkpoint_metadata_enrichment.json` and `results.md`.

One fresh-Base/zero-control-branch run completed exactly 30,000 updates, seed 1,
batch 1,024, with 512 windows/update from unchanged random-100 data and 512 from
the already-retained fixed-1000 data. Base weights were bit-exact through
training. This is not a pure replay-mixture comparison: fixed training data was
added and random windows/update were halved relative to the earlier pilot.

| Profile | Random validation / 50 | Fixed validation / 50 |
|---|---:|---:|
| Fresh selected Base | 4 | 36 |
| Constrain update 5,000 | 13 | 31 |
| Constrain update 10,000 | 10 | 32 |
| Constrain update 15,000 (frozen-rule selection) | **15** | **38** |
| Constrain update 20,000 | 12 | 39 |
| Constrain update 25,000 | 12 | 35 |
| Constrain update 30,000 | 10 | 34 |

The preregistered fixed feasibility boundary was at least Base fixed count −2;
feasible updates were 15k, 20k, 25k and 30k. Among them, update 15k maximized
random success. Its paired transitions were random Base-only 1 / Constrain-only
12 / both 3 / neither 34 (**+11/50, +22 pp**), and fixed Base-only 3 /
Constrain-only 5 / both 33 / neither 9 (**+2/50, +4 pp**). Thus the declared
exploratory target (random gain ≥10/50 plus fixed feasibility) was met on these
validation cohorts. This does not qualify the Base, formally complete Ticket
04, complete Tickets 05/06, or unlock PlugCharger. The original Base remains
unqualified; no random-test-100 or fixed-retention-100 episodes were run.

All 14 Base/candidate cohorts (700 episodes total) exited successfully under
the preregistered 300-control protocol. The diagnostic was `not_reached` in all
episodes; this means its proximity trigger did not fire, not that physical
insertion was never reached. Package versions were recorded before evaluation;
direct runtime source hashes and all training/catalog inputs match before and
after each cohort. Package versions were not captured in the pre-optimization
registration, a documented provenance limitation. See `results.md` for
preflight/evaluation failures and bounded cleanup; no validation cohort was
repeated.

## Peg orientation ConstrainNet exploratory pilot (2026-11-16; unqualified Base waiver)

The user explicitly authorized this **exploratory-only** run despite the current
Base being unqualified. This narrow waiver does not qualify Ticket 02, measure
the independent 80/100 fixed-retention criterion, complete Ticket 04, unlock
Tickets 05/06, or authorize PlugCharger. The Base remains 36/50 on fixed
validation; its formal retention test was not run. The original thresholds remain
unchanged.

The frozen preregistration and input identities are in
`outputs/exp/20261116_164019_peg_orientation_constrain_exploratory/preregistration.json`.
The run starts from the current selected Base EMA
`outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt`
(SHA-256 `067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea`)
and the unchanged random-100 dataset (HDF5 SHA-256
`c4d6d2c9a8f85ebb39bcb003a987f7bd29baed24ff7a680d7eaf3726b5786f25`). Training
used seed 1 and exactly 30,000 updates, saving six checkpoints every 5,000;
the preflight's two optimizer updates were discarded and training restarted
from scratch. It uses Peg's full 39d hidden-target state and native 7d actions,
not the generic StackCube 48d/4d adapter. No random test or fixed-retention test
scenes were used.

Base validation under the consistent 300-control protocol was random **4/50**
and fixed **36/50**. Checkpoint selection used random validation only and chose
update 25,000 (**16/50**, SHA-256 recorded in `results.json`); random paired
outcomes were Base-only 0, Constrain-only 12, both 4, neither 34 (**+24 pp**).
The selected checkpoint scored **21/50** fixed validation versus Base 36/50;
paired fixed outcomes were Base-only 18, Constrain-only 3, both 18, neither 11
(**−30 pp**). Thus this exploratory sample shows a random-validation gain and a
substantial fixed-validation loss; it does not pass the original retention
criterion or support formal completion. The yaw diagnostic was unavailable
(`not_reached` for all episodes); this does not establish that physical
insertion was never reached, and no sampled yaw-error claim is available. Five preregistered
equal-width random target-yaw bins and all per-episode records are in
`results.json` and the validation directories. The full command/timeout/failure
record and pair metrics are in that file; the test suite passed 53 tests.

Recovery addendum: `recovery_verification_report.json` reconciles all nine
retained 50-episode cohorts, seeds, caps, native counts, summaries, and paired
outcomes without modifying episode records, preregistration, or `results.json`.
It separates hashes verifiable now from missing historical checkpoint/source/
catalog identities. The 5,000-update cohort's checkpoint association remains
unverified and its selection contribution provisional; pre-execution source
hashes were not recorded for any cohort. No historical hashes are backfilled or
represented as contemporaneous evidence. Forward episode headers capture
evaluator/checkpoint/catalog identities at evaluation time.

## Official PegInsertionSide-v1 Diffusion Policy baseline (2026-11-12)

This is the unmodified official state baseline, not the fixed-yaw 39d custom
base and not a qualification of Ticket 02. The run is
`outputs/exp/20261112_181319_peg_official_dp/`; weights are physically in
`outputs/ckpt/20261112_181319_peg_official_dp/` through the pre-created
`checkpoints` symlink. Official selection saved `best_eval_success_once.pt`
at TensorBoard step 85000, which is after 85,001 optimizer updates:
**36/100** `success_once` (SHA-256
`1d25af0f585ee696162eff049ed95e43bed99493acecced61bf06fbb6f0b2e6c`).
`best_eval_success_at_end.pt` from that same eval is **34/100** (SHA-256
`44097e9bad6d9310154f9219ce5c8e7a57600ac5dcdacb2998f60f26fc1f6561`).
The closing eval after exactly 100,000 updates scored 15/100 and 12/100 on the
then-current weights; it is not a reload of the saved best checkpoint. No
independent re-evaluation was added. The custom 80/100 fixed-retention gate
was not used. `review_corrections.md` in the experiment directory corrects the
original audit's evaluation-seed explanation: 0..9 seed spaces, not environment
resets. Details are in
`.scratch/orientation-conditioned-insertion/issues/00-official-peg-baseline.md`.

## Fixed-yaw Base with official Peg training config (2026-11-16)

This resumes Ticket 02. It is not the official 43d baseline and does not qualify
the base. The historical 30,000-update run remains the first budget. The new
from-scratch run is `outputs/exp/20261116_123901_peg_orientation_base_official_config/`;
weights are in `outputs/ckpt/20261116_123901_peg_orientation_base_official_config/`.
Training used the resolved official Peg optimizer, model, and 100,000-update
budget, with the documented method exceptions: fixed 90° yaw, 39d hidden-goal
observations, the existing 1,000 demos, frozen 50-scene validation, and a 300/300
native cap. It finished in 2,076.12 s with exactly 100,000 optimizer updates.
All 20 checkpoints were validated. The latest-tie selection is update 100,000
at **36/50**, EMA SHA-256
`067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea`.
That is below 40/50, so retention was not run and this is not a formal 80/100
measurement. Ticket 02 stays paused and Ticket 04 blocked. The 13/50 result at
500 controls is not directly comparable. A round-21 accounting-only evaluator
fix now records completed actions per chunk. The historical
`executed_chunk_lengths` entries in 483/1,000 validation records were planned
lengths, so their sums overstate completed controls for early-success chunks;
the retained `executed_controls` values, scores, selected checkpoint, and hashes
are unchanged. No evaluation was rerun and no historical record was rewritten.
See `round21_accounting_review.md` in the experiment directory and Ticket 02
Execution.

## Peg orientation-conditioned insertion BaseDP (2026-11-11)

Ticket 02's data prerequisite now has exactly 1,000 officially replay-verified
fixed-yaw rows at `demos/PegInsertionSide-orientation-fixed-1000/`: the original
50 remain unchanged, with 950 new rows. Combined trajectory HDF5 SHA-256 is
`54b4a0dd7a62ef9cdd61ca5d5545051e3c769a3a179f0f7f965fddc8fe453631`;
manifest and full yaw/geometry coverage are in the dataset directory. Six
additional/recovery source packages retain expert attempts and native replay
acceptance/rejection audits. Batch logs and the additive fixed-seed reservation
supplement are in
`outputs/exp/20261111_174907_peg_orientation_base_data/`. The original frozen
split and scene catalogs were not rewritten. BaseDP has now completed its
preregistered 30,000-update from-scratch run but failed the qualification gate.
The original all-six 0/50 comparison and update-30,000 selection used an invalid
100-step cap and are retained as historical outcomes only. The corrected
validation-only checkpoint comparison is recorded below. The selected base is
not qualified for downstream use; Ticket 02 is paused and Ticket 04 blocked.

The timestamped run is `outputs/exp/20261111_184900_peg_orientation_base/`;
weights are in `outputs/ckpt/20261111_184900_peg_orientation_base/`. The
pre-registration, six validation results, selected checkpoint identity, full
fixed-retention episode records, failure diagnostics, and test-exposure note are
in `preregistration.json` and `selection_and_gate.json`. Selected update-30,000
EMA SHA-256 is
`db6184ca4a467cc4a2163c82cc27e5f64e7dd17148757d64ec727539751597f4`. The
100-episode rollout video is
`outputs/exp/20261111_184900_peg_orientation_base/fixed_retention_selected_30000_run2/rollouts.mp4`
(SHA-256 `6a64fe44da3b4a2183b83b1ac89bafb36c17c82cb8517ddd95e9dabbbcd09189`).
See `.scratch/orientation-conditioned-insertion/issues/02-peg-base.md` for
commands, the 0/100 outcome, the representative grasp/transport diagnostic,
regression evidence, and the disclosed partial first test attempt.

Round 8 corrected the evaluator's pre-insertion diagnostic capture: the prior
`preinsertion_not_reached` labels were sticky first-sample metadata, not verified
native failure types. The original native outcomes remain 0/50 validation and
0/100 retention, unchanged. A zero-update checkpoint probe (128 training
windows; four frozen validation initial states) is recorded at
`outputs/exp/20261111_203000_peg_orientation_base_diagnosis/`; it found close
EMA/online denoising losses. The initial positive gripper output was matched
against phase-equivalent expert starts and is correct for the open phase.
Follow-up found sampler-dependent gripper timing at the first close boundary,
but did not establish that as the 0% root cause: four policy chunks around one
successful training grasp all grasped within two controls of expert. A successful
native expert replay validated the saved-state observation/action chain. Root
cause remains unestablished; `followup_diagnosis.json` records measurements,
exact commands, and its noise-only next-test proposal (superseded below). No
optimizer updates were used.

The subsequent training-only horizon audit found that 999/1000 expert trajectories
exceed 100 actions and 973/1000 native successes occur after action100. A
training-derived 250-action horizon was preregistered before one selected-checkpoint
validation rerun. That run scored **9/50** at pinned seed1907; it is below the
independent 80/100 gate and cannot be compared causally to old 0/50 because the
old sampling seed was not registered. It confirms the 100-step cap was a protocol
blocker, but the remaining 41/50 failures leave Base unqualified. Full audit,
preregistration, results, and command are in
`outputs/exp/20261111_221500_peg_orientation_base_horizon/`. Fixed retention
remains 0/100 and was not rerun; no optimizer updates or test scenes were used.

The authorized round-9 controlled validation used per-scene sampling seeds
`1907 + scene_index` and a fixed 1,000-control environment TimeLimit, with
evaluator caps 250/500/1000. The selected checkpoint scored **12/50, 13/50,
13/50**; first successes after control 250 consisted of one scene at control 367,
with no additional success through 1000. The run therefore plateaus at 13/50 on
this validation cohort and remains unqualified; fixed-retention scenes were not
accessed. Preregistration, per-scene records, result hashes, logs, and exact
commands are in
`outputs/exp/20261111_210357_peg_orientation_extended_horizon/`. The evaluator
records actual control counts and first native success control. Required
regressions passed **32 tests** before the runs. Ticket 02 remains paused and
Ticket 04 blocked.

Round 10 froze five ranked hypotheses and a deterministic 12-episode training
sample before probing. Pinned teacher-forced preinsertion predictions show
8-action rotation-Z MAE 0.0427 versus first-action MAE 0.0162, which is not
physical closed-loop evidence for chunk drift. A follow-up used separate
environments with expert-prefix replay to establish actual native contact before
each branch. Eight matched post-grasp branches across two training episodes all
retained contact for eight controls; policy-2 replanning improved XY progress in
one paired comparison and regressed in two. Thus the preregistered chunk-drift
criterion was not met, and no validation horizon comparison or inference
correction was made. The 500-control cap was reached; an additional prefix ended
midway only because the budget was exhausted, not because of a task failure. The
then-authorized continuation compared full policy rollouts from six exact training
initial states, with 500 controls per arm. Execute-8 achieved **5/6** native
successes versus **1/6** for replan-2; both grasped and reached the XY proximity
trigger in 6/6, while contact-loss transitions were **1 vs 18**. This is evidence
that the combined frequent-replanning profile harms post-grasp stability on this
sample, not proof that it explains the validation plateau; RNG draw frequency
and observation replanning remain confounded. The preregistered improvement gate
for validation was not met, so no validation run or inference correction was
made. Full traces, hashes, and the remaining causal question (noise resampling
versus observation cadence) are in
`outputs/exp/20261111_214832_peg_orientation_base_diagnosis/` and Ticket 02
Execution. The required suite passed **33 tests**; Ticket 02 remains
paused and Ticket 04 blocked.

### Round 11 — corrected fixed-validation checkpoint re-selection

The original six-way checkpoint selection was invalidated because all six
checkpoints tied at 0/50 under the incorrect 100-step cap; the training audit
showed 973/1,000 native expert successes after step100. Before new rollout, the
reselection preregistration froze all six checkpoint hashes, data/catalog and
evaluator identities, seed rule, native success metric, and selection rule. The
common budget is 500 controls, justified by validation-only update30000 results
of 13/50 at both 500 and 1000, with no success after control 500. All six were
compared on the same fixed 50 validation scenes with per-scene seed `1907 + i`
and unchanged 8-action inference; the existing update30000/500 result was reused
after exact provenance verification. Scores were **0/50, 5/50, 6/50, 10/50,
10/50, 13/50** at updates 5k through 30k. Corrected selection remains update
30,000 EMA SHA-256
`db6184ca4a467cc4a2163c82cc27e5f64e7dd17148757d64ec727539751597f4`.

This validation-informed score is below the independent 80/100 qualification
gate and below the 40/50 validation marker for every checkpoint. No retention or
test scenes, new training, data collection, or checkpoint mutation occurred.
Ticket 02 remains paused-below-qualification and Ticket 04 blocked. Preregistration,
per-scene rows, logs, all result hashes and the full selection decision are in
`outputs/exp/20261111_223000_peg_orientation_base_reselection/selection_and_results.json`
and the neighboring run artifacts. The five new evaluation commands each
returned zero under a 3,600-second timeout; no cleanup was required.

### Round 12 — fixed-validation stage diagnosis

Before rollout, five hypotheses and cohort indices 0..5 were frozen for the selected
update-30000 checkpoint at 500 controls, 8-action inference, and seed `1907+index`.
Index 0 reproduced the known official failure (expected diagnostic exit 3), including
the historical terminal distance. A zero-control restore then showed the diagnostic
hole slice started at index 35, which is the hole radius; the hole center is at
36:39. That label was corrected before indices 1..5. Scene 0 was not rerun.

The six scenes have **0/6** official successes. All six acquire and hold native
grasp, move the peg closer after grasp, and remain at least 0.142 m from the hole.
Five end farther than their closest approach while still grasped. Replan-2 was not
used. This is not qualification and not a controlled comparison with the training
execute-8 successes. Evidence is in
`outputs/exp/20261112_010814_peg_orientation_validation_phase/`. Ticket 02 remains
paused and Ticket 04 blocked. The required suite passed **36 tests**.

### Round 13 — scene 0 insertion-window trace

Before rollout, hypotheses and a 200-control budget were frozen for scene 0 only.
The replay matched the round-12 pose at control 191 exactly (`0.1419355650506847` m)
and did not need the remaining 300 controls. The official peg head is about 20 mm
short of the axial insertion threshold and already inside the hole radius in y/z;
peg-center XY is not that residual. All nine following translations are below the
preregistered stall cutoff. No large insertion command was available to test a
contact block. The base remains unqualified. Evidence is in
`outputs/exp/20261112_013447_peg_orientation_scene0_insertion_window/`. The
required suite passed **38 tests**.

### Round 14 — scene 0 hole-+X intervention

One authorized diagnostic arm, not a policy score. After the frozen policy
matched the round-13 scene-0 branch at control 190, controls 191..198 replaced
only translation with a 5 mm hole-+X request. Rotation and gripper stayed the
stored baseline values. Official success did not occur. The request was unclipped,
but TCP motion was 0.2–0.8 mm per control and official head x advanced 2.2 mm,
leaving an 18.1 mm axial gap. This does not establish contact and does not
qualify the base. Evidence is in
`outputs/exp/20261112_103851_peg_orientation_scene0_hole_x_intervention/`.
The required suite passed **39 tests**.

### Round 15 — expert translation scale, read-only

No native control, training, or intervention. Before extraction,
`outputs/exp/20261112_110405_peg_orientation_expert_translation_scale/analysis_plan.json`
froze the [0.04, 0.06] band, all 1,000 episodes, the T/T+1 TCP comparison, and
the quantile rule. On gripper-request-negative rows the expert median realized
TCP is 1.47 mm against a 4–6 mm physical request; the median ratio is 0.294, so
the broad 5 mm-request-to-5 mm-TCP assumption is refuted. The intervention's
largest step, 0.843 mm, is below the expert 10th percentile, 0.927 mm. None of
those 8,168 rows is in the frozen insertion-proximal window, so the comparison
does not identify contact or IK. Details are in
`examples/baselines/constrain/PEG_ORIENTATION.md`.

### Round 16 — proximal expert requests, read-only

No native control, training, or intervention. Before extraction,
`outputs/exp/20261112_112038_peg_orientation_proximal_expert_requests/analysis_plan.json`
froze the round-15 head predicate, pre-action timing, and the median comparison
to the 0.05 intervention request and the round-13 window maximum 0.0373341420915306.
All 1,652 proximal states are actionable; none is a terminal state without an
action, and none is in `[0.04, 0.06]`. Their median stored translation request
is 0.303, so the frozen rule calls the expert requests larger than both anchors.
That is descriptive only and does not identify the 13/50 cause. Details are in
`examples/baselines/constrain/PEG_ORIENTATION.md`.

### Round 17 — proximal policy first actions, offline

No native control, training, or intervention. Before inference,
`outputs/exp/20261112_113926_peg_orientation_proximal_policy_first_actions/analysis_plan.json`
froze the same 1,652-row predicate, official 39d history `t-1, t`, first executed
action from `Agent.get_action`, seed 1907, and fixed batch 64. The batched noise
profile is not the closed-loop scene RNG. The update-30000 EMA median translation
norm is 0.302, above 0.05, so the frozen rule says the low-command profile is
not reproduced on these expert states. Zero rows are at or below 0.05. This does
not identify the 13/50 cause. Details are in
`examples/baselines/constrain/PEG_ORIENTATION.md`.

### Round 18 — scene 0 actual sampler history

One unchanged 200-control replay of validation scene 0. No intervention, training,
retention, or test scene. Before physics,
`outputs/exp/20261112_115958_peg_orientation_scene0_sampler_history/preregistration.json`
froze the update-30000 EMA, seed 1907, execute-8, and the two scheduled sampler
calls covering controls 185..200. The replay matched the round-13 anchors exactly,
including control 191 pose XY `0.1419355650506847` m. Both captured histories match
native qpos, TCP, peg, and hole center at max absolute error 0. Restoring the
saved pre-call RNG and calling the official sampler reproduces both executed
chunks at error 0. The mismatch prediction is falsified. The call before control
193 still requests translation L2 0.0336, while round 17's teacher-forced proximal
median is 0.302. That is not a matched comparison and not an out-of-distribution
proof. No implementation bug was found. The base remains unqualified and Ticket 04
blocked. Details are in `examples/baselines/constrain/PEG_ORIENTATION.md`.

Ticket 01's fixed-yaw and random-yaw datasets are retained separately; each
contains exactly 50 officially replay-verified `pd_ee_delta_pose` episodes.
Collection used the native joint-position expert and ManiSkill's official
joint-to-EE conversion/replay path. The complete paired analysis, attempt
coverage, and runtime evidence are in
`outputs/exp/20261111_164512_peg_orientation_preflight/summary.json`; task
schema, reproduction commands, and findings are in
`examples/baselines/constrain/PEG_ORIENTATION.md`.

| Dataset | Valid episodes | Final trajectory HDF5 SHA-256 | Provenance |
|---|---:|---|---|
| `demos/PegInsertionSide-orientation-fixed/` | 50 | `715f76393b661f70ba8958b9c87ace0f0a23192b216c6d4ebf1c1407115d1606` | `dataset_manifest.json`, `collection_report.json`, `replay_audit.json` |
| `demos/PegInsertionSide-orientation-random/` | 50 | `ab04a2ccd3d640e1aff40f1e2efd8b59888c26023bee3ba5a5851895bc034940` | `dataset_manifest.json`, `collection_report.json`, `replay_audit.json` |

Ticket 03 preserves that random 50-row package byte-for-byte and adds
`demos/PegInsertionSide-orientation-random-100/` (100 unique officially
replay-verified episodes; combined HDF5 SHA-256
`c4d6d2c9a8f85ebb39bcb003a987f7bd29baed24ff7a680d7eaf3726b5786f25`). Its
manifest records the original and new source identities; `training_coverage.json`
records all episode seeds, yaw and geometry coverage. The additional random
collection/recovery sources remain in their own demo directories with complete
attempt and replay journals. Frozen validation/test/retention scenes and the
hint intervention assignment are under
`outputs/exp/20261111_172604_peg_orientation_adaptation/`; see
`examples/baselines/constrain/PEG_ORIENTATION.md` and Ticket 03 Execution for
scene counts, identities, split chronology, and verification.

The bounded source-expert collection yielded 65/100 fixed and 63/100 random
expert successes. Official converted replay then accepted 62/65 and 62/63,
respectively; rejected expert/replay attempts remain recorded and were excluded
from the two final 50-episode HDF5 files. Across 100 paired seeds, maximum
fixed-vs-random hole-center difference was `1.49e-8 m`.

## One best result per experiment and method (2026-11-03)

Each task and method now has one retained result. Non-best ConstrainNet
evaluations, videos, and checkpoints were removed. StackCube has no rendered
experiment video.

| Experiment / method | Result | Video |
|---|---|---|
| PickCube BaseDP | `outputs/exp/PickCube/BaseDP/` | `outputs/video/PickCube_BaseDP/` |
| PickCube ConstrainNet | `outputs/exp/PickCube/ConstrainNet/` (28/30, XY 30/30) | `outputs/video/PickCube_ConstrainNet/` |
| PlaceSphere BaseDP | `outputs/exp/PlaceSphere/BaseDP/` | `outputs/video/PlaceSphere_BaseDP/` |
| PlaceSphere ConstrainNet | `outputs/exp/PlaceSphere/ConstrainNet/` (24/30, XY 30/30) | `outputs/video/PlaceSphere_ConstrainNet/` |
| StackCube BaseDP | `outputs/exp/StackCube/paired_eval/base_dp/` (12/100 stack, 0/100 safe) | `outputs/video/StackCube_BaseDP/` |
| StackCube ConstrainNet | `outputs/exp/StackCube/paired_eval/constrain_dp/` (100/100 stack, 88/100 safe) | `outputs/video/StackCube_ConstrainNet/` |
| Wristcam BaseDP | `outputs/exp/Wristcam/BaseDP/` | `outputs/video/Wristcam_BaseDP/` |
| Wristcam ConstrainNet | `outputs/exp/Wristcam/ConstrainNet/` | `outputs/video/Wristcam_ConstrainNet/` |

As of 2026-11-03, `outputs/exp` contained only these four experiment directories. Old timestamp paths are historical sources; the move map is `outputs/exp/migration.json`. The later Peg preflight is indexed above.

## Retained results and inspection videos (2026-11-03)

Experiment results remain under `outputs/exp`. Only rendered videos are under
`outputs/video`, named `<method>_<experiment>`. Existing videos were moved, not
regenerated. The selected 80% results were replayed from retained filtered
actions; all 60 replays matched recorded native success.

| Kept result | Location |
|---|---|
| PickCube selected XY, 28/30 and 30/30 XY-zero | `outputs/exp/20261102_172918_selected_xy_80_tracefix_r59/pickcube_rows/` |
| PlaceSphere selected XY, 24/30 and 30/30 XY-zero | `outputs/exp/20261102_172918_selected_xy_80_tracefix_r59/placesphere_rows/` |
| Fixed evaluation scenes | `outputs/exp/20261102_133924_initial_clear_contract_r52/` |
| PickCube videos | `outputs/video/ConstrainNet-bufferedXY-event-replan_pickcube-selected-xy-80/` |
| PlaceSphere videos | `outputs/video/ConstrainNet-bufferedXY_placesphere-selected-xy-80/` |

Selected checkpoints and the frozen 1,000/500 training packages remain. Older
completed StackCube, wristcam, and 2026-10-22 paired-evaluation results were
not intermediate trials and were kept; their representative videos moved to
`outputs/video/`. Optimization probes, failed rounds, unused checkpoints, and
superseded demo packages were removed. Historical sections below retain their
original paths as provenance; those paths are not all still on disk. Video move
hashes are in `outputs/video/migration.json`.

## Round 59 — passive progress-guard trace index correction and selected-profile regression verification (2026-11-02)

R59 fixes a trace-annotation off-by-one in the optional PlaceSphere progress-guard
path. The evaluator increments the actual executed task-action counter before
terminal handling (preserving R58's terminal-action accounting), so annotations
must use the count captured before that step. They now use that pre-step count for
the one-based `task_action_index` and zero-based stride-eight chunk/offset. A
paused detour therefore points to the still-pending task action without consuming
it; the first action is `(1, chunk 0, offset 0)`, action eight is `(8, 0, 7)`, and
action nine is `(9, 1, 0)`. This is a passive metadata correction only. The
progress-guard-selected negative R57 cohort remains failed and unchanged; no
method result was relabeled.

The rollout-seam regression in
`tests/test_evaluate_placesphere_paired.py` observes final per-step trace rows for
the first/last stride actions, a paused pending action, its resumption, and a
terminal executed action. It failed before the fix because action one was recorded
as index two, then passed after the fix. Focused evaluator tests: **8 passed**;
required full suite: **654 passed, 30 warnings in 99.74 s** under `maniskill`.

Because the runtime source tree changed, R59 preregistered both full 30-scene
selected-profile regression cohorts before either rollout in
`outputs/exp/20261102_172918_selected_xy_80_tracefix_r59/`. Both registrations
freeze `selected_xy_80_v1`, indices `0..29`, exact R52 scene catalogs, R58-selected
models and task-specific flags, current runtime source snapshots, transitive
sources, and inputs. This is a **regression verification of the already-selected
method**, not a new candidate or optimization trial. The PickCube selected R54
event-replan and PlaceSphere selected R53 stride-eight methods were unchanged.

| Task | Registration identity / file SHA-256 | Rows SHA-256 | Native success | Runtime errors | XY-zero | Controls | Minimum actual sample / chord clearance |
|---|---|---|---:|---:|---:|---:|---:|
| PickCube | `5733ffe8e64ab11d33c065c4eed6cae9527f919dac171058679dbcea31d2d818` / `e3b4742240590423a7c11c7939f24a0ada829dae7eadfa15526f306e40bd9b05` | `bbd1c6b91031f65b5733aaea94027fc91ade4dc7901548a60b03d47aa7f7624a` | **28/30** | 0 | **30/30** | 4,078 | +0.854 / +0.854 mm |
| PlaceSphere | `e3a9aa750e4acad40e2915e64d4e49c61d117d04aa90f8157e6993ae4d1df55b` / `e78614af2f26f8f0489e07334a2b82c7c1017ba8924dd37f8c0963e2ebcf94f7` | `32f2a990661179597aa12e706c9295e5dc2be06b9e9f94e5742b2d2d6b7840f8` | **24/30** | 0 | **30/30** | 4,297 | +0.554 / +0.554 mm |

Post-run registration validation rechecked every source, snapshot, and direct
input hash. Independent geometry/trace auditing checked every actual TCP sample
and chord against each scene's restored cylinder radius + 5 mm, fixed center,
executed control ordering, and restoration identity: zero geometry violations
and zero counter mismatches. Every one of the 60 R59 episodes exactly matches its
R58 action array and native outcome (60/60); denominators and success counts are
unchanged. The two R59 row summaries hash to PickCube
`8301215b8bbd1bcc654b376c43852471af7f8bf782cc816eb790a6b0c207dbd5` and
PlaceSphere `b1744bdf0c878e1c7f2a34e06f44a1007db8603ed45f90015bdc7ea649c856b9`.
The corrected PlaceSphere evaluator source SHA-256 is
`3bec959ef5e50f1b2bd632228bc8311c55b1f115cb833cbe3c0d4a5df2737083`; the
regression test SHA-256 is
`89b9b06619ed5949b34fc50214c9ebca55e9fc770cbef76354d2d677814ec43a`. R58's
registration and evidence remain immutable. No scene, model, data, ManiSkill
source, XY geometry, task clock, physical budget, or sampling behavior changed.

Independent R59 spec review found no blockers: the passive annotation fix and
pause/resume/terminal regression test are consistent, both current-source
selected cohorts satisfy the predeclared 24/30 native plus all-30 XY-zero gate,
and the retained source/input identities validate. The current ≥80% development
goal is closed; no further optimization or formal 100-scene run was started.

## Round 58 — selected XY profile, 80% development gate (2026-11-02)

The user-authorized development gate is **at least 24/30 native successes per
task**, 30/30 initial/sample/chord-clear TCP XY traces, and zero runtime errors.
This supersedes the historical 27/30 development threshold in earlier round
notes; the original formal 90% contract and Tickets 07–15 were not run or
reclassified. No data, model, layout, or additional method scan was created.

The CLI preset `--selected-profile selected_xy_80_v1` pins the R52 corrected
catalogs and selected checkpoint hashes, rejects conflicting rollout switches
and registration/profile mismatches, and records both task-specific methods and
the 24/30 gate before outcomes. PickCube selects R54 event replanning
(`ConstrainNet+bufferedXY+event_replan_v1`); PlaceSphere selects the R53 buffered
stride-eight path (`ConstrainNet+XY constraint`, event replanning off). Both use
the unchanged actual cylinder radius + 5 mm acceptance disk, 3 mm working buffer,
native controller bounds (PickCube ±1, PlaceSphere ±0.35), DDPM-100 / DDIM-100
eta=0 samplers, and their original 500 / 200 physical-control caps. Cylinder
collision shapes remain absent and visible geometry is checked against restored
task geometry. PlaceSphere keeps its historical action stream exactly: all 30
action arrays and native outcomes match R53 scene-by-scene.

### Frozen inputs and results

Run directory: `outputs/exp/20261102_085808_selected_xy_80_r58/`. Both complete
index lists `0..29`, source-byte snapshots, transitive runtime sources, controller,
normalizers, checkpoints, training registrations/data, and scene HDF5/manifests
were registered before either task executed. Registration identities:

| Task | Registration SHA-256 | Scene manifest / HDF5 SHA-256 | Selected checkpoint SHA-256 |
|---|---|---|---|
| PickCube | `0306bf4f49ca739740fcb66f7a1e45d5586dfb341782e475c3813b59cad34afd` | `33520b09c500bb851c0a2f24a692fbdd4dff82f265c0e74513b3be54ad6d83ab` / `a99e8d0183a62dadeb43491f0090b9e5d47bf48c9a6f2c6e8d8ec7b07767ae2f` | `fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769` |
| PlaceSphere | `97a7298991d73113a2ac17f8c9ff28e8953ae5d71a8970c272e5bc253e0c71ef` | `8e24d2c53376496d64ab3e80e3dc7361b583b6cfad2380e11d44a0dcf008aa6e` / `c586bb02ae03bb35e937869180e1beb02153051b4bc076420454e9e3a83236b1` | `6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f` |

| Task | Native success | Runtime errors | XY-zero | Physical controls | Minimum TCP sample / chord clearance |
|---|---:|---:|---:|---:|---:|
| PickCube (selected R54) | **28/30 pass** | 0 | **30/30 pass** | 4,078 | +0.854 / +0.854 mm |
| PlaceSphere (selected R53) | **24/30 pass** | 0 | **30/30 pass** | 4,297 | +0.554 / +0.554 mm |

Independent offline recomputation covered all 60 full traces, initial TCP
samples, every actual TCP control sample/chord through first native success or
failed-task termination, visible render-only geometry, zero collision shapes,
restoration flags, and PlaceSphere task-action counts. It found zero XY violations,
zero counter mismatches, and zero PlaceSphere action-stream/outcome mismatches
against R53. The failed-scene denominator remains 30 per task; no old successes
or smoke rows were pooled. Prior PlaceSphere negatives R55 (14/30), R56 (20/30),
and R57 (19/30), plus R53/R54, remain separately preserved in their historical sections. This passes
the current development acceptance only; it is not the original formal gate or a
generalization claim.

### Reproduction

Run after activating `maniskill`. Registration must finish for **both** tasks
before either rollout. The evaluator bounds each episode to 300 s and the
30-scene task run to 3,600 s; the outer command has a bounded 3,640 s timeout.
The following are the retained invocation paths. To rerun, replace the R58
experiment root in all output/registration arguments with a new timestamped
directory: existing registrations and results are immutable and reject reuse.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
indices=$(seq -s ' ' 0 29)
timeout --signal=TERM --kill-after=20s 120s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --selected-profile selected_xy_80_v1 --indices $indices --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --output outputs/exp/20261102_085808_selected_xy_80_r58/pickcube_rows --registration outputs/exp/20261102_085808_selected_xy_80_r58/pickcube_registration.json --device cuda --register
timeout --signal=TERM --kill-after=20s 120s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --selected-profile selected_xy_80_v1 --indices $indices --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --output outputs/exp/20261102_085808_selected_xy_80_r58/placesphere_rows --registration outputs/exp/20261102_085808_selected_xy_80_r58/placesphere_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3640s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --selected-profile selected_xy_80_v1 --indices $indices --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --output outputs/exp/20261102_085808_selected_xy_80_r58/pickcube_rows --registration outputs/exp/20261102_085808_selected_xy_80_r58/pickcube_registration.json --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3640s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --selected-profile selected_xy_80_v1 --indices $indices --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --output outputs/exp/20261102_085808_selected_xy_80_r58/placesphere_rows --registration outputs/exp/20261102_085808_selected_xy_80_r58/placesphere_registration.json --device cuda
```

Focused evaluator tests: **37 passed**. Required full suite:
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
→ **653 passed, 30 warnings in 95.55 s**. Registrations and snapshots were
revalidated after both runs. No ManiSkill source, data, checkpoint, or scene was
modified. Formal 100-scene evaluation remains unstarted.

Runtime source IDs: selected runner `render_only_xy_eval.py`
`eea3d1449e2ac5be89238a49be4e98f216062d6e996e0a40e9ea15cd1e009f0b`, paired
PlaceSphere evaluator `evaluate_placesphere_paired.py`
`6ea787311b723e2f9f8d6430b349c7220d5ead0fddf0fb3bf1472099c0cdeb34`, and focused
render-only tests `test_render_only_xy_eval.py`
`80471ede122ff36faf6bb23d3d16db5f09407044cdd3b532beb1fa4422d56b69`. Full source
and transitive-input snapshots are in the two registration-specific
`*_registration_sources/` directories.

Parent completion audit: the independent spec reviewer recomputed all 60 raw
episode traces and confirmed both tasks pass the predeclared ≥80% development
gate. The local tracker README and Tickets 05/06 record that this optimization
is complete; the formal data/training/100-scene evaluation chain remains
unexecuted. No additional method trial is needed to chase the superseded 90%
development target. The goal branch is retained pending user confirmation to
merge.

## Round 57 — PlaceSphere bounded task-progress tangent guard (2026-11-02)

R57 evaluates the separately named **ConstrainNet+bufferedXY+bounded_progress_guard_v1** on the unchanged fixed R52 PlaceSphere cohort, with the R30 EMA and matched Base checkpoint. No task demonstrations, model weights, scenes, goal/task-state logic, or ManiSkill source changed. Pick remains the selected R54 result at 28/30 native and 30/30 XY-zero; it was not rerun. The selected Place reference remains R54's 21/30, not the Pick 28/30 result.

When the existing XY filter changes a learned XY request beyond epsilon, the pending learned action stays at the same task-action/chunk index. A deterministic circle detour is chosen from the pending learned XY direction and live circle geometry, and its winding is held through that pause. The detour path radius is 41 mm for the R52 Place obstacle (actual obstacle radius 30 mm, acceptance radius 35 mm, working radius 38 mm); it advances in 5 mm arc targets, transitions to tangent motion within 0.5 mm radial tolerance, and requires at least 1 mm measured angular/tangential progress per arc control. It resumes only when the **original pending learned XY chord** clears the 38 mm working disk. Every pause sets Z and rotation deltas to zero and holds the last executed native gripper command; before the first learned command, the hold is derived from actual Panda finger qpos using the native `[-0.01, 0.04] m` to `[-1,1]` mapping. Pauses consume physical controls against the original 200-control Place limit. The limits are 30 controls per pause and 60 total per episode; exhausted or insufficient progress is recorded as a structured task failure. The R52 actual-radius + 5 mm acceptance disk, 3 mm working buffer, native XY bounds, learned policy, and first-success scoring remain unchanged.

### Registration, attempts, and measured outcomes

The final preregistered attempt is `outputs/exp/20261102_162253_progress_guard_xy_r57_stall_capture_fix/`. Both the diagnostic smoke `[0,1,2,5,12,15]` and full indices `0..29` were registered before rollout. Full registration identity is `a6928d72addf35068798983d31d9147646c7d12b26fc94b355db24820d4afea7`; smoke identity is `9bfb0bcfe3ea8a1f02b0851ceacfe34bed1e9dd8a787805d1d0ebed3e634aed3`. Post-run registration/source-snapshot/input checks passed; runtime sources were not edited after registration. Source and input hashes are retained in `post_execution_audit.json`.

| Evaluation | Native success | Runtime errors | XY-zero | Minimum actual TCP sample / chord gap | Physical controls | Tangent-pause controls |
|---|---:|---:|---:|---:|---:|---:|
| Final smoke (six fixed diagnostic rows) | 1/6 | 0 | 6/6 | +2.829 / +2.829 mm | see per-row files | bounded by 30 |
| Full fixed Place cohort | 19/30 **fail ≥27 gate** | 0 | 30/30 | +2.700 / +2.700 mm | 4,002 total; max 200/row | 386 total; max single pause 30, max episode total 31 |

Smoke phases were task success 1, no grasp 3, and grasp/no lift 2; index 1 recorded a bounded stall. Smoke is diagnostic only and is not pooled into the full denominator.

Four full-cohort rows (indices 1, 3, 10, 29) reached the declared 30-control pause limit with the original learned chord still blocked. Each is retained as a structured `bounded_progress_stall` task failure with actual TCP, pending XY, arc progress, physical index, and pause budget; none violates the XY acceptance disk or extends the controller budget. Full terminal phases were task success 19, no grasp 7, grasp/no lift 3, and release/stability failure 1. Independent scoring found zero actual sample/chord violations. The smallest measured clearance was +2.700 mm against the unchanged actual-radius + 5 mm disk.

Against the retained Place results, R57's 19/30 is two fewer native successes than R54 event replanning (21/30), five fewer than R53 (24/30), one fewer than R56 XY auxiliary feedback (20/30), and five more than R55 common latent (14/30). It does not meet the 27/30 native goal. The unfiltered reference ceiling remains 28/30 but only 18/30 XY-zero, so it does not satisfy the safety gate. Pick remains R54's 28/30 and 30/30 XY-zero.

Two earlier R57 smoke attempts are retained separately and excluded from the final denominator: `outputs/exp/20261102_160501_progress_guard_xy_r57/` exposed filter-only asymptotic stalling and a stall-classification defect; `outputs/exp/20261102_161424_progress_guard_xy_r57_radial_tolerance_fix/` fixed radial/tangent transition but still surfaced the stall as a runtime error. Each attempt has its own frozen registrations and source snapshots. Their outcomes were not pooled or relabeled.

The final full analysis is `full_analysis_r57.json` (SHA-256 `43c40dd2348fbed46a757cef08ef214f31833b744aba458fef4ed62658fe3a34`; rows SHA-256 `ea6a88030f08510c9ea9fbb66408ef7ce8218918f9f3786759508b173f08c199`); smoke rows SHA-256 is `3c7c881ccdefbdd10e93918c2ad69a8b5f15f04fd6de2f6aba701d461da8cb73`. The per-control task indices and execution flags are consistent, but the row-level `task_action_count` summary omits the terminal executed action on 26 rows because its increment follows terminal handling; the per-step trace is authoritative and the mismatch list is retained in `full_analysis_r57.json`. This does not alter the native success denominator or physical-control accounting.

Focused evaluator tests passed 28; the final required suite passed **649 tests, 30 warnings in 94.61 s** under `maniskill`. The final source hashes are recorded in the R57 full registration. No branch change, commit, push, or Pick rerun was made.

Parent round closure: independent spec review confirms this is a retained
negative experiment, not PlaceSphere goal completion. The terminal task-action
summary defect remains an open follow-up; no runtime-source change or further
trial followed the user's request to finish this round and report progress.
The local tracker is present at `.scratch/transport-detour-optimization/` and
was checked by direct directory/file reads. Its README and Tickets 05/06 now
record the current user-authorized render-only / initial-clear evaluation scope,
task-specific results, and the unstarted formal chain. Historical pilot-report
completion markers do not mean the current goal or formal gates passed.

## Round 56 — PlaceSphere scheduled non-XY / auxiliary XY feedback (2026-11-02)

R56 implements and evaluates the separately named combined method
**ConstrainNet+bufferedXY+Place scheduled-nonXY XY-feedback**. It is not an
isolated replanning effect. On a filter intervention, the currently scheduled
8-action chunk continues: the original plan supplies Z, rotation, and gripper for
all remaining controls; fresh-observation plans supply XY only. Auxiliary
inference is enclosed in `torch.random.fork_rng` and seeded by
`(scene_seed * 1000003 + control_index) % (2**63 - 1)`, so scheduled full plans
retain their independent scene-seeded DDIM-100, eta=0 stream and exact 8-control
stride. No expert gripper logic, teacher action, waypoint, or task-state
teleport was introduced. The 3 mm working buffer, 35 mm acceptance radius for
the frozen 30 mm cylinder, and native XY bounds `[-0.35, 0.35]` are unchanged.
Each finished episode is atomically persisted with partial native trace on
episode failures. Traces preserve the original scheduled action, selected
auxiliary XY, executed action, control/chunk provenance, auxiliary condition
history hash, RNG seed formula/seed, and measured TCP.

The evaluator also adds explicit `--xy-constraint off` for the paired **unfiltered
ConstrainNet render-only reference**. It leaves actor actions unchanged and still
records/scorers visible render-only geometry and actual TCP samples/chords. It is
a diagnostic control, not an XY-safe accepted method and not BaseDP. The prior
`control_enabled=False` behavior that still applies the constraint is untouched.

### Fixed-cohort diagnosis and preregistration

Before any R56 rollout, the saved initial TCP and object-start metadata were
checked against the frozen cylinder disk on the exact R52 PlaceSphere catalog.
The straight initial TCP-to-object segment intersects the actual-radius + 5 mm
disk in **14/30** scenes (indices 1, 2, 5, 6, 10, 12, 14, 15, 17, 21, 23, 25,
28, 29). Five of R53's six native failures are among those 14; index 27 is the
one R53 failure whose initial chord does not intersect it. This is diagnostic
context only: no row was resampled, rejected, or excluded.

All three registrations (six-row diagnostic smoke, candidate full 30, and
unfiltered reference full 30) and their source/input snapshots were frozen and
verified before rollout. Trial plan:
`outputs/exp/20261102_152921_xy_feedback_r56/trial_plan.json` (SHA-256
`22c30212f264d49caac3026dff1713dc3c3ea698b68dd416ee43ea835812e5a4`). Candidate
full registration identity is
`ea70d09dae05306607e729066274232db2fb286d64beaa08664faa1cd296007e`; reference
identity is `f2d3312a2bd9eed2566895295f32181f3cd11a47b01c0487028b3f404120c232`.
R52 scene manifest/HDF5 remain
`8e24d2c53376496d64ab3e80e3dc7361b583b6cfad2380e11d44a0dcf008aa6e` /
`c586bb02ae03bb35e937869180e1beb02153051b4bc076420454e9e3a83236b1`. The
unchanged R30 EMA and matched Base hashes are
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f` /
`491970382f3ed8dac7ed88ad1ab3f5cd23fb2d29065291a507ff766c75a6f071`; the
fixed training cohort remains 1,000 direct + 500 obstacle examples, with HDF5 /
manifest hashes `796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`
/ `aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`.

### Complete results

| Evaluation | Native success | Runtime errors | XY-zero | Minimum actual TCP sample / chord gap | Controls |
|---|---:|---:|---:|---:|---:|
| XY-feedback smoke (indices 0,1,2,5,12,15; diagnostic only) | 2/6 | 0 | 6/6 | +1.161 / +1.161 mm | 1,156 |
| XY-feedback full fixed cohort | 20/30 **fail ≥27 gate** | 0 | 30/30 | +0.554 / +0.554 mm | 4,568 |
| Unfiltered ConstrainNet reference, same 30 scenes | 28/30 (diagnostic only) | 0 | 18/30; **12 rows violate XY** | -31.147 / -32.984 mm | 3,731 |

Candidate phases were task success 20, release/stability failure 3, grasp/no lift
4, and no grasp 3. The candidate made 258 XY filter interventions and 232
auxiliary queries; the remaining 26 interventions were on the last control of a
chunk or episode, with no subsequent in-chunk control to update. Independent
actual geometry scoring found zero candidate sample or chord violations. The
reference had 45 violating sample points and 57 violating chords across 12 rows;
these violations are retained and are not masked by the native-success metric.
The reference's 28/30 native result shows the frozen actor can reach the native
gate when unfiltered, while its 12 XY-violating rows show that it is not a valid
safe result. In the paired candidate/reference outcomes, 19 succeeded in both,
one candidate-only (index 29), nine reference-only, and one failed in both.

Against R54, candidate XY feedback had 19 paired successes, one new success,
two regressions, and eight failures on both profiles (20/30 vs R54's 21/30).
Against R53 it had 19 paired successes, one new success, five regressions, and
five persistent failures (20/30 vs 24/30). R55 common-latent remains 14/30 and
was not reused. The selected R54 Pick result remains 28/30 native and 30/30
XY-zero; no Pick rerun or Pick policy path change was made.

Post-run checks validated all three registrations and every source/input
snapshot, exact 30-row ordered denominators, unchanged R52 scene hashes, and
selected checkpoint/data identities. Across all candidate controls, executed
Z/rotation/gripper exactly matched the original scheduled chunk (zero
mismatches), scheduled chunk boundaries followed the fixed 8-control stride
(zero mismatches), all XY requests remained within native bounds, and native
success control accounting matched the trace. All 30 initial scene TCPs satisfy
the frozen initial-clearance metadata. The reference passed exact requested to
executed action passthrough on every control. Full details and paired transitions
are in `full_analysis_r56.json` (SHA-256
`e4f7ebbe7d7b403d2dfe2009048db12eb41f5fe15f04c677e7d7f743c3e878c9`). Candidate
full rows hash to
`82452d14811df2345f9b98db522b9f29ef9493706a1e6a044ab5014aefbe16a3`; reference
full rows hash to
`2c5a84efd3555cb1844339bbe1bb31c8ca0949a95c40de62876e97734ecc206b`.
Registrations, source snapshots, smoke/full rows, and per-row atomic episode
traces are retained in the same R56 directory.

The final focused evaluator tests passed **25 tests**; the required complete
suite passed **641 tests, 30 warnings in 86.67 s** under `maniskill` before any
registration or trial. The three modified runtime/test sources have SHA-256
`38abef79afe910ab05e1f8da8fa194c7633184320448ca7350df43d224baa098`
(`render_only_xy_eval.py`),
`c6104b0bbbfb1ec692e0d759e0da56eb781f8df0250b3664e14456792eb21b92`
(`evaluate_placesphere_paired.py`), and
`616515d0e14b3fe58e4482ad0c3eacf0cc08c782bb22874dc46fcd63d4217f80`
(`test_render_only_xy_eval.py`). No ManiSkill source, training data, model, or
scene was changed. R56 therefore **does not meet the PlaceSphere 27/30 native
gate**, despite passing the XY gate. The smallest useful next step is a fixed-row
control-trace differential on the nine reference-only successes, locating each
first XY intervention and subsequent task-phase divergence; this should guide a
single revised mechanism before any additional candidate trial, rather than a
parameter/latent scan.

No branch change, commit, or push was made.

## Round 55 — PlaceSphere common-latent event replanning (2026-11-02)

R55 first performed a read-only control-trace comparison of all R53/R54
PlaceSphere outcomes (R53 failures, R54 regressions, and selected prior
successes). The complete per-control table is
`outputs/exp/20261102_150121_placesphere_coherence_r55/r53_r54_trace_comparison.json`
(SHA-256 `e896fb646811ad10cc42b2330e26ffd7c46df7fe19f9b03317d2243266091490`).
It includes each requested/filtered gripper channel (identical in every row),
gripper sign flips, first negative request, first native `is_obj_grasped`,
replans requested/consumed and while requesting negative grip, XY interventions
before/after grasp, per-control task phase, and derived TCP-to-cylinder-center
distance. The retained traces do not include per-control object height, so exact
first-lift controls cannot be recovered; they retain only final `ever_lifted` and
per-control grasp flags. Object-to-bin distance was not recorded, and no physical
cause is inferred from that missing context.

On R54 Place regressions 12/15/21/29, first negative requested grip moved from
R53 controls 22/62/64/94 to R54 controls 33/39/27/36; these R54 rows had 14/29/30/44
replans before first grasp, which was never observed. Grip sign flips were 15/10/3/3;
replans during negative requested grip were 0/0/4/9. Thus the replan-triggered full
model calls alter temporal gripper sequences as conditioning changes, in addition
to consuming new Gaussian initial noise. This supports testing the noise/temporal
interaction, but does not establish a physical causal mechanism. Index 27 remains
an independent release/stability failure: zero replans in both R53 and R54.
R54's full Pick+Place cohorts requested 963 replans total (704 PickCube + 259
PlaceSphere); all 963 were consumed at a subsequent policy inference.
Holding replans until grasp was not selected: it cannot react to the R53
grasp-without-lift failures (indices 6/10/14) when those trajectories never
achieve a grasp that would release the hold.

The sole R55 candidate was preregistered as **ConstrainNet+bufferedXY+
event_replan+common_latent_v1**. It is a combined method, not an isolated
replanning ablation. PlaceSphere only: the first scheduled inference captures the
scene-seeded RNG state; subsequent XY-event replans restore that state inside
scoped `torch.random.fork_rng`, while every call uses its newly updated observation
condition. There is one latent per plan, no candidate sampling or seed search;
the seven learned action channels, DDIM 100-step sampler, eta=0, R30 checkpoint,
XY buffer, controller budget, and non-scripted Z/rotation/gripper behavior remain
unchanged. `first_plan_r30_equivalence.json` (SHA-256
`fe45d32d20bdba40e9323f2d2bb9ae403dc4c0fe84eabd4c175fe7b03030bb93`) exercises the
actual R30 PlaceSphere adapter on a fixed R52 observation/RNG: wrapped and
unwrapped first-plan tensors are bitwise identical (max difference 0), and
post-inference CPU/CUDA RNG states match. Each native episode records the exact
scene seed, RNG-state digest, strategy, sampler profile, and plan count.
`per_episode_profile_manifest.json` (SHA-256
`cc5e670a686d15c56109d3b6442a368e8e9f39fc7455aca26fea54a616ceaf2b`) binds all
36 full/smoke rows to their registration identities, source and snapshot hashes,
input hashes, task options, scene seed, and per-episode noise metadata.

### Preregistered outcomes

The R52 PlaceSphere catalog and all 30 indices `0..29` were registered before
evaluation. The separately registered diagnostic smoke order was `2,5,12,15,0,1`
(prior failures, then prior successes); it is not pooled into the full denominator.
Both registrations retain source snapshots and frozen inputs. Full registration
identity is
`49687ffbf30cde57039a606add174a369de264b1cc03b51f1c24d00ac04eafbd`, registration
file SHA-256 `fe4db2e4af13a1ec91e32f239ca450d06be601839531ad4c9cfe28e62e06e8fd`;
full rows SHA-256 is
`df640c390489edcc54f877ef6b70f37234c6dd1e0313d8fc070e91ec6f9cea37`; the runner
source SHA-256 is
`cde0cbfbdc4a06954919df661b8aaddb5c0beedce8d85cd3aab8dca7041497b4`. The smoke
registration identity is
`dc83e1373e897736cdc4a842948e92445165c8c49a7c7f16b693ec440f1f0cd2`.

| Evaluation | Native success | Runtime errors | XY-zero | Minimum actual TCP sample / chord gap | Replans requested / consumed |
|---|---:|---:|---:|---:|---:|
| Smoke (6 diagnostic rows) | 1/6 | 0 | 6/6 | +2.405 mm minimum sample gap | 126 / 126 |
| Full fixed cohort (30 rows) | 14/30 **fail ≥27 gate** | 0 | 30/30 | +0.404 / +0.404 mm | 283 / 283 |

Full analysis is `full_analysis.json` (SHA-256
`b813a001ca0e4c7e72726e1fcdd3e13c9817a4db1f84e39cdc1bb8c88c13a079`). Every
episode had the registered common-latent profile and an exact scene seed; all 283
requested replans were consumed. The full 30-row denominator was retained, all
300-second per-episode / 3,600-second trial guards were active, every environment
closed in `finally`, and no native action was issued beyond PlaceSphere's original
200-control task limit. Independent catalog scoring included the initial TCP and
every measured sample/chord: zero sample/chord violations, with the smallest
positive gap 0.404 mm. R55 therefore preserves the XY gate but **does not meet the
native-success goal**, and common latent is not accepted as a successful fix.

R55 native terminal phases: task success 14, no grasp 6, grasp/no lift 4,
release/stability failure 4, transport/no release 1, and grasped/no transport 1.
Compared with R54's 21/30, common-latent R55 has 14/30. This result does not prove
the cause, but common noise alone did not stabilize the replanned action sequence.
The next smallest evidence-based candidate is Place-only XY-event replanning that
keeps the outstanding model-predicted non-XY actions from the current eight-action
chunk and replaces only XY with the fresh-observation plan. This directly tests
the observed repeated gripper phase changes (e.g. R55 index 12 has 30 sign flips
and no grasp) without scripted expert gripper logic or geometric relaxation. It
must be treated as a separately named method and preregistered before any further
native result; no additional seed/noise scan is justified.

Frozen direct inputs remain the R52 PlaceSphere catalog
`8e24d2c53376496d64ab3e80e3dc7361b583b6cfad2380e11d44a0dcf008aa6e` /
`c586bb02ae03bb35e937869180e1beb02153051b4bc076420454e9e3a83236b1`, R30 EMA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`, matched Base
`491970382f3ed8dac7ed88ad1ab3f5cd23fb2d29065291a507ff766c75a6f071`, and unchanged
training HDF5/manifest
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961` /
`aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9` (the existing
1,000 direct + 500 obstacle training cohorts). R52 layout/preflight, visible
render-only geometry, and strict initial-clearance contract were not changed.
All R55 source snapshots and nine direct inputs validated after full execution.
`post_execution_audit.json` (SHA-256
`53730b697bb08b97cf9313e3a21e5f66922824c6f2248995a136f409540b8533`) records
the post-run source/input validation, exact row sets, R52 strict-clear preflight,
episode action limits, and recorded seeds/profiles.
The R54 Pick result remains 28/30 and XY 30/30; Place-only installation has a
regression test proving the Pick path does not install or consume common-latent
noise, so the retained Pick cohort was not blindly rerun.

Focused tests passed 22; the required suite passed **639 tests, 30 warnings in
94.23 s** after the final code change. The geometry analyzer now requires finite
two-element center vectors and rejects truncated, extended, and non-finite centers.
No ManiSkill source, training data, model, or scene was changed. No branch change,
commit, or push was made.

## Round 54 — event-triggered replanning on corrected-XY cohorts (2026-11-02)

R54 applied one control-loop change to the fixed R52 PickCube and PlaceSphere
cohorts: after each executed action, the buffered XY filter compares its output
with the native-bounds-clipped request using epsilon `1e-6` action units. If XY
changed beyond epsilon, the current eight-action chunk ends after that control;
the next plan is inferred from the updated two-frame observation history. No
control step is skipped, and the original eight-control stride remains when no
filter intervention occurs. The first native-success control still terminates
immediately. Replan requests and their next-plan consumption control indices are
recorded per trace step. The physical/default paired evaluators do not enable the
hook; R54 explicitly enables it via `--event-replan`.

R53 traces provided sufficient evidence without a selected unfiltered reference:
reconstructing the registered stride-8 boundaries and counting changes against
the native-bounds-clipped request at epsilon `1e-6` found the first intervention
at controls 1–3 in nine of ten failures, leaving 5–7 actions in each first
affected chunk. In total, the nine affected failures contain 76 action chunks
with interventions and 523 nominal actions remaining after each chunk's first
intervention. Interventions recurred during later chunks. PickCube failure phases
were `grasp_no_lift` (index 10), `lift_no_transport` (14), and `no_grasp` (15,16);
PlaceSphere failures were `release_or_stability_failed` (2,27), `no_grasp` (5),
and `grasp_no_lift` (6,10,14). PlaceSphere index 27 had zero XY interventions,
so that release/stability failure is a policy failure control, not a stale-filter
case. The first intervention in these failures occurs early in policy execution;
the eventual task phase distinguishes grasp/lift/transport/release outcomes, but
the trace does not support assigning a fabricated object-to-target distance.
This diagnoses a plausible mechanism, not proof that every failure was caused by
it. R54 changed only the rollout replanning behavior and trace metadata; task targets, controller budget
(PickCube 500 / PlaceSphere 200), models, data, scenes, filter geometry, 3 mm
working buffer, and actual-radius + 5 mm acceptance disk remain fixed.

### Preregistered outcomes

Full fixed cohorts were registered before either task run and evaluated
sequentially at indices `0..29`. The separately retained smoke was predeclared as
PickCube `10,14,0,1` and PlaceSphere `2,5,0,1` (two prior failures then two prior
successes); it is diagnostic only and is not pooled into the full denominator.

| Task | Native success (gate ≥27) | Runtime errors | XY-zero (gate 30) | Minimum actual TCP sample / chord gap | Filter replans requested / consumed | Failure phases |
|---|---:|---:|---:|---:|---:|---|
| PickCube | 28/30 **pass** | 0 | 30/30 **pass** | +0.854 / +0.854 mm | 704 / 704 | grasp_no_lift 2 |
| PlaceSphere | 21/30 **fail** | 0 | 30/30 **pass** | +0.097 / +0.097 mm | 259 / 259 | no_grasp 5; grasp_no_lift 3; release_or_stability_failed 1 |

The PickCube minimum is scene 27, control 80, measured TCP XY
`[0.060052, 0.031994] m`; the PlaceSphere minimum is scene 7, control 5,
`[-0.063067, 0.038683] m`. All 60 trace geometries were independently compared
against their explicit R52 catalog center and radius; all actual samples and
chords remained outside the unchanged acceptance disk. Every requested replan
was consumed once at the next policy plan boundary. No selected-episode result
is included in these 30-scene metrics.

Against R53, PickCube indices `10,14,15` changed from failure to success, while
index `11` regressed from success to `grasp_no_lift`; total success rose from
26/30 to 28/30. PlaceSphere index `14` changed from failure to success, while
previous successes `12,15,21,29` regressed to `no_grasp`; total success fell from
24/30 to 21/30. Its R53 failures `2,5,6,10,27` persisted as failures (index 2
phase changed to `grasp_no_lift`). The native gate therefore remains unmet for
PlaceSphere; the R54 result does not establish the round goal.

The PlaceSphere regression traces show why the next step should diagnose policy
phase timing before another intervention rule is considered: indices `12,15,21,29`
all regressed from success to `no_grasp`, with their first replan at controls
1–2, first negative gripper request at controls `33,39,27,36`, and no observed
grasp by timeout. PlaceSphere failures `2` and `6` reached a grasp at controls
22 and 35 but did not lift; failure `10` first requested closure at control 19
and first grasped at 82. Index `27` had no replan and still failed release or
stability. The smallest next diagnostic is a per-control comparison of the
retained R53/R54 gripper requests and native grasp/lift transitions on these
fixed rows, to determine whether replanning delayed closure or whether the
PlaceSphere policy independently fails its grasp phase. Do not pool these
selected rows into a headline metric or change gripper/task targets.

Smoke rows retain PickCube `4/4` native successes and PlaceSphere `2/4`; both
tasks have `4/4` XY-zero rows. All smoke replan requests were consumed. No smoke
outcome is pooled with the full trial.

### Registrations, immutable inputs, and analysis

The artifact directory is
`outputs/exp/20261102_142741_event_replan_xy_r54/`. Full registration identities
are PickCube `70db753a9ba461d359ab9b3f7a22c86964b99dca569f530ab4f689cc640d3136`
and PlaceSphere
`871d1ce60279c8f17c1c824bfcfda6927c59ab6069eac4cf9fac98d681d4466f`. The
registration file SHA-256 values are respectively
`8b7e417e97ac41482689e1a4a90f78cd902d4ad29460ccce8d50720f5f9d2df4` and
`757bd48f698e619d5c3c06aa5bd84fcdbdf21ae158aab8d099b41f9dbd6e9286`. Full row
files hash to PickCube
`2ebe54c359ea0afadaa98d7a5896ac1be58124cb1da4ff3ab79b9ca09c64e37b` and
PlaceSphere `5b8e7952027243dc88e5847b8ecd08955783e3572bffa8f14a1524c147b7a0ea`.
The separate smoke registrations are in the same directory; their identities
are PickCube `e293a37a53db8fc9a3bf9a21cff4998299bce8f64d6478922a5dfade8b505cf0`
and PlaceSphere `fc685ec1b1d0fd28324ff9c341a2e7654ffefc8cfd6fe90d7642c091f1965b3c`.

All source snapshots and direct inputs revalidated after execution. The changed
runtime identities are `render_only_xy_eval.py`
`66b2cb124be221b0a4c30bfd5b8bca6c884bbee65467627c4fff1d6ac250ced9`,
PickCube paired rollout `d2f0def062ba64c28360f3478f676ba9cf2fef776f778d2b2d97caaeb21ece04`,
and PlaceSphere paired rollout
`025eb8e19414c0b84f2658863188484eebd3427bfd829fa88ad944c26288edec`. The
filter remains `cb227bf472fae871db1befecdbd34cc0ae99ad8fad461a879a522b68115c4840`.
Complete per-registration source and input hashes are preserved in each
registration and `offline_analysis_r54.json`; analysis SHA-256 is
`cb1a5313130accda09186401e5bdcfeb4511410765c17e5b95f379ce792299df`. R52
catalog and training-data identities remain those recorded above; no data,
scene, model, or training artifact was modified.

The focused tests passed (15), and the required final suite passed **632 tests,
30 warnings in 87.54 s** under `maniskill`:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line
```

The suite ran before registrations and episodes; no runtime or test source was
edited afterward. No commit, branch change, push, or merge was made.

## Round 53 — preregistered corrected-XY full cohorts (2026-11-02)

Both fixed R52 task cohorts were run sequentially, all indices `0..29`, at the
R52 final runtime/profile without changing evaluator, filter, scenes, models,
controller limits, demos, or training data. The preregistration timestamp and
artifact directory are
`outputs/exp/20261102_135852_corrected_xy_full_r53/`. The prior directory label
`20261102_230000_corrected_xy_full_r53/` is retained as a symlink; migration
details are in `MIGRATION.json`. Registrations and source snapshots preserve
their original paths and bytes. Both registrations, their direct inputs, and
source snapshots validated after execution.

### Fixed inputs and acceptance

The only method is **ConstrainNet+XY constraint**, an engineering result, not a
formal performance or generalization claim. Frozen models are PickCube R44 EMA
`fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769` and
PlaceSphere R30 EMA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`. PickCube
uses the R41 normalizer `d14d56e3645504b81db4a41f32cf6869b805dbf003194b26433cfceb6023f543`;
PlaceSphere uses its raw/identity observation normalizer in the registered actor
source. Training registrations remain
`7969ff268f2780a98ecd5005a8ce9cf5d52cac9d702356ad6773a7556200c425` (PickCube)
and `eb4a824224f649881a3511dc229d751d38c926255820d16c2016839e14f549a5`
(PlaceSphere). Demo HDF5/manifest identities are unchanged from R52: PickCube
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f` /
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`; PlaceSphere
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961` /
`aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`. Dataset
sizes remain 1,000 direct + 500 obstacle episodes per task; no demos were
created.

The unchanged acceptance disk is `tcp_xy_cylinder_radius_plus_5mm_v1`: measured
TCP XY samples and every chord must stay outside the actual restored cylinder
radius + 5 mm, from the initial state through first native success, or through
the full failed-task timeout. The distinct working path profile is
`tcp_xy_working_clearance_3mm_v2` (3 mm); it is not substituted for acceptance.
The native controller remains `pd_ee_delta_pose`, 0.1 m per XY action unit,
PickCube bounds ±1 and PlaceSphere bounds ±0.35; other channels pass through.
Both R52 zero-step preflights passed 30/30 with task-layout bounds, PickCube
goal Z=0.12 m, exact restore/admission identities, visible cylinder agreement,
zero collision shapes, and initial TCP gaps strictly positive against R+5 mm.
Their minima were +4.004 mm (PickCube) and +8.542 mm (PlaceSphere).

### Complete native results

| Task | Fixed denominator / evaluated | Native success (gate ≥27) | Runtime errors | XY-zero (gate 30) | Minimum actual sample / chord gap | Filter interventions / controls | Native XY clipping / controls | Terminal phases |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PickCube | 30 / 30 | 26/30 **fail** | 0 | 30/30 **pass** | +1.059 / +1.059 mm | 972 / 4,917 (19.77%) | 0 / 4,917 | task_success 26; grasp_no_lift 1; lift_no_transport 1; no_grasp 2 |
| PlaceSphere | 30 / 30 | 24/30 **fail** | 0 | 30/30 **pass** | +0.554 / +0.554 mm | 207 / 4,297 (4.82%) | 0 / 4,297 | task_success 24; grasp_no_lift 3; no_grasp 1; release_or_stability_failed 2 |

The fixed denominators include all native-task timeouts: PickCube indices
10/14/15/16 at 500 controls and PlaceSphere indices 2/5/6/10/14/27 at 200
controls. Every successful row includes its first native-success control and
preceding chord in the geometry audit. Rows preserve full requested/filtered
actions, live geometry, measured TCP controls, native results and restoration
identity. Offline analysis checked all 60 rows against their catalog geometry;
the runner also requires live task-state and visible-cylinder center agreement,
visible cylinder, and zero obstacle collision shapes for each environment.
All 60 full trajectories had zero actual-radius+5-mm sample or chord violations.
The historical physical-object-crossing latch remains separate (25 PickCube,
1 PlaceSphere); it is not the TCP XY gate. Body contact was 0 in both cohorts
and is not a gate. Native success retains original task semantics: PickCube
holds the cube at its task goal; PlaceSphere requires the sphere on-bin, static,
and not grasped.

The preregistration identities are PickCube
`a395d7822bbfced991050d785c7e938e4a2a789b1af445621ded299dcc269d86` (18 source
entries, 7 input entries) and PlaceSphere
`0cd1542bcbafa64f25d26952e1c89b7539994382f662ef24a4dc803462141f5f` (14 source
entries, 9 input entries). Their complete byte snapshots are retained under
`pickcube_registration_sources/` and `placesphere_registration_sources/`.
Registration files have SHA-256 `d88e8b491dadef7b840d22a2805e6a71f68667492ce50ac38e50e3e1bc6b67ae`
and `23b9125eaabd795aca624ee053510e7d82d7216c81906a20748ae888b8a722fc`.
Catalog manifest/HDF5 hashes match the R52 identities above. Full row files
hash to PickCube `2e1f29741b724edecce835c1f1005c59b87cc286f34dd822df696a104b3e782d`
and PlaceSphere `59bf3c1ad465bef56f65b72f16116cbf893a3546c43c9e902681d4ecbdc6ebbb`.

### Reproduction and offline audit

Both complete index lists were registered before either policy rollout; tasks
were then run sequentially. The commands below use the final explicit catalogs,
checkpoints and profile/source snapshot set described above:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_135852_corrected_xy_full_r53/pickcube_rows --registration outputs/exp/20261102_135852_corrected_xy_full_r53/pickcube_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_135852_corrected_xy_full_r53/placesphere_rows --registration outputs/exp/20261102_135852_corrected_xy_full_r53/placesphere_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_135852_corrected_xy_full_r53/pickcube_rows --registration outputs/exp/20261102_135852_corrected_xy_full_r53/pickcube_registration.json --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_135852_corrected_xy_full_r53/placesphere_rows --registration outputs/exp/20261102_135852_corrected_xy_full_r53/placesphere_registration.json --device cuda
```

The bounded offline analyzer is
`examples/baselines/constrain/analyze_corrected_xy_full_r53.py` (SHA-256
`76759c0e99b1b68e6f4d0e9441a3abe5e0f9d6ed599fabf5cbb39d72dce1a3b7`); it
validates all row counts/order, registration and source/input/snapshot byte
hashes, task/catalog geometry, native restoration IDs, trace sample/chord
clearance including first success and timeouts, result phase counts and action
filter-vs-bound clipping counts. Its report is
`offline_analysis_r53.json` (SHA-256
`48752bbcad430a42a66dab05352db4bfac35b82ab83dbcebf7d2f0b2f8d047ae`). It
confirmed zero runtime errors and all XY gates, but native success gates failed
for both tasks. This bounded row/provenance check passed; the prior 628-test
R52 suite remains the valid suite result because no runtime source changed.
R48/R49 native-reset OOD cohorts remain untouched and are not pooled here.

## Round 52 — restored task-layout sampling and lag-aware XY diagnostic (2026-11-02)

R52 resolves the independently confirmed R51 task-definition blocker and adds one
evidence-derived working-clearance profile. Changes are confined to
`examples/baselines/constrain`; no ManiSkill source, demos, trained weights, or
model parameters were changed. Existing R48/R49 poor and negative cohorts remain
in place as out-of-distribution evidence; they were not deleted, cherry-picked,
or relabeled as R52. The immutable physical training sets remain 1,000 direct +
500 obstacle episodes per task. PickCube training manifest/HDF5 SHA-256 remain
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486` /
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`; PlaceSphere
remain `aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9` /
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`.

### Fixed task-layout sampler and restored admission

`initial_clear_xy_scenes.py` now draws the task object and goal explicitly from
the original task layouts, rather than accepting arbitrary native-reset goals.
The PickCube contract is start X `[-0.18,-0.02]`, Y `[-0.06,0.12]`, goal X
`[0.02,0.18]`, Y `[-0.06,0.12]`, goal Z exactly `0.12 m`, table bounds
`[-0.25,-0.25,0.25,0.35]`, at least `0.14 m` transport and `0.06 m` base-line
clearance. PlaceSphere uses start X `[-0.19,-0.08]`, Y `[-0.13,0.13]`, container
X `[0.04,0.15]`, Y `[-0.13,0.13]`, at least `0.14 m` transport and `0.06 m`
base-line clearance; the full container footprint must remain within table bounds.
The manifest records `layout_contract_version=transport_detour_training_bounds_v1`
and direct goal XYZ. The obstacle remains radius `0.03 m`, on the direct start-goal
segment at a seeded fraction in `[0.4,0.6]`, with the fixed render heights `0.14 m`
(PickCube) and `0.08 m` (PlaceSphere).

The deterministic task streams use seeds 521100/522100, 1,000-attempt maximum,
600-second sampler wall limit and 300-second reset alarm. They accept 30/30 in
200 PickCube attempts (31 base-line, 24 too-short, 46 initial-inside and 69
duplicate rejections) and 255 PlaceSphere attempts (30 base-line, 1 too-short,
61 initial-inside and 133 duplicate rejections). Dedup checks include the frozen
training manifests, old development catalogs, old R46 registration/rows, the
entire frozen R48 task cohort, and already accepted rows; they reject same seed or
start **and** goal componentwise within 2 cm. All input/source identities and
per-attempt decisions are retained in each task directory. Sampling produced no
policy calls or demo files.

The sampling-time snapshot of `render_only_xy_eval.py` is SHA-256
`bc0eee0c292ac9efaf9ba64f168e319ba221de945f6a523e8a489359439d888f`; before
final policy registration it received only the registration metadata correction
from a single shared `xy_action_limit` field to task-specific `xy_action_bounds`.
The final runtime source and snapshot are both
`a984fcfa56bb42c97546dfa1b778bf991a4c33fb220c2bfed2c12e08e808e1df`. The sampling
behavior itself was unchanged by that metadata correction; this source transition
is preserved rather than rewriting the generation-time registration.

The selected new catalogs are
`outputs/exp/20261102_133924_initial_clear_contract_r52/{pickcube,placesphere}/`.
Their manifests/HDF5 hashes are PickCube
`33520b09c500bb851c0a2f24a692fbdd4dff82f265c0e74513b3be54ad6d83ab` /
`a99e8d0183a62dadeb43491f0090b9e5d47bf48c9a6f2c6e8d8ec7b07767ae2f`, and
PlaceSphere `8e24d2c53376496d64ab3e80e3dc7361b583b6cfad2380e11d44a0dcf008aa6e` /
`c586bb02ae03bb35e937869180e1beb02153051b4bc076420454e9e3a83236b1`. Final
native restore preflights (`preflight_r52.json`) passed **30/30 per task**, with
exact saved state/controller hashes, restored and visible center agreement, zero
obstacle collision shapes, original layout assertions, and true measured initial
TCP clearance against cylinder radius + 5 mm. Minimum restored signed gaps are
PickCube `+0.0040043598 m` and PlaceSphere `+0.0085417679 m`; all 60 states are
strictly outside before any policy call. Preflight report hashes are
`1438528ec3b835ab701f651a280a58249d11b9234bcc030b61a3b8dcdf44636a` and
`917e54e4a8fdfe9a5b263e6052969b02b07b2770d8c5e18058e4146716e69afe`.

### Single lag-buffer profile and frozen native smoke

The existing scoring and acceptance disk is unchanged:
`tcp_xy_cylinder_radius_plus_5mm_v1`, actual cylinder radius + `0.005 m`, TCP XY
samples **and every chord**, including the initial and first-success points. The
separate working profile is `tcp_xy_working_clearance_3mm_v2`, a fixed `0.003 m`
positive target-path clearance. Its evidence basis is R50's maximum measured
2.614 mm endpoint/chord response error plus a declared 0.386 mm allowance. It is
not claimed to prove native plant safety. The filter intervenes before a requested
path enters this working disk; tangent segments preserve the buffer, and an
already-in-buffer but acceptance-clear TCP is moved radially outward rather than
rejected. A start inside the unchanged acceptance disk still fails closed. XY
controller bounds are per task: PickCube `[-1,1]` native normalized units and
PlaceSphere `[-0.35,0.35]`; no unrelated PickCube ±0.35 clipping is applied. Z,
rotation, gripper and native controller frame pass through. Body collision is not
a gate; cylinders are render-only with zero collision shapes.

Before final smoke, the complete two-index-per-task list (indices 0 and 1) was
registered against the final sampler outputs, source snapshots, selected frozen
models, training registrations/data, and native controller. PickCube remains R44
EMA SHA-256 `fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769`
with R41 normalizer `d14d56e3645504b81db4a41f32cf6869b805dbf003194b26433cfceb6023f543`;
PlaceSphere remains R30 EMA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f` and matched
Base `491970382f3ed8dac7ed88ad1ab3f5cd23fb2d29065291a507ff766c75a6f071`. The
final registrations are `pickcube_smoke_final_registration.json` and
`placesphere_smoke_final_registration.json`; identities are
`ab077491d94c37fde03680a3235a12c61ef02fec294a71132a45a23fa06bac96` and
`d02f4f0f8ed0c38f39151a4ace4499d5b8f2d24cd92e996e1dd346d8a88e739d`. They freeze
the exact final `0.003 m` profile, native per-task bounds and the actual source
snapshots. Training-data identities are pinned directly in these registrations.

The final frozen native smoke results (diagnostic only, **not** a 27/30 headline)
are PickCube **2/2 native successes**, PlaceSphere **2/2**, and zero measured TCP
sample/chord violations in all four rows. PickCube steps were 104 and 94; minimum
actual acceptance-disk sample/chord gaps were `+8.182 mm` / `+7.687 mm` and
`+2.962 mm` / `+2.962 mm`; XY interventions were 0 and 41. PlaceSphere steps were
166 and 167; minimum gaps were `+29.977 mm` / `+29.976 mm` and `+1.903 mm` /
`+1.903 mm`; interventions were 0 and 30. All smoke initial gaps were positive
(PickCube 38.584/44.039 mm; PlaceSphere 37.846/31.551 mm). The rows retain full
requested/filtered actions, actual TCP samples/chords and native results. Final
row hashes are PickCube
`514649ed7cfc03c9cbb9bf673f21402b209f4238de8d251ce3ceebbf37b0faed` and PlaceSphere
`fb2da250c48086c1685fce62841241a09c2813f5d8c6f99f5d8e09cb9d031dd6`.

An earlier smoke attempt using an intermediate target-only buffer implementation
is also retained as `*_smoke_rows/`; it is superseded by the final registered
profile and is not pooled with these rows. Its PlaceSphere index 1 timed out at
200 controls. The final tangent-path implementation was tested, registered under
new final registration files, and reran the same preregistered frozen indices; all
four final outcomes above are from the final source hashes. The original R48/R49
cohorts and outcomes remain separately retained and are not retroactively
replaced. No full 30-scene R52 trial was run. Proceed to full evaluation only
after an independent source/spec review of this diagnostic slice.

Focused tests after the final filter change passed **23 tests** across scene
sampling, filter, and evaluator seams. The required full suite passed **628 tests,
30 warnings in 94.36 s** in the `maniskill` environment:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line
```

Current R52 source SHA-256 values:
sampler `53bd6c3f0c6af733bb973401ec7c3aa6ef2696d5b2407cb7fa76f93ca9dea4bd`, layout
geometry `3c5422ba3d29393241aa5494f8a24f7f45b6b03d7e99489ef5c61664e0c9f577`,
preflight `89bd4c6cf76dc05fb2566766d2c8ca750950ead6955ccb40712803e1b28517cb`, filter
`cb227bf472fae871db1befecdbd34cc0ae99ad8fad461a879a522b68115c4840`, evaluator
`a984fcfa56bb42c97546dfa1b778bf991a4c33fb220c2bfed2c12e08e808e1df`, sampler tests
`b23e8dd9081070df66cab9b833d6d042b8c6399a83f1216a36347bc88a71155c`, and filter
tests `95cebb228cd0287e907dc3a42da6706124cfd9105974aeb412e287a8d8c615ee`.

Exact bounded commands (each run sourced `maniskill`; sampler ≤600 s/task,
preflight ≤300 s/task, native smoke ≤300 s/episode and ≤600 s/task):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 600s python -m examples.baselines.constrain.initial_clear_xy_scenes --task pickcube --output-dir outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --seed 521100 --target 30 --max-attempts 1000 --max-wall-seconds 600
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 600s python -m examples.baselines.constrain.initial_clear_xy_scenes --task placesphere --output-dir outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --seed 522100 --target 30 --max-attempts 1000 --max-wall-seconds 600
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.initial_clear_xy_preflight_r48 --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --output outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube/preflight_r52.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.initial_clear_xy_preflight_r48 --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --output outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere/preflight_r52.json
timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --indices 0 1 --output outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_smoke_final_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_smoke_final_registration.json --device cuda --register
timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --indices 0 1 --output outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_smoke_final_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_smoke_final_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 600s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --indices 0 1 --output outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_smoke_final_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_smoke_final_registration.json --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 600s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --indices 0 1 --output outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_smoke_final_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_smoke_final_registration.json --device cuda
```

The early R52 preflight files are retained alongside `preflight_r52.json`; the
final R52-named preflights include the restored layout-version/goal-height checks.
No branch change, commit, or push was made.

After independent review, the next full-cohort trial must preregister the complete
index list and run each task against the same frozen R52 catalog with new exclusive
row/registration paths. The exact command shape is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_full30_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_full30_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_full30_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/pickcube_full30_registration.json --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_full30_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_full30_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --output outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_full30_rows --registration outputs/exp/20261102_133924_initial_clear_contract_r52/placesphere_full30_registration.json --device cuda
```

## Round 51 — actor protocol and frozen-task distribution diagnosis (2026-11-02)

R51 ran a zero-environment-step actor differential and read-only PickCube feature
comparison before any shield tuning. No checkpoint, training/demo data, scene,
sampler, or task source was changed. The selected R48 cohorts remain byte-frozen;
there was no native diagnostic or full trial because the feature check exposed
an unresolved task-definition mismatch.

### Original-method protocol proof

The proof program is
`examples/baselines/constrain/actor_protocol_proof_r51.py` (SHA-256
`7c112fa72b2551034da062d3b900aad94e0c76c82217a7bfcd0b88a4f3fcadb2`). It
validated every PickCube R50 registered source, source snapshot, and input hash;
the R44 EMA remains `fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769`
and the R41 normalizer file remains `d14d56e3645504b81db4a41f32cf6869b805dbf003194b26433cfceb6023f543`.
The immutable R50 actor snapshot and current R44 actor source both hash to
`606298af35e6e0f58113d0b4e0ed1cd6a520d5f56615747eb9563011e4d93f2c`.

The historical `Horizon8ClearanceActor` constructor was attempted but its
`load_actor` metadata gate rejected the checkpoint. Instead, the proof explicitly
validated registered bytes/inputs, typed format/model/horizon/update/EMA fields,
normalizer content and identity, precheck/fit/Base hashes, strict model-state
loading, EMA restoration, and native 100-step squared-cosine DDPM scheduler. It
then called the unchanged original `Horizon8ClearanceActor.get_action` method
directly on a `__new__` instance whose properties were those validated values.
This proves method equivalence for the checked runtime objects; it does **not**
claim constructor equivalence. Against R50's selected evaluator adapter protocol,
the same saved two-frame R48 initial observation, checkpoint, device, and per-index
RNG state were compared for indices 0, 1, and 3: all three eight-action tensors
were exactly equal (maximum absolute error `0.0`) and post-call CPU/CUDA RNG states
were exact. R41/R40 profile installation repeated twice in evaluator order and
then reverse order left profile identities unchanged. No environment step ran.
Proof JSON SHA-256:
`55f1f2931eece3b254d2b6cfa15fb194ea9535d3d026b4447f427d9fe4530fea`.

### Frozen PickCube feature/task-contract finding

The read-only report and reproducible program are
`outputs/exp/20261102_200000_xy_protocol_buffer_r51/pickcube_feature_distribution_diagnosis.json`
(SHA-256 `be35f63b6b0cddab83be02766a2fecc9cb99be0efdd98f3fbf28a85c12f6ebd8`) and
`examples/baselines/constrain/pickcube_feature_distribution_r51.py` (SHA-256
`b79f8664207ee88ccde710cafd63211fcd343536b78b2343c767870a7981f8ce`).
It compares all 30 frozen R48 raw observations with the initial observations of
the unchanged 1,000-direct/500-obstacle demonstrations. R48 manifest/HDF5 hashes
are `6ba3fac915f68464862ababbba602ecd0612023908efa8134eff1e1cafe31fa7` /
`5dc7c8808787c7edaf9b64a3adc9b02a7afaf22aab8b75df09fe134f53449089`; frozen
training manifest/HDF5 hashes are `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486` /
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`.

The critical mismatch is not limited to height. Every frozen demonstration initial
observation has goal Z `0.12 m`; R48 native-reset goal Z has min/median/max
`0.026707/0.215295/0.314346 m`, with 5/30 at or below `0.08 m` and 19/30 above
`0.16 m`. Frozen training goal XY spans X `0.0483–0.1800`, Y `-0.0600–0.1200`;
R48 reset-goal XY spans X `-0.0987–0.0994`, Y `-0.0924–0.0944`. The R46 historical
scene manifest has goal XY X `0.1068–0.1795`, Y `-0.0591–0.1197`; its height rule
is per-scene obstacle geometry, not a new goal-height distribution. R48
`sample_task` reads the goal actor from the first native reset and passes that
XYZ back as `goal_pos` on its saved reset (`initial_clear_xy_scenes.py`, lines
195–212). Thus the frozen native-reset cohort differs from the historical
training/layout goal contract in both XY and Z. R48 cube quaternion components
remain normalized; R48 qpos scene 0 differs from training-median qpos by a maximum
`0.0360 rad` (the full per-feature distributions are in the report).

This is a legitimate task-distribution conflict, not grounds to resample, filter,
or discard catalog rows. Consequently R51 did not implement or register a new
buffered profile and did not run selected-index smoke or 30-scene episodes. The
R50 trace evidence still supports testing a working buffer separately after the
task contract is resolved: PickCube 3's filtered target predicted `+2.471 mm`
clearance while the actual endpoint was `0.143 mm` inside and chord `0.158 mm`
inside (response error `2.614 mm`); PickCube 1 and PlaceSphere 0 also crossed
after `1 µm` predicted clearance. A `3 mm` working buffer would cover the largest
observed R50 endpoint error plus a declared `0.386 mm` allowance, but this is an
evidence-derived candidate only, not a validated universal margin. The acceptance
radius remains cylinder radius + `5 mm`; no radius relaxation or initial-state
waiver is proposed. R50 PickCube 0 stayed at least `45.344 mm` outside the profile
and stalled at `no_grasp`, so persistent tangent behavior is not evidenced.
Controller evidence remains R50's native `pd_ee_delta_pose`: normalized root
translation maps at `0.1 m/unit` with physical `±0.1 m` controller bounds, root-
aligned XYZ rotation norm-limited then scaled by `-0.1 rad`, while the R50 XY
filter action cap is `±0.35`; non-XY channels must remain passthrough.

R51 verification command (under `maniskill`; both proof and feature diagnosis
were bounded to 300 seconds):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.actor_protocol_proof_r51
timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.pickcube_feature_distribution_r51
```

The required constrain suite passed **624 tests, 30 warnings in 93.45 s**:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line
```

No buffer regression test or native gate was appropriate before resolving the
goal-layout contract. No source in ManiSkill was modified; no new training,
checkpoint, demo, scene/layout or episode artifact was produced. No commit,
branch change, or push was made.

## Round 50 — bounded native runtime diagnosis (2026-11-02)

This is a four-episode instrumentation diagnostic, **not** a 30-scene gate or
selective performance headline. R48 cohorts, models, task behavior, XY filter,
and training data were not changed. Selected indices were preregistered from
ascending R49 rows before any R50 outcome: PickCube runtime failures `1` and `3`,
PickCube's first completed native failure `0` (`no_grasp`, 500-step timeout),
and PlaceSphere runtime failure `0`. R49 artifacts remain unchanged.

The plan is `outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/diagnostic_plan.json`
(SHA-256 `55dd29d30377deafb919139ff94bf6af9c9bdcc7e2b114a3e20173982b82471c`).
Task registrations and source snapshots were made before episodes: PickCube
registration identity `ea4fd85ce5bc1ec9b07ad90629226c5b3bdd96b66332d65353ee6a299c163a9d`,
PlaceSphere `97373526318fc1d971bc77f1810ff5a82a3fbb2c95c70efd6267d7d32497930b`.
Each pins exact R48 catalog, selected checkpoint, training inputs and runtime
sources. PickCube and PlaceSphere checkpoints retain R44 EMA SHA-256
`fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769` and R30 EMA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`.
R48 manifest/HDF5 hashes remain the frozen identities recorded in R48 above.
Registrations validated after the runs; their source/input hashes still match.

The evaluator now snapshots the restored initial TCP, appends each attempted
control before filtering, and retains executed requested/filtered actions and
actual TCP before/after even when native `env.step` raises. Filter-rejected
attempts identify the control index and explicitly say `env_step_executed: false`;
an exception from an executed native step retains its latest measurable pose.
Runtime-error rows retain native flags, inferred task phase, independent
catalog-geometry sample/chord scoring, exception type/message, and do not
fabricate success. The first-native-success sample and preceding chord remain
in the independent score. The row denominator and nonzero CLI exit on runtime
errors are preserved. Existing evaluator `finally` blocks close each native
environment.

### Registered episode commands and retained outcomes

Exact registrations and runs (source the required conda environment before each
command; each episode alarm is 300 s, and the combined native-run ceiling is
1,200 s):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
timeout --signal=TERM --kill-after=20s 120s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/pickcube --indices 1 3 0 --output outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/pickcube_rows --registration outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/pickcube_registration.json --device cuda --register
timeout --signal=TERM --kill-after=20s 120s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/placesphere --indices 0 --output outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/placesphere_rows --registration outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/placesphere_registration.json --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 900s python -m examples.baselines.constrain.render_only_xy_eval --task pickcube --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/pickcube --indices 1 3 0 --output outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/pickcube_rows --registration outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/pickcube_registration.json --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.render_only_xy_eval --task placesphere --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/placesphere --indices 0 --output outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/placesphere_rows --registration outputs/exp/20261102_130536_xy_runtime_diagnosis_r50/placesphere_registration.json --device cuda
```

Both run commands returned nonzero after retaining all attempted rows, as
required: PickCube `1` and `3` failed at controls `92` and `9`; PickCube `0`
completed all 500 controls and timed out with native phase `no_grasp`; PlaceSphere
`0` failed at control `4`. No R50 native-success claim is made.

### Trace evidence and controller diagnosis

Independent scoring uses the frozen cylinder center/radius plus 5 mm and every
actual TCP sample and chord. Initial TCP points were clear. The first post-step
violations were:

| Task/index | First violating executed control | Actual sample gap (m) | Actual chord gap (m) | Filtered linear-target gap (m) | Unfiltered requested linear-target gap (m) | Native/task phase |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PickCube 1 | 91 | -0.000144245 | -0.000144245 | +0.000001000 | -0.013247121 | grasp_no_lift; native grasp false, crossing true |
| PickCube 3 | 8 | -0.000142840 | -0.000157739 | +0.002471041 | +0.002471041 | no_grasp |
| PlaceSphere 0 | 3 | -0.000024310 | -0.000024310 | +0.000001000 | -0.007238105 | no_grasp |

At those controls PhysX actually executed the filtered actions; the next
requested control then hit `TCP starts inside the forbidden disk`. These traces
confirm the boundary/lag hypothesis for PickCube 1 and PlaceSphere 0: a predicted
1 micrometre clearance did not absorb measured motion error. PickCube 3 is
stronger evidence of coupled response error: the unmodified filtered target
predicted 2.471 mm clearance while the measured endpoint was 0.143 mm inside,
with a 0.158 mm chord violation. The registered Panda `pd_ee_delta_pose`
controller uses normalized root-translation action with ±0.1 m physical bounds;
its root-aligned XYZ rotation vector is norm-limited then scaled by -0.1 rad.
The XY filter maps normalized XY by 0.1 m/action unit and additionally clips to
±0.35. Z/rotation pass through, so their IK/controller response and actuator
settling remain coupled to measured XY motion. The current evidence justifies a
lag-aware positive clearance buffer; it does **not** calibrate a universally
sufficient margin or justify changing it without a separately registered
follow-up.

PickCube scene `0` never approached the cylinder: minimum actual center distance
was 0.080344 m (minimum forbidden-disk gap 0.045344 m), yet it remained `no_grasp`
for 500 steps. The first negative requested gripper command was control 54;
there was no grasp, lift, or transport. It is a grasp-stage stall, not a loop at
the detour circle. R50's 500 requested native actions exactly match R49 scene 0
in sequence and SHA-256
`e5bfa49048e060bfce2ea066047e80bb1e15349378724c4ed1cc2553a4f8d126`; seed, scene
state, restored observation and the registered actor/rollout sources also match.
This rules out a selected-adapter/action mismatch relative to the R49 R44 actor
path on this identical saved two-frame initial condition and RNG seed. PlaceSphere
retained the registered R30 actor, codec, sampler and rollout source bytes; its
requested pre-filter native actions are retained, but the failed R49 index 0 has
no old action trace for an exact trace-to-trace comparison.

Offline calculations, including every first-failure boundary record, are
`offline_analysis.json` (SHA-256
`18fd6c37b813db983c6adb529c60b79e2058ba7b1fc12bd776d6998075142b5d`); its
reproducible analysis program is `analyze_diagnostic.py`. Row hashes are
PickCube `16852d1d07c9ebc6f6bcb7f1cc2ede3ed325417484b0b875eef7f9af890d1c17`
and PlaceSphere `eb9a668e3db266d130a05753413f3e9407ed1300151098656e132c4b7494d77d`.
These four selected scenes are diagnostic only and do not replace, rerun, or
qualify the R48 cohorts.

TDD evidence: the new regression tests first failed against the old wrapper
because no executed trace/failed control metadata survived the target failure.
After instrumentation, the focused evaluator file passed **11 tests**, including
the induced actuator-lag/filter-failure trace, native-step exception pose capture,
failure-row truthfulness, and first-success sample/chord endpoint. The required
full suite passed **624 tests, 30 warnings in 94.94 s** in `maniskill`:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line
```

No behavior source was edited after the R50 registrations or native runs. No
cohort, checkpoint, demo, or training artifact changed; no commit, branch change,
or push was made.

## Round 49 — selected-policy trials on frozen initial-clear cohorts (2026-11-02)

Both selected policies were run sequentially against all scene indices `0..29`
from the unchanged R48 catalogs in `outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/`.
The sole trial directory is
`outputs/exp/20261102_170000_initial_clear_xy_eval_r49/`; no scenes, training,
demos, checkpoints, policies, or evaluator code were changed for these runs.
The frozen checkpoints are PickCube R44 EMA
`fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769` and
PlaceSphere R30 EMA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`.

Each complete index list was preregistered before either task's first episode.
PickCube registration identity is
`19e4d6d1549d07012f3932f577aec4c8a6a7ba14eafaca6941e5b63a7c242ef4`
(17 source snapshots); PlaceSphere is
`0284bfe190ef53c379e89edf8cc20e004a2e15c1235db6f75c802a740aab41e3`
(13 source snapshots). They pin the checkpoint(s), cached normalizer, R48
manifest/HDF5, training registration, training/demo manifest and HDF5, and the
evaluator's actor, codec, scene loader, rollout and native controller source
entries. Registration and all directly registered source/input/snapshot hashes
validated again after both runs. Checkpoints and training/demo/catalog hashes
remain equal to their R48 identities. Registrations use the fixed
`tcp_xy_cylinder_radius_plus_5mm_v1` profile, native `pd_ee_delta_pose`,
render-only visible cylinder with zero collision shapes, and do not gate body
contact. Each evaluated row retains native action requests, filtered actions,
per-control live obstacle center/radius, measured before/after TCP XY, native
flags, restoration identity, and independent catalog-geometry sample/chord
scoring.

Exact preregistration and trial commands (run one task at a time):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s \
  python -m examples.baselines.constrain.render_only_xy_eval --task pickcube \
  --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/pickcube \
  --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --output outputs/exp/20261102_170000_initial_clear_xy_eval_r49/pickcube_rows \
  --registration outputs/exp/20261102_170000_initial_clear_xy_eval_r49/pickcube_registration.json \
  --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s \
  python -m examples.baselines.constrain.render_only_xy_eval --task pickcube \
  --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/pickcube \
  --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --output outputs/exp/20261102_170000_initial_clear_xy_eval_r49/pickcube_rows \
  --registration outputs/exp/20261102_170000_initial_clear_xy_eval_r49/pickcube_registration.json \
  --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s \
  python -m examples.baselines.constrain.render_only_xy_eval --task placesphere \
  --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/placesphere \
  --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --output outputs/exp/20261102_170000_initial_clear_xy_eval_r49/placesphere_rows \
  --registration outputs/exp/20261102_170000_initial_clear_xy_eval_r49/placesphere_registration.json \
  --device cuda --register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s \
  python -m examples.baselines.constrain.render_only_xy_eval --task placesphere \
  --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/placesphere \
  --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --output outputs/exp/20261102_170000_initial_clear_xy_eval_r49/placesphere_rows \
  --registration outputs/exp/20261102_170000_initial_clear_xy_eval_r49/placesphere_registration.json \
  --device cuda
```

### Native outcomes and XY evidence

| Task | Evaluated / fixed denominator | Native successes | Runtime errors | Zero-XY among evaluated | Min sample gap / min chord gap (m) | Filtered XY steps / controls | Evaluated terminal phases |
|---|---:|---:|---:|---:|---:|---:|---|
| PickCube | 13 / 30 | 0 / 30 | 17 / 30 | 13 / 13 | 0.00123237 / 0.00123161 | 47 / 6,500 (0.7231%) | no_grasp 13; timeout 13 |
| PlaceSphere | 20 / 30 | 16 / 30 | 10 / 30 | 20 / 20 | 0.00067899 / 0.00067899 | 2 / 2,673 (0.0748%) | task_success 16; release_or_stability_failed 4; timeout 4 |

All 27 runtime-error rows (PickCube indices `1,3,5,6,7,9,11,12,13,17,19,20,22,24,25,27,28`; PlaceSphere indices `0,8,10,12,13,14,17,21,26,28`) report
`ValueError: TCP starts inside the forbidden disk; XY constraint cannot recover`.
These errors retain their full denominator and are not counted as successes.
R48 admission and preflight gaps were positive for all initial scenes; the
runtime error therefore arose after restoration during policy execution, rather
than from an initially inadmissible catalog entry. However, the evaluator only
attaches `_render_only_xy_trace` after `run_episode` returns. Its caught
exceptions consequently saved no partial trace for those 27 failures. Their
first problematic control step, actual prior sample/chord gaps, intervention
history, task phase and any failed-row native flags are **unavailable** in this
immutable run. The failures are consistent with native post-step TCP movement
entering the disk after an admissible filtered action, but the missing partial
trace prevents confirming that transition or identifying its control index.
The recorded minima and zero-XY counts above apply only to completed rows and
must not be generalized to the full 30-scene cohorts. Body contact was 0 in
completed rows and remains non-gating.

Both task processes returned nonzero after writing all 30 rows because runtime
errors are propagated. The native-success targets (at least 27/30) were missed:
PickCube 0/30 and PlaceSphere 16/30. The all-30 zero-XY criterion is not
established due to the 27 failed rows without traces; it does not pass. No
reruns, selective rows, scene changes, margin tuning, or policy selection were
made. The next smallest evidence-producing change is to retain the live partial
TCP/action trace, current control index and phase in the evaluator's exception
row (while preserving nonzero failure propagation), then register that source
change and run a new complete 30-attempt cohort per task. Any further margin or
policy adjustment should follow that trace evidence, not precede it.

Immutable artifacts: PickCube `pickcube_rows/rows.json` SHA-256
`5a7989aea5e4b3da932a23881b1fb51b5cbcd12b45b75321dd06b41675eb2e00`, summary
`5866dc87043c0b121324561f0e808ebf29f5e77c797b300bb412e4c917f26496`; PlaceSphere
`placesphere_rows/rows.json` SHA-256
`73cbda8f80dee9d432190bf65354f65ebc1e254062636b13f94c3f40a722bed2`, summary
`dfb72423acef60c2b8eda556a3802d672c002db3c57abad68b1a284d6747e739`. Offline
aggregate calculations are in `offline_analysis.json`. No evaluator runtime code
was edited after preregistration. No implementation or test suite was run; the
work was native evidence collection and offline aggregation only.

## Round 48 — initial-clear render-only XY cohorts (2026-11-02)

Two independent 30-scene native reset catalogs are frozen at
`outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/{pickcube,placesphere}/`.
They are evaluation snapshots, not expert demonstrations: sampling used native
task reset/layout generation, captured the complete native state, controller,
and task observations, and made **zero task steps, policy calls, or demo writes**.
The cylinder center is on the direct object-to-goal XY segment at a seeded
fraction in `[0.4, 0.6]`; radius is fixed at 0.03 m. Cylinder visual heights are
frozen per task (PickCube 0.14 m; PlaceSphere 0.08 m). Direct-path endpoints are
outside the cylinder disk. Both catalogs use the explicit existing R47
TCP-only profile `tcp_xy_cylinder_radius_plus_5mm_v1`; admission uses the
measured native TCP after reset and requires a strictly positive signed gap to
radius + 0.005 m. No legacy egress waiver is used.

| Task | Seed | Accepted / attempts | Rejections | Minimum restored signed TCP gap | Manifest SHA-256 | HDF5 SHA-256 |
|---|---:|---:|---|---:|---|---|
| PickCube | 481100 | 30 / 149 | 92 too-short transport; 25 inside; 2 duplicate layout | 0.000822878 m | `6ba3fac915f68464862ababbba602ecd0612023908efa8134eff1e1cafe31fa7` | `5dc7c8808787c7edaf9b64a3adc9b02a7afaf22aab8b75df09fe134f53449089` |
| PlaceSphere | 482100 | 30 / 89 | 24 too-short transport; 25 inside; 10 duplicate layout | 0.000642924 m | `5e677558d10bfb2d993090d8acc217c70ada24962a6f0cacc9f1586572d94081` | `864a7a7de25ee19c0a905200abadde5a6e439b62cf99248e12e4b23ef53cd66e` |

Each generation was bounded by 1,000 attempts, 600 s wall time and a 300 s
per-reset alarm. One deterministic seed stream per task was used. Attempt journals
include the native reset attempt, seed, rejection/admission reason, and elapsed
time. Duplicate filtering checked the frozen task training manifest (direct and
obstacle records), the previous task-specific evaluation catalog, and prior
accepted samples: same seed, or both start and goal coordinates componentwise
within 0.02 m. This is the exact identity scope checked; it makes no broader
claim about other task datasets. Training HDF5 and manifests remained unchanged.

The true zero-step native restoration audit passed **30/30 per task** from the
saved catalog, with exact native state and controller hashes, post-restore task
and visible-cylinder center agreement, zero cylinder collision shapes, and
catalog-matched observations (PickCube exact; PlaceSphere within the catalog's
native observation tolerance). Audits:
`pickcube/preflight.json` (SHA-256
`8f7863b080e20e132b81d7a751bbf5850ea4390065873c5683a8d3a006e8f8dd`) and
`placesphere/preflight.json` (SHA-256
`3c496265d382f4c37d8428f9ad70ff8c9d5e3efb7e6249e20eb541a4c52c2870`). Minimum
post-restore profile gaps are 0.822878 mm and 0.642924 mm respectively. Source
snapshots, sampling registrations, input identities, preflight registrations,
and all attempt journals are retained beside each catalog. Sampling registration
identities are PickCube `312f911603512e14c4af3ffd9a16e929b0641866f81e2d0c1fb4e5023c69a629`
and PlaceSphere `3abc7508072258005bbcf23fb774c63a27b4f7e3e38a4465ce128a1d7bec7946`;
all registered data inputs and source snapshots revalidated byte-identically.
No policy episodes were run; native success rates remain unevaluated.

Earlier R48 attempts remain unmodified under `20261102_140000_initial_clear_xy_r48/`,
`20261102_143000_initial_clear_xy_r48_final/`, and
`20261102_150000_initial_clear_xy_r48_registered/`; they are not the selected
cohort. The first PickCube sampler process exited before writing its sampling
registration due to a local argument-shadowing error. A first no-step preflight
also exited while formatting PickCube's native reset return; both were corrected
in a fresh attempt with new source/input registrations. The first PlaceSphere
catalog used 0.12 m visual height; it was retained, then superseded by the
registered 0.08 m task-matched cohort in the selected output. None of these
attempts ran policy actions or changed old scenes/training data.

R48 intentionally leaves the frozen R44 PickCube EMA
`fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769` and R30
PlaceSphere EMA `6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`
unchanged. For the next round, first preregister each full index list with
`render_only_xy_eval --register --scenes <catalog>`, then validate the
registration and run the evaluator with the same `--scenes` path, all indices
`0..29`, an exclusive new output directory, 300 s per episode and 3,600 s per
task. Do not start policy evaluation until both registrations and their source/
input hashes are frozen. The required command shape is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 3600s \
  python -m examples.baselines.constrain.render_only_xy_eval --task <pickcube|placesphere> \
  --scenes outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/<task> \
  --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --output outputs/exp/YYYYMMDD_HHMMSS_render_only_xy_r48/<task>_rows \
  --registration outputs/exp/YYYYMMDD_HHMMSS_render_only_xy_r48/<task>_registration.json \
  --device cuda
```

Registration command uses the same arguments plus `--register`; run it once per
task before the evaluator command. The R48 acceptance gate is still a future
result: at least 27 native successes per 30 and zero measured TCP XY sample or
chord violations from each initial state through first native success.

## Round 47 — restored-geometry TCP-only XY preflight (2026-11-02)

**Decision gate: native trials stopped.** The required no-step preflight restored
all 30 saved scenes per task using the native catalog restore and render-only
environment, then checked saved state/controller hashes, task-state center,
visible cylinder pose, cylinder visibility and collision-shape count. All 60
state/controller/center checks matched the frozen catalog; all cylinders were
visible and had zero collision shapes. The report is
`outputs/exp/20261102_130000_restored_tcp_xy_r47/preflight.json` (SHA-256
`8c29494a56585787495dfe6b6318f021c22c005846134d90ae51bffbaa2d58cf`).

The R47 profile is explicitly named/versioned
`tcp_xy_cylinder_radius_plus_5mm_v1`: forbidden XY disk = restored visible
cylinder radius + 0.005 m. It is TCP-only and excludes cube/sphere footprints;
this differs from R45's object-footprint profile and from the historical
physical object-crossing latch. Actual cylinder radius was 0.030 m for every
scene and the optional margin was 0.005 m. Negative signed gaps below are
measured against the **actual cylinder** (before margin); the second value is
the gap to the R47 profile:

| Task | Initial TCP already inside actual cylinder: scene index (actual gap m; R47 gap m) | Outside cylinder but within margin |
|---|---|---|
| PickCube | 14 (-0.014206; -0.019206), 18 (-0.014020; -0.019020), 23 (-0.012045; -0.017045) | none |
| PlaceSphere | 1 (-0.003890; -0.008890), 5 (-0.000432; -0.005432), 8 (-0.014738; -0.019738), 10 (-0.004290; -0.009290), 15 (-0.000779; -0.005779), 16 (-0.003465; -0.008465), 17 (-0.018105; -0.023105), 19 (-0.007132; -0.012132), 22 (-0.013552; -0.018552), 26 (-0.006783; -0.011783) | 7 (+0.002496; -0.002504), 13 (+0.000326; -0.004674), 18 (+0.002462; -0.002538) |

Because several saved TCP starts are already inside the actual cylinder disk,
all-time zero-XY violation is impossible on the unchanged 30-scene cohorts.
No scene/cylinder was moved, no initial point was waived, and no smoke or full
policy rollout was run. This is a user-decision blocker for the requested
acceptance; native task success >=27/30 is consequently not newly evaluated.
The exact bounded command was
`source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && timeout --signal=TERM --kill-after=20s 300s python -m examples.baselines.constrain.restored_xy_preflight_r47 --output outputs/exp/20261102_130000_restored_tcp_xy_r47/preflight.json`
and completed successfully.

The evaluator fix now reads restored task-state center and cylinder radius at
each native control step, verifies the visible actor's center, records that
geometry with each measured TCP endpoint, and independently scores every sample
and chord against the frozen scene center/radius. Registrations use format
`render_only_xy_eval_r47_v1`, capture exact source-byte snapshots, directly hash
the relevant actor/codec/normalizer/evaluator/scene/controller sources and
training/demo/checkpoint/scene inputs, and validate those hashes after execution.
Frozen final registrations are PickCube
`pickcube_registration_attempt03.json` (identity
`65552169de3881b85c7b535c97d4d2d81072552a1a71150db1180e3bf81609c5`; 17
direct source entries and snapshots) and PlaceSphere
`placesphere_registration_attempt02.json` (identity
`7f87926b30c1476684b86d8a45c37e8187ca7020d4220ee1fe461e8eb11db7ff`; 13
direct source entries and snapshots). Both include all 30 indices and Git HEAD
`c48e82b32c9a8cc9b16f60211662d8b426fe2f0e`. PickCube registers its standalone
position normalizer artifact. PlaceSphere uses identity/raw observation
conditioning, pinned directly to `placesphere_constrain.py` as its normalizer
definition. Registration validation passed for all input and source hashes.
Execution is bounded to 300 s per episode and 3,600 s per task; timeout rows
propagate a nonzero process exit, and `SystemExit`, `KeyboardInterrupt`, and
other `BaseException` subclasses are not swallowed. The final full suite passed
617 tests and 30 warnings in 87.29 s. Exact command:
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
Focused regression tests cover pre-first-step scene restoration, wrong cached
center rejection, and chord scoring. Registrations are frozen, but no episodes
or result directories were created because the precondition failed. Current
source hashes: runner
`4815e96c253763b8ef6690e20aa23abdd8125b83d59ef0484cd8f8c4baa24118`,
preflight `cd8c1475fb69282ad96aee8228ce3f31c772df08089fcbcedf110748a7c309b8`,
test `536c73fd2c7cb3726e6a01bd08eb63cc16109ddf6333feb10b4547f1bd03199f`.
Base HEAD was `c48e82b32c9a8cc9b16f60211662d8b426fe2f0e`; all R47 work is
uncommitted. No checkpoints, training, demos, or scene files were changed.

## Round 46 — selected-policy render-only XY-constrained development trial (2026-11-02)

**Review correction — XY evidence invalid:** instrumentation captured the
obstacle center before scene restoration. PickCube scene 0 was filtered and
scored against `[0, 0.20000000]`, whereas its catalog and restored cylinder
center are `[0.03462751, 0.03612041]` (approximately 0.167 m apart).
Consequently, the XY counts and XY-qualified results below describe the wrong
circle, not avoidance of the restored obstacle, and cannot satisfy acceptance
for either task. Native task outcomes remain recorded; the old rows and
registrations are preserved. The scene-11 filter rejection likewise used the
cached default center and does not establish an initial violation of the
restored cylinder. The next slice must use live post-restoration geometry and
the explicitly TCP-only disk (visible cylinder radius + 5 mm margin), then
register exact source snapshots and run new trials. No obstacle or saved
initial state is to be moved to obtain a pass.

This is an engineering trial labeled **ConstrainNet+XY constraint**, not a formal
performance or generalization claim. Frozen physical data remain unchanged at
1,000 direct + 500 obstacle demonstrations per task; no checkpoint was trained,
no scene sampled, and no demonstration/replay generated. Both existing 30-scene
catalogs were restored in `obstacle_mode="render_only"`; each runtime audit
found a visible cylinder and zero cylinder collision shapes. Native task success
remains the task-specific gate; body collision is not an acceptance gate. The
historical `obstacle_crossing` latch is retained separately from the fresh XY
trace score.

### Implementation and frozen inputs

`render_only_xy_eval.py` wraps the selected actor's native `env.step` seam. For
each actually executed control action it reads the current native TCP world XY,
filters only XY, steps the environment once, and records the measured next TCP,
requested and filtered seven-channel actions, and native `info`. Z, rotation,
and gripper pass through. The historical (invalid for R47 acceptance)
object-footprint radii were
`cylinder + sqrt(2)*cube_half_size + 0.005 m` (PickCube) and
`cylinder + sphere_radius + 0.005 m` (PlaceSphere). Runtime registrations pin the
source and selected checkpoint/normalizer/scene identities before episodes.
DDPM-100 epsilon, squared-cosine schedule, clipped samples, action decoding,
selected EMA and `[1:9]` slice/execution protocol are preserved for PickCube R44;
PlaceSphere uses R30 DDIM-100 eta=0, single plan and its existing decoded native
actions. The controller source specifies ±0.1 m position bounds; the filter
uses the native normalized ±0.35 XY action range (0.1 m/action unit) and does
not clamp or recode other channels.

Selected checkpoints: PickCube R44 Control EMA
`outputs/ckpt/20261029_160000_pickcube_horizon_clearance_r44_control/final.pt`
(SHA-256 `fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769`);
PlaceSphere R30 Constrain final EMA
`outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt`
(SHA-256 `6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`),
with matched Base identity `491970382f3ed8dac7ed88ad1ab3f5cd23fb2d29065291a507ff766c75a6f071`.
The existing scene catalogs are PickCube
`outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes/`
(manifest `ffb2b54b6bf4f389d8bd81b0d5f282d833a297c6375ae2189c44b8359a9ee676`,
HDF5 `3f2312dba450110a4deb839ea856aa95e41bbf77f1ea53de386b5a759f3dc1c8`) and
PlaceSphere `outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes/`
(manifest `9e1d8ee4786d2336f01e07d664ca37c3dcc7ca455e7876176def65c18b63c119`,
HDF5 `adc7d1b2a16745c459db53d2c8f818ac87d1ffffe21fd44c54509f77cef342d7`).
The immutable data identities remain PickCube HDF5/manifest
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f` /
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, and
PlaceSphere HDF5/manifest `796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961` /
`aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`.

Fresh full-cohort registrations (30 indices, 0–29) are
`outputs/exp/20261102_111155_render_only_xy_r46/pickcube_full_registration.json`
(identity `48f6777f653c77eed9ad64b14a848c848d39e24b7fc2943aa27fd1f07a5976b3`) and
`placesphere_full_registration.json` (identity
`3ea969e26ae8c3cce82b950b1ecb9e66c4287a50e22678efedd508208dfba10f`). They
label the method separately and set `formal_or_generalization_claim: false`.
Successful smoke rows are retained at `pickcube_smoke02/` and
`placesphere_smoke03/`; both native-successfully completed with zero measured
TCP-XY violations (120 and 128 controls respectively). Earlier pre-episode
setup failures and their immutable error rows remain under the separate
`*_smoke/` and `*_smoke02/` attempt directories; they were not overwritten.

### Constrained 30-scene outcomes

| Task | Evaluated / denominator | Native success | XY-qualified native success | zero measured TCP-XY | Body contact (reported only) | Runtime error |
|---|---:|---:|---:|---:|---:|
| PickCube | 29 / 30 | 29 / 30 | 29 / 30 | 29 / 30 | 0 / 30 | 1 / 30 |
| PlaceSphere | 30 / 30 | 27 / 30 | 27 / 30 | 30 / 30 | 0 / 30 | 0 / 30 |

PickCube failed before its first action on saved scene index 11 because the
restored initial TCP was already inside the exact forbidden disk; the filter
correctly refused to claim recovery. A no-step inspection of that retained
scene found TCP `[0.01031554, 0.00547008]`, center `[0.04309617, 0.02342713]`,
radius `0.06328427 m`, initial signed gap `-0.02590747 m`; state SHA-256
`7d942ee613be8e5f1d900572c14b9abd635b73ec89db39b7a5363e3b91b0509a`. This stays
in the 30-scene denominator and is not rerun. PickCube's live historical crossing
latch was true on 7 evaluated scenes even though all 29 executed TCP traces
passed the independent sample-and-chord test; these are separate measurements.
PlaceSphere's latch was false on all 30. Thus the native-success threshold is
met for both (29/30 and 27/30), but the across-all-30 zero-XY criterion is **not
met for PickCube** because of its initial violating state. No safety or
generalization claim follows.

Full retained rows: `pickcube_constrained30/rows.json` SHA-256
`d3069c7b0df41e2f655ff3e409a355d8aefc8304d285c616496655b0e7d1b180`;
`placesphere_constrained30/rows.json` SHA-256
`6bf0546c2f8768d6dc1ef6202deed76b96e1d6ac8a7633808993c2c5e8156645`.
Rows retain restored state/observation hashes, every executed native request and
filtered action, per-control TCP before/after, native info, and actual trace
validation through first success. Runtime errors count against the full 30.

TDD: new tests first failed collection because the integration module did not
exist; behavior tests then caught fixture-contract assumptions before reaching
4/4 passing. They cover measured per-step TCP, unchanged non-XY channels, radius
preservation, filtering when Control is disabled, immutable source guards, and
failure denominators. The required full command
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
passed **615 tests, 30 warnings, 97.82 s** in `maniskill`.

Execution used finite per-episode `timeout --signal=TERM --kill-after=20s 300s`
smoke bounds and finite full-trial `... 3600s` bounds. A source-metadata-only
edit to the runner followed the completed trials to add explicit constraint and
physical-data identities to subsequent registration creation; the full trial
registrations record runner hash `eb134252f77f6dae79325a666f92de3ad2c1917b9546aa63d344922d94ec7e5c`,
while the current runner is `76b39032bd4e5aa9c268d9ead0e22cf42868d7ca2f7f09b809d47915c8099e62`.
The edit did not change the rollout path, but the original source bytes were not
snapshotted, so this post-trial source-file mismatch is retained as a provenance
limitation. Also, the frozen pre-episode registrations directly pin checkpoint,
normalizer and scene files, but do not directly list the demonstration HDF5,
manifest, or training-registration hashes; these identities are recorded above
from their frozen artifacts/checkpoint lineage, not from those trial JSONs.
Historical physical-mode results remain untouched; a same-mode unfiltered
30-scene comparator was not run.

## Round 45 — render-only geometry and XY constraint prototype (2026-11-02)

This partial engineering slice changes only the experimental PickCube and
PlaceSphere obstacle environments. Both accept `obstacle_mode="physical"`
(unchanged default) or `"render_only"`; the latter builds the same visual
cylinder with zero collision shapes. State dictionaries retain the configured
cylinder radius, height, center, enable flag, and task-specific state, so saved
scene restore remains geometrically identical. In render-only mode environment
`success` is native task success; physical mode retains the historical
collision/crossing-qualified success. No data, layouts, or checkpoints were
created or changed.

The standalone seam is
`examples/baselines/constrain/inference_xy_constraint_r45.py`: `XYConstraint`
uses measured TCP XY plus one native `pd_ee_delta_pose` action. XY action units
are bounded to ±0.35 and map to ±0.035 m per step using the Panda 0.1 m/unit
translation scale. Z, rotation, and gripper are passed through. It keeps the
existing forbidden radius exactly: cylinder radius + cube circumscribed XY
radius + 0.005 m for PickCube, and cylinder radius + sphere radius + 0.005 m for
PlaceSphere. `validate_tcp_trace` scores every actual XY sample and chord,
including the first-success sample. These interfaces are unit-tested, but the
filter is not yet wired into either selected policy evaluator.

TDD evidence: `tests/test_render_only_xy_constraint.py` first failed collection
because the intended new public module did not exist. Once implemented, the
focused suite passed (6 passed); an intermediate SAPIEN API assumption in the
visual assertion was corrected to inspect the actual `RenderBodyComponent`.
The complete command
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
passed: 611 passed, 30 warnings, 86.19 s (conda environment `maniskill`).

**Native acceptance is not complete.** No round-45 source/input registration,
saved-scene policy rollout, full executed TCP trace, per-task success check, or
30-scene evaluator was produced. Therefore the >=27/30-per-task and zero-TCP-XY
violation criteria remain unverified, and there is no 30-scene trial command
yet. Historical physical collision-aware reports and identities above are
unchanged; this prototype has the explicit label `render_only_xy_r45` and must
not be pooled with those reports. This is recorded as a blocker for the next
round rather than as passed acceptance.

## Round 44 — SE(3) horizon-8 clearance Control (2026-10-29)

Round 44 completed one fixed response fit, one **150,000-update Control-only**
training run from the immutable R40 Base, and one paired 30-scene trial (60 native
episodes). The final analysis is
`outputs/exp/20261029_160000_pickcube_horizon_clearance_r44/native_analysis_final_02.json`
(SHA-256 `dc86724e8871090101d569621628e316e85826589ea7806ea1f1d8dfdd50bffc`).
It is exploratory development evidence, not a formal performance or safety claim.

### Frozen sources, response, and precheck

The original pre-fit registration is `preregistration.json` (SHA-256
`211789597dd795db2f32e1eee4751b3a01a47a05e29770bef07139df2b9d244b`). The fixed
float64 ridge fit SHA-256 is
`f052a2b187e25fdf70ab98866049e196929562b5acce94b8c7a58ade2c45751c`. The fit
uses frozen obstacle IDs 1000–1499 only: 20,245 current-grasped carry windows
from the 400 `id % 5 != 0` episodes and 5,118 validation windows from the 100
`id % 5 == 0` episodes. It excludes 16,977 ungrasped and 3,500 incomplete-tail
windows; grasp approach is outside its coverage claim. Its 72 inputs are eight
planned native 6-D actions in physical units, current qpos9/qvel9, and the
previous-to-current TCP world pose delta. Its 48 outputs are the next eight TCP
world position deltas and world-rotation logs. No future observations, fitted
policy rollouts, new labels, or synthetic states are inputs.

The Panda source verifies root translation scale `+0.1 m` and root-aligned XYZ
rotation scale `rot_lower=-0.1 rad` with controller norm limiting. Since the
grasped carry labels contain no rotation-command variation, the frozen ridge
residual is combined with this exact known controller Euler-command rotation
map; the coefficients were not refit or tuned. This deterministic controller
map is part of the frozen response runtime and is distinct from an inference
shield. All current-response geometry uses the current finger q; actual future
finger q is used only when calculating the registered validation error.
The initial fit implementation had a positive rotation-action scale, but all
eligible carry-window rotation labels are identically zero (their fitted feature
std is the registered `1e-12` floor). The controller sign correction therefore
left the fixed coefficients bit-identical. The pre-fit registration/fit remain
byte-preserved; source-closure versions 2–8 revalidated the same fixed fit with
the signed controller map and completed Phase B/C source closure.

Source-closure versions and their fixed-model fresh prechecks are retained as
`preregistration_sourceclosed*.json` and `precheck/sourceclosed*/`. Final admitted
report SHA-256 is `011b0daf7fe1782319b713ae033d88a8e3aa64ddf8607bff5ce7c7865e14666f`
with closure SHA-256
`7c5a1f09088ac09b14d6816f2edf4ef271388a1255f206806df13c1e5bf48580`. It passes
the unchanged response criteria: native four-link point-transform max
`6.03e-8 m`; body-point p95 1.549 mm overall and 0.398–1.937 mm at horizons 1–8;
98.55% mean improvement over zero motion; TCP position p95 1.614 mm (maximum
6.046 mm). The eight forecast instants are discrete samples, not a continuous
sweep proof; the hold-carry precheck does not claim grasp-approach coverage.

### Registered Control training

Training registration SHA-256 is
`6d610e31e68fef78162c03290989f6c12b7f6e0c7baf2c6caa58835b4cc66683`. The typed
model kind is `SE3_horizon8_clearance`; R41 hint-9, 42-D position-unit global
condition, common-XY gauge seed 424243, native seven-channel actions and 100-step
DDPM sampler are retained. The objective is unweighted epsilon MSE plus
`0.01 * min(step/10000, 1)` times the sum of eight per-pose normalized body-SDF
and carried-cube XY hinge costs, averaged over current-grasped rows with eight
real future actions/states. Proxy rows are restricted to diffusion steps 10–30;
epsilon training still uses all sampled windows. The loss uses eight discrete
forecast positions, not continuous collision checking. No Base update occurred.

The ten real-batch GPU benchmark recorded 0.701 s for the cold first batch and
0.0553 s median / 0.0583 s p90 for the remaining nine. Retaining startup once
gives an 8,747 s projected p90 within the registered 15,000 s finite wall cap;
peak allocated CUDA memory was 579,302,400 bytes. Disabled-Control/Base epsilon
and DDPM sample max differences were both 0. The real minibatch gradient audit
reported XYZ norm 4.453, rotation norm 0.0294, and gripper norm 0; epsilon/body/
cube parameter-gradient norms were 0.395/3.548/119.297.

Exactly 150,000 updates completed in **11,824.64 s** on the RTX 5070. Batch 1,024,
seed 1, AdamW `1e-4`, warmup 500, cosine schedule, EMA `.75`, R40 final EMA
initialization, and R41 training profile were retained. All 150,000 loss rows
were finite. The final EMA checkpoint is
`outputs/ckpt/20261029_160000_pickcube_horizon_clearance_r44_control/final.pt`
(SHA-256 `fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769`); 10k
recovery checkpoints include model, EMA, optimizer, scheduler and RNG state. The
final checkpoint was atomically saved before post-training audits. The frozen R40
Base state remained `5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`.
Post-training EMA gradient audit at diffusion step 20 on 1,024 real eligible
windows passed: XYZ norm 0.00420, rotation norm 0.00101, gripper norm 0; epsilon/
body/cube component parameter-gradient norms 0.0332/0.3603/0.0566. No optimizer
update occurred during that audit. Final EMA planned-action abs-value fractions
over 0.35 were 0.87% XYZ-X, 0% Y, 4.39% Z, and 0% for rotation; full per-channel
label/prediction range summaries are in the final analysis. The training-log
mean x0 abs-value fraction above 0.35 was 5.94% (max 7.95%).

### One paired 30-scene native trial

Trial registration SHA-256 is
`e064ca54643606be479709cd5bcc0afa0556281ecdc0d2ef94cea09a3ee0043d`. It pins the
final EMA, R40 Base, frozen R41 and R43 reference cohorts, existing R36 30-scene
set, and DDPM-100 / slice `[1:9]` / execute-eight protocol. One evaluator
registration-source correction was required before any episode row was written;
the first setup invocation failed immediately with
`ValueError: Round 37 registered source bytes changed` before creating any
episode rows. The original runtime/engine registration and its correction record
are retained as `native_engine_registration_before_source_correction.json` and
`native_engine_registration_source_correction.json`; zero episodes were
repeated. Exactly 30 Base and 30 R44 Control rows completed with zero runtime
errors in 416.62 s. R44 Base traces and outcomes match the R43/R40 Base exactly.
Each episode row retains actual flags, action traces, scene/seed, checkpoint and
profile hashes, first-contact and terminal-phase data, and native protocol;
contact entity-pair positions remain unknown because the recorder does not
expose them. Runtime clearance-cost calls were zero.

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R40 matched Base (R43-exact) | 27/30 | 0/30 | 30/30 | 21/30 |
| R43 swept-clearance Control (primary) | 30/30 | 14/30 | 16/30 | 5/30 |
| R41 hint-9 Control (secondary) | 29/30 | 17/30 | 13/30 | 6/30 |
| R44 SE(3) horizon-8 Control | 30/30 | 17/30 | 13/30 | 7/30 |

The preregistered gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`)
**failed**: task 30 passed; safe success 17, contact 13, and XY crossing 7 failed.
Against R43 Control, R44 transitions were task both 30; safe improved 7,
regressed 4, both 10, neither 9; contact cleared 7, introduced 4, both 9,
neither 10; XY cleared 2, introduced 4, both 3, neither 21. Against R41 Control,
safe improved 4 / regressed 4 / both 13 / neither 9; contact cleared 4 /
introduced 4 / both 9 / neither 13; XY cleared 1 / introduced 2 / both 5 /
neither 22. This is a combined training-objective/horizon/carry-mask/SE(3)
response proposal comparison, not an isolated lambda effect. No further trial or
checkpoint selection followed the failed gate.

The matched 1,000 direct / 500 obstacle demonstration source remained unchanged
(HDF5 `0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`); R40 Base
remained unchanged. Full retained rows are under `paired_native_r44/`; terminal
and first-contact counts are distinct in the report. Ticket 05 has no local
`.scratch/` issue file in this checkout.

Verification: full constrain suite **605 passed, 28 warnings** in 86.47 s;
frozen data and Base hashes match before/after. No new episodes, replay, demos,
layouts, or training cohorts were created. All code changes are confined to
`examples/baselines/constrain`; no agents were used and no commit/branch/push was
performed.

Commit-reference correction: the `Ref` footer of implementation commit
`0df133ed88954203002d391c1bb0d8747b9dd2cc` mistakenly names
`05-pickcube-clearance-round44.md`. The originating issue is the historical
`.scratch/transport-detour-optimization/issues/05-pickcube-learning-pilot.md`;
no separate Round 44 ticket was created. The original commit is retained.

## Round 43 — swept clearance-proxy Control trial (2026-10-29)

The independently registered R43 frozen-data precheck is in
`outputs/exp/20261029_052324_pickcube_clearance_r43/`; model weights are in
`outputs/ckpt/20261029_052324_pickcube_clearance_r43_constrain/`. R43 retained
all three immutable precheck stages. The final source registration is
`preregistration_final_02.json` (identity
`db246e9d4ce56a53a1e2a4cc1aa8b91f70e4befbb800adb99b61934565d5d1ee`) and its
fresh final-source report is `precheck_final_02/geometry_response_precheck.json`
(SHA-256 `0feb6120d69b4a6c857e5b50a238fb6f32eded5acc4bb6ab526993bd8f556579`).
The first R43 preregistration/precheck and the R42 false-positive report plus
both R42 correction records remain byte-preserved.

The fixed-frame point validation now applies `T_tcp_link` in the forward
direction. Across 36 saved-state native FK queries, the maximum point error was
`6.03e-8 m` (hand `3.16e-8`, link7 `6.03e-8`, left finger `3.71e-8`, right
finger `2.19e-8`). Finger links were identified as prismatic joints and their
TCP-relative translation axes calibrated from two distinct existing qpos states;
rotation spread and translation residuals on the validation states were below
`1.4e-6 m/rad` (recorded individually in the report). The registered 5 mm
sample-sphere padding was applied separately to the analytic finite-cylinder
SDF; the unpadded physical-center SDF is separately reported. All four links
use deterministic samples capped at 256 per link. The carried-cube forbidden
radius uses the native `collision_radius + sqrt(2)*cube_half_size + safety_margin`
formula, and its XY-only current-state component was measured on 29,363 grasped
frames without a height waiver. Full mesh/URDF provenance, deterministic
sampling indices, dynamic finger q references/axes, finite-caps SDF quantiles,
and fit/validation cohort metrics are in the report.

The unchanged R42 21-feature ridge response artifact retained the same 96.60%
mean translation improvement and 0.682 mm TCP p95. On held-out trajectories,
translation-only predicted next-state body points (current TCP orientation and
finger q held fixed) had 0.568 mm mean / 1.542 mm p95 error, with 92.79% mean
improvement over zero motion. These are sampled geometry-proxy results, not
PhysX collision labels or a runtime safety guarantee. The frozen HDF5, manifest,
R40 Base, Panda source/assets and R42 response-fit records were read-only.

### One guarded Control training

The typed training admission is `train_preregistration.json` (identity
`49b8849d75ec97552ae3f6a03b0907de30c509796770e10c04796204bfe6f2ce`). It pins
the final precheck report, source/code/geometry/data hashes, R40 Base, normalizer,
R42 response fit, ten-real-batch benchmark, and the sole objective:
`epsilon MSE + 0.01 * min(step/10000, 1) * (swept_body + swept_cube_xy)`.
The proxy uses timesteps 10–30, scheduler-clipped `x0`, action index 1, four
sweep fractions `[.25,.5,.75,1]`, 5 mm point padding, 2 cm body margin and 1 cm
cube XY margin. It is training-only; actor inference has no clearance cost or
shield call. Before training, disabled Control matched the independent frozen
R40 Base exactly (epsilon and DDPM-100 sample max absolute difference both 0).
The ten-batch benchmark projected 9,909 s p90, within the registered 15,000 s
budget, with 550,867,968 bytes peak allocated.

Exactly 150,000 Control updates ran in 5,811.79 s on the RTX 5070. Batch size
1,024, seed 1, AdamW `1e-4`, warmup 500, cosine schedule, EMA power `.75`, R40
common-XY gauge seed 424243, R41 position-unit normalizer and raw hint-9 were
retained. No Base gradients/updates or new physical data were used. All logged
epsilon/clearance values and gradients were finite; proxy active rows and body/
cube component costs were recorded per step. The frozen R40 EMA state hash
remained `5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`.
The final EMA checkpoint SHA-256 is
`da818a9267a19a6ab8369a2d6b55b3e2116a28514c72662087607df635e53d48`; ten
thousand-step atomic recovery checkpoints and a step-0 strict metadata/EMA/
optimizer/scheduler export round-trip fixture are retained.

### One paired 30-scene native trial

Trial registration `native_trial_preregistration.json` (identity
`b6b5376d4a84d3b80685ca41c678ace52b530f767f4a16851fc890899eb7f09a`) pins the
final EMA, training/precheck registrations, unchanged R40 Base, R41 primary
tree, existing 30 R36 restored scenes and the exact DDPM-100/native-action
protocol. The pretrial real R41 row context check and disabled-Control/Base
equivalence passed. Exactly 30 rows per policy completed in 404.14 s with zero
runtime errors. Base action traces and outcomes match R41 **exactly** scene by
scene. All paired restored states and normalizer conditions matched; no rollout
was repeated. Raw rows are under `paired_native_r43/`; offline immutable-row
analysis and hash snapshot are `native_analysis_final.json` and
`native_analysis_inputs.json`. The native evaluator's summary-only report
assembly record is preserved separately; final analysis was produced from the
retained 60 rows, without rerunning native evaluation. The source-only report
repair chronology is `native_report_assembly_correction.json`.

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R40 matched Base (R41-exact) | 27/30 | 0/30 | 30/30 | 21/30 |
| R43 clearance Control | 30/30 | 14/30 | 16/30 | 5/30 |
| R41 Control primary | 29/30 | 17/30 | 13/30 | 6/30 |

R41 Control → R43 Control transitions (old/new-only, both, neither): task
`improved 1, regressed 0, both 29, neither 0`; safe `improved 4, regressed 7,
both 10, neither 9`; contact `cleared 4, introduced 7, both 9, neither 10`;
XY `cleared 4, introduced 3, both 2, neither 21`. Against the matched Base,
R43 cleared 14 contacts and 19 XY crossings, introduced 3 XY crossings, and
improved task success in 3 scenes.

The frozen gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`) **failed**:
task 30 passed; safe 14, contact 16 and XY crossing 5 failed. The sole trial is
complete exploratory evidence, not a formal performance claim or safety
guarantee. No further objective/training/native trial was run.

## Round 42 — preregistered training-only clearance proxy precheck (2026-10-29)

R42 **stopped before training and native development evaluation**. The one
precheck's final fail-closed analysis found that the fixed hand/link-7 frames
matched native FK, but the sampled point-cloud transform validation did not
meet tolerance, declared point padding was not applied, registered finger samples
were omitted, and the carried-cube forbidden-disk component was not implemented.
Therefore the surrogate was not credible as a complete
training objective, regardless of the response fit's strong held-out numeric
score. No λ tuning, geometry/dynamics refit, retry, checkpoint, policy action,
development label, simulator episode, or new demonstration was used.

### Frozen protocol and preserved attempt

The R42 output is
`outputs/exp/20261029_050700_pickcube_clearance_r42/`. Registration
`preregistration.json` was frozen before geometry/dynamics analysis (identity
`e1f843e385ed2975136ca15e1f0cedffeab24e7593c46add53f36129d9839f47`). The
final implementation source was pinned in the pre-analysis correction record
(`precheck_source_correction.json`, next registration identity
`6314040c6d53672d3788a155451c3ff7a3a49078f16b2c7d14f041875dcb89e3`); the
initial registration is retained unchanged. The single recorded precheck is
`precheck/geometry_response_precheck.json` (SHA-256
`804012ccf11b5e5e2ed63ff8fa70344150fa2d67d3a8c7befdb90cbdd5098eb2`). The
analysis-only invalidations are `precheck/precheck_correction.json` and
`precheck/precheck_correction_02.json`; the latter supersedes the first and
records the padding/finger audit. Both reference the first result by hash and
record no additional precheck attempt. Report-source deltas are independently recorded in
`analysis_source_correction*.json`.

Inputs were the exact R41 frozen 500 source: HDF5 SHA-256
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, and
validation `f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`.
The source has explicit per-trajectory cylinder metadata for all 500 obstacle
IDs (center, radius 0.03 m, and measured height; range 0.10698–0.16833 m).
The frozen initializer was the unchanged R40 Base final EMA
`9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719`; R41
Control was not resumed. No direct or dev states/actions were used to fit.

### Measured response and proxy evidence

The registered 21-feature, centered float64 ridge fit (`λ=1e-6`) used 36,697
frames from IDs `1000..1499 where id % 5 != 0`; the held-out-by-trajectory
validation used 9,143 frames from `id % 5 == 0`. Current-frame action index 1
alignment and target next-recorded TCP pose were used. Validation position error
was mean **0.0002623 m**, p95 **0.0006817 m**, maximum **0.0041635 m**; mean
position error improved **96.60%** over the preregistered zero-displacement
baseline. Angular p95 was **0.0009019 rad**. This meets the response-fit numeric
thresholds but is only one-step recorded-action prediction, not rollout or
collision-prediction accuracy.

The 36-state read-only FK check found fixed TCP→hand and TCP→link7 relation
maximum absolute errors `2.3842e-7` and `3.5763e-7`, respectively. However, the
recorded point-cloud/native transform comparison was **0.424431 m**, far above
the `1e-5 m` acceptance tolerance; its saved-TCP comparison field was conflated
with that point-cloud maximum, so the surface-point transform validation is
invalid. The original report also recorded 130,557 negative sampled point-SDF
values among 23,470,080 points; because the transforms failed and samples are
not exact surfaces, these cannot be interpreted as robot collisions or used to
assert the safe demonstrations are geometrically invalid. The proxy geometry
was limited to sampled URDF collision-mesh points from hand and link7. Although
the preregistration declared 5 mm point-sphere padding, the precheck did not
apply it; this is another recorded blocker. Registered left/right finger
collision samples were also absent. It was not a full Panda collision model and
never represented PhysX collision ground truth.

The registration additionally specified a carried-cube XY forbidden-disk
constraint, active only when the current raw grasp flag is true, using the
native PickCube radius formula. That component was **not implemented or
validated** in this one attempt. This is an independent hard failure. The
recorded attempted point cloud/hash/mesh sources and response coefficients are
retained under `precheck/`; they are not approved training artifacts.

### Decision and scope

The corrected result is `status: failed`, `training_permitted: false`, with
failure reasons: point-cloud transform validation above tolerance, missing
carried-cube proxy, declared sample padding not applied, and omitted finger
samples. As required, the 150,000-update R42 training, R41-equivalent
30-scene paired native trial, and gate were **not run**. The proposed
`epsilon MSE + λ_eff clearance` objective (λ final 0.01, warmup 10,000) was
never registered as a training phase or applied. Inference behavior is
unchanged. There are no R42 model weights or outcome counts. The preregistered
precheck attempt limit was one; no second model/geometry fit was attempted.
Frozen HDF5, manifest, validation, R40 Base, R41 Control, ManiSkill/Panda source,
PlaceSphere source, and existing demos remain unchanged.

## Round 41 — fixed Panda body-frame PickCube hint (2026-10-29)

R41 completed one 150,000-update Control-only training run on the frozen 500
obstacle demonstrations and one paired 30-scene development trial (60 native
episodes total). It reuses the exact R40 gauge-trained Base and its position-unit
profile. The sole treatment is the Control hint representation: append fixed
Panda hand and link-7 point coordinates to the unchanged three-value
obstacle-relative-to-TCP hint. This is exploratory development evidence, not a
formal performance or generalization claim. No demonstrations, derived HDF5,
episode labels, layouts, replay materializations, or ManiSkill/PlaceSphere source
changes were made.

### Artifacts, source chronology, and frozen inputs

The experiment is
`outputs/exp/20261029_031500_pickcube_bodyhint_r41/`; the Control checkpoint and
10,000-update recoveries are in
`outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_constrain/`. The reused Base
is `outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt`, SHA-256
`9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719`. R41
registration SHA-256 is
`45f6f9e0a11c4c414c339b942d31fe47694bb4c453fbfd3ce2c71fd3393b832e`; its
pre-training correction history and byte-preserved initial registration are
under the same experiment directory. A post-training strict-loader correction
is separately recorded there; it changed no weights and occurred before the
native trial.

The unchanged physical source is
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5 SHA-256
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, validation
`f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`. Training
still selects direct IDs 0–999 (83,004 windows) for the retained Base identity
and obstacle IDs 1000–1499 (45,840 windows) for Control. No Base retraining was
performed. All three physical-data hashes matched after training and native
evaluation.

### Fixed-frame calibration and hint contract

Calibration used only existing HDF5 T+1 raw 45-D observations and their saved
qpos, with read-only native forward kinematics. Nine existing trajectories were
sampled at four frames each (36 states); there were no `env.step` calls, new
scene states, data-fitted transforms, or runtime FK/environment queries. The
URDF is `mani_skill/assets/robots/panda/panda_v2.urdf`, SHA-256
`524c6986dbbabe244e05801f262862205e880834d01459b06ee273016d0579ba`; robot
model source `mani_skill/agents/robots/panda/panda.py`, SHA-256
`bf85577be75b7513802e676decb5543e3e65f2a3dfbc26175dd8b77ae57691f3`.
`panda_hand_tcp` quaternion order in the raw observation is `wxyz`.

| Fixed transform | Matrix (meters, row-major) | Native-FK validation |
| --- | --- | --- |
| `T_tcp_hand` | `[[1,0,0,0],[0,1,0,0],[0,0,1,-0.1034],[0,0,0,1]]` | relation spread `2.38e-7`; constant/native max `2.29e-8` |
| `T_tcp_link7` | `[[.70710678,-.70710678,0,0],[.70710678,.70710678,0,0],[0,0,1,-.2104],[0,0,0,1]]` | relation spread `3.58e-7`; constant/native max `1.19e-7` |

The maximum saved TCP pose versus restored-qpos FK difference was `1.83e-7`
(tolerance `1e-5`). Full details are in `frame_calibration.json` and in the
registered `body_hint` metadata.

Hint-9 order is:

1. Original `[obstacle_x-tcp_x, obstacle_y-tcp_y, radius]` (unchanged raw
   meter-valued calculation, scales `[0.5, 0.5, 0.08]`, clip `[-1,1]`).
2. `[obstacle_x-hand_x, obstacle_y-hand_y, hand_world_z]`.
3. `[obstacle_x-link7_x, obstacle_y-link7_y, link7_world_z]`.

The added six values are world-axis meters divided by `0.5` and clipped to
`[-1,1]`. Z is a derived world-height feature; it is **not** obstacle surface
clearance or a crossing oracle. Every feature uses only the latest raw 45-D
observation and fixed profile constants at runtime. No additional observation,
privileged pose/state query, cylinder-height feature, action filter, cost,
shield, or grip heuristic is used. The raw observation stays 45-D and the global
state encoder remains 42-D / 84-D history.

The R40 training-only common XY-origin augmentation and dedicated seed `424243`
are unchanged. It applies to the global condition; R41's relative/body hint is
computed from the original raw observation. A scan of all 45,840 obstacle
windows found finite, nondegenerate hint values; common-XY gauge invariance max
error was `5.96e-8`. All profile/frame metadata is standalone in the checkpoint,
and the typed format `pickcube_body_hint9_constrain_v1` strictly rejects legacy
three-channel checkpoints. The original default 3-channel PickCube condition
and PlaceSphere profiles remain unchanged.

### Control-only training

R41 trained exactly once for 150,000 updates from the frozen R40 Base final EMA;
the Base was bit-identical before and after training (embedded state hash
`5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`). The sole
model dimension change is `cond_dim: 3 -> 9` in the Control hint encoder; global
condition `84`, actions `7`, and copied down/mid channels `[64,128,256]` remain
fixed. Training remains native identity `pd_ee_delta_pose`, unweighted epsilon
DDPM-100 loss, batch 1,024, seed 1, AdamW `1e-4`, warmup 500, cosine schedule,
EMA power 0.75, and final EMA selected before training. There is no new action
clamp, codec, decoder, or runtime augmentation.

| Evidence | Result |
| --- | --- |
| Updates / wall budget | 150,000 / 9,000 s; completed in 4,413.55 s |
| Loss | first `0.03867678`, final `0.00173361`; all finite |
| Augmentation draws | 153,600,000 window draws (in-memory calls, not episodes) |
| Frozen Base | exact R40 EMA; unchanged |
| Control learning | copied encoder and zero-residual weights moved; EMA strict load passed |
| Final EMA | `outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_constrain/final.pt`, SHA-256 `838745ceb9437fa9db9c5c289a88a032cdf8a12dca42145e896d3c7ac3529642` |
| GPU | RTX 5070; peak allocated `520,562,688` bytes |

Atomic complete-state recovery checkpoints exist every 10,000 updates. Logs,
run config, training summary, augmentation statistics, phase argv/budget/exit,
and checkpoint records are under `train_constrain/` and the phase runtime JSON.

### One paired native trial and result

The one actual native trial used the same 30 restored R36 development scenes,
identical per-policy scene seeds, 500-control-step cap, 300-second per-episode
timeout, and a 36,000-second outer budget. Sampling was DDPM-100, epsilon,
`squaredcos_cap_v2`, clipping enabled, predicted slice `[1:9]`, execute/replan 8,
native seven-channel actions, with inference augmentation off. Base and Control
conditions pair across all scenes; the R41 Base's native action traces and
outcomes match R40 **exactly**. All 30 rows per policy completed with zero
runtime errors.

| Policy | Task success | Safe success | Obstacle contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R40 matched Base (unchanged checkpoint) | 27/30 | 0/30 | 30/30 | 21/30 |
| R40 Control primary | 29/30 | 15/30 | 15/30 | 5/30 |
| R41 Control hint-9 | 29/30 | 17/30 | 13/30 | 6/30 |

R40 Control → R41 Control transitions (both / neither; improved-or-cleared old
only / regressed-or-introduced new only) were: task `28 / 0; improved 1,
regressed 1`; safe `12 / 10; improved 5, regressed 3`; contact `10 / 12;
cleared 5, introduced 3`; XY crossing `4 / 23; cleared 1, introduced 2`.
All transition marginals reconcile to the 30-row cohorts. R41 Control first
contacts were transport 7, goal stabilization 4, grasp approach 1, lift 1;
contact links were Panda hand 10, cube 1, right finger 2. Contact entity-pair
positions are unknown because the recorder does not expose them. Terminal
phases were 29 task successes and one lift-no-transport. First-success steps
min/median/p90/max were `58/87/134/324`.

The frozen gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`) **failed**:
task `29` passed, safe `17`, contact `13`, and XY crossing `6` failed. This
single hint representation trial modestly improved safe success/contact against
R40 while leaving task success unchanged and increasing XY crossings by one.
It does not meet the gate and does not warrant a hint-grid sweep. The next
engineering slice should address the retained transport/goal-stabilization
contact mechanism using complete contact-phase evidence.

The first evaluator invocation failed in adapter construction before any
environment rollout (zero rows); its preregistration/runtime record is preserved.
The second invocation completed exactly one 60-episode trial in 413.34 s and
saved all rows. A historical report assembly error (`summarize_pilot` referenced
on the training module) happened only after the rows were complete. The report
was rebuilt offline from those immutable rows; `analysis_report_inputs.json`
pins row hashes and the R40 comparison tree, and `paired/status.json` records
`native_evaluation_rerun: false`. The analysis input snapshot SHA-256 is
`4c25c6a4de2a7886f48908228e75bee8c357d046f7ce34703a60367ca02a8ba3`; final
`paired/analysis.json` file SHA-256 is
`65c263a4387b1383e35c317c452505118ebafef4846953f699b977eebf5370c2`.
Complete episode action/flag traces, raw initial hint-9 values/hashes, checkpoint
and source identity, per-episode wall times, phase/contact links, and final
cube/goal XYZ are retained under `paired/base_dp/` and `paired/constrain_dp/`.

Verification: full constrain suite **586 passed, 28 warnings**; focused R41
feature/profile tests **6 passed**. The calibration, frozen-window preflight,
real saved-R40-row aggregation fixture, strict checkpoint load, and disabled
Control action equivalence all passed. Frozen source hashes remained unchanged;
ManiSkill/Panda and PlaceSphere files were read-only.

## Round 40 — matched PickCube global-XY gauge augmentation (2026-10-29)

R40 completed one matched Base/Control retraining and one paired 30-scene native
development trial (60 episodes total). It is exploratory development evidence,
not a formal performance or generalization claim. The single treatment is a
**training-distribution change**: each training window sees one uniformly
sampled XY coordinate-origin translation of the same physical episode. It is
not a claim that the training distribution or training examples were unchanged.
There are still exactly 1,000 direct and 500 obstacle physical demonstrations;
no demonstrations, HDF5 variants, labels, layouts, task geometries, or replay
materializations were created. The R40 augmentation draw counts below describe
in-memory training calls, not additional data episodes.

### Frozen source and preregistration chronology

The R40 experiment lives at
`outputs/exp/20261029_004500_pickcube_xy_gauge_r40/`; checkpoints are in
`outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_{base,constrain}/`.
The initial registration was frozen before Base training, with registration
identity `b7ea9f8d80049f8fbca71ac920d8c0c34548f088fc50f91e0b22073d9c5befb6`
and source aggregate
`82b54a6650fff1f2540c79bb777b2b02ab7aa2a76a1eb24521238ec3e9ef83af`.
Base's phase preregistration identity is
`49e14ee3f05be3a5280fd3636f75ea39198f419d3ef64bbb0666bd0f440ff53c`.

The Control preflight found a missing import for the explicit original-raw-hint
invariance check before Control training began. That precise source correction
is recorded in `post_base_source_correction.json`; the original Base registration
and source identity are byte-preserved at
`preregistration_base_training_frozen.json`. The R40 Base checkpoint was not
changed or retrained. The new Control registration identity is
`eafac1a12fa6fcae4641787d65bc7355bded3fc65857fb3353a48cfb493e1c35`, and its
phase preregistration identity is
`b37fb7743c1340f2e9258a2130e10f57fa90fb5900d0a32854d1d35a6a01e02d`.

After both training phases, a report-adapter source correction was recorded
before any native rollout in `post_training_analysis_correction.json`; the
Control-training registration is preserved at
`preregistration_control_training_frozen.json`. Neither model was retrained.
The native evaluation preregistration identity is
`4ef5212bb7d235a37fdeabd18fdfa91c1fe04471b0e0098254ac0fb88acd468e`, using
source aggregate
`0fdd4b7b6dc0927a59e4a82bfe490b58878e43e69c10c28575106f74cc2f79cb`.
The actual frozen native registration is
`3168e00fbdfde80bc96471203ab68ebc66a8833774e7bd1f118c65b7cfb64f0e`.

The immutable training source is
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5
SHA-256 `0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`,
manifest `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`,
validation `f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`.
Training uses direct IDs 0–999 (83,004 windows) and obstacle IDs 1000–1499
(45,840 windows); the same 333-task/500-obstacle cohort is unchanged. All three
hashes matched after both trainings and native evaluation. PickCube source stayed
at SHA-256 `53bfed0229cd4de2c7e363e7f3c531d5c5c84803c28e1345a2aea1a4972f3168`;
ManiSkill task/controller code and all PlaceSphere files were untouched.

### Fixed coordinate-origin transform and invariants

The fixed profile is `global_xy_gauge_v1`, with one always-on training draw
`delta_xy ~ Uniform([-0.02,0.02]^2)` meters per batch window. In raw observations,
the same delta is added to TCP world XY, goal world XY, and object world XY; for
raw 45-D Control observations, it is also added to obstacle-center XY. Indices
are derived from `NAMED_FIELDS`, not unsupported offsets. The delta is shared
across both history frames and all world-position fields in that window, and is
independent across windows. It is applied to a cloned minibatch before the fixed
R38 position-unit encoder.

This represents translating the coordinate origin of the entire scene,
including its implicit robot-base/table frame; it does not move an object or
cylinder relative to the robot. Z, joint positions/velocities, grasp flags,
quaternions, all relative vectors, cylinder radius, and all seven native action
labels remain exactly unchanged. Therefore TCP-to-object and object-to-goal
vectors, object-to-cylinder XY displacement, and the cylinder disk-clearance
margin are invariant. No axes rotation, Z shift, per-frame/independent
target/obstacle shift, label perturbation, new geometry, input feature, or
inference augmentation is used. Model inputs retain the 42-D R38 position-unit
condition and Control's same three-value relative hint; the latter is computed
from the original unshifted raw 45-D frame for bit consistency. Real-data
preflight measured zero hint change and zero disabled-Control epsilon difference.

The hypothesis is exploratory: a small common-origin randomization may reduce
absolute global-position memorization and improve conditional generalization
across the 500 demonstrations and 333 task variants. R38/R39 do not causally
establish that mechanism. The model has no explicit base-pose observation;
unchanged joint coordinates plus shifted TCP coordinates make this a deliberately
scoped coordinate-gauge hypothesis, not a sensor-noise or physical robot-motion
augmentation.

The dedicated RNG is `torch.Generator(device="cpu")`, seed
`training_seed + 424242` (424243 for this training seed). Once per batch it draws
`torch.rand((B,2), dtype=torch.float32, generator=cpu_generator)`, consumed in
row-major order `[window0-x, window0-y, window1-x, window1-y, ...]`, then maps
each value by `(2u-1)*0.02`. Its state is saved/restored with every recovery
checkpoint, independently of model initialization, diffusion noise/timestep,
sampler, and DataLoader RNG. The exact algorithm is repeated in both checkpoint
metadata blocks. Inference is unshifted in the original world frame.

### Matched training and checkpoints

The unchanged pipeline is Base direct-only for 30,000 updates, then Control
obstacle-only for 150,000 updates initialized from this new Base final EMA, with
all Base parameters frozen. Both keep batch 1,024, seed 1, AdamW `1e-4`, betas
`(0.95,0.999)`, weight decay `1e-6`, warmup 500, cosine LR, EMA power 0.75,
unweighted native epsilon MSE, DDPM-100 squared-cosine epsilon prediction, and
the R38 model/profile/action dimensions. Actions remain native identity
`pd_ee_delta_pose`; no new clamp, codec, shield, or phase weighting was added.

| Policy | Updates | Wall time | First → final loss | Final EMA SHA-256 |
| --- | ---: | ---: | ---: | --- |
| R40 Base | 30,000 | 1,061.30 s | 1.17822170 → 0.00282031 | `9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719` |
| R40 Control | 150,000 | 4,363.34 s | 0.03726619 → 0.00181645 | `2215f9f7d8bbec933bec53b2880758206af304a171297c5369ad85dcadfe1374` |

Both phase summaries report finite losses and complete 10,000-update recovery
checkpoints (3 Base, 15 Control). Loss logs have exactly 30,000 and 150,000
rows. Both checkpoint metadata objects carry exactly the same registered
augmentation profile and dedicated RNG state. Control's frozen-Base tensor hash
matches the R40 Base EMA (`5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`).
The selected weights are the fixed final EMA, not a searched seed/step/δ setting.

Saved aggregate δ statistics (one entry per window draw, **not data quantity**):

| Policy | Draws | X min / mean / max (m) | Y min / mean / max (m) |
| --- | ---: | --- | --- |
| Base | 30,720,000 | -0.02000000 / -7.38e-8 / 0.01999999 | -0.02000000 / 3.30e-7 / 0.02000000 |
| Control | 153,600,000 | -0.02000000 / -6.80e-8 / 0.02000000 | -0.02000000 / 1.48e-6 / 0.02000000 |

Recovery state and summaries are under the R40 `outputs/ckpt/` and
`outputs/exp/` paths above. Training budgets were 3,000 s Base and 9,000 s
Control; both finished within budget. The first Base invocation exited before
any updates due only to a wrapper-created nonempty output directory (exit 1,
zero updates); its exact invocation record is preserved as
`phase_runtime_base_training_attempt01.json`. The corrected 30k launch used the
same preregistered settings and completed once.

### One paired native trial and primary comparison

Both EMA loaders strictly checked the same R38 typed position normalizer plus
registered augmentation metadata before evaluation. An independently seeded
Base-vs-disabled-Control native action preflight matched exactly (max absolute
difference 0.0). Runtime augmentation was off. The archived DDPM sampler stayed
at 100 steps, epsilon, squared-cosine, clipping enabled, predicted slice
`[1:9]`, execute/replan 8, seven native action channels, with no extra action
clamp or grip shield. The trial used the same 30 R36 development scene states,
controller/raw-observation hashes, position-encoded condition hashes and
per-scene seeds as R38, 500 control steps, 300 s per-episode limit and 36,000 s
overall limit. Exactly 30/30 Base and 30/30 Control rows are retained; all were
evaluated with zero runtime errors. Episode wall times were Base
4.163/6.995/15.322/17.702 s and Control 4.445/5.599/9.702/24.976 s
(min/median/p90/max).

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| Primary R38 Base | 22/30 | 0/30 | 30/30 | 23/30 |
| R40 Base | 27/30 | 0/30 | 30/30 | 21/30 |
| Primary R38 Control | 29/30 | 13/30 | 16/30 | 6/30 |
| R40 gauge-augmented Control | 29/30 | 15/30 | 15/30 | 5/30 |
| R39 guidance, scale 1.5 (negative result, excluded) | 29/30 | 11/30 | 17/30 | 6/30 |

R38 → R40 Control paired transitions (both / neither; positive outcomes
improved-new-only / regressed-old-only; violations cleared-old-only /
introduced-new-only): task `28 / 0; 1 / 1`; safe `9 / 11; 6 / 4`; contact
`10 / 9; 6 / 5`; XY crossing `4 / 23; 2 / 1`. Each integer table reconciles
to both 30-row marginals. The Base transitions were task `20 both, 7 improved,
2 regressed, 1 neither`; safe `0 both, 30 neither`; contact `30 both`; XY
`20 both, 3 cleared, 1 introduced, 6 neither`. The per-scene R38 tree was
preregistered and hashed before native execution; its aggregate tree hash is
`208f69a4dc85b4f1b7d96ee7fc9d74ad606d1e82c60c1ee84463f0adfbe2ea4f`.

R40 Control first-contact phases were transport 11, goal stabilization 2,
grasp approach 1, lift 1; link counts were panda hand 10, cube 1, link7 1,
right finger 3. Contact entity-pair positions are unknown because the native
recorder does not expose them. The Control terminal phases were 29 task success
and 1 lift-no-transport; first-success steps were min/median/p90/max
74/98/164/237. Final 3-D goal-distance min/median/p90/max was
0.00465/0.00824/0.01553/0.56811 m. All per-episode wall time, final cube/goal
XYZ, terminal flags, contact/stage traces, and raw rows remain in
`paired/constrain_dp/` and `paired/base_dp/`.

The preregistered gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`)
**failed**: task passed at 29, safe was 15, contact 15, and XY crossing 5.
The paired development result is exploratory and does not isolate an
augmentation-only causal effect because both Base and Control were retrained.
No extra trial was launched after the failed gate.

The native runner completed and retained all 60 rows, then its historical report
assembly raised `KeyError: 'phase_counts'` after those rows were saved (the
summary object did not contain that legacy key). The first native command exit
status is retained in `paired/phase_runtime.json`; no episode was repeated.
`analysis_report_correction.json` records the analysis-only source change,
`analysis_report_inputs.json` freezes all 60 row hashes and the R38 result-tree
hash, and `paired/analysis.json` / `paired/status.json` were rebuilt offline.
`native_evaluation_rerun` is false, and the repaired report now validates exact
row/marginal counts, phase counts, terminal flags and R38 identity. Its analysis
input snapshot SHA-256 is
`57d21f04882832558d07619a4485b73df55329237721bd6f86d28be03d08f675` and final
analysis SHA-256 is
`9200e31106bb13e9d4666d048fe14e250628cdb36f915f00f14979bc2ab38bb4`.

Verification: full constrain suite **580 passed, 28 warnings** in 350.73 s;
focused gauge/report tests **5 passed**. The final R40 evaluator/report code is
under `examples/baselines/constrain/pickcube_xy_gauge_round40.py`; the reusable
named-field transform/RNG is `pickcube_xy_gauge.py`, matched trainer changes are
in `pickcube_zscore_round37.py`, and invariant/strict-metadata/report-contract
tests are in `tests/test_pickcube_xy_gauge.py`. All native rows and checkpoint
hashes are retained. No extra training, replay, evaluation scene, or process
remains.

## Round 39 — one-scale learned residual guidance trial (2026-10-29)

Round 39 performed **one** inference-only native trial using the exact R38 EMA
checkpoints, fixed position/controller-unit state profile, frozen dataset and
existing 30 R36 development scenes. No demonstrations/layouts were collected,
no model was trained, no checkpoint was written, and ManiSkill/PlaceSphere
sources were untouched. The experiment is exploratory and makes no formal
performance or guaranteed-safety claim.

### Preregistration and runtime operation

Registration:
`outputs/exp/20261029_000001_pickcube_residual_guidance_r39/preregistration.json`
(file SHA-256
`c0d56a00f24a85ccbf6283b815157ab48f31bcfa587af34388a2ec883808bdb9`, internal
registration identity `2f2b133098cf48db685fde3e19c2c22e626f1f5e4d388ce58c05a5260136173d`).
It was created before native execution and pins source aggregate
`654ae12d5c9868dd429e8669e1eab9043f36fdbd960d86658acb15af3c33e7f8`, exact
Base/Control final EMA SHA-256 values from R38 (`6a7c1af0…3d9869` /
`1c3276e3…63bf42`), the position profile file SHA-256
`d14d56e3…23f543` and profile identity
`8bcb1df3…b3d7fef`, frozen HDF5/manifest/validation identities
(`0c74b88e…349f4f` / `6b9d2c75…e5c486` / `f4756f69…d5b8f`), scene manifest/HDF5,
and per-file hashes for immutable snapshots of every R38 Base and Control row.
The R38 report had identified an analysis-only report assembly mismatch; its
retained rows, not a rerun, were snapshotted and used here.

The sole treatment is epsilon-space
`eps_base + 1.5 * (eps_control - eps_base)`: both outputs use the same current
sample, timestep, encoded 84-D global condition and raw-observation-derived
hint; `eps_base` calls the frozen Base branch (`enabled=False`) of the same
Control checkpoint and `eps_control` calls its learned enabled branch. This is
**learned conditional residual guidance**, not classifier-free guidance (there
is no unconditional branch trained for this use). Base is unchanged and has
guidance N/A. Runtime scale is not part of or written into either checkpoint.
No action/pose/grip correction, data feedback, or geometry shield is applied.

DDPM remained 100/100 steps with the squared-cosine beta schedule, epsilon
prediction and original clipping; predicted slice `[1:9]`, native 7-D identity
actions, execute/replan 8. Per-scene seed reset and one Gaussian latent draw per
decision are unchanged; the extra deterministic Base forward consumes no RNG.
An offline scale-1 check on the first retained R38 Control scene reproduces its
saved first native action bit-for-bit (max absolute difference 0.0), confirming
the default runtime fast path preserves the historical action output.
The fixed limits were 600 s per episode per policy and 36,000 s overall. Runtime
sources and all frozen input/checkpoint hashes matched after the trial.

### Native result and paired analysis

Exactly 30 existing scenes per policy completed (60 total), with 0 runtime
errors, 0 wall timeouts, and total elapsed time 602.26 s on an RTX 5070. Base
reproduced R38 exactly scene-by-scene, including native step traces/outcomes;
the guided comparison is therefore against a validated unchanged Base. Counts
below use denominator 30 in every case:

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R38 Base / R39 Base | 22 | 0 | 30 | 23 |
| R38 Control | 29 | 13 | 16 | 6 |
| R39 learned residual guidance, scale 1.5 | 29 | 11 | 17 | 6 |

R38 Control → R39 guided paired transitions (both / neither; old-only / new-only)
were task `28 / 0; regressed 1 / improved 1`, safe `9 / 15; regressed 4 /
improved 2`, contact `14 / 11; cleared 2 / introduced 3`, and XY `3 / 21;
cleared 3 / introduced 3`. The exact marginals reconcile to each transition
table. Against R39 Base, guidance transitions were task `21 both, 8 improved,
1 regressed`; safe `11 improved`; contact `13 cleared, 17 both`; and XY `17
cleared, 6 both`.

Guided first-contact phases: transport 11, goal stabilization 4, grasp approach
1, lift 1; first-contact link counts: hand 13, cube 1, left finger 1, link7 1,
right finger 1. The native recorder does not retain contact entity-pair
positions, so these remain unknown. First-success step min/median/p90/max was
78 / 107 / 218 / 334. Per-episode wall time min/median/p90/max was 5.952 /
8.241 / 20.266 / 35.163 s for guided Control and 3.396 / 7.555 / 15.605 /
15.782 s for Base. Guided final 3-D goal-error min/median/p90/max was 0.00332 /
0.00895 / 0.01781 / 0.09949 m.

The preregistered gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`) **did
not pass**: task passed; safe, contact and XY failed. Scale 1.5 is rejected for
this trial; it does not justify trying another guidance scale. The result is
consistent with a small safety regression versus R38 Control and does not
establish route-preserving control.

Retained native rows and summaries are in `paired/base_dp/` and
`paired/guided_control/`. The report-only analysis was enriched from these rows
without rerunning the environment. Final `paired/analysis.json` SHA-256 is
`80cba3052d92dbbd3cb1632af06061076069b19ae08933d0b5ed53fcf482c0e3`; the
pre-enrichment native report is preserved byte-for-byte as
`paired/native_analysis_before_report_enrichment.json` (SHA-256
`0f3ea358326897a5b89a9302f752d6df3e560ed41981610d65366d42ec62573d`) and the
report-only chronology is in `paired/analysis_report_fix.json`. Post-run audit
confirms frozen data, both checkpoints, profile and scene files all retain
their preregistered hashes. No policy/evaluation processes remain active.

## Round 38 — fixed PickCube position/controller-unit matched retraining (2026-10-28)

Round 38 completed exactly one matched Base/Control retraining and one paired
30-scene native development trial. It is a repeated-development engineering
result, not a formal performance/generalization claim. No demos, HDF5 variants,
or layouts were created; ManiSkill task/controller code and PlaceSphere artifacts
were not modified. Ticket 05 has no local issue file in this checkout; this is
the next-slice execution record.

### Frozen contract and fixed observation profile

The preregistration is
`outputs/exp/20261028_221700_pickcube_position_units_r38/preregistration.json`
(file SHA-256 `49b659941b3e7a36c492dc17f83cf27428593ac6f647113efc170fa26ec9dcd9`,
registration identity `65963d4c4c01f95e497e4387051b17153a63384584af0b824461131cce4b1b38`).
The fixed profile is `pickcube_position_controller_units_v1`, profile identity
`8bcb1df3252e86b085d51507f72ce4d7722eadea7cc46d676d6da7d33b3d7fef`, stored at
`position_encoding.json` (file SHA-256
`d14d56e3645504b81db4a41f32cf6869b805dbf003194b26433cfceb6023f543`). It has
42 dimensions, zero offset, no fitted/data-derived statistics, no clipping, and
the following scale vector, derived from `NAMED_FIELDS` (not bare offsets):

| Named field slice | Scale | Reason |
| --- | ---: | --- |
| `agent/qpos[0:9]`, `agent/qvel[0:9]`, `extra/is_grasped[0]` | 1 | unchanged |
| `extra/tcp_pose[0:3]` | 10 | meters / 0.1 m controller translation reference |
| `extra/tcp_pose[3:7]` | 1 | quaternion unchanged |
| `extra/goal_pos[0:3]` | 10 | meters / 0.1 m reference |
| `extra/obj_pose[0:3]` | 10 | meters / 0.1 m reference |
| `extra/obj_pose[3:7]` | 1 | quaternion unchanged |
| `extra/tcp_to_obj_pos[0:3]`, `extra/obj_to_goal_pos[0:3]` | 10 | meters / 0.1 m reference |

Thus the scaled state indices are exactly 19–21, 26–31, and 36–41; all other
state channels are bitwise identity. The `0.1 m` reference is the checked Panda
`pd_ee_delta_pose` contract (`translation_scale_m: 0.1`), with runtime named
observation schema verification passing at 45 raw / 42 policy values. The
obstacle hint remains the existing latest-frame relative condition computed
from raw 45-D world-meter observations. Encoding is applied once at the common
42-D training/inference condition seam; no encoded observations are stored.

Action representation is unchanged native identity 7-D `pd_ee_delta_pose`:
DDPM samples are used directly, without codec, decode, explicit clamp, new
action meaning, or policy shield. The frozen-data reader still enforces the
existing `[-1,1]` action bounds and CLOSED (`-1`) obstacle terminal padding.
Base and Control have distinct typed checkpoint formats and both store the
same profile identity; a legacy identity or R37 z-score Base cannot initialize
this Control contract despite identical tensor dimensions.

Frozen data remains
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, validation
`f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`; exactly
direct IDs 0–999 / 83,004 windows and obstacle IDs 1000–1499 / 45,840 windows.
The R36 paired primary comparator and its same 30 restored development scenes
remain unchanged. The optional R37-training-vs-initial-dev 5-sigma feature
diagnostic was not run; no dev-derived quantities selected the fixed factor.

### Matched training and audit

Both networks retain the 42-D state / 84-D two-frame condition / 7-D action
dimensions and official architecture. Base trained direct-only for 30,000
updates, then Control trained obstacle-only for 150,000 from that Base final
EMA with Base frozen. Seed 1, batch 1,024, AdamW `1e-4`, betas `(0.95,0.999)`,
weight decay `1e-6`, warmup 500, cosine schedule, EMA power 0.75, and unweighted
native epsilon MSE were held fixed. Both use squared-cosine DDPM-100 epsilon
diffusion and final EMA selected before training. Atomic complete-state recovery
checkpoints exist every 10,000 updates (3 Base, 15 Control); loss logs contain
exactly 30,000 / 150,000 finite rows.

| Policy | Updates | Elapsed | First → final loss | Final checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Base | 30,000 | 842.53 s | 1.17828774 → 0.00278578 | `6a7c1af04970d3eb871c7cf607e2028dd847706070a03ae26e8e6aa8053d9869` |
| Control | 150,000 | 4,233.00 s | 0.03769969 → 0.00168165 | `1c3276e3d52efcb4f349c75612253750898a564f4ceb523c422b61c01763bf42` |

Logs/config/summaries are under `outputs/exp/20261028_221700_pickcube_position_units_r38/train_base/`
and `train_constrain/`; checkpoints/recovery states are under matching
`outputs/ckpt/20261028_221700_pickcube_position_units_r38_{base,constrain}/`.
Real-data Base and Control preflights were finite, with raw 45-D hint separation
and zero-control epsilon max difference 0.0. Strict standalone checkpoint loads
passed; independently seeded Base-vs-disabled-Control inference also matched
exactly (max difference 0.0). Post-training source aggregate SHA-256
`c4f468885eaa21ed741ac8ba48d064c77678eecce51b1fde4cb9c7ab39c03afd` and both
training-data identities match preregistration. GPU allocator peak was not
recorded.

### One paired native trial and result

Native preregistration was written before stepping at
`paired/preregistration.json` (file SHA-256
`c2202604a1442c5deec78c2ab8e477f2b9df53586e053b72763f05c3d64edd57`,
evaluation identity `1c99b817656e519469f9915786681750298d4f786ac95c84ad3150c1bb60b3ed`).
It pins both checkpoint hashes, profile, source aggregate
`c4f468885eaa21ed741ac8ba48d064c77678eecce51b1fde4cb9c7ab39c03afd`, R36 scene
manifest/HDF5, 30 episodes each, 500 control steps, per-episode 300 s and overall
36,000 s limits. Sampling is archived DDPM-100, epsilon, squared-cosine, slice
`[1:9]`, execute/replan 8; each policy is separately seeded from the same scene
seed. All 30/30 per policy completed with zero runtime errors; native state,
controller, raw observation and encoded-condition hashes pair scene-by-scene and
match R36 restored states. Every episode retains actual terminated/truncated
flags, native action-step traces, grasp/lift/transport/success stages, first
contact phase, final cube/goal XYZ and 3-D goal error. Complete contact entity
pair positions were not exposed by this record API and are not inferred.

| Policy | Task success | Safe success | Contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Position-unit Base | 22/30 | 0/30 | 30/30 | 23/30 | 0 |
| Position-unit Control | 29/30 | 13/30 | 16/30 | 6/30 | 0 |
| Primary R36 Base | 10/30 | 0/30 | 30/30 | 15/30 | 0 |
| Primary R36 Control | 25/30 | 16/30 | 13/30 | 4/30 | 0 |
| Secondary R37 z-score Control | 7/30 | 3/30 | 13/30 | 2/30 | 0 |

Current Base → Control paired transitions (improved/cleared, regressed/introduced,
both, neither): task `8,1,21,0`; safe `13,0,0,17`; contact `14,0,16,0`; XY
`19,2,4,5`. Against R36 Control, this Control transitions task `5 improved / 1
regressed / 24 both / 0 neither`; safe `2 / 5 / 11 / 12`; contact cleared /
introduced / both / neither `2 / 5 / 11 / 12`; XY `2 / 4 / 2 / 22`. These are
paired development comparisons, not isolated single-factor causal estimates.

The frozen gate **fails**: task 29 ≥27 passes; safe 13 <26, contact 16 >3, and
XY 6 >3 fail. The run is complete, but the blocker remains. Control terminal
phases were 29 task success / 1 lift-no-transport; first contacts were 2 grasp
approach / 1 lift / 11 transport / 2 goal stabilization. First-success steps
min/median/p90/max were 58/104/181/269. Per-policy wall seconds min/median/p90/max
were Base 3.422/7.554/15.736/16.749 and Control 3.440/5.291/9.333/23.466.
Final 3-D goal error min/median/p90/max was Base 0.0060/0.0243/0.1551/0.2167 m
and Control 0.0041/0.0076/0.0123/0.1190 m.

The native evaluator completed and retained all 60 rows but its first report
assembly hit an analysis-only missing-key mismatch (`interpretation` versus
`comparison_interpretation`). No rollout was repeated. The report was rebuilt
from immutable episode rows at `paired/analysis.json` (SHA-256
`00f3e90ce90ed949ef36d799cd231f757866633d7ac3189f95d1ddf6226de7ee`); status
records `analysis_repaired_from_retained_rows: true` and
`native_evaluation_rerun: false`. Only the R37 report assembly fallback changed
after native execution; preregistered runtime source identity remains intact in
the paired preregistration and analysis. Post-report source SHA-256 is
`249fc078e43a05116c4beaee58a2f23e2389e440fd0b7216d1edec0b99c48eae`; all other
registered files match. Offline transition analysis validates integer cells,
both cohort marginals, and the complete 30-row table; `cleared_old_only` is
explicitly old-true/new-false and `introduced_new_only` old-false/new-true.

Verification: full constrain suite **567 passed, 28 warnings** in 85.34 s after
the report-only correction and exact-marginal validator; `git diff --check`
passed. No evaluation worker remains active.

## Round 37 — matched PickCube direct-observation z-score retraining (2026-10-28)

Round 37 completed one matched Base/Constrain retraining and one paired 30-scene
native trial. The single changed factor was the shared 42-D observation encoding:
both policies were trained from scratch/initialized from the newly trained Base
EMA with `pickcube_direct_zscore_v1`. This is an end-to-end observation-profile
comparison, **not** an isolated Constrain-control or data-effect attribution.
No demonstrations, layouts, or encoded HDF5 files were created; the frozen
500-scene physical dataset and PlaceSphere artifacts were not modified.

### Frozen inputs, observation profile, and preregistration

The experiment registration is
`outputs/exp/20261028_200000_pickcube_direct_zscore_r37/preregistration.json`
(SHA-256 `d480bae967ae59d3ed3619e1da4b16ef6a42adcf99d83202ba7ac6d736494284`).
The fitted artifact is
`outputs/exp/20261028_220000_pickcube_zscore_r37/normalizer_stats.json`, file
SHA-256 `730d2e297253f9b4b14a3059902921ae226f7abe20094917d2e2a99f6396642f`,
statistics identity `d0bbcd4c00d6320635f97972d3211c1ed2b150b22b46a59780ad6d6259d08645`.
It fits float64 population statistics over all **85,004 native T+1 frames** in
the 1,000 direct trajectories only (IDs 0–999); every feature count is 85,004.
The transform is `(raw42 - mean42) / max(population_std42, 0.01)`, cast to
float32 at runtime, with no observation clipping. The 42-value named schema is
the existing PickCube state prefix; the named 3-value obstacle suffix is never
in the Base/Constrain state condition. Binary and quaternion values remain
linear channels; no features or model dimensions were added. The old relative
obstacle hint remains clipped according to its existing config and is calculated
from the raw 45-D latest frame, never from normalized TCP coordinates.

| Feature (native unit) | Mean | Population std | Minimum | RMS | Floored at 0.01 |
| --- | ---: | ---: | ---: | ---: | :---: |
| `agent/qpos[0]` (rad) | 0.044140794 | 0.106621053 | -0.248344913 | 0.115396961 | no |
| `agent/qpos[1]` (rad) | 0.463923214 | 0.178997469 | 0.0439996161 | 0.497257320 | no |
| `agent/qpos[2]` (rad) | 0.0107171889 | 0.0749889268 | -0.205471188 | 0.0757508896 | no |
| `agent/qpos[3]` (rad) | -2.09797907 | 0.360908903 | -2.59803200 | 2.12879577 | no |
| `agent/qpos[4]` (rad) | -0.00823821533 | 0.0640477291 | -0.221264020 | 0.0645753807 | no |
| `agent/qpos[5]` (rad) | 2.55457824 | 0.219548003 | 1.90018857 | 2.56399518 | no |
| `agent/qpos[6]` (rad) | 0.843373261 | 0.377329590 | -0.0951657072 | 0.923935105 | no |
| `agent/qpos[7]` (m) | 0.0268357181 | 0.0105686175 | 0.0182817224 | 0.0288418350 | no |
| `agent/qpos[8]` (m) | 0.0268155991 | 0.0105849011 | 0.0182602946 | 0.0288290910 | no |
| `agent/qvel[0]` (rad/s) | 0.00802788499 | 0.101574173 | -0.501390398 | 0.101890920 | no |
| `agent/qvel[1]` (rad/s) | 0.105793320 | 0.319833605 | -0.767346501 | 0.336876478 | no |
| `agent/qvel[2]` (rad/s) | 0.00151065718 | 0.0481899872 | -0.302984983 | 0.0482136594 | no |
| `agent/qvel[3]` (rad/s) | 0.148600861 | 0.433286034 | -1.15135872 | 0.458060043 | no |
| `agent/qvel[4]` (rad/s) | -0.00197533956 | 0.0493963883 | -0.375561386 | 0.0494358690 | no |
| `agent/qvel[5]` (rad/s) | -0.0529015899 | 0.292213551 | -1.07203209 | 0.296963529 | no |
| `agent/qvel[6]` (rad/s) | 0.00711125312 | 0.157346644 | -0.841103375 | 0.157507258 | no |
| `agent/qvel[7]` (m/s) | -0.0186116932 | 0.0299030074 | -0.379620582 | 0.0352219388 | no |
| `agent/qvel[8]` (m/s) | 0.00439317390 | 0.0310723083 | -0.418345451 | 0.0313813371 | no |
| `extra/is_grasped[0]` (binary) | 0.597430709 | 0.490415392 | 0 | 0.772936420 | no |
| `extra/tcp_pose[0]` (m) | -0.0473682268 | 0.0850955015 | -0.184164703 | 0.0973909302 | no |
| `extra/tcp_pose[1]` (m) | 0.0310782897 | 0.0506820379 | -0.0610247739 | 0.0594519054 | no |
| `extra/tcp_pose[2]` (m) | 0.0948521102 | 0.0440245605 | 0.0216526166 | 0.104570956 | no |
| `extra/tcp_pose[3]` (quaternion) | -0.000136261643 | 0.00183554629 | -0.0294772387 | 0.00184059703 | **yes** |
| `extra/tcp_pose[4]` (quaternion) | 0.980205160 | 0.0206232690 | 0.911496222 | 0.980422090 | no |
| `extra/tcp_pose[5]` (quaternion) | 0.00140957371 | 0.196833353 | -0.411276847 | 0.196838400 | no |
| `extra/tcp_pose[6]` (quaternion) | -0.00264409096 | 0.00410619275 | -0.0698821619 | 0.00488385461 | **yes** |
| `extra/goal_pos[0]` (m) | 0.147484437 | 0.0246798569 | 0.0482905619 | 0.149535128 | no |
| `extra/goal_pos[1]` (m) | 0.0321654257 | 0.0555496288 | -0.0599714406 | 0.0641901540 | no |
| `extra/goal_pos[2]` (m) | 0.119999997 | 0 | 0.119999997 | 0.119999997 | **yes** |
| `extra/obj_pose[0]` (m) | -0.0527428751 | 0.0880923124 | -0.183461741 | 0.102674565 | no |
| `extra/obj_pose[1]` (m) | 0.0330848460 | 0.0521981803 | -0.0616050027 | 0.0618001380 | no |
| `extra/obj_pose[2]` (m) | 0.0700521921 | 0.0543431891 | 0.0199918468 | 0.0886594147 | no |
| `extra/obj_pose[3]` (quaternion) | 0.645150211 | 0.297919692 | -0.00562614622 | 0.710615886 | no |
| `extra/obj_pose[4]` (quaternion) | 0.000126041122 | 0.00139330437 | -0.00717262086 | 0.00139899372 | **yes** |
| `extra/obj_pose[5]` (quaternion) | 0.000421706013 | 0.00136281550 | -0.00555317104 | 0.00142657002 | **yes** |
| `extra/obj_pose[6]` (quaternion) | -0.0380681045 | 0.702546763 | -1 | 0.703577383 | no |
| `extra/tcp_to_obj_pos[0]` (m) | -0.00537464825 | 0.0236640297 | -0.203784332 | 0.0242667085 | no |
| `extra/tcp_to_obj_pos[1]` (m) | 0.00200655628 | 0.0136905664 | -0.0874460638 | 0.0138368304 | no |
| `extra/tcp_to_obj_pos[2]` (m) | -0.0247999181 | 0.0347967981 | -0.204242811 | 0.0427300023 | no |
| `extra/obj_to_goal_pos[0]` (m) | 0.200227312 | 0.0897430491 | 0.00101563334 | 0.219419214 | no |
| `extra/obj_to_goal_pos[1]` (m) | -0.000919420356 | 0.0734646972 | -0.176836476 | 0.0734704503 | no |
| `extra/obj_to_goal_pos[2]` (m) | 0.0499478052 | 0.0543431890 | -0.0183884129 | 0.0738103342 | no |

The physical dataset remains
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5 SHA-256
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, validation
`f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`. The
selected classes are direct IDs 0–999 and obstacle IDs 1000–1499, with the
registered 83,004 / 45,840 window counts. Raw HDF5 actions remain native
seven-channel identity values; obstacle terminal gripper padding remains -1.

The registered runtime/checkpoint profile is typed separately from legacy
identity-normalized checkpoints: Base `pickcube_direct_zscore_base_v1`, Constrain
`pickcube_direct_zscore_constrain_v1`. Model dimensions remain Base 42-D / 7-D,
Constrain 84-D condition / 7-D action, horizons 2/16/8 and channels 64/128/256.
Training used 30,000 direct-only Base updates, then 150,000 obstacle-only
Constrain updates initialized from that new Base EMA with the Base frozen. Both
used seed 1, batch 1,024, AdamW `1e-4`, betas `(0.95,0.999)`, weight decay
`1e-6`, 500-step warmup, cosine schedule, EMA power 0.75, unweighted epsilon
MSE, and final EMA fixed before training. The inference protocol is archived
DDPM-100 with `squaredcos_cap_v2`, epsilon prediction, `clip_sample=True`,
prediction slice `[1:9]`, execute/replan 8, direct native actions and no decode
or explicit clamp.

### Training and artifact identities

Base run/logs/config: `outputs/exp/20261028_200000_pickcube_direct_zscore_r37/train_base/`;
recovery/final checkpoints:
`outputs/ckpt/20261028_200000_pickcube_direct_zscore_r37_base/`. It completed
30,000/30,000 updates in **841.96 s**, all losses finite, first `1.16361868`,
last `0.00290657`; final checkpoint SHA-256
`7498cbeaffd3675b504b4b448927e3288b9e20703fe62803e6d134e0ee9e8cbf`.
Atomic recovery checkpoints were retained at 10,000-update intervals.

Constrain run/logs/config:
`outputs/exp/20261028_200000_pickcube_direct_zscore_r37/train_constrain/`;
recovery/final checkpoints:
`outputs/ckpt/20261028_200000_pickcube_direct_zscore_r37_constrain/`. It
completed 150,000/150,000 updates in **4,236.80 s**, all losses finite, first
`0.03401160`, last `0.00181310`; strict final EMA load passed and the frozen
Base state hash is `4a1d8b29b65f08f526960088652b724737f0bbad1c9c88aef949156474a00fa6`.
Final checkpoint SHA-256 is
`20d5bb6e2027b34275a9a4d9d81b46d0d4d6c24ad01b91fe2c4cb318d4279ba3`.
Both post-training source and frozen-data identities match preregistration.
The real-data Base and Constrain preflights were finite with the planned schemas;
disabled-control epsilon max difference was 0.0.
CUDA allocator peak memory was not recorded by these trainers and cannot be
reconstructed from the retained artifacts.

### Paired native trial and gate

The one native trial is retained under
`outputs/exp/20261028_200000_pickcube_direct_zscore_r37/paired/`. Its evaluation
preregistration (written before the first native step) pins both final checkpoint
hashes, normalizer file/statistics hashes, sampler and the existing 30 R36
development-scene state/controller/observation source. All **30/30 per policy**
completed with zero runtime errors; all 30 initial native state, controller,
observation and normalized-condition identities paired across policies, and the
restored physical scene hashes match R36 scene-by-scene.

| Policy | Task | Safe | Obstacle contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized Base DP, 30k | 1/30 | 0/30 | 25/30 | 11/30 | 0 |
| Normalized Constrain, 150k | 7/30 | 3/30 | 13/30 | 2/30 | 0 |
| R36 Base reference | 10/30 | 0/30 | 30/30 | 15/30 | 0 |
| R36 Constrain reference | 25/30 | 16/30 | 13/30 | 4/30 | 0 |

Constrain versus Base transitions (task / safe / contact / XY): task improved 7,
regressed 1, both 0, neither 22; safe improved 3, regressed 0, both 0, neither
27; contact cleared 15, introduced 3, both 10, neither 2; XY cleared 10,
introduced 1, both 1, neither 18.

Against R36, normalized Base transitions were task improved 1 / regressed 10 /
both 0 / neither 19; contact cleared 5 / introduced 0 / both 25 / neither 0;
XY cleared 8 / introduced 4 / both 7 / neither 11. Normalized Constrain versus
R36 Constrain transitions were task improved 1 / regressed 19 / both 6 / neither
4; safe improved 1 / regressed 14 / both 2 / neither 13; contact cleared 9 /
introduced 9 / both 4 / neither 8; XY cleared 4 / introduced 2 / both 0 /
neither 24. The frozen gate **failed**: 7 task <27, 3 safe <26, 13 contacts >3,
and 2 XY crossings <=3. This is a repeated-dev engineering result, not a formal
performance or generalization claim.

Terminal phases (Base / Constrain) were no-grasp 4/0, grasp-no-lift 3/1,
lift-no-transport 10/20, transport/place failure 12/2, task success 1/7.
First-contact phases (Base / Constrain) were grasp approach 2/1, lift 15/1,
transport 8/9, goal stabilization 0/2. Contact links were retained per episode;
complete contact entity-pair positions were unavailable. Per-episode wall time,
native action/flag traces, final cube XYZ, goal XYZ and 3-D goal distance are in
the episode JSONs. Wall time min/median/p90/max was 8.925/15.587/16.008/16.125 s
(Base) and 4.820/23.082/23.964/24.227 s (Constrain). Final goal-distance
min/median/p90/max was 0.0247/0.1406/0.3180/0.3672 m (Base) and
0.0069/0.2510/0.3866/0.4305 m (Constrain). Success step was 279 for the single
Base success; Constrain success step min/median/p90/max was 91/110/158.4/225.

The native evaluator retained all 60 episodes before its report aggregation hit
an analysis-only missing phase-summary field. No rollout was repeated. The
report was recomputed from those retained rows at
`paired/analysis.json`; `paired/preregistration.json` preserves the exact
runtime source identity. The subsequent analysis-only correction changed only
`pickcube_zscore_round37.py` (registered SHA-256
`768ac8e45e94b49521ecc3a771febf0b6b764dd23dc2cebd3e0d581738094310`, post-report
SHA-256 `5cbae523c95130a8161c92ef52a804c2b37651bcce94d2c8857616fb1adfd906`);
all other registered source hashes remained identical. It adds an offline
reader for retained rows and does not change inference/training. Dataset hashes
were rechecked after the trial and remain unchanged. The machine-readable data,
checkpoint, episode-count, scene-pairing and source-difference audit passed at
`paired/post_trial_audit.json`. Registered source aggregate SHA-256 was
`c8262d9a03ba3d8028097c792c29c396692536a9a53f3143fbb7f5048eaba1ac`; the
post-report source aggregate is `1d974752580036acf582e10d66897fb9e304989470c7648ae2dddf4d8659045f`.
No process was intentionally left running.

## Round 36 — R35 report correction and archived PickCube sampler recovery (2026-10-28)

R35's original paired records, preregistration, and analysis JSON are preserved
unchanged. A separately written `outputs/exp/20261028_145500_pickcube_native_control_r35/paired/correction_r36.json`
recomputes the old-90k-Control → R35-Control transitions directly from all 30
raw `result` records. The original report had the adverse-event labels reversed
for contact and XY and had incorrect safe/task transition values. Correct values:

| Metric | Old → R35 marginals | Transitions |
| --- | --- | --- |
| Task | 27 → 26 | improved 2 / regressed 3 / both 24 / neither 1 |
| Safe | 15 → 16 | improved 7 / regressed 6 / both 9 / neither 8 |
| Contact | 15 → 13 | cleared 8 / introduced 6 / both 7 / neither 9 |
| XY crossing | 3 → 9 | cleared 2 / introduced 8 / both 1 / neither 19 |

The corrected counts reconcile exactly to both cohorts' positive-outcome or
violation marginals. They are an analysis correction, not a native rerun. In
particular, the original R35 100/500 result remains R35's recorded result and is
not relabeled as DDPM-100.

Archived source recovery resolves the earlier protocol uncertainty: Git blob
`43b08ab:examples/baselines/constrain/pickcube_learning_pilot.py` has SHA-256
`24522af7d02d3cdce98742ad7df0f3e7bbfba62321272fa1de8d509388e3c0ec`, exactly
matching the R13 preregistration's source identity. That historical adapter
uses `DDPMScheduler`, 100 training/inference timesteps and the native
`[1:9]` prediction slice executed directly in chunks of eight. It does not use
the later DDIM-50 profile or a 0.35 decode/clamp. This archival proof establishes
the intended same-weights R36 restoration protocol; it does not by itself prove
runtime equivalence or native outcomes.

### R36 native trial

Registration was frozen at
`outputs/exp/20261028_191000_pickcube_archived_ddpm100_r36/preregistration.json`
before native evaluation. The one trial used the exact R35 Base and Control
checkpoint hashes above, all 30 frozen scenes, 500 control steps, and 300 s per
episode. The selected outer timeout was 36,000 s. Runtime profile confirms
DDPM-100, stride 8, prediction `[1:9]`, direct native actions and source identity
matching registration. The original frozen dataset HDF5 and manifest hashes
were verified again after evaluation. No training, data creation, new layouts,
or checkpoint writes occurred.

Both cohorts completed 30/30 with zero runtime errors. Base scored **10 task / 0
safe / 30 contact / 15 XY**; Control scored **25 / 16 / 13 / 4**. On the exact
same R35 scenes and weights, restored state/controller/observation hashes match
scene-by-scene. R36 Base exactly reproduced the retained 90k experiment's old
Base outcomes scene-by-scene (10/0/30/15), supporting the restored native
DDPM-100 path. Control versus R35 transitions (task/safe/contact/XY) were task
2 improved / 3 regressed / 23 both / 2 neither; safe 6 improved / 6 regressed /
10 both / 8 neither; contact 6 cleared / 6 introduced / 7 both / 11 neither;
XY 6 cleared / 1 introduced / 3 both / 20 neither. Compared to the old 90k
Control, R36 Control transitions were task 2 improved / 4 regressed / 23 both /
1 neither; safe 4 improved / 3 regressed / 12 both / 11 neither; contact 5
cleared / 3 introduced / 10 both / 12 neither; XY 2 cleared / 3 introduced / 1
both / 24 neither. This secondary comparison still combines 500-vs-300 data and
150k-vs-90k updates.

R36's frozen gate is **pause**: task 25/30 <27, safe 16/30 <26, contact 13/30
>3, and XY 4/30 >3; Ticket 07 remains blocked. Control's first-contact phases
were transport 5, goal stabilization 3, grasp approach 3, lift 2. Terminal
phases were task success 25, transport/place failure 2, lift/no transport 3.
Success termination steps had min/median/p90/max 58/98/130.8/223. Per-episode
wall time, final cube XYZ/3-D goal distance, and full contact-pair positions are
not available in this evaluator and were not reconstructed. Full results and
all transitions are in
`outputs/exp/20261028_191000_pickcube_archived_ddpm100_r36/paired/archived_protocol_comparison.json`;
native records and traces are in `paired/base_dp/` and `paired/constrain_dp/`.
The R35 original native records and original report remain unchanged; its
corrected post-hoc transition table is separately preserved as
`outputs/exp/20261028_145500_pickcube_native_control_r35/paired/correction_r36.json`.

Verification: focused PickCube pilot tests passed 48/48; the full constrain
suite passed **555/555** (28 warnings) in 99.66 s before the final analysis-only
marginal-validation addition; focused pilot tests passed again 48/48 afterward.
The validation changed the analysis source hash after the R36 run, not the
registered sampler/evaluator path: registered runtime source identity
`cb2737f357e85a55652cb13b3696c70359ee806894006aca3676e0cce602c0b7` matches the
preregistration and saved inference profile; current analysis source identity is
`761f518ac65ae3ac10f201bce013d27a1be0931f885bd94fb5c57c1029a202d9`. Offline same-weight adapter
checks executed the exact archived `get_action` AST methods, bound to the loaded
current model/scheduler/normalization namespaces; with a fixed observation and
RNG seed both configured adapters matched the archived methods exactly (max
absolute action difference 0), returning finite `(1,8,7)` native actions.
Evaluation
and gate were run serially under the activated `maniskill` environment; the
gate remains pause. No worker was intentionally left running.

## Cleanup status

## Round 35 — PickCube frozen 1000/500 native-action control-only retrain (2026-10-28)

One native-coordinate ControlNet retrain was completed on the frozen 500-obstacle
package, initialized from the retained original native-action 1000-direct Base
EMA. No Base retraining, data/layout creation, replay, action codec, PlaceSphere
change, or ManiSkill source edit occurred. This remains a repeated-dev engineering
result, not formal performance or generalization evidence.

### Frozen inputs, native representation, and source protocol

The frozen package is `demos/PickCube-transport-detour/20261028_164500_frozen500/`:
HDF5 `0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`,
manifest `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`,
validation `f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`.
The opt-in native manifest reader requires all 1,500 IDs/classes, successful
held-at-goal metadata, safe obstacle records, the native seven-channel action
contract, closed `-1` obstacle terminal grip, and native window padding. It
returned exactly **83,004 direct / 45,840 obstacle** windows. The 1,000 original
direct observation and action arrays were individually compared byte-for-byte
against the source used to train the Base; all 1,000 matched. That source is
`demos/PickCube-transport-detour/20261024_104348_expansion/trajectory.h5`
(`2af959e3d3ebeba8bb046db6d9583d300e1533fc638c7d36eb6393dd28024dff`), with
manifest `26cb2b3f8c371315805cb6219e0a4314dce3461bac3aaa872d35ce2e5ca7fd45`.

The selected Base is the original
`outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt`, SHA-256
`dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d`. Its strict
checkpoint contract is `pickcube_pilot_base_v1`, step 30,000, selected EMA,
direct IDs 0–999, 42-D state condition, 7-D native `pd_ee_delta_pose`, identity
action normalization, and no action encoding. The newly trained control
checkpoint records a distinct typed native-identity contract; raw actions are
used directly, with no `0.35` codec scaling or post-sampling decode. Before and
after training, independently loaded Base and disabled-control epsilon and
DDIM-50 samples matched exactly (maximum absolute difference **0**).

Retained 90k preregistration, paired indices, and command records do not identify
the old inference scheduler, step count, or executed slice. Therefore the run is
explicitly labeled the predeclared fallback hypothesis **DDIM-50, eta 0,
100-training-timestep squared-cosine schedule, prediction `[1:9]`, execute 8,
replan every 8**, with the same scene-seed reset. It is **not claimed to reproduce
the exact old inference protocol**. This protocol limitation is material: the
same frozen-scene Base scored 8/30 here versus the retained old Base 10/30. The
checkpoint, scenes, and native initial states match, but old protocol evidence is
insufficient to treat that difference as a failed exact-protocol reproduction.
The old baseline was copied byte-for-byte and its tree SHA-256 verified before
the new paired evaluation: `98ba0bbaa5c74d6548ede8aa4a014e10a4456b9e55aae4343403d1d25450ca3d`.

### Training and recovery record

Original preregistration:
`outputs/exp/20261028_145500_pickcube_native_control_r35/preregistration.json`.
It freezes the dataset, Base, source identity, 30 dev scenes, final-EMA selection,
DDPM-100 epsilon objective, AdamW `1e-4`, batch 1,024, seed 1, warmup 500,
cosine schedule, EMA power 0.75, unweighted loss, and exactly 150,000 control-only
updates. Equal-batch exposure is approximately 3,351 obstacle-window epochs,
close to 3,342 for the retained 90k/300-window comparison; data size and update
budget both differ, so this is not a single-factor causal comparison.

Two completed 150k attempts are preserved with loss logs and failure records:
the first reached its post-training source audit but did not finalize because the
audit did not consult the already-frozen recovery identity; the second exposed
the missing explicit identity-action checkpoint field before it could save a
checkpoint. Neither was evaluated or used for checkpoint selection. Recovery
identity/preregistration history is retained as `recovery_identity_r1.json`,
`recovery_identity_r2.json`, `recovery_identity.json`, and
`recovery_preregistration.json` through `recovery_preregistration_r3.json`.
The selected completed attempt is
`outputs/exp/20261028_145500_pickcube_native_control_r35/train_constrain_retry2/`
with weights at
`outputs/ckpt/20261028_145500_pickcube_fullscene500_native_constrain_retry2/final.pt`,
SHA-256 `be5b2b740a59eaf428c71a906e15eb5c83d1cf807e1629c3bdf060ac31b8db93`.
It completed 150,000/150,000 updates in 4,497.32 recorded seconds; all losses
were finite (first `0.0308330`, final `0.00227748`, final-100 mean `0.00217241`).
The embedded Base state remained unchanged, trainable control parameters moved,
and strict EMA loading passed. Source identity was recomputed after training and
matched recovery SHA `6ab1502551991e49342be5053ca0e9b7d723370919c36b88d638b018d0e43c4e`.

### Paired native evaluation and gate

The evaluation used the exact same frozen 30 scene seeds and restored native
states for both policies, with matching state/controller/observation identity
hashes scene-by-scene. Both cohorts completed 30/30 with zero runtime errors.
Complete actual per-step native-action/flag traces are retained under
`paired/base_dp/` and `paired/constrain_dp/`; the original old records are in
`paired/baseline_snapshot/paired/`.

| Policy / reference | Task | Safe | Contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current Base, DDIM-50/8 hypothesis | 8/30 | 0/30 | 30/30 | 20/30 | 0 |
| Native Control, 150k / 500 obstacles | 26/30 | 16/30 | 13/30 | 9/30 | 0 |
| Retained old 90k Control | 27/30 | 15/30 | 15/30 | 3/30 | 0 |

Current Control versus current Base transitions (task/safe/contact/XY): task
20 improved / 2 regressed / 6 both / 2 neither; safe 16 improved / 0 regressed /
0 both / 14 neither; contact 17 cleared / 0 introduced / 13 both / 0 neither;
XY 12 cleared / 1 introduced / 8 both / 9 neither.

Against the retained old 90k Control, raw-record transitions are task 2 improved
/ 3 regressed / 24 both / 1 neither; safe 7 improved / 6 regressed / 9 both / 8
neither; contact 8 cleared / 6 introduced / 7 both / 9 neither; XY 2 cleared / 8
introduced / 1 both / 19 neither. The earlier R35 analysis JSON swapped the
adverse-event labels and misreported safe transitions; the immutable correction
is `paired/correction_r36.json`. Contact and XY remain worse than the old
Control. The exact old inference protocol is resolved by R36's archived-source
evidence, but R35 itself remains the recorded DDIM-50 run; R36 is the separate
same-weight DDPM-100 restoration.

The fixed gate is **pause / Ticket 07 remains blocked**: task 26/30 <27, safe
16/30 <26, contact 13/30 >3, and XY crossing 9/30 >3. New Control successes
terminated at median step 89.5 / p90 219.5 / max 293; first grasp median/p90
28.5/47.1 and lift 42.0/63.2. Its 13 first contacts were transport 9, goal
stabilization 3, lift 1 (reported contact links: Panda hand 9, fingers 3, cube
1). Four episodes terminated in transport/place failure. Full traces contain
native actions, grasp/lift/transport/success, collision/crossing, actual
termination/truncation, and contact phase/link. The existing trace schema does
not contain per-episode wall time, final cube XYZ/3-D goal distance, or complete
contact entity-pair geometry; these are not reconstructed. Aggregate details,
per-scene transitions and audit hashes are in `paired/evidence_audit.json` and
`paired/native_primary_analysis.json`; gate is `paired/gate.json`.

The 553-test constrain suite passed (28 existing warnings) before the frozen
recovery training and evaluation. No further training, data collection, layouts,
or evaluation trial was started. The PlaceSphere frozen configuration and
artifacts are untouched.

## Round 34 — fixed-weight PickCube 50→100 DDIM inference (2026-10-28)

This was one paired, exploratory inference-only comparison on the same 30 frozen
development scenes and the same R33 EMA checkpoints. R33's **actual** registered
native run used DDIM 50, eta 0, stride 2 for both policies; it did not execute
the 100-step inference originally requested for that round. Its preregistration
and native records are preserved unchanged and snapshotted under the R34 output.
R34 changes only inference from 50 to 100 DDIM steps. Training remains 100-step
DDPM with the same squared-cosine schedule; both checkpoint hashes, the codec,
data, episode limit, controller and held native-success condition remain fixed.
This is not a formal or generalization result.

The R33 training evidence inventory resolved a real directory-path mismatch:
Base's 30,000-step run/config/loss/summary are at the registered
`20261028_111705.../train_base/`; Constrain's actual 150,000-step run/config/loss/
summary are at `20261028_113532_pickcube_fullscene500_action_codec_retry1/
train_constrain/`, while R33 preregistration pointed at the separate
`20261028_125137.../train_constrain/` path. The retry tree contains exactly
150,000 contiguous finite loss rows and a matching summary/checkpoint; it was
catalogued in place, not fabricated or copied over the registered path. The
Base standalone checkpoint stores model/EMA and step but not optimizer,
scheduler, or RNG states. The Constrain checkpoint contains optimizer step
150,000, scheduler and RNG states, plus 100 EMA shadow tensors. Details and all
file hashes are in `r33_training_evidence.json`; the historical preregistration
remains the source of its original frozen identities.

### R34 preregistered execution and result

Registration: `outputs/exp/20261028_170000_pickcube_ddim100_paired_r34/
preregistration.json`; immutable R33 comparison inputs:
`r33_50step_snapshot/`; actual paired records and traces:
`paired_eval/`; derived paired analysis and gate:
`analysis.json`. The run strictly loaded the existing Base EMA
`2bd91abc5484559e3d7406b27c3e75e12d307e4df206d2361ed956a46533e1c2` and
Constrain EMA `e66c5ac700998648ec293ee29c2ca664fc00b1ad6cdfc2514140ac8533d236b3`.
The unchanged frozen package is HDF5
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f` and
manifest `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`.
All source identities were recomputed after the run and matched the new
registration (`2abd51ea93ce7bf7ea795bf441d44cc603d7de584722f244bc3dd28b8cc9e03b`).
R33's 50-step records were copied byte-for-byte before analysis; both paired
tree hashes and the preregistration/hash match their originals.

| Policy | Inference | Task | Safe | Contact | XY crossing | Runtime errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | R33 50 → R34 100 | 14/30 → 11/30 | 0 → 0 | 30 → 30 | 19 → 19 | 0 |
| Constrain | R33 50 → R34 100 | 17/30 → 15/30 | 9 → 6 | 21 → 24 | 4 → 6 | 0 |

Each 100-step episode paired exactly on scene, seed, restored native-state,
controller and observation hashes; all 60 completed, with no runtime errors.
Constrain's task transitions (50→100) were both 14, improved 1, regressed 3,
neither 12; safe success both 6, regressed 3, neither 21; contact both 21,
introduced 3, neither 6; XY crossing both 3, cleared 1, introduced 3, neither
23. The Base had 3 task regressions and no task improvements; its contact stayed
30/30 and XY had one cleared and one introduced scene. R34 Constrain first
contact phases were transport 21, grasp approach 1, lift 1, goal stabilization
1; reported links were Panda hand 21, cube 1, and each finger 1. Terminal
phases were 15 task successes, 14 transport/place failures, and 1 lift/no
transport failure. First grasp/lift steps had medians 31.5/41; successful native
termination step median was 90 among 15 successes. Full step/action traces and
per-episode first-grasp, lift, transport, success, contact, phase and flags are
retained. The evaluator does not save final native poses/heights, 3D goal
distance, or complete contact entity-pair geometry; per-episode wall time was
not logged (the frozen wall deadline was 600 seconds). Those values are not
reconstructed.

The frozen exploratory gate (task ≥27, safe ≥26, contact ≤3, XY ≤3, complete
30/30 and zero runtime errors) is **pause**: Constrain scored 15/30 task, 6/30
safe, 24/30 contact and 6/30 XY crossing. The 100-step result is lower on task
and safe success and higher on contact and crossing than the R33 50-step result.
The prior 90k comparison remains secondary and multi-factor, not a causal
inference-step comparator. No training, new data/layouts, or checkpoint selection
occurred. PlaceSphere outputs/configuration and ManiSkill source were untouched.

Reproduction (bounded native run; run under the activated `maniskill` env):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 36000s \
python -m examples.baselines.constrain.pickcube_learning_pilot evaluate \
  --registration outputs/exp/20261028_170000_pickcube_ddim100_paired_r34/preregistration.json \
  --scene-dir outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes \
  --base-checkpoint outputs/ckpt/20261028_111705_pickcube_fullscene500_base/final.pt \
  --constrain-checkpoint outputs/ckpt/20261028_113532_pickcube_fullscene500_constrain/final.pt \
  --output-dir outputs/exp/20261028_170000_pickcube_ddim100_paired_r34/paired_eval \
  --inference-steps 100 --device cuda
```

Focused codec/pilot tests passed 48/48; the complete constrain suite passed
551/551 with 28 warnings in 83.23 seconds. A source-identity-checked native run
completed both 30-episode policy cohorts, and `git diff --check` passed.

## Round 32 — PickCube full-scene existing-pool inventory (2026-10-28)

The user-approved target is 500 distinct real demonstrations, not 500 task-
unique layouts. Within training, task conflict is same seed or componentwise
start/goal XY within 0.02 m. Full-scene conflict additionally requires cylinder
center XY within 0.005 m per component, height within 0.005 m, and radius within
0.001 m. Frozen dev30 and Ticket 03 exclusions remain strict task identity
regardless of cylinder geometry.

### Phase 1 — versioned, geometry-reconciled inventory

The initial inventory is retained at
`outputs/exp/20261028_143000_pickcube_fullscene_inventory/`; it has an obsolete
1070 candidate count because the old 300-package manifest omitted cylinder
height/radius. It was not rewritten. The corrected versioned report is
`outputs/exp/20261028_164000_pickcube_fullscene_inventory/inventory_geometry_reconciled.json`;
its pre-audit correction registration is
`registration_geometry_reconciliation.json`. Old 300 cylinder identities were
reconciled from the pinned retained/merged source manifests and catalog records.
Corrected eligible catalog additions: **706**, all from the 900-record direct-
scene catalog; the 64 pair, 127 increment, and 200 deficit catalogs contribute
zero after geometry-aware first-qualified selection. This leaves 506 unused candidates after the
successful one-pass selection. The old strict task-only result remains 103
additional task-unique candidates; the 706 count is full-scene eligible and is
not called task-unique.

Reconciliation inventory command (after activating `maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 300s \
  python -m examples.baselines.constrain.audit_pickcube_existing_pool \
  --output outputs/exp/20261028_164000_pickcube_fullscene_inventory/inventory_geometry_reconciled.json --full-scene
```

Corrected inventory SHA-256:
`b2e3c310b3086294e96271dd191568861ca9d99a7e3e682c7fd74b7003a6443d`.

### Phase 2 — frozen native replay and exact 1,000/500 package

Before native execution, `outputs/exp/20261028_164500_pickcube_fullscene500/registration.json`
froze the 706-candidate ordered pool, every candidate's source/catalog ID,
seed/start/goal/cylinder/waypoint, initial-state/action/observation fingerprints,
old/direct dataset hashes, external manifests, runtime versions, and replay
source identities. Registration SHA-256:
`de4fe563f356735bce5e0ada49057734e41dd615f53ee5fae02d97bb5f6b5462`.
The first three original-layout canaries passed fresh-environment initial
state/controller/45-D observation restoration and exact first-transition state,
controller, 45-D observation, info, and flags. All 200 selected candidates were
then attempted in stable source/manifest order; **200 accepted, 0 rejected, 0
runtime errors**, 157.20 seconds. All 200 came from the existing
`direct_scenes/assembled` catalog; no seed/layout/cylinder/waypoint was sampled
or modified. The accepted IDs and 506 unused IDs are recoverable from the full
registration and `materialization/replay_report.json`.

The package is `demos/PickCube-transport-detour/20261028_164500_frozen500/`:
exactly 1,000 direct IDs 0–999 and 500 obstacle IDs 1000–1499, all 1,500 IDs
covered once. Old direct/obstacle arrays for IDs 0–1299 match their source
arrays; all direct arrays also match the frozen 1,000-direct source. The old
1,000/300 source HDF5 remains unchanged at SHA-256
`3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`.
Obstacle counts are **333 task-unique / 500 full-scene-unique**; the 500 figure
is not represented as 500 task-unique layouts. New 200 are physically executed
native demonstrations, not copied trajectories. New cylinder geometry is
identical to the corresponding frozen catalog record; radius remains 0.03 m.

Final identities: `trajectory.h5`
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`,
`trajectory.json`
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, and
`validation.json` is retained beside them. The independent audit verified exact
class IDs/counts, full-scene/task counts, all 1,300 old-source arrays, every new
T actions/T+1 raw-45/policy-42/full-state/controller/info/terminated/truncated
trace, native held success, collision/crossing-free prefix through first held
success, and all **200/200 exact initial plus first-transition restore audits**.
Every recorded controller snapshot is `{}`; runtime arm/gripper `config.use_target`
was present and false at every saved state.

Official PickCube action/window profile from the package audit:

| Class | Episodes | DP windows (obs=2, pred=16, before=1, after=14) | Length min/median/p90/p95/max | 6 pose-channel min/max | Grip min/max | Per-episode RMS quantiles (0/5/50/95/100%) |
| --- | ---: | ---: | --- | --- | --- | --- |
| Direct | 1,000 | 83,004 | 55 / 83 / 107.1 / 112 / 123 | each `[-0.35, +0.35]` | `[-1, +1]` | per-channel arrays in `validation.json` |
| Obstacle | 500 | 45,840 | 66 / 90 / 114 / 119 / 125 | each `[-0.35, +0.35]` | `[-1, +1]` | per-channel arrays in `validation.json` |

The first six stored values are the native `pd_ee_delta_pose` expert commands
(the collector's documented 0.35 action cap); the seventh is the native gripper
command. These are controller-native command coordinates, not meters/radians
or the PlaceSphere codec. Every native obstacle episode ends with grip `-1`
(closed) at held success. The opt-in official-window reader preserves the final
grip value in tail padding, so it does not invent an open-gripper terminal tail.
Future training should consume this version through
`PickCubeFullScene500Dataset` with both HDF5 and manifest SHA pins; retain native
action coordinates until a separate codec audit establishes a different
normalization. No training or evaluation was run in this round.

Focused tests passed 7/7; reader smoke loaded both classes and returned exactly
83,004 / 45,840 windows with `(2,42)` observations and `(16,7)` actions. Final
full suite: `timeout --signal=TERM --kill-after=20s 700s python -m pytest
examples/baselines/constrain/tests -q --tb=line` — **544 passed, 28 warnings,
86.51 s**. `git diff --check` passed. No replay, audit, or training worker
remained after verification.
One earlier registered packaging attempt at
`outputs/exp/20261028_163500_pickcube_fullscene500/` replayed 200 candidates but
failed during HDF5 state-series serialization; its incomplete package remains
retained as a failed attempt. The corrected 164500 registration persisted each
replay before assembly and produced the audited final dataset above.

## Round 33 — matched PickCube action codec, training, and paired dev (2026-10-28)

Ticket 05 has no local issue file in this checkout; this is the execution record
for its next bounded optimization slice. The retained data was not collected,
replayed, copied, or rewritten. Training used exactly the frozen direct IDs
0–999 and obstacle IDs 1000–1499 from the dataset above, with the same 30
development scenes as the retained PickCube 90k exposure experiment. PlaceSphere
data, weights, and configs were not used or changed. ManiSkill source was not
modified.

### Frozen action representation and matched training

The before-training profile is `pickcube_controller_range_v1`:

```json
{"name":"pickcube_controller_range_v1","version":1,
 "scale":[0.35,0.35,0.35,0.35,0.35,0.35,1.0],"offset":[0,0,0,0,0,0,0]}
```

Profile SHA-256: `f64e54ecd8eb26749fd4d0dbb18512ace929169d180334473d4e6e85a7053163`.
It maps the six stored native controller channels into `[-1,1]` and keeps the
native gripper channel unchanged. These are dimensionless normalized controller
coordinates, not meters/radians. The audit recomputed all source actions from
HDF5 before registration: both classes reach `[-0.349999994,+0.349999994]` in
each of the first six channels and `[-1,+1]` for grip; their encoded ranges are
`[-1,+1]`. Direct/obstacle episode RMS quantiles and the recomputed values are
frozen in each preregistration. Every obstacle terminal grip is `-1` (CLOSED).
Padding remains six zero pose values plus the final CLOSED grip; the PlaceSphere
OPEN-tail convention was not imported. Raw demonstration arrays were encoded
transiently when windows were read; no encoded HDF5 was made. Prediction decoding
is multiplication only, without clipping, and rejects any decoded native action
outside the existing `[-1,1]` controller bound.

The initial Base preregistration is
`outputs/exp/20261028_111705_pickcube_fullscene500_action_codec/preregistration.json`
(SHA-256 `35da4d1045cb138216294129d2fe514ca91f5f47bc7f5f7c569af5baf9907b1a`,
source identity `4c14662c929dd96c46d46ef1a0b61655a86fd29d840860056afd0e502b214751`).
Base trained from scratch on direct-only IDs 0–999 for 30,000 updates, seed 1,
batch 1,024, AdamW `1e-4`, warmup 500, cosine schedule, EMA power 0.75. It used
the existing 42-D state input, 7-D encoded action, `2/16/8` horizons and the
100-step squared-cosine DDPM epsilon objective. Final EMA was selected before
training. It completed in 867.82 s with 30,000 finite losses (first
`1.14611423`, last `0.00446880`, mean final 100 `0.00472909`); 83,004 windows
were sampled. Checkpoint:
`outputs/ckpt/20261028_111705_pickcube_fullscene500_base/final.pt`, SHA-256
`2bd91abc5484559e3d7406b27c3e75e12d307e4df206d2361ed956a46533e1c2`.

The matched control preregistration is
`outputs/exp/20261028_113532_pickcube_fullscene500_action_codec_retry1/preregistration.json`
(SHA-256 `eae50f326255fda93f09ebecc4a2c521b0bda37640b641ab42fde20cf77a536c`,
source identity `40cc412da7821cb3d46d8ccb16d03e487e33d7f676503ad86367ac8788e66707`).
It binds the exact Base EMA hash above, the same codec/profile hash, obstacle
IDs 1000–1499, HDF5/manifest hashes, and all source paths. The first Control
attempt under the earlier 111705 registration stopped before update 1 because
`NAMED_FIELDS` was missing from the config writer; it left only empty attempt
directories and is retained at
`outputs/exp/20261028_111705_pickcube_fullscene500_action_codec/train_constrain/`
and `outputs/ckpt/20261028_111705_pickcube_fullscene500_constrain/`. The import
was fixed and tested, and the corrected source/Base identities were frozen in
the retry registration before the next training command. No checkpoint was
written in the failed attempt.

Constrain trained only the existing control branch from that Base EMA, on the
500 obstacle episodes, for 150,000 updates (same seed/batch/optimizer/schedule,
unweighted epsilon MSE); 45,840 windows were sampled. It completed in 4,288.78 s
with 150,000 finite losses (first `0.11801428`, last `0.00433297`, mean final
100 `0.00454327`). The checkpoint's metadata records the actual 150,000 update
tuple and selected final EMA. Frozen Base state hash remained
`14b37834c0148ef02261c440a12b771e78e1f23c26eb5c27367aace1962d6cf3`, matching
an independent load of the matched Base EMA; all 100 reconstructed trainable
control tensors differ from their seed-1 initialized values. Before training,
zero-control epsilon and 50-step DDIM outputs matched the Base exactly (`0`
max absolute difference); after training, disabled-control epsilon/DDIM still
matched exactly (`0`). Checkpoint:
`outputs/ckpt/20261028_113532_pickcube_fullscene500_constrain/final.pt`, SHA-256
`e66c5ac700998648ec293ee29c2ca664fc00b1ad6cdfc2514140ac8533d236b3`.

### Paired 30-scene native evaluation

The final before-scene preregistration is
`outputs/exp/20261028_125137_pickcube_fullscene500_action_codec_paired_r2/preregistration.json`
(SHA-256 `8cd2821fc6c256305ba52f464f70408d6659bf829bdb497f2394a707839fc2a9`,
current evaluation-source identity
`b7494c1fdd1b4db700f0ba7329ccd9491bcf8a6714f4cc2204f0de0315819dbd`). It pins
the Base/Constrain hashes, training preregistration, retained old paired-record
tree, 30-scene manifest/HDF5, current ManiSkill PickCube source hash, all 1,500
training IDs and the 500-step/18,000-second evaluation budget. Two earlier
evaluation preregistration drafts were not run on scenes: 124911 preceded the
explicit decoded-native-bound guard; 125034 exposed a manifest-format reader
assumption before environment initialization. The corrected evaluator was
focused-tested and preregistered as 125137 before any native evaluation.

Both policies used a single 16-action DDIM plan per call (50 inference steps,
eta 0, stride 2), executed actions `[1:9]` two at a time, and refreshed the
latest-two-frame history. Base received the same 42-D state condition used in
training; Constrain received 42-D state plus the existing 3-value normalized
obstacle hint from the 45-D observation. Actions were decoded before native
execution. The existing native held-success condition was used with a 500-step
cap; there was no added release requirement, safety shield, grasp override, or
hand-coded pose/action. Each requested policy/scene episode is retained, with
native action trace, first contact, grasp/lift/transport/success steps, actual
terminated/truncated flags, and state/controller/observation restore hashes.
All 30 scenes ran for both policies, with zero runtime errors. The paired Base
and Constrain initial native state, controller state, and observation hashes
match scene-by-scene. Record span from Base episode 000 through the paired
summary is approximately 20m35s; the per-episode cap was 300 seconds.

| Policy | Task success | Safe success | Obstacle contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| New encoded Base DP | 14/30 (46.7%) | 0/30 | 30/30 (100%) | 19/30 (63.3%) | 0/30 |
| New encoded Constrain | 17/30 (56.7%) | 9/30 (30.0%) | 21/30 (70.0%) | 4/30 (13.3%) | 0/30 |

Within this new pair (Base → Constrain), task success transitions were both 8,
Constrain-only/improved 9, Base-only/regressed 6, neither 7. Safe success was
both 0, improved 9, regressed 0, neither 21. Contact was both 21, cleared 9,
introduced 0, neither 0. XY crossing was both 3, cleared 16, introduced 1,
neither 10.

The preregistered PRIMARY engineering comparator is the retained old PickCube
90,000-update Constrain run at
`outputs/exp/20261025_065159_pickcube_exposure90000/paired/`, with exact old
Constrain SHA `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`
and Base SHA `dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d`.
Old Base was 10/30 task, 0 safe, 30 contact, 15 XY; old Constrain was 27/30
task, 15 safe, 15 contact, 3 XY. Current Base transitions vs old Base:
task both 5 / improved 9 / regressed 5 / neither 11; safe neither 30; contact
both 30; XY both 8 / cleared 7 / introduced 11 / neither 4. Current Constrain
transitions vs old Constrain: task both 16 / improved 1 / regressed 11 / neither
2; safe both 5 / improved 4 / regressed 10 / neither 11; contact both 11 /
cleared 4 / introduced 10 / neither 5; XY both 1 / cleared 2 / introduced 3 /
neither 24. This is an overall engineering-pipeline comparison, not an isolated
codec/data/inference effect: the 500-obstacle data, action codec, 150k control
training, and DDIM stride-2 actor all differ from the old 300-obstacle/90k,
native-coordinate/DDPM-100 inference baseline. Repeated use of the frozen dev30
is engineering feedback, not independent generalization evidence.

The registered performance gate is **pause / Ticket 07 remains blocked**:
Constrain misses all four thresholds (task 56.7% <90%, safe 30.0% <85%, contact
70.0% >10%, XY 13.3% >10%). Against the current Base it improves task by 3,
safe by 9, contact by 9 fewer, and XY by 15 fewer; against the retained old
Constrain it regresses task by 10, safe by 6, contact by 6, and XY by 1. First
contact phases for the new Constrain were transport 17, lift 2, grasp approach
1, goal stabilization 1 (among 21 contacts); most remaining failures still
involve transport-phase contact. No additional independent trial was run.

Authoritative outputs are in
`outputs/exp/20261028_125137_pickcube_fullscene500_action_codec_paired_r2/paired/`:
`paired_summary.json` SHA-256
`56d1c1a2682aafdd440ebfa537e82926ea38e2e1c7fd5805fe8f0f6d0e50f4e7`,
`engineering_comparison.json` SHA-256
`5fbac5093b210149c0e05bc6b10aef899bddcf2857a52be767b9159acd81c887`, and
`gate.json` SHA-256
`0ca9b9f10f33a28830502aa90ee1a18d6f2d64ad5306b0f82c31ac6320d265ff`. The
historical paired directory was copied and its tree hash verified before
comparison; `pairing_failure_scene_indices` is empty. Final HDF5, manifest,
validation hashes remain unchanged at `0c74b88e…c349349f4f`,
`6b9d2c75…76e5c486`, and `f4756f69…d46d5b8f`. No new demos or layouts were
created. No evaluation was run for PlaceSphere.

Verification: complete constrain suite before the final eval-only corrections
passed 549 tests / 28 warnings in 529.60 s. After those bounded corrections,
codec, PickCube pilot, and paired-evaluator focused tests passed 56/56; both
edited Python modules compile and `git diff --check` passes. CPU strict-load
checks rejected an old native-action Base, incorrect codec identity and wrong
step count; both new actor adapters returned `(1,8,7)` finite native actions.
The final native-trace audit independently recomputed all policy counts, bounds,
success/contact/crossing flags and initial-state pairing from the 60 episode
records. No worker remained after the runs.

## Round 31 — PickCube existing-pool audit (2026-10-28)

Read-only audit at `outputs/exp/20261028_110000_pickcube_existing_pool_audit/`
using `examples/baselines/constrain/audit_pickcube_existing_pool.py`. It did
not initialize an environment, sample a layout, or replay an action. The
inventory records the exact source-file hashes, every packaged episode ID,
task identities (same seed or componentwise start/goal within 0.02 m), per-episode T/action/observation shape and
termination counts, aggregate seven-channel action ranges/RMS, and whether the
previously recorded source catalogs still exist. Reproduction (after activating
`maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 300s \
  python -m examples.baselines.constrain.audit_pickcube_existing_pool \
  --output outputs/exp/20261028_110000_pickcube_existing_pool_audit/inventory.json
```

| Existing source | Packaged counts | Obstacle layout findings |
| --- | ---: | --- |
| `demos/PickCube-transport-detour/20261024_104348_expansion/` | 1,000 direct / 100 obstacle | 100 task-unique; all 100 also occur in the 300 package |
| `demos/PickCube-transport-detour/20261024_213325_obstacle300/` | 1,000 direct / 300 obstacle | 300 task-unique; source-qualified cohort remains unchanged |

The expansion package HDF5 SHA-256 remains
`2af959e3d3ebeba8bb046db6d9583d300e1533fc638c7d36eb6393dd28024dff`; its
manifest SHA-256 is
`26cb2b3f8c371315805cb6219e0a4314dce3461bac3aaa872d35ce2e5ca7fd45`. The
300 package HDF5 SHA-256 remains
`3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`; its
manifest SHA-256 is
`3be137138176a4c243684c081ceda62f08c5792a3f3778a07460a3c2fa2642f5`.
The retained source catalogs contain 900 expansion direct-scene records, 64
expansion pair-scene records, and 127 + 200 obstacle-increment records. Their
catalog manifests and HDF5 identities are pinned in `inventory.json`. The
increment catalog manifests prove 103 geometrically unique candidates remain
after the old 300 obstacle cohort, frozen dev30, Ticket 03, and earlier
source-catalog identities are excluded under the seed-or-componentwise-2-cm
rule. The 900-record direct-scene source contributes 103 of these; the other
catalogs contribute none. Each is marked native-safe in its original scene
metadata with the fixed 0.03 m radius and full seven-channel `expert_actions`,
but catalog entries preserve only the initial native state (not T+1 native
states/controller/info records), so they require bounded original-layout
replay before packaging. With at most 103 additional candidates, the frozen
pool cannot supply the required 200; the maximum possible cohort is 403 and
the remaining gap is 97. No replay or package materialization was started,
because it cannot meet the requested 500 count. No new layouts were sampled,
and no existing data/checkpoint/configuration was changed. Ticket 05 has no
local `.scratch` issue file in this checkout, so this execution note is the
current decision record pending parent direction.

Catalog-aware audit JSON SHA-256:
`d5fd75829c14b266a8e0d1664eaba609a5b374671cb19255474f8fe4eb57b27d`.
External layout exclusions were pinned before candidate counting:
frozen dev30 manifest
`ffb2b54b6bf4f389d8bd81b0d5f282d833a297c6375ae2189c44b8359a9ee676`
(30 scenes) and Ticket 03 manifest
`82dc69a6f0e8581e96c25e677a9f812bc20e92ea757295b9b336a61891b2c0b1`
(8 scenes). The ordered 103 candidate IDs and each catalog manifest/HDF5 hash
are enumerated in the audit report. They were not preregistered for materializing
because this finite pool is short of the 200-episode increment prerequisite.
All retained actions are finite 7-D; raw observations are 45-D with T+1
rows. In the unchanged 300 obstacle cohort, lengths are min/median/p90/p95/max
`67/90/114.1/119/125`; each of the first six action channels reaches both
±0.35 and the gripper reaches ±1. The 1,000 direct class has 337 task-unique
identities by the same seed-or-componentwise-2-cm rule; these repeats are not
misrepresented as unique. Per-channel min/max and episode-RMS distributions for
both source classes, plus every individual length/shape/finite result, are in
the audit JSON. These existing action values do not authorize action
normalization or retraining in this round.

The audit’s 2-cm rounded key is an inventory/dedup diagnostic, not a substitute
for the original source acceptance contract. The original 300-package
qualification and array identity remain authoritative. The direct 1,000 class
is not reselected or represented as 1,000 unique task layouts. A future learning
slice must use the existing frozen package unless a compliant path to the 200
missing obstacle trajectories is explicitly resolved without generating new
layouts.

Focused audit tests in `tests/test_audit_pickcube_existing_pool.py` passed 4/4.
The complete constrain suite passed 541 tests (28 warnings) in 83.60 s;
`git diff --check` passed. No audit/replay worker processes remained after
verification.

The 2026-10-23 cleanup retains one final valid paired experiment per task,
the six selected Base DP / ConstrainNet weights, and the formal data, training
records, fixed scenes and verification needed to reproduce those experiments.
Safety success is the primary comparison metric; the paired Base DP comparator
is retained even where its safety result is worse. No checkpoint or scene was
reselected by looking at individual evaluation outcomes.

Pilot/calibration/probe/interface-smoke artifacts, failed or superseded runs,
the failed PlaceSphere ConstrainNet checkpoint and the periodic PickCube Base DP
checkpoint were deleted. The cleanup removed **992 files / 450,850,395 bytes**
(about 430 MiB); **765 retained file identities** were verified unchanged.

Cleanup record:
`outputs/exp/20261023_105159_final_artifact_cleanup/cleanup_manifest.json`.
It records all retained and deleted file paths, sizes and SHA-256 hashes, and
each completed deletion. `previous_OUTPUTS.md` in that directory preserves the
previous full output index. Historical manifests/reports were not rewritten:
references to removed diagnostics are historical, and resolve to deletion
records rather than current files. The earlier completion audit describes the
pre-cleanup acceptance evidence, including pilots that have now been removed.
`final_verification.json` in the cleanup directory records a second retained-hash
check and an independent recomputation of all six policy/task metric rows from
the 600 retained episode records; all match their authoritative summaries.

## Selected weights — exactly six files

| Task | Model | Current path |
| --- | --- | --- |
| StackCube (`panda_wristcam` evaluation) | Base DP | `outputs/ckpt/20260930_base_dp_stackcube_980/best_eval_success_at_end.pt` |
| StackCube | ConstrainNet | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/final.pt` |
| PickCube | Base DP | `outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt` |
| PickCube | ConstrainNet | `outputs/ckpt/20261022_144835_pickcube_constrain/final.pt` |
| PlaceSphere | Base DP | `outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt` |
| PlaceSphere | ConstrainNet | `outputs/ckpt/20261022_201100_placesphere_constrain/final.pt` |

The weight files keep their original bytes and hashes. StackCube uses the
previously selected `ema_agent` / final EMA; the two new tasks use their
preselected final EMA after 30,000 updates per selected model.

StackCube's historical runs have date-level provenance, so their weight folders
use `YYYYMMDD_` prefixes (2026-09-30 and 2026-10-21) without inventing a start
time. The 2026-10-23 renames preserve both checkpoint hashes. Their exact
old/new file mappings and hashes are in
`outputs/exp/20261023_105159_final_artifact_cleanup/path_migrations.json`;
the cleanup verifier uses this sidecar to resolve the original retained paths.

## Final data and validation

| Task | Retained formal data | Retained validation records |
| --- | --- | --- |
| StackCube | `demos/StackCube-obstacle/` (100 obstacle demos); Base DP source `demos/StackCube-v1/motionplanning/trajectory.filtered.state.pd_ee_delta_pos.physx_cpu.h5` (980 demos) | Dataset `validation.json`, training and scene audits below |
| PickCube | `demos/PickCube-obstacle/20261022_132753_formal/` (1,000 direct + 100 obstacle) | `outputs/exp/20261022_132753_pickcube_formal/` |
| PlaceSphere | `demos/PlaceSphere-obstacle/20261022_184558_formal/` (1,000 direct + 100 obstacle) | `outputs/exp/20261022_184558_placesphere_formal/` |

The historical `demos/StackCube-v1/` source/replay archive is preserved under
the repository's original-data traceability rules. No retained data was moved
or reserialized. PickCube and PlaceSphere pilot/intermediate datasets are gone.

## Final experiment directories

All paths in this table are below `outputs/exp/`.

| Task | Base training | ConstrainNet training | Fixed scenes | Final paired evaluation |
| --- | --- | --- | --- | --- |
| StackCube | `20260930_base_dp_stackcube_980_seed1/` | `20261021_constrainnet_wristcam_seed1/` | `20261021_wristcam_scenes/` | `20261021_wristcam_paired_eval/` |
| PickCube | `20261022_144000_pickcube_base_dp/` | `20261022_144835_pickcube_constrain/` | `20261022_152157_pickcube_scenes/` | `20261022_162002_pickcube_paired_eval/` |
| PlaceSphere | `20261022_193900_placesphere_base_dp/` | `20261022_201100_placesphere_constrain/` | `20261022_205500_placesphere_scenes/` | `20261022_213500_placesphere_paired_eval/` |

Each paired experiment retains both policies' complete 100-episode records,
summaries, parsed configuration, identity/restore evidence and representative
videos. PickCube's `termination_correction.json` remains authoritative for its
18/17 historical 500-step-limit outcomes; original flags were not reconstructed.
PlaceSphere's real time-limit truncations are 3/13 at the 200-step limit.

Also retained:

- `outputs/exp/20261022_221641_multitask_obstacle_completion_audit/`: original
  12-ticket acceptance audit, whose pre-cleanup scope is unchanged.
- `outputs/exp/20261023_105159_final_artifact_cleanup/`: current pruning and
  retained-identity audit. This supersedes historical *retention* descriptions.
- `outputs/exp/20261023_122842_pickcube_contact_diagnosis/`: Ticket 01 contact
  diagnosis for 28 old PickCube scenes. It does not replace
  `20261022_162002_pickcube_paired_eval/`. The earlier incomplete run
  `20261023_122333_pickcube_contact_diagnosis/` is not the evidence set.
- `outputs/exp/20261023_141712_placesphere_failure_diagnosis/`: Ticket 02
  crossing and terminal diagnosis for 28 old PlaceSphere scenes. It does not
  replace `20261022_213500_placesphere_paired_eval/`.
- `outputs/exp/20261023_153133_pickcube_transport_detour/`: retained Ticket 03
  batch. It does not close the parent end-sweep, join, or source-identity gaps.
  Bytes are unchanged. Matching trajectories remain
  `demos/PickCube-transport-detour/20261023_153133/`.
- `outputs/exp/20261023_165956_pickcube_transport_detour/`: Ticket 03 acceptance
  batch, 8/22 attempts. It supersedes `20261023_153133` for those three gaps
  and does not replace `20261022_152157_pickcube_scenes/` or
  `20261022_162002_pickcube_paired_eval/`. Matching trajectories are
  `demos/PickCube-transport-detour/20261023_165956/`. Incomplete same-day runs
  `20261023_164921_pickcube_transport_detour/` and
  `20261023_165430_pickcube_transport_detour/` are not this evidence set.
- `outputs/exp/20261023_235744_pickcube_learning_pilot_r2/`: Ticket 05
  learnability pilot, round 2 of 2. Gate is **pause**. It does not replace
  `20261022_162002_pickcube_paired_eval/` and does not admit Ticket 07.
  Round 1 `outputs/exp/20261023_183243_pickcube_learning_pilot/` is retained
  and was not used for training. Weights:
  `outputs/ckpt/20261023_235744_pickcube_pilot_base/final.pt`,
  `outputs/ckpt/20261023_235744_pickcube_pilot_constrain/final.pt`.
   Training demos: `demos/PickCube-transport-detour/20261023_235744_pilot_r2/`.
   These are not formal performance results.
- `outputs/exp/20261024_104348_pickcube_data_expansion/`: Ticket 05 authorized
  data-expansion round on the same frozen 30 dev scenes. Gate is **pause**.
  It does not replace `20261022_162002_pickcube_paired_eval/` or the round-2
  pilot, and does not admit Ticket 07. Weights:
  `outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt`,
  `outputs/ckpt/20261024_104348_pickcube_expansion_constrain/final.pt`.
   Training demos: `demos/PickCube-transport-detour/20261024_104348_expansion/`
   (1,000 direct + 100 obstacle, including the original 20+20). Not a formal
   performance or generalization result.
- `outputs/exp/20261024_202858_pickcube_proximity_weight/`: Ticket 05
  authorized training-optimization slice on the same 1,000 direct + 100
  obstacle data and the same frozen 30 dev scenes. One control-only retrain
  from the expansion Base EMA, with a frozen TCP-obstacle proximity weight on
  the epsilon loss. Gate is **pause**. It does not replace the expansion run
  or admit Ticket 07. No new demos. Weight:
  `outputs/ckpt/20261024_202858_pickcube_proximity_weight_constrain/final.pt`
  `22d3624941645ecac5bab4bd8679a2e40f77498b531b40f5e1d3c4be79d2a037`.
   Same-scene result versus expansion Constrain is worse: task 22/26, safe
   11/13, contact 18/17, XY 4/3. Not a formal performance result.
 - `outputs/exp/20261024_213325_pickcube_obstacle300/`: Ticket 05 authorized
   100→300 obstacle increment on the same frozen 30 dev scenes. The existing
   1,000 direct arrays were copied, not re-collected, and Base was not
   retrained. Unweighted epsilon; the rejected proximity weight stayed off.
   Gate is **pause**. It does not replace the expansion run or admit Ticket 07.
   Weight: `outputs/ckpt/20261024_213325_pickcube_obstacle300_constrain/final.pt`
   `bd7757122b75ad0bcc9cc5bd88a4b5174d48dfc520d1c8459332eea157732801`.
   Training demos: `demos/PickCube-transport-detour/20261024_213325_obstacle300/`
   `3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`
   (1,000 copied direct, not claimed unique; 300 obstacle = retained 100 plus
   200 new unique scenes). Same-scene result versus expansion Constrain
   26/13/17/3 is 25/12/18/2 task/safe/contact/XY. Not a formal performance
   result. Preregistration
   `3d37d27ac2145733c091094a5dd5245e9a237294abaebef6c37c1d29e28060a9`.
    Gate `296088e5e0b7118952d7dc020ca603dcfe98eeda3a1b0dc7cba430c56828e71a`.
    Comparison `f20d5c182c3d7079ad50e2411857670d96c77e6d7ee3b6b0e6f1474b7ee5d92e`.
  - `outputs/exp/20261025_062737_pickcube_window_gap/`: Ticket 05 expert-window
    diagnosis of the frozen 100 and 300 EMAs. No collection and no training.
    Same sample seeds and diffusion timesteps. Index
    `diagnosis/index.json`
    `35d9525892d3a2fc36dc75b905f0675b09669a82195867a8ff4d7edca44f8ed8`.
    Preregistration
    `db3a3479632664205578ab8f6eba00ed4bc0053f751ecb5d96c31661f2019bfb`.
    It does not replace the 300 trial or admit Ticket 07.
  - `outputs/exp/20261025_065159_pickcube_exposure90000/`: Ticket 05 authorized
    scratch 90,000-update trial from the matched expansion Base EMA. The saved
    30,000-step cosine last learning rate was 0, so resume was not used.
    Unweighted epsilon; proximity stayed off. No new demos. Gate is **pause**.
    It does not replace the 100 or 300 runs or admit Ticket 07. Weight:
    `outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt`
    `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`.
    Same-scene result versus expansion Constrain 26/13/17/3 is 27/15/15/3
    task/safe/contact/XY. Not a formal performance result. Preregistration
    `4e1be76719c9c22388db372c464cb3f222a4d1665343ef15c6c975ecfc3cfeca`.
    Gate `4f93ef34f0e6670246c7b5fd46c90e60fe9f54c31ab54dda9545093e22c42081`.
    Comparison `a8f213c327731fa2688b533c396a96c20f8b41b3932e5985ec6b2f6d67aad0c5`.
   - `outputs/exp/20261025_081913_pickcube_closed_loop_diagnosis/`: Ticket 05
     closed-loop diagnosis of the 15 exposure contact episodes. No collection,
     no training, and no stride trial. It does not replace the 90k eval or
     admit Ticket 07. Checkpoint unchanged
     `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`.
     Preregistration
     `68e84320a807face3b1a1f25a3da6246137e656725330627399913eda95679b6`.
     Index `diagnosis/index.json`
     `46a00982c870c3af71b897f7dedd20285849088d6cdf2bfc307a4dfc6ccf30ae`.
     Gate remains pause. Staleness was not isolated (0/15 queued overshoot).
  - `outputs/exp/20261024_014459_placesphere_learning_pilot/`: Ticket 06
  learnability pilot, round 1 of 2. Gate is **pause**. It does not replace
  `20261022_213500_placesphere_paired_eval/` and does not admit Ticket 08.
  No second round was started. Weights:
  `outputs/ckpt/20261024_014459_placesphere_pilot_base/final.pt`,
  `outputs/ckpt/20261024_014459_placesphere_pilot_constrain/final.pt`.
  Training demos: `demos/PlaceSphere-transport-detour/20261024_014459_pilot_r1/`.
  These are not formal performance results.
- `outputs/exp/20261024_155244_placesphere_data_expansion/`: Ticket 06
  authorized data-expansion round on the same frozen 30 dev scenes. Gate is
  **pause**. It does not replace `20261022_213500_placesphere_paired_eval/` or
  the round-1 pilot, and does not admit Ticket 08. Weights:
  `outputs/ckpt/20261024_155244_placesphere_expansion_base/final.pt`
  `2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640`,
  `outputs/ckpt/20261024_155244_placesphere_expansion_constrain/final.pt`
  `36518c84c0aacaadaf33eb7e35161b0e869c0d18d62e9425337021a763bfa43c`.
  Training demos: `demos/PlaceSphere-transport-detour/20261024_155244_expansion/`
  `df01aec45a5fcf6f04b0f0e2cb91d4bdf58d238ce41163b40cf15d1388d253ed`
  (1,000 direct + 100 obstacle, including the original 20+20). The packaged
  1,000 is not 1,000 unique task layouts (greedy unique 425, conflicts 575).
  Not a formal performance or generalization result. Preregistration
  `270ecc8712aef405f94cfdf14c6ad5f7cc1e30f4a7faf9d1d696a0185da45100`.
  Gate `fb768c089f124780eda1a477fbe5db26544d161b7faf9822f06b3032aa854391`.
  Comparison `3864d18c94ad775b18dfd2e5a0093df7ccae1ae7f342be574f6e79ac7612fbb0`.
  Coverage-writer fix validation, not a rerun:
  `outputs/exp/20261024_185754_placesphere_coverage_writer_fix/`.
  It does not replace this experiment or change its bytes. Experiment source
  `63d17eb3889b92b3638a42ee84ae3ada765b3dcf9b013307f3e4bba0d222d844` stays the
  preregistration identity. New writer identity
  `c370f36b3c1bb1d3ceb87fe6db200c754c0cf934d7cbe91fac566c6850c1b655` is not
  retroactive. Validation
  `2a59194181b48870889492aae53424523b5eca99958ece386fc3e166d65aec61`.
- `outputs/exp/20261025_085101_placesphere_obstacle300_90k/`: Ticket 06
  authorized combined 100→300 obstacle and 90,000-update slice on the same
  frozen 30 dev scenes. Factors cannot be separated. Gate is **pause**. It
  does not replace the expansion run or admit Ticket 08. No automatic
  300→1000. Base was not retrained. Weight:
  `outputs/ckpt/20261025_085101_placesphere_obstacle300_90k_constrain/final.pt`
  `70cf5e6f00bfec8b7d0945f65805fba06bbddd95356778d73009bdc5fed2326b`.
  Training demos:
  `demos/PlaceSphere-transport-detour/20261025_085101_obstacle300/`
  `f9dd209868d800fbce41eae276112a23dffc9aec02e91196ca900d46fc42aa02`
  (1,000 copied direct + 100 copied obstacle, arrays equal to
  `df01aec45a5fcf6f04b0f0e2cb91d4bdf58d238ce41163b40cf15d1388d253ed`;
  200 new unique obstacle). Same-scene Constrain versus expansion 25/21/7/0
  is 27/21/6/1 task/safe/contact/XY. Not a formal performance result.
  Preregistration
  `a2708b53736ad21d4242d94afa371cab37ab8456d454f6aea525d04e12779382`.
  Gate `b640d91352b28dc503303c35c3a96085835a4c761e75727700d0cd51d839c9e1`.
   Comparison `e0bd371ae6c32dfe7331f1d44a918bebe8b3fc2fa9f4d3640271f8e26597bb3f`.
- `outputs/exp/20261025_114030_placesphere_release_diagnosis/`: Ticket 06
  indexed diagnosis of the frozen `70cf5e6f` EMA. No training and no new data.
  It does not replace the 300/90k eval. Index
  `10a2ec98844bf9afaa7ea28f1ccf48c8049d017d28b8ed6d90d54e17f55cab38`.
  Evidence `8e93d6c9886e7e69e86accfc12622cc3b20f7205d4ccc28d06d6c947fa1a6bec`.
  Window index is `window_index.jsonl`, 40104 entries.
- `outputs/exp/20261025_114648_placesphere_phase_balance/`: Ticket 06
  one phase-inverse-frequency trial on the unchanged 300 demos, from Base EMA
  `2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640`, 90000
  updates, final EMA. Gate is **pause**. It does not replace the 300/90k run
  or admit Ticket 08. Weight:
  `outputs/ckpt/20261025_114648_placesphere_phase_balance_constrain/final.pt`
  `3385414d139d587252dfca36fdd6dd4501d7f1ac7fdfcd220e95fa08ed3a58bf`.
  Same-scene Constrain versus `70cf5e6f` 27/21/6/1 is 27/21/7/0
  task/safe/contact/XY. Contact cleared/introduced/both/neither = 2/3/4/21.
  Not a formal performance result. Preregistration
  `9c1a630dd7f4e29c09463fe29e49652f1daf08298aa0eb90fc8542723032432b`.
  Gate `1f0dd11faa7893d7e8b55f388e753da988cffa43902e932cc8d4cbee7f6ca493`.
  Comparison `e2444277f5d755d0b4a230e0631ac31828c25ed4541f8c97d71ba0b518f36b75`.
- `outputs/exp/20261025_131040_placesphere_coverage/`: Ticket 06 offline
  coverage of the six frozen `70cf5e6f` contacts against the existing 300
  obstacle windows only. No new demos, no training, no dev labels in the
  reference set. Prior diagnosis traces had no full observation window, so
  the frozen 30 were replayed; contact steps matched 1/90, 3/146, 5/113,
  9/146, 19/92, 21/162. Train windows 40104 on episodes 1000..1299:
  approach 16899, transport 17952, held_near_bin 4953, release 300.
  Execution-state rule would collect; policy-window rule would not
  (5/6 vs 3/6 outside). Nearest expert actions are still closed-grip
  negative delta-z. Not a reward or formal result. Preregistration
  `c4d7df16feae6bc5d9bf208f040769b8a7c2b4df6f99b888a0945e7351335480`.
  Coverage `566542a0146eda49df58420cf0fec371e5215cb0e57dcdbaaff5d9e78f281089`.
  Pilot identity `f238ddc29af22eab6daee9adf9e74d3d2132b0d2e2694e23530fc3cfcc2043e3`.
  Labeler identity `c53a60af0f3f698ffc2e497a6acba202d94e01f3dcbe53866c5612d6e328a6ce`.
- `outputs/exp/20261025_131730_placesphere_recovery_probe/`: one train-scene
  feasibility probe, not a packaged demo. A 2 cm lateral offset at the
  closed-grip container approach still commands a closed descent to the same
  target. Collection of this expert is not a recovery dataset. Probe
   `49facecbf0e06a09f75e970830f4401d017cf6a26b39b491232d99410985397e`.
- `outputs/exp/20261025_134916_placesphere_matched_terminal/`: Ticket 06
  opt-in above-release feasibility. Not a training set and not collection-ready.
  Legacy `terminal_mode` default remains `legacy`. Selected margin 0.015 m on
  8/8 calibration obstacle runs after 0.005 m failed the vertex filter on 2/8.
  Confirmation obstacle 16/16 native-safe, 0 contact. Dev30 after the freeze,
  not used for selection or training: 30/30 native-safe, 0 contact.
  Perturbation denominator 64 compared on 8 calibration layouts: legacy
  contact 32, new contact 10, introduced 0, cleared 22. Pre-terminal ±1/±2 cm
  contact was 0/32 for both. The contact drop is only at nearest-approach
  midstates, where new still contacts 10/32. Direct suffix accepted 0/8 and
  0/16. `confirmation.json` `collection_ready: true` is obstacle-only and is
  overruled by `collection_decision.json`. No demos written. Old 300 and
  direct h5 files were not overwritten. Run identity
  `aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed`.
   The later flag fix is not that identity. Radius, native thresholds, and
   cylinder geometry were not changed. No training. That run's
   `confirmation.json` `collection_ready: true` is obstacle-only and is not
   rewritten. It is not this round's identity.
- `outputs/exp/20261026_100200_placesphere_matched_terminal_r19/`: Ticket 06
   round 19 role split. Not a training set. Collection is **not ready**.
   Source identity `75e474697e1960c655d548945d7423ad28c6087ee65281f20ad5620ab7e33bf3`.
   Previous runtime `aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed`
   was not rewritten. Margin 0.015 m selected on obstacle positives only
   (0.005 failed 2/8 `hand_below_cylinder_top`). Calibration role bundle 8/8.
   Confirmation admission 13/16, `collection_ready` false in both
    `confirmation.json` and `collection_decision.json`. Held-out dev30 26/30,
    not used for selection or training. Negative direct 16/16 confirmation
    transport-blocked and admits no demos. No full collection and no training.
- `outputs/exp/20261026_103421_placesphere_terminal_round20_phase_a/`: Ticket 06
  Round-20 Phase A measurement before the motion correction. Final trace
  identity `4cc1780646a8ffe6514fd911a393279f4e4d466808591ac10598d45d519692b2`;
  diagnosis SHA `4977e33094005879408f17d1890dcf0d0c49dfbbf2979afd48c35d2948eb80a4`.
  R19 full failures are angular: 085 was 0.001252 m/s linear and 1.87655 rad/s
  angular; dev 003 was 0.000353 m/s and 4.54203 rad/s; dev 019 was 0.000361
  m/s and 7.39852 rad/s. After 200 neutral steps, 003 remained at 2.16161
  rad/s and 019 at 4.48883 rad/s. 085 reached native at +69 steps. Actual-
  cylinder suffixes 046/123 reached their own first native endpoints; dev-006
  did so after five neutral steps. Dev-000 measured shared z 0.151770565 m
  against direct requirement 0.151873754 m, a 0.103189 mm shortfall (vertex
  gap is approximate, not a surface proof).
- `outputs/exp/20261026_110038_placesphere_round20_long_settle_probe/`: focused
  continuation from retained full R19 actions, diagnostic only. Dev-003 reached
  native after 361 additional neutral steps; dev-019 after 785. No thresholds or
  geometry changed.
- `outputs/exp/20261026_104648_placesphere_matched_terminal_r20/`: preserved
  earlier complete Round-20 validation, identity
  `b437c3f2d2bb426936459921de248c5e0bb32a459e4dcfaa03865e560135993b`.
  Calibration 8/8 and confirmation 16/16 passed, but dev30 was 27/30. This is
  a historical recovery point and was not rewritten.
- `outputs/exp/20261026_105656_placesphere_round20_dev_engineering_feedback/`:
  non-selecting repeated-dev feedback for predeclared 8/16-step holds on
  dev-000/003/019. It did not repair those cases and is not independent
  validation. Recovery files are in
  `outputs/exp/20261026_110010_placesphere_matched_terminal_r20_recovery/`.
- `outputs/exp/20261026_110408_placesphere_matched_terminal_r20b/`: final
  Round-20 validation identity
  `3edc929330f644c407b94f6a23244520ae96b639706b954258336f82c3441958`.
  Margin 0.015 m; selected 0-step open hold on calibration 8 after all 0/8/16
  candidates tied 8/8. Confirmation 16/16 matched positives and 16/16 blocked
  negatives. Frozen dev30 admitted 30/30 without selection/training use and
  had 30/30 blocked negatives. Direct/obstacle release z matched exactly.
  Expert neutral settle is capped at 800 steps (1,000 total expert actions) for
  the measured spin cases; policy evaluation remains 200. Perturbation scope:
  64/64 finite comparisons on calibration 8; new contact 10/64, native 32/64,
  crossing 0/64 versus legacy contact 32/64, native 30/64; cleared/introduced/
  both/neither contact transitions 22/0/10/32. Preterminal native 30/32→32/32;
  nearest-approach native 0/32 for both. This is a bounded
  calibration-layout scope, not a broad rate. No full collection or training.
- `demos/PlaceSphere-transport-detour/20261026_111952_round20_matched_tplus1/`
  and `outputs/exp/20261026_111952_placesphere_round20_tplus1/`: one matched
  direct/obstacle T+1 state/controller/action artifact after the gates. Both
  roles exactly restored initial state/controller and first-action state,
  controller, policy observation, terminated, and truncated values. HDF5 SHA
  `51cb2f0109898c7477d6d522251ea58086559e3dd11c921aedc40e332544dacc`; runner
  SHA `f3001d7fa8032d526453c1bf41ccefcb33d9fa39b57c551d7c28e673d162ed39`.
  This is a T+1 proof artifact, not the 1,000+300 collection and not used for
  training. The existing formal generator still uses the legacy terminal path;
  that was the state before the Round-21 opt-in integration below.
- Round 21 matched-terminal collection, source integration and exact commands:
  `outputs/exp/20261026_133000_placesphere_matched_collection_r21b/`.
  The final pre-collection registration is
  `preregistration.json` (`a1f6ff02fac852ea1d73bfce951047e17f6dc74b22a2f4f3643e4dd176ad0a29`),
  source identity `6a8091fc694e43fffe84fd50d7d60280f649d927986a143445f26d91a420b048`.
  It pins the collector and all experiment helpers, native PlaceSphere source
  hashes, environment package versions, field schemas, action/observation
  dimensions, source layout catalogs, task-identity dedup and finite budgets.
  ManiSkill source and historical datasets were not modified. Existing geometry
  challenge, 3 cm cylinder radius, native thresholds, legacy terminal default,
  and 200-step policy budget remain fixed.

  The bounded native pilot at `pilot_subset/` regenerated 80 DIRECT + 25
  OBSTACLE demos across 25 matched pair layouts from the frozen final-candidate
  lists. All 105 T+1 initial/first-transition restore audits passed. There was
  one rejected expert case, retained in the journal; no policy-performance
  filter was used. 6/80 DIRECT and 2/25 OBSTACLE episodes exceeded 200 actions;
  median/p90/p95/max lengths were 105/147/160/909 and 116/179/529/633. Actions
  beyond the 200-step horizon were 13.4% and 20.9% of each role's action volume.
  Open near-zero-motion actions were 19.8% DIRECT and 26.9% OBSTACLE. The
  preregistered native-static 8-action windows were 0/1,765; action-neutral
  fractions and phase counts are retained in `pilot_subset/report.json`. This
  does not imply policy convergence; all complete native-success tails remain
  in the data.

  The first full 1,000+300 attempt at
  `demos/PlaceSphere-transport-detour/20261026_123000_matched_terminal_r21/`
  is preserved as a nonqualifying diagnostic package: its mixed-cohort selector
  produced only 213 greedy-unique OBSTACLE task layouts (87 duplicates). Its
  trajectory SHA was `61d5d06543bb1e4db13a9083bb10910c65c0d0b4ae0dc532fa72f6747809efd7`;
  it is superseded below and must not be used for training.

  The corrected, qualifying candidate package is
  `demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/`.
  Primary OBSTACLE sources were frozen as the 100 matched pair-catalog layouts
  plus the 200 `catalog_full` obstacle layouts; all 300 pass the task-identity
  dedup rule. A separately frozen 212-layout task-unique replacement pool was
  available; 3 replacements were used after 3 positive OBSTACLE native-terminal
  failures. The DIRECT class contains 300 matched positive DIRECT partners plus
  700 DIRECT-only trajectories from the separate source pool. Actual source
  counts: DIRECT = 703 `direct_scenes/assembled`, 99 `pair_scenes/merged`, 198
  `catalog_full`; OBSTACLE = 3 `direct_scenes/assembled` replacements, 99
  `pair_scenes/merged`, 198 `catalog_full`. All source scene IDs, seeds, and
  role-to-episode mappings are in `collection_full/layout_id_mapping.json`.

  Collection completed in 3,165.2 s: 1,000 DIRECT + 300 OBSTACLE, 300 paired
  layouts, 1,005 layout attempts, 3 preregistered obstacle replacements, and 5
  rejected cases (3 positive obstacle terminal failures; 2 positive direct
  native-placement failures). No exception, runtime error, state-restore
  failure, timeout, or dev/Ticket04 layout conflict was admitted. All 300
  accepted OBSTACLE layouts are task-unique and all 300 have paired DIRECT,
  negative transport-block, and suffix-replay proofs. The DIRECT class has
  472 greedy-unique task layouts and 528 within-class task repeats; source
  identity repeats are not presented as new unique layouts. This is comparable
  in scale to the prior 425 unique / 575 repeat DIRECT package, with the changed
  paired source mix disclosed above. External conflicts against the frozen
  dev30 and Ticket 04 catalogs are 0.

  Independent saved-artifact audit passed all 1,300 episodes: 7-D legal native
  actions, named 39-D policy / 42-D raw observations, T actions and T+1 native
  state/controller/info/termination records, first native success, and 1,300/1,300
  exact initial + first-transition restore audits. Every obstacle prefix is
  contact/crossing-free from execution start. Radius/native thresholds and
  obstacle geometry are unchanged; vertex clearances remain approximations,
  with PhysX execution as the safety evidence. Full layout mapping and per-episode
  findings are in `collection_full/dataset_validation.json`.

  Across all 1,300 episodes, 89 exceeded 200 actions: 60/1,000 DIRECT and
  29/300 OBSTACLE. Role medians/p90/p95/max are DIRECT 105/147/321/909 and
  OBSTACLE 126/185/544/889. Above-200 actions account for 15.8% DIRECT and
  20.7% OBSTACLE action volume. Near-zero open actions account for 21.6% and
  25.5%; the strict 8-action native-static window count is 0/22,364 before the
  first native-success stopping point. Step-phase action totals (DIRECT /
  OBSTACLE) are grasp approach 42,100 / 12,562; grasp pose 5,000 / 1,500;
  gripper close 8,000 / 2,400; lift 8,028 / 2,404; transport 29,015 / 15,021;
  clearance approach 4,112 / 1,228; stabilize 2,001 / 600; release 1,000 / 300;
  retreat 6,101 / 1,807; settle 27,723 / 12,555. Phase labels refer to the
  expert's current executed phase, with no future-state phase inference. Full
  per-phase current-action labels, episode lengths and action-window records are
  in `report.json`; no success tail was truncated or removed. Do not infer that
  the unchanged 200-step policy budget solves these longer expert tails.

  Identity hashes: `trajectory.h5`
  `eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0`,
  `trajectory.json`
  `ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86`,
  `attempts.jsonl`
  `b13b796325af0a838a803bf169d9dfc0daade6a394a7f9ef264da318d47ff42a`,
  dataset report `9d8e4c9231e37017df17c916df8a00b8cfad06d502f51a86e9f67d077fa3e44d`,
  independent validation `da77e933fa6f887594800c187d169255724c65ef82957513f3db4d9a84bfe089`,
  dataset identity `f0b9dd6d5b966744f44608528aa4bbab89baf9ebeb5b01972d315ad5ad78150e`,
  and layout mapping `a519b542faf497c49d0342851343c67dc7edb19a37c881d84f747ecf65811825`.
  Complete preregistration, pilot and full-collection command logs are retained
  in the matching `outputs/exp/20261026_133000_placesphere_matched_collection_r21b/`
  directory. No training or policy evaluation was run. The dataset is ready for
  independent parent review. Its HDF5 IDs follow collection order and are
  interleaved DIRECT/OBSTACLE, so the legacy trusted-formal trainers (fixed to
  the historical 1,000/100 IDs and hashes) must not be pointed at this version.
  After review, the matched training slice should add an opt-in reader keyed by
  the new manifest's class IDs, preregister the 1,000/300 identities, then run
  Base DP for 30,000 updates on DIRECT and ConstrainNet for 90,000 updates from
  that matched Base EMA on OBSTACLE. Use separate `outputs/exp/` and
  `outputs/ckpt/` paths. This is a proposed next plan, not a training result or
  an executed command.
- `outputs/exp/20261023_175529_placesphere_transport_detour/`: Ticket 04
  acceptance batch, 4/8 attempts. It does not replace
  `20261022_205500_placesphere_scenes/` or
  `20261022_213500_placesphere_paired_eval/`. Matching trajectories are
  `demos/PlaceSphere-transport-detour/20261023_175529/`. Same-day incomplete
  runs `20261023_174606_placesphere_transport_detour/` and
  `20261023_175100_placesphere_transport_detour/`, and the smaller successful
  probe `20261023_175419_placesphere_transport_detour/`, are not this evidence
  set.

## Reports

- [Current three-task results](THREE_TASK_RESULTS.md)
- [StackCube paired results](WRISTCAM_EVALUATION.md)
- [PickCube paired results](PICKCUBE_EVALUATION.md)
- [PlaceSphere paired results](PLACE_SPHERE_EVALUATION.md)
- [PickCube / PlaceSphere collection and training summary](MULTITASK_OBSTACLE_RESULTS.md)

Older pilot and task reports remain source-controlled documentation. Their
statements that diagnostic artifacts were retained describe their original
publication state, not the post-cleanup filesystem.

## Historical relocation map

These mappings still resolve retained final artifacts without rewriting old
manifest/checkpoint paths. More specific checkpoint mappings take precedence.

| Historical path | Current path |
| --- | --- |
| `outputs/ckpt/base_dp_stackcube_980/` | `outputs/ckpt/20260930_base_dp_stackcube_980/` |
| `outputs/ckpt/constrainnet_wristcam_seed1/` | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/` |
| `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt` | `outputs/ckpt/20260930_base_dp_stackcube_980/best_eval_success_at_end.pt` |
| `outputs/runs/dp_StackCube-v1_980_1/` (training records) | `outputs/exp/20260930_base_dp_stackcube_980_seed1/` |
| `outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt` | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/final.pt` |
| `outputs/constrain_runs/wristcam_full30000_seed1/` (training records) | `outputs/exp/20261021_constrainnet_wristcam_seed1/` |
| `outputs/wristcam_formal_20261021/` or `outputs/data/wristcam_formal_20261021/` | `demos/StackCube-obstacle/` |
| `outputs/wristcam_eval_scenes_20261021/` | `outputs/exp/20261021_wristcam_scenes/` |
| `outputs/wristcam_paired_eval_20261021_recovery/` | `outputs/exp/20261021_wristcam_paired_eval/` |

Activate `maniskill` before any execution. New data must use an explicit path
under `demos/`; new weights belong in `outputs/ckpt/`, and experiment records in
  `outputs/exp/YYYYMMDD_HHMMSS_<name>/`. Do not overwrite the retained final runs.

## Round 22 — matched PlaceSphere DIRECT/OBSTACLE training and paired dev

This is the authorized matched-data optimization continuation of Ticket 06,
not formal evaluation or generalization evidence. The branch remains
`goal/transport-detour-optimization` at base `730b744495914b82f3c19415cfaf30f536779ba2`;
no checkout, branch change, commit, or push occurred. ManiSkill source and all
historical datasets/checkpoints were left unchanged. The earlier
`20261026_123000_matched_terminal_r21` dataset (213 unique obstacle layouts / 87
repeats) was excluded. The only training data were the qualifying
`20261026_133000_matched_terminal_r21b` package:

- HDF5 `eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0`;
  manifest `ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86`.
- Manifest-selected IDs are pinned in preregistration; the actual HDF5 order is
  interleaved (DIRECT IDs are even, OBSTACLE IDs odd). Counts are exactly 1,000
  DIRECT / 300 OBSTACLE, disjoint and covering 0–1299. The opt-in versioned
  reader cross-checks the manifest labels against HDF5 `kind`, rejects missing
  labels, wrong counts and unpinned identities, and verifies T+1 39/42 state,
  seven-channel native actions and successful terminal evidence. Historical
  formal readers remain locked to their original hashes and 1,000/100 IDs.
- Actual official training windows: 132,080 DIRECT / 50,077 OBSTACLE. The
  separate stride-8 diagnostic has 16,189 / 6,175 windows and is not the
  training-window denominator. Episode length median/p90/p95/max is
  105/147/321/909 DIRECT and 126/185/544/889 OBSTACLE. Actions beyond 200 are
  15.83% / 20.71% of action volume; open near-zero actions are 21.58% / 25.52%
  (28,723 / 12,855 actions). Since the tails are below 50% and the action-neutral
  share is independently substantial, the preregistered choice was uniform
   official DP window sampling: no phase weight, trajectory reweighting, or
   truncation. Expert tails do not prove policy sufficiency at 200 steps.

  A separate, non-selecting quality audit was calculated from each episode's
  saved current-action phase and per-step linear/angular speeds in the HDF5. It
  uses fixed non-overlapping 8-action windows and no future phase inference. The
  action-only open/near-zero predicate is all eight gripper actions open and
  max norm of the six pose channels ≤0.02; linear and angular native speed checks
  are counted independently (≤0.01 m/s and ≤0.5 rad/s):

  | Role | All stride-8 windows | Open/near-zero action-only windows | Linear static among action-only | Angular static among action-only | Linear-static / angular-only failure | Both native-static |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | DIRECT | 16,189 | 3,311 (20.45%) | 3,301/3,311 (99.70%) | 0/3,311 | 3,301/3,311 (99.70%) | 0/3,311 |
  | OBSTACLE | 6,175 | 1,513 (24.50%) | 1,506/1,513 (99.54%) | 0/1,513 | 1,506/1,513 (99.54%) | 0/1,513 |

  Reproduction command (read-only against the pinned HDF5):

  ```bash
  source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s \
    python -u -m examples.baselines.constrain.audit_placesphere_matched_windows \
    --data demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
    --output outputs/exp/20261026_153000_placesphere_matched_paired_eval/matched_window_quality_posthoc.json
  ```

  All action-only windows are current-phase `settle`; 10 DIRECT and 7 OBSTACLE
  action-only windows fail the linear static check. Full report:
  `outputs/exp/20261026_153000_placesphere_matched_paired_eval/matched_window_quality_posthoc.json`.
  SHA `b0afc25631f84c745ddf486d203dd009438b19c71e15772fc35b203d2c0a6f5c`.
  This analysis was completed after preregistered training/evaluation, so it did
  not affect sampling, checkpoint choice, or gate interpretation. It is a
  process deviation from the requested order to resolve this quality gap before
  training; these rates are post-hoc, not preregistered independent
  angular-window coverage. Expert angular settling remains a bottleneck for
  parent review.

The before-training registration is
`outputs/exp/20261026_153000_placesphere_matched_paired_eval/preregistration_matched_r3.json`
(SHA `d0b7b88d5514198a5cae44ec4aff41b5bb1968b3162f580ff2956bc8144f0e02`),
which pins the dataset IDs/hashes, actual source file bytes, final-EMA selection,
30-scene seeds, budgets, and output paths. Training-source identity
`7c9bec6b3310ac6e3f2715a11f63a4b889b433f79dbce7f80ebbf98d6d0d0f79` was
verified before the Base and 90k runs. The earlier r1/r2 registration files in
the same experiment folder are superseded preregistration drafts: r1 had not
yet recorded calculated DP window totals, and r2 was generated before a trainer
syntax cleanup. Neither was used for any training or evaluation command.
Before evaluation, the adverse contact/XY
transition labels in the summary writer were corrected (episode inference was
unchanged) and re-frozen in preregistration r4, SHA
`f1a49951d652cfb30fd40a33b867f0664d0cc19efd1039d95ff63cf6b9a53adf`, source
identity `8fab4e962a21c8b67de70bd5272e6994f85595ad29ee7cf0e4e8b71b97527a0e`.

Preflight loaded the new Base EMA strictly into the PlaceSphere control model:
epsilon and full 100-step disabled-control equivalence both had max absolute
difference 0.0. The 300 obstacle IDs yielded 50,077 windows. Two finite
preflight updates verified finite loss, movement in the zero-initialized
control path, nonzero zero-convolution gradients and an unchanged frozen base.
An initial overly strong preflight assertion expected gradients in all upstream
and hint modules before those zero convolutions moved; those gradients are
structurally zero at initialization. No training had started in those attempts;
the corrected two-update check passed. All attempts terminated cleanly.
Before training, `py_compile` also caught an indentation error in the optional
trainer snapshot list; it was fixed and compilation/focused tests passed before
the final training registration. The first reader sample assertion expected a
42-D policy observation; the actual contract is 39-D policy plus 42-D raw, and
the assertion was corrected before registration. One process-inspection shell
check found `rg` unavailable; it was rerun successfully with `ps` piped to
Python. None of these issues changed dataset bytes or launched an unregistered
training/evaluation run. No `.tmp` checkpoints or experiment processes remained.

Exact finite-budget commands (all after activating `maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 2400s \
  python -u -m examples.baselines.constrain.train_placesphere_base_dp \
  --demo-path demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.json \
  --run-dir outputs/exp/20261026_153000_placesphere_matched_base_dp \
  --checkpoint-dir outputs/ckpt/20261026_153000_placesphere_matched_base_dp \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 --matched-r21b

source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 6000s \
  python -u -m examples.baselines.constrain.placesphere_constrain \
  --demo-path demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.json \
  --base-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt \
  --run-dir outputs/exp/20261026_153000_placesphere_matched_constrain_90k \
  --checkpoint-dir outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k \
  --device cuda --total-iters 90000 --batch-size 1024 --seed 1 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 --matched-r21b
```

Base completed 30,000 updates in 886.36 s; losses were finite (first
1.16264045, last 0.00248973). Final EMA:
`outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt`, SHA
`4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942`.
Base run config SHA `6cf5c963f1bb475489b826c7515b93591522c4b8f54d5f78ca692461ff7e6aff`;
summary SHA `a5cede581b8cb747d942159eff4f675c8162399c34428da88ef200dcaef49eb6`.
Control completed 90,000 updates in 2,533.85 s; losses were finite (first
0.02094785, last 0.00165384), frozen Base unchanged, control parameters moved,
and control-down, mid and hint gradients were nonzero. Pre/post disabled-control
epsilon and sampling equivalence max difference was 0.0. Final EMA:
`outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt`, SHA
`4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88`.
Control run config SHA `ef7fe1925a735817c57263d8376759d301b703500749d93366dfbc53dd700be6`;
summary SHA `ed67aa7c98d3c11ce50714dd64fd8887e1dde6cec9ca3186784296d6a5a8d042`.

Paired evaluation reused the original frozen 30-scene development set from
`outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes` (the same
set used by the 2026-10-25 300/90k run), with identical restored states,
controllers, observations and per-scene seeds; max 200 control steps, 120 s per
episode, 8,000 s overall. The primary comparison is the old 300/90k EMA; the
old 100-obstacle EMA is secondary. New evaluation command:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 8000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt \
  --constrain-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt \
  --output-dir outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval \
  --primary-baseline outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired \
  --secondary-baseline outputs/exp/20261024_155244_placesphere_data_expansion/paired \
  --device cuda
```

Both new policies evaluated all 30/30 episodes, zero runtime errors. Base was
6/30 task, 0/30 safe, 30/30 contact, 24/30 XY; old 300/90k Base was
5/30, 0/30, 30/30, 24/30. New Constrain was 26/30 task, 25/30 safe, 1/30
contact and 0/30 XY. Against the old 300/90k Constrain
27/21/6/1: task improved/regressed/both/neither = 2/3/24/1; safe = 7/3/18/2;
contact cleared/introduced/both/neither = 6/1/0/23; XY = 1/0/0/29. Against
the old 100-obstacle Constrain 25/21/7/0: task = 5/4/21/0; safe = 8/4/17/1;
contact = 6/0/1/23; XY = 0/0/0/30. New Base versus old 300/90k Base task
transitions = 4/3/2/21; safe = 0/0/0/30; contact = 0/0/30/0; XY = 1/1/23/5.

The frozen gate is **pause / fail**: required task ≥27/30 and safe ≥26/30,
contact ≤3/30 and XY ≤3/30; observed task 26 and safe 25 fail, while contact
and XY pass. These are development results, not formal/generalization claims.
Compared with the old 300/90k run, this matched dataset plus layout-replacement
package improves safety/contact outcomes but does not isolate a pure expert
causal effect. Do not proceed to Ticket 08 or PickCube formal work from this
result without parent review.

Episode-level evidence is retained in
`paired_eval/{base_dp,constrain_dp}/episode_*.json`; derived summaries are in
`paired_summary.json` (SHA `4ea9288fa5fb6506d92b5fb0d753b21a4747857c1a92a9ac2b517a53dbcf67b7`)
and `episode_diagnostics.json` (SHA
`eb241ecd060233a743d3c8607b40bed97a8a17eed06ab12a81c4235fc9081537`). New
Constrain had one first contact: grasp-approach, Panda left finger/gripper on
the top cap. It ended ungrasped; 30 releases were recorded, with 2
`incorrect_release` and 2 `not_stabilized` failures. Successful native completion
steps: 26/30, median 107, p90 160, max 165; all successes occurred within the
200-step policy budget. Base first contact occurred 30/30 (29 transport, one
release/stabilization); Base had 24 incorrect-release failures. Terminal speed,
grasp, contact entity/phase/location and release records remain per-episode.
Frozen evaluation config SHA
`edc7e458d9caad81bb7e73d6d0bf638ac7442c659a8225960295b7c5f1bc81f6`. Base and
Constrain episode-directory identity hashes are respectively
`f2d668093cdb60cdd0420feb0ad7c8f1cc2f558b9dac67fa6c27c8e21c4fe896` and
`877cd6ad6c5faca05da52cea50970227a1f06ddaee4bc7d66c10c40dc6b8b2da`.

Full constrain test suite after the post-hoc window-audit check:
`timeout --signal=TERM --kill-after=20s 700s python -m pytest
examples/baselines/constrain/tests -q --tb=line` — **484 passed, 28 warnings,
83.23 s**. `git diff --check` passed. GPU training/evaluation processes
were gone at verification; the RTX 5070 returned to idle. No runtime checkpoint
or evaluation errors occurred. Files changed are limited to
`examples/baselines/constrain/` plus this index and Ticket 06 execution notes.

## Round 23 — measured PlaceSphere hold-8 continuation

User-authorized continuous Ticket 06 optimization on
`goal/transport-detour-optimization`, starting HEAD
`a402e6c6b01bd2622468e8f827fb13a076795d2a`. No checkout, commit, or push.
ManiSkill source and all retained Round-22 artifacts were unchanged. This is a
development experiment, not formal evaluation or generalization evidence.

### Current matched-checkpoint diagnosis

`placesphere_matched_diagnosis.py` is pinned to the Round-22 Base EMA
`4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942`, Constrain
EMA `4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88`, and the
unchanged 30-scene paired evaluation manifest. It traces indices 12/13/21/22/25/28
and safe Constrain controls 0/2/4/6 for both policies, with full raw observations,
queued seven-D actions, TCP/sphere/bin/cylinder fields, action queue/replan
indices, grasp/release/contact/crossing transitions, and post-action linear and
angular velocity vectors. Controller/action slice `[1,9)`, horizons 2/16/8,
scene-seed inference, 200-step horizon, and native limits (0.01 m/s, 0.5 rad/s)
are unchanged. All 20/20 traces per policy reproduced the saved success/safety/
contact/crossing/termination/step outcomes across 10 selected scenes per policy;
0 mismatches and 0 runtime errors.

| Constrain dev index | Result | First contact / release | Terminal linear / angular speed |
| ---: | --- | --- | ---: |
| 12 | native task success, unsafe | grasp-approach contact step 4, queue 3; left finger/top cap | 0.008181 m/s / 0.470576 rad/s |
| 13 | release/stability timeout | release step 86, queue 5 | 0.025526 m/s / 1.276365 rad/s |
| 21 | release/stability timeout | release step 169, queue 0 | 0.581996 m/s / 29.099939 rad/s |
| 22 | release/stability timeout | release step 114, queue 1 | 0.000720 m/s / 4.987443 rad/s |
| 25 | safe native success | release step 111, queue 6 | 0.002481 m/s / 0.126532 rad/s |
| 28 | release/stability timeout | release step 99, queue 2 | 0.001383 m/s / 5.122324 rad/s |

The sole contact is early grasp approach, not release. Four current 200-step
failures are angular/stability failures, including two with sub-threshold
terminal linear speed. That did not isolate queued-action staleness as the shared
cause. Full corrected diagnostic:
`outputs/exp/20261026_210000_placesphere_matched_diagnosis_r23d/`; diagnosis SHA
`d3780c27bd156af222527e7e0a63b9d34d5807e47b747e92a017c2e4b840711e`, preregistration
SHA `9f66f2827b4db02f0defc45cb9e460d76aaee2c790f303dde22a204fc48f11f8`.
The full T+1 expert velocity/action join SHA is
`956e5a5ad0b00018a1c37294709536d7969631d98288cea48b51e3d15569fa4e`. It aligns
each action to its *post-action* native state and checks against recorded
per-action velocity norms. Among the 300 matched OBSTACLE training releases,
linear-speed median/p90/max was 0.368188/0.373907/0.375557 m/s (300/300 above
native linear limit); angular-speed median/p90/max was 0.403734/0.438300/0.501188
rad/s (1/300 above angular limit). Of the 12,555 post-release settle
transitions, 12,457 exceeded the native angular limit and 138 exceeded the
linear limit; retained expert tails do eventually reach their first native
endpoints.

### Physics selection, data regeneration, and training

Before any dev-set result was inspected, a physics-only replay compared 0/8/16
open-neutral holds after the actual release transition on frozen training
OBSTACLE IDs 1/3/5/7/9/11/13/15. Each candidate reused the same full initial
environment/controller state and native action prefix, inserting only the
registered hold. Safe native success was 6/8, 7/8, 7/8; contact and crossing were
0/8 for all candidates; terminal speed-threshold passes were 6/8, 7/8, 7/8.
The frozen tie-break selected the shorter 8-step hold. Probe result
`outputs/exp/20261026_181000_placesphere_matched_hold_probe_r23f/result.json`
SHA `eb45994e780b7553b79ec966fb6ccf37f1628eb7ee4c6a78830d313f0b29b769`;
preregistration SHA
`adb7dfdb2bc4ef8f3c65ac378d3e2ab0d4da90c1c650f701df19368e37fadbbf`.

The full collector registration
`outputs/exp/20261026_183000_placesphere_hold8_registration_r23.json` SHA
`0ae6445a848cf5a046b1602fdebcca1364233f766a42a70a5a1aa73574e69af6` freezes
hold 8, source-catalog identity, direct/obstacle role rules, task-layout dedup,
and finite serial budgets. A 20+20 paired native collector pilot completed in
73 s and validated 20 unique paired obstacle layouts and 40/40 restore audits;
it is retained separately at `demos/PlaceSphere-transport-detour/20261026_183500_hold8_pilot_r23/`
and excluded from full training.

The regenerated full data is
`demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23/`:
1,000 DIRECT + 300 OBSTACLE, 1,004 layout attempts, 2 predeclared replacements,
4 rejected layouts, 3,104.39 s, 300 unique matched obstacle layouts, and 474/1,000
greedy-unique direct layouts (526 repeats, not claimed unique coverage). No old
trajectory arrays were copied. Independent validation passed 1,300/1,300 exact
T+1 restore audits, native seven-D actions, 39/42 state observation schema,
first-native-success evidence, full-prefix contact/crossing checks, and paired
negative/replay proofs. Dataset identity: HDF5 SHA
`2bd725179ed61c8c32b97cdc9b89ce348add0bbf6b4d7a912cfa8bad30b9a5a1`, manifest
SHA `8685d2303cb269c5e9ccda9964586c25406e40afdcd83c49ad6fd3adbb24fbad`,
training identity SHA `76fd854fcea553bd020f8e22ea25ab078552c903afe5bafd0bf3125ad61af70a`.
Independent validation SHA
`c7508ec2709252c33ad534df24be390a902c922baf8fde622dc90a6a11e607a1`;
collection report SHA `aa292bcaec9a0671a3df3074ddab29ec4cbb493b610d8d5a0349160ed5c0c08f`.

The before-training trial registration is
`outputs/exp/20261026_195500_placesphere_hold8_training_prereg_r23b.json` SHA
`32057e428d24e301b78e485a8cbfd1c8ddec56cd7c6d3962d80dab314e27ac84`; registered
source identity `8baa3958d6341d6e5e632b74fd8150297dc3fae9bedeb703c96f43ad6eafa059`
was rechecked with no source drift before training. It pins uniform official
window sampling, new matching data identity, seed 1, Base 30,000 updates,
Constrain 90,000 updates from the new Base final EMA, final-EMA weight selection,
the frozen 30 dev scenes, 200 policy steps, 120 s per episode, 8,000 s overall,
output paths, and original gate thresholds.

Base: 30,000 updates in 881.56 s, all losses finite (1.159990 → 0.002054), final
EMA SHA `92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`.
Run config SHA `bfcde1a99d908b29cdb569eb3af401ae33119aca8f96b3db100fbec449ffbaee`;
summary SHA `ddab69156d38f80e2f524497659470d2ecec56d6ca36878d1774ac1831cda35c`.
Constrain: 90,000 updates in 2,518.40 s, all losses finite (0.015319 →
0.001245), frozen Base unchanged, control branch moved with nonzero gradients,
pre/post disabled-control epsilon and 100-step sampler equivalence exact.
Final EMA SHA `c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada`;
run config SHA `7bbe0dc6c4e89e9f1f5b5df332a083d6eda19a0cc398c041f2a938dce8385d2f`;
summary SHA `8fab4eead8d240205a6db64b3f9395b2899d115453eb47a82d89784aae12303d`.

### One paired dev trial and gate

The one registered evaluation used the exact Round-22 frozen 30-scene manifest,
200 steps, both policies, and zero runtime errors (30/30 each). Hold-8 Constrain
scored 25/30 task, 25/30 safe, 0/30 contact, 0/30 XY; successful native
completion steps were median/p90/max 113/160/184. Against the primary current
matched Round-22 Constrain (26/25/1/0): task improved/regressed/both/neither
4/5/21/0; safe 5/5/20/0; contact cleared/introduced/both/neither 1/0/0/29;
XY 0/0/0/30. The previous contact cleared at scene 12, but native task success
regressed at scenes 5/10/11/17/24 while former failures 13/21/22/28 recovered.
The five current failures were all release/stability timeouts at the 200-step
limit: scenes 5/10/24 stayed on-bin but spinning; scenes 11/17 were high-motion
off-bin releases. The hold removes the single contact but does not clear the
remaining release/stability bottleneck.

**Gate: pause/fail.** Task 25 < 27 and safe 25 < 26; contact and XY pass.
This is a development result, not formal evaluation. Paired summary SHA
`63f316e6188c1ce2ea4b0c454c1552de67b63510bbca0aae13aa7348a54c8f04`; frozen
config SHA `fef33e7ca0f8319d043cfc7bdf5fcb74654a1ac12bf104e7825df94a6f8c3d88`.
The matched current EMA baseline and per-episode files remain intact.
Persistent Round-23 argv/budget/outcome ledger:
`outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23/round23_execution.json`
SHA `a30b1436e315413befa5013e30ee906d3310e27d9f0325c647d5556739f5f41f`;
all frozen training source hashes still match.

Final full constrain suite after Round-23 code additions:
`timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
— **498 passed, 28 warnings, 82.41 s**. `git diff --check` passed. Training,
evaluation, diagnostic and collection processes exited; no leftover `.tmp`
checkpoints or experiment processes were present. Ticket 08 remains blocked. The
next data-backed seam is current angular release/stability under the 200-step
policy budget; no additional R23 factor or formal evaluation was started.

## Round 24 — indexed release mechanics and exploratory execution stride 2

User-authorized single-developer continuation on
`goal/transport-detour-optimization`, starting at base
`9f190f80665810d55e35c3436710a0aeb2b2d95b` (integration branch `dev`). No
checkout, branch change, commit, or push. ManiSkill source, training data, and
checkpoint bytes were not changed. No training, data collection, formal
evaluation, or PickCube work was run.

### Release-mechanics evidence

Native Base DP and Constrain episodes were instrumented with full per-step
seven-channel actions, observation/replan step, queue index, TCP xyz/quaternion,
sphere xyz/quaternion and linear/angular velocity, grasp/open state, obstacle
contact/crossing flags, control timestep, and termination flags. The Constrain
trace reproduces all five R23 failures (scenes 5, 10, 24, 11, 17) and includes
the preregistered matched-success controls (0–4). Full traces remain inside
`paired_eval/{base_dp,constrain_dp}/episode_*.json`; indexed open-command and
physical `is_grasped: true→false` event joins are in
`release_causal_measurements.json`.

The training-expert audit reads the regenerated R23 hold-8 300-OBSTACLE HDF5,
not the older R22 expert. It verifies 48,982 action/state pairs: action index
`t` is compared to the T+1 native state `t+1`, and each of the saved linear and
angular speed values matches that same post-action state. The 300 open-action /
ungrasped transitions have pre-open linear speed median/p90 0.00549/0.00678 m/s
and angular speed 0.02665/0.03084 rad/s; same-index post-action medians/p90 are
0.36818/0.37391 m/s and 0.40389/0.43830 rad/s. This is a synchronized
before/after kinematic association, not an impulse or proof of cause. Expert
per-action timestep is absent, so expert `g·dt` is not calculated.

The five Constrain failure replays show pre-open motion ranging from 0.0030 to
0.0223 m/s linear and 0.0247 to 0.3560 rad/s angular. At the physical
ungrasped transition, downward linear velocity magnitude is about 0.364–0.378
m/s. The following 0.05 s displacement is about −0.0329 to −0.0336 m, compared
with −0.0123 m under the zero-initial-velocity `½g·dt²` comparison; observed
`dz/dt` and `g·dt` are recorded per event. This does not isolate release
actuation, contact, gravity, or pre-existing motion. Per-step sphere/robot
contact is not exposed in saved native step info (obstacle contact is); joint
torque and impulse are not measured. Do not infer a dynamic impulse from these
records.

### Preregistered stride-2 trial

Before evaluation, registration
`examples/baselines/constrain/placesphere_stride2_preregistration.json` pinned
the 30-scene manifest, exact source hashes, Base EMA
`92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`, and
Constrain EMA
`c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada`. One
factor changed: each policy call still predicts its original 16×7 sequence and
selects the original `[1:9]` actions; execution consumes two actions, appends
the latest two observations, and replans. Observation history remains 2, all 7
action dimensions, weights, sampler, scene seeds, clamps and safety behavior
are unchanged. More calls naturally consume more sequential diffusion RNG
draws; this coupled sampler-noise change is disclosed. This is explicitly
**EXPLORATORY**: more frequent feedback may help release errors, but R23 did not
establish that stale queued actions caused them.

Both policies ran all 30 scenes with the same restored state/controller/seed,
200 control steps, native first-success termination, a 300 s episode wall cap,
and a fixed 30 denominator. The 18,000 s outer timeout is compute-only; no
scene was dropped and no result-based stopping/selection occurred. The exact
command and exit/cleanup details are in `execution_record.json`.

| Policy | Task | Safe | Contact | XY crossing | Runtime errors | Successful steps median/p90/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base DP stride 2 | 8/30 | 0/30 | 30/30 | 25/30 | 0 | 160.5/185.5/189 |
| Constrain stride 2 | 26/30 | 26/30 | 1/30 | 1/30 | 0 | 135.5/158.5/171 |

Against primary R23 Constrain stride 8 (25/25/0/0), stride 2 has task/safe
transitions improved/regressed/both/neither = 4/3/22/1, and contact and XY
cleared/introduced/both/neither = 0/1/0/29 for each. Four of the five former
release/stability failures (5, 10, 24, 11) now succeed; scene 17 remains a
release/stability timeout at step 200. The one new contact/XY episode is scene
25. Against secondary R22 matched hold-0 Constrain (26/25/1/0), transitions are
task 4/4/22/0, safe 5/4/21/0, contact 1/1/0/28, and XY 0/1/0/29. R22 is
secondary and not a one-factor comparison.

The frozen gate remains **pause/fail**: task 26/30 is below 27; safe 26/30,
contact 1/30 and XY 1/30 meet their thresholds. The remaining scene-17 failure
and single introduced contact/crossing are retained; no rerun or additional
stride trial was started. New result directory:
`outputs/exp/20261026_235900_placesphere_stride2_r24/` (registration,
`paired_eval/`, per-episode full traces, `release_causal_measurements.json`, and
`execution_record.json`). Paired summary SHA
`3c537c1ba646e5d483e2fd4075524ba36bc0ea08eae4356cc4e0231276c2b0cd`; expert
HDF5 SHA `2bd725179ed61c8c32b97cdc9b89ce348add0bbf6b4d7a912cfa8bad30b9a5a1`.

Verification: constrain suite **502 passed, 28 warnings, 83.65 s**; `git
diff --check` passed. Full evaluation exited 0 with 60/60 episodes and no
runtime errors; model/evaluation child processes exited. Next minimal slice is
parent review of the indexed event traces, especially scene 17 and why the
post-action expert velocity jump appears in the matching T+1 record. Do not
claim causality or launch further trials in this round.

## Round 25 — candidate-collector interruption diagnosis and bounded resume canary

The obstacle1000 collection did **not** reach matched demonstration collection,
training, or paired evaluation. The checked-out branch remains
`goal/transport-detour-optimization` at HEAD
`6aeee1ecf559aa9e247e3348ff11f308595e94fe`; no branch operation, commit, push,
or ManiSkill source edit occurred. All native commands ran after activating
`maniskill` and had finite outer/per-attempt timeouts.

### Exact stall and retained work

The before-collection plan is
`outputs/exp/20261026_231800_placesphere_obstacle1000_r25/preregistration_collection.json`
(SHA `34d0b21272b34fefb7a5f0b5a4fc2469d73541877c57ca84455d2e87dc8e1cbe`). The
first geometry-candidate run in
`outputs/exp/20261026_231800_placesphere_obstacle1000_r25/candidate_source/`
requested 700 layouts under a 21,600 s wall / 10,000-attempt cap. It stopped
after 21,648.57 s: 2,941 attempts, 223 accepted safe geometry layouts (7.58%),
and 2,718 retained rejections. These are scene/layout proposals, not the new
hold-8 matched training episodes. Its seed interval was 2026102712–2026105652.
HDF5 SHA `cc92ddcc86faf5ebb6a2c2ac1b21fe30a55975bc6072ada81a55b9dbc477a2d4`,
manifest SHA `910cb1c8db1d5bd675624e7854c0f9f0c4f0f73ccd9bd0e05192a67175946fca`,
attempt journal SHA `6676d01ac0752b040a25f053fb8952127f214bf68b5c7fcad83d05426b4720d5`.
Rejection counts: 1,036 task-layout duplicate; 638 direct-template
unsuccessful; 552 side-detour column infeasible; 262 direct block at endpoint;
160 direct container unsuccessful; 27 waypoint out of bounds; 15 obstacle
native unsuccessful; 15 side-template unsuccessful; 13 obstacle contact or
crossing.

The next run reused the initial seed argument rather than starting after the
last journaled seed. Its 3,197 attempts were 2026102713–2026105909, overlapping
2,940 of the first run’s seeds and replaying those expensive physics searches.
It consumed a further 24,114.62 s, retained 16 accepted scenes and all 3,197
journal rows, but was interrupted before its final report/manifest write. The
root cause was the resume cursor, not training or GPU capacity. The incomplete
continuation HDF5 SHA is
`4a7394345c576994eb7709c3d6696ab8c42689ca6881ab6aa018966886ee45a9`; journal
SHA `4c81fdbfbdbdd08baa883930da6e79dfab037a35947c285dc95c91ecacc47db6`.
Those 16 journal-accepted scenes are at IDs `placesphere-000..015`; each has
matching seed metadata, finite 7-D expert actions, T/T+1 39/42 observations,
native success, and full-prefix safe-success flags. They were task-unique against
the first 223 and the preregistered exclusion catalogs. They are retained as
candidate-scene provenance only, not as training demonstrations. Total valid
candidate-layout records recovered: **239/700**, not 223+16 discarded or
recollected.

Recovery manifest
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/recovered_candidates_239/manifest.json`
(SHA `89b180a4d2cd6e3685d4b7b1eec4a76b0312bf4c7917b5d3ed530251af246ba3`)
joins the 223 complete first-run records and 16 safely recovered continuation
groups without editing either source directory. All 239 pass the frozen
seed-or-both-endpoints-within-0.02-m identity rule. Their native source arrays
remain in the two original candidate HDF5s.

### Bounded resume repair and canary

The repaired cursor scans every prior attempt journal, including rejected
attempts, rejects overlapping seed histories, and resumes strictly at
`max(attempted_seed)+1`. For the two long-run journals this is seed
2026105910; the canary then consumed seeds 2026105910–2026105917, so the next
full-batch first seed is **2026105918**. A cheap preregistered proposal replay
uses the same `default_rng(seed)` start/goal draws and geometry checks before
opening PhysX. It skips only geometry-exhausted or task-identity duplicate
proposals; all other proposals continue through the unchanged expert physics
and post-execution safety predicate. The candidate generator now atomically
writes `recovery_checkpoint.json` each attempt with accepted scene IDs/layouts,
last/next seed, attempted count, wall, source identity and journal byte count.
Forensic reconciliation checks accepted journal rows against recovered HDF5
groups and fails closed on a count/identity mismatch.

The resume/canary registration is
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/candidate_resume_canary_preregistration_r2.json`
(SHA `a6a7cd357d6aaa214a6b75b30f2f8e1e4558209a1a35cbf6f5a4b380172e8ba8`). It
records the 2,940-seed overlap, all source/artifact hashes and the exact cursor.
The 8-proposal / 300-s canary ran 20.67 s: 5 known duplicates were rejected
before environment construction, 3 new proposals reached physics, 0 were
accepted (one direct-template failure and two side-column failures), and all 8
attempts were durably recorded. This proves fresh-seed progression and the
preflight/progress path; it does not establish a new expert acceptance rate.
Canary journal SHA
`bcdffec1bb21b7be66f3a4dd8f815a1b1900f423978ed46bf62a1699d4f58499`, report SHA
`e7ca605d33c6f16c854fe5f8dc9c806bc31262efd84b5773493a03bf716a05db`, recovery
checkpoint SHA `3f5f4bbfd0ec98f14642a8ffa19153a5f2198b7a311625ba1a4f41612f6b464b`.

The next bounded continuation plan is frozen at
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/full_candidate_continuation_preregistration_r3.json`
(SHA `3895c80446ee163fc0c32ef599944145b20487acc7565a35ee2eebfaac64dd67`). It
uses first seed 2026105918, targets the remaining 461 candidate layouts, and
has an 8,836-attempt / 65,000-s budget derived from 16 accepts in 265 unseen
continuation+canary seeds, plus a fixed 1,200-attempt buffer. There was **no
full continuation, matched 700-pair collection, dataset assembly, training,
checkpoint verification, or R25 paired evaluation** in this bounded repair
round. The original 1,000 DIRECT + 300 OBSTACLE hold-8 dataset and both R23
checkpoints remain unchanged. R25 gate status is therefore not evaluated; the
fixed R24 stride-2 gate remains the last measured result.

The above R25 continuation is **superseded** by the user's newer fixed 1,000
DIRECT / 500 OBSTACLE limit. No 461-layout continuation, candidate expansion,
or 1,000-obstacle training was launched. The following R26 section records the
active frozen-data continuation and its final development result.

Verification after the resume fix: focused recovery/profile suite **5 passed**;
full constrain suite **507 passed, 28 warnings, 617.70 s**; `compileall` and
`git diff --check` passed. The canary and test child processes exited; no
collector, trainer or evaluator process remained. No episode was selected using
policy metrics. Detailed source edits are confined to
`examples/baselines/constrain/`; hashes are in the preregistrations and
canary/checkpoint records above.

## Round 26 — frozen 1,000/500 matched-data continuation

The user's newest constraint is fixed at **1,000 DIRECT / 500 OBSTACLE** and
prohibits adding layout candidates. This supersedes the historical 1,000-obstacle
target and the R25 461-candidate continuation plan above. No layout proposal was
sampled in R26; only layouts in the recovered, immutable 239-candidate pool were
replayed. The branch remained `goal/transport-detour-optimization` at
`6aeee1ecf559aa9e247e3348ff11f308595e94fe`. No checkout, branch change,
commit, push, or ManiSkill-source edit occurred. The existing Lightnav process
was left untouched.

### Frozen candidate replay, external-identity correction, and final data

The frozen-pool replay registration is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/frozen_replay_selection_r5.json`
(SHA `42eba7cd617cc7d47b8fc31d3f9fdaf4a35f48a8b2b2877a091748fe562f96a6`). It
pins the recovered pool manifest (`89b180a4d2cd6e3685d4b7b1eec4a76b0312bf4c7917b5d3ed530251af246ba3`),
both candidate-source HDF5 hashes, replay code hashes, deterministic
`(seed, original_id)` ordering, margin 0.015 m, hold-8 terminal contract, and a
one-pass maximum of 239 existing layouts. The ordered first 200 candidates were
replayed serially in 1,289.89 s; all 200 matched direct/obstacle/negative/suffix
admissions and both positive T+1 restore audits passed. No expert/restore
rejection or timeout occurred. The separate candidate-obstacle staging HDF5 SHA
is `dd5de15459201edb275ada9002c557d84474e95acd0ad718d1ec3a58a86cbd8a`; its
manifest SHA is `35652421107ec68a5f62b800f59c24e63217404ae047354e123c95f5616c569a`.
The complete first-pass ledger SHA is
`4ac08310aa66cf82c1b01d852ebbf70f07349baa320452019a03f1facdd28721`.

An independent external task-identity audit then found four of those admitted
layouts conflicted with Ticket 04: candidate IDs **1, 80, 84, 159**. They were
excluded from training regardless of their positive expert outcome. Their prior
package and hashes remain preserved as non-training history; the correction
sidecar explicitly supersedes its stale admission claim. Before replay, a second
registration
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/replacement_replay_preregistration_r2.json`
(SHA `f97a60c43ca7893596f82987cbf81e0182927925b610ec9f08755fe02a236392`)
froze only the remaining existing pool IDs, external manifests, replacement
order, code hashes, and four-replacement target. Existing candidates **200–203**
were replayed in order in 46.92 s; all four passed. Candidate IDs **204–238**
remain unused and are not training data. No new layout was sampled. An initial
standalone candidate-0 state-restore smoke was retained separately as
`first_candidate_attempt.{json,h5}` and is not a second training demonstration.

The active audited package is
`demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/`.
The four Ticket-04-conflicting candidates are absent. The new OBSTACLE class
uses source candidate IDs 0–199 excluding 1/80/84/159, plus 200–203, mapped to
final HDF5 IDs **1300–1499**. Its final HDF5 SHA is
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`, manifest
SHA `aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`, and
training-identity SHA
`648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78`. The
independent saved-artifact audit
`independent_r26_audit.json` (SHA
`35bd68b10bd4139e0e2033394e69f936fa1a568e40a89897bf5088d7dbb7e4e3`)
verified 1,500 total episodes, 1,000/500 class counts, exact original IDs
0–1299 HDF5 group-value preservation, all 200 new T/T+1 native 7-D / 39-D / 42-D
contracts, first-native-success and no-truncation evidence, 400 positive role
T+1 restore proofs, 200 negative transport-block proofs, 200 actual-cylinder
terminal-suffix proofs, and full-prefix obstacle contact/crossing safety.
Final task-identity intersections are **0** against the original obstacle-300,
frozen dev30, and Ticket 04 sets. OBSTACLE layouts are **500 unique / 0 repeats**.
The retained DIRECT class has **474 greedy-unique / 526 repeated** layouts;
these repeats are historical DIRECT content, not new unique-layout claims.

The initial r2 assembled package remains unchanged but is **not** an active
training input because of the four Ticket 04 overlaps. Its correction record is
`training_package_r3/external_task_identity_correction.json` (SHA
`4d5b987011bbd3130c17da57cfb8766fb3d129ed8f88ab31da4d21a94d8abbc3`). Neither
the historical HDF5 nor its original manifest/training-identity hashes were
rewritten.

### Before-training window audit and control-only training

Before the training registration, the active 500-OBSTACLE identity reader loaded
exactly 500 IDs and **80,151 official windows**. The matched R23 300-OBSTACLE
source has 48,682 windows; the ratio is 1.646. Training uses the preregistered
150,000 updates (`90,000 × 500/300`), yielding 1,916.38 cohort-equivalent window
passes versus 1,893.10 for the old 90k/300 run (about 1.2% higher). Data volume
and update budget change together; this is not a separated causal data-volume
effect.

The final package's complete, non-truncated action/window audit is in
`dataset_validation.json` (SHA
`edd744337e4ee1cdc0a6d1a29788ec7277f9196e08dd972293d966f5b0182e89`). DIRECT /
OBSTACLE episodes longer than 200 actions: **64/1,000** and **44/500**; action
volume beyond step 200: **17.83% / 17.59%**. Open near-zero actions: **27.17%
/ 24.97%** of action volume. Fixed stride-8 open/near-zero action-only windows:
DIRECT **3,739/16,610**, OBSTACLE **2,055/9,859**. Separate linear-static counts
are **3,647/3,739** and **2,012/2,055**; angular-static counts are **0/3,739**
and **0/2,055**; both-native-static counts are zero. Uniform official-window
sampling was retained; no phase weighting, trajectory weighting, tail
truncation, policy-performance filter, or artificial neutral window was used.

The before-training registration is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/training_preregistration_r1.json`
(SHA `08f361e950586758283ebd7f472d264fbb77b2a1e5ea29e55047feffcebb7a7a`). It
pins all dataset/source/EMA/dev/baseline hashes, the exact 150,000-update
training and single stride-2 evaluation plan, exposure calculations, budgets,
and gate. The two-update CUDA preflight record is
`training_preflight.json`: finite losses, moved zero-convolution weights,
nonzero zero-convolution gradients, moved trainable control, bit-unchanged
Base, and zero disabled-control epsilon/sampler difference. No preflight
checkpoint was selected or used.

Exact training command (9000-s timeout):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 9000s \
  python -u -m examples.baselines.constrain.placesphere_constrain \
  --demo-path demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.json \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --run-dir outputs/exp/20261027_140000_placesphere_hold8_500_r26/training_control_150k \
  --checkpoint-dir outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k \
  --device cuda --total-iters 150000 --batch-size 1024 --seed 1 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 \
  --matched-data-identity demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/training_identity.json
```

Training completed 150,000 updates in **4,316.23 s**. Losses were finite
(0.02049344 → 0.00134501); frozen Base unchanged; control parameters and all
control-down/mid/hint gradient families moved; disabled-control epsilon and
100-step sampler maximum difference were both 0.0. Final EMA:
`outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt`,
SHA `ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71`.
Run-config SHA `a560490d198d7b62ab790556a05a243ba739beae4fe98a41518ccd29ea744489`;
training-summary SHA `7dd5ba8bc1ca5bcfc7440fd1256028d47dffe798be29bdbceeba43e3218365bc`.

### One paired stride-2 dev run

The sole evaluation was preregistered in
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/evaluation_preregistration.json`
(SHA `1b304218b462e6cde8ff63dab1463e8a2da51cb73920d7a3a19839fdb6513a81`). It
reuses the current frozen 30-scene dev states/controllers/observations/seeds,
stride 2, prediction horizon 16, executed action slice [1,9], 200 control steps,
300 s per episode, fixed denominator 30, and the R24 stride-2 baseline. No
stride grid, scene addition, checkpoint selection, or second paired run occurred.

Exact evaluation command (18,000-s outer timeout):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --constrain-checkpoint outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt \
  --output-dir outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2 \
  --primary-baseline outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval \
  --device cuda --replan-stride 2
```

Both policies completed **30/30** with zero runtime errors. Base: **8 task / 0
safe / 30 contact / 25 XY**. Constrain: **25 task / 25 safe / 0 contact / 0
XY**. Against R24 stride-2 Constrain (26/26/1/1), task improved/regressed/both/
neither = **2/3/23/2**, safe = **2/3/23/2**, contact cleared/introduced/both/
neither = **1/0/0/29**, XY = **1/0/0/29**. The five new failures are release/
stability timeouts, with no contact or crossing: scenes 8 (release step 155,
0.3740 m/s, 0.3204 rad/s, off-bin), 13 (109, 0.3341 m/s, 1.2531 rad/s,
off-bin), 15 (106, 0.3717 m/s, 0.2945 rad/s, off-bin; terminal on-bin but
spinning at 3.6033 rad/s), 17 (73, 0.3892 m/s, 0.3047 rad/s, off-bin), and 25
(117, 0.3183 m/s, 0.9206 rad/s, off-bin). Successful native steps were median
114, p90 173, max 179. Episode-level release/contact/terminal records and file
hashes are indexed in `paired_eval_index.json` (SHA
`e860a8e8721b4392e0ac18be708f85c0d19a3d8f4495d654e9ede75e428f48cd`). Paired
summary SHA `cac80011768efd2f94ed823cfcdef6646f4c01d0e1348abd9747f0fc95cef07e`;
frozen config SHA `e82bc5df0e70e4ca7bf9286d1cfe36fbfcedbe81f1d864020c1b00c7abb97d9d`.

**Gate: pause/fail.** The fixed requirements are task ≥27/30 and safe ≥26/30;
observed 25/30 for each. Contact and XY are 0/30 and pass. This is repeated
development-set evidence only, not a formal/generalization claim. No new-data
collection is authorized by this result. Continue only with optimization on the
frozen 1,000/500 data and the same dev set after parent review.

Final constrain suite: **512 passed, 28 warnings, 346.62 s**. `git diff --check`
passed. Training, replay, and evaluation children exited; the RTX 5070 returned
idle. Machine-readable command/budget/outcome and input/output hashes are in
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/round26_execution_record.json`
(SHA `acee5cade32864ac2af672fa72877117e63e85711c34d69cdf0147070b56ec97`).

After the data/training/evaluation run, all old R25 expansion/canary/461-layout
planning and execution entry points in `placesphere_obstacle1000_expansion.py`
were made fail-closed under the active freeze. No data, training, or evaluation
was rerun after that guard-only change. The R26 replay and replacement
registrations retain the exact source hashes used at those stages; the
post-run source transition and final verification are recorded in
`postrun_obsolete_candidate_guard_audit.json` (SHA
`deeea2e823295cecd9121f2e2d616fd8eec56adceb42870c140d8eab601ff1a8`). The final
post-guard test run is **512 passed, 28 warnings, 309.59 s**.

## Round 27 — frozen-data DDIM inference trial (preregistered)

The active authorization remains fixed at **1,000 DIRECT / 500 OBSTACLE**; no
new data, layouts, or trajectory collection is included. Round 27 registers
exactly one inference-only change for both already-trained policies: runtime
`DDIMScheduler`, 100 inference steps, `eta=0`, created from the loaded
training-time DDPM scheduler config (squared-cosine betas, epsilon prediction,
sample clipping). The training checkpoint contract remains DDPM. Reverse
variance draws are removed, so later global RNG positions also differ; this is
one sampler-configuration comparison, not an isolated variance-only causal
effect. Default DDPM adapter call paths remain unchanged.

The preregistration is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/inference_preregistration.json`.
It pins the exact r26 HDF5/manifest/training-identity, Base EMA and final
Constrain EMA hashes and paths, all 30 frozen dev scenes, R24 stride-2 baseline,
runtime source hashes, one 60-policy-episode paired budget, per-episode limits,
and the gate. Both policies retain stride 2, prediction horizon 16, executed
slice `[1,9]`, same saved state/controller/observation/seed reset, native 7-D
actions, and existing action mapping/clamps. Runtime errors remain in the fixed
denominator. The evaluator fails closed on any preregistered artifact/source
identity mismatch before creating an evaluation output or executing a scene.

The sole allowed command is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --constrain-checkpoint outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt \
  --output-dir outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval \
  --primary-baseline outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval \
  --device cuda --replan-stride 2 --sampler ddim \
  --inference-preregistration outputs/exp/20261027_204500_placesphere_ddim100_r27/inference_preregistration.json
```

Registration-time checks: focused sampler/PlaceSphere/pilot tests **64 passed**;
full constrain suite **517 passed, 28 warnings, 328.18 s**; CLI import/help and
`git diff --check` passed.

### Single paired run and result

The sole registered command completed normally from **2026-10-27 21:00:48.570
to 21:22:35.919 +08:00** (1,307.348 s, below the 18,000 s cap). It evaluated
**30/30 episodes for each policy**, retained zero runtime errors, and had 60/60
bit-exact scene restores with matching paired scene IDs/seeds. No episode was
dropped/retried. The RTX 5070 returned idle; no unrelated compute process was
started or stopped. The frozen data, training identity, and both checkpoint
hashes were identical before and after.

DDIM Base achieved **6 task / 0 safe / 30 contact / 24 XY**. DDIM Constrain
achieved **26 task / 26 safe / 0 contact / 0 XY**. Its 26 successful native
episodes had median / p90 / max success step **125 / 171 / 171**. Against R24
stride-2 Constrain (**26/26/1/1**), task and safe transitions were
improved/regressed/both/neither **3/3/23/1**; contact and XY were each
cleared/introduced/both/neither **1/0/0/29**. Three R24 failures (indices
17/23/25) cleared and three prior successes (6/11/14) regressed; index 9 failed
in both. All four new failures were incorrect releases, with no contact or
crossing: indices **6, 9, 11, 14**, release steps **136, 161, 78, 82**. Release
linear speeds were **0.389, 0.405, 0.425, 0.367 m/s**; every release was off-bin
and high above the bin (relative z **0.151, 0.155, 0.109, 0.140 m**). Per-step
contact/release/terminal mechanism traces are retained in their episode JSONs.

**Gate: fail/pause.** Task success 26/30 is one short of ≥27; safe 26/30 passes
≥26; contact 0 and XY 0 pass ≤3. This is frozen-development-set evidence only.
No formal 08 evaluation or new collection follows. Parent review selects the
next frozen-data optimization slice before any further experiment.

The machine-readable execution record is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/round27_execution_record.json`;
SHA `5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. It pins
the exact command/budgets/timestamps, all input and output summary hashes, the
64-file evaluation-tree digest, resource/cleanup state, comparisons, failure
indices/mechanisms, and gate transition. The full paired summary is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/paired_summary.json`
(SHA `ef24daa752f99ce31d5063325397c1276f3e6bc3d085dcaeafa824904378346e`); the
evaluation log SHA is `fed54e6d0256e7878b164a32a7fc6e776a0d5c0f5a23ddf999a9d3fb12a989cd`;
the paired-output tree SHA is
`6c47aab68cfe5947d0671b491a317e61355b422ee4d02559933fbecb15cd80c5`.

## Round 27 comparison correction — R26 is primary; R24 is secondary

**Correction to the Round 27 interpretation above:** the DDIM trial was compared
there to R24 as if it were the primary baseline. That was wrong. The required
primary is the immediately preceding **R26 DDPM100 stride-2** run, which uses
the exact same Base and Constrain checkpoints, frozen scenes, seeds, and rollout
limits. R24 has different data/checkpoint generation and is secondary context,
not evidence for this sampler comparison.

The actual R26 episode path is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2/`.
Its frozen config SHA is
`e82bc5df0e70e4ca7bf9286d1cfe36fbfcedbe81f1d864020c1b00c7abb97d9d`, paired
summary SHA `cac80011768efd2f94ed823cfcdef6646f4c01d0e1348abd9747f0fc95cef07e`,
and its registered R26 evaluation preregistration SHA is
`1b304218b462e6cde8ff63dab1463e8a2da51cb73920d7a3a19839fdb6513a81`. R26
records Base **8/0/30/25** and Constrain **25/25/0/0** (task/safe/contact/XY).
Its original command omitted a sampler override; the registered R26 evaluator,
Base adapter, and Constrain adapter source hashes establish the original
DDPM100 path. The R26 and R27 runs pin the same Base EMA SHA
`92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`, Constrain
EMA SHA `ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71`,
scene manifest SHA `9e1d8ee4786d2336f01e07d664ca37c3dcc7ca455e7876176def65c18b63c119`,
and scene HDF5 SHA `adc7d1b2a16745c459db53d2c8f818ac87d1ffffe21fd44c54509f77cef342d7`.
Both use 30 scenes, stride 2, prediction horizon 16, executed slice `[1,9]`,
200 control steps, and 300 s per-episode wall cap. The correction analysis
verified all per-policy scene IDs/seeds, bit-exact restores, and exact initial
policy observations across R26 and R27.

Correct primary transitions, ordered improved/regressed/both/neither for
task/safe and cleared/introduced/both/neither for contact/XY:

| Policy | R26 DDPM100 task/safe/contact/XY | R27 DDIM100 task/safe/contact/XY | Task | Safe | Contact | XY |
|---|---:|---:|---:|---:|---:|---:|
| Base | 8/0/30/25 | 6/0/30/24 | 3/5/3/19 | 0/0/0/30 | 0/0/30/0 | 3/2/22/3 |
| Constrain | 25/25/0/0 | 26/26/0/0 | 5/4/21/0 | 5/4/21/0 | 0/0/0/30 | 0/0/0/30 |

Constrain failures changed from R26 indices **8, 13, 15, 17, 25** to R27
indices **6, 9, 11, 14**. The corrected comparison therefore shows task and safe
success each increased by one (25→26), with five R26 failures resolved and four
previous successes regressed. Contact and XY remained zero in both. The fixed
gate still fails task ≥27 by one; safe ≥26 and both safety ceilings pass.
Base failure-index sets and both policies' complete metric transitions are in
the machine-readable correction result.

R24 remains preserved as secondary comparison: it reports 26/26/1/1 for its
Constrain checkpoint/data and its old R27 transition counts remain in the
original execution record, but are confounded for sampler inference. The
original R27 preregistration still names R24 as primary; it has **not** been
rewritten or represented as retrospectively corrected.

The read-only correction was preregistered separately at
`outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json`
(SHA `5594ff8af2a1831fbb78a8f3d60c1dcfd1699e57b34b1b2a2f43bda5d43ea8e9`). Its
result is `primary_comparison.json` (SHA
`b37577c8442f6ee1a1749c3fd2283df5670d6e9dc7710691fadd49f91cf78b26`). The
timestamped record hashes both 64-file episode trees and all source/input
registrations. A prior analyzer preflight stopped before computing transitions;
its log is preserved, and the successful analysis used the superseding
registration. The correction was analysis-only: no native evaluation, replay,
training, checkpoint/data mutation, or additional scene execution occurred.
The Round 27 execution record was appended with this correction, the R24
reclassification, provenance hashes, and corrected transition table; current
execution-record SHA is
`5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. The
bounded analysis regression tests passed: **2 passed**.

### R27 correction provenance rerun — immutable registered execution-record input

The first correction registration is retained unchanged at
`outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json`
(SHA `5594ff8af2a1831fbb78a8f3d60c1dcfd1699e57b34b1b2a2f43bda5d43ea8e9`), as is
its result (SHA `b37577c8442f6ee1a1749c3fd2283df5670d6e9dc7710691fadd49f91cf78b26`).
Its registered R27 execution-record input pinned SHA
`bc3483a62d0a1496b93d003cd73bb2dcdce1081c08c2d608b6aa90ff3187fd9d`, but the
same path was later appended with the correction and had SHA
`5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. A direct
rerun therefore failed its registered-input hash gate at
`r27_original_execution_record`; the earlier registration/result remain honest
historical records, not silently repaired artifacts.

The superseding registration is
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json`
(SHA `7d294ee9876420d5ef9d26bb0dc9575a05c5ab3ae43f63a9287652cb51da4249`). It
registers a read-only snapshot at
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/inputs/round27_execution_record.current.json`
(SHA `5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`),
copied byte-for-byte from the R27 record before adding the outputs-only pointer
to this rerun. The current mutable R27 execution record SHA is now
`a10abed8c1e96cb66e898397c3e0dfcd25b3654e3ad50e43709c47c814ec3562`; it is not
an analysis input. All other registered file and 64-file episode-tree hashes,
including the R26/R27 scenes and checkpoint identities, pass unchanged.

The reproducible CLI, run twice with exit 0 and byte-identical stdout SHA
`43d56e96dbe930ad98b47397db3117df831777fb00e4b2b58ad50ea24f6cda6b`, was:

```sh
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && timeout --signal=TERM --kill-after=15s 180s python -m examples.baselines.constrain.placesphere_r27_primary_analysis --registration outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json
```

The saved result is
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/primary_comparison.json`
(SHA `43d56e96dbe930ad98b47397db3117df831777fb00e4b2b58ad50ea24f6cda6b`). It
reproduces R26 Base **8/0/30/25** vs R27 Base **6/0/30/24**, and R26 Constrain
**25/25/0/0** vs R27 Constrain **26/26/0/0** (task/safe/contact/XY), including
the transition counts above. R24 remains secondary and the original R27
preregistration's R24-primary deviation is explicitly preserved.

The analyzer source SHA is
`149035f8fc8715219956f1f15350de96e04498f53f7f1116ed1b5360600012d6`; the
regression-test source SHA is
`691d8ae35b7b42fa7cbc60f4ca7f9f5bef81f365179e9515d34e7e8a6fb87ea8`. Focused
tests passed **4/4**, including reproducing the retained stale-pin failure,
accepting the real frozen snapshot, and rejecting post-freeze mutation. No
native episodes, replay, training, data, checkpoint, or scene were created or
changed.

## Round 29 — fixed four-plan DDIM ensemble (inference-only)

Continued on `goal/transport-detour-optimization`, HEAD/base
`a1dcc014fe38976e384610df1f933719672256de`; no checkout, commit, or push.
Training remains frozen at **1,000 DIRECT / 500 OBSTACLE**. No new demos, layouts,
weights, or ManiSkill source edits. One registered inference change was used for
both policies: four independent Gaussian `(16,7)` initial plans, identical latest
two-frame condition, 100-step DDIM eta 0 denoising in one batch, arithmetic mean
of all seven channels after denoising over `[1:9]`, then the existing action map
and first-two-action execution/replan. No sample ranking, clamping changes, or
route selection. Defaults remain one plan and existing DDPM call paths.

Registration: `outputs/exp/20261027_230000_placesphere_ensemble4_r29/`
`inference_preregistration_r2.json` (SHA
`25b4a8d1e34d52b7dd410f7546a4fd807182321f4592c31b5192744ca531ae49`). The
initial registration was retained at `inference_preregistration.json`; its first
launch stopped before output creation or any scene execution because the
ensemble-size guard expected the field under `protocol`. The corrected r2
registration moved that same fixed value there; the failed preflight log remains
at `evaluation_command.log` (SHA
`14aa56cb8b6341ca4ae48d62459a534bbe6733ab530723f21c883c60f33498ef`). The one
native trial then ran with the fixed 18,000 s outer limit, 300 s/episode, 200
control steps, serial CUDA inference and 30-scene denominator per policy. Exact
command and complete input/source identities are in `round29_execution_record.json`.

The immutable primary is R27 DDIM100 eta0 single-plan at
`outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/`, with identical
weights and frozen scenes. The evaluator preflight checked all 30 scene IDs,
seeds, bit-exact restore flags, checkpoint hashes, and equal initial 42-D policy
observations. After the run, all 30 initial observations per policy matched R27
again. All 60 episodes completed; 0 runtime errors, dropped episodes, or retries.
Elapsed wall time was approximately 1,260 s (22:02:59–22:23:59 local). Maximum
single synchronized ensemble inference time was 0.3745 s for Constrain and 0.4712
s for Base. The GPU returned idle after execution.

| Policy | Task | Safe | Contact | XY | Runtime errors |
|---|---:|---:|---:|---:|---:|
| Base | 3/30 | 0/30 | 24/30 | 23/30 | 0 |
| Constrain | 25/30 | 25/30 | 0/30 | 0/30 | 0 |

R27 Constrain was 26/26/0/0. R29 vs R27 transitions (task and safe are
improved/regressed/both/neither; contact and XY are cleared/introduced/both/neither)
were task **4/5/21/0**, safe **4/5/21/0**, contact **0/0/0/30**, XY **0/0/0/30**.
The five current Constrain failures are scenes 2, 12, 16, 19, 29: four
`incorrect_release` (2/12/16/29), one `not_stabilized` (19). The four incorrect
releases were off-bin; scene 19 released with 0.484 m/s linear speed and remained
unstabilized. There were no contacts or XY crossings. Base vs R27 was task
1/4/2/23, safe 0/0/0/30, contact 6/0/24/0, XY 5/4/19/2.

**Gate: fail/pause.** Task 25/30 is below 27 and safe 25/30 is below 26;
contact and XY ceilings pass at 0. No formal evaluation follows from this
repeated-development-set result. Keep the dataset frozen and diagnose the actual
release/stabilization regressions before selecting a next optimization slice.
This trial changes ensemble size, random draw consumption, and compute together;
it does not establish that stochastic dispersion caused the earlier XY release
errors or that action averaging preserves route topology.

The evaluator's unmodified raw `paired_summary.json` has legacy R24 comparison
labels and an old 90% gate even though the supplied primary directory was R27.
Those episode outcomes were not edited. The separately registered read-only
`primary_analysis.json` is authoritative for R27 transitions and the 27/26/3/3
gate. Analysis registration SHA
`55c1a454f4c5e8c6e0f72069f80d453d0f11fed1f598c7b4c41c2466b6c94e2d`; analysis SHA
`33fa28df35422b703c2880eb2980273cbec7dee3ef765109e70663023a7ee846`. R29 paired
summary SHA `459d169f4e22317a9056a8304302f212c98ddace90cd8da48c5abac85f204c09`.
The Base EMA, Constrain EMA, HDF5, manifest and training-identity hashes were
recomputed after the run and remain the registered R26 identities. The complete
run, failure list, paired transitions, hashes and cleanup record are in
`round29_execution_record.json`.

Verification after the changes:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s \
  python -m pytest examples/baselines/constrain/tests -q --tb=line
```

Result: **528 passed, 28 warnings, 82.59 s**. This includes a bitwise equality
check of the existing single-plan DDIM path against the pre-ensemble known-latent
reference loop. `git diff --check` passed. No
training, replay, or additional native episode was started after the registered
trial. Execution record SHA
`44ee060c7a29be94c9ab52007809a6c31be377634c49f071b831e3025dce4637`.

## Round 30 — fixed-range action representation optimization

Continued serially on `goal/transport-detour-optimization`, starting at
`60ccdf0572016d9d1090bb1382d54de9c80e11d9`; integration branch remains `dev`.
No branch switch, commit, or push occurred. This round used only the immutable
R26 dataset with exactly **1,000 DIRECT / 500 OBSTACLE** episodes; it generated
no demonstrations, candidate layouts, or seeds. ManiSkill and other baseline
source files were not changed. Experiment artifacts are under
`outputs/exp/20261027_224727_placesphere_action_codec_r30/` and the new model
weights are under `outputs/ckpt/20261027_224727_placesphere_action_codec_r30*/`.

### Full-data action audit and codec contract

Before registration, a full read-only audit checked the frozen HDF5 and
manifest-selected class IDs. Its output is `action_range_audit.json` (SHA
`961915a1193fa60fa01e31b97e25a5e919f1ecb0682cdff2812de173966dc01f`). The data
identity remains HDF5
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`, manifest
`aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`, and training
identity `648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78`.
All 136,417 DIRECT and 80,651 OBSTACLE native actions fit pose `[-0.35,0.35]`
and gripper `[-1,1]`; no value was clipped or changed. The pose cap is in the
native normalized `pd_ee_delta_pose` controller coordinates, **not meters or
radians**. Selected audit values:

| Class | Channel | Native RMS | Exact zero fraction | Absolute q90 | Absolute q99 | Absolute max |
|---|---|---:|---:|---:|---:|---:|
| DIRECT | delta_x | 0.12710 | 0.33796 | 0.20000 | 0.35000 | 0.35000 |
| DIRECT | delta_y | 0.08169 | 0.33766 | 0.12983 | 0.35000 | 0.35000 |
| DIRECT | delta_z | 0.12814 | 0.33773 | 0.34092 | 0.35000 | 0.35000 |
| DIRECT | delta_euler_x/y/z | 0.02889 / 0.04644 / 0.19305 | 0.55026 each | 0.01249 / 0.05737 / 0.35000 | 0.14666 / 0.23929 / 0.35000 | 0.35000 each |
| DIRECT | gripper | 1.00000 | 0 | 1.00000 | 1.00000 | 1.00000 |
| OBSTACLE | delta_x | 0.11137 | 0.30602 | 0.20000 | 0.35000 | 0.35000 |
| OBSTACLE | delta_y | 0.12018 | 0.30575 | 0.20000 | 0.35000 | 0.35000 |
| OBSTACLE | delta_z | 0.11811 | 0.30577 | 0.29482 | 0.35000 | 0.35000 |
| OBSTACLE | delta_euler_x/y/z | 0.02663 / 0.04248 / 0.17738 | 0.61863 each | 0.01044 / 0.04851 / 0.35000 | 0.13141 / 0.20760 / 0.35000 | 0.35000 each |
| OBSTACLE | gripper | 1.00000 | 0 | 1.00000 | 1.00000 | 1.00000 |

For each class, the audit also records signed min/max, encoded RMS/variance,
zero fractions, and channel-wise forward-diffusion SNR at timesteps 0/10/25/50/
75/99. The SNR is descriptive per-channel variance under the standard noising
process, not a causal claim. At timestep 50, encoded pose-channel SNR ranges
from 0.0053–0.1903 DIRECT and 0.0053–0.1725 OBSTACLE, while the gripper is
0.8593 / 0.9075; this exposes a substantial scale imbalance motivating the
fixed controller-range encoding.

The opt-in, versioned profile is
`placesphere_controller_range_v1`, scale `[.35,.35,.35,.35,.35,.35,1]`, zero
offset. It encodes native action targets before the unchanged epsilon/DDPM
objective and decodes model outputs to native seven-channel controller actions
before the existing action map/clamp. Observation identity normalization,
39-D base observation, 42-D hint input and hint normalization, model channels,
sampling schedule, padding, gripper behavior and environment scaling are
unchanged. Native HDF5 arrays remain untouched. Identity-profile historical
checkpoints and their default inference path remain strict/bit-exact.

Training registration
`training_preregistration.json` (SHA
`eb4a824224f649881a3511dc229d751d38c926255820d16c2016839e14f549a5`) pins the
frozen IDs/hashes, codec profile, exact 30k Base + 150k Constrain final-EMA
models, source hashes, preflight outputs, budgets, R27 reference and 27/26/3/3
gate before long training. Two-update CUDA preflights passed for Base and
Constrain; disabled-control epsilon and 100-step sampler matched the new Base
exactly (max difference 0). The control branch moved, all losses were finite,
all recorded gradient families were nonzero, and the frozen Base stayed
unchanged.

### Matched training and metadata correction

Base trained for 30,000 DIRECT updates in **900.52 s**: loss 1.160607 →
0.003348 (min 0.002550, max 1.160881). Initial final-EMA checkpoint
`outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base/final.pt`, SHA
`08879f41d1524a30f5aba9bfaaf2f107a5417b5fdc4d7463c8e274c7edda41cc`.
Constrain trained for 150,000 OBSTACLE updates from this matched Base EMA in
**4,217.93 s**: loss 0.082727 → 0.002647 (min 0.001801, max 0.115430). Initial
Constrain final EMA
`outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_150k/final.pt`,
SHA `476d582a5993d55a80382dffec24fe7059bf3621b0ed2bede9a2f74166c345ff`.

The first evaluation stopped before creating an environment or running any
scene because strict validation caught a Base checkpoint metadata defect: its
actual R26 HDF5/manifest/training identity and selected 1,000 DIRECT IDs were
correct, but its `dataset_version` field retained the older R23 constant. Both
original checkpoints were preserved. A hash-guarded correction created
metadata-only copies after verifying the Base EMA, Constrain model and Constrain
EMA tensors remained unchanged. The correction record is
`checkpoint_metadata_correction.json` (SHA
`bf6942e8771ef53de6edac66dac295f7c4810b44c07c1e8227a135f0c218457c`). Corrected
Base EMA SHA `491970382f3ed8dac7ed88ad1ab3f5cd23fb2d29065291a507ff766c75a6f071`;
corrected Constrain SHA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`. The
trainer now records the matched identity's dataset version directly, and runtime
loaders pin the full R26 data identity and exact direct/obstacle ID lists.

### One R27-paired DDIM evaluation and authoritative gate

Evaluation preregistration `evaluation_preregistration_r3.json` (SHA
`a0ec9488355b68912b04aa18e620bb360ceffaaf019865bf0ae8e95d5be70075`) registered
the corrected checkpoints before any native episode. The sole evaluation ran
DDIM100 eta 0, single plan, stride 2, slice `[1:9]`, 200 control steps, 300 s per
episode, 30 fixed canonical scenes per policy; elapsed time was approximately
612 s. It completed 60/60 episodes with 0 runtime errors, drops or retries and
bit-exact scene restores. The primary was the immutable R27 DDIM100 single-plan
run at `outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/`; scene
IDs, seeds and initial 42-D observations matched per policy. The comparison is
end-to-end because both policies' weights changed to the new codec; it does not
isolate Constrain-only attribution.

| Policy | Task | Safe | Contact | XY crossing | Runtime errors | Successful steps median / p90 / max |
|---|---:|---:|---:|---:|---:|---:|
| Base DP | 4/30 | 0/30 | 29/30 | 27/30 | 0 | 124.5 / 137 / 137 |
| Constrain | 29/30 | 29/30 | 1/30 | 1/30 | 0 | 106 / 154 / 170 |

Constrain failed only scene index **3** (`placesphere-003`), a 200-step timeout
with release/stabilization contact; its full release/contact/crossing trace is
in the episode record. Against R27 Constrain 26/26/0/0, task and safe
improved/regressed/both/neither are **4/1/25/0**; contact and XY
cleared/introduced/both/neither are **0/1/0/29**. The fixed gate passes:
**29 task, 29 safe, 1 contact, 1 XY** against thresholds 27/26/3/3. The new
Base result is 4/0/29/27 task/safe/contact/XY. Paired channel-wise encoded and
predicted-native actions were captured on every executed trace action and
round-trip verified (3,542 Constrain and 5,700 Base actions).

The raw paired summary uses the explicit protocol
`placesphere_action_codec_r30_primary_r27_v1` and labels R27 as the primary; it
does not reuse R29's legacy R24 comparison or gate. The separately registered
authoritative analysis has immutable byte-for-byte R27 and R30 evaluation
snapshots at `analysis_inputs/`, registration
`analysis_registration_r2.json` (SHA
`dabd5a8d7900961f04ff9647615aa2350e2d115a1a615774bc9326bb72466b06`), and result
`primary_analysis_r2.json` (SHA
`45e54c4f921893c29314148d22523c2904ba87415fa1be2ac8f8d21afe059342`). The
machine-readable command, budgets, attempt history, checkpoint provenance,
results and cleanup status are in `round30_execution_record.json`.

Final constrain suite: **537 passed, 28 warnings**; `compileall` and
`git diff --check` passed. GPU returned idle and experiment processes exited.
This is a repeated-development-set result, not formal/generalization evidence.
The gate permits **parent review before formal08 preparation only**; no formal
08 evaluation or new scene/data run was started.

## Round 67 — terminal random-500 50k paired entry-bank observation

Fresh registration and source archive:
`outputs/exp/20261119_063749_peg_entry_50k_comparison/`. This observation
compares Base, random-500 update-30k, and the preregistered terminal random-500
update-50k checkpoint on the existing 60-case pose-window initialization bank.
The historical bank directory and contents were not changed. Three inference
draws per case produced **180 unique paired groups / 540 arm outcomes**. Native
execution completed exit 0 in 3,446.54 s with **155,029 / 334,800** registered
physical controls. Independent reconstruction verified all 180 complete
groups, source episode/seed joins, exact handoff-state equality, and unchanged
registered checkpoint/input/source hashes. Results and the full condition,
seed, paired-transition, and control tables are in `results_summary.md` and
`independent_audit.json`.

Successes: Base **73/180**, 500/30k **85/180**, 500/50k **84/180**. Terminal
50k minus 30k is **−0.6 percentage points** overall; by profile it is random
**28/90 vs 25/90 (+3.3 pp)** and fixed **56/90 vs 60/90 (−4.4 pp)**. Across
conditions the 50k-minus-30k deltas vary in sign; paired results are descriptive
repeated draws from six origins per profile (12 total), not 180 independent
scenes. This is a terminal-checkpoint comparison, not a pure update-count
causal/scaling claim: the continuation schedule changed during the additional
20k. No intermediate checkpoint evaluation, checkpoint selection, training,
bank regeneration, ManiSkill source edit, or commit was performed. The
post-run focused suite passed **141 tests**.

## PlugCharger Ticket 07 Round 69 review blockers — development only

Corrected actual receptacle orientation (preserving sampled center) and custom
state restoration for both `(4,)` and `(1, 4)` target records. Replay now uses the
verified public replay output naming rule because `replay_trajectory.main()`
returns `None`; replay controls share a stage counter across created environments.
Behavioral regression tests plus the required 16-file Peg suite passed **151
 tests**. Raw suite log and Round 69 code snapshots are in
`outputs/exp/20261119_095824_plug_orientation_round69/`. No native work ran.
Important: external TERM/KILL supervision and one wall-clock budget spanning
multi-stage orchestration are still absent; this is not authorization or a
native-ready run protocol. Tickets 08/09 remain blocked.

## PlugCharger Ticket 07 Round 77 — reset/state/live-expert isolation

The preregistered A/B/live comparison completed both fixed cases in the order
reset-only, restore-source-row-0-once then replay the exact source joint-action
stream, and a fresh official motion-planning solve. It used **1,281 controls**
within the per-case 600, aggregate 6,000, and outer 1,800 s TERM / 20 s KILL
caps. For source episode 0, both open-loop arms and the live solve succeeded
(222/222/226 controls). For episode 2, both open-loop arms failed at 17.737 mm
goal distance after 203 controls, while the fresh official live solve succeeded
at 1.961 mm after 205 controls. Captured common pre-action physical and
controller fields match between A and B in each case. Source-row-0 restoration
does not explain this pair's outcome. Fresh live execution differs, but generated
trajectory, planner stochasticity and execution pathway were not isolated; its
mechanism remains unresolved. This does not establish hidden buffer identity.

The fresh expert outputs are diagnostic-only. The historical source bundle is
unchanged; Ticket 07 is **not passed**, and no training was started. The new
online expert records require a separate replay/acceptance audit before any data
admission. Registration, source archive, traces, outputs and full discussion:
`outputs/exp/20261120_000000_plug_reset_live_diagnosis_r77/`. Required Peg+Plug
suite initially reported 160 passes using a noncanonical selection. Independent
review ran the correct 17 modules: **171 passed in 13.55 s**, raw output retained
as `reviewer_canonical_17_suite.raw.log`. Focused Plug suite: **30 passed**.
For each A/B pair, captured initial snapshots differ only by at most 1.49e-8 in
one exposed qpos scalar (state-restore roundoff); outcomes are identical.

## PlugCharger Ticket 08 — Goal-loop Round 92 fixed-data production collection

- Dataset: `demos/PlugCharger-orientation-fixed-1000_20261119_184307/`.
- Collection logs, source archive, independent data admission, pre/post regression
  tests, timing and catalog audit:
  `outputs/exp/20261119_184307_plug_orientation_collection_envelope/`.
- Native collection registration/results:
  `outputs/exp/20261119_184307_plug_orientation_collection/` (1,066 ordered
  attempts; first 1,000 eligible accepted; 2,581.512 s outer wall; exit 0).
- Frozen reset-only catalog set:
  `outputs/exp/20261119_184307_plug_orientation_catalogs/` (50 fixed validation,
  50 random validation, 100 random final test, 100 fixed Base test/retention).
- Dataset HDF5 SHA-256:
  `d909986dd2319f41e5a3e098b13744cb6eda5e3caf2b779d66fe3340165d11c1`;
  JSON SHA-256:
  `190dd31f0787b58871fe4963e36931ab65c33ab7b3eb05a6dd160a29a8b8088f`;
  catalog-set SHA-256:
  `3f75e6ec5d3729e142e87117e34bfd00a39f921283c21000486e30800f34481f`.
- Independent admission passed all 1,000 source/replay pairs and the official
  dataset reader (279,625 windows). All 300 reset/restore observations are
  finite 46-D and restore within 1.1921e-7; all target-yaw roundtrips match
  within 1.45e-8 rad. Required 22-file test suite passed
  216 before and after; source hashes were unchanged. No training/checkpoint or
  policy evaluation was run. Ticket 08 remains in progress; exact next Base
  command and gates are recorded in `PLUG_ORIENTATION.md` Round 92.

## PlugCharger Ticket 08 — Goal-loop Round 95 saved-validation diagnosis

- Offline failure-phase analysis of the unchanged 250 traces from all five fixed
  validation checkpoints is in
  `outputs/exp/20261119_220651_plug_base_validation_failure_analysis/`.
- The bundle contains per-episode/per-control geometric landmarks, same-control
  position/quaternion errors, 80k-vs-100k paired seeds, event/input hashes,
  execution/source hashes, and the exact 23-file regression log. Result:
  exploratory post-hoc only; native controls/model queries/training were zero.
- Dominant failure signature: aligned near-goal position stall around −16 mm on
  world X for 20/23 (60k), 19/20 (80k), and 21/23 (100k) failures. Missing contact
  force/grasp telemetry prevents causal attribution. The 80k selection and 40/50
  fixed-validation gate remain unchanged; neither test catalog was opened.
- Full phase breakdown and the separately proposed, unexecuted next diagnostic are
  recorded in `PLUG_ORIENTATION.md` Round 95 and the bundle's `diagnosis.md`.
