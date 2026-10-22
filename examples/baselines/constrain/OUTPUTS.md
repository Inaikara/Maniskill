# Output layout and retained artifacts

Training demonstrations and their collection/validation metadata live under
`demos/<dataset-name>/`. Formal obstacle datasets are `demos/StackCube-obstacle/`
and `demos/PickCube-obstacle/`. PlaceSphere pilot artifacts live under
`demos/PlaceSphere-obstacle/<timestamp>_pilot/` and remain separate from formal
collection until Ticket 08 freezes that task's contract.
`outputs/` has two purpose-specific directories:

- `ckpt/`: model weights, separate from training logs.
- `exp/`: training logs, fixed evaluation scenes, results, videos and verification.

New experiment directories start with `YYYYMMDD_HHMMSS_`. Retained historical
runs use the known date as `YYYYMMDD_`; no unknown start time is invented.
The Base DP date below comes from its TensorBoard event timestamp.

## Current locations

```text
demos/
├── StackCube-obstacle/
├── PickCube-obstacle/20261022_120016_pilot/  # retained incomplete attempt
├── PickCube-obstacle/20261022_130058_fresh_scene_diag/  # fresh-scene restoration check
├── PickCube-obstacle/20261022_130157_contact_link_diag/  # 1 direct + 2 obstacle attempts
├── PickCube-obstacle/20261022_130242_contact_phase_diag/  # contact link/step/phase evidence
├── PickCube-obstacle/20261022_130356_layout_clearance_diag/  # one safe obstacle, one prefilter rejection
├── PickCube-obstacle/20261022_130421_pilot/  # complete 20 + 20 pilot
├── PickCube-obstacle/20261022_153000_pilot_diag/  # interrupted diagnostic (bug found)
├── PickCube-obstacle/20261022_153200_pilot_diag/  # 1 direct + 3 obstacle attempts
├── PickCube-obstacle/20261022_153500_pilot_diag/  # interrupted diagnostic (bug found)
├── PickCube-obstacle/20261022_153700_pilot_diag/  # 1 direct + 6 obstacle attempts
├── PickCube-obstacle/20261022_154000_pilot_diag/  # 1 direct + 20 obstacle attempts
├── PickCube-obstacle/20261022_154500_pilot_diag/  # 1 direct + 6 obstacle attempts
├── PickCube-obstacle/20261022_160000_pilot_diag/  # 1 direct + 6 obstacle attempts
├── PickCube-obstacle/20261022_160200_pilot_diag/  # 1 direct + 6 obstacle attempts
├── PickCube-obstacle/20261022_160500_pilot_diag/  # 1 direct + 6 obstacle attempts
├── PickCube-obstacle/20261022_132753_formal/  # Ticket 02 formal 1,000 + 100 collection
└── PlaceSphere-obstacle/
    ├── 20261022_171100_pilot/  # invalid diagnostic seed; no complete attempt
    ├── 20261022_171300_pilot_calibration/
    ├── 20261022_171500_pilot_calibration/
    ├── 20261022_171700_pilot_calibration/
    ├── 20261022_172000_pilot_calibration/
    ├── 20261022_172200_pilot_calibration/
    ├── 20261022_172400_pilot_calibration/  # seven recorded calibration attempts
    ├── 20261022_172910_pilot/  # preflight stopped before first attempt
    ├── 20261022_173015_pilot/  # preflight stopped before first attempt
    └── 20261022_173110_pilot/  # complete Ticket 07: 20 + 20

outputs/
├── ckpt/
│   ├── base_dp_stackcube_980/best_eval_success_at_end.pt
│   ├── constrainnet_wristcam_seed1/final.pt
│   ├── 20261022_144000_pickcube_base_dp/final.pt
│   └── 20261022_144835_pickcube_constrain/final.pt
└── exp/
    ├── 20260930_base_dp_stackcube_980_seed1/
    ├── 20261021_constrainnet_wristcam_seed1/
    ├── 20261021_wristcam_scenes/
    ├── 20261021_wristcam_paired_eval/
    ├── 20261022_132753_pickcube_formal/  # 1,000 + 100 formal data and restore manifest
    ├── 20261022_151350_pickcube_scenes_smoke/  # one-scene generation/restore smoke incl. bounded failures
    ├── 20261022_152157_pickcube_scenes/  # Ticket 05: 100 fresh expert-accepted fixed PickCube obstacle scenes
    ├── 20261022_155725_pickcube_paired_eval/  # retained failed preflight: invalid state-fingerprint normalization, no policy actions
    ├── 20261022_160037_pickcube_paired_eval/  # retained failed attempt: Base DP condition and GPU video conversion errors
    ├── 20261022_162002_pickcube_paired_eval/  # Ticket 06: frozen Base DP / ConstrainNet paired evaluation
    ├── 20261022_143200_pickcube_base_dp_probe/  # bounded 45-second batch/memory probe
    ├── 20261022_144000_pickcube_base_dp/  # 30,000-update Base DP training
    ├── 20261022_161500_pickcube_base_dp_rollout/  # short interface rollout, not evaluation
    ├── 20261022_144800_pickcube_constrain_probe/  # separate 45-second non-formal throughput probe
    ├── 20261022_144835_pickcube_constrain/  # 30,000-update ControlNet run, logs/config/verification
    ├── 20261022_150600_pickcube_constrain_rollout/  # 32-step real-scene interface rollout, not evaluation
    ├── 20261022_171100_placesphere_pilot/  # invalid-seed diagnostic
    ├── 20261022_171300_placesphere_pilot_calibration/
    ├── 20261022_171500_placesphere_pilot_calibration/
    ├── 20261022_171700_placesphere_pilot_calibration/
    ├── 20261022_172000_placesphere_pilot_calibration/
    ├── 20261022_172200_placesphere_pilot_calibration/
    ├── 20261022_172400_placesphere_pilot_calibration/
    ├── 20261022_172910_placesphere_pilot/  # preflight failure; zero complete attempts
    ├── 20261022_173015_placesphere_pilot/  # preflight failure; zero complete attempts
    └── 20261022_173110_placesphere_pilot/  # Ticket 07 validation, logs, hashes and videos
```

The ConstrainNet experiment retains its loss log, provenance, run configuration,
training summary and independent verification. The scene set retains its
standalone `scenes.h5`, manifest, round-trip evidence and expert-success
trajectories. The paired evaluation retains both policies' 100 episode records,
manifests, summaries, videos and final audit. Superseded exploratory profiling
outputs and the large intermediate scene-template report were removed during
cleanup; listed PickCube diagnostic and pilot runs are retained.

## PickCube pilot attempts

`demos/PickCube-obstacle/20261022_120016_pilot/` retains the historical 20 accepted direct
trajectories and the complete attempt report. The obstacle phase exhausted its
200-attempt budget (197 cylinder-footprint prefilter rejections and 3 exact
initial-state restoration mismatches); it did not meet the ticket's 20-obstacle
acceptance target and is not a usable obstacle dataset. The data, contract,
configuration, report and artifact hashes are retained for traceability.
Its validation diagnostics and saved-batch model check are
under `outputs/exp/20261022_120016_pickcube_pilot/`. Round-8 code changes and
bounded live diagnostics are retained. The round-9 resolution and measured
failure evidence are documented in
`examples/baselines/constrain/PICKCUBE_PILOT.md`. Those attempts accepted no
obstacle examples and remain immutable historical evidence.

The complete round-9 pilot at `demos/PickCube-obstacle/20261022_130421_pilot/`
contains 20 accepted direct and 20 accepted obstacle trajectories (68 complete
obstacle attempts). It includes configuration/contract, per-attempt and
aggregate timings/counters, trajectory, report, and artifact hashes. Validation,
minibatch evidence, and both representative videos are under
`outputs/exp/20261022_130421_pilot/`, with a video/validation identity record in
`diagnostics/pilot_evidence.json`. The saved official epsilon-loss is finite;
see `PICKCUBE_PILOT.md` for restoration and safety diagnosis.
Round-10 retrospective source/recipe provenance and per-trajectory base XY
side audit are new sidecars in that same dataset directory; they preserve the
identities and historical status of the original collection metadata.
The separate Ticket 02 formal run (1,000 direct + 100 safe obstacle accepts,
pilot excluded) is documented with exact commands and contract in
`PICKCUBE_DATA.md`; its frozen source/config snapshot precedes collection.
Collection took 1,503.52 s. All 1,100 episodes passed full native-state,
controller-state, observation, terminal/safety, and training-window validation;
the manifest and detailed evidence are under
`outputs/exp/20261022_132753_pickcube_formal/validation_full_contract/`.

Ticket 03's independent PickCube Base DP, full configuration, complete update
log, checkpoint and short interface rollout are documented in
`PICKCUBE_TRAINING.md`. The 1,000 direct demonstrations alone were used; the
100 obstacle examples were filtered out by the frozen episode manifest. The
full 30,000-update run completed in 896.09 seconds and its final EMA checkpoint
is stored separately under `outputs/ckpt/`.

Ticket 04's task-specific PickCube ConstrainNet uses exactly the 100 formal safe
obstacle episodes (IDs 1000–1099) and the trusted Ticket 03 final EMA. Its
complete configuration, 30,000 finite per-step losses, independent-base and
control-update evidence are under
`outputs/exp/20261022_144835_pickcube_constrain/`; the standalone final EMA is
under `outputs/ckpt/20261022_144835_pickcube_constrain/`. The separate bounded
throughput probe and 32-step real-scene rollout are retained in the listed
experiment directories. Reproduction details, full artifact identities, exact
condition/action contract, rollout outcome and verification are in
`PICKCUBE_TRAINING.md` under Ticket 04.

Ticket 05's independently generated 100-scene PickCube obstacle catalog,
attempt/failure and phase timing report (100/349 actual runner invocations;
the immutable original report says 100/350 loop iterations and is corrected by
an accounting sidecar), native controller/runtime identity,
strict fresh-environment restoration audit, public load/restore API, exact
reproduction command and artifact hashes are documented in
`PICKCUBE_SCENES.md`. The complete run accepted 100/349 actual runner
invocations in 304.33 s; the original 100/350-loop-iteration report remains
unchanged and the corrected denominator is hash-pinned by its sidecar.
all 100 restores passed. The small smoke run and its bounded failed attempts
are retained separately for the implementation check.

Ticket 06's first frozen runner attempt at
`outputs/exp/20261022_155725_pickcube_paired_eval/` completed 0 policy actions;
all 200 records capture a state-fingerprint normalization error. The native
state snapshots were hashed without flattening the vector-env batch dimension,
matching the scene catalog's identity contract. That failed attempt is retained
unchanged. The corrected formal attempt is at
`outputs/exp/20261022_160037_pickcube_paired_eval/` retained 100 Base DP
condition-shape errors and three ConstrainNet video conversion errors; the 97
other ConstrainNet episodes were recorded, and all files remain unchanged. The
next attempt corrects those runner interface issues without changing scenes or
checkpoints and is at `outputs/exp/20261022_162002_pickcube_paired_eval/`.

Ticket 06's PickCube-only paired evaluation uses the two recorded final EMA
checkpoints on that immutable 100-scene catalog. Run configuration, complete
per-scene outcome/action records, restoration fingerprints, fixed-index videos,
per-policy summaries and paired comparison are under
`outputs/exp/20261022_162002_pickcube_paired_eval/`; reproduction and metric
definitions are in `PICKCUBE_EVALUATION.md`. Both policies completed 100/100
records with no runtime errors. Base DP/ConstrainNet task success was 82/83,
safe success 2/38, obstacle contact 97/53, and XY crossing/timeouts 0/0 for
each. A `termination_correction.json` sidecar corrects the 18/17 historical
500-step episodes to control-step-limit outcomes (not environment terminals);
the original episode records and summaries remain byte-identical and their
hashes are pinned in the sidecar. Paired task outcomes were 71 both, 11
Base-only, 12 Constrain-only, 6
neither; paired safe outcomes were 1 both, 1 Base-only, 37 Constrain-only, 61
neither. All full-state/controller/observation restore fingerprints matched
across policy pairs and observations matched the catalog.

Bounded calibration records are retained at
`demos/PickCube-obstacle/20261022_115754_pilot_calibration/` and
`demos/PickCube-obstacle/20261022_115841_pilot_calibration/`; the second run's
diagnostics are under
`outputs/exp/20261022_115841_pickcube_pilot_calibration/`. Their directory
timestamps were corrected to the recorded local start times without changing
artifact bytes.

## Relocation map

Completed artifact bytes are unchanged. Paths embedded in original checkpoints,
manifests, logs and audit records identify their historical location; resolve
them with this map instead of rewriting them and invalidating their hashes.

| Historical path | Current path |
| --- | --- |
| `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt` | `outputs/ckpt/base_dp_stackcube_980/best_eval_success_at_end.pt` |
| `outputs/runs/dp_StackCube-v1_980_1/` (training records) | `outputs/exp/20260930_base_dp_stackcube_980_seed1/` |
| `outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt` | `outputs/ckpt/constrainnet_wristcam_seed1/final.pt` |
| `outputs/constrain_runs/wristcam_full30000_seed1/` (training records) | `outputs/exp/20261021_constrainnet_wristcam_seed1/` |
| `outputs/wristcam_formal_20261021/` | `demos/StackCube-obstacle/` |
| `outputs/data/wristcam_formal_20261021/` (intermediate location) | `demos/StackCube-obstacle/` |
| `outputs/wristcam_eval_scenes_20261021/` | `outputs/exp/20261021_wristcam_scenes/` |
| `outputs/wristcam_paired_eval_20261021_recovery/` | `outputs/exp/20261021_wristcam_paired_eval/` |

The more specific checkpoint mappings take precedence over their parent-directory
mapping. Historical paths for deleted intermediate outputs have no replacement.

### Identity checks (SHA-256)

| Artifact | SHA-256 |
| --- | --- |
| Base DP checkpoint | `ba65c507ac48aa76a911fa1af28b7155ca3c4f2d7393c775930241dbcde27c25` |
| ConstrainNet final EMA checkpoint | `74de460532bf4188b7b019f9bb15ecd027af2a7a3b5bef14c2fc6fd4c59b7d02` |
| Formal `trajectory.h5` | `c4811f0d1069fdc4bd247770978a6f87423b9404cfddc04567db456b0ce734c8` |
| Scene `scenes.h5` | `b1c736ccfbd4411b20dcb1495ffbf220afd6d7b08dbf46528c4047358d139caf` |
| Scene `manifest.json` (file bytes) | `71381bfaad5c7e36235f457b3489e30d96f508c8d7bb17db037fae34eec65c89` |
| Evaluation `paired_summary.json` | `f4714253e9ac8980fa4f3e7599ac54f82abc1af9ab417c019372dce1597c733d` |

## New runs

Activate `maniskill`, then pass explicit paths. For example:

```bash
python -m examples.baselines.constrain.train_constrain \
  --demo-path demos/StackCube-obstacle/trajectory.h5 \
  --base-checkpoint outputs/ckpt/base_dp_stackcube_980/best_eval_success_at_end.pt \
  --run-dir outputs/exp/YYYYMMDD_HHMMSS_constrainnet_training \
  --checkpoint-dir outputs/ckpt/constrainnet_training
```

Replace the timestamp placeholder before running. Training defaults also split
weights and logs; evaluation defaults use timestamp-prefixed experiment paths.
The frozen demonstration collector is called with an explicit `--output-dir`
under `demos/<dataset-name>/`, preserving the recorded generator source fingerprint.

## PlaceSphere Ticket 07 pilot

`demos/PlaceSphere-obstacle/20261022_173110_pilot/` is the complete 20 direct +
20 safe-obstacle pilot. It used 30 direct and 31 obstacle complete attempts in
128.984 seconds; details, per-attempt diagnostics, stage counters, phase timings,
seeds, T/T+1 state/controller/info evidence, source/native pins, and SHA-256
identities are in its report and `PLACE_SPHERE_PILOT.md`. The matching
`outputs/exp/20261022_173110_placesphere_pilot/` contains the frozen contract and
run log, both class minibatch finite epsilon-loss evidence, and the direct and
obstacle representative videos. The trajectory hash is
`0a415d203ee023368c357bcb34c9173128633230cca210ac9abf6f562aca8220`; video
hashes and class-specific losses are recorded in `pilot_evidence.json`. No
formal PlaceSphere dataset collection or training was run.
`post_collection_source_delta.json` documents the one later configuration
validation guard and preserves the pre-collection generator hash; it did not
change the dataset or the recorded pilot contract.
