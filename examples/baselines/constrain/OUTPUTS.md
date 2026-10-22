# Output layout and retained artifacts

Training demonstrations and their collection/validation metadata live under
`demos/<dataset-name>/`. The formal obstacle dataset is `demos/StackCube-obstacle/`.
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
└── PickCube-obstacle/20261022_160500_pilot_diag/  # 1 direct + 6 obstacle attempts

outputs/
├── ckpt/
│   ├── base_dp_stackcube_980/best_eval_success_at_end.pt
│   └── constrainnet_wristcam_seed1/final.pt
└── exp/
    ├── 20260930_base_dp_stackcube_980_seed1/
    ├── 20261021_constrainnet_wristcam_seed1/
    ├── 20261021_wristcam_scenes/
    └── 20261021_wristcam_paired_eval/
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
