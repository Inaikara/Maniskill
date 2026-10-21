# Fixed-scene wristcam paired evaluation — Ticket 05

## Scope

This measures the Base DP and formal wristcam ConstrainNet on the same 100
randomly generated, expert-verified feasible scenes in the Ticket 03 immutable
scene store. This is performance on scenes from the same robot and generation
mechanism; it is **not** a training-set-isolation or generalization claim.
Loading and evaluation read the independent scene store only; no scene sampling,
expert execution, or demonstrations are used.

The evaluation entry point is `evaluate_fixed_scenes.py`. It creates an
environment with each scene's recorded cylinder radius before restoring the
complete saved `env_states`, then applies the native state reset/restore path.
Each policy gets one CPU PhysX environment, the native `panda_wristcam` reset
contract and `robot_init_qpos_noise=0.02`, `state` observations, and
`pd_ee_delta_pose` actions. The two-frame history begins as two copies of the
restored initial observation; actions execute in horizon-8 chunks through the
existing adapter seam, with rotations mapped to zero. Torch is reseeded for
each policy episode with that scene's manifest `seed`. There are no actions
after the first `stack_success`; episodes without stacking stop at 500 control
steps. Safety flags accumulate through the stack-success step, so a same-step
contact/crossing is a stack but not a safe success.

Progress is measured over that same episode window. `ever_grasped` is the
environment's `is_cubeA_grasped` flag; `ever_lifted` means cube A rose more than
0.04 m above its restored initial Z; `ever_transported` means cube A came within
0.10 m in XY of cube B at least once. The disjoint failure bins are no grasp,
grasp without lift, lift followed by transport/place failure, and stack (split
into safe and unsafe). Obstacle contact/crossing refer only to the environment's
obstacle safety flags, not normal grasp or table contact.

## Frozen identities and run

Scene store: `/home/xuwj/dev/maniskill/outputs/wristcam_eval_scenes_20261021/`.
The full store identity is recorded in `WRISTCAM_SCENES.md`; this run checked
canonical manifest identity
`f1aa2175bd70a86ba522b5d441b1a9897330fdbd17a5484d7a06f5eccb8335b3` and HDF5
SHA-256 `b1c736ccfbd4411b20dcb1495ffbf220afd6d7b08dbf46528c4047358d139caf`.

| Policy | Checkpoint / weights | SHA-256 |
| --- | --- | --- |
| Base DP | `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt`, `ema_agent` | `ba65c507ac48aa76a911fa1af28b7155ca3c4f2d7393c775930241dbcde27c25` |
| ConstrainNet | `outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt`, EMA, step 30000 | `74de460532bf4188b7b019f9bb15ecd027af2a7a3b5bef14c2fc6fd4c59b7d02` |

ConstrainNet inference uses the saved 2-frame / 51-feature observation, 48-base
condition plus its relative-condition hint, 16 prediction horizon, 8 action
horizon, 100-step squared-cos epsilon DDPM and saved normalization. No model,
hyperparameter, scene or checkpoint was changed after results were observed.
Both networks ran on CUDA; physics used CPU PhysX and a single environment.

Representative video scene indexes were fixed in the evaluator before any
policy rollout: `0, 17, 42, 77`. The policies use the same scene ordering and
native render camera. Videos are
`outputs/wristcam_paired_eval_20261021_recovery/{base_dp,constrain_dp}/representative.mp4`.

The first attempt used a single environment across scenes and recorded scene 0
as complete but all remaining episodes as explicit runtime errors because the
environment retained scene 0's cylinder radius. That output is preserved at
`outputs/wristcam_paired_eval_20261021/`. After fixing per-radius environment
recreation, recovery was run to a new directory and reused only the two
completed scene-0 records whose scene identity and initial observation evidence
matched; all failed indexes were re-executed. No failed episode was dropped.
Recovery wall time was 1,979.182 s (initial diagnostic attempt: 24.852 s).

## Results

Every rate uses the full requested denominator of 100. Each policy has 100
evaluated episodes and zero runtime errors.

| Measure | Base DP | ConstrainNet |
| --- | ---: | ---: |
| Stack success | 12/100 (12%) | 100/100 (100%) |
| Safe success | 0/100 (0%) | 88/100 (88%) |
| Obstacle contact | 100/100 (100%) | 9/100 (9%) |
| Forbidden crossing | 59/100 (59%) | 4/100 (4%) |
| Timeout (500 steps without stack) | 88/100 | 0/100 |
| Episode control steps, min–max (sum) | 169–500 (47,166) | 83–136 (9,582) |
| Ever grasped / lifted / transported | 100 / 100 / 55 | 100 / 100 / 100 |
| No grasp | 0 | 0 |
| Grasp without lift | 0 | 0 |
| Lift then transport/place failure | 88 | 0 |
| Safe stack / unsafe stack | 0 / 12 | 88 / 12 |

The 12 unsafe ConstrainNet stacks include same-step/episode-window obstacle
violations and are not reported as safe successes. Counts, flags, progress
indicators, step counts, scene IDs, seeds, initial-state checks and first action
are retained per episode. The machine-readable summaries can be recomputed from
the 200 `episode_NNN.json` records; each policy also has `summary.json` and
`manifest.json`, with the paired evidence in `paired_summary.json`.

## Verification

- `paired_summary.json` reports all 100 actual restored-state fingerprints
  equal between policies and all restored observations within `1e-6` of the
  scene-store observation (observed maximum absolute error: `1.1921e-7`).
- All 100 scene IDs/seeds were present for both policies; each policy had 100
  episode files, 100 unique restored-state fingerprints, and zero runtime-error
  records. Episode lengths were within `[1, 500]`.
- Complete-state round-trip validation for the frozen store is independently
  recorded in `WRISTCAM_SCENES.md` / `roundtrip_validation.json`.
- Focused tests cover disjoint progress bins, fixed denominator on runtime
  errors, same-step stack plus obstacle violation, strict progress inputs, and
  checkpoint-file identity changing when weights change. Public `run_one`
  rollouts use a finite fake env/adapter to verify same-step unsafe stack
  accounting and no post-success actions, exact step-cap handling across a
  partial final chunk, fresh observation history on the next episode, and
  environment recreation when scene radius changes. Resume requires
  matching policy/checkpoint SHA-256, scene manifest, seed, state evidence and
  runtime/inference configuration; legacy records derive the checkpoint hash
  from their recorded adapter weight source.
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
  examples/baselines/constrain/tests -q`: **162 passed** (focused module:
  **9 passed**).
- A bounded audit re-read all 200 episode JSON files: 100 evaluated records per
  policy, zero runtime errors, 100 distinct scene-state hashes, and matching
  restored-state fingerprints for all 100 pairs. Checkpoint and scene identities
  matched their manifests. Detailed evidence:
  `outputs/wristcam_paired_eval_20261021_recovery/final_audit.json`. Metrics are
  unchanged: Base DP 12 stacks / 0 safe / 100 contacts / 59 crossings / 88
  timeouts; ConstrainNet 100 / 88 / 9 / 4 / 0.

## Reproduction

Run from repository root in the `maniskill` conda environment. The recovery
output already exists and is intentionally not overwritten; choose a new
output directory for a fresh run.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=10s 14400s \
  python -m examples.baselines.constrain.evaluate_fixed_scenes \
  --scene-dir /home/xuwj/dev/maniskill/outputs/wristcam_eval_scenes_20261021 \
  --base-checkpoint outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt \
  --constrain-checkpoint outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt \
  --output-dir outputs/wristcam_paired_eval_fresh_20261021 \
  --device cuda --num-scenes 100 --max-steps 500
```

For an interrupted run, set `--resume-from` to the existing paired output root;
only completed records matching current policy, checkpoint, scene, seed, and
inference/runtime configuration are reused. The original completed scene-0
records satisfy the strengthened identity guard:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=5s 14400s \
  python -m examples.baselines.constrain.evaluate_fixed_scenes \
  --scene-dir /home/xuwj/dev/maniskill/outputs/wristcam_eval_scenes_20261021 \
  --base-checkpoint outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt \
  --constrain-checkpoint outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt \
  --output-dir outputs/wristcam_paired_eval_resume_20261021 \
  --resume-from outputs/wristcam_paired_eval_20261021_recovery \
  --device cuda --num-scenes 100 --max-steps 500
```
