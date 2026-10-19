# RESULTS — ConstrainNet on StackCubeObstacle (Ticket 04)

Completed full-budget training and paired closed-loop simulation results for
the obstacle-conditioned ConstrainNet policy (control branch on the frozen
official base DP UNet). Everything below reports **actual executed runs**;
short runs are never presented as full runs.

## Environment

All commands run inside the `maniskill` conda environment
(`source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill`).
Code lives under `examples/baselines/constrain/`; no ManiSkill core file was
modified. Repo base commit for this round:
`2724c64fccad9f3d8d5d71a0a0b85f87c6af2d17` ("feat: [Ticket 03] train
relative-obstacle ConstrainNet"), integration branch, working tree used for
the Ticket 04 additions.

## Full training run

Run directory: `outputs/constrain_runs/full30000_seed1/`

| Item | Value |
| --- | --- |
| Updates executed | 30000 (target 30000; `train_summary.json` `step: 30000`) |
| Demonstrations | `outputs/stackcube_obstacle/trajectory.h5` — all 100 trajectories, 10356 official DP windows |
| Seed | 1 |
| Batch size | 1024 (peak VRAM ≈ 0.45 GiB on the RTX 5070, batch 1024 kept as the default) |
| Learning rate | 1e-4, AdamW betas (0.95, 0.999), weight decay 1e-6 |
| LR schedule | cosine, 500-step warmup |
| EMA | power 0.75 (evaluation weights) |
| Network device | CUDA; CPU physics (training uses no sim) |
| Frozen base | `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt`, subkey `ema_agent`, sha16 `8010257bd96dc866` |
| Loss (per-step JSONL) | first 0.3101 → last 0.0030; mean of last 1000 updates 0.0034; all 30000 recorded losses finite |
| Wall time | ≈ 19 min total net compute |

### Run-directory contents (all durable)

- `run_config.json` — exact hyperparameters, demo path, data counts
  (`num_demos 100`, `num_windows 10356`), base checkpoint path + sha16.
- `logs/loss.jsonl` — exactly 30000 JSON lines, steps 1..30000 in order, one
  per executed update (loss + lr + UTC timestamp), flushed per step.
- `progress.json` — atomically replaced step/elapsed/mean-loss progress.
- `checkpoints/periodic.pt` — rolling recovery checkpoint, saved atomically
  every 1000 updates (tmp file + `os.replace`; a mid-write crash can never
  leave a partial file).
- `checkpoints/final.pt` — final standalone checkpoint (model incl. frozen
  base, EMA, optimizer, lr scheduler, step, RNG, full config).
- `train_summary.json` — evidence block: `all_finite: true`,
  `zero_conv_weights_moved: true`, upstream gradients reached
  `control_down_modules` / `control_mid_modules` / `hint_encoder`,
  `frozen_base_unchanged: true`.

### Execution note (honest record)

The first launch was killed externally at update 3478 by a shell timeout —
after the step-3000 periodic checkpoint had been saved atomically. Training
was resumed from `checkpoints/periodic.pt` (step 3000) into the same run
directory: the resume re-truncated the per-step JSONL to the checkpoint step
(dropping the 478 stale lines), restored optimizer/EMA/lr-scheduler/step/RNG
and re-verified the frozen-base sha, then continued to 30000. The remaining
27k updates ran in one uninterrupted process. `logs/loss.jsonl` therefore
records each update exactly once. A resumed run restarts its batch sampler
from the restored RNG stream, so the exact batch order after the resume point
is not bit-identical to a hypothetical uninterrupted run (Ticket 03 resume
semantics); the update count, loss stream and all invariants are unaffected.

## Post-training verification (independent process, disk-loaded)

`outputs/constrain_runs/full30000_seed1/verify_evidence.json`, produced by
`python -m examples.baselines.constrain.verify_constrain --checkpoint
outputs/constrain_runs/full30000_seed1/checkpoints/final.pt --device cpu`.
The verifier builds the official base DP agent **independently from the
original base checkpoint file** (never from the constrain checkpoint's own
copy) and loads the constrain checkpoint **from disk** (EMA weights):

| Check | Result |
| --- | --- |
| Frozen base bit-unchanged (per-tensor `torch.equal` + sha16 vs independent official load) | **true** (`8010257bd96dc866` on both sides) |
| Disabled-control epsilon output vs official net at identical (sample, timestep, global cond, hint) inputs, 8 batches | max abs diff **0.0** |
| Disabled-control closed-loop `get_action` vs official `Agent.get_action` under the same noise seed | max abs diff **0.0** |
| Enabled control branch movement vs base (recorded, not gated) | mean abs diff 0.154 |

This confirms the ticket invariant: with the control branch disabled the
trained artifact is exactly the frozen base DP, and the base weights were
never modified during training.

## Paired closed-loop evaluation (same scenes for both policies)

Run directory: `outputs/constrain_runs/paired_eval_full30000/`

- Scenes sampled **once** with `base_seed 10000` (scene *i* uses seed
  `10000 + i`), 100 scenes, saved to `scenes_manifest.json` and fed to both
  policies; `paired_summary.json` records `scene_equality_verified: true`
  (per-policy manifests contain identical scene configs).
- Both policies run closed-loop in a single CPU `StackCubeObstacle-v1` env,
  serial, 500-control-step cap, network on CUDA, per-episode torch reseeding
  from the scene seed (comparable noise streams).
- `base_dp` policy: official checkpoint loaded independently;
  `constrain_dp` policy: loaded strictly from
  `outputs/constrain_runs/full30000_seed1/checkpoints/final.pt` (EMA, step
  30000) in this separate process. Per-policy adapter metadata (checkpoint,
  weights, step, normalization) is recorded in both `paired_summary.json` and
  each policy's `manifest.json`.
- Preview videos: recorded only for the predesignated scene indexes
  0, 17, 42, 77 (documented subset, `video_scene_indexes` in the manifest) —
  `base_dp/preview.mp4`, `constrain_dp/preview.mp4`.
- No evaluation scene was added to training, no demonstration holdout was
  used, and the final checkpoint was not selected on evaluation results.
- No scene-sampling failures occurred (all 100 scenes sampled within budget)
  and no runtime failures occurred (both policies `status: completed`):
  **actual completed count is 100/100 for each policy.**
- Wall time: base_dp 1635.7 s (27.3 min), constrain_dp 2463.8 s (41.1 min).

## Metrics (100 completed episodes per policy, same 100 scenes)

| Metric | Base DP | ConstrainNet (final EMA) |
| --- | --- | --- |
| Completed episodes | 100/100 | 100/100 |
| Stack success rate | 0.11 (11/100) | 0.00 (0/100) |
| Obstacle contact rate | 1.00 (100/100) | 0.86 (86/100) |
| Forbidden-disk crossing rate | 0.34 (34/100) | 0.01 (1/100) |
| Safe success rate (stack ∧ no contact ∧ no crossing) | 0.00 | 0.00 |
| Termination reasons | max_steps ×100 | max_steps ×100 |

Per-scene records: `base_dp/episodes/episode_XXX.json`,
`constrain_dp/episodes/episode_XXX.json` (each with the exact scene geometry,
applied collision radius, per-episode flags, first action and seeds).
Aggregates: `base_dp/summary.json`, `constrain_dp/summary.json`,
`paired_summary.json`.

Observed behavior (descriptive, no acceptance gate): the trained ConstrainNet
crosses the forbidden disk far less often than the base DP (34% → 1% of
episodes) and makes contact less often (100% → 86%), i.e. the conditioning
did shift the policy away from the obstacle; but it never reached a stack on
these evaluation scenes (base DP stacked in 11%), and no policy achieved the
env's held `success` termination. This is one training run evaluated on one
scene distribution, not a solvability guarantee of the scenes.

## Known boundaries

- Random scenes pass the geometric filters only (separation, no overlap,
  endpoint clearance, blocks the direct A→B segment); **they are not verified
  to be expert-solvable**, and scene solvability was never checked.
- The evaluation distribution (uniform random valid blocking layouts) differs
  from the action-first demonstration distribution (scenes constructed around
  successful expert motions).
- Metrics are behavior records of the two policies; no minimum success or
  relative-improvement threshold was applied, and the ConstrainNet stack
  regression vs base DP on these scenes is a real measured result of this
  single run (not tuned, not retrained, not checkpoint-selected).
- Scene sampling failures are recorded separately from policy failures and
  never counted as policy failures; the requested episode count is never
  silently reduced. In this round: zero sampling failures, zero runtime
  failures, 100/100 episodes completed per policy.
