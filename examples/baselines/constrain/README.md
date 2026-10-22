# StackCubeObstacle action-first data generation

## Current retained outputs

See [OUTPUTS.md](OUTPUTS.md) for the current `demos/`, `outputs/ckpt/`, `outputs/exp/` layout,
artifact relocation map, and current training/evaluation commands. The sections
below include historical experiments whose original paths may have been moved
or removed. Current wristcam results are in [WRISTCAM_EVALUATION.md](WRISTCAM_EVALUATION.md).

Generate obstacle-avoidance demonstrations with direct `pd_ee_delta_pose`
feedback waypoint control. Each action template is executed successfully without
an obstacle first; cylinders are then sampled around its actual motion and the
same policy is executed again with each cylinder present.

## Environment

Activate the environment before development, generation, testing, or rendering:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
```

`StackCubeObstacle-v1` supports `physx_cpu`, one environment, and Panda robots.
Collision and forbidden-disk checks run at physics-substep frequency. The
generator uses `panda`, flat `state` observations, and a 500-control-step episode
limit. It does not require MPLib or a separate controller-conversion replay.

## Generate data and a preview video

From the repository root:

```bash
bash bash/generate_stackcube_obstacle_demos.sh
```

The script activates `maniskill`, collects demonstrations, and after successful
collection renders `dataset.mp4` in the same output directory. Edit the parameter
block in the script to change the output directory, target count, reuse count,
seed, or budgets. The current target is 100 episodes in
`outputs/stackcube_obstacle/`.

The output directory contains:

| File | Contents |
| --- | --- |
| `trajectory.h5` | Accepted `pd_ee_delta_pose` actions, states, and observations |
| `trajectory.json` | Episode configuration, validity flags, plans, and provenance |
| `report.json` | Generation timings, template reuse, clearances, and failures |
| `dataset.mp4` | All episodes concatenated in numeric ID order |

Existing collection artifacts are not overwritten; choose a new output directory
for another run. Failed candidate recordings are discarded from temporary
storage. Each accepted trajectory has T actions and T+1 states/observations.

For a small run with explicit parameters:

```bash
python -m examples.baselines.constrain.collect_constructive \
    --output-dir outputs/stackcube_obstacle_preview \
    --target-successes 10 --reuse-count 2 --seed 0
python -m examples.baselines.constrain.render_dataset_video \
    --dataset-dir outputs/stackcube_obstacle_preview
```

Collection displays one progress bar counting accepted trajectories, with the
current phase, template number, and obstacle execution attempts. Rendering has
its own frame progress bar. The preview renders saved states at the recorded
control frequency without re-executing actions. Re-rendering replaces only
`dataset.mp4` after rendering succeeds.

## Generation algorithm

1. **Sanity check:** complete one pick/carry/place episode with the cylinder
   parked outside the workspace.
2. **Action template:** sample cube positions A/B and a side-detour waypoint W
   independently of any cylinder. Execute hover, vertical descend, grasp,
   vertical lift, A → W → B carry, vertical placement, and release/retreat.
   Keep only successfully stacked templates.
3. **Record actual motion:** collect robot collision-geometry vertices below
   the cylinder top and carried-cube corners, projected onto XY, plus the actual
   carry TCP/cube traces. This accounts for the palm and arm as well as the TCP.
4. **Place cylinders:** sample in the direct A–B blocking strip, outside the
   recorded footprint with clearance. A cylinder must block straight transport
   and be tall enough to require side avoidance. Clearance categories are
   `roomy` (at least 3 cm) and `tight` (at least 1.5 cm).
5. **Execute with the cylinder:** reuse the exact template seed, cube layout,
   and waypoint plan. Check initial robot/cube states against the template's
   recorded initial-state hash and between accepted reuses. Accept only a
   successful stack with no cylinder contact or post-grasp forbidden-disk
   crossing.

`--reuse-count` limits accepted cylinder scenes per action template, default 2.
Thus 10 episodes require at least 5 templates. Position deltas use feedback
`clip((target - tcp) / 0.1 m)` with an additional action cap. Rotation deltas are
zero, preserving the reset downward orientation approximately. Cube yaw sampling
does not rotate the TCP. The two horizontal carry phases command the same height;
observed height and orientation errors are recorded, not strict acceptance gates.

The footprint is a sampled geometric prefilter; final acceptance comes from real
physics execution. Templates, cylinder searches, episode attempts, and waypoint
tracking all have finite budgets. `Config` in `collect_constructive.py` defines
sampling ranges, tracking parameters, cylinder height (default 0.18 m), and
clearance thresholds. Cylinder radius is sampled from 0.02–0.04 m; cubes have
4 cm sides. Cylinder height has one configuration source shared by sampling and
environment construction.

## Code and tests

| File | Responsibility |
| --- | --- |
| `collect_constructive.py` | Sampling, feedback control, template reuse, collection, and reporting |
| `stackcube_obstacle.py` | Cylinder environment, state restoration, contact/crossing validity |
| `obstacle_record.py` | Official recording wrapper with per-episode geometry and validity |
| `render_dataset_video.py` | Streaming preview rendering from saved states |
| `base_dp.py` | Base DP policy adapter: strict checkpoint load, 51d->48d condition trim, 4d->7d action mapping |
| `rollout_base_dp.py` | Reproducible single-episode closed-loop rollout of the base policy in the obstacle env |
| `evaluate_base_dp.py` | Randomized-scene serial evaluation: fixed-seed scene sampling, filtering, per-episode closed-loop runs, summary metrics, preview video, replay |
| `constrain_condition.py` | Relative obstacle condition: latest-frame `[obstacle_x - tcp_x, obstacle_y - tcp_y, radius]`, fixed configurable normalization (shared train/inference) |
| `constrain_denoiser.py` | ControlNet-style ConstrainNet: frozen official DP UNet + trainable control copy, hint encoder, zero-init multiscale skip/mid residuals |
| `constrain_dataset.py` | Raw 100-trajectory demo reader: zero-rotation check, 7d->4d mapping, official DP windowing/padding |
| `train_constrain.py` | Training CLI (frozen base, epsilon objective) with standalone checkpoints, durable run logging (`run_config.json`, per-step `logs/loss.jsonl`, atomic `progress.json`), atomic checkpoint saves and resume |
| `train_logging.py` | `RunLogger`: per-step JSONL loss log, atomic progress/config files, crash-recovery log truncation on resume |
| `constrain_adapter.py` | `ConstrainDPAdapter`: the constrain policy on the identical rollout/evaluator seam (`base_cond`/`get_action`/`map_action_ee_delta_pose`) |
| `rollout_constrain.py` | Reproducible single-episode closed-loop rollout from a standalone constrain checkpoint |
| `verify_constrain.py` | Post-training verification against the *independently loaded* official base: frozen-base bit equality + disabled-control epsilon/rollout equivalence at identical inputs |
| `evaluate_paired.py` | Paired evaluation driver: one shared sampled scene list fed to base DP and ConstrainNet, merged `paired_summary.json` with scene-equality evidence and timings |

```bash
python -m pytest examples/baselines/constrain/tests -q
```

The retained tests cover action-first ordering, fixed-plan reuse, initial-state
identity, sampling clearance, tracking statistics, finite budgets, failure
cleanup, data structure, real environment contacts/crossings, state restoration,
per-episode recording metadata, and preview rendering.

## Verified example

`outputs/stackcube_obstacle_constructive_action_first/` contains 10 accepted
episodes from 5 templates with 2 cylinder layouts each, plus the preview video.
That run took about 24.2 s for generation: 1.1 s sanity, 5.3 s templates,
7.6 s cylinder sampling, and 10.1 s obstacle execution. All 10 obstacle physics
attempts passed. Video rendering time is excluded; this is one small-run result,
not a general success-rate guarantee.

## Base DP policy adapter and closed-loop rollout

`base_dp.py` reuses the official ManiSkill `BC_Diffusion_state_UNet` DP for the
obstacle environment without touching ManiSkill core code:

- Loads the trained checkpoint
  `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt`
  with its EMA copy (`ema_agent` subkey, configurable weight source recorded in
  `BaseDPAdapter.metadata`). `load_state_dict` is strict: mismatched weight
  tensors fail the build instead of being silently ignored.
- Keeps the official semantics exactly: epsilon prediction, squared-cos DDPM
  scheduler with 100 diffusion steps, two-frame obs history, 16-step prediction
  horizon, and 8-step executed horizon. The network channels come from the
  checkpoint itself (the condition encoder reads a 160-wide input, i.e.
  `obs_horizon * obs_dim = 2 * 48` plus the 64-d diffusion embedding).
- Trims the queued 2x51 obstacle flat obs to the base 2x48 condition (the
  appended obstacle x/y/radius features map nowhere), and maps the 4d base
  action (xyz delta + gripper) to the 7d `pd_ee_delta_pose` action by
  appending three zero rotations on the same device.
- Device notes: the official `Agent.get_action` samples noise on
  `obs_seq.device`, so inference is correct on CPU or CUDA without editing
  official source; only training (`compute_loss`) uses a module-global device
  and is never called here.

### Verification commands

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# adapter seam + tests (57 tests incl. the base DP adapter)
python -m pytest examples/baselines/constrain/tests -q

# independent denoise equivalence: a separate official Agent built and loaded
# straight from the checkpoint on disk, compared at identical noise streams,
# epsilon inputs and diffusion timesteps with the adapter network

# reproducible single-episode closed-loop rollout (CPU sim; network device configurable)
python -m examples.baselines.constrain.rollout_base_dp --seed 11 --device cpu
python -m examples.baselines.constrain.rollout_base_dp --seed 11 --device cuda
```

`rollout_base_dp` runs one real `StackCubeObstacle-v1` episode with CPU physics
on the 500-step cap, obstacle height 0.18 m and radius sampled in 0.02-0.04 m,
and records termination reason, stack/contact/crossing flags and the safe
success flag in a JSON file under `outputs/base_dp_rollout/` (rolls a numbered
suffix instead of overwriting existing files). This is a behavior record, not
an avoidance acceptance gate.

### Recorded rollout evidence (2026-10-19)

| Run | JSON | Result |
| --- | --- | --- |
| seed 11, net CPU | `outputs/base_dp_rollout/rollout_seed11_cpu.json` | success at step 98, no contact, no crossing, `safe_success: true`, radius 0.0226 |
| seed 11, net CUDA | `outputs/base_dp_rollout/rollout_seed11_cuda.json` | success at step 110, no contact, no crossing, `safe_success: true`, radius 0.0226 |

The two runs sample the same deterministic noise stream per device; the
step-count difference is the CPU/CUDA floating-point difference in the cached
denoising stream. The 57-test suite passes with the base DP adapter tests
included (numerical trim/mapping/epsilon/get_action equivalence, strict
mismatched-weight rejection, and a separate physical controller-mapping check
that verifies the mapped 7d action in real `StackCubeObstacle-v1` physics:
axis-predominant TCP translation, constant TCP orientation for zero rotation
deltas, and real finger open/close behavior).

## Randomized-scene base DP evaluation

`evaluate_base_dp.py` serially evaluates the base DP policy on a fixed seed
list of fresh random scenes in one CPU `StackCubeObstacle-v1` environment
(503 steps of policy inference per call make CPU runs slow; plan run sizes
accordingly). Actions come only from the policy closed loop through
`BaseDPAdapter.base_cond` / `get_action` / `map_action_ee_delta_pose` on the
full 2x51 stacked history — no expert, demonstration action, demonstration
replay, or training-demonstration holdout is involved, and the same adapter
seam accepts a future ConstrainNet checkpoint.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# default: 100 episodes, 500 control steps each (the Ticket 04 full run)
python -m examples.baselines.constrain.evaluate_base_dp --device cpu

# small smoke run
python -m examples.baselines.constrain.evaluate_base_dp \
    --num-episodes 3 --max-steps 100 --device cpu

# reproduce one recorded episode exactly from its manifest entry
python -m examples.baselines.constrain.evaluate_base_dp \
    --replay-episode outputs/base_dp_eval/episodes/episode_000.json
```

Outputs go to a new unique run directory (`outputs/base_dp_eval`,
`base_dp_eval-1`, ...; existing assets are never overwritten):

| File | Contents |
| --- | --- |
| `manifest.json` | Run settings (seeds, device, checkpoint, env params, filter rules) + exact world-coordinate scenes |
| `episodes/episode_XXX.json` | Per-episode scene, applied geometry (real collision radius read back), result flags, first action |
| `summary.json` | Aggregate metrics: stack success, obstacle contact, forbidden crossing, safe stack success rates |
| `preview.mp4` | Preview video rendered from the real rollouts |

### Scene sampling and geometric filters

Scene `i` uses seed `base_seed + i` (fixed configurable seed list via
`--seed` / `--num-episodes`); the same rng also seeds the env reset and the
torch noise stream for that episode, so replaying a recorded episode entry
reproduces the exact layout and result. Cube positions, yaws, cylinder centre
and radius are drawn uniformly inside the `collect_constructive` generator
ranges — `cube_xy_range` `(-0.15, 0.05, 0.18, 0.30)`, `obstacle_xy_range`
`(-0.15, 0.15, 0.18, 0.30)`, radius `0.02-0.04` m, obstacle height `0.18` m —
and then filtered (all recorded in `manifest.json`):

1. **Cube separation:** `0.25 m <= |cubeB - cubeA| <= 0.30 m`.
2. **No initial overlap:** cylinder centre at least `radius + cube corner
   radius` from both cube centres.
3. **Endpoint clearance:** at least `radius + cube corner radius +
   safety_margin (0.005) + 0.005` from both cube centres.
4. **Blocks the direct route:** the cylinder centre lies within
   `radius + cube corner radius + safety_margin - min_segment_clearance
   (0.01)` of the direct cubeA-cubeB segment, so a straight carry would enter
   the forbidden disk.

Sampling has a finite retry budget (5000 candidate draws per scene). Exhaustion
raises `SceneSamplingError` and is recorded under `sampling_failures` — the
requested episode count is never silently reduced and a sampling failure is
never counted as a policy failure.

### Metrics and limits

Per episode: stack success (stack achieved at any step), obstacle contact and
forbidden crossing accumulated over the whole episode, and safe success =
stack success AND no contact AND no crossing. Episodes end on env safe success
or the control-step cap.

Geometric validity is **not** expert solvability: the filters only guarantee a
physically sensible blocking layout, not that any policy can solve the scene.
The evaluation distribution (uniform random valid layouts) is also not the
action-first demonstration distribution used for training, which constructs
scenes around successful expert motions. The base policy has no obstacle
awareness, so contact/crossing rates measure behavior, not an acceptance gate.

### Verified smoke run (2026-10-19)

`outputs/base_dp_eval-1/` — 3 scenes (seeds 0-2), max 100 control steps each,
base DP checkpoint on CPU net, CPU physics, with preview video:

| Metric | Value |
| --- | --- |
| Stack success rate | 0/3 |
| Obstacle contact rate | 2/3 (first contact steps 89 and 18) |
| Forbidden crossing rate | 0/3 |
| Safe stack success rate | 0/3 |

The base policy does not avoid the sampled cylinders (expected: it is
obstacle-blind). Replay of `episodes/episode_000.json` reproduced the recorded
result exactly (elapsed steps, contact step, all flags), verifying scene
manifest replay. CPU inference costs roughly 30-45 s per 8-step DP chunk on
this machine under load; the full 100x500 run (Ticket 04) should use a
proportionate time budget or the CUDA device for network inference.

## Obstacle-conditional ConstrainNet (Ticket 03)

`constrain_denoiser.py` reuses the official ManiSkill DP UNet as a frozen base
and adds a ControlNet-style trainable control branch — no LeRobot policy
classes, no image inputs, no velocity mode, no collision loss, no inference
safety projection:

- The frozen base is the official `ConditionalUnet1D` loaded from the Ticket 01
  checkpoint source (`outputs/runs/dp_StackCube-v1_980_1/checkpoints/
  best_eval_success_at_end.pt`, `ema_agent` subkey). All 148 base tensors are
  frozen (`requires_grad=False`) and verified bit-unchanged during and after
  training (sha16 of the frozen state stored in every checkpoint and
  re-verified on resume).
- The control branch is an exact weight copy of the base `down_modules` and
  `mid_modules` plus a small hint encoder mapping the repeated 3d relative
  condition to the UNet input channels. Zero-initialized 1x1 residual
  convolutions add control features to every decoder-consumed down-path skip
  connection and to the frozen mid output. Note: the official (colab-derived)
  UNet's up path pops only the last two down-level skip connections, so the
  first down-level skip is dead in the base architecture and correctly
  carries no residual.
- At initialization the control residuals are exactly zero, so the denoiser
  output and the disabled-control output both equal the base DP output at the
  same sample/timestep/global condition (tested). Only the zero convs update
  on the first step; from the second update on, gradients reach the upstream
  control branch (standard ControlNet dynamic, tested).
- The condition is `[obstacle_x - tcp_x, obstacle_y - tcp_y, radius]` in
  meters read from each window's *latest* observation frame of the proven
  flat layout (TCP world xy at indices 18/19; appended obstacle
  center/radius at 48-51; verified against the real environment), centered on
  the TCP but keeping world XY axes (never rotated into the TCP frame), and
  repeated over all 16 predicted steps. Normalization is a fixed, configurable
  affine map to [-1, 1] with clipping — `rel_xy_range (-1, 1)` m,
  `radius_range (0, 0.08)` m preserving real radius semantics; no per-batch
  statistics, and the exact transform is recorded in the checkpoint and shared
  verbatim by training and inference.

### Training data and objective

`constrain_dataset.py` reads the raw 100-trajectory
`outputs/stackcube_obstacle/trajectory.h5` directly (no holdout, no rewritten
files), verifies all three rotation channels are zero at load, maps the 7d
`pd_ee_delta_pose` actions to the base 4d `[x, y, z, gripper]`, and keeps the
official DP windowing: `(traj_idx, start, end)` slices, pad-before repeats the
first observation/action, pad-after zeroes the xyz delta and holds the last
gripper value. The loss is the official epsilon objective in
`compute_epsilon_loss` (MSE against the sampled noise).

### Checkpoints and commands

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# tests (all constrain + prior tickets, 32s)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest examples/baselines/constrain/tests -q

# short training run (defaults mirror official config; full 30000-iter budget
# is the Ticket 04 run)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.train_constrain \
    --total-iters 300 --batch-size 256 --device cuda \
    --run-dir outputs/constrain_runs/short300 --save-every 150

# resume: --total-iters is the target total update count; the frozen base and
# its recorded checksum, optimizer/EMA/scheduler states, step and RNG streams
# are restored and re-verified from the standalone checkpoint
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.train_constrain \
    --resume outputs/constrain_runs/short300/checkpoints/final.pt \
    --total-iters 350 --batch-size 256 --device cuda --run-dir outputs/constrain_runs/short350

# closed-loop rollout from the saved checkpoint in an independent process
# (CPU sim; CUDA net allowed); --disable-control runs the base-DP behavior
# under the same seeds for comparison
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.rollout_constrain \
    --checkpoint outputs/constrain_runs/short350/checkpoints/final.pt \
    --seed 11 --device cuda --max-steps 300
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.rollout_constrain \
    --checkpoint outputs/constrain_runs/short350/checkpoints/final.pt \
    --seed 11 --device cuda --max-steps 300 --disable-control
```

Each checkpoint (`checkpoints/final.pt` / `periodic.pt`) is standalone: model
(incl. frozen base), EMA, optimizer, lr scheduler, step, RNG streams, model
config, policy horizons, normalization config and the base weight source
(path + subkey + sha16). Reloads are strict and inference depends only on the
checkpoint file. `train_summary.json` in each run dir records the evidence
block (finite-loss guarantee, zero-conv movement, upstream gradients, frozen
base sha verification).

### Evidence (2026-10-19)

| Artifact | Result |
| --- | --- |
| Short training `outputs/constrain_runs/short300/` | 300 updates on all 100 demos (batch 256, CUDA net, CPU sim unused), all recorded losses finite, loss history 0.139 -> 0.083 |
| Training evidence block | zero convs moved, gradients reach `control_down_modules` / `control_mid_modules` / `hint_encoder`, frozen base sha unchanged |
| Resume run `outputs/constrain_runs/short350/` | steps 300 -> 350 from `--resume`, all invariants re-verified green, loss 0.076 |
| Rollout (constrain, CPU sim + CUDA net) | `outputs/constrain_rollout/rollout_constrain_seed11_cuda.json`: 300-step cap, no obstacle contact, no forbidden crossing; stack not yet achieved at this tiny training budget |
| Rollout (disabled control, same seeds) | `outputs/constrain_rollout/rollout_disabled_seed11_cuda.json`: safe stack success at step 110 — identical behavior class to the base DP evidence above |

The 350-update budget is deliberately small (full 30000-update training and
the 100-episode randomized evaluation are Ticket 04 work); the rollout is a
behavior record, not an avoidance acceptance gate.

## PickCube Base DP (Ticket 03)

PickCube uses the independent `pickcube_base_dp.py` adapter and
`train_pickcube_base_dp.py` entry point. It trains a fresh official UNet at the
native PickCube 42d state / 7d `pd_ee_delta_pose` contract, using only the
formal 1,000 direct demos; it does not reuse the StackCube checkpoint or
StackCube adapter contract. The 45d obstacle observation is schema-checked at
runtime and only its verified 42d state prefix enters the base network.

The completed 30,000-update configuration, dataset/checkpoint identities,
training losses and independent short obstacle-scene rollout are recorded in
[PICKCUBE_TRAINING.md](PICKCUBE_TRAINING.md). The checkpoint's EMA network can
be loaded through `PickCubeBaseDPAdapter.initialize_constrain_base()` for the
task-specific ConstrainNet branch.

## Full training, verification and paired evaluation (Ticket 04)

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# full training run: all 100 demos, 30000 updates, official defaults
# (batch 1024, lr 1e-4, warmup 500, cosine, EMA power 0.75, seed 1), CUDA net
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.train_constrain \
    --total-iters 30000 --batch-size 1024 --seed 1 --device cuda --save-every 1000 \
    --run-dir outputs/constrain_runs/full30000_seed1

# resume any interrupted run from a checkpoint (periodic.pt is replaced
# atomically every --save-every steps; final.pt is always written at the end);
# --total-iters stays the target total update count
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.train_constrain \
    --resume outputs/constrain_runs/full30000_seed1/checkpoints/periodic.pt \
    --total-iters 30000 --batch-size 1024 --seed 1 --device cuda --save-every 1000 \
    --run-dir outputs/constrain_runs/full30000_seed1

# post-training verification in an independent process, loading final.pt from
# disk and the official base from its own checkpoint file: frozen-base bit
# equality, disabled-control epsilon equivalence and disabled-control rollout
# equivalence at identical inputs; writes verify_evidence.json next to the run
python -m examples.baselines.constrain.verify_constrain \
    --checkpoint outputs/constrain_runs/full30000_seed1/checkpoints/final.pt --device cpu

# paired evaluation: 100 fresh random scenes (base_seed 10000, scene i uses
# seed 10000+i), sampled ONCE and fed to BOTH policies, 500-step cap, one CPU
# env, serial; preview videos recorded only for predesignated scene indexes
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m examples.baselines.constrain.evaluate_paired \
    --constrain-checkpoint outputs/constrain_runs/full30000_seed1/checkpoints/final.pt \
    --num-episodes 100 --max-steps 500 --seed 10000 --device cuda \
    --output-dir outputs/constrain_runs/paired_eval_full30000 \
    --video-episodes 0,17,42,77
```

Each training run dir contains `run_config.json` (exact hyperparameters, demo
path, data counts, base checkpoint + sha16), `logs/loss.jsonl` (one JSON line
per update, flushed per step), `progress.json` (atomic step/elapsed/loss
progress), `checkpoints/periodic.pt` (rolling, atomic) and
`checkpoints/final.pt` (standalone, incl. EMA), plus `train_summary.json` with
the evidence block. The paired evaluation writes `scenes_manifest.json` (the
shared scene configs), per-policy `base_dp/` and `constrain_dp/` evaluator
outputs and a merged `paired_summary.json` (metrics, scene-equality evidence,
actual completed counts, timings, per-policy adapter metadata incl.
checkpoint+normalization).

Results of the completed Ticket 04 runs: [RESULTS.md](RESULTS.md).
