# PickCube Base DP (Ticket 03)

> Ticket 04's obstacle-conditioned ControlNet policy and its training evidence
> are documented in [PickCube ConstrainNet (Ticket 04)](#pickcube-constrainnet-ticket-04).

## Reproduction and fixed configuration

This is an independent official-architecture BC diffusion policy, initialized
from scratch for PickCube. No StackCube checkpoint is read or used. Training
uses only the 1,000 `direct` episodes in
`demos/PickCube-obstacle/20261022_132753_formal/trajectory.h5`; the 100
obstacle episodes are excluded by manifest episode IDs before windows are
constructed. The loader follows the official ManiSkill DP temporal alignment
and end padding. Training and inference both use the runtime-verified 42-value
native state prefix; the 3 named `extra/pickcube_obstacle` values are never a
Base DP input. Native `pd_ee_delta_pose` actions retain all 7 channels.

The formal run's complete settings are in
`outputs/exp/20261022_144000_pickcube_base_dp/run_config.json`. It fixes 30,000
updates, seed 1, batch 1,024, AdamW (lr `1e-4`, betas `(0.95, 0.999)`, weight
decay `1e-6`), cosine schedule with 500 warmup steps, EMA power `0.75`, identity
observation/action normalization, 2/16/8 observation/prediction/execution
horizons, official 64/128/256 channels, 64d diffusion embedding, 8 GroupNorm
groups, 100-step squared-cosine DDPM epsilon prediction, no gradient clipping,
and final EMA as the selected checkpoint. This selection was fixed before
training; there was no policy evaluation or checkpoint tuning.

The independent, non-formal probe was capped at 45 seconds: batch 1,024,
2,081 updates, 46.23 updates/s and 540,697,600 peak CUDA bytes. The formal run
completed all 30,000 optimizer updates in 896.09 seconds (901.15 seconds
process wall). Its 30,000 per-update losses are flushed to
`outputs/exp/20261022_144000_pickcube_base_dp/logs/loss.jsonl`: every loss was
finite; first `1.1496141`, last `0.00279629`, minimum `0.00219548`, maximum
`1.1496141`, mean of final 100 `0.00313653`. Initial and final raw model hashes
differ, confirming parameter updates.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Formal HDF5 (`trajectory.h5`) | `fa846f52b5ae08de0243cfa52b40a4b0e7ce160881aa9db59ebaf4952c02904e` |
| Formal episode manifest (`trajectory.json`) | `8ef0d2739d6bf5675316ab37f42f1437cc19d80739e6f9102973beeabd6d1e58` |
| Frozen collection contract | `e1d98a8934203a2dcb397dc604736cb7877f9134cca4ef84ee502f8325f4f46e` |
| Formal collection report | `a11fa82d71e571656b86a411b020006630dcc7341bdad5b28a6ac7650951cf31` |
| Complete run configuration (written before updates; status/contract metadata finalized after) | `71a6294323c39e296e83dc39758d8a3edc75e32c1904a8b69e0ca890f8d7827e` |
| Per-update loss log | `b8e60ee3750b5e9662d1ba33b2ccc0b50976b3aaee30a967d132a0074da813d8` |
| Final EMA checkpoint | `b379573970712bcc17279ed237602a7be823acce9f7134ba125041ba5478389c` |
| Short rollout evidence (full initial observation/state/controller snapshot) | `67462a898d488a19718d733710492fb59d81446d6e3790bab4116aa72f1195b9` |

The run configuration and checkpoint record the frozen data contract hash,
full HDF5/manifest hashes, all 1,000 selected IDs, count and class selection;
the frozen contract carries the source-file hashes and its own git pin. The
config records the runtime named schema:
`agent/qpos:9`, `agent/qvel:9`, and ordered `extra` fields
`is_grasped:1,tcp_pose:7,goal_pos:3,obj_pose:7,tcp_to_obj_pos:3,
obj_to_goal_pos:3,pickcube_obstacle:3`. Runtime construction verified 45 total
dimensions and that the final three exactly equal the named obstacle field.

## Checkpoint interface and verification

`PickCubeBaseDPAdapter` in `pickcube_base_dp.py` strictly validates checkpoint
format, complete task/controller/model/diffusion contract, named observation
schema and tensor shapes. `base_cond` removes only the verified obstacle tail;
`get_action` consumes the matching two-frame 42d state, and
`initialize_constrain_base(denoiser)` strictly copies the trained EMA
ConditionalUnet1D into a future task-matched ConstrainNet base branch.

The independent-process short rollout loaded `final.pt` from disk and used
formal obstacle episode 1000's scene (chosen from the frozen manifest before
inference). It executed 32 real native actions in `PickCubeObstacle-v1`; all
were finite, seven-dimensional and in `[-1,1]`. Perturbing only obstacle
values left the base condition exactly unchanged. A fresh env reset reproduced
the initial 45d observation exactly. Full initial state and controller state
identities and all actions/flags are saved in
`outputs/exp/20261022_161500_pickcube_base_dp_rollout/short_rollout.json`.
The controller snapshot was empty, as expected for this controller. The
rollout continued through the interface cap with no success, obstacle-contact
or crossing flag; it is not a success-rate result or substitute for the later
100-scene evaluation.

## Commands

From the repository root after activating `maniskill`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.train_pickcube_base_dp \
  --demo-path demos/PickCube-obstacle/20261022_132753_formal/trajectory.h5 \
  --manifest demos/PickCube-obstacle/20261022_132753_formal/trajectory.json \
  --run-dir outputs/exp/YYYYMMDD_HHMMSS_pickcube_base_dp \
  --checkpoint-dir outputs/ckpt/YYYYMMDD_HHMMSS_pickcube_base_dp \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 \
  --lr 0.0001 --warmup 500 --ema-power 0.75 --save-every 1000

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m examples.baselines.constrain.rollout_pickcube_base_dp \
  --checkpoint outputs/ckpt/YYYYMMDD_HHMMSS_pickcube_base_dp/final.pt \
  --device cuda --steps 32 --output-dir outputs/exp/YYYYMMDD_HHMMSS_pickcube_base_dp_rollout

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest examples/baselines/constrain/tests -q
```

## PickCube ConstrainNet (Ticket 04)

### Fixed task contract and training

The ConstrainNet starts from the Ticket 03 final EMA Base DP checkpoint and
trains only its copied control down/mid path, hint encoder, and zero-initialized
residual convolutions. The 42-value base state excludes the named obstacle
field; the native `pd_ee_delta_pose` 7-channel output is retained without
rotation dropping or padding. Training and rollout share the latest-frame
condition extracted using the named `extra/tcp_pose[:2]` and
`extra/pickcube_obstacle[x,y,radius]` fields. The world-parallel relative
condition is `[obstacle_x-tcp_x, obstacle_y-tcp_y, radius]` in meters,
normalized by `[0.5 m, 0.5 m, 0.08 m]` and clipped to `[-1,1]`.

Only the formal obstacle episodes 1000–1099 were loaded. Dataset loading checks
the exact Ticket 02 HDF5 and manifest SHA-256 values, all formal IDs/classes,
successful terminal records, finite native 7D actions, and recorded obstacle
safety flags. The Base DP loader additionally requires the exact direct IDs
0–999 and trusted formal HDF5 hash. Both checkpoints contain the full native
robot/state/action/horizon/model/diffusion contract. The ConstrainNet checkpoint
contains its full Base DP weights and loads for inference without the Base DP
checkpoint file; optional independent verification compares it to the recorded
trusted Ticket 03 EMA checkpoint.

The fixed run config is written before updates. It specifies 30,000 updates,
seed 1, batch 1,024, AdamW `1e-4` / betas `(0.95, 0.999)` / weight decay
`1e-6`, cosine schedule with 500 warmup steps, EMA power `0.75`, no gradient
clipping, official 100-step squared-cosine DDPM epsilon objective, 2/16/8
horizons, identity state/action normalization, and final EMA selection fixed
before training. There was no evaluation-based checkpoint selection. A separate
45-second non-formal GPU probe completed 1,653 updates at 36.72 updates/s
(534,474,240 peak CUDA bytes); its record is
`outputs/exp/20261022_144800_pickcube_constrain_probe/probe_summary.json`.

The formal run completed 30,000/30,000 updates in 818.20 seconds process wall.
All 30,000 per-update losses are finite: first `0.09474345`, last `0.00286592`,
minimum `0.00188586`, maximum `0.10731530`. Logs and run configuration are
under `outputs/exp/20261022_144835_pickcube_constrain/`; final EMA and full
standalone state are `outputs/ckpt/20261022_144835_pickcube_constrain/final.pt`.
The checkpoint is 73,859,580 bytes.

### Verification and short rollout

Before training, zero-residual epsilon and the full 100-step DDPM output matched
an independently loaded Base EMA under controlled identical noise (both max
absolute differences `0`). After training, the frozen base state hash exactly
matches the independently loaded Base EMA; the disabled-control epsilon output
has max absolute difference `0`. The raw control state hash changed from
`a8b583a3c275a9811ea18f8aa3d1bbe36976529a1cb0a231e4c0b26dd666ab48` to
`640dedaa60ddf88d88915fa76f7791ed8ebf3499b0cde0f21d2cdec1681948de`; zero
residual weights moved, and gradients are nonzero in the copied down path, mid
path, and hint encoder. The trained enabled branch changes the probe epsilon
output by mean absolute `0.07391`.

A 32-action real rollout used formal obstacle episode 1000, selected before
inference. It emitted finite native actions with shape `(32,7)` and remained
within action bounds. Exact restoration matched for the initial observation,
full native state, and controller state. Native success was not reached within
32 steps; no obstacle contact or crossing was recorded. This is only a bounded
interface rollout, not a formal evaluation or success-rate claim. Details and
the complete action trace are in
`outputs/exp/20261022_150600_pickcube_constrain_rollout/short_rollout.json`.

### Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Ticket 03 Base DP final EMA | `b379573970712bcc17279ed237602a7be823acce9f7134ba125041ba5478389c` |
| Formal obstacle HDF5 | `fa846f52b5ae08de0243cfa52b40a4b0e7ce160881aa9db59ebaf4952c02904e` |
| Formal manifest | `8ef0d2739d6bf5675316ab37f42f1437cc19d80739e6f9102973beeabd6d1e58` |
| Pre-update run config | `f2f0269cfff226758c22d6471d18e8adef798b37c91c0751a170ed00f973e17d` |
| 30,000-step loss log | `d4ff41963b4e2c9dcd54b67b8fd08d20731535e1040a4a19d3f57e971ef5af48` |
| Training/verification summary | `0a3d64526d531538895eadfe12ac99d3cf0b4db0729b1c685d2fedf3cee65d44` |
| Final standalone EMA checkpoint | `da94d795f9cc5622b9ad87c7c19178e3e05ec1e93837bfe673400a71208d0ef4` |
| Short rollout evidence | `819631d0d8d49aef762754bc98d677207127c0a1596c14934e091b90b1c5deb4` |

### Reproduction

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 10800s \
  python -m examples.baselines.constrain.pickcube_constrain \
  --demo-path demos/PickCube-obstacle/20261022_132753_formal/trajectory.h5 \
  --manifest demos/PickCube-obstacle/20261022_132753_formal/trajectory.json \
  --base-checkpoint outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt \
  --run-dir outputs/exp/YYYYMMDD_HHMMSS_pickcube_constrain \
  --checkpoint-dir outputs/ckpt/YYYYMMDD_HHMMSS_pickcube_constrain \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 \
  --lr 0.0001 --warmup 500 --ema-power 0.75
```
