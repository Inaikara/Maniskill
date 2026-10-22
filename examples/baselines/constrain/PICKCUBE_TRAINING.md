# PickCube Base DP (Ticket 03)

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
