# PlaceSphere Base DP (Ticket 09)

## Frozen task and data contract

`placesphere_base_dp.py` and `train_placesphere_base_dp.py` implement an
independent, fresh-initialized official ConditionalUnet1D Base DP. The training
run loaded no checkpoint. Its only demonstrations are the 1,000 formal direct
episodes with IDs 0–999 from Ticket 08; all 100 obstacle episodes and all pilot
episodes are excluded. The frozen training source identities, complete
configuration, selected IDs, and dataset hashes were recorded before update 1
in `outputs/exp/20261022_193900_placesphere_base_dp/`.

The runtime named state has 42 raw values: `agent/qpos:9`, `agent/qvel:9`,
`extra/is_grasped:1`, `extra/tcp_pose:7`, `extra/bin_pos:3`,
`extra/obj_pose:7`, `extra/tcp_to_obj_pos:3`, and the named
`extra/placesphere_obstacle:3`. The 39-D policy observation is formed by
excluding that obstacle field. At runtime the exact live field order is checked;
the adapter takes only this verified named base-state prefix, and condition
tests show changing obstacle values cannot change the base condition.

Actions retain all seven native `pd_ee_delta_pose` channels:
`[delta_x, delta_y, delta_z, delta_euler_x, delta_euler_y, delta_euler_z,
gripper]`. Identity normalization is used for dataset actions. The official
diffusion output is clipped by its scheduler to `[-1,1]`; the PlaceSphere
execution adapter additionally clips the six pose values to the formal expert's
native `[-0.35,0.35]` range and retains gripper in `[-1,1]`. Rotations are not
removed or replaced. The gripper convention is close `-1`, open `+1`.
The frozen runtime uses Panda with `robot_init_qpos_noise=0.02`, single-env
PhysX CPU, and a 200-control-step episode limit. Translation is XYZ delta
divided by 0.1 m and clipped to ±0.35; XYZ Euler delta is divided by the signed
runtime controller `rot_lower=-0.1` and clipped to ±0.35. The frozen obstacle
is a static cylinder of radius 0.03 m and height 0.08 m.

The dataset loader asserts exact manifest IDs/classes, HDF5 class tags, saved
policy/raw observation relation, `T+1` states, finite legal actions, and native
successful termination. Official window alignment repeats the first
observation/action for initial history padding; every saved release and retreat
action remains part of its episode. Only action positions beyond the real
terminal action sequence are padded, with six zero pose coordinates and open
gripper `+1`. This avoids inheriting PickCube's hold-object padding convention.

The model contract is 39 condition dimensions, seven action dimensions,
observation/prediction/execution horizons 2/16/8, 64/128/256 UNet channels,
64-D timestep embedding, GroupNorm 8, and 100-step squared-cosine DDPM with
epsilon prediction. State/action normalization is identity. Training used seed
1, 30,000 updates, batch 1,024, AdamW (`lr=1e-4`, betas `(0.95,0.999)`, weight
decay `1e-6`), 500-step warmup then cosine schedule, EMA power 0.75, and final
EMA chosen before training. No task evaluation or metric-selected checkpoint
was used.

## Training and checkpoint evidence

The separate non-formal 45-second CUDA probe made 1,821 updates at 40.47
updates/s and peaked at 540,006,400 CUDA bytes. It is stored under
`outputs/exp/20261022_193725_placesphere_base_dp_probe/` and did not contribute
formal training steps.

The formal run completed 30,000/30,000 actual updates in 972.88 seconds. All
30,000 logged losses are finite: first `1.16113663`, last `0.00238671`, minimum
`0.00204123`, maximum `1.16113663`. Initial raw model SHA-256
`d5de206525d193c028cd7d751709b12791e96211810dc07f405e20e89f69cec6` differs
from final raw model SHA-256
`6d54ba1e32c40a7c021b8c21fdaed63feb7a7dd0b3c22fa8337a8bdafbabe466`.
The final EMA checkpoint is independently loadable at
`outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt`.

| Artifact | SHA-256 |
| --- | --- |
| Formal HDF5 | `fec4620c496ae3c5c0dec7db469333f302f7170363e11f66cba961cbb5d56ec4` |
| Formal manifest | `740379c6a575a470f930b249494cb603a1ff36e2c71c5235e93d6eb3ec4c11ba` |
| Frozen data contract | `ff11b940ede86e290928d26ccd4ac96e7ba480995af774d46a018dbda8b6e761` |
| Collection report | `d861bf23e813e06e77f8791cd11e94844b000de8586f5169d33467f2c8df4b67` |
| Final run configuration (completed status) | `31a36c45d3d3b9882f814832626bea332d5aa84bd1c00697549e1de7e9dab342` |
| Per-update loss JSONL | `466e51621567f4f5b49a983cb48e5c2917d2c91cab2d2c2c36c17527f3dbdd4d` |
| Final EMA checkpoint | `07f93942cfcfd7d28120b0b210fd7c323098140f670bbfce4c9a83ff76b9fb30` |

The run config includes every selected episode ID, the four trusted data
identities, resolved model/training parameters, output paths, runtime named
schema, and the executed source hashes. Exact source files are archived in
`source_snapshot/`, including the official builder `diffusion_policy/train.py`.
The run summary's config hash
(`7b017851d2a9525bf7f933153f2b1530ac3ca99862b5d4df26ad9dcae2382383`)
identifies its pre-finalization
config snapshot; the final on-disk completed run config is hash-pinned in the
table above.

## Strict loading and short obstacle rollout

`PlaceSphereBaseDPAdapter` validates its dedicated format tag and complete
training contract, data hashes, exact direct IDs, named observation schema,
task dimensions and strict state-dict shapes. It rejects PickCube checkpoints,
incorrect task/model/horizon/action contracts, altered data identities, and
non-trusted formal HDF5/manifest/report identities. It exposes a strict
`initialize_constrain_base(denoiser)` seam for a future matching 39-condition,
7-action PlaceSphere ControlNet.

The independent short rollout loaded the final checkpoint from disk and chose
formal obstacle episode 1000 and its scene before inference. It restored the
full saved native state and controller, then ran 32 real native actions in the
fresh obstacle geometry. Restore max absolute differences were `1.4901161e-8`
for native state, `0` for controller state, and `0` for policy observation,
within `1e-7`, `1e-7`, and `1e-6` respectively. All 32 actions had shape `(7,)`
and remained in the formal native action ranges. Obstacle perturbation left the
base condition invariant. A second, fresh CPU load returned finite `(1,8,7)`
actions within native execution bounds.

The recorded 32-step outcome is `native_success=false`, `success=false`,
`is_obj_on_bin=false`, `is_obj_static=true`, `is_obj_grasped=false`, with no
obstacle contact or forbidden crossing. This is an interface diagnostic, not a
success gate or success-rate estimate. The first rollout before adding the
native pose bound is retained at
`outputs/exp/20261022_193900_placesphere_base_dp_rollout/`; after that run,
the exact pre-run source remained archived and the focused mapping delta was
captured as `post_training_native_action_bounds.diff`. The corrected rollout is
at `outputs/exp/20261022_195600_placesphere_base_dp_rollout/` (SHA-256
`5f690a178679dadef5a3a0e6e87a776a392d0278ed1fee25f6a8bc4aa2128d98`).

## Reproduction commands

Run from repository root with `maniskill` active. These paths identify the
completed run; choose a new timestamp for a new run because outputs are
write-once.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 10800s \
  python -m examples.baselines.constrain.train_placesphere_base_dp \
  --demo-path demos/PlaceSphere-obstacle/20261022_184558_formal/trajectory.h5 \
  --manifest demos/PlaceSphere-obstacle/20261022_184558_formal/trajectory.json \
  --run-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_base_dp \
  --checkpoint-dir outputs/ckpt/YYYYMMDD_HHMMSS_placesphere_base_dp \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 \
  --lr 0.0001 --warmup 500 --ema-power 0.75

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 300s \
  python -m examples.baselines.constrain.rollout_placesphere_base_dp \
  --checkpoint outputs/ckpt/YYYYMMDD_HHMMSS_placesphere_base_dp/final.pt \
  --device cuda --steps 32 \
  --output-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_base_dp_rollout

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 600s \
  python -m pytest examples/baselines/constrain/tests -q
```

The completed regression command passed all 257 tests. No formal PlaceSphere
policy evaluation or ControlNet training was run.
