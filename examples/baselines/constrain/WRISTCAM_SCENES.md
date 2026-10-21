# Native wristcam feasible evaluation scenes — Ticket 03

## Scope and frozen generation contract

This artifact is a **randomly generated, expert-verified feasible scene
evaluation set**. It does not claim training-set isolation or universal planning
solvability. A rejected candidate is only a failure under the bounded collector
and its stated expert; it is not evidence that the scene is unsolvable.

The scene set reuses the frozen Ticket 02 action-first generator, expert,
sampling prior, exact cKDTree swept-footprint clearance, layout and radius
ranges, waypoint/speed/lift parameters, and safety acceptance. Frozen config
fingerprint (formal seed 0):
`a5390e1777aa3ea0f708bf8b414d38ac6727ed520c53e3ce53ef9c05da19d140`.
All four generator source hashes are verified against the Ticket 02 freeze by
the builder and are embedded in `manifest.json` and `build_identity.json`.
The evaluation-set run uses seed `10000`; its run-specific config fingerprint
is `d62bb29e012eaa822ac4d2c5fbc9b92451415c78023e7806232b381f2bc78841`.

Environment contract: `StackCubeObstacle-v1`, native `panda_wristcam` reset
pose/quaternion and `robot_init_qpos_noise=0.02`, `state` observations,
`pd_ee_delta_pose`, CPU PhysX, one environment, zero commanded rotation, and
500 control steps. Success stops at the first `stack_success`; a same-step
collision or forbidden crossing is rejected. Only a successful no-obstacle
template and a subsequent safe expert obstacle execution are retained, with one
layout per template.

Finite run limits were target 100 scenes, reuse 1, 2,000 templates, 400
physical attempts, 40 sampler calls per template slot, 500 steps per episode,
and a 3,600-second process timeout. The run used 744.08 seconds process wall
time (collector elapsed 735.79 seconds; the process includes serialization and
exit). No budget was expanded.

## Collection results

Current directory: `outputs/exp/20261021_wristcam_scenes/` (see [OUTPUTS.md](OUTPUTS.md)).
Expert-success trajectories and metadata remain under its `witness/` child;
the large intermediate template report was removed during cleanup. The independent
scene store is `scenes.h5` plus `manifest.json`; loading it does not read
`witness/`, `trajectory.h5`, the training demonstrations, or any earlier output.

| Measure | Result | Denominator / meaning |
| --- | ---: | --- |
| Candidate templates | 420 | bounded candidates reached before 100 outputs |
| No-obstacle expert successful templates | 392 | 392/420 (93.33%) |
| Candidate failures before a successful template | 28 | 27 `convergence_timeout`, 1 `waypoint_infeasible` |
| Cylinder sampler calls | 13,666 | across candidate templates until target |
| `no_feasible_cylinder` rejects | 13,566 | rejected calls; exact frozen geometry filters retained |
| Feasible placements selected | 100 | one for each accepted template |
| Obstacle expert attempts / safe successes | 100 / 100 | all 100 had stack success and no collision/crossing |
| Distinct output templates / scenes | 100 / 100 | reuse count one |
| Episode control steps | 83–107 | stopped at first stack success |
| Sanity / templates / sampling / obstacle stage | 1.28 / 565.50 / 40.13 / 126.90 s | collector stage timers |

The 100 saved layouts have cube separation 0.2501–0.2990 m, cylinder radius
0.0200–0.0397 m, and recorded geometric swept-footprint clearance
0.0150–0.0315 m. Cube A XY spans x `[-0.1483, 0.1797]`, y
`[0.0505, 0.2992]`; cylinder center spans x `[-0.0065, 0.1649]`, y
`[0.1510, 0.2250]`. These are descriptive sample statistics, not extra
acceptance filters. No deduplication, distance filtering, or isolation check was
performed.

## Independent scene-store and restoration API

`SceneSet(directory).load(index)` returns a `Scene` with its full nested
`initial_state`, initial observation, scene parameters and witness metadata.
The HDF5 state includes every `env_states` leaf from row zero (robot
articulation, both cubes, and obstacle center/radius). The manifest binds each
scene to hashes of all initial-state leaves and of its robot/cube state; it also
records the cube layout, full cylinder geometry, seed, expert flags/steps,
environment contract and frozen generator source/config identities. The
manifest includes a stable scene-data SHA-256 and a canonical manifest identity
(SHA-256 over the manifest payload excluding its own `manifest_sha256` field).
`SceneSet` checks the manifest/data and per-scene state hashes while loading.

Build the matching environment with the scene's recorded cylinder radius and
height before restoration. `restore_scene(env, scene)` then resets the native
task/controller and sticky safety trackers, and restores the complete state
dictionary. It checks the restored initial observation against the stored one.
Example for a shared Base DP / ConstrainNet evaluator:

```python
import gymnasium as gym
import examples.baselines.constrain  # register StackCubeObstacle-v1
from examples.baselines.constrain.scene_set import SceneSet, restore_scene
from examples.baselines.constrain.stackcube_obstacle import ENV_ID

scenes = SceneSet("outputs/exp/20261021_wristcam_scenes")
scene = scenes.load(0)  # no expert or collection is run here
cylinder = scene.metadata["cylinder"]
env = gym.make(
    ENV_ID, num_envs=1, obs_mode="state", sim_backend="physx_cpu",
    robot_uids="panda_wristcam", robot_init_qpos_noise=0.02,
    control_mode="pd_ee_delta_pose", obstacle_radius=cylinder["radius"],
    obstacle_height=cylinder["height"], obstacle_center=cylinder["center_xy"],
    max_episode_steps=500,
)
initial_observation = restore_scene(env, scene)
# Run any policy from this same restored observation/state.
env.close()
```

## Round-trip evidence and reproduction

`roundtrip_validation.json` records PASS for all 100 standalone loads and real
environment restorations. Validation ran no expert. Every state leaf differed
by at most `1.49e-8` (floating-point articulation normalization); every initial
observation differed by at most `1.20e-7`. All 100 cylinder collision radii
matched the constructed geometry within the environment's `1e-6` radius
contract. State, observation, witness safety, and geometry mismatches fail the
validator; candidates are never silently dropped.

Reproduction from repository root (commands run in the `maniskill` environment;
all workloads are explicitly time-bounded):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' \
  timeout --kill-after=5s 3600s python -m examples.baselines.constrain.build_wristcam_scenes \
   --output-dir outputs/exp/YYYYMMDD_HHMMSS_wristcam_scenes --seed 10000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -f 'ROUNDTRIP_WALL_SECONDS=%e' \
  timeout --kill-after=5s 600s python -m examples.baselines.constrain.validate_wristcam_scenes \
   --scene-dir outputs/exp/20261021_wristcam_scenes
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=5s 180s \
  python -m pytest examples/baselines/constrain/tests -q
```

Recorded SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| `scenes.h5` | `b1c736ccfbd4411b20dcb1495ffbf220afd6d7b08dbf46528c4047358d139caf` |
| `manifest.json` file bytes | `71381bfaad5c7e36235f457b3489e30d96f508c8d7bb17db037fae34eec65c89` |
| `manifest.json` canonical identity | `f1aa2175bd70a86ba522b5d441b1a9897330fdbd17a5484d7a06f5eccb8335b3` |
| `roundtrip_validation.json` | `5f4f810f5dc8bf3266f9666208b69d2e23fd1c645b868798b59da00733517ccf` |

The constrain test suite passed: 152 tests in 43.26 seconds. No policy training
or policy evaluation was run as part of this ticket.
