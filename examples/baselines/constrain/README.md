# StackCubeObstacle action-first data generation

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
