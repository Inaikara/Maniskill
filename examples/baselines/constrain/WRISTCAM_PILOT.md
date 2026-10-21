# Native wristcam pilot

**Status:** pilot complete; 5/5 target trajectories collected. This report is
for user review; no formal collection or training was started.

## Contract and pre-run budget

The Base DP filtered demonstration metadata is
`demos/StackCube-v1/motionplanning/trajectory.filtered.state.pd_ee_delta_pos.physx_cpu.json`.
Its `env_info` records `StackCube-v1`, `obs_mode=state`,
`control_mode=pd_ee_delta_pos`, `sim_backend=physx_cpu`, `num_envs=1`, and the
registered 50-step environment limit. The source task
`mani_skill/envs/tasks/tabletop/stack_cube.py` defaults to
`robot_uids=panda_wristcam` and `robot_init_qpos_noise=0.02`; its agent root is
at `(-0.615, 0, 0)`. The pilot explicitly uses that robot and noise, but does
not replace its reset pose or quaternion. `pd_ee_delta_pose`, state observations,
500 control steps, CPU physics and one environment are pilot-specific settings.
Wristcam-link collision shapes are included by footprint extraction.

Before the first collection process, the following ceilings were selected:

| Budget | Value |
| --- | ---: |
| Safe successful trajectories | 5 |
| Accepted reuses per template | 1 |
| Candidate templates | 96 |
| Physical obstacle episode attempts | 100 |
| Cylinder sampling calls per reuse slot | 40 (20 preferred + 20 fallback) |
| Control steps per episode | 500 |
| Collection process wall time | 600 seconds, hard timeout |

These bounds allow a pilot-sized attempt near the previous 96-template
reference without multiplying recordings through template reuse. No budget will
be raised after exhaustion. Waypoint or speed/clearance changes, if any, will be
finite, described here before the rerun, and reported separately. The configured
pilot values are recorded with each collection report.

## Results

Run in the activated `maniskill` conda environment:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=5s 180s \
  python -m pytest examples/baselines/constrain/tests -q

timeout --kill-after=5s 600s python -m examples.baselines.constrain.collect_constructive \
  --output-dir outputs/wristcam_pilot_20261021_verified \
  --target-successes 5 --reuse-count 1 --max-templates 96 \
  --max-episodes 100 --max-cylinder-attempts 40 --seed 0

timeout --kill-after=5s 180s python -m examples.baselines.constrain.render_dataset_video \
  --dataset-dir outputs/wristcam_pilot_20261021_verified
```

Collection completed successfully in 169.2 s (inside the 600 s hard timeout).
No waypoint, speed, lift, or clearance parameters were tuned. The final values
are the existing defaults recorded in `report.json`; thus the initial TCP
orientation came from each native reset and all commanded rotation channels
remained zero.

| Stage / measure | Result | Denominator |
| --- | ---: | --- |
| Candidate templates attempted | 9 | templates considered before reaching target |
| No-obstacle expert stack successes | 9 | 9 candidate templates (100%) |
| Cylinder sampling calls | 265 | across those templates |
| No feasible cylinder | 260 | 260/265 sampling calls rejected geometrically |
| Feasible cylinder placements selected | 5 | one selected per produced trajectory |
| Obstacle physics attempts / safe successes | 5 / 5 | 5 feasible placements executed (100%) |
| Distinct templates with output | 5 | 5 accepted trajectories; 1 reuse each |
| Stage elapsed (sanity / templates / sampling / physics) | 1.36 / 11.62 / 149.73 / 6.43 s | stage timings in `report.json` |
| Total elapsed / accepted per minute | 169.21 s / 1.77 | 5 accepted / total elapsed |

Failed sample reason: `no_feasible_cylinder` (260). There were no failed
no-obstacle templates and no failed physical obstacle attempts. Stage-time
components sum to the total within reporting overhead. The output is
`outputs/wristcam_pilot_20261021_verified/` (`trajectory.h5`, `trajectory.json`,
`report.json`, and `dataset.mp4`).

### Validation and interpretation

- The saved metadata resolves to `StackCubeObstacle-v1`, CPU PhysX, one
  environment, `panda_wristcam`, state observations, `pd_ee_delta_pose`, native
  `robot_init_qpos_noise=0.02`, and 500 steps. The wrapper applies no custom
  robot pose/quaternion reset.
- All five records have 7-channel actions (xyz, three zero rotations, gripper),
  51-dimensional observations, and T+1 states/observations. Every rotation
  maximum is exactly zero. Episode lengths are 100, 88, 92, 92, and 90 steps.
- Every episode has `stack_success=true`, `obstacle_collision=false`, and
  `obstacle_crossing=false`. The existing 4D `ConstrainDemoDataset` reader
  loaded all five trajectories into 457 training windows; this was validation
  only and no training was run.
- `dataset.mp4` rendered all 467 saved frames in about 2 s from recorded states.

The historical references are 82/96 no-obstacle template successes (85.4%),
50/96 final successful template outputs (52.1%), and 100/100 physics successes
after geometric filtering. This pilot observed 9/9 no-obstacle successes,
5/9 output templates (55.6%), and 5/5 accepted physical runs after filtering.
These pilot rates are descriptive only: nine candidate templates are too few to
claim a stable acceptance rate. The former experiment retained two trajectories
per template; this pilot intentionally retained only one, so episode and
template-output denominators are not interchangeable.

### Development notes

The first collection process exited 1 during sanity artifact signature loading:
it assumed the articulation key `panda`, while the wristcam recording stores
`panda_wristcam`. Its temporary directory was cleaned and no aggregate files
were written. The signature reader now discovers the single recorded robot
articulation, with a dedicated regression test.

The first successful diagnostic pilot was discarded after a real-environment
check showed the camera link uses a PhysX box collision shape, which does not expose
`get_vertices()`. The footprint sampler now derives the box's eight local
vertices from its half-size and includes them with all robot-link collision
geometry. A real wristcam environment test checks that camera-link vertices are
present in the sampling cache. The verified final collection was run after this
correction, and those are the retained deliverables.

The test-first stack-success regression failed against the prior behavior, then
passed after `run_episode` began stopping at the first step-info
`stack_success`; sticky collision/crossing flags from that same step remain in
the returned trace and disqualify it from safe acceptance. Targeted collection
tests pass (21 tests); the full required suite passes (147 tests in 39.91 s).

The preliminary failed/diagnostic output directories were removed after the
verified run; only `outputs/wristcam_pilot_20261021_verified/` is retained.

## Supplemental geometry-prior optimization (round 2)

The user approved the original five trajectories. This supplemental run retains
the original artifacts above and writes to
`outputs/wristcam_pilot_20261021_geometry_prior_optimized/`. The new run is
awaiting a separate user review/approval. Ticket 02, formal 100-trajectory
collection, training, and branch integration have not been started.

### Change and distribution

Profiling evidence from the first run identified repeated dense
candidate-by-footprint distance arrays in `sample_cylinder` as the costly
geometry operation. The implementation now builds a SciPy `cKDTree` over the
complete recorded swept footprint and queries exact nearest footprint points
for each surviving candidate. Candidate lateral offsets now use a directional
prior: rank the two sides of the direct A→B segment by the nearest footprint
projection to a 0.04 m lateral reference, prefer the less occupied side, and
draw offset magnitude uniformly from 50–100% of the unchanged blocking limit.
Longitudinal `t` remains uniform over 0.30–0.70; configured radius, endpoint,
box, direct-blocking, full footprint, and clearance tests are unchanged. The
selected center is still only a geometric candidate and must pass real physics
and safety flags. This deliberately changes the cylinder center distribution
toward the corridor opposite the recorded sweep; it does not change cube,
waypoint, reset, expert, or safety distributions/parameters. The resolved prior
and SciPy-backed exact query are recorded in the new `report.json` config.

Red/green: the new seeded geometry behavior test failed against the old
symmetric sampler (positive-side fraction 0.633 versus the required 0.8), then
passed with the conditioned prior; existing exact geometry tests passed. The
full required suite passed: 148 tests in 39.50 s.

### Same-seed measured comparison

Both trials use seed 0, target 5, reuse 1, at most 96 templates, 100 physical
episodes, 40 cylinder sampling calls per reuse slot, and 500 steps per episode.
Neither trial changed waypoint, speed, lift, reset, radius/box bounds, or safety
thresholds.

| Measure | Original verified run | Geometry-prior run | Denominator / interpretation |
| --- | ---: | ---: | --- |
| Candidate templates | 9 | 17 | considered before target |
| No-obstacle expert successes | 9 | 17 | successes / candidates: 9/9; 17/17 |
| Distinct output templates / accepted trajectories | 5 / 5 | 5 / 5 | each template retained once |
| Cylinder sampling calls | 265 | 585 | calls through successful target completion |
| `no_feasible_cylinder` | 260 | 580 | failed calls / calls: 260/265; 580/585 |
| Geometrically feasible placements | 5 | 5 | selected placements |
| Obstacle physics attempts / safe successes | 5 / 5 | 5 / 5 | successes / attempts: 100% both |
| Sampling stage | 149.73 s | 1.80 s | cumulative actual collection stage |
| Total elapsed | 169.21 s | 30.45 s | cumulative end-to-end collection |

Sampling time decreased by 98.8% (about 83.4×), despite more candidate templates
and more sampler calls. End-to-end time decreased by 82.0% (about 5.6×). The
sampler-call geometric yield was 5/265 (1.89%) and 5/585 (0.85%); the higher
call count is exposed rather than hidden. The actual five-trajectory sampling
stage met the 60-second goal. These small pilot outcomes do not establish stable
acceptance rates.

### Artifact checks and reproducibility

The saved `validation.json` is machine-readable evidence from the following
validation command. It checks the reported five distinct templates, all five
stack-success/no-collision/no-crossing flags in both report and trajectory
metadata, 7D actions, 51D observations, T+1 states/observations, exactly zero
rotation channels, and successful 4D/51D `ConstrainDemoDataset` loading (486
training windows; no training run):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
timeout --kill-after=5s 30s python -m examples.baselines.constrain.validate_wristcam_pilot \
  --dataset-dir outputs/wristcam_pilot_20261021_geometry_prior_optimized
```

The render command completed within its 180-second limit and produced
`dataset.mp4` (496 frames). The original verified artifact directory remains
intact. The prior and algorithm details above should be frozen before any
subsequent larger collection; this report requests fresh user confirmation
before Ticket 02.
