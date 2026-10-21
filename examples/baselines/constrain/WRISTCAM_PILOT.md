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

## End-to-end collection and simulation throughput (round 3)

This round used the round-2 geometry-prior collector as its baseline, followed
by one bounded optimization and one matching real 5-trajectory run. Both runs
used the same machine, seed 0, target 5, reuse 1, at most 96 templates, 100
physical episodes, 40 cylinder attempts per reuse slot, and the unchanged 500
control-step cap. Outputs are separate; the round-2 assets above remain intact.
No robot, reset, waypoint, speed, clearance, physics, or safety settings were
changed.

### Baseline measurement and bottleneck

The new baseline was collected to
`outputs/wristcam_pilot_20261021_round3_baseline/`. Measurement boundaries:

- Process wall time is `/usr/bin/time` around Python process startup through
  process exit, including imports, environment creation, HDF5 recording,
  aggregate/report JSON serialization, and cleanup.
- Collector elapsed is the `collect()` timer captured at the start of
  `write_report()`, **before** final report JSON serialization and atomic rename.
  The round-3 text above incorrectly called this “through report write”; it does
  not include that final write. Process wall time does include it. Existing stage
  timers cover sanity, no-obstacle template recorder calls, cylinder geometry
  sampling, and obstacle recorder calls. Template/obstacle calls include env
  creation, reconfigure/reset, control and safety checks, per-step recording,
  HDF5 flush, and close; the recorder's `elapsed_seconds` is narrower and
  excludes env creation and final flush. cProfile is needed to distinguish costs
  within those stage boundaries.
- The end-of-run `report.json` was 20,228,241 bytes. Its 17 successful template
  traces retained 165,597 unique swept-footprint points both in `trace` and in
  a duplicate top-level `footprint` field. This was repeated report data, not
  additional safety coverage. It identified a concrete serialization/storage
  cost that could be removed without changing geometry or RNG consumption.

| Measure | Round-3 baseline | Optimized | Denominator / boundary |
| --- | ---: | ---: | --- |
| Process wall | 32.91 s | 33.03 s | process start through exit |
| Collector elapsed | 30.68 s | 31.02 s | `collect()` start through final report serialization start |
| Sanity recorder stage | 1.39 s | 1.32 s | one sanity attempt |
| Successful template recorder stage | 21.61 s | 21.76 s | 17 template attempts |
| Cylinder geometry sampling | 1.79 s | 1.91 s | 585 sampler calls |
| Obstacle recorder stage | 5.83 s | 5.96 s | 5 physical attempts |
| Obstacle control steps | 491 | 491 | sum across 5 attempts |
| Control steps / obstacle-stage second | 84.2/s | 82.4/s | 491 / obstacle recorder stage; includes recorder overhead |
| Report JSON size | 20.23 MB | 10.88 MB | bytes on disk |

The timings show template collection dominates the measured stages (about 70%
of the collector elapsed); cylinder geometry is about 6%. However, the available
stage boundary includes setup, reset/reconfiguration, physics, per-step safety,
footprint recording, and HDF5 work, so this evidence does not identify which of
those dominates within a template. Obstacle-stage control-step rate is an
end-to-end recorder rate, not a pure simulation throughput benchmark. The
measured wall time did not improve (32.91 → 33.03 s, +0.4%); there is no evidence
of a simulation-kernel speedup, and this small single-run difference is timing
noise. No safety checks were removed or weakened.

### Minimal optimization and behavior equivalence

The template footprint remains in its existing full `trace.footprint` record,
which is also the sole source used to construct the sampler's footprint array.
The duplicate top-level template `footprint` copy was removed. This keeps every
point, exact cKDTree query input, prior, geometric filter, sampling order, and
random-number consumption unchanged. It removes one JSON serialization of
165,597 points (report size fell 46.2%); cylinder calls, layouts, selected
templates, control steps, and safety outcomes matched the baseline exactly.
The improvement is report size/serialization work only. The measured end-to-end
time did not get faster, and no claim of simulation throughput improvement is
made.

### Round-3 optimized trial, validation, and reproduction

Optimized artifacts are in
`outputs/wristcam_pilot_20261021_round3_throughput_optimized/`. Results: 17/17
no-obstacle template stacks; 585 sampler calls (580 `no_feasible_cylinder`, 5
feasible placements); 5/5 physical attempts safe and successful, from five
distinct templates; 491 obstacle control steps; no failed template or physical
attempt. The 5/17 output-template ratio describes only this finite pilot.
Sampling stage was 1.91 s. The run reached 5/5 within all specified budgets.

Validation evidence is retained as `validation.json`: five safe successes, five
unique output templates, seven-channel actions, 51-dimensional observations,
zero rotation channels, and T+1 rows for each state/observation leaf. The
existing `ConstrainDemoDataset` reader loaded all five, producing 486 training
windows with 51D observations and 4D actions; no training was run. The
representative `dataset.mp4` contains all 496 saved frames (about 1.00 MB).

Red/green: the new collection regression test failed before the optimization
because the template footprint appeared both at the top level and in its trace;
it passed after the duplicate was removed, with the full 2D trace footprint
still available. Targeted collector tests passed (24 tests); the full required
suite passed (149 tests in 40.68 s).

Exact commands, from the repository root (each workload is hard bounded):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
/usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' timeout --kill-after=5s 600s \
  python -m examples.baselines.constrain.collect_constructive \
  --output-dir outputs/wristcam_pilot_20261021_round3_baseline \
  --target-successes 5 --reuse-count 1 --max-templates 96 \
  --max-episodes 100 --max-cylinder-attempts 40 --seed 0
/usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' timeout --kill-after=5s 600s \
  python -m examples.baselines.constrain.collect_constructive \
  --output-dir outputs/wristcam_pilot_20261021_round3_throughput_optimized \
  --target-successes 5 --reuse-count 1 --max-templates 96 \
  --max-episodes 100 --max-cylinder-attempts 40 --seed 0
timeout --kill-after=5s 30s python -m examples.baselines.constrain.validate_wristcam_pilot \
  --dataset-dir outputs/wristcam_pilot_20261021_round3_throughput_optimized
timeout --kill-after=5s 180s python -m examples.baselines.constrain.render_dataset_video \
  --dataset-dir outputs/wristcam_pilot_20261021_round3_throughput_optimized
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=5s 180s \
  python -m pytest examples/baselines/constrain/tests -q
```

**Decision requested:** review the optimized pilot and decide whether to
proceed to Ticket 02. No formal 100-trajectory collection, training, or merge
was started; fresh user confirmation is required before any of them.

## End-to-end profiling and measured footprint optimization (round 4)

This round addresses the remaining request to profile the actual full collection
and make a measured attempt at simulation/collection throughput. Ticket 02 and
formal collection/training remain unstarted. It retains the round-3 baseline and
optimized datasets. Round-4 output is
`outputs/wristcam_pilot_20261021_round4_footprint_optimized/`.

### Profile method and findings

The full bounded seed-0, target-5 collection was run under `cProfile` at
`outputs/wristcam_pilot_20261021_round4_profile_run/`; the binary
`outputs/wristcam_pilot_20261021_round4_profile.prof` is the machine-readable
profile. `cProfile` adds material overhead, so its 36.83s profiled function time
is diagnostic only; unprofiled `/usr/bin/time` process wall is the comparison
metric. The profile records 2,234 control calls and 11,170 physics-substep
callbacks. Cumulative profile results (call counts and seconds) were:

| Operation | Calls | Baseline cumulative s | Scope / interpretation |
| --- | ---: | ---: | --- |
| Env `__init__` / `gym.make` | 23 each | 13.85 | includes initial scene setup; nested work, do not add to other rows |
| `reset` → `_reconfigure` | 46 | 15.95 / 15.73 | each attempt first resets then auto-reconfigures; nested cumulative values |
| Scene setup | 46 | 5.51 | SAPIEN scene rebuild portion of reconfigure |
| `gc.collect` during clear | 69 | 6.60 | Python GC invoked from environment clear |
| Recorder-wrapped control `step` | 2,234 | 11.24 | includes task control and simulation, not a pure PhysX kernel timer |
| Per-substep forbidden-crossing check | 11,170 | 5.27 | safety logic; 5.20s in optimized profile |
| Per-substep obstacle contact check | 11,170 | 0.16 | contact scan remains enabled |
| Template swept-footprint extraction | 1,361 | 1.89 | Python/Numpy transform/filter work, includes cProfile overhead |
| Cylinder geometry sampler | 585 | 2.05 | actual geometry sampling, independent stage timer below is 1.79s |
| JSON dump calls | 126 | 1.07 | all metadata/report writes under cProfile |
| Recorder HDF5 flush | 23 | 0.04 | does not include all recorder buffer work or aggregate copying |

The template recorder stage is the largest end-to-end stage (~21.6s), but its
largest controllable internal hotspot found was repeated Python shape-level work
in swept-footprint extraction. Reconfigure/scene setup and Python GC are also
large, and task safety checks cost ~5.3 profiled seconds. They are not modified:
reusing/rebuilding environments or changing per-substep safety execution needs
separate lifecycle/safety equivalence work. No claim is made that PhysX itself
was the dominant CPU cost.

### Optimization, red/green, and measured result

The `collect_footprint` seam now caches each collision shape's vertices already
transformed into its link frame, concatenated once per link. Each physics step
therefore performs one live link transform and one height filter per link rather
than Python-looping over every collision shape. The complete vertices remain
included, including the wrist camera; the z cutoff, cube corners, footprint
consumers, geometry tests, safety checks, and sampling prior are unchanged. The
test first failed against the old cache shape (the link entry contained a list
instead of the required homogeneous per-link vertex array), then passed after
the implementation. The real wristcam cache/footprint regression and full suite
also pass.

Baseline and round-4 runs have the same seed/config and identical candidate,
success, sampling-call, physics-attempt, and control-step counts, so this is the
same observed work, not an acceptance-rate or workload reduction:

| Measure | Round-3 optimized baseline | Round-4 optimized | Difference |
| --- | ---: | ---: | ---: |
| Candidates / no-obstacle successes | 17 / 17 | 17 / 17 | identical |
| Cylinder calls / no-feasible | 585 / 580 | 585 / 580 | identical |
| Obstacle attempts / safe successes | 5 / 5 | 5 / 5 | identical |
| Obstacle control steps | 491 | 491 | identical |
| Sanity / template / sampling / obstacle stage s | 1.317 / 21.764 / 1.910 / 5.956 | 1.344 / 21.601 / 1.783 / 5.914 | template -0.8%; sampling -6.7% |
| `collect()` elapsed field s | 31.024 | 30.703 | -1.0%; stops before final report write |
| Process start-to-exit wall s | 33.03 | 32.73 | -0.9%; includes imports, initialization, output writes, exit |
| Profiled `collect_footprint` cumulative s | 1.886 | 1.686 | -10.6%, under cProfile |
| Obstacle steps / obstacle-stage second | 82.4/s | 83.0/s | +0.7%; stage includes recorder overhead |

This is a small measured improvement, not evidence of a major simulation-kernel
speedup: stage and process changes are near run-to-run noise, and control-step
throughput improved only 0.7%. Safety checks and simulator configuration work
remain prominent. No safety precision, native wristcam reset, physics/control
frequency, solver settings, expert behavior, trajectory state coverage, or
collection budgets were reduced. Same seeded layout/selected cylinder/action
comparison is preserved by the unchanged RNG call sequence; per-step states are
not asserted bit-identical because floating-point execution can vary.

### Validation and reproduction

The round-4 validator passed: 5 unique templates, 5/5 stack successes with no
collision/crossing, 7-channel actions, 51D observations, exact zero rotation,
T+1 state/observation rows, and 486 windows loaded through the existing dataset
reader. `validation.json` records the evidence. The exact bounded commands were:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
/usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' timeout --kill-after=5s 600s \
  python -m cProfile -o outputs/wristcam_pilot_20261021_round4_profile.prof \
  -m examples.baselines.constrain.collect_constructive \
  --output-dir outputs/wristcam_pilot_20261021_round4_profile_run \
  --target-successes 5 --reuse-count 1 --max-templates 96 \
  --max-episodes 100 --max-cylinder-attempts 40 --seed 0
/usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' timeout --kill-after=5s 600s \
  python -m examples.baselines.constrain.collect_constructive \
  --output-dir outputs/wristcam_pilot_20261021_round4_footprint_optimized \
  --target-successes 5 --reuse-count 1 --max-templates 96 \
  --max-episodes 100 --max-cylinder-attempts 40 --seed 0
timeout --kill-after=5s 30s python -m examples.baselines.constrain.validate_wristcam_pilot \
  --dataset-dir outputs/wristcam_pilot_20261021_round4_footprint_optimized
timeout --kill-after=5s 180s python -m examples.baselines.constrain.render_dataset_video \
  --dataset-dir outputs/wristcam_pilot_20261021_round4_footprint_optimized
python -c "import pstats; p=pstats.Stats('outputs/wristcam_pilot_20261021_round4_profile.prof'); p.strip_dirs().sort_stats('cumulative').print_stats(35)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=5s 180s \
  python -m pytest examples/baselines/constrain/tests -q
```

The profile run was 37.42s process wall; its cProfile summary is diagnostic and
not compared as an unprofiled runtime. The unprofiled optimized collection was
32.73s process wall and 30.703s through the collector timer boundary. Final
report serialization is after that boundary; process wall correctly includes
it. The renderer produced the full 496-frame `dataset.mp4` within the 180s limit.
Validation passed, and the full constrain suite passed (149 tests in 40.48s).
No formal collection, training, or merge was started; this report requests user
review before Ticket 02.
