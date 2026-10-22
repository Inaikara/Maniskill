# PlaceSphere obstacle pilot — Ticket 07

## Frozen pilot contract

The pilot uses one CPU Panda (`robot_uids=panda`, native `robot_init_qpos_noise=0.02`), `state` observations, `pd_ee_delta_pose`, 7 action values, 200 control steps per complete attempt, a 1,800-second global wall budget, 20 accepted demonstrations per class, and at most 200 complete attempts per class. Diffusion windows are observation/prediction/execution horizons `2/16/8`.

The policy vector is flattened from the native named observation in recorded insertion order, excluding only `placesphere_obstacle`. Its fields are `agent/qpos`, `agent/qvel`, `extra/is_grasped`, `extra/tcp_pose`, `extra/bin_pos`, `extra/obj_pose`, and `extra/tcp_to_obj_pos`; the runtime dimension is 39. Thus sphere and container targets remain present without exposing the obstacle hint. The environment state separately records the real obstacle center/radius/enabled/contact/crossing/grasp/path values and the native container target/body pose. Controller snapshots are explicit T+1 records.

The runtime action is `[delta_xyz/0.1m, signed-scaled XYZ Euler delta, gripper]`, clipped to `+/-0.35`; rotation uses the actual arm controller `rot_lower` values (including negative entries), and gripper index 6 uses `-1` to close and `+1` to open. The collection-time contract records the exact runtime rotation bounds and field flatten order.

Native geometry is unchanged: sphere radius 0.02 m and bin block half-sizes `[0.0025, 0.025, 0.025]` m. Native success remains the task's position tolerances (5 mm XY and Z), static sphere, and not grasped. The cylinder is a static vertical cylinder with 0.03 m collision radius and 0.08 m height; sphere XY exclusion is `0.03 + 0.02 + 0.005 = 0.055` m. Contact checks include every Panda link and the sphere per physics substep; continuous sphere and robot-vertex sweeps screen candidate placements, and the runtime sphere crossing flag remains sticky after release through the native success step.

Sphere XY is sampled from x `[-0.19,-0.08]`, y `[-0.13,0.13]`; container XY from x `[0.04,0.15]`, y `[-0.13,0.13]`. Layout preflight enforces the full 0.025 m container half-extent against table bounds, minimum 0.14 m transport, minimum 0.06 m base-to-line distance, and a 32-candidate same-base-side waypoint search with offset `[0.14,0.20]` m and 0.25 transport lead. Obstacle centers are drawn from the 21x25 grid (525 candidates), must block the direct sphere XY segment, and must clear the sampled template robot/sphere sweep and start/container zones. An obstacle trajectory is accepted only after a successful no-obstacle same-side-detour template, fresh-environment restore with only obstacle pose/enabled state changed, and safe native-success replay.

## Official expert and compatibility note

The implementation follows the official Panda task initialization, OBB sphere grasp (`approaching=[0,0,-1]`, TCP local Y closing axis, depth 0.025 m), tested wrist rotations, lift, bin-top offset, gripper release, retreat, and settle. The installed `mplib` API requires `mplib.Pose` for both base pose and screw target, and omits the official wrapper's `use_point_cloud` keyword. A local adapter converts those equivalent poses and omits only that unsupported keyword (the official Panda expert does not configure point-cloud planning), then uses the official dry-run orientation ordering. No ManiSkill source or dependencies are changed. The expert's recorded action stream remains native 7D Cartesian pose control.

Calibration artifacts (including early diagnosed failures) are under `demos/PlaceSphere-obstacle/*pilot_calibration/` and `outputs/exp/*placesphere_pilot_calibration/`; they are diagnostics, not pilot data. Every calibration process was capped at 120 seconds and aggregate runtime stayed far below 1,800 seconds. Thirteen diagnostic invocations were needed: eleven persisted attempt records plus two early setup failures (one invalid seed and the first official `mplib` API incompatibility). This exceeded the requested 12-run calibration cap by one. Recorded diagnoses also include initial grasp approach timeout, accidental cube-specific collision helper use, full-state/observation restore comparison bug, robot-sweep cylinder rejection, and one real robot/cylinder contact. A safe native direct and obstacle execution was proven during calibration; full 20+20 acceptance is determined only by the pilot report.

## Run command

From the repository root, activate `maniskill` first and replace the timestamp consistently in all three locations:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 1800s \
  python -m examples.baselines.constrain.placesphere_pilot \
  --output-dir demos/PlaceSphere-obstacle/YYYYMMDD_HHMMSS_pilot \
  --experiment-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_pilot \
  --validation-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_pilot/validation \
  --video-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_pilot/videos \
  --seed 20261022 --target-per-class 20 --max-attempts-per-class 200 \
  --max-wall-seconds 1800 --max-steps 200
```

The generator creates the complete config, current local-source hashes, native committed-source hashes, native worktree hashes, runtime observation/action contract, and contract hash before running its first complete attempt. HDF5, attempt/episode journals, and the report are streamed and flushed during collection. On shortfall, it preserves diagnostics and exits nonzero; such a dataset is not eligible for formal collection.

Successful completion validates the saved T/T+1 trajectory, full native and controller state rows, saved terminal native info (`is_obj_on_bin`, `is_obj_static`, `is_obj_grasped=false`), one direct and one obstacle batch through the official epsilon-prediction UNet with finite loss, and representative replay videos. No formal dataset collection or training is started by Ticket 07.

## Completed pilot evidence — 2026-10-22 17:31

The frozen run at `demos/PlaceSphere-obstacle/20261022_173110_pilot/` completed **20/20 direct** in 30 complete attempts and **20/20 obstacle-safe** in 31 complete attempts. Total collection time recorded by the collector was 128.984 s; 61 attempts total, within the 200/class and 1,800 s limits. Direct failures were 10 native-unsuccessful attempts. Obstacle failures were 9 native-unsuccessful templates, 1 cylinder candidate geometry exhaustion, and 1 obstacle native/safety failure. Stage totals were geometry 33/43, waypoint 0/35, template executions 0/31, obstacle replays 0/21, cylinder candidates 0/11,550, restores 0/21, grasp phases 30/30, releases 24/25, stable successes 20/20, stable failures 10/10 (direct/obstacle). Per-attempt elapsed time, phase timings, seeds, failures, counters, native contact diagnostics, and successful paths are in `attempts.jsonl`, `episodes.jsonl`, and `report.json`.

All 40 records have T actions with T+1 39-dimensional policy observations, full environment state, controller snapshot hashes, info, success and termination rows; none are truncated. Every saved last info proves the native sphere is on the bin, static, and not grasped. All 20 obstacle initial-state, controller, and policy-observation restore identities match their no-obstacle templates except for the named cylinder condition; all waypoints are on the Panda base's transport-line side. The policy input excludes the obstacle field. Both classes pass an actual batch of shape `(2,2,39)` observations and `(2,16,7)` actions through the official ConditionalUnet1D epsilon objective: direct loss `1.1481501`, obstacle loss `1.2529047`, both finite. This is interface validation only, not training.

Artifacts are `demos/PlaceSphere-obstacle/20261022_173110_pilot/trajectory.h5` (SHA-256 `0a415d203ee023368c357bcb34c9173128633230cca210ac9abf6f562aca8220`) and `outputs/exp/20261022_173110_placesphere_pilot/`. Video identities: `direct.mp4` `25765cd19beeaf2527ac5f3408585bfbb5d6d404f3bcedadefe18444defc1c44`; `obstacle.mp4` `4951918d972286ca4809310af58dc7108468263be11d1e31e681a46bd0746979`. Frozen-contract SHA-256 is `284a1e5e44e1b6f358d07dee04a47ebff9e8d25c7af7c7f5e0511dbfd6e54d53`; report SHA-256 is `e832f0bdb7d48becdf8493f218ef4b1fd10f8e96c410462c9c17a5442940412e`.

The pre-run generator source pins are recorded verbatim in the frozen contract: `layout_geometry.py` `3c5422ba3d29393241aa5494f8a24f7f45b6b03d7e99489ef5c61664e0c9f577`, `pickcube_expert.py` `1124e941a6735dd1c11b96ec3c94afd802167240821303607645dde7b2f7cbd2`, `placesphere_obstacle.py` `1ea3d488c0cea2c42d23417f88dcf236620d523b2894ec22deaa6dd68d8de12a`, and pre-run `placesphere_pilot.py` `98503e18cb47256bfcc2eabe87630795d2fd7fe59287f1429db321097bfc0460`. Native ManiSkill and official expert source pins were recorded at commit `da47dbb89a897f5b69d24427e698e162c4679142`; committed and worktree hashes match. After collection, the pilot module gained one input-validation guard; `outputs/exp/20261022_173110_placesphere_pilot/post_collection_source_delta.json` maps the executed and current hashes and records its red/green evidence. It rejects invalid cylinder radius/height/margin before creating an output directory; the frozen run used valid values and the saved trajectory identity is unchanged. The actual invocation is the command above with timestamp `20261022_173110` substituted.

Final regression command, run with the `maniskill` environment and explicit 600 s timeout: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 600s python -m pytest examples/baselines/constrain/tests -q` — **246 passed** in 58.00 s. The safety-geometry validator also had a focused test-first red (`DID NOT RAISE ValueError`) followed by green after the guard; the final full suite includes that test.
