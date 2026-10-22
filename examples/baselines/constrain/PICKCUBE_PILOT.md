# PickCube obstacle pilot — round 8 evidence

## Live diagnostic after round-7 pilot

The retained `demos/PickCube-obstacle/20261022_120016_pilot/` bytes were not
modified. Runtime state inspection confirmed state-mode observation size 45;
its final named state observation values are `pickcube_obstacle=[x,y,radius]`.
`get_state_dict()` returns `actors/{table-workspace,cube,goal_site}`, the
`articulations` tree, `pickcube_obstacle` (9 values), and `pickcube_goal`.
The restoration identity comparison now permits only obstacle center XY and
the obstacle-enabled bit to differ. Its canonical SHA-256 uses that same
normalization, and exact equality is required for all other leaves.

The first bounded live diagnostic is
`demos/PickCube-obstacle/20261022_153200_pilot_diag/` (seed 1422, one direct
target and three obstacle attempts): direct 1/1; obstacle 0/3 (one exact
robot-articulation restore mismatch, two cylinder-prefilter misses). The
restoration difference was at `/articulations/panda`; this is retained as a
rejected mismatch, not masked with a tolerance. A subsequent bounded diagnostic
`.../20261022_153700_pilot_diag/` (one direct target, six obstacle attempts)
accepted direct 1/1 and obstacle 0/6: two exact restoration mismatches, three
physical collision/crossing failures, and one native unsuccessful attempt.
It tested 3150 cylinder candidates. This exposed that using the entire robot
link sweep in the *object-path candidate prefilter* excluded the corridor;
candidate ranking now uses the measured cube-center sweep and the final native
physics execution remains authoritative for all robot links and the cube.
The rotated cube footprint now follows its runtime quaternion. These diagnostic
attempts are not a completed dataset.

Two subsequent bounded diagnostics (each one direct target plus six obstacle
attempts) tested the exact-restoration residual correction and full-sweep
candidate ranking. The correction now makes full native state exact, confirmed
by attempt accounting (`state_restore_failures: 0`). The observation guard then
identified an exact mismatch at flattened index 18 (template 0, restored 1)
for two attempts; the other four obstacle attempts reached physics and failed
the native robot-obstacle contact gate. The named observation assembly is
`qpos[9]`, `qvel[9]`, `is_grasped[1]`, `tcp_pose[7]`, then native task fields
and the final obstacle `[x,y,radius]`; index 18 is `is_grasped`, not an
obstacle-conditioning coordinate. This time-zero change remains a hard
rejection. The detailed values, candidate center,
and full-robot sweep clearance are in
`demos/PickCube-obstacle/20261022_160200_pilot_diag/report.json`. Measured
full-robot sweep clearance was only 9–18 mm, below the cylinder collision
radius; runtime contact checks confirm these placements are blocked.
An additional six-attempt check used a deterministic same-seed reset instead
of simulator-state replay; it reproduced the same two `is_grasped` mismatches
and four physics rejections, so hidden contact-state restoration is not yet
resolved by that reset seam either. Full state identity and controller identity
are still checked before acceptance.

The native `pd_ee_delta_pose` contract is preserved: Panda, qpos initialization
noise 0.02, state observations, 7 actions (XYZ position deltas scaled by 0.1 m,
XYZ Euler deltas divided by signed controller `rot_lower`, gripper index 6),
and native PickCube static-placement success. Obstacle conditions are raw world
XY/radius in meters; no StackCube observation indices are used. Collection is
finite: geometry tries are capped at 32, candidate grid at 525, per-class
attempts are capped by configuration, per-attempt control steps by 500, and
overall elapsed time by `max_wall_seconds` (default 1800; must be positive and
finite). Attempt records preserve failure diagnostics and phase counters.

## Current status

The pilot is **not closed**. Full state restoration now verifies bit-exactly,
but time-zero `is_grasped` changes and lack of safe obstacle successes remain
blocking. No new accepted pilot, representative video, or saved-batch
model evidence is claimed. No formal data collection or training was started.
The required full regression passed (200 tests; one upstream deprecation
warning). See `OUTPUTS.md` for all retained paths.
