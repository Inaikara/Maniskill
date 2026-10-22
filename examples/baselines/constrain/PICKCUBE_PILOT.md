# PickCube obstacle pilot — round 9 evidence

## Round 10 provenance and base-side audit

The complete retrospective supplement is
`demos/PickCube-obstacle/20261022_130421_pilot/provenance_supplement.json`;
its canonical resolved recipe SHA-256 is recorded there. The full resolved
contract includes start/goal sampling ranges, shared layout filters, waypoint
sampling, cylinder candidate grid and both clearance filters, Panda expert
phases/action scaling/budgets, safety parameters, state semantics, and horizons.
Committed generator source pins are for `d9288268b634e24f91a6c00332dfa057280cd7ee`
and include per-file SHA-256 values; these match the current round-9 generator
files. This is a retrospective audit supplement, not collection-time metadata.
Existing files were hash-identified in place and were not rewritten.

`base_xy_side_audit.json` reconstructs the base XY from each native initial
Panda articulation state and cross-checks S/G, signed base side, line distance,
and obstacle waypoint side from the saved per-episode layout/trace. It reports
20/20 line-distance-filter passes for each class, direct min/max 0.06261/0.35389
m and obstacle min/max 0.08537/0.51536 m. Base-side signs vary in both classes
(direct 12/8, obstacle 16/4 for negative/positive), while only obstacle episodes
have waypoints (20/20), demonstrating the shared base-distance filter and
class-specific same-base-side detour. Recreate the supplement and audit with:

```bash
python -m examples.baselines.constrain.pickcube_provenance \
  demos/PickCube-obstacle/20261022_130421_pilot
```

The source pin is the committed generator; current future-collector output
generation is not asserted by this retrospective evidence.

## Diagnosis and fix

The retained round-8 runs showed exact native state/controller restoration but
`is_grasped` changed in the initial flattened observation after replay/reset in
the same CPU scene. ManiSkill's native `Scene.get_contacts()` delegates to
`PhysxCpuSystem.get_contacts()`, which reads that scene's current contact cache.
Round 9 restores the template state and controller in a newly constructed scene;
the initial observation is rebuilt from the native unflattened observation tree
and compared by field name, excluding only `extra/pickcube_obstacle`. The exact
state and controller checks remain in force. The successful pilot had zero state,
controller, or initial-observation restoration failures.

Instrumented actual failed obstacle executions reported first contact as
`panda_hand`, at control step 1 in `grasp_approach`. Their candidate robot-sweep
clearances were 4.7–6.1 mm, well below the 30 mm cylinder radius. Candidate
selection now requires both cube-path clearance and measured full-robot-sweep
clearance of at least the configured cylinder radius. No native contact gate,
cube XY forbidden region, or success condition was relaxed. The same-base-side
waypoint range was changed from 0.13–0.17 m to 0.20–0.25 m; this experiment
configuration is persisted in the contract and collection config. Radius and
height remain 0.03 m and 0.25 m.

Bounded calibration `20261022_130356_layout_clearance_diag` accepted one direct
and one safe obstacle trajectory in two obstacle attempts. It also retained a
rejected candidate with zero robot-clear candidates (maximum measured clearance
11.1 mm), confirming the new candidate gate rejects it before obstacle execution.
The preceding contact-phase diagnostic `20261022_130242_contact_phase_diag`
retained the link/step/phase evidence for the hand contacts. A second failure
mode in that diagnostic was native unsuccessful completion after the actual
collision, also rejected.

## Finite pilot result

`demos/PickCube-obstacle/20261022_130421_pilot/` contains the complete pilot:

- Direct: 20/20 accepted in 20 complete attempts.
- Obstacle: 20/20 accepted in 68 complete attempts (39 geometric cylinder
  clearance rejections, 9 executed collision/crossing failures).
- Zero restoration mismatches; all 20 obstacle successes passed native
  PickCube success, robot/cube contact checks, and cube forbidden-region checks.
- Attempt-level and aggregate elapsed time, geometry/template/cylinder/obstacle
  timings, and phase counters are retained in `report.json`.
- Actions, observations, success, termination/truncation, full state, native
  infos, seed, and traces are stored; the loader checked T actions and T+1
  observations/full states/terminal evidence.

The saved batch validation is
`outputs/exp/20261022_130421_pilot/validation/minibatch_validation.json`:
official ConditionalUnet1D epsilon-loss 1.1643102, finite, on a batch with
observation shape `[2,2,45]` and action shape `[2,16,7]`. Representative videos
are `outputs/exp/20261022_130421_pilot/videos/{direct,obstacle}.mp4`. Artifact
SHA-256 values are in the dataset's `artifact_hashes.json`; the trajectory HDF5
hash is `b04d68af0db7cd44f66055849724a81d626be7470394fdb84914ae9613c599d1`.
The validation and video hashes are recorded in
`outputs/exp/20261022_130421_pilot/diagnostics/pilot_evidence.json`.

The official task contract remains Panda, `pd_ee_delta_pose`, state observation,
7D actions preserving signed XYZ Euler rotation scaling, native PickCube static
placement success, 500 control steps, and single-cylinder obstacle geometry.
The exact layout and cylinder parameters are frozen in `contract.json`. No
formal 1,000/100 collection or training was started. ManiSkill source was read
for diagnosis and not modified.
