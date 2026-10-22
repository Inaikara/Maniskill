# PickCube and PlaceSphere obstacle extensions — final results index

The two tasks are independent evaluations and are reported separately. There is
no combined success rate. Evaluation scenes are fixed, newly generated,
expert-solvable same-distribution obstacle layouts; all data uses modified
object/goal position distributions. The obstacle is a static cylinder and the
detour is constrained to the robot-base side. No left/right route-choice,
training/test isolation, cross-distribution generalization, or safety guarantee
is claimed.

## Formal expert data and model contracts

| Task | Direct demonstrations | Obstacle demonstrations | Direct acceptance | Obstacle acceptance | Policy observation / action |
| --- | ---: | ---: | ---: | ---: | --- |
| PickCube | 1,000 / 1,000 attempts | 100 / 402 attempts | 100% | 24.88% | native 42-D state / 7-D `pd_ee_delta_pose` |
| PlaceSphere | 1,000 / 1,438 attempts | 100 / 163 attempts | 69.54% | 61.35% | named native 39-D state from 42-D raw / 7-D `pd_ee_delta_pose` |

These counts exclude pilots and failed/intermediate datasets. PickCube keeps its
native pick-and-place task criterion and does not add release. PlaceSphere keeps
its native on-bin, static, ungrasped criterion and retains full release and
post-release stability behavior. Training/action contracts are task-specific;
each model preserves all seven native pose/gripper channels. PlaceSphere's
separate obstacle hint uses latest TCP world-parallel XY and the named cylinder
field with fixed `[0.5,0.5,0.08]` normalization.

Formal source datasets and identities:

| Task | Dataset | HDF5 SHA-256 | Manifest SHA-256 |
| --- | --- | --- | --- |
| PickCube | `demos/PickCube-obstacle/20261022_132753_formal/` | `fa846f52b5ae08de0243cfa52b40a4b0e7ce160881aa9db59ebaf4952c02904e` | `8ef0d2739d6bf5675316ab37f42f1437cc19d80739e6f9102973beeabd6d1e58` |
| PlaceSphere | `demos/PlaceSphere-obstacle/20261022_184558_formal/` | `fec4620c496ae3c5c0dec7db469333f302f7170363e11f66cba961cbb5d56ec4` | `740379c6a575a470f930b249494cb603a1ff36e2c71c5235e93d6eb3ec4c11ba` |

The same formal PlaceSphere dataset supplies only direct IDs 0–999 to its Base
DP and only safe obstacle IDs 1000–1099 to ConstrainNet. PickCube uses the
equivalent direct/obstacle split. Exact contracts and source pins are in
`PICKCUBE_DATA.md` and `PLACE_SPHERE_DATA.md`.

## Training budgets and selected checkpoints

Both tasks independently trained Base DP and ConstrainNet for 30,000 formal
updates each, seed 1, batch 1,024, AdamW `1e-4`, 500 warmup updates, EMA power
0.75, and a final EMA selected before training. Selection did not depend on
policy evaluation.

| Task/model | Completed formal updates | Selected final EMA | SHA-256 | Formal elapsed |
| --- | ---: | --- | --- | ---: |
| PickCube Base DP | 30,000 | `outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt` | `b379573970712bcc17279ed237602a7be823acce9f7134ba125041ba5478389c` | 896.09 s |
| PickCube ConstrainNet | 30,000 | `outputs/ckpt/20261022_144835_pickcube_constrain/final.pt` | `da94d795f9cc5622b9ad87c7c19178e3e05ec1e93837bfe673400a71208d0ef4` | 818.20 s process wall |
| PlaceSphere Base DP | 30,000 | `outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt` | `07f93942cfcfd7d28120b0b210fd7c323098140f670bbfce4c9a83ff76b9fb30` | 972.88 s |
| PlaceSphere ConstrainNet | 30,000 | `outputs/ckpt/20261022_201100_placesphere_constrain/final.pt` | `0d20daee00ed90863ca4e68d1a7c1093774e4b7c04d20c4d470c3470dd715284` | 944.36 s |

One additional PlaceSphere ConstrainNet attempt completed 30,000 updates but
failed its post-training verification and is excluded from checkpoint selection
and policy results. This is an additional 30,000 completed PlaceSphere updates
(90,000 total across PlaceSphere Base, selected ConstrainNet and the failed
ConstrainNet attempt), not zero-cost diagnostic work. Its elapsed training time
was not written to a successful summary because verification failed; the exact
30,000 updates remain visible in its loss log. Its log/config/checkpoint
remain under `outputs/exp/20261022_200800_placesphere_constrain/` and
`outputs/ckpt/20261022_200800_placesphere_constrain/`; the successful selected
model and verification are under the 20:11 run. A separate 20:07 attempt stopped
before training.

## Obstacle-only paired evaluation

| Task (frozen scenes) | Base DP: task / safe | ConstrainNet: task / safe | Other safety outcomes | Evaluation records |
| --- | ---: | ---: | --- | --- |
| PickCube (`outputs/exp/20261022_152157_pickcube_scenes/`) | 82/100; 2/100 | 83/100; 38/100 | Contacts 97/100 vs 53/100; XY crossings 0/100 both | 100 each, zero runtime errors |
| PlaceSphere (`outputs/exp/20261022_205500_placesphere_scenes/`) | 97/100; 73/100 | 87/100; 86/100 | Contacts 2/100 vs 6/100; XY crossings 25/100 vs 6/100 | 100 each, zero runtime errors |

PickCube's corrected step-cap audit reports 18 Base / 17 Constrain control-step
limits; its historical records omitted native flags, so this correction uses
the saved 500-step count and frozen limit and does not reconstruct flags. The
completed PlaceSphere run recorded 3 / 13 real time-limit truncations, no
environment terminals, and no 200-step cap without truncation. Full distinct
task/safe paired counts, phases, safety outcomes, actual restore hashes, fixed
videos and checksums are in `PICKCUBE_EVALUATION.md` and
`PLACE_SPHERE_EVALUATION.md`. The sample sizes are 100 scenes per task; they are
not combined into a pooled rate.

## Ticket and artifact evidence map

The local plan in `.scratch/multitask-obstacle/README.md` and tickets
`.scratch/multitask-obstacle/issues/01-*.md` through `12-*.md` remain planning
metadata, not implementation or evidence. Completed implementation/evidence
for the accepted sequence is indexed here:

| Ticket | Evidence |
| --- | --- |
| 01 | `PICKCUBE_PILOT.md`; `demos/PickCube-obstacle/20261022_130421_pilot/`; `outputs/exp/20261022_130421_pilot/` |
| 02 | `PICKCUBE_DATA.md`; `demos/PickCube-obstacle/20261022_132753_formal/`; `outputs/exp/20261022_132753_pickcube_formal/` |
| 03 | `PICKCUBE_TRAINING.md`; `outputs/exp/20261022_144000_pickcube_base_dp/`; `outputs/ckpt/20261022_144000_pickcube_base_dp/` |
| 04 | `PICKCUBE_TRAINING.md`; `outputs/exp/20261022_144835_pickcube_constrain/`; `outputs/ckpt/20261022_144835_pickcube_constrain/` |
| 05 | `PICKCUBE_SCENES.md`; `outputs/exp/20261022_152157_pickcube_scenes/` |
| 06 | `PICKCUBE_EVALUATION.md`; corrected results pinned in `outputs/exp/20261022_162002_pickcube_paired_eval/termination_correction.json` |
| 07 | `PLACE_SPHERE_PILOT.md`; `demos/PlaceSphere-obstacle/20261022_173110_pilot/`; `outputs/exp/20261022_173110_placesphere_pilot/` |
| 08 | `PLACE_SPHERE_DATA.md`; `demos/PlaceSphere-obstacle/20261022_184558_formal/`; `outputs/exp/20261022_184558_placesphere_formal/` |
| 09–10 | `PLACE_SPHERE_TRAINING.md`; `outputs/exp/20261022_193900_placesphere_base_dp/`, `outputs/exp/20261022_201100_placesphere_constrain/`; selected weights in matching `outputs/ckpt/` folders |
| 11 | `PLACE_SPHERE_SCENES.md`; `outputs/exp/20261022_205500_placesphere_scenes/`, including `restore_audit.json` |
| 12 | `PLACE_SPHERE_EVALUATION.md`; `outputs/exp/20261022_213500_placesphere_paired_eval/` |

## Interpretation and limitations

The positions/distribution were intentionally modified, both expert classes
share that distribution, and the obstacle training/evaluation catalogs are not
claimed isolated. The scenes are individually expert-solvable; paths use the
fixed base-side detour rather than testing a choice among route sides. The
observed task, safety, contact and crossing results describe these models on
these fixed obstacle scenes only. They imply neither a general path-selection
capability nor a safety guarantee.

## Goal completion audit

The 12-ticket acceptance and retained-artifact identity audit is recorded at
`outputs/exp/20261022_221641_multitask_obstacle_completion_audit/audit.json`
(SHA-256 `c10fd9a5e256c6b7ac5f08f2eb401c55ecda853415b8cb2d26b50c3eee0ea15a`).
It maps each issue acceptance bullet to its evidence, records checked dataset,
checkpoint, scene, paired-summary, correction and video identities, and lists
known caveats. The true goal-start base is
`fb21bd47f3a4d8c11ac71b6be4759aca5b5018cf`; round-25 base and audited HEAD are
`f09237a42d4e2fefbb661644918373991a2da640`. The complete goal-base-to-HEAD
diff contains 50 paths, all under `examples/baselines/constrain/`. Round 25 is
still uncommitted, so no final commit SHA exists; the goal branch is not merged.
