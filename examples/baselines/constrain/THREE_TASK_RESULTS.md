# Three-task final obstacle results

These are the final valid paired runs retained after the 2026-10-23 cleanup.
Each task uses its own fixed 100 expert-verified obstacle scenes. Each policy
has a fixed denominator of 100 and zero runtime errors. Task success uses each
task's native criterion; safe success additionally requires no obstacle contact
and no object XY forbidden-region violation through the full success step.

## Results

| Task | Policy | Task success | Safe success | Obstacle contact | XY forbidden crossing | Step/time-limit failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| StackCube | Base DP | 12% | 0% | 100% | 59% | 88 |
| StackCube | ConstrainNet | 100% | 88% | 9% | 4% | 0 |
| PickCube | Base DP | 82% | 2% | 97% | 0% | 18 |
| PickCube | ConstrainNet | 83% | 38% | 53% | 0% | 17 |
| PlaceSphere | Base DP | 97% | 73% | 2% | 25% | 3 |
| PlaceSphere | ConstrainNet | 87% | 86% | 6% | 6% | 13 |

- **StackCube:** ConstrainNet improves safe success by **88 percentage points**
  and native stack success by 88 points. Evaluation uses `panda_wristcam`.
- **PickCube:** safe success improves by **36 points** and task success by one
  point, but contact remains high at 53%; this is the weakest task by the
  ConstrainNet safe-success metric.
- **PlaceSphere:** safe success improves by **13 points**, while task success
  falls by 10 points and contact rises from 2% to 6%. The control branch reduces
  XY violations but does not improve every metric.

The limit column groups actual task-budget exhaustion for readability:
StackCube uses 500-step timeouts, PickCube uses the audited 500-step cap
classification (historical Gym flags were not saved), and PlaceSphere records
actual 200-step time-limit truncations. It does not group runtime errors, which
are zero for all six policy/task results.

## Retained best valid experiments

| Task | Paired result root | Fixed scene root |
| --- | --- | --- |
| StackCube | `outputs/exp/20261021_wristcam_paired_eval/` | `outputs/exp/20261021_wristcam_scenes/` |
| PickCube | `outputs/exp/20261022_162002_pickcube_paired_eval/` | `outputs/exp/20261022_152157_pickcube_scenes/` |
| PlaceSphere | `outputs/exp/20261022_213500_placesphere_paired_eval/` | `outputs/exp/20261022_205500_placesphere_scenes/` |

For StackCube and PlaceSphere, the machine-readable metrics are in each root's
`paired_summary.json`. PickCube's final corrected metrics are in
`termination_correction.json` under `corrected_policy_summaries`; its original
paired summary remains unchanged. Both policies' representative videos live in
`{base_dp,constrain_dp}/representative.mp4` beneath each result root.

The six selected final weights, corresponding formal datasets and training
records are indexed in [OUTPUTS.md](OUTPUTS.md). Base DP final weights remain as
the benchmark comparator and as the matching control-branch initialization.
Intermediate checkpoints, failed runs, probes and pilots have been pruned.

The cleanup inventory is
`outputs/exp/20261023_105159_final_artifact_cleanup/cleanup_manifest.json`:
992 deleted files, 450,850,395 bytes removed, and 765 retained file identities
verified unchanged. Deleted paths/hashes remain in that inventory; original
manifests and result bytes were not rewritten.
The adjacent `final_verification.json` independently recomputes the table from
all 600 retained episode records and verifies the six representative videos.

## Interpretation

The three tasks have different native success conditions and are not pooled
into a single success rate. PickCube adds no release requirement; PlaceSphere
requires placement, static motion and an ungrasped sphere. The new tasks use
modified object/goal distributions and fixed base-side detours. Their scenes
are expert-solvable and not claimed isolated from training data. These results
do not establish general path selection, cross-distribution generalization or
a safety guarantee.
