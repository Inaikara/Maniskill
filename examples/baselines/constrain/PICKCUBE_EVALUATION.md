# PickCube paired obstacle evaluation — Ticket 06

The formal paired comparison evaluates the two frozen EMA policies only on the
100 expert-accepted obstacle scenes in
`outputs/exp/20261022_152157_pickcube_scenes/`. It does not generate scenes,
collect demonstrations, evaluate obstacle-free behavior, or select scenes
after seeing policy outcomes. The catalog and full-initial-state restoration
contract are documented in `PICKCUBE_SCENES.md`.

## Frozen run

Run from the repository root with the `maniskill` environment active:

```bash
PYTHONPATH=. python -m examples.baselines.constrain.evaluate_pickcube_paired \
  --scene-dir outputs/exp/20261022_152157_pickcube_scenes \
  --base-checkpoint outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt \
  --constrain-checkpoint outputs/ckpt/20261022_144835_pickcube_constrain/final.pt \
  --device cuda \
  --output-dir outputs/exp/20261022_162002_pickcube_paired_eval
```

The scene manifest and HDF5, checkpoint byte hashes, parsed adapter metadata,
runtime versions, device, 500-step cap, 2-frame history, 16-step Base DP
prediction horizon, 8-step execution chunk, per-scene inference seed rule, and
video indexes `(0, 1, 2)` are frozen to `frozen_config.json` before rollout.
Every policy/scene gets a fresh native PickCube environment. The catalog
restoration method restores full native state and controller state; the runner
checks their SHA-256 fingerprints and the exact 45-D initial observation
against the catalog before action inference. Episode records retain native
actions, the initial restoration fingerprints, termination, sticky contact and
XY-crossing outcomes, grasp/lift/transport phase indicators, control steps, and
per-episode runtime errors. Per-record, per-policy-summary, and paired-summary
JSON writes are atomic.

Failure-stage diagnostics are separate from task success: `ever_grasped` is the
native `is_grasped` flag; `ever_lifted` means cube Z exceeded its restored
initial Z by 0.04 m; `ever_transported` means cube XY came within 0.10 m of the
static goal XY at any step. Non-success episodes are binned as no grasp, grasp
without lift, lift without transport, or transport/placement failure. These
thresholds are diagnostic bins only and never substitute for native success.

Native PickCube `native_success` is the task outcome. The runner finishes the
entire control step on which it first becomes true, then stops. Contact and XY
crossing are sticky; a same-step task success plus violation remains a task
success and is not a safe success. A task episode terminates only at native
success, environment termination/truncation, or 500 executed steps. Failures
remain records in the fixed denominator of 100. Paired counts report both,
base-only, constrain-only, and neither outcomes for task and safe success.

The 100 scenes share one generator and use a fixed base-side detour; this does
not establish left/right path-choice ability or train/test isolation. Video
episodes 0, 1, and 2 are fixed before policy evaluation, for both policies.

## Results

The formal run is saved under
`outputs/exp/20261022_162002_pickcube_paired_eval/`: frozen inputs,
`base_dp/episode_NNN.json`, `constrain_dp/episode_NNN.json`, per-policy
summaries and representative videos, and `paired_summary.json`. The summary
contains measured results and restoration-equality evidence; the results below
are copied from that completed run.

The completed run recorded 100/100 episodes for each policy, with zero runtime
errors. Base DP achieved 82/100 native task successes and 2/100 safe successes;
ConstrainNet achieved 83/100 task successes and 38/100 safe successes. Obstacle
contact counts were 97 and 53 respectively; XY forbidden crossings and
timeouts were zero for both. Environment-terminal counts were 18 (Base DP) and
17 (ConstrainNet); total executed control steps were 19,287 and 18,191. Paired
task outcomes were both 71, Base-only 11, Constrain-only 12, neither 6. Paired
safe outcomes were both 1, Base-only 1, Constrain-only 37, neither 61. All 100
scene IDs, restored full-state hashes, controller-state hashes and initial
observation hashes match across policy runs; all 200 initial observations match
the scene catalog exactly.

The representative videos contain 671 frames (Base DP) and 484 frames
(ConstrainNet), both covering the preselected scene indexes 0, 1, and 2.

SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Scene manifest embedded identity | `ce3bd0ccf30e240664799e85f55a0351f60b37d1ea6e2891b96f20f81266c6f4` |
| Scene manifest file bytes | `ea1ae25961ad4af673be865399d000717032452a5fb24fc7c77443f42558c346` |
| Scene HDF5 | `0a9740c5882335ed02eebd58cb4b27f61f0bd9ff8e75eac292825d1a1c7ca6e7` |
| Base DP final EMA | `b379573970712bcc17279ed237602a7be823acce9f7134ba125041ba5478389c` |
| ConstrainNet final EMA | `da94d795f9cc5622b9ad87c7c19178e3e05ec1e93837bfe673400a71208d0ef4` |
| Frozen evaluation config | `a8c020e11640d6a10169bfabf790e2d79f10eb33665805a6525d58ae4868a95b` |
| Paired summary | `32ba7c34535e772d95c6d2cafbe705709eb2953bc186d39838c13e5439c9c9bc` |
| Base DP video | `5fc0768927dd44ab0418ac187a8addb242ae5288b565c5f4f6b2f3085374d403` |
| ConstrainNet video | `c5d5d5944950bcaaee8d0bc601f028b9b7e249e568e69714857ac3540176de43` |
