# PlaceSphere fixed obstacle scenes — Ticket 11

`placesphere_scenes.py` generates 100 new, expert-accepted fixed obstacle
scenes from the frozen formal contract at
`demos/PlaceSphere-obstacle/20261022_184558_formal/frozen_contract.json`.
It does not sample policy performance, load a checkpoint, or use the 100 formal
obstacle initial states as its evaluation set. No training/evaluation isolation
or cross-set deduplication is asserted or required.

## Frozen task and generation recipe

The task is native Panda PlaceSphere with the official initialization
(`robot_init_qpos_noise=0.02`), state observations, CPU PhysX, and
`pd_ee_delta_pose` (7D). The native expert uses the installed MPLib-compatible
official Panda solver, official sphere OBB grasp/wrist orientation ordering,
lift, a same-base-side waypoint, the native bin-top pose, release/retreat, and
settling until native success. Native success requires the sphere on the bin,
static and ungrasped. The policy observation is 39D; the full saved raw
observation is 42D.

The one static vertical cylinder is reconstructed as actual collision geometry:
radius 0.03 m, height 0.08 m. Sphere forbidden XY radius is 0.055 m
(`0.03 + 0.02 + 0.005`). Start ranges are x `[-0.19,-0.08]`, y
`[-0.13,0.13]`; container ranges are x `[0.04,0.15]`, y `[-0.13,0.13]`.
Layout checks use the 0.14 m minimum transport, 0.06 m minimum base-to-line
distance, the full 0.025 m bin half extent, and same-base-side waypoint offset
`[0.14,0.20]`. The expert first executes the detour template without an
obstacle; its actual sphere and all-link collision-vertex sweeps determine
candidate cylinder positions in the direct-path strip. The exact template
initial state/controller are then restored in a fresh environment with only
the obstacle condition changed, and the expert executes it again.

Safety is sticky from first grasp through release until the native success
step. Every accepted scene requires native terminal `on_container`, `static`,
`ungrasped`, a reached release/retreat phase, post-release checks through native
success, no robot/sphere cylinder contact, and no sphere forbidden-disk crossing.
The full saved expert actions and raw/policy observation trajectories support
auditing. Each row records the post-release proof and actual measured cylinder
geometry.

## API and artifacts

Public API:

```python
from examples.baselines.constrain.placesphere_scenes import PlaceSphereSceneSet

catalog = PlaceSphereSceneSet("outputs/exp/YYYYMMDD_HHMMSS_placesphere_scenes")
scene = catalog.load(0)                  # verified state/controller/observations/actions
env, restored = catalog.restore(0)        # new native env with actual scene cylinder
env.close()
```

`load()` validates manifest identity, full HDF5/config/provenance/report/journal
hashes, frozen-contract identity, captured source bytes, per-scene native-state,
controller, raw-observation, policy-observation and expert-action hashes,
dimensions, cylinder flags/size, expert native-success evidence, and post-release
safety proof before returning a scene. `restore()` creates a fresh environment
with the recorded physical radius/height, resets task/controller/safety/elapsed
state, restores the complete native state and controller, and compares the
full state, controller, raw 42D observation and policy 39D observation against
the recorded values. Tolerances are explicit in `config.json` and
`manifest.json` (initial target 1e-7); every norm is recorded in the 100-scene
restore audit. The audit performs two separate fresh restores per scene and
reports both actual full-state/controller/raw/policy fingerprints alongside
recorded fingerprints, exact-hash match results, and all four maximum-norm
differences. It requires both actual restore fingerprints to reproduce exactly.
It does not normalize away controller or native-state mismatch.

The experiment directory contains `scenes.h5`, `manifest.json`, `config.json`,
`frozen_contract.json`, `provenance.json`, `source_snapshot/`, `attempts.jsonl`,
`report.json`, `checksums.json`, and (after complete audit) `restore_audit.json`
plus its SHA-256 sidecar. HDF5 stores each full initial state, controller,
initial raw/policy observation, expert action sequence and expert observation
sequences. Manifest rows carry start/container/base XY, base side, waypoint,
actual cylinder geometry, scene seed, native/safety proof, expert trace, and
identities. The journal records each actual runner invocation exactly once,
including failure reason, stage counters, per-phase time and diagnostics.

The generator captures frozen contract bytes and exact current local/native
source bytes before runtime preflight or attempts. Their SHA-256 fingerprints
and the frozen contract's historical source pins are both retained; any source
delta from the formal collection-time pins remains visible rather than being
rewritten. Output paths are exclusive and failed/incomplete runs retain their
journal/report and exit nonzero. Default limits are seed `40017`, target 100,
1,000 complete attempts and 3,600 s wall time.

## Commands and acceptance

Run from repository root after activating `maniskill`:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 3600s \
  python -m examples.baselines.constrain.placesphere_scenes \
  --output-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_scenes \
  --seed 40017 --target 100 --max-attempts 1000 --max-wall-seconds 3600
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 600s \
  python -c 'from examples.baselines.constrain.placesphere_scenes import audit; import sys; r=audit(sys.argv[1]); assert r["checked"] == 100 and r["all_passed"]' \
  outputs/exp/YYYYMMDD_HHMMSS_placesphere_scenes
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 600s \
  python -m pytest examples/baselines/constrain/tests -q
```

Acceptance requires exactly 100 unique scene IDs, complete=100, actual runner
invocations equal the journal/report denominator and no greater than 1,000,
elapsed time within 3,600 s, all per-stage and phase timing/failure statistics,
100 native expert successes with post-release safety proof, immutable artifact
hashes, and 100 successful fresh-environment restores within recorded
tolerances. This catalog is fixed expert-verified evaluation input; no formal
policy evaluation or policy video is part of Ticket 11.

## Completed generation — 2026-10-22

Output: `outputs/exp/20261022_205500_placesphere_scenes/`.

| Evidence | Result |
| --- | --- |
| Fixed scenes | 100/100, IDs `placesphere-000`–`placesphere-099` |
| Actual complete runner invocations | 166/1,000; acceptance rate 100/166 = 0.60240964 |
| Generation wall time | 620.515 s / 3,600 s |
| Failures retained | 63 `native_unsuccessful`, 1 `cylinder_geometry_retry_exhausted`, 2 `obstacle_native_or_safety_failure` |
| Fresh restore audit | 100/100 pass; paired independent restores have exactly identical actual state/controller/raw/policy fingerprints for all 100 |
| Maximum restore absolute differences | full native state 0; controller 0; raw observation 0; policy observation 0 (each declared limit 1e-7) |
| Actual vs recorded state/controller exact hashes | 100/100 state, 100/100 controller |
| Safety/expert acceptance | 100/100 native success, reached release/retreat, checked post-release through native terminal, no contact/crossing, terminal on-bin/static/ungrasped |

Aggregate stage counters: geometry 194, waypoint 181, template executions
166, cylinder candidates 54,075, obstacle executions 102, restores 102,
grasp phases 165, release phases 130, stable successes 101, stable failures
64. Summed phase elapsed seconds: geometry 0.014, template 57.409, cylinder
sampling 384.146, obstacle execution 58.298, complete attempts 529.440.
The report's actual runner count and attempts journal length both equal 166;
there is no loop-index offset.

Frozen formal contract SHA-256:
`ff11b940ede86e290928d26ccd4ac96e7ba480995af774d46a018dbda8b6e761`.
Current collection-time local source pins match `layout_geometry.py`,
`pickcube_expert.py`, and `placesphere_obstacle.py`; exact actual source bytes
are archived and fingerprinted. The frozen historical pins differ for
`placesphere_formal.py` (`15fe2080…` frozen, `66cd8150…` actual) and
`placesphere_pilot.py` (`83cf79a7…` frozen, `ed6bfb1c…` actual). These source
deltas remain explicit in `provenance.json`; the frozen contract was not
rewritten. Both native ManiSkill source files match their frozen pins exactly.

Artifact SHA-256: `scenes.h5`
`ed4e09429a29884cbcc2d77cd5cd1d8ec01c49d74a00a64f02dc7b6929937ece`;
`manifest.json`
`1606a307fb117f82ecf3e798d170b394a099b8c4c653eee96c9ee2d6b2649be3`;
`attempts.jsonl`
`42bd83fda6dc5c2ec925f952a909d4a13321ea2c9af52d49f032403ed53df350`;
`report.json`
`b6b139b990856abf325b6596bff3f4f708281a3655a086fc8e10448a009d0132`;
`restore_audit.json`
`7e2f29e2d33ee06a42045e154a5325b8ddadd6259e20e0adf5a247701ca2747b`.
The generation source snapshot hash for `placesphere_scenes.py` is
`904e767222fb8b8cce18ed5021bcf46cfdabec8d69d7ab8c2bd057ceba79e430`;
the separately fingerprinted two-restore audit implementation hash is
`4caa3fd5becc4e0e22538da8a285b8eff495b9f45bb38af3ae2dde0f249087d7`.

Generation command used the documented values and the path above. The full
suite command is
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 600s python -m pytest examples/baselines/constrain/tests -q`.
