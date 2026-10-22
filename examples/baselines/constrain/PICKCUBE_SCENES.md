# PickCube fixed obstacle scenes — Ticket 05

The immutable 100-scene set is stored at
`outputs/exp/20261022_152157_pickcube_scenes/` (never under `demos/`). It was
generated from the formal Ticket 02 contract at
`demos/PickCube-obstacle/20261022_132753_formal/frozen_contract.json` using
generation seed `20017`. This is a newly generated set, not the 100 training
initial states reused wholesale. It makes no train/eval isolation, overlap, or
generalization claim.

## Generation and verification

Run from the repository root after activating `maniskill`:

```bash
python -m examples.baselines.constrain.pickcube_scenes \
  --output-dir outputs/exp/YYYYMMDD_HHMMSS_pickcube_scenes \
  --seed 20017 --target 100 --max-attempts 1000 --max-wall-seconds 3600
python -m examples.baselines.constrain.pickcube_scenes \
  --output-dir outputs/exp/YYYYMMDD_HHMMSS_pickcube_scenes --audit
```

The experiment snapshots configuration, formal contract and native runtime
controller resolution before the first physics attempt. It checks all pinned
local and ManiSkill controller source hashes, plus live `pd_ee_delta_pose`
position/rotation bounds against the previously resolved runtime. Each outer
attempt calls the existing `_default_attempt("obstacle")`: at most 32 geometry
layouts, 525 cylinder candidates, and 500 total expert control steps using one
same-base-side waypoint plan. The candidate cylinder is constructed in a fresh
scene with its actual 0.03 m collision radius and 0.25 m height. Acceptance
requires native PickCube success, no robot/cube obstacle contact, and no XY
forbidden crossing. No checkpoint or policy is loaded or consulted.

The historical `report.json` and manifest `generation` record the original
algorithm's 350 loop iterations (28.57%). The immutable `attempts.jsonl` has 349
actual runner invocations: 100 accepted and 249 rejected (198 exhausted
525-candidate geometry searches, 49 real obstacle collision or crossing
rejections, and 2 unsuccessful no-obstacle templates). The canonical corrected
acceptance rate is **100 / 349 (28.65%)**. The attempt-count accounting
correction is recorded separately in
`outputs/exp/20261022_152157_pickcube_scenes/accounting_correction.json`; it
pins the original report and journal hashes without rewriting historical
files. Generation elapsed time was 304.33 s. Attempts were bounded by 1000 / 3600 s.
Phase totals are in `report.json`; each actual outer attempt's seed, elapsed
time, failure, counters, and phase times are in `attempts.jsonl`.

Every scene stores the complete native state tree (including obstacle flags,
goal and robot articulation), complete flat `state` initial observation and its
shape/dtype/hash, runtime controller snapshot/hash, actual cylinder geometry,
seed and layout, and expert actions in `scenes.h5`. Manifest expert records
include native/safety acceptance, waypoint/candidate/clearance evidence and
initial runtime identity. All manifest, HDF5, configuration and source
identities are SHA-256 protected.

Independent fresh-environment restore audit: **100/100**. Restore constructs
the environment with the recorded physical radius/height, resets safety and
elapsed-step state, applies and exactly verifies all native state leaves using
the established Panda root residual correction, restores the controller using
its public state API, then requires bit-exact initial observation equality.
The actual frozen controller currently reports an empty state snapshot `{}`;
that snapshot is nevertheless captured, hashed, and restored for every scene.

## Scene API

```python
from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet
from examples.baselines.constrain.pickcube_expert import make_pickcube_env

scenes = PickCubeSceneSet("outputs/exp/20261022_152157_pickcube_scenes")
scene = scenes.load(0)  # immutable metadata, native state, observation, controller
env = make_pickcube_env(
    obstacle_radius=scene["metadata"]["cylinder"]["radius"],
    obstacle_height=scene["metadata"]["cylinder"]["height"],
)
env, initial_observation = scenes.restore(0, env)
# Or: env, initial_observation = scenes.restore(0)  # caller owns/closes fresh env
```

`load(index)` checks manifest/config/formal/native runtime/HDF5 and per-scene
state/observation/controller hashes, and rejects unsafe initial safety flags.
`restore(index, env=None)` rejects a supplied environment whose *actual
collision geometry* differs. It restores both into an existing environment and
returns `(env, observation)`; with `env=None` it constructs and returns a fresh
environment which the caller closes. It does not load models, run an expert, or
filter scenes based on policy behavior. The paired-scene consumer contract is
one immutable catalog plus an index shared by both policy runs: call
`catalog.restore(index)` once per policy in its own fresh environment, and
record the same `scene_id`, state hash, and index in both episode records. Do
not reuse a stepped environment or resample layouts between policies.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| `scenes.h5` | `0a9740c5882335ed02eebd58cb4b27f61f0bd9ff8e75eac292825d1a1c7ca6e7` |
| `manifest.json` | `ea1ae25961ad4af673be865399d000717032452a5fb24fc7c77443f42558c346` |
| `config.json` | `c76ecd552d04dc3887f3c8a79805f154121aa6830588c2d701296f36ecebe89f` |
| `report.json` | `da60c67224c158a4367e414eb403c2f007f8678a6dd65a27d01422ca0970e628` |
| `attempts.jsonl` | `e8806f42dd2213d9e65cc9673afa126e8f4d473f1eb94b721bbd0c21b0652e1e` |
| `restore_audit.json` | `61dce8f9d99eeedf56fa9d888f5e2c2bb677d0d2d8b021cf26d80eb55f5fefe9` |
| `accounting_correction.json` | `45271734d1f0c8ca1ec5bfae01be060bea77795f2c1080b859e507ec2e32e066` |
| frozen formal contract | `e1d98a8934203a2dcb397dc604736cb7877f9134cca4ef84ee502f8325f4f46e` |
| frozen native runtime resolution | `b965a888d9edc00ffc2f6f0dfb45452048ae21b5b269a2e9cab26685b5d5be3b` |

The set uses the formal same-layout/base-side rules. It is not isolated from
training data, is not claimed disjoint, and supports no cross-distribution
generalization claim.
