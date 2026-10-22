# PlaceSphere formal demonstrations — Ticket 08

## Frozen collection contract

The formal dataset is a fresh collection of **1,000 direct and 100 safe-obstacle
episodes**. Ticket 07's 20+20 pilot is provenance only and is not copied or
included. Both classes use the same sphere/container distributions and layout
prefilter; obstacle episodes add a same-Panda-base-side waypoint, measured
template sweep and static-cylinder placement, fresh-scene restoration, and
safe native replay. The generator accepts no downloaded or PickCube demonstration
input.

`placesphere_formal.py` is the reusable formal CLI; it calls the same native
PlaceSphere expert and collector as the proven pilot. It freezes the contract
before the first attempt, including the full Git commit, committed native source
hashes, current native worktree hashes, local generator/adapter SHA-256 pins,
MPLib compatibility/version, runtime dimensions and rotation bounds, geometry,
layout/waypoint/cylinder distributions, and all retry/attempt/wall limits.
The pre-attempt record at `frozen_contract.json` is authoritative. Direct and
obstacle targets are explicit independent values (1,000/100), each has a
2,000-complete-attempt cap, and the shared wall budget is 6,000 seconds. Each
geometry, waypoint, cylinder-grid, expert control, grasp, release/settle and
outer-attempt loop is bounded. `attempts.jsonl` is flushed after every complete
invocation; accepted HDF5 episodes and JSON metadata are streamed and flushed,
so interrupted/short collections remain diagnosable and are not overwritten.

The frozen policy input is the actual 39-D native named state flattening with
only `placesphere_obstacle` excluded. Each HDF5 episode stores both that policy
input and the full 42-D native `raw_obs` at T+1, including the 3-D obstacle
hint. `frozen_contract.json` gives every native observation field path, shape,
flattened width, and the explicit raw-to-policy field selection. HDF5 also
retains full environment/controller state at T+1, raw infos/terminal safety
evidence in the JSON manifest, and the original executed seven-dimensional
actions. No action conversion is applied. Accepted obstacle episodes require zero accumulated
robot/cylinder contacts, zero sticky sphere forbidden-crossing flags (including
post-release motion), a native successful placement, a static ungrasped sphere,
and a same-side waypoint.

## Exact commands

Run from the repository root with the `maniskill` conda environment activated.
The paths below are the Ticket 08 run identities:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 6100s \
  python -m examples.baselines.constrain.placesphere_formal \
  --dataset-dir demos/PlaceSphere-obstacle/20261022_184558_formal \
  --experiment-dir outputs/exp/20261022_184558_placesphere_formal \
  --seed 20261022 --max-attempts-per-class 2000 \
  --max-wall-seconds 6000 --max-steps 200
```

The collector's full native-state/controller restoration, all-episode contract,
training-window, and five-per-class saved-action physics replay check is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 2410s \
  python -m examples.baselines.constrain.placesphere_formal \
  --dataset-dir demos/PlaceSphere-obstacle/20261022_184558_formal \
  --experiment-dir outputs/exp/20261022_184558_placesphere_formal \
  --validate-only
```

Use `--validate-only` to repeat the independent audit without changing the
dataset. Collection and validation write their reports, complete logs,
artifact hashes, and verification records under the explicit experiment path.
The dataset contract/config, per-attempt journal, streamed episode manifest,
HDF5 trajectory, final report and identity hashes are kept under `demos/`.

## Formal results

The completed, qualifying collection at
`demos/PlaceSphere-obstacle/20261022_184558_formal/` contains exactly **1,000
direct and 100 safe-obstacle episodes**, with no pilot episodes. It used
1,438/2,000 direct complete attempts (1,000 accepted, 438 native unsuccessful)
and 163/2,000 obstacle attempts (100 accepted; 57 native-unsuccessful template
executions, 2 cylinder-geometry exhaustions, and 4 final obstacle rollouts
rejected). The 1,601 attempt journal rows equal the actual complete runner
invocations. Collection elapsed time was 1,349.90 s, within the 6,000 s wall
budget.

Counts and phase timing evidence in `report.json`:

| Measurement | Direct | Obstacle |
| --- | ---: | ---: |
| Accepted / target | 1,000 / 1,000 | 100 / 100 |
| Complete attempts / budget | 1,438 / 2,000 | 163 / 2,000 |
| Geometry candidates | 1,691 | 199 |
| Same-side waypoint samples | 0 | 174 |
| Template executions | 0 | 163 |
| Cylinder candidates | 0 | 55,650 (104 x 525; 2 layouts exhausted before a grid) |
| Fresh restore executions | 0 | 104 |
| Obstacle executions | 0 | 104 |
| Grasp / release phases | 1,438 / 1,151 | 161 / 132 |
| Stable successes / failures | 1,000 / 438 | 101 / 60 |
| Runner elapsed time | 746.43 s | 596.50 s |

Obstacle diagnostics split the four final-rollout rejections into one native
success rejected for a sticky sphere XY crossing and three native-unsuccessful
executions. All four had zero obstacle-contact flag; the **100 accepted**
obstacle trajectories have zero collision and crossing flags throughout, and
all terminate with native success, sphere on the container, static, and
ungrasped. The attempt journal and aggregate report retain each seed, per-phase
counter/timing, trace, and failure detail.

Independent saved-artifact validation passed all **1,100** episodes: action
shape `(T, 7)`, policy observation shape `(T+1, 39)`, full raw native observation
shape `(T+1, 42)` including the named `/extra/placesphere_obstacle` hint, finite
and legal values, and exact equality between the raw observation with only that
three-value field removed and the 39-D policy input. The frozen contract records
the task-specific native field paths and flatten order; no shared PickCube
indices are used. Full native state and controller structure at T+1, complete
T+1 info, terminal evidence, obstacle safety flags, workspace/transport/base-side
geometry, and training windows also passed. Exact class window counts are 93,848
direct and 10,635 obstacle. All
1,100 initial native states, controller snapshots and policy observations were
restored from HDF5 and checked; numeric comparison tolerances were 1e-7 for
native/controller state and 1e-6 for the flattened policy observation. The
measured maximum absolute differences were `1.4901161e-8` for native state,
`0` for controller snapshots, and `1.1920929e-7` for policy observations (one
float32 ULP). Five direct and five
obstacle episodes replayed their exact saved 7-D actions from restored initial
states and reached native/safe physics success. One additional candidate replay
(obstacle episode 1001) did not terminate successfully: its sphere linear speed
was 0.01072 m/s, just above the native 0.01 m/s static threshold. It is retained
as a failed replay diagnostic and was not counted among the five successful
representatives; no tolerance was applied to native success.

The qualifying trajectory HDF5 SHA-256 is
`fec4620c496ae3c5c0dec7db469333f302f7170363e11f66cba961cbb5d56ec4`; the
frozen-contract SHA-256 is
`ff11b940ede86e290928d26ccd4ac96e7ba480995af774d46a018dbda8b6e761`. The
trajectory JSON SHA-256 is
`740379c6a575a470f930b249494cb603a1ff36e2c71c5235e93d6eb3ec4c11ba`; report
SHA-256 is
`d861bf23e813e06e77f8791cd11e94844b000de8586f5169d33467f2c8df4b67`; and
attempt journal SHA-256 is
`a0caa8e41dac0e008f0e597b17140fe9aacf0b88cdf19a92fae953024d044747`.
The final validation record SHA-256 is
`e5545208f1bf132b604c815c5e4dbcf9e1ff3fb19c4afe5119c7c8f29e60df0f`.

The full dataset/report/attempt/episode SHA-256 mapping and final validation identity
are in `outputs/exp/20261022_184558_placesphere_formal/formal_identity.json`.
The executed full Git commit is
`e07d427cd48d092d8f6978f2839e494f39903577`; native PlaceSphere/expert source
pins are in the frozen contract and match the worktree. MPLib was `0.2.1`,
Gymnasium `0.29.1`, NumPy `1.26.4`, Python `3.11.16`, Torch `2.14.0+cu130`, and
runtime raw/policy observation/action dimensions were 42/39/7. The complete
execution/current local source map is in `formal_identity.json`. The pilot's
trajectory SHA remains
`0a415d203ee023368c357bcb34c9173128633230cca210ac9abf6f562aca8220`; it is
excluded from all formal counts and files.

## Retrospective source audit

Ticket review found that the frozen contract's pre-attempt pin for
`placesphere_formal.py` (`15fe2080…`) differed from the then-current file
(`66cd8150…`) and that the executed source bytes had not been retained. The exact
executed source has now been recovered from the workspace OpenCode snapshot Git
object `85e38256b2c857d568b8b5b04171a1f7ec744421`; the recovered file SHA-256 is
**`15fe2080f5a0c8a3d2ecbac7157190b3c799def9543f0c3a30c427881cca4135`**, an
exact match to the frozen contract. The immutable copy is
`outputs/exp/20261022_184558_placesphere_formal/provenance_source/placesphere_formal.py`.

The complete diff to current source is
`provenance_source/post_collection_source.diff` (SHA-256
`ad78f949d1e4b09e2a457df92da99273f868c9a822b1ac442601e73862f4f74c`). It adds
only measured restore-delta diagnostics to validation output; collector,
expert, action, and episode-generation behavior are unchanged. The other frozen
local pins match the worktree, except `placesphere_pilot.py` which is recovered
from committed source at `aa985ab:examples/baselines/constrain/placesphere_pilot.py`
with the exact frozen SHA-256. Its current diff is separately reviewable in
`provenance_source/current_pilot_snapshotting.diff` and adds only future
pre-attempt source archiving. Both native source pins match their committed and
current bytes. The complete mapping, object identities, recovered hash proof,
all original dataset hashes, and change classifications are in
`provenance_source/source_recovery_manifest.json`.

Future PlaceSphere collections now archive all local source bytes under
`<dataset>/source_snapshot/` before runtime preflight, contract creation, and
the first attempt. The manifest SHA is pinned in the frozen contract, and each
snapshot is included in the dataset artifact hash list. A regression test
asserts that snapshots and their hashes exist and match before an attempt runner
is invoked.

The first formal preflight at `20261022_180137` is preserved separately. It
returned one successful expert rollout but raised before writing that attempt
row because validation incorrectly applied the ±0.35 pose-channel bound to the
±1 gripper channel. Its report consequently says zero rows; the output
`failure_accounting.json` records the observed one invocation and the mismatch.
It accepted zero episodes and its empty HDF5 remains unchanged. The corrected
formal source explicitly bounds pose channels 0–5 and gripper channel 6
separately.

The intermediate 1,000+100 collection at `20261022_180516` is also retained but
is not the qualifying Ticket 08 dataset: it predated storage of the full raw
42-D native observation. Its HDF5 has a distinct identity and must not be used as
the formal deliverable. The later `20261022_184558_formal` dataset stores both
raw and policy observations and is the one validated above.

No formal training was run. The source pins and values in the saved frozen
contract, rather than current source values, define the executed collection.
