# PlaceSphere paired obstacle evaluation — Ticket 12

The final paired evaluation uses the preselected Ticket 09 Base DP final EMA
and Ticket 10 ConstrainNet final EMA on the fixed 100-scene expert-verified
catalog from Ticket 11. It evaluates only obstacle scenes. Both policies receive
fresh native PlaceSphere environments restored from the same complete saved
simulation/controller state and 42-D raw observation; restore fingerprints are
checked against the catalog and then compared scene-by-scene between policies.

## Frozen contract and reproduction

The source run is
`outputs/exp/20261022_213500_placesphere_paired_eval/`. It freezes both
checkpoint hashes, the scene manifest/HDF5 hashes, runtime source hashes and
copies of the actual source bytes, adapter metadata, Python/Torch/device,
per-scene seed rule, 200-control-step cap, 120-second per-episode wall deadline,
and video indexes `(0,1,2)` before the first episode. The run is write-once;
there is no resume path. Every policy has exactly 100 index-addressed records;
errors remain in the requested 100 denominator.

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 6000s \
  python -m examples.baselines.constrain.evaluate_placesphere_paired \
  --scene-dir outputs/exp/20261022_205500_placesphere_scenes \
  --base-checkpoint outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt \
  --constrain-checkpoint outputs/ckpt/20261022_201100_placesphere_constrain/final.pt \
  --output-dir outputs/exp/YYYYMMDD_HHMMSS_placesphere_paired_eval --device cuda
```

The actual run took 1,048.83 seconds, within the 6,000-second process deadline.
Both models use state observations and native `pd_ee_delta_pose`: 42-D raw
observation, 39-D ordered named-state policy condition, 7 native action
channels (all XYZ/Euler pose values and gripper), 2/16/8 observation/prediction/
execution horizons, and 100-step squared-cosine epsilon DDPM. The Base condition
is `(batch,2,39)`; ConstrainNet's is flattened `(batch,78)`. Its separate latest
TCP-anchored world-parallel XY hint is normalized by `[0.5,0.5,0.08]`, clipped
to `[-1,1]`. Native action mapping clips six pose channels to `±0.35` and keeps
the gripper in `[-1,1]`, preserving release actions.

Each episode initializes Torch inference RNG from that scene's seed immediately
before inference, independently for each policy. Native task success is the
original PlaceSphere condition: on-bin, static, and ungrasped. The loop finishes
the entire first successful control step, accumulates sticky robot/sphere
cylinder-contact and forbidden-XY-crossing flags through that step (including
release, fall, roll and settling), then stops. A successful but unsafe episode
is a task success and not a safe success. Actual `terminated` and `truncated`
flags from `env.step` are retained. Truncation is `timeout`, native termination
is `environment_terminal`, and reaching the 200-step cap is separately
classified as `control_step_limit` if Gymnasium did not already truncate.
Episode wall-deadline failures are runtime errors, never successes.

Failure phases are diagnostic, not substitutes for task success. They retain
actual native grasp state, on-bin/static terminal state, ever-grasped/lifted/
transported/released evidence, and distinguish no grasp, grasp/no lift,
grasped/no transport, transport/no release, and release/stability failure.
Lift (`sphere Z > restored start Z + 0.04 m`) and transport (within 0.10 m of
static bin XY) are explicitly diagnostic thresholds only. A runtime error is
reported separately and is not relabeled as a task-phase failure.

## Measured outcomes

All 200 requested episodes were evaluated: 100/100 per policy, zero runtime
errors and zero episode-wall-deadline errors. Rates below use the requested
denominator of 100, not only completed successes.

| Outcome | Base DP | ConstrainNet |
| --- | ---: | ---: |
| Native task success | 97/100 (97%) | 87/100 (87%) |
| Safe success | 73/100 (73%) | 86/100 (86%) |
| Robot/sphere obstacle contact | 2/100 (2%) | 6/100 (6%) |
| Forbidden XY crossing | 25/100 (25%) | 6/100 (6%) |
| Time-limit truncation | 3 | 13 |
| Native environment terminal | 0 | 0 |
| Control-step limit without truncation | 0 | 0 |
| Runtime errors / wall deadline | 0 / 0 | 0 / 0 |
| Total control steps | 9,559 | 12,010 |
| Mean steps per evaluated episode | 95.59 | 120.10 |

Failure phase counts: Base DP — no grasp 0, grasp/no lift 1, grasped/no
transport 0, transport/no release 0, release/stability failure 2, task success
97. ConstrainNet — 0, 1, 0, 0, 12, and 87 respectively. These measured outcomes
are reported as observed, including ConstrainNet's lower task success and
longer rollout time.

Paired scene-level counts (both / Base-only / Constrain-only / neither) are
85/12/2/1 for native task success and 68/5/18/9 for safe success. All 100 pairs
match actual scene ID, seed, restored full native state, controller, raw
observation and policy observation fingerprints. Each restore also passed the
catalog's independent fresh-scene restoration tolerances before inference.

Videos were fixed before outcomes and cover scenes 0, 1 and 2 for both models.
The Base DP video has 367 frames and ConstrainNet has 406; these equal the
selected episodes' executed control steps plus three initial frames, so every
selected terminal control-step frame is included. Videos show the real native
scene and cylinder. Their byte hashes are in the table below and `paired_summary`.

## Artifacts and identities

The experiment folder retains `frozen_config.json`, `manifest.json`, paired and
per-policy summaries, 200 per-scene JSON records with native actions and
restore evidence, both videos, `source_snapshot/`, and `checksums.json` covering
the completed run. The source copy of this evaluator is SHA-256
`f66ee7dc626d578b106d845ddfa1ce3d0a21d1126575ad9dbbd78cafdc23941e`.

| Artifact | SHA-256 |
| --- | --- |
| Scene HDF5 | `ed4e09429a29884cbcc2d77cd5cd1d8ec01c49d74a00a64f02dc7b6929937ece` |
| Scene manifest | `1606a307fb117f82ecf3e798d170b394a099b8c4c653eee96c9ee2d6b2649be3` |
| Base DP final EMA | `07f93942cfcfd7d28120b0b210fd7c323098140f670bbfce4c9a83ff76b9fb30` |
| ConstrainNet final EMA | `0d20daee00ed90863ca4e68d1a7c1093774e4b7c04d20c4d470c3470dd715284` |
| Frozen evaluation config | `174625a2cd49363ef16059c4d9acf9d6e5ee41bb4236bcfd58e422d535633f98` |
| Paired summary | `bca66ea0efa1e5f8f3282510a56f8c912ab0dc3e5bb1beac8aa31f4f6ee4b5b0` |
| Final checksums index | `49462c5c8c0770e029731c8ea0d20fa89caf2f92394d4bc5bb752213d742d4da` |
| Final run manifest | `b3b16f77cb5475c5ee664d4d30bef874611dd39d1f4755d8fc0db9de82f9645e` |
| Base DP video | `4d2887209d25a5e2d50b084f26408fe5e20b1388f4bd073a6987f0a247cb220d` |
| ConstrainNet video | `6bad861544b2eef9b079e40974aa201c6bb0b0759893b70c2984e26d2e634a46` |

Three earlier evaluator attempts are retained as immutable failed evidence; none
contributes to these metrics. The first stopped in pre-episode source-path
validation; the next failed on NumPy-boolean JSON serialization; the third
recorded episodes but exposed the Base adapter's distinct condition shape in
the paired audit. The fourth directory is the completed corrected run described
above. Each restart used a new output directory; no result-driven scene, weight,
or configuration selection was performed. After the run, a reporting-only
checksum-index correction excluded `manifest.json` from its own recursive index
and updated the manifest to pin the final index. No episode, model, scene,
configuration or video bytes changed. The earlier index, exact captured
evaluator bytes and post-run delta are preserved as
`checksums_pre_finalization.json` and `post_run_reporting_source/`;
`post_run_integrity_audit.json` records the source and original-sidecar hashes.
It also resolves the remaining runtime helper/model source bytes to the pinned
scene, Base DP and ConstrainNet source archives (including `pickcube_expert.py`,
`base_dp.py`, official `train.py`, `constrain_ckpt.py`, and
`conditional_unet1d.py`).
The final index covers all final files other than itself and the manifest (which
pins the index hash).
