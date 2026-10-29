# Final retained artifacts — three obstacle tasks

## Round 44 — SE(3) horizon-8 clearance Control (2026-10-29)

Round 44 completed one fixed response fit, one **150,000-update Control-only**
training run from the immutable R40 Base, and one paired 30-scene trial (60 native
episodes). The final analysis is
`outputs/exp/20261029_160000_pickcube_horizon_clearance_r44/native_analysis_final_02.json`
(SHA-256 `dc86724e8871090101d569621628e316e85826589ea7806ea1f1d8dfdd50bffc`).
It is exploratory development evidence, not a formal performance or safety claim.

### Frozen sources, response, and precheck

The original pre-fit registration is `preregistration.json` (SHA-256
`211789597dd795db2f32e1eee4751b3a01a47a05e29770bef07139df2b9d244b`). The fixed
float64 ridge fit SHA-256 is
`f052a2b187e25fdf70ab98866049e196929562b5acce94b8c7a58ade2c45751c`. The fit
uses frozen obstacle IDs 1000–1499 only: 20,245 current-grasped carry windows
from the 400 `id % 5 != 0` episodes and 5,118 validation windows from the 100
`id % 5 == 0` episodes. It excludes 16,977 ungrasped and 3,500 incomplete-tail
windows; grasp approach is outside its coverage claim. Its 72 inputs are eight
planned native 6-D actions in physical units, current qpos9/qvel9, and the
previous-to-current TCP world pose delta. Its 48 outputs are the next eight TCP
world position deltas and world-rotation logs. No future observations, fitted
policy rollouts, new labels, or synthetic states are inputs.

The Panda source verifies root translation scale `+0.1 m` and root-aligned XYZ
rotation scale `rot_lower=-0.1 rad` with controller norm limiting. Since the
grasped carry labels contain no rotation-command variation, the frozen ridge
residual is combined with this exact known controller Euler-command rotation
map; the coefficients were not refit or tuned. This deterministic controller
map is part of the frozen response runtime and is distinct from an inference
shield. All current-response geometry uses the current finger q; actual future
finger q is used only when calculating the registered validation error.
The initial fit implementation had a positive rotation-action scale, but all
eligible carry-window rotation labels are identically zero (their fitted feature
std is the registered `1e-12` floor). The controller sign correction therefore
left the fixed coefficients bit-identical. The pre-fit registration/fit remain
byte-preserved; source-closure versions 2–8 revalidated the same fixed fit with
the signed controller map and completed Phase B/C source closure.

Source-closure versions and their fixed-model fresh prechecks are retained as
`preregistration_sourceclosed*.json` and `precheck/sourceclosed*/`. Final admitted
report SHA-256 is `011b0daf7fe1782319b713ae033d88a8e3aa64ddf8607bff5ce7c7865e14666f`
with closure SHA-256
`7c5a1f09088ac09b14d6816f2edf4ef271388a1255f206806df13c1e5bf48580`. It passes
the unchanged response criteria: native four-link point-transform max
`6.03e-8 m`; body-point p95 1.549 mm overall and 0.398–1.937 mm at horizons 1–8;
98.55% mean improvement over zero motion; TCP position p95 1.614 mm (maximum
6.046 mm). The eight forecast instants are discrete samples, not a continuous
sweep proof; the hold-carry precheck does not claim grasp-approach coverage.

### Registered Control training

Training registration SHA-256 is
`6d610e31e68fef78162c03290989f6c12b7f6e0c7baf2c6caa58835b4cc66683`. The typed
model kind is `SE3_horizon8_clearance`; R41 hint-9, 42-D position-unit global
condition, common-XY gauge seed 424243, native seven-channel actions and 100-step
DDPM sampler are retained. The objective is unweighted epsilon MSE plus
`0.01 * min(step/10000, 1)` times the sum of eight per-pose normalized body-SDF
and carried-cube XY hinge costs, averaged over current-grasped rows with eight
real future actions/states. Proxy rows are restricted to diffusion steps 10–30;
epsilon training still uses all sampled windows. The loss uses eight discrete
forecast positions, not continuous collision checking. No Base update occurred.

The ten real-batch GPU benchmark recorded 0.701 s for the cold first batch and
0.0553 s median / 0.0583 s p90 for the remaining nine. Retaining startup once
gives an 8,747 s projected p90 within the registered 15,000 s finite wall cap;
peak allocated CUDA memory was 579,302,400 bytes. Disabled-Control/Base epsilon
and DDPM sample max differences were both 0. The real minibatch gradient audit
reported XYZ norm 4.453, rotation norm 0.0294, and gripper norm 0; epsilon/body/
cube parameter-gradient norms were 0.395/3.548/119.297.

Exactly 150,000 updates completed in **11,824.64 s** on the RTX 5070. Batch 1,024,
seed 1, AdamW `1e-4`, warmup 500, cosine schedule, EMA `.75`, R40 final EMA
initialization, and R41 training profile were retained. All 150,000 loss rows
were finite. The final EMA checkpoint is
`outputs/ckpt/20261029_160000_pickcube_horizon_clearance_r44_control/final.pt`
(SHA-256 `fae59fa81362caa64b16ef0e882c63250a5f28a8894280dcd77610691c54a769`); 10k
recovery checkpoints include model, EMA, optimizer, scheduler and RNG state. The
final checkpoint was atomically saved before post-training audits. The frozen R40
Base state remained `5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`.
Post-training EMA gradient audit at diffusion step 20 on 1,024 real eligible
windows passed: XYZ norm 0.00420, rotation norm 0.00101, gripper norm 0; epsilon/
body/cube component parameter-gradient norms 0.0332/0.3603/0.0566. No optimizer
update occurred during that audit. Final EMA planned-action abs-value fractions
over 0.35 were 0.87% XYZ-X, 0% Y, 4.39% Z, and 0% for rotation; full per-channel
label/prediction range summaries are in the final analysis. The training-log
mean x0 abs-value fraction above 0.35 was 5.94% (max 7.95%).

### One paired 30-scene native trial

Trial registration SHA-256 is
`e064ca54643606be479709cd5bcc0afa0556281ecdc0d2ef94cea09a3ee0043d`. It pins the
final EMA, R40 Base, frozen R41 and R43 reference cohorts, existing R36 30-scene
set, and DDPM-100 / slice `[1:9]` / execute-eight protocol. One evaluator
registration-source correction was required before any episode row was written;
the first setup invocation failed immediately with
`ValueError: Round 37 registered source bytes changed` before creating any
episode rows. The original runtime/engine registration and its correction record
are retained as `native_engine_registration_before_source_correction.json` and
`native_engine_registration_source_correction.json`; zero episodes were
repeated. Exactly 30 Base and 30 R44 Control rows completed with zero runtime
errors in 416.62 s. R44 Base traces and outcomes match the R43/R40 Base exactly.
Each episode row retains actual flags, action traces, scene/seed, checkpoint and
profile hashes, first-contact and terminal-phase data, and native protocol;
contact entity-pair positions remain unknown because the recorder does not
expose them. Runtime clearance-cost calls were zero.

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R40 matched Base (R43-exact) | 27/30 | 0/30 | 30/30 | 21/30 |
| R43 swept-clearance Control (primary) | 30/30 | 14/30 | 16/30 | 5/30 |
| R41 hint-9 Control (secondary) | 29/30 | 17/30 | 13/30 | 6/30 |
| R44 SE(3) horizon-8 Control | 30/30 | 17/30 | 13/30 | 7/30 |

The preregistered gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`)
**failed**: task 30 passed; safe success 17, contact 13, and XY crossing 7 failed.
Against R43 Control, R44 transitions were task both 30; safe improved 7,
regressed 4, both 10, neither 9; contact cleared 7, introduced 4, both 9,
neither 10; XY cleared 2, introduced 4, both 3, neither 21. Against R41 Control,
safe improved 4 / regressed 4 / both 13 / neither 9; contact cleared 4 /
introduced 4 / both 9 / neither 13; XY cleared 1 / introduced 2 / both 5 /
neither 22. This is a combined training-objective/horizon/carry-mask/SE(3)
response proposal comparison, not an isolated lambda effect. No further trial or
checkpoint selection followed the failed gate.

The matched 1,000 direct / 500 obstacle demonstration source remained unchanged
(HDF5 `0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`); R40 Base
remained unchanged. Full retained rows are under `paired_native_r44/`; terminal
and first-contact counts are distinct in the report. Ticket 05 has no local
`.scratch/` issue file in this checkout.

Verification: full constrain suite **605 passed, 28 warnings** in 86.47 s;
frozen data and Base hashes match before/after. No new episodes, replay, demos,
layouts, or training cohorts were created. All code changes are confined to
`examples/baselines/constrain`; no agents were used and no commit/branch/push was
performed.

## Round 43 — swept clearance-proxy Control trial (2026-10-29)

The independently registered R43 frozen-data precheck is in
`outputs/exp/20261029_052324_pickcube_clearance_r43/`; model weights are in
`outputs/ckpt/20261029_052324_pickcube_clearance_r43_constrain/`. R43 retained
all three immutable precheck stages. The final source registration is
`preregistration_final_02.json` (identity
`db246e9d4ce56a53a1e2a4cc1aa8b91f70e4befbb800adb99b61934565d5d1ee`) and its
fresh final-source report is `precheck_final_02/geometry_response_precheck.json`
(SHA-256 `0feb6120d69b4a6c857e5b50a238fb6f32eded5acc4bb6ab526993bd8f556579`).
The first R43 preregistration/precheck and the R42 false-positive report plus
both R42 correction records remain byte-preserved.

The fixed-frame point validation now applies `T_tcp_link` in the forward
direction. Across 36 saved-state native FK queries, the maximum point error was
`6.03e-8 m` (hand `3.16e-8`, link7 `6.03e-8`, left finger `3.71e-8`, right
finger `2.19e-8`). Finger links were identified as prismatic joints and their
TCP-relative translation axes calibrated from two distinct existing qpos states;
rotation spread and translation residuals on the validation states were below
`1.4e-6 m/rad` (recorded individually in the report). The registered 5 mm
sample-sphere padding was applied separately to the analytic finite-cylinder
SDF; the unpadded physical-center SDF is separately reported. All four links
use deterministic samples capped at 256 per link. The carried-cube forbidden
radius uses the native `collision_radius + sqrt(2)*cube_half_size + safety_margin`
formula, and its XY-only current-state component was measured on 29,363 grasped
frames without a height waiver. Full mesh/URDF provenance, deterministic
sampling indices, dynamic finger q references/axes, finite-caps SDF quantiles,
and fit/validation cohort metrics are in the report.

The unchanged R42 21-feature ridge response artifact retained the same 96.60%
mean translation improvement and 0.682 mm TCP p95. On held-out trajectories,
translation-only predicted next-state body points (current TCP orientation and
finger q held fixed) had 0.568 mm mean / 1.542 mm p95 error, with 92.79% mean
improvement over zero motion. These are sampled geometry-proxy results, not
PhysX collision labels or a runtime safety guarantee. The frozen HDF5, manifest,
R40 Base, Panda source/assets and R42 response-fit records were read-only.

### One guarded Control training

The typed training admission is `train_preregistration.json` (identity
`49b8849d75ec97552ae3f6a03b0907de30c509796770e10c04796204bfe6f2ce`). It pins
the final precheck report, source/code/geometry/data hashes, R40 Base, normalizer,
R42 response fit, ten-real-batch benchmark, and the sole objective:
`epsilon MSE + 0.01 * min(step/10000, 1) * (swept_body + swept_cube_xy)`.
The proxy uses timesteps 10–30, scheduler-clipped `x0`, action index 1, four
sweep fractions `[.25,.5,.75,1]`, 5 mm point padding, 2 cm body margin and 1 cm
cube XY margin. It is training-only; actor inference has no clearance cost or
shield call. Before training, disabled Control matched the independent frozen
R40 Base exactly (epsilon and DDPM-100 sample max absolute difference both 0).
The ten-batch benchmark projected 9,909 s p90, within the registered 15,000 s
budget, with 550,867,968 bytes peak allocated.

Exactly 150,000 Control updates ran in 5,811.79 s on the RTX 5070. Batch size
1,024, seed 1, AdamW `1e-4`, warmup 500, cosine schedule, EMA power `.75`, R40
common-XY gauge seed 424243, R41 position-unit normalizer and raw hint-9 were
retained. No Base gradients/updates or new physical data were used. All logged
epsilon/clearance values and gradients were finite; proxy active rows and body/
cube component costs were recorded per step. The frozen R40 EMA state hash
remained `5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`.
The final EMA checkpoint SHA-256 is
`da818a9267a19a6ab8369a2d6b55b3e2116a28514c72662087607df635e53d48`; ten
thousand-step atomic recovery checkpoints and a step-0 strict metadata/EMA/
optimizer/scheduler export round-trip fixture are retained.

### One paired 30-scene native trial

Trial registration `native_trial_preregistration.json` (identity
`b6b5376d4a84d3b80685ca41c678ace52b530f767f4a16851fc890899eb7f09a`) pins the
final EMA, training/precheck registrations, unchanged R40 Base, R41 primary
tree, existing 30 R36 restored scenes and the exact DDPM-100/native-action
protocol. The pretrial real R41 row context check and disabled-Control/Base
equivalence passed. Exactly 30 rows per policy completed in 404.14 s with zero
runtime errors. Base action traces and outcomes match R41 **exactly** scene by
scene. All paired restored states and normalizer conditions matched; no rollout
was repeated. Raw rows are under `paired_native_r43/`; offline immutable-row
analysis and hash snapshot are `native_analysis_final.json` and
`native_analysis_inputs.json`. The native evaluator's summary-only report
assembly record is preserved separately; final analysis was produced from the
retained 60 rows, without rerunning native evaluation. The source-only report
repair chronology is `native_report_assembly_correction.json`.

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R40 matched Base (R41-exact) | 27/30 | 0/30 | 30/30 | 21/30 |
| R43 clearance Control | 30/30 | 14/30 | 16/30 | 5/30 |
| R41 Control primary | 29/30 | 17/30 | 13/30 | 6/30 |

R41 Control → R43 Control transitions (old/new-only, both, neither): task
`improved 1, regressed 0, both 29, neither 0`; safe `improved 4, regressed 7,
both 10, neither 9`; contact `cleared 4, introduced 7, both 9, neither 10`;
XY `cleared 4, introduced 3, both 2, neither 21`. Against the matched Base,
R43 cleared 14 contacts and 19 XY crossings, introduced 3 XY crossings, and
improved task success in 3 scenes.

The frozen gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`) **failed**:
task 30 passed; safe 14, contact 16 and XY crossing 5 failed. The sole trial is
complete exploratory evidence, not a formal performance claim or safety
guarantee. No further objective/training/native trial was run.

## Round 42 — preregistered training-only clearance proxy precheck (2026-10-29)

R42 **stopped before training and native development evaluation**. The one
precheck's final fail-closed analysis found that the fixed hand/link-7 frames
matched native FK, but the sampled point-cloud transform validation did not
meet tolerance, declared point padding was not applied, registered finger samples
were omitted, and the carried-cube forbidden-disk component was not implemented.
Therefore the surrogate was not credible as a complete
training objective, regardless of the response fit's strong held-out numeric
score. No λ tuning, geometry/dynamics refit, retry, checkpoint, policy action,
development label, simulator episode, or new demonstration was used.

### Frozen protocol and preserved attempt

The R42 output is
`outputs/exp/20261029_050700_pickcube_clearance_r42/`. Registration
`preregistration.json` was frozen before geometry/dynamics analysis (identity
`e1f843e385ed2975136ca15e1f0cedffeab24e7593c46add53f36129d9839f47`). The
final implementation source was pinned in the pre-analysis correction record
(`precheck_source_correction.json`, next registration identity
`6314040c6d53672d3788a155451c3ff7a3a49078f16b2c7d14f041875dcb89e3`); the
initial registration is retained unchanged. The single recorded precheck is
`precheck/geometry_response_precheck.json` (SHA-256
`804012ccf11b5e5e2ed63ff8fa70344150fa2d67d3a8c7befdb90cbdd5098eb2`). The
analysis-only invalidations are `precheck/precheck_correction.json` and
`precheck/precheck_correction_02.json`; the latter supersedes the first and
records the padding/finger audit. Both reference the first result by hash and
record no additional precheck attempt. Report-source deltas are independently recorded in
`analysis_source_correction*.json`.

Inputs were the exact R41 frozen 500 source: HDF5 SHA-256
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, and
validation `f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`.
The source has explicit per-trajectory cylinder metadata for all 500 obstacle
IDs (center, radius 0.03 m, and measured height; range 0.10698–0.16833 m).
The frozen initializer was the unchanged R40 Base final EMA
`9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719`; R41
Control was not resumed. No direct or dev states/actions were used to fit.

### Measured response and proxy evidence

The registered 21-feature, centered float64 ridge fit (`λ=1e-6`) used 36,697
frames from IDs `1000..1499 where id % 5 != 0`; the held-out-by-trajectory
validation used 9,143 frames from `id % 5 == 0`. Current-frame action index 1
alignment and target next-recorded TCP pose were used. Validation position error
was mean **0.0002623 m**, p95 **0.0006817 m**, maximum **0.0041635 m**; mean
position error improved **96.60%** over the preregistered zero-displacement
baseline. Angular p95 was **0.0009019 rad**. This meets the response-fit numeric
thresholds but is only one-step recorded-action prediction, not rollout or
collision-prediction accuracy.

The 36-state read-only FK check found fixed TCP→hand and TCP→link7 relation
maximum absolute errors `2.3842e-7` and `3.5763e-7`, respectively. However, the
recorded point-cloud/native transform comparison was **0.424431 m**, far above
the `1e-5 m` acceptance tolerance; its saved-TCP comparison field was conflated
with that point-cloud maximum, so the surface-point transform validation is
invalid. The original report also recorded 130,557 negative sampled point-SDF
values among 23,470,080 points; because the transforms failed and samples are
not exact surfaces, these cannot be interpreted as robot collisions or used to
assert the safe demonstrations are geometrically invalid. The proxy geometry
was limited to sampled URDF collision-mesh points from hand and link7. Although
the preregistration declared 5 mm point-sphere padding, the precheck did not
apply it; this is another recorded blocker. Registered left/right finger
collision samples were also absent. It was not a full Panda collision model and
never represented PhysX collision ground truth.

The registration additionally specified a carried-cube XY forbidden-disk
constraint, active only when the current raw grasp flag is true, using the
native PickCube radius formula. That component was **not implemented or
validated** in this one attempt. This is an independent hard failure. The
recorded attempted point cloud/hash/mesh sources and response coefficients are
retained under `precheck/`; they are not approved training artifacts.

### Decision and scope

The corrected result is `status: failed`, `training_permitted: false`, with
failure reasons: point-cloud transform validation above tolerance, missing
carried-cube proxy, declared sample padding not applied, and omitted finger
samples. As required, the 150,000-update R42 training, R41-equivalent
30-scene paired native trial, and gate were **not run**. The proposed
`epsilon MSE + λ_eff clearance` objective (λ final 0.01, warmup 10,000) was
never registered as a training phase or applied. Inference behavior is
unchanged. There are no R42 model weights or outcome counts. The preregistered
precheck attempt limit was one; no second model/geometry fit was attempted.
Frozen HDF5, manifest, validation, R40 Base, R41 Control, ManiSkill/Panda source,
PlaceSphere source, and existing demos remain unchanged.

## Round 41 — fixed Panda body-frame PickCube hint (2026-10-29)

R41 completed one 150,000-update Control-only training run on the frozen 500
obstacle demonstrations and one paired 30-scene development trial (60 native
episodes total). It reuses the exact R40 gauge-trained Base and its position-unit
profile. The sole treatment is the Control hint representation: append fixed
Panda hand and link-7 point coordinates to the unchanged three-value
obstacle-relative-to-TCP hint. This is exploratory development evidence, not a
formal performance or generalization claim. No demonstrations, derived HDF5,
episode labels, layouts, replay materializations, or ManiSkill/PlaceSphere source
changes were made.

### Artifacts, source chronology, and frozen inputs

The experiment is
`outputs/exp/20261029_031500_pickcube_bodyhint_r41/`; the Control checkpoint and
10,000-update recoveries are in
`outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_constrain/`. The reused Base
is `outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt`, SHA-256
`9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719`. R41
registration SHA-256 is
`45f6f9e0a11c4c414c339b942d31fe47694bb4c453fbfd3ce2c71fd3393b832e`; its
pre-training correction history and byte-preserved initial registration are
under the same experiment directory. A post-training strict-loader correction
is separately recorded there; it changed no weights and occurred before the
native trial.

The unchanged physical source is
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5 SHA-256
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, validation
`f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`. Training
still selects direct IDs 0–999 (83,004 windows) for the retained Base identity
and obstacle IDs 1000–1499 (45,840 windows) for Control. No Base retraining was
performed. All three physical-data hashes matched after training and native
evaluation.

### Fixed-frame calibration and hint contract

Calibration used only existing HDF5 T+1 raw 45-D observations and their saved
qpos, with read-only native forward kinematics. Nine existing trajectories were
sampled at four frames each (36 states); there were no `env.step` calls, new
scene states, data-fitted transforms, or runtime FK/environment queries. The
URDF is `mani_skill/assets/robots/panda/panda_v2.urdf`, SHA-256
`524c6986dbbabe244e05801f262862205e880834d01459b06ee273016d0579ba`; robot
model source `mani_skill/agents/robots/panda/panda.py`, SHA-256
`bf85577be75b7513802e676decb5543e3e65f2a3dfbc26175dd8b77ae57691f3`.
`panda_hand_tcp` quaternion order in the raw observation is `wxyz`.

| Fixed transform | Matrix (meters, row-major) | Native-FK validation |
| --- | --- | --- |
| `T_tcp_hand` | `[[1,0,0,0],[0,1,0,0],[0,0,1,-0.1034],[0,0,0,1]]` | relation spread `2.38e-7`; constant/native max `2.29e-8` |
| `T_tcp_link7` | `[[.70710678,-.70710678,0,0],[.70710678,.70710678,0,0],[0,0,1,-.2104],[0,0,0,1]]` | relation spread `3.58e-7`; constant/native max `1.19e-7` |

The maximum saved TCP pose versus restored-qpos FK difference was `1.83e-7`
(tolerance `1e-5`). Full details are in `frame_calibration.json` and in the
registered `body_hint` metadata.

Hint-9 order is:

1. Original `[obstacle_x-tcp_x, obstacle_y-tcp_y, radius]` (unchanged raw
   meter-valued calculation, scales `[0.5, 0.5, 0.08]`, clip `[-1,1]`).
2. `[obstacle_x-hand_x, obstacle_y-hand_y, hand_world_z]`.
3. `[obstacle_x-link7_x, obstacle_y-link7_y, link7_world_z]`.

The added six values are world-axis meters divided by `0.5` and clipped to
`[-1,1]`. Z is a derived world-height feature; it is **not** obstacle surface
clearance or a crossing oracle. Every feature uses only the latest raw 45-D
observation and fixed profile constants at runtime. No additional observation,
privileged pose/state query, cylinder-height feature, action filter, cost,
shield, or grip heuristic is used. The raw observation stays 45-D and the global
state encoder remains 42-D / 84-D history.

The R40 training-only common XY-origin augmentation and dedicated seed `424243`
are unchanged. It applies to the global condition; R41's relative/body hint is
computed from the original raw observation. A scan of all 45,840 obstacle
windows found finite, nondegenerate hint values; common-XY gauge invariance max
error was `5.96e-8`. All profile/frame metadata is standalone in the checkpoint,
and the typed format `pickcube_body_hint9_constrain_v1` strictly rejects legacy
three-channel checkpoints. The original default 3-channel PickCube condition
and PlaceSphere profiles remain unchanged.

### Control-only training

R41 trained exactly once for 150,000 updates from the frozen R40 Base final EMA;
the Base was bit-identical before and after training (embedded state hash
`5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`). The sole
model dimension change is `cond_dim: 3 -> 9` in the Control hint encoder; global
condition `84`, actions `7`, and copied down/mid channels `[64,128,256]` remain
fixed. Training remains native identity `pd_ee_delta_pose`, unweighted epsilon
DDPM-100 loss, batch 1,024, seed 1, AdamW `1e-4`, warmup 500, cosine schedule,
EMA power 0.75, and final EMA selected before training. There is no new action
clamp, codec, decoder, or runtime augmentation.

| Evidence | Result |
| --- | --- |
| Updates / wall budget | 150,000 / 9,000 s; completed in 4,413.55 s |
| Loss | first `0.03867678`, final `0.00173361`; all finite |
| Augmentation draws | 153,600,000 window draws (in-memory calls, not episodes) |
| Frozen Base | exact R40 EMA; unchanged |
| Control learning | copied encoder and zero-residual weights moved; EMA strict load passed |
| Final EMA | `outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_constrain/final.pt`, SHA-256 `838745ceb9437fa9db9c5c289a88a032cdf8a12dca42145e896d3c7ac3529642` |
| GPU | RTX 5070; peak allocated `520,562,688` bytes |

Atomic complete-state recovery checkpoints exist every 10,000 updates. Logs,
run config, training summary, augmentation statistics, phase argv/budget/exit,
and checkpoint records are under `train_constrain/` and the phase runtime JSON.

### One paired native trial and result

The one actual native trial used the same 30 restored R36 development scenes,
identical per-policy scene seeds, 500-control-step cap, 300-second per-episode
timeout, and a 36,000-second outer budget. Sampling was DDPM-100, epsilon,
`squaredcos_cap_v2`, clipping enabled, predicted slice `[1:9]`, execute/replan 8,
native seven-channel actions, with inference augmentation off. Base and Control
conditions pair across all scenes; the R41 Base's native action traces and
outcomes match R40 **exactly**. All 30 rows per policy completed with zero
runtime errors.

| Policy | Task success | Safe success | Obstacle contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R40 matched Base (unchanged checkpoint) | 27/30 | 0/30 | 30/30 | 21/30 |
| R40 Control primary | 29/30 | 15/30 | 15/30 | 5/30 |
| R41 Control hint-9 | 29/30 | 17/30 | 13/30 | 6/30 |

R40 Control → R41 Control transitions (both / neither; improved-or-cleared old
only / regressed-or-introduced new only) were: task `28 / 0; improved 1,
regressed 1`; safe `12 / 10; improved 5, regressed 3`; contact `10 / 12;
cleared 5, introduced 3`; XY crossing `4 / 23; cleared 1, introduced 2`.
All transition marginals reconcile to the 30-row cohorts. R41 Control first
contacts were transport 7, goal stabilization 4, grasp approach 1, lift 1;
contact links were Panda hand 10, cube 1, right finger 2. Contact entity-pair
positions are unknown because the recorder does not expose them. Terminal
phases were 29 task successes and one lift-no-transport. First-success steps
min/median/p90/max were `58/87/134/324`.

The frozen gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`) **failed**:
task `29` passed, safe `17`, contact `13`, and XY crossing `6` failed. This
single hint representation trial modestly improved safe success/contact against
R40 while leaving task success unchanged and increasing XY crossings by one.
It does not meet the gate and does not warrant a hint-grid sweep. The next
engineering slice should address the retained transport/goal-stabilization
contact mechanism using complete contact-phase evidence.

The first evaluator invocation failed in adapter construction before any
environment rollout (zero rows); its preregistration/runtime record is preserved.
The second invocation completed exactly one 60-episode trial in 413.34 s and
saved all rows. A historical report assembly error (`summarize_pilot` referenced
on the training module) happened only after the rows were complete. The report
was rebuilt offline from those immutable rows; `analysis_report_inputs.json`
pins row hashes and the R40 comparison tree, and `paired/status.json` records
`native_evaluation_rerun: false`. The analysis input snapshot SHA-256 is
`4c25c6a4de2a7886f48908228e75bee8c357d046f7ce34703a60367ca02a8ba3`; final
`paired/analysis.json` file SHA-256 is
`65c263a4387b1383e35c317c452505118ebafef4846953f699b977eebf5370c2`.
Complete episode action/flag traces, raw initial hint-9 values/hashes, checkpoint
and source identity, per-episode wall times, phase/contact links, and final
cube/goal XYZ are retained under `paired/base_dp/` and `paired/constrain_dp/`.

Verification: full constrain suite **586 passed, 28 warnings**; focused R41
feature/profile tests **6 passed**. The calibration, frozen-window preflight,
real saved-R40-row aggregation fixture, strict checkpoint load, and disabled
Control action equivalence all passed. Frozen source hashes remained unchanged;
ManiSkill/Panda and PlaceSphere files were read-only.

## Round 40 — matched PickCube global-XY gauge augmentation (2026-10-29)

R40 completed one matched Base/Control retraining and one paired 30-scene native
development trial (60 episodes total). It is exploratory development evidence,
not a formal performance or generalization claim. The single treatment is a
**training-distribution change**: each training window sees one uniformly
sampled XY coordinate-origin translation of the same physical episode. It is
not a claim that the training distribution or training examples were unchanged.
There are still exactly 1,000 direct and 500 obstacle physical demonstrations;
no demonstrations, HDF5 variants, labels, layouts, task geometries, or replay
materializations were created. The R40 augmentation draw counts below describe
in-memory training calls, not additional data episodes.

### Frozen source and preregistration chronology

The R40 experiment lives at
`outputs/exp/20261029_004500_pickcube_xy_gauge_r40/`; checkpoints are in
`outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_{base,constrain}/`.
The initial registration was frozen before Base training, with registration
identity `b7ea9f8d80049f8fbca71ac920d8c0c34548f088fc50f91e0b22073d9c5befb6`
and source aggregate
`82b54a6650fff1f2540c79bb777b2b02ab7aa2a76a1eb24521238ec3e9ef83af`.
Base's phase preregistration identity is
`49e14ee3f05be3a5280fd3636f75ea39198f419d3ef64bbb0666bd0f440ff53c`.

The Control preflight found a missing import for the explicit original-raw-hint
invariance check before Control training began. That precise source correction
is recorded in `post_base_source_correction.json`; the original Base registration
and source identity are byte-preserved at
`preregistration_base_training_frozen.json`. The R40 Base checkpoint was not
changed or retrained. The new Control registration identity is
`eafac1a12fa6fcae4641787d65bc7355bded3fc65857fb3353a48cfb493e1c35`, and its
phase preregistration identity is
`b37fb7743c1340f2e9258a2130e10f57fa90fb5900d0a32854d1d35a6a01e02d`.

After both training phases, a report-adapter source correction was recorded
before any native rollout in `post_training_analysis_correction.json`; the
Control-training registration is preserved at
`preregistration_control_training_frozen.json`. Neither model was retrained.
The native evaluation preregistration identity is
`4ef5212bb7d235a37fdeabd18fdfa91c1fe04471b0e0098254ac0fb88acd468e`, using
source aggregate
`0fdd4b7b6dc0927a59e4a82bfe490b58878e43e69c10c28575106f74cc2f79cb`.
The actual frozen native registration is
`3168e00fbdfde80bc96471203ab68ebc66a8833774e7bd1f118c65b7cfb64f0e`.

The immutable training source is
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5
SHA-256 `0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`,
manifest `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`,
validation `f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`.
Training uses direct IDs 0–999 (83,004 windows) and obstacle IDs 1000–1499
(45,840 windows); the same 333-task/500-obstacle cohort is unchanged. All three
hashes matched after both trainings and native evaluation. PickCube source stayed
at SHA-256 `53bfed0229cd4de2c7e363e7f3c531d5c5c84803c28e1345a2aea1a4972f3168`;
ManiSkill task/controller code and all PlaceSphere files were untouched.

### Fixed coordinate-origin transform and invariants

The fixed profile is `global_xy_gauge_v1`, with one always-on training draw
`delta_xy ~ Uniform([-0.02,0.02]^2)` meters per batch window. In raw observations,
the same delta is added to TCP world XY, goal world XY, and object world XY; for
raw 45-D Control observations, it is also added to obstacle-center XY. Indices
are derived from `NAMED_FIELDS`, not unsupported offsets. The delta is shared
across both history frames and all world-position fields in that window, and is
independent across windows. It is applied to a cloned minibatch before the fixed
R38 position-unit encoder.

This represents translating the coordinate origin of the entire scene,
including its implicit robot-base/table frame; it does not move an object or
cylinder relative to the robot. Z, joint positions/velocities, grasp flags,
quaternions, all relative vectors, cylinder radius, and all seven native action
labels remain exactly unchanged. Therefore TCP-to-object and object-to-goal
vectors, object-to-cylinder XY displacement, and the cylinder disk-clearance
margin are invariant. No axes rotation, Z shift, per-frame/independent
target/obstacle shift, label perturbation, new geometry, input feature, or
inference augmentation is used. Model inputs retain the 42-D R38 position-unit
condition and Control's same three-value relative hint; the latter is computed
from the original unshifted raw 45-D frame for bit consistency. Real-data
preflight measured zero hint change and zero disabled-Control epsilon difference.

The hypothesis is exploratory: a small common-origin randomization may reduce
absolute global-position memorization and improve conditional generalization
across the 500 demonstrations and 333 task variants. R38/R39 do not causally
establish that mechanism. The model has no explicit base-pose observation;
unchanged joint coordinates plus shifted TCP coordinates make this a deliberately
scoped coordinate-gauge hypothesis, not a sensor-noise or physical robot-motion
augmentation.

The dedicated RNG is `torch.Generator(device="cpu")`, seed
`training_seed + 424242` (424243 for this training seed). Once per batch it draws
`torch.rand((B,2), dtype=torch.float32, generator=cpu_generator)`, consumed in
row-major order `[window0-x, window0-y, window1-x, window1-y, ...]`, then maps
each value by `(2u-1)*0.02`. Its state is saved/restored with every recovery
checkpoint, independently of model initialization, diffusion noise/timestep,
sampler, and DataLoader RNG. The exact algorithm is repeated in both checkpoint
metadata blocks. Inference is unshifted in the original world frame.

### Matched training and checkpoints

The unchanged pipeline is Base direct-only for 30,000 updates, then Control
obstacle-only for 150,000 updates initialized from this new Base final EMA, with
all Base parameters frozen. Both keep batch 1,024, seed 1, AdamW `1e-4`, betas
`(0.95,0.999)`, weight decay `1e-6`, warmup 500, cosine LR, EMA power 0.75,
unweighted native epsilon MSE, DDPM-100 squared-cosine epsilon prediction, and
the R38 model/profile/action dimensions. Actions remain native identity
`pd_ee_delta_pose`; no new clamp, codec, shield, or phase weighting was added.

| Policy | Updates | Wall time | First → final loss | Final EMA SHA-256 |
| --- | ---: | ---: | ---: | --- |
| R40 Base | 30,000 | 1,061.30 s | 1.17822170 → 0.00282031 | `9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719` |
| R40 Control | 150,000 | 4,363.34 s | 0.03726619 → 0.00181645 | `2215f9f7d8bbec933bec53b2880758206af304a171297c5369ad85dcadfe1374` |

Both phase summaries report finite losses and complete 10,000-update recovery
checkpoints (3 Base, 15 Control). Loss logs have exactly 30,000 and 150,000
rows. Both checkpoint metadata objects carry exactly the same registered
augmentation profile and dedicated RNG state. Control's frozen-Base tensor hash
matches the R40 Base EMA (`5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d`).
The selected weights are the fixed final EMA, not a searched seed/step/δ setting.

Saved aggregate δ statistics (one entry per window draw, **not data quantity**):

| Policy | Draws | X min / mean / max (m) | Y min / mean / max (m) |
| --- | ---: | --- | --- |
| Base | 30,720,000 | -0.02000000 / -7.38e-8 / 0.01999999 | -0.02000000 / 3.30e-7 / 0.02000000 |
| Control | 153,600,000 | -0.02000000 / -6.80e-8 / 0.02000000 | -0.02000000 / 1.48e-6 / 0.02000000 |

Recovery state and summaries are under the R40 `outputs/ckpt/` and
`outputs/exp/` paths above. Training budgets were 3,000 s Base and 9,000 s
Control; both finished within budget. The first Base invocation exited before
any updates due only to a wrapper-created nonempty output directory (exit 1,
zero updates); its exact invocation record is preserved as
`phase_runtime_base_training_attempt01.json`. The corrected 30k launch used the
same preregistered settings and completed once.

### One paired native trial and primary comparison

Both EMA loaders strictly checked the same R38 typed position normalizer plus
registered augmentation metadata before evaluation. An independently seeded
Base-vs-disabled-Control native action preflight matched exactly (max absolute
difference 0.0). Runtime augmentation was off. The archived DDPM sampler stayed
at 100 steps, epsilon, squared-cosine, clipping enabled, predicted slice
`[1:9]`, execute/replan 8, seven native action channels, with no extra action
clamp or grip shield. The trial used the same 30 R36 development scene states,
controller/raw-observation hashes, position-encoded condition hashes and
per-scene seeds as R38, 500 control steps, 300 s per-episode limit and 36,000 s
overall limit. Exactly 30/30 Base and 30/30 Control rows are retained; all were
evaluated with zero runtime errors. Episode wall times were Base
4.163/6.995/15.322/17.702 s and Control 4.445/5.599/9.702/24.976 s
(min/median/p90/max).

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| Primary R38 Base | 22/30 | 0/30 | 30/30 | 23/30 |
| R40 Base | 27/30 | 0/30 | 30/30 | 21/30 |
| Primary R38 Control | 29/30 | 13/30 | 16/30 | 6/30 |
| R40 gauge-augmented Control | 29/30 | 15/30 | 15/30 | 5/30 |
| R39 guidance, scale 1.5 (negative result, excluded) | 29/30 | 11/30 | 17/30 | 6/30 |

R38 → R40 Control paired transitions (both / neither; positive outcomes
improved-new-only / regressed-old-only; violations cleared-old-only /
introduced-new-only): task `28 / 0; 1 / 1`; safe `9 / 11; 6 / 4`; contact
`10 / 9; 6 / 5`; XY crossing `4 / 23; 2 / 1`. Each integer table reconciles
to both 30-row marginals. The Base transitions were task `20 both, 7 improved,
2 regressed, 1 neither`; safe `0 both, 30 neither`; contact `30 both`; XY
`20 both, 3 cleared, 1 introduced, 6 neither`. The per-scene R38 tree was
preregistered and hashed before native execution; its aggregate tree hash is
`208f69a4dc85b4f1b7d96ee7fc9d74ad606d1e82c60c1ee84463f0adfbe2ea4f`.

R40 Control first-contact phases were transport 11, goal stabilization 2,
grasp approach 1, lift 1; link counts were panda hand 10, cube 1, link7 1,
right finger 3. Contact entity-pair positions are unknown because the native
recorder does not expose them. The Control terminal phases were 29 task success
and 1 lift-no-transport; first-success steps were min/median/p90/max
74/98/164/237. Final 3-D goal-distance min/median/p90/max was
0.00465/0.00824/0.01553/0.56811 m. All per-episode wall time, final cube/goal
XYZ, terminal flags, contact/stage traces, and raw rows remain in
`paired/constrain_dp/` and `paired/base_dp/`.

The preregistered gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`)
**failed**: task passed at 29, safe was 15, contact 15, and XY crossing 5.
The paired development result is exploratory and does not isolate an
augmentation-only causal effect because both Base and Control were retrained.
No extra trial was launched after the failed gate.

The native runner completed and retained all 60 rows, then its historical report
assembly raised `KeyError: 'phase_counts'` after those rows were saved (the
summary object did not contain that legacy key). The first native command exit
status is retained in `paired/phase_runtime.json`; no episode was repeated.
`analysis_report_correction.json` records the analysis-only source change,
`analysis_report_inputs.json` freezes all 60 row hashes and the R38 result-tree
hash, and `paired/analysis.json` / `paired/status.json` were rebuilt offline.
`native_evaluation_rerun` is false, and the repaired report now validates exact
row/marginal counts, phase counts, terminal flags and R38 identity. Its analysis
input snapshot SHA-256 is
`57d21f04882832558d07619a4485b73df55329237721bd6f86d28be03d08f675` and final
analysis SHA-256 is
`9200e31106bb13e9d4666d048fe14e250628cdb36f915f00f14979bc2ab38bb4`.

Verification: full constrain suite **580 passed, 28 warnings** in 350.73 s;
focused gauge/report tests **5 passed**. The final R40 evaluator/report code is
under `examples/baselines/constrain/pickcube_xy_gauge_round40.py`; the reusable
named-field transform/RNG is `pickcube_xy_gauge.py`, matched trainer changes are
in `pickcube_zscore_round37.py`, and invariant/strict-metadata/report-contract
tests are in `tests/test_pickcube_xy_gauge.py`. All native rows and checkpoint
hashes are retained. No extra training, replay, evaluation scene, or process
remains.

## Round 39 — one-scale learned residual guidance trial (2026-10-29)

Round 39 performed **one** inference-only native trial using the exact R38 EMA
checkpoints, fixed position/controller-unit state profile, frozen dataset and
existing 30 R36 development scenes. No demonstrations/layouts were collected,
no model was trained, no checkpoint was written, and ManiSkill/PlaceSphere
sources were untouched. The experiment is exploratory and makes no formal
performance or guaranteed-safety claim.

### Preregistration and runtime operation

Registration:
`outputs/exp/20261029_000001_pickcube_residual_guidance_r39/preregistration.json`
(file SHA-256
`c0d56a00f24a85ccbf6283b815157ab48f31bcfa587af34388a2ec883808bdb9`, internal
registration identity `2f2b133098cf48db685fde3e19c2c22e626f1f5e4d388ce58c05a5260136173d`).
It was created before native execution and pins source aggregate
`654ae12d5c9868dd429e8669e1eab9043f36fdbd960d86658acb15af3c33e7f8`, exact
Base/Control final EMA SHA-256 values from R38 (`6a7c1af0…3d9869` /
`1c3276e3…63bf42`), the position profile file SHA-256
`d14d56e3…23f543` and profile identity
`8bcb1df3…b3d7fef`, frozen HDF5/manifest/validation identities
(`0c74b88e…349f4f` / `6b9d2c75…e5c486` / `f4756f69…d5b8f`), scene manifest/HDF5,
and per-file hashes for immutable snapshots of every R38 Base and Control row.
The R38 report had identified an analysis-only report assembly mismatch; its
retained rows, not a rerun, were snapshotted and used here.

The sole treatment is epsilon-space
`eps_base + 1.5 * (eps_control - eps_base)`: both outputs use the same current
sample, timestep, encoded 84-D global condition and raw-observation-derived
hint; `eps_base` calls the frozen Base branch (`enabled=False`) of the same
Control checkpoint and `eps_control` calls its learned enabled branch. This is
**learned conditional residual guidance**, not classifier-free guidance (there
is no unconditional branch trained for this use). Base is unchanged and has
guidance N/A. Runtime scale is not part of or written into either checkpoint.
No action/pose/grip correction, data feedback, or geometry shield is applied.

DDPM remained 100/100 steps with the squared-cosine beta schedule, epsilon
prediction and original clipping; predicted slice `[1:9]`, native 7-D identity
actions, execute/replan 8. Per-scene seed reset and one Gaussian latent draw per
decision are unchanged; the extra deterministic Base forward consumes no RNG.
An offline scale-1 check on the first retained R38 Control scene reproduces its
saved first native action bit-for-bit (max absolute difference 0.0), confirming
the default runtime fast path preserves the historical action output.
The fixed limits were 600 s per episode per policy and 36,000 s overall. Runtime
sources and all frozen input/checkpoint hashes matched after the trial.

### Native result and paired analysis

Exactly 30 existing scenes per policy completed (60 total), with 0 runtime
errors, 0 wall timeouts, and total elapsed time 602.26 s on an RTX 5070. Base
reproduced R38 exactly scene-by-scene, including native step traces/outcomes;
the guided comparison is therefore against a validated unchanged Base. Counts
below use denominator 30 in every case:

| Policy | Task success | Safe success | Contact | XY crossing |
| --- | ---: | ---: | ---: | ---: |
| R38 Base / R39 Base | 22 | 0 | 30 | 23 |
| R38 Control | 29 | 13 | 16 | 6 |
| R39 learned residual guidance, scale 1.5 | 29 | 11 | 17 | 6 |

R38 Control → R39 guided paired transitions (both / neither; old-only / new-only)
were task `28 / 0; regressed 1 / improved 1`, safe `9 / 15; regressed 4 /
improved 2`, contact `14 / 11; cleared 2 / introduced 3`, and XY `3 / 21;
cleared 3 / introduced 3`. The exact marginals reconcile to each transition
table. Against R39 Base, guidance transitions were task `21 both, 8 improved,
1 regressed`; safe `11 improved`; contact `13 cleared, 17 both`; and XY `17
cleared, 6 both`.

Guided first-contact phases: transport 11, goal stabilization 4, grasp approach
1, lift 1; first-contact link counts: hand 13, cube 1, left finger 1, link7 1,
right finger 1. The native recorder does not retain contact entity-pair
positions, so these remain unknown. First-success step min/median/p90/max was
78 / 107 / 218 / 334. Per-episode wall time min/median/p90/max was 5.952 /
8.241 / 20.266 / 35.163 s for guided Control and 3.396 / 7.555 / 15.605 /
15.782 s for Base. Guided final 3-D goal-error min/median/p90/max was 0.00332 /
0.00895 / 0.01781 / 0.09949 m.

The preregistered gate (`task >=27`, `safe >=26`, `contact <=3`, `XY <=3`) **did
not pass**: task passed; safe, contact and XY failed. Scale 1.5 is rejected for
this trial; it does not justify trying another guidance scale. The result is
consistent with a small safety regression versus R38 Control and does not
establish route-preserving control.

Retained native rows and summaries are in `paired/base_dp/` and
`paired/guided_control/`. The report-only analysis was enriched from these rows
without rerunning the environment. Final `paired/analysis.json` SHA-256 is
`80cba3052d92dbbd3cb1632af06061076069b19ae08933d0b5ed53fcf482c0e3`; the
pre-enrichment native report is preserved byte-for-byte as
`paired/native_analysis_before_report_enrichment.json` (SHA-256
`0f3ea358326897a5b89a9302f752d6df3e560ed41981610d65366d42ec62573d`) and the
report-only chronology is in `paired/analysis_report_fix.json`. Post-run audit
confirms frozen data, both checkpoints, profile and scene files all retain
their preregistered hashes. No policy/evaluation processes remain active.

## Round 38 — fixed PickCube position/controller-unit matched retraining (2026-10-28)

Round 38 completed exactly one matched Base/Control retraining and one paired
30-scene native development trial. It is a repeated-development engineering
result, not a formal performance/generalization claim. No demos, HDF5 variants,
or layouts were created; ManiSkill task/controller code and PlaceSphere artifacts
were not modified. Ticket 05 has no local issue file in this checkout; this is
the next-slice execution record.

### Frozen contract and fixed observation profile

The preregistration is
`outputs/exp/20261028_221700_pickcube_position_units_r38/preregistration.json`
(file SHA-256 `49b659941b3e7a36c492dc17f83cf27428593ac6f647113efc170fa26ec9dcd9`,
registration identity `65963d4c4c01f95e497e4387051b17153a63384584af0b824461131cce4b1b38`).
The fixed profile is `pickcube_position_controller_units_v1`, profile identity
`8bcb1df3252e86b085d51507f72ce4d7722eadea7cc46d676d6da7d33b3d7fef`, stored at
`position_encoding.json` (file SHA-256
`d14d56e3645504b81db4a41f32cf6869b805dbf003194b26433cfceb6023f543`). It has
42 dimensions, zero offset, no fitted/data-derived statistics, no clipping, and
the following scale vector, derived from `NAMED_FIELDS` (not bare offsets):

| Named field slice | Scale | Reason |
| --- | ---: | --- |
| `agent/qpos[0:9]`, `agent/qvel[0:9]`, `extra/is_grasped[0]` | 1 | unchanged |
| `extra/tcp_pose[0:3]` | 10 | meters / 0.1 m controller translation reference |
| `extra/tcp_pose[3:7]` | 1 | quaternion unchanged |
| `extra/goal_pos[0:3]` | 10 | meters / 0.1 m reference |
| `extra/obj_pose[0:3]` | 10 | meters / 0.1 m reference |
| `extra/obj_pose[3:7]` | 1 | quaternion unchanged |
| `extra/tcp_to_obj_pos[0:3]`, `extra/obj_to_goal_pos[0:3]` | 10 | meters / 0.1 m reference |

Thus the scaled state indices are exactly 19–21, 26–31, and 36–41; all other
state channels are bitwise identity. The `0.1 m` reference is the checked Panda
`pd_ee_delta_pose` contract (`translation_scale_m: 0.1`), with runtime named
observation schema verification passing at 45 raw / 42 policy values. The
obstacle hint remains the existing latest-frame relative condition computed
from raw 45-D world-meter observations. Encoding is applied once at the common
42-D training/inference condition seam; no encoded observations are stored.

Action representation is unchanged native identity 7-D `pd_ee_delta_pose`:
DDPM samples are used directly, without codec, decode, explicit clamp, new
action meaning, or policy shield. The frozen-data reader still enforces the
existing `[-1,1]` action bounds and CLOSED (`-1`) obstacle terminal padding.
Base and Control have distinct typed checkpoint formats and both store the
same profile identity; a legacy identity or R37 z-score Base cannot initialize
this Control contract despite identical tensor dimensions.

Frozen data remains
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, validation
`f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`; exactly
direct IDs 0–999 / 83,004 windows and obstacle IDs 1000–1499 / 45,840 windows.
The R36 paired primary comparator and its same 30 restored development scenes
remain unchanged. The optional R37-training-vs-initial-dev 5-sigma feature
diagnostic was not run; no dev-derived quantities selected the fixed factor.

### Matched training and audit

Both networks retain the 42-D state / 84-D two-frame condition / 7-D action
dimensions and official architecture. Base trained direct-only for 30,000
updates, then Control trained obstacle-only for 150,000 from that Base final
EMA with Base frozen. Seed 1, batch 1,024, AdamW `1e-4`, betas `(0.95,0.999)`,
weight decay `1e-6`, warmup 500, cosine schedule, EMA power 0.75, and unweighted
native epsilon MSE were held fixed. Both use squared-cosine DDPM-100 epsilon
diffusion and final EMA selected before training. Atomic complete-state recovery
checkpoints exist every 10,000 updates (3 Base, 15 Control); loss logs contain
exactly 30,000 / 150,000 finite rows.

| Policy | Updates | Elapsed | First → final loss | Final checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Base | 30,000 | 842.53 s | 1.17828774 → 0.00278578 | `6a7c1af04970d3eb871c7cf607e2028dd847706070a03ae26e8e6aa8053d9869` |
| Control | 150,000 | 4,233.00 s | 0.03769969 → 0.00168165 | `1c3276e3d52efcb4f349c75612253750898a564f4ceb523c422b61c01763bf42` |

Logs/config/summaries are under `outputs/exp/20261028_221700_pickcube_position_units_r38/train_base/`
and `train_constrain/`; checkpoints/recovery states are under matching
`outputs/ckpt/20261028_221700_pickcube_position_units_r38_{base,constrain}/`.
Real-data Base and Control preflights were finite, with raw 45-D hint separation
and zero-control epsilon max difference 0.0. Strict standalone checkpoint loads
passed; independently seeded Base-vs-disabled-Control inference also matched
exactly (max difference 0.0). Post-training source aggregate SHA-256
`c4f468885eaa21ed741ac8ba48d064c77678eecce51b1fde4cb9c7ab39c03afd` and both
training-data identities match preregistration. GPU allocator peak was not
recorded.

### One paired native trial and result

Native preregistration was written before stepping at
`paired/preregistration.json` (file SHA-256
`c2202604a1442c5deec78c2ab8e477f2b9df53586e053b72763f05c3d64edd57`,
evaluation identity `1c99b817656e519469f9915786681750298d4f786ac95c84ad3150c1bb60b3ed`).
It pins both checkpoint hashes, profile, source aggregate
`c4f468885eaa21ed741ac8ba48d064c77678eecce51b1fde4cb9c7ab39c03afd`, R36 scene
manifest/HDF5, 30 episodes each, 500 control steps, per-episode 300 s and overall
36,000 s limits. Sampling is archived DDPM-100, epsilon, squared-cosine, slice
`[1:9]`, execute/replan 8; each policy is separately seeded from the same scene
seed. All 30/30 per policy completed with zero runtime errors; native state,
controller, raw observation and encoded-condition hashes pair scene-by-scene and
match R36 restored states. Every episode retains actual terminated/truncated
flags, native action-step traces, grasp/lift/transport/success stages, first
contact phase, final cube/goal XYZ and 3-D goal error. Complete contact entity
pair positions were not exposed by this record API and are not inferred.

| Policy | Task success | Safe success | Contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Position-unit Base | 22/30 | 0/30 | 30/30 | 23/30 | 0 |
| Position-unit Control | 29/30 | 13/30 | 16/30 | 6/30 | 0 |
| Primary R36 Base | 10/30 | 0/30 | 30/30 | 15/30 | 0 |
| Primary R36 Control | 25/30 | 16/30 | 13/30 | 4/30 | 0 |
| Secondary R37 z-score Control | 7/30 | 3/30 | 13/30 | 2/30 | 0 |

Current Base → Control paired transitions (improved/cleared, regressed/introduced,
both, neither): task `8,1,21,0`; safe `13,0,0,17`; contact `14,0,16,0`; XY
`19,2,4,5`. Against R36 Control, this Control transitions task `5 improved / 1
regressed / 24 both / 0 neither`; safe `2 / 5 / 11 / 12`; contact cleared /
introduced / both / neither `2 / 5 / 11 / 12`; XY `2 / 4 / 2 / 22`. These are
paired development comparisons, not isolated single-factor causal estimates.

The frozen gate **fails**: task 29 ≥27 passes; safe 13 <26, contact 16 >3, and
XY 6 >3 fail. The run is complete, but the blocker remains. Control terminal
phases were 29 task success / 1 lift-no-transport; first contacts were 2 grasp
approach / 1 lift / 11 transport / 2 goal stabilization. First-success steps
min/median/p90/max were 58/104/181/269. Per-policy wall seconds min/median/p90/max
were Base 3.422/7.554/15.736/16.749 and Control 3.440/5.291/9.333/23.466.
Final 3-D goal error min/median/p90/max was Base 0.0060/0.0243/0.1551/0.2167 m
and Control 0.0041/0.0076/0.0123/0.1190 m.

The native evaluator completed and retained all 60 rows but its first report
assembly hit an analysis-only missing-key mismatch (`interpretation` versus
`comparison_interpretation`). No rollout was repeated. The report was rebuilt
from immutable episode rows at `paired/analysis.json` (SHA-256
`00f3e90ce90ed949ef36d799cd231f757866633d7ac3189f95d1ddf6226de7ee`); status
records `analysis_repaired_from_retained_rows: true` and
`native_evaluation_rerun: false`. Only the R37 report assembly fallback changed
after native execution; preregistered runtime source identity remains intact in
the paired preregistration and analysis. Post-report source SHA-256 is
`249fc078e43a05116c4beaee58a2f23e2389e440fd0b7216d1edec0b99c48eae`; all other
registered files match. Offline transition analysis validates integer cells,
both cohort marginals, and the complete 30-row table; `cleared_old_only` is
explicitly old-true/new-false and `introduced_new_only` old-false/new-true.

Verification: full constrain suite **567 passed, 28 warnings** in 85.34 s after
the report-only correction and exact-marginal validator; `git diff --check`
passed. No evaluation worker remains active.

## Round 37 — matched PickCube direct-observation z-score retraining (2026-10-28)

Round 37 completed one matched Base/Constrain retraining and one paired 30-scene
native trial. The single changed factor was the shared 42-D observation encoding:
both policies were trained from scratch/initialized from the newly trained Base
EMA with `pickcube_direct_zscore_v1`. This is an end-to-end observation-profile
comparison, **not** an isolated Constrain-control or data-effect attribution.
No demonstrations, layouts, or encoded HDF5 files were created; the frozen
500-scene physical dataset and PlaceSphere artifacts were not modified.

### Frozen inputs, observation profile, and preregistration

The experiment registration is
`outputs/exp/20261028_200000_pickcube_direct_zscore_r37/preregistration.json`
(SHA-256 `d480bae967ae59d3ed3619e1da4b16ef6a42adcf99d83202ba7ac6d736494284`).
The fitted artifact is
`outputs/exp/20261028_220000_pickcube_zscore_r37/normalizer_stats.json`, file
SHA-256 `730d2e297253f9b4b14a3059902921ae226f7abe20094917d2e2a99f6396642f`,
statistics identity `d0bbcd4c00d6320635f97972d3211c1ed2b150b22b46a59780ad6d6259d08645`.
It fits float64 population statistics over all **85,004 native T+1 frames** in
the 1,000 direct trajectories only (IDs 0–999); every feature count is 85,004.
The transform is `(raw42 - mean42) / max(population_std42, 0.01)`, cast to
float32 at runtime, with no observation clipping. The 42-value named schema is
the existing PickCube state prefix; the named 3-value obstacle suffix is never
in the Base/Constrain state condition. Binary and quaternion values remain
linear channels; no features or model dimensions were added. The old relative
obstacle hint remains clipped according to its existing config and is calculated
from the raw 45-D latest frame, never from normalized TCP coordinates.

| Feature (native unit) | Mean | Population std | Minimum | RMS | Floored at 0.01 |
| --- | ---: | ---: | ---: | ---: | :---: |
| `agent/qpos[0]` (rad) | 0.044140794 | 0.106621053 | -0.248344913 | 0.115396961 | no |
| `agent/qpos[1]` (rad) | 0.463923214 | 0.178997469 | 0.0439996161 | 0.497257320 | no |
| `agent/qpos[2]` (rad) | 0.0107171889 | 0.0749889268 | -0.205471188 | 0.0757508896 | no |
| `agent/qpos[3]` (rad) | -2.09797907 | 0.360908903 | -2.59803200 | 2.12879577 | no |
| `agent/qpos[4]` (rad) | -0.00823821533 | 0.0640477291 | -0.221264020 | 0.0645753807 | no |
| `agent/qpos[5]` (rad) | 2.55457824 | 0.219548003 | 1.90018857 | 2.56399518 | no |
| `agent/qpos[6]` (rad) | 0.843373261 | 0.377329590 | -0.0951657072 | 0.923935105 | no |
| `agent/qpos[7]` (m) | 0.0268357181 | 0.0105686175 | 0.0182817224 | 0.0288418350 | no |
| `agent/qpos[8]` (m) | 0.0268155991 | 0.0105849011 | 0.0182602946 | 0.0288290910 | no |
| `agent/qvel[0]` (rad/s) | 0.00802788499 | 0.101574173 | -0.501390398 | 0.101890920 | no |
| `agent/qvel[1]` (rad/s) | 0.105793320 | 0.319833605 | -0.767346501 | 0.336876478 | no |
| `agent/qvel[2]` (rad/s) | 0.00151065718 | 0.0481899872 | -0.302984983 | 0.0482136594 | no |
| `agent/qvel[3]` (rad/s) | 0.148600861 | 0.433286034 | -1.15135872 | 0.458060043 | no |
| `agent/qvel[4]` (rad/s) | -0.00197533956 | 0.0493963883 | -0.375561386 | 0.0494358690 | no |
| `agent/qvel[5]` (rad/s) | -0.0529015899 | 0.292213551 | -1.07203209 | 0.296963529 | no |
| `agent/qvel[6]` (rad/s) | 0.00711125312 | 0.157346644 | -0.841103375 | 0.157507258 | no |
| `agent/qvel[7]` (m/s) | -0.0186116932 | 0.0299030074 | -0.379620582 | 0.0352219388 | no |
| `agent/qvel[8]` (m/s) | 0.00439317390 | 0.0310723083 | -0.418345451 | 0.0313813371 | no |
| `extra/is_grasped[0]` (binary) | 0.597430709 | 0.490415392 | 0 | 0.772936420 | no |
| `extra/tcp_pose[0]` (m) | -0.0473682268 | 0.0850955015 | -0.184164703 | 0.0973909302 | no |
| `extra/tcp_pose[1]` (m) | 0.0310782897 | 0.0506820379 | -0.0610247739 | 0.0594519054 | no |
| `extra/tcp_pose[2]` (m) | 0.0948521102 | 0.0440245605 | 0.0216526166 | 0.104570956 | no |
| `extra/tcp_pose[3]` (quaternion) | -0.000136261643 | 0.00183554629 | -0.0294772387 | 0.00184059703 | **yes** |
| `extra/tcp_pose[4]` (quaternion) | 0.980205160 | 0.0206232690 | 0.911496222 | 0.980422090 | no |
| `extra/tcp_pose[5]` (quaternion) | 0.00140957371 | 0.196833353 | -0.411276847 | 0.196838400 | no |
| `extra/tcp_pose[6]` (quaternion) | -0.00264409096 | 0.00410619275 | -0.0698821619 | 0.00488385461 | **yes** |
| `extra/goal_pos[0]` (m) | 0.147484437 | 0.0246798569 | 0.0482905619 | 0.149535128 | no |
| `extra/goal_pos[1]` (m) | 0.0321654257 | 0.0555496288 | -0.0599714406 | 0.0641901540 | no |
| `extra/goal_pos[2]` (m) | 0.119999997 | 0 | 0.119999997 | 0.119999997 | **yes** |
| `extra/obj_pose[0]` (m) | -0.0527428751 | 0.0880923124 | -0.183461741 | 0.102674565 | no |
| `extra/obj_pose[1]` (m) | 0.0330848460 | 0.0521981803 | -0.0616050027 | 0.0618001380 | no |
| `extra/obj_pose[2]` (m) | 0.0700521921 | 0.0543431891 | 0.0199918468 | 0.0886594147 | no |
| `extra/obj_pose[3]` (quaternion) | 0.645150211 | 0.297919692 | -0.00562614622 | 0.710615886 | no |
| `extra/obj_pose[4]` (quaternion) | 0.000126041122 | 0.00139330437 | -0.00717262086 | 0.00139899372 | **yes** |
| `extra/obj_pose[5]` (quaternion) | 0.000421706013 | 0.00136281550 | -0.00555317104 | 0.00142657002 | **yes** |
| `extra/obj_pose[6]` (quaternion) | -0.0380681045 | 0.702546763 | -1 | 0.703577383 | no |
| `extra/tcp_to_obj_pos[0]` (m) | -0.00537464825 | 0.0236640297 | -0.203784332 | 0.0242667085 | no |
| `extra/tcp_to_obj_pos[1]` (m) | 0.00200655628 | 0.0136905664 | -0.0874460638 | 0.0138368304 | no |
| `extra/tcp_to_obj_pos[2]` (m) | -0.0247999181 | 0.0347967981 | -0.204242811 | 0.0427300023 | no |
| `extra/obj_to_goal_pos[0]` (m) | 0.200227312 | 0.0897430491 | 0.00101563334 | 0.219419214 | no |
| `extra/obj_to_goal_pos[1]` (m) | -0.000919420356 | 0.0734646972 | -0.176836476 | 0.0734704503 | no |
| `extra/obj_to_goal_pos[2]` (m) | 0.0499478052 | 0.0543431890 | -0.0183884129 | 0.0738103342 | no |

The physical dataset remains
`demos/PickCube-transport-detour/20261028_164500_frozen500/`: HDF5 SHA-256
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`, manifest
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, validation
`f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`. The
selected classes are direct IDs 0–999 and obstacle IDs 1000–1499, with the
registered 83,004 / 45,840 window counts. Raw HDF5 actions remain native
seven-channel identity values; obstacle terminal gripper padding remains -1.

The registered runtime/checkpoint profile is typed separately from legacy
identity-normalized checkpoints: Base `pickcube_direct_zscore_base_v1`, Constrain
`pickcube_direct_zscore_constrain_v1`. Model dimensions remain Base 42-D / 7-D,
Constrain 84-D condition / 7-D action, horizons 2/16/8 and channels 64/128/256.
Training used 30,000 direct-only Base updates, then 150,000 obstacle-only
Constrain updates initialized from that new Base EMA with the Base frozen. Both
used seed 1, batch 1,024, AdamW `1e-4`, betas `(0.95,0.999)`, weight decay
`1e-6`, 500-step warmup, cosine schedule, EMA power 0.75, unweighted epsilon
MSE, and final EMA fixed before training. The inference protocol is archived
DDPM-100 with `squaredcos_cap_v2`, epsilon prediction, `clip_sample=True`,
prediction slice `[1:9]`, execute/replan 8, direct native actions and no decode
or explicit clamp.

### Training and artifact identities

Base run/logs/config: `outputs/exp/20261028_200000_pickcube_direct_zscore_r37/train_base/`;
recovery/final checkpoints:
`outputs/ckpt/20261028_200000_pickcube_direct_zscore_r37_base/`. It completed
30,000/30,000 updates in **841.96 s**, all losses finite, first `1.16361868`,
last `0.00290657`; final checkpoint SHA-256
`7498cbeaffd3675b504b4b448927e3288b9e20703fe62803e6d134e0ee9e8cbf`.
Atomic recovery checkpoints were retained at 10,000-update intervals.

Constrain run/logs/config:
`outputs/exp/20261028_200000_pickcube_direct_zscore_r37/train_constrain/`;
recovery/final checkpoints:
`outputs/ckpt/20261028_200000_pickcube_direct_zscore_r37_constrain/`. It
completed 150,000/150,000 updates in **4,236.80 s**, all losses finite, first
`0.03401160`, last `0.00181310`; strict final EMA load passed and the frozen
Base state hash is `4a1d8b29b65f08f526960088652b724737f0bbad1c9c88aef949156474a00fa6`.
Final checkpoint SHA-256 is
`20d5bb6e2027b34275a9a4d9d81b46d0d4d6c24ad01b91fe2c4cb318d4279ba3`.
Both post-training source and frozen-data identities match preregistration.
The real-data Base and Constrain preflights were finite with the planned schemas;
disabled-control epsilon max difference was 0.0.
CUDA allocator peak memory was not recorded by these trainers and cannot be
reconstructed from the retained artifacts.

### Paired native trial and gate

The one native trial is retained under
`outputs/exp/20261028_200000_pickcube_direct_zscore_r37/paired/`. Its evaluation
preregistration (written before the first native step) pins both final checkpoint
hashes, normalizer file/statistics hashes, sampler and the existing 30 R36
development-scene state/controller/observation source. All **30/30 per policy**
completed with zero runtime errors; all 30 initial native state, controller,
observation and normalized-condition identities paired across policies, and the
restored physical scene hashes match R36 scene-by-scene.

| Policy | Task | Safe | Obstacle contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized Base DP, 30k | 1/30 | 0/30 | 25/30 | 11/30 | 0 |
| Normalized Constrain, 150k | 7/30 | 3/30 | 13/30 | 2/30 | 0 |
| R36 Base reference | 10/30 | 0/30 | 30/30 | 15/30 | 0 |
| R36 Constrain reference | 25/30 | 16/30 | 13/30 | 4/30 | 0 |

Constrain versus Base transitions (task / safe / contact / XY): task improved 7,
regressed 1, both 0, neither 22; safe improved 3, regressed 0, both 0, neither
27; contact cleared 15, introduced 3, both 10, neither 2; XY cleared 10,
introduced 1, both 1, neither 18.

Against R36, normalized Base transitions were task improved 1 / regressed 10 /
both 0 / neither 19; contact cleared 5 / introduced 0 / both 25 / neither 0;
XY cleared 8 / introduced 4 / both 7 / neither 11. Normalized Constrain versus
R36 Constrain transitions were task improved 1 / regressed 19 / both 6 / neither
4; safe improved 1 / regressed 14 / both 2 / neither 13; contact cleared 9 /
introduced 9 / both 4 / neither 8; XY cleared 4 / introduced 2 / both 0 /
neither 24. The frozen gate **failed**: 7 task <27, 3 safe <26, 13 contacts >3,
and 2 XY crossings <=3. This is a repeated-dev engineering result, not a formal
performance or generalization claim.

Terminal phases (Base / Constrain) were no-grasp 4/0, grasp-no-lift 3/1,
lift-no-transport 10/20, transport/place failure 12/2, task success 1/7.
First-contact phases (Base / Constrain) were grasp approach 2/1, lift 15/1,
transport 8/9, goal stabilization 0/2. Contact links were retained per episode;
complete contact entity-pair positions were unavailable. Per-episode wall time,
native action/flag traces, final cube XYZ, goal XYZ and 3-D goal distance are in
the episode JSONs. Wall time min/median/p90/max was 8.925/15.587/16.008/16.125 s
(Base) and 4.820/23.082/23.964/24.227 s (Constrain). Final goal-distance
min/median/p90/max was 0.0247/0.1406/0.3180/0.3672 m (Base) and
0.0069/0.2510/0.3866/0.4305 m (Constrain). Success step was 279 for the single
Base success; Constrain success step min/median/p90/max was 91/110/158.4/225.

The native evaluator retained all 60 episodes before its report aggregation hit
an analysis-only missing phase-summary field. No rollout was repeated. The
report was recomputed from those retained rows at
`paired/analysis.json`; `paired/preregistration.json` preserves the exact
runtime source identity. The subsequent analysis-only correction changed only
`pickcube_zscore_round37.py` (registered SHA-256
`768ac8e45e94b49521ecc3a771febf0b6b764dd23dc2cebd3e0d581738094310`, post-report
SHA-256 `5cbae523c95130a8161c92ef52a804c2b37651bcce94d2c8857616fb1adfd906`);
all other registered source hashes remained identical. It adds an offline
reader for retained rows and does not change inference/training. Dataset hashes
were rechecked after the trial and remain unchanged. The machine-readable data,
checkpoint, episode-count, scene-pairing and source-difference audit passed at
`paired/post_trial_audit.json`. Registered source aggregate SHA-256 was
`c8262d9a03ba3d8028097c792c29c396692536a9a53f3143fbb7f5048eaba1ac`; the
post-report source aggregate is `1d974752580036acf582e10d66897fb9e304989470c7648ae2dddf4d8659045f`.
No process was intentionally left running.

## Round 36 — R35 report correction and archived PickCube sampler recovery (2026-10-28)

R35's original paired records, preregistration, and analysis JSON are preserved
unchanged. A separately written `outputs/exp/20261028_145500_pickcube_native_control_r35/paired/correction_r36.json`
recomputes the old-90k-Control → R35-Control transitions directly from all 30
raw `result` records. The original report had the adverse-event labels reversed
for contact and XY and had incorrect safe/task transition values. Correct values:

| Metric | Old → R35 marginals | Transitions |
| --- | --- | --- |
| Task | 27 → 26 | improved 2 / regressed 3 / both 24 / neither 1 |
| Safe | 15 → 16 | improved 7 / regressed 6 / both 9 / neither 8 |
| Contact | 15 → 13 | cleared 8 / introduced 6 / both 7 / neither 9 |
| XY crossing | 3 → 9 | cleared 2 / introduced 8 / both 1 / neither 19 |

The corrected counts reconcile exactly to both cohorts' positive-outcome or
violation marginals. They are an analysis correction, not a native rerun. In
particular, the original R35 100/500 result remains R35's recorded result and is
not relabeled as DDPM-100.

Archived source recovery resolves the earlier protocol uncertainty: Git blob
`43b08ab:examples/baselines/constrain/pickcube_learning_pilot.py` has SHA-256
`24522af7d02d3cdce98742ad7df0f3e7bbfba62321272fa1de8d509388e3c0ec`, exactly
matching the R13 preregistration's source identity. That historical adapter
uses `DDPMScheduler`, 100 training/inference timesteps and the native
`[1:9]` prediction slice executed directly in chunks of eight. It does not use
the later DDIM-50 profile or a 0.35 decode/clamp. This archival proof establishes
the intended same-weights R36 restoration protocol; it does not by itself prove
runtime equivalence or native outcomes.

### R36 native trial

Registration was frozen at
`outputs/exp/20261028_191000_pickcube_archived_ddpm100_r36/preregistration.json`
before native evaluation. The one trial used the exact R35 Base and Control
checkpoint hashes above, all 30 frozen scenes, 500 control steps, and 300 s per
episode. The selected outer timeout was 36,000 s. Runtime profile confirms
DDPM-100, stride 8, prediction `[1:9]`, direct native actions and source identity
matching registration. The original frozen dataset HDF5 and manifest hashes
were verified again after evaluation. No training, data creation, new layouts,
or checkpoint writes occurred.

Both cohorts completed 30/30 with zero runtime errors. Base scored **10 task / 0
safe / 30 contact / 15 XY**; Control scored **25 / 16 / 13 / 4**. On the exact
same R35 scenes and weights, restored state/controller/observation hashes match
scene-by-scene. R36 Base exactly reproduced the retained 90k experiment's old
Base outcomes scene-by-scene (10/0/30/15), supporting the restored native
DDPM-100 path. Control versus R35 transitions (task/safe/contact/XY) were task
2 improved / 3 regressed / 23 both / 2 neither; safe 6 improved / 6 regressed /
10 both / 8 neither; contact 6 cleared / 6 introduced / 7 both / 11 neither;
XY 6 cleared / 1 introduced / 3 both / 20 neither. Compared to the old 90k
Control, R36 Control transitions were task 2 improved / 4 regressed / 23 both /
1 neither; safe 4 improved / 3 regressed / 12 both / 11 neither; contact 5
cleared / 3 introduced / 10 both / 12 neither; XY 2 cleared / 3 introduced / 1
both / 24 neither. This secondary comparison still combines 500-vs-300 data and
150k-vs-90k updates.

R36's frozen gate is **pause**: task 25/30 <27, safe 16/30 <26, contact 13/30
>3, and XY 4/30 >3; Ticket 07 remains blocked. Control's first-contact phases
were transport 5, goal stabilization 3, grasp approach 3, lift 2. Terminal
phases were task success 25, transport/place failure 2, lift/no transport 3.
Success termination steps had min/median/p90/max 58/98/130.8/223. Per-episode
wall time, final cube XYZ/3-D goal distance, and full contact-pair positions are
not available in this evaluator and were not reconstructed. Full results and
all transitions are in
`outputs/exp/20261028_191000_pickcube_archived_ddpm100_r36/paired/archived_protocol_comparison.json`;
native records and traces are in `paired/base_dp/` and `paired/constrain_dp/`.
The R35 original native records and original report remain unchanged; its
corrected post-hoc transition table is separately preserved as
`outputs/exp/20261028_145500_pickcube_native_control_r35/paired/correction_r36.json`.

Verification: focused PickCube pilot tests passed 48/48; the full constrain
suite passed **555/555** (28 warnings) in 99.66 s before the final analysis-only
marginal-validation addition; focused pilot tests passed again 48/48 afterward.
The validation changed the analysis source hash after the R36 run, not the
registered sampler/evaluator path: registered runtime source identity
`cb2737f357e85a55652cb13b3696c70359ee806894006aca3676e0cce602c0b7` matches the
preregistration and saved inference profile; current analysis source identity is
`761f518ac65ae3ac10f201bce013d27a1be0931f885bd94fb5c57c1029a202d9`. Offline same-weight adapter
checks executed the exact archived `get_action` AST methods, bound to the loaded
current model/scheduler/normalization namespaces; with a fixed observation and
RNG seed both configured adapters matched the archived methods exactly (max
absolute action difference 0), returning finite `(1,8,7)` native actions.
Evaluation
and gate were run serially under the activated `maniskill` environment; the
gate remains pause. No worker was intentionally left running.

## Cleanup status

## Round 35 — PickCube frozen 1000/500 native-action control-only retrain (2026-10-28)

One native-coordinate ControlNet retrain was completed on the frozen 500-obstacle
package, initialized from the retained original native-action 1000-direct Base
EMA. No Base retraining, data/layout creation, replay, action codec, PlaceSphere
change, or ManiSkill source edit occurred. This remains a repeated-dev engineering
result, not formal performance or generalization evidence.

### Frozen inputs, native representation, and source protocol

The frozen package is `demos/PickCube-transport-detour/20261028_164500_frozen500/`:
HDF5 `0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`,
manifest `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`,
validation `f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f`.
The opt-in native manifest reader requires all 1,500 IDs/classes, successful
held-at-goal metadata, safe obstacle records, the native seven-channel action
contract, closed `-1` obstacle terminal grip, and native window padding. It
returned exactly **83,004 direct / 45,840 obstacle** windows. The 1,000 original
direct observation and action arrays were individually compared byte-for-byte
against the source used to train the Base; all 1,000 matched. That source is
`demos/PickCube-transport-detour/20261024_104348_expansion/trajectory.h5`
(`2af959e3d3ebeba8bb046db6d9583d300e1533fc638c7d36eb6393dd28024dff`), with
manifest `26cb2b3f8c371315805cb6219e0a4314dce3461bac3aaa872d35ce2e5ca7fd45`.

The selected Base is the original
`outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt`, SHA-256
`dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d`. Its strict
checkpoint contract is `pickcube_pilot_base_v1`, step 30,000, selected EMA,
direct IDs 0–999, 42-D state condition, 7-D native `pd_ee_delta_pose`, identity
action normalization, and no action encoding. The newly trained control
checkpoint records a distinct typed native-identity contract; raw actions are
used directly, with no `0.35` codec scaling or post-sampling decode. Before and
after training, independently loaded Base and disabled-control epsilon and
DDIM-50 samples matched exactly (maximum absolute difference **0**).

Retained 90k preregistration, paired indices, and command records do not identify
the old inference scheduler, step count, or executed slice. Therefore the run is
explicitly labeled the predeclared fallback hypothesis **DDIM-50, eta 0,
100-training-timestep squared-cosine schedule, prediction `[1:9]`, execute 8,
replan every 8**, with the same scene-seed reset. It is **not claimed to reproduce
the exact old inference protocol**. This protocol limitation is material: the
same frozen-scene Base scored 8/30 here versus the retained old Base 10/30. The
checkpoint, scenes, and native initial states match, but old protocol evidence is
insufficient to treat that difference as a failed exact-protocol reproduction.
The old baseline was copied byte-for-byte and its tree SHA-256 verified before
the new paired evaluation: `98ba0bbaa5c74d6548ede8aa4a014e10a4456b9e55aae4343403d1d25450ca3d`.

### Training and recovery record

Original preregistration:
`outputs/exp/20261028_145500_pickcube_native_control_r35/preregistration.json`.
It freezes the dataset, Base, source identity, 30 dev scenes, final-EMA selection,
DDPM-100 epsilon objective, AdamW `1e-4`, batch 1,024, seed 1, warmup 500,
cosine schedule, EMA power 0.75, unweighted loss, and exactly 150,000 control-only
updates. Equal-batch exposure is approximately 3,351 obstacle-window epochs,
close to 3,342 for the retained 90k/300-window comparison; data size and update
budget both differ, so this is not a single-factor causal comparison.

Two completed 150k attempts are preserved with loss logs and failure records:
the first reached its post-training source audit but did not finalize because the
audit did not consult the already-frozen recovery identity; the second exposed
the missing explicit identity-action checkpoint field before it could save a
checkpoint. Neither was evaluated or used for checkpoint selection. Recovery
identity/preregistration history is retained as `recovery_identity_r1.json`,
`recovery_identity_r2.json`, `recovery_identity.json`, and
`recovery_preregistration.json` through `recovery_preregistration_r3.json`.
The selected completed attempt is
`outputs/exp/20261028_145500_pickcube_native_control_r35/train_constrain_retry2/`
with weights at
`outputs/ckpt/20261028_145500_pickcube_fullscene500_native_constrain_retry2/final.pt`,
SHA-256 `be5b2b740a59eaf428c71a906e15eb5c83d1cf807e1629c3bdf060ac31b8db93`.
It completed 150,000/150,000 updates in 4,497.32 recorded seconds; all losses
were finite (first `0.0308330`, final `0.00227748`, final-100 mean `0.00217241`).
The embedded Base state remained unchanged, trainable control parameters moved,
and strict EMA loading passed. Source identity was recomputed after training and
matched recovery SHA `6ab1502551991e49342be5053ca0e9b7d723370919c36b88d638b018d0e43c4e`.

### Paired native evaluation and gate

The evaluation used the exact same frozen 30 scene seeds and restored native
states for both policies, with matching state/controller/observation identity
hashes scene-by-scene. Both cohorts completed 30/30 with zero runtime errors.
Complete actual per-step native-action/flag traces are retained under
`paired/base_dp/` and `paired/constrain_dp/`; the original old records are in
`paired/baseline_snapshot/paired/`.

| Policy / reference | Task | Safe | Contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current Base, DDIM-50/8 hypothesis | 8/30 | 0/30 | 30/30 | 20/30 | 0 |
| Native Control, 150k / 500 obstacles | 26/30 | 16/30 | 13/30 | 9/30 | 0 |
| Retained old 90k Control | 27/30 | 15/30 | 15/30 | 3/30 | 0 |

Current Control versus current Base transitions (task/safe/contact/XY): task
20 improved / 2 regressed / 6 both / 2 neither; safe 16 improved / 0 regressed /
0 both / 14 neither; contact 17 cleared / 0 introduced / 13 both / 0 neither;
XY 12 cleared / 1 introduced / 8 both / 9 neither.

Against the retained old 90k Control, raw-record transitions are task 2 improved
/ 3 regressed / 24 both / 1 neither; safe 7 improved / 6 regressed / 9 both / 8
neither; contact 8 cleared / 6 introduced / 7 both / 9 neither; XY 2 cleared / 8
introduced / 1 both / 19 neither. The earlier R35 analysis JSON swapped the
adverse-event labels and misreported safe transitions; the immutable correction
is `paired/correction_r36.json`. Contact and XY remain worse than the old
Control. The exact old inference protocol is resolved by R36's archived-source
evidence, but R35 itself remains the recorded DDIM-50 run; R36 is the separate
same-weight DDPM-100 restoration.

The fixed gate is **pause / Ticket 07 remains blocked**: task 26/30 <27, safe
16/30 <26, contact 13/30 >3, and XY crossing 9/30 >3. New Control successes
terminated at median step 89.5 / p90 219.5 / max 293; first grasp median/p90
28.5/47.1 and lift 42.0/63.2. Its 13 first contacts were transport 9, goal
stabilization 3, lift 1 (reported contact links: Panda hand 9, fingers 3, cube
1). Four episodes terminated in transport/place failure. Full traces contain
native actions, grasp/lift/transport/success, collision/crossing, actual
termination/truncation, and contact phase/link. The existing trace schema does
not contain per-episode wall time, final cube XYZ/3-D goal distance, or complete
contact entity-pair geometry; these are not reconstructed. Aggregate details,
per-scene transitions and audit hashes are in `paired/evidence_audit.json` and
`paired/native_primary_analysis.json`; gate is `paired/gate.json`.

The 553-test constrain suite passed (28 existing warnings) before the frozen
recovery training and evaluation. No further training, data collection, layouts,
or evaluation trial was started. The PlaceSphere frozen configuration and
artifacts are untouched.

## Round 34 — fixed-weight PickCube 50→100 DDIM inference (2026-10-28)

This was one paired, exploratory inference-only comparison on the same 30 frozen
development scenes and the same R33 EMA checkpoints. R33's **actual** registered
native run used DDIM 50, eta 0, stride 2 for both policies; it did not execute
the 100-step inference originally requested for that round. Its preregistration
and native records are preserved unchanged and snapshotted under the R34 output.
R34 changes only inference from 50 to 100 DDIM steps. Training remains 100-step
DDPM with the same squared-cosine schedule; both checkpoint hashes, the codec,
data, episode limit, controller and held native-success condition remain fixed.
This is not a formal or generalization result.

The R33 training evidence inventory resolved a real directory-path mismatch:
Base's 30,000-step run/config/loss/summary are at the registered
`20261028_111705.../train_base/`; Constrain's actual 150,000-step run/config/loss/
summary are at `20261028_113532_pickcube_fullscene500_action_codec_retry1/
train_constrain/`, while R33 preregistration pointed at the separate
`20261028_125137.../train_constrain/` path. The retry tree contains exactly
150,000 contiguous finite loss rows and a matching summary/checkpoint; it was
catalogued in place, not fabricated or copied over the registered path. The
Base standalone checkpoint stores model/EMA and step but not optimizer,
scheduler, or RNG states. The Constrain checkpoint contains optimizer step
150,000, scheduler and RNG states, plus 100 EMA shadow tensors. Details and all
file hashes are in `r33_training_evidence.json`; the historical preregistration
remains the source of its original frozen identities.

### R34 preregistered execution and result

Registration: `outputs/exp/20261028_170000_pickcube_ddim100_paired_r34/
preregistration.json`; immutable R33 comparison inputs:
`r33_50step_snapshot/`; actual paired records and traces:
`paired_eval/`; derived paired analysis and gate:
`analysis.json`. The run strictly loaded the existing Base EMA
`2bd91abc5484559e3d7406b27c3e75e12d307e4df206d2361ed956a46533e1c2` and
Constrain EMA `e66c5ac700998648ec293ee29c2ca664fc00b1ad6cdfc2514140ac8533d236b3`.
The unchanged frozen package is HDF5
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f` and
manifest `6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`.
All source identities were recomputed after the run and matched the new
registration (`2abd51ea93ce7bf7ea795bf441d44cc603d7de584722f244bc3dd28b8cc9e03b`).
R33's 50-step records were copied byte-for-byte before analysis; both paired
tree hashes and the preregistration/hash match their originals.

| Policy | Inference | Task | Safe | Contact | XY crossing | Runtime errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | R33 50 → R34 100 | 14/30 → 11/30 | 0 → 0 | 30 → 30 | 19 → 19 | 0 |
| Constrain | R33 50 → R34 100 | 17/30 → 15/30 | 9 → 6 | 21 → 24 | 4 → 6 | 0 |

Each 100-step episode paired exactly on scene, seed, restored native-state,
controller and observation hashes; all 60 completed, with no runtime errors.
Constrain's task transitions (50→100) were both 14, improved 1, regressed 3,
neither 12; safe success both 6, regressed 3, neither 21; contact both 21,
introduced 3, neither 6; XY crossing both 3, cleared 1, introduced 3, neither
23. The Base had 3 task regressions and no task improvements; its contact stayed
30/30 and XY had one cleared and one introduced scene. R34 Constrain first
contact phases were transport 21, grasp approach 1, lift 1, goal stabilization
1; reported links were Panda hand 21, cube 1, and each finger 1. Terminal
phases were 15 task successes, 14 transport/place failures, and 1 lift/no
transport failure. First grasp/lift steps had medians 31.5/41; successful native
termination step median was 90 among 15 successes. Full step/action traces and
per-episode first-grasp, lift, transport, success, contact, phase and flags are
retained. The evaluator does not save final native poses/heights, 3D goal
distance, or complete contact entity-pair geometry; per-episode wall time was
not logged (the frozen wall deadline was 600 seconds). Those values are not
reconstructed.

The frozen exploratory gate (task ≥27, safe ≥26, contact ≤3, XY ≤3, complete
30/30 and zero runtime errors) is **pause**: Constrain scored 15/30 task, 6/30
safe, 24/30 contact and 6/30 XY crossing. The 100-step result is lower on task
and safe success and higher on contact and crossing than the R33 50-step result.
The prior 90k comparison remains secondary and multi-factor, not a causal
inference-step comparator. No training, new data/layouts, or checkpoint selection
occurred. PlaceSphere outputs/configuration and ManiSkill source were untouched.

Reproduction (bounded native run; run under the activated `maniskill` env):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 36000s \
python -m examples.baselines.constrain.pickcube_learning_pilot evaluate \
  --registration outputs/exp/20261028_170000_pickcube_ddim100_paired_r34/preregistration.json \
  --scene-dir outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes \
  --base-checkpoint outputs/ckpt/20261028_111705_pickcube_fullscene500_base/final.pt \
  --constrain-checkpoint outputs/ckpt/20261028_113532_pickcube_fullscene500_constrain/final.pt \
  --output-dir outputs/exp/20261028_170000_pickcube_ddim100_paired_r34/paired_eval \
  --inference-steps 100 --device cuda
```

Focused codec/pilot tests passed 48/48; the complete constrain suite passed
551/551 with 28 warnings in 83.23 seconds. A source-identity-checked native run
completed both 30-episode policy cohorts, and `git diff --check` passed.

## Round 32 — PickCube full-scene existing-pool inventory (2026-10-28)

The user-approved target is 500 distinct real demonstrations, not 500 task-
unique layouts. Within training, task conflict is same seed or componentwise
start/goal XY within 0.02 m. Full-scene conflict additionally requires cylinder
center XY within 0.005 m per component, height within 0.005 m, and radius within
0.001 m. Frozen dev30 and Ticket 03 exclusions remain strict task identity
regardless of cylinder geometry.

### Phase 1 — versioned, geometry-reconciled inventory

The initial inventory is retained at
`outputs/exp/20261028_143000_pickcube_fullscene_inventory/`; it has an obsolete
1070 candidate count because the old 300-package manifest omitted cylinder
height/radius. It was not rewritten. The corrected versioned report is
`outputs/exp/20261028_164000_pickcube_fullscene_inventory/inventory_geometry_reconciled.json`;
its pre-audit correction registration is
`registration_geometry_reconciliation.json`. Old 300 cylinder identities were
reconciled from the pinned retained/merged source manifests and catalog records.
Corrected eligible catalog additions: **706**, all from the 900-record direct-
scene catalog; the 64 pair, 127 increment, and 200 deficit catalogs contribute
zero after geometry-aware first-qualified selection. This leaves 506 unused candidates after the
successful one-pass selection. The old strict task-only result remains 103
additional task-unique candidates; the 706 count is full-scene eligible and is
not called task-unique.

Reconciliation inventory command (after activating `maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 300s \
  python -m examples.baselines.constrain.audit_pickcube_existing_pool \
  --output outputs/exp/20261028_164000_pickcube_fullscene_inventory/inventory_geometry_reconciled.json --full-scene
```

Corrected inventory SHA-256:
`b2e3c310b3086294e96271dd191568861ca9d99a7e3e682c7fd74b7003a6443d`.

### Phase 2 — frozen native replay and exact 1,000/500 package

Before native execution, `outputs/exp/20261028_164500_pickcube_fullscene500/registration.json`
froze the 706-candidate ordered pool, every candidate's source/catalog ID,
seed/start/goal/cylinder/waypoint, initial-state/action/observation fingerprints,
old/direct dataset hashes, external manifests, runtime versions, and replay
source identities. Registration SHA-256:
`de4fe563f356735bce5e0ada49057734e41dd615f53ee5fae02d97bb5f6b5462`.
The first three original-layout canaries passed fresh-environment initial
state/controller/45-D observation restoration and exact first-transition state,
controller, 45-D observation, info, and flags. All 200 selected candidates were
then attempted in stable source/manifest order; **200 accepted, 0 rejected, 0
runtime errors**, 157.20 seconds. All 200 came from the existing
`direct_scenes/assembled` catalog; no seed/layout/cylinder/waypoint was sampled
or modified. The accepted IDs and 506 unused IDs are recoverable from the full
registration and `materialization/replay_report.json`.

The package is `demos/PickCube-transport-detour/20261028_164500_frozen500/`:
exactly 1,000 direct IDs 0–999 and 500 obstacle IDs 1000–1499, all 1,500 IDs
covered once. Old direct/obstacle arrays for IDs 0–1299 match their source
arrays; all direct arrays also match the frozen 1,000-direct source. The old
1,000/300 source HDF5 remains unchanged at SHA-256
`3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`.
Obstacle counts are **333 task-unique / 500 full-scene-unique**; the 500 figure
is not represented as 500 task-unique layouts. New 200 are physically executed
native demonstrations, not copied trajectories. New cylinder geometry is
identical to the corresponding frozen catalog record; radius remains 0.03 m.

Final identities: `trajectory.h5`
`0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f`,
`trajectory.json`
`6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486`, and
`validation.json` is retained beside them. The independent audit verified exact
class IDs/counts, full-scene/task counts, all 1,300 old-source arrays, every new
T actions/T+1 raw-45/policy-42/full-state/controller/info/terminated/truncated
trace, native held success, collision/crossing-free prefix through first held
success, and all **200/200 exact initial plus first-transition restore audits**.
Every recorded controller snapshot is `{}`; runtime arm/gripper `config.use_target`
was present and false at every saved state.

Official PickCube action/window profile from the package audit:

| Class | Episodes | DP windows (obs=2, pred=16, before=1, after=14) | Length min/median/p90/p95/max | 6 pose-channel min/max | Grip min/max | Per-episode RMS quantiles (0/5/50/95/100%) |
| --- | ---: | ---: | --- | --- | --- | --- |
| Direct | 1,000 | 83,004 | 55 / 83 / 107.1 / 112 / 123 | each `[-0.35, +0.35]` | `[-1, +1]` | per-channel arrays in `validation.json` |
| Obstacle | 500 | 45,840 | 66 / 90 / 114 / 119 / 125 | each `[-0.35, +0.35]` | `[-1, +1]` | per-channel arrays in `validation.json` |

The first six stored values are the native `pd_ee_delta_pose` expert commands
(the collector's documented 0.35 action cap); the seventh is the native gripper
command. These are controller-native command coordinates, not meters/radians
or the PlaceSphere codec. Every native obstacle episode ends with grip `-1`
(closed) at held success. The opt-in official-window reader preserves the final
grip value in tail padding, so it does not invent an open-gripper terminal tail.
Future training should consume this version through
`PickCubeFullScene500Dataset` with both HDF5 and manifest SHA pins; retain native
action coordinates until a separate codec audit establishes a different
normalization. No training or evaluation was run in this round.

Focused tests passed 7/7; reader smoke loaded both classes and returned exactly
83,004 / 45,840 windows with `(2,42)` observations and `(16,7)` actions. Final
full suite: `timeout --signal=TERM --kill-after=20s 700s python -m pytest
examples/baselines/constrain/tests -q --tb=line` — **544 passed, 28 warnings,
86.51 s**. `git diff --check` passed. No replay, audit, or training worker
remained after verification.
One earlier registered packaging attempt at
`outputs/exp/20261028_163500_pickcube_fullscene500/` replayed 200 candidates but
failed during HDF5 state-series serialization; its incomplete package remains
retained as a failed attempt. The corrected 164500 registration persisted each
replay before assembly and produced the audited final dataset above.

## Round 33 — matched PickCube action codec, training, and paired dev (2026-10-28)

Ticket 05 has no local issue file in this checkout; this is the execution record
for its next bounded optimization slice. The retained data was not collected,
replayed, copied, or rewritten. Training used exactly the frozen direct IDs
0–999 and obstacle IDs 1000–1499 from the dataset above, with the same 30
development scenes as the retained PickCube 90k exposure experiment. PlaceSphere
data, weights, and configs were not used or changed. ManiSkill source was not
modified.

### Frozen action representation and matched training

The before-training profile is `pickcube_controller_range_v1`:

```json
{"name":"pickcube_controller_range_v1","version":1,
 "scale":[0.35,0.35,0.35,0.35,0.35,0.35,1.0],"offset":[0,0,0,0,0,0,0]}
```

Profile SHA-256: `f64e54ecd8eb26749fd4d0dbb18512ace929169d180334473d4e6e85a7053163`.
It maps the six stored native controller channels into `[-1,1]` and keeps the
native gripper channel unchanged. These are dimensionless normalized controller
coordinates, not meters/radians. The audit recomputed all source actions from
HDF5 before registration: both classes reach `[-0.349999994,+0.349999994]` in
each of the first six channels and `[-1,+1]` for grip; their encoded ranges are
`[-1,+1]`. Direct/obstacle episode RMS quantiles and the recomputed values are
frozen in each preregistration. Every obstacle terminal grip is `-1` (CLOSED).
Padding remains six zero pose values plus the final CLOSED grip; the PlaceSphere
OPEN-tail convention was not imported. Raw demonstration arrays were encoded
transiently when windows were read; no encoded HDF5 was made. Prediction decoding
is multiplication only, without clipping, and rejects any decoded native action
outside the existing `[-1,1]` controller bound.

The initial Base preregistration is
`outputs/exp/20261028_111705_pickcube_fullscene500_action_codec/preregistration.json`
(SHA-256 `35da4d1045cb138216294129d2fe514ca91f5f47bc7f5f7c569af5baf9907b1a`,
source identity `4c14662c929dd96c46d46ef1a0b61655a86fd29d840860056afd0e502b214751`).
Base trained from scratch on direct-only IDs 0–999 for 30,000 updates, seed 1,
batch 1,024, AdamW `1e-4`, warmup 500, cosine schedule, EMA power 0.75. It used
the existing 42-D state input, 7-D encoded action, `2/16/8` horizons and the
100-step squared-cosine DDPM epsilon objective. Final EMA was selected before
training. It completed in 867.82 s with 30,000 finite losses (first
`1.14611423`, last `0.00446880`, mean final 100 `0.00472909`); 83,004 windows
were sampled. Checkpoint:
`outputs/ckpt/20261028_111705_pickcube_fullscene500_base/final.pt`, SHA-256
`2bd91abc5484559e3d7406b27c3e75e12d307e4df206d2361ed956a46533e1c2`.

The matched control preregistration is
`outputs/exp/20261028_113532_pickcube_fullscene500_action_codec_retry1/preregistration.json`
(SHA-256 `eae50f326255fda93f09ebecc4a2c521b0bda37640b641ab42fde20cf77a536c`,
source identity `40cc412da7821cb3d46d8ccb16d03e487e33d7f676503ad86367ac8788e66707`).
It binds the exact Base EMA hash above, the same codec/profile hash, obstacle
IDs 1000–1499, HDF5/manifest hashes, and all source paths. The first Control
attempt under the earlier 111705 registration stopped before update 1 because
`NAMED_FIELDS` was missing from the config writer; it left only empty attempt
directories and is retained at
`outputs/exp/20261028_111705_pickcube_fullscene500_action_codec/train_constrain/`
and `outputs/ckpt/20261028_111705_pickcube_fullscene500_constrain/`. The import
was fixed and tested, and the corrected source/Base identities were frozen in
the retry registration before the next training command. No checkpoint was
written in the failed attempt.

Constrain trained only the existing control branch from that Base EMA, on the
500 obstacle episodes, for 150,000 updates (same seed/batch/optimizer/schedule,
unweighted epsilon MSE); 45,840 windows were sampled. It completed in 4,288.78 s
with 150,000 finite losses (first `0.11801428`, last `0.00433297`, mean final
100 `0.00454327`). The checkpoint's metadata records the actual 150,000 update
tuple and selected final EMA. Frozen Base state hash remained
`14b37834c0148ef02261c440a12b771e78e1f23c26eb5c27367aace1962d6cf3`, matching
an independent load of the matched Base EMA; all 100 reconstructed trainable
control tensors differ from their seed-1 initialized values. Before training,
zero-control epsilon and 50-step DDIM outputs matched the Base exactly (`0`
max absolute difference); after training, disabled-control epsilon/DDIM still
matched exactly (`0`). Checkpoint:
`outputs/ckpt/20261028_113532_pickcube_fullscene500_constrain/final.pt`, SHA-256
`e66c5ac700998648ec293ee29c2ca664fc00b1ad6cdfc2514140ac8533d236b3`.

### Paired 30-scene native evaluation

The final before-scene preregistration is
`outputs/exp/20261028_125137_pickcube_fullscene500_action_codec_paired_r2/preregistration.json`
(SHA-256 `8cd2821fc6c256305ba52f464f70408d6659bf829bdb497f2394a707839fc2a9`,
current evaluation-source identity
`b7494c1fdd1b4db700f0ba7329ccd9491bcf8a6714f4cc2204f0de0315819dbd`). It pins
the Base/Constrain hashes, training preregistration, retained old paired-record
tree, 30-scene manifest/HDF5, current ManiSkill PickCube source hash, all 1,500
training IDs and the 500-step/18,000-second evaluation budget. Two earlier
evaluation preregistration drafts were not run on scenes: 124911 preceded the
explicit decoded-native-bound guard; 125034 exposed a manifest-format reader
assumption before environment initialization. The corrected evaluator was
focused-tested and preregistered as 125137 before any native evaluation.

Both policies used a single 16-action DDIM plan per call (50 inference steps,
eta 0, stride 2), executed actions `[1:9]` two at a time, and refreshed the
latest-two-frame history. Base received the same 42-D state condition used in
training; Constrain received 42-D state plus the existing 3-value normalized
obstacle hint from the 45-D observation. Actions were decoded before native
execution. The existing native held-success condition was used with a 500-step
cap; there was no added release requirement, safety shield, grasp override, or
hand-coded pose/action. Each requested policy/scene episode is retained, with
native action trace, first contact, grasp/lift/transport/success steps, actual
terminated/truncated flags, and state/controller/observation restore hashes.
All 30 scenes ran for both policies, with zero runtime errors. The paired Base
and Constrain initial native state, controller state, and observation hashes
match scene-by-scene. Record span from Base episode 000 through the paired
summary is approximately 20m35s; the per-episode cap was 300 seconds.

| Policy | Task success | Safe success | Obstacle contact | XY crossing | Runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| New encoded Base DP | 14/30 (46.7%) | 0/30 | 30/30 (100%) | 19/30 (63.3%) | 0/30 |
| New encoded Constrain | 17/30 (56.7%) | 9/30 (30.0%) | 21/30 (70.0%) | 4/30 (13.3%) | 0/30 |

Within this new pair (Base → Constrain), task success transitions were both 8,
Constrain-only/improved 9, Base-only/regressed 6, neither 7. Safe success was
both 0, improved 9, regressed 0, neither 21. Contact was both 21, cleared 9,
introduced 0, neither 0. XY crossing was both 3, cleared 16, introduced 1,
neither 10.

The preregistered PRIMARY engineering comparator is the retained old PickCube
90,000-update Constrain run at
`outputs/exp/20261025_065159_pickcube_exposure90000/paired/`, with exact old
Constrain SHA `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`
and Base SHA `dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d`.
Old Base was 10/30 task, 0 safe, 30 contact, 15 XY; old Constrain was 27/30
task, 15 safe, 15 contact, 3 XY. Current Base transitions vs old Base:
task both 5 / improved 9 / regressed 5 / neither 11; safe neither 30; contact
both 30; XY both 8 / cleared 7 / introduced 11 / neither 4. Current Constrain
transitions vs old Constrain: task both 16 / improved 1 / regressed 11 / neither
2; safe both 5 / improved 4 / regressed 10 / neither 11; contact both 11 /
cleared 4 / introduced 10 / neither 5; XY both 1 / cleared 2 / introduced 3 /
neither 24. This is an overall engineering-pipeline comparison, not an isolated
codec/data/inference effect: the 500-obstacle data, action codec, 150k control
training, and DDIM stride-2 actor all differ from the old 300-obstacle/90k,
native-coordinate/DDPM-100 inference baseline. Repeated use of the frozen dev30
is engineering feedback, not independent generalization evidence.

The registered performance gate is **pause / Ticket 07 remains blocked**:
Constrain misses all four thresholds (task 56.7% <90%, safe 30.0% <85%, contact
70.0% >10%, XY 13.3% >10%). Against the current Base it improves task by 3,
safe by 9, contact by 9 fewer, and XY by 15 fewer; against the retained old
Constrain it regresses task by 10, safe by 6, contact by 6, and XY by 1. First
contact phases for the new Constrain were transport 17, lift 2, grasp approach
1, goal stabilization 1 (among 21 contacts); most remaining failures still
involve transport-phase contact. No additional independent trial was run.

Authoritative outputs are in
`outputs/exp/20261028_125137_pickcube_fullscene500_action_codec_paired_r2/paired/`:
`paired_summary.json` SHA-256
`56d1c1a2682aafdd440ebfa537e82926ea38e2e1c7fd5805fe8f0f6d0e50f4e7`,
`engineering_comparison.json` SHA-256
`5fbac5093b210149c0e05bc6b10aef899bddcf2857a52be767b9159acd81c887`, and
`gate.json` SHA-256
`0ca9b9f10f33a28830502aa90ee1a18d6f2d64ad5306b0f82c31ac6320d265ff`. The
historical paired directory was copied and its tree hash verified before
comparison; `pairing_failure_scene_indices` is empty. Final HDF5, manifest,
validation hashes remain unchanged at `0c74b88e…c349349f4f`,
`6b9d2c75…76e5c486`, and `f4756f69…d46d5b8f`. No new demos or layouts were
created. No evaluation was run for PlaceSphere.

Verification: complete constrain suite before the final eval-only corrections
passed 549 tests / 28 warnings in 529.60 s. After those bounded corrections,
codec, PickCube pilot, and paired-evaluator focused tests passed 56/56; both
edited Python modules compile and `git diff --check` passes. CPU strict-load
checks rejected an old native-action Base, incorrect codec identity and wrong
step count; both new actor adapters returned `(1,8,7)` finite native actions.
The final native-trace audit independently recomputed all policy counts, bounds,
success/contact/crossing flags and initial-state pairing from the 60 episode
records. No worker remained after the runs.

## Round 31 — PickCube existing-pool audit (2026-10-28)

Read-only audit at `outputs/exp/20261028_110000_pickcube_existing_pool_audit/`
using `examples/baselines/constrain/audit_pickcube_existing_pool.py`. It did
not initialize an environment, sample a layout, or replay an action. The
inventory records the exact source-file hashes, every packaged episode ID,
task identities (same seed or componentwise start/goal within 0.02 m), per-episode T/action/observation shape and
termination counts, aggregate seven-channel action ranges/RMS, and whether the
previously recorded source catalogs still exist. Reproduction (after activating
`maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 300s \
  python -m examples.baselines.constrain.audit_pickcube_existing_pool \
  --output outputs/exp/20261028_110000_pickcube_existing_pool_audit/inventory.json
```

| Existing source | Packaged counts | Obstacle layout findings |
| --- | ---: | --- |
| `demos/PickCube-transport-detour/20261024_104348_expansion/` | 1,000 direct / 100 obstacle | 100 task-unique; all 100 also occur in the 300 package |
| `demos/PickCube-transport-detour/20261024_213325_obstacle300/` | 1,000 direct / 300 obstacle | 300 task-unique; source-qualified cohort remains unchanged |

The expansion package HDF5 SHA-256 remains
`2af959e3d3ebeba8bb046db6d9583d300e1533fc638c7d36eb6393dd28024dff`; its
manifest SHA-256 is
`26cb2b3f8c371315805cb6219e0a4314dce3461bac3aaa872d35ce2e5ca7fd45`. The
300 package HDF5 SHA-256 remains
`3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`; its
manifest SHA-256 is
`3be137138176a4c243684c081ceda62f08c5792a3f3778a07460a3c2fa2642f5`.
The retained source catalogs contain 900 expansion direct-scene records, 64
expansion pair-scene records, and 127 + 200 obstacle-increment records. Their
catalog manifests and HDF5 identities are pinned in `inventory.json`. The
increment catalog manifests prove 103 geometrically unique candidates remain
after the old 300 obstacle cohort, frozen dev30, Ticket 03, and earlier
source-catalog identities are excluded under the seed-or-componentwise-2-cm
rule. The 900-record direct-scene source contributes 103 of these; the other
catalogs contribute none. Each is marked native-safe in its original scene
metadata with the fixed 0.03 m radius and full seven-channel `expert_actions`,
but catalog entries preserve only the initial native state (not T+1 native
states/controller/info records), so they require bounded original-layout
replay before packaging. With at most 103 additional candidates, the frozen
pool cannot supply the required 200; the maximum possible cohort is 403 and
the remaining gap is 97. No replay or package materialization was started,
because it cannot meet the requested 500 count. No new layouts were sampled,
and no existing data/checkpoint/configuration was changed. Ticket 05 has no
local `.scratch` issue file in this checkout, so this execution note is the
current decision record pending parent direction.

Catalog-aware audit JSON SHA-256:
`d5fd75829c14b266a8e0d1664eaba609a5b374671cb19255474f8fe4eb57b27d`.
External layout exclusions were pinned before candidate counting:
frozen dev30 manifest
`ffb2b54b6bf4f389d8bd81b0d5f282d833a297c6375ae2189c44b8359a9ee676`
(30 scenes) and Ticket 03 manifest
`82dc69a6f0e8581e96c25e677a9f812bc20e92ea757295b9b336a61891b2c0b1`
(8 scenes). The ordered 103 candidate IDs and each catalog manifest/HDF5 hash
are enumerated in the audit report. They were not preregistered for materializing
because this finite pool is short of the 200-episode increment prerequisite.
All retained actions are finite 7-D; raw observations are 45-D with T+1
rows. In the unchanged 300 obstacle cohort, lengths are min/median/p90/p95/max
`67/90/114.1/119/125`; each of the first six action channels reaches both
±0.35 and the gripper reaches ±1. The 1,000 direct class has 337 task-unique
identities by the same seed-or-componentwise-2-cm rule; these repeats are not
misrepresented as unique. Per-channel min/max and episode-RMS distributions for
both source classes, plus every individual length/shape/finite result, are in
the audit JSON. These existing action values do not authorize action
normalization or retraining in this round.

The audit’s 2-cm rounded key is an inventory/dedup diagnostic, not a substitute
for the original source acceptance contract. The original 300-package
qualification and array identity remain authoritative. The direct 1,000 class
is not reselected or represented as 1,000 unique task layouts. A future learning
slice must use the existing frozen package unless a compliant path to the 200
missing obstacle trajectories is explicitly resolved without generating new
layouts.

Focused audit tests in `tests/test_audit_pickcube_existing_pool.py` passed 4/4.
The complete constrain suite passed 541 tests (28 warnings) in 83.60 s;
`git diff --check` passed. No audit/replay worker processes remained after
verification.

The 2026-10-23 cleanup retains one final valid paired experiment per task,
the six selected Base DP / ConstrainNet weights, and the formal data, training
records, fixed scenes and verification needed to reproduce those experiments.
Safety success is the primary comparison metric; the paired Base DP comparator
is retained even where its safety result is worse. No checkpoint or scene was
reselected by looking at individual evaluation outcomes.

Pilot/calibration/probe/interface-smoke artifacts, failed or superseded runs,
the failed PlaceSphere ConstrainNet checkpoint and the periodic PickCube Base DP
checkpoint were deleted. The cleanup removed **992 files / 450,850,395 bytes**
(about 430 MiB); **765 retained file identities** were verified unchanged.

Cleanup record:
`outputs/exp/20261023_105159_final_artifact_cleanup/cleanup_manifest.json`.
It records all retained and deleted file paths, sizes and SHA-256 hashes, and
each completed deletion. `previous_OUTPUTS.md` in that directory preserves the
previous full output index. Historical manifests/reports were not rewritten:
references to removed diagnostics are historical, and resolve to deletion
records rather than current files. The earlier completion audit describes the
pre-cleanup acceptance evidence, including pilots that have now been removed.
`final_verification.json` in the cleanup directory records a second retained-hash
check and an independent recomputation of all six policy/task metric rows from
the 600 retained episode records; all match their authoritative summaries.

## Selected weights — exactly six files

| Task | Model | Current path |
| --- | --- | --- |
| StackCube (`panda_wristcam` evaluation) | Base DP | `outputs/ckpt/20260930_base_dp_stackcube_980/best_eval_success_at_end.pt` |
| StackCube | ConstrainNet | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/final.pt` |
| PickCube | Base DP | `outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt` |
| PickCube | ConstrainNet | `outputs/ckpt/20261022_144835_pickcube_constrain/final.pt` |
| PlaceSphere | Base DP | `outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt` |
| PlaceSphere | ConstrainNet | `outputs/ckpt/20261022_201100_placesphere_constrain/final.pt` |

The weight files keep their original bytes and hashes. StackCube uses the
previously selected `ema_agent` / final EMA; the two new tasks use their
preselected final EMA after 30,000 updates per selected model.

StackCube's historical runs have date-level provenance, so their weight folders
use `YYYYMMDD_` prefixes (2026-09-30 and 2026-10-21) without inventing a start
time. The 2026-10-23 renames preserve both checkpoint hashes. Their exact
old/new file mappings and hashes are in
`outputs/exp/20261023_105159_final_artifact_cleanup/path_migrations.json`;
the cleanup verifier uses this sidecar to resolve the original retained paths.

## Final data and validation

| Task | Retained formal data | Retained validation records |
| --- | --- | --- |
| StackCube | `demos/StackCube-obstacle/` (100 obstacle demos); Base DP source `demos/StackCube-v1/motionplanning/trajectory.filtered.state.pd_ee_delta_pos.physx_cpu.h5` (980 demos) | Dataset `validation.json`, training and scene audits below |
| PickCube | `demos/PickCube-obstacle/20261022_132753_formal/` (1,000 direct + 100 obstacle) | `outputs/exp/20261022_132753_pickcube_formal/` |
| PlaceSphere | `demos/PlaceSphere-obstacle/20261022_184558_formal/` (1,000 direct + 100 obstacle) | `outputs/exp/20261022_184558_placesphere_formal/` |

The historical `demos/StackCube-v1/` source/replay archive is preserved under
the repository's original-data traceability rules. No retained data was moved
or reserialized. PickCube and PlaceSphere pilot/intermediate datasets are gone.

## Final experiment directories

All paths in this table are below `outputs/exp/`.

| Task | Base training | ConstrainNet training | Fixed scenes | Final paired evaluation |
| --- | --- | --- | --- | --- |
| StackCube | `20260930_base_dp_stackcube_980_seed1/` | `20261021_constrainnet_wristcam_seed1/` | `20261021_wristcam_scenes/` | `20261021_wristcam_paired_eval/` |
| PickCube | `20261022_144000_pickcube_base_dp/` | `20261022_144835_pickcube_constrain/` | `20261022_152157_pickcube_scenes/` | `20261022_162002_pickcube_paired_eval/` |
| PlaceSphere | `20261022_193900_placesphere_base_dp/` | `20261022_201100_placesphere_constrain/` | `20261022_205500_placesphere_scenes/` | `20261022_213500_placesphere_paired_eval/` |

Each paired experiment retains both policies' complete 100-episode records,
summaries, parsed configuration, identity/restore evidence and representative
videos. PickCube's `termination_correction.json` remains authoritative for its
18/17 historical 500-step-limit outcomes; original flags were not reconstructed.
PlaceSphere's real time-limit truncations are 3/13 at the 200-step limit.

Also retained:

- `outputs/exp/20261022_221641_multitask_obstacle_completion_audit/`: original
  12-ticket acceptance audit, whose pre-cleanup scope is unchanged.
- `outputs/exp/20261023_105159_final_artifact_cleanup/`: current pruning and
  retained-identity audit. This supersedes historical *retention* descriptions.
- `outputs/exp/20261023_122842_pickcube_contact_diagnosis/`: Ticket 01 contact
  diagnosis for 28 old PickCube scenes. It does not replace
  `20261022_162002_pickcube_paired_eval/`. The earlier incomplete run
  `20261023_122333_pickcube_contact_diagnosis/` is not the evidence set.
- `outputs/exp/20261023_141712_placesphere_failure_diagnosis/`: Ticket 02
  crossing and terminal diagnosis for 28 old PlaceSphere scenes. It does not
  replace `20261022_213500_placesphere_paired_eval/`.
- `outputs/exp/20261023_153133_pickcube_transport_detour/`: retained Ticket 03
  batch. It does not close the parent end-sweep, join, or source-identity gaps.
  Bytes are unchanged. Matching trajectories remain
  `demos/PickCube-transport-detour/20261023_153133/`.
- `outputs/exp/20261023_165956_pickcube_transport_detour/`: Ticket 03 acceptance
  batch, 8/22 attempts. It supersedes `20261023_153133` for those three gaps
  and does not replace `20261022_152157_pickcube_scenes/` or
  `20261022_162002_pickcube_paired_eval/`. Matching trajectories are
  `demos/PickCube-transport-detour/20261023_165956/`. Incomplete same-day runs
  `20261023_164921_pickcube_transport_detour/` and
  `20261023_165430_pickcube_transport_detour/` are not this evidence set.
- `outputs/exp/20261023_235744_pickcube_learning_pilot_r2/`: Ticket 05
  learnability pilot, round 2 of 2. Gate is **pause**. It does not replace
  `20261022_162002_pickcube_paired_eval/` and does not admit Ticket 07.
  Round 1 `outputs/exp/20261023_183243_pickcube_learning_pilot/` is retained
  and was not used for training. Weights:
  `outputs/ckpt/20261023_235744_pickcube_pilot_base/final.pt`,
  `outputs/ckpt/20261023_235744_pickcube_pilot_constrain/final.pt`.
   Training demos: `demos/PickCube-transport-detour/20261023_235744_pilot_r2/`.
   These are not formal performance results.
- `outputs/exp/20261024_104348_pickcube_data_expansion/`: Ticket 05 authorized
  data-expansion round on the same frozen 30 dev scenes. Gate is **pause**.
  It does not replace `20261022_162002_pickcube_paired_eval/` or the round-2
  pilot, and does not admit Ticket 07. Weights:
  `outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt`,
  `outputs/ckpt/20261024_104348_pickcube_expansion_constrain/final.pt`.
   Training demos: `demos/PickCube-transport-detour/20261024_104348_expansion/`
   (1,000 direct + 100 obstacle, including the original 20+20). Not a formal
   performance or generalization result.
- `outputs/exp/20261024_202858_pickcube_proximity_weight/`: Ticket 05
  authorized training-optimization slice on the same 1,000 direct + 100
  obstacle data and the same frozen 30 dev scenes. One control-only retrain
  from the expansion Base EMA, with a frozen TCP-obstacle proximity weight on
  the epsilon loss. Gate is **pause**. It does not replace the expansion run
  or admit Ticket 07. No new demos. Weight:
  `outputs/ckpt/20261024_202858_pickcube_proximity_weight_constrain/final.pt`
  `22d3624941645ecac5bab4bd8679a2e40f77498b531b40f5e1d3c4be79d2a037`.
   Same-scene result versus expansion Constrain is worse: task 22/26, safe
   11/13, contact 18/17, XY 4/3. Not a formal performance result.
 - `outputs/exp/20261024_213325_pickcube_obstacle300/`: Ticket 05 authorized
   100→300 obstacle increment on the same frozen 30 dev scenes. The existing
   1,000 direct arrays were copied, not re-collected, and Base was not
   retrained. Unweighted epsilon; the rejected proximity weight stayed off.
   Gate is **pause**. It does not replace the expansion run or admit Ticket 07.
   Weight: `outputs/ckpt/20261024_213325_pickcube_obstacle300_constrain/final.pt`
   `bd7757122b75ad0bcc9cc5bd88a4b5174d48dfc520d1c8459332eea157732801`.
   Training demos: `demos/PickCube-transport-detour/20261024_213325_obstacle300/`
   `3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`
   (1,000 copied direct, not claimed unique; 300 obstacle = retained 100 plus
   200 new unique scenes). Same-scene result versus expansion Constrain
   26/13/17/3 is 25/12/18/2 task/safe/contact/XY. Not a formal performance
   result. Preregistration
   `3d37d27ac2145733c091094a5dd5245e9a237294abaebef6c37c1d29e28060a9`.
    Gate `296088e5e0b7118952d7dc020ca603dcfe98eeda3a1b0dc7cba430c56828e71a`.
    Comparison `f20d5c182c3d7079ad50e2411857670d96c77e6d7ee3b6b0e6f1474b7ee5d92e`.
  - `outputs/exp/20261025_062737_pickcube_window_gap/`: Ticket 05 expert-window
    diagnosis of the frozen 100 and 300 EMAs. No collection and no training.
    Same sample seeds and diffusion timesteps. Index
    `diagnosis/index.json`
    `35d9525892d3a2fc36dc75b905f0675b09669a82195867a8ff4d7edca44f8ed8`.
    Preregistration
    `db3a3479632664205578ab8f6eba00ed4bc0053f751ecb5d96c31661f2019bfb`.
    It does not replace the 300 trial or admit Ticket 07.
  - `outputs/exp/20261025_065159_pickcube_exposure90000/`: Ticket 05 authorized
    scratch 90,000-update trial from the matched expansion Base EMA. The saved
    30,000-step cosine last learning rate was 0, so resume was not used.
    Unweighted epsilon; proximity stayed off. No new demos. Gate is **pause**.
    It does not replace the 100 or 300 runs or admit Ticket 07. Weight:
    `outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt`
    `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`.
    Same-scene result versus expansion Constrain 26/13/17/3 is 27/15/15/3
    task/safe/contact/XY. Not a formal performance result. Preregistration
    `4e1be76719c9c22388db372c464cb3f222a4d1665343ef15c6c975ecfc3cfeca`.
    Gate `4f93ef34f0e6670246c7b5fd46c90e60fe9f54c31ab54dda9545093e22c42081`.
    Comparison `a8f213c327731fa2688b533c396a96c20f8b41b3932e5985ec6b2f6d67aad0c5`.
   - `outputs/exp/20261025_081913_pickcube_closed_loop_diagnosis/`: Ticket 05
     closed-loop diagnosis of the 15 exposure contact episodes. No collection,
     no training, and no stride trial. It does not replace the 90k eval or
     admit Ticket 07. Checkpoint unchanged
     `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`.
     Preregistration
     `68e84320a807face3b1a1f25a3da6246137e656725330627399913eda95679b6`.
     Index `diagnosis/index.json`
     `46a00982c870c3af71b897f7dedd20285849088d6cdf2bfc307a4dfc6ccf30ae`.
     Gate remains pause. Staleness was not isolated (0/15 queued overshoot).
  - `outputs/exp/20261024_014459_placesphere_learning_pilot/`: Ticket 06
  learnability pilot, round 1 of 2. Gate is **pause**. It does not replace
  `20261022_213500_placesphere_paired_eval/` and does not admit Ticket 08.
  No second round was started. Weights:
  `outputs/ckpt/20261024_014459_placesphere_pilot_base/final.pt`,
  `outputs/ckpt/20261024_014459_placesphere_pilot_constrain/final.pt`.
  Training demos: `demos/PlaceSphere-transport-detour/20261024_014459_pilot_r1/`.
  These are not formal performance results.
- `outputs/exp/20261024_155244_placesphere_data_expansion/`: Ticket 06
  authorized data-expansion round on the same frozen 30 dev scenes. Gate is
  **pause**. It does not replace `20261022_213500_placesphere_paired_eval/` or
  the round-1 pilot, and does not admit Ticket 08. Weights:
  `outputs/ckpt/20261024_155244_placesphere_expansion_base/final.pt`
  `2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640`,
  `outputs/ckpt/20261024_155244_placesphere_expansion_constrain/final.pt`
  `36518c84c0aacaadaf33eb7e35161b0e869c0d18d62e9425337021a763bfa43c`.
  Training demos: `demos/PlaceSphere-transport-detour/20261024_155244_expansion/`
  `df01aec45a5fcf6f04b0f0e2cb91d4bdf58d238ce41163b40cf15d1388d253ed`
  (1,000 direct + 100 obstacle, including the original 20+20). The packaged
  1,000 is not 1,000 unique task layouts (greedy unique 425, conflicts 575).
  Not a formal performance or generalization result. Preregistration
  `270ecc8712aef405f94cfdf14c6ad5f7cc1e30f4a7faf9d1d696a0185da45100`.
  Gate `fb768c089f124780eda1a477fbe5db26544d161b7faf9822f06b3032aa854391`.
  Comparison `3864d18c94ad775b18dfd2e5a0093df7ccae1ae7f342be574f6e79ac7612fbb0`.
  Coverage-writer fix validation, not a rerun:
  `outputs/exp/20261024_185754_placesphere_coverage_writer_fix/`.
  It does not replace this experiment or change its bytes. Experiment source
  `63d17eb3889b92b3638a42ee84ae3ada765b3dcf9b013307f3e4bba0d222d844` stays the
  preregistration identity. New writer identity
  `c370f36b3c1bb1d3ceb87fe6db200c754c0cf934d7cbe91fac566c6850c1b655` is not
  retroactive. Validation
  `2a59194181b48870889492aae53424523b5eca99958ece386fc3e166d65aec61`.
- `outputs/exp/20261025_085101_placesphere_obstacle300_90k/`: Ticket 06
  authorized combined 100→300 obstacle and 90,000-update slice on the same
  frozen 30 dev scenes. Factors cannot be separated. Gate is **pause**. It
  does not replace the expansion run or admit Ticket 08. No automatic
  300→1000. Base was not retrained. Weight:
  `outputs/ckpt/20261025_085101_placesphere_obstacle300_90k_constrain/final.pt`
  `70cf5e6f00bfec8b7d0945f65805fba06bbddd95356778d73009bdc5fed2326b`.
  Training demos:
  `demos/PlaceSphere-transport-detour/20261025_085101_obstacle300/`
  `f9dd209868d800fbce41eae276112a23dffc9aec02e91196ca900d46fc42aa02`
  (1,000 copied direct + 100 copied obstacle, arrays equal to
  `df01aec45a5fcf6f04b0f0e2cb91d4bdf58d238ce41163b40cf15d1388d253ed`;
  200 new unique obstacle). Same-scene Constrain versus expansion 25/21/7/0
  is 27/21/6/1 task/safe/contact/XY. Not a formal performance result.
  Preregistration
  `a2708b53736ad21d4242d94afa371cab37ab8456d454f6aea525d04e12779382`.
  Gate `b640d91352b28dc503303c35c3a96085835a4c761e75727700d0cd51d839c9e1`.
   Comparison `e0bd371ae6c32dfe7331f1d44a918bebe8b3fc2fa9f4d3640271f8e26597bb3f`.
- `outputs/exp/20261025_114030_placesphere_release_diagnosis/`: Ticket 06
  indexed diagnosis of the frozen `70cf5e6f` EMA. No training and no new data.
  It does not replace the 300/90k eval. Index
  `10a2ec98844bf9afaa7ea28f1ccf48c8049d017d28b8ed6d90d54e17f55cab38`.
  Evidence `8e93d6c9886e7e69e86accfc12622cc3b20f7205d4ccc28d06d6c947fa1a6bec`.
  Window index is `window_index.jsonl`, 40104 entries.
- `outputs/exp/20261025_114648_placesphere_phase_balance/`: Ticket 06
  one phase-inverse-frequency trial on the unchanged 300 demos, from Base EMA
  `2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640`, 90000
  updates, final EMA. Gate is **pause**. It does not replace the 300/90k run
  or admit Ticket 08. Weight:
  `outputs/ckpt/20261025_114648_placesphere_phase_balance_constrain/final.pt`
  `3385414d139d587252dfca36fdd6dd4501d7f1ac7fdfcd220e95fa08ed3a58bf`.
  Same-scene Constrain versus `70cf5e6f` 27/21/6/1 is 27/21/7/0
  task/safe/contact/XY. Contact cleared/introduced/both/neither = 2/3/4/21.
  Not a formal performance result. Preregistration
  `9c1a630dd7f4e29c09463fe29e49652f1daf08298aa0eb90fc8542723032432b`.
  Gate `1f0dd11faa7893d7e8b55f388e753da988cffa43902e932cc8d4cbee7f6ca493`.
  Comparison `e2444277f5d755d0b4a230e0631ac31828c25ed4541f8c97d71ba0b518f36b75`.
- `outputs/exp/20261025_131040_placesphere_coverage/`: Ticket 06 offline
  coverage of the six frozen `70cf5e6f` contacts against the existing 300
  obstacle windows only. No new demos, no training, no dev labels in the
  reference set. Prior diagnosis traces had no full observation window, so
  the frozen 30 were replayed; contact steps matched 1/90, 3/146, 5/113,
  9/146, 19/92, 21/162. Train windows 40104 on episodes 1000..1299:
  approach 16899, transport 17952, held_near_bin 4953, release 300.
  Execution-state rule would collect; policy-window rule would not
  (5/6 vs 3/6 outside). Nearest expert actions are still closed-grip
  negative delta-z. Not a reward or formal result. Preregistration
  `c4d7df16feae6bc5d9bf208f040769b8a7c2b4df6f99b888a0945e7351335480`.
  Coverage `566542a0146eda49df58420cf0fec371e5215cb0e57dcdbaaff5d9e78f281089`.
  Pilot identity `f238ddc29af22eab6daee9adf9e74d3d2132b0d2e2694e23530fc3cfcc2043e3`.
  Labeler identity `c53a60af0f3f698ffc2e497a6acba202d94e01f3dcbe53866c5612d6e328a6ce`.
- `outputs/exp/20261025_131730_placesphere_recovery_probe/`: one train-scene
  feasibility probe, not a packaged demo. A 2 cm lateral offset at the
  closed-grip container approach still commands a closed descent to the same
  target. Collection of this expert is not a recovery dataset. Probe
   `49facecbf0e06a09f75e970830f4401d017cf6a26b39b491232d99410985397e`.
- `outputs/exp/20261025_134916_placesphere_matched_terminal/`: Ticket 06
  opt-in above-release feasibility. Not a training set and not collection-ready.
  Legacy `terminal_mode` default remains `legacy`. Selected margin 0.015 m on
  8/8 calibration obstacle runs after 0.005 m failed the vertex filter on 2/8.
  Confirmation obstacle 16/16 native-safe, 0 contact. Dev30 after the freeze,
  not used for selection or training: 30/30 native-safe, 0 contact.
  Perturbation denominator 64 compared on 8 calibration layouts: legacy
  contact 32, new contact 10, introduced 0, cleared 22. Pre-terminal ±1/±2 cm
  contact was 0/32 for both. The contact drop is only at nearest-approach
  midstates, where new still contacts 10/32. Direct suffix accepted 0/8 and
  0/16. `confirmation.json` `collection_ready: true` is obstacle-only and is
  overruled by `collection_decision.json`. No demos written. Old 300 and
  direct h5 files were not overwritten. Run identity
  `aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed`.
   The later flag fix is not that identity. Radius, native thresholds, and
   cylinder geometry were not changed. No training. That run's
   `confirmation.json` `collection_ready: true` is obstacle-only and is not
   rewritten. It is not this round's identity.
- `outputs/exp/20261026_100200_placesphere_matched_terminal_r19/`: Ticket 06
   round 19 role split. Not a training set. Collection is **not ready**.
   Source identity `75e474697e1960c655d548945d7423ad28c6087ee65281f20ad5620ab7e33bf3`.
   Previous runtime `aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed`
   was not rewritten. Margin 0.015 m selected on obstacle positives only
   (0.005 failed 2/8 `hand_below_cylinder_top`). Calibration role bundle 8/8.
   Confirmation admission 13/16, `collection_ready` false in both
    `confirmation.json` and `collection_decision.json`. Held-out dev30 26/30,
    not used for selection or training. Negative direct 16/16 confirmation
    transport-blocked and admits no demos. No full collection and no training.
- `outputs/exp/20261026_103421_placesphere_terminal_round20_phase_a/`: Ticket 06
  Round-20 Phase A measurement before the motion correction. Final trace
  identity `4cc1780646a8ffe6514fd911a393279f4e4d466808591ac10598d45d519692b2`;
  diagnosis SHA `4977e33094005879408f17d1890dcf0d0c49dfbbf2979afd48c35d2948eb80a4`.
  R19 full failures are angular: 085 was 0.001252 m/s linear and 1.87655 rad/s
  angular; dev 003 was 0.000353 m/s and 4.54203 rad/s; dev 019 was 0.000361
  m/s and 7.39852 rad/s. After 200 neutral steps, 003 remained at 2.16161
  rad/s and 019 at 4.48883 rad/s. 085 reached native at +69 steps. Actual-
  cylinder suffixes 046/123 reached their own first native endpoints; dev-006
  did so after five neutral steps. Dev-000 measured shared z 0.151770565 m
  against direct requirement 0.151873754 m, a 0.103189 mm shortfall (vertex
  gap is approximate, not a surface proof).
- `outputs/exp/20261026_110038_placesphere_round20_long_settle_probe/`: focused
  continuation from retained full R19 actions, diagnostic only. Dev-003 reached
  native after 361 additional neutral steps; dev-019 after 785. No thresholds or
  geometry changed.
- `outputs/exp/20261026_104648_placesphere_matched_terminal_r20/`: preserved
  earlier complete Round-20 validation, identity
  `b437c3f2d2bb426936459921de248c5e0bb32a459e4dcfaa03865e560135993b`.
  Calibration 8/8 and confirmation 16/16 passed, but dev30 was 27/30. This is
  a historical recovery point and was not rewritten.
- `outputs/exp/20261026_105656_placesphere_round20_dev_engineering_feedback/`:
  non-selecting repeated-dev feedback for predeclared 8/16-step holds on
  dev-000/003/019. It did not repair those cases and is not independent
  validation. Recovery files are in
  `outputs/exp/20261026_110010_placesphere_matched_terminal_r20_recovery/`.
- `outputs/exp/20261026_110408_placesphere_matched_terminal_r20b/`: final
  Round-20 validation identity
  `3edc929330f644c407b94f6a23244520ae96b639706b954258336f82c3441958`.
  Margin 0.015 m; selected 0-step open hold on calibration 8 after all 0/8/16
  candidates tied 8/8. Confirmation 16/16 matched positives and 16/16 blocked
  negatives. Frozen dev30 admitted 30/30 without selection/training use and
  had 30/30 blocked negatives. Direct/obstacle release z matched exactly.
  Expert neutral settle is capped at 800 steps (1,000 total expert actions) for
  the measured spin cases; policy evaluation remains 200. Perturbation scope:
  64/64 finite comparisons on calibration 8; new contact 10/64, native 32/64,
  crossing 0/64 versus legacy contact 32/64, native 30/64; cleared/introduced/
  both/neither contact transitions 22/0/10/32. Preterminal native 30/32→32/32;
  nearest-approach native 0/32 for both. This is a bounded
  calibration-layout scope, not a broad rate. No full collection or training.
- `demos/PlaceSphere-transport-detour/20261026_111952_round20_matched_tplus1/`
  and `outputs/exp/20261026_111952_placesphere_round20_tplus1/`: one matched
  direct/obstacle T+1 state/controller/action artifact after the gates. Both
  roles exactly restored initial state/controller and first-action state,
  controller, policy observation, terminated, and truncated values. HDF5 SHA
  `51cb2f0109898c7477d6d522251ea58086559e3dd11c921aedc40e332544dacc`; runner
  SHA `f3001d7fa8032d526453c1bf41ccefcb33d9fa39b57c551d7c28e673d162ed39`.
  This is a T+1 proof artifact, not the 1,000+300 collection and not used for
  training. The existing formal generator still uses the legacy terminal path;
  that was the state before the Round-21 opt-in integration below.
- Round 21 matched-terminal collection, source integration and exact commands:
  `outputs/exp/20261026_133000_placesphere_matched_collection_r21b/`.
  The final pre-collection registration is
  `preregistration.json` (`a1f6ff02fac852ea1d73bfce951047e17f6dc74b22a2f4f3643e4dd176ad0a29`),
  source identity `6a8091fc694e43fffe84fd50d7d60280f649d927986a143445f26d91a420b048`.
  It pins the collector and all experiment helpers, native PlaceSphere source
  hashes, environment package versions, field schemas, action/observation
  dimensions, source layout catalogs, task-identity dedup and finite budgets.
  ManiSkill source and historical datasets were not modified. Existing geometry
  challenge, 3 cm cylinder radius, native thresholds, legacy terminal default,
  and 200-step policy budget remain fixed.

  The bounded native pilot at `pilot_subset/` regenerated 80 DIRECT + 25
  OBSTACLE demos across 25 matched pair layouts from the frozen final-candidate
  lists. All 105 T+1 initial/first-transition restore audits passed. There was
  one rejected expert case, retained in the journal; no policy-performance
  filter was used. 6/80 DIRECT and 2/25 OBSTACLE episodes exceeded 200 actions;
  median/p90/p95/max lengths were 105/147/160/909 and 116/179/529/633. Actions
  beyond the 200-step horizon were 13.4% and 20.9% of each role's action volume.
  Open near-zero-motion actions were 19.8% DIRECT and 26.9% OBSTACLE. The
  preregistered native-static 8-action windows were 0/1,765; action-neutral
  fractions and phase counts are retained in `pilot_subset/report.json`. This
  does not imply policy convergence; all complete native-success tails remain
  in the data.

  The first full 1,000+300 attempt at
  `demos/PlaceSphere-transport-detour/20261026_123000_matched_terminal_r21/`
  is preserved as a nonqualifying diagnostic package: its mixed-cohort selector
  produced only 213 greedy-unique OBSTACLE task layouts (87 duplicates). Its
  trajectory SHA was `61d5d06543bb1e4db13a9083bb10910c65c0d0b4ae0dc532fa72f6747809efd7`;
  it is superseded below and must not be used for training.

  The corrected, qualifying candidate package is
  `demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/`.
  Primary OBSTACLE sources were frozen as the 100 matched pair-catalog layouts
  plus the 200 `catalog_full` obstacle layouts; all 300 pass the task-identity
  dedup rule. A separately frozen 212-layout task-unique replacement pool was
  available; 3 replacements were used after 3 positive OBSTACLE native-terminal
  failures. The DIRECT class contains 300 matched positive DIRECT partners plus
  700 DIRECT-only trajectories from the separate source pool. Actual source
  counts: DIRECT = 703 `direct_scenes/assembled`, 99 `pair_scenes/merged`, 198
  `catalog_full`; OBSTACLE = 3 `direct_scenes/assembled` replacements, 99
  `pair_scenes/merged`, 198 `catalog_full`. All source scene IDs, seeds, and
  role-to-episode mappings are in `collection_full/layout_id_mapping.json`.

  Collection completed in 3,165.2 s: 1,000 DIRECT + 300 OBSTACLE, 300 paired
  layouts, 1,005 layout attempts, 3 preregistered obstacle replacements, and 5
  rejected cases (3 positive obstacle terminal failures; 2 positive direct
  native-placement failures). No exception, runtime error, state-restore
  failure, timeout, or dev/Ticket04 layout conflict was admitted. All 300
  accepted OBSTACLE layouts are task-unique and all 300 have paired DIRECT,
  negative transport-block, and suffix-replay proofs. The DIRECT class has
  472 greedy-unique task layouts and 528 within-class task repeats; source
  identity repeats are not presented as new unique layouts. This is comparable
  in scale to the prior 425 unique / 575 repeat DIRECT package, with the changed
  paired source mix disclosed above. External conflicts against the frozen
  dev30 and Ticket 04 catalogs are 0.

  Independent saved-artifact audit passed all 1,300 episodes: 7-D legal native
  actions, named 39-D policy / 42-D raw observations, T actions and T+1 native
  state/controller/info/termination records, first native success, and 1,300/1,300
  exact initial + first-transition restore audits. Every obstacle prefix is
  contact/crossing-free from execution start. Radius/native thresholds and
  obstacle geometry are unchanged; vertex clearances remain approximations,
  with PhysX execution as the safety evidence. Full layout mapping and per-episode
  findings are in `collection_full/dataset_validation.json`.

  Across all 1,300 episodes, 89 exceeded 200 actions: 60/1,000 DIRECT and
  29/300 OBSTACLE. Role medians/p90/p95/max are DIRECT 105/147/321/909 and
  OBSTACLE 126/185/544/889. Above-200 actions account for 15.8% DIRECT and
  20.7% OBSTACLE action volume. Near-zero open actions account for 21.6% and
  25.5%; the strict 8-action native-static window count is 0/22,364 before the
  first native-success stopping point. Step-phase action totals (DIRECT /
  OBSTACLE) are grasp approach 42,100 / 12,562; grasp pose 5,000 / 1,500;
  gripper close 8,000 / 2,400; lift 8,028 / 2,404; transport 29,015 / 15,021;
  clearance approach 4,112 / 1,228; stabilize 2,001 / 600; release 1,000 / 300;
  retreat 6,101 / 1,807; settle 27,723 / 12,555. Phase labels refer to the
  expert's current executed phase, with no future-state phase inference. Full
  per-phase current-action labels, episode lengths and action-window records are
  in `report.json`; no success tail was truncated or removed. Do not infer that
  the unchanged 200-step policy budget solves these longer expert tails.

  Identity hashes: `trajectory.h5`
  `eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0`,
  `trajectory.json`
  `ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86`,
  `attempts.jsonl`
  `b13b796325af0a838a803bf169d9dfc0daade6a394a7f9ef264da318d47ff42a`,
  dataset report `9d8e4c9231e37017df17c916df8a00b8cfad06d502f51a86e9f67d077fa3e44d`,
  independent validation `da77e933fa6f887594800c187d169255724c65ef82957513f3db4d9a84bfe089`,
  dataset identity `f0b9dd6d5b966744f44608528aa4bbab89baf9ebeb5b01972d315ad5ad78150e`,
  and layout mapping `a519b542faf497c49d0342851343c67dc7edb19a37c881d84f747ecf65811825`.
  Complete preregistration, pilot and full-collection command logs are retained
  in the matching `outputs/exp/20261026_133000_placesphere_matched_collection_r21b/`
  directory. No training or policy evaluation was run. The dataset is ready for
  independent parent review. Its HDF5 IDs follow collection order and are
  interleaved DIRECT/OBSTACLE, so the legacy trusted-formal trainers (fixed to
  the historical 1,000/100 IDs and hashes) must not be pointed at this version.
  After review, the matched training slice should add an opt-in reader keyed by
  the new manifest's class IDs, preregister the 1,000/300 identities, then run
  Base DP for 30,000 updates on DIRECT and ConstrainNet for 90,000 updates from
  that matched Base EMA on OBSTACLE. Use separate `outputs/exp/` and
  `outputs/ckpt/` paths. This is a proposed next plan, not a training result or
  an executed command.
- `outputs/exp/20261023_175529_placesphere_transport_detour/`: Ticket 04
  acceptance batch, 4/8 attempts. It does not replace
  `20261022_205500_placesphere_scenes/` or
  `20261022_213500_placesphere_paired_eval/`. Matching trajectories are
  `demos/PlaceSphere-transport-detour/20261023_175529/`. Same-day incomplete
  runs `20261023_174606_placesphere_transport_detour/` and
  `20261023_175100_placesphere_transport_detour/`, and the smaller successful
  probe `20261023_175419_placesphere_transport_detour/`, are not this evidence
  set.

## Reports

- [Current three-task results](THREE_TASK_RESULTS.md)
- [StackCube paired results](WRISTCAM_EVALUATION.md)
- [PickCube paired results](PICKCUBE_EVALUATION.md)
- [PlaceSphere paired results](PLACE_SPHERE_EVALUATION.md)
- [PickCube / PlaceSphere collection and training summary](MULTITASK_OBSTACLE_RESULTS.md)

Older pilot and task reports remain source-controlled documentation. Their
statements that diagnostic artifacts were retained describe their original
publication state, not the post-cleanup filesystem.

## Historical relocation map

These mappings still resolve retained final artifacts without rewriting old
manifest/checkpoint paths. More specific checkpoint mappings take precedence.

| Historical path | Current path |
| --- | --- |
| `outputs/ckpt/base_dp_stackcube_980/` | `outputs/ckpt/20260930_base_dp_stackcube_980/` |
| `outputs/ckpt/constrainnet_wristcam_seed1/` | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/` |
| `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt` | `outputs/ckpt/20260930_base_dp_stackcube_980/best_eval_success_at_end.pt` |
| `outputs/runs/dp_StackCube-v1_980_1/` (training records) | `outputs/exp/20260930_base_dp_stackcube_980_seed1/` |
| `outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt` | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/final.pt` |
| `outputs/constrain_runs/wristcam_full30000_seed1/` (training records) | `outputs/exp/20261021_constrainnet_wristcam_seed1/` |
| `outputs/wristcam_formal_20261021/` or `outputs/data/wristcam_formal_20261021/` | `demos/StackCube-obstacle/` |
| `outputs/wristcam_eval_scenes_20261021/` | `outputs/exp/20261021_wristcam_scenes/` |
| `outputs/wristcam_paired_eval_20261021_recovery/` | `outputs/exp/20261021_wristcam_paired_eval/` |

Activate `maniskill` before any execution. New data must use an explicit path
under `demos/`; new weights belong in `outputs/ckpt/`, and experiment records in
  `outputs/exp/YYYYMMDD_HHMMSS_<name>/`. Do not overwrite the retained final runs.

## Round 22 — matched PlaceSphere DIRECT/OBSTACLE training and paired dev

This is the authorized matched-data optimization continuation of Ticket 06,
not formal evaluation or generalization evidence. The branch remains
`goal/transport-detour-optimization` at base `730b744495914b82f3c19415cfaf30f536779ba2`;
no checkout, branch change, commit, or push occurred. ManiSkill source and all
historical datasets/checkpoints were left unchanged. The earlier
`20261026_123000_matched_terminal_r21` dataset (213 unique obstacle layouts / 87
repeats) was excluded. The only training data were the qualifying
`20261026_133000_matched_terminal_r21b` package:

- HDF5 `eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0`;
  manifest `ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86`.
- Manifest-selected IDs are pinned in preregistration; the actual HDF5 order is
  interleaved (DIRECT IDs are even, OBSTACLE IDs odd). Counts are exactly 1,000
  DIRECT / 300 OBSTACLE, disjoint and covering 0–1299. The opt-in versioned
  reader cross-checks the manifest labels against HDF5 `kind`, rejects missing
  labels, wrong counts and unpinned identities, and verifies T+1 39/42 state,
  seven-channel native actions and successful terminal evidence. Historical
  formal readers remain locked to their original hashes and 1,000/100 IDs.
- Actual official training windows: 132,080 DIRECT / 50,077 OBSTACLE. The
  separate stride-8 diagnostic has 16,189 / 6,175 windows and is not the
  training-window denominator. Episode length median/p90/p95/max is
  105/147/321/909 DIRECT and 126/185/544/889 OBSTACLE. Actions beyond 200 are
  15.83% / 20.71% of action volume; open near-zero actions are 21.58% / 25.52%
  (28,723 / 12,855 actions). Since the tails are below 50% and the action-neutral
  share is independently substantial, the preregistered choice was uniform
   official DP window sampling: no phase weight, trajectory reweighting, or
   truncation. Expert tails do not prove policy sufficiency at 200 steps.

  A separate, non-selecting quality audit was calculated from each episode's
  saved current-action phase and per-step linear/angular speeds in the HDF5. It
  uses fixed non-overlapping 8-action windows and no future phase inference. The
  action-only open/near-zero predicate is all eight gripper actions open and
  max norm of the six pose channels ≤0.02; linear and angular native speed checks
  are counted independently (≤0.01 m/s and ≤0.5 rad/s):

  | Role | All stride-8 windows | Open/near-zero action-only windows | Linear static among action-only | Angular static among action-only | Linear-static / angular-only failure | Both native-static |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | DIRECT | 16,189 | 3,311 (20.45%) | 3,301/3,311 (99.70%) | 0/3,311 | 3,301/3,311 (99.70%) | 0/3,311 |
  | OBSTACLE | 6,175 | 1,513 (24.50%) | 1,506/1,513 (99.54%) | 0/1,513 | 1,506/1,513 (99.54%) | 0/1,513 |

  Reproduction command (read-only against the pinned HDF5):

  ```bash
  source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s \
    python -u -m examples.baselines.constrain.audit_placesphere_matched_windows \
    --data demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
    --output outputs/exp/20261026_153000_placesphere_matched_paired_eval/matched_window_quality_posthoc.json
  ```

  All action-only windows are current-phase `settle`; 10 DIRECT and 7 OBSTACLE
  action-only windows fail the linear static check. Full report:
  `outputs/exp/20261026_153000_placesphere_matched_paired_eval/matched_window_quality_posthoc.json`.
  SHA `b0afc25631f84c745ddf486d203dd009438b19c71e15772fc35b203d2c0a6f5c`.
  This analysis was completed after preregistered training/evaluation, so it did
  not affect sampling, checkpoint choice, or gate interpretation. It is a
  process deviation from the requested order to resolve this quality gap before
  training; these rates are post-hoc, not preregistered independent
  angular-window coverage. Expert angular settling remains a bottleneck for
  parent review.

The before-training registration is
`outputs/exp/20261026_153000_placesphere_matched_paired_eval/preregistration_matched_r3.json`
(SHA `d0b7b88d5514198a5cae44ec4aff41b5bb1968b3162f580ff2956bc8144f0e02`),
which pins the dataset IDs/hashes, actual source file bytes, final-EMA selection,
30-scene seeds, budgets, and output paths. Training-source identity
`7c9bec6b3310ac6e3f2715a11f63a4b889b433f79dbce7f80ebbf98d6d0d0f79` was
verified before the Base and 90k runs. The earlier r1/r2 registration files in
the same experiment folder are superseded preregistration drafts: r1 had not
yet recorded calculated DP window totals, and r2 was generated before a trainer
syntax cleanup. Neither was used for any training or evaluation command.
Before evaluation, the adverse contact/XY
transition labels in the summary writer were corrected (episode inference was
unchanged) and re-frozen in preregistration r4, SHA
`f1a49951d652cfb30fd40a33b867f0664d0cc19efd1039d95ff63cf6b9a53adf`, source
identity `8fab4e962a21c8b67de70bd5272e6994f85595ad29ee7cf0e4e8b71b97527a0e`.

Preflight loaded the new Base EMA strictly into the PlaceSphere control model:
epsilon and full 100-step disabled-control equivalence both had max absolute
difference 0.0. The 300 obstacle IDs yielded 50,077 windows. Two finite
preflight updates verified finite loss, movement in the zero-initialized
control path, nonzero zero-convolution gradients and an unchanged frozen base.
An initial overly strong preflight assertion expected gradients in all upstream
and hint modules before those zero convolutions moved; those gradients are
structurally zero at initialization. No training had started in those attempts;
the corrected two-update check passed. All attempts terminated cleanly.
Before training, `py_compile` also caught an indentation error in the optional
trainer snapshot list; it was fixed and compilation/focused tests passed before
the final training registration. The first reader sample assertion expected a
42-D policy observation; the actual contract is 39-D policy plus 42-D raw, and
the assertion was corrected before registration. One process-inspection shell
check found `rg` unavailable; it was rerun successfully with `ps` piped to
Python. None of these issues changed dataset bytes or launched an unregistered
training/evaluation run. No `.tmp` checkpoints or experiment processes remained.

Exact finite-budget commands (all after activating `maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 2400s \
  python -u -m examples.baselines.constrain.train_placesphere_base_dp \
  --demo-path demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.json \
  --run-dir outputs/exp/20261026_153000_placesphere_matched_base_dp \
  --checkpoint-dir outputs/ckpt/20261026_153000_placesphere_matched_base_dp \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 --matched-r21b

source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 6000s \
  python -u -m examples.baselines.constrain.placesphere_constrain \
  --demo-path demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.json \
  --base-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt \
  --run-dir outputs/exp/20261026_153000_placesphere_matched_constrain_90k \
  --checkpoint-dir outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k \
  --device cuda --total-iters 90000 --batch-size 1024 --seed 1 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 --matched-r21b
```

Base completed 30,000 updates in 886.36 s; losses were finite (first
1.16264045, last 0.00248973). Final EMA:
`outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt`, SHA
`4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942`.
Base run config SHA `6cf5c963f1bb475489b826c7515b93591522c4b8f54d5f78ca692461ff7e6aff`;
summary SHA `a5cede581b8cb747d942159eff4f675c8162399c34428da88ef200dcaef49eb6`.
Control completed 90,000 updates in 2,533.85 s; losses were finite (first
0.02094785, last 0.00165384), frozen Base unchanged, control parameters moved,
and control-down, mid and hint gradients were nonzero. Pre/post disabled-control
epsilon and sampling equivalence max difference was 0.0. Final EMA:
`outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt`, SHA
`4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88`.
Control run config SHA `ef7fe1925a735817c57263d8376759d301b703500749d93366dfbc53dd700be6`;
summary SHA `ed67aa7c98d3c11ce50714dd64fd8887e1dde6cec9ca3186784296d6a5a8d042`.

Paired evaluation reused the original frozen 30-scene development set from
`outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes` (the same
set used by the 2026-10-25 300/90k run), with identical restored states,
controllers, observations and per-scene seeds; max 200 control steps, 120 s per
episode, 8,000 s overall. The primary comparison is the old 300/90k EMA; the
old 100-obstacle EMA is secondary. New evaluation command:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 8000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt \
  --constrain-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt \
  --output-dir outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval \
  --primary-baseline outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired \
  --secondary-baseline outputs/exp/20261024_155244_placesphere_data_expansion/paired \
  --device cuda
```

Both new policies evaluated all 30/30 episodes, zero runtime errors. Base was
6/30 task, 0/30 safe, 30/30 contact, 24/30 XY; old 300/90k Base was
5/30, 0/30, 30/30, 24/30. New Constrain was 26/30 task, 25/30 safe, 1/30
contact and 0/30 XY. Against the old 300/90k Constrain
27/21/6/1: task improved/regressed/both/neither = 2/3/24/1; safe = 7/3/18/2;
contact cleared/introduced/both/neither = 6/1/0/23; XY = 1/0/0/29. Against
the old 100-obstacle Constrain 25/21/7/0: task = 5/4/21/0; safe = 8/4/17/1;
contact = 6/0/1/23; XY = 0/0/0/30. New Base versus old 300/90k Base task
transitions = 4/3/2/21; safe = 0/0/0/30; contact = 0/0/30/0; XY = 1/1/23/5.

The frozen gate is **pause / fail**: required task ≥27/30 and safe ≥26/30,
contact ≤3/30 and XY ≤3/30; observed task 26 and safe 25 fail, while contact
and XY pass. These are development results, not formal/generalization claims.
Compared with the old 300/90k run, this matched dataset plus layout-replacement
package improves safety/contact outcomes but does not isolate a pure expert
causal effect. Do not proceed to Ticket 08 or PickCube formal work from this
result without parent review.

Episode-level evidence is retained in
`paired_eval/{base_dp,constrain_dp}/episode_*.json`; derived summaries are in
`paired_summary.json` (SHA `4ea9288fa5fb6506d92b5fb0d753b21a4747857c1a92a9ac2b517a53dbcf67b7`)
and `episode_diagnostics.json` (SHA
`eb241ecd060233a743d3c8607b40bed97a8a17eed06ab12a81c4235fc9081537`). New
Constrain had one first contact: grasp-approach, Panda left finger/gripper on
the top cap. It ended ungrasped; 30 releases were recorded, with 2
`incorrect_release` and 2 `not_stabilized` failures. Successful native completion
steps: 26/30, median 107, p90 160, max 165; all successes occurred within the
200-step policy budget. Base first contact occurred 30/30 (29 transport, one
release/stabilization); Base had 24 incorrect-release failures. Terminal speed,
grasp, contact entity/phase/location and release records remain per-episode.
Frozen evaluation config SHA
`edc7e458d9caad81bb7e73d6d0bf638ac7442c659a8225960295b7c5f1bc81f6`. Base and
Constrain episode-directory identity hashes are respectively
`f2d668093cdb60cdd0420feb0ad7c8f1cc2f558b9dac67fa6c27c8e21c4fe896` and
`877cd6ad6c5faca05da52cea50970227a1f06ddaee4bc7d66c10c40dc6b8b2da`.

Full constrain test suite after the post-hoc window-audit check:
`timeout --signal=TERM --kill-after=20s 700s python -m pytest
examples/baselines/constrain/tests -q --tb=line` — **484 passed, 28 warnings,
83.23 s**. `git diff --check` passed. GPU training/evaluation processes
were gone at verification; the RTX 5070 returned to idle. No runtime checkpoint
or evaluation errors occurred. Files changed are limited to
`examples/baselines/constrain/` plus this index and Ticket 06 execution notes.

## Round 23 — measured PlaceSphere hold-8 continuation

User-authorized continuous Ticket 06 optimization on
`goal/transport-detour-optimization`, starting HEAD
`a402e6c6b01bd2622468e8f827fb13a076795d2a`. No checkout, commit, or push.
ManiSkill source and all retained Round-22 artifacts were unchanged. This is a
development experiment, not formal evaluation or generalization evidence.

### Current matched-checkpoint diagnosis

`placesphere_matched_diagnosis.py` is pinned to the Round-22 Base EMA
`4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942`, Constrain
EMA `4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88`, and the
unchanged 30-scene paired evaluation manifest. It traces indices 12/13/21/22/25/28
and safe Constrain controls 0/2/4/6 for both policies, with full raw observations,
queued seven-D actions, TCP/sphere/bin/cylinder fields, action queue/replan
indices, grasp/release/contact/crossing transitions, and post-action linear and
angular velocity vectors. Controller/action slice `[1,9)`, horizons 2/16/8,
scene-seed inference, 200-step horizon, and native limits (0.01 m/s, 0.5 rad/s)
are unchanged. All 20/20 traces per policy reproduced the saved success/safety/
contact/crossing/termination/step outcomes across 10 selected scenes per policy;
0 mismatches and 0 runtime errors.

| Constrain dev index | Result | First contact / release | Terminal linear / angular speed |
| ---: | --- | --- | ---: |
| 12 | native task success, unsafe | grasp-approach contact step 4, queue 3; left finger/top cap | 0.008181 m/s / 0.470576 rad/s |
| 13 | release/stability timeout | release step 86, queue 5 | 0.025526 m/s / 1.276365 rad/s |
| 21 | release/stability timeout | release step 169, queue 0 | 0.581996 m/s / 29.099939 rad/s |
| 22 | release/stability timeout | release step 114, queue 1 | 0.000720 m/s / 4.987443 rad/s |
| 25 | safe native success | release step 111, queue 6 | 0.002481 m/s / 0.126532 rad/s |
| 28 | release/stability timeout | release step 99, queue 2 | 0.001383 m/s / 5.122324 rad/s |

The sole contact is early grasp approach, not release. Four current 200-step
failures are angular/stability failures, including two with sub-threshold
terminal linear speed. That did not isolate queued-action staleness as the shared
cause. Full corrected diagnostic:
`outputs/exp/20261026_210000_placesphere_matched_diagnosis_r23d/`; diagnosis SHA
`d3780c27bd156af222527e7e0a63b9d34d5807e47b747e92a017c2e4b840711e`, preregistration
SHA `9f66f2827b4db02f0defc45cb9e460d76aaee2c790f303dde22a204fc48f11f8`.
The full T+1 expert velocity/action join SHA is
`956e5a5ad0b00018a1c37294709536d7969631d98288cea48b51e3d15569fa4e`. It aligns
each action to its *post-action* native state and checks against recorded
per-action velocity norms. Among the 300 matched OBSTACLE training releases,
linear-speed median/p90/max was 0.368188/0.373907/0.375557 m/s (300/300 above
native linear limit); angular-speed median/p90/max was 0.403734/0.438300/0.501188
rad/s (1/300 above angular limit). Of the 12,555 post-release settle
transitions, 12,457 exceeded the native angular limit and 138 exceeded the
linear limit; retained expert tails do eventually reach their first native
endpoints.

### Physics selection, data regeneration, and training

Before any dev-set result was inspected, a physics-only replay compared 0/8/16
open-neutral holds after the actual release transition on frozen training
OBSTACLE IDs 1/3/5/7/9/11/13/15. Each candidate reused the same full initial
environment/controller state and native action prefix, inserting only the
registered hold. Safe native success was 6/8, 7/8, 7/8; contact and crossing were
0/8 for all candidates; terminal speed-threshold passes were 6/8, 7/8, 7/8.
The frozen tie-break selected the shorter 8-step hold. Probe result
`outputs/exp/20261026_181000_placesphere_matched_hold_probe_r23f/result.json`
SHA `eb45994e780b7553b79ec966fb6ccf37f1628eb7ee4c6a78830d313f0b29b769`;
preregistration SHA
`adb7dfdb2bc4ef8f3c65ac378d3e2ab0d4da90c1c650f701df19368e37fadbbf`.

The full collector registration
`outputs/exp/20261026_183000_placesphere_hold8_registration_r23.json` SHA
`0ae6445a848cf5a046b1602fdebcca1364233f766a42a70a5a1aa73574e69af6` freezes
hold 8, source-catalog identity, direct/obstacle role rules, task-layout dedup,
and finite serial budgets. A 20+20 paired native collector pilot completed in
73 s and validated 20 unique paired obstacle layouts and 40/40 restore audits;
it is retained separately at `demos/PlaceSphere-transport-detour/20261026_183500_hold8_pilot_r23/`
and excluded from full training.

The regenerated full data is
`demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23/`:
1,000 DIRECT + 300 OBSTACLE, 1,004 layout attempts, 2 predeclared replacements,
4 rejected layouts, 3,104.39 s, 300 unique matched obstacle layouts, and 474/1,000
greedy-unique direct layouts (526 repeats, not claimed unique coverage). No old
trajectory arrays were copied. Independent validation passed 1,300/1,300 exact
T+1 restore audits, native seven-D actions, 39/42 state observation schema,
first-native-success evidence, full-prefix contact/crossing checks, and paired
negative/replay proofs. Dataset identity: HDF5 SHA
`2bd725179ed61c8c32b97cdc9b89ce348add0bbf6b4d7a912cfa8bad30b9a5a1`, manifest
SHA `8685d2303cb269c5e9ccda9964586c25406e40afdcd83c49ad6fd3adbb24fbad`,
training identity SHA `76fd854fcea553bd020f8e22ea25ab078552c903afe5bafd0bf3125ad61af70a`.
Independent validation SHA
`c7508ec2709252c33ad534df24be390a902c922baf8fde622dc90a6a11e607a1`;
collection report SHA `aa292bcaec9a0671a3df3074ddab29ec4cbb493b610d8d5a0349160ed5c0c08f`.

The before-training trial registration is
`outputs/exp/20261026_195500_placesphere_hold8_training_prereg_r23b.json` SHA
`32057e428d24e301b78e485a8cbfd1c8ddec56cd7c6d3962d80dab314e27ac84`; registered
source identity `8baa3958d6341d6e5e632b74fd8150297dc3fae9bedeb703c96f43ad6eafa059`
was rechecked with no source drift before training. It pins uniform official
window sampling, new matching data identity, seed 1, Base 30,000 updates,
Constrain 90,000 updates from the new Base final EMA, final-EMA weight selection,
the frozen 30 dev scenes, 200 policy steps, 120 s per episode, 8,000 s overall,
output paths, and original gate thresholds.

Base: 30,000 updates in 881.56 s, all losses finite (1.159990 → 0.002054), final
EMA SHA `92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`.
Run config SHA `bfcde1a99d908b29cdb569eb3af401ae33119aca8f96b3db100fbec449ffbaee`;
summary SHA `ddab69156d38f80e2f524497659470d2ecec56d6ca36878d1774ac1831cda35c`.
Constrain: 90,000 updates in 2,518.40 s, all losses finite (0.015319 →
0.001245), frozen Base unchanged, control branch moved with nonzero gradients,
pre/post disabled-control epsilon and 100-step sampler equivalence exact.
Final EMA SHA `c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada`;
run config SHA `7bbe0dc6c4e89e9f1f5b5df332a083d6eda19a0cc398c041f2a938dce8385d2f`;
summary SHA `8fab4eead8d240205a6db64b3f9395b2899d115453eb47a82d89784aae12303d`.

### One paired dev trial and gate

The one registered evaluation used the exact Round-22 frozen 30-scene manifest,
200 steps, both policies, and zero runtime errors (30/30 each). Hold-8 Constrain
scored 25/30 task, 25/30 safe, 0/30 contact, 0/30 XY; successful native
completion steps were median/p90/max 113/160/184. Against the primary current
matched Round-22 Constrain (26/25/1/0): task improved/regressed/both/neither
4/5/21/0; safe 5/5/20/0; contact cleared/introduced/both/neither 1/0/0/29;
XY 0/0/0/30. The previous contact cleared at scene 12, but native task success
regressed at scenes 5/10/11/17/24 while former failures 13/21/22/28 recovered.
The five current failures were all release/stability timeouts at the 200-step
limit: scenes 5/10/24 stayed on-bin but spinning; scenes 11/17 were high-motion
off-bin releases. The hold removes the single contact but does not clear the
remaining release/stability bottleneck.

**Gate: pause/fail.** Task 25 < 27 and safe 25 < 26; contact and XY pass.
This is a development result, not formal evaluation. Paired summary SHA
`63f316e6188c1ce2ea4b0c454c1552de67b63510bbca0aae13aa7348a54c8f04`; frozen
config SHA `fef33e7ca0f8319d043cfc7bdf5fcb74654a1ac12bf104e7825df94a6f8c3d88`.
The matched current EMA baseline and per-episode files remain intact.
Persistent Round-23 argv/budget/outcome ledger:
`outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23/round23_execution.json`
SHA `a30b1436e315413befa5013e30ee906d3310e27d9f0325c647d5556739f5f41f`;
all frozen training source hashes still match.

Final full constrain suite after Round-23 code additions:
`timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
— **498 passed, 28 warnings, 82.41 s**. `git diff --check` passed. Training,
evaluation, diagnostic and collection processes exited; no leftover `.tmp`
checkpoints or experiment processes were present. Ticket 08 remains blocked. The
next data-backed seam is current angular release/stability under the 200-step
policy budget; no additional R23 factor or formal evaluation was started.

## Round 24 — indexed release mechanics and exploratory execution stride 2

User-authorized single-developer continuation on
`goal/transport-detour-optimization`, starting at base
`9f190f80665810d55e35c3436710a0aeb2b2d95b` (integration branch `dev`). No
checkout, branch change, commit, or push. ManiSkill source, training data, and
checkpoint bytes were not changed. No training, data collection, formal
evaluation, or PickCube work was run.

### Release-mechanics evidence

Native Base DP and Constrain episodes were instrumented with full per-step
seven-channel actions, observation/replan step, queue index, TCP xyz/quaternion,
sphere xyz/quaternion and linear/angular velocity, grasp/open state, obstacle
contact/crossing flags, control timestep, and termination flags. The Constrain
trace reproduces all five R23 failures (scenes 5, 10, 24, 11, 17) and includes
the preregistered matched-success controls (0–4). Full traces remain inside
`paired_eval/{base_dp,constrain_dp}/episode_*.json`; indexed open-command and
physical `is_grasped: true→false` event joins are in
`release_causal_measurements.json`.

The training-expert audit reads the regenerated R23 hold-8 300-OBSTACLE HDF5,
not the older R22 expert. It verifies 48,982 action/state pairs: action index
`t` is compared to the T+1 native state `t+1`, and each of the saved linear and
angular speed values matches that same post-action state. The 300 open-action /
ungrasped transitions have pre-open linear speed median/p90 0.00549/0.00678 m/s
and angular speed 0.02665/0.03084 rad/s; same-index post-action medians/p90 are
0.36818/0.37391 m/s and 0.40389/0.43830 rad/s. This is a synchronized
before/after kinematic association, not an impulse or proof of cause. Expert
per-action timestep is absent, so expert `g·dt` is not calculated.

The five Constrain failure replays show pre-open motion ranging from 0.0030 to
0.0223 m/s linear and 0.0247 to 0.3560 rad/s angular. At the physical
ungrasped transition, downward linear velocity magnitude is about 0.364–0.378
m/s. The following 0.05 s displacement is about −0.0329 to −0.0336 m, compared
with −0.0123 m under the zero-initial-velocity `½g·dt²` comparison; observed
`dz/dt` and `g·dt` are recorded per event. This does not isolate release
actuation, contact, gravity, or pre-existing motion. Per-step sphere/robot
contact is not exposed in saved native step info (obstacle contact is); joint
torque and impulse are not measured. Do not infer a dynamic impulse from these
records.

### Preregistered stride-2 trial

Before evaluation, registration
`examples/baselines/constrain/placesphere_stride2_preregistration.json` pinned
the 30-scene manifest, exact source hashes, Base EMA
`92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`, and
Constrain EMA
`c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada`. One
factor changed: each policy call still predicts its original 16×7 sequence and
selects the original `[1:9]` actions; execution consumes two actions, appends
the latest two observations, and replans. Observation history remains 2, all 7
action dimensions, weights, sampler, scene seeds, clamps and safety behavior
are unchanged. More calls naturally consume more sequential diffusion RNG
draws; this coupled sampler-noise change is disclosed. This is explicitly
**EXPLORATORY**: more frequent feedback may help release errors, but R23 did not
establish that stale queued actions caused them.

Both policies ran all 30 scenes with the same restored state/controller/seed,
200 control steps, native first-success termination, a 300 s episode wall cap,
and a fixed 30 denominator. The 18,000 s outer timeout is compute-only; no
scene was dropped and no result-based stopping/selection occurred. The exact
command and exit/cleanup details are in `execution_record.json`.

| Policy | Task | Safe | Contact | XY crossing | Runtime errors | Successful steps median/p90/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base DP stride 2 | 8/30 | 0/30 | 30/30 | 25/30 | 0 | 160.5/185.5/189 |
| Constrain stride 2 | 26/30 | 26/30 | 1/30 | 1/30 | 0 | 135.5/158.5/171 |

Against primary R23 Constrain stride 8 (25/25/0/0), stride 2 has task/safe
transitions improved/regressed/both/neither = 4/3/22/1, and contact and XY
cleared/introduced/both/neither = 0/1/0/29 for each. Four of the five former
release/stability failures (5, 10, 24, 11) now succeed; scene 17 remains a
release/stability timeout at step 200. The one new contact/XY episode is scene
25. Against secondary R22 matched hold-0 Constrain (26/25/1/0), transitions are
task 4/4/22/0, safe 5/4/21/0, contact 1/1/0/28, and XY 0/1/0/29. R22 is
secondary and not a one-factor comparison.

The frozen gate remains **pause/fail**: task 26/30 is below 27; safe 26/30,
contact 1/30 and XY 1/30 meet their thresholds. The remaining scene-17 failure
and single introduced contact/crossing are retained; no rerun or additional
stride trial was started. New result directory:
`outputs/exp/20261026_235900_placesphere_stride2_r24/` (registration,
`paired_eval/`, per-episode full traces, `release_causal_measurements.json`, and
`execution_record.json`). Paired summary SHA
`3c537c1ba646e5d483e2fd4075524ba36bc0ea08eae4356cc4e0231276c2b0cd`; expert
HDF5 SHA `2bd725179ed61c8c32b97cdc9b89ce348add0bbf6b4d7a912cfa8bad30b9a5a1`.

Verification: constrain suite **502 passed, 28 warnings, 83.65 s**; `git
diff --check` passed. Full evaluation exited 0 with 60/60 episodes and no
runtime errors; model/evaluation child processes exited. Next minimal slice is
parent review of the indexed event traces, especially scene 17 and why the
post-action expert velocity jump appears in the matching T+1 record. Do not
claim causality or launch further trials in this round.

## Round 25 — candidate-collector interruption diagnosis and bounded resume canary

The obstacle1000 collection did **not** reach matched demonstration collection,
training, or paired evaluation. The checked-out branch remains
`goal/transport-detour-optimization` at HEAD
`6aeee1ecf559aa9e247e3348ff11f308595e94fe`; no branch operation, commit, push,
or ManiSkill source edit occurred. All native commands ran after activating
`maniskill` and had finite outer/per-attempt timeouts.

### Exact stall and retained work

The before-collection plan is
`outputs/exp/20261026_231800_placesphere_obstacle1000_r25/preregistration_collection.json`
(SHA `34d0b21272b34fefb7a5f0b5a4fc2469d73541877c57ca84455d2e87dc8e1cbe`). The
first geometry-candidate run in
`outputs/exp/20261026_231800_placesphere_obstacle1000_r25/candidate_source/`
requested 700 layouts under a 21,600 s wall / 10,000-attempt cap. It stopped
after 21,648.57 s: 2,941 attempts, 223 accepted safe geometry layouts (7.58%),
and 2,718 retained rejections. These are scene/layout proposals, not the new
hold-8 matched training episodes. Its seed interval was 2026102712–2026105652.
HDF5 SHA `cc92ddcc86faf5ebb6a2c2ac1b21fe30a55975bc6072ada81a55b9dbc477a2d4`,
manifest SHA `910cb1c8db1d5bd675624e7854c0f9f0c4f0f73ccd9bd0e05192a67175946fca`,
attempt journal SHA `6676d01ac0752b040a25f053fb8952127f214bf68b5c7fcad83d05426b4720d5`.
Rejection counts: 1,036 task-layout duplicate; 638 direct-template
unsuccessful; 552 side-detour column infeasible; 262 direct block at endpoint;
160 direct container unsuccessful; 27 waypoint out of bounds; 15 obstacle
native unsuccessful; 15 side-template unsuccessful; 13 obstacle contact or
crossing.

The next run reused the initial seed argument rather than starting after the
last journaled seed. Its 3,197 attempts were 2026102713–2026105909, overlapping
2,940 of the first run’s seeds and replaying those expensive physics searches.
It consumed a further 24,114.62 s, retained 16 accepted scenes and all 3,197
journal rows, but was interrupted before its final report/manifest write. The
root cause was the resume cursor, not training or GPU capacity. The incomplete
continuation HDF5 SHA is
`4a7394345c576994eb7709c3d6696ab8c42689ca6881ab6aa018966886ee45a9`; journal
SHA `4c81fdbfbdbdd08baa883930da6e79dfab037a35947c285dc95c91ecacc47db6`.
Those 16 journal-accepted scenes are at IDs `placesphere-000..015`; each has
matching seed metadata, finite 7-D expert actions, T/T+1 39/42 observations,
native success, and full-prefix safe-success flags. They were task-unique against
the first 223 and the preregistered exclusion catalogs. They are retained as
candidate-scene provenance only, not as training demonstrations. Total valid
candidate-layout records recovered: **239/700**, not 223+16 discarded or
recollected.

Recovery manifest
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/recovered_candidates_239/manifest.json`
(SHA `89b180a4d2cd6e3685d4b7b1eec4a76b0312bf4c7917b5d3ed530251af246ba3`)
joins the 223 complete first-run records and 16 safely recovered continuation
groups without editing either source directory. All 239 pass the frozen
seed-or-both-endpoints-within-0.02-m identity rule. Their native source arrays
remain in the two original candidate HDF5s.

### Bounded resume repair and canary

The repaired cursor scans every prior attempt journal, including rejected
attempts, rejects overlapping seed histories, and resumes strictly at
`max(attempted_seed)+1`. For the two long-run journals this is seed
2026105910; the canary then consumed seeds 2026105910–2026105917, so the next
full-batch first seed is **2026105918**. A cheap preregistered proposal replay
uses the same `default_rng(seed)` start/goal draws and geometry checks before
opening PhysX. It skips only geometry-exhausted or task-identity duplicate
proposals; all other proposals continue through the unchanged expert physics
and post-execution safety predicate. The candidate generator now atomically
writes `recovery_checkpoint.json` each attempt with accepted scene IDs/layouts,
last/next seed, attempted count, wall, source identity and journal byte count.
Forensic reconciliation checks accepted journal rows against recovered HDF5
groups and fails closed on a count/identity mismatch.

The resume/canary registration is
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/candidate_resume_canary_preregistration_r2.json`
(SHA `a6a7cd357d6aaa214a6b75b30f2f8e1e4558209a1a35cbf6f5a4b380172e8ba8`). It
records the 2,940-seed overlap, all source/artifact hashes and the exact cursor.
The 8-proposal / 300-s canary ran 20.67 s: 5 known duplicates were rejected
before environment construction, 3 new proposals reached physics, 0 were
accepted (one direct-template failure and two side-column failures), and all 8
attempts were durably recorded. This proves fresh-seed progression and the
preflight/progress path; it does not establish a new expert acceptance rate.
Canary journal SHA
`bcdffec1bb21b7be66f3a4dd8f815a1b1900f423978ed46bf62a1699d4f58499`, report SHA
`e7ca605d33c6f16c854fe5f8dc9c806bc31262efd84b5773493a03bf716a05db`, recovery
checkpoint SHA `3f5f4bbfd0ec98f14642a8ffa19153a5f2198b7a311625ba1a4f41612f6b464b`.

The next bounded continuation plan is frozen at
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/full_candidate_continuation_preregistration_r3.json`
(SHA `3895c80446ee163fc0c32ef599944145b20487acc7565a35ee2eebfaac64dd67`). It
uses first seed 2026105918, targets the remaining 461 candidate layouts, and
has an 8,836-attempt / 65,000-s budget derived from 16 accepts in 265 unseen
continuation+canary seeds, plus a fixed 1,200-attempt buffer. There was **no
full continuation, matched 700-pair collection, dataset assembly, training,
checkpoint verification, or R25 paired evaluation** in this bounded repair
round. The original 1,000 DIRECT + 300 OBSTACLE hold-8 dataset and both R23
checkpoints remain unchanged. R25 gate status is therefore not evaluated; the
fixed R24 stride-2 gate remains the last measured result.

The above R25 continuation is **superseded** by the user's newer fixed 1,000
DIRECT / 500 OBSTACLE limit. No 461-layout continuation, candidate expansion,
or 1,000-obstacle training was launched. The following R26 section records the
active frozen-data continuation and its final development result.

Verification after the resume fix: focused recovery/profile suite **5 passed**;
full constrain suite **507 passed, 28 warnings, 617.70 s**; `compileall` and
`git diff --check` passed. The canary and test child processes exited; no
collector, trainer or evaluator process remained. No episode was selected using
policy metrics. Detailed source edits are confined to
`examples/baselines/constrain/`; hashes are in the preregistrations and
canary/checkpoint records above.

## Round 26 — frozen 1,000/500 matched-data continuation

The user's newest constraint is fixed at **1,000 DIRECT / 500 OBSTACLE** and
prohibits adding layout candidates. This supersedes the historical 1,000-obstacle
target and the R25 461-candidate continuation plan above. No layout proposal was
sampled in R26; only layouts in the recovered, immutable 239-candidate pool were
replayed. The branch remained `goal/transport-detour-optimization` at
`6aeee1ecf559aa9e247e3348ff11f308595e94fe`. No checkout, branch change,
commit, push, or ManiSkill-source edit occurred. The existing Lightnav process
was left untouched.

### Frozen candidate replay, external-identity correction, and final data

The frozen-pool replay registration is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/frozen_replay_selection_r5.json`
(SHA `42eba7cd617cc7d47b8fc31d3f9fdaf4a35f48a8b2b2877a091748fe562f96a6`). It
pins the recovered pool manifest (`89b180a4d2cd6e3685d4b7b1eec4a76b0312bf4c7917b5d3ed530251af246ba3`),
both candidate-source HDF5 hashes, replay code hashes, deterministic
`(seed, original_id)` ordering, margin 0.015 m, hold-8 terminal contract, and a
one-pass maximum of 239 existing layouts. The ordered first 200 candidates were
replayed serially in 1,289.89 s; all 200 matched direct/obstacle/negative/suffix
admissions and both positive T+1 restore audits passed. No expert/restore
rejection or timeout occurred. The separate candidate-obstacle staging HDF5 SHA
is `dd5de15459201edb275ada9002c557d84474e95acd0ad718d1ec3a58a86cbd8a`; its
manifest SHA is `35652421107ec68a5f62b800f59c24e63217404ae047354e123c95f5616c569a`.
The complete first-pass ledger SHA is
`4ac08310aa66cf82c1b01d852ebbf70f07349baa320452019a03f1facdd28721`.

An independent external task-identity audit then found four of those admitted
layouts conflicted with Ticket 04: candidate IDs **1, 80, 84, 159**. They were
excluded from training regardless of their positive expert outcome. Their prior
package and hashes remain preserved as non-training history; the correction
sidecar explicitly supersedes its stale admission claim. Before replay, a second
registration
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/replacement_replay_preregistration_r2.json`
(SHA `f97a60c43ca7893596f82987cbf81e0182927925b610ec9f08755fe02a236392`)
froze only the remaining existing pool IDs, external manifests, replacement
order, code hashes, and four-replacement target. Existing candidates **200–203**
were replayed in order in 46.92 s; all four passed. Candidate IDs **204–238**
remain unused and are not training data. No new layout was sampled. An initial
standalone candidate-0 state-restore smoke was retained separately as
`first_candidate_attempt.{json,h5}` and is not a second training demonstration.

The active audited package is
`demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/`.
The four Ticket-04-conflicting candidates are absent. The new OBSTACLE class
uses source candidate IDs 0–199 excluding 1/80/84/159, plus 200–203, mapped to
final HDF5 IDs **1300–1499**. Its final HDF5 SHA is
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`, manifest
SHA `aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`, and
training-identity SHA
`648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78`. The
independent saved-artifact audit
`independent_r26_audit.json` (SHA
`35bd68b10bd4139e0e2033394e69f936fa1a568e40a89897bf5088d7dbb7e4e3`)
verified 1,500 total episodes, 1,000/500 class counts, exact original IDs
0–1299 HDF5 group-value preservation, all 200 new T/T+1 native 7-D / 39-D / 42-D
contracts, first-native-success and no-truncation evidence, 400 positive role
T+1 restore proofs, 200 negative transport-block proofs, 200 actual-cylinder
terminal-suffix proofs, and full-prefix obstacle contact/crossing safety.
Final task-identity intersections are **0** against the original obstacle-300,
frozen dev30, and Ticket 04 sets. OBSTACLE layouts are **500 unique / 0 repeats**.
The retained DIRECT class has **474 greedy-unique / 526 repeated** layouts;
these repeats are historical DIRECT content, not new unique-layout claims.

The initial r2 assembled package remains unchanged but is **not** an active
training input because of the four Ticket 04 overlaps. Its correction record is
`training_package_r3/external_task_identity_correction.json` (SHA
`4d5b987011bbd3130c17da57cfb8766fb3d129ed8f88ab31da4d21a94d8abbc3`). Neither
the historical HDF5 nor its original manifest/training-identity hashes were
rewritten.

### Before-training window audit and control-only training

Before the training registration, the active 500-OBSTACLE identity reader loaded
exactly 500 IDs and **80,151 official windows**. The matched R23 300-OBSTACLE
source has 48,682 windows; the ratio is 1.646. Training uses the preregistered
150,000 updates (`90,000 × 500/300`), yielding 1,916.38 cohort-equivalent window
passes versus 1,893.10 for the old 90k/300 run (about 1.2% higher). Data volume
and update budget change together; this is not a separated causal data-volume
effect.

The final package's complete, non-truncated action/window audit is in
`dataset_validation.json` (SHA
`edd744337e4ee1cdc0a6d1a29788ec7277f9196e08dd972293d966f5b0182e89`). DIRECT /
OBSTACLE episodes longer than 200 actions: **64/1,000** and **44/500**; action
volume beyond step 200: **17.83% / 17.59%**. Open near-zero actions: **27.17%
/ 24.97%** of action volume. Fixed stride-8 open/near-zero action-only windows:
DIRECT **3,739/16,610**, OBSTACLE **2,055/9,859**. Separate linear-static counts
are **3,647/3,739** and **2,012/2,055**; angular-static counts are **0/3,739**
and **0/2,055**; both-native-static counts are zero. Uniform official-window
sampling was retained; no phase weighting, trajectory weighting, tail
truncation, policy-performance filter, or artificial neutral window was used.

The before-training registration is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/training_preregistration_r1.json`
(SHA `08f361e950586758283ebd7f472d264fbb77b2a1e5ea29e55047feffcebb7a7a`). It
pins all dataset/source/EMA/dev/baseline hashes, the exact 150,000-update
training and single stride-2 evaluation plan, exposure calculations, budgets,
and gate. The two-update CUDA preflight record is
`training_preflight.json`: finite losses, moved zero-convolution weights,
nonzero zero-convolution gradients, moved trainable control, bit-unchanged
Base, and zero disabled-control epsilon/sampler difference. No preflight
checkpoint was selected or used.

Exact training command (9000-s timeout):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 9000s \
  python -u -m examples.baselines.constrain.placesphere_constrain \
  --demo-path demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.json \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --run-dir outputs/exp/20261027_140000_placesphere_hold8_500_r26/training_control_150k \
  --checkpoint-dir outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k \
  --device cuda --total-iters 150000 --batch-size 1024 --seed 1 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 \
  --matched-data-identity demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/training_identity.json
```

Training completed 150,000 updates in **4,316.23 s**. Losses were finite
(0.02049344 → 0.00134501); frozen Base unchanged; control parameters and all
control-down/mid/hint gradient families moved; disabled-control epsilon and
100-step sampler maximum difference were both 0.0. Final EMA:
`outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt`,
SHA `ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71`.
Run-config SHA `a560490d198d7b62ab790556a05a243ba739beae4fe98a41518ccd29ea744489`;
training-summary SHA `7dd5ba8bc1ca5bcfc7440fd1256028d47dffe798be29bdbceeba43e3218365bc`.

### One paired stride-2 dev run

The sole evaluation was preregistered in
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/evaluation_preregistration.json`
(SHA `1b304218b462e6cde8ff63dab1463e8a2da51cb73920d7a3a19839fdb6513a81`). It
reuses the current frozen 30-scene dev states/controllers/observations/seeds,
stride 2, prediction horizon 16, executed action slice [1,9], 200 control steps,
300 s per episode, fixed denominator 30, and the R24 stride-2 baseline. No
stride grid, scene addition, checkpoint selection, or second paired run occurred.

Exact evaluation command (18,000-s outer timeout):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --constrain-checkpoint outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt \
  --output-dir outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2 \
  --primary-baseline outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval \
  --device cuda --replan-stride 2
```

Both policies completed **30/30** with zero runtime errors. Base: **8 task / 0
safe / 30 contact / 25 XY**. Constrain: **25 task / 25 safe / 0 contact / 0
XY**. Against R24 stride-2 Constrain (26/26/1/1), task improved/regressed/both/
neither = **2/3/23/2**, safe = **2/3/23/2**, contact cleared/introduced/both/
neither = **1/0/0/29**, XY = **1/0/0/29**. The five new failures are release/
stability timeouts, with no contact or crossing: scenes 8 (release step 155,
0.3740 m/s, 0.3204 rad/s, off-bin), 13 (109, 0.3341 m/s, 1.2531 rad/s,
off-bin), 15 (106, 0.3717 m/s, 0.2945 rad/s, off-bin; terminal on-bin but
spinning at 3.6033 rad/s), 17 (73, 0.3892 m/s, 0.3047 rad/s, off-bin), and 25
(117, 0.3183 m/s, 0.9206 rad/s, off-bin). Successful native steps were median
114, p90 173, max 179. Episode-level release/contact/terminal records and file
hashes are indexed in `paired_eval_index.json` (SHA
`e860a8e8721b4392e0ac18be708f85c0d19a3d8f4495d654e9ede75e428f48cd`). Paired
summary SHA `cac80011768efd2f94ed823cfcdef6646f4c01d0e1348abd9747f0fc95cef07e`;
frozen config SHA `e82bc5df0e70e4ca7bf9286d1cfe36fbfcedbe81f1d864020c1b00c7abb97d9d`.

**Gate: pause/fail.** The fixed requirements are task ≥27/30 and safe ≥26/30;
observed 25/30 for each. Contact and XY are 0/30 and pass. This is repeated
development-set evidence only, not a formal/generalization claim. No new-data
collection is authorized by this result. Continue only with optimization on the
frozen 1,000/500 data and the same dev set after parent review.

Final constrain suite: **512 passed, 28 warnings, 346.62 s**. `git diff --check`
passed. Training, replay, and evaluation children exited; the RTX 5070 returned
idle. Machine-readable command/budget/outcome and input/output hashes are in
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/round26_execution_record.json`
(SHA `acee5cade32864ac2af672fa72877117e63e85711c34d69cdf0147070b56ec97`).

After the data/training/evaluation run, all old R25 expansion/canary/461-layout
planning and execution entry points in `placesphere_obstacle1000_expansion.py`
were made fail-closed under the active freeze. No data, training, or evaluation
was rerun after that guard-only change. The R26 replay and replacement
registrations retain the exact source hashes used at those stages; the
post-run source transition and final verification are recorded in
`postrun_obsolete_candidate_guard_audit.json` (SHA
`deeea2e823295cecd9121f2e2d616fd8eec56adceb42870c140d8eab601ff1a8`). The final
post-guard test run is **512 passed, 28 warnings, 309.59 s**.

## Round 27 — frozen-data DDIM inference trial (preregistered)

The active authorization remains fixed at **1,000 DIRECT / 500 OBSTACLE**; no
new data, layouts, or trajectory collection is included. Round 27 registers
exactly one inference-only change for both already-trained policies: runtime
`DDIMScheduler`, 100 inference steps, `eta=0`, created from the loaded
training-time DDPM scheduler config (squared-cosine betas, epsilon prediction,
sample clipping). The training checkpoint contract remains DDPM. Reverse
variance draws are removed, so later global RNG positions also differ; this is
one sampler-configuration comparison, not an isolated variance-only causal
effect. Default DDPM adapter call paths remain unchanged.

The preregistration is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/inference_preregistration.json`.
It pins the exact r26 HDF5/manifest/training-identity, Base EMA and final
Constrain EMA hashes and paths, all 30 frozen dev scenes, R24 stride-2 baseline,
runtime source hashes, one 60-policy-episode paired budget, per-episode limits,
and the gate. Both policies retain stride 2, prediction horizon 16, executed
slice `[1,9]`, same saved state/controller/observation/seed reset, native 7-D
actions, and existing action mapping/clamps. Runtime errors remain in the fixed
denominator. The evaluator fails closed on any preregistered artifact/source
identity mismatch before creating an evaluation output or executing a scene.

The sole allowed command is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --constrain-checkpoint outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt \
  --output-dir outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval \
  --primary-baseline outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval \
  --device cuda --replan-stride 2 --sampler ddim \
  --inference-preregistration outputs/exp/20261027_204500_placesphere_ddim100_r27/inference_preregistration.json
```

Registration-time checks: focused sampler/PlaceSphere/pilot tests **64 passed**;
full constrain suite **517 passed, 28 warnings, 328.18 s**; CLI import/help and
`git diff --check` passed.

### Single paired run and result

The sole registered command completed normally from **2026-10-27 21:00:48.570
to 21:22:35.919 +08:00** (1,307.348 s, below the 18,000 s cap). It evaluated
**30/30 episodes for each policy**, retained zero runtime errors, and had 60/60
bit-exact scene restores with matching paired scene IDs/seeds. No episode was
dropped/retried. The RTX 5070 returned idle; no unrelated compute process was
started or stopped. The frozen data, training identity, and both checkpoint
hashes were identical before and after.

DDIM Base achieved **6 task / 0 safe / 30 contact / 24 XY**. DDIM Constrain
achieved **26 task / 26 safe / 0 contact / 0 XY**. Its 26 successful native
episodes had median / p90 / max success step **125 / 171 / 171**. Against R24
stride-2 Constrain (**26/26/1/1**), task and safe transitions were
improved/regressed/both/neither **3/3/23/1**; contact and XY were each
cleared/introduced/both/neither **1/0/0/29**. Three R24 failures (indices
17/23/25) cleared and three prior successes (6/11/14) regressed; index 9 failed
in both. All four new failures were incorrect releases, with no contact or
crossing: indices **6, 9, 11, 14**, release steps **136, 161, 78, 82**. Release
linear speeds were **0.389, 0.405, 0.425, 0.367 m/s**; every release was off-bin
and high above the bin (relative z **0.151, 0.155, 0.109, 0.140 m**). Per-step
contact/release/terminal mechanism traces are retained in their episode JSONs.

**Gate: fail/pause.** Task success 26/30 is one short of ≥27; safe 26/30 passes
≥26; contact 0 and XY 0 pass ≤3. This is frozen-development-set evidence only.
No formal 08 evaluation or new collection follows. Parent review selects the
next frozen-data optimization slice before any further experiment.

The machine-readable execution record is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/round27_execution_record.json`;
SHA `5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. It pins
the exact command/budgets/timestamps, all input and output summary hashes, the
64-file evaluation-tree digest, resource/cleanup state, comparisons, failure
indices/mechanisms, and gate transition. The full paired summary is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/paired_summary.json`
(SHA `ef24daa752f99ce31d5063325397c1276f3e6bc3d085dcaeafa824904378346e`); the
evaluation log SHA is `fed54e6d0256e7878b164a32a7fc6e776a0d5c0f5a23ddf999a9d3fb12a989cd`;
the paired-output tree SHA is
`6c47aab68cfe5947d0671b491a317e61355b422ee4d02559933fbecb15cd80c5`.

## Round 27 comparison correction — R26 is primary; R24 is secondary

**Correction to the Round 27 interpretation above:** the DDIM trial was compared
there to R24 as if it were the primary baseline. That was wrong. The required
primary is the immediately preceding **R26 DDPM100 stride-2** run, which uses
the exact same Base and Constrain checkpoints, frozen scenes, seeds, and rollout
limits. R24 has different data/checkpoint generation and is secondary context,
not evidence for this sampler comparison.

The actual R26 episode path is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2/`.
Its frozen config SHA is
`e82bc5df0e70e4ca7bf9286d1cfe36fbfcedbe81f1d864020c1b00c7abb97d9d`, paired
summary SHA `cac80011768efd2f94ed823cfcdef6646f4c01d0e1348abd9747f0fc95cef07e`,
and its registered R26 evaluation preregistration SHA is
`1b304218b462e6cde8ff63dab1463e8a2da51cb73920d7a3a19839fdb6513a81`. R26
records Base **8/0/30/25** and Constrain **25/25/0/0** (task/safe/contact/XY).
Its original command omitted a sampler override; the registered R26 evaluator,
Base adapter, and Constrain adapter source hashes establish the original
DDPM100 path. The R26 and R27 runs pin the same Base EMA SHA
`92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`, Constrain
EMA SHA `ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71`,
scene manifest SHA `9e1d8ee4786d2336f01e07d664ca37c3dcc7ca455e7876176def65c18b63c119`,
and scene HDF5 SHA `adc7d1b2a16745c459db53d2c8f818ac87d1ffffe21fd44c54509f77cef342d7`.
Both use 30 scenes, stride 2, prediction horizon 16, executed slice `[1,9]`,
200 control steps, and 300 s per-episode wall cap. The correction analysis
verified all per-policy scene IDs/seeds, bit-exact restores, and exact initial
policy observations across R26 and R27.

Correct primary transitions, ordered improved/regressed/both/neither for
task/safe and cleared/introduced/both/neither for contact/XY:

| Policy | R26 DDPM100 task/safe/contact/XY | R27 DDIM100 task/safe/contact/XY | Task | Safe | Contact | XY |
|---|---:|---:|---:|---:|---:|---:|
| Base | 8/0/30/25 | 6/0/30/24 | 3/5/3/19 | 0/0/0/30 | 0/0/30/0 | 3/2/22/3 |
| Constrain | 25/25/0/0 | 26/26/0/0 | 5/4/21/0 | 5/4/21/0 | 0/0/0/30 | 0/0/0/30 |

Constrain failures changed from R26 indices **8, 13, 15, 17, 25** to R27
indices **6, 9, 11, 14**. The corrected comparison therefore shows task and safe
success each increased by one (25→26), with five R26 failures resolved and four
previous successes regressed. Contact and XY remained zero in both. The fixed
gate still fails task ≥27 by one; safe ≥26 and both safety ceilings pass.
Base failure-index sets and both policies' complete metric transitions are in
the machine-readable correction result.

R24 remains preserved as secondary comparison: it reports 26/26/1/1 for its
Constrain checkpoint/data and its old R27 transition counts remain in the
original execution record, but are confounded for sampler inference. The
original R27 preregistration still names R24 as primary; it has **not** been
rewritten or represented as retrospectively corrected.

The read-only correction was preregistered separately at
`outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json`
(SHA `5594ff8af2a1831fbb78a8f3d60c1dcfd1699e57b34b1b2a2f43bda5d43ea8e9`). Its
result is `primary_comparison.json` (SHA
`b37577c8442f6ee1a1749c3fd2283df5670d6e9dc7710691fadd49f91cf78b26`). The
timestamped record hashes both 64-file episode trees and all source/input
registrations. A prior analyzer preflight stopped before computing transitions;
its log is preserved, and the successful analysis used the superseding
registration. The correction was analysis-only: no native evaluation, replay,
training, checkpoint/data mutation, or additional scene execution occurred.
The Round 27 execution record was appended with this correction, the R24
reclassification, provenance hashes, and corrected transition table; current
execution-record SHA is
`5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. The
bounded analysis regression tests passed: **2 passed**.

### R27 correction provenance rerun — immutable registered execution-record input

The first correction registration is retained unchanged at
`outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json`
(SHA `5594ff8af2a1831fbb78a8f3d60c1dcfd1699e57b34b1b2a2f43bda5d43ea8e9`), as is
its result (SHA `b37577c8442f6ee1a1749c3fd2283df5670d6e9dc7710691fadd49f91cf78b26`).
Its registered R27 execution-record input pinned SHA
`bc3483a62d0a1496b93d003cd73bb2dcdce1081c08c2d608b6aa90ff3187fd9d`, but the
same path was later appended with the correction and had SHA
`5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. A direct
rerun therefore failed its registered-input hash gate at
`r27_original_execution_record`; the earlier registration/result remain honest
historical records, not silently repaired artifacts.

The superseding registration is
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json`
(SHA `7d294ee9876420d5ef9d26bb0dc9575a05c5ab3ae43f63a9287652cb51da4249`). It
registers a read-only snapshot at
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/inputs/round27_execution_record.current.json`
(SHA `5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`),
copied byte-for-byte from the R27 record before adding the outputs-only pointer
to this rerun. The current mutable R27 execution record SHA is now
`a10abed8c1e96cb66e898397c3e0dfcd25b3654e3ad50e43709c47c814ec3562`; it is not
an analysis input. All other registered file and 64-file episode-tree hashes,
including the R26/R27 scenes and checkpoint identities, pass unchanged.

The reproducible CLI, run twice with exit 0 and byte-identical stdout SHA
`43d56e96dbe930ad98b47397db3117df831777fb00e4b2b58ad50ea24f6cda6b`, was:

```sh
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && timeout --signal=TERM --kill-after=15s 180s python -m examples.baselines.constrain.placesphere_r27_primary_analysis --registration outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json
```

The saved result is
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/primary_comparison.json`
(SHA `43d56e96dbe930ad98b47397db3117df831777fb00e4b2b58ad50ea24f6cda6b`). It
reproduces R26 Base **8/0/30/25** vs R27 Base **6/0/30/24**, and R26 Constrain
**25/25/0/0** vs R27 Constrain **26/26/0/0** (task/safe/contact/XY), including
the transition counts above. R24 remains secondary and the original R27
preregistration's R24-primary deviation is explicitly preserved.

The analyzer source SHA is
`149035f8fc8715219956f1f15350de96e04498f53f7f1116ed1b5360600012d6`; the
regression-test source SHA is
`691d8ae35b7b42fa7cbc60f4ca7f9f5bef81f365179e9515d34e7e8a6fb87ea8`. Focused
tests passed **4/4**, including reproducing the retained stale-pin failure,
accepting the real frozen snapshot, and rejecting post-freeze mutation. No
native episodes, replay, training, data, checkpoint, or scene were created or
changed.

## Round 29 — fixed four-plan DDIM ensemble (inference-only)

Continued on `goal/transport-detour-optimization`, HEAD/base
`a1dcc014fe38976e384610df1f933719672256de`; no checkout, commit, or push.
Training remains frozen at **1,000 DIRECT / 500 OBSTACLE**. No new demos, layouts,
weights, or ManiSkill source edits. One registered inference change was used for
both policies: four independent Gaussian `(16,7)` initial plans, identical latest
two-frame condition, 100-step DDIM eta 0 denoising in one batch, arithmetic mean
of all seven channels after denoising over `[1:9]`, then the existing action map
and first-two-action execution/replan. No sample ranking, clamping changes, or
route selection. Defaults remain one plan and existing DDPM call paths.

Registration: `outputs/exp/20261027_230000_placesphere_ensemble4_r29/`
`inference_preregistration_r2.json` (SHA
`25b4a8d1e34d52b7dd410f7546a4fd807182321f4592c31b5192744ca531ae49`). The
initial registration was retained at `inference_preregistration.json`; its first
launch stopped before output creation or any scene execution because the
ensemble-size guard expected the field under `protocol`. The corrected r2
registration moved that same fixed value there; the failed preflight log remains
at `evaluation_command.log` (SHA
`14aa56cb8b6341ca4ae48d62459a534bbe6733ab530723f21c883c60f33498ef`). The one
native trial then ran with the fixed 18,000 s outer limit, 300 s/episode, 200
control steps, serial CUDA inference and 30-scene denominator per policy. Exact
command and complete input/source identities are in `round29_execution_record.json`.

The immutable primary is R27 DDIM100 eta0 single-plan at
`outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/`, with identical
weights and frozen scenes. The evaluator preflight checked all 30 scene IDs,
seeds, bit-exact restore flags, checkpoint hashes, and equal initial 42-D policy
observations. After the run, all 30 initial observations per policy matched R27
again. All 60 episodes completed; 0 runtime errors, dropped episodes, or retries.
Elapsed wall time was approximately 1,260 s (22:02:59–22:23:59 local). Maximum
single synchronized ensemble inference time was 0.3745 s for Constrain and 0.4712
s for Base. The GPU returned idle after execution.

| Policy | Task | Safe | Contact | XY | Runtime errors |
|---|---:|---:|---:|---:|---:|
| Base | 3/30 | 0/30 | 24/30 | 23/30 | 0 |
| Constrain | 25/30 | 25/30 | 0/30 | 0/30 | 0 |

R27 Constrain was 26/26/0/0. R29 vs R27 transitions (task and safe are
improved/regressed/both/neither; contact and XY are cleared/introduced/both/neither)
were task **4/5/21/0**, safe **4/5/21/0**, contact **0/0/0/30**, XY **0/0/0/30**.
The five current Constrain failures are scenes 2, 12, 16, 19, 29: four
`incorrect_release` (2/12/16/29), one `not_stabilized` (19). The four incorrect
releases were off-bin; scene 19 released with 0.484 m/s linear speed and remained
unstabilized. There were no contacts or XY crossings. Base vs R27 was task
1/4/2/23, safe 0/0/0/30, contact 6/0/24/0, XY 5/4/19/2.

**Gate: fail/pause.** Task 25/30 is below 27 and safe 25/30 is below 26;
contact and XY ceilings pass at 0. No formal evaluation follows from this
repeated-development-set result. Keep the dataset frozen and diagnose the actual
release/stabilization regressions before selecting a next optimization slice.
This trial changes ensemble size, random draw consumption, and compute together;
it does not establish that stochastic dispersion caused the earlier XY release
errors or that action averaging preserves route topology.

The evaluator's unmodified raw `paired_summary.json` has legacy R24 comparison
labels and an old 90% gate even though the supplied primary directory was R27.
Those episode outcomes were not edited. The separately registered read-only
`primary_analysis.json` is authoritative for R27 transitions and the 27/26/3/3
gate. Analysis registration SHA
`55c1a454f4c5e8c6e0f72069f80d453d0f11fed1f598c7b4c41c2466b6c94e2d`; analysis SHA
`33fa28df35422b703c2880eb2980273cbec7dee3ef765109e70663023a7ee846`. R29 paired
summary SHA `459d169f4e22317a9056a8304302f212c98ddace90cd8da48c5abac85f204c09`.
The Base EMA, Constrain EMA, HDF5, manifest and training-identity hashes were
recomputed after the run and remain the registered R26 identities. The complete
run, failure list, paired transitions, hashes and cleanup record are in
`round29_execution_record.json`.

Verification after the changes:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s \
  python -m pytest examples/baselines/constrain/tests -q --tb=line
```

Result: **528 passed, 28 warnings, 82.59 s**. This includes a bitwise equality
check of the existing single-plan DDIM path against the pre-ensemble known-latent
reference loop. `git diff --check` passed. No
training, replay, or additional native episode was started after the registered
trial. Execution record SHA
`44ee060c7a29be94c9ab52007809a6c31be377634c49f071b831e3025dce4637`.

## Round 30 — fixed-range action representation optimization

Continued serially on `goal/transport-detour-optimization`, starting at
`60ccdf0572016d9d1090bb1382d54de9c80e11d9`; integration branch remains `dev`.
No branch switch, commit, or push occurred. This round used only the immutable
R26 dataset with exactly **1,000 DIRECT / 500 OBSTACLE** episodes; it generated
no demonstrations, candidate layouts, or seeds. ManiSkill and other baseline
source files were not changed. Experiment artifacts are under
`outputs/exp/20261027_224727_placesphere_action_codec_r30/` and the new model
weights are under `outputs/ckpt/20261027_224727_placesphere_action_codec_r30*/`.

### Full-data action audit and codec contract

Before registration, a full read-only audit checked the frozen HDF5 and
manifest-selected class IDs. Its output is `action_range_audit.json` (SHA
`961915a1193fa60fa01e31b97e25a5e919f1ecb0682cdff2812de173966dc01f`). The data
identity remains HDF5
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`, manifest
`aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`, and training
identity `648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78`.
All 136,417 DIRECT and 80,651 OBSTACLE native actions fit pose `[-0.35,0.35]`
and gripper `[-1,1]`; no value was clipped or changed. The pose cap is in the
native normalized `pd_ee_delta_pose` controller coordinates, **not meters or
radians**. Selected audit values:

| Class | Channel | Native RMS | Exact zero fraction | Absolute q90 | Absolute q99 | Absolute max |
|---|---|---:|---:|---:|---:|---:|
| DIRECT | delta_x | 0.12710 | 0.33796 | 0.20000 | 0.35000 | 0.35000 |
| DIRECT | delta_y | 0.08169 | 0.33766 | 0.12983 | 0.35000 | 0.35000 |
| DIRECT | delta_z | 0.12814 | 0.33773 | 0.34092 | 0.35000 | 0.35000 |
| DIRECT | delta_euler_x/y/z | 0.02889 / 0.04644 / 0.19305 | 0.55026 each | 0.01249 / 0.05737 / 0.35000 | 0.14666 / 0.23929 / 0.35000 | 0.35000 each |
| DIRECT | gripper | 1.00000 | 0 | 1.00000 | 1.00000 | 1.00000 |
| OBSTACLE | delta_x | 0.11137 | 0.30602 | 0.20000 | 0.35000 | 0.35000 |
| OBSTACLE | delta_y | 0.12018 | 0.30575 | 0.20000 | 0.35000 | 0.35000 |
| OBSTACLE | delta_z | 0.11811 | 0.30577 | 0.29482 | 0.35000 | 0.35000 |
| OBSTACLE | delta_euler_x/y/z | 0.02663 / 0.04248 / 0.17738 | 0.61863 each | 0.01044 / 0.04851 / 0.35000 | 0.13141 / 0.20760 / 0.35000 | 0.35000 each |
| OBSTACLE | gripper | 1.00000 | 0 | 1.00000 | 1.00000 | 1.00000 |

For each class, the audit also records signed min/max, encoded RMS/variance,
zero fractions, and channel-wise forward-diffusion SNR at timesteps 0/10/25/50/
75/99. The SNR is descriptive per-channel variance under the standard noising
process, not a causal claim. At timestep 50, encoded pose-channel SNR ranges
from 0.0053–0.1903 DIRECT and 0.0053–0.1725 OBSTACLE, while the gripper is
0.8593 / 0.9075; this exposes a substantial scale imbalance motivating the
fixed controller-range encoding.

The opt-in, versioned profile is
`placesphere_controller_range_v1`, scale `[.35,.35,.35,.35,.35,.35,1]`, zero
offset. It encodes native action targets before the unchanged epsilon/DDPM
objective and decodes model outputs to native seven-channel controller actions
before the existing action map/clamp. Observation identity normalization,
39-D base observation, 42-D hint input and hint normalization, model channels,
sampling schedule, padding, gripper behavior and environment scaling are
unchanged. Native HDF5 arrays remain untouched. Identity-profile historical
checkpoints and their default inference path remain strict/bit-exact.

Training registration
`training_preregistration.json` (SHA
`eb4a824224f649881a3511dc229d751d38c926255820d16c2016839e14f549a5`) pins the
frozen IDs/hashes, codec profile, exact 30k Base + 150k Constrain final-EMA
models, source hashes, preflight outputs, budgets, R27 reference and 27/26/3/3
gate before long training. Two-update CUDA preflights passed for Base and
Constrain; disabled-control epsilon and 100-step sampler matched the new Base
exactly (max difference 0). The control branch moved, all losses were finite,
all recorded gradient families were nonzero, and the frozen Base stayed
unchanged.

### Matched training and metadata correction

Base trained for 30,000 DIRECT updates in **900.52 s**: loss 1.160607 →
0.003348 (min 0.002550, max 1.160881). Initial final-EMA checkpoint
`outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base/final.pt`, SHA
`08879f41d1524a30f5aba9bfaaf2f107a5417b5fdc4d7463c8e274c7edda41cc`.
Constrain trained for 150,000 OBSTACLE updates from this matched Base EMA in
**4,217.93 s**: loss 0.082727 → 0.002647 (min 0.001801, max 0.115430). Initial
Constrain final EMA
`outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_150k/final.pt`,
SHA `476d582a5993d55a80382dffec24fe7059bf3621b0ed2bede9a2f74166c345ff`.

The first evaluation stopped before creating an environment or running any
scene because strict validation caught a Base checkpoint metadata defect: its
actual R26 HDF5/manifest/training identity and selected 1,000 DIRECT IDs were
correct, but its `dataset_version` field retained the older R23 constant. Both
original checkpoints were preserved. A hash-guarded correction created
metadata-only copies after verifying the Base EMA, Constrain model and Constrain
EMA tensors remained unchanged. The correction record is
`checkpoint_metadata_correction.json` (SHA
`bf6942e8771ef53de6edac66dac295f7c4810b44c07c1e8227a135f0c218457c`). Corrected
Base EMA SHA `491970382f3ed8dac7ed88ad1ab3f5cd23fb2d29065291a507ff766c75a6f071`;
corrected Constrain SHA
`6d66a875fd56bff8f9ec646ee5d89a8ebffe5c2d4401b5a541e67e49093a3e6f`. The
trainer now records the matched identity's dataset version directly, and runtime
loaders pin the full R26 data identity and exact direct/obstacle ID lists.

### One R27-paired DDIM evaluation and authoritative gate

Evaluation preregistration `evaluation_preregistration_r3.json` (SHA
`a0ec9488355b68912b04aa18e620bb360ceffaaf019865bf0ae8e95d5be70075`) registered
the corrected checkpoints before any native episode. The sole evaluation ran
DDIM100 eta 0, single plan, stride 2, slice `[1:9]`, 200 control steps, 300 s per
episode, 30 fixed canonical scenes per policy; elapsed time was approximately
612 s. It completed 60/60 episodes with 0 runtime errors, drops or retries and
bit-exact scene restores. The primary was the immutable R27 DDIM100 single-plan
run at `outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/`; scene
IDs, seeds and initial 42-D observations matched per policy. The comparison is
end-to-end because both policies' weights changed to the new codec; it does not
isolate Constrain-only attribution.

| Policy | Task | Safe | Contact | XY crossing | Runtime errors | Successful steps median / p90 / max |
|---|---:|---:|---:|---:|---:|---:|
| Base DP | 4/30 | 0/30 | 29/30 | 27/30 | 0 | 124.5 / 137 / 137 |
| Constrain | 29/30 | 29/30 | 1/30 | 1/30 | 0 | 106 / 154 / 170 |

Constrain failed only scene index **3** (`placesphere-003`), a 200-step timeout
with release/stabilization contact; its full release/contact/crossing trace is
in the episode record. Against R27 Constrain 26/26/0/0, task and safe
improved/regressed/both/neither are **4/1/25/0**; contact and XY
cleared/introduced/both/neither are **0/1/0/29**. The fixed gate passes:
**29 task, 29 safe, 1 contact, 1 XY** against thresholds 27/26/3/3. The new
Base result is 4/0/29/27 task/safe/contact/XY. Paired channel-wise encoded and
predicted-native actions were captured on every executed trace action and
round-trip verified (3,542 Constrain and 5,700 Base actions).

The raw paired summary uses the explicit protocol
`placesphere_action_codec_r30_primary_r27_v1` and labels R27 as the primary; it
does not reuse R29's legacy R24 comparison or gate. The separately registered
authoritative analysis has immutable byte-for-byte R27 and R30 evaluation
snapshots at `analysis_inputs/`, registration
`analysis_registration_r2.json` (SHA
`dabd5a8d7900961f04ff9647615aa2350e2d115a1a615774bc9326bb72466b06`), and result
`primary_analysis_r2.json` (SHA
`45e54c4f921893c29314148d22523c2904ba87415fa1be2ac8f8d21afe059342`). The
machine-readable command, budgets, attempt history, checkpoint provenance,
results and cleanup status are in `round30_execution_record.json`.

Final constrain suite: **537 passed, 28 warnings**; `compileall` and
`git diff --check` passed. GPU returned idle and experiment processes exited.
This is a repeated-development-set result, not formal/generalization evidence.
The gate permits **parent review before formal08 preparation only**; no formal
08 evaluation or new scene/data run was started.
