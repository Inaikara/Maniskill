# PickCube formal expert data (Ticket 02)

The formal dataset uses the accepted round-9 pilot contract from
`demos/PickCube-obstacle/20261022_130421_pilot/`; that 20+20 pilot is excluded
from formal counts. The sampling, expert, robot, physics, action, obstacle,
safety and horizon values are frozen by `frozen_contract.json` in the formal
dataset directory before its first attempt. Its source fingerprints are hashes
of the actual working-tree generator files at that point (the collector is a
working-tree change, so committed-source hashes alone would be misleading).

## Frozen formal run and reproduction

Dataset: `demos/PickCube-obstacle/20261022_132753_formal/`  
Run evidence: `outputs/exp/20261022_132753_pickcube_formal/`

The finite collection ceilings are 1,000 direct accepts / 2,000 direct complete
attempts and 100 safe obstacle accepts / 1,000 obstacle complete attempts, with
a shared 6,000-second wall limit and 500 control steps per complete attempt.
The 6,000-second estimate extrapolates the pilot's measured 16.03 s for 20
direct accepts and 95.12 s for 20 safe obstacle accepts. No pilot trajectories
are included. Layout geometry retries (32), waypoint candidates (32), cylinder
grid candidates (21×25), and control-step limits remain separately bounded.
There is no opposite-side fallback or relaxed contact/safety acceptance.

Run from the repository root:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' \
  timeout --signal=TERM --kill-after=10s 6000s \
  python -m examples.baselines.constrain.pickcube_pilot \
  --output-dir demos/PickCube-obstacle/20261022_132753_formal \
  --diagnostics-dir outputs/exp/20261022_132753_pickcube_formal/collection \
  --seed 9917 --target-per-class 1000 --direct-target 1000 --obstacle-target 100 \
  --max-attempts-per-class 1000 --direct-max-attempts 2000 \
  --obstacle-max-attempts 1000 --max-wall-seconds 6000
timeout --signal=TERM --kill-after=10s 1800s \
  python -m examples.baselines.constrain.validate_pickcube_formal \
  --dataset-dir demos/PickCube-obstacle/20261022_132753_formal \
  --output-dir outputs/exp/20261022_132753_pickcube_formal/validation_full_contract
timeout --signal=TERM --kill-after=10s 600s \
  python -m examples.baselines.constrain.validate_pickcube_formal \
  --dataset-dir demos/PickCube-obstacle/20261022_132753_formal \
  --output-dir outputs/exp/20261022_132753_pickcube_formal/validation_full_contract \
  --replay-representatives
timeout --signal=TERM --kill-after=10s 60s \
  python -m examples.baselines.constrain.validate_pickcube_formal \
  --dataset-dir demos/PickCube-obstacle/20261022_132753_formal \
  --output-dir outputs/exp/20261022_132753_pickcube_formal/validation_full_contract \
  --resolve-runtime-contract
```

`trajectory.h5` is flushed after each accepted episode. `attempts.jsonl` and
`episodes.jsonl` are append-only durable journals; `report.json` is atomically
updated during collection and finalized with aggregate counters/hashes. The
final `trajectory.json` retains layout, waypoint, obstacle, base XY, native and
controller initial-state identities, named observation-field shapes, and
episode safety evidence. `frozen_contract.json` is written before any sampling
attempt. Validation walks every episode and every state leaf, checks finite
7D bounded actions and T+1 observations/states/terminal flags, loads every
training window through `PickCubePilotDataset`, and restores each initial native
state and actual serialized controller state in a fresh scene while checking
the exact saved initial observation. Validation must report exactly 1,000
direct and 100 obstacle records; otherwise Ticket 02 remains incomplete.

## Contract

- Both classes use the same cube and target ranges, table/workspace bounds,
  transport-distance filter, base-to-SG-line clearance, and same-base-side
  waypoint distribution. Direct demonstrations do not insert a waypoint.
- Obstacle demonstrations use the pilot's static 0.03 m-radius, 0.25 m-high
  cylinder, 0.005 m safety margin, and `1e-6` contact impulse threshold. A
  candidate must block the direct XY segment while clearing the measured full
  robot sweep; physical execution is authoritative. Contact and forbidden XY
  crossing each reject the episode.
- Panda native reset noise remains 0.02. Control is `pd_ee_delta_pose`, state
  observations, CPU PhysX, seven action channels, native signed Euler scaling,
  500-step attempt cap, and 2/16/8 observation/prediction/execution horizons.
- Task success and terminal evidence remain native. Obstacle success additionally
  requires no robot/cube obstacle contact and no forbidden XY crossing. No
  action conversion is applied; training consumes the native seven channels.

Formal counts, elapsed times, acceptance denominators, failures, stage counters,
per-file SHA-256 identities, and restore results are recorded in the dataset and
experiment directories. The pilot remains separately identified and is never
added to formal acceptance counts.

## Run result

Collection completed in 1,503.52 s process wall. It accepted 1,000 direct
episodes from 1,000 complete attempts and 100 safe obstacle episodes from 402
complete attempts. Obstacle rejections were 248 finite cylinder candidate
exhaustions, 52 physical collision/crossing failures, and 2 unsuccessful native
templates; no restoration mismatch occurred. Per-stage timings, all 1,402
attempts/seeds, and aggregate counters are in the dataset `report.json`;
`outputs/exp/20261022_132753_pickcube_formal/validation_full_contract/manifest.json`
contains the accepted seed manifest, source fingerprints, and artifact hashes.

Full validation passed on all 1,100 episodes: observation dimension 45, 7D
finite legal actions, T+1 observation/state/terminal arrays, native terminal
success and obstacle safety flags, 1,000 direct / 100 obstacle geometry and
waypoint checks, exact initial observation plus native-state/controller-state
fresh-scene restoration, and 80,003 loaded training windows. The minimum
observed base-to-SG-line distances were 0.06081 m direct and 0.06460 m obstacle;
all 100 obstacle waypoints remained on the Panda-base side. Controller state
snapshots were captured from the live controller API per episode (the measured
snapshot is empty for this controller); its saved identity was verified on
restore. `trajectory.h5` SHA-256 is
`fa846f52b5ae08de0243cfa52b40a4b0e7ce160881aa9db59ebaf4952c02904e` and the
frozen contract SHA-256 is
`e1d98a8934203a2dcb397dc604736cb7877f9134cca4ef84ee502f8325f4f46e`.
The `trajectory.json` SHA-256 is
`8ef0d2739d6bf5675316ab37f42f1437cc19d80739e6f9102973beeabd6d1e58`; the
validation manifest SHA-256 is
`0f50709791258b42e836b4ece5bee0be12d816b2623153a69d578afa949a8b78`; the
representative replay evidence SHA-256 is
`1ce33bd47e91b219948a6fb88a7806bdcc8b848372f4b86ca1c8c8f893f46181`. Five
direct and five obstacle demonstrations replayed their saved native actions
from restored initial states; all ten achieved native success, and all obstacle
replays had zero contact and zero forbidden crossing. No action conversion is
used. Summed complete-attempt elapsed was 892.57 s direct and 601.42 s obstacle;
the process-wall boundary additionally includes startup, artifact writes, and
exit.

The live runtime controller configuration was independently resolved and saved
at `runtime_contract_resolution.json`: position bounds are ±0.1 m and signed
XYZ Euler rotation bounds are ±0.1 rad, with `rot_lower=[-0.1,-0.1,-0.1]`.
That file pins the ManiSkill Panda/controller source bytes; its SHA-256 is
`b965a888d9edc00ffc2f6f0dfb45452048ae21b5b269a2e9cab26685b5d5be3b`.
