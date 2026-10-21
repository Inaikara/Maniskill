# Formal native wristcam dataset — Ticket 02

## Approval and pre-run freeze

**Approval recorded before generation:** the user explicitly approved the pilot
and optimizations and instructed: “OK 现在开始ticket02 按顺序完成后续ticket 如非必要不再需要征求我的意见”. This authorizes this formal Ticket 02 collection and sequential later tickets without a new approval gate. This round performs Ticket 02 only; no training or evaluation is part of this collection round.

The formal generator is frozen at repository revision
`e744d1592dbe1826bb7e04192d57871e0f410bf7` (tree
`97f35b0b7506d223d8d338641d21a7750e2f69b2`). Source SHA-256 values, computed
before generation:

| Source | SHA-256 |
| --- | --- |
| `examples/baselines/constrain/collect_constructive.py` | `b989607c6f0c434ceceba3951ec3d38cee6ec95b29473a1483ab99bc696c2142` |
| `examples/baselines/constrain/stackcube_obstacle.py` | `f5c62090f968fa65fc616171dcab0109452e02d1b4d9dfd47623b3d19fde8d85` |
| `examples/baselines/constrain/obstacle_record.py` | `918e2478da96bda0afdf81f2bd700a13c4e03249c79b8f20c5e626f90f86d3cb` |
| `examples/baselines/constrain/constrain_dataset.py` | `4ffdbaad2d522857129d4d0d68bc2698b793937eba8353be017999a64d73a561` |

The frozen generation distribution and contract are the optimized round-5 pilot
rules in `WRISTCAM_PILOT.md`: native `StackCubeObstacle-v1`, `panda_wristcam`,
native reset pose/noise `0.02`, `state`, `pd_ee_delta_pose`, CPU PhysX, one
environment, 500 control steps, zero commanded rotation, cached full
wristcam-inclusive swept collision footprint, exact cKDTree nearest-point
clearance, least-occupied-side directional prior, and unchanged geometric
blocking/clearance conditions. The expert waypoint/speed/lift parameters and
first `stack_success` stop are unchanged. A success is accepted only when both
collision and crossing flags are false. Each retained trajectory comes from a
distinct successful template with reuse count one.

Frozen configuration fingerprint (canonical `Config.fingerprint()`):
`a5390e1777aa3ea0f708bf8b414d38ac6727ed520c53e3ce53ef9c05da19d140`.
Run-specific limits are target 100, reuse 1, seed 0, maximum 2,000 candidate
templates, maximum 400 physical episodes, maximum 40 cylinder sampler calls
per template/reuse slot, and 500 steps per episode. Collection has a hard
3,600-second wall limit; neither the limits nor the distribution will be
expanded if exhausted. The output is the new, non-overwriting path
`outputs/wristcam_formal_20261021/`.

Runtime freeze in activated `maniskill`: Python 3.11.16, NumPy 1.26.4, SciPy
1.17.1, h5py 3.16.0, Gymnasium 0.29.1. The source and config fingerprints
above must match the executed report. The complete native expert/config values
are embedded in the run's `report.json`.

## Collection and verification

Run date: 2026-10-21. The frozen config fingerprint matched the collector's
`report.json`. The hard timeout was not reached; process wall includes imports,
environment setup, collection, report/HDF5/JSON writes, and exit.

| Stage / measure | Result | Denominator / interpretation |
| --- | ---: | --- |
| Candidate templates | 295 | templates considered before reaching target |
| No-obstacle successful templates | 276 | 276/295 candidates (93.56%) |
| Templates with accepted output / trajectories | 100 / 100 | 100 distinct templates, reuse 1 |
| Cylinder sampling calls | 9,076 | across candidate templates until target reached |
| `no_feasible_cylinder` | 8,976 | 8,976/9,076 calls rejected geometrically |
| Feasible placements selected | 100 | one per output template |
| Obstacle physics attempts / safe successes | 100 / 100 | 100% of selected feasible placements |
| Other failures | 19 | 18 `convergence_timeout`, 1 `grasp_failed`; these account for the 19 candidates without a successful no-obstacle template |
| Sampling calls with feasible output | 100/9,076 (1.10%) | not the candidate-template acceptance rate |
| Sanity / templates / sampling / obstacle stage | 1.25 / 391.52 / 27.44 / 121.58 s | stage timers from `report.json` |
| Collector elapsed | 543.67 s | stops before final report serialization |
| Process wall | 548.94 s | `/usr/bin/time`, start to process exit |

The run reached 100/100 within the frozen 2,000-template, 400-episode,
40-sampler-call-per-template and 3,600-second limits. Stage time sums to
541.79 seconds; the remaining collector time is stage/report orchestration.
This finite run's 100/295 output-template rate is descriptive only. No budgets
or acceptance filters were expanded or relaxed.

`validation.json` records PASS for all 100 records: 100 unique template IDs,
stack success and no obstacle collision/crossing in both report and trajectory
metadata, 7D actions, 51D observations, exact zero rotation channels, and T+1
rows for every recorded state leaf. The existing `ConstrainDemoDataset` loaded
all 100 trajectories (9,461 windows, 51D observations and 4D actions). Episode
lengths are recorded in that file (82–108 steps). No training or evaluation
was run.

### Artifact identities

SHA-256 (exact files under `outputs/wristcam_formal_20261021/`):

| Artifact | SHA-256 |
| --- | --- |
| `trajectory.h5` | `c4811f0d1069fdc4bd247770978a6f87423b9404cfddc04567db456b0ce734c8` |
| `trajectory.json` | `9ebaa31eff4e4ad1f542a097a4624d29a4becdc54da25da58aa5721608282d66` |
| `report.json` | `d935ff3bdc6fa95fea7b9bbac4af7ed41a07b4be19155d29f26807565f4677c5` |
| `validation.json` | `db104dd5ce4d5d47a6a2876a9be5d105f6abfb5432c7d20afeba4722cdcb05fc` |

### Reproduction commands

From repository root, with hard timeouts:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -f 'PROCESS_WALL_SECONDS=%e' \
  timeout --kill-after=5s 3600s python -m examples.baselines.constrain.collect_constructive \
  --output-dir outputs/wristcam_formal_20261021 \
  --target-successes 100 --reuse-count 1 --max-templates 2000 \
  --max-episodes 400 --max-cylinder-attempts 40 --seed 0
```

Validation was run in the activated environment with a 120-second timeout. Its
complete per-trajectory evidence and PASS assertions are retained in
`outputs/wristcam_formal_20261021/validation.json`; the checks call
`validate_aggregate(..., expected_count=100)` and `ConstrainDemoDataset`, then
assert every metadata safety flag, unique template, action/observation shape,
and exact zero rotation for each HDF5 trajectory.

The downstream collection identity is the frozen source/config contract in this
report, not the incidental formal run output path. Subsequent tickets should
retain these generation rules while changing only their explicitly defined
run-specific target/seed as required by their ticket.
