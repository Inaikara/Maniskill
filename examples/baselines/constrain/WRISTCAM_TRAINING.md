# Formal wristcam ConstrainNet retraining — Ticket 04

## Run identity and configuration

Completed a fresh run on 2026-10-21 from the original Base DP `ema_agent`; no
Panda ConstrainNet checkpoint was resumed. Run artifacts are in
`outputs/constrain_runs/wristcam_full30000_seed1/` (ignored runtime data).
The formal source contains all 100 demonstrations and 9,461 official windows.

| Identity | Value |
| --- | --- |
| Dataset `trajectory.h5` SHA-256 | `c4811f0d1069fdc4bd247770978a6f87423b9404cfddc04567db456b0ce734c8` |
| Paired `trajectory.json` SHA-256 | `9ebaa31eff4e4ad1f542a097a4624d29a4becdc54da25da58aa5721608282d66` |
| Collection report SHA-256 | `d935ff3bdc6fa95fea7b9bbac4af7ed41a07b4be19155d29f26807565f4677c5` |
| Generator revision / tree | `e744d1592dbe1826bb7e04192d57871e0f410bf7` / `97f35b0b7506d223d8d338641d21a7750e2f69b2` |
| Frozen generator config fingerprint | `a5390e1777aa3ea0f708bf8b414d38ac6727ed520c53e3ce53ef9c05da19d140` |
| Base checkpoint | `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt`, subkey `ema_agent` |
| Base checkpoint file SHA-256 | `ba65c507ac48aa76a911fa1af28b7155ca3c4f2d7393c775930241dbcde27c25` |
| Frozen base state sha16 | `8010257bd96dc866` |

`provenance.json` records the dataset/report identities, frozen generator source
hashes and configuration, native environment contract, and base checkpoint
identity. The same compact object is embedded in `run_config.json` and the
standalone final checkpoint.

Native collection contract: `StackCubeObstacle-v1`, `panda_wristcam`, reset
noise `0.02`, `state`, `pd_ee_delta_pose`, CPU PhysX, one environment, 500-step
episode cap, zero commanded rotation. Generation source SHA-256 values match
the Ticket 02 freeze in `WRISTCAM_DATASET.md`.

Effective training configuration: fresh seed 1; 30,000 updates; batch 1,024;
AdamW learning rate `1e-4`, betas `(0.95, 0.999)`, weight decay `1e-6`; cosine
schedule with 500 warmup steps; EMA power `0.75`; CUDA RTX 5070; save interval
1,000. Model, action/observation interface, and normalization are the existing
ConstrainNet configuration. No hyperparameter selection or scene evaluation
was performed.

## Completion and evidence

Training exited successfully after 893.95 seconds process wall (890.3 seconds
reported loop elapsed), at exactly step 30,000. All 30,000 per-update losses
are in `logs/loss.jsonl` and finite.

| Loss statistic | Value |
| --- | ---: |
| First update | 0.20492896437644958 |
| Final update | 0.0029789158143103123 |
| Mean across 30,000 | 0.009373630191331419 |
| Mean over final 1,000 | 0.0032573870706837623 |
| Range | 0.002273283898830414 – 0.24043482542037964 |

Final EMA standalone checkpoint:
`outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt`
(74,295,931 bytes, SHA-256
`74de460532bf4188b7b019f9bb15ecd027af2a7a3b5bef14c2fc6fd4c59b7d02`).
Independent CPU load confirms step 30,000. The base state sha16 matches the
independently loaded original checkpoint exactly; disabled epsilon and rollout
outputs match exactly (`max_abs_diff=0`). The enabled control branch changes
epsilon output (mean absolute difference `0.1492032409`), and enabled EMA
inference returned finite actions of shape `[1, 8, 4]`. Training evidence also
confirms finite loss, moved zero-convolution weights, nonzero upstream control
and hint-encoder gradients, and unchanged frozen base. Full evidence is in
`verify_evidence.json` and `train_summary.json`.

## Reproduction commands

Run from the repository root after activating `maniskill`; all commands use
finite timeouts:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=10s 10800s \
  python -m examples.baselines.constrain.train_constrain \
  --demo-path outputs/wristcam_formal_20261021/trajectory.h5 \
  --base-checkpoint outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt \
  --run-dir outputs/constrain_runs/wristcam_full30000_seed1 \
  --provenance-json outputs/constrain_runs/wristcam_full30000_seed1/provenance.json \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 \
  --lr 1e-4 --warmup 500 --ema-power 0.75 --save-every 1000
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=10s 300s \
  python -m examples.baselines.constrain.verify_constrain \
  --checkpoint outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt \
  --base-checkpoint outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt \
  --device cpu
```

Verification ran no scenes and selected no checkpoint by evaluation performance.
