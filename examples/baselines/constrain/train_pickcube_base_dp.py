"""Train a fresh official-architecture PickCube Base DP on direct demos only."""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from examples.baselines.constrain.output_paths import CKPT_ROOT, experiment_dir
from examples.baselines.constrain.pickcube_base_dp import (
    FORMAL_CONTRACT, FORMAL_H5, FORMAL_MANIFEST, FORMAL_REPORT, MODEL_CONFIG,
    PickCubeBaseDataset, build_agent, contract, sha256_file,
    verify_runtime_observation_contract,
)
from examples.baselines.constrain.train_logging import RunLogger, atomic_write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-path", type=Path, default=FORMAL_H5)
    p.add_argument("--manifest", type=Path, default=FORMAL_MANIFEST)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-iters", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema-power", type=float, default=0.75)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--profile-seconds", type=float, default=0,
                   help="bounded standalone throughput/memory probe; performs no formal updates")
    return p.parse_args(argv)


def _sha_params(module):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_scheduler():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if not args.profile_seconds:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=False)
    dataset = PickCubeBaseDataset(args.demo_path, args.manifest)
    runtime_schema = verify_runtime_observation_contract()
    agent = build_agent(args.device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    lr_sched = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=args.warmup,
                             num_training_steps=args.total_iters)
    ema = EMAModel(parameters=agent.parameters(), power=args.ema_power)
    scheduler = make_scheduler()
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=RandomSampler(dataset),
                        drop_last=True, num_workers=0, pin_memory=args.device.startswith("cuda"))
    batch = next(iter(loader))
    # Run config is durably fixed before any optimizer update.
    run_config = {
        "status": "profiling" if args.profile_seconds else "running",
        "training": {"seed": args.seed, "planned_updates": args.total_iters,
                     "batch_size": args.batch_size, "optimizer": "AdamW",
                     "betas": [0.95, 0.999], "weight_decay": 1e-6,
                     "learning_rate": args.lr, "warmup_steps": args.warmup,
                     "lr_schedule": "cosine", "ema_power": args.ema_power,
                     "gradient_clip_norm": None, "profile_seconds": args.profile_seconds,
                     "selected_weights": "final EMA fixed before training; no task evaluation"},
        "policy_contract": contract(),
        "runtime_observation_schema": runtime_schema,
        "data": {"path": str(args.demo_path), "manifest": str(args.manifest),
                 "selected_class": "direct", "selected_episode_ids": dataset.trajectory_ids,
                 "episodes": len(dataset.trajectory_ids), "windows": len(dataset),
                 "h5_sha256": sha256_file(args.demo_path),
                 "manifest_sha256": sha256_file(args.manifest),
                 "frozen_contract_sha256": sha256_file(FORMAL_CONTRACT),
                 "collection_report_sha256": sha256_file(FORMAL_REPORT)},
        "architecture": dict(MODEL_CONFIG),
        "normalization": {"observation": "identity", "action": "identity"},
        "outputs": {"experiment": str(args.run_dir), "checkpoint": str(args.checkpoint_dir)},
        "initial_model_sha256": _sha_params(agent),
    }
    atomic_write_json(args.run_dir / "run_config.json", run_config)
    logger = RunLogger(args.run_dir)
    print(f"run config written before updates: {args.run_dir / 'run_config.json'}", flush=True)

    def update(batch, step):
        obs = batch["observations"].to(args.device, non_blocking=True)
        actions = batch["actions"].to(args.device, non_blocking=True)
        cond = obs.flatten(start_dim=1)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(actions),), device=args.device).long()
        noisy = scheduler.add_noise(actions, noise, timesteps)
        prediction = agent.noise_pred_net(noisy, timesteps, global_cond=cond)
        loss = torch.nn.functional.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        lr_sched.step(); ema.step(agent.parameters())
        return float(loss.detach().cpu())

    if args.profile_seconds:
        started = time.monotonic(); updates = 0; losses = []
        while time.monotonic() - started < min(args.profile_seconds, 60):
            losses.append(update(batch, updates + 1)); updates += 1
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated()
            else:
                peak = None
        evidence = {"kind": "non-formal-throughput-probe", "updates": updates,
                    "seconds": time.monotonic()-started, "updates_per_second": updates/(time.monotonic()-started),
                    "peak_cuda_bytes": peak, "loss_first": losses[0], "loss_last": losses[-1],
                    "batch_size": args.batch_size}
        atomic_write_json(args.run_dir / "profile.json", evidence)
        print(json.dumps(evidence, indent=2), flush=True)
        return evidence

    started = time.monotonic(); step = 0; losses = []; last_progress = started
    iterator = iter(loader)
    while step < args.total_iters:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        step += 1
        loss = update(batch, step); losses.append(loss)
        logger.log_step(step, loss, lr=float(lr_sched.get_last_lr()[0]))
        now = time.monotonic()
        if now-last_progress >= 30 or step == args.total_iters:
            logger.log_progress(step, args.total_iters,
                                {"elapsed_seconds": now-started, "loss_last": loss,
                                 "mean_loss_last100": float(np.mean(losses[-100:]))})
            print(f"step {step}/{args.total_iters} loss={loss:.6f} elapsed={now-started:.1f}s", flush=True)
            last_progress = now
        if step % args.save_every == 0:
            _save(args.checkpoint_dir / "periodic.pt", agent, ema, step, run_config, "model")

    final_path = args.checkpoint_dir / "final.pt"
    _save(final_path, agent, ema, step, run_config, "ema")
    summary = {"status": "complete", "actual_updates": step, "all_losses_finite": bool(losses) and bool(np.isfinite(losses).all()),
               "loss_count": len(losses), "loss_first": losses[0], "loss_last": losses[-1],
               "loss_min": float(np.min(losses)), "loss_max": float(np.max(losses)),
               "elapsed_seconds": time.monotonic()-started, "checkpoint": str(final_path),
               "checkpoint_sha256": sha256_file(final_path), "initial_model_sha256": run_config["initial_model_sha256"],
               "final_model_sha256": _sha_params(agent), "evaluation_weights": "EMA", "config_sha256": sha256_file(args.run_dir / "run_config.json")}
    atomic_write_json(args.run_dir / "train_summary.json", summary)
    logger.log_progress(step, args.total_iters, summary)
    run_config["status"] = "complete"; atomic_write_json(args.run_dir / "run_config.json", run_config)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _save(path, agent, ema, step, config, selector):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    model_state = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
    ema_agent = copy.deepcopy(agent)
    ema.copy_to(ema_agent.parameters())
    ema_state = {k: v.detach().cpu() for k, v in ema_agent.state_dict().items()}
    torch.save({"format": "pickcube_base_dp_v1", "contract": contract(),
                "named_state_schema": config["runtime_observation_schema"],
                "weights": {"model": model_state, "ema": ema_state},
                "step": int(step), "selected_weights": selector,
                "data_identity": config["data"], "training_config": config["training"]}, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
