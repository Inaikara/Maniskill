"""Train a fresh official-architecture PlaceSphere Base DP on direct demos."""

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from examples.baselines.constrain.placesphere_base_dp import (
    FORMAL_CONTRACT, FORMAL_H5, FORMAL_MANIFEST, FORMAL_REPORT, MODEL_CONFIG,
    TRUSTED_MANIFEST_SHA256, TRUSTED_REPORT_SHA256, PlaceSphereBaseDataset, build_agent, contract, sha256_file,
    verify_runtime_observation_contract,
)
from examples.baselines.constrain.train_logging import RunLogger, atomic_write_json
from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_H5_SHA256, MATCHED_MANIFEST_SHA256, MATCHED_VERSION, MATCHED_HOLD8_VERSION,
    PlaceSphereManifestDataset, load_matched_training_identity,
)
from examples.baselines.constrain.placesphere_action_codec import (
    IDENTITY as IDENTITY_ACTION_ENCODING, CONTROLLER_RANGE,
    encode_native, validate_profile,
)


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
    p.add_argument("--profile-seconds", type=float, default=0)
    p.add_argument("--matched-r21b", action="store_true")
    p.add_argument("--matched-data-identity", type=Path)
    p.add_argument("--action-encoding", choices=("identity_native_v1", "placesphere_controller_range_v1"),
                   default="identity_native_v1")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args(argv)


def _sha_params(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_scheduler():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")


def _save(path, agent, ema, step, config, selector, action_encoding):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    ema_agent = copy.deepcopy(agent)
    ema.copy_to(ema_agent.parameters())
    torch.save({"format": "placesphere_base_dp_v1", "contract": contract(action_encoding),
                "action_encoding": action_encoding,
                "named_state_schema": config["runtime_observation_schema"],
                "weights": {"ema": {k: v.detach().cpu() for k, v in ema_agent.state_dict().items()},
                            "model": {k: v.detach().cpu() for k, v in agent.state_dict().items()}},
                "step": int(step), "selected_weights": selector,
                "data_identity": config["data"], "training_config": config["training"]}, tmp)
    tmp.replace(path)


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.matched_r21b and args.matched_data_identity:
        raise ValueError("choose one matched dataset identity mode")
    matched_hold8 = args.matched_data_identity is not None
    action_encoding = (CONTROLLER_RANGE if args.action_encoding == CONTROLLER_RANGE["name"]
                       else IDENTITY_ACTION_ENCODING)
    validate_profile(action_encoding)
    identity = (load_matched_training_identity(args.matched_data_identity, args.demo_path, args.manifest,
                                                expected_version=json.loads(args.matched_data_identity.read_text()).get("dataset_version"))
                if matched_hold8 else None)
    if args.matched_r21b:
        if sha256_file(args.demo_path) != MATCHED_H5_SHA256 or sha256_file(args.manifest) != MATCHED_MANIFEST_SHA256:
            raise ValueError("matched PlaceSphere r21b data identity mismatch")
    elif not matched_hold8:
        if sha256_file(args.demo_path) != "fec4620c496ae3c5c0dec7db469333f302f7170363e11f66cba961cbb5d56ec4":
            raise ValueError("formal PlaceSphere HDF5 SHA-256 mismatch")
        if sha256_file(FORMAL_CONTRACT) != "ff11b940ede86e290928d26ccd4ac96e7ba480995af774d46a018dbda8b6e761":
            raise ValueError("formal PlaceSphere frozen contract SHA-256 mismatch")
        if sha256_file(args.manifest) != TRUSTED_MANIFEST_SHA256:
            raise ValueError("formal PlaceSphere manifest SHA-256 mismatch")
        if sha256_file(FORMAL_REPORT) != TRUSTED_REPORT_SHA256:
            raise ValueError("formal PlaceSphere report SHA-256 mismatch")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if not args.profile_seconds and not args.preflight_only:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=False)
    # Snapshot executed local sources before preflight/configuration or updates.
    source_files = [Path(__file__), Path(__import__("examples.baselines.constrain.placesphere_base_dp", fromlist=["__file__"]).__file__),
                    Path(__import__("examples.baselines.constrain.base_dp", fromlist=["__file__"]).__file__),
                     Path(__import__("examples.baselines.constrain.train_logging", fromlist=["__file__"]).__file__),
                      Path(__import__("examples.baselines.constrain.placesphere_manifest_dataset", fromlist=["__file__"]).__file__),
                      Path(__import__("examples.baselines.constrain.placesphere_action_codec", fromlist=["__file__"]).__file__),
                      Path(__import__("examples.baselines.constrain.placesphere_base_dp", fromlist=["OFFICIAL_DP"]).OFFICIAL_DP) / "train.py"]
    snapshot = args.run_dir / "source_snapshot"
    snapshot.mkdir()
    source_identity = {}
    for source in source_files:
        shutil.copy2(source, snapshot / source.name)
        source_identity[str(source.relative_to(Path.cwd()))] = sha256_file(source)
    atomic_write_json(snapshot / "source_hashes.json", source_identity)

    dataset = (PlaceSphereManifestDataset(
        args.demo_path, args.manifest, "direct",
        expected_h5_sha256=identity["h5_sha256"],
        expected_manifest_sha256=identity["manifest_sha256"],
        max_release_hold_steps=8, expected_version=identity["dataset_version"])
        if matched_hold8 else
        PlaceSphereManifestDataset(args.demo_path, args.manifest, "direct")
        if args.matched_r21b else PlaceSphereBaseDataset(args.demo_path, args.manifest))
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
    run_config = {
         "status": "preflight" if args.preflight_only else "profiling" if args.profile_seconds else "running",
         "training": {"seed": args.seed, "planned_updates": args.total_iters,
                     "batch_size": args.batch_size, "optimizer": "AdamW", "betas": [0.95, 0.999],
                     "weight_decay": 1e-6, "learning_rate": args.lr, "warmup_steps": args.warmup,
                     "lr_schedule": "cosine", "ema_power": args.ema_power,
                     "gradient_clip_norm": None, "profile_seconds": args.profile_seconds,
                      "selected_weights": "final EMA fixed before training; no task evaluation",
                      "action_encoding": action_encoding},
         "policy_contract": contract(action_encoding), "action_encoding": action_encoding,
         "runtime_observation_schema": runtime_schema,
         "data": {"path": str(args.demo_path), "manifest": str(args.manifest),
                  "selected_class": "direct", "selected_episode_ids": dataset.trajectory_ids,
                  "episodes": len(dataset.trajectory_ids), "windows": len(dataset),
                  "h5_sha256": sha256_file(args.demo_path), "manifest_sha256": sha256_file(args.manifest),
                   "frozen_contract_sha256": None if (args.matched_r21b or matched_hold8) else sha256_file(FORMAL_CONTRACT),
                   "collection_report_sha256": None if (args.matched_r21b or matched_hold8) else sha256_file(FORMAL_REPORT),
                    **({"dataset_version": MATCHED_VERSION} if args.matched_r21b else {}),
                    **({"dataset_version": identity["dataset_version"], "release_hold_steps": 8,
                       "training_identity_sha256": sha256_file(args.matched_data_identity)} if matched_hold8 else {})},
         "architecture": dict(MODEL_CONFIG), "normalization": {"observation": "identity", "action": action_encoding},
        "outputs": {"experiment": str(args.run_dir), "checkpoint": str(args.checkpoint_dir)},
        "executed_source_sha256": source_identity, "initial_model_sha256": _sha_params(agent),
    }
    atomic_write_json(args.run_dir / "run_config.json", run_config)
    logger = RunLogger(args.run_dir)
    print(f"run config written before updates: {args.run_dir / 'run_config.json'}", flush=True)

    def update(batch):
        obs = batch["observations"].to(args.device)
        actions = encode_native(batch["actions"].to(args.device), action_encoding)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(actions),), device=args.device).long()
        noisy = scheduler.add_noise(actions, noise, timesteps)
        prediction = agent.noise_pred_net(noisy, timesteps, global_cond=obs.flatten(start_dim=1))
        loss = torch.nn.functional.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {len(losses) + 1}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        lr_sched.step(); ema.step(agent.parameters())
        return float(loss.detach().cpu())

    if args.profile_seconds:
        batch = next(iter(loader)); losses = []; started = time.monotonic()
        while time.monotonic() - started < min(args.profile_seconds, 60):
            losses.append(update(batch))
        evidence = {"kind": "non-formal-throughput-probe", "updates": len(losses),
                    "seconds": time.monotonic() - started, "updates_per_second": len(losses)/(time.monotonic()-started),
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                    "loss_first": losses[0], "loss_last": losses[-1], "batch_size": args.batch_size}
        atomic_write_json(args.run_dir / "profile.json", evidence)
        return evidence

    started = time.monotonic(); step = 0; losses = []; last_progress = started
    iterator = iter(loader)
    if args.preflight_only:
        preflight_losses = []
        for _ in range(2):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            preflight_losses.append(update(batch))
        report = {"status": "preflight_complete", "updates": 2,
                  "all_losses_finite": bool(np.isfinite(preflight_losses).all()),
                  "losses": preflight_losses, "action_encoding": action_encoding,
                  "dataset_h5_sha256": sha256_file(args.demo_path),
                  "dataset_manifest_sha256": sha256_file(args.manifest)}
        atomic_write_json(args.run_dir / "preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    while step < args.total_iters:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        step += 1
        loss = update(batch); losses.append(loss)
        logger.log_step(step, loss, lr=float(lr_sched.get_last_lr()[0]))
        now = time.monotonic()
        if now - last_progress >= 30 or step == args.total_iters:
            logger.log_progress(step, args.total_iters, {"elapsed_seconds": now-started,
                                "loss_last": loss, "mean_loss_last100": float(np.mean(losses[-100:]))})
            print(f"step {step}/{args.total_iters} loss={loss:.6f} elapsed={now-started:.1f}s", flush=True)
            last_progress = now
    final_path = args.checkpoint_dir / "final.pt"
    _save(final_path, agent, ema, step, run_config, "ema", action_encoding)
    summary = {"status": "complete", "actual_updates": step,
               "all_losses_finite": bool(losses) and bool(np.isfinite(losses).all()),
               "loss_count": len(losses), "loss_first": losses[0], "loss_last": losses[-1],
               "loss_min": float(np.min(losses)), "loss_max": float(np.max(losses)),
               "elapsed_seconds": time.monotonic()-started, "checkpoint": str(final_path),
               "checkpoint_sha256": sha256_file(final_path), "initial_model_sha256": run_config["initial_model_sha256"],
               "final_model_sha256": _sha_params(agent), "evaluation_weights": "EMA",
               "config_sha256": sha256_file(args.run_dir / "run_config.json")}
    atomic_write_json(args.run_dir / "train_summary.json", summary)
    logger.log_progress(step, args.total_iters, summary)
    run_config["status"] = "complete"; atomic_write_json(args.run_dir / "run_config.json", run_config)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
