"""Matched PickCube Round 37 z-score training and runtime profile.

This profile is opt-in and has checkpoint formats distinct from legacy identity
normalization artifacts. Raw environment observations and seven native actions
remain untouched at the environment boundary.
"""

import copy
import argparse
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
from examples.baselines.constrain.pickcube_base_dp import (
    MODEL_CONFIG as BASE_CONFIG, NAMED_FIELDS, build_agent, contract, sha256_file,
)
from examples.baselines.constrain.pickcube_observation_norm import (
    DIRECT_IDS, PROFILE, normalize_observation, validate_stats,
)
from examples.baselines.constrain.materialize_pickcube_fullscene500 import PickCubeFullScene500Dataset

BASE_FORMAT = "pickcube_direct_zscore_base_v1"
CONSTRAIN_FORMAT = "pickcube_direct_zscore_constrain_v1"
DATA_H5 = "demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.h5"
DATA_MANIFEST = "demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.json"
DATA_H5_SHA = "0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f"
DATA_MANIFEST_SHA = "6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486"
BASE_UPDATES = 30000
CONSTRAIN_UPDATES = 150000
TRAINING = {"seed": 1, "batch_size": 1024, "lr": 1e-4, "warmup": 500,
            "betas": [0.95, 0.999], "weight_decay": 1e-6, "ema_power": 0.75,
            "selected_weights": "final EMA fixed before training", "loss_weight": None}
DIFFUSION = {"training_scheduler": "DDPMScheduler", "inference_scheduler": "DDPMScheduler",
             "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
             "prediction_type": "epsilon", "num_train_timesteps": 100,
             "inference_steps": 100, "executed_prediction_slice": [1, 9],
              "execution_stride": 8}
EPISODE_TIMEOUT_SECONDS = 120
CONSTRAIN_CONFIG = {"input_dim": 7, "global_cond_dim": 84,
                    "diffusion_step_embed_dim": 64, "down_dims": [64, 128, 256],
                    "n_groups": 8, "cond_dim": 3, "num_diffusion_iters": 100}
POLICY = {"robot": "panda", "obs_dim": 42, "obstacle_obs_dim": 45, "act_dim": 7,
          "obs_horizon": 2, "pred_horizon": 16, "act_horizon": 8,
          "control_mode": "pd_ee_delta_pose", "observation_mode": "state",
          "named_state_fields": NAMED_FIELDS}


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_stats(path, expected_sha=None):
    path = Path(path)
    if expected_sha and sha256_file(path) != expected_sha:
        raise ValueError("normalizer artifact file hash differs from registration")
    stats = validate_stats(json.loads(path.read_text()))
    if stats["source_identity"]["h5_sha256"] != DATA_H5_SHA or stats["source_identity"]["manifest_sha256"] != DATA_MANIFEST_SHA:
        raise ValueError("normalizer source is not the frozen Round 37 dataset")
    return stats


def native_dataset(kind, include_obstacle=False):
    return PickCubeFullScene500Dataset(
        DATA_H5, DATA_MANIFEST, expected_h5_sha256=DATA_H5_SHA,
        expected_manifest_sha256=DATA_MANIFEST_SHA, kind=kind,
        include_obstacle=include_obstacle,
    )


def prepare_control_observation(raw_observation, stats):
    """Encode base features once while deriving the hint from raw 45-D state."""
    from examples.baselines.constrain.pickcube_constrain import (
        PickCubeConditionNorm, pickcube_relative_condition,
    )
    raw = torch.as_tensor(raw_observation, dtype=torch.float32)
    if raw.shape[-1] != 45:
        raise ValueError("Constrain input preparation requires raw native 45-D observations")
    encoded = normalize_observation(raw[..., :42], stats)
    hint = pickcube_relative_condition(raw, PickCubeConditionNorm())
    return encoded, hint


def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_base(path, agent, ema, optimizer, lr_sched, step, stats, data_identity,
               provenance, training_augmentation=None, augmentation_generator=None,
               augmentation_delta_stats=None):
    shadow = copy.deepcopy(agent)
    ema.copy_to(shadow.parameters())
    payload = {
        "format": BASE_FORMAT, "profile": PROFILE, "contract": contract(),
        "model_config": dict(BASE_CONFIG), "normalizer": stats,
        "normalizer_sha256": stats["sha256"], "data_identity": data_identity,
        "normalization": {"observation_profile": PROFILE,
                          "observation_stats_sha256": stats["sha256"],
                          "action": "identity native seven-channel"},
        "action_encoding": None, "action_encoding_identity": True,
        "training": dict(TRAINING, updates=BASE_UPDATES), "step": step,
        "training_augmentation": training_augmentation,
        "augmentation_rng_state": (augmentation_generator.get_state()
                                   if augmentation_generator is not None else None),
        "augmentation_delta_stats": augmentation_delta_stats,
        "selected_weights": "ema", "weights": {
            "model": {k: v.detach().cpu() for k, v in agent.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in shadow.state_dict().items()},
        }, "optimizer": optimizer.state_dict(), "lr_scheduler": lr_sched.state_dict(),
        "ema": ema.state_dict(), "rng": _rng_state(), "provenance": provenance,
    }
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp); os.replace(tmp, path)


def _base_contract(ck, stats, require_final=True, training_augmentation=None):
    from examples.baselines.constrain.pickcube_xy_gauge import validate_checkpoint_metadata
    validate_checkpoint_metadata(ck.get("training_augmentation"), training_augmentation)
    identity = ck.get("data_identity", {})
    if (ck.get("format") != BASE_FORMAT or ck.get("profile") != PROFILE
            or ck.get("contract") != contract() or ck.get("model_config") != BASE_CONFIG
            or ck.get("normalizer_sha256") != stats["sha256"]
            or ck.get("normalizer") != stats or identity.get("h5_sha256") != DATA_H5_SHA
            or ck.get("normalization") != {"observation_profile": PROFILE,
                                            "observation_stats_sha256": stats["sha256"],
                                            "action": "identity native seven-channel"}
            or ck.get("action_encoding") is not None or ck.get("action_encoding_identity") is not True
            or identity.get("manifest_sha256") != DATA_MANIFEST_SHA
            or identity.get("selected_class") != "direct"
            or identity.get("selected_episode_ids") != DIRECT_IDS
            or identity.get("episodes") != 1000 or identity.get("windows") != 83004
            or ck.get("training") != dict(TRAINING, updates=BASE_UPDATES)
            or ck.get("training_augmentation") != training_augmentation
            or (training_augmentation is not None and ck.get("augmentation_rng_state") is None)
            or (training_augmentation is not None and ck.get("augmentation_delta_stats") is None)
            or ck.get("selected_weights") != "ema"
            or (require_final and ck.get("step") != BASE_UPDATES)):
        raise ValueError("normalized PickCube Base checkpoint contract mismatch")
    return ck


def _make_schedule(total_updates, warmup, optimizer):
    from diffusers.optimization import get_scheduler
    return get_scheduler("cosine", optimizer=optimizer,
                         num_warmup_steps=warmup, num_training_steps=total_updates)


def train_base(stats_path, run_dir, checkpoint_dir, device="cuda", resume=None,
               total_updates=BASE_UPDATES, provenance=None, training_augmentation=None):
    stats = load_stats(stats_path)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=resume is not None)
    checkpoint_dir.mkdir(parents=True, exist_ok=resume is not None)
    if not resume:
        random.seed(TRAINING["seed"]); np.random.seed(TRAINING["seed"])
        torch.manual_seed(TRAINING["seed"])
        if str(device).startswith("cuda"): torch.cuda.manual_seed_all(TRAINING["seed"])
    augmentation_generator = None
    if training_augmentation is not None:
        from examples.baselines.constrain.pickcube_xy_gauge import (
            empty_delta_stats, make_generator, validate_metadata,
        )
        validate_metadata(training_augmentation)
        augmentation_generator = make_generator(TRAINING["seed"])
        augmentation_delta_stats = empty_delta_stats()
    else:
        augmentation_delta_stats = None
    dataset = native_dataset("direct")
    if len(dataset) != 83004:
        raise ValueError(f"direct window count mismatch: {len(dataset)} != 83004")
    runtime = build_agent(device)
    optimizer = torch.optim.AdamW(runtime.parameters(), lr=TRAINING["lr"],
                                  betas=tuple(TRAINING["betas"]), weight_decay=TRAINING["weight_decay"])
    lr_sched = _make_schedule(total_updates, TRAINING["warmup"], optimizer)
    from diffusers.training_utils import EMAModel
    ema = EMAModel(parameters=runtime.parameters(), power=TRAINING["ema_power"])
    start_step = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        _base_contract(ck, stats, require_final=False,
                       training_augmentation=training_augmentation)
        runtime.load_state_dict(ck["weights"]["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"]); lr_sched.load_state_dict(ck["lr_scheduler"])
        ema.load_state_dict(ck["ema"]); start_step = int(ck["step"]); _restore_rng(ck["rng"])
        if augmentation_generator is not None:
            augmentation_generator.set_state(ck["augmentation_rng_state"])
            augmentation_delta_stats = ck["augmentation_delta_stats"]
    scheduler = _training_scheduler()
    loader = DataLoader(dataset, batch_size=TRAINING["batch_size"],
                        sampler=RandomSampler(dataset), drop_last=True, num_workers=0,
                        pin_memory=str(device).startswith("cuda"))
    identity = {"selected_class": "direct", "selected_episode_ids": DIRECT_IDS,
                "episodes": 1000, "windows": len(dataset), "h5_sha256": DATA_H5_SHA,
                "manifest_sha256": DATA_MANIFEST_SHA, "data_path": DATA_H5,
                "manifest_path": DATA_MANIFEST}
    provenance = provenance or {}
    cfg = {"profile": PROFILE, "normalizer_sha256": stats["sha256"],
           "normalizer": stats, "training": dict(TRAINING, updates=total_updates),
            "data_identity": identity, "initial_model_sha256": state_sha(runtime.state_dict()),
            "training_augmentation": training_augmentation,
           "provenance": provenance,
           "outputs": {"run_dir": str(run_dir), "checkpoint_dir": str(checkpoint_dir)}}
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    diffusion = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, prediction_type="epsilon")
    log = (run_dir / "loss.jsonl").open("a" if start_step else "w")
    iterator = iter(loader); start = time.monotonic(); losses = []
    for step in range(start_step + 1, total_updates + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        raw_observations = batch["observations"].to(device)
        if augmentation_generator is not None:
            from examples.baselines.constrain.pickcube_xy_gauge import (
                augment_window_observations, update_delta_stats,
            )
            raw_observations, delta = augment_window_observations(raw_observations, augmentation_generator)
            augmentation_delta_stats = update_delta_stats(augmentation_delta_stats, delta)
        obs = normalize_observation(raw_observations, stats)
        actions = batch["actions"].to(device)
        noise = torch.randn_like(actions); t = torch.randint(0, 100, (len(actions),), device=device)
        loss = torch.nn.functional.mse_loss(
            runtime.noise_pred_net(diffusion.add_noise(actions, noise, t), t,
                                   global_cond=obs.flatten(1)), noise)
        if not torch.isfinite(loss): raise FloatingPointError(f"Base nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); lr_sched.step()
        ema.step(runtime.parameters()); value = float(loss.detach().cpu()); losses.append(value)
        log.write(json.dumps({"step": step, "loss": value,
                              "lr": lr_sched.get_last_lr()[0]}) + "\n")
        if step % 100 == 0: log.flush()
        if step % 10000 == 0 or step == total_updates:
            log.flush(); _save_base(checkpoint_dir / f"recovery_{step:06d}.pt", runtime,
                                    ema, optimizer, lr_sched, step, stats, identity, provenance,
                                    training_augmentation, augmentation_generator,
                                    augmentation_delta_stats)
            print(f"Base {step}/{total_updates} loss={value:.6f}", flush=True)
    log.close()
    _save_base(checkpoint_dir / "final.pt", runtime, ema, optimizer, lr_sched,
               total_updates, stats, identity, provenance, training_augmentation,
               augmentation_generator, augmentation_delta_stats)
    _base_contract(torch.load(checkpoint_dir / "final.pt", map_location="cpu", weights_only=False),
                   stats, training_augmentation=training_augmentation)
    result = {"step": total_updates, "elapsed_seconds": time.monotonic()-start,
            "checkpoint": str(checkpoint_dir / "final.pt"),
            "checkpoint_sha256": sha256_file(checkpoint_dir / "final.pt"),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "all_losses_finite": all(np.isfinite(losses))}
    if augmentation_delta_stats is not None:
        from examples.baselines.constrain.pickcube_xy_gauge import finalize_delta_stats
        result["augmentation_delta_stats"] = finalize_delta_stats(augmentation_delta_stats)
    return result


def _training_scheduler():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")


def preflight_base(stats_path, output, device="cpu", training_augmentation=None):
    """Exercise the production Base loader/model/loss seam on one real batch."""
    stats = load_stats(stats_path)
    dataset = native_dataset("direct")
    if len(dataset) != 83004:
        raise ValueError("Base preflight direct-window count differs from frozen contract")
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    raw, actions = batch["observations"].to(device), batch["actions"].to(device)
    if training_augmentation is not None:
        from examples.baselines.constrain.pickcube_xy_gauge import (
            augment_window_observations, make_generator, validate_metadata,
        )
        validate_metadata(training_augmentation)
        raw, sampled_delta = augment_window_observations(raw, make_generator(TRAINING["seed"]))
    else:
        sampled_delta = None
    normalized = normalize_observation(raw, stats)
    agent = build_agent(device)
    scheduler = _training_scheduler()
    noise = torch.zeros_like(actions); timesteps = torch.zeros(len(actions), dtype=torch.long, device=device)
    eps = agent.noise_pred_net(scheduler.add_noise(actions, noise, timesteps), timesteps,
                               global_cond=normalized.flatten(1))
    evidence = {"profile": PROFILE, "normalizer_sha256": stats["sha256"],
                "planned_training": {"updates": BASE_UPDATES, **TRAINING},
                "data_identity": {"h5_sha256": DATA_H5_SHA, "manifest_sha256": DATA_MANIFEST_SHA,
                                  "class": "direct", "first_ids": [0, 1]},
                "raw_obs_shape": list(raw.shape), "normalized_obs_shape": list(normalized.shape),
                "action_shape": list(actions.shape), "epsilon_shape": list(eps.shape),
                "finite": bool(torch.isfinite(normalized).all() and torch.isfinite(eps).all()),
                 "one_normalization_application": True}
    if sampled_delta is not None:
        evidence["training_augmentation"] = training_augmentation
        evidence["delta_min_m"] = sampled_delta.min(dim=0).values.tolist()
        evidence["delta_max_m"] = sampled_delta.max(dim=0).values.tolist()
    if not evidence["finite"] or normalized.shape[-1] != 42 or eps.shape != actions.shape:
        raise RuntimeError(f"Base real-minibatch preflight failed: {evidence}")
    Path(output).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def preflight_control(stats_path, base_checkpoint, output, device="cpu",
                      training_augmentation=None):
    """Validate typed Base metadata and run real raw-obstacle/hint minibatch."""
    from examples.baselines.constrain.pickcube_constrain import (
        MODEL_CONFIG as PICKCUBE_CONTROL_CONFIG, PickCubeConditionNorm,
        pickcube_relative_condition,
    )
    stats = load_stats(stats_path)
    base_ck = _base_contract(torch.load(base_checkpoint, map_location="cpu", weights_only=False),
                             stats, training_augmentation=training_augmentation)
    dataset = native_dataset("obstacle", include_obstacle=True)
    if len(dataset) != 45840:
        raise ValueError("Constrain preflight obstacle-window count differs from frozen contract")
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    raw, actions = batch["observations"].to(device), batch["actions"].to(device)
    if raw.shape[-1] != 45:
        raise RuntimeError("preflight loader must preserve raw 45-D observations for the hint")
    original_hint = pickcube_relative_condition(raw, PickCubeConditionNorm())
    if training_augmentation is not None:
        from examples.baselines.constrain.pickcube_xy_gauge import (
            augment_window_observations, make_generator, validate_metadata,
        )
        validate_metadata(training_augmentation)
        raw, sampled_delta = augment_window_observations(raw, make_generator(TRAINING["seed"]))
    else:
        sampled_delta = None
    normalized, transformed_hint = prepare_control_observation(raw, stats)
    hint_condition = original_hint
    hint_invariant_error = float((hint_condition - original_hint).abs().max())
    hint = hint_condition[:, None, :].expand(-1, 16, -1)
    base_agent = build_agent(device); base_agent.load_state_dict(base_ck["weights"]["ema"], strict=True)
    denoiser = build_constrain_denoiser(PICKCUBE_CONTROL_CONFIG, device=device)
    denoiser.base_net.load_state_dict(base_agent.noise_pred_net.state_dict(), strict=True)
    scheduler = _training_scheduler(); noise = torch.zeros_like(actions)
    timesteps = torch.zeros(len(actions), dtype=torch.long, device=device)
    noised = scheduler.add_noise(actions, noise, timesteps)
    epsilon = denoiser(noised, timesteps, global_cond=normalized.flatten(1), hint=hint)
    # Independent standard Base and zero-control path share this normalized condition.
    probe = torch.randn(2,16,7,device=device); t = torch.tensor([13,71],device=device)
    cond = normalized.flatten(1)
    with torch.no_grad():
        reference = base_agent.noise_pred_net(probe,t,global_cond=cond)
        disabled = denoiser(probe,t,global_cond=cond,hint=hint,enabled=False)
    evidence = {"profile": PROFILE, "normalizer_sha256": stats["sha256"],
                "planned_training": {"updates": CONSTRAIN_UPDATES, **TRAINING},
                "base_checkpoint_sha256": sha256_file(base_checkpoint),
                "base_format": base_ck["format"], "base_step": base_ck["step"],
                "data_identity": {"h5_sha256": DATA_H5_SHA, "manifest_sha256": DATA_MANIFEST_SHA,
                                  "class": "obstacle", "first_ids": [1000,1001]},
                "raw_obs_shape": list(raw.shape), "normalized_obs_shape": list(normalized.shape),
                "hint_shape": list(hint.shape), "actions_shape": list(actions.shape),
                "epsilon_shape": list(epsilon.shape), "hint_source": "raw native 45-D latest frame",
                 "disabled_control_max_abs_diff": float((reference-disabled).abs().max()),
                 "hint_invariance_max_abs_diff": hint_invariant_error,
                 "transformed_hint_max_abs_diff": float((transformed_hint - original_hint).abs().max()),
                 "finite": bool(torch.isfinite(epsilon).all())}
    if sampled_delta is not None:
        evidence["training_augmentation"] = training_augmentation
        evidence["delta_min_m"] = sampled_delta.min(dim=0).values.tolist()
        evidence["delta_max_m"] = sampled_delta.max(dim=0).values.tolist()
    if (normalized.shape[-1] != 42 or hint.shape != (len(raw),16,3)
            or not evidence["finite"] or evidence["disabled_control_max_abs_diff"] > 1e-6
            or evidence["hint_invariance_max_abs_diff"] > 1e-6):
        raise RuntimeError(f"Constrain real-minibatch preflight failed: {evidence}")
    Path(output).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def _save_control(path, denoiser, ema, optimizer, lr_sched, step, rng, config,
                  augmentation_generator=None, augmentation_delta_stats=None):
    from examples.baselines.constrain.constrain_ckpt import save_constrain_checkpoint
    if augmentation_generator is not None:
        rng = dict(rng, gauge_augmentation=augmentation_generator.get_state(),
                   gauge_delta_stats=augmentation_delta_stats)
    return save_constrain_checkpoint(path, denoiser, ema, optimizer, lr_sched, step,
                                     rng, config, checkpoint_format=CONSTRAIN_FORMAT)


def train_constrain(stats_path, base_checkpoint, run_dir, checkpoint_dir,
                    device="cuda", resume=None, total_updates=CONSTRAIN_UPDATES,
                    provenance=None, training_augmentation=None):
    from examples.baselines.constrain.pickcube_constrain import (
        MODEL_CONFIG as PICKCUBE_CONTROL_CONFIG, PickCubeConditionNorm,
        pickcube_relative_condition,
    )
    stats = load_stats(stats_path)
    base_ck = _base_contract(torch.load(base_checkpoint, map_location="cpu", weights_only=False),
                             stats, training_augmentation=training_augmentation)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=resume is not None)
    checkpoint_dir.mkdir(parents=True, exist_ok=resume is not None)
    if not resume:
        random.seed(TRAINING["seed"]); np.random.seed(TRAINING["seed"])
        torch.manual_seed(TRAINING["seed"])
        if str(device).startswith("cuda"): torch.cuda.manual_seed_all(TRAINING["seed"])
    augmentation_generator = None
    if training_augmentation is not None:
        from examples.baselines.constrain.pickcube_xy_gauge import (
            empty_delta_stats, make_generator, validate_metadata,
        )
        validate_metadata(training_augmentation)
        augmentation_generator = make_generator(TRAINING["seed"])
        augmentation_delta_stats = empty_delta_stats()
    else:
        augmentation_delta_stats = None
    dataset = native_dataset("obstacle", include_obstacle=True)
    if len(dataset) != 45840:
        raise ValueError(f"obstacle window count mismatch: {len(dataset)} != 45840")
    agent = build_agent("cpu"); agent.load_state_dict(base_ck["weights"]["ema"], strict=True)
    denoiser = build_constrain_denoiser(PICKCUBE_CONTROL_CONFIG, device="cpu")
    denoiser.base_net.load_state_dict(agent.noise_pred_net.state_dict(), strict=True)
    denoiser.to(device)
    frozen_base_sha = state_sha(denoiser.base_net.state_dict())
    params = denoiser.trainable_parameters()
    if any(p.requires_grad for p in denoiser.base_net.parameters()):
        raise RuntimeError("matched Base branch must remain frozen")
    optimizer = torch.optim.AdamW(params, lr=TRAINING["lr"], betas=tuple(TRAINING["betas"]),
                                  weight_decay=TRAINING["weight_decay"])
    lr_sched = _make_schedule(total_updates, TRAINING["warmup"], optimizer)
    from diffusers.training_utils import EMAModel
    ema = EMAModel(parameters=params, power=TRAINING["ema_power"])
    step0 = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        _validate_control(ck, stats, base_checkpoint, require_final=False,
                          training_augmentation=training_augmentation)
        denoiser.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"]); lr_sched.load_state_dict(ck["lr_scheduler"])
        ema.load_state_dict(ck["ema"]); step0 = int(ck["step"]); _restore_rng(ck["rng"].get("round37", {}))
        if augmentation_generator is not None:
            augmentation_generator.set_state(ck["rng"]["gauge_augmentation"])
            augmentation_delta_stats = ck["rng"]["gauge_delta_stats"]
    norm = PickCubeConditionNorm()
    data_identity = {"selected_class": "obstacle", "selected_episode_ids": list(range(1000,1500)),
                     "episodes": 500, "windows": len(dataset), "h5_sha256": DATA_H5_SHA,
                     "manifest_sha256": DATA_MANIFEST_SHA, "data_path": DATA_H5,
                     "manifest_path": DATA_MANIFEST}
    provenance = provenance or {}
    config = {
        "profile": PROFILE, "model": dict(PICKCUBE_CONTROL_CONFIG),
        "policy": POLICY,
        "normalization": {"observation_profile": PROFILE, "observation_stats": stats,
                          "observation_stats_sha256": stats["sha256"],
                          "action": "identity native seven-channel" , "hint": norm.to_dict()},
        "diffusion": dict(DIFFUSION), "base_weights": {
            "path": str(base_checkpoint), "sha256": sha256_file(base_checkpoint),
            "subkey": "weights.ema", "state_sha256": frozen_base_sha,
            "updates": BASE_UPDATES},
        "data_identity": data_identity,
        "training": dict(TRAINING, updates=total_updates),
        "training_augmentation": training_augmentation,
        "provenance": provenance,
        "selected_weights": "ema", "model_action_space": "native identity coordinates",
        "action_encoding": None, "action_encoding_identity": True,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    loader = DataLoader(dataset, batch_size=TRAINING["batch_size"],
                        sampler=RandomSampler(dataset), drop_last=True, num_workers=0)
    diffusion = _training_scheduler(); iterator = iter(loader); start = time.monotonic()
    log = (run_dir / "loss.jsonl").open("a" if step0 else "w")
    losses = []
    for step in range(step0 + 1, total_updates + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        raw = batch["observations"].to(device)
        if raw.shape[-1] != 45: raise ValueError("obstacle training requires raw native 45-D frames")
        original_raw = raw
        if augmentation_generator is not None:
            from examples.baselines.constrain.pickcube_xy_gauge import (
                augment_window_observations, update_delta_stats,
            )
            raw, delta = augment_window_observations(raw, augmentation_generator)
            augmentation_delta_stats = update_delta_stats(augmentation_delta_stats, delta)
        obs, _ = prepare_control_observation(raw, stats)
        # Keep the original raw-frame hint byte-identical; translation leaves it invariant.
        hint_condition = pickcube_relative_condition(original_raw, PickCubeConditionNorm())
        # Hint semantics deliberately consume unnormalized raw TCP/obstacle values.
        hint = hint_condition[:, None, :].expand(-1, 16, -1)
        actions = batch["actions"].to(device)
        noise = torch.randn_like(actions); t = torch.randint(0, 100, (len(actions),), device=device)
        noised = diffusion.add_noise(actions, noise, t)
        prediction = denoiser(noised, t, global_cond=obs.flatten(1), hint=hint)
        loss = torch.nn.functional.mse_loss(prediction, noise)
        if not torch.isfinite(loss): raise FloatingPointError(f"Constrain nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); lr_sched.step(); ema.step(params)
        value = float(loss.detach().cpu()); losses.append(value)
        log.write(json.dumps({"step": step, "loss": value, "lr": lr_sched.get_last_lr()[0]}) + "\n")
        if step % 100 == 0: log.flush()
        if step % 10000 == 0 or step == total_updates:
            log.flush()
            _save_control(checkpoint_dir / f"recovery_{step:06d}.pt", denoiser, ema,
                          optimizer, lr_sched, step, {"round37": _rng_state()}, config,
                          augmentation_generator, augmentation_delta_stats)
            print(f"Constrain {step}/{total_updates} loss={value:.6f}", flush=True)
    log.close()
    final = checkpoint_dir / "final.pt"
    _save_control(final, denoiser, ema, optimizer, lr_sched, total_updates,
                  {"round37": _rng_state()}, config, augmentation_generator,
                  augmentation_delta_stats)
    # A durable final artifact exists before post-training validation.
    if state_sha(denoiser.base_net.state_dict()) != frozen_base_sha:
        raise RuntimeError(f"frozen base changed; preserve pending checkpoint {final}")
    loaded, ck = load_control(final, stats, device, expected_base_sha=sha256_file(base_checkpoint),
                              expected_augmentation=training_augmentation)
    result = {"step": total_updates, "elapsed_seconds": time.monotonic()-start,
            "checkpoint": str(final), "checkpoint_sha256": sha256_file(final),
            "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
            "all_losses_finite": all(np.isfinite(losses)), "ema_loaded": loaded is not None,
            "base_state_sha256": frozen_base_sha, "checkpoint_step": ck["step"]}
    if augmentation_delta_stats is not None:
        from examples.baselines.constrain.pickcube_xy_gauge import finalize_delta_stats
        result["augmentation_delta_stats"] = finalize_delta_stats(augmentation_delta_stats)
    return result


def _validate_control(ck, stats, base_checkpoint, require_final=True,
                      training_augmentation=None):
    from examples.baselines.constrain.pickcube_xy_gauge import validate_checkpoint_metadata
    validate_checkpoint_metadata(ck.get("config", {}).get("training_augmentation"),
                                 training_augmentation)
    cfg = ck.get("config", {}); data = cfg.get("data_identity", {})
    norm = cfg.get("normalization", {})
    if (ck.get("format") != CONSTRAIN_FORMAT or cfg.get("profile") != PROFILE
            or norm.get("observation_profile") != PROFILE
            or norm.get("observation_stats_sha256") != stats["sha256"]
            or norm.get("observation_stats") != stats
            or cfg.get("model") != CONSTRAIN_CONFIG
            or cfg.get("diffusion") != DIFFUSION
            or data.get("selected_class") != "obstacle"
            or data.get("selected_episode_ids") != list(range(1000,1500))
            or data.get("episodes") != 500 or data.get("windows") != 45840
            or data.get("h5_sha256") != DATA_H5_SHA
            or data.get("manifest_sha256") != DATA_MANIFEST_SHA
            or cfg.get("base_weights", {}).get("sha256") != sha256_file(base_checkpoint)
            or cfg.get("training") != dict(TRAINING, updates=CONSTRAIN_UPDATES)
            or cfg.get("training_augmentation") != training_augmentation
            or (training_augmentation is not None
                and (ck.get("rng", {}).get("gauge_augmentation") is None
                     or ck.get("rng", {}).get("gauge_delta_stats") is None))
            or cfg.get("policy") != POLICY
            or cfg.get("selected_weights") != "ema"
            or (require_final and ck.get("step") != CONSTRAIN_UPDATES)):
        raise ValueError("normalized PickCube Constrain checkpoint contract mismatch")
    return ck


def load_control(checkpoint, stats, device="cpu", expected_base_sha=None,
                 expected_augmentation=None):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck.get("config", {}); normalizer = config.get("normalization", {})
    from examples.baselines.constrain.pickcube_xy_gauge import validate_checkpoint_metadata
    validate_checkpoint_metadata(config.get("training_augmentation"), expected_augmentation)
    if (ck.get("format") != CONSTRAIN_FORMAT or config.get("profile") != PROFILE
            or normalizer.get("observation_profile") != PROFILE
            or normalizer.get("observation_stats_sha256") != stats["sha256"]
            or normalizer.get("observation_stats") != stats
            or config.get("model") != CONSTRAIN_CONFIG
            or config.get("diffusion") != DIFFUSION
            or config.get("policy") != POLICY
            or config.get("model_action_space") != "native identity coordinates"
            or config.get("action_encoding") is not None
            or config.get("action_encoding_identity") is not True
            or config.get("data_identity", {}).get("selected_episode_ids") != list(range(1000,1500))
            or config.get("data_identity", {}).get("episodes") != 500
            or config.get("data_identity", {}).get("windows") != 45840
            or config.get("data_identity", {}).get("episodes") != 500
            or config.get("data_identity", {}).get("windows") != 45840
            or config.get("data_identity", {}).get("h5_sha256") != DATA_H5_SHA
            or config.get("data_identity", {}).get("manifest_sha256") != DATA_MANIFEST_SHA
            or config.get("base_weights", {}).get("subkey") != "weights.ema"
            or config.get("base_weights", {}).get("updates") != BASE_UPDATES
            or config.get("training") != dict(TRAINING, updates=CONSTRAIN_UPDATES)
            or config.get("training_augmentation") != expected_augmentation
            or config.get("selected_weights") != "ema"
            or ck.get("step") != CONSTRAIN_UPDATES):
        raise ValueError("normalized PickCube Constrain checkpoint contract mismatch")
    if expected_base_sha and config.get("base_weights", {}).get("sha256") != expected_base_sha:
        raise ValueError("normalized Constrain matched Base checkpoint hash mismatch")
    denoiser = build_constrain_denoiser(CONSTRAIN_CONFIG, device=device)
    denoiser.load_state_dict(ck["model"], strict=True)
    from diffusers.training_utils import EMAModel
    ema = EMAModel(parameters=denoiser.trainable_parameters(), power=0.75)
    ema.load_state_dict(ck["ema"]); ema.copy_to(denoiser.trainable_parameters())
    denoiser.eval()
    return denoiser, ck


class ZScoreBaseAdapter:
    def __init__(self, checkpoint, stats_path, device="cpu", expected_augmentation=None):
        self.stats = load_stats(stats_path)
        ck = _base_contract(torch.load(checkpoint, map_location="cpu", weights_only=False),
                            self.stats, training_augmentation=expected_augmentation)
        self._agent = build_agent(device); self._agent.load_state_dict(ck["weights"]["ema"], strict=True)
        self._agent.eval(); self.device = torch.device(device); self.stats_sha256 = self.stats["sha256"]
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                       clip_sample=True, prediction_type="epsilon")
        self.scheduler.set_timesteps(100)
        self.metadata = {"format": BASE_FORMAT, "profile": PROFILE, "normalizer_sha256": self.stats_sha256,
                         "checkpoint_sha256": sha256_file(checkpoint), "weights": "ema",
                         "execution_horizon": 8}

    def base_cond(self, raw_queued):
        raw = torch.as_tensor(raw_queued, dtype=torch.float32, device=self.device)
        if raw.shape[-2:] != (2,45): raise ValueError("expected raw native (2,45) history")
        return raw

    @staticmethod
    def map_action(action):
        action = torch.as_tensor(action)
        if action.shape[-1] != 7 or not torch.isfinite(action).all():
            raise ValueError("expected finite native seven-channel PickCube actions")
        if bool((action.abs() > 1.000001).any()):
            raise ValueError("native seven-channel actions must remain in [-1,1]")
        return action

    def get_action(self, raw_queued):
        raw = torch.as_tensor(raw_queued, dtype=torch.float32, device=self.device)
        if raw.ndim == 2: raw = raw[None]
        if raw.shape[-2:] != (2,45): raise ValueError("expected raw native (batch,2,45) observations")
        condition = normalize_observation(raw[..., :42], self.stats).flatten(1)
        sample = torch.randn((raw.shape[0],16,7), device=self.device)
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                eps = self._agent.noise_pred_net(sample, t, global_cond=condition)
                sample = self.scheduler.step(eps,t,sample).prev_sample
        return sample[:,1:9]


class ZScoreConstrainAdapter:
    def __init__(self, checkpoint, stats_path, base_checkpoint, device="cpu", control_enabled=True,
                 expected_augmentation=None):
        self.stats = load_stats(stats_path)
        self.denoiser, self.ck = load_control(checkpoint, self.stats, device,
                                              expected_base_sha=sha256_file(base_checkpoint),
                                              expected_augmentation=expected_augmentation)
        self.device = torch.device(device); self.control_enabled = control_enabled
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                       clip_sample=True, prediction_type="epsilon")
        self.scheduler.set_timesteps(100)
        self.metadata = {"format": CONSTRAIN_FORMAT, "profile": PROFILE,
                         "normalizer_sha256": self.stats["sha256"],
                         "checkpoint_sha256": sha256_file(checkpoint), "weights": "ema",
                         "control_enabled": control_enabled, "execution_horizon": 8}

    def base_cond(self, raw_queued):
        raw = torch.as_tensor(raw_queued, dtype=torch.float32, device=self.device)
        if raw.shape[-2:] != (2,45): raise ValueError("expected raw native (2,45) history")
        return raw

    @staticmethod
    def map_action(action):
        action = torch.as_tensor(action)
        if action.shape[-1] != 7 or not torch.isfinite(action).all():
            raise ValueError("expected finite native seven-channel PickCube actions")
        if bool((action.abs() > 1.000001).any()):
            raise ValueError("native seven-channel actions must remain in [-1,1]")
        return action

    def get_action(self, raw_queued):
        raw = torch.as_tensor(raw_queued, dtype=torch.float32, device=self.device)
        if raw.ndim == 2: raw = raw[None]
        if raw.shape[-2:] != (2,45): raise ValueError("expected raw native (batch,2,45) observations")
        obs42, hint_condition = prepare_control_observation(raw, self.stats)
        condition = obs42.flatten(1)
        # Critical: calculate relative geometry from the raw latest 45-D frame.
        hint = hint_condition[:,None,:].expand(-1,16,-1)
        sample = torch.randn((raw.shape[0],16,7), device=self.device)
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                eps = self.denoiser(sample,t,global_cond=condition,hint=hint,
                                    enabled=self.control_enabled)
                sample = self.scheduler.step(eps,t,sample).prev_sample
        actions = sample[:,1:9]
        if actions.shape[-1] != 7 or not torch.isfinite(actions).all():
            raise ValueError("Constrain emitted invalid native action sequence")
        return actions


ROUND37_EXP = Path("outputs/exp/20261028_200000_pickcube_direct_zscore_r37")
ROUND37_BASE_CKPT = Path("outputs/ckpt/20261028_200000_pickcube_direct_zscore_r37_base")
ROUND37_CONTROL_CKPT = Path("outputs/ckpt/20261028_200000_pickcube_direct_zscore_r37_constrain")
R36_REGISTRATION = Path("outputs/exp/20261028_191000_pickcube_archived_ddpm100_r36/preregistration.json")
R36_PAIRED = Path("outputs/exp/20261028_191000_pickcube_archived_ddpm100_r36/paired")
R36_DEV = Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes")
SOURCE_FILES = (
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/pickcube_observation_norm.py",
    "examples/baselines/constrain/fit_pickcube_observation_norm.py",
    "examples/baselines/constrain/materialize_pickcube_fullscene500.py",
    "examples/baselines/constrain/pickcube_base_dp.py",
    "examples/baselines/constrain/pickcube_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/constrain_ckpt.py",
    "examples/baselines/constrain/pickcube_learning_pilot.py",
    "examples/baselines/constrain/pickcube_transport_detour.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/diagnose_pickcube_contact.py",
    "mani_skill/envs/tasks/tabletop/pick_cube.py",
)


def source_identity():
    files = {name: sha256_file(name) for name in SOURCE_FILES}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "files_sha256": hashlib.sha256(encoded).hexdigest()}


def register_round37(stats_path, exp_dir=ROUND37_EXP):
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet

    exp_dir = Path(exp_dir); exp_dir.mkdir(parents=True, exist_ok=False)
    stats_path = Path(stats_path)
    stats = load_stats(stats_path)
    r36 = json.loads(R36_REGISTRATION.read_text())
    scene_dir = Path(r36["paths"]["dev_scenes"])
    catalog = TransportDetourSceneSet(scene_dir)
    if len(catalog) != 30:
        raise ValueError("Round 37 registration requires the exact existing 30-scene R36 dev set")
    current_native_source_sha = sha256_file("mani_skill/envs/tasks/tabletop/pick_cube.py")
    if current_native_source_sha != r36["validation"]["maniskill_pickcube_source"]["sha256"]:
        raise ValueError("PickCube native task source differs from the archived R36 evaluation identity")
    scenes_manifest_sha = sha256_file(scene_dir / "manifest.json")
    if scenes_manifest_sha != r36["validation"]["dev_manifest"]["sha256"]:
        raise ValueError("frozen R36 development manifest hash mismatch")
    scenes_h5_sha = sha256_file(scene_dir / "scenes.h5")
    if scenes_h5_sha != r36["validation"]["dev_scenes_h5"]["sha256"]:
        raise ValueError("frozen R36 development scene HDF5 hash mismatch")
    frozen_validation = Path(DATA_MANIFEST).with_name("validation.json")
    if sha256_file(frozen_validation) != "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f":
        raise ValueError("frozen 1000/500 validation identity mismatch")
    paths = {"experiment": str(exp_dir), "registration": str(exp_dir / "preregistration.json"),
             "stats": str(stats_path), "base_run": str(exp_dir / "train_base"),
             "base_checkpoint_dir": str(ROUND37_BASE_CKPT),
             "base_checkpoint": str(ROUND37_BASE_CKPT / "final.pt"),
             "constrain_run": str(exp_dir / "train_constrain"),
             "constrain_checkpoint_dir": str(ROUND37_CONTROL_CKPT),
             "constrain_checkpoint": str(ROUND37_CONTROL_CKPT / "final.pt"),
             "preflight": str(exp_dir / "preflight"), "paired": str(exp_dir / "paired"),
             "dev_scenes": str(scene_dir), "r36_paired": str(R36_PAIRED)}
    body = {
        "round": 37, "status": "registered_before_training_and_native_evaluation",
        "profile": PROFILE, "paths": paths,
        "dataset": {"h5_path": DATA_H5, "h5_sha256": DATA_H5_SHA,
                    "manifest_path": DATA_MANIFEST, "manifest_sha256": DATA_MANIFEST_SHA,
                    "validation_path": str(frozen_validation),
                    "validation_sha256": sha256_file(frozen_validation),
                    "episodes": 1500, "direct_ids": DIRECT_IDS,
                    "obstacle_ids": list(range(1000, 1500)),
                    "direct_windows": 83004, "obstacle_windows": 45840,
                    "actions": "native identity 7-D; no codec/decode/clamp",
                    "terminal_gripper_padding": -1},
        "normalizer": {"path": str(stats_path), "file_sha256": sha256_file(stats_path),
                       "statistics_sha256": stats["sha256"], "fit_source_ids": DIRECT_IDS,
                       "fit_frame_count": stats["fit_frame_count"], "floor": stats["floor"],
                       "floored_fields": stats["floored_fields"],
                       "algorithm": stats["fit_algorithm"]},
        "training": {"base_updates": BASE_UPDATES, "constrain_updates": CONSTRAIN_UPDATES,
                     **TRAINING, "base_class": "direct", "constrain_class": "obstacle",
                     "base_frozen_in_constrain": True, "selected_weights": "final EMA",
                     "loss": "unweighted epsilon MSE", "recovery_every_updates": 10000,
                     "max_wall_seconds": {"base": 3000, "constrain": 9000}},
        "model": {"base": dict(BASE_CONFIG), "constrain": dict(CONSTRAIN_CONFIG),
                  "observation_dim": 42, "condition_dim": 84, "action_dim": 7},
        "sampler": dict(DIFFUSION),
        "native_evaluation": {"scene_count": 30, "max_control_steps": 500,
                              "scene_manifest_sha256": scenes_manifest_sha,
                              "scene_h5_sha256": scenes_h5_sha,
                              "scene_state_source": str(scene_dir),
                              "noise_seed_rule": "torch.manual_seed(scene seed) separately for each policy",
                              "wall_timeout_seconds_per_episode": 120,
                              "outer_timeout_seconds": 18000,
                              "primary_reference": {"r36_base": {"task": 10, "safe": 0, "contact": 30, "xy": 15},
                                                    "r36_constrain": {"task": 25, "safe": 16, "contact": 13, "xy": 4}},
                              "gate": {"task_gte": 27, "safe_gte": 26,
                                       "contact_lte": 3, "xy_lte": 3}},
        "source_identity": source_identity(),
        "r36_preregistration_sha256": sha256_file(R36_REGISTRATION),
        "formal_performance_claim": False,
        "comparison_interpretation": "matched end-to-end observation-profile retraining; not pure control/data attribution",
    }
    body["registration_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = Path(paths["registration"])
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def verify_registration(path):
    path = Path(path); registration = json.loads(path.read_text())
    expected_hash = registration.get("registration_sha256")
    body = dict(registration); body.pop("registration_sha256", None)
    actual_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("Round 37 preregistration checksum mismatch")
    if registration.get("source_identity") != source_identity():
        raise ValueError("Round 37 registered source bytes changed")
    if sha256_file(registration["paths"]["stats"]) != registration["normalizer"]["file_sha256"]:
        raise ValueError("Round 37 fitted statistics artifact changed")
    if load_stats(registration["paths"]["stats"])["sha256"] != registration["normalizer"]["statistics_sha256"]:
        raise ValueError("Round 37 statistics identity changed")
    for name, expected in ((DATA_H5, DATA_H5_SHA), (DATA_MANIFEST, DATA_MANIFEST_SHA)):
        if sha256_file(name) != expected:
            raise ValueError(f"frozen training data changed: {name}")
    return registration


def _registration_provenance(path, registration):
    return {"registration_path": str(path),
            "registration_sha256": registration["registration_sha256"],
            "source_identity": registration["source_identity"],
            "registered_at_phase": "before_training"}


def preflight_action_equivalence(base_checkpoint, control_checkpoint, stats_path, device="cuda"):
    stats = load_stats(stats_path)
    base = ZScoreBaseAdapter(base_checkpoint, stats_path, device)
    control = ZScoreConstrainAdapter(control_checkpoint, stats_path, base_checkpoint,
                                     device, control_enabled=False)
    with open(Path(R36_DEV) / "manifest.json") as stream:
        scene = json.load(stream)["scenes"][0]
    catalog_cls = __import__("examples.baselines.constrain.pickcube_transport_detour",
                             fromlist=["TransportDetourSceneSet"]).TransportDetourSceneSet
    catalog = catalog_cls(R36_DEV)
    raw = torch.as_tensor(catalog.load(0)["observation"], dtype=torch.float32, device=device)
    queued = raw.reshape(1,45).repeat(2,1)
    torch.manual_seed(1947); a = base.get_action(queued)
    torch.manual_seed(1947); b = control.get_action(queued)
    difference = float((a-b).abs().max())
    evidence = {"base_checkpoint_sha256": sha256_file(base_checkpoint),
                "constrain_checkpoint_sha256": sha256_file(control_checkpoint),
                "normalizer_sha256": stats["sha256"], "seed": 1947,
                "raw_observation_sha256": hashlib.sha256(queued[0,-1].cpu().numpy().tobytes()).hexdigest(),
                "disabled_control_max_abs_diff": difference,
                "matches": difference <= 1e-6, "native_action_shape": list(a.shape)}
    if not evidence["matches"] or a.shape != (1,8,7):
        raise RuntimeError(f"independent normalized Base/disabled-Control equivalence failed: {evidence}")
    return evidence


def evaluate_round37(registration_path):
    from examples.baselines.constrain.pickcube_learning_pilot import run_pilot_episode, summarize_pilot
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet

    registration = verify_registration(registration_path)
    paths = registration["paths"]
    base_ck = Path(paths["base_checkpoint"]); control_ck = Path(paths["constrain_checkpoint"])
    if not base_ck.is_file() or not control_ck.is_file():
        raise FileNotFoundError("both registered final EMA checkpoints are required before evaluation")
    stats_path = paths["stats"]
    base = ZScoreBaseAdapter(base_ck, stats_path, "cuda")
    control = ZScoreConstrainAdapter(control_ck, stats_path, base_ck, "cuda")
    equivalence = preflight_action_equivalence(base_ck, control_ck, stats_path, "cuda")
    scene_set = TransportDetourSceneSet(paths["dev_scenes"])
    if len(scene_set) != 30: raise ValueError("fixed native trial requires 30 existing R36 development scenes")
    output = Path(paths["paired"])
    output.mkdir(parents=True, exist_ok=False)
    source_before = source_identity()
    prereg = {
        "status": "frozen_before_native_evaluation", "registration_sha256": registration["registration_sha256"],
        "base_checkpoint_sha256": sha256_file(base_ck), "constrain_checkpoint_sha256": sha256_file(control_ck),
        "normalizer_file_sha256": sha256_file(stats_path),
        "normalizer_statistics_sha256": load_stats(stats_path)["sha256"],
        "scene_manifest_sha256": sha256_file(Path(paths["dev_scenes"]) / "manifest.json"),
        "scene_h5_sha256": sha256_file(Path(paths["dev_scenes"]) / "scenes.h5"),
        "source_identity_before_native": source_before,
        "sampler": dict(DIFFUSION), "episodes_per_policy": 30,
        "max_control_steps": 500, "control_chunk": 8,
        "policy_order": ["base_dp", "constrain_dp"],
        "disabled_control_equivalence": equivalence,
        "success_rule": "native held-at-goal PickCube success; stop after first success step",
        "actions": "raw native identity pd_ee_delta_pose, no codec/decode/clamp",
    }
    prereg["evaluation_registration_sha256"] = hashlib.sha256(
        json.dumps(prereg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    _atomic_json(output / "frozen_config.json", prereg)
    rows_by_policy = {}
    policies = (("base_dp", base), ("constrain_dp", control))
    try:
        for policy, adapter in policies:
            policy_dir = output / policy; policy_dir.mkdir()
            rows = []
            for index in range(30):
                scene = scene_set.load(index)
                path = policy_dir / f"episode_{index:03d}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    if row.get("checkpoint_sha256") != adapter.metadata["checkpoint_sha256"]:
                        raise ValueError(f"existing {policy} row {index} checkpoint identity mismatch")
                else:
                    started = time.monotonic()
                    prior_handler = signal.signal(signal.SIGALRM, _episode_alarm)
                    signal.alarm(EPISODE_TIMEOUT_SECONDS)
                    try:
                        row = run_pilot_episode(adapter, scene_set, index, "cuda", max_steps=500)
                    except Exception as exc:
                        row = {"scene_index": index, "scene_id": scene["metadata"]["scene_id"],
                               "seed": int(scene["metadata"]["seed"]), "status": "runtime_error",
                               "error": f"{type(exc).__name__}: {exc}"}
                        if isinstance(exc, TimeoutError): row["error_kind"] = "wall_timeout"
                    finally:
                        signal.alarm(0); signal.signal(signal.SIGALRM, prior_handler)
                    obs = np.asarray(scene["observation"], dtype=np.float32).reshape(-1)
                    normalized = normalize_observation(torch.as_tensor(obs[:42]), load_stats(stats_path))
                    row["normalizer_sha256"] = load_stats(stats_path)["sha256"]
                    row["normalizer_condition_sha256"] = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
                    row["checkpoint_sha256"] = adapter.metadata["checkpoint_sha256"]
                    row["raw_restored_observation_sha256"] = row.get("restored_observation_sha256")
                    row["wall_seconds"] = round(time.monotonic() - started, 3)
                    _atomic_json(path, row)
                rows.append(row)
            rows_by_policy[policy] = rows
            _atomic_json(policy_dir / "summary.json", summarize_pilot(rows, 30))
    except BaseException:
        _atomic_json(output / "status.json", {"status": "incomplete", "rows": {
            key: len(value) for key, value in rows_by_policy.items()}})
        raise
    base_rows, control_rows = rows_by_policy["base_dp"], rows_by_policy["constrain_dp"]
    state_pairing = all(
        a.get("status") == b.get("status") == "evaluated"
        and a.get("scene_id") == b.get("scene_id")
        and a.get("restored_state_sha256") == b.get("restored_state_sha256")
        and a.get("restored_controller_state_sha256") == b.get("restored_controller_state_sha256")
        and a.get("restored_observation_sha256") == b.get("restored_observation_sha256")
        for a,b in zip(base_rows,control_rows))
    stats = {key: summarize_pilot(rows,30) for key,rows in rows_by_policy.items()}
    analysis = {"complete": all(s["complete"] for s in stats.values()),
                "runtime_errors": {k:s["runtime_errors"] for k,s in stats.items()},
                "pairing_identity_matches_scene_by_scene": state_pairing,
                "normalizer_hashes": {k: load_stats(stats_path)["sha256"] for k in stats},
                "policy_metrics": stats, "disabled_control_equivalence": equivalence,
                "r36_reference": registration["native_evaluation"]["primary_reference"],
                "formal_performance_claim": False,
                "comparison_interpretation": registration.get(
                    "comparison_interpretation", registration.get("interpretation"))}
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    analysis["paired_base_vs_constrain"] = {
        field: _transitions(base_rows, control_rows, field) for field in fields}
    old_rows = {policy: [_read_json(Path(paths["r36_paired"]) / policy / f"episode_{i:03d}.json")
                         for i in range(30)] for policy in ("base_dp", "constrain_dp")}
    analysis["vs_r36"] = {policy: {field: _transitions(old_rows[policy], rows_by_policy[policy], field)
                                    for field in fields}
                          for policy in ("base_dp", "constrain_dp")}
    for policy, rows in rows_by_policy.items():
        for index, row in enumerate(rows):
            old = old_rows[policy][index]
            if row.get("status") == "evaluated" and (
                    row.get("scene_id") != old.get("scene_id")
                    or row.get("restored_state_sha256") != old.get("restored_state_sha256")
                    or row.get("restored_controller_state_sha256") != old.get("restored_controller_state_sha256")
                    or row.get("restored_observation_sha256") != old.get("restored_observation_sha256")):
                raise ValueError(f"Round 37 restored state differs from R36 at {policy} scene {index}")
    base_m, control_m = stats["base_dp"], stats["constrain_dp"]
    gate = {"task": control_m["task_success_count"] >= 27,
            "safe": control_m["safe_success_count"] >= 26,
            "contact": control_m["obstacle_contact_count"] <= 3,
            "xy": control_m["xy_crossing_count"] <= 3,
            "complete": analysis["complete"], "paired": state_pairing}
    analysis["gate"] = {"passed": all(gate.values()), "criteria": gate}
    analysis["wall_seconds"] = {policy: [row.get("wall_seconds") for row in rows]
                                 for policy,rows in rows_by_policy.items()}
    # Terminal phases are a property of retained episode results, not the
    # summarize_pilot metric object.  Keep this independent of first contact.
    analysis["phase_counts"] = {
        policy: _terminal_phase_counts(rows) for policy, rows in rows_by_policy.items()}
    analysis["contact_first_phase_counts"] = {
        policy: _phase_counts(rows) for policy,rows in rows_by_policy.items()}
    analysis["source_identity_after_native"] = source_identity()
    _atomic_json(output / "analysis.json", analysis)
    _atomic_json(output / "status.json", {"status": "complete" if analysis["complete"] else "complete_with_errors",
                                          "paired_identity_matches": state_pairing,
                                          "gate_passed": gate["complete"] and all(gate.values())})
    return analysis


def _episode_alarm(_signum, _frame):
    raise TimeoutError(f"native episode exceeded {EPISODE_TIMEOUT_SECONDS}-second wall timeout")


def _phase_counts(rows):
    result = {}
    for row in rows:
        phase = (row.get("first_contact") or {}).get("phase")
        if phase: result[phase] = result.get(phase, 0) + 1
    return result


def _terminal_phase_counts(rows):
    """Count recorded terminal phases; report unknown only when metadata is absent."""
    result = {}
    for row in rows:
        if row.get("status") != "evaluated":
            continue
        phase = row.get("result", {}).get("phase")
        phase = phase if phase is not None else "unknown"
        result[phase] = result.get(phase, 0) + 1
    return result


def _transitions(left_rows, right_rows, field):
    left = [bool(row.get("result", {}).get(field, False)) for row in left_rows]
    right = [bool(row.get("result", {}).get(field, False)) for row in right_rows]
    positive = field in ("native_success", "safe_success")
    if positive:
        table = {"improved_new_only": sum(not a and b for a,b in zip(left,right)),
                 "regressed_old_only": sum(a and not b for a,b in zip(left,right)),
                 "both": sum(a and b for a,b in zip(left,right)),
                 "neither": sum(not a and not b for a,b in zip(left,right)),
                 "old_marginal": sum(left), "new_marginal": sum(right)}
        categories = ("improved_new_only", "regressed_old_only", "both", "neither")
    else:
        table = {"cleared_old_only": sum(a and not b for a,b in zip(left,right)),
                 "introduced_new_only": sum(not a and b for a,b in zip(left,right)),
                 "both": sum(a and b for a,b in zip(left,right)),
                 "neither": sum(not a and not b for a,b in zip(left,right)),
                 "old_marginal": sum(left), "new_marginal": sum(right)}
        categories = ("cleared_old_only", "introduced_new_only", "both", "neither")
    if (any(not isinstance(table[key], int) for key in table)
            or sum(table[key] for key in categories) != len(left)
            or table["old_marginal"] != (table["regressed_old_only"] + table["both"] if positive
                                         else table["cleared_old_only"] + table["both"])
            or table["new_marginal"] != (table["improved_new_only"] + table["both"] if positive
                                         else table["introduced_new_only"] + table["both"])):
        raise ValueError("paired transition table does not reconcile to exact integer marginals")
    return table


def _read_json(path):
    return json.loads(Path(path).read_text())


def analyze_existing_round37(registration_path):
    """Offline-only analysis of retained native rows; performs no rollout."""
    registration_path = Path(registration_path)
    registration = _read_json(registration_path)
    reg_body = dict(registration); reg_sha = reg_body.pop("registration_sha256", None)
    actual_reg_sha = hashlib.sha256(json.dumps(
        reg_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if reg_sha != actual_reg_sha:
        raise ValueError("Round 37 preregistration checksum mismatch during offline analysis")
    paths = registration["paths"]
    for name, expected in ((DATA_H5, DATA_H5_SHA), (DATA_MANIFEST, DATA_MANIFEST_SHA)):
        if sha256_file(name) != expected:
            raise ValueError(f"frozen dataset changed after native evaluation: {name}")
    paired_dir = Path(paths["paired"])
    prereg = _read_json(paired_dir / "preregistration.json")
    base_checkpoint = Path(paths["base_checkpoint"]); control_checkpoint = Path(paths["constrain_checkpoint"])
    if (sha256_file(base_checkpoint) != prereg["base_checkpoint_sha256"]
            or sha256_file(control_checkpoint) != prereg["constrain_checkpoint_sha256"]):
        raise ValueError("native episode checkpoint hashes differ from the frozen evaluation registration")
    if load_stats(paths["stats"])["sha256"] != prereg["normalizer_statistics_sha256"]:
        raise ValueError("fitted normalizer identity differs from the frozen native evaluation")
    source_before = prereg["source_identity_before_native"]
    source_after = source_identity()
    changed_sources = {name: {"registered": value, "current": source_after["files"].get(name)}
                       for name, value in source_before["files"].items()
                       if source_after["files"].get(name) != value}
    if set(changed_sources) - {"examples/baselines/constrain/pickcube_zscore_round37.py"}:
        raise ValueError("runtime/evaluation source changed after native run; offline report cannot assume equivalence")
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot
    rows_by_policy = {}
    for policy in ("base_dp", "constrain_dp"):
        rows = [_read_json(paired_dir / policy / f"episode_{i:03d}.json") for i in range(30)]
        if any(row.get("scene_index") != index for index, row in enumerate(rows)):
            raise ValueError(f"retained {policy} records do not cover exact ordered indices 0..29")
        expected_ckpt = prereg[f"{ 'base' if policy == 'base_dp' else 'constrain' }_checkpoint_sha256"]
        if any(row.get("checkpoint_sha256") != expected_ckpt for row in rows):
            raise ValueError(f"retained {policy} record has a different checkpoint identity")
        rows_by_policy[policy] = rows
    summaries = {policy: summarize_pilot(rows, 30) for policy, rows in rows_by_policy.items()}
    base_rows, control_rows = rows_by_policy["base_dp"], rows_by_policy["constrain_dp"]
    pairing_failures = [i for i,(a,b) in enumerate(zip(base_rows, control_rows))
                        if a.get("status") != "evaluated" or b.get("status") != "evaluated"
                        or any(a.get(key) != b.get(key) for key in (
                            "scene_id", "restored_state_sha256", "restored_controller_state_sha256",
                            "restored_observation_sha256"))]
    state_pairing = not pairing_failures
    # Compare the byte hash of normalized conditions between the two policies.
    normalized_pairing = all(a.get("normalizer_condition_sha256") == b.get("normalizer_condition_sha256")
                             for a,b in zip(base_rows,control_rows))
    old_rows = {policy: [_read_json(Path(paths["r36_paired"]) / policy / f"episode_{i:03d}.json")
                         for i in range(30)] for policy in ("base_dp", "constrain_dp")}
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    r36_mismatches = []
    for policy in ("base_dp", "constrain_dp"):
        for index,(old,new) in enumerate(zip(old_rows[policy], rows_by_policy[policy])):
            if new.get("status") != "evaluated" or any(
                old.get(a) != new.get(b) for a,b in (
                    ("scene_id","scene_id"), ("restored_state_sha256","restored_state_sha256"),
                    ("restored_controller_state_sha256","restored_controller_state_sha256"),
                    ("restored_observation_sha256","restored_observation_sha256"))):
                r36_mismatches.append({"policy":policy,"scene_index":index})
    analyses = {
        "complete": all(value["complete"] for value in summaries.values()),
        "runtime_errors": {policy:value["runtime_errors"] for policy,value in summaries.items()},
        "policy_metrics": summaries,
        "pairing_identity_matches_scene_by_scene": state_pairing,
        "pairing_failures": pairing_failures,
        "normalized_condition_hashes_match_between_policies": normalized_pairing,
        "normalizer_statistics_sha256": prereg["normalizer_statistics_sha256"],
        "r36_reference": registration["native_evaluation"]["primary_reference"],
        "r36_state_identity_mismatches": r36_mismatches,
        "vs_r36": {policy: {field:_transitions(old_rows[policy], rows_by_policy[policy], field)
                             for field in fields} for policy in ("base_dp","constrain_dp")},
        "paired_base_vs_constrain": {field:_transitions(base_rows,control_rows,field) for field in fields},
        "gate": {"criteria": {
            "task": summaries["constrain_dp"]["task_success_count"] >= 27,
            "safe": summaries["constrain_dp"]["safe_success_count"] >= 26,
            "contact": summaries["constrain_dp"]["obstacle_contact_count"] <= 3,
            "xy": summaries["constrain_dp"]["xy_crossing_count"] <= 3,
            "complete": all(value["complete"] for value in summaries.values()),
            "paired": state_pairing,
            "same_r36_restored_states": not r36_mismatches,
        }},
        "wall_seconds": {policy:[row.get("wall_seconds") for row in rows]
                         for policy,rows in rows_by_policy.items()},
        "terminal_phase_counts": {policy:_count_values(
            [row.get("result",{}).get("phase") for row in rows])
            for policy,rows in rows_by_policy.items()},
        "contact_first_phase_counts": {policy:_phase_counts(rows)
                                       for policy,rows in rows_by_policy.items()},
        "final_cube_xyz_and_goal_distance_present": all(
            row.get("status") != "evaluated" or ("final_cube_xyz" in row
            and "goal_xyz" in row and "final_goal_distance_3d" in row)
            for rows in rows_by_policy.values() for row in rows),
        "runtime_source_identity_at_native_evaluation": source_before,
        "analysis_source_identity_after_reporting_fix": source_after,
        "analysis_only_source_changes": changed_sources,
        "formal_performance_claim": False,
        "interpretation": registration.get(
            "comparison_interpretation", registration.get("interpretation")),
        "contact_geometry_gap": "native records retain first contact link/phase; complete contact entity-pair positions were unavailable",
    }
    analyses["gate"]["passed"] = all(analyses["gate"]["criteria"].values())
    analyses["native_evaluation_registration_sha256"] = prereg["evaluation_registration_sha256"]
    analyses["base_checkpoint_sha256"] = prereg["base_checkpoint_sha256"]
    analyses["constrain_checkpoint_sha256"] = prereg["constrain_checkpoint_sha256"]
    _atomic_json(paired_dir / "analysis.json", analyses)
    _atomic_json(paired_dir / "status.json", {
        "status": "complete" if analyses["complete"] else "complete_with_errors",
        "analysis_repaired_from_retained_rows": True,
        "native_evaluation_rerun": False,
        "gate_passed": analyses["gate"]["passed"],
        "source_identity_at_native_evaluation_preserved": True,
    })
    return analyses


def _count_values(values):
    counts = {}
    for value in values:
        if value is not None: counts[value] = counts.get(value,0) + 1
    return counts


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "preflight-base", "train-base",
        "preflight-control", "train-control", "evaluate", "analyze-existing"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.command == "register":
        reg = register_round37(args.stats or Path("outputs/exp/20261028_220000_pickcube_zscore_r37/normalizer_stats.json"),
                               args.registration.parent)
        print(json.dumps({"registration": str(args.registration),
                          "registration_sha256": reg["registration_sha256"],
                          "source_identity": reg["source_identity"]}, indent=2)); return reg
    if args.command == "analyze-existing":
        return analyze_existing_round37(args.registration)
    reg = verify_registration(args.registration); paths = reg["paths"]
    if args.command == "preflight-base":
        Path(paths["preflight"]).mkdir(parents=True, exist_ok=True)
        return preflight_base(paths["stats"], Path(paths["preflight"]) / "base.json", args.device)
    if args.command == "train-base":
        provenance = _registration_provenance(args.registration, reg)
        summary = train_base(paths["stats"], paths["base_run"], paths["base_checkpoint_dir"],
                             args.device, args.resume, provenance=provenance)
        _atomic_json(Path(paths["base_run"]) / "train_summary.json", summary)
        summary["source_identity_after_training"] = source_identity()
        summary["source_identity_matches_registration"] = summary["source_identity_after_training"] == reg["source_identity"]
        summary["dataset_identity_after_training"] = {
            "h5_sha256": sha256_file(DATA_H5), "manifest_sha256": sha256_file(DATA_MANIFEST)}
        summary["dataset_identity_matches_registration"] = (
            summary["dataset_identity_after_training"]["h5_sha256"] == DATA_H5_SHA
            and summary["dataset_identity_after_training"]["manifest_sha256"] == DATA_MANIFEST_SHA)
        summary["status"] = ("complete" if summary["source_identity_matches_registration"]
                             and summary["dataset_identity_matches_registration"] else "pending_audit")
        _atomic_json(Path(paths["base_run"]) / "train_summary.json", summary)
        return summary
    if args.command == "preflight-control":
        Path(paths["preflight"]).mkdir(parents=True, exist_ok=True)
        return preflight_control(paths["stats"], paths["base_checkpoint"],
                                 Path(paths["preflight"]) / "constrain.json", args.device)
    if args.command == "train-control":
        provenance = _registration_provenance(args.registration, reg)
        summary = train_constrain(paths["stats"], paths["base_checkpoint"],
                                  paths["constrain_run"], paths["constrain_checkpoint_dir"],
                                  args.device, args.resume, provenance=provenance)
        _atomic_json(Path(paths["constrain_run"]) / "train_summary.json", summary)
        summary["source_identity_after_training"] = source_identity()
        summary["source_identity_matches_registration"] = summary["source_identity_after_training"] == reg["source_identity"]
        summary["dataset_identity_after_training"] = {
            "h5_sha256": sha256_file(DATA_H5), "manifest_sha256": sha256_file(DATA_MANIFEST)}
        summary["dataset_identity_matches_registration"] = (
            summary["dataset_identity_after_training"]["h5_sha256"] == DATA_H5_SHA
            and summary["dataset_identity_after_training"]["manifest_sha256"] == DATA_MANIFEST_SHA)
        summary["status"] = ("complete" if summary["source_identity_matches_registration"]
                             and summary["dataset_identity_matches_registration"] else "pending_audit")
        _atomic_json(Path(paths["constrain_run"]) / "train_summary.json", summary)
        return summary
    return evaluate_round37(args.registration)


if __name__ == "__main__":
    main()
