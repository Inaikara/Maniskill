"""Small-scale PickCube transport-detour learnability pilot.

Formal Base/Constrain trainers stay locked to the trusted 1000/100 dataset.
This module trains a matched new-distribution pair and scores an independent
development set. It does not start Ticket 07.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import (
    build_agent, contract, sha256_file,
)

LAYOUT_TOLERANCE_M = 0.02
OBSTACLE_PLAN_CAP = 100
TRAIN_OBSTACLE_COUNT = 20
TRAIN_DIRECT_COUNT = 20
DEV_SCENE_COUNT = 30
ROUND_CAP = 2
PRIOR_MANIFEST = "outputs/exp/20261023_165956_pickcube_transport_detour/manifest.json"
SELECTED_WEIGHTS = "final EMA fixed before training; no task evaluation"
BASE_FORMAT = "pickcube_pilot_base_v1"
CONSTRAIN_FORMAT = "pickcube_pilot_constrain_v1"
TICKET01_REFERENCE = {
    "distribution": "old PickCube obstacle scenes, not this pilot",
    "attribution": "rates are not a same-distribution improvement delta",
    "full_set": {
        "base_contact": "97/100", "base_xy": "0/100", "base_task": "82/100", "base_safe": "2/100",
        "constrain_contact": "53/100", "constrain_xy": "0/100",
        "constrain_task": "83/100", "constrain_safe": "38/100",
    },
    "sample_mechanism": (
        "distal arm or gripper links contact the cylinder during transport or "
        "goal stabilization while the cube center stays outside the XY disk"
    ),
    "zero_xy_is_not_robot_safety": True,
}


def build_preregistration(round_index, version, paths, reason=None, prior_version=None,
                          batch_size=1024, dedup_mode="cylinder"):
    if int(round_index) < 1 or int(round_index) > ROUND_CAP:
        raise ValueError("pilot round must be 1 or 2")
    if int(round_index) == 2 and not reason:
        raise ValueError("round 2 requires a recorded reason")
    if int(round_index) == 2 and dedup_mode != "task":
        raise ValueError("round 2 requires the revised task dedup")
    if dedup_mode not in ("cylinder", "task"):
        raise ValueError("unknown dedup mode")
    if int(batch_size) < 1:
        raise ValueError("batch size must be positive")
    return {
        "round_index": int(round_index),
        "round_cap": ROUND_CAP,
        "version": version,
        "reason": reason,
        "prior_version": prior_version,
        "prior_manifest": PRIOR_MANIFEST,
        "paths": dict(paths),
        "obstacle_plan_cap": OBSTACLE_PLAN_CAP,
        "train_obstacle_count": TRAIN_OBSTACLE_COUNT,
        "train_direct_count": TRAIN_DIRECT_COUNT,
        "selected_direct_ids": list(range(TRAIN_DIRECT_COUNT)),
        "selected_obstacle_ids": list(range(1000, 1000 + TRAIN_OBSTACLE_COUNT)),
        "dev_scene_count": DEV_SCENE_COUNT,
        "layout_dedup": {
            "tolerance_m": LAYOUT_TOLERANCE_M,
            "metric": "chebyshev",
            "mode": dedup_mode,
            "cylinder_proximity_is_conflict": dedup_mode == "cylinder",
            "within_run_enforced": True,
            "conflict_if": (
                "same seed, or start and goal both within tolerance"
                if dedup_mode == "task"
                else "same seed, or start+goal both within tolerance, or cylinder center within tolerance"
            ),
            "splits": "train, dev, and Ticket 03 prior source are mutually independent",
            "exclude_prior_source": PRIOR_MANIFEST,
        },
        "training": {
            "seed": 1,
            "total_iters": 30000,
            "batch_size": int(batch_size),
            "lr": 1e-4,
            "warmup": 500,
            "ema_power": 0.75,
            "optimizer": "AdamW",
            "betas": [0.95, 0.999],
            "weight_decay": 1e-6,
            "lr_schedule": "cosine",
            "architecture": "existing PickCube ConditionalUnet1D and ConstrainDenoiser",
            "selected_weights": SELECTED_WEIGHTS,
            "base_class": "direct",
            "constrain_class": "obstacle",
            "constrain_init": "matched base final EMA",
            "trainable": "control branch only",
            "differences_from_formal": [
                "episode counts are the predeclared subset, not 1000 direct and 100 obstacle",
                "data identity is this version, not the trusted formal hashes",
                "checkpoint format is pilot-specific so formal loaders reject it",
                "expert may stop on evaluate() success, so terminated[-1] need not be 1",
            ],
        },
        "collection": {
            "train_seed": 2026106001 if dedup_mode == "task" else 2026102401,
            "dev_seed": 2026107001 if dedup_mode == "task" else 2026103001,
            "train_max_attempts": 160 if dedup_mode == "task" else 140,
            "dev_max_attempts": 280 if dedup_mode == "task" else 200,
            "train_wall_seconds": 2800 if dedup_mode == "task" else 3200,
            "dev_wall_seconds": 4500,
            "per_attempt_deadline": 150,
        },
        "eval": {
            "max_steps": 500,
            "episode_wall_seconds": 120,
            "denominator": DEV_SCENE_COUNT,
            "noise_seed_rule": "torch.manual_seed(scene seed) before each scene",
            "video_scene_indexes_fixed_before_results": [0, 1, 2],
        },
        "gate": {
            "task_success_rate_gte": 0.90,
            "safe_success_rate_gte": 0.85,
            "contact_rate_lte": 0.10,
            "xy_crossing_rate_lte": 0.10,
            "require_complete": True,
            "require_zero_runtime_errors": True,
            "formal_performance_claim": False,
            "base_failure_threshold": None,
            "benefit_threshold": None,
            "on_fail": "at most one further collection+training round, then pause Ticket 07",
        },
    }


def write_preregistration(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_preregistration(path):
    payload = json.loads(Path(path).read_text())
    required = (
        "round_index", "train_obstacle_count", "train_direct_count", "dev_scene_count",
        "obstacle_plan_cap", "layout_dedup", "training", "gate", "selected_direct_ids",
        "selected_obstacle_ids",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"preregistration missing {missing}")
    if payload["train_obstacle_count"] > payload["obstacle_plan_cap"]:
        raise ValueError("obstacle subset exceeds the 100-plan")
    if payload["gate"].get("formal_performance_claim") is not False:
        raise ValueError("pilot gate must not claim formal performance")
    if payload["training"].get("selected_weights") != SELECTED_WEIGHTS:
        raise ValueError("selected weights are not the pre-registered final EMA")
    return payload


def layout_view(value):
    if value is None:
        raise ValueError("missing layout")
    if "center_xy" in value and "start_xy" in value:
        return {
            "seed": value.get("seed"),
            "start_xy": value["start_xy"],
            "goal_xy": value["goal_xy"],
            "center_xy": value["center_xy"],
        }
    cylinder = value.get("cylinder") or {}
    return {
        "seed": value.get("seed"),
        "start_xy": value.get("start_xy"),
        "goal_xy": value.get("goal_xy"),
        "center_xy": cylinder.get("center_xy", value.get("center_xy")),
    }


def _within(left, right, tolerance):
    a = np.asarray(left, dtype=np.float64).reshape(2)
    b = np.asarray(right, dtype=np.float64).reshape(2)
    return bool(np.max(np.abs(a - b)) <= float(tolerance))


def layouts_conflict(left, right, tolerance=LAYOUT_TOLERANCE_M, cylinder=True):
    left, right = layout_view(left), layout_view(right)
    if left.get("seed") is not None and left.get("seed") == right.get("seed"):
        return True
    same_task = _within(left["start_xy"], right["start_xy"], tolerance) and _within(
        left["goal_xy"], right["goal_xy"], tolerance)
    if not cylinder:
        return same_task
    same_cylinder = _within(left["center_xy"], right["center_xy"], tolerance)
    return same_task or same_cylinder


def split_is_independent(train, dev, prior, tolerance=LAYOUT_TOLERANCE_M, cylinder=True):
    groups = (("train", train), ("dev", dev), ("prior", prior))
    for name, rows in groups:
        for index, row in enumerate(rows):
            for other in rows[index + 1:]:
                if layouts_conflict(row, other, tolerance, cylinder=cylinder):
                    return False
    for left_name, left_rows in groups:
        for right_name, right_rows in groups:
            if left_name >= right_name:
                continue
            for left in left_rows:
                for right in right_rows:
                    if layouts_conflict(left, right, tolerance, cylinder=cylinder):
                        return False
    return True


def decide_gate(summary):
    failures = []
    if summary.get("task_success_rate", 0.0) < 0.90:
        failures.append("task_success_rate")
    if summary.get("safe_success_rate", 0.0) < 0.85:
        failures.append("safe_success_rate")
    if summary.get("obstacle_contact_rate", 1.0) > 0.10:
        failures.append("obstacle_contact_rate")
    if summary.get("xy_crossing_rate", 1.0) > 0.10:
        failures.append("xy_crossing_rate")
    if not summary.get("complete", False) or int(summary.get("runtime_errors", 1)) != 0:
        failures.append("paired_execution")
    return {
        "decision": "pause" if failures else "pass",
        "failures": failures,
        "formal_performance_claim": False,
        "ticket07": "blocked" if failures else "eligible",
    }


def package_records(records, dest, registration):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    expected = {("direct", int(i)) for i in registration["selected_direct_ids"]}
    expected.update(("obstacle", int(i)) for i in registration["selected_obstacle_ids"])
    if len(registration["selected_obstacle_ids"]) > int(registration["obstacle_plan_cap"]):
        raise ValueError("obstacle subset exceeds the 100-plan")
    found = [(record["kind"], int(record["episode_id"])) for record in records]
    if len(found) != len(set(found)):
        raise ValueError("duplicate training episode id")
    extra = [item for item in found if item not in expected]
    if extra:
        raise ValueError(f"undeclared training trajectories: {extra}")
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"missing predeclared trajectories: {missing}")
    episodes = []
    with h5py.File(dest / "trajectory.h5", "w") as handle:
        for record in sorted(records, key=lambda item: int(item["episode_id"])):
            if not record["native_success"] or not record["held_at_success"]:
                raise ValueError("training trajectory is not a held native success")
            if record["kind"] == "obstacle" and (
                record["obstacle_collision"] or record["obstacle_crossing"]
            ):
                raise ValueError("unsafe obstacle trajectory")
            key = f"traj_{int(record['episode_id'])}"
            group = handle.create_group(key)
            group.attrs["kind"] = record["kind"]
            group.create_dataset("obs", data=np.asarray(record["obs"], dtype=np.float32))
            group.create_dataset("actions", data=np.asarray(record["actions"], dtype=np.float32))
            group.create_dataset("terminated", data=np.asarray(record["terminated"], dtype=np.uint8))
            group.create_dataset("truncated", data=np.asarray(record["truncated"], dtype=np.uint8))
            success = np.zeros(len(record["obs"]), dtype=np.uint8)
            success[-1] = 1
            group.create_dataset("success", data=success)
            episodes.append({
                "episode_id": int(record["episode_id"]),
                "kind": record["kind"],
                "scene_id": record["scene_id"],
                "layout": record["layout"],
                "native_success": True,
                "held_at_success": True,
                "obstacle_collision": bool(record["obstacle_collision"]),
                "obstacle_crossing": bool(record["obstacle_crossing"]),
            })
    manifest = {"episodes": episodes, "format": "pickcube_pilot_demos_v1"}
    (dest / "trajectory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dest


def _window_slices(length, obs_horizon=2, pred_horizon=16):
    before, after = obs_horizon - 1, pred_horizon - obs_horizon
    return [(start, start + pred_horizon) for start in range(-before, length - pred_horizon + after)]


class PilotTrajectoryDataset(torch.utils.data.Dataset):
    """Strict windows for one predeclared class. Refuses extra trajectories."""

    def __init__(self, data_path, manifest_path, kind, expected_ids):
        self.kind = kind
        self.expected_ids = [int(item) for item in expected_ids]
        manifest = json.loads(Path(manifest_path).read_text())
        episodes = manifest["episodes"]
        ids = [int(item["episode_id"]) for item in episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest has duplicate episode ids")
        selected = [int(item["episode_id"]) for item in episodes if item["kind"] == kind]
        if selected != self.expected_ids:
            raise ValueError(f"{kind} episode ids are not the predeclared subset")
        other = [item for item in episodes if item["kind"] not in ("direct", "obstacle")]
        if other:
            raise ValueError("undeclared training kind")
        self.h5_sha256 = sha256_file(data_path)
        self.manifest_sha256 = sha256_file(manifest_path)
        self.observations, self.actions, self.slices = [], [], []
        with h5py.File(data_path, "r") as handle:
            if set(handle.keys()) != {f"traj_{item}" for item in ids}:
                raise ValueError("HDF5 contains undeclared trajectories")
            for episode_id in self.expected_ids:
                group = handle[f"traj_{episode_id}"]
                if group.attrs.get("kind") != kind:
                    raise ValueError(f"traj_{episode_id} kind does not match the selection")
                obs, actions = group["obs"][:], group["actions"][:]
                if obs.shape[1] != 45 or actions.shape[1] != 7 or len(obs) != len(actions) + 1:
                    raise ValueError(f"traj_{episode_id} schema mismatch")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError(f"traj_{episode_id} is non-finite")
                if np.max(np.abs(actions)) > 1.00001:
                    raise ValueError(f"traj_{episode_id} action outside [-1, 1]")
                if not bool(group["success"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"traj_{episode_id} lacks successful non-truncated evidence")
                self.observations.append(torch.as_tensor(obs, dtype=torch.float32))
                self.actions.append(torch.as_tensor(actions, dtype=torch.float32))
                for start, end in _window_slices(len(actions)):
                    self.slices.append((len(self.actions) - 1, start, end))
        self.trajectory_ids = list(self.expected_ids)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start, end = self.slices[index]
        obs, actions = self.observations[trajectory], self.actions[trajectory]
        length = len(actions)
        obs_seq = obs[max(0, start):start + 2]
        act_seq = actions[max(0, start):end]
        if start < 0:
            obs_seq = torch.cat((obs_seq[:1].repeat(-start, 1), obs_seq))
            act_seq = torch.cat((act_seq[:1].repeat(-start, 1), act_seq))
        if end > length:
            pad = torch.cat((torch.zeros(6), act_seq[-1, 6:7]))
            act_seq = torch.cat((act_seq, pad.repeat(end - length, 1)))
        if self.kind == "direct":
            obs_seq = obs_seq[..., :42]
        return {"observations": obs_seq, "actions": act_seq}


def _state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _save_base(path, agent, ema, step, payload):
    import copy
    path = Path(path)
    model_state = {key: value.detach().cpu() for key, value in agent.state_dict().items()}
    ema_agent = copy.deepcopy(agent)
    ema.copy_to(ema_agent.parameters())
    ema_state = {key: value.detach().cpu() for key, value in ema_agent.state_dict().items()}
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "format": BASE_FORMAT, "contract": contract(), "weights": {"model": model_state, "ema": ema_state},
        "step": int(step), "selected_weights": "ema", "payload": payload,
    }, tmp)
    os.replace(tmp, path)


def train_base(demo_path, manifest_path, run_dir, checkpoint_dir, registration, device="cuda"):
    from torch.utils.data import DataLoader, RandomSampler
    training = registration["training"]
    if training["base_class"] != "direct" or training["selected_weights"] != SELECTED_WEIGHTS:
        raise ValueError("base training contract mismatch")
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(training["seed"]); np.random.seed(training["seed"]); torch.manual_seed(training["seed"])
    dataset = PilotTrajectoryDataset(
        demo_path, manifest_path, "direct", registration["selected_direct_ids"])
    if len(dataset) < training["batch_size"]:
        raise RuntimeError("direct windows are fewer than the pre-registered batch size")
    agent = build_agent(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(), lr=training["lr"], betas=tuple(training["betas"]), weight_decay=training["weight_decay"])
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    lr_sched = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=training["warmup"],
        num_training_steps=training["total_iters"])
    ema = EMAModel(parameters=agent.parameters(), power=training["ema_power"])
    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    loader = DataLoader(
        dataset, batch_size=training["batch_size"], sampler=RandomSampler(dataset),
        drop_last=True, num_workers=0)
    payload = {
        "selected_class": "direct", "episodes": len(dataset.trajectory_ids),
        "selected_episode_ids": dataset.trajectory_ids, "h5_sha256": dataset.h5_sha256,
        "manifest_sha256": dataset.manifest_sha256, "windows": len(dataset),
        "training": training, "registration_version": registration["version"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    iterator = iter(loader)
    losses = []
    started = time.monotonic()
    for step in range(1, training["total_iters"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        obs = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        if obs.shape[-1] != 42:
            raise RuntimeError("base training saw obstacle channels")
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(actions),), device=device).long()
        noisy = scheduler.add_noise(actions, noise, timesteps)
        loss = torch.nn.functional.mse_loss(agent.noise_pred_net(noisy, timesteps, global_cond=obs.flatten(1)), noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_sched.step()
        ema.step(agent.parameters())
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step == training["total_iters"] or step % 1000 == 0:
            print(f"base step {step}/{training['total_iters']} loss={losses[-1]:.6f}", flush=True)
    final = checkpoint_dir / "final.pt"
    _save_base(final, agent, ema, training["total_iters"], payload)
    summary = {
        "status": "complete", "actual_updates": training["total_iters"],
        "selected_weights": "ema", "loss_first": losses[0], "loss_last": losses[-1],
        "all_losses_finite": bool(np.isfinite(losses).all()),
        "elapsed_seconds": time.monotonic() - started, "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _load_base_ema(checkpoint, registration):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = ck.get("payload") or {}
    if (ck.get("format") != BASE_FORMAT or ck.get("selected_weights") != "ema"
            or int(ck.get("step", -1)) != int(registration["training"]["total_iters"])
            or payload.get("selected_class") != "direct"
            or payload.get("selected_episode_ids") != list(registration["selected_direct_ids"])):
        raise ValueError("pilot base checkpoint is not the matched final EMA")
    agent = build_agent("cpu")
    agent.load_state_dict(ck["weights"]["ema"], strict=True)
    return ck, agent


def train_constrain(demo_path, manifest_path, base_checkpoint, run_dir, checkpoint_dir,
                    registration, device="cuda"):
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import BatchSampler, RandomSampler
    from examples.baselines.constrain.constrain_ckpt import save_constrain_checkpoint
    from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
    from examples.baselines.constrain.pickcube_constrain import (
        MODEL_CONFIG, PickCubeConditionNorm, pickcube_relative_condition,
    )
    training = registration["training"]
    if training["constrain_class"] != "obstacle" or training["constrain_init"] != "matched base final EMA":
        raise ValueError("constrain training contract mismatch")
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _, base_agent = _load_base_ema(base_checkpoint, registration)
    dataset = PilotTrajectoryDataset(
        demo_path, manifest_path, "obstacle", registration["selected_obstacle_ids"])
    if len(dataset) < training["batch_size"]:
        raise RuntimeError("obstacle windows are fewer than the pre-registered batch size")
    denoiser = build_constrain_denoiser(MODEL_CONFIG, device="cpu")
    denoiser.base_net.load_state_dict(base_agent.noise_pred_net.state_dict(), strict=True)
    denoiser.to(device)
    base_sha = _state_sha(denoiser.base_net.state_dict())
    params = denoiser.trainable_parameters()
    if any(param.requires_grad for param in denoiser.base_net.parameters()):
        raise RuntimeError("base network is trainable")
    initial = [param.detach().clone() for param in params]
    random.seed(training["seed"]); np.random.seed(training["seed"]); torch.manual_seed(training["seed"])
    optimizer = torch.optim.AdamW(
        params, lr=training["lr"], betas=tuple(training["betas"]), weight_decay=training["weight_decay"])
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    lr_sched = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=training["warmup"],
        num_training_steps=training["total_iters"])
    ema = EMAModel(parameters=params, power=training["ema_power"])
    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(RandomSampler(dataset), training["batch_size"], drop_last=True),
        num_workers=0)
    norm = PickCubeConditionNorm()
    config = {
        "model": dict(MODEL_CONFIG),
        "normalization": norm.to_dict(),
        "base_weights": {
            "source": str(base_checkpoint), "sha256": sha256_file(base_checkpoint),
            "subkey": "weights.ema", "state_sha256": base_sha, "updates": training["total_iters"],
        },
        "data_identity": {
            "selected_class": "obstacle", "episodes": len(dataset.trajectory_ids),
            "selected_episode_ids": dataset.trajectory_ids, "h5_sha256": dataset.h5_sha256,
            "manifest_sha256": dataset.manifest_sha256,
        },
        "training": training,
        "selected_weights": SELECTED_WEIGHTS,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    iterator = iter(loader)
    losses = []
    started = time.monotonic()
    for step in range(1, training["total_iters"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        obs = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        if obs.shape[-1] != 45:
            raise RuntimeError("constrain training did not see the obstacle channels")
        hint = pickcube_relative_condition(obs, norm)[:, None, :].expand(-1, actions.shape[1], -1)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(actions),), device=device)
        noisy = scheduler.add_noise(actions, noise, timesteps)
        loss = torch.nn.functional.mse_loss(
            denoiser(noisy, timesteps, global_cond=obs[..., :42].flatten(1), hint=hint), noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_sched.step()
        ema.step(params)
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step == training["total_iters"] or step % 1000 == 0:
            print(f"constrain step {step}/{training['total_iters']} loss={losses[-1]:.6f}", flush=True)
    if _state_sha(denoiser.base_net.state_dict()) != base_sha:
        raise RuntimeError("frozen base changed during constrain training")
    if not any(not torch.equal(param, start) for param, start in zip(denoiser.trainable_parameters(), initial)):
        raise RuntimeError("control branch did not change")
    final = checkpoint_dir / "final.pt"
    save_constrain_checkpoint(
        final, denoiser, ema, optimizer, lr_sched, training["total_iters"],
        {"torch": torch.get_rng_state()}, config, checkpoint_format=CONSTRAIN_FORMAT)
    summary = {
        "status": "complete", "actual_updates": training["total_iters"],
        "selected_weights": SELECTED_WEIGHTS, "base_unchanged": True,
        "control_parameter_changed": True, "loss_first": losses[0], "loss_last": losses[-1],
        "all_losses_finite": bool(np.isfinite(losses).all()),
        "elapsed_seconds": time.monotonic() - started, "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def compare_ticket01(contact_rows):
    phases, links = {}, {}
    for row in contact_rows:
        if not row:
            continue
        phases[row.get("phase")] = phases.get(row.get("phase"), 0) + 1
        links[row.get("link")] = links.get(row.get("link"), 0) + 1
    return {
        "pilot_first_contact_phase_counts": phases,
        "pilot_first_contact_link_counts": links,
        "ticket01": TICKET01_REFERENCE,
        "formal_performance_claim": False,
    }


def layouts_from_manifest(path):
    payload = json.loads(Path(path).read_text())
    return [layout_view(scene) for scene in payload["scenes"]]


def collect_split(output_dir, *, seed, target, max_attempts, max_wall_seconds,
                  per_attempt_deadline, exclude_layouts, cylinder=True):
    from examples.baselines.constrain.pickcube_transport_detour import generate

    seen = [layout_view(row) for row in exclude_layouts]

    def reject(metadata):
        current = layout_view(metadata)
        for other in seen:
            if layouts_conflict(current, other, cylinder=cylinder):
                return "layout_duplicate"
        return None

    def note_accepted(row):
        seen.append(layout_view(row))

    return generate(
        output_dir, seed=seed, target=target, max_attempts=max_attempts,
        max_wall_seconds=max_wall_seconds, per_attempt_deadline=per_attempt_deadline,
        reject_metadata=reject, note_accepted=note_accepted, demo_dir=None,
    )


def merge_scene_catalogs(sources, dest, seed):
    """Copy accepted scenes into one catalog. Refuses layout conflicts. Does not invent scenes."""
    import shutil

    import h5py

    from examples.baselines.constrain.pickcube_transport_detour import _write_manifest

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    layouts = []
    with h5py.File(dest / "scenes.h5", "w") as out:
        for source in sources:
            source = Path(source)
            manifest = json.loads((source / "manifest.json").read_text())
            with h5py.File(source / "scenes.h5", "r") as src:
                for row in manifest["scenes"]:
                    layout = layout_view(row)
                    if any(layouts_conflict(layout, other) for other in layouts):
                        raise ValueError("merged catalogs are not layout-independent")
                    layouts.append(layout)
                    scene_id = f"pickcube-{len(rows):03d}"
                    src.copy(row["scene_id"], out, scene_id)
                    copied = dict(row)
                    copied["scene_id"] = scene_id
                    rows.append(copied)
    first = Path(sources[0])
    shutil.copy(first / "source_identity.json", dest / "source_identity.json")
    (dest / "config.json").write_text(json.dumps({
        "merged_from": [str(path) for path in sources],
        "generation_seed": seed,
        "scene_count": len(rows),
    }, indent=2, sort_keys=True) + "\n")
    _write_manifest(dest, rows, seed)
    return len(rows)


def records_from_scene_dir(scene_dir, registration):
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    if len(manifest["scenes"]) != registration["train_obstacle_count"]:
        raise ValueError("training scene count is not the predeclared obstacle subset")
    records = []
    with h5py.File(scene_dir / "scenes.h5", "r") as handle:
        for index, scene in enumerate(manifest["scenes"]):
            group = handle[scene["scene_id"]]
            if "direct_actions" not in group or "direct_obs" not in group:
                raise ValueError(f"{scene['scene_id']} is missing the matching direct trajectory")
            expert = scene["expert"]
            if not expert.get("direct_transport_blocked") or expert.get("obstacle_collision") is not False:
                raise ValueError(f"{scene['scene_id']} failed the expert premise")
            layout = layout_view(scene)
            direct_id = registration["selected_direct_ids"][index]
            obstacle_id = registration["selected_obstacle_ids"][index]
            records.append({
                "kind": "direct", "episode_id": direct_id, "scene_id": scene["scene_id"],
                "obs": np.asarray(group["direct_obs"]), "actions": np.asarray(group["direct_actions"]),
                "terminated": np.asarray(group["direct_terminated"]),
                "truncated": np.asarray(group["direct_truncated"]),
                "native_success": bool(int(group.attrs["direct_native_success"])),
                "held_at_success": bool(int(group.attrs["direct_held_at_success"])),
                "obstacle_collision": False, "obstacle_crossing": False, "layout": layout,
            })
            records.append({
                "kind": "obstacle", "episode_id": obstacle_id, "scene_id": scene["scene_id"],
                "obs": np.asarray(group["observations"]), "actions": np.asarray(group["expert_actions"]),
                "terminated": np.asarray(group["obstacle_terminated"]),
                "truncated": np.asarray(group["obstacle_truncated"]),
                "native_success": bool(expert.get("native_success")),
                "held_at_success": bool(expert.get("held_at_success")),
                "obstacle_collision": bool(expert.get("obstacle_collision")),
                "obstacle_crossing": bool(expert.get("obstacle_crossing")),
                "layout": layout,
            })
    return records


def _json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class PilotBaseAdapter:
    def __init__(self, checkpoint, registration, device="cpu"):
        ck, self.agent = _load_base_ema(checkpoint, registration)
        self.agent.to(device).eval()
        self.device = torch.device(device)
        self.metadata = {"format": BASE_FORMAT, "selected_weights": "ema",
                         "checkpoint_sha256": sha256_file(checkpoint)}

    def get_action(self, queued):
        obs = torch.as_tensor(queued, dtype=torch.float32, device=self.device)
        if obs.shape[-2:] != (2, 45):
            raise ValueError(f"expected (2,45) observations, got {tuple(obs.shape)}")
        if obs.ndim == 2:
            obs = obs[None]
        with torch.no_grad():
            return self.agent.get_action(obs[..., :42])


class PilotConstrainAdapter:
    def __init__(self, checkpoint, registration, device="cpu"):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.training_utils import EMAModel
        from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
        from examples.baselines.constrain.pickcube_constrain import MODEL_CONFIG, PickCubeConditionNorm
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ck.get("config") or {}
        data = config.get("data_identity") or {}
        if (ck.get("format") != CONSTRAIN_FORMAT or int(ck.get("step", -1)) != registration["training"]["total_iters"]
                or config.get("selected_weights") != SELECTED_WEIGHTS
                or data.get("selected_class") != "obstacle"
                or data.get("selected_episode_ids") != list(registration["selected_obstacle_ids"])):
            raise ValueError("pilot constrain checkpoint failed strict load")
        self.denoiser = build_constrain_denoiser(MODEL_CONFIG, device=device)
        self.denoiser.load_state_dict(ck["model"], strict=True)
        ema = EMAModel(parameters=self.denoiser.trainable_parameters(), power=0.75)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(self.denoiser.trainable_parameters())
        self.denoiser.eval()
        self.device = torch.device(device)
        self.norm = PickCubeConditionNorm.from_dict(config["normalization"])
        self.scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        self.scheduler.set_timesteps(100)
        self.metadata = {"format": CONSTRAIN_FORMAT, "selected_weights": SELECTED_WEIGHTS,
                         "checkpoint_sha256": sha256_file(checkpoint)}

    def get_action(self, queued):
        from examples.baselines.constrain.pickcube_constrain import pickcube_relative_condition
        obs = torch.as_tensor(queued, dtype=torch.float32, device=self.device)
        if obs.ndim == 2:
            obs = obs[None]
        hint = pickcube_relative_condition(obs, self.norm)[:, None, :].expand(-1, 16, -1)
        sample = torch.randn(obs.shape[0], 16, 7, device=self.device)
        cond = obs[..., :42].flatten(start_dim=obs.ndim - 2)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                prediction = self.denoiser(sample, timestep, global_cond=cond, hint=hint, enabled=True)
                sample = self.scheduler.step(prediction, timestep, sample).prev_sample
        action = sample[:, 1:9]
        if bool((action.abs() > 1.000001).any()):
            raise ValueError("constrain action left [-1, 1]")
        return action


def _bool_info(info, name):
    value = info[name]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


class EpisodeDeadline(TimeoutError):
    pass


def _episode_deadline(_signum, _frame):
    raise EpisodeDeadline("episode wall exceeded")


def run_pilot_episode(adapter, scene_set, index, device, max_steps=500):
    from examples.baselines.constrain.diagnose_pickcube_contact import classify_contact_phase
    from examples.baselines.constrain.evaluate_pickcube_paired import classify_termination
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    from examples.baselines.constrain.pickcube_scenes import _tree_hash

    scene = scene_set.load(index)
    geometry = scene["metadata"]["cylinder"]
    env = make_pickcube_env(obstacle_radius=geometry["radius"], obstacle_height=geometry["height"])
    try:
        _, obs = scene_set.restore(index, env)
        obs = np.asarray(obs.detach().cpu() if torch.is_tensor(obs) else obs, dtype=np.float32).reshape(-1)
        if _tree_hash(env.unwrapped.get_state_dict()) != scene["metadata"]["state_sha256"]:
            raise ValueError("restored state mismatch")
        if not np.array_equal(obs, scene["observation"].reshape(-1)):
            raise ValueError("restored observation mismatch")
        history = np.repeat(obs[None, :], 2, axis=0)
        torch.manual_seed(int(scene["metadata"]["seed"]))
        elapsed = 0
        collision = crossing = grasped = lifted = transported = False
        initial_cube_z = float(env.unwrapped.cube.pose.p[0, 2].detach().cpu())
        goal_xy = env.unwrapped.goal_site.pose.p[0, :2].detach().cpu().numpy()
        first_contact = None
        native_success = terminated = truncated = False
        termination = None
        while elapsed < max_steps:
            prediction = adapter.get_action(history)
            for action in prediction[0][:min(8, max_steps - elapsed)]:
                action_cpu = action.detach().cpu().numpy()
                obs, _, step_terminated, step_truncated, info = env.step(action_cpu[None])
                elapsed += 1
                terminated = bool(terminated or step_terminated)
                truncated = bool(truncated or step_truncated)
                obs = np.asarray(obs.detach().cpu() if torch.is_tensor(obs) else obs, dtype=np.float32).reshape(-1)
                history = np.concatenate((history[1:], obs[None]), axis=0)
                step_success = _bool_info(info, "native_success")
                native_success |= step_success
                step_collision = _bool_info(info, "obstacle_collision")
                collision |= step_collision
                crossing |= _bool_info(info, "obstacle_crossing")
                grasped_now = _bool_info(info, "is_grasped")
                grasped |= grasped_now
                cube = env.unwrapped.cube.pose.p[0].detach().cpu().numpy()
                lifted_now = bool(cube[2] > initial_cube_z + 0.04)
                near_goal = bool(np.linalg.norm(cube[:2] - goal_xy) < 0.10)
                lifted |= lifted_now
                transported |= near_goal
                if step_collision and first_contact is None:
                    stored = getattr(env.unwrapped, "_first_obstacle_contact", None) or {}
                    first_contact = {
                        "step": elapsed,
                        "link": stored.get("other"),
                        "phase": classify_contact_phase(
                            grasped=grasped_now, lifted=lifted_now, near_goal=near_goal,
                            native_success=step_success,
                        ),
                    }
                termination = classify_termination(
                    step_success, step_terminated, step_truncated, elapsed, max_steps)
                if termination is not None:
                    break
            if termination is not None:
                break
        if termination is None:
            termination = classify_termination(native_success, terminated, truncated, elapsed, max_steps)
        return {
            "scene_index": index, "scene_id": scene["metadata"]["scene_id"],
            "seed": int(scene["metadata"]["seed"]), "status": "evaluated",
            "restored_state_sha256": scene["metadata"]["state_sha256"],
            "first_contact": first_contact,
            "result": {
                "termination_reason": termination, "elapsed_steps": elapsed,
                "terminated": terminated, "truncated": truncated,
                "native_success": native_success,
                "safe_success": bool(native_success and not collision and not crossing),
                "obstacle_contact": collision, "xy_forbidden_crossing": crossing,
                "phase": ("task_success" if native_success else "no_grasp" if not grasped
                          else "grasp_no_lift" if not lifted else "lift_no_transport"
                          if not transported else "transport_or_place_failed"),
            },
        }
    finally:
        env.close()


def summarize_pilot(rows, requested):
    evaluated = [row["result"] for row in rows if row.get("status") == "evaluated"]
    count = len(evaluated)
    return {
        "requested_denominator": requested, "evaluated": count,
        "runtime_errors": requested - count,
        "task_success_rate": sum(item["native_success"] for item in evaluated) / requested,
        "safe_success_rate": sum(item["safe_success"] for item in evaluated) / requested,
        "obstacle_contact_rate": sum(item["obstacle_contact"] for item in evaluated) / requested,
        "xy_crossing_rate": sum(item["xy_forbidden_crossing"] for item in evaluated) / requested,
        "task_success_count": sum(item["native_success"] for item in evaluated),
        "safe_success_count": sum(item["safe_success"] for item in evaluated),
        "obstacle_contact_count": sum(item["obstacle_contact"] for item in evaluated),
        "xy_crossing_count": sum(item["xy_forbidden_crossing"] for item in evaluated),
        "complete": count == requested,
    }


def paired_counts(left_rows, right_rows, field):
    left = [bool(row.get("result", {}).get(field, False)) for row in left_rows]
    right = [bool(row.get("result", {}).get(field, False)) for row in right_rows]
    return {
        "both": sum(a and b for a, b in zip(left, right)),
        "base_only": sum(a and not b for a, b in zip(left, right)),
        "constrain_only": sum(b and not a for a, b in zip(left, right)),
        "neither": sum(not a and not b for a, b in zip(left, right)),
    }


def freeze_pilot_identity(repo_root=None):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = (
        "examples/baselines/constrain/pickcube_learning_pilot.py",
        "examples/baselines/constrain/pickcube_transport_detour.py",
        "examples/baselines/constrain/pickcube_expert.py",
        "examples/baselines/constrain/pickcube_base_dp.py",
        "examples/baselines/constrain/pickcube_constrain.py",
        "examples/baselines/constrain/constrain_denoiser.py",
    )
    hashed = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in files}
    return {
        "basis": "working-tree bytes frozen before pilot runs",
        "git_head_is_code_identity": False,
        "files": hashed,
        "files_sha256": hashlib.sha256(json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "preregister", "collect-train", "collect-dev", "package", "train-base",
        "train-constrain", "evaluate", "gate",
    ))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--version", default="r1")
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--reason")
    parser.add_argument("--prior-version")
    parser.add_argument("--dedup-mode", default="cylinder")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--demo-dir", type=Path)
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--constrain-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--max-wall-seconds", type=float, default=2400)
    parser.add_argument("--per-attempt-deadline", type=float, default=150)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "preregister":
        exp = args.registration.parent
        stamp = exp.name.split("_pickcube")[0]
        payload = build_preregistration(
            args.round_index, args.version, {
                "exp": str(exp),
                "train_scenes": str(exp / "train_scenes"),
                "dev_scenes": str(exp / "dev_scenes"),
                "demos": f"demos/PickCube-transport-detour/{stamp}_pilot_r{args.round_index}",
                "ckpt_base": f"outputs/ckpt/{stamp}_pickcube_pilot_base",
                "ckpt_constrain": f"outputs/ckpt/{stamp}_pickcube_pilot_constrain",
                "train_base": str(exp / "train_base"),
                "train_constrain": str(exp / "train_constrain"),
                "paired": str(exp / "paired"),
            }, reason=args.reason, prior_version=args.prior_version, dedup_mode=args.dedup_mode)
        payload["source_identity"] = freeze_pilot_identity()
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    registration = load_preregistration(args.registration)
    if args.command == "collect-train":
        spec = registration["collection"]
        if (args.seed != spec["train_seed"] or args.max_attempts != spec["train_max_attempts"]
                or args.max_wall_seconds != spec["train_wall_seconds"]
                or args.per_attempt_deadline != spec["per_attempt_deadline"]):
            raise SystemExit("collect-train budget does not match preregistration")
        exclude = layouts_from_manifest(PRIOR_MANIFEST)
        report = collect_split(
            args.output_dir, seed=spec["train_seed"], target=registration["train_obstacle_count"],
            max_attempts=spec["train_max_attempts"], max_wall_seconds=spec["train_wall_seconds"],
            per_attempt_deadline=spec["per_attempt_deadline"], exclude_layouts=exclude,
            cylinder=registration["layout_dedup"]["cylinder_proximity_is_conflict"])
        print(json.dumps(report, indent=2), flush=True)
        if not report["complete"]:
            raise SystemExit("training collection incomplete")
        return report
    if args.command == "collect-dev":
        exclude = layouts_from_manifest(PRIOR_MANIFEST)
        for path in args.exclude_manifest:
            exclude.extend(layouts_from_manifest(path))
        spec = dict(registration["collection"])
        addendum_path = args.registration.parent / "collection_budget_addendum.json"
        if addendum_path.exists():
            addendum = json.loads(addendum_path.read_text())
            if addendum.get("reason") != "measured_yield_below_preregistered_attempt_budget":
                raise SystemExit("unrecognized collection budget addendum")
            if addendum.get("does_not_change") != ["counts", "dedup", "gate", "training"]:
                raise SystemExit("collection addendum changes a frozen rule")
            spec.update(addendum["collection"])
        if (args.seed != spec["dev_seed"] or args.max_attempts != spec["dev_max_attempts"]
                or args.max_wall_seconds != spec["dev_wall_seconds"]
                or args.per_attempt_deadline != spec["per_attempt_deadline"]):
            raise SystemExit("collect-dev budget does not match preregistration")
        report = collect_split(
            args.output_dir, seed=spec["dev_seed"], target=registration["dev_scene_count"],
            max_attempts=spec["dev_max_attempts"], max_wall_seconds=spec["dev_wall_seconds"],
            per_attempt_deadline=spec["per_attempt_deadline"], exclude_layouts=exclude,
            cylinder=registration["layout_dedup"]["cylinder_proximity_is_conflict"])
        print(json.dumps(report, indent=2), flush=True)
        if not report["complete"]:
            raise SystemExit("dev collection incomplete")
        return report
    if args.command == "package":
        records = records_from_scene_dir(args.scene_dir, registration)
        dest = package_records(records, args.demo_dir, registration)
        print(dest, flush=True)
        return dest
    if args.command == "train-base":
        return train_base(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.run_dir, args.checkpoint_dir, registration, args.device)
    if args.command == "train-constrain":
        return train_constrain(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.base_checkpoint, args.run_dir, args.checkpoint_dir, registration, args.device)
    if args.command == "evaluate":
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
        catalog = TransportDetourSceneSet(args.scene_dir)
        expected_dev = registration["dev_scene_count"]
        count_addendum = args.registration.parent / "dev_count_addendum.json"
        if count_addendum.exists():
            recorded = json.loads(count_addendum.read_text())
            if recorded.get("reason") != "yield_saturated_under_frozen_dedup":
                raise ValueError("unrecognized dev-count addendum")
            if recorded.get("does_not_change_gate_thresholds") is not True:
                raise ValueError("dev-count addendum must keep the gate thresholds")
            expected_dev = int(recorded["achieved_dev_count"])
        if len(catalog) != expected_dev:
            raise ValueError("dev scene count does not match the frozen denominator")
        train_layouts = layouts_from_manifest(args.exclude_manifest[0]) if args.exclude_manifest else []
        dev_layouts = layouts_from_manifest(args.scene_dir / "manifest.json")
        prior = layouts_from_manifest(PRIOR_MANIFEST)
        cylinder = registration["layout_dedup"]["cylinder_proximity_is_conflict"]

        def _hit(rows, others):
            return any(layouts_conflict(left, right, cylinder=cylinder) for left in rows for right in others)

        def _internal(rows):
            return any(layouts_conflict(rows[i], rows[j], cylinder=cylinder)
                       for i in range(len(rows)) for j in range(i + 1, len(rows)))

        if train_layouts and (
            _internal(train_layouts) or _internal(dev_layouts) or _hit(train_layouts, dev_layouts)
            or _hit(train_layouts, prior) or _hit(dev_layouts, prior)
        ):
            raise ValueError("dev scenes are not independent of train or prior source")
        adapters = {
            "base_dp": PilotBaseAdapter(args.base_checkpoint, registration, args.device),
            "constrain_dp": PilotConstrainAdapter(args.constrain_checkpoint, registration, args.device),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        summaries = {}
        stored = {}
        import signal
        episode_wall = registration["eval"]["episode_wall_seconds"]
        for name, adapter in adapters.items():
            rows = []
            policy_dir = args.output_dir / name
            policy_dir.mkdir()
            for index in range(len(catalog)):
                previous = signal.signal(signal.SIGALRM, _episode_deadline)
                signal.setitimer(signal.ITIMER_REAL, episode_wall)
                try:
                    row = run_pilot_episode(adapter, catalog, index, args.device, max_steps=500)
                except Exception as exc:
                    row = {"scene_index": index, "status": "runtime_error",
                           "error_kind": "wall_timeout" if isinstance(exc, EpisodeDeadline) else "unexpected",
                           "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                _json_atomic(policy_dir / f"episode_{index:03d}.json", row)
                rows.append(row)
                print(f"{name} scene {index} {row.get('status')}", flush=True)
            summaries[name] = summarize_pilot(rows, len(catalog))
            stored[name] = rows
            _json_atomic(policy_dir / "summary.json", summaries[name])
        paired = {
            "policies": summaries,
            "paired_counts": {
                "task_success": paired_counts(stored["base_dp"], stored["constrain_dp"], "native_success"),
                "safe_success": paired_counts(stored["base_dp"], stored["constrain_dp"], "safe_success"),
            },
            "ticket01_comparison": {
                "base_dp": compare_ticket01([row.get("first_contact") for row in stored["base_dp"]]),
                "constrain_dp": compare_ticket01([row.get("first_contact") for row in stored["constrain_dp"]]),
            },
            "formal_performance_claim": False,
        }
        _json_atomic(args.output_dir / "paired_summary.json", paired)
        print(json.dumps(summaries, indent=2), flush=True)
        return paired
    if args.command == "gate":
        paired = json.loads((args.output_dir / "paired_summary.json").read_text())
        decision = decide_gate(paired["policies"]["constrain_dp"])
        decision["ticket01_comparison"] = paired.get("ticket01_comparison")
        decision["paired_counts"] = paired.get("paired_counts")
        _json_atomic(args.output_dir / "gate.json", decision)
        print(json.dumps(decision, indent=2), flush=True)
        return decision
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
