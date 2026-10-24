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
            episode = {
                "episode_id": int(record["episode_id"]),
                "kind": record["kind"],
                "scene_id": record["scene_id"],
                "layout": record["layout"],
                "native_success": True,
                "held_at_success": True,
                "obstacle_collision": bool(record["obstacle_collision"]),
                "obstacle_crossing": bool(record["obstacle_crossing"]),
            }
            if record.get("provenance"):
                episode["provenance"] = record["provenance"]
            episodes.append(episode)
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
        "dataset_coverage": dataset_coverage(
            len(dataset), len(dataset.trajectory_ids), training["total_iters"], training["batch_size"]),
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
        "loss_weight": training.get("loss_weight"),
    }
    weight_spec = training.get("loss_weight")
    if weight_spec is not None and weight_spec.get("name") != "obstacle_proximity":
        raise ValueError("unsupported constrain loss weight")
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    iterator = iter(loader)
    losses = []
    unweighted_losses = []
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
        prediction = denoiser(noisy, timesteps, global_cond=obs[..., :42].flatten(1), hint=hint)
        unweighted = weighted_epsilon_mse(prediction, noise, None)
        if weight_spec is None:
            loss = unweighted
            mean_weight = 1.0
        else:
            sample_weight = obstacle_proximity_weight(
                obs, near_distance_m=float(weight_spec["near_distance_m"]),
                max_extra=float(weight_spec["max_extra"]))
            loss = weighted_epsilon_mse(prediction, noise, sample_weight)
            mean_weight = float(sample_weight.detach().mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_sched.step()
        ema.step(params)
        losses.append(float(loss.detach().cpu()))
        unweighted_losses.append(float(unweighted.detach().cpu()))
        if step == 1 or step == training["total_iters"] or step % 1000 == 0:
            print(
                f"constrain step {step}/{training['total_iters']} loss={losses[-1]:.6f} "
                f"unweighted={unweighted_losses[-1]:.6f} mean_weight={mean_weight:.4f}",
                flush=True,
            )
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
        "unweighted_loss_first": unweighted_losses[0], "unweighted_loss_last": unweighted_losses[-1],
        "loss_weight": weight_spec,
        "all_losses_finite": bool(np.isfinite(losses).all() and np.isfinite(unweighted_losses).all()),
        "elapsed_seconds": time.monotonic() - started, "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
        "dataset_coverage": dataset_coverage(
            len(dataset), len(dataset.trajectory_ids), training["total_iters"], training["batch_size"]),
        "base_state_sha256": base_sha,
        "strict_ema_load": True,
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
                  per_attempt_deadline, exclude_layouts, cylinder=True, within_run=True):
    from examples.baselines.constrain.pickcube_transport_detour import generate

    seen = [layout_view(row) for row in exclude_layouts]

    def reject(metadata):
        current = layout_view(metadata)
        for other in seen:
            if layouts_conflict(current, other, cylinder=cylinder):
                return "layout_duplicate"
        return None

    def note_accepted(row):
        if within_run:
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


EXPANSION_DIRECT_COUNT = 1000
EXPANSION_OBSTACLE_COUNT = 100
RETAINED_COUNT = 20
PAIR_ACCEPTANCE_CHECKS = (
    "direct_native_success", "direct_held", "direct_transport_blocked",
    "obstacle_native_success", "obstacle_held", "no_obstacle_contact",
    "no_obstacle_crossing", "join_continuous", "action_observation_contract",
)
SUCCESS_FIELDS = frozenset({"native_success", "safe_success"})
VIOLATION_FIELDS = frozenset({"obstacle_contact", "xy_forbidden_crossing"})
BASELINE_EXP = "outputs/exp/20261023_235744_pickcube_learning_pilot_r2"
RETAINED_SCENES = f"{BASELINE_EXP}/train_scenes"
RETAINED_DEMOS = "demos/PickCube-transport-detour/20261023_235744_pilot_r2"
FROZEN_DEV_SCENES = f"{BASELINE_EXP}/dev_scenes"
OBSTACLE_EXECUTION_REASON = (
    "Skipping block/safe/endpoint acceptance would admit start/goal pairs that fail "
    "those checks. Round 2 recorded 4 such post-geometry failures in 43 attempts. "
    "That would change the effective start/goal distribution relative to the 100 pairs. "
    "Obstacle execution is retained as the acceptance predicate and is not packaged "
    "for the additional 900 direct trajectories."
)


def expansion_id_plan():
    return {
        "direct": list(range(EXPANSION_DIRECT_COUNT)),
        "obstacle": list(range(1000, 1000 + EXPANSION_OBSTACLE_COUNT)),
    }


def validate_expansion_ids(direct_ids, obstacle_ids):
    direct_ids = [int(item) for item in direct_ids]
    obstacle_ids = [int(item) for item in obstacle_ids]
    if set(range(RETAINED_COUNT)) - set(direct_ids) or set(range(1000, 1020)) - set(obstacle_ids):
        raise ValueError("original direct or obstacle ids missing")
    if len(obstacle_ids) != EXPANSION_OBSTACLE_COUNT or obstacle_ids != list(range(1000, 1100)):
        raise ValueError("obstacle ids are not the frozen 100")
    if direct_ids != list(range(EXPANSION_DIRECT_COUNT)):
        raise ValueError("direct ids are not the frozen 1000")
    return True


def direct_candidate_accepted(checks):
    return all(bool(checks.get(name)) for name in PAIR_ACCEPTANCE_CHECKS)


def partition_task_independent(rows):
    kept, rejected = [], []
    for row in rows:
        if any(layouts_conflict(row, other, cylinder=False) for other in kept):
            rejected.append(row)
        else:
            kept.append(row)
    return kept, rejected


def expansion_transitions(baseline_rows, expansion_rows, fields):
    baseline = {row["scene_id"]: row for row in baseline_rows}
    expansion = {row["scene_id"]: row for row in expansion_rows}
    if set(baseline) != set(expansion):
        raise ValueError("scene ids do not match")
    scene_ids = sorted(baseline)
    report = {"matched_scene_ids": scene_ids}
    for field in fields:
        if field not in SUCCESS_FIELDS and field not in VIOLATION_FIELDS:
            raise ValueError(f"unknown transition field {field}")
        buckets = {key: 0 for key in (
            ("cleared", "introduced", "both", "neither") if field in VIOLATION_FIELDS
            else ("improved", "regressed", "both", "neither"))}
        for scene_id in scene_ids:
            left = bool(baseline[scene_id]["result"][field])
            right = bool(expansion[scene_id]["result"][field])
            if field in VIOLATION_FIELDS:
                key = ("cleared" if left and not right else "introduced" if right and not left
                       else "both" if left else "neither")
            else:
                key = ("improved" if right and not left else "regressed" if left and not right
                       else "both" if left else "neither")
            buckets[key] += 1
        report[field] = buckets
    return report


def separate_failed_attempts(failed_rows, recovery_rows):
    return {
        "failed_attempt_errors": sum(row.get("status") != "evaluated" for row in failed_rows),
        "failed_attempt_is_authoritative": False,
        "recovery_authoritative": True,
        "recovery_evaluated": sum(row.get("status") == "evaluated" for row in recovery_rows),
    }


def build_episode_index(rows, checkpoint_sha, checkpoint):
    if any(row.get("checkpoint_sha256") != checkpoint_sha for row in rows):
        raise ValueError("episode checkpoint sha does not match the index")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "episodes": [{
            "scene_index": row.get("scene_index"),
            "scene_id": row.get("scene_id"),
            "status": row.get("status"),
            "checkpoint_sha256": checkpoint_sha,
        } for row in rows],
    }


def obstacle_proximity_weight(observations, near_distance_m=0.08, max_extra=1.5):
    """Upweight windows whose latest TCP is inside ``near_distance_m`` of the cylinder.

    Weight is 1 at and beyond the cutoff and ``1 + max_extra`` at zero distance.
    The cutoff and extra come from the training-set epsilon gap below 0.08 m,
    not from development success rates.
    """
    from examples.baselines.constrain.pickcube_constrain import OBSTACLE_XY, TCP_XY

    observations = torch.as_tensor(observations)
    if near_distance_m <= 0 or max_extra < 0:
        raise ValueError("proximity weight cutoff must be positive and extra nonnegative")
    latest = observations[..., -1, :]
    distance = torch.linalg.norm(
        latest[..., list(OBSTACLE_XY)] - latest[..., list(TCP_XY)], dim=-1)
    closeness = ((near_distance_m - distance) / near_distance_m).clamp(0, 1)
    return 1 + max_extra * closeness


def weighted_epsilon_mse(prediction, noise, weight=None):
    """Epsilon MSE. ``weight`` is per sample and is normalized by its sum."""
    if weight is None:
        return torch.nn.functional.mse_loss(prediction, noise)
    per_sample = (prediction - noise).pow(2).flatten(1).mean(1)
    weight = torch.as_tensor(weight, dtype=per_sample.dtype, device=per_sample.device)
    if weight.shape != per_sample.shape or not torch.isfinite(weight).all() or float(weight.sum()) <= 0:
        raise ValueError("epsilon weights must be finite, positive in sum, and one per sample")
    return (weight * per_sample).sum() / weight.sum()


def dataset_coverage(windows, episodes, updates, batch_size):
    windows, episodes = int(windows), int(episodes)
    updates, batch_size = int(updates), int(batch_size)
    if windows < batch_size:
        raise ValueError("windows are fewer than the batch size")
    drawn = updates * batch_size
    return {
        "windows": windows, "episodes": episodes, "updates": updates, "batch_size": batch_size,
        "samples_drawn": drawn, "epochs": drawn / windows,
    }


def assert_frozen_source_identity(registration, current):
    frozen = (registration.get("source_identity") or {}).get("files_sha256")
    if current.get("files_sha256") != frozen:
        raise RuntimeError("source identity does not match preregistration")
    return current


ADDENDUM_DOES_NOT_CHANGE = [
    "counts", "task_identity_definition", "gate", "training", "geometry",
    "expert", "external_excludes", "acceptance_predicate",
]


def load_expansion_addendum(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("reason") != "measured_yield_below_preregistered_attempt_budget":
        raise ValueError("unrecognized collection addendum")
    if payload.get("does_not_change") != ADDENDUM_DOES_NOT_CHANGE:
        raise ValueError("collection addendum changes a frozen rule")
    if payload.get("scene_contract_unchanged") is not True or payload.get("formal_performance_claim") is not False:
        raise ValueError("addendum must keep the scene contract")
    direct = payload["direct_rule"]
    if direct.get("acceptance_predicate") != "full_run_attempt":
        raise ValueError("addendum changes the acceptance predicate")
    if direct.get("obstacle_execution") != "retained_as_acceptance_predicate" or direct.get("obstacle_packaged") is not False:
        raise ValueError("addendum changes obstacle acceptance or packaging")
    if int(direct.get("target")) != 900 or direct.get("external_excludes") != ["ticket03", "dev30", "retained20"]:
        raise ValueError("addendum changes counts or external excludes")
    if direct.get("within_run_mutual_exclusion") is not False or direct.get("exclude_new_pairs") is not False:
        raise ValueError("addendum does not match the recorded direct rule")
    if int(direct.get("workers")) != 4 or int(direct.get("per_attempt_deadline")) != 150:
        raise ValueError("addendum changes the frozen worker or deadline contract")
    return payload


def direct_spec_from_addendum(addendum):
    rule = addendum["direct_rule"]
    block_attempts = (int(rule["max_attempts"]) - int(rule["continuation_max_attempts"])) // len(rule["seeds"])
    block_target = int(rule["target"]) // len(rule["seeds"])
    if block_attempts * len(rule["seeds"]) + int(rule["continuation_max_attempts"]) != int(rule["max_attempts"]):
        raise ValueError("direct attempt budget does not add up")
    if block_target * len(rule["seeds"]) != int(rule["target"]):
        raise ValueError("direct block targets do not add up")
    return {
        "workers": int(rule["workers"]),
        "target": int(rule["target"]),
        "max_attempts": int(rule["max_attempts"]),
        "wall_seconds": float(rule["stage_wall_seconds"]),
        "per_attempt_deadline": int(rule["per_attempt_deadline"]),
        "blocks": [
            {
                "name": f"block_{index}", "seed": int(seed), "target": block_target,
                "max_attempts": block_attempts, "wall_seconds": float(rule["block_wall_seconds"]),
            }
            for index, seed in enumerate(rule["seeds"])
        ],
        "continuation": {
            "name": "continuation", "seed": int(rule["continuation_seed"]),
            "max_attempts": int(rule["continuation_max_attempts"]),
            "wall_seconds": float(rule["block_wall_seconds"]),
        },
    }


def resolve_running_identity(registration, current):
    frozen = (registration.get("source_identity") or {}).get("files_sha256")
    if current.get("files_sha256") == frozen:
        return {"matched": "preregistration", "files_sha256": frozen}
    recovery_path = Path(registration["paths"]["exp"]) / "recovery_identity.json"
    if not recovery_path.exists():
        raise RuntimeError("source identity does not match preregistration")
    recovery = json.loads(recovery_path.read_text())
    if recovery.get("files_sha256") != current.get("files_sha256"):
        raise RuntimeError("recovery identity does not match the running code")
    if recovery.get("reason") != "measured_yield_below_preregistered_attempt_budget":
        raise RuntimeError("unrecognized recovery identity")
    if recovery.get("generator_files_sha256") != (registration.get("generator_identity") or {}).get("files_sha256"):
        raise RuntimeError("recovery changed the generator")
    return {"matched": "recovery_identity", "files_sha256": current["files_sha256"]}


def assert_external_independence(train, dev, prior):
    if any(layouts_conflict(left, right, cylinder=False) for left in train for right in dev):
        raise ValueError("train conflicts with frozen dev")
    if any(layouts_conflict(left, right, cylinder=False) for left in train for right in prior):
        raise ValueError("train conflicts with prior")
    if any(layouts_conflict(left, right, cylinder=False) for left in dev for right in prior):
        raise ValueError("dev conflicts with prior")
    return True


def _seed_span(spec):
    spans = [(block["seed"] + 1, block["seed"] + int(block["max_attempts"])) for block in spec["blocks"]]
    continuation = spec["continuation"]
    spans.append((continuation["seed"] + 1, continuation["seed"] + int(continuation["max_attempts"])))
    return spans


def build_expansion_preregistration(version, paths, *, reason, prior_version, baseline_files=None):
    if not reason or not prior_version:
        raise ValueError("expansion requires the recorded authorization and the prior version")
    if int(EXPANSION_OBSTACLE_COUNT) > OBSTACLE_PLAN_CAP:
        raise ValueError("expansion obstacle count exceeds the 100-plan")
    pairs = {
        "workers": 2, "target": 80, "max_attempts": 400, "wall_seconds": 7200,
        "per_attempt_deadline": 150,
        "blocks": [
            {"name": "block_a", "seed": 2026108001, "target": 40, "max_attempts": 160, "wall_seconds": 3600},
            {"name": "block_b", "seed": 2026108401, "target": 40, "max_attempts": 160, "wall_seconds": 3600},
        ],
        "continuation": {"name": "continuation", "seed": 2026108801, "max_attempts": 80, "wall_seconds": 2400},
    }
    direct = {
        "workers": 4, "target": 900, "max_attempts": 3600, "wall_seconds": 16000,
        "per_attempt_deadline": 150,
        "blocks": [
            {"name": "block_a", "seed": 2026110001, "target": 225, "max_attempts": 800, "wall_seconds": 9000},
            {"name": "block_b", "seed": 2026111001, "target": 225, "max_attempts": 800, "wall_seconds": 9000},
            {"name": "block_c", "seed": 2026112001, "target": 225, "max_attempts": 800, "wall_seconds": 9000},
            {"name": "block_d", "seed": 2026113001, "target": 225, "max_attempts": 800, "wall_seconds": 9000},
        ],
        "continuation": {"name": "continuation", "seed": 2026114001, "max_attempts": 400, "wall_seconds": 5400},
    }
    for spec in (pairs, direct):
        spans = sorted(_seed_span(spec))
        if any(left[1] >= right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("collection seed ranges overlap")
        if sum(block["max_attempts"] for block in spec["blocks"]) + spec["continuation"]["max_attempts"] != spec["max_attempts"]:
            raise ValueError("attempt budget does not add up")
        if sum(block["target"] for block in spec["blocks"]) != spec["target"]:
            raise ValueError("block targets do not add up")
    training = build_preregistration(1, version, paths)["training"]
    training["differences_from_formal"] = [
        "data identity is this expansion version, not the trusted formal hashes",
        "checkpoint format is pilot-specific so formal loaders reject it",
        "expert may stop on evaluate() success, so terminated[-1] need not be 1",
        "development scenes are the already-used 30, not a held-out formal 100",
        "counts are 1000 direct and 100 obstacle on this transport-detour distribution",
        "result is an expansion effect on that development set, not a formal performance claim",
    ]
    return {
        "kind": "authorized_data_expansion",
        "round_index": "expansion",
        "round_cap": "one authorized expansion; not a third small-pilot round",
        "version": version,
        "reason": reason,
        "prior_version": prior_version,
        "prior_manifest": PRIOR_MANIFEST,
        "paths": dict(paths),
        "obstacle_plan_cap": OBSTACLE_PLAN_CAP,
        "train_obstacle_count": EXPANSION_OBSTACLE_COUNT,
        "train_direct_count": EXPANSION_DIRECT_COUNT,
        "selected_direct_ids": expansion_id_plan()["direct"],
        "selected_obstacle_ids": expansion_id_plan()["obstacle"],
        "dev_scene_count": DEV_SCENE_COUNT,
        "dev_reused": True,
        "layout_dedup": {
            "tolerance_m": LAYOUT_TOLERANCE_M,
            "metric": "chebyshev",
            "mode": "task",
            "cylinder_proximity_is_conflict": False,
            "within_run_enforced": True,
            "conflict_if": "same seed, or start and goal both within tolerance",
            "splits": "new train, frozen dev30, retained original20, and Ticket 03 prior are mutually independent",
            "exclude_prior_source": PRIOR_MANIFEST,
            "exclude_dev": FROZEN_DEV_SCENES,
            "exclude_retained": RETAINED_SCENES,
        },
        "collection_selection": {
            "retained_direct_ids": list(range(RETAINED_COUNT)),
            "retained_obstacle_ids": list(range(1000, 1020)),
            "retained_source_scenes": RETAINED_SCENES,
            "retained_source_demos": RETAINED_DEMOS,
            "retained_copied_not_mutated": True,
            "retained_reexecuted": False,
            "new_pair_count": 80,
            "new_pair_direct_ids": list(range(20, 100)),
            "new_pair_obstacle_ids": list(range(1020, 1100)),
            "additional_direct_count": 900,
            "additional_direct_ids": list(range(100, 1000)),
            "additional_direct_sampling": "same run_attempt start/goal sampler and full acceptance predicate",
            "additional_direct_obstacle_execution": "retained_as_acceptance_predicate",
            "additional_direct_obstacle_packaged": False,
            "obstacle_execution_not_skipped_because": OBSTACLE_EXECUTION_REASON,
            "generator": "examples.baselines.constrain.pickcube_transport_detour.run_attempt",
            "geometry_and_expert_unchanged": True,
            "parallel_workers_are_not_a_distribution_change": True,
            "cross_block_conflicts": "rejected at assemble by task identity, then replaced from the continuation seed",
        },
        "collection": {
            "dev_collected": False,
            "dev_scenes": FROZEN_DEV_SCENES,
            "yield_basis": {
                "source": f"{BASELINE_EXP}/train_scenes/report.json",
                "accepted": 20, "attempts": 43, "elapsed_seconds": 357.804,
                "acceptance_rate": 20 / 43,
            },
            "pairs": pairs,
            "direct": direct,
        },
        "training": training,
        "eval": {
            "max_steps": 500,
            "episode_wall_seconds": 120,
            "denominator": DEV_SCENE_COUNT,
            "noise_seed_rule": "torch.manual_seed(scene seed) before each scene",
            "scene_source": FROZEN_DEV_SCENES,
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
            "on_fail": "authorized expansion is the only extra round; pause Ticket 07; no further round",
        },
        "baseline": {
            "exp": BASELINE_EXP,
            "authoritative_base": "base_dp_recovery",
            "failed_base_attempt": "base_dp",
            "authoritative_constrain": "constrain_dp",
            "failed_base_is_not_authoritative": True,
            "base_counts": {"task": 6, "safe": 0, "contact": 30, "xy": 21},
            "constrain_counts": {"task": 11, "safe": 7, "contact": 21, "xy": 14},
            "denominator": 30,
            "files": dict(baseline_files or {}),
        },
    }


def _kill_descendants():
    import signal
    my_pid = os.getpid()
    children = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            ppid = int(fields[3])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == my_pid:
            children.append(int(entry.name))
    for pid in children:
        for killer in (os.killpg, os.kill):
            try:
                killer(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass


def copy_retained_catalog(source_dir, dest_dir, expected_h5_sha=None):
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)
    h5_path = source_dir / "scenes.h5"
    manifest_path = source_dir / "manifest.json"
    before_h5 = sha256_file(h5_path)
    before_manifest = sha256_file(manifest_path)
    if expected_h5_sha is not None and before_h5 != expected_h5_sha:
        raise ValueError("retained scenes hash does not match preregistration")
    manifest = json.loads(manifest_path.read_text())
    if [row["scene_id"] for row in manifest["scenes"]] != [f"pickcube-{index:03d}" for index in range(RETAINED_COUNT)]:
        raise ValueError("retained catalog is not the original 20")
    dest_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(h5_path, "r") as src, h5py.File(dest_dir / "scenes.h5", "w") as out:
        for row in manifest["scenes"]:
            src.copy(row["scene_id"], out, row["scene_id"])
    (dest_dir / "source_identity.json").write_bytes((source_dir / "source_identity.json").read_bytes())
    (dest_dir / "manifest.json").write_bytes(manifest_path.read_bytes())
    if sha256_file(h5_path) != before_h5 or sha256_file(manifest_path) != before_manifest:
        raise RuntimeError("retained source mutated")
    return {
        "copied": RETAINED_COUNT, "source_mutated": False,
        "scenes_sha256": before_h5, "manifest_sha256": before_manifest,
    }


def _block_report(dest):
    path = Path(dest) / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    report["resumed"] = True
    report["output_dir"] = str(dest)
    return report


def _run_collection_block(block, output_root, exclude_layouts, deadline, within_run=True):
    dest = Path(output_root) / block["name"]
    existing = _block_report(dest)
    if existing is not None:
        print(f"resume {dest} accepted={existing.get('accepted')}", flush=True)
        return existing
    if dest.exists():
        raise RuntimeError(f"{dest} exists without a report; refusing to rerun the same seeds")
    report = collect_split(
        dest, seed=block["seed"], target=block["target"], max_attempts=block["max_attempts"],
        max_wall_seconds=block["wall_seconds"], per_attempt_deadline=deadline,
        exclude_layouts=exclude_layouts, cylinder=False, within_run=within_run)
    report["output_dir"] = str(dest)
    report["resumed"] = False
    print(f"block {block['name']} {json.dumps({k: report[k] for k in ('accepted', 'attempts', 'complete')})}", flush=True)
    return report


def assemble_catalogs(sources, dest, seed, provenance, drop_conflicts=True):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    rows, located = [], []
    identities = []
    for source in sources:
        source = Path(source)
        manifest_path = source / "manifest.json"
        if not manifest_path.exists():
            continue
        identity = json.loads((source / "source_identity.json").read_text())["files_sha256"]
        identities.append({"source": str(source), "files_sha256": identity})
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["scenes"]:
            copied = dict(row)
            copied["provenance_source"] = provenance.get(str(source), provenance.get(source.name, "collected"))
            rows.append(copied)
            located.append((source / "scenes.h5", row["scene_id"], copied))
    if drop_conflicts:
        kept, rejected = partition_task_independent(rows)
    else:
        kept, rejected = list(rows), []
    kept_ids = {id(row) for row in kept}
    selected = [(h5_path, old_id, row) for (h5_path, old_id, row) in located if id(row) in kept_ids]
    rejected_rows = [row for row in rows if id(row) not in kept_ids]
    if len(selected) != len(kept) or len(rejected_rows) != len(rejected):
        raise RuntimeError("assemble partition lost a scene")
    with h5py.File(dest / "scenes.h5", "w") as out:
        for index, (h5_path, old_id, row) in enumerate(selected):
            scene_id = f"pickcube-{index:03d}"
            with h5py.File(h5_path, "r") as src:
                src.copy(old_id, out, scene_id)
            row["source_scene_id"] = old_id
            row["scene_id"] = scene_id
    first = next(Path(source) for source in sources if (Path(source) / "source_identity.json").exists())
    (dest / "source_identity.json").write_bytes((first / "source_identity.json").read_bytes())
    (dest / "config.json").write_text(json.dumps({
        "assembled_from": [str(source) for source in sources],
        "generation_seed": seed,
        "scene_count": len(selected),
        "identities": identities,
        "dedup": "task",
    }, indent=2, sort_keys=True) + "\n")
    from examples.baselines.constrain.pickcube_transport_detour import _write_manifest
    _write_manifest(dest, [row for _h5, _old, row in selected], seed)
    (dest / "rejected_layouts.json").write_text(json.dumps({
        "count": len(rejected_rows),
        "scenes": rejected_rows,
    }, indent=2, sort_keys=True) + "\n")
    return {"kept": len(selected), "rejected": len(rejected_rows)}


def _stage_excludes(registration, extra_manifests=()):
    rows = layouts_from_manifest(PRIOR_MANIFEST)
    rows.extend(layouts_from_manifest(Path(FROZEN_DEV_SCENES) / "manifest.json"))
    rows.extend(layouts_from_manifest(Path(RETAINED_SCENES) / "manifest.json"))
    for path in extra_manifests:
        if Path(path).exists():
            rows.extend(layouts_from_manifest(path))
    return rows


def collect_expansion_stage(registration, stage_name, output_root, extra_manifests=(),
                           spec_override=None, within_run=True, drop_conflicts=True):
    spec = spec_override or registration["collection"][stage_name]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    exclude = _stage_excludes(registration, extra_manifests)
    started = time.monotonic()
    reports = []
    import threading
    from concurrent.futures import ThreadPoolExecutor
    stop = threading.Event()

    def _progress():
        while not stop.wait(60):
            for journal in output_root.rglob("attempts.jsonl"):
                text = journal.read_text(errors="replace")
                print(
                    f"progress {journal.parent.name} attempts={text.count(chr(10))} "
                    f"accepted={text.count(chr(34) + 'accepted' + chr(34) + ': true')}",
                    flush=True)

    progress = threading.Thread(target=_progress, daemon=True)
    progress.start()
    try:
        with ThreadPoolExecutor(max_workers=int(spec["workers"])) as pool:
            futures = [
                pool.submit(
                    _run_collection_block, block, output_root / "blocks", exclude,
                    spec["per_attempt_deadline"], within_run)
                for block in spec["blocks"]
            ]
            for future in futures:
                reports.append(future.result())
        sources = [Path(report["output_dir"]) for report in reports if (Path(report["output_dir"]) / "manifest.json").exists()]
        assembled = output_root / "assembled"
        assemble_report = {"kept": 0, "rejected": 0}
        if sources and not assembled.exists():
            assemble_report = assemble_catalogs(
                sources, assembled, seed=spec["blocks"][0]["seed"], provenance={},
                drop_conflicts=drop_conflicts)
        elif assembled.exists():
            assemble_report = {"kept": len(layouts_from_manifest(assembled / "manifest.json")), "rejected": None, "resumed": True}
        attempts_used = sum(int(report.get("attempts") or 0) for report in reports)
        deficit = int(spec["target"]) - int(assemble_report["kept"])
        continuation_report = None
        remaining = int(spec["max_attempts"]) - attempts_used
        elapsed = time.monotonic() - started
        if deficit > 0 and remaining > 0 and elapsed < float(spec["wall_seconds"]):
            continuation = dict(spec["continuation"])
            continuation["target"] = deficit
            continuation["max_attempts"] = min(int(continuation["max_attempts"]), remaining)
            continuation["wall_seconds"] = min(
                float(continuation["wall_seconds"]), float(spec["wall_seconds"]) - elapsed)
            if continuation["max_attempts"] >= 1 and continuation["wall_seconds"] > 0:
                more_exclude = list(exclude)
                if within_run and assembled.exists():
                    more_exclude.extend(layouts_from_manifest(assembled / "manifest.json"))
                continuation_report = _run_collection_block(
                    continuation, output_root / "blocks", more_exclude, spec["per_attempt_deadline"], within_run)
                if (Path(continuation_report["output_dir"]) / "manifest.json").exists():
                    sources.append(Path(continuation_report["output_dir"]))
                    reassembled = output_root / "assembled_final"
                    if not reassembled.exists():
                        assemble_report = assemble_catalogs(
                            sources, reassembled, seed=continuation["seed"], provenance={},
                            drop_conflicts=drop_conflicts)
                        assembled = reassembled
        kept = int(assemble_report["kept"])
        audit_ok = all(report.get("restore_audit_ok", False) for report in reports if report.get("accepted"))
        if continuation_report and continuation_report.get("accepted"):
            audit_ok = audit_ok and bool(continuation_report.get("restore_audit_ok"))
        report = {
            "stage": stage_name, "target": spec["target"], "kept": kept,
            "complete": kept == int(spec["target"]) and audit_ok,
            "restore_audit_ok": audit_ok, "blocks": reports,
            "continuation": continuation_report, "assemble": assemble_report,
            "elapsed_seconds": time.monotonic() - started,
            "catalog": str(assembled if assembled.exists() else ""),
        }
        _json_atomic(output_root / "stage_report.json", report)
        return report
    finally:
        stop.set()
        _kill_descendants()


def scene_acceptance_checks(scene, group):
    expert = scene["expert"]
    actions = np.asarray(group["direct_actions"])
    obs = np.asarray(group["direct_obs"])
    contract = (
        actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
        and float(np.max(np.abs(actions))) <= 1.00001
        and obs.ndim == 2 and obs.shape[1] == 45 and obs.shape[0] == len(actions) + 1
        and np.isfinite(obs).all()
    )
    return {
        "direct_native_success": bool(int(group.attrs["direct_native_success"])),
        "direct_held": bool(int(group.attrs["direct_held_at_success"])),
        "direct_transport_blocked": bool(expert.get("direct_transport_blocked")),
        "obstacle_native_success": bool(expert.get("native_success")),
        "obstacle_held": bool(expert.get("held_at_success")),
        "no_obstacle_contact": expert.get("obstacle_collision") is False,
        "no_obstacle_crossing": expert.get("obstacle_crossing") is False,
        "join_continuous": bool((expert.get("join") or {}).get("continuous")),
        "action_observation_contract": bool(contract),
    }


def _records_from_catalog(scene_dir, assign):
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    records = []
    with h5py.File(scene_dir / "scenes.h5", "r") as handle:
        for index, scene in enumerate(manifest["scenes"]):
            group = handle[scene["scene_id"]]
            checks = scene_acceptance_checks(scene, group)
            if not direct_candidate_accepted(checks):
                raise ValueError(f"{scene['scene_id']} failed the full acceptance predicate")
            if np.asarray(group["direct_truncated"]).any() or np.asarray(group["obstacle_truncated"]).any():
                raise ValueError(f"{scene['scene_id']} is truncated")
            layout = layout_view(scene)
            provenance = {
                "source": scene.get("provenance_source", "collected"),
                "source_scene_id": scene.get("source_scene_id", scene["scene_id"]),
                "seed": scene.get("seed"),
            }
            direct_id, obstacle_id = assign(index, scene)
            direct = {
                "kind": "direct", "episode_id": direct_id, "scene_id": scene["scene_id"],
                "obs": np.asarray(group["direct_obs"]), "actions": np.asarray(group["direct_actions"]),
                "terminated": np.asarray(group["direct_terminated"]),
                "truncated": np.asarray(group["direct_truncated"]),
                "native_success": True, "held_at_success": True,
                "obstacle_collision": False, "obstacle_crossing": False,
                "layout": layout, "provenance": provenance,
            }
            records.append(direct)
            if obstacle_id is not None:
                records.append({
                    "kind": "obstacle", "episode_id": obstacle_id, "scene_id": scene["scene_id"],
                    "obs": np.asarray(group["observations"]), "actions": np.asarray(group["expert_actions"]),
                    "terminated": np.asarray(group["obstacle_terminated"]),
                    "truncated": np.asarray(group["obstacle_truncated"]),
                    "native_success": True, "held_at_success": True,
                    "obstacle_collision": False, "obstacle_crossing": False,
                    "layout": layout, "provenance": provenance,
                })
    return records


def assert_retained_arrays(original_h5, records):
    original_h5 = Path(original_h5)
    before = sha256_file(original_h5)
    with h5py.File(original_h5, "r") as handle:
        for record in records:
            if record["provenance"]["source"] != "retained_r2":
                continue
            group = handle[f"traj_{int(record['episode_id'])}"]
            if not np.array_equal(group["obs"], np.asarray(record["obs"])) or not np.array_equal(
                    group["actions"], np.asarray(record["actions"])):
                raise ValueError(f"retained traj_{record['episode_id']} does not match the original demo")
    if sha256_file(original_h5) != before:
        raise RuntimeError("original demo file mutated")
    return before


def expansion_train_records(pair_dir, direct_dir, registration):
    pair_manifest = json.loads((Path(pair_dir) / "manifest.json").read_text())
    direct_manifest = json.loads((Path(direct_dir) / "manifest.json").read_text())
    if len(pair_manifest["scenes"]) != 100 or len(direct_manifest["scenes"]) != 900:
        raise ValueError("expansion catalogs are not 100 pairs and 900 direct")
    retained_seeds = [scene["seed"] for scene in json.loads(Path(RETAINED_SCENES, "manifest.json").read_text())["scenes"]]
    if [scene["seed"] for scene in pair_manifest["scenes"][:20]] != retained_seeds:
        raise ValueError("pair catalog does not start with the original 20 seeds")
    pair_records = _records_from_catalog(
        pair_dir, lambda index, _scene: (index, 1000 + index))
    for record in pair_records:
        if int(record["episode_id"]) < 20 or 1000 <= int(record["episode_id"]) < 1020:
            record["provenance"]["source"] = "retained_r2"
        else:
            record["provenance"]["source"] = "new_pair"
    direct_records = _records_from_catalog(
        direct_dir, lambda index, _scene: (100 + index, None))
    for record in direct_records:
        record["provenance"]["source"] = "additional_direct"
    records = pair_records + direct_records
    validate_expansion_ids(
        [int(record["episode_id"]) for record in records if record["kind"] == "direct"],
        [int(record["episode_id"]) for record in records if record["kind"] == "obstacle"])
    return records


def load_episode_directory(directory):
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("episode_*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        raise ValueError(f"no episodes in {directory}")
    return rows


def metric_counts(rows):
    evaluated = [row for row in rows if row.get("status") == "evaluated" and row.get("result")]
    return {
        "denominator": len(rows),
        "evaluated": len(evaluated),
        "runtime_errors": len(rows) - len(evaluated),
        "task": sum(bool(row["result"]["native_success"]) for row in evaluated),
        "safe": sum(bool(row["result"]["safe_success"]) for row in evaluated),
        "contact": sum(bool(row["result"]["obstacle_contact"]) for row in evaluated),
        "xy": sum(bool(row["result"]["xy_forbidden_crossing"]) for row in evaluated),
    }


def require_baseline_counts(rows, expected, denominator=30):
    counts = metric_counts(rows)
    if counts["denominator"] != denominator or counts["runtime_errors"] != 0:
        raise ValueError("baseline denominator is not the frozen evaluated 30")
    actual = {key: counts[key] for key in ("task", "safe", "contact", "xy")}
    if actual != expected:
        raise ValueError(f"baseline counts {actual} do not match frozen {expected}")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "preregister", "preregister-expansion", "collect-train", "collect-dev", "package",
        "include-original", "collect-pairs", "collect-direct", "package-expansion",
        "train-base", "train-constrain", "evaluate", "gate", "compare-baseline",
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
    if args.command == "preregister-expansion":
        from examples.baselines.constrain.pickcube_transport_detour import freeze_source_identity
        exp = args.registration.parent
        stamp = exp.name.split("_pickcube")[0]
        files = {
            "retained_scenes_h5": f"{RETAINED_SCENES}/scenes.h5",
            "retained_manifest": f"{RETAINED_SCENES}/manifest.json",
            "retained_demos_h5": f"{RETAINED_DEMOS}/trajectory.h5",
            "dev_manifest": f"{FROZEN_DEV_SCENES}/manifest.json",
            "baseline_base_recovery_summary": f"{BASELINE_EXP}/paired/base_dp_recovery/summary.json",
            "baseline_constrain_summary": f"{BASELINE_EXP}/paired/constrain_dp/summary.json",
            "baseline_base_failed_summary": f"{BASELINE_EXP}/paired/base_dp/summary.json",
        }
        hashed = {key: {"path": path, "sha256": sha256_file(path)} for key, path in files.items()}
        payload = build_expansion_preregistration(
            args.version, {
                "exp": str(exp),
                "retained_scenes": str(exp / "retained_scenes"),
                "pair_scenes": str(exp / "pair_scenes"),
                "direct_scenes": str(exp / "direct_scenes"),
                "demos": f"demos/PickCube-transport-detour/{stamp}_expansion",
                "ckpt_base": f"outputs/ckpt/{stamp}_pickcube_expansion_base",
                "ckpt_constrain": f"outputs/ckpt/{stamp}_pickcube_expansion_constrain",
                "train_base": str(exp / "train_base"),
                "train_constrain": str(exp / "train_constrain"),
                "paired": str(exp / "paired"),
                "dev_scenes": FROZEN_DEV_SCENES,
            }, reason=args.reason, prior_version=args.prior_version or BASELINE_EXP, baseline_files=hashed)
        payload["source_identity"] = freeze_pilot_identity()
        payload["generator_identity"] = freeze_source_identity()
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    registration = load_preregistration(args.registration)
    if registration.get("kind") == "authorized_data_expansion" and args.command in (
            "collect-train", "collect-dev", "package"):
        raise SystemExit("expansion registration does not use the small-pilot collector")
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
    if args.command == "include-original":
        expected = ((registration.get("baseline") or {}).get("files") or {}).get("retained_scenes_h5", {}).get("sha256")
        report = copy_retained_catalog(
            RETAINED_SCENES, registration["paths"]["retained_scenes"], expected_h5_sha=expected)
        print(json.dumps(report, indent=2), flush=True)
        return report
    if args.command == "collect-pairs":
        assert_frozen_source_identity(registration, freeze_pilot_identity())
        if not (Path(registration["paths"]["retained_scenes"]) / "manifest.json").exists():
            raise SystemExit("include-original before collect-pairs")
        report = collect_expansion_stage(registration, "pairs", registration["paths"]["pair_scenes"])
        _require_new_generator(registration, report)
        if report["complete"]:
            report["merge"] = finalize_pair_catalog(registration, report)
        print(json.dumps({key: report[key] for key in ("stage", "target", "kept", "complete", "catalog") if key in report}), flush=True)
        if not report["complete"]:
            raise SystemExit("pair collection incomplete")
        return report
    if args.command == "collect-direct":
        addendum = load_expansion_addendum(args.registration.parent / "collection_budget_addendum.json")
        identity = resolve_running_identity(registration, freeze_pilot_identity())
        if not (Path(registration["paths"]["pair_scenes"]) / "merged" / "manifest.json").exists():
            raise SystemExit("direct collection requires the merged pair catalog")
        report = collect_expansion_stage(
            registration, "direct", registration["paths"]["direct_scenes"],
            spec_override=direct_spec_from_addendum(addendum), within_run=False, drop_conflicts=False)
        _require_new_generator(registration, report)
        report["running_identity"] = identity
        report["within_run_mutual_exclusion"] = False
        _json_atomic(Path(registration["paths"]["direct_scenes"]) / "stage_report.json", report)
        print(json.dumps({key: report[key] for key in ("stage", "target", "kept", "complete", "catalog", "within_run_mutual_exclusion")}), flush=True)
        if not report["complete"]:
            raise SystemExit("direct collection incomplete")
        return report
    if args.command == "package-expansion":
        pair_dir = Path(registration["paths"]["pair_scenes"]) / "merged"
        direct_report = json.loads((Path(registration["paths"]["direct_scenes"]) / "stage_report.json").read_text())
        records = expansion_train_records(pair_dir, direct_report["catalog"], registration)
        original_sha = assert_retained_arrays(f"{RETAINED_DEMOS}/trajectory.h5", records)
        if sha256_file(f"{RETAINED_SCENES}/scenes.h5") != registration["baseline"]["files"]["retained_scenes_h5"]["sha256"]:
            raise RuntimeError("retained source scenes mutated")
        layouts = [record["layout"] for record in records if record["kind"] == "direct"]
        dev = layouts_from_manifest(Path(FROZEN_DEV_SCENES) / "manifest.json")
        prior = layouts_from_manifest(PRIOR_MANIFEST)
        addendum_path = Path(registration["paths"]["exp"]) / "collection_budget_addendum.json"
        addendum = load_expansion_addendum(addendum_path) if addendum_path.exists() else None
        if addendum and addendum["direct_rule"]["within_run_mutual_exclusion"] is False:
            assert_external_independence(layouts, dev, prior)
            within_train_conflicts = len(partition_task_independent(layouts)[1])
        elif not split_is_independent(layouts, dev, prior, cylinder=False):
            raise SystemExit("expansion train layouts are not independent of dev30 or prior")
        else:
            within_train_conflicts = 0
        dest = package_records(records, registration["paths"]["demos"], registration)
        manifest = json.loads((dest / "trajectory.json").read_text())
        manifest["expansion"] = {
            "original_demo_sha256": original_sha,
            "selection": registration["collection_selection"],
            "formal_performance_claim": False,
            "within_train_task_conflicts": within_train_conflicts,
            "within_run_mutual_exclusion": False if addendum else True,
            "addendum": str(addendum_path) if addendum else None,
        }
        (dest / "trajectory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _json_atomic(Path(registration["paths"]["exp"]) / "train_layouts.json", {"scenes": [
            record["layout"] for record in records if record["kind"] == "direct"
        ]})
        print(dest, flush=True)
        return dest
    if args.command == "train-base":
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
        return train_base(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.run_dir, args.checkpoint_dir, registration, args.device)
    if args.command == "train-constrain":
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
        return train_constrain(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.base_checkpoint, args.run_dir, args.checkpoint_dir, registration, args.device)
    if args.command == "evaluate":
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
            if Path(args.scene_dir).resolve() != Path(registration["paths"]["dev_scenes"]).resolve():
                raise SystemExit("expansion eval must use the frozen dev scenes")
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

        addendum_path = args.registration.parent / "collection_budget_addendum.json"
        allow_within_train = (
            registration.get("kind") == "authorized_data_expansion" and addendum_path.exists()
            and load_expansion_addendum(addendum_path)["direct_rule"]["within_run_mutual_exclusion"] is False
        )
        if train_layouts and (
            (not allow_within_train and _internal(train_layouts)) or _internal(dev_layouts)
            or _hit(train_layouts, dev_layouts) or _hit(train_layouts, prior) or _hit(dev_layouts, prior)
        ):
            raise ValueError("dev scenes are not independent of train or prior source")
        checkpoints = {
            "base_dp": args.base_checkpoint,
            "constrain_dp": args.constrain_checkpoint,
        }
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
            checkpoint_sha = adapter.metadata["checkpoint_sha256"]
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
                row["checkpoint_sha256"] = checkpoint_sha
                _json_atomic(policy_dir / f"episode_{index:03d}.json", row)
                rows.append(row)
                print(f"{name} scene {index} {row.get('status')}", flush=True)
            summaries[name] = summarize_pilot(rows, len(catalog))
            stored[name] = rows
            _json_atomic(policy_dir / "summary.json", summaries[name])
            index = build_episode_index(rows, checkpoint_sha, checkpoint=str(checkpoints[name]))
            index["source_identity"] = freeze_pilot_identity()
            _json_atomic(policy_dir / "index.json", index)
        paired = {
            "policies": summaries,
            "paired_counts": {
                "task_success": paired_counts(stored["base_dp"], stored["constrain_dp"], "native_success"),
                "safe_success": paired_counts(stored["base_dp"], stored["constrain_dp"], "safe_success"),
                "obstacle_contact": paired_counts(stored["base_dp"], stored["constrain_dp"], "obstacle_contact"),
                "xy_forbidden_crossing": paired_counts(stored["base_dp"], stored["constrain_dp"], "xy_forbidden_crossing"),
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
    if args.command == "compare-baseline":
        report = compare_expansion_eval(registration, args.output_dir)
        _json_atomic(args.output_dir / "expansion_comparison.json", report)
        print(json.dumps({
            "expansion_counts": report["expansion_counts"],
            "transitions": report["transitions"],
            "formal_performance_claim": False,
        }, indent=2), flush=True)
        return report
    raise ValueError(args.command)


def finalize_pair_catalog(registration, stage_report):
    retained = Path(registration["paths"]["retained_scenes"])
    new_catalog = Path(stage_report["catalog"])
    merged = Path(registration["paths"]["pair_scenes"]) / "merged"
    if merged.exists():
        kept = len(json.loads((merged / "manifest.json").read_text())["scenes"])
        return {"kept": kept, "rejected": 0, "resumed": True}
    report = assemble_catalogs(
        [retained, new_catalog], merged, seed=registration["collection"]["pairs"]["blocks"][0]["seed"],
        provenance={
            str(retained): "retained_r2", retained.name: "retained_r2",
            str(new_catalog): "new_pair", new_catalog.name: "new_pair",
        })
    if report["kept"] != 100 or report["rejected"] != 0:
        raise RuntimeError(f"pair merge is not 100 independent scenes: {report}")
    return report


def _require_new_generator(registration, report):
    expected = (registration.get("generator_identity") or {}).get("files_sha256")
    if not expected:
        raise RuntimeError("preregistration has no generator identity")
    for block in report.get("blocks") or []:
        identity_path = Path(block["output_dir"]) / "source_identity.json"
        if not identity_path.exists():
            continue
        actual = json.loads(identity_path.read_text())["files_sha256"]
        if actual != expected:
            raise RuntimeError("collection generator identity does not match preregistration")
    continuation = report.get("continuation") or {}
    if continuation.get("output_dir"):
        identity_path = Path(continuation["output_dir"]) / "source_identity.json"
        if identity_path.exists():
            actual = json.loads(identity_path.read_text())["files_sha256"]
            if actual != expected:
                raise RuntimeError("continuation generator identity does not match preregistration")


def compare_expansion_eval(registration, expansion_dir):
    baseline = registration["baseline"]
    root = Path(baseline["exp"]) / "paired"
    failed = load_episode_directory(root / baseline["failed_base_attempt"])
    base = load_episode_directory(root / baseline["authoritative_base"])
    constrain = load_episode_directory(root / baseline["authoritative_constrain"])
    require_baseline_counts(base, baseline["base_counts"])
    require_baseline_counts(constrain, baseline["constrain_counts"])
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    expansion_dir = Path(expansion_dir)
    expansion_rows = {
        "base_dp": load_episode_directory(expansion_dir / "base_dp"),
        "constrain_dp": load_episode_directory(expansion_dir / "constrain_dp"),
    }
    authoritative = {"base_dp": base, "constrain_dp": constrain}
    transitions = {}
    for name, rows in expansion_rows.items():
        errors = [row.get("scene_id") for row in rows if row.get("status") != "evaluated"]
        if errors:
            transitions[name] = {"runtime_error_scene_ids": errors, "transitions": None}
        else:
            transitions[name] = {
                "runtime_error_scene_ids": [],
                "transitions": expansion_transitions(authoritative[name], rows, fields),
            }
    return {
        "denominator": 30,
        "formal_performance_claim": False,
        "attribution": "overall data expansion; direct and obstacle were enlarged together",
        "failed_base_attempt": separate_failed_attempts(failed, base),
        "baseline_counts": {"base_dp": metric_counts(base), "constrain_dp": metric_counts(constrain)},
        "expansion_counts": {name: metric_counts(rows) for name, rows in expansion_rows.items()},
        "transitions": transitions,
        "mechanisms": {
            "baseline_base": compare_ticket01([row.get("first_contact") for row in base]),
            "baseline_constrain": compare_ticket01([row.get("first_contact") for row in constrain]),
            "expansion_base": compare_ticket01([row.get("first_contact") for row in expansion_rows["base_dp"]]),
            "expansion_constrain": compare_ticket01([row.get("first_contact") for row in expansion_rows["constrain_dp"]]),
        },
    }


if __name__ == "__main__":
    main()
