"""Small-scale PlaceSphere transport-detour learnability pilot.

Formal Base/Constrain trainers stay locked to the trusted 1000/100 dataset.
This module trains a matched new-distribution pair and scores an independent
development set. It does not start Ticket 08.
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

from examples.baselines.constrain.placesphere_base_dp import (
    PlaceSphereBaseDPAdapter, build_agent, contract, sha256_file,
)

LAYOUT_TOLERANCE_M = 0.02
OBSTACLE_PLAN_CAP = 100
TRAIN_OBSTACLE_COUNT = 20
TRAIN_DIRECT_COUNT = 20
DEV_SCENE_COUNT = 30
ROUND_CAP = 2
POLICY_OBS_DIM = 39
RAW_OBS_DIM = 42
MAX_STEPS = 200
PRIOR_MANIFEST = "outputs/exp/20261023_175529_placesphere_transport_detour/manifest.json"
SELECTED_WEIGHTS = "final EMA fixed before training; no task evaluation"
BASE_FORMAT = "placesphere_pilot_base_v1"
CONSTRAIN_FORMAT = "placesphere_pilot_constrain_v1"
NATIVE_SUCCESS = "on_bin and static and not grasped"
TICKET02_REFERENCE = {
    "distribution": "old PlaceSphere obstacle scenes, not this pilot",
    "attribution": "rates are not a same-distribution improvement delta",
    "sample_not_full_set": True,
    "mechanisms": {
        "base_xy_measured": "17/17 geometric_above_top, contactless",
        "constrain_xy": "3 geometric_above_top, 2 forbidden_margin_only, 1 table-height contact",
        "contact": "base grasp-approach panda_hand top cap; constrain also release-phase sphere side",
        "terminal": "not_stabilized with timeout; incorrect_release; post_release_deviation unobserved",
    },
    "native_success": NATIVE_SUCCESS,
    "diagnostic_bins_do_not_replace_native_success": True,
}


def build_preregistration(round_index, version, paths, reason=None, prior_version=None,
                          batch_size=1024, dedup_mode="task", collection=None):
    if int(round_index) < 1 or int(round_index) > ROUND_CAP:
        raise ValueError("pilot round must be 1 or 2")
    if int(round_index) == 2 and not reason:
        raise ValueError("round 2 requires a recorded reason")
    if dedup_mode != "task":
        raise ValueError("PlaceSphere pilot dedup is task identity; cylinder proximity is not identity")
    if int(batch_size) < 1:
        raise ValueError("batch size must be positive")
    spec = {
        "train_seed": 2026108001, "dev_seed": 2026109001,
        "train_max_attempts": 80, "dev_max_attempts": 140,
        "train_wall_seconds": 2400, "dev_wall_seconds": 4200,
        "per_attempt_deadline": 160,
    }
    if collection:
        allowed = set(spec)
        unknown = set(collection) - allowed
        if unknown:
            raise ValueError(f"collection override changes a frozen field: {sorted(unknown)}")
        spec.update(collection)
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
        "native_success": NATIVE_SUCCESS,
        "layout_dedup": {
            "tolerance_m": LAYOUT_TOLERANCE_M,
            "metric": "chebyshev",
            "mode": "task",
            "cylinder_proximity_is_conflict": False,
            "within_run_enforced": True,
            "conflict_if": "same seed, or start and container both within tolerance",
            "splits": "train, dev, and Ticket 04 prior source are mutually independent",
            "exclude_prior_source": PRIOR_MANIFEST,
            "justification": (
                "Ticket 05 round 1 showed a 2 cm cylinder clause made 20/30 infeasible. "
                "PlaceSphere cylinders are searched on the same mid-segment, so that clause "
                "would collapse independent tasks. Cylinder proximity is not identity."
            ),
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
            "architecture": "existing PlaceSphere ConditionalUnet1D and ConstrainDenoiser",
            "selected_weights": SELECTED_WEIGHTS,
            "base_class": "direct",
            "constrain_class": "obstacle",
            "constrain_init": "matched base final EMA",
            "trainable": "control branch only",
            "differences_from_formal": [
                "episode counts are the predeclared subset, not 1000 direct and 100 obstacle",
                "data identity is this version, not the trusted formal hashes",
                "checkpoint format is pilot-specific so formal loaders reject it",
                "native success is on-bin, static, and ungrasped; saved actions include release, retreat, and settle",
                "observation is 39 policy / 42 raw",
                "evaluation budget is 200 steps",
                "dedup is task identity from round 1; cylinder proximity is not identity",
            ],
        },
        "collection": spec,
        "eval": {
            "max_steps": MAX_STEPS,
            "episode_wall_seconds": 120,
            "denominator": DEV_SCENE_COUNT,
            "noise_seed_rule": "torch.manual_seed and cuda.manual_seed_all(scene seed) before each scene, independently per policy",
            "action_contract": "coerce to (batch, horizon, 7); reject other widths; map pose to [-0.35, 0.35] and gripper to [-1, 1]",
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
            "on_fail": "pause Ticket 08; no third collection+training round",
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
        "selected_obstacle_ids", "native_success", "eval",
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
    if payload["native_success"] != NATIVE_SUCCESS:
        raise ValueError("native success contract changed")
    if int(payload["eval"].get("max_steps", 0)) != MAX_STEPS:
        raise ValueError("evaluation step budget is not 200")
    if payload["layout_dedup"].get("cylinder_proximity_is_conflict") is not False:
        raise ValueError("cylinder proximity is not a PlaceSphere identity conflict")
    return payload


def layout_view(value):
    if value is None:
        raise ValueError("missing layout")
    cylinder = value.get("cylinder") or {}
    container = value.get("container_xy", value.get("goal_xy"))
    center = value.get("center_xy", cylinder.get("center_xy"))
    if value.get("start_xy") is None or container is None or center is None:
        raise ValueError("layout is missing start, container, or cylinder center")
    return {
        "seed": value.get("seed"),
        "start_xy": value["start_xy"],
        "container_xy": container,
        "center_xy": center,
    }


def _within(left, right, tolerance):
    a = np.asarray(left, dtype=np.float64).reshape(2)
    b = np.asarray(right, dtype=np.float64).reshape(2)
    return bool(np.max(np.abs(a - b)) <= float(tolerance))


def layouts_conflict(left, right, tolerance=LAYOUT_TOLERANCE_M, cylinder=False):
    left, right = layout_view(left), layout_view(right)
    if left.get("seed") is not None and left.get("seed") == right.get("seed"):
        return True
    same_task = _within(left["start_xy"], right["start_xy"], tolerance) and _within(
        left["container_xy"], right["container_xy"], tolerance)
    if not cylinder:
        return same_task
    return same_task or _within(left["center_xy"], right["center_xy"], tolerance)


def split_is_independent(train, dev, prior, tolerance=LAYOUT_TOLERANCE_M, cylinder=False):
    groups = (("train", train), ("dev", dev), ("prior", prior))
    for _name, rows in groups:
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


def new_splits_isolated(train, dev, prior, tolerance=LAYOUT_TOLERANCE_M, cylinder=False):
    """Train and dev must be internally and mutually independent of each other and of prior.

    A historical prior catalog that conflicts with itself does not fail new scenes.
    """
    for rows in (train, dev):
        for index, row in enumerate(rows):
            for other in rows[index + 1:]:
                if layouts_conflict(row, other, tolerance, cylinder=cylinder):
                    return False
    for left in train:
        for right in list(dev) + list(prior):
            if layouts_conflict(left, right, tolerance, cylinder=cylinder):
                return False
    for left in dev:
        for right in prior:
            if layouts_conflict(left, right, tolerance, cylinder=cylinder):
                return False
    return True


def trajectory_has_release_retreat_settle(actions, phases, *, ungrasped, static, on_bin):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 1:
        return False
    if not (bool(ungrasped) and bool(static) and bool(on_bin)):
        return False
    if not {"retreat", "release", "settle"} <= set(phases):
        return False
    closed = bool(np.any(actions[:, 6] < -0.5))
    opened = bool(np.any(actions[:, 6] > 0.5))
    return closed and opened


def dev_premises_hold(scenes):
    for row in scenes:
        expert = row.get("expert") or {}
        clearance = expert.get("endpoint_clearance") or {}
        radius = (row.get("cylinder") or {}).get("radius")
        if not all(expert.get(key) is True for key in (
                "direct_transport_blocked", "native_success", "safe_success", "on_bin", "static",
                "ungrasped", "release_phase_reached", "retreat_phase_reached", "settle_phase_reached")):
            return False
        if expert.get("obstacle_collision") is not False or expert.get("obstacle_crossing") is not False:
            return False
        if expert.get("alternate_side_fallback") is not False:
            return False
        if (expert.get("direct_block_contact") or {}).get("phase") != "transport":
            return False
        if clearance.get("method") != "full_robot_phase_vertices" or clearance.get("paths") != ["direct", "side"]:
            return False
        if radius is None or abs(float(radius) - 0.03) > 1e-12:
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
        "ticket08": "blocked" if failures else "eligible",
    }


def compare_ticket02(rows):
    phases, locations, links, relations, mechanisms = {}, {}, {}, {}, {}
    for row in rows:
        contact = (row or {}).get("first_contact") or {}
        if contact:
            phases[contact.get("phase")] = phases.get(contact.get("phase"), 0) + 1
            locations[contact.get("location")] = locations.get(contact.get("location"), 0) + 1
            links[contact.get("link_group")] = links.get(contact.get("link_group"), 0) + 1
        relation = ((row or {}).get("crossing") or {}).get("relation")
        if relation:
            relations[relation] = relations.get(relation, 0) + 1
        mechanism = ((row or {}).get("terminal_mechanism") or {}).get("mechanism")
        if mechanism:
            mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
    return {
        "pilot_first_contact_phase_counts": phases,
        "pilot_first_contact_location_counts": locations,
        "pilot_first_contact_link_group_counts": links,
        "pilot_crossing_relation_counts": relations,
        "pilot_terminal_mechanism_counts": mechanisms,
        "ticket02": TICKET02_REFERENCE,
        "same_distribution_delta": False,
        "formal_performance_claim": False,
    }


def batched_observation(queued, raw_dim=RAW_OBS_DIM):
    obs = torch.as_tensor(queued, dtype=torch.float32)
    if obs.shape[-1] != raw_dim:
        raise ValueError(f"expected raw observation width {raw_dim}, got {tuple(obs.shape)}")
    if obs.ndim == 2:
        obs = obs[None]
    if obs.ndim != 3 or obs.shape[-2:] != (2, raw_dim):
        raise ValueError(f"expected batched (batch, 2, {raw_dim}) observations, got {tuple(obs.shape)}")
    return obs


def coerce_action_sequence(action):
    action = torch.as_tensor(action)
    if action.shape[-1] != 7:
        raise ValueError(f"expected action width 7, got {tuple(action.shape)}")
    if action.ndim == 2:
        action = action[None]
    if action.ndim != 3 or action.shape[0] < 1 or action.shape[1] < 1:
        raise ValueError(f"expected batched action horizon, got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("non-finite action")
    return action


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
            if not (record["native_success"] and record["on_bin"] and record["static"] and record["ungrasped"]):
                raise ValueError("training trajectory is not ungrasped, static, and on-bin")
            phases = [name for name, flag in (
                ("retreat", "retreat_reached"), ("release", "release_reached"), ("settle", "settle_reached"),
            ) if record.get(flag)]
            if not trajectory_has_release_retreat_settle(
                    record["actions"], phases, ungrasped=True, static=True, on_bin=True):
                raise ValueError("training trajectory is missing release, retreat, or settle")
            if record["kind"] == "obstacle" and (record["obstacle_collision"] or record["obstacle_crossing"]):
                raise ValueError("unsafe obstacle trajectory")
            width = POLICY_OBS_DIM if record["kind"] == "direct" else RAW_OBS_DIM
            obs = np.asarray(record["obs"], dtype=np.float32)
            actions = np.asarray(record["actions"], dtype=np.float32)
            if obs.shape[1] != width or actions.shape[1] != 7:
                raise ValueError("training trajectory schema mismatch")
            key = f"traj_{int(record['episode_id'])}"
            group = handle.create_group(key)
            group.attrs["kind"] = record["kind"]
            group.attrs["ungrasped"] = 1
            group.attrs["static"] = 1
            group.attrs["on_bin"] = 1
            group.attrs["settle_reached"] = 1
            group.attrs["release_reached"] = 1
            group.attrs["retreat_reached"] = 1
            group.create_dataset("obs", data=obs)
            group.create_dataset("actions", data=actions)
            group.create_dataset("terminated", data=np.asarray(record["terminated"], dtype=np.uint8))
            group.create_dataset("truncated", data=np.asarray(record["truncated"], dtype=np.uint8))
            success = np.zeros(len(obs), dtype=np.uint8)
            success[-1] = 1
            group.create_dataset("success", data=success)
            episodes.append({
                "episode_id": int(record["episode_id"]),
                "kind": record["kind"],
                "scene_id": record["scene_id"],
                "layout": record["layout"],
                "native_success": True,
                "on_bin": True,
                "static": True,
                "ungrasped": True,
                "release_reached": True,
                "retreat_reached": True,
                "settle_reached": True,
                "obstacle_collision": bool(record["obstacle_collision"]),
                "obstacle_crossing": bool(record["obstacle_crossing"]),
            })
    manifest = {"episodes": episodes, "format": "placesphere_pilot_demos_v1"}
    (dest / "trajectory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dest


def _window_slices(length, obs_horizon=2, pred_horizon=16):
    before, after = obs_horizon - 1, pred_horizon - obs_horizon
    return [(start, start + pred_horizon) for start in range(-before, length - pred_horizon + after)]


class PilotTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, manifest_path, kind, expected_ids):
        self.kind = kind
        self.expected_ids = [int(item) for item in expected_ids]
        self.width = POLICY_OBS_DIM if kind == "direct" else RAW_OBS_DIM
        manifest = json.loads(Path(manifest_path).read_text())
        episodes = manifest["episodes"]
        ids = [int(item["episode_id"]) for item in episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest has duplicate episode ids")
        selected = [int(item["episode_id"]) for item in episodes if item["kind"] == kind]
        if selected != self.expected_ids:
            raise ValueError(f"{kind} episode ids are not the predeclared subset")
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
                if obs.shape[1] != self.width or actions.shape[1] != 7 or len(obs) != len(actions) + 1:
                    raise ValueError(f"traj_{episode_id} schema mismatch")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError(f"traj_{episode_id} is non-finite")
                if np.max(np.abs(actions)) > 1.00001 or np.max(np.abs(actions[:, :6])) > 0.350001:
                    raise ValueError(f"traj_{episode_id} action outside the PlaceSphere controller bound")
                if int(group.attrs.get("ungrasped", 0)) != 1 or int(group.attrs.get("settle_reached", 0)) != 1:
                    raise ValueError(f"traj_{episode_id} is missing ungrasped settle evidence")
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
            pad = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=act_seq.dtype)
            act_seq = torch.cat((act_seq, pad.repeat(end - length, 1)))
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


def _require_cuda(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def train_base(demo_path, manifest_path, run_dir, checkpoint_dir, registration, device="cuda"):
    from torch.utils.data import DataLoader, RandomSampler
    training = registration["training"]
    if training["base_class"] != "direct" or training["selected_weights"] != SELECTED_WEIGHTS:
        raise ValueError("base training contract mismatch")
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    _require_cuda(device)
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
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, training["total_iters"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        obs = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        if obs.shape[-1] != POLICY_OBS_DIM:
            raise RuntimeError("base training saw a non-policy observation width")
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
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else None,
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
    from examples.baselines.constrain.placesphere_constrain import MODEL, placesphere_hint
    training = registration["training"]
    if training["constrain_class"] != "obstacle" or training["constrain_init"] != "matched base final EMA":
        raise ValueError("constrain training contract mismatch")
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    _require_cuda(device)
    _, base_agent = _load_base_ema(base_checkpoint, registration)
    dataset = PilotTrajectoryDataset(
        demo_path, manifest_path, "obstacle", registration["selected_obstacle_ids"])
    if len(dataset) < training["batch_size"]:
        raise RuntimeError("obstacle windows are fewer than the pre-registered batch size")
    denoiser = build_constrain_denoiser(MODEL, device="cpu")
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
    config = {
        "model": dict(MODEL),
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
        if obs.shape[-1] != RAW_OBS_DIM:
            raise RuntimeError("constrain training did not see the raw obstacle observation")
        hint = placesphere_hint(obs)[:, None, :].expand(-1, actions.shape[1], -1)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(actions),), device=device)
        noisy = scheduler.add_noise(actions, noise, timesteps)
        loss = torch.nn.functional.mse_loss(
            denoiser(noisy, timesteps, global_cond=obs[..., :POLICY_OBS_DIM].flatten(1), hint=hint), noise)
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


def layouts_from_manifest(path):
    payload = json.loads(Path(path).read_text())
    return [layout_view(scene) for scene in payload["scenes"]]


def collect_split(output_dir, *, seed, target, max_attempts, max_wall_seconds,
                  per_attempt_deadline, exclude_layouts):
    from examples.baselines.constrain.placesphere_transport_detour import generate

    seen = [layout_view(row) for row in exclude_layouts]

    def reject(metadata):
        current = layout_view(metadata)
        for other in seen:
            if layouts_conflict(current, other, cylinder=False):
                return "layout_duplicate"
        return None

    def note_accepted(row):
        seen.append(layout_view(row))

    return generate(
        output_dir, seed=seed, target=target, max_attempts=max_attempts,
        max_wall_seconds=max_wall_seconds, per_attempt_deadline=per_attempt_deadline,
        reject_metadata=reject, note_accepted=note_accepted, demo_dir=None,
    )


def _attr_or_expert(group, scene, attr, meta_key):
    if attr in group.attrs:
        return bool(int(group.attrs[attr]))
    return bool((scene.get("expert") or {}).get(meta_key))


def records_from_scene_dir(scene_dir, registration):
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    scenes = manifest["scenes"]
    if len(scenes) != registration["train_obstacle_count"]:
        raise ValueError("training scene count is not the predeclared obstacle subset")
    if not dev_premises_hold(scenes):
        raise ValueError("training scenes do not meet the Ticket 04 premises")
    records = []
    with h5py.File(scene_dir / "scenes.h5", "r") as handle:
        for index, scene in enumerate(scenes):
            group = handle[scene["scene_id"]]
            if "direct_actions" not in group or "direct_policy_observations" not in group:
                raise ValueError(f"{scene['scene_id']} is missing the matching direct trajectory")
            expert = scene["expert"]
            layout = layout_view(scene)
            direct_phases = [name for name, attr, key in (
                ("retreat", "direct_retreat_reached", "direct_retreat_phase_reached"),
                ("release", "direct_release_reached", "direct_release_phase_reached"),
                ("settle", "direct_settle_reached", "direct_settle_phase_reached"),
            ) if _attr_or_expert(group, scene, attr, key)]
            obstacle_phases = [name for name, key in (
                ("retreat", "retreat_phase_reached"),
                ("release", "release_phase_reached"),
                ("settle", "settle_phase_reached"),
            ) if expert.get(key)]
            records.append({
                "kind": "direct", "episode_id": registration["selected_direct_ids"][index],
                "scene_id": scene["scene_id"],
                "obs": np.asarray(group["direct_policy_observations"]),
                "actions": np.asarray(group["direct_actions"]),
                "terminated": np.asarray(group["direct_terminated"]),
                "truncated": np.asarray(group["direct_truncated"]),
                "native_success": _attr_or_expert(group, scene, "direct_native_success", "direct_native_success"),
                "on_bin": _attr_or_expert(group, scene, "direct_on_bin", "direct_on_bin"),
                "static": _attr_or_expert(group, scene, "direct_static", "direct_static"),
                "ungrasped": _attr_or_expert(group, scene, "direct_ungrasped", "direct_ungrasped"),
                "retreat_reached": "retreat" in direct_phases,
                "release_reached": "release" in direct_phases,
                "settle_reached": "settle" in direct_phases,
                "obstacle_collision": False, "obstacle_crossing": False, "layout": layout,
            })
            records.append({
                "kind": "obstacle", "episode_id": registration["selected_obstacle_ids"][index],
                "scene_id": scene["scene_id"],
                "obs": np.asarray(group["raw_observations"]),
                "actions": np.asarray(group["expert_actions"]),
                "terminated": np.asarray(group["terminated"]),
                "truncated": np.asarray(group["truncated"]),
                "native_success": bool(expert.get("native_success")),
                "on_bin": bool(expert.get("on_bin")),
                "static": bool(expert.get("static")),
                "ungrasped": bool(expert.get("ungrasped")),
                "retreat_reached": "retreat" in obstacle_phases,
                "release_reached": "release" in obstacle_phases,
                "settle_reached": "settle" in obstacle_phases,
                "obstacle_collision": bool(expert.get("obstacle_collision")),
                "obstacle_crossing": bool(expert.get("obstacle_crossing")),
                "layout": layout,
            })
    return records


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class PilotBaseAdapter:
    def __init__(self, checkpoint, registration, device="cpu"):
        _ck, self.agent = _load_base_ema(checkpoint, registration)
        self.agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(100)
        self.device = torch.device(device)
        self.metadata = {"format": BASE_FORMAT, "selected_weights": "ema",
                         "checkpoint_sha256": sha256_file(checkpoint)}

    def get_action(self, queued):
        obs = batched_observation(queued).to(self.device)
        with torch.no_grad():
            return coerce_action_sequence(self.agent.get_action(obs[..., :POLICY_OBS_DIM]))


class PilotConstrainAdapter:
    def __init__(self, checkpoint, registration, device="cpu"):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.training_utils import EMAModel
        from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
        from examples.baselines.constrain.placesphere_constrain import MODEL
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ck.get("config") or {}
        data = config.get("data_identity") or {}
        if (ck.get("format") != CONSTRAIN_FORMAT or int(ck.get("step", -1)) != registration["training"]["total_iters"]
                or config.get("selected_weights") != SELECTED_WEIGHTS
                or data.get("selected_class") != "obstacle"
                or data.get("selected_episode_ids") != list(registration["selected_obstacle_ids"])):
            raise ValueError("pilot constrain checkpoint failed strict load")
        self.denoiser = build_constrain_denoiser(MODEL, device=device)
        self.denoiser.load_state_dict(ck["model"], strict=True)
        ema = EMAModel(parameters=self.denoiser.trainable_parameters(), power=0.75)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(self.denoiser.trainable_parameters())
        self.denoiser.eval()
        self.device = torch.device(device)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        self.metadata = {"format": CONSTRAIN_FORMAT, "selected_weights": SELECTED_WEIGHTS,
                         "checkpoint_sha256": sha256_file(checkpoint)}

    def get_action(self, queued):
        from examples.baselines.constrain.placesphere_constrain import placesphere_hint
        obs = batched_observation(queued).to(self.device)
        hint = placesphere_hint(obs)[:, None, :].expand(-1, 16, -1)
        sample = torch.randn(obs.shape[0], 16, 7, device=self.device)
        cond = obs[..., :POLICY_OBS_DIM].flatten(start_dim=1)
        self.scheduler.set_timesteps(100)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                prediction = self.denoiser(sample, timestep, global_cond=cond, hint=hint, enabled=True)
                sample = self.scheduler.step(prediction, timestep, sample).prev_sample
        return coerce_action_sequence(sample[:, 1:9])


def _bool_info(info, name):
    value = info[name]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


class EpisodeDeadline(TimeoutError):
    pass


def _episode_deadline(_signum, _frame):
    raise EpisodeDeadline("episode wall exceeded")


def run_pilot_episode(adapter, scene_set, index, device, max_steps=MAX_STEPS):
    from examples.baselines.constrain.diagnose_placesphere_failure import (
        _terminal_pose, classify_contact_location, classify_contact_phase, classify_crossing,
        classify_terminal_mechanism, diagnostic_lifted, diagnostic_near_goal, read_obstacle_contact,
    )
    from examples.baselines.constrain.evaluate_placesphere_paired import classify_phase, classify_termination
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    scene = scene_set.load(index)
    geometry = scene["metadata"]["cylinder"]
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=max_steps, obstacle_radius=geometry["radius"], obstacle_height=geometry["height"]))
    try:
        env, restored = scene_set.restore(index, env)
        if not restored.get("bit_exact"):
            raise ValueError(f"restored state is not bit-exact: {restored.get('norms')}")
        base = env.unwrapped
        info = base.evaluate()
        from examples.baselines.constrain.placesphere_pilot import _obs_vector
        obs = np.asarray(_obs_vector(base.get_obs(info=info, unflattened=True)), dtype=np.float32).reshape(-1)
        if obs.shape != (RAW_OBS_DIM,) or not np.array_equal(obs, scene["raw_observation"].reshape(-1)):
            raise ValueError("restored raw observation mismatch")
        history = np.repeat(obs[None, :], 2, axis=0)
        seed = int(scene["metadata"]["seed"])
        torch.manual_seed(seed)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elapsed = 0
        collision = crossing = grasped = lifted = transported = released = False
        initial_z = float(base.obj.pose.p[0, 2].detach().cpu())
        goal_xy = base.bin.pose.p[0, :2].detach().cpu().numpy()
        center = np.asarray(geometry["center_xy"], dtype=np.float64)
        radius = float(geometry["radius"])
        height = float(geometry["height"])
        sphere_radius = float(base.radius)
        forbidden = float(base._forbidden_radius)
        prev_xy = base.obj.pose.p[0, :2].detach().cpu().numpy().astype(np.float64)
        prev_z = initial_z
        first_contact = None
        crossing_record = None
        release_record = None
        native_success = terminated = truncated = False
        termination = None
        info = {}
        while elapsed < max_steps:
            prediction = coerce_action_sequence(adapter.get_action(history))
            native = PlaceSphereBaseDPAdapter.map_action_ee_delta_pose(prediction)
            if native.ndim != 3 or native.shape[-1] != 7:
                raise ValueError(f"mapped action is not (batch, horizon, 7): {tuple(native.shape)}")
            for action in native[0][:min(8, max_steps - elapsed)]:
                action_cpu = action.detach().cpu().numpy().reshape(7)
                _obs, _, step_terminated, step_truncated, info = env.step(action_cpu[None])
                elapsed += 1
                terminated = bool(terminated or step_terminated)
                truncated = bool(truncated or step_truncated)
                step_success = _bool_info(info, "native_success")
                native_success = bool(native_success or step_success)
                collision_before = collision
                step_collision = _bool_info(info, "obstacle_collision")
                step_crossing = _bool_info(info, "obstacle_crossing")
                collision = bool(collision or step_collision)
                new_crossing = bool(step_crossing and not crossing)
                crossing = bool(crossing or step_crossing)
                now_grasped = _bool_info(info, "is_obj_grasped")
                grasped = bool(grasped or now_grasped)
                released = bool(released or (grasped and not now_grasped))
                sphere = base.obj.pose.p[0].detach().cpu().numpy()
                lifted_now = diagnostic_lifted(sphere[2], initial_z)
                near_goal = diagnostic_near_goal(sphere[:2], goal_xy)
                lifted = bool(lifted or lifted_now)
                transported = bool(transported or near_goal)
                phase = classify_contact_phase(
                    grasped=now_grasped, lifted=lifted_now, near_goal=near_goal,
                    released=bool(grasped and not now_grasped), native_success=step_success)
                if first_contact is None and step_collision and not collision_before:
                    try:
                        hit = read_obstacle_contact(env)
                    except Exception as exc:
                        hit = None
                        first_contact = {"step": elapsed, "contact_unreadable": f"{type(exc).__name__}: {exc}",
                                         "phase": phase}
                    if hit and hit.get("position") is not None:
                        first_contact = {
                            "step": elapsed, "entity": hit.get("entity"),
                            "link_group": hit.get("link_group"), "phase": phase,
                            "location": classify_contact_location(
                                hit["position"], center_xy=center, radius=radius, height=height),
                            "position": hit.get("position"), "surface": hit.get("surface"),
                        }
                    elif first_contact is None:
                        first_contact = {"step": elapsed, "contact_unreadable": True, "phase": phase}
                if crossing_record is None and new_crossing:
                    crossing_record = classify_crossing(
                        sphere_center_z=float(sphere[2]), previous_sphere_center_z=prev_z,
                        sphere_radius=sphere_radius, obstacle_height=height,
                        start_xy=prev_xy, end_xy=sphere[:2], center_xy=center,
                        obstacle_radius=radius, forbidden_radius=forbidden,
                        contact_on_step=bool(step_collision and not collision_before))
                    crossing_record["step"] = elapsed
                    crossing_record["phase"] = phase
                if release_record is None and grasped and not now_grasped:
                    release_record = _terminal_pose(base, info, elapsed)
                    release_record["on_bin"] = _bool_info(info, "is_obj_on_bin")
                obs = np.asarray(_obs.detach().cpu() if torch.is_tensor(_obs) else _obs, dtype=np.float32).reshape(-1)
                if obs.shape != (RAW_OBS_DIM,):
                    raise ValueError(f"step observation is not {RAW_OBS_DIM}d")
                history = np.concatenate((history[1:], obs[None]), axis=0)
                prev_xy = sphere[:2].astype(np.float64)
                prev_z = float(sphere[2])
                termination = classify_termination(step_success, step_terminated, step_truncated, elapsed, max_steps)
                if termination is not None:
                    break
            if termination is not None:
                break
        if termination is None:
            termination = classify_termination(native_success, terminated, truncated, elapsed, max_steps)
        if not info:
            info = base.evaluate()
        terminal = _terminal_pose(base, info, elapsed)
        terminal_grasped = _bool_info(info, "is_obj_grasped")
        terminal_on_bin = _bool_info(info, "is_obj_on_bin")
        terminal_static = _bool_info(info, "is_obj_static")
        mechanism = classify_terminal_mechanism(
            native_success=termination == "native_success", ever_released=released,
            release_on_bin=None if release_record is None else release_record.get("on_bin"),
            terminal_on_bin=terminal_on_bin, terminal_static=terminal_static,
            terminal_grasped=terminal_grasped, termination_reason=termination,
            elapsed_steps=elapsed, max_steps=max_steps)
        return {
            "scene_index": index, "scene_id": scene["metadata"]["scene_id"],
            "seed": seed, "status": "evaluated",
            "checkpoint_sha256": adapter.metadata["checkpoint_sha256"],
            "restored_bit_exact": True,
            "first_contact": first_contact,
            "crossing": crossing_record,
            "release": release_record,
            "terminal": terminal,
            "terminal_mechanism": mechanism,
            "result": {
                "termination_reason": termination, "elapsed_steps": elapsed,
                "terminated": terminated, "truncated": truncated,
                "native_success": termination == "native_success",
                "safe_success": bool(termination == "native_success" and not collision and not crossing),
                "obstacle_contact": collision, "xy_forbidden_crossing": crossing,
                "terminal_on_bin": terminal_on_bin, "terminal_static": terminal_static,
                "terminal_grasped": terminal_grasped, "terminal_ungrasped": not terminal_grasped,
                "linear_speed_m_s": terminal.get("linear_speed_m_s"),
                "angular_speed_rad_s": terminal.get("angular_speed_rad_s"),
                "ever_released": released,
                "phase": classify_phase(grasped, lifted, transported, released, termination == "native_success"),
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
        "examples/baselines/constrain/placesphere_learning_pilot.py",
        "examples/baselines/constrain/placesphere_transport_detour.py",
        "examples/baselines/constrain/placesphere_pilot.py",
        "examples/baselines/constrain/placesphere_base_dp.py",
        "examples/baselines/constrain/placesphere_constrain.py",
        "examples/baselines/constrain/constrain_denoiser.py",
        "examples/baselines/constrain/diagnose_placesphere_failure.py",
    )
    hashed = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in files}
    return {
        "basis": "working-tree bytes frozen at this command",
        "git_head_is_code_identity": False,
        "files": hashed,
        "files_sha256": hashlib.sha256(json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def _collection_spec(registration, registration_path):
    spec = dict(registration["collection"])
    addendum_path = Path(registration_path).parent / "collection_budget_addendum.json"
    if addendum_path.exists():
        addendum = json.loads(addendum_path.read_text())
        if addendum.get("reason") != "measured_yield_below_preregistered_attempt_budget":
            raise SystemExit("unrecognized collection budget addendum")
        if addendum.get("does_not_change") != ["counts", "dedup", "gate", "training"]:
            raise SystemExit("collection addendum changes a frozen rule")
        spec.update(addendum["collection"])
    return spec


def _smoke_action(adapter):
    action = coerce_action_sequence(adapter.get_action(np.zeros((2, RAW_OBS_DIM), np.float32)))
    native = PlaceSphereBaseDPAdapter.map_action_ee_delta_pose(action)
    if tuple(native.shape) != (1, 8, 7):
        raise RuntimeError(f"pilot action contract failed: {tuple(native.shape)}")
    return {"action_shape": list(native.shape), "finite": bool(torch.isfinite(native).all())}


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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--demo-dir", type=Path)
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--constrain-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--max-wall-seconds", type=float, default=2400)
    parser.add_argument("--per-attempt-deadline", type=float, default=160)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "preregister":
        exp = args.registration.parent
        stamp = exp.name.split("_placesphere")[0]
        payload = build_preregistration(
            args.round_index, args.version, {
                "exp": str(exp),
                "train_scenes": str(exp / "train_scenes"),
                "dev_scenes": str(exp / "dev_scenes"),
                "demos": f"demos/PlaceSphere-transport-detour/{stamp}_pilot_r{args.round_index}",
                "ckpt_base": f"outputs/ckpt/{stamp}_placesphere_pilot_base",
                "ckpt_constrain": f"outputs/ckpt/{stamp}_placesphere_pilot_constrain",
                "train_base": str(exp / "train_base"),
                "train_constrain": str(exp / "train_constrain"),
                "paired": str(exp / "paired"),
            }, reason=args.reason, prior_version=args.prior_version)
        payload["source_identity"] = freeze_pilot_identity()
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    registration = load_preregistration(args.registration)
    if args.command == "collect-train":
        spec = _collection_spec(registration, args.registration)
        if (args.seed != spec["train_seed"] or args.max_attempts != spec["train_max_attempts"]
                or args.max_wall_seconds != spec["train_wall_seconds"]
                or args.per_attempt_deadline != spec["per_attempt_deadline"]):
            raise SystemExit("collect-train budget does not match preregistration")
        exclude = layouts_from_manifest(PRIOR_MANIFEST)
        report = collect_split(
            args.output_dir, seed=spec["train_seed"], target=registration["train_obstacle_count"],
            max_attempts=spec["train_max_attempts"], max_wall_seconds=spec["train_wall_seconds"],
            per_attempt_deadline=spec["per_attempt_deadline"], exclude_layouts=exclude)
        print(json.dumps(report, indent=2), flush=True)
        if not report["complete"]:
            raise SystemExit("training collection incomplete")
        return report
    if args.command == "collect-dev":
        exclude = layouts_from_manifest(PRIOR_MANIFEST)
        for path in args.exclude_manifest:
            exclude.extend(layouts_from_manifest(path))
        spec = _collection_spec(registration, args.registration)
        if (args.seed != spec["dev_seed"] or args.max_attempts != spec["dev_max_attempts"]
                or args.max_wall_seconds != spec["dev_wall_seconds"]
                or args.per_attempt_deadline != spec["per_attempt_deadline"]):
            raise SystemExit("collect-dev budget does not match preregistration")
        report = collect_split(
            args.output_dir, seed=spec["dev_seed"], target=registration["dev_scene_count"],
            max_attempts=spec["dev_max_attempts"], max_wall_seconds=spec["dev_wall_seconds"],
            per_attempt_deadline=spec["per_attempt_deadline"], exclude_layouts=exclude)
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
        import signal
        from examples.baselines.constrain.diagnose_placesphere_failure import (
            _child_pids, _terminate_new_children,
        )
        from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
        catalog = TransportDetourSceneSet(args.scene_dir)
        if len(catalog) != registration["dev_scene_count"]:
            raise ValueError("dev scene count does not match the frozen denominator")
        scenes = catalog.manifest["scenes"]
        if not dev_premises_hold(scenes):
            raise ValueError("dev scenes do not meet the Ticket 04 premises")
        train_layouts = layouts_from_manifest(args.exclude_manifest[0]) if args.exclude_manifest else []
        dev_layouts = layouts_from_manifest(args.scene_dir / "manifest.json")
        prior = layouts_from_manifest(PRIOR_MANIFEST)
        if train_layouts and not new_splits_isolated(train_layouts, dev_layouts, prior, cylinder=False):
            raise ValueError("dev scenes are not independent of train or prior source")
        adapters = {
            "base_dp": PilotBaseAdapter(args.base_checkpoint, registration, args.device),
            "constrain_dp": PilotConstrainAdapter(args.constrain_checkpoint, registration, args.device),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        smoke = {name: _smoke_action(adapter) for name, adapter in adapters.items()}
        _json_atomic(args.output_dir / "action_contract.json", smoke)
        summaries, stored = {}, {}
        episode_wall = registration["eval"]["episode_wall_seconds"]
        max_steps = registration["eval"]["max_steps"]
        for name, adapter in adapters.items():
            rows = []
            policy_dir = args.output_dir / name
            policy_dir.mkdir()
            for index in range(len(catalog)):
                before = set(_child_pids())
                previous = signal.signal(signal.SIGALRM, _episode_deadline)
                signal.setitimer(signal.ITIMER_REAL, episode_wall)
                try:
                    row = run_pilot_episode(adapter, catalog, index, args.device, max_steps=max_steps)
                except Exception as exc:
                    row = {"scene_index": index, "status": "runtime_error",
                           "checkpoint_sha256": adapter.metadata["checkpoint_sha256"],
                           "error_kind": "wall_timeout" if isinstance(exc, EpisodeDeadline) else "unexpected",
                           "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                    _terminate_new_children(before)
                _json_atomic(policy_dir / f"episode_{index:03d}.json", row)
                rows.append(row)
                print(f"{name} scene {index} {row.get('status')}", flush=True)
            summaries[name] = summarize_pilot(rows, len(catalog))
            stored[name] = rows
            _json_atomic(policy_dir / "summary.json", summaries[name])
        evaluation_source = freeze_pilot_identity()
        registered = registration.get("source_identity") or {}
        paired = {
            "policies": summaries,
            "paired_counts": {
                "task_success": paired_counts(stored["base_dp"], stored["constrain_dp"], "native_success"),
                "safe_success": paired_counts(stored["base_dp"], stored["constrain_dp"], "safe_success"),
            },
            "ticket02_comparison": {
                "base_dp": compare_ticket02(stored["base_dp"]),
                "constrain_dp": compare_ticket02(stored["constrain_dp"]),
            },
            "checkpoint_sha256": {
                "base_dp": adapters["base_dp"].metadata["checkpoint_sha256"],
                "constrain_dp": adapters["constrain_dp"].metadata["checkpoint_sha256"],
            },
            "evaluation_source": evaluation_source,
            "preregistration_source_files_sha256": registered.get("files_sha256"),
            "evaluation_source_matches_preregistration": evaluation_source.get("files_sha256") == registered.get("files_sha256"),
            "formal_performance_claim": False,
        }
        _json_atomic(args.output_dir / "paired_summary.json", paired)
        print(json.dumps(summaries, indent=2), flush=True)
        return paired
    if args.command == "gate":
        paired = json.loads((args.output_dir / "paired_summary.json").read_text())
        decision = decide_gate(paired["policies"]["constrain_dp"])
        decision["ticket02_comparison"] = paired.get("ticket02_comparison")
        decision["paired_counts"] = paired.get("paired_counts")
        decision["evaluation_source_matches_preregistration"] = paired.get("evaluation_source_matches_preregistration")
        decision["checkpoint_sha256"] = paired.get("checkpoint_sha256")
        _json_atomic(args.output_dir / "gate.json", decision)
        print(json.dumps(decision, indent=2), flush=True)
        return decision
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
