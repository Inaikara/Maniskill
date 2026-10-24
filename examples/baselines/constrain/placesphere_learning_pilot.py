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


EXPANSION_DIRECT_COUNT = 1000
EXPANSION_OBSTACLE_COUNT = 100
RETAINED_COUNT = 20
BASELINE_EXP = "outputs/exp/20261024_014459_placesphere_learning_pilot"
RETAINED_SCENES = f"{BASELINE_EXP}/train_scenes"
RETAINED_DEMOS = "demos/PlaceSphere-transport-detour/20261024_014459_pilot_r1"
FROZEN_DEV_SCENES = f"{BASELINE_EXP}/dev_scenes"
SUCCESS_FIELDS = frozenset({"native_success", "safe_success"})
VIOLATION_FIELDS = frozenset({"obstacle_contact", "xy_forbidden_crossing"})
PAIR_ACCEPTANCE_CHECKS = (
    "direct_native_success", "direct_on_bin", "direct_static", "direct_ungrasped",
    "direct_release_retreat_settle", "direct_transport_blocked",
    "obstacle_native_success", "obstacle_on_bin", "obstacle_static", "obstacle_ungrasped",
    "obstacle_release_retreat_settle", "no_obstacle_contact", "no_obstacle_crossing",
    "endpoint_clearance", "radius_0_03", "block_in_transport", "action_observation_contract",
)
ADDENDUM_DOES_NOT_CHANGE = [
    "counts", "task_identity_definition", "gate", "training", "geometry",
    "expert", "external_excludes", "acceptance_predicate", "within_train_duplicate_policy",
]
OBSTACLE_EXECUTION_REASON = (
    "Pilot train 20/46 failures included direct_block_contact_in_endpoint, "
    "side_detour_occupies_cylinder_column, and direct_template_unsuccessful. "
    "Dev also recorded direct_container_phase_unsuccessful. Skipping block, safe, "
    "or endpoint acceptance would admit start/container pairs that fail those checks "
    "and change the distribution relative to the 100 pairs. Obstacle execution stays "
    "the acceptance predicate and is not packaged for the additional 900 direct trajectories."
)
DUPLICATE_POLICY_REASON = (
    "PickCube within-run strict uniqueness made 1000 direct infeasible under the 2 cm "
    "task-identity rule; the authorized addendum before that 900-run allowed train-layout "
    "repeats. PlaceSphere uses the same rule on a similar start/container box, so 1000 "
    "unique is not assumed. The 100 pairs stay unique. The 900 direct may repeat train "
    "task layouts. This is declared before any PlaceSphere expansion attempt."
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


def task_duplicate_report(rows):
    kept, rejected = partition_task_independent(rows)
    return {
        "packaged_count": len(rows),
        "greedy_unique": len(kept),
        "conflicts": len(rejected),
        "packaged_count_is_unique_count": len(rejected) == 0,
    }


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
        labels = (("cleared", "introduced", "both", "neither") if field in VIOLATION_FIELDS
                  else ("improved", "regressed", "both", "neither"))
        buckets = {key: 0 for key in labels}
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
    if recovery.get("generator_files_sha256") != (registration.get("generator_identity") or {}).get("files_sha256"):
        raise RuntimeError("recovery changed the generator")
    if recovery.get("recorded_before_rerun") is not True or "recorded_at" not in recovery:
        raise RuntimeError("recovery was not dated before rerun")
    return {
        "matched": "recovery_identity",
        "files_sha256": current["files_sha256"],
        "recorded_at": recovery["recorded_at"],
    }


def assert_external_independence(train, dev, prior):
    if any(layouts_conflict(left, right, cylinder=False) for left in train for right in dev):
        raise ValueError("train conflicts with frozen dev")
    if any(layouts_conflict(left, right, cylinder=False) for left in train for right in prior):
        raise ValueError("train conflicts with prior")
    if any(layouts_conflict(left, right, cylinder=False) for left in dev for right in prior):
        raise ValueError("dev conflicts with prior")
    return True


def load_expansion_addendum(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("reason") != "measured_yield_below_preregistered_attempt_budget":
        raise ValueError("unrecognized collection addendum")
    if payload.get("does_not_change") != ADDENDUM_DOES_NOT_CHANGE:
        raise ValueError("collection addendum changes a frozen rule")
    if payload.get("recorded_before_rerun") is not True or "recorded_at" not in payload:
        raise ValueError("addendum was not dated before rerun")
    if payload.get("scene_contract_unchanged") is not True or payload.get("formal_performance_claim") is not False:
        raise ValueError("addendum must keep the scene contract")
    direct = payload["direct_rule"]
    if direct.get("acceptance_predicate") != "full_run_attempt":
        raise ValueError("addendum changes the acceptance predicate")
    if (direct.get("obstacle_execution") != "retained_as_acceptance_predicate"
            or direct.get("obstacle_packaged") is not False):
        raise ValueError("addendum changes obstacle acceptance or packaging")
    if int(direct.get("target")) != 900 or direct.get("external_excludes") != ["ticket04", "dev30", "retained20"]:
        raise ValueError("addendum changes counts or external excludes")
    if direct.get("within_run_mutual_exclusion") is not False or direct.get("exclude_new_pairs") is not False:
        raise ValueError("addendum does not match the recorded direct rule")
    if int(direct.get("workers")) != 4 or int(direct.get("per_attempt_deadline")) != 160:
        raise ValueError("addendum changes the frozen worker or deadline contract")
    return payload


def _seed_span(spec):
    spans = [(int(block["seed"]) + 1, int(block["seed"]) + int(block["max_attempts"])) for block in spec["blocks"]]
    continuation = spec["continuation"]
    spans.append((int(continuation["seed"]) + 1, int(continuation["seed"]) + int(continuation["max_attempts"])))
    return spans


def build_expansion_preregistration(version, paths, *, reason, prior_version, baseline_files=None):
    if not reason or not prior_version:
        raise ValueError("expansion requires the recorded authorization and the prior version")
    if int(EXPANSION_OBSTACLE_COUNT) > OBSTACLE_PLAN_CAP or int(EXPANSION_OBSTACLE_COUNT) == 300:
        raise ValueError("expansion obstacle count exceeds the 100-plan")
    pairs = {
        "workers": 2, "target": 80, "max_attempts": 480, "wall_seconds": 12000,
        "per_attempt_deadline": 160, "within_run_mutual_exclusion": True, "drop_conflicts": True,
        "blocks": [
            {"name": "block_a", "seed": 2026128001, "target": 40, "max_attempts": 180, "wall_seconds": 5000},
            {"name": "block_b", "seed": 2026128401, "target": 40, "max_attempts": 180, "wall_seconds": 5000},
        ],
        "continuation": {"name": "continuation", "seed": 2026128801, "max_attempts": 120, "wall_seconds": 4000},
    }
    direct = {
        "workers": 4, "target": 900, "max_attempts": 3200, "wall_seconds": 24000,
        "per_attempt_deadline": 160, "within_run_mutual_exclusion": False, "drop_conflicts": False,
        "blocks": [
            {"name": "block_a", "seed": 2026130001, "target": 225, "max_attempts": 700, "wall_seconds": 14000},
            {"name": "block_b", "seed": 2026131001, "target": 225, "max_attempts": 700, "wall_seconds": 14000},
            {"name": "block_c", "seed": 2026132001, "target": 225, "max_attempts": 700, "wall_seconds": 14000},
            {"name": "block_d", "seed": 2026133001, "target": 225, "max_attempts": 700, "wall_seconds": 14000},
        ],
        "continuation": {"name": "continuation", "seed": 2026134001, "max_attempts": 400, "wall_seconds": 8000},
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
        "native success is on-bin, static, and ungrasped; saved actions include release, retreat, and settle",
        "observation is 39 policy / 42 raw",
        "evaluation budget is 200 steps",
        "counts are 1000 direct and 100 obstacle; the direct count is not a unique-layout count",
        "the 100 pairs are task-unique; the 900 direct may repeat train task layouts",
        "development scenes are the already-used 30, not a held-out formal 100",
        "result is an overall matched data-expansion effect on that development set, not a formal or generalization claim",
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
        "native_success": NATIVE_SUCCESS,
        "sweeps": "none",
        "layout_dedup": {
            "tolerance_m": LAYOUT_TOLERANCE_M,
            "metric": "chebyshev",
            "mode": "task",
            "cylinder_proximity_is_conflict": False,
            "within_run_enforced_for_pairs": True,
            "within_run_enforced_for_additional_direct": False,
            "conflict_if": "same seed, or start and container both within tolerance",
            "splits": "new train externally independent of frozen dev30, retained original20, and Ticket 04",
            "exclude_prior_source": PRIOR_MANIFEST,
            "exclude_dev": FROZEN_DEV_SCENES,
            "exclude_retained": RETAINED_SCENES,
        },
        "within_train_duplicates": {
            "pairs_within_run_mutual_exclusion": True,
            "direct_within_run_mutual_exclusion": False,
            "direct_may_repeat_train_task_layouts": True,
            "packaged_count_is_unique_count": False,
            "log": "greedy_unique_and_conflicts",
            "external_excludes": ["ticket04", "dev30", "retained20"],
            "exclude_new_pairs": False,
            "why_predeclared": DUPLICATE_POLICY_REASON,
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
            "additional_direct_sampling": "same run_attempt start/container sampler and full acceptance predicate",
            "additional_direct_obstacle_execution": "retained_as_acceptance_predicate",
            "additional_direct_obstacle_packaged": False,
            "obstacle_execution_not_skipped_because": OBSTACLE_EXECUTION_REASON,
            "generator": "examples.baselines.constrain.placesphere_transport_detour.run_attempt",
            "geometry_and_expert_unchanged": True,
            "parallel_workers_are_not_a_distribution_change": True,
        },
        "collection": {
            "dev_collected": False,
            "dev_scenes": FROZEN_DEV_SCENES,
            "yield_basis": {
                "source": f"{BASELINE_EXP}/train_scenes/report.json",
                "accepted": 20, "attempts": 46, "elapsed_seconds": 295.44346061500255,
                "acceptance_rate": 20 / 46,
            },
            "pairs": pairs,
            "direct": direct,
        },
        "training": training,
        "eval": {
            "max_steps": MAX_STEPS,
            "episode_wall_seconds": 120,
            "denominator": DEV_SCENE_COUNT,
            "noise_seed_rule": "torch.manual_seed and cuda.manual_seed_all(scene seed) before each scene, independently per policy",
            "action_contract": "coerce to (batch, horizon, 7); reject other widths; map pose to [-0.35, 0.35] and gripper to [-1, 1]",
            "scene_source": FROZEN_DEV_SCENES,
            "smoke_before_episodes": [1, 8, 7],
        },
        "gate": {
            "task_success_rate_gte": 0.90,
            "safe_success_rate_gte": 0.85,
            "contact_rate_lte": 0.10,
            "xy_crossing_rate_lte": 0.10,
            "require_complete": True,
            "require_zero_runtime_errors": True,
            "formal_performance_claim": False,
            "generalization_claim": False,
            "base_failure_threshold": None,
            "benefit_threshold": None,
            "on_fail": "authorized expansion is the only extra round; pause Ticket 08; no further round",
        },
        "baseline": {
            "exp": BASELINE_EXP,
            "authoritative_base": "base_dp",
            "failed_base_attempt": None,
            "authoritative_constrain": "constrain_dp",
            "base_counts": {"task": 0, "safe": 0, "contact": 29, "xy": 23},
            "constrain_counts": {"task": 16, "safe": 11, "contact": 8, "xy": 3},
            "denominator": 30,
            "files": dict(baseline_files or {}),
        },
        "timeouts_s": {
            "collect_pairs": 14000, "collect_direct": 26000, "train": 2400, "paired": 8000, "grace_kill": 60,
        },
        "attribution": "overall matched data expansion; not an isolated direct or obstacle contribution",
    }


def assert_retained_arrays(original_h5, records):
    original_h5 = Path(original_h5)
    before = sha256_file(original_h5)
    with h5py.File(original_h5, "r") as handle:
        for record in records:
            if record.get("provenance", {}).get("source") != "retained_r1":
                continue
            group = handle[f"traj_{int(record['episode_id'])}"]
            if not np.array_equal(group["obs"], np.asarray(record["obs"])) or not np.array_equal(
                    group["actions"], np.asarray(record["actions"])):
                raise ValueError(f"retained traj_{record['episode_id']} does not match the original demo")
    if sha256_file(original_h5) != before:
        raise RuntimeError("original demo file mutated")
    return before


def load_episode_directory(directory):
    directory = Path(directory)
    rows = [json.loads(path.read_text()) for path in sorted(directory.glob("episode_*.json"))]
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
        raise ValueError("baseline denominator is not the frozen evaluated set")
    actual = {key: counts[key] for key in ("task", "safe", "contact", "xy")}
    if actual != expected:
        raise ValueError(f"baseline counts {actual} do not match frozen {expected}")
    return counts


def _kill_descendants():
    import signal
    my_pid = os.getpid()
    children = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = int((entry / "stat").read_text().split()[3])
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
    expected_ids = [f"placesphere-{index:03d}" for index in range(RETAINED_COUNT)]
    if [row["scene_id"] for row in manifest["scenes"]] != expected_ids:
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


def collect_expansion_split(output_dir, *, seed, target, max_attempts, max_wall_seconds,
                            per_attempt_deadline, exclude_layouts, within_run):
    from examples.baselines.constrain.placesphere_transport_detour import generate

    external = [layout_view(row) for row in exclude_layouts]
    seen = list(external)

    def reject(metadata):
        current = layout_view(metadata)
        pool = seen if within_run else external
        for other in pool:
            if layouts_conflict(current, other, cylinder=False):
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


def _block_report(dest):
    path = Path(dest) / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    report["resumed"] = True
    report["output_dir"] = str(dest)
    return report


def _run_collection_block(block, output_root, exclude_layouts, deadline, within_run):
    dest = Path(output_root) / block["name"]
    existing = _block_report(dest)
    if existing is not None:
        print(f"resume {dest} accepted={existing.get('accepted')}", flush=True)
        return existing
    if dest.exists():
        raise RuntimeError(f"{dest} exists without a report; refusing to rerun the same seeds")
    report = collect_expansion_split(
        dest, seed=block["seed"], target=block["target"], max_attempts=block["max_attempts"],
        max_wall_seconds=block["wall_seconds"], per_attempt_deadline=deadline,
        exclude_layouts=exclude_layouts, within_run=within_run)
    report["output_dir"] = str(dest)
    report["resumed"] = False
    print(f"block {block['name']} accepted={report.get('accepted')} attempts={report.get('attempts')}", flush=True)
    return report


def assemble_catalogs(sources, dest, seed, provenance, drop_conflicts=True, identity_source=None):
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    temporary = dest.with_name(dest.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"{temporary} exists; refusing to hide a partial assemble")
    temporary.mkdir(parents=True)
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
    selected = [(h5_path, old_id, row) for h5_path, old_id, row in located if id(row) in kept_ids]
    rejected_rows = [row for row in rows if id(row) not in kept_ids]
    if len(selected) != len(kept) or len(rejected_rows) != len(rejected):
        raise RuntimeError("assemble partition lost a scene")
    with h5py.File(temporary / "scenes.h5", "w") as out:
        for index, (h5_path, old_id, row) in enumerate(selected):
            scene_id = f"placesphere-{index:03d}"
            with h5py.File(h5_path, "r") as src:
                src.copy(old_id, out, scene_id)
            row["source_scene_id"] = old_id
            row["scene_id"] = scene_id
    identity_file = None
    if identity_source is not None and (Path(identity_source) / "source_identity.json").exists():
        identity_file = Path(identity_source) / "source_identity.json"
    else:
        identity_file = next(Path(source) / "source_identity.json" for source in sources
                             if (Path(source) / "source_identity.json").exists())
    (temporary / "source_identity.json").write_bytes(identity_file.read_bytes())
    duplicate_report = task_duplicate_report([row for _h5, _old, row in selected])
    (temporary / "config.json").write_text(json.dumps({
        "assembled_from": [str(source) for source in sources],
        "generation_seed": seed,
        "scene_count": len(selected),
        "identities": identities,
        "dedup": "task",
        "drop_conflicts": drop_conflicts,
        "duplicate_accounting": duplicate_report,
        "packaged_count_is_unique_count": False,
    }, indent=2, sort_keys=True) + "\n")
    from examples.baselines.constrain.placesphere_transport_detour import _write_manifest
    _write_manifest(temporary, [row for _h5, _old, row in selected], seed)
    (temporary / "rejected_layouts.json").write_text(json.dumps(_jsonable({
        "count": len(rejected_rows), "scenes": rejected_rows,
    }), indent=2, sort_keys=True) + "\n")
    temporary.rename(dest)
    return {
        "kept": len(selected), "rejected": len(rejected_rows),
        "duplicate_accounting": duplicate_report,
    }


def _stage_excludes(extra_manifests=()):
    rows = layouts_from_manifest(PRIOR_MANIFEST)
    rows.extend(layouts_from_manifest(Path(FROZEN_DEV_SCENES) / "manifest.json"))
    rows.extend(layouts_from_manifest(Path(RETAINED_SCENES) / "manifest.json"))
    for path in extra_manifests:
        if Path(path).exists():
            rows.extend(layouts_from_manifest(path))
    return rows


def collect_expansion_stage(registration, stage_name, output_root, extra_manifests=(), spec_override=None):
    spec = spec_override or registration["collection"][stage_name]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    exclude = _stage_excludes(extra_manifests)
    within_run = bool(spec["within_run_mutual_exclusion"])
    drop_conflicts = bool(spec["drop_conflicts"])
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
        sources = [Path(report["output_dir"]) for report in reports
                   if (Path(report["output_dir"]) / "manifest.json").exists()]
        assembled = output_root / "assembled"
        assemble_report = {"kept": 0, "rejected": 0}
        if sources and not assembled.exists():
            assemble_report = assemble_catalogs(
                sources, assembled, seed=spec["blocks"][0]["seed"], provenance={},
                drop_conflicts=drop_conflicts)
        elif assembled.exists():
            assemble_report = {
                "kept": len(layouts_from_manifest(assembled / "manifest.json")),
                "rejected": None, "resumed": True,
            }
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
                    continuation, output_root / "blocks", more_exclude,
                    spec["per_attempt_deadline"], within_run)
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
        catalog = assembled if assembled.exists() else Path()
        duplicate = None
        if catalog and (catalog / "manifest.json").exists():
            duplicate = task_duplicate_report(layouts_from_manifest(catalog / "manifest.json"))
        report = {
            "stage": stage_name, "target": spec["target"], "kept": kept,
            "complete": kept == int(spec["target"]) and audit_ok,
            "restore_audit_ok": audit_ok, "blocks": reports,
            "continuation": continuation_report, "assemble": assemble_report,
            "elapsed_seconds": time.monotonic() - started,
            "catalog": str(catalog if catalog and (catalog / "manifest.json").exists() else ""),
            "within_run_mutual_exclusion": within_run,
            "drop_conflicts": drop_conflicts,
            "duplicate_accounting": duplicate,
            "packaged_count_is_unique_count": False,
        }
        _json_atomic(output_root / "stage_report.json", report)
        return report
    finally:
        stop.set()
        _kill_descendants()


def scene_acceptance_checks(scene, group):
    expert = scene.get("expert") or {}
    direct_actions = np.asarray(group["direct_actions"])
    direct_obs = np.asarray(group["direct_policy_observations"])
    obstacle_actions = np.asarray(group["expert_actions"])
    raw = np.asarray(group["raw_observations"])
    direct_truncated = np.asarray(group["direct_truncated"])
    obstacle_truncated = np.asarray(group["truncated"])

    def _bounded(actions):
        return (
            actions.ndim == 2 and actions.shape[1] == 7 and len(actions) >= 1
            and np.isfinite(actions).all()
            and float(np.max(np.abs(actions))) <= 1.00001
            and float(np.max(np.abs(actions[:, :6]))) <= 0.350001
        )

    contract = (
        _bounded(direct_actions) and _bounded(obstacle_actions)
        and direct_obs.ndim == 2 and direct_obs.shape[1] == POLICY_OBS_DIM
        and direct_obs.shape[0] == len(direct_actions) + 1 and np.isfinite(direct_obs).all()
        and raw.ndim == 2 and raw.shape[1] == RAW_OBS_DIM
        and raw.shape[0] == len(obstacle_actions) + 1 and np.isfinite(raw).all()
        and not bool(direct_truncated.any()) and not bool(obstacle_truncated.any())
    )
    direct_phases = [name for name, attr in (
        ("retreat", "direct_retreat_reached"),
        ("release", "direct_release_reached"),
        ("settle", "direct_settle_reached"),
    ) if attr in group.attrs and bool(int(group.attrs[attr]))]
    obstacle_phases = [name for name, key in (
        ("retreat", "retreat_phase_reached"),
        ("release", "release_phase_reached"),
        ("settle", "settle_phase_reached"),
    ) if expert.get(key)]
    clearance = expert.get("endpoint_clearance") or {}
    radius = (scene.get("cylinder") or {}).get("radius")
    direct_flags = {
        "ungrasped": bool(int(group.attrs.get("direct_ungrasped", 0))),
        "static": bool(int(group.attrs.get("direct_static", 0))),
        "on_bin": bool(int(group.attrs.get("direct_on_bin", 0))),
    }
    return {
        "direct_native_success": bool(int(group.attrs.get("direct_native_success", 0))),
        "direct_on_bin": direct_flags["on_bin"],
        "direct_static": direct_flags["static"],
        "direct_ungrasped": direct_flags["ungrasped"],
        "direct_release_retreat_settle": trajectory_has_release_retreat_settle(
            direct_actions, direct_phases, **direct_flags),
        "direct_transport_blocked": expert.get("direct_transport_blocked") is True,
        "obstacle_native_success": expert.get("native_success") is True,
        "obstacle_on_bin": expert.get("on_bin") is True,
        "obstacle_static": expert.get("static") is True,
        "obstacle_ungrasped": expert.get("ungrasped") is True,
        "obstacle_release_retreat_settle": trajectory_has_release_retreat_settle(
            obstacle_actions, obstacle_phases,
            ungrasped=bool(expert.get("ungrasped")), static=bool(expert.get("static")),
            on_bin=bool(expert.get("on_bin"))),
        "no_obstacle_contact": expert.get("obstacle_collision") is False,
        "no_obstacle_crossing": expert.get("obstacle_crossing") is False,
        "endpoint_clearance": (
            clearance.get("method") == "full_robot_phase_vertices"
            and clearance.get("paths") == ["direct", "side"]
        ),
        "radius_0_03": radius is not None and abs(float(radius) - 0.03) <= 1e-12,
        "block_in_transport": (expert.get("direct_block_contact") or {}).get("phase") == "transport",
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
                missing = [name for name in PAIR_ACCEPTANCE_CHECKS if not checks.get(name)]
                raise ValueError(f"{scene['scene_id']} failed the full acceptance predicate: {missing}")
            layout = layout_view(scene)
            provenance = {
                "source": scene.get("provenance_source", "collected"),
                "source_scene_id": scene.get("source_scene_id", scene["scene_id"]),
                "seed": scene.get("seed"),
            }
            direct_id, obstacle_id = assign(index, scene)
            expert = scene["expert"]
            records.append({
                "kind": "direct", "episode_id": direct_id, "scene_id": scene["scene_id"],
                "obs": np.asarray(group["direct_policy_observations"]),
                "actions": np.asarray(group["direct_actions"]),
                "terminated": np.asarray(group["direct_terminated"]),
                "truncated": np.asarray(group["direct_truncated"]),
                "native_success": True, "on_bin": True, "static": True, "ungrasped": True,
                "retreat_reached": True, "release_reached": True, "settle_reached": True,
                "obstacle_collision": False, "obstacle_crossing": False,
                "layout": layout, "provenance": provenance,
            })
            if obstacle_id is not None:
                records.append({
                    "kind": "obstacle", "episode_id": obstacle_id, "scene_id": scene["scene_id"],
                    "obs": np.asarray(group["raw_observations"]),
                    "actions": np.asarray(group["expert_actions"]),
                    "terminated": np.asarray(group["terminated"]),
                    "truncated": np.asarray(group["truncated"]),
                    "native_success": bool(expert.get("native_success")),
                    "on_bin": bool(expert.get("on_bin")),
                    "static": bool(expert.get("static")),
                    "ungrasped": bool(expert.get("ungrasped")),
                    "retreat_reached": bool(expert.get("retreat_phase_reached")),
                    "release_reached": bool(expert.get("release_phase_reached")),
                    "settle_reached": bool(expert.get("settle_phase_reached")),
                    "obstacle_collision": bool(expert.get("obstacle_collision")),
                    "obstacle_crossing": bool(expert.get("obstacle_crossing")),
                    "layout": layout, "provenance": provenance,
                })
    return records


def expansion_train_records(pair_dir, direct_dir, registration):
    pair_manifest = json.loads((Path(pair_dir) / "manifest.json").read_text())
    direct_manifest = json.loads((Path(direct_dir) / "manifest.json").read_text())
    if len(pair_manifest["scenes"]) != 100 or len(direct_manifest["scenes"]) != 900:
        raise ValueError("expansion catalogs are not 100 pairs and 900 direct")
    retained_seeds = [scene["seed"] for scene in json.loads(
        Path(RETAINED_SCENES, "manifest.json").read_text())["scenes"]]
    if [scene["seed"] for scene in pair_manifest["scenes"][:20]] != retained_seeds:
        raise ValueError("pair catalog does not start with the original 20 seeds")
    pair_records = _records_from_catalog(pair_dir, lambda index, _scene: (index, 1000 + index))
    for record in pair_records:
        episode_id = int(record["episode_id"])
        record["provenance"]["source"] = "retained_r1" if episode_id < 20 or 1000 <= episode_id < 1020 else "new_pair"
    direct_records = _records_from_catalog(direct_dir, lambda index, _scene: (100 + index, None))
    for record in direct_records:
        record["provenance"]["source"] = "additional_direct"
    records = pair_records + direct_records
    validate_expansion_ids(
        [int(record["episode_id"]) for record in records if record["kind"] == "direct"],
        [int(record["episode_id"]) for record in records if record["kind"] == "obstacle"])
    return records


def assert_direct_duplicate_policy(layouts, dev, prior):
    if len(layouts) != EXPANSION_DIRECT_COUNT:
        raise ValueError("direct layout count is not 1000")
    if task_duplicate_report(layouts[:EXPANSION_OBSTACLE_COUNT])["conflicts"] != 0:
        raise ValueError("pair catalog is not unique")
    extra = layouts[EXPANSION_OBSTACLE_COUNT:]
    retained = layouts[:RETAINED_COUNT]
    if any(layouts_conflict(left, right, cylinder=False) for left in extra for right in retained):
        raise ValueError("additional direct repeats a retained layout")
    assert_external_independence(layouts, dev, prior)
    report = task_duplicate_report(layouts)
    report["policy_packaged_count_is_unique_count"] = False
    return report


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
            str(retained): "retained_r1", retained.name: "retained_r1",
            str(new_catalog): "new_pair", new_catalog.name: "new_pair",
        },
        identity_source=new_catalog)
    if report["kept"] != 100 or report["rejected"] != 0:
        raise RuntimeError(f"pair merge is not 100 independent scenes: {report}")
    return report


def _require_new_generator(registration, report):
    expected = (registration.get("generator_identity") or {}).get("files_sha256")
    if not expected:
        raise RuntimeError("preregistration has no generator identity")
    blocks = list(report.get("blocks") or [])
    continuation = report.get("continuation") or {}
    if continuation.get("output_dir"):
        blocks.append(continuation)
    for block in blocks:
        identity_path = Path(block["output_dir"]) / "source_identity.json"
        if not identity_path.exists():
            continue
        actual = json.loads(identity_path.read_text())["files_sha256"]
        if actual != expected:
            raise RuntimeError("collection generator identity does not match preregistration")


def stage_spec_from_addendum(registration, stage_name, addendum):
    if not addendum or addendum.get("stage") not in (None, stage_name):
        return None
    override = addendum.get(f"{stage_name}_spec")
    if not override:
        return None
    frozen = registration["collection"][stage_name]
    for key in ("target", "workers", "per_attempt_deadline", "within_run_mutual_exclusion", "drop_conflicts"):
        if override.get(key) != frozen.get(key):
            raise ValueError(f"addendum changes frozen {stage_name} field {key}")
    if override.get("acceptance_predicate", "full_run_attempt") != "full_run_attempt":
        raise ValueError("addendum changes the acceptance predicate")
    return override


def _positive_count(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _agreed_budget(config, registration):
    recorded = config.get("training")
    frozen = (registration or {}).get("training")
    if not isinstance(recorded, dict) or not isinstance(frozen, dict):
        raise ValueError("run config is missing the training budget")
    if any(key not in recorded or key not in frozen for key in ("total_iters", "batch_size")):
        raise ValueError("run config is missing the training budget")
    updates = _positive_count(frozen["total_iters"], "updates")
    batch_size = _positive_count(frozen["batch_size"], "batch_size")
    if (_positive_count(recorded["total_iters"], "updates") != updates
            or _positive_count(recorded["batch_size"], "batch_size") != batch_size):
        raise ValueError("run config training budget does not match the registration")
    return updates, batch_size


def _measured_constrain_windows(identity, registration, episodes):
    if identity.get("selected_class") != "obstacle":
        raise ValueError("constrain coverage selected_class is not obstacle")
    ids = identity.get("selected_episode_ids")
    if not isinstance(ids, list) or len(ids) != episodes or any(type(item) is not int for item in ids):
        raise ValueError("constrain episodes do not match selected_episode_ids")
    expected = (registration or {}).get("selected_obstacle_ids")
    if list(expected or []) != ids:
        raise ValueError("registration obstacle ids do not match the constrain run config")
    demo_dir = ((registration or {}).get("paths") or {}).get("demos")
    if not demo_dir:
        raise ValueError("registration is missing the demo path needed to count windows")
    h5_path = Path(demo_dir) / "trajectory.h5"
    manifest_path = Path(demo_dir) / "trajectory.json"
    if not h5_path.is_file() or not manifest_path.is_file():
        raise ValueError("constrain run config is missing windows and the demo files needed to count them")
    if sha256_file(h5_path) != identity.get("h5_sha256"):
        raise ValueError("demo h5 hash does not match the constrain run config")
    if sha256_file(manifest_path) != identity.get("manifest_sha256"):
        raise ValueError("demo manifest hash does not match the constrain run config")
    dataset = PilotTrajectoryDataset(h5_path, manifest_path, "obstacle", ids)
    windows = len(dataset)
    if windows <= 0:
        raise ValueError("windows must be a positive integer")
    if len(dataset.trajectory_ids) != episodes:
        raise ValueError("measured constrain episodes do not match the run config")
    return windows


def _coverage_counts(config, registration):
    identity = config.get("data_identity")
    nested = isinstance(identity, dict)
    if nested and ("windows" in config or "episodes" in config):
        raise ValueError("run config mixes top-level and nested coverage fields")
    if "windows" in config or "episodes" in config:
        if "windows" not in config or "episodes" not in config:
            raise ValueError("base run config is missing windows or episodes")
        return _positive_count(config["windows"], "windows"), _positive_count(config["episodes"], "episodes")
    if not nested or "episodes" not in identity:
        raise ValueError("constrain run config is missing windows")
    episodes = _positive_count(identity["episodes"], "episodes")
    if "windows" in identity:
        ids = identity.get("selected_episode_ids")
        if not isinstance(ids, list) or len(ids) != episodes:
            raise ValueError("constrain episodes do not match selected_episode_ids")
        return _positive_count(identity["windows"], "windows"), episodes
    return _measured_constrain_windows(identity, registration, episodes), episodes


def write_coverage(run_dir, registration):
    config = json.loads((Path(run_dir) / "run_config.json").read_text())
    windows, episodes = _coverage_counts(config, registration)
    updates, batch_size = _agreed_budget(config, registration)
    coverage = dataset_coverage(windows, episodes, updates, batch_size)
    coverage["formal_performance_claim"] = False
    coverage["generalization_claim"] = False
    _json_atomic(Path(run_dir) / "coverage.json", coverage)
    return coverage


def episode_dir_sha(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("episode_*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compare_expansion_eval(registration, expansion_dir):
    baseline = registration["baseline"]
    root = Path(baseline["exp"]) / "paired"
    files = baseline.get("files") or {}
    for key, dirname in (("baseline_base_episodes", baseline["authoritative_base"]),
                         ("baseline_constrain_episodes", baseline["authoritative_constrain"])):
        recorded = (files.get(key) or {}).get("sha256")
        if recorded and episode_dir_sha(root / dirname) != recorded:
            raise RuntimeError(f"{key} bytes do not match preregistration")
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
        "generalization_claim": False,
        "attribution": registration.get("attribution"),
        "failed_attempt": None,
        "baseline_counts": {"base_dp": metric_counts(base), "constrain_dp": metric_counts(constrain)},
        "expansion_counts": {name: metric_counts(rows) for name, rows in expansion_rows.items()},
        "transitions": transitions,
        "mechanisms": {
            "baseline_base": compare_ticket02(base),
            "baseline_constrain": compare_ticket02(constrain),
            "expansion_base": compare_ticket02(expansion_rows["base_dp"]),
            "expansion_constrain": compare_ticket02(expansion_rows["constrain_dp"]),
        },
        "within_train_duplicates": registration.get("within_train_duplicates"),
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
        "preregister", "preregister-expansion", "collect-train", "collect-dev", "package",
        "include-original", "collect-pairs", "collect-direct", "package-expansion",
        "train-base", "train-constrain", "evaluate", "gate", "compare-baseline",
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
    if args.command == "preregister-expansion":
        from examples.baselines.constrain.placesphere_transport_detour import freeze_source_identity
        exp = args.registration.parent
        stamp = exp.name.split("_placesphere")[0]
        files = {
            "retained_scenes_h5": f"{RETAINED_SCENES}/scenes.h5",
            "retained_manifest": f"{RETAINED_SCENES}/manifest.json",
            "retained_demos_h5": f"{RETAINED_DEMOS}/trajectory.h5",
            "dev_scenes_h5": f"{FROZEN_DEV_SCENES}/scenes.h5",
            "dev_manifest": f"{FROZEN_DEV_SCENES}/manifest.json",
            "baseline_base_summary": f"{BASELINE_EXP}/paired/base_dp/summary.json",
            "baseline_constrain_summary": f"{BASELINE_EXP}/paired/constrain_dp/summary.json",
        }
        hashed = {key: {"path": path, "sha256": sha256_file(path)} for key, path in files.items()}
        hashed["baseline_base_episodes"] = {
            "path": f"{BASELINE_EXP}/paired/base_dp",
            "sha256": episode_dir_sha(f"{BASELINE_EXP}/paired/base_dp"),
        }
        hashed["baseline_constrain_episodes"] = {
            "path": f"{BASELINE_EXP}/paired/constrain_dp",
            "sha256": episode_dir_sha(f"{BASELINE_EXP}/paired/constrain_dp"),
        }
        payload = build_expansion_preregistration(
            args.version, {
                "exp": str(exp),
                "retained_scenes": str(exp / "retained_scenes"),
                "pair_scenes": str(exp / "pair_scenes"),
                "direct_scenes": str(exp / "direct_scenes"),
                "demos": f"demos/PlaceSphere-transport-detour/{stamp}_expansion",
                "ckpt_base": f"outputs/ckpt/{stamp}_placesphere_expansion_base",
                "ckpt_constrain": f"outputs/ckpt/{stamp}_placesphere_expansion_constrain",
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
    if args.command == "include-original":
        expected = ((registration.get("baseline") or {}).get("files") or {}).get("retained_scenes_h5", {}).get("sha256")
        report = copy_retained_catalog(
            RETAINED_SCENES, registration["paths"]["retained_scenes"], expected_h5_sha=expected)
        print(json.dumps(report, indent=2), flush=True)
        return report
    if args.command == "collect-pairs":
        resolve_running_identity(registration, freeze_pilot_identity())
        if not (Path(registration["paths"]["retained_scenes"]) / "manifest.json").exists():
            raise SystemExit("include-original before collect-pairs")
        addendum_path = args.registration.parent / "collection_budget_addendum.json"
        addendum = load_expansion_addendum(addendum_path) if addendum_path.exists() else None
        report = collect_expansion_stage(
            registration, "pairs", registration["paths"]["pair_scenes"],
            spec_override=stage_spec_from_addendum(registration, "pairs", addendum))
        _require_new_generator(registration, report)
        if report["complete"]:
            report["merge"] = finalize_pair_catalog(registration, report)
        _json_atomic(Path(registration["paths"]["pair_scenes"]) / "stage_report.json", report)
        print(json.dumps({key: report.get(key) for key in ("stage", "target", "kept", "complete", "catalog")}), flush=True)
        if not report["complete"]:
            raise SystemExit("pair collection incomplete")
        return report
    if args.command == "collect-direct":
        resolve_running_identity(registration, freeze_pilot_identity())
        if not (Path(registration["paths"]["pair_scenes"]) / "merged" / "manifest.json").exists():
            raise SystemExit("direct collection requires the merged pair catalog")
        addendum_path = args.registration.parent / "collection_budget_addendum.json"
        addendum = load_expansion_addendum(addendum_path) if addendum_path.exists() else None
        report = collect_expansion_stage(
            registration, "direct", registration["paths"]["direct_scenes"],
            spec_override=stage_spec_from_addendum(registration, "direct", addendum))
        _require_new_generator(registration, report)
        report["running_identity"] = resolve_running_identity(registration, freeze_pilot_identity())
        _json_atomic(Path(registration["paths"]["direct_scenes"]) / "stage_report.json", report)
        print(json.dumps({
            key: report.get(key) for key in (
                "stage", "target", "kept", "complete", "catalog", "within_run_mutual_exclusion",
                "duplicate_accounting")
        }), flush=True)
        if not report["complete"]:
            raise SystemExit("direct collection incomplete")
        return report
    if args.command == "package-expansion":
        resolve_running_identity(registration, freeze_pilot_identity())
        pair_dir = Path(registration["paths"]["pair_scenes"]) / "merged"
        direct_report = json.loads((Path(registration["paths"]["direct_scenes"]) / "stage_report.json").read_text())
        records = expansion_train_records(pair_dir, direct_report["catalog"], registration)
        original_sha = assert_retained_arrays(f"{RETAINED_DEMOS}/trajectory.h5", records)
        if sha256_file(f"{RETAINED_SCENES}/scenes.h5") != registration["baseline"]["files"]["retained_scenes_h5"]["sha256"]:
            raise RuntimeError("retained source scenes mutated")
        layouts = [record["layout"] for record in records if record["kind"] == "direct"]
        dev = layouts_from_manifest(Path(FROZEN_DEV_SCENES) / "manifest.json")
        prior = layouts_from_manifest(PRIOR_MANIFEST)
        duplicate_report = assert_direct_duplicate_policy(layouts, dev, prior)
        dest = package_records(records, registration["paths"]["demos"], registration)
        manifest = json.loads((dest / "trajectory.json").read_text())
        manifest["expansion"] = {
            "original_demo_sha256": original_sha,
            "selection": registration["collection_selection"],
            "formal_performance_claim": False,
            "generalization_claim": False,
            "duplicate_accounting": duplicate_report,
            "packaged_direct_count": 1000,
            "packaged_count_is_unique_count": False,
        }
        (dest / "trajectory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _json_atomic(Path(registration["paths"]["exp"]) / "train_layouts.json", {"scenes": layouts})
        print(dest, flush=True)
        return dest
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
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
        summary = train_base(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.run_dir, args.checkpoint_dir, registration, args.device)
        if registration.get("kind") == "authorized_data_expansion":
            summary["coverage"] = write_coverage(args.run_dir, registration)
        return summary
    if args.command == "train-constrain":
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
        summary = train_constrain(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.base_checkpoint, args.run_dir, args.checkpoint_dir, registration, args.device)
        if registration.get("kind") == "authorized_data_expansion":
            summary["coverage"] = write_coverage(args.run_dir, registration)
        return summary
    if args.command == "evaluate":
        import signal
        from examples.baselines.constrain.diagnose_placesphere_failure import (
            _child_pids, _terminate_new_children,
        )
        from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
        expansion = registration.get("kind") == "authorized_data_expansion"
        running_identity = None
        if expansion:
            running_identity = resolve_running_identity(registration, freeze_pilot_identity())
            if Path(args.scene_dir).resolve() != Path(registration["paths"]["dev_scenes"]).resolve():
                raise SystemExit("expansion eval must use the frozen dev scenes")
            if not args.exclude_manifest:
                raise SystemExit("expansion eval requires packaged train layouts")
        catalog = TransportDetourSceneSet(args.scene_dir)
        if len(catalog) != registration["dev_scene_count"]:
            raise ValueError("dev scene count does not match the frozen denominator")
        scenes = catalog.manifest["scenes"]
        if not dev_premises_hold(scenes):
            raise ValueError("dev scenes do not meet the Ticket 04 premises")
        train_layouts = layouts_from_manifest(args.exclude_manifest[0]) if args.exclude_manifest else []
        dev_layouts = layouts_from_manifest(args.scene_dir / "manifest.json")
        prior = layouts_from_manifest(PRIOR_MANIFEST)
        if expansion:
            assert_external_independence(train_layouts, dev_layouts, prior)
        elif train_layouts and not new_splits_isolated(train_layouts, dev_layouts, prior, cylinder=False):
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
            checkpoint = args.base_checkpoint if name == "base_dp" else args.constrain_checkpoint
            index = build_episode_index(
                rows, adapter.metadata["checkpoint_sha256"], checkpoint=str(checkpoint))
            index["source_identity"] = freeze_pilot_identity()
            _json_atomic(policy_dir / "index.json", index)
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
            "running_identity": running_identity,
            "formal_performance_claim": False,
            "generalization_claim": False,
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
        if registration.get("kind") == "authorized_data_expansion":
            decision["on_fail"] = registration["gate"]["on_fail"]
            decision["attribution"] = registration.get("attribution")
            decision["formal_performance_claim"] = False
            decision["generalization_claim"] = False
        _json_atomic(args.output_dir / "gate.json", decision)
        print(json.dumps(decision, indent=2), flush=True)
        return decision
    if args.command == "compare-baseline":
        resolve_running_identity(registration, freeze_pilot_identity())
        report = compare_expansion_eval(registration, args.output_dir)
        _json_atomic(args.output_dir / "expansion_comparison.json", report)
        print(json.dumps({
            "expansion_counts": report["expansion_counts"],
            "transitions": report["transitions"],
            "formal_performance_claim": False,
        }, indent=2), flush=True)
        return report
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
