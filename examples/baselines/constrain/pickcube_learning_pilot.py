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
    NAMED_FIELDS, build_agent, contract, sha256_file,
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
FULLSCENE_BASE_FORMAT = "pickcube_fullscene_base_v2"
FULLSCENE_CONSTRAIN_FORMAT = "pickcube_fullscene_constrain_v2"
NATIVE500_KIND = "frozen_fullscene500_native"
NATIVE500_BASE_FORMAT = "pickcube_fullscene500_native_base_v1"
NATIVE500_CONSTRAIN_FORMAT = "pickcube_fullscene500_native_constrain_v1"
NATIVE500_H5 = "demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.h5"
NATIVE500_MANIFEST = "demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.json"
NATIVE500_H5_SHA256 = "0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f"
NATIVE500_MANIFEST_SHA256 = "6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486"
NATIVE500_VALIDATION_SHA256 = "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f"
NATIVE500_BASE = "outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt"
NATIVE500_BASE_SHA256 = "dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d"
NATIVE500_BASE_H5 = "demos/PickCube-transport-detour/20261024_104348_expansion/trajectory.h5"
NATIVE500_BASE_H5_SHA256 = "2af959e3d3ebeba8bb046db6d9583d300e1533fc638c7d36eb6393dd28024dff"
NATIVE500_BASE_MANIFEST_SHA256 = "26cb2b3f8c371315805cb6219e0a4314dce3461bac3aaa872d35ce2e5ca7fd45"
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


def build_fullscene500_preregistration(paths, identities, source_identity):
    """Freeze the matched 1000/500 encoded-action engineering trial."""
    from examples.baselines.constrain.pickcube_action_codec import CONTROLLER_RANGE, profile_sha256

    training = {
        "seed": 1, "base_updates": 30000, "constrain_updates": 150000,
        "total_iters": 30000, "batch_size": 1024, "lr": 1e-4, "warmup": 500,
        "ema_power": 0.75, "optimizer": "AdamW", "betas": [0.95, 0.999],
        "weight_decay": 1e-6, "lr_schedule": "cosine",
        "architecture": "PickCube ConditionalUnet1D + ConstrainDenoiser",
        "selected_weights": SELECTED_WEIGHTS, "base_class": "direct",
        "constrain_class": "obstacle", "constrain_init": "matched base final EMA",
        "trainable": "control branch only", "loss_weight": None,
        "training_diffusion": {"scheduler": "DDPMScheduler", "steps": 100,
                                "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon"},
        "inference": {"scheduler": "DDIMScheduler", "steps": 50, "eta": 0.0,
                      "prediction_horizon": 16, "execution_horizon": 2,
                      "executed_prediction_slice": [1, 9], "plans_per_call": 1},
    }
    return {
        "format": "pickcube_fullscene500_action_codec_preregistration_v1",
        "kind": "frozen_fullscene500_action_codec", "version": "r33",
        "round_index": 33, "round_cap": 33,
        "train_direct_count": 1000, "train_obstacle_count": 500,
        "selected_direct_ids": list(range(1000)),
        "selected_obstacle_ids": list(range(1000, 1500)),
        "obstacle_plan_cap": 500, "dev_scene_count": 30,
        "layout_dedup": {"mode": "frozen_fullscene500", "within_run_enforced": True,
                          "cylinder_proximity_is_conflict": False,
                          "dev_and_ticket03_task_identity_strict": True},
        "action_encoding": CONTROLLER_RANGE,
        "action_encoding_sha256": profile_sha256(CONTROLLER_RANGE),
        "training": training,
        "paths": dict(paths), "datasets": dict(identities["datasets"]),
        "baseline": dict(identities["baseline"]),
        "matched_base_checkpoint": dict(identities.get("matched_base_checkpoint") or {}),
        "matched_constrain_checkpoint": dict(identities.get("matched_constrain_checkpoint") or {}),
        "source_identity": source_identity, "validation": dict(identities["validation"]),
        "evaluation": {"scene_manifest": paths["dev_scenes"] + "/manifest.json",
                       "scene_count": 30, "max_steps": 500, "episode_wall_seconds": 300,
                       "same_initial_native_state_and_controller": True,
                       "noise_seed_rule": "torch.manual_seed(scene seed) before each policy episode",
                       "runtime_errors_remain_in_denominator": True},
        "compute_budget_seconds": {"base": 4500, "constrain": 9000, "paired_eval": 18000},
        "selection": "final EMA only; selected before training; no evaluation-based checkpoint selection",
        "claims": {"formal_performance": False, "generalization": False,
                   "engineering_pipeline_comparison": True,
                   "confounds": ["training data size", "action encoding", "inference DDIM stride-2"]},
        "gate": {"task_success_rate_gte": 0.90, "safe_success_rate_gte": 0.85,
                 "contact_rate_lte": 0.10, "xy_crossing_rate_lte": 0.10,
                 "formal_performance_claim": False, "require_complete": True,
                 "require_zero_runtime_errors": True},
    }


def build_fullscene500_native_preregistration(paths, identities, source_identity):
    """Freeze the one native-action control-only trial; no action codec is used."""
    return {
        "format": "pickcube_fullscene500_native_preregistration_v1",
        "kind": NATIVE500_KIND,
        "version": "r35-native-control",
        "round_index": 35,
        "round_cap": 35,
        "train_direct_count": 1000,
        "train_obstacle_count": 500,
        "selected_direct_ids": list(range(1000)),
        "selected_obstacle_ids": list(range(1000, 1500)),
        "obstacle_plan_cap": 500,
        "dev_scene_count": 30,
        "layout_dedup": {"mode": "frozen_fullscene500", "within_run_enforced": True,
                          "cylinder_proximity_is_conflict": False,
                          "dev_and_ticket03_task_identity_strict": True},
        "paths": dict(paths),
        "datasets": dict(identities["datasets"]),
        "baseline": dict(identities.get("baseline", {})),
        "validation": dict(identities["validation"]),
        "source_identity": source_identity,
        "base_checkpoint": {"path": NATIVE500_BASE, "sha256": NATIVE500_BASE_SHA256,
                            "format": BASE_FORMAT, "step": 30000,
                            "selected_weights": "ema", "selected_class": "direct",
                            "selected_episode_ids": list(range(1000)),
                            "action_encoding": None, "model_action_space": "native identity coordinates"},
        "base_training_data": {"h5_path": NATIVE500_BASE_H5, "h5_sha256": NATIVE500_BASE_H5_SHA256,
                               "manifest_sha256": NATIVE500_BASE_MANIFEST_SHA256},
        "matched_base_checkpoint": {"path": NATIVE500_BASE, "sha256": NATIVE500_BASE_SHA256},
        "action_encoding": None,
        "action_space": "native pd_ee_delta_pose coordinates; identity normalization; no codec",
        "training": {
            "seed": 1, "base_updates": 30000, "constrain_updates": 150000,
            "total_iters": 150000, "batch_size": 1024, "lr": 1e-4,
            "warmup": 500, "ema_power": 0.75, "optimizer": "AdamW",
            "betas": [0.95, 0.999], "weight_decay": 1e-6,
            "lr_schedule": "cosine", "loss_weight": None,
            "base_retrained": False, "trainable": "control branch only",
            "base_class": "direct", "constrain_class": "obstacle",
            "constrain_init": "matched base final EMA",
            "selected_weights": SELECTED_WEIGHTS,
            "training_diffusion": {"scheduler": "DDPMScheduler", "steps": 100,
                                    "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon"},
        },
        "inference_hypothesis": {
            "status": "explicit hypothesis; old sampler protocol is not recoverable from retained records",
            "sampler": "DDIMScheduler", "num_train_timesteps": 100,
            "inference_steps": 50, "eta": 0.0, "prediction_horizon": 16,
            "executed_prediction_slice": [1, 9], "execution_horizon": 8,
            "replan_stride": 8, "noise_seed_rule": "torch.manual_seed(scene seed) before each policy episode",
        },
        "evaluation": {"scene_count": 30, "max_steps": 500,
                       "episode_wall_seconds": 300, "same_initial_native_state_and_controller": True,
                       "runtime_errors_remain_in_denominator": True},
        "gate": {"task_success_count_gte": 27, "safe_success_count_gte": 26,
                 "contact_count_lte": 3, "xy_crossing_count_lte": 3,
                 "require_complete": True, "require_zero_runtime_errors": True,
                 "formal_performance_claim": False},
        "comparison_note": "combined dataset-size and update-budget comparison, not a single-factor causal estimate",
        "selection": "final EMA fixed before training; no evaluation-based checkpoint selection",
    }


def build_r36_archived_ddpm_preregistration(paths, identities, source_identity):
    """Freeze same-weight R36 inference using the recovered archived sampler."""
    payload = build_fullscene500_native_preregistration(paths, identities, source_identity)
    payload.update({"version": "r36-archived-ddpm-restoration", "round_index": 36,
                    "round_cap": 36})
    payload["inference_hypothesis"] = {
        "status": "archived protocol supported by exact preregistered source identity",
        "sampler": "DDPMScheduler", "num_train_timesteps": 100,
        "inference_steps": 100, "prediction_horizon": 16,
        "executed_prediction_slice": [1, 9], "execution_horizon": 8,
        "replan_stride": 8,
        "native_action_representation": "identity pd_ee_delta_pose; direct environment action; no codec/decode/clamp",
        "noise_seed_rule": "torch.manual_seed(scene seed) before each policy episode",
    }
    payload["comparison_note"] = (
        "same-weight restoration of archived native DDPM-100; compared with R35 DDIM-50, "
        "sampler and inference-step package both change; stride remains 8. Secondary old 90k/300 "
        "comparison still combines dataset size and training budget differences.")
    return payload
    return {
        "format": "pickcube_fullscene500_action_codec_preregistration_v1",
        "kind": "frozen_fullscene500_action_codec", "version": "r33",
        "round_index": 33, "round_cap": 33,
        "train_direct_count": 1000, "train_obstacle_count": 500,
        "selected_direct_ids": list(range(1000)),
        "selected_obstacle_ids": list(range(1000, 1500)),
        "obstacle_plan_cap": 500, "dev_scene_count": 30,
        "layout_dedup": {"mode": "frozen_fullscene500", "within_run_enforced": True,
                          "cylinder_proximity_is_conflict": False,
                          "dev_and_ticket03_task_identity_strict": True},
        "action_encoding": CONTROLLER_RANGE,
        "action_encoding_sha256": profile_sha256(CONTROLLER_RANGE),
        "training": training,
        "paths": dict(paths), "datasets": dict(identities["datasets"]),
        "baseline": dict(identities["baseline"]),
        "matched_base_checkpoint": dict(identities.get("matched_base_checkpoint") or {}),
        "matched_constrain_checkpoint": dict(identities.get("matched_constrain_checkpoint") or {}),
        "source_identity": source_identity,
        "validation": dict(identities["validation"]),
        "evaluation": {"scene_manifest": paths["dev_scenes"] + "/manifest.json",
                       "scene_count": 30, "max_steps": 500,
                       "episode_wall_seconds": 300,
                       "same_initial_native_state_and_controller": True,
                       "noise_seed_rule": "torch.manual_seed(scene seed) before each policy episode",
                       "runtime_errors_remain_in_denominator": True},
        "compute_budget_seconds": {"base": 4500, "constrain": 9000,
                                    "paired_eval": 18000},
        "selection": "final EMA only; selected before training; no evaluation-based selection",
        "claims": {"formal_performance": False, "generalization": False,
                   "engineering_pipeline_comparison": True,
                   "confounds": ["training data size", "action encoding", "inference DDIM stride-2"]},
        "gate": {"task_success_rate_gte": 0.90, "safe_success_rate_gte": 0.85,
                 "contact_rate_lte": 0.10, "xy_crossing_rate_lte": 0.10,
                 "formal_performance_claim": False, "require_complete": True,
                 "require_zero_runtime_errors": True},
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
    if (payload.get("kind") != "frozen_fullscene500_action_codec"
            and payload["train_obstacle_count"] > payload["obstacle_plan_cap"]):
        raise ValueError("obstacle subset exceeds the 100-plan")
    if payload["gate"].get("formal_performance_claim") is not False:
        raise ValueError("pilot gate must not claim formal performance")
    if payload["training"].get("selected_weights") != SELECTED_WEIGHTS:
        raise ValueError("selected weights are not the pre-registered final EMA")
    if payload.get("kind") == NATIVE500_KIND:
        if (payload.get("action_encoding") is not None
                or payload.get("training", {}).get("constrain_updates") != 150000
                or payload.get("inference_hypothesis", {}).get("sampler") not in ("DDIMScheduler", "DDPMScheduler")
                or payload.get("inference_hypothesis", {}).get("inference_steps") not in (50, 100)
                or payload.get("selected_direct_ids") != list(range(1000))
                or payload.get("selected_obstacle_ids") != list(range(1000, 1500))):
            raise ValueError("native full-scene500 preregistration contract mismatch")
    if payload.get("kind") == "frozen_fullscene500_action_codec":
        from examples.baselines.constrain.pickcube_action_codec import profile_sha256
        if (payload["train_direct_count"] != 1000 or payload["train_obstacle_count"] != 500
                or payload["selected_direct_ids"] != list(range(1000))
                or payload["selected_obstacle_ids"] != list(range(1000, 1500))
                or payload["training"].get("base_updates") != 30000
                or payload["training"].get("constrain_updates") != 150000
                or profile_sha256(payload.get("action_encoding")) != payload.get("action_encoding_sha256")):
            raise ValueError("full-scene500 preregistration contract mismatch")
    return payload


def _assert_registered_dataset(dataset, registration):
    if registration.get("kind") not in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
        return
    identity = registration["datasets"]
    if (dataset.data_path.resolve() != Path(identity["h5"]["path"]).resolve()
            or dataset.manifest_path.resolve() != Path(identity["manifest"]["path"]).resolve()):
        raise ValueError("full-scene500 training data paths differ from preregistration")
    if (dataset.h5_sha256 != identity["h5"]["sha256"]
            or dataset.manifest_sha256 != identity["manifest"]["sha256"]):
        raise ValueError("full-scene500 training data hashes differ from preregistration")
    expected = registration["selected_direct_ids"] if dataset.kind == "direct" else registration["selected_obstacle_ids"]
    if dataset.trajectory_ids != expected:
        raise ValueError("full-scene500 selected class IDs differ from preregistration")
    if registration.get("kind") == NATIVE500_KIND:
        expected_windows = 45840 if dataset.kind == "obstacle" else 83004
        if len(dataset) != expected_windows or not dataset.native_fullscene500:
            raise ValueError("native full-scene500 official window count/profile mismatch")
    for spec in registration["validation"].values():
        if isinstance(spec, dict) and "path" in spec and sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError("full-scene500 pinned validation/source identity changed")


def audit_fullscene_action_encoding(data_path, manifest_path):
    """Recompute native and encoded action ranges/RMS directly from the frozen HDF5."""
    from examples.baselines.constrain.pickcube_action_codec import CONTROLLER_RANGE, encode_native

    manifest = json.loads(Path(manifest_path).read_text())
    by_id = {int(row["episode_id"]): row for row in manifest["episodes"]}
    if sorted(by_id) != list(range(1500)) or len(manifest["episodes"]) != 1500:
        raise ValueError("frozen manifest must cover IDs 0..1499 exactly once")
    stats = {}
    with h5py.File(data_path, "r") as handle:
        if len(handle) != 1500:
            raise ValueError("frozen HDF5 must have exactly 1500 episodes")
        for kind, ids in (("direct", range(1000)), ("obstacle", range(1000, 1500))):
            chunks, rms, encoded_min, encoded_max, terminal_grip = [], [], [], [], []
            for episode_id in ids:
                row = by_id[episode_id]
                group = handle[f"traj_{episode_id}"]
                actions = np.asarray(group["actions"][:], dtype=np.float32)
                if (row.get("kind") != kind or group.attrs.get("kind") != kind
                        or actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all()):
                    raise ValueError(f"frozen action class/schema mismatch at episode {episode_id}")
                encoded = encode_native(torch.from_numpy(actions), CONTROLLER_RANGE).numpy()
                chunks.append(actions)
                rms.append(np.sqrt(np.mean(actions.astype(np.float64) ** 2, axis=0)))
                encoded_min.append(encoded.min(axis=0))
                encoded_max.append(encoded.max(axis=0))
                terminal_grip.append(float(actions[-1, 6]))
            native = np.concatenate(chunks, axis=0)
            stats[kind] = {
                "episodes": len(chunks), "native_min": native.min(axis=0).tolist(),
                "native_max": native.max(axis=0).tolist(),
                "episode_rms_quantiles_0_05_50_95_100": np.quantile(
                    np.stack(rms), [0, .05, .5, .95, 1], axis=0).tolist(),
                "encoded_min": np.min(encoded_min, axis=0).tolist(),
                "encoded_max": np.max(encoded_max, axis=0).tolist(),
                "terminal_gripper_values": sorted(set(terminal_grip)),
            }
    if stats["obstacle"]["terminal_gripper_values"] != [-1.0]:
        raise ValueError("frozen PickCube obstacle tails must remain CLOSED (-1)")
    return {"source": "recomputed all frozen HDF5 action arrays before training",
            "profile": CONTROLLER_RANGE, "classes": stats}


def _recompute_registered_source(registration):
    if registration.get("kind") not in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
        return None
    current = freeze_pilot_identity(fullscene=True)
    try:
        enforce_runtime_source_identity(registration, current, "post-training-source-audit")
    except RuntimeError as exc:
        raise RuntimeError("experiment source changed after preregistration/recovery freeze") from exc
    return current


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


def materialize_window(obs, actions, start, end):
    """Training window padding: repeat the first step before 0, hold gripper after the end."""
    obs = torch.as_tensor(obs, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    length = len(actions)
    obs_seq = obs[max(0, start):start + 2]
    act_seq = actions[max(0, start):end]
    if start < 0:
        obs_seq = torch.cat((obs_seq[:1].repeat(-start, 1), obs_seq))
        act_seq = torch.cat((act_seq[:1].repeat(-start, 1), act_seq))
    if end > length:
        pad = torch.cat((torch.zeros(6), act_seq[-1, 6:7]))
        act_seq = torch.cat((act_seq, pad.repeat(end - length, 1)))
    return obs_seq, act_seq


class PilotTrajectoryDataset(torch.utils.data.Dataset):
    """Strict windows for one predeclared class. Refuses extra trajectories."""

    def __init__(self, data_path, manifest_path, kind, expected_ids, action_profile=None,
                 native_fullscene500=False):
        from examples.baselines.constrain.pickcube_action_codec import encode_native

        self.kind = kind
        self.data_path, self.manifest_path = Path(data_path), Path(manifest_path)
        self.action_profile = action_profile
        self.native_fullscene500 = bool(native_fullscene500)
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
        if self.native_fullscene500:
            if (len(episodes) != 1500 or sorted(ids) != list(range(1500))
                    or len(set(ids)) != 1500 or action_profile is not None):
                raise ValueError("native full-scene500 requires the exact 1500-ID manifest and identity actions")
            for row in episodes:
                if (not row.get("native_success") or not row.get("held_at_success")
                        or (row["kind"] == "obstacle" and
                            (row.get("obstacle_collision") or row.get("obstacle_crossing")))):
                    raise ValueError(f"episode {row['episode_id']}: unsafe or non-held native success")
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
                if self.native_fullscene500 and (np.max(np.abs(actions[:, :6])) > 0.350001
                        or (kind == "obstacle" and float(actions[-1, 6]) != -1.0)):
                    raise ValueError(f"traj_{episode_id}: native action range or CLOSED terminal grip mismatch")
                if not bool(group["success"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"traj_{episode_id} lacks successful non-truncated evidence")
                self.observations.append(torch.as_tensor(obs, dtype=torch.float32))
                action_tensor = torch.as_tensor(actions, dtype=torch.float32)
                if action_profile is not None:
                    action_tensor = encode_native(action_tensor, action_profile)
                self.actions.append(action_tensor)
                for start, end in _window_slices(len(actions)):
                    self.slices.append((len(self.actions) - 1, start, end))
        self.trajectory_ids = list(self.expected_ids)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start, end = self.slices[index]
        obs, actions = self.observations[trajectory], self.actions[trajectory]
        obs_seq, act_seq = materialize_window(obs, actions, start, end)
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


def fullscene_disabled_equivalence(base_agent, denoiser, device="cpu", seed=3351):
    """Compare independent Base EMA and disabled-control paths under identical DDIM noise."""
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    reference = base_agent.noise_pred_net.to(device).eval()
    denoiser.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn((1, 16, 7), generator=generator).to(device)
    condition = torch.randn((1, 84), generator=generator).to(device)
    hint = torch.zeros((1, 16, 3), device=device)
    timestep = torch.tensor([37], device=device)
    with torch.no_grad():
        expected = reference(sample, timestep, global_cond=condition)
        observed = denoiser(sample, timestep, global_cond=condition, hint=hint, enabled=False)
    epsilon_difference = float((expected - observed).abs().max())
    expected_sample, observed_sample = sample.clone(), sample.clone()
    scheduler = DDIMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon")
    scheduler.set_timesteps(50)
    with torch.no_grad():
        for current_timestep in scheduler.timesteps:
            expected_epsilon = reference(expected_sample, current_timestep, global_cond=condition)
            observed_epsilon = denoiser(
                observed_sample, current_timestep, global_cond=condition, hint=hint, enabled=False)
            expected_sample = scheduler.step(
                expected_epsilon, current_timestep, expected_sample, eta=0.0).prev_sample
            observed_sample = scheduler.step(
                observed_epsilon, current_timestep, observed_sample, eta=0.0).prev_sample
    sample_difference = float((expected_sample - observed_sample).abs().max())
    return {"epsilon_max_abs_diff": epsilon_difference,
            "ddim50_max_abs_diff": sample_difference,
            "epsilon_matches": epsilon_difference <= 1e-6,
            "sampling_matches": sample_difference <= 1e-6,
            "noise_seed": seed, "inference_steps": 50, "eta": 0.0}


def _save_base(path, agent, ema, step, payload, checkpoint_format=BASE_FORMAT):
    import copy
    path = Path(path)
    model_state = {key: value.detach().cpu() for key, value in agent.state_dict().items()}
    ema_agent = copy.deepcopy(agent)
    ema.copy_to(ema_agent.parameters())
    ema_state = {key: value.detach().cpu() for key, value in ema_agent.state_dict().items()}
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "format": checkpoint_format, "contract": contract(), "weights": {"model": model_state, "ema": ema_state},
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
    profile = registration.get("action_encoding")
    dataset = PilotTrajectoryDataset(
        demo_path, manifest_path, "direct", registration["selected_direct_ids"], profile)
    _assert_registered_dataset(dataset, registration)
    if len(dataset) < training["batch_size"]:
        raise RuntimeError("direct windows are fewer than the pre-registered batch size")
    agent = build_agent(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(), lr=training["lr"], betas=tuple(training["betas"]), weight_decay=training["weight_decay"])
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    base_updates = expected_base_updates(registration)
    lr_sched = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=training["warmup"],
        num_training_steps=base_updates)
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
        "data_path": str(dataset.data_path), "manifest_path": str(dataset.manifest_path),
        "training": training, "registration_version": registration["version"],
        "action_encoding": profile,
        "action_encoding_sha256": registration.get("action_encoding_sha256"),
        "model_action_space": "encoded controller-range coordinates" if profile else "native identity coordinates",
        "actual_training": {"updates": base_updates, "selected_weights": "ema"},
    }
    checkpoint_format = FULLSCENE_BASE_FORMAT if profile is not None else BASE_FORMAT
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n")
    iterator = iter(loader)
    losses = []
    started = time.monotonic()
    loss_log = (run_dir / "loss.jsonl").open("w")
    for step in range(1, base_updates + 1):
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
        loss_log.write(json.dumps({"step": step, "loss": losses[-1]}) + "\n")
        if step % 1000 == 0 or step == base_updates:
            loss_log.flush()
        if step == 1 or step == base_updates or step % 1000 == 0:
            print(f"base step {step}/{base_updates} loss={losses[-1]:.6f}", flush=True)
    loss_log.close()
    payload["source_identity_after_training"] = _recompute_registered_source(registration)
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    final = checkpoint_dir / "final.pt"
    _save_base(final, agent, ema, base_updates, payload, checkpoint_format)
    summary = {
        "status": "complete", "actual_updates": base_updates,
        "selected_weights": "ema", "loss_first": losses[0], "loss_last": losses[-1],
        "all_losses_finite": bool(np.isfinite(losses).all()),
        "elapsed_seconds": time.monotonic() - started, "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
        "dataset_coverage": dataset_coverage(
            len(dataset), len(dataset.trajectory_ids), base_updates, training["batch_size"]),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def expected_base_updates(registration):
    training = registration["training"]
    if "base_updates" in training:
        return int(training["base_updates"])
    return int(training["total_iters"])


def expected_constrain_updates(registration):
    training = registration["training"]
    return int(training.get("constrain_updates", training["total_iters"]))


def _load_base_ema(checkpoint, registration):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = ck.get("payload") or {}
    native500 = registration.get("kind") == NATIVE500_KIND
    expected_format = FULLSCENE_BASE_FORMAT if registration.get("action_encoding") else BASE_FORMAT
    if (ck.get("format") != expected_format or ck.get("selected_weights") != "ema"
            or int(ck.get("step", -1)) != expected_base_updates(registration)
            or payload.get("selected_class") != "direct"
            or payload.get("selected_episode_ids") != list(registration["selected_direct_ids"])
            or (registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND) and (
                payload.get("action_encoding") != registration.get("action_encoding")
                or (not native500 and (
                    payload.get("action_encoding_sha256") != registration.get("action_encoding_sha256")
                    or payload.get("model_action_space") != "encoded controller-range coordinates"
                    or payload.get("actual_training") != {"updates": expected_base_updates(registration), "selected_weights": "ema"}))
                or (native500 and (payload.get("model_action_space") not in (None, "native identity coordinates")
                                   or payload.get("selected_class") != "direct"))))):
        raise ValueError("pilot base checkpoint is not the matched final EMA")
    matched_base = registration.get("matched_base_checkpoint") or {}
    if matched_base and (Path(checkpoint).resolve() != Path(matched_base["path"]).resolve()
                         or sha256_file(checkpoint) != matched_base["sha256"]):
        raise ValueError("Base EMA path/hash differs from the preregistered matched checkpoint")
    if registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
        identity = registration["datasets"]
        if native500:
            base_identity = registration["base_checkpoint"]
            if (Path(checkpoint).resolve() != Path(base_identity["path"]).resolve()
                    or sha256_file(checkpoint) != base_identity["sha256"]
                    or ck.get("format") != BASE_FORMAT or ck.get("step") != 30000
                    or ck.get("selected_weights") != "ema"
                    or payload.get("selected_episode_ids") != list(range(1000))
                    or payload.get("selected_class") != "direct"
                    or payload.get("action_encoding") is not None):
                raise ValueError("native control requires the exact original native 1000-direct Base EMA")
            base_data = registration["base_training_data"]
            if (payload.get("h5_sha256") != base_data["h5_sha256"]
                    or payload.get("manifest_sha256") != base_data["manifest_sha256"]
                    or sha256_file(base_data["h5_path"]) != base_data["h5_sha256"]):
                raise ValueError("native Base training data identity differs from its retained source")
            _verify_native_direct_subset(identity["h5"]["path"], base_data["h5_path"])
        elif (payload.get("h5_sha256") != identity["h5"]["sha256"]
              or payload.get("manifest_sha256") != identity["manifest"]["sha256"]
              or Path(payload.get("data_path", "")).resolve() != Path(identity["h5"]["path"]).resolve()
              or Path(payload.get("manifest_path", "")).resolve() != Path(identity["manifest"]["path"]).resolve()):
            raise ValueError("base checkpoint data hashes differ from full-scene500 registration")
    agent = build_agent("cpu")
    agent.load_state_dict(ck["weights"]["ema"], strict=True)
    return ck, agent


def _verify_native_direct_subset(frozen_h5, base_h5):
    """Prove each original direct observation/action array is byte-identical."""
    with h5py.File(frozen_h5, "r") as frozen, h5py.File(base_h5, "r") as base:
        for episode_id in range(1000):
            key = f"traj_{episode_id}"
            left, right = frozen[key], base[key]
            if left.attrs.get("kind") != "direct" or right.attrs.get("kind") != "direct":
                raise ValueError(f"{key}: native Base direct-class identity mismatch")
            for field in ("obs", "actions"):
                a, b = left[field][:], right[field][:]
                if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                    raise ValueError(f"{key}/{field}: frozen direct array differs from the Base training source")


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
    profile = registration.get("action_encoding")
    native500 = registration.get("kind") == NATIVE500_KIND
    dataset = PilotTrajectoryDataset(
        demo_path, manifest_path, "obstacle", registration["selected_obstacle_ids"], profile,
        native_fullscene500=registration.get("kind") == NATIVE500_KIND)
    _assert_registered_dataset(dataset, registration)
    if len(dataset) < training["batch_size"]:
        raise RuntimeError("obstacle windows are fewer than the pre-registered batch size")
    if profile is not None:
        random.seed(training["seed"]); np.random.seed(training["seed"]); torch.manual_seed(training["seed"])
    denoiser = build_constrain_denoiser(MODEL_CONFIG, device="cpu")
    denoiser.base_net.load_state_dict(base_agent.noise_pred_net.state_dict(), strict=True)
    denoiser.to(device)
    base_sha = _state_sha(denoiser.base_net.state_dict())
    equivalence_before = fullscene_disabled_equivalence(base_agent, denoiser, device) if profile is not None or native500 else None
    if equivalence_before is not None and not (
            equivalence_before["epsilon_matches"] and equivalence_before["sampling_matches"]):
        raise RuntimeError("new Base EMA and disabled Constrain path differ before training")
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
    constrain_updates = expected_constrain_updates(registration)
    lr_sched = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=training["warmup"],
        num_training_steps=constrain_updates)
    ema = EMAModel(parameters=params, power=training["ema_power"])
    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(RandomSampler(dataset), training["batch_size"], drop_last=True),
        num_workers=0)
    norm = PickCubeConditionNorm()
    checkpoint_format = (NATIVE500_CONSTRAIN_FORMAT if native500 else
                         FULLSCENE_CONSTRAIN_FORMAT if profile is not None else CONSTRAIN_FORMAT)
    config = {
        "model": dict(MODEL_CONFIG),
        "policy": {
            "robot": "panda", "obs_dim": 42, "obstacle_obs_dim": 45, "act_dim": 7,
            "obs_horizon": 2, "pred_horizon": 16, "act_horizon": 8,
            "control_mode": "pd_ee_delta_pose", "observation_mode": "state",
            "named_state_fields": NAMED_FIELDS,
        },
        "diffusion": {"training_scheduler": "DDPMScheduler", "inference_scheduler": "DDIMScheduler" if profile is not None or native500 else "DDPMScheduler",
                      "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
                      "prediction_type": "epsilon", "num_train_timesteps": 100,
                      "inference_steps": registration.get("inference_hypothesis", {}).get("inference_steps", 50)
                      if profile is not None or native500 else 100,
                      "eta": 0.0 if profile is not None or native500 else None},
        "normalization": norm.to_dict(),
        "base_weights": {
            "source": str(base_checkpoint), "sha256": sha256_file(base_checkpoint),
            "subkey": "weights.ema", "state_sha256": base_sha, "updates": expected_base_updates(registration),
        },
        "data_identity": {
            "selected_class": "obstacle", "episodes": len(dataset.trajectory_ids),
            "selected_episode_ids": dataset.trajectory_ids, "h5_sha256": dataset.h5_sha256,
            "manifest_sha256": dataset.manifest_sha256,
            "data_path": str(dataset.data_path), "manifest_path": str(dataset.manifest_path),
        },
        "training": dict(training, total_iters=constrain_updates),
        "selected_weights": SELECTED_WEIGHTS,
        "loss_weight": training.get("loss_weight"),
        "action_encoding": profile,
        "action_encoding_sha256": registration.get("action_encoding_sha256"),
        "model_action_space": "encoded controller-range coordinates" if profile else "native identity coordinates",
        "action_encoding_identity": True,
        "actual_training": {"updates": expected_constrain_updates(registration), "selected_weights": "ema"},
        "pretraining_disabled_control_equivalence": equivalence_before,
    }
    weight_spec = training.get("loss_weight")
    if weight_spec is not None and weight_spec.get("name") != "obstacle_proximity":
        raise ValueError("unsupported constrain loss weight")
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    iterator = iter(loader)
    losses = []
    unweighted_losses = []
    started = time.monotonic()
    loss_log_path = run_dir / "loss.jsonl"
    loss_log = loss_log_path.open("w")
    for step in range(1, constrain_updates + 1):
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
        loss_log.write(json.dumps({"step": step, "loss": losses[-1],
                                   "unweighted_loss": unweighted_losses[-1]}) + "\n")
        if step % 1000 == 0 or step == constrain_updates:
            loss_log.flush()
        if step == 1 or step == constrain_updates or step % 1000 == 0:
            print(
                f"constrain step {step}/{constrain_updates} loss={losses[-1]:.6f} "
                f"unweighted={unweighted_losses[-1]:.6f} mean_weight={mean_weight:.4f}",
                flush=True,
            )
    loss_log.close()
    config["source_identity_after_training"] = _recompute_registered_source(registration)
    _json_atomic(run_dir / "run_config.json", config)
    if _state_sha(denoiser.base_net.state_dict()) != base_sha:
        raise RuntimeError("frozen base changed during constrain training")
    if not any(not torch.equal(param, start) for param, start in zip(denoiser.trainable_parameters(), initial)):
        raise RuntimeError("control branch did not change")
    equivalence_after = fullscene_disabled_equivalence(base_agent, denoiser, device) if profile is not None or native500 else None
    if equivalence_after is not None and not (
            equivalence_after["epsilon_matches"] and equivalence_after["sampling_matches"]):
        raise RuntimeError("frozen base/disabled control equivalence failed after training")
    final = checkpoint_dir / "final.pt"
    save_constrain_checkpoint(
        final, denoiser, ema, optimizer, lr_sched, constrain_updates,
        {"torch": torch.get_rng_state()}, config, checkpoint_format=checkpoint_format)
    summary = {
        "status": "complete", "actual_updates": constrain_updates,
        "selected_weights": SELECTED_WEIGHTS, "base_unchanged": True,
        "control_parameter_changed": True, "loss_first": losses[0], "loss_last": losses[-1],
        "unweighted_loss_first": unweighted_losses[0], "unweighted_loss_last": unweighted_losses[-1],
        "loss_weight": weight_spec,
        "all_losses_finite": bool(np.isfinite(losses).all() and np.isfinite(unweighted_losses).all()),
        "elapsed_seconds": time.monotonic() - started, "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
        "dataset_coverage": dataset_coverage(
            len(dataset), len(dataset.trajectory_ids), constrain_updates, training["batch_size"]),
        "base_state_sha256": base_sha,
        "strict_ema_load": True,
        "disabled_control_equivalence_after_training": equivalence_after,
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


def _pickcube_ddim_scheduler(inference_steps=50):
    if not isinstance(inference_steps, int) or inference_steps not in (50, 100):
        raise ValueError("PickCube action-codec DDIM inference supports exactly 50 or 100 steps")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon")
    scheduler.set_timesteps(inference_steps)
    expected_first = 99 if inference_steps == 100 else 98
    if len(scheduler.timesteps) != inference_steps or int(scheduler.timesteps[0]) != expected_first:
        raise RuntimeError("DDIM scheduler did not construct the registered endpoint sequence")
    return scheduler


class PilotBaseAdapter:
    def __init__(self, checkpoint, registration, device="cpu", inference_steps=50):
        ck, self.agent = _load_base_ema(checkpoint, registration)
        self.agent.to(device).eval()
        self.device = torch.device(device)
        self.action_profile = registration.get("action_encoding")
        native500 = registration.get("kind") == NATIVE500_KIND
        protocol = registration.get("inference_hypothesis", {})
        action_slice = protocol.get("executed_prediction_slice", [1, 9])
        self.action_slice = slice(action_slice[0], action_slice[1])
        self.scheduler = None
        protocol = registration.get("inference_hypothesis", {})
        self.native_ddpm = native500 and protocol.get("sampler") == "DDPMScheduler"
        if self.action_profile is not None or (native500 and not self.native_ddpm):
            self.scheduler = _pickcube_ddim_scheduler(inference_steps)
        self.inference_steps = inference_steps if self.scheduler is not None else 100
        self.metadata = {"format": FULLSCENE_BASE_FORMAT if self.action_profile else BASE_FORMAT,
                         "selected_weights": "ema",
                         "checkpoint_sha256": sha256_file(checkpoint),
                         "inference_scheduler": "DDPMScheduler" if self.native_ddpm or not (self.action_profile or native500) else "DDIMScheduler",
                         "inference_steps": self.inference_steps,
                         "eta": 0.0 if self.action_profile or native500 else None,
                         "execution_horizon": registration.get("inference_hypothesis", {}).get("execution_horizon", 8)}

    def get_action(self, queued):
        obs = torch.as_tensor(queued, dtype=torch.float32, device=self.device)
        if obs.shape[-2:] != (2, 45):
            raise ValueError(f"expected (2,45) observations, got {tuple(obs.shape)}")
        if obs.ndim == 2:
            obs = obs[None]
        if self.scheduler is not None:
            sample = torch.randn((obs.shape[0], 16, 7), device=self.device)
            condition = obs[..., :42].flatten(start_dim=obs.ndim - 2)
            with torch.no_grad():
                for timestep in self.scheduler.timesteps:
                    prediction = self.agent.noise_pred_net(
                        sample, timestep, global_cond=condition)
                    sample = self.scheduler.step(prediction, timestep, sample, eta=0.0).prev_sample
            action = sample[:, self.action_slice]
            if self.action_profile is not None:
                from examples.baselines.constrain.pickcube_action_codec import decode_prediction_to_native
                return decode_prediction_to_native(action, self.action_profile)
            return action
        with torch.no_grad():
            return self.agent.get_action(obs[..., :42])


class PilotConstrainAdapter:
    def __init__(self, checkpoint, registration, device="cpu", inference_steps=50):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.training_utils import EMAModel
        from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
        from examples.baselines.constrain.pickcube_constrain import MODEL_CONFIG, PickCubeConditionNorm
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ck.get("config") or {}
        data = config.get("data_identity") or {}
        matched_control = registration.get("matched_constrain_checkpoint") or {}
        native500 = registration.get("kind") == NATIVE500_KIND
        expected_format = (NATIVE500_CONSTRAIN_FORMAT if native500 else
                           FULLSCENE_CONSTRAIN_FORMAT if registration.get("action_encoding") else CONSTRAIN_FORMAT)
        if (ck.get("format") != expected_format or int(ck.get("step", -1)) != expected_constrain_updates(registration)
                or config.get("selected_weights") != SELECTED_WEIGHTS
                or (registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND)
                    and (config.get("diffusion", {}).get("training_scheduler") != "DDPMScheduler"
                         or config.get("diffusion", {}).get("num_train_timesteps") != 100
                         or config.get("diffusion", {}).get("beta_schedule") != "squaredcos_cap_v2"
                         or config.get("diffusion", {}).get("prediction_type") != "epsilon"
                         or (not native500 and config.get("diffusion") != {
                        "training_scheduler": "DDPMScheduler", "inference_scheduler": "DDIMScheduler",
                        "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
                        "prediction_type": "epsilon", "num_train_timesteps": 100,
                        "inference_steps": registration.get("inference_hypothesis", {}).get("inference_steps", 50), "eta": 0.0})
                         or config.get("policy", {}).get("obs_dim") != 42
                          or config.get("policy", {}).get("act_dim") != 7
                          or (not native500 and config.get("model_action_space") != "encoded controller-range coordinates")
                          or (native500 and (config.get("model_action_space") != "native identity coordinates"
                                             or config.get("action_encoding_identity") is not True))))
                or data.get("selected_class") != "obstacle"
                or data.get("selected_episode_ids") != list(registration["selected_obstacle_ids"])
                or config.get("action_encoding") != registration.get("action_encoding")
                or config.get("action_encoding_sha256") != registration.get("action_encoding_sha256")
                 or (registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND) and (
                    (not native500 and config.get("model_action_space") != "encoded controller-range coordinates")
                    or (native500 and config.get("model_action_space") != "native identity coordinates")
                    or (native500 and config.get("action_encoding_identity") is not True)
                    or config.get("actual_training") != {"updates": expected_constrain_updates(registration), "selected_weights": "ema"}))
                or (matched_control and (
                    Path(checkpoint).resolve() != Path(matched_control["path"]).resolve()
                    or sha256_file(checkpoint) != matched_control["sha256"]))
        or (registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND)
            and (data.get("h5_sha256") != registration["datasets"]["h5"]["sha256"]
                 or data.get("manifest_sha256") != registration["datasets"]["manifest"]["sha256"]
                 or Path(data.get("data_path", "")).resolve() != Path(registration["datasets"]["h5"]["path"]).resolve()
                 or Path(data.get("manifest_path", "")).resolve() != Path(registration["datasets"]["manifest"]["path"]).resolve()
                 or config.get("base_weights", {}).get("sha256") != sha256_file(registration["paths"]["base_checkpoint"])) )):
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
        self.inference_scheduler = None
        steps = inference_steps
        protocol = registration.get("inference_hypothesis", {})
        native_ddpm = native500 and protocol.get("sampler") == "DDPMScheduler"
        if registration.get("action_encoding") or (native500 and not native_ddpm):
            steps = registration.get("inference_hypothesis", {}).get("inference_steps", inference_steps)
            self.inference_scheduler = _pickcube_ddim_scheduler(steps)
        else:
            self.scheduler.set_timesteps(steps if native_ddpm else 100)
        self.action_profile = registration.get("action_encoding")
        protocol = registration.get("inference_hypothesis", {})
        action_slice = protocol.get("executed_prediction_slice", [1, 9])
        self.action_slice = slice(action_slice[0], action_slice[1])
        self.inference_steps = steps if self.inference_scheduler is not None else 100
        self.metadata = {"format": expected_format, "selected_weights": SELECTED_WEIGHTS,
                         "checkpoint_sha256": sha256_file(checkpoint),
                         "inference_scheduler": "DDPMScheduler" if native_ddpm or not (self.action_profile is not None or native500) else "DDIMScheduler",
                         "inference_steps": self.inference_steps,
                         "eta": 0.0 if self.action_profile is not None or (native500 and not native_ddpm) else None,
                         "execution_horizon": registration.get("inference_hypothesis", {}).get("execution_horizon", 8)}

    def get_action(self, queued):
        from examples.baselines.constrain.pickcube_constrain import pickcube_relative_condition
        from examples.baselines.constrain.pickcube_action_codec import decode_prediction_to_native
        obs = torch.as_tensor(queued, dtype=torch.float32, device=self.device)
        if obs.ndim == 2:
            obs = obs[None]
        hint = pickcube_relative_condition(obs, self.norm)[:, None, :].expand(-1, 16, -1)
        sample = torch.randn(obs.shape[0], 16, 7, device=self.device)
        cond = obs[..., :42].flatten(start_dim=obs.ndim - 2)
        with torch.no_grad():
            scheduler = self.inference_scheduler or self.scheduler
            for timestep in scheduler.timesteps:
                prediction = self.denoiser(sample, timestep, global_cond=cond, hint=hint, enabled=True)
                if self.inference_scheduler is not None:
                    sample = scheduler.step(prediction, timestep, sample, eta=0.0).prev_sample
                else:
                    sample = scheduler.step(prediction, timestep, sample).prev_sample
        action = sample[:, self.action_slice]
        if self.action_profile is not None:
            return decode_prediction_to_native(action, self.action_profile)
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
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env, snapshot_execution_state
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
        initial_state, initial_controller = snapshot_execution_state(env)
        restored_state_sha = _tree_hash(initial_state)
        restored_controller_sha = _tree_hash(initial_controller)
        restored_observation_sha = hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()
        history = np.repeat(obs[None, :], 2, axis=0)
        torch.manual_seed(int(scene["metadata"]["seed"]))
        elapsed = 0
        collision = crossing = grasped = lifted = transported = False
        initial_cube_z = float(env.unwrapped.cube.pose.p[0, 2].detach().cpu())
        goal_xy = env.unwrapped.goal_site.pose.p[0, :2].detach().cpu().numpy()
        first_contact = None
        native_success = terminated = truncated = False
        termination = None
        first_grasp_step = first_lift_step = first_transport_step = first_success_step = None
        step_trace = []
        while elapsed < max_steps:
            prediction = adapter.get_action(history)
            execution_horizon = int(getattr(adapter, "metadata", {}).get(
                "execution_horizon", 2 if getattr(adapter, "action_profile", None) is not None else 8))
            for action in prediction[0][:min(execution_horizon, max_steps - elapsed)]:
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
                if grasped_now and first_grasp_step is None:
                    first_grasp_step = elapsed
                if lifted_now and first_lift_step is None:
                    first_lift_step = elapsed
                if near_goal and first_transport_step is None:
                    first_transport_step = elapsed
                if step_success and first_success_step is None:
                    first_success_step = elapsed
                step_trace.append({
                    "step": elapsed, "native_action": action_cpu.tolist(),
                    "grasped": grasped_now, "lifted": lifted_now, "near_goal": near_goal,
                    "native_success": step_success, "terminated": bool(step_terminated),
                    "truncated": bool(step_truncated), "obstacle_collision": step_collision,
                    "obstacle_crossing": _bool_info(info, "obstacle_crossing"),
                })
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
        final_cube_xyz = env.unwrapped.cube.pose.p[0].detach().cpu().numpy()
        final_goal_xyz = env.unwrapped.goal_site.pose.p[0].detach().cpu().numpy()
        final_goal_distance_3d = float(np.linalg.norm(final_cube_xyz - final_goal_xyz))
        return {
            "scene_index": index, "scene_id": scene["metadata"]["scene_id"],
            "seed": int(scene["metadata"]["seed"]), "status": "evaluated",
            "restored_state_sha256": restored_state_sha,
            "restored_controller_state_sha256": restored_controller_sha,
            "restored_observation_sha256": restored_observation_sha,
            "first_contact": first_contact,
            "step_trace": step_trace,
            "first_grasp_step": first_grasp_step, "first_lift_step": first_lift_step,
            "first_transport_step": first_transport_step, "first_native_success_step": first_success_step,
            "final_cube_xyz": final_cube_xyz.tolist(), "goal_xyz": final_goal_xyz.tolist(),
            "final_goal_distance_3d": final_goal_distance_3d,
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


def fullscene500_engineering_comparison(registration, output_dir, current_rows):
    """Compare final EMA outcomes to the preregistered retained 90k trial."""
    from examples.baselines.constrain.materialize_pickcube_fullscene500 import file_tree_hash

    paired_root = Path(output_dir) / "baseline_snapshot" / "paired"
    if file_tree_hash(paired_root) != registration["baseline"]["paired_episode_tree_sha256"]:
        raise ValueError("historical baseline snapshot changed after pinning")
    baselines = {
        "base_dp": load_episode_directory(paired_root / "base_dp"),
        "constrain_dp": load_episode_directory(paired_root / "constrain_dp"),
    }
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    transitions = {}
    pairing_failures = []
    new_base = sorted(current_rows["base_dp"], key=lambda row: int(row["scene_index"]))
    new_constrain = sorted(current_rows["constrain_dp"], key=lambda row: int(row["scene_index"]))
    for left, right in zip(new_base, new_constrain):
        if left.get("status") != "evaluated" or right.get("status") != "evaluated" or any(left.get(field) != right.get(field) for field in (
                "scene_id", "restored_state_sha256", "restored_controller_state_sha256",
                "restored_observation_sha256")):
            pairing_failures.append(int(left["scene_index"]))
    for policy, current in current_rows.items():
        old = baselines[policy]
        if len(old) != 30 or len(current) != 30:
            raise ValueError("primary engineering comparison requires all 30 paired scenes")
        old = sorted(old, key=lambda row: int(row["scene_index"]))
        current = sorted(current, key=lambda row: int(row["scene_index"]))
        for left, right in zip(old, current):
            if (left.get("status") != "evaluated" or right.get("status") != "evaluated"
                    or left.get("scene_id") != right.get("scene_id")
                    or left.get("restored_state_sha256") != right.get("restored_state_sha256")):
                pairing_failures.append(int(right["scene_index"]))
        if pairing_failures:
            transitions[policy] = None
        else:
            transitions[policy] = {}
            for name in fields:
                counts = paired_counts(old, current, name)
                if name in ("native_success", "safe_success"):
                    transitions[policy][name] = {
                        "improved_new_only": counts["constrain_only"],
                        "regressed_baseline_only": counts["base_only"],
                        "both": counts["both"], "neither": counts["neither"],
                    }
                else:
                    transitions[policy][name] = {
                        "cleared_new_only": counts["base_only"],
                        "introduced_new_only": counts["constrain_only"],
                        "both": counts["both"], "neither": counts["neither"],
                    }
            normalized = {}
            for field, counts in transitions[policy].items():
                violation = field not in ("native_success", "safe_success")
                normalized[field] = {
                    ("cleared" if violation else "improved"): counts[
                        "cleared_new_only" if violation else "improved_new_only"],
                    ("introduced" if violation else "regressed"): counts[
                        "introduced_new_only" if violation else "regressed_baseline_only"],
                    "both": counts["both"], "neither": counts["neither"],
                }
            validate_transition_marginals(
                normalized,
                {field: sum(bool(row["result"][field]) for row in old) for field in fields},
                {field: sum(bool(row["result"][field]) for row in current) for field in fields},
                30,
            )
    return {
        "comparison_type": "engineering pipeline comparison; not isolated codec/data/inference effect",
        "formal_performance_claim": False, "generalization_claim": False,
        "baseline": registration["baseline"],
        "new_checkpoints": {
            name: {"path": registration["paths"][f"{name.split('_')[0]}_checkpoint"],
                   "sha256": sha256_file(registration["paths"][f"{name.split('_')[0]}_checkpoint"])}
            for name in ("base_dp", "constrain_dp")
        },
        "metrics": {"baseline": {name: metric_counts(rows) for name, rows in baselines.items()},
                    "new": {name: metric_counts(rows) for name, rows in current_rows.items()}},
        "transitions": transitions,
        "pairing_failure_scene_indices": sorted(set(pairing_failures)),
        "transition_order": "both, baseline_only (cleared for violations), new_only (introduced for violations), neither",
    }


def freeze_pilot_identity(repo_root=None, fullscene=False):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = [
        "examples/baselines/constrain/pickcube_learning_pilot.py",
        "examples/baselines/constrain/pickcube_transport_detour.py",
        "examples/baselines/constrain/pickcube_expert.py",
        "examples/baselines/constrain/pickcube_base_dp.py",
        "examples/baselines/constrain/pickcube_constrain.py",
        "examples/baselines/constrain/constrain_denoiser.py",
    ]
    if fullscene:
        files.extend((
            "examples/baselines/constrain/pickcube_action_codec.py",
            "examples/baselines/constrain/pickcube_obstacle.py",
            "examples/baselines/constrain/constrain_ckpt.py",
            "examples/baselines/constrain/diagnose_pickcube_contact.py",
            "examples/baselines/constrain/evaluate_pickcube_paired.py",
            "examples/baselines/constrain/pickcube_scenes.py",
            "examples/baselines/constrain/materialize_pickcube_fullscene500.py",
        ))
    hashed = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in files}
    return {
        "basis": "working-tree bytes frozen before pilot runs",
        "git_head_is_code_identity": False,
        "files": hashed,
        "files_sha256": hashlib.sha256(json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


EXPANSION_DIRECT_COUNT = 1000
EXPANSION_OBSTACLE_COUNT = 100
OBSTACLE_AUTHORIZED_CAP = 1000
OBSTACLE_INCREMENT_COUNT = 300
OBSTACLE_INCREMENT_NEW = 200
EXPANSION_EXP = "outputs/exp/20261024_104348_pickcube_data_expansion"
EXPANSION_DATASET = "demos/PickCube-transport-detour/20261024_104348_expansion"
EXPANSION_BASE_CKPT = "outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt"
EXPANSION_BASE_SHA = "dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d"
EXPANSION_CONSTRAIN_CKPT = "outputs/ckpt/20261024_104348_pickcube_expansion_constrain/final.pt"
EXPANSION_CONSTRAIN_SHA = "9ca9022aee201ae7f9ffe702d941863d6ffd4e4c6056f220fcdb2d2fe23c830d"
EXPANSION_H5_SHA = "2af959e3d3ebeba8bb046db6d9583d300e1533fc638c7d36eb6393dd28024dff"
PROXIMITY_CKPT_SHA = "22d3624941645ecac5bab4bd8679a2e40f77498b531b40f5e1d3c4be79d2a037"
OBSTACLE300_EXP = "outputs/exp/20261024_213325_pickcube_obstacle300"
OBSTACLE300_DATASET = "demos/PickCube-transport-detour/20261024_213325_obstacle300"
OBSTACLE300_CKPT = "outputs/ckpt/20261024_213325_pickcube_obstacle300_constrain/final.pt"
OBSTACLE300_CKPT_SHA = "bd7757122b75ad0bcc9cc5bd88a4b5174d48dfc520d1c8459332eea157732801"
OBSTACLE300_H5_SHA = "3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6"
WINDOW_GAP_PROTOCOL = {
    "sample_seed": 2026102501,
    "open_loop_seed": 2026102502,
    "sampling_std_seed": 2026102503,
    "sampling_std_windows": 64,
    "diffusion_timesteps": 100,
    "num_train_timesteps": 100,
    "pred_horizon": 16,
    "act_horizon": 8,
    "executed_slice": [1, 9],
    "near_rotation_m": 0.12,
    "near_epsilon_m": 0.08,
    "pos_action_scale_m": 0.1,
    "rot_action_scale_rad": 0.1,
    "batch_size": 64,
}
GAP_EPSILON_RATIO_MAX = 2.0
GAP_NEAR_ROTATION_MAE_MAX = 0.02
GAP_SHARED_EPSILON_FACTOR = 1.25
LONGER_BUDGET_UPDATES = 90000
SHARED_WINDOW_COUNT = 8965
TRAIN300_WINDOW_COUNT = 27588
EXISTING_OBSTACLE_MANIFEST = f"{EXPANSION_EXP}/pair_scenes/merged/manifest.json"
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


def obstacle_increment_id_plan():
    return {
        "direct": list(range(EXPANSION_DIRECT_COUNT)),
        "obstacle": list(range(1000, 1000 + OBSTACLE_INCREMENT_COUNT)),
        "retained_obstacle": list(range(1000, 1100)),
        "new_obstacle": list(range(1100, 1000 + OBSTACLE_INCREMENT_COUNT)),
    }


def validate_obstacle_increment_ids(direct_ids, obstacle_ids):
    direct_ids = [int(item) for item in direct_ids]
    obstacle_ids = [int(item) for item in obstacle_ids]
    if set(range(1000, 1100)) - set(obstacle_ids):
        raise ValueError("original obstacle ids missing")
    if obstacle_ids != list(range(1000, 1000 + OBSTACLE_INCREMENT_COUNT)):
        raise ValueError("obstacle ids are not the frozen 300")
    if direct_ids != list(range(EXPANSION_DIRECT_COUNT)):
        raise ValueError("direct ids are not the frozen 1000")
    return True


def assert_obstacle_increment_independence(existing, new, dev, prior):
    combined = list(existing) + list(new)
    kept, rejected = partition_task_independent(combined)
    if rejected or len(kept) != len(combined):
        raise ValueError("obstacle layouts are not unique")
    assert_external_independence(new, dev, prior)
    assert_external_independence(existing, dev, prior)
    return True


def base_replay_reusable(rows, scenes, expected_sha, expansion_identity, current_identity):
    if (current_identity or {}).get("files_sha256") != (expansion_identity or {}).get("files_sha256"):
        return False
    if len(rows) != len(scenes):
        return False
    ordered = sorted(rows, key=lambda row: row.get("scene_index", 0))
    for row, scene in zip(ordered, scenes):
        if row.get("status") != "evaluated" or row.get("checkpoint_sha256") != expected_sha:
            return False
        if int(row.get("seed")) != int(scene.get("seed")):
            return False
        if row.get("restored_state_sha256") != scene.get("state_sha256"):
            return False
    return True


def retained_expansion_records(source_h5, manifest, expected_direct, expected_obstacle):
    source_h5 = Path(source_h5)
    before = sha256_file(source_h5)
    records = []
    with h5py.File(source_h5, "r") as handle:
        for episode in manifest["episodes"]:
            episode_id = int(episode["episode_id"])
            group = handle[f"traj_{episode_id}"]
            if group.attrs.get("kind") != episode["kind"]:
                raise ValueError(f"traj_{episode_id} kind does not match the source manifest")
            records.append({
                "kind": episode["kind"],
                "episode_id": episode_id,
                "scene_id": episode["scene_id"],
                "obs": np.asarray(group["obs"]),
                "actions": np.asarray(group["actions"]),
                "terminated": np.asarray(group["terminated"]),
                "truncated": np.asarray(group["truncated"]),
                "native_success": True,
                "held_at_success": True,
                "obstacle_collision": bool(episode.get("obstacle_collision", False)),
                "obstacle_crossing": bool(episode.get("obstacle_crossing", False)),
                "layout": episode["layout"],
                "provenance": dict(episode.get("provenance") or {}),
            })
    if sha256_file(source_h5) != before:
        raise RuntimeError("expansion source mutated")
    direct = [row for row in records if row["kind"] == "direct"]
    obstacle = [row for row in records if row["kind"] == "obstacle"]
    if len(direct) != int(expected_direct) or len(obstacle) != int(expected_obstacle):
        raise ValueError("retained counts do not match the expansion source")
    conflicts = len(partition_task_independent([row["layout"] for row in direct])[1])
    recorded = (manifest.get("expansion") or {}).get("within_train_task_conflicts")
    if recorded is not None and conflicts != int(recorded):
        raise ValueError("direct duplicate count changed; refusing to drop or relabel them")
    return records, {
        "direct_unique_claim": False,
        "within_direct_task_conflicts": conflicts,
        "direct_count": len(direct),
        "retained_obstacle_count": len(obstacle),
        "source_h5_sha256": before,
    }


def assert_unweighted_increment(registration):
    training = registration["training"]
    if training.get("loss_weight") is not None or training.get("proximity_weight") != "off":
        raise RuntimeError("proximity weight must be off")
    if training.get("base_retrained") is not False:
        raise RuntimeError("this slice does not retrain base")
    if int(registration["train_obstacle_count"]) != OBSTACLE_INCREMENT_COUNT:
        raise RuntimeError("this slice is 300 obstacle demonstrations")
    return True


def layouts_from_episodes(path, kind):
    payload = json.loads(Path(path).read_text())
    return [layout_view(episode["layout"]) for episode in payload["episodes"] if episode["kind"] == kind]


OBSTACLE_INCREMENT_ADDENDUM_DOES_NOT_CHANGE = [
    "counts", "task_identity_definition", "gate", "training", "geometry",
    "expert", "external_excludes", "within_run_dedup", "acceptance_predicate",
]


def load_obstacle_increment_addendum(path, registration):
    payload = json.loads(Path(path).read_text())
    if payload.get("reason") != "measured_yield_below_preregistered_attempt_budget":
        raise ValueError("unrecognized obstacle increment addendum")
    if payload.get("does_not_change") != OBSTACLE_INCREMENT_ADDENDUM_DOES_NOT_CHANGE:
        raise ValueError("obstacle increment addendum changes a frozen rule")
    if payload.get("within_run_mutual_exclusion") is not True:
        raise ValueError("addendum relaxes within-run identity")
    if int(payload.get("target")) != OBSTACLE_INCREMENT_NEW:
        raise ValueError("addendum changes the 200 target")
    extra = payload["continuation"]
    if int(extra["per_attempt_deadline"]) != 150:
        raise ValueError("addendum changes the attempt deadline")
    if int(extra["max_attempts"]) < 1 or float(extra["wall_seconds"]) <= 0:
        raise ValueError("addendum budget is not finite")
    extra_span = (int(extra["seed"]) + 1, int(extra["seed"]) + int(extra["max_attempts"]))
    for span in _seed_span(registration["collection"]["obstacle"]):
        if not (extra_span[1] < span[0] or span[1] < extra_span[0]):
            raise ValueError("addendum seed range overlaps the preregistered seeds")
    return payload


def _obstacle_increment_collection():
    obstacle = {
        "workers": 2,
        "target": OBSTACLE_INCREMENT_NEW,
        "max_attempts": 3000,
        "wall_seconds": 28000,
        "per_attempt_deadline": 150,
        "blocks": [
            {"name": "block_a", "seed": 2026130001, "target": 50, "max_attempts": 500, "wall_seconds": 7200},
            {"name": "block_b", "seed": 2026131001, "target": 50, "max_attempts": 500, "wall_seconds": 7200},
            {"name": "block_c", "seed": 2026132001, "target": 50, "max_attempts": 500, "wall_seconds": 7200},
            {"name": "block_d", "seed": 2026133001, "target": 50, "max_attempts": 500, "wall_seconds": 7200},
        ],
        "continuation": {"name": "continuation", "seed": 2026134001, "max_attempts": 1000, "wall_seconds": 8000},
    }
    spans = sorted(_seed_span(obstacle))
    if any(left[1] >= right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("collection seed ranges overlap")
    if sum(block["max_attempts"] for block in obstacle["blocks"]) + obstacle["continuation"]["max_attempts"] != obstacle["max_attempts"]:
        raise ValueError("attempt budget does not add up")
    if sum(block["target"] for block in obstacle["blocks"]) != obstacle["target"]:
        raise ValueError("block targets do not add up")
    return obstacle


def build_obstacle_increment_preregistration(version, paths, *, reason, prior_version, baseline_files=None):
    if not reason or not prior_version:
        raise ValueError("obstacle increment requires the recorded authorization and the prior version")
    if OBSTACLE_INCREMENT_COUNT != 300 or OBSTACLE_INCREMENT_COUNT > OBSTACLE_AUTHORIZED_CAP:
        raise ValueError("this slice is not the authorized 300")
    plan = obstacle_increment_id_plan()
    training = build_preregistration(1, version, paths)["training"]
    training["loss_weight"] = None
    training["proximity_weight"] = "off"
    training["base_retrained"] = False
    training["differences_from_formal"] = [
        "direct arrays are the existing 1000, copied and not retrained; they are not claimed unique",
        "obstacle count is the authorized 300 slice, not the 100-plan and not the 1000 cap",
        "control branch is trained from the matched expansion Base EMA with unweighted epsilon",
        "development scenes are the already-used 30; this is not a formal performance claim",
    ]
    return {
        "kind": "authorized_obstacle_increment",
        "round_index": "obstacle300",
        "version": version,
        "reason": reason,
        "prior_version": prior_version,
        "prior_manifest": PRIOR_MANIFEST,
        "paths": dict(paths),
        "obstacle_plan_cap": OBSTACLE_INCREMENT_COUNT,
        "authorized_obstacle_cap": OBSTACLE_AUTHORIZED_CAP,
        "slice_is_not_the_authorized_cap": True,
        "train_obstacle_count": OBSTACLE_INCREMENT_COUNT,
        "train_direct_count": EXPANSION_DIRECT_COUNT,
        "selected_direct_ids": plan["direct"],
        "selected_obstacle_ids": plan["obstacle"],
        "dev_scene_count": DEV_SCENE_COUNT,
        "dev_reused": True,
        "direct_unique_claim": False,
        "known_within_direct_task_conflicts": 663,
        "layout_dedup": {
            "tolerance_m": LAYOUT_TOLERANCE_M,
            "metric": "chebyshev",
            "mode": "task",
            "cylinder_proximity_is_conflict": False,
            "within_run_enforced": True,
            "external_excludes": ["ticket03", "dev30", "existing100"],
            "conflict_if": "same seed, or start and goal both within tolerance",
            "exclude_prior_source": PRIOR_MANIFEST,
            "exclude_dev": FROZEN_DEV_SCENES,
            "exclude_existing_obstacle": EXISTING_OBSTACLE_MANIFEST,
        },
        "collection": {
            "dev_collected": False,
            "dev_scenes": FROZEN_DEV_SCENES,
            "direct_recollected": False,
            "obstacle": _obstacle_increment_collection(),
            "acceptance_predicate": "full_run_attempt",
            "package_new_direct": False,
        },
        "training": training,
        "eval": {
            "max_steps": 500,
            "episode_wall_seconds": 120,
            "denominator": DEV_SCENE_COUNT,
            "noise_seed_rule": "torch.manual_seed(scene seed) before each scene",
            "scene_source": FROZEN_DEV_SCENES,
            "base_reuse": "only if checkpoint sha, restored state sha, scene seed, and source identity match",
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
            "on_fail": "dev-only pause; does not admit Ticket 07; a later slice is a new round",
        },
        "comparison": {
            "name": "expansion_100_obstacle",
            "expansion_paired": f"{EXPANSION_EXP}/paired",
            "constrain_counts": {"task": 26, "safe": 13, "contact": 17, "xy": 3},
            "base_counts": {"task": 10, "safe": 0, "contact": 30, "xy": 15},
            "rejected_proximity_is_not_baseline": True,
            "rejected_proximity_counts": {"task": 22, "safe": 11, "contact": 18, "xy": 4},
            "rejected_proximity_sha256": PROXIMITY_CKPT_SHA,
            "expansion_constrain_sha256": EXPANSION_CONSTRAIN_SHA,
        },
        "frozen_identities": {
            "base_checkpoint_sha256": EXPANSION_BASE_SHA,
            "expansion_h5_sha256": EXPANSION_H5_SHA,
            "expansion_constrain_sha256": EXPANSION_CONSTRAIN_SHA,
            "rejected_proximity_sha256": PROXIMITY_CKPT_SHA,
        },
        "files": dict(baseline_files or {}),
        "formal_performance_claim": False,
    }


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


def validate_transition_marginals(transitions, baseline_counts, current_counts, denominator):
    """Reject paired transition tables that cannot produce their cohort marginals."""
    for field, buckets in transitions.items():
        if field not in baseline_counts or field not in current_counts:
            raise ValueError(f"missing marginal count for {field}")
        if sum(int(value) for value in buckets.values()) != int(denominator):
            raise ValueError(f"{field} transitions do not sum to denominator")
        if field in VIOLATION_FIELDS:
            baseline_total = buckets["cleared"] + buckets["both"]
            current_total = buckets["introduced"] + buckets["both"]
        else:
            baseline_total = buckets["regressed"] + buckets["both"]
            current_total = buckets["improved"] + buckets["both"]
        if baseline_total != int(baseline_counts[field]) or current_total != int(current_counts[field]):
            raise ValueError(f"{field} transitions contradict cohort marginals")


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


def fixed_window_draw(count, seed, pred_horizon=16, act_dim=7, num_timesteps=100):
    count = int(count)
    if count < 1:
        raise ValueError("draw count must be positive")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    noise = torch.randn(count, int(pred_horizon), int(act_dim), generator=generator)
    timesteps = torch.randint(0, int(num_timesteps), (count,), generator=generator)
    return noise, timesteps


def draw_sha256(noise, timesteps):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(noise.detach().cpu().numpy()).tobytes())
    digest.update(np.ascontiguousarray(timesteps.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def rotation_gap(predicted, expert, distance, near_m=0.12):
    predicted = torch.as_tensor(predicted, dtype=torch.float32)
    expert = torch.as_tensor(expert, dtype=torch.float32)
    distance = torch.as_tensor(distance, dtype=torch.float32).reshape(-1)
    if predicted.shape != expert.shape or predicted.ndim != 3 or predicted.shape[1:] != (8, 7):
        raise ValueError("rotation gap expects the 8-step executed slice")
    if len(distance) != len(predicted):
        raise ValueError("rotation gap distance does not match the windows")
    per = (predicted[..., 3:6] - expert[..., 3:6]).abs().reshape(len(predicted), -1).mean(1)
    near = distance < float(near_m)
    far = ~near

    def _mean(mask):
        if not bool(mask.any()):
            return None
        return float(per[mask].mean())

    return {
        "near_m": float(near_m),
        "near_count": int(near.sum()),
        "far_count": int(far.sum()),
        "near_rotation_mae": _mean(near),
        "far_rotation_mae": _mean(far),
    }


def _gap_fields(metrics, name):
    required = ("train_epsilon", "dev_epsilon", "shared_epsilon", "near_rotation_mae", "far_rotation_mae")
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        raise ValueError(f"{name} missing {missing}")
    if any(float(metrics[key]) < 0 for key in required):
        raise ValueError(f"{name} metric is negative")
    values = {key: float(metrics[key]) for key in required}
    if values["train_epsilon"] <= 0 or values["shared_epsilon"] <= 0:
        raise ValueError(f"{name} epsilon baseline must be positive")
    return values


def classify_window_gap(metrics_300, metrics_100):
    """Pre-registered from the published 100-obstacle gaps, not from this run.

    The prior dev/train epsilon ratio was about 4 and the dev near rotation MAE
    was 0.090 against a far MAE of 0.005. Closed means both are inside the
    stricter bars below. A worse shared-window fit is underfit even if those
    ratios look closed.
    """
    current = _gap_fields(metrics_300, "obstacle300")
    baseline = _gap_fields(metrics_100, "obstacle100")
    epsilon_ratio = current["dev_epsilon"] / current["train_epsilon"]
    epsilon_gap = epsilon_ratio > GAP_EPSILON_RATIO_MAX
    rotation_gap_remains = current["near_rotation_mae"] > GAP_NEAR_ROTATION_MAE_MAX
    shared_worse = current["shared_epsilon"] > GAP_SHARED_EPSILON_FACTOR * baseline["shared_epsilon"]
    gap_closed = not epsilon_gap and not rotation_gap_remains
    underfit = (not gap_closed) or shared_worse
    return {
        "rule": "pre-registered window gap",
        "epsilon_ratio": epsilon_ratio,
        "epsilon_ratio_max": GAP_EPSILON_RATIO_MAX,
        "near_rotation_mae_max": GAP_NEAR_ROTATION_MAE_MAX,
        "shared_epsilon_factor": GAP_SHARED_EPSILON_FACTOR,
        "epsilon_gap": epsilon_gap,
        "rotation_gap_remains": rotation_gap_remains,
        "shared_fit_worse": shared_worse,
        "gap_closed": gap_closed,
        "underfit": underfit,
        "next": "longer_budget" if underfit else "closed_loop",
        "formal_performance_claim": False,
    }


def longer_budget_start(scheduler_state, optimizer_present, chosen_updates=LONGER_BUDGET_UPDATES):
    last = (scheduler_state or {}).get("_last_lr") or []
    last_lr = float(last[0]) if last else None
    if optimizer_present and last_lr is not None and last_lr > 0:
        return {
            "start": "resume",
            "chosen_updates": int(chosen_updates),
            "last_lr": last_lr,
            "reason": "optimizer state and a nonzero saved learning rate are both present",
        }
    if optimizer_present:
        reason = (
            "optimizer state is present but the saved cosine last_lr is 0 at the finished "
            "30000-step schedule; retaining that scheduler cannot deliver additional exposure. "
            "Scratch 90000 from the matched Base EMA uses a cosine sized for 90000."
        )
    else:
        reason = (
            "optimizer state is absent, so exact resume is not supported. "
            "Scratch 90000 from the matched Base EMA."
        )
    return {
        "start": "scratch",
        "chosen_updates": int(chosen_updates),
        "last_lr": last_lr,
        "reason": reason,
    }


def build_window_gap_preregistration(version, paths):
    return {
        "kind": "authorized_window_gap_diagnosis",
        "version": version,
        "paths": dict(paths),
        "no_collection": True,
        "no_training": True,
        "formal_performance_claim": False,
        "changed_factor": "none",
        "protocol": dict(WINDOW_GAP_PROTOCOL),
        "decision_rule": {
            "name": "pre-registered window gap",
            "epsilon_ratio_max": GAP_EPSILON_RATIO_MAX,
            "near_rotation_mae_max": GAP_NEAR_ROTATION_MAE_MAX,
            "shared_epsilon_factor": GAP_SHARED_EPSILON_FACTOR,
            "longer_budget_updates": LONGER_BUDGET_UPDATES,
            "source": (
                "prior 100 measurement: dev/train epsilon about 4x; "
                "dev near rotation MAE 0.090 versus far 0.005"
            ),
        },
        "checkpoints": {
            "obstacle100": {"path": EXPANSION_CONSTRAIN_CKPT, "sha256": EXPANSION_CONSTRAIN_SHA, "episode_ids": list(range(1000, 1100))},
            "obstacle300": {"path": OBSTACLE300_CKPT, "sha256": OBSTACLE300_CKPT_SHA, "episode_ids": list(range(1000, 1300))},
        },
        "datasets": {
            "expansion_h5": {"path": f"{EXPANSION_DATASET}/trajectory.h5", "sha256": EXPANSION_H5_SHA},
            "obstacle300_h5": {"path": f"{OBSTACLE300_DATASET}/trajectory.h5", "sha256": OBSTACLE300_H5_SHA},
            "dev_scenes": FROZEN_DEV_SCENES,
        },
        "base_checkpoint_sha256": EXPANSION_BASE_SHA,
        "comparison_counts": {
            "obstacle100": {"task": 26, "safe": 13, "contact": 17, "xy": 3},
            "obstacle300": {"task": 25, "safe": 12, "contact": 18, "xy": 2},
        },
    }


def load_window_gap_preregistration(path):
    payload = json.loads(Path(path).read_text())
    expected = build_window_gap_preregistration(payload.get("version"), payload.get("paths") or {})
    for key in (
        "kind", "no_collection", "no_training", "formal_performance_claim", "protocol",
        "decision_rule", "checkpoints", "datasets", "base_checkpoint_sha256",
    ):
        if payload.get(key) != expected.get(key):
            raise ValueError(f"window-gap preregistration field changed: {key}")
    if not (payload.get("source_identity") or {}).get("files_sha256"):
        raise ValueError("window-gap preregistration has no source identity")
    return payload


def build_window_gap_index(splits, protocol, source_identity, decision):
    if not (source_identity or {}).get("files_sha256"):
        raise ValueError("source identity missing")
    if (decision or {}).get("rule") != "pre-registered window gap":
        raise ValueError("decision rule was not the frozen rule")
    for name in ("shared100", "dev30"):
        split = splits.get(name) or {}
        if "obstacle100" not in split or "obstacle300" not in split:
            raise ValueError(f"{name} is missing obstacle100 or obstacle300")
        if not split.get("draw_sha256") or not split.get("sample_sha256"):
            raise ValueError(f"{name} has no reproducible draw")
        left = split["obstacle100"].get("draw_sha256")
        right = split["obstacle300"].get("draw_sha256")
        if left != right or left != split["draw_sha256"]:
            raise ValueError(f"{name} sample seeds or diffusion timesteps differ")
    train = splits.get("train300") or {}
    if "obstacle300" not in train or not train.get("draw_sha256"):
        raise ValueError("train300 fit missing")
    return {
        "protocol": protocol,
        "source_identity": source_identity,
        "splits": splits,
        "decision": decision,
        "same_sample_seeds": True,
        "same_diffusion_timesteps": True,
        "prose_only": False,
    }


def assert_exposure_budget(registration):
    training = registration["training"]
    if registration.get("changed_factor") != "constrain_updates":
        raise RuntimeError("exposure slice changed factor is not the update budget")
    if training.get("start") != "scratch":
        raise RuntimeError("this slice is scratch; resume was not selected")
    if int(training.get("total_iters", -1)) != LONGER_BUDGET_UPDATES:
        raise RuntimeError("budget is not the preregistered 90000")
    if expected_base_updates(registration) != 30000:
        raise RuntimeError("base updates are not the frozen 30000")
    if training.get("loss_weight") is not None or training.get("proximity_weight") != "off":
        raise RuntimeError("proximity weight must stay off")
    if training.get("base_retrained") is not False:
        raise RuntimeError("this slice does not retrain base")
    return True


def build_exposure_preregistration(version, paths, diagnosis_index, diagnosis_path):
    decision = (diagnosis_index or {}).get("decision") or {}
    if decision.get("next") != "longer_budget" or decision.get("underfit") is not True:
        raise ValueError("exposure trial requires the underfit diagnosis")
    if decision.get("rule") != "pre-registered window gap":
        raise ValueError("diagnosis rule mismatch")
    resume = (diagnosis_index or {}).get("resume_decision") or {}
    if resume.get("start") != "scratch" or float(resume.get("last_lr", -1)) != 0.0:
        raise ValueError("exposure trial requires the recorded scratch decision")
    training = build_preregistration(1, version, paths)["training"]
    training["total_iters"] = LONGER_BUDGET_UPDATES
    training["base_updates"] = 30000
    training["loss_weight"] = None
    training["proximity_weight"] = "off"
    training["base_retrained"] = False
    training["start"] = "scratch"
    training["differences_from_formal"] = [
        "only the constrain update budget changes, from 30000 to 90000",
        "scratch from the matched Base EMA; the saved 30000-step cosine last_lr is 0",
        "direct arrays stay the copied 1000 and obstacle stays the frozen 300; no new collection",
        "development scenes are the already-used 30; this is not a formal performance claim",
    ]
    plan = obstacle_increment_id_plan()
    return {
        "kind": "authorized_training_optimization",
        "version": version,
        "paths": dict(paths),
        "changed_factor": "constrain_updates",
        "updates_before": 30000,
        "updates_after": LONGER_BUDGET_UPDATES,
        "no_collection": True,
        "formal_performance_claim": False,
        "round_index": "exposure90000",
        "train_direct_count": EXPANSION_DIRECT_COUNT,
        "train_obstacle_count": OBSTACLE_INCREMENT_COUNT,
        "obstacle_plan_cap": OBSTACLE_INCREMENT_COUNT,
        "dev_scene_count": DEV_SCENE_COUNT,
        "selected_direct_ids": plan["direct"],
        "selected_obstacle_ids": plan["obstacle"],
        "layout_dedup": {"mode": "task", "cylinder_proximity_is_conflict": False},
        "training": training,
        "gate": {
            "task_success_rate_gte": 0.90,
            "safe_success_rate_gte": 0.85,
            "contact_rate_lte": 0.10,
            "xy_crossing_rate_lte": 0.10,
            "require_complete": True,
            "require_zero_runtime_errors": True,
            "formal_performance_claim": False,
            "on_fail": "dev-only pause; does not admit Ticket 07; do not start another slice in this run",
        },
        "eval": {
            "max_steps": 500,
            "episode_wall_seconds": 120,
            "denominator": DEV_SCENE_COUNT,
            "noise_seed_rule": "torch.manual_seed(scene seed) before each scene",
            "scene_source": FROZEN_DEV_SCENES,
        },
        "hypothesis": (
            "300 shared-window epsilon is higher than the 100 checkpoint on the same "
            "noise and timesteps after 1114 epochs versus 3427. One scratch 90000-update "
            "cosine from the matched Base EMA tests that exposure gap. Final EMA is "
            "preselected. Dev success does not select the checkpoint."
        ),
        "diagnosis_index": diagnosis_path,
        "base_checkpoint_sha256": EXPANSION_BASE_SHA,
        "datasets": {
            "obstacle300_h5": {"path": f"{OBSTACLE300_DATASET}/trajectory.h5", "sha256": OBSTACLE300_H5_SHA},
            "expansion_h5": {"path": f"{EXPANSION_DATASET}/trajectory.h5", "sha256": EXPANSION_H5_SHA},
        },
        "comparison": {
            "obstacle100": {"task": 26, "safe": 13, "contact": 17, "xy": 3},
            "obstacle300": {"task": 25, "safe": 12, "contact": 18, "xy": 2},
            "obstacle100_paired": f"{EXPANSION_EXP}/paired",
            "obstacle300_paired": f"{OBSTACLE300_EXP}/paired",
            "obstacle100_sha256": EXPANSION_CONSTRAIN_SHA,
            "obstacle300_sha256": OBSTACLE300_CKPT_SHA,
            "rejected_proximity_is_not_baseline": True,
            "rejected_proximity_sha256": PROXIMITY_CKPT_SHA,
        },
        "wall": {"train_seconds": 5000, "expected_train_seconds": 2500, "eval_seconds": 8000},
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


def enforce_runtime_source_identity(registration, current, command):
    """Every runtime command checks the working-tree hash. No kind skips it."""
    if registration.get("kind") == "authorized_data_expansion":
        resolved = resolve_running_identity(registration, current)
        resolved["command"] = command
        return resolved
    frozen = (registration.get("source_identity") or {}).get("files_sha256")
    if current.get("files_sha256") == frozen:
        return {
            "matched": "preregistration", "files_sha256": frozen,
            "kind": registration.get("kind"), "command": command,
        }
    recovery_path = Path(registration["paths"]["exp"]) / "recovery_identity.json"
    if not recovery_path.exists():
        raise RuntimeError("source identity does not match preregistration")
    recovery = json.loads(recovery_path.read_text())
    if recovery.get("exempt") is True or recovery.get("skip_identity") is True:
        raise RuntimeError("recovery identity cannot bypass the identity gate")
    if recovery.get("files_sha256") != current.get("files_sha256"):
        raise RuntimeError("recovery identity does not match the running code")
    if not recovery.get("reason"):
        raise RuntimeError("recovery identity requires a recorded reason")
    if recovery.get("preregistration_files_sha256") != frozen:
        raise RuntimeError("recovery identity does not cite the preregistered hash")
    return {
        "matched": "recovery_identity", "files_sha256": current["files_sha256"],
        "kind": registration.get("kind"), "command": command,
    }


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
                           spec_override=None, within_run=True, drop_conflicts=True,
                           extra_layouts=()):
    spec = spec_override or registration["collection"][stage_name]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    exclude = _stage_excludes(registration, extra_manifests)
    exclude.extend(extra_layouts)
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


def _stack_windows(rows):
    if not rows:
        raise ValueError("no expert windows")
    return {
        "episode_id": [row["episode_id"] for row in rows],
        "start": [int(row["start"]) for row in rows],
        "observations": torch.stack([row["observations"] for row in rows]),
        "actions": torch.stack([row["actions"] for row in rows]),
    }


def _append_windows(rows, episode_id, observations, actions):
    observations = torch.as_tensor(observations, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if observations.ndim != 2 or observations.shape[1] != 45 or actions.shape[1] != 7 or len(observations) != len(actions) + 1:
        raise ValueError(f"{episode_id} expert schema mismatch")
    for start, end in _window_slices(len(actions)):
        obs_seq, act_seq = materialize_window(observations, actions, start, end)
        rows.append({"episode_id": episode_id, "start": int(start), "observations": obs_seq, "actions": act_seq})


def windows_from_h5(path, episode_ids):
    rows = []
    with h5py.File(path, "r") as handle:
        for episode_id in episode_ids:
            group = handle[f"traj_{int(episode_id)}"]
            _append_windows(rows, int(episode_id), group["obs"][:], group["actions"][:])
    return _stack_windows(rows)


def windows_from_dev_scenes(scene_dir):
    rows = []
    with h5py.File(Path(scene_dir) / "scenes.h5", "r") as handle:
        scene_ids = sorted(handle.keys())
        if len(scene_ids) != DEV_SCENE_COUNT:
            raise ValueError("dev scene count is not 30")
        for scene_id in scene_ids:
            group = handle[scene_id]
            _append_windows(rows, scene_id, group["observations"][:], group["expert_actions"][:])
    return _stack_windows(rows)


def tcp_obstacle_distance(observations):
    from examples.baselines.constrain.pickcube_constrain import OBSTACLE_XY, TCP_XY

    latest = observations[..., -1, :]
    return torch.linalg.norm(latest[..., list(OBSTACLE_XY)] - latest[..., list(TCP_XY)], dim=-1)


def load_gap_checkpoint(path, expected_sha, expected_ids, device):
    if sha256_file(path) != expected_sha:
        raise ValueError(f"checkpoint sha mismatch for {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    config = ck.get("config") or {}
    data = config.get("data_identity") or {}
    if (ck.get("format") != CONSTRAIN_FORMAT or int(ck.get("step", -1)) != 30000
            or config.get("selected_weights") != SELECTED_WEIGHTS
            or list(data.get("selected_episode_ids") or []) != list(expected_ids)
            or (config.get("base_weights") or {}).get("sha256") != EXPANSION_BASE_SHA
            or config.get("loss_weight") is not None):
        raise ValueError(f"gap checkpoint failed strict identity: {path}")
    from diffusers.training_utils import EMAModel
    from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
    from examples.baselines.constrain.pickcube_constrain import MODEL_CONFIG, PickCubeConditionNorm

    denoiser = build_constrain_denoiser(MODEL_CONFIG, device=device)
    denoiser.load_state_dict(ck["model"], strict=True)
    ema = EMAModel(parameters=denoiser.trainable_parameters(), power=0.75)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(denoiser.trainable_parameters())
    denoiser.eval()
    return denoiser, PickCubeConditionNorm.from_dict(config["normalization"]), ck


def _diffusion_scheduler():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon")


def score_epsilon_draw(denoiser, norm, observations, actions, noise, timesteps, device, batch_size):
    from examples.baselines.constrain.pickcube_constrain import pickcube_relative_condition

    scheduler = _diffusion_scheduler()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            stop = start + batch_size
            obs = observations[start:stop].to(device)
            act = actions[start:stop].to(device)
            drawn_noise = noise[start:stop].to(device)
            drawn_steps = timesteps[start:stop].to(device)
            hint = pickcube_relative_condition(obs, norm)[:, None, :].expand(-1, act.shape[1], -1)
            noisy = scheduler.add_noise(act, drawn_noise, drawn_steps)
            prediction = denoiser(
                noisy, drawn_steps, global_cond=obs[..., :42].flatten(1), hint=hint, enabled=True)
            parts.append((prediction - drawn_noise).pow(2).flatten(1).mean(1).detach().cpu())
    return torch.cat(parts)


def sample_executed_slice(denoiser, norm, observations, init_noise, device, batch_size, label):
    from examples.baselines.constrain.pickcube_constrain import pickcube_relative_condition

    scheduler = _diffusion_scheduler()
    scheduler.set_timesteps(100)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            stop = min(start + batch_size, len(observations))
            obs = observations[start:stop].to(device)
            sample = init_noise[start:stop].to(device)
            hint = pickcube_relative_condition(obs, norm)[:, None, :].expand(-1, 16, -1)
            cond = obs[..., :42].flatten(1)
            for timestep in scheduler.timesteps:
                prediction = denoiser(sample, timestep, global_cond=cond, hint=hint, enabled=True)
                sample = scheduler.step(prediction, timestep, sample).prev_sample
            executed = sample[:, 1:9]
            if bool((executed.abs() > 1.000001).any()):
                raise ValueError("sampled action left [-1, 1]")
            parts.append(executed.detach().cpu())
            print(f"{label} sampled {stop}/{len(observations)}", flush=True)
    return torch.cat(parts)


def _split_score(epsilon, distance, executed, actions, protocol):
    near = distance < float(protocol["near_epsilon_m"])
    report = rotation_gap(executed, actions[:, 1:9], distance, near_m=protocol["near_rotation_m"])
    report.update({
        "epsilon": float(epsilon.mean()),
        "count": int(epsilon.shape[0]),
        "near_epsilon_m": float(protocol["near_epsilon_m"]),
        "near_epsilon_count": int(near.sum()),
        "near_epsilon": float(epsilon[near].mean()) if bool(near.any()) else None,
    })
    return report


def _score_model(name, denoiser, norm, splits, draws, device, protocol):
    batch = int(protocol["batch_size"])
    scored = {}
    for split_name, windows in splits.items():
        noise, timesteps = draws[split_name]["epsilon"]
        init_noise = draws[split_name]["sample"]
        epsilon = score_epsilon_draw(
            denoiser, norm, windows["observations"], windows["actions"], noise, timesteps, device, batch)
        executed = sample_executed_slice(
            denoiser, norm, windows["observations"], init_noise, device, batch, f"{name}:{split_name}")
        distance = tcp_obstacle_distance(windows["observations"])
        metrics = _split_score(epsilon, distance, executed, windows["actions"], protocol)
        metrics["draw_sha256"] = draws[split_name]["draw_sha256"]
        metrics["sample_sha256"] = draws[split_name]["sample_sha256"]
        scored[split_name] = metrics
        print(f"{name} {split_name} epsilon={metrics['epsilon']:.6f} near_rot={metrics['near_rotation_mae']}", flush=True)
    return scored


def _sampling_std(first, second, protocol, count):
    delta = (first[:count] - second[:count]).abs()
    return {
        "windows": int(count),
        "xyz_mm": float(delta[..., :3].mean() * float(protocol["pos_action_scale_m"]) * 1000),
        "rotation_rad": float(delta[..., 3:6].mean() * float(protocol["rot_action_scale_rad"])),
    }


def run_window_gap_diagnosis(registration, device="cuda"):
    try:
        return _run_window_gap_diagnosis(registration, device)
    finally:
        _kill_descendants()


def _run_window_gap_diagnosis(registration, device):
    protocol = registration["protocol"]
    datasets = registration["datasets"]
    checkpoints = registration["checkpoints"]
    expansion_h5 = datasets["expansion_h5"]["path"]
    obstacle_h5 = datasets["obstacle300_h5"]["path"]
    if sha256_file(expansion_h5) != EXPANSION_H5_SHA or sha256_file(obstacle_h5) != OBSTACLE300_H5_SHA:
        raise RuntimeError("frozen dataset hash changed")
    if sha256_file(EXPANSION_BASE_CKPT) != EXPANSION_BASE_SHA:
        raise RuntimeError("matched Base EMA hash changed")
    copied_ids = list(range(1000)) + list(range(1000, 1100))
    assert_packaged_matches_source(expansion_h5, obstacle_h5, copied_ids)
    shared = windows_from_h5(expansion_h5, checkpoints["obstacle100"]["episode_ids"])
    train300 = windows_from_h5(obstacle_h5, checkpoints["obstacle300"]["episode_ids"])
    dev = windows_from_dev_scenes(datasets["dev_scenes"])
    if len(shared["observations"]) != SHARED_WINDOW_COUNT or len(train300["observations"]) != TRAIN300_WINDOW_COUNT:
        raise RuntimeError("expert window count does not match the frozen training summaries")
    draws = {}
    for name, windows, seed_offset in (
        ("shared100", shared, 0), ("dev30", dev, 1), ("train300", train300, 2),
    ):
        noise, timesteps = fixed_window_draw(len(windows["observations"]), protocol["sample_seed"] + seed_offset)
        sample_noise, _ = fixed_window_draw(len(windows["observations"]), protocol["open_loop_seed"] + seed_offset)
        draws[name] = {
            "epsilon": (noise, timesteps),
            "sample": sample_noise,
            "draw_sha256": draw_sha256(noise, timesteps),
            "sample_sha256": draw_sha256(sample_noise, timesteps),
        }
    output = Path(registration["paths"]["diagnosis"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    denoiser, norm, ck100 = load_gap_checkpoint(
        checkpoints["obstacle100"]["path"], checkpoints["obstacle100"]["sha256"],
        checkpoints["obstacle100"]["episode_ids"], device)
    scored100 = _score_model("obstacle100", denoiser, norm, {"shared100": shared, "dev30": dev}, draws, device, protocol)
    del denoiser, ck100
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    denoiser, norm, ck300 = load_gap_checkpoint(
        checkpoints["obstacle300"]["path"], checkpoints["obstacle300"]["sha256"],
        checkpoints["obstacle300"]["episode_ids"], device)
    scored300 = _score_model(
        "obstacle300", denoiser, norm,
        {"shared100": shared, "dev30": dev, "train300": train300}, draws, device, protocol)
    std_count = min(int(protocol["sampling_std_windows"]), len(dev["observations"]))
    std_noise, _ = fixed_window_draw(len(dev["observations"]), protocol["sampling_std_seed"])
    std_executed = sample_executed_slice(
        denoiser, norm, dev["observations"][:std_count], std_noise[:std_count], device, protocol["batch_size"],
        "obstacle300:std")
    first_dev = sample_executed_slice(
        denoiser, norm, dev["observations"][:std_count], draws["dev30"]["sample"][:std_count], device,
        protocol["batch_size"], "obstacle300:std-base")
    sampling = _sampling_std(first_dev, std_executed, protocol, std_count)
    scheduler_decision = longer_budget_start(ck300.get("lr_scheduler"), ck300.get("optimizer") is not None)
    del denoiser, ck300
    metrics100 = {
        "train_epsilon": scored100["shared100"]["epsilon"],
        "dev_epsilon": scored100["dev30"]["epsilon"],
        "shared_epsilon": scored100["shared100"]["epsilon"],
        "near_rotation_mae": scored100["dev30"]["near_rotation_mae"],
        "far_rotation_mae": scored100["dev30"]["far_rotation_mae"],
    }
    metrics300 = {
        "train_epsilon": scored300["train300"]["epsilon"],
        "dev_epsilon": scored300["dev30"]["epsilon"],
        "shared_epsilon": scored300["shared100"]["epsilon"],
        "near_rotation_mae": scored300["dev30"]["near_rotation_mae"],
        "far_rotation_mae": scored300["dev30"]["far_rotation_mae"],
    }
    decision = classify_window_gap(metrics300, metrics100)
    decision["metrics_obstacle100"] = metrics100
    decision["metrics_obstacle300"] = metrics300
    splits = {}
    for name in ("shared100", "dev30"):
        splits[name] = {
            "draw_sha256": draws[name]["draw_sha256"],
            "sample_sha256": draws[name]["sample_sha256"],
            "window_count": int(len(shared["observations"] if name == "shared100" else dev["observations"])),
            "obstacle100": scored100[name],
            "obstacle300": scored300[name],
        }
    splits["train300"] = {
        "draw_sha256": draws["train300"]["draw_sha256"],
        "sample_sha256": draws["train300"]["sample_sha256"],
        "window_count": int(len(train300["observations"])),
        "obstacle300": scored300["train300"],
    }
    if sha256_file(expansion_h5) != EXPANSION_H5_SHA or sha256_file(obstacle_h5) != OBSTACLE300_H5_SHA:
        raise RuntimeError("frozen dataset hash changed during diagnosis")
    index = build_window_gap_index(splits, protocol, freeze_pilot_identity(), decision)
    index["sampling_std_obstacle300_dev"] = sampling
    index["resume_decision"] = scheduler_decision
    index["copied_100_matches_expansion"] = True
    index["elapsed_seconds"] = time.monotonic() - started
    index["formal_performance_claim"] = False
    _json_atomic(output / "index.json", index)
    _json_atomic(output / "obstacle100.json", {"checkpoint_sha256": EXPANSION_CONSTRAIN_SHA, "splits": scored100})
    _json_atomic(output / "obstacle300.json", {
        "checkpoint_sha256": OBSTACLE300_CKPT_SHA,
        "splits": scored300,
        "resume_decision": scheduler_decision,
        "sampling_std": sampling,
    })
    print(json.dumps({"decision": decision["next"], "underfit": decision["underfit"], "gap_closed": decision["gap_closed"]}), flush=True)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "preregister", "preregister-fullscene500", "preregister-native500", "preregister-r36-ddpm", "preregister-expansion", "preregister-obstacle-increment",
        "collect-train", "collect-dev", "package",
        "include-original", "collect-pairs", "collect-direct", "package-expansion",
        "collect-obstacle-increment", "package-obstacle-increment",
        "train-base", "train-constrain", "evaluate", "gate", "compare-baseline",
        "compare-increment", "preregister-window-gap", "diagnose-window-gap",
        "preregister-exposure", "compare-exposure",
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
    parser.add_argument("--inference-steps", type=int, default=50,
                        help="opt-in DDIM inference steps for the frozen action-codec pair")
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
    if args.command == "preregister-fullscene500":
        import importlib.util
        from examples.baselines.constrain.pickcube_action_codec import profile_sha256
        from examples.baselines.constrain.materialize_pickcube_fullscene500 import file_tree_hash
        exp = args.registration.parent
        dataset = Path("demos/PickCube-transport-detour/20261028_164500_frozen500")
        dev = Path(FROZEN_DEV_SCENES)
        baseline_exp = Path("outputs/exp/20261025_065159_pickcube_exposure90000")
        baseline = {
            "base_checkpoint": "outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt",
            "base_checkpoint_sha256": sha256_file("outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt"),
            "constrain_checkpoint": "outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt",
            "constrain_checkpoint_sha256": sha256_file("outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt"),
            "paired_base": str(baseline_exp / "paired/base_dp"),
            "paired_constrain": str(baseline_exp / "paired/constrain_dp"),
            "paired_base_summary_sha256": sha256_file(baseline_exp / "paired/base_dp/summary.json"),
            "paired_constrain_summary_sha256": sha256_file(baseline_exp / "paired/constrain_dp/summary.json"),
            "paired_episode_tree_sha256": file_tree_hash(baseline_exp / "paired"),
            "baseline_summary": {"base": {"task": "10/30", "safe": "0/30", "contact": "30/30", "xy": "15/30"},
                                 "constrain": {"task": "27/30", "safe": "15/30", "contact": "15/30", "xy": "3/30"}},
            "comparison_interpretation": "engineering pipeline comparison; data, codec, training budget and DDIM stride differ",
        }
        paths = {
            "exp": str(exp), "demos": str(dataset), "dev_scenes": str(dev),
            "base_run": str(exp / "train_base"), "constrain_run": str(exp / "train_constrain"),
            "base_checkpoint_dir": f"outputs/ckpt/{exp.name.split('_pickcube')[0]}_pickcube_fullscene500_base",
            "constrain_checkpoint_dir": f"outputs/ckpt/{exp.name.split('_pickcube')[0]}_pickcube_fullscene500_constrain",
            "base_checkpoint": f"outputs/ckpt/{exp.name.split('_pickcube')[0]}_pickcube_fullscene500_base/final.pt",
            "constrain_checkpoint": f"outputs/ckpt/{exp.name.split('_pickcube')[0]}_pickcube_fullscene500_constrain/final.pt",
            "paired": str(exp / "paired"),
        }
        matched_base_identity = None
        matched_constrain_identity = None
        if args.base_checkpoint is not None:
            base_path = Path(args.base_checkpoint).resolve()
            base_ck = torch.load(base_path, map_location="cpu", weights_only=False)
            base_payload = base_ck.get("payload") or {}
            if (base_ck.get("format") != FULLSCENE_BASE_FORMAT
                    or base_ck.get("selected_weights") != "ema"
                    or int(base_ck.get("step", -1)) != 30000
                    or base_payload.get("action_encoding_sha256") != profile_sha256()
                    or base_payload.get("h5_sha256") != sha256_file(dataset / "trajectory.h5")
                    or base_payload.get("manifest_sha256") != sha256_file(dataset / "trajectory.json")):
                raise ValueError("reused Base is not the completed 1000-direct full-scene500 EMA")
            matched_base_identity = {"path": str(base_path), "sha256": sha256_file(base_path),
                                     "step": 30000, "selected_weights": "ema"}
            paths["base_checkpoint"] = str(base_path)
            paths["base_checkpoint_dir"] = str(base_path.parent)
            paths["base_run"] = "outputs/exp/20261028_111705_pickcube_fullscene500_action_codec/train_base"
        if args.constrain_checkpoint is not None:
            control_path = Path(args.constrain_checkpoint).resolve()
            control_ck = torch.load(control_path, map_location="cpu", weights_only=False)
            control_config = control_ck.get("config") or {}
            if (control_ck.get("format") != FULLSCENE_CONSTRAIN_FORMAT
                    or int(control_ck.get("step", -1)) != 150000
                    or control_config.get("action_encoding_sha256") != profile_sha256()
                    or control_config.get("data_identity", {}).get("h5_sha256") != sha256_file(dataset / "trajectory.h5")
                    or control_config.get("base_weights", {}).get("sha256")
                    != (matched_base_identity or {}).get("sha256")):
                raise ValueError("reused Constrain checkpoint is not the completed matched 500-obstacle EMA")
            matched_constrain_identity = {"path": str(control_path), "sha256": sha256_file(control_path),
                                          "step": 150000, "selected_weights": "ema"}
            paths["constrain_checkpoint"] = str(control_path)
            paths["constrain_checkpoint_dir"] = str(control_path.parent)
        source = freeze_pilot_identity(fullscene=True)
        identities = {
            "datasets": {
                "h5": {"path": str(dataset / "trajectory.h5"), "sha256": sha256_file(dataset / "trajectory.h5")},
                "manifest": {"path": str(dataset / "trajectory.json"), "sha256": sha256_file(dataset / "trajectory.json")},
                "validation": {"path": str(dataset / "validation.json"), "sha256": sha256_file(dataset / "validation.json")},
            },
            "baseline": baseline,
            "matched_base_checkpoint": matched_base_identity,
            "matched_constrain_checkpoint": matched_constrain_identity,
            "validation": {
                "dev_manifest": {"path": str(dev / "manifest.json"), "sha256": sha256_file(dev / "manifest.json")},
                "dev_scenes_h5": {"path": str(dev / "scenes.h5"), "sha256": sha256_file(dev / "scenes.h5")},
                "dataset_validation_reference": "frozen validation.json: native channels 0..5 [-.35,.35], gripper [-1,1], held-success tail CLOSED",
                "action_encoding_audit": audit_fullscene_action_encoding(
                    dataset / "trajectory.h5", dataset / "trajectory.json"),
            },
        }
        runtime_module = importlib.util.find_spec("mani_skill.envs.tasks.tabletop.pick_cube")
        if runtime_module is None or runtime_module.origin is None:
            raise RuntimeError("cannot resolve the ManiSkill PickCube task source")
        identities["validation"]["maniskill_pickcube_source"] = {
            "path": runtime_module.origin, "sha256": sha256_file(runtime_module.origin),
        }
        payload = build_fullscene500_preregistration(paths, identities, source)
        if args.prior_version:
            prior_path = Path(args.prior_version)
            payload["training_preregistration"] = {
                "path": str(prior_path), "sha256": sha256_file(prior_path),
                "source_identity_sha256": json.loads(prior_path.read_text())["source_identity"]["files_sha256"],
            }
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    if args.command == "preregister-native500":
        import importlib.util
        from examples.baselines.constrain.materialize_pickcube_fullscene500 import file_tree_hash
        exp = args.registration.parent
        dataset = Path(NATIVE500_H5).parent
        dev = Path(FROZEN_DEV_SCENES)
        baseline_exp = Path("outputs/exp/20261025_065159_pickcube_exposure90000")
        base_path = Path(NATIVE500_BASE)
        if sha256_file(base_path) != NATIVE500_BASE_SHA256:
            raise RuntimeError("original native direct Base EMA hash changed")
        base_ck = torch.load(base_path, map_location="cpu", weights_only=False)
        base_payload = base_ck.get("payload") or {}
        if (base_ck.get("format") != BASE_FORMAT or base_ck.get("step") != 30000
                or base_ck.get("selected_weights") != "ema"
                or base_payload.get("selected_class") != "direct"
                or base_payload.get("selected_episode_ids") != list(range(1000))
                or base_payload.get("action_encoding") is not None):
            raise RuntimeError("original Base EMA is not the expected 1000-direct native checkpoint")
        if (base_payload.get("h5_sha256") != NATIVE500_BASE_H5_SHA256
                or base_payload.get("manifest_sha256") != NATIVE500_BASE_MANIFEST_SHA256
                or sha256_file(NATIVE500_BASE_H5) != NATIVE500_BASE_H5_SHA256):
            raise RuntimeError("original Base data identity is not the retained expansion source")
        _verify_native_direct_subset(dataset / "trajectory.h5", NATIVE500_BASE_H5)
        baseline = {
            "base_checkpoint": NATIVE500_BASE, "base_checkpoint_sha256": NATIVE500_BASE_SHA256,
            "constrain_checkpoint": "outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt",
            "constrain_checkpoint_sha256": sha256_file("outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt"),
            "paired_base": str(baseline_exp / "paired/base_dp"),
            "paired_constrain": str(baseline_exp / "paired/constrain_dp"),
            "paired_episode_tree_sha256": file_tree_hash(baseline_exp / "paired"),
            "protocol_recovery": "not recoverable from retained preregistration/index/commands; this trial uses the explicit DDIM50 hypothesis",
            "counts": {"base": {"task": 10, "safe": 0, "contact": 30, "xy": 15},
                       "constrain": {"task": 27, "safe": 15, "contact": 15, "xy": 3}},
        }
        paths = {
            "exp": str(exp), "demos": str(dataset), "dev_scenes": str(dev),
            "constrain_run": str(exp / "train_constrain"),
            "constrain_checkpoint_dir": f"outputs/ckpt/{exp.name.split('_pickcube')[0]}_pickcube_fullscene500_native_constrain",
            "constrain_checkpoint": f"outputs/ckpt/{exp.name.split('_pickcube')[0]}_pickcube_fullscene500_native_constrain/final.pt",
            "base_checkpoint": NATIVE500_BASE, "paired": str(exp / "paired"),
        }
        runtime_module = importlib.util.find_spec("mani_skill.envs.tasks.tabletop.pick_cube")
        if runtime_module is None or runtime_module.origin is None:
            raise RuntimeError("cannot resolve the ManiSkill PickCube task source")
        identities = {
            "datasets": {
                "h5": {"path": NATIVE500_H5, "sha256": sha256_file(NATIVE500_H5)},
                "manifest": {"path": NATIVE500_MANIFEST, "sha256": sha256_file(NATIVE500_MANIFEST)},
            },
            "validation": {
                "dataset_validation": {"path": str(dataset / "validation.json"),
                                       "sha256": sha256_file(dataset / "validation.json")},
                "dev_manifest": {"path": str(dev / "manifest.json"), "sha256": sha256_file(dev / "manifest.json")},
                "dev_scenes_h5": {"path": str(dev / "scenes.h5"), "sha256": sha256_file(dev / "scenes.h5")},
                "maniskill_pickcube_source": {"path": runtime_module.origin,
                                               "sha256": sha256_file(runtime_module.origin)},
            },
            "baseline": baseline,
        }
        payload = build_fullscene500_native_preregistration(
            paths, identities, freeze_pilot_identity(fullscene=True))
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    if args.command == "preregister-r36-ddpm":
        import importlib.util
        from examples.baselines.constrain.materialize_pickcube_fullscene500 import file_tree_hash
        exp = args.registration.parent
        dataset = Path(NATIVE500_H5).parent
        dev = Path(FROZEN_DEV_SCENES)
        r35 = Path("outputs/exp/20261028_145500_pickcube_native_control_r35")
        base_path = Path(NATIVE500_BASE)
        constrain_path = Path("outputs/ckpt/20261028_145500_pickcube_fullscene500_native_constrain_retry2/final.pt")
        if base_path.is_file() is False or sha256_file(base_path) != NATIVE500_BASE_SHA256:
            raise RuntimeError("R36 Base checkpoint identity mismatch")
        if constrain_path.is_file() is False or sha256_file(constrain_path) != "be5b2b740a59eaf428c71a906e15eb5c83d1cf807e1629c3bdf060ac31b8db93":
            raise RuntimeError("R36 Constrain checkpoint identity mismatch")
        baseline_paired = r35 / "paired"
        baseline = {
            "paired_base": str(baseline_paired / "base_dp"),
            "paired_constrain": str(baseline_paired / "constrain_dp"),
            "paired_episode_tree_sha256": file_tree_hash(baseline_paired),
            "base_checkpoint_sha256": NATIVE500_BASE_SHA256,
            "constrain_checkpoint_sha256": "be5b2b740a59eaf428c71a906e15eb5c83d1cf807e1629c3bdf060ac31b8db93",
            "protocol": "R35 actual DDIM-50, stride 8; frozen same-weight primary comparator",
        }
        paths = {
            "exp": str(exp), "demos": str(dataset), "dev_scenes": str(dev),
            "base_checkpoint": str(base_path), "constrain_checkpoint": str(constrain_path),
            "paired": str(exp / "paired"),
        }
        runtime_module = importlib.util.find_spec("mani_skill.envs.tasks.tabletop.pick_cube")
        if runtime_module is None or runtime_module.origin is None:
            raise RuntimeError("cannot resolve the ManiSkill PickCube task source")
        identities = {
            "datasets": {"h5": {"path": NATIVE500_H5, "sha256": sha256_file(NATIVE500_H5)},
                         "manifest": {"path": NATIVE500_MANIFEST, "sha256": sha256_file(NATIVE500_MANIFEST)}},
            "validation": {
                "dataset_validation": {"path": str(dataset / "validation.json"), "sha256": sha256_file(dataset / "validation.json")},
                "dev_manifest": {"path": str(dev / "manifest.json"), "sha256": sha256_file(dev / "manifest.json")},
                "dev_scenes_h5": {"path": str(dev / "scenes.h5"), "sha256": sha256_file(dev / "scenes.h5")},
                "maniskill_pickcube_source": {"path": runtime_module.origin, "sha256": sha256_file(runtime_module.origin)},
            },
            "baseline": baseline,
            "matched_base_checkpoint": {"path": str(base_path), "sha256": NATIVE500_BASE_SHA256},
            "matched_constrain_checkpoint": {"path": str(constrain_path), "sha256": baseline["constrain_checkpoint_sha256"]},
        }
        payload = build_r36_archived_ddpm_preregistration(paths, identities, freeze_pilot_identity(fullscene=True))
        payload["baseline"] = baseline
        payload["matched_constrain_checkpoint"] = identities["matched_constrain_checkpoint"]
        payload["gate"] = {"task_success_count_gte": 27, "safe_success_count_gte": 26,
                            "contact_count_lte": 3, "xy_crossing_count_lte": 3,
                            "require_complete": True, "require_zero_runtime_errors": True,
                            "formal_performance_claim": False}
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
    if args.command == "preregister-obstacle-increment":
        from examples.baselines.constrain.pickcube_transport_detour import freeze_source_identity
        exp = args.registration.parent
        stamp = exp.name.split("_pickcube")[0]
        expected = {
            "expansion_h5": (f"{EXPANSION_DATASET}/trajectory.h5", EXPANSION_H5_SHA),
            "expansion_json": (f"{EXPANSION_DATASET}/trajectory.json", None),
            "base_checkpoint": (EXPANSION_BASE_CKPT, EXPANSION_BASE_SHA),
            "expansion_constrain_checkpoint": (EXPANSION_CONSTRAIN_CKPT, EXPANSION_CONSTRAIN_SHA),
            "existing_obstacle_manifest": (EXISTING_OBSTACLE_MANIFEST, None),
            "dev_manifest": (f"{FROZEN_DEV_SCENES}/manifest.json", None),
            "dev_scenes_h5": (f"{FROZEN_DEV_SCENES}/scenes.h5", None),
            "ticket03_manifest": (PRIOR_MANIFEST, None),
            "rejected_proximity_checkpoint": (
                "outputs/ckpt/20261024_202858_pickcube_proximity_weight_constrain/final.pt",
                PROXIMITY_CKPT_SHA,
            ),
        }
        hashed = {}
        for key, (path, digest) in expected.items():
            actual = sha256_file(path)
            if digest is not None and actual != digest:
                raise RuntimeError(f"{path} hash does not match the frozen identity")
            hashed[key] = {"path": path, "sha256": actual}
        payload = build_obstacle_increment_preregistration(
            args.version, {
                "exp": str(exp),
                "obstacle_scenes": str(exp / "obstacle_scenes"),
                "demos": f"demos/PickCube-transport-detour/{stamp}_obstacle300",
                "ckpt_base": EXPANSION_BASE_CKPT,
                "ckpt_constrain": f"outputs/ckpt/{stamp}_pickcube_obstacle300_constrain",
                "train_constrain": str(exp / "train_constrain"),
                "paired": str(exp / "paired"),
                "dev_scenes": FROZEN_DEV_SCENES,
            }, reason=args.reason, prior_version=args.prior_version or EXPANSION_EXP, baseline_files=hashed)
        payload["source_identity"] = freeze_pilot_identity()
        payload["generator_identity"] = freeze_source_identity()
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    if args.command == "preregister-window-gap":
        exp = args.registration.parent
        payload = build_window_gap_preregistration(args.version, {
            "exp": str(exp),
            "diagnosis": str(exp / "diagnosis"),
        })
        payload["source_identity"] = freeze_pilot_identity()
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    if args.command == "preregister-exposure":
        diagnosis_path = Path(args.prior_version)
        diagnosis = json.loads(diagnosis_path.read_text())
        exp = args.registration.parent
        stamp = exp.name.split("_pickcube")[0]
        payload = build_exposure_preregistration(args.version, {
            "exp": str(exp),
            "demos": OBSTACLE300_DATASET,
            "ckpt_base": EXPANSION_BASE_CKPT,
            "ckpt_constrain": f"outputs/ckpt/{stamp}_pickcube_exposure90000_constrain",
            "train_constrain": str(exp / "train_constrain"),
            "paired": str(exp / "paired"),
            "dev_scenes": FROZEN_DEV_SCENES,
        }, diagnosis, str(diagnosis_path))
        for key, spec in payload["datasets"].items():
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"{key} hash does not match the frozen identity")
        if sha256_file(EXPANSION_BASE_CKPT) != EXPANSION_BASE_SHA:
            raise RuntimeError("matched Base EMA hash changed")
        payload["source_identity"] = freeze_pilot_identity()
        payload["diagnosis_index_sha256"] = sha256_file(diagnosis_path)
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    if args.command == "diagnose-window-gap":
        registration = load_window_gap_preregistration(args.registration)
        enforce_runtime_source_identity(registration, freeze_pilot_identity(), args.command)
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return run_window_gap_diagnosis(registration, args.device)
    registration = load_preregistration(args.registration)
    enforce_runtime_source_identity(
        registration,
        freeze_pilot_identity(fullscene=registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND)),
        args.command,
    )
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
    if args.command == "collect-obstacle-increment":
        if registration.get("kind") != "authorized_obstacle_increment":
            raise SystemExit("obstacle increment collection requires its preregistration")
        extra = layouts_from_episodes(registration["files"]["expansion_json"]["path"], "obstacle")
        if len(extra) != EXPANSION_OBSTACLE_COUNT:
            raise SystemExit("existing obstacle layouts are not the frozen 100")
        report = collect_expansion_stage(
            registration, "obstacle", registration["paths"]["obstacle_scenes"],
            extra_layouts=extra, within_run=True, drop_conflicts=True)
        _require_new_generator(registration, report)
        addendum_path = Path(registration["paths"]["exp"]) / "obstacle_increment_addendum.json"
        if not report["complete"] and addendum_path.exists():
            addendum = load_obstacle_increment_addendum(addendum_path, registration)
            initial = Path(registration["paths"]["obstacle_scenes"]) / "stage_report_initial.json"
            if not initial.exists():
                initial.write_bytes((Path(registration["paths"]["obstacle_scenes"]) / "stage_report.json").read_bytes())
            report = collect_obstacle_increment_deficit(registration, report, addendum)
            identity_path = Path(report["deficit_block"]["output_dir"]) / "source_identity.json"
            expected = (registration.get("generator_identity") or {}).get("files_sha256")
            if identity_path.exists() and json.loads(identity_path.read_text())["files_sha256"] != expected:
                raise RuntimeError("deficit generator identity does not match preregistration")
            _json_atomic(Path(registration["paths"]["obstacle_scenes"]) / "stage_report.json", report)
        print(json.dumps({
            "stage": report["stage"], "target": report["target"], "kept": report["kept"],
            "complete": report["complete"], "restore_audit_ok": report["restore_audit_ok"],
        }), flush=True)
        if not report["complete"]:
            raise SystemExit("obstacle increment collection incomplete")
        return report
    if args.command == "package-obstacle-increment":
        if registration.get("kind") != "authorized_obstacle_increment":
            raise SystemExit("obstacle increment packaging requires its preregistration")
        source_h5 = Path(registration["files"]["expansion_h5"]["path"])
        if sha256_file(source_h5) != registration["frozen_identities"]["expansion_h5_sha256"]:
            raise RuntimeError("expansion direct source hash changed")
        stage = json.loads((Path(registration["paths"]["obstacle_scenes"]) / "stage_report.json").read_text())
        if not stage.get("complete") or int(stage.get("kept") or 0) != OBSTACLE_INCREMENT_NEW:
            raise RuntimeError("new obstacle catalog is not the preregistered 200")
        if not stage.get("restore_audit_ok"):
            raise RuntimeError("new obstacle restore audit failed")
        source_manifest = json.loads(Path(registration["files"]["expansion_json"]["path"]).read_text())
        retained, quality = retained_expansion_records(
            source_h5, source_manifest, expected_direct=1000, expected_obstacle=100)
        new_records = new_obstacle_records_from_catalog(stage["catalog"])
        if len(new_records) != OBSTACLE_INCREMENT_NEW:
            raise RuntimeError("new obstacle records are not 200")
        dev = layouts_from_manifest(Path(FROZEN_DEV_SCENES) / "manifest.json")
        prior = layouts_from_manifest(PRIOR_MANIFEST)
        existing = [row["layout"] for row in retained if row["kind"] == "obstacle"]
        assert_obstacle_increment_independence(
            existing, [row["layout"] for row in new_records], dev, prior)
        records = retained + new_records
        validate_obstacle_increment_ids(
            [int(row["episode_id"]) for row in records if row["kind"] == "direct"],
            [int(row["episode_id"]) for row in records if row["kind"] == "obstacle"])
        dest = package_records(records, registration["paths"]["demos"], registration)
        assert_packaged_matches_source(
            source_h5, dest / "trajectory.h5", list(range(1000)) + list(range(1000, 1100)))
        if sha256_file(source_h5) != registration["frozen_identities"]["expansion_h5_sha256"]:
            raise RuntimeError("expansion source mutated during packaging")
        manifest = json.loads((dest / "trajectory.json").read_text())
        manifest["obstacle_increment"] = {
            "source_h5_sha256": registration["frozen_identities"]["expansion_h5_sha256"],
            "direct_copied": True,
            "direct_unique_claim": False,
            "within_direct_task_conflicts": quality["within_direct_task_conflicts"],
            "retained_obstacle_ids": list(range(1000, 1100)),
            "new_obstacle_ids": list(range(1100, 1300)),
            "new_obstacle_unique": True,
            "restore_audit_ok": True,
            "formal_performance_claim": False,
        }
        (dest / "trajectory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        quality.update({
            "new_obstacle_count": len(new_records),
            "obstacle_count": 300,
            "new_obstacle_unique": True,
            "restore_audit_ok": True,
            "source_mutated": False,
            "formal_performance_claim": False,
        })
        _json_atomic(Path(registration["paths"]["exp"]) / "data_quality.json", quality)
        print(dest, flush=True)
        return dest
    if args.command == "train-base":
        if registration.get("kind") == NATIVE500_KIND:
            raise SystemExit("native full-scene500 trial freezes the original Base; base retraining is forbidden")
        if registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
            expected = registration["paths"]
            if (Path(args.run_dir).resolve() != Path(expected["base_run"]).resolve()
                    or Path(args.checkpoint_dir).resolve() != Path(expected["base_checkpoint_dir"]).resolve()
                    or Path(args.demo_dir).resolve() != Path(expected["demos"]).resolve()):
                raise SystemExit("base training output/data paths differ from preregistration")
        if registration.get("changed_factor") == "constrain_updates":
            raise SystemExit("this slice does not retrain base")
        if registration.get("kind") == "authorized_obstacle_increment":
            raise SystemExit("this slice does not retrain base")
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
        return train_base(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.run_dir, args.checkpoint_dir, registration, args.device)
    if args.command == "train-constrain":
        if registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
            expected = registration["paths"]
            if (Path(args.run_dir).resolve() != Path(expected["constrain_run"]).resolve()
                    or Path(args.checkpoint_dir).resolve() != Path(expected["constrain_checkpoint_dir"]).resolve()
                    or Path(args.base_checkpoint).resolve() != Path(expected["base_checkpoint"]).resolve()
                    or Path(args.demo_dir).resolve() != Path(expected["demos"]).resolve()):
                raise SystemExit("constrain training output/base/data paths differ from preregistration")
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
        if registration.get("kind") == "authorized_obstacle_increment":
            assert_unweighted_increment(registration)
            if sha256_file(args.base_checkpoint) != registration["frozen_identities"]["base_checkpoint_sha256"]:
                raise SystemExit("base checkpoint is not the matched expansion EMA")
        if registration.get("changed_factor") == "constrain_updates":
            assert_exposure_budget(registration)
            if sha256_file(args.base_checkpoint) != registration["base_checkpoint_sha256"]:
                raise RuntimeError("base checkpoint is not the matched expansion EMA")
            datasets = registration["datasets"]
            if sha256_file(datasets["obstacle300_h5"]["path"]) != datasets["obstacle300_h5"]["sha256"]:
                raise RuntimeError("frozen 300 dataset hash changed")
            if sha256_file(datasets["expansion_h5"]["path"]) != datasets["expansion_h5"]["sha256"]:
                raise RuntimeError("frozen 100 dataset hash changed")
        return train_constrain(
            args.demo_dir / "trajectory.h5", args.demo_dir / "trajectory.json",
            args.base_checkpoint, args.run_dir, args.checkpoint_dir, registration, args.device)
    if args.command == "evaluate":
        import shutil
        from examples.baselines.constrain.materialize_pickcube_fullscene500 import file_tree_hash
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
        native500 = registration.get("kind") == NATIVE500_KIND
        if registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
            expected = registration["paths"]
            if (Path(args.output_dir).resolve() != Path(expected["paired"]).resolve()
                    or Path(args.base_checkpoint).resolve() != Path(expected["base_checkpoint"]).resolve()
                    or Path(args.constrain_checkpoint).resolve() != Path(expected["constrain_checkpoint"]).resolve()):
                raise SystemExit("paired evaluation output/checkpoint paths differ from preregistration")
            if Path(args.scene_dir).resolve() != Path(registration["paths"]["dev_scenes"]).resolve():
                raise SystemExit("full-scene500 evaluation must use the preregistered dev scenes")
            for spec in registration["validation"].values():
                if isinstance(spec, dict) and "path" in spec and sha256_file(spec["path"]) != spec["sha256"]:
                    raise RuntimeError("frozen dev scene identity differs from preregistration")
            for key in ("h5", "manifest"):
                spec = registration["datasets"][key]
                if sha256_file(spec["path"]) != spec["sha256"]:
                    raise RuntimeError("frozen training dataset changed after preregistration")
        if registration.get("kind") == "authorized_data_expansion":
            resolve_running_identity(registration, freeze_pilot_identity())
            if Path(args.scene_dir).resolve() != Path(registration["paths"]["dev_scenes"]).resolve():
                raise SystemExit("expansion eval must use the frozen dev scenes")
        if registration.get("changed_factor") == "constrain_updates":
            if Path(args.scene_dir).resolve() != Path(FROZEN_DEV_SCENES).resolve():
                raise SystemExit("exposure eval must use the frozen dev scenes")
            assert_exposure_budget(registration)
        if registration.get("kind") == "authorized_obstacle_increment":
            if Path(args.scene_dir).resolve() != Path(registration["paths"]["dev_scenes"]).resolve():
                raise SystemExit("increment eval must use the frozen dev scenes")
            demo_json = Path(registration["paths"]["demos"]) / "trajectory.json"
            if demo_json.exists():
                assert_obstacle_increment_independence(
                    layouts_from_episodes(demo_json, "obstacle"), [],
                    layouts_from_manifest(args.scene_dir / "manifest.json"),
                    layouts_from_manifest(PRIOR_MANIFEST))
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
        if not args.exclude_manifest:
            train_layouts = []
        elif registration.get("kind") == "frozen_fullscene500_action_codec":
            train_layouts = (layouts_from_episodes(args.exclude_manifest[0], "direct")
                             + layouts_from_episodes(args.exclude_manifest[0], "obstacle"))
        else:
            train_layouts = layouts_from_manifest(args.exclude_manifest[0])
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
        ) or registration.get("kind") == "frozen_fullscene500_action_codec"
        if train_layouts and (
            (not allow_within_train and _internal(train_layouts)) or _internal(dev_layouts)
            or _hit(train_layouts, dev_layouts) or _hit(train_layouts, prior) or _hit(dev_layouts, prior)
        ):
            raise ValueError("dev scenes are not independent of train or prior source")
        checkpoints = {
            "base_dp": args.base_checkpoint,
            "constrain_dp": args.constrain_checkpoint,
        }
        if args.inference_steps not in (50, 100):
            raise SystemExit("registered action-codec inference supports only 50 or 100 DDIM steps")
        if native500 and args.inference_steps != registration["inference_hypothesis"]["inference_steps"]:
            raise SystemExit("native500 inference must match its preregistered hypothesis")
        adapters = {
            "base_dp": PilotBaseAdapter(args.base_checkpoint, registration, args.device, args.inference_steps),
            "constrain_dp": PilotConstrainAdapter(
                args.constrain_checkpoint, registration, args.device, args.inference_steps),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        registered_sampler = registration.get("inference_hypothesis", {}).get("sampler", "DDPMScheduler")
        _json_atomic(args.output_dir / "inference_profile.json", {
            "scheduler": registered_sampler,
            "inference_steps": args.inference_steps if registration.get("action_encoding") is not None or native500 else 100,
            "training_timesteps": 100, "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon", "eta": 0.0 if registered_sampler == "DDIMScheduler" else None,
            "prediction_horizon": 16, "executed_prediction_slice": [1, 9],
            "execution_stride": registration.get("inference_hypothesis", {}).get("replan_stride", 2), "plans_per_call": 1,
            "action_encoding_sha256": registration.get("action_encoding_sha256"),
            "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
            "constrain_checkpoint_sha256": sha256_file(args.constrain_checkpoint),
            "scene_manifest_sha256": sha256_file(args.scene_dir / "manifest.json"),
            "source_identity": freeze_pilot_identity(fullscene=True),
        })
        if registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND):
            baseline_root = args.output_dir / "baseline_snapshot"
            shutil.copytree(Path(registration["baseline"]["paired_base"]).parent, baseline_root / "paired")
            if file_tree_hash(baseline_root / "paired") != registration["baseline"]["paired_episode_tree_sha256"]:
                raise RuntimeError("immutable historical paired-record snapshot hash mismatch")
        summaries = {}
        stored = {}
        reuse_rows = None
        if registration.get("kind") == "authorized_obstacle_increment":
            expansion_rows = load_episode_directory(
                Path(registration["comparison"]["expansion_paired"]) / "base_dp")
            scenes = json.loads((Path(FROZEN_DEV_SCENES) / "manifest.json").read_text())["scenes"]
            expansion_index = json.loads(
                (Path(registration["comparison"]["expansion_paired"]) / "base_dp" / "index.json").read_text())
            current_identity = freeze_pilot_identity()
            reusable = base_replay_reusable(
                expansion_rows, scenes,
                registration["frozen_identities"]["base_checkpoint_sha256"],
                expansion_index.get("source_identity") or {}, current_identity)
            _json_atomic(args.output_dir / "base_reuse.json", {
                "reusable": reusable,
                "reevaluated": not reusable,
                "reason": (
                    "matched checkpoint, restored state, seed, and source identity"
                    if reusable else
                    "identity, restored state, or seed does not match; reevaluated"
                ),
            })
            if reusable:
                reuse_rows = expansion_rows
                adapters.pop("base_dp")
        import signal
        episode_wall = (registration["evaluation"]["episode_wall_seconds"]
                        if registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND)
                        else registration["eval"]["episode_wall_seconds"])
        if reuse_rows is not None:
            policy_dir = args.output_dir / "base_dp"
            policy_dir.mkdir()
            checkpoint_sha = registration["frozen_identities"]["base_checkpoint_sha256"]
            rows = []
            for row in sorted(reuse_rows, key=lambda item: item["scene_index"]):
                copied = dict(row)
                copied["checkpoint_sha256"] = checkpoint_sha
                copied["reused_expansion_base"] = True
                _json_atomic(policy_dir / f"episode_{copied['scene_index']:03d}.json", copied)
                rows.append(copied)
            summaries["base_dp"] = summarize_pilot(rows, len(catalog))
            stored["base_dp"] = rows
            index = build_episode_index(rows, checkpoint_sha, checkpoint=str(checkpoints["base_dp"]))
            index["source_identity"] = freeze_pilot_identity(
                fullscene=registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND))
            index["reused_expansion_base"] = True
            _json_atomic(policy_dir / "summary.json", summaries["base_dp"])
            _json_atomic(policy_dir / "index.json", index)
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
            index["source_identity"] = freeze_pilot_identity(
                fullscene=registration.get("kind") in ("frozen_fullscene500_action_codec", NATIVE500_KIND))
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
        if registration.get("kind") == "frozen_fullscene500_action_codec":
            comparison = fullscene500_engineering_comparison(registration, args.output_dir, stored)
            _json_atomic(args.output_dir / "engineering_comparison.json", comparison)
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
    if args.command == "compare-increment":
        report = compare_obstacle_increment(registration, args.output_dir)
        _json_atomic(args.output_dir / "obstacle_increment_comparison.json", report)
        print(json.dumps({
            "increment_counts": report["increment_counts"],
            "transitions": report["transitions"],
            "formal_performance_claim": False,
        }, indent=2), flush=True)
        return report
    if args.command == "compare-exposure":
        report = compare_exposure_eval(registration, args.output_dir)
        _json_atomic(args.output_dir / "exposure_comparison.json", report)
        print(json.dumps({
            "exposure_counts": report["exposure_counts"],
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


def collect_obstacle_increment_deficit(registration, stage_report, addendum):
    catalog = Path(stage_report["catalog"])
    if not (catalog / "manifest.json").exists():
        raise RuntimeError("incomplete stage has no catalog")
    deficit = int(stage_report["target"]) - int(stage_report["kept"])
    if deficit < 1:
        raise RuntimeError("obstacle increment has no deficit")
    extra = addendum["continuation"]
    exclude = layouts_from_episodes(registration["files"]["expansion_json"]["path"], "obstacle")
    exclude.extend(layouts_from_manifest(Path(FROZEN_DEV_SCENES) / "manifest.json"))
    exclude.extend(layouts_from_manifest(PRIOR_MANIFEST))
    exclude.extend(layouts_from_manifest(catalog / "manifest.json"))
    block = {
        "name": extra["name"],
        "seed": int(extra["seed"]),
        "target": deficit,
        "max_attempts": int(extra["max_attempts"]),
        "wall_seconds": float(extra["wall_seconds"]),
    }
    report = _run_collection_block(
        block, Path(registration["paths"]["obstacle_scenes"]) / "blocks", exclude,
        int(extra["per_attempt_deadline"]), True)
    sources = [catalog]
    if (Path(report["output_dir"]) / "manifest.json").exists():
        sources.append(Path(report["output_dir"]))
    merged = Path(registration["paths"]["obstacle_scenes"]) / "assembled_deficit"
    if not merged.exists():
        assemble = assemble_catalogs(
            sources, merged, seed=block["seed"], provenance={}, drop_conflicts=True)
    else:
        assemble = {
            "kept": len(layouts_from_manifest(merged / "manifest.json")),
            "resumed": True,
        }
    audit_ok = bool(stage_report.get("restore_audit_ok")) and (
        not report.get("accepted") or bool(report.get("restore_audit_ok")))
    kept = int(assemble["kept"])
    updated = {
        "stage": "obstacle",
        "target": int(stage_report["target"]),
        "kept": kept,
        "complete": kept == int(stage_report["target"]) and audit_ok,
        "restore_audit_ok": audit_ok,
        "catalog": str(merged),
        "prior_kept": int(stage_report["kept"]),
        "prior_catalog": str(catalog),
        "deficit_block": report,
        "assemble": assemble,
        "identity_rules_unchanged": True,
    }
    _json_atomic(Path(registration["paths"]["obstacle_scenes"]) / "deficit_report.json", updated)
    return updated


def new_obstacle_records_from_catalog(catalog, start_id=1100):
    paired = _records_from_catalog(
        catalog, lambda index, _scene: (1_000_000 + index, start_id + index))
    records = [row for row in paired if row["kind"] == "obstacle"]
    expected = list(range(start_id, start_id + len(records)))
    if [int(row["episode_id"]) for row in records] != expected:
        raise ValueError("new obstacle ids are not contiguous")
    for row in records:
        row["provenance"]["source"] = "new_obstacle"
        if row["obstacle_collision"] or row["obstacle_crossing"] or not row["native_success"]:
            raise ValueError("new obstacle trajectory is not expert-safe")
    return records


def assert_packaged_matches_source(source_h5, packaged_h5, episode_ids):
    with h5py.File(source_h5, "r") as source, h5py.File(packaged_h5, "r") as packaged:
        for episode_id in episode_ids:
            left = source[f"traj_{int(episode_id)}"]
            right = packaged[f"traj_{int(episode_id)}"]
            for name in ("obs", "actions", "terminated", "truncated"):
                if not np.array_equal(left[name], right[name]):
                    raise ValueError(f"traj_{episode_id} {name} is not the source array")
    return True


def failure_mechanisms(rows):
    phases, links, outcomes = {}, {}, {}
    for row in rows:
        if row.get("status") != "evaluated":
            continue
        contact = row.get("first_contact") or {}
        if contact:
            phases[contact.get("phase")] = phases.get(contact.get("phase"), 0) + 1
            links[contact.get("link")] = links.get(contact.get("link"), 0) + 1
        result = row.get("result") or {}
        if not result.get("safe_success"):
            key = result.get("phase")
            outcomes[key] = outcomes.get(key, 0) + 1
    return {
        "first_contact_phase": phases,
        "first_contact_link": links,
        "unsafe_or_failed_phase": outcomes,
    }


def compare_exposure_eval(registration, output_dir):
    comparison = registration["comparison"]
    if comparison.get("rejected_proximity_is_not_baseline") is not True:
        raise ValueError("proximity result is not the baseline")
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    baselines = {}
    for name in ("obstacle100", "obstacle300"):
        rows = load_episode_directory(Path(comparison[f"{name}_paired"]) / "constrain_dp")
        require_baseline_counts(rows, comparison[name])
        if any(row.get("checkpoint_sha256") != comparison[f"{name}_sha256"] for row in rows):
            raise ValueError(f"{name} constrain checkpoint is not the frozen EMA")
        if any(row.get("checkpoint_sha256") == comparison["rejected_proximity_sha256"] for row in rows):
            raise ValueError("proximity checkpoint is not the baseline")
        baselines[name] = rows
    current = load_episode_directory(Path(output_dir) / "constrain_dp")
    errors = [row.get("scene_id") for row in current if row.get("status") != "evaluated"]
    transitions = None if errors else {
        name: expansion_transitions(rows, current, fields) for name, rows in baselines.items()
    }
    return {
        "denominator": 30,
        "formal_performance_claim": False,
        "rejected_proximity_is_not_baseline": True,
        "baseline_counts": {name: metric_counts(rows) for name, rows in baselines.items()},
        "exposure_counts": metric_counts(current),
        "transitions": transitions,
        "runtime_error_scene_ids": errors,
        "mechanisms": {
            "obstacle100": failure_mechanisms(baselines["obstacle100"]),
            "obstacle300": failure_mechanisms(baselines["obstacle300"]),
            "exposure": failure_mechanisms(current),
        },
    }


def compare_obstacle_increment(registration, output_dir):
    comparison = registration["comparison"]
    if comparison.get("rejected_proximity_is_not_baseline") is not True:
        raise ValueError("proximity result is not the baseline")
    root = Path(comparison["expansion_paired"])
    constrain = load_episode_directory(root / "constrain_dp")
    base = load_episode_directory(root / "base_dp")
    require_baseline_counts(constrain, comparison["constrain_counts"])
    require_baseline_counts(base, comparison["base_counts"])
    if any(row.get("checkpoint_sha256") == comparison["rejected_proximity_sha256"] for row in constrain):
        raise ValueError("proximity checkpoint is not the baseline")
    if any(row.get("checkpoint_sha256") != comparison["expansion_constrain_sha256"] for row in constrain):
        raise ValueError("baseline constrain checkpoint is not the expansion EMA")
    output_dir = Path(output_dir)
    current = {
        "base_dp": load_episode_directory(output_dir / "base_dp"),
        "constrain_dp": load_episode_directory(output_dir / "constrain_dp"),
    }
    authoritative = {"base_dp": base, "constrain_dp": constrain}
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    transitions = {}
    for name, rows in current.items():
        errors = [row.get("scene_id") for row in rows if row.get("status") != "evaluated"]
        transitions[name] = {
            "runtime_error_scene_ids": errors,
            "transitions": None if errors else expansion_transitions(authoritative[name], rows, fields),
        }
    return {
        "denominator": 30,
        "formal_performance_claim": False,
        "baseline": "expansion_100_obstacle",
        "rejected_proximity_is_not_baseline": True,
        "baseline_counts": {"base_dp": metric_counts(base), "constrain_dp": metric_counts(constrain)},
        "increment_counts": {name: metric_counts(rows) for name, rows in current.items()},
        "transitions": transitions,
        "mechanisms": {
            "baseline_constrain": failure_mechanisms(constrain),
            "increment_constrain": failure_mechanisms(current["constrain_dp"]),
            "increment_base": failure_mechanisms(current["base_dp"]),
        },
    }


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
