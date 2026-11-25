"""From-scratch official Diffusion Policy baseline for fixed-yaw peg insertion."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transforms3d.quaternions import quat2mat

from examples.baselines.constrain.base_dp import BasePolicyArgs, _PolicyEnvStub
from examples.baselines.constrain.peg_orientation_data import (
    PegSceneCatalog,
    restore_peg_scene,
)

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DP = ROOT / "examples/baselines/diffusion_policy"
DEMO_DIR = ROOT / "demos/PegInsertionSide-orientation-fixed-1000"
DATA = DEMO_DIR / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
DATA_JSON = DEMO_DIR / "trajectory.state.pd_ee_delta_pose.physx_cpu.json"
OBS_DIM, ACT_DIM = 39, 7
OBS_HORIZON, PRED_HORIZON, ACT_HORIZON = 2, 16, 8
EVAL_ENV_TIME_LIMIT = 1000
VALIDATION_CATALOG = ROOT / "outputs/exp/20261111_172604_peg_orientation_adaptation/validation_fixed_scenes"
FORMAT = "peg_orientation_fixed_base_dp_v1"


def resolved_official_peg_training():
    """Official Peg Args plus the baselines.sh Peg overrides.

    Method exceptions are explicit. This is not a claim that the run is the
    official 43d baseline.
    """
    if str(OFFICIAL_DP) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DP))
    from train import Args

    official = Args()
    return {
        "seed": official.seed,
        "batch_size": official.batch_size,
        "total_iters": 100_000,
        "lr": official.lr,
        "betas": (0.95, 0.999),
        "weight_decay": 1e-6,
        "warmup_steps": 500,
        "scheduler": "cosine",
        "ema_power": 0.75,
        "obs_horizon": official.obs_horizon,
        "act_horizon": official.act_horizon,
        "pred_horizon": official.pred_horizon,
        "unet_dims": list(official.unet_dims),
        "diffusion_step_embed_dim": official.diffusion_step_embed_dim,
        "n_groups": official.n_groups,
        "num_diffusion_iters": 100,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "clip_sample": True,
        "normalization": "identity",
        "num_dataload_workers": official.num_dataload_workers,
        "log_freq": official.log_freq,
        "max_episode_steps": 300,
        "env_time_limit": 300,
        "checkpoint_every": 5000,
        "control_mode": "pd_ee_delta_pose",
        "sim_backend": "physx_cpu",
        "method_exceptions": {
            "num_demos": 1000,
            "official_num_demos": 100,
            "observation_dim": 39,
            "official_observation_dim": 43,
            "evaluation": "frozen_validation_50",
            "checkpoint_policy": "save_every_5000_not_official_best_only",
            "selection": "max_native_success_tie_latest",
            "inference_seed_rule": "1 + scene_index",
            "task": "fixed_90deg_yaw_hidden_goal",
            "training_seed_history": "previous custom seed 2026111102; this run uses Args.seed 1",
            "eval_timing": "offline checkpoints at exact 5000..100000, not official eval after N+1 updates",
            "rng": "no official eval envs are constructed before training, so this is not a clone of the 43d RNG",
        },
    }


def official_iteration_batches(dataset_len, batch_size, total_iters):
    """Official RandomSampler, drop-last BatchSampler, and iteration wrapper."""
    if str(OFFICIAL_DP) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DP))
    from torch.utils.data import BatchSampler, RandomSampler
    from diffusion_policy.utils import IterationBasedBatchSampler

    class _Len:
        def __len__(self):
            return dataset_len

    sampler = RandomSampler(_Len(), replacement=False)
    batches = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    return IterationBasedBatchSampler(batches, total_iters)


def select_checkpoint(rows):
    """Highest native success count; ties take the latest update."""
    return max(rows, key=lambda row: (row["successes"], row["update"]))


def retention_authorized(successes, total=50, threshold=40):
    if total != 50 or threshold != 40:
        raise ValueError("retention authorization is the fixed 40/50 validation rule")
    return int(successes) >= threshold


def _require_official_budget(args):
    cfg = resolved_official_peg_training()
    if (args.updates != cfg["total_iters"] or args.seed != cfg["seed"]
            or args.batch_size != cfg["batch_size"]
            or args.checkpoint_every != cfg["checkpoint_every"]):
        raise ValueError(
            "training command must match the resolved official Peg budget "
            f"(updates={cfg['total_iters']}, seed={cfg['seed']}, "
            f"batch_size={cfg['batch_size']}, checkpoint_every={cfg['checkpoint_every']})"
        )
    return cfg


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PreinsertionDiagnostic:
    """Capture the first pre-action proximity sample and its terminal outcome."""

    def __init__(self, hole_center, target_yaw, trigger_distance_m=0.1):
        self.hole_center = np.asarray(hole_center)
        self.target_yaw = float(target_yaw)
        self.trigger_distance_m = float(trigger_distance_m)
        self.sample = None
        self.last_observation = None

    def observe(self, peg_position, peg_yaw, step_index):
        if self.sample is None:
            from examples.baselines.constrain.peg_orientation_data import preinsertion_yaw_diagnostic

            sample = preinsertion_yaw_diagnostic(
                peg_position, self.hole_center, peg_yaw, self.target_yaw,
                trigger_distance_m=self.trigger_distance_m, step_index=step_index,
            )
            self.last_observation = sample
            if sample["status"] == "sampled":
                self.sample = sample

    def finalize(self, native_success, failure_type):
        if self.sample is None:
            result = dict(self.last_observation or {
                "status": "not_reached", "distance_m": None,
                "trigger_distance_m": self.trigger_distance_m, "yaw_error_rad": None,
                "sample_step": None,
            })
            result.update(terminal=True, native_success=bool(native_success), failure_type=failure_type)
        else:
            result = dict(self.sample)
            result.update(terminal=True, native_success=bool(native_success), failure_type=failure_type)
        return result


class PegBaseDataset(Dataset):
    """Official DP episode windows over 39d hidden-target state and native 7d actions."""

    def __init__(self, data_path=DATA, manifest_path=DATA_JSON, expected_episodes=1000):
        episodes = json.loads(Path(manifest_path).read_text())["episodes"]
        episode_ids = [int(row["episode_id"]) for row in episodes]
        if len(episode_ids) != expected_episodes or sorted(episode_ids) != list(range(expected_episodes)):
            raise ValueError(f"fixed Peg training manifest must identify {expected_episodes} episodes")
        self.observations, self.actions, self.slices = [], [], []
        with h5py.File(data_path, "r") as handle:
            if len(handle) != expected_episodes:
                raise ValueError(f"fixed Peg training HDF5 must contain {expected_episodes} episodes")
            for episode_id in episode_ids:
                group = handle[f"traj_{episode_id}"]
                obs, actions = group["obs"][:], group["actions"][:]
                if obs.shape != (len(actions) + 1, OBS_DIM) or actions.ndim != 2 or actions.shape[1] != ACT_DIM:
                    raise ValueError(f"traj_{episode_id}: expected T+1 39d observations and T 7d actions")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.00001:
                    raise ValueError(f"traj_{episode_id}: invalid observations or native actions")
                self.observations.append(torch.as_tensor(obs, dtype=torch.float32))
                self.actions.append(torch.as_tensor(actions, dtype=torch.float32))
                length = len(actions)
                for start in range(-(OBS_HORIZON - 1), length - PRED_HORIZON + (PRED_HORIZON - OBS_HORIZON)):
                    self.slices.append((episode_id, start, start + PRED_HORIZON))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        episode, start, end = self.slices[index]
        obs, actions = self.observations[episode], self.actions[episode]
        length = len(actions)
        obs_seq = obs[max(0, start):start + OBS_HORIZON]
        action_seq = actions[max(0, start):end]
        if start < 0:
            obs_seq = torch.cat((obs_seq[0].repeat(-start, 1), obs_seq))
            action_seq = torch.cat((action_seq[0].repeat(-start, 1), action_seq))
        if end > length:
            pad = torch.cat((torch.zeros(ACT_DIM - 1), action_seq[-1, -1:]))
            action_seq = torch.cat((action_seq, pad.repeat(end - length, 1)))
        return {"observations": obs_seq, "actions": action_seq}


def build_agent(device="cpu"):
    if str(OFFICIAL_DP) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DP))
    from train import Agent

    args = BasePolicyArgs(OBS_DIM, ACT_DIM, OBS_HORIZON, ACT_HORIZON, PRED_HORIZON)
    args.unet_dims = [64, 128, 256]
    return Agent(_PolicyEnvStub(OBS_DIM, ACT_DIM, OBS_HORIZON), args).to(device)


class PegBaseAdapter:
    def __init__(self, checkpoint, device="cpu"):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        if ck.get("format") != FORMAT or ck.get("observation_dim") != OBS_DIM or ck.get("action_dim") != ACT_DIM:
            raise ValueError("Peg BaseDP checkpoint contract mismatch")
        self.agent = build_agent(device)
        self.agent.load_state_dict(ck["ema_agent"], strict=True)
        self.agent.eval()
        self.device = torch.device(device)
        self.metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                         "weights": "ema_agent", "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
                         "obs_horizon": OBS_HORIZON, "pred_horizon": PRED_HORIZON,
                         "act_horizon": ACT_HORIZON}

    def get_action(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if observations.shape[-2:] != (OBS_HORIZON, OBS_DIM):
            raise ValueError(f"expected (..., 2, 39) observations, got {tuple(observations.shape)}")
        with torch.no_grad():
            return self.agent.get_action(observations)


def observation_peg_and_hole(flat):
    """Peg XYZ and hole center from the 39d state layout.

    Extra fields are tcp pose, peg pose, peg half-size, hole radius, then hole center.
    The radius occupies index 35, so the hole starts at 36.
    """
    flat = np.asarray(flat)
    return flat[..., 25:28], flat[..., 36:39]


def _observation_xy(flat):
    peg, hole = observation_peg_and_hole(flat)
    return float(np.linalg.norm(peg[:2] - hole[:2]))


def _float_list(value, count):
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return [float(item) for item in np.asarray(array, dtype=np.float64).reshape(-1)[:count]]


def _stage_geometry(task):
    """Descriptive native geometry snapshot; does not define a success threshold."""
    peg_pose = _float_list(task.peg.pose.raw_pose, 7)
    hole_pose = _float_list(task.box_hole_pose.raw_pose, 7)
    peg_axis = quat2mat(peg_pose[3:7])[:, 0]
    hole_axis = quat2mat(hole_pose[3:7])[:, 0]
    peg_yaw = float(np.arctan2(peg_axis[1], peg_axis[0]))
    hole_yaw = float(np.arctan2(hole_axis[1], hole_axis[0]))
    inserted, head = task.has_peg_inserted()
    return {
        "peg_center_to_hole_center_distance_m": float(np.linalg.norm(
            np.asarray(peg_pose[:3]) - np.asarray(hole_pose[:3])
        )),
        "official_peg_head_in_hole_m": _float_list(head, 3),
        "official_insertion": bool(np.asarray(inserted.detach().cpu()).reshape(-1)[0]),
        "peg_yaw_rad": peg_yaw,
        "hole_yaw_rad": hole_yaw,
        "wrapped_peg_plus_x_minus_box_plus_x_yaw_rad": float(
            np.arctan2(np.sin(peg_yaw - hole_yaw), np.cos(peg_yaw - hole_yaw))
        ),
    }


def commanded_translation_in_hole(action_xyz, root_quat, hole_quat):
    """Rotate a root-frame translation command into the official hole frame.

    Panda ``pd_ee_delta_pose`` applies action[:3] in the robot root frame.
    Hole-frame +X is the insertion axis used by ``has_peg_inserted``.
    """
    world = quat2mat(np.asarray(root_quat, dtype=np.float64)) @ np.asarray(action_xyz, dtype=np.float64)
    return quat2mat(np.asarray(hole_quat, dtype=np.float64)).T @ world


def normalized_root_translation_for_hole_plus_x(hole_quat, root_quat, pos_lower, pos_upper, meters):
    """Map a physical hole-+X delta into an unclipped normalized root translation.

    The controller applies ``clip_and_scale`` to action[:3] and adds that delta in
    the robot root frame. This is the inverse of that scale plus
    ``commanded_translation_in_hole``. It raises instead of clipping.
    """
    hole_delta = np.array([float(meters), 0.0, 0.0], dtype=np.float64)
    world = quat2mat(np.asarray(hole_quat, dtype=np.float64)) @ hole_delta
    root_delta = quat2mat(np.asarray(root_quat, dtype=np.float64)).T @ world
    low, high = float(pos_lower), float(pos_upper)
    scale = 0.5 * (high - low)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("controller position bounds cannot scale a translation")
    normalized = (root_delta - 0.5 * (high + low)) / scale
    if np.any(np.abs(normalized) > 1.0 + 1e-8):
        raise ValueError("hole +X request exceeds controller bounds and would be clipped")
    physical = 0.5 * (high + low) + scale * normalized
    recovered = commanded_translation_in_hole(physical, root_quat, hole_quat)
    if not np.allclose(recovered, hole_delta, atol=1e-8):
        raise ValueError("hole to root translation did not round-trip")
    return normalized, physical


OBSERVATION_SLICES = {
    "qpos": (0, 9),
    "qvel": (9, 18),
    "tcp_pose": (18, 25),
    "peg_pose": (25, 32),
    "peg_half_size": (32, 35),
    "hole_radius": (35, 36),
    "hole_center": (36, 39),
}


def _rng_snapshot(device):
    snapshot = {"torch_cpu": [int(item) for item in torch.get_rng_state().tolist()]}
    if torch.device(device).type == "cuda":
        snapshot["torch_cuda"] = [int(item) for item in torch.cuda.get_rng_state().tolist()]
        snapshot["cuda_device_index"] = int(torch.cuda.current_device())
    return snapshot


def _restore_rng(snapshot, device):
    torch.set_rng_state(torch.tensor(snapshot["torch_cpu"], dtype=torch.uint8))
    if torch.device(device).type != "cuda":
        return
    if "torch_cuda" not in snapshot:
        raise ValueError("cuda sampler call has no saved cuda rng state")
    if int(snapshot["cuda_device_index"]) != int(torch.cuda.current_device()):
        raise ValueError("saved cuda rng device does not match the current device")
    torch.cuda.set_rng_state(torch.tensor(snapshot["torch_cuda"], dtype=torch.uint8))


def _native_observation_fields(task):
    hole_pose = _float_list(task.box_hole_pose.raw_pose, 7)
    return {
        "qpos": _float_list(task.agent.robot.get_qpos(), 9),
        "qvel": _float_list(task.agent.robot.get_qvel(), 9),
        "tcp_pose": _float_list(task.agent.tcp.pose.raw_pose, 7),
        "peg_pose": _float_list(task.peg.pose.raw_pose, 7),
        "peg_half_size": _float_list(task.peg_half_sizes, 3),
        "hole_radius": _float_list(task.box_hole_radii, 1),
        "hole_center": hole_pose[:3],
        "box_hole_pose": hole_pose,
    }


def _history_consistency(history, native):
    current = np.asarray(history[1], dtype=np.float64)
    prior = np.asarray(history[0], dtype=np.float64)
    errors = {}
    for name, (start, end) in OBSERVATION_SLICES.items():
        errors[name] = float(np.max(np.abs(current[start:end] - np.asarray(native[name], dtype=np.float64))))
    return errors, float(np.max(np.abs(prior - current)))


def reproduce_sampler_chunk(get_action, capture, device):
    """Restore the pre-call RNG and call the same sampler once. No environment step."""
    _restore_rng(capture["rng"], device)
    history = np.asarray(capture["history_39d"], dtype=np.float32)
    planned = get_action(torch.as_tensor(history[None], device=device))[0]
    length = int(capture["executed_chunk_length"])
    got = np.asarray(planned[:length].detach().cpu().numpy(), dtype=np.float32)
    expected = np.asarray(capture["executed_chunk_7d"], dtype=np.float32)
    error = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
    return {
        "preaction_step_index": capture["preaction_step_index"],
        "first_completed_control": capture["first_completed_control"],
        "last_completed_control": capture["last_completed_control"],
        "max_abs_error": error,
        "matched": error == 0.0,
    }


def _controller_frame(task):
    config = task.agent.controller.controllers["arm"].config
    return {
        "frame": config.frame,
        "use_delta": bool(config.use_delta),
        "use_target": bool(config.use_target),
        "pos_lower": float(np.asarray(config.pos_lower).reshape(-1)[0]),
        "pos_upper": float(np.asarray(config.pos_upper).reshape(-1)[0]),
        "rot_lower": float(np.asarray(config.rot_lower).reshape(-1)[0]),
    }


def _max_abs_error(live, expected):
    return float(np.max(np.abs(np.asarray(live, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))


def _intervention_branch_match(task, spec, hole_center, last_action, initial_error):
    tolerance = float(spec["tolerance"])
    peg = _float_list(task.peg.pose.raw_pose, 7)
    tcp = _float_list(task.agent.tcp.pose.raw_pose, 7)
    hole = _float_list(task.box_hole_pose.raw_pose, 7)
    root = _float_list(task.agent.robot.pose.raw_pose, 7)
    _inserted, head = task.has_peg_inserted()
    head = _float_list(head, 3)
    peg_position = np.asarray(task.peg.pose.p[0].detach().cpu().numpy(), dtype=np.float64)
    pose_xy = float(np.linalg.norm(peg_position[:2] - np.asarray(hole_center, dtype=np.float64)[:2]))
    grasping = bool(task.agent.is_grasping(task.peg).detach().cpu().reshape(-1)[0])
    expected = spec["match"]
    errors = {
        "peg_pose": _max_abs_error(peg, expected["peg_pose"]),
        "tcp_pose": _max_abs_error(tcp, expected["tcp_pose"]),
        "hole_pose": _max_abs_error(hole, expected["hole_pose"]),
        "root_pose": _max_abs_error(root, expected["root_pose"]),
        "peg_head_in_hole_m": _max_abs_error(head, expected["peg_head_in_hole_m"]),
        "pose_xy_m": abs(pose_xy - float(expected["pose_xy_m"])),
        "prefix_action_7d": _max_abs_error(last_action, spec["prefix_action_7d"]),
        "initial_observation": None if initial_error is None else float(initial_error),
        "grasping": grasping,
    }
    pose_ok = all(errors[key] <= tolerance for key in (
        "peg_pose", "tcp_pose", "hole_pose", "root_pose", "peg_head_in_hole_m",
        "pose_xy_m", "prefix_action_7d",
    ))
    observation_ok = initial_error is not None and float(initial_error) <= float(spec["max_initial_observation_error"])
    return {"matched": bool(pose_ok and observation_ok and grasping == bool(expected["grasping"])), "errors": errors}


def _hole_plus_x_replacement(task, spec, index):
    frame = _controller_frame(task)
    if (frame["frame"] != "root_translation:root_aligned_body_rotation"
            or not frame["use_delta"] or frame["use_target"]):
        raise ValueError("controller frame cannot map a hole +X translation")
    root_pose = _float_list(task.agent.robot.pose.raw_pose, 7)
    normalized, physical = normalized_root_translation_for_hole_plus_x(
        spec["frozen_hole_quat"], root_pose[3:7], frame["pos_lower"], frame["pos_upper"],
        spec["physical_hole_plus_x_m"],
    )
    baseline = np.asarray(spec["baseline_actions_7d"][index], dtype=np.float64)
    action = baseline.copy()
    action[:3] = normalized
    return action, {
        "baseline_action_7d": [float(item) for item in baseline],
        "requested_physical_root_delta_m": [float(item) for item in physical],
        "requested_physical_hole_plus_x_m": float(spec["physical_hole_plus_x_m"]),
        "frozen_hole_quat": [float(item) for item in spec["frozen_hole_quat"]],
        "live_root_quat": root_pose[3:7],
        "live_hole_pose": _float_list(task.box_hole_pose.raw_pose, 7),
        "preaction_tcp_pose": _float_list(task.agent.tcp.pose.raw_pose, 7),
        "preaction_peg_pose": _float_list(task.peg.pose.raw_pose, 7),
        "clipped": False,
        "pos_lower": frame["pos_lower"],
        "pos_upper": frame["pos_upper"],
    }


def evaluate(checkpoint, catalog_dir, output_dir, device="cuda", video=False,
             max_episode_steps=100, seed=1907, scene_indices=None, record_phase_trace=False,
             insertion_trace_window=None, translation_intervention=None, env_time_limit=None,
             policy_factory=None, hint_mode="correct", hint_assignment=None,
             record_stage_geometry=False, video_scene_indices=None):
    import gymnasium as gym
    import imageio.v2 as imageio
    from examples.baselines.constrain import peg_orientation
    record_phase_trace = record_phase_trace or record_stage_geometry

    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    if env_time_limit is None:
        env_time_limit = EVAL_ENV_TIME_LIMIT
    env_time_limit = int(env_time_limit)
    if insertion_trace_window is not None:
        start, end = int(insertion_trace_window[0]), int(insertion_trace_window[1])
        if not record_phase_trace or start < 1 or end < start:
            raise ValueError("insertion_trace_window requires a phase trace and an inclusive 1-based range")
        insertion_trace_window = (start, end)
    catalog = PegSceneCatalog(catalog_dir)
    if hint_mode not in ("correct", "fixed_standard", "shuffled"):
        raise ValueError(f"unknown hint mode: {hint_mode}")
    if hint_mode == "shuffled" and (hint_assignment is None
                                      or len(hint_assignment.get("target_yaws", [])) != len(catalog)):
        raise ValueError("shuffled evaluation requires one frozen pseudo-target per catalog scene")
    assignment_sha256 = None
    if hint_assignment is not None:
        assignment_sha256 = hashlib.sha256(json.dumps(
            hint_assignment, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
    if scene_indices is None:
        selected = list(range(len(catalog)))
    else:
        selected = [int(index) for index in scene_indices]
        if (not selected or len(selected) != len(set(selected))
                or any(index < 0 or index >= len(catalog) for index in selected)):
            raise ValueError("scene_indices must be unique in-range catalog indices")
    video_indices = None if video_scene_indices is None else {int(index) for index in video_scene_indices}
    if video_indices is not None and any(index < 0 or index >= len(catalog) for index in video_indices):
        raise ValueError("video_scene_indices must be in-range catalog indices")
    env = gym.make(peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", reconfiguration_freq=1, render_mode="rgb_array",
                    max_episode_steps=env_time_limit)
    writer = None
    try:
        adapter = policy_factory(env.unwrapped) if policy_factory else PegBaseAdapter(checkpoint, device)
        evaluation_identity = dict(getattr(adapter, "metadata", {}))
        evaluation_identity.update({"hint_mode": hint_mode,
                                    "hint_assignment_sha256": assignment_sha256})
        catalog_manifest = Path(catalog_dir) / "manifest.json"
        evaluation_identity["catalog_manifest_sha256"] = (
            sha256_file(catalog_manifest) if catalog_manifest.is_file() else None
        )
        evaluation_identity["evaluator_source_sha256"] = sha256_file(__file__)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        records = []
        if video:
            writer = imageio.get_writer(str(output_dir / "rollouts.mp4"), fps=20, codec="libx264")
        for index in selected:
            capture_video = writer is not None and (video_indices is None or index in video_indices)
            scene = catalog.load(index)
            restored = restore_peg_scene(env, scene).detach().cpu().numpy().reshape(-1)
            pseudo_target = (hint_assignment["target_yaws"][index]
                             if hint_mode == "shuffled" else None)
            if hasattr(adapter, "set_episode_context"):
                adapter.set_episode_context(index, scene[0], pseudo_target)
            obs = restored
            scene_seed = int(seed) + index
            torch.manual_seed(scene_seed)
            torch.cuda.manual_seed_all(scene_seed)
            history = [obs.copy(), obs.copy()]
            if capture_video:
                writer.append_data(_video_frame(env))
            meta = scene[0]
            diagnostic = PreinsertionDiagnostic(meta["hole_center_m"], meta["target_yaw_rad"])
            success = False
            failure_type = "max_episode_steps"
            step_index = 0
            first_success_control = None
            phase_trace = []
            sampler_chunks = []
            sampler_input_window = []
            initial_flat = obs.copy()
            initial_error = None
            if record_phase_trace and scene[2] is not None:
                initial_error = float(np.max(np.abs(obs - np.asarray(scene[2]).reshape(-1))))
            executed_chunk_lengths = []
            ever_grasped = False
            previous_grasped = False
            contact_losses = 0
            grasped_controls = 0
            first_grasp_control = None
            first_pose_xy = None
            first_obs_xy = None
            min_pose_xy = None
            min_obs_xy = _observation_xy(initial_flat) if record_phase_trace else None
            max_source_diff = 0.0
            peak_lift = 0.0
            peak_grasped_lift = 0.0
            hole_center = np.asarray(meta["hole_center_m"], dtype=np.float64)
            controller_frame = _controller_frame(env.unwrapped) if insertion_trace_window else None
            intervention = translation_intervention
            intervention_used = 0
            last_executed_action = None
            pending_intervention = None
            intervention_report = None
            if intervention is not None:
                intervention_report = {
                    "prefix_matched": None,
                    "match_errors": None,
                    "intervention_controls_executed": 0,
                    "stopped_on_official_success": False,
                    "stopped_for_mismatch": False,
                    "transform_rejected": None,
                    "success_label": None,
                    "counts_as_policy_success": False,
                }
            while step_index < max_episode_steps:
                if intervention is not None and step_index >= intervention["prefix_controls"]:
                    if intervention_report["prefix_matched"] is None:
                        match = _intervention_branch_match(
                            env.unwrapped, intervention, hole_center, last_executed_action, initial_error,
                        )
                        intervention_report["prefix_matched"] = match["matched"]
                        intervention_report["match_errors"] = match["errors"]
                        if not match["matched"]:
                            intervention_report["stopped_for_mismatch"] = True
                            failure_type = "prefix_mismatch"
                            break
                    if intervention_used >= intervention["count"] or success:
                        break
                    try:
                        action_np, pending_intervention = _hole_plus_x_replacement(
                            env.unwrapped, intervention, intervention_used,
                        )
                    except ValueError as err:
                        intervention_report["transform_rejected"] = str(err)
                        failure_type = "transform_rejected"
                        break
                    chunk = [torch.tensor(action_np, dtype=torch.float32)]
                else:
                    pending_intervention = None
                    first_control = step_index + 1
                    remaining = max_episode_steps - step_index
                    scheduled_last = step_index + min(ACT_HORIZON, remaining)
                    capture_call = (
                        insertion_trace_window is not None
                        and first_control <= insertion_trace_window[1]
                        and scheduled_last >= insertion_trace_window[0]
                    )
                    history_copy = native = head_before = rng_before = None
                    if capture_call:
                        history_copy = np.asarray(history, dtype=np.float32).copy()
                        native = _native_observation_fields(env.unwrapped)
                        inserted, head = env.unwrapped.has_peg_inserted()
                        head_before = (
                            bool(np.asarray(inserted.detach().cpu()).reshape(-1)[0]),
                            _float_list(head, 3),
                        )
                        rng_before = _rng_snapshot(device)
                    planned = adapter.get_action(torch.as_tensor(np.asarray(history)[None], device=device))[0]
                    if intervention is not None:
                        remaining = min(remaining, intervention["prefix_controls"] - step_index)
                    chunk = planned[:remaining]
                    if record_phase_trace:
                        sampler_chunks.append(int(chunk.shape[0]))
                    actual_last = step_index + int(chunk.shape[0])
                    if (capture_call and chunk.shape[0] and first_control <= insertion_trace_window[1]
                            and actual_last >= insertion_trace_window[0]):
                        errors, prior_gap = _history_consistency(history_copy, native)
                        sampler_input_window.append({
                            "scene_index": index,
                            "catalog_seed": int(scene[0]["seed"]),
                            "policy_sampling_seed": scene_seed,
                            "preaction_step_index": step_index,
                            "first_completed_control": first_control,
                            "last_completed_control": actual_last,
                            "executed_chunk_length": int(chunk.shape[0]),
                            "returned_action_count": int(planned.shape[0]),
                            "history_39d": history_copy.tolist(),
                            "executed_chunk_7d": np.asarray(
                                chunk.detach().cpu().numpy(), dtype=np.float32
                            ).reshape(int(chunk.shape[0]), 7).tolist(),
                            "rng": rng_before,
                            "device": str(device),
                            "official_peg_head_in_hole_m": head_before[1],
                            "official_insertion": head_before[0],
                            "box_hole_pose": native["box_hole_pose"],
                            "box_hole_radius_m": native["hole_radius"][0],
                            "history_layout": {
                                "shape": [2, 39],
                                "prior_index": 0,
                                "current_index": 1,
                                "semantics": (
                                    "index 0 is the previous observation, repeated from the initial "
                                    "observation on the first call; index 1 is the current observation"
                                ),
                                "fields": {name: [start, end] for name, (start, end) in OBSERVATION_SLICES.items()},
                            },
                            "current_vs_native_max_abs": errors,
                            "prior_vs_current_max_abs": prior_gap,
                        })
                completed_chunk_controls = 0
                for action in chunk:
                    last_executed_action = [
                        float(item) for item in np.asarray(action.detach().cpu()).reshape(-1)[:7]
                    ]
                    task = env.unwrapped
                    peg_position = task.peg.pose.p[0].detach().cpu().numpy()
                    peg_axis = quat2mat(task.peg.pose.q[0].detach().cpu().numpy())[:, 0]
                    peg_yaw = float(np.arctan2(peg_axis[1], peg_axis[0]))
                    if record_phase_trace:
                        pose_xy = float(np.linalg.norm(np.asarray(peg_position)[:2] - hole_center[:2]))
                        obs_xy = _observation_xy(history[-1])
                        max_source_diff = max(max_source_diff, abs(pose_xy - obs_xy))
                        min_pose_xy = pose_xy if min_pose_xy is None else min(min_pose_xy, pose_xy)
                        min_obs_xy = min(min_obs_xy, obs_xy)
                        if first_pose_xy is None and pose_xy <= 0.1:
                            first_pose_xy = step_index
                        if first_obs_xy is None and obs_xy <= 0.1:
                            first_obs_xy = step_index
                    diagnostic.observe(peg_position, peg_yaw, step_index)
                    completed_control = step_index + 1
                    window_pre = None
                    if (insertion_trace_window is not None
                            and insertion_trace_window[0] <= completed_control <= insertion_trace_window[1]):
                        inserted, head = task.has_peg_inserted()
                        action_7d = np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)
                        tcp_pose = _float_list(task.agent.tcp.pose.raw_pose, 7)
                        peg_pose = _float_list(task.peg.pose.raw_pose, 7)
                        hole_pose = _float_list(task.box_hole_pose.raw_pose, 7)
                        root_pose = _float_list(task.agent.robot.pose.raw_pose, 7)
                        window_pre = {
                            "executed_action_7d": [float(item) for item in action_7d[:7]],
                            "action_translation_l2": float(np.linalg.norm(action_7d[:3])),
                            "action_rotation_l2": float(np.linalg.norm(action_7d[3:6])),
                            "preaction_peg_head_in_hole_m": _float_list(head, 3),
                            "preaction_official_insertion": bool(np.asarray(
                                inserted.detach().cpu()).reshape(-1)[0]),
                            "preaction_tcp_pose": tcp_pose,
                            "preaction_peg_pose": peg_pose,
                            "preaction_box_hole_pose": hole_pose,
                            "preaction_root_pose": root_pose,
                            "commanded_translation_in_hole": [
                                float(item) for item in commanded_translation_in_hole(
                                    action_7d[:3], root_pose[3:7], hole_pose[3:7])
                            ],
                            "box_hole_radius_m": float(np.asarray(
                                task.box_hole_radii.detach().cpu()).reshape(-1)[0]),
                        }
                    stage_geometry_before = _stage_geometry(task) if record_stage_geometry else None
                    obs, _, terminated, truncated, info = env.step(action.detach().cpu().numpy())
                    completed_chunk_controls += 1
                    flat = obs.detach().cpu().numpy().reshape(-1)
                    history = [history[-1], flat]
                    success = success or bool(info.get("success", False))
                    step_index += 1
                    if pending_intervention is not None:
                        intervention_used += 1
                    if success and first_success_control is None:
                        first_success_control = step_index
                    if record_phase_trace:
                        grasp_value = task.agent.is_grasping(task.peg)
                        is_grasping = bool(grasp_value.detach().cpu().reshape(-1)[0])
                        contact_lost = previous_grasped and not is_grasping
                        if is_grasping and first_grasp_control is None:
                            first_grasp_control = step_index
                        if contact_lost:
                            contact_losses += 1
                        ever_grasped = ever_grasped or is_grasping
                        grasped_controls += int(is_grasping)
                        previous_grasped = is_grasping
                        post_peg, post_hole = observation_peg_and_hole(flat)
                        post_xy = float(np.linalg.norm(post_peg[:2] - post_hole[:2]))
                        min_obs_xy = min(min_obs_xy, post_xy)
                        lift = float(post_peg[2] - observation_peg_and_hole(initial_flat)[0][2])
                        peak_lift = max(peak_lift, lift)
                        if is_grasping:
                            peak_grasped_lift = max(peak_grasped_lift, lift)
                        phase_trace.append({
                            "control": step_index,
                            "preaction_control_index": step_index - 1,
                            "is_grasping": is_grasping,
                            "contact_lost": bool(contact_lost),
                            "preaction_pose_peg_hole_xy_m": pose_xy,
                            "preaction_observation_peg_hole_xy_m": obs_xy,
                            "preaction_xy_source_abs_difference_m": abs(pose_xy - obs_xy),
                            "observation_peg_hole_xy_m": post_xy,
                            "observation_peg_hole_center_distance_m": float(np.linalg.norm(post_peg - post_hole)),
                            "observation_tcp_to_peg_distance_m": float(np.linalg.norm(flat[18:21] - flat[25:28])),
                            "observation_peg_lift_m": lift,
                            "gripper_action": float(np.asarray(action.detach().cpu()).reshape(-1)[-1]),
                            "native_success": success,
                        })
                        if stage_geometry_before is not None:
                            phase_trace[-1]["stage_geometry_preaction"] = stage_geometry_before
                            phase_trace[-1]["stage_geometry_postaction"] = _stage_geometry(task)
                        if window_pre is not None:
                            post = info.get("peg_head_pos_at_hole")
                            post_list = None if post is None else _float_list(post, 3)
                            post_tcp = _float_list(task.agent.tcp.pose.raw_pose, 7)
                            post_peg = _float_list(task.peg.pose.raw_pose, 7)
                            post_root = _float_list(task.agent.robot.pose.raw_pose, 7)
                            phase_trace[-1].update(window_pre)
                            phase_trace[-1]["postaction_official_peg_head_in_hole_m"] = post_list
                            phase_trace[-1]["postaction_peg_head_source"] = None if post is None else "info"
                            phase_trace[-1]["postaction_box_hole_pose"] = _float_list(task.box_hole_pose.raw_pose, 7)
                            phase_trace[-1]["postaction_tcp_pose"] = post_tcp
                            phase_trace[-1]["postaction_peg_pose"] = post_peg
                            phase_trace[-1]["postaction_root_pose"] = post_root
                            phase_trace[-1]["tcp_displacement_m"] = [
                                float(post_tcp[axis] - window_pre["preaction_tcp_pose"][axis])
                                for axis in range(3)
                            ]
                            phase_trace[-1]["peg_center_displacement_m"] = [
                                float(post_peg[axis] - window_pre["preaction_peg_pose"][axis])
                                for axis in range(3)
                            ]
                            phase_trace[-1]["root_displacement_m"] = [
                                float(post_root[axis] - window_pre["preaction_root_pose"][axis])
                                for axis in range(3)
                            ]
                            phase_trace[-1]["peg_head_displacement_in_hole_m"] = None if post_list is None else [
                                float(post_list[axis] - window_pre["preaction_peg_head_in_hole_m"][axis])
                                for axis in range(3)
                            ]
                        if pending_intervention is not None:
                            post_tcp = _float_list(task.agent.tcp.pose.raw_pose, 7)
                            post_peg = _float_list(task.peg.pose.raw_pose, 7)
                            phase_trace[-1].update(pending_intervention)
                            phase_trace[-1]["translation_replaced"] = True
                            phase_trace[-1]["executed_action_7d"] = last_executed_action
                            phase_trace[-1]["postaction_tcp_pose"] = post_tcp
                            phase_trace[-1]["postaction_peg_pose"] = post_peg
                            phase_trace[-1]["tcp_displacement_m"] = [
                                float(post_tcp[axis] - pending_intervention["preaction_tcp_pose"][axis])
                                for axis in range(3)
                            ]
                            phase_trace[-1]["peg_center_displacement_m"] = [
                                float(post_peg[axis] - pending_intervention["preaction_peg_pose"][axis])
                                for axis in range(3)
                            ]
                    if capture_video and not (terminated or truncated):
                        writer.append_data(_video_frame(env))
                    if terminated or truncated:
                        failure_type = ("success" if success else "max_episode_steps"
                                        if truncated and step_index >= max_episode_steps
                                        else str(info.get("fail", "native_termination")))
                        break
                executed_chunk_lengths.append(completed_chunk_controls)
                if terminated or truncated:
                    break
            if intervention_report is not None:
                intervention_report["intervention_controls_executed"] = intervention_used
                if success and intervention_used:
                    intervention_report["stopped_on_official_success"] = True
                    intervention_report["success_label"] = "scripted_diagnostic_not_model_performance"
                elif (not success and intervention_used >= intervention["count"]
                        and failure_type == "max_episode_steps"):
                    failure_type = "intervention_complete"
            diagnostic = diagnostic.finalize(success, failure_type)
            record = {"scene_index": index, "seed": int(scene[0]["seed"]),
                      "policy_sampling_seed": scene_seed, "executed_controls": step_index,
                      "executed_chunk_lengths": executed_chunk_lengths,
                      "first_success_control": first_success_control, "success": success,
                      "failure_type": failure_type, "preinsertion": diagnostic}
            record["evaluation_identity"] = evaluation_identity
            if hasattr(adapter, "hint_samples"):
                record["hint_samples"] = list(adapter.hint_samples)
            if intervention_report is not None:
                record["translation_intervention"] = intervention_report
            if record_phase_trace:
                record["phase"] = {
                    "ever_grasped": ever_grasped,
                    "first_grasp_control": first_grasp_control,
                    "contact_losses": contact_losses,
                    "grasped_controls": grasped_controls,
                    "first_pose_xy_proximity_control": first_pose_xy,
                    "first_observation_xy_proximity_control": first_obs_xy,
                    "minimum_pose_peg_hole_xy_m": min_pose_xy,
                    "minimum_observation_peg_hole_xy_m": min_obs_xy,
                    "observation_xy_reduction_m": (
                        None if min_obs_xy is None else float(_observation_xy(initial_flat) - min_obs_xy)
                    ),
                    "max_preaction_xy_source_abs_difference_m": max_source_diff,
                    "peak_observation_peg_lift_m": peak_lift,
                    "peak_grasped_lift_m": peak_grasped_lift,
                    "sampler_calls": len(sampler_chunks),
                    "actions_per_sampler_call": sampler_chunks,
                    "initial_observation_max_abs_error": initial_error,
                    "xy_le_0.1_meaning": "proximity_trigger_not_official_success_or_axis_alignment",
                }
                if insertion_trace_window is not None:
                    record["phase"]["insertion_trace_window"] = list(insertion_trace_window)
                    record["phase"]["controller_frame"] = controller_frame
                    record["phase"]["sampler_input_calls"] = len(sampler_input_window)
                    record["phase"]["sampler_input_boundaries"] = [
                        [row["first_completed_control"], row["last_completed_control"]]
                        for row in sampler_input_window
                    ]
                    record["sampler_input_window"] = sampler_input_window
                record["phase_trace"] = phase_trace
            if record_stage_geometry:
                record["stage_geometry_terminal"] = _stage_geometry(env.unwrapped)
            records.append(record)
            (output_dir / f"episode_{index:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
    finally:
        if writer is not None:
            writer.close()
        env.close()
    successes = sum(row["success"] for row in records)
    summary = {"episodes": len(records), "successes": successes, "success_rate": successes / len(records),
               "failure_types": {kind: sum(row["failure_type"] == kind for row in records)
                                  for kind in sorted({row["failure_type"] for row in records})},
               "checkpoint": adapter.metadata, "scene_catalog": str(catalog_dir),
                "evaluation_settings": {"max_episode_steps": int(max_episode_steps),
                                        "env_time_limit": int(env_time_limit),
                                        "policy_sampling_seed": int(seed),
                                        "per_scene_policy_seed": "base_seed + scene_index"},
               "records": records}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _video_frame(env):
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    return np.asarray(frame, dtype=np.uint8)


def _smoke_contract(agent, batch, device, catalog_dir, dataset_windows):
    import gymnasium as gym
    from examples.baselines.constrain import peg_orientation

    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    if observations.shape[1:] != (OBS_HORIZON, OBS_DIM) or actions.shape[1:] != (PRED_HORIZON, ACT_DIM):
        raise ValueError("public data window is not the official 2/16 horizon on 39d/7d")
    loss = agent.compute_loss(observations, actions)
    loss.backward()
    if not torch.isfinite(loss) or any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in agent.parameters()
    ):
        raise ValueError("training smoke produced non-finite loss or gradients")
    with torch.no_grad():
        predicted = agent.get_action(observations[:1])
    if predicted.shape[-2:] != (ACT_HORIZON, ACT_DIM) or not torch.isfinite(predicted).all():
        raise ValueError("training smoke produced non-finite actions")
    env = gym.make(
        peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu", max_episode_steps=300,
    )
    try:
        scene = PegSceneCatalog(catalog_dir).load(0)
        restored = restore_peg_scene(env, scene).detach().cpu().numpy().reshape(-1)
        if restored.shape != (OBS_DIM,):
            raise ValueError(f"restored observation is {restored.shape}, expected 39d")
        if abs(float(scene[0]["target_yaw_rad"]) - (np.pi / 2)) > 1e-5:
            raise ValueError("restored scene is not the fixed 90 degree target")
        space = env.action_space
        low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        if space.shape[-1] != ACT_DIM or not np.allclose(low[:ACT_DIM], -1) or not np.allclose(high[:ACT_DIM], 1):
            raise ValueError(f"controller action space is not native 7d [-1, 1]: {space}")
        env.step(np.zeros(space.shape, dtype=np.float32))
    finally:
        env.close()
    return {
        "smoke_only": True, "optimizer_updates": 0, "loss": float(loss.detach().cpu()),
        "action_shape": list(predicted.shape), "device": str(device),
        "dataset_windows": dataset_windows, "restored_observation_dim": int(restored.shape[0]),
        "action_dim": ACT_DIM, "control_mode": "pd_ee_delta_pose",
        "contract_step": "one_zero_action_not_a_scored_evaluation",
    }


def train(args):
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from torch.utils.tensorboard import SummaryWriter

    cfg = _require_official_budget(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(OFFICIAL_DP) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DP))
    from diffusion_policy.utils import worker_init_fn
    import train as official_train
    official_train.device = device
    dataset = PegBaseDataset(args.data, args.manifest)
    loader = DataLoader(
        dataset,
        batch_sampler=official_iteration_batches(len(dataset), args.batch_size, args.updates),
        num_workers=cfg["num_dataload_workers"],
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    agent = build_agent(device)
    if args.smoke_only:
        print(json.dumps(_smoke_contract(
            agent, next(iter(loader)), device, args.smoke_catalog, len(dataset),
        )))
        return
    ema = EMAModel(parameters=agent.parameters(), power=cfg["ema_power"])
    ema_agent = build_agent(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"],
    )
    scheduler = get_scheduler(
        cfg["scheduler"], optimizer, num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=cfg["total_iters"],
    )
    writer = SummaryWriter(str(args.log_dir))
    iterator = iter(loader)
    started = time.time()
    for update in range(1, args.updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        observations = batch["observations"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        loss = agent.compute_loss(observations, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.step(agent.parameters())
        if update % cfg["log_freq"] == 0:
            writer.add_scalar("train/loss", loss.item(), update)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], update)
        if update % args.checkpoint_every == 0 or update == args.updates:
            ema.copy_to(ema_agent.parameters())
            path = args.ckpt_dir / f"update_{update:05d}.pt"
            torch.save({"format": FORMAT, "observation_dim": OBS_DIM, "action_dim": ACT_DIM,
                        "update": update, "seed": args.seed, "agent": agent.state_dict(),
                        "ema_agent": ema_agent.state_dict(),
                        "data": {"h5_sha256": sha256_file(args.data),
                                 "manifest_sha256": sha256_file(args.manifest)}}, path)
            writer.add_scalar("train/checkpoint_update", update, update)
            print(json.dumps({"update": update, "loss": loss.item(), "checkpoint": str(path),
                              "elapsed_seconds": time.time() - started}), flush=True)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", type=Path, default=DATA)
    fit.add_argument("--manifest", type=Path, default=DATA_JSON)
    fit.add_argument("--updates", type=int, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--batch-size", type=int, default=1024)
    fit.add_argument("--checkpoint-every", type=int, default=5000)
    fit.add_argument("--ckpt-dir", type=Path, required=True)
    fit.add_argument("--log-dir", type=Path, required=True)
    fit.add_argument("--smoke-only", action="store_true")
    fit.add_argument("--smoke-catalog", type=Path, default=VALIDATION_CATALOG)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--catalog", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--video", action="store_true")
    ev.add_argument("--max-episode-steps", type=int, default=100)
    ev.add_argument("--env-time-limit", type=int, default=None)
    ev.add_argument("--seed", type=int, default=1907)
    ev.add_argument("--scene-indices", type=int, nargs="+")
    ev.add_argument("--record-phase-trace", action="store_true")
    ev.add_argument("--record-stage-geometry", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        print(json.dumps(evaluate(
            args.checkpoint, args.catalog, args.output_dir, args.device, args.video,
            args.max_episode_steps, args.seed, scene_indices=args.scene_indices,
            record_phase_trace=args.record_phase_trace,
            record_stage_geometry=args.record_stage_geometry,
            env_time_limit=args.env_time_limit,
        ), indent=2))


if __name__ == "__main__":
    main()
