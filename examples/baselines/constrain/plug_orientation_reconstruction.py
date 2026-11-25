"""Experiment-only source-state-to-EE diagnostic trajectory reconstruction.

This deliberately does not modify the source demonstrations. It restores only
the recorded initial state, computes all remaining EE targets with Pinocchio,
then follows them using ordinary native controller actions and live physics.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from transforms3d.quaternions import quat2axangle, qinverse, qmult

from examples.baselines.constrain import plug_orientation  # register task IDs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, gym_utils
from mani_skill.utils.structs import Pose


ROBOT_UID = "panda_wristcam"
MAX_TOTAL_CONTROLS = 6000
MAX_CONTROLS_PER_TRAJECTORY = 600
MAX_CONTROLS_PER_WAYPOINT = 3
POSITION_TOLERANCE_M = 0.0005
ROTATION_TOLERANCE_RAD = 0.5 * np.pi / 180


def split_source_trajectory(states, actions):
    """Return the immutable initial row and T following target rows."""
    if len(states) != len(actions) + 1:
        raise ValueError(
            f"reconstruction requires T+1 states for T actions; "
            f"got {len(states)} states and {len(actions)} actions"
        )
    if len(states) < 2:
        raise ValueError("source trajectory must contain at least one action")
    return states[0], states[1:]


def encode_native_pose_delta(controller, current_pose, target_pose):
    """Encode world TCP target as ManiSkill's normalized native 6D delta action.

    Mirrors the public conversion helper's frame and quaternion order. In
    particular, the EE controller multiplies rotational action by rot_lower,
    which is negative for this task.
    """
    current_pose = np.asarray(current_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    delta_position = target_pose[:3] - current_pose[:3]
    delta_quaternion = qmult(current_pose[3:], qinverse(target_pose[3:]))
    axis, theta = quat2axangle(delta_quaternion)
    if theta > np.pi:
        theta -= 2 * np.pi
    delta = np.r_[delta_position, axis * theta]
    low = controller.action_space_low.detach().cpu().numpy()
    high = controller.action_space_high.detach().cpu().numpy()
    action = gym_utils.inv_scale_action(delta, low, high)
    return np.asarray(action, dtype=np.float32)


def run_capped_waypoint_loop(
    env,
    target,
    *,
    initial_observation,
    initial_state,
    make_action,
    error,
    tolerance,
    max_controls,
    gripper,
):
    """Run at least one ordinary control step; never exceed the local cap."""
    observations = [np.array(initial_observation, copy=True)]
    states = [initial_state]
    actions = []
    for _ in range(max_controls):
        action = np.asarray(make_action(target, gripper), dtype=np.float32).copy()
        if not np.isfinite(action).all():
            raise ValueError("native reconstructed action is non-finite")
        result = env.step(action)
        observations.append(np.asarray(result[0], dtype=np.float32).copy())
        actions.append(action)
        live_env = getattr(env, "unwrapped", env)
        if hasattr(live_env, "get_state_dict"):
            states.append(live_env.get_state_dict())
        else:
            states.append(None)
        if error(target) <= tolerance:
            break
        if result[2] or result[3]:
            break
    return observations, states, actions


def _pose_error(actual, target):
    actual, target = np.asarray(actual), np.asarray(target)
    position = float(np.linalg.norm(actual[:3] - target[:3]))
    dq = qmult(qinverse(actual[3:]), target[3:])
    angle = float(2 * np.arctan2(np.linalg.norm(dq[1:]), abs(dq[0])))
    return position, angle


def _state_pose_targets(env, state_rows):
    """Offline FK of saved t+1 joint states; does not set live articulation."""
    articulation = env.unwrapped.agent.robot
    model = articulation.create_pinocchio_model()
    ee_link_index = env.unwrapped.agent.tcp.index
    targets = []
    dof = articulation.max_dof
    for row in state_rows:
        packed = np.asarray(row["articulations"][ROBOT_UID], dtype=np.float64)
        if len(packed) < 13 + 2 * dof:
            raise ValueError(f"packed articulation state too short: {packed.shape}")
        root_pose = Pose.create(packed[:7])
        qpos = packed[13 : 13 + dof]
        model.compute_forward_kinematics(qpos)
        local_tcp = model.get_link_pose(ee_link_index)
        world_tcp = root_pose * Pose.create(local_tcp)
        pose = world_tcp.raw_pose[0].detach().cpu().numpy().copy()
        if not np.isfinite(pose).all():
            raise ValueError("source-state FK produced non-finite TCP pose")
        targets.append(pose)
    return targets


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_episode(
    source_h5, episode_id, output_dir, *, control_cap=600,
    corrections_per_waypoint=MAX_CONTROLS_PER_WAYPOINT,
):
    """Reconstruct a single source episode with native CPU physics controls."""
    source_h5 = Path(source_h5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_meta = json.loads(source_h5.with_suffix(".json").read_text())
    episode_meta = next(
        row for row in source_meta["episodes"]
        if int(row["episode_id"]) == int(episode_id)
    )
    with h5py.File(source_h5, "r") as h5:
        group = h5[f"traj_{episode_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        source_states = trajectory_utils.dict_to_list_of_dicts(group["env_states"])
    initial_state, target_states = split_source_trajectory(source_states, actions)

    env = gym.make(
        source_meta["env_info"]["env_id"], obs_mode="state",
        control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
        reward_mode="sparse", max_episode_steps=control_cap + 1,
    )
    control_trace = []
    observations, states, native_actions = [], [], []
    target_poses = []
    started = time.monotonic()
    attempt_path = output_dir / "pre_step_attempts.jsonl"
    trace_path = output_dir / "control_trace.jsonl"
    try:
        env.reset(seed=int(episode_meta["episode_seed"]))
        # The only state restoration: source row zero, before any control step.
        env.unwrapped.set_state_dict(common.batch(initial_state))
        initial_obs = env.unwrapped.get_obs()
        if isinstance(initial_obs, dict):
            initial_obs = common.flatten_state_dict(initial_obs, use_torch=True)
        initial_obs = common.to_numpy(initial_obs).reshape(-1)
        states.append(env.unwrapped.get_state_dict())
        observations.append(np.array(initial_obs, dtype=np.float32, copy=True))
        target_poses = _state_pose_targets(env, target_states)
        controller = env.unwrapped.agent.controller.controllers["arm"]
        composite = env.unwrapped.agent.controller
        max_steps = min(int(control_cap), MAX_CONTROLS_PER_TRAJECTORY)
        used = 0
        for waypoint, (target_pose, source_action) in enumerate(zip(target_poses, actions)):
            if used >= max_steps:
                break
            grip = float(source_action[-1])
            for attempt in range(min(int(corrections_per_waypoint), MAX_CONTROLS_PER_WAYPOINT)):
                current = env.unwrapped.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
                pos_error, rot_error = _pose_error(current, target_pose)
                # Always issue one control for each source waypoint so the copied
                # gripper stream is applied even for nearly identical EE targets.
                if attempt and pos_error <= POSITION_TOLERANCE_M and rot_error <= ROTATION_TOLERANCE_RAD:
                    break
                arm_action = encode_native_pose_delta(controller, current, target_pose)
                grip_action = np.asarray([grip], dtype=np.float32)
                action_dict = {
                    "arm": common.to_tensor(arm_action, device=env.unwrapped.device).reshape(1, -1),
                    "gripper": common.to_tensor(grip_action, device=env.unwrapped.device).reshape(1, -1),
                }
                action = common.to_numpy(composite.from_action_dict(action_dict)).reshape(-1)
                if used >= max_steps or used >= MAX_TOTAL_CONTROLS:
                    break
                before = {
                    "tcp_pose": current.tolist(),
                    "target_pose": target_pose.tolist(),
                    "position_error_m": pos_error,
                    "rotation_error_rad": rot_error,
                }
                used += 1  # Count immediately before the physical step.
                with attempt_path.open("a") as journal:
                    journal.write(json.dumps({
                        "control": used - 1, "waypoint": waypoint,
                        "correction": attempt, "action": action.tolist(),
                        "before": before,
                    }, sort_keys=True, allow_nan=False) + "\n")
                    journal.flush()
                obs, _, terminated, truncated, info = env.step(action)
                observations.append(np.asarray(common.to_numpy(obs)).reshape(-1).astype(np.float32))
                native_actions.append(np.array(action, dtype=np.float32, copy=True))
                states.append(env.unwrapped.get_state_dict())
                after = env.unwrapped.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
                p_err, r_err = _pose_error(after, target_pose)
                control_trace.append({
                    "control": used - 1, "waypoint": waypoint, "correction": attempt,
                    "action": action.tolist(), "gripper_source_action": grip,
                    "before": before, "after_tcp_pose": after.tolist(),
                    "after_position_error_m": p_err,
                    "after_rotation_error_rad": r_err,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "info": _jsonable(info),
                })
                with trace_path.open("a") as journal:
                    journal.write(json.dumps(control_trace[-1], sort_keys=True, allow_nan=False) + "\n")
                    journal.flush()
                if terminated or truncated:
                    break
            if used >= max_steps:
                break
        final_info = env.unwrapped.evaluate()
        report = {
            "method": "source_state_t_plus_1_pinocchio_fk_world_tcp_native_pd_ee_delta_pose",
            "identity": "reconstructed target stream; not original actions/replay",
            "source_h5": str(source_h5), "source_h5_sha256": _sha256(source_h5),
            "source_json_sha256": _sha256(source_h5.with_suffix(".json")),
            "episode_id": int(episode_id), "seed": int(episode_meta["episode_seed"]),
            "source_action_count": int(len(actions)), "source_state_count": int(len(source_states)),
            "target_stream_sha256": hashlib.sha256(np.asarray(target_poses, dtype=np.float32).tobytes()).hexdigest(),
            "initial_state_restores": 1, "intermediate_state_restores": 0,
            "controls": used, "control_cap": max_steps,
            "corrections_per_waypoint_cap": int(corrections_per_waypoint),
            "target_waypoints": len(target_poses),
            "observations_shape": list(np.asarray(observations).shape),
            "actions_shape": list(np.asarray(native_actions).shape),
            "states_count": len(states), "elapsed_seconds": time.monotonic() - started,
            "terminal": _jsonable(final_info),
            "controller": {
                "frame": controller.config.frame, "use_delta": controller.config.use_delta,
                "use_target": controller.config.use_target,
                "rot_lower": _jsonable(controller.config.rot_lower),
                "action_low": _jsonable(controller.action_space_low),
                "action_high": _jsonable(controller.action_space_high),
            },
            "control_trace": control_trace,
        }
        if len(states) != len(native_actions) + 1 or len(observations) != len(native_actions) + 1:
            raise AssertionError("reconstruction recorder lost T+1 initial/state/observation row")
        if np.asarray(observations).shape[1] != 46:
            raise AssertionError(f"expected 46D state observations, got {np.asarray(observations).shape}")
        with h5py.File(output_dir / "reconstruction.h5", "w") as h5:
            h5.create_dataset("actions", data=np.asarray(native_actions, dtype=np.float32))
            h5.create_dataset("obs", data=np.asarray(observations, dtype=np.float32))
            h5.create_dataset("source_target_poses_world", data=np.asarray(target_poses, dtype=np.float32))
            h5.create_dataset("source_target_index_per_control", data=np.asarray([x["waypoint"] for x in control_trace], dtype=np.int32))
            h5.create_dataset("env_states_json", data=np.asarray([
                json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) for row in states
            ], dtype=h5py.string_dtype("utf-8")))
        (output_dir / "reconstruction_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        return report
    finally:
        env.close()
