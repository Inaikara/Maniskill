"""Build an outside-entry physical initialization bank for Peg yaw/lateral probes."""

import argparse
import hashlib
import json
import os
import time
from itertools import combinations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from examples.baselines.constrain import peg_orientation
from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog, restore_peg_scene
from examples.baselines.constrain.peg_orientation_near_hole_eval import (
    BANK, CATALOG_ROOT, DATA_DIRS, _source_actions, _bank_rows, sha256, jsonable,
)
from examples.baselines.constrain.peg_orientation_takeover import capture_physical_state, native_grasped

ROOT = Path(__file__).resolve().parents[3]
PROFILES = ("random", "fixed")
CONDITIONS = ("aligned", "lateral_holeY_plus5mm", "lateral_holeY_minus5mm",
              "yaw_worldZ_plus5deg", "yaw_worldZ_minus5deg")
WITHDRAWAL_GAP_M = 0.020
LATERAL_M = 0.005
YAW_RAD = np.deg2rad(5.0)
MAX_INIT_CONTROLS = 120
WITHDRAW_CAP = 60
STABILIZE_WINDOW = 5
POSE_WINDOW_SAMPLES = 11
HEAD_AXIS_TOLERANCE_M = np.array([.001, .002, .003])
HEAD_NORM_TOLERANCE_M = .003
REGISTERED_PATHS = (
    "examples/baselines/constrain/peg_orientation_near_hole_perturb.py",
    "examples/baselines/constrain/tests/test_peg_orientation_near_hole_perturb.py",
    "examples/baselines/constrain/peg_orientation_near_hole_eval.py",
    "examples/baselines/constrain/peg_orientation_near_hole.py",
    "examples/baselines/constrain/peg_orientation_data.py",
    "examples/baselines/constrain/peg_orientation_takeover.py",
    "examples/baselines/constrain/peg_orientation.py",
    "mani_skill/agents/controllers/pd_ee_pose.py",
    "mani_skill/agents/controllers/base_controller.py",
    "mani_skill/agents/robots/panda/panda.py",
    "mani_skill/envs/tasks/tabletop/peg_insertion_side.py",
    "examples/baselines/constrain/tests/test_peg_orientation.py",
    "examples/baselines/constrain/tests/test_base_dp.py",
    "examples/baselines/constrain/tests/test_peg_orientation_preflight.py",
    "examples/baselines/constrain/tests/test_peg_orientation_data.py",
    "examples/baselines/constrain/tests/test_peg_orientation_base_dp.py",
    "examples/baselines/constrain/tests/test_peg_official_dp.py",
    "examples/baselines/constrain/tests/test_peg_orientation_constrain.py",
    "examples/baselines/constrain/tests/test_peg_orientation_failure_analysis.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover_study.py",
    "examples/baselines/constrain/tests/test_peg_orientation_near_hole.py",
    "examples/baselines/constrain/tests/test_peg_orientation_near_hole_eval.py",
    "examples/baselines/constrain/tests/test_peg_orientation_near_hole_hold.py",
)


@dataclass
class PoseValue:
    position: np.ndarray
    rotation: np.ndarray  # scalar-first quaternion


@dataclass
class NativePoseCommandGeometry:
    pos_lower: np.ndarray
    pos_upper: np.ndarray
    rot_lower: np.ndarray
    rot_upper: np.ndarray
    frame: str
    use_delta: bool


def quat_from_euler_xyz(euler):
    xyzw = Rotation.from_euler("XYZ", euler).as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def _rotation(q):
    return Rotation.from_quat(np.r_[q[1:], q[0]])


def _euler_xyz(rotation):
    return rotation.as_euler("XYZ")


def compute_native_target(current, action, config):
    """Pure equivalent of PDEEPoseController.compute_target_pose for this config."""
    action = np.asarray(action, dtype=np.float64)
    # Match Controller._preprocess_action -> _clip_and_scale_action before target-pose math.
    if config.use_delta:
        normalized_rotation = action[3:6].copy()
        norm = np.linalg.norm(normalized_rotation)
        if norm > 1:
            normalized_rotation /= norm
        action = np.r_[config.pos_lower + (action[:3] + 1) *
                       (config.pos_upper - config.pos_lower) / 2,
                       normalized_rotation * config.rot_lower, action[6:]]
    if not config.use_delta:
        return PoseValue(action[:3].copy(), quat_from_euler_xyz(action[3:6]))
    delta = Rotation.from_euler("XYZ", action[3:6])
    current_rotation = _rotation(current.rotation)
    if "root_aligned_body_rotation" in config.frame:
        rotation = delta * current_rotation
    elif "body_aligned_body_rotation" in config.frame:
        rotation = current_rotation * delta
    else:
        raise ValueError(f"unsupported native rotation frame {config.frame!r}")
    if "root_translation" in config.frame:
        position = current.position + action[:3]
    elif "body_translation" in config.frame:
        position = current.position + current_rotation.apply(action[:3])
    else:
        raise ValueError(f"unsupported native translation frame {config.frame!r}")
    return PoseValue(position, np.r_[rotation.as_quat()[3], rotation.as_quat()[:3]])


def native_pose_action(current, desired, config, root_rotation=None):
    """Invert native normalized PD EE delta scaling, returning audit details."""
    root_rotation = root_rotation or Rotation.identity()
    if config.use_delta:
        current_position = np.asarray(current.position, dtype=np.float64)
        desired_position = np.asarray(desired.position, dtype=np.float64)
        if "root_translation" in config.frame:
            translation = root_rotation.inv().apply(desired_position - current_position)
        elif "body_translation" in config.frame:
            translation = _rotation(current.rotation).inv().apply(desired.position - current.position)
        else:
            raise ValueError(config.frame)
        desired_root = root_rotation.inv() * _rotation(desired.rotation)
        current_root = root_rotation.inv() * _rotation(current.rotation)
        if "root_aligned_body_rotation" in config.frame:
            delta = desired_root * current_root.inv()
        else:
            delta = current_root.inv() * desired_root
        rotation_euler = _euler_xyz(delta)
        normalized_rotation = rotation_euler / config.rot_lower
        clipped_translation = np.clip(translation, config.pos_lower, config.pos_upper)
        clipped_rotation = normalized_rotation / max(1.0, np.linalg.norm(normalized_rotation))
        normalized_translation = 2 * (clipped_translation - config.pos_lower) / (config.pos_upper - config.pos_lower) - 1
        action = np.r_[normalized_translation, clipped_rotation, -1.0].astype(np.float32)
        actual_translation = config.pos_lower + (normalized_translation + 1) * (config.pos_upper - config.pos_lower) / 2
        actual_euler = clipped_rotation * config.rot_lower
        return {"action": action, "requested_translation_root_m": translation.tolist(),
                "realized_translation_root_m": actual_translation.tolist(),
                "requested_rotation_euler_xyz_rad": rotation_euler.tolist(),
                "realized_rotation_euler_xyz_rad": actual_euler.tolist(),
                "action_clipped": bool(not np.allclose(translation, actual_translation, atol=1e-8) or
                                        not np.allclose(rotation_euler, actual_euler, atol=1e-8))}
    raise ValueError("initializer requires native delta pose controller")


def rotate_tcp_about_head(tcp, head_world, yaw_delta):
    pivot = np.asarray(head_world, dtype=np.float64)
    yaw = Rotation.from_rotvec([0, 0, yaw_delta])
    rotation = yaw * _rotation(tcp.rotation)
    position = pivot + yaw.apply(tcp.position - pivot)
    q = rotation.as_quat()
    return PoseValue(position, np.r_[q[3], q[:3]])


def entry_clearance(head_hole_xyz, peg_half_length):
    """Positive means the head remains outside the true front plane x=-half-length."""
    return float(-np.asarray(head_hole_xyz)[0] - float(peg_half_length))


def head_position_eligible(head_xyz, target_xyz):
    error = np.asarray(head_xyz, dtype=np.float64) - np.asarray(target_xyz, dtype=np.float64)
    return bool(np.linalg.norm(error) <= HEAD_NORM_TOLERANCE_M and
                np.all(np.abs(error) <= HEAD_AXIS_TOLERANCE_M))


def lateral_admission_eligible(delta_y_m, direction):
    """Inclusive signed band around the registered ±5 mm lateral target."""
    delta_y_m = float(delta_y_m)
    direction = int(direction)
    if direction not in (-1, 1):
        raise ValueError("lateral direction must be -1 or +1")
    measured_signed = direction * delta_y_m
    return bool(.003 <= measured_signed <= .007)


def minimal_window_admission(rows, control_timestep, *, expected_lateral_direction=None):
    """Validate the prospective consecutive-control pose window; velocity is diagnostic only."""
    if len(rows) != POSE_WINDOW_SAMPLES or not np.isfinite(control_timestep) or control_timestep <= 0:
        return False
    indices = [int(row["control_index"]) for row in rows]
    if any(b - a != 1 for a, b in zip(indices, indices[1:])):
        return False
    times = [float(row["monotonic_seconds"]) for row in rows]
    sim_times = [float(row["simulation_elapsed_seconds"]) for row in rows]
    if (not np.isfinite(times + sim_times).all() or any(b <= a for a, b in zip(times, times[1:])) or
            any(b <= a for a, b in zip(sim_times, sim_times[1:]))):
        return False
    if abs((sim_times[-1] - sim_times[0]) - 10 * float(control_timestep)) > max(1e-9, control_timestep * 1e-6):
        return False
    try:
        for row in rows:
            head_error = np.asarray(row["head_error_xyz_m"], dtype=float)
            head = np.asarray(row["head_hole_xyz_m"], dtype=float)
            center = np.asarray(row["peg_center_world_m"], dtype=float)
            quat = np.asarray(row["peg_quaternion_wxyz"], dtype=float)
            joints = np.asarray(row["arm_qpos_rad"], dtype=float)
            clearance = float(row["entry_clearance_m"])
            yaw_error = float(row["yaw_error_rad"])
            if (head_error.shape != (3,) or head.shape != (3,) or center.shape != (3,) or
                    quat.shape != (4,) or joints.shape != (7,) or
                    not all(np.isfinite(x).all() for x in (head_error, head, center, quat, joints)) or
                    not np.isfinite([clearance, yaw_error]).all() or np.linalg.norm(quat) <= 1e-12 or
                    np.linalg.norm(head_error) > .003 or
                    np.any(np.abs(head_error) > np.array([.001, .002, .003])) or
                    not .019 <= clearance <= .021 or abs(yaw_error) > np.deg2rad(1) or
                    not bool(row["grasped"]) or bool(row["official_inserted"])):
                return False
            if expected_lateral_direction is not None and not lateral_admission_eligible(
                    float(row["head_hole_xyz_m"][1]) - float(row["base_head_y_m"]),
                    expected_lateral_direction):
                return False
        heads = np.asarray([r["head_hole_xyz_m"] for r in rows], dtype=float)
        centers = np.asarray([r["peg_center_world_m"] for r in rows], dtype=float)
        quats = np.asarray([r["peg_quaternion_wxyz"] for r in rows], dtype=float)
        joints = np.asarray([r["arm_qpos_rad"] for r in rows], dtype=float)
        if not all(np.isfinite(x).all() for x in (heads, centers, quats, joints)):
            return False
        if max(np.linalg.norm(a-b) for a,b in combinations(heads, 2)) > .00025:
            return False
        if max(np.linalg.norm(a-b) for a,b in combinations(centers, 2)) > .00025:
            return False
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        angles = [2*np.arccos(np.clip(abs(float(np.dot(a,b))), -1, 1)) for a,b in combinations(quats, 2)]
        if max(angles) > np.deg2rad(.25) or np.any(np.ptp(joints, axis=0) > .002):
            return False
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return True


def pose_qualified(head, head_target, yaw_error, grasped, inserted):
    return bool(grasped and not inserted and head_position_eligible(head, head_target)
                and abs(float(yaw_error)) <= np.deg2rad(1))


def condition_target(name, base_head, base_yaw, peg_half_length):
    head = np.asarray(base_head, dtype=np.float64).copy()
    head[0] = -float(peg_half_length) - WITHDRAWAL_GAP_M
    yaw = base_yaw
    if name == "lateral_holeY_plus5mm":
        head[1] += LATERAL_M
    elif name == "lateral_holeY_minus5mm":
        head[1] -= LATERAL_M
    elif name == "yaw_worldZ_plus5deg":
        yaw += YAW_RAD
    elif name == "yaw_worldZ_minus5deg":
        yaw -= YAW_RAD
    elif name != "aligned":
        raise ValueError(f"unknown condition {name}")
    return head, float(yaw)


def _as_np(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).copy()


def _pose_value(pose):
    raw = _as_np(pose.raw_pose).reshape(-1)
    return PoseValue(raw[:3], raw[3:7])


def _task_head(task):
    return _as_np((task.box_hole_pose.inv() * task.peg_head_pose).p[0])


def _yaw_error(task, target_yaw):
    peg_yaw = float(np.arctan2(_rotation(_pose_value(task.peg.pose).rotation).as_matrix()[1, 0],
                               _rotation(_pose_value(task.peg.pose).rotation).as_matrix()[0, 0]))
    hole = _pose_value(task.box_hole_pose)
    hole_yaw = float(_rotation(hole.rotation).as_euler("xyz")[2])
    relative = peg_yaw - hole_yaw
    return float(np.arctan2(np.sin(relative - target_yaw), np.cos(relative - target_yaw)))


def _native_config(env):
    combined = env.unwrapped.agent.controller
    controller = combined.controllers.get("arm", combined)
    c = controller.config
    def vec(value): return np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()
    if not c.use_delta or c.frame != "root_translation:root_aligned_body_rotation":
        raise ValueError(f"unexpected controller contract: use_delta={c.use_delta}, frame={c.frame}")
    return controller, NativePoseCommandGeometry(vec(c.pos_lower), vec(c.pos_upper),
        vec(c.rot_lower), vec(c.rot_upper), c.frame, c.use_delta)


def _native_controller_preflight(env):
    from mani_skill.utils.structs import Pose

    controller, geometry = _native_config(env)
    if bool(getattr(controller.config, "use_target", False)):
        raise ValueError("native controller unexpectedly uses a target buffer")
    if not callable(getattr(env.unwrapped.peg, "get_linear_velocity", None)) or not callable(
            getattr(env.unwrapped.peg, "get_angular_velocity", None)):
        raise RuntimeError("required public peg velocity APIs are unavailable")
    current_pose = controller.ee_pose_at_base
    current = _pose_value(current_pose)
    for sign in (-1, 1):
        delta_angle = sign * np.deg2rad(1)
        desired_rotation = Rotation.from_rotvec([0, 0, delta_angle]) * _rotation(current.rotation)
        q = desired_rotation.as_quat()
        desired = PoseValue(current.position + np.array([.001, 0, 0]), np.r_[q[3], q[:3]])
        command = native_pose_action(current, desired, geometry)
        raw = torch.as_tensor(command["action"][:6][None], dtype=torch.float32, device=env.unwrapped.device)
        scaled = controller._clip_and_scale_action(raw)
        target = controller.compute_target_pose(current_pose, scaled)
        if not np.allclose(_as_np(target.p)[0], desired.position, atol=1e-6, rtol=0):
            raise ValueError("actual native controller position normalization mismatch")
        target_q = _as_np(target.q)[0]
        if abs(float(np.dot(target_q, desired.rotation))) < 1 - 1e-6:
            raise ValueError("actual native controller rotation sign/frame/units mismatch")
    return {"controller_type": f"{type(controller).__module__}.{type(controller).__qualname__}",
            "frame": geometry.frame, "use_delta": geometry.use_delta,
            "use_target": bool(controller.config.use_target),
            "pos_lower": geometry.pos_lower.tolist(), "pos_upper": geometry.pos_upper.tolist(),
            "rot_lower": geometry.rot_lower.tolist(), "rot_upper": geometry.rot_upper.tolist(),
            "actual_controller_positive_negative_yaw_preflight": "passed",
            "public_peg_velocity_apis": ["get_linear_velocity", "get_angular_velocity"]}


def _controller_step(env, desired_tcp):
    task = env.unwrapped
    controller, geometry = _native_config(env)
    current = _pose_value(task.agent.tcp.pose)
    root_rotation = _rotation(_pose_value(task.agent.robot.pose).rotation)
    command = native_pose_action(current, desired_tcp, geometry, root_rotation)
    before_head = _task_head(task)
    obs, _, terminated, truncated, info = env.step(command["action"])
    return obs, {"command": command, "terminated": bool(np.asarray(terminated).any()),
                 "truncated": bool(np.asarray(truncated).any()), "native_success": bool(np.asarray(info.get("success", False)).any()),
                 "head_hole_xyz_m": _task_head(task).tolist(), "tcp_pose": jsonable(task.agent.tcp.pose.raw_pose),
                 "head_before_m": before_head.tolist()}


def _joint_velocity(task):
    return _as_np(task.agent.robot.get_qvel())[0, :7]


def _actor_velocity(actor, method):
    value = getattr(actor, method, None)
    if value is None:
        raise RuntimeError(f"public actor API {method} is unavailable")
    return _as_np(value())


def _velocity_measurement(task):
    arm_qvel = _joint_velocity(task)
    peg_linear = _actor_velocity(task.peg, "get_linear_velocity").reshape(-1)
    peg_angular = _actor_velocity(task.peg, "get_angular_velocity").reshape(-1)
    return {"arm_qvel_rad_s": arm_qvel.tolist(),
            "max_arm_qvel_rad_s": float(np.max(np.abs(arm_qvel))),
            "peg_linear_velocity_m_s": peg_linear.tolist(),
            "peg_linear_speed_m_s": float(np.linalg.norm(peg_linear)),
            "peg_angular_velocity_rad_s": peg_angular.tolist(),
            "peg_angular_speed_rad_s": float(np.linalg.norm(peg_angular))}


def _settled(task, head_target, yaw_target):
    head = _task_head(task)
    head_error_vector = head - head_target
    head_error = float(np.linalg.norm(head_error_vector))
    yaw_error = abs(_yaw_error(task, yaw_target))
    velocity = _velocity_measurement(task)
    inserted = bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any())
    ok = (pose_qualified(head, head_target, yaw_error, native_grasped(task), inserted) and
          velocity["max_arm_qvel_rad_s"] <= .02 and velocity["peg_linear_speed_m_s"] <= .005 and
          velocity["peg_angular_speed_rad_s"] <= .05)
    return ok, {"head_error_m": head_error, "head_error_xyz_m": head_error_vector.tolist(),
                "yaw_error_rad": yaw_error, **velocity,
                "grasped": native_grasped(task), "official_inserted": inserted, "head_hole_xyz_m": head.tolist()}


def _tcp_target_from_head(task, target_head_hole, desired_peg_rotation):
    """Use the current physical peg-to-TCP transform, including grasp slip."""
    hole = _pose_value(task.box_hole_pose)
    hole_rotation = _rotation(hole.rotation)
    target_head_world = hole.position + hole_rotation.apply(target_head_hole)
    peg_pose = _pose_value(task.peg.pose)
    tcp_pose = _pose_value(task.agent.tcp.pose)
    peg_rotation = _rotation(peg_pose.rotation)
    head_offset_peg = peg_rotation.inv().apply(_as_np(task.peg_head_pose.p[0]) - peg_pose.position)
    tcp_offset_peg = peg_rotation.inv().apply(tcp_pose.position - peg_pose.position)
    desired_peg_position = target_head_world - desired_peg_rotation.apply(head_offset_peg)
    desired_tcp_position = desired_peg_position + desired_peg_rotation.apply(tcp_offset_peg)
    grasp_relative_rotation = peg_rotation.inv() * _rotation(tcp_pose.rotation)
    desired_tcp_rotation = desired_peg_rotation * grasp_relative_rotation
    quat = desired_tcp_rotation.as_quat()
    return PoseValue(desired_tcp_position, np.r_[quat[3], quat[:3]])


def _run_condition_body(env, scene, actions, bank_row, name, budget, telemetry, settling_strategy="continuous", admission_mode="instant_velocity"):
    settling_strategy = telemetry.get("settling_strategy", settling_strategy)
    admission_mode = telemetry.get("admission_mode", admission_mode)
    if settling_strategy not in ("continuous", "fixed_tcp"):
        raise ValueError(f"unknown settling strategy {settling_strategy!r}")
    if admission_mode not in ("instant_velocity", "pose_window"):
        raise ValueError(f"unknown admission mode {admission_mode!r}")
    telemetry["admission_mode"] = admission_mode
    control_timestep = float(env.control_timestep)
    telemetry["control_timestep_s"] = control_timestep
    task = env.unwrapped
    calibration = _native_controller_preflight(env)
    telemetry["controller_calibration"] = calibration
    obs = restore_peg_scene(env, scene)
    history = [_as_np(obs).reshape(-1)]
    telemetry["observation_history"] = history
    prefix_hash = hashlib.sha256()
    prefix_actions = np.asarray(actions[:int(bank_row["prefix_controls"])], dtype=np.float32)
    condition_controls = 0
    for action in prefix_actions:
        if condition_controls >= 420 or budget[0] >= budget[2] or time.monotonic() >= budget[1]:
            raise TimeoutError("global initialization control/deadline bound reached")
        prefix_hash.update(np.asarray(action).tobytes())
        obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        budget[0] += 1
        condition_controls += 1
        history.append(_as_np(obs).reshape(-1))
        telemetry["executed_prefix_actions"] = prefix_actions[:condition_controls]
        if bool(np.asarray(info.get("success", False)).any()) or terminated or truncated:
            return {"admitted": False, "reason": "prefix_terminated_or_inserted", "controls": condition_controls,
                    "prefix_actions": prefix_actions.tolist(), "replay_trace": [x.tolist() for x in history]}
    if prefix_hash.hexdigest() != bank_row["prefix_actions_sha256"]:
        raise ValueError("expert prefix action hash differs from source bank")
    if not native_grasped(task):
        return {"admitted": False, "reason": "prefix_grasp_lost", "controls": condition_controls,
                "prefix_actions": prefix_actions.tolist(), "replay_trace": [x.tolist() for x in history]}
    if not np.allclose(np.asarray(history[-2:]), bank_row["observation_history"], atol=1e-5, rtol=0):
        raise ValueError("replayed prefix observation history mismatch")
    if max(np.max(np.abs(_as_np(v) - np.asarray(bank_row["physical_state"][k])))
           for k, v in capture_physical_state(task).items()) > 1e-5:
        raise ValueError("replayed prefix physical state mismatch")
    base_head = _task_head(task)
    peg_pose_initial = _pose_value(task.peg.pose)
    hole_pose_initial = _pose_value(task.box_hole_pose)
    peg_yaw_initial = _rotation(peg_pose_initial.rotation).as_euler("xyz")[2]
    hole_yaw_initial = _rotation(hole_pose_initial.rotation).as_euler("xyz")[2]
    initial_yaw = float(np.arctan2(np.sin(peg_yaw_initial-hole_yaw_initial),
                                   np.cos(peg_yaw_initial-hole_yaw_initial)))
    half_length = float(task.peg_half_sizes[0, 0].detach().cpu())
    head_target, yaw_target = condition_target(name, base_head, initial_yaw, half_length)
    tcp = _pose_value(task.agent.tcp.pose)
    peg = _pose_value(task.peg.pose)
    peg_yaw = _rotation(peg.rotation).as_euler("xyz")[2]
    target_peg_yaw = hole_yaw_initial + yaw_target
    yaw_move = target_peg_yaw - peg_yaw
    world_z_delta = yaw_move  # requested perturbation is world-Z, not hole-axis rotation
    initial_peg_rotation = _rotation(peg_pose_initial.rotation)
    desired_peg_rotation = Rotation.from_rotvec([0, 0, world_z_delta]) * initial_peg_rotation
    records, phases = [], []
    telemetry.update({"trace": records, "measurement_windows": {}, "control_measurements": [],
                      "prefix_action_sha256": prefix_hash.hexdigest()})
    fixed_tcp_target = None
    latch_step = None
    # Withdraw first. Keep the original peg axis and lateral coordinates until true front-plane clearance.
    for phase in ("withdraw", "perturb_settle"):
        cap = WITHDRAW_CAP if phase == "withdraw" else MAX_INIT_CONTROLS - WITHDRAW_CAP
        stable = 0
        stability_rows = []
        for _ in range(cap):
            if condition_controls >= 420 or budget[0] >= budget[2] or time.monotonic() >= budget[1]:
                raise TimeoutError("global initialization control/deadline bound reached")
            current_head = _task_head(task)
            tcp = _pose_value(task.agent.tcp.pose)
            # Continuous mode recomputes the grasp offset; fixed mode latches one measured world TCP pose.
            hole_pose = _pose_value(task.box_hole_pose)
            desired_head = head_target if phase == "perturb_settle" else np.array(
                [-half_length-WITHDRAWAL_GAP_M, base_head[1], base_head[2]])
            if phase == "withdraw":
                head_world_target = hole_pose.position + _rotation(hole_pose.rotation).apply(desired_head)
                current_head_world = _as_np(task.peg_head_pose.p[0])
                delta_world = head_world_target - current_head_world
                desired_tcp = PoseValue(tcp.position + delta_world, tcp.rotation)
            else:
                desired_tcp = (fixed_tcp_target if fixed_tcp_target is not None else
                               _tcp_target_from_head(task, desired_head, desired_peg_rotation))
            obs, step_row = _controller_step(env, desired_tcp)
            budget[0] += 1
            condition_controls += 1
            history.append(_as_np(obs).reshape(-1))
            telemetry["executed_prefix_actions"] = prefix_actions
            telemetry["observation_history"] = history
            telemetry["executed_initializer_actions"] = [r["command"]["action"] for r in records] + [step_row["command"]["action"]]
            clearance = entry_clearance(_task_head(task), half_length)
            step_row.update({"phase": phase, "entry_clearance_m": clearance,
                             "head_target_hole_xyz_m": head_target.tolist(),
                             "yaw_target_rad": yaw_target})
            records.append(step_row)
            _, physical_measurement = _settled(task, head_target, yaw_target)
            if phase == "perturb_settle" and settling_strategy == "fixed_tcp" and fixed_tcp_target is None:
                inserted_now = bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any())
                if pose_qualified(_task_head(task), head_target, _yaw_error(task, yaw_target),
                                  native_grasped(task), inserted_now):
                    actual_tcp = _pose_value(task.agent.tcp.pose)
                    fixed_tcp_target = PoseValue(actual_tcp.position.copy(), actual_tcp.rotation.copy())
                    latch_step = len(records)
            step_row["settling_strategy"] = settling_strategy
            step_row["fixed_tcp_target_latched"] = fixed_tcp_target is not None
            step_row["fixed_tcp_target_pose"] = (None if fixed_tcp_target is None else {
                "position": fixed_tcp_target.position.tolist(), "rotation": fixed_tcp_target.rotation.tolist()})
            step_row["fixed_tcp_target_latch_control"] = latch_step
            telemetry["control_measurements"].append({
                **physical_measurement,
                "actual_yaw_error_rad": _yaw_error(task, yaw_target),
                "entry_clearance_m": clearance,
                "physical_state": capture_physical_state(task),
                "requested_translation_root_m": step_row["command"]["requested_translation_root_m"],
                "realized_translation_root_m": step_row["command"]["realized_translation_root_m"],
                "requested_rotation_euler_xyz_rad": step_row["command"]["requested_rotation_euler_xyz_rad"],
                "realized_rotation_euler_xyz_rad": step_row["command"]["realized_rotation_euler_xyz_rad"],
                "action_clipped": step_row["command"]["action_clipped"],
                "requested_action": step_row["command"]["action"],
                "desired_head_hole_xyz_m": head_target.tolist(), "desired_yaw_rad": yaw_target,
                "control_index": condition_controls, "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "monotonic_seconds": time.monotonic(), "simulation_elapsed_seconds": condition_controls * control_timestep,
                "peg_center_world_m": _pose_value(task.peg.pose).position.tolist(),
                "peg_quaternion_wxyz": _pose_value(task.peg.pose).rotation.tolist(),
                "arm_qpos_rad": _as_np(task.agent.robot.get_qpos())[0, :7].tolist(),
                "base_head_y_m": float(base_head[1]),
            })
            if step_row["native_success"] or step_row["terminated"] or step_row["truncated"]:
                return {"admitted": False, "reason": "success_or_termination_during_init", "trace": records,
                        "prefix_actions": prefix_actions.tolist(), "prefix_action_sha256": prefix_hash.hexdigest(),
                        "source_prefix_controls": len(prefix_actions)}
            if phase == "withdraw" and clearance >= WITHDRAWAL_GAP_M - .001:
                telemetry["phase_entry_after_withdrawal"] = {
                    "physical_state": capture_physical_state(task),
                    "observation_history_last_two": [x.tolist() for x in history[-2:]],
                    "head_hole_xyz_m": _task_head(task).tolist(),
                    "tcp_pose": jsonable(task.agent.tcp.pose.raw_pose),
                    "prefix_action_sha256": prefix_hash.hexdigest(),
                    "prefix_controls_executed": condition_controls - len(records),
                    "initializer_controls_before_perturb": len(records),
                }
                phases.append({"phase": phase, "controls": len(records)})
                break
            if phase == "perturb_settle":
                is_settled, measurement = _settled(task, head_target, yaw_target)
                measurement.update(telemetry["control_measurements"][-1])
                basic_pose_ok = (pose_qualified(_task_head(task), head_target, _yaw_error(task, yaw_target),
                                                native_grasped(task), measurement["official_inserted"]) and
                                 .019 <= clearance <= .021 and lateral_admission_eligible(
                                     float(_task_head(task)[1] - base_head[1]),
                                     (1 if name.endswith("plus5mm") else -1)) if name.startswith("lateral_holeY_") else
                                 pose_qualified(_task_head(task), head_target, _yaw_error(task, yaw_target),
                                                native_grasped(task), measurement["official_inserted"]) and .019 <= clearance <= .021)
                if admission_mode == "pose_window":
                    stability_rows = ((stability_rows + [measurement])[-POSE_WINDOW_SAMPLES:]
                                      if basic_pose_ok else [])
                    direction = ((1 if name.endswith("plus5mm") else -1)
                                 if name.startswith("lateral_holeY_") else None)
                    window_qualifies = minimal_window_admission(
                        stability_rows, control_timestep, expected_lateral_direction=direction)
                    telemetry["pose_window_candidate"] = stability_rows
                else:
                    stability_rows.append(measurement)
                    window_qualifies = False
                telemetry["measurement_windows"][phase] = stability_rows
                stable = stable + 1 if is_settled else 0
                reached = (window_qualifies if admission_mode == "pose_window" else stable >= STABILIZE_WINDOW)
                if reached:
                    phases.append({"phase": phase, "controls": len(stability_rows)})
                    break
        else:
            return {"admitted": False, "reason": f"{phase}_control_cap", "trace": records}
        if phase == "withdraw" and entry_clearance(_task_head(task), half_length) < .019:
            return {"admitted": False, "reason": "withdrawal_not_physically_outside", "trace": records}
    # Settle toward fixed targets, not toward the drifting current state.
    window = stability_rows[-(POSE_WINDOW_SAMPLES if admission_mode == "pose_window" else STABILIZE_WINDOW):]
    ok, final = _settled(task, head_target, yaw_target)
    clearance = entry_clearance(_task_head(task), half_length)
    lateral_ok = True
    if name.startswith("lateral_holeY_"):
        direction = 1 if name.endswith("plus5mm") else -1
        lateral_ok = lateral_admission_eligible(float(_task_head(task)[1] - base_head[1]), direction)
    if admission_mode == "pose_window":
        direction = (1 if name.endswith("plus5mm") else -1) if name.startswith("lateral_holeY_") else None
        admitted = minimal_window_admission(window, control_timestep, expected_lateral_direction=direction)
    else:
        admitted = bool(ok and .019 <= clearance <= .021 and len(window) == STABILIZE_WINDOW and all(
        np.linalg.norm(r["head_error_xyz_m"]) <= HEAD_NORM_TOLERANCE_M and
        np.all(np.abs(r["head_error_xyz_m"]) <= HEAD_AXIS_TOLERANCE_M) and
        r["yaw_error_rad"] <= np.deg2rad(1) and
        r["max_arm_qvel_rad_s"] <= .02 and r["peg_linear_speed_m_s"] <= .005 and
        r["peg_angular_speed_rad_s"] <= .05 and r["grasped"] and not r["official_inserted"] for r in window)
        and lateral_ok)
    return {"admitted": admitted, "reason": None if admitted else "stability_criteria_not_met",
             "admission_mode": admission_mode, "control_timestep_s": control_timestep,
            "controls": condition_controls, "scripted_controls": len(records), "prefix_controls": int(bank_row["prefix_controls"]),
            "initial_head_hole_xyz_m": base_head.tolist(), "desired_head_hole_xyz_m": head_target.tolist(),
            "actual_head_hole_xyz_m": _task_head(task).tolist(),
            "actual_entry_clearance_m": entry_clearance(_task_head(task), half_length),
            "peg_half_length_m": half_length, "true_entry_front_plane_x_m": -half_length,
             "actual_relative_peg_hole_yaw_rad": _yaw_error(task, 0.0),
             "lateral_target_delta_y_m": float(head_target[1] - base_head[1]),
             "actual_lateral_delta_y_m": float(_task_head(task)[1] - base_head[1]),
             "lateral_admission_eligible": lateral_ok,
            "desired_yaw_rad": yaw_target, "final_metrics": final, "last5_stability_metrics": window,
            "source_prefix_actions": prefix_actions.tolist(),
            "prefix_action_sha256": prefix_hash.hexdigest(),
            "full_action_sha256": hashlib.sha256(prefix_actions.tobytes() +
                np.asarray([r["command"]["action"] for r in records], dtype=np.float32).tobytes()).hexdigest(),
            "actions": [r["command"]["action"] for r in records], "trace": records,
            "last_two_observations": [x.tolist() for x in history[-2:]],
            "physical_state": capture_physical_state(task),
            "controller_mode": str(task.control_mode), "public_controller_state": jsonable(task.agent.controller.get_state()),
             "controller_calibration": calibration,
             "phase_controls": phases, "initial_history_sha256": hashlib.sha256(np.asarray(bank_row["observation_history"]).tobytes()).hexdigest()}


def _run_condition(env, scene, actions, bank_row, name, budget, settling_strategy="continuous", admission_mode="instant_velocity"):
    """Finalize every attempted case with a physical snapshot, including early exits."""
    task = env.unwrapped
    result = None
    error = None
    telemetry = {"executed_prefix_actions": [], "executed_initializer_actions": [],
                 "observation_history": [], "trace": [], "measurement_windows": {},
                 "control_measurements": [], "settling_strategy": settling_strategy, "admission_mode": admission_mode}
    try:
        result = _run_condition_body(env, scene, actions, bank_row, name, budget, telemetry)
    except Exception as exc:
        error = exc
    evidence = {"case_finalized": True, "condition": name,
                "source_prefix_controls": int(bank_row.get("prefix_controls", 0)),
                "source_prefix_actions": np.asarray(actions[:int(bank_row.get("prefix_controls", 0))],
                                                     dtype=np.float32).tolist(),
                "source_prefix_action_sha256_expected": bank_row.get("prefix_actions_sha256"),
                "actual_observation": None, "physical_state": None,
                "final_head_hole_xyz_m": None, "final_entry_clearance_m": None,
                "controller_calibration": None, "partial_case": error is not None,
                "executed_prefix_actions": telemetry["executed_prefix_actions"],
                "executed_initializer_actions": telemetry["executed_initializer_actions"],
                 "observation_history": telemetry["observation_history"],
                 "trace": telemetry["trace"], "measurement_windows": telemetry["measurement_windows"],
                 "admission_mode": admission_mode, "control_timestep_s": telemetry.get("control_timestep_s"),
        "phase_entry_after_withdrawal": telemetry.get("phase_entry_after_withdrawal")}
    evidence["last_pose_window_candidate"] = telemetry.get("pose_window_candidate", [])
    evidence["control_measurements"] = telemetry["control_measurements"]
    evidence["prefix_action_sha256"] = telemetry.get("prefix_action_sha256")
    evidence["controller_calibration"] = telemetry.get("controller_calibration")
    if telemetry["control_measurements"]:
        evidence["desired_head_hole_xyz_m"] = telemetry["control_measurements"][-1]["desired_head_hole_xyz_m"]
        evidence["desired_yaw_rad"] = telemetry["control_measurements"][-1]["desired_yaw_rad"]
        evidence["last5_control_measurements"] = telemetry["control_measurements"][-STABILIZE_WINDOW:]
    else:
        evidence["desired_head_hole_xyz_m"] = "unknown: target not computed before case stopped"
        evidence["desired_yaw_rad"] = "unknown: target not computed before case stopped"
        evidence["last5_control_measurements"] = []
    try:
        evidence["actual_observation"] = _as_np(env.get_obs()).reshape(-1).tolist()
        evidence["physical_state"] = capture_physical_state(task)
        evidence["final_head_hole_xyz_m"] = _task_head(task).tolist()
        try:
            half_length = float(task.peg_half_sizes[0, 0].detach().cpu())
            evidence["final_entry_clearance_m"] = entry_clearance(_task_head(task), half_length)
            evidence["peg_half_length_m"] = half_length
        except Exception:
            evidence["final_entry_clearance_m"] = "unknown"
        evidence["native_grasped"] = native_grasped(task)
        evidence["official_inserted"] = bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any())
        evidence["controller_mode"] = str(task.control_mode)
        evidence["public_controller_state"] = jsonable(task.agent.controller.get_state())
    except Exception as capture_exc:
        evidence["capture_error"] = {"type": type(capture_exc).__name__, "message": str(capture_exc)}
    if error is not None:
        error.case_evidence = jsonable(evidence)
        raise error
    result.update(evidence)
    return result


def run_bank(output, deadline_seconds=3600, selected_ids=None, strategies=("continuous",),
             max_total_controls=25200, admission_mode="instant_velocity"):
    output = Path(output).resolve()
    if not output.is_dir() or any((output / name).exists() for name in
                                  ("preregistration.json", "results.json", "journal.jsonl", "source_archive")):
        raise FileExistsError(f"output directory must be precreated wrapper-only and otherwise unused: {output}")
    overall_deadline = time.monotonic() + min(float(deadline_seconds), 3600)
    groups, envs = {}, []
    started = time.monotonic()
    budget = [0, overall_deadline, int(max_total_controls)]
    try:
        catalog = {"random": PegSceneCatalog(CATALOG_ROOT / "validation_random_scenes"),
                   "fixed": PegSceneCatalog(CATALOG_ROOT / "validation_fixed_scenes")}
        selected = {}
        for profile in PROFILES:
            rows = sorted((r for r in _bank_rows(profile) if r["eligible"]), key=lambda r: r["catalog_index"])
            if selected_ids is None:
                selected[profile] = rows[:6]
            else:
                wanted = list(selected_ids[profile])
                by_id = {int(row["catalog_index"]): row for row in rows}
                if any(index not in by_id for index in wanted):
                    raise ValueError(f"requested source-bank ID is not eligible for {profile}: {wanted}")
                selected[profile] = [by_id[index] for index in wanted]
        source_hashes = {"new_module": sha256(__file__), "bank_random": sha256(BANK / "random_bank/branches_reconciled.jsonl"),
                         "bank_fixed": sha256(BANK / "fixed_bank/branches.jsonl")}
        identities = {rel: sha256(ROOT / rel) for rel in REGISTERED_PATHS}
        for profile in PROFILES:
            for path in sorted((BANK / f"{profile}_bank").iterdir()):
                if path.is_file():
                    identities[str(path.relative_to(ROOT))] = sha256(path)
            catalog_path = CATALOG_ROOT / ("validation_random_scenes" if profile == "random" else "validation_fixed_scenes")
            for path in sorted(catalog_path.iterdir()):
                if path.is_file():
                    identities[str(path.relative_to(ROOT))] = sha256(path)
            for path in sorted(DATA_DIRS[profile].iterdir()):
                if path.is_file():
                    identities[str(path.relative_to(ROOT))] = sha256(path)
        checkpoint_refs = {
            "outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt":
                "067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea",
            "outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_25000.pt":
                "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df",
        }
        for rel, expected in checkpoint_refs.items():
            actual = sha256(ROOT / rel)
            if actual != expected:
                raise ValueError(f"frozen checkpoint reference digest mismatch: {rel}")
            identities[rel] = actual
        prereg = {"objective": "Build physical expert-reachable perturbed initializations outside the actual hole entry plane; no policy inference.",
                  "selected_catalog_ids": {p: [r["catalog_index"] for r in rows] for p, rows in selected.items()},
                  "source_bank_admitted_ids": {p: [r["catalog_index"] for r in sorted(
                      (row for row in _bank_rows(p) if row["eligible"]), key=lambda row: row["catalog_index"])]
                      for p in PROFILES},
                   "admitted_not_selected_ids": {p: [r["catalog_index"] for r in sorted(
                       (row for row in _bank_rows(p) if row["eligible"]), key=lambda row: row["catalog_index"])
                       if r["catalog_index"] not in {x["catalog_index"] for x in selected[p]}]
                       for p in PROFILES},
                  "source_catalog_denominator": 20,
                   "conditions": CONDITIONS, "settling_strategies": list(strategies), "admission_mode": admission_mode,
                  "candidate_count": sum(len(v) for v in selected.values()) * len(CONDITIONS) * len(strategies),
                  "condition_target": {"outside_gap_m": .020, "lateral_hole_y_m": .005, "world_z_yaw_rad": YAW_RAD},
                   "selection": ("explicit frozen source ID subset per profile; no replacements" if selected_ids is not None else "first six already-admitted source-bank indices per profile, sorted before new outcomes; no replacements"),
                  "source_hashes": source_hashes, "registered_inputs_and_sources": identities,
                  "model_weights_loaded": False, "policies_queried": False,
                  "frozen_weight_references_only": {
                      "base_ema_sha256": "067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea",
                      "constrain_random500_update25000_ema_sha256": "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df"},
                  "native_controller": "unchanged pd_ee_delta_pose; normalized action inverted from native config; closed-loop scripted diagnostic initializer only",
                  "budgets": {"controls_per_condition_including_prefix_max": 420, "scripted_controls_max": MAX_INIT_CONTROLS,
                              "withdraw_controls_max": WITHDRAW_CAP, "total_controls_max": int(max_total_controls),
                              "stage_wall_seconds_max": min(deadline_seconds, 3600),
                              "overall_wall_seconds_max": min(deadline_seconds, 3600)},
                  "calibration_protocol": {"fixed_tcp_latch": "after withdrawal, first pose-only qualifying state; copy actual world TCP position+quaternion once; immutable target for remaining settle controls",
                      "pose_only_latch": {"head_norm_m_max": HEAD_NORM_TOLERANCE_M,
                          "head_axis_m_max": HEAD_AXIS_TOLERANCE_M.tolist(), "yaw_error_deg_max": 1,
                          "grasp_required": True, "official_inserted": False},
                      "continuous_strategy": "original per-control current peg-to-TCP offset recomputation",
                      "signed_lateral_target_band_m": {"minus": [-.007, -.003], "plus": [.003, .007]},
                      "strategy_order": list(strategies), "condition_order": list(CONDITIONS),
                      "no_pose_or_velocity_teleports": True, "weights_hashed_not_loaded": True},
                   "stability": {"consecutive_controls": (POSE_WINDOW_SAMPLES if admission_mode == "pose_window" else 5), "head_error_norm_m_max": HEAD_NORM_TOLERANCE_M,
                                 "head_error_axis_m_max": HEAD_AXIS_TOLERANCE_M.tolist(), "yaw_error_deg_max": 1,
                                "arm_qvel_rad_s_max": .02, "peg_linear_m_s_max": .005, "peg_angular_rad_s_max": .05,
                                 "outside_entry_clearance_m": [.019, .021]},
                  "admission_criteria": ({"mode": "pose_window", "samples": 11, "intervals": 10,
                      "control_timestep_source": "actual env.control_timestep", "head_pairwise_diameter_m_max": .00025,
                      "peg_world_center_pairwise_diameter_m_max": .00025,
                      "peg_quaternion_rotational_diameter_deg_max": .25, "arm_joint_qpos_range_rad_max": .002,
                      "per_sample_head_norm_m_max": .003, "per_sample_head_axis_m_max": [.001,.002,.003],
                      "clearance_m": [.019,.021], "yaw_error_deg_max": 1,
                      "signed_lateral_delta_from_original_base_head_y_m": {"minus": [-.007,-.003], "plus": [.003,.007]},
                      "instantaneous_velocity": "recorded diagnostic only; no hard gate"} if admission_mode == "pose_window" else
                      {"mode": "instant_velocity", "samples": 5, "peg_linear_m_s_max": .005,
                       "peg_angular_rad_s_max": .05, "arm_qvel_rad_s_max": .02}),
                  "near_mouth_interpretation_correction": "Old bank head x near -0.04 to -0.05 m lies inside peg entry plane x=-peg_half_length; official success x>=-0.015 m is near box center. This new bank withdraws outside the physical front plane.",
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        prereg["runtime"] = {"python": os.sys.version, "torch": torch.__version__, "numpy": np.__version__}
        (output / "preregistration.json").write_text(json.dumps(jsonable(prereg), indent=2, sort_keys=True)+"\n")
        for rel in REGISTERED_PATHS:
            dest = output / "source_archive" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / rel).read_bytes())
        (output / "source_closure.json").write_text(json.dumps(identities, indent=2, sort_keys=True)+"\n")
        (output / "pre_native_input_hashes.json").write_text(json.dumps(identities, indent=2, sort_keys=True)+"\n")
        wrapper_path = output / "attempt_001.sh"
        if wrapper_path.exists():
            wrapper_sha = sha256(wrapper_path)
            archived_wrapper = output / "source_archive/outer_attempt_001.sh"
            archived_wrapper.write_bytes(wrapper_path.read_bytes())
            prereg["outer_wrapper_sha256"] = wrapper_sha
            closure = dict(identities)
            closure["outer_attempt_001.sh"] = wrapper_sha
            (output / "source_closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True)+"\n")
            prereg["source_closure_sha256"] = sha256(output / "source_closure.json")
            (output / "preregistration.json").write_text(json.dumps(jsonable(prereg), indent=2, sort_keys=True)+"\n")
            (output / "preregistration.sha256").write_text(sha256(output / "preregistration.json")+"  preregistration.json\n")
        (output / "preregistration.sha256").write_text(sha256(output / "preregistration.json")+"  preregistration.json\n")
        groups = {p: [] for p in PROFILES}
        with (output / "journal.jsonl").open("a") as journal:
            for profile in PROFILES:
                for row in selected[profile]:
                    index = int(row["catalog_index"])
                    scene = catalog[profile].load(index)
                    data_path = DATA_DIRS[profile] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
                    actions = _source_actions(data_path, row["source_episode_id"])
                    for condition in CONDITIONS:
                      for strategy in strategies:
                        if time.monotonic() >= overall_deadline:
                            raise TimeoutError("overall initialization deadline exceeded")
                        env = gym.make(peg_orientation.RANDOM_ENV_ID if profile == "random" else peg_orientation.ENV_ID,
                            obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                            reconfiguration_freq=1, max_episode_steps=600)
                        envs.append(env)
                        try:
                            result = _run_condition(env, scene, actions, row, condition, budget, strategy, admission_mode)
                            entry = {"profile": profile, "catalog_index": index, "source_scene_id": row["scene_id"],
                                     "condition": condition, "strategy": strategy,
                                     "result": jsonable(result), "total_native_controls": budget[0]}
                        except Exception as exc:
                            entry = {"profile": profile, "catalog_index": index, "condition": condition, "strategy": strategy,
                                     "error": {"type": type(exc).__name__, "message": str(exc),
                                               "partial_case": getattr(exc, "case_evidence", None)},
                                     "total_native_controls": budget[0]}
                            journal.write(json.dumps(jsonable(entry), allow_nan=False)+"\n"); journal.flush(); os.fsync(journal.fileno())
                            raise
                        finally:
                            env.close(); envs.remove(env)
                        groups[profile].append(entry)
                        journal.write(json.dumps(jsonable(entry), allow_nan=False, sort_keys=True)+"\n")
                        journal.flush(); os.fsync(journal.fileno())
        result = {"candidates": sum(len(v) for v in groups.values()), "native_controls": budget[0],
                  "elapsed_seconds": time.monotonic()-started, "groups": groups,
                   "admitted_counts": {p: {s: {c: sum(bool(x["result"]["admitted"]) for x in rows
                       if x["condition"] == c and x.get("strategy") == s) for c in CONDITIONS}
                       for s in strategies} for p, rows in groups.items()},
                  "interpretation": "Initialization feasibility only; no policy queries, no Base/CN rollouts or comparative outcomes."}
        (output / "results.json").write_text(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False)+"\n")
        post_identities = {rel: sha256(ROOT / rel) for rel in REGISTERED_PATHS}
        for rel in checkpoint_refs:
            post_identities[rel] = sha256(ROOT / rel)
        for profile in PROFILES:
            for path in sorted((BANK / f"{profile}_bank").iterdir()):
                if path.is_file():
                    post_identities[str(path.relative_to(ROOT))] = sha256(path)
            catalog_path = CATALOG_ROOT / ("validation_random_scenes" if profile == "random" else "validation_fixed_scenes")
            for path in sorted(catalog_path.iterdir()):
                if path.is_file():
                    post_identities[str(path.relative_to(ROOT))] = sha256(path)
            for path in sorted(DATA_DIRS[profile].iterdir()):
                if path.is_file():
                    post_identities[str(path.relative_to(ROOT))] = sha256(path)
        (output / "post_native_input_hashes.json").write_text(json.dumps(post_identities, indent=2, sort_keys=True)+"\n")
        if identities != post_identities:
            raise ValueError("registered source or input hashes changed during native initialization")
        return result
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc),
            "physical_controls_including_prefix_replay": budget[0],
            "elapsed_seconds": time.monotonic()-started,
            "completed_case_count": sum(len(rows) for rows in groups.values())}, indent=2)+"\n")
        raise
    finally:
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-random-ids", type=int, nargs="*")
    parser.add_argument("--selected-fixed-ids", type=int, nargs="*")
    parser.add_argument("--strategies", nargs="+", choices=("continuous", "fixed_tcp"), default=["continuous"])
    parser.add_argument("--max-total-controls", type=int, default=25200)
    parser.add_argument("--admission-mode", choices=("instant_velocity", "pose_window"), default="instant_velocity")
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    args = parser.parse_args()
    selected_ids = None
    if args.selected_random_ids is not None or args.selected_fixed_ids is not None:
        if args.selected_random_ids is None or args.selected_fixed_ids is None:
            parser.error("both selected ID lists must be supplied")
        selected_ids = {"random": args.selected_random_ids, "fixed": args.selected_fixed_ids}
    run_bank(args.output, deadline_seconds=args.deadline_seconds, selected_ids=selected_ids,
              strategies=tuple(args.strategies), max_total_controls=args.max_total_controls,
              admission_mode=args.admission_mode)


if __name__ == "__main__":
    main()
