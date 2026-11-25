"""Bounded live-state expert continuation for Peg orientation diagnostics."""

import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import to_sapien_pose


STATE_TOLERANCE = 1e-5
CONTROLLER_MODE = "pd_joint_pos"


def _task(env):
    return env.unwrapped


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).copy()


def capture_physical_state(task):
    """Capture physical branch anchors via public task/agent pose and state APIs."""
    return {
        "robot_qpos": _array(task.agent.robot.get_qpos()),
        "robot_qvel": _array(task.agent.robot.get_qvel()),
        "robot_root_pose": _array(task.agent.robot.pose.raw_pose),
        "tcp_pose": _array(task.agent.tcp.pose.raw_pose),
        "peg_pose": _array(task.peg.pose.raw_pose),
        "hole_pose": _array(task.box_hole_pose.raw_pose),
    }


def native_grasped(task):
    value = task.agent.is_grasping(task.peg)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).all())


class _NativeSuccess(Exception):
    pass


class _ControlCap(Exception):
    pass


class _NativeTerminal(Exception):
    pass


class _CountedJointEnv:
    """Public env facade that counts solver controls and uses batched joint actions."""

    def __init__(self, env, max_controls):
        self._env = env
        self.max_controls = int(max_controls)
        self.controls = 0

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        if self.controls >= self.max_controls:
            raise _ControlCap
        action = np.asarray(action, dtype=np.float64)
        if action.shape == (8,):
            action = action[None, :]
        if action.shape != (1, 8):
            raise ValueError(f"expected batched pd_joint_pos [arm7, gripper], got {action.shape}")
        result = self._env.step({"control_mode": CONTROLLER_MODE, "action": action})
        self.controls += 1
        info = result[-1]
        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.detach().cpu().numpy()
        if bool(np.asarray(success).any()):
            raise _NativeSuccess
        if bool(np.asarray(result[2]).any()) or bool(np.asarray(result[3]).any()):
            raise _NativeTerminal
        return result


def run_live_expert_continuation(
    env,
    planner_factory,
    continuation,
    *,
    max_controls,
    state_reader=capture_physical_state,
    grasp_check=native_grasped,
    tolerance=STATE_TOLERANCE,
):
    """Switch the live held-peg branch to joint control and continue without reset."""
    if max_controls < 0:
        raise ValueError("max_controls must be nonnegative")
    task = _task(env)
    if not grasp_check(task):
        raise ValueError("live expert continuation requires an already-grasped peg")

    mode_before = task.control_mode
    physical_before = state_reader(task)
    task.agent.set_control_mode(CONTROLLER_MODE)
    task.agent.controller.reset()
    physical_after = state_reader(task)
    if not grasp_check(task):
        raise RuntimeError("controller switch lost the native grasp without a physical control")
    errors = {}
    for key, before in physical_before.items():
        after = physical_after[key]
        if np.shape(before) != np.shape(after):
            raise RuntimeError(f"controller switch changed state shape for {key}")
        errors[key] = float(np.max(np.abs(np.asarray(after) - np.asarray(before)))) if np.size(before) else 0.0
    max_error = max(errors.values(), default=0.0)
    if max_error > tolerance:
        raise RuntimeError(f"controller switch changed physical branch state: {errors}")

    counted_env = _CountedJointEnv(env, max_controls)
    planner = None
    status = "plan_complete"
    plan_result = None
    try:
        planner = planner_factory(counted_env)
        planner.gripper_state = planner.CLOSED
        try:
            plan_result = continuation(task, planner)
            if isinstance(plan_result, (int, np.integer)) and plan_result == -1:
                status = "plan_failure"
        except _NativeSuccess:
            status = "native_success"
        except _ControlCap:
            status = "control_cap"
        except _NativeTerminal:
            status = "native_terminal"
    finally:
        if planner is not None:
            planner.close()

    return {
        "status": status,
        "plan_result": plan_result,
        "controls": counted_env.controls,
        "controller_mode_before": mode_before,
        "controller_mode_after": task.control_mode,
        "controller_switch_physical_steps": 0,
        "switch_state_max_abs_error": max_error,
        "switch_state_errors": errors,
        "native_grasp_before": True,
        "native_grasp_after": bool(grasp_check(task)),
        "planner_gripper_state": planner.gripper_state if planner is not None else None,
    }


def run_official_geometry_continuation(task, planner):
    """Apply official Peg planner waypoints from the current grasp transform."""
    tcp_pose = to_sapien_pose(task.agent.tcp.pose)
    peg_pose = to_sapien_pose(task.peg.pose)
    goal_pose = to_sapien_pose(task.goal_pose)
    # Preserve the actual current peg-to-TCP transform rather than assuming the
    # grasp pose from the expert's reset-time pick phase.
    insert_pose = goal_pose * peg_pose.inv() * tcp_pose
    half_length = float(_array(task.peg_half_sizes)[0, 0])
    offset = sapien.Pose([-0.01 - half_length, 0, 0])
    pre_insert_pose = insert_pose * offset
    result = planner.move_to_pose_with_screw(pre_insert_pose)
    if result == -1:
        return -1
    for _ in range(3):
        delta_pose = goal_pose * offset * to_sapien_pose(task.peg.pose).inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        result = planner.move_to_pose_with_screw(pre_insert_pose)
        if result == -1:
            return -1
    result = planner.move_to_pose_with_screw(insert_pose * sapien.Pose([0.05, 0, 0]))
    return result


def build_official_planner(env):
    """Construct the stock Panda solver after the supported live controller switch."""
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    from examples.baselines.constrain.peg_orientation_preflight import (
        install_mplib_base_pose_compat,
    )

    install_mplib_base_pose_compat()
    task = env.unwrapped
    return PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=task.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
