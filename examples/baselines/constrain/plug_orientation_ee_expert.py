"""Experimental closed-loop PlugCharger teacher using native EE actions only."""

import numpy as np
import sapien.core as sapien
import trimesh
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat
import json
import random
from pathlib import Path
import traceback
import time

from examples.baselines.constrain.peg_orientation_near_hole_perturb import (
    PoseValue, _native_config, _pose_value, _rotation, native_pose_action,
)

MAX_CONTROLS = 600
MAX_ALIGN_CONTROLS = 180
MAX_INSERT_CONTROLS = 160
PHASES = ("reach", "grasp", "lift", "align", "insert", "settle")
GRASP_TILT_DEG = 15.0
GRIPPER_CLOSE_STEPS = 12
TEACHER_ALGORITHM = "native_ee_stream_final_target_hold_v4_object_pose_lift_tracking"
POSITION_TOLERANCE = 0.0015
ROTATION_TOLERANCE = 0.03


def _pose(raw):
    return sapien.Pose(np.asarray(raw[:3], dtype=float), np.asarray(raw[3:], dtype=float))


def charger_relative_tcp_target(charger_pose, tcp_pose, desired_charger_pose):
    """Apply the observed charger→TCP transform to a desired full charger pose."""
    return _pose_array(
        _pose(desired_charger_pose) * _pose(charger_pose).inv() * _pose(tcp_pose)
    )


def _pose_array(pose):
    return np.r_[np.asarray(pose.p), np.asarray(pose.q)]


def pose_error(current, target):
    dp = float(np.linalg.norm(np.asarray(current[:3]) - np.asarray(target[:3])))
    dq = Rotation.from_quat(np.r_[target[4:7], target[3]]) * Rotation.from_quat(
        np.r_[current[4:7], current[3]]
    ).inv()
    return dp, float(dq.magnitude())


def _tcp(task):
    return np.asarray(task.agent.tcp.pose.raw_pose[0].detach().cpu(), dtype=float).copy()


def _actor_pose(actor):
    return np.asarray(actor.pose.raw_pose[0].detach().cpu(), dtype=float).copy()


def _step(env, target, gripper, trace, phase, *, waypoint_index=None,
          planned_target=None, target_hold=False):
    if len(trace) >= MAX_CONTROLS:
        raise RuntimeError("native teacher exceeded 600-control episode cap")
    task = env.unwrapped
    controller, geometry = _native_config(env)
    current = _pose_value(task.agent.tcp.pose)
    current_raw = _tcp(task)
    root = _rotation(_pose_value(task.agent.robot.pose).rotation)
    target = _bounded_target(current, _pose(target))
    bounded_raw = np.r_[target.position, target.rotation].astype(float)
    cmd = native_pose_action(current, target, geometry, root)
    # Native pose encoder normalizes/clips the controller command. Refuse to
    # silently issue a physically different target on these small-step phases.
    if cmd["action_clipped"]:
        raise RuntimeError(f"native EE command clipped in {phase}")
    action = np.r_[cmd["action"][:6], float(gripper)].astype(np.float32)
    row = {
        "phase": phase, "action": action.copy(),
        "measured_tcp_before": current_raw,
        "planned_world_tcp_target": np.asarray(
            planned_target if planned_target is not None else target.p if hasattr(target, "p") else target.position,
            dtype=float).copy(),
        "bounded_encoded_command_target": bounded_raw,
        "waypoint_index": waypoint_index, "target_hold": bool(target_hold),
        "controller_config": {"frame": geometry.frame, "use_delta": bool(geometry.use_delta),
                              "pos_lower": np.asarray(geometry.pos_lower).copy(),
                              "pos_upper": np.asarray(geometry.pos_upper).copy(),
                              "rot_lower": np.asarray(geometry.rot_lower).copy(),
                              "rot_upper": np.asarray(geometry.rot_upper).copy()},
        "action_clipped": bool(cmd["action_clipped"]),
        "control_attempted": True, "control_completed": False,
    }
    trace.append(row)
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except BaseException as exc:
        row["step_exception"] = f"{type(exc).__name__}: {exc}"
        raise
    measured = _tcp(task)
    evaluation = task.evaluate()
    row.update({
        "tcp_pose": measured, "measured_tcp_after": measured.copy(),
        "position_error_m": pose_error(measured, bounded_raw)[0],
        "rotation_error_rad": pose_error(measured, bounded_raw)[1],
        "charger_pose": _actor_pose(task.charger),
        "orientation_hint": np.asarray(task.get_orientation_hint(), dtype=float).copy(),
        "native_is_grasping": bool(task.agent.is_grasping(task.charger)),
        "control_completed": True,
        "official_success": bool(np.asarray(evaluation["success"]).any()),
        "official_position_error_m": float(np.asarray(evaluation["obj_to_goal_dist"]).reshape(-1)[0]),
        "official_rotation_error_rad": float(np.asarray(evaluation["obj_to_goal_angle"]).reshape(-1)[0]),
    })
    # TimeLimit truncation ends this recorded episode but is not itself a planned
    # waypoint tracking failure; terminate is an environment-level hard stop.
    return obs, bool(np.asarray(terminated).any()), info


def _bounded_target(current, desired):
    current_position = current.position if hasattr(current, "position") else current.p
    current_rotation = current.rotation if hasattr(current, "rotation") else current.q
    desired_position = desired.position if hasattr(desired, "position") else desired.p
    desired_rotation = desired.rotation if hasattr(desired, "rotation") else desired.q
    delta = np.asarray(desired_position) - np.asarray(current_position)
    delta = np.clip(delta, -0.005, 0.005)
    current_rot, desired_rot = _rotation(current_rotation), _rotation(desired_rotation)
    rotation_delta = (desired_rot * current_rot.inv()).as_rotvec()
    norm = np.linalg.norm(rotation_delta)
    if norm > 0.03:
        rotation_delta *= 0.03 / norm
    rotation = Rotation.from_rotvec(rotation_delta) * current_rot
    q = rotation.as_quat()
    return PoseValue(np.asarray(current_position) + delta, np.r_[q[3], q[:3]])


def _fk_waypoint_targets(task, qpos_path):
    robot = task.agent.robot
    model = robot.create_pinocchio_model()
    link_index = task.agent.tcp.index
    current_q = robot.get_qpos()[0].detach().cpu().numpy().copy()
    root = task.agent.robot.pose
    targets = []
    for arm_q in np.asarray(qpos_path):
        full_q = current_q.copy()
        full_q[:7] = arm_q[:7]
        model.compute_forward_kinematics(full_q)
        local = model.get_link_pose(link_index)
        world = root * type(root).create(local)
        targets.append(np.asarray(world.raw_pose[0].detach().cpu(), dtype=float).copy())
    return targets


def _seed_planner(seed):
    import mplib
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    mplib.set_global_seed(int(seed))


def _plan_arm_path(env, target_pose, planner_seed):
    from examples.baselines.constrain.peg_orientation_preflight import install_mplib_base_pose_compat
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    install_mplib_base_pose_compat()
    _seed_planner(planner_seed)
    task = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=task.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5,
    )
    try:
        result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
        if isinstance(result, int) and result == -1:
            raise RuntimeError("MPLib screw planning failed")
        if result.get("status") != "Success":
            raise RuntimeError(f"MPLib screw planning status {result.get('status')}")
        return _fk_waypoint_targets(task, result["position"])
    finally:
        planner.close()


def _follow_planned_targets(env, targets, gripper, trace, phase):
    if not targets:
        raise RuntimeError(f"MPLib returned an empty {phase} path")
    for waypoint_index, target in enumerate(targets):
        _, done, _ = _step(env, target, gripper, trace, phase,
                           waypoint_index=waypoint_index, planned_target=target)
        if done:
            return None
    # Native EE control is closed-loop per target. Intermediate FK points are
    # streamed once; tracking quality is diagnostic, not a planner-path gate.
    target = np.asarray(targets[-1], dtype=float).copy()
    for _ in range(10):
        _, done, _ = _step(env, target, gripper, trace, phase,
                           waypoint_index=len(targets) - 1, planned_target=target,
                           target_hold=True)
        if done:
            return None
    return target


def solve(env, seed, target_yaw, *, control_cap=MAX_CONTROLS, trace=None):
    """Reset and execute a privileged-goal online teacher in pd_ee_delta_pose."""
    from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb

    if trace is None:
        trace = []
    if control_cap != MAX_CONTROLS:
        raise ValueError("the preregistered native cap is fixed at 600 controls")
    planner_seed = int(seed) + 1
    _seed_planner(planner_seed)
    env.reset(seed=int(seed), options={"target_yaw": float(target_yaw)})
    _seed_planner(planner_seed)
    task = env.unwrapped
    if task.control_mode != "pd_ee_delta_pose":
        raise ValueError(f"teacher requires pd_ee_delta_pose, got {task.control_mode}")

    finger_length = 0.025
    obb = trimesh.primitives.Box(
        extents=np.asarray(task._base_size) * 2,
        transform=task.charger_base_pose.sp.to_transformation_matrix(),
    )
    approach = np.array([0., 0., -1.])
    closing = task.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approach, target_closing=closing, depth=finger_length
    )
    grasp = task.agent.build_grasp_pose(approach, grasp_info["closing"], grasp_info["center"])
    grasp = grasp * sapien.Pose(q=euler2quat(0, np.deg2rad(GRASP_TILT_DEG), 0))
    reach = grasp * sapien.Pose([0, 0, -0.05])

    for phase, pose, grip in (("reach", reach, 1.), ("grasp", grasp, 1.)):
        path = _plan_arm_path(env, pose, planner_seed)
        planned_final_tcp = _follow_planned_targets(env, path, grip, trace, phase)
        if planned_final_tcp is None:
            return _result(env, trace, False, f"{phase}_tracking_failed")

    for _ in range(GRIPPER_CLOSE_STEPS):
        _step(env, planned_final_tcp, -1., trace, "grasp_close",
              waypoint_index=len(path) - 1, planned_target=planned_final_tcp,
              target_hold=True)
    charger = _actor_pose(task.charger)
    lifted = charger.copy()
    lifted[2] += 0.03
    if not _charger_target_loop(env, lifted, -1., trace, "lift", 80):
        return _result(env, trace, False, "lift_tracking_failed")

    goal = _pose_array(task.goal_pose.sp)
    standoff = goal.copy()
    standoff[:3] += _rotation(goal[3:]).apply([-0.05, 0, 0])
    if not _charger_target_loop(env, standoff, -1., trace, "align", MAX_ALIGN_CONTROLS):
        return _result(env, trace, False, "align_failed")
    if not _charger_target_loop(env, goal, -1., trace, "insert", MAX_INSERT_CONTROLS,
                                insertion=True):
        return _result(env, trace, False, "insert_failed")
    for _ in range(10):
        charger_now, tcp_now = _actor_pose(task.charger), _tcp(task)
        tcp_target = charger_relative_tcp_target(charger_now, tcp_now, goal)
        _step(env, tcp_target, -1., trace, "settle")
    success = bool(np.asarray(task.evaluate()["success"]).any())
    return _result(env, trace, success, "native_success" if success else "official_success_false")


def _charger_target_loop(env, desired_charger, gripper, trace, phase, budget, insertion=False):
    task = env.unwrapped
    gap = 0.05
    start_charger = _actor_pose(task.charger)
    for _ in range(budget):
        if insertion:
            desired = _pose_array(_pose(desired_charger) * sapien.Pose([-gap, 0, 0]))
        else:
            desired = np.asarray(desired_charger, dtype=float)
        charger = _actor_pose(task.charger)
        tcp = _tcp(task)
        target = charger_relative_tcp_target(charger, tcp, desired)
        _step(env, target, gripper, trace, phase)
        live_charger = _actor_pose(task.charger)
        pos, rot = pose_error(live_charger, desired)
        trace[-1].update({
            "charger_target_pose": np.asarray(desired, dtype=float).copy(),
            "charger_position_error_m": pos,
            "charger_rotation_error_rad": rot,
            "native_is_grasping_diagnostic": bool(task.agent.is_grasping(task.charger)),
        })
        if phase == "lift":
            trace[-1].update({
                "lift_start_charger_height_m": float(start_charger[2]),
                "lift_target_charger_height_m": float(desired_charger[2]),
                "lift_current_charger_height_m": float(live_charger[2]),
                "lift_height_progress_m": float(live_charger[2] - start_charger[2]),
            })
        if pos <= POSITION_TOLERANCE and rot <= ROTATION_TOLERANCE:
            if not insertion:
                return True
            gap = max(0., gap - 0.001)
            if gap == 0.:
                official = task.evaluate()
                if bool(np.asarray(official["success"]).any()):
                    return True
    return False


def _result(env, trace, success, reason):
    task = env.unwrapped
    evaluation = task.evaluate()
    lift_rows = [row for row in trace if row.get("phase") == "lift"]
    lift = {}
    if lift_rows:
        first, last = lift_rows[0], lift_rows[-1]
        lift = {
            "lift_start_charger_height_m": float(first["lift_start_charger_height_m"]),
            "lift_end_charger_height_m": float(last["lift_current_charger_height_m"]),
            "lift_target_charger_height_m": float(last["lift_target_charger_height_m"]),
            "lift_height_progress_m": float(last["lift_current_charger_height_m"]
                                              - first["lift_start_charger_height_m"]),
            "lift_final_position_error_m": float(last["charger_position_error_m"]),
            "lift_final_rotation_error_rad": float(last["charger_rotation_error_rad"]),
        }
    return {"success": bool(success), "reason": reason, "controls": len(trace), **lift,
            "official_success": bool(np.asarray(evaluation["success"]).any()),
            "official_position_error_m": float(np.asarray(evaluation["obj_to_goal_dist"]).reshape(-1)[0]),
            "official_rotation_error_rad": float(np.asarray(evaluation["obj_to_goal_angle"]).reshape(-1)[0]),
            "trace": trace}


def run_case(seed, output_dir, target_yaw=0.0, deadline=None):
    """Capture one online teacher and independently replay its exact recorded actions."""
    import gymnasium as gym
    import h5py
    import torch
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common
    from examples.baselines.constrain import plug_orientation

    def enforce_deadline(env):
        original_step = env.step
        def step(action):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("registered collection wall-clock cap exhausted before control")
            return original_step(action)
        env.step = step
        return env

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = enforce_deadline(gym.make(plug_orientation.ENV_ID, obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                   reward_mode="sparse", max_episode_steps=1000))
    recorder = RecordEpisode(env, output_dir=str(output), trajectory_name="expert",
                             save_trajectory=True, save_video=False, save_on_reset=False,
                              record_env_state=True,
                               source_type="experimental_online_ee_teacher_v4",
                               source_desc=("Privileged-goal teacher v4: stream planned FK targets, "
                                           "hold final planned TCP target, then close; native 7D EE actions; "
                                           "native grasp classifier diagnostic only, charger full-pose tracking gates lift"))
    trace = []
    expert_started = time.monotonic()
    try:
        result = solve(recorder, seed, target_yaw, trace=trace)
        recorder.flush_trajectory(save=True)
    except BaseException as exc:
        recorder.flush_trajectory(save=True)
        partial = {"seed": int(seed), "expert_controls": len(trace),
                   "stage": "expert_failed_before_replay", "replay_status": "not_started",
                   "replay_controls": 0,
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(),
                    "trace": [_jsonable(row) for row in trace]}
        (output / "partial_failure.json").write_text(
            json.dumps(partial, indent=2, allow_nan=False) + "\n"
        )
        raise
    finally:
        recorder.close()
    expert_elapsed = time.monotonic() - expert_started
    source = output / "expert.h5"
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as h5:
        traj = h5["traj_0"]
        actions = np.asarray(traj["actions"], dtype=np.float32).copy()
        states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
        obs_rows = trajectory_utils.dict_to_list_of_dicts(traj["obs"])
    if actions.shape != (len(trace), 7) or not np.isfinite(actions).all():
        raise ValueError(f"recorded online actions mismatch trace: {actions.shape} vs {len(trace)}")
    if len(states) != len(actions) + 1 or len(obs_rows) != len(actions) + 1:
        raise ValueError("online capture must contain T+1 states and observations")
    if not all(_finite_tree(row) for row in states):
        raise ValueError("online recorded states contain non-finite values")
    flat_observations = [_observation_vector(row, common) for row in obs_rows]
    if any(row.shape[-1] != 46 or not np.isfinite(row).all() for row in flat_observations):
        raise ValueError("online base observations must be finite 46-D state vectors")
    replay_env = enforce_deadline(gym.make(plug_orientation.ENV_ID, obs_mode="state",
                          control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                          reward_mode="sparse", max_episode_steps=1000))
    replay_dir = output / "replay"
    replay_dir.mkdir()
    replay_started = time.monotonic()
    replay_recorder = RecordEpisode(
        replay_env, output_dir=str(output / "replay"), trajectory_name="replay",
        save_trajectory=True, save_video=False, save_on_reset=False,
        record_env_state=True, source_type="independent_exact_ee_action_replay",
        source_desc="Unmodified recorded online v4 7D actions from canonical initial state",
    )
    try:
        replay_recorder.reset(seed=int(seed), options={"target_yaw": float(target_yaw)})
        replay_initial = replay_recorder.unwrapped.get_state_dict()
        source_initial = common.batch(states[0])
        if not _tree_close(source_initial, replay_initial, 1e-5):
            raise ValueError("fresh seeded replay state differs from expert state 0")
        replay_obs0 = replay_recorder.unwrapped.get_obs(unflattened=True)
        replay_flat = _observation_vector(replay_obs0, common)
        if replay_flat.shape[-1] != 46 or not np.isfinite(replay_flat).all():
            raise ValueError("replay actual initial observation is not finite 46-D")
        initial_obs_error = float(np.max(np.abs(replay_flat - flat_observations[0])))
        if initial_obs_error > 1e-5:
            raise ValueError(f"fresh seeded replay observation differs from expert obs 0: {initial_obs_error}")
        for action in actions:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("registered collection wall-clock cap exhausted before replay control")
            replay_recorder.step(action.copy())
        replay_recorder.flush_trajectory(save=True)
        replay_eval = replay_recorder.unwrapped.evaluate()
        replay_summary = {
            "success": bool(torch.as_tensor(replay_eval["success"]).any()),
            "official_position_error_m": float(torch.as_tensor(replay_eval["obj_to_goal_dist"]).reshape(-1)[0]),
            "official_rotation_error_rad": float(torch.as_tensor(replay_eval["obj_to_goal_angle"]).reshape(-1)[0]),
        }
    finally:
        replay_recorder.close()
    replay_elapsed = time.monotonic() - replay_started
    replay_h5 = replay_dir / "replay.h5"
    with h5py.File(replay_h5, "r") as h5:
        replay_traj = h5["traj_0"]
        replay_actions = np.asarray(replay_traj["actions"], dtype=np.float32)
        replay_states = trajectory_utils.dict_to_list_of_dicts(replay_traj["env_states"])
        replay_obs = trajectory_utils.dict_to_list_of_dicts(replay_traj["obs"])
    if (not np.array_equal(actions, replay_actions) or replay_actions.shape != (len(actions), 7)
            or len(replay_states) != len(actions) + 1 or len(replay_obs) != len(actions) + 1):
        raise ValueError("independent replay did not preserve exact actions and T+1 alignment")
    if not all(_finite_tree(row) for row in replay_states):
        raise ValueError("replay states contain non-finite values")
    replay_obs_flat = [_observation_vector(row, common) for row in replay_obs]
    if any(row.shape[-1] != 46 or not np.isfinite(row).all() for row in replay_obs_flat):
        raise ValueError("replay observations are not finite 46-D vectors")
    replay_result = {
        **replay_summary, "controls": int(len(replay_actions)),
        "action_shape": list(replay_actions.shape), "states_t_plus_one": len(replay_states),
        "observations_t_plus_one": len(replay_obs), "base_observation_dim": 46,
        "initial_observation_match_max_abs": float(np.max(np.abs(replay_flat-flat_observations[0]))),
        "actions_exact_match": True,
    }
    payload = {"seed": int(seed), "target_yaw": float(target_yaw), "teacher_algorithm": TEACHER_ALGORITHM,
               "expert": {k: v for k, v in result.items() if k != "trace"},
               "replay": replay_result,
               "replay_status": "replayed" if result["success"] else "diagnostic_expert_failed",
                "expert_controls": len(actions),
                "replay_controls": len(actions), "expert_elapsed_seconds": expert_elapsed,
                "replay_elapsed_seconds": replay_elapsed, "trace": [
                    _jsonable(row) for row in trace]}
    (output / "case.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _tree_close(left, right, atol):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _tree_close(left[key], right[key], atol) for key in left
        )
    return np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=0)


def _observation_vector(observation, common):
    """Normalize a recorded flat state row or a genuinely structured observation."""
    if isinstance(observation, dict):
        # BaseEnv.get_obs(unflattened=True) already returns batched tensors.
        value = common.flatten_state_dict(observation, use_torch=True)
        return value.detach().cpu().numpy()
    if hasattr(observation, "detach"):
        observation = observation.detach().cpu().numpy()
    return np.asarray(observation)


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(child) for child in value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return bool(np.isfinite(np.asarray(value)).all())
    except (TypeError, ValueError):
        return True


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
