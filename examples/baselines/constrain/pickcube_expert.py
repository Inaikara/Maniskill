"""Bounded PickCube direct and same-base-side expert execution."""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

import examples.baselines.constrain  # register the obstacle task
from examples.baselines.constrain.layout_geometry import (
    LayoutGeometryConfig,
    sample_same_side_waypoint,
    validate_layout,
)
from examples.baselines.constrain.pickcube_obstacle import ENV_ID
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


@dataclass(frozen=True)
class PickCubeExpertConfig:
    max_steps: int = 500
    max_steps_per_waypoint: int = 80
    waypoint_tolerance: float = 0.006
    action_cap: float = 0.35
    lift_height: float = 0.12
    grasp_steps: int = 10
    settle_steps: int = 10
    seed: int = 0


def snapshot_execution_state(env):
    """Capture native simulation state and controller state separately."""
    base = env.unwrapped
    state = base.get_state_dict()
    controller = base.agent.controller.get_state()
    return state, controller


def restore_execution_state(env, snapshot):
    state, controller = snapshot
    base = env.unwrapped
    _set_state_dict_exact(base, state)
    base.agent.controller.set_state(controller)
    # Continuing a restored episode must not inherit its previous rollout's
    # Gym TimeLimit/native elapsed-step counters.
    for wrapper in (env, base):
        if hasattr(wrapper, "_elapsed_steps"):
            wrapper._elapsed_steps = 0
    if torch.is_tensor(getattr(base, "_elapsed_steps", None)):
        base._elapsed_steps.zero_()


def run_pickcube_expert(
    env,
    *,
    cube_xy,
    goal_xy,
    detour=False,
    obstacle_enabled=False,
    obstacle_center=(0.8, 0.8),
    config=PickCubeExpertConfig(),
    initial_snapshot=None,
    deterministic_reset_restore=False,
    layout_config=LayoutGeometryConfig(
        table_bounds=(-0.25, -0.25, 0.25, 0.35),
        min_transport_distance=0.14,
        min_base_line_distance=0.06,
        waypoint_offset_range=(0.13, 0.17),
        waypoint_lead_fraction=0.25,
        max_attempts=32,
    ),
):
    """Execute one Panda task and return actions, T+1 state/observations and trace."""
    base = env.unwrapped
    expert_started = time.monotonic()
    cube_xy = np.asarray(cube_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    base_xy = base.agent.robot.pose.p[0, :2].detach().cpu().numpy()
    layout = validate_layout(cube_xy, goal_xy, base_xy, layout_config)
    if not layout.valid:
        raise ValueError(f"invalid_pickcube_layout:{layout.reason}")
    waypoint = None
    waypoint_attempts = 0
    if detour:
        sampled = sample_same_side_waypoint(
            cube_xy, goal_xy, base_xy, np.random.default_rng(config.seed), layout_config
        )
        if sampled.waypoint_xy is None:
            raise ValueError(f"invalid_pickcube_waypoint:{sampled.reason}")
        waypoint = sampled.waypoint_xy
        waypoint_attempts = sampled.attempts

    if initial_snapshot is None:
        obs, reset_info = env.reset(
            seed=config.seed,
            options={
                "cube_xy": cube_xy,
                "goal_pos": [*goal_xy, 0.12],
                "obstacle_enabled": obstacle_enabled,
                "obstacle_center": obstacle_center,
            },
        )
    else:
        state, controller = initial_snapshot
        state = _clone_tree(state)
        if deterministic_reset_restore:
            obs, reset_info = env.reset(
                seed=config.seed,
                options={"cube_xy": cube_xy, "goal_pos": [*goal_xy, 0.12],
                         "obstacle_enabled": obstacle_enabled,
                         "obstacle_center": obstacle_center},
            )
            base.agent.controller.set_state(_clone_tree(controller))
        else:
            if obstacle_enabled:
                extra = state["pickcube_obstacle"]
                extra[0, :2] = torch.as_tensor(obstacle_center, dtype=extra.dtype, device=extra.device)
                extra[0, 3] = 1
                extra[0, 4:7] = 0
            _set_state_dict_exact(base, state)
            base.agent.controller.set_state(_clone_tree(controller))
            for wrapper in (env, base):
                if hasattr(wrapper, "_elapsed_steps"):
                    wrapper._elapsed_steps = 0
            if torch.is_tensor(getattr(base, "_elapsed_steps", None)):
                base._elapsed_steps.zero_()
            obs = base.get_obs()
            reset_info = base.evaluate()
    # Official Panda PickCube expert geometry and grasp convention.
    approaching = np.array([0.0, 0.0, -1.0])
    closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp = compute_grasp_info_by_obb(
        get_actor_obb(base.cube), approaching=approaching, target_closing=closing, depth=0.025
    )
    grasp_pose = base.agent.build_grasp_pose(
        approaching, grasp["closing"], base.cube.pose.sp.p
    )
    actions, states, observations = [], [_clone_tree(base.get_state_dict())], [_clone_tree(obs)]
    infos = [_clone_tree(reset_info)]
    initial_observation_named = _clone_tree(
        base.get_obs(info=reset_info, unflattened=True)
    )
    terminated_rows, truncated_rows = [False], [False]
    initial_snapshot = snapshot_execution_state(env)
    trace = {
        "swept_xy": [],
        "cube_xy": [],
        "waypoint_xy": None if waypoint is None else waypoint.tolist(),
        "waypoint_sampling_attempts": waypoint_attempts,
        "phase_status": [],
    }
    trace["phase_timings"] = {"geometry": time.monotonic() - expert_started,
                              "expert_execution": 0.0}
    execution_started = time.monotonic()
    current_phase = "grasp_approach"
    last_info = reset_info
    done = False

    def step(action):
        nonlocal obs, last_info, done
        if done or len(actions) >= config.max_steps:
            done = True
            return True
        action = torch.as_tensor(action, dtype=torch.float32, device=base.device)
        obs, _, terminated, truncated, last_info = env.step(action)
        actions.append(action.detach().cpu().numpy().copy())
        contact = getattr(base, "_first_obstacle_contact", None)
        if contact is not None and "first_obstacle_contact" not in trace:
            trace["first_obstacle_contact"] = {
                **contact, "control_step": len(actions), "phase": current_phase,
            }
        states.append(_clone_tree(base.get_state_dict()))
        observations.append(_clone_tree(obs))
        infos.append(_clone_tree(last_info))
        terminated_rows.append(bool(terminated))
        truncated_rows.append(bool(truncated))
        trace["cube_xy"].append(base.cube.pose.p[0, :2].detach().cpu().numpy().copy())
        trace["swept_xy"].extend(
            _collision_vertices_xy(base, max_height=float(base.obstacle_height))
        )
        done = bool(terminated) or bool(truncated) or bool(
            base.evaluate()["native_success"][0].item()
        )
        return done

    def move_to(target, grip, tolerance=None, target_quat=None):
        tolerance = config.waypoint_tolerance if tolerance is None else tolerance
        for _ in range(config.max_steps_per_waypoint):
            if done or len(actions) >= config.max_steps:
                return "episode_ended"
            tcp = base.agent.tcp.pose.p[0]
            delta = target - tcp
            rotation = torch.zeros(3, device=base.device)
            rotation_done = True
            if target_quat is not None:
                current_q = base.agent.tcp.pose.q[0].detach().cpu().numpy()
                target_xyzw = np.asarray(target_quat)[[1, 2, 3, 0]]
                current_xyzw = current_q[[1, 2, 3, 0]]
                delta_rotation = (
                    Rotation.from_quat(target_xyzw)
                    * Rotation.from_quat(current_xyzw).inv()
                )
                delta_euler = delta_rotation.as_euler("XYZ")
                arm_controller = base.agent.controller.controllers["arm"]
                rotation = torch.as_tensor(
                    scale_native_euler_action(
                        delta_euler, arm_controller.config.rot_lower, config.action_cap
                    ),
                    dtype=torch.float32,
                    device=base.device,
                )
                rotation_done = np.linalg.norm(delta_rotation.as_rotvec()) < 0.03
            if float(torch.linalg.norm(delta).item()) < tolerance and rotation_done:
                return None
            pos = (delta / 0.1).clamp(-config.action_cap, config.action_cap)
            if step(torch.cat([pos, rotation, torch.tensor([grip], device=base.device)])):
                return "episode_ended"
        if len(actions) >= config.max_steps:
            return "episode_ended"
        return "timeout"

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    grasp_position = torch.as_tensor(reach_pose.p, dtype=torch.float32, device=base.device)
    cube_z = float(base.cube.pose.p[0, 2].item())
    phases = [
        (grasp_position, 1.0, None),
        (torch.as_tensor(grasp_pose.p, dtype=torch.float32, device=base.device), 1.0, None),
    ]
    grasp_pose_reached = True
    for phase_name, (target, grip, tolerance) in zip(
        ("grasp_approach", "grasp_pose"), phases
    ):
        current_phase = phase_name
        status = move_to(target, grip, tolerance, grasp_pose.q)
        trace["phase_status"].append((phase_name, status or "reached"))
        if status is not None:
            grasp_pose_reached = False
            break
    if grasp_pose_reached:
        for _ in range(config.grasp_steps):
            current_phase = "gripper_close"
            if step(torch.tensor([0, 0, 0, 0, 0, 0, -1.0], device=base.device)):
                grasp_pose_reached = False
                trace["phase_status"].append(("gripper_close", "episode_ended"))
                break
    lift = grasp_pose.p.copy()
    lift[2] += config.lift_height
    targets = [lift]
    if waypoint is not None:
        targets.append(np.array([waypoint[0], waypoint[1], cube_z + config.lift_height]))
    targets.append(base.goal_site.pose.p[0].detach().cpu().numpy())
    if grasp_pose_reached:
        for target_name, target in zip(
            ("lift", "waypoint", "goal") if waypoint is not None else ("lift", "goal"),
            targets,
        ):
            current_phase = target_name
            if len(actions) >= config.max_steps:
                trace["phase_status"].append((target_name, "max_steps"))
                break
            status = move_to(
                torch.as_tensor(target, dtype=torch.float32, device=base.device), -1.0
            )
            trace["phase_status"].append((target_name, status or "reached"))
            if status is not None:
                break
    if grasp_pose_reached:
        current_phase = "settle"
        for _ in range(config.settle_steps):
            if len(actions) >= config.max_steps or step(
                torch.tensor([0, 0, 0, 0, 0, 0, -1.0], device=base.device)
            ):
                break
    native = base.evaluate()["native_success"]
    trace["phase_timings"]["expert_execution"] = time.monotonic() - execution_started
    safety = {
        "native_success": bool(native[0].item()),
        "obstacle_collision": bool(base._obstacle_collision[0].item()),
        "obstacle_crossing": bool(base._forbidden_crossing[0].item()),
        "safe_success": bool(base.evaluate()["success"][0].item()),
    }
    return {
        "actions": np.asarray(actions),
        "states": states,
        "observations": observations,
        "initial_state": states[0],
        "initial_observation": observations[0],
        "initial_observation_named": initial_observation_named,
        "initial_snapshot": initial_snapshot,
        "safety": safety,
        "native_info": last_info,
        "infos": infos,
        "terminated": np.asarray(terminated_rows, dtype=np.uint8),
        "truncated": np.asarray(truncated_rows, dtype=np.uint8),
        "trace": trace,
        "layout": {"start_xy": cube_xy.tolist(), "goal_xy": goal_xy.tolist()},
    }


def _clone_tree(value):
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def _set_state_dict_exact(env, expected):
    """Correct only measured SAPIEN root-pose round-trip drift, then verify all leaves."""
    candidate = _clone_tree(expected)
    for _ in range(4):
        env.set_state_dict(candidate)
        actual = env.get_state_dict()
        if _state_tree_equal(expected, actual):
            return
        wanted_root = expected.get("articulations", {}).get("panda")
        actual_root = actual.get("articulations", {}).get("panda")
        if (not torch.is_tensor(wanted_root) or not torch.is_tensor(actual_root)
                or wanted_root.shape != actual_root.shape or wanted_root.shape[-1] < 7):
            break
        normalized_actual = _clone_tree(actual)
        normalized_actual["articulations"]["panda"][..., :3] = wanted_root[..., :3]
        if not _state_tree_equal(expected, normalized_actual):
            break
        # SAPIEN's float32 world-pose round trip can accumulate a few ulps in
        # the Panda root translation. Feed back the measured residual; never
        # accept it as equal unless the complete resulting state is bit-exact.
        candidate["articulations"]["panda"][..., :3] += (
            wanted_root[..., :3] - actual_root[..., :3]
        )
    raise RuntimeError("exact native state restore failed after root-pose residual correction")


def _state_tree_equal(expected, actual):
    if not isinstance(expected, dict) or not isinstance(actual, dict) or expected.keys() != actual.keys():
        return False
    for key in expected:
        left, right = expected[key], actual[key]
        if isinstance(left, dict):
            if not _state_tree_equal(left, right):
                return False
        elif torch.is_tensor(left):
            if not torch.is_tensor(right) or not torch.equal(left, right):
                return False
        elif not np.array_equal(left, right):
            return False
    return True


def project_collision_vertices_xy(vertices_world, max_height):
    """Project collision vertices occupying the cylinder's vertical envelope."""
    vertices_world = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
    return vertices_world[vertices_world[:, 2] <= max_height, :2]


def scale_native_euler_action(delta_euler, rot_lower, action_cap):
    """Map native XYZ Euler deltas through the signed controller lower scale."""
    return np.clip(
        np.asarray(delta_euler, dtype=np.float64) / np.asarray(rot_lower, dtype=np.float64),
        -action_cap,
        action_cap,
    )


def _collision_vertices_xy(env, max_height):
    points = []
    for link in env.agent.robot.links:
        for shape in link._bodies[0].collision_shapes:
            try:
                vertices = np.asarray(shape.get_vertices())
            except AttributeError:
                get_half_size = getattr(shape, "get_half_size", None)
                if get_half_size is None:
                    raise RuntimeError(
                        f"cannot enumerate collision geometry for robot link {link.name}"
                    )
                half = np.asarray(get_half_size()).reshape(3)
                vertices = np.array(
                    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                ) * half
            local = shape.get_local_pose().to_transformation_matrix()
            homogeneous = np.concatenate([vertices.T, np.ones((1, len(vertices)))])
            world = link.pose.to_transformation_matrix()[0].cpu().numpy() @ (
                np.asarray(local) @ homogeneous
            )
            points.extend(project_collision_vertices_xy(world[:3].T, max_height).tolist())
    half = float(env.cube_half_size)
    cube_pose = env.cube.pose.p[0].detach().cpu().numpy()
    local_cube_vertices = np.asarray([
        [x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)
    ])
    cube_rotation = env.cube.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    cube_vertices = local_cube_vertices @ cube_rotation.T + cube_pose
    points.extend(project_collision_vertices_xy(cube_vertices, max_height).tolist())
    return points


def make_pickcube_env(render_mode=None, *, obstacle_radius=0.03, obstacle_height=0.25):
    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cpu",
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        control_mode="pd_ee_delta_pose",
        max_episode_steps=500,
        render_mode=render_mode,
        obstacle_radius=obstacle_radius,
        obstacle_height=obstacle_height,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = PickCubeExpertConfig(seed=args.seed)
    (output / "config.json").write_text(
        json.dumps(
            {
                "config": dataclasses.asdict(config),
                "robot": "panda",
                "robot_init_qpos_noise": 0.02,
                "control_mode": "pd_ee_delta_pose",
                "official_solver_dependency_available": True,
                "control_rationale": (
                    "official Panda solver emits joint-position actions; retain its "
                    "OBB grasp geometry and task phases with 7d pose feedback"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    results = {}
    for name, detour in (("direct", False), ("same_base_side_detour", True)):
        env = make_pickcube_env()
        try:
            result = run_pickcube_expert(
                env,
                cube_xy=(-0.12, 0.07),
                goal_xy=(0.12, 0.07),
                detour=detour,
                obstacle_enabled=False,
                config=config,
            )
            results[name] = {
                "steps": len(result["actions"]),
                "safety": result["safety"],
                "layout": result["layout"],
                "waypoint_xy": result["trace"]["waypoint_xy"],
                "swept_xy_points": len(result["trace"]["swept_xy"]),
                "cube_trace_points": len(result["trace"]["cube_xy"]),
                "action_shape": list(result["actions"].shape),
                "cube_final_xyz": env.unwrapped.cube.pose.p[0].detach().cpu().tolist(),
                "goal_xyz": env.unwrapped.goal_site.pose.p[0].detach().cpu().tolist(),
                "tcp_final_xyz": env.unwrapped.agent.tcp.pose.p[0].detach().cpu().tolist(),
                "native_info": {
                    key: value.detach().cpu().tolist()
                    for key, value in result["native_info"].items()
                    if torch.is_tensor(value)
                },
            }
            (output / "smoke.json").write_text(json.dumps(results, indent=2) + "\n")
        finally:
            env.close()
    (output / "smoke.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(item["safety"]["native_success"] for item in results.values()):
        raise RuntimeError("one or more PickCube expert smoke paths missed native success")


if __name__ == "__main__":
    main()
