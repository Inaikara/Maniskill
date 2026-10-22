"""Bounded PlaceSphere expert pilot with native state/action semantics."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import mplib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import examples.baselines.constrain.placesphere_obstacle as obstacle_task
from examples.baselines.constrain.pickcube_expert import (
    _clone_tree, _set_state_dict_exact,
    scale_native_euler_action,
)
from examples.baselines.constrain.layout_geometry import (
    LayoutGeometryConfig, sample_same_side_waypoint, validate_layout,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
import sapien


class _CompatiblePandaArmMotionPlanningSolver:
    """Official solver with the installed mplib Pose-valued base-pose seam adapted."""

    def __new__(cls, *args, **kwargs):
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

        class Compatible(PandaArmMotionPlanningSolver):
            def setup_planner(self):
                move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
                link_names = [link.get_name() for link in self.robot.get_links()]
                joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
                planner = mplib.Planner(
                    urdf=self.env_agent.urdf_path,
                    srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                    user_link_names=link_names,
                    user_joint_names=joint_names,
                    move_group=move_group,
                )
                planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
                planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
                planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
                return planner

            def move_to_pose_with_screw(self, pose, dry_run=False, refine_steps=0):
                # mplib's installed API takes mplib.Pose and has no
                # use_point_cloud keyword; official solver passes an array and
                # that keyword. The native expert uses no point-cloud planner.
                from mani_skill.utils.structs.pose import to_sapien_pose

                pose = to_sapien_pose(pose)
                self._update_grasp_visual(pose)
                pose = self._transform_pose_for_planning(pose)
                goal = mplib.Pose(pose.p, pose.q)
                current = self.robot.get_qpos().cpu().numpy()[0]
                result = self.planner.plan_screw(
                    goal, current, time_step=self.base_env.control_timestep,
                    wrt_world=True)
                if result["status"] != "Success":
                    result = self.planner.plan_screw(
                        goal, current, time_step=self.base_env.control_timestep,
                        wrt_world=True)
                    if result["status"] != "Success":
                        return -1
                if dry_run:
                    return result
                return self.follow_path(result, refine_steps=refine_steps)

        return Compatible(*args, **kwargs)


@dataclass(frozen=True)
class PlaceSphereLayoutConfig:
    table_bounds: tuple = (-0.25, -0.25, 0.25, 0.25)
    min_transport_distance: float = 0.14
    min_base_line_distance: float = 0.06
    waypoint_offset_range: tuple = (0.14, 0.20)
    max_attempts: int = 32


def validate_placesphere_layout(start, goal, base, config, *, container_half_extent):
    base_config = LayoutGeometryConfig(
        table_bounds=config.table_bounds,
        min_transport_distance=config.min_transport_distance,
        min_base_line_distance=config.min_base_line_distance,
        waypoint_offset_range=config.waypoint_offset_range,
        waypoint_lead_fraction=0.25, max_attempts=config.max_attempts)
    result = validate_layout(start, goal, base, base_config)
    xmin, ymin, xmax, ymax = config.table_bounds
    x, y = np.asarray(goal, dtype=float)
    inside = (xmin + container_half_extent <= x <= xmax - container_half_extent and
              ymin + container_half_extent <= y <= ymax - container_half_extent)
    return (result.valid and inside), (None if result.valid and inside else
            result.reason or "container_bounds"), result, base_config


@dataclass(frozen=True)
class PlaceSpherePilotConfig:
    seed: int = 20261022
    target_per_class: int = 20
    max_attempts_per_class: int = 200
    max_steps: int = 200
    max_wall_seconds: float = 1800.0
    obs_horizon: int = 2
    pred_horizon: int = 16
    execution_horizon: int = 8
    geometry_attempts: int = 32
    waypoint_attempts: int = 32
    cylinder_candidates: int = 525
    obstacle_radius: float = 0.03
    obstacle_height: float = 0.08
    safety_margin: float = 0.005


def make_env(render_mode=None, config=PlaceSpherePilotConfig()):
    return gym.make(obstacle_task.ENV_ID, num_envs=1, obs_mode="state",
        sim_backend="physx_cpu", robot_uids="panda", robot_init_qpos_noise=0.02,
        control_mode="pd_ee_delta_pose", max_episode_steps=config.max_steps,
        render_mode=render_mode, obstacle_radius=config.obstacle_radius,
        obstacle_height=config.obstacle_height, safety_margin=config.safety_margin)


def _obs_vector(obs):
    if isinstance(obs, dict):
        return np.concatenate([_obs_vector(obs[key]).reshape(-1) for key in obs])
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _policy_observation(env, info):
    """Flatten native named state fields while excluding the obstacle hint."""
    named=env.get_obs(info=info,unflattened=True)
    def visit(node):
        if isinstance(node,dict):
            return [visit(value) for key,value in node.items()
                    if key!=obstacle_task.OBSTACLE_KEY]
        return [np.asarray(node.detach().cpu() if torch.is_tensor(node) else node,
                           dtype=np.float32).reshape(-1)]
    pieces=[]
    def collect(nodes):
        for item in nodes:
            if isinstance(item,list):collect(item)
            else:pieces.append(item)
    collect(visit(named))
    return np.concatenate(pieces).astype(np.float32,copy=False)


def _policy_observation_schema(env, info):
    named=env.get_obs(info=info,unflattened=True);fields=[]
    def visit(node,path=""):
        if isinstance(node,dict):
            for key,value in node.items():
                if key==obstacle_task.OBSTACLE_KEY:continue
                visit(value,f"{path}/{key}")
        else:
            array=node.detach().cpu().numpy() if torch.is_tensor(node) else np.asarray(node)
            fields.append({"path":path,"shape":list(array.shape[1:]),"flattened_dimension":int(array[0].size)})
    visit(named)
    return fields


def _official_grasp_angles():
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2) + np.pi / 4
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    return [float(angle) for angle in angles]


def _equal_tree(left, right, ignored=frozenset()):
    if isinstance(left, dict) and isinstance(right, dict):
        lk, rk = left.keys() - ignored, right.keys() - ignored
        return lk == rk and all(_equal_tree(left[k], right[k], ignored) for k in lk)
    if torch.is_tensor(left) or torch.is_tensor(right):
        a, b = torch.as_tensor(left), torch.as_tensor(right)
        return a.shape == b.shape and torch.equal(a, b)
    return np.array_equal(left, right)


def _first_difference(left, right, ignored=frozenset(), path=""):
    if isinstance(left,dict) and isinstance(right,dict):
        lk,rk=left.keys()-ignored,right.keys()-ignored
        if lk!=rk:return {"path":path or "/","expected_keys":sorted(lk),"actual_keys":sorted(rk)}
        for key in sorted(lk):
            difference=_first_difference(left[key],right[key],ignored,f"{path}/{key}")
            if difference:return difference
        return None
    if not _equal_tree(left,right):
        a=left.detach().cpu().tolist() if torch.is_tensor(left) else np.asarray(left).tolist()
        b=right.detach().cpu().tolist() if torch.is_tensor(right) else np.asarray(right).tolist()
        return {"path":path or "/","expected":a,"actual":b}
    return None


def _tree_sha256(tree, ignore_obstacle_condition=False):
    digest=hashlib.sha256()
    def visit(node,path=""):
        if isinstance(node,dict):
            for key in sorted(node):
                value=node[key]
                if ignore_obstacle_condition and key==obstacle_task.OBSTACLE_KEY:
                    value=torch.as_tensor(value).clone();value[..., [0,1,3]]=0
                visit(value,f"{path}/{key}")
        else:
            value=node.detach().cpu().numpy() if torch.is_tensor(node) else np.asarray(node)
            digest.update(path.encode());digest.update(np.ascontiguousarray(value).tobytes())
    visit(tree)
    return digest.hexdigest()


def _restore_matches_except_obstacle(template, restored):
    if template.keys() != restored.keys():
        return False
    for key in template:
        if key == obstacle_task.OBSTACLE_KEY:
            a, b = torch.as_tensor(template[key]), torch.as_tensor(restored[key])
            # Only cylinder centre and enabled bit are allowed to differ.
            if a.shape != b.shape or not torch.equal(a[..., [2, 4, 5, 6, 7, 8]],
                                                      b[..., [2, 4, 5, 6, 7, 8]]):
                return False
        elif not _equal_tree(template[key], restored[key]):
            return False
    return True


def _robot_collision_vertices_xy(env, max_height):
    """World XY vertices of all Panda collision shapes below cylinder height."""
    points=[]
    for link in env.agent.robot.links:
        for shape in link._bodies[0].collision_shapes:
            try: vertices=np.asarray(shape.get_vertices())
            except AttributeError:
                get_half=getattr(shape,"get_half_size",None)
                if get_half is None:raise RuntimeError(f"cannot enumerate collision geometry for {link.name}")
                half=np.asarray(get_half()).reshape(3)
                vertices=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
            local=shape.get_local_pose().to_transformation_matrix()
            homogeneous=np.concatenate((vertices.T,np.ones((1,len(vertices)))))
            world=link.pose.to_transformation_matrix()[0].cpu().numpy() @ (np.asarray(local) @ homogeneous)
            points.extend(world[:3].T.tolist())
    return points


def _continuous_sweep_clearance(center, samples, max_height=None):
    """Minimum XY distance from a point to sampled continuous vertex paths."""
    points=np.asarray(samples,dtype=np.float64)
    if points.ndim==2:
        if len(points)==0:return float("inf")
        if len(points)==1:return float(np.linalg.norm(points[0,:2]-center))
        points=points[:,None,:]
    if points.ndim!=3 or points.shape[-1] not in (2,3):
        raise ValueError("sweep samples must have shape (T, points, 2/3) or (T, 2/3)")
    xy=points[...,:2]
    if len(points)==1:
        eligible=np.ones(points.shape[1],dtype=bool) if max_height is None or points.shape[-1]==2 else points[0,:,2]<=max_height
        return float(np.min(np.linalg.norm(xy[0,eligible]-center,axis=-1))) if eligible.any() else float("inf")
    start=xy[:-1];direction=xy[1:]-start
    length_sq=np.sum(direction*direction,axis=-1).clip(1e-24)
    t=np.clip(np.sum((center-start)*direction,axis=-1)/length_sq,0,1)
    nearest=start+t[...,None]*direction
    distance=np.linalg.norm(nearest-center,axis=-1)
    if max_height is not None and points.shape[-1]==3:
        eligible=np.minimum(points[:-1,:,2],points[1:,:,2])<=max_height
        distance=distance[eligible]
    return float(np.min(distance)) if distance.size else float("inf")


def run_expert(env, start_xy, container_xy, *, detour, seed, max_steps,
               waypoint=None, obstacle_enabled=False, obstacle_center=None,
               initial_snapshot=None):
    base = env.unwrapped
    if initial_snapshot is None:
        obs, info = env.reset(seed=seed, options={
            "sphere_xy": start_xy, "container_xy": container_xy,
            "obstacle_enabled": obstacle_enabled,
            "obstacle_center": obstacle_center or (0.0, 0.2)})
    else:
        state, controller = initial_snapshot
        _set_state_dict_exact(base, _clone_tree(state))
        base.agent.controller.set_state(_clone_tree(controller))
        for wrapper in (env, base):
            if hasattr(wrapper, "_elapsed_steps"):
                wrapper._elapsed_steps = 0
        if torch.is_tensor(getattr(base, "_elapsed_steps", None)):
            base._elapsed_steps.zero_()
        obs = base.get_obs()
        info = base.evaluate()
    obs = _policy_observation(base, info)
    # Official PlaceSphere OBB grasp/orientation convention: downward approach,
    # TCP's local closing axis, 25 mm finger depth and tested Z wrist rotations.
    approaching = np.array([0.0, 0.0, -1.0])
    closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        get_actor_obb(base.obj), approaching=approaching, target_closing=closing, depth=0.025)
    grasp_pose = base.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])
    planner = _CompatiblePandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False)
    chosen = None
    tested = []
    for angle in _official_grasp_angles():
        candidate = grasp_pose * sapien.Pose(q=__import__("transforms3d.euler", fromlist=["euler2quat"]).euler2quat(0, 0, float(angle)))
        tested.append(float(angle))
        if planner.move_to_pose_with_screw(candidate, dry_run=True) == -1:
            continue
        chosen = candidate
        break
    if chosen is None:
        return None
    actions, states, observations, infos = [], [_clone_tree(base.get_state_dict())], [_clone_tree(obs)], [_clone_tree(info)]
    controller_states=[_clone_tree(base.agent.controller.get_state())]
    terminated, truncated = [False], [False]
    initial_state = _clone_tree(base.get_state_dict())
    initial_controller = _clone_tree(base.agent.controller.get_state())
    initial_obs_named = _clone_tree(base.get_obs(info=info, unflattened=True))
    phase = "approach"
    trace = {"phases": [], "grasp_rotation_rad": tested[-1],
        "waypoint_xy": None if waypoint is None else list(waypoint),
        "sphere_xy": [base.obj.pose.p[0,:2].detach().cpu().numpy().tolist()]}
    done = False

    def step(action):
        nonlocal obs, info, done
        if done or len(actions) >= max_steps:
            return True
        _, _, term, trunc, info = env.step(torch.as_tensor(action, dtype=torch.float32, device=base.device))
        obs = _policy_observation(base, info)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        states.append(_clone_tree(base.get_state_dict())); observations.append(_clone_tree(obs)); infos.append(_clone_tree(info))
        controller_states.append(_clone_tree(base.agent.controller.get_state()))
        terminated.append(bool(term)); truncated.append(bool(trunc))
        if base._first_obstacle_contact is not None and "first_obstacle_contact" not in trace:
            trace["first_obstacle_contact"] = {**base._first_obstacle_contact,
                "control_step":len(actions),"phase":phase}
        trace["sphere_xy"].append(base.obj.pose.p[0, :2].detach().cpu().numpy().tolist())
        done = bool(term or trunc or base.evaluate()["native_success"][0].item())
        return done

    def move(target, target_q=None, grip=1.0, tolerance=0.006, budget=80):
        for _ in range(budget):
            if done or len(actions) >= max_steps:
                return False
            delta = target - base.agent.tcp.pose.p[0]
            rot = torch.zeros(3, device=base.device)
            rot_done = True
            if target_q is not None:
                current = base.agent.tcp.pose.q[0].detach().cpu().numpy()
                dq = Rotation.from_quat(np.asarray(target_q)[[1, 2, 3, 0]]) * Rotation.from_quat(current[[1, 2, 3, 0]]).inv()
                euler = dq.as_euler("XYZ")
                arm = base.agent.controller.controllers["arm"]
                rot = torch.as_tensor(scale_native_euler_action(euler, arm.config.rot_lower, 0.35), device=base.device, dtype=torch.float32)
                rot_done = np.linalg.norm(dq.as_rotvec()) < 0.03
            if torch.linalg.norm(delta).item() < tolerance and rot_done:
                return True
            pos = (delta / 0.1).clamp(-0.35, 0.35)
            if step(torch.cat((pos, rot, torch.tensor([grip], device=base.device)))):
                return bool(base.evaluate()["native_success"][0].item())
        return False

    pre = chosen * sapien.Pose([0, 0, -0.05])
    sphere_z = float(base.obj.pose.p[0, 2].item())
    stages = [("approach", pre.p, pre.q), ("grasp", chosen.p, chosen.q)]
    ok = True
    for phase, pos, quat in stages:
        ok = move(torch.as_tensor(pos, device=base.device, dtype=torch.float32), quat)
        trace["phases"].append({"phase": phase, "reached": ok})
        if not ok: break
    if ok:
        for _ in range(8):
            if step([0, 0, 0, 0, 0, 0, -1]): ok = False; break
    lift = np.asarray(chosen.p).copy(); lift[2] += 0.10
    targets = [("lift", lift)]
    if waypoint is not None: targets.append(("waypoint", [waypoint[0], waypoint[1], sphere_z + 0.10]))
    # Official expert alignment target is bin pose * [0,0,2*bin_half_z].
    goal_pose = base.bin.pose.sp * sapien.Pose([0, 0, 2 * base.block_half_size[2]])
    offset = goal_pose.p - base.obj.pose.p[0].detach().cpu().numpy()
    goal = lift + offset
    targets.append(("container_align", goal))
    for phase, target in targets:
        if not ok: break
        ok = move(torch.as_tensor(target, device=base.device, dtype=torch.float32), chosen.q, grip=-1.0)
        trace["phases"].append({"phase": phase, "reached": ok})
    if ok:
        phase = "release_retreat"
        retreat = goal.copy(); retreat[2] += 0.08
        retreat_reached = move(torch.as_tensor(retreat, device=base.device, dtype=torch.float32), chosen.q, grip=1.0)
        trace["phases"].append({"phase": "release_open_and_retreat", "reached": retreat_reached})
        phase = "settle"
        settle_steps = 0
        for _ in range(12):
            finished = step([0, 0, 0, 0, 0, 0, 1])
            settle_steps += 1
            if finished: break
        trace["phases"].append({"phase": "settle", "control_steps": settle_steps})
    native = bool(base.evaluate()["native_success"][0].item())
    final_info = infos[-1]
    return {"actions": np.asarray(actions, dtype=np.float32), "states": states,
        "observations": observations, "infos": infos, "terminated": np.asarray(terminated, dtype=np.uint8),
        "truncated": np.asarray(truncated, dtype=np.uint8), "native_success": native,
        "safe_success": bool(base.evaluate()["success"][0].item()), "initial_state": initial_state,
        "initial_controller": initial_controller, "initial_obs_named": initial_obs_named,
        "controller_states":controller_states,
        "trace": trace, "final_info": final_info}


def _attempt(kind, seed, rng, config):
    attempt_started=time.monotonic()
    env = make_env(config=config)
    try:
        base = env.unwrapped
        base_xy = base.agent.robot.pose.p[0, :2].detach().cpu().numpy()
        geom = PlaceSphereLayoutConfig()
        container_half = 0.025
        geometry_started=time.monotonic();geometry_tries=0
        for _ in range(config.geometry_attempts):
            geometry_tries+=1
            start = np.array([rng.uniform(-0.19, -0.08), rng.uniform(-0.13, 0.13)])
            goal = np.array([rng.uniform(0.04, 0.15), rng.uniform(-0.13, 0.13)])
            valid, _, _, layout_config = validate_placesphere_layout(
                start, goal, base_xy, geom, container_half_extent=container_half)
            if valid: break
        else: return {"failure": "geometry_retry_exhausted","counters":{"geometry_tries":geometry_tries},"phase_timings":{"geometry":time.monotonic()-geometry_started}}, None
        waypoint = None
        waypoint_tries=0
        if kind == "obstacle":
            sampled = sample_same_side_waypoint(start, goal, base_xy, rng, layout_config)
            waypoint_tries=sampled.attempts
            if sampled.waypoint_xy is None: return {"failure": "waypoint_retry_exhausted","counters":{"geometry_tries":geometry_tries,"waypoint_tries":waypoint_tries}}, None
            waypoint = sampled.waypoint_xy
        geometry_elapsed=time.monotonic()-geometry_started
        template_started=time.monotonic()
        result = run_expert(env, start, goal, detour=kind == "obstacle", seed=seed,
                            max_steps=config.max_steps, waypoint=waypoint)
        template_elapsed=time.monotonic()-template_started
        if result is None or not result["native_success"]:
            return {"failure": "native_unsuccessful", "counters": {"geometry_tries": geometry_tries,"waypoint_tries":waypoint_tries,"template_executions":int(kind=="obstacle"),"grasp_phases":int(result is not None),"release_phases":int(bool(result and any(p.get("phase")=="release_open_and_retreat" for p in result["trace"]["phases"]))),"stable_successes":0,"stable_failures":1},
                    "phase_timings":{"geometry":geometry_elapsed,"template":template_elapsed},
                    "diagnostics": {"layout": {"start": start.tolist(), "container": goal.tolist()},
                                    "trace": result["trace"] if result else None,
                                    "action_count": len(result["actions"]) if result else 0,
                                    "terminated": result["terminated"].tolist() if result else None,
                                    "truncated": result["truncated"].tolist() if result else None,
                                    "final_info": _json(result["final_info"]) if result else None,
                                    "final_sphere_pose": base.obj.pose.raw_pose.detach().cpu().tolist()}}, None
        if kind == "obstacle":
            cylinder_started=time.monotonic()
            sweep = np.asarray(result["trace"]["sphere_xy"],dtype=np.float64)[:,None,:]
            line = goal - start; norm = np.linalg.norm(line)
            lateral = np.array([-line[1], line[0]]) / norm
            base_side = np.sign(np.cross(line, base_xy - start))
            centers = [start + t * line - base_side * lateral * off
                       for t in np.linspace(.4, .6, 21) for off in np.linspace(0, .06, 25)]
            blocking = [np.linalg.norm(c - (start + np.clip(np.dot(c-start, line)/(norm**2),0,1)*line)) <= .055 and
                        np.linalg.norm(c-start) > .07 and np.linalg.norm(c-goal) > .07 for c in centers]
            robot_sweep = []
            # Conservative measured link collision vertices, including all links.
            for state in result["states"]:
                _set_state_dict_exact(base, state)
                robot_sweep.append(np.asarray(
                    _robot_collision_vertices_xy(base, config.obstacle_height),dtype=np.float64))
            if len({len(points) for points in robot_sweep}) != 1:
                return {"failure":"robot_sweep_geometry_changed","counters":{
                    "geometry_tries":geometry_tries,"waypoint_tries":waypoint_tries,
                    "template_executions":1},"diagnostics":{"sweep_vertex_counts":[len(x) for x in robot_sweep]}},None
            robot_sweep=np.stack(robot_sweep)
            feasible = []
            for i, center in enumerate(centers):
                sphere_clearance=_continuous_sweep_clearance(center,sweep)
                robot_clearance=_continuous_sweep_clearance(center,robot_sweep,config.obstacle_height+0.02)
                if not blocking[i] or sphere_clearance < .055: continue
                if robot_clearance < config.obstacle_radius: continue
                feasible.append((center,robot_clearance,sphere_clearance))
            cylinder_elapsed=time.monotonic()-cylinder_started
            if not feasible: return {"failure": "cylinder_geometry_retry_exhausted",
                "counters": {"geometry_tries":geometry_tries,"waypoint_tries":waypoint_tries,"template_executions":1,"cylinder_candidates": len(centers)}, "phase_timings":{"geometry":geometry_elapsed,"template":template_elapsed,"cylinder_sampling":cylinder_elapsed}, "diagnostics": {
                    "blocking_candidates": int(sum(blocking)),
                    "sphere_clear_candidates": int(sum(
                        blocking[i] and _continuous_sweep_clearance(center,sweep) >= .055
                        for i,center in enumerate(centers))),
                    "robot_clear_candidates": int(sum(
                        blocking[i] and _continuous_sweep_clearance(center,robot_sweep,config.obstacle_height+0.02) >= config.obstacle_radius
                        for i,center in enumerate(centers))),
                    "minimum_robot_distance_m": float(min(_continuous_sweep_clearance(c,robot_sweep,config.obstacle_height+0.02) for c in centers)),
                    "maximum_obstacle_offset_m": .06}}, None
            center, robot_clearance, sphere_clearance = max(feasible,key=lambda item:item[1])
            # Fresh environment prevents stale contact-cache contamination. Restore
            # the exact template full state and controller, differing only in obstacle.
            template_state, template_controller = result["initial_state"], result["initial_controller"]
            template_obs = result["initial_obs_named"]
            template_policy_obs = result["observations"][0].copy()
            env.close(); env = make_env(config=config); base = env.unwrapped
            obs, _ = env.reset(seed=seed, options={"sphere_xy": start, "container_xy": goal,
                "obstacle_enabled": True, "obstacle_center": center})
            restored = _clone_tree(template_state)
            restored[obstacle_task.OBSTACLE_KEY][0, :2] = torch.as_tensor(center, device=base.device)
            restored[obstacle_task.OBSTACLE_KEY][0, 3] = 1
            _set_state_dict_exact(base, restored)
            base.agent.controller.set_state(_clone_tree(template_controller))
            base._prev_sphere_xy[:] = base.obj.pose.p[:, :2]
            # Execute a fresh same-side expert rollout against the frozen scene.
            obstacle_state = _clone_tree(base.get_state_dict())
            obstacle_state[obstacle_task.OBSTACLE_KEY][0, :2] = torch.as_tensor(center, device=base.device)
            obstacle_state[obstacle_task.OBSTACLE_KEY][0, 3] = 1
            obstacle_state[obstacle_task.OBSTACLE_KEY][0, 4:7] = 0
            obstacle_started=time.monotonic()
            result = run_expert(env, start, goal, detour=True, seed=seed,
                max_steps=config.max_steps, waypoint=waypoint, obstacle_enabled=True,
                obstacle_center=center, initial_snapshot=(obstacle_state, template_controller))
            obstacle_elapsed=time.monotonic()-obstacle_started
            if not _restore_matches_except_obstacle(template_state, result["initial_state"]):
                return {"failure": "initial_state_restore_mismatch",
                    "counters":{"geometry_tries":geometry_tries,"waypoint_tries":waypoint_tries,
                        "template_executions":1,"restore_executions":1,"cylinder_candidates":len(centers)},
                    "phase_timings":{"geometry":geometry_elapsed,"template":template_elapsed,
                        "cylinder_sampling":cylinder_elapsed,"obstacle_execution":time.monotonic()-obstacle_started},
                    "diagnostics":{"native_state_equal_except_obstacle":False}}, None
            if not _equal_tree(template_controller, result["initial_controller"]):
                return {"failure":"controller_restore_mismatch","counters":{"template_executions":1,"restore_executions":1,"cylinder_candidates":len(centers)},
                    "phase_timings":{"geometry":geometry_elapsed,"template":template_elapsed,"cylinder_sampling":cylinder_elapsed,"obstacle_execution":time.monotonic()-obstacle_started}}, None
            if not _equal_tree(template_obs, result["initial_obs_named"],
                               frozenset({obstacle_task.OBSTACLE_KEY})):
                return {"failure": "initial_observation_restore_mismatch", "counters":{"template_executions":1,"restore_executions":1,"cylinder_candidates":len(centers)},
                    "phase_timings":{"geometry":geometry_elapsed,"template":template_elapsed,"cylinder_sampling":cylinder_elapsed,"obstacle_execution":time.monotonic()-obstacle_started}, "diagnostics": {
                    "first_difference": _first_difference(
                        template_obs, result["initial_obs_named"],
                        frozenset({obstacle_task.OBSTACLE_KEY}))}}, None
            result["trace"]["obstacle_center"] = center.tolist()
            result["trace"]["obstacle_sweep_clearance_m"]={
                "full_robot_vertex_paths":robot_clearance,"sphere_center_path":sphere_clearance}
            if not result["native_success"] or not result["safe_success"]:
                return {"failure": "obstacle_native_or_safety_failure",
                    "counters": {"geometry_tries":geometry_tries,"waypoint_tries":waypoint_tries,"template_executions":1,"obstacle_executions":1,"cylinder_candidates": len(centers),"restore_executions":1,"grasp_phases":1,"release_phases":int(any(p.get("phase")=="release_open_and_retreat" for p in result["trace"]["phases"])),"stable_successes":int(result["native_success"]),"stable_failures":int(not result["native_success"])}, "phase_timings":{"geometry":geometry_elapsed,"template":template_elapsed,"cylinder_sampling":cylinder_elapsed,"obstacle_execution":obstacle_elapsed}, "diagnostics": {
                        "native_success": result["native_success"],
                        "safe_success": result["safe_success"],
                        "final_info": _json(result["final_info"]),
                        "trace": result["trace"],
                        "final_sphere_pose": base.obj.pose.raw_pose.detach().cpu().tolist()}}, None
            result["trace"]["initial_state_identity"]={
                "template_native_state_sha256_excluding_obstacle_pose_condition":_tree_sha256(template_state,True),
                "obstacle_native_state_sha256_excluding_obstacle_pose_condition":_tree_sha256(result["initial_state"],True),
                "template_controller_sha256":_tree_sha256(template_controller),
                "obstacle_controller_sha256":_tree_sha256(result["initial_controller"]),
                "template_policy_observation_sha256":_tree_sha256(template_policy_obs),
                "obstacle_policy_observation_sha256":_tree_sha256(result["observations"][0]),
                "full_native_state_equal_except_named_obstacle_fields":True,
                "policy_observation_equal_obstacle_hint_excluded":True}
        else:
            result["trace"]["initial_state_identity"]={
                "native_state_sha256":_tree_sha256(result["initial_state"]),
                "controller_state_sha256":_tree_sha256(result["initial_controller"]),
                "named_observation_sha256":_tree_sha256(result["initial_obs_named"])}
        counters={"geometry_tries":geometry_tries,"waypoint_tries":waypoint_tries,
                  "template_executions":int(kind=="obstacle"),
                  "obstacle_executions":int(kind=="obstacle"),
                  "cylinder_candidates":config.cylinder_candidates if kind=="obstacle" else 0,
                  "restore_executions":int(kind=="obstacle"),"grasp_phases":1,
                  "release_phases":int(any(p.get("phase")=="release_open_and_retreat" for p in result["trace"]["phases"])),
                  "stable_successes":1,"stable_failures":0}
        timings={"geometry":geometry_elapsed,"template":template_elapsed,
                 "cylinder_sampling":cylinder_elapsed if kind=="obstacle" else 0.0,
                 "obstacle_execution":obstacle_elapsed if kind=="obstacle" else 0.0,
                 "complete_attempt":time.monotonic()-attempt_started}
        return {"actions": result["actions"], "obs": np.stack([_obs_vector(x) for x in result["observations"]]),
            "states": result["states"], "controller_states":result["controller_states"], "infos": result["infos"], "terminated": result["terminated"],
            "truncated": result["truncated"], "native_success": result["native_success"],
            "safe_success": result["safe_success"], "layout": {"start_xy": start.tolist(), "container_xy": goal.tolist()},
            "trace": result["trace"], "seed": seed,"counters":counters,"phase_timings":timings}, None
    finally: env.close()


def _state_write(group, key, value):
    if isinstance(value, dict):
        child = group.create_group(key)
        for subkey, subvalue in value.items(): _state_write(child, subkey, subvalue)
    else:
        group.create_dataset(key, data=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))


def collect_pilot(output_dir, config=PlaceSpherePilotConfig(), attempt_runner=None):
    positive = (config.target_per_class, config.max_attempts_per_class,
                config.max_steps, config.obs_horizon, config.pred_horizon,
                config.execution_horizon, config.geometry_attempts,
                config.waypoint_attempts, config.cylinder_candidates)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in positive):
        raise ValueError("pilot counts, horizons, retries, and step budget must be positive integers")
    if not np.isfinite(config.max_wall_seconds) or config.max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive and finite")
    obstacle_geometry = np.asarray(
        [config.obstacle_radius, config.obstacle_height, config.safety_margin],
        dtype=np.float64)
    if (not np.isfinite(obstacle_geometry).all() or config.obstacle_radius <= 0
            or config.obstacle_height <= 0 or config.safety_margin < 0):
        raise ValueError("obstacle geometry must be finite with positive radius/height and nonnegative margin")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "collection_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    repo = Path(__file__).resolve().parents[3]
    local_paths = ("placesphere_obstacle.py", "placesphere_pilot.py",
                   "pickcube_expert.py", "layout_geometry.py")
    local_pins = {
        f"examples/baselines/constrain/{name}": hashlib.sha256(
            (repo / "examples/baselines/constrain" / name).read_bytes()).hexdigest()
        for name in local_paths
    }
    import subprocess
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    native_paths = ("mani_skill/envs/tasks/tabletop/place_sphere.py",
                    "mani_skill/examples/motionplanning/panda/solutions/place_sphere.py")
    native_git_pins = {
        name: hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{revision}:{name}"], cwd=repo)).hexdigest()
        for name in native_paths
    }
    native_worktree_pins = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in native_paths
    }
    runtime = make_env(config=config)
    try:
        _, runtime_info = runtime.reset(seed=config.seed)
        runtime_task = runtime.unwrapped
        policy_schema = _policy_observation_schema(runtime_task, runtime_info)
        obs_dim = len(_policy_observation(runtime_task, runtime_info))
        rotation_lower = runtime_task.agent.controller.controllers["arm"].config.rot_lower
        rotation_lower = np.asarray(rotation_lower.detach().cpu() if torch.is_tensor(rotation_lower)
                                    else rotation_lower).tolist()
        actual_radius = float(runtime_task.radius)
        actual_bin_half = list(runtime_task.block_half_size)
    finally:
        runtime.close()

    contract = {
        "generator_version": "placesphere-pilot-v1",
        "git_commit": revision,
        "native_git_pins_sha256": native_git_pins,
        "native_worktree_source_sha256": native_worktree_pins,
        "local_source_sha256_before_collection": local_pins,
        "config": asdict(config),
        "native_geometry": {
            "sphere_radius_m": actual_radius,
            "bin_block_half_sizes_m": actual_bin_half,
            "native_success": "PlaceSphereEnv.evaluate: XY offset <= 0.005 m; Z offset error <= 0.005 m; sphere static; sphere not grasped",
        },
        "expert": {
            "robot": "Panda", "robot_init_qpos_noise": 0.02,
            "grasp": {"geometry": "official sphere OBB", "approaching_xyz": [0, 0, -1],
                      "closing_axis": "runtime TCP local Y", "finger_depth_m": 0.025,
                      "wrist_angles_rad": _official_grasp_angles()},
            "phases": ["approach", "grasp", "close", "lift", "same-side waypoint if obstacle",
                       "native bin-top offset", "release while retreating", "settle until native success"],
            "motion_planner_compatibility": {
                "mplib_version": getattr(mplib, "__version__", "unknown"),
                "base_pose": "convert official solver base pose array to installed mplib.Pose",
                "screw_target": "convert official solver target to installed mplib.Pose",
                "unsupported_keyword": "omit use_point_cloud; official Panda expert does not configure point-cloud planning",
            },
        },
        "state_fields": {
            "environment": "native PlaceSphere state dictionary plus custom fields; no task-specific flattened indexes",
            "placesphere_obstacle_order": ["center_x", "center_y", "radius", "enabled",
                "collision", "crossing", "grasp_started", "previous_sphere_x", "previous_sphere_y"],
            "placesphere_goal": "container body world XYZ; native body pose also retained in environment state",
            "controller": "explicit controller_states T+1 snapshots; initial controller is separately identity-pinned",
        },
        "policy_observation": {
            "dimension": obs_dim,
            "flatten_order": policy_schema,
            "obstacle_hint_excluded": True,
            "retained_fields": ["agent/qpos", "agent/qvel", "extra/is_grasped",
                "extra/tcp_pose", "extra/bin_pos", "extra/obj_pose", "extra/tcp_to_obj_pos"],
            "method": "flatten native named unflattened fields in insertion order, omitting only placesphere_obstacle",
        },
        "runtime_action": {
            "control_mode": "pd_ee_delta_pose",
            "dimension": 7,
            "position": "XYZ delta / 0.1 m, clipped to +/-0.35",
            "rotation": {"mapping": "XYZ Euler delta divided by signed runtime arm controller.rot_lower then clipped to +/-0.35",
                         "runtime_rot_lower": rotation_lower},
            "gripper": "index 6; close -1, open +1",
        },
        "horizons": {"observation": config.obs_horizon,
                     "prediction": config.pred_horizon,
                     "execution": config.execution_horizon},
        "sampling": {
            "sphere_x_m": [-0.19, -0.08], "sphere_y_m": [-0.13, 0.13],
            "container_x_m": [0.04, 0.15], "container_y_m": [-0.13, 0.13],
            "full_container_xy_half_extent_m": max(actual_bin_half[1:]),
            "geometry_candidates": config.geometry_attempts,
            "waypoint_candidates": config.waypoint_attempts,
            "same_base_side_only": True,
            "minimum_transport_distance_m": 0.14,
            "minimum_base_line_distance_m": 0.06,
        },
        "obstacle": {
            "shape": "static vertical cylinder",
            "radius_m": config.obstacle_radius,
            "height_m": config.obstacle_height,
            "sphere_xy_forbidden_radius_m": config.obstacle_radius + actual_radius + config.safety_margin,
            "candidate_grid": [21, 25],
            "contacts": "all Panda links and sphere per physics substep",
            "sphere_tracking": "sticky from first grasp through release until native success or episode end",
        },
        "limits": {"target_per_class": config.target_per_class,
                   "complete_attempts_per_class": config.max_attempts_per_class,
                   "control_steps_per_attempt": config.max_steps,
                   "total_wall_seconds": config.max_wall_seconds},
    }
    contract_path = root / "frozen_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    runner = attempt_runner or _attempt
    report = {"complete": False, "classes": {}, "attempts": [],
              "frozen_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()}
    stage_names = ("geometry_tries", "waypoint_tries", "template_executions",
        "obstacle_executions", "cylinder_candidates", "restore_executions",
        "grasp_phases", "release_phases", "stable_successes", "stable_failures")
    start_time = time.monotonic()
    episodes = []

    def persist_attempt(row):
        with (root / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def persist_episode(episode):
        with (root / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(episode) + "\n")

    try:
        with h5py.File(root / "trajectory.h5", "w") as h5:
            for class_index, kind in enumerate(("direct", "obstacle")):
                rng = np.random.default_rng(config.seed + class_index)
                accepted, failures = 0, {}
                stage_counts = dict.fromkeys(stage_names, 0)
                for attempt_index in range(config.max_attempts_per_class):
                    if time.monotonic() - start_time >= config.max_wall_seconds:
                        report["wall_timeout"] = True
                        break
                    seed = int(rng.integers(0, 2**31 - 1))
                    attempt_started = time.monotonic()
                    try:
                        outcome, reason = runner(kind, seed, rng, config)
                    except Exception as error:
                        import traceback
                        outcome, reason = {
                            "failure": "runner_exception",
                            "diagnostics": {"error": repr(error),
                                            "traceback": traceback.format_exc()},
                        }, "runner_exception"
                    row = {"class": kind, "attempt": attempt_index + 1, "seed": seed,
                           "elapsed_seconds": time.monotonic() - attempt_started}
                    if outcome is None or "failure" in outcome:
                        reason = reason or (outcome or {}).get("failure", "unspecified")
                        row.update(failure=reason,
                            counters=(outcome or {}).get("counters", {}),
                            phase_timings=(outcome or {}).get("phase_timings", {}),
                            diagnostics=(outcome or {}).get("diagnostics", {}))
                        failures[reason] = failures.get(reason, 0) + 1
                    else:
                        actions = np.asarray(outcome["actions"])
                        observations = np.asarray(outcome["obs"])
                        count = len(actions)
                        if actions.ndim != 2 or actions.shape[1] != 7:
                            raise ValueError("pilot trajectory actions must have dimension seven")
                        if observations.shape[0] != count + 1 or len(outcome["states"]) != count + 1:
                            raise ValueError("pilot trajectory requires T actions and T+1 observations/states")
                        if len(outcome["controller_states"]) != count + 1:
                            raise ValueError("pilot trajectory requires T+1 explicit controller snapshots")
                        if not outcome["native_success"] or (kind == "obstacle" and not outcome["safe_success"]):
                            raise ValueError("runner returned an unqualified native/safety result")
                        terminal = outcome["infos"][-1]
                        native_success = bool(torch.as_tensor(terminal.get(
                            "native_success", terminal.get("success", False))).reshape(-1)[0])
                        on_container = bool(torch.as_tensor(terminal["is_obj_on_bin"]).reshape(-1)[0])
                        static = bool(torch.as_tensor(terminal["is_obj_static"]).reshape(-1)[0])
                        grasped = bool(torch.as_tensor(terminal["is_obj_grasped"]).reshape(-1)[0])
                        if not (native_success and on_container and static and not grasped):
                            raise ValueError("saved last info does not prove native PlaceSphere success")
                        success_rows = np.asarray([
                            bool(torch.as_tensor(info.get("success", False)).reshape(-1)[0])
                            for info in outcome["infos"]], dtype=np.uint8)
                        if len(success_rows) != count + 1 or not success_rows[-1]:
                            raise ValueError("saved T+1 infos lack native terminal success")
                        if bool(np.asarray(outcome["truncated"]).any()) or not bool(np.asarray(outcome["terminated"])[-1]):
                            raise ValueError("native PlaceSphere success must terminate before the control-step limit")
                        index = len(episodes)
                        group = h5.create_group(f"traj_{index}")
                        group.attrs["kind"], group.attrs["seed"] = kind, seed
                        group.create_dataset("actions", data=actions)
                        group.create_dataset("obs", data=observations)
                        group.create_dataset("success", data=success_rows)
                        group.create_dataset("terminated", data=outcome["terminated"])
                        group.create_dataset("truncated", data=outcome["truncated"])
                        states = group.create_group("env_states")
                        for key, value in _stack_state_rows(outcome["states"]).items():
                            _state_write(states, key, value)
                        controllers = group.create_group("controller_states")
                        for key, value in _stack_state_rows(outcome["controller_states"]).items():
                            _state_write(controllers, key, value)
                        controllers.create_dataset("snapshot_sha256",data=np.asarray([
                            _tree_sha256(state).encode("ascii")
                            for state in outcome["controller_states"]],dtype="S64"))
                        episode = {"episode_id": index, "kind": kind, "seed": seed,
                            "layout": outcome["layout"], "trace": outcome["trace"],
                            "infos": _json(outcome["infos"]),
                            "terminated": outcome["terminated"].tolist(),
                            "truncated": outcome["truncated"].tolist()}
                        episodes.append(episode)
                        persist_episode(episode)
                        row.update(accepted=True, steps=count,
                            counters=outcome.get("counters", {}),
                            phase_timings=outcome.get("phase_timings", {}))
                        accepted += 1
                        h5.flush()

                    report["attempts"].append(row)
                    persist_attempt(row)
                    for name in stage_names:
                        stage_counts[name] += int(row.get("counters", {}).get(name, 0))
                    snapshot = {"complete": False, "attempt_count": len(report["attempts"]),
                        "classes": {**report["classes"], kind: {
                            "accepted": accepted, "target": config.target_per_class,
                            "complete_attempts": attempt_index + 1,
                            "failures": failures, "stage_counts": stage_counts}}}
                    (root / "report.json").write_text(json.dumps(snapshot, indent=2) + "\n")
                    if accepted == config.target_per_class:
                        break

                class_attempts = [row for row in report["attempts"] if row["class"] == kind]
                report["classes"][kind] = {
                    "accepted": accepted, "target": config.target_per_class,
                    "complete_attempts": len(class_attempts),
                    "attempt_budget": config.max_attempts_per_class,
                    "failures": failures, "stage_counts": stage_counts,
                    "phase_elapsed_seconds": {
                        phase: sum(float(row.get("phase_timings", {}).get(phase, 0))
                                   for row in class_attempts)
                        for phase in ("geometry", "template", "cylinder_sampling",
                                      "obstacle_execution", "complete_attempt")}}

        report["complete"] = all(
            report["classes"].get(kind, {}).get("accepted") == config.target_per_class
            for kind in ("direct", "obstacle"))
        report["elapsed_seconds"] = time.monotonic() - start_time
        (root / "trajectory.json").write_text(json.dumps({"episodes": episodes}, indent=2) + "\n")
        report["trajectory_sha256"] = hashlib.sha256((root / "trajectory.h5").read_bytes()).hexdigest()
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        names = ("trajectory.h5", "trajectory.json", "frozen_contract.json",
                 "collection_config.json", "report.json")
        (root / "artifact_hashes.json").write_text(json.dumps({
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
        }, indent=2) + "\n")
        return report
    except BaseException:
        report["interrupted"] = True
        report["attempt_count"] = len(report["attempts"])
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


def _stack_state_rows(rows):
    if isinstance(rows[0],dict):return {k:_stack_state_rows([r[k] for r in rows]) for k in rows[0]}
    return np.concatenate([x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x) for x in rows],axis=0)


def _json(x):
    if torch.is_tensor(x):return x.detach().cpu().tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_json(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def base_bin_sizes():return [0.0025,0.025,0.025]


class PlaceSpherePilotDataset(Dataset):
    def __init__(self,path,obs_horizon=2,pred_horizon=16):
        self.obs=[];self.actions=[];self.windows=[];self.class_windows={"direct":[],"obstacle":[]}
        with h5py.File(path,"r") as f:
            for key in sorted(f,key=lambda k:int(k.rsplit("_",1)[-1])):
                g=f[key];o=g["obs"][:];a=g["actions"][:]
                if a.ndim!=2 or a.shape[1]!=7 or len(o)!=len(a)+1:raise ValueError(f"{key}: expected 7D action and T+1 observations")
                if not g["success"][-1] or len(g["success"])!=len(a)+1:raise ValueError(f"{key}: missing terminal success")
                if bool(g["truncated"][:].any()) or not bool(g["terminated"][-1]):
                    raise ValueError(f"{key}: success must be a native termination, not a time limit")
                self.obs.append(torch.tensor(o,dtype=torch.float32));self.actions.append(torch.tensor(a,dtype=torch.float32))
                kind=g.attrs["kind"];kind=kind.decode() if isinstance(kind,bytes) else str(kind)
                for i in range(-obs_horizon+1,len(a)):
                    item=(len(self.obs)-1,i);self.windows.append(item);self.class_windows[kind].append(item)
                def check(group):
                    for v in group.values():
                        if isinstance(v,h5py.Group):check(v)
                        elif len(v)!=len(a)+1:raise ValueError(f"{key}: state trajectory is not T+1")
                check(g["env_states"])
                if "controller_states" not in g:raise ValueError(f"{key}: missing controller snapshots")
                check(g["controller_states"])
        self.obs_horizon,self.pred_horizon=obs_horizon,pred_horizon
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):
        t,s=self.windows[i];o=self.obs[t];a=self.actions[t]
        return {"observations":o[torch.arange(s,s+self.obs_horizon).clamp(0,len(o)-1)],"actions":a[torch.arange(s,s+self.pred_horizon).clamp(0,len(a)-1)]}


def validate_minibatch(dataset_path, output_dir, expected_obs_dim):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from examples.baselines.constrain.constrain_denoiser import _official_conditional_unet1d
    ds=PlaceSpherePilotDataset(dataset_path)
    if not ds.class_windows["direct"] or not ds.class_windows["obstacle"]:
        raise ValueError("saved minibatch validation requires both accepted classes")
    batches={}
    for kind in ("direct","obstacle"):
        selected=ds.class_windows[kind][:2]
        batches[kind]={key:torch.stack([ds[i][key] for i,_ in selected]) for key in ("observations","actions")}
    obs=batches["direct"]["observations"]
    if obs.shape[-1]!=expected_obs_dim or any(x["actions"].shape[-1]!=7 for x in batches.values()):raise ValueError("saved PlaceSphere batch dimension mismatch")
    model=_official_conditional_unet1d()(input_dim=7,global_cond_dim=obs.shape[1]*obs.shape[2],diffusion_step_embed_dim=64,down_dims=[64,128,256],n_groups=8)
    scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    evidence={}
    for kind,batch in batches.items():
        obs,actions=batch["observations"],batch["actions"]
        if obs.shape[-1]!=expected_obs_dim:raise ValueError("saved PlaceSphere observation dimension mismatch")
        noise=torch.randn_like(actions);t=torch.randint(0,100,(len(actions),));pred=model(scheduler.add_noise(actions,noise,t),t,global_cond=obs.flatten(1));loss=torch.nn.functional.mse_loss(pred,noise)
        evidence[kind]={"batch_size":len(actions),"observation_shape":list(obs.shape),"action_shape":list(actions.shape),"epsilon_loss":float(loss),"loss_finite":bool(torch.isfinite(loss))}
        if not evidence[kind]["loss_finite"]:raise FloatingPointError(f"{kind} epsilon loss is non-finite")
    result={"classes":evidence,"model":"official ConditionalUnet1D epsilon objective; interface validation only"}
    Path(output_dir).mkdir(parents=True,exist_ok=False);(Path(output_dir)/"minibatch_validation.json").write_text(json.dumps(result,indent=2)+"\n");return result


def _h5_state(node, index):
    if isinstance(node,h5py.Group):return {key:_h5_state(value,index) for key,value in node.items()}
    return node[index]


def render_representative_videos(dataset_path, output_dir, config=PlaceSpherePilotConfig()):
    import imageio.v2 as imageio
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    env=make_env(render_mode="rgb_array",config=config)
    try:
        env.reset(seed=config.seed)
        with h5py.File(dataset_path,"r") as h5:
            selected={}
            for key in sorted(h5,key=lambda k:int(k.rsplit("_",1)[-1])):
                group=h5[key];kind=group.attrs["kind"];kind=kind.decode() if isinstance(kind,bytes) else str(kind)
                if kind not in selected:selected[kind]=key
            if set(selected)!={"direct","obstacle"}:raise ValueError("representative videos require a saved episode from both classes")
            for kind,key in selected.items():
                group=h5[key]
                with imageio.get_writer(str(root/f"{kind}.mp4"),fps=env.unwrapped.control_freq,codec="libx264",macro_block_size=1) as writer:
                    for index in range(len(group["actions"])+1):
                        env.unwrapped.set_state_dict(_h5_state(group["env_states"],index))
                        frame=env.render()
                        if torch.is_tensor(frame):frame=frame.detach().cpu().numpy()
                        frame=np.asarray(frame,dtype=np.uint8)
                        if frame.ndim==4:frame=frame[0]
                        writer.append_data(frame)
    finally:env.close()
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("direct.mp4","obstacle.mp4")}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True)
    p.add_argument("--experiment-dir",required=True)
    p.add_argument("--validation-dir",required=True)
    p.add_argument("--video-dir",required=True)
    p.add_argument("--seed",type=int,default=20261022)
    p.add_argument("--target-per-class",type=int,default=20)
    p.add_argument("--max-attempts-per-class",type=int,default=200)
    p.add_argument("--max-wall-seconds",type=float,default=1800)
    p.add_argument("--max-steps",type=int,default=200)
    a=p.parse_args(argv)
    c=PlaceSpherePilotConfig(seed=a.seed,target_per_class=a.target_per_class,max_attempts_per_class=a.max_attempts_per_class,max_wall_seconds=a.max_wall_seconds,max_steps=a.max_steps)
    experiment=Path(a.experiment_dir)
    experiment.mkdir(parents=True,exist_ok=False)
    report=collect_pilot(a.output_dir,c)
    import shutil
    for name in ("collection_config.json","frozen_contract.json","report.json",
                 "attempts.jsonl","artifact_hashes.json"):
        source=Path(a.output_dir)/name
        if source.exists():shutil.copy2(source,experiment/name)
    if not report["complete"]:
        raise RuntimeError(f"pilot incomplete; not eligible for formal collection: {report['classes']}")
    env=make_env(config=c)
    try:
        _,info=env.reset(seed=c.seed)
        dim=len(_policy_observation(env.unwrapped,info))
    finally:
        env.close()
    validation=validate_minibatch(Path(a.output_dir)/"trajectory.h5",a.validation_dir,dim)
    videos=render_representative_videos(Path(a.output_dir)/"trajectory.h5",a.video_dir,c)
    evidence={"dataset":str(Path(a.output_dir)/"trajectory.h5"),
        "dataset_sha256":hashlib.sha256((Path(a.output_dir)/"trajectory.h5").read_bytes()).hexdigest(),
        "frozen_contract_sha256":hashlib.sha256((Path(a.output_dir)/"frozen_contract.json").read_bytes()).hexdigest(),
        "validation":validation,"videos":videos}
    evidence_path=Path(a.validation_dir)/"pilot_evidence.json"
    evidence_path.write_text(json.dumps(evidence,indent=2)+"\n")
    shutil.copy2(evidence_path,experiment/"pilot_evidence.json")
    for name in ("trajectory.h5","trajectory.json","collection_config.json",
                 "frozen_contract.json","report.json","artifact_hashes.json"):
        path=Path(a.output_dir)/name
        if path.exists():evidence.setdefault("artifact_hashes",{})[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (experiment/"pilot_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    return evidence


if __name__=="__main__":main()
