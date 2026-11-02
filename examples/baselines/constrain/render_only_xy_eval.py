"""Frozen selected-policy evaluation with a measured-TCP, per-step XY filter."""

import argparse
import hashlib
import json
import signal
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from examples.baselines.constrain.inference_xy_constraint_r45 import XYConstraint, validate_tcp_trace

_ACTIVE_TRACES = []
_ACTIVE_SCENE_GEOMETRY = None
EPISODE_WALL_SECONDS = 300
TRIAL_WALL_SECONDS = 3600
TCP_XY_MARGIN_M = 0.005
XY_WORKING_CLEARANCE_M = 0.003
XY_WORKING_PROFILE = "tcp_xy_working_clearance_3mm_v2"
FILTER_INTERVENTION_EPSILON = 1e-6
COMMON_LATENT_PROFILE = "common_latent_v1"
XY_FEEDBACK_PROFILE = "scheduled_non_xy_tail_auxiliary_xy_v1"
PROGRESS_GUARD_PROFILE = "bounded_task_progress_tangent_v1"
PROGRESS_GUARD_PER_PAUSE_LIMIT = 30
PROGRESS_GUARD_TOTAL_LIMIT = 60
PROGRESS_GUARD_ARC_STEP_M = 0.005
PROGRESS_GUARD_MIN_ARC_PROGRESS_M = 0.001
PROGRESS_GUARD_PATH_RADIAL_TOLERANCE_M = 0.0005


class BoundedProgressGuard:
    """Geometrically detour around the working disk without consuming task actions."""

    def __init__(self, working_radius, pending_xy, center_xy=(0.0, 0.0),
                 xy_bounds=(-0.35, 0.35), translation_scale_m=0.1):
        self.working_radius = float(working_radius)
        self.path_radius = self.working_radius + 0.003
        self.pending_xy = np.asarray(pending_xy, dtype=np.float64).reshape(2).copy()
        self.center_xy = np.asarray(center_xy, dtype=np.float64).reshape(2).copy()
        self.xy_bounds = tuple(float(x) for x in xy_bounds)
        self.translation_scale_m = float(translation_scale_m)
        self.total_controls = 0
        self.pause_controls = 0
        self.start_xy = None
        self.winding = None
        self.last_angle = None
        self.last_actual_angle = None
        self.arc_progress_m = 0.0

    def start(self, tcp_xy):
        tcp = np.asarray(tcp_xy, dtype=np.float64).reshape(2)
        radial_vector = tcp - self.center_xy
        radial = radial_vector / np.linalg.norm(radial_vector)
        cross = radial[0] * self.pending_xy[1] - radial[1] * self.pending_xy[0]
        self.winding = 1.0 if cross >= 0 else -1.0
        self.start_xy = tcp.copy()
        self.last_angle = float(np.arctan2(radial_vector[1], radial_vector[0]))
        self.last_actual_angle = self.last_angle
        self.pause_controls = 0

    def original_chord_clear(self, tcp_xy):
        from examples.baselines.constrain.inference_xy_constraint_r45 import XYConstraint
        tcp = np.asarray(tcp_xy, dtype=np.float64).reshape(2)
        requested = np.clip(self.pending_xy, *self.xy_bounds)
        endpoint = tcp + requested * self.translation_scale_m
        return not XYConstraint.segment_violates(
            tcp, endpoint, self.center_xy, self.working_radius)

    def next_xy_action(self, tcp_xy):
        tcp = np.asarray(tcp_xy, dtype=np.float64).reshape(2)
        if self.total_controls >= PROGRESS_GUARD_TOTAL_LIMIT:
            return None
        if self.pause_controls >= PROGRESS_GUARD_PER_PAUSE_LIMIT:
            return None
        radial = tcp - self.center_xy
        radius = float(np.linalg.norm(radial))
        angle = float(np.arctan2(radial[1], radial[0]))
        if radius > self.path_radius + PROGRESS_GUARD_PATH_RADIAL_TOLERANCE_M:
            target = self.center_xy + radial * (self.path_radius / radius)
        elif radius < self.path_radius - PROGRESS_GUARD_PATH_RADIAL_TOLERANCE_M:
            target = self.center_xy + radial * (self.path_radius / radius)
        else:
            arc = min(PROGRESS_GUARD_ARC_STEP_M / self.path_radius,
                      2 * np.arccos(self.working_radius / self.path_radius))
            target_angle = angle + self.winding * arc
            target = self.center_xy + self.path_radius * np.array(
                [np.cos(target_angle), np.sin(target_angle)])
            self.last_angle = target_angle
        action = (target - tcp) / self.translation_scale_m
        action = np.clip(action, self.xy_bounds[0], self.xy_bounds[1])
        self.total_controls += 1
        self.pause_controls += 1
        return action

    def observe(self, tcp_xy, previous_radius):
        radial = np.asarray(tcp_xy, dtype=np.float64).reshape(2) - self.center_xy
        radius = float(np.linalg.norm(radial))
        angle = float(np.arctan2(radial[1], radial[0]))
        progress = 0.0
        if abs(float(previous_radius) - self.path_radius) <= PROGRESS_GUARD_PATH_RADIAL_TOLERANCE_M:
            delta = (angle - self.last_actual_angle + np.pi) % (2 * np.pi) - np.pi
            progress = delta * self.winding * self.path_radius
            if progress < PROGRESS_GUARD_MIN_ARC_PROGRESS_M:
                return False, progress, radius
            self.arc_progress_m += progress
        self.last_actual_angle = angle
        return True, progress, radius


class TaskProgressStall(RuntimeError):
    """A bounded XY pause could not clear the pending learned direction."""


def native_gripper_hold_from_qpos(qpos, lower=-0.01, upper=0.04):
    """Encode the current Panda finger position in its normalized native action."""
    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.isfinite(values).all() or upper <= lower:
        raise ValueError("finite native gripper joint position is required")
    return float(np.clip(2.0 * (values[0] - lower) / (upper - lower) - 1.0, -1.0, 1.0))


class EpisodeTimeout(TimeoutError):
    pass


class CommonInitialNoise:
    """Reuse each episode's first inference RNG state for PlaceSphere replans."""

    def __init__(self, adapter, device):
        import torch
        self.adapter = adapter
        self.original_get_action = adapter.get_action
        self.device = torch.device(device)
        self.plan_count = 0
        self.cpu_state = None
        self.cuda_device = None
        self.cuda_state = None
        self.initial_state_sha256 = None

    def get_action(self, condition):
        import torch
        if self.cpu_state is None:
            self.cpu_state = torch.get_rng_state().clone()
            states = [self.cpu_state.numpy().tobytes()]
            if self.device.type == "cuda":
                self.cuda_device = (self.device.index if self.device.index is not None
                                    else torch.cuda.current_device())
                self.cuda_state = torch.cuda.get_rng_state(self.cuda_device).clone()
                states.append(self.cuda_state.cpu().numpy().tobytes())
            self.initial_state_sha256 = hashlib.sha256(b"".join(states)).hexdigest()
            result = self.original_get_action(condition)
        else:
            devices = [] if self.cuda_device is None else [self.cuda_device]
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(self.cpu_state)
                if self.cuda_device is not None:
                    torch.cuda.set_rng_state(self.cuda_state, self.cuda_device)
                result = self.original_get_action(condition)
        self.plan_count += 1
        return result

    def metadata(self, episode_seed):
        return {"profile": COMMON_LATENT_PROFILE,
                "strategy": "restore_first_scheduled_inference_rng_state_per_replan",
                "episode_seed": int(episode_seed), "plan_count": self.plan_count,
                "first_inference_rng_state_sha256": self.initial_state_sha256,
                "actor_sampler": "DDIM eta=0; 100 diffusion steps; unchanged checkpoint"}


def configure_place_common_noise(adapter, task, event_replan, device):
    if task != "placesphere" or not event_replan:
        return None
    policy = CommonInitialNoise(adapter, device)
    adapter.get_action = policy.get_action
    return policy


class PlaceXYFeedback:
    """Keep scheduled non-XY channels while accepting auxiliary-plan XY only."""

    def __init__(self, adapter, scene_seed, device):
        self.adapter = adapter
        self.scene_seed = int(scene_seed)
        self.device = device
        self.scheduled_chunk_count = 0
        self.auxiliary_queries = []

    def start_chunk(self, scheduled_actions):
        self.scheduled_chunk_count += 1
        self.scheduled_actions = scheduled_actions

    def action_for_control(self, offset, auxiliary_plan=None, auxiliary_offset=0):
        import torch
        scheduled = self.scheduled_actions[0, offset].clone()
        if auxiliary_plan is not None:
            scheduled[:2] = auxiliary_plan[0, auxiliary_offset, :2]
        return scheduled


def auxiliary_xy_plan(adapter, condition, scene_seed, control_index, device):
    """Infer one updated-condition plan without advancing scheduled-plan RNG."""
    import torch
    device = torch.device(device)
    seed = (int(scene_seed) * 1_000_003 + int(control_index)) % (2**63 - 1)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        prediction = adapter.get_action(condition)
        mapper = getattr(adapter, "map_action", None) or adapter.map_action_ee_delta_pose
        return mapper(prediction)


def _episode_timeout(_signum, _frame):
    raise EpisodeTimeout(f"episode exceeded {EPISODE_WALL_SECONDS}s wall limit")

def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _radius(base, task):
    return float(base.obstacle_collision_radius) + TCP_XY_MARGIN_M


def _update_task_phase(env, task, info):
    base = env.unwrapped
    object_name = "cube" if task == "pickcube" else "obj"
    target_name = "goal_site" if task == "pickcube" else "bin"
    if not hasattr(base, object_name) or not hasattr(base, target_name):
        env._render_only_xy_native_flags = _jsonable(info)
        return
    state = getattr(env, "_render_only_xy_phase_state", None)
    if state is None:
        state = {"grasped": False, "lifted": False, "transported": False,
                 "released": False, "initial_object_z": None}
        env._render_only_xy_phase_state = state
    if task == "pickcube":
        obj, goal = base.cube.pose.p[0], base.goal_site.pose.p[0, :2]
        grasp_flag = info.get("is_grasped", False)
    else:
        obj, goal = base.obj.pose.p[0], base.bin.pose.p[0, :2]
        grasp_flag = info.get("is_obj_grasped", False)
    obj_xyz = _array(obj).reshape(-1)
    goal_xy = _array(goal).reshape(-1)[:2]
    if state["initial_object_z"] is None:
        state["initial_object_z"] = float(obj_xyz[2])
    grasped_now = bool(np.asarray(_array(grasp_flag)).reshape(-1)[0])
    state["grasped"] |= grasped_now
    state["lifted"] |= float(obj_xyz[2]) > state["initial_object_z"] + 0.04
    state["transported"] |= np.linalg.norm(obj_xyz[:2] - goal_xy) < 0.10
    if task == "placesphere":
        state["released"] |= bool(state["grasped"] and not grasped_now)
    success = bool(np.asarray(_array(info.get("native_success", False))).reshape(-1)[0])
    if success:
        phase = "task_success"
    elif not state["grasped"]:
        phase = "no_grasp"
    elif not state["lifted"]:
        phase = "grasp_no_lift"
    elif not state["transported"]:
        phase = "grasped_no_transport"
    elif task == "placesphere" and state["released"]:
        phase = "release_or_stability_failed"
    else:
        phase = "transport_or_place_failed"
    env._render_only_xy_task_phase = phase
    env._render_only_xy_native_flags = _jsonable(info)


def score_native_xy_trace(trace, initial_tcp_xy, expected_center_xy, expected_radius):
    """Independently score the native TCP samples/chords against frozen scene geometry."""
    from examples.baselines.constrain.inference_xy_constraint_r45 import validate_tcp_trace

    center = np.asarray(expected_center_xy, dtype=np.float64).reshape(2)
    radius = float(expected_radius)
    points = [np.asarray(initial_tcp_xy, dtype=np.float64).reshape(2)]
    for step in trace:
        live_center = np.asarray(step["obstacle_center_xy"], dtype=np.float64).reshape(2)
        live_radius = float(step["forbidden_radius_m"])
        if not np.allclose(live_center, center, atol=1e-6, rtol=0) or abs(live_radius-radius) > 1e-7:
            raise ValueError("live obstacle geometry differs from frozen scene geometry")
        points.append(np.asarray(step["tcp_xy_after"], dtype=np.float64).reshape(2))
    return validate_tcp_trace(points, center, radius,
                              first_success_step=next((i for i, step in enumerate(trace, 1)
                                                      if step.get("native_success", False)), None))


def instrument_env(env, task, *, control_enabled=True, event_replan=False,
                   xy_feedback=False, xy_constraint=True, progress_guard=False):
    """Wrap the native step seam; filter each just-in-time action using live TCP."""
    base = env.unwrapped
    if getattr(base, "obstacle_mode", None) != "render_only":
        raise ValueError("XY evaluation requires an explicitly render_only environment")
    obstacle = base.obstacle
    if any(len(body.collision_shapes) != 0 for body in obstacle._bodies):
        raise ValueError("render_only obstacle unexpectedly has collision shapes")
    import sapien
    visual = obstacle._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
    if visual is None or visual.visibility <= 0:
        raise ValueError("render_only obstacle visual is absent or invisible")
    actor = getattr(base.agent, "tcp", None)
    if actor is None:
        actor = base.agent.robot.links_map["panda_hand_tcp"]
    trace = []
    attempts = []
    original_step = env.step
    constraint = XYConstraint(translation_scale_m=0.1, action_limit=0.35,
                              working_clearance_m=XY_WORKING_CLEARANCE_M,
                              xy_action_bounds=(-1.0, 1.0) if task == "pickcube" else (-0.35, 0.35))
    guard = None
    if progress_guard:
        if task != "placesphere":
            raise ValueError("bounded progress guard is PlaceSphere-only")
        finger_qpos = _array(base.agent.robot.get_qpos()).reshape(-1, 9)[0, -2:]
        initial_gripper_hold = native_gripper_hold_from_qpos(finger_qpos)
    else:
        initial_gripper_hold = None
    last_task_gripper = initial_gripper_hold
    physical_control_index = 0
    guard_total_controls = 0

    def step(action):
        nonlocal guard, last_task_gripper, physical_control_index, guard_total_controls
        requested = _array(action).reshape(1, -1)[0].astype(float)
        tcp = _array(actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
        center = _array(base._obstacle_center).reshape(-1, 2)[0].astype(float)
        radius = _radius(base, task)
        env._render_only_xy_center = center.tolist()
        env._render_only_xy_radius = float(radius)
        if env._render_only_xy_initial_tcp_xy is None:
            env._render_only_xy_initial_tcp_xy = tcp.tolist()
        visual_actor = obstacle._objs[0]
        if hasattr(visual_actor, "pose"):
            visual_center = _array(visual_actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
            if not np.allclose(center, visual_center, atol=1e-6, rtol=0):
                raise ValueError("live task-state center differs from visible cylinder pose")
        attempt = {"control_index": len(trace) + 1, "tcp_xy_before": tcp.tolist(),
                   "tcp_xy_after": tcp.tolist(), "requested_action": requested.tolist(),
                   "filtered_action": None, "env_step_executed": False,
                   "obstacle_center_xy": center.tolist(), "forbidden_radius_m": radius,
                   "native_success": False,
                   "task_phase": getattr(env, "_render_only_xy_task_phase", None),
                   "last_native_flags": getattr(env, "_render_only_xy_native_flags", None)}
        attempts.append(attempt)
        env._render_only_xy_failed_control_index = attempt["control_index"]
        filtered = (constraint.filter_action(requested, tcp, center, radius)
                    if xy_constraint else requested.copy())
        bounds = np.asarray(constraint.xy_action_bounds, dtype=float)
        bounded_requested = np.clip(requested[:2], bounds[0], bounds[1])
        changed = bool(np.any(np.abs(filtered[:2] - bounded_requested) > FILTER_INTERVENTION_EPSILON))
        task_action_executed = True
        pause_action = None
        if progress_guard and xy_constraint:
            if guard is None and changed:
                guard = BoundedProgressGuard(
                    radius + XY_WORKING_CLEARANCE_M, requested[:2], center,
                    xy_bounds=constraint.xy_action_bounds,
                    translation_scale_m=constraint.translation_scale_m)
                guard.total_controls = guard_total_controls
                guard.start(tcp)
            if guard is not None:
                if guard.original_chord_clear(tcp):
                    attempt["progress_guard_event"] = "resume_pending_task_action"
                    attempt["progress_guard_arc_progress_m"] = guard.arc_progress_m
                    guard = None
                else:
                    detour_xy = guard.next_xy_action(tcp)
                    if detour_xy is None:
                        attempt.update(progress_guard_event="bounded_progress_stall",
                                       task_action_executed=False, env_step_executed=False,
                                       progress_guard_total_controls=guard.total_controls,
                                       progress_guard_pause_controls=guard.pause_controls)
                        env._render_only_xy_progress_stall = {
                            "type": "bounded_progress_stall",
                            "physical_control_index": physical_control_index,
                            "scheduled_task_action_index": attempt["control_index"],
                            "pending_xy": guard.pending_xy.tolist(),
                            "tcp_xy": tcp.tolist(), "arc_progress_m": guard.arc_progress_m,
                            "pause_controls": guard.pause_controls,
                            "total_pause_controls": guard.total_controls,
                        }
                        raise TaskProgressStall("bounded XY progress guard exhausted")
                    pause_action = np.zeros(7, dtype=float)
                    pause_action[:2] = detour_xy
                    pause_action[6] = last_task_gripper
                    pause_action = constraint.filter_action(pause_action, tcp, center, radius)
                    guard_total_controls = guard.total_controls
                    task_action_executed = False
                    attempt["progress_guard_event"] = "tangent_detour_pause"
                    attempt["progress_guard_pending_xy"] = guard.pending_xy.tolist()
                    attempt["progress_guard_winding"] = guard.winding
                    attempt["progress_guard_arc_progress_m"] = guard.arc_progress_m
                    attempt["progress_guard_pause_control"] = guard.pause_controls
                    attempt["progress_guard_total_control"] = guard.total_controls
                    attempt["progress_guard_tcp_radius_before_m"] = float(
                        np.linalg.norm(tcp - center))
                    attempt["pending_task_action_index"] = attempt["control_index"]
                    attempt["gripper_hold_native"] = float(last_task_gripper)
                    filtered = pause_action
        attempt["filtered_action"] = filtered.tolist()
        attempt["task_action_executed"] = task_action_executed
        attempt["physical_control_index"] = physical_control_index + 1
        attempt["requested_replan"] = bool(event_replan and changed)
        attempt["xy_feedback_requested"] = bool(xy_feedback and changed)
        attempt["replan_consumed_at_control_index"] = None
        env._render_only_xy_replan_requested = bool(event_replan and changed)
        attempt["env_step_executed"] = True
        try:
            result = original_step(filtered.reshape(1, 7))
        except Exception:
            attempt["tcp_xy_after"] = _array(actor.pose.p).reshape(-1, 3)[0, :2].astype(float).tolist()
            attempt["env_step_error"] = True
            attempt["step"] = len(trace) + 1
            trace.append(attempt)
            raise
        info = result[-1]
        physical_control_index += 1
        if task_action_executed:
            last_task_gripper = float(np.clip(requested[6], -1.0, 1.0))
        _update_task_phase(env, task, info)
        tcp_after = _array(actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
        if guard is not None and not task_action_executed:
            ok, arc_progress, tcp_radius = guard.observe(
                tcp_after, attempt["progress_guard_tcp_radius_before_m"])
            attempt["progress_guard_actual_arc_progress_m"] = float(arc_progress)
            attempt["progress_guard_tcp_radius_after_m"] = float(tcp_radius)
            if not ok or tcp_radius < guard.working_radius - 1e-9:
                env._render_only_xy_progress_stall = {
                    "type": "bounded_progress_stall",
                    "reason": "minimum_tangent_progress_or_acceptance_radius_failed",
                    "physical_control_index": physical_control_index,
                    "scheduled_task_action_index": attempt["control_index"],
                    "pending_xy": guard.pending_xy.tolist(), "tcp_xy": tcp_after.tolist(),
                    "arc_progress_m": guard.arc_progress_m,
                    "actual_arc_progress_m": float(arc_progress),
                    "pause_controls": guard.pause_controls,
                    "total_pause_controls": guard.total_controls,
                }
        attempt.update(step=len(trace) + 1, tcp_xy_after=tcp_after.tolist(),
                       native_success=bool(np.asarray(info.get("native_success", False)).reshape(-1)[0]),
                       info=_jsonable(info))
        trace.append(attempt)
        env._render_only_xy_last_attempt = attempt
        env._render_only_xy_failed_control_index = None
        return result

    env.step = step
    env._render_only_xy_trace = trace
    env._render_only_xy_attempts = attempts
    env._render_only_xy_initial_tcp_xy = None
    env._render_only_xy_failed_control_index = None
    env._render_only_xy_center = None
    env._render_only_xy_radius = None
    env._render_only_xy_task_phase = None
    env._render_only_xy_native_flags = None
    env._render_only_xy_phase_state = None
    env._render_only_xy_event_replan = bool(event_replan)
    env._render_only_xy_feedback = bool(xy_feedback)
    env._render_only_xy_constraint = bool(xy_constraint)
    env._render_only_xy_replan_requested = False
    env._render_only_xy_last_attempt = None
    env._render_only_xy_progress_stall = None
    env._render_only_xy_progress_guard = bool(progress_guard)
    env._render_only_xy_expected_geometry = _ACTIVE_SCENE_GEOMETRY
    _ACTIVE_TRACES.append((trace, attempts, env))
    return trace


def consume_replan_request(env, control_index):
    """Consume a filter-triggered replan before the next policy inference."""
    if not getattr(env, "_render_only_xy_replan_requested", False):
        return False
    env._render_only_xy_replan_requested = False
    attempt = getattr(env, "_render_only_xy_last_attempt", None)
    if attempt is not None:
        attempt["replan_consumed_at_control_index"] = int(control_index)
    return True


def should_replan_after_control(env, termination):
    return termination is None and bool(getattr(env, "_render_only_xy_replan_requested", False))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    value = _array(value)
    return value.item() if value.ndim == 0 else value.tolist()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_jsonable) + "\n")
    temporary.replace(path)


def freeze_registration(path, task, indices, sources, inputs=None, *, event_replan=False,
                        common_latent_replan=False, xy_feedback=False, xy_constraint=True,
                        progress_guard=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_root = path.parent / (path.stem + "_sources")
    if snapshot_root.exists():
        raise FileExistsError(snapshot_root)
    snapshot_root.mkdir(parents=True)
    snapshots = {}
    for name, src in sorted(sources.items()):
        target = snapshot_root / f"{name}_{Path(src).name}"
        shutil.copyfile(src, target)
        snapshots[name] = {"path": str(target), "sha256": sha256(target)}
    repo_root = Path(__file__).resolve().parents[3]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                       text=True).strip()
    dirty_paths = subprocess.check_output(["git", "status", "--short"], cwd=repo_root,
                                           text=True).splitlines()
    body = {"format": "render_only_xy_eval_r47_v1", "task": task,
             "method": ("ConstrainNet+bufferedXY+bounded_progress_guard_v1"
                        if progress_guard else
                        "ConstrainNet+unfiltered_render_only_reference"
                        if not xy_constraint else
                        "ConstrainNet+bufferedXY+scheduled_non_xy_xy_feedback_v1"
                        if xy_feedback else
                        "ConstrainNet+bufferedXY+event_replan+common_latent_v1"
                        if common_latent_replan else
                       "ConstrainNet+bufferedXY+event_replan_v1" if event_replan
                       else "ConstrainNet+XY constraint"),
            "indices": [int(i) for i in indices],
            "git_head_context": git_head, "worktree_dirty_paths": dirty_paths,
            "constraint_profile": {"name": "tcp_xy_cylinder_radius_plus_5mm_v1",
                 "working_clearance_profile": XY_WORKING_PROFILE,
                 "working_clearance_m": XY_WORKING_CLEARANCE_M,
                   "event_replan": bool(event_replan),
                   "common_latent_replan": bool(common_latent_replan),
                    "xy_feedback": bool(xy_feedback),
                    "bounded_progress_guard": bool(progress_guard),
                    "progress_guard_profile": (PROGRESS_GUARD_PROFILE if progress_guard else None),
                    "progress_guard_per_pause_controls": (PROGRESS_GUARD_PER_PAUSE_LIMIT
                                                          if progress_guard else None),
                    "progress_guard_total_controls": (PROGRESS_GUARD_TOTAL_LIMIT
                                                       if progress_guard else None),
                    "progress_guard_arc_step_m": (PROGRESS_GUARD_ARC_STEP_M
                                                   if progress_guard else None),
                    "progress_guard_path_radial_tolerance_m": (
                        PROGRESS_GUARD_PATH_RADIAL_TOLERANCE_M if progress_guard else None),
                    "progress_guard_path_radius_extra_m": (XY_WORKING_CLEARANCE_M
                                                           if progress_guard else None),
                   "xy_constraint": bool(xy_constraint),
                   "xy_feedback_rng": ("forked_torch_cpu_cuda_rng_scene_seed_control_index_v1"
                                      if xy_feedback else None),
                  "noise_strategy": ("restore_first_scheduled_inference_rng_state_per_replan"
                                     if common_latent_replan else "independent_per_inference"),
                 "filter_intervention_epsilon": FILTER_INTERVENTION_EPSILON,
                "controller": "pd_ee_delta_pose", "frame": "world XY",
                "translation_scale_m_per_action_unit": 0.1,
                "xy_action_bounds": [-1.0, 1.0] if task == "pickcube" else [-0.35, 0.35],
                "non_xy_channels": "pass through",
                "forbidden_radius": "restored visible cylinder radius + 0.005 m",
                "excluded_profiles": ["tcp_xy_object_footprint_r45", "physical_object_latch"],
                "body_collision_gate": False, "obstacle_mode": "render_only"},
            "formal_or_generalization_claim": False,
            "source_snapshots": snapshots,
            "inputs": inputs or {},
            "sources": {name: {"path": str(Path(src)), "sha256": sha256(src)}
                        for name, src in sorted(sources.items())}}
    body["registration_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def validate_registration(path):
    reg = json.loads(Path(path).read_text())
    body = dict(reg)
    recorded = body.pop("registration_sha256", None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if recorded != actual:
        raise ValueError("registration checksum mismatch")
    for name, item in reg["sources"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"registered source mismatch: {name}")
    for name, item in reg["inputs"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"registered input mismatch: {name}")
    for name, item in reg["source_snapshots"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"source snapshot mismatch: {name}")
    return reg


def summarize_rows(rows, requested):
    complete = [r for r in rows if r.get("status") == "evaluated"]
    return {"denominator": int(requested), "evaluated": len(complete),
            "runtime_errors": int(requested) - len(complete),
            "native_success_count": sum(bool(r.get("native_success")) for r in complete),
            "xy_zero_count": sum(bool(r.get("xy_zero")) for r in complete)}


def _runtime_error_row(index, exc):
    active = _ACTIVE_TRACES[-1] if _ACTIVE_TRACES else ([], [], None)
    trace, attempts, env = active
    initial = getattr(env, "_render_only_xy_initial_tcp_xy", None) if env is not None else None
    expected = getattr(env, "_render_only_xy_expected_geometry", None) if env is not None else None
    center = expected["center"] if expected is not None else None
    radius = expected["radius"] if expected is not None else None
    if initial is not None and center is not None and radius is not None:
        validation = score_native_xy_trace(trace, initial, center, radius)
    else:
        validation = {"zero_xy_violations": False, "violating_samples": [],
                      "violating_segments": [], "geometry_captured": False}
    return {"scene_index": index, "status": "runtime_error",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_control_index": (getattr(env, "_render_only_xy_failed_control_index", None)
                                      if env is not None else None),
            "initial_tcp_xy": initial, "attempted_actions": attempts,
            "executed_trace": trace, "first_native_success_control_index": next(
                (step["step"] for step in trace if step.get("native_success")), None),
             "native_task_phase": (attempts[-1].get("task_phase") if attempts else None),
             "task_progress_stall": (getattr(env, "_render_only_xy_progress_stall", None)
                                     if env is not None else None),
             "xy_validation": validation, "xy_zero": bool(validation.get("zero_xy_violations", False)),
            "native_success": False}


def validate_explicit_catalog(registration, task, indices, scenes_dir):
    """Require explicit execution catalogs to be the exact preregistered inputs."""
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("scene indices must be unique and non-negative")
    scenes_dir = Path(scenes_dir)
    inputs = registration.get("inputs", {})
    for key, path in (("scene_manifest", scenes_dir / "manifest.json"),
                      ("scene_h5", scenes_dir / "scenes.h5")):
        item = inputs.get(key)
        if item is None or Path(item["path"]).resolve() != path.resolve():
            raise ValueError("explicit scenes differ from registered scene catalog")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"registered input mismatch: {key}")
    module_name = ("examples.baselines.constrain.pickcube_transport_detour" if task == "pickcube"
                   else "examples.baselines.constrain.placesphere_transport_detour")
    module = __import__(module_name, fromlist=["TransportDetourSceneSet"])
    catalog = module.TransportDetourSceneSet(scenes_dir)
    if any(index >= len(catalog) for index in indices):
        raise ValueError("requested scene index is outside explicit catalog")
    return catalog


def _install_render_only_step_wrapper(task, *, event_replan=False, xy_feedback=False,
                                      xy_constraint=True, progress_guard=False):
    import gymnasium as gym
    original_make = gym.make

    def make(*args, **kwargs):
        kwargs["obstacle_mode"] = "render_only"
        env = original_make(*args, **kwargs)
        instrument_env(env, task, event_replan=event_replan, xy_feedback=xy_feedback,
                       xy_constraint=xy_constraint, progress_guard=progress_guard)
        return env
    gym.make = make
    return original_make


def _evaluate_one(task, index, device, scenes_dir=None, *, common_latent_replan=False,
                  xy_feedback=False, xy_constraint=True, progress_guard=False):
    global _ACTIVE_SCENE_GEOMETRY
    _ACTIVE_TRACES.clear()
    if task == "pickcube":
        from examples.baselines.constrain import pickcube_body_hint_round41 as r41
        from examples.baselines.constrain import pickcube_xy_gauge_round40 as r40
        from examples.baselines.constrain import pickcube_zscore_round37 as engine
        from examples.baselines.constrain.pickcube_clearance_eval_r44 import (
            CONTROL, STATS_PATH, BASE, DEV_SCENES,
        )
        from examples.baselines.constrain.pickcube_clearance_train_r44 import (
            TRAIN_FORMAT, MODEL_KIND, TOTAL, EMA_POWER,
        )
        from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG, body_hint_from_raw
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
        from examples.baselines.constrain.evaluate_pickcube_paired import PickCubePairedEvaluator
        import torch
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.training_utils import EMAModel
        r41._install_profile(); r40._install_evaluation_loaders()
        ck = torch.load(CONTROL, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        if (ck.get("format") != TRAIN_FORMAT or cfg.get("model_kind") != MODEL_KIND
                or cfg.get("horizon") != 8 or ck.get("step") != TOTAL
                or ck.get("ema") is None or cfg.get("selected_weights") != "ema"):
            raise ValueError("R44 selected EMA checkpoint profile mismatch")
        model = engine.build_constrain_denoiser(CONTROL_CONFIG, device="cpu")
        model.load_state_dict(ck["model"], strict=True)
        ema = EMAModel(parameters=model.trainable_parameters(), power=EMA_POWER)
        ema.load_state_dict(ck["ema"]); ema.copy_to(model.trainable_parameters())
        model.to(device).eval()
        stats = engine.load_stats(STATS_PATH)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                  clip_sample=True, prediction_type="epsilon")
        scheduler.set_timesteps(100)
        class Adapter:
            def __init__(self):
                self.device = torch.device(device); self.model = model; self.stats = stats
                self.scheduler = scheduler
                self.metadata = {"execution_horizon": 8}
            def base_cond(self, raw):
                raw = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
                if raw.ndim == 2: raw = raw[None]
                if raw.shape[-2:] != (2, 45): raise ValueError("R44 actor requires (B,2,45)")
                return raw
            def get_action(self, raw):
                cond = engine.normalize_observation(raw[..., :42], self.stats).flatten(1)
                hint = body_hint_from_raw(raw)[:, None].expand(-1, 16, -1)
                sample = torch.randn((len(raw), 16, 7), device=self.device)
                with torch.no_grad():
                    for timestep in self.scheduler.timesteps:
                        sample = self.scheduler.step(
                            self.model(sample, timestep, global_cond=cond, hint=hint),
                            timestep, sample).prev_sample
                return sample[:, 1:9]
            @staticmethod
            def map_action_ee_delta_pose(action): return torch.as_tensor(action)
        catalog = TransportDetourSceneSet(scenes_dir or DEV_SCENES)
        scene = catalog.load(index)
        _ACTIVE_SCENE_GEOMETRY = {"center": scene["metadata"]["cylinder"]["center_xy"],
                                  "radius": float(scene["metadata"]["cylinder"]["radius"]) + TCP_XY_MARGIN_M}
        runner = PickCubePairedEvaluator(Adapter(), catalog, "constrain_dp", ".", device, max_steps=500)
        row = runner.run_episode(index, scene)
    else:
        from examples.baselines.constrain import placesphere_base_dp as base_mod
        from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
        from examples.baselines.constrain.evaluate_placesphere_paired import PlaceSpherePairedEvaluator
        ckpt = Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt")
        base = Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base_verified/final.pt")
        adapter = PlaceSphereConstrainAdapter(ckpt, device=device, control_enabled=True,
                                              base_checkpoint=base, sampler="ddim", eta=0.0)
        noise_policy = (configure_place_common_noise(adapter, task, True, device)
                        if common_latent_replan else None)
        from examples.baselines.constrain.placesphere_action_codec import CONTROLLER_RANGE
        from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
        transport_catalog = TransportDetourSceneSet(
            scenes_dir or "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
        class Catalog:
            def __len__(self): return len(transport_catalog)
            def load(self, i): return transport_catalog.load(i)
            def restore(self, i, env):
                env, result = transport_catalog.restore(i, env)
                scene = transport_catalog.load(i)
                return env, {"raw_observation": scene["raw_observation"],
                             "bit_exact": result["bit_exact"], "norms": result["norms"]}
        catalog = Catalog()
        scene = catalog.load(index)
        _ACTIVE_SCENE_GEOMETRY = {"center": scene["metadata"]["cylinder"]["center_xy"],
                                  "radius": float(scene["metadata"]["cylinder"]["radius"]) + TCP_XY_MARGIN_M}
        if xy_feedback:
            runner = PlaceSpherePairedEvaluator(adapter, catalog, ".", device)
            runner.xy_feedback = True
        else:
            runner = PlaceSpherePairedEvaluator(adapter, catalog, ".", device)
        runner.progress_guard = progress_guard
        row = runner.run_episode(index, scene)
        if noise_policy is not None:
            row["policy_sampling"] = noise_policy.metadata(scene["metadata"]["seed"])
    traces = [item for item in _ACTIVE_TRACES if item[0]]
    trace = traces[-1][0] if traces else []
    center = np.asarray(scene["metadata"]["cylinder"]["center_xy"], dtype=float)
    radius = float(scene["metadata"]["cylinder"]["radius"]) + 0.005
    initial_tcp = trace[0]["tcp_xy_before"] if trace else []
    report = score_native_xy_trace(trace, initial_tcp, center, radius) if trace else {
        "zero_xy_violations": False, "violating_samples": [], "violating_segments": []}
    row["native_success"] = bool(row.get("result", {}).get("native_success", False))
    row["actual_tcp_trace"] = trace
    if xy_feedback:
        row["policy_sampling"] = {
            "profile": XY_FEEDBACK_PROFILE,
            "scene_seed": int(scene["metadata"]["seed"]),
            "rng_strategy": "torch.random.fork_rng; auxiliary seed=(scene_seed*1000003+control_index)%(2**63-1)",
            "scheduled_plan_stride_controls": 8,
            "scheduled_full_plan_count": (len(row.get("actions_native_pd_ee_delta_pose", [])) + 7) // 8,
            "auxiliary_xy_query_count": sum(bool(step.get("auxiliary_query")) for step in trace),
            "auxiliary_channels": ["x", "y"],
            "sampler": "DDIM eta=0; 100 steps; unchanged R30 EMA",
        }
    if progress_guard:
        row["progress_guard_profile"] = {
            "name": PROGRESS_GUARD_PROFILE,
            "winding": "sign(cross(initial_radial, pending_learned_xy)); positive on tie",
            "path_radius_m": "live_forbidden_radius + 0.005 + 0.003 + 0.003",
            "arc_step_m": PROGRESS_GUARD_ARC_STEP_M,
            "minimum_actual_arc_progress_m": PROGRESS_GUARD_MIN_ARC_PROGRESS_M,
            "path_radial_tolerance_m": PROGRESS_GUARD_PATH_RADIAL_TOLERANCE_M,
            "minimum_actual_working_radius_m": "live_forbidden_radius + 0.003",
            "per_pause_controls": PROGRESS_GUARD_PER_PAUSE_LIMIT,
            "total_controls": PROGRESS_GUARD_TOTAL_LIMIT,
            "resume_condition": "original pending learned XY chord clears working radius",
            "task_action_clock": "advances only when original scheduled action executes",
        }
    row["xy_constraint"] = {"center_xy": center.tolist(), "forbidden_radius_m": radius,
                             "controller_translation_scale_m": 0.1,
            "controller_pose_action_limit": (1.0 if task == "pickcube" else 0.35),
            "working_clearance_profile": XY_WORKING_PROFILE,
            "working_clearance_m": XY_WORKING_CLEARANCE_M,
                             "visual_only_obstacle": True, "collision_shapes": 0}
    row["xy_validation"] = report
    row["xy_zero"] = bool(report["zero_xy_violations"])
    row["method_label"] = ("ConstrainNet+bufferedXY+bounded_progress_guard_v1"
                           if progress_guard else
                           "ConstrainNet unfiltered render-only reference"
                           if not xy_constraint else
                           "ConstrainNet+bufferedXY+Place scheduled-nonXY XY-feedback"
                           if xy_feedback else "ConstrainNet+XY constraint")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pickcube", "placesphere"), required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--scenes", type=Path,
                        help="explicit task-native scene catalog (defaults to the historical catalog)")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--event-replan", action="store_true")
    parser.add_argument("--common-latent-replan", action="store_true")
    parser.add_argument("--xy-feedback", action="store_true")
    parser.add_argument("--bounded-progress-guard", action="store_true")
    parser.add_argument("--xy-constraint", choices=("on", "off"), default="on",
                        help="off preserves every actor action and scores actual violations")
    args = parser.parse_args(argv)
    if args.common_latent_replan and (args.task != "placesphere" or not args.event_replan):
        raise ValueError("common latent replanning is PlaceSphere-only and requires --event-replan")
    if args.xy_feedback and args.task != "placesphere":
        raise ValueError("XY feedback is PlaceSphere-only")
    if args.bounded_progress_guard and (args.task != "placesphere" or args.xy_feedback
                                        or args.common_latent_replan or args.event_replan):
        raise ValueError("bounded progress guard is a separate PlaceSphere-only profile")
    xy_constraint = args.xy_constraint == "on"
    if not xy_constraint and (args.event_replan or args.common_latent_replan or args.xy_feedback):
        raise ValueError("unfiltered reference cannot enable filtering/replanning modes")
    if args.output.exists():
        raise FileExistsError(args.output)
    source_paths = {"runner": Path(__file__),
                    "diagnostic_test": Path(__file__).parent / "tests" / "test_render_only_xy_eval.py",
                    "filter": Path(__file__).with_name("inference_xy_constraint_r45.py"),
                    "controller": Path("mani_skill/agents/robots/panda/panda.py")}
    if args.task == "pickcube":
        from examples.baselines.constrain.pickcube_clearance_eval_r44 import CONTROL, DEV_SCENES, STATS_PATH
        checkpoint = CONTROL
        scenes = args.scenes or Path(DEV_SCENES)
        source_paths.update({"task": Path("examples/baselines/constrain/pickcube_obstacle.py"),
                             "actor": Path("examples/baselines/constrain/pickcube_clearance_train_r44.py"),
                             "rollout": Path("examples/baselines/constrain/evaluate_pickcube_paired.py"),
                             "codec": Path("examples/baselines/constrain/pickcube_action_codec.py"),
                             "normalizer_engine": Path("examples/baselines/constrain/pickcube_zscore_round37.py"),
                             "profile_r40": Path("examples/baselines/constrain/pickcube_xy_gauge_round40.py"),
                             "profile_r41": Path("examples/baselines/constrain/pickcube_body_hint_round41.py"),
                             "base_actor": Path("examples/baselines/constrain/pickcube_base_dp.py"),
                             "normalizer": Path("examples/baselines/constrain/pickcube_observation_norm.py"),
                             "denoiser": Path("examples/baselines/constrain/constrain_denoiser.py"),
                             "dataset_decoder": Path("examples/baselines/constrain/materialize_pickcube_fullscene500.py"),
                             "scene_catalog": Path("examples/baselines/constrain/pickcube_scenes.py"),
                             "scene_loader": Path("examples/baselines/constrain/pickcube_transport_detour.py"),
                             "task_env_factory": Path("examples/baselines/constrain/pickcube_expert.py")})
        input_paths = {"checkpoint": checkpoint, "normalizer": STATS_PATH,
                       "scene_manifest": scenes / "manifest.json", "scene_h5": scenes / "scenes.h5"}
        from examples.baselines.constrain.pickcube_clearance_train_r44 import TRAIN_REG
        from examples.baselines.constrain import pickcube_zscore_round37 as engine
        input_paths.update({"training_registration": TRAIN_REG,
                            "data_h5": Path(engine.DATA_H5),
                            "data_manifest": Path(engine.DATA_MANIFEST)})
    else:
        checkpoint = Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt")
        scenes = args.scenes or Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
        source_paths.update({"task": Path("examples/baselines/constrain/placesphere_obstacle.py"),
                             "actor": Path("examples/baselines/constrain/placesphere_constrain.py"),
                             "rollout": Path("examples/baselines/constrain/evaluate_placesphere_paired.py"),
                             "scene_catalog": Path("examples/baselines/constrain/placesphere_scenes.py"),
                             "scene_loader": Path("examples/baselines/constrain/placesphere_transport_detour.py"),
                             "codec": Path("examples/baselines/constrain/placesphere_action_codec.py"),
                             "base_actor": Path("examples/baselines/constrain/placesphere_base_dp.py"),
                             "inference_sampler": Path("examples/baselines/constrain/placesphere_inference_sampler.py"),
                             "manifest_dataset": Path("examples/baselines/constrain/placesphere_manifest_dataset.py"),
                             "task_config": Path("examples/baselines/constrain/placesphere_pilot.py")})
        input_paths = {"checkpoint": checkpoint,
                       "base_checkpoint": Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base_verified/final.pt"),
                       "normalizer": Path("examples/baselines/constrain/placesphere_constrain.py"),
                       "scene_manifest": scenes / "manifest.json", "scene_h5": scenes / "scenes.h5"}
        input_paths.update({"training_registration": Path("outputs/exp/20261027_224727_placesphere_action_codec_r30/training_preregistration.json"),
                            "data_h5": Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.h5"),
                             "data_manifest": Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.json"),
                            "checkpoint_correction": Path("outputs/exp/20261027_224727_placesphere_action_codec_r30/checkpoint_metadata_correction.json")})
    if len(set(args.indices)) != len(args.indices) or any(i < 0 for i in args.indices):
        raise ValueError("scene indices must be unique and non-negative")
    if args.register:
        if args.scenes is not None:
            input_paths["scene_manifest"] = scenes / "manifest.json"
            input_paths["scene_h5"] = scenes / "scenes.h5"
        if args.output.exists():
            raise FileExistsError(args.output)
        freeze_registration(args.registration, args.task, args.indices, source_paths,
                            {name: {"path": str(path), "sha256": sha256(path)}
                             for name, path in input_paths.items()},
                            event_replan=args.event_replan,
                            common_latent_replan=args.common_latent_replan,
                            xy_feedback=args.xy_feedback, xy_constraint=xy_constraint,
                            progress_guard=args.bounded_progress_guard)
        return
    reg = validate_registration(args.registration)
    registered_replan = bool(reg.get("constraint_profile", {}).get("event_replan", False))
    if registered_replan != bool(args.event_replan):
        raise ValueError("event-replan runtime flag differs from frozen registration")
    registered_common = bool(reg.get("constraint_profile", {}).get("common_latent_replan", False))
    if registered_common != bool(args.common_latent_replan):
        raise ValueError("common-latent runtime flag differs from frozen registration")
    registered_feedback = bool(reg.get("constraint_profile", {}).get("xy_feedback", False))
    if registered_feedback != bool(args.xy_feedback):
        raise ValueError("XY-feedback runtime flag differs from frozen registration")
    registered_constraint = bool(reg.get("constraint_profile", {}).get("xy_constraint", True))
    if registered_constraint != xy_constraint:
        raise ValueError("XY-constraint runtime flag differs from frozen registration")
    if bool(reg.get("constraint_profile", {}).get("bounded_progress_guard", False)) != args.bounded_progress_guard:
        raise ValueError("bounded progress guard differs from frozen registration")
    if args.scenes is not None:
        validate_explicit_catalog(reg, args.task, args.indices, scenes)
    if (reg.get("task") != args.task or reg.get("indices") != args.indices
            or any(sha256(path) != item["sha256"] for name, item in reg["inputs"].items()
                   for path in [item["path"]])):
        raise ValueError("registered task/indices/input identity mismatch")
    original_make = _install_render_only_step_wrapper(
        args.task, event_replan=args.event_replan, xy_feedback=args.xy_feedback,
        xy_constraint=xy_constraint, progress_guard=args.bounded_progress_guard)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    failures = False
    started = time.monotonic()
    try:
        for index in args.indices:
            if time.monotonic() - started >= TRIAL_WALL_SECONDS:
                failures = True
                rows.append({"scene_index": index, "status": "runtime_error",
                             "error": f"trial exceeded {TRIAL_WALL_SECONDS}s wall limit"})
                _atomic_json(args.output / f"episode_{index:03d}.json", rows[-1])
                continue
            previous_handler = signal.signal(signal.SIGALRM, _episode_timeout)
            remaining = TRIAL_WALL_SECONDS - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, min(EPISODE_WALL_SECONDS, remaining))
            try:
                row = _evaluate_one(args.task, index, args.device, scenes,
                                     common_latent_replan=args.common_latent_replan,
                                     xy_feedback=args.xy_feedback, xy_constraint=xy_constraint,
                                     progress_guard=args.bounded_progress_guard)
                rows.append(row)
                failures |= row.get("status") != "evaluated"
            except Exception as exc:
                failures = True
                rows.append(_runtime_error_row(index, exc))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
            _atomic_json(args.output / f"episode_{index:03d}.json", rows[-1])
    finally:
        import gymnasium as gym
        gym.make = original_make
    # Integration recording is completed per scene in the evaluators' native traces.
    summary = summarize_rows(rows, len(args.indices))
    _atomic_json(args.output / "rows.json", rows)
    _atomic_json(args.output / "summary.json", summary)
    validate_registration(args.registration)
    print(json.dumps(summary), flush=True)
    if failures:
        raise RuntimeError(f"evaluation had runtime errors: {summary}")
    return summary


if __name__ == "__main__":
    main()
