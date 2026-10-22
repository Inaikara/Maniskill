"""Bounded PickCube direct and obstacle pilot collection and 7d data loading."""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

from examples.baselines.constrain.pickcube_expert import (
    PickCubeExpertConfig,
    make_pickcube_env,
    run_pickcube_expert,
)
from examples.baselines.constrain.layout_geometry import LayoutGeometryConfig


@dataclass(frozen=True)
class PickCubePilotConfig:
    seed: int = 42
    target_per_class: int = 20
    max_attempts_per_class: int = 200
    max_steps: int = 500
    obs_horizon: int = 2
    pred_horizon: int = 16
    max_wall_seconds: float = 1800.0
    waypoint_offset_range: tuple = (0.20, 0.25)
    obstacle_radius: float = 0.03
    obstacle_height: float = 0.25
    direct_target: int = None
    obstacle_target: int = None
    direct_max_attempts: int = None
    obstacle_max_attempts: int = None

    def target(self, kind):
        value = getattr(self, f"{kind}_target")
        return self.target_per_class if value is None else value


def _default_attempt(kind, seed, rng, config):
    """Run a real bounded expert attempt; failed or truncated tasks are rejected."""
    env = make_pickcube_env(
        obstacle_radius=config.obstacle_radius, obstacle_height=config.obstacle_height
    )
    try:
        base_xy = env.unwrapped.agent.robot.pose.p[0, :2].detach().cpu().numpy()
        # Sample a varied, reachable layout on the robot-facing side of the
        # table; validate_layout remains the geometric authority.
        geometry_tries = 0
        waypoint_tries = 0
        template_executions = 0
        obstacle_executions = 0
        cylinder_candidates = 0
        phase_timings = {"geometry": 0.0, "template": 0.0,
                         "cylinder_sampling": 0.0, "obstacle_execution": 0.0}
        geometry_started = time.monotonic()
        for _ in range(32):
            geometry_tries += 1
            start = np.array([rng.uniform(-0.18, -0.02), rng.uniform(-0.06, 0.12)])
            goal = np.array([rng.uniform(0.02, 0.18), rng.uniform(-0.06, 0.12)])
            try:
                result = run_pickcube_expert(
                    env, cube_xy=start, goal_xy=goal, detour=(kind == "obstacle"),
                    obstacle_enabled=False, config=PickCubeExpertConfig(
                        seed=seed, max_steps=config.max_steps if kind == "direct" else max(1, config.max_steps // 2)
                    ),
                    layout_config=LayoutGeometryConfig(
                        table_bounds=(-0.25, -0.25, 0.25, 0.35),
                        min_transport_distance=0.14, min_base_line_distance=0.06,
                        waypoint_offset_range=config.waypoint_offset_range,
                        waypoint_lead_fraction=0.25, max_attempts=32,
                    ),
                )
                waypoint_tries += int(result["trace"]["waypoint_sampling_attempts"])
                break
            except ValueError as exc:
                if kind == "obstacle" and "invalid_pickcube_waypoint" in str(exc):
                    waypoint_tries += 32
                if "invalid_pickcube_layout" not in str(exc) and "invalid_pickcube_waypoint" not in str(exc):
                    raise
        else:
            return {"failure": "geometry_retry_exhausted", "counters": {
                "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries}}, None
        geometry_elapsed = time.monotonic() - geometry_started
        template_elapsed = float(result["trace"]["phase_timings"]["expert_execution"])
        phase_timings["geometry"] = max(0.0, geometry_elapsed - template_elapsed)
        if kind == "obstacle":
            template_executions += 1
            phase_timings["template"] = template_elapsed
            if not result["safety"]["native_success"]:
                return {"failure": "template_native_timeout_or_unsuccessful", "counters": {
                    "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                    "template_executions": template_executions}}, None
            template_steps = len(result["actions"])
            if template_steps >= config.max_steps:
                return None, "attempt_control_step_budget_exhausted"
            template_state, template_controller = result["initial_snapshot"]
            template_observation = result["initial_observation_named"]
            # Candidate cylinders block the straight S-G object segment but
            # are kept away from the measured successful detour sweep.
            from examples.baselines.constrain.collect_constructive import segment_point_distance
            # The object-path broad phase must not use a union of every robot
            # link: that swept envelope can cover the entire direct corridor.
            # Candidate execution below remains the authority for all robot
            # links and the cube, including continuous-step crossing checks.
            cylinder_started = time.monotonic()
            sweep = np.asarray(result["trace"]["cube_xy"], dtype=np.float64)
            center = None
            side = np.sign(np.cross(goal - start, base_xy - start))
            direction = (goal - start) / np.linalg.norm(goal - start)
            lateral = np.array([-direction[1], direction[0]])
            candidates = np.asarray([
                start + t * (goal - start) - side * lateral * offset
                for t in np.linspace(0.40, 0.60, 21)
                for offset in np.linspace(0.0, 0.06, 25)
            ])
            cylinder_candidates = len(candidates)
            distances, _ = cKDTree(sweep).query(candidates, k=1, workers=1)
            blocking = np.asarray([
                segment_point_distance(candidate, start, goal) <= 0.063
                and np.linalg.norm(candidate - start) > 0.068
                and np.linalg.norm(candidate - goal) > 0.068
                and -0.25 <= candidate[0] <= 0.25
                and -0.25 <= candidate[1] <= 0.35
                for candidate in candidates
            ])
            robot_sweep = np.asarray(result["trace"]["swept_xy"], dtype=np.float64)
            robot_distances, _ = cKDTree(robot_sweep).query(candidates, k=1, workers=1)
            robot_clear = robot_distances >= config.obstacle_radius
            feasible = _safe_cylinder_candidates(
                blocking, distances, robot_distances,
                required_robot_clearance=config.obstacle_radius,
                required_cube_clearance=0.063,
            )
            if len(feasible):
                # Rank viable object-blocking locations by clearance from the
                # measured full robot sweep; final execution still decides.
                chosen = feasible[np.argmax(robot_distances[feasible])]
                center = candidates[chosen]
            if center is None:
                phase_timings["cylinder_sampling"] = time.monotonic() - cylinder_started
                return {"failure": f"cylinder_geometry_retry_exhausted:{len(candidates)}", "counters": {
                    "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                    "template_executions": template_executions,
                    "cylinder_candidates": cylinder_candidates}, "phase_timings": phase_timings,
                    "diagnostics": {
                    "candidate_count": len(candidates), "blocking_count": int(blocking.sum()),
                    "sweep_clear_candidates": int((distances >= 0.035).sum()),
                    "robot_clear_candidates": int(robot_clear.sum()),
                    "required_robot_clearance_m": config.obstacle_radius,
                    "maximum_robot_clearance_m": float(robot_distances.max())}}, None
            phase_timings["cylinder_sampling"] = time.monotonic() - cylinder_started
            # CPU observations read the live PhysX scene contact cache. Run the
            # restored execution in a fresh scene to discard template contacts.
            env.close()
            env = make_pickcube_env(
                obstacle_radius=config.obstacle_radius, obstacle_height=config.obstacle_height
            )
            env.reset(seed=seed, options={"cube_xy": start, "goal_pos": [*goal, 0.12],
                                          "obstacle_enabled": True, "obstacle_center": center})
            obstacle_executions += 1
            result = run_pickcube_expert(
                env, cube_xy=start, goal_xy=goal, detour=True, obstacle_enabled=True,
                obstacle_center=center, initial_snapshot=(template_state, template_controller),
                config=PickCubeExpertConfig(seed=seed, max_steps=config.max_steps - template_steps),
                layout_config=LayoutGeometryConfig(
                    table_bounds=(-0.25, -0.25, 0.25, 0.35),
                    min_transport_distance=0.14, min_base_line_distance=0.06,
                    waypoint_offset_range=config.waypoint_offset_range,
                    waypoint_lead_fraction=0.25, max_attempts=32,
                ),
            )
            phase_timings["obstacle_execution"] = float(
                result["trace"]["phase_timings"]["expert_execution"]
            )
            if not _same_initial_except_obstacle(template_state, result["initial_state"]):
                return {"failure": "initial_state_restore_mismatch", "counters": {
                    "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                    "template_executions": template_executions,
                    "obstacle_executions": obstacle_executions,
                    "cylinder_candidates": cylinder_candidates}, "diagnostics": {
                    "first_difference": _first_tree_difference(template_state, result["initial_state"])}}, None
            if not _tree_equal(template_controller, result["initial_snapshot"][1]):
                return {"failure": "controller_restore_mismatch", "counters": {
                    "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                    "template_executions": template_executions,
                    "obstacle_executions": obstacle_executions,
                    "cylinder_candidates": cylinder_candidates}}, None
            if not _observation_matches_except_obstacle(
                template_observation, result["initial_observation_named"]
            ):
                return {"failure": "initial_observation_restore_mismatch", "counters": {
                    "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                    "template_executions": template_executions,
                    "obstacle_executions": obstacle_executions,
                    "cylinder_candidates": cylinder_candidates}, "diagnostics": {
                    **_observation_mismatch_details(
                        template_observation, result["initial_observation_named"]
                    )}}, None
            if not result["safety"]["safe_success"]:
                if not result["safety"]["native_success"]:
                    reason = "obstacle_native_timeout_or_unsuccessful"
                else:
                    reason = "obstacle_collision_or_crossing"
                return {"failure": reason, "counters": {
                    "geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                    "template_executions": template_executions,
                    "obstacle_executions": obstacle_executions,
                    "cylinder_candidates": cylinder_candidates}, "phase_timings": phase_timings,
                    "diagnostics": {
                    "safety": result["safety"], "obstacle_center": center.tolist(),
                    "cube_clearance_m": float(distances[chosen]),
                    "robot_sweep_clearance_m": float(robot_distances[chosen]),
                    "first_obstacle_contact": result["trace"].get("first_obstacle_contact")}}, None
            result["trace"]["obstacle_center"] = center.tolist()
            result["trace"]["cylinder_candidates_tested"] = int(len(candidates))
            result["trace"]["cylinder_cube_clearance_m"] = float(distances[chosen])
            result["trace"]["cylinder_robot_sweep_clearance_m"] = float(robot_distances[chosen])
            result["trace"]["initial_state_identity"] = {
                "template_native_state_sha256": _value_hash(template_state, ignore_obstacle_pose=True),
                "obstacle_native_state_sha256": _value_hash(result["initial_state"], ignore_obstacle_pose=True),
                "template_controller_sha256": _value_hash(template_controller),
                "obstacle_controller_sha256": _value_hash(result["initial_snapshot"][1]),
                "native_observation_equal_except_named_pickcube_obstacle_field": True,
            }
        elif not result["safety"]["native_success"]:
            return {"failure": "native_timeout_or_unsuccessful", "counters": {
                "geometry_tries": geometry_tries}}, None
        result["trace"]["runtime_initial_identity"] = {
            "native_state_sha256": _value_hash(result["initial_snapshot"][0]),
            "controller_state_sha256": _value_hash(result["initial_snapshot"][1]),
            "controller_state_snapshot": _jsonable(result["initial_snapshot"][1]),
            "named_observation_fields": _observation_field_shapes(result["initial_observation_named"]),
            "named_observation_sha256": _value_hash(result["initial_observation_named"]),
        }
        result["layout"]["base_xy"] = base_xy.tolist()
        return {
            "actions": result["actions"],
            "obs": np.stack([np.asarray(x).reshape(-1) for x in result["observations"]]),
            "states": result["states"],
            "infos": result["infos"],
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "success": True,
            "info": {"native_success": True, "safe_success": True},
            "seed": seed,
            "layout": result["layout"],
            "trace": result["trace"],
            "counters": {"geometry_tries": geometry_tries, "waypoint_tries": waypoint_tries,
                         "template_executions": template_executions,
                         "obstacle_executions": obstacle_executions,
                         "cylinder_candidates": cylinder_candidates},
            "phase_timings": phase_timings,
            "info": {"native_success": result["safety"]["native_success"],
                     "safe_success": result["safety"]["safe_success"]},
        }, None
    finally:
        env.close()


def _write_value(group, key, value):
    if isinstance(value, dict):
        child = group.create_group(str(key))
        for subkey, subvalue in value.items():
            _write_value(child, subkey, subvalue)
    else:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        group.create_dataset(str(key), data=np.asarray(value))


def _same_initial_except_obstacle(expected, actual):
    if not isinstance(expected, dict) or not isinstance(actual, dict) or expected.keys() != actual.keys():
        return False
    for key in expected:
        left, right = expected[key], actual[key]
        if isinstance(left, dict):
            if not _same_initial_except_obstacle(left, right):
                return False
        elif key == "pickcube_obstacle":
            left, right = torch.as_tensor(left), torch.as_tensor(right)
            if left.shape != right.shape or left.shape[-1] != 9:
                return False
            mask = torch.ones(left.shape[-1], dtype=torch.bool, device=left.device)
            mask[[0, 1, 3]] = False  # obstacle center and enabled condition only
            if not torch.equal(left[..., mask], right[..., mask]):
                return False
        elif not _tree_equal(left, right):
            return False
    return True


def _first_tree_difference(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return f"{prefix or '/'}:key-set"
        for key in expected:
            path = f"{prefix}/{key}"
            difference = _first_tree_difference(expected[key], actual[key], path)
            if difference:
                return difference
        return None
    if not _tree_equal(expected, actual):
        return prefix or "/"
    return None


def _tree_equal(expected, actual):
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _tree_equal(expected[key], actual[key]) for key in expected
        )
    if torch.is_tensor(expected):
        return torch.equal(expected, actual)
    return np.array_equal(expected, actual)


def _value_hash(value, ignore_obstacle_pose=False):
    digest = hashlib.sha256()

    def visit(node, prefix=""):
        if isinstance(node, dict):
            for name in sorted(node):
                child = node[name]
                if ignore_obstacle_pose and name == "pickcube_obstacle":
                    child = torch.as_tensor(child).clone()
                    child[..., [0, 1, 3]] = 0
                visit(child, f"{prefix}/{name}")
        else:
            array = node.detach().cpu().numpy() if torch.is_tensor(node) else np.asarray(node)
            digest.update(prefix.encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.hexdigest()


def _observation_matches_except_obstacle(template_obs, obstacle_obs):
    if not isinstance(template_obs, dict) or not isinstance(obstacle_obs, dict):
        return False
    if not _contains_key(template_obs, "pickcube_obstacle") or not _contains_key(
        obstacle_obs, "pickcube_obstacle"
    ):
        return False
    return _observation_tree_equal(template_obs, obstacle_obs)


def _observation_mismatch_details(template_obs, obstacle_obs):
    return {"first_difference": _first_observation_difference(template_obs, obstacle_obs)}


def _safe_cylinder_candidates(blocking, cube_distances, robot_distances,
                              *, required_robot_clearance, required_cube_clearance):
    blocking = np.asarray(blocking, dtype=bool)
    cube_distances = np.asarray(cube_distances, dtype=np.float64)
    robot_distances = np.asarray(robot_distances, dtype=np.float64)
    if blocking.shape != cube_distances.shape or blocking.shape != robot_distances.shape:
        raise ValueError("candidate masks and sweep distances must have matching shapes")
    safe = (blocking & (cube_distances >= required_cube_clearance)
            & (robot_distances >= required_robot_clearance))
    return np.flatnonzero(safe)


def _contains_key(tree, name):
    return isinstance(tree, dict) and (
        name in tree or any(_contains_key(value, name) for value in tree.values())
    )


def _observation_tree_equal(expected, actual):
    if isinstance(expected, dict) and isinstance(actual, dict):
        left_keys = expected.keys() - {"pickcube_obstacle"}
        right_keys = actual.keys() - {"pickcube_obstacle"}
        return left_keys == right_keys and all(
            _observation_tree_equal(expected[key], actual[key]) for key in left_keys
        )
    if torch.is_tensor(expected) or torch.is_tensor(actual):
        left, right = torch.as_tensor(expected), torch.as_tensor(actual)
        return left.shape == right.shape and torch.equal(left, right)
    return np.array_equal(expected, actual)


def _first_observation_difference(expected, actual, path=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        left_keys = expected.keys() - {"pickcube_obstacle"}
        right_keys = actual.keys() - {"pickcube_obstacle"}
        if left_keys != right_keys:
            return {"path": path or "/", "template_keys": sorted(left_keys),
                    "restored_keys": sorted(right_keys)}
        for key in sorted(left_keys):
            difference = _first_observation_difference(expected[key], actual[key], f"{path}/{key}")
            if difference:
                return difference
        return None
    if not _observation_tree_equal(expected, actual):
        left = expected.detach().cpu().tolist() if torch.is_tensor(expected) else np.asarray(expected).tolist()
        right = actual.detach().cpu().tolist() if torch.is_tensor(actual) else np.asarray(actual).tolist()
        return {"path": path or "/", "template": left, "restored": right}
    return None


def _stack_states(states):
    if isinstance(states[0], dict):
        return {key: _stack_states([state[key] for state in states]) for key in states[0]}
    values = [value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value) for value in states]
    return np.concatenate(values, axis=0)


def _clone_state(state):
    if isinstance(state, dict):
        return {key: _clone_state(value) for key, value in state.items()}
    return state.detach().clone() if torch.is_tensor(state) else np.array(state, copy=True)


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _trace_metadata(trace):
    trace = dict(trace)
    if "swept_xy" in trace:
        trace["swept_xy_point_count"] = len(trace.pop("swept_xy"))
    return _jsonable(trace)


def _observation_field_shapes(tree, prefix=""):
    if isinstance(tree, dict):
        return {f"{prefix}/{key}": shape for key, value in tree.items()
                for shape in [_observation_field_shapes(value, f"{prefix}/{key}")]}
    shape = list(tree.shape) if hasattr(tree, "shape") else list(np.asarray(tree).shape)
    return shape


def collect_pilot(output_dir, config=PickCubePilotConfig(), attempt_runner=None):
    positive = (config.target_per_class, config.max_attempts_per_class, config.max_steps,
                config.obs_horizon, config.pred_horizon)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
           for value in positive):
        raise ValueError("pilot counts, horizons, and step budget must be positive integers")
    if not np.isfinite(config.max_wall_seconds) or config.max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive and finite")
    targets = {kind: config.target(kind) for kind in ("direct", "obstacle")}
    attempt_budgets = {kind: getattr(config, f"{kind}_max_attempts") or config.max_attempts_per_class
                       for kind in ("direct", "obstacle")}
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
           for value in (*targets.values(), *attempt_budgets.values())):
        raise ValueError("class targets and attempt budgets must be positive integers")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    h5_path = output_dir / "trajectory.h5"
    (output_dir / "collection_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    from examples.baselines.constrain.pickcube_provenance import current_source_fingerprint
    frozen_contract = {
        "generator_version": "pickcube-formal-streaming-v1",
        "source_fingerprints": current_source_fingerprint(Path(__file__).resolve().parents[3]),
        "collection": asdict(config),
        "sampling": {"start_x": [-0.18, -0.02], "start_y": [-0.06, 0.12],
                     "goal_x": [0.02, 0.18], "goal_y": [-0.06, 0.12],
                     "table_bounds_xy": [-0.25, -0.25, 0.25, 0.35],
                     "minimum_transport_distance_m": 0.14,
                     "minimum_base_line_distance_m": 0.06,
                     "waypoint_offset_range_m": list(config.waypoint_offset_range),
                     "waypoint_lead_fraction": 0.25, "waypoint_candidates": 32,
                     "same_base_side_only": True, "geometry_attempts": 32},
        "obstacle": {"shape": "vertical cylinder", "radius_m": config.obstacle_radius,
                     "height_m": config.obstacle_height, "safety_margin_m": 0.005,
                     "contact_impulse_threshold": 1e-6, "candidate_grid": [21, 25],
                     "blocking_segment_distance_m": 0.063,
                     "endpoint_exclusion_m": 0.068, "cube_clearance_m": 0.063,
                     "robot_sweep_clearance_m": config.obstacle_radius},
        "execution": {"robot": "panda", "robot_init_qpos_noise": 0.02,
                      "obs_mode": "state", "control_mode": "pd_ee_delta_pose",
                      "sim_backend": "physx_cpu", "action_dim": 7,
                      "action_scale_translation_m": 0.1, "action_cap": 0.35,
                      "rotation_scale": "signed controller.rot_lower XYZ Euler",
                      "expert": asdict(PickCubeExpertConfig()), "rollout_control_steps": config.max_steps,
                      "controller_runtime_snapshot": "saved per trajectory in metadata hashes",
                      "obs_horizon": config.obs_horizon, "prediction_horizon": config.pred_horizon,
                      "execution_horizon": 8},
        "acceptance": {"direct": "native task success", "obstacle":
                       "native success, zero robot/cube obstacle contact, zero XY forbidden crossing"},
    }
    (output_dir / "frozen_contract.json").write_text(json.dumps(frozen_contract, indent=2, sort_keys=True) + "\n")
    report = {
        "config": asdict(config), "frozen_contract_sha256": hashlib.sha256(
            (output_dir / "frozen_contract.json").read_bytes()).hexdigest(),
        "started_unix": time.time(), "classes": {},
        "complete": False, "attempts": [],
    }
    runner = attempt_runner or _default_attempt
    episodes = []
    def persist_progress():
        report["classes"] = {
            kind: {"complete_attempts": sum(a["class"] == kind for a in report["attempts"]),
                   "accepted": sum(a["class"] == kind and a.get("accepted", False)
                                   for a in report["attempts"]),
                   "target": targets[kind],
                   "attempt_budget": attempt_budgets[kind],
                   "failures": dict(Counter(a["failure"] for a in report["attempts"]
                                             if a["class"] == kind and "failure" in a)),
                   "stage_counts": {name: sum(int(a.get(name, a.get("counters", {}).get(name, 0)))
                                              for a in report["attempts"] if a["class"] == kind)
                                    for name in ("geometry_tries", "waypoint_tries",
                                                 "template_executions", "obstacle_executions",
                                                 "cylinder_candidates_tested")},
                   "phase_elapsed_seconds": {phase: sum(float(a.get("phase_timings", {}).get(phase, 0))
                                                          for a in report["attempts"] if a["class"] == kind)
                                              for phase in ("geometry", "template", "cylinder_sampling",
                                                            "obstacle_execution")}}
            for kind in ("direct", "obstacle")
        }
        report["complete"] = all(report["classes"][kind]["accepted"] == targets[kind]
                                 for kind in targets)
        report["updated_unix"] = time.time()
        temporary = output_dir / "report.json.tmp"
        snapshot = {key: value for key, value in report.items() if key != "attempts"}
        snapshot["attempt_count"] = len(report["attempts"])
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.replace(output_dir / "report.json")
    def persist_attempt(record):
        with (output_dir / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    try:
        overall_started = time.monotonic()
        with h5py.File(h5_path, "w") as handle:
          for class_index, kind in enumerate(("direct", "obstacle")):
            rng = np.random.default_rng(config.seed + class_index)
            counts = Counter()
            class_accepted = 0
            started = time.monotonic()
            for attempt_index in range(attempt_budgets[kind]):
                if time.monotonic() - overall_started >= config.max_wall_seconds:
                    report["wall_timeout"] = True
                    break
                seed = int(rng.integers(0, 2**31 - 1))
                attempt_started = time.monotonic()
                outcome, reason = runner(kind, seed, rng, config)
                diagnostics = {}
                counters = {}
                phase_timings = {}
                if isinstance(outcome, dict) and "failure" in outcome and "actions" not in outcome:
                    reason = outcome["failure"]
                    diagnostics = outcome.get("diagnostics", {})
                    counters = outcome.get("counters", {})
                    phase_timings = outcome.get("phase_timings", {})
                    trajectory = None
                else:
                    trajectory = outcome
                counts["complete_attempts"] += 1
                record = {"class": kind, "attempt": attempt_index + 1, "seed": seed,
                          "elapsed_seconds": time.monotonic() - attempt_started}
                record["phase_timings"] = phase_timings
                if trajectory is None:
                    counts[f"failure:{reason or 'unspecified'}"] += 1
                    record["failure"] = reason or "unspecified"
                    if diagnostics:
                        record["diagnostics"] = diagnostics
                    record["counters"] = counters
                    record.update(counters)
                    report["attempts"].append(record)
                    persist_attempt(record)
                    persist_progress()
                    continue
                if not trajectory.get("success") or not trajectory.get("info", {}).get("native_success"):
                    counts["failure:native_unsuccessful"] += 1
                    record["failure"] = "native_unsuccessful"
                    report["attempts"].append(record)
                    persist_attempt(record)
                    persist_progress()
                    continue
                if kind == "obstacle" and not trajectory.get("info", {}).get("safe_success"):
                    counts["failure:unsafe"] += 1
                    record["failure"] = "unsafe"
                    report["attempts"].append(record)
                    persist_attempt(record)
                    persist_progress()
                    continue
                actions = np.asarray(trajectory["actions"])
                obs = np.asarray(trajectory["obs"])
                if actions.ndim != 2 or actions.shape[1] != 7 or obs.shape[0] != len(actions) + 1:
                    raise ValueError("accepted trajectory must have 7d actions and T+1 observations")
                if len(trajectory["states"]) != len(actions) + 1:
                    raise ValueError("accepted trajectory must have T+1 full states")
                trajectory.update(kind=kind, seed=seed)
                class_accepted += 1
                counts["accepted"] += 1
                record.update(accepted=True, steps=len(actions))
                record["counters"] = trajectory.get("counters", {})
                record["phase_timings"] = trajectory.get("phase_timings", {})
                for counter, value in record["counters"].items():
                    record[counter] = value
                record["cylinder_candidates_tested"] = int(
                    trajectory.get("trace", {}).get("cylinder_candidates_tested", 0)
                )
                report["attempts"].append(record)
                persist_attempt(record)
                i = len(episodes)
                episodes.append({"episode_id": i, "kind": kind, "seed": seed,
                    "layout": _jsonable(trajectory.get("layout")),
                    "trace": _trace_metadata(trajectory.get("trace", {})),
                    "infos": _jsonable(trajectory.get("infos", [])),
                    "terminated": trajectory.get("terminated", []).tolist(),
                    "truncated": trajectory.get("truncated", []).tolist()})
                with (output_dir / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(episodes[-1]) + "\n")
                group = handle.create_group(f"traj_{i}")
                group.attrs["kind"], group.attrs["seed"] = kind, seed
                group.create_dataset("actions", data=actions)
                group.create_dataset("obs", data=obs)
                success = np.zeros(len(actions) + 1, dtype=np.uint8)
                success[-1] = 1
                group.create_dataset("success", data=success)
                group.create_dataset("terminated", data=trajectory["terminated"])
                group.create_dataset("truncated", data=trajectory["truncated"])
                states = group.create_group("env_states")
                for key, value in _stack_states(trajectory["states"]).items():
                    _write_value(states, key, value)
                handle.flush()
                persist_progress()
                if class_accepted >= targets[kind]:
                    break
            report["classes"][kind] = {
                "complete_attempts": counts["complete_attempts"],
                "accepted": class_accepted,
                "failures": {key[8:]: value for key, value in counts.items() if key.startswith("failure:")},
                "stage_counts": {
                    "geometry_failures": sum(
                        value for key, value in counts.items()
                        if key.startswith("failure:cylinder_geometry")
                    ),
                    "waypoint_failures": sum(
                        value for key, value in counts.items()
                        if key.startswith("failure:waypoint")
                    ),
                    "state_restore_failures": counts.get("failure:initial_state_restore_mismatch", 0),
                    "cylinder_candidate_evaluations": sum(
                        int(attempt.get("cylinder_candidates", attempt.get("cylinder_candidates_tested", 0)))
                        for attempt in report["attempts"] if attempt["class"] == kind
                    ),
                    "geometry_tries": sum(int(attempt.get("geometry_tries", 0))
                                           for attempt in report["attempts"] if attempt["class"] == kind),
                    "waypoint_tries": sum(int(attempt.get("waypoint_tries", 0))
                                           for attempt in report["attempts"] if attempt["class"] == kind),
                    "template_executions": sum(int(attempt.get("template_executions", 0))
                                                for attempt in report["attempts"] if attempt["class"] == kind),
                    "obstacle_executions": sum(int(attempt.get("obstacle_executions", 0))
                                                for attempt in report["attempts"] if attempt["class"] == kind),
                },
                "phase_elapsed_seconds": {
                    phase: sum(float(attempt.get("phase_timings", {}).get(phase, 0.0))
                               for attempt in report["attempts"] if attempt["class"] == kind)
                    for phase in ("geometry", "template", "cylinder_sampling", "obstacle_execution")
                },
                "elapsed_seconds": time.monotonic() - started,
            }
        report["complete"] = all(
            report["classes"].get(kind, {}).get("accepted") == targets[kind]
            for kind in ("direct", "obstacle")
        )
        if len(episodes):
            report["trajectory_sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()
        report["finished_unix"] = time.time()
        (output_dir / "trajectory.json").write_text(json.dumps({"episodes": episodes}, indent=2) + "\n")
        (output_dir / "contract.json").write_text(json.dumps({
            "robot": "panda", "robot_init_qpos_noise": 0.02,
            "control_mode": "pd_ee_delta_pose", "obs_mode": "state",
            "state_semantics": "native ManiSkill state dictionary plus pickcube_obstacle and pickcube_goal; no StackCube indexes",
            "obstacle_observation": "world XY center and collision radius in meters (raw state; no normalization)",
            "action": {"dimension": 7, "position": "first 3 normalized by 0.1m",
                       "rotation": "XYZ Euler delta divided componentwise by controller.rot_lower, including its signed negative values",
                       "gripper": "index 6"},
            "horizons": {"obs": config.obs_horizon, "prediction": config.pred_horizon,
                         "execution": 8},
            "rollout_control_step_cap_per_complete_attempt": config.max_steps,
            "obstacle_geometry": {"cylinder_radius_m": config.obstacle_radius,
                                  "cylinder_height_m": config.obstacle_height},
            "layout_geometry": {"waypoint_offset_range_m": list(config.waypoint_offset_range),
                                "same_base_side_only": True,
                                "minimum_transport_distance_m": 0.14,
                                "minimum_base_line_distance_m": 0.06},
        }, indent=2) + "\n")
        persist_progress()
        report["artifact_sha256"] = {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("trajectory.h5", "trajectory.json", "collection_config.json", "contract.json")
            if (output_dir / name).exists()
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output_dir / "artifact_hashes.json").write_text(json.dumps({
            **report["artifact_sha256"],
            "report.json": hashlib.sha256((output_dir / "report.json").read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        return report
    except BaseException:
        snapshot = {key: value for key, value in report.items() if key != "attempts"}
        snapshot.update(attempt_count=len(report["attempts"]), interrupted=True)
        (output_dir / "report.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        raise


class PickCubePilotDataset(Dataset):
    """Task-specific native state-action windows; preserves all 7 action values."""

    def __init__(self, path, obs_horizon=2, pred_horizon=16):
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.observations, self.actions = [], []
        with h5py.File(path, "r") as handle:
            self.keys = sorted(handle.keys(), key=lambda key: int(key.rsplit("_", 1)[1]))
            self.slices = []
            for index, key in enumerate(self.keys):
                group = handle[key]
                obs, actions = group["obs"][:], group["actions"][:]
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"{key}: expected native seven-dimensional actions")
                if obs.shape[0] != actions.shape[0] + 1:
                    raise ValueError(f"{key}: expected T+1 observations")
                if "success" not in group or not np.all(group["success"][:]):
                    if "success" not in group or not bool(group["success"][-1]):
                        raise ValueError(f"{key}: missing successful terminal evidence")
                for flag in ("terminated", "truncated"):
                    if flag not in group or len(group[flag]) != len(actions) + 1:
                        raise ValueError(f"{key}: missing T+1 {flag} evidence")
                if bool(group["truncated"][:].any()) or not bool(group["terminated"][-1]):
                    raise ValueError(f"{key}: successful terminal cannot be a timeout")
                def check_state_lengths(node):
                    if isinstance(node, h5py.Dataset):
                        if len(node) != len(actions) + 1:
                            raise ValueError(f"{key}: full state must have T+1 rows")
                    else:
                        for child in node.values():
                            check_state_lengths(child)
                if "env_states" not in group:
                    raise ValueError(f"{key}: missing full states")
                check_state_lengths(group["env_states"])
                self.observations.append(torch.as_tensor(obs, dtype=torch.float32))
                self.actions.append(torch.as_tensor(actions, dtype=torch.float32))
                L = len(actions)
                self.slices.extend((index, start) for start in range(-(obs_horizon-1), L))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start = self.slices[index]
        obs, act = self.observations[trajectory], self.actions[trajectory]
        obs_indices = torch.arange(start, start + self.obs_horizon).clamp(0, len(obs) - 1)
        act_indices = torch.arange(start, start + self.pred_horizon).clamp(0, len(act) - 1)
        obs_seq = obs[obs_indices]
        act_seq = act[act_indices]
        return {"observations": obs_seq, "actions": act_seq}


def validate_minibatch(dataset_path, validation_dir, *, expected_obs_dim, device="cpu"):
    """Run one saved batch through a dimension-matched official epsilon model."""
    from torch.utils.data import DataLoader
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from examples.baselines.constrain.constrain_denoiser import _official_conditional_unet1d

    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=False)
    dataset = PickCubePilotDataset(dataset_path)
    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)))))
    obs, actions = batch["observations"], batch["actions"]
    if obs.shape[-1] != expected_obs_dim or actions.shape[-1] != 7:
        raise ValueError("saved minibatch does not match initialized PickCube env/action dimensions")
    network = _official_conditional_unet1d()(
        input_dim=7, global_cond_dim=obs.shape[1] * obs.shape[2],
        diffusion_step_embed_dim=64, down_dims=[64, 128, 256], n_groups=8,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon",
    )
    obs, actions = obs.to(device), actions.to(device)
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, 100, (len(actions),), device=device).long()
    noisy = scheduler.add_noise(actions, noise, timesteps)
    prediction = network(noisy, timesteps, global_cond=obs.flatten(start_dim=1))
    loss = torch.nn.functional.mse_loss(prediction, noise)
    evidence = {"batch_size": len(actions), "obs_shape": list(obs.shape),
                "action_shape": list(actions.shape), "loss": float(loss.detach().cpu()),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "model": "official ConditionalUnet1D, epsilon objective",
                "device": str(device)}
    if not evidence["loss_finite"]:
        raise FloatingPointError("saved minibatch epsilon loss is non-finite")
    (validation_dir / "minibatch_validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-dir")
    parser.add_argument("--diagnostics-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-per-class", type=int, default=20)
    parser.add_argument("--direct-target", type=int)
    parser.add_argument("--obstacle-target", type=int)
    parser.add_argument("--max-attempts-per-class", type=int, default=200)
    parser.add_argument("--direct-max-attempts", type=int)
    parser.add_argument("--obstacle-max-attempts", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--video-dir")
    args = parser.parse_args(argv)
    pilot_config = PickCubePilotConfig(
        seed=args.seed, target_per_class=args.target_per_class,
        direct_target=args.direct_target, obstacle_target=args.obstacle_target,
        max_attempts_per_class=args.max_attempts_per_class,
        direct_max_attempts=args.direct_max_attempts,
        obstacle_max_attempts=args.obstacle_max_attempts,
        max_wall_seconds=args.max_wall_seconds)
    report = collect_pilot(args.output_dir, pilot_config)
    if args.diagnostics_dir:
        diagnostics = Path(args.diagnostics_dir)
        diagnostics.mkdir(parents=True, exist_ok=False)
        (diagnostics / "collection_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if not report["complete"]:
        return
    if not any((args.validation_dir, args.video_dir)):
        return
    env = make_pickcube_env(obstacle_radius=pilot_config.obstacle_radius,
                            obstacle_height=pilot_config.obstacle_height)
    try:
        obs = env.reset(seed=args.seed)[0]
        expected_obs_dim = int(np.asarray(obs).reshape(-1).shape[0])
    finally:
        env.close()
    if args.validation_dir:
        validate_minibatch(Path(args.output_dir) / "trajectory.h5", args.validation_dir,
                           expected_obs_dim=expected_obs_dim)
    if args.video_dir:
        render_representative_videos(Path(args.output_dir) / "trajectory.h5", args.video_dir,
                                      obstacle_radius=pilot_config.obstacle_radius,
                                      obstacle_height=pilot_config.obstacle_height)
    if args.diagnostics_dir and args.validation_dir and args.video_dir:
        _write_pilot_evidence(args.output_dir, args.validation_dir, args.video_dir,
                              Path(args.diagnostics_dir) / "pilot_evidence.json")


def _write_pilot_evidence(dataset_dir, validation_dir, video_dir, destination):
    paths = {
        "trajectory.h5": Path(dataset_dir) / "trajectory.h5",
        "artifact_hashes.json": Path(dataset_dir) / "artifact_hashes.json",
        "minibatch_validation.json": Path(validation_dir) / "minibatch_validation.json",
        "direct.mp4": Path(video_dir) / "direct.mp4",
        "obstacle.mp4": Path(video_dir) / "obstacle.mp4",
    }
    evidence = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in paths.items()}
    destination = Path(destination)
    destination.write_text(json.dumps(evidence, indent=2) + "\n")


def render_representative_videos(dataset_path, output_dir, *, obstacle_radius=0.03,
                                 obstacle_height=0.25):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    env = make_pickcube_env(render_mode="rgb_array", obstacle_radius=obstacle_radius,
                            obstacle_height=obstacle_height)
    try:
        env.reset(seed=0)
        with h5py.File(dataset_path, "r") as handle:
            selected = {}
            for key in sorted(handle.keys(), key=lambda name: int(name.rsplit("_", 1)[1])):
                group = handle[key]
                kind = str(group.attrs["kind"])
                if kind not in selected:
                    selected[kind] = key
                if len(selected) == 2:
                    break
            if set(selected) != {"direct", "obstacle"}:
                raise ValueError("representative videos require accepted direct and obstacle trajectories")
            for kind, key in selected.items():
                destination = output_dir / f"{kind}.mp4"
                with imageio.get_writer(str(destination), fps=env.unwrapped.control_freq,
                                        codec="libx264", macro_block_size=1) as writer:
                    for state in range(len(handle[f"{key}/actions"]) + 1):
                        env.unwrapped.set_state_dict({
                            name: _index_state(value, state)
                            for name, value in handle[f"{key}/env_states"].items()
                        })
                        frame = env.render()
                        if torch.is_tensor(frame):
                            frame = frame.detach().cpu().numpy()
                        frame = np.asarray(frame, dtype=np.uint8)
                        if frame.ndim == 4:
                            frame = frame[0]
                        writer.append_data(frame)
    finally:
        env.close()


def _index_state(node, index):
    if isinstance(node, h5py.Group):
        return {key: _index_state(value, index) for key, value in node.items()}
    return node[index]


if __name__ == "__main__":
    main()
