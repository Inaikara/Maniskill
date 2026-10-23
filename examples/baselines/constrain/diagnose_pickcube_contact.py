"""Repeatable PickCube contact diagnosis over retained paired evaluations."""

import argparse
import hashlib
import json
import signal
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

DEFAULT_SAMPLE_TARGET = 28
LIFT_Z_M = 0.04
NEAR_GOAL_XY_M = 0.10
LOCATION_BAND_M = 0.01
MANDATORY_ALL = ("base_safe_success", "both_task_failure")
LIMITED_QUOTAS = (
    ("base_only_task_failure", 4),
    ("constrain_only_task_failure", 4),
    ("base_contact_constrain_safe", 6),
)
FILL_STRATA = ("both_contact_task_success", "remainder")
STRATA = MANDATORY_ALL + tuple(name for name, _ in LIMITED_QUOTAS) + FILL_STRATA
POLICIES = ("base_dp", "constrain_dp")


class EpisodeDeadline(TimeoutError):
    pass


def _deadline(_signum, _frame):
    raise EpisodeDeadline("episode wall limit")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _episode(eval_dir, policy, index):
    return json.loads((Path(eval_dir) / policy / f"episode_{index:03d}.json").read_text())


def _policy_counts(rows):
    return {
        "requested": len(rows),
        "obstacle_contact_count": sum(row["obstacle_contact"] for row in rows),
        "xy_crossing_count": sum(row["xy_forbidden_crossing"] for row in rows),
        "task_success_count": sum(row["native_success"] for row in rows),
        "safe_success_count": sum(row["safe_success"] for row in rows),
        "historical_termination_counts": dict(Counter(row["termination_reason"] for row in rows)),
    }


def _paired(left, right, field):
    pairs = [(bool(a[field]), bool(b[field])) for a, b in zip(left, right)]
    return {
        "both": sum(a and b for a, b in pairs),
        "base_only": sum(a and not b for a, b in pairs),
        "constrain_only": sum(b and not a for a, b in pairs),
        "neither": sum(not a and not b for a, b in pairs),
    }


def audit_retained(eval_dir, scene_dir):
    """Read retained paired records. Do not treat outcome phase as contact phase."""
    eval_dir = Path(eval_dir)
    scene_dir = Path(scene_dir)
    frozen = json.loads((eval_dir / "frozen_config.json").read_text())
    correction = json.loads((eval_dir / "termination_correction.json").read_text())
    base_rows = [_episode(eval_dir, "base_dp", index) for index in range(100)]
    constrain_rows = [_episode(eval_dir, "constrain_dp", index) for index in range(100)]
    if any("first_contact" in row or "first_obstacle_contact" in row for row in base_rows + constrain_rows):
        raise ValueError("retained records unexpectedly contain contact detail; audit must be revisited")
    base_results = [row["result"] for row in base_rows]
    constrain_results = [row["result"] for row in constrain_rows]
    corrected = correction["corrected_policy_summaries"]
    full = {
        "requested": 100,
        "base_dp": {
            **_policy_counts(base_results),
            "corrected_control_step_limit_count": corrected["base_dp"]["control_step_limit_count"],
            "corrected_environment_terminal_count": corrected["base_dp"]["environment_terminal_count"],
            "corrected_timeout_count": corrected["base_dp"]["timeout_count"],
        },
        "constrain_dp": {
            **_policy_counts(constrain_results),
            "corrected_control_step_limit_count": corrected["constrain_dp"]["control_step_limit_count"],
            "corrected_environment_terminal_count": corrected["constrain_dp"]["environment_terminal_count"],
            "corrected_timeout_count": corrected["constrain_dp"]["timeout_count"],
        },
        "paired_task_success": _paired(base_results, constrain_results, "native_success"),
        "paired_safe_success": _paired(base_results, constrain_results, "safe_success"),
        "authoritative_termination": "termination_correction.json; episode JSON environment_terminal labels are historical and were not re-observed native flags",
    }
    from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet

    catalog = PickCubeSceneSet(scene_dir)
    checkpoint_sha = {name: _sha(frozen["checkpoint_paths"][name]) for name in POLICIES}
    if checkpoint_sha != frozen["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes no longer match frozen paired-evaluation identity")
    identities = {
        "eval_dir": str(eval_dir),
        "scene_dir": str(scene_dir),
        "checkpoint_paths": frozen["checkpoint_paths"],
        "checkpoint_sha256": checkpoint_sha,
        "scene_manifest_file_sha256": _sha(scene_dir / "manifest.json"),
        "scene_manifest_embedded_sha256": catalog.manifest["manifest_sha256"],
        "scene_hdf5_sha256": _sha(catalog.path),
        "frozen_config_sha256": _sha(eval_dir / "frozen_config.json"),
        "paired_summary_sha256": _sha(eval_dir / "paired_summary.json"),
        "termination_correction_sha256": _sha(eval_dir / "termination_correction.json"),
        "frozen_max_steps": frozen["max_steps"],
        "frozen_device": frozen["device"],
        "frozen_python": frozen["python"],
        "frozen_torch": frozen["torch"],
    }
    rows = []
    for index, (base, constrain) in enumerate(zip(base_rows, constrain_rows)):
        if base["scene_index"] != index or constrain["scene_index"] != index:
            raise ValueError(f"episode index mismatch at {index}")
        if base["scene_id"] != constrain["scene_id"]:
            raise ValueError(f"paired scene id mismatch at {index}")
        rows.append({
            "scene_index": index,
            "scene_id": base["scene_id"],
            "base": base["result"],
            "constrain": constrain["result"],
        })
    return {
        "full_set": full,
        "identities": identities,
        "retained_contact_detail": {
            "has_contact_step": False,
            "has_contact_entity": False,
            "has_contact_position": False,
            "has_contact_phase": False,
            "fields_absent": ["first_contact", "first_obstacle_contact"],
        },
        "retained_records_sufficient_for_mechanism": False,
        "outcome_phase_is_contact_phase": False,
        "zero_xy_crossing_is_robot_safety_proof": False,
        "rows": rows,
    }


def _stratum(row):
    base, constrain = row["base"], row["constrain"]
    if base["safe_success"]:
        return "base_safe_success"
    if not base["native_success"] and not constrain["native_success"]:
        return "both_task_failure"
    if not base["native_success"] and constrain["native_success"]:
        return "base_only_task_failure"
    if base["native_success"] and not constrain["native_success"]:
        return "constrain_only_task_failure"
    if (base["obstacle_contact"] and constrain["safe_success"]
            and base["native_success"] and constrain["native_success"]):
        return "base_contact_constrain_safe"
    if (base["obstacle_contact"] and constrain["obstacle_contact"]
            and base["native_success"] and constrain["native_success"]):
        return "both_contact_task_success"
    return "remainder"


def select_representative(rows, *, target=DEFAULT_SAMPLE_TARGET):
    """Deterministic 20–30 scene sample. Sample counts are not full-set rates."""
    if not 20 <= int(target) <= 30:
        raise ValueError("representative sample target must be 20-30")
    target = int(target)
    grouped = {name: [] for name in STRATA}
    for row in sorted(rows, key=lambda item: item["scene_index"]):
        grouped[_stratum(row)].append(row)
    selected = []
    seen = set()
    exceeded = False

    def take(name, limit):
        nonlocal exceeded
        taken = 0
        for row in grouped[name]:
            if row["scene_index"] in seen:
                continue
            if limit is not None and taken >= limit:
                break
            if name not in MANDATORY_ALL and len(selected) >= target:
                break
            selected.append({"scene_index": row["scene_index"], "stratum": name, "row": row})
            seen.add(row["scene_index"])
            taken += 1
            if len(selected) > target:
                exceeded = True

    for name in MANDATORY_ALL:
        take(name, None)
    for name, limit in LIMITED_QUOTAS:
        take(name, limit)
    for name in FILL_STRATA:
        take(name, None)
    return {
        "target": target,
        "indexes": [item["scene_index"] for item in selected],
        "assignments": [{"scene_index": item["scene_index"], "stratum": item["stratum"]}
                        for item in selected],
        "sample_rates_are_not_full_set_rates": True,
        "exceeded_target_for_mandatory_strata": exceeded,
        "full_set_count": len(rows),
        "sample_count": len(selected),
        "full_set_counts": {name: len(grouped[name]) for name in STRATA},
        "sample_counts": dict(Counter(item["stratum"] for item in selected)),
        "sample_contact_count": sum(bool(item["row"]["base"]["obstacle_contact"]) for item in selected),
        "rule": (
            "lowest scene index within stratum; all base safe successes and all both-policy "
            "task failures; up to 4 base-only failures, 4 constrain-only failures, and 6 "
            "base-contact/constrain-safe successes; fill to target from both-contact successes "
            "then remainder. Sample rates are not full-set rates."
        ),
    }


def diagnostic_lifted(cube_z, initial_cube_z):
    return float(cube_z) > float(initial_cube_z) + LIFT_Z_M


def diagnostic_near_goal(cube_xy, goal_xy):
    return float(np.linalg.norm(np.asarray(cube_xy, dtype=np.float64)
                                - np.asarray(goal_xy, dtype=np.float64))) < NEAR_GOAL_XY_M


def classify_contact_phase(*, grasped, lifted, near_goal, native_success):
    """Contact-time phase. Thresholds match evaluator diagnostic bins, not native placement."""
    if native_success or (grasped and near_goal):
        return "goal_stabilization"
    if lifted:
        return "transport"
    if grasped:
        return "lift"
    return "grasp_approach"


def geometry_xy_clearance(point_xy, center_xy, radius):
    distance = float(np.linalg.norm(np.asarray(point_xy, dtype=np.float64)
                                    - np.asarray(center_xy, dtype=np.float64)))
    radius = float(radius)
    return {
        "method": "geometry_approximation_xy_distance_minus_radius",
        "is_surface_clearance": False,
        "center_distance_m": distance,
        "radius_m": radius,
        "clearance_m": distance - radius,
    }


def classify_contact_location(position, *, center_xy, radius, height, band_m=LOCATION_BAND_M):
    point = np.asarray(position, dtype=np.float64).reshape(-1)
    radial = float(np.linalg.norm(point[:2] - np.asarray(center_xy, dtype=np.float64).reshape(2)))
    if float(point[2]) >= float(height) - float(band_m) and radial <= float(radius) + float(band_m):
        return "top_cap"
    if abs(radial - float(radius)) <= float(band_m) and float(point[2]) <= float(height) + float(band_m):
        return "side"
    return "other"


def link_group(name):
    if name == "cube":
        return "cube"
    if "finger" in name or name.endswith("hand"):
        return "gripper"
    if "link" in name:
        return "arm"
    return "other"


def _obstacle_contact_name(base, other):
    if other is base.cube._bodies[0].entity:
        return "cube"
    matches = [link.name for link in base.agent.robot.links if other is link._bodies[0].entity]
    return matches[0] if matches else None


def _obstacle_contact_points(env, *, require_impulse):
    """Read PhysX contacts. Does not write obstacle flags or first-contact state."""
    base = env.unwrapped
    if not bool(base._obstacle_enabled[0]):
        return []
    obstacle_entity = base.obstacle._objs[0]
    hits = []
    for contact in base.scene.get_contacts():
        bodies = [body.entity for body in contact.bodies]
        if obstacle_entity not in bodies:
            continue
        other = bodies[1] if bodies[0] is obstacle_entity else bodies[0]
        name = _obstacle_contact_name(base, other)
        if name is None:
            continue
        for point in contact.points:
            impulse = float(np.linalg.norm(np.asarray(point.impulse)))
            if require_impulse and impulse <= float(base.contact_impulse_threshold):
                continue
            position = np.asarray(point.position, dtype=np.float64).reshape(-1)[:3]
            separation = float(np.asarray(point.separation).reshape(-1)[0])
            cleared = not require_impulse
            hits.append({
                "entity": name,
                "link_group": link_group(name),
                "position": position.tolist(),
                "impulse": impulse,
                "separation_m": separation,
                "surface": {
                    "method": ("post_step_physx_contact_point_after_impulse_clear" if cleared
                               else "physx_contact_point_separation"),
                    "is_surface_clearance": not cleared,
                    "separation_m": separation,
                },
            })
    return hits


def observe_obstacle_contacts(env):
    return _obstacle_contact_points(env, require_impulse=True)


def load_expert_actions(scene_set, index):
    row = scene_set.manifest["scenes"][index]
    with h5py.File(scene_set.path, "r") as handle:
        return np.asarray(handle[row["scene_id"]]["expert_actions"])


def _outcome_phase(native_success, grasped, lifted, transported):
    if native_success:
        return "task_success"
    if not grasped:
        return "no_grasp"
    if not lifted:
        return "grasp_no_lift"
    if not transported:
        return "lift_no_transport"
    return "transport_or_place_failed"


def _nearby_links(base, center, radius, required_name):
    links = []
    for link in base.agent.robot.links:
        xyz = link.pose.p[0].detach().cpu().numpy()
        distance = float(np.linalg.norm(xyz[:2] - center))
        if distance <= radius + 0.15 or link.name == required_name:
            clearance = geometry_xy_clearance(xyz[:2], center, radius)
            links.append({
                "name": link.name,
                "link_group": link_group(link.name),
                "xyz": xyz.tolist(),
                "geometry_xy_clearance_m": clearance["clearance_m"],
                "clearance_method": clearance["method"],
                "is_surface_clearance": False,
                "relation_window_m": 0.15,
            })
    return links


def _vertex_clearance(base, center, radius, height):
    from examples.baselines.constrain.pickcube_expert import _collision_vertices_xy

    points = np.asarray(_collision_vertices_xy(base, max_height=height), dtype=np.float64)
    if len(points) == 0:
        return {"method": "collision_vertex_xy_projection_distance_minus_radius",
                "is_surface_clearance": False, "min_clearance_m": None, "point_count": 0}
    distance = float(np.linalg.norm(points - np.asarray(center, dtype=np.float64), axis=1).min())
    return {
        "method": "collision_vertex_xy_projection_distance_minus_radius",
        "is_surface_clearance": False,
        "min_clearance_m": distance - float(radius),
        "point_count": int(len(points)),
    }


def replay_saved_actions(scene_set, index, actions, *, wall_seconds=120):
    """Restore the frozen scene and step saved actions. Scoring comes only from env info."""
    from examples.baselines.constrain.evaluate_pickcube_paired import (
        _bool, _native_env, classify_termination,
    )

    scene = scene_set.load(index)
    env = _native_env(scene)
    previous = signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, wall_seconds)
    try:
        scene_set.restore(index, env)
        base = env.unwrapped
        initial_cube_z = float(base.cube.pose.p[0, 2].detach().cpu())
        goal_xy = base.goal_site.pose.p[0, :2].detach().cpu().numpy()
        center = base._obstacle_center[0].detach().cpu().numpy().astype(np.float64)
        radius = float(base.obstacle_collision_radius)
        height = float(base.obstacle_height)
        elapsed = 0
        collision = crossing = grasped_ever = lifted_ever = transported = False
        native_success = terminated = truncated = False
        termination = None
        first_contact = None
        cube_path, tcp_path = [], []
        min_cube = min_tcp = None
        for action in actions:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            _, _, step_terminated, step_truncated, info = env.step(action[None])
            elapsed += 1
            terminated = bool(terminated or step_terminated)
            truncated = bool(truncated or step_truncated)
            step_success = _bool(info, "native_success")
            native_success = bool(native_success or step_success)
            collision = bool(collision or _bool(info, "obstacle_collision"))
            crossing = bool(crossing or _bool(info, "obstacle_crossing"))
            grasped_now = _bool(info, "is_grasped")
            grasped_ever = bool(grasped_ever or grasped_now)
            cube = base.cube.pose.p[0].detach().cpu().numpy()
            tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
            cube_path.append(cube[:2].astype(float).tolist())
            tcp_path.append(tcp[:2].astype(float).tolist())
            lifted_now = diagnostic_lifted(cube[2], initial_cube_z)
            near_goal = diagnostic_near_goal(cube[:2], goal_xy)
            lifted_ever = bool(lifted_ever or lifted_now)
            transported = bool(transported or near_goal)
            cube_clear = geometry_xy_clearance(cube[:2], center, radius)
            tcp_clear = geometry_xy_clearance(tcp[:2], center, radius)
            min_cube = cube_clear["clearance_m"] if min_cube is None else min(min_cube, cube_clear["clearance_m"])
            min_tcp = tcp_clear["clearance_m"] if min_tcp is None else min(min_tcp, tcp_clear["clearance_m"])
            if first_contact is None and collision:
                hits = observe_obstacle_contacts(env)
                env_first = getattr(base, "_first_obstacle_contact", None)
                position_source = "post_step_physx_contact_with_impulse"
                impulse_source = "post_step_physx_impulse"
                if not hits and env_first:
                    residual = [hit for hit in _obstacle_contact_points(env, require_impulse=False)
                                if hit["entity"] == env_first.get("other")]
                    if residual:
                        hits = residual
                        position_source = "post_step_contact_cache_impulse_cleared"
                        impulse_source = "env_in_step_max_impulse"
                if not hits:
                    raise RuntimeError(
                        f"scene {index} step {elapsed} flagged obstacle contact without a readable PhysX contact"
                    )
                primary = (min(hits, key=lambda hit: hit["separation_m"])
                           if position_source == "post_step_contact_cache_impulse_cleared"
                           else max(hits, key=lambda hit: hit["impulse"]))
                if impulse_source == "env_in_step_max_impulse":
                    primary = {**primary, "impulse": float(env_first["max_impulse"])}
                try:
                    vertex = _vertex_clearance(base, center, radius, height)
                except Exception as exc:
                    vertex = {"method": "collision_vertex_xy_projection_distance_minus_radius",
                              "is_surface_clearance": False, "error": f"{type(exc).__name__}: {exc}"}
                relation = {
                    "cube_xyz": cube.astype(float).tolist(),
                    "tcp_xyz": tcp.astype(float).tolist(),
                    "obstacle": {"center_xy": center.astype(float).tolist(), "radius": radius, "height": height},
                    "nearby_links": _nearby_links(base, center, radius, primary["entity"]),
                    "geometry_clearance": {
                        "method": "geometry_approximation_xy_distance_minus_radius",
                        "is_surface_clearance": False,
                        "cube_m": cube_clear["clearance_m"],
                        "tcp_m": tcp_clear["clearance_m"],
                        "contact_point_m": geometry_xy_clearance(primary["position"][:2], center, radius)["clearance_m"],
                    },
                    "collision_vertex_clearance": vertex,
                }
                first_contact = {
                    "control_step": elapsed,
                    "entity": primary["entity"],
                    "link_group": primary["link_group"],
                    "entities": sorted({hit["entity"] for hit in hits}),
                    "position": primary["position"],
                    "impulse": primary["impulse"],
                    "separation_m": primary["separation_m"],
                    "surface": primary["surface"],
                    "position_source": position_source,
                    "impulse_source": impulse_source,
                    "phase": classify_contact_phase(
                        grasped=grasped_now, lifted=lifted_now, near_goal=near_goal,
                        native_success=step_success),
                    "location": classify_contact_location(
                        primary["position"], center_xy=center, radius=radius, height=height),
                    "location_band_m": LOCATION_BAND_M,
                    "env_first_entity": None if not env_first else env_first.get("other"),
                    "contacts": [{"entity": hit["entity"], "position": hit["position"],
                                  "impulse": hit["impulse"], "separation_m": hit["separation_m"]}
                                 for hit in hits],
                    "relation": relation,
                }
            reason = classify_termination(step_success, step_terminated, step_truncated, elapsed, 500)
            if reason is not None:
                termination = reason
                break
        if termination is None:
            termination = classify_termination(native_success, terminated, truncated, elapsed, 500)
        if first_contact is None:
            cube = base.cube.pose.p[0].detach().cpu().numpy()
            tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
            relation = {
                "cube_xyz": cube.astype(float).tolist(),
                "tcp_xyz": tcp.astype(float).tolist(),
                "obstacle": {"center_xy": center.astype(float).tolist(), "radius": radius, "height": height},
                "nearby_links": _nearby_links(base, center, radius, ""),
                "geometry_clearance": {
                    "method": "geometry_approximation_xy_distance_minus_radius",
                    "is_surface_clearance": False,
                    "cube_m": geometry_xy_clearance(cube[:2], center, radius)["clearance_m"],
                    "tcp_m": geometry_xy_clearance(tcp[:2], center, radius)["clearance_m"],
                },
            }
        else:
            relation = first_contact["relation"]
        return {
            "scene_index": index,
            "scene_id": scene["metadata"]["scene_id"],
            "scoring": {
                "termination_reason": termination,
                "elapsed_steps": elapsed,
                "terminated": terminated,
                "truncated": truncated,
                "native_success": native_success,
                "safe_success": bool(native_success and not collision and not crossing),
                "obstacle_contact": collision,
                "xy_forbidden_crossing": crossing,
                "ever_grasped": grasped_ever,
                "ever_lifted": lifted_ever,
                "ever_transported": transported,
                "outcome_phase": _outcome_phase(native_success, grasped_ever, lifted_ever, transported),
            },
            "scoring_source": "env.step info flags; observer does not write collision, crossing, or success",
            "first_contact": first_contact,
            "relation": relation,
            "paths": {"cube_xy": cube_path, "tcp_xy": tcp_path},
            "min_geometry_clearance_m": {
                "method": "geometry_approximation_xy_distance_minus_radius",
                "is_surface_clearance": False,
                "cube": min_cube,
                "tcp": min_tcp,
            },
            "zero_xy_crossing_is_robot_safety_proof": False,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        env.close()


def _mean_nearest(source, reference):
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    if len(source) == 0 or len(reference) == 0:
        return None
    return float(np.mean([np.linalg.norm(reference - point, axis=1).min() for point in source]))


def compare_scene(base, constrain, expert):
    expert_paths = expert.get("paths") or {}
    return {
        "zero_xy_crossing_is_robot_safety_proof": False,
        "expert_clearance": {
            "cube": {
                "method": "retained_cylinder_center_to_cube_xy_path_distance",
                "is_surface_clearance": False,
                "clearance_m": expert.get("cube_clearance_m"),
            },
            "robot_sweep": {
                "method": "retained_cylinder_center_to_collision_vertex_xy_distance",
                "is_surface_clearance": False,
                "clearance_m": expert.get("robot_sweep_clearance_m"),
            },
        },
        "path_deviation": {
            "method": "mean_nearest_xy_distance",
            "is_surface_clearance": False,
            "base_to_expert_cube_xy_mean_nearest_m": _mean_nearest(
                base["paths"]["cube_xy"], expert_paths.get("cube_xy", [])),
            "constrain_to_expert_cube_xy_mean_nearest_m": _mean_nearest(
                constrain["paths"]["cube_xy"], expert_paths.get("cube_xy", [])),
            "base_to_expert_tcp_xy_mean_nearest_m": _mean_nearest(
                base["paths"]["tcp_xy"], expert_paths.get("tcp_xy", [])),
            "constrain_to_expert_tcp_xy_mean_nearest_m": _mean_nearest(
                constrain["paths"]["tcp_xy"], expert_paths.get("tcp_xy", [])),
        },
        "base_contact_phase": None if not base.get("first_contact") else base["first_contact"].get("phase"),
        "constrain_contact_phase": None if not constrain.get("first_contact") else constrain["first_contact"].get("phase"),
        "expert_waypoint_xy": expert.get("waypoint_xy"),
    }


def _contact_counts(scenes, policy):
    entities, phases, locations = Counter(), Counter(), Counter()
    contacted = 0
    for scene in scenes:
        contact = (scene.get(policy) or {}).get("first_contact")
        if not contact:
            continue
        contacted += 1
        entities[contact["entity"]] += 1
        phases[contact["phase"]] += 1
        if contact.get("location"):
            locations[contact["location"]] += 1
    return {
        "contact_count": contacted,
        "contact_entity_counts": dict(entities),
        "contact_phase_counts": dict(phases),
        "contact_location_counts": dict(locations),
    }


def summarize_contact_evidence(scenes, audit):
    return {
        "full_set": audit.get("full_set", {}),
        "sample": {
            "scene_count": len(scenes),
            "base_dp": _contact_counts(scenes, "base"),
            "constrain_dp": _contact_counts(scenes, "constrain"),
        },
        "sample_counts_are_full_set_rates": False,
        "zero_xy_crossing_is_robot_safety_proof": False,
        "confirmed_mechanism": None,
    }


def _retained_expert(metadata):
    expert = metadata["expert"]
    return {
        "safe_success": expert["safe_success"],
        "native_success": expert["native_success"],
        "obstacle_collision": expert["obstacle_collision"],
        "obstacle_crossing": expert["obstacle_crossing"],
        "first_obstacle_contact": expert["first_obstacle_contact"],
        "cube_clearance_m": expert["cube_clearance_m"],
        "robot_sweep_clearance_m": expert["robot_sweep_clearance_m"],
        "waypoint_xy": expert["waypoint_xy"],
        "steps": expert["steps"],
        "clearance_note": "retained scalars are cylinder-center distances to recorded paths, not surface clearance",
    }


def _matches(scoring, retained):
    return all(scoring[key] == retained[key] for key in (
        "native_success", "obstacle_contact", "xy_forbidden_crossing", "safe_success", "elapsed_steps"))


def diagnose_sample(eval_dir, scene_dir, selection, *, episode_wall_seconds, max_wall_seconds, started):
    from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet

    catalog = PickCubeSceneSet(scene_dir)
    scenes, errors = [], []
    for index in selection["indexes"]:
        remaining = max_wall_seconds - (time.monotonic() - started)
        if remaining < 5:
            errors.append({"scene_index": index, "error": "overall wall budget exhausted before scene"})
            break
        assignment = next(item for item in selection["assignments"] if item["scene_index"] == index)
        print(f"replay scene {index} {assignment['stratum']}", flush=True)
        record = {"scene_index": index, "stratum": assignment["stratum"]}
        try:
            base_record = _episode(eval_dir, "base_dp", index)
            constrain_record = _episode(eval_dir, "constrain_dp", index)
            record["scene_id"] = base_record["scene_id"]
            record["base"] = replay_saved_actions(
                catalog, index, base_record["actions_native_pd_ee_delta_pose"],
                wall_seconds=episode_wall_seconds)
            record["constrain"] = replay_saved_actions(
                catalog, index, constrain_record["actions_native_pd_ee_delta_pose"],
                wall_seconds=episode_wall_seconds)
            retained = _retained_expert(catalog.load(index)["metadata"])
            expert_trace = replay_saved_actions(
                catalog, index, load_expert_actions(catalog, index),
                wall_seconds=episode_wall_seconds)
            record["expert"] = {
                "retained": retained,
                "replay": expert_trace,
                "replay_matches_retained_safety": bool(
                    expert_trace["scoring"]["native_success"] and retained["native_success"]
                    and not expert_trace["scoring"]["obstacle_contact"]
                    and retained["obstacle_collision"] is False
                    and not expert_trace["scoring"]["xy_forbidden_crossing"]
                    and retained["obstacle_crossing"] is False
                    and expert_trace["scoring"]["safe_success"] and retained["safe_success"]),
            }
            record["comparison"] = compare_scene(
                record["base"], record["constrain"],
                {**retained, "paths": expert_trace["paths"]})
            record["base_scoring_matches_retained"] = _matches(record["base"]["scoring"], base_record["result"])
            record["constrain_scoring_matches_retained"] = _matches(
                record["constrain"]["scoring"], constrain_record["result"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"scene_index": index, "error": record["error"]})
        scenes.append(record)
    return scenes, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_TARGET)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--episode-wall-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    def _on_term(signum, _frame):
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, _on_term)
    try:
        started = time.monotonic()
        audit = audit_retained(args.eval_dir, args.scene_dir)
        selection = select_representative(audit["rows"], target=args.sample_size)
        _write(args.output_dir / "retained_audit.json",
               {key: value for key, value in audit.items() if key != "rows"})
        _write(args.output_dir / "sample_selection.json", selection)
        index = {
            "status": "audit_only" if args.audit_only else "running",
            "retained_records_sufficient_for_mechanism": audit["retained_records_sufficient_for_mechanism"],
            "replay_performed": False,
            "eval_dir": str(args.eval_dir),
            "scene_dir": str(args.scene_dir),
            "sample_indexes": selection["indexes"],
            "sample_count": selection["sample_count"],
            "identities": audit["identities"],
            "zero_xy_crossing_is_robot_safety_proof": False,
            "reproduction": (
                "saved native pd_ee_delta_pose actions on PickCubeSceneSet.restore in a fresh "
                "PickCubeObstacle environment; scoring read from env.step info"
            ),
            "artifacts": ["retained_audit.json", "sample_selection.json", "index.json"],
            "diagnosis_source_sha256": _sha(__file__),
        }
        errors = []
        if not args.audit_only:
            scenes, errors = diagnose_sample(
                args.eval_dir, args.scene_dir, selection,
                episode_wall_seconds=args.episode_wall_seconds,
                max_wall_seconds=args.max_wall_seconds, started=started)
            for record in scenes:
                if "error" not in record:
                    _write(args.output_dir / "scenes" / f"scene_{record['scene_index']:03d}.json", record)
            findings = summarize_contact_evidence(scenes, audit)
            findings["replay_fidelity"] = {
                "scenes_attempted": len(scenes),
                "scene_errors": len(errors),
                "base_scoring_matches_retained": sum(
                    bool(row.get("base_scoring_matches_retained")) for row in scenes),
                "constrain_scoring_matches_retained": sum(
                    bool(row.get("constrain_scoring_matches_retained")) for row in scenes),
                "expert_replay_matches_retained_safety": sum(
                    bool((row.get("expert") or {}).get("replay_matches_retained_safety")) for row in scenes),
                "errors": errors,
            }
            _write(args.output_dir / "findings.json", findings)
            index["replay_performed"] = True
            index["artifacts"] = [
                "retained_audit.json", "sample_selection.json", "findings.json", "scenes/", "index.json"]
            index["replay_fidelity"] = findings["replay_fidelity"]
            index["status"] = "complete" if not errors else "incomplete"
        index["wall_seconds"] = round(time.monotonic() - started, 3)
        _write(args.output_dir / "index.json", index)
        if errors:
            raise SystemExit(f"diagnosis incomplete: {len(errors)} scene errors")
        return index
    finally:
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
