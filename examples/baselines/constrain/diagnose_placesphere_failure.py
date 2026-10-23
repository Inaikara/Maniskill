"""Repeatable PlaceSphere crossing and terminal diagnosis over retained paired evaluations."""

import argparse
import hashlib
import json
import os
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
MAX_STEPS = 200
MANDATORY_ALL = ("contact", "release_or_stability_failure")
LIMITED_QUOTAS = (("xy_violation", 8),)
FILL_STRATA = ("both_safe_success", "remainder")
STRATA = MANDATORY_ALL + tuple(name for name, _ in LIMITED_QUOTAS) + FILL_STRATA
POLICIES = ("base_dp", "constrain_dp")
MATCH_FIELDS = (
    "native_success", "obstacle_contact", "xy_forbidden_crossing", "safe_success",
    "elapsed_steps", "terminal_is_grasped", "terminal_on_bin", "terminal_static",
    "ever_released", "phase", "termination_reason",
)


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


def _child_pids():
    me = os.getpid()
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except OSError:
            continue
        if ppid == me:
            found.append(int(entry.name))
    return found


def _terminate_new_children(before):
    before = set(before)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in _child_pids():
            if pid in before:
                continue
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def _policy_counts(rows):
    return {
        "requested": len(rows),
        "obstacle_contact_count": sum(bool(row["obstacle_contact"]) for row in rows),
        "xy_crossing_count": sum(bool(row["xy_forbidden_crossing"]) for row in rows),
        "task_success_count": sum(bool(row["native_success"]) for row in rows),
        "safe_success_count": sum(bool(row["safe_success"]) for row in rows),
        "release_or_stability_failed_count": sum(
            row["phase"] == "release_or_stability_failed" for row in rows),
        "timeout_count": sum(row["termination_reason"] == "timeout" for row in rows),
        "phase_counts": dict(Counter(row["phase"] for row in rows)),
        "termination_counts": dict(Counter(row["termination_reason"] for row in rows)),
    }


def _paired(left, right, field):
    pairs = [(bool(a[field]), bool(b[field])) for a, b in zip(left, right)]
    return {
        "both": sum(a and b for a, b in pairs),
        "base_only": sum(a and not b for a, b in pairs),
        "constrain_only": sum(b and not a for a, b in pairs),
        "neither": sum(not a and not b for a, b in pairs),
    }


def _records_have(rows, *needles):
    for row in rows:
        text = json.dumps(row)
        if any(needle in text for needle in needles):
            return True
    return False


def audit_retained(eval_dir, scene_dir):
    """Read retained paired records. They do not contain crossing height or release timing."""
    eval_dir = Path(eval_dir)
    scene_dir = Path(scene_dir)
    frozen = json.loads((eval_dir / "frozen_config.json").read_text())
    base_rows = [_episode(eval_dir, "base_dp", index) for index in range(100)]
    constrain_rows = [_episode(eval_dir, "constrain_dp", index) for index in range(100)]
    base_results = [row["result"] for row in base_rows]
    constrain_results = [row["result"] for row in constrain_rows]
    absent_needles = (
        "crossing_height", "sphere_center_z", "first_contact", "first_obstacle_contact",
        "release_step", "terminal_pose", "terminal_velocity", "linear_velocity",
    )
    if _records_have(base_rows + constrain_rows, *absent_needles):
        raise ValueError("retained records unexpectedly contain crossing or terminal detail")
    from examples.baselines.constrain.placesphere_scenes import PlaceSphereSceneSet

    catalog = PlaceSphereSceneSet(scene_dir)
    checkpoint_sha = {name: _sha(frozen["checkpoint_paths"][name]) for name in POLICIES}
    if checkpoint_sha != frozen["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes no longer match frozen paired-evaluation identity")
    manifest_sha = _sha(scene_dir / "manifest.json")
    hdf5_sha = _sha(catalog.path)
    if manifest_sha != frozen["scene_manifest_sha256"] or hdf5_sha != frozen["scene_file_sha256"]:
        raise ValueError("scene bytes no longer match frozen paired-evaluation identity")
    identities = {
        "eval_dir": str(eval_dir),
        "scene_dir": str(scene_dir),
        "checkpoint_paths": frozen["checkpoint_paths"],
        "checkpoint_sha256": checkpoint_sha,
        "scene_manifest_file_sha256": manifest_sha,
        "scene_manifest_identity_sha256": catalog.manifest["manifest_identity_sha256"],
        "scene_hdf5_sha256": hdf5_sha,
        "frozen_config_sha256": _sha(eval_dir / "frozen_config.json"),
        "paired_summary_sha256": _sha(eval_dir / "paired_summary.json"),
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
        "full_set": {
            "requested": 100,
            "base_dp": _policy_counts(base_results),
            "constrain_dp": _policy_counts(constrain_results),
            "paired_task_success": _paired(base_results, constrain_results, "native_success"),
            "paired_safe_success": _paired(base_results, constrain_results, "safe_success"),
            "paired_obstacle_contact": _paired(base_results, constrain_results, "obstacle_contact"),
            "paired_xy_forbidden_crossing": _paired(
                base_results, constrain_results, "xy_forbidden_crossing"),
            "note": "full-set counts are retained 100-scene records, not sample estimates",
        },
        "identities": identities,
        "retained_detail": {
            "has_crossing_height": False,
            "has_contact_step": False,
            "has_release_step": False,
            "has_terminal_pose": False,
            "has_terminal_velocity": False,
            "fields_absent": list(absent_needles),
        },
        "retained_records_sufficient_for_mechanism": False,
        "contactless_crossing_preclassified_as_over_top": False,
        "diagnostic_bins_do_not_replace_native_success": True,
        "rows": rows,
    }


def _stratum(row):
    base, constrain = row["base"], row["constrain"]
    if base["obstacle_contact"] or constrain["obstacle_contact"]:
        return "contact"
    if (base.get("phase") == "release_or_stability_failed"
            or constrain.get("phase") == "release_or_stability_failed"):
        return "release_or_stability_failure"
    if base["xy_forbidden_crossing"] or constrain["xy_forbidden_crossing"]:
        return "xy_violation"
    if base["safe_success"] and constrain["safe_success"]:
        return "both_safe_success"
    return "remainder"


def select_representative(rows, *, target=DEFAULT_SAMPLE_TARGET):
    """Deterministic 20–30 scene sample. Sample counts are not full-set rates."""
    if not 20 <= int(target) <= 30:
        raise ValueError("representative sample target must be 20-30")
    target = int(target)
    grouped = {name: [] for name in STRATA}
    for row in sorted(rows, key=lambda item: item["scene_index"]):
        grouped[_stratum(row)].append(row)
    selected, seen, exceeded = [], set(), False

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
        "compares_both_policies_on_same_scene": True,
        "exceeded_target_for_mandatory_strata": exceeded,
        "full_set_count": len(rows),
        "sample_count": len(selected),
        "full_set_counts": {name: len(grouped[name]) for name in STRATA},
        "sample_counts": dict(Counter(item["stratum"] for item in selected)),
        "rule": (
            "lowest scene index within stratum; all either-policy contacts and all "
            "either-policy release/stability failures; up to 8 remaining XY violations; "
            "fill to target from both-policy safe successes then remainder. "
            "Both policies are compared on each selected scene. Sample rates are not full-set rates."
        ),
    }


def diagnostic_lifted(sphere_z, initial_sphere_z):
    return float(sphere_z) > float(initial_sphere_z) + LIFT_Z_M


def diagnostic_near_goal(sphere_xy, goal_xy):
    return float(np.linalg.norm(np.asarray(sphere_xy, dtype=np.float64)
                                - np.asarray(goal_xy, dtype=np.float64))) < NEAR_GOAL_XY_M


def classify_contact_phase(*, grasped, lifted, near_goal, released, native_success):
    """Contact-time phase. Thresholds match evaluator diagnostic bins, not native placement."""
    if native_success or released or (grasped and near_goal):
        return "release_or_stabilization"
    if lifted:
        return "transport"
    if grasped:
        return "lift"
    return "grasp_approach"


def classify_contact_location(position, *, center_xy, radius, height, band_m=LOCATION_BAND_M):
    point = np.asarray(position, dtype=np.float64).reshape(-1)
    radial = float(np.linalg.norm(point[:2] - np.asarray(center_xy, dtype=np.float64).reshape(2)))
    if float(point[2]) >= float(height) - float(band_m) and radial <= float(radius) + float(band_m):
        return "top_cap"
    if abs(radial - float(radius)) <= float(band_m) and float(point[2]) <= float(height) + float(band_m):
        return "side"
    return "other"


def _segment_nearest(start, end, center):
    start, end, center = (np.asarray(value, dtype=np.float64).reshape(2)
                          for value in (start, end, center))
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom < 1e-24:
        return float(np.linalg.norm(start - center))
    t = float(np.clip(np.dot(center - start, direction) / denom, 0.0, 1.0))
    return float(np.linalg.norm(start + t * direction - center))


def classify_crossing(*, sphere_center_z, previous_sphere_center_z, sphere_radius,
                      obstacle_height, start_xy, end_xy, center_xy, obstacle_radius,
                      forbidden_radius, contact_on_step):
    """Record measured height and contact. Absence of contact is not an over-top label."""
    zs = [float(sphere_center_z)]
    if previous_sphere_center_z is not None:
        zs.append(float(previous_sphere_center_z))
    bottom = min(zs) - float(sphere_radius)
    distance = _segment_nearest(start_xy, end_xy, center_xy)
    above = bottom >= float(obstacle_height)
    body_overlaps = distance < float(obstacle_radius) + float(sphere_radius)
    in_forbidden = distance < float(forbidden_radius)
    if contact_on_step:
        relation = "contact"
    elif above:
        relation = "geometric_above_top"
    elif body_overlaps:
        relation = "xy_overlap_without_top_clearance"
    elif in_forbidden:
        relation = "forbidden_margin_only"
    else:
        relation = "outside_forbidden_disk"
    return {
        "sphere_center_z_m": float(sphere_center_z),
        "previous_sphere_center_z_m": None if previous_sphere_center_z is None else float(previous_sphere_center_z),
        "sphere_radius_m": float(sphere_radius),
        "sphere_bottom_z_m": bottom,
        "obstacle_height_m": float(obstacle_height),
        "bottom_minus_obstacle_top_m": bottom - float(obstacle_height),
        "center_distance_m": distance,
        "obstacle_radius_m": float(obstacle_radius),
        "forbidden_radius_m": float(forbidden_radius),
        "contact_on_step": bool(contact_on_step),
        "relation": relation,
        "over_top": relation == "geometric_above_top",
        "contactless_crossing_preclassified_as_over_top": False,
        "height_source": "minimum_endpoint_center_z_minus_radius",
    }


def classify_terminal_mechanism(*, native_success, ever_released, release_on_bin,
                                terminal_on_bin, terminal_static, terminal_grasped,
                                termination_reason, elapsed_steps, max_steps):
    """Evidenced bins. They do not replace native on-bin, static, and ungrasped success."""
    result = {
        "replaces_native_success": False,
        "native_success": bool(native_success),
        "mechanism": None,
        "insufficient_time": False,
    }
    if native_success:
        return result
    timed_out = (termination_reason in ("timeout", "control_step_limit")
                 and int(elapsed_steps) >= int(max_steps))
    if terminal_grasped or not ever_released:
        result["mechanism"] = "incorrect_release"
        return result
    if release_on_bin is None:
        result["unresolved"] = True
        result["hypothesis"] = "release_location_unobserved"
        return result
    if release_on_bin and not terminal_on_bin:
        result["mechanism"] = "post_release_deviation"
        return result
    if not terminal_on_bin:
        result["mechanism"] = "incorrect_release"
        return result
    if not terminal_static:
        result["mechanism"] = "not_stabilized"
        result["insufficient_time"] = bool(timed_out)
        return result
    result["unresolved"] = True
    return result


def link_group(name):
    if name == "sphere":
        return "sphere"
    if "finger" in name or name.endswith("hand"):
        return "gripper"
    if "link" in name:
        return "arm"
    return "other"


def _obstacle_contact_name(base, other):
    if other is base.obj._bodies[0].entity:
        return "sphere"
    matches = [link.name for link in base.agent.robot.links if other is link._bodies[0].entity]
    return matches[0] if matches else None


def _obstacle_contact_points(env, *, require_impulse):
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
            hits.append({
                "entity": name,
                "link_group": link_group(name),
                "position": position.tolist(),
                "impulse": impulse,
                "separation_m": separation,
            })
    return hits


def read_obstacle_contact(env):
    """Read one obstacle contact without writing collision, crossing, or success flags."""
    base = env.unwrapped
    hits = _obstacle_contact_points(env, require_impulse=True)
    position_source = "post_step_physx_contact_with_impulse"
    impulse_source = "post_step_physx_impulse"
    surface_clearance = True
    env_first = getattr(base, "_first_obstacle_contact", None)
    if not hits and env_first:
        residual = [hit for hit in _obstacle_contact_points(env, require_impulse=False)
                    if hit["entity"] == env_first.get("other")]
        if residual:
            hits = residual
            position_source = "post_step_contact_cache_impulse_cleared"
            impulse_source = "env_in_step_max_impulse"
            surface_clearance = False
    if not hits:
        return None
    if position_source == "post_step_contact_cache_impulse_cleared":
        primary = min(hits, key=lambda hit: hit["separation_m"])
        primary = {**primary, "impulse": float(env_first["max_impulse"])}
    else:
        primary = max(hits, key=lambda hit: hit["impulse"])
    return {
        **primary,
        "position_source": position_source,
        "impulse_source": impulse_source,
        "surface": {
            "method": position_source,
            "is_surface_clearance": surface_clearance,
            "separation_m": primary["separation_m"],
        },
        "env_first_entity": None if not env_first else env_first.get("other"),
    }


def _at_end(phase, sphere_xy, start_xy, container_xy):
    near_start = diagnostic_near_goal(sphere_xy, start_xy)
    near_container = diagnostic_near_goal(sphere_xy, container_xy)
    return bool(phase in ("grasp_approach", "lift", "release_or_stabilization")
                or near_start or near_container)


def _xyz(value):
    return np.asarray(value, dtype=np.float64).reshape(-1)


def replay_saved_actions(scene_set, index, actions, *, wall_seconds=120, max_steps=MAX_STEPS):
    """Restore the frozen scene and step saved actions. Scoring comes only from env info."""
    from examples.baselines.constrain.evaluate_placesphere_paired import (
        _bool, classify_phase, classify_termination,
    )

    baseline = set(_child_pids())
    scene = scene_set.load(index)
    env = None
    previous = signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, wall_seconds)
    try:
        env, _restored = scene_set.restore(index)
        base = env.unwrapped
        meta = scene["metadata"]
        initial_z = float(base.obj.pose.p[0, 2].detach().cpu())
        goal_xy = base.bin.pose.p[0, :2].detach().cpu().numpy()
        center = base._obstacle_center[0].detach().cpu().numpy().astype(np.float64)
        radius = float(base.obstacle_collision_radius)
        height = float(base.obstacle_collision_height)
        sphere_radius = float(base.radius)
        forbidden = float(base._forbidden_radius)
        start_xy = np.asarray(meta["start_xy"], dtype=np.float64)
        container_xy = np.asarray(meta["container_xy"], dtype=np.float64)
        prev_xy = base.obj.pose.p[0, :2].detach().cpu().numpy().astype(np.float64)
        prev_z = initial_z
        elapsed = 0
        collision = crossing = grasped_ever = lifted_ever = transported = released = False
        native_success = terminated = truncated = False
        termination = None
        first_contact = None
        crossing_record = None
        release_record = None
        info = {}
        for action in actions:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            collision_before = collision
            _, _, step_terminated, step_truncated, info = env.step(action[None])
            elapsed += 1
            terminated = bool(terminated or step_terminated)
            truncated = bool(truncated or step_truncated)
            step_success = _bool(info, "native_success")
            native_success = bool(native_success or step_success)
            step_collision = _bool(info, "obstacle_collision")
            step_crossing = _bool(info, "obstacle_crossing")
            new_collision = bool(step_collision and not collision_before)
            collision = bool(collision or step_collision)
            new_crossing = bool(step_crossing and not crossing)
            crossing = bool(crossing or step_crossing)
            now_grasped = _bool(info, "is_obj_grasped")
            grasped_ever = bool(grasped_ever or now_grasped)
            released = bool(released or (grasped_ever and not now_grasped))
            sphere = base.obj.pose.p[0].detach().cpu().numpy()
            lifted_now = diagnostic_lifted(sphere[2], initial_z)
            near_goal = diagnostic_near_goal(sphere[:2], goal_xy)
            lifted_ever = bool(lifted_ever or lifted_now)
            transported = bool(transported or near_goal)
            on_bin = _bool(info, "is_obj_on_bin")
            static = _bool(info, "is_obj_static")
            phase = classify_contact_phase(
                grasped=now_grasped, lifted=lifted_now, near_goal=near_goal,
                released=bool(grasped_ever and not now_grasped), native_success=step_success)
            if first_contact is None and new_collision:
                hit = read_obstacle_contact(env)
                if hit is None:
                    raise RuntimeError(
                        f"scene {index} step {elapsed} flagged obstacle contact without a readable PhysX contact")
                first_contact = {
                    "control_step": elapsed,
                    "entity": hit["entity"],
                    "link_group": hit["link_group"],
                    "position": hit["position"],
                    "impulse": hit["impulse"],
                    "separation_m": hit["separation_m"],
                    "surface": hit["surface"],
                    "position_source": hit["position_source"],
                    "impulse_source": hit["impulse_source"],
                    "phase": phase,
                    "location": classify_contact_location(
                        hit["position"], center_xy=center, radius=radius, height=height),
                    "location_band_m": LOCATION_BAND_M,
                    "at_end": _at_end(phase, sphere[:2], start_xy, container_xy),
                    "env_first_entity": hit.get("env_first_entity"),
                }
            if crossing_record is None and new_crossing:
                fresh = _obstacle_contact_points(env, require_impulse=True)
                contact_on_step = bool(fresh) or new_collision
                if new_collision and not fresh and first_contact is None:
                    raise RuntimeError(
                        f"scene {index} step {elapsed} crossing contact flag has no readable PhysX contact")
                crossing_record = classify_crossing(
                    sphere_center_z=float(sphere[2]), previous_sphere_center_z=prev_z,
                    sphere_radius=sphere_radius, obstacle_height=height,
                    start_xy=prev_xy, end_xy=sphere[:2], center_xy=center,
                    obstacle_radius=radius, forbidden_radius=forbidden,
                    contact_on_step=contact_on_step)
                crossing_record.update({
                    "control_step": elapsed,
                    "phase": phase,
                    "at_end": _at_end(phase, sphere[:2], start_xy, container_xy),
                    "earlier_contact_sticky": bool(collision_before),
                    "sphere_xyz": sphere.astype(float).tolist(),
                    "distance_to_start_m": float(np.linalg.norm(sphere[:2] - start_xy)),
                    "distance_to_container_m": float(np.linalg.norm(sphere[:2] - container_xy)),
                })
            if release_record is None and grasped_ever and not now_grasped:
                release_record = _terminal_pose(base, info, elapsed)
                release_record["on_bin"] = on_bin
                release_record["static"] = static
                release_record["grasped"] = False
            prev_xy = sphere[:2].astype(np.float64)
            prev_z = float(sphere[2])
            reason = classify_termination(step_success, step_terminated, step_truncated, elapsed, max_steps)
            if reason is not None:
                termination = reason
                break
        if termination is None:
            termination = classify_termination(native_success, terminated, truncated, elapsed, max_steps)
        if not info:
            info = base.evaluate()
        terminal_grasped = _bool(info, "is_obj_grasped")
        terminal_on_bin = _bool(info, "is_obj_on_bin")
        terminal_static = _bool(info, "is_obj_static")
        outcome = classify_phase(grasped_ever, lifted_ever, transported, released, termination == "native_success")
        terminal = _terminal_pose(base, info, elapsed)
        terminal.update({
            "grasped": terminal_grasped,
            "on_bin": terminal_on_bin,
            "static": terminal_static,
            "release_step": None if release_record is None else release_record["control_step"],
            "release_state": release_record,
            "mechanism": classify_terminal_mechanism(
                native_success=termination == "native_success",
                ever_released=released,
                release_on_bin=None if release_record is None else release_record["on_bin"],
                terminal_on_bin=terminal_on_bin,
                terminal_static=terminal_static,
                terminal_grasped=terminal_grasped,
                termination_reason=termination,
                elapsed_steps=elapsed,
                max_steps=max_steps),
        })
        return {
            "scene_index": index,
            "scene_id": meta["scene_id"],
            "scoring": {
                "termination_reason": termination,
                "elapsed_steps": elapsed,
                "terminated": terminated,
                "truncated": truncated,
                "native_success": termination == "native_success",
                "safe_success": bool(termination == "native_success" and not collision and not crossing),
                "obstacle_contact": collision,
                "xy_forbidden_crossing": crossing,
                "ever_grasped": grasped_ever,
                "ever_lifted": lifted_ever,
                "ever_transported": transported,
                "ever_released": released,
                "terminal_is_grasped": terminal_grasped,
                "terminal_on_bin": terminal_on_bin,
                "terminal_static": terminal_static,
                "phase": outcome,
            },
            "scoring_source": "env.step info flags; observer does not write collision, crossing, or success",
            "first_contact": first_contact,
            "crossing": crossing_record,
            "terminal": terminal,
            "obstacle": {
                "center_xy": center.astype(float).tolist(),
                "radius_m": radius,
                "height_m": height,
                "sphere_radius_m": sphere_radius,
                "forbidden_radius_m": forbidden,
            },
            "contactless_crossing_preclassified_as_over_top": False,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if env is not None:
            env.close()
        _terminate_new_children(baseline)


def _terminal_pose(base, info, step):
    sphere = base.obj.pose.p[0].detach().cpu().numpy().astype(float)
    bin_xyz = base.bin.pose.p[0].detach().cpu().numpy().astype(float)
    linear = base.obj.linear_velocity[0].detach().cpu().numpy().astype(float)
    angular = base.obj.angular_velocity[0].detach().cpu().numpy().astype(float)
    return {
        "control_step": step,
        "sphere_xyz": sphere.tolist(),
        "bin_xyz": bin_xyz.tolist(),
        "relative_xyz": (sphere - bin_xyz).tolist(),
        "linear_velocity": linear.tolist(),
        "angular_velocity": angular.tolist(),
        "linear_speed_m_s": float(np.linalg.norm(linear)),
        "angular_speed_rad_s": float(np.linalg.norm(angular)),
        "native_static_lin_thresh_m_s": 1e-2,
        "native_static_ang_thresh_rad_s": 0.5,
        "native_flags_from_env_info": True,
        "info_on_bin": bool(np.asarray(info["is_obj_on_bin"]).reshape(-1)[0]) if "is_obj_on_bin" in info else None,
    }


def load_expert_actions(scene_set, index):
    row = scene_set.manifest["scenes"][index]
    with h5py.File(scene_set.path, "r") as handle:
        return np.asarray(handle[row["scene_id"]]["expert_actions"])


def _retained_expert(metadata):
    expert = metadata["expert"]
    trace = expert.get("trace") or {}
    sweep = trace.get("obstacle_sweep_clearance_m") or {}
    return {
        "safe_success": expert["safe_success"],
        "native_success": expert["native_success"],
        "obstacle_collision": expert["obstacle_collision"],
        "obstacle_crossing": expert["obstacle_crossing"],
        "sphere_clearance_m": sweep.get("sphere_center_path"),
        "robot_sweep_clearance_m": sweep.get("full_robot_vertex_paths"),
        "release_control_step": trace.get("release_control_step"),
        "post_release_control_steps": expert.get("post_release_control_steps"),
        "clearance_note": "retained scalars are cylinder-center distances to recorded paths, not surface clearance",
    }


def compare_scene(base, constrain, expert):
    def mechanism(trace):
        return ((trace.get("terminal") or {}).get("mechanism") or {}).get("mechanism")

    return {
        "contactless_crossing_preclassified_as_over_top": False,
        "expert_clearance": {
            "sphere": {
                "method": "retained_cylinder_center_to_sphere_xy_path_distance",
                "is_surface_clearance": False,
                "clearance_m": expert.get("sphere_clearance_m"),
            },
            "robot_sweep": {
                "method": "retained_cylinder_center_to_collision_vertex_xy_distance",
                "is_surface_clearance": False,
                "clearance_m": expert.get("robot_sweep_clearance_m"),
            },
        },
        "base_crossing_relation": None if not base.get("crossing") else base["crossing"].get("relation"),
        "constrain_crossing_relation": None if not constrain.get("crossing") else constrain["crossing"].get("relation"),
        "base_contact_phase": None if not base.get("first_contact") else base["first_contact"].get("phase"),
        "constrain_contact_phase": None if not constrain.get("first_contact") else constrain["first_contact"].get("phase"),
        "base_terminal_mechanism": mechanism(base),
        "constrain_terminal_mechanism": mechanism(constrain),
        "expert_release_control_step": expert.get("release_control_step"),
        "expert_post_release_control_steps": expert.get("post_release_control_steps"),
    }


def adjustment_suggestions(scenes):
    crossings, contacts, failures = [], [], []
    expert_safe = False
    for scene in scenes:
        expert = scene.get("expert") or {}
        expert_safe = expert_safe or bool(expert.get("replay_matches_retained_safety"))
        for policy in ("base", "constrain"):
            trace = scene.get(policy) or {}
            if trace.get("crossing"):
                crossings.append(trace["crossing"])
            if trace.get("first_contact"):
                contacts.append(trace["first_contact"])
            mechanism = ((trace.get("terminal") or {}).get("mechanism") or {})
            if mechanism.get("mechanism"):
                failures.append(mechanism)
    height_hit = any(item.get("over_top") for item in crossings) or any(
        item.get("location") == "top_cap" for item in contacts)
    end_hit = any(item.get("at_end") for item in list(crossings) + contacts) or any(
        item.get("phase") in ("grasp_approach", "lift", "release_or_stabilization") for item in contacts)
    terminal_hit = expert_safe and any(
        item.get("mechanism") in ("not_stabilized", "post_release_deviation") for item in failures)
    clearances = [item.get("bottom_minus_obstacle_top_m") for item in crossings if item.get("over_top")]
    if height_hit:
        height = {"status": "evidenced",
                  "statement": "at least one crossing clears the obstacle top or a contact is on the top cap; "
                               "raise the obstacle above the measured sphere bottom of those crossings. "
                               f"measured bottom-minus-top values: {clearances}"}
    else:
        height = {"status": "hypothesis",
                  "statement": "no sampled crossing has sphere bottom at or above the obstacle top and no contact "
                               "is on the top cap; a height change is unconfirmed"}
    if end_hit:
        end = {"status": "evidenced",
               "statement": "at least one contact or crossing is at grasp, lift, or release/stabilization, "
                            "or within 0.10 m of an end; end sweep clearance is implicated"}
    else:
        end = {"status": "hypothesis",
               "statement": "sampled contacts and crossings are not at the grasp or goal ends; "
                            "an end-clearance change is unconfirmed"}
    if terminal_hit:
        shared = {"status": "evidenced",
                  "statement": "a sampled failure is not stabilized or left the bin after an on-bin release, "
                               "and the expert replay stayed safe through release and settle; keep that shared "
                               "expert terminal rather than stopping at release"}
    else:
        shared = {"status": "hypothesis",
                  "statement": "sampled failures are not evidenced as a missing expert settle; "
                               "changing the shared expert terminal is unconfirmed"}
    return {
        "height": height,
        "end_clearance": end,
        "shared_expert_terminal": shared,
        "contactless_crossing_preclassified_as_over_top": False,
    }


def summarize_failure_evidence(scenes, audit):
    relations = {"base_dp": Counter(), "constrain_dp": Counter()}
    mechanisms = {"base_dp": Counter(), "constrain_dp": Counter()}
    contacts = {"base_dp": Counter(), "constrain_dp": Counter()}
    phases = {"base_dp": Counter(), "constrain_dp": Counter()}
    for scene in scenes:
        for policy, key in (("base", "base_dp"), ("constrain", "constrain_dp")):
            trace = scene.get(policy) or {}
            relation = (trace.get("crossing") or {}).get("relation")
            if relation:
                relations[key][relation] += 1
            mechanism = ((trace.get("terminal") or {}).get("mechanism") or {}).get("mechanism")
            if mechanism:
                mechanisms[key][mechanism] += 1
            entity = (trace.get("first_contact") or {}).get("entity")
            if entity:
                contacts[key][entity] += 1
            phase = (trace.get("first_contact") or {}).get("phase")
            if phase:
                phases[key][phase] += 1
    confirmed = sorted({name for counts in mechanisms.values() for name in counts})
    suggestions = adjustment_suggestions(scenes)
    hypotheses = [f"{topic}: {item['statement']}" for topic, item in suggestions.items()
                  if isinstance(item, dict) and item.get("status") == "hypothesis"]
    return {
        "full_set": audit.get("full_set", {}),
        "sample": {
            "scene_count": len(scenes),
            "crossing_relation_counts": {key: dict(value) for key, value in relations.items()},
            "terminal_mechanism_counts": {key: dict(value) for key, value in mechanisms.items()},
            "contact_entity_counts": {key: dict(value) for key, value in contacts.items()},
            "contact_phase_counts": {key: dict(value) for key, value in phases.items()},
        },
        "sample_counts_are_full_set_rates": False,
        "contactless_crossing_preclassified_as_over_top": False,
        "diagnostic_bins_do_not_replace_native_success": True,
        "confirmed_mechanisms": confirmed,
        "hypotheses": hypotheses,
        "adjustments": suggestions,
    }


def measured_conclusions(scenes, full_set=None):
    """Split measured crossing and terminal facts from unconfirmed adjustments."""
    full_set = full_set or {}
    crossings, contacts, failures = [], [], []
    for scene in scenes:
        for policy in ("base", "constrain"):
            trace = scene.get(policy) or {}
            if trace.get("crossing"):
                crossings.append({"policy": policy, "scene_index": scene.get("scene_index"), **trace["crossing"]})
            if trace.get("first_contact"):
                contacts.append({"policy": policy, "scene_index": scene.get("scene_index"), **trace["first_contact"]})
            mechanism = ((trace.get("terminal") or {}).get("mechanism") or {})
            if mechanism.get("mechanism"):
                failures.append({"policy": policy, "scene_index": scene.get("scene_index"), **mechanism,
                                 "release_on_bin": ((trace.get("terminal") or {}).get("release_state") or {}).get("on_bin")})
    over_top = [item for item in crossings if item.get("relation") == "geometric_above_top"]
    not_over_top = [item for item in crossings if item.get("relation") != "geometric_above_top"]
    mid_transport = [item for item in over_top if item.get("phase") == "transport" and not item.get("at_end")]
    approach_contacts = [item for item in contacts if item.get("phase") == "grasp_approach"]
    release_contacts = [item for item in contacts if item.get("phase") == "release_or_stabilization"]
    spinning = [item for item in failures if item.get("mechanism") == "not_stabilized"]
    wrong_release = [item for item in failures if item.get("mechanism") == "incorrect_release"]
    departed = [item for item in failures if item.get("mechanism") == "post_release_deviation"]
    clearances = [float(item["bottom_minus_obstacle_top_m"]) for item in over_top
                  if item.get("bottom_minus_obstacle_top_m") is not None]
    return {
        "contactless_crossing_preclassified_as_over_top": False,
        "diagnostic_bins_do_not_replace_native_success": True,
        "sample_counts_are_full_set_rates": False,
        "full_set_xy_crossing_count": (full_set.get("base_dp") or {}).get("xy_crossing_count"),
        "full_set_requested": (full_set.get("base_dp") or {}).get("requested"),
        "measured_crossing_count": len(crossings),
        "geometric_above_top_count": len(over_top),
        "not_over_top_count": len(not_over_top),
        "not_over_top_relations": sorted({item.get("relation") for item in not_over_top}),
        "mid_transport_over_top_count": len(mid_transport),
        "over_top_bottom_clearance_m": None if not clearances else {
            "min": min(clearances), "max": max(clearances)},
        "approach_contact_count": len(approach_contacts),
        "release_contact_count": len(release_contacts),
        "not_stabilized_count": len(spinning),
        "incorrect_release_count": len(wrong_release),
        "post_release_deviation_count": len(departed),
        "height": {
            "status": "evidenced" if over_top else "hypothesis",
            "statement": ("measured geometric-above-top crossings clear the obstacle; "
                          "margin-only and below-top contacts are not over-top"
                          if over_top else "no measured crossing cleared the obstacle top"),
        },
        "end_clearance": {
            "status": "evidenced" if approach_contacts or release_contacts else "hypothesis",
            "statement": ("grasp-approach or release contacts implicate the ends; "
                          "mid-transport over-top crossings do not"
                          if approach_contacts or release_contacts
                          else "no measured contact is at grasp approach or release"),
        },
        "shared_expert_terminal": {
            "status": "evidenced" if spinning else "hypothesis",
            "statement": ("on-bin unstabilized timeouts are the evidenced expert-terminal gap; "
                          "off-bin releases are not fixed by a longer settle"
                          if spinning else "no measured failure remained on the bin and unstabilized"),
        },
        "post_release_deviation": {
            "status": "evidenced" if departed else "hypothesis",
            "statement": ("a release on the bin later left the bin"
                          if departed else "no measured release was on the bin and later off it"),
        },
    }


def render_measured_report(scenes, audit):
    conclusions = measured_conclusions(scenes, audit.get("full_set"))
    lines = [
        "# PlaceSphere failure diagnosis",
        "",
        "Sample counts are not full-set rates.",
        "Contactless XY crossings are not preclassified as over-top.",
        "Diagnostic bins do not replace native on-bin, static, and ungrasped success.",
        "",
        f"Measured crossings: {conclusions['measured_crossing_count']}.",
        f"Geometric above top: {conclusions['geometric_above_top_count']}.",
        f"Not over-top: {conclusions['not_over_top_count']} {conclusions['not_over_top_relations']}.",
        f"Mid-transport over-top: {conclusions['mid_transport_over_top_count']}.",
        f"Post-release deviation: {conclusions['post_release_deviation_count']}.",
        f"Incorrect release: {conclusions['incorrect_release_count']}.",
        f"Not stabilized: {conclusions['not_stabilized_count']}.",
        f"Height {conclusions['height']['status']}: {conclusions['height']['statement']}",
        f"End clearance {conclusions['end_clearance']['status']}: {conclusions['end_clearance']['statement']}",
        f"Shared expert terminal {conclusions['shared_expert_terminal']['status']}: "
        f"{conclusions['shared_expert_terminal']['statement']}",
        f"Post-release deviation {conclusions['post_release_deviation']['status']}: "
        f"{conclusions['post_release_deviation']['statement']}",
        "",
    ]
    return "\n".join(lines), conclusions


def render_findings(findings):
    lines = [
        "# PlaceSphere failure diagnosis",
        "",
        "Sample counts are not full-set rates.",
        "Contactless XY crossings are not preclassified as over-top.",
        "Diagnostic bins do not replace native on-bin, static, and ungrasped success.",
        "",
    ]
    full = findings.get("full_set") or {}
    for policy in ("base_dp", "constrain_dp"):
        row = full.get(policy) or {}
        if not row:
            continue
        lines.append(
            f"Full set {policy}: contact {row.get('obstacle_contact_count')}/{row.get('requested')}, "
            f"XY {row.get('xy_crossing_count')}/{row.get('requested')}, "
            f"task {row.get('task_success_count')}/{row.get('requested')}, "
            f"safe {row.get('safe_success_count')}/{row.get('requested')}, "
            f"release/stability phase {row.get('release_or_stability_failed_count')}/{row.get('requested')}.")
    sample = findings.get("sample") or {}
    lines.append(f"Sample scenes: {sample.get('scene_count')}.")
    for policy, counts in (sample.get("crossing_relation_counts") or {}).items():
        for name, count in counts.items():
            lines.append(f"Sample {policy} crossing relation {name}: {count}.")
    for policy, counts in (sample.get("terminal_mechanism_counts") or {}).items():
        for name, count in counts.items():
            lines.append(f"Sample {policy} terminal mechanism {name}: {count}.")
    for topic, item in (findings.get("adjustments") or {}).items():
        if isinstance(item, dict) and item.get("status"):
            lines.append(f"{topic} {item['status']}: {item.get('statement')}")
    if findings.get("hypotheses"):
        lines.append("Hypotheses:")
        lines.extend(f"- {item}" for item in findings["hypotheses"])
    lines.append("")
    return "\n".join(lines)


def _matches(scoring, retained):
    return all(scoring[key] == retained[key] for key in MATCH_FIELDS)


def diagnose_sample(eval_dir, scene_dir, selection, *, episode_wall_seconds, max_wall_seconds, started):
    from examples.baselines.constrain.placesphere_scenes import PlaceSphereSceneSet

    catalog = PlaceSphereSceneSet(scene_dir)
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
                wall_seconds=min(episode_wall_seconds, remaining))
            record["constrain"] = replay_saved_actions(
                catalog, index, constrain_record["actions_native_pd_ee_delta_pose"],
                wall_seconds=min(episode_wall_seconds, remaining))
            retained = _retained_expert(catalog.load(index)["metadata"])
            expert_trace = replay_saved_actions(
                catalog, index, load_expert_actions(catalog, index),
                wall_seconds=min(episode_wall_seconds, remaining))
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
            record["comparison"] = compare_scene(record["base"], record["constrain"], retained)
            record["base_scoring_matches_retained"] = _matches(record["base"]["scoring"], base_record["result"])
            record["constrain_scoring_matches_retained"] = _matches(
                record["constrain"]["scoring"], constrain_record["result"])
            if not record["base_scoring_matches_retained"] or not record["constrain_scoring_matches_retained"]:
                raise RuntimeError(f"scene {index} replay scoring does not match retained record")
            if not record["expert"]["replay_matches_retained_safety"]:
                raise RuntimeError(f"scene {index} expert replay is not the retained safe success")
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
    parser.add_argument("--max-wall-seconds", type=float, default=900)
    parser.add_argument("--episode-wall-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    baseline = set(_child_pids())

    def _on_term(signum, _frame):
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, _on_term)
    index = {"status": "starting", "output_dir": str(args.output_dir)}
    try:
        started = time.monotonic()
        audit = audit_retained(args.eval_dir, args.scene_dir)
        selection = select_representative(audit["rows"], target=args.sample_size)
        _write(args.output_dir / "retained_audit.json",
               {key: value for key, value in audit.items() if key != "rows"})
        _write(args.output_dir / "sample_selection.json",
               {key: value for key, value in selection.items() if key != "rows"})
        index = {
            "status": "audit_only" if args.audit_only else "running",
            "retained_records_sufficient_for_mechanism": audit["retained_records_sufficient_for_mechanism"],
            "contactless_crossing_preclassified_as_over_top": False,
            "diagnostic_bins_do_not_replace_native_success": True,
            "replay_performed": False,
            "eval_dir": str(args.eval_dir),
            "scene_dir": str(args.scene_dir),
            "sample_indexes": selection["indexes"],
            "sample_count": selection["sample_count"],
            "identities": audit["identities"],
            "episode_wall_seconds": args.episode_wall_seconds,
            "max_wall_seconds": args.max_wall_seconds,
            "reproduction": (
                "saved native pd_ee_delta_pose actions on PlaceSphereSceneSet.restore in a fresh "
                "PlaceSphereObstacle environment; scoring read from env.step info"
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
            complete = [row for row in scenes if "error" not in row]
            findings = summarize_failure_evidence(complete, audit)
            measured_text, measured = render_measured_report(complete, audit)
            findings["measured"] = measured
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
            (args.output_dir / "findings.md").write_text(render_findings(findings) + "\n" + measured_text)
            index["replay_performed"] = True
            index["artifacts"] = [
                "retained_audit.json", "sample_selection.json", "findings.json", "findings.md",
                "scenes/", "index.json"]
            index["replay_fidelity"] = findings["replay_fidelity"]
            index["status"] = "complete" if not errors else "incomplete"
        index["wall_seconds"] = round(time.monotonic() - started, 3)
        _write(args.output_dir / "index.json", index)
        if errors:
            raise SystemExit(f"diagnosis incomplete: {len(errors)} scene errors")
        return index
    except SystemExit as exc:
        if isinstance(exc.code, int):
            index["status"] = "interrupted"
            index["error"] = f"signal {exc.code - 128}"
            if args.output_dir.exists():
                _write(args.output_dir / "index.json", index)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        _terminate_new_children(baseline)


if __name__ == "__main__":
    main()
