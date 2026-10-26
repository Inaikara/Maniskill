"""Opt-in PlaceSphere terminal shared by direct and obstacle experts.

Legacy retreat-while-grasped behavior stays the default of run_transport_expert.
This module does not change radius, cylinder pose, or native success thresholds.
Vertex clearance is a geometric filter, not a surface-clearance proof.
"""

import hashlib
import json
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np

RADIUS_M = 0.03
NATIVE_XY_M = 0.005
NATIVE_Z_TOL_M = 0.005
NATIVE_LIN_THRESH_M_S = 1e-2
NATIVE_ANG_THRESH_RAD_S = 0.5
ABOVE_RELEASE_MARGINS_M = (0.005, 0.015, 0.030)
SIDE_OFFSET_HEADINGS_RAD = (0.0, 0.6, -0.6)
SIDE_OFFSET_M = 0.05
PERTURB_DISTANCES_M = (0.01, 0.02)
ACTION_CAP = 0.35
POS_SCALE_M = 0.1
MAX_COMMAND_M = ACTION_CAP * POS_SCALE_M
PRE_RELEASE_XY_M = 0.003
RETREAT_DZ_M = 0.08
STABILIZE_BUDGET = 16
SETTLE_BUDGET = 48
HEIGHT_HIGH_M = 0.15
HAND_LINK_NAMES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
PAIR_CATALOG = "outputs/exp/20261024_155244_placesphere_data_expansion/pair_scenes/merged"
OBSTACLE200_CATALOG = "outputs/exp/20261025_085101_placesphere_obstacle300_90k/obstacle_scenes/catalog_full"
DEV_SCENES = "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes"
IDENTITY_FILES = (
    "examples/baselines/constrain/placesphere_matched_terminal.py",
    "examples/baselines/constrain/placesphere_transport_detour.py",
    "examples/baselines/constrain/placesphere_pilot.py",
    "examples/baselines/constrain/placesphere_obstacle.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/tests/test_placesphere_matched_terminal.py",
)
ROLE_POSITIVE_DIRECT = "positive_direct"
ROLE_POSITIVE_OBSTACLE = "positive_obstacle"
ROLE_NEGATIVE_DIRECT = "negative_direct"
ROLE_TERMINAL_REPLAY = "terminal_replay"
PREVIOUS_RUNTIME_IDENTITY = "aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed"
COLLECTION_READY_IFF = (
    "every confirmation layout admits only when positive_direct is parked, native, "
    "and contact-free from execution start, positive_obstacle keeps the actual cylinder "
    "and is native-safe from execution start, both share the planned-geometry terminal "
    "targets, and the direct terminal replay at that actual cylinder is native and "
    "contact-free without treating a cleared prefix latch as full-prefix safety; "
    "negative_direct never admits"
)
TARGET_KEYS = ("lift_xyz", "transport_exit_xyz", "clearance_xyz", "release_tcp_z", "retreat_xyz")


def terminal_phase_plan(mode):
    if mode == "legacy":
        return ("container_approach", "retreat", "release", "settle")
    if mode == "above_release":
        return ("clearance_approach", "stabilize", "release", "settle", "retreat")
    if mode == "side_offset":
        return ("side_offset_approach", "side_descend", "bin_translate", "stabilize", "release", "settle", "retreat")
    raise ValueError(f"unknown terminal mode {mode}")


def clearance_tcp_z(cylinder_top_z, *, tcp_z, lowest_hand_vertex_z, margin_m):
    hand_below = max(0.0, float(tcp_z) - float(lowest_hand_vertex_z))
    return float(cylinder_top_z) + hand_below + float(margin_m)


def above_release_candidates(cylinder_top_z, *, tcp_z, lowest_hand_vertex_z, transport_exit_z,
                             margins=ABOVE_RELEASE_MARGINS_M):
    if tuple(float(value) for value in margins) != tuple(float(value) for value in ABOVE_RELEASE_MARGINS_M):
        raise ValueError("above-release margins are the predeclared triple")
    exit_z = float(transport_exit_z)
    rows = []
    for margin in margins:
        geometric = clearance_tcp_z(
            cylinder_top_z, tcp_z=tcp_z, lowest_hand_vertex_z=lowest_hand_vertex_z, margin_m=margin)
        release_z = max(geometric, exit_z)
        rows.append({
            "kind": "above_release",
            "margin_m": float(margin),
            "geometric_tcp_z": float(geometric),
            "release_tcp_z": float(release_z),
            "descends_below_transport_exit": bool(release_z < exit_z - 1e-6),
            "geometric_is_surface_clearance": False,
            "radius_m": RADIUS_M,
            "native_xy_m": NATIVE_XY_M,
            "native_z_tol_m": NATIVE_Z_TOL_M,
            "native_lin_thresh_m_s": NATIVE_LIN_THRESH_M_S,
            "native_ang_thresh_rad_s": NATIVE_ANG_THRESH_RAD_S,
            "native_thresholds_changed": False,
        })
    return rows


def retreat_after_release(*, native_success_before_retreat, retreat_steps_executed, retreat_steps_planned,
                          native_success_during_retreat=False):
    planned = int(retreat_steps_planned)
    if planned < 1:
        raise ValueError("retreat plan is missing")
    if native_success_before_retreat and int(retreat_steps_executed) != 0:
        raise ValueError("retreat continued after the first native success")
    executed = 0 if native_success_before_retreat else int(retreat_steps_executed)
    return {
        "retreat_required": True,
        "retreat_executed": executed > 0,
        "truncated_after_native_success": bool(native_success_before_retreat or native_success_during_retreat),
        "retreat_steps_executed": executed,
        "retreat_steps_planned": planned,
        "continued_after_first_native_success": False,
    }


def pose_delta_action(delta_xyz, grip, *, cap=ACTION_CAP, pos_scale=POS_SCALE_M):
    delta = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    pos = np.clip(delta / float(pos_scale), -float(cap), float(cap))
    action = np.zeros(7, dtype=np.float32)
    action[:3] = pos.astype(np.float32)
    action[6] = np.float32(grip)
    return action


def bounded_xy_actions(offset_xy, *, grip=-1.0):
    offset = np.asarray(offset_xy, dtype=np.float64).reshape(2)
    if float(np.max(np.abs(offset))) > 0.02 + 1e-9:
        raise ValueError("perturbation exceeds the predeclared 2 cm bound")
    remaining = offset.copy()
    actions = []
    while float(np.linalg.norm(remaining)) > 1e-8:
        step = remaining.copy()
        norm = float(np.linalg.norm(step))
        if norm > MAX_COMMAND_M:
            step = step / norm * MAX_COMMAND_M
        action = pose_delta_action([step[0], step[1], 0.0], grip)
        commanded = np.asarray(action[:2], dtype=np.float64) * POS_SCALE_M
        if float(np.linalg.norm(commanded - step)) > 1e-5:
            raise ValueError("perturbation action would saturate")
        actions.append(action)
        remaining -= step
        if len(actions) > 2:
            raise ValueError("perturbation needs more than the predeclared steps")
    return actions


def _passes(row):
    return bool(
        row.get("native_success") and row.get("on_bin") and row.get("static") and row.get("ungrasped")
        and not row.get("obstacle_contact") and not row.get("obstacle_crossing")
        and not row.get("closed_descent_below_clearance") and row.get("from_execution_start")
        and row.get("native_xy_m") == NATIVE_XY_M and row.get("radius_m") == RADIUS_M
        and not row.get("native_thresholds_changed"))


def select_terminal_candidate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["margin_m"]), []).append(row)
    passing = sorted(margin for margin, group in grouped.items() if group and all(_passes(row) for row in group))
    attempted = set(grouped) == set(ABOVE_RELEASE_MARGINS_M)
    selected = passing[0] if passing else None
    return {
        "selected_margin_m": selected,
        "passing_margins_m": passing,
        "attempted_margins_m": sorted(grouped),
        "side_offset_authorized": bool(selected is None and attempted),
        "collection_ready": False,
        "native_thresholds_changed": False,
        "radius_m": RADIUS_M,
    }


def side_offset_candidates(bin_xy, cylinder_xy, *, hand_xy_extent_m, radius_m, offset_m=SIDE_OFFSET_M):
    if abs(float(radius_m) - RADIUS_M) > 1e-12:
        raise ValueError("radius is fixed at 0.03 m")
    bin_xy = np.asarray(bin_xy, dtype=np.float64).reshape(2)
    cylinder_xy = np.asarray(cylinder_xy, dtype=np.float64).reshape(2)
    away = bin_xy - cylinder_xy
    norm = float(np.linalg.norm(away))
    away = np.array([1.0, 0.0]) if norm < 1e-9 else away / norm
    rows = []
    for heading in SIDE_OFFSET_HEADINGS_RAD:
        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ])
        approach = bin_xy + (rotation @ away) * float(offset_m)
        distance = float(np.linalg.norm(approach - cylinder_xy))
        clear = distance > float(radius_m) + float(hand_xy_extent_m)
        rows.append({
            "kind": "side_offset",
            "heading_rad": float(heading),
            "offset_m": float(offset_m),
            "approach_xy": approach.tolist(),
            "cylinder_distance_m": distance,
            "column_clear": bool(clear),
            "executable": bool(clear),
            "radius_m": RADIUS_M,
            "geometric_is_surface_clearance": False,
            "native_thresholds_changed": False,
            "native_xy_m": NATIVE_XY_M,
        })
    return rows


def shared_matched_terminal(direct, obstacle):
    if direct.get("waypoint_xyz") is not None:
        return False, "direct_has_waypoint"
    if obstacle.get("waypoint_xyz") is None:
        return False, "detour_missing_waypoint"
    for key in ("lift_xyz", "transport_exit_xyz", "clearance_xyz", "release_tcp_z", "retreat_xyz"):
        if not np.allclose(direct["targets"][key], obstacle["targets"][key], atol=1e-5, rtol=0):
            return False, f"{key}_mismatch"
    for record in (direct, obstacle):
        if record.get("radius_m") != RADIUS_M:
            return False, "radius_changed"
        if not (record.get("native_success") and record.get("on_bin") and record.get("static")):
            return False, "native_success_not_place"
        if not record.get("ungrasped"):
            return False, "native_success_not_released"
    return True, None


def _native_place(record):
    return bool(
        record.get("native_success") and record.get("on_bin") and record.get("static") and record.get("ungrasped")
        and record.get("release_reached") and not record.get("continued_after_first_native_success")
        and record.get("radius_m") == RADIUS_M and record.get("native_xy_m") == NATIVE_XY_M
        and record.get("native_thresholds_changed") is not True)


def _full_prefix_safe(record):
    return bool(
        record.get("from_execution_start") is True
        and record.get("from_execution_start_safe") is True
        and record.get("prefix_latch_cleared_as_proof") is not True
        and not record.get("obstacle_contact") and not record.get("obstacle_crossing")
        and not record.get("transport_contact") and not record.get("terminal_contact"))


def positive_direct_accepted(record):
    return bool(
        _native_place(record) and _full_prefix_safe(record)
        and record.get("role") == ROLE_POSITIVE_DIRECT
        and record.get("cylinder_parked") is True
        and record.get("actual_cylinder") is False
        and record.get("obstacle_relocated") is not True
        and record.get("planned_cylinder_recorded") is True
        and record.get("release_height_from_planned_geometry") is True
        and record.get("planned_cylinder_contact_proof") is False
        and record.get("path_uses_waypoint") is not True
        and record.get("action_width") == 7)


def positive_obstacle_accepted(record):
    retreat_ok = bool(record.get("truncated_after_native_success") or record.get("retreat_reached"))
    return bool(
        _native_place(record) and _full_prefix_safe(record)
        and record.get("role") == ROLE_POSITIVE_OBSTACLE
        and record.get("cylinder_parked") is not True
        and record.get("actual_cylinder") is True
        and record.get("obstacle_relocated") is not True
        and record.get("contact_checks_ran") is True
        and record.get("path_uses_waypoint") is True
        and record.get("retreat_required") and retreat_ok
        and record.get("action_width") == 7)


def matched_terminal_accepted(record):
    return positive_obstacle_accepted(record)


def direct_terminal_suffix_accepted(record):
    return positive_direct_accepted(record)


def negative_direct_admits(record):
    return False


def terminal_replay_accepted(record):
    return bool(
        record.get("role") == ROLE_TERMINAL_REPLAY
        and record.get("actual_cylinder") is True
        and record.get("cylinder_parked") is not True
        and record.get("prefix_latches_labeled") is True
        and record.get("prefix_latch_cleared_as_proof") is not True
        and record.get("full_prefix_safe") is False
        and record.get("native_success") and record.get("on_bin") and record.get("static") and record.get("ungrasped")
        and not record.get("terminal_contact") and not record.get("terminal_crossing")
        and record.get("radius_m") == RADIUS_M and record.get("native_xy_m") == NATIVE_XY_M
        and record.get("native_thresholds_changed") is not True
        and record.get("action_width") == 7
        and record.get("same_planned_geometry") is True)


def collection_admission(direct, obstacle, replay, negative=None):
    reasons = []
    if not isinstance(direct, dict) or not positive_direct_accepted(direct):
        reasons.append("positive_direct_rejected")
    if not isinstance(obstacle, dict) or not positive_obstacle_accepted(obstacle):
        reasons.append("positive_obstacle_rejected")
    try:
        shared, why = shared_matched_terminal(direct or {}, obstacle or {})
    except (TypeError, KeyError, ValueError):
        shared, why = False, "shared_terminal_error"
    if not shared:
        reasons.append(why or "shared_terminal_rejected")
    if isinstance(direct, dict) and isinstance(obstacle, dict) and isinstance(replay, dict):
        values = [direct.get("release_tcp_z"), obstacle.get("release_tcp_z"), replay.get("release_tcp_z")]
        if any(value is None for value in values) or max(values) - min(values) > 1e-5:
            reasons.append("release_tcp_z_mismatch")
    if not isinstance(replay, dict) or not terminal_replay_accepted(replay):
        reasons.append("terminal_replay_rejected")
    if negative_direct_admits(negative):
        reasons.append("negative_admitted")
    return {
        "collection_ready": not reasons,
        "reasons": reasons,
        "negative_admits_demos": False,
        "negative_is_feasibility_failure": False,
    }


def confirmation_decision(rows):
    if not rows:
        return {"collection_ready": False, "ready_count": 0, "denominator": 0, "reasons": ["empty"],
                "negative_admits_demos": False}
    decisions = [
        collection_admission(
            row.get("positive_direct"), row.get("positive_obstacle"),
            row.get("terminal_replay"), row.get("negative_direct"))
        for row in rows
    ]
    ready = [item["collection_ready"] for item in decisions]
    return {
        "collection_ready": bool(ready) and all(ready),
        "ready_count": int(sum(ready)),
        "denominator": len(rows),
        "reasons": [] if all(ready) else ["layout_rejected"],
        "negative_admits_demos": False,
        "layout_reasons": [item["reasons"] for item in decisions],
    }


def select_representative(rows, *, count, excluded_seeds, seed):
    excluded = {int(value) for value in excluded_seeds}
    pool = [dict(row) for row in rows if int(row["seed"]) not in excluded]
    if len(pool) < int(count):
        raise ValueError("representative pool is smaller than the requested count")
    distances = np.array([float(row["distance_m"]) for row in pool], dtype=np.float64)
    cuts = np.quantile(distances, [1.0 / 3.0, 2.0 / 3.0])

    def band(row):
        if float(row["distance_m"]) <= float(cuts[0]):
            return 0
        if float(row["distance_m"]) <= float(cuts[1]):
            return 1
        return 2

    cells = {}
    for row in pool:
        key = (row["source"], "high" if float(row["height_m"]) >= HEIGHT_HIGH_M else "typical", band(row))
        cells.setdefault(key, []).append(row)
    rng = np.random.default_rng(int(seed))
    for key in cells:
        order = rng.permutation(len(cells[key]))
        cells[key] = [cells[key][int(index)] for index in order]
    keys = sorted(cells, key=lambda item: (str(item[0]), str(item[1]), int(item[2])))
    picked = []
    seen = set()

    def take(row):
        ident = (row["source"], int(row["index"]), int(row["seed"]))
        if ident in seen:
            return False
        picked.append(row)
        seen.add(ident)
        return True

    for source in sorted({row["source"] for row in pool}):
        for key in keys:
            if key[0] == source and cells[key]:
                take(cells[key].pop(0))
                break
        if len(picked) == int(count):
            return picked
    while len(picked) < int(count):
        progressed = False
        for key in keys:
            if not cells[key]:
                continue
            if take(cells[key].pop(0)):
                progressed = True
            if len(picked) == int(count):
                return picked
        if not progressed:
            raise ValueError("could not fill the representative count")
    return picked


def lowest_hand_vertex_z(bodies):
    points = []
    for name, value in bodies.items():
        if name in HAND_LINK_NAMES or "finger" in name or name == "panda_hand":
            points.append(np.asarray(value, dtype=np.float64).reshape(-1, 3))
    if not points:
        raise ValueError("hand collision vertices were not recorded")
    return float(np.min(np.concatenate(points, axis=0)[:, 2]))


def hand_xy_extent_m(bodies, tcp_xy):
    tcp_xy = np.asarray(tcp_xy, dtype=np.float64).reshape(2)
    points = []
    for name, value in bodies.items():
        if name in HAND_LINK_NAMES or "finger" in name or name == "panda_hand":
            points.append(np.asarray(value, dtype=np.float64).reshape(-1, 3))
    if not points:
        raise ValueError("hand collision vertices were not recorded")
    stacked = np.concatenate(points, axis=0)
    return float(np.max(np.linalg.norm(stacked[:, :2] - tcp_xy, axis=1)))


def freeze_identity(repo_root=None):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in IDENTITY_FILES}
    return {
        "basis": "working-tree bytes frozen before physics",
        "git_head_is_code_identity": False,
        "files": files,
        "files_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def _native(base):
    return bool(base.evaluate()["native_success"][0].item())


def _mark(trace, name, status):
    trace["phase_status"].append((name, status))


def execute_above_release(*, step, move, base, grasp_quat, shared_targets, trace, cylinder_top_z,
                          margin_m, is_done, action_count, bodies_fn):
    """Reach bin XY above the measured hand clearance, open, then retreat."""
    if is_done():
        return {"ok": False, "reason": "already_done", "release_begin": None}
    tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
    bodies = bodies_fn()
    lowest = lowest_hand_vertex_z(bodies)
    hand_below = max(0.0, float(tcp[2]) - lowest)
    geometric = float(cylinder_top_z) + hand_below + float(margin_m)
    approach = np.asarray(shared_targets["container_approach_xyz"], dtype=np.float64)
    exit_z = float(np.asarray(shared_targets["transport_exit_xyz"], dtype=np.float64)[2])
    if shared_targets.get("terminal_mode") == "above_release" and shared_targets.get("release_tcp_z") is not None:
        release_z = float(shared_targets["release_tcp_z"])
    else:
        release_z = max(geometric, exit_z)
        shared_targets["terminal_mode"] = "above_release"
        shared_targets["release_tcp_z"] = float(release_z)
        shared_targets["clearance_xyz"] = [float(approach[0]), float(approach[1]), float(release_z)]
        shared_targets["matched_retreat_xyz"] = [float(approach[0]), float(approach[1]), float(release_z) + RETREAT_DZ_M]
        shared_targets["hand_below_tcp_m"] = float(hand_below)
        shared_targets["geometric_is_surface_clearance"] = False
        shared_targets["radius_m"] = RADIUS_M
        shared_targets["native_xy_m"] = NATIVE_XY_M
    clearance = np.array([float(approach[0]), float(approach[1]), float(release_z)], dtype=np.float64)
    if float(clearance[2]) + 1e-4 < geometric:
        _mark(trace, "clearance_approach", "below_measured_clearance")
        return {"ok": False, "reason": "shared_release_below_measured_clearance", "release_begin": None,
                "release_tcp_z": float(release_z), "geometric_tcp_z": float(geometric)}
    trace["container_approach_action_index"] = action_count()
    trace["release_tcp_z"] = float(release_z)
    trace["geometric_tcp_z"] = float(geometric)
    trace["geometric_is_surface_clearance"] = False
    reached = move(clearance, grasp_quat, grip=-1.0, tolerance=0.006, budget=40)
    _mark(trace, "clearance_approach", "reached" if reached and not is_done() else ("episode_ended" if is_done() else "timeout"))
    if not reached or is_done():
        return {"ok": False, "reason": "clearance_not_reached", "release_begin": None, "release_tcp_z": float(release_z)}
    bodies = bodies_fn()
    held_low = lowest_hand_vertex_z(bodies)
    trace["hand_vertex_above_top_m"] = float(held_low - float(cylinder_top_z))
    trace["hand_vertex_is_surface_clearance"] = False
    if held_low <= float(cylinder_top_z):
        _mark(trace, "stabilize", "hand_below_cylinder_top")
        return {"ok": False, "reason": "hand_below_cylinder_top", "release_begin": None, "release_tcp_z": float(release_z)}
    bin_xy = base.bin.pose.p[0, :2].detach().cpu().numpy()
    stable = False
    for _ in range(STABILIZE_BUDGET):
        if is_done():
            break
        sphere = base.obj.pose.p[0].detach().cpu().numpy()
        velocity = base.obj.linear_velocity[0].detach().cpu().numpy()
        error = bin_xy - sphere[:2]
        if float(np.linalg.norm(error)) <= PRE_RELEASE_XY_M and float(np.linalg.norm(velocity)) <= NATIVE_LIN_THRESH_M_S:
            stable = True
            break
        current = base.agent.tcp.pose.p[0].detach().cpu().numpy()
        target = current.copy()
        target[:2] = current[:2] + error
        target[2] = float(release_z)
        move(target, grasp_quat, grip=-1.0, tolerance=0.003, budget=3)
    if not stable:
        for _ in range(6):
            if is_done() or step([0, 0, 0, 0, 0, 0, -1]):
                break
            sphere = base.obj.pose.p[0].detach().cpu().numpy()
            velocity = base.obj.linear_velocity[0].detach().cpu().numpy()
            if float(np.linalg.norm(bin_xy - sphere[:2])) <= PRE_RELEASE_XY_M and float(np.linalg.norm(velocity)) <= NATIVE_LIN_THRESH_M_S:
                stable = True
                break
    _mark(trace, "stabilize", "reached" if stable else "timeout")
    if not stable or is_done():
        return {"ok": False, "reason": "sphere_not_stable_before_release", "release_begin": None,
                "release_tcp_z": float(release_z)}
    release_begin = action_count()
    step([0, 0, 0, 0, 0, 0, 1])
    _mark(trace, "release", "episode_ended" if is_done() and not _native(base) else "reached")
    retreat_target = clearance.copy()
    retreat_target[2] += RETREAT_DZ_M
    planned = 20
    if _native(base):
        retreat = retreat_after_release(
            native_success_before_retreat=True, retreat_steps_executed=0, retreat_steps_planned=planned)
        _mark(trace, "retreat", "truncated_after_native_success")
        _mark(trace, "settle", "truncated_after_native_success")
    else:
        begin = action_count()
        move(retreat_target, grasp_quat, grip=1.0, tolerance=0.006, budget=planned)
        executed = action_count() - begin
        tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
        at_target = float(np.linalg.norm(tcp - retreat_target)) < 0.006
        during = bool(_native(base) and not at_target)
        retreat = retreat_after_release(
            native_success_before_retreat=False, retreat_steps_executed=executed, retreat_steps_planned=planned,
            native_success_during_retreat=during)
        _mark(trace, "retreat", "truncated_after_native_success" if during else ("reached" if at_target else "timeout"))
        if _native(base):
            _mark(trace, "settle", "truncated_after_native_success")
        else:
            for _ in range(SETTLE_BUDGET):
                if step([0, 0, 0, 0, 0, 0, 1]):
                    break
            _mark(trace, "settle", "episode_ended" if is_done() else "reached")
    trace["retreat_contract"] = retreat
    trace["release_tcp_z"] = float(release_z)
    return {
        "ok": bool(_native(base)),
        "reason": None if _native(base) else "native_success_not_reached",
        "release_begin": release_begin,
        "release_tcp_z": float(release_z),
        "retreat": retreat,
    }


def load_catalog_rows(pair_dir=PAIR_CATALOG, obstacle_dir=OBSTACLE200_CATALOG, dev_dir=DEV_SCENES):
    rows = []
    for source, directory in (("pair100", pair_dir), ("obstacle200", obstacle_dir)):
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        for index, scene in enumerate(manifest["scenes"]):
            center = np.asarray(scene["cylinder"]["center_xy"], dtype=np.float64)
            goal = np.asarray(scene["container_xy"], dtype=np.float64)
            rows.append({
                "source": source,
                "index": index,
                "seed": int(scene["seed"]),
                "scene_id": scene["scene_id"],
                "height_m": float(scene["cylinder"]["height"]),
                "distance_m": float(np.linalg.norm(center - goal)),
                "catalog": str(directory),
            })
    dev = json.loads((Path(dev_dir) / "manifest.json").read_text())
    excluded = {int(scene["seed"]) for scene in dev["scenes"]}
    return rows, excluded


def write_registration(path, *, count=24, calibration_count=8, seed=2026102518):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rows, excluded = load_catalog_rows()
    chosen = select_representative(rows, count=count, excluded_seeds=excluded, seed=seed)
    identity = freeze_identity()
    payload = {
        "kind": "matched_terminal_feasibility",
        "version": "r19_role_split",
        "previous_runtime_identity": PREVIOUS_RUNTIME_IDENTITY,
        "previous_identity_rewritten": False,
        "roles": {
            ROLE_POSITIVE_DIRECT: "cylinder parked/absent; same initial task state; terminal height from planned cylinder geometry; full prefix native and contact-free; does not admit alone",
            ROLE_POSITIVE_OBSTACLE: "catalog cylinder not relocated; full prefix safe from execution start; does not admit alone",
            ROLE_NEGATIVE_DIRECT: "straight path into the blocking cylinder; expected transport block; never admits demos and is not a data-feasibility failure",
            ROLE_TERMINAL_REPLAY: "direct terminal actions replayed with the actual cylinder; prefix latches labeled; not a full-prefix proof",
        },
        "hypothesis": (
            "Closed-grip negative-Z near the bin is the copied top-cap mechanism. "
            "An opt-in shared terminal that holds bin XY at a TCP Z where measured hand "
            "collision vertices clear the recorded cylinder top, stabilizes, opens, and "
            "retreats only after release will keep native on-bin/static/ungrasped success "
            "without a closed descent through the cap. Vertex clearance is approximate. "
            "PhysX contact and native outcome are the proof. Direct positive runs with the "
            "cylinder parked. A straight path into the blocking cylinder is a negative probe."
        ),
        "candidates": {
            "above_release_margins_m": list(ABOVE_RELEASE_MARGINS_M),
            "side_offset_headings_rad": list(SIDE_OFFSET_HEADINGS_RAD),
            "side_offset_m": SIDE_OFFSET_M,
            "side_offset_only_if_all_above_release_fail": True,
            "never_descend_below_measured_clearance_while_closed": True,
        },
        "native_thresholds": {
            "xy_m": NATIVE_XY_M,
            "z_tol_m": NATIVE_Z_TOL_M,
            "lin_m_s": NATIVE_LIN_THRESH_M_S,
            "ang_rad_s": NATIVE_ANG_THRESH_RAD_S,
            "ungrasped": True,
            "changed": False,
        },
        "radius_m": RADIUS_M,
        "cylinder_pose_and_height_changed": False,
        "selection": {
            "calibration_count": calibration_count,
            "confirmation_count": count - calibration_count,
            "rule": "lowest margin whose every calibration obstacle run is native-safe from execution start; confirmation does not re-pick",
            "collection_ready_iff": COLLECTION_READY_IFF,
            "perturbations_m": list(PERTURB_DISTANCES_M),
            "perturbation_axes": ["toward_cylinder", "away_from_cylinder"],
            "perturbation_is_finite_actions": True,
            "state_teleport_forbidden": True,
            "smoke_not_in_denominator": True,
        },
        "layouts": chosen,
        "excluded_dev_seed_count": len(excluded),
        "dev_not_used_for_selection": True,
        "identity": identity,
        "historical_identities_rewritten": False,
        "full_collect": False,
        "training": False,
        "budgets": {
            "episode_wall_s": 160,
            "calibration_phase_wall_s": 2600,
            "confirmation_phase_wall_s": 2400,
            "heldout_phase_wall_s": 2200,
            "max_steps": 360,
            "action_width": 7,
        },
        "prior_runs_not_in_denominator": [
            "outputs/exp/20261025_134512_placesphere_matched_terminal/smoke.json",
            "outputs/exp/20261025_134614_placesphere_matched_terminal/smoke.json",
        ],
        "prior_smoke_note": (
            "Not used to choose margins or drop layouts. The first smoke was a summary "
            "ValueError. The second, on frozen layout 0, had obstacle native success, "
            "no contact, release_tcp_z 0.262, hand vertex gap 0.0034 m which is not "
            "surface clearance, and a direct transport timeout."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def assert_registration(path):
    registration = json.loads(Path(path).read_text())
    if registration.get("skip_identity") or registration.get("exempt") or registration.get("identity_bypass"):
        raise RuntimeError("identity bypass is rejected")
    current = freeze_identity()
    if current["files_sha256"] != registration["identity"]["files_sha256"]:
        raise RuntimeError("source identity does not match preregistration")
    if registration.get("selection", {}).get("collection_ready_iff") != COLLECTION_READY_IFF:
        raise RuntimeError("collection_ready_iff does not match the runtime predicate")
    if registration.get("native_thresholds", {}).get("changed") or registration.get("radius_m") != RADIUS_M:
        raise RuntimeError("registration loosens a fixed contract")
    rows, excluded = load_catalog_rows()
    expected = select_representative(
        rows, count=len(registration["layouts"]), excluded_seeds=excluded, seed=2026102518)
    if [(row["source"], row["index"], row["seed"]) for row in expected] != [
            (row["source"], row["index"], row["seed"]) for row in registration["layouts"]]:
        raise RuntimeError("frozen layout subset does not match the selector")
    return registration, current


class EpisodeDeadline(Exception):
    pass


def _alarm(_signum, _frame):
    raise EpisodeDeadline("episode wall exceeded")


def _child_pids():
    me = os.getpid()
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except OSError:
            continue
        if ppid == me:
            found.append(int(entry.name))
    return found


def _kill_new_children(before):
    before = set(before)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in _child_pids():
            if pid in before:
                continue
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def _scene_row(layout):
    manifest = json.loads((Path(layout["catalog"]) / "manifest.json").read_text())
    scene = manifest["scenes"][int(layout["index"])]
    if int(scene["seed"]) != int(layout["seed"]):
        raise RuntimeError("catalog index does not match the frozen seed")
    return scene


def _open_scene(layout):
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
    from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
    from examples.baselines.constrain.pickcube_expert import snapshot_execution_state

    scene = _scene_row(layout)
    catalog = TransportDetourSceneSet(layout["catalog"])
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=400, obstacle_radius=float(scene["cylinder"]["radius"]),
        obstacle_height=float(scene["cylinder"]["height"])))
    try:
        env, restored = catalog.restore(int(layout["index"]), env)
    except BaseException:
        env.close()
        raise
    if not restored.get("bit_exact"):
        env.close()
        raise RuntimeError("scene restore is not bit-exact")
    return env, scene, snapshot_execution_state(env)


def _planned_cylinder(scene):
    cylinder = scene["cylinder"]
    height = float(cylinder["height"])
    return {
        "center_xy": [float(cylinder["center_xy"][0]), float(cylinder["center_xy"][1])],
        "height_m": height,
        "radius_m": float(cylinder["radius"]),
        "top_z": height,
    }


def park_execution_snapshot(snapshot):
    from examples.baselines.constrain.pickcube_expert import _clone_tree
    from examples.baselines.constrain.placesphere_obstacle import OBSTACLE_KEY, PARKED_CENTER

    state, controller = snapshot
    state = _clone_tree(state)
    controller = _clone_tree(controller)
    extra = state[OBSTACLE_KEY].detach().clone()
    info = {
        "initial_contact_latch": bool(float(extra[0, 4]) > 0.5),
        "initial_crossing_latch": bool(float(extra[0, 5]) > 0.5),
        "parked_center": [float(PARKED_CENTER[0]), float(PARKED_CENTER[1])],
    }
    extra[0, 0] = float(PARKED_CENTER[0])
    extra[0, 1] = float(PARKED_CENTER[1])
    extra[0, 3] = 0.0
    state[OBSTACLE_KEY] = extra
    return (state, controller), info


def _action_digest(actions):
    if actions is None:
        return {"action_width": None, "action_count": 0, "actions_sha256": None, "actions": None}
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2:
        return {"action_width": None, "action_count": 0, "actions_sha256": None, "actions": None}
    width = int(array.shape[1])
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() if width == 7 else None
    return {
        "action_width": width,
        "action_count": int(array.shape[0]),
        "actions_sha256": digest,
        "actions": array.tolist() if width == 7 and array.shape[0] <= 360 else None,
    }


def _first_contact(contact, actions):
    if not contact:
        return None
    step = contact.get("control_step")
    action = None
    if step is not None and actions is not None:
        index = int(step) - 1
        array = np.asarray(actions)
        if 0 <= index < len(array):
            action = np.asarray(array[index], dtype=np.float64).reshape(-1).tolist()
    return {
        "phase": contact.get("phase"),
        "step": None if step is None else int(step),
        "link": contact.get("other"),
        "tcp_xyz": contact.get("tcp_xyz"),
        "sphere_xyz": contact.get("sphere_xyz"),
        "max_impulse": contact.get("max_impulse"),
        "action": action,
    }


def _summary(result, *, role, planned, parked, relocated, contact_checks_ran, latch_info=None,
             release_height_from_planned_geometry=False, execution_center=None):
    terminal = result.get("terminal") or {}
    safety = result.get("safety") or {}
    trace = result.get("trace") or {}
    phases = {name for name, status in trace.get("phase_status", [])}
    retreat = trace.get("retreat_contract") or {}
    contact = trace.get("first_obstacle_contact") or {}
    terminal_phases = {
        "clearance_approach", "stabilize", "release", "settle", "retreat", "container_approach"}
    actions = None if result.get("actions") is None else np.asarray(result["actions"])
    digest = _action_digest(actions)
    shared = trace.get("shared_targets") or {}
    path_uses_waypoint = role == ROLE_POSITIVE_OBSTACLE
    targets = {key: shared.get(key) for key in TARGET_KEYS}
    if targets["release_tcp_z"] is None:
        targets["release_tcp_z"] = trace.get("release_tcp_z")
    initial_latch = bool((latch_info or {}).get("initial_contact_latch") or (latch_info or {}).get("initial_crossing_latch"))
    run_contact = bool(safety.get("obstacle_collision"))
    run_crossing = bool(safety.get("obstacle_crossing"))
    record = {
        "role": role,
        "kind": role,
        "native_success": bool(terminal.get("native_success")),
        "on_bin": bool(terminal.get("on_bin")),
        "static": bool(terminal.get("static")),
        "ungrasped": bool(terminal.get("ungrasped")),
        "obstacle_contact": run_contact,
        "obstacle_crossing": run_crossing,
        "terminal_contact": bool(contact.get("phase") in terminal_phases),
        "terminal_crossing": run_crossing,
        "transport_contact": bool(contact.get("phase") == "transport"),
        "release_reached": "release" in phases,
        "retreat_reached": any(status == "reached" for name, status in trace.get("phase_status", []) if name == "retreat"),
        "retreat_required": True,
        "truncated_after_native_success": bool(retreat.get("truncated_after_native_success")),
        "continued_after_first_native_success": bool(retreat.get("continued_after_first_native_success")),
        "radius_m": RADIUS_M,
        "native_xy_m": NATIVE_XY_M,
        "native_thresholds_changed": False,
        "from_execution_start": True,
        "from_execution_start_safe": bool(not run_contact and not run_crossing and not initial_latch),
        "prefix_latch_cleared_as_proof": False,
        "cylinder_parked": bool(parked),
        "actual_cylinder": bool(role == ROLE_POSITIVE_OBSTACLE and not parked and not relocated),
        "obstacle_relocated": bool(relocated),
        "planned_cylinder_recorded": planned is not None,
        "planned_cylinder": planned,
        "release_height_from_planned_geometry": bool(release_height_from_planned_geometry),
        "planned_cylinder_contact_proof": bool(role == ROLE_POSITIVE_OBSTACLE and contact_checks_ran and not parked),
        "contact_checks_ran": bool(contact_checks_ran),
        "path_uses_waypoint": path_uses_waypoint,
        "execution_center_xy": None if execution_center is None else [float(execution_center[0]), float(execution_center[1])],
        "release_tcp_z": trace.get("release_tcp_z"),
        "geometric_tcp_z": trace.get("geometric_tcp_z"),
        "geometric_is_surface_clearance": False,
        "hand_vertex_above_top_m": trace.get("hand_vertex_above_top_m"),
        "matched_reason": (trace.get("matched_terminal") or {}).get("reason"),
        "steps": digest["action_count"],
        "phase_status": [list(item) for item in (trace.get("phase_status") or [])],
        "first_contact": _first_contact(contact, actions),
        "retreat": retreat,
        "targets": targets,
        "waypoint_xyz": shared.get("waypoint_xyz") if path_uses_waypoint else None,
        "action_width": digest["action_width"],
        "actions_sha256": digest["actions_sha256"],
        "actions": digest["actions"],
        "initial_latches": latch_info,
        "safety_scope": "parked_execution" if parked else "actual_cylinder_full_prefix",
        "negative_is_feasibility_failure": False,
        "expected_transport_block": role == ROLE_NEGATIVE_DIRECT,
    }
    tcp = trace.get("tcp_xyz")
    tcp = np.asarray([] if tcp is None else tcp)
    release_z = trace.get("release_tcp_z")
    closed_down = False
    if release_z is not None and actions is not None and len(actions) and len(tcp) == len(actions) + 1:
        phases_at = trace.get("step_phase") or []
        for action, z, phase in zip(actions, tcp[1:], phases_at):
            if phase in {"clearance_approach", "stabilize"} and float(action[6]) < 0 and float(action[2]) < -0.05 and float(z) < float(release_z) - 0.005:
                closed_down = True
    record["closed_descent_below_clearance"] = closed_down
    record["accepted_obstacle"] = positive_obstacle_accepted(record) if role == ROLE_POSITIVE_OBSTACLE else None
    record["accepted_direct_suffix"] = positive_direct_accepted(record) if role == ROLE_POSITIVE_DIRECT else False
    record["admits_demos"] = False if role == ROLE_NEGATIVE_DIRECT else None
    return record


def _rejected(layout, reason, *, role, error=None):
    return {
        "ok": False, "rejected": True, "reason": reason, "role": role,
        "scene_id": None if layout is None else layout.get("scene_id"),
        "error": error, "admits_demos": False, "from_execution_start_safe": False,
        "negative_is_feasibility_failure": False,
    }


def run_layout_expert(layout, *, margin_m, role, shared_targets=None, episode_wall_s=160):
    from examples.baselines.constrain.placesphere_obstacle import PARKED_CENTER
    from examples.baselines.constrain.placesphere_transport_detour import run_transport_expert

    if role not in (ROLE_POSITIVE_DIRECT, ROLE_POSITIVE_OBSTACLE, ROLE_NEGATIVE_DIRECT):
        raise ValueError(f"unknown execution role {role}")
    before = set(_child_pids())
    env = None
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
    try:
        env, scene, snapshot = _open_scene(layout)
        planned = _planned_cylinder(scene)
        if abs(planned["radius_m"] - RADIUS_M) > 1e-6:
            return _rejected(layout, "radius_changed", role=role)
        latch_info = None
        parked = role == ROLE_POSITIVE_DIRECT
        detour = role == ROLE_POSITIVE_OBSTACLE
        if parked:
            snapshot, latch_info = park_execution_snapshot(snapshot)
        targets = dict(scene["expert"]["fork_targets"] if shared_targets is None else shared_targets)
        waypoint = np.asarray(targets["waypoint_xyz"], dtype=np.float64)[:2] if detour else None
        center = np.asarray(PARKED_CENTER if parked else planned["center_xy"], dtype=np.float64)
        result = run_transport_expert(
            env, np.asarray(scene["start_xy"]), np.asarray(scene["container_xy"]),
            detour=detour, waypoint=waypoint, seed=int(scene["seed"]), max_steps=360,
            obstacle_enabled=not parked, obstacle_center=center,
            initial_snapshot=snapshot, shared_targets=targets, terminal_mode="above_release",
            release_margin_m=float(margin_m), record_body_sweep=False,
            stop_on_contact=role == ROLE_NEGATIVE_DIRECT)
        if result is None:
            return _rejected(layout, "expert_returned_none", role=role)
        base = env.unwrapped
        executed = base._obstacle_center[0].detach().cpu().numpy()
        constructed = float(base.obstacle_collision_height)
        enabled = bool(base._obstacle_enabled[0].item())
        relocated = bool(
            role == ROLE_POSITIVE_OBSTACLE
            and not np.allclose(executed, planned["center_xy"], atol=1e-5))
        summary = _summary(
            result, role=role, planned=planned, parked=parked and not enabled,
            relocated=relocated, contact_checks_ran=enabled, latch_info=latch_info,
            release_height_from_planned_geometry=abs(constructed - planned["height_m"]) < 1e-5,
            execution_center=executed)
        summary["scene_id"] = scene["scene_id"]
        summary["seed"] = int(scene["seed"])
        summary["shared_targets"] = result["trace"].get("shared_targets")
        summary["bit_exact_restore"] = True
        summary["constructed_cylinder_height_m"] = constructed
        summary["_raw"] = result
        if role == ROLE_NEGATIVE_DIRECT:
            summary["blocked_in_transport"] = bool(summary.get("transport_contact") and not summary.get("native_success"))
            summary["admits_demos"] = False
        return summary
    except EpisodeDeadline:
        return _rejected(layout, "episode_deadline", role=role)
    except Exception as exc:
        rejected = _rejected(layout, "exception", role=role, error=f"{type(exc).__name__}: {exc}")
        rejected["traceback"] = traceback.format_exc()
        return rejected
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if env is not None:
            env.close()
        _kill_new_children(before)


def replay_direct_terminal(layout, direct_result, *, episode_wall_s=160):
    import torch
    from examples.baselines.constrain.pickcube_expert import restore_execution_state
    from examples.baselines.constrain.placesphere_transport_detour import _terminal

    raw = (direct_result or {}).get("_raw") or {}
    trace = raw.get("trace") or {}
    snapshot = trace.get("container_approach_snapshot")
    index = trace.get("container_approach_action_index")
    actions = raw.get("actions")
    release_z = None if direct_result is None else direct_result.get("release_tcp_z")
    planned = None if direct_result is None else direct_result.get("planned_cylinder")
    if snapshot is None or index is None or actions is None or len(np.asarray(actions)) <= int(index):
        return {
            "role": ROLE_TERMINAL_REPLAY, "rejected": True, "reason": "terminal_not_recorded",
            "full_prefix_safe": False, "prefix_latch_cleared_as_proof": False,
            "prefix_latches_labeled": True, "actual_cylinder": True, "cylinder_parked": False,
            "release_tcp_z": release_z, "scene_id": None if layout is None else layout.get("scene_id"),
            "admits_demos": False, "native_success": False,
        }
    before = set(_child_pids())
    env = None
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
    try:
        env, scene, _snapshot = _open_scene(layout)
        planned = _planned_cylinder(scene)
        restore_execution_state(env, snapshot)
        base = env.unwrapped
        prefix_contact = bool(base._obstacle_collision[0].item())
        prefix_crossing = bool(base._forbidden_crossing[0].item())
        center = np.asarray(planned["center_xy"], dtype=np.float64)
        base._obstacle_enabled[:] = True
        base._set_obstacle_pose(center)
        base._obstacle_collision[:] = False
        base._forbidden_crossing[:] = False
        base._first_obstacle_contact = None
        contact = False
        crossing = False
        suffix = np.asarray(actions[int(index):], dtype=np.float32)
        if suffix.ndim != 2 or suffix.shape[1] != 7:
            return {
                "role": ROLE_TERMINAL_REPLAY, "rejected": True, "reason": "action_width",
                "full_prefix_safe": False, "prefix_latches_labeled": True,
                "prefix_latch_cleared_as_proof": False, "action_width": None if suffix.ndim != 2 else int(suffix.shape[-1]),
                "scene_id": scene["scene_id"],
            }
        for action in suffix:
            env.step(torch.as_tensor(action, device=base.device))
            contact = contact or bool(base._obstacle_collision[0].item())
            crossing = crossing or bool(base._forbidden_crossing[0].item())
        terminal = _terminal(base.evaluate())
        executed = base._obstacle_center[0].detach().cpu().numpy()
        same = bool(
            np.allclose(executed, center, atol=1e-4)
            and abs(float(base.obstacle_collision_height) - planned["height_m"]) < 1e-5)
        digest = _action_digest(suffix)
        return {
            "role": ROLE_TERMINAL_REPLAY,
            "scene_id": scene["scene_id"],
            "seed": int(scene["seed"]),
            "actual_cylinder": True,
            "cylinder_parked": False,
            "obstacle_relocated": not same,
            "same_planned_geometry": same,
            "planned_cylinder": planned,
            "prefix_contact_latch": prefix_contact,
            "prefix_crossing_latch": prefix_crossing,
            "prefix_latches_labeled": True,
            "prefix_latch_cleared_as_proof": False,
            "latch_cleared_for_terminal_check": True,
            "full_prefix_safe": False,
            "terminal_contact": contact,
            "terminal_crossing": crossing,
            "native_success": terminal["native_success"],
            "on_bin": terminal["on_bin"],
            "static": terminal["static"],
            "ungrasped": terminal["ungrasped"],
            "radius_m": RADIUS_M,
            "native_xy_m": NATIVE_XY_M,
            "native_thresholds_changed": False,
            "action_width": digest["action_width"],
            "actions_sha256": digest["actions_sha256"],
            "actions": digest["actions"],
            "release_tcp_z": release_z,
            "geometric_is_surface_clearance": False,
            "admits_demos": False,
        }
    except EpisodeDeadline:
        return {"role": ROLE_TERMINAL_REPLAY, "rejected": True, "reason": "episode_deadline",
                "full_prefix_safe": False, "prefix_latches_labeled": True,
                "prefix_latch_cleared_as_proof": False, "scene_id": layout.get("scene_id")}
    except Exception as exc:
        return {
            "role": ROLE_TERMINAL_REPLAY, "rejected": True, "reason": "exception",
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "full_prefix_safe": False, "prefix_latches_labeled": True,
            "prefix_latch_cleared_as_proof": False, "scene_id": layout.get("scene_id"),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if env is not None:
            env.close()
        _kill_new_children(before)


def _public_record(record):
    if not isinstance(record, dict):
        return record
    return {key: value for key, value in record.items() if key not in {"_raw", "shared_targets"}}


def run_role_bundle(layout, *, margin_m, episode_wall_s=160):
    obstacle = run_layout_expert(layout, margin_m=margin_m, role=ROLE_POSITIVE_OBSTACLE, episode_wall_s=episode_wall_s)
    direct = run_layout_expert(
        layout, margin_m=margin_m, role=ROLE_POSITIVE_DIRECT,
        shared_targets=obstacle.get("shared_targets"), episode_wall_s=episode_wall_s)
    replay = replay_direct_terminal(layout, direct, episode_wall_s=episode_wall_s)
    negative = run_layout_expert(layout, margin_m=margin_m, role=ROLE_NEGATIVE_DIRECT, episode_wall_s=episode_wall_s)
    decision = collection_admission(direct, obstacle, replay, negative)
    return {
        "layout": layout,
        "positive_obstacle": _public_record(obstacle),
        "positive_direct": _public_record(direct),
        "terminal_replay": replay,
        "negative_direct": _public_record(negative),
        "admission": decision,
        "rejected": not decision["collection_ready"],
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_smoke(registration_path, output_dir):
    registration, identity = assert_registration(registration_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    layout = registration["layouts"][0]
    margin = ABOVE_RELEASE_MARGINS_M[0]
    started = time.monotonic()
    bundle = run_role_bundle(layout, margin_m=margin, episode_wall_s=160)
    payload = {
        "in_selection_denominator": False,
        "margin_m": margin,
        "layout": layout,
        "identity": identity["files_sha256"],
        "elapsed_s": time.monotonic() - started,
        "bundle": bundle,
        "collection_ready": False,
        "radius_m": RADIUS_M,
        "native_thresholds_changed": False,
    }
    _write_json(output / "smoke.json", payload)
    return payload


def _layout_key(layout):
    return f"{layout['source']}-{int(layout['index']):03d}"


def run_calibration(registration_path, output_dir, *, episode_wall_s=160, phase_wall_s=2600):
    registration, identity = assert_registration(registration_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = int(registration["selection"]["calibration_count"])
    layouts = registration["layouts"][:count]
    deadline = time.monotonic() + float(phase_wall_s)
    rows = []
    exceptions = []
    for margin in ABOVE_RELEASE_MARGINS_M:
        margin_rows = []
        for layout in layouts:
            if time.monotonic() > deadline:
                payload = {"complete": False, "reason": "phase_wall", "rows": rows, "identity": identity["files_sha256"]}
                _write_json(output / "calibration.json", payload)
                return payload
            obstacle = run_layout_expert(
                layout, margin_m=margin, role=ROLE_POSITIVE_OBSTACLE, episode_wall_s=episode_wall_s)
            if obstacle.get("reason") == "exception":
                exceptions.append(obstacle.get("error"))
            row = {
                "margin_m": margin,
                "layout": layout,
                "obstacle": _public_record(obstacle),
                "role": ROLE_POSITIVE_OBSTACLE,
                "rejected": not bool(obstacle.get("accepted_obstacle")),
            }
            margin_rows.append(row)
            rows.append(row)
            _write_json(output / "calibration_partial.json", {"rows": rows, "identity": identity["files_sha256"]})
            if exceptions.count(exceptions[-1]) >= 3 if exceptions else False:
                payload = {"complete": False, "reason": "repeated_exception", "error": exceptions[-1], "rows": rows}
                _write_json(output / "calibration.json", payload)
                return payload
        decision_rows = []
        for item in margin_rows:
            obstacle = item["obstacle"]
            decision_rows.append({
                "margin_m": margin,
                "native_success": bool(obstacle.get("native_success")),
                "on_bin": bool(obstacle.get("on_bin")),
                "static": bool(obstacle.get("static")),
                "ungrasped": bool(obstacle.get("ungrasped")),
                "obstacle_contact": bool(obstacle.get("obstacle_contact")),
                "obstacle_crossing": bool(obstacle.get("obstacle_crossing")),
                "closed_descent_below_clearance": bool(obstacle.get("closed_descent_below_clearance")),
                "from_execution_start": True,
                "native_xy_m": NATIVE_XY_M,
                "radius_m": RADIUS_M,
                "native_thresholds_changed": False,
            })
        if all(_passes(item) for item in decision_rows):
            chosen = select_terminal_candidate(decision_rows)
            bundles = []
            for layout in layouts:
                if time.monotonic() > deadline:
                    payload = {
                        "complete": False, "reason": "phase_wall_during_role_bundle",
                        "rows": rows, "role_bundles": bundles, "selected_margin_m": margin,
                        "decision": chosen, "identity": identity["files_sha256"],
                        "collection_ready": False,
                    }
                    _write_json(output / "calibration.json", payload)
                    return payload
                bundles.append(run_role_bundle(layout, margin_m=margin, episode_wall_s=episode_wall_s))
                _write_json(output / "calibration_partial.json", {
                    "rows": rows, "role_bundles": bundles, "identity": identity["files_sha256"],
                })
            payload = {
                "complete": True, "rows": rows, "role_bundles": bundles, "selected_margin_m": margin,
                "decision": chosen, "identity": identity["files_sha256"],
                "denominator": len(margin_rows), "native_thresholds_changed": False,
                "collection_ready": False,
                "role_bundle_ready": int(sum(not row["rejected"] for row in bundles)),
                "negative_admits_demos": False,
                "selection_uses_obstacle_only": True,
            }
            _write_json(output / "calibration.json", payload)
            return payload
    decision = select_terminal_candidate([
        {
            "margin_m": item["margin_m"],
            "native_success": bool(item["obstacle"].get("native_success")),
            "on_bin": bool(item["obstacle"].get("on_bin")),
            "static": bool(item["obstacle"].get("static")),
            "ungrasped": bool(item["obstacle"].get("ungrasped")),
            "obstacle_contact": bool(item["obstacle"].get("obstacle_contact")),
            "obstacle_crossing": bool(item["obstacle"].get("obstacle_crossing")),
            "closed_descent_below_clearance": bool(item["obstacle"].get("closed_descent_below_clearance")),
            "from_execution_start": True,
            "native_xy_m": NATIVE_XY_M,
            "radius_m": RADIUS_M,
            "native_thresholds_changed": False,
        }
        for item in rows
    ])
    payload = {
        "complete": True, "rows": rows, "selected_margin_m": decision["selected_margin_m"],
        "decision": decision, "identity": identity["files_sha256"], "denominator": count,
    }
    _write_json(output / "calibration.json", payload)
    return payload


def _replay_to_exit(env, layout, actions):
    import torch
    base = env.unwrapped
    exit_xyz = np.asarray(layout and _scene_row(layout)["expert"]["fork_targets"]["transport_exit_xyz"], dtype=np.float64)
    center = np.asarray(_scene_row(layout)["cylinder"]["center_xy"], dtype=np.float64)
    radius = float(_scene_row(layout)["cylinder"]["radius"])
    nearest = None
    nearest_gap = 1e9
    stop = None
    for index, action in enumerate(actions):
        env.step(torch.as_tensor(np.asarray(action, dtype=np.float32), device=base.device))
        tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
        gap = float(np.linalg.norm(tcp[:2] - center) - radius)
        if gap < nearest_gap:
            nearest_gap = gap
            nearest = index + 1
        if float(np.linalg.norm(tcp - exit_xyz)) < 0.015:
            stop = index + 1
            break
    return stop, nearest if nearest_gap <= 0.06 else None, nearest_gap


def _continue_from_snapshot(env, snapshot, *, mode, targets, margin_m, grasp_quat, max_steps=120):
    import torch
    from examples.baselines.constrain.pickcube_expert import restore_execution_state
    from examples.baselines.constrain.placesphere_pilot import scale_native_euler_action
    from scipy.spatial.transform import Rotation

    restore_execution_state(env, snapshot)
    base = env.unwrapped
    snapshot_latch_contact = bool(base._obstacle_collision[0].item())
    base._obstacle_collision[:] = False
    base._forbidden_crossing[:] = False
    base._first_obstacle_contact = None
    done = False
    steps = 0
    contact = False
    crossing = False

    def step(action):
        nonlocal done, steps, contact, crossing
        if done or steps >= max_steps:
            done = True
            return True
        action = np.asarray(action, dtype=np.float32).reshape(7)
        env.step(torch.as_tensor(action, device=base.device))
        steps += 1
        contact = contact or bool(base._obstacle_collision[0].item())
        crossing = crossing or bool(base._forbidden_crossing[0].item())
        done = bool(base.evaluate()["native_success"][0].item())
        return done

    def move(target, target_q=None, grip=-1.0, tolerance=0.006, budget=40):
        target = torch.as_tensor(target, dtype=torch.float32, device=base.device)
        for _ in range(budget):
            if done or steps >= max_steps:
                return False
            delta = target - base.agent.tcp.pose.p[0]
            rot = torch.zeros(3, device=base.device)
            rot_done = True
            if target_q is not None:
                current = base.agent.tcp.pose.q[0].detach().cpu().numpy()
                dq = Rotation.from_quat(np.asarray(target_q)[[1, 2, 3, 0]]) * Rotation.from_quat(current[[1, 2, 3, 0]]).inv()
                rot = torch.as_tensor(
                    scale_native_euler_action(dq.as_euler("XYZ"), base.agent.controller.controllers["arm"].config.rot_lower, 0.35),
                    device=base.device, dtype=torch.float32)
                rot_done = float(np.linalg.norm(dq.as_rotvec())) < 0.03
            if float(torch.linalg.norm(delta)) < tolerance and rot_done:
                return True
            pos = (delta / 0.1).clamp(-0.35, 0.35)
            if step(torch.cat((pos, rot, torch.tensor([grip], device=base.device)))):
                return _native(base)
        return False

    if mode == "legacy":
        move(targets["container_approach_xyz"], grasp_quat, grip=-1.0, budget=60)
        if not done:
            move(targets["retreat_xyz"], grasp_quat, grip=-1.0, budget=30)
        if not done:
            step([0, 0, 0, 0, 0, 0, 1])
            for _ in range(36):
                if step([0, 0, 0, 0, 0, 0, 1]):
                    break
    else:
        trace = {"phase_status": [], "tcp_xyz": [], "step_phase": []}
        execute_above_release(
            step=step, move=move, base=base, grasp_quat=grasp_quat, shared_targets=dict(targets),
            trace=trace, cylinder_top_z=float(base.obstacle_collision_height), margin_m=margin_m,
            is_done=lambda: done, action_count=lambda: steps, bodies_fn=lambda: __import__(
                "examples.baselines.constrain.placesphere_transport_detour", fromlist=["_collision_bodies"]
            )._collision_bodies(base))
    return {
        "native_success": _native(base),
        "obstacle_contact": contact,
        "obstacle_crossing": crossing,
        "snapshot_latch_contact": snapshot_latch_contact,
        "latch_cleared_before_continuation": True,
        "steps": steps,
        "mode": mode,
    }


def run_perturbations(registration_path, output_dir, margin_m, *, episode_wall_s=120):
    import h5py
    from examples.baselines.constrain.pickcube_expert import snapshot_execution_state

    registration, identity = assert_registration(registration_path)
    output = Path(output_dir)
    count = int(registration["selection"]["calibration_count"])
    attempts = []
    for layout in registration["layouts"][:count]:
        env = None
        scene = None
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
        try:
            env, scene, _snapshot = _open_scene(layout)
            with h5py.File(Path(layout["catalog"]) / "scenes.h5", "r") as handle:
                actions = np.asarray(handle[scene["scene_id"]]["expert_actions"])
            stop, nearest, gap = _replay_to_exit(env, layout, actions)
            targets = dict(scene["expert"]["fork_targets"])
            grasp_quat = np.asarray(targets["grasp_quat"], dtype=np.float64)
            center = np.asarray(scene["cylinder"]["center_xy"], dtype=np.float64)
            bin_xy = np.asarray(scene["container_xy"], dtype=np.float64)
            away = bin_xy - center
            away = away / max(float(np.linalg.norm(away)), 1e-9)
            probes = [("pre_terminal", stop)]
            if nearest is not None:
                probes.append(("nearest_approach", nearest))
            for label, index in probes:
                if index is None:
                    attempts.append({"layout": layout, "probe": label, "rejected": True, "reason": "midstate_not_reached"})
                    continue
                env.close()
                env, scene, initial = _open_scene(layout)
                base = env.unwrapped
                import torch
                for action in actions[:index]:
                    env.step(torch.as_tensor(np.asarray(action, dtype=np.float32), device=base.device))
                arrived = snapshot_execution_state(env)
                for distance in PERTURB_DISTANCES_M:
                    for sign, axis in ((1.0, "away_from_cylinder"), (-1.0, "toward_cylinder")):
                        offset = away * sign * float(distance)
                        perturb = bounded_xy_actions(offset)
                        from examples.baselines.constrain.pickcube_expert import restore_execution_state
                        restore_execution_state(env, arrived)
                        for action in perturb:
                            env.step(torch.as_tensor(action, device=base.device))
                        perturbed = snapshot_execution_state(env)
                        legacy = _continue_from_snapshot(
                            env, perturbed, mode="legacy", targets=targets, margin_m=margin_m, grasp_quat=grasp_quat)
                        new = _continue_from_snapshot(
                            env, perturbed, mode="above_release", targets=targets, margin_m=margin_m, grasp_quat=grasp_quat)
                        attempts.append({
                            "layout": _layout_key(layout), "probe": label, "axis": axis,
                            "distance_m": distance, "rejected": False, "legacy": legacy, "new": new,
                            "state_teleport": False,
                        })
            attempts.append({"layout": _layout_key(layout), "nearest_gap_m": gap, "pre_terminal_index": stop})
        except EpisodeDeadline:
            attempts.append({"layout": _layout_key(layout), "rejected": True, "reason": "episode_deadline"})
        except Exception as exc:
            attempts.append({
                "layout": _layout_key(layout), "rejected": True, "reason": "exception",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if env is not None:
                env.close()
    compared = [row for row in attempts if row.get("legacy") and row.get("new")]
    payload = {
        "identity": identity["files_sha256"],
        "margin_m": margin_m,
        "attempt_count": len(attempts),
        "compared_count": len(compared),
        "legacy_contact": sum(bool(row["legacy"]["obstacle_contact"]) for row in compared),
        "new_contact": sum(bool(row["new"]["obstacle_contact"]) for row in compared),
        "legacy_native": sum(bool(row["legacy"]["native_success"]) for row in compared),
        "new_native": sum(bool(row["new"]["native_success"]) for row in compared),
        "new_crossing": sum(bool(row["new"]["obstacle_crossing"]) for row in compared),
        "attempts": attempts,
        "scope": "calibration layouts only; not a broad rate",
    }
    _write_json(output / "perturbations.json", payload)
    return payload


def _frozen_margin(output_dir, margin_m):
    path = Path(output_dir) / "calibration.json"
    if not path.exists():
        raise RuntimeError("confirmation requires the calibration file from this output directory")
    selected = json.loads(path.read_text()).get("selected_margin_m")
    if selected is None or abs(float(selected) - float(margin_m)) > 1e-9:
        raise RuntimeError("confirmation margin does not match the frozen calibration")
    return float(selected)


def run_confirmation(registration_path, output_dir, margin_m, *, episode_wall_s=160, phase_wall_s=2400):
    registration, identity = assert_registration(registration_path)
    output = Path(output_dir)
    frozen = _frozen_margin(output, margin_m)
    count = int(registration["selection"]["calibration_count"])
    layouts = registration["layouts"][count:]
    deadline = time.monotonic() + float(phase_wall_s)
    rows = []
    for layout in layouts:
        if time.monotonic() > deadline:
            decision = confirmation_decision(rows)
            payload = {
                "complete": False, "reason": "phase_wall", "rows": rows,
                "identity": identity["files_sha256"], "collection_ready": False,
                "decision": decision,
            }
            _write_json(output / "confirmation.json", payload)
            _write_json(output / "collection_decision.json", payload)
            return payload
        rows.append(run_role_bundle(layout, margin_m=frozen, episode_wall_s=episode_wall_s))
        _write_json(output / "confirmation_partial.json", {"rows": rows})
    decision = confirmation_decision(rows)
    ready = bool(decision["collection_ready"] and len(rows) == len(layouts))
    payload = {
        "complete": len(rows) == len(layouts),
        "denominator": len(layouts),
        "collection_ready": ready,
        "decision": decision,
        "rows": rows,
        "margin_m": frozen,
        "identity": identity["files_sha256"],
        "dev_scenes_used": False,
        "native_thresholds_changed": False,
        "negative_admits_demos": False,
        "obstacle_only_flag_rejected": True,
    }
    _write_json(output / "confirmation.json", payload)
    _write_json(output / "collection_decision.json", {
        "collection_ready": ready,
        "decision": decision,
        "identity": identity["files_sha256"],
        "margin_m": frozen,
        "denominator": len(layouts),
        "negative_admits_demos": False,
        "source": "confirmation_decision",
    })
    return payload


def dev_layouts():
    manifest = json.loads((Path(DEV_SCENES) / "manifest.json").read_text())
    rows = []
    for index, scene in enumerate(manifest["scenes"]):
        center = np.asarray(scene["cylinder"]["center_xy"], dtype=np.float64)
        goal = np.asarray(scene["container_xy"], dtype=np.float64)
        rows.append({
            "source": "dev30", "index": index, "seed": int(scene["seed"]),
            "scene_id": scene["scene_id"], "height_m": float(scene["cylinder"]["height"]),
            "distance_m": float(np.linalg.norm(center - goal)), "catalog": DEV_SCENES,
        })
    return rows


def run_heldout(registration_path, output_dir, margin_m, *, episode_wall_s=160, phase_wall_s=2200):
    _registration, identity = assert_registration(registration_path)
    output = Path(output_dir)
    frozen = _frozen_margin(output, margin_m)
    layouts = dev_layouts()
    deadline = time.monotonic() + float(phase_wall_s)
    rows = []
    for layout in layouts:
        if time.monotonic() > deadline:
            payload = {
                "complete": False, "reason": "phase_wall", "rows": rows,
                "used_for_selection": False, "used_for_training": False,
                "collection_ready": False, "identity": identity["files_sha256"],
            }
            _write_json(output / "dev30_validation.json", payload)
            return payload
        obstacle = run_layout_expert(layout, margin_m=frozen, role=ROLE_POSITIVE_OBSTACLE, episode_wall_s=episode_wall_s)
        direct = run_layout_expert(
            layout, margin_m=frozen, role=ROLE_POSITIVE_DIRECT,
            shared_targets=obstacle.get("shared_targets"), episode_wall_s=episode_wall_s)
        replay = replay_direct_terminal(layout, direct, episode_wall_s=episode_wall_s)
        rows.append({
            "layout": layout,
            "positive_obstacle": _public_record(obstacle),
            "positive_direct": _public_record(direct),
            "terminal_replay": replay,
            "admission": collection_admission(direct, obstacle, replay, None),
        })
        _write_json(output / "dev30_partial.json", {"rows": rows, "used_for_selection": False})
    decision = confirmation_decision(rows)
    payload = {
        "complete": len(rows) == len(layouts),
        "denominator": len(layouts),
        "ready_count": decision["ready_count"],
        "collection_ready": False,
        "heldout_matches_admission": decision["collection_ready"],
        "used_for_selection": False,
        "used_for_training": False,
        "margin_m": frozen,
        "identity": identity["files_sha256"],
        "rows": rows,
    }
    _write_json(output / "dev30_validation.json", payload)
    return payload


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write-registration", "smoke", "calibrate", "perturb", "confirm", "heldout"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--registration")
    parser.add_argument("--margin", type=float)
    args = parser.parse_args(argv)
    if args.command == "write-registration":
        payload = write_registration(args.output)
        print(payload["identity"]["files_sha256"])
        return
    if not args.registration:
        raise SystemExit("command requires --registration")
    if args.command == "calibrate":
        payload = run_calibration(args.registration, args.output)
        print(json.dumps({
            "complete": payload.get("complete"), "selected_margin_m": payload.get("selected_margin_m"),
            "reason": payload.get("reason"),
        }, sort_keys=True))
        return
    if args.command == "perturb":
        if args.margin is None:
            raise SystemExit("perturb requires --margin")
        payload = run_perturbations(args.registration, args.output, args.margin)
        print(json.dumps({
            "compared": payload["compared_count"], "legacy_contact": payload["legacy_contact"],
            "new_contact": payload["new_contact"], "legacy_native": payload["legacy_native"],
            "new_native": payload["new_native"],
        }, sort_keys=True))
        return
    if args.command == "confirm":
        if args.margin is None:
            raise SystemExit("confirm requires --margin")
        payload = run_confirmation(args.registration, args.output, args.margin)
        print(json.dumps({
            "complete": payload.get("complete"),
            "denominator": payload.get("denominator"),
            "collection_ready": payload.get("collection_ready"),
            "ready_count": (payload.get("decision") or {}).get("ready_count"),
        }, sort_keys=True))
        return
    if args.command == "heldout":
        if args.margin is None:
            raise SystemExit("heldout requires --margin")
        payload = run_heldout(args.registration, args.output, args.margin)
        print(json.dumps({
            "complete": payload.get("complete"),
            "ready_count": payload.get("ready_count"),
            "denominator": payload.get("denominator"),
            "used_for_selection": False,
            "collection_ready": False,
        }, sort_keys=True))
        return
    payload = run_smoke(args.registration, args.output)
    print(json.dumps({
        "collection_ready": False,
        "admission": (payload.get("bundle") or {}).get("admission"),
        "elapsed_s": payload["elapsed_s"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
