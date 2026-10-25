"""Indexed release-contact diagnosis for the frozen PlaceSphere 300/90k EMA.

XY proximity is not 3D closeness. A top-cap label requires the PhysX point to
lie on the cylinder top plane. A body-origin or vertex gap is an approximation.
"""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

POSE_ACTION_CAP = 0.35
CONTROLLER_POS_SCALE_M = 0.1
CONTROLLER_ROT_SCALE_RAD = 0.1
LOCATION_BAND_M = 0.01
NEAR_RADIAL_GAP_M = 0.08
NEAR_BIN_XY_M = 0.05
EXPERT_TCP_CENTER_LEAVE_M = 0.03
RELEASE_FRACTION_MAX = 0.25
RELEASE_ERROR_RATIO_MIN = 1.5
EXECUTION_TRACK_MIN = 5
PREDICTION_FAR_MIN = 5
UPDATES = 90000
BASE_SHA = "2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640"
FROZEN_CKPT = "outputs/ckpt/20261025_085101_placesphere_obstacle300_90k_constrain/final.pt"
FROZEN_CKPT_SHA = "70cf5e6f00bfec8b7d0945f65805fba06bbddd95356778d73009bdc5fed2326b"
FROZEN_EVAL = "outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired/constrain_dp"
FROZEN_REGISTRATION = "outputs/exp/20261025_085101_placesphere_obstacle300_90k/preregistration.json"
OBSTACLE_H5 = "demos/PlaceSphere-transport-detour/20261025_085101_obstacle300/trajectory.h5"
PROBLEM_INDICES = (1, 3, 5, 9, 13, 14, 16, 19, 21)
SAFE_CONTROL_INDICES = (0, 2, 4, 6)
CHANNELS = (
    "delta_x", "delta_y", "delta_z", "delta_euler_x", "delta_euler_y", "delta_euler_z", "gripper",
)
GRASP_OFFSET = 18
TCP_OFFSET = 19
BIN_OFFSET = 26
OBJ_OFFSET = 29
OBSTACLE_OFFSET = 39
EXPECTED_CONTROLLER = {
    "control_mode": "pd_ee_delta_pose", "normalize_action": True, "use_delta": True,
    "use_target": False, "pos_lower": -0.1, "pos_upper": 0.1, "rot_lower": -0.1,
    "rot_upper": 0.1, "gripper_lower": -0.01, "gripper_upper": 0.04,
    "executed_slice": [1, 9], "obs_horizon": 2, "pred_horizon": 16, "act_horizon": 8,
    "seed_rule": "scene_seed",
}


def _xyz(value):
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (3,):
        raise ValueError("expected an xyz point")
    return point


def reconcile_native_failure(*, phase, mechanism, ever_released, terminal_grasped,
                             transported, terminal_on_bin):
    """The coarse incorrect-release bin is not a grasp-no-transport label."""
    mistaken = bool(
        phase == "grasped_no_transport" and mechanism == "incorrect_release"
        and not ever_released and terminal_grasped and not transported)
    if mistaken:
        operative = "grasped_no_transport"
    elif phase == "release_or_stability_failed" and ever_released and not terminal_grasped and not terminal_on_bin:
        operative = "incorrect_release"
    elif phase == "grasped_no_transport":
        operative = "grasped_no_transport"
    else:
        operative = mechanism
    return {
        "operative": operative,
        "published_mechanism": mechanism,
        "published_phase": phase,
        "mistaken_incorrect_release": mistaken,
        "replaces_native_success": False,
    }


def body_origin_gap(position, *, center_xy, radius, top_z):
    point = _xyz(position)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    xy_distance = float(np.linalg.norm(point[:2] - center))
    return {
        "xyz": point.tolist(),
        "xy_distance_m": xy_distance,
        "xy_within_3cm": xy_distance <= 0.03,
        "vertical_gap_m": float(point[2] - float(top_z)),
        "radial_gap_m": float(xy_distance - float(radius)),
        "is_3d_close": False,
        "is_surface_clearance": False,
        "xy_within_3cm_is_3d_close": False,
        "method": "body_origin_vs_cylinder_axis",
    }


def contact_surface_record(hit, *, center_xy, radius, top_z, band_m=LOCATION_BAND_M):
    point = _xyz(hit["position"])
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    radial = float(np.linalg.norm(point[:2] - center))
    vertical = float(point[2] - float(top_z))
    on_plane = abs(vertical) <= float(band_m)
    in_disk = radial <= float(radius) + float(band_m)
    on_side = abs(radial - float(radius)) <= float(band_m) and point[2] <= float(top_z) + float(band_m)
    if on_plane and in_disk:
        location = "top_cap"
    elif vertical > float(band_m) and in_disk:
        location = "above_column"
    elif on_side:
        location = "side"
    else:
        location = "other"
    method = (hit.get("surface") or {}).get("method")
    exact = method == "post_step_physx_contact_with_impulse"
    return {
        "entity": hit.get("entity"),
        "position": point.tolist(),
        "separation_m": None if hit.get("separation_m") is None else float(hit["separation_m"]),
        "impulse": None if hit.get("impulse") is None else float(hit["impulse"]),
        "vertical_gap_m": vertical,
        "radial_gap_m": radial - float(radius),
        "location": location,
        "location_rule": "top plane band and radial disk; height above the plane is not a cap",
        "is_surface_clearance": bool(exact and (hit.get("surface") or {}).get("is_surface_clearance") is True),
        "is_3d_close": False,
        "xy_within_3cm_is_3d_close": False,
        "method": method,
    }


def action_semantics(action):
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (7,):
        raise ValueError("PlaceSphere action is 7-wide")
    grip = float(values[6])
    if grip > 0:
        sign = "open"
    elif grip < 0:
        sign = "close"
    else:
        sign = "hold"
    return {
        "channels": list(CHANNELS),
        "values": values.tolist(),
        "xyz": values[:3].tolist(),
        "rotation_xyz": values[3:6].tolist(),
        "grip": grip,
        "grip_sign": sign,
        "rotation_is_meters": False,
        "translation_is_already_meters": False,
        "pose_action_cap": POSE_ACTION_CAP,
        "controller_pos_scale_m": CONTROLLER_POS_SCALE_M,
        "controller_rot_scale_rad": CONTROLLER_ROT_SCALE_RAD,
        "effective_translation_m": (values[:3] * CONTROLLER_POS_SCALE_M).tolist(),
        "effective_rotation_rad": (values[3:6] * CONTROLLER_ROT_SCALE_RAD).tolist(),
    }


def _window_phase(obs, actions, start):
    obs = np.asarray(obs, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 42 or actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("window labeling expects (T+1, 42) observations and (T, 7) actions")
    start = int(start)
    latest = int(np.clip(start + 1, 0, len(obs) - 1))
    span = obs[max(0, start):min(len(obs), start + 16)]
    real = actions[max(0, start):min(len(actions), start + 16)]
    grasped = bool(span.shape[0] and np.any(span[:, GRASP_OFFSET] > 0.5))
    opening = bool(real.shape[0] and np.any(real[:, 6] > 0))
    if grasped and opening:
        return "release"
    if obs[latest, GRASP_OFFSET] > 0.5:
        distance = float(np.linalg.norm(obs[latest, OBJ_OFFSET:OBJ_OFFSET + 2] - obs[latest, BIN_OFFSET:BIN_OFFSET + 2]))
        return "held_near_bin" if distance <= NEAR_BIN_XY_M else "transport"
    return "approach"


def label_training_window(obs, actions, *, start, episode_id):
    obs = np.asarray(obs, dtype=np.float64)
    latest = int(np.clip(int(start) + 1, 0, len(obs) - 1))
    tcp = obs[latest, TCP_OFFSET:TCP_OFFSET + 2]
    obstacle = obs[latest, OBSTACLE_OFFSET:OBSTACLE_OFFSET + 3]
    radial = float(np.linalg.norm(tcp - obstacle[:2]) - obstacle[2])
    return {
        "episode_id": int(episode_id),
        "window_start": int(start),
        "phase": _window_phase(obs, actions, start),
        "near": radial < NEAR_RADIAL_GAP_M,
        "radial_gap_m": radial,
        "far_near_basis": "tcp_center_radial_gap",
        "height_in_observation": False,
        "is_3d_close": False,
        "is_surface_clearance": False,
        "xy_within_3cm_is_3d_close": False,
    }


def phase_inverse_weights(counts):
    counts = {str(key): int(value) for key, value in counts.items()}
    if not counts or any(value <= 0 for value in counts.values()):
        raise ValueError("phase weights need positive recorded counts")
    total = sum(counts.values())
    phases = len(counts)
    return {key: total / (phases * value) for key, value in counts.items()}


def vertex_column_gap(vertices, *, center_xy, radius, top_z):
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    inside = np.linalg.norm(points[:, :2] - center, axis=1) <= float(radius)
    chosen = points[inside]
    return {
        "vertices_in_column": int(chosen.shape[0]),
        "min_vertical_gap_m": None if chosen.size == 0 else float(np.min(chosen[:, 2]) - float(top_z)),
        "is_surface_clearance": False,
        "is_3d_close": False,
        "xy_within_3cm_is_3d_close": False,
        "method": "collision_vertices_inside_cylinder_column",
    }


def perturbation_required(*, separation_m, margin_inferred):
    if separation_m is not None and float(separation_m) < 0:
        return False
    return bool(margin_inferred)


def contact_vs_expert(tcp_xyz, expert_release_tcp, *, grip_opening, expert_physx):
    if expert_release_tcp is None:
        return {
            "class": "uncompared", "tcp_center_distance_m": None, "is_3d_close": False,
            "is_surface_clearance": False, "xy_within_3cm_is_3d_close": False,
            "basis": "tcp_center_distance_not_surface", "leave_m": EXPERT_TCP_CENTER_LEAVE_M,
        }
    distance = float(np.linalg.norm(_xyz(tcp_xyz) - _xyz(expert_release_tcp)))
    if distance < EXPERT_TCP_CENTER_LEAVE_M and grip_opening and expert_physx is False:
        kind = "execution_track"
    elif distance >= EXPERT_TCP_CENTER_LEAVE_M:
        kind = "prediction_far"
    else:
        kind = "not_isolated"
    return {
        "class": kind, "tcp_center_distance_m": distance, "is_3d_close": False,
        "is_surface_clearance": False, "xy_within_3cm_is_3d_close": False,
        "basis": "tcp_center_distance_not_surface", "leave_m": EXPERT_TCP_CENTER_LEAVE_M,
    }


def controller_contract(config):
    mismatch = None
    for key, expected in EXPECTED_CONTROLLER.items():
        if config.get(key) != expected:
            mismatch = key
            break
    normalized = config.get("normalize_action") is True and config.get("rot_lower") == -0.1
    return {
        "matches_expert_contract": mismatch is None,
        "mismatch": mismatch,
        "pose_0.35_is_radians": False,
        "rotation_action_is_normalized_then_scaled_by_rot_lower": normalized,
        "executed_slice": list(EXPECTED_CONTROLLER["executed_slice"]),
        "seed_rule": EXPECTED_CONTROLLER["seed_rule"],
        "horizons": {"obs": 2, "pred": 16, "act": 8},
    }


def decide_release_factor(evidence):
    decision = {
        "trial": False, "factor": None, "stride": None, "stride_swept": False,
        "architecture_change": False, "action_shield": False, "new_data": False,
        "updates": UPDATES, "base_checkpoint_sha256": BASE_SHA, "same_checkpoint": False,
        "weights": None, "reason": None, "budget_seconds": 2500, "outer_timeout_seconds": 6000,
    }
    if (evidence.get("reproduction_mismatch") or evidence.get("expert_physx_contact_count")
            or not evidence.get("window_index_saved")):
        decision["reason"] = "evidence_incomplete"
        return decision
    labels_ok = (
        evidence.get("failure_16_operative") == "grasped_no_transport"
        and evidence.get("failure_16_mistaken_incorrect_release") is True
        and evidence.get("failure_13_operative") == "incorrect_release"
        and evidence.get("failure_14_operative") == "incorrect_release")
    if not labels_ok or int(evidence.get("contact_count", 0)) != 6:
        decision["reason"] = "failure_phase_unverified"
        return decision
    execution = int(evidence["execution_track_contacts"])
    prediction = int(evidence["prediction_far_contacts"])
    mismatch = evidence.get("controller_mismatch")
    if execution >= EXECUTION_TRACK_MIN and mismatch:
        decision.update(trial=True, factor=mismatch, same_checkpoint=True, reason="named_controller_mismatch")
        return decision
    if execution >= EXECUTION_TRACK_MIN:
        decision["reason"] = "execution_tracking_not_a_named_factor"
        return decision
    if evidence.get("release_epsilon") is None or evidence.get("other_epsilon") in (None, 0):
        decision["reason"] = "release_supervision_not_rare_or_not_worse"
        return decision
    other = float(evidence["other_epsilon"])
    ratio = float(evidence["release_epsilon"]) / other
    rare = float(evidence["release_window_fraction"]) <= RELEASE_FRACTION_MAX and ratio >= RELEASE_ERROR_RATIO_MIN
    if rare and prediction >= PREDICTION_FAR_MIN and execution < EXECUTION_TRACK_MIN:
        weights = evidence.get("phase_weights") or {}
        if "release" not in weights:
            decision["reason"] = "weights_missing"
            return decision
        decision.update(
            trial=True, factor="phase_inverse_frequency", weights=dict(weights),
            reason="rare_worse_release_supervision")
        return decision
    decision["reason"] = "release_supervision_not_rare_or_not_worse"
    return decision


def _sha_file(path):
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
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def diagnosis_source_identity(repo_root=None):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = (
        "examples/baselines/constrain/placesphere_release_diagnosis.py",
        "examples/baselines/constrain/placesphere_learning_pilot.py",
        "examples/baselines/constrain/placesphere_constrain.py",
        "examples/baselines/constrain/placesphere_transport_detour.py",
        "examples/baselines/constrain/diagnose_placesphere_failure.py",
        "examples/baselines/constrain/evaluate_placesphere_paired.py",
    )
    hashed = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in files}
    return {
        "basis": "working-tree bytes frozen before the release diagnosis",
        "git_head_is_code_identity": False,
        "files": hashed,
        "files_sha256": hashlib.sha256(
            json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def _live_controller(env):
    base = env.unwrapped
    arm = base.agent.controller.controllers["arm"].config
    grip = base.agent.controller.controllers["gripper"].config
    return {
        "control_mode": "pd_ee_delta_pose",
        "normalize_action": bool(arm.normalize_action),
        "use_delta": bool(arm.use_delta),
        "use_target": bool(arm.use_target),
        "pos_lower": float(np.broadcast_to(arm.pos_lower, 3)[0]),
        "pos_upper": float(np.broadcast_to(arm.pos_upper, 3)[0]),
        "rot_lower": float(np.broadcast_to(arm.rot_lower, 3)[0]),
        "rot_upper": float(np.broadcast_to(arm.rot_upper, 3)[0]),
        "gripper_lower": float(np.broadcast_to(grip.lower, 1)[0]) if np.ndim(grip.lower) else float(grip.lower),
        "gripper_upper": float(np.broadcast_to(grip.upper, 1)[0]) if np.ndim(grip.upper) else float(grip.upper),
        "ee_link": str(arm.ee_link),
        "executed_slice": [1, 9],
        "obs_horizon": 2,
        "pred_horizon": 16,
        "act_horizon": 8,
        "seed_rule": "scene_seed",
        "snapshot_empty_is_not_a_scale": True,
    }


def _body_records(base, center, radius, top_z):
    rows = []
    tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
    rows.append({"name": "tcp", "role": "tcp", **body_origin_gap(tcp, center_xy=center, radius=radius, top_z=top_z)})
    for link in base.agent.robot.links:
        name = link.name
        if name == "panda_hand":
            role = "hand"
        elif name == "panda_link8":
            role = "flange"
        elif "finger" in name:
            role = "finger"
        elif name == "panda_hand_tcp":
            role = "tcp_link"
        else:
            role = "link"
        origin = link.pose.p[0].detach().cpu().numpy()
        rows.append({"name": name, "role": role, **body_origin_gap(origin, center_xy=center, radius=radius, top_z=top_z)})
    sphere = base.obj.pose.p[0].detach().cpu().numpy()
    rows.append({"name": "sphere", "role": "sphere", **body_origin_gap(sphere, center_xy=center, radius=radius, top_z=top_z)})
    return rows


def _vertex_records(env, center, radius, top_z):
    from examples.baselines.constrain.pickcube_expert import _shape_world_vertices

    wanted = {"panda_hand", "panda_link8", "panda_leftfinger", "panda_rightfinger", "panda_hand_tcp"}
    rows = []
    for link in env.unwrapped.agent.robot.links:
        if link.name not in wanted:
            continue
        points = []
        for shape in link._bodies[0].collision_shapes:
            try:
                points.append(_shape_world_vertices(link, shape))
            except Exception as exc:
                rows.append({"name": link.name, "error": f"{type(exc).__name__}: {exc}",
                             "is_surface_clearance": False, "is_3d_close": False})
        if points:
            rows.append({"name": link.name, **vertex_column_gap(
                np.concatenate(points, axis=0), center_xy=center, radius=radius, top_z=top_z)})
    return rows


def _load_expert_actions(scene_set, index):
    import h5py

    row = scene_set.manifest["scenes"][index]
    with h5py.File(scene_set.directory / scene_set.manifest["scenes_file"], "r") as handle:
        return np.asarray(handle[row["scene_id"]]["expert_actions"])


def instrument_episode(adapter, scene_set, index, device, max_steps=200):
    import torch
    from examples.baselines.constrain.diagnose_placesphere_failure import (
        _obstacle_contact_points, _terminal_pose, classify_contact_phase, diagnostic_lifted,
        diagnostic_near_goal, read_obstacle_contact,
    )
    from examples.baselines.constrain.evaluate_placesphere_paired import classify_phase, classify_termination
    from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter
    from examples.baselines.constrain.placesphere_learning_pilot import RAW_OBS_DIM, _bool_info, coerce_action_sequence
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, _obs_vector, make_env

    scene = scene_set.load(index)
    geometry = scene["metadata"]["cylinder"]
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=max_steps, obstacle_radius=geometry["radius"], obstacle_height=geometry["height"]))
    try:
        env, restored = scene_set.restore(index, env)
        if not restored.get("bit_exact"):
            raise ValueError(f"restored state is not bit-exact: {restored.get('norms')}")
        base = env.unwrapped
        controller = _live_controller(env)
        controller["saved_snapshot_empty"] = scene["controller"] == {}
        info = base.evaluate()
        obs = np.asarray(_obs_vector(base.get_obs(info=info, unflattened=True)), dtype=np.float32).reshape(-1)
        if obs.shape != (RAW_OBS_DIM,) or not np.array_equal(obs, scene["raw_observation"].reshape(-1)):
            raise ValueError("restored raw observation mismatch")
        history = np.repeat(obs[None, :], 2, axis=0)
        seed = int(scene["metadata"]["seed"])
        torch.manual_seed(seed)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        center = np.asarray(geometry["center_xy"], dtype=np.float64)
        radius = float(geometry["radius"])
        top_z = float(geometry["height"])
        initial_z = float(base.obj.pose.p[0, 2].detach().cpu())
        goal_xy = base.bin.pose.p[0, :2].detach().cpu().numpy()
        elapsed = 0
        collision = crossing = grasped = lifted = transported = released = False
        first_contact = None
        release_step = None
        native_success = terminated = truncated = False
        termination = None
        info = {}
        replans = []
        min_bin_xy = None
        contact_geometry = None
        release_geometry = None
        while elapsed < max_steps:
            prediction = coerce_action_sequence(adapter.get_action(history))
            mapped = PlaceSphereBaseDPAdapter.map_action_ee_delta_pose(prediction)
            raw = prediction[0].detach().cpu().numpy()
            native = mapped[0].detach().cpu().numpy()
            if native.shape != (8, 7):
                raise ValueError(f"expected queued (8, 7), got {native.shape}")
            replans.append({
                "step": elapsed,
                "queue": native.tolist(),
                "pre_map_pose_outside_cap": bool(np.max(np.abs(raw[:, :6])) > POSE_ACTION_CAP + 1e-6),
                "semantics": [action_semantics(row) for row in native],
            })
            for queue_index, action in enumerate(native[:min(8, max_steps - elapsed)]):
                _obs, _, step_terminated, step_truncated, info = env.step(action[None])
                elapsed += 1
                terminated = bool(terminated or step_terminated)
                truncated = bool(truncated or step_truncated)
                step_success = _bool_info(info, "native_success")
                native_success = bool(native_success or step_success)
                collision_before = collision
                step_collision = _bool_info(info, "obstacle_collision")
                collision = bool(collision or step_collision)
                crossing = bool(crossing or _bool_info(info, "obstacle_crossing"))
                now_grasped = _bool_info(info, "is_obj_grasped")
                grasped = bool(grasped or now_grasped)
                just_released = bool(grasped and not now_grasped)
                released = bool(released or just_released)
                sphere = base.obj.pose.p[0].detach().cpu().numpy()
                bin_xy = float(np.linalg.norm(sphere[:2] - goal_xy))
                min_bin_xy = bin_xy if min_bin_xy is None else min(min_bin_xy, bin_xy)
                lifted_now = diagnostic_lifted(sphere[2], initial_z)
                near_goal = diagnostic_near_goal(sphere[:2], goal_xy)
                lifted = bool(lifted or lifted_now)
                transported = bool(transported or near_goal)
                phase = classify_contact_phase(
                    grasped=now_grasped, lifted=lifted_now, near_goal=near_goal,
                    released=just_released, native_success=step_success)
                if first_contact is None and step_collision and not collision_before:
                    hit = read_obstacle_contact(env)
                    hits = _obstacle_contact_points(env, require_impulse=True)
                    surface = None if hit is None else contact_surface_record(
                        hit, center_xy=center, radius=radius, top_z=top_z)
                    first_contact = {
                        "step": elapsed, "queue_index": queue_index, "replan_step": replans[-1]["step"],
                        "phase": phase, "action": action_semantics(action),
                        "surface": surface, "entities": [item["entity"] for item in hits],
                        "contact_unreadable": hit is None,
                    }
                    contact_geometry = {
                        "bodies": _body_records(base, center, radius, top_z),
                        "vertices": _vertex_records(env, center, radius, top_z),
                        "margin_inferred": False,
                        "perturbation_ran": perturbation_required(
                            separation_m=None if surface is None else surface["separation_m"], margin_inferred=False),
                    }
                if release_step is None and just_released:
                    release_step = elapsed
                    release_geometry = {
                        "step": elapsed, "action": action_semantics(action),
                        "bodies": _body_records(base, center, radius, top_z),
                        "on_bin": _bool_info(info, "is_obj_on_bin"),
                    }
                obs = np.asarray(_obs.detach().cpu() if torch.is_tensor(_obs) else _obs, dtype=np.float32).reshape(-1)
                history = np.concatenate((history[1:], obs[None]), axis=0)
                termination = classify_termination(step_success, step_terminated, step_truncated, elapsed, max_steps)
                if termination is not None:
                    break
            if termination is not None:
                break
        if termination is None:
            termination = classify_termination(native_success, terminated, truncated, elapsed, max_steps)
        if not info:
            info = base.evaluate()
        terminal = _terminal_pose(base, info, elapsed)
        terminal_grasped = _bool_info(info, "is_obj_grasped")
        terminal_on_bin = _bool_info(info, "is_obj_on_bin")
        terminal_static = _bool_info(info, "is_obj_static")
        from examples.baselines.constrain.diagnose_placesphere_failure import classify_terminal_mechanism
        mechanism = classify_terminal_mechanism(
            native_success=termination == "native_success", ever_released=released,
            release_on_bin=None if release_geometry is None else release_geometry["on_bin"],
            terminal_on_bin=terminal_on_bin, terminal_static=terminal_static,
            terminal_grasped=terminal_grasped, termination_reason=termination,
            elapsed_steps=elapsed, max_steps=max_steps)
        phase_name = classify_phase(grasped, lifted, transported, released, termination == "native_success")
        reconciled = reconcile_native_failure(
            phase=phase_name, mechanism=mechanism.get("mechanism"), ever_released=released,
            terminal_grasped=terminal_grasped, transported=transported, terminal_on_bin=terminal_on_bin)
        return {
            "scene_index": index, "scene_id": scene["metadata"]["scene_id"], "seed": seed,
            "status": "evaluated", "restored_bit_exact": True,
            "checkpoint_sha256": adapter.metadata["checkpoint_sha256"],
            "controller": controller_contract(controller) | {"live": controller},
            "cylinder": {"center_xy": center.tolist(), "radius_m": radius, "top_z_m": top_z,
                         "height_source": geometry.get("height_source"),
                         "geometric_top_cover_is_surface_clearance": geometry.get("geometric_top_cover_is_surface_clearance")},
            "first_contact": first_contact, "contact_geometry": contact_geometry,
            "release_geometry": release_geometry, "replans": replans,
            "min_sphere_bin_xy_m": min_bin_xy, "terminal": terminal,
            "terminal_mechanism": mechanism, "reconciled_failure": reconciled,
            "result": {
                "termination_reason": termination, "elapsed_steps": elapsed,
                "native_success": termination == "native_success",
                "safe_success": bool(termination == "native_success" and not collision and not crossing),
                "obstacle_contact": collision, "xy_forbidden_crossing": crossing,
                "terminal_grasped": terminal_grasped, "terminal_on_bin": terminal_on_bin,
                "terminal_static": terminal_static, "ever_released": released,
                "transported": transported, "phase": phase_name,
                "linear_speed_m_s": terminal.get("linear_speed_m_s"),
            },
        }
    finally:
        env.close()


def replay_expert(scene_set, index):
    import torch
    from examples.baselines.constrain.diagnose_placesphere_failure import read_obstacle_contact
    from examples.baselines.constrain.placesphere_learning_pilot import _bool_info
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    scene = scene_set.load(index)
    geometry = scene["metadata"]["cylinder"]
    actions = _load_expert_actions(scene_set, index)
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=max(len(actions), 1), obstacle_radius=geometry["radius"], obstacle_height=geometry["height"]))
    try:
        scene_set.restore(index, env)
        base = env.unwrapped
        center = np.asarray(geometry["center_xy"], dtype=np.float64)
        radius = float(geometry["radius"])
        top_z = float(geometry["height"])
        grasped = False
        release = None
        physx = False
        for step, action in enumerate(actions, start=1):
            action = np.asarray(action, dtype=np.float32).reshape(7)
            _, _, _, _, info = env.step(action[None])
            now = _bool_info(info, "is_obj_grasped")
            grasped = grasped or now
            hit = read_obstacle_contact(env) if _bool_info(info, "obstacle_collision") else None
            physx = physx or hit is not None
            if release is None and grasped and float(action[6]) > 0:
                tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
                release = {
                    "step": step, "tcp_xyz": tcp.astype(float).tolist(),
                    "action": action_semantics(action),
                    "bodies": _body_records(base, center, radius, top_z),
                    "physx": None if hit is None else contact_surface_record(
                        hit, center_xy=center, radius=radius, top_z=top_z),
                }
            if _bool_info(info, "native_success"):
                break
        return {"physx_contact": physx, "release": release, "steps": step if actions.size else 0}
    finally:
        env.close()


def _published(index):
    return json.loads(Path(FROZEN_EVAL, f"episode_{index:03d}.json").read_text())


def reproduction_mismatch(traced, published):
    contact = traced.get("first_contact") or {}
    published_contact = published.get("first_contact") or {}
    result = traced["result"]
    published_result = published["result"]
    checks = {
        "native_success": result["native_success"] == published_result["native_success"],
        "obstacle_contact": result["obstacle_contact"] == published_result["obstacle_contact"],
        "elapsed_steps": result["elapsed_steps"] == published_result["elapsed_steps"],
        "phase": result["phase"] == published_result["phase"],
        "ever_released": result["ever_released"] == published_result["ever_released"],
        "terminal_grasped": result["terminal_grasped"] == published_result["terminal_grasped"],
        "contact_step": contact.get("step") == published_contact.get("step"),
    }
    return {"mismatch": not all(checks.values()), "checks": checks}


def score_windows(adapter, h5_path, episode_ids, device):
    import h5py
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from examples.baselines.constrain.placesphere_constrain import placesphere_hint
    from examples.baselines.constrain.placesphere_learning_pilot import POLICY_OBS_DIM, _window_slices

    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026102516)
    rows = []
    batch_obs, batch_act, batch_meta = [], [], []

    def flush():
        if not batch_obs:
            return
        obs = torch.stack(batch_obs).to(device)
        actions = torch.stack(batch_act).to(device)
        noise = torch.randn(actions.shape, generator=generator).to(device)
        timesteps = torch.full((len(actions),), 50, device=device, dtype=torch.long)
        noisy = scheduler.add_noise(actions, noise, timesteps)
        hint = placesphere_hint(obs)[:, None, :].expand(-1, actions.shape[1], -1)
        with torch.no_grad():
            prediction = adapter.denoiser(
                noisy, timesteps, global_cond=obs[..., :POLICY_OBS_DIM].flatten(1), hint=hint, enabled=True)
        error = (prediction - noise).pow(2).flatten(1).mean(1).detach().cpu().numpy()
        for meta, value in zip(batch_meta, error):
            meta["epsilon"] = float(value)
            rows.append(meta)
        batch_obs.clear()
        batch_act.clear()
        batch_meta.clear()

    with h5py.File(h5_path, "r") as handle:
        for episode_id in episode_ids:
            group = handle[f"traj_{int(episode_id)}"]
            obs = np.asarray(group["obs"])
            actions = np.asarray(group["actions"])
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
            tensor_act = torch.as_tensor(actions, dtype=torch.float32)
            length = len(actions)
            for start, end in _window_slices(length):
                label = label_training_window(obs, actions, start=start, episode_id=episode_id)
                obs_seq = tensor_obs[max(0, start):start + 2]
                act_seq = tensor_act[max(0, start):end]
                if start < 0:
                    obs_seq = torch.cat((obs_seq[:1].repeat(-start, 1), obs_seq))
                    act_seq = torch.cat((act_seq[:1].repeat(-start, 1), act_seq))
                if end > length:
                    pad = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=act_seq.dtype)
                    act_seq = torch.cat((act_seq, pad.repeat(end - length, 1)))
                batch_obs.append(obs_seq)
                batch_act.append(act_seq)
                batch_meta.append(label)
                if len(batch_obs) == 256:
                    flush()
        flush()
    return rows


def _group_error(rows, predicate):
    chosen = [row["epsilon"] for row in rows if predicate(row)]
    if not chosen:
        return {"count": 0, "mean_epsilon": None}
    return {"count": len(chosen), "mean_epsilon": float(np.mean(chosen))}


def assemble_evidence(episodes, windows):
    contacts = [row for row in episodes if row["result"]["obstacle_contact"]]
    failures = {row["scene_index"]: row for row in episodes if not row["result"]["native_success"]}
    counts = {}
    for row in windows:
        counts[row["phase"]] = counts.get(row["phase"], 0) + 1
    weights = phase_inverse_weights(counts) if counts and all(value > 0 for value in counts.values()) else {}
    release = _group_error(windows, lambda row: row["phase"] == "release")
    other = _group_error(windows, lambda row: row["phase"] != "release")
    near = _group_error(windows, lambda row: row["phase"] == "release" and row["near"])
    far = _group_error(windows, lambda row: row["phase"] == "release" and not row["near"])
    return {
        "reproduction_mismatch": any(row["reproduction"]["mismatch"] for row in episodes),
        "expert_physx_contact_count": sum(bool(row["expert"]["physx_contact"]) for row in episodes),
        "window_index_saved": True,
        "release_window_fraction": 0.0 if not windows else counts.get("release", 0) / len(windows),
        "release_epsilon": release["mean_epsilon"],
        "other_epsilon": other["mean_epsilon"],
        "release_near_epsilon": near["mean_epsilon"],
        "release_far_epsilon": far["mean_epsilon"],
        "prediction_far_contacts": sum(
            (row.get("expert_comparison") or {}).get("class") == "prediction_far" for row in contacts),
        "execution_track_contacts": sum(
            (row.get("expert_comparison") or {}).get("class") == "execution_track" for row in contacts),
        "contact_count": len(contacts),
        "failure_13_operative": failures[13]["reconciled_failure"]["operative"],
        "failure_14_operative": failures[14]["reconciled_failure"]["operative"],
        "failure_16_operative": failures[16]["reconciled_failure"]["operative"],
        "failure_16_mistaken_incorrect_release": failures[16]["reconciled_failure"]["mistaken_incorrect_release"],
        "phase_counts": counts,
        "phase_weights": weights,
        "controller_mismatch": next(
            (row["controller"]["mismatch"] for row in episodes if row["controller"]["mismatch"]), None),
        "window_count": len(windows),
    }


def run_diagnosis(output_dir, device="cuda"):
    import torch
    from examples.baselines.constrain.diagnose_placesphere_failure import _child_pids, _terminate_new_children
    from examples.baselines.constrain.placesphere_learning_pilot import (
        PilotConstrainAdapter, load_preregistration,
    )
    from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    identity = diagnosis_source_identity()
    registration = load_preregistration(FROZEN_REGISTRATION)
    if _sha_file(FROZEN_CKPT) != FROZEN_CKPT_SHA:
        raise RuntimeError("frozen checkpoint sha changed")
    protocol = {
        "kind": "placesphere_release_diagnosis",
        "checkpoint": FROZEN_CKPT,
        "checkpoint_sha256": FROZEN_CKPT_SHA,
        "problem_indices": list(PROBLEM_INDICES),
        "safe_control_indices": list(SAFE_CONTROL_INDICES),
        "safe_control_rule": "first four published safe successes, frozen before geometry",
        "criteria": {
            "release_fraction_max": RELEASE_FRACTION_MAX,
            "release_error_ratio_min": RELEASE_ERROR_RATIO_MIN,
            "expert_tcp_center_leave_m": EXPERT_TCP_CENTER_LEAVE_M,
            "leave_is_not_surface_clearance": True,
            "xy_within_3cm_is_3d_close": False,
            "no_stride_sweep": True,
        },
        "source_identity": identity,
        "trial_started": False,
    }
    _write_json(output / "preregistration.json", protocol)
    catalog = TransportDetourSceneSet(registration["collection"]["dev_scenes"])
    adapter = PilotConstrainAdapter(FROZEN_CKPT, registration, device)
    if adapter.metadata["checkpoint_sha256"] != FROZEN_CKPT_SHA:
        raise RuntimeError("loaded checkpoint sha mismatch")
    indices = list(PROBLEM_INDICES) + list(SAFE_CONTROL_INDICES)
    episodes = []
    for index in indices:
        before = set(_child_pids())
        previous = signal.signal(signal.SIGALRM, _deadline)
        signal.setitimer(signal.ITIMER_REAL, 120)
        try:
            traced = instrument_episode(adapter, catalog, index, device)
        except TimeoutError as exc:
            traced = {"scene_index": index, "scene_id": None, "status": "runtime_error",
                      "error": f"{type(exc).__name__}: {exc}",
                      "result": {"obstacle_contact": False, "native_success": False, "phase": None},
                      "reproduction": {"mismatch": True, "checks": {}},
                      "expert": {"physx_contact": True, "release": None},
                      "expert_comparison": {"class": "uncompared"},
                      "reconciled_failure": {}, "controller": {"mismatch": "runtime_error"}}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            _terminate_new_children(before)
        if traced.get("status") == "evaluated":
            signal.setitimer(signal.ITIMER_REAL, 120)
            try:
                expert = replay_expert(catalog, index)
            except Exception as exc:
                expert = {"physx_contact": True, "error": f"{type(exc).__name__}: {exc}", "release": None}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                _terminate_new_children(before)
        else:
            expert = traced["expert"]
        if traced.get("status") != "evaluated":
            _write_json(output / f"episode_{index:03d}.json", traced)
            episodes.append(traced)
            print(f"scene {index} status={traced.get('status')} error={traced.get('error')}", flush=True)
            continue
        published = _published(index)
        contact = traced.get("first_contact") or {}
        surface = contact.get("surface") or {}
        expert_tcp = None if not expert.get("release") else expert["release"]["tcp_xyz"]
        tcp = None
        if traced.get("contact_geometry"):
            tcp = next(row["xyz"] for row in traced["contact_geometry"]["bodies"] if row["role"] == "tcp")
        comparison = contact_vs_expert(
            tcp or [0, 0, 0], expert_tcp,
            grip_opening=bool((contact.get("action") or {}).get("grip_sign") == "open"),
            expert_physx=bool(expert["physx_contact"])) if tcp is not None else {
                "class": "no_contact", "is_3d_close": False, "is_surface_clearance": False}
        if surface.get("location") != "top_cap" or surface.get("entity") != "panda_hand":
            if comparison.get("class") in ("prediction_far", "execution_track"):
                comparison = {**comparison, "class": "other_contact"}
        traced["expert"] = expert
        traced["expert_comparison"] = comparison
        traced["reproduction"] = reproduction_mismatch(traced, published)
        traced["published_checkpoint_sha256"] = published.get("checkpoint_sha256")
        _write_json(output / f"episode_{index:03d}.json", traced)
        episodes.append(traced)
        print(f"scene {index} contact={traced['result']['obstacle_contact']} "
              f"phase={traced['result']['phase']} mismatch={traced['reproduction']['mismatch']}", flush=True)
    windows = score_windows(adapter, OBSTACLE_H5, range(1000, 1300), device)
    index_path = output / "window_index.jsonl"
    with index_path.open("w") as stream:
        for row in windows:
            stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    evidence = assemble_evidence(episodes, windows)
    evidence["window_index"] = str(index_path)
    decision = decide_release_factor(evidence)
    _write_json(output / "evidence.json", evidence)
    _write_json(output / "decision.json", decision)
    _write_json(output / "index.json", {
        "episodes": [{"scene_index": row["scene_index"], "scene_id": row["scene_id"],
                      "role": "problem" if row["scene_index"] in PROBLEM_INDICES else "safe_control",
                      "reproduction_mismatch": row["reproduction"]["mismatch"],
                      "expert_comparison": row["expert_comparison"],
                      "reconciled_failure": row["reconciled_failure"]} for row in episodes],
        "evidence": evidence, "decision": decision, "source_identity": identity,
        "checkpoint_sha256": FROZEN_CKPT_SHA,
    })
    return {"evidence": evidence, "decision": decision}


def _deadline(_signum, _frame):
    raise TimeoutError("episode wall exceeded")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = run_diagnosis(args.output_dir, args.device)
    print(json.dumps({"decision": result["decision"], "elapsed_s": time.monotonic() - started}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
