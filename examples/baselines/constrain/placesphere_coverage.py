"""Offline coverage of PlaceSphere contact states against the 300 expert windows.

XY distance is not 3D closeness. The near bar is a train leave-one-scene-out
quantile, not a reward and not an arbitrary meter threshold.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

CONTACT_SCENE_INDICES = (1, 3, 5, 9, 19, 21)
SAFE_CONTROL_INDICES = (0, 2, 4, 6)
TRAIN_OBSTACLE_ID_MIN = 1000
TRAIN_OBSTACLE_ID_MAX = 1299
RADIUS_QUANTILE = 0.95
NEAR_BIN_XY_M = 0.05
POLICY_DIM = 39
RAW_DIM = 42
LABELER_FILES = (
    "examples/baselines/constrain/placesphere_coverage.py",
    "examples/baselines/constrain/placesphere_release_diagnosis.py",
)
SCALE_NAMES = (
    "position_3d", "object_position_3d", "rotation", "proprio_qpos", "proprio_qvel", "normalized_hint",
)
HINT_SCALE = np.array([0.5, 0.5, 0.08], dtype=np.float64)


def freeze_labeler_identity(repo_root=None):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    hashed = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in LABELER_FILES}
    return {
        "basis": "labeler file list frozen at this command",
        "git_head_is_code_identity": False,
        "files": hashed,
        "files_sha256": hashlib.sha256(
            json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def traces_have_full_observations(episodes):
    if not episodes:
        return False
    for row in episodes:
        window = row.get("decision_window") or {}
        if window.get("frames") != 2 or window.get("policy_condition_shape") != [2, 39]:
            return False
        if window.get("latest_state_index") != 1 or window.get("future_frames_used") != 0:
            return False
    return True


def quat_geodesic(left, right):
    first = np.asarray(left, dtype=np.float64).reshape(4)
    second = np.asarray(right, dtype=np.float64).reshape(4)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _hint(latest):
    tcp = np.asarray(latest[19:21], dtype=np.float64)
    obstacle = np.asarray(latest[39:42], dtype=np.float64)
    raw = np.array([obstacle[0] - tcp[0], obstacle[1] - tcp[1], obstacle[2]], dtype=np.float64)
    return np.clip(raw / HINT_SCALE, -1.0, 1.0)


def state_phase(latest):
    latest = np.asarray(latest, dtype=np.float64).reshape(-1)
    if float(latest[18]) <= 0.5:
        return "approach"
    distance = float(np.linalg.norm(latest[29:31] - latest[26:28]))
    if distance <= NEAR_BIN_XY_M:
        return "held_near_bin"
    return "transport"


def policy_phase(latest, action):
    action = np.asarray(action, dtype=np.float64).reshape(7)
    if float(np.asarray(latest).reshape(-1)[18]) > 0.5 and float(action[6]) > 0:
        return "release"
    return state_phase(latest)


def grip_sign(grip):
    value = float(grip)
    if value > 0:
        return "open"
    if value < 0:
        return "close"
    return "hold"


def delta_z_sign(value):
    number = float(value)
    if number > 0:
        return "up"
    if number < 0:
        return "down"
    return "hold"


def expert_training_window(obs, actions, start):
    """Pair the training window with the executed action, index 1 of the prediction.

    The saved action at ``start`` was already applied to reach the latest frame.
    Inference executes ``sample[:, 1:9]``, so the no-leak label is ``actions[start + 1]``.
    """
    obs = np.asarray(obs, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    start = int(start)
    if start >= 0:
        if start + 1 >= len(actions) or start + 1 >= len(obs):
            raise ValueError("executed action is missing")
        window = np.array(obs[start:start + 2], dtype=np.float64, copy=True)
        action = np.array(actions[start + 1], dtype=np.float64, copy=True)
    else:
        window = np.repeat(np.array(obs[0:1], dtype=np.float64, copy=True), 2, axis=0)
        action = np.array(actions[0], dtype=np.float64, copy=True)
    record = decision_window(window, action.reshape(1, 7), 0)
    record["executed_action_offset"] = 1
    record["previous_action_used_as_label"] = False
    return record


def decision_window(obs, actions, start):
    obs = np.asarray(obs, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != RAW_DIM or actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("decision window expects (T+1, 42) observations and (T, 7) actions")
    start = int(start)
    if start >= 0:
        if start + 1 >= len(obs) or start >= len(actions):
            raise ValueError("decision state is missing")
        window = np.array(obs[start:start + 2], dtype=np.float64, copy=True)
        action = np.array(actions[start], dtype=np.float64, copy=True)
    else:
        window = np.repeat(np.array(obs[0:1], dtype=np.float64, copy=True), 2, axis=0)
        action = np.array(actions[0], dtype=np.float64, copy=True)
    latest = window[-1]
    condition = window[:, :POLICY_DIM]
    return {
        "frames": 2,
        "raw_dim": RAW_DIM,
        "policy_condition_dim": int(condition.size),
        "policy_condition_shape": [2, POLICY_DIM],
        "latest_state_index": 1,
        "future_frames_used": 0,
        "window": window,
        "policy_condition": condition,
        "latest_state": latest,
        "action": action,
        "grip": float(action[6]),
        "grip_sign": grip_sign(action[6]),
        "delta_z": float(action[2]),
        "delta_z_sign": delta_z_sign(action[2]),
        "state_phase": state_phase(latest),
        "policy_phase": policy_phase(latest, action),
    }


def component_distances(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != (RAW_DIM,) or right.shape != (RAW_DIM,):
        raise ValueError("component distance expects 42-d latest states")
    tcp = left[19:22] - right[19:22]
    obj = left[29:32] - right[29:32]
    xy = float(np.linalg.norm(tcp[:2]))
    return {
        "position_3d_m": float(np.linalg.norm(tcp)),
        "object_position_3d_m": float(np.linalg.norm(obj)),
        "xy_only_m": xy,
        "xy_within_3cm": xy <= 0.03,
        "xy_only_is_3d": False,
        "xy_within_3cm_is_3d_close": False,
        "is_3d_close": False,
        "is_surface_clearance": False,
        "rotation_rad": float(quat_geodesic(left[22:26], right[22:26]) + quat_geodesic(left[32:36], right[32:36])),
        "proprio_qpos": float(np.linalg.norm(left[0:9] - right[0:9])),
        "proprio_qvel": float(np.linalg.norm(left[9:18] - right[9:18])),
        "normalized_hint": float(np.linalg.norm(_hint(left) - _hint(right))),
        "normalized_hint_is_3d": False,
        "normalized_hint_axes": "xy_relative_and_radius",
    }


def scaled_features(latest, scales):
    latest = np.asarray(latest, dtype=np.float64).reshape(-1)
    missing = [name for name in SCALE_NAMES if not scales.get(name)]
    if missing:
        raise ValueError(f"coverage scales are missing or zero: {missing}")
    quat_tcp = latest[22:26] / np.linalg.norm(latest[22:26])
    quat_obj = latest[32:36] / np.linalg.norm(latest[32:36])
    if quat_tcp[0] < 0:
        quat_tcp = -quat_tcp
    if quat_obj[0] < 0:
        quat_obj = -quat_obj
    return np.concatenate((
        latest[19:22] / float(scales["position_3d"]),
        latest[29:32] / float(scales["object_position_3d"]),
        quat_tcp / float(scales["rotation"]),
        quat_obj / float(scales["rotation"]),
        latest[0:9] / float(scales["proprio_qpos"]),
        latest[9:18] / float(scales["proprio_qvel"]),
        _hint(latest) / float(scales["normalized_hint"]),
    ))


def composite_distance(left, right, scales):
    return float(np.linalg.norm(scaled_features(left, scales) - scaled_features(right, scales)))


def _require_train_windows(windows):
    if any(row.get("split") == "dev" for row in windows):
        raise ValueError("dev labels are not training coverage")
    ids = [int(row["episode_id"]) for row in windows]
    if any(item < TRAIN_OBSTACLE_ID_MIN or item > TRAIN_OBSTACLE_ID_MAX for item in ids):
        raise ValueError("coverage reference is the 300 obstacle episodes only")
    return ids


def leave_one_scene_out(windows, scales, quantile=RADIUS_QUANTILE):
    _require_train_windows(windows)
    grouped = {}
    for row in windows:
        grouped.setdefault(row["policy_phase"], []).append(row)
    composite_mins = []
    position_mins = []
    for rows in grouped.values():
        by_scene = {}
        for row in rows:
            by_scene.setdefault(int(row["episode_id"]), []).append(row)
        scenes = list(by_scene)
        if len(scenes) < 2:
            raise ValueError("leave-one-scene-out needs two training scenes in the phase")
        features = {
            scene: np.stack([scaled_features(item["latest"], scales) for item in group])
            for scene, group in by_scene.items()
        }
        positions = {
            scene: np.stack([np.asarray(item["latest"], dtype=np.float64).reshape(-1)[19:22] for item in group])
            for scene, group in by_scene.items()
        }
        for scene in scenes:
            other_feat = np.concatenate([features[other] for other in scenes if other != scene], axis=0)
            other_pos = np.concatenate([positions[other] for other in scenes if other != scene], axis=0)
            best_composite = np.inf
            best_position = np.inf
            for start in range(0, len(other_feat), 2048):
                chunk = other_feat[start:start + 2048]
                best_composite = min(best_composite, float(np.linalg.norm(
                    features[scene][:, None, :] - chunk[None, :, :], axis=-1).min()))
                pos = other_pos[start:start + 2048]
                best_position = min(best_position, float(np.linalg.norm(
                    positions[scene][:, None, :] - pos[None, :, :], axis=-1).min()))
            composite_mins.append(best_composite)
            position_mins.append(best_position)
    return {
        "radius": float(np.quantile(np.asarray(composite_mins, dtype=np.float64), float(quantile))),
        "position_3d_radius_m": float(np.quantile(np.asarray(position_mins, dtype=np.float64), float(quantile))),
        "quantile": float(quantile),
        "scene_count": len({int(row["episode_id"]) for row in windows}),
        "window_count": len(windows),
        "source": "train_leave_one_scene_out_same_policy_phase",
        "uses_dev_labels": False,
        "reward_claim": False,
        "xy_only_is_3d": False,
    }


def nearest_coverage(query, windows, scales, radius, position_bar):
    _require_train_windows(windows)
    ranked = []
    for row in windows:
        components = component_distances(query["latest"], row["latest"])
        distance = composite_distance(query["latest"], row["latest"], scales)
        ranked.append((distance, int(row["episode_id"]), row, components))
    ranked.sort(key=lambda item: (item[0], item[1]))
    distance, episode_id, row, components = ranked[0]
    inside = [item for item in ranked if item[0] <= float(radius)]
    signs = {grip_sign(item[2]["action"][6]) for item in inside}
    motions = {delta_z_sign(item[2]["action"][2]) for item in inside}
    return {
        "reward_claim": False,
        "xy_only_is_3d": False,
        "within_radius": bool(distance <= float(radius)),
        "is_3d_close": bool(components["position_3d_m"] <= float(position_bar)),
        "ambiguous_grip": "open" in signs and "close" in signs,
        "ambiguous_delta_z": "up" in motions and "down" in motions,
        "neighbor_count_within_radius": len(inside),
        "radius": float(radius),
        "position_bar_m": float(position_bar),
        "composite": distance,
        "components": components,
        "nearest": {
            "episode_id": episode_id,
            "grip": float(row["action"][6]),
            "grip_sign": grip_sign(row["action"][6]),
            "delta_z": float(row["action"][2]),
            "delta_z_sign": delta_z_sign(row["action"][2]),
            "state_phase": row["state_phase"],
            "policy_phase": row["policy_phase"],
        },
    }


PUBLISHED_CONTACT_STEPS = {1: 90, 3: 146, 5: 113, 9: 146, 19: 92, 21: 162}
FROZEN_CKPT = "outputs/ckpt/20261025_085101_placesphere_obstacle300_90k_constrain/final.pt"
FROZEN_CKPT_SHA = "70cf5e6f00bfec8b7d0945f65805fba06bbddd95356778d73009bdc5fed2326b"
FROZEN_DEV = "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes"
FROZEN_ADAPTER_REGISTRATION = "outputs/exp/20261025_085101_placesphere_obstacle300_90k/preregistration.json"
OBSTACLE_H5 = "demos/PlaceSphere-transport-detour/20261025_085101_obstacle300/trajectory.h5"
OBSTACLE_H5_SHA = "f9dd209868d800fbce41eae276112a23dffc9aec02e91196ca900d46fc42aa02"


def contact_identity(rows, expected_steps=None):
    expected = {int(key): int(value) for key, value in (expected_steps or PUBLISHED_CONTACT_STEPS).items()}
    found = {}
    for row in rows:
        if row.get("obstacle_contact"):
            found[int(row["scene_index"])] = None if row.get("contact_step") is None else int(row["contact_step"])
    return {
        "match": found == expected,
        "found": found,
        "expected": expected,
        "count": len(found),
        "expected_count": len(expected),
    }


def feature_scales(states):
    stacked = np.stack([np.asarray(item, dtype=np.float64).reshape(-1) for item in states])
    hints = np.stack([_hint(row) for row in stacked])

    def std(block):
        value = float(np.std(block))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("coverage scale is not positive")
        return value

    return {
        "position_3d": std(stacked[:, 19:22]),
        "object_position_3d": std(stacked[:, 29:32]),
        "rotation": std(stacked[:, 22:26]),
        "proprio_qpos": std(stacked[:, 0:9]),
        "proprio_qvel": std(stacked[:, 9:18]),
        "normalized_hint": std(hints),
        "uses_dev_labels": False,
    }


def coverage_decision(contact_reports, safe_reports):
    if len(contact_reports) != 6 or len(safe_reports) != 4:
        raise ValueError("coverage decision expects six contacts and four safe controls")
    outside = sum(not row["within_radius"] for row in contact_reports)
    inside_safe = sum(bool(row["within_radius"]) for row in safe_reports)
    ambiguity = any(
        row["within_radius"] and (row["ambiguous_grip"] or row["ambiguous_delta_z"])
        for row in contact_reports)
    strong = outside >= 5 and inside_safe >= 3
    return {
        "contacts_outside_radius": outside,
        "safe_inside_radius": inside_safe,
        "strong_off_path": strong,
        "action_ambiguity": ambiguity,
        "collect_recovery": bool(strong and not ambiguity),
        "reward_claim": False,
        "strong_rule": "at least 5 of 6 contacts outside the train LOSO radius and at least 3 of 4 safe controls inside",
    }


def index_queries(queries, train_windows, scales):
    phases = []
    for query in queries:
        if query["policy_phase"] not in phases:
            phases.append(query["policy_phase"])
    bars = {}
    unsupported = set()
    for phase in phases:
        subset = [row for row in train_windows if row["policy_phase"] == phase]
        if len({int(row["episode_id"]) for row in subset}) < 2:
            unsupported.add(phase)
            continue
        bars[phase] = leave_one_scene_out(subset, scales)
    reports = []
    for query in queries:
        if query["policy_phase"] in unsupported:
            reports.append({
                "scene_index": query.get("scene_index"),
                "role": query.get("role"),
                "query_policy_phase": query["policy_phase"],
                "query_state_phase": query.get("state_phase"),
                "within_radius": False,
                "is_3d_close": False,
                "ambiguous_grip": False,
                "ambiguous_delta_z": False,
                "neighbor_count_within_radius": 0,
                "reward_claim": False,
                "xy_only_is_3d": False,
                "components": {"xy_only_is_3d": False, "xy_within_3cm_is_3d_close": False, "is_3d_close": False},
                "nearest": None,
                "radius_choice": {
                    "quantile": RADIUS_QUANTILE,
                    "source": "no_train_phase_support",
                    "radius": None,
                    "position_3d_radius_m": None,
                    "uses_dev_labels": False,
                    "reward_claim": False,
                    "xy_only_is_3d": False,
                },
            })
            continue
        bar = bars[query["policy_phase"]]
        subset = [row for row in train_windows if row["policy_phase"] == query["policy_phase"]]
        hit = nearest_coverage(
            query, subset, scales, bar["radius"], bar["position_3d_radius_m"])
        hit["scene_index"] = query.get("scene_index")
        hit["role"] = query.get("role")
        hit["query_policy_phase"] = query["policy_phase"]
        hit["query_state_phase"] = query.get("state_phase")
        hit["query_grip"] = float(np.asarray(query["action"]).reshape(-1)[6])
        hit["query_grip_sign"] = grip_sign(hit["query_grip"])
        hit["query_delta_z"] = float(np.asarray(query["action"]).reshape(-1)[2])
        hit["radius_choice"] = {
            "quantile": bar["quantile"],
            "source": bar["source"],
            "radius": bar["radius"],
            "position_3d_radius_m": bar["position_3d_radius_m"],
            "uses_dev_labels": False,
            "reward_claim": False,
            "xy_only_is_3d": False,
        }
        reports.append(hit)
    return reports


def _json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def build_coverage_preregistration(paths, pilot_identity, labeler_identity):
    if not pilot_identity.get("files") or not labeler_identity.get("files"):
        raise ValueError("coverage preregistration needs pilot and labeler file lists")
    if set(labeler_identity["files"]) != set(LABELER_FILES):
        raise ValueError("labeler file list is not the coverage labeler list")
    return {
        "kind": "authorized_coverage",
        "reward_claim": False,
        "uses_dev_labels": False,
        "formal_performance_claim": False,
        "generalization_claim": False,
        "new_data": False,
        "training_trial": False,
        "checkpoint": FROZEN_CKPT,
        "checkpoint_sha256": FROZEN_CKPT_SHA,
        "dev_scenes": FROZEN_DEV,
        "dev_scene_count": 30,
        "adapter_registration": FROZEN_ADAPTER_REGISTRATION,
        "obstacle_h5": OBSTACLE_H5,
        "obstacle_h5_sha256": OBSTACLE_H5_SHA,
        "train_episode_ids": [1000, 1299],
        "train_episode_id_rule": "inclusive 1000..1299 obstacle only",
        "contact_scene_indices": list(CONTACT_SCENE_INDICES),
        "safe_control_indices": list(SAFE_CONTROL_INDICES),
        "expected_contact_steps": {str(key): value for key, value in PUBLISHED_CONTACT_STEPS.items()},
        "radius_quantile": RADIUS_QUANTILE,
        "radius_source": "train_leave_one_scene_out_same_policy_phase",
        "xy_only_is_3d": False,
        "episode_wall_seconds": 120,
        "record_wall_seconds": 3600,
        "index_wall_seconds": 1200,
        "max_steps": 200,
        "paths": dict(paths),
        "pilot_identity": pilot_identity,
        "labeler_identity": labeler_identity,
        "hypothesis": (
            "six frozen closed-loop contact states are off the 300-expert window support, "
            "or they are near and the action is ambiguous; the LOSO radius decides, not 0.03 m"
        ),
        "on_strong_off_path": "one recovery collection on train layouts only, or a feasibility probe if unsafe",
        "on_near": "do not collect; isolate action ambiguity or return the blocker",
    }


def query_from_window(window, *, scene_index, role):
    if not traces_have_full_observations([{"decision_window": window}]):
        raise ValueError("recorded window is not a full no-leak observation")
    action = np.asarray(window["action"], dtype=np.float64).reshape(7)
    return {
        "scene_index": int(scene_index),
        "role": role,
        "split": "query",
        "latest": np.asarray(window["latest_state"], dtype=np.float64),
        "action": action,
        "policy_phase": window["policy_phase"],
        "state_phase": window["state_phase"],
    }


def train_windows_from_h5(path, expected_sha):
    import h5py
    from examples.baselines.constrain.placesphere_learning_pilot import _window_slices, sha256_file

    if sha256_file(path) != expected_sha:
        raise RuntimeError("obstacle demo hash changed")
    rows = []
    with h5py.File(path, "r") as handle:
        for episode_id in range(TRAIN_OBSTACLE_ID_MIN, TRAIN_OBSTACLE_ID_MAX + 1):
            key = f"traj_{episode_id}"
            if key not in handle:
                raise ValueError(f"missing training episode {episode_id}")
            obs = np.asarray(handle[key]["obs"])
            actions = np.asarray(handle[key]["actions"])
            if obs.shape[1] != RAW_DIM or actions.shape[1] != 7 or len(obs) != len(actions) + 1:
                raise ValueError(f"traj_{episode_id} schema mismatch")
            for start, _end in _window_slices(len(actions)):
                record = expert_training_window(obs, actions, start)
                rows.append({
                    "episode_id": episode_id,
                    "split": "train",
                    "window_start": int(start),
                    "latest": record["latest_state"],
                    "action": record["action"],
                    "policy_phase": record["policy_phase"],
                    "state_phase": record["state_phase"],
                })
    if not rows:
        raise ValueError("coverage reference is the 300 obstacle episodes only")
    return rows


def _deadline(_signum, _frame):
    raise TimeoutError("coverage wall exceeded")


def record_coverage(registration, device="cuda"):
    import signal
    import time
    from examples.baselines.constrain.diagnose_placesphere_failure import _child_pids, _terminate_new_children
    from examples.baselines.constrain.placesphere_learning_pilot import (
        PilotConstrainAdapter, load_preregistration, runtime_identity_gate, sha256_file,
    )
    from examples.baselines.constrain.placesphere_release_diagnosis import instrument_episode
    from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet

    identity = runtime_identity_gate(registration, "record-coverage")
    if sha256_file(registration["checkpoint"]) != registration["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint sha changed")
    if sha256_file(registration["obstacle_h5"]) != registration["obstacle_h5_sha256"]:
        raise RuntimeError("obstacle demo hash changed")
    output = Path(registration["paths"]["exp"]) / "recorded"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    adapter_registration = load_preregistration(registration["adapter_registration"])
    catalog = TransportDetourSceneSet(registration["dev_scenes"])
    if len(catalog) != 30:
        raise ValueError("frozen dev scene count is not 30")
    adapter = PilotConstrainAdapter(registration["checkpoint"], adapter_registration, device)
    if adapter.metadata["checkpoint_sha256"] != FROZEN_CKPT_SHA:
        raise RuntimeError("loaded checkpoint sha mismatch")
    expected = {int(key): int(value) for key, value in registration["expected_contact_steps"].items()}
    rows = []
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    try:
        for index in range(30):
            if time.monotonic() - started > float(registration["record_wall_seconds"]):
                raise TimeoutError("coverage wall exceeded")
            before = set(_child_pids())
            previous = signal.signal(signal.SIGALRM, _deadline)
            signal.setitimer(signal.ITIMER_REAL, float(registration["episode_wall_seconds"]))
            try:
                traced = instrument_episode(adapter, catalog, index, device, max_steps=200, record_windows=True)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
                _terminate_new_children(before)
            contact = traced.get("first_contact") or {}
            release = traced.get("release_geometry") or {}
            row = {
                "scene_index": index,
                "scene_id": traced.get("scene_id"),
                "status": traced.get("status"),
                "obstacle_contact": bool((traced.get("result") or {}).get("obstacle_contact")),
                "native_success": bool((traced.get("result") or {}).get("native_success")),
                "contact_step": contact.get("step"),
                "queue_index": contact.get("queue_index"),
                "policy_window": contact.get("policy_window"),
                "execution_window": contact.get("execution_window"),
                "release_policy_window": release.get("policy_window"),
                "release_execution_window": release.get("execution_window"),
                "checkpoint_sha256": traced.get("checkpoint_sha256"),
            }
            rows.append(row)
            _json_atomic(output / f"episode_{index:03d}.json", row)
            print(f"scene {index} contact={row['obstacle_contact']} step={row['contact_step']}", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    identity_report = contact_identity(rows, expected)
    payload = {
        "running_identity": identity,
        "contact_identity": identity_report,
        "elapsed_s": time.monotonic() - started,
        "reward_claim": False,
        "episodes": rows,
    }
    _json_atomic(output / "record.json", payload)
    if not identity_report["match"]:
        raise RuntimeError(f"contact identity mismatch: {identity_report}")
    for index in CONTACT_SCENE_INDICES:
        row = rows[index]
        if not traces_have_full_observations([{"decision_window": row["execution_window"]}]):
            raise RuntimeError(f"scene {index} execution window is incomplete")
        if not traces_have_full_observations([{"decision_window": row["policy_window"]}]):
            raise RuntimeError(f"scene {index} policy window is incomplete")
    for index in SAFE_CONTROL_INDICES:
        window = rows[index]["release_execution_window"]
        if not traces_have_full_observations([{"decision_window": window}]):
            raise RuntimeError(f"safe scene {index} release window is incomplete")
    return payload


def index_coverage(registration):
    import time
    from examples.baselines.constrain.placesphere_learning_pilot import runtime_identity_gate, sha256_file

    identity = runtime_identity_gate(registration, "index-coverage")
    if sha256_file(registration["obstacle_h5"]) != registration["obstacle_h5_sha256"]:
        raise RuntimeError("obstacle demo hash changed")
    recorded = json.loads((Path(registration["paths"]["exp"]) / "recorded" / "record.json").read_text())
    if not recorded["contact_identity"]["match"]:
        raise RuntimeError("refusing to index a contact-identity mismatch")
    rows = recorded["episodes"]
    queries = []
    for index in CONTACT_SCENE_INDICES:
        queries.append(query_from_window(rows[index]["execution_window"], scene_index=index, role="contact_execution"))
        queries.append(query_from_window(rows[index]["policy_window"], scene_index=index, role="contact_policy"))
    for index in SAFE_CONTROL_INDICES:
        queries.append(query_from_window(
            rows[index]["release_execution_window"], scene_index=index, role="safe_release_execution"))
    started = time.monotonic()
    train_rows = train_windows_from_h5(registration["obstacle_h5"], registration["obstacle_h5_sha256"])
    scales = feature_scales([row["latest"] for row in train_rows])
    phase_counts = {}
    for row in train_rows:
        phase_counts[row["policy_phase"]] = phase_counts.get(row["policy_phase"], 0) + 1
    reports = index_queries(queries, train_rows, scales)
    execution = [row for row in reports if row["role"] == "contact_execution"]
    policy = [row for row in reports if row["role"] == "contact_policy"]
    safe = [row for row in reports if row["role"] == "safe_release_execution"]
    decision = coverage_decision(execution, safe)
    decision["policy_window_decision"] = coverage_decision(policy, safe)
    decision["execution_and_policy_agree"] = (
        decision["strong_off_path"] == decision["policy_window_decision"]["strong_off_path"]
        and decision["action_ambiguity"] == decision["policy_window_decision"]["action_ambiguity"])
    report = {
        "running_identity": identity,
        "scales": scales,
        "phase_counts": phase_counts,
        "window_count": len(train_rows),
        "train_episode_span": [TRAIN_OBSTACLE_ID_MIN, TRAIN_OBSTACLE_ID_MAX],
        "uses_dev_labels": False,
        "reward_claim": False,
        "xy_only_is_3d": False,
        "queries": reports,
        "decision": decision,
        "elapsed_s": time.monotonic() - started,
        "ambiguity_radius_choice": reports[0]["radius_choice"] if reports else None,
    }
    dest = Path(registration["paths"]["exp"]) / "coverage.json"
    _json_atomic(dest, report)
    return report


def main(argv=None):
    import argparse
    import time
    from examples.baselines.constrain.placesphere_learning_pilot import freeze_pilot_identity

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "record", "index"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.command == "preregister":
        if args.registration.exists():
            raise FileExistsError(args.registration)
        exp = args.registration.parent
        payload = build_coverage_preregistration(
            {"exp": str(exp), "recorded": str(exp / "recorded"), "coverage": str(exp / "coverage.json")},
            freeze_pilot_identity(), freeze_labeler_identity())
        _json_atomic(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    registration = json.loads(args.registration.read_text())
    if registration.get("kind") != "authorized_coverage":
        raise ValueError("registration is not the coverage kind")
    if args.command == "record":
        result = record_coverage(registration, args.device)
    else:
        result = index_coverage(registration)
    print(json.dumps({"command": args.command, "elapsed_s": time.monotonic() - started,
                      "reward_claim": False}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
