"""Opt-in collection and audit of matched shared-terminal PlaceSphere experts.

The historical formal and transport-detour collectors remain unchanged. This
collector reads frozen existing layout catalogs, but regenerates every accepted
training trajectory with the matched expert.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain import placesphere_matched_terminal as matched


SOURCE_FILES = (
    "examples/baselines/constrain/placesphere_matched_collection.py",
    "examples/baselines/constrain/placesphere_matched_terminal.py",
    "examples/baselines/constrain/placesphere_transport_detour.py",
    "examples/baselines/constrain/placesphere_pilot.py",
    "examples/baselines/constrain/placesphere_obstacle.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/placesphere_round20_tplusone.py",
    "examples/baselines/constrain/placesphere_scenes.py",
    "examples/baselines/constrain/placesphere_learning_pilot.py",
)
SOURCE_LAYOUTS = (
    "outputs/exp/20261024_155244_placesphere_data_expansion/direct_scenes/assembled",
    "outputs/exp/20261024_155244_placesphere_data_expansion/pair_scenes/merged",
    "outputs/exp/20261025_085101_placesphere_obstacle300_90k/obstacle_scenes/catalog_full",
)
EXTERNAL_DEDUP_CATALOGS = (
    "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes",
    "outputs/exp/20261023_175529_placesphere_transport_detour",
)
NATIVE_SOURCE_FILES = (
    "mani_skill/envs/tasks/tabletop/place_sphere.py",
    "mani_skill/examples/motionplanning/panda/solutions/place_sphere.py",
)
MARGIN_M = 0.015
HOLD_STEPS = 0
DIRECT_TARGET = 1000
OBSTACLE_TARGET = 300
LAYOUT_OVERLAP_TARGET = 300
POLICY_STEPS = 200
EPISODE_WALL_S = 160
OUTER_WALL_S = 72000
MAX_ATTEMPTS = 5000
NEUTRAL_WINDOW = {
    "definition": "executed open-gripper actions with XYZ position and rotation action norm <= 0.02 and sphere linear/angular velocity each <= native PlaceSphere limits; counted per consecutive fixed 8-action window, labeled by the current expert phase",
    "window_actions": 8,
    "max_position_rotation_action_norm": 0.02,
    "max_linear_velocity_m_s": matched.NATIVE_LIN_THRESH_M_S,
    "max_angular_velocity_rad_s": matched.NATIVE_ANG_THRESH_RAD_S,
    "gripper_open": "action[6] > 0",
    "phase_source": "step_phase at the same control transition; no future-state phase inference",
}


def source_identity(repo_root=None):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def layout_rows(catalogs=SOURCE_LAYOUTS):
    """Read original source identities from catalogs without altering their files."""
    rows, seen = [], set()
    for catalog in catalogs:
        path = Path(catalog)
        manifest = json.loads((path / "manifest.json").read_text())
        for index, scene in enumerate(manifest["scenes"]):
            key = (str(path), int(index), int(scene["seed"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "source": str(path), "index": index, "seed": int(scene["seed"]),
                "scene_id": scene["scene_id"], "catalog": str(path),
                "start_xy": scene["start_xy"], "container_xy": scene["container_xy"],
                "cylinder": scene["cylinder"],
            })
    return rows


def _task_layout_conflict(left, right, tolerance=0.02):
    if int(left["seed"]) == int(right["seed"]):
        return True
    start_match = np.max(np.abs(np.asarray(left["start_xy"]) - np.asarray(right["start_xy"]))) <= tolerance
    goal_match = np.max(np.abs(np.asarray(left["container_xy"]) - np.asarray(right["container_xy"]))) <= tolerance
    return bool(start_match and goal_match)


def _greedy_task_identity(rows):
    unique, conflicts = [], []
    for row in rows:
        if any(_task_layout_conflict(row, kept) for kept in unique):
            conflicts.append(row)
        else:
            unique.append(row)
    return {"count": len(rows), "greedy_unique": len(unique), "conflicts": len(conflicts)}


def select_role_candidates(rows, *, obstacle_target=OBSTACLE_TARGET):
    paired = [row for row in rows if row["catalog"].endswith("pair_scenes/merged")]
    paired.extend(row for row in rows if row["catalog"].endswith("catalog_full"))
    direct = [row for row in rows if row["catalog"].endswith("direct_scenes/assembled")]
    if len(paired) != int(obstacle_target) or _greedy_task_identity(paired)["conflicts"]:
        raise ValueError("paired obstacle role candidates must be task-unique and match the obstacle target")
    backups, seen = [], list(paired)
    for row in direct:
        if any(_task_layout_conflict(row, existing) for existing in seen):
            continue
        backups.append(row)
        seen.append(row)
    return {"paired_primary": paired, "direct_only": direct, "obstacle_replacements": backups}


def _runtime_contract():
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSpherePilotConfig, _obs_vector, _policy_observation, _policy_observation_schema, make_env,
    )

    repo = Path(__file__).resolve().parents[3]
    native = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in NATIVE_SOURCE_FILES}
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=matched.EXPERT_MAX_STEPS, obstacle_radius=matched.RADIUS_M,
        obstacle_height=0.08))
    try:
        _obs, info = env.reset(seed=2026102621, options={
            "sphere_xy": np.array([-0.14, 0.0]), "container_xy": np.array([0.12, 0.0]),
            "obstacle_enabled": True, "obstacle_center": np.array([0.0, 0.12]),
        })
        base = env.unwrapped
        policy_schema = _policy_observation_schema(base, info)
        raw_schema = _policy_observation_schema(base, info, include_obstacle_hint=True)
        action_space = env.action_space
        controller = base.agent.controller.controllers["arm"].config
        return {
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "torch", "gymnasium", "mani-skill", "mplib")},
            "native_source_sha256": native,
            "control_mode": str(base.control_mode),
            "action_dimension": int(np.prod(action_space.shape)),
            "action_space_low": np.asarray(action_space.low).tolist(),
            "action_space_high": np.asarray(action_space.high).tolist(),
            "policy_observation_dimension": len(_policy_observation(base, info)),
            "raw_observation_dimension": len(_obs_vector(base.get_obs(info=info, unflattened=True))),
            "policy_observation_schema": policy_schema,
            "raw_observation_schema": raw_schema,
            "arm_rotation_lower": np.asarray(
                controller.rot_lower.detach().cpu() if hasattr(controller.rot_lower, "detach")
                else controller.rot_lower).tolist(),
            "expected_contract": {"policy_dim": 39, "raw_dim": 42, "action_dim": 7,
                                  "obstacle_hint_path": "/extra/placesphere_obstacle"},
        }
    finally:
        env.close()


def preregister(path, *, catalogs=SOURCE_LAYOUTS, seed=2026102621,
                release_hold_steps=HOLD_STEPS):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if int(release_hold_steps) not in matched.RELEASE_HOLD_STEPS_CANDIDATES:
        raise ValueError("release_hold_steps must use a preregistered value (0, 8, or 16)")
    rows = layout_rows(catalogs)
    if len(rows) < DIRECT_TARGET:
        raise ValueError(f"frozen catalogs provide only {len(rows)} layouts; need {DIRECT_TARGET}")
    external = []
    external_records = []
    for catalog in EXTERNAL_DEDUP_CATALOGS:
        manifest_bytes = (Path(catalog) / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        external.extend(manifest["scenes"])
        external_records.append({"path": catalog, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                                 "scene_count": len(manifest["scenes"])})
    external_conflicts = [row for row in rows if any(_task_layout_conflict(row, other) for other in external)]
    if external_conflicts:
        raise ValueError(f"frozen training catalogs conflict with dev/Ticket04: {len(external_conflicts)}")
    legacy_direct_order = [row for row in rows if row["catalog"].endswith("pair_scenes/merged")]
    legacy_direct_order.extend(row for row in rows if row["catalog"].endswith("direct_scenes/assembled"))
    role_candidates = select_role_candidates(rows)
    payload = {
        "kind": "placesphere_matched_shared_terminal_training_dataset",
        "version": "r21_matched_expert_1000_direct_300_obstacle",
        "source_identity": source_identity(),
        "runtime_contract": _runtime_contract(),
        "catalogs": [{"path": str(p), "manifest_sha256": hashlib.sha256(
            (Path(p) / "manifest.json").read_bytes()).hexdigest()} for p in catalogs],
        "catalog_layouts": rows,
        "role_candidate_layouts": role_candidates,
        "external_dedup_catalogs": external_records,
        "external_dedup": {"rule": "same seed, or start and container each within 0.02 m componentwise",
                            "candidate_conflicts": 0, "frozen_dev_and_ticket04_excluded": True},
        "source_task_identity_audit": {
            "all_catalog_rows": _greedy_task_identity(rows),
            "original_1000_direct_order_pair100_then_direct900": _greedy_task_identity(legacy_direct_order),
            "cylinder_proximity_is_not_identity": True,
        },
        "targets": {"direct": DIRECT_TARGET, "obstacle": OBSTACLE_TARGET,
                    "paired_layouts": LAYOUT_OVERLAP_TARGET},
        "pilot": {"planned_direct": 80, "planned_obstacle": 25,
                  "planned_paired_layouts": 25, "paired_candidate_prefix": 30,
                  "direct_only_candidate_prefix": 100,
                  "pilot_trajectories_are_final_dataset_subset": True,
                  "dev_scenes_used": False},
        "selection": "300 primary paired layouts are exactly the 100 pair catalog plus 200 catalog_full obstacle layouts; 900 direct-only candidates are separate; obstacle failure replacements use only the preregistered task-unique direct-catalog backup list; failed layouts remain in the rejection journal",
        "replacement_order_seed": int(seed),
        "terminal": {"mode": "above_release", "margin_m": MARGIN_M,
                     "release_hold_steps": int(release_hold_steps),
                     "common_z": "max(measured direct requirement, measured actual-cylinder obstacle requirement, transport-exit z)",
                     "settle_budget_steps": matched.SETTLE_BUDGET,
                     "expert_action_budget_steps": matched.EXPERT_MAX_STEPS,
                     "policy_evaluation_budget_steps": POLICY_STEPS},
        "unchanged": {"cylinder_radius_m": matched.RADIUS_M, "native_xy_m": matched.NATIVE_XY_M,
                      "native_z_tolerance_m": matched.NATIVE_Z_TOL_M,
                      "native_linear_m_s": matched.NATIVE_LIN_THRESH_M_S,
                      "native_angular_rad_s": matched.NATIVE_ANG_THRESH_RAD_S,
                      "geometry_or_challenge_height_changed": False,
                      "legacy_default_changed": False},
        "roles": {"DIRECT": "positive_direct; parked cylinder but planned cylinder retained for shared release measurement",
                  "OBSTACLE": "positive_obstacle; actual cylinder never relocated; same-side waypoint; contact-free from execution start",
                  "negative_direct": "blocking probe only; never admitted", "terminal_replay": "suffix proof only; never admitted"},
        "admission": matched.COLLECTION_READY_IFF,
        "storage": {"observations": "39-D policy and 42-D native raw, T+1",
                    "actions": "native executed 7-D, T", "state_controller_info": "all T+1 snapshots/infos",
                    "terminal": "first native success; full trajectory preserved"},
        "neutral_window_definition": NEUTRAL_WINDOW,
        "budgets": {"episode_wall_seconds": EPISODE_WALL_S, "outer_wall_seconds": OUTER_WALL_S,
                    "max_attempts": MAX_ATTEMPTS, "workers": 1},
        "dedup": {"task_layout_identity": "same seed, or start and container each within 0.02 m componentwise; cylinder proximity is not identity",
                  "old_Tplusone_is_training": False,
                  "old_trajectories_copied": False,
                  "external_train_dev_ticket04_checks": "recorded from catalog source metadata; any duplicate is retained as a source-identity conflict, never relabeled unique"},
        "training": False,
        "created_before_collection": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def assert_preregistration(path):
    registration = json.loads(Path(path).read_text())
    if registration.get("source_identity") != source_identity():
        raise RuntimeError("matched collector source differs from frozen preregistration")
    if registration.get("runtime_contract") != _runtime_contract():
        raise RuntimeError("runtime package, native source, action, or observation contract changed")
    if registration.get("training") is not False or registration.get("created_before_collection") is not True:
        raise RuntimeError("registration is not a valid pre-collection freeze")
    if registration.get("terminal", {}).get("release_hold_steps") not in matched.RELEASE_HOLD_STEPS_CANDIDATES:
        raise RuntimeError("registration has an unsupported release hold")
    for catalog in registration["catalogs"]:
        digest = hashlib.sha256((Path(catalog["path"]) / "manifest.json").read_bytes()).hexdigest()
        if digest != catalog["manifest_sha256"]:
            raise RuntimeError(f"source layout catalog changed: {catalog['path']}")
    for catalog in registration["external_dedup_catalogs"]:
        digest = hashlib.sha256((Path(catalog["path"]) / "manifest.json").read_bytes()).hexdigest()
        if digest != catalog["manifest_sha256"]:
            raise RuntimeError(f"external dedup catalog changed: {catalog['path']}")
    current_rows = layout_rows([item["path"] for item in registration["catalogs"]])
    expected = [(_layout_key(row), row["scene_id"]) for row in registration["catalog_layouts"]]
    actual = [(_layout_key(row), row["scene_id"]) for row in current_rows]
    if actual != expected:
        raise RuntimeError("frozen source layout identity/order changed")
    current_by_key = {_layout_key(row): row for row in current_rows}
    candidates = registration["role_candidate_layouts"]
    for role in ("paired_primary", "direct_only", "obstacle_replacements"):
        if any(current_by_key.get(_layout_key(row)) != row for row in candidates[role]):
            raise RuntimeError(f"frozen {role} selection differs from the source catalog")
    if len(candidates["paired_primary"]) != OBSTACLE_TARGET:
        raise RuntimeError("the frozen primary obstacle layout cohort is not exactly 300")
    paired = candidates["paired_primary"]
    backups = candidates["obstacle_replacements"]
    if (_greedy_task_identity(paired)["conflicts"]
            or any(_task_layout_conflict(row, primary) for row in backups for primary in paired)
            or _greedy_task_identity(paired + backups)["conflicts"]):
        raise RuntimeError("obstacle primary/replacement layouts are not task-unique")
    external = []
    for catalog in registration["external_dedup_catalogs"]:
        external.extend(json.loads((Path(catalog["path"]) / "manifest.json").read_text())["scenes"])
    if any(_task_layout_conflict(row, other) for row in current_rows for other in external):
        raise RuntimeError("frozen source layouts overlap dev/Ticket04 under the task identity rule")
    return registration


def _layout_key(layout):
    return (str(layout["catalog"]), int(layout["index"]), int(layout["seed"]))


def _pair(layout, episode_wall_s, release_hold_steps):
    obstacle = matched.run_layout_expert(
        layout, margin_m=MARGIN_M, role=matched.ROLE_POSITIVE_OBSTACLE,
        release_hold_steps=release_hold_steps, episode_wall_s=episode_wall_s)
    direct = matched.run_layout_expert(
        layout, margin_m=MARGIN_M, role=matched.ROLE_POSITIVE_DIRECT,
        shared_targets=obstacle.get("shared_targets"), release_hold_steps=release_hold_steps,
        episode_wall_s=episode_wall_s)
    common = matched._common_release_targets(obstacle.get("shared_targets") or {}, direct, obstacle)
    if common is not None:
        common["release_hold_steps"] = int(release_hold_steps)
        if obstacle.get("release_tcp_z") is None or abs(float(obstacle["release_tcp_z"]) - common["release_tcp_z"]) > 1e-6:
            obstacle = matched.run_layout_expert(
                layout, margin_m=MARGIN_M, role=matched.ROLE_POSITIVE_OBSTACLE,
                shared_targets=common, release_hold_steps=release_hold_steps,
                episode_wall_s=episode_wall_s)
            direct = matched.run_layout_expert(
                layout, margin_m=MARGIN_M, role=matched.ROLE_POSITIVE_DIRECT,
                shared_targets=common, release_hold_steps=release_hold_steps,
                episode_wall_s=episode_wall_s)
    replay = matched.replay_direct_terminal(layout, direct, episode_wall_s=episode_wall_s)
    negative = matched.run_layout_expert(
        layout, margin_m=MARGIN_M, role=matched.ROLE_NEGATIVE_DIRECT,
        episode_wall_s=episode_wall_s)
    admission = matched.collection_admission(direct, obstacle, replay, negative)
    return direct, obstacle, replay, negative, common, admission


def _write_episode(group, record, kind, metadata):
    from examples.baselines.constrain.placesphere_pilot import _stack_state_rows, _state_write, _tree_sha256

    raw = record["_raw"]
    actions = np.asarray(raw["actions"], dtype=np.float32)
    policy = np.asarray(raw["policy_observations"], dtype=np.float32)
    raw_obs = np.asarray(raw["raw_observations"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or policy.shape != (len(actions) + 1, 39) or raw_obs.shape != (len(actions) + 1, 42):
        raise ValueError("matched expert violated T/T+1 action/observation contract")
    if (not np.isfinite(actions).all() or not np.isfinite(policy).all() or not np.isfinite(raw_obs).all()
            or len(raw["infos"]) != len(actions) + 1
            or len(raw["states"]) != len(actions) + 1
            or len(raw["controller_states"]) != len(actions) + 1
            or len(raw["terminated"]) != len(actions) + 1
            or len(raw["truncated"]) != len(actions) + 1
            or np.asarray(raw["truncated"]).any() or not bool(np.asarray(raw["terminated"])[-1])):
        raise ValueError("matched expert is missing finite native T+1 evidence or native termination")
    if record.get("observation_contract_ok") is not True:
        raise ValueError("matched expert failed named 39-D policy / 42-D raw observation contract")
    if not record.get("native_success") or not record.get("on_bin") or not record.get("static") or not record.get("ungrasped"):
        raise ValueError("matched expert did not reach first native ungrasped static placement")
    if kind == "obstacle" and (not record.get("from_execution_start_safe") or record.get("obstacle_contact") or record.get("obstacle_crossing")):
        raise ValueError("obstacle episode is not full-prefix contact/crossing free")
    group.attrs["kind"] = kind
    group.attrs["seed"] = int(record["seed"])
    group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    group.create_dataset("actions", data=actions)
    group.create_dataset("obs", data=policy)
    group.create_dataset("raw_obs", data=raw_obs)
    group.create_dataset("success", data=np.asarray([bool(info.get("native_success")) for info in raw["infos"]], dtype=np.uint8))
    group.create_dataset("terminated", data=np.asarray(raw["terminated"], dtype=np.uint8))
    group.create_dataset("truncated", data=np.asarray(raw["truncated"], dtype=np.uint8))
    for name, rows in (("env_states", raw["states"]), ("controller_states", raw["controller_states"])):
        node = group.create_group(name)
        for key, value in _stack_state_rows(rows).items():
            _state_write(node, key, value)
    group["controller_states"].create_dataset("snapshot_sha256", data=np.asarray(
        [_tree_sha256(row).encode("ascii") for row in raw["controller_states"]], dtype="S64"))


def run_collection(registration_path, dataset_dir, experiment_dir, *, target_direct=DIRECT_TARGET,
                   target_obstacle=OBSTACLE_TARGET, pilot_layouts=0, pair_layout_target=None,
                   episode_wall_s=EPISODE_WALL_S,
                   outer_wall_s=OUTER_WALL_S, max_attempts=MAX_ATTEMPTS):
    registration = assert_preregistration(registration_path)
    if pilot_layouts and (pilot_layouts < 20 or pilot_layouts > 100):
        raise ValueError("pilot_layouts must be between 20 and 100")
    if target_direct < 1 or target_obstacle < 1 or target_obstacle > target_direct:
        raise ValueError("targets must be positive and obstacle <= direct")
    dataset, experiment = Path(dataset_dir), Path(experiment_dir)
    dataset.mkdir(parents=True, exist_ok=False)
    experiment.mkdir(parents=True, exist_ok=False)
    frozen = {"registration_sha256": hashlib.sha256(Path(registration_path).read_bytes()).hexdigest(),
              "source_identity": registration["source_identity"], "targets": {"direct": target_direct,
              "obstacle": target_obstacle}, "pilot_layouts": pilot_layouts,
              "episode_wall_s": episode_wall_s, "outer_wall_s": outer_wall_s,
              "max_attempts": max_attempts, "workers": 1}
    (dataset / "frozen_contract.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    (dataset / "collection_config.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    rng = np.random.default_rng(int(registration["replacement_order_seed"]))
    role_candidates = registration["role_candidate_layouts"]

    def shuffled(rows):
        return [rows[int(i)] for i in rng.permutation(len(rows))]

    pair_candidates = shuffled(role_candidates["paired_primary"])
    direct_candidates = shuffled(role_candidates["direct_only"])
    obstacle_backups = shuffled(role_candidates["obstacle_replacements"])
    if pilot_layouts:
        pair_candidates = pair_candidates[:min(30, len(pair_candidates))]
        direct_candidates = direct_candidates[:pilot_layouts]
    targets = {"direct": int(target_direct), "obstacle": int(target_obstacle)}
    pair_layout_target = targets["obstacle"] if pair_layout_target is None else int(pair_layout_target)
    if pair_layout_target != targets["obstacle"]:
        raise ValueError("every obstacle training episode must belong to a matched positive pair")
    accepted = {"direct": [], "obstacle": []}
    failures, tried, paired, used_backups = [], set(), set(), set()
    started = time.monotonic()
    with h5py.File(dataset / "trajectory.h5", "w") as h5, (dataset / "trajectory.json").open("w") as manifest, (dataset / "attempts.jsonl").open("w") as journal:
        manifest.write('{"episodes":[')
        episode_id = 0
        pair_cursor = 0
        direct_cursor = 0
        backup_cursor = 0
        attempt_count = 0

        def persist(kind, layout, record, audit, admission, common, episode_id):
            meta = {"episode_id": episode_id, "kind": kind, "layout": layout,
                    "source_identity": registration["source_identity"],
                    "matched_admission": admission,
                    "t_plus_one_restore_audit": audit,
                    "common_release_measurements": None if common is None else common.get("release_target_measurements"),
                    "step_phase": record["_raw"]["trace"]["step_phase"],
                    "sphere_linear_speed_m_s": record["_raw"]["trace"]["sphere_linear_speed_m_s"],
                    "sphere_angular_speed_rad_s": record["_raw"]["trace"]["sphere_angular_speed_rad_s"],
                    "expert_record": matched._public_record(record)}
            _write_episode(h5.create_group(f"traj_{episode_id}"), record, kind, meta)
            episode = {"episode_id": episode_id, "kind": kind, "seed": record["seed"],
                       "layout": layout, "trace": record["phase_status"],
                       "infos": _json_infos(record["_raw"]["infos"])}
            manifest.write(("," if episode_id else "") + json.dumps(episode, sort_keys=True))
            manifest.flush()

        while (len(accepted["direct"]) < targets["direct"] or len(accepted["obstacle"]) < targets["obstacle"]):
            if attempt_count >= max_attempts:
                break
            if time.monotonic() - started >= outer_wall_s:
                break
            is_pair_layout = len(accepted["obstacle"]) < targets["obstacle"]
            if is_pair_layout:
                if pair_cursor < len(pair_candidates):
                    layout = pair_candidates[pair_cursor]
                    from_backup = False
                    pair_cursor += 1
                elif backup_cursor < len(obstacle_backups):
                    layout = obstacle_backups[backup_cursor]
                    from_backup = True
                    backup_cursor += 1
                else:
                    break
            else:
                while direct_cursor < len(direct_candidates) and _layout_key(direct_candidates[direct_cursor]) in used_backups:
                    direct_cursor += 1
                if direct_cursor >= len(direct_candidates):
                    break
                layout = direct_candidates[direct_cursor]
                from_backup = False
                direct_cursor += 1
            key = _layout_key(layout)
            tried.add(key)
            attempt_count += 1
            if is_pair_layout:
                direct, obstacle, replay, negative, common, admission = _pair(
                    layout, episode_wall_s, int(registration["terminal"]["release_hold_steps"]))
            else:
                direct = matched.run_layout_expert(
                    layout, margin_m=MARGIN_M, role=matched.ROLE_POSITIVE_DIRECT,
                    release_hold_steps=int(registration["terminal"]["release_hold_steps"]),
                    episode_wall_s=episode_wall_s)
                admission = {"collection_ready": bool(matched.positive_direct_accepted(direct)),
                             "reasons": [] if matched.positive_direct_accepted(direct) else ["positive_direct_rejected"]}
                obstacle = replay = negative = common = None
            if not admission["collection_ready"]:
                row = {"layout": layout, "accepted": False, "reason": admission["reasons"],
                       "roles": {"direct": matched._public_record(direct),
                                 "obstacle": None if obstacle is None else matched._public_record(obstacle),
                                  "replay": replay, "negative": None if negative is None else matched._public_record(negative)}}
                failures.append(row)
                journal.write(json.dumps(row, sort_keys=True) + "\n"); journal.flush()
                continue
            audit_direct = None
            audit_obstacle = None
            if len(accepted["direct"]) < targets["direct"]:
                audit_direct = _restore_one_step(layout, direct)
            if is_pair_layout and len(accepted["obstacle"]) < targets["obstacle"]:
                audit_obstacle = _restore_one_step(layout, obstacle)
            if ((audit_direct is not None and not audit_direct["passed"])
                    or (audit_obstacle is not None and not audit_obstacle["passed"])):
                row = {"layout": layout, "accepted": False, "reason": "t_plus_one_restore_mismatch",
                       "restore_audits": {"direct": audit_direct, "obstacle": audit_obstacle}}
                failures.append(row)
                journal.write(json.dumps(row, sort_keys=True) + "\n"); journal.flush()
                continue
            if from_backup:
                used_backups.add(key)
            if len(accepted["direct"]) < targets["direct"]:
                accepted["direct"].append(key)
                persist("direct", layout, direct, audit_direct, admission, common, episode_id)
                episode_id += 1
            if is_pair_layout:
                accepted["obstacle"].append(key); paired.add(key)
                persist("obstacle", layout, obstacle, audit_obstacle, admission, common, episode_id)
                episode_id += 1
            journal.write(json.dumps({"layout": layout, "accepted": True, "replacement": from_backup,
                                      "accepted_counts": {key: len(value) for key, value in accepted.items()},
                                      "negative_blocked_in_transport": None if negative is None else negative.get("blocked_in_transport"),
                                      "terminal_replay": replay,
                                      "elapsed_seconds": time.monotonic() - started}, sort_keys=True) + "\n")
            journal.flush(); h5.flush()
        manifest.write("]}\n")
    report = {"complete": all(len(accepted[k]) == targets[k] for k in targets), "targets": targets,
              "accepted": {key: len(value) for key, value in accepted.items()},
              "unique_layouts": len(set(accepted["direct"] + accepted["obstacle"])),
              "paired_layouts": len(paired), "paired_layout_target": pair_layout_target,
              "attempted_layouts": attempt_count, "replacement_layouts": len(used_backups), "rejections": failures,
              "elapsed_seconds": time.monotonic() - started,
              "training": False, "source_identity": registration["source_identity"],
              "task_identity": {
                  "direct": _greedy_task_identity([next(row for row in registration["catalog_layouts"] if _layout_key(row) == key)
                                                   for key in accepted["direct"]]),
                  "obstacle": _greedy_task_identity([next(row for row in registration["catalog_layouts"] if _layout_key(row) == key)
                                                     for key in accepted["obstacle"]]),
                  "external_dev_ticket04_conflicts": 0,
              },
              "trajectory_sha256": hashlib.sha256((dataset / "trajectory.h5").read_bytes()).hexdigest()}
    report["window_statistics"] = _window_statistics(dataset / "trajectory.h5", report["accepted"])
    (dataset / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (experiment / "collection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _json_infos(infos):
    from examples.baselines.constrain.placesphere_pilot import _json
    return _json(infos)


def _restore_one_step(layout, record):
    from examples.baselines.constrain.placesphere_round20_tplusone import _restore_one_step as audit
    return audit(layout, record)


def _window_statistics(path, accepted):
    rows = {"episodes": [], "neutral_definition": NEUTRAL_WINDOW}
    neutral_by_phase = {}
    with h5py.File(path, "r") as h5:
        for key in sorted(h5, key=lambda item: int(item.split("_")[1])):
            group = h5[key]
            actions = np.asarray(group["actions"])
            metadata = json.loads(group.attrs["metadata_json"])
            current_phases = metadata["step_phase"]
            linear = metadata["sphere_linear_speed_m_s"]
            angular = metadata["sphere_angular_speed_rad_s"]
            neutral_actions = (actions[:, 6] > 0) & (np.linalg.norm(actions[:, :6], axis=1) <= 0.02)
            for phase, is_neutral in zip(current_phases, neutral_actions):
                bucket = neutral_by_phase.setdefault(phase, {"actions": 0, "neutral_actions": 0})
                bucket["actions"] += 1
                bucket["neutral_actions"] += int(is_neutral)
            windows = []
            for begin in range(0, len(actions) - 7, 8):
                chunk = actions[begin:begin + 8]
                if len(chunk) != 8 or len(current_phases) < begin + 8 or len(linear) < begin + 8 or len(angular) < begin + 8:
                    continue
                neutral = bool(np.all(chunk[:, 6] > 0) and np.max(np.linalg.norm(chunk[:, :6], axis=1)) <= 0.02
                               and max(linear[begin:begin + 8]) <= matched.NATIVE_LIN_THRESH_M_S
                               and max(angular[begin:begin + 8]) <= matched.NATIVE_ANG_THRESH_RAD_S)
                windows.append({"start": begin, "phase": current_phases[begin], "neutral": neutral})
            rows["episodes"].append({"id": key, "kind": str(group.attrs["kind"]), "T": len(actions),
                                     "le_200": len(actions) <= POLICY_STEPS, "windows": windows,
                                     "neutral_action_count": int(neutral_actions.sum()),
                                     "neutral_action_fraction": float(neutral_actions.mean()) if len(actions) else 0.0})
    durations = [row["T"] for row in rows["episodes"]]
    neutral_counts = {}
    for row in rows["episodes"]:
        for window in row["windows"]:
            bucket = neutral_counts.setdefault(window["phase"], {"windows": 0, "neutral": 0})
            bucket["windows"] += 1
            bucket["neutral"] += int(window["neutral"])
    rows["lengths"] = {
        "count": len(durations),
        "at_or_below_200": sum(value <= POLICY_STEPS for value in durations),
        "above_200": sum(value > POLICY_STEPS for value in durations),
        "max": max(durations, default=0),
        "median": float(np.median(durations)) if durations else None,
        "p90": float(np.percentile(durations, 90)) if durations else None,
        "p95": float(np.percentile(durations, 95)) if durations else None,
    }
    rows["lengths_by_role"] = {}
    for kind in ("direct", "obstacle"):
        values = [row["T"] for row in rows["episodes"] if row["kind"] == kind]
        rows["lengths_by_role"][kind] = {
            "count": len(values), "at_or_below_200": sum(value <= POLICY_STEPS for value in values),
            "above_200": sum(value > POLICY_STEPS for value in values),
            "max": max(values, default=0), "median": float(np.median(values)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "action_fraction_above_200": float(sum(max(0, value - POLICY_STEPS) for value in values)
                                                   / sum(values)) if values and sum(values) else 0.0,
        }
    rows["neutral_windows_by_phase"] = neutral_counts
    rows["neutral_actions_by_phase"] = {
        phase: {**counts, "fraction": counts["neutral_actions"] / counts["actions"]}
        for phase, counts in neutral_by_phase.items()
    }
    rows["neutral_action_fraction_by_role"] = {}
    for kind in ("direct", "obstacle"):
        episodes = [row for row in rows["episodes"] if row["kind"] == kind]
        total_steps = sum(row["T"] for row in episodes)
        neutral_steps = sum(row["neutral_action_count"] for row in episodes)
        rows["neutral_action_fraction_by_role"][kind] = {
            "neutral_actions": neutral_steps, "actions": total_steps,
            "fraction": neutral_steps / total_steps if total_steps else 0.0,
        }
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "pilot", "collect"))
    parser.add_argument("--registration", required=True)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--pilot-layouts", type=int, default=0)
    parser.add_argument("--pair-layouts", type=int)
    parser.add_argument("--outer-wall-s", type=float, default=OUTER_WALL_S)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--direct", type=int, default=DIRECT_TARGET)
    parser.add_argument("--obstacle", type=int, default=OBSTACLE_TARGET)
    parser.add_argument("--release-hold-steps", type=int, choices=matched.RELEASE_HOLD_STEPS_CANDIDATES,
                        default=HOLD_STEPS)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        preregister(args.registration, release_hold_steps=args.release_hold_steps)
        return
    if not args.dataset_dir or not args.experiment_dir:
        parser.error("pilot/collect require --dataset-dir and --experiment-dir")
    pilot = args.command == "pilot"
    report = run_collection(args.registration, args.dataset_dir, args.experiment_dir,
                            target_direct=args.direct, target_obstacle=args.obstacle,
                            pilot_layouts=args.pilot_layouts if pilot else 0,
                            pair_layout_target=args.pair_layouts if args.pair_layouts is not None else (25 if pilot else None),
                            outer_wall_s=args.outer_wall_s, max_attempts=args.max_attempts)
    print(json.dumps({key: report[key] for key in ("complete", "targets", "accepted", "attempted_layouts", "paired_layouts")}, sort_keys=True))


if __name__ == "__main__":
    main()
